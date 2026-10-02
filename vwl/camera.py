"""Film timing and map cameras, shared by the single-person and two-person renderers.

* `segments(plan)`: the film as (kind, slot, first frame, end frame) runs.
* `Track`: one person's movement; `where(t)` is the gap-aware position.
* `continuous_motion(...)`: the dot's position and the map clock for every frame.  The dot rests at
  each photo's place mid-photo and travels the recorded route to the next one, easing out and in,
  over the rest of the hold and the transition — the map never jumps.
* `follow_camera(...)`: a smooth camera framing where the dots have just been and are about to go.
"""
from __future__ import annotations

import numpy as np

from .mapview import min_span, to_world, zoom_for_bbox

GAP = 6 * 3600          # interpolate the position only between fixes at most this far apart
STALE = 2 * 86400       # further than this from any fix, the position is unknown
SPAN_MIN = 1.3e-4       # smallest map view (world units), ≈ 3-5 km
DAILY_AFTER = 2 * 86400 # between photos further apart than this, the route is one point per day
MAX_ROUTE = 300         # at most this many route points per hop
ZOOM_LAG = 2.0          # zoom is smoothed this many times more slowly than the pan


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def segments(plan: dict):
    """[(kind, slot, f0, f1)] for intro, each hold and transition, and outro; and the frame count."""
    fps = plan["fps"]
    slots = plan["slots"]
    segs, f = [], 0

    def add(kind, i, sec):
        nonlocal f
        n = max(0, int(round(sec * fps)))
        segs.append((kind, i, f, f + n))
        f += n

    add("intro", -1, plan["intro"])
    for i, s in enumerate(slots):
        add("hold", i, s["hold"])
        if s["trans"] > 0 and i + 1 < len(slots):
            add("trans", i, s["trans"])
    add("outro", len(slots) - 1, plan["outro"])
    return segs, f


class Track:
    def __init__(self, loc: dict):
        self.t = loc["t"]
        self.mx, self.my = to_world(loc["lat"], loc["lon"])
        self.has = len(self.t) > 0
        self._daily = None

    def where(self, tq) -> np.ndarray:
        """Gap-aware position at times tq: linear between fixes at most GAP apart; across a longer
        gap, the last known fix; NaN when no fix lies within STALE (before or after)."""
        t, mx, my = self.t, self.mx, self.my
        tq = np.atleast_1d(np.asarray(tq, np.float64))
        out = np.full((len(tq), 2), np.nan)
        if not self.has:
            return out
        k = np.searchsorted(t, tq, side="right")
        prv, nxt = np.clip(k - 1, 0, len(t) - 1), np.clip(k, 0, len(t) - 1)
        dt_prev = np.where(k > 0, tq - t[prv], np.inf)
        dt_next = np.where(k < len(t), t[nxt] - tq, np.inf)
        span = t[nxt] - t[prv]
        w = np.where((k > 0) & (k < len(t)) & (span > 0), dt_prev / np.maximum(span, 1e-9), 0.0)
        lerp = (k > 0) & (k < len(t)) & (span <= GAP)
        out[:, 0] = np.where(lerp, mx[prv] + w * (mx[nxt] - mx[prv]), np.where(dt_prev <= dt_next, mx[prv], mx[nxt]))
        out[:, 1] = np.where(lerp, my[prv] + w * (my[nxt] - my[prv]), np.where(dt_prev <= dt_next, my[prv], my[nxt]))
        # across a long gap stay at the last fix rather than jumping early to the next one
        hold_prev = ~lerp & (k > 0) & (dt_prev <= STALE)
        out[hold_prev, 0], out[hold_prev, 1] = mx[prv][hold_prev], my[prv][hold_prev]
        out[np.minimum(dt_prev, dt_next) > STALE] = np.nan
        return out

    def daily(self):
        """(time, mx, my) of each day's median position: the route at the scale of days."""
        if self._daily is None:
            day = (self.t // 86400).astype(np.int64)
            starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1] if len(day) else np.zeros(0, int)
            ends = np.r_[starts[1:], len(day)] if len(day) else starts
            self._daily = (np.array([np.median(self.t[a:b]) for a, b in zip(starts, ends)]),
                           np.array([np.median(self.mx[a:b]) for a, b in zip(starts, ends)]),
                           np.array([np.median(self.my[a:b]) for a, b in zip(starts, ends)]))
        return self._daily

    def route(self, ta: float, tb: float):
        """Recorded points strictly between ta and tb: raw for short hops, daily medians for long ones."""
        if tb - ta > DAILY_AFTER:
            t, mx, my = self.daily()
        else:
            t, mx, my = self.t, self.mx, self.my
        i0, i1 = np.searchsorted(t, [ta, tb], side="right")
        idx = np.arange(i0, i1)
        if len(idx) > MAX_ROUTE:
            idx = idx[np.linspace(0, len(idx) - 1, MAX_ROUTE).astype(int)]
        return t[idx], mx[idx], my[idx]


def anchors(plan: dict, segs) -> list[tuple[int, float, bool]]:
    """(frame, life time, video) keyframes: mid-hold for a photo; the start and end of a video clip,
    between which the clock runs in real time."""
    out = []
    slots = plan["slots"]
    for kind, i, f0, f1 in segs:
        if kind != "hold":
            continue
        s = slots[i]
        if s.get("kind") == "video":
            out += [(f0, s["ts"], True), (max(f0, f1 - 1), s["ts"] + s["hold"], False)]
        else:
            out.append(((f0 + f1) // 2, s["ts"], False))
    return out


def continuous_motion(track: Track, plan: dict, segs, N: int):
    """(tau, pos): the map clock (life time) and the dot's world position for every frame."""
    keys = anchors(plan, segs)
    tau = np.zeros(N)
    pos = np.full((N, 2), np.nan)
    if not keys:
        return tau, pos
    for (fa, ta, video), (fb, tb, _) in zip(keys[:-1], keys[1:]):
        n = fb - fa
        if n <= 0:
            continue
        u = np.arange(n) / n
        if video:                                  # inside a clip: real time, the dot where it was
            tau[fa:fb] = ta + (tb - ta) * u
            pos[fa:fb] = track.where(tau[fa:fb])
            continue
        A, B = track.where(ta)[0], track.where(tb)[0]
        if np.isnan(A[0]) or np.isnan(B[0]):       # no position at one end: hidden, clock still runs
            tau[fa:fb] = ta + (tb - ta) * smoothstep(u)
            half = fa + n // 2
            pos[fa:half], pos[half:fb] = A, B
            continue
        rt, rx, ry = track.route(ta, tb)
        px, py, pt = np.r_[A[0], rx, B[0]], np.r_[A[1], ry, B[1]], np.r_[ta, rt, tb]
        cum = np.r_[0, np.cumsum(np.hypot(np.diff(px), np.diff(py)))]
        if cum[-1] <= 0:                           # stayed put: just let the clock run
            tau[fa:fb] = ta + (tb - ta) * smoothstep(u)
            pos[fa:fb] = A
            continue
        sq = smoothstep(u) * cum[-1]               # even speed along the route, easing in and out
        pos[fa:fb, 0] = np.interp(sq, cum, px)
        pos[fa:fb, 1] = np.interp(sq, cum, py)
        tau[fa:fb] = np.maximum.accumulate(np.interp(sq, cum, pt))
    f_first, t_first, _ = keys[0]
    f_last, t_last, _ = keys[-1]
    tau[:f_first], pos[:f_first] = t_first, track.where(t_first)[0]
    tau[f_last:], pos[f_last:] = t_last, track.where(t_last)[0]
    return tau, pos


def _kernel(sg):
    k = np.exp(-0.5 * (np.arange(-int(3 * sg), int(3 * sg) + 1) / sg) ** 2)
    return k / k.sum()


def _smooth(v, k):
    return np.convolve(np.pad(v, (len(k) // 2, len(k) // 2), mode="edge"), k, mode="valid")[: len(v)]


def follow_camera(segs, N: int, dots: list[np.ndarray], fps: int, slot_period: float, pw: int, ph: int,
                  tile_px: int):
    """Per-frame camera (cx, cy, zoom), visibility (0..1) and a has-camera mask for a panel showing `dots`.

    The target frames everything the dots cover within ± half a photo period, so the view zooms out
    while they travel far and back in while they stay; centre and zoom are then smoothed (zoom more
    slowly), and the view eases open wherever a dot would leave the central 80%.  The panel is hidden
    only where no dot has a position (no data), and the camera never jumps except across such gaps.
    """
    shown = np.zeros(N, bool)
    for kind, i, f0, f1 in segs:
        shown[f0:f1] = kind in ("hold", "trans")
    have = np.logical_or.reduce([~np.isnan(d[:, 0]) for d in dots])
    shown &= have
    w = max(2, int(0.5 * slot_period * fps))
    # bounding box of all known dot positions in a sliding window (min/max over a window, per axis)
    lo = np.full((N, 2), np.inf)
    hi = np.full((N, 2), -np.inf)
    for d in dots:
        for c in range(2):
            v = d[:, c]
            vl = np.where(np.isnan(v), np.inf, v)
            vh = np.where(np.isnan(v), -np.inf, v)
            pl = np.pad(vl, (w, w), constant_values=np.inf)
            ph_ = np.pad(vh, (w, w), constant_values=-np.inf)
            win_l = np.lib.stride_tricks.sliding_window_view(pl, 2 * w + 1).min(axis=1)
            win_h = np.lib.stride_tricks.sliding_window_view(ph_, 2 * w + 1).max(axis=1)
            lo[:, c] = np.minimum(lo[:, c], win_l)
            hi[:, c] = np.maximum(hi[:, c], win_h)
    key = shown & np.isfinite(lo[:, 0])
    target = np.full((N, 4), np.nan)
    for f in np.flatnonzero(key):
        target[f] = min_span((lo[f, 0], lo[f, 1], hi[f, 0], hi[f, 1]), SPAN_MIN)
    # runs between hidden stretches are smoothed separately: across no data the camera may jump
    edges = np.flatnonzero(np.diff(shown.astype(np.int8))) + 1
    bounds = np.r_[0, edges, N]
    sig = max(2.0, min(0.5 * fps, 0.4 * slot_period * fps))
    k_pan, k_zoom = _kernel(sig), _kernel(ZOOM_LAG * sig)
    ctr = np.full((N, 2), np.nan)
    lsz = np.full((N, 2), np.nan)
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        kf = np.flatnonzero(key[r0:r1])
        if not len(kf):
            continue
        tb = target[r0:r1][kf]
        fr = np.arange(r1 - r0)
        for c in range(2):
            ctr[r0:r1, c] = _smooth(np.interp(fr, kf, (tb[:, c] + tb[:, c + 2]) / 2), k_pan)
            lsz[r0:r1, c] = _smooth(np.interp(fr, kf, np.log(tb[:, c + 2] - tb[:, c])), k_zoom)
    half = np.exp(lsz) / 2
    cx, cy = ctr[:, 0], ctr[:, 1]
    hw, hh = half[:, 0], half[:, 1]
    # containment: every dot inside the central 80%; the widening is spread over neighbouring frames
    need = np.ones(N)
    with np.errstate(invalid="ignore", divide="ignore"):
        for d in dots:
            nd = np.fmax(np.abs(d[:, 0] - cx) / (0.8 * hw), np.abs(d[:, 1] - cy) / (0.8 * hh))
            need = np.fmax(need, np.nan_to_num(nd, nan=1.0))
    need = np.log(np.clip(need, 1.0, None))
    grow = np.zeros(N)
    wk = int(2 * sig)
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        v = need[r0:r1]
        if not v.any():
            continue
        mx_f = np.lib.stride_tricks.sliding_window_view(np.pad(v, (wk, wk), mode="edge"), 2 * wk + 1).max(axis=1)
        grow[r0:r1] = np.maximum(_smooth(mx_f, k_pan), v)
    g = np.exp(grow)
    boxes = np.c_[cx - hw * g, cy - hh * g, cx + hw * g, cy + hh * g]
    boxes[~shown] = np.nan
    cam = np.full((N, 3), np.nan)
    ok = ~np.isnan(boxes[:, 0])
    cam[ok, 0] = (boxes[ok, 0] + boxes[ok, 2]) / 2
    cam[ok, 1] = (boxes[ok, 1] + boxes[ok, 3]) / 2
    cam[ok, 2] = [zoom_for_bbox(*b, pw, ph, tile_px) for b in boxes[ok]]
    vis = ok.astype(np.float64)
    fk = max(1, int(0.25 * fps))
    vis = np.convolve(np.pad(vis, (fk, fk), mode="edge"), np.ones(2 * fk + 1) / (2 * fk + 1), mode="valid")
    vis[~ok] = 0.0
    if ok.any():  # frames without a camera reuse the nearest one, so tile lookup stays defined
        idx = np.maximum.accumulate(np.where(ok, np.arange(N), 0))
        idx[:np.flatnonzero(ok)[0]] = np.flatnonzero(ok)[0]
        cam = cam[idx]
    return cam, vis, ok


def trail(pos: np.ndarray, f: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """The dot's path over the last n frames up to f: (points, age 0..1, 1 = oldest), or None."""
    seg = pos[max(0, f - n):f + 1]
    ages = np.linspace(1, 0, len(seg)) * (len(seg) - 1) / max(n, 1)
    ok = ~np.isnan(seg[:, 0])
    if ok.sum() < 2:
        return None
    return seg[ok], ages[ok]
