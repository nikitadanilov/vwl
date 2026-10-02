"""Rendering a two-person plan (see duo_layout.py for the screen geometry).

The upper part of the frame shows a spread: a row of 1-3 photos/videos while together, or each
person's own photos on their half while apart.  The map band below never overlaps the photos: one
wide map with both routes as parallel lanes while together, two half-width maps (each following
its own person, on that person's clock) while apart.  The layout switches during the transition
between spreads, together with the morph of the photos.
"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np

from . import overlay as O
from .camera import SPAN_MIN, Track, continuous_motion, follow_camera, segments, smoothstep, trail
from .duo_layout import DuoLayout, row_layout
from .mapview import MapView, min_span, robust_bbox, union, zoom_for_bbox
from .morph import Morph, ken_burns
from .render import TRAIL_S, XFADE, Renderer, load_track
from .video import ClipReader

SWITCH_S = 0.4      # minimum seconds over which the map band switches between one and two maps


def build_duo_timeline(plan: dict, locs: list[dict], lay: DuoLayout, tile_px: int, cp: dict):
    fps = plan["fps"]
    slots = plan["slots"]
    segs, N = segments(plan)
    tracks = [Track(loc) for loc in locs]
    t_life0, t_life1 = plan["range"]
    # each person's map runs on their own clock: the time of the photos on their side
    slot_period = min((s["hold"] + s["trans"] for s in slots), default=1.0)
    people = []
    for k, tr in enumerate(tracks):
        pk = copy.copy(plan)
        pk["slots"] = [dict(s, ts=s.get("ts_by", [s["ts"]] * 2)[k]) for s in slots]
        tau, pos = continuous_motion(tr, pk, segs, N)
        cam, vis, ok = follow_camera(segs, N, [pos], fps, slot_period, lay.half_w, lay.mh, tile_px)
        people.append({"tau": tau, "pos": pos, "cam": cam, "vis": vis, "has": tr.has and bool(ok.any())})
    wide = lay.map_together[2]
    camj, visj, _ = follow_camera(segs, N, [people[0]["pos"], people[1]["pos"]], fps, slot_period,
                                  wide, lay.mh, tile_px)
    # one map while the spread is "together", two while "apart"; switch mid-transition, eased
    mode = np.zeros(N)
    for kind, i, f0, f1 in segs:
        if kind == "hold":
            mode[f0:f1] = slots[i]["mode"] == "together"
        elif kind == "trans":
            h = (f0 + f1) // 2
            mode[f0:h] = slots[i]["mode"] == "together"
            mode[h:f1] = slots[i + 1]["mode"] == "together"
        elif kind == "intro":
            mode[f0:f1] = slots[0]["mode"] == "together"
        else:
            mode[f0:f1] = slots[-1]["mode"] == "together"
    k = max(1, int(SWITCH_S * fps / 2))
    merge = smoothstep(np.convolve(np.pad(mode, (k, k), mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid"))

    full_cam = None
    bbs = [robust_bbox(tr.mx, tr.my, 0.5, 99.5) for tr in tracks if tr.has]
    if bbs:
        bb = bbs[0]
        for other in bbs[1:]:
            bb = union(bb, other)
        bb = min_span(bb, SPAN_MIN * 10)
        full_cam = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, zoom_for_bbox(*bb, lay.W, lay.H, tile_px, pad=1.2))
    views = [(people[0]["cam"], lay.half_w, lay.mh), (people[1]["cam"], lay.half_w, lay.mh), (camj, wide, lay.mh)]
    return {"segs": segs, "N": N, "tau": people[0]["tau"], "people": people, "camj": camj, "visj": visj,
            "merge": merge, "full_cam": full_cam, "t_life": (t_life0, t_life1), "views": views,
            "has_loc": any(p["has"] for p in people)}


def rounded_mask(w, h, r):
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    return mask.astype(np.float32) / 255


def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale to cover w×h and centre-crop (cells already have the photo's aspect, so little is lost)."""
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    x0, y0 = (nw - w) // 2, (nh - h) // 2
    return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])


class DuoRenderer(Renderer):
    BG = (11, 12, 16)

    def setup_maps(self):
        W, H = self.plan["size"]
        self.lay = lay = DuoLayout(W, H)
        self.persons = self.plan["persons"]
        self.locs = [load_track(Path(p["work"]), self.plan) for p in self.persons]
        z = np.load(self.work / "duo.npz")
        self.cp = {k: z[k] for k in z.files}
        self.tl = build_duo_timeline(self.plan, self.locs, lay, self.tiles.px, self.cp)
        self.tracks = [Track(loc) for loc in self.locs]
        self.view_half = [MapView(self.tiles, self.locs[k], lay.half_w, lay.mh) for k in (0, 1)]
        self.view_wide = MapView(self.tiles, self.locs[0], lay.map_together[2], lay.mh)
        self.full = MapView(self.tiles, self.locs[0], W, H)
        r = int(16 * lay.s)
        self.mask_half, self.mask_wide = rounded_mask(lay.half_w, lay.mh, r), rounded_mask(lay.map_together[2], lay.mh, r)
        t0, t1 = self.tl["t_life"]
        self.strip, self.x_of = O.timeline_strip_duo(self.cp["t"], self.cp["state"], self.cp["dist"], t0, t1,
                                                     W, lay.strip_h, lay.s)
        self.cell_cache = {}
        self.photo_cache = {}
        self.clip_readers = {}

    # --- spreads
    def cells(self, i):
        """[(item, (x, y, w, h))] for spread i."""
        if i not in self.cell_cache:
            s, lay = self.slots[i], self.lay
            groups = [(s["items"], lay.photos_together)] if s["mode"] == "together" else \
                [(s["left"], lay.photos_left), (s["right"], lay.photos_right)]
            out = []
            for items, region in groups:
                if items:
                    out += list(zip(items, row_layout([it["aspect"] for it in items], region, lay.gap)))
            self.cell_cache[i] = out
            if len(self.cell_cache) > 6:
                self.cell_cache.pop(min(self.cell_cache))
        return self.cell_cache[i]

    def has_video(self, i) -> bool:
        return any(it["kind"] == "video" for it, _ in self.cells(i))

    def _photo(self, it, w, h):
        key = (it["file"], w, h)
        img = self.photo_cache.get(key)
        if img is None:
            src = cv2.cvtColor(cv2.imread(str(self.work / it["file"])), cv2.COLOR_BGR2RGB)
            img = _cover(src, w, h)
            self.photo_cache[key] = img
            if len(self.photo_cache) > 24:
                self.photo_cache.pop(next(iter(self.photo_cache)))
        return img

    def _video(self, it, k, w, h):
        r = self.clip_readers.get(it["file"])
        if r is None:
            if len(self.clip_readers) > 6:
                self.clip_readers.pop(next(iter(self.clip_readers))).close()
            r = self.clip_readers[it["file"]] = ClipReader(str(self.work / it["file"]))
        return _cover(r.frame(k), w, h)

    def compose(self, i, u, k):
        """The photo area of spread i at hold progress u (Ken Burns) and video frame k."""
        canvas = np.empty((self.lay.H, self.lay.W, 3), np.uint8)
        canvas[:] = self.BG
        for j, (it, (x, y, w, h)) in enumerate(self.cells(i)):
            if it["kind"] == "video":
                cell = self._video(it, k, w, h)
            else:
                drift = tuple(np.random.default_rng(i * 7 + j).uniform(-0.5, 0.5, 2))
                cell = ken_burns(self._photo(it, w, h), u, 0.05, drift)
            canvas[y:y + h, x:x + w] = cell
        return canvas

    def start_canvas(self, i):
        return self.compose(i, 0.0, 0)

    def end_canvas(self, i):
        return self.compose(i, 1.0, self.hold_frames[i] - 1)

    def item_caption(self, it):
        key = ("icap", it["file"])
        if key not in self.text:
            sc = self.lay.s
            lines = [it.get("owner_name", ""), it["date"]] + ([it["place"]] if it.get("place") else [])
            sizes = [int(20 * sc), int(30 * sc)] + [int(20 * sc)] * (len(lines) - 2)
            weights = ["SemiBold", "SemiBold"] + ["Regular"] * (len(lines) - 2)
            cols = [tuple(it.get("owner_color", (255, 255, 255))), (255, 255, 255)] + [(230, 230, 230)] * (len(lines) - 2)
            self.text[key] = O.shadowed_text(lines, sizes, weights, cols, gap=3, pad=14)
        return self.text[key]

    def captions(self, frame, i, alpha):
        if alpha <= 0.003:
            return
        for it, (x, y, w, h) in self.cells(i):
            O.blend(frame, self.item_caption(it), x - 6, y - 6, alpha)

    # --- maps
    def _person(self, k, f):
        p = self.tl["people"][k]
        pos = p["pos"][f]
        tr = self.tracks[k]
        return {"track": (tr.t, tr.mx, tr.my), "tau": p["tau"][f],
                "history": trail(p["pos"], f, int(TRAIL_S * self.plan["fps"])),
                "pos": None if np.isnan(pos[0]) else pos, "color": tuple(self.persons[k]["color"])}

    def _put(self, frame, img, rect, mask, alpha):
        x, y, w, h = rect
        region = frame[y:y + h, x:x + w]
        a = (mask * alpha)[..., None]
        region[:] = (region * (1 - a) + img * a).astype(np.uint8)

    def _label(self, img, names):
        x = int(12 * self.lay.s)
        fs, th = 0.7 * self.lay.s, 2
        (_, th0), base = cv2.getTextSize(" ".join(n for n, _ in names), cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        pad = int(7 * self.lay.s)
        y = int(28 * self.lay.s)
        widths = [cv2.getTextSize(n, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] for n, _ in names]
        gap = int(22 * self.lay.s)
        x1 = x + sum(widths) + gap * (len(names) - 1)
        box = img[max(0, y - th0 - pad):y + base + pad, max(0, x - pad):x1 + pad]
        box[:] = (box * 0.25 + np.array([14, 15, 20]) * 0.75).astype(np.uint8)   # dark label background
        for (name, col), w in zip(names, widths):
            cv2.putText(img, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, tuple(int(c) for c in col), th, cv2.LINE_AA)
            x += w + gap

    def map_band(self, f, frame, alpha=1.0):
        tl, lay = self.tl, self.lay
        if not tl["has_loc"] or alpha <= 0.003:
            return
        m = float(tl["merge"][f])
        pulse = (f / self.plan["fps"] * 0.8) % 1.0
        if m < 0.997:
            for k, rect in ((0, lay.map_left), (1, lay.map_right)):
                p = tl["people"][k]
                vis = float(p["vis"][f])
                cx, cy, z = p["cam"][f]
                person = self._person(k, f)
                img = self.view_half[k].draw_people(cx, cy, z, [person], pulse=pulse,
                                                    dim=0.85 if vis > 0.5 else 0.45)
                self.locator().paste(img, cx, cy, z, self.tiles.px, [(person["pos"], person["color"])])
                self._label(img, [(self.persons[k]["name"] + ("" if vis > 0.5 else "  - no data"),
                                   self.persons[k]["color"])])
                self._put(frame, img, rect, self.mask_half, alpha * (1 - m))
        if m > 0.003:
            cx, cy, z = tl["camj"][f]
            both = [self._person(0, f), self._person(1, f)]
            img = self.view_wide.draw_people(cx, cy, z, both, pulse=pulse, lanes=3.0 * self.lay.s)
            self.locator().paste(img, cx, cy, z, self.tiles.px, [(p["pos"], p["color"]) for p in both])
            self._label(img, [(p["name"], p["color"]) for p in self.persons])
            self._put(frame, img, lay.map_together, self.mask_wide, alpha * m)

    def full_map(self, f, reveal: bool):
        tl = self.tl
        if not tl["full_cam"]:
            return np.full((self.lay.H, self.lay.W, 3), 12, np.uint8)
        cx, cy, z = tl["full_cam"]
        upto = tl["tau"][f] if reveal else tl["t_life"][1]
        people = []
        for k in (0, 1):
            tr = self.tracks[k]
            people.append({"track": (tr.t, tr.mx, tr.my), "tau": upto, "pos": None,
                           "color": tuple(self.persons[k]["color"])})
        return self.full.draw_people(cx, cy, z, people, dim=0.7, trace_only=True)

    # --- frames
    def frame(self, f) -> np.ndarray:
        fps, tl, lay = self.plan["fps"], self.tl, self.lay
        kind, i, f0, f1 = next(s for s in tl["segs"] if s[2] <= f < s[3])
        u = (f - f0) / max(1, f1 - f0)
        sec_in = (f - f0) / fps
        if kind in ("intro", "outro"):
            frame = self.intro_background(u) if kind == "intro" else self.outro_background(u)
            title = self.title_img(final=kind == "outro")
            if kind == "intro":
                O.blend(frame, title, (lay.W - title.shape[1]) // 2, (lay.H - title.shape[0]) // 2,
                        float(smoothstep(sec_in / 0.8)))
                frame = (frame * smoothstep(sec_in / 0.6)).astype(np.uint8)
                left = (f1 - f) / fps
                if left < XFADE:
                    a = float(smoothstep(1 - left / XFADE))
                    nxt = self.start_canvas(0)
                    self.map_band(f, nxt)
                    frame = cv2.addWeighted(frame, 1 - a, nxt, a, 0)
                return frame
            if sec_in < XFADE:
                a = float(smoothstep(sec_in / XFADE))
                prev = self.end_canvas(i)
                self.map_band(f, prev)
                frame = cv2.addWeighted(prev, 1 - a, frame, a, 0)
            return (frame * smoothstep((f1 - f) / fps / 1.2)).astype(np.uint8)
        if kind == "hold":
            frame = self.compose(i, u, f - f0)
        else:
            if i not in self.morphs:
                self.morphs.clear()
                self.morphs[i] = Morph(self.end_canvas(i), self.start_canvas(i + 1), strength=self.opts["morph"])
            live = self.compose(i, 1.0, self.hold_frames[i] + f - f0) if self.has_video(i) else None
            frame = self.morphs[i].frame(float(u), live)
            frame[lay.my - lay.gap // 2:] = self.BG     # the morph must not smear photos into the map band
        frame = np.ascontiguousarray(frame)
        self.map_band(f, frame)
        # per-photo captions: fade in at hold start, cross-fade during the transition
        if kind == "hold":
            self.captions(frame, i, float(smoothstep(sec_in / 0.4)))
        else:
            self.captions(frame, i, float(1 - smoothstep(u / 0.5)))
            self.captions(frame, i + 1, float(smoothstep((u - 0.5) / 0.5)))
        self.cards(frame, kind, i, u, sec_in)
        O.blend(frame, self.strip, 0, lay.H - lay.strip_h, 1.0)
        x = self.x_of(tl["tau"][f])
        cv2.line(frame, (x, lay.H - lay.strip_h), (x, lay.H - 1), (255, 255, 255), max(2, int(3 * lay.s)), cv2.LINE_AA)
        return frame

    def cards(self, frame, kind, i, u, sec_in):
        """Year and chapter cards, centred on the photo area."""
        lay = self.lay
        cy = lay.margin + lay.photo_h // 2
        if kind == "trans" and self.slots[i + 1]["year"] != self.slots[i]["year"]:
            img = self.year_img(self.slots[i + 1]["year"])
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, cy - img.shape[0] // 2, float(smoothstep(u / 0.4)) * 0.9)
        elif kind == "hold" and (i == 0 or self.slots[i - 1]["year"] != self.slots[i]["year"]):
            img = self.year_img(self.slots[i]["year"])
            a = 1 - smoothstep((sec_in - 0.6) / 0.6)
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, cy - img.shape[0] // 2, float(a) * 0.9)
        if kind == "hold" and self.slots[i].get("chapter"):
            img = self.chapter_img(i)
            a = smoothstep(sec_in / 0.3) * (1 - smoothstep((sec_in - 1.6) / 0.5))
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, cy + int(80 * lay.s), float(a))

    def title_img(self, final=False):
        key = ("title", final)
        if key not in self.text:
            p, st, sc = self.plan, self.plan["stats"], self.lay.s
            span = f"{st['span'][0]} — {st['span'][1]}" if st.get("span") else ""
            nums = [f"{st['days_together']:,} days together"]
            if st.get("km_together"):
                nums.append(f"{st['km_together']:,} km together")
            if st.get("countries_together"):
                nums.append(f"{st['countries_together']} countries together")
            extra = []
            if st.get("longest_apart_days"):
                extra.append(f"longest apart: {st['longest_apart_days']} days")
            if st.get("furthest_km"):
                extra.append(f"furthest apart: {st['furthest_km']:,} km")
            lines = [p["title"], span, "  ·  ".join(nums)] + (["  ·  ".join(extra)] if extra else [])
            sizes = [int(72 * sc), int(40 * sc), int(26 * sc)] + [int(22 * sc)] * bool(extra)
            weights = ["Bold", "Light", "Regular"] + ["Regular"] * bool(extra)
            self.text[key] = O.shadowed_text(lines, sizes, weights, [(255, 255, 255)] * len(lines), gap=18, pad=40)
        return self.text[key]
