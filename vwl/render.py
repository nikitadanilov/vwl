"""`vwl render`: plan.json + index → video.

Timeline:  [intro: whole-life map revealed] → per photo: hold (Ken Burns) → morph
→ … → [outro: whole-life map + totals].  "Life time" τ advances slowly during a
hold and sweeps to the next photo's time during the morph, so the map panel animates
the actual movement between the two photos, zooming out for trips and back in.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import overlay as O
from .archive import process_pool
from .camera import SPAN_MIN, Track, continuous_motion, follow_camera, segments, smoothstep, trail
from .mapview import Locator, MapView, Tiles, min_span, robust_bbox, view_tiles, world_tiles, zoom_for_bbox
from .morph import Morph, ken_burns, make_canvas
from .video import ClipReader

XFADE = 0.8
TRAIL_S = 5.0   # seconds of film over which the dot's trail fades out  # intro→first photo and last photo→outro cross-fade, seconds


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


class Layout:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.s = s = H / 1080
        self.margin = int(40 * s)
        self.strip_h = int(46 * s)
        self.pw, self.ph = int(W * 0.30) // 2 * 2, int(H * 0.38) // 2 * 2
        self.px = W - self.pw - self.margin
        self.py = H - self.strip_h - self.ph - self.margin


def build_timeline(plan: dict, loc: dict, lay: Layout, tile_px: int):
    fps = plan["fps"]
    slots = plan["slots"]
    segs, N = segments(plan)
    track = Track(loc)
    t, mx, my, has_loc = track.t, track.mx, track.my, track.has
    ts = np.array([s["ts"] for s in slots])
    t_life0 = min(t[0], ts[0]) if has_loc else ts[0]
    t_life1 = max(t[-1], ts[-1]) if has_loc else ts[-1]
    slot_period = min((s["hold"] + s["trans"] for s in slots), default=1.0)

    tau, pos = continuous_motion(track, plan, segs, N)
    cam, vis, ok = follow_camera(segs, N, [pos], fps, slot_period, lay.pw, lay.ph, tile_px)
    life_bb = robust_bbox(mx, my, 0.5, 99.5) if has_loc else None
    full_cam = None
    if life_bb:
        bb = min_span(life_bb, SPAN_MIN * 10)
        full_cam = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, zoom_for_bbox(*bb, lay.W, lay.H, tile_px, pad=1.2))
    return {"segs": segs, "N": N, "tau": tau, "pos": pos, "cam": cam, "panel_alpha": vis, "full_cam": full_cam,
            "has_loc": has_loc and bool(ok.any()), "t_life": (t_life0, t_life1)}


def needed_tiles(tl, lay, tile_px):
    need = set(world_tiles())  # the locator maps
    views = tl.get("views") or [(tl["cam"], lay.pw, lay.ph)]
    for cam, w, h in views:
        for f in range(0, tl["N"]):
            cx, cy, z = cam[f]
            if np.isnan(cx):
                continue
            for zz in (z, max(0.0, z - 1.0)):  # base tiles, and the coarser tiles labels are drawn from
                zi, _, _, _, (tx0, tx1, ty0, ty1) = view_tiles(cx, cy, zz, w, h, tile_px)
                n = 1 << zi
                need |= {(zi, x % n, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1) if 0 <= y < n}
    if tl["full_cam"]:
        cx, cy, z = tl["full_cam"]
        for zz in (z, max(0.0, z - 1.0)):
            zi, _, _, _, (tx0, tx1, ty0, ty1) = view_tiles(cx, cy, zz, lay.W, lay.H, tile_px)
            n = 1 << zi
            need |= {(zi, x % n, y) for x in range(tx0, tx1 + 1) for y in range(ty0, ty1 + 1) if 0 <= y < n}
    return need


# ---------------------------------------------------------------- worker

class Renderer:
    def __init__(self, work: str, opts: dict):
        self.work = Path(work)
        self.opts = opts
        plan = self.plan = json.load(open(self.work / "plan.json"))
        W, H = plan["size"]
        self.lay = lay = Layout(W, H)
        self.tiles = Tiles(opts["style"], offline=True, url=opts.get("tile_url"))
        self.slots = plan["slots"]
        self.setup_maps()
        self.canvas = {}
        self.morphs = {}
        self.readers = {}
        self.hold_frames = {i: f1 - f0 for kind, i, f0, f1 in self.tl["segs"] if kind == "hold"}
        self.text = {}
        mask = np.zeros((lay.ph, lay.pw), np.uint8)
        r = int(18 * lay.s)
        cv2.rectangle(mask, (r, 0), (lay.pw - r, lay.ph), 255, -1)
        cv2.rectangle(mask, (0, r), (lay.pw, lay.ph - r), 255, -1)
        for cx, cy in ((r, r), (lay.pw - r - 1, r), (r, lay.ph - r - 1), (lay.pw - r - 1, lay.ph - r - 1)):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
        self.panel_mask = mask

    def setup_maps(self):
        """Track, timeline, map views and the timeline strip (the two-person renderer overrides this)."""
        lay, plan = self.lay, self.plan
        self.loc = load_track(self.work, plan)
        self.tl = build_timeline(plan, self.loc, lay, self.tiles.px)
        self.track = Track(self.loc)
        self.panel = MapView(self.tiles, self.loc, lay.pw, lay.ph)
        self.full = MapView(self.tiles, self.loc, lay.W, lay.H)
        t0 = min(self.slots[0]["ts"], self.tl["t_life"][0])
        t1 = max(self.slots[-1]["ts"], self.tl["t_life"][1])
        self.strip, self.x_of = O.timeline_strip(self.slots, self.loc["t"], t0, t1, lay.W, lay.strip_h, lay.s)

    # --- cached per-slot assets
    def get_canvas(self, i):
        if i not in self.canvas:
            if len(self.canvas) > 4:
                self.canvas.pop(min(self.canvas))
            img = cv2.cvtColor(cv2.imread(str(self.work / self.slots[i]["file"])), cv2.COLOR_BGR2RGB)
            self.canvas[i] = make_canvas(img, self.lay.W, self.lay.H)
        return self.canvas[i]

    def drift(self, i):
        r = np.random.default_rng(i)
        return tuple(r.uniform(-0.5, 0.5, 2))

    def kb(self, i, u):
        return ken_burns(self.get_canvas(i), u, 0.06, self.drift(i))

    def is_video(self, i) -> bool:
        return self.slots[i].get("kind") == "video"

    def video_canvas(self, i, k):
        """Frame k of slot i's clip, fitted to the screen (beyond the clip's end: its last frame)."""
        r = self.readers.get(i)
        if r is None:
            for j in [j for j in self.readers if abs(j - i) > 1]:
                self.readers.pop(j).close()
            r = self.readers[i] = ClipReader(str(self.work / self.slots[i]["file"]))
        return make_canvas(r.frame(k), self.lay.W, self.lay.H)

    def start_canvas(self, i):
        return self.video_canvas(i, 0) if self.is_video(i) else self.kb(i, 0.0)

    def end_canvas(self, i):
        return self.video_canvas(i, self.hold_frames[i] - 1) if self.is_video(i) else self.kb(i, 1.0)

    def get_morph(self, i):
        if i not in self.morphs:
            self.morphs.clear()
            self.morphs[i] = Morph(self.end_canvas(i), self.start_canvas(i + 1), strength=self.opts["morph"])
        return self.morphs[i]

    def caption(self, i):
        key = ("cap", i)
        if key not in self.text:
            self.text[key] = self._caption(self.slots[i])
        return self.text[key]

    def _caption(self, s):
        sc = self.lay.s
        lines, sizes, weights, cols = [s["date"]], [int(50 * sc)], ["SemiBold"], [(255, 255, 255)]
        if s.get("owner_name"):
            lines.insert(0, s["owner_name"]); sizes.insert(0, int(24 * sc)); weights.insert(0, "SemiBold")
            cols.insert(0, tuple(s.get("owner_color", (255, 255, 255))))
        if s.get("place"):
            lines.append(s["place"]); sizes.append(int(28 * sc)); weights.append("Regular"); cols.append((225, 225, 225))
        if s.get("caption"):
            lines.append(f"“{s['caption'][:90]}”"); sizes.append(int(24 * sc)); weights.append("Italic"); cols.append((210, 210, 210))
        if s["source"] in O.PLATFORM:
            lines.append(f"posted on {O.PLATFORM[s['source']][0]}"); sizes.append(int(20 * sc)); weights.append("Medium"); cols.append(O.PLATFORM[s["source"]][1])
        return O.shadowed_text(lines, sizes, weights, cols)

    def chapter_img(self, i):
        key = ("chapter", i)
        if key not in self.text:
            self.text[key] = O.shadowed_text([self.slots[i]["chapter"]], [int(56 * self.lay.s)], ["SemiBold"],
                                             [(255, 255, 255)], pad=40)
        return self.text[key]

    def year_img(self, y):
        key = ("year", y)
        if key not in self.text:
            self.text[key] = O.shadowed_text([str(y)], [int(150 * self.lay.s)], ["Bold"], [(255, 255, 255)], pad=40)
        return self.text[key]

    def title_img(self, final=False):
        key = ("title", final)
        if key not in self.text:
            p, st, sc = self.plan, self.plan["stats"], self.lay.s
            span = f"{st['span'][0]} — {st['span'][1]}" if st.get("span") else ""
            nums = [f"{st['photos']:,} photos"]
            if st.get("videos"):
                nums.append(f"{st['videos']:,} videos")
            if st.get("location_points"):
                nums.append(f"{st['location_points']:,} location fixes")
            extra = []
            if st.get("km"):
                extra.append(f"{st['km']:,} km between recorded places")
            if st.get("countries"):
                extra.append(f"{st['countries']} countries")
            lines = [p.get("title") or "A life, as recorded", span, "  ·  ".join(nums)]
            sizes, weights = [int(72 * sc), int(40 * sc), int(26 * sc)], ["Bold", "Light", "Regular"]
            if extra:
                lines.append("  ·  ".join(extra)); sizes.append(int(26 * sc)); weights.append("Regular")
            self.text[key] = O.shadowed_text(lines, sizes, weights, [(255, 255, 255)] * len(lines), gap=18, pad=40)
        return self.text[key]

    # --- frame composition
    def map_panel(self, f, alpha, frame):
        tl = self.tl
        alpha *= tl["panel_alpha"][f]
        if not tl["has_loc"] or alpha <= 0.003:
            return
        cx, cy, z = tl["cam"][f]
        pos = tl["pos"][f]
        pulse = (f / self.plan["fps"] * 0.8) % 1.0
        person = {"track": (self.track.t, self.track.mx, self.track.my), "tau": tl["tau"][f],
                  "history": trail(tl["pos"], f, int(TRAIL_S * self.plan["fps"])),
                  "pos": None if np.isnan(pos[0]) else pos, "color": MapView.TRAIL, "trace_color": MapView.TRACE}
        img = self.panel.draw_people(cx, cy, z, [person], pulse=pulse)
        self.locator().paste(img, cx, cy, z, self.tiles.px, [(person["pos"], MapView.TRAIL)])
        lay = self.lay
        region = frame[lay.py:lay.py + lay.ph, lay.px:lay.px + lay.pw]
        a = (self.panel_mask.astype(np.float32) / 255 * alpha * 0.96)[..., None]
        region[:] = (region * (1 - a) + img * a).astype(np.uint8)

    def locator(self) -> Locator:
        """The small world map in the corner of map panels (built on first use: tiles must be fetched)."""
        if getattr(self, "_locator", None) is None:
            self._locator = Locator(self.tiles, int(150 * self.lay.s) // 2 * 2, int(80 * self.lay.s) // 2 * 2)
        return self._locator

    def full_map(self, f, reveal: bool):
        tl = self.tl
        if not tl["full_cam"]:
            return np.full((self.lay.H, self.lay.W, 3), 12, np.uint8)
        cx, cy, z = tl["full_cam"]
        upto = tl["tau"][f] if reveal else tl["t_life"][1]
        person = {"track": (self.track.t, self.track.mx, self.track.my), "tau": upto, "color": MapView.TRAIL,
                  "trace_color": MapView.TRACE}
        return self.full.draw_people(cx, cy, z, [person], dim=0.7, trace_only=True)

    def ui(self, f, frame, kind, i, u, sec_in, seg_len):
        lay = self.lay
        # captions: fade in at hold start, cross-fade during morph
        if kind == "hold":
            blend_a = smoothstep(sec_in / 0.5)
            O.blend(frame, self.caption(i), lay.margin - 24, lay.margin - 24, blend_a)
        elif kind == "trans":
            O.blend(frame, self.caption(i), lay.margin - 24, lay.margin - 24, 1 - smoothstep(u / 0.5))
            O.blend(frame, self.caption(i + 1), lay.margin - 24, lay.margin - 24, smoothstep((u - 0.5) / 0.5))
        # year card on year change
        if kind == "trans" and self.slots[i + 1]["year"] != self.slots[i]["year"]:
            img = self.year_img(self.slots[i + 1]["year"])
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, (lay.H - img.shape[0]) // 2 - int(60 * lay.s),
                    float(smoothstep(u / 0.4)) * 0.9)
        elif kind == "hold" and (i == 0 or self.slots[i - 1]["year"] != self.slots[i]["year"]):
            img = self.year_img(self.slots[i]["year"])
            a = 1 - smoothstep((sec_in - 0.6) / 0.6)
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, (lay.H - img.shape[0]) // 2 - int(60 * lay.s), float(a) * 0.9)
        # chapter card (two-person films): first time together, reunions
        if kind == "hold" and self.slots[i].get("chapter"):
            img = self.chapter_img(i)
            a = smoothstep(sec_in / 0.3) * (1 - smoothstep((sec_in - 1.6) / 0.5))
            O.blend(frame, img, (lay.W - img.shape[1]) // 2, (lay.H - img.shape[0]) // 2 + int(70 * lay.s), float(a))
        # timeline strip + playhead
        O.blend(frame, self.strip, 0, lay.H - lay.strip_h, 1.0)
        x = self.x_of(self.tl["tau"][f])
        cv2.line(frame, (x, lay.H - lay.strip_h), (x, lay.H - 1), (255, 146, 64), max(2, int(3 * lay.s)), cv2.LINE_AA)

    def frame(self, f) -> np.ndarray:
        fps = self.plan["fps"]
        tl = self.tl
        seg = next(s for s in tl["segs"] if s[2] <= f < s[3])
        kind, i, f0, f1 = seg
        n = max(1, f1 - f0)
        u = (f - f0) / n
        sec_in = (f - f0) / fps
        if kind == "intro":
            frame = self.intro_background(u)
            O.blend(frame, self.title_img(), (self.lay.W - self.title_img().shape[1]) // 2,
                    (self.lay.H - self.title_img().shape[0]) // 2, float(smoothstep(sec_in / 0.8)))
            fade_in = smoothstep(sec_in / 0.6)
            frame = (frame * fade_in).astype(np.uint8)
            left = (f1 - f) / fps
            if left < XFADE:
                a = float(smoothstep(1 - left / XFADE))
                frame = cv2.addWeighted(frame, 1 - a, self.start_canvas(0), a, 0)
            return frame
        if kind == "outro":
            frame = self.outro_background(u)
            if sec_in < XFADE:
                a = float(smoothstep(sec_in / XFADE))
                frame = cv2.addWeighted(self.end_canvas(i), 1 - a, frame, a, 0)
            left = (f1 - f) / fps
            return (frame * smoothstep(left / 1.2)).astype(np.uint8)
        if kind == "hold":
            frame = self.video_canvas(i, f - f0) if self.is_video(i) else self.kb(i, u)
        else:
            # an outgoing clip keeps playing through the morph
            live = self.video_canvas(i, self.hold_frames[i] + f - f0) if self.is_video(i) else None
            frame = self.get_morph(i).frame(float(u), live)
        frame = np.ascontiguousarray(frame)
        # darken top-left & bottom for legibility
        self._vignette(frame)
        panel_a = 1.0
        if kind == "hold" and i == 0:
            panel_a = float(smoothstep(sec_in / 0.6))
        self.map_panel(f, panel_a, frame)
        self.ui(f, frame, kind, i, u, sec_in, n / fps)
        return frame

    def intro_background(self, u: float) -> np.ndarray:
        """The film's photos as a mosaic (work/mosaic.jpg), drifting slowly inward, dimmed, and dimmed
        much more in a soft area around the title so the text stays readable."""
        if not hasattr(self, "_mosaic"):
            W, H = self.lay.W, self.lay.H
            img = cv2.imread(str(self.work / "mosaic.jpg"))
            self._mosaic = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (W, H)) if img is not None else None
            th, tw = self.title_img().shape[:2]
            mask = np.zeros((H, W), np.float32)
            x0, y0 = (W - tw) // 2, (H - th) // 2
            mask[max(0, y0):y0 + th, max(0, x0):x0 + tw] = 1
            mask = cv2.GaussianBlur(mask, (0, 0), 0.12 * H)
            mask /= max(mask.max(), 1e-6)
            self._intro_dim = (0.55 * (1 - 0.8 * mask))[..., None]   # 0.55 at the edges, ~0.11 behind the text
        if self._mosaic is None:
            return self.full_map(0, reveal=False) if self.tl.get("full_cam") else np.zeros((self.lay.H, self.lay.W, 3), np.uint8)
        frame = ken_burns(self._mosaic, float(u), 0.06)
        return (frame * self._intro_dim).astype(np.uint8)

    def outro_background(self, u: float) -> np.ndarray:
        """Back to the mosaic of all the film's photos, no text: it drifts slowly outward to the full
        grid while the film fades out."""
        self.intro_background(0.0)  # loads the mosaic
        if self._mosaic is None:
            return self.full_map(0, reveal=False) if self.tl.get("full_cam") else np.zeros((self.lay.H, self.lay.W, 3), np.uint8)
        return (ken_burns(self._mosaic, 1.0 - float(u), 0.06) * 0.85).astype(np.uint8)

    def _vignette(self, frame):
        if not hasattr(self, "_vig"):
            H, W = frame.shape[:2]
            y = np.linspace(0, 1, H)[:, None]
            x = np.linspace(0, 1, W)[None, :]
            top = np.clip(1 - y / 0.30, 0, 1) ** 2 * np.clip(1 - x / 0.7, 0, 1) * 0.55
            bot = np.clip((y - 0.62) / 0.38, 0, 1) ** 2 * 0.65
            self._vig = (1 - np.maximum(top, bot))[..., None].astype(np.float32)
        frame[:] = (frame * self._vig).astype(np.uint8)


def load_track(work: Path, plan: dict) -> dict:
    """The movement track, limited to the plan's date range."""
    z = np.load(Path(work) / "index" / "locations.npz")
    loc = {k: z[k] for k in z.files}
    t_from, t_to = plan.get("range") or (None, None)
    m = np.ones(len(loc["t"]), bool)
    if t_from is not None:
        m &= loc["t"] >= t_from
    if t_to is not None:
        m &= loc["t"] <= t_to
    return {k: v[m] for k, v in loc.items()}


def _encoder(path, W, H, fps, crf, preset):
    return subprocess.Popen(
        [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE)


def renderer_for(work: str, opts: dict) -> "Renderer":
    """The two-person renderer for a plan with `persons`, else the single-person one."""
    if json.load(open(Path(work) / "plan.json")).get("persons"):
        from .render_duo import DuoRenderer
        return DuoRenderer(work, opts)
    return Renderer(work, opts)


def _render_chunk(args):
    work, opts, f0, f1, out = args
    cv2.setNumThreads(1)
    r = renderer_for(work, opts)
    W, H = r.plan["size"]
    enc = _encoder(out, W, H, r.plan["fps"], opts["crf"], opts["preset"])
    for f in range(f0, f1):
        enc.stdin.write(r.frame(f).tobytes())
        if (f - f0) % 300 == 0:
            print(f"[vwl] chunk {os.path.basename(out)}: {f - f0}/{f1 - f0}", file=sys.stderr, flush=True)
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")
    return out


def run(work: Path, out: Path, style="dark", workers=8, crf=18, preset="medium", morph=1.0,
        music: str | None = None, preview_at: float | None = None, offline=False,
        tile_url: str | None = None):
    opts = {"style": style, "crf": crf, "preset": preset, "morph": morph, "tile_url": tile_url}
    plan = json.load(open(work / "plan.json"))
    tiles = Tiles(style, offline=offline, url=tile_url)
    if not (work / "mosaic.jpg").exists() and plan["slots"]:
        from .plan import mosaic, plan_items
        print("[vwl] building the photo mosaic for the intro …")
        mosaic(work, plan_items(plan), plan["size"], workers)
    if plan.get("persons") and any(s.get("kind") != "spread" for s in plan["slots"]):
        raise SystemExit(f"{work / 'plan.json'} is from an older version (photo pairs instead of spreads): "
                         "re-run `vwl plan` with the same options")
    if plan.get("persons"):
        from .duo_layout import DuoLayout
        from .render_duo import build_duo_timeline
        lay = DuoLayout(*plan["size"])
        locs = [load_track(Path(p["work"]), plan) for p in plan["persons"]]
        z = np.load(work / "duo.npz")
        tl = build_duo_timeline(plan, locs, lay, tiles.px, {k: z[k] for k in z.files})
    else:
        lay = Layout(*plan["size"])
        tl = build_timeline(plan, load_track(work, plan), lay, tiles.px)
    if tl["has_loc"]:
        tiles.prefetch(needed_tiles(tl, lay, tiles.px))
    if preview_at is not None:
        r = renderer_for(str(work), opts)
        f = min(tl["N"] - 1, int(preview_at * plan["fps"]))
        img = r.frame(f)
        cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"[vwl] wrote preview frame {f} → {out}")
        return
    N = tl["N"]
    # chunk at slot boundaries so each worker builds few morphs twice
    bounds = [0]
    starts = [s[2] for s in tl["segs"] if s[0] == "hold"]
    for w in range(1, workers):
        target = N * w // workers
        b = min(starts, key=lambda x: abs(x - target)) if starts else target
        if b > bounds[-1]:
            bounds.append(b)
    bounds.append(N)
    seg_dir = work / "segments"
    seg_dir.mkdir(exist_ok=True)
    jobs = [(str(work), opts, a, b, str(seg_dir / f"seg{k:03d}.mp4")) for k, (a, b) in enumerate(zip(bounds, bounds[1:])) if b > a]
    print(f"[vwl] rendering {N} frames ({N / plan['fps']:.0f}s) in {len(jobs)} chunks")
    with process_pool(workers) as ex:
        segs = list(ex.map(_render_chunk, jobs))
    lst = seg_dir / "list.txt"
    lst.write_text("".join(f"file '{os.path.abspath(s)}'\n" for s in segs))
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if music:
        dur = N / plan["fps"]
        cmd += ["-i", music, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-af", f"afade=t=in:d=1.5,afade=t=out:st={max(0, dur - 4):.2f}:d=4", "-shortest"]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)
    for seg in segs:
        os.remove(seg)
    print(f"[vwl] wrote {out}")
