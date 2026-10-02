"""Slippy-map rendering with a disk tile cache, a smoothed camera, and track drawing.

Tiles are fetched once (politely, with a User-Agent, a few threads) before rendering
starts and then read from ~/.cache/vwl/tiles.  The default styles use Esri's keyless
ArcGIS Online basemaps (attribution required; fine for personal, non-commercial use).
CARTO's CDN now answers every keyless request with an "API KEY REQUIRED" image, and
the OSM tile servers forbid bulk downloads — for those, or Stadia/MapTiler, pass a
keyed URL template with --tile-url.
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import requests

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
# style: ([base, overlay…] URL templates, tile px, attribution). Overlays are transparent label layers.
STYLES = {
    "dark": ([ESRI.format("Canvas/World_Dark_Gray_Base"), ESRI.format("Canvas/World_Dark_Gray_Reference")], 256,
             "Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "light": ([ESRI.format("Canvas/World_Light_Gray_Base"), ESRI.format("Canvas/World_Light_Gray_Reference")], 256,
              "Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "satellite": ([ESRI.format("World_Imagery"), ESRI.format("Reference/World_Boundaries_and_Places")], 256,
                  "Esri, Maxar, Earthstar Geographics"),
    "topo": ([ESRI.format("World_Topo_Map")], 256, "Esri, HERE, Garmin, © OpenStreetMap contributors"),
    "osm": (["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], 256, "© OpenStreetMap contributors"),
}
MAX_Z = 16
UA = "virtualworldlines/0.1 (personal life-video renderer)"


def to_world(lat, lon):
    """Web-Mercator in [0,1]² (x east, y south)."""
    lat = np.clip(lat, -85.0511, 85.0511)
    x = (np.asarray(lon) + 180.0) / 360.0
    s = np.sin(np.radians(lat))
    y = 0.5 - np.log((1 + s) / (1 - s)) / (4 * np.pi)
    return x, y


class Tiles:
    def __init__(self, style: str = "dark", cache_dir: str | None = None, offline: bool = False, mem_tiles: int = 600,
                 url: str | None = None):
        if url:  # custom provider, e.g. a keyed Stadia/MapTiler/CARTO URL
            self.urls, self.px, self.attribution = [url], 256, "map tiles: see provider terms"
            style = "custom-" + hashlib.sha1(url.encode()).hexdigest()[:8]
        else:
            self.urls, self.px, self.attribution = STYLES[style]
        self.dir = Path(cache_dir or os.path.expanduser("~/.cache/vwl/tiles")) / style
        self.offline = offline
        self.mem = OrderedDict()
        self.mem_tiles = mem_tiles
        self._blank = None

    def path(self, layer, z, x, y) -> Path:
        return self.dir / str(layer) / str(z) / str(x) / f"{y}.img"

    def fetch(self, layer, z, x, y, session=None) -> bool:
        p = self.path(layer, z, x, y)
        if p.exists():
            return True
        if self.offline:
            return False
        url = self.urls[layer].format(s="abc"[(x + y) % 3], z=z, x=x, y=y)
        try:
            r = (session or requests).get(url, headers={"User-Agent": UA}, timeout=20)
            if r.status_code != 200 or not r.content or not r.headers.get("content-type", "").startswith("image"):
                return False
        except requests.RequestException:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(r.content)
        os.replace(tmp, p)
        return True

    def prefetch(self, zxys, threads: int = 6):
        todo = [(l, *t) for t in zxys for l in range(len(self.urls)) if not self.path(l, *t).exists()]
        if not todo or self.offline:
            return 0
        print(f"[vwl] fetching {len(todo)} map tiles …")
        local = threading.local()

        def job(t):
            if not hasattr(local, "s"):
                local.s = requests.Session()
            return self.fetch(*t, session=local.s)

        with ThreadPoolExecutor(threads) as ex:
            ok = sum(ex.map(job, todo))
        if ok < len(todo):
            print(f"[vwl] {len(todo) - ok} tiles unavailable (offline or outside coverage) — drawn blank")
        return ok

    def _read(self, layer, z, x, y):
        p = self.path(layer, z, x, y)
        if not p.exists():
            return None
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[0] != self.px:
            img = cv2.resize(img, (self.px, self.px), interpolation=cv2.INTER_AREA)
        return img

    def get(self, z, x, y) -> np.ndarray:
        n = 1 << z
        x %= n
        key = (z, x, y)
        t = self.mem.get(key)
        if t is not None:
            self.mem.move_to_end(key)
            return t
        img = None
        if 0 <= y < n:
            base = self._read(0, z, x, y)
            if base is not None:
                img = cv2.cvtColor(base[..., :3], cv2.COLOR_BGR2RGB)
        if img is None:
            if self._blank is None:
                self._blank = np.full((self.px, self.px, 3), (22, 24, 30), np.uint8)
            img = self._blank
        self.mem[key] = img
        if len(self.mem) > self.mem_tiles:
            self.mem.popitem(last=False)
        return img

    def get_labels(self, z, x, y) -> np.ndarray | None:
        """The style's transparent label layers (place names, borders) as one RGBA tile, or None."""
        if len(self.urls) < 2:
            return None
        n = 1 << z
        x %= n
        key = ("labels", z, x, y)
        if key in self.mem:
            self.mem.move_to_end(key)
            return self.mem[key]
        out = None
        if 0 <= y < n:
            for layer in range(1, len(self.urls)):
                ov = self._read(layer, z, x, y)
                if ov is None or ov.shape[2] != 4:
                    continue
                ov = cv2.cvtColor(ov, cv2.COLOR_BGRA2RGBA)
                if out is None:
                    out = ov
                else:
                    a = ov[..., 3:4].astype(np.float32) / 255
                    out = (out * (1 - a) + ov * a).astype(np.uint8)
        self.mem[key] = out
        if len(self.mem) > self.mem_tiles:
            self.mem.popitem(last=False)
        return out


def view_tiles(cx, cy, z, w, h, tile_px):
    """Integer zoom, fractional scale and tile index ranges covering a view."""
    zi = int(min(MAX_Z, max(0, math.ceil(z - 1e-6))))
    s = 2 ** (z - zi)  # ≤ 1: render at zi and shrink
    world = tile_px * (1 << zi)
    hw, hh = w / 2 / s, h / 2 / s
    x0, x1 = cx * world - hw, cx * world + hw
    y0, y1 = cy * world - hh, cy * world + hh
    return zi, s, world, (x0, y0, x1, y1), (int(x0 // tile_px), int(x1 // tile_px), int(y0 // tile_px), int(y1 // tile_px))


def zoom_for_bbox(x0, y0, x1, y1, w, h, tile_px, pad=1.35, zmin=1.0, zmax=14.5):
    sx, sy = max(x1 - x0, 1e-9) * pad, max(y1 - y0, 1e-9) * pad
    z = math.log2(min(w / (sx * tile_px), h / (sy * tile_px)))
    return float(np.clip(z, zmin, zmax))


class MapView:
    """Draws one map panel: basemap, lifetime trace (faint), recent trail, position marker."""

    TRACE = np.array([90, 200, 255], np.float32)
    TRAIL = (255, 146, 64)

    def __init__(self, tiles: Tiles, loc: dict, w: int, h: int):
        self.tiles, self.w, self.h = tiles, w, h
        self.t = loc["t"]
        self.mx, self.my = to_world(loc["lat"], loc["lon"])
        self.grid_cache = None

    def basemap(self, cx, cy, z) -> np.ndarray:
        tp = self.tiles.px
        zi, s, world, (x0, y0, x1, y1), (tx0, tx1, ty0, ty1) = view_tiles(cx, cy, z, self.w, self.h, tp)
        cw, ch = (tx1 - tx0 + 1) * tp, (ty1 - ty0 + 1) * tp
        canvas = np.empty((ch, cw, 3), np.uint8)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                canvas[(ty - ty0) * tp:(ty - ty0 + 1) * tp, (tx - tx0) * tp:(tx - tx0 + 1) * tp] = self.tiles.get(zi, tx, ty)
        # affine: canvas px -> panel px
        ox, oy = x0 - tx0 * tp, y0 - ty0 * tp
        M = np.array([[s, 0, -ox * s], [0, s, -oy * s]], np.float32)
        interp = cv2.INTER_AREA if s < 0.999 else cv2.INTER_LINEAR
        if s < 0.999:  # INTER_AREA isn't supported by warpAffine; pre-shrink
            small = cv2.resize(canvas, None, fx=s, fy=s, interpolation=interp)
            M = np.array([[1, 0, -ox * s], [0, 1, -oy * s]], np.float32)
            return cv2.warpAffine(small, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.warpAffine(canvas, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def project(self, mx, my, cx, cy, z):
        scale = self.tiles.px * 2 ** z
        return (mx - cx) * scale + self.w / 2, (my - cy) * scale + self.h / 2

    def labels(self, cx, cy, z):
        """Place-name layer rendered from tiles one zoom level coarser, i.e. 1-2× the native text
        size (the base map itself is drawn from finer tiles and shrunk, which makes text tiny)."""
        tp = self.tiles.px
        zl = max(0.0, z - 1.0)
        zi, s, world, (x0, y0, x1, y1), (tx0, tx1, ty0, ty1) = view_tiles(cx, cy, zl, self.w, self.h, tp)
        s *= 2 ** (z - zl)  # scale from the coarser tiles to the requested zoom
        cw, ch = (tx1 - tx0 + 1) * tp, (ty1 - ty0 + 1) * tp
        canvas = np.zeros((ch, cw, 4), np.uint8)
        anything = False
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                t = self.tiles.get_labels(zi, tx, ty)
                if t is not None:
                    canvas[(ty - ty0) * tp:(ty - ty0 + 1) * tp, (tx - tx0) * tp:(tx - tx0 + 1) * tp] = t
                    anything = True
        if not anything:
            return None
        world_z = tp * 2 ** z
        ox = cx * world_z - self.w / 2 - tx0 * tp * s
        oy = cy * world_z - self.h / 2 - ty0 * tp * s
        M = np.array([[s, 0, -ox], [0, s, -oy]], np.float32)
        return cv2.warpAffine(canvas, M, (self.w, self.h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _labels_over(self, img, cx, cy, z):
        lab = self.labels(cx, cy, z)
        if lab is None:
            return img
        a = np.clip(lab[..., 3:4].astype(np.float32) / 255 * 1.6, 0, 1)
        rgb = 255 - (255 - lab[..., :3].astype(np.float32)) * 0.35   # brighten grey text toward white
        return (img * (1 - a) + rgb * a).astype(np.uint8)

    def draw_people(self, cx, cy, z, people: list[dict], dim=0.85, pulse=0.0, lanes: float = 0.0,
                    trace_only=False) -> np.ndarray:
        """One or more people on one map.  Each dict: track (t, mx, my), tau (life time: the faint
        lifetime trace is drawn up to it), history ((points, ages) from camera.trail, or None), pos
        (or None), color, and optionally trace_color.  `lanes` > 0 offsets each trail sideways by that
        many pixels, so two people travelling together read as two lanes of one road."""
        img = self.basemap(cx, cy, z)
        if dim != 1.0:
            img = cv2.convertScaleAbs(img, alpha=dim)
        for p in people:
            tc = p.get("trace_color")
            trace_col = np.array(tc if tc is not None else np.array(p["color"], np.float32) * 0.55 + 255 * 0.25,
                                 np.float32)
            img = self._trace(img, p["track"], p["tau"], None, cx, cy, z, trace_col)
        img = self._labels_over(img, cx, cy, z)
        if not trace_only:
            n = len(people)
            for k, p in enumerate(people):
                if p.get("history") is not None:
                    img = self._trail(img, *p["history"], cx, cy, z, p["color"], (k - (n - 1) / 2) * lanes)
            for k, p in enumerate(people):
                if p.get("pos") is not None:
                    off = (k - (n - 1) / 2) * lanes * 1.6
                    img = self._dot(img, p["pos"], cx, cy, z, p["color"], pulse, shift=(off, 0))
        return self._finish(img, cy, z)

    def _trace(self, img, track, upto, reveal_from, cx, cy, z, color):
        """Lifetime trace up to `upto`: a faint glow, density-accumulated."""
        t, mx, my = track
        w, h = self.w, self.h
        k = int(np.searchsorted(t, upto, side="right"))
        k0 = 0 if reveal_from is None else int(np.searchsorted(t, reveal_from))
        if k > k0:
            px, py = self.project(mx[k0:k], my[k0:k], cx, cy, z)
            m = (px >= 0) & (px < w) & (py >= 0) & (py < h)
            if m.any():
                idx = py[m].astype(np.int32) * w + px[m].astype(np.int32)
                cnt = np.bincount(idx, minlength=w * h).reshape(h, w).astype(np.float32)
                heat = 1 - np.exp(-cnt * 0.6)
                heat = cv2.GaussianBlur(heat, (0, 0), 1.2) * 1.6 + heat * 0.6
                heat = np.clip(heat, 0, 1)[..., None] * 0.75
                img = (img * (1 - heat) + color * heat).astype(np.uint8)
        return img

    def _trail(self, img, pts, ages, cx, cy, z, color, offset: float = 0.0, buckets: int = 8):
        """The dot's recent path; parts fade (darken into the map) with their age 0..1 (1 = oldest).
        Long hops (flights, data gaps) are drawn as arcs; `offset` shifts the path sideways (px)."""
        if len(pts) < 2:
            return img
        w, h = self.w, self.h
        px, py = self.project(pts[:, 0], pts[:, 1], cx, cy, z)
        xy = np.stack([px, py], 1)
        jump = max(w, h) * 0.35
        overlay = img.copy()
        b = np.minimum((ages * buckets).astype(int), buckets - 1)
        for k in range(buckets):  # oldest bucket first, faintest
            kk = buckets - 1 - k
            idx = np.flatnonzero(b[:-1] == kk)
            if not len(idx):
                continue
            fresh = 1 - (kk + 0.5) / buckets
            col = tuple(int(v * (0.15 + 0.85 * fresh ** 1.2)) for v in color)
            # consecutive segments in a bucket become one polyline
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for r in runs:
                seg = xy[r[0]:r[-1] + 2]
                seg = _with_arcs(seg, jump)
                if offset:
                    seg = _offset(seg, offset)
                cv2.polylines(overlay, [np.round(seg * 16).astype(np.int32)], False, col, 3, cv2.LINE_AA, shift=4)
        return cv2.addWeighted(overlay, 0.9, img, 0.1, 0)

    def _dot(self, img, pos, cx, cy, z, color, pulse, shift=(0.0, 0.0)):
        x, y = self.project(np.array([pos[0]]), np.array([pos[1]]), cx, cy, z)
        x, y = int((x[0] + shift[0]) * 16), int((y[0] + shift[1]) * 16)
        r = int((10 + 14 * pulse) * 16)
        ring = img.copy()
        cv2.circle(ring, (x, y), r, color, 2, cv2.LINE_AA, shift=4)
        img = cv2.addWeighted(ring, 1 - pulse, img, pulse, 0)
        cv2.circle(img, (x, y), 7 * 16, color, -1, cv2.LINE_AA, shift=4)
        cv2.circle(img, (x, y), 3 * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
        return img

    def _finish(self, img, cy, z):
        self._scalebar(img, cy, z)
        cv2.putText(img, self.tiles.attribution, (8, self.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1,
                    cv2.LINE_AA)
        return img

    def _scalebar(self, img, cy, z):
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * cy))))
        mpp = 40075016.686 * math.cos(math.radians(lat)) / (self.tiles.px * 2 ** z)
        target = mpp * self.w * 0.22
        e = 10 ** math.floor(math.log10(target))
        nice = max(v * e for v in (1, 2, 5) if v * e <= target)
        px = int(nice / mpp)
        x0, y0 = self.w - px - 14, self.h - 14
        cv2.line(img, (x0, y0), (x0 + px, y0), (230, 230, 230), 2, cv2.LINE_AA)
        for xx in (x0, x0 + px):
            cv2.line(img, (xx, y0 - 5), (xx, y0), (230, 230, 230), 2, cv2.LINE_AA)
        label = f"{nice / 1000:g} km" if nice >= 1000 else f"{nice:g} m"
        cv2.putText(img, label, (x0, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def _offset(poly: np.ndarray, d: float) -> np.ndarray:
    """Shift a polyline sideways by d pixels along its local normals."""
    if len(poly) < 2 or not d:
        return poly
    tang = np.gradient(poly, axis=0)
    norm = np.hypot(tang[:, 0], tang[:, 1])[:, None]
    tang = np.where(norm > 1e-6, tang / np.maximum(norm, 1e-6), 0)
    return poly + d * np.c_[-tang[:, 1], tang[:, 0]]


def _with_arcs(pts: np.ndarray, jump: float) -> np.ndarray:
    """Replace long straight hops (flights, data gaps) by gentle arcs."""
    if len(pts) < 2:
        return pts
    d = np.hypot(*np.diff(pts, axis=0).T)
    if not (d > jump).any():
        return pts
    out = [pts[:1]]
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        if d[i - 1] > jump:
            mid = (p0 + p1) / 2
            nrm = np.array([-(p1 - p0)[1], (p1 - p0)[0]]) * 0.18
            ctrl = mid + (nrm if nrm[1] < 0 else -nrm)  # bow upwards
            s = np.linspace(0, 1, 24)[1:, None]
            out.append((1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s ** 2 * p1)
        else:
            out.append(p1[None])
    return np.concatenate(out)


# ---------------------------------------------------------------- camera

def robust_bbox(mx, my, lo=2, hi=98):
    if len(mx) == 0:
        return None
    if len(mx) > 20:
        return (np.percentile(mx, lo), np.percentile(my, lo), np.percentile(mx, hi), np.percentile(my, hi))
    return mx.min(), my.min(), mx.max(), my.max()


def union(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def min_span(bb, span):
    x0, y0, x1, y1 = bb
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) / 2, span / 2), max((y1 - y0) / 2, span / 2)
    return cx - hw, cy - hh, cx + hw, cy + hh


class Locator:
    """A small world map showing where a big map is looking: a rectangle for the view (a ring when the
    view is too small to see at world scale) and a dot per person."""

    def __init__(self, tiles: Tiles, w: int, h: int):
        self.w, self.h = w, h
        self.cy = 0.47                              # a little north of the equator: more land, less ocean
        self.y_top = self.cy - h / (2 * w)
        view = MapView(tiles, {"t": np.zeros(0), "lat": np.zeros(0), "lon": np.zeros(0)}, w, h)
        z = np.log2(w / tiles.px)                   # the whole world is exactly w pixels wide
        base = view.basemap(0.5, self.cy, z)
        self.base = cv2.convertScaleAbs(base, alpha=1.7, beta=6)   # the dark style is too dark this small
        cv2.rectangle(self.base, (0, 0), (w - 1, h - 1), (120, 120, 128), 1, cv2.LINE_AA)

    def to_px(self, mx, my):
        return mx * self.w, (my - self.y_top) * self.w

    def render(self, cx, cy, z, view_w, view_h, tile_px, dots=()) -> np.ndarray:
        img = self.base.copy()
        scale = tile_px * 2 ** z                    # big-map pixels per world unit
        hw, hh = view_w / 2 / scale, view_h / 2 / scale
        x0, y0 = self.to_px(cx - hw, cy - hh)
        x1, y1 = self.to_px(cx + hw, cy + hh)
        if x1 - x0 >= 5 and y1 - y0 >= 4:
            cv2.rectangle(img, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))),
                          (255, 255, 255), 1, cv2.LINE_AA)
        else:
            px, py = self.to_px(cx, cy)
            cv2.circle(img, (int(round(px)), int(round(py))), 5, (255, 255, 255), 1, cv2.LINE_AA)
        for pos, color in dots:
            if pos is not None:
                px, py = self.to_px(pos[0], pos[1])
                cv2.circle(img, (int(round(px)), int(round(py))), 2, tuple(int(c) for c in color), -1, cv2.LINE_AA)
        return img

    def paste(self, panel: np.ndarray, cx, cy, z, tile_px, dots=(), margin: int = 8):
        """Draw the locator into the upper-right corner of a rendered map panel (in place)."""
        ph, pw = panel.shape[:2]
        if pw < self.w + 2 * margin or ph < self.h + 2 * margin:
            return
        img = self.render(cx, cy, z, pw, ph, tile_px, dots)
        region = panel[margin:margin + self.h, pw - margin - self.w:pw - margin]
        region[:] = (region * 0.15 + img * 0.85).astype(np.uint8)


def world_tiles() -> set:
    """Tiles the locator map is drawn from (zoom 0 and 1)."""
    return {(0, 0, 0)} | {(1, x, y) for x in range(2) for y in range(2)}
