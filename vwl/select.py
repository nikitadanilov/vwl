"""Pick a few hundred photos that represent a life, out of tens of thousands.

1. Drop obvious non-memories (screenshots, tiny images, received memes get a penalty).
2. Group photos into *events*: runs separated by >4 h or by a >30 km jump.
3. Give every year a quota ∝ sqrt(photos that year): photo-heavy recent years
   don't drown out the sparse early ones, and every year that has photos appears.
4. Inside a year, rank events by sqrt(size) × (trip bonus if far from home) and give
   the quota to the top events, spread through the year.
5. Decode a few candidates per chosen slot, score sharpness/exposure/colour and
   metadata (favourite, shared to social, views, caption), drop near-duplicates by
   perceptual hash, keep the best — spread in time inside the event.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import video as V
from .archive import process_pool, read_uri
from .model import Photo

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

SKIP = re.compile(r"(?i)screenshot|screen[ _-]?shot|screen[ _-]?record|rpreplay|снимок экрана|скриншот|bildschirmfoto|captura")
RECEIVED = re.compile(r"(?i)-WA\d+|^received_|telegram|viber|^FB_IMG|^Messenger_")


def load_image(uri: str, max_side: int | None = None, draft: bool = False) -> Image.Image:
    im = Image.open(BytesIO(read_uri(uri)))
    if draft and max_side and im.format == "JPEG":
        im.draft("RGB", (max_side, max_side))  # fast DCT-domain downscale
    im = ImageOps.exif_transpose(im)
    im = im.convert("RGB")
    if max_side and max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    return im


def year_of(ts: float) -> int:
    return datetime.fromtimestamp(ts, timezone.utc).year


@dataclass
class Event:
    photos: list[Photo]
    lat: float | None
    lon: float | None

    @property
    def t0(self):
        return self.photos[0].ts

    @property
    def t1(self):
        return self.photos[-1].ts


def _km(a_lat, a_lon, b_lat, b_lon):
    p = math.pi / 180
    x = (b_lon - a_lon) * p * math.cos((a_lat + b_lat) * p / 2)
    y = (b_lat - a_lat) * p
    return 6371 * math.hypot(x, y)


def events_of(photos: list[Photo]) -> list[Event]:
    events, cur, last_geo = [], [], None
    for p in photos:
        new = bool(cur) and p.ts - cur[-1].ts > 4 * 3600
        if cur and p.lat is not None and last_geo and _km(p.lat, p.lon, *last_geo) > 30:
            new = True
        if new:
            events.append(cur)
            cur = []
        cur.append(p)
        if p.lat is not None:
            last_geo = (p.lat, p.lon)
    if cur:
        events.append(cur)
    out = []
    for ev in events:
        geo = [(p.lat, p.lon) for p in ev if p.lat is not None]
        lat = float(np.median([g[0] for g in geo])) if geo else None
        lon = float(np.median([g[1] for g in geo])) if geo else None
        out.append(Event(ev, lat, lon))
    return out


def home_of(events: list[Event]):
    """Most photographed ~10 km cell."""
    cells = defaultdict(int)
    for e in events:
        if e.lat is not None:
            cells[(round(e.lat * 10), round(e.lon * 10))] += len(e.photos)
    if not cells:
        return None
    c = max(cells, key=cells.get)
    return c[0] / 10, c[1] / 10


def meta_score(p: Photo) -> float:
    s = 0.0
    s += 3.0 if p.favorite else 0
    s += 2.0 if p.shared else 0
    s += 0.6 * math.log1p(p.views)
    s += 1.0 if p.description else 0
    s += 0.3 if p.lat is not None else 0
    s -= 1.5 if RECEIVED.search(p.title) else 0
    return s


def _largest_remainder(weights: dict, total: int, caps: dict | None = None) -> dict:
    tw = sum(weights.values()) or 1
    raw = {k: total * w / tw for k, w in weights.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    if caps:
        alloc = {k: min(v, caps[k]) for k, v in alloc.items()}
    rest = total - sum(alloc.values())
    for k in sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True):
        if rest <= 0:
            break
        if not caps or alloc[k] < caps[k]:
            alloc[k] += 1
            rest -= 1
    return alloc


_nsfw = None
NSFW_CLASSES = {"FEMALE_BREAST_EXPOSED", "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED",
                "BUTTOCKS_EXPOSED", "ANUS_EXPOSED"}


def _square_crops(im: np.ndarray):
    """NudeNet letterboxes to a 320 px square, which shrinks people in tall portraits to
    nothing; scan overlapping square crops along the long side instead."""
    h, w = im.shape[:2]
    s = min(h, w)
    n = max(1, round(max(h, w) / s * 1.5))
    for k in range(n + 1):
        o = int((max(h, w) - s) * k / n)
        yield im[o:o + s, :] if h > w else im[:, o:o + s]


def is_nsfw(rgb: np.ndarray, threshold: float = 0.35) -> bool:
    """Best-effort local NudeNet check (nothing leaves the machine); still review the
    contact sheet.  False if NudeNet isn't installed."""
    global _nsfw
    if _nsfw is None:
        try:
            import onnxruntime
            from nudenet import NudeDetector
            onnxruntime.set_default_logger_severity(3)
            _nsfw = NudeDetector()
            # one inference thread per worker process (the default uses every core in every worker)
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            path = _nsfw.onnx_session._model_path if hasattr(_nsfw.onnx_session, "_model_path") else None
            if path:
                _nsfw.onnx_session = onnxruntime.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except ImportError:
            _nsfw = False
    if not _nsfw:
        return False
    for crop in _square_crops(rgb):
        bgra = cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_RGB2BGRA)
        if any(d["class"] in NSFW_CLASSES and d["score"] >= threshold for d in _nsfw.detect(bgra)):
            return True
    return False


def quality(uri: str, nsfw_filter: bool = True):
    """(score, dhash, size) from a small decode; None if undecodable or filtered out."""
    try:
        im = load_image(uri, 640, draft=True)
    except Exception:  # noqa: BLE001
        return None
    if nsfw_filter and is_nsfw(np.asarray(im)):
        return "nsfw"
    im.thumbnail((384, 384))
    w, h = im.size
    a = np.asarray(im, dtype=np.float32)
    g = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_32F).var()
    mean, std = g.mean(), g.std()
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colorful = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
    s = (1.2 * min(math.log1p(sharp) / 6.0, 1.3)       # focus
         - 2.0 * max(0, abs(mean - 118) / 118 - 0.45)   # badly exposed
         + 0.8 * min(std / 60, 1.0)                     # contrast
         + 0.5 * min(colorful / 50, 1.2))               # not a grey document
    # documents, receipts, screens: mostly bright paper/backlight and little colour
    if float((g > 170).mean()) > 0.4 and colorful < 30:
        s -= 3.0
    ar = max(w, h) / min(w, h)
    if ar > 2.3:                                        # panoramas/strips crop badly
        s -= 1.0 if ar < 3 else 3.0
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten()), 2)
    return s, dh, (w, h)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def select(photos: list[Photo], n: int, t_from=None, t_to=None, workers: int = 8, seed: int = 0,
           exclude: set[str] = frozenset(), nsfw_filter: bool = True, video_share: float = 0.15,
           hold_video: float = 4.0, cache_dir: Path | None = None) -> list[Photo]:
    """Returns the picked items; for picked videos, select.video_info[uri] holds the clip window."""
    rng = np.random.default_rng(seed)
    max_videos = math.floor(video_share * n + 1e-9)
    cands = [p for p in photos
             if not SKIP.search(p.title) and (t_from is None or p.ts >= t_from) and (t_to is None or p.ts <= t_to)
             and p.uri not in exclude and p.title not in exclude and (p.kind == "image" or max_videos > 0)]
    if not cands:
        return []
    events = events_of(cands)
    home = home_of(events)
    by_year = defaultdict(list)
    for e in events:
        by_year[year_of(e.t0)].append(e)
    year_w = {y: math.sqrt(sum(len(e.photos) for e in evs)) for y, evs in by_year.items()}
    year_caps = {y: sum(len(e.photos) for e in evs) for y, evs in by_year.items()}
    quota = _largest_remainder(year_w, min(n, len(cands)), year_caps)
    # guarantee each year with photos at least one slot when n allows
    if n >= len(by_year):
        for y in quota:
            if quota[y] == 0:
                donor = max(quota, key=quota.get)
                quota[donor] -= 1
                quota[y] = 1

    slots: list[tuple[Event, int]] = []
    for y, evs in by_year.items():
        q = quota[y]
        if q <= 0:
            continue

        def ev_w(e: Event):
            w = math.sqrt(len(e.photos)) * (1 + 0.15 * sum(meta_score(p) > 1.5 for p in e.photos))
            if home and e.lat is not None and _km(e.lat, e.lon, *home) > 100:
                w *= 1.8  # trips matter
            return w * (0.9 + 0.2 * rng.random())

        ranked = sorted(evs, key=ev_w, reverse=True)
        chosen = ranked[:q]
        weights = {i: ev_w(e) for i, e in enumerate(chosen)}
        caps = {i: max(1, len(e.photos) // 3) for i, e in enumerate(chosen)}
        per = _largest_remainder(weights, q, caps)
        if sum(per.values()) < q:  # small events everywhere: take next-ranked events
            extra = ranked[q:q + (q - sum(per.values()))]
            chosen += extra
            per.update({len(chosen) - len(extra) + i: 1 for i in range(len(extra))})
        slots += [(e, per.get(i, 1)) for i, e in enumerate(chosen) if per.get(i, 1) > 0]

    # candidates: up to 4 per slot, spread in time, favourites first
    to_score = {}
    plan = []
    for e, k in slots:
        ps = e.photos
        pool = sorted(ps, key=meta_score, reverse=True)[:k]
        m = min(len(ps), 4 * k)
        pool += [ps[int(i)] for i in np.linspace(0, len(ps) - 1, m)]
        if max_videos:  # an event's best videos always get a chance, not only if sampled
            pool += sorted((p for p in ps if p.kind == "video"), key=meta_score, reverse=True)[:2]
        pool = list({p.uri: p for p in pool}.values())
        plan.append((e, k, pool))
        for p in pool:
            to_score[p.uri] = p
    uris = [u for u, p in to_score.items() if p.kind == "image"]
    # videos are expensive to score (extract + decode): only the most promising 2× the cap
    vids = sorted((p for p in to_score.values() if p.kind == "video"), key=meta_score, reverse=True)[:2 * max_videos]
    vdir = Path(cache_dir or Path("work/cache/videos"))
    vargs = [(p.uri, p.favorite, hold_video, str(vdir), nsfw_filter) for p in vids]
    # scores persist across runs (work/cache/scores.pkl): re-planning only scores what's new
    store = ScoreCache(vdir.parent / "scores.pkl")
    ikey = lambda u: ("img", u, nsfw_filter)  # noqa: E731
    vkey = lambda a: ("vid", a[0], a[2], a[4])  # noqa: E731
    todo_i = [u for u in uris if ikey(u) not in store]
    todo_v = [a for a in vargs if not store.video_ok(vkey(a))]
    if len(todo_i) < len(uris) or len(todo_v) < len(vargs):
        print(f"[vwl] scores cached from earlier runs: {len(uris) - len(todo_i)} photos, {len(vargs) - len(todo_v)} videos")
    with process_pool(workers) as ex:
        for u, r in zip(todo_i, _progress(ex.map(quality, todo_i, [nsfw_filter] * len(todo_i), chunksize=4),
                                          len(todo_i), "photo candidates scored")):
            store[ikey(u)] = r
        for a, r in zip(todo_v, _progress(ex.map(video_quality, todo_v, chunksize=1), len(todo_v),
                                          "video candidates scored")):
            store[vkey(a)] = r
    store.save()
    q = {u: store[ikey(u)] for u in uris}
    vres = [store[vkey(a)] for a in vargs]
    select.video_info = {}
    for p, r in zip(vids, vres):
        if isinstance(r, tuple):
            q[p.uri] = r[:3]
            select.video_info[p.uri] = r[3]
        elif r == "nsfw":
            q[p.uri] = r
    if vids:
        print(f"[vwl] {len(select.video_info)} of {len(vids)} video candidates usable")
    # the cap is a ceiling: only the best-scoring usable videos may be used at all
    allowed = set(sorted(select.video_info, key=lambda u: q[u][0] + meta_score(to_score[u]),
                         reverse=True)[:max_videos])
    flagged = [u for u, v in q.items() if v == "nsfw"]
    if flagged:
        print(f"[vwl] {len(flagged)} candidates flagged as nudity and skipped (list: nsfw_flagged.txt)")
    select.flagged = flagged
    q = {u: v for u, v in q.items() if v != "nsfw"}

    picked: list[Photo] = []
    for e, k, pool in plan:
        scored = [(p, q[p.uri]) for p in pool if q.get(p.uri) and (p.kind == "image" or p.uri in allowed)]
        if not scored:
            continue
        t0, t1 = e.t0, e.t1 + 1
        chosen = []
        for p, (s, dh, size) in sorted(scored, key=lambda x: x[1][0] + meta_score(x[0]), reverse=True):
            if len(chosen) >= k:
                break
            if any(_hamming(dh, d2) < 12 for _, d2, _ in chosen):
                continue
            # spread: penalise photos within 1/(2k) of the event span of an already chosen one
            if any(abs(p.ts - c.ts) < (t1 - t0) / (3 * k) for c, _, _ in chosen) and len(scored) > 2 * k:
                continue
            chosen.append((p, dh, size))
        picked += [c for c, _, _ in chosen]
    return sorted(picked, key=lambda p: p.ts)


class ScoreCache:
    """Candidate scores kept between runs.  Keys: ("img", uri, nsfw_filter) and
    ("vid", uri, hold_video, nsfw_filter); a cached video result is only reused while its extracted
    file still exists (the video cache is pruned)."""

    def __init__(self, path: Path):
        import pickle
        self.path = path
        self.d = {}
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self.d = pickle.load(f)
            except Exception:  # noqa: BLE001 - a corrupt cache is just rebuilt
                self.d = {}

    def __contains__(self, k):
        return k in self.d

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v

    def video_ok(self, k) -> bool:
        r = self.d.get(k, False)
        if r is False:
            return False
        return not isinstance(r, tuple) or Path(r[3]["local"]).exists()

    def save(self):
        import os
        import pickle
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.d, f)
        os.replace(tmp, self.path)


def _progress(it, total: int, what: str, every: float = 0.1):
    """Pass items through, printing '[vwl] k/total what' about every 10%."""
    step = max(1, int(total * every))
    for k, x in enumerate(it, 1):
        if total >= 50 and (k % step == 0 or k == total):
            print(f"[vwl] {k}/{total} {what}", flush=True)
        yield x


def video_quality(args):
    """(score, dhash, size, clip info) for a video candidate; None if unusable, "nsfw" if flagged."""
    uri, favorite, hold_video, cache_dir, nsfw_filter = args
    try:
        if V.member_size(uri) > V.MAX_BYTES and not favorite:
            return None
        path = V.local_path(uri, Path(cache_dir))
        info = V.probe(path)
        if not info or info["duration"] < V.MIN_CLIP:
            return None
        times, frames = V.sample(path, info)
        if not len(frames):
            return None
        qs, shake = V.frame_scores(frames)
        start, score = V.best_window(times, qs, shake, info["duration"], hold_video)
        length = min(hold_video, info["duration"] - start)
        mid = frames[int(np.argmin(np.abs(times - (start + length / 2))))]
        if nsfw_filter:
            for t in (start + 0.2 * length, start + 0.5 * length, start + 0.8 * length):
                f = V.grab(path, t, 640, info)
                if f is not None and is_nsfw(f):
                    return "nsfw"
    except Exception:  # noqa: BLE001 - a broken video must not stop planning
        return None
    g = cv2.cvtColor(mid, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    dh = int("".join("1" if b else "0" for b in (small[:, 1:] > small[:, :-1]).flatten()), 2)
    info.update(start=start, length=length, local=str(path))
    return score, dh, (info["width"], info["height"]), info
