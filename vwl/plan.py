"""`vwl plan`: choose photos, captions and pacing → <work>/plan.json.

plan.json is meant to be hand-edited: delete a slot, change a caption or a hold
time — then `vwl render` again.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import index
from .archive import process_pool
from . import video as V
from .select import load_image, select


def local_dt(ts: float, lon: float | None) -> datetime:
    """Approximate local time from longitude (good enough for a date caption)."""
    off = 0 if lon is None else round(lon / 15)
    return datetime.fromtimestamp(ts, timezone.utc) + timedelta(hours=off)


def fmt_date(ts, lon=None) -> str:
    d = local_dt(ts, lon)
    return f"{d.day} {d.strftime('%B %Y')}"


class Track:
    def __init__(self, loc: dict):
        self.t, self.lat, self.lon = loc["t"], loc["lat"], loc["lon"]

    def at(self, ts: float, max_gap: float = 3 * 3600):
        if len(self.t) == 0:
            return None
        i = int(np.searchsorted(self.t, ts))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(self.t[j] - ts) <= max_gap:
                if best is None or abs(self.t[j] - ts) < abs(self.t[best] - ts):
                    best = j
        return None if best is None else (float(self.lat[best]), float(self.lon[best]))


class Geocoder:
    def __init__(self):
        try:
            import reverse_geocoder as rg
            self.rg = rg
        except ImportError:
            self.rg = None
        try:
            import pycountry
            self.pc = pycountry
        except ImportError:
            self.pc = None

    def country(self, cc):
        if self.pc:
            c = self.pc.countries.get(alpha_2=cc)
            if c:
                return getattr(c, "common_name", None) or c.name
        return cc

    def many(self, coords):
        if not self.rg or not coords:
            return [None] * len(coords)
        return self.rg.search(coords, mode=1, verbose=False)


def _export(args):
    """Photo → local JPEG; video → transcoded clip + a poster frame for the contact sheet."""
    s, work, size, fps = args
    if s.get("kind") == "video":
        src = Path(s["local"])
        if not src.exists():  # pruned from the cache since it was scored: extract again
            src = V.local_path(s["uri"], work / "cache" / "videos")
        info = {"width": s["width"], "height": s["height"]}
        # the outgoing clip keeps playing through the morph that follows it
        V.transcode(src, work / s["file"], s["clip_start"], s["hold"] + s["trans"], size, fps, s["hdr"])
        f = V.grab(src, s["clip_start"] + s["hold"] / 2, 480, info)
        if f is not None:
            Image.fromarray(f).save(work / s["poster"], quality=88)
        return
    im = load_image(s["uri"], int(max(size) * 1.25))
    im.save(work / s["file"], quality=92)


def describe(picked, loc, places, geo, hold: float, trans: float, vinfo: dict) -> list[dict]:
    """Slots for picked items: where (own GPS, else the track), place name, date, timing, clip window."""
    track = Track(loc)
    coords = []
    for p in picked:
        ll = (p.lat, p.lon) if p.lat is not None else track.at(p.ts)
        coords.append(ll)
    rg_res = geo.many([c for c in coords if c])
    it = iter(rg_res)
    rg_by_i = [next(it) if c else None for c in coords]
    ccs = [r["cc"] for r in rg_by_i if r]
    home_cc = max(set(ccs), key=ccs.count) if ccs else None
    place_t0 = [pl.t0 for pl in places]

    def place_name(i, p):
        ll = coords[i]
        name = ""
        j = bisect_left(place_t0, p.ts - 86400)
        while j < len(places) and places[j].t0 <= p.ts + 1800:
            pl = places[j]
            if pl.t0 - 1800 <= p.ts <= pl.t1 + 1800 and (not ll or abs(pl.lat - ll[0]) + abs(pl.lon - ll[1]) < 0.02):
                name = {"Inferred Home": "Home", "Inferred Work": "Work"}.get(pl.name, pl.name)
            j += 1
        r = rg_by_i[i]
        if r:
            city = r["name"]
            where = f"{city}, {r['admin1']}" if r["cc"] == home_cc and r["admin1"] else f"{city}, {geo.country(r['cc'])}"
            name = f"{name} · {where}" if name and name != city else where
        return name

    slots = []
    for i, p in enumerate(picked):
        lon = coords[i][1] if coords[i] else None
        s = {
            "uri": p.uri, "ts": p.ts, "kind": p.kind, "lat": coords[i][0] if coords[i] else None, "lon": lon,
            "source": p.source, "date": fmt_date(p.ts, lon), "year": local_dt(p.ts, lon).year,
            "place": place_name(i, p), "caption": p.description[:160],
            "hold": hold, "trans": trans if i + 1 < len(picked) else 0.0,
        }
        if p.kind == "video":
            v = vinfo[p.uri]
            # the clip is shown for its own length; the map's clock follows the footage
            s.update(ts=p.ts + v["start"], clip_start=round(v["start"], 3), hold=round(v["length"], 3),
                     local=v["local"], width=v["width"], height=v["height"], hdr=v["hdr"],
                     duration=round(v["duration"], 3))
        slots.append(s)
    return slots


def export_media(work: Path, items: list[dict], size, fps: int, workers: int):
    """Photos → work/photos/NNNN.jpg; videos → work/clips/NNNN.mp4 + poster; sets each item's "aspect"."""
    for d in ("photos", "clips"):
        (work / d).mkdir(exist_ok=True)
    for i, it in enumerate(items):
        if it["kind"] == "video":
            it["file"], it["poster"] = f"clips/{i:04d}.mp4", f"photos/{i:04d}.jpg"
        else:
            it["file"] = f"photos/{i:04d}.jpg"
    with process_pool(workers) as ex:
        list(ex.map(_export, [(it, work, tuple(size), fps) for it in items], chunksize=1))
    for it in items:
        if it["kind"] == "video":
            it["aspect"] = round(it["width"] / it["height"], 4)
        else:
            with Image.open(work / it["file"]) as im:
                it["aspect"] = round(im.width / im.height, 4)
    freed = V.prune_cache(work / "cache" / "videos", {it["local"] for it in items if it["kind"] == "video"})
    if freed:
        print(f"[vwl] video cache: freed {freed / 2**30:.1f} GB (kept ≤ 20 GB, least recently used first)")


def run(work: Path, images: int | float = 120, hold: float = 2.2, trans: float = 1.0, fps: int = 30,
        size=(1920, 1080), t_from=None, t_to=None, workers: int = 8, title: str = "",
        nsfw_filter: bool = True, hold_video: float = 4.0, video_share: float = 0.15) -> dict:
    """Pick `images` photos and videos in [t_from, t_to] (an int, or a float in (0, 1] = that fraction
    of the items in range; at most `video_share` of them videos).  A photo is shown for `hold` s, a
    video for up to `hold_video` s (shorter videos play in full); each is followed by a `trans` s morph."""
    photos, _social, places, loc = index.load(work)
    in_range = [p for p in photos if (t_from is None or p.ts >= t_from) and (t_to is None or p.ts <= t_to)]
    if not in_range:
        raise SystemExit("no photos in the requested date range (or nothing indexed — run `vwl index`)")
    # an open end of the range defaults to the first/last photo, so the map and strip don't run on
    # into years that only have location data
    t_from = in_range[0].ts if t_from is None else t_from
    t_to = in_range[-1].ts + 86400 if t_to is None else t_to
    m = np.ones(len(loc["t"]), bool)
    if t_from is not None:
        m &= loc["t"] >= t_from
    if t_to is not None:
        m &= loc["t"] <= t_to
    loc = {k: v[m] for k, v in loc.items()}
    if isinstance(images, float):
        pct = images
        images = max(2, round(pct * len(in_range)))
        print(f"[vwl] {pct * 100:.4g}% of {len(in_range)} photos and videos in range → {images}")
    intro, outro = 5.0, 6.0
    n_vid = sum(p.kind == "video" for p in in_range)
    print(f"[vwl] selecting {images} of {len(in_range) - n_vid} photos and {n_vid} videos "
          f"(videos: at most {video_share * 100:.3g}%)")
    ex_file = work / "exclude.txt"
    exclude = set()
    if ex_file.exists():
        exclude = {l.strip() for l in ex_file.read_text().splitlines() if l.strip() and not l.startswith("#")}
        print(f"[vwl] {len(exclude)} entries in {ex_file}")
    picked = select(in_range, images, workers=workers, exclude=exclude, nsfw_filter=nsfw_filter,
                    video_share=video_share, hold_video=hold_video, cache_dir=work / "cache" / "videos")
    vinfo = getattr(select, "video_info", {})
    if len(picked) < images:
        print(f"[vwl] only {len(picked)} suitable items in range")
    (work / "nsfw_flagged.txt").write_text("".join(u + "\n" for u in getattr(select, "flagged", [])))

    geo = Geocoder()
    slots = describe(picked, loc, places, geo, hold, trans, vinfo)

    # --- export locally: render workers shouldn't have to seek in 50 GB zips
    export_media(work, slots, size, fps, workers)

    stats = {
        "photos": len(in_range) - n_vid, "videos": n_vid, "location_points": int(m.sum()),
        "span": [slots[0]["year"], slots[-1]["year"]] if slots else None,
        "km": _track_km(loc),
        "countries": len({r["cc"] for r in geo.many(_country_sample(loc)) if r}) if len(loc["t"]) else 0,
    }
    out = {"fps": fps, "size": list(size), "intro": intro, "outro": outro, "title": title,
           "range": [t_from, t_to], "stats": stats, "slots": slots}
    with open(work / "plan.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    contact_sheet(work, slots)
    mosaic(work, slots, size, workers)
    total = intro + outro + sum(s["hold"] + s["trans"] for s in slots)
    vids = [s for s in slots if s["kind"] == "video"]
    parts = [f"{len(slots) - len(vids)} photos × {hold:g}s"]
    if vids:
        parts.append(f"{len(vids)} clips = {sum(s['hold'] for s in vids):.1f}s")
    parts.append(f"{len(slots) - 1} transitions × {trans:g}s + {intro + outro:g}s intro/outro")
    print(f"[vwl] plan: {' + '.join(parts)} → {total:.0f}s ({int(total // 60)}:{int(total % 60):02d})")
    return out


def _track_km(loc) -> int:
    """Great-circle km between daily median positions: a lower bound that ignores GPS jitter."""
    t, la, lo = loc["t"], loc["lat"], loc["lon"]
    if len(t) < 2:
        return 0
    from .index import _haversine
    day = (t // 86400).astype(np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1, len(t)]
    mla = np.array([np.median(la[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    mlo = np.array([np.median(lo[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    return int(_haversine(mla[:-1], mlo[:-1], mla[1:], mlo[1:]).sum() / 1000)


def _country_sample(loc, n=20000):
    idx = np.linspace(0, len(loc["t"]) - 1, min(n, len(loc["t"]))).astype(int)
    return [(float(loc["lat"][i]), float(loc["lon"][i])) for i in idx]


def contact_sheet(work: Path, slots: list[dict], cols: int = 8, tw: int = 240, per_page: int = 400):
    """work/contact_sheet.jpg (or contact_sheet_001.jpg, … when there are more than `per_page` items):
    every chosen photo, numbered, for a quick review.  To drop one, add its file title or its uri
    (in plan.json) to work/exclude.txt and re-plan.  A failure here only warns: the plan is saved."""
    for old in work.glob("contact_sheet*.jpg"):
        old.unlink()
    pages = [slots[k:k + per_page] for k in range(0, len(slots), per_page)] or [[]]
    names = ["contact_sheet.jpg"] if len(pages) == 1 else [f"contact_sheet_{p + 1:03d}.jpg" for p in range(len(pages))]
    try:
        for p, (page, name) in enumerate(zip(pages, names)):
            _sheet(work, page, p * per_page, cols, tw).save(work / name, quality=85)
    except Exception as e:  # noqa: BLE001
        print(f"[vwl] warning: contact sheet failed ({e}); the plan itself is saved")
        return
    shown = names[0] if len(names) == 1 else f"{names[0]} … {names[-1]}"
    print(f"[vwl] review {work / shown}; exclude photos via {work / 'exclude.txt'}")


def plan_items(plan: dict) -> list[dict]:
    """Every distinct photo/video of a plan, in film order (spreads flattened)."""
    out, seen = [], set()
    for s in plan["slots"]:
        for it in (s.get("items", []) + s.get("left", []) + s.get("right", []) if s.get("kind") == "spread" else [s]):
            if it["file"] not in seen:
                seen.add(it["file"])
                out.append(it)
    return out


def _thumb(args):
    path, w, h = args
    try:
        with Image.open(path) as im:
            im.draft("RGB", (w * 2, h * 2))       # fast JPEG decode at reduced size
            im = im.convert("RGB")
            s = max(w / im.width, h / im.height)
            im = im.resize((max(w, round(im.width * s)), max(h, round(im.height * s))), Image.LANCZOS)
            x0, y0 = (im.width - w) // 2, (im.height - h) // 2
            return np.asarray(im.crop((x0, y0, x0 + w, y0 + h)))
    except OSError:
        return np.full((h, w, 3), 30, np.uint8)


def mosaic(work: Path, items: list[dict], size, workers: int = 8) -> Path | None:
    """work/mosaic.jpg: every photo of the film as one screen-sized grid, in time order — the
    background of the opening title.  Cells shrink as the count grows (5,000 photos ≈ 20 px)."""
    W, H = size
    n = len(items)
    if not n:
        return None
    cols = max(1, math.ceil(math.sqrt(n * W / H)))
    rows = math.ceil(n / cols)
    xs = np.linspace(0, W, cols + 1).round().astype(int)
    ys = np.linspace(0, H, rows + 1).round().astype(int)
    last = n - cols * (rows - 1)                         # the last row is stretched to the full width
    xs_last = np.linspace(0, W, last + 1).round().astype(int)
    cells = []
    for k in range(n):
        r, c = divmod(k, cols)
        x = xs_last if r == rows - 1 else xs
        cells.append((x[c], ys[r], x[c + 1] - x[c], ys[r + 1] - ys[r]))
    jobs = [(str(work / it.get("poster", it["file"])), w, h)
            for it, (_, _, w, h) in zip(sorted(items, key=lambda x: x["ts"]), cells)]
    with process_pool(workers) as ex:
        thumbs = list(ex.map(_thumb, jobs, chunksize=32))
    out = np.full((H, W, 3), 12, np.uint8)
    for t, (x, y, w, h) in zip(thumbs, cells):
        out[y:y + h, x:x + w] = t
    path = work / "mosaic.jpg"
    Image.fromarray(out).save(path, quality=90)
    return path


def _sheet(work: Path, slots: list[dict], first: int, cols: int, tw: int):
    from PIL import Image, ImageDraw
    from .overlay import font
    th = tw * 3 // 4
    rows = max(1, (len(slots) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * tw, rows * (th + 22)), (16, 16, 18))
    d = ImageDraw.Draw(sheet)
    f = font(13, "Medium")
    for i, s in enumerate(slots):
        try:
            im = Image.open(work / s.get("poster", s["file"]))
        except OSError:
            im = Image.new("RGB", (tw, th), (40, 40, 44))
        im.thumbnail((tw - 6, th - 6))
        x, y = (i % cols) * tw, (i // cols) * (th + 22)
        sheet.paste(im, (x + (tw - im.width) // 2, y + (th - im.height) // 2))
        if s.get("kind") == "video":  # ▶ badge
            cx, cy = x + tw // 2, y + th // 2
            d.ellipse((cx - 18, cy - 18, cx + 18, cy + 18), fill=(0, 0, 0))
            d.polygon([(cx - 6, cy - 10), (cx - 6, cy + 10), (cx + 11, cy)], fill=(255, 255, 255))
        d.text((x + 4, y + th + 3), f"{first + i:04d}  {s['date']}", font=f, fill=(210, 210, 210))
    return sheet
