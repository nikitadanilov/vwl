"""`vwl index`: scan every export once, cache per container, merge into one timeline.

Outputs in <work>/index/:
  photos.jsonl     candidate photos (all sources), time-sorted
  social.jsonl     posts, comments, messages (all sources), time-sorted
  places.jsonl     named places with time spans (semantic visits, check-ins)
  locations.npz    t, lat, lon, src — the merged, cleaned movement track
  summary.json     counts, time range, detected account names
"""
from __future__ import annotations

import json
import os
import pickle
from concurrent.futures import as_completed
from pathlib import Path

import numpy as np

from . import ingest_google as G
from . import ingest_meta as M
from . import ingest_other as O
from .archive import Container, discover, process_pool, warn
from .exif import HEAD_BYTES, read_exif
from .model import (LOC_CHECKIN, LOC_PHOTO, LOC_POST, Photo, Place, Social, time_from_filename)

CACHE_VERSION = 6


def scan_container(c: Container | str) -> dict:
    c = Container(c) if isinstance(c, str) else c
    path = c.path
    members = c.members()
    out = {"path": path, "kind": "unknown", "photos": [], "social": [], "places": [], "raw_msgs": {},
           "gp_media": [], "gp_sidecars": [], "loc": None}
    loc_buf = G.LocBuf()
    if G.is_takeout(members):
        out["kind"] = "google"
        out["gp_media"], out["gp_sidecars"] = G.scan_photos(members)
        out["loc"], out["places"] = G.scan_locations(members)
    elif (platform := M.detect(path, members)):
        out["kind"] = platform
        photos, social, places, raw = M.scan(platform, members)
        out["photos"], out["social"], out["places"] = photos, social, places
        out["raw_msgs"] = {platform: raw}
    elif O.is_x(members):
        out["kind"] = "x"
        out["photos"], out["social"] = O.scan_x(members)
    else:
        # standalone Timeline.json exported from a phone (Android or iOS), GPX, photo folders
        arrays, places = G.scan_locations(members)  # includes GPX/TCX
        out["places"] = places
        out["photos"] = O.scan_generic_photos(members)
        extra = loc_buf.arrays()
        out["loc"] = {k: np.concatenate([arrays[k], extra[k]]) for k in arrays}
        out["kind"] = "generic"
    if out["loc"] is None:
        out["loc"] = loc_buf.arrays()
    return out


def _cached_scan(c: Container, fp: str, cache_dir: Path) -> dict:
    f = cache_dir / f"{fp}.v{CACHE_VERSION}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    res = scan_container(c)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(res, fh)
    os.replace(tmp, f)
    return res


def _clean_track(t, lat, lon, src):
    order = np.argsort(t, kind="stable")
    t, lat, lon, src = t[order], lat[order], lon[order], src[order]
    if len(t) < 3:
        return t, lat, lon, src
    # drop GPS teleports: points that imply >1200 km/h both into and out of them
    for _ in range(2):
        d_prev = _haversine(lat[:-1], lon[:-1], lat[1:], lon[1:])
        dt = np.maximum(np.diff(t), 1.0)
        v = d_prev / dt * 3.6  # km/h
        bad = np.zeros(len(t), bool)
        bad[1:-1] = (v[:-1] > 1200) & (v[1:] > 1200) & (d_prev[:-1] > 50_000)
        keep = ~bad
        t, lat, lon, src = t[keep], lat[keep], lon[keep], src[keep]
    # thin: keep a point if >=30 m from the last kept one or >=20 min later
    keep = np.zeros(len(t), bool)
    keep[0] = True
    lt, la, lo = t[0], lat[0], lon[0]
    cosl = np.cos(np.radians(lat))
    for i in range(1, len(t)):
        dy = (lat[i] - la) * 111_000
        dx = (lon[i] - lo) * 111_000 * cosl[i]
        if dx * dx + dy * dy > 900 or t[i] - lt > 1200 or src[i] in (LOC_PHOTO, LOC_CHECKIN, LOC_POST):
            keep[i] = True
            lt, la, lo = t[i], lat[i], lon[i]
    keep[-1] = True
    return t[keep], lat[keep], lon[keep], src[keep]


def _haversine(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _exif_fallback(m: dict) -> Photo | None:
    from .archive import read_uri
    ex = {}
    try:
        if m.get("kind", "image") == "image":
            ex = read_exif(read_uri(m["uri"], HEAD_BYTES))
    except Exception:  # noqa: BLE001
        pass
    ts, origin = ex.get("ts"), "exif"
    if ts is None:
        ts, origin = time_from_filename(m["name"]), "filename"
    if ts is None:
        return None
    return Photo(uri=m["uri"], ts=ts, lat=ex.get("lat"), lon=ex.get("lon"), title=m["name"],
                 album=m["dir"].rsplit("/", 1)[-1], ts_origin=origin, kind=m.get("kind", "image"))


SOCIAL = {"facebook", "instagram", "x"}


def fold_posted_copies(photos: list[Photo]) -> tuple[list[Photo], int]:
    """A photo you posted exists twice: the original in your library and the platform's
    downscaled copy.  When the copy kept the camera's capture time (ts_origin "exif"), it matches the
    original to the second: mark the original as shared (the strongest "this mattered" signal), give
    it the post's caption, and drop the copy.  Copies without a capture time stay as they are."""
    orig = [p for p in photos if p.source not in SOCIAL]
    ots = np.array([p.ts for p in orig])
    drop = set()
    for c in photos:
        if c.source not in SOCIAL or c.ts_origin != "exif" or not len(ots):
            continue
        i0, i1 = np.searchsorted(ots, [c.ts - 2, c.ts + 2])
        same = [o for o in orig[i0:i1] if o.kind == c.kind]
        if len(same) == 1:  # ambiguous bursts are left alone
            o = same[0]
            o.shared = True
            o.description = o.description or c.description
            if o.lat is None and c.lat is not None:
                o.lat, o.lon = c.lat, c.lon
            drop.add(id(c))
    return [p for p in photos if id(p) not in drop], len(drop)


def run(paths: list[str], work: Path, workers: int = 8, self_names: dict | None = None) -> dict:
    out_dir = work / "index"
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    containers = list(discover(paths))
    print(f"[vwl] {len(containers)} containers")
    results = []
    with process_pool(workers) as ex:
        futs = {ex.submit(_cached_scan, c, c.fingerprint(), cache_dir): c.path for c in containers}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001 - a corrupt part must not kill the run
                warn(f"failed to scan {futs[f]}: {e!r}")
                continue
            results.append(r)
            n_loc = len(r["loc"]["t"]) if r["loc"] is not None else 0
            print(f"[vwl] {i}/{len(futs)} {r['kind']:9s} {os.path.basename(r['path'])}: "
                  f"{len(r['gp_media']) + len(r['photos'])} images, {len(r['gp_sidecars'])} sidecars, "
                  f"{n_loc} loc pts, {len(r['social'])} social, "
                  f"{sum(len(v) for v in r['raw_msgs'].values())} msgs")

    # --- photos
    gp_media = [m for r in results for m in r["gp_media"]]
    # iPhone Live Photos export as IMG_1234.HEIC + IMG_1234.MOV (a 2-3 s clip): keep only the photo
    stills = {(m["dir"], m["name"].rsplit(".", 1)[0].lower()) for m in gp_media if m.get("kind", "image") == "image"}
    n0 = len(gp_media)
    gp_media = [m for m in gp_media if m.get("kind", "image") == "image"
                or (m["dir"], m["name"].rsplit(".", 1)[0].lower()) not in stills]
    if len(gp_media) < n0:
        print(f"[vwl] {n0 - len(gp_media)} Live Photo clips folded into their stills")
    gp_side = [s for r in results for s in r["gp_sidecars"]]
    paired, unpaired = G.pair_photos(gp_media, gp_side)
    photos: list[Photo] = []
    n_comp = 0
    for m, s in paired:
        if s.get("origin", "").startswith("composition"):
            n_comp += 1  # Google-made collages/animations/"cinematic" clips, not a moment you captured
            continue
        photos.append(Photo(uri=m["uri"], ts=s["ts"], lat=s["lat"], lon=s["lon"], title=s["title"],
                            description=s["description"], favorite=s["favorite"], views=s["views"],
                            album=m["dir"].rsplit("/", 1)[-1], kind=m.get("kind", "image"),
                            origin=s.get("origin", "")))
    if n_comp:
        print(f"[vwl] {n_comp} Google-made compositions skipped")
    if unpaired:
        print(f"[vwl] {len(unpaired)} Google Photos images without sidecar (yet) — using EXIF/filename")
        with process_pool(workers) as ex:
            photos += [p for p in ex.map(_exif_fallback, unpaired, chunksize=64) if p]
    for r in results:
        photos += r["photos"]
    # album folders duplicate "Photos from YYYY" entries: keep one per (name, time)
    seen, uniq = {}, []
    for p in sorted(photos, key=lambda p: (not p.album.startswith("Photos from"), p.uri)):
        key = (G.EDITED.sub("", p.title.rsplit(".", 1)[0]).lower(), round(p.ts), p.kind)
        if key in seen:
            q = seen[key]
            q.favorite |= p.favorite
            q.views = max(q.views, p.views)
            q.description = q.description or p.description
            continue
        seen[key] = p
        uniq.append(p)
    photos = sorted(uniq, key=lambda p: p.ts)
    photos, n_folded = fold_posted_copies(photos)
    if n_folded:
        print(f"[vwl] {n_folded} photos posted to Facebook/Instagram/X matched to their originals")

    # --- social
    social: list[Social] = [s for r in results for s in r["social"]]
    names = {}
    raw_by_platform = {}
    for r in results:
        for plat, msgs in r["raw_msgs"].items():
            raw_by_platform.setdefault(plat, []).extend(msgs)
    for plat, msgs in raw_by_platform.items():
        msgs_social, names[plat] = M.messages_to_social(plat, msgs, (self_names or {}).get(plat))
        social += msgs_social
    social.sort(key=lambda s: s.ts)
    places: list[Place] = sorted((p for r in results for p in r["places"]), key=lambda p: p.t0)

    # --- locations: history + every geotagged photo/post/check-in
    parts = [r["loc"] for r in results if r["loc"] is not None and len(r["loc"]["t"])]
    extra_t, extra_la, extra_lo, extra_s = [], [], [], []
    for p in photos:
        if p.lat is not None and p.ts_origin != "post":
            extra_t.append(p.ts); extra_la.append(p.lat); extra_lo.append(p.lon); extra_s.append(LOC_PHOTO)
    for s in social:
        if s.lat is not None:
            extra_t.append(s.ts); extra_la.append(s.lat); extra_lo.append(s.lon); extra_s.append(LOC_POST)
    for pl in places:
        if pl.source == "facebook":
            extra_t.append(pl.t0); extra_la.append(pl.lat); extra_lo.append(pl.lon); extra_s.append(LOC_CHECKIN)
    t = np.concatenate([p["t"] for p in parts] + [np.asarray(extra_t, np.float64)])
    lat = np.concatenate([p["lat"] for p in parts] + [np.asarray(extra_la, np.float64)])
    lon = np.concatenate([p["lon"] for p in parts] + [np.asarray(extra_lo, np.float64)])
    src = np.concatenate([p["src"] for p in parts] + [np.asarray(extra_s, np.uint8)])
    raw_n = len(t)
    t, lat, lon, src = _clean_track(t, lat, lon, src)

    # --- write
    with open(out_dir / "photos.jsonl", "w") as f:
        for p in photos:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "social.jsonl", "w") as f:
        for s in social:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    with open(out_dir / "places.jsonl", "w") as f:
        for p in places:
            f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
    np.savez_compressed(out_dir / "locations.npz", t=t, lat=lat, lon=lon, src=src)
    kinds = {}
    for p in social:
        kinds[f"{p.source}/{p.kind}"] = kinds.get(f"{p.source}/{p.kind}", 0) + 1
    srcs = {}
    for p in photos:
        srcs[p.source] = srcs.get(p.source, 0) + 1
    n_video = sum(p.kind == "video" for p in photos)
    summary = {
        "containers": {r["path"]: r["kind"] for r in results},
        "photos": len(photos) - n_video, "videos": n_video, "photos_by_source": srcs,
        "photos_geotagged": sum(p.lat is not None for p in photos),
        "photo_range": [photos[0].ts, photos[-1].ts] if photos else None,
        "social": len(social), "social_by_kind": kinds,
        "places": len(places),
        "location_points_raw": int(raw_n), "location_points": int(len(t)),
        "self_names": names,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def load(work: Path):
    d = work / "index"
    photos = [Photo(**json.loads(l)) for l in open(d / "photos.jsonl")]
    social = [Social(**json.loads(l)) for l in open(d / "social.jsonl")]
    places = [Place(**json.loads(l)) for l in open(d / "places.jsonl")]
    z = np.load(d / "locations.npz")
    return photos, social, places, {k: z[k] for k in z.files}
