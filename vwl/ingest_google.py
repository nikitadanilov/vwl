"""Google Takeout: Google Photos + every generation of Location History.

Google Photos
  Takeout/Google Photos/<Photos from YYYY | album>/<IMG>.jpg with a JSON sidecar next
  to it.  The sidecar has been called, over the years:
      IMG.jpg.json                         (until ~2024)
      IMG.jpg.supplemental-metadata.json   (late 2024+)
      IMG.jpg.supplemental-met.json        (names truncated to 51 chars)
      IMG.jpg(1).json  <->  IMG(1).jpg     (duplicate names)
      IMG-edited.jpg uses IMG.jpg's sidecar (suffix is localized)
  and it may sit in a *different zip part* than its image, so pairing happens only
  after all parts are indexed.  The authoritative original name is the sidecar's
  "title"; photoTakenTime is UTC; geoData is 0,0 when unknown.

Location History, in order of age
  Records.json               {"locations":[{latitudeE7, longitudeE7, timestamp|timestampMs}]}
  Semantic Location History/YYYY/YYYY_MONTH.json  {"timelineObjects":[{placeVisit}|{activitySegment}]}
  Timeline.json (Android on-device export, 2024+)  {"semanticSegments":[…], "rawSignals":[…]}
  Timeline.json (iOS on-device export; older iOS versions named it location-history.json)
                             [{startTime, visit|activity|timelinePath}] with "geo:lat,lon" points
Since 2024 Google keeps Timeline on the phone, so newer takeouts often contain no
location history at all — export it from the phone (see docs/DATA_SOURCES.md).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .archive import Member, warn
from .model import (LOC_FIT, media_kind, LOC_RECORDS, LOC_SEMANTIC, LOC_TIMELINE, Place, parse_latlng,
                    parse_time, valid_latlon)

PHOTO_PRODUCT = re.compile(r"(?i)google\s*(photos|fotos|фото|foto)|^photos$")
LOC_FILE = re.compile(r"(?i)(^|/)(records|timeline|location[-_ ]history)[^/]*\.json$")
SEMANTIC_FILE = re.compile(r"(^|/)\d{4}_[A-Z]+\.json$")
DUP = re.compile(r"^(.*)\((\d+)\)$")
EDITED = re.compile(r"(?i)-(edited|bearbeitet|modifié|editado|modificato|bewerkt|redigerad|"
                    r"muokattu|edytowane|upravené|измененный|изменено|отредактировано)$")


def is_takeout(members: list[Member]) -> bool:
    """A Takeout zip part, or an extracted Takeout/ folder (given directly or via its parent)."""
    if members and members[0].container.rstrip("/").endswith("/Takeout"):
        return True
    return any(m.name.startswith("Takeout/") for m in members[:2000])


def _product(m: Member) -> str:
    parts = m.name.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "Takeout" else parts[0]


# ---------------------------------------------------------------- Google Photos

def scan_photos(members: list[Member]):
    """Return (media, sidecars) for one container. Pairing happens in pair_photos()."""
    media, sidecars = [], []
    for m in members:
        if not PHOTO_PRODUCT.search(_product(m)):
            continue
        low = m.basename.lower()
        ext = low[low.rfind("."):] if "." in low else ""
        kind = media_kind(low)
        if kind:
            media.append({"uri": m.uri, "dir": m.dirname, "name": m.basename, "kind": kind})
        elif ext == ".json" and m.size_hint < 64 * 1024:  # -1 (unknown) passes
            try:
                j = json.loads(m.read())
            except (ValueError, UnicodeDecodeError):
                continue
            if not isinstance(j, dict) or "photoTakenTime" not in j:
                continue  # album metadata.json, print-subscriptions.json, …
            geo = j.get("geoData") or {}
            lat, lon = geo.get("latitude"), geo.get("longitude")
            if not valid_latlon(lat, lon):
                geo = j.get("geoDataExif") or {}
                lat, lon = geo.get("latitude"), geo.get("longitude")
            dm = DUP.match(m.basename[:-5])
            sidecars.append({
                "dir": m.dirname,
                "title": j.get("title", ""),
                "dup": int(dm.group(2)) if dm else 0,
                "ts": parse_time((j.get("photoTakenTime") or {}).get("timestamp")),
                "lat": lat if valid_latlon(lat, lon) else None,
                "lon": lon if valid_latlon(lat, lon) else None,
                "description": j.get("description") or "",
                "favorite": bool(j.get("favorited")),
                "views": int(j.get("imageViews") or 0),
                "origin": _origin(j.get("googlePhotosOrigin")),
            })
    return media, sidecars


def _origin(o) -> str:
    """{"mobileUpload": {"deviceType": "ANDROID_PHONE"}} → "mobileUpload:ANDROID_PHONE"; {"picasa": {}} →
    "picasa".  Seen: mobileUpload, picasa, driveDesktopUploader, photosDesktopUploader, driveSync, webUpload,
    composition (Google-made collages/animations); received items should mention partner/shared."""
    if not isinstance(o, dict) or not o:
        return ""
    key = next(iter(o))
    dev = (o.get(key) or {}).get("deviceType", "") if isinstance(o.get(key), dict) else ""
    return f"{key}:{dev}" if dev else key


def pair_photos(media: list[dict], sidecars: list[dict]):
    """Match images to sidecars across all containers. Returns (paired, unpaired_media)."""
    by_key, by_dir = {}, {}
    for s in sidecars:
        by_key[(s["dir"], s["title"], s["dup"])] = s
        by_dir.setdefault(s["dir"], []).append(s)
    paired, unpaired = [], []
    for m in media:
        d, name = m["dir"], m["name"]
        stem, ext = (name.rsplit(".", 1) + [""])[:2]
        dup = 0
        dm = DUP.match(stem)
        if dm:
            stem, dup = dm.group(1), int(dm.group(2))
        base = EDITED.sub("", stem)
        s = (by_key.get((d, f"{base}.{ext}", dup)) or by_key.get((d, f"{base}.{ext}", 0)))
        if s is None:
            # truncated file name, or extension case/format changed (HEIC→JPG on download)
            cands = [c for c in by_dir.get(d, ()) if c["title"].startswith(base) and c["dup"] == dup]
            if len(cands) == 1 or (cands and all(c["title"] == cands[0]["title"] for c in cands)):
                s = cands[0]
        if s is None or s["ts"] is None:
            unpaired.append(m)
        else:
            paired.append((m, s))
    return paired, unpaired


# ---------------------------------------------------------------- Location History

class LocBuf:
    def __init__(self):
        self.t, self.la, self.lo, self.src, self.acc = [], [], [], [], []

    def add(self, t, lat, lon, src, acc=0):
        if t is not None and valid_latlon(lat, lon):
            self.t.append(t); self.la.append(lat); self.lo.append(lon)
            self.src.append(src); self.acc.append(acc or 0)

    def arrays(self):
        return {
            "t": np.asarray(self.t, np.float64), "lat": np.asarray(self.la, np.float64),
            "lon": np.asarray(self.lo, np.float64), "src": np.asarray(self.src, np.uint8),
            "acc": np.asarray(self.acc, np.float32),
        }


def _records(m: Member, buf: LocBuf):
    try:
        import ijson
        with m.open() as f:
            for r in ijson.items(f, "locations.item", use_float=True):
                t = parse_time(r.get("timestamp") or r.get("timestampMs"))
                ll = parse_latlng(r)
                if ll:
                    buf.add(t, ll[0], ll[1], LOC_RECORDS, r.get("accuracy"))
    except ImportError:
        for r in json.loads(m.read()).get("locations", []):
            ll = parse_latlng(r)
            if ll:
                buf.add(parse_time(r.get("timestamp") or r.get("timestampMs")), *ll, LOC_RECORDS, r.get("accuracy"))


def _dur(d: dict):
    d = d or {}
    return (parse_time(d.get("startTimestamp") or d.get("startTimestampMs")),
            parse_time(d.get("endTimestamp") or d.get("endTimestampMs")))


def _semantic(j: dict, buf: LocBuf, places: list[Place]):
    for o in j.get("timelineObjects", []):
        if "placeVisit" in o:
            v = o["placeVisit"]
            loc = v.get("location") or {}
            ll = parse_latlng(loc)
            t0, t1 = _dur(v.get("duration"))
            if ll and t0:
                buf.add(t0, *ll, LOC_SEMANTIC)
                buf.add(t1, *ll, LOC_SEMANTIC)
                name = loc.get("name") or (loc.get("address") or "").split("\n")[0]
                if name:
                    places.append(Place(t0, t1 or t0, ll[0], ll[1], name, "google_semantic"))
        elif "activitySegment" in o:
            a = o["activitySegment"]
            t0, t1 = _dur(a.get("duration"))
            if not t0:
                continue
            raw = (a.get("simplifiedRawPath") or {}).get("points") or []
            if raw:
                for p in raw:
                    ll = parse_latlng(p)
                    if ll:
                        buf.add(parse_time(p.get("timestamp") or p.get("timestampMs")), *ll, LOC_SEMANTIC)
            pts = [parse_latlng(a.get("startLocation"))]
            pts += [parse_latlng(w) for w in (a.get("waypointPath") or {}).get("waypoints", [])]
            pts += [parse_latlng(a.get("endLocation"))]
            pts = [p for p in pts if p]
            t1 = t1 or t0
            for i, p in enumerate(pts):  # waypoints carry no times: spread evenly
                buf.add(t0 + (t1 - t0) * i / max(1, len(pts) - 1), *p, LOC_SEMANTIC)


def _segments(segs: list, buf: LocBuf, places: list[Place]):
    """Android Timeline.json's semanticSegments and the iOS export's top-level list share this shape."""
    for s in segs:
        t0, t1 = parse_time(s.get("startTime")), parse_time(s.get("endTime"))
        if t0 is None:
            continue
        t1 = t1 or t0
        if "visit" in s:
            c = (s["visit"] or {}).get("topCandidate") or {}
            ll = parse_latlng(c.get("placeLocation"))
            if ll:
                buf.add(t0, *ll, LOC_TIMELINE)
                buf.add(t1, *ll, LOC_TIMELINE)
                sem = c.get("semanticType") or ""
                if sem and sem.upper() not in ("UNKNOWN", "SEARCHED_ADDRESS", "ALIASED_LOCATION"):
                    places.append(Place(t0, t1, ll[0], ll[1], sem.replace("_", " ").title(), "google_timeline"))
        if "activity" in s:
            a = s["activity"] or {}
            for key, t in (("start", t0), ("end", t1)):
                ll = parse_latlng(a.get(key))
                if ll:
                    buf.add(t, *ll, LOC_TIMELINE)
        for p in s.get("timelinePath") or []:
            ll = parse_latlng(p.get("point"))
            t = parse_time(p.get("time"))
            if t is None and "durationMinutesOffsetFromStartTime" in p:
                t = t0 + 60 * float(p["durationMinutesOffsetFromStartTime"])
            if ll and t:
                buf.add(t, *ll, LOC_TIMELINE)


FIT_LOC = re.compile(r"(^|/)Fit/All data/[^/]*location\.sample[^/]*\.json$")


def _fit_samples(m: Member, buf: LocBuf):
    """Google Fit: {"Data Points": [{"startTimeNanos", "fitValue": [lat, lon, accuracy, altitude]}]}."""
    try:
        j = json.loads(m.read())
    except ValueError:
        return
    for p in j.get("Data Points", []):
        v = p.get("fitValue") or []
        if len(v) >= 2:
            lat, lon = (v[0].get("value") or {}).get("fpVal"), (v[1].get("value") or {}).get("fpVal")
            acc = (v[2].get("value") or {}).get("fpVal") if len(v) > 2 else 0
            if acc is not None and acc > 500:
                continue
            buf.add(int(p["startTimeNanos"]) / 1e9, lat, lon, LOC_FIT, acc)


def scan_locations(members: list[Member]):
    from .ingest_other import scan_gpx
    buf, places = LocBuf(), []
    scan_gpx(members, buf)
    for m in members:
        n = m.name
        if FIT_LOC.search(n):
            _fit_samples(m, buf)
            continue
        if n.endswith(".json") and LOC_FILE.search(n):
            head = m.read(4096).lstrip()
            try:
                if b'"locations"' in head[:200]:
                    _records(m, buf)
                    continue
                j = json.loads(m.read())
            except Exception as e:  # noqa: BLE001 - one bad file must not stop indexing
                warn(f"cannot parse {m.uri}: {e}")
                continue
            if isinstance(j, dict) and "semanticSegments" in j:
                _segments(j["semanticSegments"], buf, places)
                for r in j.get("rawSignals", []):
                    pos = r.get("position")
                    if pos:
                        ll = parse_latlng(pos)
                        if ll:
                            buf.add(parse_time(pos.get("timestamp")), *ll, LOC_TIMELINE, pos.get("accuracyMeters"))
            elif isinstance(j, dict) and "timelineObjects" in j:
                _semantic(j, buf, places)
            elif isinstance(j, list) and j and isinstance(j[0], dict) and "startTime" in j[0]:
                _segments(j, buf, places)
        elif n.endswith(".json") and SEMANTIC_FILE.search(n) and m.size_hint < 200 * 2**20:
            try:
                j = json.loads(m.read())
            except ValueError:
                continue
            if isinstance(j, dict) and "timelineObjects" in j:
                _semantic(j, buf, places)
    return buf.arrays(), places
