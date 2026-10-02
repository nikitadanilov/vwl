"""X/Twitter archives, GPX tracks, and plain photo folders (iCloud export, phone backup…)."""
from __future__ import annotations

import json
import re

from .archive import Member, warn
from .exif import HEAD_BYTES, read_exif
from .model import LOC_GPX, Photo, Social, media_kind, parse_time, time_from_filename, valid_latlon

# ---------------------------------------------------------------- X / Twitter
# data/tweets.js (older: data/tweet.js, plus tweets-part1.js …) is JavaScript:
#   window.YTD.tweets.part0 = [ {"tweet": {...}} , ... ]


def is_x(members: list[Member]) -> bool:
    return any(re.search(r"(^|/)data/(manifest|account)\.js$", m.name) for m in members)


def _js(m: Member):
    raw = m.read().decode("utf-8", "replace")
    return json.loads(raw[raw.index("=") + 1:])


def scan_x(members: list[Member]):
    social, photos = [], []
    me = None
    media_by_id = {}
    for m in members:
        if re.search(r"(^|/)data/account\.js$", m.name):
            try:
                me = _js(m)[0]["account"]["accountId"]
            except Exception:  # noqa: BLE001
                pass
        mm = re.search(r"data/tweets?_media/(\d+)-[^/]+$", m.name)
        if mm:
            media_by_id.setdefault(mm.group(1), []).append(m)
    for m in members:
        if re.search(r"(^|/)data/tweets?(-part\d+)?\.js$", m.name):
            try:
                items = _js(m)
            except Exception as e:  # noqa: BLE001
                warn(f"cannot parse {m.uri}: {e}")
                continue
            for it in items:
                t = it.get("tweet", it)
                text = t.get("full_text") or t.get("text") or ""
                if text.startswith("RT @"):
                    continue
                ts = parse_time(t.get("created_at"))
                text = re.sub(r"\s*https://t\.co/\w+", "", text).strip()
                lat = lon = None
                coords = (t.get("coordinates") or {}).get("coordinates")
                if coords and valid_latlon(float(coords[1]), float(coords[0])):
                    lat, lon = float(coords[1]), float(coords[0])
                kind = "comment" if t.get("in_reply_to_screen_name") else "post"
                if ts and text:
                    social.append(Social(ts, "x", kind, text, who=t.get("in_reply_to_screen_name") or "",
                                         place=(t.get("place") or {}).get("full_name", ""), lat=lat, lon=lon))
                for med in media_by_id.get(t.get("id_str", ""), []):
                    kind = media_kind(med.basename)
                    if kind and ts:
                        photos.append(Photo(uri=med.uri, ts=ts, lat=lat, lon=lon, source="x", title=med.basename,
                                            description=text, shared=True, ts_origin="post", kind=kind))
        elif re.search(r"(^|/)data/direct-messages(-part\d+)?\.js$", m.name) and me:
            try:
                convs = _js(m)
            except Exception:  # noqa: BLE001
                continue
            for c in convs:
                for msg in (c.get("dmConversation") or {}).get("messages", []):
                    mc = msg.get("messageCreate")
                    if not mc or not mc.get("text"):
                        continue
                    mine = mc.get("senderId") == me
                    social.append(Social(parse_time(mc.get("createdAt")), "x", "message_out" if mine else "message_in",
                                         re.sub(r"\s*https://t\.co/\w+", "", mc["text"]),
                                         who=mc.get("recipientId") if mine else mc.get("senderId")))
    return photos, [s for s in social if s.ts]


# ---------------------------------------------------------------- GPX (Strava, Garmin, …)
_TRKPT = re.compile(rb'<(?:trkpt|rtept|wpt)\s+[^>]*?lat="([-\d.]+)"\s+lon="([-\d.]+)"[^>]*>(.*?)</(?:trkpt|rtept|wpt)>', re.S)
_TRKPT2 = re.compile(rb'<(?:trkpt|rtept|wpt)\s+[^>]*?lon="([-\d.]+)"\s+lat="([-\d.]+)"[^>]*>(.*?)</(?:trkpt|rtept|wpt)>', re.S)
_TIME = re.compile(rb"<time>([^<]+)</time>")


_TCX_TRKPT = re.compile(rb"<Trackpoint>(.*?)</Trackpoint>", re.S)
_TCX_LAT = re.compile(rb"<LatitudeDegrees>([-\d.]+)</LatitudeDegrees>")
_TCX_LON = re.compile(rb"<LongitudeDegrees>([-\d.]+)</LongitudeDegrees>")
_TCX_TIME = re.compile(rb"<Time>([^<]+)</Time>")


def tcx_points(data: bytes):
    """(time, lat, lon) per <Trackpoint> that has a position.  Each trackpoint is parsed on its own:
    some have a time but no position (GPS not yet locked), and must not borrow the next one's."""
    for body in _TCX_TRKPT.findall(data):
        tm, la, lo = _TCX_TIME.search(body), _TCX_LAT.search(body), _TCX_LON.search(body)
        if tm and la and lo:
            yield tm.group(1).decode(), float(la.group(1)), float(lo.group(1))


def scan_gpx(members: list[Member], buf):
    """GPX (Strava, Garmin, …) and TCX (Google Fit Activities/*.tcx, Garmin)."""
    for m in members:
        low = m.name.lower()
        if low.endswith(".tcx"):
            for t, la, lo in tcx_points(m.read()):
                buf.add(parse_time(t), la, lo, LOC_GPX)
            continue
        if not low.endswith(".gpx"):
            continue
        data = m.read()
        for rx, swap in ((_TRKPT, False), (_TRKPT2, True)):
            for a, b, body in rx.findall(data):
                lat, lon = (float(b), float(a)) if swap else (float(a), float(b))
                tm = _TIME.search(body)
                if tm:
                    buf.add(parse_time(tm.group(1).decode()), lat, lon, LOC_GPX)


# ---------------------------------------------------------------- plain photo folders

def scan_generic_photos(members: list[Member], source="photos"):
    """Any images with EXIF (Apple 'iCloud Photos Part N of M.zip', DCIM backups, …)."""
    photos = []
    for m in members:
        low = m.basename.lower()
        kind = media_kind(low)
        if kind == "video":  # no EXIF to read cheaply; the file name must carry the time
            ts = time_from_filename(m.basename)
            if ts is not None:
                photos.append(Photo(uri=m.uri, ts=ts, source=source, title=m.basename, kind="video",
                                    album=m.dirname.rsplit("/", 1)[-1], ts_origin="filename"))
            continue
        if kind != "image" or 0 <= m.size_hint < 50_000:
            continue
        ex = read_exif(m.read(HEAD_BYTES))
        ts, origin = ex.get("ts"), "exif"
        if ts is None:
            ts, origin = time_from_filename(m.basename), "filename"
        if ts is None:
            continue
        photos.append(Photo(uri=m.uri, ts=ts, lat=ex.get("lat"), lon=ex.get("lon"), source=source,
                            title=m.basename, album=m.dirname.rsplit("/", 1)[-1], ts_origin=origin))
    return photos
