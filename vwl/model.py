"""Plain records shared by all ingesters, plus small parsing helpers."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".gif", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".avi", ".mkv", ".mts", ".webm"}


def media_kind(name: str) -> str | None:
    """'image', 'video', or None for anything else, judged by extension."""
    ext = name[name.rfind("."):].lower() if "." in name else ""
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None

# Location sample origin codes (stored as uint8 in locations.npz)
LOC_RECORDS, LOC_SEMANTIC, LOC_TIMELINE, LOC_PHOTO, LOC_CHECKIN, LOC_GPX, LOC_POST, LOC_FIT = range(8)


@dataclass
class Photo:
    uri: str                       # container::member of the image bytes
    ts: float                      # capture time, unix seconds UTC
    lat: float | None = None
    lon: float | None = None
    source: str = "google_photos"  # google_photos | instagram | facebook | x | photos
    title: str = ""
    description: str = ""
    favorite: bool = False
    views: int = 0
    album: str = ""
    shared: bool = False           # was posted to a social network (strong "this mattered" signal)
    ts_origin: str = "sidecar"     # sidecar | exif | filename | post
    kind: str = "image"            # image | video
    origin: str = ""               # Google Photos upload origin, e.g. "mobileUpload:ANDROID_PHONE", "picasa"

    def to_dict(self):
        return asdict(self)


@dataclass
class Social:
    ts: float
    source: str                    # facebook | instagram | x
    kind: str                      # post | comment | message_out | message_in | story | checkin
    text: str
    who: str = ""                  # counterpart (thread title / replied-to) — anonymized at render time
    place: str = ""
    lat: float | None = None
    lon: float | None = None
    media: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Place:
    t0: float
    t1: float
    lat: float
    lon: float
    name: str
    source: str

    def to_dict(self):
        return asdict(self)


def parse_time(v) -> float | None:
    """Accept unix s/ms (int/str) or ISO-8601 ('Z' or offset). Returns unix seconds."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x / 1000.0 if x > 1e11 else x
    s = str(v).strip()
    if s.isdigit():
        return parse_time(int(s))
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%a %b %d %H:%M:%S %z %Y",      # X/Twitter created_at
                    "%Y:%m:%d %H:%M:%S",             # EXIF
                    "%d %b %Y, %H:%M:%S %Z"):        # Takeout "formatted"
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_FLOAT = r"[-+]?\d+(?:\.\d+)?"
_LATLNG = re.compile(rf"({_FLOAT})\s*°?\s*,\s*({_FLOAT})")


def parse_latlng(v) -> tuple[float, float] | None:
    """'52.52°, 13.40°' (Android Timeline), 'geo:52.52,13.40' (iOS), or {latitudeE7,..}."""
    if v is None:
        return None
    if isinstance(v, dict):
        if "latitudeE7" in v and "longitudeE7" in v:
            return v["latitudeE7"] / 1e7, v["longitudeE7"] / 1e7
        if "latE7" in v and "lngE7" in v:
            return v["latE7"] / 1e7, v["lngE7"] / 1e7
        if "latitude" in v and "longitude" in v:
            return float(v["latitude"]), float(v["longitude"])
        for k in ("latLng", "LatLng", "point", "placeLocation"):
            if k in v:
                return parse_latlng(v[k])
        return None
    m = _LATLNG.search(str(v))
    if not m:
        return None
    lat, lon = float(m.group(1)), float(m.group(2))
    return (lat, lon) if valid_latlon(lat, lon) else None


def valid_latlon(lat, lon) -> bool:
    return (lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
            and not (abs(lat) < 1e-6 and abs(lon) < 1e-6))


def fix_mojibake(s: str) -> str:
    """Meta exports write UTF-8 bytes as \\u00XX escapes; undo that."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def fix_tree(o):
    if isinstance(o, str):
        return fix_mojibake(o)
    if isinstance(o, list):
        return [fix_tree(x) for x in o]
    if isinstance(o, dict):
        return {fix_mojibake(k): fix_tree(v) for k, v in o.items()}
    return o


_FN_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})[-_.]?(\d{2})[-_.]?(\d{2})[-_ T.]?(\d{2})[-_.:]?(\d{2})[-_.:]?(\d{2})(?!\d{3,})")


def time_from_filename(name: str) -> float | None:
    """IMG_20230513_165021.jpg, PXL_20230513_165021123.jpg, 2023-05-13 16.50.21.jpg, ..."""
    m = _FN_DATE.search(name)
    if not m:
        return None
    y, mo, d, h, mi, se = map(int, m.groups())
    try:
        return datetime(y, mo, d, h, mi, se, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None
