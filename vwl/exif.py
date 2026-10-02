"""Minimal EXIF reading (time + GPS) from the head of an image file."""
from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

from PIL import Image

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # HEIC just won't decode
    pass

HEAD_BYTES = 256 * 1024


def _ratio(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def _dms(v, ref) -> float | None:
    try:
        d, m, s = (_ratio(t) for t in v)
    except (TypeError, ValueError):
        return None
    x = d + m / 60 + s / 3600
    if x != x:
        return None
    return -x if ref in ("S", "W", b"S", b"W") else x


def read_exif(data: bytes) -> dict:
    """Return {'ts':…, 'lat':…, 'lon':…} for whatever could be found."""
    out = {}
    try:
        im = Image.open(io.BytesIO(data))
        ex = im.getexif()
    except Exception:
        return out
    sub = ex.get_ifd(0x8769) if ex else {}
    dt = sub.get(0x9003) or ex.get(0x0132)  # DateTimeOriginal, DateTime
    if dt:
        try:
            t = datetime.strptime(str(dt).strip("\x00 ")[:19], "%Y:%m:%d %H:%M:%S")
            off = sub.get(0x9011)  # OffsetTimeOriginal, e.g. "+02:00"
            tz = timezone.utc
            if off and len(str(off)) >= 6:
                o = str(off)
                sign = -1 if o[0] == "-" else 1
                tz = timezone(sign * timedelta(hours=int(o[1:3]), minutes=int(o[4:6])))
            out["ts"] = t.replace(tzinfo=tz).timestamp()
            out["ts_local_guess"] = off is None
        except (ValueError, IndexError):
            pass
    gps = ex.get_ifd(0x8825) if ex else {}
    if gps and 2 in gps and 4 in gps:
        lat, lon = _dms(gps[2], gps.get(1)), _dms(gps[4], gps.get(3))
        if lat is not None and lon is not None and not (lat == 0 and lon == 0):
            out["lat"], out["lon"] = lat, lon
    return out
