"""Text and UI layers, pre-rendered as RGBA once and alpha-blended per frame."""
from __future__ import annotations

import functools
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PLATFORM = {
    "facebook": ("Facebook", (66, 133, 244)),
    "instagram": ("Instagram", (225, 48, 108)),
    "x": ("X", (200, 200, 200)),
}
EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍\U000E0000-\U000E007F]")


@functools.lru_cache(None)
def font_path(spec: str) -> str | None:
    try:
        return subprocess.run(["fc-match", "-f", "%{file}", spec], capture_output=True, text=True, timeout=5).stdout or None
    except (OSError, subprocess.TimeoutExpired):
        return None


@functools.lru_cache(None)
def font(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    for spec in (f"Inter:style={weight}", f"Noto Sans:style={weight}", "DejaVu Sans"):
        p = font_path(spec)
        if p:
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def clean(s: str) -> str:
    return EMOJI.sub("", s).strip()


def to_rgba(im: Image.Image) -> np.ndarray:
    return np.asarray(im, dtype=np.uint8)


def blend(dst: np.ndarray, rgba: np.ndarray, x: int, y: int, alpha: float = 1.0):
    """In-place alpha blend of an RGBA layer onto an RGB frame at (x, y)."""
    if alpha <= 0.003:
        return
    h, w = rgba.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    src = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = src[..., 3:4].astype(np.float32) * (alpha / 255.0)
    region = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (region * (1 - a) + src[..., :3].astype(np.float32) * a).astype(np.uint8)


def shadowed_text(lines, sizes, weights, colors, gap=6, pad=24) -> np.ndarray:
    """Stack of text lines with a soft drop shadow, as RGBA."""
    fonts = [font(s, w) for s, w in zip(sizes, weights)]
    lines = [clean(l) for l in lines]
    boxes = [f.getbbox(l or " ") for f, l in zip(fonts, lines)]
    W = max(b[2] for b in boxes) + 2 * pad
    H = sum(f.size + gap for f in fonts) + 2 * pad
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sh = Image.new("L", (W, H), 0)
    d, ds = ImageDraw.Draw(im), ImageDraw.Draw(sh)
    y = pad
    for f, l, c in zip(fonts, lines, colors):
        ds.text((pad + 2, y + 3), l, font=f, fill=200)
        d.text((pad, y), l, font=f, fill=c + (255,))
        y += f.size + gap
    sh = sh.filter(ImageFilter.GaussianBlur(7))
    base = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    base.putalpha(sh)
    return to_rgba(Image.alpha_composite(base, im))


def _ticks(d, t0: float, t1: float, x_of, height: int, scale: float, max_labels: int = 12):
    """Tick marks suited to the span: years for a long film, months for a year or two, days for an
    excerpt of a few weeks.  A year boundary is always labelled with the year."""
    days = (t1 - t0) / 86400
    a, b = datetime.fromtimestamp(t0, timezone.utc), datetime.fromtimestamp(t1, timezone.utc)
    if days > 3 * 365:
        marks = [datetime(y, 1, 1, tzinfo=timezone.utc) for y in range(a.year + 1, b.year + 1)]
        label = lambda m: str(m.year)  # noqa: E731
    elif days > 75:
        marks = []
        y, mo = a.year, a.month
        while True:
            mo += 1
            if mo > 12:
                y, mo = y + 1, 1
            m = datetime(y, mo, 1, tzinfo=timezone.utc)
            if m > b:
                break
            marks.append(m)
        label = lambda m: str(m.year) if m.month == 1 else m.strftime("%b")  # noqa: E731
    else:
        start = datetime(a.year, a.month, a.day, tzinfo=timezone.utc)
        marks = [datetime.fromtimestamp(start.timestamp() + k * 86400, timezone.utc)
                 for k in range(1, int(days) + 2)]
        marks = [m for m in marks if m.timestamp() <= t1]
        label = lambda m: str(m.year) if (m.month, m.day) == (1, 1) else f"{m.day} {m.strftime('%b')}"  # noqa: E731
    f = font(int(15 * scale), "Medium")
    step = max(1, -(-len(marks) // max_labels))
    for k, m in enumerate(marks):
        x = x_of(m.timestamp())
        major = (m.month, m.day) == (1, 1)
        d.line((x, 0, x, height), fill=(255, 255, 255, 90 if major else 40))
        if k % step == 0 or major:
            d.text((x + 4, 2), label(m), font=f, fill=(200, 200, 200, 200))


def plural(n: int, word: str) -> str:
    """'1 day', '7 days', '1,821 days' (regular English plurals only)."""
    return f"{n:,} {word}" if n == 1 else f"{n:,} {word}s"


def span_text(span) -> str:
    """'2017 — 2025', or just '2018' when it starts and ends in the same year."""
    if not span:
        return ""
    return str(span[0]) if span[0] == span[1] else f"{span[0]} — {span[1]}"


def timeline_strip(slots, loc_ts, t0, t1, width, height, scale) -> tuple[np.ndarray, callable]:
    """Static histogram (shown photos above, location-data density below) + a mapper ts→x."""
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, width, height), fill=(10, 12, 16, 150))
    span = max(t1 - t0, 1)

    def x_of(ts):
        return int((ts - t0) / span * (width - 1))
    bins = max(40, width // 6)
    mid = height // 2 + int(4 * scale)
    for arr, col, up in ((np.array([s["ts"] for s in slots]), (255, 255, 255, 150), True),
                         (np.asarray(loc_ts), (255, 146, 64, 170), False)):
        if len(arr) == 0:
            continue
        hist, _ = np.histogram(arr, bins=bins, range=(t0, t1))
        hist = np.sqrt(hist / max(hist.max(), 1))
        bw = width / bins
        for i, v in enumerate(hist):
            if v <= 0:
                continue
            hpx = max(1, int(v * (height / 2 - 6 * scale)))
            x0 = int(i * bw) + 1
            if up:
                d.rectangle((x0, mid - hpx, int((i + 1) * bw) - 1, mid - 1), fill=col)
            else:
                d.rectangle((x0, mid + 1, int((i + 1) * bw) - 1, mid + max(1, hpx // 2)), fill=col)
    _ticks(d, t0, t1, x_of, height, scale)
    return to_rgba(im), x_of


def timeline_strip_duo(cp_t, cp_state, cp_dist, t0, t1, width, height, scale) -> tuple[np.ndarray, callable]:
    """Distance between the two people over time (log scale, taller = further apart); together is a
    low gold bar, no data is grey.  Returns the strip and a mapper ts→x."""
    im = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, width, height), fill=(10, 12, 16, 150))
    span = max(t1 - t0, 1)

    def x_of(ts):
        return int((ts - t0) / span * (width - 1))
    inside = (cp_t >= t0) & (cp_t <= t1)       # an excerpt's strip covers only its own dates
    cp_t, cp_state, cp_dist = cp_t[inside], cp_state[inside], cp_dist[inside]
    col = np.clip(((cp_t - t0) / span * width).astype(int), 0, width - 1)
    known = cp_state > 0
    n_all = np.bincount(col, minlength=width)
    n_known = np.bincount(col, weights=known, minlength=width)
    n_tog = np.bincount(col, weights=cp_state == 2, minlength=width)
    top = int(18 * scale)
    base = height - int(4 * scale)
    for x in range(width):
        if n_all[x] == 0:
            continue
        if n_known[x] < 0.2 * n_all[x]:
            d.line((x, base - int(3 * scale), x, base), fill=(90, 90, 96, 140))
            continue
        m = (col == x) & known
        km = float(np.median(cp_dist[m])) / 1000 if m.any() else 0.0
        frac_t = n_tog[x] / max(n_known[x], 1)
        h = (np.clip(np.log10(km + 0.1) + 1, 0, 5.3) / 5.3) * (base - top)
        h = max(h * (1 - frac_t), 2 * scale)
        c = tuple(int(a * frac_t + b * (1 - frac_t)) for a, b in zip((255, 214, 110), (205, 205, 215)))
        d.line((x, base - int(h), x, base), fill=c + (215,))
    _ticks(d, t0, t1, x_of, height, scale)
    return to_rgba(im), x_of
