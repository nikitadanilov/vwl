"""Video clips: extract from the export, probe, pick the best window, transcode for rendering.

Takeout deflates videos inside its zips, so ffmpeg cannot seek in them in place: a candidate
is first streamed to a local cache file.  Only candidates for chosen slots are touched, never
the whole library (1,600+ videos, ~85 GB in a typical Takeout).
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .archive import open_uri

MIN_CLIP = 2.0            # shorter videos are skipped (accidental taps, Live-Photo-like clips)
MAX_BYTES = 1 << 30       # larger videos are only considered when favourited
SAMPLE_W = 192


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def member_size(uri: str) -> int:
    container, name = uri.split("::", 1)
    if os.path.isdir(container):
        return os.path.getsize(os.path.join(container, name))
    with zipfile.ZipFile(container) as zf:
        return zf.getinfo(name).file_size


def local_path(uri: str, cache_dir: Path) -> Path:
    """A seekable local file for the video at `uri` (extracted from its zip once, then cached)."""
    container, name = uri.split("::", 1)
    if os.path.isdir(container):
        return Path(container) / name
    ext = name[name.rfind("."):].lower()
    dst = cache_dir / (hashlib.sha1(uri.encode()).hexdigest()[:20] + ext)
    if dst.exists():
        os.utime(dst)  # mark as recently used for prune_cache()
        return dst
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".{os.getpid()}.part")
    with open_uri(uri) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, 4 << 20)
    os.replace(tmp, dst)
    return dst


def prune_cache(cache_dir: Path, keep: set[str], max_bytes: int = 20 << 30) -> int:
    """Delete least recently used cached videos until the cache fits `max_bytes`; never those in `keep`."""
    if not cache_dir.exists():
        return 0
    files = sorted(cache_dir.iterdir(), key=lambda f: f.stat().st_mtime)
    total = sum(f.stat().st_size for f in files)
    freed = 0
    for f in files:
        if total <= max_bytes:
            break
        if str(f) in keep or str(f.resolve()) in keep:
            continue
        sz = f.stat().st_size
        f.unlink()
        total -= sz
        freed += sz
    return freed


_DUR = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_VID = re.compile(r"Stream #\S+.*?: Video: (.*)")
_WH = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_ROT = re.compile(r"(?:rotate\s*:\s*(-?\d+))|(?:rotation of (-?\d+(?:\.\d+)?) degrees)")


def probe(path: Path) -> dict | None:
    """Duration, display size (rotation applied) and HDR flag, parsed from `ffmpeg -i` output."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                       errors="replace", timeout=60)
    err = r.stderr
    d, v = _DUR.search(err), _VID.search(err)
    if not d or not v:
        return None
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3))
    wh = _WH.search(v.group(1))
    if not wh or dur <= 0:
        return None
    w, h = int(wh.group(1)), int(wh.group(2))
    rot = 0.0
    for m in _ROT.finditer(err):
        rot = float(m.group(1) or m.group(2))
    if round(abs(rot)) % 180 == 90:
        w, h = h, w
    hdr = bool(re.search(r"arib-std-b67|smpte2084|bt2020", v.group(1)))
    return {"duration": dur, "width": w, "height": h, "hdr": hdr}


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def sample(path: Path, info: dict, width: int = SAMPLE_W, max_samples: int = 120):
    """RGB frames (times, array[n, h, w, 3]) for scoring: the keyframes only (phones write about one
    per second), decoded single-threaded — several times cheaper than decoding every frame of 4K
    video.  Long videos are sampled by seeking to `max_samples` keyframes instead."""
    w, h = width, _even(width * info["height"] / info["width"])
    dur = info["duration"]
    fast = [ffmpeg_exe(), "-v", "error", "-threads", "1", "-skip_frame", "nokey"]
    if dur <= 2 * max_samples:
        cmd = fast + ["-i", str(path), "-vf", f"scale={w}:{h}", "-fps_mode", "vfr",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
        k = len(raw) // (w * h * 3)
        frames = np.frombuffer(raw[: k * w * h * 3], np.uint8).reshape(k, h, w, 3)
        times = (np.arange(k) + 0.5) * dur / max(k, 1)   # keyframes are close to evenly spaced
        return times, frames
    times = np.linspace(0.5, dur - 0.5, max_samples)
    out = []
    for t in times:
        cmd = [ffmpeg_exe(), "-v", "error", "-threads", "1", "-ss", f"{t:.3f}", "-skip_frame", "nokey",
               "-i", str(path), "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
        out.append(np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3) if len(raw) >= w * h * 3
                   else np.zeros((h, w, 3), np.uint8))
    return times, np.stack(out)


def frame_scores(frames: np.ndarray):
    """Per-sample quality (sharpness, exposure, colour) and shake (global shift to the next sample)."""
    q, gray = [], []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        gray.append(g.astype(np.float32))
        a = f.astype(np.float32)
        rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
        colorful = math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean())
        sharp = cv2.Laplacian(g, cv2.CV_32F).var()
        q.append(1.2 * min(math.log1p(sharp) / 6.0, 1.3) - 2.0 * max(0, abs(g.mean() - 118) / 118 - 0.45)
                 + 0.8 * min(g.std() / 60, 1.0) + 0.5 * min(colorful / 50, 1.2))
    shake = np.zeros(len(frames))
    if len(gray) > 1:
        win = cv2.createHanningWindow(gray[0].shape[::-1], cv2.CV_32F)
        for i in range(len(gray) - 1):
            (dx, dy), _ = cv2.phaseCorrelate(gray[i], gray[i + 1], win)
            shake[i] = math.hypot(dx, dy) / gray[0].shape[1]  # fraction of the frame width per sample
        shake[-1] = shake[-2]
    return np.asarray(q), shake


def best_window(times, q, shake, duration: float, length: float):
    """(start, score) of the steadiest, sharpest `length`-second window, avoiding the first and
    last second (camera fumbling) when the video is long enough."""
    length = min(length, duration)
    margin = 1.0 if duration >= length + 2.0 else max(0.0, (duration - length) / 2)
    starts = np.arange(margin, max(margin, duration - length - margin) + 1e-6, 0.25)
    best = (margin, -1e9)
    for s in starts:
        m = (times >= s) & (times <= s + length)
        if not m.any():
            m = np.abs(times - (s + length / 2)) == np.abs(times - (s + length / 2)).min()
        sc = q[m].mean() - 6.0 * np.clip(shake[m] - 0.02, 0, None).mean()
        if sc > best[1]:
            best = (float(s), float(sc))
    return best


HDR_CHAIN = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
             "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,")


def transcode(src: Path, dst: Path, start: float, length: float, size: tuple[int, int], fps: int, hdr: bool):
    """Cut [start, start+length] to a small constant-frame-rate SDR H.264 file without audio,
    scaled to fit `size` (rotation is applied by ffmpeg's autorotate)."""
    W, H = size
    vf = (HDR_CHAIN if hdr else "") + (f"scale=w={W}:h={H}:force_original_aspect_ratio=decrease:"
                                       f"force_divisible_by=2,fps={fps},format=yuv420p")
    tmp = dst.with_suffix(".part.mp4")
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{length:.3f}",
           "-vf", vf, "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 and hdr:  # tone mapping failed on an unusual colour space: plain conversion
        cmd[cmd.index("-vf") + 1] = vf[len(HDR_CHAIN):]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src}: {r.stderr[-400:]}")
    os.replace(tmp, dst)


class ClipReader:
    """Sequential RGB frame access to a transcoded clip; frame k beyond the end repeats the last."""

    def __init__(self, path: str):
        self.path = path
        self.cap = None
        self.k = -1
        self.cur = None

    def frame(self, k: int) -> np.ndarray:
        if self.cap is None or k < self.k:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(self.path)
            self.k, self.cur = -1, None
        while self.k < k:
            ok, f = self.cap.read()
            if not ok:
                break
            self.k += 1
            self.cur = f
        if self.cur is None:
            raise RuntimeError(f"cannot read {self.path}")
        return cv2.cvtColor(self.cur, cv2.COLOR_BGR2RGB)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def grab(path: Path, t: float, width: int, info: dict) -> np.ndarray | None:
    """One RGB frame at time t, `width` pixels wide."""
    w, h = width, _even(width * info["height"] / info["width"])
    cmd = [ffmpeg_exe(), "-v", "error", "-threads", "1", "-ss", f"{max(0.0, t):.3f}", "-i", str(path),
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).copy() if len(raw) >= w * h * 3 else None
