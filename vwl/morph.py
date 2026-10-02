"""Photo canvases, Ken Burns drift, and optical-flow morphing between two photos.

The morph estimates dense optical flow A→B and B→A (DIS, on a downscaled copy),
smooths it heavily so unrelated photos "melt" into each other instead of tearing,
then warps both images part-way along the flow and cross-dissolves.  For photos
of the same scene the flow is meaningful and the morph looks like motion.
"""
from __future__ import annotations

import functools

import cv2
import numpy as np


def make_canvas(img_rgb: np.ndarray, w: int, h: int) -> np.ndarray:
    """Fit a photo into w×h: cover when aspect is close, else contain over a blurred fill."""
    ih, iw = img_rgb.shape[:2]
    ar, car = iw / ih, w / h
    cover = max(w / iw, h / ih)
    if abs(np.log(ar / car)) < 0.2:
        s = cover
        r = cv2.resize(img_rgb, (int(round(iw * s)), int(round(ih * s))), interpolation=cv2.INTER_AREA)
        y0, x0 = (r.shape[0] - h) // 2, (r.shape[1] - w) // 2
        return np.ascontiguousarray(r[y0:y0 + h, x0:x0 + w])
    small = cv2.resize(img_rgb, (max(1, int(iw * cover / 12)), max(1, int(ih * cover / 12))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6)
    bg = cv2.resize(small, (int(round(iw * cover)), int(round(ih * cover))), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (bg.shape[0] - h) // 2, (bg.shape[1] - w) // 2
    bg = (bg[y0:y0 + h, x0:x0 + w].astype(np.float32) * 0.42).astype(np.uint8)
    s = min(w / iw, h / ih) * 0.94
    fw, fh = int(round(iw * s)), int(round(ih * s))
    fg = cv2.resize(img_rgb, (fw, fh), interpolation=cv2.INTER_AREA)
    fx, fy = (w - fw) // 2, (h - fh) // 2
    bg = (bg * _shadow(w, h, fx, fy, fw, fh)).astype(np.uint8)
    bg[fy:fy + fh, fx:fx + fw] = fg
    return bg


@functools.lru_cache(maxsize=8)
def _shadow(w, h, fx, fy, fw, fh) -> np.ndarray:
    """Multiplier (h, w, 1) that darkens a soft drop shadow under the fitted photo.  Cached: for
    video every frame of a clip has the same geometry, and the blur is the expensive part."""
    sh = np.zeros((h, w), np.float32)
    sh[fy + 8:fy + fh + 8, fx + 8:fx + fw + 8] = 1
    return 1 - cv2.GaussianBlur(sh, (0, 0), 18)[..., None] * 0.55


def ken_burns(canvas: np.ndarray, u: float, zoom=0.06, drift=(0.0, 0.0)) -> np.ndarray:
    """Slow zoom-in (and drift) over u∈[0,1]."""
    h, w = canvas.shape[:2]
    s = 1 + zoom * u
    dx, dy = drift[0] * u * w * zoom, drift[1] * u * h * zoom
    M = np.array([[s, 0, (1 - s) * w / 2 + dx], [0, s, (1 - s) * h / 2 + dy]], np.float32)
    return cv2.warpAffine(canvas, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


class Morph:
    def __init__(self, a: np.ndarray, b: np.ndarray, strength: float = 1.0, flow_width: int = 480):
        self.a, self.b = a, b
        h, w = a.shape[:2]
        fs = flow_width / w
        ga = cv2.cvtColor(cv2.resize(a, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, None, fx=fs, fy=fs, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        fab = dis.calc(ga, gb, None)
        fba = dis.calc(gb, ga, None)
        sig = flow_width / 60

        def prep(f):
            f = cv2.GaussianBlur(f, (0, 0), sig)
            mag = np.linalg.norm(f, axis=2, keepdims=True)
            cap = flow_width * 0.12
            f = f * np.minimum(1, cap / np.maximum(mag, 1e-6))
            f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / fs
            return (f * strength).astype(np.float32)

        self.fab, self.fba = prep(fab), prep(fba)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.gx, self.gy = gx, gy

    def frame(self, t: float, a: np.ndarray | None = None) -> np.ndarray:
        """Blend at t∈[0,1].  `a` replaces the outgoing image, e.g. a video that keeps playing
        through the transition; the flow estimated at the transition's start is reused for it."""
        a = self.a if a is None else a
        a_w = cv2.remap(a, self.gx - t * self.fab[..., 0], self.gy - t * self.fab[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        b_w = cv2.remap(self.b, self.gx - (1 - t) * self.fba[..., 0], self.gy - (1 - t) * self.fba[..., 1],
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        k = t * t * (3 - 2 * t)
        return cv2.addWeighted(a_w, 1 - k, b_w, k, 0)
