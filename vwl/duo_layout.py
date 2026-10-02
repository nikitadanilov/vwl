"""Screen geometry of a two-person film, shared by the planner (which packs photos into rows of the
right shape) and the renderer.

    ┌───────────────────────────────┐   together            ┌──────────────┬──────────────┐   apart
    │  photo  │  photo  │  photo     │                      │ A's photos   │ B's photos   │
    ├───────────────────────────────┤                       ├──────────────┼──────────────┤
    │           one wide map         │                      │   A's map    │   B's map    │
    ├───────────────────────────────┤                       ├──────────────┴──────────────┤
    │ timeline strip                 │                      │ timeline strip               │
    └───────────────────────────────┘                       └──────────────────────────────┘
"""
from __future__ import annotations

MAP_FRAC = 0.36          # share of the frame height given to the map band


class DuoLayout:
    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.s = s = H / 1080
        self.margin = int(24 * s)
        self.gap = int(14 * s)
        self.strip_h = int(46 * s)
        self.mh = int(H * MAP_FRAC) // 2 * 2                       # map band height
        self.my = H - self.strip_h - self.mh - self.gap // 2        # map band top
        self.photo_h = self.my - self.gap - self.margin             # photo area height
        self.half_w = (W - 2 * self.margin - self.gap) // 2
        # rectangles (x, y, w, h)
        self.photos_together = (self.margin, self.margin, W - 2 * self.margin, self.photo_h)
        self.photos_left = (self.margin, self.margin, self.half_w, self.photo_h)
        self.photos_right = (W - self.margin - self.half_w, self.margin, self.half_w, self.photo_h)
        self.map_together = (self.margin, self.my, W - 2 * self.margin, self.mh)
        self.map_left = (self.margin, self.my, self.half_w, self.mh)
        self.map_right = (W - self.margin - self.half_w, self.my, self.half_w, self.mh)


def row_layout(aspects: list[float], region: tuple, gap: int) -> list[tuple[int, int, int, int]]:
    """Cells for photos side by side in one row, each at its own aspect ratio, as large as the
    region allows, centred.  Returns (x, y, w, h) per photo."""
    x0, y0, W, H = region
    n = len(aspects)
    h = min(H, (W - gap * (n - 1)) / max(sum(aspects), 1e-6))
    widths = [a * h for a in aspects]
    total = sum(widths) + gap * (n - 1)
    x = x0 + (W - total) / 2
    y = y0 + (H - h) / 2
    cells = []
    for w in widths:
        cells.append((int(round(x)), int(round(y)), int(round(w)), int(round(h))))
        x += w + gap
    return cells


def row_fill(aspects: list[float], region: tuple, gap: int) -> float:
    """Fraction of the region a row of these photos covers."""
    cells = row_layout(aspects, region, gap)
    return sum(w * h for _, _, w, h in cells) / (region[2] * region[3])


def best_count(aspects: list[float], region: tuple, gap: int, kmax: int) -> int:
    """How many of the next photos to show together: the count that covers the region best, with a
    small bonus per photo so two portraits beat one squeezed landscape."""
    best, best_k = -1.0, 1
    for k in range(1, min(kmax, len(aspects)) + 1):
        score = row_fill(aspects[:k], region, gap) + 0.08 * k
        if score > best:
            best, best_k = score, k
    return best_k
