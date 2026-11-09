"""Find the column of numbered colour swatches printed on a paint-by-number sheet.

Swatch *i* from the top is printed number *i + 1*, so the order of the returned colours
is the numbering the painter sees on the sheet.
"""
from dataclasses import dataclass

import cv2
import numpy as np

WHITE_THRESHOLD = 235     # a pixel whose darkest channel is below this is "not paper"
MAX_SWATCH_FRACTION = 0.1  # swatches are small next to the page; the canvas is not
MIN_SOLIDITY = 0.85        # a swatch is a filled rectangle


class PaletteNotFound(ValueError):
    """No column of evenly spaced swatches could be located on the sheet."""


@dataclass(frozen=True)
class Palette:
    colors: np.ndarray  # (N, 3) uint8; row i is printed number i + 1
    box: tuple          # (x0, y0, x1, y1) of the swatch column


def _solid_blocks(rgb):
    """Bounding boxes (x, y, w, h) of solid non-white rectangles on the page."""
    h, w = rgb.shape[:2]
    ink = (rgb.min(axis=2) < WHITE_THRESHOLD).astype(np.uint8)
    k = max(3, min(h, w) // 400)  # opening removes digits, hairlines and the page frame
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=4)
    blocks = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if bw > w * MAX_SWATCH_FRACTION or bh > h * MAX_SWATCH_FRACTION:
            continue
        if bw < k * 2 or bh < k * 2 or area < MIN_SOLIDITY * bw * bh:
            continue
        blocks.append((x, y, bw, bh))
    return blocks


def _best_column(blocks):
    """The largest set of equally sized blocks that share a left edge."""
    best = []
    for rx, _, rw, rh in blocks:
        group = [b for b in blocks
                 if abs(b[0] - rx) <= max(3, 0.15 * rw)
                 and abs(b[2] - rw) <= 0.15 * rw
                 and abs(b[3] - rh) <= 0.2 * rh]
        if len(group) > len(best):
            best = group
    return sorted(best, key=lambda b: b[1])


def find_palette(rgb: np.ndarray, min_swatches: int = 4) -> Palette:
    column = _best_column(_solid_blocks(rgb))
    if len(column) < min_swatches:
        raise PaletteNotFound(
            f"Found only {len(column)} swatch-like blocks; expected a column of at least "
            f"{min_swatches} evenly spaced colour swatches.")

    xs = np.array([b[0] for b in column])
    ys = np.array([b[1] for b in column])
    bw = int(np.median([b[2] for b in column]))
    bh = int(np.median([b[3] for b in column]))
    pitch = float(np.median(np.diff(ys)))
    if pitch < bh:
        raise PaletteNotFound("Swatch blocks overlap; this does not look like a palette column.")

    # Sample on a regular grid so a swatch too pale to be detected (near-white) still counts
    # and the numbering cannot silently shift.
    count = int(round((ys[-1] - ys[0]) / pitch)) + 1
    cx = int(np.median(xs)) + bw // 2
    colors = []
    for i in range(count):
        cy = int(round(ys[0] + i * pitch)) + bh // 2
        patch = rgb[cy - bh // 4:cy + bh // 4 + 1, cx - bw // 4:cx + bw // 4 + 1]
        colors.append(np.median(patch.reshape(-1, 3), axis=0))

    box = (int(xs.min()), int(ys.min()), int(xs.max()) + bw, int(ys[-1]) + bh)
    return Palette(np.round(colors).astype(np.uint8), box)
