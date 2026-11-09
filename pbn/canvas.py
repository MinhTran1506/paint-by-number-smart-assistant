"""Turn a photographed/scanned paint-by-number sheet into a per-pixel palette-number map.

Printed canvases show every region in its swatch colour *faded towards white*
(``canvas = 255 - s * (255 - swatch)``). We fit that one opacity ``s`` by matching the
swatches one-to-one against the flat fill colours found on the canvas, then label each
pixel with the number whose fill colour it matches.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from pbn.palette import find_palette
from pbn.regions import clean_labels

PAPER = 246          # canvas pixels have a channel below this; blank paper does not
PEAK_SEPARATION = 3.5  # RGB distance below which two colour peaks are the same fill
UNMATCHED_COST = 60.0  # matching cost of a swatch with no plausible canvas colour
MATCH_TOLERANCE = 12.0  # a swatch counts as found if its fill is this close to the model
PIXEL_TOLERANCE = 7.0   # a pixel is "fill" if this close to a fill colour, else "ink"


@dataclass
class Analysis:
    canvas: np.ndarray   # (H, W, 3) uint8, the canvas at processing size
    labels: np.ndarray   # (H, W) uint8, palette index 0..N-1 (printed number = index + 1)
    ink: np.ndarray      # (H, W) bool, outlines / digits / anti-aliasing (not flat fill)
    palette: np.ndarray  # (N, 3) uint8, true swatch colours
    area: np.ndarray     # (N,) float, share of the canvas per number
    matched: np.ndarray  # (N,) bool, a flat fill matching the number was found
    fade: float          # fitted opacity of the printed colours over white
    box: tuple           # (x0, y0, x1, y1) of the canvas in the uploaded image
    scale: float         # processing size / uploaded size


def find_canvas_box(rgb: np.ndarray) -> tuple:
    """Bounding box (x0, y0, x1, y1) of the main printed area, in pixels of ``rgb``."""
    h, w = rgb.shape[:2]
    s = min(1.0, 1000 / max(h, w))
    small = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else rgb
    mask = (small.min(axis=2) < PAPER).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # drop frame/logo
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n < 2:
        raise ValueError("No printed canvas area found on the sheet.")
    x, y, bw, bh, _ = (int(v) for v in stats[1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])])

    # Refine to the exact edge at full resolution.
    pad = int(np.ceil(1 / s)) + 2
    X0, Y0 = max(0, int(x / s) - pad), max(0, int(y / s) - pad)
    X1, Y1 = min(w, int((x + bw) / s) + pad), min(h, int((y + bh) / s) + pad)
    painted = rgb[Y0:Y1, X0:X1].min(axis=2) < PAPER
    cols = np.flatnonzero(painted.mean(axis=0) > 0.5)
    rows = np.flatnonzero(painted.mean(axis=1) > 0.5)
    return X0 + int(cols[0]), Y0 + int(rows[0]), X0 + int(cols[-1]) + 1, Y0 + int(rows[-1]) + 1


def _pack(rgb_flat):
    c = rgb_flat.astype(np.uint32)
    return (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]


def _unpack(keys):
    return np.stack([(keys >> 16) & 255, (keys >> 8) & 255, keys & 255], axis=1).astype(np.float64)


def _dominant_colors(canvas, max_peaks=300):
    """Most common flat colours, strongest first, with near-duplicates merged."""
    keys = _pack(canvas.reshape(-1, 3))
    uniq, counts = np.unique(keys, return_counts=True)
    min_count = max(20, int(2e-5 * keys.size))
    peaks = np.empty((0, 3))
    for j in np.argsort(-counts):
        if counts[j] < min_count or len(peaks) >= max_peaks:
            break
        color = _unpack(uniq[j:j + 1])
        if len(peaks) and cdist(color, peaks).min() < PEAK_SEPARATION:
            continue
        peaks = np.vstack([peaks, color])
    return peaks


def _faded(swatches, s):
    return 255.0 - s * (255.0 - swatches)


def _match(swatches, peaks, s):
    cost = np.minimum(cdist(_faded(swatches, s), peaks), UNMATCHED_COST)
    rows, cols = linear_sum_assignment(cost)
    total = cost[rows, cols].sum() + UNMATCHED_COST * (len(swatches) - len(rows))
    return total, rows, cols


def fit_fade(swatches: np.ndarray, peaks: np.ndarray):
    """Best opacity ``s`` and the one-to-one swatch -> peak assignment for it."""
    sw = swatches.astype(np.float64)
    coarse = np.arange(0.05, 1.0001, 0.005)
    s0 = min(coarse, key=lambda s: _match(sw, peaks, s)[0])
    fine = np.arange(max(0.01, s0 - 0.005), min(1.0, s0 + 0.005) + 1e-9, 0.0005)
    s = min(fine, key=lambda s: _match(sw, peaks, s)[0])
    _, rows, cols = _match(sw, peaks, s)
    return float(s), rows, cols


def _label_pixels(canvas, centers):
    """Nearest fill colour per pixel (-1 when no fill colour is close enough)."""
    keys = _pack(canvas.reshape(-1, 3))
    uniq, inverse = np.unique(keys, return_inverse=True)
    dist = cdist(_unpack(uniq), centers)
    nearest = dist.argmin(axis=1)
    nearest[dist.min(axis=1) > PIXEL_TOLERANCE] = -1
    return nearest[inverse].reshape(canvas.shape[:2])


def analyze(rgb: np.ndarray, max_side: int = 2000) -> Analysis:
    palette = find_palette(rgb)
    x0, y0, x1, y1 = find_canvas_box(rgb)
    canvas = rgb[y0:y1, x0:x1]
    scale = min(1.0, max_side / max(canvas.shape[:2]))
    if scale < 1:
        size = (round(canvas.shape[1] * scale), round(canvas.shape[0] * scale))
        canvas = cv2.resize(canvas, size, interpolation=cv2.INTER_AREA)
    canvas = np.ascontiguousarray(canvas)

    peaks = _dominant_colors(canvas)
    if len(peaks) == 0:
        raise ValueError("The canvas has no flat colour regions to match against the palette.")
    s, rows, cols = fit_fade(palette.colors, peaks)

    centers = _faded(palette.colors.astype(np.float64), s)  # fallback for unmatched numbers
    matched = np.zeros(len(palette.colors), bool)
    for r, c in zip(rows, cols):
        if np.linalg.norm(centers[r] - peaks[c]) <= MATCH_TOLERANCE:
            centers[r] = peaks[c]  # use the real fill colour so JPEG/print drift cancels out
            matched[r] = True

    labels = _label_pixels(canvas, centers)
    ink = labels < 0
    # Anti-aliased edges blend ink with the fill and can land near a *different* fill colour,
    # so distrust the 1px halo around ink as well as the ink itself.
    halo = cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if halo.all():
        raise ValueError("No canvas pixel matches any palette colour.")
    if halo.any():  # lines, digits and edges take the label of the nearest trustworthy pixel
        near = distance_transform_edt(halo, return_distances=False, return_indices=True)
        labels = labels[near[0], near[1]]

    labels = clean_labels(labels.astype(np.uint8), len(palette.colors))
    area = np.bincount(labels.ravel(), minlength=len(palette.colors)) / labels.size
    return Analysis(canvas, labels, ink, palette.colors, area, matched, s,
                    (x0, y0, x1, y1), float(scale))
