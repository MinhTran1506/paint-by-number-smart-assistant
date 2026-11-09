"""Draw the canvas so the eye lands on what is left to paint.

Selected number: its *unpainted* regions fill with the true swatch colour and get a thin edge
outline. Everything else is dimmed to a neutral grey whose brightness is chosen to contrast with
the selected swatch. Painted regions, of any number, show softly in their true colour so
progress is visible. One optional region (the target of "next region") gets its own outline.
Printed digits and outlines (``ink``) are never painted over, only recoloured for contrast.
"""
import cv2
import numpy as np

from pbn.canvas import Analysis
from pbn.regions import Regions

OUTLINE_COLOR = (255, 40, 120)
FOCUS_COLOR = (255, 230, 0)
DONE_MIX = 0.55          # share of the true swatch colour blended into painted regions
DIM_LIGHT, DIM_DARK = 226.0, 72.0


def _luminance(rgb):
    rgb = np.asarray(rgb, np.float32)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _edge_band(mask: np.ndarray, thickness: int) -> np.ndarray:
    """Pixels of ``mask`` within ``thickness`` px of its boundary."""
    k = np.ones((2 * thickness + 1, 2 * thickness + 1), np.uint8)
    return mask & (cv2.erode(mask.astype(np.uint8), k) == 0)


def render_frame(a: Analysis, regions: Regions, selected: int, painted, focus=None) -> np.ndarray:
    labels, ink, palette = a.labels, a.ink, a.palette.astype(np.float32)
    swatch = palette[selected]

    # Dim backdrop: dark behind a light swatch, light behind a dark one.
    light_backdrop = _luminance(swatch) <= 150
    base = DIM_LIGHT if light_backdrop else DIM_DARK
    grey = np.clip(base + 0.25 * (_luminance(a.canvas) - 190.0), 0, 255)
    out = np.repeat(grey[..., None], 3, axis=2)
    out[ink] = (grey[ink] + (-55.0 if light_backdrop else 60.0))[:, None]  # faint digits

    is_painted = np.zeros(regions.count + 1, bool)
    if painted:
        is_painted[list(painted)] = True
    is_painted = is_painted[regions.ids]

    soft = is_painted & ~ink
    out[soft] = DONE_MIX * palette[labels[soft]] + (1 - DONE_MIX) * out[soft]

    chosen = (labels == selected) & ~is_painted
    out[chosen & ~ink] = swatch
    out[chosen & ink] = 255.0 if _luminance(swatch) < 110 else 25.0  # digits stay readable

    t = max(2, max(labels.shape) // 700)
    out[_edge_band(chosen, t)] = OUTLINE_COLOR
    if focus:
        out[_edge_band(regions.ids == focus, t)] = FOCUS_COLOR  # same width: keep tiny fills visible
    return np.clip(out, 0, 255).astype(np.uint8)
