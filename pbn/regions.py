"""Split each palette number into its separate paintable regions.

A region is one 8-connected blob of a single colour. Painters track progress region by region,
so every region gets a stable integer id (1..count; 0 means "no region", used for specks too
small to paint).
"""
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_position

NO_COLOR = 255
SEAM_DEPTH = 1.0  # regions this shallow (about 3px wide or less) are resampling seams, not cells


@dataclass
class Regions:
    ids: np.ndarray      # (H, W) int32 region id per pixel, 0 = none
    label: np.ndarray    # (R + 1,) uint8 palette index per region (index 0 unused)
    area: np.ndarray     # (R + 1,) int64 pixels
    anchor: np.ndarray   # (R + 1, 2) int32 (y, x) of the first pixel in reading order
    bbox: np.ndarray     # (R + 1, 4) int32 (x, y, w, h)
    pole: np.ndarray     # (R + 1, 2) int32 (y, x) of the pixel deepest inside the region
    _by_label: list = field(default_factory=list, repr=False)

    @property
    def count(self) -> int:
        return len(self.label) - 1

    def of_label(self, color: int) -> np.ndarray:
        """Region ids of a palette index, in reading order (top to bottom, left to right)."""
        if 0 <= color < len(self._by_label):
            return self._by_label[color]
        return np.empty(0, np.int32)


def build_regions(labels: np.ndarray, n_colors: int, min_area: int = 3,
                  min_depth: float = 0.0) -> Regions:
    """Regions of every colour. ``min_depth`` drops regions that never get deeper than that
    many pixels from their own boundary (thin seams); 0 keeps everything."""
    h, w = labels.shape
    ids = np.zeros((h, w), np.int32)
    label, area, anchor, bbox = [NO_COLOR], [0], [(0, 0)], [(0, 0, 0, 0)]

    for color in range(n_colors):
        mask = labels == color
        if not mask.any():
            continue
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        lut = np.zeros(n, np.int32)
        for i in range(1, n):
            x, y, bw, bh, px = (int(v) for v in stats[i])
            if px < min_area:
                continue
            lut[i] = len(label)
            label.append(color)
            area.append(px)
            bbox.append((x, y, bw, bh))
            anchor.append((y, x + int(np.argmax(cc[y, x:x + bw] == i))))  # top row, leftmost
        if lut.any():
            ids[mask] = lut[cc[mask]]

    label, area = np.array(label, np.uint8), np.array(area, np.int64)
    anchor, bbox = np.array(anchor, np.int32), np.array(bbox, np.int32)
    pole, depth = _poles(ids, len(label) - 1)

    if min_depth > 0:
        keep = depth > min_depth
        keep[0] = True
        if not keep.all():
            lut = np.zeros(len(label), np.int32)
            lut[keep] = np.arange(int(keep.sum()), dtype=np.int32)  # index 0 stays 0
            ids = lut[ids]
            label, area, anchor, bbox, pole = (a[keep] for a in (label, area, anchor, bbox, pole))

    by_label = []
    for color in range(n_colors):
        members = np.flatnonzero(label == color).astype(np.int32)
        order = np.argsort(anchor[members, 0].astype(np.int64) * w + anchor[members, 1])
        by_label.append(members[order])
    return Regions(ids, label, area, anchor, bbox, pole, by_label)


def clean_labels(labels: np.ndarray, n_colors: int, min_depth: float = SEAM_DEPTH) -> np.ndarray:
    """Hand thin seam regions to the nearest real region.

    Downscaling blends neighbouring fills along their boundary, and a blend can land close to a
    third fill colour, leaving hairline "regions" that nobody could paint.
    """
    regions = build_regions(labels, n_colors, min_area=1, min_depth=min_depth)
    gone = regions.ids == 0
    if not gone.any() or gone.all():
        return labels
    near = distance_transform_edt(gone, return_distances=False, return_indices=True)
    return labels[near[0], near[1]]


def _poles(ids: np.ndarray, count: int):
    """Per region, the pixel farthest from any boundary (always inside it) and that distance."""
    pole = np.zeros((count + 1, 2), np.int32)
    depth_at = np.zeros(count + 1)
    if count == 0:
        return pole, depth_at
    padded = np.pad(ids, 1)  # the canvas edge counts as a boundary too
    edge = np.zeros(padded.shape, bool)
    across = padded[:, 1:] != padded[:, :-1]
    edge[:, 1:] |= across
    edge[:, :-1] |= across
    down = padded[1:, :] != padded[:-1, :]
    edge[1:, :] |= down
    edge[:-1, :] |= down
    depth = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    pole[1:] = np.array(maximum_position(depth, labels=ids, index=np.arange(1, count + 1)), np.int32)
    depth_at[1:] = depth[pole[1:, 0], pole[1:, 1]]
    return pole, depth_at
