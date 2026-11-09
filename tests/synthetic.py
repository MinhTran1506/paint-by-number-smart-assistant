"""A synthetic paint-by-number sheet that mimics real kits (see data/test_image.jpg).

Layout: white page with a thin outer frame, a logo-ish scribble, a column of numbered
swatches on the left, and a canvas whose cells are the swatch colours faded towards
white, separated by thin ink lines and carrying small printed digits.
"""
from dataclasses import dataclass

import cv2
import numpy as np

FADE = 0.35  # opacity of the original colour laid over white
INK = (150, 80, 80)

# Index i is printed number i + 1. Entries 0, 1 and 3 are near-identical darks, like the
# real kit, where several black-ish swatches fade to almost the same grey.
SWATCHES = np.array(
    [
        [20, 20, 22], [0, 0, 0], [60, 63, 68], [28, 29, 31],
        [130, 116, 107], [197, 177, 188], [195, 233, 252], [221, 141, 70],
        [133, 40, 85], [114, 112, 187], [242, 218, 172], [26, 61, 54],
    ],
    np.uint8,
)

CELL = 60
COLS, ROWS = 13, 18
CANVAS_BOX = (130, 100, 130 + COLS * CELL, 100 + ROWS * CELL)  # x0, y0, x1, y1
PAGE_W, PAGE_H = 1000, 1300


def fade(color, s=FADE):
    return np.round(255 - s * (255 - np.asarray(color, float))).astype(np.uint8)


def cell_label(row, col, n=len(SWATCHES)):
    return (row * 5 + col * 3) % n  # neighbours always differ, every label is used


@dataclass
class Synthetic:
    image: np.ndarray        # (H, W, 3) RGB uint8
    swatches: np.ndarray     # (N, 3) true swatch colours
    truth: np.ndarray        # (H, W) int, -1 outside the canvas
    box: tuple
    fade: float


def make_canvas(with_palette=True, scale=1, fade_s=FADE) -> Synthetic:
    img = np.full((PAGE_H, PAGE_W, 3), 255, np.uint8)
    truth = np.full((PAGE_H, PAGE_W), -1, np.int16)

    cv2.rectangle(img, (0, 0), (PAGE_W - 1, PAGE_H - 1), (120, 120, 120), 2)  # page frame
    cv2.putText(img, "HJ", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3)  # logo
    cv2.putText(img, "YN124", (130, 1235), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1)

    if with_palette:
        for i, color in enumerate(SWATCHES):
            y = 140 + i * 46
            img[y:y + 30, 30:70] = color
            cv2.putText(img, str(i + 1), (6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (30, 30, 30), 1)

    x0, y0, _, _ = CANVAS_BOX
    for r in range(ROWS):
        for c in range(COLS):
            lab = cell_label(r, c)
            xs, ys = x0 + c * CELL, y0 + r * CELL
            img[ys:ys + CELL, xs:xs + CELL] = fade(SWATCHES[lab], fade_s)
            truth[ys:ys + CELL, xs:xs + CELL] = lab
    for r in range(ROWS + 1):
        img[y0 + r * CELL - (r == ROWS):y0 + r * CELL + 1 - (r == ROWS), x0:CANVAS_BOX[2]] = INK
    for c in range(COLS + 1):
        img[y0:CANVAS_BOX[3], x0 + c * CELL - (c == COLS):x0 + c * CELL + 1 - (c == COLS)] = INK
    for r in range(ROWS):
        for c in range(COLS):
            cv2.putText(img, str(cell_label(r, c) + 1), (x0 + c * CELL + 20, y0 + r * CELL + 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, INK, 1)

    box = CANVAS_BOX
    if scale != 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        truth = cv2.resize(truth, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        box = tuple(int(v * scale) for v in box)
    return Synthetic(img, SWATCHES.copy(), truth, box, fade_s)


def jpeg_roundtrip(rgb, quality=85):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
