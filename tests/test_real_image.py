"""Smoke test on the real kit in data/test_image.jpg (skipped if the file is absent)."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from pbn.canvas import analyze

IMAGE = Path(__file__).resolve().parent.parent / "data" / "test_image.jpg"


@unittest.skipUnless(IMAGE.exists(), "data/test_image.jpg not available")
class RealKitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bgr = cv2.imdecode(np.fromfile(str(IMAGE), np.uint8), cv2.IMREAD_COLOR)
        cls.a = analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 1400)

    def test_reads_all_36_numbers_in_printed_order(self):
        self.assertEqual(len(self.a.palette), 36)
        # spot-check swatches against the printed strip: #3 dark grey, #9 pale blue, #33 cream
        for number, rgb in {3: (60, 63, 68), 9: (195, 233, 252), 33: (242, 218, 172)}.items():
            self.assertLessEqual(np.abs(self.a.palette[number - 1].astype(int) - rgb).max(), 6)

    def test_recovers_the_print_fade(self):
        self.assertAlmostEqual(self.a.fade, 0.349, delta=0.02)

    def test_every_number_is_found_on_the_canvas(self):
        self.assertTrue(self.a.matched.all(), f"unmatched: {np.flatnonzero(~self.a.matched) + 1}")
        self.assertTrue((self.a.area > 1e-4).all(), f"tiny: {np.flatnonzero(self.a.area <= 1e-4) + 1}")

    def test_canvas_excludes_margin_palette_and_logo(self):
        x0, y0, x1, y1 = self.a.box
        self.assertGreater(x0, 300)   # palette strip sits left of the canvas
        self.assertGreater(y0, 300)   # logo sits above it
        self.assertLess(y1 - y0, 6733 - 600)

    def test_most_of_the_canvas_is_flat_fill_rather_than_ink(self):
        self.assertLess(self.a.ink.mean(), 0.25)


if __name__ == "__main__":
    unittest.main()
