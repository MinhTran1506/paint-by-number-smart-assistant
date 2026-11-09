import unittest

import numpy as np

from pbn.palette import PaletteNotFound, find_palette
from tests.synthetic import jpeg_roundtrip, make_canvas


class FindPaletteTests(unittest.TestCase):
    def test_returns_every_swatch_in_printed_order(self):
        syn = make_canvas()
        pal = find_palette(syn.image)
        self.assertEqual(pal.colors.shape, (len(syn.swatches), 3))
        self.assertLessEqual(np.abs(pal.colors.astype(int) - syn.swatches).max(), 2)

    def test_survives_jpeg_compression(self):
        syn = make_canvas()
        pal = find_palette(jpeg_roundtrip(syn.image, 80))
        self.assertEqual(len(pal.colors), len(syn.swatches))
        self.assertLessEqual(np.abs(pal.colors.astype(int) - syn.swatches).max(), 8)

    def test_works_on_a_much_larger_sheet(self):
        syn = make_canvas(scale=3)
        pal = find_palette(syn.image)
        self.assertEqual(len(pal.colors), len(syn.swatches))
        self.assertLessEqual(np.abs(pal.colors.astype(int) - syn.swatches).max(), 2)

    def test_box_encloses_the_strip_but_not_the_canvas(self):
        syn = make_canvas()
        x0, y0, x1, y1 = find_palette(syn.image).box
        self.assertLess(x1, syn.box[0])
        self.assertLessEqual(x0, 30)
        self.assertGreaterEqual(x1, 70)
        self.assertLessEqual(y0, 140)
        self.assertGreaterEqual(y1, 140 + 11 * 46 + 30)

    def test_raises_when_the_sheet_has_no_palette(self):
        with self.assertRaises(PaletteNotFound):
            find_palette(make_canvas(with_palette=False).image)

    def test_raises_on_a_blank_page(self):
        with self.assertRaises(PaletteNotFound):
            find_palette(np.full((400, 300, 3), 255, np.uint8))


if __name__ == "__main__":
    unittest.main()
