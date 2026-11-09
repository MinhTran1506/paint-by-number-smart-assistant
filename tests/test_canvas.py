import unittest

import numpy as np

from pbn.canvas import analyze, find_canvas_box
from pbn.palette import PaletteNotFound
from tests.synthetic import CELL, jpeg_roundtrip, make_canvas

BIG = 5000  # larger than the synthetic sheet, so analysis runs at 1:1


def truth_crop(syn):
    x0, y0, x1, y1 = syn.box
    return syn.truth[y0:y1, x0:x1]


def interior(shape, margin=2):
    """Pixels away from cell borders; a border pixel belongs equally to both neighbours."""
    ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
    ry, rx = ys % CELL, xs % CELL
    return (ry >= margin) & (ry < CELL - margin) & (rx >= margin) & (rx < CELL - margin)


class FindCanvasBoxTests(unittest.TestCase):
    def test_box_matches_the_printed_area_not_frame_logo_or_palette(self):
        syn = make_canvas()
        got = find_canvas_box(syn.image)
        for g, want in zip(got, syn.box):
            self.assertLessEqual(abs(g - want), 1)

    def test_box_is_exact_on_a_larger_sheet(self):
        syn = make_canvas(scale=3)
        got = find_canvas_box(syn.image)
        for g, want in zip(got, syn.box):
            self.assertLessEqual(abs(g - want), 1)


class AnalyzeTests(unittest.TestCase):
    def test_every_pixel_gets_the_palette_number_of_its_cell(self):
        syn = make_canvas()
        a = analyze(syn.image, BIG)
        truth = truth_crop(syn)
        self.assertEqual(a.labels.shape, truth.shape)
        inner = interior(truth.shape)
        self.assertGreaterEqual((a.labels == truth)[inner].mean(), 0.999)

    def test_near_identical_dark_swatches_are_told_apart(self):
        syn = make_canvas()
        a = analyze(syn.image, BIG)
        truth = truth_crop(syn)
        inner = interior(truth.shape)
        for dark in (0, 1, 3):  # swatches 1, 2 and 4 fade to greys only 3-7 levels apart
            self.assertGreater(((a.labels == dark) == (truth == dark))[inner].mean(), 0.999)

    def test_numbering_follows_the_printed_palette_order(self):
        syn = make_canvas()
        a = analyze(syn.image, BIG)
        self.assertLessEqual(np.abs(a.palette.astype(int) - syn.swatches).max(), 2)

    def test_fade_is_recovered(self):
        for s in (0.35, 0.5):
            a = analyze(make_canvas(fade_s=s).image, BIG)
            self.assertAlmostEqual(a.fade, s, delta=0.02)

    def test_digits_and_lines_are_ink_and_take_the_surrounding_label(self):
        syn = make_canvas()
        a = analyze(syn.image, BIG)
        truth = truth_crop(syn)
        self.assertGreater(a.ink.mean(), 0.005)
        self.assertLess(a.ink.mean(), 0.2)
        digits = a.ink & interior(truth.shape, margin=3)
        self.assertGreater(digits.sum(), 1000)
        self.assertGreaterEqual((a.labels == truth)[digits].mean(), 0.99)

    def test_area_fractions_cover_the_canvas(self):
        a = analyze(make_canvas().image, BIG)
        self.assertAlmostEqual(float(a.area.sum()), 1.0, places=6)
        self.assertTrue((a.area > 0).all())

    def test_every_number_is_reported_as_matched(self):
        a = analyze(make_canvas().image, BIG)
        self.assertTrue(a.matched.all())

    def test_works_after_jpeg_compression(self):
        syn = make_canvas()
        a = analyze(jpeg_roundtrip(syn.image, 85), BIG)
        truth = truth_crop(syn)
        self.assertGreaterEqual((a.labels == truth)[interior(truth.shape, 3)].mean(), 0.98)

    def test_large_sheet_is_downscaled_to_the_processing_size(self):
        syn = make_canvas(scale=3)
        a = analyze(syn.image, 800)
        self.assertEqual(max(a.canvas.shape[:2]), 800)
        self.assertEqual(a.labels.shape, a.canvas.shape[:2])
        self.assertLess(a.scale, 1)
        self.assertAlmostEqual(a.area[0], 1 / len(syn.swatches), delta=0.05)

    def test_sheet_without_a_palette_is_rejected(self):
        with self.assertRaises(PaletteNotFound):
            analyze(make_canvas(with_palette=False).image, BIG)


if __name__ == "__main__":
    unittest.main()
