import unittest

import numpy as np

from pbn.canvas import analyze
from pbn.regions import build_regions
from pbn.render import FOCUS_COLOR, OUTLINE_COLOR, render_frame
from tests.synthetic import CELL, COLS, ROWS, cell_label, make_canvas

LIGHT, DARK = 10, 2  # swatch 11 (cream, #11) and swatch 3 (dark grey, #3) in the fixture


def lum(rgb):
    rgb = np.asarray(rgb, float)
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


class RenderFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = analyze(make_canvas().image, 5000)
        cls.r = build_regions(cls.a.labels, len(cls.a.palette))

    def fill(self, number):
        return (self.a.labels == number) & ~self.a.ink

    def cells(self, number):
        """Region ids of every printed cell of one colour."""
        return [int(self.r.ids[row * CELL + CELL // 2, col * CELL + CELL // 2])
                for row in range(ROWS) for col in range(COLS) if cell_label(row, col) == number]

    def region_fill(self, rid):
        return (self.r.ids == rid) & ~self.a.ink

    def outline(self, out):
        return np.all(out == OUTLINE_COLOR, axis=2)

    def test_selected_fill_is_the_true_swatch_colour(self):
        out = render_frame(self.a, self.r, 4, set())
        inner = self.fill(4) & ~self.outline(out)
        self.assertGreater(inner.sum(), 1000)
        self.assertTrue((out[inner] == self.a.palette[4]).all())

    def test_other_colours_are_dimmed_to_neutral_grey(self):
        out = render_frame(self.a, self.r, 4, set()).astype(int)
        others = self.fill(5)
        spread = out[others].max(axis=1) - out[others].min(axis=1)
        self.assertLessEqual(spread.max(), 2)

    def test_background_turns_dark_behind_a_light_swatch_and_light_behind_a_dark_one(self):
        light = render_frame(self.a, self.r, LIGHT, set())
        dark = render_frame(self.a, self.r, DARK, set())
        other = self.fill(5)
        self.assertLess(lum(light[other]).mean(), 120)
        self.assertGreater(lum(dark[other]).mean(), 200)

    def test_a_painted_region_shows_softly_in_its_true_hue(self):
        rid = self.cells(7)[0]
        plain = render_frame(self.a, self.r, 4, set()).astype(int)
        done = render_frame(self.a, self.r, 4, {rid}).astype(int)
        px = self.region_fill(rid)
        target = self.a.palette[7].astype(int)
        self.assertTrue((np.abs(done[px] - target).sum(1) < np.abs(plain[px] - target).sum(1)).all())
        self.assertTrue((np.abs(done[px] - target).sum(1) > 0).all())  # softer than the selection
        sibling = self.region_fill(self.cells(7)[1])
        self.assertTrue((done[sibling] == plain[sibling]).all())  # other regions are untouched

    def test_painted_regions_of_the_selected_colour_stop_standing_out(self):
        first, second = self.cells(4)[:2]
        out = render_frame(self.a, self.r, 4, {first})
        painted, left = self.region_fill(first), self.region_fill(second)
        self.assertFalse((out[painted] == self.a.palette[4]).all(axis=1).any())  # soft tint only
        self.assertFalse(self.outline(out)[self.r.ids == first].any())            # no outline
        inner = left & ~self.outline(out)
        self.assertTrue((out[inner] == self.a.palette[4]).all())                  # still vivid
        self.assertTrue(self.outline(out)[self.r.ids == second].any())            # still outlined

    def test_region_edges_are_outlined_inside_the_region_only(self):
        out = render_frame(self.a, self.r, 4, set())
        outline = self.outline(out)
        self.assertGreater(outline.sum(), 100)
        self.assertFalse((outline & (self.a.labels != 4)).any())

    def test_digits_stay_readable_inside_the_selected_region(self):
        for number in (LIGHT, DARK):
            out = render_frame(self.a, self.r, number, set())
            digits = (self.a.labels == number) & self.a.ink & ~self.outline(out)
            self.assertGreater(digits.sum(), 50)
            contrast = np.abs(lum(out[digits]) - lum(self.a.palette[number]))
            self.assertGreater(np.median(contrast), 100)

    def test_digits_remain_visible_in_dimmed_areas(self):
        out = render_frame(self.a, self.r, 4, set())
        digits = (self.a.labels == 5) & self.a.ink
        bg = lum(out[self.fill(5)]).mean()
        self.assertGreater(np.abs(lum(out[digits]) - bg).mean(), 25)

    def test_the_focused_region_gets_its_own_outline_inside_the_region(self):
        rid = self.cells(4)[3]
        out = render_frame(self.a, self.r, 4, set(), focus=rid)
        focus = np.all(out == FOCUS_COLOR, axis=2)
        self.assertGreater(focus.sum(), 100)
        self.assertFalse((focus & (self.r.ids != rid)).any())
        # the other unpainted regions keep the normal outline
        self.assertTrue(self.outline(out)[self.r.ids == self.cells(4)[0]].any())
        self.assertFalse(self.outline(out)[self.r.ids == rid].any())

    def test_does_not_modify_its_inputs(self):
        before = self.a.canvas.copy(), self.a.labels.copy(), self.r.ids.copy()
        painted = {1, 2}
        render_frame(self.a, self.r, 4, painted, focus=3)
        self.assertTrue((before[0] == self.a.canvas).all())
        self.assertTrue((before[1] == self.a.labels).all())
        self.assertTrue((before[2] == self.r.ids).all())
        self.assertEqual(painted, {1, 2})

    def test_output_matches_canvas_shape_and_type(self):
        out = render_frame(self.a, self.r, 0, set())
        self.assertEqual(out.shape, self.a.canvas.shape)
        self.assertEqual(out.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
