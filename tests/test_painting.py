import unittest

import numpy as np

from pbn.painting import PaintState
from pbn.regions import build_regions


def make_regions():
    """Colour 0 = left half, colour 1 = three separate blobs in reading order, 4x2 each."""
    labels = np.zeros((10, 40), np.uint8)
    for x0 in (4, 14, 24):
        labels[2:6, x0:x0 + 4] = 1
    labels[7:9, 30:36] = 1                       # a fourth blob, further down
    return labels, build_regions(labels, 2, min_area=1)


class PaintStateTests(unittest.TestCase):
    def setUp(self):
        self.labels, self.r = make_regions()
        self.ps = PaintState(self.r)
        self.blobs = list(self.r.of_label(1))   # reading order

    def test_toggle_marks_and_unmarks_a_region(self):
        self.ps.toggle(self.blobs[0])
        self.assertIn(self.blobs[0], self.ps.painted)
        self.ps.toggle(self.blobs[0])
        self.assertNotIn(self.blobs[0], self.ps.painted)

    def test_toggling_the_no_region_id_is_ignored(self):
        self.ps.toggle(0)
        self.assertEqual(self.ps.painted, set())

    def test_left_counts_unpainted_regions_of_a_colour(self):
        self.assertEqual(self.ps.left(1), 4)
        self.ps.toggle(self.blobs[1])
        self.assertEqual(self.ps.left(1), 3)
        self.assertFalse(self.ps.is_done(1))

    def test_a_colour_is_done_when_nothing_is_left(self):
        for rid in self.blobs:
            self.ps.toggle(rid)
        self.assertTrue(self.ps.is_done(1))
        self.assertFalse(self.ps.is_done(0))

    def test_set_color_paints_and_unpaints_every_region(self):
        self.ps.toggle(self.blobs[0])
        self.ps.set_color(1, True)
        self.assertEqual(self.ps.left(1), 0)
        self.ps.set_color(1, False)
        self.assertEqual(self.ps.left(1), 4)

    def test_undo_reverses_the_last_action_step_by_step(self):
        self.ps.toggle(self.blobs[0])
        self.ps.set_color(1, True)
        self.assertTrue(self.ps.undo())
        self.assertEqual(self.ps.painted, {self.blobs[0]})
        self.assertTrue(self.ps.undo())
        self.assertEqual(self.ps.painted, set())
        self.assertFalse(self.ps.undo())

    def test_undo_of_a_bulk_unpaint_restores_exactly_what_was_painted(self):
        self.ps.toggle(self.blobs[2])
        self.ps.set_color(1, False)
        self.ps.undo()
        self.assertEqual(self.ps.painted, {self.blobs[2]})

    def test_painted_share_is_weighted_by_area(self):
        total = self.r.area[1:].sum()
        self.ps.toggle(self.blobs[3])            # the 6x2 blob
        self.assertAlmostEqual(self.ps.painted_fraction(), self.r.area[self.blobs[3]] / total)
        self.assertEqual(PaintState(self.r).painted_fraction(), 0.0)

    def test_next_walks_unpainted_regions_in_reading_order_and_wraps(self):
        self.assertEqual(self.ps.next_unpainted(1, after=None), self.blobs[0])
        self.assertEqual(self.ps.next_unpainted(1, after=self.blobs[0]), self.blobs[1])
        self.assertEqual(self.ps.next_unpainted(1, after=self.blobs[3]), self.blobs[0])  # wraps

    def test_next_skips_painted_regions(self):
        self.ps.toggle(self.blobs[1])
        self.assertEqual(self.ps.next_unpainted(1, after=self.blobs[0]), self.blobs[2])

    def test_next_works_from_a_region_that_has_just_been_painted(self):
        self.ps.toggle(self.blobs[1])
        self.assertEqual(self.ps.next_unpainted(1, after=self.blobs[1]), self.blobs[2])

    def test_next_is_none_when_the_colour_is_finished(self):
        self.ps.set_color(1, True)
        self.assertIsNone(self.ps.next_unpainted(1, after=None))

    def test_can_undo_reports_whether_there_is_anything_to_undo(self):
        self.assertFalse(self.ps.can_undo)
        self.ps.toggle(self.blobs[0])
        self.assertTrue(self.ps.can_undo)
        self.ps.undo()
        self.assertFalse(self.ps.can_undo)

    def test_can_start_from_previously_saved_regions(self):
        ps = PaintState(self.r, painted={self.blobs[0], self.blobs[1]})
        self.assertEqual(ps.left(1), 2)
        self.assertFalse(ps.undo())  # restored progress is not an undoable action


if __name__ == "__main__":
    unittest.main()
