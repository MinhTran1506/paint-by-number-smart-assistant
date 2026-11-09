import unittest

import numpy as np

from pbn.canvas import analyze
from pbn.regions import build_regions, clean_labels
from tests.synthetic import CELL, COLS, ROWS, cell_label, make_canvas

BIG = 5000


class BuildRegionsOnTinyArraysTests(unittest.TestCase):
    def test_each_connected_blob_of_a_colour_is_its_own_region(self):
        labels = np.zeros((6, 12), np.uint8)
        labels[:, 4:8] = 1                       # a vertical bar of colour 1 splits colour 0 in two
        r = build_regions(labels, 2, min_area=1)
        self.assertEqual(r.count, 3)
        self.assertEqual(sorted(r.label[1:].tolist()), [0, 0, 1])

    def test_diagonal_neighbours_are_one_region(self):
        labels = np.zeros((4, 4), np.uint8)
        labels[0, 0] = labels[1, 1] = labels[2, 2] = 1
        r = build_regions(labels, 2, min_area=1)
        self.assertEqual(int((r.label[1:] == 1).sum()), 1)

    def test_specks_below_min_area_are_not_regions(self):
        labels = np.zeros((10, 10), np.uint8)
        labels[:, 5:] = 1
        labels[2, 2] = 1                          # 1px speck of colour 1 inside colour 0
        r = build_regions(labels, 2, min_area=3)
        self.assertEqual(r.count, 2)
        self.assertEqual(r.ids[2, 2], 0)

    def test_regions_of_a_colour_come_back_in_reading_order(self):
        labels = np.zeros((9, 9), np.uint8)
        labels[1, 6:8] = 1; labels[1, 1:3] = 1; labels[5, 3:5] = 1; labels[3, 4:6] = 1
        r = build_regions(labels, 2, min_area=1)
        anchors = [(int(r.anchor[i][0]), int(r.anchor[i][1])) for i in r.of_label(1)]
        self.assertEqual(anchors, [(1, 1), (1, 6), (3, 4), (5, 3)])

    def test_anchor_is_the_first_pixel_in_raster_order(self):
        labels = np.zeros((6, 6), np.uint8)
        labels[2, 3] = labels[3, 1] = labels[3, 2] = labels[3, 3] = 1
        r = build_regions(labels, 2, min_area=1)
        rid = r.of_label(1)[0]
        self.assertEqual(tuple(int(v) for v in r.anchor[rid]), (2, 3))


class ThinRegionTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.zeros((20, 30), np.uint8)
        self.labels[:, 15:] = 1
        self.labels[5:15, 3:5] = 2    # a 2px-wide stripe: a seam, not a paintable cell
        self.labels[5:15, 20:28] = 2  # a proper 8px-wide cell

    def test_thin_regions_are_dropped_when_a_minimum_depth_is_asked_for(self):
        kept = build_regions(self.labels, 3, min_area=1, min_depth=1.0)
        self.assertEqual(int((kept.label[1:] == 2).sum()), 1)
        self.assertEqual(kept.ids[10, 3], 0)

    def test_by_default_nothing_thin_is_dropped(self):
        self.assertEqual(int((build_regions(self.labels, 3, min_area=1).label[1:] == 2).sum()), 2)

    def test_clean_labels_hands_thin_pixels_to_the_surrounding_region(self):
        cleaned = clean_labels(self.labels, 3, min_depth=1.0)
        self.assertEqual(cleaned[10, 3], 0)       # the stripe is absorbed by its surroundings
        self.assertEqual(cleaned[10, 24], 2)      # the real cell is untouched
        self.assertEqual(cleaned.shape, self.labels.shape)

    def test_clean_labels_leaves_a_clean_map_alone(self):
        clean = np.zeros((20, 30), np.uint8)
        clean[:, 15:] = 1
        self.assertTrue((clean_labels(clean, 2) == clean).all())

    def test_downscaled_sheet_does_not_grow_phantom_seam_regions(self):
        sheet = make_canvas(scale=3)
        a = analyze(sheet.image, 3000)  # mild downscale: blended seams land between fills
        r = build_regions(a.labels, len(a.palette))
        self.assertGreaterEqual(r.count, ROWS * COLS)
        self.assertLessEqual(r.count, int(ROWS * COLS * 1.12))  # was 479 before cleaning


class BuildRegionsOnTheSyntheticSheetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = analyze(make_canvas().image, BIG)
        cls.r = build_regions(cls.a.labels, len(cls.a.palette))

    def cell_center(self, row, col):
        return row * CELL + CELL // 2, col * CELL + CELL // 2

    def test_every_printed_cell_is_one_region_of_the_right_colour(self):
        self.assertEqual(self.r.count, ROWS * COLS)
        for row, col in [(0, 0), (3, 7), (ROWS - 1, COLS - 1)]:
            y, x = self.cell_center(row, col)
            rid = self.r.ids[y, x]
            self.assertGreater(rid, 0)
            self.assertEqual(self.r.label[rid], cell_label(row, col))

    def test_region_areas_match_the_cells(self):
        areas = self.r.area[1:]
        self.assertTrue(((areas > CELL * CELL * 0.9) & (areas < CELL * CELL * 1.1)).all())

    def test_pole_is_well_inside_the_region(self):
        for rid in (1, 50, self.r.count):
            py, px = (int(v) for v in self.r.pole[rid])
            self.assertEqual(self.r.ids[py, px], rid)
            y0, x0, bh, bw = self.r.bbox[rid][1], self.r.bbox[rid][0], self.r.bbox[rid][3], self.r.bbox[rid][2]
            self.assertGreaterEqual(min(py - y0, y0 + bh - 1 - py, px - x0, x0 + bw - 1 - px), 20)

    def test_every_pixel_of_a_region_has_that_regions_colour(self):
        self.assertTrue((self.r.label[self.r.ids] == self.a.labels)[self.r.ids > 0].all())

    def test_each_colours_regions_are_listed_in_reading_order(self):
        for lab in range(len(self.a.palette)):
            keys = [int(self.r.anchor[i][0]) * self.a.labels.shape[1] + int(self.r.anchor[i][1])
                    for i in self.r.of_label(lab)]
            self.assertEqual(keys, sorted(keys))
            self.assertGreater(len(keys), 0)


if __name__ == "__main__":
    unittest.main()
