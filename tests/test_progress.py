import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pbn.canvas import analyze
from pbn.progress import ProgressStore, marks_from_painted, painted_from_marks
from pbn.regions import build_regions
from tests.synthetic import ROWS, make_canvas

KEY = "0123456789abcdef0123456789abcdef"


class ProgressStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = ProgressStore(Path(self._tmp.name) / "progress")

    def tearDown(self):
        self._tmp.cleanup()

    def test_saved_marks_come_back(self):
        marks = [(0.25, 0.5, 3), (0.75, 0.125, 11)]
        self.store.save(KEY, marks)
        self.assertEqual(self.store.load(KEY), marks)

    def test_unknown_image_has_no_marks(self):
        self.assertEqual(self.store.load(KEY), [])

    def test_corrupt_file_is_ignored_not_fatal(self):
        self.store.save(KEY, [(0.1, 0.1, 1)])
        self.store.path(KEY).write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.load(KEY), [])

    def test_file_with_an_unknown_version_is_ignored(self):
        self.store.save(KEY, [(0.1, 0.1, 1)])
        data = json.loads(self.store.path(KEY).read_text(encoding="utf-8"))
        data["version"] = 99
        self.store.path(KEY).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load(KEY), [])

    def test_malformed_entries_are_skipped(self):
        self.store.save(KEY, [(0.1, 0.1, 1)])
        data = json.loads(self.store.path(KEY).read_text(encoding="utf-8"))
        data["painted"] += [["x", 1, 2], [0.5], "junk", [2.0, 0.5, 1]]
        self.store.path(KEY).write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.store.load(KEY), [(0.1, 0.1, 1)])

    def test_save_leaves_no_temporary_files_and_overwrites(self):
        self.store.save(KEY, [(0.1, 0.1, 1)])
        self.store.save(KEY, [(0.2, 0.2, 2)])
        self.assertEqual(self.store.load(KEY), [(0.2, 0.2, 2)])
        self.assertEqual([p.name for p in self.store.directory.iterdir()], [f"{KEY}.json"])

    def test_keys_cannot_escape_the_directory(self):
        with self.assertRaises(ValueError):
            self.store.path("../evil")


class MarkConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sheet = make_canvas(scale=3)
        cls.big = analyze(sheet.image, 3000)     # 1:1 with the 3x sheet
        cls.small = analyze(sheet.image, 700)    # a much coarser processing size
        cls.big_r = build_regions(cls.big.labels, len(cls.big.palette))
        cls.small_r = build_regions(cls.small.labels, len(cls.small.palette))

    def test_painted_regions_round_trip_at_the_same_size(self):
        painted = {1, 20, 77, self.big_r.count}
        marks = marks_from_painted(self.big_r, painted, self.big.labels.shape)
        got, dropped = painted_from_marks(self.big_r, marks, self.big.labels.shape)
        self.assertEqual(got, painted)
        self.assertEqual(dropped, 0)

    def test_progress_survives_a_change_of_processing_size(self):
        cells = [i for i in range(1, self.big_r.count + 1)
                 if self.big_r.area[i] > 0.8 * (self.big.labels.shape[0] / ROWS) ** 2]
        painted = set(cells[::5])
        marks = marks_from_painted(self.big_r, painted, self.big.labels.shape)
        got, dropped = painted_from_marks(self.small_r, marks, self.small.labels.shape)
        self.assertEqual(dropped, 0)
        self.assertEqual(len(got), len(painted))
        # the restored regions are the same physical cells: same colour and place
        for rid in got:
            self.assertIn(int(self.small_r.label[rid]), range(len(self.small.palette)))
        back = marks_from_painted(self.small_r, got, self.small.labels.shape)
        again, _ = painted_from_marks(self.big_r, back, self.big.labels.shape)
        self.assertEqual(again, painted)

    def test_marks_with_the_wrong_colour_are_dropped_and_counted(self):
        marks = marks_from_painted(self.big_r, {1, 2, 3}, self.big.labels.shape)
        wrong = [(fx, fy, (lab + 1) % len(self.big.palette)) for fx, fy, lab in marks]
        got, dropped = painted_from_marks(self.big_r, wrong, self.big.labels.shape)
        self.assertEqual(got, set())
        self.assertEqual(dropped, 3)

    def test_marks_are_plain_fractions_and_a_colour_index(self):
        (fx, fy, lab), = marks_from_painted(self.big_r, {5}, self.big.labels.shape)
        self.assertTrue(0 < fx < 1 and 0 < fy < 1)
        self.assertEqual(lab, int(self.big_r.label[5]))


if __name__ == "__main__":
    unittest.main()
