"""Drive the real app.py on data/test_image.jpg through Streamlit's headless test harness.

The harness feeds the file uploader, and swaps the browser viewer for a stand-in that returns
scripted clicks and records the focus requests the app sends it. Real browser behaviour
(wheel, drag) is covered separately by tests/viewer.test.js.
"""
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

from pbn.canvas import analyze
from pbn.painting import PaintState
from pbn.regions import build_regions

ROOT = Path(__file__).resolve().parent.parent
IMAGE = ROOT / "data" / "test_image.jpg"

HARNESS = f'''
import io, pathlib, streamlit as st
import pbn.viewer
IMG = pathlib.Path(r"{IMAGE}").read_bytes()
st.file_uploader = lambda *a, **k: io.BytesIO(IMG)
def fake_viewer(frame, view_id, height=800, focus=None, key="canvas_viewer"):
    st.session_state["_seen"] = {{"focus": focus, "shape": frame.shape}}
    c = st.session_state.get("_click")
    return None if c is None else {{"x": c[0], "y": c[1], "n": c[2], "view": view_id}}
pbn.viewer.viewer = fake_viewer
exec(compile(pathlib.Path(r"{ROOT / 'app.py'}").read_text(encoding="utf-8"), "app.py", "exec"))
'''


def new_app():
    return AppTest.from_string(HARNESS, default_timeout=300)


@unittest.skipUnless(IMAGE.exists(), "data/test_image.jpg not available")
class AppFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bgr = cv2.imdecode(np.fromfile(str(IMAGE), np.uint8), cv2.IMREAD_COLOR)
        cls.a = analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 2000)   # the app's default size
        cls.r = build_regions(cls.a.labels, len(cls.a.palette))

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.environ.get("PBN_PROGRESS_DIR")
        os.environ["PBN_PROGRESS_DIR"] = self._tmp.name
        self.at = new_app().run()
        self.assertFalse(self.at.exception, [e.value for e in self.at.exception])

    def tearDown(self):
        if self._old is None:
            os.environ.pop("PBN_PROGRESS_DIR", None)
        else:
            os.environ["PBN_PROGRESS_DIR"] = self._old
        self._tmp.cleanup()

    # -- helpers --------------------------------------------------------------------------
    def paint(self, at=None):
        return (at or self.at).session_state["paint"]

    def big_regions(self, color, n=3):
        ids = sorted(self.r.of_label(color).tolist(), key=lambda i: -self.r.area[i])
        return ids[:n]

    def click(self, rid, n, at=None):
        at = at or self.at
        py, px = (int(v) for v in self.r.pole[rid])
        at.session_state["_click"] = (px, py, n)
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])

    def button(self, key, at=None):
        return (at or self.at).button(key=key)

    def saved_files(self):
        return sorted(Path(self._tmp.name).glob("*.json"))

    # -- tests ----------------------------------------------------------------------------
    def test_starts_with_the_whole_printed_palette_and_nothing_painted(self):
        picks = [b for b in self.at.button if b.key and b.key.startswith("pick_")]
        self.assertEqual(len(picks), 36)
        self.assertEqual(self.paint().painted, set())
        self.assertIn("left", picks[0].label)

    def test_clicking_a_region_of_the_selected_colour_paints_it_and_saves_progress(self):
        rid = self.big_regions(0)[0]
        self.click(rid, 1)
        self.assertEqual(self.paint().painted, {rid})
        self.assertEqual(len(self.saved_files()), 1)

    def test_clicking_the_same_region_again_unpaints_it(self):
        rid = self.big_regions(0)[0]
        self.click(rid, 1)
        self.click(rid, 2)
        self.assertEqual(self.paint().painted, set())

    def test_clicking_a_region_of_another_colour_selects_that_colour_without_painting(self):
        rid = self.big_regions(5)[0]
        self.click(rid, 1)
        self.assertEqual(self.at.session_state["selected"], 5)
        self.assertEqual(self.paint().painted, set())

    def test_progress_comes_back_in_a_new_session(self):
        first, second = self.big_regions(0)[:2]
        self.click(first, 1)
        self.click(second, 2)
        fresh = new_app().run()
        self.assertFalse(fresh.exception)
        self.assertEqual(self.paint(fresh).painted, {first, second})

    def test_next_jumps_to_unpainted_regions_of_the_colour_in_reading_order(self):
        order = self.r.of_label(0).tolist()
        self.button("next_region").click().run()
        seen = self.at.session_state["_seen"]["focus"]
        py, px = (int(v) for v in self.r.pole[order[0]])
        self.assertEqual((int(seen["x"]), int(seen["y"])), (px, py))
        self.assertEqual(seen["w"], int(self.r.bbox[order[0]][2]))
        first_n = seen["n"]

        self.button("next_region").click().run()
        seen = self.at.session_state["_seen"]["focus"]
        py, px = (int(v) for v in self.r.pole[order[1]])
        self.assertEqual((int(seen["x"]), int(seen["y"])), (px, py))
        self.assertNotEqual(seen["n"], first_n)

    def test_next_skips_regions_that_are_already_painted(self):
        order = self.r.of_label(0).tolist()
        self.click(order[0], 1)
        self.button("next_region").click().run()
        seen = self.at.session_state["_seen"]["focus"]
        py, px = (int(v) for v in self.r.pole[order[1]])
        self.assertEqual((int(seen["x"]), int(seen["y"])), (px, py))

    def test_painting_the_focused_region_clears_the_focus_and_next_moves_on(self):
        order = self.r.of_label(0).tolist()
        self.button("next_region").click().run()
        self.click(order[0], 1)
        self.assertIsNone(self.at.session_state["_seen"]["focus"])
        self.button("next_region").click().run()
        seen = self.at.session_state["_seen"]["focus"]
        py, px = (int(v) for v in self.r.pole[order[1]])
        self.assertEqual((int(seen["x"]), int(seen["y"])), (px, py))

    def test_undo_takes_back_the_last_mark(self):
        rid = self.big_regions(0)[0]
        self.click(rid, 1)
        self.button("undo").click().run()
        self.assertEqual(self.paint().painted, set())
        self.assertTrue(self.button("undo").disabled)

    def test_marking_a_colour_done_paints_all_of_it_and_hides_it_on_request(self):
        self.button("mark_done").click().run()
        ps = self.paint()
        self.assertTrue(ps.is_done(0))
        self.assertEqual(ps.left(0), 0)
        self.assertTrue(self.button("next_region").disabled)
        self.at.toggle(key="hide_done").set_value(True).run()
        picks = lambda: [b.key for b in self.at.button if b.key and b.key.startswith("pick_")]
        self.assertIn("pick_0", picks())            # the colour you are on never vanishes
        self.button("pick_1").click().run()
        self.assertNotIn("pick_0", picks())         # once you move on, the finished one hides
        self.assertEqual(len(picks()), 35)

    def test_palette_can_be_sorted_by_what_is_left(self):
        self.at.selectbox(key="sort_by").set_value("Most left").run()
        keys = [int(b.key.split("_")[1]) for b in self.at.button if b.key and b.key.startswith("pick_")]
        lefts = [len(self.r.of_label(i)) for i in keys]
        self.assertEqual(lefts, sorted(lefts, reverse=True))

    def test_ticking_a_colours_done_box_paints_every_region_of_it(self):
        self.at.checkbox(key="done_3_0").check().run()
        self.assertTrue(self.paint().is_done(3))
        self.at.checkbox(key="done_3_1").uncheck().run()
        self.assertEqual(self.paint().left(3), len(self.r.of_label(3)))


if __name__ == "__main__":
    unittest.main()
