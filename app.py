"""
Paint-By-Number Smart Assistant
Run:  streamlit run app.py

Upload a printed canvas that includes its numbered color palette (see data/test_image.jpg).
The app reads the palette, maps every canvas region to its printed number, and highlights what
is left to paint for one number at a time on a zoomable view. Progress is tracked per region
and saved, so you can close the app and pick up where you left off.
"""
import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from pbn.canvas import analyze
from pbn.painting import PaintState
from pbn.palette import PaletteNotFound
from pbn.progress import ProgressStore, marks_from_painted, painted_from_marks
from pbn.regions import build_regions
from pbn.render import render_frame
from pbn.viewer import viewer

st.set_page_config(page_title="Paint-By-Number Assistant", layout="wide")

STORE = ProgressStore(os.environ.get("PBN_PROGRESS_DIR")
                      or Path(__file__).resolve().parent / "progress")
SORTS = ["Number", "Largest area", "Most left"]


@st.cache_data(show_spinner="Reading the palette and mapping canvas regions...", max_entries=2)
def load(file_bytes: bytes, max_side: int):
    bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("That file could not be decoded as an image.")
    analysis = analyze(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), max_side)
    return analysis, build_regions(analysis.labels, len(analysis.palette))


# --------------------------------------------------------------------------
# State and actions (callbacks run before the script body on the next rerun)
# --------------------------------------------------------------------------
for key, default in (("selected", 0), ("last_click", None), ("sig", None), ("focus", None),
                     ("focus_n", 0), ("cursor", {}), ("flash", None)):
    st.session_state.setdefault(key, default)


def persist():
    ps = st.session_state.paint
    STORE.save(st.session_state.file_key,
               marks_from_painted(ps.regions, ps.painted, ps.regions.ids.shape))


def select_color(i):
    st.session_state.selected = i
    st.session_state.focus = None


def toggle_region(rid):
    ps = st.session_state.paint
    ps.toggle(rid)
    if rid in ps.painted:
        st.session_state.cursor[st.session_state.selected] = rid  # Next continues from here
    if st.session_state.focus and st.session_state.focus["rid"] == rid:
        st.session_state.focus = None
    persist()


def mark_selected_done():
    st.session_state.paint.set_color(st.session_state.selected, True)
    st.session_state.focus = None
    persist()


def set_color_done(i, was_done):
    st.session_state.paint.set_color(i, not was_done)
    st.session_state.focus = None
    persist()


def undo():
    st.session_state.paint.undo()
    st.session_state.focus = None
    persist()


def next_region():
    sel = st.session_state.selected
    rid = st.session_state.paint.next_unpainted(sel, st.session_state.cursor.get(sel))
    if rid is None:
        st.session_state.focus = None
        st.session_state.flash = f"Every region of #{sel + 1} is painted. 🎉"
        return
    st.session_state.cursor[sel] = rid
    st.session_state.focus_n += 1
    st.session_state.focus = {"rid": rid, "n": st.session_state.focus_n}


def reset_progress():
    st.session_state.paint = PaintState(st.session_state.paint.regions)
    st.session_state.focus = None
    st.session_state.cursor = {}
    persist()


# --------------------------------------------------------------------------
# Sidebar: upload + settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Canvas")
    canvas_file = st.file_uploader("Printed canvas with its color palette",
                                   type=["jpg", "jpeg", "png"])

    st.header("2. Settings")
    max_side = st.select_slider("Processing size (max px)", [1000, 1400, 2000, 2600, 3200], 2000,
                                help="Higher resolves smaller cells and keeps them sharp when "
                                     "zoomed in, but is slower.")
    view_height = st.slider("Viewer height (px)", 500, 1400, 800, 50)

if canvas_file is None:
    st.title("🎨 Paint-By-Number Smart Assistant")
    st.info("Upload your printed canvas in the sidebar. It must include the numbered color "
            "palette strip, so colors can be matched to the numbers printed on your kit.")
    st.stop()

file_bytes = canvas_file.getvalue()
try:
    a, regions = load(file_bytes, max_side)
except PaletteNotFound as err:
    st.error(f"Could not find the numbered color palette on this image. {err}")
    st.stop()
except ValueError as err:
    st.error(str(err))
    st.stop()

n_colors = len(a.palette)

# New image or processing size: start from the progress saved for this image.
file_key = hashlib.md5(file_bytes).hexdigest()
sig = f"{file_key}:{max_side}"
if st.session_state.sig != sig:
    marks = STORE.load(file_key)
    painted, dropped = painted_from_marks(regions, marks, a.labels.shape)
    st.session_state.paint = PaintState(regions, painted)
    st.session_state.update(sig=sig, file_key=file_key, selected=0, last_click=None,
                            focus=None, cursor={})
    if marks:
        st.session_state.flash = (f"Restored {len(painted)} painted regions."
                                  + (f" {dropped} could not be matched at this size." if dropped else ""))

ps: PaintState = st.session_state.paint
if st.session_state.selected >= n_colors:
    st.session_state.selected = 0
sel = st.session_state.selected

if st.session_state.flash:
    st.toast(st.session_state.flash)
    st.session_state.flash = None

# --------------------------------------------------------------------------
# Sidebar: palette (numbers match the strip printed on the canvas)
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("3. Palette")
    done_colors = sum(ps.is_done(i) for i in range(n_colors))
    st.progress(min(ps.painted_fraction(), 1.0),
                text=f"Canvas painted: {ps.painted_fraction() * 100:.1f}% "
                     f"({done_colors}/{n_colors} colors done)")

    f1, f2 = st.columns([3, 2], vertical_alignment="bottom")
    sort_by = f1.selectbox("Sort by", SORTS, key="sort_by")
    hide_done = f2.toggle("Hide done", key="hide_done")

    order = list(range(n_colors))
    if sort_by == "Largest area":
        order.sort(key=lambda i: -a.area[i])
    elif sort_by == "Most left":
        order.sort(key=lambda i: -ps.left(i))
    if hide_done:
        order = [i for i in order if not ps.is_done(i) or i == sel]

    box = st.container(height=440)
    with box:
        for i in order:
            r, g, b = (int(v) for v in a.palette[i])
            done = ps.is_done(i)
            left = ps.left(i)
            c1, c2, c3 = st.columns([1, 4, 1], vertical_alignment="center")
            c1.markdown(
                f"<div style='width:28px;height:28px;border-radius:6px;"
                f"background:rgb({r},{g},{b});opacity:{0.3 if done else 1};"
                f"border:{'3px solid #ff2878' if i == sel else '1px solid #888'}'></div>",
                unsafe_allow_html=True)
            label = (f"{'✅ ' if done else ''}{'⚠️ ' if not a.matched[i] else ''}"
                     f"#{i + 1} · {a.area[i] * 100:.1f}% · "
                     f"{'done' if done else f'{left} left'}")
            c2.button(label, key=f"pick_{i}", on_click=select_color, args=(i,),
                      use_container_width=True, type="primary" if i == sel else "secondary",
                      help=None if a.matched[i] else "No flat fill matching this number was "
                                                     "found on the canvas.")
            # The key carries the derived state, so the box is rebuilt when regions change it.
            c3.checkbox("done", value=done, key=f"done_{i}_{int(done)}",
                        on_change=set_color_done, args=(i, done),
                        label_visibility="collapsed",
                        help="Mark every region of this color as painted (untick to undo)")

    st.button("Reset progress", on_click=reset_progress, use_container_width=True)

    with st.expander("Detection details"):
        st.caption(f"{n_colors} colors read from the palette strip · "
                   f"{int(a.matched.sum())} found on the canvas · "
                   f"print fade {a.fade * 100:.0f}% · "
                   f"canvas {a.box[2] - a.box[0]}×{a.box[3] - a.box[1]} px "
                   f"→ {a.canvas.shape[1]}×{a.canvas.shape[0]} px · "
                   f"{regions.count} regions tracked. Cells thinner than ~3 px at this size "
                   f"are not tracked; raise Processing size to resolve smaller cells.")

# --------------------------------------------------------------------------
# Main panel
# --------------------------------------------------------------------------
st.title("🎨 Paint-By-Number Smart Assistant")

r, g, b = (int(v) for v in a.palette[sel])
left, total = ps.left(sel), ps.total(sel)
info = st.columns([1, 4])
info[0].markdown(
    f"<div style='width:56px;height:56px;border-radius:10px;background:rgb({r},{g},{b});"
    f"border:1px solid #888'></div>", unsafe_allow_html=True)
info[1].markdown(f"### Color #{sel + 1}  \n`RGB({r}, {g}, {b})` · "
                 f"{a.area[sel] * 100:.1f}% of canvas · "
                 f"{'✅ all' if left == 0 else left} of {total} regions "
                 f"{'painted' if left == 0 else 'left'}")

actions = st.columns(3)
actions[0].button("⏭ Next unpainted region", key="next_region", on_click=next_region,
                  use_container_width=True, type="primary", disabled=left == 0)
actions[1].button("✅ Mark this color done", key="mark_done", on_click=mark_selected_done,
                  use_container_width=True, disabled=left == 0)
actions[2].button("↩ Undo last mark", key="undo", on_click=undo, use_container_width=True,
                  disabled=not ps.can_undo)

# The jump target is only meaningful while it is still unpainted and of the selected color.
focus = st.session_state.focus
if focus and (focus["rid"] in ps.painted or int(regions.label[focus["rid"]]) != sel):
    focus = None
focus_payload = None
if focus:
    rid = focus["rid"]
    py, px = (int(v) for v in regions.pole[rid])
    focus_payload = {"x": px + 0.5, "y": py + 0.5, "w": int(regions.bbox[rid][2]),
                     "h": int(regions.bbox[rid][3]), "n": focus["n"]}

st.caption("Click a highlighted region to mark it painted (click again to undo). "
           "Click any other color to switch to it. Scroll to zoom, drag to pan.")
frame = render_frame(a, regions, sel, ps.painted, focus=focus["rid"] if focus else None)
click = viewer(frame, view_id=sig, height=view_height, focus=focus_payload)

if click and click.get("view") == sig and click != st.session_state.last_click:
    st.session_state.last_click = click
    h, w = a.labels.shape
    x, y = min(max(int(click["x"]), 0), w - 1), min(max(int(click["y"]), 0), h - 1)
    clicked_color, clicked_region = int(a.labels[y, x]), int(regions.ids[y, x])
    if clicked_color != sel:
        select_color(clicked_color)
        st.rerun()
    elif clicked_region > 0:
        toggle_region(clicked_region)
        st.rerun()
