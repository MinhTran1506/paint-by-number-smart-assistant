# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A Streamlit app that helps someone paint a paint-by-number kit. The user uploads one image of the **printed canvas including its numbered color-palette strip** (see `data/test_image.jpg`: ~5500×6700 px, 36 swatches). The app reads the palette, maps every canvas region to its printed number, and highlights what is left to paint for one number at a time on a zoomable view. Progress is tracked **per region** (click a region to mark it painted), saved to disk, and a *Next unpainted region* button jumps to the next one.

## Commands

```powershell
.venv\Scripts\Activate.ps1
streamlit run app.py

python -m unittest discover -s tests -t .      # Python tests (run from repo root)
python -m unittest tests.test_canvas           # one module
python -m unittest tests.test_canvas.AnalyzeTests.test_fade_is_recovered   # one test
node --test tests/viewer.test.js               # viewer JS logic (stub DOM, Node 22+)

# Tests that run app.py (tests/test_app_flow.py) take ~40s; set PBN_PROGRESS_DIR to redirect saved progress
```

Tests use stdlib `unittest` (pytest is not installed in the venv). `tests/test_real_image.py` and `tests/test_app_flow.py` run on `data/test_image.jpg` and skip if it is absent. Dependencies are in `requirements.txt`. Do not run Python with `-I` here, since the `pbn` package is imported from the repo root.

## Architecture

The image logic lives in the `pbn/` package so it is testable without Streamlit; `app.py` is only UI and session state.

- **`pbn/palette.py`**: `find_palette()` locates the swatch column (largest group of equal-size, left-aligned solid blocks after a morphological open that removes digits/hairlines/frames). Swatch *i* from the top **is** printed number *i + 1*; no OCR. It samples on a regular grid, so a near-white swatch that is not detected as a block still keeps the numbering from shifting. Raises `PaletteNotFound` otherwise.
- **`pbn/canvas.py`**: `analyze()` is the pipeline. It finds the canvas box, downsizes to `max_side`, and then relies on the key insight: **printed fills are the swatch colors faded toward white**, `canvas = 255 - s·(255 - swatch)`. It finds the dominant flat colors on the canvas, fits the single opacity `s` by one-to-one (Hungarian) matching against the swatches (this is what separates near-identical dark swatches), and labels each pixel with the nearest matched fill color. Pixels that match no fill are `ink` (outlines, printed digits); ink plus a 1px halo is relabeled from the nearest trustworthy pixel, because anti-aliased ink/fill blends can mimic a *different* fill color. Returns an `Analysis` (labels, ink mask, palette, per-number area, `matched` flags, fitted fade).
- **`pbn/regions.py`**: `build_regions()` splits each color into 8-connected regions with stable ids (0 = none) and per-region `anchor` (first pixel in reading order, used for ordering), `bbox` and `pole` (deepest interior pixel, used for focusing and for saving progress). `clean_labels()` runs inside `analyze()`: resizing blends neighboring fills, and a blend can mimic a *third* fill color, leaving hairline phantom regions. Regions that never get deeper than ~1 px (≈3 px wide) are handed to the nearest real region. Real tiny cells need a larger Processing size to be resolved.
- **`pbn/painting.py`**: `PaintState` is the plain-Python progress model: `toggle`, `set_color`, `undo`, `left`/`is_done` per color, area-weighted `painted_fraction`, and `next_unpainted(color, after)` (reading order, wraps, works from a just-painted region).
- **`pbn/progress.py`**: `ProgressStore` saves/loads `progress/<md5 of image bytes>.json` (atomic write; corrupt or unknown-version files are ignored). Painted regions are stored as `(fx, fy, color)`: the region's pole as fractions of the canvas plus its color, **not** region ids, so progress survives a change of Processing size.
- **`pbn/render.py`**: `render_frame(a, regions, selected, painted, focus)` draws the selected number's *unpainted* regions in the true swatch color with an edge outline, dims everything else to a neutral grey whose brightness contrasts with the swatch (dark backdrop behind light swatches and vice versa), softly tints painted regions of any color, outlines the `focus` region in yellow, and keeps `ink` pixels visible but recolored for contrast.
- **`pbn/viewer.py` + `components/viewer/index.html`**: a local Streamlit component (plain JS, no build step) for wheel zoom toward the cursor, drag pan, +/−/fit buttons, double-click to reset. Zoom/pan are client-side only (no rerun). A click (not a drag) returns image-pixel coordinates `{x, y, n, view}`. A `focus` arg `{x, y, w, h, n}` makes it center/zoom on a region once per distinct `n` (re-renders with the same `n` leave the user's view alone). A new `frame` with the same `view_id` swaps the picture but keeps zoom/pan; a new `view_id` re-fits.
- **`app.py`**: `load()` is `@st.cache_data` keyed by `(file bytes, max_side)` and returns `(Analysis, Regions)`. A signature `sig = md5(file):max_side` resets selection/focus and reloads saved progress when it changes; the `PaintState` lives in `st.session_state.paint` and every change calls `persist()`. Click semantics: clicking a region of the *selected* color toggles it painted, clicking another color switches to it. Clicks are applied only if their `view` equals the current `sig` and they differ from `last_click` (component values persist across reruns). The palette `done` checkbox key embeds the derived state (`done_<i>_<0|1>`) so it rebuilds when region clicks change that state.

Coordinates throughout are in the **processed canvas space** (cropped to the canvas box, then resized), not the uploaded image.

## Test fixtures

`tests/test_app_flow.py` swaps the browser viewer for a stand-in (returns scripted clicks, records the `focus` the app sends). `tests/synthetic.py` builds a fake sheet (page frame, logo, swatch strip with three near-identical dark swatches, faded cell fills, ink lines and digits). Compare labels only on cell *interiors* (`interior()` in `tests/test_canvas.py`); a pixel on a cell border legitimately belongs to either neighbor.
