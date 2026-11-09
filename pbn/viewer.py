"""Zoomable, pannable, clickable canvas viewer (a local Streamlit component, no build step).

Zoom and pan happen entirely in the browser, so they never trigger a Streamlit rerun. Only a
click (not a drag) is sent back, as image-pixel coordinates.
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import streamlit.components.v1 as components

_FRONTEND = Path(__file__).resolve().parent.parent / "components" / "viewer"
_component = components.declare_component("pbn_viewer", path=str(_FRONTEND))


def _data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise ValueError("Could not encode the canvas frame.")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def viewer(frame: np.ndarray, view_id: str, height: int = 800, focus=None,
           key: str = "canvas_viewer"):
    """Show ``frame``; returns the latest click as ``{"x", "y", "n", "view"}`` or ``None``.

    ``view_id`` identifies the image being worked on: when it changes the view re-fits,
    otherwise a new ``frame`` keeps the user's current zoom and pan.

    ``focus`` is ``{"x", "y", "w", "h", "n"}`` (a region's centre and size in image pixels):
    the view jumps to it once per distinct ``n``.
    """
    h, w = frame.shape[:2]
    return _component(src=_data_url(frame), img_w=w, img_h=h, view_id=view_id,
                      height=height, focus=focus, key=key, default=None)
