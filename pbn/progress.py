"""Save and restore which regions are painted.

Regions are stored by position, as fractions of the canvas, plus their palette index, so
progress still maps onto the right cells if the processing size (and therefore the region ids)
changes between sessions.
"""
import json
import os
import re
import tempfile
from pathlib import Path

from pbn.regions import Regions

VERSION = 1
_KEY = re.compile(r"[0-9a-fA-F]{8,64}")


class ProgressStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, key: str) -> Path:
        if not _KEY.fullmatch(key):
            raise ValueError("Progress key must be a hex digest.")
        return self.directory / f"{key}.json"

    def load(self, key: str) -> list:
        """Saved marks as ``(fx, fy, colour)``; empty if there is no usable file."""
        try:
            data = json.loads(self.path(key).read_text(encoding="utf-8"))
            if data.get("version") != VERSION:
                return []
            raw = data["painted"]
        except (OSError, ValueError, KeyError, AttributeError, TypeError):
            return []
        marks = []
        for item in raw:
            try:
                fx, fy, color = item
                fx, fy, color = float(fx), float(fy), int(color)
            except (TypeError, ValueError):
                continue
            if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0 and color >= 0:
                marks.append((fx, fy, color))
        return marks

    def save(self, key: str, marks) -> None:
        target = self.path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"version": VERSION, "painted": [[fx, fy, color] for fx, fy, color in marks]}
        fd, tmp = tempfile.mkstemp(dir=self.directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, target)  # atomic: a crash never leaves a half-written file
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def marks_from_painted(regions: Regions, painted, shape) -> list:
    h, w = shape
    marks = []
    for rid in sorted(painted):
        py, px = (int(v) for v in regions.pole[rid])
        marks.append(((px + 0.5) / w, (py + 0.5) / h, int(regions.label[rid])))
    return marks


def painted_from_marks(regions: Regions, marks, shape):
    """Region ids for saved marks, and how many marks could not be matched to a region."""
    h, w = shape
    painted, dropped = set(), 0
    for fx, fy, color in marks:
        x = min(max(int(fx * w), 0), w - 1)
        y = min(max(int(fy * h), 0), h - 1)
        rid = int(regions.ids[y, x])
        if rid > 0 and int(regions.label[rid]) == color:
            painted.add(rid)
        else:
            dropped += 1
    return painted, dropped
