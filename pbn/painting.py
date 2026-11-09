"""Which regions the painter has finished, with undo and "next region" navigation.

Pure Python state (no Streamlit), so the rules can be tested on their own.
"""
import numpy as np

from pbn.regions import Regions


class PaintState:
    def __init__(self, regions: Regions, painted=()):
        self.regions = regions
        self.painted = set(painted)
        self._history = []  # each entry: (ids that were added, ids that were removed)

    # -- changing -----------------------------------------------------------------------
    def _apply(self, add, remove):
        add = {i for i in add if i not in self.painted}
        remove = {i for i in remove if i in self.painted}
        if not add and not remove:
            return
        self.painted |= add
        self.painted -= remove
        self._history.append((add, remove))

    def toggle(self, rid: int) -> None:
        if rid <= 0 or rid > self.regions.count:
            return
        if rid in self.painted:
            self._apply((), (rid,))
        else:
            self._apply((rid,), ())

    def set_color(self, color: int, painted: bool) -> None:
        ids = self.regions.of_label(color).tolist()
        if painted:
            self._apply(ids, ())
        else:
            self._apply((), ids)

    @property
    def can_undo(self) -> bool:
        return bool(self._history)

    def undo(self) -> bool:
        if not self._history:
            return False
        add, remove = self._history.pop()
        self.painted -= add
        self.painted |= remove
        return True

    # -- reading ------------------------------------------------------------------------
    def left(self, color: int) -> int:
        return sum(1 for i in self.regions.of_label(color).tolist() if i not in self.painted)

    def total(self, color: int) -> int:
        return len(self.regions.of_label(color))

    def is_done(self, color: int) -> bool:
        return self.total(color) > 0 and self.left(color) == 0

    def painted_fraction(self) -> float:
        total = int(self.regions.area[1:].sum())
        if total == 0:
            return 0.0
        return float(self.regions.area[sorted(self.painted)].sum()) / total if self.painted else 0.0

    def next_unpainted(self, color: int, after=None):
        """The next unpainted region of ``color`` in reading order after region ``after``.

        Wraps around; ``after`` may be any region id (even one just painted). None when the
        colour has nothing left.
        """
        todo = [i for i in self.regions.of_label(color).tolist() if i not in self.painted]
        if not todo:
            return None
        if after is None or after <= 0 or after > self.regions.count:
            return todo[0]
        w_key = self.regions.ids.shape[1]
        anchors = self.regions.anchor
        keys = np.array([int(anchors[i, 0]) * w_key + int(anchors[i, 1]) for i in todo])
        after_key = int(anchors[after, 0]) * w_key + int(anchors[after, 1])
        idx = int(np.searchsorted(keys, after_key, side="right"))
        return todo[idx % len(todo)]
