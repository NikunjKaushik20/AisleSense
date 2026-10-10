"""1 m floor grid accumulating person-seconds, track births and track deaths (streaming).

Person-seconds: each observation of a track is weighted by the time since that track's previous
observation (capped at max_gap); the first observation gets nominal_dt. A person standing still
for T seconds sampled every dt therefore contributes T (+- one sample) to their cell.
"""
from __future__ import annotations

import numpy as np


class Grid:
    def __init__(self, x0: float, y0: float, width: float, height: float, cell: float = 1.0,
                 nominal_dt: float = 1.0, max_gap: float = 2.0, death_gap: float = 5.0):
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)
        self.nx = max(1, int(np.ceil(width / cell)))
        self.ny = max(1, int(np.ceil(height / cell)))
        self.nominal_dt, self.max_gap, self.death_gap = float(nominal_dt), float(max_gap), float(death_gap)
        self.person_s = np.zeros((self.ny, self.nx))
        self.births = np.zeros((self.ny, self.nx), np.int64)
        self.deaths = np.zeros((self.ny, self.nx), np.int64)
        self._last = {}  # track -> (t, iy, ix)  (iy/ix may be -1 when off-grid)
        self.t_max = -np.inf
        self.n_events = 0
        self.n_off_grid = 0

    @classmethod
    def from_floor(cls, floor: dict, **kw) -> "Grid":
        return cls(floor["x0"], floor["y0"], floor["width"], floor["height"], floor.get("cell", 1.0), **kw)

    def cell_of(self, x: float, y: float):
        ix = int(np.floor((x - self.x0) / self.cell))
        iy = int(np.floor((y - self.y0) / self.cell))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            return iy, ix
        return -1, -1

    def add(self, t: float, track: str, x: float, y: float) -> None:
        self.n_events += 1
        iy, ix = self.cell_of(x, y)
        prev = self._last.get(track)
        if prev is None or t - prev[0] > self.death_gap:
            if prev is not None:
                self._die(prev)
            w = self.nominal_dt
            if iy >= 0:
                self.births[iy, ix] += 1
        else:
            w = min(max(t - prev[0], 0.0), self.max_gap)
        if iy >= 0:
            self.person_s[iy, ix] += w
        else:
            self.n_off_grid += 1
        self._last[track] = (t, iy, ix)
        if t > self.t_max:
            self.t_max = t

    def add_many(self, events) -> None:
        for e in events:
            self.add(e.t, e.track, e.x, e.y)

    def _die(self, prev) -> None:
        _, iy, ix = prev
        if iy >= 0:
            self.deaths[iy, ix] += 1

    def expire(self, now: float | None = None) -> int:
        """Close tracks not seen for > death_gap (all tracks if now is None). Returns count closed."""
        dead = [k for k, v in self._last.items() if now is None or now - v[0] > self.death_gap]
        for k in dead:
            self._die(self._last.pop(k))
        return len(dead)

    def active_tracks(self) -> dict:
        return dict(self._last)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "cell": self.cell, "nx": self.nx, "ny": self.ny,
                "person_s": self.person_s.ravel().round(3).tolist(),
                "births": self.births.ravel().tolist(), "deaths": self.deaths.ravel().tolist()}
