"""Floor heatmap of person-seconds per 1 m cell (live grid or recomputed from stored events)."""
from __future__ import annotations

from aisle.geo.grid import Grid


def grid_for(site, source=None) -> Grid:
    kw = {}
    if source is not None:
        kw = dict(nominal_dt=source.nominal_dt, max_gap=source.max_gap, death_gap=source.death_gap)
    return Grid.from_floor(site.floor, **kw)


def grid_from_events(site, events, source_params: dict | None = None) -> Grid:
    g = Grid.from_floor(site.floor, **(source_params or {}))
    g.add_many(events)
    g.expire()
    return g


def payload(grid: Grid, session: str | None, site_id: str, t0=None, t1=None) -> dict:
    cells = grid.person_s.ravel()
    return {
        "session": session, "site": site_id, "t0": t0, "t1": t1,
        "unit": "person-seconds", "x0": grid.x0, "y0": grid.y0, "cell": grid.cell,
        "nx": grid.nx, "ny": grid.ny, "order": "row-major, row = y index from y0",
        "cells": cells.round(3).tolist(), "max": float(cells.max()) if cells.size else 0.0,
        "births": grid.births.ravel().tolist(), "deaths": grid.deaths.ravel().tolist(),
    }
