"""Section-to-section journeys -> Sankey JSON (directed; A->B and B->A both kept).

Per track (sorted by time): map each observation to its section (site.zone_of, sections only), drop off-section samples,
collapse runs (a section counts only after min_samples consecutive observations, so jitter on a border is not a trip),
count consecutive distinct pairs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def contains_vec(poly, x, y) -> np.ndarray:
    """Vectorised ray casting, same rule as schema.Zone.contains."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    inside = np.zeros(len(x), bool)
    pts = [tuple(map(float, p)) for p in poly]
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]; xj, yj = pts[j]
        cross = (yi > y) != (yj > y)
        inside ^= cross & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        j = i
    return inside


def transitions(site, events, min_samples: int = 2) -> Counter:
    from aisle.analytics.block import arrays
    a = events if isinstance(events, dict) else arrays(events)
    secs = [z for z in site.zones if z.kind == "section"]
    n = len(a["t"])
    if n == 0 or not secs:
        return Counter()
    lab = np.full(n, -1)
    for j, z in enumerate(secs):
        m = lab < 0
        idx = np.flatnonzero(m)
        inside = contains_vec(z.polygon, a["x"][idx], a["y"][idx])
        lab[idx[inside]] = j
    c: Counter = Counter()
    k = a["k"]
    starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
    ends = np.r_[starts[1:], n]
    for s, e in zip(starts, ends):
        seq = lab[s:e]
        seq = seq[seq >= 0]
        if len(seq) == 0:
            continue
        b = np.flatnonzero(np.r_[True, seq[1:] != seq[:-1]])
        runs = np.diff(np.r_[b, len(seq)])
        path = [int(seq[i]) for i, r in zip(b, runs) if r >= min_samples]
        path = [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]
        for u, v in zip(path[:-1], path[1:]):
            c[(secs[u].name, secs[v].name)] += 1
    return c


def payload(site, events, session: str) -> dict:
    c = transitions(site, events)
    names = [z.name for z in site.zones if z.kind == "section"]
    ix = {nm: i for i, nm in enumerate(names)}
    links = [{"source": ix[u], "target": ix[v], "value": int(n)} for (u, v), n in sorted(c.items(), key=lambda kv: -kv[1])]
    return {"session": session, "site": site.id, "nodes": [{"name": nm} for nm in names], "links": links,
            "unit": "track transitions between sections (tracker IDs; an ID break can hide a trip)"}
