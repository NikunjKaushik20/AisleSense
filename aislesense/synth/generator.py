"""Synthetic scenario generator: scenario yaml -> labelled position rows.

M1 scope: passers (walk the main lane end to end) and lingerers (walk to a shelf front, stand, leave).
Returns rows (t, track, x, y) sorted by t, plus truth visits [{track, zone, t_in, t_out}].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def _bbox(poly):
    a = np.asarray(poly, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def _walk(p0, p1, speed, t0, hz):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dur = max(np.linalg.norm(p1 - p0) / speed, 1.0 / hz)
    n = max(int(np.ceil(dur * hz)), 1)
    ts = t0 + np.arange(n) / hz
    f = (ts - t0) / dur
    return ts, p0 + f[:, None] * (p1 - p0), t0 + n / hz


def generate(scenario: dict, site) -> tuple[list, list]:
    rng = np.random.default_rng(int(scenario.get("seed", 0)))
    T = float(scenario.get("duration_s", 600))
    hz = float(scenario.get("hz", 5))
    W, H = site.floor["width"], site.floor["height"]
    x0, y0 = site.floor["x0"], site.floor["y0"]
    rows, truth, nid = [], [], 0

    p = scenario.get("passers") or {}
    rate = float(p.get("rate_per_s", 0))
    smin, smax = p.get("speed_mps", [1.0, 1.5])
    ly0, ly1 = p.get("lane_y", [y0 + H * 0.25, y0 + H * 0.45])
    t = rng.exponential(1 / rate) if rate > 0 else T
    while t < T:
        nid += 1
        y = rng.uniform(ly0, ly1)
        left = rng.random() < 0.5
        a, b = ([x0 + 0.05, y], [x0 + W - 0.05, y]) if left else ([x0 + W - 0.05, y], [x0 + 0.05, y])
        ts, ps, _ = _walk(a, b, rng.uniform(smin, smax), t, hz)
        rows += [(float(tt), f"s{nid}", float(q[0]), float(q[1])) for tt, q in zip(ts, ps)]
        t += rng.exponential(1 / rate)

    zones = {z.name: z for z in site.zones}
    entry = np.array([x0 + 1.0, ly0])
    exit_ = np.array([x0 + W - 0.5, ly0])
    for L in scenario.get("lingerers") or []:
        zname = L["zone"]
        tgt = zones.get(f"{zname} front") or zones[zname]
        bx0, by0, bx1, by1 = _bbox(tgt.polygon)
        r = float(L.get("rate_per_s", 0.01))
        s0, s1 = L.get("stay_s", [20, 60])
        t = rng.exponential(1 / r)
        while t < T:
            nid += 1
            spot = np.array([rng.uniform(bx0 + 0.2, bx1 - 0.2), rng.uniform(by0 + 0.1, by1 - 0.1)])
            ts1, ps1, t_arr = _walk(entry, spot, 1.2, t, hz)
            stay = rng.uniform(s0, s1)
            n = max(int(stay * hz), 1)
            ts2 = t_arr + np.arange(n) / hz
            ps2 = spot + rng.normal(0, 0.05, (n, 2))
            ts3, ps3, _ = _walk(spot, exit_, 1.2, t_arr + n / hz, hz)
            for ts, ps in ((ts1, ps1), (ts2, ps2), (ts3, ps3)):
                rows += [(float(tt), f"l{nid}", float(q[0]), float(q[1])) for tt, q in zip(ts, ps)]
            truth.append({"track": f"l{nid}", "zone": zname, "t_in": float(t_arr), "t_out": float(t_arr + n / hz)})
            t += rng.exponential(1 / r)

    rows = [r for r in rows if r[0] < T]
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows, truth


def load_scenario(site) -> dict:
    return yaml.safe_load((Path(site.path) / site.extra.get("scenario", "scenario.yaml")).read_text(encoding="utf-8"))
