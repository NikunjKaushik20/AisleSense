"""D-2 walkway-vs-shelf flags (rule frozen before scoring [A35]).

Lingerer = a track episode with speed < 0.3 m/s lasting >= 5 s.
  lingerer inside a `shelf_front` polygon                                       -> BROWSE (green; engagement, never BLOCK)
  lingerer inside a `walkway` polygon while >= 2 OTHER moving tracks pass
  within 2 m of it during the episode                                           -> BLOCK candidate (red)
  anything else                                                                 -> unflagged
Overlapping flags of one type within 2 m are merged (a group of 3 is one flag). Evidence text only describes what was
seen (the X6 flow readers disagreed, kappa ~0, so they never decide).
"""
from __future__ import annotations

import numpy as np

RULE = {"slow_mps": 0.3, "min_s": 5.0, "pass_radius_m": 2.0, "min_passers": 2, "max_gap_s": 2.0, "speed_lag_s": 1.0,
        "text": "lingerer = speed < 0.3 m/s for >= 5 s; shelf_front lingerer -> BROWSE; walkway lingerer with >= 2 other "
                "moving tracks within 2 m during its stay -> BLOCK candidate; else unflagged"}


def arrays(events) -> dict:
    """Event objects or (t, track, x, y) rows -> arrays sorted by (track, t)."""
    if len(events) and hasattr(events[0], "track"):
        t = np.array([e.t for e in events], float); k = np.array([str(e.track) for e in events])
        x = np.array([e.x for e in events], float); y = np.array([e.y for e in events], float)
    else:
        t = np.array([r[0] for r in events], float); k = np.array([str(r[1]) for r in events])
        x = np.array([r[2] for r in events], float); y = np.array([r[3] for r in events], float)
    names, ki = np.unique(k, return_inverse=True) if len(k) else (np.array([]), np.array([], int))
    o = np.lexsort((t, ki))
    return {"t": t[o], "k": ki[o], "x": x[o], "y": y[o], "names": names}


def estimate_hz(a) -> float:
    if len(a["t"]) < 2:
        return 1.0
    same = a["k"][1:] == a["k"][:-1]
    dt = np.diff(a["t"])[same]
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else 1.0


def speeds(a, lag_s: float = 1.0, hz: float = 5.0) -> np.ndarray:
    """Per-sample speed over a ~lag_s displacement within one track (backward; forward at track start)."""
    n = len(a["t"])
    L = max(int(round(lag_s * hz)), 1)
    v = np.full(n, np.nan)
    if n > L:
        same = a["k"][L:] == a["k"][:-L]
        dt = a["t"][L:] - a["t"][:-L]
        d = np.hypot(a["x"][L:] - a["x"][:-L], a["y"][L:] - a["y"][:-L]) / np.maximum(dt, 1e-6)
        ok = same & (dt > 0) & (dt <= 2 * lag_s * max(1.0, 1.0 / (hz * lag_s)))
        v[L:][ok] = d[ok]
        back = np.full(n, np.nan)
        back[:-L][ok] = d[ok]
        v = np.where(np.isnan(v), back, v)
    return np.nan_to_num(v, nan=0.0)


def lingers(a, hz: float | None = None, rule: dict | None = None) -> list:
    """Episodes (track_idx, i0, i1 exclusive, t0, t1, x, y) of slow motion lasting >= min_s."""
    r = {**RULE, **(rule or {})}
    n = len(a["t"])
    if n == 0:
        return []
    hz = hz or estimate_hz(a)
    v = speeds(a, r["speed_lag_s"], hz)
    slow = v < r["slow_mps"]
    brk = np.ones(n, bool)
    brk[1:] = (a["k"][1:] != a["k"][:-1]) | (np.diff(a["t"]) > r["max_gap_s"]) | (slow[1:] != slow[:-1])
    starts = np.flatnonzero(brk)
    ends = np.r_[starts[1:], n]
    out = []
    for i0, i1 in zip(starts, ends):
        if not slow[i0]:
            continue
        t0, t1 = a["t"][i0], a["t"][i1 - 1]
        if t1 - t0 >= r["min_s"]:
            out.append((int(a["k"][i0]), int(i0), int(i1), float(t0), float(t1),
                        float(np.median(a["x"][i0:i1])), float(np.median(a["y"][i0:i1]))))
    return out


def _zone(site, x, y, kind):
    for z in site.zones:
        if z.kind == kind and z.contains(x, y):
            return z.name
    return None


def raw_flags(a, site, hz: float | None = None, rule: dict | None = None) -> list:
    """One flag dict per lingerer episode (before merging)."""
    r = {**RULE, **(rule or {})}
    hz = hz or estimate_hz(a)
    eps = lingers(a, hz, r)
    if not eps:
        return []
    v = speeds(a, r["speed_lag_s"], hz)
    order = np.argsort(a["t"], kind="stable")
    ts = a["t"][order]
    out = []
    for k, i0, i1, t0, t1, x, y in eps:
        shelf = _zone(site, x, y, "shelf_front")
        if shelf is not None:
            out.append({"type": "BROWSE", "zone": shelf, "x": x, "y": y, "t0": t0, "t1": t1, "tracks": {k}, "passers": set(),
                        "pass_speed": []})
            continue
        walk = _zone(site, x, y, "walkway")
        if walk is None:
            continue
        lo, hi = np.searchsorted(ts, t0, "left"), np.searchsorted(ts, t1, "right")
        idx = order[lo:hi]
        near = (np.hypot(a["x"][idx] - x, a["y"][idx] - y) <= r["pass_radius_m"]) & (a["k"][idx] != k) & (v[idx] >= r["slow_mps"])
        passers = set(a["k"][idx][near].tolist())
        if len(passers) >= r["min_passers"]:
            out.append({"type": "BLOCK", "zone": walk, "x": x, "y": y, "t0": t0, "t1": t1, "tracks": {k}, "passers": passers,
                        "pass_speed": v[idx][near].tolist()})
    return out


def merge(fl: list, radius: float = 2.0) -> list:
    fl = sorted(fl, key=lambda f: (f["type"], f["t0"]))
    out: list = []
    for f in fl:
        for g in reversed(out):
            if g["type"] == f["type"] and g["zone"] == f["zone"] and f["t0"] <= g["t1"] and f["t1"] >= g["t0"] \
                    and np.hypot(g["x"] - f["x"], g["y"] - f["y"]) <= radius:
                n = len(g["tracks"])
                g["x"] = (g["x"] * n + f["x"]) / (n + 1); g["y"] = (g["y"] * n + f["y"]) / (n + 1)
                g["t0"], g["t1"] = min(g["t0"], f["t0"]), max(g["t1"], f["t1"])
                g["tracks"] |= f["tracks"]; g["passers"] |= f["passers"]; g["pass_speed"] += f["pass_speed"]
                break
        else:
            out.append({**f, "tracks": set(f["tracks"]), "passers": set(f["passers"]), "pass_speed": list(f["pass_speed"])})
    return out


def flags(events_or_arrays, site, hz: float | None = None, rule: dict | None = None) -> list:
    a = events_or_arrays if isinstance(events_or_arrays, dict) else arrays(events_or_arrays)
    out = []
    for i, f in enumerate(sorted(merge(raw_flags(a, site, hz, rule)), key=lambda f: f["t0"])):
        nl, npass = len(f["tracks"]), len(f["passers"])
        dur = f["t1"] - f["t0"]
        if f["type"] == "BLOCK":
            ps = float(np.mean(f["pass_speed"])) if f["pass_speed"] else 0.0
            ev = (f"{nl} {'person' if nl == 1 else 'people'} stood {dur:.0f} s in {f['zone']} while {npass} others passed "
                  f"within 2 m (mean passer speed {ps:.2f} m/s)")
        else:
            ev = f"{nl} {'person' if nl == 1 else 'people'} lingered {dur:.0f} s at {f['zone']}"
        out.append({"id": f"{f['type'][:2].lower()}{i}-{int(f['t0'])}", "type": f["type"], "zone": f["zone"],
                    "x": round(f["x"], 2), "y": round(f["y"], 2), "t0": round(f["t0"], 2), "t1": round(f["t1"], 2),
                    "n_lingerers": nl, "n_passers": npass, "evidence": ev})
    return out


def as_scoring_flags(fl: list, cell: float = 1.0) -> list:
    """BLOCK flags -> eval.common.score_events tuples (iy, ix, w, t0, t1, cx, cy). BROWSE is not a bottleneck flag."""
    return [(int(f["y"] // cell), int(f["x"] // cell), 0, f["t0"], f["t1"], f["x"], f["y"]) for f in fl if f["type"] == "BLOCK"]
