"""S2 ghost hold (X8): when a track dies with no exit evidence, a ghost holds its presence at the last point.

Rows are (t, track, x, y) in any planar frame (floor metres in the app, image pixels in the MOT20 eval).
A ghost lives from the track's last sample t1 to t1 + TTL (capped at the end of the data), as in X8.
Rebirth candidates: tracks born within the TTL and within k * radius of the ghost point.
Stitch: a reborn track inherits the dead track's identity only if the link is unique both ways (one ghost claims
it and it is the ghost's only candidate); otherwise no identity is merged (abstain). Dwell counts person-seconds per
identity, so a ghost that overlaps its own stitched rebirth is not double counted.

TTL and k were picked on MERL val subjects 21-26 only (grid {0,1,2,4,8} s x {1,2}; X8) and are frozen here.
"""
from __future__ import annotations

import math

TTL_S = 8.0  # frozen: MERL val grid pick (X8)
K = 1.0      # frozen: MERL val grid pick (X8); k did not matter on val
RADIUS_M = 1.0  # one person-scale on the floor (X8 used k x box height in pixels)


def track_ends(rows) -> dict:
    """track -> {t0, t1, x0, y0, x1, y1} from (t, track, x, y) rows (any order)."""
    T = {}
    for t, k, x, y in rows:
        d = T.get(k)
        if d is None:
            T[k] = {"t0": t, "t1": t, "x0": x, "y0": y, "x1": x, "y1": y}
            continue
        if t < d["t0"]:
            d.update(t0=t, x0=x, y0=y)
        if t >= d["t1"]:
            d.update(t1=t, x1=x, y1=y)
    return T


def floor_exit_fn(floor: dict, margin_m: float = 0.5):
    """Exit evidence on the floor: last point within margin of the mapped floor rectangle's edge."""
    x0, y0, W, H = floor["x0"], floor["y0"], floor["width"], floor["height"]

    def exited(track, x, y):
        return x < x0 + margin_m or y < y0 + margin_m or x > x0 + W - margin_m or y > y0 + H - margin_m
    return exited


def _is_exit(exited, track, d) -> bool:
    if exited is None:
        return False
    if callable(exited):
        return bool(exited(track, d["x1"], d["y1"]))
    return track in exited


def holds(rows, ttl_s: float = TTL_S, k: float = K, radius=RADIUS_M, exited=None, t_end: float | None = None) -> list:
    """Ghost holds for every non-exit death. radius: float, or dict track -> scale (e.g. box height px).

    exited: None (no exit evidence anywhere), a set of track ids, or callable(track, x, y) -> bool.
    Returns [{track, t0, t1, x, y, reborn (track id or None), candidates: [track ids]}]."""
    if ttl_s <= 0:
        return []
    T = track_ends(rows)
    births = sorted((d["t0"], kk, d["x0"], d["y0"]) for kk, d in T.items())
    t_last = math.inf if t_end is None else float(t_end)
    import bisect
    out = []
    bt = [b[0] for b in births]
    for kk, d in T.items():
        if _is_exit(exited, kk, d):
            continue
        r = radius.get(kk, RADIUS_M) if isinstance(radius, dict) else float(radius)
        lim = k * max(r, 1e-9)
        end = min(d["t1"] + ttl_s, t_last)
        cands = []
        i = bisect.bisect_right(bt, d["t1"])
        while i < len(births) and births[i][0] <= d["t1"] + ttl_s:
            tb, kb, xb, yb = births[i]
            if kb != kk and math.hypot(xb - d["x1"], yb - d["y1"]) <= lim:
                cands.append((tb, kb))
            i += 1
        reborn = cands[0][1] if len(cands) == 1 else None
        if end > d["t1"]:
            out.append({"track": kk, "t0": d["t1"], "t1": end, "x": d["x1"], "y": d["y1"], "reborn": reborn,
                        "candidates": [c[1] for c in cands]})
    return out


def stitch(hs: list) -> dict:
    """reborn track -> root identity, only for links unique both ways (abstain when ambiguous)."""
    claims = {}
    for h in hs:
        for c in h.get("candidates", []):
            claims.setdefault(c, []).append(h["track"])
    parent = {}
    for h in hs:
        c = h["candidates"]
        if len(c) == 1 and len(claims[c[0]]) == 1:
            parent[c[0]] = h["track"]

    def root(x):
        seen = set()
        while x in parent and x not in seen:
            seen.add(x)
            x = parent[x]
        return x
    return {kb: root(kb) for kb in parent}


def ghost_rows(hs: list, hz: float) -> list:
    """Synthetic presence samples (t, track, x, y) every 1/hz inside each hold (t0 exclusive, t1 inclusive)."""
    dt = 1.0 / hz
    out = []
    for h in hs:
        span = (h["t1"] - h["t0"]) / dt
        n = int(math.floor(span + 1e-6))
        for i in range(1, n + 1):
            out.append((h["t0"] + i * dt, h["track"], h["x"], h["y"]))
    return out


def apply(rows, hz: float, ttl_s: float = TTL_S, k: float = K, radius=RADIUS_M, exited=None, t_end=None):
    """rows + ghost samples (sorted by t, track) and the holds."""
    rows = list(rows)
    hs = holds(rows, ttl_s, k, radius, exited, t_end)
    out = rows + ghost_rows(hs, hz)
    out.sort(key=lambda r: (r[0], str(r[1])))
    return out, hs
