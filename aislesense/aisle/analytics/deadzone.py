"""D-1 coverage-aware dead zones (frozen rule v2 [X7][X9]; injection protocol of X11).

Per 1 m cell: visitors = distinct track IDs, terminations = first + last in-grid cell of each track.
  termination share = terminations / visitors > TAU            -> UNSEEN (camera edge or weak view; "add or move a camera")
  else DEAD iff Poisson-low vs the 5x5 neighbourhood median (BH-FDR q) AND visitors < RATIO x that median
  a DEAD cell not DEAD in the previous session of the same site -> WATCH (persistence, F8)
  observed and none of the above -> ok;  no visitors -> unknown.
Baseline (the default entry): decile rule = observed cell with visitors <= 10th percentile of observed cells.

Arrays here are indexed [ix, iy] (x first) like the eval scripts; payload() converts to the API row-major order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage, stats

RULE = {"tau": 0.25, "fdr_q": 0.05, "ratio": 0.25, "window": 5, "min_neigh": 5, "version": "v2 (frozen after X7)",
        "text": "UNSEEN if (births+deaths)/visitors > 0.25; else DEAD iff Poisson-low vs 5x5 neighbour median (BH-FDR 0.05) "
                "AND visitors < 0.25 x that median; DEAD not dead in previous session -> WATCH"}
KEEPS = (1.0, 0.15, 0.05, 0.01)
STATES = ("ok", "dead", "watch", "unseen", "unknown")


# ---------------------------------------------------------------- counts
def counts(ids, x, y, x0: float, y0: float, nx: int, ny: int, cell: float = 1.0):
    """ids/x/y sorted by (id, time). Returns vis, term arrays of shape (nx, ny)."""
    cx = np.floor((np.asarray(x, float) - x0) / cell).astype(int)
    cy = np.floor((np.asarray(y, float) - y0) / cell).astype(int)
    m = (cx >= 0) & (cx < nx) & (cy >= 0) & (cy < ny)
    ids = np.asarray(ids)[m]; cx, cy = cx[m], cy[m]
    vis = np.zeros((nx, ny)); term = np.zeros((nx, ny))
    if len(ids) == 0:
        return vis, term
    u = pd.DataFrame({"id": ids, "cx": cx, "cy": cy}).drop_duplicates()
    np.add.at(vis, (u.cx.values, u.cy.values), 1)
    g = pd.DataFrame({"id": ids, "cx": cx, "cy": cy}).groupby("id", sort=False)
    for e in (g.first(), g.last()):
        np.add.at(term, (e.cx.values, e.cy.values), 1)
    return vis, term


# ---------------------------------------------------------------- rules
def decile_rule(vis, term=None):
    obs = vis >= 1
    if not obs.any():
        return obs
    return obs & (vis <= np.percentile(vis[obs], 10))


def _neigh_median(vals, w: int, min_n: int):
    c = (w * w) // 2

    def f(win):
        r = np.r_[win[:c], win[c + 1:]]
        return np.nanmedian(r) if np.isfinite(r).sum() >= min_n else np.nan
    return ndimage.generic_filter(vals, f, size=w, mode="constant", cval=np.nan)


def aware(vis, term, rule: dict | None = None) -> dict:
    """Returns masks: obs, unseen, dead_v1 (no effect size), dead (v2), plus lam (neighbour median) and term share."""
    r = {**RULE, **(rule or {})}
    obs = vis >= 1
    ts = np.where(obs, term / np.maximum(vis, 1), 0.0)
    unseen = obs & (ts > r["tau"])
    cov = obs & ~unseen
    lam = _neigh_median(np.where(cov, vis, np.nan), r["window"], r["min_neigh"])
    ok = cov & np.isfinite(lam)
    p = np.where(ok, stats.poisson.cdf(vis, np.nan_to_num(lam, nan=1.0)), 1.0)
    srt = np.sort(p[ok]); m = len(srt)
    k = np.where(srt <= r["fdr_q"] * np.arange(1, m + 1) / max(m, 1))[0]
    thr = srt[k.max()] if len(k) else -1.0
    dead1 = ok & (p <= thr)
    dead = dead1 & (vis < r["ratio"] * np.nan_to_num(lam, nan=0.0))
    return {"obs": obs, "unseen": unseen, "dead_v1": dead1, "dead": dead, "lam": lam, "term_share": ts}


def aware_rule(vis, term):
    """RULES-compatible arm for eval/atc_deadzone.py: bool DEAD mask (v2)."""
    return aware(vis, term)["dead"]


def states(vis, term, prev_dead=None) -> tuple[np.ndarray, dict]:
    a = aware(vis, term)
    st = np.full(vis.shape, "unknown", dtype=object)
    st[a["obs"]] = "ok"
    st[a["unseen"]] = "unseen"
    st[a["dead"]] = "dead"
    if prev_dead is not None and prev_dead.shape == vis.shape:
        st[a["dead"] & ~prev_dead] = "watch"
    return st, a


def coverage(a: dict) -> np.ndarray:
    """0..1 confidence that the cell's counts reflect the floor: 1 - termination share (0 where unobserved)."""
    return np.where(a["obs"], np.clip(1.0 - a["term_share"], 0.0, 1.0), 0.0)


# ---------------------------------------------------------------- X11 injection (RNG order identical to atc_inject_seeds.py)
def patches(obs, dist, vis, seed: int, max_cells: int = 48):
    busy = obs & (dist >= 4) & (vis >= np.median(vis[obs]) if obs.any() else obs)
    cand = np.argwhere(busy)
    patch = np.zeros_like(obs)
    if len(cand) == 0:
        return patch
    rng = np.random.default_rng(seed)
    tries = 0
    while patch.sum() < max_cells and tries < 1000:
        i, j = cand[rng.integers(len(cand))]
        tries += 1
        sl = (slice(i, i + 4), slice(j, j + 4))
        if busy[sl].shape == (4, 4) and busy[sl].all() and not ndimage.binary_dilation(patch, np.ones((5, 5)))[sl].any():
            patch[sl] = True
    return patch


def inject(b: pd.DataFrame, patch, x0: float, y0: float, seed: int, keep: float, cell: float = 1.0):
    """b: columns id, s (int time), x, y sorted by (id, s). Returns (blind_b, ignored_b)."""
    nx, ny = patch.shape
    cx = ((b.x - x0) // cell).astype(int).clip(0, nx - 1).values
    cy = ((b.y - y0) // cell).astype(int).clip(0, ny - 1).values
    inp = patch[cx, cy]
    rng = np.random.default_rng(100 + seed)
    k = ~inp | (rng.random(len(b)) < keep)
    bi = b[k].copy(); ik = inp[k]
    brk = (bi.id.values[1:] != bi.id.values[:-1]) | (ik[1:] != ik[:-1]) | ((bi.s.values[1:] - bi.s.values[:-1]) > 1)
    bi["id"] = np.r_[0, np.cumsum(brk)]
    ids = np.unique(b.id.values[inp])
    kid = set(ids[rng.random(len(ids)) < keep])
    ign = b[~inp | b.id.isin(kid).values]
    return bi, ign


# ---------------------------------------------------------------- app view
def frame(events) -> pd.DataFrame:
    """Event objects or (t, track, x, y) rows -> DataFrame id, s, x, y sorted by (id, s) (1 Hz, per-second mean)."""
    if len(events) and hasattr(events[0], "track"):
        d = pd.DataFrame({"t": [e.t for e in events], "id": [str(e.track) for e in events],
                          "x": [e.x for e in events], "y": [e.y for e in events]})
    else:
        d = pd.DataFrame(list(events), columns=["t", "id", "x", "y"]) if len(events) else pd.DataFrame(columns=["t", "id", "x", "y"])
        d["id"] = d["id"].astype(str)
    if d.empty:
        return pd.DataFrame({"id": pd.Series(dtype=str), "s": pd.Series(dtype=np.int64), "x": pd.Series(dtype=float), "y": pd.Series(dtype=float)})
    d["s"] = np.floor(d.t.astype(float)).astype(np.int64)
    b = d.groupby(["id", "s"], sort=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index()
    return b


def analyse(site, events, keep: float = 1.0, prev_events=None, seed: int = 0) -> dict:
    fl = site.floor
    c = float(fl.get("cell", 1.0))
    nx, ny = int(np.ceil(fl["width"] / c)), int(np.ceil(fl["height"] / c))
    b = frame(events)
    vis, term = counts(b.id.values, b.x.values, b.y.values, fl["x0"], fl["y0"], nx, ny, c)
    patch = np.zeros((nx, ny), bool)
    if keep < 1.0 and len(b):
        obs = vis >= 1
        patch = patches(obs, ndimage.distance_transform_edt(obs), vis, seed)
        if patch.any():
            b, _ = inject(b, patch, fl["x0"], fl["y0"], seed, keep, c)
            vis, term = counts(b.id.values, b.x.values, b.y.values, fl["x0"], fl["y0"], nx, ny, c)
    prev_dead = None
    if prev_events is not None:
        pb = frame(prev_events)
        if len(pb):
            pv, pt = counts(pb.id.values, pb.x.values, pb.y.values, fl["x0"], fl["y0"], nx, ny, c)
            prev_dead = aware(pv, pt)["dead"]
    st, a = states(vis, term, prev_dead)
    return {"nx": nx, "ny": ny, "cell": c, "vis": vis, "term": term, "state": st, "aware": a,
            "decile": decile_rule(vis, term), "coverage": coverage(a), "patch": patch, "has_prev": prev_dead is not None}


def _rm(arr):
    """(nx, ny) -> row-major list (row = y index), the /api/heatmap layout."""
    return np.asarray(arr).T.ravel().tolist()


def pooled_from_results(res: dict | None) -> dict | None:
    if not res:
        return None
    m = res.get("metric", {})
    out = {}
    for test in ("blind", "ignored"):
        arm = m.get(test)
        if not arm or "aware" not in arm:
            return None
        out[test] = {"n": arm["cells_pooled"], "aware_dead": arm["aware"]["dead_pooled"],
                     "aware_unknown": arm.get("aware_unseen_pooled"), "decile_dead": arm["decile"]["dead_pooled"]}
    return out


def payload(site, events, session: str, keep: float = 1.0, prev_events=None, results: dict | None = None) -> dict:
    r = analyse(site, events, keep, prev_events)
    pooled = pooled_from_results(results)
    st = r["state"]
    return {"session": session, "site": site.id, "keep": keep, "x0": site.floor["x0"], "y0": site.floor["y0"],
            "cell": r["cell"], "nx": r["nx"], "ny": r["ny"], "order": "row-major, row = y index from y0",
            "state": _rm(st), "decile": [bool(v) for v in _rm(r["decile"])],
            "coverage": [round(float(v), 3) for v in _rm(r["coverage"])],
            "patch": [bool(v) for v in _rm(r["patch"])],
            "counts": {s: int((st == s).sum()) for s in STATES} | {"decile_dead": int(r["decile"].sum()),
                                                                    "patch_cells": int(r["patch"].sum()),
                                                                    "patch_decile_dead": int((r["decile"] & r["patch"]).sum()),
                                                                    "patch_aware_dead": int((np.isin(st, ["dead", "watch"]) & r["patch"]).sum()),
                                                                    "patch_unseen": int(((st == "unseen") & r["patch"]).sum())},
            "previous_session": bool(r["has_prev"]), "rule": RULE["text"],
            "pooled": pooled, "source": "results/atc_deadzone.json" if pooled else None}


def zone_cells(site, st, kinds=("section", "walkway", "shelf_front")) -> dict:
    """{state: {zone_name|None: n_cells}} from an (nx, ny) state array."""
    fl = site.floor; c = float(fl.get("cell", 1.0))
    out: dict = {}
    nx, ny = st.shape
    for ix in range(nx):
        for iy in range(ny):
            s = st[ix, iy]
            if s in ("ok", "unknown"):
                continue
            z = site.zone_of(fl["x0"] + (ix + 0.5) * c, fl["y0"] + (iy + 0.5) * c, kinds=kinds)
            out.setdefault(s, {}).setdefault(z, 0)
            out[s][z] += 1
    return out
