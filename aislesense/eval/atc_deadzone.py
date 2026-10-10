"""ATC dead-zone eval: decile baseline + blind / ignored injection harness (X2, X9, X11 protocol).

Data: ATC mall trajectories 2012-11-14 (day 1) and 2012-11-21 (day 2), 1 Hz per person via the app ingest
(aisle.ingest.sources.atc_1hz). Grid: 1 m cells, 90 x 53, origin fixed to the day-1 minimum (as atc_heldout.py) so
cells line up across days. Visitors = distinct IDs per cell per day; terminations = first + last cell of each ID.

Decile rule (baseline): observed cell (>= 1 visitor) with visitors <= 10th percentile of observed cells -> DEAD.
Injection (10 patch seeds x 2 days, 1% kept, up to 3 busy interior 4x4 m patches per seed):
  blind   = weak camera: each person-second inside the patch kept w.p. 1%, new ID at every gap or patch crossing.
            Truth: busy floor, so DEAD is a false alarm.
  ignored = people truly skip the patch, camera fine: each visiting ID keeps its patch rows w.p. 1% (by ID), no new IDs.
            Truth: dead floor, so DEAD is correct.
Rule arms are pluggable: RULES maps name -> fn(vis, term) -> bool DEAD mask. M2 ships `decile`; M4 adds `aware` (aisle.analytics.deadzone, rule v2 frozen).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import fingerprint, write_result  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage  # noqa: E402

from aisle.analytics import deadzone as dz  # noqa: E402
from aisle.ingest.sources import atc_1hz  # noqa: E402
from aisle.schema import data_root  # noqa: E402

NX, NY = 90, 53
DAYS = ("20121114", "20121121")
SEEDS = range(10)
KEEP = 0.01
EXPECTED = {"blind_decile_dead": 0.634, "ignored_decile_dead": 0.655, "day1_decile_on_edge": 0.958,
            "blind_aware_dead": 0.0, "blind_aware_unknown": 0.939, "ignored_aware_dead": 0.625,
            "blind_cells_pooled": 656}


def day_path(day: str) -> Path:
    return data_root() / "atc_x" / f"atc-{day}.npz"


def load_day(day: str) -> pd.DataFrame:
    d = atc_1hz(day_path(day))
    b = pd.DataFrame({"id": d["id"], "s": d["s"], "x": d["x"].astype(float), "y": d["y"].astype(float)})
    return b.sort_values(["id", "s"], kind="stable").reset_index(drop=True)


def cells(b: pd.DataFrame, X0: float, Y0: float):
    return dz.counts(b.id.values, b.x.values, b.y.values, X0, Y0, NX, NY)


decile_rule = dz.decile_rule
RULES = {"decile": dz.decile_rule, "aware": dz.aware_rule}


def edge_mask(obs):
    return obs & ndimage.binary_dilation(~obs, np.ones((3, 3)))


def patches(obs, dist, vis, seed):
    return dz.patches(obs, dist, vis, seed)


def inject(b, patch, X0, Y0, seed, keep=KEEP):
    """Returns (blind_b, ignored_b) for one patch seed. RNG order identical to atc_inject_seeds.py."""
    return dz.inject(b, patch, X0, Y0, seed, keep)


def run(rules: dict | None = None, days=DAYS, seeds=SEEDS) -> dict:
    rules = rules or RULES
    b0 = load_day(days[0])
    X0, Y0 = float(b0.x.min()), float(b0.y.min())
    rows, per_day = [], {}
    for day in days:
        b = b0 if day == days[0] else load_day(day)
        vis, term = cells(b, X0, Y0)
        obs = vis >= 1
        dist = ndimage.distance_transform_edt(obs)
        edge = edge_mask(obs)
        base = {name: fn(vis, term) for name, fn in rules.items()}
        per_day[day] = {"person_s": int(len(b)), "ids": int(b.id.nunique()), "observed_cells": int(obs.sum()),
                        **{f"{n}_dead": int(m.sum()) for n, m in base.items()},
                        **{f"{n}_dead_on_edge_share": float((m & edge).sum() / max(m.sum(), 1)) for n, m in base.items()},
                        **{f"{n}_dead_within_2m_of_edge_share": float((m & (dist <= 2)).sum() / max(m.sum(), 1)) for n, m in base.items()}}
        for seed in seeds:
            patch = patches(obs, dist, vis, seed)
            bl, ig = inject(b, patch, X0, Y0, seed)
            for test, bb in (("blind", bl), ("ignored", ig)):
                v2, t2 = cells(bb, X0, Y0)
                r = {"day": day, "seed": seed, "test": test, "cells": int(patch.sum()),
                     "visitors_before": float(vis[patch].mean()), "visitors_after": float(v2[patch].mean()),
                     "unobserved": int((patch & (v2 < 1)).sum()),
                     "aware_unseen": int((dz.aware(v2, t2)["unseen"] & patch).sum())}
                for name, fn in rules.items():
                    r[f"{name}_dead"] = int((fn(v2, t2) & patch).sum())
                rows.append(r)
        print(f"atc {day} done", flush=True)
    R = pd.DataFrame(rows)
    out = {"arms": {}, "per_day_baseline": per_day}
    for test in ("blind", "ignored"):
        g = R[R.test == test]
        cellsn = int(g.cells.sum())
        arm = {"cells_pooled": cellsn, "patches_pooled": cellsn // 16,
               "aware_unseen_pooled": int(g.aware_unseen.sum()), "aware_unseen_rate_pooled": g.aware_unseen.sum() / max(cellsn, 1),
               "unobserved_pooled": int(g.unobserved.sum())}
        for name in rules:
            arm[name] = {"dead_pooled": int(g[f"{name}_dead"].sum()), "dead_rate_pooled": g[f"{name}_dead"].sum() / max(cellsn, 1),
                         "per_day": {d: {"dead": int(gg[f"{name}_dead"].sum()), "cells": int(gg.cells.sum()),
                                         "rate": gg[f"{name}_dead"].sum() / max(gg.cells.sum(), 1)} for d, gg in g.groupby("day")},
                         "per_seed_dead_range": [int(g[f"{name}_dead"].min()), int(g[f"{name}_dead"].max())]}
        out["arms"][test] = arm
    out["rows"] = R.to_dict("records")
    out["origin"] = {"X0": X0, "Y0": Y0, "nx": NX, "ny": NY, "cell_m": 1.0, "fixed_to": days[0]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="same protocol (all 10 seeds x 2 days); kept for run_all symmetry")
    ap.parse_args(argv)
    o = run()
    bl, ig = o["arms"]["blind"], o["arms"]["ignored"]
    res = {
        "data": "ATC mall trajectories (real sensor tracks, trajectory replay; no CV) 2012-11-14 + 2012-11-21; injection is synthetic on real tracks",
        "config": {"cell_m": 1.0, "grid": [NX, NY], "keep": KEEP, "seeds": list(SEEDS), "days": list(DAYS),
                   "decile_rule": "observed cell with visitors <= 10th percentile of observed cells",
                   "aware_rule": dz.RULE["text"]},
        "seed": "patch seeds 0-9, keep rng 100+seed", "data_hash": fingerprint([day_path(d) for d in DAYS]),
        "n": {"blind_cells_pooled": bl["cells_pooled"], "ignored_cells_pooled": ig["cells_pooled"]},
        "metric": {"blind": bl, "ignored": ig, "per_day_baseline": o["per_day_baseline"]},
        "prior": {"name": "decile rule (the common baseline)",
                  "blind_dead_rate": bl["decile"]["dead_rate_pooled"], "ignored_dead_rate": ig["decile"]["dead_rate_pooled"]},
        "headline": [
            {"metric": "blind busy cells called DEAD (false alarm), pooled", "arm": "decile", "value": bl["decile"]["dead_rate_pooled"],
             "count": bl["decile"]["dead_pooled"], "of": bl["cells_pooled"], "unit": "per injected cell, 10 seeds x 2 days"},
            {"metric": "ignored cells called DEAD (hit), pooled", "arm": "decile", "value": ig["decile"]["dead_rate_pooled"],
             "count": ig["decile"]["dead_pooled"], "of": ig["cells_pooled"]},
            {"metric": "blind busy cells called DEAD (false alarm), pooled", "arm": "aware", "value": bl["aware"]["dead_rate_pooled"],
             "count": bl["aware"]["dead_pooled"], "of": bl["cells_pooled"], "per_seed_dead_range": bl["aware"]["per_seed_dead_range"],
             "unit": "per injected cell, 10 seeds x 2 days"},
            {"metric": "blind busy cells shaded UNSEEN (coverage gate), pooled", "arm": "aware", "value": bl["aware_unseen_rate_pooled"],
             "count": bl["aware_unseen_pooled"], "of": bl["cells_pooled"]},
            {"metric": "ignored cells called DEAD (hit), pooled", "arm": "aware", "value": ig["aware"]["dead_rate_pooled"],
             "count": ig["aware"]["dead_pooled"], "of": ig["cells_pooled"]},
            {"metric": "day-1 decile DEAD cells on coverage edge", "arm": "decile",
             "value": o["per_day_baseline"][DAYS[0]]["decile_dead_on_edge_share"]},
        ],
        "expected_X2_X11": EXPECTED,
        "origin": o["origin"],
        "rows": o["rows"],
    }
    write_result("atc_deadzone", res)
    print(f"atc_deadzone: blind decile DEAD {bl['decile']['dead_pooled']}/{bl['cells_pooled']} ({bl['decile']['dead_rate_pooled']:.3f}); "
          f"ignored {ig['decile']['dead_rate_pooled']:.3f} | aware blind DEAD {bl['aware']['dead_pooled']}/{bl['cells_pooled']} "
          f"UNSEEN {bl['aware_unseen_rate_pooled']:.3f}, ignored DEAD {ig['aware']['dead_rate_pooled']:.3f}; day1 edge {o['per_day_baseline'][DAYS[0]]['decile_dead_on_edge_share']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
