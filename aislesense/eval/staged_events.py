"""Staged block / browse events harness (20+ of each [A29]); labels/events.csv = session,type,t0,t1,zone.

Tune (tune sessions only): the density baseline's window length from {15, 30, 60} s by block recall minus browse
false alarm. Score (score sessions only, once; lock in var/locks/): block recall and browse false-alarm rate per arm.
Arms: density (baseline, eval.common.density_flags); walkway (M4 D-2, aisle.analytics.block, nothing tuned).
Trivial references: flag nothing (recall 0, FA 0) and flag every event (recall 1, FA 1).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import DENSITY_RULE, check_and_lock, code_hash, density_flags, rows_to_arrays, score_events, write_result  # noqa: E402
from eval.harness import make_guard, result_tag, site_hash  # noqa: E402

from aisle.analytics import block  # noqa: E402

WINDOW_GRID = (15.0, 30.0, 60.0)


def density_arm(a, site, hz, params):
    return density_flags(a, site.floor, hz, window_s=params.get("window_s"))[0]


def walkway_arm(a, site, hz, params):
    """D-2 rule (frozen, no tuned parameter): BLOCK flags only; shelf lingerers become BROWSE and never count."""
    return block.as_scoring_flags(block.flags(a, site, hz))


ARMS = {"density": density_arm, "walkway": walkway_arm}


def tune(guard, site) -> dict:
    guard.phase = "tune"
    zb = {z.name: z for z in site.zones}
    data = [guard.load(s) for s in guard.tune]
    grid = {}
    for w in WINDOW_GRID:
        hb = nb = hw = nw = 0
        for d in data:
            r = score_events(density_flags(rows_to_arrays(d["rows"]), site.floor, d["hz"], window_s=w)[0], d["events"], zb)
            hb += r["block"]["flagged"]; nb += r["block"]["n"]; hw += r["browse"]["flagged"]; nw += r["browse"]["n"]
        grid[w] = hb / max(nb, 1) - hw / max(nw, 1)
    pick = max(grid, key=grid.get)
    return {"window_s": pick, "window_grid_tune_recall_minus_fa": grid, "tune_sessions": list(guard.tune)}


def score(guard, site, params, arms=None) -> dict:
    arms = arms or ARMS
    guard.phase = "score"
    zb = {z.name: z for z in site.zones}
    out = {}
    for s in guard.score:
        d = guard.load(s)
        a = rows_to_arrays(d["rows"])
        out[s] = {name: score_events(fn(a, site, d["hz"], params), d["events"], zb) for name, fn in arms.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="synth_aisle")
    ap.add_argument("--force-rescore", action="store_true")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    guard, site, ss = make_guard(a.site)
    params = tune(guard, site)
    import eval.common as _common
    rule = code_hash(block, _common, __file__)
    synth_only = all(isinstance(v, dict) and "synth_scenario" in v for v in (ss.get("sources") or {}).values())
    lock = check_and_lock(a.site, "+".join(guard.score) + "_events", a.force_rescore, {"script": "staged_events", "tuned": params},
                          rule_hash=rule, synthetic=synth_only)
    sc = score(guard, site, params)
    arms = {}
    for name in ARMS:
        nb = sum(v[name]["block"]["n"] for v in sc.values()); fb = sum(v[name]["block"]["flagged"] for v in sc.values())
        nw = sum(v[name]["browse"]["n"] for v in sc.values()); fw = sum(v[name]["browse"]["flagged"] for v in sc.values())
        arms[name] = {"block_recall": fb / max(nb, 1), "browse_false_alarm": fw / max(nw, 1), "n_block": nb, "n_browse": nw,
                      "blocks_flagged": fb, "browses_flagged": fw}
    d = arms["density"]
    res = {
        "data": "synthetic" if a.site == "synth_aisle" else f"team footage site {a.site}",
        "note": "On synth_aisle this proves the harness runs end to end; it is not block accuracy [A29].",
        "config": {"site": a.site, "density_rule": DENSITY_RULE, "window_grid": list(WINDOW_GRID), "walkway_rule": block.RULE},
        "seed": {s: (ss.get("sources") or {}).get(s) for s in guard.tune + guard.score}, "data_hash": site_hash(a.site),
        "sessions": {"tune": guard.tune, "score": guard.score}, "access_log": guard.log, "tuned": params,
        "n": {"block_events": d["n_block"], "browse_events": d["n_browse"]},
        "metric": {"arms": arms, "per_session": sc},
        "prior": {"flag_nothing": {"block_recall": 0.0, "browse_false_alarm": 0.0},
                  "flag_every_event": {"block_recall": 1.0, "browse_false_alarm": 1.0}},
        "headline": [{"metric": "block recall / browse false alarm (density baseline)", "arm": "density",
                      "value": d["block_recall"], "browse_false_alarm": d["browse_false_alarm"], "data": "synthetic" if a.site == "synth_aisle" else a.site},
                     {"metric": "block recall / browse false alarm (walkway rule)", "arm": "walkway",
                      "value": arms["walkway"]["block_recall"], "browse_false_alarm": arms["walkway"]["browse_false_alarm"],
                      "data": "synthetic" if a.site == "synth_aisle" else a.site}],
        "score_lock": lock,
    }
    write_result(f"staged_{result_tag(a.site)}", res)
    print(f"staged_{result_tag(a.site)}: density block recall {d['block_recall']:.3f}, browse FA {d['browse_false_alarm']:.3f}; walkway {arms['walkway']['block_recall']:.3f} / {arms['walkway']['browse_false_alarm']:.3f} (window {params['window_s']}s from tune; run #{lock['n_scored_runs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
