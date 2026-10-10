"""Stopwatch dwell harness (frozen protocol [A34]): section dwell from tracks vs hand stopwatch, prior beside it.

Tune (tune sessions only): prior = mean stopwatch dwell per section; choose the per-ID timer's minimum visit length
from {0, 1, 2, 5} s by per-section MAE. Score (score sessions only, once; lock in var/locks/): per-section mean dwell
from the baseline per-ID timer (track runs inside a section polygon) vs the stopwatch mean, and the tune prior's error.
Unit: absolute error of the per-section mean dwell, averaged over sections with >= 1 stopwatch visit.

  python eval/stopwatch.py --site synth_aisle      (now; synthetic sessions with tracker-like ID breaks)
  python eval/stopwatch.py --site team             (after filming; same code)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import check_and_lock, code_hash, write_result  # noqa: E402
from eval.harness import make_guard, result_tag, site_hash  # noqa: E402

import numpy as np  # noqa: E402

from synth.labelled import section_visits  # noqa: E402

MIN_VISIT_GRID = (0.0, 1.0, 2.0, 5.0)


def section_means(visits) -> dict:
    by = {}
    for v in visits:
        by.setdefault(v["section"], []).append(v["t_out"] - v["t_in"])
    return {k: float(np.mean(x)) for k, x in sorted(by.items())}


def errors(est: dict, truth: dict, fallback: float) -> dict:
    return {s: abs(est.get(s, fallback) - t) for s, t in truth.items()}


def per_id_timer(rows, site, hz, min_visit_s):
    return section_visits(rows, site, min_visit_s, hz)


def tune(guard, site) -> dict:
    """Reads only tune sessions (guard enforces it)."""
    guard.phase = "tune"
    data = [guard.load(s) for s in guard.tune]
    sw = [v for d in data for v in d["stopwatch"]]
    prior = section_means(sw)
    prior_global = float(np.mean([v["t_out"] - v["t_in"] for v in sw])) if sw else 0.0
    grid = {}
    for mv in MIN_VISIT_GRID:
        errs = []
        for d in data:
            est = section_means(per_id_timer(d["rows"], site, d["hz"], mv))
            errs += list(errors(est, section_means(d["stopwatch"]), 0.0).values())
        grid[mv] = float(np.mean(errs)) if errs else float("inf")
    pick = min(grid, key=grid.get)
    return {"prior_per_section": prior, "prior_global": prior_global, "min_visit_grid_tune_MAE": grid, "min_visit_s": pick,
            "tune_sessions": list(guard.tune)}


def score(guard, site, params) -> dict:
    guard.phase = "score"
    out = {}
    for s in guard.score:
        d = guard.load(s)
        truth = section_means(d["stopwatch"])
        tv = per_id_timer(d["rows"], site, d["hz"], params["min_visit_s"])
        est = section_means(tv)
        e_est = errors(est, truth, 0.0)
        e_pri = errors(params["prior_per_section"], truth, params["prior_global"])
        n_true = {k: sum(v["section"] == k for v in d["stopwatch"]) for k in truth}
        n_est = {k: sum(v["section"] == k for v in tv) for k in truth}
        out[s] = {"per_section": {k: {"stopwatch_mean_s": truth[k], "tracker_mean_s": est.get(k), "prior_mean_s": params["prior_per_section"].get(k, params["prior_global"]),
                                      "n_stopwatch": n_true[k], "n_tracker_visits": n_est[k]} for k in truth},
                  "tracker_MAE_s": float(np.mean(list(e_est.values()))), "prior_MAE_s": float(np.mean(list(e_pri.values()))),
                  "n_sections": len(truth), "n_stopwatch_visits": len(d["stopwatch"])}
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
    import synth.labelled as _lab
    synth_only = all(isinstance(v, dict) and "synth_scenario" in v for v in (ss.get("sources") or {}).values())
    lock = check_and_lock(a.site, "+".join(guard.score), a.force_rescore, {"script": "stopwatch", "tuned": {"min_visit_s": params["min_visit_s"]}},
                          rule_hash=code_hash(_common, _lab, __file__), synthetic=synth_only)
    sc = score(guard, site, params)
    tr = float(np.mean([v["tracker_MAE_s"] for v in sc.values()]))
    pr = float(np.mean([v["prior_MAE_s"] for v in sc.values()]))
    res = {
        "data": "synthetic" if a.site == "synth_aisle" else f"team footage site {a.site}",
        "config": {"site": a.site, "protocol": "stopwatch per section [A34]", "estimator": "baseline per-ID timer (track runs in section polygon)",
                   "min_visit_grid": list(MIN_VISIT_GRID), "unit": "abs error of per-section mean dwell, mean over sections"},
        "seed": {s: (ss.get("sources") or {}).get(s) for s in guard.tune + guard.score}, "data_hash": site_hash(a.site),
        "sessions": {"tune": guard.tune, "score": guard.score}, "access_log": guard.log, "tuned": params,
        "n": {s: {"sections": v["n_sections"], "stopwatch_visits": v["n_stopwatch_visits"]} for s, v in sc.items()},
        "metric": {"tracker_MAE_s": tr, "per_session": sc},
        "prior": {"name": "tune-session stopwatch mean per section", "MAE_s": pr},
        "headline": [{"metric": "section mean dwell MAE (per-ID timer)", "value": tr, "prior": pr, "unit": "s per section",
                      "beats_prior": tr < pr}],
        "score_lock": lock,
    }
    write_result(f"stopwatch_{result_tag(a.site)}", res)
    print(f"stopwatch_{result_tag(a.site)}: tracker MAE {tr:.2f}s vs prior {pr:.2f}s (min_visit {params['min_visit_s']}s from tune; run #{lock['n_scored_runs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
