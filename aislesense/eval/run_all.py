"""Run every eval script and write results/summary.json: every headline metric with its prior or baseline beside it.

  python eval/run_all.py --fast     (cached tracks for MERL / MOT20; seconds to minutes; CPU only)
  python eval/run_all.py            (re-tracks MOT20 from det.txt; MERL still from the dump - use merl_dwell.py --clips N for live CV)

The synthetic harness sessions (synth_aisle) are re-scored on every run with --force-rescore, which each result
records; team sites are never re-scored by run_all. summary.json has no timestamps or lock counters, so a rerun with
the same seeds reproduces it byte for byte.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eval.common import read_result, write_result  # noqa: E402

STEPS = [
    ("merl_dwell", ["eval/merl_dwell.py", "--from-dump"]),
    ("mot20_dwell", ["eval/mot20_dwell.py"]),
    ("atc_deadzone", ["eval/atc_deadzone.py"]),
    ("synth_block", ["eval/synth_block.py"]),
    ("persistence", ["eval/persistence.py"]),  # M4: ATC day 1 -> day 2 dead-zone persistence
    ("tips_demo", ["eval/tips_demo.py"]),  # M4: tips + before/after diff on synthetic sessions
    ("stopwatch_synth", ["eval/stopwatch.py", "--site", "synth_aisle", "--force-rescore"]),
    ("staged_synth", ["eval/staged_events.py", "--site", "synth_aisle", "--force-rescore"]),
    ("swaps", ["eval/swaps.py"]),  # M3: one row per arm (baseline, +S1, +S1+S2, +S1+S2+S3), prior beside each
]


def build_summary() -> dict:
    m = read_result("merl_dwell"); mo = read_result("mot20_dwell"); at = read_result("atc_deadzone")
    sb = read_result("synth_block"); sw = read_result("stopwatch_synth"); se = read_result("staged_synth")
    M = m["metric"]; MO = mo["metric"]
    rows = [
        {"id": "merl_frame_browse_F1", "file": "merl_dwell.json", "data": "MERL test 28 clips (real)", "unit": "frame F1",
         "value": M["frame_browse"]["tracked"]["F1"], "prior": M["frame_browse"]["prior_all_browse"]["F1"], "prior_name": "all-browse"},
        {"id": "merl_ids_per_clip", "file": "merl_dwell.json", "data": "MERL test (1 shopper per clip)", "unit": "IDs per clip",
         "value": M["ids_per_clip"]["mean"], "prior": 1, "prior_name": "truth"},
        {"id": "merl_longest_id_MAE", "file": "merl_dwell.json", "data": "MERL test", "unit": "s per clip (browse seconds)",
         "value": M["per_clip_browse_s"]["longest_id"]["MAE"], "prior": M["per_clip_browse_s"]["prior"]["MAE"], "prior_name": "val share x clip length"},
        {"id": "merl_tracked_time_MAE", "file": "merl_dwell.json", "data": "MERL test", "unit": "s per clip (browse seconds)",
         "value": M["per_clip_browse_s"]["tracked_time"]["MAE"], "prior": M["per_clip_browse_s"]["prior"]["MAE"], "prior_name": "val share x clip length"},
        {"id": "mot20_span_MAE", "file": "mot20_dwell.json", "data": "MOT20 train 4 seqs (station crowd)", "unit": "s per GT visit >= 5 s",
         "value": MO["span_MAE_s"], "prior": mo["prior"]["span_MAE_s"], "prior_name": "LOSO global median"},
        {"id": "mot20_long_dweller_AUROC", "file": "mot20_dwell.json", "data": "MOT20 train", "unit": "AUROC (GT >= 15 s)",
         "value": MO["AUROC_ge15s"], "prior": 0.5, "prior_name": "constant"},
        {"id": "atc_blind_decile_dead", "file": "atc_deadzone.json", "data": "ATC 2 days, blind injection, 10 seeds", "unit": "share of injected busy cells called DEAD",
         "value": at["metric"]["blind"]["decile"]["dead_rate_pooled"], "count": at["metric"]["blind"]["decile"]["dead_pooled"], "of": at["metric"]["blind"]["cells_pooled"],
         "prior": None, "prior_name": "this is the baseline (aware arm at M4)"},
        {"id": "atc_ignored_decile_dead", "file": "atc_deadzone.json", "data": "ATC 2 days, ignored injection, 10 seeds", "unit": "share of ignored cells called DEAD",
         "value": at["metric"]["ignored"]["decile"]["dead_rate_pooled"], "prior": None, "prior_name": "this is the baseline (aware arm at M4)"},
        {"id": "atc_day1_decile_on_edge", "file": "atc_deadzone.json", "data": "ATC 2012-11-14", "unit": "share of decile DEAD cells on coverage edge",
         "value": at["metric"]["per_day_baseline"]["20121114"]["decile_dead_on_edge_share"], "prior": None, "prior_name": "baseline"},
        {"id": "synth_density_browse_fa", "file": "synth_block.json", "data": "synthetic", "unit": "share of shelf browses flagged as bottleneck",
         "value": sb["metric"]["density"]["browse_false_alarm"], "prior": None, "prior_name": "baseline (walkway arm at M4)"},
        {"id": "synth_density_block_recall", "file": "synth_block.json", "data": "synthetic", "unit": "share of walkway blocks flagged",
         "value": sb["metric"]["density"]["block_recall"], "prior": None, "prior_name": "baseline (walkway arm at M4)"},
        {"id": "stopwatch_synth_MAE", "file": "stopwatch_synth.json", "data": "synthetic sessions (harness proof)", "unit": "s per section mean dwell",
         "value": sw["metric"]["tracker_MAE_s"], "prior": sw["prior"]["MAE_s"], "prior_name": sw["prior"]["name"]},
        {"id": "staged_synth_density_block_recall", "file": "staged_synth.json", "data": "synthetic sessions (harness proof)", "unit": "share of staged blocks flagged",
         "value": se["metric"]["arms"]["density"]["block_recall"], "browse_false_alarm": se["metric"]["arms"]["density"]["browse_false_alarm"],
         "prior": 0.0, "prior_name": "flag nothing (flag-all gives 1.0 / FA 1.0)"},
    ]
    for r in rows:
        if r["prior"] is not None and r["id"] not in ("merl_ids_per_clip",):
            higher_better = "F1" in r["unit"] or "AUROC" in r["unit"] or "recall" in r["id"]
            r["beats_prior"] = (r["value"] > r["prior"]) if higher_better else (r["value"] < r["prior"])
    honest = [r["id"] for r in rows if r.get("beats_prior") is False]
    return {
        "milestone": "M2 baseline (ByteTrack buffer 30, no ghost hold, decile dead zones, density bottlenecks)",
        "metrics": rows,
        "baseline_loses_to_prior": honest,
        "statement": "On public data the baseline does not beat the trivial prior on frame browse F1 (MERL), per-clip dwell (MERL) "
                     "or per-visit dwell (MOT20); it beats the constant on long-dweller ranking (MOT20 AUROC).",
        "force_rescore_synth": {"stopwatch_synth": sw["score_lock"]["force_rescore"], "staged_synth": se["score_lock"]["force_rescore"]},
        "swaps": swaps_summary(),
    }


def swaps_summary() -> dict | None:
    """M3 arms from results/swaps.json, copied (not recomputed), with its generated statement."""
    try:
        w = read_result("swaps")
    except FileNotFoundError:
        return None
    return {
        "file": "swaps.json",
        "arms": [{"arm": r["arm"], "mot20_span_MAE_s": r["mot20"]["span_MAE_s"], "mot20_prior_MAE_s": r["mot20_prior"]["span_MAE_s"],
                  "mot20_minus_prior_CI95": r["mot20"]["minus_prior_CI95"], "mot20_AUROC_ge15s": r["mot20"]["AUROC_ge15s"],
                  "merl_frame_F1": r["merl"]["frame_browse"]["F1"], "merl_prior_F1": r["merl_prior"]["frame_browse_F1"],
                  "merl_per_clip_MAE_s": r["merl"]["per_clip_browse_s"]["MAE"], "merl_prior_MAE_s": r["merl_prior"]["per_clip_MAE_s"],
                  "synth_section_MAE_s": r["synth"]["section_MAE_s"], "synth_prior_MAE_s": r["synth_prior"]["section_MAE_s"]}
                 for r in w["arms"]],
        "any_arm_beats_prior_per_visit": w["any_arm_beats_prior_per_visit"],
        "statement": w["statement"],
        "fallback_applied": w["fallback"]["applied"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    py = sys.executable
    failed = []
    for name, cmd in STEPS:
        args = [py, *cmd] + (["--fast"] if a.fast else [])
        t = time.time()
        r = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True)
        last = (r.stdout.strip().splitlines() or [""])[-1]
        print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {name} {time.time() - t:.0f}s  {last}", flush=True)
        if r.returncode != 0:
            failed.append(name)
            print(r.stderr[-2000:], file=sys.stderr)
    if failed:
        print(f"run_all: FAILED {failed}")
        return 1
    s = build_summary()
    write_result("summary", s)
    print(f"run_all: {len(s['metrics'])} metrics -> results/summary.json; baseline loses to prior on {s['baseline_loses_to_prior']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
