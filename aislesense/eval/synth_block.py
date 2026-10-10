"""Synthetic block-vs-browse sanity check (data: synthetic, never claimed as block accuracy [A29]).

Generator synth/scenarios/block_eval.yaml on sites/synth_aisle: 24 walkway blocks (2-3 lingerers in the walkway while
>= 3 passers squeeze past) + 24 shelf-front browses + plain traffic, 5 seeds. Arms score the same rows:
  density (baseline): eval.common.density_flags (top-decile person-seconds AND slow, per 1 m x 30 s cell-window).
  walkway (M4, D-2): aisle.analytics.block BLOCK flags (lingerer in walkway + >= 2 movers within 2 m; shelf lingerers -> BROWSE).
Block recall = share of block events with a bottleneck flag on them; browse false alarm = share of browse events
flagged as a bottleneck (the density rule cannot tell them apart, F5 / X2).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import density_flags, fingerprint, rows_to_arrays, score_events, write_result  # noqa: E402

import numpy as np  # noqa: E402

from aisle.analytics import block  # noqa: E402
from aisle.schema import load_site  # noqa: E402
from synth.labelled import SCEN_DIR, generate_labelled, load  # noqa: E402

SEEDS = range(5)


def density_arm(a, site, hz):
    return density_flags(a, site.floor, hz)[0]


def walkway_arm(a, site, hz):
    return block.as_scoring_flags(block.flags(a, site, hz))


ARMS = {"density": density_arm, "walkway": walkway_arm}


def run(arms=None, seeds=SEEDS, scenario="block_eval") -> dict:
    arms = arms or ARMS
    site = load_site("synth_aisle")
    sc = load(scenario) if isinstance(scenario, str) else dict(scenario)
    hz = float(sc.get("hz", 5))
    zb = {z.name: z for z in site.zones}
    per_seed = []
    for seed in seeds:
        rows, events, _ = generate_labelled(sc, site, seed)
        a = rows_to_arrays(rows)
        rec = {"seed": seed, "n_rows": len(rows), "n_block": sum(e["type"] == "block" for e in events),
               "n_browse": sum(e["type"] == "browse" for e in events)}
        for name, fn in arms.items():
            rec[name] = score_events(fn(a, site, hz), events, zb)
        per_seed.append(rec)
    out = {}
    for name in arms:
        nb = sum(r[name]["block"]["n"] for r in per_seed)
        fb = sum(r[name]["block"]["flagged"] for r in per_seed)
        nw = sum(r[name]["browse"]["n"] for r in per_seed)
        fw = sum(r[name]["browse"]["flagged"] for r in per_seed)
        out[name] = {"block_recall": fb / max(nb, 1), "browse_false_alarm": fw / max(nw, 1), "n_block": nb, "n_browse": nw,
                     "blocks_flagged": fb, "browses_flagged": fw,
                     "flags_outside_events": int(sum(r[name]["flags_outside_events"] for r in per_seed)),
                     "per_seed_browse_fa": [r[name]["browse"]["rate"] for r in per_seed],
                     "per_seed_block_recall": [r[name]["block"]["rate"] for r in per_seed]}
    return {"arms": out, "per_seed": per_seed, "scenario": sc}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="same 5 seeds (the run takes seconds)")
    ap.parse_args(argv)
    o = run()
    d = o["arms"]["density"]
    w = o["arms"]["walkway"]
    v1 = run(arms={"density": density_arm}, scenario={**load("block_eval"), "browse_lingerers": [1, 3]})["arms"]["density"]
    res = {
        "data": "synthetic",
        "note": "Sanity check, not evidence: proves what each rule does on staged synthetic events; never claimed as block accuracy [A29].",
        "config": {"scenario": "synth/scenarios/block_eval.yaml", "site": "synth_aisle", "seeds": list(SEEDS),
                   "density_rule": __import__("eval.common", fromlist=["DENSITY_RULE"]).DENSITY_RULE,
                   "walkway_rule": block.RULE, "scenario_values": o["scenario"]},
        "seed": list(SEEDS), "data_hash": fingerprint([SCEN_DIR / "block_eval.yaml", Path(__file__).resolve().parent.parent / "sites/synth_aisle/zones.geojson"]),
        "n": {"block_events": d["n_block"], "browse_events": d["n_browse"]},
        "metric": o["arms"],
        "prior": {"name": "density rule (baseline)", "block_recall": d["block_recall"], "browse_false_alarm": d["browse_false_alarm"]},
        "headline": [{"metric": "shelf browses flagged as bottleneck by density rule", "arm": "density", "value": d["browse_false_alarm"],
                      "count": d["browses_flagged"], "of": d["n_browse"], "data": "synthetic"},
                     {"metric": "walkway blocks flagged by density rule", "arm": "density", "value": d["block_recall"],
                      "count": d["blocks_flagged"], "of": d["n_block"], "data": "synthetic"},
                     {"metric": "shelf browses flagged as bottleneck by walkway rule", "arm": "walkway", "value": w["browse_false_alarm"],
                      "count": w["browses_flagged"], "of": w["n_browse"], "data": "synthetic"},
                     {"metric": "walkway blocks flagged by walkway rule", "arm": "walkway", "value": w["block_recall"],
                      "count": w["blocks_flagged"], "of": w["n_block"], "data": "synthetic"}],
        "per_seed": o["per_seed"],
        "scenario_history": [{"version": "v1 (browse groups 1-3 people, block groups 2-3)", "status": "superseded",
                              "why": "v1 confounded group size with location; v2 matches browse group size to block (2-3) so only "
                                     "location differs. Changed after seeing v1; both kept here.",
                              "density": {k: v1[k] for k in ("block_recall", "browse_false_alarm", "n_block", "n_browse")}}],
    }
    write_result("synth_block", res)
    print(f"synth_block: density block recall {d['block_recall']:.3f} ({d['blocks_flagged']}/{d['n_block']}), "
          f"browse FA {d['browse_false_alarm']:.3f} ({d['browses_flagged']}/{d['n_browse']}) | walkway block recall "
          f"{w['block_recall']:.3f} ({w['blocks_flagged']}/{w['n_block']}), browse FA {w['browse_false_alarm']:.3f} ({w['browses_flagged']}/{w['n_browse']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
