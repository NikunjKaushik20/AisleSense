"""Tips + before/after demo on synthetic sessions (data: synthetic; shows what the tips and diff say, not accuracy).

before = synth/scenarios/block_eval.yaml seed 0 (walkway blocks + shelf browses + traffic).
after  = the same scenario and seed with the walkway blocks removed (n_block: 0), i.e. "carts moved out of the walkway".
Both go through the app code: block.flags, deadzone.analyse, dwell.payload, tips.from_parts, diff.compare.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import fingerprint, write_result  # noqa: E402

from aisle.analytics import block, deadzone as dz, diff, dwell, tips  # noqa: E402
from aisle.schema import Event, load_site  # noqa: E402
from synth.labelled import SCEN_DIR, generate_labelled, load  # noqa: E402

SEED = 0


def session_tips(site, rows, hz, sid):
    fl = block.flags(rows, site, hz)
    st = dz.analyse(site, rows)["state"]
    ev = [Event(session=sid, cam="synth", t=t, track=str(k), x=x, y=y, conf=1.0) for t, k, x, y in rows]
    dw = dwell.payload(site, ev, sid, hz=hz)
    return fl, tips.from_parts(site, st, fl, dw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.parse_args(argv)
    site = load_site("synth_aisle")
    sc = load("block_eval")
    hz = float(sc.get("hz", 5))
    rb, eb, _ = generate_labelled(sc, site, SEED)
    ra, ea, _ = generate_labelled({**sc, "n_block": 0}, site, SEED)
    fb, tb = session_tips(site, rb, hz, "before")
    fa, ta = session_tips(site, ra, hz, "after")
    d = diff.compare(tb, ta)
    cnt = lambda fl, t: sum(f["type"] == t for f in fl)  # noqa: E731
    res = {
        "data": "synthetic (labelled generator rows, not real people)",
        "note": "Demo of tip and diff output; never claimed as accuracy [A29].",
        "config": {"scenario": "block_eval", "seed": SEED, "after": "same scenario and seed with n_block: 0",
                   "walkway_rule": block.RULE["text"], "deadzone_rule": dz.RULE["text"],
                   "dwell_ratio": tips.DWELL_RATIO, "min_dead_cells": tips.MIN_DEAD_CELLS},
        "seed": SEED, "data_hash": fingerprint([SCEN_DIR / "block_eval.yaml"]),
        "n": {"before_rows": len(rb), "after_rows": len(ra), "before_staged_blocks": sum(e["type"] == "block" for e in eb),
              "after_staged_blocks": sum(e["type"] == "block" for e in ea)},
        "metric": {"before": {"block_flags": cnt(fb, "BLOCK"), "browse_flags": cnt(fb, "BROWSE"), "tips": tb},
                   "after": {"block_flags": cnt(fa, "BLOCK"), "browse_flags": cnt(fa, "BROWSE"), "tips": ta},
                   "diff": d},
        "prior": None,
        "headline": [{"metric": "BLOCK tips cleared after removing walkway blocks", "data": "synthetic",
                      "count": sum(x["kind"] == "BLOCK" for x in d["cleared"]),
                      "before_block_flags": cnt(fb, "BLOCK"), "after_block_flags": cnt(fa, "BLOCK")}],
    }
    write_result("tips_demo", res)
    print(f"tips_demo: before {cnt(fb, 'BLOCK')} BLOCK / {cnt(fb, 'BROWSE')} BROWSE flags, {len(tb)} tips; after "
          f"{cnt(fa, 'BLOCK')} BLOCK, {len(ta)} tips; cleared {[(x['kind'], x['zone']) for x in d['cleared']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
