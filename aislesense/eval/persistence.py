"""Dead-zone persistence, ATC day 1 (2012-11-14) -> day 2 (2012-11-21) (X9 protocol, F8 guard).

Rule v2 frozen (aisle.analytics.deadzone). Grid origin fixed to the day-1 minimum so 1 m cells line up across days.
For each day-1 aware DEAD cell: still DEAD on day 2, UNSEEN on day 2, or busy on day 2 (>= day-2 median visitors over
observed cells). Day-2 DEAD cells not DEAD on day 1 are what the app shows as WATCH. Decile baseline beside it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.atc_deadzone import DAYS, NX, NY, cells, day_path, load_day  # noqa: E402
from eval.common import fingerprint, write_result  # noqa: E402

import numpy as np  # noqa: E402

from aisle.analytics import deadzone as dz  # noqa: E402

EXPECTED = {"day1_aware_dead": 118, "busy_on_day2": 23, "tolerance": 2}


def jac(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="same protocol; kept for run_all symmetry")
    ap.parse_args(argv)
    b0 = load_day(DAYS[0])
    X0, Y0 = float(b0.x.min()), float(b0.y.min())
    b1 = load_day(DAYS[1])
    (v0, t0), (v1, t1) = cells(b0, X0, Y0), cells(b1, X0, Y0)
    a0, a1 = dz.aware(v0, t0), dz.aware(v1, t1)
    n0, n1 = dz.decile_rule(v0), dz.decile_rule(v1)
    med1 = float(np.median(v1[a1["obs"]]))
    busy1 = v1 >= med1
    d0, d1 = a0["dead"], a1["dead"]
    st1, _ = dz.states(v1, t1, prev_dead=d0)
    aw = {"day1_dead": int(d0.sum()), "day2_dead": int(d1.sum()), "still_dead_day2": int((d0 & d1).sum()),
          "unseen_day2": int((d0 & a1["unseen"]).sum()), "busy_day2": int((d0 & busy1).sum()),
          "busy_day2_share": float((d0 & busy1).sum() / max(d0.sum(), 1)), "jaccard": jac(d0, d1),
          "day2_watch": int((st1 == "watch").sum()), "day2_dead_and_dead_day1": int((st1 == "dead").sum())}
    de = {"day1_dead": int(n0.sum()), "busy_day2": int((n0 & busy1).sum()), "jaccard": jac(n0, n1)}
    ok = abs(aw["busy_day2"] - EXPECTED["busy_on_day2"]) <= EXPECTED["tolerance"] and aw["day1_dead"] == EXPECTED["day1_aware_dead"]
    res = {
        "data": "ATC mall trajectories (real sensor tracks, no CV), 2012-11-14 -> 2012-11-21",
        "config": {"rule": dz.RULE, "origin": {"X0": X0, "Y0": Y0, "nx": NX, "ny": NY, "fixed_to": DAYS[0]},
                   "busy": "day-2 visitors >= day-2 median over observed cells", "day2_median_visitors": med1},
        "seed": None, "data_hash": fingerprint([day_path(d) for d in DAYS]),
        "n": {"day1_aware_dead": aw["day1_dead"], "day1_decile_dead": de["day1_dead"]},
        "metric": {"aware": aw, "decile": de, "unseen_jaccard": jac(a0["unseen"], a1["unseen"])},
        "prior": {"name": "decile rule (the common baseline)", **de},
        "headline": [{"metric": "day-1 aware DEAD cells busy one week later", "arm": "aware", "value": aw["busy_day2_share"],
                      "count": aw["busy_day2"], "of": aw["day1_dead"], "unit": "1 m cells"},
                     {"metric": "day-2 DEAD cells shown as WATCH (not dead the week before)", "arm": "aware",
                      "count": aw["day2_watch"], "of": aw["day2_dead"]}],
        "expected_X9": EXPECTED, "matches_expected": bool(ok),
    }
    write_result("persistence", res)
    print(f"persistence: day-1 aware DEAD {aw['day1_dead']}; busy on day 2 {aw['busy_day2']} ({aw['busy_day2_share']:.3f}), "
          f"still dead {aw['still_dead_day2']}, unseen {aw['unseen_day2']}; day-2 WATCH {aw['day2_watch']}/{aw['day2_dead']}; "
          f"decile busy {de['busy_day2']}/{de['day1_dead']}; expected 23+-2 -> {'ok' if ok else 'MISS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
