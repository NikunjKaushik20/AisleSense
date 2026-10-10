"""Write sites/synth_aisle/labels/{stopwatch,events}.csv from the generator truth of each session in sessions.yaml.

Run once (labels are committed; eval scripts read only the CSVs, as they will for team footage):
  python synth/make_labels.py [--site synth_aisle]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import yaml  # noqa: E402

from aisle.schema import SITES_DIR, load_site  # noqa: E402
from synth.labelled import generate_labelled, load  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="synth_aisle")
    a = ap.parse_args(argv)
    site = load_site(a.site)
    ss = yaml.safe_load((SITES_DIR / a.site / "sessions.yaml").read_text(encoding="utf-8"))
    out = SITES_DIR / a.site / "labels"
    out.mkdir(parents=True, exist_ok=True)
    sw, ev = [], []
    for sess in list(ss.get("tune") or []) + list(ss.get("score") or []):
        src = ss["sources"][sess]
        _, events, visits = generate_labelled(load(src["synth_scenario"]), site, int(src["seed"]))
        sw += [(sess, v["section"], v["person"], round(v["t_in"], 2), round(v["t_out"], 2)) for v in visits]
        ev += [(sess, e["type"], round(e["t0"], 2), round(e["t1"], 2), e["zone"]) for e in events]
    with open(out / "stopwatch.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "section", "person", "t_in", "t_out"]); w.writerows(sw)
    with open(out / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["session", "type", "t0", "t1", "zone"]); w.writerows(ev)
    print(f"labels: {len(sw)} stopwatch visits, {len(ev)} staged events -> {out}")


if __name__ == "__main__":
    main()
