"""M4 numbers from results/*.json against the plan (BUILD.md M4 'Expected numbers', X9/X11). Run the evals first."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.json"
    if not p.exists():
        pytest.skip(f"{p.name} missing: run eval/{name}.py")
    return json.loads(p.read_text(encoding="utf-8"))


def test_atc_aware_blind_zero_of_656():
    m = load("atc_deadzone")["metric"]
    bl, ig = m["blind"], m["ignored"]
    assert bl["cells_pooled"] == 656
    assert bl["aware"]["dead_pooled"] == 0 and bl["aware"]["per_seed_dead_range"] == [0, 0]
    assert all(v["dead"] == 0 for v in bl["aware"]["per_day"].values())
    assert abs(bl["aware_unseen_rate_pooled"] - 0.939) <= 0.01
    assert abs(bl["decile"]["dead_rate_pooled"] - 0.634) <= 0.01


def test_atc_aware_ignored_keeps_sensitivity():
    ig = load("atc_deadzone")["metric"]["ignored"]
    assert abs(ig["aware"]["dead_rate_pooled"] - 0.625) <= 0.01
    assert abs(ig["decile"]["dead_rate_pooled"] - 0.655) <= 0.01
    assert set(ig["aware"]["per_day"]) == {"20121114", "20121121"}


def test_persistence_23_of_118():
    a = load("persistence")["metric"]["aware"]
    assert a["day1_dead"] == 118 and abs(a["busy_day2"] - 23) <= 2


def test_synth_walkway_rule_beats_density():
    m = load("synth_block")["metric"]
    assert m["walkway"]["block_recall"] >= 0.9 and m["walkway"]["browse_false_alarm"] <= 0.1
    assert m["density"]["browse_false_alarm"] >= 0.5


def test_tips_demo_written():
    d = load("tips_demo")
    assert d["metric"]["diff"]["cleared"] and d["data"].startswith("synthetic")
