"""M2: the baseline numbers in results/*.json match the plan's expected values (BUILD.md M2) within tolerance.

Reads the JSON files written by `python eval/run_all.py --fast`; it never recomputes or edits them."""
import json
from pathlib import Path

import pytest

R = Path(__file__).resolve().parent.parent / "results"


def load(name):
    p = R / f"{name}.json"
    if not p.exists():
        pytest.fail(f"{p} missing: run `python eval/run_all.py --fast` first")
    return json.loads(p.read_text(encoding="utf-8"))


def test_all_m2_files_exist_and_carry_config_seed_hash_metric_prior_n():
    for n in ["merl_dwell", "mot20_dwell", "atc_deadzone", "synth_block", "stopwatch_synth", "staged_synth"]:
        d = load(n)
        for k in ("config", "seed", "data_hash", "metric", "prior", "n"):
            assert k in d, f"{n}.json lacks {k}"
    s = load("summary")
    assert len(s["metrics"]) >= 10
    for r in s["metrics"]:
        assert "prior" in r and "prior_name" in r, r["id"]


def test_merl_baseline_matches_X1_X8():
    m = load("merl_dwell")["metric"]
    assert m["frame_browse"]["tracked"]["F1"] == pytest.approx(0.787, abs=0.01)
    assert m["frame_browse"]["prior_all_browse"]["F1"] == pytest.approx(0.866, abs=0.005)
    assert m["ids_per_clip"]["mean"] == pytest.approx(14.64, abs=0.5)
    assert m["per_clip_browse_s"]["longest_id"]["MAE"] == pytest.approx(74.1, abs=2)
    assert m["per_clip_browse_s"]["prior"]["MAE"] == pytest.approx(6.5, abs=0.5)
    assert load("merl_dwell")["n"]["test_clips"] == 28


def test_mot20_baseline_matches_X5():
    d = load("mot20_dwell")
    assert d["config"]["tracker"]["track_buffer"] == 30
    assert d["metric"]["span_MAE_s"] == pytest.approx(4.70, abs=0.1)
    assert d["prior"]["span_MAE_s"] == pytest.approx(3.96, abs=0.05)
    assert d["metric"]["AUROC_ge15s"] == pytest.approx(0.712, abs=0.02)
    assert d["n"]["gt_visits"] == 3615


def test_atc_decile_matches_X11_X2():
    d = load("atc_deadzone")["metric"]
    assert d["blind"]["cells_pooled"] == 656
    assert d["blind"]["decile"]["dead_rate_pooled"] == pytest.approx(0.634, abs=0.01)
    assert d["ignored"]["decile"]["dead_rate_pooled"] == pytest.approx(0.655, abs=0.01)
    assert d["per_day_baseline"]["20121114"]["decile_dead_on_edge_share"] == pytest.approx(0.958, abs=0.01)


def test_synth_density_rule_cannot_tell_browse_from_block():
    d = load("synth_block")
    assert d["data"] == "synthetic"
    den = d["metric"]["density"]
    assert den["n_block"] >= 20 and den["n_browse"] >= 20
    assert den["browse_false_alarm"] >= 0.5  # F5 sanity check, not evidence


def test_harness_results_on_synth():
    sw, se = load("stopwatch_synth"), load("staged_synth")
    assert sw["data"] == se["data"] == "synthetic"
    assert sw["sessions"] == {"tune": ["s1"], "score": ["s2"]}
    assert "MAE_s" in sw["prior"] and sw["metric"]["tracker_MAE_s"] >= 0
    assert se["n"]["block_events"] >= 20 and se["n"]["browse_events"] >= 20


def test_summary_states_baseline_vs_prior_honestly():
    s = load("summary")
    ids = {r["id"]: r for r in s["metrics"]}
    for k in ("merl_frame_browse_F1", "merl_longest_id_MAE", "mot20_span_MAE"):
        assert ids[k]["beats_prior"] is False and k in s["baseline_loses_to_prior"]
    assert ids["mot20_long_dweller_AUROC"]["beats_prior"] is True
