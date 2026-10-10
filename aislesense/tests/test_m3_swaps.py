"""M3 swaps: numbers reproduce X5/X8 within tolerance, swap parameters come from val/LOSO only, honest summary.
Needs `python eval/swaps.py --fast` (and `eval/run_all.py --fast` for summary.json) to have run."""
import json

import pytest

from aisle.cv import ghost, track
from aisle.schema import RESULTS_DIR


def load(n):
    return json.loads((RESULTS_DIR / f"{n}.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sw():
    return load("swaps")


def arm(sw, name):
    return next(r for r in sw["arms"] if r["arm"] == name)


def test_one_row_per_arm_each_with_prior(sw):
    assert [r["arm"] for r in sw["arms"]] == ["baseline", "+S1", "+S1+S2", "+S1+S2+S3"]
    for r in sw["arms"]:
        for ds in ("mot20", "merl", "synth"):
            assert r[f"{ds}_prior"], (r["arm"], ds)
        assert r["mot20_prior"]["span_MAE_s"] == pytest.approx(3.96, abs=0.05)


def test_mot20_numbers_X5(sw):
    assert arm(sw, "baseline")["mot20"]["span_MAE_s"] == pytest.approx(4.70, abs=0.1)
    s1 = arm(sw, "+S1")["mot20"]
    assert s1["span_MAE_s"] == pytest.approx(4.10, abs=0.1)
    assert s1["AUROC_ge15s"] == pytest.approx(0.754, abs=0.02)
    assert set(s1["buffer"].values()) == {200}


def test_merl_numbers_X8(sw):
    h = arm(sw, "+S1+S2")["merl"]
    p = arm(sw, "+S1+S2")["merl_prior"]
    assert h["frame_browse"]["F1"] == pytest.approx(0.877, abs=0.01)
    assert p["frame_browse_F1"] == pytest.approx(0.866, abs=0.01)
    assert h["per_clip_browse_s"]["MAE"] == pytest.approx(9.3, abs=1.0)
    assert p["per_clip_MAE_s"] == pytest.approx(6.5, abs=0.1)
    assert arm(sw, "baseline")["merl"]["frame_browse"]["F1"] == pytest.approx(0.787, abs=0.01)
    assert sw["merl_n"]["test_clips"] == 28


def test_params_from_val_or_loso_never_scored_split(sw):
    from eval import merl_dwell as md
    from eval import swaps
    g = sw["params"]["S2_ghost"]
    assert g["chosen_on"] == "MERL val subjects 21-26"
    assert g["val_clips"] and all(md.split_of(c) == "val" for c in g["val_clips"])
    assert (g["pick"]["ttl_s"], g["pick"]["k"]) == (ghost.TTL_S, ghost.K) == (8.0, 1.0)
    assert sw["params"]["S3_merge_ttl"]["ttl_s"] == ghost.TTL_S
    # the chooser refuses a test clip
    with pytest.raises(AssertionError):
        swaps.choose_ghost([{"clip": "27_1", "split": "test", "fi": None, "g": None, "rows": [], "exited": set(), "t_end": 0}])
    # S1: each held-out sequence's buffer is picked on the other sequences, and the app default is that pick
    lo = sw["params"]["S1_buffer"]
    assert "leave-one-sequence-out" in lo["chosen_on"]
    for seq, v in lo["loso"].items():
        assert int(min(v["other_seq_MAE"], key=v["other_seq_MAE"].get)) == v["pick"], seq
    assert {v["pick"] for v in lo["loso"].values()} == {track.DEFAULT_BUFFER}
    # synth: tuning read only the tune session, scoring only the score session
    sd = sw["synth_detail"]
    assert sd["sessions"] == {"tune": ["s1"], "score": ["s2"]}
    assert all((ph == "tune" and s == "s1") or (ph == "score" and s == "s2") for ph, s in sd["access_log"])


def test_honest_framing_in_swaps_and_summary(sw):
    assert sw["any_arm_beats_prior_per_visit"] is False
    assert not any(v["merl_per_clip"] for v in sw["per_visit_beats_prior"].values())
    for r in sw["arms"]:  # a 'beat' needs the whole CI below 0
        assert sw["per_visit_beats_prior"][r["arm"]]["mot20_span"] == (r["mot20"]["minus_prior_CI95"][1] < 0)
    s = load("summary")
    assert s["swaps"] is not None and s["swaps"]["any_arm_beats_prior_per_visit"] is False
    assert "no arm beats" in s["swaps"]["statement"].lower()
    assert s["swaps"]["statement"] == sw["statement"]


def test_fallback_decision_recorded(sw):
    f = sw["fallback"]
    assert f["applied"] == (f["synth_S1S2_MAE_s"] > f["synth_S1_MAE_s"])
    assert 0 <= f["stitch_precision"] <= 1
