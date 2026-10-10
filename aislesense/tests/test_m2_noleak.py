"""M2 leakage discipline: tuning never reads a score session, scored runs lock, MERL test never picks parameters,
labels never reach a rule."""
import json
from pathlib import Path

import pytest

from eval import common, staged_events, stopwatch
from eval.common import LeakError, SessionGuard, check_and_lock
from eval.harness import make_guard

ROOT = Path(__file__).resolve().parent.parent


def _spy_guard(site="synth_aisle"):
    guard, s, ss = make_guard(site)
    seen = []
    inner = guard.loader

    def spy(session):
        seen.append((guard.phase, session))
        return inner(session)
    guard.loader = spy
    return guard, s, seen


def test_guard_refuses_score_session_in_tune_phase_and_vice_versa():
    g = SessionGuard({"tune": ["a"], "score": ["b"]}, lambda s: s)
    assert g.load("a") == "a"
    with pytest.raises(LeakError):
        g.load("b")
    g.phase = "score"
    with pytest.raises(LeakError):
        g.load("a")
    with pytest.raises(LeakError):
        SessionGuard({"tune": ["a"], "score": ["a"]}, lambda s: s)


@pytest.mark.parametrize("mod", [stopwatch, staged_events])
def test_tuning_reads_only_tune_sessions(mod):
    guard, site, seen = _spy_guard()
    mod.tune(guard, site)
    assert seen and all(s in guard.tune for _, s in seen)
    assert not any(s in guard.score for _, s in seen)


@pytest.mark.parametrize("mod", [stopwatch, staged_events])
def test_lock_checked_before_any_score_session_is_read(mod, monkeypatch, tmp_path):
    monkeypatch.setenv("AISLE_LOCK_DIR", str(tmp_path))
    seen = []
    real = mod.make_guard

    def mk(site):
        g, s, ss = real(site)
        inner = g.loader
        g.loader = lambda sess: (seen.append((g.phase, sess)), inner(sess))[1]
        return g, s, ss
    monkeypatch.setattr(mod, "make_guard", mk)
    monkeypatch.setattr(mod, "write_result", lambda *a, **k: None)
    (tmp_path / ("synth_aisle_s2.lock" if mod is stopwatch else "synth_aisle_s2_events.lock")).write_text("{}")
    with pytest.raises(SystemExit) as e:
        mod.main(["--site", "synth_aisle"])
    assert "LOCKED" in str(e.value)
    assert all(p == "tune" for p, _ in seen), seen


def test_second_scored_run_needs_force_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setenv("AISLE_LOCK_DIR", str(tmp_path))
    r1 = check_and_lock("siteX", "s2", False, {})
    assert r1["n_scored_runs"] == 1 and r1["force_rescore"] is False
    with pytest.raises(SystemExit):
        check_and_lock("siteX", "s2", False, {})
    r3 = check_and_lock("siteX", "s2", True, {})
    assert r3["force_rescore"] is True and r3["n_scored_runs"] == 2


def test_merl_test_subjects_never_choose_parameters():
    from eval import merl_dwell as md
    with pytest.raises(AssertionError):
        md.choose_prior_share([{"clip": "21_1", "gt_s": 1, "dur_s": 2}, {"clip": "30_1", "gt_s": 1, "dur_s": 2}])
    d = json.loads((ROOT / "results" / "merl_dwell.json").read_text())
    assert set(d["prior_from_subjects"]) <= set(range(21, 27))
    assert set(d["scored_subjects"]) <= set(range(27, 42))
    assert not set(d["prior_from_subjects"]) & set(d["scored_subjects"])


def test_mot20_prior_is_leave_one_sequence_out():
    d = json.loads((ROOT / "results" / "mot20_dwell.json").read_text())
    meds = {s: v["prior_median_from_other_seqs"] for s, v in d["per_seq"].items()}
    assert len(meds) == 4 and len(set(meds.values())) > 1  # each scored seq gets a prior from the other three


def test_labels_never_reach_rules():
    """Rule inputs are (t, track, x, y) rows only; track IDs are opaque (no role prefix, no label person IDs)."""
    guard, site, _ = _spy_guard()
    d = guard.load("s1")
    assert all(len(r) == 4 for r in d["rows"][:1000])
    ids = {r[1].split("~")[0] for r in d["rows"]}
    persons = {v["person"] for v in d["stopwatch"]}
    assert not ids & persons
    assert all(i.startswith("t") for i in ids)
    a = common.rows_to_arrays(d["rows"])
    assert set(a) == {"t", "k", "x", "y", "names"}
