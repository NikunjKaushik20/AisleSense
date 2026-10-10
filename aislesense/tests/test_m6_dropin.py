"""M6 drop-in proven: sites/team_template copied to a temp sites/team_ci filled from MERL clips (the documented team
path, eval/mock_team_site.py); stopwatch and staged_events run end to end on it through the app CV path, with the
score lock. Also the score-lock hardening: a re-score after a rule change is refused on a real site."""
from __future__ import annotations

import json
import shutil

import pytest

from eval import common, staged_events, stopwatch
from eval.mock_team_site import build
from aisle.schema import SITES_DIR

SITE = "team_ci"
CLIPS = ["25_3", "21_2"]  # MERL val subjects (never test subjects 27-41), shortest clips


@pytest.fixture(scope="module")
def team_ci(tmp_path_factory):
    d = build(SITE, CLIPS)
    yield d
    shutil.rmtree(d, ignore_errors=True)  # rows cache var/cache/team_ci_* is kept so reruns skip the GPU pass


@pytest.fixture
def iso(monkeypatch, tmp_path):
    monkeypatch.setenv("AISLE_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setattr(common, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


@pytest.mark.needs_data("merl")
def test_template_copy_has_every_dropin_file(team_ci):
    for f in ("site.yaml", "zones.geojson", "sessions.yaml", "labels/stopwatch.csv", "labels/events.csv"):
        assert (team_ci / f).exists(), f
    tpl = set(p.relative_to(SITES_DIR / "team_template").as_posix() for p in (SITES_DIR / "team_template").rglob("*") if p.is_file())
    got = set(p.relative_to(team_ci).as_posix() for p in team_ci.rglob("*") if p.is_file())
    assert tpl <= got


@pytest.mark.needs_data("merl")
def test_stopwatch_and_staged_end_to_end_with_lock(team_ci, iso):
    assert stopwatch.main(["--site", SITE]) == 0
    assert staged_events.main(["--site", SITE]) == 0
    sw = json.loads((iso / "results" / f"stopwatch_{SITE}.json").read_text())
    se = json.loads((iso / "results" / f"staged_{SITE}.json").read_text())
    for r in (sw, se):
        assert r["data"] == f"team footage site {SITE}"
        assert r["access_log"] == [["tune", "s1"], ["score", "s2"]]
        assert r["score_lock"]["n_scored_runs"] == 1 and r["score_lock"]["synthetic"] is False
        assert r["score_lock"]["rule_hash"]
    assert se["n"]["browse_events"] > 0
    assert (iso / "locks" / f"{SITE}_s2.lock").exists() and (iso / "locks" / f"{SITE}_s2_events.lock").exists()
    for mod in (stopwatch, staged_events):  # second scored run without the flag is refused
        with pytest.raises(SystemExit) as e:
            mod.main(["--site", SITE])
        assert "LOCKED" in str(e.value)
    assert staged_events.main(["--site", SITE, "--force-rescore"]) == 0  # same code: allowed, recorded
    se2 = json.loads((iso / "results" / f"staged_{SITE}.json").read_text())
    assert se2["score_lock"]["n_scored_runs"] == 2 and se2["score_lock"]["force_rescore"] is True
    assert se2["metric"]["arms"] == se["metric"]["arms"]  # identical code reproduces, it cannot move the number
    led = (iso / "locks" / "ledger.jsonl").read_text().strip().splitlines()
    assert len(led) == 3  # 2 first scores + 1 forced re-score


def test_rule_change_rescore_refused_on_real_site_flagged_on_synthetic(iso):
    common.check_and_lock("realsite", "s2", False, {}, rule_hash="aaa", synthetic=False)
    with pytest.raises(SystemExit) as e:
        common.check_and_lock("realsite", "s2", True, {}, rule_hash="bbb", synthetic=False)
    assert "RULE CHANGED" in str(e.value)
    common.check_and_lock("synth_aisle", "s2", False, {}, rule_hash="aaa")
    r = common.check_and_lock("synth_aisle", "s2", True, {}, rule_hash="bbb")
    assert r["rule_changed_since_first_score"] is True and r["synthetic"] is True and r["rule_hash_first"] == "aaa"
