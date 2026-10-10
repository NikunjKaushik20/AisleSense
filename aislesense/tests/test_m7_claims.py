"""M7: every doc number is rebuilt from results/*.json; banned claims, free numbers and missing priors fail."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_claims as cc  # noqa: E402

pytestmark = pytest.mark.skipif(not cc.SRC.exists(), reason="docs/ is not in git; claims check runs only where docs/src exists")


def test_docs_match_results():
    assert cc.check() == []


def test_every_key_resolves():
    for k in cc.K:
        assert cc.value(k) not in ("", None), k


def test_first_slide_is_0_of_656_from_atc_deadzone():
    key = cc.FIRST_SLIDE[1]
    assert cc.K[key][0] == "atc_deadzone.json"
    assert cc.value(key) == "0 of 656"
    demo = (ROOT / "docs" / "demo_script.md").read_text(encoding="utf-8")
    assert "First-slide number: 0 of 656" in demo


def _copy_outputs(tmp_path):
    outs = {}
    for name, out in cc.OUTPUTS.items():
        dst = tmp_path / "out" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out, dst)
        outs[name] = dst
    claims = tmp_path / "out" / "claims.md"
    shutil.copy(cc.CLAIMS_MD, claims)
    return outs, claims


def test_hand_edited_number_fails(tmp_path):
    outs, claims = _copy_outputs(tmp_path)
    readme = outs["README.md"]
    text = readme.read_text(encoding="utf-8")
    assert "63.4%" in text
    readme.write_text(text.replace("63.4%", "73.4%", 1), encoding="utf-8")
    errs = cc.check(outputs=outs, claims_md=claims)
    assert any("README.md" in e and "differs" in e for e in errs)


def test_changed_result_fails(tmp_path):
    res = tmp_path / "results"
    shutil.copytree(cc.RESULTS, res)
    fp = res / "atc_deadzone.json"
    d = json.loads(fp.read_text(encoding="utf-8"))
    d["metric"]["blind"]["aware"]["dead_pooled"] = 5
    fp.write_text(json.dumps(d), encoding="utf-8")
    outs, claims = _copy_outputs(tmp_path)
    errs = cc.check(outputs=outs, claims_md=claims, results=res)
    assert any("first-slide" in e for e in errs)
    assert any("differs" in e for e in errs)


def test_free_number_detected():
    assert cc.free_numbers("Our tracker is 42% better.")
    assert cc.free_numbers("It reaches 0.91 F1.")
    assert not cc.free_numbers("Stand still for 30 s in the walkway, 20+ events, `python run.py --port 8000`.")
    assert not cc.free_numbers("Ours is {{mot_best}} against {{mot_prior}} [X5].")


def test_headline_needs_prior():
    assert cc.prior_adjacency("Dwell error is {{mot_best}}.")
    assert not cc.prior_adjacency("Dwell error is {{mot_best}} against {{mot_prior}}.")
    assert cc.prior_adjacency("| row | {{atc_aware_blind_dead}} |\n| row | {{atc_decile_blind_pct}} |")


@pytest.mark.parametrize("line", [
    "AisleSense is the first system to map dwell.",
    "It works in dense crowds.",
    "Here is our detector on ATC.",
    "ATC proves our detector is right.",
    "A state-of-the-art tracker.",
])
def test_banned_phrases(line):
    assert cc.banned(line)


def test_detects_blocks_banned_until_team_scored():
    assert cc.banned("AisleSense detects blocks.", team_scored=False)
    assert not cc.banned("AisleSense detects blocks.", team_scored=True)


def test_quote_region_exempt():
    txt = "<!-- claims:quote -->\n- \"Works in dense crowds.\"\n<!-- /claims:quote -->\nfine"
    assert not cc.banned(txt)


def test_unknown_key_fails():
    errs = []
    cc.render("{{no_such_key}}", errors=errs)
    assert errs
