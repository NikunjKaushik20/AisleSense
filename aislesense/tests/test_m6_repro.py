"""M6 reproducibility: seeds in configs/seeds.yaml are the ones the code uses; lock files exist."""
from __future__ import annotations

import yaml

from aisle.schema import CONFIGS_DIR, ROOT, SITES_DIR


def test_seeds_match_code():
    from eval import atc_deadzone, synth_block, tips_demo
    s = yaml.safe_load((CONFIGS_DIR / "seeds.yaml").read_text(encoding="utf-8"))
    assert list(atc_deadzone.SEEDS) == s["atc_deadzone_injection"]
    assert list(synth_block.SEEDS) == s["synth_block_scenarios"]
    assert tips_demo.SEED == s["tips_demo"]
    ss = yaml.safe_load((SITES_DIR / "synth_aisle" / "sessions.yaml").read_text(encoding="utf-8"))
    assert {k: v["seed"] for k, v in ss["sources"].items()} == s["synth_aisle_sessions"]


def test_lock_files_present():
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8")
    for pkg in ("torch==", "ultralytics==", "fastapi==", "numpy==", "scipy=="):
        assert pkg in lock
    assert (ROOT / "frontend" / "package-lock.json").exists()
    assert (ROOT / "run.ps1").exists()
