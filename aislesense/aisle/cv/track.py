"""ByteTrack (ultralytics built-in) configured by a tracker yaml.

M1/M2 baseline buffer = 30 frames. M3 swap S1: buffer 200 (8 s at 25 fps), picked leave-one-sequence-out on MOT20
(every held-out sequence picks 200 from {30, 100, 200}; X5, eval/swaps.py). BASELINE_BUFFER stays for the eval arms."""
from __future__ import annotations

from aisle.schema import CONFIGS_DIR

BASELINE_BUFFER = 30
DEFAULT_BUFFER = 200  # S1 (M3)

BASE = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def tracker_yaml(track_buffer: int = DEFAULT_BUFFER) -> str:
    """Write (once) and return configs/bytetrack_b<N>.yaml."""
    import yaml
    p = CONFIGS_DIR / f"bytetrack_b{int(track_buffer)}.yaml"
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump({**BASE, "track_buffer": int(track_buffer)}, sort_keys=False), encoding="utf-8")
    return str(p)
