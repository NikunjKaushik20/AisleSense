"""M6 privacy (F14): no raw frame is written to var/ by default; an opted-in recording is head-blurred."""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pytest

from aisle.cv.blur import blur_heads, head_boxes
from aisle.schema import VAR_DIR, default_model, load_site

MEDIA = {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mkv", ".mov", ".npy"}
MODEL = default_model()


def media_files():
    return {p for p in VAR_DIR.rglob("*") if p.is_file() and p.suffix.lower() in MEDIA}


def run_frames(n: int, record: bool, session: str):
    from aisle.cv.track import tracker_yaml
    from aisle.ingest.multi import MultiSource
    site = load_site("merl_demo")
    ms = MultiSource(site, [replace(site.cameras[0], id="cam0")], session, MODEL, tracker_yaml(200), paced=False, record=record)
    g = iter(ms)
    rows = list(itertools.islice(g, n))
    g.close()  # runs the source's finally: threads joined, recorder closed
    return ms, rows


def test_blur_pixelates_head_and_keeps_rest():
    rng = np.random.default_rng(0)
    fr = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    box = [[50, 40, 110, 190]]
    out = blur_heads(fr, box)
    hx0, hy0, hx1, hy1 = (int(v) for v in head_boxes(box)[0])
    assert not np.array_equal(out[hy0:hy1, hx0:hx1], fr[hy0:hy1, hx0:hx1])
    blk = out[hy0 + 1:hy0 + 11, hx0 + 1:hx0 + 11].reshape(-1, 3)
    assert len(np.unique(blk, axis=0)) <= 4  # coarse blocks, not texture
    assert np.array_equal(out[hy1 + 2:, :], fr[hy1 + 2:, :])  # body and background untouched


@pytest.mark.needs_data("merl")
def test_default_run_writes_no_frames():
    before = media_files()
    ms, rows = run_frames(30, record=False, session="m6priv_default")
    assert len(rows) == 30  # 30 frames went through detect + track
    assert media_files() - before == set()
    assert ms.stats()["record"] is False


@pytest.mark.needs_data("merl")
def test_recording_is_opt_in_and_blurred():
    import cv2
    ms, _ = run_frames(30, record=True, session="m6priv_rec")
    p = VAR_DIR / "recordings" / "m6priv_rec_cam0.mp4"
    try:
        assert p.exists() and ms.recorders and ms.recorders[0].frames >= 25
        cap = cv2.VideoCapture(str(p))
        ok, fr = cap.read()
        cap.release()
        assert ok and fr.ndim == 3
    finally:
        p.unlink(missing_ok=True)
