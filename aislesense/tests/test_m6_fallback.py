"""M6 live fallback (F13 [A37]): a phone-style URL stream that drops for > 3 s switches the session to a replay of
its recorded event rows with a banner; when the phone comes back the stream goes live and the banner clears.
The phone is the MJPEG stand-in (IP Webcam URL shape) in its own process; killing it = phone gone."""
from __future__ import annotations

import time

import pytest

from aisle.ingest.mjpeg import spawn
from aisle.ingest.runner import Session
from aisle.store.sqlite import Store

CLIP = "merl_x/Videos_MERL_Shopping_Dataset/27_1_crop.mp4"
PORT = 8755


def wait(pred, timeout):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if pred():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.needs_data("merl")
def test_drop_switches_to_replay_and_back(tmp_path):
    proc, url = spawn(CLIP, PORT)
    s = Session("merl_demo", mode="live", sources=[url], paced=True, store=Store(tmp_path / "f.sqlite")).start()
    try:
        st = lambda: s.source.stats()["streams"][0]["state"] if s.source.streams else "connecting"  # noqa: E731
        assert wait(lambda: st() == "live", 40), s.summary()
        time.sleep(4)  # record some rows to replay
        assert s.live()["fallback"] is None and s.summary()["fallback"] is None
        n_live = s.n_events
        assert n_live > 0
        proc.kill()
        t_kill = time.monotonic()
        assert wait(lambda: st() == "replay", 10)
        took = time.monotonic() - t_kill
        assert 2.5 <= took <= 8.0, took  # 3 s drop threshold (+ detection slack)
        msg = s.live()
        assert msg["fallback"] and "replay" in msg["fallback"].lower()
        assert msg["streams"][0]["state"] == "replay"
        assert "replay" in s.summary()["fallback"].lower()
        assert wait(lambda: s.source.streams[0].replay_batches > 5, 10)  # the map keeps moving
        proc, url = spawn(CLIP, PORT)  # phone back on the same URL
        assert wait(lambda: st() == "live", 30)
        assert s.live()["fallback"] is None
        assert s.source.stats()["streams"][0]["drops"] == 1
    finally:
        s.stop()
        proc.kill()


@pytest.mark.needs_data("merl")
def test_three_concurrent_sources_one_session(tmp_path):
    """1-3 concurrent cameras in one session: per-stream health is reported and track ids stay per camera."""
    clips = [f"merl_x/Videos_MERL_Shopping_Dataset/{c}_crop.mp4" for c in ("27_1", "28_1", "29_1")]
    from aisle.schema import load_site
    import yaml
    from aisle.schema import SITES_DIR
    site = SITES_DIR / "m6_multi_tmp"
    site.mkdir(exist_ok=True)
    try:
        cfg = yaml.safe_load((SITES_DIR / "merl_demo" / "site.yaml").read_text(encoding="utf-8"))
        cfg["id"] = "m6_multi_tmp"
        cfg["cameras"] = [{**cfg["cameras"][0], "id": f"cam{i}", "source": c} for i, c in enumerate(clips)]
        (site / "site.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        (site / "zones.geojson").write_text((SITES_DIR / "merl_demo" / "zones.geojson").read_text(encoding="utf-8"), encoding="utf-8")
        assert len(load_site("m6_multi_tmp").cameras) == 3
        s = Session("m6_multi_tmp", mode="live", paced=True, store=Store(tmp_path / "m.sqlite")).start()
        try:
            assert wait(lambda: len(s.source.streams) == 3 and all(x.state == "live" for x in s.source.streams), 40)
            time.sleep(3)
            m = s.live()
            assert [x["cam"] for x in m["streams"]] == ["cam0", "cam1", "cam2"]
            assert all(x["fps"] > 5 for x in m["streams"]), m["streams"]
            cams = {t["id"].split(":")[0] for t in m["tracks"]}
            assert cams <= {"cam0", "cam1", "cam2"}
        finally:
            s.stop()
    finally:
        import shutil
        shutil.rmtree(site, ignore_errors=True)
