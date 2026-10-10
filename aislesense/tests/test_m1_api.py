import time

import pytest
from fastapi.testclient import TestClient

from aisle.api.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(store_path=tmp_path / "t.sqlite")
    with TestClient(app) as c:
        yield c
        app.state.mgr.stop()


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_sites_and_site(client):
    ids = {s["id"] for s in client.get("/api/sites").json()}
    assert {"merl_demo", "atc", "synth_aisle"} <= ids
    s = client.get("/api/site/synth_aisle").json()
    assert s["floor"]["width"] == 20 and s["zones"]
    assert client.get("/api/site/nope").status_code == 404


def test_heatmap_returns_nx_times_ny_cells(client):
    r = client.post("/api/session", json={"site": "synth_aisle", "mode": "replay", "speed": 20})
    assert r.status_code == 200, r.text
    time.sleep(1.0)
    h = client.get("/api/heatmap").json()
    assert len(h["cells"]) == h["nx"] * h["ny"] == 200
    assert h["unit"] == "person-seconds" and h["max"] > 0
    # window query from the store
    hw = client.get("/api/heatmap", params={"session": h["session"], "t0": 0, "t1": 1e9}).json()
    assert len(hw["cells"]) == hw["nx"] * hw["ny"]


def test_later_endpoints_need_a_session(client):
    for p in ("deadzones", "flags", "journeys", "tips"):  # live since M4 (shapes in test_m4_api.py)
        r = client.get(f"/api/{p}")
        assert r.status_code == 404


def test_ws_live_sends_within_5s_on_synth(client):
    client.post("/api/session", json={"site": "synth_aisle", "mode": "live"})
    t0 = time.monotonic()
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
    assert time.monotonic() - t0 < 5.0
    assert set(msg) >= {"t", "fps", "tracks", "flags"}


def test_ws_live_tracks_have_zone_fields(client):
    client.post("/api/session", json={"site": "synth_aisle", "mode": "replay", "speed": 30})
    deadline = time.monotonic() + 5
    with client.websocket_connect("/ws/live") as ws:
        while time.monotonic() < deadline:
            msg = ws.receive_json()
            if msg["tracks"]:
                break
    assert msg["tracks"], "no live tracks within 5 s"
    assert set(msg["tracks"][0]) == {"id", "x", "y", "zone"}


def test_bad_session_requests(client):
    assert client.post("/api/session", json={"site": "nope"}).status_code == 404
    assert client.post("/api/session", json={"site": "synth_aisle", "mode": "x"}).status_code == 422
    assert client.post("/api/session", json={"site": "merl_demo", "mode": "replay"}).status_code == 400
