"""M4 API: /api/deadzones, /api/flags, /api/journeys, /api/tips, /api/diff return 200 with the docs/api.md shapes."""
import time

import pytest
from fastapi.testclient import TestClient

from aisle.api.app import create_app

SHAPES = {
    "deadzones": {"session", "keep", "x0", "y0", "cell", "nx", "ny", "state", "decile", "coverage", "pooled", "source"},
    "flags": {"session", "flags"},
    "journeys": {"session", "nodes", "links"},
    "tips": {"session", "tips"},
}


@pytest.fixture()
def client(tmp_path):
    app = create_app(store_path=tmp_path / "t.sqlite")
    with TestClient(app) as c:
        yield c
        app.state.mgr.stop()


def _run(client, site, speed, until_events=2000, timeout=40):
    r = client.post("/api/session", json={"site": site, "mode": "replay", "speed": speed})
    assert r.status_code == 200, r.text
    sid = r.json()["session"]
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if client.get("/api/session").json().get("n_events", 0) >= until_events:
            break
        time.sleep(0.3)
    return sid


def test_m4_endpoints_shapes_on_synth(client):
    sid = _run(client, "synth_aisle", 60, until_events=20000)
    for path, keys in SHAPES.items():
        r = client.get(f"/api/{path}", params={"session": sid})
        assert r.status_code == 200, (path, r.text)
        assert keys <= set(r.json()), (path, set(r.json()))
    dz = client.get("/api/deadzones", params={"session": sid, "keep": 0.01}).json()
    assert len(dz["state"]) == len(dz["decile"]) == len(dz["coverage"]) == dz["nx"] * dz["ny"]
    assert set(dz["state"]) <= {"ok", "dead", "watch", "unseen", "unknown"}
    assert client.get("/api/deadzones", params={"session": sid, "keep": 0.5}).status_code == 422
    fl = client.get("/api/flags", params={"session": sid}).json()["flags"]
    for f in fl:
        assert {"id", "type", "zone", "x", "y", "t0", "t1", "evidence"} <= set(f) and f["type"] in ("BLOCK", "BROWSE")
    j = client.get("/api/journeys", params={"session": sid}).json()
    assert all(0 <= l["source"] < len(j["nodes"]) and 0 <= l["target"] < len(j["nodes"]) and l["value"] > 0 for l in j["links"])
    for t in client.get("/api/tips", params={"session": sid}).json()["tips"]:
        assert {"id", "kind", "text", "severity"} <= set(t)
        assert t["kind"] in ("DEAD", "UNSEEN", "WATCH", "BLOCK", "BROWSE", "DWELL")
    # current session works without ?session
    assert client.get("/api/flags").status_code == 200


def test_diff_between_two_sessions(client):
    a = _run(client, "synth_aisle", 60, until_events=5000)
    b = _run(client, "synth_aisle", 60, until_events=5000)
    client.post("/api/session", json={"site": "synth_aisle", "mode": "replay", "speed": 1})  # stop b
    r = client.get("/api/diff", params={"a": a, "b": b})
    assert r.status_code == 200, r.text
    assert {"a", "b", "cleared", "added", "kept"} <= set(r.json())
    assert client.get("/api/diff", params={"a": a, "b": "nope"}).status_code == 404


@pytest.mark.needs_data("atc")
def test_deadzones_atc_pooled_from_results(client):
    sid = _run(client, "atc", 600, until_events=20000)
    d = client.get("/api/deadzones", params={"session": sid, "keep": 0.01}).json()
    assert d["source"] == "results/atc_deadzone.json"
    assert d["pooled"]["blind"]["n"] == 656 and d["pooled"]["blind"]["aware_dead"] == 0
    assert d["pooled"]["blind"]["decile_dead"] > 0
