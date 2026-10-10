"""M5 smoke: the app serves the built SPA, and every endpoint the dashboard calls answers with the
contract shape (or the documented 501) on synth_aisle and atc. Needs `npm --prefix frontend run build`."""
import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from aisle import schema
from aisle.api import static
from aisle.api.app import create_app

DIST = schema.ROOT / "frontend" / "dist"
NOT_IMPL = {"status": "not_implemented"}
# later endpoints: required keys once they return 200 (the M5 proposal in docs/api.md)
LATER = {
    "dwell": {"session", "unit", "prior", "sections", "long_stays"},
    "deadzones": {"session", "keep", "nx", "ny", "state", "decile", "coverage", "pooled"},
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


@pytest.mark.skipif(not (DIST / "index.html").exists(), reason="frontend not built")
def test_root_serves_spa_and_assets(client):
    r = client.get("/")
    assert r.status_code == 200 and 'id="root"' in r.text
    deep = client.get("/some/deep/link")
    assert deep.status_code == 200 and 'id="root"' in deep.text
    asset = next((DIST / "assets").glob("*.js"))
    a = client.get(f"/assets/{asset.name}")
    assert a.status_code == 200 and "javascript" in a.headers["content-type"]
    assert client.get("/api/nope").status_code == 404


@pytest.mark.parametrize("site,speed", [("synth_aisle", 30), pytest.param("atc", 600, marks=pytest.mark.needs_data("atc"))])
def test_ui_endpoints_answer_contract(client, site, speed):
    assert client.get("/api/health").json()["status"] == "ok"
    assert site in {s["id"] for s in client.get("/api/sites").json()}
    s = client.get(f"/api/site/{site}").json()
    assert {"floor", "zones", "cameras", "kind", "label"} <= set(s)
    if site == "atc":
        assert "replay" in s["label"].lower()
    r = client.post("/api/session", json={"site": site, "mode": "replay", "speed": speed})
    assert r.status_code == 200, r.text
    sid = r.json()["session"]
    deadline = time.monotonic() + 30
    h = None
    while time.monotonic() < deadline:
        h = client.get("/api/heatmap").json()
        if h.get("max", 0) > 0:
            break
        time.sleep(0.3)
    assert h and len(h["cells"]) == h["nx"] * h["ny"] and h["max"] > 0
    cur = client.get("/api/session").json()
    assert cur["session"] == sid and {"running", "fps_effective", "n_tracks", "media_seconds"} <= set(cur)
    assert any(x["id"] == sid for x in client.get("/api/sessions").json())
    for path, keys in LATER.items():
        q = f"/api/{path}?session={sid}" + ("&keep=0.01" if path == "deadzones" else "")
        resp = client.get(q)
        if resp.status_code == 501:
            assert resp.json() == NOT_IMPL
        else:
            assert resp.status_code == 200, (path, resp.status_code)
            assert keys <= set(resp.json()), (path, set(resp.json()))
    with client.websocket_connect("/ws/live") as ws:
        msg = ws.receive_json()
    assert {"session", "t", "fps", "tracks", "flags"} <= set(msg)


def test_zone_and_camera_save_round_trip(client, tmp_path, monkeypatch):
    sites = tmp_path / "sites"
    shutil.copytree(schema.SITES_DIR / "merl_demo", sites / "edit_me")
    monkeypatch.setattr(schema, "SITES_DIR", sites)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "Aisle 1", "kind": "walkway"},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 1]]]}}]}
    r = client.put("/api/site/edit_me/zones", json=fc)
    assert r.status_code == 200, r.text
    saved = json.loads((sites / "edit_me" / "zones.geojson").read_text())
    ring = saved["features"][0]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and saved["features"][0]["properties"]["kind"] == "walkway"
    assert schema.load_site("edit_me").zones[0].name == "Aisle 1"
    bad = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "x", "kind": "nope"},
                                                       "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]}}]}
    assert client.put("/api/site/edit_me/zones", json=bad).status_code == 422
    cam = {"image_points": [[0, 0], [10, 0], [10, 10], [0, 10]], "floor_points": [[0, 0], [1, 0], [1, 1], [0, 1]]}
    assert client.put("/api/site/edit_me/camera/cam0", json=cam).status_code == 200
    assert schema.load_site("edit_me").cameras[0].image_points == cam["image_points"]
    assert client.put("/api/site/edit_me/camera/cam0", json={**cam, "image_points": cam["image_points"][:3]}).status_code == 422
    assert client.put("/api/site/../zones", json=fc).status_code in (404, 405)
    assert client.put("/api/site/nope/zones", json=fc).status_code == 404


def test_validate_zones_closes_rings():
    feats = static.validate_zones({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "A", "kind": "section"},
         "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1]]]}}]})
    assert feats[0]["geometry"]["coordinates"][0][-1] == [0.0, 0.0]
