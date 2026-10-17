import numpy as np
import pytest
from fastapi.testclient import TestClient

from aisle.api.app import create_app
from aisle.cv.preview import render_jpeg


def test_preview_is_jpeg_with_heads_pixelated():
    import cv2
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    box = [200, 100, 300, 400]
    jpg = render_jpeg(frame, [box], [7], width=640, quality=95)
    assert jpg[:2] == b"\xff\xd8"
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == frame.shape
    # the head region (top 28% of the box) is blocky: far less pixel-to-pixel variation than the raw noise
    head = img[120:170, 215:285].astype(float)
    raw = frame[120:170, 215:285].astype(float)
    assert np.abs(np.diff(head, axis=1)).mean() < 0.5 * np.abs(np.diff(raw, axis=1)).mean()


def test_preview_does_not_change_the_frame():
    frame = np.zeros((120, 160, 3), np.uint8)
    render_jpeg(frame, [[10, 10, 60, 110]], [1])
    assert not frame.any()


@pytest.fixture()
def client(tmp_path):
    app = create_app(store_path=tmp_path / "t.sqlite")
    with TestClient(app) as c:
        yield c
        app.state.mgr.stop()


def test_camera_endpoint_empty_without_video(client):
    assert client.get("/api/camera").status_code == 204
    client.post("/api/session", json={"site": "synth_aisle", "mode": "replay", "speed": 30})
    r = client.get("/api/camera")
    assert r.status_code == 204 and not r.content
