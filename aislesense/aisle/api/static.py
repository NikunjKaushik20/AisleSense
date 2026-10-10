"""M5: serve the built dashboard (frontend/dist) as an SPA, plus the small setup endpoints the
zone/homography editor and the before/after picker need.

Installed by app.create_app via `install(app, store)`. Mount order matters: `install` must run
after every /api route is registered, because the SPA catch-all answers everything else.

Endpoints added here (documented in docs/api.md, "M5 additions"):
  GET  /api/sessions                          stored sessions, newest last
  GET  /api/site/{id}/camera/{cam}/frame      one JPEG still from the camera source (setup only)
  PUT  /api/site/{id}/zones                   GeoJSON FeatureCollection -> sites/<id>/zones.geojson
  PUT  /api/site/{id}/camera/{cam}            {image_points, floor_points} -> sites/<id>/site.yaml
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from aisle import schema

DIST = schema.ROOT / "frontend" / "dist"
ZONE_KINDS = {"section", "walkway", "shelf_front"}


def _sites_dir() -> Path:
    return schema.SITES_DIR


def _site_dir(site_id: str) -> Path:
    if not site_id.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(404, f"unknown site {site_id}")
    d = _sites_dir() / site_id
    if not (d / "site.yaml").exists():
        raise HTTPException(404, f"unknown site {site_id}")
    return d


class CameraCalib(BaseModel):
    image_points: list[list[float]]
    floor_points: list[list[float]]


def validate_zones(gj: dict) -> list[dict]:
    """Return the cleaned feature list or raise 422. Polygons need >= 3 distinct points."""
    if not isinstance(gj, dict) or gj.get("type") != "FeatureCollection" or not isinstance(gj.get("features"), list):
        raise HTTPException(422, "body must be a GeoJSON FeatureCollection")
    out = []
    for i, f in enumerate(gj["features"]):
        p = (f or {}).get("properties") or {}
        g = (f or {}).get("geometry") or {}
        name, kind = str(p.get("name", "")).strip(), p.get("kind")
        if not name or kind not in ZONE_KINDS:
            raise HTTPException(422, f"feature {i}: needs a name and kind in {sorted(ZONE_KINDS)}")
        if g.get("type") != "Polygon" or not g.get("coordinates"):
            raise HTTPException(422, f"feature {i}: geometry must be a Polygon")
        ring = [[float(q[0]), float(q[1])] for q in g["coordinates"][0]]
        if len({tuple(q) for q in ring}) < 3:
            raise HTTPException(422, f"feature {i}: polygon needs at least 3 distinct points")
        if ring[0] != ring[-1]:
            ring.append(list(ring[0]))
        out.append({"type": "Feature", "properties": {"name": name, "kind": kind},
                    "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return out


def router(store=None) -> APIRouter:
    r = APIRouter()

    @r.get("/api/sessions")
    def sessions():
        if store is None:
            return []
        return [{"id": s["id"], "site": s["site"], "mode": s["mode"], "started": s["started"]} for s in store.sessions()]

    @r.put("/api/site/{site_id}/zones")
    async def save_zones(site_id: str, request: Request):
        d = _site_dir(site_id)
        try:
            gj = await request.json()
        except Exception:
            raise HTTPException(422, "body is not JSON")
        feats = validate_zones(gj)
        (d / "zones.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1), encoding="utf-8")
        return {"saved": True, "site": site_id, "n_zones": len(feats), "path": f"sites/{site_id}/zones.geojson"}

    @r.put("/api/site/{site_id}/camera/{cam_id}")
    def save_camera(site_id: str, cam_id: str, body: CameraCalib):
        d = _site_dir(site_id)
        if len(body.image_points) != 4 or len(body.floor_points) != 4 or any(len(q) != 2 for q in body.image_points + body.floor_points):
            raise HTTPException(422, "need exactly 4 image points and 4 floor points, each [a, b]")
        cfg = yaml.safe_load((d / "site.yaml").read_text(encoding="utf-8"))
        cams = cfg.get("cameras") or []
        cam = next((c for c in cams if str(c.get("id")) == cam_id), None)
        if cam is None:
            raise HTTPException(404, f"unknown camera {cam_id}")
        cam["image_points"] = [[round(a, 2), round(b, 2)] for a, b in body.image_points]
        cam["floor_points"] = [[round(a, 3), round(b, 3)] for a, b in body.floor_points]
        (d / "site.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        return {"saved": True, "site": site_id, "camera": cam_id, "path": f"sites/{site_id}/site.yaml"}

    @r.get("/api/site/{site_id}/camera/{cam_id}/frame")
    def camera_frame(site_id: str, cam_id: str):
        d = _site_dir(site_id)
        cfg = yaml.safe_load((d / "site.yaml").read_text(encoding="utf-8"))
        cam = next((c for c in cfg.get("cameras") or [] if str(c.get("id")) == cam_id), None)
        if cam is None:
            raise HTTPException(404, f"unknown camera {cam_id}")
        import cv2  # local import: only the editor needs it
        src = str(cam.get("source", ""))
        path = src if "://" in src else str(schema.resolve_data_path(src))
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404, f"no frame from camera {cam_id}")
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Frame-Size": f"{frame.shape[1]}x{frame.shape[0]}"})

    return r


def install(app: FastAPI, store=None, dist: Path | None = None) -> None:
    """Add the M5 routes, then the SPA: real files from dist, index.html for any other non-API path."""
    app.include_router(router(store))
    dist = dist or DIST

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith(("api/", "ws/")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index = dist / "index.html"
        if not index.exists():
            return JSONResponse({"detail": "frontend not built: npm --prefix frontend ci && npm --prefix frontend run build"}, status_code=404)
        f = (dist / path).resolve()
        if path and f.is_file() and dist.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(index, media_type="text/html")
