"""Core data types: Event row, Zone, Camera, Site, and site loading from sites/<id>/."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SITES_DIR = ROOT / "sites"
RESULTS_DIR = ROOT / "results"
VAR_DIR = ROOT / "var"
CONFIGS_DIR = ROOT / "configs"
# Git repo root (one level above aislesense/). data/ and models/ live here, outside version control.
REPO_ROOT = ROOT.parent
EVENT_FIELDS = ("session", "cam", "t", "track", "x", "y", "conf")


def data_root() -> Path:
    return Path(os.environ.get("AISLE_DATA", REPO_ROOT / "data"))


def default_model() -> str:
    """AISLE_MODEL, else <repo>/models/yolo11n.pt, else the bare name (ultralytics downloads it)."""
    env = os.environ.get("AISLE_MODEL")
    if env:
        return env
    local = REPO_ROOT / "models" / "yolo11n.pt"
    return str(local) if local.exists() else "yolo11n.pt"


@dataclass(frozen=True)
class Event:
    """One observation of one person on the floor. x, y in metres (floor frame), t in seconds."""
    session: str
    cam: str
    t: float
    track: str
    x: float
    y: float
    conf: float = 1.0

    def as_row(self) -> tuple:
        return (self.session, self.cam, float(self.t), str(self.track), float(self.x), float(self.y), float(self.conf))

    @classmethod
    def from_row(cls, row) -> "Event":
        s, c, t, k, x, y, cf = row
        return cls(str(s), str(c), float(t), str(k), float(x), float(y), float(cf))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Zone:
    name: str
    kind: str  # section | walkway | shelf_front
    polygon: list  # [[x, y], ...] in metres

    def contains(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon (no shapely dependency on the hot path)."""
        pts = self.polygon
        inside = False
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside


@dataclass
class Camera:
    id: str
    source: str
    image_points: list = field(default_factory=list)  # 4 x [u, v] px
    floor_points: list = field(default_factory=list)  # 4 x [x, y] m


@dataclass
class Site:
    id: str
    name: str
    kind: str  # video | atc | synth
    floor: dict  # {x0, y0, width, height, cell}
    cameras: list
    zones: list
    extra: dict = field(default_factory=dict)
    path: Path | None = None

    def zone_of(self, x: float, y: float, kinds=("section",)) -> str | None:
        for z in self.zones:
            if z.kind in kinds and z.contains(x, y):
                return z.name
        return None

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "kind": self.kind, "floor": self.floor,
            "cameras": [{"id": c.id, "source": c.source} for c in self.cameras],
            "zones": [{"name": z.name, "kind": z.kind, "polygon": z.polygon} for z in self.zones],
            "label": self.extra.get("label"),
        }


def load_zones(path: Path) -> list:
    if not path.exists():
        return []
    gj = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for f in gj.get("features", []):
        g = f.get("geometry") or {}
        if g.get("type") != "Polygon" or not g.get("coordinates"):
            continue
        p = f.get("properties", {})
        out.append(Zone(name=str(p.get("name", "zone")), kind=str(p.get("kind", "section")), polygon=[list(map(float, q)) for q in g["coordinates"][0]]))
    return out


def list_sites() -> list:
    return sorted(p.name for p in SITES_DIR.iterdir() if (p / "site.yaml").exists())


def load_site(site_id: str) -> Site:
    d = SITES_DIR / site_id
    cfg = yaml.safe_load((d / "site.yaml").read_text(encoding="utf-8"))
    cams = [Camera(id=str(c["id"]), source=str(c.get("source", "")), image_points=c.get("image_points", []), floor_points=c.get("floor_points", []))
            for c in cfg.get("cameras", []) or []]
    floor = dict(cfg["floor"])
    floor.setdefault("cell", 1.0)
    known = {"id", "name", "kind", "floor", "cameras"}
    return Site(id=str(cfg.get("id", site_id)), name=str(cfg.get("name", site_id)), kind=str(cfg["kind"]), floor=floor,
                cameras=cams, zones=load_zones(d / "zones.geojson"), extra={k: v for k, v in cfg.items() if k not in known}, path=d)


def resolve_data_path(p: str) -> Path:
    """Paths in site.yaml may use ${AISLE_DATA}; relative paths resolve against the data root."""
    p = p.replace("${AISLE_DATA}", str(data_root()))
    q = Path(p)
    return q if q.is_absolute() else data_root() / q
