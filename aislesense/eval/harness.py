"""Team-footage harness plumbing shared by eval/stopwatch.py and eval/staged_events.py.

A site with sessions.yaml (tune / score / sources) and labels/{stopwatch,events}.csv. Session sources:
  {synth_scenario: <name>, seed: <int>}   synthetic generator rows (sites/synth_aisle now)
  {video: <path or url>} or a plain string  team footage through the app CV path (aisle VideoSource), cached as
                                            var/cache/<site>_<session>_rows.csv
All session data (rows AND labels) goes through eval.common.SessionGuard, so tuning code cannot read a score session.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import SessionGuard, fingerprint  # noqa: E402

import yaml  # noqa: E402

from aisle.schema import SITES_DIR, VAR_DIR, default_model, load_site  # noqa: E402


def result_tag(site_id: str) -> str:
    return "synth" if site_id == "synth_aisle" else site_id


def load_sessions(site_id: str) -> dict:
    p = SITES_DIR / site_id / "sessions.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def read_labels(site_id: str, kind: str) -> list:
    p = SITES_DIR / site_id / "labels" / f"{kind}.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _video_rows(site, session: str, src) -> tuple[list, float]:
    from dataclasses import replace
    from aisle.cv.track import tracker_yaml
    from aisle.ingest.sources import VideoSource
    cache = VAR_DIR / "cache" / f"{site.id}_{session}_rows.csv"
    stride = int(site.extra.get("stride", 2))
    hz = float(site.extra.get("fps", 30)) / stride
    if cache.exists():
        with open(cache, newline="") as f:
            return [(float(r[0]), r[1], float(r[2]), float(r[3])) for r in csv.reader(f)], hz
    path = src["video"] if isinstance(src, dict) else str(src)
    cam = replace(site.cameras[0], source=path)
    vs = VideoSource(site, cam, session, site.extra.get("model", default_model()),
                     tracker_yaml(int(site.extra.get("track_buffer", 30))), stride=stride, loop=False, paced=False)
    rows = [(e.t, e.track, e.x, e.y) for _, ev in vs for e in ev]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return rows, hz


def make_guard(site_id: str) -> tuple[SessionGuard, object, dict]:
    """Returns (guard, site, sessions). guard.load(s) -> {rows, hz, stopwatch, events} for session s."""
    site = load_site(site_id)
    ss = load_sessions(site_id)
    sw_all = read_labels(site_id, "stopwatch")
    ev_all = read_labels(site_id, "events")

    def loader(session):
        src = (ss.get("sources") or {}).get(session)
        if src is None:
            raise KeyError(f"no source for session {session} in sites/{site_id}/sessions.yaml")
        if isinstance(src, dict) and "synth_scenario" in src:
            from synth.labelled import generate_labelled, load
            sc = load(src["synth_scenario"])
            rows, _, _ = generate_labelled(sc, site, int(src["seed"]))
            hz = float(sc.get("hz", 5))
        else:
            rows, hz = _video_rows(site, session, src)
        return {"rows": rows, "hz": hz,
                "stopwatch": [{"section": r["section"], "person": r["person"], "t_in": float(r["t_in"]), "t_out": float(r["t_out"])}
                              for r in sw_all if r["session"] == session],
                "events": [{"type": r["type"], "t0": float(r["t0"]), "t1": float(r["t1"]), "zone": r["zone"]}
                           for r in ev_all if r["session"] == session]}
    return SessionGuard(ss, loader), site, ss


def site_hash(site_id: str) -> str:
    d = SITES_DIR / site_id
    files = [p for p in [d / "sessions.yaml", d / "zones.geojson", d / "site.yaml", d / "labels" / "stopwatch.csv", d / "labels" / "events.csv"] if p.exists()]
    return fingerprint(files)
