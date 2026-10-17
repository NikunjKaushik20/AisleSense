"""FastAPI app: REST + /ws/live, serves frontend/dist as static. Contract in docs/api.md."""
from __future__ import annotations

import asyncio
import threading

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from aisle.analytics import heatmap
from aisle.ingest.runner import Session
from aisle.schema import VAR_DIR, list_sites, load_site
from aisle.store.sqlite import Store

VERSION = "1.0.0"
NOT_IMPL = {"status": "not_implemented"}


class SessionReq(BaseModel):
    site: str
    mode: str = "live"
    sources: list[str] | None = None
    speed: float | None = None


class Manager:
    def __init__(self, store: Store, cfg: dict | None = None):
        self.store, self.cfg = store, cfg or {}
        self.current: Session | None = None
        self.sessions: dict[str, Session] = {}
        self.lock = threading.Lock()

    def start(self, site: str, mode: str = "live", sources=None, speed: float | None = None) -> Session:
        with self.lock:
            if self.current is not None:
                self.current.stop()
            s = Session(site, mode=mode, speed=speed or 1.0, sources=sources, paced=True, store=self.store, cfg=self.cfg).start()
            self.current = self.sessions[s.id] = s
            return s

    def stop(self) -> None:
        with self.lock:
            if self.current is not None:
                self.current.stop()


def create_app(store_path=None, autostart: dict | None = None, cfg: dict | None = None,
               prefetch: bool | None = None) -> FastAPI:
    store = Store(store_path)
    mgr = Manager(store, cfg)
    app = FastAPI(title="AisleSense", version=VERSION)
    app.state.mgr = mgr

    @app.on_event("startup")
    def _startup():
        if autostart:
            mgr.start(**autostart)

    @app.on_event("shutdown")
    def _shutdown():
        mgr.stop()
        store.flush()

    @app.get("/api/health")
    def health():
        cur = mgr.current
        return {"status": "ok", "version": VERSION, "session": cur.id if cur else None,
                "running": bool(cur and cur.running)}

    @app.get("/api/sites")
    def sites():
        out = []
        for sid in list_sites():
            s = load_site(sid)
            out.append({"id": s.id, "name": s.name, "kind": s.kind, "label": s.extra.get("label")})
        return out

    @app.get("/api/site/{site_id}")
    def site(site_id: str):
        if site_id not in list_sites():
            raise HTTPException(404, f"unknown site {site_id}")
        return load_site(site_id).to_dict()

    @app.post("/api/session")
    def new_session(req: SessionReq):
        if req.site not in list_sites():
            raise HTTPException(404, f"unknown site {req.site}")
        if req.mode not in ("live", "replay"):
            raise HTTPException(422, "mode must be live or replay")
        try:
            s = mgr.start(req.site, req.mode, req.sources, req.speed)
        except (ValueError, FileNotFoundError, NotImplementedError) as ex:
            raise HTTPException(400, str(ex))
        return {"session": s.id, "site": s.site.id, "mode": s.mode, "speed": s.speed, "label": s.source.label, "started": True}

    @app.get("/api/session")
    def cur_session():
        cur = mgr.current
        return cur.summary() if cur else {"session": None, "running": False}

    @app.get("/api/camera")
    def camera(cam: str | None = None):
        """Live preview JPEG of a video session's camera (heads pixelated). Built in memory per request, never saved.
        204 when there is no live frame to show."""
        cur = mgr.current
        src = cur.source if cur else None
        jpg = src.preview_jpeg(cam) if src is not None and hasattr(src, "preview_jpeg") else None
        if jpg is None:  # not a video session, or no live frame yet: a normal state, so no error status
            return Response(status_code=204, headers={"Cache-Control": "no-store"})
        return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/heatmap")
    def get_heatmap(session: str | None = None, t0: float | None = None, t1: float | None = None):
        s = mgr.sessions.get(session) if session else mgr.current
        if session and s is None:
            meta = next((m for m in store.sessions() if m["id"] == session), None)
            if meta is None:
                raise HTTPException(404, f"unknown session {session}")
            site_ = load_site(meta["site"])
            g = heatmap.grid_from_events(site_, store.events(session, t0, t1))
            return heatmap.payload(g, session, site_.id, t0, t1)
        if s is None:
            raise HTTPException(404, "no session; POST /api/session first")
        if t0 is None and t1 is None:
            with s.lock:
                return heatmap.payload(s.grid, s.id, s.site.id)
        store.flush()
        src = s.source
        g = heatmap.grid_from_events(s.site, store.events(s.id, t0, t1),
                                     dict(nominal_dt=src.nominal_dt, max_gap=src.max_gap, death_gap=src.death_gap))
        return heatmap.payload(g, s.id, s.site.id, t0, t1)

    # ---- M3 dwell + M4 dead zones (D-1), walkway/shelf flags (D-2), journeys, tips, before/after diff
    # Results are cached per session version: a stored session never changes; a running one gets a new version every
    # REFRESH_S seconds (a full ATC day is ~1M rows, so every poll cannot reload it).
    # M6: panels are served stale-while-revalidate, and a prefetch thread rebuilds every panel of the current session
    # in the background whenever its version changes, so a dashboard tab switch does not wait on a 3-10 s rebuild
    # (eval/api_latency.py -> results/api_latency.json). AISLE_PREFETCH=0 (or prefetch=False) gives the M4 behaviour.
    import os as _os
    import time as _time
    cache: dict = {}
    latest: dict = {}  # (kind, sid) -> (ver, value): last good panel per session, served while a newer one builds
    building: set = set()
    clock = threading.Lock()
    REFRESH_S = 10.0
    swr_on = prefetch if prefetch is not None else _os.environ.get("AISLE_PREFETCH", "1") != "0"
    app.state.panel_stats = {"prefetch": swr_on, "builds": 0, "stale_served": 0, "build_s": 0.0}

    def _cached(key, fn):
        with clock:
            if key in cache:
                return cache[key]
        v = fn()
        with clock:
            if len(cache) > 96:
                cache.clear()
            cache[key] = v
        return v

    def _version(s):
        return ("t", int(_time.monotonic() // REFRESH_S)) if s.running else ("done", s.n_events)

    def _resolve(session: str | None):
        """-> (site, session id, events() loader, hz, version). Current session when `session` is None."""
        s = mgr.sessions.get(session) if session else mgr.current
        if s is not None:
            ver = _version(s)
            sid = s.id

            def load():
                store.flush()
                return store.events(sid)
            return s.site, sid, (lambda: _cached(("ev", sid, ver), load)), 1.0 / s.source.nominal_dt, ver
        if session is None:
            raise HTTPException(404, "no session; POST /api/session first")
        meta = next((m for m in store.sessions() if m["id"] == session), None)
        if meta is None:
            raise HTTPException(404, f"unknown session {session}")
        return load_site(meta["site"]), session, (lambda: _cached(("ev", session, "stored"), lambda: store.events(session))), None, "stored"

    dead_masks: dict = {}  # finished session id -> its DEAD mask (None = no rows). Finished sessions never change.

    def _dead_mask(site, sid: str):
        """A finished session's DEAD mask, from memory, then var/cache, then (once) its stored rows. Loading a full
        ATC day from SQLite takes ~15 s, so without this the first panel after a site switch waits that long."""
        import numpy as np
        from aisle.analytics import deadzone as dz
        if sid in dead_masks:
            return dead_masks[sid]
        f = VAR_DIR / "cache" / f"dead_{sid}.npy"
        m = None
        try:
            a = np.load(f)
            m = a.astype(bool) if a.size else None
        except (OSError, ValueError):
            store.flush()
            m = dz.dead_mask(site, store.events(sid))
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, m if m is not None else np.zeros(0, bool))
            except OSError:
                pass
        dead_masks[sid] = m
        return m

    def _previous(site, sid: str):
        """Most recent earlier stored session of the same site with rows: (id, DEAD mask) or None."""
        ss = store.sessions()
        idx = next((i for i, m in enumerate(ss) if m["id"] == sid), None)
        if idx is None:
            return None
        for m in reversed(ss[:idx]):
            if m["site"] == site.id:
                mask = _dead_mask(site, m["id"])
                if mask is not None:
                    return m["id"], mask
        return None

    def _warm_previous():
        """Startup: build the mask of each site's latest finished session, the one the next session compares with."""
        seen = set()
        for m in reversed(store.sessions()):
            cur = mgr.current
            if m["site"] in seen or (cur is not None and m["id"] == cur.id):
                continue
            try:
                site_ = load_site(m["site"])
            except Exception:
                continue
            if _dead_mask(site_, m["id"]) is not None:
                seen.add(m["site"])

    def _build(kind, site, sid, events, hz, ver):
        """One panel payload. kind: flags | journeys | tips | dwell | dz:<keep>."""
        import json
        from aisle.analytics import block, deadzone as dz, dwell, journeys, tips
        from aisle.schema import RESULTS_DIR
        if kind == "flags":
            return block.flags(events(), site, hz)
        if kind == "journeys":
            return journeys.payload(site, events(), sid)
        if kind == "dwell":
            return dwell.payload(site, events(), sid, hz=hz)
        if kind == "tips":
            ev = events()
            prev = _previous(site, sid)
            st = dz.analyse(site, ev, 1.0, prev_dead=prev[1] if prev else None)["state"]
            dw = _panel("dwell", site, sid, events, hz, ver, background=True) if ev else None
            return tips.payload(site, sid, st, _panel("flags", site, sid, events, hz, ver, background=True), dw)
        if kind.startswith("dz:"):
            prev = _previous(site, sid)
            p = RESULTS_DIR / "atc_deadzone.json"
            res = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
            return dz.payload(site, events(), sid, float(kind[3:]), None, res, prev_dead=prev[1] if prev else None)
        raise KeyError(kind)

    def _timed_build(kind, site, sid, events, hz, ver):
        t0 = _time.perf_counter()
        v = _build(kind, site, sid, events, hz, ver)
        with clock:
            st = app.state.panel_stats
            st["builds"] += 1
            st["build_s"] = round(st["build_s"] + _time.perf_counter() - t0, 3)
        return v

    def _store(key, ver, v):
        with clock:
            if len(latest) > 256:
                latest.clear()
            latest[key] = (ver, v)

    def _panel(kind, site, sid, events, hz, ver, background: bool = False):
        """Exact version cached -> return it. Older version cached (and SWR on) -> return it now and rebuild in a
        background thread. Nothing cached -> build now. `background` = caller is already a builder: build inline."""
        key = (kind, sid)
        with clock:
            have = latest.get(key)
        if have is not None and have[0] == ver:
            return have[1]
        if have is not None and swr_on and not background:
            with clock:
                start = key not in building
                building.add(key)
                app.state.panel_stats["stale_served"] += 1
            if start:
                def job():
                    try:
                        _store(key, ver, _timed_build(kind, site, sid, events, hz, ver))
                    finally:
                        with clock:
                            building.discard(key)
                threading.Thread(target=job, daemon=True, name=f"panel-{kind}").start()
            return have[1]
        v = _timed_build(kind, site, sid, events, hz, ver)
        _store(key, ver, v)
        return v

    def _p(kind, session):
        site, sid, events, hz, ver = _resolve(session)
        return _panel(kind, site, sid, events, hz, ver), sid

    def _tips(session: str | None):
        """Tips panel, with the BLOCK / BROWSE tips recounted from the same flags panel the map shows. The full tips
        build (dead zones + dwell) can trail the flags by a refresh, and a tip saying "blocked 2 times" next to a
        BLOCK x5 badge reads as a bug."""
        from aisle.analytics import tips as tp
        site, sid, events, hz, ver = _resolve(session)
        t = _panel("tips", site, sid, events, hz, ver)
        fl = _panel("flags", site, sid, events, hz, ver)
        fresh = tp.from_parts(site, None, fl, None)  # no dead-zone state or dwell: only BLOCK / BROWSE tips
        keep = [x for x in t["tips"] if x["kind"] not in ("BLOCK", "BROWSE")]
        return {**t, "tips": sorted(keep + fresh, key=lambda x: (-x["severity"], x["kind"], str(x["zone"])))}

    PANELS = ("dwell", "flags", "tips", "journeys", "dz:1.0", "dz:0.15", "dz:0.05", "dz:0.01")
    stop_prefetch = threading.Event()

    def _prefetch_loop():
        while not stop_prefetch.is_set():
            cur = mgr.current
            try:
                if cur is not None and cur.n_events:
                    site, sid, events, hz, ver = _resolve(cur.id)
                    for kind in PANELS:
                        if stop_prefetch.is_set() or mgr.current is not cur:
                            break
                        _panel(kind, site, sid, events, hz, ver, background=True)
            except Exception:  # a failed prefetch only costs latency; the endpoint surfaces the error itself
                pass
            stop_prefetch.wait(1.0)

    if swr_on:
        @app.on_event("startup")
        def _start_prefetch():
            threading.Thread(target=_prefetch_loop, daemon=True, name="panel-prefetch").start()
            threading.Thread(target=_warm_previous, daemon=True, name="warm-previous").start()

        @app.on_event("shutdown")
        def _stop_prefetch():
            stop_prefetch.set()

    @app.get("/api/panel_stats")
    def panel_stats():
        with clock:
            return {**app.state.panel_stats, "cached_panels": len(latest)}

    @app.get("/api/dwell")
    def get_dwell(session: str | None = None):
        """M3 S3: section dwell = person-seconds / merged entries, prior column from tune sessions (docs/api.md)."""
        return _p("dwell", session)[0]

    @app.get("/api/deadzones")
    def get_deadzones(session: str | None = None, keep: float = 1.0):
        from aisle.analytics import deadzone as dz
        k = next((k for k in dz.KEEPS if abs(keep - k) < 1e-9), None)
        if k is None:
            raise HTTPException(422, f"keep must be one of {list(dz.KEEPS)}")
        return _p(f"dz:{float(k)}", session)[0]

    @app.get("/api/flags")
    def get_flags(session: str | None = None):
        from aisle.analytics import block
        fl, sid = _p("flags", session)
        return {"session": sid, "rule": block.RULE["text"], "flags": fl}

    @app.get("/api/journeys")
    def get_journeys(session: str | None = None):
        return _p("journeys", session)[0]

    @app.get("/api/tips")
    def get_tips(session: str | None = None):
        return _tips(session)

    @app.get("/api/diff")
    def get_diff(a: str, b: str):
        from aisle.analytics import diff
        ta, tb = _tips(a), _tips(b)
        return diff.payload(a, b, ta["tips"], tb["tips"])

    @app.websocket("/ws/live")
    async def ws_live(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                cur = mgr.current
                msg = cur.live() if cur else {"session": None, "t": None, "fps": 0.0, "tracks": [], "flags": []}
                await ws.send_json(msg)
                await asyncio.sleep(0.5)
        except (WebSocketDisconnect, RuntimeError):
            return

    # M5: setup endpoints + SPA catch-all (aisle/api/static.py). Keep this LAST: add new /api routes above it.
    from aisle.api import static
    static.install(app, store)
    return app
