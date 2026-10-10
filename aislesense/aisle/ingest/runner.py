"""Session runner: pulls a Source, stores events, updates the live grid and live track state."""
from __future__ import annotations

import threading
import time

from aisle.analytics.heatmap import grid_for
from aisle.ingest.sources import make_source
from aisle.schema import load_site
from aisle.store.sqlite import Store


class Session:
    def __init__(self, site_id: str, mode: str = "live", speed: float = 1.0, sources=None, paced: bool = True,
                 store: Store | None = None, cfg: dict | None = None):
        self.site = load_site(site_id)
        self.mode, self.speed, self.paced = mode, float(speed), paced
        self.store = store or Store()
        self.id = self.store.new_session(site_id, mode, {"speed": speed, "sources": sources, "cfg": cfg or {}})
        self.source = make_source(self.site, self.id, mode, sources=sources, paced=paced, cfg=cfg)
        self.grid = grid_for(self.site, self.source)
        self.live_window = max(2.0 * self.source.nominal_dt, 1.0)
        self.lock = threading.Lock()
        self.tracks = {}  # track -> (t, x, y)
        self.n_events = 0
        self.all_tracks = set()
        self.batches = 0
        self.t_media0 = self.t_media = None
        self.wall_first = self.wall_last = None
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        self._fps_hist = []

    # ---- control
    def start(self) -> "Session":
        self._thread = threading.Thread(target=self._run, name=f"session-{self.id}", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        try:
            self.source.close()  # unblocks multi-stream sources that are waiting for frames
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ---- loop
    def _run(self) -> None:
        try:
            pace = self.paced and self.source.paced
            for t, evs in self.source:
                if self._stop.is_set():
                    break
                now = time.monotonic()
                if self.wall_first is None:
                    self.wall_first, self.t_media0 = now, t
                if pace:
                    ahead = (t - self.t_media0) / self.speed - (now - self.wall_first)
                    if ahead > 0.02:
                        if self._stop.wait(ahead):
                            break
                self.store.insert(evs)
                with self.lock:
                    self.grid.add_many(evs)
                    for e in evs:
                        self.tracks[e.track] = (e.t, e.x, e.y)
                        self.all_tracks.add(e.track)
                    self.n_events += len(evs)
                    self.batches += 1
                    self.t_media = t
                    self.wall_last = time.monotonic()
                    self._fps_hist.append(self.wall_last)
                    if len(self._fps_hist) > 60:
                        self._fps_hist = self._fps_hist[-60:]
                    if self.batches % 20 == 0:
                        self.grid.expire(now=t)
                        cut = t - self.live_window
                        self.tracks = {k: v for k, v in self.tracks.items() if v[0] >= cut}
        except Exception as ex:  # surfaced through /api/session and the smoke JSON
            self.error = f"{type(ex).__name__}: {ex}"
        finally:
            self.store.flush()
            self.source.close()

    # ---- views
    def live(self) -> dict:
        with self.lock:
            t = self.t_media
            cut = (t or 0) - self.live_window
            tr = [{"id": k, "x": round(v[1], 3), "y": round(v[2], 3), "zone": self.site.zone_of(v[1], v[2])}
                  for k, v in self.tracks.items() if t is not None and v[0] >= cut]
            h = self._fps_hist
            fps = (len(h) - 1) / (h[-1] - h[0]) if len(h) > 1 and h[-1] > h[0] else 0.0
        st = self.source.stats() if hasattr(self.source, "streams") else {}
        msg = {"session": self.id, "t": t, "fps": round(fps, 2), "tracks": tr, "flags": []}
        if st:  # M6 multi-stream: per-stream health and the replay-fallback banner (null while every stream is live)
            msg["streams"] = [{k: s[k] for k in ("cam", "state", "fps", "since_frame_s")} for s in st["streams"]]
            msg["fallback"] = st["fallback"]
        return msg

    def summary(self) -> dict:
        with self.lock:
            wall = (self.wall_last - self.wall_first) if self.wall_first and self.wall_last else 0.0
            media = (self.t_media - self.t_media0) if self.t_media is not None else 0.0
            return {
                "session": self.id, "site": self.site.id, "mode": self.mode, "speed": self.speed,
                "label": self.source.label, "running": self.running, "error": self.error,
                "n_events": self.n_events, "n_tracks": len(self.all_tracks), "batches": self.batches,
                "media_seconds": round(media, 3), "wall_seconds": round(wall, 3),
                "fps_effective": round((self.batches - 1) / wall, 3) if wall > 0 else 0.0,
                "realtime_factor": round(media / wall, 3) if wall > 0 else 0.0,
                "source": self.source.stats(),
                "fallback": self.source.stats().get("fallback"),
            }
