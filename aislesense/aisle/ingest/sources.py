"""Event sources. Each yields (t_media, [Event]) in media-time order.

paced=True means the runner throttles to `speed` x real time (replays, synthetic, live file loops in
server mode). Video sources do their own work per frame and are throttled only when paced.
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from aisle.schema import Event, VAR_DIR, default_model, resolve_data_path


class Source:
    paced = True
    label = ""
    nominal_dt = 1.0
    max_gap = 2.0
    death_gap = 5.0

    def __iter__(self):  # pragma: no cover - interface
        raise NotImplementedError

    def stats(self) -> dict:
        return {}

    def close(self) -> None:
        pass


class SynthSource(Source):
    """Replays generator rows; loops with a media-time offset so a live demo never runs dry.
    Each loop re-generates with the next seed, so loop 2 is new shoppers, not a rerun of loop 1."""

    def __init__(self, site, session: str, loop: bool = True):
        from synth.generator import generate, load_scenario
        sc = load_scenario(site)
        self.rows, self.truth = generate(sc, site)
        self._regen = lambda k: generate({**sc, "seed": int(sc.get("seed", 0)) + k}, site)[0]
        self.T = float(sc.get("duration_s", 600))
        self.nominal_dt = 1.0 / float(sc.get("hz", 5))
        self.max_gap, self.death_gap = 1.0, 3.0
        self.session, self.loop = session, loop
        self.label = site.extra.get("label", "synthetic")

    def __iter__(self):
        for k in itertools.count():
            off = k * self.T
            rows = self.rows if k == 0 else self._regen(k)
            for t, grp in itertools.groupby(rows, key=lambda r: r[0]):
                yield t + off, [Event(self.session, "synth", t + off, f"{tr}#{k}" if k else tr, x, y, 1.0) for _, tr, x, y in grp]
            if not self.loop:
                return


def atc_1hz(npz_path: Path) -> dict:
    """ATC day -> 1 Hz per-person mean positions in metres. Cached under var/cache/."""
    cache = VAR_DIR / "cache" / (npz_path.stem + "_1hz.npz")
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    import pandas as pd
    z = np.load(npz_path)
    a = pd.DataFrame({"s": np.floor(z["t"]).astype(np.int64), "id": z["id"], "x": z["x"], "y": z["y"]})
    b = a.groupby(["s", "id"], sort=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index()
    out = {"s": b.s.to_numpy(), "id": b.id.to_numpy(), "x": (b.x.to_numpy() / 1e3).astype(np.float32), "y": (b.y.to_numpy() / 1e3).astype(np.float32)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out


class AtcSource(Source):
    """ATC mall trajectories (mm -> m, 1 Hz). No CV: always labelled trajectory replay [D10]."""
    nominal_dt, max_gap, death_gap = 1.0, 2.0, 5.0

    def __init__(self, site, session: str, start_offset_s: float = 0.0):
        self.d = atc_1hz(resolve_data_path(site.extra["day"]))
        self.session = session
        self.label = site.extra.get("label", "Trajectory replay")
        s = self.d["s"]
        self.i0 = int(np.searchsorted(s, s[0] + start_offset_s))
        # boundaries of each 1 s group
        self.bounds = np.flatnonzero(np.diff(s[self.i0:])) + 1 + self.i0

    def __iter__(self):
        s, ids, x, y = self.d["s"], self.d["id"], self.d["x"], self.d["y"]
        starts = np.r_[self.i0, self.bounds]
        ends = np.r_[self.bounds, len(s)]
        for a, b in zip(starts, ends):
            t = float(s[a])
            yield t, [Event(self.session, "atc", t, str(ids[i]), float(x[i]), float(y[i]), 1.0) for i in range(a, b)]


class VideoSource(Source):
    """Video file (looped) or phone/RTSP URL -> YOLO11n person boxes -> ByteTrack -> foot point -> floor.

    No frames are stored (privacy default, F14)."""

    def __init__(self, site, cam, session: str, model_path: str, tracker_cfg: str, stride: int = 2,
                 loop: bool = True, paced: bool = True, device=None, conf: float = 0.1, imgsz: int = 640):
        import cv2
        from aisle.geo import homography as hg
        from aisle.cv.detect import load_model
        self.cv2, self.hg = cv2, hg
        self.site, self.cam, self.session = site, cam, session
        src = cam.source
        self.is_url = "://" in src
        self.path = src if self.is_url else str(resolve_data_path(src))
        if not self.is_url and not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self.H = hg.fit(cam.image_points, cam.floor_points)
        self.model = load_model(model_path)
        self.tracker_cfg, self.stride, self.loop, self.paced = tracker_cfg, int(stride), loop and not self.is_url, paced
        self.device, self.conf, self.imgsz = device, conf, imgsz
        self.fps = float(site.extra.get("fps", 30))
        self.nominal_dt = self.stride / self.fps
        self.max_gap, self.death_gap = 1.0, 3.0
        self.label = site.extra.get("label", "video")
        self.frames = 0
        self.infer_s = 0.0

    def _reset_tracker(self):
        p = getattr(self.model, "predictor", None)
        for tr in getattr(p, "trackers", None) or []:
            tr.reset()

    def __iter__(self):
        cv2 = self.cv2
        for loop_k in itertools.count():
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open {self.path}")
            fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
            nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            dur = nfr / fps if nfr > 0 else 0.0
            if loop_k:
                self._reset_tracker()
            wall0 = time.monotonic()
            fi = -1
            try:
                while True:
                    ok = True
                    for _ in range(self.stride - 1):
                        ok = cap.grab() and ok
                        fi += 1
                    ok, frame = cap.read()
                    fi += 1
                    if not ok:
                        break
                    t = (time.monotonic() - wall0) if self.is_url else loop_k * dur + fi / fps
                    yield t, self._events(frame, t, loop_k)
            finally:
                cap.release()
            if not self.loop:
                return

    def _events(self, frame, t, loop_k):
        t0 = time.perf_counter()
        r = self.model.track(frame, persist=True, classes=[0], conf=self.conf, imgsz=self.imgsz,
                             tracker=self.tracker_cfg, verbose=False, device=self.device)[0]
        self.infer_s += time.perf_counter() - t0
        self.frames += 1
        b = r.boxes
        self.last_xyxy = b.xyxy.cpu().numpy() if b is not None and len(b) else []
        self.last_ids = b.id.int().tolist() if b is not None and b.id is not None and len(b) else []
        if b is None or b.id is None or len(b) == 0:
            return []
        ids = self.last_ids
        conf = b.conf.cpu().numpy()
        fl = self.hg.apply(self.H, self.hg.foot_points(b.xyxy.cpu().numpy()))
        return [Event(self.session, self.cam.id, float(t), f"{self.cam.id}:{loop_k}:{i}", float(p[0]), float(p[1]), float(c))
                for i, p, c in zip(ids, fl, conf)]

    def stats(self) -> dict:
        return {"frames": self.frames, "infer_s": round(self.infer_s, 3), "stride": self.stride}


def make_source(site, session: str, mode: str, sources=None, paced: bool = True, cfg: dict | None = None) -> Source:
    from aisle.cv.track import DEFAULT_BUFFER, tracker_yaml
    cfg = cfg or {}
    if site.kind == "synth":
        return SynthSource(site, session)
    if site.kind == "atc":
        return AtcSource(site, session, float(cfg.get("start_offset_s", site.extra.get("start_offset_s", 0))))
    if site.kind == "video":
        if mode != "live":
            raise ValueError("video sites run in live mode; use atc or synth_aisle for event replay")
        cams = list(site.cameras)
        if sources:
            from dataclasses import replace
            cams = [replace(c, source=s) for c, s in zip(cams, sources)]
        tb = tracker_yaml(int(cfg.get("track_buffer") or DEFAULT_BUFFER))
        model = cfg.get("model", default_model())
        if len(cams) > 1 or paced or any("://" in c.source for c in cams):
            # M6: 1-3 concurrent cameras, reconnect, replay fallback, optional blurred recording
            from aisle.ingest.multi import MultiSource
            if len(cams) > 3:
                raise ValueError("at most 3 concurrent streams (results/fps.json)")
            return MultiSource(site, cams, session, model, tb, paced=paced,
                               record=bool(cfg.get("record", site.extra.get("record", False))))
        return VideoSource(site, cams[0], session, cfg.get("model", default_model()),
                           tb, stride=int(site.extra.get("stride", 2)),
                           loop=bool(site.extra.get("loop", True)), paced=paced)
    raise ValueError(f"unknown site kind {site.kind}")
