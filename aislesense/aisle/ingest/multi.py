"""M6 multi-stream live ingest: 1-3 concurrent cameras (phone IP-cam URLs or looped files), each with its own YOLO
instance and ByteTrack state, per-stream fps, reconnect, and replay fallback (F13 [A37]).

Per camera two threads:
  reader  opens the source (file or http:// / rtsp:// URL), applies the stride, hands frames on. Live (URL) and paced
          file sources keep only the newest frame, like a real camera (a slow GPU drops frames, never lags). Unpaced
          files (eval/fps.py capacity runs) hand over every frame. A URL that fails is reopened every RECONNECT_S.
  worker  runs detection + tracking on the newest frame and emits (t, events). If no frame has arrived for DROP_S
          seconds the camera switches to REPLAY: it loops the event rows it recorded while live (time-shifted, track
          ids suffixed) so the floor map keeps moving, and the session shows a banner. When frames return it goes LIVE.

Time base: paced sessions use wall seconds since session start for every camera (one floor clock for all streams);
unpaced capacity runs use each stream's media time. No frames are stored unless `record` is set, and recorded frames
are head-blurred (aisle/cv/blur.py, F14)."""
from __future__ import annotations

import itertools
import queue
import threading
import time
from pathlib import Path

from aisle.ingest.sources import Source, VideoSource
from aisle.schema import Event, VAR_DIR

DROP_S = 3.0
RECONNECT_S = 1.0
OPEN_TIMEOUT_MS = 3000
READ_TIMEOUT_MS = 3000
REPLAY_KEEP_S = 600.0


def open_capture(path: str, is_url: bool):
    import cv2
    if is_url:
        return cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, OPEN_TIMEOUT_MS,
                                                        cv2.CAP_PROP_READ_TIMEOUT_MSEC, READ_TIMEOUT_MS])
    return cv2.VideoCapture(path)


class CamStream:
    def __init__(self, vs: VideoSource, clock0: float, paced: bool, stop: threading.Event, out: queue.Queue,
                 recorder=None, drop_s: float = DROP_S):
        self.vs, self.clock0, self.paced, self.stop, self.out = vs, clock0, paced, stop, out
        self.recorder, self.drop_s = recorder, drop_s
        self.latest = paced or vs.is_url
        self.frames_q: queue.Queue = queue.Queue(maxsize=1 if self.latest else 4)
        self.state = "connecting"  # connecting | live | replay | ended
        self.last_frame_wall = None
        self.dropped_at = None
        self.n_drops = self.n_reconnects = self.frames_read = self.frames_skipped = 0
        self.replay_batches = 0
        self.media_s = 0.0
        self.history: list = []  # recorded (t, [(track, x, y, conf)]) while live, for the replay fallback
        self._fps_hist: list = []
        self.reader_done = False
        self.t_first = None
        self._loop = 0

    # ---------------------------------------------------------------- reader thread
    def _put(self, item) -> None:
        if self.latest:
            try:
                self.frames_q.get_nowait()
                self.frames_skipped += 1
            except queue.Empty:
                pass
            self.frames_q.put(item)
        else:
            while not self.stop.is_set():
                try:
                    self.frames_q.put(item, timeout=0.2)
                    return
                except queue.Full:
                    continue

    def reader(self) -> None:
        vs = self.vs
        try:
            for loop_k in itertools.count():
                if self.stop.is_set():
                    return
                cap = open_capture(vs.path, vs.is_url)
                if not cap.isOpened():
                    cap.release()
                    if not vs.is_url:
                        raise RuntimeError(f"cannot open {vs.path}")
                    self.n_reconnects += 1
                    self.stop.wait(RECONNECT_S)
                    continue
                import cv2
                fps = cap.get(cv2.CAP_PROP_FPS) or vs.fps
                if not (1.0 <= fps <= 120.0):
                    fps = vs.fps
                nfr = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                dur = nfr / fps if nfr > 0 else 0.0
                wall0 = time.monotonic()
                fi = -1
                try:
                    while not self.stop.is_set():
                        for _ in range(vs.stride - 1):
                            cap.grab()
                            fi += 1
                        ok, frame = cap.read()
                        fi += 1
                        if not ok:
                            break
                        self.frames_read += 1
                        media = fi / fps
                        if self.paced and not vs.is_url:  # a file played as a camera: real time
                            ahead = media - (time.monotonic() - wall0)
                            if ahead > 0 and self.stop.wait(ahead):
                                return
                        t_media = loop_k * dur + media
                        self._put((t_media, loop_k, frame))
                finally:
                    cap.release()
                if not vs.is_url and not vs.loop:
                    return
                if vs.is_url:
                    self.n_reconnects += 1
                    self.stop.wait(RECONNECT_S)
        finally:
            self.reader_done = True

    # ---------------------------------------------------------------- worker thread
    def now(self) -> float:
        return time.monotonic() - self.clock0

    def worker(self) -> None:
        vs = self.vs
        cam = vs.cam.id
        rk = 0
        while not self.stop.is_set():
            try:
                t_media, loop_k, frame = self.frames_q.get(timeout=0.1)
            except queue.Empty:
                if self.reader_done and self.frames_q.empty():
                    self.state = "ended"
                    return
                since = self.now() - (self.last_frame_wall if self.last_frame_wall is not None else 0.0)
                if self.state == "live" and since > self.drop_s:
                    self.n_drops += 1
                    self.state, self.dropped_at = "replay", self.now()
                if self.state == "replay" and self.history:
                    rk += 1
                    self._replay_once(rk)
                continue
            if loop_k != self._loop:
                self._loop = loop_k
                vs._reset_tracker()
            t = self.now() if self.paced else t_media
            evs = vs._events(frame, t, loop_k)
            if self.recorder is not None:
                self.recorder.write(frame, vs.last_xyxy)
            wall = self.now()
            if self.t_first is None:
                self.t_first = wall
            self.last_frame_wall = wall
            self.state = "live"
            self.media_s = t_media
            self._fps_hist.append(wall)
            if len(self._fps_hist) > 60:
                self._fps_hist = self._fps_hist[-60:]
            self.history.append((t, [(e.track, e.x, e.y, e.conf) for e in evs]))
            if self.history and self.history[-1][0] - self.history[0][0] > REPLAY_KEEP_S:
                self.history = self.history[len(self.history) // 10:]
            self.out.put((t, evs))

    def _replay_once(self, rk: int) -> None:
        """One pass over the recorded rows, shifted to the floor clock; stops early when frames come back."""
        h = self.history
        t0, base = h[0][0], self.now()
        sess, cam = self.vs.session, self.vs.cam.id
        for t, rows in h:
            if self.stop.is_set() or not self.frames_q.empty():
                return
            due = base + (t - t0)
            wait = due - self.now()
            if wait > 0 and self.stop.wait(min(wait, 1.0)):
                return
            tt = self.now()
            self.replay_batches += 1
            self.out.put((tt, [Event(sess, cam, tt, f"{k}#r{rk}", x, y, c) for k, x, y, c in rows]))

    def stats(self) -> dict:
        h = self._fps_hist
        fps = (len(h) - 1) / (h[-1] - h[0]) if len(h) > 1 and h[-1] > h[0] and self.state == "live" else 0.0
        wall = (self.last_frame_wall - self.t_first) if self.t_first is not None and self.last_frame_wall else 0.0
        return {"cam": self.vs.cam.id, "source": self.vs.path, "url": self.vs.is_url, "state": self.state,
                "fps": round(fps, 2), "frames": self.vs.frames, "frames_read": self.frames_read,
                "frames_skipped": self.frames_skipped, "infer_s": round(self.vs.infer_s, 3), "stride": self.vs.stride,
                "media_s": round(self.media_s, 3), "wall_s": round(wall, 3), "drops": self.n_drops,
                "reconnects": self.n_reconnects, "replay_batches": self.replay_batches,
                "since_frame_s": round(self.now() - self.last_frame_wall, 2) if self.last_frame_wall is not None else None}


class MultiSource(Source):
    """1-3 cameras merged into one stream of (t, events) batches. Pacing is done per camera, not by the runner."""
    paced = False

    def __init__(self, site, cams, session: str, model_path: str, tracker_cfg: str, paced: bool = True,
                 record: bool = False, drop_s: float = DROP_S, device=None):
        self.site, self.session, self.paced_streams = site, session, paced
        self.subs = [VideoSource(site, c, session, model_path, tracker_cfg, stride=int(site.extra.get("stride", 2)),
                                 loop=bool(site.extra.get("loop", True)), paced=paced, device=device) for c in cams]
        v0 = self.subs[0]
        self.nominal_dt, self.max_gap, self.death_gap = v0.nominal_dt, v0.max_gap, v0.death_gap
        self.label = site.extra.get("label", "video")
        self.record, self.drop_s = record, drop_s
        self._stop = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=256)
        self.streams: list[CamStream] = []
        self.recorders = []

    def __iter__(self):
        clock0 = time.monotonic()
        threads = []
        for vs in self.subs:
            rec = None
            if self.record:
                from aisle.cv.blur import Recorder
                rec = Recorder(VAR_DIR / "recordings" / f"{self.session}_{vs.cam.id}.mp4", vs.fps / vs.stride)
                self.recorders.append(rec)
            cs = CamStream(vs, clock0, self.paced_streams, self._stop, self.q, rec, self.drop_s)
            self.streams.append(cs)
            for fn in (cs.reader, cs.worker):
                th = threading.Thread(target=fn, name=f"{vs.cam.id}-{fn.__name__}", daemon=True)
                th.start()
                threads.append(th)
        try:
            while True:
                try:
                    yield self.q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set() or (not any(th.is_alive() for th in threads) and self.q.empty()):
                        return
        finally:
            self._stop.set()
            for th in threads:
                th.join(5)
            for r in self.recorders:
                r.close()

    def close(self) -> None:
        self._stop.set()

    def fallback(self) -> str | None:
        rep = [s for s in self.streams if s.state == "replay"]
        if not rep:
            return None
        names = ", ".join(s.vs.cam.id for s in rep)
        return (f"Live stream {names} dropped: showing a replay of its recorded event rows until it reconnects "
                f"(no new data)")

    def stats(self) -> dict:
        return {"streams": [s.stats() for s in self.streams], "fallback": self.fallback(), "n_streams": len(self.subs),
                "record": self.record}
