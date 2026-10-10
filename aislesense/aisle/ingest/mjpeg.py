"""Stand-in phone camera: serves a looped video file as an MJPEG HTTP stream at real time, on the same URL shape as
the Android "IP Webcam" app (http://HOST:PORT/video, multipart/x-mixed-replace JPEG). Lets the live path, eval/fps.py
and the fallback test run through a real network URL without a phone.

  python -m aisle.ingest.mjpeg --file merl_x/Videos_MERL_Shopping_Dataset/27_1_crop.mp4 --port 8081

Run it in its own process (spawn()), like a real phone: cv2.VideoCapture(url) holds the GIL while it opens, so an
in-process server thread would starve and the open would time out.
"""
from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BOUNDARY = b"frame"


class MjpegCam:
    def __init__(self, path: str, host: str = "127.0.0.1", port: int = 0, fps: float | None = None, quality: int = 80):
        from aisle.schema import resolve_data_path
        self.path = str(resolve_data_path(path)) if "://" not in path else path
        self.fps, self.quality = fps, quality
        cam = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] != "/video":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    cam._stream(self.wfile)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    return

        self.httpd = ThreadingHTTPServer((host, port), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}/video"
        self._stopped = threading.Event()
        self._thread = None

    def _stream(self, w) -> None:
        import cv2
        while not self._stopped.is_set():
            cap = cv2.VideoCapture(self.path)
            fps = self.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
            t0, i = time.monotonic(), 0
            while not self._stopped.is_set():
                ok, fr = cap.read()
                if not ok:
                    break
                ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                w.write(b"--" + BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpg)).encode() + b"\r\n\r\n" + jpg.tobytes() + b"\r\n")
                i += 1
                d = t0 + i / fps - time.monotonic()
                if d > 0:
                    time.sleep(d)
            cap.release()

    def start(self) -> "MjpegCam":
        self._stopped.clear()
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Phone goes away: open connections end and new ones are refused."""
        self._stopped.set()
        self.httpd.shutdown()
        self.httpd.server_close()


def spawn(path: str, port: int, host: str = "127.0.0.1"):
    """Start a stand-in phone camera in a child process. Returns (Popen, url); kill the Popen to 'drop' the phone."""
    import subprocess
    import sys
    from aisle.schema import ROOT
    p = subprocess.Popen([sys.executable, "-m", "aisle.ingest.mjpeg", "--file", path, "--host", host, "--port", str(port)],
                         cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    url = p.stdout.readline().strip()
    if not url.startswith("http"):
        p.kill()
        raise RuntimeError(f"mjpeg stand-in failed on port {port}")
    return p, url


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    a = ap.parse_args(argv)
    c = MjpegCam(a.file, a.host, a.port).start()
    print(c.url, flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        c.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
