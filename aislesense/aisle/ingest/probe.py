"""Check a camera URL (or file) before a demo: opens it the way the app does and reports resolution and delivered fps.

  python -m aisle.ingest.probe http://192.168.1.23:8080/video --seconds 5
Exit 0 when frames arrive, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time


def probe(src: str, seconds: float = 5.0) -> dict:
    from aisle.ingest.multi import open_capture
    t0 = time.monotonic()
    cap = open_capture(src, "://" in src)
    opened = cap.isOpened()
    t_open = time.monotonic() - t0
    n, shape = 0, None
    t1 = time.monotonic()
    while opened and time.monotonic() - t1 < seconds:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        shape = fr.shape
    wall = time.monotonic() - t1
    cap.release()
    return {"source": src, "opened": opened, "open_s": round(t_open, 2), "frames": n,
            "fps": round(n / wall, 2) if wall > 0 else 0.0, "width": shape[1] if shape else None,
            "height": shape[0] if shape else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--seconds", type=float, default=5.0)
    a = ap.parse_args(argv)
    r = probe(a.source, a.seconds)
    print(r)
    return 0 if r["frames"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
