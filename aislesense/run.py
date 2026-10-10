"""AisleSense one command.

  python run.py --site synth_aisle                      # server on http://127.0.0.1:8000/ with that site live
  python run.py --site merl_demo --mode live --headless --seconds 60   # smoke run -> results/m1_smoke.json
  python run.py --site atc --mode replay --headless --seconds 30 --speed 600
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from aisle.schema import RESULTS_DIR, load_site


def default_mode(site_id: str) -> str:
    return "live" if load_site(site_id).kind == "video" else "replay"


def headless(a) -> int:
    from aisle.ingest.runner import Session
    from aisle.store.sqlite import Store
    t_start = time.monotonic()
    store = Store(a.db) if a.db else Store()
    s = Session(a.site, mode=a.mode, speed=a.speed, paced=a.mode == "replay", store=store,
                cfg={"track_buffer": a.track_buffer, "model": a.model}).start()
    t_load = time.monotonic()
    deadline = t_load + a.seconds
    while time.monotonic() < deadline and s.running:
        time.sleep(0.2)
    s.stop()
    store.close()
    out = s.summary()
    out.update({"seconds_requested": a.seconds, "wall_total_s": round(time.monotonic() - t_start, 3),
                "notes": "fps_effective/realtime_factor measured from first batch to last (excludes model load); "
                         "video headless runs unpaced, so realtime_factor is processing capacity"})
    if s.site.kind == "video":
        out.update({"model": a.model, "track_buffer": a.track_buffer})
    name = a.out or ("m1_smoke" if a.site == "merl_demo" else f"m1_smoke_{a.site}")
    RESULTS_DIR.mkdir(exist_ok=True)
    p = RESULTS_DIR / f"{name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"{p.name}: n_events={out['n_events']} n_tracks={out['n_tracks']} fps_effective={out['fps_effective']} "
          f"realtime_factor={out['realtime_factor']} error={out['error']}")
    return 0 if out["error"] is None and out["n_events"] > 0 else 1


def serve(a) -> int:
    import uvicorn
    from aisle.api.app import create_app
    app = create_app(store_path=a.db, autostart={"site": a.site, "mode": a.mode, "speed": a.speed},
                     cfg={"track_buffer": a.track_buffer, "model": a.model})
    print(f"AisleSense on http://{a.host}:{a.port}/  (site={a.site}, mode={a.mode})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="AisleSense")
    ap.add_argument("--site", default="synth_aisle")
    ap.add_argument("--mode", choices=["live", "replay"])
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=None, help="sqlite path (default var/aisle.sqlite)")
    ap.add_argument("--out", default=None, help="results/<out>.json name for headless runs")
    ap.add_argument("--model", default=None, help="YOLO weights (default: AISLE_MODEL, models/yolo11n.pt, or auto-download)")
    ap.add_argument("--track-buffer", type=int, default=200)  # M3 S1 (aisle.cv.track.DEFAULT_BUFFER)
    a = ap.parse_args(argv)
    if a.model is None:
        from aisle.schema import default_model
        a.model = default_model()
    a.mode = a.mode or default_mode(a.site)
    return headless(a) if a.headless else serve(a)


if __name__ == "__main__":
    sys.exit(main())
