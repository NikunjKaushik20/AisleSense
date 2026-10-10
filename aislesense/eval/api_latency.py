"""M6: dashboard panel latency on a running ATC replay, M4 behaviour (rebuild on request) vs M6 prefetch + stale-while-
revalidate -> results/api_latency.json.

Each arm starts a fresh app (own sqlite under var/cache/), replays ATC 2012-11-14 at --speed (the demo setting), lets it
ingest for --warm seconds, then plays a user switching dashboard tabs: one panel request every --every seconds for
--measure seconds, cycling dwell, tips, dead zones (keep 1 and 1%), flags, journeys. Wall latency per request.

  python eval/api_latency.py --warm 60 --measure 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import write_result  # noqa: E402

import numpy as np  # noqa: E402

from aisle.schema import VAR_DIR  # noqa: E402

TABS = [("/api/dwell", {}), ("/api/tips", {}), ("/api/deadzones", {"keep": 1.0}), ("/api/flags", {}),
        ("/api/deadzones", {"keep": 0.01}), ("/api/journeys", {})]


def arm(prefetch: bool, speed: float, warm: float, measure: float, every: float) -> dict:
    from fastapi.testclient import TestClient
    from aisle.api.app import create_app
    db = VAR_DIR / "cache" / f"latency_{'on' if prefetch else 'off'}.sqlite"
    for suf in ("", "-wal", "-shm"):
        Path(str(db) + suf).unlink(missing_ok=True)
    app = create_app(store_path=str(db), autostart={"site": "atc", "mode": "replay", "speed": speed}, prefetch=prefetch)
    lat = []
    with TestClient(app) as c:
        time.sleep(warm)
        t_end = time.monotonic() + measure
        i = 0
        while time.monotonic() < t_end:
            path, q = TABS[i % len(TABS)]
            t0 = time.perf_counter()
            r = c.get(path, params=q)
            dt = time.perf_counter() - t0
            assert r.status_code == 200, (path, r.status_code, r.text[:200])
            lat.append({"path": path + (f"?keep={q['keep']}" if q else ""), "s": round(dt, 4)})
            i += 1
            rest = every - dt
            if rest > 0:
                time.sleep(rest)
        sess = c.get("/api/session").json()
        stats = c.get("/api/panel_stats").json()
    c.app.state.mgr.stop()
    x = np.array([v["s"] for v in lat])
    return {"prefetch": prefetch, "n_requests": len(lat), "p50_s": round(float(np.median(x)), 4),
            "p95_s": round(float(np.percentile(x, 95)), 4), "max_s": round(float(x.max()), 4),
            "share_under_1s": round(float((x < 1.0).mean()), 3),
            "events_at_end": sess.get("n_events"), "media_hours_at_end": round(sess.get("media_seconds", 0) / 3600, 2),
            "panel_stats": stats, "requests": lat}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=600)
    ap.add_argument("--warm", type=float, default=60)
    ap.add_argument("--measure", type=float, default=60)
    ap.add_argument("--every", type=float, default=3.0)
    a = ap.parse_args(argv)
    before = arm(False, a.speed, a.warm, a.measure, a.every)
    print(f"before (M4, rebuild on request): p50 {before['p50_s']} s p95 {before['p95_s']} s max {before['max_s']} s", flush=True)
    after = arm(True, a.speed, a.warm, a.measure, a.every)
    print(f"after (M6 prefetch + SWR): p50 {after['p50_s']} s p95 {after['p95_s']} s max {after['max_s']} s", flush=True)
    res = {"data": "ATC 2012-11-14 trajectory replay (running session)",
           "config": {"speed": a.speed, "warm_s": a.warm, "measure_s": a.measure, "every_s": a.every,
                      "tabs": [p + (f"?keep={q['keep']}" if q else "") for p, q in TABS], "client": "fastapi TestClient, in process"},
           "seed": None, "n": {"requests_before": before["n_requests"], "requests_after": after["n_requests"]},
           "metric": {"before": before, "after": after},
           "prior": {"name": "M4 behaviour (rebuild a running session's panel on the first request after each 10 s version)",
                     "p95_s": before["p95_s"]},
           "notes": "after = panels of the current session are rebuilt by a background prefetch thread; a request gets the "
                    "newest finished build (at most one refresh interval old while the session runs)."}
    p = write_result("api_latency", res)
    print(p.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
