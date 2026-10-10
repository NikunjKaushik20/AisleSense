"""M6 multi-stream throughput on this GPU -> results/fps.json.

For N in --streams (default 1,2,3) concurrent cameras, each a different MERL test clip, yolo11n + ByteTrack b200,
stride 2, through aisle.ingest.multi.MultiSource (the code path the server uses):

  file  unpaced looped files: capacity. realtime_factor = media seconds processed / wall seconds, per stream.
        A stream count is demo-ok when every stream has realtime_factor >= 1.0 (BUILD M6).
  http  each clip served as a phone-style MJPEG URL (aisle.ingest.mjpeg, IP Webcam URL shape, own process, real
        time). The camera does not wait for us, so slowness shows as skipped frames: keep_ratio = frames processed /
        frames the stride asks for (1.0 = no drops). This is the honest live number.

  python eval/fps.py --streams 1,2,3 --seconds 60 [--transport file,http]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import write_result  # noqa: E402

from aisle.cv.track import DEFAULT_BUFFER, tracker_yaml  # noqa: E402
from aisle.ingest.multi import MultiSource  # noqa: E402
from aisle.schema import default_model, load_site  # noqa: E402

CLIPS = ["merl_x/Videos_MERL_Shopping_Dataset/27_1_crop.mp4", "merl_x/Videos_MERL_Shopping_Dataset/28_1_crop.mp4",
         "merl_x/Videos_MERL_Shopping_Dataset/29_1_crop.mp4"]
MODEL = default_model()
WARMUP_S = 5.0
BASE_PORT = 8710


def run(n: int, seconds: float, transport: str) -> dict:
    site = load_site("merl_demo")
    procs, srcs = [], []
    if transport == "http":
        from aisle.ingest.mjpeg import spawn
        for i in range(n):
            p, url = spawn(CLIPS[i], BASE_PORT + i)
            procs.append(p)
            srcs.append(url)
    else:
        srcs = CLIPS[:n]
    cams = [replace(site.cameras[0], id=f"cam{i}", source=s) for i, s in enumerate(srcs)]
    ms = MultiSource(site, cams, f"fps_{transport}_{n}", MODEL, tracker_yaml(DEFAULT_BUFFER), paced=transport == "http")
    t_start = time.monotonic()
    snap0 = None
    n_events = 0
    try:
        for _, evs in ms:
            el = time.monotonic() - t_start
            n_events += len(evs)
            if snap0 is None and el >= WARMUP_S and all(s.state == "live" for s in ms.streams) and len(ms.streams) == n:
                snap0 = (time.monotonic(), [dict(s.stats()) for s in ms.streams])
            if snap0 is not None and time.monotonic() - snap0[0] >= seconds:
                break
            if el > seconds + 120:
                raise RuntimeError("streams never all went live")
        snap1 = (time.monotonic(), [dict(s.stats()) for s in ms.streams])
    finally:
        ms.close()
        for p in procs:
            p.kill()
    wall = snap1[0] - snap0[0]
    per = []
    for a, b in zip(snap0[1], snap1[1]):
        frames = b["frames"] - a["frames"]
        row = {"cam": b["cam"], "source": Path(b["source"]).name if not b["url"] else b["source"], "frames": frames,
               "processed_fps": round(frames / wall, 2), "infer_ms_per_frame": round(1e3 * (b["infer_s"] - a["infer_s"]) / max(frames, 1), 1),
               "drops": b["drops"]}
        if transport == "file":
            media = b["media_s"] - a["media_s"]
            if media < 0:  # wrapped a loop: count frames instead (stride 2 at 30 fps)
                media = frames * b["stride"] / 30.0
            row["realtime_factor"] = round(media / wall, 3)
        else:
            wanted = (b["frames_read"] - a["frames_read"])  # frames delivered by the camera after the stride
            row["frames_from_camera"] = wanted
            row["keep_ratio"] = round(frames / max(wanted, 1), 3)
        per.append(row)
    key = "realtime_factor" if transport == "file" else "keep_ratio"
    worst = min(r[key] for r in per)
    return {"streams": n, "transport": transport, "wall_s": round(wall, 2), "per_stream": per,
            f"min_{key}": worst, "total_processed_fps": round(sum(r["processed_fps"] for r in per), 2),
            "demo_ok": bool(worst >= (1.0 if transport == "file" else 0.95))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="1,2,3")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--transport", default="file,http")
    a = ap.parse_args(argv)
    import torch
    rows = []
    for tr in a.transport.split(","):
        for n in [int(x) for x in a.streams.split(",")]:
            r = run(n, a.seconds, tr)
            rows.append(r)
            k = "realtime_factor" if tr == "file" else "keep_ratio"
            print(f"{tr} x{n}: min {k} {r['min_' + k]}, total {r['total_processed_fps']} fps, demo_ok {r['demo_ok']}", flush=True)
    files = [r for r in rows if r["transport"] == "file"]
    http = [r for r in rows if r["transport"] == "http"]
    res = {
        "data": "MERL Shopping test clips 27_1, 28_1, 29_1 (one per stream), looped",
        "config": {"model": "yolo11n.pt", "tracker": f"bytetrack b{DEFAULT_BUFFER}", "stride": 2, "imgsz": 640,
                   "warmup_s": WARMUP_S, "seconds": a.seconds, "code_path": "aisle.ingest.multi.MultiSource",
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"},
        "seed": None, "n": {"runs": len(rows)},
        "metric": {"runs": rows},
        "prior": {"name": "1-stream realtime factor in X1 (phase C script)", "value": 2.5, "tolerance": 0.7},
        "max_demo_streams_file": max([r["streams"] for r in files if r["demo_ok"]], default=0),
        "max_demo_streams_http": max([r["streams"] for r in http if r["demo_ok"]], default=0),
        "notes": "file = capacity (unpaced); http = phone-like MJPEG URL at real time, keep_ratio < 1 means frames were "
                 "skipped (the floor map still updates, at a lower rate). Demo uses stream counts that are demo_ok.",
    }
    p = write_result("fps", res)
    print(f"{p.name}: max demo streams file={res['max_demo_streams_file']} http={res['max_demo_streams_http']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
