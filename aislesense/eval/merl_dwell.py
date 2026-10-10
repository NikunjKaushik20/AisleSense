"""MERL Shopping dwell / browse eval (baseline: yolo11n + ByteTrack buffer 30, stride 2, no ghost hold).

Split frozen as X8: val = subjects 21-26 (only place anything is chosen: the prior's browse share), test = 27-41 (28 clips,
scored). Metrics on test: frame browse F1 (person tracked) vs all-browse prior, IDs per clip (1 shopper per clip),
per-clip browse seconds MAE for longest ID and summed tracked time vs prior (val browse share x clip length).

  --from-dump (default)  read per-frame tracks from nikunj/scripts/out/merl_dump/ (same cfg as X1), mapped to floor events
                         through the app homography (aisle.geo.homography) and scored from aisle Event rows.
  --clips N              run the live CV path (aisle.ingest.sources.VideoSource, GPU) on the first N test clips and check
                         IDs per clip against nikunj/scripts/out/merl_rows.json -> results/merl_dwell_live.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import NIKUNJ_OUT, fingerprint, prf, write_result  # noqa: E402

import numpy as np  # noqa: E402

from aisle.geo import homography as hg  # noqa: E402
from aisle.schema import Event, data_root, default_model, load_site  # noqa: E402

FPS, STRIDE = 30.0, 2
VAL_SUBJECTS = range(21, 27)
TEST_SUBJECTS = range(27, 42)
EXPECTED = {"tracked_F1": 0.787, "prior_F1": 0.866, "ids_per_clip": 14.64, "longest_id_MAE_s": 74.1, "prior_MAE_s": 6.5}


def subject(clip: str) -> int:
    return int(clip.split("_")[0])


def split_of(clip: str) -> str:
    s = subject(clip)
    return "val" if s in VAL_SUBJECTS else "test" if s in TEST_SUBJECTS else "other"


def merl_dir() -> Path:
    return data_root() / "merl_x"


def gt_mask(clip: str, n_frames: int) -> np.ndarray:
    import scipy.io as sio
    tl = sio.loadmat(str(merl_dir() / "Labels_MERL_Shopping_Dataset" / f"{clip}_label.mat"))["tlabs"]
    g = np.zeros(n_frames + STRIDE + 1, bool)
    for k in range(5):
        for s, e in tl[k][0].reshape(-1, 2):
            g[int(s) - 1:int(e)] = True
    return g


def clip_metrics(frame_idx: np.ndarray, events_per_frame: list, gt: np.ndarray) -> dict:
    """Score one clip from per-processed-frame lists of aisle Events. Used by both the dump and the live path."""
    g = gt[frame_idx]
    present = np.array([len(ev) > 0 for ev in events_per_frame], bool)
    counts = {}
    for ev in events_per_frame:
        for e in {e.track for e in ev}:
            counts[e] = counts.get(e, 0) + 1
    dt = STRIDE / FPS

    def tri(pred):
        return [int((pred & g).sum()), int((pred & ~g).sum()), int((~pred & g).sum())]
    return {"frames": int(len(frame_idx)), "dur_s": len(frame_idx) * dt, "gt_s": float(g.sum() * dt),
            "trk_s": float(present.sum() * dt), "n_ids": len(counts),
            "n_ids_2s": int(sum(c * dt >= 2.0 for c in counts.values())),
            "longest_id_s": max(counts.values(), default=0) * dt,
            "pres": tri(present), "prior": tri(np.ones_like(g))}


def dump_events(clip: str, H, session: str = "merl_eval"):
    a = np.load(NIKUNJ_OUT / "merl_dump" / f"{clip}.npy")
    last = int(a[a[:, 1] == -9][0, 0])
    a = a[a[:, 1] != -9]
    fi = np.arange(0, last + 1, STRIDE)
    by = {int(f): [] for f in fi}
    if len(a):
        fl = hg.apply(H, hg.foot_points(a[:, 2:6]))
        for r, p in zip(a, fl):
            by.setdefault(int(r[0]), []).append(Event(session, "cam0", r[0] / FPS, f"cam0:0:{int(r[1])}", float(p[0]), float(p[1]), float(r[6])))
    return fi, [by[int(f)] for f in fi]


def choose_prior_share(val_rows: list) -> float:
    """The only fitted quantity of the baseline eval. Must only ever see val clips (asserted)."""
    assert all(split_of(r["clip"]) == "val" for r in val_rows), "prior share must come from val subjects 21-26 only"
    return sum(r["gt_s"] for r in val_rows) / sum(r["dur_s"] for r in val_rows)


def summarize(test_rows: list, share: float) -> dict:
    A = lambda k: np.array([r[k] for r in test_rows], float)  # noqa: E731
    g = A("gt_s")
    tot = lambda k: np.sum([r[k] for r in test_rows], 0)  # noqa: E731
    prior_est = share * A("dur_s")
    return {
        "frame_browse": {"tracked": prf(*tot("pres")), "prior_all_browse": prf(*tot("prior"))},
        "ids_per_clip": {"mean": A("n_ids").mean(), "median": float(np.median(A("n_ids"))), "max": A("n_ids").max(),
                         "ids_2s_mean": A("n_ids_2s").mean(), "true_shoppers_per_clip": 1},
        "per_clip_browse_s": {
            "longest_id": {"MAE": np.abs(A("longest_id_s") - g).mean(), "bias": np.mean(A("longest_id_s") - g)},
            "tracked_time": {"MAE": np.abs(A("trk_s") - g).mean(), "bias": np.mean(A("trk_s") - g)},
            "prior": {"MAE": np.abs(prior_est - g).mean(), "bias": np.mean(prior_est - g), "val_browse_share": share},
        },
        "gt_browse_share_test": g.sum() / A("dur_s").sum(),
    }


def run_dump(args) -> dict:
    site = load_site("merl_demo")
    cam = site.cameras[0]
    H = hg.fit(cam.image_points, cam.floor_points)
    files = sorted(glob.glob(str(NIKUNJ_OUT / "merl_dump" / "*.npy")))
    rows = []
    for f in files:
        clip = os.path.basename(f)[:-4]
        if split_of(clip) == "other":
            continue
        fi, evs = dump_events(clip, H)
        r = clip_metrics(fi, evs, gt_mask(clip, int(fi[-1])))
        rows.append({"clip": clip, "split": split_of(clip), **r})
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    share = choose_prior_share(val)
    S = summarize(test, share)
    out = {
        "data": "MERL Shopping (real overhead store video), test subjects 27-41",
        "source": "tracks from nikunj/scripts/out/merl_dump (yolo11n, ByteTrack buffer 30, conf 0.1, imgsz 640, stride 2) "
                  "-> aisle homography -> aisle Event rows",
        "config": {"model": "yolo11n", "track_buffer": 30, "stride": STRIDE, "fps": FPS, "ghost_hold": None,
                   "choose_split": "val subjects 21-26", "score_split": "test subjects 27-41"},
        "seed": 0, "data_hash": fingerprint(files),
        "n": {"test_clips": len(test), "val_clips": len(val), "test_seconds": sum(r["dur_s"] for r in test)},
        "prior_from_subjects": sorted({subject(r["clip"]) for r in val}),
        "scored_subjects": sorted({subject(r["clip"]) for r in test}),
        "metric": S,
        "prior": {"frame_browse_F1": S["frame_browse"]["prior_all_browse"]["F1"], "frame_browse_name": "every frame is browse",
                  "per_clip_MAE_s": S["per_clip_browse_s"]["prior"]["MAE"], "per_clip_name": "val browse share x clip length",
                  "val_browse_share": share},
        "headline": [
            {"metric": "frame browse F1", "value": S["frame_browse"]["tracked"]["F1"],
             "prior": S["frame_browse"]["prior_all_browse"]["F1"], "prior_name": "all-browse", "beats_prior": S["frame_browse"]["tracked"]["F1"] > S["frame_browse"]["prior_all_browse"]["F1"]},
            {"metric": "per-clip browse s MAE (longest ID)", "value": S["per_clip_browse_s"]["longest_id"]["MAE"],
             "prior": S["per_clip_browse_s"]["prior"]["MAE"], "prior_name": "val share x clip length", "unit": "s per clip",
             "beats_prior": S["per_clip_browse_s"]["longest_id"]["MAE"] < S["per_clip_browse_s"]["prior"]["MAE"]},
            {"metric": "IDs per clip (1 true shopper)", "value": S["ids_per_clip"]["mean"], "prior": 1.0, "prior_name": "truth"},
        ],
        "expected_X1_X8": EXPECTED,
        "per_clip": test,
    }
    write_result("merl_dwell", out)
    return out


def run_live(n: int) -> dict:
    """Live CV path through the app's VideoSource on the first n test clips."""
    import time
    from dataclasses import replace
    from aisle.cv.track import tracker_yaml
    from aisle.ingest.sources import VideoSource
    site = load_site("merl_demo")
    ref = {r["clip"]: r for r in json.loads((NIKUNJ_OUT / "merl_rows.json").read_text())}
    clips = sorted(c for c in ref if split_of(c) == "test")[:n]
    rows = []
    t_all = time.time()
    for clip in clips:
        cam = replace(site.cameras[0], source=f"merl_x/Videos_MERL_Shopping_Dataset/{clip}_crop.mp4")
        src = VideoSource(site, cam, "merl_live", default_model(), tracker_yaml(30), stride=STRIDE, loop=False, paced=False)
        fi, evs = [], []
        for t, ev in src:
            fi.append(int(round(t * FPS)))
            evs.append(ev)
        fi = np.asarray(fi)
        r = clip_metrics(fi, evs, gt_mask(clip, int(fi[-1]) + STRIDE))
        rows.append({"clip": clip, **r, "ref_n_ids": ref[clip]["n_ids"], "ref_trk_s": ref[clip]["trk_s"],
                     "infer_s": src.stats()["infer_s"]})
        print(f"{clip}: ids {r['n_ids']} (ref {ref[clip]['n_ids']}) trk_s {r['trk_s']:.1f} (ref {ref[clip]['trk_s']:.1f})", flush=True)
    ids = np.array([r["n_ids"] for r in rows], float)
    rid = np.array([r["ref_n_ids"] for r in rows], float)
    rel = float(np.mean(np.abs(ids - rid) / np.maximum(rid, 1)))
    out = {
        "data": "MERL Shopping test clips, live CV path (aisle.ingest.sources.VideoSource on GPU)",
        "config": {"model": "yolo11n", "track_buffer": 30, "stride": STRIDE, "clips": clips,
                   "note": "app VideoSource path vs reference merl_yolo.py rows (same model, tracker cfg, stride); IDs and tracked seconds compared per clip"},
        "seed": 0, "n": {"clips": len(rows)},
        "metric": {"ids_per_clip_mean": ids.mean(), "ref_ids_per_clip_mean": rid.mean(), "mean_rel_diff_ids": rel,
                   "trk_s_mae_vs_ref": float(np.mean([abs(r["trk_s"] - r["ref_trk_s"]) for r in rows])),
                   "frame_browse_tracked": prf(*np.sum([r["pres"] for r in rows], 0)),
                   "frame_browse_prior": prf(*np.sum([r["prior"] for r in rows], 0))},
        "tolerance": {"mean_rel_diff_ids_max": 0.5},
        "wall_s": round(time.time() - t_all, 1),
        "per_clip": rows,
    }
    out["pass"] = rel <= 0.5
    write_result("merl_dwell_live", out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dump", action="store_true", help="(default) score cached per-frame tracks")
    ap.add_argument("--clips", type=int, default=0, help="run live CV path on N test clips")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args(argv)
    if a.clips:
        o = run_live(a.clips)
        print(f"merl live: ids/clip {o['metric']['ids_per_clip_mean']:.2f} vs ref {o['metric']['ref_ids_per_clip_mean']:.2f}, rel diff {o['metric']['mean_rel_diff_ids']:.3f}, pass={o['pass']}")
        return 0 if o["pass"] else 1
    o = run_dump(a)
    S = o["metric"]
    print(f"merl_dwell: F1 tracked {S['frame_browse']['tracked']['F1']:.3f} vs prior {S['frame_browse']['prior_all_browse']['F1']:.3f}; "
          f"IDs/clip {S['ids_per_clip']['mean']:.2f}; longest-ID MAE {S['per_clip_browse_s']['longest_id']['MAE']:.1f}s vs prior {S['per_clip_browse_s']['prior']['MAE']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
