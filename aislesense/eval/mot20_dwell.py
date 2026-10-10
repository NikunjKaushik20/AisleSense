"""MOT20 train dwell eval (same zones, matching and metric as X5).

GT dwell visits: runs of a GT pedestrian in one 3x3 image zone, >= 5 s. Tracker: ByteTrack (aisle.cv.track.BASE config,
buffer from --buffer, default 30 = baseline) on the MOT20 public detections. Each GT visit is matched (IoU >= 0.5,
Hungarian per frame) to the tracker visit covering most of its frames; the estimate is that tracker visit's span.
Prior: leave-one-sequence-out global median of GT dwell (the scored sequence never sets its own prior).
Long-dweller ranking: AUROC of the span for GT >= 15 s (a constant prior has 0.5).

  --fast       read cached tracks nikunj/scripts/out/mot20_trk_<seq>_b<buffer>.csv (produced with the same config)
  (default)    re-run ByteTrack on det.txt, cache under var/cache/
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.common import NIKUNJ_OUT, auroc, fingerprint, write_result  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aisle.cv.track import BASE  # noqa: E402
from aisle.schema import VAR_DIR, data_root  # noqa: E402

EXPECTED = {30: {"span_MAE_s": 4.70, "prior_MAE_s": 3.96, "AUROC_ge15s": 0.712},
            200: {"span_MAE_s": 4.10, "prior_MAE_s": 3.96, "AUROC_ge15s": 0.754}}


def mot_root() -> Path:
    return data_root() / "mot20_labels_x" / "MOT20Labels" / "train"


def zone(x1, y1, x2, y2, W, H):
    return (np.clip(((x1 + x2) / 2) // (W / 3), 0, 2) * 3 + np.clip(y2 // (H / 3), 0, 2)).astype(int)


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    A = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); B = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (A[:, None] + B[None, :] - inter + 1e-9)


class _Dets:
    def __init__(s, xyxy, conf):
        s.xyxy = xyxy; s.conf = conf; s.cls = np.zeros(len(conf))

    @property
    def xywh(s):
        x = s.xyxy
        return np.c_[(x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2, x[:, 2] - x[:, 0], x[:, 3] - x[:, 1]]

    def __len__(s):
        return len(s.conf)

    def __getitem__(s, i):
        return _Dets(s.xyxy[i], s.conf[i])


def seqinfo(sd: Path) -> dict:
    ini = (sd / "seqinfo.ini").read_text()
    g = lambda k: int(ini.split(k + "=")[1].split()[0])  # noqa: E731
    return {"fps": g("frameRate"), "W": g("imWidth"), "H": g("imHeight"), "L": g("seqLength")}


def tracks(sd: Path, buf: int, fast: bool) -> tuple[pd.DataFrame, str]:
    name = f"mot20_trk_{sd.name}_b{buf}.csv"
    if fast and (NIKUNJ_OUT / name).exists():
        return pd.read_csv(NIKUNJ_OUT / name), "nikunj cache"
    fo = VAR_DIR / "cache" / name
    if fo.exists():
        return pd.read_csv(fo), "var cache"
    from ultralytics.trackers.byte_tracker import BYTETracker
    si = seqinfo(sd)
    d = pd.read_csv(sd / "det" / "det.txt", header=None).iloc[:, :7]
    d.columns = ["f", "i", "x", "y", "w", "h", "c"]
    d["c"] = np.where(d.c > 0, 0.9, 0.5)  # binary public-det confidence remapped (A25)
    cfg = {**BASE, "track_buffer": int(buf)}
    tr = BYTETracker(SimpleNamespace(**{k: v for k, v in cfg.items() if k != "tracker_type"}), frame_rate=si["fps"])
    dg = dict(tuple(d.groupby("f")))
    out = []
    for f in range(1, si["L"] + 1):
        df = dg.get(f)
        D = _Dets(np.c_[df.x, df.y, df.x + df.w, df.y + df.h].astype(float), df.c.values.astype(float)) if df is not None else _Dets(np.zeros((0, 4)), np.zeros(0))
        for r in tr.update(D):
            out.append((f, int(r[4]), *r[:4]))
    T = pd.DataFrame(out, columns=["f", "tid", "x1", "y1", "x2", "y2"])
    fo.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(fo, index=False)
    return T, "computed"


def tracker_visits(T: pd.DataFrame, W, H):
    T = T.sort_values(["tid", "f"]).reset_index(drop=True)
    T["z"] = zone(T.x1, T.y1, T.x2, T.y2, W, H)
    T["run"] = ((T.z != T.z.shift()) | (T.tid != T.tid.shift())).cumsum()
    V = T.groupby("run").agg(tid=("tid", "first"), z=("z", "first"), f0=("f", "min"), f1=("f", "max"))
    return T, V


def gt_visits_matched(sd: Path, T: pd.DataFrame, V: pd.DataFrame) -> pd.DataFrame:
    from scipy.optimize import linear_sum_assignment
    si = seqinfo(sd)
    W, H, fps = si["W"], si["H"], si["fps"]
    g = pd.read_csv(sd / "gt" / "gt.txt", header=None, names=["f", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
    g = g[(g.cls == 1) & (g.flag == 1)].copy()
    g["x2"] = g.x + g.w; g["y2"] = g.y + g.h
    g["zone"] = zone(g.x, g.y, g.x2, g.y2, W, H)
    g = g.sort_values(["id", "f"]).reset_index(drop=True)
    g["run"] = ((g.zone != g.zone.shift()) | (g.id != g.id.shift()) | (g.f - g.f.shift() > 1)).cumsum()
    gi = g.groupby("f").indices
    T = T.copy()
    T["vi"] = T.run.map(dict(zip(V.index, range(len(V))))).values
    tg = dict(tuple(T.groupby("f")))
    vcol = np.full(len(g), -1)
    for f, ix in gi.items():
        t = tg.get(f)
        if t is None:
            continue
        M = iou(g.loc[ix, ["x", "y", "x2", "y2"]].values, t[["x1", "y1", "x2", "y2"]].values)
        r, c = linear_sum_assignment(-M)
        ok = M[r, c] >= 0.5
        vcol[ix[r[ok]]] = t.vi.values[c[ok]]
    g["vi"] = vcol
    GV = g.groupby("run").agg(n=("f", "size"), vis=("vis", "mean"))
    GV["gts"] = GV.n / fps
    best = g[g.vi >= 0].groupby(["run", "vi"]).size().rename("c").reset_index().sort_values("c").drop_duplicates("run", keep="last").set_index("run").vi
    GV["vi"] = best.reindex(GV.index).fillna(-1).astype(int)
    GV = GV[GV.gts >= 5].copy()
    has = (GV.vi >= 0).values
    span = (V.f1 - V.f0 + 1).values
    GV["span"] = np.where(has, span[GV.vi.clip(lower=0).values], 0) / fps
    GV["matched"] = has
    GV["seq"] = sd.name
    return GV.reset_index()


def evaluate(buf: int = 30, fast: bool = True) -> dict:
    seqs = sorted(Path(p) for p in glob.glob(str(mot_root() / "MOT20-*")))
    E, src = [], {}
    for sd in seqs:
        si = seqinfo(sd)
        T, src[sd.name] = tracks(sd, buf, fast)
        T, V = tracker_visits(T, si["W"], si["H"])
        E.append(gt_visits_matched(sd, T, V))
    E = pd.concat(E, ignore_index=True)
    S = sorted(E.seq.unique())
    err = np.abs(E.span - E.gts)
    prior_err = pd.concat([np.abs(E[E.seq == s].gts - E[E.seq != s].gts.median()) for s in S])
    y = E.gts.values >= 15
    per_seq = {s: {"n": int((E.seq == s).sum()), "span_MAE": float(err[E.seq == s].mean()),
                   "prior_MAE": float(np.abs(E[E.seq == s].gts - E[E.seq != s].gts.median()).mean()),
                   "prior_median_from_other_seqs": float(E[E.seq != s].gts.median())} for s in S}
    files = [sd / "gt" / "gt.txt" for sd in seqs] + [sd / "det" / "det.txt" for sd in seqs]
    return {
        "E": E,
        "result": {
            "data": "MOT20 train (4 seqs, station crowd, public detections) - not a store",
            "config": {"tracker": {**BASE, "track_buffer": int(buf)}, "zones": "3x3 image grid (A24)", "match": "IoU>=0.5 Hungarian",
                       "gt_visit_min_s": 5, "track_source": src, "prior": "leave-one-sequence-out global median GT dwell"},
            "seed": 0, "data_hash": fingerprint(files),
            "n": {"gt_visits": int(len(E)), "long_dwellers_ge15s": int(y.sum()), "matched_share": float(E.matched.mean())},
            "metric": {"span_MAE_s": float(err.mean()), "span_bias_s": float((E.span - E.gts).mean()),
                       "under_half_truth": float((E.span < 0.5 * E.gts).mean()),
                       "AUROC_ge15s": auroc(y, E.span.values),
                       "recall_at_est_ge10s": float((E.span.values[y] >= 10).mean()),
                       "fpr_at_est_ge10s": float((E.span.values[~y] >= 10).mean())},
            "prior": {"span_MAE_s": float(prior_err.mean()), "AUROC_ge15s": 0.5},
            "per_seq": per_seq,
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--buffer", type=int, default=30)
    a = ap.parse_args(argv)
    r = evaluate(a.buffer, a.fast)["result"]
    m, p = r["metric"], r["prior"]
    r["headline"] = [
        {"metric": "per-visit span MAE", "value": m["span_MAE_s"], "prior": p["span_MAE_s"], "prior_name": "LOSO global median",
         "unit": "s per GT visit >= 5 s", "beats_prior": m["span_MAE_s"] < p["span_MAE_s"]},
        {"metric": "long-dweller AUROC (GT >= 15 s)", "value": m["AUROC_ge15s"], "prior": 0.5, "prior_name": "constant", "beats_prior": m["AUROC_ge15s"] > 0.5},
    ]
    r["expected_X5"] = EXPECTED.get(a.buffer)
    name = "mot20_dwell" if a.buffer == 30 else f"mot20_dwell_b{a.buffer}"
    write_result(name, r)
    print(f"{name}: span MAE {m['span_MAE_s']:.2f}s vs prior {p['span_MAE_s']:.2f}s; AUROC {m['AUROC_ge15s']:.3f}; n {r['n']['gt_visits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
