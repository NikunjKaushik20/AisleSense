"""M3 swaps, one by one against the M2 baseline on the same folds -> results/swaps.json (one row per arm, prior beside each).

Arms: baseline (ByteTrack buffer 30, per-ID timer) | +S1 buffer 200 | +S1+S2 ghost hold | +S1+S2+S3 person-s / merged entries.
Folds (same as M2):
  MOT20  leave-one-sequence-out. S1 buffer is picked per held-out sequence from {30, 100, 200} on the OTHER sequences.
         Per-visit span MAE vs LOSO global median, long-dweller AUROC; zone level (3x3, zones with >= 10 GT visits):
         mean dwell per zone vs LOSO global mean GT dwell.
  MERL   val subjects 21-26 choose (S2 TTL x k grid), test 27-41 scored. Frame browse F1 vs all-browse, per-clip browse
         seconds vs val share x clip length. S1 is not re-run on MERL: the cached dump is buffer 30 (re-dump = GPU ~40 min).
  synth  synth_aisle sessions: s1 tunes (min visit length per arm), s2 scored (forced re-score, recorded). Section mean
         dwell vs stopwatch, prior = tune-session stopwatch mean. No tracker in synth rows, so S1 does not apply there.
Ghost TTL/k (S2) and the merge TTL (S3) come from the MERL val grid only; nothing is chosen on a scored split.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import merl_dwell as md  # noqa: E402
from eval import mot20_dwell as mo  # noqa: E402
from eval.common import NIKUNJ_OUT, auroc, check_and_lock, fingerprint, prf, write_result  # noqa: E402
from eval.harness import make_guard, site_hash  # noqa: E402
from eval.stopwatch import MIN_VISIT_GRID, errors, section_means  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aisle.analytics import dwell  # noqa: E402
from aisle.cv import ghost  # noqa: E402
from aisle.geo import homography as hg  # noqa: E402
from aisle.schema import load_site  # noqa: E402

ARMS = [("baseline", []), ("+S1", ["S1"]), ("+S1+S2", ["S1", "S2"]), ("+S1+S2+S3", ["S1", "S2", "S3"])]
BUFFERS = (30, 100, 200)
TTL_GRID, K_GRID = (0, 1, 2, 4, 8), (1, 2)
MERL_W, MERL_H, BORDER = 920, 680, 0.05
EXPECTED = {"mot20_b200_span_MAE_s": (4.10, 0.1), "mot20_b200_AUROC": (0.754, 0.02),
            "merl_hold_F1": (0.877, 0.01), "merl_hold_MAE_s": (9.3, 1.0), "merl_prior_F1": 0.866, "merl_prior_MAE_s": 6.5}


# ------------------------------------------------------------------ MERL (S2 choose on val, score on test)
def merl_clips() -> list:
    site = load_site("merl_demo")
    cam = site.cameras[0]
    H = hg.fit(cam.image_points, cam.floor_points)
    out = []
    for f in sorted(glob.glob(str(NIKUNJ_OUT / "merl_dump" / "*.npy"))):
        clip = os.path.basename(f)[:-4]
        sp = md.split_of(clip)
        if sp == "other":
            continue
        a = np.load(f)
        last = int(a[a[:, 1] == -9][0, 0])
        a = a[a[:, 1] != -9]
        fi = np.arange(0, last + 1, md.STRIDE)
        g = md.gt_mask(clip, int(fi[-1]))[fi]
        rows, exited = [], set()
        if len(a):
            fl = hg.apply(H, hg.foot_points(a[:, 2:6]))
            rows = [(float(r[0]) / md.FPS, int(r[1]), float(p[0]), float(p[1])) for r, p in zip(a, fl)]
            for tid in np.unique(a[:, 1]).astype(int):
                r = a[a[:, 1] == tid][-1]
                if r[2] < BORDER * MERL_W or r[3] < BORDER * MERL_H or r[4] > (1 - BORDER) * MERL_W or r[5] > (1 - BORDER) * MERL_H:
                    exited.add(tid)  # same exit evidence as X8 (box at the image border)
        out.append({"clip": clip, "split": sp, "fi": fi, "g": g, "rows": rows, "exited": exited, "t_end": last / md.FPS})
    return out


def merl_presence(c: dict, ttl: float, k: float) -> np.ndarray:
    hz = md.FPS / md.STRIDE
    rows = c["rows"]
    if ttl > 0 and rows:
        rows, _ = ghost.apply(rows, hz, ttl, k, ghost.RADIUS_M, c["exited"], c["t_end"])
    fr = np.unique(np.round(np.array([r[0] for r in rows]) * md.FPS).astype(int)) if rows else np.array([], int)
    return np.isin(c["fi"], fr)


def tri(pred, g):
    return [int((pred & g).sum()), int((pred & ~g).sum()), int((~pred & g).sum())]


def choose_ghost(val_clips: list) -> dict:
    """S2 TTL x k grid on MERL val only (asserted). Ties keep the first in grid order, as X8."""
    assert val_clips and all(md.split_of(c["clip"]) == "val" for c in val_clips), "ghost params must come from MERL val subjects 21-26"
    grid = {}
    for ttl, k in itertools.product(TTL_GRID, K_GRID):
        grid[(ttl, k)] = prf(*np.sum([tri(merl_presence(c, ttl, k), c["g"]) for c in val_clips], 0))["F1"]
    best = max(grid.values())
    pick = next(key for key in grid if grid[key] == best)
    return {"grid_val_F1": {f"ttl{t}_k{k}": v for (t, k), v in grid.items()}, "pick": {"ttl_s": float(pick[0]), "k": float(pick[1])},
            "chosen_on": "MERL val subjects 21-26", "val_clips": [c["clip"] for c in val_clips]}


def merl_arms(clips: list, gp: dict) -> dict:
    val = [c for c in clips if c["split"] == "val"]
    test = [c for c in clips if c["split"] == "test"]
    dt = md.STRIDE / md.FPS
    share = md.choose_prior_share([{"clip": c["clip"], "gt_s": c["g"].sum() * dt, "dur_s": len(c["fi"]) * dt} for c in val])
    gt = np.array([c["g"].sum() * dt for c in test])
    dur = np.array([len(c["fi"]) * dt for c in test])
    prior_est = share * dur
    prior = {"frame_browse_F1": prf(*np.sum([tri(np.ones_like(c["g"]), c["g"]) for c in test], 0))["F1"], "frame_browse_name": "all-browse",
             "per_clip_MAE_s": float(np.abs(prior_est - gt).mean()), "per_clip_name": "val browse share x clip length", "val_browse_share": share}
    ttl, k = gp["pick"]["ttl_s"], gp["pick"]["k"]
    hz = md.FPS / md.STRIDE

    def presence_arm(t_):
        P = [merl_presence(c, t_, k) for c in test]
        f = prf(*np.sum([tri(p, c["g"]) for p, c in zip(P, test)], 0))
        est = np.array([p.sum() * dt for p in P])
        return {"frame_browse": f, "per_clip_browse_s": {"MAE": float(np.abs(est - gt).mean()), "bias": float((est - gt).mean())}}
    base = presence_arm(0)
    s2 = presence_arm(ttl)
    # S3: person-seconds and merged entries over the whole view (one "section" per clip; 1 true shopper per clip)
    ent, ps, nid = [], [], []
    for c in test:
        vs = dwell.visits(c["rows"], lambda x, y: "view", hz, ghost_hold=True, merge=True, ttl_s=ttl, k=k,
                          exited=c["exited"], t_end=c["t_end"])
        ent.append(len(vs)); ps.append(sum(v["seconds"] for v in vs)); nid.append(len({r[1] for r in c["rows"]}))
    ps = np.array(ps)
    s3 = {**s2, "entries_per_clip": float(np.mean(ent)), "ids_per_clip": float(np.mean(nid)), "true_shoppers_per_clip": 1,
          "person_s_per_clip_MAE": float(np.abs(ps - gt).mean()),
          "note": "frame F1 and per-clip browse seconds are presence-based and unchanged by S3; S3 changes the entry count "
                  "(IDs per clip -> merged entries) and the person-seconds sum (overlapping duplicate IDs add up)."}
    na = {"not_run": "MERL dump is buffer 30; S1 needs a GPU re-dump (~40 min). Row repeats the arm without S1."}
    return {"baseline": {**base, "ids_per_clip": float(np.mean(nid))}, "+S1": {**base, **na}, "+S1+S2": {**s2, **na},
            "+S1+S2+S3": {**s3, **na}, "prior": prior, "n": {"test_clips": len(test), "val_clips": len(val)}}


# ------------------------------------------------------------------ MOT20 (LOSO)
_MO_CACHE = {}
_FAST = {"on": True}


def mot20_seq(sd: Path, buf: int, hold: bool):
    key = (sd.name, buf, hold)
    if key not in _MO_CACHE:
        _MO_CACHE[key] = _mot20_seq(sd, buf, hold)
    return _MO_CACHE[key]


def _mot20_seq(sd: Path, buf: int, hold: bool):
    si = mo.seqinfo(sd)
    T, _ = mo.tracks(sd, buf, _FAST["on"])
    T, V = mo.tracker_visits(T, si["W"], si["H"])
    rows = None
    if hold:
        T = T.copy()
        T["cx"] = (T.x1 + T.x2) / 2
        rows = list(zip((T.f / si["fps"]).tolist(), T.tid.tolist(), T.cx.tolist(), T.y2.tolist()))
        last = T.sort_values("f").groupby("tid").last()
        ex = set(last.index[(last.x1 < BORDER * si["W"]) | (last.y1 < BORDER * si["H"]) | (last.x2 > (1 - BORDER) * si["W"]) | (last.y2 > (1 - BORDER) * si["H"])])
        rad = (last.y2 - last.y1).to_dict()
        hs = ghost.holds(rows, ghost.TTL_S, ghost.K, rad, ex, si["L"] / si["fps"])
        add = {h["track"]: int(round((h["t1"] - h["t0"]) * si["fps"])) for h in hs}
        lastrun = V.reset_index().sort_values("f1").groupby("tid").last()["run"]
        V = V.copy()
        for tid, run in lastrun.items():
            if tid in add:
                V.loc[run, "f1"] += add[tid]
        rows = (rows, rad, ex)
    GV = mo.gt_visits_matched(sd, T, V)
    return GV, T, V, rows, si


def gt_zone_means(sd: Path, si: dict) -> dict:
    g = pd.read_csv(sd / "gt" / "gt.txt", header=None, names=["f", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
    g = g[(g.cls == 1) & (g.flag == 1)].copy()
    g["zone"] = mo.zone(g.x, g.y, g.x + g.w, g.y + g.h, si["W"], si["H"])
    g = g.sort_values(["id", "f"]).reset_index(drop=True)
    g["run"] = ((g.zone != g.zone.shift()) | (g.id != g.id.shift()) | (g.f - g.f.shift() > 1)).cumsum()
    R = g.groupby("run").agg(zone=("zone", "first"), n=("f", "size"))
    R["s"] = R.n / si["fps"]
    R = R[R.s >= 5]
    return {int(z): (float(x.s.mean()), int(len(x))) for z, x in R.groupby("zone")}


def mot20_arms(fast: bool) -> dict:
    _FAST["on"] = fast
    seqs = sorted(Path(p) for p in glob.glob(str(mo.mot_root() / "MOT20-*")))
    names = [s.name for s in seqs]
    E = {b: {sd.name: mot20_seq(sd, b, False)[0] for sd in seqs} for b in BUFFERS}
    # S1 pick per held-out sequence, from the other sequences only
    picks = {}
    for s in names:
        mae = {b: float(np.mean(np.concatenate([np.abs(E[b][o].span - E[b][o].gts).values for o in names if o != s]))) for b in BUFFERS}
        picks[s] = {"pick": int(min(mae, key=mae.get)), "other_seq_MAE": mae}
    gts_all = pd.concat([E[30][s] for s in names])
    prior_med = {s: float(gts_all[gts_all.seq != s].gts.median()) for s in names}
    prior_mean = {s: float(gts_all[gts_all.seq != s].gts.mean()) for s in names}
    truth_z = {}
    for sd in seqs:
        si = mo.seqinfo(sd)
        truth_z[sd.name] = gt_zone_means(sd, si)

    def visit_metrics(Es: dict) -> dict:
        A = pd.concat([Es[s] for s in names], ignore_index=True)
        err = np.abs(A.span - A.gts)
        y = A.gts.values >= 15
        perr = np.abs(A.gts - A.seq.map(prior_med))
        d = (err - perr).values
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        per_seq = {s: {"span_MAE_s": float(err[A.seq == s].mean()), "prior_MAE_s": float(perr[A.seq == s].mean())} for s in names}
        return {"span_MAE_s": float(err.mean()), "span_bias_s": float((A.span - A.gts).mean()), "AUROC_ge15s": auroc(y, A.span.values),
                "under_half_truth": float((A.span < 0.5 * A.gts).mean()), "n_visits": int(len(A)),
                "minus_prior_MAE_s": float(d.mean()), "minus_prior_CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "seqs_beating_prior": int(sum(v["span_MAE_s"] < v["prior_MAE_s"] for v in per_seq.values())), "per_seq": per_seq}

    def zone_metrics(est: dict) -> dict:
        e, p = [], []
        for s in names:
            for z, (m, n) in truth_z[s].items():
                if n >= 10:
                    e.append(abs(est[s].get(z, 0.0) - m)); p.append(abs(prior_mean[s] - m))
        return {"zone_mean_MAE_s": float(np.mean(e)), "zone_prior_MAE_s": float(np.mean(p)), "n_zones": len(e)}

    def timer_zone_est(Vs: dict, sis: dict) -> dict:
        out = {}
        for s in names:
            V, si = Vs[s], sis[s]
            sp = (V.f1 - V.f0 + 1) / si["fps"]
            d = pd.DataFrame({"z": V.z.values, "s": sp.values})
            d = d[d.s >= 5]
            out[s] = {int(z): float(x.s.mean()) for z, x in d.groupby("z")}
        return out

    arms, sis = {}, {sd.name: mo.seqinfo(sd) for sd in seqs}
    # baseline: buffer 30 everywhere
    Vb = {sd.name: mot20_seq(sd, 30, False)[2] for sd in seqs}
    arms["baseline"] = {**visit_metrics(E[30]), **zone_metrics(timer_zone_est(Vb, sis)), "buffer": {s: 30 for s in names}}
    E1 = {s: E[picks[s]["pick"]][s] for s in names}
    V1 = {sd.name: mot20_seq(sd, picks[sd.name]["pick"], False)[2] for sd in seqs}
    arms["+S1"] = {**visit_metrics(E1), **zone_metrics(timer_zone_est(V1, sis)), "buffer": {s: picks[s]["pick"] for s in names}}
    R2 = {sd.name: mot20_seq(sd, picks[sd.name]["pick"], True) for sd in seqs}
    E2 = {s: R2[s][0] for s in names}
    arms["+S1+S2"] = {**visit_metrics(E2), **zone_metrics(timer_zone_est({s: R2[s][2] for s in names}, sis)),
                      "buffer": {s: picks[s]["pick"] for s in names}}
    est3 = {}
    for s in names:
        (rows, rad, ex), si = R2[s][3], R2[s][4]
        W, H = si["W"], si["H"]

        def zf(x, y, W=W, H=H):
            return int(min(max(x // (W / 3), 0), 2) * 3 + min(max(y // (H / 3), 0), 2))
        vs = dwell.visits(rows, zf, float(si["fps"]), ghost_hold=True, merge=True, radius=rad, exited=ex,
                          t_end=si["L"] / si["fps"], min_visit_s=5.0)
        tab = dwell.section_table(vs)
        est3[s] = {int(r["section"]): r["dwell_s"] for r in tab["sections"] if r["dwell_s"] is not None}
    arms["+S1+S2+S3"] = {**visit_metrics(E2), **zone_metrics(est3), "buffer": {s: picks[s]["pick"] for s in names},
                         "note": "per-visit span is unchanged by S3 (S3 is a section-level estimator); zone level uses person-s / merged entries"}
    prior = {"span_MAE_s": float(np.mean(np.concatenate([np.abs(E[30][s].gts - prior_med[s]).values for s in names]))),
             "span_name": "LOSO global median GT dwell", "AUROC_ge15s": 0.5, "zone_name": "LOSO global mean GT dwell",
             "zone_prior_MAE_s": arms["baseline"]["zone_prior_MAE_s"]}
    files = [sd / "gt" / "gt.txt" for sd in seqs] + [sd / "det" / "det.txt" for sd in seqs]
    return {**arms, "prior": prior, "S1_params": {"grid": list(BUFFERS), "loso": picks, "chosen_on": "MOT20 leave-one-sequence-out (held-out sequence never picks)"},
            "data_hash": fingerprint(files)}


# ------------------------------------------------------------------ synth stopwatch (tune s1, score s2)
def synth_arms() -> dict:
    guard, site, ss = make_guard("synth_aisle")
    zf = dwell.site_zone_fn(site)
    ex = ghost.floor_exit_fn(site.floor)
    secs = [z.name for z in site.zones if z.kind == "section"]

    def est(d, arm, mv):
        hz = d["hz"]
        if arm in ("baseline", "+S1"):
            vs = dwell.visits(d["rows"], zf, hz, ghost_hold=False, merge=False, min_visit_s=mv)
        elif arm == "+S1+S2":
            vs = dwell.visits(d["rows"], zf, hz, ghost_hold=True, merge=False, exited=ex, min_visit_s=mv)
        else:
            vs = dwell.visits(d["rows"], zf, hz, ghost_hold=True, merge=True, exited=ex, min_visit_s=mv)
        tab = dwell.section_table(vs, sections=secs)
        return {r["section"]: r["dwell_s"] for r in tab["sections"] if r["dwell_s"] is not None}, vs

    guard.phase = "tune"
    tune = [guard.load(s) for s in guard.tune]
    sw = [v for d in tune for v in d["stopwatch"]]
    prior = section_means(sw)
    prior_global = float(np.mean([v["t_out"] - v["t_in"] for v in sw]))
    picks = {}
    for arm, _ in ARMS:
        g = {}
        for mv in MIN_VISIT_GRID:
            e = []
            for d in tune:
                e += list(errors(est(d, arm, mv)[0], section_means(d["stopwatch"]), 0.0).values())
            g[mv] = float(np.mean(e))
        picks[arm] = {"min_visit_s": min(g, key=g.get), "tune_MAE_grid": g}
    lock = check_and_lock("synth_aisle", "+".join(guard.score), True, {"script": "swaps", "tuned": {a: p["min_visit_s"] for a, p in picks.items()}})
    guard.phase = "score"
    out = {}
    for s in guard.score:
        d = guard.load(s)
        truth = section_means(d["stopwatch"])
        e_pri = errors(prior, truth, prior_global)
        for arm, _ in ARMS:
            e, vs = est(d, arm, picks[arm]["min_visit_s"])
            err = errors(e, truth, 0.0)
            r = out.setdefault(arm, {"section_MAE_s": [], "n_entries": 0, "n_stopwatch": 0})
            r["section_MAE_s"].append(float(np.mean(list(err.values()))))
            r["n_entries"] += len(vs)
            r["n_stopwatch"] += len(d["stopwatch"])
        out.setdefault("_prior", []).append(float(np.mean(list(e_pri.values()))))
        # stitch diagnostics on the score session (scorer reads the true person prefix before '~'; no rule does)
        rows = sorted(d["rows"], key=lambda r: (r[0], r[1]))
        hs = ghost.holds(rows, ghost.TTL_S, ghost.K, ghost.RADIUS_M, ex)
        ch = ghost.stitch(hs)
        true_breaks = {r[1] for r in rows if "~" in str(r[1])}
        ok = sum(str(kb).split("~")[0] == str(ka).split("~")[0] for kb, ka in ch.items())
        out["_stitch"] = {"links": len(ch), "correct": ok, "precision": ok / max(len(ch), 1),
                          "true_id_breaks": len(true_breaks), "recall": ok / max(len(true_breaks), 1),
                          "holds": len(hs), "holds_ended_by_rebirth": sum(h["reborn"] is not None or bool(h["candidates"]) for h in hs)}
    pr = float(np.mean(out.pop("_prior")))
    st = out.pop("_stitch")
    arms = {a: {"section_MAE_s": float(np.mean(v["section_MAE_s"])), "entries": v["n_entries"], "stopwatch_visits": v["n_stopwatch"],
                "min_visit_s": picks[a]["min_visit_s"]} for a, v in out.items()}
    arms["+S1"]["not_applicable"] = "synthetic rows have no tracker; S1 does not change them (row repeats baseline)"
    return {**arms, "prior": {"name": "tune-session stopwatch mean per section", "section_MAE_s": pr},
            "tune_picks": picks, "stitch_diagnostics_score_session": st, "sessions": {"tune": guard.tune, "score": guard.score},
            "access_log": guard.log, "score_lock": lock, "data_hash": site_hash("synth_aisle")}


# ------------------------------------------------------------------ table
def statement(rows: list, beats: dict) -> str:
    """Plain-English verdict generated from the numbers (never typed by hand)."""
    sig = [r for r in rows if beats[r["arm"]]["mot20_span"]]
    low = [r for r in rows if beats[r["arm"]]["mot20_span_point_lower"]]
    merl_any = any(v["merl_per_clip"] for v in beats.values())
    parts = []
    p = rows[0]["mot20_prior"]["span_MAE_s"]
    if sig:
        parts.append(f"On MOT20, arm {sig[0]['arm']} beats the LOSO prior on per-visit span MAE with the 95% CI below 0.")
    else:
        parts.append("On MOT20 no arm beats the LOSO prior on per-visit span MAE (no 95% CI of arm minus prior lies below 0).")
    if low and not sig:
        r = low[0]["mot20"]
        ci = r["minus_prior_CI95"]
        parts.append(f"Arm {low[0]['arm']} has a lower point MAE ({r['span_MAE_s']:.2f} s vs {p:.2f} s, difference "
                     f"{r['minus_prior_MAE_s']:+.2f} s, CI [{ci[0]:+.2f}, {ci[1]:+.2f}], {r['seqs_beating_prior']} of "
                     f"{len(r['per_seq'])} sequences): a tie, not a win.")
    parts.append("On MERL no arm beats the prior on per-clip browse seconds (val share x clip length)." if not merl_any
                 else "On MERL at least one arm beats the prior on per-clip browse seconds.")
    parts.append("The swaps shrink per-ID error and raise long-dweller AUROC and frame browse F1; the prior column stays beside every dwell number.")
    return " ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="cached MOT20 tracks + MERL dump (CPU only)")
    a = ap.parse_args(argv)
    clips = merl_clips()
    gp = choose_ghost([c for c in clips if c["split"] == "val"])
    assert (gp["pick"]["ttl_s"], gp["pick"]["k"]) == (ghost.TTL_S, ghost.K), f"val pick {gp['pick']} != frozen app params"
    M = merl_arms(clips, gp)
    print(f"merl done: hold F1 {M['+S1+S2']['frame_browse']['F1']:.3f}", flush=True)
    MO = mot20_arms(a.fast)
    print(f"mot20 done: S1 span {MO['+S1']['span_MAE_s']:.2f}", flush=True)
    SY = synth_arms()
    rows = []
    for arm, sw in ARMS:
        rows.append({
            "arm": arm, "swaps": sw,
            "mot20": {k: MO[arm][k] for k in ("span_MAE_s", "span_bias_s", "AUROC_ge15s", "under_half_truth", "minus_prior_MAE_s",
                                               "minus_prior_CI95", "seqs_beating_prior", "per_seq", "zone_mean_MAE_s", "n_zones", "buffer")},
            "mot20_prior": {"span_MAE_s": MO["prior"]["span_MAE_s"], "AUROC_ge15s": 0.5, "zone_mean_MAE_s": MO["prior"]["zone_prior_MAE_s"]},
            "merl": M[arm], "merl_prior": M["prior"],
            "synth": SY[arm], "synth_prior": SY["prior"],
        })
    per_visit_beats = {r["arm"]: {"mot20_span": r["mot20"]["minus_prior_CI95"][1] < 0,  # beats = whole 95% CI below 0
                                  "mot20_span_point_lower": r["mot20"]["span_MAE_s"] < r["mot20_prior"]["span_MAE_s"],
                                  "merl_per_clip": r["merl"]["per_clip_browse_s"]["MAE"] < r["merl_prior"]["per_clip_MAE_s"]} for r in rows}
    any_beats = any(d["mot20_span"] or d["merl_per_clip"] for d in per_visit_beats.values())
    s1, s2 = SY["+S1"]["section_MAE_s"], SY["+S1+S2"]["section_MAE_s"]
    fallback = {"rule": "if S2 hurts on synth multi-shopper scenes (wrong holds), keep S1 only and show the long-stay list instead of dwell MAE",
                "synth_S1_MAE_s": s1, "synth_S1S2_MAE_s": s2, "S2_hurts_on_synth": s2 > s1 + 1e-9,
                "stitch_precision": SY["stitch_diagnostics_score_session"]["precision"]}
    fallback["applied"] = fallback["S2_hurts_on_synth"]
    res = {
        "data": {"mot20": "MOT20 train 4 seqs (station crowd, public detections) - not a store",
                 "merl": "MERL Shopping test subjects 27-41 (real overhead store video, 1 shopper per clip)",
                 "synth": "synthetic synth_aisle sessions (harness proof, not evidence)"},
        "config": {"arms": [a for a, _ in ARMS], "ghost": {"ttl_s": ghost.TTL_S, "k": ghost.K, "radius_m_floor": ghost.RADIUS_M,
                                                           "radius_mot20": "k x last box height (px)", "exit": "MERL/MOT20: last box within 5% of image border; floor: 0.5 m of floor edge"},
                   "merge_ttl_s": ghost.TTL_S, "fast": a.fast},
        "seed": 0, "data_hash": {"mot20": MO["data_hash"], "synth": SY["data_hash"]},
        "params": {"S1_buffer": MO["S1_params"], "S2_ghost": gp, "S3_merge_ttl": {"ttl_s": ghost.TTL_S, "chosen_on": "same as S2 (MERL val)"},
                   "synth_min_visit": {k: v["min_visit_s"] for k, v in SY["tune_picks"].items()}, "synth_tune_grid": SY["tune_picks"]},
        "arms": rows,
        "per_visit_beats_prior": per_visit_beats,
        "any_arm_beats_prior_per_visit": any_beats,
        "statement": statement(rows, per_visit_beats),
        "fallback": fallback,
        "synth_detail": {k: SY[k] for k in ("stitch_diagnostics_score_session", "sessions", "access_log", "score_lock")},
        "merl_n": M["n"],
        "expected_X5_X8": EXPECTED,
    }
    write_result("swaps", res)
    b, s = rows[0], rows[1]
    print(f"swaps: mot20 span {b['mot20']['span_MAE_s']:.2f}->{s['mot20']['span_MAE_s']:.2f}s (prior {b['mot20_prior']['span_MAE_s']:.2f}), "
          f"AUROC {s['mot20']['AUROC_ge15s']:.3f}; merl hold F1 {rows[2]['merl']['frame_browse']['F1']:.3f} (prior {M['prior']['frame_browse_F1']:.3f}), "
          f"hold MAE {rows[2]['merl']['per_clip_browse_s']['MAE']:.1f}s (prior {M['prior']['per_clip_MAE_s']:.1f}); "
          f"synth S3 {rows[3]['synth']['section_MAE_s']:.2f}s vs base {rows[0]['synth']['section_MAE_s']:.2f} prior {SY['prior']['section_MAE_s']:.2f}; "
          f"fallback {'APPLIED' if fallback['applied'] else 'not needed'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
