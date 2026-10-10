"""Shared eval helpers: result writing, data fingerprints, tune/score session guard, score locks, density rule.

Every eval script writes exactly one JSON in results/ with: name, config, seed, data (fingerprint), metric, prior
(or baseline), n. No timestamps in result files, so reruns with the same seeds are byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from aisle.schema import REPO_ROOT, RESULTS_DIR, VAR_DIR  # noqa: E402

# Cached research dumps (MERL detections, MOT20 tracks). Not in git; evals that need them skip without them.
NIKUNJ_OUT = Path(os.environ.get("NIKUNJ_OUT", REPO_ROOT / "nikunj" / "scripts" / "out"))


# ---------------------------------------------------------------- results
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if not np.isfinite(f) else round(f, 6)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, Path):
        return str(o).replace("\\", "/")
    return o


def write_result(name: str, payload: dict, results_dir: Path | None = None) -> Path:
    d = Path(results_dir or RESULTS_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.json"
    p.write_text(json.dumps(_clean({"name": name, **payload}), indent=1) + "\n", encoding="utf-8")
    return p


def read_result(name: str, results_dir: Path | None = None) -> dict:
    return json.loads((Path(results_dir or RESULTS_DIR) / f"{name}.json").read_text(encoding="utf-8"))


def fingerprint(paths) -> str:
    """sha1 over name + size + content (files > 32 MB: first and last 4 MB only). Order-independent of input order."""
    h = hashlib.sha1()
    for p in sorted(Path(x) for x in paths):
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}".encode())
        with open(p, "rb") as f:
            if st.st_size <= 32 << 20:
                h.update(f.read())
            else:
                h.update(f.read(4 << 20))
                f.seek(-(4 << 20), 2)
                h.update(f.read())
    return h.hexdigest()[:16]


def prf(tp, fp, fn) -> dict:
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"P": p, "R": r, "F1": 2 * p * r / max(p + r, 1e-9), "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def auroc(y, s) -> float:
    """Mann-Whitney AUROC with tie handling (no sklearn dependency)."""
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- tune / score guard (F11)
class LeakError(RuntimeError):
    pass


class SessionGuard:
    """Hands out session data only for sessions allowed in the current phase.

    phase 'tune' -> only sessions.yaml `tune`; phase 'score' -> only `score`. Every access is logged in .log."""

    def __init__(self, sessions: dict, loader):
        self.tune = list(sessions.get("tune") or [])
        self.score = list(sessions.get("score") or [])
        if set(self.tune) & set(self.score):
            raise LeakError(f"session in both tune and score: {set(self.tune) & set(self.score)}")
        self.loader = loader
        self.phase = "tune"
        self.log = []

    def load(self, session: str):
        allowed = self.tune if self.phase == "tune" else self.score
        if session not in allowed:
            raise LeakError(f"{self.phase} phase may not read session {session!r} (allowed {allowed})")
        self.log.append((self.phase, session))
        return self.loader(session)


def lock_dir() -> Path:
    return Path(os.environ.get("AISLE_LOCK_DIR", str(VAR_DIR / "locks")))


def check_and_lock(site: str, session: str, force: bool, info: dict, rule_hash: str | None = None,
                   synthetic: bool | None = None) -> dict:
    """Refuse a second scored run of (site, session) unless force; returns lock record for the result JSON.

    M6 hardening (F11): `rule_hash` fingerprints the scoring code (rules + harness). The first scored run pins it. On a
    real (non-synthetic) site a --force-rescore with a DIFFERENT rule hash is refused: changing the rule after seeing
    the score session and re-scoring is tuning on the scored set. Re-running identical code stays allowed (it can only
    reproduce the number). Synthetic sites (synth_aisle, labels from the generator) may re-score after a rule change,
    but the record says so (`rule_changed_since_first_score`). Every scored run is appended to var/locks/ledger.jsonl.
    """
    import datetime
    synthetic = (site == "synth_aisle") if synthetic is None else bool(synthetic)
    d = lock_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{site}_{session}.lock"
    prev = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    if prev is not None and not force:
        raise SystemExit(f"LOCKED: {p} exists (scored before). Re-scoring needs --force-rescore, which is recorded in the result.")
    prev = prev or {}
    first = prev.get("rule_hash_first") or (None if prev.get("n_scored_runs") else rule_hash)
    changed = bool(rule_hash and first and rule_hash != first)
    if changed and not synthetic:
        raise SystemExit(f"RULE CHANGED after scoring: {p} was first scored with rule {first}, now {rule_hash}. Re-scoring a "
                         "real score session with a changed rule is tuning on the scored set; freeze the rule on tune sessions "
                         "and film a new score session instead.")
    n = prev.get("n_scored_runs", 0) + 1
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hist = list(prev.get("history") or []) + [{"run": n, "rule_hash": rule_hash, "force": bool(force and n > 1), "utc": utc}]
    rec = {"lock": f"var/locks/{p.name}", "n_scored_runs": n, "force_rescore": bool(force and n > 1), "synthetic": synthetic,
           "rule_hash": rule_hash, "rule_hash_first": first or rule_hash, "rule_changed_since_first_score": changed,
           "runs_before_rule_hash": prev.get("runs_before_rule_hash", 0 if first else n - 1), **info, "history": hist[-50:]}
    p.write_text(json.dumps(_clean(rec), indent=1), encoding="utf-8")
    with open(d / "ledger.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"lock": p.name, "run": n, "rule_hash": rule_hash, "force": rec["force_rescore"],
                            "synthetic": synthetic, "utc": utc}) + "\n")
    return {k: v for k, v in rec.items() if k != "history"}


def code_hash(*modules) -> str:
    """Fingerprint of the python files that define a scoring rule (modules or paths)."""
    return fingerprint([m.__file__ if hasattr(m, "__file__") else m for m in modules])


# ---------------------------------------------------------------- tracks from event rows
def rows_to_arrays(rows):
    """rows: (t, track, x, y) -> dict of numpy arrays sorted by (track, t)."""
    t = np.array([r[0] for r in rows], float)
    k = np.array([r[1] for r in rows], dtype=object)
    x = np.array([r[2] for r in rows], float)
    y = np.array([r[3] for r in rows], float)
    codes, ki = np.unique(k.astype(str), return_inverse=True)
    o = np.lexsort((t, ki))
    return {"t": t[o], "k": ki[o], "x": x[o], "y": y[o], "names": codes}


def speeds(a, lag_s: float = 1.0, hz: float = 5.0) -> np.ndarray:
    """Per-sample speed (m/s) over a ~lag_s displacement within the same track (backward, then forward at track start)."""
    n = len(a["t"])
    L = max(int(round(lag_s * hz)), 1)
    v = np.full(n, np.nan)
    if n > L:
        same = a["k"][L:] == a["k"][:-L]
        dt = a["t"][L:] - a["t"][:-L]
        d = np.hypot(a["x"][L:] - a["x"][:-L], a["y"][L:] - a["y"][:-L]) / np.maximum(dt, 1e-6)
        ok = same & (dt > 0) & (dt <= 2 * lag_s)
        v[L:][ok] = d[ok]
        back = np.full(n, np.nan)
        back[:-L][ok] = d[ok]
        v = np.where(np.isnan(v), back, v)
    return np.nan_to_num(v, nan=0.0)


# ---------------------------------------------------------------- baseline density rule (F5: cannot tell browse from block)
DENSITY_RULE = {"cell_m": 1.0, "window_s": 30.0, "min_person_s": 5.0, "density_q": 0.90, "slow_mps": 0.3,
                "text": "cell-window (1 m x window) flagged as bottleneck iff person-seconds >= top decile of eligible "
                        "cell-windows (>= 5 person-s) AND mean speed <= 0.3 m/s"}


def density_flags(a, floor: dict, hz: float, window_s: float | None = None, rule: dict | None = None):
    """Baseline bottleneck rule over cell-windows. Returns list of (iy, ix, w, t0, t1, cx, cy) flagged and stats."""
    r = {**DENSITY_RULE, **(rule or {})}
    if window_s is not None:
        r["window_s"] = float(window_s)
    c = r["cell_m"]
    nx = int(np.ceil(floor["width"] / c))
    ny = int(np.ceil(floor["height"] / c))
    ix = np.floor((a["x"] - floor["x0"]) / c).astype(int)
    iy = np.floor((a["y"] - floor["y0"]) / c).astype(int)
    m = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    w = np.floor(a["t"] / r["window_s"]).astype(int)
    v = speeds(a, 1.0, hz)
    key = (w[m].astype(np.int64) * ny + iy[m]) * nx + ix[m]
    uk, inv = np.unique(key, return_inverse=True)
    ps = np.bincount(inv, minlength=len(uk)) / hz
    vm = np.bincount(inv, weights=v[m], minlength=len(uk)) / np.maximum(np.bincount(inv, minlength=len(uk)), 1)
    elig = ps >= r["min_person_s"]
    if not elig.any():
        return [], {"eligible": 0, "flagged": 0, "density_thr": None, **r}
    thr = float(np.quantile(ps[elig], r["density_q"]))
    flag = elig & (ps >= thr) & (vm <= r["slow_mps"])
    out = []
    for kk in uk[flag]:
        wi, rest = divmod(int(kk), ny * nx)
        yi, xi = divmod(rest, nx)
        out.append((yi, xi, wi, wi * r["window_s"], (wi + 1) * r["window_s"],
                    floor["x0"] + (xi + 0.5) * c, floor["y0"] + (yi + 0.5) * c))
    return out, {"eligible": int(elig.sum()), "flagged": int(flag.sum()), "density_thr": thr, **r}


def score_events(flags, events, zones_by_name: dict, margin_m: float = 0.25):
    """An event (type, t0, t1, zone) is flagged iff some flagged cell-window overlaps it in time and its cell centre lies
    inside the event zone polygon (or within margin_m of its bounding box). Returns per-type hit rates + unmatched flags."""
    hits = []
    used = np.zeros(len(flags), bool)
    for ev in events:
        z = zones_by_name[ev["zone"]]
        pts = np.asarray(z.polygon, float)
        bx0, by0 = pts.min(0) - margin_m
        bx1, by1 = pts.max(0) + margin_m
        hit = False
        for j, f in enumerate(flags):
            _, _, _, ft0, ft1, cx, cy = f
            if ft1 > ev["t0"] and ft0 < ev["t1"] and (z.contains(cx, cy) or (bx0 <= cx <= bx1 and by0 <= cy <= by1)):
                hit = True
                used[j] = True
        hits.append(hit)
    hits = np.asarray(hits, bool)
    typ = np.asarray([e["type"] for e in events])
    out = {}
    for t in ("block", "browse"):
        mm = typ == t
        out[t] = {"n": int(mm.sum()), "flagged": int(hits[mm].sum()), "rate": float(hits[mm].mean()) if mm.any() else None}
    out["flags_total"] = int(len(flags))
    out["flags_outside_events"] = int((~used).sum())
    return out
