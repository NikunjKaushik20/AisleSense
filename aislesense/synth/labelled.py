"""Labelled synthetic scenarios for the M2 eval: walkway blocks, shelf-front browses and plain traffic.

Events are staged one per time slot (no overlap), in shuffled order:
  block  : 2-3 lingerers stand together in the walkway (speed ~0) for stay_s while >= 3 passers squeeze past slowly.
  browse : browse_lingerers (default 1-3) stand at one shelf front for stay_s; background traffic continues.
Background: passers walk the main lane end to end at 1.0-1.5 m/s (plain traffic).
Optional ID breaks (id_breaks_per_min) split a true track into fragments with new IDs, to mimic tracker breaks.

generate_labelled(scenario, site, seed) -> (rows, events, visits)
  rows   : [(t, track, x, y)] sorted by t (tracks are post-break IDs, what a tracker would emit)
  events : [{type, t0, t1, zone, x, y, n_lingerers, n_passers}]  (staged-event labels, events.csv protocol)
  visits : [{person, section, t_in, t_out}] section visits of TRUE people >= min_visit_s (stopwatch.csv protocol)
Nothing here is ever read by a rule; labels are only read by scorers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

SCEN_DIR = Path(__file__).resolve().parent / "scenarios"


def load(name: str) -> dict:
    return yaml.safe_load((SCEN_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _walk(p0, p1, speed, t0, hz):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dur = max(np.linalg.norm(p1 - p0) / speed, 1.0 / hz)
    n = max(int(np.ceil(dur * hz)), 1)
    ts = t0 + np.arange(n) / hz
    f = (ts - t0) / dur
    return ts, p0 + f[:, None] * (p1 - p0), t0 + n / hz


def _stand(spot, t0, dur, hz, rng, jitter=0.05):
    n = max(int(dur * hz), 1)
    return t0 + np.arange(n) / hz, spot + rng.normal(0, jitter, (n, 2)), t0 + n / hz


def _bbox(poly):
    a = np.asarray(poly, float)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def generate_labelled(sc: dict, site, seed: int):
    rng = np.random.default_rng(int(seed))
    hz = float(sc.get("hz", 5))
    fl = site.floor
    x0, W = fl["x0"], fl["width"]
    zones = {z.name: z for z in site.zones}
    walk = zones[sc.get("walkway", "Main walkway")]
    wx0, wy0, wx1, wy1 = _bbox(walk.polygon)
    lane = sc.get("lane_y", [2.5, 4.5])
    slot = float(sc.get("slot_s", 90))
    types = ["block"] * int(sc.get("n_block", 24)) + ["browse"] * int(sc.get("n_browse", 24))
    rng.shuffle(types)
    T = slot * len(types) + slot
    true_paths = {}  # person -> (ts, ps)
    events = []
    nid = 0

    def add(ts, ps, pid):
        a = true_paths.setdefault(pid, ([], []))
        a[0].append(ts); a[1].append(ps)

    # plain traffic
    p = sc.get("passers", {})
    rate = float(p.get("rate_per_s", 0.1))
    smin, smax = p.get("speed_mps", [1.0, 1.5])
    t = rng.exponential(1 / rate)
    while t < T:
        nid += 1
        y = rng.uniform(*lane)
        a, b = ([x0 + 0.05, y], [x0 + W - 0.05, y]) if rng.random() < 0.5 else ([x0 + W - 0.05, y], [x0 + 0.05, y])
        ts, ps, _ = _walk(a, b, rng.uniform(smin, smax), t, hz)
        add(ts, ps, f"p{nid}")
        t += rng.exponential(1 / rate)

    shelf_fronts = [z for z in site.zones if z.kind == "shelf_front"]
    entry = np.array([x0 + 1.0, lane[0]])
    s0, s1 = sc.get("stay_s", [20, 60])
    for i, typ in enumerate(types):
        ts0 = slot * (i + 0.5)
        stay = float(rng.uniform(s0, s1))
        if typ == "block":
            spot = np.array([rng.uniform(wx0 + 2.5, wx1 - 2.5), rng.uniform(lane[0] + 0.5, lane[1] - 0.5)])
            bl = sc.get("block_lingerers", [2, 3])
            zname, nl = walk.name, int(rng.integers(bl[0], bl[1] + 1))
        else:
            z = shelf_fronts[int(rng.integers(len(shelf_fronts)))]
            bx0, by0, bx1, by1 = _bbox(z.polygon)
            spot = np.array([rng.uniform(bx0 + 0.6, bx1 - 0.6), rng.uniform(by0 + 0.3, by1 - 0.3)])
            wl = sc.get("browse_lingerers", [1, 3])
            zname, nl = z.name, int(rng.integers(wl[0], wl[1] + 1))
        arrive = ts0 + 8.0
        for j in range(nl):
            nid += 1
            pid = f"l{nid}"
            sp = spot + np.array([0.45 * (j - (nl - 1) / 2), rng.normal(0, 0.1)])
            w1t, w1p, ta = _walk(entry, sp, 1.2, arrive - np.linalg.norm(sp - entry) / 1.2, hz)
            st, sp_, tl = _stand(sp, ta, stay, hz, rng)
            w2t, w2p, _ = _walk(sp, [x0 + W - 0.5, lane[0]], 1.2, tl, hz)
            for ts, ps in ((w1t, w1p), (st, sp_), (w2t, w2p)):
                add(ts, ps, pid)
        npass = 0
        if typ == "block":
            # passers forced through the blocked spot, slowed while squeezing past (congestion)
            k = int(sc.get("block_passers", [3, 6])[0] + rng.integers(0, sc.get("block_passers", [3, 6])[1] - sc.get("block_passers", [3, 6])[0] + 1))
            for q in range(k):
                nid += 1
                tq = arrive + (q + 0.5) * stay / k
                y = spot[1] + rng.uniform(-0.8, 0.8)
                left = rng.random() < 0.5
                a = np.array([spot[0] - 4.0, y]) if left else np.array([spot[0] + 4.0, y])
                bpt = np.array([spot[0] + 4.0, y]) if left else np.array([spot[0] - 4.0, y])
                m = (a + bpt) / 2
                t1_, p1_, te = _walk(a, m - np.sign(bpt - a) * np.array([1.5, 0]), 1.2, tq, hz)
                t2_, p2_, te2 = _walk(p1_[-1], m + np.sign(bpt - a) * np.array([1.5, 0]), float(rng.uniform(0.3, 0.5)), te, hz)
                t3_, p3_, _ = _walk(p2_[-1], bpt, 1.2, te2, hz)
                for ts, ps in ((t1_, p1_), (t2_, p2_), (t3_, p3_)):
                    add(ts, ps, f"q{nid}")
                npass += 1
        events.append({"type": typ, "t0": float(arrive), "t1": float(arrive + stay), "zone": zname,
                       "x": float(spot[0]), "y": float(spot[1]), "n_lingerers": nl, "n_passers": npass})

    # assemble true rows, clip to floor
    true_rows = []
    for pid, (tl_, pl_) in true_paths.items():
        ts = np.concatenate(tl_); ps = np.concatenate(pl_)
        ps[:, 0] = np.clip(ps[:, 0], x0 + 0.01, x0 + W - 0.01)
        ps[:, 1] = np.clip(ps[:, 1], fl["y0"] + 0.01, fl["y0"] + fl["height"] - 0.01)
        o = np.argsort(ts, kind="stable")
        for tt, (xx, yy) in zip(ts[o], ps[o]):
            if tt < T:
                true_rows.append((float(tt), pid, float(xx), float(yy)))

    visits = section_visits(true_rows, site, float(sc.get("min_visit_s", 2.0)), hz)
    # opaque track IDs (no role prefix reaches a rule), then tracker-like breaks
    pids = sorted(true_paths)
    anon = {pid: f"t{i}" for pid, i in zip(pids, rng.permutation(len(pids)))}
    rows = apply_id_breaks([(t, anon[k], x, y) for t, k, x, y in true_rows], float(sc.get("id_breaks_per_min", 0.0)), rng)
    rows.sort(key=lambda r: (r[0], r[1]))
    events.sort(key=lambda e: e["t0"])
    return rows, events, visits


def apply_id_breaks(rows, per_min: float, rng):
    if per_min <= 0:
        return list(rows)
    by = {}
    for r in rows:
        by.setdefault(r[1], []).append(r)
    out = []
    for pid in sorted(by):
        rr = sorted(by[pid])
        frag, t_next = 0, rr[0][0] + rng.exponential(60.0 / per_min)
        for t, _, x, y in rr:
            while t >= t_next:
                frag += 1
                t_next += rng.exponential(60.0 / per_min)
            out.append((t, f"{pid}~{frag}" if frag else pid, x, y))
    return out


def section_visits(rows, site, min_visit_s: float, hz: float):
    """Runs of one person inside one section polygon (gap <= 2 samples), >= min_visit_s."""
    secs = [z for z in site.zones if z.kind == "section"]
    by = {}
    for t, k, x, y in rows:
        by.setdefault(k, []).append((t, x, y))
    out = []
    for k in sorted(by):
        cur = None
        for t, x, y in sorted(by[k]):
            z = next((s.name for s in secs if s.contains(x, y)), None)
            if cur and (z != cur[0] or t - cur[2] > 2.0 / hz):
                if cur[0] and cur[2] - cur[1] >= min_visit_s:
                    out.append({"person": k, "section": cur[0], "t_in": cur[1], "t_out": cur[2]})
                cur = None
            if cur is None:
                cur = [z, t, t]
            else:
                cur[2] = t
        if cur and cur[0] and cur[2] - cur[1] >= min_visit_s:
            out.append({"person": k, "section": cur[0], "t_in": cur[1], "t_out": cur[2]})
    return out
