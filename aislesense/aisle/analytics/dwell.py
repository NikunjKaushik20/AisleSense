"""S3 section dwell: person-seconds / entries, entries re-entering the same zone within the TTL merged [A34].

Pipeline over (t, track, x, y) rows:
  1. (S2) ghost hold fills presence after non-exit deaths; unique rebirths are stitched to one identity.
  2. Person-seconds: each sample weighs the time since that identity's previous sample (capped at max_gap);
     the first sample weighs 1/hz (heatmap rule, per identity so a ghost overlapping its stitched rebirth adds 0).
  3. Visits: per identity and zone, presence intervals; with merge, intervals of one identity in the same zone
     separated by <= TTL are one entry (a step out of the polygon or an ID break is not a new visit).
  4. Section table: person_s, entries, dwell_s = person_s / entries, median visit, long stays, prior beside it.
The baseline per-ID timer is the same code with ghost_hold=False, merge=False.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aisle.cv import ghost

LONG_MIN_S = 60.0  # a visit this long (or 2x the section prior, if larger) is a long stay


def _weights(rows, hz: float, max_gap: float):
    last = {}
    out = []
    for t, k, x, y in rows:
        p = last.get(k)
        w = 1.0 / hz if p is None else min(max(t - p, 0.0), max_gap)
        last[k] = t
        out.append(w)
    return out


def visits(rows, zone_fn, hz: float, ghost_hold: bool = True, merge: bool = True, ttl_s: float = ghost.TTL_S,
           k: float = ghost.K, radius=ghost.RADIUS_M, exited=None, t_end=None, max_gap: float | None = None,
           min_visit_s: float = 0.0) -> list:
    """[{person, track, zone, t_in, t_out, seconds}] ; seconds = person-seconds inside the visit."""
    rows = sorted(rows, key=lambda r: (r[0], str(r[1])))
    chain = {}
    if ghost_hold:
        rows, hs = ghost.apply(rows, hz, ttl_s, k, radius, exited, t_end)
        if merge:
            chain = ghost.stitch(hs)
    mg = max_gap if max_gap is not None else 2.0 / hz
    w = _weights([(t, chain.get(kk, kk), x, y) for t, kk, x, y in rows], hz, mg)
    gap = ttl_s if merge else 2.0 / hz + 1e-9
    # per (identity, zone): ordered samples -> intervals
    seq = {}
    for (t, kk, x, y), wi in zip(rows, w):
        z = zone_fn(x, y)
        pid = chain.get(kk, kk)
        seq.setdefault(pid, []).append((t, z, wi, kk))
    out = []
    for pid, ss in seq.items():
        ss.sort(key=lambda r: r[0])
        cur = {}  # zone -> open visit
        prev_z = None
        for t, z, wi, kk in ss:
            if not merge and prev_z is not None and z != prev_z and prev_z in cur:
                out.append(cur.pop(prev_z))  # baseline: leaving the zone closes the visit
            prev_z = z
            if z is None:
                continue
            v = cur.get(z)
            if v is not None and t - v["t_out"] > gap:
                out.append(cur.pop(z))
                v = None
            if v is None:
                cur[z] = {"person": pid, "track": kk, "zone": z, "t_in": t, "t_out": t, "seconds": wi}
            else:
                v["t_out"] = t
                v["seconds"] += wi
        out.extend(cur.values())
    out = [v for v in out if v["seconds"] >= min_visit_s]
    out.sort(key=lambda v: (v["t_in"], str(v["person"]), str(v["zone"])))
    return out


def section_table(vs: list, prior: dict | None = None, long_s: float = LONG_MIN_S, sections=None) -> dict:
    prior = prior or {}
    by = {}
    for v in vs:
        by.setdefault(v["zone"], []).append(v)
    names = list(sections) if sections is not None else sorted(by, key=str)
    rows, longs = [], []
    for s in names:
        L = by.get(s, [])
        ps = float(sum(v["seconds"] for v in L))
        p = prior.get(s)
        thr = max(long_s, 2.0 * p) if p else long_s
        lg = [v for v in L if v["seconds"] >= thr]
        longs += [{"track": str(v["track"]), "section": s, "seconds": round(v["seconds"], 2), "t_in": round(v["t_in"], 2)} for v in lg]
        rows.append({"section": s, "person_s": round(ps, 2), "entries": len(L),
                     "dwell_s": round(ps / len(L), 2) if L else None,
                     "median_s": round(float(np.median([v["seconds"] for v in L])), 2) if L else None,
                     "prior_s": None if p is None else round(float(p), 2), "n_long": len(lg)})
    longs.sort(key=lambda r: -r["seconds"])
    return {"sections": rows, "long_stays": longs}


def site_zone_fn(site):
    secs = [z for z in site.zones if z.kind == "section"]

    def f(x, y):
        for z in secs:
            if z.contains(x, y):
                return z.name
        return None
    return f


def site_prior(site_id: str, results_dir: Path | None = None) -> tuple[dict, dict]:
    """Per-section prior from the site's tune sessions (results/stopwatch_<tag>.json), never from the session shown."""
    from aisle.schema import RESULTS_DIR
    tag = "synth" if site_id == "synth_aisle" else site_id
    p = Path(results_dir or RESULTS_DIR) / f"stopwatch_{tag}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        return (d.get("tuned", {}).get("prior_per_section") or {},
                {"name": "Hand-timed average",
                 "source": f"Mean stay per section, timed by hand with a stopwatch in a separate tuning session (results/{p.name})"})
    return {}, {"name": "Hand-timed average (none yet)", "source": None}


def payload(site, events, session: str | None, hz: float | None = None) -> dict:
    """GET /api/dwell body (docs/api.md M5 additions) from stored Event rows of one session."""
    rows = [(e.t, e.track, e.x, e.y) for e in events]
    if hz is None:
        ts = np.unique(np.round([r[0] for r in rows], 4)) if rows else np.array([])
        hz = 1.0 / float(np.median(np.diff(ts))) if len(ts) > 2 else 1.0
    t_end = max((r[0] for r in rows), default=None)
    vs = visits(rows, site_zone_fn(site), hz, exited=ghost.floor_exit_fn(site.floor), t_end=t_end, min_visit_s=2.0)
    prior, pinfo = site_prior(site.id)
    tab = section_table(vs, prior, sections=[z.name for z in site.zones if z.kind == "section"])
    return {"session": session, "unit": "person-seconds per entry (ID breaks and re-entries within 8 s merged)",
            "method": {"ghost_hold_s": ghost.TTL_S, "merge_ttl_s": ghost.TTL_S, "min_visit_s": 2.0, "hz": round(hz, 3)},
            "prior": pinfo, **tab}
