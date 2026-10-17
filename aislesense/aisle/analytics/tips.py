"""Plain-language tips over DEAD / UNSEEN / WATCH / BLOCK / BROWSE / dwell outliers.

One tip per (kind, zone), id = "<KIND>:<zone>", so diff.py can match tips across two sessions. Severity 3 = act now,
2 = worth a look, 1 = note. Tips state only counts from this session; no accuracy claim is made in the text.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from aisle.analytics import deadzone as dz

DWELL_RATIO = 2.0      # section dwell > 2x prior (or 2x the median section when no prior) -> DWELL tip
MIN_ENTRIES = 5
MIN_DEAD_CELLS = 2     # one quiet square metre is noise; two or more adjacent-ish is a pattern


def _where(z):
    return z if z else "the open floor"


def from_parts(site, dz_state, flags: list, dwell: dict | None) -> list:
    """dz_state: (nx, ny) array of cell states (aisle.analytics.deadzone.states)."""
    tips = []
    zc = dz.zone_cells(site, dz_state) if dz_state is not None else {}
    for z, n in sorted(zc.get("dead", {}).items(), key=lambda kv: -kv[1]):
        if n >= MIN_DEAD_CELLS:
            tips.append({"kind": "DEAD", "zone": z, "severity": 3 if n >= 5 else 2, "cells": n,
                         "text": f"{n} m² of {_where(z)} get under a quarter of their neighbours' visitors while the camera "
                                 f"sees them well. Move a promotion or a high-demand item there, or open the route to it."})
    for z, n in sorted(zc.get("watch", {}).items(), key=lambda kv: -kv[1]):
        tips.append({"kind": "WATCH", "zone": z, "severity": 1, "cells": n,
                     "text": f"{n} m² of {_where(z)} look dead today but were not in the previous session. "
                             f"Watch one more session before moving stock."})
    for z, n in sorted(zc.get("unseen", {}).items(), key=lambda kv: -kv[1]):
        tips.append({"kind": "UNSEEN", "zone": z, "severity": 2 if n >= 10 else 1, "cells": n,
                     "text": f"{n} m² of {_where(z)} sit where tracks start or stop (camera edge or weak view), so their "
                             f"counts are not trusted. Add or move a camera before calling them dead."})
    blocks = Counter(f["zone"] for f in flags if f["type"] == "BLOCK")
    for z, n in blocks.most_common():
        tips.append({"kind": "BLOCK", "zone": z, "severity": 3, "count": n,
                     "text": f"{_where(z)} was blocked {n} time{'s' if n != 1 else ''}: people stood in the walkway while "
                             f"others squeezed past. Move carts, stacked stock or displays out of the walkway."})
    browses = Counter(f["zone"] for f in flags if f["type"] == "BROWSE")
    for z, n in browses.most_common():
        tips.append({"kind": "BROWSE", "zone": z, "severity": 1, "count": n,
                     "text": f"{n} browsing group{'s' if n != 1 else ''} lingered at {_where(z)}. That is shelf engagement, "
                             f"not a bottleneck: keep it stocked and faced."})
    if dwell:
        rows = [r for r in dwell.get("sections", []) if r.get("dwell_s") is not None and r.get("entries", 0) >= MIN_ENTRIES]
        med = float(np.median([r["dwell_s"] for r in rows])) if rows else None
        for r in rows:
            ref, what = (r["prior_s"], "the hand-timed average") if r.get("prior_s") else (med, "the median section")
            if ref and r["dwell_s"] > DWELL_RATIO * ref:
                tips.append({"kind": "DWELL", "zone": r["section"], "severity": 2,
                             "text": f"People spend {r['dwell_s']:.0f} s per visit in {r['section']}, over {DWELL_RATIO:.0f}x "
                                     f"{what} ({ref:.0f} s). Check for a queue or confusing layout, or a display that works."})
    for t in tips:
        t["id"] = f"{t['kind']}:{t['zone'] or 'floor'}"
    tips.sort(key=lambda t: (-t["severity"], t["kind"], str(t["zone"])))
    return tips


def payload(site, session: str, dz_state, flags: list, dwell: dict | None) -> dict:
    return {"session": session, "site": site.id, "tips": from_parts(site, dz_state, flags, dwell)}
