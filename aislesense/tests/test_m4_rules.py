"""M4 unit tests: D-1 coverage gate, D-2 walkway vs shelf, journeys, tips, diff."""
import numpy as np
import pandas as pd
from scipy import ndimage

from aisle.analytics import block, deadzone as dz, diff, journeys, tips
from aisle.schema import load_site


# ---------------------------------------------------------------- D-1
def _walkers(n=3000, size=30, seed=0):
    """Straight walks across a size x size floor at 1 Hz; returns DataFrame id, s, x, y sorted by (id, s)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a, b = rng.uniform(0, size, 2), rng.uniform(0, size, 2)
        # start/end on the border so terminations sit on the edge, like a real camera view
        side = rng.integers(4)
        a[side % 2] = 0.01 if side < 2 else size - 0.01
        b[(side + 1) % 2] = size - 0.01 if side < 2 else 0.01
        steps = max(int(np.linalg.norm(b - a) / 1.2), 2)
        t0 = int(rng.integers(0, 3600))
        for k in range(steps + 1):
            p = a + (b - a) * k / steps
            out.append((i, t0 + k, p[0], p[1]))
    return pd.DataFrame(out, columns=["id", "s", "x", "y"]).sort_values(["id", "s"]).reset_index(drop=True)


def _grid(b, size=30):
    return dz.counts(b.id.values, b.x.values, b.y.values, 0.0, 0.0, size, size)


def test_blind_patch_never_turns_dead():
    b = _walkers()
    vis, term = _grid(b)
    obs = vis >= 1
    hits_decile = 0
    for seed in range(3):
        patch = dz.patches(obs, ndimage.distance_transform_edt(obs), vis, seed)
        assert patch.sum() >= 16
        bl, ig = dz.inject(b, patch, 0.0, 0.0, seed, 0.01)
        v2, t2 = _grid(bl)
        a = dz.aware(v2, t2)
        assert not (a["dead"] & patch).any(), "a blind (weak-camera) patch must never be called DEAD"
        assert (a["unseen"] & patch).sum() >= 0.8 * (patch & (v2 >= 1)).sum()
        hits_decile += int((dz.decile_rule(v2) & patch).sum())
    assert hits_decile > 0  # the baseline does fall for it on the same data


def test_ignored_patch_can_turn_dead():
    b = _walkers()
    vis, term = _grid(b)
    obs = vis >= 1
    patch = dz.patches(obs, ndimage.distance_transform_edt(obs), vis, 0)
    _, ig = dz.inject(b, patch, 0.0, 0.0, 0, 0.01)
    v2, t2 = _grid(ig)
    assert (dz.aware(v2, t2)["dead"] & patch).sum() > 0


def test_watch_when_not_dead_in_previous_session():
    b = _walkers()
    vis, term = _grid(b)
    obs = vis >= 1
    patch = dz.patches(obs, ndimage.distance_transform_edt(obs), vis, 0)
    _, ig = dz.inject(b, patch, 0.0, 0.0, 0, 0.01)
    v2, t2 = _grid(ig)
    prev = dz.aware(vis, term)["dead"]
    st, a = dz.states(v2, t2, prev_dead=prev)
    assert ((st == "watch") & patch).sum() == (a["dead"] & patch & ~prev).sum() > 0
    st2, _ = dz.states(v2, t2, prev_dead=a["dead"])
    assert not (st2 == "watch").any()


def test_payload_row_major_and_keep():
    site = load_site("synth_aisle")
    rows = [(float(t), f"p{i}", 1.0 + 0.5 * t, 3.0) for i in range(5) for t in range(30)]
    p = dz.payload(site, rows, "s", keep=1.0)
    assert len(p["state"]) == len(p["decile"]) == len(p["coverage"]) == p["nx"] * p["ny"] == 200
    assert set(p["state"]) <= set(dz.STATES)
    # cell (ix=1, iy=3) is visited -> row-major index 3*20+1
    assert p["state"][3 * 20 + 1] != "unknown" and p["state"][0] == "unknown"


# ---------------------------------------------------------------- D-2
def _stand(pid, x, y, t0, dur, hz=5):
    return [(t0 + k / hz, pid, x + 0.02 * np.sin(k), y) for k in range(int(dur * hz))]


def _pass(pid, y, t0, x0=2.0, x1=19.0, v=1.2, hz=5):
    n = int((x1 - x0) / v * hz)
    return [(t0 + k / hz, pid, x0 + (x1 - x0) * k / n, y) for k in range(n + 1)]


def _passers(y, t0, n=4):
    return [r for q in range(n) for r in _pass(f"q{q}", y, t0 + 3 * q)]


def test_lingerer_at_shelf_never_block():
    site = load_site("synth_aisle")
    # shelf front 'Drinks front' is y 5..6; passers walk the walkway edge at y=4.8 (< 2 m away)
    rows = _stand("a", 10.0, 5.5, 0.0, 30) + _passers(4.8, 2.0)
    fl = block.flags(rows, site, 5)
    assert any(f["type"] == "BROWSE" and f["zone"] == "Drinks front" for f in fl)
    assert not any(f["type"] == "BLOCK" for f in fl)


def test_walkway_lingerer_with_passers_is_block():
    site = load_site("synth_aisle")
    rows = _stand("cart", 10.0, 3.5, 0.0, 30) + _passers(3.0, 2.0)
    fl = block.flags(rows, site, 5)
    assert [f["type"] for f in fl] == ["BLOCK"] and fl[0]["zone"] == "Main walkway" and fl[0]["n_passers"] >= 2


def test_walkway_lingerer_alone_is_unflagged():
    site = load_site("synth_aisle")
    assert block.flags(_stand("a", 10.0, 3.5, 0.0, 30), site, 5) == []


def test_short_stop_is_not_a_lingerer():
    site = load_site("synth_aisle")
    rows = _stand("a", 10.0, 3.5, 0.0, 3) + _passers(3.0, 0.0)
    assert block.flags(rows, site, 5) == []


# ---------------------------------------------------------------- journeys, tips, diff
def test_journeys_counts_section_transitions():
    site = load_site("synth_aisle")
    path = [(1.5, 2.5)] * 3 + [(3.0, 7.0)] * 3 + [(10.0, 7.0)] * 3 + [(3.0, 7.0)] * 3
    rows = [(float(i), "p", x, y) for i, (x, y) in enumerate(path)]
    j = journeys.payload(site, rows, "s")
    names = [n["name"] for n in j["nodes"]]
    links = {(names[l["source"]], names[l["target"]]): l["value"] for l in j["links"]}
    assert links == {("Entrance", "Snacks"): 1, ("Snacks", "Drinks"): 1, ("Drinks", "Snacks"): 1}


def test_removed_cart_clears_in_diff():
    site = load_site("synth_aisle")
    before = _stand("cart", 10.0, 3.5, 0.0, 30) + _passers(3.0, 2.0)
    after = _passers(3.0, 2.0)
    ta = tips.from_parts(site, None, block.flags(before, site, 5), None)
    tb = tips.from_parts(site, None, block.flags(after, site, 5), None)
    d = diff.payload("a", "b", ta, tb)
    assert {"kind": "BLOCK", "zone": "Main walkway"}.items() <= d["cleared"][0].items()
    assert d["added"] == [] and d["kept"] == []


def test_tips_kinds_and_ids():
    site = load_site("synth_aisle")
    st = np.full((20, 10), "ok", dtype=object)
    st[0:3, 7] = "dead"; st[15, 0:12 // 2] = "unseen"; st[8, 8] = "watch"
    fl = [{"type": "BROWSE", "zone": "Dairy front"}]
    dw = {"sections": [{"section": "Snacks", "dwell_s": 100.0, "entries": 9, "prior_s": 20.0},
                       {"section": "Drinks", "dwell_s": 10.0, "entries": 9, "prior_s": 20.0}]}
    t = tips.from_parts(site, st, fl, dw)
    kinds = {x["kind"] for x in t}
    assert kinds == {"DEAD", "UNSEEN", "WATCH", "BROWSE", "DWELL"}
    assert all(x["id"] == f"{x['kind']}:{x['zone'] or 'floor'}" and x["severity"] in (1, 2, 3) for x in t)
    assert [x for x in t if x["kind"] == "DWELL"][0]["zone"] == "Snacks"
