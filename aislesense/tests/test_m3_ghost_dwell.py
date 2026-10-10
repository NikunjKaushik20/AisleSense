"""M3 unit tests: ghost hold (S2) and section dwell = person-seconds / merged entries (S3)."""
import pytest

from aisle.analytics import dwell
from aisle.cv import ghost, track

HZ = 5.0


def stand(track_id, t0, t1, x, y, hz=HZ):
    n = int(round((t1 - t0) * hz))
    return [(t0 + i / hz, track_id, x, y) for i in range(n)]


def zone_left(x, y):
    return "A" if x < 5 else None


def test_default_buffer_is_s1_and_yaml_written():
    assert track.DEFAULT_BUFFER == 200
    p = track.tracker_yaml()
    assert p.endswith("bytetrack_b200.yaml")
    import yaml
    assert yaml.safe_load(open(p))["track_buffer"] == 200


def test_frozen_ghost_params():
    assert (ghost.TTL_S, ghost.K) == (8.0, 1.0)


def test_hold_after_non_exit_death_until_ttl():
    rows = stand("a", 0, 10, 2, 2)
    hs = ghost.holds(rows, ttl_s=8, k=1, radius=1.0, exited=set())
    assert len(hs) == 1 and hs[0]["track"] == "a"
    assert hs[0]["t1"] - hs[0]["t0"] == pytest.approx(8.0)
    assert (hs[0]["x"], hs[0]["y"]) == (2, 2)


def test_no_hold_after_exit():
    rows = stand("a", 0, 10, 2, 2)
    assert ghost.holds(rows, ttl_s=8, exited={"a"}) == []


def test_nearby_rebirth_stitches_hold_keeps_ttl():
    rows = stand("a", 0, 10, 2, 2) + stand("b", 13, 20, 2.3, 2)
    hs = ghost.holds(rows, ttl_s=8, k=1, radius=1.0, exited={"b"})
    assert len(hs) == 1 and hs[0]["t1"] - hs[0]["t0"] == pytest.approx(8.0) and hs[0]["reborn"] == "b"
    chain = ghost.stitch(hs)
    assert chain["b"] == "a"


def test_far_rebirth_does_not_end_hold_or_stitch():
    rows = stand("a", 0, 10, 2, 2) + stand("b", 13, 20, 8, 2)
    hs = ghost.holds(rows, ttl_s=8, k=1, radius=1.0, exited={"b"})
    assert hs[0]["reborn"] is None and hs[0]["candidates"] == []
    assert ghost.stitch(hs) == {}


def test_ambiguous_rebirth_abstains_from_stitch():
    rows = stand("a", 0, 10, 2, 2) + stand("c", 0, 10, 2.4, 2) + stand("b", 12, 20, 2.2, 2)
    hs = ghost.holds(rows, ttl_s=8, k=1, radius=1.0, exited={"b"})
    assert ghost.stitch(hs) == {}  # b is claimed by two ghosts -> no link


def test_held_rows_fill_ghost_samples():
    rows = stand("a", 0, 10, 2, 2)
    out, hs = ghost.apply(rows, HZ, ttl_s=8, exited=set())
    extra = [r for r in out if r[0] > rows[-1][0]]
    assert len(extra) == pytest.approx(8 * HZ, abs=1) and all(r[1] == "a" for r in extra)
    out2, _ = ghost.apply(rows, HZ, ttl_s=8, exited=set(), t_end=12.0)
    assert max(r[0] for r in out2) <= 12.0 + 1e-9  # never holds past the end of the data


def test_dwell_person_seconds_over_merged_entries():
    # one person in zone A for 10 s, ID breaks at 10 s, reborn 3 s later nearby for 10 s more
    rows = stand("a", 0, 10, 2, 2) + stand("b", 13, 23, 2.2, 2) + stand("z", 0, 1, 9, 9)
    v_naive = dwell.visits(rows, zone_left, HZ, ghost_hold=False, merge=False)
    v_s3 = dwell.visits(rows, zone_left, HZ, ghost_hold=True, merge=True, exited={"b", "z"})
    assert len([v for v in v_naive if v["zone"] == "A"]) == 2
    a = [v for v in v_s3 if v["zone"] == "A"]
    assert len(a) == 1 and a[0]["seconds"] == pytest.approx(23.0, abs=0.5)


def test_reentry_within_ttl_merges_same_person():
    rows = stand("a", 0, 5, 2, 2) + stand("a", 5, 8, 7, 2) + stand("a", 8, 13, 2, 2)
    v = dwell.visits(rows, zone_left, HZ, ghost_hold=False, merge=True, ttl_s=8)
    assert len([x for x in v if x["zone"] == "A"]) == 1
    v2 = dwell.visits(rows, zone_left, HZ, ghost_hold=False, merge=False)
    assert len([x for x in v2 if x["zone"] == "A"]) == 2


def test_section_table_shape_and_long_stays():
    rows = stand("a", 0, 100, 2, 2) + stand("b", 0, 10, 3, 3)
    tab = dwell.section_table(dwell.visits(rows, zone_left, HZ, ghost_hold=False, merge=False), prior={"A": 20.0}, long_s=60)
    s = tab["sections"][0]
    assert {"section", "person_s", "entries", "dwell_s", "median_s", "prior_s", "n_long"} <= set(s)
    assert s["entries"] == 2 and s["dwell_s"] == pytest.approx(55, abs=1) and s["prior_s"] == 20.0 and s["n_long"] == 1
    assert tab["long_stays"][0]["track"] == "a"
