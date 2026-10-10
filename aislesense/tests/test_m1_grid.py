import numpy as np
import pytest

from aisle.geo.grid import Grid

FPS, STRIDE = 30.0, 2
DT = STRIDE / FPS


def test_standing_person_10s_at_stride2_gives_10_person_seconds():
    g = Grid(0, 0, 10, 10, 1.0, nominal_dt=DT, max_gap=1.0, death_gap=3.0)
    n = int(round(10.0 / DT))
    for k in range(n):
        g.add(k * DT, "p1", 4.5 + 0.01 * np.sin(k), 5.5)
    g.expire()
    assert g.person_s.sum() == pytest.approx(10.0, abs=0.5)
    assert g.person_s[5, 4] == pytest.approx(10.0, abs=0.5)
    assert g.births[5, 4] == 1 and g.deaths[5, 4] == 1
    assert g.births.sum() == 1 and g.deaths.sum() == 1


def test_walker_spreads_time_and_births_deaths_at_ends():
    g = Grid(0, 0, 10, 2, 1.0, nominal_dt=0.2, max_gap=1.0, death_gap=3.0)
    for k in range(50):  # 1 m/s over 10 m at 5 Hz
        g.add(k * 0.2, "w", 0.1 + k * 0.2, 1.5)
    g.expire()
    assert g.person_s.sum() == pytest.approx(10.0, abs=0.3)
    assert (g.person_s[1] > 0.5).sum() == 10
    assert g.births[1, 0] == 1 and g.deaths[1, 9] == 1


def test_gap_longer_than_death_gap_splits_track():
    g = Grid(0, 0, 4, 4, 1.0, nominal_dt=1.0, max_gap=2.0, death_gap=5.0)
    g.add(0, "a", 1.5, 1.5)
    g.add(20, "a", 2.5, 2.5)
    g.expire()
    assert g.births.sum() == 2 and g.deaths.sum() == 2
    assert g.person_s.sum() == pytest.approx(2.0)


def test_off_grid_dropped_and_counted():
    g = Grid(0, 0, 2, 2, 1.0)
    g.add(0, "a", -5, 1)
    assert g.person_s.sum() == 0 and g.n_off_grid == 1


def test_expire_with_now_only_closes_stale():
    g = Grid(0, 0, 4, 4, 1.0, death_gap=5.0)
    g.add(0, "old", 1, 1)
    g.add(9, "new", 2, 2)
    assert g.expire(now=10) == 1
    assert list(g.active_tracks()) == ["new"]
