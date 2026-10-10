import numpy as np

from aisle.geo import homography as hg

IMG = [[100, 620], [820, 620], [700, 120], [220, 120]]
FLOOR = [[0, 0], [4, 0], [4, 6], [0, 6]]


def test_round_trip_on_control_points_below_1e6_m():
    H = hg.fit(IMG, FLOOR)
    out = hg.apply(H, IMG)
    assert np.abs(out - np.asarray(FLOOR, float)).max() < 1e-6
    back = hg.apply(hg.inverse(H), out)
    assert np.abs(back - np.asarray(IMG, float)).max() < 1e-6


def test_inverse_round_trip_on_interior_points():
    H = hg.fit(IMG, FLOOR)
    pts = np.array([[400, 400], [300, 200], [600, 550]], float)
    assert np.abs(hg.apply(hg.inverse(H), hg.apply(H, pts)) - pts).max() < 1e-6


def test_foot_point_is_bottom_centre():
    fp = hg.foot_points([[10, 20, 30, 80], [0, 0, 4, 4]])
    assert fp.tolist() == [[20, 80], [2, 4]]
    assert hg.foot_points(np.zeros((0, 4))).shape == (0, 2)


def test_bad_input_raises():
    import pytest
    with pytest.raises(ValueError):
        hg.fit(IMG[:3], FLOOR[:3])
