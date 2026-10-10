import pytest

from aisle.schema import Event, EVENT_FIELDS, list_sites, load_site
from aisle.geo import homography as hg


def test_event_row_round_trip():
    e = Event("s1", "cam0", 1.5, "7", 2.0, 3.0, 0.9)
    assert len(e.as_row()) == len(EVENT_FIELDS)
    assert Event.from_row(e.as_row()) == e
    assert set(e.to_dict()) == set(EVENT_FIELDS)


def test_all_required_sites_load():
    sites = list_sites()
    for sid in ("merl_demo", "atc", "synth_aisle", "team_template"):
        assert sid in sites
        s = load_site(sid)
        assert s.floor["width"] > 0 and s.floor["height"] > 0
        assert s.kind in ("video", "atc", "synth")


def test_video_site_cameras_have_valid_homography():
    s = load_site("merl_demo")
    assert s.cameras
    for c in s.cameras:
        H = hg.fit(c.image_points, c.floor_points)
        assert abs(hg.apply(H, c.image_points) - __import__("numpy").asarray(c.floor_points, float)).max() < 1e-6


def test_zones_parse_and_point_lookup():
    s = load_site("synth_aisle")
    kinds = {z.kind for z in s.zones}
    assert {"section", "walkway", "shelf_front"} <= kinds
    sec = next(z for z in s.zones if z.kind == "section")
    xs = [p[0] for p in sec.polygon]; ys = [p[1] for p in sec.polygon]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    assert s.zone_of(cx, cy) == sec.name
    assert s.zone_of(-100, -100) is None


def test_atc_site_is_labelled_trajectory_replay():
    s = load_site("atc")
    assert "trajectory replay" in s.to_dict()["label"].lower()
