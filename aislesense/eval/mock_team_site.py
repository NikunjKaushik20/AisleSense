"""Build a mock team-footage site from existing MERL clips, through the exact drop-in path in docs/footage_dropin.md:
copy sites/team_template/ -> sites/<id>/, point each session at a video file, fill zones and labels.

It proves the plumbing (video -> app CV path -> rows cache -> tune/score split -> lock), not accuracy: MERL has one
shopper per clip and no blocking, so labels are mock: stopwatch = the shopper in the aisle for the whole clip; browse
events = MERL shelf-interaction intervals (5 action classes, merged when < 2 s apart); no block events.

  python eval/mock_team_site.py --site team_mock --clips 21_1,22_1
  python eval/stopwatch.py --site team_mock ; python eval/staged_events.py --site team_mock
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from aisle.schema import SITES_DIR, data_root  # noqa: E402

MERL_VID = "merl_x/Videos_MERL_Shopping_Dataset/{clip}_crop.mp4"
FPS = 30.0
ZONES = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"name": "Aisle", "kind": "section"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4.6, 0], [4.6, 3.4], [0, 3.4], [0, 0]]]}},
    {"type": "Feature", "properties": {"name": "Shelf front", "kind": "shelf_front"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0.5], [4.6, 0.5], [4.6, 3.4], [0, 3.4], [0, 0.5]]]}},
    {"type": "Feature", "properties": {"name": "Walkway", "kind": "walkway"},
     "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4.6, 0], [4.6, 0.5], [0, 0.5], [0, 0]]]}}]}


def clip_info(clip: str) -> tuple[float, list]:
    """(duration s, merged browse intervals in s) from the MERL video and label file."""
    import cv2
    import numpy as np
    import scipy.io as sio
    cap = cv2.VideoCapture(str(data_root() / MERL_VID.format(clip=clip)))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    tl = sio.loadmat(str(data_root() / "merl_x" / "Labels_MERL_Shopping_Dataset" / f"{clip}_label.mat"))["tlabs"]
    iv = sorted((float(s - 1) / FPS, float(e) / FPS) for k in range(5) for s, e in np.asarray(tl[k][0]).reshape(-1, 2))
    merged = []
    for s, e in iv:
        if merged and s - merged[-1][1] < 2.0:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return n / FPS, merged


def build(site_id: str, clips: list[str], tune_n: int = 1) -> Path:
    d = SITES_DIR / site_id
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SITES_DIR / "team_template", d)  # the drop-in step a team member does by hand
    site = yaml.safe_load((d / "site.yaml").read_text(encoding="utf-8"))
    merl = yaml.safe_load((SITES_DIR / "merl_demo" / "site.yaml").read_text(encoding="utf-8"))
    site.update({"id": site_id, "name": f"Mock team site from MERL clips {', '.join(clips)}",
                 "label": "Mock team footage (MERL clips; plumbing proof, not accuracy)", "floor": merl["floor"],
                 "track_buffer": 200})
    site["cameras"] = [{**merl["cameras"][0], "id": "phone1", "source": MERL_VID.format(clip=clips[0])}]
    (d / "site.yaml").write_text(yaml.safe_dump(site, sort_keys=False), encoding="utf-8")
    (d / "zones.geojson").write_text(json.dumps(ZONES), encoding="utf-8")
    names = [f"s{i + 1}" for i in range(len(clips))]
    sess = {"tune": names[:tune_n], "score": names[tune_n:],
            "sources": {s: {"video": MERL_VID.format(clip=c)} for s, c in zip(names, clips)}}
    (d / "sessions.yaml").write_text(yaml.safe_dump(sess, sort_keys=False), encoding="utf-8")
    sw = ["session,section,person,t_in,t_out"]
    ev = ["session,type,t0,t1,zone"]
    for s, c in zip(names, clips):
        dur, browse = clip_info(c)
        sw.append(f"{s},Aisle,{c},0.0,{dur:.2f}")
        ev += [f"{s},browse,{a:.2f},{b:.2f},Shelf front" for a, b in browse]
    (d / "labels" / "stopwatch.csv").write_text("\n".join(sw) + "\n", encoding="utf-8")
    (d / "labels" / "events.csv").write_text("\n".join(ev) + "\n", encoding="utf-8")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="team_mock")
    ap.add_argument("--clips", default="21_1,22_1")
    a = ap.parse_args(argv)
    d = build(a.site, a.clips.split(","))
    print(f"wrote {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
