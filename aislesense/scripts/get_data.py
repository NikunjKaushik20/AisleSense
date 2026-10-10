"""Download and unpack the public datasets into <repo>/data (or AISLE_DATA).

    python scripts/get_data.py              # MERL Shopping + MOT20 labels + one ATC day (about 2.1 GB)
    python scripts/get_data.py --heldout    # also ATC part 1 (4.2 GB) for the unseen day 2012-11-21
    python scripts/get_data.py --mot20-video  # also MOT20 video (5.0 GB), only for eval/mot20_video-style runs

Downloads resume after a dropped connection and are skipped when the file is already complete.
Layout produced (what the sites and evals read):
    data/merl_x/Videos_MERL_Shopping_Dataset/*.mp4, data/merl_x/Labels_MERL_Shopping_Dataset/*.mat
    data/mot20_labels_x/MOT20Labels/train/...
    data/atc_x/atc-YYYYMMDD.npz      (t, id, x, y columns; the CSV is deleted after conversion)

Licences: MERL Shopping (MERL research licence), MOT20 (CC BY-NC-SA 3.0), ATC (research use only;
cite Brscic et al., 2013). Check them before any non-research use.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aisle.schema import data_root  # noqa: E402

MERL = "https://www.merl.com/pub/tmarks/MERL_Shopping_Dataset/"
MOT = "https://motchallenge.net/data/"
ATC = "https://dil.atr.jp/crest2010_HRI/ATC_dataset/"


def fetch(url: str, dest: Path) -> Path:
    """Download url to dest, resuming a partial file; skip when the size already matches."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        total = int(r.headers.get("Content-Length", 0))
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        print(f"have  {dest.name} ({total / 1e9:.2f} GB)")
        return dest
    for attempt in range(30):
        have = dest.stat().st_size if dest.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range: start over
                    f.truncate(0)
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    print(f"\rget   {dest.name} {f.tell() / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
            print()
        except OSError as e:
            print(f"\n  connection dropped ({e}); resuming, attempt {attempt + 1}")
            continue
        if not total or dest.stat().st_size >= total:
            return dest
    raise SystemExit(f"could not download {url}")


def unzip(zf: Path, out: Path, marker: str) -> None:
    if (out / marker).exists():
        print(f"have  {out.name}/{marker}")
        return
    print(f"unzip {zf.name} -> {out}")
    with zipfile.ZipFile(zf) as z:
        z.extractall(out)


def atc_day(archive: Path, day: str, out: Path) -> None:
    """Pull one day's CSV out of an ATC .7z and store t,id,x,y as npz (the CSVs are large)."""
    npz = out / f"atc-{day}.npz"
    if npz.exists():
        print(f"have  {npz.name}")
        return
    import numpy as np
    import pandas as pd
    import py7zr
    out.mkdir(parents=True, exist_ok=True)
    name = f"atc-{day}.csv"
    print(f"7z    {archive.name}: {name}")
    with py7zr.SevenZipFile(archive) as z:
        z.extract(out, targets=[name])
    a = pd.read_csv(out / name, header=None, usecols=[0, 1, 2, 3], names=["t", "id", "x", "y"],
                    dtype={"t": "float64", "id": "int64", "x": "float32", "y": "float32"})
    np.savez(npz, **{c: a[c].values for c in a.columns})
    (out / name).unlink()
    print(f"      {npz.name}: {len(a)} rows, {a.id.nunique()} people")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--heldout", action="store_true", help="also ATC part 1 (4.2 GB) for the unseen day 2012-11-21")
    ap.add_argument("--mot20-video", action="store_true", help="also the MOT20 video (5.0 GB)")
    a = ap.parse_args(argv)
    d = data_root()
    print(f"data root: {d}")

    for name in ("Videos_MERL_Shopping_Dataset.zip", "Labels_MERL_Shopping_Dataset.zip"):
        zf = fetch(MERL + name, d / "merl_shopping" / name)
        unzip(zf, d / "merl_x", name.removesuffix(".zip"))
    unzip(fetch(MOT + "MOT20Labels.zip", d / "mot20" / "MOT20Labels.zip"), d / "mot20_labels_x", "MOT20Labels")
    atc_day(fetch(ATC + "atc-20121114.7z", d / "atc" / "atc-20121114.7z"), "20121114", d / "atc_x")
    if a.heldout:
        atc_day(fetch(ATC + "atc-tracking-part1.7z", d / "atc" / "atc-tracking-part1.7z"), "20121121", d / "atc_x")
    if a.mot20_video:
        fetch(MOT + "MOT20.zip", d / "mot20" / "MOT20.zip")  # read in place by the MOT20 video eval
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
