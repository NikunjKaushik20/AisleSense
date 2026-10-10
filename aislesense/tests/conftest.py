import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Datasets live outside git. Tests marked needs_data("merl") etc. skip until scripts/get_data.py has run.
DATASETS = {
    "merl": "merl_x/Videos_MERL_Shopping_Dataset",
    "atc": "atc_x/atc-20121114.npz",
}


def pytest_configure(config):
    config.addinivalue_line("markers", "needs_data(*names): skip unless these datasets are downloaded (scripts/get_data.py)")


def pytest_collection_modifyitems(config, items):
    from aisle.schema import data_root
    root = data_root()
    for item in items:
        for mark in item.iter_markers("needs_data"):
            missing = [n for n in mark.args if not (root / DATASETS[n]).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(reason=f"needs {', '.join(missing)} data: run scripts/get_data.py"))
