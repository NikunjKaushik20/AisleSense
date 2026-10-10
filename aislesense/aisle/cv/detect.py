"""YOLO11n person detector (person class only).

Each video source gets its own YOLO object, because ultralytics keeps tracker state on the
predictor; sharing one object across streams would mix their track IDs."""
from __future__ import annotations

from aisle.schema import default_model

DEFAULT_MODEL = default_model()


def load_model(path: str = DEFAULT_MODEL):
    from ultralytics import YOLO
    return YOLO(path)
