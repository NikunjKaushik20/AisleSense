"""Live camera preview for the dashboard: newest frame, heads pixelated, person boxes and track ids drawn.

Rendered on request from the frame the worker already holds in memory and returned as a JPEG. Nothing is written
to disk, so the default "no frames stored" guarantee (F14) still holds; heads are blurred before anything leaves
the process."""
from __future__ import annotations

import numpy as np

from aisle.cv.blur import blur_heads

BOX_BGR = (143, 63, 43)     # --accent #2b3f8f
LABEL_BGR = (252, 251, 247)  # --panel


def render_jpeg(frame: np.ndarray, xyxy, ids, width: int = 640, quality: int = 72) -> bytes:
    import cv2
    img = blur_heads(frame, xyxy)
    boxes = np.asarray(xyxy, float).reshape(-1, 4)
    ids = list(ids) if ids is not None and len(ids) == len(boxes) else [None] * len(boxes)
    th = max(3, round(img.shape[1] / 320))
    fs = max(0.55, img.shape[1] / 1100)
    for (x0, y0, x1, y1), k in zip(boxes.astype(int), ids):
        cv2.rectangle(img, (x0, y0), (x1, y1), BOX_BGR, th)
        if k is not None:
            text = f"#{k}"
            (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            yb = max(y0, tht + 8)
            cv2.rectangle(img, (x0, yb - tht - 8), (x0 + tw + 8, yb), BOX_BGR, -1)
            cv2.putText(img, text, (x0 + 4, yb - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, LABEL_BGR, 2, cv2.LINE_AA)
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()
