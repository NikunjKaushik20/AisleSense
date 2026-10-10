"""Face / head blur for any frame that is saved (F14). Default config saves no frames at all.

The detector already gives person boxes, so the head region is taken as the top `head_frac` of each person box
(it covers the face in oblique phone views and the head from above); it is pixelated, which a deblur cannot undo
the way it can a light Gaussian. No extra face model, so it also works on backs of heads and small faces."""
from __future__ import annotations

import numpy as np

HEAD_FRAC = 0.28
BLOCK_PX = 12


def head_boxes(xyxy, head_frac: float = HEAD_FRAC, pad: float = 0.1) -> np.ndarray:
    """Person boxes (N, 4) -> head boxes (N, 4): top head_frac of the box, widened by pad on each side."""
    b = np.asarray(xyxy, float).reshape(-1, 4)
    if not len(b):
        return b
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.stack([b[:, 0] - pad * w, b[:, 1] - pad * h * head_frac, b[:, 2] + pad * w, b[:, 1] + head_frac * h], 1)


def pixelate(frame: np.ndarray, box, block: int = BLOCK_PX) -> None:
    """In place: replace box with coarse blocks."""
    import cv2
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = frame[y0:y1, x0:x1]
    sw, sh = max(1, (x1 - x0) // block), max(1, (y1 - y0) // block)
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA)
    frame[y0:y1, x0:x1] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)


def blur_heads(frame: np.ndarray, person_xyxy, head_frac: float = HEAD_FRAC) -> np.ndarray:
    """Copy of frame with every head region pixelated."""
    out = frame.copy()
    for hb in head_boxes(person_xyxy, head_frac):
        pixelate(out, hb)
    return out


class Recorder:
    """Writes head-blurred frames to an mp4. Only created when a site or session sets `record: true`."""

    def __init__(self, path, fps: float):
        self.path, self.fps, self.w = str(path), float(fps), None
        self.frames = 0

    def write(self, frame, person_xyxy) -> None:
        import cv2
        if self.w is None:
            from pathlib import Path
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.w = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.w.write(blur_heads(frame, person_xyxy))
        self.frames += 1

    def close(self) -> None:
        if self.w is not None:
            self.w.release()
            self.w = None
