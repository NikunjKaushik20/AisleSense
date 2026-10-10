"""4-point image->floor homography per camera (DLT, exact for 4 points)."""
from __future__ import annotations

import numpy as np


def fit(image_points, floor_points) -> np.ndarray:
    """Return 3x3 H mapping image px [u, v] to floor metres [x, y]. Needs >= 4 non-collinear points."""
    src = np.asarray(image_points, float)
    dst = np.asarray(floor_points, float)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("need >= 4 matching [u,v] and [x,y] points")
    # normalise for numerical stability
    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    Ts, Td = norm(src), norm(dst)
    sh = (Ts @ np.c_[src, np.ones(len(src))].T).T
    dh = (Td @ np.c_[dst, np.ones(len(dst))].T).T
    A = []
    for (u, v, _), (x, y, _) in zip(sh, dh):
        A.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        A.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply(H: np.ndarray, pts) -> np.ndarray:
    p = np.atleast_2d(np.asarray(pts, float))
    q = np.c_[p, np.ones(len(p))] @ np.asarray(H).T
    return q[:, :2] / q[:, 2:3]


def inverse(H: np.ndarray) -> np.ndarray:
    Hi = np.linalg.inv(H)
    return Hi / Hi[2, 2]


def foot_points(xyxy) -> np.ndarray:
    """Bottom-centre of each [x1, y1, x2, y2] box (where the person stands)."""
    b = np.atleast_2d(np.asarray(xyxy, float))
    if b.size == 0:
        return np.zeros((0, 2))
    return np.c_[(b[:, 0] + b[:, 2]) / 2.0, b[:, 3]]
