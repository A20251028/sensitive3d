"""Small CPU triangle rasterizer (numba).

Used for three things: rendering orthographic views of a mesh (detection and
repair views), mapping texels back to the triangles that own them (UV-space
rasterization), and computing coverage masks.

Pixel ``(row, col)`` has its centre at continuous coordinate
``(x, y) = (col + 0.5, row + 0.5)``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def _raster_into(xy, depth, faces, face_ok, near, far, depth_test, zbuf, fid, bary, pid, part_id):
    H, W = zbuf.shape
    eps = 1e-7
    for f in range(faces.shape[0]):
        if not face_ok[f]:
            continue
        i0 = faces[f, 0]
        i1 = faces[f, 1]
        i2 = faces[f, 2]
        x0 = xy[i0, 0]
        y0 = xy[i0, 1]
        x1 = xy[i1, 0]
        y1 = xy[i1, 1]
        x2 = xy[i2, 0]
        y2 = xy[i2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue
        inv = 1.0 / area
        cmin = max(int(math.ceil(min(x0, min(x1, x2)) - 0.5)), 0)
        cmax = min(int(math.floor(max(x0, max(x1, x2)) - 0.5)), W - 1)
        rmin = max(int(math.ceil(min(y0, min(y1, y2)) - 0.5)), 0)
        rmax = min(int(math.floor(max(y0, max(y1, y2)) - 0.5)), H - 1)
        if cmin > cmax or rmin > rmax:
            continue
        d0 = depth[i0]
        d1 = depth[i1]
        d2 = depth[i2]
        for r in range(rmin, rmax + 1):
            py = r + 0.5
            for c in range(cmin, cmax + 1):
                px = c + 0.5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * inv
                if w0 < -eps:
                    continue
                w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * inv
                if w1 < -eps:
                    continue
                w2 = 1.0 - w0 - w1
                if w2 < -eps:
                    continue
                z = w0 * d0 + w1 * d1 + w2 * d2
                if z < near or z > far:
                    continue
                if depth_test and z >= zbuf[r, c]:
                    continue
                zbuf[r, c] = z
                fid[r, c] = f
                pid[r, c] = part_id
                bary[r, c, 0] = w0
                bary[r, c, 1] = w1
                bary[r, c, 2] = w2


class RasterBuffers:
    """Shared z-buffer, face/part id and barycentric buffers."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self.zbuf = np.full((self.height, self.width), np.inf, dtype=np.float64)
        self.face = np.full((self.height, self.width), -1, dtype=np.int64)
        self.part = np.full((self.height, self.width), -1, dtype=np.int64)
        self.bary = np.zeros((self.height, self.width, 3), dtype=np.float64)

    @property
    def valid(self) -> np.ndarray:
        return self.face >= 0

    def draw(self, xy, depth, faces, face_ok=None, part_id=0, near=-np.inf, far=np.inf, depth_test=True):
        faces = np.ascontiguousarray(faces, dtype=np.int64)
        if face_ok is None:
            face_ok = np.ones(len(faces), dtype=np.bool_)
        _raster_into(
            np.ascontiguousarray(xy, dtype=np.float64),
            np.ascontiguousarray(depth, dtype=np.float64),
            faces,
            np.ascontiguousarray(face_ok, dtype=np.bool_),
            float(near),
            float(far),
            bool(depth_test),
            self.zbuf,
            self.face,
            self.bary,
            self.part,
            int(part_id),
        )


def rasterize_uv(uv_px: np.ndarray, faces: np.ndarray, width: int, height: int, face_ok=None) -> RasterBuffers:
    """Rasterize triangles in texture space; returns texel -> face ownership."""
    buf = RasterBuffers(width, height)
    buf.draw(uv_px, np.zeros(len(uv_px)), faces, face_ok=face_ok, depth_test=False)
    return buf
