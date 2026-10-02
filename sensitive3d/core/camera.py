"""Orthographic camera used for detection and repair views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-20)


@dataclass
class OrthoCamera:
    """Orthographic view.

    ``forward`` is the viewing direction.  Image x grows along ``right`` and
    image y grows along ``-up``.  ``near``/``far`` bound the depth (measured
    along ``forward`` from ``center``) of what gets rendered.
    """

    center: np.ndarray
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    pixel_size: float
    width: int
    height: int
    near: float = -np.inf
    far: float = np.inf

    @staticmethod
    def looking(
        target,
        direction,
        extent_w: float,
        extent_h: float,
        pixel_size: float,
        up_hint=(0.0, 0.0, 1.0),
        near: float = -np.inf,
        far: float = np.inf,
        max_pixels: int = 4096,
    ) -> "OrthoCamera":
        fwd = _normalize(direction)
        up_hint = _normalize(up_hint)
        if abs(float(np.dot(fwd, up_hint))) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0]) if abs(fwd[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        right = _normalize(np.cross(fwd, up_hint))
        up = _normalize(np.cross(right, fwd))
        pixel_size = float(pixel_size)
        largest = max(extent_w, extent_h) / pixel_size
        if largest > max_pixels:
            pixel_size *= largest / max_pixels
        w = max(int(np.ceil(extent_w / pixel_size)), 1)
        h = max(int(np.ceil(extent_h / pixel_size)), 1)
        return OrthoCamera(np.asarray(target, dtype=np.float64), right, up, fwd, pixel_size, w, h, near, far)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        d = np.asarray(points, dtype=np.float64) - self.center
        x = d @ self.right / self.pixel_size + self.width / 2.0
        y = -(d @ self.up) / self.pixel_size + self.height / 2.0
        return np.stack([x, y], axis=-1), d @ self.forward

    def unproject(self, xy: np.ndarray, depth: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        a = (xy[..., 0] - self.width / 2.0) * self.pixel_size
        b = -(xy[..., 1] - self.height / 2.0) * self.pixel_size
        return (
            self.center
            + a[..., None] * self.right
            + b[..., None] * self.up
            + np.asarray(depth, dtype=np.float64)[..., None] * self.forward
        )

    def pixel_centers(self) -> np.ndarray:
        ys, xs = np.mgrid[0 : self.height, 0 : self.width]
        return np.stack([xs + 0.5, ys + 0.5], axis=-1).astype(np.float64)

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "right": self.right.tolist(),
            "up": self.up.tolist(),
            "forward": self.forward.tolist(),
            "pixel_size": self.pixel_size,
            "width": self.width,
            "height": self.height,
            "near": None if not np.isfinite(self.near) else self.near,
            "far": None if not np.isfinite(self.far) else self.far,
        }
