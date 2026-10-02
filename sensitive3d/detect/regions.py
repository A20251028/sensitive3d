"""3D sensitive regions: oriented plate boxes lifted from 2D detections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

CATEGORY_LABELS = {
    "speed_limit": "限速标志",
    "height_limit": "限高标志",
    "weight_limit": "限重标志",
    "width_limit": "限宽标志",
    "prohibitory": "禁令标志(限速/限重等)",
    "road_name": "路牌/指路标志",
    "guide": "指路标志",
    "warning": "警告标志",
    "other_sign": "其他标志",
}


def plane_axes(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-plane axes: ``u`` horizontal (viewer's right), ``v`` as vertical as possible."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    u = np.cross(z, n)
    if np.linalg.norm(u) < 0.2:  # plate lying flat (road marking): use x as u
        u = np.cross(np.array([0.0, 1.0, 0.0]), n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v / np.linalg.norm(v)


def fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA plane: returns (centroid, unit normal, rms residual)."""
    c = np.median(points, axis=0)
    q = points - c
    _, s, vt = np.linalg.svd(q[:: max(1, len(q) // 20000)], full_matrices=False)
    n = vt[2]
    rms = float(s[2] / np.sqrt(max(len(q[:: max(1, len(q) // 20000)]), 1)))
    return c, n, rms


@dataclass
class SignRegion:
    center: np.ndarray
    normal: np.ndarray
    half_u: float
    half_v: float
    category: str = "other_sign"
    score: float = 0.0
    label: str = ""
    id: int = -1
    mount: str = "unknown"
    sources: list = field(default_factory=list)
    shape: str = "rect"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        n = np.asarray(self.normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)

    @property
    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        return plane_axes(self.normal)

    @property
    def size(self) -> float:
        return 2.0 * max(self.half_u, self.half_v)

    def local(self, points: np.ndarray) -> np.ndarray:
        u, v = self.axes
        d = np.asarray(points, dtype=np.float64) - self.center
        return np.stack([d @ u, d @ v, d @ self.normal], axis=-1)

    def contains(self, points: np.ndarray, margin: float = 0.0, front: float = 0.08, back: float = 0.15) -> np.ndarray:
        """Points inside the plate box (in-plane margin, ``front`` / ``back`` along the normal)."""
        q = self.local(points)
        return (
            (np.abs(q[:, 0]) <= self.half_u + margin)
            & (np.abs(q[:, 1]) <= self.half_v + margin)
            & (q[:, 2] <= front)
            & (q[:, 2] >= -back)
        )

    def corners(self, margin: float = 0.0, front: float = 0.08, back: float = 0.15) -> np.ndarray:
        u, v = self.axes
        out = []
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (front, -back):
                    out.append(self.center + a * (self.half_u + margin) * u + b * (self.half_v + margin) * v + c * self.normal)
        return np.array(out)

    def aabb(self, margin: float = 0.0, front: float = 0.08, back: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
        c = self.corners(margin, front, back)
        return c.min(axis=0), c.max(axis=0)

    def to_dict(self) -> dict:
        u, v = self.axes
        return {
            "id": self.id,
            "category": self.category,
            "category_label": CATEGORY_LABELS.get(self.category, self.category),
            "label": self.label,
            "score": round(float(self.score), 3),
            "center": [round(float(x), 4) for x in self.center],
            "normal": [round(float(x), 4) for x in self.normal],
            "axis_u": [round(float(x), 4) for x in u],
            "axis_v": [round(float(x), 4) for x in v],
            "width": round(2 * self.half_u, 3),
            "height": round(2 * self.half_v, 3),
            "shape": self.shape,
            "mount": self.mount,
        }

    @staticmethod
    def from_dict(d: dict) -> "SignRegion":
        return SignRegion(
            center=np.array(d["center"]),
            normal=np.array(d["normal"]),
            half_u=d["width"] / 2,
            half_v=d["height"] / 2,
            category=d.get("category", "other_sign"),
            score=d.get("score", 1.0),
            label=d.get("label", ""),
            id=d.get("id", -1),
            mount=d.get("mount", "unknown"),
            shape=d.get("shape", "rect"),
        )


def region_from_points(points: np.ndarray, view_dir: Optional[np.ndarray] = None, **kw) -> Optional[SignRegion]:
    """Fit an oriented plate region to surface points of one sign face."""
    if len(points) < 12:
        return None
    c, n, _ = fit_plane(points)
    if view_dir is not None and np.dot(n, view_dir) > 0:
        n = -n  # normal points towards the viewer
    u, v = plane_axes(n)
    d = points - c
    a, b = d @ u, d @ v
    a0, a1 = np.percentile(a, [1, 99])
    b0, b1 = np.percentile(b, [1, 99])
    center = c + (a0 + a1) / 2 * u + (b0 + b1) / 2 * v
    # put the centre on the median front surface
    center = center + np.median(d @ n) * n
    return SignRegion(center, n, (a1 - a0) / 2, (b1 - b0) / 2, **kw)


def overlap(a: SignRegion, b: SignRegion) -> bool:
    if np.dot(a.normal, b.normal) < 0.5:
        return False
    d = a.local(b.center[None])[0]
    return abs(d[0]) < a.half_u + 0.5 * b.half_u and abs(d[1]) < a.half_v + 0.5 * b.half_v and abs(d[2]) < 0.3


def merge_regions(regions: list[SignRegion]) -> list[SignRegion]:
    """Greedy merge of duplicates (same plate seen from several candidates / views)."""
    regions = sorted(regions, key=lambda r: -r.score)
    kept: list[SignRegion] = []
    for r in regions:
        dup = next((k for k in kept if overlap(k, r) or overlap(r, k)), None)
        if dup is None:
            kept.append(r)
            continue
        # grow the kept region to the union of both plate rectangles
        u, v = dup.axes
        pts = np.concatenate([dup.corners(front=0, back=0), r.corners(front=0, back=0)])
        q = dup.local(pts)
        a0, a1 = q[:, 0].min(), q[:, 0].max()
        b0, b1 = q[:, 1].min(), q[:, 1].max()
        dup.center = dup.center + (a0 + a1) / 2 * u + (b0 + b1) / 2 * v
        dup.half_u, dup.half_v = (a1 - a0) / 2, (b1 - b0) / 2
        dup.sources.extend(r.sources)
    for i, r in enumerate(kept):
        r.id = i
    return kept
