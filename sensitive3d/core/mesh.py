"""In-memory mesh model shared by every reader, writer and algorithm.

Conventions
-----------
* All algorithms work in a right-handed, Z-up, metric world frame.  Readers
  convert their native frame into it (OSGB/OBJ are already Z-up, glTF is
  Y-up) and writers convert back.
* Vertex attributes are per vertex (OSG style).  A mesh with UV seams simply
  has duplicated positions.
* Textures are ``uint8`` arrays of shape ``(H, W, C)`` with row 0 at the top,
  i.e. texture coordinate ``(u, v)`` maps to pixel ``(u * W, (1 - v) * H)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Texture:
    image: np.ndarray
    name: str = ""
    encoding: str = "jpg"
    dirty: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def uv_to_px(self, uv: np.ndarray) -> np.ndarray:
        """Continuous pixel coordinates (x right, y down, pixel centres at +0.5)."""
        uv = np.asarray(uv, dtype=np.float64)
        return np.stack([uv[..., 0] * self.width, (1.0 - uv[..., 1]) * self.height], axis=-1)

    def px_to_uv(self, px: np.ndarray) -> np.ndarray:
        px = np.asarray(px, dtype=np.float64)
        return np.stack([px[..., 0] / self.width, 1.0 - px[..., 1] / self.height], axis=-1)


@dataclass
class MeshPart:
    vertices: np.ndarray
    faces: np.ndarray
    uvs: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    texture: Optional[int] = None
    has_finer: bool = False
    index: int = 0
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4))
    name: str = ""
    dirty: bool = False

    def __post_init__(self) -> None:
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.ascontiguousarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.uvs is not None:
            self.uvs = np.ascontiguousarray(self.uvs, dtype=np.float64).reshape(-1, 2)

    @property
    def is_leaf(self) -> bool:
        return not self.has_finer

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        used = self.vertices[np.unique(self.faces)] if len(self.faces) else self.vertices
        return used.min(axis=0), used.max(axis=0)

    def face_normals(self) -> np.ndarray:
        return face_normals(self.vertices, self.faces)

    def face_areas(self) -> np.ndarray:
        v = self.vertices[self.faces]
        return 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)

    def centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    def recompute_normals(self) -> None:
        self.normals = vertex_normals(self.vertices, self.faces)


@dataclass
class MeshFile:
    """One mesh file of a dataset (an .osgb file, an .obj, a .glb ...)."""

    path: str
    format: str
    parts: list[MeshPart]
    textures: list[Texture]
    meta: dict = field(default_factory=dict)

    @property
    def dirty(self) -> bool:
        return any(p.dirty for p in self.parts) or any(t.dirty for t in self.textures)

    def leaf_parts(self) -> list[MeshPart]:
        return [p for p in self.parts if p.is_leaf]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bs = [p.bounds() for p in self.parts if len(p.faces)]
        if not bs:
            return np.zeros(3), np.zeros(3)
        return np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0)

    def texture_of(self, part: MeshPart) -> Optional[Texture]:
        if part.texture is None or part.texture < 0 or part.texture >= len(self.textures):
            return None
        return self.textures[part.texture]


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    return n / np.maximum(norm, 1e-20)


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v = vertices[faces]
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])  # area weighted
    out = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(out, faces[:, k], n)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / np.maximum(norm, 1e-20)
    out[norm[:, 0] < 1e-20] = (0.0, 0.0, 1.0)
    return out


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 4x4 matrix in OSG row-vector convention (p' = [p, 1] @ M)."""
    p = np.asarray(points, dtype=np.float64)
    return p @ matrix[:3, :3] + matrix[3, :3]


def transform_directions(dirs: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    d = np.asarray(dirs, dtype=np.float64) @ matrix[:3, :3]
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-20)
