"""Texture sampling and texel <-> surface mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

from .mesh import MeshPart, Texture
from .raster import rasterize_uv


def sample_bilinear_px(image: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample at continuous pixel coords (centres at +0.5). Returns float (N, C)."""
    img = image if image.ndim == 3 else image[..., None]
    H, W = img.shape[:2]
    x = np.clip(np.asarray(xy[..., 0], dtype=np.float64) - 0.5, 0, W - 1)
    y = np.clip(np.asarray(xy[..., 1], dtype=np.float64) - 0.5, 0, H - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = img[y0, x0].astype(np.float64)
    b = img[y0, x1].astype(np.float64)
    c = img[y1, x0].astype(np.float64)
    d = img[y1, x1].astype(np.float64)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def sample_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    H, W = image.shape[:2]
    uv = np.asarray(uv, dtype=np.float64)
    xy = np.stack([uv[..., 0] * W, (1.0 - uv[..., 1]) * H], axis=-1)
    return sample_bilinear_px(image, xy)


@dataclass
class TexelMap:
    """Texels of one texture owned by (a subset of) the faces of one part."""

    rows: np.ndarray
    cols: np.ndarray
    faces: np.ndarray
    positions: np.ndarray  # world positions of the texel centres

    def __len__(self) -> int:
        return len(self.rows)


def texel_map(part: MeshPart, texture: Texture, face_ok: Optional[np.ndarray] = None) -> TexelMap:
    """Rasterize ``part``'s faces in UV space and lift every covered texel to 3D."""
    empty = TexelMap(np.zeros(0, int), np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3)))
    if part.uvs is None or len(part.faces) == 0:
        return empty
    if face_ok is not None and not np.any(face_ok):
        return empty
    uv_px = texture.uv_to_px(part.uvs)
    buf = rasterize_uv(uv_px, part.faces, texture.width, texture.height, face_ok=face_ok)
    rows, cols = np.nonzero(buf.valid)
    fids = buf.face[rows, cols]
    b = buf.bary[rows, cols]
    pos = np.einsum("nk,nkd->nd", b, part.vertices[part.faces[fids]])
    return TexelMap(rows, cols, fids, pos)


def coverage_mask(parts: list[MeshPart], texture: Texture, texture_index: int) -> np.ndarray:
    """Texels covered by any face of the given parts that use ``texture_index``."""
    mask = np.zeros((texture.height, texture.width), dtype=bool)
    for p in parts:
        if p.texture != texture_index or p.uvs is None or len(p.faces) == 0:
            continue
        buf = rasterize_uv(texture.uv_to_px(p.uvs), p.faces, texture.width, texture.height)
        mask |= buf.valid
    return mask


def texel_size(part: MeshPart, texture: Texture, face_ok: Optional[np.ndarray] = None) -> float:
    """Median world size (m) of one texel over the selected faces."""
    if part.uvs is None or len(part.faces) == 0:
        return 0.01
    f = part.faces if face_ok is None else part.faces[face_ok]
    if len(f) == 0:
        return 0.01
    v = part.vertices[f]
    a3 = 0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1)
    t = texture.uv_to_px(part.uvs)[f]
    a2 = 0.5 * np.abs(
        (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1]) - (t[:, 2, 0] - t[:, 0, 0]) * (t[:, 1, 1] - t[:, 0, 1])
    )
    ok = (a2 > 1e-6) & (a3 > 1e-12)
    if not ok.any():
        return 0.01
    return float(np.median(np.sqrt(a3[ok] / a2[ok])))


def fill_from_nearest(image: np.ndarray, known: np.ndarray, target: np.ndarray) -> None:
    """In place: give every ``target`` texel the colour of the nearest ``known`` texel."""
    if not target.any() or not known.any():
        return
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    image[target] = image[iy[target], ix[target]]
