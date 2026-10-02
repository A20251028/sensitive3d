"""Textured orthographic rendering of mesh parts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .camera import OrthoCamera
from .mesh import MeshPart
from .raster import RasterBuffers
from .texture import sample_bilinear


@dataclass
class RenderItem:
    part: MeshPart
    image: Optional[np.ndarray] = None
    face_ok: Optional[np.ndarray] = None


@dataclass
class RenderResult:
    camera: OrthoCamera
    color: np.ndarray  # (H, W, 3) uint8
    valid: np.ndarray  # (H, W) bool
    part: np.ndarray  # (H, W) index into the rendered item list, -1 = empty
    face: np.ndarray  # (H, W) face index within the part, -1 = empty
    bary: np.ndarray  # (H, W, 3)
    depth: np.ndarray  # (H, W)
    position: np.ndarray  # (H, W, 3) world position, NaN when empty
    normal: np.ndarray  # (H, W, 3) geometric face normal, 0 when empty


def render(
    items: Sequence[RenderItem],
    camera: OrthoCamera,
    cull_backfaces: bool = False,
    background=(0, 0, 0),
    shade: bool = False,
) -> RenderResult:
    buf = RasterBuffers(camera.width, camera.height)
    projected = []
    for k, it in enumerate(items):
        part = it.part
        if len(part.faces) == 0:
            projected.append(None)
            continue
        xy, depth = camera.project(part.vertices)
        ok = np.ones(len(part.faces), dtype=bool) if it.face_ok is None else it.face_ok.astype(bool).copy()
        if cull_backfaces:
            ok &= part.face_normals() @ camera.forward < 0.0
        buf.draw(xy, depth, part.faces, ok, part_id=k, near=camera.near, far=camera.far)
        projected.append(xy)

    H, W = camera.height, camera.width
    color = np.empty((H, W, 3), dtype=np.uint8)
    color[:] = np.asarray(background, dtype=np.uint8)
    position = np.full((H, W, 3), np.nan, dtype=np.float64)
    normal = np.zeros((H, W, 3), dtype=np.float64)
    valid = buf.valid
    for k, it in enumerate(items):
        sel = valid & (buf.part == k)
        if not sel.any():
            continue
        part = it.part
        fids = buf.face[sel]
        b = buf.bary[sel]
        tri = part.faces[fids]
        position[sel] = np.einsum("nk,nkd->nd", b, part.vertices[tri])
        fn = part.face_normals()[fids]
        normal[sel] = fn
        if it.image is not None and part.uvs is not None:
            uv = np.einsum("nk,nkd->nd", b, part.uvs[tri])
            col = sample_bilinear(it.image, uv)[:, :3]
        else:
            col = np.full((len(fids), 3), 180.0)
        if shade:
            light = np.array([0.3, -0.4, 0.86])
            col = col * (0.55 + 0.45 * np.abs(fn @ light))[:, None]
        color[sel] = np.clip(col, 0, 255).astype(np.uint8)
    depth = np.where(valid, buf.zbuf, np.nan)
    return RenderResult(camera, color, valid, buf.part, buf.face, buf.bary, depth, position, normal)
