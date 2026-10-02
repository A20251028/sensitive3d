"""Before / after previews: PNG close-ups and small GLB crops for the web viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

from .core.camera import OrthoCamera
from .core.mesh import MeshFile, MeshPart, Texture
from .core.render import RenderItem, render


def save_png(img: np.ndarray, path: str | Path, max_side: int = 512) -> None:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (max(int(w * s), 1), max(int(h * s), 1)), interpolation=cv2.INTER_AREA)
    Image.fromarray(img).save(path)


def closeup(items: Sequence[RenderItem], camera: OrthoCamera) -> np.ndarray:
    return render(items, camera, background=(235, 238, 242)).color


def context_camera(center: np.ndarray, normal: np.ndarray, size: float, ground_z: Optional[float] = None) -> OrthoCamera:
    """A slightly oblique view showing the sign and what is around / below it."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    d = -n + np.array([0, 0, -0.35])
    d /= np.linalg.norm(d)
    span = max(size * 2.5, 3.0)
    target = np.array(center, float)
    if ground_z is not None:
        target[2] = (target[2] + ground_z) / 2
        span = max(span, (center[2] - ground_z) * 1.6)
    return OrthoCamera.looking(target, d, span, span, span / 480, near=-span, far=span)


def crop_parts(mesh_parts: Sequence[tuple[MeshPart, Optional[Texture]]], lo: np.ndarray, hi: np.ndarray, max_tex: int = 1024) -> MeshFile:
    """Faces with a vertex inside [lo, hi], with downscaled textures, as one MeshFile."""
    parts, textures, tex_ids = [], [], {}
    for part, tex in mesh_parts:
        if len(part.faces) == 0:
            continue
        v = part.vertices[part.faces]
        inside = np.any(np.all((v >= lo) & (v <= hi), axis=2), axis=1)
        if not inside.any():
            continue
        f = part.faces[inside]
        vids, inv = np.unique(f, return_inverse=True)
        tid = None
        if tex is not None and part.uvs is not None:
            if id(tex) not in tex_ids:
                img = tex.image[..., :3]
                s = min(1.0, max_tex / max(img.shape[:2]))
                if s < 1.0:
                    img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
                tex_ids[id(tex)] = len(textures)
                textures.append(Texture(np.ascontiguousarray(img), name=f"t{len(textures)}.jpg"))
            tid = tex_ids[id(tex)]
        parts.append(
            MeshPart(
                part.vertices[vids],
                inv.reshape(-1, 3),
                uvs=part.uvs[vids] if part.uvs is not None else None,
                texture=tid,
                index=len(parts),
                name=f"p{len(parts)}",
            )
        )
    return MeshFile("preview", "glb", parts, textures)


def write_glb(mesh: MeshFile, path: str | Path, origin: Optional[np.ndarray] = None) -> None:
    """GLB for three.js; vertices are shifted by ``origin`` to keep float precision."""
    from .io.gltf import write_gltf

    if origin is not None:
        for p in mesh.parts:
            p.vertices = p.vertices - origin
    write_gltf(mesh, path)
