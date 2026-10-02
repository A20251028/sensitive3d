"""glTF / GLB reader and writer (via trimesh).

glTF is Y-up; parts are converted to the Z-up working frame on read and
back on write.  Node transforms are baked into the vertices.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..core.mesh import MeshFile, MeshPart, Texture

# glTF (Y-up) -> working frame (Z-up): (x, y, z) -> (x, -z, y) as row vectors: p_zup = p_yup @ _YUP_TO_ZUP
_YUP_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _material_image(material) -> Optional[Image.Image]:
    for attr in ("baseColorTexture", "image"):
        img = getattr(material, attr, None)
        if img is not None:
            return img
    return None


def read_gltf(path: str | Path, rel_path: Optional[str] = None) -> MeshFile:
    import trimesh

    path = Path(path)
    scene = trimesh.load(str(path), force="scene", process=False)
    parts: list[MeshPart] = []
    textures: list[Texture] = []
    tex_ids: dict[int, int] = {}
    for k, node in enumerate(scene.graph.nodes_geometry):
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name]
        if not hasattr(geom, "faces") or len(geom.faces) == 0:
            continue
        v = np.asarray(geom.vertices, dtype=np.float64)
        v = v @ transform[:3, :3].T + transform[:3, 3]
        uv = None
        tex = None
        visual = geom.visual
        if getattr(visual, "kind", None) == "texture" and visual.uv is not None and len(visual.uv) == len(v):
            uv = np.asarray(visual.uv, dtype=np.float64)
            img = _material_image(visual.material)
            if img is not None:
                key = id(img)
                if key not in tex_ids:
                    tex_ids[key] = len(textures)
                    textures.append(Texture(np.asarray(img.convert("RGB")).copy(), name=f"texture{len(textures)}", encoding="jpg"))
                tex = tex_ids[key]
        parts.append(
            MeshPart(
                vertices=v @ _YUP_TO_ZUP,
                faces=np.asarray(geom.faces, dtype=np.int64),
                uvs=uv,
                texture=tex,
                index=k,
                name=str(geom_name),
            )
        )
    return MeshFile(rel_path or path.name, "glb", parts, textures, {"source": str(path)})


def write_gltf(mesh: MeshFile, dst: str | Path, src: Optional[str | Path] = None) -> None:
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual.texture import TextureVisuals

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None and not mesh.dirty and Path(src).suffix.lower() == dst.suffix.lower():
        shutil.copyfile(src, dst)
        return
    pil_cache: dict[int, Image.Image] = {}
    scene = trimesh.Scene()
    for k, part in enumerate(mesh.parts):
        if len(part.faces) == 0:
            continue
        visual = None
        if part.uvs is not None and part.texture is not None:
            if part.texture not in pil_cache:
                pil_cache[part.texture] = Image.fromarray(mesh.textures[part.texture].image[..., :3])
            mat = PBRMaterial(baseColorTexture=pil_cache[part.texture], metallicFactor=0.0, roughnessFactor=1.0)
            visual = TextureVisuals(uv=part.uvs, material=mat)
        tm = trimesh.Trimesh(vertices=part.vertices @ _YUP_TO_ZUP.T, faces=part.faces, visual=visual, process=False)
        scene.add_geometry(tm, geom_name=part.name or f"part{k}")
    data = scene.export(file_type="glb")
    dst.write_bytes(data)
