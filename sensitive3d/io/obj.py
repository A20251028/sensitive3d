"""Wavefront OBJ (+MTL +textures) reader and writer.

One :class:`MeshPart` is created per material; vertices are split so that
every (position, uv, normal) combination becomes one per-vertex record.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..core.mesh import MeshFile, MeshPart, Texture


def _parse_mtl(path: Path) -> dict[str, Optional[str]]:
    mats: dict[str, Optional[str]] = {}
    cur = None
    if not path.is_file():
        return mats
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip().split()
        if not s:
            continue
        if s[0] == "newmtl":
            cur = " ".join(s[1:])
            mats[cur] = None
        elif s[0] == "map_Kd" and cur is not None:
            mats[cur] = s[-1]
    return mats


def read_obj(path: str | Path, rel_path: Optional[str] = None) -> MeshFile:
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pos, uvs, nrm = [], [], []
    groups: dict[str, list[list[tuple[int, int, int]]]] = {}
    material_files: dict[str, Optional[str]] = {}
    mtl_libs = []
    cur = "default"
    for line in text:
        if not line or line[0] == "#":
            continue
        s = line.split()
        if not s:
            continue
        tag = s[0]
        if tag == "v":
            pos.append((float(s[1]), float(s[2]), float(s[3])))
        elif tag == "vt":
            uvs.append((float(s[1]), float(s[2]) if len(s) > 2 else 0.0))
        elif tag == "vn":
            nrm.append((float(s[1]), float(s[2]), float(s[3])))
        elif tag == "f":
            poly = []
            for tok in s[1:]:
                p = tok.split("/")
                vi = int(p[0])
                ti = int(p[1]) if len(p) > 1 and p[1] else 0
                ni = int(p[2]) if len(p) > 2 and p[2] else 0
                vi = vi - 1 if vi > 0 else len(pos) + vi
                ti = ti - 1 if ti > 0 else (len(uvs) + ti if ti < 0 else -1)
                ni = ni - 1 if ni > 0 else (len(nrm) + ni if ni < 0 else -1)
                poly.append((vi, ti, ni))
            groups.setdefault(cur, []).append(poly)
        elif tag == "usemtl":
            cur = " ".join(s[1:])
        elif tag == "mtllib":
            mtl_libs.append(" ".join(s[1:]))
    for lib in mtl_libs:
        material_files.update(_parse_mtl(path.parent / lib))

    P = np.array(pos, dtype=np.float64).reshape(-1, 3)
    T = np.array(uvs, dtype=np.float64).reshape(-1, 2)
    N = np.array(nrm, dtype=np.float64).reshape(-1, 3)
    textures: list[Texture] = []
    tex_index: dict[str, int] = {}
    parts: list[MeshPart] = []
    for gi, (mat, polys) in enumerate(groups.items()):
        tris = []
        for poly in polys:
            for k in range(1, len(poly) - 1):
                tris.append((poly[0], poly[k], poly[k + 1]))
        if not tris:
            continue
        keys = np.array(tris, dtype=np.int64).reshape(-1, 3)  # corners x (v, t, n)
        uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        has_uv = len(T) > 0 and (uniq[:, 1] >= 0).all()
        has_n = len(N) > 0 and (uniq[:, 2] >= 0).all()
        tex_id = None
        tex_file = material_files.get(mat)
        if tex_file:
            if tex_file not in tex_index:
                tp = path.parent / tex_file
                if tp.is_file():
                    img = np.asarray(Image.open(tp).convert("RGB"))
                    tex_index[tex_file] = len(textures)
                    enc = "png" if tp.suffix.lower() == ".png" else "jpg"
                    textures.append(Texture(img.copy(), name=tex_file, encoding=enc))
            tex_id = tex_index.get(tex_file)
        parts.append(
            MeshPart(
                vertices=P[uniq[:, 0]],
                faces=faces,
                uvs=T[uniq[:, 1]] if has_uv else None,
                normals=N[uniq[:, 2]] if has_n else None,
                texture=tex_id,
                index=gi,
                name=mat,
            )
        )
    return MeshFile(rel_path or path.name, "obj", parts, textures, {"source": str(path), "mtllibs": mtl_libs})


def _save_texture(t: Texture, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(t.image[..., :3])
    if out.suffix.lower() == ".png":
        pil.save(out)
    else:
        pil.save(out, quality=95, subsampling=0)


def write_obj(mesh: MeshFile, dst: str | Path, src: Optional[str | Path] = None) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    geometry_dirty = any(p.dirty for p in mesh.parts)
    if src is not None and not geometry_dirty:
        # geometry untouched: .obj / .mtl stay byte-identical, only changed textures are re-encoded
        src = Path(src)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        for lib in mesh.meta.get("mtllibs", []):
            if (src.parent / lib).is_file() and (src.parent / lib).resolve() != (dst.parent / lib).resolve():
                (dst.parent / lib).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src.parent / lib, dst.parent / lib)
        for t in mesh.textures:
            if not t.name:
                continue
            if t.dirty:
                _save_texture(t, dst.parent / t.name)
            elif (src.parent / t.name).is_file() and (src.parent / t.name).resolve() != (dst.parent / t.name).resolve():
                (dst.parent / t.name).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src.parent / t.name, dst.parent / t.name)
        return
    mtl_name = dst.stem + ".mtl"
    tex_names = []
    for k, t in enumerate(mesh.textures):
        name = t.name or f"{dst.stem}_tex{k}.{'png' if t.encoding == 'png' else 'jpg'}"
        tex_names.append(name)
        _save_texture(t, dst.parent / name)
    with open(dst.parent / mtl_name, "w", encoding="utf-8") as f:
        for k, part in enumerate(mesh.parts):
            f.write(f"newmtl {part.name or f'mat{k}'}\nKa 1 1 1\nKd 1 1 1\nillum 1\n")
            if part.texture is not None:
                f.write(f"map_Kd {tex_names[part.texture]}\n")
            f.write("\n")
    lines = [f"mtllib {mtl_name}\n"]
    vbase = 0
    for k, part in enumerate(mesh.parts):
        if len(part.faces) == 0:
            continue
        lines.append(f"o part{k}\nusemtl {part.name or f'mat{k}'}\n")
        lines.append("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in part.vertices))
        has_uv = part.uvs is not None
        has_n = part.normals is not None
        if has_uv:
            lines.append("".join(f"vt {u:.7f} {v:.7f}\n" for u, v in part.uvs))
        if has_n:
            lines.append("".join(f"vn {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in part.normals))
        f1 = part.faces + vbase + 1
        if has_uv and has_n:
            lines.append("".join(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n" for a, b, c in f1))
        elif has_uv:
            lines.append("".join(f"f {a}/{a} {b}/{b} {c}/{c}\n" for a, b, c in f1))
        elif has_n:
            lines.append("".join(f"f {a}//{a} {b}//{b} {c}//{c}\n" for a, b, c in f1))
        else:
            lines.append("".join(f"f {a} {b} {c}\n" for a, b, c in f1))
        vbase += len(part.vertices)
    dst.write_text("".join(lines), encoding="utf-8")
