"""OSGB reading / patching through the native ``osgb_bridge`` helper."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from ..core.mesh import MeshFile, MeshPart, Texture, transform_directions, transform_points

_REPO_ROOT = Path(__file__).resolve().parents[2]


class BridgeError(RuntimeError):
    pass


def find_bridge() -> str:
    env = os.environ.get("S3D_OSGB_BRIDGE")
    candidates = [env] if env else []
    candidates += [
        str(_REPO_ROOT / "native" / "osgb_bridge" / "build" / "osgb_bridge"),
        str(_REPO_ROOT / "native" / "osgb_bridge" / "build" / "Release" / "osgb_bridge.exe"),
        shutil.which("osgb_bridge") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise BridgeError(
        "osgb_bridge not found. Build it with scripts/build_bridge.sh (needs OpenSceneGraph) "
        "or set S3D_OSGB_BRIDGE to its path."
    )


def bridge_available() -> bool:
    try:
        find_bridge()
        return True
    except BridgeError:
        return False


def _run(args: list[str]) -> str:
    proc = subprocess.run([find_bridge(), *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise BridgeError(f"osgb_bridge {args[0]} failed ({proc.returncode}): {proc.stderr.strip()}")
    return proc.stdout


def scan_osgb(paths: Iterable[str | Path], batch: int = 200) -> list[dict]:
    """Fast structural scan (bounds, LOD flags, children) without decoding textures."""
    paths = [str(p) for p in paths]
    out: list[dict] = []
    for i in range(0, len(paths), batch):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("\n".join(paths[i : i + batch]) + "\n")
            list_file = f.name
        try:
            text = _run(["info", list_file])
        finally:
            os.unlink(list_file)
        out.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return out


def read_osgb(path: str | Path, rel_path: Optional[str] = None) -> MeshFile:
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="s3d_exp_") as tmp:
        _run(["export", str(path), tmp])
        t = Path(tmp)
        man = json.loads((t / "manifest.json").read_text(encoding="utf-8"))
        textures = []
        for tex in man["textures"]:
            if tex["file"]:
                img = np.fromfile(t / tex["file"], dtype=np.uint8).reshape(tex["height"], tex["width"], tex["channels"])
            else:
                img = np.full((4, 4, 3), 128, np.uint8)
            enc = tex.get("encoding", "jpg")
            textures.append(Texture(img, name=tex.get("name", ""), encoding="png" if enc == "png" else "jpg"))
        parts = []
        for g in man["geometries"]:
            if not g["vertices"]:
                continue
            m = np.array(g["matrix"], dtype=np.float64).reshape(4, 4)
            v = np.fromfile(t / g["vertices"], dtype=np.float32).reshape(-1, 3).astype(np.float64)
            f = np.fromfile(t / g["triangles"], dtype=np.uint32).reshape(-1, 3).astype(np.int64)
            uv = np.fromfile(t / g["texcoords"], dtype=np.float32).reshape(-1, 2) if g["texcoords"] else None
            n = np.fromfile(t / g["normals"], dtype=np.float32).reshape(-1, 3) if g["normals"] else None
            c = np.fromfile(t / g["colors"], dtype=np.float32).reshape(-1, 4) if g["colors"] else None
            parts.append(
                MeshPart(
                    vertices=transform_points(v, m),
                    faces=f,
                    uvs=uv,
                    normals=transform_directions(n, m) if n is not None else None,
                    colors=c.astype(np.float64) if c is not None else None,
                    texture=g["texture"] if g["texture"] >= 0 else None,
                    has_finer=bool(g["has_finer"]),
                    index=int(g["index"]),
                    matrix=m,
                    name=g.get("name", ""),
                )
            )
        meta = {"lods": man["lods"], "children": sorted({c["file"] for l in man["lods"] for c in l["children"] if c["file"]})}
        return MeshFile(rel_path or path.name, "osgb", parts, textures, meta)


def encode_texture(tex: Texture) -> tuple[bytes, str]:
    img = tex.image
    pil = Image.fromarray(img if img.shape[2] in (3, 4) else img[..., :3])
    buf = io.BytesIO()
    if tex.encoding == "png" or img.shape[2] == 4:
        pil.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), "png"
    pil.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue(), "jpg"


def write_osgb(mesh: MeshFile, src: str | Path, dst: str | Path) -> None:
    """Write ``mesh`` to ``dst`` by patching ``src``; unchanged files are copied."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not mesh.dirty:
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return
    with tempfile.TemporaryDirectory(prefix="s3d_patch_") as tmp:
        t = Path(tmp)
        lines = []
        for part in mesh.parts:
            if not part.dirty:
                continue
            inv = np.linalg.inv(part.matrix)
            base = f"g{part.index}"
            transform_points(part.vertices, inv).astype(np.float32).tofile(t / f"{base}_v.f32")
            part.faces.astype(np.uint32).tofile(t / f"{base}_t.u32")
            uvf = nf = cf = "-"
            if part.uvs is not None:
                part.uvs.astype(np.float32).tofile(t / f"{base}_uv.f32")
                uvf = f"{base}_uv.f32"
            if part.normals is not None:
                transform_directions(part.normals, inv).astype(np.float32).tofile(t / f"{base}_n.f32")
                nf = f"{base}_n.f32"
            if part.colors is not None:
                part.colors.astype(np.float32).tofile(t / f"{base}_c.f32")
                cf = f"{base}_c.f32"
            lines.append(f"geometry {part.index} {base}_v.f32 {uvf} {nf} {cf} {base}_t.u32")
        for k, tex in enumerate(mesh.textures):
            if not tex.dirty:
                continue
            data, ext = encode_texture(tex)
            name = f"tex{k}.{ext}"
            (t / name).write_bytes(data)
            lines.append(f"texture {k} {name}")
        (t / "patch.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run(["patch", str(src), str(t), str(dst)])


def build_osgb(description: str, workdir: str | Path, dst: str | Path) -> None:
    """Build an OSGB file from a ``build.txt`` description (see osgb_bridge.cpp)."""
    workdir = Path(workdir)
    desc = workdir / "build.txt"
    desc.write_text(description, encoding="utf-8")
    _run(["build", str(desc), str(dst)])
