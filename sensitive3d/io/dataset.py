"""Dataset abstraction: a folder of OSGB tiles, or OBJ / glTF mesh files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.mesh import MeshFile
from . import gltf, obj, osgb

MESH_SUFFIXES = {".osgb": "osgb", ".obj": "obj", ".glb": "glb", ".gltf": "glb"}


@dataclass
class FileInfo:
    rel: str
    tile: str
    bmin: Optional[np.ndarray] = None
    bmax: Optional[np.ndarray] = None
    leaf_bmin: Optional[np.ndarray] = None
    leaf_bmax: Optional[np.ndarray] = None
    has_leaf: bool = True
    num_triangles: int = 0
    children: list[str] = field(default_factory=list)

    def intersects(self, lo: np.ndarray, hi: np.ndarray, leaf_only: bool = False) -> bool:
        a, b = (self.leaf_bmin, self.leaf_bmax) if leaf_only else (self.bmin, self.bmax)
        if a is None or b is None:
            return True
        return bool(np.all(a <= hi) and np.all(b >= lo))


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.is_file():
            self.single_file = self.root
            self.root = self.root.parent
        else:
            self.single_file = None
        self.kind = self._detect_kind()
        self.files: list[FileInfo] = []

    # -- discovery -----------------------------------------------------------
    def _candidates(self) -> list[Path]:
        if self.single_file is not None:
            return [self.single_file]
        return sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in MESH_SUFFIXES)

    def _detect_kind(self) -> str:
        kinds = {MESH_SUFFIXES[p.suffix.lower()] for p in self._candidates()}
        for k in ("osgb", "obj", "glb"):
            if k in kinds:
                return k
        raise ValueError(f"no .osgb / .obj / .glb / .gltf files found under {self.root}")

    def _tile_of(self, rel: Path) -> str:
        parts = rel.parts
        if self.kind == "osgb":
            if "Data" in parts[:-1]:
                i = parts.index("Data")
                if i + 1 < len(parts) - 1:
                    return parts[i + 1]
            return str(rel.parent) if len(parts) > 1 else rel.stem
        return rel.stem

    def scan(self) -> list[FileInfo]:
        paths = [p for p in self._candidates() if MESH_SUFFIXES[p.suffix.lower()] == self.kind]
        infos: list[FileInfo] = []
        if self.kind == "osgb":
            for p, rec in zip(paths, osgb.scan_osgb(paths)):
                rel = p.relative_to(self.root)
                info = FileInfo(rel.as_posix(), self._tile_of(rel))
                if rec.get("ok"):
                    geoms = [g for g in rec["geometries"] if "min" in g]
                    if geoms:
                        info.bmin = np.min([g["min"] for g in geoms], axis=0)
                        info.bmax = np.max([g["max"] for g in geoms], axis=0)
                    leaves = [g for g in geoms if not g["has_finer"]]
                    info.has_leaf = bool(leaves)
                    if leaves:
                        info.leaf_bmin = np.min([g["min"] for g in leaves], axis=0)
                        info.leaf_bmax = np.max([g["max"] for g in leaves], axis=0)
                    info.num_triangles = int(sum(g["num_triangles"] for g in rec["geometries"]))
                    info.children = list(rec.get("children", []))
                infos.append(info)
        else:
            for p in paths:
                rel = p.relative_to(self.root)
                info = FileInfo(rel.as_posix(), self._tile_of(rel))
                mesh = self.load(info.rel)
                info.bmin, info.bmax = mesh.bounds()
                info.leaf_bmin, info.leaf_bmax = info.bmin, info.bmax
                info.num_triangles = int(sum(len(pt.faces) for pt in mesh.parts))
                infos.append(info)
        self.files = infos
        return infos

    def tiles(self) -> dict[str, list[FileInfo]]:
        out: dict[str, list[FileInfo]] = {}
        for f in self.files:
            out.setdefault(f.tile, []).append(f)
        return out

    # -- io ------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        return self.root / rel

    def load(self, rel: str) -> MeshFile:
        p = self.path(rel)
        if self.kind == "osgb":
            return osgb.read_osgb(p, rel)
        if self.kind == "obj":
            return obj.read_obj(p, rel)
        return gltf.read_gltf(p, rel)

    def save(self, mesh: MeshFile, out_root: str | Path) -> None:
        dst = Path(out_root) / mesh.path
        src = self.path(mesh.path)
        if self.kind == "osgb":
            osgb.write_osgb(mesh, src, dst)
        elif self.kind == "obj":
            obj.write_obj(mesh, dst, src)
        else:
            gltf.write_gltf(mesh, dst, src)

    def copy_all(self, out_root: str | Path) -> None:
        """Copy the whole input (metadata.xml, untouched tiles ...) to the output."""
        out_root = Path(out_root)
        if self.single_file is not None:
            out_root.mkdir(parents=True, exist_ok=True)
            # sidecar files of a single mesh (mtl, textures) live next to it
            for p in self.root.iterdir():
                if p.is_file():
                    shutil.copy2(p, out_root / p.name)
            return
        shutil.copytree(self.root, out_root, dirs_exist_ok=True)

    def metadata_xml(self) -> Optional[str]:
        p = self.root / "metadata.xml"
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
