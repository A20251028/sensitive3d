"""Dataset abstraction: a folder of OSGB tiles, or OBJ / glTF mesh files.

:meth:`Dataset.scan` is strict: files that cannot be read, PagedLOD
references that do not resolve, reference cycles and missing / unreadable
textures are recorded in :attr:`Dataset.report` (a :class:`ScanReport`) and
make ``report.ok_for_repair`` False.  Failed files are never treated as
valid leaves.  OS sidecar files (``._*``, ``.DS_Store``, ``__MACOSX`` ...)
are never dataset files (see :mod:`sensitive3d.io.inputs`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import numpy as np

from ..core.mesh import MeshFile
from . import gltf, inputs, obj, osgb
from .lodgraph import LodGraph

MESH_SUFFIXES = {".osgb": "osgb", ".obj": "obj", ".glb": "glb", ".gltf": "glb"}
_EXAMPLES = 5


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
    ok: bool = True
    error: Optional[str] = None
    depth: Optional[int] = None  # shortest PagedLOD distance from a root file
    parents: list[str] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)  # image records of the bridge "info" scan
    texture_issues: list[str] = field(default_factory=list)
    decoded_bytes: int = 0  # RGBA bytes of the textures with a known size

    def intersects(self, lo: np.ndarray, hi: np.ndarray, leaf_only: bool = False) -> bool:
        a, b = (self.leaf_bmin, self.leaf_bmax) if leaf_only else (self.bmin, self.bmax)
        if a is None or b is None:
            return True
        return bool(np.all(a <= hi) and np.all(b >= lo))

    def to_dict(self) -> dict:
        return {
            "rel": self.rel,
            "tile": self.tile,
            "ok": self.ok,
            "error": self.error,
            "depth": self.depth,
            "has_leaf": self.has_leaf,
            "num_triangles": self.num_triangles,
            "children": list(self.children),
            "parents": list(self.parents),
            "texture_issues": list(self.texture_issues),
            "decoded_bytes": self.decoded_bytes,
        }


@dataclass
class ScanReport:
    """Coverage and integrity of a scanned dataset (see :meth:`Dataset.scan`)."""

    kind: str
    root: str
    files_total: int = 0
    files_ok: int = 0
    failed: list[dict] = field(default_factory=list)  # [{rel, error}]
    ignored: list[str] = field(default_factory=list)  # OS sidecar files left out
    skipped: list[dict] = field(default_factory=list)  # symlinks not followed [{rel, reason}]
    missing_refs: list[dict] = field(default_factory=list)  # [{parent, ref, resolved}]
    multi_parent: list[dict] = field(default_factory=list)  # [{rel, parents}]
    cycles: list[list[str]] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    roots: list[str] = field(default_factory=list)
    levels: dict[int, int] = field(default_factory=dict)  # depth -> number of files
    leaf_files: int = 0  # readable files holding finest-level geometry
    texture_issues: list[dict] = field(default_factory=list)  # [{rel, issues}]
    decoded_bytes_total: int = 0
    decoded_bytes_max: int = 0  # largest single file
    texture_info: bool = True  # False: the (old) bridge reported no image records, texture checks were not possible
    warnings: list[str] = field(default_factory=list)
    blocking_errors: list[str] = field(default_factory=list)
    files: list[FileInfo] = field(default_factory=list)

    @property
    def ok_for_repair(self) -> bool:
        return not self.blocking_errors

    def to_dict(self, include_files: bool = True) -> dict:
        d = {
            "kind": self.kind,
            "root": self.root,
            "ok_for_repair": self.ok_for_repair,
            "files_total": self.files_total,
            "files_ok": self.files_ok,
            "failed": list(self.failed),
            "ignored": list(self.ignored),
            "skipped": list(self.skipped),
            "missing_refs": list(self.missing_refs),
            "multi_parent": list(self.multi_parent),
            "cycles": [list(c) for c in self.cycles],
            "unreachable": list(self.unreachable),
            "roots": list(self.roots),
            "levels": dict(self.levels),
            "leaf_files": self.leaf_files,
            "texture_issues": list(self.texture_issues),
            "decoded_bytes_total": self.decoded_bytes_total,
            "decoded_bytes_max": self.decoded_bytes_max,
            "texture_info": self.texture_info,
            "warnings": list(self.warnings),
            "blocking_errors": list(self.blocking_errors),
        }
        if include_files:
            d["files"] = [f.to_dict() for f in self.files]
        return d


def _examples(items: list[str]) -> str:
    s = "; ".join(items[:_EXAMPLES])
    return s + (f" 等 {len(items)} 项" if len(items) > _EXAMPLES else "")


def _decoded_bytes(images: list[dict]) -> int:
    return int(sum(4 * im["width"] * im["height"] for im in images if im.get("width", -1) > 0 and im.get("height", -1) > 0))


def _osgb_texture_issues(rec: dict) -> list[str]:
    """Texture problems reported by the bridge (fields absent with an old bridge)."""
    issues = []
    images = rec.get("images") or []
    bad = [str(gi) for gi, g in enumerate(rec.get("geometries") or []) if g.get("has_uv") and g.get("texture_missing")]
    if bad:
        issues.append(f"{len(bad)} 个几何有 UV 但贴图缺失或无法读取 (几何 {_examples(bad)})")
    for im in images:
        if im.get("ok") is False:
            issues.append(f"贴图 {im.get('index', '?')} {im.get('name') or ''} 无法读取 ({im.get('encoding', '?')})".replace("  ", " "))
    return issues


def _obj_texture_issues(path: Path, mesh: MeshFile) -> list[str]:
    issues = []
    for lib in mesh.meta.get("mtllibs", []):
        mtl = path.parent / lib
        if not mtl.is_file():
            issues.append(f"材质库缺失: {lib}")
            continue
        for mat, tex in obj._parse_mtl(mtl).items():
            # read_obj resolves map_Kd against the OBJ's directory
            if tex and not (path.parent / tex).is_file():
                issues.append(f"材质 {mat} 的贴图缺失: {tex}")
    return issues


def _gltf_external_uris(path: Path) -> list[str]:
    """External (non data:) buffer / image URIs of a .gltf file."""
    if path.suffix.lower() != ".gltf":
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    uris = []
    for key in ("buffers", "images"):
        for item in doc.get(key) or []:
            uri = item.get("uri") if isinstance(item, dict) else None
            if uri and not uri.startswith("data:"):
                uris.append(unquote(uri))
    return uris


class Dataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if self.root.is_file():
            self.single_file = self.root
            self.root = self.root.parent
            if inputs.is_ignored(self.single_file.name):
                raise ValueError(f"{self.single_file.name} 是系统附属文件 (._* / .DS_Store 等), 不是网格文件")
        else:
            self.single_file = None
        self.ignored: list[str] = []
        self.skipped: list[dict] = []
        self._discover()
        self.kind = self._detect_kind()
        self.files: list[FileInfo] = []
        self.graph: Optional[LodGraph] = None
        self.report: Optional[ScanReport] = None
        self._texture_info = True

    # -- discovery -----------------------------------------------------------
    def _discover(self) -> None:
        """List candidate mesh files; sidecars go to ``self.ignored``."""
        self.ignored, self.skipped = [], []
        if self.single_file is not None:
            self.ignored = sorted(p.name for p in self.root.iterdir() if p.is_file() and inputs.is_ignored(p.name))
            self._files = [self.single_file]
            return
        files = inputs.iter_files(self.root, self.ignored, self.skipped)
        self._files = [p for p in files if p.suffix.lower() in MESH_SUFFIXES]

    def _candidates(self) -> list[Path]:
        return list(self._files)

    def _detect_kind(self) -> str:
        kinds = {MESH_SUFFIXES.get(p.suffix.lower()) for p in self._candidates()}
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

    def _osgb_records(self, paths: list[Path]) -> list[dict]:
        """One info record per path, matched by file name; gaps become failures."""
        strs = [str(p) for p in paths]
        try:
            recs = osgb.scan_osgb(paths)
        except (osgb.BridgeError, ValueError):  # crash, or non-JSON output
            osgb.find_bridge()  # no bridge at all: let that error through
            recs = []
            for s in strs:  # isolate the file that breaks the batch
                try:
                    recs.extend(osgb.scan_osgb([s]))
                except (osgb.BridgeError, ValueError) as e:
                    recs.append({"file": s, "ok": False, "error": f"osgb_bridge 异常: {e}"})
        def key(f) -> str:
            return os.path.normcase(os.path.abspath(str(f)))

        by_file = {str(r.get("file")): r for r in recs}
        by_key = {key(r.get("file")): r for r in recs if r.get("file")}
        same_count = len(recs) == len(strs)
        out = []
        for i, s in enumerate(strs):
            r = by_file.get(s) or by_key.get(key(s)) or (recs[i] if same_count else None)
            if r is None:
                r = {"file": s, "ok": False, "error": f"结构扫描未返回该文件的记录 (返回 {len(recs)} 条 / 期望 {len(strs)} 条)"}
            out.append(r)
        return out

    def scan(self) -> list[FileInfo]:
        self._discover()
        paths = [p for p in self._candidates() if MESH_SUFFIXES.get(p.suffix.lower()) == self.kind]
        infos: list[FileInfo] = []
        self._texture_info = self.kind != "osgb"
        if self.kind == "osgb":
            for p, rec in zip(paths, self._osgb_records(paths)):
                rel = p.relative_to(self.root)
                info = FileInfo(rel.as_posix(), self._tile_of(rel))
                if rec.get("ok"):
                    geoms = [g for g in rec.get("geometries", []) if "min" in g]
                    if geoms:
                        info.bmin = np.min([g["min"] for g in geoms], axis=0)
                        info.bmax = np.max([g["max"] for g in geoms], axis=0)
                    leaves = [g for g in geoms if not g["has_finer"]]
                    info.has_leaf = bool(leaves)
                    if leaves:
                        info.leaf_bmin = np.min([g["min"] for g in leaves], axis=0)
                        info.leaf_bmax = np.max([g["max"] for g in leaves], axis=0)
                    info.num_triangles = int(sum(g.get("num_triangles", 0) for g in rec.get("geometries", [])))
                    info.children = list(rec.get("children", []))
                    info.images = list(rec.get("images") or [])
                    self._texture_info = self._texture_info or "images" in rec
                    info.texture_issues = _osgb_texture_issues(rec)
                    info.decoded_bytes = _decoded_bytes(info.images)
                else:
                    self._fail(info, rec.get("error") or "osgb_bridge 无法读取该文件: 文件损坏或格式不支持")
                infos.append(info)
        else:
            for p in paths:
                rel = p.relative_to(self.root)
                info = FileInfo(rel.as_posix(), self._tile_of(rel))
                try:
                    mesh = self.load(info.rel)
                    info.bmin, info.bmax = mesh.bounds()
                    info.leaf_bmin, info.leaf_bmax = info.bmin, info.bmax
                    info.num_triangles = int(sum(len(pt.faces) for pt in mesh.parts))
                    info.decoded_bytes = int(sum(4 * t.image.shape[0] * t.image.shape[1] for t in mesh.textures))
                    if self.kind == "obj":
                        info.texture_issues = _obj_texture_issues(p, mesh)
                    else:
                        info.texture_issues = [f"外部文件缺失: {u}" for u in _gltf_external_uris(p) if not (p.parent / u).is_file()]
                    info.has_leaf = info.num_triangles > 0
                except Exception as e:  # any reader error makes the file unusable, not the scan
                    self._fail(info, f"{type(e).__name__}: {e}")
                infos.append(info)
        self.files = infos
        self.graph = LodGraph({f.rel: f.children for f in infos if f.ok}, existing=[f.rel for f in infos])
        for f in infos:
            f.depth = self.graph.depth.get(f.rel)
            f.parents = self.graph.parents.get(f.rel, [])
        self.report = self._build_report()
        return infos

    @staticmethod
    def _fail(info: FileInfo, error: str) -> None:
        info.ok = False
        info.error = error
        info.has_leaf = False
        info.bmin = info.bmax = info.leaf_bmin = info.leaf_bmax = None

    def _build_report(self) -> ScanReport:
        g = self.graph
        files = self.files
        rep = ScanReport(self.kind, str(self.single_file or self.root), files=files)
        rep.files_total = len(files)
        rep.files_ok = sum(1 for f in files if f.ok)
        rep.failed = [{"rel": f.rel, "error": f.error} for f in files if not f.ok]
        rep.ignored = list(self.ignored)
        rep.skipped = list(self.skipped)
        rep.multi_parent = list(g.multi_parent)
        rep.cycles = [list(c) for c in g.cycles]
        rep.unreachable = list(g.unreachable)
        rep.roots = list(g.roots)
        rep.levels = {d: len(v) for d, v in g.levels().items()}
        rep.leaf_files = sum(1 for f in files if f.ok and f.has_leaf)
        rep.texture_issues = [{"rel": f.rel, "issues": list(f.texture_issues)} for f in files if f.texture_issues]
        rep.decoded_bytes_total = int(sum(f.decoded_bytes for f in files))
        rep.decoded_bytes_max = int(max((f.decoded_bytes for f in files), default=0))
        rep.texture_info = self._texture_info or rep.files_ok == 0
        rep.warnings = list(g.warnings)

        missing = list(g.missing)
        if self.single_file is not None:
            # a single picked file: children on disk were just not part of the scan
            outside = [m for m in missing if m["resolved"] and (self.root / m["resolved"]).is_file()]
            missing = [m for m in missing if m not in outside]
            if outside:
                rep.warnings.append(f"单文件模式: {len(outside)} 个子文件存在但未扫描 (例如 {outside[0]['resolved']})")
        rep.missing_refs = missing

        if not rep.texture_info:
            rep.warnings.append("osgb_bridge 未报告贴图信息 (旧版本): 未能检查贴图缺失, 解码内存未知, 请更新并重新编译 bridge")
        if rep.ignored:
            rep.warnings.append(f"忽略 {len(rep.ignored)} 个系统附属文件 (._* / .DS_Store / __MACOSX 等), 原始目录中未删除")
        if rep.skipped:
            rep.warnings.append(f"跳过 {len(rep.skipped)} 个符号链接: {_examples([s['rel'] for s in rep.skipped])}")
        if rep.multi_parent:
            rep.warnings.append(f"{len(rep.multi_parent)} 个文件被多个父文件引用 (已去重, 深度取最短路径)")
        if rep.unreachable:
            rep.warnings.append(f"{len(rep.unreachable)} 个文件无法从任何根文件到达: {_examples(rep.unreachable)}")
        if self.kind == "osgb":
            tile_of = {f.rel: f.tile for f in files}
            per_tile: dict[str, int] = {}
            for r in rep.roots:
                per_tile[tile_of[r]] = per_tile.get(tile_of[r], 0) + 1
            multi = sorted(t for t, n in per_tile.items() if n > 1)
            if multi:
                rep.warnings.append(f"{len(multi)} 个瓦片有多个未被引用的根文件 (可能是孤立文件): {_examples(multi)}")

        err = rep.blocking_errors
        if not files:
            err.append("没有可处理的网格文件")
        elif rep.files_ok == 0:
            err.append("所有网格文件都无法读取")
        if rep.failed:
            err.append(f"{len(rep.failed)} 个文件读取失败: " + _examples([f"{x['rel']} ({x['error']})" for x in rep.failed]))
        if rep.missing_refs:
            err.append(f"{len(rep.missing_refs)} 处子文件引用缺失: " + _examples([f"{m['parent']} -> {m['ref']}" for m in rep.missing_refs]))
        if rep.cycles:
            err.append(f"{len(rep.cycles)} 处循环引用: {_examples([' -> '.join(c + c[:1]) for c in rep.cycles])}")
        if rep.texture_issues:
            err.append(f"{len(rep.texture_issues)} 个文件贴图缺失或损坏: " + _examples([f"{t['rel']}: {t['issues'][0]}" for t in rep.texture_issues]))
        return rep

    def tiles(self) -> dict[str, list[FileInfo]]:
        """Readable files grouped by tile (failed files are left out)."""
        out: dict[str, list[FileInfo]] = {}
        for f in self.files:
            if f.ok:
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

    def _single_file_sidecars(self) -> list[str]:
        """The picked file, its same-directory siblings and referenced mtl / textures / buffers."""
        root_real = self.root.resolve()
        rels = set()
        for p in self.root.iterdir():
            if p.is_file() and not inputs.is_ignored(p.name) and inputs.is_inside(p, root_real):
                rels.add(p.name)
        sf = self.single_file
        refs: list[str] = []
        if sf.suffix.lower() == ".obj":
            for line in sf.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("mtllib"):
                    lib = line[6:].strip()
                    refs.append(lib)
                    mtl = self.root / lib
                    if mtl.is_file():
                        refs += [str(Path(lib).parent / t) for t in obj._parse_mtl(mtl).values() if t]
        else:
            refs = _gltf_external_uris(sf)
        for ref in refs:
            r = inputs.safe_rel(os.path.normpath(ref).replace("\\", "/"))
            if r is not None and (self.root / r).is_file() and inputs.is_inside(self.root / r, root_real):
                rels.add(r.as_posix())
        return sorted(rels)

    def copy_all(self, out_root: str | Path) -> dict:
        """Copy the whole input (metadata.xml, untouched tiles ...) to the output.

        OS sidecar files are not copied; the source is never modified.
        Returns the manifest of the copied files (see :func:`inputs.copy_tree`).
        """
        out_root = Path(out_root)
        if self.single_file is not None:
            out_root.mkdir(parents=True, exist_ok=True)
            return inputs.copy_files(self.root, self._single_file_sidecars(), out_root)
        return inputs.copy_tree(self.root, out_root)

    def metadata_xml(self) -> Optional[str]:
        p = self.root / "metadata.xml"
        return p.read_text(encoding="utf-8", errors="replace") if p.is_file() else None
