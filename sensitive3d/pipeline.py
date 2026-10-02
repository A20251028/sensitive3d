"""End-to-end processing: scan -> detect -> repair every LOD -> write -> report."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .core.mesh import MeshFile
from .core.render import RenderItem
from .core.texture import texel_size
from .detect.detector import DetectionConfig, SignDetector
from .detect.regions import CATEGORY_LABELS, SignRegion, merge_regions
from .io.dataset import Dataset, FileInfo
from .preview import closeup, context_camera, crop_parts, save_png, write_glb
from .repair.geometry import GeometryConfig, RegionGeometry, analyse_region, remove_objects
from .repair.texture import TextureConfig, build_views, paint_targets, part_targets, wipe_freed_texels

ProgressFn = Callable[[float, str], None]


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    remove_geometry: bool = True
    previews: bool = True
    verify: bool = True
    regions: Optional[list] = None  # pre-defined regions (dicts) instead of detection

    @staticmethod
    def from_dict(d: dict) -> "PipelineConfig":
        cfg = PipelineConfig()
        for key, sub in (("detection", cfg.detection), ("geometry", cfg.geometry), ("texture", cfg.texture)):
            for k, v in (d.get(key) or {}).items():
                if hasattr(sub, k):
                    setattr(sub, k, v)
        for k in ("remove_geometry", "previews", "verify", "regions"):
            if k in d:
                setattr(cfg, k, d[k])
        return cfg


def _influence(region: SignRegion, info: Optional[RegionGeometry], extra: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = region.aabb(margin=0.6 + extra, front=0.6 + extra, back=0.8 + extra)
    lo = lo.copy()
    if info is not None and info.ground_z is not None:
        lo[2] = min(lo[2], info.ground_z - 0.5)
    else:
        lo[2] -= 8.0  # signs stand at most a few metres above the ground
    return lo, hi


def _files_near(files: list[FileInfo], lo, hi, leaf_only: bool) -> list[FileInfo]:
    out = []
    for f in files:
        if leaf_only and not f.has_leaf:
            continue
        if f.intersects(lo, hi, leaf_only=leaf_only):
            out.append(f)
    return out


class Pipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, progress: Optional[ProgressFn] = None, log: Optional[Callable[[str], None]] = None):
        self.cfg = config or PipelineConfig()
        self._progress = progress or (lambda f, m: None)
        self._log = log or (lambda m: None)

    def progress(self, frac: float, msg: str) -> None:
        self._progress(float(min(max(frac, 0.0), 1.0)), msg)
        self._log(msg)

    # -- detection ---------------------------------------------------------
    def detect(self, ds: Dataset) -> tuple[list[SignRegion], dict]:
        if self.cfg.regions:
            regions = [SignRegion.from_dict(d) for d in self.cfg.regions]
            for i, r in enumerate(regions):
                r.id = i
            return regions, {}
        detector = SignDetector(self.cfg.detection, log=self._log)
        tiles = ds.tiles()
        regions: list[SignRegion] = []
        evidence = {}
        for k, (tile, files) in enumerate(sorted(tiles.items())):
            leaf_files = [f for f in files if f.has_leaf]
            if not leaf_files:
                continue
            self.progress(0.05 + 0.35 * k / max(len(tiles), 1), f"检测 {tile}: 读取 {len(leaf_files)} 个最精细层级文件")
            parts = []
            for f in leaf_files:
                m = ds.load(f.rel)
                parts += [(p, m.texture_of(p)) for p in m.leaf_parts()]
            res = detector.detect(parts)
            for r in res.regions:
                ev = res.evidence.get(r.id)
                r.id = len(regions)
                regions.append(r)
                if ev is not None:
                    evidence[r.id] = ev
            self._log(f"  {tile}: {len(res.regions)} 个敏感标志")
        # signs on tile borders are seen from both tiles
        old_ids = {id(r): r.id for r in regions}
        merged = merge_regions(regions)
        evidence = {r.id: evidence[old_ids[id(r)]] for r in merged if old_ids[id(r)] in evidence}
        return merged, evidence

    # -- main --------------------------------------------------------------
    def run(self, input_path: str | Path, output_path: str | Path, work_dir: Optional[str | Path] = None) -> dict:
        t0 = time.time()
        output_path = Path(output_path)
        work_dir = Path(work_dir) if work_dir else output_path.parent / (output_path.name + "_s3d")
        work_dir.mkdir(parents=True, exist_ok=True)
        prev_dir = work_dir / "previews"
        prev_dir.mkdir(exist_ok=True)
        report: dict = {"input": str(input_path), "output": str(output_path), "regions": [], "warnings": []}

        self.progress(0.01, "扫描数据集")
        ds = Dataset(input_path)
        files = ds.scan()
        report["format"] = ds.kind
        report["files"] = len(files)
        report["tiles"] = len(ds.tiles())
        report["leaf_files"] = sum(1 for f in files if f.has_leaf)
        self._log(f"{ds.kind}: {len(files)} 个文件, {report['tiles']} 个瓦片")

        regions, evidence = self.detect(ds)
        report["detection_seconds"] = round(time.time() - t0, 1)
        self.progress(0.42, f"共检测到 {len(regions)} 个敏感标志")

        self.progress(0.43, "复制原始数据到输出目录")
        ds.copy_all(output_path)
        if not regions:
            report["seconds"] = round(time.time() - t0, 1)
            (work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
            self.progress(1.0, "未发现敏感标志, 输出与输入一致")
            return report

        # ---- finest level ------------------------------------------------
        leaf_sel: dict[str, FileInfo] = {}
        for r in regions:
            lo, hi = _influence(r, None, extra=self.cfg.texture.context)
            for f in _files_near(files, lo, hi, leaf_only=True):
                leaf_sel[f.rel] = f
        self.progress(0.45, f"修复最精细层级: {len(leaf_sel)} 个文件")
        leaf_meshes: dict[str, MeshFile] = {rel: ds.load(rel) for rel in leaf_sel}
        leaf_pairs = [(p, m.texture_of(p)) for m in leaf_meshes.values() for p in m.leaf_parts()]
        orig_items = [RenderItem(p, t.image.copy() if t is not None else None) for p, t in leaf_pairs]
        orig_items = [RenderItem(_snapshot(it.part), it.image) for it in orig_items]

        infos: dict[int, RegionGeometry] = {}
        for r in regions:
            lo, hi = _influence(r, None)
            near = [p for p, _ in leaf_pairs if _overlaps(p, lo, hi)]
            infos[r.id] = analyse_region(near, r, self.cfg.geometry) if self.cfg.remove_geometry else RegionGeometry(r.id, "none")
            self._log(f"  标志 {r.id} ({r.label or r.category}): 安装方式 {infos[r.id].mount}")

        # previews (before)
        cams = {}
        if self.cfg.previews:
            for r in regions:
                info = infos[r.id]
                cams[r.id] = context_camera(r.center, r.normal, r.size, info.ground_z)
                save_png(closeup(orig_items, cams[r.id]), prev_dir / f"region_{r.id}_before.png")

        leaf_edits = {}
        if self.cfg.remove_geometry:
            for rel, mesh in leaf_meshes.items():
                leaf_edits[rel] = remove_objects(mesh, regions, infos, self.cfg.geometry)
        self.progress(0.55, "生成修复视图并补全纹理")

        # targets on the finest level and repair views
        leaf_targets: dict[str, dict] = {}
        for rel, mesh in leaf_meshes.items():
            tg = {}
            for pi, part in enumerate(mesh.parts):
                tex = mesh.texture_of(part)
                if tex is None:
                    continue
                t = part_targets(part, tex, leaf_edits.get(rel, {}).get(pi), regions, self.cfg.texture)
                if t is not None:
                    tg[pi] = t
            leaf_targets[rel] = tg
        edited_items = [RenderItem(p, t.image if t is not None else None) for p, t in leaf_pairs]
        views = {}
        for k, r in enumerate(regions):
            tlist = []
            texels = []
            for rel, tg in leaf_targets.items():
                mesh = leaf_meshes[rel]
                for pi, t in tg.items():
                    if (t.region == r.id).any():
                        tlist.append((mesh.parts[pi], t))
                        texels.append(texel_size(mesh.parts[pi], mesh.texture_of(mesh.parts[pi])))
            views[r.id] = build_views(
                r, tlist, edited_items, self.cfg.texture,
                texel=float(np.median(texels)) if texels else 0.02, info=infos.get(r.id), log=self._log,
            )
            self.progress(0.55 + 0.2 * (k + 1) / len(regions), f"修复视图 {k + 1}/{len(regions)}")
        if self.cfg.previews:
            for r in regions:
                for j, v in enumerate(views.get(r.id, [])):
                    save_png(v.before, prev_dir / f"region_{r.id}_view{j}_before.png")
                    save_png(v.image, prev_dir / f"region_{r.id}_view{j}_after.png")

        stats = {"files_modified": 0, "texels_painted": 0, "faces_removed": 0, "faces_added": 0}
        for rel, mesh in leaf_meshes.items():
            stats["texels_painted"] += paint_targets(mesh, leaf_targets[rel], views, coarse=False, cfg=self.cfg.texture)
            edits = leaf_edits.get(rel, {})
            wipe_freed_texels(mesh, edits)
            stats["faces_removed"] += sum(e.removed for e in edits.values())
            stats["faces_added"] += sum(len(e.new_faces) for e in edits.values())

        # previews (after) + verification on the repaired finest level
        if self.cfg.previews:
            after_items = [RenderItem(p, t.image if t is not None else None) for p, t in leaf_pairs]
            for r in regions:
                save_png(closeup(after_items, cams[r.id]), prev_dir / f"region_{r.id}_after.png")
                self._region_glbs(r, infos[r.id], orig_items, leaf_pairs, prev_dir)
        if self.cfg.verify:
            residual = self._verify(regions, leaf_pairs)
            report["residual_detections"] = residual
            if residual:
                report["warnings"].append(f"修复后复检仍发现 {len(residual)} 处疑似标志, 请人工查看")

        for rel, mesh in leaf_meshes.items():
            if mesh.dirty:
                ds.save(mesh, output_path)
                stats["files_modified"] += 1
        self.progress(0.8, "修复其余 LOD 层级")

        # ---- coarser levels ----------------------------------------------
        coarse_sel: dict[str, FileInfo] = {}
        for r in regions:
            lo, hi = _influence(r, infos[r.id])
            for f in _files_near(files, lo, hi, leaf_only=False):
                if f.rel not in leaf_meshes:
                    coarse_sel[f.rel] = f
        for k, rel in enumerate(sorted(coarse_sel)):
            mesh = ds.load(rel)
            edits = remove_objects(mesh, regions, infos, self.cfg.geometry) if self.cfg.remove_geometry else {}
            tg = {}
            for pi, part in enumerate(mesh.parts):
                tex = mesh.texture_of(part)
                if tex is None:
                    continue
                t = part_targets(part, tex, edits.get(pi), regions, self.cfg.texture)
                if t is not None:
                    tg[pi] = t
            stats["texels_painted"] += paint_targets(mesh, tg, views, coarse=True, cfg=self.cfg.texture)
            wipe_freed_texels(mesh, edits)
            stats["faces_removed"] += sum(e.removed for e in edits.values())
            stats["faces_added"] += sum(len(e.new_faces) for e in edits.values())
            if mesh.dirty:
                ds.save(mesh, output_path)
                stats["files_modified"] += 1
            self.progress(0.8 + 0.18 * (k + 1) / max(len(coarse_sel), 1), f"LOD 文件 {k + 1}/{len(coarse_sel)}: {rel}")

        report.update(stats)
        for r in regions:
            d = r.to_dict()
            info = infos[r.id]
            d["mount"] = info.mount
            d["ground_z"] = info.ground_z
            d["views"] = len(views.get(r.id, []))
            if self.cfg.previews:
                d["preview_before"] = f"previews/region_{r.id}_before.png"
                d["preview_after"] = f"previews/region_{r.id}_after.png"
                d["glb_before"] = f"previews/region_{r.id}_before.glb"
                d["glb_after"] = f"previews/region_{r.id}_after.glb"
            report["regions"].append(d)
        report["summary"] = _summary(regions)
        report["seconds"] = round(time.time() - t0, 1)
        (work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        self.progress(1.0, f"完成: 去除 {len(regions)} 个敏感标志, 修改 {stats['files_modified']} 个文件")
        return report

    # -- helpers -----------------------------------------------------------
    def _region_glbs(self, r, info, orig_items, leaf_pairs, prev_dir) -> None:
        span = max(3.0, r.size * 3)
        lo = r.center - span
        hi = r.center + span
        if info.ground_z is not None:
            lo[2] = info.ground_z - 0.5
        origin = r.center.copy()
        write_glb(crop_parts([(it.part, _ImgTex(it.image)) for it in orig_items], lo, hi), prev_dir / f"region_{r.id}_before.glb", origin)
        write_glb(crop_parts(leaf_pairs, lo, hi), prev_dir / f"region_{r.id}_after.glb", origin)

    def _verify(self, regions, leaf_pairs) -> list[dict]:
        """Re-run detection around the repaired regions."""
        near = []
        for p, t in leaf_pairs:
            if t is None:
                continue
            for r in regions:
                lo, hi = _influence(r, None, extra=0.2)
                if _overlaps(p, lo, hi):
                    near.append((p, t))
                    break
        if not near:
            return []
        res = SignDetector(self.cfg.detection).detect(near)
        out = []
        for rr in res.regions:
            if any(np.linalg.norm(rr.center - r.center) < r.size + 0.5 for r in regions):
                out.append(rr.to_dict())
        return out


class _ImgTex:
    """Minimal texture stand-in for preview cropping."""

    def __init__(self, image):
        self.image = image


def _snapshot(part):
    from .core.mesh import MeshPart

    return MeshPart(part.vertices.copy(), part.faces.copy(), uvs=None if part.uvs is None else part.uvs.copy(), texture=part.texture, has_finer=part.has_finer)


def _overlaps(part, lo, hi) -> bool:
    a, b = part.bounds()
    return bool(np.all(a <= hi) and np.all(b >= lo))


def _summary(regions: list[SignRegion]) -> dict:
    out: dict = {}
    for r in regions:
        key = CATEGORY_LABELS.get(r.category, r.category)
        out[key] = out.get(key, 0) + 1
    return out


def run(input_path, output_path, config: Optional[PipelineConfig] = None, progress: Optional[ProgressFn] = None, log=None, work_dir=None) -> dict:
    return Pipeline(config, progress, log).run(input_path, output_path, work_dir)
