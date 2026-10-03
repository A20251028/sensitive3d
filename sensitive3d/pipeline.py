"""Processing stages: scan -> detect -> review -> repair.

Each stage writes its own JSON into the work directory so it can be audited
and resumed independently:

* ``scan.json``        – structure of the input (LOD graph, failures, textures)
                          + ``input_manifest.json`` (SHA-256 of every input file);
* ``detection.json``   – candidates with provenance (tile, files, LOD, cameras,
                          2D/3D mapping) and evidence images under ``evidence/``;
                          nothing in the dataset is written;
* ``review.json``      – the decisions that were executed (see :mod:`sensitive3d.review`);
* ``repair.json``      – what was changed per file, read-back checks, protected
                          area checks, residual checks and the input hash check.

:meth:`Pipeline.run` keeps the fully automatic behaviour (only candidates whose
status is ``auto`` are repaired; the others are reported as pending).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image

from .core.mesh import MeshFile, MeshPart
from .core.render import RenderItem
from .core.texture import texel_map, texel_size
from .detect.detector import ALL_CATEGORIES, DetectionConfig, SignDetector
from .detect.heuristic import color_masks
from .detect.regions import CATEGORY_LABELS, SignRegion, merge_regions
from .io.dataset import Dataset, FileInfo
from .preview import closeup, context_camera, crop_parts, save_png, write_glb
from .repair.geometry import GeometryConfig, RegionGeometry, analyse_region, remove_objects
from .repair.texture import TextureConfig, build_views, paint_targets, part_targets, wipe_freed_texels
from .review import RepairTarget, ReviewError, parse_protected, resolve_targets
from .runctx import BudgetExceeded, Cancelled, RunContext, RunLimits, StopRun

ProgressFn = Callable[[float, str], None]
SIGN_COLORS = ("red", "blue", "green", "yellow")


class ScanBlocked(RuntimeError):
    """The input has structural problems; automatic repair is refused."""

    def __init__(self, problems: list[str]):
        super().__init__("输入数据存在阻断性问题, 已停止自动修复: " + "; ".join(problems[:5]) + (" …" if len(problems) > 5 else ""))
        self.problems = problems


@dataclass
class PipelineConfig:
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    texture: TextureConfig = field(default_factory=TextureConfig)
    limits: RunLimits = field(default_factory=RunLimits)
    remove_geometry: bool = True  # default operation of auto-accepted candidates
    previews: bool = True
    verify: bool = True
    regions: Optional[list] = None  # pre-defined regions (dicts) instead of detection
    auto_accept_score: float = 0.85  # below this a candidate needs review
    allow_scan_errors: bool = False  # repair even if the scan reported blocking problems (CLI only)
    diagnostics: bool = False  # keep close-ups + reasons of rejected colour candidates
    hash_inputs: bool = True  # SHA-256 manifest of the input before / after

    @staticmethod
    def from_dict(d: dict) -> "PipelineConfig":
        cfg = PipelineConfig()
        for key, sub in (("detection", cfg.detection), ("geometry", cfg.geometry), ("texture", cfg.texture)):
            for k, v in (d.get(key) or {}).items():
                if hasattr(sub, k):
                    setattr(sub, k, tuple(v) if k == "categories" else v)
        if d.get("limits"):
            cfg.limits = RunLimits.from_dict(d["limits"])
        for k in ("remove_geometry", "previews", "verify", "regions", "auto_accept_score", "diagnostics", "hash_inputs"):
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
        if not getattr(f, "ok", True):
            continue
        if leaf_only and not f.has_leaf:
            continue
        if f.intersects(lo, hi, leaf_only=leaf_only):
            out.append(f)
    return out


def _overlaps(part, lo, hi) -> bool:
    a, b = part.bounds()
    return bool(np.all(a <= hi) and np.all(b >= lo))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _snapshot(part: MeshPart) -> MeshPart:
    return MeshPart(part.vertices.copy(), part.faces.copy(), uvs=None if part.uvs is None else part.uvs.copy(), texture=part.texture, has_finer=part.has_finer)


class _ImgTex:
    """Minimal texture stand-in for preview cropping."""

    def __init__(self, image):
        self.image = image


def _summary(regions) -> dict:
    out: dict = {}
    for r in regions:
        cat = r.category if isinstance(r, SignRegion) else r["category"]
        key = CATEGORY_LABELS.get(cat, cat)
        out[key] = out.get(key, 0) + 1
    return out


def _overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, (255, 30, 30), 2)
    return out


class Pipeline:
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        progress: Optional[ProgressFn] = None,
        log: Optional[Callable[[str], None]] = None,
        cancel_event=None,
    ):
        self.cfg = config or PipelineConfig()
        self.ctx = RunContext(self.cfg.limits, progress, log, cancel_event)

    # convenience used by older callers
    def progress(self, frac: float, msg: str) -> None:
        self.ctx.progress(frac, msg)

    def _bridge(self, fn, *args, what: str = ""):
        """Run an io call with timeout / cancel; map bridge stops to run stops."""
        from .io import osgb

        try:
            return fn(*args, **self.ctx.bridge_kwargs())
        except osgb.BridgeCancelled as e:
            raise Cancelled(f"已取消 ({what})") from e
        except osgb.BridgeTimeout as e:
            raise BudgetExceeded(f"{what} 超过单次读写超时: {e}") from e

    def _load(self, ds: Dataset, rel: str) -> MeshFile:
        self.ctx.check(f"读取 {rel}")
        return self._bridge(ds.load, rel, what=f"读取 {rel}")

    def _save(self, ds: Dataset, mesh: MeshFile, output_path) -> None:
        self.ctx.check(f"写出 {mesh.path}")
        self._bridge(ds.save, mesh, output_path, what=f"写出 {mesh.path}")

    # =====================================================================
    # stage 1: import check
    # =====================================================================
    def scan(self, input_path, work_dir: Optional[Path] = None) -> tuple[Dataset, dict]:
        ctx = self.ctx
        ctx.progress(0.0, "导入检查: 扫描数据集结构")
        ds = Dataset(input_path)
        files = self._bridge(ds.scan, what="结构扫描")
        rep = ds.report.to_dict() if getattr(ds, "report", None) is not None else {}
        rep["file_list"] = rep.pop("files", [])
        rep.update(
            {
                "stage": "scan",
                "input": str(input_path),
                "format": ds.kind,
                "files": len(files),
                "files_ok": sum(1 for f in files if getattr(f, "ok", True)),
                "tiles": len(ds.tiles()),
                "leaf_files": sum(1 for f in files if f.has_leaf and getattr(f, "ok", True)),
                "triangles": int(sum(f.num_triangles for f in files)),
            }
        )
        blocking = list(getattr(getattr(ds, "report", None), "blocking_errors", []) or [])
        rep["blocking_errors"] = blocking
        rep["ok_for_repair"] = not blocking
        if self.cfg.hash_inputs:
            ctx.progress(0.5, "导入检查: 计算输入文件哈希")
            from .io.inputs import file_manifest

            manifest = file_manifest(ds.root)
            rep["input_manifest"] = {"files": len(manifest["files"]), "bytes": manifest["total_bytes"], "ignored": len(manifest.get("ignored", []))}
            if work_dir is not None:
                _write_json(Path(work_dir) / "input_manifest.json", manifest)
        rep["run"] = ctx.stats()
        if work_dir is not None:
            _write_json(Path(work_dir) / "scan.json", rep)
        ctx.progress(1.0, f"导入检查完成: {rep['files_ok']}/{len(files)} 个文件可读" + (f", {len(blocking)} 个阻断问题" if blocking else ""))
        return ds, rep

    # =====================================================================
    # stage 2: detection only (never writes the dataset)
    # =====================================================================
    def detect(self, ds: Dataset, work_dir: Optional[Path] = None) -> dict:
        ctx = self.ctx
        cfg = self.cfg
        work_dir = Path(work_dir) if work_dir is not None else None
        ev_dir = work_dir / "evidence" if work_dir is not None else None
        if ev_dir is not None:
            ev_dir.mkdir(parents=True, exist_ok=True)
        if not ds.files:
            self._bridge(ds.scan, what="结构扫描")
        t0 = time.time()
        records: list[dict] = []  # per (tile-level) region, before merging
        regions: list[SignRegion] = []
        rejected_total = 0
        candidates_total = 0
        stopped = None
        detector = SignDetector(cfg.detection, log=ctx.log)
        tiles = sorted(ds.tiles().items())
        try:
            if cfg.regions:
                self._predefined(ds, records, regions)
            else:
                for k, (tile, files) in enumerate(tiles):
                    leaf_files = [f for f in files if f.has_leaf and getattr(f, "ok", True)]
                    if not leaf_files:
                        continue
                    ctx.check(f"检测 {tile}")
                    mb = sum(getattr(f, "decoded_bytes", 0) or 0 for f in leaf_files) / 2**20
                    ctx.check_decoded(mb, f"瓦片 {tile} 的最精细层级")
                    ctx.progress(k / max(len(tiles), 1), f"检测 {tile}: 读取 {len(leaf_files)} 个最精细层级文件")
                    parts, owner = [], {}
                    for f in leaf_files:
                        m = self._load(ds, f.rel)
                        for p in m.leaf_parts():
                            parts.append((p, m.texture_of(p)))
                            owner[id(p)] = f
                    res = detector.detect(parts, ctx=ctx, diagnostics=cfg.diagnostics)
                    candidates_total += res.candidates
                    rejected_total += len(res.rejected)
                    if cfg.diagnostics and ev_dir is not None:
                        self._save_rejected(ev_dir, tile, res.rejected)
                    for r in res.regions:
                        ev = res.evidence.get(r.id)
                        self._record(r, ev, tile, parts, owner, records, regions, ev_dir)
                    ctx.log(f"  {tile}: {res.candidates} 个颜色候选, {len(res.regions)} 个标志")
        except StopRun as e:
            stopped = {"reason": e.reason, "message": str(e)}
            ctx.log(f"检测提前停止: {e}")

        # signs on tile borders are seen from both tiles: merge, union the provenance
        for i, r in enumerate(regions):
            r.sources = [i]
        merged = merge_regions(regions)
        candidates = []
        selected = set(cfg.detection.categories)
        for r in merged:
            idxs = r.sources or []
            best = records[idxs[0]]
            rec = dict(best)
            rec["id"] = r.id
            rec["region"] = r.to_dict()
            prov = dict(best["provenance"])
            prov["tiles"] = sorted({t for i in idxs for t in records[i]["provenance"]["tiles"]})
            prov["files"] = sorted({f for i in idxs for f in records[i]["provenance"]["files"]})
            prov["lod"] = {"finest_only": True, "depths": {f: d for i in idxs for f, d in records[i]["provenance"]["lod"]["depths"].items()}}
            rec["provenance"] = prov
            rec["review_reasons"] = list(dict.fromkeys(list(r.review_reasons) + list(best.get("review_reasons", []))))
            self._classify(rec, selected)
            if ev_dir is not None:
                rec["evidence"] = self._rename_evidence(ev_dir, best.get("_ev_key"), r.id)
            rec.pop("_ev_key", None)
            candidates.append(rec)
        for r in merged:
            r.sources = []

        report = {
            "stage": "detect",
            "format": ds.kind,
            "candidates": candidates,
            "counts": {
                "total": len(candidates),
                "auto": sum(1 for c in candidates if c["status"] == "auto"),
                "review": sum(1 for c in candidates if c["status"] == "review"),
                "colour_candidates": candidates_total,
                "colour_candidates_without_sign": rejected_total if cfg.diagnostics else None,
            },
            "summary": _summary(candidates),
            "metrics": {"recall": None, "precision": None, "f1": None, "note": "没有独立人工标注 (漏检分母未知), 召回率/F1 不可计算; 置信度不是准确率"},
            "config": {
                "categories": list(cfg.detection.categories),
                "min_score": cfg.detection.min_score,
                "auto_accept_score": cfg.auto_accept_score,
                "yolo_model": cfg.detection.yolo_model,
                "predefined_regions": bool(cfg.regions),
            },
            "environment": _environment(),
            "seconds": round(time.time() - t0, 1),
            "run": ctx.stats(),
            "stopped": stopped,
        }
        if work_dir is not None:
            _write_json(work_dir / "detection.json", report)
        ctx.progress(1.0, f"检测完成: {report['counts']['auto']} 个可自动处理, {report['counts']['review']} 个待审核" + (" (提前停止)" if stopped else ""))
        return report

    def _predefined(self, ds: Dataset, records: list, regions: list) -> None:
        """Candidates from user supplied regions (status auto: a person chose them)."""
        for i, d in enumerate(self.cfg.regions):
            r = SignRegion.from_dict({"width": 0.8, "height": 0.8, **d})
            r.id = i
            r.score = float(d.get("score", 1.0))
            r.mapping = {"status": "ok", "ok": True, "provisional": False, "reason": "人工给定区域"}
            lo, hi = _influence(r, None)
            files = _files_near(ds.files, lo, hi, leaf_only=True)
            parts, owner = [], {}
            for f in files:
                m = self._load(ds, f.rel)
                for p in m.leaf_parts():
                    parts.append((p, m.texture_of(p)))
                    owner[id(p)] = f
            self._record(r, None, files[0].tile if files else "", parts, owner, records, regions, None)
            records[-1]["predefined"] = True
            if "operation" in d:
                records[-1]["requested_operation"] = d["operation"]

    def _record(self, r: SignRegion, ev, tile: str, parts, owner, records: list, regions: list, ev_dir: Optional[Path]) -> None:
        """Provenance + mounting analysis of one detected region (finest level of one tile)."""
        cfg = self.cfg
        lo, hi = _influence(r, None)
        near = [(p, t) for p, t in parts if _overlaps(p, lo, hi)]
        info = analyse_region([p for p, _ in near], r, cfg.geometry)
        rlo, rhi = r.aabb(margin=0.05, front=0.12, back=0.3)
        files = sorted({owner[id(p)].rel for p, _ in near if _overlaps(p, rlo, rhi) and r.contains(p.centroids(), margin=0.05, front=0.12, back=0.3).any()})
        depths = {owner[id(p)].rel: getattr(owner[id(p)], "depth", None) for p, _ in near if owner[id(p)].rel in files}
        rec = {
            "category": r.category,
            "category_label": CATEGORY_LABELS.get(r.category, r.category),
            "label": r.label,
            "score": round(float(r.score), 3),
            "mapping": dict(r.mapping),
            "mount": info.to_dict(),
            "review_reasons": list(r.review_reasons),
            "provenance": {
                "tiles": [tile] if tile else [],
                "files": files,
                "lod": {"finest_only": True, "depths": depths},
                "detector": ev.detection.source if ev is not None else "predefined",
                "colour": ev.candidate_color if ev is not None else None,
                "camera": ev.camera.to_dict() if ev is not None else None,
                "bbox2d": [int(x) for x in ev.detection.bbox] if ev is not None else None,
                "context_camera": context_camera(r.center, r.normal, r.size, info.ground_z).to_dict(),
            },
        }
        key = f"t{len(records)}"
        rec["_ev_key"] = key
        if ev_dir is not None:
            if ev is not None and ev.closeup is not None:
                save_png(ev.closeup, ev_dir / f"{key}_closeup.png", max_side=512)
                Image.fromarray((ev.detection.mask * 255).astype(np.uint8)).save(ev_dir / f"{key}_mask.png")
                save_png(_overlay(ev.closeup, ev.detection.mask), ev_dir / f"{key}_overlay.png", max_side=512)
            if parts:
                items = [RenderItem(p, t.image if t is not None else None) for p, t in near]
                cam = context_camera(r.center, r.normal, r.size, info.ground_z)
                save_png(closeup(items, cam), ev_dir / f"{key}_context.png")
        records.append(rec)
        regions.append(r)

    def _classify(self, rec: dict, selected: set) -> None:
        """status auto / review + suggested decision."""
        cfg = self.cfg
        reasons = list(rec.get("review_reasons", []))
        mapping_ok = bool(rec["mapping"].get("ok"))
        if rec.get("predefined"):
            reasons = [x for x in reasons if not x.startswith("置信度")]
        elif rec["score"] < cfg.auto_accept_score:
            reasons.append(f"置信度 {rec['score']:.2f} 低于自动处理阈值 {cfg.auto_accept_score:.2f}")
        if not mapping_ok and not any("映射" in x for x in reasons):
            reasons.append("二维→三维映射失败")
        mount = rec["mount"]
        if mount.get("confidence") != "high":
            reasons.append(f"安装方式不确定 ({mount.get('type')})")
        if rec["category"] not in selected:
            reasons.append(f"类别 {rec['category_label']} 未被选为去除对象")
        rec["review_reasons"] = list(dict.fromkeys(reasons))
        rec["status"] = "auto" if not rec["review_reasons"] else "review"
        if not mapping_ok:
            op = "none"
        elif rec.get("requested_operation") in ("geometry", "texture"):
            op = rec["requested_operation"]
        elif cfg.remove_geometry and mount.get("type") in ("pole", "wall", "free") and mount.get("confidence") == "high":
            op = "geometry"
        else:
            op = "texture"
        rec["suggested"] = {"accept": rec["status"] == "auto" and op != "none", "operation": op}

    def _rename_evidence(self, ev_dir: Path, key: Optional[str], cid: int) -> dict:
        out = {}
        for kind in ("closeup", "mask", "overlay", "context"):
            src = ev_dir / f"{key}_{kind}.png"
            if key and src.is_file():
                dst = ev_dir / f"cand_{cid}_{kind}.png"
                src.replace(dst)
                out[kind] = f"evidence/{dst.name}"
        return out

    def _save_rejected(self, ev_dir: Path, tile: str, rejected: list) -> None:
        rdir = ev_dir / "rejected"
        rdir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "+-_" else "_" for c in tile)
        index_path = rdir / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else []
        for rj in rejected:
            name = f"{safe}_c{rj['candidate']}"
            save_png(rj["closeup"], rdir / f"{name}.png", max_side=512)
            entry = {k: v for k, v in rj.items() if k != "closeup"}
            entry.update(tile=tile, image=f"evidence/rejected/{name}.png")
            index.append(entry)
        _write_json(index_path, index)

    # =====================================================================
    # stage 3+4: execute reviewed decisions
    # =====================================================================
    def repair(
        self,
        ds: Dataset,
        detection: dict,
        review: Optional[dict],
        output_path,
        work_dir: Optional[Path] = None,
        auto: bool = False,
    ) -> dict:
        ctx = self.ctx
        cfg = self.cfg
        t0 = time.time()
        output_path = Path(output_path)
        work_dir = Path(work_dir) if work_dir is not None else output_path.parent / (output_path.name + "_s3d")
        prev_dir = work_dir / "previews"
        prev_dir.mkdir(parents=True, exist_ok=True)
        if not ds.files:
            self._bridge(ds.scan, what="结构扫描")
        blocking = list(getattr(getattr(ds, "report", None), "blocking_errors", []) or [])
        if blocking and not cfg.allow_scan_errors:
            raise ScanBlocked(blocking)
        targets, skipped = resolve_targets(detection, review, auto)
        protected = parse_protected(review)
        if review is not None and work_dir is not None:
            _write_json(work_dir / "review.json", review)
        report: dict = {
            "stage": "repair",
            "input": str(ds.root if ds.single_file is None else ds.single_file),
            "output": str(output_path),
            "format": ds.kind,
            "targets": [{"id": t.candidate_id, "operation": t.operation, "decided_by": t.decided_by, "mount": t.mount.mount} for t in targets],
            "skipped": skipped,
            "protected": [{"min": lo.tolist(), "max": hi.tolist()} for lo, hi in protected],
            "warnings": [],
            "files": [],
            "stopped": None,
        }
        ctx.progress(0.0, "复制原始数据到输出目录 (工作副本)")
        ds.copy_all(output_path)
        if not targets:
            report.update(files_modified=0, texels_painted=0, faces_removed=0, faces_added=0, seconds=round(time.time() - t0, 1))
            report["input_check"] = self._input_check(ds, work_dir)
            _write_json(work_dir / "repair.json", report)
            ctx.progress(1.0, "没有确认的修复目标, 输出与输入一致")
            return report
        try:
            self._repair_targets(ds, targets, protected, output_path, work_dir, prev_dir, report)
        except StopRun as e:
            report["stopped"] = {"reason": e.reason, "message": str(e)}
            report["warnings"].append(f"修复提前停止: {e}. 输出目录不完整, 不能使用; 已保存的证据见报告")
        report["input_check"] = self._input_check(ds, work_dir)
        if report["input_check"].get("ok") is False:
            report["warnings"].append("输入文件哈希发生变化! 请检查是否有其他程序修改了输入目录")
        report["seconds"] = round(time.time() - t0, 1)
        report["run"] = ctx.stats()
        _write_json(work_dir / "repair.json", report)
        done = report.get("files_modified", 0)
        ctx.progress(1.0, ("修复已停止" if report["stopped"] else "修复完成") + f": {len(targets)} 个目标, 修改 {done} 个文件")
        return report

    def _repair_targets(self, ds, targets: list[RepairTarget], protected, output_path, work_dir, prev_dir, report) -> None:
        ctx = self.ctx
        cfg = self.cfg
        files = ds.files
        regions = [t.region for t in targets]
        infos = {t.candidate_id: t.mount for t in targets}
        geo_regions = [t.region for t in targets if t.operation == "geometry"]

        # ---- finest level ------------------------------------------------
        leaf_sel: dict[str, FileInfo] = {}
        for r in regions:
            lo, hi = _influence(r, infos[r.id], extra=cfg.texture.context)
            for f in _files_near(files, lo, hi, leaf_only=True):
                leaf_sel[f.rel] = f
        mb = sum(getattr(f, "decoded_bytes", 0) or 0 for f in leaf_sel.values()) / 2**20
        ctx.check_decoded(mb, "修复区域的最精细层级文件")
        ctx.progress(0.05, f"修复最精细层级: {len(leaf_sel)} 个文件")
        leaf_meshes: dict[str, MeshFile] = {}
        for i, rel in enumerate(sorted(leaf_sel)):
            leaf_meshes[rel] = self._load(ds, rel)
            ctx.progress(0.05 + 0.1 * (i + 1) / len(leaf_sel), f"读取 {i + 1}/{len(leaf_sel)}: {rel}")
        leaf_pairs = [(p, m.texture_of(p)) for m in leaf_meshes.values() for p in m.leaf_parts()]
        orig_items = [RenderItem(_snapshot(p), t.image.copy() if t is not None else None) for p, t in leaf_pairs]

        cams = {}
        if cfg.previews:
            for r in regions:
                info = infos[r.id]
                cams[r.id] = context_camera(r.center, r.normal, r.size, info.ground_z)
                save_png(closeup(orig_items, cams[r.id]), prev_dir / f"region_{r.id}_before.png")

        leaf_edits: dict[str, dict] = {}
        for rel, mesh in leaf_meshes.items():
            ctx.check("删除标志几何")
            leaf_edits[rel] = remove_objects(mesh, geo_regions, infos, cfg.geometry, protected=protected) if geo_regions else {}
        ctx.progress(0.25, "生成修复视图并补全纹理")

        leaf_targets: dict[str, dict] = {}
        for rel, mesh in leaf_meshes.items():
            tg = {}
            for pi, part in enumerate(mesh.parts):
                tex = mesh.texture_of(part)
                if tex is None:
                    continue
                t = part_targets(part, tex, leaf_edits.get(rel, {}).get(pi), regions, cfg.texture, protected=protected)
                if t is not None:
                    tg[pi] = t
            leaf_targets[rel] = tg
        edited_items = [RenderItem(p, t.image if t is not None else None) for p, t in leaf_pairs]
        views = {}
        for k, r in enumerate(regions):
            ctx.check("修复视图")
            tlist, texels = [], []
            for rel, tg in leaf_targets.items():
                mesh = leaf_meshes[rel]
                for pi, t in tg.items():
                    if (t.region == r.id).any():
                        tlist.append((mesh.parts[pi], t))
                        texels.append(texel_size(mesh.parts[pi], mesh.texture_of(mesh.parts[pi])))
            views[r.id] = build_views(r, tlist, edited_items, cfg.texture, texel=float(np.median(texels)) if texels else 0.02, info=infos.get(r.id), log=ctx.log)
            ctx.progress(0.25 + 0.25 * (k + 1) / len(regions), f"修复视图 {k + 1}/{len(regions)}")
        if cfg.previews:
            for r in regions:
                for j, v in enumerate(views.get(r.id, [])):
                    save_png(v.before, prev_dir / f"region_{r.id}_view{j}_before.png")
                    save_png(v.image, prev_dir / f"region_{r.id}_view{j}_after.png")

        totals = {"files_modified": 0, "texels_painted": 0, "faces_removed": 0, "faces_added": 0, "vertices_added": 0, "protected_faces_kept": 0}
        atlas = []
        for rel, mesh in leaf_meshes.items():
            painted = paint_targets(mesh, leaf_targets[rel], views, coarse=False, cfg=cfg.texture)
            edits = leaf_edits.get(rel, {})
            wipe_freed_texels(mesh, edits)
            atlas.extend(_atlas_residual(rel, mesh, edits, leaf_targets[rel]))
            self._account(rel, mesh, edits, painted, totals, report)

        if cfg.previews:
            after_items = [RenderItem(p, t.image if t is not None else None) for p, t in leaf_pairs]
            for r in regions:
                save_png(closeup(after_items, cams[r.id]), prev_dir / f"region_{r.id}_after.png")
                self._region_glbs(r, infos[r.id], orig_items, leaf_pairs, prev_dir)
        if cfg.verify:
            residual = self._verify(regions, leaf_pairs)
            report["residual_detections"] = residual
            if residual:
                report["warnings"].append(f"修复后复检仍发现 {len(residual)} 处疑似标志, 请人工查看 (复检只是辅助手段)")
        for rel, mesh in leaf_meshes.items():
            if mesh.dirty:
                self._save(ds, mesh, output_path)
                totals["files_modified"] += 1
                self._readback(ds, rel, output_path, mesh, report)
        ctx.progress(0.6, "修复其余 LOD 层级")

        # ---- coarser levels ----------------------------------------------
        coarse_sel: dict[str, FileInfo] = {}
        for r in regions:
            lo, hi = _influence(r, infos[r.id])
            for f in _files_near(files, lo, hi, leaf_only=False):
                if f.rel not in leaf_meshes:
                    coarse_sel[f.rel] = f
        for k, rel in enumerate(sorted(coarse_sel)):
            ctx.check_decoded((getattr(coarse_sel[rel], "decoded_bytes", 0) or 0) / 2**20, rel)
            mesh = self._load(ds, rel)
            edits = remove_objects(mesh, geo_regions, infos, cfg.geometry, protected=protected) if geo_regions else {}
            tg = {}
            for pi, part in enumerate(mesh.parts):
                tex = mesh.texture_of(part)
                if tex is None:
                    continue
                t = part_targets(part, tex, edits.get(pi), regions, cfg.texture, protected=protected)
                if t is not None:
                    tg[pi] = t
            painted = paint_targets(mesh, tg, views, coarse=True, cfg=cfg.texture)
            wipe_freed_texels(mesh, edits)
            atlas.extend(_atlas_residual(rel, mesh, edits, tg))
            self._account(rel, mesh, edits, painted, totals, report)
            if mesh.dirty:
                self._save(ds, mesh, output_path)
                totals["files_modified"] += 1
                self._readback(ds, rel, output_path, mesh, report)
            ctx.progress(0.6 + 0.3 * (k + 1) / max(len(coarse_sel), 1), f"LOD 文件 {k + 1}/{len(coarse_sel)}: {rel}")

        report.update(totals)
        report["atlas_residual"] = atlas
        bad_atlas = [a for a in atlas if a["sign_colour_fraction"] > 0.02]
        if bad_atlas:
            report["warnings"].append(f"{len(bad_atlas)} 个纹理在被修改/释放的区域仍有较多标志颜色像素, 请人工检查")
        if protected:
            ctx.progress(0.92, "检查保护区域")
            report["protected_check"] = self._protected_check(ds, output_path, protected, list(leaf_meshes) + sorted(coarse_sel))
            if any(not p["ok"] for p in report["protected_check"]):
                report["warnings"].append("保护区域内检测到变化, 不能判定为通过")
        failed_rb = [f for f in report["files"] if f.get("readback", {}).get("ok") is False]
        if failed_rb:
            report["warnings"].append(f"{len(failed_rb)} 个输出文件回读校验失败")
        report["regions"] = []
        for t in targets:
            r = t.region
            d = r.to_dict()
            d.update(
                id=t.candidate_id,
                operation=t.operation,
                decided_by=t.decided_by,
                mount=t.mount.mount,
                mount_confidence=t.mount.confidence,
                ground_z=t.mount.ground_z,
                views=len(views.get(r.id, [])),
            )
            if cfg.previews:
                d.update(
                    preview_before=f"previews/region_{r.id}_before.png",
                    preview_after=f"previews/region_{r.id}_after.png",
                    glb_before=f"previews/region_{r.id}_before.glb",
                    glb_after=f"previews/region_{r.id}_after.glb",
                    context_camera=cams[r.id].to_dict() if r.id in cams else None,
                )
            report["regions"].append(d)
        report["summary"] = _summary(regions)

    # -- accounting / checks -------------------------------------------------
    def _account(self, rel: str, mesh: MeshFile, edits: dict, painted: int, totals: dict, report: dict) -> None:
        rec = {
            "rel": rel,
            "modified": mesh.dirty,
            "geometry_changed": any(p.dirty for p in mesh.parts),
            "faces_removed": int(sum(e.removed for e in edits.values())),
            "faces_added": int(sum(len(e.new_faces) for e in edits.values())),
            "vertices_added": int(sum(e.new_vertices for e in edits.values())),
            "protected_faces_kept": int(sum(e.protected_kept for e in edits.values())),
            "texels_painted": int(painted),
            "textures_modified": [k for k, t in enumerate(mesh.textures) if t.dirty],
            "atlas_growth": list(mesh.meta.get("atlas_growth", [])),
            "uv_remapped": bool(mesh.meta.get("atlas_growth")),
        }
        if not mesh.dirty:
            return
        for k in ("faces_removed", "faces_added", "vertices_added", "protected_faces_kept", "texels_painted"):
            totals[k] = totals.get(k, 0) + rec[k]
        report["files"].append(rec)

    def _output_dataset(self, ds: Dataset, output_path: Path) -> Dataset:
        cached = getattr(self, "_out_ds", None)
        if cached is None or cached.root != Path(output_path):
            cached = Dataset(output_path)
            cached.kind = ds.kind
            self._out_ds = cached
        return cached

    def _readback(self, ds: Dataset, rel: str, output_path: Path, mesh: MeshFile, report: dict) -> None:
        """Re-read a written file; texture-only files must keep geometry and UVs bit-identical."""
        rec = next(f for f in report["files"] if f["rel"] == rel)
        problems = []
        try:
            back = self._output_dataset(ds, output_path).load(rel)
        except Exception as e:  # noqa: BLE001 - reported, not hidden
            rec["readback"] = {"ok": False, "problems": [f"无法回读: {e}"]}
            return
        if len(back.parts) != len(mesh.parts):
            problems.append(f"几何体数量 {len(back.parts)} != {len(mesh.parts)}")
        else:
            for a, b in zip(mesh.parts, back.parts):
                # unreferenced vertices may be dropped by a format (OBJ), so compare what faces use
                if len(a.faces) != len(b.faces):
                    problems.append(f"几何体 {a.index}: 三角形数 {len(b.faces)} != {len(a.faces)}")
                    continue
                if not len(a.faces):
                    continue
                (alo, ahi), (blo, bhi) = a.bounds(), b.bounds()
                if not (np.allclose(alo, blo, atol=1e-3) and np.allclose(ahi, bhi, atol=1e-3)):
                    problems.append(f"几何体 {a.index}: 包围盒不一致")
                aa, ba = float(a.face_areas().sum()), float(b.face_areas().sum())
                if abs(aa - ba) > 1e-4 * max(aa, 1e-9) + 1e-6:
                    problems.append(f"几何体 {a.index}: 表面积 {ba:.4f} != {aa:.4f}")
        for k, (ta, tb) in enumerate(zip(mesh.textures, back.textures)):
            if ta.image.shape[:2] != tb.image.shape[:2]:
                problems.append(f"纹理 {k}: 尺寸 {tb.image.shape[:2]} != {ta.image.shape[:2]}")
        if not rec["geometry_changed"]:
            try:
                orig = ds.load(rel)
                for a, b in zip(orig.parts, back.parts):
                    if a.vertices.shape != b.vertices.shape or not np.allclose(a.vertices, b.vertices, rtol=0, atol=1e-6):
                        problems.append(f"仅纹理修复却改变了几何体 {a.index} 的顶点")
                    if not np.array_equal(a.faces, b.faces):
                        problems.append(f"仅纹理修复却改变了几何体 {a.index} 的三角形")
                    if (a.uvs is None) != (b.uvs is None) or (a.uvs is not None and not np.allclose(a.uvs, b.uvs, rtol=0, atol=1e-7)):
                        problems.append(f"仅纹理修复却改变了几何体 {a.index} 的 UV")
            except Exception as e:  # noqa: BLE001
                problems.append(f"无法读取原文件对比: {e}")
        rec["readback"] = {"ok": not problems, "problems": problems}

    def _protected_check(self, ds: Dataset, output_path: Path, protected, rels: list[str]) -> list[dict]:
        out_ds = self._output_dataset(ds, output_path)
        results = []
        for i, (plo, phi) in enumerate(protected):
            stat = {"index": i, "files": 0, "faces_before": 0, "faces_after": 0, "texels": 0, "texels_changed": 0, "mean_abs_diff": 0.0}
            diffs = []
            for rel in rels:
                try:
                    a = ds.load(rel)
                    b = out_ds.load(rel)
                except Exception as e:  # noqa: BLE001
                    stat.setdefault("errors", []).append(f"{rel}: {e}")
                    continue
                touched = False
                for pa, pb in zip(a.parts, b.parts):
                    ina = np.any(np.all((pa.vertices[pa.faces] >= plo) & (pa.vertices[pa.faces] <= phi), axis=2), axis=1)
                    inb = np.any(np.all((pb.vertices[pb.faces] >= plo) & (pb.vertices[pb.faces] <= phi), axis=2), axis=1)
                    if ina.any() or inb.any():
                        touched = True
                    stat["faces_before"] += int(ina.sum())
                    stat["faces_after"] += int(inb.sum())
                    ta, tb = a.texture_of(pa), b.texture_of(pb)
                    if ta is None or tb is None or pa.uvs is None or not ina.any():
                        continue
                    tm = texel_map(pa, ta, face_ok=ina)
                    sel = np.all((tm.positions >= plo) & (tm.positions <= phi), axis=1)
                    if not sel.any() or ta.image.shape != tb.image.shape:
                        continue
                    d = np.abs(ta.image[tm.rows[sel], tm.cols[sel], :3].astype(int) - tb.image[tm.rows[sel], tm.cols[sel], :3].astype(int)).max(axis=1)
                    diffs.append(d)
                if touched:
                    stat["files"] += 1
            if diffs:
                d = np.concatenate(diffs)
                stat["texels"] = int(len(d))
                stat["texels_changed"] = int((d > 20).sum())
                stat["mean_abs_diff"] = round(float(d.mean()), 2)
            # JPEG re-encoding of a modified atlas adds small noise everywhere; a real edit shows up as large differences
            stat["ok"] = stat["faces_before"] == stat["faces_after"] and (stat["texels"] == 0 or stat["texels_changed"] <= 0.005 * stat["texels"])
            results.append(stat)
        return results

    def _input_check(self, ds: Dataset, work_dir: Path) -> dict:
        mpath = Path(work_dir) / "input_manifest.json"
        if not mpath.is_file():
            return {"ok": None, "note": "未记录输入哈希 (未执行导入检查阶段)"}
        from .io.inputs import verify_manifest

        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        res = verify_manifest(manifest, ds.root)
        return {k: res[k] for k in ("ok", "changed", "missing", "added") if k in res}

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
        """Re-run detection around the repaired regions (an auxiliary check only)."""
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
        res = SignDetector(self.cfg.detection).detect(near, ctx=self.ctx)
        out = []
        for rr in res.regions:
            if any(np.linalg.norm(rr.center - r.center) < r.size + 0.5 for r in regions):
                out.append(rr.to_dict())
        return out

    # =====================================================================
    # fully automatic run (CLI "process", tests)
    # =====================================================================
    def run(self, input_path, output_path, work_dir: Optional[str | Path] = None) -> dict:
        t0 = time.time()
        output_path = Path(output_path)
        work_dir = Path(work_dir) if work_dir else output_path.parent / (output_path.name + "_s3d")
        work_dir.mkdir(parents=True, exist_ok=True)
        ctx = self.ctx
        ctx.window(0.0, 0.1)
        ds, scan = self.scan(input_path, work_dir)
        ctx.window(0.1, 0.45)
        detection = self.detect(ds, work_dir)
        ctx.window(0.45, 1.0)
        report = {"input": str(input_path), "output": str(output_path), "warnings": [], "regions": []}
        if detection.get("stopped"):
            report["stopped"] = detection["stopped"]
        if detection.get("stopped") is None:
            rep = self.repair(ds, detection, None, output_path, work_dir, auto=True)
            report.update({k: v for k, v in rep.items() if k not in ("input", "output")})
        report.update(
            format=ds.kind,
            files=scan["files"],
            files_ok=scan.get("files_ok"),
            tiles=scan["tiles"],
            leaf_files=scan["leaf_files"],
            scan_blocking_errors=scan.get("blocking_errors", []),
            detection_seconds=detection.get("seconds"),
            candidates=len(detection["candidates"]),
            pending=[{"id": c["id"], "label": c.get("label") or c["category_label"], "reasons": c["review_reasons"]} for c in detection["candidates"] if c["status"] != "auto"],
            seconds=round(time.time() - t0, 1),
        )
        report.setdefault("residual_detections", [])
        report.setdefault("summary", {})
        for k in ("files_modified", "texels_painted", "faces_removed", "faces_added"):
            report.setdefault(k, 0)
        _write_json(work_dir / "report.json", report)
        return report


def _environment() -> dict:
    env: dict = {}
    try:
        from . import __version__

        env["sensitive3d"] = __version__
    except ImportError:
        pass
    try:
        from .fonts import fonts_report  # type: ignore[attr-defined]

        env["fonts"] = fonts_report()
    except Exception:  # noqa: BLE001 - optional module
        env["fonts"] = None
    try:
        from .io.osgb import bridge_version  # type: ignore[attr-defined]

        env["bridge"] = bridge_version()
    except Exception:  # noqa: BLE001
        env["bridge"] = None
    return env


def _atlas_residual(rel: str, mesh: MeshFile, edits: dict, targets: dict) -> list[dict]:
    """Sign-coloured pixels left in texels that were repainted or freed (privacy check of the atlas)."""
    from .core.raster import rasterize_uv

    out = []
    by_tex: dict[int, np.ndarray] = {}
    for pi, e in edits.items():
        part = mesh.parts[pi]
        if part.texture is None or not e.removed_uv_tris:
            continue
        tex = mesh.textures[part.texture]
        tris = np.concatenate(e.removed_uv_tris)
        if not len(tris):
            continue
        m = rasterize_uv(tris.reshape(-1, 2), np.arange(len(tris) * 3).reshape(-1, 3), tex.width, tex.height).valid
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # include gutters
        by_tex[part.texture] = by_tex.get(part.texture, np.zeros_like(m)) | m
    for pi, t in targets.items():
        part = mesh.parts[pi]
        if part.texture is None:
            continue
        tex = mesh.textures[part.texture]
        m = by_tex.get(part.texture, np.zeros((tex.height, tex.width), bool))
        m[t.rows, t.cols] = True
        by_tex[part.texture] = m
    for ti, m in by_tex.items():
        tex = mesh.textures[ti]
        if m.shape != tex.image.shape[:2] or not m.any():
            continue
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        masks = color_masks(tex.image[y0:y1, x0:x1, :3], loose=True)
        sign = np.zeros((y1 - y0, x1 - x0), bool)
        for c in SIGN_COLORS:
            sign |= masks[c]
        sub = m[y0:y1, x0:x1]
        n = int(sub.sum())
        k = int((sign & sub).sum())
        out.append({"rel": rel, "texture": ti, "texels_checked": n, "sign_colour_texels": k, "sign_colour_fraction": round(k / max(n, 1), 4)})
    return out


def run(input_path, output_path, config: Optional[PipelineConfig] = None, progress: Optional[ProgressFn] = None, log=None, work_dir=None, cancel_event=None) -> dict:
    """Fully automatic: scan, detect, repair the ``auto`` candidates."""
    return Pipeline(config, progress, log, cancel_event).run(input_path, output_path, work_dir)


__all__ = ["ALL_CATEGORIES", "Pipeline", "PipelineConfig", "ReviewError", "ScanBlocked", "run"]
