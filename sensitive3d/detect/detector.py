"""Fully automatic sign detection on the finest level of detail.

1. candidates: 3D clusters of sign-coloured texels (and, when a learned
   model is configured, detections in oblique "sweep" views);
2. verification: each candidate is rendered head-on (orthographic close-up)
   and passed to the 2D detectors;
3. lifting: the 2D sign mask is mapped back to 3D surface points and fitted
   with an oriented plate region; duplicates are merged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..core.camera import OrthoCamera
from ..core.mesh import MeshPart, Texture
from ..core.render import RenderItem, RenderResult, render
from .candidates import Candidate, find_candidates
from .heuristic import Detection2D, HeuristicSignDetector
from .regions import SignRegion, fit_plane, merge_regions, plane_axes, region_from_points

ALL_CATEGORIES = (
    "speed_limit",
    "height_limit",
    "weight_limit",
    "width_limit",
    "prohibitory",
    "road_name",
    "guide",
    "warning",
    "other_sign",
)


@dataclass
class DetectionConfig:
    categories: Sequence[str] = ALL_CATEGORIES
    min_score: float = 0.55
    yolo_model: Optional[str] = None
    yolo_conf: float = 0.35
    sweep: Optional[bool] = None  # default: on when a learned model is configured
    sweep_pixel_size: float = 0.03
    closeup_resolution: int = 360


@dataclass
class Evidence:
    camera: OrthoCamera
    detection: Detection2D
    closeup: Optional[np.ndarray] = None  # rendered close-up (RGB) the 2D detector saw
    candidate_color: str = ""


@dataclass
class DetectionResult:
    regions: list[SignRegion]
    candidates: int = 0
    evidence: dict = field(default_factory=dict)  # region id -> Evidence
    rejected: list = field(default_factory=list)  # per candidate diagnostics (when requested)


def _items_near(items: Sequence[RenderItem], lo: np.ndarray, hi: np.ndarray) -> list[RenderItem]:
    out = []
    for it in items:
        a, b = it.part.bounds()
        if np.all(a <= hi) and np.all(b >= lo):
            out.append(it)
    return out


def closeup_camera(points: np.ndarray, normals: np.ndarray, resolution: int = 360) -> OrthoCamera:
    c, n, _ = fit_plane(points)
    mean_n = normals.mean(axis=0)
    if np.linalg.norm(mean_n) > 0.3 and np.dot(n, mean_n) < 0:
        n = -n
    if np.linalg.norm(mean_n) > 0.3 and abs(np.dot(n, mean_n / np.linalg.norm(mean_n))) < 0.5:
        n = mean_n / np.linalg.norm(mean_n)  # not a clean plane: look along the mean surface normal
    u, v = plane_axes(n)
    d = points - c
    ext = max(np.ptp(d @ u), np.ptp(d @ v), 0.2)
    size = ext * 1.6 + 0.4
    ps = float(np.clip(size / resolution, 0.004, 0.03))
    target = c + ((d @ u).max() + (d @ u).min()) / 2 * u + ((d @ v).max() + (d @ v).min()) / 2 * v
    return OrthoCamera.looking(target, -n, size, size, ps, up_hint=v, near=-1.5, far=1.0)


def lift_detection(res: RenderResult, det: Detection2D, fallback_points: Optional[np.ndarray] = None) -> tuple[Optional[SignRegion], dict]:
    """Map a 2D sign mask to an oriented 3D plate region.

    Returns ``(region, stats)``.  ``stats["status"]`` is ``ok``, ``non_planar``
    (region kept, needs review), ``mapping_failed`` (too few mapped pixels; a
    provisional region is fitted to ``fallback_points`` when given, needs
    review) or ``rejected`` (implausible for a traffic sign: size, lying flat;
    region is None).
    """
    m = det.mask & res.valid
    stats = {"mask_pixels": int(det.mask.sum()), "mapped_pixels": int(m.sum()), "ok": False, "provisional": False}

    def provisional(reason: str):
        stats.update(status="mapping_failed", reason=reason)
        if fallback_points is None or len(fallback_points) < 12:
            return None, stats
        region = region_from_points(fallback_points, view_dir=res.camera.forward, category=det.category, score=det.score, label=det.label)
        if region is not None:
            stats["provisional"] = True
            region.mapping = dict(stats)
        return region, stats

    if m.sum() < 12:
        return provisional(f"掩膜内只有 {int(m.sum())} 个像素落在模型表面")
    pts = res.position[m]
    # drop pixels that hit something far behind / in front of the sign face
    depth = res.depth[m]
    med = np.median(depth)
    keep = np.abs(depth - med) < 0.35
    stats["depth_kept"] = int(keep.sum())
    pts = pts[keep]
    if len(pts) < 12:
        return provisional("深度一致的像素过少")
    region = region_from_points(pts, view_dir=res.camera.forward, category=det.category, score=det.score, label=det.label)
    if region is None:
        return provisional("平面拟合失败")
    region.shape = "disk" if det.category in ("speed_limit", "height_limit", "weight_limit", "width_limit", "prohibitory") else "rect"
    size = region.size
    stats.update(size=round(float(size), 3), normal_z=round(float(region.normal[2]), 3))
    if not (0.15 <= size <= 8.0) or min(region.half_u, region.half_v) < 0.05:
        stats.update(status="rejected", reason=f"尺寸 {size:.2f} m 不像交通标志")
        return None, stats
    # traffic signs are (near) vertical plates; flat colour patches on the ground are road paint
    if abs(region.normal[2]) > 0.6:
        stats.update(status="rejected", reason="平躺在地面上 (路面标线)")
        return None, stats
    q = region.local(pts)
    p80 = float(np.percentile(np.abs(q[:, 2]), 80))
    stats["plane_p80"] = round(p80, 4)
    if p80 > max(0.06, 0.12 * size):
        stats.update(status="non_planar", reason=f"牌面不平整 (80% 偏差 {p80:.3f} m)")
        region.mapping = dict(stats)
        return region, stats
    stats.update(status="ok", ok=True, reason="")
    region.mapping = dict(stats)
    return region, stats


class SignDetector:
    def __init__(self, config: Optional[DetectionConfig] = None, log: Callable[[str], None] = lambda s: None):
        self.config = config or DetectionConfig()
        self.log = log
        self.detectors = [HeuristicSignDetector()]
        if self.config.yolo_model:
            from .yolo import YoloOnnxDetector

            self.detectors.append(YoloOnnxDetector(self.config.yolo_model, conf=self.config.yolo_conf))
        self.sweep = self.config.sweep if self.config.sweep is not None else bool(self.config.yolo_model)

    def detect_2d(self, img, valid, pixel_size, diagnostics: Optional[list] = None) -> list[Detection2D]:
        dets = []
        for d in self.detectors:
            if diagnostics is not None and d.name == "heuristic":
                dets.extend(d.detect(img, valid, pixel_size, diagnostics=diagnostics))
            else:
                dets.extend(d.detect(img, valid, pixel_size))
        return dets

    # -- sweep views (learned detector only) ---------------------------------
    def _sweep_candidates(self, items: Sequence[RenderItem]) -> list[Candidate]:
        learned = [d for d in self.detectors if d.name != "heuristic"]
        if not learned:
            return []
        bounds = [it.part.bounds() for it in items if len(it.part.faces)]
        lo = np.min([b[0] for b in bounds], axis=0)
        hi = np.max([b[1] for b in bounds], axis=0)
        center = (lo + hi) / 2
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        out = []
        el = np.radians(20.0)
        for az in np.radians(np.arange(0, 360, 45)):
            fwd = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), -np.sin(el)])
            cam0 = OrthoCamera.looking(center, fwd, 1, 1, self.config.sweep_pixel_size)
            q = corners - center
            w = np.ptp(q @ cam0.right) + 1
            h = np.ptp(q @ cam0.up) + 1
            dmin, dmax = (q @ cam0.forward).min(), (q @ cam0.forward).max()
            step = 15.0
            for near in np.arange(dmin, dmax, step):
                cam = OrthoCamera.looking(center, fwd, w, h, self.config.sweep_pixel_size, near=near - 1.0, far=near + step + 1.0, max_pixels=6000)
                res = render(items, cam, cull_backfaces=True)
                for det in self._tiled(learned, res):
                    m = det.mask & res.valid
                    if m.sum() < 8:
                        continue
                    pts = res.position[m]
                    nrm = res.normal[m]
                    out.append(Candidate("learned", pts, nrm, pts.mean(axis=0), np.ptp(pts, axis=0)))
        return out

    def _tiled(self, detectors, res: RenderResult, tile: int = 640, overlap: int = 128) -> list[Detection2D]:
        H, W = res.valid.shape
        dets = []
        for y in range(0, max(H - overlap, 1), tile - overlap):
            for x in range(0, max(W - overlap, 1), tile - overlap):
                sl = (slice(y, min(y + tile, H)), slice(x, min(x + tile, W)))
                if not res.valid[sl].any():
                    continue
                for d in detectors:
                    for det in d.detect(res.color[sl], res.valid[sl], res.camera.pixel_size):
                        full = np.zeros((H, W), bool)
                        full[sl] = det.mask
                        det.mask = full
                        dets.append(det)
        return dets

    # -- main entry ----------------------------------------------------------
    def detect(self, parts: Sequence[tuple[MeshPart, Texture]], ctx=None, diagnostics: bool = False) -> DetectionResult:
        """Detect signs on the given (finest level) parts.

        All categories are reported; choosing what to remove is the review
        step's job.  ``ctx`` (a :class:`sensitive3d.runctx.RunContext`) is
        checked between candidates.  With ``diagnostics`` every colour
        candidate that produced no sign keeps its close-up and the 2D
        detector's per-component decisions in ``DetectionResult.rejected``.
        """
        cfg = self.config
        items = [RenderItem(p, t.image if t is not None else None) for p, t in parts]
        cands = find_candidates([(p, t) for p, t in parts if t is not None])
        if self.sweep:
            cands += self._sweep_candidates(items)
        self.log(f"  {len(cands)} candidates")
        regions: list[SignRegion] = []
        evidence: list[Evidence] = []
        rejected: list[dict] = []
        for ci, cand in enumerate(cands):
            if ctx is not None:
                ctx.check("检测候选区域")
            cam = closeup_camera(cand.points, cand.normals, cfg.closeup_resolution)
            reach = np.array([cam.width, cam.height, 0]).max() * cam.pixel_size
            near_items = _items_near(items, cand.center - reach - 1.5, cand.center + reach + 1.5)
            res = render(near_items, cam, cull_backfaces=False)
            diag: Optional[list] = [] if diagnostics else None
            lifts = []
            kept = 0
            for det in self.detect_2d(res.color, res.valid, cam.pixel_size, diag):
                if det.score < cfg.min_score:
                    lifts.append({"category": det.category, "score": det.score, "status": "rejected", "reason": f"置信度 {det.score:.2f} < {cfg.min_score}"})
                    continue
                region, stats = lift_detection(res, det, cand.points)
                lifts.append({"category": det.category, "score": round(det.score, 3), **stats})
                if region is None:
                    continue
                if stats["status"] == "mapping_failed":
                    region.review_reasons.append("二维→三维映射失败: " + stats.get("reason", ""))
                elif stats["status"] == "non_planar":
                    region.review_reasons.append(stats.get("reason", "牌面不平整"))
                region.sources.append(len(evidence))
                evidence.append(Evidence(cam, det, res.color, cand.color))
                regions.append(region)
                kept += 1
            if diagnostics and kept == 0:
                rejected.append(
                    {
                        "candidate": ci,
                        "color": cand.color,
                        "center": [round(float(x), 3) for x in cand.center],
                        "extent": [round(float(x), 3) for x in cand.extent],
                        "points": int(len(cand.points)),
                        "camera": cam.to_dict(),
                        "closeup": res.color,
                        "components": diag or [],
                        "lifts": lifts,
                    }
                )
        merged = merge_regions(regions)
        ev = {r.id: evidence[r.sources[0]] for r in merged if r.sources}
        for r in merged:
            r.sources = []
        return DetectionResult(merged, len(cands), ev, rejected)
