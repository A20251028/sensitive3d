"""Texture repair: synthesise the surface behind / around removed signs.

For each region one or more *repair views* are rendered from the edited
finest LOD: orthographic images looking straight at the surfaces that need
new content (fill patches and residual sign texels).  Each view is
inpainted once, and every LOD then samples its target texels from these
views, so all levels of detail get the same, consistent result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from ..core.camera import OrthoCamera
from ..core.mesh import MeshFile, MeshPart, Texture
from ..core.raster import rasterize_uv
from ..core.render import RenderItem, render
from ..core.texture import fill_from_nearest, sample_bilinear_px, texel_map, texel_size
from ..detect.regions import SignRegion, plane_axes
from .geometry import PartEdit, RegionGeometry
from .inpaint import inpaint


@dataclass
class TextureConfig:
    method: str = "auto"
    lama_model: Optional[str] = None
    residual_margin: float = 0.06  # extra margin around the plate for residual sign texels (m)
    context: float = 0.8  # context around the targets in repair views (m, at least)
    depth_tolerance: float = 0.12  # finest LOD texel vs repair view surface (m)
    coarse_depth_tolerance: float = 0.6  # coarser LODs deviate more from the finest surface


@dataclass
class RepairView:
    region_id: int
    camera: OrthoCamera
    before: np.ndarray
    image: np.ndarray
    mask: np.ndarray
    valid: np.ndarray
    depth: np.ndarray


@dataclass
class Target:
    """Texels of one part that must be (re)painted."""

    rows: np.ndarray
    cols: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    region: np.ndarray  # region id per texel


def residual_mask(points: np.ndarray, region: SignRegion, cfg: TextureConfig, extra: float = 0.0) -> np.ndarray:
    return region.contains(points, margin=cfg.residual_margin + extra, front=0.12 + extra, back=0.3 + extra)


def part_targets(
    part: MeshPart,
    tex: Texture,
    edit: Optional[PartEdit],
    regions: Sequence[SignRegion],
    cfg: TextureConfig,
) -> Optional[Target]:
    """New fill faces + texels of remaining faces lying inside a sign region."""
    if part.uvs is None or len(part.faces) == 0:
        return None
    lo, hi = part.bounds()
    near = []
    grow = 2.0 * texel_size(part, tex) if part.has_finer else 0.0
    for r in regions:
        rlo, rhi = r.aabb(margin=cfg.residual_margin + 0.05 + grow, front=0.15 + grow, back=0.35 + grow)
        if np.all(hi >= rlo) and np.all(lo <= rhi):
            near.append(r)
    new_faces = np.zeros(len(part.faces), bool)
    if edit is not None and len(edit.new_faces):
        new_faces[edit.new_faces] = True
    if not near and not new_faces.any():
        return None
    # limit the UV rasterization to faces close to any region
    c = part.centroids()
    close = new_faces.copy()
    for r in near:
        close |= r.contains(c, margin=cfg.residual_margin + 0.3 + grow, front=0.5 + grow, back=0.6 + grow)
    if not close.any():
        return None
    tm = texel_map(part, tex, face_ok=close)
    if len(tm) == 0:
        return None
    # coarse levels: texels are large and the surface deviates from the
    # finest one, so grow the residual box by a couple of texels
    extra = 2.0 * texel_size(part, tex) if part.has_finer else 0.0
    reg = np.full(len(tm), -1, np.int64)
    for r in near:
        reg[(reg < 0) & residual_mask(tm.positions, r, cfg, extra)] = r.id
    is_new = new_faces[tm.faces]
    if is_new.any() and regions:
        # fill faces belong to the nearest region
        centers = np.array([r.center for r in regions])
        ids = np.array([r.id for r in regions])
        d = np.linalg.norm(tm.positions[is_new][:, None, :] - centers[None], axis=2)
        reg[is_new] = ids[np.argmin(d, axis=1)]
    sel = reg >= 0
    if not sel.any():
        return None
    fn = part.face_normals()[tm.faces[sel]]
    return Target(tm.rows[sel], tm.cols[sel], tm.positions[sel], fn, reg[sel])


def _view_specs(region: SignRegion, info: Optional[RegionGeometry], targets_pts: np.ndarray, cfg: TextureConfig):
    """Repair view placements: a front view of the plate area and, for signs on
    a pole, a top-down view of the patched ground at the pole base."""
    specs = []
    n = region.normal
    u, v = plane_axes(n)
    w = 2 * region.half_u + 2 * max(cfg.context, 0.75 * region.size)
    h = 2 * region.half_v + 2 * max(cfg.context, 0.75 * region.size)
    specs.append(("front", region.center, -n, w, h, v, -0.35, 0.45))
    if info is not None and info.mount == "pole" and info.ground_z is not None:
        xy = info.pole_xy if info.pole_xy is not None else region.center[:2]
        base = np.array([xy[0], xy[1], info.ground_z])
        size = 2 * (0.5 + max(cfg.context, 0.6))
        # include every target near the ground
        if len(targets_pts):
            low = targets_pts[targets_pts[:, 2] < info.ground_z + 0.6]
            if len(low):
                size = max(size, 2 * (np.abs(low[:, :2] - xy).max() + max(cfg.context, 0.6)))
        specs.append(("ground", base, np.array([0.0, 0.0, -1.0]), size, size, np.array([0.0, 1.0, 0.0]), -0.6, 0.6))
    return specs


def build_views(
    region: SignRegion,
    targets: list[tuple[MeshPart, Target]],
    items: Sequence[RenderItem],
    cfg: TextureConfig,
    texel: float = 0.02,
    info: Optional[RegionGeometry] = None,
    log=lambda s: None,
) -> list[RepairView]:
    """Render + inpaint the repair views of one region from the edited finest LOD.

    ``texel`` is the texel size (m) of the finest LOD around the region; the
    views are rendered slightly finer than that.
    """
    pts = [t.positions[t.region == region.id] for _, t in targets]
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    ps = float(np.clip(texel * 0.75, 0.004, 0.05))
    views = []
    for kind, center, fwd, w, h, up, near, far in _view_specs(region, info, P, cfg):
        cam = OrthoCamera.looking(center, fwd, w, h, ps, up_hint=up, near=near, far=far, max_pixels=1600)
        reach = max(w, h) + 1.0
        lo, hi = center - reach, center + reach
        near_items = [it for it in items if np.all(it.part.bounds()[0] <= hi) and np.all(it.part.bounds()[1] >= lo)]
        res = render(near_items, cam, cull_backfaces=True)
        mask = np.zeros(res.valid.shape, bool)
        if len(P):
            xy, depth = cam.project(P)
            inb = (xy[:, 0] >= 0) & (xy[:, 0] < cam.width) & (xy[:, 1] >= 0) & (xy[:, 1] < cam.height) & (depth > cam.near) & (depth < cam.far)
            mask[xy[inb, 1].astype(int), xy[inb, 0].astype(int)] = True
        inside = res.valid.copy()
        inside[res.valid] = region.contains(res.position[res.valid], margin=cfg.residual_margin, front=0.12, back=0.3)
        mask |= inside
        if not mask.any():
            continue
        k = max(int(round(0.03 / cam.pixel_size)), 1)
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((2 * k + 1,) * 2, np.uint8)).astype(bool)
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        valid = res.valid | mask  # target pixels without a rendered surface (holes) are synthesised too
        out = inpaint(res.color.copy(), mask, valid, method=cfg.method, lama_model=cfg.lama_model)
        # synthesised pixels without a rendered surface keep depth NaN: any
        # target texel projecting there is accepted
        views.append(RepairView(region.id, cam, res.color, out, mask, valid, res.depth.copy()))
        log(f"    region {region.id}: {kind} view {cam.width}x{cam.height} px, {int(mask.sum())} px synthesised")
    return views


def paint_targets(
    mesh: MeshFile,
    targets: dict[int, Target],
    views: dict[int, list[RepairView]],
    coarse: bool,
    cfg: TextureConfig,
) -> int:
    """Write target texels of ``mesh`` from the repair views; returns texels painted."""
    painted = 0
    tol = cfg.coarse_depth_tolerance if coarse else cfg.depth_tolerance
    leftovers: dict[int, list] = {}
    painted_by_tex: dict[int, list] = {}
    for pi, t in targets.items():
        part = mesh.parts[pi]
        tex = mesh.texture_of(part)
        if tex is None:
            continue
        colors = np.zeros((len(t.rows), 3))
        best = np.full(len(t.rows), -np.inf)
        # pass 1: views that see the surface; pass 2: also grazing surfaces
        # (plate rims, blobby coarse faces) as long as the depth matches
        for min_facing in (0.2, -0.35):
            todo = ~np.isfinite(best)
            if not todo.any():
                break
            for rid in np.unique(t.region[todo]):
                sel = np.nonzero(todo & (t.region == rid))[0]
                for view in views.get(int(rid), []):
                    cam = view.camera
                    xy, depth = cam.project(t.positions[sel])
                    facing = -(t.normals[sel] @ cam.forward)
                    inb = (xy[:, 0] >= 0) & (xy[:, 0] < cam.width) & (xy[:, 1] >= 0) & (xy[:, 1] < cam.height)
                    ok = inb & (facing > min_facing)
                    if not ok.any():
                        continue
                    xi = np.clip(xy[:, 0].astype(int), 0, cam.width - 1)
                    yi = np.clip(xy[:, 1].astype(int), 0, cam.height - 1)
                    vd = view.depth[yi, xi]
                    synth = view.mask[yi, xi] & ~np.isfinite(vd)
                    depth_ok = synth | (np.isfinite(vd) & (np.abs(vd - depth) < tol))
                    ok &= view.valid[yi, xi] & depth_ok
                    better = ok & (facing > best[sel])
                    if better.any():
                        idx = sel[better]
                        colors[idx] = sample_bilinear_px(view.image, xy[better])[:, :3]
                        best[idx] = facing[better]
        done = np.isfinite(best)
        if done.any():
            img = tex.image
            img[t.rows[done], t.cols[done], :3] = np.clip(colors[done], 0, 255).astype(np.uint8)
            tex.dirty = True
            painted += int(done.sum())
            painted_by_tex.setdefault(part.texture, []).append((t.rows[done], t.cols[done]))
        if (~done).any():
            leftovers.setdefault(part.texture, []).append((t.rows[~done], t.cols[~done]))
    # bilinear filtering also reads the gutter texels around thin charts:
    # give uncovered texels next to painted ones the repaired colour
    for ti, chunks in painted_by_tex.items():
        tex = mesh.textures[ti]
        rows = np.concatenate([c[0] for c in chunks])
        cols = np.concatenate([c[1] for c in chunks])
        pad = 4
        y0, y1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, tex.height)
        x0, x1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, tex.width)
        P = np.zeros((y1 - y0, x1 - x0), bool)
        P[rows - y0, cols - x0] = True
        covered = np.zeros_like(P)
        for p in mesh.parts:
            if p.texture == ti and p.uvs is not None and len(p.faces):
                uvpx = tex.uv_to_px(p.uvs) - np.array([x0, y0])
                covered |= rasterize_uv(uvpx, p.faces, x1 - x0, y1 - y0).valid
        ring = cv2.dilate(P.astype(np.uint8), np.ones((2 * pad + 1,) * 2, np.uint8)).astype(bool) & ~covered & ~P
        if ring.any():
            sub = tex.image[y0:y1, x0:x1]
            fill_from_nearest(sub, P, ring)
            tex.image[y0:y1, x0:x1] = sub
    # texels no view could see: inpaint in texture space
    for ti, chunks in leftovers.items():
        tex = mesh.textures[ti]
        m = np.zeros((tex.height, tex.width), bool)
        for r, c in chunks:
            m[r, c] = True
        if not m.any():
            continue
        ys, xs = np.nonzero(m)
        pad = 24
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, tex.height)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, tex.width)
        crop = np.ascontiguousarray(tex.image[y0:y1, x0:x1, :3])
        tex.image[y0:y1, x0:x1, :3] = cv2.inpaint(crop, m[y0:y1, x0:x1].astype(np.uint8) * 255, 5, cv2.INPAINT_TELEA)
        tex.dirty = True
        painted += int(m.sum())
    return painted


def wipe_freed_texels(mesh: MeshFile, edits: dict[int, PartEdit], pad: int = 4) -> None:
    """Overwrite atlas texels that belonged to deleted faces (no sign pixels may survive)."""
    by_tex: dict[int, list] = {}
    for pi, e in edits.items():
        part = mesh.parts[pi]
        if part.texture is None or not e.removed_uv_tris:
            continue
        by_tex.setdefault(part.texture, []).extend(e.removed_uv_tris)
    for ti, tri_lists in by_tex.items():
        tex = mesh.textures[ti]
        tris = np.concatenate(tri_lists)
        if len(tris) == 0:
            continue
        pts = tris.reshape(-1, 2)
        faces = np.arange(len(pts)).reshape(-1, 3)
        removed = rasterize_uv(pts, faces, tex.width, tex.height).valid
        # conservative: include the gutter around deleted charts
        removed = cv2.dilate(removed.astype(np.uint8), np.ones((2 * pad + 1,) * 2, np.uint8)).astype(bool)
        covered = np.zeros_like(removed)
        for p in mesh.parts:
            if p.texture == ti and p.uvs is not None and len(p.faces):
                covered |= rasterize_uv(tex.uv_to_px(p.uvs), p.faces, tex.width, tex.height).valid
        target = removed & ~covered
        if not target.any():
            continue
        ys, xs = np.nonzero(removed)
        y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 9, tex.height)
        x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 9, tex.width)
        sub = tex.image[y0:y1, x0:x1]
        fill_from_nearest(sub, covered[y0:y1, x0:x1], target[y0:y1, x0:x1])
        tex.image[y0:y1, x0:x1] = sub
        tex.dirty = True
