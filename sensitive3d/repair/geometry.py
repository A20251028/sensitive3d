"""Geometry removal: delete sign plates (and their poles) and close the holes.

For every region the faces of the sign plate are removed.  When the plate
stands free on a pole, the pole below it is removed down to the ground;
when it is mounted on a wall (surface right behind the plate) only the
plate goes.  Every hole opened by the removal is closed with a planar
patch that gets its own chart in the texture atlas (re-using the space
freed by the deleted faces when possible).  The texture of the patches is
synthesised later by :mod:`sensitive3d.repair.texture`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import mapbox_earcut
import numpy as np

from ..core.mesh import MeshFile, MeshPart, face_normals
from ..core.raster import rasterize_uv
from ..core.texture import texel_size
from ..detect.regions import SignRegion, plane_axes


@dataclass
class GeometryConfig:
    plate_margin: float = 0.05  # in-plane margin around the detected sign face (m)
    plate_front: float = 0.10
    plate_back: float = 0.20
    remove_poles: bool = True
    pole_margin: float = 0.25  # horizontal margin of the pole search prism around the plate (m)
    min_height: float = 0.05  # keep everything below ground + this (m)
    wall_gap: float = 0.5  # a surface within this distance behind the plate means "wall mounted"
    atlas_pad: int = 3


@dataclass
class RegionGeometry:
    """Per-region facts shared by all LODs (derived from the finest level)."""

    region_id: int
    mount: str = "unknown"  # pole | wall | free
    ground_z: Optional[float] = None
    pole_xy: Optional[np.ndarray] = None


@dataclass
class PartEdit:
    removed: int = 0
    new_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    removed_uv_tris: list = field(default_factory=list)  # uv triangles of deleted faces (to wipe)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def plate_face_mask(part: MeshPart, region: SignRegion, cfg: GeometryConfig) -> np.ndarray:
    if len(part.faces) == 0:
        return np.zeros(0, bool)
    c = part.centroids()
    return region.contains(c, margin=cfg.plate_margin, front=cfg.plate_front, back=cfg.plate_back)


def _footprint_dist(points_xy: np.ndarray, region: SignRegion) -> np.ndarray:
    """Horizontal distance from the plate's footprint segment."""
    u, _ = region.axes
    uh = np.array([u[0], u[1]])
    if np.linalg.norm(uh) < 1e-6:
        uh = np.array([1.0, 0.0])
    uh = uh / np.linalg.norm(uh)
    c = region.center[:2]
    d = points_xy - c
    t = np.clip(d @ uh, -region.half_u, region.half_u)
    return np.linalg.norm(d - t[:, None] * uh, axis=1)


def analyse_region(parts: list[MeshPart], region: SignRegion, cfg: GeometryConfig) -> RegionGeometry:
    """Classify the mounting of a sign and find the ground level (finest LOD)."""
    info = RegionGeometry(region.id)
    u, v = region.axes
    n = region.normal
    plate_bottom = region.center[2] - region.half_v * abs(v[2]) - region.half_u * abs(u[2])
    cents, norms, areas = [], [], []
    for p in parts:
        if len(p.faces):
            cents.append(p.centroids())
            norms.append(p.face_normals())
            areas.append(p.face_areas())
    if not cents:
        return info
    C, N, A = np.concatenate(cents), np.concatenate(norms), np.concatenate(areas)

    # wall: surface parallel to the plate right behind it, extending beyond its edges
    q = region.local(C)
    ring = (
        (np.abs(q[:, 0]) < region.half_u + 0.4)
        & (np.abs(q[:, 1]) < region.half_v + 0.4)
        & ((np.abs(q[:, 0]) > region.half_u + cfg.plate_margin) | (np.abs(q[:, 1]) > region.half_v + cfg.plate_margin))
        & (q[:, 2] < 0.05)
        & (q[:, 2] > -cfg.wall_gap)
        & (N @ n > 0.7)
    )
    ring_area = A[ring].sum()
    expected = (2 * region.half_u + 0.8) * (2 * region.half_v + 0.8) - (2 * region.half_u) * (2 * region.half_v)
    if ring_area > 0.35 * expected:
        info.mount = "wall"
        return info

    # ground level around the footprint
    dist = _footprint_dist(C[:, :2], region)
    up = (N[:, 2] > 0.8) & (dist < cfg.pole_margin + 1.2) & (C[:, 2] < plate_bottom - 0.4)
    if up.any():
        info.ground_z = float(np.percentile(C[up, 2], 50))
    else:
        below = (dist < cfg.pole_margin) & (C[:, 2] < plate_bottom)
        info.ground_z = float(C[below, 2].min()) if below.any() else None
    if info.ground_z is None or not cfg.remove_poles:
        info.mount = "free"
        return info

    # pole: something inside the prism under the plate but nothing around it
    mid = (C[:, 2] > info.ground_z + 0.4) & (C[:, 2] < plate_bottom - 0.15)
    inside = mid & (dist < cfg.pole_margin)
    around = mid & (dist >= cfg.pole_margin) & (dist < cfg.pole_margin + 0.4)
    if A[inside].sum() > 0.005 and A[around].sum() < 0.5 * A[inside].sum() + 0.02:
        info.mount = "pole"
        sel = inside
        info.pole_xy = np.average(C[sel, :2], axis=0, weights=A[sel])
    else:
        info.mount = "free"
    return info


def _object_points_mask(points: np.ndarray, region: SignRegion, info: RegionGeometry, cfg: GeometryConfig, slack: float = 0.0) -> np.ndarray:
    mask = region.contains(points, margin=cfg.plate_margin + slack, front=cfg.plate_front + slack, back=cfg.plate_back + slack)
    if info.mount == "pole" and info.ground_z is not None:
        u, v = region.axes
        plate_top = region.center[2] + region.half_v * abs(v[2]) + region.half_u * abs(u[2])
        dist = _footprint_dist(points[:, :2], region)
        mask |= (dist < cfg.pole_margin + slack) & (points[:, 2] > info.ground_z + cfg.min_height - slack) & (points[:, 2] < plate_top + 0.1 + slack)
    return mask


def object_face_mask(part: MeshPart, region: SignRegion, info: RegionGeometry, cfg: GeometryConfig) -> np.ndarray:
    """Faces of the sign object (plate, plus pole when free standing).

    On the finest level a face belongs to the object when its centroid does.
    Coarser levels have large faces that reach far beyond the sign; there a
    face is only removed when all of its vertices lie in (a slightly grown)
    object volume, and wall-mounted plates are left to texture repair since
    they are flush with the wall at that resolution.
    """
    if len(part.faces) == 0:
        return np.zeros(0, bool)
    if not part.has_finer:
        mask = plate_face_mask(part, region, cfg)
        if info.mount == "pole" and info.ground_z is not None:
            c = part.centroids()
            u, v = region.axes
            plate_top = region.center[2] + region.half_v * abs(v[2]) + region.half_u * abs(u[2])
            dist = _footprint_dist(c[:, :2], region)
            mask |= (dist < cfg.pole_margin) & (c[:, 2] > info.ground_z + cfg.min_height) & (c[:, 2] < plate_top + 0.1)
        return mask
    if info.mount == "wall":
        return np.zeros(len(part.faces), bool)
    inside = _object_points_mask(part.vertices, region, info, cfg, slack=0.3)
    mask = inside[part.faces].all(axis=1)
    if info.mount == "pole" and info.ground_z is not None:
        # keep faces lying on the ground inside the prism
        c = part.centroids()
        mask &= c[:, 2] > info.ground_z + cfg.min_height
    return mask


# ---------------------------------------------------------------------------
# hole filling
# ---------------------------------------------------------------------------


def _edge_keys(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return e, np.sort(e, axis=1)


def _undirected_counts(faces: np.ndarray) -> dict:
    _, k = _edge_keys(faces)
    uniq, cnt = np.unique(k, axis=0, return_counts=True)
    return {(int(a), int(b)): int(c) for (a, b), c in zip(uniq, cnt)}


def new_boundary_loops(old_faces: np.ndarray, new_faces: np.ndarray, positions: np.ndarray) -> list[list[int]]:
    """Directed loops (in the orientation of the remaining faces) of holes opened by a removal.

    Vertices are welded by position first, because per-vertex UVs split
    the surface along chart seams.
    """
    if len(new_faces) == 0:
        return []
    _, weld = np.unique(np.round(positions / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    weld = weld.ravel()
    old_w = weld[old_faces]
    new_w = weld[new_faces]
    old_cnt = _undirected_counts(old_w)
    e, k = _edge_keys(new_w)
    uniq, inv, cnt = np.unique(k, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    is_b = cnt[inv] == 1
    # representative (unwelded) vertex for each welded id, taken from the remaining faces
    rep = {}
    for f in range(len(new_faces)):
        for j in range(3):
            rep.setdefault(int(new_w[f, j]), int(new_faces[f, j]))
    nxt: dict[int, list[int]] = {}
    for idx in np.nonzero(is_b)[0]:
        a, b = int(e[idx, 0]), int(e[idx, 1])
        key = (min(a, b), max(a, b))
        if old_cnt.get(key, 0) < 2:
            continue  # was already a border (tile edge) before the removal
        nxt.setdefault(a, []).append(b)
    loops = []
    used = set()
    for start in list(nxt.keys()):
        for first in nxt[start]:
            if (start, first) in used:
                continue
            loop = [start]
            used.add((start, first))
            cur = first
            ok = False
            for _ in range(100000):
                if cur == start:
                    ok = True
                    break
                loop.append(cur)
                cands = [b for b in nxt.get(cur, []) if (cur, b) not in used]
                if not cands:
                    break
                used.add((cur, cands[0]))
                cur = cands[0]
            if ok and len(loop) >= 3:
                loops.append([rep[w] for w in loop])
    return loops


def triangulate_loop(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Triangulate a 3D loop projected on the plane with ``normal``; returns local indices."""
    u, v = plane_axes(normal)
    p2 = np.stack([points @ u, points @ v], axis=1)
    tris = None
    try:
        idx = mapbox_earcut.triangulate_float64(p2.astype(np.float64), np.array([len(p2)], dtype=np.uint32))
        if len(idx) == 3 * (len(p2) - 2):
            tris = np.asarray(idx, dtype=np.int64).reshape(-1, 3)
    except Exception:  # pragma: no cover - earcut is robust, fan is the fallback
        tris = None
    if tris is None:
        k = len(p2)
        tris = np.stack([np.zeros(k - 2, np.int64), np.arange(1, k - 1), np.arange(2, k)], axis=1)
    return tris


# ---------------------------------------------------------------------------
# atlas space
# ---------------------------------------------------------------------------


def find_free_rect(occupied: np.ndarray, w: int, h: int) -> Optional[tuple[int, int]]:
    H, W = occupied.shape
    if w > W or h > H or w <= 0 or h <= 0:
        return None
    sat = np.zeros((H + 1, W + 1), np.int64)
    sat[1:, 1:] = occupied.astype(np.int64).cumsum(0).cumsum(1)
    s = sat[h:, w:] - sat[:-h, w:] - sat[h:, :-w] + sat[:-h, :-w]
    ys, xs = np.nonzero(s == 0)
    if len(ys) == 0:
        return None
    i = int(np.argmin(ys * W + xs))
    return int(xs[i]), int(ys[i])


def grow_texture(mesh: MeshFile, tex_index: int, extra_rows: int) -> None:
    """Append rows at the bottom of a texture, remapping v of every part that uses it."""
    tex = mesh.textures[tex_index]
    H = tex.height
    fill = tex.image.reshape(-1, tex.image.shape[2]).mean(axis=0).astype(np.uint8)
    pad = np.empty((extra_rows, tex.width, tex.image.shape[2]), np.uint8)
    pad[:] = fill
    tex.image = np.concatenate([tex.image, pad], axis=0)
    tex.dirty = True
    for p in mesh.parts:
        if p.texture == tex_index and p.uvs is not None:
            p.uvs[:, 1] = 1.0 - (1.0 - p.uvs[:, 1]) * H / (H + extra_rows)
            p.dirty = True


class AtlasAllocator:
    """Free-space bookkeeping for one texture of one mesh file."""

    def __init__(self, mesh: MeshFile, tex_index: int, pad: int = 3):
        self.mesh = mesh
        self.tex_index = tex_index
        self.pad = pad
        self.refresh()

    def refresh(self) -> None:
        tex = self.mesh.textures[self.tex_index]
        occ = np.zeros((tex.height, tex.width), bool)
        for p in self.mesh.parts:
            if p.texture == self.tex_index and p.uvs is not None and len(p.faces):
                occ |= rasterize_uv(tex.uv_to_px(p.uvs), p.faces, tex.width, tex.height).valid
        if self.pad > 0 and occ.any():
            import cv2

            occ = cv2.dilate(occ.astype(np.uint8), np.ones((2 * self.pad + 1,) * 2, np.uint8)).astype(bool)
        self.occupied = occ

    def allocate(self, w: int, h: int) -> tuple[int, int]:
        pos = find_free_rect(self.occupied, w, h)
        if pos is None:
            tex = self.mesh.textures[self.tex_index]
            grow = max(h, 16)
            grow_texture(self.mesh, self.tex_index, grow)
            self.occupied = np.concatenate([self.occupied, np.zeros((grow, tex.width), bool)], axis=0)
            pos = find_free_rect(self.occupied, w, h)
            if pos is None:
                raise RuntimeError("texture too narrow for a fill patch")
        x, y = pos
        self.occupied[y : y + h, x : x + w] = True
        return x, y


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def remove_objects(
    mesh: MeshFile,
    regions: list[SignRegion],
    infos: dict[int, RegionGeometry],
    cfg: Optional[GeometryConfig] = None,
) -> dict[int, PartEdit]:
    """Remove sign geometry of all ``regions`` from every part of ``mesh`` and fill holes.

    Returns an edit record per part index (position in ``mesh.parts``).
    """
    cfg = cfg or GeometryConfig()
    edits: dict[int, PartEdit] = {}
    allocators: dict[int, AtlasAllocator] = {}
    for pi, part in enumerate(mesh.parts):
        if len(part.faces) == 0:
            continue
        lo, hi = part.bounds()
        remove = np.zeros(len(part.faces), bool)
        for r in regions:
            info = infos.get(r.id, RegionGeometry(r.id))
            rlo, rhi = r.aabb(margin=cfg.pole_margin + 0.1, front=cfg.plate_front, back=cfg.plate_back)
            if info.mount == "pole" and info.ground_z is not None:
                rlo = rlo.copy()
                rlo[2] = min(rlo[2], info.ground_z - 0.5)
            if np.any(hi < rlo) or np.any(lo > rhi):
                continue
            remove |= object_face_mask(part, r, info, cfg)
        if not remove.any():
            continue
        tex = mesh.texture_of(part)
        edit = PartEdit(removed=int(remove.sum()))
        if tex is not None and part.uvs is not None:
            edit.removed_uv_tris = [tex.uv_to_px(part.uvs)[part.faces[remove]]]
        old_faces = part.faces
        kept = old_faces[~remove]
        loops = new_boundary_loops(old_faces, kept, part.vertices)
        fn_all = face_normals(part.vertices, kept) if len(kept) else np.zeros((0, 3))
        new_tris = []
        new_vert_pos = []
        new_vert_src = []
        new_uv = []
        base = len(part.vertices)
        if tex is not None and part.uvs is not None and part.texture not in allocators:
            allocators[part.texture] = AtlasAllocator(mesh, part.texture, cfg.atlas_pad)
        alloc = allocators.get(part.texture) if tex is not None and part.uvs is not None else None
        if alloc is not None:
            # deleted faces no longer occupy the atlas
            alloc.occupied &= _occupied_by(mesh, part.texture, exclude=(pi, remove), pad=cfg.atlas_pad)
        ts = texel_size(part, tex, ~remove) if tex is not None and part.uvs is not None else 0.02
        for loop in loops:
            loop = np.array(loop, dtype=np.int64)
            pts = part.vertices[loop]
            # orientation: mean normal of remaining faces touching the loop
            touch = np.isin(kept, loop).any(axis=1)
            nrm = fn_all[touch].mean(axis=0) if touch.any() else np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if np.linalg.norm(nrm) < 1e-9:
                continue
            nrm = nrm / np.linalg.norm(nrm)
            tri = triangulate_loop(pts, nrm)
            # consistent winding with the surrounding surface
            fnorm = np.cross(pts[tri[:, 1]] - pts[tri[:, 0]], pts[tri[:, 2]] - pts[tri[:, 0]])
            if (fnorm @ nrm).sum() < 0:
                tri = tri[:, ::-1]
            vbase = base + sum(len(x) for x in new_vert_pos)
            new_vert_pos.append(pts)
            new_vert_src.append(loop)
            new_tris.append(tri + vbase)
            if alloc is not None:
                u, v = plane_axes(nrm)
                p2 = np.stack([pts @ u, -(pts @ v)], axis=1)
                p2 -= p2.min(axis=0)
                scale = ts
                for _ in range(4):
                    w = int(np.ceil(p2[:, 0].max() / scale)) + 1 + 2 * cfg.atlas_pad
                    h = int(np.ceil(p2[:, 1].max() / scale)) + 1 + 2 * cfg.atlas_pad
                    if max(w, h) <= max(tex.width, 64) // 2:
                        break
                    scale *= 2
                x0, y0 = alloc.allocate(w, h)
                # keep pixel coordinates: the atlas may still grow (rows are
                # appended at the bottom, existing pixels keep their place)
                new_uv.append(p2 / scale + np.array([x0 + cfg.atlas_pad + 0.5, y0 + cfg.atlas_pad + 0.5]))
        if new_tris:
            pos = np.concatenate(new_vert_pos)
            src = np.concatenate(new_vert_src)
            part.vertices = np.concatenate([part.vertices, pos])
            if part.uvs is not None:
                if new_uv:
                    tex = mesh.texture_of(part)
                    uv_new = tex.px_to_uv(np.concatenate(new_uv))
                else:
                    uv_new = part.uvs[src]
                part.uvs = np.concatenate([part.uvs, uv_new])
            if part.colors is not None:
                part.colors = np.concatenate([part.colors, part.colors[src]])
            if part.normals is not None:
                part.normals = np.concatenate([part.normals, part.normals[src]])
            tris = np.concatenate(new_tris)
            part.faces = np.concatenate([kept, tris])
            edit.new_faces = np.arange(len(kept), len(part.faces))
        else:
            part.faces = kept
        if part.normals is not None:
            part.recompute_normals()
        part.dirty = True
        edits[pi] = edit
    return edits


def _occupied_by(mesh: MeshFile, tex_index: int, exclude: tuple[int, np.ndarray], pad: int) -> np.ndarray:
    """Occupancy of a texture by all faces except the excluded ones of one part."""
    import cv2

    tex = mesh.textures[tex_index]
    occ = np.zeros((tex.height, tex.width), bool)
    for pi, p in enumerate(mesh.parts):
        if p.texture != tex_index or p.uvs is None or len(p.faces) == 0:
            continue
        ok = None
        if pi == exclude[0]:
            ok = ~exclude[1]
            if len(ok) != len(p.faces):
                ok = None
        occ |= rasterize_uv(tex.uv_to_px(p.uvs), p.faces, tex.width, tex.height, face_ok=ok).valid
    if pad > 0 and occ.any():
        occ = cv2.dilate(occ.astype(np.uint8), np.ones((2 * pad + 1,) * 2, np.uint8)).astype(bool)
    return occ
