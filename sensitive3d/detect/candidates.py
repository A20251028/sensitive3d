"""Candidate generation: 3D clusters of sign-coloured texels."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..core.mesh import MeshPart, Texture
from ..core.texture import texel_map
from .heuristic import color_masks

SIGN_COLORS = ("red", "blue", "green", "yellow")


@dataclass
class Candidate:
    color: str
    points: np.ndarray  # sample of 3D points
    normals: np.ndarray  # face normals at those points
    center: np.ndarray
    extent: np.ndarray  # bounding box size


def colored_texels(part: MeshPart, tex: Texture) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """3D positions + face normals of sign-coloured texels, per colour."""
    masks = color_masks(tex.image, loose=True)
    any_mask = np.zeros(masks["red"].shape, bool)
    kernel = np.ones((3, 3), np.uint8)
    for c in SIGN_COLORS:
        # drop isolated speckles (texture noise on glass, foliage ...)
        masks[c] = cv2.morphologyEx(masks[c].astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        any_mask |= masks[c]
    if not any_mask.any():
        return {}
    tm = texel_map(part, tex)
    if len(tm) == 0:
        return {}
    fn = part.face_normals()
    out = {}
    for c in SIGN_COLORS:
        sel = masks[c][tm.rows, tm.cols]
        if sel.any():
            out[c] = (tm.positions[sel], fn[tm.faces[sel]])
    return out


def _cluster_voxels(points: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    """Connected components (26-neighbourhood) of occupied voxels.

    Returns the component label of every point and the number of occupied
    voxels of every component (a proxy for surface area).
    """
    keys = np.floor(points / voxel).astype(np.int64)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    lo = uniq.min(axis=0)
    span = uniq.max(axis=0) - lo + 3
    enc = lambda k: ((k[:, 0] - lo[0] + 1) * span[1] + (k[:, 1] - lo[1] + 1)) * span[2] + (k[:, 2] - lo[2] + 1)  # noqa: E731
    codes = enc(uniq)
    order = np.argsort(codes)
    sorted_codes = codes[order]
    rows, cols = [], []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) > (0, 0, 0)]
    for off in offsets:
        nb = enc(uniq + np.array(off))
        pos = np.searchsorted(sorted_codes, nb)
        pos = np.clip(pos, 0, len(sorted_codes) - 1)
        hit = sorted_codes[pos] == nb
        rows.append(np.nonzero(hit)[0])
        cols.append(order[pos[hit]])
    rows = np.concatenate(rows) if rows else np.zeros(0, int)
    cols = np.concatenate(cols) if cols else np.zeros(0, int)
    n = len(uniq)
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab[inv], np.bincount(lab)


def find_candidates(
    items: list[tuple[MeshPart, Texture]],
    voxel: float = 0.04,
    min_area: float = 0.03,
    min_extent: float = 0.12,
    max_extent: float = 8.0,
    max_points: int = 4000,
) -> list[Candidate]:
    per_color: dict[str, list] = {c: [] for c in SIGN_COLORS}
    for part, tex in items:
        for c, (p, n) in colored_texels(part, tex).items():
            per_color[c].append((p, n))
    rng = np.random.default_rng(0)
    out: list[Candidate] = []
    for c, chunks in per_color.items():
        if not chunks:
            continue
        P = np.concatenate([x[0] for x in chunks])
        N = np.concatenate([x[1] for x in chunks])
        labels, nvox = _cluster_voxels(P, voxel)
        for lab in np.nonzero(nvox * voxel * voxel >= min_area)[0]:
            sel = np.nonzero(labels == lab)[0]
            pts = P[sel]
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            ext = hi - lo
            if ext.max() < min_extent or ext.max() > max_extent:
                continue
            if len(sel) > max_points:
                pick = rng.choice(len(sel), max_points, replace=False)
                sel = sel[pick]
            out.append(Candidate(c, P[sel], N[sel], (lo + hi) / 2, ext))
    return out
