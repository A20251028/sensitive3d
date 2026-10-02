"""Image inpainting backends.

``auto`` uses a multi-scale exemplar (patch) synthesis that copies real
surrounding texture into the hole, followed by Poisson-style seam blending;
small holes fall back to OpenCV's Telea method.  ``lama`` runs a LaMa ONNX
model when one is configured (``S3D_LAMA_ONNX`` or the ``lama_model``
option).
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from numba import njit


def inpaint_telea(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    return cv2.inpaint(img, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


@njit(cache=True)
def _patch_dist(img, ty, tx, sy, sx, r, best):
    H, W, C = img.shape
    d = 0.0
    n = 0
    for dy in range(-r, r + 1):
        y1 = ty + dy
        y2 = sy + dy
        if y1 < 0 or y1 >= H or y2 < 0 or y2 >= H:
            continue
        for dx in range(-r, r + 1):
            x1 = tx + dx
            x2 = sx + dx
            if x1 < 0 or x1 >= W or x2 < 0 or x2 >= W:
                continue
            for c in range(C):
                t = img[y1, x1, c] - img[y2, x2, c]
                d += t * t
            n += 1
        if n > 0 and d / n > best:
            return 1e30
    if n == 0:
        return 1e30
    return d / n


@njit(cache=True)
def _pm_search(img, ty, tx, source_ok, nnf, cost, r, iters, seed):
    """PatchMatch nearest-neighbour search for the target pixels (ty, tx)."""
    np.random.seed(seed)
    H, W, C = img.shape
    n = len(ty)
    for k in range(n):
        cost[k] = _patch_dist(img, ty[k], tx[k], nnf[k, 0], nnf[k, 1], r, 1e30)
    # index of each target pixel for propagation lookups
    idx = -np.ones((H, W), np.int64)
    for k in range(n):
        idx[ty[k], tx[k]] = k
    for it in range(iters):
        step = 1 if it % 2 == 0 else -1
        for kk in range(n):
            k = kk if step == 1 else n - 1 - kk
            y = ty[k]
            x = tx[k]
            by = nnf[k, 0]
            bx = nnf[k, 1]
            best = cost[k]
            for o in range(2):
                oy = -step if o == 0 else 0
                ox = 0 if o == 0 else -step
                py = y + oy
                px = x + ox
                if 0 <= py < H and 0 <= px < W and idx[py, px] >= 0:
                    j = idx[py, px]
                    cy = nnf[j, 0] - oy
                    cx = nnf[j, 1] - ox
                    if 0 <= cy < H and 0 <= cx < W and source_ok[cy, cx]:
                        d = _patch_dist(img, y, x, cy, cx, r, best)
                        if d < best:
                            best = d
                            by = cy
                            bx = cx
            rad = max(H, W)
            while rad >= 1:
                cy = by + np.random.randint(-rad, rad + 1)
                cx = bx + np.random.randint(-rad, rad + 1)
                if 0 <= cy < H and 0 <= cx < W and source_ok[cy, cx]:
                    d = _patch_dist(img, y, x, cy, cx, r, best)
                    if d < best:
                        best = d
                        by = cy
                        bx = cx
                rad //= 2
            nnf[k, 0] = by
            nnf[k, 1] = bx
            cost[k] = best


@njit(cache=True)
def _vote(img, hole, ty, tx, nnf, cost, r, sharp):
    """Every hole pixel becomes the weighted mean of the source pixels of all patches covering it."""
    H, W, C = img.shape
    acc = np.zeros((H, W, C))
    wsum = np.zeros((H, W))
    med = np.median(cost) + 1e-6
    for k in range(len(ty)):
        w = np.exp(-cost[k] / med * sharp)
        for dy in range(-r, r + 1):
            py = ty[k] + dy
            qy = nnf[k, 0] + dy
            if py < 0 or py >= H or qy < 0 or qy >= H:
                continue
            for dx in range(-r, r + 1):
                px = tx[k] + dx
                qx = nnf[k, 1] + dx
                if px < 0 or px >= W or qx < 0 or qx >= W or not hole[py, px]:
                    continue
                for c in range(C):
                    acc[py, px, c] += w * img[qy, qx, c]
                wsum[py, px] += w
    for y in range(H):
        for x in range(W):
            if hole[y, x] and wsum[y, x] > 0:
                for c in range(C):
                    img[y, x, c] = acc[y, x, c] / wsum[y, x]


def _complete_level(work, hole, source, r, em_iters, rng, sharp_last=True):
    if source.sum() < 20 or not hole.any():
        return work
    tgt = cv2.dilate(hole.astype(np.uint8), np.ones((2 * r + 1,) * 2, np.uint8)).astype(bool)
    ty, tx = np.nonzero(tgt)
    sy, sx = np.nonzero(source)
    pick = rng.integers(0, len(sy), size=len(ty))
    nnf = np.stack([sy[pick], sx[pick]], axis=1).astype(np.int64)
    cost = np.zeros(len(ty))
    ty = ty.astype(np.int64)
    tx = tx.astype(np.int64)
    for it in range(em_iters):
        _pm_search(work, ty, tx, source, nnf, cost, r, 3 if it == 0 else 2, int(rng.integers(1 << 30)))
        _vote(work, hole, ty, tx, nnf, cost, r, 3.0 if (sharp_last and it == em_iters - 1) else 0.5)
    if sharp_last:
        # voting averages away fine grain; mix the matched pixels back in
        _pm_search(work, ty, tx, source, nnf, cost, r, 1, int(rng.integers(1 << 30)))
        inside = hole[ty, tx]
        copy = work[nnf[inside, 0], nnf[inside, 1]]
        work[ty[inside], tx[inside]] = 0.5 * work[ty[inside], tx[inside]] + 0.5 * copy
    return work


def inpaint_exemplar(img: np.ndarray, mask: np.ndarray, valid: Optional[np.ndarray] = None, patch: int = 3, seed: int = 0) -> np.ndarray:
    """Multi-scale EM / PatchMatch image completion (Wexler et al.).

    The hole is filled with real texture patches from the surrounding
    ``valid`` pixels; overlapping patches vote, coarse to fine, so that
    structures such as joints or lane lines continue through the hole.
    """
    if valid is None:
        valid = np.ones(mask.shape, bool)
    hole = mask & valid
    if not hole.any():
        return img.copy()
    k = np.ones((2 * patch + 1,) * 2, np.uint8)
    source = valid & ~cv2.dilate(hole.astype(np.uint8), k).astype(bool)
    source &= cv2.erode(valid.astype(np.uint8), k).astype(bool)
    if source.sum() < 50:
        return inpaint_telea(img, mask)
    levels = [(img.astype(np.float64), hole, source)]

    def thickness(h):
        return cv2.distanceTransform(h.astype(np.uint8), cv2.DIST_L2, 3).max()

    # go down until the hole is only a few pixels thick so that the coarsest
    # level sees the global structure (lines, joints) around it
    while thickness(levels[-1][1]) > patch and min(levels[-1][0].shape[:2]) >= 4 * (2 * patch + 1) and len(levels) < 7:
        im, ho, so = levels[-1]
        h2, w2 = im.shape[0] // 2, im.shape[1] // 2
        im2 = cv2.resize(im, (w2, h2), interpolation=cv2.INTER_AREA)
        ho2 = cv2.resize(ho.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_AREA) > 0
        so2 = cv2.erode(cv2.resize(so.astype(np.uint8), (w2, h2), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)).astype(bool) & ~ho2
        levels.append((im2, ho2, so2))
    rng = np.random.default_rng(seed)
    prev = None
    nlev = len(levels)
    for li, (im, ho, so) in enumerate(reversed(levels)):
        work = im.copy()
        if prev is None:
            smooth = inpaint_telea(np.clip(work, 0, 255).astype(np.uint8), ho, 3).astype(np.float64)
            work[ho] = smooth[ho]
        else:
            up = cv2.resize(prev, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
            work[ho] = up[ho]
        em = 8 if li < 2 else (4 if li < nlev - 1 else 3)
        prev = _complete_level(work, ho, so, patch, em, rng, sharp_last=(li == nlev - 1))
    out = img.astype(np.float64).copy()
    out[hole] = prev[hole]
    return np.clip(out, 0, 255).astype(np.uint8)


def harmonize(result: np.ndarray, original: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Match the mean colour of the filled region to its surrounding ring (lighting continuity)."""
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~mask & valid
    inner_ring = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    if ring.sum() < 10 or inner_ring.sum() < 10:
        return result
    delta = original[ring].astype(float).mean(axis=0) - result[inner_ring].astype(float).mean(axis=0)
    if np.abs(delta).max() < 2:
        return result
    out = result.astype(float)
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    w = np.exp(-dist / 6.0)[..., None]
    out[mask] += (w[mask] * delta)
    return np.clip(out, 0, 255).astype(np.uint8)


_LAMA = {}


def inpaint_lama(img: np.ndarray, mask: np.ndarray, model_path: str) -> np.ndarray:
    import onnxruntime as ort

    if model_path not in _LAMA:
        _LAMA[model_path] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    sess = _LAMA[model_path]
    H, W = mask.shape
    size = 512
    inputs = sess.get_inputs()
    x = cv2.resize(img[..., :3], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    m = (cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
    feeds = {inputs[0].name: x, inputs[1].name: m}
    y = sess.run(None, feeds)[0][0].transpose(1, 2, 0)
    if y.max() <= 1.5:
        y = y * 255.0
    y = cv2.resize(np.clip(y, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    out[mask] = y[mask]
    return out


def inpaint(img: np.ndarray, mask: np.ndarray, valid: Optional[np.ndarray] = None, method: str = "auto", lama_model: Optional[str] = None) -> np.ndarray:
    """Fill ``mask`` pixels of ``img`` (uint8 RGB) from the surrounding ``valid`` pixels."""
    if valid is None:
        valid = np.ones(mask.shape, bool)
    mask = mask & valid
    if not mask.any():
        return img.copy()
    lama_model = lama_model or os.environ.get("S3D_LAMA_ONNX")
    if method == "lama" or (method == "auto" and lama_model and os.path.isfile(lama_model)):
        if not lama_model:
            raise ValueError("method 'lama' needs a LaMa ONNX model (lama_model / S3D_LAMA_ONNX)")
        out = inpaint_lama(img, mask, lama_model)
    elif method == "telea":
        out = inpaint_telea(img, mask)
    elif method == "ns":
        out = cv2.inpaint(img, mask.astype(np.uint8) * 255, 5, cv2.INPAINT_NS)
    else:  # auto / exemplar
        if mask.sum() < 30:
            out = inpaint_telea(img, mask, 3)
        else:
            out = inpaint_exemplar(img, mask, valid)
    return harmonize(out, img, mask, valid)
