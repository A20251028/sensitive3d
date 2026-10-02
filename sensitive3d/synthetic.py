"""Synthetic, photogrammetry-like test dataset.

The scene is a street (road, sidewalks, grass, buildings) with traffic signs
(speed / height / weight limit, road-name plates on a pole and on a wall)
plus a few red / blue distractors that must *not* be removed.

To look like an oblique-photogrammetry reconstruction:

* the neighbourhood of every sign is rebuilt with marching cubes from a
  signed distance field, so plate, pole, ground and wall are one continuous
  "melted" surface;
* textures are baked into fragmented atlases (box-projected charts);
* four levels of detail are written as an OSGB PagedLOD hierarchy
  (``Data/Tile_*/Tile_*.osgb`` ... ``*_L3_*.osgb``) with ``metadata.xml``.

``ground_truth.json`` lists every sign so tests can score detection.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import fast_simplification
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes

from .core.mesh import MeshFile, MeshPart, Texture, face_normals
from .core.raster import rasterize_uv
from .core.texture import fill_from_nearest

GRID = 0.25  # ground / facade grid spacing (m); marching-cubes boxes snap to it
VOXEL = 0.025  # marching-cubes voxel (m)

# ---------------------------------------------------------------------------
# sign graphics
# ---------------------------------------------------------------------------

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
]


def _font(size: int) -> ImageFont.ImageFont:
    for f in _FONT_CANDIDATES:
        if Path(f).is_file():
            try:
                return ImageFont.truetype(f, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _center_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((xy[0] - w / 2 - box[0], xy[1] - h / 2 - box[1]), text, font=font, fill=fill)


def prohibitory_graphic(text: str, kind: str = "speed", size: int = 512) -> np.ndarray:
    """White disk with a red ring and black text (GB 5768 style)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    d = ImageDraw.Draw(img)
    r = size / 2
    d.ellipse([0, 0, size - 1, size - 1], fill=(250, 250, 250))
    edge = size * 0.025
    d.ellipse([edge, edge, size - edge, size - edge], fill=(200, 25, 30))
    ring = size * 0.11
    d.ellipse([edge + ring, edge + ring, size - edge - ring, size - edge - ring], fill=(250, 250, 250))
    fs = int(size * (0.42 if len(text) <= 2 else 0.3))
    _center_text(d, (r, r), text, _font(fs), (15, 15, 15))
    if kind == "height":
        t = size * 0.07
        d.polygon([(r - t, size * 0.2), (r + t, size * 0.2), (r, size * 0.2 + t * 1.2)], fill=(15, 15, 15))
        d.polygon([(r - t, size * 0.8), (r + t, size * 0.8), (r, size * 0.8 - t * 1.2)], fill=(15, 15, 15))
    elif kind == "width":
        t = size * 0.07
        d.polygon([(size * 0.2, r - t), (size * 0.2, r + t), (size * 0.2 + t * 1.2, r)], fill=(15, 15, 15))
        d.polygon([(size * 0.8, r - t), (size * 0.8, r + t), (size * 0.8 - t * 1.2, r)], fill=(15, 15, 15))
    return np.asarray(img)


def road_name_graphic(name: str, sub: str, width: int = 840, height: int = 270) -> np.ndarray:
    img = Image.new("RGB", (width, height), (10, 70, 160))
    d = ImageDraw.Draw(img)
    m = int(height * 0.06)
    d.rectangle([m, m, width - 1 - m, height - 1 - m], outline=(245, 245, 245), width=max(3, m // 2))
    _center_text(d, (width / 2, height * 0.42), name, _font(int(height * 0.42)), (245, 245, 245))
    _center_text(d, (width / 2, height * 0.78), sub, _font(int(height * 0.14)), (245, 245, 245))
    return np.asarray(img)


# ---------------------------------------------------------------------------
# noise helpers
# ---------------------------------------------------------------------------


def _hash3(ix, iy, iz) -> np.ndarray:
    v = np.sin(ix * 12.9898 + iy * 78.233 + iz * 37.719) * 43758.5453
    return v - np.floor(v)


def _noise(p: np.ndarray, scale: float) -> np.ndarray:
    q = np.floor(p / scale)
    return _hash3(q[:, 0], q[:, 1], q[:, 2])


# ---------------------------------------------------------------------------
# primitives (signed distance + colour)
# ---------------------------------------------------------------------------


@dataclass
class Prim:
    name: str
    sdf: Callable[[np.ndarray], np.ndarray]
    color: Callable[[np.ndarray, np.ndarray], np.ndarray]
    bmin: np.ndarray
    bmax: np.ndarray


def ground_height(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h = np.full(np.shape(x), 0.10)
    side = ((y >= 9.5) & (y < 12.0)) | ((y > 20.0) & (y <= 22.5))
    road = (y >= 12.0) & (y <= 20.0)
    h = np.where(side, 0.15, h)
    h = np.where(road, 0.0, h)
    return h


def ground_color(p: np.ndarray, n: np.ndarray) -> np.ndarray:
    x, y = p[:, 0], p[:, 1]
    fine = _noise(p, 0.02)
    mid = _noise(p, 0.3)
    col = np.stack([70 + 25 * fine + 15 * mid, 108 + 30 * fine + 20 * mid, 48 + 15 * fine], axis=1)  # grass
    side = ((y >= 9.5) & (y < 12.0)) | ((y > 20.0) & (y <= 22.5))
    joint = (np.abs(((x + 0.25) % 0.5) - 0.25) < 0.012) | (np.abs(((y + 0.25) % 0.5) - 0.25) < 0.012)
    tile = np.stack([168 + 12 * fine, 162 + 12 * fine, 150 + 12 * fine], axis=1)
    tile[joint] *= 0.75
    col = np.where(side[:, None], tile, col)
    road = (y >= 12.0) & (y <= 20.0)
    asphalt = np.repeat((72 + 22 * fine + 10 * mid)[:, None], 3, axis=1) + np.array([0, 0, 3])
    yellow = (np.abs(y - 16.0) > 0.08) & (np.abs(y - 16.0) < 0.22)
    white = ((np.abs(y - 14.0) < 0.075) | (np.abs(y - 18.0) < 0.075)) & ((x % 6.0) < 3.0)
    white |= (np.abs(y - 12.35) < 0.075) | (np.abs(y - 19.65) < 0.075)
    white |= (x > 44.0) & (x < 47.0) & ((np.floor((y - 12.0) / 0.6) % 2) == 0) & (y > 12.6) & (y < 19.4)
    asphalt[yellow] = (205, 170, 45)
    asphalt[white] = (215, 215, 210)
    col = np.where(road[:, None], asphalt, col)
    curb = (np.abs(y - 12.0) < 0.12) | (np.abs(y - 20.0) < 0.12)
    col[curb] = (190, 188, 182)
    return col


def box_sdf(center, half) -> Callable[[np.ndarray], np.ndarray]:
    c = np.asarray(center, float)
    h = np.asarray(half, float)

    def f(p):
        q = np.abs(p - c) - h
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)

    return f


def facade_color(base, center, half, garage=None) -> Callable:
    base = np.asarray(base, float)
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    top = c[2] + h[2]

    def f(p, n):
        fine = _noise(p, 0.02)
        col = base[None, :] * (0.92 + 0.12 * fine[:, None])
        roof = (n[:, 2] > 0.7) & (p[:, 2] > top - 0.05)
        along = np.where(np.abs(n[:, 0]) > np.abs(n[:, 1]), p[:, 1], p[:, 0])
        z = p[:, 2]
        win = ((along % 2.5) > 0.65) & ((along % 2.5) < 1.85) & ((z % 3.0) > 1.0) & ((z % 3.0) < 2.4) & (z < top - 0.6)
        glass = np.stack([55 + 25 * fine, 70 + 25 * fine, 88 + 25 * fine], axis=1)
        col = np.where((win & ~roof)[:, None], glass, col)
        if garage is not None:
            gx0, gx1, gy, gz1 = garage
            g = (p[:, 0] > gx0) & (p[:, 0] < gx1) & (np.abs(p[:, 1] - gy) < 0.05) & (z < gz1)
            col[g] = np.stack([30 + 15 * fine[g], 75 + 15 * fine[g], 165 + 20 * fine[g]], axis=1)
        col[roof] = np.stack([88 + 25 * fine[roof]] * 3, axis=1)
        return col

    return f


def pole_sdf(x, y, r, z0, z1) -> Callable:
    def f(p):
        dxy = np.hypot(p[:, 0] - x, p[:, 1] - y) - r
        dz = np.maximum(z0 - p[:, 2], p[:, 2] - z1)
        out = np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0))
        return out + np.minimum(np.maximum(dxy, dz), 0)

    return f


def metal_color(level=150) -> Callable:
    def f(p, n):
        fine = _noise(p, 0.02)
        return np.repeat((level + 18 * fine)[:, None], 3, axis=1)

    return f


@dataclass
class Plate:
    center: np.ndarray
    normal: np.ndarray
    width: float
    height: float
    shape: str  # "disk" | "rect"
    graphic: np.ndarray
    thickness: float = 0.06

    @property
    def u(self) -> np.ndarray:
        u = np.cross([0.0, 0.0, 1.0], self.normal)
        return u / np.linalg.norm(u)

    @property
    def v(self) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0])

    def local(self, p):
        d = p - self.center
        return d @ self.u, d @ self.v, d @ self.normal

    def sdf(self, p):
        a, b, c = self.local(p)
        dc = np.abs(c) - self.thickness / 2
        if self.shape == "disk":
            dr = np.hypot(a, b) - self.width / 2
        else:
            qa = np.abs(a) - self.width / 2
            qb = np.abs(b) - self.height / 2
            dr = np.hypot(np.maximum(qa, 0), np.maximum(qb, 0)) + np.minimum(np.maximum(qa, qb), 0)
        return np.hypot(np.maximum(dr, 0), np.maximum(dc, 0)) + np.minimum(np.maximum(dr, dc), 0)

    def color(self, p, n):
        a, b, c = self.local(p)
        back = metal_color(165)(p, n)
        front = (n @ self.normal > 0.35) & (c > 0)
        if not front.any():
            return back
        H, W = self.graphic.shape[:2]
        gx = np.clip((a[front] / self.width + 0.5) * W, 0, W - 1).astype(int)
        gy = np.clip((0.5 - b[front] / self.height) * H, 0, H - 1).astype(int)
        col = back.copy()
        col[front] = self.graphic[gy, gx].astype(float)
        return col


@dataclass
class SignSpec:
    id: str
    category: str
    label: str
    plate: Plate
    mount: str  # "pole" | "wall"
    pole: Optional[tuple] = None  # (x, y, r, z0, z1)


@dataclass
class Scene:
    extent: tuple = (0.0, 48.0, 0.0, 32.0)
    prims: list = field(default_factory=list)
    signs: list = field(default_factory=list)
    buildings: list = field(default_factory=list)  # (center, half, prim index)
    mc_boxes: list = field(default_factory=list)  # (bmin, bmax)
    tiles: list = field(default_factory=list)  # (name, xmin, xmax, ymin, ymax)


def _snap_box(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    lo = np.floor(np.asarray(lo) / GRID) * GRID
    hi = np.ceil(np.asarray(hi) / GRID) * GRID
    return lo, hi


def build_scene() -> Scene:
    s = Scene()
    s.prims.append(
        Prim(
            "ground",
            lambda p: p[:, 2] - ground_height(p[:, 0], p[:, 1]),
            ground_color,
            np.array([0, 0, -1.0]),
            np.array([48, 32, 0.5]),
        )
    )
    blds = [
        ((9.0, 5.0, 5.0), (6.0, 3.0, 5.0), (205, 188, 160), None),
        ((35.0, 27.25, 7.0), (7.0, 2.75, 7.0), (150, 160, 176), None),
        ((20.0, 27.0, 3.0), (2.0, 2.0, 3.0), (165, 95, 75), (19.0, 21.0, 25.0, 2.4)),
    ]
    for c, h, base, garage in blds:
        c = np.array(c)
        h = np.array(h)
        c[2] = h[2]  # sits on z = 0 (ground under buildings is lower than their base)
        s.buildings.append((c, h, len(s.prims)))
        s.prims.append(Prim("building", box_sdf(c, h), facade_color(base, c, h, garage), c - h, c + h))
    # distractor: a red "car" on the road
    car_c, car_h = np.array([22.0, 14.0, 0.75]), np.array([2.0, 0.9, 0.75])

    def car_color(p, n):
        fine = _noise(p, 0.02)
        col = np.stack([170 + 30 * fine, 28 + 10 * fine, 30 + 10 * fine], axis=1)
        glass = (p[:, 2] > 1.05) & (np.abs(n[:, 2]) < 0.5)
        col[glass] = (40, 45, 55)
        return col

    s.prims.append(Prim("car", box_sdf(car_c, car_h), car_color, car_c - car_h, car_c + car_h))
    s.buildings.append((car_c, car_h, len(s.prims) - 1))

    def pole_sign(sid, cat, label, x, y, normal, plate_z, w, h, shape, graphic):
        normal = np.asarray(normal, float)
        r = 0.05
        gz = float(ground_height(np.array([x]), np.array([y]))[0])
        top = plate_z + h / 2
        center = np.array([x, y, plate_z]) + normal * (r + 0.035)
        plate = Plate(center, normal, w, h, shape, graphic)
        pole = (x, y, r, gz - 0.3, top + 0.05)
        s.signs.append(SignSpec(sid, cat, label, plate, "pole", pole))
        s.prims.append(Prim("pole", pole_sdf(x, y, r, gz - 0.3, top + 0.05), metal_color(140),
                            np.array([x - r, y - r, gz - 0.3]), np.array([x + r, y + r, top + 0.05])))
        ext = np.array([w / 2, w / 2, h / 2])
        s.prims.append(Prim("plate", plate.sdf, plate.color, center - ext - 0.1, center + ext + 0.1))
        lo, hi = _snap_box([x - w / 2 - 0.3, y - w / 2 - 0.3, gz - 0.3], [x + w / 2 + 0.3, y + w / 2 + 0.3, top + 0.3])
        s.mc_boxes.append((lo, hi))

    pole_sign("speed_limit_60", "speed_limit", "限速 60", 8.0, 11.2, (-1, 0, 0), 2.6, 0.8, 0.8, "disk",
              prohibitory_graphic("60", "speed"))
    pole_sign("height_limit_4.5", "height_limit", "限高 4.5m", 30.0, 20.8, (1, 0, 0), 2.6, 0.8, 0.8, "disk",
              prohibitory_graphic("4.5m", "height"))
    pole_sign("weight_limit_20t", "weight_limit", "限重 20t", 40.0, 11.2, (-1, 0, 0), 2.6, 0.8, 0.8, "disk",
              prohibitory_graphic("20t", "weight"))
    pole_sign("road_name_jianshe", "road_name", "路牌 建设大道", 15.0, 21.6, (0, -1, 0), 2.9, 1.4, 0.45, "rect",
              road_name_graphic("建设大道", "JIANSHE AVE"))

    # wall mounted road-name plate on building 1 (wall y = 8 faces +y)
    plate = Plate(np.array([9.0, 8.0 + 0.03, 3.2]), np.array([0.0, 1.0, 0.0]), 1.0, 0.33, "rect",
                  road_name_graphic("和平街", "HEPING ST", 840, 280))
    s.signs.append(SignSpec("road_name_heping", "road_name", "路牌 和平街", plate, "wall"))
    s.prims.append(Prim("plate", plate.sdf, plate.color, plate.center - 0.7, plate.center + 0.7))
    lo, hi = _snap_box([8.25, 7.75, 2.75], [9.75, 8.5, 3.75])
    s.mc_boxes.append((lo, hi))

    s.tiles = [("Tile_+000_+000", 0.0, 24.0, 0.0, 32.0), ("Tile_+001_+000", 24.0, 48.0, 0.0, 32.0)]
    return s


def scene_sdf_color(scene: Scene, p: np.ndarray, n: np.ndarray) -> np.ndarray:
    d = np.stack([np.abs(pr.sdf(p)) for pr in scene.prims], axis=1)
    which = np.argmin(d, axis=1)
    col = np.zeros((len(p), 3))
    for k, pr in enumerate(scene.prims):
        sel = which == k
        if sel.any():
            col[sel] = pr.color(p[sel], n[sel])
    return col


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def _grid_faces(nx: int, ny: int, keep: np.ndarray) -> np.ndarray:
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    a = i * (ny + 1) + j
    b = (i + 1) * (ny + 1) + j
    c = (i + 1) * (ny + 1) + j + 1
    d = i * (ny + 1) + j + 1
    k = keep.ravel()
    f1 = np.stack([a.ravel(), b.ravel(), c.ravel()], axis=1)[k]
    f2 = np.stack([a.ravel(), c.ravel(), d.ravel()], axis=1)[k]
    return np.concatenate([f1, f2])


def _inside_any(points: np.ndarray, boxes) -> np.ndarray:
    m = np.zeros(len(points), dtype=bool)
    for lo, hi in boxes:
        m |= np.all((points > lo) & (points < hi), axis=1)
    return m


def ground_mesh(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = scene.extent
    nx, ny = int(round((x1 - x0) / GRID)), int(round((y1 - y0) / GRID))
    xs = x0 + np.arange(nx + 1) * GRID
    ys = y0 + np.arange(ny + 1) * GRID
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    V = np.stack([X.ravel(), Y.ravel(), ground_height(X.ravel(), Y.ravel())], axis=1)
    cx, cy = np.meshgrid(xs[:-1] + GRID / 2, ys[:-1] + GRID / 2, indexing="ij")
    cent = np.stack([cx.ravel(), cy.ravel(), ground_height(cx.ravel(), cy.ravel())], axis=1)
    keep = ~_inside_any(cent, scene.mc_boxes)
    for c, h, _ in scene.buildings:  # no ground under buildings / car
        keep &= ~np.all(np.abs(cent[:, :2] - c[:2]) < h[:2] - 1e-6, axis=1)
    return V, _grid_faces(nx, ny, keep.reshape(nx, ny))


def box_mesh(center, half, holes) -> tuple[np.ndarray, np.ndarray]:
    c, h = np.asarray(center, float), np.asarray(half, float)
    lo, hi = c - h, c + h
    Vs, Fs, base = [], [], 0
    # (origin, axis_u, axis_v) for 4 walls + roof, oriented outward (u x v = normal)
    faces = [
        ((lo[0], lo[1], lo[2]), (hi[0] - lo[0], 0, 0), (0, 0, hi[2] - lo[2])),  # -y
        ((hi[0], hi[1], lo[2]), (lo[0] - hi[0], 0, 0), (0, 0, hi[2] - lo[2])),  # +y
        ((lo[0], hi[1], lo[2]), (0, lo[1] - hi[1], 0), (0, 0, hi[2] - lo[2])),  # -x
        ((hi[0], lo[1], lo[2]), (0, hi[1] - lo[1], 0), (0, 0, hi[2] - lo[2])),  # +x
        ((lo[0], lo[1], hi[2]), (hi[0] - lo[0], 0, 0), (0, hi[1] - lo[1], 0)),  # roof
    ]
    for o, u, v in faces:
        o, u, v = np.array(o), np.array(u), np.array(v)
        nu = max(int(round(np.linalg.norm(u) / GRID)), 1)
        nv = max(int(round(np.linalg.norm(v) / GRID)), 1)
        a = np.linspace(0, 1, nu + 1)
        b = np.linspace(0, 1, nv + 1)
        A, B = np.meshgrid(a, b, indexing="ij")
        V = o + A.ravel()[:, None] * u + B.ravel()[:, None] * v
        ca, cb = np.meshgrid((a[:-1] + a[1:]) / 2, (b[:-1] + b[1:]) / 2, indexing="ij")
        cent = o + ca.ravel()[:, None] * u + cb.ravel()[:, None] * v
        keep = ~_inside_any(cent, holes)
        F = _grid_faces(nu, nv, keep.reshape(nu, nv))
        Vs.append(V)
        Fs.append(F + base)
        base += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def mc_patch(scene: Scene, lo: np.ndarray, hi: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = np.round((hi - lo) / VOXEL).astype(int) + 1
    axes = [lo[k] + np.arange(n[k]) * VOXEL for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    prims = [pr for pr in scene.prims if np.all(pr.bmin <= hi + 0.5) and np.all(pr.bmax >= lo - 0.5)]
    vol = np.min(np.stack([pr.sdf(G) for pr in prims], axis=1), axis=1).reshape(n)
    V, F, _, _ = marching_cubes(vol, 0.0, spacing=(VOXEL,) * 3)
    V = V.astype(np.float64) + lo
    F = F.astype(np.int64)  # default "descent" winding is outward for a signed distance field
    Vd, Fd = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32), target_reduction=0.6)
    V, F = np.asarray(Vd, np.float64), np.asarray(Fd, np.int64)
    border = np.any((V <= lo + 1e-4) | (V >= hi - 1e-4), axis=1)
    V[~border] += rng.normal(0, 0.003, size=(int((~border).sum()), 3))
    return V, F


def scene_mesh(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    Vs, Fs, base = [], [], 0

    def add(V, F):
        nonlocal base
        Vs.append(V)
        Fs.append(F + base)
        base += len(V)

    add(*ground_mesh(scene))
    for c, h, _ in scene.buildings:
        add(*box_mesh(c, h, scene.mc_boxes))
    for lo, hi in scene.mc_boxes:
        add(*mc_patch(scene, lo, hi, rng))
    return np.concatenate(Vs), np.concatenate(Fs)


# ---------------------------------------------------------------------------
# texture atlas
# ---------------------------------------------------------------------------


def _pow2(n: int) -> int:
    return 1 << max(int(math.ceil(math.log2(max(n, 1)))), 4)


def make_atlas(V, F, texel: float, page: int = 2048, max_chart: float = 4.0, pad: int = 3):
    """Box-projected charts packed into atlas pages.

    Returns a list of ``(vertices, uvs, faces, height, width)`` per page.
    """
    if len(F) == 0:
        return []
    fn = face_normals(V, F)
    axis = np.argmax(np.abs(fn), axis=1)
    label = axis * 2 + (fn[np.arange(len(F)), axis] > 0)
    nF = len(F)
    rows = np.repeat(np.arange(nF), 3)
    inc = coo_matrix((np.ones(nF * 3), (rows, F.ravel())), shape=(nF, len(V))).tocsr()
    charts = []
    for lab in np.unique(label):
        fsel = np.nonzero(label == lab)[0]
        sub = inc[fsel]
        adj = sub @ sub.T
        ncomp, comp = connected_components(adj, directed=False)
        ax = lab // 2
        keep_axes = [k for k in range(3) if k != ax]
        cent2 = V[F[fsel]].mean(axis=1)[:, keep_axes]
        cell = np.floor(cent2 / max_chart).astype(np.int64)
        key = comp.astype(np.int64) * 1_000_003 + cell[:, 0] * 1009 + cell[:, 1]
        for k in np.unique(key):
            charts.append((fsel[key == k], keep_axes))
    rects = []
    for fids, keep_axes in charts:
        vids, local = np.unique(F[fids], return_inverse=True)
        p2 = V[vids][:, keep_axes] / texel
        p2 = p2 - p2.min(axis=0)
        w = int(math.ceil(p2[:, 0].max())) + 1 + 2 * pad
        h = int(math.ceil(p2[:, 1].max())) + 1 + 2 * pad
        if max(w, h) > page:
            s = (page - 2 * pad - 2) / max(w, h)
            p2 *= s
            w = int(math.ceil(p2[:, 0].max())) + 1 + 2 * pad
            h = int(math.ceil(p2[:, 1].max())) + 1 + 2 * pad
        rects.append((fids, vids, local.reshape(-1, 3), p2, w, h))
    order = sorted(range(len(rects)), key=lambda i: -rects[i][5])
    pages: list[list] = []
    cur = None
    for i in order:
        fids, vids, local, p2, w, h = rects[i]
        if cur is None or cur["x"] + w > page:
            if cur is not None:
                cur["y"] += cur["row"]
                cur["x"] = 0
                cur["row"] = 0
            if cur is None or cur["y"] + h > page:
                cur = {"x": 0, "y": 0, "row": 0, "items": []}
                pages.append(cur)
        cur["items"].append((fids, vids, local, p2 + [cur["x"] + pad, cur["y"] + pad]))
        cur["x"] += w
        cur["row"] = max(cur["row"], h)
    out = []
    for pg in pages:
        Vs, UVs, Fs, base = [], [], [], 0
        used_h = max(int(np.ceil(it[3][:, 1].max())) for it in pg["items"]) + pad + 1
        used_w = max(int(np.ceil(it[3][:, 0].max())) for it in pg["items"]) + pad + 1
        W, H = _pow2(used_w), _pow2(used_h)
        for fids, vids, local, p2 in pg["items"]:
            Vs.append(V[vids])
            UVs.append(np.stack([p2[:, 0] / W, 1.0 - p2[:, 1] / H], axis=1))
            Fs.append(local + base)
            base += len(vids)
        out.append((np.concatenate(Vs), np.concatenate(UVs), np.concatenate(Fs), H, W))
    return out


def bake(scene: Scene, V, UV, F, H, W, rng=None) -> np.ndarray:
    tex = Texture(np.zeros((H, W, 3), np.uint8))
    buf = rasterize_uv(tex.uv_to_px(UV), F, W, H)
    rows, cols = np.nonzero(buf.valid)
    fids = buf.face[rows, cols]
    pos = np.einsum("nk,nkd->nd", buf.bary[rows, cols], V[F[fids]])
    nrm = face_normals(V, F)[fids]
    col = scene_sdf_color(scene, pos, nrm)
    light = np.array([0.45, -0.35, 0.82])
    light /= np.linalg.norm(light)
    shade = 0.62 + 0.38 * np.clip(nrm @ light, 0, 1)
    col = col * shade[:, None]
    img = np.zeros((H, W, 3), np.float64)
    img[rows, cols] = col
    img = np.clip(img, 0, 255).astype(np.uint8)
    fill_from_nearest(img, buf.valid, ~buf.valid)
    return img


# ---------------------------------------------------------------------------
# dataset writers
# ---------------------------------------------------------------------------

LEVELS = [
    {"name": "L0", "texel": 0.16, "reduction": 0.97, "cells": (1, 1), "page": 512},
    {"name": "L1", "texel": 0.08, "reduction": 0.90, "cells": (1, 1), "page": 1024},
    {"name": "L2", "texel": 0.04, "reduction": 0.70, "cells": (1, 2), "page": 2048},
    {"name": "L3", "texel": 0.02, "reduction": 0.00, "cells": (2, 2), "page": 2048},
]


def _cell_bounds(tile, cells):
    _, x0, x1, y0, y1 = tile
    nx, ny = cells
    out = []
    for j in range(ny):
        for i in range(nx):
            out.append((x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx, y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny))
    return out


def _submesh(V, F, fmask):
    f = F[fmask]
    vids, inv = np.unique(f, return_inverse=True)
    return V[vids], inv.reshape(-1, 3)


def level_parts(scene, V, F, cell, level) -> list[MeshPart]:
    cx0, cx1, cy0, cy1 = cell
    c = V[F].mean(axis=1)
    m = (c[:, 0] >= cx0) & (c[:, 0] < cx1) & (c[:, 1] >= cy0) & (c[:, 1] < cy1)
    if not m.any():
        return []
    v, f = _submesh(V, F, m)
    if level["reduction"] > 0:
        vd, fd = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=level["reduction"])
        v, f = np.asarray(vd, np.float64), np.asarray(fd, np.int64)
    parts = []
    for pv, puv, pf, H, W in make_atlas(v, f, level["texel"], page=level["page"]):
        img = bake(scene, pv, puv, pf, H, W)
        parts.append(MeshPart(pv, pf, uvs=puv, texture=None, has_finer=level is not LEVELS[-1]))
        parts[-1].meta_image = img  # type: ignore[attr-defined]
    return parts


def _sphere(parts):
    pts = np.concatenate([p.vertices for p in parts])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = (lo + hi) / 2
    return c, float(np.linalg.norm(hi - lo) / 2)


def write_osgb_dataset(scene: Scene, V, F, out: Path, work: Path, log=print) -> dict:
    from .io.osgb import build_osgb

    (out / "Data").mkdir(parents=True, exist_ok=True)
    (out / "metadata.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<ModelMetadata version="1">\n'
        "\t<!--Spatial Reference System-->\n\t<SRS>ENU:31.23040,121.47370</SRS>\n"
        "\t<!--Origin in Spatial Reference System-->\n\t<SRSOrigin>0,0,0</SRSOrigin>\n"
        "\t<Texture>\n\t\t<ColorSource>Visible</ColorSource>\n\t</Texture>\n</ModelMetadata>\n",
        encoding="utf-8",
    )
    leaf_parts_all = []
    stats = {"files": 0, "triangles": 0}
    for tile in scene.tiles:
        name = tile[0]
        tdir = out / "Data" / name
        tdir.mkdir(parents=True, exist_ok=True)
        # parts[level][cell] -> list of MeshPart
        per_level = [[level_parts(scene, V, F, cell, lv) for cell in _cell_bounds(tile, lv["cells"])] for lv in LEVELS]
        bin_dir = work / name
        bin_dir.mkdir(parents=True, exist_ok=True)
        counter = [0]

        def geode(parts):
            lines = ["BEGIN_GEODE"]
            for p in parts:
                k = counter[0]
                counter[0] += 1
                p.vertices.astype(np.float32).tofile(bin_dir / f"v{k}.f32")
                p.uvs.astype(np.float32).tofile(bin_dir / f"uv{k}.f32")
                p.faces.astype(np.uint32).tofile(bin_dir / f"t{k}.u32")
                Image.fromarray(p.meta_image).save(bin_dir / f"{name}_{k}.jpg", quality=90)
                lines.append(f"GEOMETRY v{k}.f32 uv{k}.f32 - t{k}.u32 {name}_{k}.jpg")
                stats["triangles"] += len(p.faces)
            lines.append("END_GEODE")
            return "\n".join(lines)

        def plod(parts, children, rng=400.0):
            c, r = _sphere(parts) if parts else (np.zeros(3), 1.0)
            s = f"BEGIN_PAGEDLOD {c[0]:.4f} {c[1]:.4f} {c[2]:.4f} {r:.4f} 1\nCHILD 0 {rng} {geode(parts)}\n"
            for ch in children:
                s += f"FILE_CHILD {ch} {rng} 1e30\n"
            return s + "END_PAGEDLOD\n"

        def emit(fname, text):
            build_osgb(text, bin_dir, tdir / fname)
            stats["files"] += 1

        emit(f"{name}.osgb", plod(per_level[0][0], [f"{name}_L1.osgb"]))
        emit(f"{name}_L1.osgb", plod(per_level[1][0], [f"{name}_L2_{i}.osgb" for i in range(2)]))
        for i in range(2):
            emit(f"{name}_L2_{i}.osgb", plod(per_level[2][i], [f"{name}_L3_{i}{j}.osgb" for j in range(2)]))
            for j in range(2):
                leaf = per_level[3][i * 2 + j]
                leaf_parts_all.extend(leaf)
                emit(f"{name}_L3_{i}{j}.osgb", geode(leaf) + "\n")
        log(f"  {name}: done")
    stats["leaf_parts"] = leaf_parts_all
    return stats


def ground_truth(scene: Scene) -> list[dict]:
    out = []
    for s in scene.signs:
        p = s.plate
        out.append(
            {
                "id": s.id,
                "category": s.category,
                "label": s.label,
                "mount": s.mount,
                "center": p.center.round(4).tolist(),
                "normal": p.normal.tolist(),
                "width": p.width,
                "height": p.height,
                "pole": list(s.pole) if s.pole else None,
            }
        )
    return out


def generate(out_dir: str | Path, formats=("osgb", "obj", "glb"), log=print) -> dict:
    import tempfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = build_scene()
    log("building scene mesh ...")
    V, F = scene_mesh(scene)
    log(f"  {len(F)} triangles at full resolution")
    gt = ground_truth(scene)
    leaf_parts = None
    result = {"ground_truth": gt}
    if "osgb" in formats:
        log("writing OSGB dataset ...")
        with tempfile.TemporaryDirectory(prefix="s3d_syn_") as tmp:
            stats = write_osgb_dataset(scene, V, F, out_dir / "osgb", Path(tmp), log=log)
        leaf_parts = stats.pop("leaf_parts")
        result["osgb"] = stats
        (out_dir / "osgb" / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    if "obj" in formats or "glb" in formats:
        if leaf_parts is None:
            leaf_parts = []
            for tile in scene.tiles:
                for cell in _cell_bounds(tile, LEVELS[-1]["cells"]):
                    leaf_parts.extend(level_parts(scene, V, F, cell, LEVELS[-1]))
        textures, parts = [], []
        for k, p in enumerate(leaf_parts):
            textures.append(Texture(p.meta_image, name=f"scene_tex{k}.jpg"))
            parts.append(MeshPart(p.vertices, p.faces, uvs=p.uvs, texture=k, index=k, name=f"mat{k}"))
        mesh = MeshFile("scene", "obj", parts, textures)
        if "obj" in formats:
            from .io.obj import write_obj

            log("writing OBJ ...")
            write_obj(mesh, out_dir / "obj" / "scene.obj")
            (out_dir / "obj" / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
        if "glb" in formats:
            from .io.gltf import write_gltf

            log("writing GLB ...")
            write_gltf(mesh, out_dir / "glb" / "scene.glb")
            (out_dir / "glb" / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
