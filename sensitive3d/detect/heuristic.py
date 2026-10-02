"""Model-free traffic-sign detector for rendered close-up views.

Works on colour and shape, following the Chinese GB 5768 sign families:

* prohibitory signs (speed / height / weight / width limit ...): white disk
  with a red ring and dark symbols;
* guide / road-name signs: blue or green rectangles carrying white text;
* warning signs: yellow triangles with a black border and symbol.

Every detection carries a filled mask of the whole sign face, its category
and a confidence score.  Physical size limits use the view's pixel size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class Detection2D:
    mask: np.ndarray  # (H, W) bool, whole sign face
    category: str
    score: float
    label: str = ""
    bbox: tuple = field(default=(0, 0, 0, 0))  # x0, y0, x1, y1
    source: str = "heuristic"


def _hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)


def color_masks(img: np.ndarray, loose: bool = False) -> dict[str, np.ndarray]:
    hsv = _hsv(img)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    smin = 100 if loose else 110
    vmin = 40 if loose else 50
    sat = (s >= smin) & (v >= vmin)
    return {
        "red": sat & ((h <= 9) | (h >= 166)),
        "blue": sat & (h >= 96) & (h <= 130),
        "green": sat & (h >= 45) & (h <= 90),
        "yellow": sat & (h >= 16) & (h <= 34) & (v >= 90),
        "white": (s <= 70) & (v >= 165),
        "dark": v <= 95,
    }


def _components(mask: np.ndarray, min_area: int):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    slices = ndimage.find_objects(lab)
    for i, (sz, sl) in enumerate(zip(sizes, slices)):
        if sz >= min_area:
            yield (lab[sl] == i + 1), sl


def _full(local: np.ndarray, sl, shape) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[sl] = local
    return out


def _pad_slice(sl, shape, pad):
    return tuple(slice(max(s.start - pad, 0), min(s.stop + pad, n)) for s, n in zip(sl, shape))


def _ellipse_fit(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 6:
        return None
    return cv2.fitEllipse(c), c


def _relative_light_dark(hsv: np.ndarray, region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Light (white paint) and dark (symbols) pixels relative to the sign's own brightness."""
    v = hsv[..., 2].astype(float)
    sat = hsv[..., 1].astype(float)
    ref = np.percentile(v[region], 95) if region.any() else 255.0
    light = (sat <= 80) & (v >= 0.72 * ref)
    dark = v <= min(0.45 * ref, 110)
    return light, dark


def detect_prohibitory(img, valid, pixel_size, masks) -> list[Detection2D]:
    out = []
    H, W = valid.shape
    hsv = _hsv(img)
    red = cv2.morphologyEx(masks["red"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    min_px = max(int((0.25 / pixel_size) ** 2 * 0.15), 20)
    for comp, sl in _components(red, min_px):
        sl2 = _pad_slice(sl, (H, W), 2)
        local = _full(comp, sl, (H, W))[sl2]
        disk = ndimage.binary_fill_holes(local)
        area = disk.sum()
        if area < min_px:
            continue
        fit = _ellipse_fit(disk)
        if fit is None:
            continue
        (cx, cy), (ea, eb), _ang = fit[0]
        if min(ea, eb) < 4:
            continue
        axis_ratio = min(ea, eb) / max(ea, eb)
        fill = area / (np.pi * ea * eb / 4)
        ring_frac = local.sum() / area
        diameter = max(ea, eb) * pixel_size
        inner = disk & ~local
        if inner.sum() < 0.1 * area:
            continue
        white, dark = _relative_light_dark(hsv[sl2], disk)
        white_frac = (white & inner).sum() / max(inner.sum(), 1)
        dark_frac = (dark & inner).sum() / max(inner.sum(), 1)
        ok = (
            axis_ratio > 0.55
            and 0.82 < fill < 1.12
            and 0.15 < ring_frac < 0.8
            and 0.25 <= diameter <= 1.8
            and white_frac > 0.25
            and 0.02 < dark_frac < 0.6
        )
        if not ok:
            continue
        score = 0.5 + 0.2 * min(axis_ratio / 0.9, 1) + 0.15 * (1 - abs(ring_frac - 0.4) / 0.4) + 0.15 * min(white_frac / 0.5, 1)
        full = np.zeros((H, W), bool)
        full[sl2] = disk
        full = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        category, label = classify_prohibitory(inner, dark)
        ys, xs = np.nonzero(full)
        out.append(Detection2D(full, category, float(min(score, 0.99)), label, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return out


_FONT_FILES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]
_VALUES = {
    "speed_limit": [str(v) for v in range(5, 125, 5)],
    "weight_limit": [f"{v}t" for v in (2, 3, 5, 7, 8, 10, 13, 15, 20, 25, 30, 35, 40, 49, 50, 55)],
    "height_limit": [f"{v}m" for v in ("2", "2.2", "2.5", "2.8", "3", "3.2", "3.5", "3.8", "4", "4.2", "4.5", "4.8", "5", "5.5")],
}
_VALUES["width_limit"] = _VALUES["height_limit"]
_LABEL = {"speed_limit": "限速", "weight_limit": "限重", "height_limit": "限高", "width_limit": "限宽"}
_GLYPHS: dict = {}
_GH = 40


def _glyph(text: str) -> np.ndarray:
    if text in _GLYPHS:
        return _GLYPHS[text]
    from PIL import Image, ImageDraw, ImageFont

    font = None
    for f in _FONT_FILES:
        try:
            font = ImageFont.truetype(f, 64)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    img = Image.new("L", (64 * len(text) + 40, 120), 255)
    ImageDraw.Draw(img).text((20, 20), text, font=font, fill=0)
    a = np.asarray(img) < 128
    ys, xs = np.nonzero(a)
    a = a[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    w = max(int(round(a.shape[1] * _GH / a.shape[0])), 1)
    g = cv2.resize(a.astype(np.uint8) * 255, (w, _GH), interpolation=cv2.INTER_AREA) > 100
    _GLYPHS[text] = g
    return g


def _read_value(content: np.ndarray, families) -> tuple[Optional[str], str, float]:
    """Template-match the dark symbol crop against rendered limit values."""
    ys, xs = np.nonzero(content)
    if len(ys) < 15:
        return None, "", 0.0
    crop = content[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    if crop.shape[0] < 6:
        return None, "", 0.0
    w = max(int(round(crop.shape[1] * _GH / crop.shape[0])), 1)
    probe = cv2.resize(crop.astype(np.uint8) * 255, (w, _GH), interpolation=cv2.INTER_AREA) > 100
    best = (None, "", 0.0)
    for fam in families:
        for text in _VALUES[fam]:
            g = _glyph(text)
            if not 0.7 < g.shape[1] / w < 1.4:
                continue
            gg = cv2.resize(g.astype(np.uint8) * 255, (w, _GH), interpolation=cv2.INTER_NEAREST) > 100
            k = np.ones((3, 3), np.uint8)
            a = cv2.dilate(probe.astype(np.uint8), k).astype(bool)
            b = cv2.dilate(gg.astype(np.uint8), k).astype(bool)
            iou = (a & b).sum() / max((a | b).sum(), 1)
            if iou > best[2]:
                best = (fam, text, float(iou))
    return best


def classify_prohibitory(inner: np.ndarray, dark: np.ndarray) -> tuple[str, str]:
    """Limit type and value of a red-ring sign.

    Height limit signs carry small triangles above and below the number,
    width limit signs left and right; the number itself is read by
    template matching (speed ``60``, weight ``20t``, height ``4.5m`` ...).
    """
    ys, xs = np.nonzero(inner)
    if len(ys) == 0:
        return "prohibitory", ""
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(inner.sum() / np.pi)
    d = dark & inner
    dy, dx = np.nonzero(d)
    if len(dy) < 10:
        return "prohibitory", ""
    lab, nlab = ndimage.label(d, structure=np.ones((3, 3)))
    area_in = inner.sum()
    pos = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    arrows = np.zeros_like(d)
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == k
        a = comp.sum()
        if a > 0.06 * area_in or a < 4:
            continue
        yy, xx = np.nonzero(comp)
        ry = (yy.mean() + sl[0].start - cy) / r
        rx = (xx.mean() + sl[1].start - cx) / r
        hit = None
        if ry < -0.5 and abs(rx) < 0.2:
            hit = "top"
        elif ry > 0.5 and abs(rx) < 0.2:
            hit = "bottom"
        elif rx < -0.55 and abs(ry) < 0.2:
            hit = "left"
        elif rx > 0.55 and abs(ry) < 0.2:
            hit = "right"
        if hit:
            pos[hit] += 1
            arrows[sl] |= comp
    layout = None
    if pos["top"] and pos["bottom"] and not (pos["left"] or pos["right"]):
        layout = "height_limit"
    elif pos["left"] and pos["right"] and not (pos["top"] or pos["bottom"]):
        layout = "width_limit"
    band = d & ~arrows
    if layout:
        trials = [(layout, [layout], band)]
    else:
        # arrows may have blurred into the digits: also read the central bands only
        dy_all, dx_all = np.nonzero(d)
        ry_all = (dy_all - cy) / r
        rx_all = (dx_all - cx) / r
        vband = np.zeros_like(d)
        hband = np.zeros_like(d)
        vband[dy_all[np.abs(ry_all) < 0.4], dx_all[np.abs(ry_all) < 0.4]] = True
        hband[dy_all[np.abs(rx_all) < 0.4], dx_all[np.abs(rx_all) < 0.4]] = True
        trials = [
            (None, ["speed_limit", "weight_limit"], band),
            ("height_limit", ["height_limit"], vband),
            ("width_limit", ["width_limit"], hband),
        ]
    best = (None, None, "", 0.0)
    for lay, fams, content in trials:
        fam, text, score = _read_value(content, fams)
        if fam is not None and score > best[3]:
            best = (lay, fam, text, score)
    lay, fam, text, score = best
    if fam is not None and score >= 0.42:
        cat = lay or fam
        return cat, f"{_LABEL[cat]} {text}"
    if layout:
        return layout, _LABEL[layout]
    return "prohibitory", "禁令标志"


def _text_pixels(hsv: np.ndarray, inside: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Pixels clearly lighter and less saturated than the sign background (white text / border)."""
    if not background.any():
        return np.zeros_like(inside)
    v = hsv[..., 2].astype(float)
    sat = hsv[..., 1].astype(float)
    vb = np.median(v[background])
    sb = np.median(sat[background])
    return inside & (v >= vb * 1.18 + 8) & (sat <= sb * 0.8)


def detect_rect_signs(img, valid, pixel_size, masks) -> list[Detection2D]:
    out = []
    H, W = valid.shape
    hsv = _hsv(img)
    for color, category in (("blue", "road_name"), ("green", "guide")):
        m = cv2.morphologyEx(masks[color].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        min_px = max(int((0.15 / pixel_size) ** 2), 30)
        for comp, sl in _components(m, min_px):
            sl2 = _pad_slice(sl, (H, W), 2)
            local = _full(comp, sl, (H, W))[sl2]
            filled = ndimage.binary_fill_holes(local)
            cnts, _ = cv2.findContours(filled.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            (rcx, rcy), (rw, rh), _ = cv2.minAreaRect(c)
            if min(rw, rh) < 4:
                continue
            rect_fill = filled.sum() / (rw * rh)
            long_m = max(rw, rh) * pixel_size
            short_m = min(rw, rh) * pixel_size
            inside = filled
            text = _text_pixels(hsv[sl2], inside, local)
            white_frac = text.sum() / max(inside.sum(), 1)
            ntext = ndimage.label(text, structure=np.ones((3, 3)))[1]
            ok = (
                rect_fill > 0.78
                and 0.12 <= short_m <= 4.0
                and long_m <= 8.0
                and long_m / max(short_m, 1e-6) <= 8.0
                and 0.03 <= white_frac <= 0.65
                and ntext >= 2
            )
            if not ok:
                continue
            score = 0.5 + 0.25 * min((rect_fill - 0.78) / 0.17, 1) + 0.25 * min(white_frac / 0.2, 1)
            full = np.zeros((H, W), bool)
            full[sl2] = filled
            full = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            ys, xs = np.nonzero(full)
            label = "路牌" if category == "road_name" else "指路标志"
            out.append(Detection2D(full, category, float(min(score, 0.99)), label, (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return out


def detect_warning(img, valid, pixel_size, masks) -> list[Detection2D]:
    out = []
    H, W = valid.shape
    m = cv2.morphologyEx(masks["yellow"].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    min_px = max(int((0.3 / pixel_size) ** 2 * 0.2), 30)
    for comp, sl in _components(m, min_px):
        sl2 = _pad_slice(sl, (H, W), 6)
        local = _full(comp, sl, (H, W))[sl2]
        # the black border surrounds the yellow face: grow into dark pixels
        grown = ndimage.binary_fill_holes(local | (cv2.dilate(local.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & masks["dark"][sl2]))
        cnts, _ = cv2.findContours(grown.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        side = np.sqrt(grown.sum() * 4 / np.sqrt(3)) * pixel_size
        dark_in = (masks["dark"][sl2] & grown & ~local).sum() / max(grown.sum(), 1)
        yfill = ndimage.binary_fill_holes(local)
        ycnts, _ = cv2.findContours(yfill.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yapprox = cv2.approxPolyDP(max(ycnts, key=cv2.contourArea), 0.06 * cv2.arcLength(max(ycnts, key=cv2.contourArea), True), True)
        border_ratio = grown.sum() / max(yfill.sum(), 1)
        if len(approx) != 3 or len(yapprox) != 3 or not (0.3 <= side <= 2.0) or dark_in < 0.05 or border_ratio > 1.8:
            continue
        full = np.zeros((H, W), bool)
        full[sl2] = grown
        full = cv2.dilate(full.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        ys, xs = np.nonzero(full)
        out.append(Detection2D(full, "warning", 0.75, "警告标志", (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return out


class HeuristicSignDetector:
    name = "heuristic"

    def __init__(self, categories=None):
        self.categories = set(categories) if categories else None

    def detect(self, img: np.ndarray, valid: np.ndarray, pixel_size: float) -> list[Detection2D]:
        masks = color_masks(img)
        for k in masks:
            masks[k] &= valid
        dets = detect_prohibitory(img, valid, pixel_size, masks)
        dets += detect_rect_signs(img, valid, pixel_size, masks)
        dets += detect_warning(img, valid, pixel_size, masks)
        return dets
