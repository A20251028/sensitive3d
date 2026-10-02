import numpy as np

from sensitive3d.detect.heuristic import HeuristicSignDetector
from sensitive3d.synthetic import prohibitory_graphic, road_name_graphic


def _scene(graphic, size_px, bg=(120, 118, 110), canvas=360, shade=1.0):
    img = np.empty((canvas, canvas, 3), np.uint8)
    img[:] = bg
    import cv2

    g = cv2.resize(graphic, (size_px[0], size_px[1]), interpolation=cv2.INTER_AREA)
    y0 = (canvas - size_px[1]) // 2
    x0 = (canvas - size_px[0]) // 2
    if graphic.shape[0] == graphic.shape[1]:  # disk: only paste inside the circle
        yy, xx = np.mgrid[: size_px[1], : size_px[0]]
        disk = (xx - size_px[0] / 2) ** 2 + (yy - size_px[1] / 2) ** 2 <= (size_px[0] / 2) ** 2
        region = img[y0 : y0 + size_px[1], x0 : x0 + size_px[0]]
        region[disk] = g[disk]
    else:
        img[y0 : y0 + size_px[1], x0 : x0 + size_px[0]] = g
    return (img * shade).astype(np.uint8)


def _detect(img, ps=0.005):
    return HeuristicSignDetector().detect(img, np.ones(img.shape[:2], bool), ps)


def test_speed_limit_sign_read():
    d = _detect(_scene(prohibitory_graphic("60", "speed"), (160, 160)))
    assert len(d) == 1 and d[0].category == "speed_limit" and "60" in d[0].label


def test_shaded_weight_and_height_limit():
    d = _detect(_scene(prohibitory_graphic("20t", "weight"), (160, 160), shade=0.62))
    assert len(d) == 1 and d[0].category == "weight_limit"
    d = _detect(_scene(prohibitory_graphic("4.5m", "height"), (160, 160)))
    assert len(d) == 1 and d[0].category == "height_limit"


def test_road_name_plate():
    d = _detect(_scene(road_name_graphic("人民路", "RENMIN RD"), (280, 90)))
    assert len(d) == 1 and d[0].category == "road_name"


def test_distractors_are_ignored():
    img = np.empty((360, 360, 3), np.uint8)
    img[:] = (120, 118, 110)
    img[100:200, 60:300] = (30, 75, 165)  # plain blue door, no text
    img[220:300, 40:200] = (175, 30, 30)  # red car side, not a ring
    assert _detect(img) == []
