"""Optional learned detector: any YOLOv5/v8-style ONNX export run with onnxruntime.

Class names come from ``<model>.names.txt`` / ``<model>.names.json`` next to
the model (or the model's own metadata).  Names are mapped to sign
categories; TT100K-style codes are understood (``pl60`` speed limit,
``ph4.5`` height limit, ``pm20`` weight limit, ``pw3`` width limit,
``p*`` prohibitory, ``w*`` warning, ``i*`` indicative/guide).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .heuristic import Detection2D


def category_for_name(name: str) -> Optional[str]:
    n = name.strip().lower()
    rules = [
        (r"^pl\d|speed", "speed_limit"),
        (r"^ph\d|height", "height_limit"),
        (r"^pm\d|weight|mass", "weight_limit"),
        (r"^pw\d|width", "width_limit"),
        (r"^p[a-z0-9]|prohib|no_|forbid", "prohibitory"),
        (r"^w[a-z0-9]|warn", "warning"),
        (r"^i[a-z0-9]|guide|direction|indic", "guide"),
        (r"road.?name|street.?sign|路牌", "road_name"),
        (r"sign", "other_sign"),
    ]
    for pat, cat in rules:
        if re.search(pat, n):
            return cat
    return None


def _load_names(model_path: Path, session) -> list[str]:
    for suffix in (".names.json", ".names.txt"):
        p = model_path.with_suffix(suffix)
        if p.is_file():
            if suffix.endswith("json"):
                data = json.loads(p.read_text(encoding="utf-8"))
                return [data[str(i)] for i in range(len(data))] if isinstance(data, dict) else list(data)
            return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = session.get_modelmeta().custom_metadata_map
    if "names" in meta:  # ultralytics stores a python dict literal
        names = re.findall(r"\d+:\s*'([^']*)'", meta["names"])
        if names:
            return names
    return []


class YoloOnnxDetector:
    name = "yolo"

    def __init__(self, model_path: str | Path, conf: float = 0.35, iou: float = 0.5, input_size: int = 640):
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else input_size
        self.names = _load_names(self.model_path, self.session)
        self.conf = conf
        self.iou = iou

    def _infer(self, img: np.ndarray):
        h, w = img.shape[:2]
        s = self.size / max(h, w)
        nh, nw = int(round(h * s)), int(round(w * s))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img[..., :3], (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0][0]
        if out.shape[0] < out.shape[1]:  # v8 layout (4 + nc, N) -> (N, 4 + nc)
            out = out.T
        nc = len(self.names) if self.names else out.shape[1] - 4
        if out.shape[1] == 5 + nc:  # v5 layout with objectness
            scores = out[:, 5:] * out[:, 4:5]
        else:
            scores = out[:, 4:]
        cls = scores.argmax(axis=1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        boxes = out[keep, :4] / s
        return boxes, conf[keep], cls[keep]

    def detect(self, img: np.ndarray, valid: np.ndarray, pixel_size: float) -> list[Detection2D]:
        boxes, conf, cls = self._infer(img)
        if len(boxes) == 0:
            return []
        xywh = [[float(b[0] - b[2] / 2), float(b[1] - b[3] / 2), float(b[2]), float(b[3])] for b in boxes]
        idx = cv2.dnn.NMSBoxes(xywh, conf.astype(float).tolist(), self.conf, self.iou)
        H, W = valid.shape
        out = []
        for i in np.array(idx).ravel():
            name = self.names[cls[i]] if cls[i] < len(self.names) else str(cls[i])
            cat = category_for_name(name)
            if cat is None:
                continue
            x, y, w, h = xywh[i]
            x0, y0 = max(int(x), 0), max(int(y), 0)
            x1, y1 = min(int(np.ceil(x + w)), W), min(int(np.ceil(y + h)), H)
            if x1 <= x0 or y1 <= y0:
                continue
            m = np.zeros((H, W), bool)
            m[y0:y1, x0:x1] = True
            out.append(Detection2D(m & valid, cat, float(conf[i]), name, (x0, y0, x1, y1), source="yolo"))
        return out
