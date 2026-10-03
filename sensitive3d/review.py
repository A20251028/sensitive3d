"""Review step: turn detection candidates + human decisions into repair targets.

A detection report (``detection.json``) lists candidates with a status:

* ``auto``   – confident, mapped to 3D, mounting known: may be repaired without
  a person in the fully automatic mode;
* ``review`` – low confidence, failed / uncertain 2D -> 3D mapping, uncertain
  mounting or a category that was not selected: never repaired unless a
  person accepts it.

A review file (``review.json``) holds the decisions::

    {"decisions": [{"id": 0, "accept": true, "operation": "geometry"},
                   {"id": 3, "accept": true, "operation": "texture", "margin": 0.05},
                   {"id": 5, "accept": false}],
     "protected": [{"name": "门牌号旁的窗户", "min": [x, y, z], "max": [x, y, z]}]}

``operation`` is ``geometry`` (remove plate / pole geometry, fill holes and
repaint) or ``texture`` (repaint only; geometry and UVs stay byte-identical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .detect.regions import SignRegion
from .repair.geometry import RegionGeometry

OPERATIONS = ("geometry", "texture")


class ReviewError(ValueError):
    """A decision that must not be executed (e.g. geometry removal on an unreliable location)."""


@dataclass
class RepairTarget:
    candidate_id: int
    region: SignRegion
    operation: str  # geometry | texture
    mount: RegionGeometry
    decided_by: str  # "auto" | "review"


def default_decisions(detection: dict) -> list[dict]:
    """A review template: accept what was auto-classified, leave the rest undecided."""
    out = []
    for c in detection.get("candidates", []):
        sug = c.get("suggested") or {}
        out.append(
            {
                "id": c["id"],
                "accept": bool(sug.get("accept")) if c.get("status") == "auto" else None,
                "operation": sug.get("operation", "texture"),
                "status": c.get("status"),
                "label": c.get("label") or c.get("category_label"),
                "reasons": c.get("review_reasons", []),
            }
        )
    return out


def parse_protected(review: Optional[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    boxes = []
    for i, p in enumerate((review or {}).get("protected", []) or []):
        try:
            lo = np.asarray(p["min"], float).reshape(3)
            hi = np.asarray(p["max"], float).reshape(3)
        except (KeyError, ValueError, TypeError) as e:
            raise ReviewError(f"保护区域 #{i} 格式错误 (需要 min/max 三维坐标): {e}") from e
        if np.any(hi < lo):
            raise ReviewError(f"保护区域 #{i} 的 max 小于 min")
        boxes.append((lo, hi))
    return boxes


def resolve_targets(detection: dict, review: Optional[dict], auto: bool) -> tuple[list[RepairTarget], list[dict]]:
    """Repair targets from candidates + decisions.

    ``auto`` (fully automatic mode) accepts candidates whose status is
    ``auto`` and that have no explicit decision.  Returns ``(targets,
    skipped)``; ``skipped`` documents every candidate that is not repaired and
    why.  Raises :class:`ReviewError` for decisions that must not run.
    """
    decisions = {}
    for d in (review or {}).get("decisions", []) or []:
        if "id" not in d:
            raise ReviewError("审核决定缺少 id")
        decisions[int(d["id"])] = d
    known = {int(c["id"]) for c in detection.get("candidates", [])}
    unknown = sorted(set(decisions) - known)
    if unknown:
        raise ReviewError(f"审核决定引用了不存在的候选: {unknown}")
    targets: list[RepairTarget] = []
    skipped: list[dict] = []
    for c in detection.get("candidates", []):
        cid = int(c["id"])
        d = decisions.get(cid)
        sug = c.get("suggested") or {}
        mapping_ok = bool((c.get("mapping") or {}).get("ok"))
        if d is None or d.get("accept") is None:
            if auto and c.get("status") == "auto" and sug.get("accept"):
                op = sug.get("operation", "texture")
                by = "auto"
            else:
                skipped.append({"id": cid, "reason": "待审核" if c.get("status") == "review" else "未选择", "status": c.get("status")})
                continue
        elif not d.get("accept"):
            skipped.append({"id": cid, "reason": "人工拒绝", "status": c.get("status")})
            continue
        else:
            op = d.get("operation") or sug.get("operation") or "texture"
            by = "review"
        if op not in OPERATIONS:
            raise ReviewError(f"候选 #{cid}: 未知操作 {op!r} (只能是 geometry / texture)")
        if not mapping_ok:
            # the 3D location is only provisional: do not touch geometry or paint blindly
            raise ReviewError(f"候选 #{cid}: 二维→三维映射失败, 位置不可靠, 不能执行{'几何删除' if op == 'geometry' else '纹理修复'}")
        region = SignRegion.from_dict(c["region"])
        region.id = cid
        margin = float(d.get("margin", 0.0)) if d else 0.0
        if margin:
            if not (-0.5 < margin < 2.0):
                raise ReviewError(f"候选 #{cid}: 外扩范围 {margin} m 不合理")
            region.half_u = max(region.half_u + margin, 0.02)
            region.half_v = max(region.half_v + margin, 0.02)
        mount = RegionGeometry.from_dict(cid, c.get("mount"))
        if op == "geometry" and mount.mount not in ("pole", "wall", "free"):
            raise ReviewError(f"候选 #{cid}: 安装方式未知, 不能执行几何删除")
        targets.append(RepairTarget(cid, region, op, mount, by))
    return targets, skipped
