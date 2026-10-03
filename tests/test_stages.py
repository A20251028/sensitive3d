"""Stage split: scan -> detect (read only) -> review -> repair, plus run control."""

import json
import threading

import numpy as np
import pytest

from sensitive3d.io.dataset import Dataset
from sensitive3d.io.inputs import file_manifest
from sensitive3d.pipeline import Pipeline, PipelineConfig, ScanBlocked
from sensitive3d.review import ReviewError, default_decisions, resolve_targets
from sensitive3d.runctx import BudgetExceeded, Cancelled, RunContext, RunLimits


def _cand(cid, status="auto", ok=True, mount="pole", conf="high", op="geometry"):
    return {
        "id": cid,
        "status": status,
        "category": "speed_limit",
        "category_label": "限速标志",
        "label": "限速 60",
        "score": 0.95,
        "review_reasons": [] if status == "auto" else ["置信度低"],
        "mapping": {"ok": ok, "status": "ok" if ok else "mapping_failed"},
        "mount": {"type": mount, "confidence": conf, "ground_z": 0.0, "pole_xy": [0.0, 0.0], "metrics": {}},
        "region": {"center": [0, 0, 2], "normal": [1, 0, 0], "width": 0.8, "height": 0.8},
        "suggested": {"accept": status == "auto" and ok, "operation": op if ok else "none"},
    }


# ---------------------------------------------------------------------------
# review rules (no data needed)
# ---------------------------------------------------------------------------


def test_auto_mode_only_takes_auto_candidates():
    det = {"candidates": [_cand(0), _cand(1, status="review")]}
    targets, skipped = resolve_targets(det, None, auto=True)
    assert [t.candidate_id for t in targets] == [0]
    assert skipped[0]["id"] == 1 and skipped[0]["reason"] == "待审核"
    # without auto nothing runs unless a person decided
    targets, _ = resolve_targets(det, None, auto=False)
    assert targets == []


def test_human_decisions_and_operations():
    det = {"candidates": [_cand(0), _cand(1, status="review", conf="low")]}
    rev = {"decisions": [{"id": 0, "accept": False}, {"id": 1, "accept": True, "operation": "texture", "margin": 0.1}]}
    targets, skipped = resolve_targets(det, rev, auto=True)
    assert [(t.candidate_id, t.operation, t.decided_by) for t in targets] == [(1, "texture", "review")]
    assert abs(targets[0].region.half_u - 0.5) < 1e-9
    assert skipped == [{"id": 0, "reason": "人工拒绝", "status": "auto"}]


def test_unreliable_location_is_refused():
    det = {"candidates": [_cand(0, ok=False)]}
    for op in ("geometry", "texture"):
        with pytest.raises(ReviewError):
            resolve_targets(det, {"decisions": [{"id": 0, "accept": True, "operation": op}]}, auto=False)
    det = {"candidates": [_cand(0, mount="unknown", conf="low")]}
    with pytest.raises(ReviewError):
        resolve_targets(det, {"decisions": [{"id": 0, "accept": True, "operation": "geometry"}]}, auto=False)
    with pytest.raises(ReviewError):
        resolve_targets(det, {"decisions": [{"id": 7, "accept": True}]}, auto=False)


def test_review_template_leaves_review_items_undecided():
    det = {"candidates": [_cand(0), _cand(1, status="review")]}
    d = default_decisions(det)
    assert d[0]["accept"] is True and d[1]["accept"] is None


# ---------------------------------------------------------------------------
# run control
# ---------------------------------------------------------------------------


def test_run_context_cancel_and_budget():
    ev = threading.Event()
    ctx = RunContext(RunLimits(max_seconds=1000), cancel_event=ev)
    ctx.check("a")
    ev.set()
    with pytest.raises(Cancelled):
        ctx.check("b")
    ctx = RunContext(RunLimits(max_memory_mb=1))
    with pytest.raises(BudgetExceeded):
        ctx.check("mem")
    ctx = RunContext(RunLimits(max_decoded_mb=10))
    ctx.check_decoded(5, "ok")
    with pytest.raises(BudgetExceeded):
        ctx.check_decoded(50, "too much")


# ---------------------------------------------------------------------------
# stages on the synthetic scene
# ---------------------------------------------------------------------------


def _osgb_or_obj(synthetic):
    return synthetic / ("osgb" if (synthetic / "osgb").is_dir() else "obj")


def test_detect_is_read_only_and_has_provenance(synthetic, tmp_path):
    src = _osgb_or_obj(synthetic)
    before = file_manifest(src)
    pipe = Pipeline(PipelineConfig())
    ds, scan = pipe.scan(src, tmp_path)
    assert scan["ok_for_repair"] and (tmp_path / "scan.json").is_file() and (tmp_path / "input_manifest.json").is_file()
    det = pipe.detect(ds, tmp_path)
    assert len(det["candidates"]) == 5
    for c in det["candidates"]:
        p = c["provenance"]
        assert p["tiles"] and p["files"] and p["camera"] and p["bbox2d"] and p["context_camera"]
        assert c["mapping"]["ok"] and c["mount"]["type"] in ("pole", "wall")
        assert c["evidence"]["closeup"] and (tmp_path / c["evidence"]["overlay"]).is_file()
    assert det["metrics"]["recall"] is None and det["metrics"]["f1"] is None
    assert file_manifest(src)["files"] == before["files"]


def test_texture_only_keeps_geometry_and_protected_area(synthetic, tmp_path):
    src = synthetic / "obj"
    pipe = Pipeline(PipelineConfig())
    ds, _ = pipe.scan(src, tmp_path / "w")
    det = pipe.detect(ds, tmp_path / "w")
    first = det["candidates"][0]
    c = np.array(first["region"]["center"])
    review = {
        "decisions": [{"id": x["id"], "accept": x["id"] == first["id"], "operation": "texture"} for x in det["candidates"]],
        # a protected box far from every sign: must stay unchanged
        "protected": [{"min": (c + [-12, 3, -3]).tolist(), "max": (c + [-9, 6, 3]).tolist()}],
    }
    rep = pipe.repair(ds, det, review, tmp_path / "out", tmp_path / "w")
    assert rep["faces_removed"] == 0 and rep["texels_painted"] > 0
    assert rep["files"] and all(f["readback"]["ok"] for f in rep["files"]), rep["files"]
    assert all(not f["geometry_changed"] for f in rep["files"])
    assert rep["protected_check"] and all(p["ok"] for p in rep["protected_check"])
    assert rep["input_check"]["ok"] is True
    assert {s["id"] for s in rep["skipped"]} == {x["id"] for x in det["candidates"]} - {first["id"]}


def test_geometry_refused_on_blocking_scan(synthetic, tmp_path):
    src = tmp_path / "bad"
    src.mkdir()
    for p in (synthetic / "obj").iterdir():
        (src / p.name).write_bytes(p.read_bytes())
    (src / "broken.obj").write_text("v 0 0 0\nv 1 0 0\nf 1 2 9\n", encoding="utf-8")  # invalid face index
    pipe = Pipeline(PipelineConfig())
    ds, scan = pipe.scan(src, tmp_path / "w")
    assert not scan["ok_for_repair"] and scan["blocking_errors"]
    det = {"candidates": [_cand(0)]}
    with pytest.raises(ScanBlocked):
        pipe.repair(ds, det, None, tmp_path / "out", tmp_path / "w", auto=True)


def test_cancel_stops_detection_and_keeps_evidence(synthetic, tmp_path):
    ev = threading.Event()
    ev.set()
    pipe = Pipeline(PipelineConfig(), cancel_event=ev)
    ds = Dataset(_osgb_or_obj(synthetic))
    ds.scan()
    det = pipe.detect(ds, tmp_path)
    assert det["stopped"]["reason"] == "cancelled"
    assert json.loads((tmp_path / "detection.json").read_text(encoding="utf-8"))["stopped"]
