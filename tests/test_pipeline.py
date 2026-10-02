"""End-to-end: detect and remove the signs of the synthetic street scene."""

import numpy as np

from conftest import needs_bridge
from sensitive3d.detect.detector import SignDetector
from sensitive3d.io.dataset import Dataset
from sensitive3d.pipeline import PipelineConfig, run


def _check_detection(report, ground_truth):
    assert len(report["regions"]) == len(ground_truth)
    for g in ground_truth:
        d = [np.linalg.norm(np.array(r["center"]) - g["center"]) for r in report["regions"]]
        r = report["regions"][int(np.argmin(d))]
        assert min(d) < 0.15, g["id"]
        assert r["category"] == g["category"], (g["id"], r["category"])
        assert abs(r["width"] - g["width"]) < 0.15
        assert r["mount"] == g["mount"], (g["id"], r["mount"])
    assert report["residual_detections"] == []


def _no_signs_left(out_dir):
    ds = Dataset(out_dir)
    ds.scan()
    parts = []
    for f in ds.files:
        if f.has_leaf:
            m = ds.load(f.rel)
            parts += [(p, m.texture_of(p)) for p in m.leaf_parts()]
    return SignDetector().detect(parts).regions == []


def test_obj_pipeline(synthetic, ground_truth, tmp_path):
    rep = run(synthetic / "obj", tmp_path / "out", work_dir=tmp_path / "work")
    _check_detection(rep, ground_truth)
    assert rep["faces_removed"] > 0 and rep["faces_added"] > 0
    assert (tmp_path / "work" / "previews" / "region_0_after.png").is_file()
    assert _no_signs_left(tmp_path / "out")


def test_texture_only_mode_keeps_geometry(synthetic, tmp_path):
    cfg = PipelineConfig()
    cfg.remove_geometry = False
    cfg.previews = False
    rep = run(synthetic / "glb", tmp_path / "out", cfg, work_dir=tmp_path / "work")
    assert rep["faces_removed"] == 0 and rep["texels_painted"] > 0


def test_predefined_regions_skip_detection(synthetic, ground_truth, tmp_path):
    g = ground_truth[0]
    cfg = PipelineConfig()
    cfg.previews = False
    cfg.verify = False
    cfg.regions = [{"center": g["center"], "normal": g["normal"], "width": g["width"], "height": g["height"], "category": g["category"]}]
    rep = run(synthetic / "obj", tmp_path / "out", cfg, work_dir=tmp_path / "work")
    assert len(rep["regions"]) == 1 and rep["regions"][0]["mount"] == "pole"


@needs_bridge
def test_osgb_pipeline_all_lods(synthetic, ground_truth, tmp_path):
    src = synthetic / "osgb"
    out = tmp_path / "out"
    rep = run(src, out, work_dir=tmp_path / "work")
    _check_detection(rep, ground_truth)
    # every file of the dataset is present in the output, the layout is unchanged
    a = sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file())
    b = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert a == b
    # coarser LOD files were repaired as well
    modified = rep["files_modified"]
    assert modified > rep["leaf_files"] // 2
    assert _no_signs_left(out)
    # the PagedLOD hierarchy is still intact
    ds = Dataset(out)
    files = ds.scan()
    assert sum(1 for f in files if f.children) == sum(1 for f in Dataset(src).scan() if f.children)
