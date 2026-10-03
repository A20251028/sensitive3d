"""Input hygiene, strict dataset scan and the PagedLOD reference graph."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import needs_bridge
from fixtures_bad import (
    SIDECARS,
    add_sidecars,
    case_mismatch_dataset,
    clean_dataset,
    cycle_dataset,
    garbage_dataset,
    missing_ref_dataset,
    multi_parent_dataset,
    zip_dataset,
)
from sensitive3d.io import inputs, osgb
from sensitive3d.io.dataset import Dataset, FileInfo
from sensitive3d.io.lodgraph import LodGraph, resolve_ref
from sensitive3d.webpreview import lod_cut

# ---------------------------------------------------------------------------
# inputs: pure helpers
# ---------------------------------------------------------------------------


def test_is_ignored():
    for rel in ("._a.osgb", "Data/._a.osgb", ".DS_Store", "x/.DS_Store", "__MACOSX/Data/a.osgb", "Thumbs.db", "a\\Desktop.ini", "a\\__MACOSX\\b"):
        assert inputs.is_ignored(rel), rel
    for rel in ("a.osgb", "Data/T/T.osgb", ".hidden/a.osgb", "_a.osgb", "MACOSX/a", "a._b.osgb", "metadata.xml"):
        assert not inputs.is_ignored(rel), rel


def test_safe_rel_matches_server_semantics():
    assert inputs.safe_rel("a/b.osgb").as_posix() == "a/b.osgb"
    assert inputs.safe_rel("a\\b\\c.osgb").as_posix() == "a/b/c.osgb"
    assert inputs.safe_rel("./a//b.osgb").as_posix() == "a/b.osgb"
    for bad in ("../x", "a/../../x", "/etc/passwd", "C:\\x\\y", "c:x", "", ".", "a/\x00b"):
        assert inputs.safe_rel(bad) is None, bad


def test_iter_files_and_manifest(tmp_path):
    root = tmp_path / "src"
    (root / "Data/T").mkdir(parents=True)
    (root / "Data/T/T.osgb").write_bytes(b"x" * 10)
    (root / "metadata.xml").write_text("m")
    add_sidecars(root)
    outside = tmp_path / "outside.osgb"
    outside.write_bytes(b"secret")
    (root / "Data/T/link_out.osgb").symlink_to(outside)
    (root / "Data/T/link_in.osgb").symlink_to(root / "Data/T/T.osgb")
    (root / "Data/linkdir").symlink_to(root / "Data/T", target_is_directory=True)

    ignored, skipped = [], []
    rels = [p.relative_to(root).as_posix() for p in inputs.iter_files(root, ignored, skipped)]
    assert rels == ["Data/T/T.osgb", "Data/T/link_in.osgb", "metadata.xml"]
    assert sorted(ignored) == sorted(SIDECARS)
    assert {s["rel"] for s in skipped} == {"Data/T/link_out.osgb", "Data/linkdir"}

    man = inputs.file_manifest(root)
    assert [f["rel"] for f in man["files"]] == rels
    assert man["total_bytes"] == 10 + 10 + 1 and all(len(f["sha256"]) == 64 for f in man["files"])
    assert inputs.verify_manifest(man, root)["ok"]

    # sidecars appearing or changing do not matter, real files do
    (root / "Data/T/._new.osgb").write_bytes(b"x")
    (root / ".DS_Store").write_bytes(b"changed")
    assert inputs.verify_manifest(man, root)["ok"]
    (root / "metadata.xml").write_text("M")
    (root / "extra.txt").write_text("e")
    (root / "Data/T/link_in.osgb").unlink()
    v = inputs.verify_manifest(man, root)
    assert not v["ok"] and v["changed"] == ["metadata.xml"] and v["added"] == ["extra.txt"] and v["missing"] == ["Data/T/link_in.osgb"]


def test_copy_tree_skips_sidecars_and_refuses_nesting(tmp_path):
    src = tmp_path / "src"
    (src / "Data/T").mkdir(parents=True)
    (src / "Data/T/T.osgb").write_bytes(b"osgb")
    add_sidecars(src)
    before = inputs.file_manifest(src)
    sidecar_bytes = {r: (src / r).read_bytes() for r in SIDECARS}

    res = inputs.copy_tree(src, tmp_path / "dst")
    assert [f["rel"] for f in res["files"]] == ["Data/T/T.osgb"]
    assert res["files"][0]["sha256"] == before["files"][0]["sha256"]
    assert sorted(res["ignored"]) == sorted(SIDECARS)
    out = sorted(p.relative_to(tmp_path / "dst").as_posix() for p in (tmp_path / "dst").rglob("*") if p.is_file())
    assert out == ["Data/T/T.osgb"]
    # the source is untouched, sidecars included
    assert inputs.verify_manifest(before, src)["ok"]
    assert all((src / r).read_bytes() == b for r, b in sidecar_bytes.items())

    with pytest.raises(ValueError):
        inputs.copy_tree(src, src / "out")
    with pytest.raises(ValueError):
        inputs.copy_tree(src, src)
    with pytest.raises(ValueError):
        inputs.copy_tree(src / "Data", src)
    assert inputs.verify_manifest(before, src)["ok"]


def test_safe_extract_zip(tmp_path):
    src = tmp_path / "src"
    (src / "Data/T").mkdir(parents=True)
    (src / "Data/T/T.osgb").write_bytes(b"osgb" * 100)
    (src / "metadata.xml").write_text("m")
    add_sidecars(src)
    z = tmp_path / "in.zip"
    zip_dataset(src, z, prefix="set/")
    dst = tmp_path / "jobs" / "input"
    res = inputs.safe_extract_zip(z, dst)
    files = sorted(p.relative_to(dst).as_posix() for p in dst.rglob("*") if p.is_file())
    assert files == ["set/Data/T/T.osgb", "set/metadata.xml"]
    assert res["files"] == 2 and res["bytes"] == 401
    assert sorted(res["ignored"]) == sorted("set/" + r for r in SIDECARS)
    assert len(res["rejected"]) == 4
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "jobs" / "evil.txt").exists() and not (tmp_path / "evil2.txt").exists()

    with pytest.raises(ValueError, match="文件数量"):
        inputs.safe_extract_zip(z, tmp_path / "d2", max_files=1)
    with pytest.raises(ValueError, match="总大小"):
        inputs.safe_extract_zip(z, tmp_path / "d3", max_bytes=100)


def test_safe_extract_zip_duplicates_never_overwrite(tmp_path):
    z = tmp_path / "dup.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("d/a.txt", b"1")
        with pytest.warns(UserWarning):
            zf.writestr("d/a.txt", b"2")
        zf.writestr("d/A.txt", b"3")  # collides only on case-insensitive file systems (macOS)
    dst = tmp_path / "out"
    res = inputs.safe_extract_zip(z, dst)
    assert "d/a.txt" in res["rejected"]
    assert res["files"] + len(res["rejected"]) == 3
    assert (dst / "d/a.txt").read_bytes() == b"1"


def test_safe_extract_zip_lying_header(tmp_path):
    """Sizes are enforced while writing, not only from the (forgeable) header."""
    z = tmp_path / "bomb.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.bin", b"\x00" * 50_000)
    data = bytearray(z.read_bytes())
    # patch the declared uncompressed size in the central directory to 10 bytes
    cd = data.rfind(b"PK\x01\x02")
    data[cd + 24 : cd + 28] = (10).to_bytes(4, "little")
    z.write_bytes(bytes(data))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        inputs.safe_extract_zip(z, tmp_path / "out", max_bytes=1000)
    out = tmp_path / "out" / "a.bin"
    assert not out.exists() or out.stat().st_size <= 1000


# ---------------------------------------------------------------------------
# lodgraph: pure unit tests
# ---------------------------------------------------------------------------


def test_resolve_ref():
    assert resolve_ref("Data/T/T.osgb", "sub/T_L1.osgb") == "Data/T/sub/T_L1.osgb"
    assert resolve_ref("Data/T/sub/L1.osgb", "L2_0.osgb") == "Data/T/sub/L2_0.osgb"
    assert resolve_ref("Data/T/sub/L1.osgb", "..\\T.osgb") == "Data/T/T.osgb"
    assert resolve_ref("Data/T/T.osgb", "./a/./b.osgb") == "Data/T/a/b.osgb"
    assert resolve_ref("root.osgb", "Data/T/T.osgb") == "Data/T/T.osgb"
    for bad in ("../../../x.osgb", "/abs/x.osgb", "C:\\x.osgb", "", "http://x/y.osgb"):
        assert resolve_ref("Data/T/T.osgb", bad) is None, bad


def test_lodgraph_structure():
    children = {
        "Data/T/T.osgb": ["sub/L1.osgb"],
        "Data/T/sub/L1.osgb": ["L2_0.osgb", "L2_1.osgb", "L2_0.osgb"],  # duplicate ref
        "Data/U/U.osgb": ["sub/L1.osgb", "gone.osgb", "../../../escape.osgb"],
        "Data/U/sub/L1.osgb": ["L2_0.osgb", "../../T/sub/l2_1.OSGB"],  # cross-tile, wrong case
        "Data/C/A.osgb": ["B.osgb"],
        "Data/C/B.osgb": ["A.osgb", "Z.osgb"],
        "Data/C/S.osgb": ["S.osgb"],
    }
    existing = list(children) + ["Data/T/sub/L2_0.osgb", "Data/T/sub/L2_1.osgb", "Data/U/sub/L2_0.osgb", "Data/C/Z.osgb", "Data/O/orphan.osgb"]
    g = LodGraph(children, existing)
    assert g.edges["Data/T/sub/L1.osgb"] == ["Data/T/sub/L2_0.osgb", "Data/T/sub/L2_1.osgb"]
    assert g.edges["Data/U/sub/L1.osgb"] == ["Data/U/sub/L2_0.osgb", "Data/T/sub/L2_1.osgb"]
    assert len(g.warnings) == 1 and "l2_1.OSGB" in g.warnings[0]
    assert [(m["ref"], m["resolved"]) for m in g.missing] == [("gone.osgb", "Data/U/gone.osgb"), ("../../../escape.osgb", None)]
    assert g.multi_parent == [{"rel": "Data/T/sub/L2_1.osgb", "parents": ["Data/T/sub/L1.osgb", "Data/U/sub/L1.osgb"]}]
    assert g.roots == ["Data/C/S.osgb", "Data/O/orphan.osgb", "Data/T/T.osgb", "Data/U/U.osgb"]
    assert g.cycles == [["Data/C/A.osgb", "Data/C/B.osgb"], ["Data/C/S.osgb"]]
    assert g.unreachable == ["Data/C/A.osgb", "Data/C/B.osgb", "Data/C/Z.osgb"]
    assert g.depth["Data/T/sub/L2_1.osgb"] == 2 and g.depth["Data/U/sub/L1.osgb"] == 1
    assert g.levels()[2] == ["Data/T/sub/L2_0.osgb", "Data/T/sub/L2_1.osgb", "Data/U/sub/L2_0.osgb"]
    assert "Data/T/sub/L2_0.osgb" in g.leaves and "Data/C/S.osgb" not in g.leaves
    assert g.descendants("Data/C/A.osgb") == ["Data/C/B.osgb", "Data/C/Z.osgb"]
    assert g.subtree("Data/C/S.osgb") == ["Data/C/S.osgb"]
    assert g.subtree("Data/U/U.osgb") == ["Data/U/U.osgb", "Data/U/sub/L1.osgb", "Data/U/sub/L2_0.osgb", "Data/T/sub/L2_1.osgb"]
    assert g.in_cycle("Data/C/A.osgb", "Data/C/B.osgb") and not g.in_cycle("Data/T/T.osgb", "Data/T/sub/L1.osgb")
    json.dumps(g.to_dict())


def test_lodgraph_long_chain_and_big_cycle_do_not_recurse():
    n = 5000
    chain = {f"f{i}.osgb": [f"f{i + 1}.osgb"] for i in range(n)}
    g = LodGraph(chain, existing=[f"f{i}.osgb" for i in range(n + 1)])
    assert g.depth[f"f{n}.osgb"] == n and g.cycles == []
    ring = {f"r{i}.osgb": [f"r{(i + 1) % n}.osgb"] for i in range(n)}
    g = LodGraph(ring)
    assert g.roots == [] and len(g.cycles) == 1 and len(g.cycles[0]) == n and len(g.unreachable) == n
    assert len(g.descendants("r0.osgb")) == n - 1


def _fi(rel, children=(), tris=2, ok=True):
    return FileInfo(rel, "t", num_triangles=tris, children=list(children), ok=ok, has_leaf=not children)


def test_lod_cut_dedupes_multi_parent_and_survives_cycles():
    files = [
        _fi("P.osgb", ["A.osgb", "B.osgb"]),
        _fi("A.osgb", ["S.osgb"]),
        _fi("B.osgb", ["B1.osgb"]),
        _fi("B1.osgb", ["S.osgb"]),
        _fi("S.osgb"),
        _fi("C.osgb", ["D.osgb"]),
        _fi("D.osgb", ["C.osgb"]),
        _fi("E.osgb", ["E.osgb", "E1.osgb"]),
        _fi("E1.osgb"),
    ]
    rels, depth = lod_cut(files, 10_000)
    assert sorted(rels) == ["E1.osgb", "S.osgb"] and len(rels) == len(set(rels))
    # budget stops the descent at a coarser complete level
    rels, depth = lod_cut(files, 4)
    assert sorted(rels) == ["E.osgb", "P.osgb"] and depth == 0


def test_lod_cut_same_names_in_different_directories():
    files = [
        _fi("Data/T/T.osgb", ["sub/L1.osgb"]),
        _fi("Data/T/sub/L1.osgb", ["L2.osgb"]),
        _fi("Data/T/sub/L2.osgb"),
        _fi("Data/U/U.osgb", ["sub/L1.osgb"]),
        _fi("Data/U/sub/L1.osgb", ["L2.osgb"]),
        _fi("Data/U/sub/L2.osgb"),
    ]
    rels, depth = lod_cut(files, 10_000)
    assert rels == ["Data/T/sub/L2.osgb", "Data/U/sub/L2.osgb"] and depth == 2


def test_lod_cut_keeps_parent_of_failed_child_and_only_back_edges():
    files = [
        _fi("R.osgb", ["K.osgb"]),
        _fi("K.osgb", ok=False),
        _fi("R2.osgb", ["Q.osgb"]),
        _fi("Q.osgb", ["Q1.osgb"]),
        _fi("Q1.osgb", ["Q.osgb"]),  # back reference to its parent
    ]
    rels, _ = lod_cut(files, 10_000)
    assert sorted(rels) == ["Q1.osgb", "R.osgb"]
    # only cycles, no root at all: every readable file, terminates
    rels, _ = lod_cut([_fi("a", ["b"]), _fi("b", ["a"])], 10)
    assert rels == ["a", "b"]


# ---------------------------------------------------------------------------
# Dataset.scan with fabricated bridge records (old and new bridge, no binary needed)
# ---------------------------------------------------------------------------


def _fake_dataset(tmp_path, names):
    root = tmp_path / "ds"
    for n in names:
        p = root / n
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fake")
    return root


def _geom(lo, hi, finer=False, **kw):
    return {"has_finer": finer, "num_triangles": 2, "depth": 1 if finer else 0, "min": lo, "max": hi, **kw}


def test_scan_old_bridge_records(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path, ["Data/T/T.osgb", "Data/T/T_L1.osgb", "Data/T/._T.osgb", ".DS_Store"])

    def fake(paths, batch=200):
        out = []
        for p in map(str, paths):
            if p.endswith("T_L1.osgb"):
                out.append({"file": p, "ok": True, "geometries": [_geom([0, 0, 0], [1, 1, 1])], "children": []})
            else:
                out.append({"file": p, "ok": True, "geometries": [_geom([0, 0, 0], [2, 2, 2], finer=True)], "children": ["T_L1.osgb"]})
        return out

    monkeypatch.setattr(osgb, "scan_osgb", fake)
    ds = Dataset(root)
    files = ds.scan()
    assert [f.rel for f in files] == ["Data/T/T.osgb", "Data/T/T_L1.osgb"]
    rep = ds.report
    assert rep.ok_for_repair and rep.blocking_errors == []
    assert sorted(rep.ignored) == [".DS_Store", "Data/T/._T.osgb"]
    assert rep.levels == {0: 1, 1: 1} and rep.leaf_files == 1 and rep.decoded_bytes_total == 0
    assert not rep.texture_info and any("贴图信息" in w for w in rep.warnings)  # old bridge: said, not hidden
    assert files[1].depth == 1 and files[1].parents == ["Data/T/T.osgb"] and files[1].images == []
    json.dumps(rep.to_dict())


def test_scan_new_bridge_texture_fields(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path, ["Data/T/a.osgb", "Data/T/b.osgb", "Data/T/c.osgb"])
    img_ok = {"index": 0, "name": "t.jpg", "encoding": "jpg", "width": 64, "height": 32, "bytes": 900, "ok": True}
    recs = {
        "a.osgb": {"ok": True, "geometries": [_geom([0, 0, 0], [1, 1, 1], has_uv=True, image=0, texture_missing=False)], "images": [img_ok], "children": []},
        "b.osgb": {"ok": True, "geometries": [_geom([0, 0, 0], [1, 1, 1], has_uv=True, image=-1, texture_missing=True)], "images": [], "children": []},
        "c.osgb": {
            "ok": True,
            "geometries": [_geom([0, 0, 0], [1, 1, 1], has_uv=True, image=0, texture_missing=False)],
            "images": [{"index": 0, "name": "x.png", "encoding": "missing", "width": -1, "height": -1, "bytes": 0, "ok": False}, dict(img_ok, index=1)],
            "children": [],
        },
    }
    monkeypatch.setattr(osgb, "scan_osgb", lambda paths, batch=200: [dict(recs[Path(p).name], file=str(p)) for p in paths])
    ds = Dataset(root)
    files = {f.rel: f for f in ds.scan()}
    assert files["Data/T/a.osgb"].texture_issues == [] and files["Data/T/a.osgb"].decoded_bytes == 64 * 32 * 4
    assert files["Data/T/b.osgb"].texture_issues and files["Data/T/c.osgb"].texture_issues
    assert files["Data/T/c.osgb"].decoded_bytes == 64 * 32 * 4  # unknown size is not counted
    rep = ds.report
    assert not rep.ok_for_repair and any("贴图" in e for e in rep.blocking_errors)
    assert [t["rel"] for t in rep.texture_issues] == ["Data/T/b.osgb", "Data/T/c.osgb"]
    assert rep.decoded_bytes_total == 2 * 64 * 32 * 4 and rep.decoded_bytes_max == 64 * 32 * 4
    assert rep.texture_info and not any("贴图信息" in w for w in rep.warnings)


def test_scan_failed_and_missing_records(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path, ["Data/T/a.osgb", "Data/T/b.osgb", "Data/T/c.osgb"])

    def short(paths, batch=200):  # old bridge: ok=false without error, one record lost
        paths = [str(p) for p in paths]
        return [{"file": paths[0], "ok": False}, {"file": paths[2], "ok": True, "geometries": [_geom([0, 0, 0], [1, 1, 1])], "children": []}]

    monkeypatch.setattr(osgb, "scan_osgb", short)
    ds = Dataset(root)
    files = ds.scan()
    a, b, c = files
    assert not a.ok and a.error and not a.has_leaf and a.bmin is None and a.leaf_bmin is None
    assert not b.ok and "记录" in b.error and not b.has_leaf
    assert c.ok and c.has_leaf
    assert list(ds.tiles()) == ["T"] and [f.rel for f in ds.tiles()["T"]] == ["Data/T/c.osgb"]
    rep = ds.report
    assert rep.files_total == 3 and rep.files_ok == 1 and {f["rel"] for f in rep.failed} == {"Data/T/a.osgb", "Data/T/b.osgb"}
    assert not rep.ok_for_repair and "读取失败" in rep.blocking_errors[0]


def test_scan_matches_records_by_path_then_order(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path, ["a.osgb", "b.osgb"])

    def reordered(paths, batch=200):  # absolute-path spelling, reversed order
        recs = [{"file": str(Path(p).resolve()), "ok": Path(p).name == "a.osgb", "geometries": [_geom([0, 0, 0], [1, 1, 1])], "children": []} for p in paths]
        return recs[::-1]

    monkeypatch.setattr(osgb, "scan_osgb", reordered)
    (tmp_path / "x").mkdir()
    a, b = Dataset(tmp_path / "x" / ".." / "ds").scan()
    assert a.rel == "a.osgb" and a.ok and not b.ok


def test_scan_bridge_crash_isolates_bad_file(tmp_path, monkeypatch):
    root = _fake_dataset(tmp_path, ["a.osgb", "b.osgb", "c.osgb"])

    def crashy(paths, batch=200):
        paths = [str(p) for p in paths]
        if any(p.endswith("b.osgb") for p in paths):
            raise osgb.BridgeError("osgb_bridge info failed (-11): segfault")
        if any(p.endswith("c.osgb") for p in paths):
            raise json.JSONDecodeError("Expecting value", "plugin noise", 0)
        return [{"file": p, "ok": True, "geometries": [_geom([0, 0, 0], [1, 1, 1])], "children": []} for p in paths]

    monkeypatch.setattr(osgb, "scan_osgb", crashy)
    monkeypatch.setattr(osgb, "find_bridge", lambda: "/bin/true")
    ds = Dataset(root)
    a, b, c = ds.scan()
    assert a.ok and not b.ok and "segfault" in b.error and not c.ok and not c.has_leaf
    assert not ds.report.ok_for_repair

    def no_bridge():
        raise osgb.BridgeError("osgb_bridge not found")

    monkeypatch.setattr(osgb, "find_bridge", no_bridge)
    with pytest.raises(osgb.BridgeError):  # a missing bridge is an error, not 3 "bad files"
        Dataset(root).scan()


# ---------------------------------------------------------------------------
# OBJ / GLB
# ---------------------------------------------------------------------------


def _write_obj(path: Path, texture: str = "tex/a.png", faces: str = "f 1/1 2/2 3/3\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"mtllib {path.stem}.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl m\n{faces}", encoding="utf-8")
    (path.parent / f"{path.stem}.mtl").write_text(f"newmtl m\nmap_Kd {texture}\n", encoding="utf-8")


def test_obj_scan_failures_and_missing_textures(tmp_path):
    root = tmp_path / "objs"
    _write_obj(root / "good.obj")
    (root / "tex").mkdir()
    Image.fromarray(np.zeros((8, 16, 3), np.uint8)).save(root / "tex/a.png")
    _write_obj(root / "notex.obj", texture="tex/missing.png")
    _write_obj(root / "broken.obj", faces="f 1/1 2/2 99/3\n")
    (root / "._good.obj").write_bytes(b"\x00\x05\x16\x07")
    (root / "bad.glb").write_bytes(b"glTF garbage")
    ds = Dataset(root)
    assert ds.kind == "obj"
    files = {f.rel: f for f in ds.scan()}
    assert sorted(files) == ["broken.obj", "good.obj", "notex.obj"]
    assert files["good.obj"].ok and files["good.obj"].texture_issues == [] and files["good.obj"].decoded_bytes == 8 * 16 * 4
    assert files["notex.obj"].ok and "tex/missing.png" in files["notex.obj"].texture_issues[0]
    assert not files["broken.obj"].ok and not files["broken.obj"].has_leaf
    rep = ds.report
    assert rep.ignored == ["._good.obj"] and not rep.ok_for_repair
    assert rep.levels == {0: 3} and rep.roots == ["broken.obj", "good.obj", "notex.obj"]


def test_glb_scan_garbage(tmp_path):
    root = tmp_path / "glbs"
    root.mkdir()
    (root / "bad.glb").write_bytes(b"glTF\x02\x00\x00\x00garbage")
    ds = Dataset(root)
    (f,) = ds.scan()
    assert not f.ok and f.error and not ds.report.ok_for_repair


def test_single_file_copy_all(tmp_path):
    root = tmp_path / "one"
    _write_obj(root / "scene.obj")
    (root / "tex").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(root / "tex/a.png")
    (root / "._scene.obj").write_bytes(b"\x00\x05\x16\x07")
    (root / "notes.txt").write_text("n")
    before = inputs.file_manifest(root)
    ds = Dataset(root / "scene.obj")
    ds.scan()
    assert ds.report.ok_for_repair and ds.report.ignored == ["._scene.obj"]
    res = ds.copy_all(tmp_path / "out")
    assert sorted(f["rel"] for f in res["files"]) == ["notes.txt", "scene.mtl", "scene.obj", "tex/a.png"]
    assert not (tmp_path / "out" / "._scene.obj").exists()
    assert inputs.verify_manifest(before, root)["ok"] and (root / "._scene.obj").is_file()
    with pytest.raises(ValueError):
        Dataset(root / "._scene.obj")


# ---------------------------------------------------------------------------
# real OSGB files built with the bridge
# ---------------------------------------------------------------------------


@needs_bridge
def test_osgb_clean_subdir_refs_and_sidecars(tmp_path):
    root = tmp_path / "clean"
    info = clean_dataset(root, tmp_path / "work")
    add_sidecars(root)
    before = inputs.file_manifest(root)
    ds = Dataset(root)
    files = ds.scan()
    rels = [f.rel for f in files]
    assert rels == sorted(rels) and len(rels) == 8
    assert not any(inputs.is_ignored(r) for r in rels)
    rep = ds.report
    assert rep.ok_for_repair, rep.blocking_errors
    assert sorted(rep.ignored) == sorted(SIDECARS)
    assert rep.roots == ["Data/T/T.osgb", "Data/U/U.osgb"] and rep.levels == {0: 2, 1: 2, 2: 4}
    assert rep.missing_refs == [] and rep.cycles == [] and rep.multi_parent == [] and rep.unreachable == []
    by = {f.rel: f for f in files}
    assert by["Data/T/T.osgb"].children == ["sub/L1.osgb"]
    assert ds.graph.children("Data/U/sub/L1.osgb") == ["Data/U/sub/L2_0.osgb", "Data/U/sub/L2_1.osgb"]
    assert by["Data/U/sub/L2_1.osgb"].parents == ["Data/U/sub/L1.osgb"] and by["Data/U/sub/L2_1.osgb"].depth == 2
    assert sorted(f.rel for f in files if f.has_leaf) == info["leaves"] and rep.leaf_files == 4
    assert set(ds.tiles()) == {"T", "U"}
    rels_cut, depth = lod_cut(files, 1_000_000, graph=ds.graph)
    assert sorted(rels_cut) == info["leaves"] and depth == 2
    json.dumps(rep.to_dict())

    out = tmp_path / "out"
    res = ds.copy_all(out)
    copied = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert copied == sorted(rels + ["metadata.xml"]) == sorted(f["rel"] for f in res["files"])
    assert inputs.verify_manifest(before, root)["ok"]
    assert inputs.file_manifest(root) == before
    assert all((root / r).is_file() for r in SIDECARS)


@needs_bridge
def test_osgb_zip_import_never_lists_sidecars(tmp_path):
    src = tmp_path / "clean"
    clean_dataset(src, tmp_path / "work")
    add_sidecars(src)
    before = inputs.file_manifest(src)
    z = tmp_path / "upload.zip"
    zip_dataset(src, z, prefix="clean/")
    res = inputs.safe_extract_zip(z, tmp_path / "job" / "input")
    assert res["ignored"] and len(res["rejected"]) == 4
    ds = Dataset(tmp_path / "job" / "input" / "clean")
    ds.scan()
    assert ds.report.ok_for_repair and ds.report.ignored == []
    assert all(not inputs.is_ignored(f.rel) for f in ds.files) and len(ds.files) == 8
    assert inputs.verify_manifest(before, src)["ok"]


@needs_bridge
def test_osgb_missing_reference_blocks(tmp_path):
    missing_ref_dataset(tmp_path / "m", tmp_path / "work")
    ds = Dataset(tmp_path / "m")
    ds.scan()
    rep = ds.report
    assert rep.missing_refs == [{"parent": "Data/M/M.osgb", "ref": "M_gone.osgb", "resolved": "Data/M/M_gone.osgb"}]
    assert not rep.ok_for_repair and any("引用缺失" in e for e in rep.blocking_errors)
    assert rep.files_ok == 2


@needs_bridge
def test_osgb_multi_parent_dedup(tmp_path):
    multi_parent_dataset(tmp_path / "p", tmp_path / "work")
    ds = Dataset(tmp_path / "p")
    files = ds.scan()
    rep = ds.report
    assert rep.ok_for_repair, rep.blocking_errors
    assert rep.multi_parent == [{"rel": "Data/P/S.osgb", "parents": ["Data/P/A.osgb", "Data/P/B1.osgb"]}]
    by = {f.rel: f for f in files}
    assert by["Data/P/S.osgb"].depth == 2 and by["Data/P/B1.osgb"].depth == 2
    assert rep.levels == {0: 1, 1: 2, 2: 2}
    assert sum(rep.levels.values()) == len(files)  # every file counted once
    rels, _ = lod_cut(files, 1_000_000, graph=ds.graph)
    assert rels == ["Data/P/S.osgb"]
    assert ds.graph.subtree("Data/P/P.osgb").count("Data/P/S.osgb") == 1


@needs_bridge
def test_osgb_cycles_block_and_terminate(tmp_path):
    cycle_dataset(tmp_path / "c", tmp_path / "work")
    ds = Dataset(tmp_path / "c")
    files = ds.scan()
    rep = ds.report
    assert rep.cycles == [["Data/C/C.osgb", "Data/C/D.osgb"], ["Data/C/E.osgb"]]
    assert rep.unreachable == ["Data/C/C.osgb", "Data/C/D.osgb"]
    assert rep.roots == ["Data/C/E.osgb"]
    assert not rep.ok_for_repair and any("循环" in e for e in rep.blocking_errors)
    rels, _ = lod_cut(files, 1_000_000, graph=ds.graph)
    assert rels == ["Data/C/E1.osgb"]


@needs_bridge
def test_osgb_garbage_file_is_failed_not_leaf(tmp_path):
    garbage_dataset(tmp_path / "g", tmp_path / "work")
    ds = Dataset(tmp_path / "g")
    files = {f.rel: f for f in ds.scan()}
    for rel in ("Data/G/G_L1.osgb", "Data/G/junk.osgb"):
        f = files[rel]
        assert not f.ok and f.error and not f.has_leaf and f.bmin is None
    assert files["Data/G/G.osgb"].ok
    rep = ds.report
    assert not rep.ok_for_repair and rep.files_ok == 1 and len(rep.failed) == 2
    assert rep.leaf_files == 0
    assert [f.rel for f in ds.tiles()["G"]] == ["Data/G/G.osgb"]
    rels, _ = lod_cut(list(files.values()), 1_000_000, graph=ds.graph)
    assert rels == ["Data/G/G.osgb"]  # the parent of the broken child is kept


@needs_bridge
def test_osgb_case_mismatch_resolves_with_warning(tmp_path):
    case_mismatch_dataset(tmp_path / "k", tmp_path / "work")
    ds = Dataset(tmp_path / "k")
    files = ds.scan()
    rep = ds.report
    assert rep.ok_for_repair, rep.blocking_errors
    assert rep.missing_refs == [] and any("k_l1.osgb" in w for w in rep.warnings)
    assert {f.rel: f.depth for f in files} == {"Data/K/K.osgb": 0, "Data/K/K_L1.osgb": 1}


@needs_bridge
def test_osgb_bridge_records_one_per_file(tmp_path):
    """The real bridge (old or new) returns exactly one record per path, in order."""
    garbage_dataset(tmp_path / "g", tmp_path / "work")
    paths = sorted((tmp_path / "g").rglob("*.osgb"))
    recs = osgb.scan_osgb(paths)
    assert [r["file"] for r in recs] == [str(p) for p in paths]
    assert [r["ok"] for r in recs] == [True, False, False]
