"""Runtime compatibility of the native bridge: version / selftest, the info
contract, broken inputs, timeouts / cancellation and environment checks."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import needs_bridge
from sensitive3d.core.mesh import Texture
from sensitive3d.io import osgb

GEODE = (
    "BEGIN_PAGEDLOD 0.5 0.5 0 1 1\n"
    "CHILD 0 100 BEGIN_GEODE\n"
    "  GEOMETRY v.f32 uv.f32 - t.u32 tex.jpg\n"
    "  GEOMETRY v.f32 - - t.u32 tex2.png\n"
    "  GEOMETRY v.f32 uv.f32 - t.u32 -\n"
    "END_GEODE\n"
    "FILE_CHILD fine.osgb 100 1e30\n"
    "END_PAGEDLOD\n"
)
CHECKS = {
    "jpeg_write", "jpeg_read", "png_write", "png_read", "osgb_write", "osgb_read", "pixels_match", "geometry_match",
}


def _inputs(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32).tofile(d / "v.f32")
    np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32).tofile(d / "uv.f32")
    np.array([0, 1, 2, 0, 2, 3], np.uint32).tofile(d / "t.u32")
    img = np.zeros((32, 64, 3), np.uint8)
    img[:16, :, 0] = 255
    Image.fromarray(img).save(d / "tex.jpg", quality=90)
    Image.fromarray(img[:, :48]).save(d / "tex2.png")


@pytest.fixture
def built(tmp_path) -> dict:
    """a.osgb with embedded JPEG + PNG textures, ext/e.osgb referencing them as external files."""
    _inputs(tmp_path)
    osgb.build_osgb(GEODE, tmp_path, tmp_path / "a.osgb")
    ext = tmp_path / "ext"
    _inputs(ext)
    osgb.build_osgb("EXTERNAL_IMAGES\n" + GEODE, ext, ext / "e.osgb")
    return {"dir": tmp_path, "a": tmp_path / "a.osgb", "e": ext / "e.osgb", "ext": ext}


def _bridge_export(src: Path, out: Path) -> dict:
    p = subprocess.run([osgb.find_bridge(), "export", str(src), str(out)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    return json.loads((out / "manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# version / selftest
# ---------------------------------------------------------------------------


@needs_bridge
def test_version_json():
    p = subprocess.run([osgb.find_bridge(), "version"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    raw = json.loads(p.stdout)
    v = osgb.bridge_version(force=True)
    assert v["bridge_version"] == raw["bridge_version"] == "2"
    assert v["osg_version"].count(".") >= 1
    assert isinstance(v["plugin_paths"], list) and v["plugin_paths"]
    assert {"osgb", "jpg", "png"} <= set(v["plugins"])
    assert v["plugins"]["osgb"]["path"], "the .osgb reader plugin must be resolvable"


@needs_bridge
def test_selftest_passes(tmp_path):
    p = subprocess.run(
        [osgb.find_bridge(), "selftest", str(tmp_path / "st")], capture_output=True, text=True, timeout=120
    )
    data = json.loads(p.stdout)
    assert p.returncode == 0 and data["ok"], data["errors"]
    assert CHECKS <= set(data["checks"]) and all(data["checks"].values())
    assert data["errors"] == [] and data["osg_version"] and isinstance(data["plugin_paths"], list)

    st = osgb.bridge_selftest(force=True)
    assert st["ok"] and st["hint"] is None
    status = osgb.bridge_status()
    assert status["found"] and status["ok"] and status["error"] is None
    assert status["selftest"]["checks"] == st["checks"]
    assert status["version"]["bridge_version"] == "2"


# ---------------------------------------------------------------------------
# info contract
# ---------------------------------------------------------------------------


@needs_bridge
def test_info_contract(built):
    rec = osgb.scan_osgb([built["a"]])[0]
    assert rec["file"] == str(built["a"]) and rec["ok"] and "error" not in rec
    assert rec["children"] == ["fine.osgb"]
    g0, g1, g2 = rec["geometries"]
    for g in (g0, g1, g2):
        assert g["has_finer"] and g["num_triangles"] == 2 and g["depth"] == 1
        assert g["min"] == [0, 0, 0] and g["max"] == [1, 1, 0]
    assert g0["has_uv"] and g0["image"] == 0 and g0["texture_missing"] is False
    assert not g1["has_uv"] and g1["image"] == 1 and g1["texture_missing"] is False
    assert g2["has_uv"] and g2["image"] == -1 and g2["texture_missing"] is False
    im0, im1 = rec["images"]
    jpg = (built["dir"] / "tex.jpg").read_bytes()
    png = (built["dir"] / "tex2.png").read_bytes()
    assert im0 == {
        "index": 0, "name": "tex.jpg", "encoding": "jpg", "format": "jpg", "external": False,
        "width": 64, "height": 32, "bytes": len(jpg), "ok": True,
    }
    assert (im1["encoding"], im1["width"], im1["height"], im1["bytes"], im1["ok"]) == ("png", 48, 32, len(png), True)

    ext = osgb.scan_osgb([built["e"]])[0]
    assert ext["ok"] and [im["encoding"] for im in ext["images"]] == ["external", "external"]
    assert [(im["format"], im["width"], im["external"], im["ok"]) for im in ext["images"]] == [
        ("jpg", 64, True, True), ("png", 48, True, True),
    ]


@needs_bridge
def test_info_bad_files_do_not_lose_the_batch(built, tmp_path):
    garbage = tmp_path / "garbage.osgb"
    garbage.write_bytes(np.random.default_rng(1).integers(0, 256, 5000, dtype=np.uint8).tobytes())
    trunc = tmp_path / "trunc.osgb"
    data = built["a"].read_bytes()
    trunc.write_bytes(data[: len(data) // 2])
    missing = tmp_path / "missing.osgb"
    paths = [built["a"], garbage, missing, trunc, built["e"]]
    expect = [True, False, False, False, True]

    # the bridge itself prints exactly one record per input line, in order
    lst = tmp_path / "list.txt"
    lst.write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
    p = subprocess.run([osgb.find_bridge(), "info", str(lst)], capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr
    recs = [json.loads(line) for line in p.stdout.splitlines()]
    assert [r["file"] for r in recs] == [str(x) for x in paths]
    assert [r["ok"] for r in recs] == expect
    for r in recs:
        if not r["ok"]:
            assert r["error"] and r["geometries"] == [] and r["images"] == [] and r["children"] == []
    assert "not found" in recs[2]["error"]

    scanned = osgb.scan_osgb(paths)
    assert [r["file"] for r in scanned] == [str(x) for x in paths]
    assert [r["ok"] for r in scanned] == expect


@needs_bridge
def test_missing_external_texture(built, tmp_path):
    e = built["e"]
    (built["ext"] / "tex.jpg").unlink()
    rec = osgb.scan_osgb([e])[0]
    assert rec["ok"]
    g0, g1, _ = rec["geometries"]
    assert g0["texture_missing"] is True and g0["image"] == 0
    assert g1["texture_missing"] is False and g1["image"] == 1
    im0, im1 = rec["images"]
    assert im0["encoding"] == "missing" and im0["ok"] is False and im0["external"] is True
    assert "tex.jpg" in im0["error"] and "not found" in im0["error"]
    assert im1["ok"] is True

    # export: no broken raw for the missing image, the other one is intact
    out = tmp_path / "exp"
    man = _bridge_export(e, out)
    t0, t1 = man["textures"]
    assert t0["file"] is None and t0["valid"] is False and "not found" in t0["error"] and t0["name"] == "tex.jpg"
    assert not (out / "tex0.raw").exists()
    assert t1["valid"] is True and (out / t1["file"]).stat().st_size == 48 * 32 * t1["channels"]
    assert man["geometries"][0]["texture_missing"] is True and man["geometries"][0]["texture"] == 0

    with pytest.raises(osgb.BridgeTextureError) as ei:
        osgb.read_osgb(e)
    assert ei.value.texture == "tex.jpg" and str(e) in str(ei.value)

    # patching must refuse to write a file that would lose the texture
    patch = tmp_path / "patch"
    patch.mkdir()
    (patch / "patch.txt").write_text("\n", encoding="utf-8")
    p = subprocess.run(
        [osgb.find_bridge(), "patch", str(e), str(patch), str(tmp_path / "out.osgb")], capture_output=True, text=True
    )
    assert p.returncode != 0 and "tex.jpg" in p.stderr and not (tmp_path / "out.osgb").exists()


@needs_bridge
def test_corrupt_embedded_texture(built, tmp_path):
    data = bytearray(built["a"].read_bytes())
    jpg = (built["dir"] / "tex.jpg").read_bytes()
    i = bytes(data).index(jpg)  # embedded byte for byte
    data[i : i + 2] = b"\0\0"  # destroy the JPEG SOI marker
    bad = tmp_path / "bad.osgb"
    bad.write_bytes(bytes(data))

    rec = osgb.scan_osgb([bad])[0]
    assert rec["ok"] and rec["geometries"][0]["texture_missing"] is True
    assert rec["images"][0]["ok"] is False and "not a valid jpg" in rec["images"][0]["error"]
    assert rec["images"][1]["ok"] is True

    out = tmp_path / "exp"
    man = _bridge_export(bad, out)
    assert man["textures"][0]["valid"] is False and "cannot decode" in man["textures"][0]["error"]
    assert not (out / "tex0.raw").exists()
    with pytest.raises(osgb.BridgeTextureError):
        osgb.read_osgb(bad)


# ---------------------------------------------------------------------------
# process control (fake bridge executables; no OpenSceneGraph needed)
# ---------------------------------------------------------------------------


def _fake_bridge(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "fake_bridge"
    p.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    p.chmod(0o755)
    return p


def _gone(pid: int, wait: float = 5.0) -> bool:
    """True once ``pid`` no longer runs (exited, or a zombie nobody reaps yet)."""
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists():
            try:
                if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                    return True
            except (OSError, IndexError):
                return True
        else:
            out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
            if not out.stdout.strip() or out.stdout.strip().startswith("Z"):
                return True
        time.sleep(0.05)
    return False


SLEEPER = """
import os, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open(sys.argv[-1] + ".tmp", "w") as f:
    f.write(str(child.pid))
os.replace(sys.argv[-1] + ".tmp", sys.argv[-1] + ".pid")  # appears atomically
print("started", file=sys.stderr, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_run_timeout_kills_process_group(tmp_path, monkeypatch):
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(_fake_bridge(tmp_path, SLEEPER)))
    marker = tmp_path / "job"
    t0 = time.monotonic()
    with pytest.raises(osgb.BridgeTimeout) as ei:
        osgb._run(["info", str(marker)], timeout=3)
    assert time.monotonic() - t0 < 10
    msg = str(ei.value)
    assert "timed out" in msg and "fake_bridge" in msg and "started" in msg  # command + stderr tail
    assert isinstance(ei.value, osgb.BridgeError)
    assert _gone(int((tmp_path / "job.pid").read_text())), "child of the bridge survived the timeout"


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_run_timeout_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(_fake_bridge(tmp_path, SLEEPER)))
    monkeypatch.setenv("S3D_BRIDGE_TIMEOUT", "1")
    with pytest.raises(osgb.BridgeTimeout):
        osgb._run(["info", str(tmp_path / "job")])
    assert _gone(int((tmp_path / "job.pid").read_text()))


def test_default_timeout_parsing(monkeypatch):
    monkeypatch.delenv("S3D_BRIDGE_TIMEOUT", raising=False)
    assert osgb.default_timeout() == osgb.DEFAULT_TIMEOUT == 600
    monkeypatch.setenv("S3D_BRIDGE_TIMEOUT", "42.5")
    assert osgb.default_timeout() == 42.5
    monkeypatch.setenv("S3D_BRIDGE_TIMEOUT", "0")
    assert osgb.default_timeout() is None
    monkeypatch.setenv("S3D_BRIDGE_TIMEOUT", "bogus")
    assert osgb.default_timeout() == 600


@pytest.mark.skipif(os.name != "posix", reason="process groups")
def test_run_cancel_event_and_callable(tmp_path, monkeypatch):
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(_fake_bridge(tmp_path, SLEEPER)))
    def set_when_started(ev, pidfile):
        end = time.monotonic() + 20
        while not pidfile.exists() and time.monotonic() < end:
            time.sleep(0.02)
        ev.set()

    ev = threading.Event()
    threading.Thread(target=set_when_started, args=(ev, tmp_path / "a.pid"), daemon=True).start()
    t0 = time.monotonic()
    with pytest.raises(osgb.BridgeCancelled) as ei:
        osgb._run(["info", str(tmp_path / "a")], timeout=60, cancel=ev)
    assert time.monotonic() - t0 < 25 and "cancelled" in str(ei.value)
    assert _gone(int((tmp_path / "a.pid").read_text()))

    with pytest.raises(osgb.BridgeCancelled):
        osgb._run(["info", str(tmp_path / "b")], timeout=60, cancel=lambda: (tmp_path / "b.pid").exists())
    assert _gone(int((tmp_path / "b.pid").read_text()))

    # already cancelled: the process is never started
    with pytest.raises(osgb.BridgeCancelled):
        osgb._run(["info", str(tmp_path / "c")], cancel=lambda: True)
    assert not (tmp_path / "c.pid").exists()
    with pytest.raises(TypeError):
        osgb._run(["info", "x"], cancel=object())


def test_run_error_message_and_hints(tmp_path, monkeypatch):
    fake = _fake_bridge(
        tmp_path,
        "import sys\n"
        "for i in range(40):\n"
        "    print(f'noise line {i}', file=sys.stderr)\n"
        "print('cannot read /data/x.osgb', file=sys.stderr)\n"
        "sys.exit(3)\n",
    )
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(fake))
    with pytest.raises(osgb.BridgeError) as ei:
        osgb._run(["export", "/data/x.osgb", "/tmp/out"])
    msg = str(ei.value)
    assert "exit code 3" in msg and "export /data/x.osgb /tmp/out" in msg and str(fake) in msg
    assert "cannot read /data/x.osgb" in msg and "noise line 39" in msg and "noise line 0\n" not in msg
    assert ei.value.returncode == 3 and "cannot read" in ei.value.stderr

    fake.write_text(
        f"#!{sys.executable}\n{NO_CORE}import os, sys\n"
        "print('dyld[123]: Library not loaded: @rpath/libosgDB.161.dylib', file=sys.stderr, flush=True)\n"
        "os.abort()\n",
        encoding="utf-8",
    )
    with pytest.raises(osgb.BridgeError) as ei:
        osgb._run(["version"])
    assert "SIGABRT" in str(ei.value) and "DYLD_LIBRARY_PATH" in ei.value.hint and "docs/macos.md" in str(ei.value)

    fake.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "print('osgb_bridge: error while loading shared libraries: libosgDB.so.161: cannot open shared object file',"
        " file=sys.stderr)\nsys.exit(127)\n",
        encoding="utf-8",
    )
    with pytest.raises(osgb.BridgeError) as ei:
        osgb._run(["version"])
    assert "LD_LIBRARY_PATH" in ei.value.hint

    # not executable
    fake.chmod(0o644)
    with pytest.raises(osgb.BridgeError) as ei:
        osgb._run(["version"])
    assert "cannot start" in str(ei.value) and ei.value.hint


NO_CORE = "import resource\nresource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"

FAKE_INFO = NO_CORE + r"""
import json, os, signal, sys, time
log, cmd, lst = sys.argv[0] + ".log", sys.argv[1], sys.argv[2]
paths = open(lst, encoding="utf-8").read().splitlines()
with open(log, "a") as f:
    f.write(json.dumps(paths) + "\n")
for i, p in enumerate(paths):
    name = os.path.basename(p)
    if name == "crash.osgb":
        os.kill(os.getpid(), signal.SIGSEGV)
    if name == "hang.osgb":
        time.sleep(60)
    if name == "short.osgb" and len(paths) > 1:
        sys.exit(0)  # silently stops: record count mismatch
    if name == "half.osgb":
        sys.stdout.write('{"file": "' + p + '", "ok": tr')
        sys.stdout.flush()
        os.kill(os.getpid(), signal.SIGSEGV)
    print(json.dumps({"file": p, "ok": True, "geometries": [], "images": [], "children": []}), flush=True)
"""


@pytest.mark.skipif(os.name != "posix", reason="signals")
def test_scan_recovers_from_crash_and_mismatch(tmp_path, monkeypatch):
    fake = _fake_bridge(tmp_path, FAKE_INFO)
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(fake))
    names = ["a", "b", "crash", "c", "short", "d", "half", "e"]
    paths = [str(tmp_path / f"{n}.osgb") for n in names]
    recs = osgb.scan_osgb(paths)
    assert [r["file"] for r in recs] == paths
    ok = {os.path.basename(r["file"]): r["ok"] for r in recs}
    assert ok == {"a.osgb": True, "b.osgb": True, "crash.osgb": False, "c.osgb": True, "short.osgb": True,
                  "d.osgb": True, "half.osgb": False, "e.osgb": True}
    by = {os.path.basename(r["file"]): r for r in recs}
    assert "SIGSEGV" in by["crash.osgb"]["error"] and "SIGSEGV" in by["half.osgb"]["error"]
    for r in recs:
        assert set(r) >= {"file", "ok", "geometries", "images", "children"}
    calls = [json.loads(line) for line in Path(str(fake) + ".log").read_text().splitlines()]
    assert calls[0] == paths  # one batch, then the files after the crash one by one
    assert all(len(c) == 1 for c in calls[1:]) and [c[0] for c in calls[1:]] == paths[2:]

    # batching: every chunk is complete as well
    Path(str(fake) + ".log").unlink()
    recs = osgb.scan_osgb(paths, batch=3)
    assert [r["file"] for r in recs] == paths and [r["ok"] for r in recs] == list(ok.values())


@pytest.mark.skipif(os.name != "posix", reason="signals")
def test_scan_isolates_a_hanging_file(tmp_path, monkeypatch):
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(_fake_bridge(tmp_path, FAKE_INFO)))
    paths = [str(tmp_path / f"{n}.osgb") for n in ("a", "hang", "b")]
    t0 = time.monotonic()
    recs = osgb.scan_osgb(paths, timeout=1.0)
    assert time.monotonic() - t0 < 20
    assert [r["ok"] for r in recs] == [True, False, True]
    assert "timed out" in recs[1]["error"]

    ev = threading.Event()
    ev.set()
    with pytest.raises(osgb.BridgeCancelled):
        osgb.scan_osgb(paths, cancel=ev)


def test_scan_rejects_unsupported_names(tmp_path, monkeypatch):
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(_fake_bridge(tmp_path, FAKE_INFO)))
    paths = [str(tmp_path / "a.osgb"), str(tmp_path / "bad\nname.osgb"), str(tmp_path / "b.osgb")]
    recs = osgb.scan_osgb(paths)
    assert [r["file"] for r in recs] == paths and [r["ok"] for r in recs] == [True, False, True]
    assert "line break" in recs[1]["error"]


# ---------------------------------------------------------------------------
# status / doctor
# ---------------------------------------------------------------------------


def test_bridge_status_missing_binary(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "osgb_bridge"
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(missing))
    assert not osgb.bridge_available()
    st = osgb.bridge_status()
    assert st["found"] is False and st["ok"] is False and st["path"] is None and st["selftest"] is None
    assert str(missing) in st["error"] and "S3D_OSGB_BRIDGE" in st["hint"]
    sel = osgb.bridge_selftest()
    assert sel["ok"] is False and sel["hint"]
    with pytest.raises(osgb.BridgeError):
        osgb.read_osgb(tmp_path / "x.osgb")


def test_bridge_status_old_binary(tmp_path, monkeypatch):
    old = _fake_bridge(
        tmp_path,
        "import sys\nprint('usage:\\n  osgb_bridge export <in.osgb> <out_dir>', file=sys.stderr)\nsys.exit(1)\n",
    )
    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(old))
    st = osgb.bridge_status(force=True)
    assert st["found"] is True and st["ok"] is False and "build_bridge.sh" in st["hint"]


def test_doctor_reports_missing_bridge(tmp_path, monkeypatch, capsys):
    from sensitive3d.cli import main
    from sensitive3d.doctor import environment_report, format_report

    monkeypatch.setenv("S3D_OSGB_BRIDGE", str(tmp_path / "missing"))
    report = environment_report()
    assert report["ok"] is False and report["bridge"]["found"] is False
    assert any("osgb_bridge" in p for p in report["problems"]) and report["hints"]
    assert report["env"]["S3D_OSGB_BRIDGE"] == str(tmp_path / "missing")
    assert set(report["packages"]) >= {"numpy", "numba", "scipy", "cv2", "PIL", "trimesh", "skimage",
                                       "fast_simplification", "mapbox_earcut", "fastapi", "uvicorn", "onnxruntime"}
    assert report["python"]["version"] and report["platform"]["machine"]
    assert "存在问题" in format_report(report)

    assert main(["doctor", "--json"]) == 1
    printed = json.loads(capsys.readouterr().out)
    assert printed["ok"] is False and printed["bridge"]["error"]


@needs_bridge
def test_doctor_ok_with_working_bridge(capsys):
    from sensitive3d.cli import main

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "自检        通过" in out and "环境正常" in out


def test_encode_texture_keeps_opaque_jpeg():
    rgba = np.full((8, 8, 4), 200, np.uint8)
    rgba[..., 3] = 255
    assert osgb.encode_texture(Texture(rgba.copy(), encoding="jpg"))[1] == "jpg"
    rgba[0, 0, 3] = 10
    assert osgb.encode_texture(Texture(rgba.copy(), encoding="jpg"))[1] == "png"
    assert osgb.encode_texture(Texture(np.zeros((4, 4, 3), np.uint8), encoding="png"))[1] == "png"
