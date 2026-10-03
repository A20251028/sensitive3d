import io
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from sensitive3d.io.inputs import file_manifest
from sensitive3d.server import _safe_rel, create_app


def test_safe_paths():
    assert _safe_rel("a/b.osgb").as_posix() == "a/b.osgb"
    assert _safe_rel("../x") is None
    assert _safe_rel("/etc/passwd") is None
    assert _safe_rel("C:\\x\\y") is None


def _wait(client, job_id, pred, timeout=300):
    job = None
    for _ in range(timeout * 2):
        job = client.get(f"/api/jobs/{job_id}").json()
        if pred(job):
            return job
        time.sleep(0.5)
    raise AssertionError(f"timeout, last state: { {k: job[k] for k in ('status', 'steps', 'message')} }")


def _idle(job):
    return job["status"] not in ("running", "queued")


def _zip_obj(synthetic, extra_junk=True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in (synthetic / "obj").iterdir():
            if p.name not in ("ground_truth.json", "synthetic_meta.json"):
                zf.write(p, f"street/{p.name}")
        if extra_junk:
            zf.writestr("__MACOSX/street/._scene.obj", b"junk")
            zf.writestr("street/._scene.obj", b"junk")
            zf.writestr("street/.DS_Store", b"junk")
    return buf.getvalue()


def test_stepwise_flow(synthetic, tmp_path):
    client = TestClient(create_app(tmp_path / "ws"))
    r = client.post("/api/jobs", files=[("files", ("street.zip", _zip_obj(synthetic), "application/zip"))], data={"options": "{}"})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    job = _wait(client, job_id, lambda j: j["steps"]["preview"]["status"] in ("done", "failed") and _idle(j))
    assert job["steps"]["import"]["status"] == "done" and job["steps"]["preview"]["status"] == "done", job["steps"]
    # nothing is detected or repaired without being asked
    assert job["steps"]["detect"]["status"] == "pending" and job["steps"]["repair"]["status"] == "pending"
    scan = client.get(f"/api/jobs/{job_id}/scan").json()
    assert scan["ok_for_repair"] and scan["files_ok"] == scan["files"] == 1
    assert len(scan["ignored"]) == 0  # junk never reached the working copy (dropped at import)
    assert client.get(f"/api/jobs/{job_id}/files/{job['overview']['before']['path']}").status_code == 200
    # repair without review is refused
    assert client.post(f"/api/jobs/{job_id}/repair").status_code == 409

    assert client.post(f"/api/jobs/{job_id}/detect", json={"options": {}}).status_code == 200
    job = _wait(client, job_id, lambda j: j["steps"]["detect"]["status"] not in ("queued", "running"))
    assert job["steps"]["detect"]["status"] == "done", job["steps"]["detect"]
    det = client.get(f"/api/jobs/{job_id}/detection").json()
    assert len(det["candidates"]) == 5
    c0 = det["candidates"][0]
    assert client.get(f"/api/jobs/{job_id}/files/{c0['evidence']['overlay']}").status_code == 200

    # invalid decision is rejected with an explanation
    bad = client.put(f"/api/jobs/{job_id}/review", json={"decisions": [{"id": 99, "accept": True}]})
    assert bad.status_code == 400
    decisions = [{"id": c["id"], "accept": c["id"] in (0, 1), "operation": "texture" if c["id"] == 1 else "geometry"} for c in det["candidates"]]
    ok = client.put(f"/api/jobs/{job_id}/review", json={"decisions": decisions, "protected": []})
    assert ok.status_code == 200, ok.text
    assert len(ok.json()["targets"]) == 2

    assert client.post(f"/api/jobs/{job_id}/repair").status_code == 200
    job = _wait(client, job_id, lambda j: j["steps"]["package"]["status"] in ("done", "failed") or j["steps"]["repair"]["status"] in ("failed", "stopped", "cancelled"))
    assert job["steps"]["package"]["status"] == "done", job["steps"]
    rep = client.get(f"/api/jobs/{job_id}/repair").json()
    assert {t["id"] for t in rep["targets"]} == {0, 1}
    assert all(f["readback"]["ok"] for f in rep["files"])
    z = client.get(f"/api/jobs/{job_id}/download")
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "scene.obj" in names and "sensitive3d_report/repair.json" in names and "sensitive3d_report/detection.json" in names
    assert not any("._" in n or "DS_Store" in n or "__MACOSX" in n for n in names)


def test_auto_mode_and_cancel(synthetic, tmp_path):
    client = TestClient(create_app(tmp_path / "ws"))
    r = client.post("/api/jobs", files=[("files", ("street.zip", _zip_obj(synthetic, False), "application/zip"))], data={"options": '{"auto": true}'})
    job_id = r.json()["id"]
    job = _wait(client, job_id, lambda j: j["steps"]["package"]["status"] in ("done", "failed"))
    assert job["steps"]["package"]["status"] == "done"
    assert job["repair"]["targets"] == 5

    # cancel a running detection: the step ends as cancelled and nothing is repaired
    assert client.post(f"/api/jobs/{job_id}/detect", json={}).status_code == 200
    _wait(client, job_id, lambda j: j["steps"]["detect"]["status"] == "running", timeout=60)
    client.post(f"/api/jobs/{job_id}/cancel")
    job = _wait(client, job_id, _idle, timeout=120)
    assert job["steps"]["detect"]["status"] in ("cancelled", "done")
    if job["steps"]["detect"]["status"] == "cancelled":
        assert job["steps"]["repair"]["status"] == "pending"


def test_local_import_rules(synthetic, tmp_path):
    src_root = tmp_path / "data"
    src = src_root / "street"
    src.mkdir(parents=True)
    for p in (synthetic / "obj").iterdir():
        (src / p.name).write_bytes(p.read_bytes())
    (src / "._scene.obj").write_bytes(b"appledouble")
    before = file_manifest(src)
    # not enabled
    client = TestClient(create_app(tmp_path / "ws0"))
    assert client.post("/api/jobs/local", json={"path": str(src)}).status_code == 403
    # workspace inside an allowed root is refused at start-up
    with pytest.raises(ValueError):
        create_app(src_root / "ws", allow_local_roots=[src_root])
    client = TestClient(create_app(tmp_path / "ws1", allow_local_roots=[src_root]))
    assert client.post("/api/jobs/local", json={"path": str(tmp_path)}).status_code == 403  # outside the root
    assert client.post("/api/jobs/local", json={"path": str(src_root / "missing")}).status_code == 400
    r = client.post("/api/jobs/local", json={"path": str(src), "name": "street"})
    assert r.status_code == 200, r.text
    job = _wait(client, r.json()["id"], lambda j: j["steps"]["preview"]["status"] in ("done", "failed") and _idle(j))
    assert job["steps"]["import"]["status"] == "done", job["steps"]
    assert job["source"]["unchanged"] is True
    after = file_manifest(src)
    assert after["files"] == before["files"] and (src / "._scene.obj").is_file()  # source untouched, junk not deleted
    assert not any(p.name.startswith("._") for p in (tmp_path / "ws1").rglob("*"))  # junk not copied
