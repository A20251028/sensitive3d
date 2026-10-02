import io
import time
import zipfile

from fastapi.testclient import TestClient

from sensitive3d.server import _safe_rel, create_app


def test_safe_paths():
    assert _safe_rel("a/b.osgb").as_posix() == "a/b.osgb"
    assert _safe_rel("../x") is None
    assert _safe_rel("/etc/passwd") is None
    assert _safe_rel("C:\\x\\y") is None


def test_upload_process_download(synthetic, tmp_path):
    app = create_app(tmp_path / "ws")
    client = TestClient(app)
    assert client.get("/api/health").json()["ok"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in (synthetic / "obj").iterdir():
            if p.name != "ground_truth.json":
                zf.write(p, f"street/{p.name}")
    r = client.post("/api/jobs", files=[("files", ("street.zip", buf.getvalue(), "application/zip"))], data={"options": '{"previews": true}'})
    assert r.status_code == 200, r.text
    job_id = r.json()["id"]
    for _ in range(300):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(1)
    assert job["status"] == "done", job.get("error")
    assert len(job["report"]["regions"]) == 5
    assert client.get(f"/api/jobs/{job_id}/files/{job['report']['overview']['after']}").status_code == 200
    assert client.get(f"/api/jobs/{job_id}/files/../job.json").status_code in (400, 404)
    z = client.get(f"/api/jobs/{job_id}/download")
    assert z.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
    assert "scene.obj" in names and "sensitive3d_report.json" in names
