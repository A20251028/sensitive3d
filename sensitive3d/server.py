"""Web service: upload a dataset, run the pipeline as a background job, preview, download.

Endpoints (JSON unless noted)::

    GET  /api/health                     bridge availability, version
    POST /api/jobs                       multipart: files[] (zip, or folder files with
                                         relative paths as file names) + options (JSON)
    POST /api/demo                       job on the built-in synthetic dataset
    GET  /api/jobs                       list jobs
    GET  /api/jobs/{id}                  status, progress, log, report
    GET  /api/jobs/{id}/files/{path}     previews (png / glb) of a job
    GET  /api/jobs/{id}/download         zip of the repaired dataset
    DELETE /api/jobs/{id}                remove a job and its files
"""

from __future__ import annotations

import json
import queue
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_LOG = 400


def _safe_rel(name: str) -> Optional[PurePosixPath]:
    """Relative upload path without traversal; None when unusable."""
    p = PurePosixPath(name.replace("\\", "/"))
    parts = [x for x in p.parts if x not in ("", ".")]
    if not parts or any(x == ".." for x in parts) or p.is_absolute() or ":" in parts[0]:
        return None
    return PurePosixPath(*parts)


def _strip_common_root(root: Path) -> Path:
    """Descend into single wrapper folders (e.g. zip of a folder) until the dataset root."""
    cur = root
    for _ in range(5):
        entries = [p for p in cur.iterdir() if not p.name.startswith((".", "__MACOSX"))]
        if len(entries) == 1 and entries[0].is_dir() and not (cur / "Data").is_dir() and not (cur / "metadata.xml").is_file():
            cur = entries[0]
        else:
            break
    return cur


class Job:
    def __init__(self, job_id: str, root: Path, options: dict, name: str):
        self.id = job_id
        self.root = root
        self.options = options
        self.name = name
        self.status = "queued"
        self.progress = 0.0
        self.message = "排队中"
        self.log: list[str] = []
        self.report: Optional[dict] = None
        self.error: Optional[str] = None
        self.created = time.time()
        self.finished: Optional[float] = None
        self.lock = threading.Lock()

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    def add_log(self, msg: str) -> None:
        with self.lock:
            self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")
            del self.log[:-MAX_LOG]

    def to_dict(self, full: bool = True) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created": self.created,
            "finished": self.finished,
            "error": self.error,
            "options": self.options,
        }
        if full:
            with self.lock:
                d["log"] = list(self.log[-200:])
            d["report"] = self.report
        return d

    def save(self) -> None:
        (self.root / "job.json").write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")


class JobManager:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        (workspace / "jobs").mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[str]" = queue.Queue()
        self._load_existing()
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_existing(self) -> None:
        for d in sorted((self.workspace / "jobs").iterdir()):
            f = d / "job.json"
            if not f.is_file():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            job = Job(data["id"], d, data.get("options", {}), data.get("name", ""))
            job.status = data.get("status", "failed")
            if job.status in ("queued", "running"):
                job.status, job.error = "failed", "服务重启, 任务中断"
            job.progress = data.get("progress", 0)
            job.message = data.get("message", "")
            job.report = data.get("report")
            job.error = data.get("error") or job.error
            job.created = data.get("created", time.time())
            job.finished = data.get("finished")
            job.log = data.get("log", [])
            self.jobs[job.id] = job

    def create(self, options: dict, name: str) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        root = self.workspace / "jobs" / job_id
        (root / "input").mkdir(parents=True)
        job = Job(job_id, root, options, name)
        self.jobs[job_id] = job
        return job

    def submit(self, job: Job) -> None:
        job.save()
        self.queue.put(job.id)

    def _worker(self) -> None:
        while True:
            job_id = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            self._run(job)

    def _run(self, job: Job) -> None:
        from .pipeline import PipelineConfig, run
        from .webpreview import write_overviews

        job.status = "running"
        job.message = "开始处理"
        job.save()

        def progress(frac: float, msg: str) -> None:
            job.progress = 0.95 * frac
            job.message = msg

        try:
            src = _strip_common_root(job.input_dir)
            cfg = PipelineConfig.from_dict(job.options)
            job.report = run(src, job.output_dir, cfg, progress=progress, log=job.add_log, work_dir=job.work_dir)
            job.message = "生成三维预览"
            job.progress = 0.96
            try:
                job.report["overview"] = write_overviews(src, job.output_dir, job.work_dir / "previews", job.report)
            except Exception as e:  # previews are optional
                job.add_log(f"预览生成失败: {e}")
            job.message = "打包结果"
            job.progress = 0.98
            archive = job.root / "result.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in sorted(job.output_dir.rglob("*")):
                    if p.is_file():
                        zf.write(p, p.relative_to(job.output_dir).as_posix())
                rep = job.work_dir / "report.json"
                if rep.is_file():
                    zf.write(rep, "sensitive3d_report.json")
            job.status = "done"
            job.progress = 1.0
            job.message = job.report.get("summary") and f"完成: 去除 {len(job.report['regions'])} 个敏感标志" or "完成: 未发现敏感标志"
        except Exception as e:
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
            job.message = "处理失败"
            job.add_log(traceback.format_exc())
        finally:
            job.finished = time.time()
            job.save()

    def delete(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        if job is not None:
            shutil.rmtree(job.root, ignore_errors=True)


def create_app(workspace: str | Path = "workspace") -> FastAPI:
    workspace = Path(workspace).resolve()
    manager = JobManager(workspace)
    app = FastAPI(title="sensitive3d", version=__version__)
    app.state.manager = manager

    @app.get("/api/health")
    def health():
        from .io.osgb import bridge_available

        return {"ok": True, "version": __version__, "osgb_bridge": bridge_available()}

    @app.post("/api/jobs")
    async def create_job(files: list[UploadFile] = File(...), options: str = Form("{}"), name: str = Form("")):
        try:
            opts = json.loads(options or "{}")
        except json.JSONDecodeError:
            raise HTTPException(400, "options must be JSON")
        job = manager.create(opts, name)
        count = 0
        try:
            for up in files:
                rel = _safe_rel(up.filename or "")
                if rel is None:
                    continue
                if rel.suffix.lower() == ".zip" and len(files) == 1:
                    tmp = job.root / "upload.zip"
                    with open(tmp, "wb") as f:
                        shutil.copyfileobj(up.file, f)
                    with zipfile.ZipFile(tmp) as zf:
                        for info in zf.infolist():
                            r = _safe_rel(info.filename)
                            if r is None or info.is_dir():
                                continue
                            dst = job.input_dir / r
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(info) as srcf, open(dst, "wb") as out:
                                shutil.copyfileobj(srcf, out)
                            count += 1
                    tmp.unlink()
                else:
                    dst = job.input_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "wb") as f:
                        shutil.copyfileobj(up.file, f)
                    count += 1
        except zipfile.BadZipFile:
            manager.delete(job.id)
            raise HTTPException(400, "无法解压 zip 文件")
        if count == 0:
            manager.delete(job.id)
            raise HTTPException(400, "没有收到有效文件")
        job.name = name or (files[0].filename or "upload").split("/")[0]
        job.add_log(f"收到 {count} 个文件")
        manager.submit(job)
        return job.to_dict(full=False)

    @app.post("/api/demo")
    def demo(options: dict = None):
        from .synthetic import generate

        demo_dir = workspace / "demo" / "osgb"
        if not (demo_dir / "metadata.xml").is_file():
            generate(workspace / "demo", formats=("osgb",))
        job = manager.create(options or {}, "示例数据 (合成街道场景)")
        shutil.copytree(demo_dir, job.input_dir, dirs_exist_ok=True)
        gt = job.input_dir / "ground_truth.json"
        if gt.is_file():
            gt.unlink()
        manager.submit(job)
        return job.to_dict(full=False)

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_dict(full=False) for j in sorted(manager.jobs.values(), key=lambda j: -j.created)]

    def _job(job_id: str) -> Job:
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _job(job_id).to_dict()

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _job(job_id)
        if job.status == "running":
            raise HTTPException(409, "job is running")
        manager.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str):
        job = _job(job_id)
        rel = _safe_rel(path)
        if rel is None:
            raise HTTPException(400, "bad path")
        p = (job.work_dir / rel).resolve()
        if not str(p).startswith(str(job.work_dir.resolve())) or not p.is_file():
            raise HTTPException(404, "file not found")
        media = {"glb": "model/gltf-binary", "png": "image/png", "json": "application/json"}.get(p.suffix[1:].lower())
        return FileResponse(p, media_type=media)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        job = _job(job_id)
        archive = job.root / "result.zip"
        if job.status != "done" or not archive.is_file():
            raise HTTPException(409, "result not ready")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (job.name or job.id))
        return FileResponse(archive, media_type="application/zip", filename=f"{safe or job.id}_repaired.zip")

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
