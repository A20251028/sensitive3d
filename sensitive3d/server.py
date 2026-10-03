"""Web service: a job walks through auditable steps.

    import (copy + structure scan) -> preview (original model) -> detect (read only)
    -> review (human decisions, saved) -> repair (only reviewed decisions) -> package

Endpoints (JSON unless noted)::

    GET    /api/health                        bridge self-test, version, local import roots
    POST   /api/jobs                          multipart upload: files[] (one .zip, or folder files with
                                              relative paths as names) + options (JSON) + name
    POST   /api/jobs/local                    {"path", "name", "options"}: copy a local folder (only when
                                              started with --allow-local-root, only from localhost)
    POST   /api/demo                          job on a freshly generated synthetic dataset
    GET    /api/jobs                          list jobs
    GET    /api/jobs/{id}                     state of every step, progress, log, summaries
    GET    /api/jobs/{id}/scan                full import-check report
    POST   /api/jobs/{id}/detect              run detection only ({"options": {...}} optional)
    GET    /api/jobs/{id}/detection           detection report (candidates + provenance + evidence paths)
    PUT    /api/jobs/{id}/review              {"decisions": [...], "protected": [...]}: validated + saved
    GET    /api/jobs/{id}/review              saved review
    POST   /api/jobs/{id}/repair              execute the saved review
    GET    /api/jobs/{id}/repair              repair report
    POST   /api/jobs/{id}/cancel              cancel the running / queued steps
    GET    /api/jobs/{id}/files/{path}        evidence / previews / reports of a job
    GET    /api/jobs/{id}/download            zip: repaired dataset + reports
    DELETE /api/jobs/{id}                     remove a job (never touches a local source folder)
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

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .io.inputs import copy_tree, file_manifest, is_ignored, safe_extract_zip, safe_rel, verify_manifest

WEB_DIR = Path(__file__).resolve().parents[1] / "web"
MAX_LOG = 600
STEPS = ("import", "preview", "detect", "repair", "package")
STEP_LABELS = {"import": "导入检查", "preview": "原始预览", "detect": "只检测", "repair": "执行修复", "package": "打包下载"}
LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}


def _safe_rel(name: str) -> Optional[PurePosixPath]:  # kept for callers of the old name
    return safe_rel(name)


def _strip_common_root(root: Path) -> Path:
    """Descend into single wrapper folders (e.g. zip of a folder) until the dataset root."""
    cur = root
    for _ in range(5):
        entries = [p for p in cur.iterdir() if not is_ignored(p.name) and not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir() and not (cur / "Data").is_dir() and not (cur / "metadata.xml").is_file():
            cur = entries[0]
        else:
            break
    return cur


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Job:
    def __init__(self, job_id: str, root: Path, options: dict, name: str):
        self.id = job_id
        self.root = root
        self.options = options
        self.name = name
        self.source: dict = {"type": "upload"}
        self.input_root: Optional[str] = None  # dataset root relative to input/
        self.steps = {s: {"status": "pending"} for s in STEPS}
        self.current: Optional[str] = None
        self.progress = 0.0
        self.message = "等待上传"
        self.log: list[str] = []
        self.scan: Optional[dict] = None  # summary
        self.detection: Optional[dict] = None  # counts
        self.repair: Optional[dict] = None  # summary
        self.overview: dict = {}
        self.created = time.time()
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def dataset_dir(self) -> Path:
        return self.input_dir / self.input_root if self.input_root else _strip_common_root(self.input_dir)

    @property
    def status(self) -> str:
        sts = [self.steps[s]["status"] for s in STEPS]
        if "running" in sts:
            return "running"
        if "queued" in sts:
            return "queued"
        if self.steps["package"]["status"] == "done":
            return "done"
        if "failed" in sts:
            return "failed"
        if any(x in ("cancelled", "stopped") for x in sts):
            return "stopped"
        return "ready"

    def add_log(self, msg: str) -> None:
        with self.lock:
            for line in str(msg).splitlines() or [""]:
                self.log.append(f"{time.strftime('%H:%M:%S')} {line}")
            del self.log[:-MAX_LOG]

    def to_dict(self, full: bool = True) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "current": self.current,
            "steps": self.steps,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created": self.created,
            "source": self.source,
            "options": self.options,
            "scan": self.scan,
            "detection": self.detection,
            "review_saved": (self.work_dir / "review.json").is_file(),
            "repair": self.repair,
            "overview": self.overview,
            "input_root": self.input_root,
        }
        if full:
            with self.lock:
                d["log"] = list(self.log[-250:])
        return d

    def save(self) -> None:
        data = self.to_dict()
        data["log"] = list(self.log)
        tmp = self.root / "job.json.tmp"
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        tmp.replace(self.root / "job.json")


class JobManager:
    def __init__(self, workspace: Path, defaults: Optional[dict] = None):
        self.workspace = workspace
        self.defaults = defaults or {}
        (workspace / "jobs").mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.queue: "queue.Queue[tuple[str, str, dict]]" = queue.Queue()
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
            for k in ("source", "input_root", "scan", "detection", "repair", "overview", "progress", "message", "created", "log"):
                if k in data and data[k] is not None:
                    setattr(job, k, data[k])
            job.steps = {s: dict((data.get("steps") or {}).get(s, {"status": "pending"})) for s in STEPS}
            for s in STEPS:
                if job.steps[s].get("status") in ("queued", "running"):
                    job.steps[s].update(status="failed", error="服务重启, 步骤中断")
            self.jobs[job.id] = job

    def create(self, options: dict, name: str) -> Job:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        root = self.workspace / "jobs" / job_id
        (root / "input").mkdir(parents=True)
        (root / "work").mkdir()
        job = Job(job_id, root, options, name)
        self.jobs[job_id] = job
        return job

    def options(self, job: Job, extra: Optional[dict] = None) -> dict:
        return _merge(_merge(self.defaults, job.options), extra or {})

    def enqueue(self, job: Job, step: str, extra: Optional[dict] = None) -> None:
        job.steps[step] = {"status": "queued", "queued": time.time()}
        job.save()
        self.queue.put((job.id, step, extra or {}))

    def cancel(self, job: Job) -> None:
        job.cancel_event.set()
        for s in STEPS:
            if job.steps[s]["status"] == "queued":
                job.steps[s].update(status="cancelled", message="已取消")
        job.add_log("收到取消请求")
        job.save()

    def _worker(self) -> None:
        while True:
            job_id, step, extra = self.queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.steps[step]["status"] != "queued":
                continue
            self._run_step(job, step, extra)

    def _run_step(self, job: Job, step: str, extra: dict) -> None:
        from .pipeline import Pipeline, PipelineConfig
        from .runctx import StopRun

        job.cancel_event.clear()
        job.current = step
        job.steps[step] = {"status": "running", "started": time.time()}
        job.message = f"{STEP_LABELS[step]}…"
        job.progress = 0.0
        job.save()

        def progress(frac: float, msg: str) -> None:
            job.progress = frac
            job.message = msg
            job.steps[step]["message"] = msg

        opts = self.options(job, extra)
        cfg = PipelineConfig.from_dict(opts)
        pipe = Pipeline(cfg, progress=progress, log=job.add_log, cancel_event=job.cancel_event)
        follow: list[str] = []
        try:
            if step == "import":
                self._import(job, pipe)
                follow = ["preview"]
                if opts.get("auto"):
                    follow.append("detect")
            elif step == "preview":
                self._preview(job, pipe)
            elif step == "detect":
                self._detect(job, pipe)
                if opts.get("auto") and job.steps["detect"].get("status") == "running":
                    self._auto_review(job)
                    follow = ["repair"]
            elif step == "repair":
                self._repair(job, pipe)
                if job.steps["repair"].get("status") == "running":
                    follow = ["package"]
            elif step == "package":
                self._package(job, pipe)
            if job.steps[step]["status"] == "running":
                job.steps[step]["status"] = "done"
        except StopRun as e:
            job.steps[step].update(status="cancelled" if e.reason == "cancelled" else "stopped", error=str(e))
            job.message = f"{STEP_LABELS[step]}已停止: {e}"
            follow = []
        except Exception as e:  # noqa: BLE001 - surfaced to the user, logged in full
            job.steps[step].update(status="failed", error=f"{type(e).__name__}: {e}")
            job.message = f"{STEP_LABELS[step]}失败: {e}"
            job.add_log(traceback.format_exc())
            follow = []
        finally:
            job.steps[step]["finished"] = time.time()
            job.current = None
            if job.steps[step]["status"] == "done":
                job.progress = 1.0
            job.save()
        if job.steps[step]["status"] == "done" and not job.cancel_event.is_set():
            for s in follow:
                self.enqueue(job, s)

    # -- steps ---------------------------------------------------------------
    def _import(self, job: Job, pipe) -> None:
        ctx = pipe.ctx
        src = job.source
        if src.get("type") == "local":
            source = Path(src["path"])
            ctx.progress(0.02, "记录本机源目录哈希 (只读)")
            before = file_manifest(source)
            ctx.check("导入")
            ctx.progress(0.3, f"复制 {len(before['files'])} 个文件到工作副本")
            copy_tree(source, job.input_dir)
            after = file_manifest(source)
            check = verify_manifest(before, source)
            src.update(files=len(before["files"]), bytes=before["total_bytes"], ignored=len(before.get("ignored", [])), unchanged=check["ok"] and after["files"] == before["files"])
            if not src["unchanged"]:
                raise RuntimeError("复制期间源目录发生变化, 请确认没有其他程序在写入后重试")
        root = _strip_common_root(job.input_dir)
        job.input_root = root.relative_to(job.input_dir).as_posix() if root != job.input_dir else ""
        ctx.window(0.4 if src.get("type") == "local" else 0.0, 1.0)
        ds, rep = pipe.scan(root, job.work_dir)
        job.scan = {
            k: rep.get(k)
            for k in ("format", "files", "files_ok", "tiles", "leaf_files", "triangles", "ok_for_repair", "blocking_errors", "warnings", "levels", "roots", "multi_parent", "missing_refs", "cycles", "unreachable", "ignored", "texture_issues", "decoded_bytes_total", "input_manifest", "failed")
        }
        job.add_log(f"导入检查: {rep.get('files_ok')}/{rep.get('files')} 个文件可读, 阻断问题 {len(rep.get('blocking_errors') or [])} 个")

    def _preview(self, job: Job, pipe) -> None:
        from .webpreview import write_overview

        pipe.ctx.progress(0.05, "生成原始模型预览")
        job.overview["before"] = write_overview(job.dataset_dir, job.work_dir / "previews", "before", ctx=pipe.ctx)

    def _dataset(self, job: Job, pipe):
        from .io.dataset import Dataset

        ds = Dataset(job.dataset_dir)
        pipe.ctx.progress(0.0, "读取数据集结构")
        pipe._bridge(ds.scan, what="结构扫描")
        return ds

    def _detect(self, job: Job, pipe) -> None:
        pipe.ctx.window(0.0, 0.1)
        ds = self._dataset(job, pipe)
        pipe.ctx.window(0.1, 1.0)
        rep = pipe.detect(ds, job.work_dir)
        job.detection = {**rep["counts"], "summary": rep["summary"], "stopped": rep.get("stopped"), "seconds": rep.get("seconds")}
        for f in ("review.json", "repair.json"):
            (job.work_dir / f).unlink(missing_ok=True)  # decisions refer to the previous candidate ids
        job.steps["repair"] = {"status": "pending"}
        job.steps["package"] = {"status": "pending"}
        job.repair = None
        if rep.get("stopped"):
            job.steps["detect"].update(status="stopped" if rep["stopped"]["reason"] != "cancelled" else "cancelled", error=rep["stopped"]["message"])

    def _auto_review(self, job: Job) -> None:
        from .review import default_decisions

        det = json.loads((job.work_dir / "detection.json").read_text(encoding="utf-8"))
        decisions = [{"id": d["id"], "accept": bool(d["accept"]), "operation": d["operation"]} for d in default_decisions(det) if d["accept"]]
        (job.work_dir / "review.json").write_text(json.dumps({"decisions": decisions, "protected": [], "auto": True}, ensure_ascii=False, indent=1), encoding="utf-8")

    def _repair(self, job: Job, pipe) -> None:
        detection = json.loads((job.work_dir / "detection.json").read_text(encoding="utf-8"))
        review = json.loads((job.work_dir / "review.json").read_text(encoding="utf-8"))
        pipe.ctx.window(0.0, 0.05)
        ds = self._dataset(job, pipe)
        pipe.ctx.window(0.05, 1.0)
        if job.output_dir.exists():
            shutil.rmtree(job.output_dir)
        rep = pipe.repair(ds, detection, review, job.output_dir, job.work_dir)
        job.repair = {
            k: rep.get(k)
            for k in ("files_modified", "faces_removed", "faces_added", "texels_painted", "warnings", "stopped", "input_check", "summary", "seconds")
        }
        job.repair["targets"] = len(rep.get("targets", []))
        job.repair["readback_failures"] = sum(1 for f in rep.get("files", []) if f.get("readback", {}).get("ok") is False)
        job.repair["protected_ok"] = all(p.get("ok") for p in rep.get("protected_check", [])) if rep.get("protected_check") else None
        if rep.get("stopped"):
            job.steps["repair"].update(status="cancelled" if rep["stopped"]["reason"] == "cancelled" else "stopped", error=rep["stopped"]["message"])

    def _package(self, job: Job, pipe) -> None:
        from .webpreview import write_overview

        rep = json.loads((job.work_dir / "repair.json").read_text(encoding="utf-8"))
        if rep.get("stopped"):
            raise RuntimeError("修复未完成, 不能打包")
        pipe.ctx.progress(0.1, "生成修复后预览")
        origin = (job.overview.get("before") or {}).get("origin")
        try:
            job.overview["after"] = write_overview(job.output_dir, job.work_dir / "previews", "after", origin=origin, ctx=pipe.ctx)
        except Exception as e:  # noqa: BLE001 - optional
            job.add_log(f"修复后预览生成失败: {e}")
        pipe.ctx.progress(0.5, "打包结果")
        archive = job.root / "result.zip"
        tmp = job.root / "result.zip.tmp"
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(job.output_dir.rglob("*")):
                if p.is_file():
                    pipe.ctx.check("打包")
                    zf.write(p, p.relative_to(job.output_dir).as_posix())
            for name in ("scan.json", "detection.json", "review.json", "repair.json"):
                rp = job.work_dir / name
                if rp.is_file():
                    zf.write(rp, f"sensitive3d_report/{name}")
        tmp.replace(archive)

    def delete(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        if job is not None:
            job.cancel_event.set()
            shutil.rmtree(job.root, ignore_errors=True)


def create_app(workspace: str | Path = "workspace", allow_local_roots: Optional[list] = None, defaults: Optional[dict] = None) -> FastAPI:
    workspace = Path(workspace).resolve()
    manager = JobManager(workspace, defaults)
    roots = [Path(r).resolve() for r in (allow_local_roots or [])]
    for r in roots:
        if workspace == r or workspace.is_relative_to(r) or r.is_relative_to(workspace):
            raise ValueError(f"允许导入的目录 {r} 不能与工作目录 {workspace} 重叠 (输入与输出必须隔离)")
    app = FastAPI(title="sensitive3d", version=__version__)
    app.state.manager = manager
    app.state.local_roots = roots

    @app.get("/api/health")
    def health():
        out = {"ok": True, "version": __version__, "local_import": [str(r) for r in roots]}
        try:
            from .io.osgb import bridge_status

            st = bridge_status()
            out["bridge"] = st
            out["osgb_bridge"] = bool(st.get("found")) and bool((st.get("selftest") or {}).get("ok"))
        except ImportError:
            from .io.osgb import bridge_available

            out["osgb_bridge"] = bridge_available()
        return out

    def _job(job_id: str) -> Job:
        job = manager.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    def _opts(options: str) -> dict:
        try:
            v = json.loads(options or "{}")
        except json.JSONDecodeError:
            raise HTTPException(400, "options must be JSON")
        if not isinstance(v, dict):
            raise HTTPException(400, "options must be a JSON object")
        return v

    @app.post("/api/jobs")
    async def create_job(files: list[UploadFile] = File(...), options: str = Form("{}"), name: str = Form("")):
        opts = _opts(options)
        job = manager.create(opts, name)
        count, ignored = 0, []
        try:
            for up in files:
                rel = safe_rel(up.filename or "")
                if rel is None:
                    ignored.append(up.filename)
                    continue
                if is_ignored(rel.as_posix()):
                    ignored.append(rel.as_posix())
                    continue
                if rel.suffix.lower() == ".zip" and len(files) == 1:
                    tmp = job.root / "upload.zip"
                    with open(tmp, "wb") as f:
                        shutil.copyfileobj(up.file, f)
                    stats = safe_extract_zip(tmp, job.input_dir)
                    tmp.unlink()
                    count += stats["files"]
                    ignored += list(stats.get("ignored", []))
                    if stats.get("rejected"):
                        job.add_log(f"zip 中 {len(stats['rejected'])} 个条目路径不安全, 已拒绝")
                else:
                    dst = job.input_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst, "wb") as f:
                        shutil.copyfileobj(up.file, f)
                    count += 1
        except (zipfile.BadZipFile, ValueError) as e:
            manager.delete(job.id)
            raise HTTPException(400, f"无法导入: {e}")
        if count == 0:
            manager.delete(job.id)
            raise HTTPException(400, "没有收到有效文件 (附属文件 ._* / .DS_Store / __MACOSX 已忽略)")
        job.name = name or (files[0].filename or "upload").split("/")[0]
        job.source = {"type": "upload", "files": count, "ignored": len(ignored)}
        job.add_log(f"收到 {count} 个文件, 忽略 {len(ignored)} 个附属文件")
        manager.enqueue(job, "import")
        return job.to_dict(full=False)

    @app.post("/api/jobs/local")
    def create_local(request: Request, payload: dict = Body(...)):
        if not roots:
            raise HTTPException(403, "未启用本机路径导入: 启动服务时用 --allow-local-root <目录> 指定允许读取的目录")
        if (request.client.host if request.client else "") not in LOCAL_HOSTS:
            raise HTTPException(403, "本机路径导入只允许从本机 (localhost) 访问")
        try:
            path = Path(str(payload.get("path", ""))).expanduser().resolve(strict=True)
        except (OSError, RuntimeError):
            raise HTTPException(400, "路径不存在")
        if not path.is_dir():
            raise HTTPException(400, "请给出数据集文件夹 (包含 Data/ 或 .obj/.glb 的目录)")
        if not any(path == r or path.is_relative_to(r) for r in roots):
            raise HTTPException(403, f"路径不在允许的目录内: {[str(r) for r in roots]}")
        if path.is_relative_to(workspace) or workspace.is_relative_to(path):
            raise HTTPException(400, "源目录不能与工作目录重叠")
        job = manager.create(dict(payload.get("options") or {}), str(payload.get("name") or path.name))
        job.source = {"type": "local", "path": str(path)}
        job.add_log(f"本机源目录 (只读复制): {path}")
        manager.enqueue(job, "import")
        return job.to_dict(full=False)

    @app.post("/api/demo")
    def demo(options: Optional[dict] = Body(None)):
        from .synthetic import generate

        # always regenerate into a fresh directory: never reuse a stale cached demo
        demo_dir = workspace / "demo" / uuid.uuid4().hex[:8]
        generate(demo_dir, formats=("osgb",), log=lambda *a: None)
        job = manager.create(options or {}, "示例数据 (合成街道场景)")
        copy_tree(demo_dir / "osgb", job.input_dir)
        for extra in ("ground_truth.json", "synthetic_meta.json"):
            if (job.input_dir / extra).is_file():
                (job.work_dir / extra).write_bytes((job.input_dir / extra).read_bytes())
                (job.input_dir / extra).unlink()
        shutil.rmtree(demo_dir, ignore_errors=True)
        job.source = {"type": "demo"}
        manager.enqueue(job, "import")
        return job.to_dict(full=False)

    @app.get("/api/jobs")
    def list_jobs():
        return [j.to_dict(full=False) for j in sorted(manager.jobs.values(), key=lambda j: -j.created)]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return _job(job_id).to_dict()

    def _report(job: Job, name: str) -> dict:
        p = job.work_dir / name
        if not p.is_file():
            raise HTTPException(404, f"{name} 尚未生成")
        return json.loads(p.read_text(encoding="utf-8"))

    @app.get("/api/jobs/{job_id}/scan")
    def get_scan(job_id: str):
        return _report(_job(job_id), "scan.json")

    def _busy(job: Job) -> None:
        if job.status in ("running", "queued"):
            raise HTTPException(409, "任务正在运行, 请等待或取消")

    @app.post("/api/jobs/{job_id}/detect")
    def detect(job_id: str, payload: Optional[dict] = Body(None)):
        job = _job(job_id)
        _busy(job)
        if job.steps["import"]["status"] != "done":
            raise HTTPException(409, "请先完成导入检查")
        extra = (payload or {}).get("options") or {}
        if extra:
            job.options = _merge(job.options, extra)
        manager.enqueue(job, "detect")
        return job.to_dict(full=False)

    @app.get("/api/jobs/{job_id}/detection")
    def get_detection(job_id: str):
        return _report(_job(job_id), "detection.json")

    @app.put("/api/jobs/{job_id}/review")
    def put_review(job_id: str, payload: dict = Body(...)):
        from .review import ReviewError, parse_protected, resolve_targets

        job = _job(job_id)
        _busy(job)
        detection = _report(job, "detection.json")
        review = {"decisions": list(payload.get("decisions") or []), "protected": list(payload.get("protected") or []), "saved": time.time()}
        try:
            parse_protected(review)
            targets, skipped = resolve_targets(detection, review, auto=False)
        except ReviewError as e:
            raise HTTPException(400, str(e))
        (job.work_dir / "review.json").write_text(json.dumps(review, ensure_ascii=False, indent=1), encoding="utf-8")
        job.steps["repair"] = {"status": "pending"}
        job.steps["package"] = {"status": "pending"}
        job.add_log(f"审核已保存: {len(targets)} 个目标执行修复, {len(skipped)} 个不处理")
        job.save()
        return {"targets": [{"id": t.candidate_id, "operation": t.operation} for t in targets], "skipped": skipped}

    @app.get("/api/jobs/{job_id}/review")
    def get_review(job_id: str):
        return _report(_job(job_id), "review.json")

    @app.post("/api/jobs/{job_id}/repair")
    def repair(job_id: str, payload: Optional[dict] = Body(None)):
        job = _job(job_id)
        _busy(job)
        if not (job.work_dir / "review.json").is_file():
            raise HTTPException(409, "请先保存审核结果")
        if job.scan and not job.scan.get("ok_for_repair"):
            raise HTTPException(409, "导入检查存在阻断性问题, 不能执行修复: " + "; ".join((job.scan.get("blocking_errors") or [])[:3]))
        extra = (payload or {}).get("options") or {}
        manager.enqueue(job, "repair", extra)
        return job.to_dict(full=False)

    @app.get("/api/jobs/{job_id}/repair")
    def get_repair(job_id: str):
        return _report(_job(job_id), "repair.json")

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        job = _job(job_id)
        manager.cancel(job)
        return job.to_dict(full=False)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = _job(job_id)
        if job.status == "running":
            raise HTTPException(409, "job is running; cancel it first")
        manager.delete(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str):
        job = _job(job_id)
        rel = safe_rel(path)
        if rel is None:
            raise HTTPException(400, "bad path")
        base = job.work_dir.resolve()
        p = (base / rel).resolve()
        if not p.is_relative_to(base) or not p.is_file():
            raise HTTPException(404, "file not found")
        media = {"glb": "model/gltf-binary", "png": "image/png", "json": "application/json"}.get(p.suffix[1:].lower())
        return FileResponse(p, media_type=media)

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str):
        job = _job(job_id)
        archive = job.root / "result.zip"
        if job.steps["package"]["status"] != "done" or not archive.is_file():
            raise HTTPException(409, "result not ready")
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (job.name or job.id))
        return FileResponse(archive, media_type="application/zip", filename=f"{safe or job.id}_repaired.zip")

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
