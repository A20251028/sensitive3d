"""OSGB reading / patching through the native ``osgb_bridge`` helper.

Every bridge call runs in its own process group with a timeout (env
``S3D_BRIDGE_TIMEOUT``, seconds, default 600, ``0`` disables it) and can be
cancelled with a callable or a ``threading.Event``; the whole group is killed
in both cases.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
from PIL import Image

from ..core.mesh import MeshFile, MeshPart, Texture, transform_directions, transform_points

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TIMEOUT = 600.0
SELFTEST_TIMEOUT = 60.0
_POLL = 0.1
_ENV_KEYS = ("S3D_OSGB_BRIDGE", "OSG_LIBRARY_PATH", "OSG_LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH")

# a callable returning True when the work should stop, or a threading.Event
Cancel = Union[Callable[[], bool], threading.Event, None]

_HINT_NOT_FOUND = (
    "未找到 osgb_bridge：运行 ./scripts/build_bridge.sh 编译（需要 OpenSceneGraph，macOS: brew install open-scene-graph cmake），"
    "或把 S3D_OSGB_BRIDGE 设为可执行文件的完整路径。详见 docs/macos.md"
)
_HINT_ENV_MISSING = (
    "S3D_OSGB_BRIDGE 指向的文件不存在：修正该路径，或取消该变量（unset S3D_OSGB_BRIDGE）以使用 "
    "native/osgb_bridge/build/osgb_bridge。详见 docs/macos.md"
)
_HINT_DYLD = (
    "osgb_bridge 无法加载 OpenSceneGraph 动态库（dyld）：确认已 brew install open-scene-graph，"
    '必要时 export DYLD_LIBRARY_PATH="$(brew --prefix)/lib"，然后重新运行 ./scripts/build_bridge.sh。详见 docs/macos.md'
)
_HINT_LDSO = (
    "osgb_bridge 无法加载 OpenSceneGraph 动态库：安装运行库（Debian/Ubuntu: sudo apt-get install openscenegraph "
    "libopenscenegraph-dev）或设置 LD_LIBRARY_PATH，然后重新运行 ./scripts/build_bridge.sh。详见 docs/macos.md"
)
_HINT_PLUGINS = (
    "OpenSceneGraph 插件（osgPlugins-<版本> 目录）未找到：把 OSG_LIBRARY_PATH 设为包含该目录的路径"
    '（Homebrew: export OSG_LIBRARY_PATH="$(brew --prefix)/lib"；Debian/Ubuntu 通常为 /usr/lib/x86_64-linux-gnu），'
    "再运行 sensitive3d doctor 检查。详见 docs/macos.md"
)
_HINT_EXEC = (
    "无法执行 osgb_bridge：检查可执行权限（chmod +x）和 CPU 架构（Apple Silicon 上需在本机重新编译："
    "./scripts/build_bridge.sh）。详见 docs/macos.md"
)
_HINT_OLD = "osgb_bridge 版本过旧，不支持该命令：重新运行 ./scripts/build_bridge.sh。"


class BridgeError(RuntimeError):
    """The native bridge is missing, failed, or produced unusable output."""

    def __init__(
        self,
        message: str,
        *,
        cmd: Optional[list[str]] = None,
        returncode: Optional[int] = None,
        stdout: str = "",
        stderr: str = "",
        hint: Optional[str] = None,
    ):
        super().__init__(message)
        self.cmd = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.hint = hint


class BridgeTimeout(BridgeError):
    """The bridge did not finish in time; its process group was killed."""


class BridgeCancelled(BridgeError):
    """The caller cancelled the bridge call; its process group was killed."""


class BridgeTextureError(BridgeError):
    """A texture of an OSGB file has no usable pixels (missing external image, corrupt data)."""

    def __init__(self, message: str, *, file: str = "", texture: str = "", index: int = -1, **kw):
        super().__init__(message, **kw)
        self.file = file
        self.texture = texture
        self.index = index


# ---------------------------------------------------------------------------
# locating and running the binary
# ---------------------------------------------------------------------------


def find_bridge() -> str:
    """Path of the bridge binary.  An explicit ``S3D_OSGB_BRIDGE`` is never silently replaced."""
    env = os.environ.get("S3D_OSGB_BRIDGE")
    if env:
        p = Path(env).expanduser()
        if not p.is_file():
            raise BridgeError(f"S3D_OSGB_BRIDGE points to a missing file: {env}", hint=_HINT_ENV_MISSING)
        return str(p)
    candidates = [
        str(_REPO_ROOT / "native" / "osgb_bridge" / "build" / "osgb_bridge"),
        str(_REPO_ROOT / "native" / "osgb_bridge" / "build" / "Release" / "osgb_bridge.exe"),
        shutil.which("osgb_bridge") or "",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    raise BridgeError(
        "osgb_bridge not found. Build it with scripts/build_bridge.sh (needs OpenSceneGraph) "
        "or set S3D_OSGB_BRIDGE to its path.",
        hint=_HINT_NOT_FOUND,
    )


def bridge_available() -> bool:
    """True when the binary exists (see :func:`bridge_status` for a real check)."""
    try:
        find_bridge()
        return True
    except BridgeError:
        return False


def default_timeout() -> Optional[float]:
    """Per-call timeout from ``S3D_BRIDGE_TIMEOUT`` (seconds); None means no limit."""
    raw = os.environ.get("S3D_BRIDGE_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else None


def _cancel_check(cancel: Cancel) -> Optional[Callable[[], bool]]:
    if cancel is None:
        return None
    is_set = getattr(cancel, "is_set", None)
    if callable(is_set):
        return is_set
    if callable(cancel):
        return cancel
    raise TypeError("cancel must be a callable returning bool or a threading.Event")


def _diagnose(text: str, returncode: Optional[int] = None) -> Optional[str]:
    """A user hint for common start-up / plugin problems found in stderr."""
    low = text.lower()
    hints = []
    if "dyld" in low or "library not loaded" in low or "image not found" in low:
        hints.append(_HINT_DYLD)
    if "error while loading shared libraries" in low or "cannot open shared object file" in low:
        hints.append(_HINT_LDSO)
    if any(
        k in low
        for k in (
            "could not find plugin",
            "unable to find a plugin",
            "not found in the osg library path",
            "no reader for",
            "no readerwriter",
        )
    ):
        hints.append(_HINT_PLUGINS)
    if "bad cpu type" in low or "exec format error" in low or "permission denied" in low:
        hints.append(_HINT_EXEC)
    if returncode == 1 and "usage:" in low and "osgb_bridge" in low:
        hints.append(_HINT_OLD)
    return "\n".join(hints) if hints else None


def _tail(text: str, lines: int = 15, width: int = 500) -> str:
    rows = [r[:width] for r in text.strip().splitlines()[-lines:]]
    return "\n".join("  " + r for r in rows)


def _describe(cmd: list[str], returncode: Optional[int]) -> str:
    what = f"osgb_bridge {cmd[1]}" if len(cmd) > 1 else "osgb_bridge"
    if returncode is None:
        return f"{what} ({shlex.join(cmd)})"
    if returncode < 0:
        try:
            sig = signal.Signals(-returncode).name
        except ValueError:
            sig = f"signal {-returncode}"
        code = f"killed by {sig}, exit code {returncode}"
    else:
        code = f"exit code {returncode}"
    return f"{what} failed ({code}): {shlex.join(cmd)}"


def _with_details(message: str, stderr: str, hint: Optional[str]) -> str:
    if stderr.strip():
        message += "\nstderr (last lines):\n" + _tail(stderr)
    if hint:
        message += "\nhint: " + hint
    return message


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _run(args: list[str], timeout: Optional[float] = None, cancel: Cancel = None) -> str:
    """Run the bridge and return its stdout.

    ``timeout`` defaults to :func:`default_timeout`; ``cancel`` is a callable or
    ``threading.Event``.  On timeout / cancellation the whole process group is
    killed and BridgeTimeout / BridgeCancelled is raised; a non-zero exit
    raises BridgeError.  Every BridgeError carries the partial stdout/stderr.
    """
    exe = find_bridge()
    cmd = [exe, *[str(a) for a in args]]
    limit = default_timeout() if timeout is None else (timeout if timeout > 0 else None)
    is_cancelled = _cancel_check(cancel)
    if is_cancelled is not None and is_cancelled():
        raise BridgeCancelled(f"{_describe(cmd, None)} cancelled before start", cmd=cmd)
    kwargs: dict = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    else:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
        )
    except OSError as e:
        hint = _diagnose(str(e)) or _HINT_EXEC
        raise BridgeError(_with_details(f"cannot start {shlex.join(cmd)}: {e}", "", hint), cmd=cmd, hint=hint) from e

    deadline = None if limit is None else time.monotonic() + limit
    reason = None
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=_POLL)
                break
            except subprocess.TimeoutExpired:
                pass
            if is_cancelled is not None and is_cancelled():
                reason = "cancelled"
            elif deadline is not None and time.monotonic() >= deadline:
                reason = "timeout"
            if reason:
                _kill_group(proc)
                try:
                    out, err = proc.communicate(timeout=10)
                except subprocess.TimeoutExpired:  # a grandchild left the group and holds the pipes
                    proc.kill()
                    out, err = b"", b""
                break
    except BaseException:  # e.g. KeyboardInterrupt or a failing cancel callable: never leave it running
        if proc.poll() is None:
            _kill_group(proc)
        try:
            proc.communicate(timeout=10)
        except Exception:
            pass
        raise

    stdout = out.decode("utf-8", "surrogateescape")
    stderr = err.decode("utf-8", "replace")
    details = dict(cmd=cmd, returncode=proc.returncode, stdout=stdout, stderr=stderr)
    if reason == "cancelled":
        msg = f"{_describe(cmd, None)} was cancelled; its process group was killed"
        raise BridgeCancelled(_with_details(msg, stderr, None), **details)
    if reason == "timeout":
        msg = (
            f"{_describe(cmd, None)} timed out after {limit:g} s (S3D_BRIDGE_TIMEOUT); "
            "its process group was killed"
        )
        raise BridgeTimeout(_with_details(msg, stderr, None), **details)
    if proc.returncode != 0:
        hint = _diagnose(stderr, proc.returncode)
        raise BridgeError(_with_details(_describe(cmd, proc.returncode), stderr, hint), hint=hint, **details)
    return stdout


# ---------------------------------------------------------------------------
# structural scan
# ---------------------------------------------------------------------------


def _failed_record(path: str, error: str) -> dict:
    return {"file": path, "ok": False, "error": error, "geometries": [], "images": [], "children": []}


def _normalize_record(rec: dict) -> dict:
    rec.setdefault("geometries", [])
    rec.setdefault("images", [])
    rec.setdefault("children", [])
    if not rec.get("ok"):
        rec["ok"] = False
        rec.setdefault("error", "unknown error")
    return rec


def _info(paths: list[str], timeout: Optional[float], cancel: Cancel) -> tuple[list[dict], Optional[BridgeError]]:
    """Run ``info``; return the complete records it printed and the error, if any."""
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(b"".join(os.fsencode(p) + b"\n" for p in paths))
        list_file = f.name
    error: Optional[BridgeError] = None
    try:
        text = _run(["info", list_file], timeout=timeout, cancel=cancel)
    except BridgeCancelled:
        raise
    except BridgeError as e:  # crash, timeout, non-zero exit: keep what was printed
        text, error = e.stdout or "", e
    finally:
        os.unlink(list_file)
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except ValueError:  # cut off by a crash
            continue
        if isinstance(rec, dict) and "file" in rec:
            records.append(rec)
    return records, error


def _scan_error(error: Optional[BridgeError], count: int) -> str:
    if error is not None:
        return str(error)
    return f"osgb_bridge info returned {count} records for 1 file"


def _scan_chunk(chunk: list[str], timeout: Optional[float], cancel: Cancel) -> list[dict]:
    result: list[Optional[dict]] = [None] * len(chunk)
    send = []
    for k, p in enumerate(chunk):
        if not p or "\n" in p or "\r" in p:
            result[k] = _failed_record(p, "unsupported file name (empty or contains a line break)")
        else:
            send.append(k)
    if send:
        records, error = _info([chunk[k] for k in send], timeout, cancel)
        matched = 0
        for k, rec in zip(send, records):
            if rec.get("file") != chunk[k]:
                break
            result[k] = _normalize_record(rec)
            matched += 1
        rest = send[matched:]
        if len(send) == 1 and rest:
            result[rest[0]] = _failed_record(chunk[rest[0]], _scan_error(error, len(records)))
        else:
            # crash / timeout / count mismatch: isolate the remaining files
            for k in rest:
                single, err1 = _info([chunk[k]], timeout, cancel)
                if len(single) == 1 and single[0].get("file") == chunk[k]:
                    result[k] = _normalize_record(single[0])
                else:
                    result[k] = _failed_record(chunk[k], _scan_error(err1, len(single)))
    return [r if r is not None else _failed_record(p, "not scanned") for r, p in zip(result, chunk)]


def scan_osgb(
    paths: Iterable[str | Path], batch: int = 200, *, timeout: Optional[float] = None, cancel: Cancel = None
) -> list[dict]:
    """Fast structural scan (bounds, LOD flags, UV / texture state, images, children) without decoding textures.

    Returns exactly one record per path, in order.  Files that cannot be read
    (or that crash / hang the bridge) yield ``{"ok": False, "error": ...}``;
    after a crash or a record-count mismatch the remaining files of the batch
    are re-scanned one by one, so one bad file never loses the batch.
    ``timeout`` applies to each bridge call.
    """
    paths = [os.fspath(p) for p in paths]
    batch = max(1, int(batch))
    out: list[dict] = []
    for i in range(0, len(paths), batch):
        out.extend(_scan_chunk(paths[i : i + batch], timeout, cancel))
    return out


# ---------------------------------------------------------------------------
# export / patch / build
# ---------------------------------------------------------------------------


def _load_texture(src: Path, tmp: Path, k: int, tex: dict) -> np.ndarray:
    name = tex.get("name", "")
    w, h, c = int(tex.get("width", 0)), int(tex.get("height", 0)), int(tex.get("channels", 0))
    if not tex.get("file") or tex.get("valid") is False or w <= 0 or h <= 0 or c <= 0:
        reason = tex.get("error") or "no pixel data"
        raise BridgeTextureError(
            f"{src}: texture #{k} '{name}' could not be decoded: {reason}", file=str(src), texture=name, index=k
        )
    raw = np.fromfile(tmp / tex["file"], dtype=np.uint8)
    if raw.size != w * h * c:
        raise BridgeTextureError(
            f"{src}: texture #{k} '{name}' has {raw.size} bytes, expected {w}x{h}x{c}",
            file=str(src),
            texture=name,
            index=k,
        )
    return raw.reshape(h, w, c)


def read_osgb(
    path: str | Path, rel_path: Optional[str] = None, *, timeout: Optional[float] = None, cancel: Cancel = None
) -> MeshFile:
    """Read one OSGB file.  Raises BridgeTextureError when a texture has no usable pixels."""
    path = Path(path)
    with tempfile.TemporaryDirectory(prefix="s3d_exp_") as tmp:
        _run(["export", str(path), tmp], timeout=timeout, cancel=cancel)
        t = Path(tmp)
        man = json.loads((t / "manifest.json").read_text(encoding="utf-8"))
        textures = []
        for k, tex in enumerate(man["textures"]):
            img = _load_texture(path, t, k, tex)
            enc = tex.get("encoding", "jpg")
            textures.append(Texture(img, name=tex.get("name", ""), encoding="png" if enc == "png" else "jpg"))
        for g in man["geometries"]:
            if g.get("texture_missing") and g.get("texture", -1) < 0:
                raise BridgeTextureError(
                    f"{path}: geometry #{g['index']} '{g.get('name', '')}' uses a texture without an image",
                    file=str(path),
                )
        parts = []
        for g in man["geometries"]:
            if not g["vertices"]:
                continue
            m = np.array(g["matrix"], dtype=np.float64).reshape(4, 4)
            v = np.fromfile(t / g["vertices"], dtype=np.float32).reshape(-1, 3).astype(np.float64)
            f = np.fromfile(t / g["triangles"], dtype=np.uint32).reshape(-1, 3).astype(np.int64)
            uv = np.fromfile(t / g["texcoords"], dtype=np.float32).reshape(-1, 2) if g["texcoords"] else None
            n = np.fromfile(t / g["normals"], dtype=np.float32).reshape(-1, 3) if g["normals"] else None
            c = np.fromfile(t / g["colors"], dtype=np.float32).reshape(-1, 4) if g["colors"] else None
            parts.append(
                MeshPart(
                    vertices=transform_points(v, m),
                    faces=f,
                    uvs=uv,
                    normals=transform_directions(n, m) if n is not None else None,
                    colors=c.astype(np.float64) if c is not None else None,
                    texture=g["texture"] if g["texture"] >= 0 else None,
                    has_finer=bool(g["has_finer"]),
                    index=int(g["index"]),
                    matrix=m,
                    name=g.get("name", ""),
                )
            )
        meta = {"lods": man["lods"], "children": sorted({c["file"] for l in man["lods"] for c in l["children"] if c["file"]})}
        return MeshFile(rel_path or path.name, "osgb", parts, textures, meta)


def encode_texture(tex: Texture) -> tuple[bytes, str]:
    img = tex.image
    if img.ndim == 3 and img.shape[2] == 4 and tex.encoding != "png" and bool((img[..., 3] == 255).all()):
        # opaque RGBA, e.g. a JPEG decoded by the macOS ImageIO plugin: keep it a JPEG
        img = img[..., :3]
    pil = Image.fromarray(img if img.shape[2] in (3, 4) else img[..., :3])
    buf = io.BytesIO()
    if tex.encoding == "png" or img.shape[2] == 4:
        pil.save(buf, format="PNG", optimize=False)
        return buf.getvalue(), "png"
    pil.convert("RGB").save(buf, format="JPEG", quality=95, subsampling=0)
    return buf.getvalue(), "jpg"


def write_osgb(
    mesh: MeshFile, src: str | Path, dst: str | Path, *, timeout: Optional[float] = None, cancel: Cancel = None
) -> None:
    """Write ``mesh`` to ``dst`` by patching ``src``; unchanged files are copied."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not mesh.dirty:
        if Path(src).resolve() != dst.resolve():
            shutil.copyfile(src, dst)
        return
    with tempfile.TemporaryDirectory(prefix="s3d_patch_") as tmp:
        t = Path(tmp)
        lines = []
        for part in mesh.parts:
            if not part.dirty:
                continue
            inv = np.linalg.inv(part.matrix)
            base = f"g{part.index}"
            transform_points(part.vertices, inv).astype(np.float32).tofile(t / f"{base}_v.f32")
            part.faces.astype(np.uint32).tofile(t / f"{base}_t.u32")
            uvf = nf = cf = "-"
            if part.uvs is not None:
                part.uvs.astype(np.float32).tofile(t / f"{base}_uv.f32")
                uvf = f"{base}_uv.f32"
            if part.normals is not None:
                transform_directions(part.normals, inv).astype(np.float32).tofile(t / f"{base}_n.f32")
                nf = f"{base}_n.f32"
            if part.colors is not None:
                part.colors.astype(np.float32).tofile(t / f"{base}_c.f32")
                cf = f"{base}_c.f32"
            lines.append(f"geometry {part.index} {base}_v.f32 {uvf} {nf} {cf} {base}_t.u32")
        for k, tex in enumerate(mesh.textures):
            if not tex.dirty:
                continue
            data, ext = encode_texture(tex)
            name = f"tex{k}.{ext}"
            (t / name).write_bytes(data)
            lines.append(f"texture {k} {name}")
        (t / "patch.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        _run(["patch", str(src), str(t), str(dst)], timeout=timeout, cancel=cancel)


def build_osgb(
    description: str, workdir: str | Path, dst: str | Path, *, timeout: Optional[float] = None, cancel: Cancel = None
) -> None:
    """Build an OSGB file from a ``build.txt`` description (see osgb_bridge.cpp)."""
    workdir = Path(workdir)
    desc = workdir / "build.txt"
    desc.write_text(description, encoding="utf-8")
    _run(["build", str(desc), str(dst)], timeout=timeout, cancel=cancel)


# ---------------------------------------------------------------------------
# runtime diagnostics
# ---------------------------------------------------------------------------

_cache: dict = {}
_cache_lock = threading.Lock()


def _cache_key(kind: str) -> tuple:
    exe = find_bridge()
    st = os.stat(exe)
    return (kind, os.path.realpath(exe), st.st_mtime_ns, st.st_size, tuple(os.environ.get(k) for k in _ENV_KEYS))


def _parse_object(text: str) -> Optional[dict]:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
            except ValueError:
                return None
            return data if isinstance(data, dict) else None
    return None


def bridge_version(force: bool = False, timeout: float = 30.0) -> dict:
    """``osgb_bridge version`` as a dict (cached per binary path / mtime); raises BridgeError."""
    key = _cache_key("version")
    with _cache_lock:
        if not force and key in _cache:
            return json.loads(json.dumps(_cache[key]))
    text = _run(["version"], timeout=timeout)
    data = _parse_object(text)
    if data is None:
        raise BridgeError(f"osgb_bridge version printed no JSON: {text[:500]!r}", stdout=text, hint=_HINT_OLD)
    data["path"] = key[1]
    with _cache_lock:
        _cache[key] = data
    return json.loads(json.dumps(data))


def bridge_selftest(force: bool = False, timeout: float = SELFTEST_TIMEOUT) -> dict:
    """Run ``osgb_bridge selftest`` (JPEG / PNG / OSGB write + read + export).

    Always returns a dict with at least ``ok``, ``checks``, ``errors`` and
    ``hint``.  Results are cached per binary path / mtime and OSG environment
    variables (timeouts are not cached); ``force`` re-runs it.
    """
    try:
        key = _cache_key("selftest")
    except (BridgeError, OSError) as e:
        return {"ok": False, "checks": {}, "errors": [str(e)], "hint": getattr(e, "hint", None) or _HINT_NOT_FOUND}
    with _cache_lock:
        if not force and key in _cache:
            return json.loads(json.dumps(_cache[key]))
    with tempfile.TemporaryDirectory(prefix="s3d_selftest_") as tmp:
        try:
            text = _run(["selftest", tmp], timeout=timeout)
            stderr = ""
        except BridgeTimeout as e:
            return {"ok": False, "checks": {}, "errors": [str(e)], "hint": None, "stderr": e.stderr[-4000:]}
        except BridgeError as e:
            text, stderr = e.stdout or "", e.stderr or ""
            data = _parse_object(text)
            if data is None:
                data = {"ok": False, "checks": {}, "errors": [str(e)], "hint": e.hint, "stderr": stderr[-4000:]}
                with _cache_lock:
                    _cache[key] = data
                return json.loads(json.dumps(data))
    data = _parse_object(text)
    if data is None:
        data = {"ok": False, "checks": {}, "errors": [f"selftest printed no JSON: {text[:500]!r}"]}
    data.setdefault("checks", {})
    data.setdefault("errors", [])
    data["ok"] = bool(data.get("ok")) and all(data["checks"].values())
    data["stderr"] = stderr[-4000:]
    data["hint"] = None if data["ok"] else (_diagnose("\n".join(data["errors"]) + "\n" + stderr) or _HINT_PLUGINS)
    data["path"] = key[1]
    with _cache_lock:
        _cache[key] = data
    return json.loads(json.dumps(data))


def bridge_status(force: bool = False, timeout: float = SELFTEST_TIMEOUT) -> dict:
    """Is the bridge usable?  Locates the binary and actually runs its selftest.

    Returns ``{"found", "path", "ok", "version", "selftest", "error", "hint"}``.
    """
    status: dict = {"found": False, "path": None, "ok": False, "version": None, "selftest": None, "error": None, "hint": None}
    try:
        status["path"] = find_bridge()
        status["found"] = True
    except BridgeError as e:
        status["error"] = str(e)
        status["hint"] = e.hint or _HINT_NOT_FOUND
        return status
    try:
        status["version"] = bridge_version(force=force)
    except BridgeError as e:
        status["error"] = str(e)
        status["hint"] = e.hint
        return status
    selftest = bridge_selftest(force=force, timeout=timeout)
    status["selftest"] = selftest
    status["ok"] = bool(selftest.get("ok"))
    if not status["ok"]:
        status["error"] = "osgb_bridge selftest failed: " + ("; ".join(selftest.get("errors") or []) or "unknown error")
        status["hint"] = selftest.get("hint")
    return status
