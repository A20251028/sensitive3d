"""Environment diagnostics behind ``sensitive3d doctor``.

Checks the Python runtime, the third-party packages and the native OSGB
bridge (which is actually executed: version + selftest), and turns the
findings into human readable problems and hints.
"""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from typing import Optional

from . import __version__

# (import name, distribution name used in the install hint)
REQUIRED_PACKAGES = (
    ("numpy", "numpy"),
    ("numba", "numba"),
    ("scipy", "scipy"),
    ("cv2", "opencv-contrib-python-headless"),
    ("PIL", "pillow"),
    ("trimesh", "trimesh"),
    ("skimage", "scikit-image"),
    ("fast_simplification", "fast-simplification"),
    ("mapbox_earcut", "mapbox-earcut"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
)
OPTIONAL_PACKAGES = (("onnxruntime", "onnxruntime"),)
ENV_VARS = ("S3D_OSGB_BRIDGE", "OSG_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH", "S3D_BRIDGE_TIMEOUT")
MIN_PYTHON = (3, 9)


def _package_version(module: str, dist: str) -> tuple[Optional[str], Optional[str]]:
    """(version, error); importing also catches broken native extensions."""
    try:
        mod = importlib.import_module(module)
    except Exception as e:  # ImportError, OSError from missing shared libraries, ...
        return None, f"{type(e).__name__}: {e}"
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as dist_version

            version = dist_version(dist)
        except Exception:
            version = "unknown"
    return str(version), None


def _rosetta() -> bool:
    """True when this (x86_64) Python runs translated on Apple Silicon."""
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() == "1"
    except Exception:
        return False


def environment_report(run_selftest: bool = True) -> dict:
    """Collect the environment state; ``report["ok"]`` is False when something blocks OSGB processing."""
    from .io.osgb import BridgeError, bridge_status, find_bridge

    problems: list[str] = []
    hints: list[str] = []

    python = {
        "version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "prefix": sys.prefix,
        "virtualenv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
    }
    if sys.version_info[:2] < MIN_PYTHON:
        problems.append(f"Python {python['version']} 过旧，需要 >= {'.'.join(map(str, MIN_PYTHON))}")
    plat = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "rosetta": _rosetta(),
    }
    if plat["rosetta"]:
        hints.append(
            "当前 Python 在 Rosetta (x86_64) 下运行，而 Homebrew OpenSceneGraph 是 arm64：请使用 arm64 的 python3.12 "
            "（/opt/homebrew/bin/python3.12）重新创建虚拟环境"
        )

    packages: dict[str, Optional[str]] = {}
    import_errors: dict[str, str] = {}
    missing_required = []
    for module, dist in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        version, error = _package_version(module, dist)
        packages[module] = version
        if error:
            import_errors[module] = error
            if (module, dist) in REQUIRED_PACKAGES:
                missing_required.append(dist)
                problems.append(f"Python 依赖 {module} 无法导入（{dist}）：{error}")
    if missing_required:
        hints.append("在项目目录的虚拟环境中安装依赖：pip install -e '.[dev]'")
    if packages.get("onnxruntime") is None:
        hints.append("onnxruntime 未安装（可选）：YOLO / LaMa ONNX 模型不可用，pip install onnxruntime 后可启用")

    if run_selftest:
        bridge = bridge_status()
    else:
        bridge = {"found": False, "path": None, "ok": None, "version": None, "selftest": None, "error": None, "hint": None}
        try:
            bridge["path"] = find_bridge()
            bridge["found"] = True
        except BridgeError as e:
            bridge["error"], bridge["hint"] = str(e), e.hint
    if not bridge["found"]:
        problems.append(f"未找到 osgb_bridge，无法读写 OSGB：{bridge['error']}")
    elif run_selftest and not bridge["ok"]:
        problems.append(f"osgb_bridge 无法正常工作：{bridge['error']}")
    if bridge.get("hint") and (not bridge["found"] or bridge["ok"] is False):
        hints.append(bridge["hint"])
    version = bridge.get("version") or {}
    for key, plugin in (version.get("plugins") or {}).items():
        if not plugin.get("path") and bridge.get("ok") is not True:
            hints.append(
                f"OpenSceneGraph 插件 {plugin.get('library')}（{key}）不在插件搜索路径中："
                "请把 OSG_LIBRARY_PATH 设为包含 osgPlugins-<版本> 目录的路径（Homebrew: \"$(brew --prefix)/lib\"）"
            )

    env = {k: os.environ.get(k) for k in ENV_VARS}
    return {
        "ok": not problems,
        "sensitive3d": __version__,
        "python": python,
        "platform": plat,
        "packages": packages,
        "import_errors": import_errors,
        "bridge": bridge,
        "env": env,
        "problems": problems,
        "hints": hints,
    }


def format_report(report: dict) -> str:
    """Human readable (Chinese) rendering of :func:`environment_report`."""
    lines = [f"sensitive3d {report['sensitive3d']} 环境检查", ""]
    py, plat = report["python"], report["platform"]
    lines.append(f"Python    {py['version']} ({py['executable']}){'  [venv]' if py['virtualenv'] else ''}")
    lines.append(f"平台      {plat['platform']} ({plat['system']} {plat['machine']}{', Rosetta' if plat['rosetta'] else ''})")
    lines.append("")
    lines.append("Python 依赖")
    optional = {m for m, _ in OPTIONAL_PACKAGES}
    for name, version in report["packages"].items():
        if version is None:
            state = "未安装（可选）" if name in optional else "缺失"
        else:
            state = version
        lines.append(f"  {name:<22}{state}")
    lines.append("")
    b = report["bridge"]
    lines.append("osgb_bridge")
    lines.append(f"  路径        {b.get('path') or '未找到'}")
    v = b.get("version") or {}
    if v:
        lines.append(f"  版本        OSG {v.get('osg_version')}，bridge {v.get('bridge_version')}")
        lines.append("  插件搜索路径")
        for p in v.get("plugin_paths") or []:
            lines.append(f"    {p}")
        lines.append("  插件")
        for key, plugin in (v.get("plugins") or {}).items():
            lines.append(f"    {key:<16}{plugin.get('path') or '未找到 (' + str(plugin.get('library')) + ')'}")
    st = b.get("selftest")
    if st:
        lines.append(f"  自检        {'通过' if st.get('ok') else '失败'}")
        for name, ok in (st.get("checks") or {}).items():
            lines.append(f"    {name:<16}{'OK' if ok else '失败'}")
        for err in st.get("errors") or []:
            lines.append(f"    ! {err}")
    elif b.get("ok") is None and b.get("found"):
        lines.append("  自检        未运行")
    lines.append("")
    lines.append("环境变量")
    for k, val in report["env"].items():
        lines.append(f"  {k} = {val if val is not None else '（未设置）'}")
    if report["problems"]:
        lines.append("")
        lines.append("问题")
        lines.extend(f"  - {p}" for p in report["problems"])
    if report["hints"]:
        lines.append("")
        lines.append("建议")
        lines.extend(f"  - {h}" for h in report["hints"])
    lines.append("")
    lines.append("结论: " + ("环境正常" if report["ok"] else "存在问题，见上方“问题”与“建议”"))
    return "\n".join(lines)
