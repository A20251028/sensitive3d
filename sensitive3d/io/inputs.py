"""Input hygiene shared by zip import, folder upload, the CLI and :class:`Dataset`.

* :func:`is_ignored` is the single rule for OS sidecar files (macOS
  AppleDouble ``._*``, ``.DS_Store``, ``__MACOSX/``, Windows ``Thumbs.db`` /
  ``desktop.ini``).  Ignored files never become dataset files and are never
  copied into a working copy, but they are never deleted from the source.
* :func:`safe_rel` turns an uploaded / zipped name into a safe relative path.
* :func:`iter_files`, :func:`file_manifest`, :func:`verify_manifest`,
  :func:`copy_tree` and :func:`safe_extract_zip` only ever *read* the source.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

IGNORED_NAMES = frozenset({"__macosx", ".ds_store", "thumbs.db", "desktop.ini"})
_DRIVE = re.compile(r"^[A-Za-z]:")
_CHUNK = 1 << 20


def is_ignored(rel: str | os.PathLike) -> bool:
    """True when any component of the relative path ``rel`` is an OS sidecar.

    Matches ``__MACOSX``, ``.DS_Store``, ``Thumbs.db``, ``desktop.ini`` (case
    insensitive) and AppleDouble names starting with ``._``.
    """
    for part in str(rel).replace("\\", "/").split("/"):
        if part.lower() in IGNORED_NAMES or part.startswith("._"):
            return True
    return False


def safe_rel(name: str) -> Optional[PurePosixPath]:
    """Relative path without traversal; None when unusable.

    Backslashes are treated as separators; absolute paths, drive letters,
    ``..`` components, NUL bytes and empty names are rejected.
    """
    name = (name or "").replace("\\", "/")
    if "\x00" in name:
        return None
    p = PurePosixPath(name)
    parts = [x for x in p.parts if x not in ("", ".")]
    if not parts or p.is_absolute() or any(x == ".." for x in parts) or _DRIVE.match(parts[0]) or ":" in parts[0]:
        return None
    return PurePosixPath(*parts)


def is_inside(path: Path, root_real: Path) -> bool:
    """True when ``path`` (symlinks resolved) lies inside the resolved directory ``root_real``."""
    try:
        return path.resolve().is_relative_to(root_real)
    except (OSError, RuntimeError):  # broken / looping symlink
        return False


def iter_files(
    root: str | Path,
    ignored: Optional[list[str]] = None,
    skipped: Optional[list[dict]] = None,
) -> list[Path]:
    """Regular files under ``root`` (sorted by relative path), sidecars excluded.

    Symlinked files are kept when they resolve inside ``root``.  Symlinks that
    resolve outside ``root`` (or are broken) and symlinked directories (never
    followed, to avoid loops and duplicates) are skipped.

    Optional out-lists: ``ignored`` receives the relative posix paths of the
    sidecar files that were left out, ``skipped`` receives
    ``{"rel": str, "reason": str}`` for every skipped symlink.
    """
    root = Path(root)
    root_real = root.resolve()
    found: list[tuple[str, Path]] = []
    ign: list[str] = []
    skp: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        for name in list(dirnames):
            p = d / name
            if p.is_symlink():
                rel = p.relative_to(root).as_posix()
                if not is_ignored(rel):
                    reason = "目录符号链接指向输入目录之外" if not is_inside(p, root_real) else "目录符号链接 (未跟随)"
                    skp.append({"rel": rel, "reason": reason})
                dirnames.remove(name)
        dirnames.sort()
        for name in filenames:
            p = d / name
            rel = p.relative_to(root).as_posix()
            if is_ignored(rel):
                ign.append(rel)
                continue
            if p.is_symlink():
                if not is_inside(p, root_real):
                    skp.append({"rel": rel, "reason": "符号链接指向输入目录之外或已失效"})
                    continue
            if not p.is_file():  # fifo, socket, broken link ...
                skp.append({"rel": rel, "reason": "不是普通文件"})
                continue
            found.append((rel, p))
    found.sort(key=lambda x: x[0])
    if ignored is not None:
        ignored.extend(sorted(ign))
    if skipped is not None:
        skipped.extend(sorted(skp, key=lambda x: x["rel"]))
    return [p for _, p in found]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(root: str | Path, hash: bool = True) -> dict:
    """``{"root", "files": [{"rel", "size", "sha256"}], "ignored", "skipped", "total_bytes"}``.

    ``sha256`` is None when ``hash`` is False.  Only reads ``root``.
    """
    root = Path(root)
    ignored: list[str] = []
    skipped: list[dict] = []
    files = []
    total = 0
    for p in iter_files(root, ignored, skipped):
        size = p.stat().st_size
        total += size
        files.append({"rel": p.relative_to(root).as_posix(), "size": size, "sha256": sha256_file(p) if hash else None})
    return {"root": str(root), "files": files, "ignored": ignored, "skipped": skipped, "total_bytes": total}


def verify_manifest(manifest: dict, root: Optional[str | Path] = None) -> dict:
    """Compare the non-ignored files under ``root`` with an earlier manifest.

    Returns ``{"ok", "changed", "missing", "added"}`` (lists of relative paths).
    Hashes are compared when the manifest has them, sizes otherwise.
    """
    root = Path(root if root is not None else manifest["root"])
    old = {f["rel"]: f for f in manifest["files"]}
    hashed = any(f.get("sha256") for f in manifest["files"])
    new = {f["rel"]: f for f in file_manifest(root, hash=hashed)["files"]}
    changed = sorted(
        r for r in old.keys() & new.keys()
        if old[r]["size"] != new[r]["size"] or (hashed and old[r].get("sha256") != new[r].get("sha256"))
    )
    missing = sorted(old.keys() - new.keys())
    added = sorted(new.keys() - old.keys())
    return {"ok": not (changed or missing or added), "changed": changed, "missing": missing, "added": added}


def _copy_file(src: Path, dst: Path, hash: bool) -> Optional[str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not hash:
        shutil.copy2(src, dst)
        return None
    h = hashlib.sha256()
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(_CHUNK):
            h.update(chunk)
            fo.write(chunk)
    shutil.copystat(src, dst)
    return h.hexdigest()


def copy_files(src_root: str | Path, rels: Iterable[str], dst_root: str | Path, hash: bool = True) -> dict:
    """Copy the given relative files from ``src_root`` to ``dst_root``.

    Ignored sidecars and unsafe names are skipped.  Never writes to the source:
    a destination that is the source file itself is refused.
    Returns a manifest of what was copied (sha256 of the copied bytes).
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    files, ignored = [], []
    total = 0
    for rel in rels:
        r = safe_rel(str(rel))
        if r is None:
            raise ValueError(f"非法的相对路径: {rel}")
        if is_ignored(r):
            ignored.append(r.as_posix())
            continue
        s, d = src_root / r, dst_root / r
        if d.exists() and d.resolve() == s.resolve():
            raise ValueError(f"输出文件与输入文件相同, 拒绝覆盖原始数据: {s}")
        digest = _copy_file(s, d, hash)
        size = s.stat().st_size
        total += size
        files.append({"rel": r.as_posix(), "size": size, "sha256": digest})
    return {"root": str(dst_root), "source": str(src_root), "files": files, "ignored": ignored, "total_bytes": total}


def copy_tree(src: str | Path, dst: str | Path, hash: bool = True) -> dict:
    """Copy every non-ignored file of ``src`` into ``dst`` (existing files are overwritten).

    Nothing in ``src`` is modified or deleted.  ``dst`` inside ``src`` (or the
    other way round, or both the same) is refused with ValueError.
    Returns a manifest of what was copied plus the ignored / skipped lists.
    """
    src, dst = Path(src), Path(dst)
    s_real, d_real = src.resolve(), dst.resolve()
    if not src.is_dir():
        raise ValueError(f"输入目录不存在: {src}")
    if s_real == d_real or d_real.is_relative_to(s_real) or s_real.is_relative_to(d_real):
        raise ValueError(f"输出目录不能与输入目录相同或互相包含: {src} -> {dst}")
    ignored: list[str] = []
    skipped: list[dict] = []
    paths = iter_files(src, ignored, skipped)
    dst.mkdir(parents=True, exist_ok=True)
    out = copy_files(src, [p.relative_to(src).as_posix() for p in paths], dst, hash=hash)
    out["ignored"] = ignored
    out["skipped"] = skipped
    return out


# ---------------------------------------------------------------------------
# zip import
# ---------------------------------------------------------------------------


def _zip_name(info: zipfile.ZipInfo) -> str:
    """Member name; legacy (non UTF-8 flagged) names are tried as UTF-8, then GBK."""
    name = info.filename
    if info.flag_bits & 0x800:
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return name
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return name


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    return info.create_system == 3 and stat.S_ISLNK(info.external_attr >> 16)


def safe_extract_zip(
    zip_path: str | Path,
    dst: str | Path,
    max_files: int = 200_000,
    max_bytes: int = 50 * 1024**3,
) -> dict:
    """Extract ``zip_path`` into ``dst`` with zip-slip and size protection.

    Members with unsafe names (absolute, drive letters, ``..``), symlinks and
    duplicates are rejected; OS sidecars are skipped.  Exceeding ``max_files``
    or ``max_bytes`` (checked against the headers and again while writing)
    raises ValueError; partially extracted files are then left in ``dst``,
    which the caller should discard.

    Returns ``{"files": n, "bytes": n, "ignored": [rel], "rejected": [name]}``.
    """
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    d_real = dst.resolve()
    gib = max_bytes / 1024**3
    ignored: list[str] = []
    rejected: list[str] = []
    todo: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    seen: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        declared = 0
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = _zip_name(info)
            rel = safe_rel(name)
            if rel is None or _is_zip_symlink(info):
                rejected.append(name)
                continue
            if is_ignored(rel):
                ignored.append(rel.as_posix())
                continue
            if rel.as_posix() in seen:
                rejected.append(name)
                continue
            seen.add(rel.as_posix())
            todo.append((info, rel))
            declared += info.file_size
            if len(todo) > max_files:
                raise ValueError(f"压缩包内文件数量超过上限 {max_files}")
            if declared > max_bytes:
                raise ValueError(f"压缩包解压后总大小超过上限 {gib:.1f} GiB")
        written = 0
        count = 0
        folded: set[str] = set()
        for info, rel in todo:
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            # a pre-existing symlinked directory inside dst must not redirect the write
            if not target.parent.resolve().is_relative_to(d_real) or target.is_symlink():
                rejected.append(rel.as_posix())
                continue
            # names differing only in case collide on case-insensitive file systems (macOS)
            key = rel.as_posix().casefold()
            if key in folded and target.exists():
                rejected.append(rel.as_posix())
                continue
            folded.add(key)
            with zf.open(info) as fi, open(target, "wb") as fo:
                while chunk := fi.read(_CHUNK):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"压缩包解压后总大小超过上限 {gib:.1f} GiB")
                    fo.write(chunk)
            count += 1
    return {"files": count, "bytes": written, "ignored": sorted(ignored), "rejected": rejected}
