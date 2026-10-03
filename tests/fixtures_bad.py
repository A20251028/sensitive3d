"""Helpers building small, partly broken datasets for the dataset-scan tests.

OSGB files are written with the bridge ``build`` command (see
``sensitive3d.io.osgb.build_osgb``): every file holds one textured quad, and
coarse files are PagedLODs whose ``FILE_CHILD`` names are stored exactly as
given, so any reference layout (sub directories, missing files, several
parents, cycles, wrong case) can be produced.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

APPLEDOUBLE = b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        " + bytes(64)


class OsgbBuilder:
    """Writes quads + build.txt into ``work`` and builds files below ``root``."""

    def __init__(self, root: Path, work: Path):
        self.root = Path(root)
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.count = 0
        tex = np.zeros((16, 16, 3), np.uint8)
        tex[..., 0] = 200
        tex[8:, :, 1] = 120
        Image.fromarray(tex).save(self.work / "tex.jpg", quality=90)

    def geode(self, x: float = 0.0, y: float = 0.0, size: float = 1.0, z: float = 0.0) -> str:
        k = self.count
        self.count += 1
        v = np.array([[x, y, z], [x + size, y, z], [x + size, y + size, z], [x, y + size, z]], np.float32)
        uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        t = np.array([0, 1, 2, 0, 2, 3], np.uint32)
        v.tofile(self.work / f"v{k}.f32")
        uv.tofile(self.work / f"uv{k}.f32")
        t.tofile(self.work / f"t{k}.u32")
        return f"BEGIN_GEODE\nGEOMETRY v{k}.f32 uv{k}.f32 - t{k}.u32 tex.jpg\nEND_GEODE"

    def plod(self, children: list[str], x: float = 0.0, y: float = 0.0, size: float = 1.0) -> str:
        c = (x + size / 2, y + size / 2, 0.0)
        s = f"BEGIN_PAGEDLOD {c[0]} {c[1]} {c[2]} {size} 1\nCHILD 0 400 {self.geode(x, y, size)}\n"
        for ch in children:
            s += f"FILE_CHILD {ch} 400 1e30\n"
        return s + "END_PAGEDLOD\n"

    def write(self, rel: str, children: list[str] | None = None, x: float = 0.0, y: float = 0.0, size: float = 1.0) -> Path:
        """A PagedLOD file referencing ``children`` (a plain leaf geode when None / empty)."""
        from sensitive3d.io.osgb import build_osgb

        dst = self.root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = self.plod(children, x, y, size) if children else self.geode(x, y, size) + "\n"
        build_osgb(text, self.work, dst)
        return dst


def metadata(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "metadata.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<ModelMetadata version="1">\n<SRS>ENU:31,121</SRS>\n'
        "<SRSOrigin>0,0,0</SRSOrigin>\n</ModelMetadata>\n",
        encoding="utf-8",
    )


def clean_dataset(root: Path, work: Path) -> dict:
    """(a) Two tiles, 3 levels, children in a sub directory, same file names in both tiles.

    Data/T/T.osgb -> sub/L1.osgb -> L2_0.osgb, L2_1.osgb (relative to sub/), same for Data/U.
    """
    b = OsgbBuilder(root, work)
    metadata(root)
    for i, t in enumerate(("T", "U")):
        x = 10.0 * i
        b.write(f"Data/{t}/{t}.osgb", ["sub/L1.osgb"], x, 0, 2)
        b.write(f"Data/{t}/sub/L1.osgb", ["L2_0.osgb", "L2_1.osgb"], x, 0, 2)
        b.write(f"Data/{t}/sub/L2_0.osgb", None, x, 0, 1)
        b.write(f"Data/{t}/sub/L2_1.osgb", None, x + 1, 0, 1)
    return {"leaves": [f"Data/{t}/sub/L2_{j}.osgb" for t in ("T", "U") for j in range(2)]}


def missing_ref_dataset(root: Path, work: Path) -> None:
    """(b) Data/M/M.osgb references M_L1.osgb (exists) and M_gone.osgb (does not)."""
    b = OsgbBuilder(root, work)
    metadata(root)
    b.write("Data/M/M.osgb", ["M_L1.osgb", "M_gone.osgb"])
    b.write("Data/M/M_L1.osgb", None)


def multi_parent_dataset(root: Path, work: Path) -> None:
    """(c) P -> A, B;  A -> S;  B -> B1 -> S.  S has two parents, shortest depth 2."""
    b = OsgbBuilder(root, work)
    metadata(root)
    b.write("Data/P/P.osgb", ["A.osgb", "B.osgb"])
    b.write("Data/P/A.osgb", ["S.osgb"])
    b.write("Data/P/B.osgb", ["B1.osgb"])
    b.write("Data/P/B1.osgb", ["S.osgb"])
    b.write("Data/P/S.osgb", None)


def cycle_dataset(root: Path, work: Path) -> None:
    """(d) C <-> D reference each other; E references itself and a leaf E1."""
    b = OsgbBuilder(root, work)
    metadata(root)
    b.write("Data/C/C.osgb", ["D.osgb"])
    b.write("Data/C/D.osgb", ["C.osgb"])
    b.write("Data/C/E.osgb", ["E.osgb", "E1.osgb"])
    b.write("Data/C/E1.osgb", None)


def garbage_dataset(root: Path, work: Path) -> None:
    """(e) A valid root whose only child is random bytes, plus a lone garbage file."""
    b = OsgbBuilder(root, work)
    metadata(root)
    b.write("Data/G/G.osgb", ["G_L1.osgb"])
    rng = np.random.default_rng(7)
    (root / "Data/G/G_L1.osgb").write_bytes(rng.integers(0, 256, 4096, dtype=np.uint8).tobytes())
    (root / "Data/G/junk.osgb").write_bytes(b"not an osgb file\x00" * 10)


def case_mismatch_dataset(root: Path, work: Path) -> None:
    """(g) Data/K/K.osgb references k_l1.osgb, the file on disk is K_L1.osgb."""
    b = OsgbBuilder(root, work)
    metadata(root)
    b.write("Data/K/K.osgb", ["k_l1.osgb"])
    b.write("Data/K/K_L1.osgb", None)


SIDECARS = [
    ".DS_Store",
    "Data/.DS_Store",
    "Data/T/._T.osgb",
    "Data/T/sub/._L1.osgb",
    "__MACOSX/Data/T/._T.osgb",
    "Data/Thumbs.db",
    "desktop.ini",
]


def add_sidecars(root: Path) -> list[str]:
    """(f) macOS / Windows sidecar files next to real data."""
    for rel in SIDECARS:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(APPLEDOUBLE if Path(rel).name.startswith("._") else b"\x00sidecar\x00")
    return list(SIDECARS)


def zip_dataset(root: Path, zip_path: Path, prefix: str = "", evil: bool = True) -> None:
    """Zip ``root`` (sidecars included) plus, optionally, unsafe member names."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, names in os.walk(root):
            for n in sorted(names):
                p = Path(dirpath) / n
                zf.write(p, prefix + p.relative_to(root).as_posix())
        if evil:
            zf.writestr("../evil.txt", b"zip slip")
            zf.writestr("/abs/evil.txt", b"absolute")
            zf.writestr("C:\\evil.txt", b"drive")
            zf.writestr(prefix + "a/../../evil2.txt", b"traversal")
