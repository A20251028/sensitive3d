"""Whole-scene before / after GLB overviews for the web viewer.

For OSGB datasets the finest "cut" through the PagedLOD tree that stays
under a triangle budget is used, so large datasets still load in a browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .core.mesh import MeshFile
from .io.dataset import Dataset, FileInfo
from .preview import crop_parts, write_glb


def lod_cut(files: list[FileInfo], budget: int) -> tuple[list[str], int]:
    """Files of the finest complete level whose total triangle count fits ``budget``."""
    by_name = {}
    for f in files:
        by_name.setdefault(Path(f.rel).name, []).append(f)
    referenced = set()
    for f in files:
        referenced.update(f.children)
    roots = [f for f in files if Path(f.rel).name not in referenced]
    if not roots:
        return [f.rel for f in files], 0
    level = roots
    chosen = [f.rel for f in level]
    depth = 0
    while True:
        nxt = []
        for f in level:
            kids = []
            for c in f.children:
                cands = by_name.get(Path(c).name, [])
                same_dir = [k for k in cands if Path(k.rel).parent == Path(f.rel).parent]
                kids.extend(same_dir or cands)
            nxt.extend(kids if kids else [f])
        if [x.rel for x in nxt] == [x.rel for x in level]:
            break
        if sum(x.num_triangles for x in nxt) > budget:
            break
        level = nxt
        chosen = [f.rel for f in level]
        depth += 1
    return chosen, depth


def _load(ds: Dataset, rels: list[str], max_tex: int) -> MeshFile:
    pairs = []
    for rel in rels:
        # every file of the cut shows its own level of detail
        m = ds.load(rel)
        pairs += [(p, m.texture_of(p)) for p in m.parts]
    lo = np.full(3, -1e18)
    hi = np.full(3, 1e18)
    return crop_parts(pairs, lo, hi, max_tex=max_tex)


def write_overviews(src: Path, out: Path, prev_dir: Path, report: Optional[dict] = None, budget: int = 400_000) -> dict:
    prev_dir.mkdir(parents=True, exist_ok=True)
    a = Dataset(src)
    files = a.scan()
    if a.kind == "osgb":
        rels, depth = lod_cut(files, budget)
    else:
        rels, depth = [f.rel for f in files], 0
    max_tex = 1024 if len(rels) <= 16 else 512
    before = _load(a, rels, max_tex)
    lo, hi = before.bounds()
    origin = (lo + hi) / 2
    write_glb(before, prev_dir / "overview_before.glb", origin)
    b = Dataset(out)
    b.scan()
    after = _load(b, rels, max_tex)
    write_glb(after, prev_dir / "overview_after.glb", origin)
    return {
        "before": "previews/overview_before.glb",
        "after": "previews/overview_after.glb",
        "origin": [float(x) for x in origin],
        "min": [float(x) for x in lo],
        "max": [float(x) for x in hi],
        "files": len(rels),
        "level": depth,
        "triangles": int(sum(len(p.faces) for p in before.parts)),
    }
