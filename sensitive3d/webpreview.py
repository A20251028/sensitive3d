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
from .io.lodgraph import LodGraph
from .preview import crop_parts, write_glb


def lod_cut(files: list[FileInfo], budget: int, graph: Optional[LodGraph] = None) -> tuple[list[str], int]:
    """Files of the finest complete level whose total triangle count fits ``budget``.

    The cut walks the PagedLOD reference graph (normalised relative paths)
    level by level from the root files.  A file referenced by several parents
    appears once, files that failed to scan are never used (their parent is
    kept instead, so no hole appears) and reference cycles cannot loop.
    ``graph`` defaults to one built from ``FileInfo.children``.
    """
    ok = {f.rel: f for f in files if getattr(f, "ok", True)}
    if graph is None:
        graph = LodGraph({f.rel: f.children for f in files if f.rel in ok}, existing=[f.rel for f in files])
    level = [r for r in graph.roots if r in ok]
    if not level:
        return sorted(ok), 0
    visited = set(level)
    chosen, depth = list(level), 0
    for _ in range(len(graph.nodes) + 1):
        nxt: list[str] = []
        placed: set[str] = set()

        def put(rel: str) -> None:
            if rel not in placed:
                placed.add(rel)
                nxt.append(rel)

        for rel in level:
            kids = [k for k in graph.children(rel) if k != rel]
            if not kids or any(k not in ok for k in kids):
                put(rel)  # finest available here (or a child is unusable)
                continue
            new = [k for k in kids if k not in visited]
            if new:
                for k in new:
                    put(k)
            elif any(graph.in_cycle(rel, k) for k in kids):
                put(rel)  # only back references: stop descending
            # else: every child is already shown via another parent
        if not nxt or nxt == level:
            break
        if sum(ok[r].num_triangles for r in nxt) > budget:
            break
        visited.update(nxt)
        level = nxt
        chosen = list(level)
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
        rels, depth = lod_cut(files, budget, graph=a.graph)
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
