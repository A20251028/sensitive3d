"""PagedLOD reference graph of a dataset, keyed by normalised relative path.

A PagedLOD stores its child file names relative to the directory of the
file that contains it.  Real datasets are not always trees: a file can be
referenced by several parents, references can be missing, differ in case
from the file on disk, or even form cycles.  :class:`LodGraph` resolves all
references once and answers structural questions without ever recursing
(every traversal is an explicit queue / stack with a visited set).
"""

from __future__ import annotations

import posixpath
import re
import unicodedata
from collections import deque
from typing import Iterable, Mapping, Optional

_DRIVE = re.compile(r"^[A-Za-z]:")


def _norm(rel: str) -> str:
    return posixpath.normpath(str(rel).replace("\\", "/"))


def _fold(rel: str) -> str:
    """Key for the case / unicode-normalisation insensitive fallback match."""
    return unicodedata.normalize("NFC", rel).casefold()


def resolve_ref(parent_rel: str, child_name: str) -> Optional[str]:
    """Dataset-relative path of ``child_name`` referenced from ``parent_rel``.

    The child is joined to the parent's directory (backslashes normalised).
    Returns None for empty / absolute / URL references and for paths that
    escape the dataset root.
    """
    child = str(child_name or "").replace("\\", "/")
    if not child.strip() or child.startswith("/") or _DRIVE.match(child) or "://" in child:
        return None
    base = posixpath.dirname(_norm(parent_rel))
    joined = posixpath.normpath(posixpath.join(base, child))
    if joined in (".", "..") or joined.startswith("../") or joined.startswith("/"):
        return None
    return joined


class LodGraph:
    """Directed graph parent -> child file of a dataset.

    ``children``: ``{rel: [child names exactly as stored]}`` (files that could
    not be read simply have no entry); ``existing``: every file of the dataset
    (defaults to the keys of ``children``).

    Attributes (all deterministic / sorted):

    * ``edges``         rel -> resolved child rels (stored order, deduplicated)
    * ``parents``       rel -> sorted parent rels
    * ``missing``       ``[{"parent", "ref", "resolved"}]`` unresolvable refs
    * ``warnings``      e.g. references that only matched case-insensitively
    * ``multi_parent``  ``[{"rel", "parents"}]`` files with more than one parent
    * ``roots``         files without parents (a self reference does not count)
    * ``cycles``        strongly connected components of size > 1 and self loops
    * ``depth``         shortest distance from any root (BFS)
    * ``unreachable``   files not reachable from any root (e.g. only inside cycles)
    * ``leaves``        files without resolved children
    """

    def __init__(self, children: Mapping[str, Iterable[str]], existing: Optional[Iterable[str]] = None):
        nodes = {_norm(r) for r in (existing if existing is not None else children.keys())}
        nodes.update(_norm(r) for r in children.keys())
        self.nodes: list[str] = sorted(nodes)
        folded: dict[str, list[str]] = {}
        for n in self.nodes:
            folded.setdefault(_fold(n), []).append(n)

        self.edges: dict[str, list[str]] = {n: [] for n in self.nodes}
        self.missing: list[dict] = []
        self.warnings: list[str] = []
        for parent in sorted(children, key=_norm):
            p = _norm(parent)
            for ref in children[parent]:
                target = resolve_ref(p, ref)
                if target is not None and target not in nodes:
                    cands = folded.get(_fold(target), [])
                    if len(cands) == 1:
                        self.warnings.append(f"引用大小写/编码与文件不一致: {p} -> {ref} 按 {cands[0]} 处理")
                        target = cands[0]
                    else:
                        if len(cands) > 1:
                            self.warnings.append(f"引用 {p} -> {ref} 匹配到多个仅大小写不同的文件: {', '.join(cands)}")
                        self.missing.append({"parent": p, "ref": ref, "resolved": target})
                        continue
                if target is None:
                    self.missing.append({"parent": p, "ref": ref, "resolved": None})
                    continue
                if target not in self.edges[p]:
                    self.edges[p].append(target)

        par: dict[str, set[str]] = {n: set() for n in self.nodes}
        for p, kids in self.edges.items():
            for k in kids:
                par[k].add(p)
        self.parents: dict[str, list[str]] = {n: sorted(v) for n, v in par.items()}
        self.multi_parent: list[dict] = [{"rel": n, "parents": ps} for n, ps in self.parents.items() if len(ps) > 1]
        self.roots: list[str] = [n for n in self.nodes if not (par[n] - {n})]
        self.leaves: list[str] = [n for n in self.nodes if not self.edges[n]]

        self.depth: dict[str, int] = {}
        queue = deque()
        for r in self.roots:
            self.depth[r] = 0
            queue.append(r)
        while queue:
            n = queue.popleft()
            for k in self.edges[n]:
                if k not in self.depth:
                    self.depth[k] = self.depth[n] + 1
                    queue.append(k)
        self.unreachable: list[str] = [n for n in self.nodes if n not in self.depth]

        self._scc, comps = self._components()
        self.cycles: list[list[str]] = sorted(sorted(c) for c in comps if len(c) > 1 or c[0] in self.edges[c[0]])

    # -- strongly connected components (iterative Tarjan) ----------------------
    def _components(self) -> tuple[dict[str, int], list[list[str]]]:
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        comp: dict[str, int] = {}
        comps: list[list[str]] = []
        counter = 0
        for start in self.nodes:
            if start in index:
                continue
            work = [(start, 0)]
            while work:
                v, i = work.pop()
                if i == 0:
                    index[v] = low[v] = counter
                    counter += 1
                    stack.append(v)
                    on_stack.add(v)
                kids = self.edges[v]
                if i < len(kids):
                    work.append((v, i + 1))
                    w = kids[i]
                    if w not in index:
                        work.append((w, 0))
                    elif w in on_stack:
                        low[v] = min(low[v], index[w])
                    continue
                # all children done: propagate low-link to the caller
                if low[v] == index[v]:
                    c = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp[w] = len(comps)
                        c.append(w)
                        if w == v:
                            break
                    comps.append(c)
                if work:
                    u = work[-1][0]
                    low[u] = min(low[u], low[v])
        return comp, comps

    # -- queries ---------------------------------------------------------------
    def children(self, rel: str) -> list[str]:
        return list(self.edges.get(_norm(rel), []))

    def in_cycle(self, a: str, b: str) -> bool:
        """True when ``a`` and ``b`` lie on a common reference cycle."""
        a, b = _norm(a), _norm(b)
        if a == b:
            return a in self.edges.get(a, [])
        return a in self._scc and self._scc.get(a) == self._scc.get(b)

    def levels(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for n in self.nodes:
            if n in self.depth:
                out.setdefault(self.depth[n], []).append(n)
        return dict(sorted(out.items()))

    def descendants(self, rel: str) -> list[str]:
        """Every file reachable from ``rel`` (BFS order, each once, ``rel`` excluded)."""
        start = _norm(rel)
        seen = {start}
        out = []
        queue = deque([start])
        while queue:
            n = queue.popleft()
            for k in self.edges.get(n, []):
                if k not in seen:
                    seen.add(k)
                    out.append(k)
                    queue.append(k)
        return out

    def subtree(self, rel: str) -> list[str]:
        return [_norm(rel)] + self.descendants(rel)

    def to_dict(self) -> dict:
        return {
            "files": len(self.nodes),
            "roots": list(self.roots),
            "levels": {d: len(v) for d, v in self.levels().items()},
            "leaves": len(self.leaves),
            "missing_refs": list(self.missing),
            "multi_parent": list(self.multi_parent),
            "cycles": [list(c) for c in self.cycles],
            "unreachable": list(self.unreachable),
            "warnings": list(self.warnings),
        }
