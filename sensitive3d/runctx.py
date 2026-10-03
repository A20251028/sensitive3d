"""Run control shared by all pipeline stages: progress, cancellation and budgets."""

from __future__ import annotations

import os
import resource
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional


class StopRun(Exception):
    """Base class: the run stopped early; evidence gathered so far is kept."""

    reason = "stopped"


class Cancelled(StopRun):
    reason = "cancelled"


class BudgetExceeded(StopRun):
    reason = "budget"


@dataclass
class RunLimits:
    max_seconds: Optional[float] = None  # wall clock for the whole stage
    max_memory_mb: Optional[float] = None  # peak resident memory of this process
    bridge_timeout: Optional[float] = None  # one osgb_bridge call
    max_decoded_mb: Optional[float] = None  # decoded texture memory loaded at once (tile / file)

    @staticmethod
    def from_dict(d: Optional[dict]) -> "RunLimits":
        lim = RunLimits()
        for k, v in (d or {}).items():
            if hasattr(lim, k) and v not in (None, ""):
                setattr(lim, k, float(v))
        return lim

    def to_dict(self) -> dict:
        return asdict(self)


def peak_memory_mb() -> float:
    """Peak resident set size of this process (macOS reports bytes, Linux KiB)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def current_memory_mb() -> float:
    """Current resident set size where the OS tells us cheaply, else the peak."""
    try:
        with open("/proc/self/statm", encoding="ascii") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return peak_memory_mb()


class RunContext:
    """Progress reporting plus cooperative cancellation / budget checks.

    Stages call :meth:`check` between files; it raises :class:`Cancelled` or
    :class:`BudgetExceeded`.  ``cancel_event`` is also handed to the bridge so
    that a running subprocess is killed promptly.
    """

    def __init__(
        self,
        limits: Optional[RunLimits] = None,
        progress: Optional[Callable[[float, str], None]] = None,
        log: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ):
        self.limits = limits or RunLimits()
        self._progress = progress or (lambda f, m: None)
        self._log = log or (lambda m: None)
        self.cancel_event = cancel_event or threading.Event()
        self.started = time.time()
        self.lo, self.hi = 0.0, 1.0  # current progress window

    # -- progress ----------------------------------------------------------
    def window(self, lo: float, hi: float) -> "RunContext":
        self.lo, self.hi = lo, hi
        return self

    def progress(self, frac: float, msg: str) -> None:
        f = self.lo + (self.hi - self.lo) * min(max(frac, 0.0), 1.0)
        self._progress(f, msg)
        self._log(msg)

    def log(self, msg: str) -> None:
        self._log(msg)

    # -- control -----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.time() - self.started

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check(self, where: str = "") -> None:
        if self.cancel_event.is_set():
            raise Cancelled(f"已取消 ({where})" if where else "已取消")
        lim = self.limits
        if lim.max_seconds is not None and self.elapsed > lim.max_seconds:
            raise BudgetExceeded(f"超过时间预算 {lim.max_seconds:.0f} 秒 ({where})")
        if lim.max_memory_mb is not None:
            mem = current_memory_mb()
            if mem > lim.max_memory_mb:
                raise BudgetExceeded(f"内存 {mem:.0f} MB 超过预算 {lim.max_memory_mb:.0f} MB ({where})")

    def check_decoded(self, mb: float, what: str) -> None:
        """Refuse to decode more texture memory than allowed before loading it."""
        lim = self.limits.max_decoded_mb
        if lim is not None and mb > lim:
            raise BudgetExceeded(f"{what} 需解码约 {mb:.0f} MB 贴图, 超过预算 {lim:.0f} MB")

    def bridge_kwargs(self) -> dict:
        """Keyword arguments for io.osgb calls (timeout + cancellation)."""
        return {"timeout": self.limits.bridge_timeout, "cancel": self.cancel_event}

    def stats(self) -> dict:
        return {"seconds": round(self.elapsed, 1), "peak_memory_mb": round(peak_memory_mb(), 1)}
