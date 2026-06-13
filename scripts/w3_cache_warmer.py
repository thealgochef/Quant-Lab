#!/usr/bin/env python3
"""Parallel per-day cache warmer for the W3 (D-036) dataset build.

Purpose
-------
The W3 serial train (``run_dashboard_session_experiment.py``) spends nearly all
its wall-clock in ``build_utility_dataset`` decoding/labeling one day at a time.
Each labeled day is written to a per-day cache
(``<data_dir>/<symbol>/<date>/ml_utility_<tag>.parquet``); a warm cache makes the
serial run effectively instant on that day. This script warms those caches in
parallel ahead of time, then exits. Tomorrow's serial train runs on warm cache.

Identity with a serial run
---------------------------
* **Config / cache tag.** The config is resolved through the SAME parser and
  resolver the experiment CLI uses (``run_dashboard_session_experiment``), so
  ``config.dataset_config_hash()`` — the ``ml_utility_<tag>`` cache key — is
  byte-identical to the train run (D-036 tag ``7850272e``). The cache PATH and
  the per-day build PATH (``_process_single_date`` -> ``process_single_date_stream``)
  are exactly the ones the serial loop calls.
* **Cache CONTENT.** ``build_utility_dataset`` threads ``prev_full_hl`` (the prior
  full trading day's PDH/PDL seed) sequentially through the days, and the per-day
  labeling depends on that seed. So building a day in isolation with the wrong
  seed would write a path-correct but content-WRONG cache that the train run would
  silently consume. We reproduce the serial seed exactly: the ``prev_full_hl``
  entering day *i* is the full (high, low) of the most-recent NON-EMPTY prior day
  in the window (or ``None`` if none) — see ``_seed_for_day``. For valid store days
  (non-empty bars) this collapses to "the previous window day's H/L", matching the
  serial loop's carry, including the empty-day fallback.

Operational properties
----------------------
* Top-level worker function (Windows ``spawn`` safe); heavy imports are inside the
  functions so spawned workers don't import Streamlit/pandas at module load.
* Resumable by construction: a day whose cache already EXISTS and LOADS is skipped.
* Self-healing write: after each build the written cache is re-read; on failure it
  is deleted and the day is rebuilt once, then reported FAILED if it still won't load.
* Per-day progress (day, rows, seconds, worker pid, peak RSS, status) is streamed to
  stdout and appended to ``W3A_WARM.log`` at the QL repo root (``*.log`` is gitignored).
* Peak RSS is read from the Windows API (``GetProcessMemoryInfo.PeakWorkingSetSize``)
  — the true per-process peak, no psutil dependency and no sampling thread.

Usage
-----
  # full warm at N workers (detached launch for the overnight run):
  python scripts/w3_cache_warmer.py --workers 4
  # smoke (memory probe): exactly two named days at N=2:
  python scripts/w3_cache_warmer.py --workers 2 --days 2026-02-10,2026-02-11
  # or first K uncached days:
  python scripts/w3_cache_warmer.py --workers 2 --limit 2
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path

# Self-contained import roots — no PYTHONPATH needed for main OR spawned workers.
_QL_ROOT = Path(__file__).resolve().parents[1]
_SRC = _QL_ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Exact D-036 config, as the ratified launch command (model/fold args don't affect
# the dataset cache tag, but are passed verbatim so the resolved config is identical).
EXP_ARGV = [
    "--preset", "all_to_ny",
    "--symbol", "NQ",
    "--bar-type", "147t",
    "--start", "2025-11-21",
    "--end", "2026-02-13",
    "--tp", "15",
    "--sl", "15",
    "--interaction-window", "5",
    "--include-approach-features",
    "--approach-window", "15",
    "--fold-scheme", "purged-days",
    "--fold-train-days", "40",
    "--fold-test-days", "5",
    "--fold-step-days", "5",
    "--fold-purge-days", "2",
    "--min-train-events", "30",
    "--pin-features",
    "int_time_within_2pts,int_absorption_ratio,app_avg_trade_size,app_large_trade_vol_pct,app_max_spread",
    "--iterations", "1000",
    "--depth", "6",
]

_LOG_PATH = _QL_ROOT / "W3A_WARM.log"


# ── Peak RSS via the Windows API (psutil-free, true peak) ─────────────────────
class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


# Bind with explicit arg/return types — without these, ctypes truncates the 64-bit
# HANDLE/pointer args on Win64 and the call silently fails (returns 0).
_kernel32 = ctypes.WinDLL("kernel32")
_psapi = ctypes.WinDLL("psapi")
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_ProcessMemoryCounters),
    wintypes.DWORD,
]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def _peak_working_set_gb() -> float:
    """This process's peak working set (RSS) in GB, from GetProcessMemoryInfo."""
    try:
        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(_ProcessMemoryCounters)
        ok = _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return 0.0
        return counters.PeakWorkingSetSize / 1e9
    except Exception:
        return 0.0


def _cache_path(data_dir: Path, symbol: str, date_str: str, cache_tag: str) -> Path:
    return data_dir / symbol / date_str / f"ml_utility_{cache_tag}.parquet"


def _quiet_unlink(path: Path) -> None:
    with contextlib.suppress(OSError):
        path.unlink()


def _loads(path: Path) -> bool:
    """True iff ``path`` exists and reads back as a parquet (rows >= 0)."""
    if not path.exists():
        return False
    try:
        import pandas as pd

        pd.read_parquet(path)
        return True
    except Exception:
        return False


def _seed_for_day(
    data_dir: Path,
    symbol: str,
    window_dates: list[str],
    date_str: str,
    util_cfg,
) -> tuple[float, float] | None:
    """Reproduce ``build_utility_dataset``'s ``prev_full_hl`` entering ``date_str``.

    The serial loop carries the full (high, low) of the most-recent NON-EMPTY prior
    day. We walk back from the immediate predecessor; the first day with non-empty
    bars yields the seed. ``_get_session_hl_for_date`` returns that day's own H/L
    when its bars are non-empty (independent of the seed we pass), so passing
    ``None`` is equivalent to the serial carry; an empty day returns ``None`` and we
    keep walking — exactly the serial fallback.
    """
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
        _get_session_hl_for_date,
    )

    try:
        idx = window_dates.index(date_str)
    except ValueError:
        return None
    for k in range(idx - 1, -1, -1):
        hl = _get_session_hl_for_date(data_dir, symbol, window_dates[k], util_cfg, None)
        if hl is not None:
            return hl
    return None


def warm_one_day(task: tuple) -> dict:
    """Build (and cache) one day's labeled dataset — the parallel worker.

    Top-level + primitive args so it is picklable under Windows ``spawn``.
    """
    date_str, window_dates, data_dir_str, symbol, util_kwargs, cache_tag = task
    pid = os.getpid()
    data_dir = Path(data_dir_str)
    cache_path = _cache_path(data_dir, symbol, date_str, cache_tag)
    t0 = time.perf_counter()

    try:
        import pandas as pd

        from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
        from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
            _process_single_date,
        )

        util_cfg = DashboardUtilityConfig(**util_kwargs)

        # Resumable: skip a day whose cache already exists AND loads.
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                return _result(date_str, len(df), 0.0, pid, "SKIP")
            except Exception:
                # Corrupt cache — drop it and rebuild below.
                _quiet_unlink(cache_path)

        seed = _seed_for_day(data_dir, symbol, window_dates, date_str, util_cfg)

        def _build_and_write():
            frame = _process_single_date(date_str, data_dir, symbol, util_cfg, seed)
            if not frame.empty:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(cache_path, index=False)
            return frame

        df = _build_and_write()
        rows = len(df)

        # Empty day: the serial builder writes no cache (matches; rebuilt fast later).
        if df.empty:
            return _result(date_str, 0, time.perf_counter() - t0, pid, "EMPTY")

        # Verify the written cache loads; on failure delete + rebuild once.
        if _loads(cache_path):
            return _result(date_str, rows, time.perf_counter() - t0, pid, "OK")
        _quiet_unlink(cache_path)
        df = _build_and_write()
        rows = len(df)
        status = "OK_RETRY" if (not df.empty and _loads(cache_path)) else "FAILED"
        if status == "FAILED":
            _quiet_unlink(cache_path)
        return _result(date_str, rows, time.perf_counter() - t0, pid, status)
    except Exception as exc:  # never let a worker crash the pool
        return _result(
            date_str,
            -1,
            time.perf_counter() - t0,
            pid,
            f"ERROR:{type(exc).__name__}",
            err=traceback.format_exc(limit=4),
        )


def _result(
    day: str, rows: int, seconds: float, pid: int, status: str, err: str | None = None
) -> dict:
    return {
        "day": day,
        "rows": rows,
        "seconds": seconds,
        "pid": pid,
        "status": status,
        "peak_rss_gb": _peak_working_set_gb(),
        "err": err,
    }


def _suggest_workers(peak_gb: float, budget_gb: float = 20.0) -> tuple[int, str]:
    """N = min(4, max(2, floor(budget / peak_per_worker)))."""
    if peak_gb <= 0:
        return 2, "peak unmeasured -> floor to 2"
    import math

    raw = math.floor(budget_gb / peak_gb)
    n = min(4, max(2, raw))
    math_str = (
        f"floor({budget_gb:.0f}GB / {peak_gb:.2f}GB peak) = {raw} "
        f"-> min(4, max(2, {raw})) = {n}"
    )
    return n, math_str


def _resolve_window():
    """Resolve (config, cache_tag, util_kwargs, symbol, data_dir, window_dates).

    Reuses the experiment CLI's parser + resolver so the cache tag and day list are
    identical to the serial train. Imported here (not at module load) because it
    pulls in Streamlit via ml_training_tab — main-process only, never the workers.
    """
    import run_dashboard_session_experiment as exp

    ns = exp._build_parser().parse_args(EXP_ARGV)
    config = exp._resolve_config(ns)
    cache_tag = config.dataset_config_hash()
    util_kwargs = config.dashboard_utility.model_dump()
    data_dir = ns.data_dir
    available = exp.get_available_dates(ns.symbol, data_dir)
    window_dates = exp._date_slice(available, ns.start, ns.end)
    return cache_tag, util_kwargs, ns.symbol, data_dir, window_dates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, help="process pool size N")
    parser.add_argument(
        "--days", help="comma-separated explicit in-window day subset (smoke)"
    )
    parser.add_argument(
        "--limit", type=int, help="process at most this many uncached days (smoke)"
    )
    args = parser.parse_args()

    cache_tag, util_kwargs, symbol, data_dir, window_dates = _resolve_window()

    # Build the target list.
    if args.days:
        requested = [d.strip() for d in args.days.split(",") if d.strip()]
        unknown = [d for d in requested if d not in window_dates]
        if unknown:
            raise SystemExit(f"--days not in D-036 window: {unknown}")
        targets = requested
    else:
        targets = [
            d
            for d in window_dates
            if not _loads(_cache_path(data_dir, symbol, d, cache_tag))
        ]
        if args.limit is not None:
            targets = targets[: args.limit]

    start = datetime.now(UTC)
    header = (
        f"# {start.isoformat()} W3 cache warmer | tag={cache_tag} "
        f"| symbol={symbol} | window={len(window_dates)}d "
        f"| targets={len(targets)} | workers={args.workers} | pid={os.getpid()}"
    )
    print(header, flush=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as logf:
        logf.write(header + "\n")
        logf.flush()

        if not targets:
            done = "# all in-window days already warm — nothing to do"
            print(done, flush=True)
            logf.write(done + "\n")
            return 0

        tasks = [
            (d, window_dates, str(data_dir), symbol, util_kwargs, cache_tag)
            for d in targets
        ]
        counts: dict[str, int] = {}
        peak_overall = 0.0
        peak_day = None
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(warm_one_day, t): t[0] for t in tasks}
            for fut in as_completed(futs):
                day = futs[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # broken worker / pool
                    r = {
                        "day": day,
                        "rows": -1,
                        "seconds": 0.0,
                        "pid": -1,
                        "status": f"POOL_ERROR:{type(exc).__name__}",
                        "peak_rss_gb": 0.0,
                        "err": traceback.format_exc(limit=4),
                    }
                done += 1
                counts[r["status"]] = counts.get(r["status"], 0) + 1
                if r["peak_rss_gb"] > peak_overall:
                    peak_overall = r["peak_rss_gb"]
                    peak_day = r["day"]
                ts = datetime.now(UTC).isoformat()
                line = (
                    f"{ts} day={r['day']} rows={r['rows']} "
                    f"secs={r['seconds']:.1f} pid={r['pid']} "
                    f"peak_rss_gb={r['peak_rss_gb']:.2f} "
                    f"status={r['status']} [{done}/{len(targets)}]"
                )
                print(line, flush=True)
                logf.write(line + "\n")
                if r.get("err"):
                    logf.write(r["err"].rstrip() + "\n")
                logf.flush()

        wall = (datetime.now(UTC) - start).total_seconds()
        _n_suggest, math_str = _suggest_workers(peak_overall)
        summary = (
            f"# {datetime.now(UTC).isoformat()} DONE wall={wall / 60:.1f}min "
            f"counts={counts} peak_rss_gb={peak_overall:.2f} (day={peak_day}) "
            f"| suggested_workers: {math_str}"
        )
        print(summary, flush=True)
        logf.write(summary + "\n")
        logf.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
