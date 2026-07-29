"""Parallel Phase-A warmer for the IFVG day artifacts (IFVG window).

Warms ``ifvg_tbars_<atag>`` / ``ifvg_levels_<atag>`` for every store day —
profile-INDEPENDENT, so this runs once per platform/scheme identity and every
capture-profile iteration reuses it.

Spawn-safe (top-level worker), resumable (existing artifact pairs are
skipped), worker-recycled (``max_tasks_per_child``) for the known per-worker
RSS creep. Seeds come from the prior day's artifact when it already exists
(fast path) or the SC store walks (fallback — workers racing ahead of their
neighbors pay one extra prior-day drain). Chain consistency is re-verified by
the Phase-C capture driver, which rebuilds any day whose stamped seeds
disagree with its predecessor's outputs.

Usage:
    PYTHONPATH=src python scripts/ifvg_artifact_warmer.py [--workers 4]
        [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit N] [--symbol NQ]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def available_store_days(symbol: str, data_dir: Path) -> list[str]:
    """The RESEARCH WINDOW: mbp1-era days only (the ratified INGEST-curated
    store; the pre-2025 MBP-10 era carries the known multi-instrument
    contamination caveats and stays out of the IFVG chain)."""
    symbol_dir = data_dir / symbol
    if not symbol_dir.is_dir():
        return []
    days = []
    for entry in sorted(symbol_dir.iterdir()):
        if not entry.is_dir():
            continue
        try:
            day = date.fromisoformat(entry.name)
        except ValueError:
            continue
        if day.isoformat() != entry.name:
            continue
        if (entry / "mbp1.parquet").exists():
            days.append(entry.name)
    return days


def warm_one(date_str: str) -> tuple[str, str, float]:
    """Top-level spawn-safe worker: (date, status, seconds)."""
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.day_artifacts import (
        build_day_artifacts,
        seeds_for_day,
        write_day_artifacts,
    )

    cfg = IfvgCaptureConfig()
    t0 = time.time()
    if cfg.bars_path(date_str).exists() and cfg.levels_path(date_str).exists():
        return date_str, "cached", 0.0
    try:
        seeds = seeds_for_day(date_str, cfg)
        artifacts = build_day_artifacts(date_str, cfg, seeds)
        write_day_artifacts(artifacts, cfg)
    except Exception as exc:  # noqa: BLE001 — a bad day must not kill the warm
        return date_str, f"ERROR: {exc!r}", time.time() - t0
    status = "ok" if artifacts.bars else "empty"
    if artifacts.reader_warnings:
        status += f" ({len(artifacts.reader_warnings)} reader warnings)"
    return date_str, status, time.time() - t0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument("--data-dir", default="data/databento")
    args = parser.parse_args()

    days = available_store_days(args.symbol, Path(args.data_dir))
    if args.start:
        days = [d for d in days if d >= args.start]
    if args.end:
        days = [d for d in days if d <= args.end]
    if args.limit:
        days = days[: args.limit]
    if not days:
        print("no store days matched")
        return 1
    # Skip already-cached days in the PARENT: a burst of instantly-returning
    # worker tasks churns max_tasks_per_child recycling into a Windows-spawn
    # wedge (observed 2026-07-28: parent alive, zero workers, log frozen at the
    # cached prefix). Workers only ever receive real builds.
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig

    _cfg = IfvgCaptureConfig()
    cached = [d for d in days if _cfg.bars_path(d).exists() and _cfg.levels_path(d).exists()]
    days = [d for d in days if d not in set(cached)]
    print(f"{len(cached)} days already cached; warming {len(days)} with {args.workers} workers")
    if not days:
        print("DONE 0 days, 0 errors, 0s total (all cached)")
        return 0

    # RSS bounding via CHUNKED POOLS (fresh executor per batch, clean teardown
    # between batches) — NOT max_tasks_per_child: in-flight worker replacement
    # wedged twice on this py3.13/Windows setup at exactly the recycle boundary
    # (parent alive, zero workers, log frozen at workers*max_tasks builds).
    done = errors = 0
    t0 = time.time()
    chunk_size = max(1, args.workers * 6)
    for start in range(0, len(days), chunk_size):
        chunk = days[start : start + chunk_size]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(warm_one, d): d for d in chunk}
            for future in as_completed(futures):
                date_str, status, seconds = future.result()
                done += 1
                if status.startswith("ERROR"):
                    errors += 1
                print(
                    f"[{done}/{len(days)}] {date_str}: {status} ({seconds:.1f}s, "
                    f"elapsed {time.time()-t0:.0f}s)",
                    flush=True,
                )
    print(f"DONE {done} days, {errors} errors, {time.time()-t0:.0f}s total")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
