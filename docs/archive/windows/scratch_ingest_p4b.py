"""INGEST P4b FRESH-DAY BUILD: one post-gap day through the QL per-day D-036 build.

Reuses ``w3_cache_warmer.warm_one_day`` VERBATIM (the seed-replicating worker) —
only the window is widened so the fresh day and its seed predecessors are in
scope. The config resolves through the same parser/resolver as the serial
train, so the cache tag is the D-036 ``ml_utility_7850272e``.

Run:  python scratch_ingest_p4b.py [2026-03-02]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_QL_ROOT = Path(__file__).resolve().parent
for _p in (str(_QL_ROOT / "src"), str(_QL_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import w3_cache_warmer as warmer  # noqa: E402


def main(day: str) -> int:
    import pandas as pd

    import run_dashboard_session_experiment as exp

    ns = exp._build_parser().parse_args(warmer.EXP_ARGV)
    config = exp._resolve_config(ns)
    cache_tag = config.dataset_config_hash()
    util_kwargs = config.dashboard_utility.model_dump()
    available = exp.get_available_dates(ns.symbol, ns.data_dir)
    window_dates = exp._date_slice(available, "2025-11-21", day)
    print(f"cache_tag={cache_tag}  (D-036 expectation: 7850272e)")
    print(f"window: {window_dates[0]}..{window_dates[-1]} ({len(window_dates)} days)")
    if day not in window_dates:
        print(f"FATAL: {day} not discovered in the store")
        return 2

    seed = warmer._seed_for_day(
        ns.data_dir, ns.symbol, window_dates,
        day, config.dashboard_utility,
    )
    print(f"prev_full_hl seed entering {day}: {seed}")

    t0 = time.perf_counter()
    result = warmer.warm_one_day(
        (day, window_dates, str(ns.data_dir), ns.symbol, util_kwargs, cache_tag)
    )
    print(json.dumps(result, indent=1))

    cache_path = warmer._cache_path(ns.data_dir, ns.symbol, day, cache_tag)
    if not cache_path.exists():
        print("FATAL: no cache written")
        return 1
    df = pd.read_parquet(cache_path)
    print(f"cache: {cache_path}")
    print(f"rows={len(df)}  wall={time.perf_counter() - t0:.0f}s")
    with pd.option_context("display.max_columns", 20, "display.width", 200):
        for col in ("level_kind", "session", "outcome", "label"):
            if col in df.columns:
                print(f"{col} counts:\n{df[col].value_counts().to_string()}")
    n_null = int(df.isna().sum().sum())
    print(f"total null cells: {n_null}")
    return 0 if result["status"] in {"OK", "SKIP"} and len(df) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "2026-03-02"))
