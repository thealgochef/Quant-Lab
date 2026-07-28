"""CACHE_SHAPE_RECON PART C step 0 — config resolution + day selection (READ-ONLY).

Resolves the exact D-036 config the way scripts/w3_cache_warmer.py does
(run_dashboard_session_experiment._build_parser/_resolve_config over the ratified
EXP_ARGV), computes the DAY_MBP10 selection from file METADATA only (os.path.getsize,
Path.exists — no data file is opened), and writes a handoff JSON consumed by
scratch_cache_shape_timing.py.

Writes ONLY: C:\\Users\\gonza\\Documents\\Claude-Quant-Lab\\_scratch_timing\\handoff.json
Never touches the databento store tree.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

_QL_ROOT = Path(__file__).resolve().parent
for _p in (str(_QL_ROOT / "src"), str(_QL_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCRATCH = Path(r"C:\Users\gonza\Documents\Claude-Quant-Lab\_scratch_timing")
_SCRATCH.mkdir(exist_ok=True)

# Exact D-036 launch argv, copied VERBATIM from scripts/w3_cache_warmer.py EXP_ARGV.
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

SEALED_START, SEALED_END = "2026-06-12", "2026-07-10"
WIN_START, WIN_END = "2025-11-21", "2026-02-13"
DAY_MBP1 = "2026-03-02"


def main() -> int:
    import strategy_core

    sc_file = strategy_core.__file__
    dist_info = Path(sc_file).parent.parent / "strategy_core-0.1.0.dist-info" / "direct_url.json"
    direct_url = json.loads(dist_info.read_text())

    import run_dashboard_session_experiment as exp

    ns = exp._build_parser().parse_args(EXP_ARGV)
    config = exp._resolve_config(ns)
    cache_tag = config.dataset_config_hash()
    util_kwargs = config.dashboard_utility.model_dump()
    data_dir = Path(ns.data_dir)
    symbol = ns.symbol

    available = exp.get_available_dates(symbol, data_dir)
    window_d036 = exp._date_slice(available, ns.start, ns.end)

    # ── DAY_MBP10 selection: metadata only ────────────────────────────────────
    # Interpretation: median over ALL window days (2025-11-21..2026-02-13 incl.)
    # that HAVE an mbp10.parquet — including thin Sunday/holiday days, since the
    # rule keys on file presence, not day quality. statistics.median (n odd ->
    # the middle element; n even would be the mean of the two central values).
    # Candidates are then restricted to days whose dir ALSO carries
    # ml_utility_<cache_tag>.parquet; pick min |size - median| (ties: none here).
    sizes: dict[str, int] = {}
    cache_present: dict[str, bool] = {}
    for d in available:
        if WIN_START <= d <= WIN_END:
            p = data_dir / symbol / d / "mbp10.parquet"
            if p.exists():
                sizes[d] = os.path.getsize(p)
                cache_present[d] = (data_dir / symbol / d / f"ml_utility_{cache_tag}.parquet").exists()
    median = statistics.median(sizes.values())
    candidates = sorted(
        (abs(s - median), d, s) for d, s in sizes.items() if cache_present[d]
    )
    delta, day_mbp10, size_mbp10 = candidates[0]

    # ── DAY_MBP1 verification: listing only ───────────────────────────────────
    mbp1_path = data_dir / symbol / DAY_MBP1 / "mbp1.parquet"
    mbp1_cache = data_dir / symbol / DAY_MBP1 / f"ml_utility_{cache_tag}.parquet"
    assert mbp1_path.exists(), f"missing {mbp1_path}"

    # Seed-walk window per day:
    #  * DAY_MBP10 sits inside the D-036 window -> the warmer's exact window list.
    #  * DAY_MBP1 (2026-03-02) is OUTSIDE the D-036 window (no ratified window
    #    contains it); its serial-equivalent seed is the most recent NON-EMPTY
    #    prior store day -> walk over ALL available store days <= DAY_MBP1.
    window_mbp1 = [d for d in available if d <= DAY_MBP1]

    sealed_touch = [d for d in (day_mbp10, DAY_MBP1) if SEALED_START <= d <= SEALED_END]
    assert not sealed_touch, f"measurement day inside sealed range: {sealed_touch}"

    handoff = {
        "python": sys.version,
        "strategy_core_file": sc_file,
        "strategy_core_direct_url": direct_url,
        "exp_argv": EXP_ARGV,
        "symbol": symbol,
        "data_dir": str(data_dir),
        "cache_tag": cache_tag,
        "util_kwargs": util_kwargs,
        "day_selection": {
            "window": [WIN_START, WIN_END],
            "n_window_days_with_mbp10": len(sizes),
            "n_candidates_with_cache": sum(1 for v in cache_present.values() if v),
            "median_bytes": median,
            "chosen_day": day_mbp10,
            "chosen_size_bytes": size_mbp10,
            "abs_delta_from_median_bytes": delta,
            "cache_present": cache_present[day_mbp10],
            "runner_up": {"day": candidates[1][1], "size": candidates[1][2], "delta": candidates[1][0]},
        },
        "day_mbp1": {
            "day": DAY_MBP1,
            "mbp1_parquet_bytes": os.path.getsize(mbp1_path),
            "cache_present": mbp1_cache.exists(),
            "cache_bytes_listed_not_opened": os.path.getsize(mbp1_cache) if mbp1_cache.exists() else None,
        },
        "window_dates": {
            day_mbp10: window_d036,
            DAY_MBP1: window_mbp1,
        },
    }
    out = _SCRATCH / "handoff.json"
    out.write_text(json.dumps(handoff, indent=2, default=str), encoding="utf-8")
    print(f"cache_tag={cache_tag}")
    print(f"strategy_core={sc_file}")
    print(f"commit={direct_url['vcs_info']['commit_id']}")
    print(f"DAY_MBP10={day_mbp10} size={size_mbp10} median={median} delta={delta} cache={cache_present[day_mbp10]}")
    print(f"DAY_MBP1={DAY_MBP1} mbp1_bytes={handoff['day_mbp1']['mbp1_parquet_bytes']} cache={handoff['day_mbp1']['cache_present']}")
    print(f"util_kwargs={util_kwargs}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
