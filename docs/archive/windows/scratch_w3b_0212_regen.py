"""W3b 02-12 settling test + cache regen.

Builds 2026-02-12 twice through the EXACT D-036 path (process_single_date_stream):
  * prev_day_hl=None        -> expect the current STALE 4-touch cache (no pdh/pdl)
  * prev_day_hl=(seed)      -> expect 5 touches incl pdl|long @ 25058.25 (serving's survivor)
Confirms None-build reproduces the existing stale cache, and seed-build adds exactly
the pdl|long touch. Writes the corrected (seed) cache ONLY if the gate passes.

Untracked scratch. Run from QL root with PYTHONPATH=src.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from alpha_lab.agents.data_infra.ml.config import DashboardUtilityConfig
from alpha_lab.agents.data_infra.ml.engine_decision import process_single_date_stream

from w3_cache_warmer import _resolve_window, _seed_for_day  # noqa: E402

DAY = "2026-02-12"
EXPECTED_SEED = (25465.25, 25058.25)
SURVIVOR = {"level_type": "pdl", "direction": "LONG", "rep_px": 25058.25}
CACHE_REL = f"data/databento/NQ/{DAY}/ml_utility_7850272e.parquet"


def touch_set(df: pd.DataFrame):
    if df.empty:
        return []
    rows = []
    for _, r in df.sort_values("event_ts").iterrows():
        rows.append(
            (
                str(r["level_type"]),
                str(r["direction"]),
                round(float(r["representative_price"]), 2),
                str(pd.Timestamp(r["event_ts"])),
                str(r.get("label")),
            )
        )
    return rows


def main() -> int:
    cache_tag, util_kwargs, symbol, data_dir, window_dates = _resolve_window()
    util_cfg = DashboardUtilityConfig(**util_kwargs)
    print(f"cache_tag={cache_tag}  symbol={symbol}  window={len(window_dates)}d")
    assert cache_tag == "7850272e", f"unexpected cache tag {cache_tag}"

    seed = _seed_for_day(Path(data_dir), symbol, window_dates, DAY, util_cfg)
    print(f"_seed_for_day({DAY}) = {seed}   expected {EXPECTED_SEED}")
    assert seed == EXPECTED_SEED, "seed mismatch — investigate before writing"

    existing = pd.read_parquet(CACHE_REL)
    print(f"\n=== EXISTING (stale) cache: {len(existing)} touches ===")
    for t in touch_set(existing):
        print("   ", t)

    print("\n=== BUILD prev_day_hl=None (expect == stale 4-touch cache) ===")
    df_none = process_single_date_stream(DAY, Path(data_dir), symbol, util_cfg, prev_day_hl=None)
    none_set = touch_set(df_none)
    print(f"none-build: {len(df_none)} touches")
    for t in none_set:
        print("   ", t)

    print("\n=== BUILD prev_day_hl=seed (expect 5 incl pdl|long @ 25058.25) ===")
    df_seed = process_single_date_stream(DAY, Path(data_dir), symbol, util_cfg, prev_day_hl=seed)
    seed_set = touch_set(df_seed)
    print(f"seed-build: {len(df_seed)} touches")
    for t in seed_set:
        print("   ", t)

    # ── Gate checks ──────────────────────────────────────────────────────────
    none_matches_stale = touch_set(existing) == none_set
    pdl_rows = df_seed[(df_seed["level_type"] == "pdl") & (df_seed["direction"].str.upper() == "LONG")]
    has_pdl = (
        len(pdl_rows) == 1
        and round(float(pdl_rows.iloc[0]["representative_price"]), 2) == SURVIVOR["rep_px"]
    )
    seed_adds_one = (len(df_seed) == len(df_none) + 1) and set(none_set).issubset(set(seed_set))
    extra = sorted(set(seed_set) - set(none_set))

    print("\n=== GATE ===")
    print(f"  none-build reproduces stale cache : {none_matches_stale}")
    print(f"  seed-build = none + exactly one    : {seed_adds_one}")
    print(f"  the one extra touch                : {extra}")
    print(f"  extra is pdl|long @ 25058.25       : {has_pdl}")
    if pdl_rows.shape[0] == 1:
        r = pdl_rows.iloc[0]
        print(f"  pdl row: event_ts={pd.Timestamp(r['event_ts'])} dir={r['direction']} "
              f"rep_px={r['representative_price']} label={r.get('label')} session={r.get('session')}")

    passed = none_matches_stale and has_pdl and seed_adds_one
    print(f"\n  SETTLING TEST: {'PASS' if passed else 'FAIL'}")

    if passed:
        df_seed.to_parquet(CACHE_REL, index=False)
        print(f"  WROTE corrected 5-touch cache -> {CACHE_REL}")
    else:
        print("  NOT writing — gate failed; stale cache left in place (backup intact).")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
