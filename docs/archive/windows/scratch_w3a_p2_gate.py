"""W3a P2 compute stop-gate: one full pipeline day under the P3 config.

Times build_utility_dataset over 2026-02-12 (fresh — no ml_utility cache for
this config hash), projects x60 days, and verdicts against the 90-minute gate.
Supplemental: times one bare canonical-reader drain (source.events()) for the
same trading day to attribute cost reader-vs-decision-layer.

Scratch harness — untracked; re-run after a reader-cost fix to re-measure the gate.
Run: PYTHONPATH=src python scratch_w3a_p2_gate.py
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

GATE_DAY = "2026-02-12"
GATE_MINUTES = 90.0
PROJECT_DAYS = 60


def main() -> int:
    from alpha_lab.agents.data_infra.ml.config import (
        DashboardUtilityConfig,
        MLPipelineConfig,
        ModelConfig,
        session_experiment_from_preset,
    )
    from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
        build_utility_dataset,
    )

    config = MLPipelineConfig(
        training_mode="dashboard_utility",
        dashboard_utility=DashboardUtilityConfig(
            tp_points=15.0,
            sl_points=15.0,
            bar_type="147t",
            interaction_window_minutes=5,
            include_approach_features=True,
            approach_window_minutes=15,
        ),
        session_experiment=session_experiment_from_preset("all_to_ny"),
        model=ModelConfig(
            iterations=1000, depth=6, rfecv_enabled=False, loss_function="MultiClass"
        ),
        tick_size=0.25,
        instrument="NQ",
    )
    data_dir = Path("data/databento")
    tag = config.dataset_config_hash()
    cache = data_dir / "NQ" / GATE_DAY / f"ml_utility_{tag}.parquet"
    print(f"config hash: {tag}; cache pre-exists: {cache.exists()}", flush=True)

    t0 = time.perf_counter()
    df = build_utility_dataset([GATE_DAY], data_dir, config)
    elapsed = time.perf_counter() - t0
    projected_min = elapsed * PROJECT_DAYS / 60.0
    print(
        f"PIPELINE {GATE_DAY}: {len(df)} rows in {elapsed:.1f}s "
        f"({elapsed / 60.0:.1f} min)",
        flush=True,
    )
    print(
        f"PROJECTION x{PROJECT_DAYS} days: {projected_min:.0f} min "
        f"({projected_min / 60.0:.1f} h)",
        flush=True,
    )

    # Supplemental attribution: one bare canonical-reader drain (no runtime,
    # no features) over the same trading day.
    from strategy_core.data.databento_parquet import DatabentoParquetSource

    t0 = time.perf_counter()
    source = DatabentoParquetSource.for_trading_day(
        data_dir / "NQ", date.fromisoformat(GATE_DAY), requested_symbol="NQ"
    )
    n_events = 0
    for _event in source.events():
        n_events += 1
    drain = time.perf_counter() - t0
    print(
        f"READER DRAIN (one pass, {n_events} events): {drain:.1f}s — the "
        f"pipeline makes TWO passes (trades+runtime, then quotes)",
        flush=True,
    )

    verdict = "GO" if projected_min <= GATE_MINUTES else "STOP"
    print(f"GATE ({GATE_MINUTES:.0f} min): {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
