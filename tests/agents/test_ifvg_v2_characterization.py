"""Characterization gates for the Quant-Lab side of the IFVG v2 repair."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from strategy_core.candles._ids import make_bar_id
from strategy_core.types import Bar, BarKind, CloseReason, Direction

from alpha_lab.agents.data_infra.ifvg.contracts import (
    IFVG_DATASET_SCHEMA_VERSION,
    RecordTable,
)
from alpha_lab.agents.data_infra.ifvg.data_access import (
    EXPLORATION_DATE_ALLOWLIST,
    DataAccessAudit,
    ExplorationDataPolicy,
)
from alpha_lab.agents.data_infra.ifvg.entry_dataset import build_candidate_label_rows
from alpha_lab.agents.data_infra.ifvg.geometry import attach_candidate_geometry
from alpha_lab.agents.data_infra.ifvg.migration import migrate_v1_experiment_config
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.trade_stats import compute_trade_stats

_DAY = date(2026, 1, 13)
_T0 = datetime(2026, 1, 12, 23, 0, tzinfo=UTC)


def _bar(index: int, o: int, h: int, low: int, c: int) -> Bar:
    logical_open = _T0 + timedelta(minutes=index)
    return Bar(
        timeframe_ticks=60,
        trading_day=_DAY,
        bar_index=index,
        bar_id=make_bar_id(60, _DAY, index, BarKind.TIME),
        open_ts_utc=logical_open,
        close_ts_utc=logical_open + timedelta(seconds=59),
        open_ticks=o,
        high_ticks=h,
        low_ticks=low,
        close_ticks=c,
        volume=1,
        trade_count=1,
        is_complete=True,
        is_partial=False,
        close_reason=CloseReason.COMPLETE,
        kind=BarKind.TIME,
        logical_open_ts_utc=logical_open,
        logical_close_ts_utc=logical_open + timedelta(minutes=1),
    )


def test_v2_dataset_contract_has_typed_tables() -> None:
    assert IFVG_DATASET_SCHEMA_VERSION == 2
    assert {
        RecordTable.SETUP_LIFECYCLE,
        RecordTable.ENTRY_CANDIDATE,
        RecordTable.CANDIDATE_LABEL,
        RecordTable.ELIGIBLE_DECISION,
        RecordTable.EXECUTED_TRADE,
        RecordTable.GEOMETRY_DOSSIER,
        RecordTable.QUARANTINE,
    } <= set(RecordTable)


def test_trade_stats_fail_closed_on_candidate_rows() -> None:
    candidate = pd.DataFrame(
        {
            "record_table": ["entry_candidate"],
            "candidate_id": ["c-1"],
            "trading_day": ["2026-01-13"],
            "_label": ["win"],
            "_realized_pts": [5.0],
        }
    )
    with pytest.raises(ValueError, match="executed_trade"):
        compute_trade_stats(
            candidate,
            cost_points=0.0,
            evaluation_config_hash="0" * 64,
        )


def test_candidate_geometry_never_falls_back_to_setup_id() -> None:
    candidates = pd.DataFrame(
        {
            "candidate_id": ["candidate-a", "candidate-b"],
            "setup_id": ["same-setup", "same-setup"],
        }
    )
    dossiers = pd.DataFrame(
        {
            "candidate_id": ["candidate-a", "candidate-b"],
            "setup_id": ["same-setup", "same-setup"],
            "entry_candle_id": ["bar-a", "bar-b"],
        }
    )
    joined = attach_candidate_geometry(candidates, dossiers)
    assert joined.set_index("candidate_id")["entry_candle_id"].to_dict() == {
        "candidate-a": "bar-a",
        "candidate-b": "bar-b",
    }

    with pytest.raises(ValueError, match="candidate_id"):
        attach_candidate_geometry(
            candidates,
            dossiers.drop(columns=["candidate_id"]),
        )


def test_r_families_get_their_own_path_metrics() -> None:
    bars = (
        _bar(1, 100, 110, 96, 108),  # 1R target reached, 2R not reached
        _bar(2, 108, 121, 105, 120),  # 2R target reached later
    )
    labels = build_candidate_label_rows(
        candidate_id="candidate-1",
        entry_ticks=100,
        stop_ticks=90,
        direction=Direction.LONG,
        entry_bar=_bar(0, 100, 105, 95, 100),
        forward_bars_1m=bars,
        tick_size=0.25,
        r_multiples=(1.0, 2.0),
    )
    by_r = {row["r_multiple"]: row for row in labels}
    assert by_r[1.0]["bars_after_entry_to_resolution"] == 1
    assert by_r[2.0]["bars_after_entry_to_resolution"] == 2
    assert by_r[1.0]["mfe_r"] != by_r[2.0]["mfe_r"]


def test_forbidden_date_is_denied_before_path_factory_or_io() -> None:
    calls: list[str] = []
    audit = DataAccessAudit()
    policy = ExplorationDataPolicy(audit=audit)

    def path_factory(day: str) -> Path:
        calls.append(day)
        return Path("data") / day / "mbp1.parquet"

    with pytest.raises(PermissionError):
        policy.resolve_source_path("2026-06-12", path_factory)
    assert calls == []
    assert audit.path_constructions == 0
    assert audit.metadata_accesses == 0
    assert audit.file_opens == 0
    assert audit.denied_dates == {"2026-06-12": 1}

    allowed = "2026-01-13"
    assert allowed in EXPLORATION_DATE_ALLOWLIST
    assert policy.resolve_source_path(allowed, path_factory) == (
        Path("data") / allowed / "mbp1.parquet"
    )
    assert calls == [allowed]


def test_legacy_trade_cap_never_becomes_execution_cap() -> None:
    migrated = migrate_v1_experiment_config(
        {
            "doc_defaults": {"apply": False},
            "filters": {"max_trades_per_day": 2},
        }
    )
    assert migrated["qualification_mode"] == "broad_capture"
    assert migrated["legacy_candidate_row_limit"] == 2
    assert migrated["max_executed_trades_per_day"] is None
    assert migrated["execution_enabled"] is False


def test_inactive_ui_values_do_not_change_section_hash() -> None:
    raw_a = {
        "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
        "max_candidates_per_day": 10,
        "retest_fraction": 0.25,
    }
    raw_b = {**raw_a, "retest_fraction": 0.75}
    a = resolve_profile_config(raw_a)
    b = resolve_profile_config(raw_b)
    assert a.section_config_hash == b.section_config_hash
    assert a.evaluation_config_hash == b.evaluation_config_hash
    assert a.diagnostics["retest_fraction"]["status"] == "diagnostic_only"


def test_executed_table_requires_unique_trade_ids() -> None:
    duplicate = pd.DataFrame(
        {
            "record_table": [RecordTable.EXECUTED_TRADE.value] * 2,
            "trade_schema_version": [2, 2],
            "trade_id": ["trade-1", "trade-1"],
            "decision_id": ["decision-1", "decision-1"],
            "status": ["resolved", "resolved"],
            "resolution": ["target", "target"],
            "trading_day": ["2026-01-13", "2026-01-13"],
            "direction": ["LONG", "LONG"],
            "_label": ["win", "win"],
            "_realized_pts": [1.0, 1.0],
            "_net_r": [1.0, 1.0],
            "_risk_points": [1.0, 1.0],
            "_mfe_r": [1.0, 1.0],
            "_mae_r": [0.0, 0.0],
            "_bars_to_res": [1, 1],
            "entry_ts_utc": [_T0, _T0],
        }
    )
    with pytest.raises(ValueError, match="trade_id"):
        compute_trade_stats(
            duplicate,
            cost_points=0.0,
            evaluation_config_hash="1" * 64,
        )
