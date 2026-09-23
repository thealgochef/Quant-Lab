"""Saved-study exports retain the original execution and metric contracts."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.charter import CostPolicy
from alpha_lab.agents.data_infra.ifvg.search.costed_exports import generate_costed_exports
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import compute_strategy_metrics
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame
from tests.agents.test_ifvg_scheduled_exit_accounting import scheduled_trade


def _metrics(frame):
    return compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: frame},
        cost_points=0.514,
        evaluation_config_hash="1" * 64,
    )


def _export(tmp_path, frame, metrics=None, dates=None, *, include_raw_evidence=True):
    return generate_costed_exports(
        executed_trades=frame,
        stored_metrics=metrics or _metrics(frame),
        cost_policy=CostPolicy(),
        core_replay_id="2" * 64,
        costed_evaluation_id="3" * 64,
        evaluation_dates=dates or sorted(frame["trading_day"].astype(str).unique()),
        out_dir=tmp_path,
        include_raw_evidence=include_raw_evidence,
    )


def test_complete_columns_actual_costs_initial_loss_and_warmup(tmp_path):
    frame = make_resolved_trades_frame(
        ("2026-01-12", "2026-01-13", "2026-01-14"),
        trades_per_day=1,
        loss_every=2,
    )
    frame["is_warmup"] = [True, False, False]
    frame["evidence_extra"] = ["excluded warmup", "retained loss", "retained win"]
    frame["direction"] = "long"
    receipt = _export(
        tmp_path,
        frame,
        dates=["2026-01-13", "2026-01-14", "2026-01-15"],
    )
    output = pd.read_parquet(tmp_path / "costed_trades_post_warmup.parquet")
    pd.testing.assert_frame_equal(output[frame.columns], frame.iloc[1:].reset_index(drop=True))
    assert receipt["all_checks_passed"]
    assert receipt["excluded_warmup_rows"] == 1
    assert receipt["post_warmup_resolved_execution_rows"] == 2
    assert output["derived_cost_r"].tolist() == pytest.approx([0.0514, 0.0514])
    assert output["derived_net_usd"].tolist() == pytest.approx([-210.28, 189.72])
    equity = pd.read_parquet(tmp_path / "closed_trade_equity.parquet")
    assert equity.loc[0, "closed_equity_before_r"] == 0
    assert equity.loc[0, "drawdown_r"] == pytest.approx(1.0514)
    daily = json.loads((tmp_path / "evaluation_day_breakdown.json").read_text())
    assert daily[-1]["executed_trades"] == 0
    assert daily[-1]["total_net_r"] == 0
    records = json.loads((tmp_path / "costed_trades_post_warmup.json").read_text())
    assert records[0]["evidence_extra"] == "retained loss"


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 8])
def test_empty_and_few_days_exact_thirds(tmp_path, count):
    dates = [f"2026-01-{13 + index:02d}" for index in range(max(count, 1))]
    frame = make_resolved_trades_frame(tuple(dates), trades_per_day=1)
    if count == 0:
        frame = frame.head(0)
    receipt = _export(tmp_path, frame, dates=dates)
    blocks = json.loads((tmp_path / "chronological_thirds_breakdown.json").read_text())
    assert len(blocks) == 3
    assert sum(row["executed_trades"] for row in blocks) == count
    assert receipt["time_block_sign_consistency"] == _metrics(frame).time_block_sign_consistency
    assert sum(row["included_in_sign_consistency"] for row in blocks) == min(count, 3)


def test_null_session_preserves_recorded_grouping_and_priority(tmp_path):
    frame = make_resolved_trades_frame(("2026-01-13", "2026-01-14"), trades_per_day=2)
    frame["entry_session"] = ["london", None, "london", None]
    frame["session_doc"] = "ny"
    receipt = _export(tmp_path, frame)
    rows = json.loads((tmp_path / "session_breakdown.json").read_text())
    assert {row["session_group"] for row in rows} == {"london", "None"}
    assert sum(row["null_source_rows"] for row in rows) == 2
    assert all(row["session_source_column"] == "entry_session" for row in rows)
    assert receipt["session_stability_score"] == _metrics(frame).session_stability_score


@pytest.mark.parametrize("fault", ["open", "warmup_null", "outside", "duplicate", "bad_metrics"])
def test_invalid_evidence_fails_before_export(tmp_path, fault):
    frame = make_resolved_trades_frame(("2026-01-13", "2026-01-14"), trades_per_day=1)
    frame["is_warmup"] = False
    metrics = _metrics(frame)
    dates = ["2026-01-13", "2026-01-14"]
    if fault == "open":
        frame.loc[0, "status"] = "open_unresolved"
    elif fault == "warmup_null":
        frame["is_warmup"] = frame["is_warmup"].astype(object)
        frame.loc[0, "is_warmup"] = None
    elif fault == "outside":
        dates = ["2026-01-13"]
    elif fault == "duplicate":
        frame.loc[1, "trade_id"] = frame.loc[0, "trade_id"]
    elif fault == "bad_metrics":
        metrics = metrics.model_copy(update={"net_expectancy_r": 900.0})
    with pytest.raises(ValueError):
        _export(tmp_path, frame, metrics, dates)
    assert not list(tmp_path.iterdir())


def test_full_equity_vector_is_verified(tmp_path):
    frame = make_resolved_trades_frame(("2026-01-13", "2026-01-14"), trades_per_day=1)
    metrics = _metrics(frame).model_dump(mode="json")
    stats = dict(metrics["trade_stats"])
    stats["equity"]["r"]["equity"][0] = 1234
    metrics["trade_stats"] = stats
    with pytest.raises(ValueError, match="full r equity vector"):
        _export(tmp_path, frame, metrics)


def test_saved_uncertainty_is_exported_without_resampling(tmp_path):
    frame = make_resolved_trades_frame(("2026-01-13", "2026-01-14"), trades_per_day=1)
    metrics = _metrics(frame)
    _export(tmp_path, frame, metrics)
    saved = json.loads((tmp_path / "original_saved_uncertainty.json").read_text())
    assert saved["net_expectancy_bootstrap_ci95"] == list(metrics.net_expectancy_bootstrap_ci95)
    assert saved["cluster_bootstrap_ci95"] == dict(metrics.trade_stats)["cluster_bootstrap_ci95"]


def test_compact_default_keeps_auditable_economics_without_raw_duplicates(tmp_path):
    frame = make_resolved_trades_frame(("2026-01-13", "2026-01-14"), trades_per_day=1)
    frame["diagnostic_dump"] = "not needed for a trade audit"
    receipt = generate_costed_exports(
        executed_trades=frame,
        stored_metrics=_metrics(frame),
        cost_policy=CostPolicy(),
        core_replay_id="2" * 64,
        costed_evaluation_id="3" * 64,
        evaluation_dates=["2026-01-13", "2026-01-14"],
        out_dir=tmp_path,
    )
    assert receipt["delivery_mode"] == "compact_audit"
    assert receipt["all_checks_passed"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "costed_trades_post_warmup.csv",
        "closed_trade_equity.csv",
        "session_breakdown.json",
        "chronological_thirds_breakdown.json",
        "evaluation_day_breakdown.json",
        "original_saved_uncertainty.json",
        "costed_export_receipt.json",
    }
    trades = pd.read_csv(tmp_path / "costed_trades_post_warmup.csv")
    assert "diagnostic_dump" not in trades
    assert trades["trade_id"].tolist() == frame["trade_id"].tolist()
    assert trades["derived_net_r"].sum() == pytest.approx(receipt["summary"]["total_net_r"])
    equity = pd.read_csv(tmp_path / "closed_trade_equity.csv")
    assert equity["closed_equity_after_usd"].iloc[-1] == pytest.approx(
        receipt["summary"]["total_net_usd"]
    )


def test_compact_export_refuses_to_mix_with_preserved_raw_evidence(tmp_path):
    frame = make_resolved_trades_frame(("2026-01-13",), trades_per_day=1)
    prior = tmp_path / "preserved_raw.json"
    prior.write_text("historical evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        _export(tmp_path, frame, include_raw_evidence=False)
    assert prior.read_text(encoding="utf-8") == "historical evidence"
    assert list(tmp_path.iterdir()) == [prior]


def test_raw_evidence_cannot_be_written_into_curated_reports(tmp_path):
    repository = tmp_path / "checkout"
    frame = make_resolved_trades_frame(("2026-01-13",), trades_per_day=1)
    with pytest.raises(ValueError, match="outside the repository"):
        generate_costed_exports(
            executed_trades=frame,
            stored_metrics=_metrics(frame),
            cost_policy=CostPolicy(),
            core_replay_id="2" * 64,
            costed_evaluation_id="3" * 64,
            evaluation_dates=["2026-01-13"],
            out_dir=repository / "reports" / "raw",
            include_raw_evidence=True,
            repo_root=repository,
        )
    assert not repository.exists()


def test_compact_scheduled_exit_preserves_actual_price_deadline_and_partial_economics(tmp_path):
    frame = scheduled_trade(realized=12)
    receipt = _export(tmp_path, frame, include_raw_evidence=False)
    exported = pd.read_csv(tmp_path / "costed_trades_post_warmup.csv")
    assert exported["exit_ticks"].tolist() == frame["exit_ticks"].tolist()
    assert (
        exported["scheduled_exit_schedule_id"].tolist()
        == frame["scheduled_exit_schedule_id"].tolist()
    )
    assert (
        exported["scheduled_exit_deadline_ts_utc"].tolist()
        == frame["scheduled_exit_deadline_ts_utc"].tolist()
    )
    assert exported["derived_net_r"].iloc[0] == pytest.approx((3.0 - 0.514) / 10.0)
    assert receipt["summary"]["scheduled_close_resolutions"] == 1
