"""A1 review export and reconciled entry/cash report views."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from strategy_core.strategies.ifvg_smc.menthorq_levels import LEVEL_COLUMN_NAMES

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.menthorq_levels import load_menthorq_levels
from alpha_lab.agents.data_infra.ifvg.menthorq_reporting import (
    NOT_PRODUCED,
    build_comparison_table,
    build_funded_cash_groups,
    build_menthorq_reports,
    record_context,
    write_context_export,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from tests.agents.data_infra.ifvg.test_menthorq_levels import _sources
from tests.agents.test_ifvg_v2_reporting_manifest import _tables


def _inputs():
    fixture = Path(__file__).parent / "fixtures" / "menthorq_a1"
    provider = load_menthorq_levels(
        fixture / "eod_gamma_levels_daily_wide.csv.fixture",
        fixture / "daily_total_dealer_gamma_and_regime.csv.fixture",
    )
    section = resolve_profile_config({"section_overrides": {
        "menthorq_context_version": "menthorq_eod_v1",
    }}).section
    tables = _tables()
    for kind in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION,
                 RecordTable.EXECUTED_TRADE):
        frame = tables[kind]
        entry_timestamps = [pd.Timestamp("2026-01-13T14:00:00Z"),
                            pd.Timestamp("2026-01-14T16:00:00Z")]
        frame["envelope_ts_utc"] = entry_timestamps
        frame["entry_ts_utc"] = (
            entry_timestamps if kind is RecordTable.EXECUTED_TRADE else pd.NaT
        )
        frame["geometry_entry_bar_open_ticks"] = frame["entry_ticks"]
        frame.loc[1, "envelope_entry_session"] = "unknown"
        frame.loc[1, "entry_session"] = "unknown"
    tables[RecordTable.ENTRY_CANDIDATE]["is_warmup"] = False
    trades = tables[RecordTable.EXECUTED_TRADE]
    trades["resolution_ts_utc"] = trades["entry_ts_utc"] + pd.Timedelta(minutes=2)
    trades["envelope_ts_utc"] = trades["resolution_ts_utc"]
    return tables, provider, section


def test_export_has_every_candidate_exact_lookup_values_and_no_outcomes(tmp_path):
    tables, provider, section = _inputs()
    candidates = tables[RecordTable.ENTRY_CANDIDATE]
    path = write_context_export(
        tmp_path / "context_export.csv", candidates, tables[RecordTable.ELIGIBLE_DECISION],
        tables[RecordTable.EXECUTED_TRADE], provider=provider, section=section,
        tick_size=0.25, run_identity={"profile_hash": "test"},
    )
    metadata = json.loads(path.read_text(encoding="utf-8").splitlines()[0][2:])
    assert metadata["archival"] is False
    assert metadata["source_file_sha256"] == dict(provider.source_file_sha256)
    exported = pd.read_csv(path, comment="#")
    columns = exported.columns.tolist()
    assert columns[columns.index("nearest_support_is_gex1") + 1] == "nearest_support_universe"
    assert exported["nearest_support_universe"].eq("all_19").all()
    expected = record_context(candidates, provider=provider, section=section, tick_size=0.25)
    assert len(exported) == len(candidates)
    assert exported["candidate_id"].tolist() == candidates["candidate_id"].tolist()
    assert exported["slot_chicago"].tolist() == expected["slot_chicago"].tolist()
    assert exported["regime"].tolist() == expected["regime"].tolist()
    assert exported.loc[0, "entry_price_points"] == expected.loc[0, "entry_price_points"]
    assert json.loads(exported.loc[0, "levels"]) == expected.loc[0, "levels"]
    assert exported["decision_id"].tolist() == (
        tables[RecordTable.ELIGIBLE_DECISION]["decision_id"].tolist()
    )
    assert not {"label", "resolution", "mae_ticks", "mfe_ticks", "realized_ticks"} & set(exported)


@pytest.mark.parametrize("universe,below,above,blocked", [
    ("all_19", 99.0, 100.5, False),
    ("studied_8", 98.0, 101.0, True),
])
def test_export_forwards_universe_to_nearest_values_and_gate(
    tmp_path, universe, below, above, blocked,
):
    tables, _, _ = _inputs()
    levels = dict.fromkeys(LEVEL_COLUMN_NAMES, "0")
    levels.update({
        "HVL": "90", "1D Min": "90", "1D Max": "110", "GEX 1": "98",
        "GEX 4": "99", "Call Resistance": "101", "GEX 5": "100.5",
    })
    provider = load_menthorq_levels(*_sources(tmp_path, level_changes=levels))
    section = resolve_profile_config({"section_overrides": {
        "menthorq_context_version": "menthorq_eod_v1",
        "nearest_support_gex1_block": True, "nearest_support_universe": universe,
    }}).section
    candidates = tables[RecordTable.ENTRY_CANDIDATE].iloc[[0]].copy()
    candidates["entry_ticks"] = 400
    path = write_context_export(
        tmp_path / "context_export.csv", candidates,
        tables[RecordTable.ELIGIBLE_DECISION].iloc[:0],
        tables[RecordTable.EXECUTED_TRADE].iloc[:0], provider=provider, section=section,
        tick_size=.25, run_identity={"profile_hash": "test"},
    )
    exported = pd.read_csv(path, comment="#")
    row = exported.iloc[0]
    assert row["nearest_support_universe"] == universe
    assert row["nearest_below_points"] == below
    assert row["nearest_above_points"] == above
    assert bool(row["nearest_support_is_gex1"]) is blocked
    assert bool(row["nearest_support_gate_blocked"]) is blocked
    empty = record_context(candidates.iloc[:0], provider=provider, section=section, tick_size=.25)
    columns = empty.columns.tolist()
    assert columns[columns.index("nearest_support_is_gex1") + 1] == "nearest_support_universe"


def test_all_grouped_tables_reconcile_counts_points_and_exact_cash_cents():
    tables, provider, section = _inputs()
    report = build_menthorq_reports(tables, provider=provider, section=section,
                                   tick_size=0.25, cost_points=0.514)
    for kind, groups in report["record_counts"].items():
        assert sum(row["count"] for row in groups) == len(tables[RecordTable(kind)])
        assert any(row["entry_session"] == "unknown" for row in groups)
    groups, total = report["executed_trade_groups"], report["total"]
    for column in ("trades", "wins", "gross_points", "net_points"):
        assert sum(row[column] for row in groups) == pytest.approx(total[column])
    for dimension in ("regime", "slot"):
        for column in ("trades", "wins", "gross_points", "net_points"):
            split_total = sum(row[column] for row in report[f"comparison_by_{dimension}"])
            assert split_total == pytest.approx(report["comparison"][column])
    comparison = build_comparison_table({"baseline": report, "all_on": report})
    assert len(comparison) == 2
    assert comparison["net_cash_cents"].eq(NOT_PRODUCED).all()
    assert comparison["roll_flag_unavailable"].all()
    assert comparison.loc[0, "regime.unknown.trades"] >= 0
    # This consumer reads a saved, validated result; it never launches a campaign.
    saved = {"validation": {"passed": True}, "summaries_cents": {
        "firm": {"net_cash_earned_cents": 39_801},
    }, "tables": {"cash_ledger": [
        {"firm_key": "firm", "kind": "account_purchase", "amount_usd": 102,
         "ts_utc": "2026-01-13T14:00:00Z"},
        {"firm_key": "firm", "kind": "payout_received", "amount_usd": 500.01,
         "ts_utc": "2026-01-14T16:00:00Z"},
    ]}}
    cash = build_funded_cash_groups(saved, provider=provider)
    assert sum(row["net_cash_cents"] for row in cash["groups"]) == 39_801
    assert cash["totals_cents"] == {"firm": 39_801}


def test_record_context_uses_final_entry_envelope_for_candidates_and_decisions():
    tables, provider, section = _inputs()
    for kind in (RecordTable.ENTRY_CANDIDATE, RecordTable.ELIGIBLE_DECISION):
        records = tables[kind].copy()
        assert records["entry_ts_utc"].isna().all()
        # Unioned optional trade fields remain null on candidate/decision rows.
        records["trade_id"] = None
        context = record_context(records, provider=provider, section=section, tick_size=.25)
        assert context["availability_ts_utc"].tolist() == [
            ts.isoformat() for ts in records["envelope_ts_utc"]
        ]
        assert context.loc[0, "regime"] == provider.snapshot(
            records.loc[0, "envelope_ts_utc"].to_pydatetime()
        ).regime
    trade = tables[RecordTable.EXECUTED_TRADE].iloc[[0]].copy()
    trade["envelope_ts_utc"] = pd.Timestamp("2026-01-13T23:00:00Z")
    context = record_context(trade, provider=provider, section=section, tick_size=.25)
    assert context.loc[0, "availability_ts_utc"] == trade.iloc[0]["entry_ts_utc"].isoformat()
    assert context.loc[0, "context_available"]
    assert context.loc[0, "gate_status"] != "not_applicable_outside_hours"


@pytest.mark.parametrize("kind", [RecordTable.ENTRY_CANDIDATE, RecordTable.EXECUTED_TRADE])
@pytest.mark.parametrize("invalid", [pd.NaT, pd.Timestamp("2026-01-13T14:00:00")])
def test_record_context_rejects_missing_or_naive_authoritative_entry_time(kind, invalid):
    tables, provider, section = _inputs()
    records = tables[kind].iloc[[0]].copy()
    column = "entry_ts_utc" if kind is RecordTable.EXECUTED_TRADE else "envelope_ts_utc"
    records[column] = invalid
    with pytest.raises(ValueError, match="aware availability timestamp"):
        record_context(records, provider=provider, section=section, tick_size=.25)


def test_preparation_review_export_uses_run_id_and_preserves_immutable_membership(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.preparation import _write_preparation_context_export

    tables, provider, section = _inputs()
    immutable_roots = (tmp_path / "v2", tmp_path / "v3")
    for root in immutable_roots:
        root.mkdir()
        (root / "manifest.json").write_text('{"immutable": true}\n')
    before = {str(path): path.read_bytes() for root in immutable_roots for path in root.iterdir()}
    run_identity = {
        "v2_artifact_id": "2" * 64, "v3_artifact_id": "3" * 64,
        "v2_manifest_payload_sha256": "4" * 64, "v3_manifest_payload_sha256": "5" * 64,
        "section_config_hash": "6" * 64, "evaluation_config_hash": "7" * 64,
        "strategy_core_commit": "8" * 40, "strategy_core_source_tree_hash": "9" * 64,
    }
    report_root = tmp_path / "jobs" / "context_profile" / "reports"
    exported = _write_preparation_context_export(
        report_root=report_root, tables=tables, provider=provider, section=section,
        tick_size=.25, run_identity=run_identity, immutable_roots=immutable_roots,
    )
    assert exported == report_root / run_identity["v2_artifact_id"] / "context_export.csv"
    metadata = json.loads(exported.read_text().splitlines()[0][2:])
    assert metadata["run_identity"] == run_identity
    assert metadata["archival"] is False
    assert len(pd.read_csv(exported, comment="#")) == len(tables[RecordTable.ENTRY_CANDIDATE])
    after = {str(path): path.read_bytes() for root in immutable_roots for path in root.iterdir()}
    assert before == after


@pytest.mark.parametrize("enabled,has_provider", [(False, True), (True, False)])
def test_preparation_review_export_disabled_or_no_provider_creates_no_folder(
    tmp_path, enabled, has_provider,
):
    from alpha_lab.agents.data_infra.ifvg.preparation import _write_preparation_context_export

    tables, provider, enabled_section = _inputs()
    section = enabled_section if enabled else resolve_profile_config({}).section
    immutable_root = tmp_path / "immutable"
    report_root = immutable_root / "reports"
    assert _write_preparation_context_export(
        report_root=report_root, tables=tables, provider=provider if has_provider else None,
        section=section, tick_size=.25, run_identity={}, immutable_roots=(immutable_root,),
    ) is None
    assert not immutable_root.exists()


def test_preparation_review_export_rejects_an_immutable_destination_before_write(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.preparation import _write_preparation_context_export

    tables, provider, section = _inputs()
    immutable = tmp_path / "immutable"
    with pytest.raises(ValueError, match="outside immutable dataset roots"):
        _write_preparation_context_export(
            report_root=immutable / "reports", tables=tables, provider=provider,
            section=section, tick_size=.25, run_identity={"v2_artifact_id": "2" * 64},
            immutable_roots=(immutable,),
        )
    assert not immutable.exists()


@pytest.mark.parametrize("mode", ["enabled", "explicit", "disabled"])
def test_persisted_preparation_forwards_review_root_and_respects_job_label(
    tmp_path, monkeypatch, mode,
):
    from alpha_lab.agents.data_infra.ifvg import preparation
    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        ArtifactPreparationStatus,
    )

    expected = tmp_path / "scratch_reports" if mode == "explicit" else (
        tmp_path / "jobs" / "context_job" / "reports"
    )
    received = []

    def fake_prepare(**kwargs):
        received.append(kwargs)
        return preparation.PreparedIfvgPair(
            pair=SimpleNamespace(), access_audit={},
            preparation_state=preparation.PreparationJobState(
                profile_name=kwargs["profile_name"], status=ArtifactPreparationStatus.CONTEXT_READY,
            ),
            context_export_path=expected / ("2" * 64) / "context_export.csv"
            if mode != "disabled" else None,
        )

    monkeypatch.setattr(preparation, "prepare_ifvg_development_pair", fake_prepare)
    overrides = {"menthorq_context_version": "menthorq_eod_v1"} if mode != "disabled" else None
    prepared = preparation.prepare_ifvg_development_pair_persisted(
        repo_root=tmp_path, job_root=Path("jobs"), job_label="context_job",
        section_overrides=overrides,
        **({"report_root": expected} if mode == "explicit" else {}),
    )
    assert len(received) == 1
    if mode == "disabled":
        assert "report_root" not in received[0]
        assert prepared.context_export_path is None
    else:
        assert received[0]["report_root"] == expected
        assert prepared.context_export_path == expected / ("2" * 64) / "context_export.csv"
    assert not expected.exists()
