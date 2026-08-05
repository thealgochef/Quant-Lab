"""V2 report separation, execution validation, and immutable persistence."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import (
    RecordTable,
    stamp_table_contract,
    validate_foreign_keys,
)
from alpha_lab.agents.data_infra.ifvg.manifest import (
    DatasetIdentity,
    RepositoryState,
    dataset_id_for,
    read_repository_state,
    save_v2_dataset_immutable,
)
from alpha_lab.agents.data_infra.ifvg.reporting import (
    build_candidate_report,
    build_decision_report,
    build_executed_trade_report,
    build_invariant_audit,
)
from alpha_lab.agents.data_infra.ifvg.trade_stats import compute_trade_stats

_PROFILE_HASH = "a" * 64
_EVALUATION_HASH = "b" * 64


def _envelope(setup: str, day: str) -> dict:
    return {
        "envelope_schema_version": 2,
        "envelope_strategy_id": "ifvg_smc",
        "envelope_strategy_version": "2",
        "envelope_profile_hash": _PROFILE_HASH,
        "envelope_profile_name": "ifvg_v2_doc_default_fresh_static_1r",
        "envelope_qualification_mode": "doc_default",
        "envelope_section_config_hash": _PROFILE_HASH,
        "envelope_entry_family": "fresh_fvg_continuation",
        "envelope_label_family": "static_1r",
        "envelope_entry_session": "ny",
        "envelope_anchor_policy": "trading_day_18et_elapsed_v1",
        "envelope_resolver_policy": "next_1m_bar_stop_first_v1",
        "envelope_causality_parent": "confirmed_after",
        "envelope_causality_opposing": "confirmed_after",
        "envelope_causality_entry": "confirmed_after",
        "envelope_timeout_policy": "parent_own_tf_40|post_inversion_1m_80",
        "envelope_setup_id": setup,
        "envelope_trading_day": day,
        "evaluation_config_hash": _EVALUATION_HASH,
    }


def _tables() -> dict[RecordTable, pd.DataFrame]:
    candidates = []
    decisions = []
    trades = []
    geometry = []
    labels = []
    for index, (day, resolution) in enumerate(
        (("2026-01-13", "target"), ("2026-01-14", "stop"))
    ):
        setup_id = f"setup-{index}"
        candidate_id = f"candidate-{index}"
        decision_id = f"decision-{index}"
        trade_id = f"trade-{index}"
        candidates.append(
            {
                **_envelope(setup_id, day),
                "candidate_id": candidate_id,
                "direction": "LONG",
                "entry_family": "fresh_fvg_continuation",
                "trigger_cursor": f"entry-{index}",
                "entry_ticks": 100,
                "proposed_stop_ticks": 90,
                "risk_ticks": 10,
                "proposed_target_ticks": 110,
                "block_reasons": "[]",
            }
        )
        decisions.append(
            {
                **_envelope(setup_id, day),
                "decision_id": decision_id,
                "candidate_id": candidate_id,
                "direction": "LONG",
                "entry_family": "fresh_fvg_continuation",
                "entry_cursor": f"entry-{index}",
                "entry_ticks": 100,
                "stop_ticks": 90,
                "risk_ticks": 10,
                "target_ticks": 110,
                "passed_guards": '["causality", "session", "positive_risk"]',
            }
        )
        trades.append(
            {
                **_envelope(setup_id, day),
                "trade_id": trade_id,
                "decision_id": decision_id,
                "candidate_id": candidate_id,
                "direction": "LONG",
                "status": "resolved",
                "resolution": resolution,
                "entry_family": "fresh_fvg_continuation",
                "entry_cursor": f"entry-{index}",
                "resolution_cursor": f"resolution-{index}",
                "entry_ts_utc": pd.Timestamp(f"{day} 14:00:00Z"),
                "resolution_ts_utc": pd.Timestamp(f"{day} 14:02:00Z"),
                "entry_ticks": 100,
                "stop_ticks": 90,
                "target_ticks": 110,
                "risk_ticks": 10,
                "bars_after_entry_to_resolution": 2,
                "mfe_ticks": 10 if resolution == "target" else 4,
                "mae_ticks": 2 if resolution == "target" else 10,
                "realized_ticks": 10 if resolution == "target" else -10,
            }
        )
        geometry.append(
            {
                **_envelope(setup_id, day),
                "candidate_id": candidate_id,
                "decision_id": decision_id,
                "trade_id": trade_id,
                "geometry_inversion_bar_cursor": f"inversion-{index}",
                "geometry_entry_bar_cursor": f"entry-{index}",
            }
        )
        labels.append(
            {
                "candidate_label_id": f"label-{index}",
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "label_family": "static_r_1_next_bar_stop_first_v1",
                "r_multiple": 1.0,
                "label": "win" if resolution == "target" else "loss",
                "bars_after_entry_to_resolution": 2,
                "mfe_r": 1.0,
                "mae_r": 0.2,
                "censored": False,
                "section_config_hash": _PROFILE_HASH,
                "evaluation_config_hash": _EVALUATION_HASH,
                "qualification_mode": "doc_default",
                "entry_family": "fresh_fvg_continuation",
                "anchor_policy": "trading_day_18et_elapsed_v1",
                "resolver_policy": "next_1m_bar_stop_first_v1",
            }
        )
    return {
        RecordTable.SETUP_LIFECYCLE: stamp_table_contract(
            RecordTable.SETUP_LIFECYCLE, pd.DataFrame()
        ),
        RecordTable.ENTRY_CANDIDATE: stamp_table_contract(
            RecordTable.ENTRY_CANDIDATE, pd.DataFrame(candidates)
        ),
        RecordTable.CANDIDATE_LABEL: stamp_table_contract(
            RecordTable.CANDIDATE_LABEL, pd.DataFrame(labels)
        ),
        RecordTable.ELIGIBLE_DECISION: stamp_table_contract(
            RecordTable.ELIGIBLE_DECISION, pd.DataFrame(decisions)
        ),
        RecordTable.EXECUTED_TRADE: stamp_table_contract(
            RecordTable.EXECUTED_TRADE, pd.DataFrame(trades)
        ),
        RecordTable.GEOMETRY_DOSSIER: stamp_table_contract(
            RecordTable.GEOMETRY_DOSSIER, pd.DataFrame(geometry)
        ),
        RecordTable.QUARANTINE: stamp_table_contract(
            RecordTable.QUARANTINE, pd.DataFrame()
        ),
    }


def test_performance_is_exact_execution_only_and_cluster_bootstrap_is_stable() -> None:
    trades = _tables()[RecordTable.EXECUTED_TRADE]
    first = compute_trade_stats(
        trades,
        cost_points=0.0,
        evaluation_config_hash=_EVALUATION_HASH,
    )
    second = compute_trade_stats(
        trades,
        cost_points=0.0,
        evaluation_config_hash=_EVALUATION_HASH,
    )
    assert first["n"] == 2
    assert first["cluster_bootstrap_ci95"] == second["cluster_bootstrap_ci95"]
    assert first["cluster_bootstrap_ci95"]["samples"] == 10_000
    assert first["cluster_bootstrap_ci95"]["available"] is True


def test_trade_stats_reject_wrong_side_stop_and_entry_bar_resolution() -> None:
    trades = _tables()[RecordTable.EXECUTED_TRADE]
    wrong_stop = trades.copy()
    wrong_stop.loc[0, "stop_ticks"] = 101
    with pytest.raises(ValueError, match="risk_ticks|wrong side"):
        compute_trade_stats(
            wrong_stop,
            cost_points=0.0,
            evaluation_config_hash=_EVALUATION_HASH,
        )
    entry_bar = trades.copy()
    entry_bar.loc[0, "bars_after_entry_to_resolution"] = 0
    with pytest.raises(ValueError, match="entry bar"):
        compute_trade_stats(
            entry_bar,
            cost_points=0.0,
            evaluation_config_hash=_EVALUATION_HASH,
        )


def test_reports_keep_candidate_and_decision_surfaces_free_of_performance() -> None:
    tables = _tables()
    candidate = build_candidate_report(
        tables[RecordTable.ENTRY_CANDIDATE],
        tables[RecordTable.CANDIDATE_LABEL],
        evaluation_config_hash=_EVALUATION_HASH,
        max_candidates_per_day=None,
    )
    decision = build_decision_report(
        tables[RecordTable.ENTRY_CANDIDATE],
        tables[RecordTable.ELIGIBLE_DECISION],
        evaluation_config_hash=_EVALUATION_HASH,
    )
    executed = build_executed_trade_report(
        tables[RecordTable.EXECUTED_TRADE],
        cost_points=0.0,
        evaluation_config_hash=_EVALUATION_HASH,
        tick_size=0.25,
    )
    for report in (candidate, decision):
        payload = json.dumps(report)
        assert report["performance_metrics_present"] is False
        assert "profit_factor" not in payload
        assert "drawdown" not in payload
        assert "equity" not in payload
    assert executed["trade_count"] == 2
    assert executed["performance"]["pnl_usd"]["all"]["profit_factor"] == 1.0


def test_fk_and_invariant_reconciliation_fail_closed() -> None:
    tables = _tables()
    validate_foreign_keys(tables)
    audit = build_invariant_audit(
        tables,
        data_access_audit={
            "allowlist": ["2026-01-13"],
            "denied_dates": {},
            "path_constructions_by_date": {"2026-01-13": 1},
            "metadata_accesses_by_date": {"2026-01-13": 1},
            "file_opens_by_date": {"2026-01-13": 1},
            "rows_read_by_date": {"2026-01-13": 100},
        },
    )
    assert audit["passed"] is True
    assert set(audit["violations"].values()) == {0}

    broken = dict(tables)
    broken_trade = tables[RecordTable.EXECUTED_TRADE].copy()
    broken_trade.loc[0, "decision_id"] = "missing"
    broken[RecordTable.EXECUTED_TRADE] = broken_trade
    with pytest.raises(ValueError, match="missing parents"):
        validate_foreign_keys(broken)


def _identity() -> DatasetIdentity:
    state = RepositoryState(
        name="Strategy-Core",
        path="C:/repo",
        head="1" * 40,
        dirty_status_sha256="2" * 64,
        source_tree_hash="3" * 64,
    )
    return DatasetIdentity(
        repositories=(state,),
        authoritative_source_blob="9b5f6f163ae060030c5695dbc0aede94e0ebebcd",
        resolved_profile_hash=_PROFILE_HASH,
        evaluation_config_hash=_EVALUATION_HASH,
        date_allowlist=("2026-01-13", "2026-01-14"),
        permitted_source_hashes=(("2026-01-13/mbp1.parquet", "4" * 64),),
    )


def test_repository_state_ignores_generated_outputs_outside_source_scope(
    tmp_path: Path,
) -> None:
    source = tmp_path / "src"
    source.mkdir()
    source_file = source / "module.py"
    source_file.write_text("VALUE = 1\n", encoding="utf-8")
    for args in (
        ("init",),
        ("config", "user.email", "ifvg-test@example.invalid"),
        ("config", "user.name", "IFVG Test"),
        ("add", "src/module.py"),
        ("commit", "-m", "initial"),
    ):
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    before = read_repository_state(
        "test",
        tmp_path,
        source_paths=("src",),
    )
    generated = tmp_path / "data" / "ifvg_datasets" / "dataset-id"
    generated.mkdir(parents=True)
    (generated / "manifest.json").write_text("{}\n", encoding="utf-8")
    after_output = read_repository_state(
        "test",
        tmp_path,
        source_paths=("src",),
    )
    assert after_output.dirty_status_sha256 == before.dirty_status_sha256
    assert after_output.source_tree_hash == before.source_tree_hash

    source_file.write_text("VALUE = 2\n", encoding="utf-8")
    after_source_change = read_repository_state(
        "test",
        tmp_path,
        source_paths=("src",),
    )
    assert after_source_change.dirty_status_sha256 != before.dirty_status_sha256
    assert after_source_change.source_tree_hash != before.source_tree_hash


def test_dataset_id_is_content_derived_and_save_is_immutable(tmp_path: Path) -> None:
    identity = _identity()
    assert dataset_id_for(identity) == dataset_id_for(identity)
    moved_state = replace(identity.repositories[0], path="D:/another-checkout")
    assert dataset_id_for(
        replace(identity, repositories=(moved_state,))
    ) == dataset_id_for(identity)
    tables = _tables()
    reports = {
        "candidate_report": build_candidate_report(
            tables[RecordTable.ENTRY_CANDIDATE],
            tables[RecordTable.CANDIDATE_LABEL],
            evaluation_config_hash=_EVALUATION_HASH,
            max_candidates_per_day=None,
        ),
        "decision_report": build_decision_report(
            tables[RecordTable.ENTRY_CANDIDATE],
            tables[RecordTable.ELIGIBLE_DECISION],
            evaluation_config_hash=_EVALUATION_HASH,
        ),
        "executed_trade_report": build_executed_trade_report(
            tables[RecordTable.EXECUTED_TRADE],
            cost_points=0.0,
            evaluation_config_hash=_EVALUATION_HASH,
            tick_size=0.25,
        ),
    }
    invariant = build_invariant_audit(
        tables,
        data_access_audit={
            "denied_dates": {},
            "path_constructions_by_date": {},
            "metadata_accesses_by_date": {},
            "file_opens_by_date": {},
            "rows_read_by_date": {},
        },
    )
    output = save_v2_dataset_immutable(
        base_dir=tmp_path,
        identity=identity,
        raw_config={"profile_name": "doc"},
        effective_config={"profile_hash": _PROFILE_HASH},
        tables=tables,
        invariant_audit=invariant,
        data_access_audit={},
        **reports,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == dataset_id_for(identity)
    assert manifest["invariant_audit_passed"] is True
    assert all(len(item["sha256"]) == 64 for item in manifest["artifacts"])
    assert (output / "executed_trade.parquet").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        save_v2_dataset_immutable(
            base_dir=tmp_path,
            identity=identity,
            raw_config={},
            effective_config={},
            tables=tables,
            invariant_audit=invariant,
            data_access_audit={},
            **reports,
        )


def test_immutable_save_refuses_failed_audit_before_writing(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="passing invariant"):
        save_v2_dataset_immutable(
            base_dir=tmp_path,
            identity=_identity(),
            raw_config={},
            effective_config={},
            tables=_tables(),
            candidate_report={},
            decision_report={},
            executed_trade_report={},
            invariant_audit={"passed": False},
            data_access_audit={},
        )
    assert list(tmp_path.iterdir()) == []
