"""Determinism, provenance, capacity, and accepted-v2 parity reports."""

from __future__ import annotations

import inspect
from pathlib import Path

import pandas as pd
import pytest
from strategy_core.strategies.ifvg_smc.context_config import (
    FEATURE_FORMULA_VERSION,
    ContextFeatureConfig,
)
from strategy_core.structures.context import canonical_json

from alpha_lab.agents.data_infra.ifvg.capture_driver import _insert_unique, normalize_context_days
from alpha_lab.agents.data_infra.ifvg.context_contracts import ContextRecordTable
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.dataset import (
    V2CaptureResult,
    V3CaptureResult,
    _performance_source_dates,
    reconcile_v3_core_to_accepted_v2,
)
from alpha_lab.agents.data_infra.ifvg.reporting import (
    build_context_capacity_report,
    build_context_coverage_report,
    build_context_identity_report,
    build_context_performance_report,
    build_context_reconciliation_report,
    build_context_validity_report,
)
from alpha_lab.agents.data_infra.ifvg.verification import run_context_verification

from .ifvg_v3_fixtures import context_fixture


def test_performance_source_dates_keep_the_fixed_26_date_denominator() -> None:
    dates = tuple(f"2026-01-{index:02d}" for index in range(1, 31))
    assert _performance_source_dates(dates) == dates[:26]
    assert _performance_source_dates(dates[:10]) == dates[:10]


def test_strategy_core_editable_provenance_uses_import_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from alpha_lab.agents.data_infra.ifvg import verification

    checkout = tmp_path / "Strategy-core"
    (checkout / ".git").mkdir(parents=True)
    imported = checkout / "src" / "strategy_core" / "__init__.py"
    monkeypatch.setattr(verification.strategy_core, "__file__", str(imported))
    monkeypatch.setattr(
        verification,
        "distribution",
        lambda _name: (_ for _ in ()).throw(AssertionError("unexpected metadata read")),
    )

    assert verification._strategy_core_repository_root(tmp_path / "Quant-Lab") == checkout


def test_strategy_core_vcs_install_must_match_sibling_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from alpha_lab.agents.data_infra.ifvg import verification

    repo_root = tmp_path / "Claude-Quant-Lab"
    repo_root.mkdir()
    checkout = tmp_path / "Strategy-core"
    (checkout / ".git").mkdir(parents=True)
    imported = (
        tmp_path
        / "Python"
        / "Lib"
        / "site-packages"
        / "strategy_core"
        / "__init__.py"
    )
    commit = "1" * 40

    class Distribution:
        @staticmethod
        def read_text(name: str) -> str:
            assert name == "direct_url.json"
            return f'{{"vcs_info": {{"commit_id": "{commit}"}}}}'

    monkeypatch.setattr(verification.strategy_core, "__file__", str(imported))
    monkeypatch.setattr(verification, "distribution", lambda _name: Distribution())
    monkeypatch.setattr(verification, "_git_head", lambda _root: commit)

    assert verification._strategy_core_repository_root(repo_root) == checkout
    monkeypatch.setattr(verification, "_git_head", lambda _root: "2" * 40)
    with pytest.raises(RuntimeError, match="differs from source checkout HEAD"):
        verification._strategy_core_repository_root(repo_root)


def _accepted_core_fixture() -> dict[RecordTable, pd.DataFrame]:
    frames = {table: pd.DataFrame() for table in RecordTable}
    frames[RecordTable.ENTRY_CANDIDATE] = pd.DataFrame(
        {"candidate_id": ["candidate-a"], "value": [1]}
    )
    frames[RecordTable.ELIGIBLE_DECISION] = pd.DataFrame(
        {"decision_id": ["decision-a"], "candidate_id": ["candidate-a"]}
    )
    return frames


def test_normalization_is_deterministic() -> None:
    day, left, _core, _emissions = context_fixture()
    right = normalize_context_days((day,), warmup_days=0)
    for table in ContextRecordTable:
        assert canonical_json(left[table].to_dict("records")) == canonical_json(
            right[table].to_dict("records")
        )


def test_referenced_object_keeps_first_day_metadata_but_record_conflicts_fail() -> None:
    rows: dict[str, dict] = {}
    first = {
        "displacement_window_id": "window-a",
        "observed_bar_count": 4,
        "source_date": "2026-01-12",
        "is_warmup": True,
        "days_of_htf_history": 9,
        "entering_context_seed_hash": "seed-a",
    }
    later_reference = {
        **first,
        "source_date": "2026-01-13",
        "is_warmup": False,
        "days_of_htf_history": 10,
        "entering_context_seed_hash": "seed-b",
    }
    _insert_unique(rows, "window-a", first, label="displacement window")
    _insert_unique(rows, "window-a", later_reference, label="displacement window")
    assert rows["window-a"] == first

    conflicting = {**later_reference, "observed_bar_count": 5}
    with pytest.raises(ValueError, match="conflicting normalized rows"):
        _insert_unique(rows, "window-a", conflicting, label="displacement window")


def test_capacity_and_performance_measurements_belong_only_to_v3_result() -> None:
    v2 = inspect.signature(V2CaptureResult.__init__).parameters
    v3 = inspect.signature(V3CaptureResult.__init__).parameters
    assert "capacity_metrics" not in v2
    assert "performance_measurements" not in v2
    assert "capacity_metrics" in v3
    assert "performance_measurements" in v3
    assert "diagnostic_timings" in v3


def test_verification_failure_reports_exact_capacity_and_performance_violations(
    monkeypatch,
    tmp_path,
) -> None:
    # The fixed-chain entrypoint must never collapse actionable gate evidence to
    # booleans. Build a minimal seam around the report stage rather than replaying.
    import alpha_lab.agents.data_infra.ifvg.verification as verification

    day, tables, core, _emissions = context_fixture()
    capture = type(
        "Capture",
        (),
        {
            "context_tables": tables,
            "core_parity_tables": core,
            "baseline_reconciliation": {"passed": True},
            "capacity_metrics": {
                "terminal_state_bytes": 6 * 1024 * 1024,
                "terminal_seed_bytes": 2 * 1024 * 1024,
                "max_transition_bytes": 40 * 1024,
            },
            "performance_measurements": {
                "disabled_replay_seconds": 1.0,
                "enabled_replay_seconds": 2.0,
                "completed_1m_step_p99_ms": 3.0,
                "multi_timeframe_callback_p99_ms": 11.0,
                "measurement_policy": "test",
            },
        },
    )()
    dates = tuple(sorted(verification.EXPLORATION_DATE_ALLOWLIST))
    monkeypatch.setattr(
        verification,
        "_verify_authoritative_blob",
        lambda _root: None,
    )
    monkeypatch.setattr(
        verification,
        "discover_allowlisted_source_files",
        lambda *_args, **_kwargs: {
            day: tmp_path / f"{day}.parquet" for day in dates
        },
    )
    monkeypatch.setattr(
        verification,
        "hash_allowlisted_source_files",
        lambda *_args, **_kwargs: {day: "a" * 64 for day in dates},
    )
    state = type(
        "State",
        (),
        {
            "name": "Strategy-Core",
            "head": "b" * 40,
            "source_tree_hash": "c" * 64,
        },
    )()
    monkeypatch.setattr(
        verification,
        "_context_repository_states",
        lambda _root: (state,),
    )
    capture_calls: list[dict] = []

    def fake_capture(*_args, **kwargs):
        capture_calls.append(kwargs)
        return capture

    monkeypatch.setattr(
        verification,
        "build_ifvg_v3_capture",
        fake_capture,
    )
    monkeypatch.setattr(
        verification,
        "build_context_validity_report",
        lambda _tables: {"passed": True},
    )
    monkeypatch.setattr(
        verification,
        "build_context_reconciliation_report",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        verification,
        "build_context_identity_report",
        lambda *_args, **_kwargs: {"passed": True},
    )

    with pytest.raises(RuntimeError) as error:
        run_context_verification(repo_root=tmp_path)
    message = str(error.value)
    assert "capacity_violations" in message
    assert "capacity_report" in message
    assert "performance_violations" in message
    assert "performance_report" in message
    assert "terminal_seed_bytes" in message
    assert "replay_slowdown_fraction" in message
    assert len(capture_calls) == 1
    assert capture_calls[0]["measure_performance"] is True


def test_accepted_v2_reconciliation_is_content_exact() -> None:
    accepted = _accepted_core_fixture()
    report = reconcile_v3_core_to_accepted_v2(accepted, accepted)
    assert report["passed"] is True
    changed = {table: frame.copy() for table, frame in accepted.items()}
    changed[RecordTable.ENTRY_CANDIDATE].loc[0, "value"] = 2
    with pytest.raises(ValueError, match="differs from accepted v2"):
        reconcile_v3_core_to_accepted_v2(changed, accepted)


def test_performance_report_enforces_median_and_repeated_run_p95_gates() -> None:
    report = build_context_performance_report(
        disabled_replay_seconds=10.0,
        enabled_replay_seconds=11.0,
        completed_1m_step_p99_ms=1.0,
        multi_timeframe_callback_p99_ms=2.0,
        repeated_run_p95_slowdown_fraction=0.26,
        measurement_protocol="two_warmups_ten_alternating_pairs_v1",
    )

    assert report["passed"] is False
    assert report["replay_slowdown_fraction"] == pytest.approx(0.10)
    assert report["measurement_protocol"] == "two_warmups_ten_alternating_pairs_v1"
    assert report["violations"] == {
        "repeated_run_p95_slowdown_fraction": {
            "observed": 0.26,
            "limit": 0.25,
        }
    }


def test_capacity_report_enforces_formula_v2_headroom_limits() -> None:
    report = build_context_capacity_report(
        {},
        terminal_state_bytes=4_194_305,
        terminal_seed_bytes=838_861,
        max_transition_bytes=26_215,
    )

    assert report["schema_version"] == 4
    assert report["passed"] is False
    assert report["limits"] == {
        "terminal_state_bytes": 4_194_304,
        "terminal_seed_bytes": 838_860,
        "max_transition_bytes": 26_214,
        "max_members_per_pool": 16,
    }
    assert set(report["violations"]) == {
        "terminal_state_bytes",
        "terminal_seed_bytes",
        "max_transition_bytes",
    }


def test_nonperformance_reports_reconcile_exact_context() -> None:
    day, tables, core, _emissions = context_fixture()
    baseline = {"passed": True, "tables": {}}
    validity = build_context_validity_report(tables)
    coverage = build_context_coverage_report(tables)
    reconciliation = build_context_reconciliation_report(
        tables,
        core_tables=core,
        baseline_reconciliation=baseline,
    )
    identity = build_context_identity_report(
        tables,
        expected={
            "feature_set_version": "ifvg_context_v1",
            "feature_formula_version": FEATURE_FORMULA_VERSION,
            "feature_schema_hash": day.end_context_seed.identity.feature_schema_hash,
            "context_config_hash": day.end_context_seed.context_config_hash,
        },
    )
    capacity = build_context_capacity_report(
        tables,
        terminal_state_bytes=len(canonical_json(day.end_context_seed).encode()),
        terminal_seed_bytes=len(canonical_json(day.end_context_seed).encode()),
        max_transition_bytes=max(event.serialized_size() for event in day.context_events),
    )
    assert validity["passed"] is True
    assert reconciliation["passed"] is True
    assert identity["passed"] is True
    assert capacity["passed"] is True
    assert coverage["candidate_link_count"] == 2
    assert coverage["decision_link_count"] == coverage["trade_link_count"] == 1


def test_formula_or_config_change_changes_only_context_identity() -> None:
    baseline = ContextFeatureConfig()
    changed = ContextFeatureConfig(swing_strength=4)
    assert baseline.feature_set_version == changed.feature_set_version
    from strategy_core.strategies.ifvg_smc.context_config import (
        context_config_hash,
        feature_schema_hash,
    )

    assert context_config_hash(baseline) != context_config_hash(changed)
    assert feature_schema_hash(baseline) != feature_schema_hash(changed)
