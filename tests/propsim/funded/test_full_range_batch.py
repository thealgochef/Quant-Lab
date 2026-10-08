"""Synthetic orchestration/authority checks; no real market observations are read."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.agents.data_infra.ifvg.search.task_b import TaskBExecutionScope
from alpha_lab.propsim.funded.comparison_plan import APPROVAL_STORE, PLAN_STORE
from alpha_lab.propsim.funded.comparison_runner import load_plan
from alpha_lab.propsim.funded.comparison_study import record_owner_approval, save_plan
from alpha_lab.propsim.funded.full_range_batch import (
    ANNOTATION,
    CONFIGURATION_NAMES,
    FullRangeBatchPlanEnvelope,
    FullRangeBatchPlanPayload,
    FullRangeConfiguration,
    FullRangeSourceRef,
    ObservedCoreDriver,
    _assert_request,
    _assert_terminal_result_binding,
    _dispatch,
    _verify_saved_print_receipts,
    entry_context,
    load_checkpoint,
    run_stream_days,
    save_checkpoint,
    validate_full_range_plan,
)
from alpha_lab.propsim.funded.pair_engine import PairRun
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

from .pair_builders import (
    BASE,
    ScriptedDriver,
    ScriptedPrints,
    SyntheticDay,
    build,
    flat,
    ledger_for,
)

REPO = Path(__file__).resolve().parents[3]
DOC = REPO / "docs/ifsm-correct-config-full-range-v01"


@pytest.fixture
def plan():
    dates = json.loads((DOC / "date_scope.json").read_text())
    # The task's explicit membership is versioned input, not a generated calendar.
    evaluation = tuple(dates["evaluation_dates"])
    warmup = tuple(dates["warmup_dates"])
    scope = TaskBExecutionScope(
        prepared_store_registry_paths=("C:/synthetic/2025.json", "C:/synthetic/2026.json"),
        artifact_provenance_dates=tuple(sorted(warmup + evaluation)),
        owner_evidence=(
            {"label": "TASK_B.md", "path": "C:/synthetic/task.md", "sha256": "1" * 64},
            {
                "label": "owner_decisions_20_21",
                "path": "C:/synthetic/decisions.md",
                "sha256": "2" * 64,
            },
        ),
        menthorq_source_file_sha256={"levels": "3" * 64, "regime": "4" * 64},
        preparation_catalog_sha256={"2025": "5" * 64, "2026": "6" * 64},
        funded_profile_sha256={key: "7" * 64 for key in FIRM_PROFILES},
        processing_policy_sha256="8" * 64,
    )
    rows = tuple(
        FullRangeConfiguration(
            batch_id=f"C{i + 1:02d}",
            name=name,
            display_name=name,
            historical_section_json="{}",
            historical_section_config_hash=f"{i + 1}" * 64,
            effective_section_json=json.dumps(
                {
                    "exit_policy": "scale_out_half_breakeven_hold_to_close_v1"
                    if i < 2
                    else "fixed_target_v1"
                }
            ),
            effective_section_config_hash=f"{i + 1}" * 64,
            instrument="micro" if i < 2 else "mini",
            quantity=10 if i < 2 else 1,
            cost_per_contract_mills=514 if i < 2 else 5140,
        )
        for i, name in enumerate(CONFIGURATION_NAMES)
    )
    return FullRangeBatchPlanPayload(
        question="Exact six-row synthetic authority fixture",
        source=FullRangeSourceRef(
            task_b_plan_id="a" * 64,
            task_b_plan_manifest_sha256="b" * 64,
            task_b_store_namespace_id="c" * 64,
            title="Synthetic registered inputs",
            warmup_dates=warmup,
            evaluation_dates=evaluation,
            cutoff_utc="2026-06-10T21:00:00Z",
        ),
        configurations=rows,
        task_b_store_root="C:/synthetic/store",
        task_b_plan_id="a" * 64,
        task_b_scope=scope,
        prepared_registration_ids={"2025": "d" * 64, "2026": "e" * 64},
        core_root="C:/synthetic/core",
        core_source={
            "base_commit": "f" * 40,
            "patch_sha256": "0" * 64,
            "branch": "task",
            "description": "Synthetic source",
        },
        runtime_source_file_sha256={},
        request_file="C:/synthetic/request.json",
        request_file_sha256="1" * 64,
        owner_evidence_file_sha256={},
        calendar_sha256="2" * 64,
    )


def test_plan_rejects_auto_baseline_changed_sizes_dates_and_firms(plan):
    raw = plan.model_dump(mode="json")
    changes = []
    extra = copy.deepcopy(raw)
    extra["configurations"].append(extra["configurations"][0])
    changes.append(extra)
    wrong = copy.deepcopy(raw)
    wrong["configurations"][0]["quantity"] = 1
    changes.append(wrong)
    wrong = copy.deepcopy(raw)
    wrong["source"]["evaluation_dates"] = wrong["source"]["evaluation_dates"][1:]
    changes.append(wrong)
    wrong = copy.deepcopy(raw)
    wrong["firm_profiles"] = wrong["firm_profiles"][:1]
    changes.append(wrong)
    for content in changes:
        with pytest.raises(ValidationError):
            FullRangeBatchPlanPayload.model_validate(content)


def test_exact_plan_saved_reopened_new_approval_and_tampering_refused(plan, tmp_path):
    from alpha_lab.propsim.funded.comparison_plan import FundedComparisonApprovalEnvelope

    envelope = FullRangeBatchPlanEnvelope.from_payload(plan)
    plan_id = save_plan(tmp_path, envelope)
    assert load_plan(tmp_path, plan_id) == plan
    approval_id = record_owner_approval(
        tmp_path,
        plan_id,
        approved_on="2026-10-04",
        channel="claude_conversation",
        statement="Owner authorizes exact six configurations.",
        scope="One frozen six-row comparison and twelve funded results.",
    )
    approval = load_verified_envelope(
        tmp_path, APPROVAL_STORE, approval_id, FundedComparisonApprovalEnvelope
    )
    assert approval.payload.funded_comparison_plan_id == plan_id
    path = tmp_path / PLAN_STORE / plan_id / "envelope.json"
    content = json.loads(path.read_text())
    content["payload"]["configurations"][0]["quantity"] = 2
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError):
        load_plan(tmp_path, plan_id)


def test_worker_refuses_wrong_import_before_input_lookup(plan, monkeypatch):
    from alpha_lab.propsim.funded import core_identity

    monkeypatch.setattr(
        core_identity,
        "core_source_identity",
        lambda: {"root": "C:/wrong/core", "base_commit": "f" * 40, "patch_sha256": "0" * 64},
    )
    with pytest.raises(PermissionError, match="different frozen compatible Core"):
        validate_full_range_plan(plan)


def test_assert_all_common_worker_fields_and_no_display_name_substitution(plan):
    request = json.loads((DOC / "RUN_REQUEST.json").read_text())
    sections = []
    rows = []
    for row, expected in zip(plan.configurations, request["configs"], strict=True):
        values = {
            **request["common_expected_effective_values"],
            **ANNOTATION,
            **{
                key: expected[key]
                for key in (
                    "entry_schedule_policy",
                    "opposing_parent_distance_ticks_max",
                    "exit_policy",
                    "tp_r_multiple",
                )
            },
        }
        sections.append(SimpleNamespace(model_dump=lambda mode, value=values: value))
        rows.append(
            row.model_copy(
                update={
                    "historical_section_config_hash": expected["reference"][
                        "historical_section_hash"
                    ]
                }
            )
        )
    bound = plan.model_copy(update={"configurations": tuple(rows)})
    _assert_request(bound, sections, request)
    sections[0].model_dump(mode="json")["parent_retest_timeout_1m_bars"] = None
    with pytest.raises(PermissionError, match="parent_retest_timeout"):
        _assert_request(bound, sections, request)


def test_checkpoint_refuses_changed_dispatch_wrong_prefix_or_modified_bytes(tmp_path):
    path = tmp_path / "checkpoint.json"
    save_checkpoint(path, {"dispatch_sha256": "a" * 64, "completed_dates": ["2025-06-02"]})
    assert load_checkpoint(path, dispatch_sha256="a" * 64, dates=("2025-06-02",))
    with pytest.raises(PermissionError, match="different plan"):
        load_checkpoint(path, dispatch_sha256="b" * 64, dates=("2025-06-02",))
    with pytest.raises(PermissionError, match="different plan"):
        load_checkpoint(path, dispatch_sha256="a" * 64, dates=("2025-06-03",))
    raw = json.loads(path.read_text())
    raw["payload"]["completed_dates"] = ["2025-06-03"]
    path.write_text(json.dumps(raw))
    with pytest.raises(PermissionError, match="bytes failed"):
        load_checkpoint(path, dispatch_sha256="a" * 64, dates=("2025-06-03",))


@pytest.mark.parametrize("reason", [None, "no_level_row", "before_0600", "after_1700"])
def test_passive_entry_annotation_preserves_signal_and_unknown_context(reason):
    from datetime import UTC, datetime, timedelta

    from strategy_core.strategies.ifvg_smc.menthorq_levels import (
        LEVEL_COLUMN_NAMES,
        MenthorqLevelSnapshot,
    )
    from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

    from alpha_lab.propsim.funded.clock import CHICAGO

    ts = datetime(2026, 1, 13, 9, tzinfo=CHICAGO)
    if reason == "before_0600":
        ts = ts.replace(hour=5)
    elif reason == "after_1700":
        ts = ts.replace(hour=17)
    ts = ts.astimezone(UTC)
    section = default_ifvg_smc_section().model_copy(update=ANNOTATION)
    # Negative regime conflicts with a hypothetical positive gate; every saved
    # gate is off, so observing it cannot affect this already-admitted entry.
    snapshot = MenthorqLevelSnapshot(
        trading_date=ts.date(),
        source_eod_date=ts.date() - timedelta(days=1),
        source_file_sha256="a" * 64,
        levels={name: None for name in LEVEL_COLUMN_NAMES},
        regime="negative" if reason is None else "unknown",
        context_available=reason is None,
        unavailable_reason=reason,
        selected_instrument_id=100,
        roll_flag=False,
    )
    provider = SimpleNamespace(snapshot=lambda _ts: snapshot, prior_cash_close_for=lambda _ts: None)
    signal = SimpleNamespace(entry_ticks=BASE, trade_id="actual-entry", direction="long")
    before = copy.deepcopy(signal.__dict__)
    bar = SimpleNamespace(availability_ts_utc=ts, open_ticks=BASE)
    out = entry_context(provider, section, bar, signal, 0.25)
    assert signal.__dict__ == before
    assert out["availability_ts_utc"] == ts.isoformat()
    assert out["context_available"] is (reason is None)
    assert out["selected_instrument_id"] == 100
    assert not out["regime_gate_blocked"] and not out["nearest_support_gate_blocked"]


def test_observer_retains_actual_partial_timestamp_and_checkpoint_counter():
    section = SimpleNamespace(exit_policy="scale_out_half_breakeven_hold_to_close_v1")
    driver = ObservedCoreDriver(section, tick_size=0.25, menthorq_provider=None)
    driver._observe_exits(
        [
            SimpleNamespace(
                trade_id="actual",
                scale_out_ts_utc="2026-01-13T15:02:00+00:00",
                scale_out_fraction=0.5,
                exit_policy=section.exit_policy,
            )
        ]
    )
    driver.breakeven_cleared = 1
    resumed = ObservedCoreDriver(section, tick_size=0.25, menthorq_provider=None)
    resumed.restore(driver.checkpoint())
    assert (
        resumed.core_exit_annotations["actual"]["scale_out_ts_utc"] == "2026-01-13T15:02:00+00:00"
    )
    assert resumed.core_exit_annotations["actual"]["scale_out_fraction"] == 0.5
    assert resumed.breakeven_cleared == 1


def test_receipt_reuse_verifies_only_registered_named_paths_and_bytes(tmp_path):
    import hashlib

    cfg = SimpleNamespace(data_dir=tmp_path, symbol="NQ")
    policy = SimpleNamespace(registrations=[SimpleNamespace(source_dates=("2025-06-02",))])
    path = tmp_path / "NQ" / "2025-06-02" / "trades.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic source receipt fixture")
    stat = path.stat()
    receipt = {
        "path": str(path.resolve()),
        "physical_date": "2025-06-02",
        "filename": "trades.parquet",
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    assert _verify_saved_print_receipts(cfg, policy, [receipt]).files[receipt["path"]] == receipt
    for escaped in (
        {**receipt, "path": str(tmp_path / "NQ" / "2026-06-11" / "trades.parquet")},
        {**receipt, "path": str(tmp_path / "other" / "2025-06-02" / "trades.parquet")},
        {**receipt, "filename": "mbp10.parquet"},
    ):
        with pytest.raises(PermissionError, match="escaped"):
            _verify_saved_print_receipts(cfg, policy, [escaped])
    with pytest.raises(PermissionError, match="changed since"):
        _verify_saved_print_receipts(cfg, policy, [{**receipt, "sha256": "0" * 64}])


def test_full_continuation_evidence_does_not_claim_saved_parity_or_actual_resume():
    from alpha_lab.propsim.funded.clock import to_ns
    from alpha_lab.propsim.funded.comparison_result import (
        build_comparison_result,
        validate_comparison,
    )

    output = {
        "configuration": "synthetic",
        "display_name": "synthetic",
        "settings_plain": [],
        "axes": {},
        "pairs": {},
        "reference": {"equivalent": None, "saved_study_trades": None, "replayed_trades": 0},
        "resumed": {},
        "prints": {
            "minutes_checked": 0,
            "minutes_rebuilt_exactly": 0,
            "missing_utc_days": [],
            "files": [],
        },
    }
    now = to_ns("2026-01-13T15:00:00Z")
    result = build_comparison_result(
        context={},
        outputs=[output],
        failures=[],
        profiles=(),
        trading_days=(),
        start_ns=now,
        cutoff_ns=now,
        settings={},
        resume_check_requested=False,
    )
    row = result["tables"]["execution_evidence"][0]
    assert row["no_account_replay_equals_saved_study"] is None
    assert row["resumed_run_identical"] is None and not row["resume_check_requested"]
    validation = validate_comparison(
        result, [output], (), now, require_resume_check=False, require_reference_check=False
    )
    assert validation["passed"]
    assert "no_account_replay_equals_saved_study" not in validation["checks"]["synthetic"]
    assert "resumed_run_identical" not in validation["checks"]["synthetic"]


def test_terminal_pointer_cannot_substitute_another_valid_plan_result(plan, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
    from alpha_lab.propsim.funded.comparison_plan import (
        RESULT_STORE,
        FundedComparisonResultEnvelope,
        FundedComparisonResultPayload,
    )
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result
    from alpha_lab.propsim.funded.full_range_batch import ENGINE_VERSION
    from alpha_lab.propsim.funded.result import canonical_json, result_sha256

    plan_id, approval_id = "a" * 64, "b" * 64
    saved = {
        "funded_comparison_plan_id": plan_id,
        "full_range_batch": {
            "plan": plan.model_dump(mode="json"),
            "worker_dispatches": [
                _dispatch(plan_id, approval_id, plan, row) for row in plan.configurations
            ],
            "failed_configurations": [],
            "print_source_receipts": [],
        },
        "tables": {
            "pair_results": [
                {"configuration": row.name, "firm_key": firm.firm_key, "status": "Completed"}
                for row in plan.configurations
                for firm in plan.firm_profiles
            ],
        },
    }
    result = FundedComparisonResultEnvelope.from_payload(
        FundedComparisonResultPayload(
            funded_comparison_plan_id=plan_id,
            funded_comparison_approval_id=approval_id,
            result_json_sha256=result_sha256(saved),
            validation_passed=True,
            engine_version=ENGINE_VERSION,
        )
    )
    save_envelope_immutable(
        tmp_path, RESULT_STORE, result, extra_files={"result.json": canonical_json(saved).encode()}
    )
    opened = load_comparison_result(tmp_path, result.funded_comparison_result_id)
    _assert_terminal_result_binding(
        opened, result, plan_id=plan_id, approval_id=approval_id, plan=plan
    )
    # The result and its sidecar really verify; changed state must still not adopt
    # another valid plan with identical inputs or a different approval mapping.
    for wrong_plan_id, wrong_approval_id, wrong_plan in (
        ("c" * 64, approval_id, plan),
        (plan_id, "d" * 64, plan),
        (plan_id, approval_id, plan.model_copy(update={"question": "another exact plan"})),
    ):
        with pytest.raises(PermissionError, match="different frozen plan or approval"):
            _assert_terminal_result_binding(
                opened,
                result,
                plan_id=wrong_plan_id,
                approval_id=wrong_approval_id,
                plan=wrong_plan,
            )
    wrong = copy.deepcopy(opened)
    wrong["full_range_batch"]["worker_dispatches"][0]["approval_id"] = "e" * 64
    with pytest.raises(PermissionError, match="worker dispatch"):
        _assert_terminal_result_binding(
            wrong, result, plan_id=plan_id, approval_id=approval_id, plan=plan
        )
    wrong = copy.deepcopy(opened)
    wrong["full_range_batch"]["worker_dispatches"] = wrong["full_range_batch"]["worker_dispatches"][
        1:
    ]
    with pytest.raises(PermissionError, match="exactly the six"):
        _assert_terminal_result_binding(
            wrong, result, plan_id=plan_id, approval_id=approval_id, plan=plan
        )
    wrong = copy.deepcopy(opened)
    wrong["tables"]["pair_results"][0]["status"] = "Not completed"
    with pytest.raises(PermissionError, match="twelve approved"):
        _assert_terminal_result_binding(
            wrong, result, plan_id=plan_id, approval_id=approval_id, plan=plan
        )


class _ObservedScript(ScriptedDriver):
    """Observation-only fixture implementing the production stream interface."""

    def __init__(self, scale_out=False):
        super().__init__(scale_out)
        self.entry_contexts = {}
        self.core_exit_annotations = {}
        self.last_day_evidence = {}
        self.day_entries = 0

    def begin_day(self, bars_by_tf, levels_for):
        self.day_entries = 0
        return super().begin_day(bars_by_tf, levels_for)

    def step(self, bar, gate):
        out = super().step(bar, gate)
        if out.entry:
            self.day_entries += 1
            self.entry_contexts[out.entry.trade_id] = {
                "context_available": False,
                "availability_ts_utc": bar.availability_ts_utc.isoformat(),
            }
        return out

    def end_day(self, trading_day, *, dataset_exhausted=False):
        result = super().end_day(trading_day, dataset_exhausted=dataset_exhausted)
        self.last_day_evidence = {"entries": self.day_entries, "stage_counters": {"fixture": 1}}
        return result

    def checkpoint(self):
        return {**super().checkpoint(), "entry_contexts": self.entry_contexts}

    def restore(self, state):
        super().restore(state)
        self.entry_contexts = state["entry_contexts"]


@pytest.mark.parametrize("scale_out", [False, True])
def test_per_day_resume_exact_cash_trades_quantities_and_independent_context(scale_out, tmp_path):
    days = [
        SyntheticDay(
            date(2026, 1, 13 + i),
            [[BASE], [BASE, BASE + 8], *flat(2)],
            signals={0: ("long", BASE - 8, BASE + 8)},
        )
        for i in range(2)
    ]
    inputs, schedule, start, cutoff = build(days)
    all_dates = tuple(day.trading_day for day in inputs)
    providers = [SimpleNamespace(_cash_closes={}, _selected_instruments={}) for _ in range(3)]

    def make_runs():
        runs = [PairRun("reference", _ObservedScript(scale_out), None)]
        for firm in FIRM_PROFILES:
            ledger = ledger_for(
                firm,
                schedule,
                start,
                cutoff,
                quantity=10 if scale_out else 1,
                tick_value_cents=50 if scale_out else 500,
                cost_per_contract_mills=514 if scale_out else 5140,
                scale_out=scale_out,
            )
            runs.append(PairRun(ledger.pair_id, _ObservedScript(scale_out), ledger))
        return runs

    def execute(runs, dates, path, stats, activity):
        run_stream_days(
            runs,
            providers,
            dates,
            lambda _day: lambda: ScriptedPrints(),
            checkpoint_path=path,
            dispatch_sha256="a" * 64,
            all_dates=all_dates,
            daily_activity=activity,
            coverage_for_day=lambda _day: {"source_status": "available"},
            stats_state=stats,
        )

    straight = make_runs()
    straight_activity = []
    execute(straight, inputs, tmp_path / "straight.json", {}, straight_activity)
    interrupted = make_runs()
    execute(interrupted, inputs[:1], tmp_path / "resumed.json", {}, [])
    checkpoint = load_checkpoint(
        tmp_path / "resumed.json", dispatch_sha256="a" * 64, dates=all_dates
    )
    resumed = make_runs()
    for run, state in zip(resumed, checkpoint["runs"], strict=True):
        run.driver.restore(state["driver"])
        if run.ledger:
            run.ledger.restore(state["ledger"])
        run.strategy_trades = state["strategy_trades"]
        run.warmup_trades = state["warmup_trades"]
        run.also_blocked = state["also_blocked"]
    activity = checkpoint["daily_activity"]
    execute(resumed, inputs[1:], tmp_path / "resumed.json", checkpoint["stats"], activity)
    assert activity == straight_activity
    assert len(activity) == 6
    for actual, wanted in zip(resumed, straight, strict=True):
        assert actual.to_state() == wanted.to_state()
        if actual.ledger:
            assert len(actual.ledger.accounts) == 1
            assert len(actual.ledger.trades) == 2
            assert all(row["costs_cents"] == 1028 for row in actual.ledger.trades)
            assert all(row["quantity"] == (10 if scale_out else 1) for row in actual.ledger.trades)
            assert all("entry_context" in row for row in actual.ledger.trades)
