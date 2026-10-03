"""The owner table is a finite plan, with independent cash and no fit stages."""

from __future__ import annotations

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    SearchCharterEnvelope,
    SearchCharterPayload,
    _example_charter_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children
from alpha_lab.agents.data_infra.ifvg.search.strategy_approval import (
    StrategySearchApprovalPayload,
    charter_intent,
    charter_intent_hash,
)
from alpha_lab.agents.data_infra.ifvg.search.task_b import (
    TASK_B_WARMUP_DATES,
    TaskBEvidenceRef,
    explicit_configuration_rows,
    task_b_configurations,
    task_b_request_fields,
    task_b_requirement_set,
    validate_task_b_request,
)
from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import (
    task_b_semantic,
)


def task_fields(tmp_path):
    return task_b_request_fields(
        charter_intent(_example_charter_payload()),
        replay_dates=(*TASK_B_WARMUP_DATES, "2025-06-16", "2026-06-10"),
        registry_paths=(tmp_path / "2025.json", tmp_path / "2026.json"),
        artifact_provenance_dates=(*TASK_B_WARMUP_DATES, "2025-06-16", "2026-06-10"),
        menthorq_source_file_sha256={"levels.csv": "a" * 64, "regime.csv": "b" * 64},
        preparation_catalog_sha256={str(tmp_path / "2025_catalog"): "c" * 64,
                                    str(tmp_path / "2026_catalog"): "d" * 64},
        owner_evidence=(
            TaskBEvidenceRef(label="TASK_B.md", path=str(tmp_path / "TASK_B.md"), sha256="a" * 64),
            TaskBEvidenceRef(label="owner_decisions_20_21", path=str(tmp_path / "decisions.md"),
                            sha256="b" * 64),
        ),
    )


def test_thirteen_named_rows_enumerate_without_cartesian_expansion(tmp_path):
    fields = task_fields(tmp_path)
    payload = SearchCharterPayload(
        **fields, owner_authorization=_example_charter_payload().owner_authorization
    )
    charter = SearchCharterEnvelope.from_payload(payload)
    children = enumerate_children(
        charter, identity_resolver=lambda child: child.resolved_section_config_hash
    )
    assert [child.configuration_name for child in children] == [
        "baseline", "context_on", "regime_positive", "regime_negative", "gex1_block",
        "slot_s1", "slot_s2", "slot_s3", "slot_s4", "midsession", "negative_s1",
        "positive_mid_gex1", "ny_only",
    ]
    assert len({child.resolved_section_config_hash for child in children}) == 13
    assert children[0].comparison_role == "baseline"
    assert all(child.comparison_role == "challenger" for child in children[1:])
    assert tuple(name for name, _ids in explicit_configuration_rows(fields)) == tuple(
        child.configuration_name for child in children
    )
    rebuilt = SearchCharterPayload.model_validate(payload.model_dump(mode="json"))
    assert charter_intent(rebuilt) == fields


def test_exact_arm_settings_and_ny_only_change_no_other_setting():
    sections = {
        row.name: resolve_profile_config({
            "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
            "section_overrides": resolve_axis_overrides(dict(row.axis_value_ids)),
        }).section for row in task_b_configurations()
    }
    baseline = sections["baseline"].model_dump(mode="json")
    ny = sections["ny_only"].model_dump(mode="json")
    assert {key for key in baseline if baseline[key] != ny[key]} == {"enabled_entry_sessions"}
    assert sections["ny_only"].enabled_entry_sessions == ("ny",)
    assert all(section.holding_policy == "scheduled_daily_close_v1"
               for section in sections.values())
    assert all(section.regime_unknown_policy == "allow" for section in sections.values())
    assert sections["gex1_block"].nearest_support_universe == "studied_8"
    assert sections["positive_mid_gex1"].regime_gate_policy == "positive_only"
    assert sections["positive_mid_gex1"].entry_schedule_windows == (("10:00", "13:30"),)


def test_no_added_rows_dates_or_fitting_scope(tmp_path):
    fields = task_fields(tmp_path)
    validate_task_b_request(fields)
    payload = SearchCharterPayload(
        **fields, owner_authorization=_example_charter_payload().owner_authorization
    )
    semantic = task_b_semantic(SearchCharterEnvelope.from_payload(payload))
    assert semantic.payload.feature_bundle_ids == ()
    assert semantic.payload.label_policy_id is None
    assert semantic.payload.fold_protocol_id is None
    assert semantic.payload.model_protocol_id is None
    assert [stage.value[:2] for stage in semantic.payload.stage_plan] == [
        "00", "01", "02", "03", "12", "14", "15"
    ]
    for mutation in (
        {"max_child_count": 14},
        {"explicit_configurations": fields["explicit_configurations"][:-1]},
        {"date_policy": {**fields["date_policy"], "warmup_dates": ("2025-06-04",)}},
    ):
        with pytest.raises(ValueError, match="Task B"):
            validate_task_b_request({**fields, **mutation})


def test_ordinary_charter_optional_defaults_preserve_intent_hash():
    original = charter_intent(_example_charter_payload())
    expanded = {**original, "explicit_configurations": (), "task_b_execution": None}
    expanded["date_policy"] = {**expanded["date_policy"], "warmup_policy_id": None}
    assert charter_intent_hash(original) == charter_intent_hash(expanded)
    assert "task_b_execution" not in original
    assert "explicit_configurations" not in original


def test_task_approval_requires_both_owner_citations(tmp_path):
    fields = task_fields(tmp_path)
    evidence = ("TASK_B.md:" + "a" * 64, "owner_decisions_20_21:" + "b" * 64)
    values = {
        "store_namespace_id": "c" * 64,
        "requirement_set_id": task_b_requirement_set(fields).requirement_set_id,
        "charter_intent_sha256": canonical_contract_sha256(fields),
        "approved_charter_json": json.dumps(fields, sort_keys=True),
        "artifact_provenance_dates": tuple(fields["date_policy"]["replay_dates"]),
        "author": "Luis", "approved_at": "2026-10-03T00:00:00Z",
        "effective_from": "2026-10-03T00:00:00Z", "reviewed_evidence_refs": evidence,
        "approval_statement": "Exact task plan preauthorization.",
    }
    StrategySearchApprovalPayload(**values)
    with pytest.raises(ValueError, match="cite Luis"):
        StrategySearchApprovalPayload(**{**values, "reviewed_evidence_refs": evidence[:1]})


def test_logical_deadline_guard_prevents_the_actual_print_loader_open(tmp_path, monkeypatch):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from alpha_lab.propsim.funded import print_minutes
    from alpha_lab.propsim.funded.comparison_run import PrintStats
    from alpha_lab.propsim.funded.pair_engine import DayInput

    opened = []
    monkeypatch.setattr(
        print_minutes, "load_print_day", lambda root, day: opened.append(day.isoformat())
    )
    bar = SimpleNamespace(
        open_ts_utc=datetime(2026, 6, 10, 23, 58, tzinfo=UTC),
        close_ts_utc=datetime(2026, 6, 10, 23, 59, tzinfo=UTC),
        logical_open_ts_utc=datetime(2026, 6, 10, 23, 59, tzinfo=UTC),
        logical_close_ts_utc=datetime(2026, 6, 11, tzinfo=UTC),
    )

    def authorize(day):
        if day.isoformat() > "2026-06-10":
            raise PermissionError("protected print partition")

    stats = PrintStats(tmp_path, authorize_source_day=authorize)
    loader = stats.factory(DayInput("2026-06-10", {60: [bar]}, None, True, None))
    with pytest.raises(PermissionError, match="protected"):
        loader()
    assert opened == ["2026-06-10"]


def test_actual_print_loader_binds_bytes_and_rechecks_saved_receipts(tmp_path, monkeypatch):
    from datetime import UTC, date, datetime
    from types import SimpleNamespace

    import pyarrow as pa
    import pyarrow.parquet as pq

    from alpha_lab.agents.data_infra.ifvg import prepared_store
    from alpha_lab.agents.data_infra.ifvg.search import task_b_execution
    from alpha_lab.propsim.funded.price_evidence import load_print_day

    raw = tmp_path / "data"
    source = raw / "NQ" / "2026-06-10" / "mbp10.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(pa.table({
        "ts_event": pa.array([datetime(2026, 6, 10, 20, tzinfo=UTC)],
                             type=pa.timestamp("ns", "UTC")),
        "action": ["T"], "price": [20000.0], "instrument_id": [1],
        "symbol": ["NQM6"], "sequence": [1],
    }), source)
    tracker = task_b_execution._PrintSourceReceipts()
    loaded = load_print_day(raw / "NQ", date(2026, 6, 10), source_file_observer=tracker.observe)
    assert loaded.file == "2026-06-10/mbp10.parquet"
    assert len(tracker.files) == 1
    receipt = next(iter(tracker.files.values()))
    assert receipt["sha256"] == task_b_execution._source_sha256(source)
    registration = SimpleNamespace(
        source_dates=("2026-06-10",), definition={"raw_data_dir": str(raw), "symbol": "NQ"}
    )
    monkeypatch.setattr(prepared_store, "load_prepared_store", lambda path: registration)
    context = SimpleNamespace(charter=SimpleNamespace(payload=SimpleNamespace(
        task_b_execution=SimpleNamespace(prepared_store_registry_paths=(tmp_path / "registry",))
    )))
    files = {"print_source_receipts.json": json.dumps(list(tracker.files.values())).encode()}
    task_b_execution._verify_print_source_receipts(context, files)
    source.write_bytes(b"changed source")
    with pytest.raises(PermissionError, match="changed"):
        tracker.observe(source)
    with pytest.raises(PermissionError, match="changed"):
        task_b_execution._verify_print_source_receipts(context, files)


def test_task_funded_requirements_cover_actual_prop_computation(tmp_path):
    fields = task_fields(tmp_path)
    requirements = task_b_requirement_set(fields)
    assert "funded:exact_account_plan" in {
        row.decision_key for row in requirements.payload.requirements
    }


@pytest.mark.parametrize(("day", "deadline_local"), (
    ("2025-07-03", "2025-07-03T15:55:00"),
    ("2025-07-04", "2025-07-03T22:55:00"),
    ("2025-12-24", "2025-12-24T15:55:00"),
    ("2025-12-25", "2025-12-24T22:55:00"),
    ("2025-12-31", "2025-12-31T15:55:00"),
    ("2026-01-01", "2025-12-31T22:55:00"),
    ("2026-04-03", "2026-04-03T08:10:00"),
))
def test_calendar_owns_existing_normal_late_and_abbreviated_logical_closures(day, deadline_local):
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
    from alpha_lab.propsim.funded.clock import to_ns

    section = resolve_profile_config({
        "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
        "section_overrides": {"holding_policy": "scheduled_daily_close_v1"},
    }).section
    schedule, start, cutoff = _calendar(section, (day,))
    assert len(schedule) == 1
    expected = datetime.fromisoformat(deadline_local).replace(tzinfo=ZoneInfo("America/Chicago"))
    assert schedule[0].trading_day == day
    assert schedule[0].deadline_ns == to_ns(expected)
    assert start < schedule[0].deadline_ns < cutoff == schedule[0].day_end_ns


@pytest.mark.parametrize("days", (
    ("2025-07-03", "2025-07-04"),
    ("2025-12-24", "2025-12-25"),
    ("2025-12-31", "2026-01-01"),
    ("2026-04-03",),
))
def test_logical_deadline_matches_core_clock_reference_and_live_funded_pairs(days):
    from datetime import UTC, date, datetime, timedelta

    from strategy_core.strategies.ifvg_smc.daily_close import (
        DailyCloseController,
        configured_daily_close_calendar,
    )
    from strategy_core.types import BarKind, CloseReason

    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
    from alpha_lab.propsim.funded.comparison_source import core_trade_key
    from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_days
    from tests.propsim.funded.pair_builders import (
        BASE,
        ScriptedDriver,
        ScriptedPrints,
        ledger_for,
        minute_bar,
    )

    section = resolve_profile_config({
        "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
        "section_overrides": {"holding_policy": "scheduled_daily_close_v1"},
    }).section
    calendar = configured_daily_close_calendar(
        section.daily_close_timezone, section.daily_close_time, section.daily_close_buffer_minutes
    )
    schedule, start_ns, cutoff_ns = _calendar(section, days)

    class CalendarDriver(ScriptedDriver):
        def begin_day(self, bars_by_tf, levels_for):
            bars = super().begin_day(bars_by_tf, levels_for)
            self.clock = DailyCloseController(calendar)
            close = calendar.first_deadline_after(
                bars[0].availability_ts_utc, bars[-1].availability_ts_utc
            )
            self._deadline = close.deadline_ts_utc
            return bars

        def step(self, bar, gate):
            self.clock.before_decision(
                bar, position_entry=self.position.entry_ts if self.position is not None else None
            )
            return super().step(bar, gate)

    inputs = []
    for closing in schedule:
        deadline = datetime.fromtimestamp(closing.deadline_ns // 1_000_000_000, tz=UTC)
        bars = [minute_bar(date.fromisoformat(closing.trading_day),
                           deadline - timedelta(minutes=3), index,
                           [BASE if index < 2 else BASE + 8]) for index in range(3)]
        for index, bar in enumerate(bars):
            bar.signal = ("long", BASE - 100, BASE + 100) if index == 0 else None
            bar.close_ts_utc -= timedelta(microseconds=1)
            bar.kind, bar.timeframe_ticks = BarKind.TIME, 60
            bar.is_complete, bar.is_partial, bar.close_reason = True, False, CloseReason.COMPLETE
        inputs.append(DayInput(closing.trading_day, {60: bars}, None, True, closing))
    reference = PairRun("reference", CalendarDriver(), None)
    runs = [PairRun(f"CFG|{firm}", CalendarDriver(),
                    ledger_for(firm, schedule, start_ns, cutoff_ns))
            for firm in ("takeprofittrader", "myfundedfutures")]
    run_days([reference, *runs], inputs, lambda _day: (lambda: ScriptedPrints()))
    assert len(reference.strategy_trades) == len(days)
    for run in runs:
        run.ledger.finish()
        assert [core_trade_key(trade) for trade in run.strategy_trades] == [
            core_trade_key(trade) for trade in reference.strategy_trades
        ]
        assert [trade["exit_ns"] for trade in run.ledger.trades] == [
            day.deadline_ns for day in schedule
        ]
        assert all(trade["exit_kind"] == "scheduled_close" for trade in run.ledger.trades)
        assert run.ledger.finished and run.ledger.position is None


def test_s12_constructs_verified_funded_output_with_pinned_core_without_exit_policy(
    tmp_path, monkeypatch,
):
    from types import SimpleNamespace

    import pandas as pd

    from alpha_lab.agents.data_infra.ifvg import day_artifacts, prepared_store
    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from alpha_lab.agents.data_infra.ifvg.search import saved_strategy_result, task_b_execution
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        load_sidecar_bytes,
        load_verified_envelope,
    )
    from alpha_lab.propsim.funded import print_minutes
    from tests.agents.ifvg_search.conftest import make_resolved_trades_frame

    fields = task_fields(tmp_path)
    charter = SearchCharterEnvelope.from_payload(SearchCharterPayload(
        **fields, owner_authorization=_example_charter_payload().owner_authorization,
    ))
    dates = tuple(charter.payload.date_policy.replay_dates)
    children, specs, contexts = [], [], {}
    for ordinal, configuration in enumerate(charter.payload.explicit_configurations):
        resolved = resolve_profile_config({
            "profile_name": charter.payload.baseline_profile_name,
            "section_overrides": resolve_axis_overrides(dict(configuration.axis_value_ids)),
        })
        assert not hasattr(resolved.section, "exit_policy")
        cfg = IfvgCaptureConfig(section=resolved.section, data_dir=tmp_path / "raw")
        key = resolved.section_config_hash
        children.append({"ordinal": ordinal, "core_replay_id": f"{ordinal:064x}",
                         "state": "completed"})
        specs.append(SimpleNamespace(ordinal=ordinal, resolved_section_config_hash=key))
        contexts[key] = (resolved, cfg, None, None)
    context = SimpleNamespace(charter=charter, children=children, specs=specs,
                              store_root=tmp_path / "synthetic_store")

    class Policy:
        registrations = (SimpleNamespace(source_dates=dates),)

        def __init__(self):
            self.dates = dates

        def assert_zero_forbidden_access(self):
            return None

    class Provider:
        def register_day_artifacts(self, artifacts, tick_size):
            assert tick_size == 0.25

        def snapshot(self, ts_utc):
            return None

    loaded_days = []

    def synthetic_day(day, cfg, *, access_policy):
        assert day in access_policy.dates
        loaded_days.append((cfg.profile_hash, day))
        return SimpleNamespace(bars=(), level_timeline=None)

    # Only source/authority seams are synthetic. The pinned Core lifecycle,
    # account engine, current ledgers, canonical validator and S12 serializer run.
    empty_trades = make_resolved_trades_frame(("2025-06-16",)).head(0)
    empty_trades["is_warmup"] = pd.Series(index=empty_trades.index, dtype=bool)
    monkeypatch.setattr(prepared_store, "load_registered_day_artifacts", synthetic_day)
    monkeypatch.setattr(day_artifacts, "levels_for_from_frame", lambda frame: lambda ts: [])
    monkeypatch.setattr(task_b_execution, "_provider", lambda cfg, scope: Provider())
    monkeypatch.setattr(task_b_execution, "validate_task_b_request", lambda *a, **kw: None)
    monkeypatch.setattr(saved_strategy_result, "load_saved_strategy_result", lambda root, core_id:
                        SimpleNamespace(tables={RecordTable.EXECUTED_TRADE: empty_trades}))
    monkeypatch.setattr(print_minutes, "load_print_day", lambda *a, **kw:
                        pytest.fail("synthetic zero-trade test opened source prints"))

    output_ids, _ = task_b_execution._funded(context, contexts, Policy)
    assert len(output_ids) == 1
    envelope = load_verified_envelope(
        context.store_root, "task_b_artifacts", output_ids[0],
        task_b_execution.TaskBArtifactEnvelope,
    )
    assert envelope.payload.kind == "funded_accounts"
    assert envelope.task_b_artifact_id == output_ids[0]
    assert envelope.payload.search_id == charter.search_id
    assert envelope.payload.core_replay_ids == tuple(row["core_replay_id"] for row in children)
    outputs = json.loads(load_sidecar_bytes(
        context.store_root, "task_b_artifacts", output_ids[0], "funded_runs.json",
    ))
    result = json.loads(load_sidecar_bytes(
        context.store_root, "task_b_artifacts", output_ids[0], "comparison_result.json",
    ))
    assert len(outputs) == 13
    assert {row["exit_policy"] for row in outputs} == {"fixed_target_v1"}
    assert all(row["reference"]["equivalent"] for row in outputs)
    assert len(result["summaries_cents"]) == 26
    assert result["validation"]["passed"] is True
    assert all(summary["status"] == "Completed" for summary in result["summaries_cents"].values())
    assert json.loads(load_sidecar_bytes(
        context.store_root, "task_b_artifacts", output_ids[0], "print_source_receipts.json",
    )) == []
    # Exercise the actual index/envelope/sidecar reload and reuse path. No print
    # sources were needed; the synthetic receipt metadata names only this fixture.
    monkeypatch.setattr(prepared_store, "load_prepared_store", lambda path: SimpleNamespace(
        source_dates=dates, definition={"raw_data_dir": str(tmp_path / "raw"), "symbol": "NQ"},
    ))
    reloaded_envelope, sidecars = task_b_execution._load(context, "funded_accounts")
    assert reloaded_envelope == envelope
    assert json.loads(sidecars["funded_runs.json"]) == outputs
    assert json.loads(sidecars["comparison_result.json"]) == result
    loads_before_reuse = list(loaded_days)
    reused_ids, explanation = task_b_execution._funded(context, contexts, Policy)
    assert reused_ids == output_ids
    assert explanation == "Verified funded account results reused"
    assert loaded_days == loads_before_reuse


@pytest.mark.parametrize("task_b", [True, False], ids=["task_b", "ordinary"])
def test_completed_capture_lifetime_preserves_verified_evidence(tmp_path, task_b):
    """Exercise private stage lifecycle; this synthetic test authorizes no run."""
    import weakref
    from dataclasses import replace

    import pandas as pd

    from alpha_lab.agents.data_infra.ifvg.search import pipeline
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        load_executed_trade_table,
    )
    from alpha_lab.agents.data_infra.ifvg.search.lineage import (
        LineageUniquenessEnvelope,
        deserialize_native_lineage_map,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        load_sidecar_bytes,
        load_verified_envelope,
        save_or_reuse_envelope,
    )
    from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

    fixture = build_pipeline_fixture(tmp_path)
    charter = fixture["charter"]
    if task_b:
        charter = SearchCharterEnvelope.from_payload(SearchCharterPayload(
            **task_fields(tmp_path),
            owner_authorization=_example_charter_payload().owner_authorization,
        ))
    base_runner = fixture["wiring"].child_runner
    references = {}
    audits = []

    class CapturedResult:
        def __init__(self, source):
            self.tables = source.tables
            self.gross_trade_stream_hash = source.gross_trade_stream_hash
            self.capture = object()
            self.audit_capture = object()
            self.day_funnels = {"2026-01-13": {"executed": 1}}

    def run_child(*, spec, core_replay_id):
        # This detects the loop's local `result` retaining the previous capture,
        # even if tables_by_child has already been cleared.
        assert all((reference() is None) == task_b for reference in references.values())
        result = CapturedResult(base_runner(spec=spec, core_replay_id=core_replay_id))
        for frame in result.tables.values():
            frame["exit_ticks"] = frame["target_ticks"].where(
                frame["resolution"].eq("target"), frame["stop_ticks"],
            )
            frame["scheduled_exit_deadline_ts_utc"] = pd.NaT
            frame["scheduled_exit_schedule_id"] = None
        references[core_replay_id] = weakref.ref(result)
        envelope = fixture["wiring"].identity_resolver(spec)
        # Model the runner's immutable publication before returning. The audit
        # callback below reloads this synthetic companion through the real store.
        save_or_reuse_envelope(
            fixture["store_root"], "core_replays", envelope,
            extra_files={"test_companion.json": json.dumps({
                "core_replay_id": core_replay_id, "published": True,
            }).encode("utf-8")},
        )
        return result

    def audit_child(row, result):
        core_id = row["core_replay_id"]
        if task_b:
            assert result is None
        else:
            assert result is references[core_id]()
        companion = json.loads(load_sidecar_bytes(
            fixture["store_root"], "core_replays", core_id, "test_companion.json",
        ))
        assert companion == {"core_replay_id": core_id, "published": True}
        audits.append(core_id)
        return (canonical_contract_sha256(companion),)

    context = pipeline._RunContext(
        semantic=fixture["semantic"], charter=charter,
        wiring=replace(fixture["wiring"], child_runner=run_child, audit_builder=audit_child),
        store_root=fixture["store_root"], state_root=fixture["state_root"],
        state=pipeline._initial_state(fixture["semantic"]), synthetic=True,
    )
    pipeline._stage_s01_prepare(context)
    replay_ids, _ = pipeline._stage_s02_replays(context)
    assert {row["state"] for row in context.children} == {"completed"}, [
        (row["ordinal"], row["explanation"]) for row in context.children
    ]
    assert len(replay_ids) == (13 if task_b else 4)
    assert all((reference() is None) == task_b for reference in references.values())
    assert set(context.tables_by_child) == (set() if task_b else set(replay_ids))
    for core_id in replay_ids:
        evidence = context.executed_trades_by_child[core_id]
        stored = load_executed_trade_table(context.store_root, evidence.executed_trade_table_id)
        pd.testing.assert_frame_equal(stored.frame, evidence.frame)
        evaluation = pipeline._child_evaluation_envelope(core_id, charter.payload.cost_policy)
        loaded_metrics = pipeline._load_child_evaluation(
            context.store_root, evaluation.costed_evaluation_id,
        )
        assert loaded_metrics is not None
        assert loaded_metrics.model_dump(mode="json") == context.metrics_by_child[
            core_id
        ].model_dump(mode="json")
        lineage = deserialize_native_lineage_map(context.lineage_sidecars[core_id])
        report = LineageUniquenessEnvelope.from_payload(lineage.uniqueness_report)
        assert load_verified_envelope(
            context.store_root, "lineage_reports", report.lineage_report_id,
            LineageUniquenessEnvelope,
        ) == report
        assert f"lineage_map_{core_id}.json" in context.stage_sidecars
        assert f"day_funnels_{core_id}.json" in context.stage_sidecars
    pipeline._stage_s03_audit(context)
    assert set(audits) == set(replay_ids)
