"""Typed Strategy-Core emissions reconcile through the Quant-Lab v2 pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from strategy_core.candles._ids import make_bar_id
from strategy_core.strategies.ifvg_smc.records import (
    EligibleDecisionRecord,
    EntryCandidateRecord,
    ExecutedTradeRecord,
    GeometryDossierRecord,
    GeometryEvidence,
    IfvgEmission,
    RecordEnvelope,
    SetupLifecycleEventRecord,
    bar_cursor,
    bar_evidence,
    make_candidate_id,
    make_decision_id,
    make_lifecycle_event_id,
    make_setup_id,
    make_trade_id,
)
from strategy_core.strategies.ifvg_smc.section import IFVG_STRATEGY_VERSION
from strategy_core.structures.fvg import Fvg, GapDirection
from strategy_core.types import Bar, BarKind, CloseReason, Direction

from alpha_lab.agents.data_infra.ifvg.capture_driver import flatten_emissions
from alpha_lab.agents.data_infra.ifvg.contracts import (
    RecordTable,
    partition_capture_tables,
    stamp_table_contract,
    validate_foreign_keys,
    validate_primary_keys,
    validate_table_identity,
)
from alpha_lab.agents.data_infra.ifvg.entry_dataset import (
    build_candidate_labels_from_tables,
)
from alpha_lab.agents.data_infra.ifvg.experiment import run_ifvg_v2_evaluation
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config

_DAY = date(2026, 1, 13)
_T0 = datetime(2026, 1, 12, 15, 0, tzinfo=UTC)


def _bar(index: int, o: int, h: int, low: int, c: int) -> Bar:
    logical_open = _T0 + timedelta(minutes=index)
    logical_close = logical_open + timedelta(minutes=1)
    return Bar(
        timeframe_ticks=60,
        trading_day=_DAY,
        bar_index=index,
        bar_id=make_bar_id(60, _DAY, index, BarKind.TIME),
        open_ts_utc=logical_open + timedelta(seconds=1),
        close_ts_utc=logical_close - timedelta(seconds=1),
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
        logical_close_ts_utc=logical_close,
    )


def _fvg(
    tf: int,
    direction: GapDirection,
    lo: int,
    hi: int,
    ident: str,
    confirmed: datetime,
) -> Fvg:
    return Fvg(
        fvg_id=f"{tf}:{direction.value}:{ident}",
        timeframe_seconds=tf,
        direction=direction,
        gap_low_ticks=lo,
        gap_high_ticks=hi,
        size_ticks=hi - lo,
        a_bar_id=f"{ident}:a",
        c_bar_id=f"{ident}:c",
        a_open_ts_utc=confirmed - timedelta(seconds=tf * 3),
        confirmed_ts_utc=confirmed,
        trading_day=_DAY,
    )


def test_typed_emissions_partition_label_report_and_reconcile() -> None:
    resolved = resolve_profile_config()
    tap_bar = _bar(0, 120, 122, 110, 118)
    lock_bar = _bar(1, 110, 112, 98, 105)
    inversion_bar = _bar(2, 102, 106, 97, 105)
    entry_bar = _bar(3, 101, 105, 95, 100)
    resolution_bar = _bar(4, 100, 111, 95, 110)
    htf = _fvg(
        3600,
        GapDirection.BULLISH,
        110,
        120,
        "htf",
        _T0 - timedelta(hours=2),
    )
    parent = _fvg(
        300,
        GapDirection.BULLISH,
        102,
        108,
        "parent",
        lock_bar.availability_ts_utc - timedelta(minutes=1),
    )
    opposing = _fvg(
        60,
        GapDirection.BEARISH,
        101,
        104,
        "opposing",
        lock_bar.availability_ts_utc,
    )
    entry_gap = _fvg(
        60,
        GapDirection.BULLISH,
        98,
        99,
        "entry",
        entry_bar.availability_ts_utc,
    )
    setup_id = make_setup_id(
        resolved.section_config_hash,
        htf.fvg_id,
        bar_cursor(tap_bar),
    )
    candidate_id = make_candidate_id(
        setup_id,
        "fresh_fvg_continuation",
        entry_gap.fvg_id,
    )
    decision_id = make_decision_id(
        candidate_id,
        resolved.section_config_hash,
    )
    trade_id = make_trade_id(decision_id, bar_cursor(entry_bar))

    def envelope(bar: Bar, entry_session: str) -> RecordEnvelope:
        section = resolved.section
        return RecordEnvelope(
            schema_version=2,
            strategy_id="ifvg_smc",
            strategy_version=IFVG_STRATEGY_VERSION,
            profile_hash=resolved.section_config_hash,
            trading_day=_DAY,
            ts_utc=bar.availability_ts_utc,
            setup_id=setup_id,
            profile_name=section.profile_name,
            qualification_mode=section.qualification_mode.value,
            section_config_hash=resolved.section_config_hash,
            entry_family=section.entry_family,
            label_family=section.label_family,
            entry_session=entry_session,
            anchor_policy=section.anchor_policy,
            resolver_policy=section.resolver_policy.value,
            causality_parent=section.causality_parent.value,
            causality_opposing=section.causality_opposing.value,
            causality_entry=section.causality_entry.value,
            timeout_policy="synthetic",
        )

    geometry = GeometryEvidence(
        htf=htf,
        parent=parent,
        opposing=opposing,
        entry_fvg=entry_gap,
        tap_bar=bar_evidence(tap_bar),
        lock_bar=bar_evidence(lock_bar),
        inversion_bar=bar_evidence(inversion_bar),
        entry_bar=bar_evidence(entry_bar),
        manipulation_swing_ticks=89,
        sl_buffer_ticks=1,
        entry_ticks=100,
        stop_ticks=90,
        target_ticks=110,
        feature_as_of_cursor=bar_cursor(entry_bar),
    )
    candidate = EntryCandidateRecord(
        envelope=envelope(entry_bar, "ny"),
        candidate_id=candidate_id,
        direction=Direction.LONG,
        entry_family="fresh_fvg_continuation",
        trigger_evidence_id=entry_gap.fvg_id,
        trigger_cursor=bar_cursor(entry_bar),
        entry_fvg=entry_gap,
        entry_ticks=100,
        proposed_stop_ticks=90,
        risk_ticks=10,
        proposed_target_ticks=110,
        bars_since_inversion=1,
        entry_to_parent_ticks=0,
        in_engine_session="ny",
        in_doc_session="ny",
        block_reasons=(),
        geometry=geometry,
    )
    decision = EligibleDecisionRecord(
        envelope=candidate.envelope,
        decision_id=decision_id,
        candidate_id=candidate_id,
        direction=Direction.LONG,
        execution_profile_hash=resolved.section_config_hash,
        entry_family="fresh_fvg_continuation",
        entry_cursor=bar_cursor(entry_bar),
        entry_ticks=100,
        stop_ticks=90,
        risk_ticks=10,
        target_ticks=110,
        passed_guards=("profile_runnable", "geometry_complete"),
        geometry=geometry,
    )
    trade = ExecutedTradeRecord(
        envelope=envelope(resolution_bar, "ny"),
        trade_id=trade_id,
        decision_id=decision_id,
        candidate_id=candidate_id,
        direction=Direction.LONG,
        status="resolved",
        resolution="target",
        entry_family="fresh_fvg_continuation",
        entry_cursor=bar_cursor(entry_bar),
        resolution_cursor=bar_cursor(resolution_bar),
        entry_ts_utc=entry_bar.availability_ts_utc,
        resolution_ts_utc=resolution_bar.availability_ts_utc,
        entry_ticks=100,
        stop_ticks=90,
        target_ticks=110,
        risk_ticks=10,
        bars_after_entry_to_resolution=1,
        mfe_ticks=11,
        mae_ticks=5,
        realized_ticks=10,
        realized_r=1.0,
        geometry=geometry,
    )
    dossier = GeometryDossierRecord(
        envelope=candidate.envelope,
        candidate_id=candidate_id,
        decision_id=decision_id,
        trade_id=trade_id,
        geometry=geometry,
    )

    def lifecycle(
        bar: Bar,
        from_phase: str,
        to_phase: str,
        transition: str,
        reason: str,
    ) -> SetupLifecycleEventRecord:
        cursor = bar_cursor(bar)
        return SetupLifecycleEventRecord(
            envelope=envelope(
                bar,
                "ny" if to_phase in {"S5", "S0"} else "none",
            ),
            lifecycle_event_id=make_lifecycle_event_id(
                setup_id,
                transition,
                reason,
                cursor,
            ),
            from_phase=from_phase,
            to_phase=to_phase,
            transition=transition,
            reason=reason,
            event_cursor=cursor,
            candidate_id=candidate_id if to_phase in {"S5", "S0"} else None,
            decision_id=decision_id if to_phase in {"S5", "S0"} else None,
            trade_id=trade_id if to_phase in {"S5", "S0"} else None,
        )

    emissions = (
        IfvgEmission(
            "setup_lifecycle_event",
            lifecycle(tap_bar, "S0", "S1", "setup_activated", "tap"),
        ),
        IfvgEmission("entry_candidate", candidate),
        IfvgEmission("eligible_decision", decision),
        IfvgEmission("geometry_dossier", dossier),
        IfvgEmission(
            "setup_lifecycle_event",
            lifecycle(entry_bar, "S4", "S5", "trade_opened", "decision"),
        ),
        IfvgEmission("executed_trade", trade),
        IfvgEmission(
            "setup_lifecycle_event",
            lifecycle(
                resolution_bar,
                "S5",
                "S0",
                "trade_resolved",
                "target",
            ),
        ),
    )
    trace = flatten_emissions(emissions, entering_seed_hash=None)
    trace["trace_ordinal"] = range(len(trace))
    tables = partition_capture_tables(trace)
    for table, frame in tuple(tables.items()):
        stamped = frame.copy()
        stamped["evaluation_config_hash"] = resolved.evaluation_config_hash
        stamped["is_warmup"] = False
        tables[table] = stamped
    labels = build_candidate_labels_from_tables(
        tables[RecordTable.ENTRY_CANDIDATE],
        bars_by_day={_DAY.isoformat(): (entry_bar, resolution_bar)},
        tick_size=0.25,
        resolved_profile=resolved,
    )
    tables[RecordTable.CANDIDATE_LABEL] = stamp_table_contract(
        RecordTable.CANDIDATE_LABEL,
        labels,
    )

    for table, frame in tables.items():
        validate_primary_keys(table, frame)
        validate_table_identity(table, frame)
    validate_foreign_keys(tables)
    reports = run_ifvg_v2_evaluation(
        tables,
        resolved_profile=resolved,
        data_access_audit={
            "allowlist": [_DAY.isoformat()],
            "denied_dates": {},
            "path_constructions_by_date": {_DAY.isoformat(): 1},
            "metadata_accesses_by_date": {_DAY.isoformat(): 1},
            "file_opens_by_date": {_DAY.isoformat(): 1},
            "rows_read_by_date": {_DAY.isoformat(): 2},
        },
        cost_points=0.0,
    )
    assert len(tables[RecordTable.CANDIDATE_LABEL]) == 3
    assert reports["candidate_report"]["raw_candidates"] == 1
    assert reports["decision_report"]["decisions"] == 1
    assert reports["executed_trade_report"]["trade_count"] == 1
    assert reports["executed_trade_report"]["performance"]["n"] == 1
    assert reports["executed_trade_report"]["evaluation_scope"] == {
        "policy": "post_warmup_candidate_entry_v1",
        "all_chain_candidates": 1,
        "evidence_candidates": 1,
        "warmup_candidates_excluded": 0,
        "all_chain_candidate_labels": 3,
        "evidence_candidate_labels": 3,
        "warmup_candidate_labels_excluded": 0,
        "all_chain_decisions": 1,
        "evidence_decisions": 1,
        "warmup_decisions_excluded": 0,
        "all_chain_executed_trades": 1,
        "evidence_executed_trades": 1,
        "warmup_executed_trades_excluded": 0,
    }
    assert reports["invariant_audit"]["passed"] is True

    warmup_tables = {
        table: frame.copy()
        for table, frame in tables.items()
    }
    warmup_tables[RecordTable.ENTRY_CANDIDATE]["is_warmup"] = True
    warmup_reports = run_ifvg_v2_evaluation(
        warmup_tables,
        resolved_profile=resolved,
        data_access_audit={
            "allowlist": [_DAY.isoformat()],
            "denied_dates": {},
            "path_constructions_by_date": {_DAY.isoformat(): 1},
            "metadata_accesses_by_date": {_DAY.isoformat(): 1},
            "file_opens_by_date": {_DAY.isoformat(): 1},
            "rows_read_by_date": {_DAY.isoformat(): 2},
        },
        cost_points=0.0,
    )
    assert warmup_reports["candidate_report"]["raw_candidates"] == 0
    assert warmup_reports["decision_report"]["decisions"] == 0
    assert warmup_reports["executed_trade_report"]["trade_count"] == 0
    assert warmup_reports["executed_trade_report"]["performance"]["n"] == 0
    assert warmup_reports["executed_trade_report"]["evaluation_scope"][
        "warmup_executed_trades_excluded"
    ] == 1
    assert warmup_reports["invariant_audit"]["passed"] is True
