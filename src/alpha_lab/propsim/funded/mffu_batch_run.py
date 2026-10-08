"""Source-bound, resumable execution of the one 64-intent MFFU study.

Each child owns an ordinary Core stream and a funded Core/account stream.  Only
immutable registered bars and the bounded EOD lookup are shared.  A child
checkpoint is written after each entire trading day and is bound to the final
plan, approval, worker intent, Core, and Quant-Lab source identities.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.propsim.funded.clock import from_ns, to_ns
from alpha_lab.propsim.funded.comparison_plan import (
    RESULT_STORE,
    FundedComparisonResultEnvelope,
    FundedComparisonResultPayload,
)
from alpha_lab.propsim.funded.comparison_run import PrintStats, pair_id_for
from alpha_lab.propsim.funded.full_range_batch import (
    _roundtrip,
    _verify_saved_print_receipts,
    load_checkpoint,
    save_checkpoint,
)
from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver
from alpha_lab.propsim.funded.mffu_batch_plan import (
    MffuBatchPlanPayload,
    MffuVariantRef,
    load_approved_mffu_batch_plan,
    verify_mffu_batch_plan,
)
from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_one_day
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.position_walk import (
    FEE_ROUNDING_POLICY_ID,
    TargetDecision,
    fill_cost_cents,
)
from alpha_lab.propsim.funded.profiles import INSTRUMENTS
from alpha_lab.propsim.funded.result import canonical_json, result_sha256
from alpha_lab.propsim.funded.runner import append_ledger, read_state, write_state

ENGINE_VERSION = "ifsm_mffu_context_64_batch_v1"
RESULT_SCHEMA = "ifsm_mffu_context_64_batch_result_v1"
CHECKPOINT_SCHEMA = "ifsm_mffu_context_day_checkpoint_v1"
FINAL_SCHEMA = "ifsm_mffu_context_child_output_v1"


def _validate_fee_postings(plan: MffuBatchPlanPayload) -> None:
    """Check the frozen per-fill rounding rule and each possible cents posting."""
    if plan.fee_rounding_policy != FEE_ROUNDING_POLICY_ID:
        raise PermissionError("batch fee rounding policy differs from the modeled rule")
    if (fill_cost_cents(6, 514), fill_cost_cents(3, 514)) != (308, 154):
        raise PermissionError("six/three-micro fee arithmetic changed")
    if fill_cost_cents(6, 514) + 2 * fill_cost_cents(3, 514) != 616:
        raise PermissionError("six/three/three fill fees must total 616 cents")
    partial_policies = {
        "scale_out_half_breakeven_hold_to_close_v1",
        "gamma_conditional_1r_v1",
        "early_positive_whole_1r_v1",
    }
    for row in plan.variants:
        if row.fee_rounding_policy != FEE_ROUNDING_POLICY_ID:
            raise PermissionError(f"{row.variant_id}: fee rounding policy changed")
        for quantity in row.possible_quantities:
            fill_quantities = {quantity}
            if row.exit_policy in partial_policies:
                fill_quantities.update((quantity // 2, quantity - quantity // 2))
            for fill_quantity in sorted(fill_quantities):
                fee = fill_cost_cents(fill_quantity, row.cost_per_contract_mills)
                if fee < 0:
                    raise PermissionError(f"{row.variant_id}: negative fill fee")


def validate_mffu_runtime(plan: MffuBatchPlanPayload, *, verify_inputs: bool = True):
    """Verify the final source, section bytes and all named Task B metadata.

    This step reads no raw market data.  The registered loader retains its own
    per-day access check during execution.
    """
    verify_mffu_batch_plan(plan)
    return resolve_mffu_registered_inputs(plan, verify_inputs=verify_inputs)


def resolve_mffu_registered_inputs(plan: MffuBatchPlanPayload, *, verify_inputs: bool = True):
    """Resolve only the saved calendar/configurations/registrations.

    Execution callers must verify their own source-bound approved plan first.
    The historical caller above retains its original source gate. The separate
    ML phase uses this same registered data contract with its new source gate.
    This function neither approves nor launches any replay.
    """
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.prepared_store import (
        PreparedStoreReplayPolicy,
        load_prepared_store,
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import require_store_namespace
    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
    from alpha_lab.propsim.funded.full_range_batch import _sha

    _validate_fee_postings(plan)
    sections = tuple(IfvgSmcSection.model_validate_json(row.effective_section_json)
                     for row in plan.variants)
    for row, section in zip(plan.variants, sections, strict=True):
        if (section.model_dump(mode="json") != json.loads(row.effective_section_json) or
                ifvg_profile_hash(section) != row.effective_section_config_hash):
            raise PermissionError(f"{row.variant_id}: worker section identity changed")
    namespace = require_store_namespace(Path(plan.task_b_store_root), expected_class="research")
    if namespace.store_namespace_id != plan.source.task_b_store_namespace_id:
        raise PermissionError("Task B namespace differs from the frozen source")
    charter = load_verified_envelope(
        Path(plan.task_b_store_root), "charters", plan.task_b_plan_id,
        SearchCharterEnvelope,
    )
    manifest = Path(plan.task_b_store_root) / "charters" / plan.task_b_plan_id / "manifest.json"
    if (_sha(manifest) != plan.source.task_b_plan_manifest_sha256 or
            charter.payload.task_b_execution != plan.task_b_scope or
            tuple(charter.payload.date_policy.replay_dates) !=
            plan.source.warmup_dates + plan.source.evaluation_dates):
        raise PermissionError("Task B charter, manifest, or date scope changed")
    paths = tuple(Path(path) for path in plan.task_b_scope.prepared_store_registry_paths)
    registrations = tuple(load_prepared_store(path) for path in paths)
    if ({str(reg.path): reg.registration_id for reg in registrations}
            != dict(plan.prepared_registration_ids)):
        raise PermissionError("prepared registrations changed")
    catalogs = {
        str(Path(reg.definition["catalog_path"]).resolve()): _sha(reg.definition["catalog_path"])
        for reg in registrations
    }
    if catalogs != dict(plan.task_b_scope.preparation_catalog_sha256):
        raise PermissionError("source-selected contract catalog changed")
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    policy = PreparedStoreReplayPolicy(dates, registry_paths=paths)
    roots = {str(Path(reg.definition["raw_data_dir"]).resolve()) for reg in registrations}
    if len(roots) != 1:
        raise PermissionError("registered segments disagree on the raw source root")
    cfg = IfvgCaptureConfig(
        section=sections[0], data_dir=Path(next(iter(roots))),
        prepared_store_registry_paths=paths,
        preparation_catalog_paths=tuple(Path(path) for path in catalogs),
    )
    calendars = tuple(_calendar(section, plan.source.evaluation_dates) for section in sections)
    if any(value != calendars[0] for value in calendars[1:]):
        raise PermissionError("the 64 workers disagree on the mandatory-close calendar")
    schedule, start_ns, cutoff_ns = calendars[0]
    if (canonical_contract_sha256([asdict(day) for day in schedule])
            != plan.calendar_sha256 or cutoff_ns != to_ns(plan.source.cutoff_utc)):
        raise PermissionError("the repaired calendar or terminal cutoff changed")
    if verify_inputs:
        policy.verify_registered_metadata(cfg)
    policy.assert_zero_forbidden_access()
    return sections, cfg, policy, schedule, start_ns, cutoff_ns


def _dispatch(plan_id: str, approval_id: str, plan: MffuBatchPlanPayload,
              row: MffuVariantRef) -> dict:
    return {
        "plan_id": plan_id, "approval_id": approval_id,
        "variant": row.model_dump(mode="json"),
        "core_root": plan.core_root,
        "core_source": plan.core_source.model_dump(mode="json"),
        "runtime_source_file_sha256": dict(plan.runtime_source_file_sha256),
        "context_archive_sha256": plan.context_archive_sha256,
        "source": plan.source.model_dump(mode="json"),
        "task_b_scope": plan.task_b_scope.model_dump(mode="json"),
        "engine_version": ENGINE_VERSION,
    }


def _verify_terminal_result_binding(saved: dict, envelope, *, plan_id: str,
                                    approval_id: str, plan: MffuBatchPlanPayload) -> None:
    """A mutable job-state pointer cannot substitute another approved result."""
    payload = envelope.payload
    batch = saved.get("mffu_batch", {})
    expected_dispatches = [
        _dispatch(plan_id, approval_id, plan, row) for row in plan.variants
    ]
    expected_ids = [row.variant_id for row in plan.variants]
    dispositions = batch.get("dispositions", ())
    from alpha_lab.propsim.funded.mffu_batch_correction_reuse import (
        COMPLETE_CHILD_STATUSES,
        correction_receipt,
    )

    correction = correction_receipt(plan)
    corrected_statuses = {"reused_nonimpact", "reused_equivalent_after_verification",
                          "replayed_changed", "replayed_equal"}
    if (batch.get("correction_lineage") != correction
            or (correction is None and any(item.get("status") in corrected_statuses
                                           for item in dispositions))):
        raise PermissionError("terminal correction result lacks its exact source-bound lineage")
    for item in dispositions:
        if item.get("status") in {"reused_nonimpact", "reused_equivalent_after_verification"}:
            proof = batch.get("reuse_proofs", {}).get(item.get("variant_id"), {})
            if (proof.get("status") != item.get("status")
                    or proof.get("original_result_id") != correction["original_result_id"]):
                raise PermissionError("terminal correction result lacks an exact reuse proof")
    if (
        payload.funded_comparison_plan_id != plan_id
        or payload.funded_comparison_approval_id != approval_id
        or payload.engine_version != ENGINE_VERSION
        or not payload.validation_passed
        or saved.get("funded_comparison_plan_id") != plan_id
        or not saved.get("validation", {}).get("passed")
        or batch.get("schema") != RESULT_SCHEMA
        or batch.get("plan") != plan.model_dump(mode="json")
        or batch.get("approval_id") != approval_id
        or batch.get("worker_dispatches") != expected_dispatches
        or [item.get("variant_id") for item in dispositions] != expected_ids
        or any(item.get("status") not in COMPLETE_CHILD_STATUSES
               for item in dispositions)
    ):
        raise PermissionError("terminal MFFU result differs from approved plan or workers")


def _load_context(plan: MffuBatchPlanPayload):
    from alpha_lab.agents.data_infra.ifvg.menthorq_asof import load_v02_eod_asof_zip

    return load_v02_eod_asof_zip(
        plan.context_archive_path,
        cutoff_utc=datetime.fromisoformat(plan.source.cutoff_utc.replace("Z", "+00:00")),
        expected_archive_sha256=plan.context_archive_sha256,
    )


def _target_selector(driver: MffuCoreDriver):
    from strategy_core.strategies.ifvg_smc.ifsm_policy_context import (
        IfsmPolicyContext,
        target_action,
    )

    def select(ns: int) -> TargetDecision:
        ts = from_ns(ns)
        context = IfsmPolicyContext(**driver.context_index.policy_context_fields(ts))
        snapshot = driver.context_index.snapshot(ts)
        return TargetDecision(
            action=target_action(driver.section.exit_policy, context),
            context={"asof": snapshot, "receipt": context.receipt()},
        )

    return select


def _verify_day_checkpoint(checkpoint: dict | None, dates: tuple[str, ...],
                           evaluation_dates: tuple[str, ...]) -> None:
    """A sealed payload must agree on days, streams and daily evidence."""
    if checkpoint is None:
        return
    completed = tuple(checkpoint.get("completed_dates", ()))
    stats = checkpoint.get("stats")
    runs = checkpoint.get("runs")
    evaluation = frozenset(evaluation_dates)
    expected_activity = [
        (day, stream) for day in completed if day in evaluation
        for stream in ("strategy", "funded")
    ]
    activity = checkpoint.get("daily_activity")
    if (checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or completed != dates[:len(completed)]
            or not isinstance(stats, dict)
            or stats.get("completed_count") != len(completed)
            or not isinstance(runs, list) or len(runs) != 2
            or not isinstance(activity, list)
            or [(row.get("evaluation_date"), row.get("stream"))
                if isinstance(row, dict) else (None, None) for row in activity]
            != expected_activity):
        raise PermissionError("MFFU child checkpoint day state differs from its date prefix")


def _annotate_new_rows(run: PairRun, before_strategy: int, before_funded: int) -> None:
    driver = run.driver
    for trade in run.strategy_trades[before_strategy:]:
        trade.update(driver.core_exit_annotations.get(trade["trade_id"], {}))
        context = driver.entry_contexts.get(trade["trade_id"])
        if context is not None:
            trade["entry_context"] = context
        quantity = driver.entry_quantities.get(trade["trade_id"])
        if quantity is not None:
            trade["quantity"] = quantity
    if run.ledger is not None:
        for trade in run.ledger.trades[before_funded:]:
            context = driver.entry_contexts.get(trade["strategy_trade_id"])
            if context is not None:
                trade["entry_context"] = context


def run_mffu_configuration(*, plan_id: str, approval_id: str,
                           plan: MffuBatchPlanPayload, row: MffuVariantRef,
                           work_root: Path, on_day=None) -> dict:
    """Execute or resume one policy over all 10+253 authorized dates."""
    from alpha_lab.agents.data_infra.ifvg.day_artifacts import levels_for_from_frame
    from alpha_lab.agents.data_infra.ifvg.prepared_store import load_registered_day_artifacts
    from alpha_lab.propsim.funded.comparison_describe import describe_section

    started = time.monotonic()
    sections, base_cfg, policy, schedule, start_ns, cutoff_ns = validate_mffu_runtime(
        plan, verify_inputs=False,
    )
    section = sections[int(row.variant_id[3:]) - 1]
    cfg = replace(base_cfg, section=section)
    context_index = _load_context(plan)
    dispatch = _dispatch(plan_id, approval_id, plan, row)
    dispatch_sha = canonical_contract_sha256(dispatch)
    work_root = Path(work_root) / row.variant_id
    checkpoint_path = work_root / "checkpoint.json"
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    checkpoint = load_checkpoint(checkpoint_path, dispatch_sha256=dispatch_sha, dates=dates)
    _verify_day_checkpoint(checkpoint, dates, plan.source.evaluation_dates)
    ordinary = MffuCoreDriver(
        section, tick_size=cfg.tick_size, context_index=context_index,
        quantity_policy=row.quantity_policy,
    )
    funded = MffuCoreDriver(
        section, tick_size=cfg.tick_size, context_index=context_index,
        quantity_policy=row.quantity_policy,
    )
    profile = plan.firm_profiles[0]
    ledger = PairLedger(
        pair_id=pair_id_for(row.name, profile.firm_key), configuration=row.name,
        profile=profile, processing=plan.processing, quantity=row.quantity,
        tick_value_cents=INSTRUMENTS[row.instrument].tick_value_cents,
        cost_per_side_cents=0, cost_per_contract_mills=row.cost_per_contract_mills,
        trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns,
        scale_out=row.exit_policy == "scale_out_half_breakeven_hold_to_close_v1",
    )
    runs = [PairRun("reference", ordinary, None), PairRun(
        pair_id_for(row.name, profile.firm_key), funded, ledger,
        entry_selector=lambda signal, _ns: {
            "quantity": funded.entry_quantities[signal.trade_id],
            "target_decision_needed": row.exit_policy in {
                "gamma_conditional_1r_v1", "early_positive_whole_1r_v1",
            },
        },
        target_selector=_target_selector(funded) if row.exit_policy in {
            "gamma_conditional_1r_v1", "early_positive_whole_1r_v1",
        } else None,
    )]
    daily_activity: list[dict] = []
    saved_stats: dict = {"completed_count": 0}
    if checkpoint is not None:
        for run, state in zip(runs, checkpoint["runs"], strict=True):
            if run.pair_id != state["pair_id"]:
                raise PermissionError("checkpoint stream membership differs from dispatch")
            run.driver.restore(state["driver"])
            if run.ledger is not None:
                run.ledger.restore(state["ledger"])
            run.strategy_trades = state["strategy_trades"]
            run.warmup_trades = state["warmup_trades"]
            run.also_blocked = state["also_blocked"]
        daily_activity, saved_stats = checkpoint["daily_activity"], checkpoint["stats"]
    print_sources = _verify_saved_print_receipts(
        cfg, policy, saved_stats.get("print_source_receipts", []),
    )
    allowed = frozenset(day for registration in policy.registrations
                        for day in registration.source_dates)

    def authorize_print_day(day):
        if day.isoformat() not in allowed or day.isoformat() > "2026-06-10":
            raise PermissionError("funded prints lie outside registered source dates")

    stats = PrintStats(
        cfg.data_dir / cfg.symbol, authorize_source_day=authorize_print_day,
        source_file_observer=print_sources.observe,
    )
    stats.minutes_checked = saved_stats.get("minutes_checked", 0)
    stats.minutes_matched = saved_stats.get("minutes_matched", 0)
    stats.files = {item["file"]: item for item in saved_stats.get("files", [])}
    stats.missing = set(saved_stats.get("missing_utc_days", []))
    evaluation = set(plan.source.evaluation_dates)
    by_day = {day.trading_day: day for day in schedule}
    completed = list(dates[:saved_stats["completed_count"]])
    day_coverage: dict[str, dict] = {}
    for index, day in enumerate(dates[len(completed):], start=len(completed)):
        artifacts = load_registered_day_artifacts(day, cfg, access_policy=policy)
        bars_by_tf: dict[int, list] = {}
        for bar in artifacts.bars:
            bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
        day_coverage[day] = {
            "source_status": "available" if bars_by_tf.get(60) else "unavailable",
        }
        day_input = DayInput(
            day, bars_by_tf, levels_for_from_frame(artifacts.level_timeline),
            day in evaluation, by_day.get(day), index == len(dates) - 1,
        )
        prints = stats.factory(day_input)
        for run in runs:
            before_strategy = len(run.strategy_trades)
            before_funded = len(run.ledger.trades) if run.ledger is not None else 0
            before_payouts = len(run.ledger.payout_events) if run.ledger is not None else 0
            before_decisions = len(run.driver.policy_decisions)
            run_one_day(run, day_input, prints)
            _annotate_new_rows(run, before_strategy, before_funded)
            if day_input.is_evaluation:
                daily_activity.append({
                    **run.driver.last_day_evidence,
                    **day_coverage[day],
                    "stream": "strategy" if run.ledger is None else "funded",
                    "firm_key": None if run.ledger is None else profile.firm_key,
                    "evaluation_date": day,
                    "account_status_at_end": (None if run.ledger is None else
                                              run.ledger.current.status),
                    "payout_events_this_day": (0 if run.ledger is None else
                                               len(run.ledger.payout_events) - before_payouts),
                    "policy_decisions_this_day": len(run.driver.policy_decisions) -
                    before_decisions,
                })
        prints.release()
        saved_stats.update({
            "completed_count": index + 1,
            "minutes_checked": stats.minutes_checked,
            "minutes_matched": stats.minutes_matched,
            "files": list(stats.files.values()),
            "missing_utc_days": sorted(stats.missing),
            "print_source_receipts": list(print_sources.files.values()),
        })
        completed.append(day)
        save_checkpoint(checkpoint_path, {
            "schema": CHECKPOINT_SCHEMA,
            "dispatch_sha256": dispatch_sha,
            "completed_dates": completed,
            "runs": [_roundtrip(run.to_state()) for run in runs],
            "daily_activity": daily_activity,
            "stats": saved_stats,
        })
        if on_day is not None:
            on_day(index, day_input)
    policy.assert_zero_forbidden_access()
    if tuple(completed) != dates:
        raise PermissionError("child did not complete the exact date scope")
    ledger.finish()
    print_sources.recheck_stats()
    output = {
        "configuration": row.name, "batch_id": row.batch_id,
        "display_name": row.display_name,
        "axes": json.loads(row.intent_json),
        "settings_plain": describe_section(section),
        "section_config_hash": row.effective_section_config_hash,
        "effective_behavior_hash": row.effective_behavior_hash,
        "exit_policy": row.exit_policy,
        "sizing": {
            "instrument": row.instrument, "instrument_label": INSTRUMENTS[row.instrument].label,
            "quantity": row.quantity,
            "quantity_policy": row.quantity_policy,
            "possible_quantities": row.possible_quantities,
            "tick_value_cents": INSTRUMENTS[row.instrument].tick_value_cents,
            "cost_per_contract_mills": row.cost_per_contract_mills,
            "fee_rounding_policy": row.fee_rounding_policy,
        },
        "strategy_trades_no_account": runs[0].strategy_trades,
        "pairs": {profile.firm_key: {
            "pair_id": runs[1].pair_id,
            "ledger": _roundtrip(ledger.snapshot()),
            "strategy_trades": runs[1].strategy_trades,
            "entry_candidates_also_blocked_by_strategy": runs[1].also_blocked,
            "forced_flat": funded.forced_flat,
            "discarded_refused_setups": funded.discarded_setups,
            "final_seed_hash": funded.seed_hash(),
            "trades_matching_reference_entries": None,
            "trades_not_in_reference": None,
        }},
        "daily_activity": daily_activity,
        "policy_decisions": {
            "strategy": ordinary.policy_decisions,
            "funded": funded.policy_decisions,
        },
        "dispatch": dispatch,
        "dispatch_sha256": dispatch_sha,
        "completed_dates": completed,
        "reference": {
            "equivalent": None,
            "compared_with": "Same-input policy replay; no old trade equality claim",
            "saved_study_trades": None,
            "replayed_trades": len(runs[0].strategy_trades),
            "warmup_trades": runs[0].warmup_trades,
        },
        "resumed": {},
        "prints": {
            "minutes_checked": stats.minutes_checked,
            "minutes_rebuilt_exactly": stats.minutes_matched,
            "files": list(stats.files.values()),
            "missing_utc_days": sorted(stats.missing),
        },
        "print_source_receipts": list(print_sources.files.values()),
        "access_audit": policy.audit_dict(),
        "seconds": round(time.monotonic() - started, 1),
    }
    save_checkpoint(work_root / "output.json", {
        "schema": FINAL_SCHEMA, "dispatch_sha256": dispatch_sha,
        "completed_dates": completed, "output": output,
    })
    return output


def _run_worker(task: dict) -> dict:
    plan = MffuBatchPlanPayload.model_validate(task["plan"])
    row = plan.variants[int(task["variant_id"][3:]) - 1]
    return run_mffu_configuration(
        plan_id=task["plan_id"], approval_id=task["approval_id"], plan=plan,
        row=row, work_root=Path(task["work_root"]),
    )


def _worker_init_core(core_root: str) -> None:
    """Make the exact task-owned Core importable in spawned Windows workers."""
    source = (Path(core_root) / "src").resolve(strict=True)
    loaded = sys.modules.get("strategy_core")
    if loaded is not None:
        # Windows spawn imports this worker module before invoking its
        # initializer. A transitive import may already have loaded Core from
        # the inherited CLI path; accept it only when it is the frozen source.
        path = getattr(loaded, "__file__", None)
        if path is None or not Path(path).resolve().is_relative_to(source):
            raise PermissionError("worker imported Core outside the frozen source")
    sys.path.insert(0, str(source))


def run_mffu_batch(*, plan_id: str, store_root: Path, state_root: Path,
                   workers: int = 4) -> dict:
    """Run the frozen study once, reusing only source-matched completed children."""
    from alpha_lab.propsim.funded.comparison_result import (
        build_comparison_result,
        validate_comparison,
    )
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result
    from alpha_lab.propsim.funded.full_range_reporting import attach_full_range_reports
    from alpha_lab.propsim.funded.mffu_batch_correction_reuse import (
        collect_print_receipts,
        correction_disposition,
        correction_receipt,
        correction_reuse_annotations,
        try_reuse_correction,
        verify_saved_correction_reuse,
    )
    from alpha_lab.propsim.funded.mffu_batch_reuse import (
        try_reuse_partial,
        verify_saved_reuse,
    )

    store_root, state_root = Path(store_root), Path(state_root)
    envelope, approval = load_approved_mffu_batch_plan(store_root, plan_id)
    plan = envelope.payload
    _sections, cfg, policy, schedule, start_ns, cutoff_ns = validate_mffu_runtime(plan)
    prior = read_state(state_root, plan_id) or {}
    work_root = state_root / plan_id / "workers"
    if prior.get("status") == "Completed" and prior.get("result_id"):
        saved = load_comparison_result(store_root, prior["result_id"])
        result_envelope = load_verified_envelope(
            store_root, RESULT_STORE, prior["result_id"],
            FundedComparisonResultEnvelope,
        )
        _verify_terminal_result_binding(
            saved, result_envelope, plan_id=plan_id,
            approval_id=approval.funded_comparison_approval_id, plan=plan,
        )
        _verify_saved_print_receipts(
            cfg, policy, saved["mffu_batch"]["print_source_receipts"],
        )
        for row in plan.variants:
            if any(item["variant_id"] == row.variant_id and item["status"] in {
                "reused_nonimpact", "reused_equivalent_after_verification",
            } for item in saved["mffu_batch"]["dispositions"]):
                sealed = load_checkpoint(
                    work_root / row.variant_id / "output.json",
                    dispatch_sha256=canonical_contract_sha256(
                        _dispatch(plan_id, saved["mffu_batch"]["approval_id"], plan, row)
                    ), dates=plan.source.warmup_dates + plan.source.evaluation_dates,
                )
                if sealed is None:
                    raise PermissionError("correction reuse output disappeared")
                verify_saved_correction_reuse(plan, row, sealed["output"])
                continue
            if not any(item["variant_id"] == row.variant_id and
                       item["status"] == "compatible_reused"
                       for item in saved["mffu_batch"]["dispositions"]):
                continue
            sealed = load_checkpoint(
                work_root / row.variant_id / "output.json",
                dispatch_sha256=canonical_contract_sha256(
                    _dispatch(plan_id, saved["mffu_batch"]["approval_id"], plan, row)
                ),
                dates=plan.source.warmup_dates + plan.source.evaluation_dates,
            )
            if sealed is None:
                raise PermissionError("reused child output disappeared after publication")
            verify_saved_reuse(plan, row, sealed["output"], work_root / row.variant_id)
        return prior
    approval_id = approval.funded_comparison_approval_id
    write_state(
        state_root, plan_id, status="Running", kind="funded_comparison",
        phase="simulating_mffu_64", pid=os.getpid(), configurations_total=64,
        configurations_done=0, started_at_utc=datetime.now(UTC).isoformat(),
    )
    outputs: list[dict] = []
    failures: list[dict] = []
    tasks: list[dict] = []
    reuse_proofs: dict[str, dict] = {}
    reuse_annotations: list[dict] = []
    correction_print_receipts: dict = {}
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    context_index = None
    for row in plan.variants:
        dispatch = _dispatch(plan_id, approval_id, plan, row)
        saved = load_checkpoint(
            work_root / row.variant_id / "output.json",
            dispatch_sha256=canonical_contract_sha256(dispatch), dates=dates,
        )
        if saved is not None:
            output = saved["output"]
            if (tuple(saved["completed_dates"]) != dates or
                    output["dispatch"] != dispatch or
                    output["section_config_hash"] != row.effective_section_config_hash or
                    output["effective_behavior_hash"] != row.effective_behavior_hash):
                raise PermissionError(f"{row.variant_id}: finished child differs from plan")
            if output.get("correction_reuse"):
                collect_print_receipts(correction_print_receipts, output["print_source_receipts"])
            else:
                _verify_saved_print_receipts(cfg, policy, output["print_source_receipts"])
            if output.get("reuse", {}).get("status") == "compatible_reused":
                proof, sidecar = verify_saved_reuse(
                    plan, row, output, work_root / row.variant_id,
                )
                reuse_proofs[row.variant_id] = proof
                reuse_annotations.extend({
                    "configuration": row.variant_id,
                    "status": "posthoc_v02_not_executed",
                    **record,
                } for record in sidecar["records"])
            if output.get("correction_reuse"):
                reuse_proofs[row.variant_id] = verify_saved_correction_reuse(plan, row, output)
                annotations, _paths = correction_reuse_annotations(plan, row, output)
                reuse_annotations.extend(annotations)
            outputs.append(output)
        else:
            reused = try_reuse_correction(plan_id, approval_id, plan, row, work_root)
            if reused is not None:
                output, proof = reused
                collect_print_receipts(correction_print_receipts, output["print_source_receipts"])
                reuse_proofs[row.variant_id] = proof
                annotations, _paths = correction_reuse_annotations(plan, row, output)
                reuse_annotations.extend(annotations)
                outputs.append(output)
                continue
            if row.variant_id in {"MCB001", "MCB025"}:
                if context_index is None:
                    context_index = _load_context(plan)
                reused = try_reuse_partial(
                    plan_id, approval_id, plan, row, work_root, context_index,
                )
                if reused is not None:
                    output, proof = reused
                    checked, sidecar = verify_saved_reuse(
                        plan, row, output, work_root / row.variant_id,
                    )
                    if checked != proof:
                        raise PermissionError("new reuse proof failed immediate readback")
                    reuse_proofs[row.variant_id] = proof
                    reuse_annotations.extend({
                        "configuration": row.variant_id,
                        "status": "posthoc_v02_not_executed",
                        **record,
                    } for record in sidecar["records"])
                    outputs.append(output)
                    continue
            tasks.append({
                "plan_id": plan_id, "approval_id": approval_id,
                "plan": plan.model_dump(mode="json"), "variant_id": row.variant_id,
                "work_root": str(work_root),
            })
    if correction_print_receipts:
        _verify_saved_print_receipts(cfg, policy, list(correction_print_receipts.values()))
    write_state(state_root, plan_id, configurations_done=len(outputs),
                phase="simulating_mffu_64")
    with ProcessPoolExecutor(
        max_workers=max(1, min(workers, 8)),
        initializer=_worker_init_core,
        initargs=(plan.core_root,),
    ) as pool:
        futures = {pool.submit(_run_worker, task): task["variant_id"] for task in tasks}
        for future in as_completed(futures):
            variant_id = futures[future]
            try:
                outputs.append(future.result())
            except Exception as error:
                failures.append({
                    "configuration": variant_id, "batch_id": variant_id,
                    "display_name": variant_id,
                    "reason": f"{type(error).__name__}: {error}",
                })
            write_state(
                state_root, plan_id, configurations_done=len(outputs) + len(failures),
                configurations_failed=len(failures),
            )
    outputs.sort(key=lambda output: output["batch_id"])
    by_completed = {output["batch_id"]: output for output in outputs}
    dispositions = [{
        "variant_id": row.variant_id,
        "status": (correction_disposition(plan, row, by_completed[row.variant_id])
                   if row.variant_id in by_completed else "failed"),
        "reason": next((f["reason"] for f in failures if f["batch_id"] == row.variant_id), None),
        "reused_from": (
            by_completed[row.variant_id]["correction_reuse"]["original_result_id"]
            if (row.variant_id in by_completed
                and by_completed[row.variant_id].get("correction_reuse")) else
            by_completed[row.variant_id]["reuse"]["reference_batch_id"]
            if row.variant_id in by_completed and by_completed[row.variant_id].get("reuse")
            else None
        ),
    } for row in plan.variants]
    context = {
        "funded_comparison_plan_id": plan_id, "purpose": plan.purpose,
        "question": plan.question,
        "run_identity": {
            "funded_comparison_plan_id": plan_id,
            "engine_version": ENGINE_VERSION,
            "core_source": plan.core_source.model_dump(mode="json"),
        },
        "approval": {
            "approved_on": approval.payload.approved_on,
            "channel": approval.payload.channel,
            "scope": approval.payload.scope,
        },
        "source": {
            "title": plan.source.title,
            "evaluation_first_day": plan.source.evaluation_dates[0],
            "evaluation_last_day": plan.source.evaluation_dates[-1],
            "evaluation_days": 253, "warmup_days": 10,
        },
        "owner_decisions": [d.model_dump(mode="json") for d in plan.owner_decisions],
        "limitations": list(plan.limitations),
    }
    settings = {
        "size_text": "Ten or six micros selected at each entry by the frozen policy",
        "sizing_by_configuration": {
            row.name: {
                "instrument": row.instrument, "quantity": row.quantity,
                "possible_quantities": row.possible_quantities,
                "quantity_policy": row.quantity_policy,
                "tick_value_cents": INSTRUMENTS[row.instrument].tick_value_cents,
                "cost_per_contract_per_fill_usd": row.cost_per_contract_mills / 1000,
                "fee_rounding_policy": row.fee_rounding_policy,
                "exit_policy": row.exit_policy,
            } for row in plan.variants
        },
        "firm_profiles": [p.model_dump(mode="json") for p in plan.firm_profiles],
        "processing_clock": plan.processing.model_dump(mode="json"),
        "execution_model": plan.execution_model.model_dump(mode="json"),
        "core_source": plan.core_source.model_dump(mode="json"),
    }
    result = build_comparison_result(
        context=context, outputs=outputs, failures=failures,
        profiles=plan.firm_profiles, trading_days=schedule,
        start_ns=start_ns, cutoff_ns=cutoff_ns, settings=settings,
        resume_check_requested=False,
    )
    result["validation"] = validate_comparison(
        result, outputs, plan.firm_profiles, cutoff_ns,
        require_resume_check=False, require_reference_check=False,
    )
    result["mffu_batch"] = {
        "schema": RESULT_SCHEMA,
        "plan": plan.model_dump(mode="json"),
        "approval_id": approval_id,
        "dispositions": dispositions,
        "reuse_proofs": reuse_proofs,
        "correction_lineage": correction_receipt(plan),
        "reuse_context_annotations": reuse_annotations,
        "worker_dispatches": [o["dispatch"] for o in outputs],
        "decision_context": [
            {
                "configuration": output["configuration"],
                "stream": stream,
                "firm_key": None if stream == "strategy" else plan.firm_profiles[0].firm_key,
                **{key: decision.get(key) for key in (
                    "event", "setup_id", "trade_id", "policy", "action", "context",
                    "reasons", "distance_ticks", "fallback_reason", "overhead",
                    "quota_before", "quota_after",
                )},
            }
            for output in outputs
            for stream, decisions in output.get("policy_decisions", {}).items()
            for decision in decisions
        ],
        "print_source_receipts": list({
            item["path"]: item for output in outputs
            for item in output["print_source_receipts"]
        }.values()),
        "failed_configurations": failures,
        "input_metadata_access_audit": policy.audit_dict(),
    }
    result = attach_full_range_reports(
        result, outputs, failures,
        configuration_names=tuple(row.name for row in plan.variants),
        evaluation_dates=plan.source.evaluation_dates,
        warmup_dates=plan.source.warmup_dates,
        cutoff_utc=plan.source.cutoff_utc,
        expected_configuration_count=64, expected_firm_count=1,
    )
    from alpha_lab.propsim.funded.mffu_batch_analysis import analyze_mffu_batch
    from alpha_lab.propsim.funded.mffu_batch_plan import HANDOFF_ROOT, _sha_bytes

    with zipfile.ZipFile(plan.handoff_zip) as handoff:
        pairs_bytes = handoff.read(HANDOFF_ROOT + "COMPARISON_PAIRS.csv")
    if _sha_bytes(pairs_bytes) != plan.source_member_sha256["COMPARISON_PAIRS.csv"]:
        raise PermissionError("declared comparison pair source changed")
    comparison_pairs = list(csv.DictReader(io.StringIO(pairs_bytes.decode("utf-8-sig"))))
    result["mffu_analysis"] = analyze_mffu_batch(
        result, variants=plan.variants, comparison_pairs=comparison_pairs,
        evaluation_dates=plan.source.evaluation_dates,
    )
    result["validation"]["checks"].update({
        "exactly_64_dispositions": len(dispositions) == 64,
        "exactly_64_mffu_rows": len(result["tables"]["pair_results"]) == 64,
        "complete_date_membership": all(tuple(o["completed_dates"]) == dates for o in outputs),
        "every_configuration_completed": not failures,
    })
    result["validation"]["configurations_not_completed"] = failures
    result["validation"]["passed"] = bool(
        result["validation"]["passed"] and
        result["validation"]["checks"]["exactly_64_dispositions"] and
        result["validation"]["checks"]["exactly_64_mffu_rows"] and
        result["validation"]["checks"]["complete_date_membership"] and
        result["full_range_reporting"]["validation"]["passed"] and not failures
    )
    payload = FundedComparisonResultPayload(
        funded_comparison_plan_id=plan_id,
        funded_comparison_approval_id=approval_id,
        result_json_sha256=result_sha256(result),
        validation_passed=result["validation"]["passed"],
        engine_version=ENGINE_VERSION,
    )
    saved_envelope = FundedComparisonResultEnvelope.from_payload(payload)
    result_id = saved_envelope.funded_comparison_result_id
    if not has_envelope(store_root, RESULT_STORE, result_id):
        from alpha_lab.propsim.funded.comparison_runner import RESULT_SIDECAR

        save_envelope_immutable(
            store_root, RESULT_STORE, saved_envelope,
            extra_files={RESULT_SIDECAR: canonical_json(result).encode()},
        )
    load_comparison_result(store_root, result_id)
    append_ledger(store_root, {
        "event_id": f"mffu_64_{plan_id}_{result_id}_saved",
        "event_type": "run_completed",
        "funded_comparison_plan_id": plan_id,
        "funded_comparison_result_id": result_id,
        "mode": plan.mode,
        "status": "completed" if not failures else "incomplete",
        "configurations_completed": len(outputs),
        "configurations_not_completed": failures,
    })
    return write_state(
        state_root, plan_id, result_id=result_id,
        status="Completed" if payload.validation_passed else "Incomplete",
        phase="completed_unpublished", configurations_done=64,
        configurations_failed=len(failures),
        completed_at_utc=datetime.now(UTC).isoformat(), review_folder=None,
        review_error=None,
    )
