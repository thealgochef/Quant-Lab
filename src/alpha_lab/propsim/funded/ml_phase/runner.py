"""Approved, day-checkpointed shadows and matched fresh-account operations."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import replace
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.day_artifacts import levels_for_from_frame
from alpha_lab.agents.data_infra.ifvg.prepared_store import load_registered_day_artifacts
from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
from alpha_lab.propsim.funded.comparison_run import PrintStats
from alpha_lab.propsim.funded.full_range_batch import (
    _roundtrip,
    _verify_saved_print_receipts,
    load_checkpoint,
    save_checkpoint,
)
from alpha_lab.propsim.funded.mffu_batch_plan import MffuBatchPlanEnvelope
from alpha_lab.propsim.funded.mffu_batch_run import _load_context, resolve_mffu_registered_inputs
from alpha_lab.propsim.funded.pair_engine import DayInput
from alpha_lab.propsim.funded.pair_ledger import PairLedger

from .benchmark import OnlineScorer, fit_benchmark, save_json
from .plan import load_approved_phase_plan
from .protocol import digest
from .stream import PhaseStream


def shadow_binding(envelope, approval, store, work):
    receipt = Path(envelope.payload.runtime_root) / "shadow_reuse.json"
    if not receipt.exists():
        return envelope, approval, work
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
    from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
    from alpha_lab.propsim.funded.comparison_runner import find_approval

    from .plan import MlPhasePlanEnvelope, verify_phase_plan
    from .runtime import sha_file

    if str(receipt) not in dict(envelope.payload.bound_files):
        raise PermissionError("unbound shadow reuse receipt")
    reuse = json.loads(receipt.read_text(encoding="utf-8"))
    old = load_verified_envelope(store, PLAN_STORE, reuse["plan_id"], MlPhasePlanEnvelope)
    verify_phase_plan(old.payload, imported=False)
    old_approval = find_approval(store, reuse["plan_id"])
    if old_approval is None or old.payload.contracts_json != envelope.payload.contracts_json:
        raise PermissionError("shadow reuse approval or contracts differ")
    for path, expected in reuse["files"].items():
        if sha_file(Path(path)) != expected:
            raise PermissionError("reused shadow changed")
    return old, old_approval, Path(reuse["work_root"])


def resolved_inputs(plan):
    inherited = MffuBatchPlanEnvelope.model_validate_json(
        Path(plan.reference_plan_path).read_text(encoding="utf-8")
    ).payload
    sections, cfg, access, schedule, _, _ = resolve_mffu_registered_inputs(inherited)
    return inherited, sections, cfg, access, schedule, _load_context(inherited)


def _prints(cfg, access, saved):
    receipts = _verify_saved_print_receipts(cfg, access, saved.get("print_source_receipts", []))
    allowed = frozenset(
        day for registration in access.registrations for day in registration.source_dates
    )

    def authorize(day):
        if day.isoformat() not in allowed or day.isoformat() > "2026-06-10":
            raise PermissionError("ML phase print date is outside registered source scope")

    stats = PrintStats(
        cfg.data_dir / cfg.symbol,
        authorize_source_day=authorize,
        source_file_observer=receipts.observe,
    )
    stats.minutes_checked = saved.get("minutes_checked", 0)
    stats.minutes_matched = saved.get("minutes_matched", 0)
    stats.files = {row["file"]: row for row in saved.get("files", [])}
    stats.missing = set(saved.get("missing_utc_days", []))
    return stats, receipts


def _stats(stats, receipts):
    return {
        "minutes_checked": stats.minutes_checked,
        "minutes_matched": stats.minutes_matched,
        "files": list(stats.files.values()),
        "missing_utc_days": sorted(stats.missing),
        "print_source_receipts": list(receipts.files.values()),
    }


def _day(day, cfg, access, schedule, evaluation, exhausted):
    artifacts = load_registered_day_artifacts(day, cfg, access_policy=access)
    bars = {}
    for bar in artifacts.bars:
        bars.setdefault(bar.timeframe_ticks, []).append(bar)
    return DayInput(
        day,
        bars,
        levels_for_from_frame(artifacts.level_timeline),
        day in evaluation,
        schedule[day],
        exhausted,
    )


def run_shadow(
    *, store: Path, plan_id: str, reference: str, work: Path, stop_after_days: int | None = None
):
    envelope, approval = load_approved_phase_plan(store, plan_id)
    envelope, approval, work = shadow_binding(envelope, approval, store, work)
    plan_id = envelope.funded_comparison_plan_id
    contracts = json.loads(envelope.payload.contracts_json)
    inherited, sections, cfg, access, _, context = resolved_inputs(envelope.payload)
    section = sections[int(reference[3:]) - 1]
    cfg = replace(cfg, section=section)
    dates = inherited.source.warmup_dates + inherited.source.evaluation_dates
    calendar, _, _ = _calendar(section, dates)
    schedule = {d.trading_day: d for d in calendar}
    evaluation = set(inherited.source.evaluation_dates)
    dispatch = digest(
        {
            "plan_id": plan_id,
            "approval_id": approval.funded_comparison_approval_id,
            "reference": reference,
            "kind": "fixed_shadow",
        }
    )
    folder = work / "shadows" / reference
    terminal = folder / "output.json"
    saved = load_checkpoint(
        terminal if terminal.exists() else folder / "checkpoint.json",
        dispatch_sha256=dispatch,
        dates=dates,
    )
    stream = PhaseStream(
        reference=reference,
        stream_id=f"SHADOW_{reference}",
        section=section,
        context_index=context,
        definitions=contracts["FEATURE_DEFINITIONS"]["rows"],
        source_id=plan_id,
    )
    completed, stat_record = [], {}
    if saved:
        stream.restore(saved["stream"])
        completed, stat_record = saved["completed_dates"], saved["stats"]
    stats, receipts = _prints(cfg, access, stat_record)
    if terminal.exists():
        if tuple(completed) != dates:
            raise PermissionError("shadow terminal does not cover the complete date scope")
        return saved
    first_scored = contracts["DATE_AND_FOLD_PLAN"]["scored_dates"][0]
    started = time.monotonic()
    for index in range(len(completed), len(dates)):
        day = dates[index]
        if day == first_scored:
            seed = stream.driver.checkpoint()
            seed_hash = stream.driver.seed_hash()
            if stream.position is not None:
                raise PermissionError("scored-period seed carries shadow exposure")
            save_checkpoint(
                folder / "scored_seed.json",
                {
                    "dispatch_sha256": dispatch,
                    "completed_dates": completed,
                    "driver": seed,
                    "seed_hash": seed_hash,
                    "first_scored_date": first_scored,
                    "no_account": True,
                    "no_position": True,
                },
            )
        inputs = _day(day, cfg, access, schedule, evaluation, index == len(dates) - 1)
        prints = stats.factory(inputs)
        stream.run_day(inputs, prints)
        prints.release()
        completed.append(day)
        payload = {
            "dispatch_sha256": dispatch,
            "completed_dates": completed,
            "stream": _roundtrip(stream.snapshot()),
            "stats": _stats(stats, receipts),
            "access_audit": access.audit_dict(),
            "elapsed_seconds": time.monotonic() - started,
        }
        save_checkpoint(folder / "checkpoint.json", payload)
        print(
            f"shadow {reference}: {index + 1}/{len(dates)} {day}; "
            f"entries={len(stream.datasets['ENTRY'])} "
            f"checkpoints={len(stream.datasets['CONTINUATION'])}",
            flush=True,
        )
        if stop_after_days is not None and index + 1 >= stop_after_days:
            return payload
    access.assert_zero_forbidden_access()
    receipts.recheck_stats()
    save_checkpoint(terminal, payload)
    return payload


def fit_all(*, store: Path, plan_id: str, work: Path):
    envelope, _ = load_approved_phase_plan(store, plan_id)
    contracts = json.loads(envelope.payload.contracts_json)
    inherited, sections, _, _, _, _ = resolved_inputs(envelope.payload)
    shadows = {
        r: run_shadow(store=store, plan_id=plan_id, reference=r, work=work)
        for r in ("MCB062", "MCB025")
    }
    datasets = {r: value["stream"]["datasets"] for r, value in shadows.items()}
    boundaries = {}
    section = sections[24]
    for fold in contracts["DATE_AND_FOLD_PLAN"]["folds"]:
        _, _, cutoff = _calendar(section, fold["train_candidate_dates"])
        _, start, end = _calendar(section, fold["test_dates"])
        boundaries[fold["fold_id"]] = {
            "cutoff_ns": cutoff,
            "test_start_ns": start,
            "test_end_ns": end,
        }
    identity = {
        "plan_id": plan_id,
        "datasets_sha256": digest(datasets),
        "environment": json.loads(envelope.payload.environment_json),
    }
    output = work / "benchmark" / "benchmark.json"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise PermissionError("saved benchmark belongs to a different source or dataset")
        for record in existing["fits"]:
            if record["model_id"]:
                from .models import RegressionFit

                fit = RegressionFit.load(work / "benchmark" / record["fit_path"])
                if fit.state["fit_id"] != record["model_id"]:
                    raise PermissionError("benchmark fit identity mismatch")
        return existing
    result = fit_benchmark(
        datasets=datasets,
        contracts=contracts,
        boundaries=boundaries,
        identity=identity,
        folder=work / "benchmark",
    )
    save_json(output, result)
    return result


def run_policies(
    *, store: Path, plan_id: str, reference: str, work: Path, stop_after_days: int | None = None
):
    envelope, approval = load_approved_phase_plan(store, plan_id)
    contracts = json.loads(envelope.payload.contracts_json)
    inherited, sections, cfg, access, _, context = resolved_inputs(envelope.payload)
    section = sections[int(reference[3:]) - 1]
    cfg = replace(cfg, section=section)
    dates = tuple(contracts["DATE_AND_FOLD_PLAN"]["scored_dates"])
    calendar, start, cutoff = _calendar(section, dates)
    schedule = {d.trading_day: d for d in calendar}
    benchmark = json.loads((work / "benchmark/benchmark.json").read_text(encoding="utf-8"))
    if benchmark["identity"]["plan_id"] != plan_id:
        raise PermissionError("models belong to a different approved plan")
    shadow_plan, shadow_approval, shadow_work = shadow_binding(envelope, approval, store, work)
    shadow_dispatch = digest(
        {
            "plan_id": shadow_plan.funded_comparison_plan_id,
            "approval_id": shadow_approval.funded_comparison_approval_id,
            "reference": reference,
            "kind": "fixed_shadow",
        }
    )
    all_dates = inherited.source.warmup_dates + inherited.source.evaluation_dates
    seed = load_checkpoint(
        shadow_work / "shadows" / reference / "scored_seed.json",
        dispatch_sha256=shadow_dispatch,
        dates=all_dates,
    )
    if not seed or not seed["no_position"] or seed["first_scored_date"] != dates[0]:
        raise PermissionError("a verified flat causal pre-score seed is required")
    dispatch = digest(
        {
            "plan_id": plan_id,
            "approval_id": approval.funded_comparison_approval_id,
            "reference": reference,
            "kind": "matched_policies",
            "seed_hash": seed["seed_hash"],
            "benchmark": digest(benchmark),
        }
    )
    folder = work / "operations" / reference
    terminal = folder / "output.json"
    saved = load_checkpoint(
        terminal if terminal.exists() else folder / "checkpoint.json",
        dispatch_sha256=dispatch,
        dates=dates,
    )
    cells = [c for c in contracts["MODEL_MATRIX"]["cells"] if c["reference"] == reference]
    streams = {}
    for cell in [None, *cells]:
        name = f"NO_ML_{reference}" if cell is None else cell["cell_id"]
        ledger = PairLedger(
            pair_id=f"{name}__myfundedfutures",
            configuration=name,
            profile=inherited.firm_profiles[0],
            processing=inherited.processing,
            quantity=10,
            tick_value_cents=50,
            cost_per_side_cents=0,
            cost_per_contract_mills=514,
            trading_days=calendar,
            start_ns=start,
            cutoff_ns=cutoff,
            scale_out=True,
        )
        scorer = (
            None if cell is None else OnlineScorer(cell, benchmark, work / "benchmark", contracts)
        )
        stream = PhaseStream(
            reference=reference,
            stream_id=name,
            section=section,
            context_index=context,
            definitions=contracts["FEATURE_DEFINITIONS"]["rows"],
            source_id=shadow_plan.funded_comparison_plan_id,
            ledger=ledger,
            scorer=scorer,
        )
        initial = copy.deepcopy(seed["driver"])
        for field in ("entry_contexts", "entry_quantities", "core_exit_annotations", "ml_entries"):
            initial[field] = {}
        initial["policy_decisions"], initial["ml_decisions"] = [], []
        stream.driver.restore(initial)
        if stream.driver.seed_hash() != seed["seed_hash"]:
            raise PermissionError("policy initialization changed the common strategy seed")
        streams[name] = stream
    completed, stat_record = [], {}
    if saved:
        if set(saved["streams"]) != set(streams):
            raise PermissionError("checkpoint policy membership changed")
        for name, state in saved["streams"].items():
            streams[name].restore(state)
        completed, stat_record = saved["completed_dates"], saved["stats"]
    stats, receipts = _prints(cfg, access, stat_record)
    if terminal.exists():
        if tuple(completed) != dates:
            raise PermissionError("operation terminal date scope is incomplete")
        return saved
    started = time.monotonic()
    for index in range(len(completed), len(dates)):
        day = dates[index]
        inputs = _day(day, cfg, access, schedule, set(dates), index == len(dates) - 1)
        prints = stats.factory(inputs)
        for stream in streams.values():
            stream.run_day(inputs, prints)
        prints.release()
        completed.append(day)
        payload = {
            "dispatch_sha256": dispatch,
            "completed_dates": completed,
            "streams": {name: _roundtrip(stream.snapshot()) for name, stream in streams.items()},
            "stats": _stats(stats, receipts),
            "access_audit": access.audit_dict(),
            "seed_hash": seed["seed_hash"],
            "start_ns": start,
            "cutoff_ns": cutoff,
            "elapsed_seconds": time.monotonic() - started,
        }
        save_checkpoint(folder / "checkpoint.json", payload)
        print(
            f"policies {reference}: {index + 1}/{len(dates)} {day}; 13 independent accounts",
            flush=True,
        )
        if stop_after_days is not None and index + 1 >= stop_after_days:
            return payload
    for stream in streams.values():
        stream.ledger.finish()
    payload["streams"] = {name: _roundtrip(stream.snapshot()) for name, stream in streams.items()}
    access.assert_zero_forbidden_access()
    receipts.recheck_stats()
    save_checkpoint(terminal, payload)
    return payload
