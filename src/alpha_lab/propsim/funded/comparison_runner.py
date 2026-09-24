"""Worker-side execution of one frozen configuration-comparison plan.

Reads ONLY the verified plan envelope (and, for a historical run, its stored
owner approval). Re-verifies the strategy source and every configuration's
resolution, runs each configuration in its own process (all selected firms,
the no-account reference replay and the resumed-state check), builds and
validates the one result, saves it immutably, appends the cumulative research
ledger and publishes the review folder. A configuration that cannot complete is
recorded as not completed with its reason; it is never shown as zero.
Progress is written to ``<state_root>/<plan_id>/state.json``.
"""

from __future__ import annotations

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_json_sidecar,
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.propsim.funded.clock import chicago_label, to_ns
from alpha_lab.propsim.funded.comparison_plan import (
    APPROVAL_STORE,
    PLAN_STORE,
    RESULT_STORE,
    FundedComparisonApprovalEnvelope,
    FundedComparisonPlanEnvelope,
    FundedComparisonPlanEnvelopeV2,
    FundedComparisonPlanPayload,
    FundedComparisonPlanPayloadV2,
    FundedComparisonResultEnvelope,
    FundedComparisonResultPayload,
)
from alpha_lab.propsim.funded.comparison_result import (
    apply_reporting_corrections,
    build_comparison_result,
    validate_comparison,
)
from alpha_lab.propsim.funded.comparison_source import (
    ComparisonConfiguration,
    ComparisonSource,
    open_comparison_source,
    resolve_configuration,
)
from alpha_lab.propsim.funded.position_walk import EXECUTION_MODEL_TEXT
from alpha_lab.propsim.funded.price_evidence import DATA_ROOT
from alpha_lab.propsim.funded.profiles import INSTRUMENTS
from alpha_lab.propsim.funded.result import canonical_json, result_sha256
from alpha_lab.propsim.funded.runner import _now, append_ledger, read_ledger, write_state
from alpha_lab.propsim.funded.sources import ARCHIVE_ROOT

__all__ = [
    "ENGINE_VERSION",
    "RESULT_SIDECAR",
    "find_approval",
    "open_plan_source",
    "run_comparison_plan",
    "load_comparison_result",
    "republish_comparison_review",
    "review_supplements",
    "publish_comparison_review",
]

ENGINE_VERSION = "funded_comparison_engine_v1"
#: parallel configuration processes when none is given (bounded for memory; each
#: process holds one configuration's day inputs, measured at well under 2 GB)
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 4))
RESULT_SIDECAR = "result.json"


def find_approval(store_root: Path, plan_id: str) -> FundedComparisonApprovalEnvelope | None:
    folder = Path(store_root) / APPROVAL_STORE
    if not folder.is_dir():
        return None
    for entry in sorted(folder.iterdir()):
        try:
            envelope = load_verified_envelope(store_root, APPROVAL_STORE, entry.name,
                                              FundedComparisonApprovalEnvelope)
        except Exception:
            continue
        if envelope.payload.funded_comparison_plan_id == plan_id:
            return envelope
    return None


def load_plan(store_root: Path, plan_id: str
              ) -> FundedComparisonPlanPayload | FundedComparisonPlanPayloadV2:
    try:
        return load_verified_envelope(store_root, PLAN_STORE, plan_id,
                                      FundedComparisonPlanEnvelope).payload
    except Exception:
        return load_verified_envelope(store_root, PLAN_STORE, plan_id,
                                      FundedComparisonPlanEnvelopeV2).payload


def is_v2(plan) -> bool:
    return getattr(plan, "plan_schema", "") == "funded_comparison_plan_v2"


def check_core_source(plan) -> None:
    """A v2 plan runs only on the exact Strategy-Core source it froze."""

    if not is_v2(plan):
        return
    from alpha_lab.propsim.funded.core_identity import core_source_identity

    actual = core_source_identity()
    wanted = plan.core_source
    if (actual["base_commit"] != wanted.base_commit
            or actual["patch_sha256"] != wanted.patch_sha256):
        raise PermissionError("the imported Strategy-Core source differs from the frozen plan")


def variant_configuration(source: ComparisonSource, variant) -> ComparisonConfiguration:
    """The v2 variant as a comparison configuration, re-resolved and checked."""

    ids = dict(variant.axis_value_ids)
    if variant.in_verified_study:
        config = source.by_name.get(variant.name)
        if config is None or config.axis_value_ids != ids:
            raise ValueError(f"configuration {variant.name} differs from the verified study")
        return config
    base = source.by_name.get(variant.cache_configuration)
    if base is None:
        raise ValueError(f"cache configuration {variant.cache_configuration} is unavailable")
    _section, cfg = resolve_configuration(ids)
    if cfg.profile_hash != variant.resolved_section_config_hash:
        raise ValueError(f"configuration {variant.name} resolves differently from the plan")
    return ComparisonConfiguration(
        name=variant.name, display_name=variant.display_name, axes={}, axis_value_ids=ids,
        resolved_section_config_hash=variant.resolved_section_config_hash,
        approval_id=base.approval_id, core_replay_id="")


def open_plan_source(plan, archive_root: Path | None = None) -> ComparisonSource:
    archive_root = ARCHIVE_ROOT if archive_root is None else Path(archive_root)
    source = None
    for candidate in archive_root.glob(f"*/{plan.source.package_root_name}"):
        source = open_comparison_source(candidate)
    if source is None or source.package.run_id != plan.source.package_run_id:
        raise ValueError("the plan's verified strategy study is unavailable")
    if source.package.manifest_sha256 != plan.source.package_manifest_sha256:
        raise ValueError("the strategy study changed after the plan was frozen")
    if (source.warmup_dates != plan.source.warmup_dates
            or source.evaluation_dates != plan.source.evaluation_dates):
        raise ValueError("the study dates differ from the frozen plan")
    known = source.by_name
    if is_v2(plan):
        for variant in plan.variants:
            variant_configuration(source, variant)
        return source
    for ref in plan.configurations:
        config = known.get(ref.name)
        if config is None:
            raise ValueError(f"configuration {ref.name} is not in the verified study")
        if (dict(ref.axis_value_ids) != config.axis_value_ids
                or ref.resolved_section_config_hash != config.resolved_section_config_hash
                or ref.approval_id != config.approval_id):
            raise ValueError(f"configuration {ref.name} differs from the verified study")
    return source


def _run_one(task: dict[str, Any]) -> dict[str, Any]:
    """Process entry point (top level so it can be spawned on Windows)."""

    from alpha_lab.propsim.funded.comparison_run import run_configuration

    data_root = None if task["data_root"] is None else Path(task["data_root"])
    if task["plan"].get("plan_schema") == "funded_comparison_plan_v2":
        plan = FundedComparisonPlanPayloadV2.model_validate(task["plan"])
        check_core_source(plan)
        source = open_plan_source(plan, task["archive_root"])
        variant = next(v for v in plan.variants if v.name == task["configuration"])
        return run_configuration(
            source=source, configuration=variant_configuration(source, variant),
            profiles=plan.firm_profiles, instrument=variant.instrument,
            quantity=variant.quantity, cost_per_side_cents=0, processing=plan.processing,
            data_root=data_root, check_resume=True,
            cost_per_contract_mills=variant.cost_per_contract_mills,
            cache_configuration=(None if variant.in_verified_study
                                 else source.by_name[variant.cache_configuration]))
    plan = FundedComparisonPlanPayload.model_validate(task["plan"])
    source = open_plan_source(plan, task["archive_root"])
    configuration = source.by_name[task["configuration"]]
    return run_configuration(
        source=source, configuration=configuration, profiles=plan.firm_profiles,
        instrument=plan.instrument, quantity=plan.quantity,
        cost_per_side_cents=plan.cost_per_side_cents, processing=plan.processing,
        data_root=data_root, check_resume=True)


def load_comparison_result(store_root: Path, result_id: str) -> dict[str, Any]:
    envelope = load_verified_envelope(store_root, RESULT_STORE, result_id,
                                      FundedComparisonResultEnvelope)
    result = load_json_sidecar(store_root, RESULT_STORE, result_id, RESULT_SIDECAR)
    if result_sha256(result) != envelope.payload.result_json_sha256:
        raise ValueError("the saved comparison result does not match its verified hash")
    # the saved bytes stay immutable; summary-only reporting corrections are
    # re-derived from the saved rows and recorded on the loaded copy
    return apply_reporting_corrections(result)


def review_supplements(plan, source: ComparisonSource, result: dict[str, Any], *,
                       plan_id: str) -> dict[str, Any]:
    """Exact bindings, the saved calendar and the trade-boundary check for the export."""

    from alpha_lab.propsim.funded.comparison_evidence import (
        configuration_bindings,
        trade_boundary_check,
        trading_calendar_rows,
    )

    return {
        "configuration_bindings": configuration_bindings(plan, source, plan_id=plan_id),
        "trading_calendar": trading_calendar_rows(source.trading_days, source.start_ns),
        "trade_boundary_check": trade_boundary_check(
            (result.get("tables") or {}).get("trades") or [], source.trading_days,
            source.start_ns),
    }


def _publish(result: dict[str, Any], result_id: str, store_root: Path,
             reports_root: Path, *, supplements: dict[str, Any] | None = None,
             review_findings: str | None = None):
    """Publish the next export version; (path, error, receipt). Never recomputes money."""

    try:
        from alpha_lab.agents.data_infra.ifvg.funded_comparison_review import (
            next_comparison_export_version,
            publish_comparison_review_folder,
        )

        published = publish_comparison_review_folder(
            result=result, result_id=result_id, reports_root=reports_root,
            ledger_entries=read_ledger(store_root),
            export_version=next_comparison_export_version(reports_root, result_id),
            review_findings=review_findings, supplements=supplements)
    except Exception as error:  # the economic result stays saved and verified
        return None, f"{type(error).__name__}: {error}", None
    return str(published.path), None, published.receipt


def republish_comparison_review(*, plan_id: str, store_root: Path, state_root: Path,
                                reports_root: Path, review_findings: str | None = None,
                                extra_supplements: dict[str, Any] | None = None,
                                archive_root: Path | None = None) -> dict[str, Any]:
    """A NEW export version of a completed, verified result (the result is not recomputed).

    Adds the exact bindings, saved calendar and trade-boundary check; the loaded
    result carries its recorded reporting corrections. The plan's frozen
    Strategy-Core source must be the one imported (bindings re-resolve through the
    normal configurator path).
    """

    from alpha_lab.propsim.funded.runner import read_state

    store_root, state_root = Path(store_root), Path(state_root)
    state = read_state(state_root, plan_id) or {}
    if state.get("status") != "Completed" or not state.get("result_id"):
        raise ValueError("only a completed, verified comparison can be published again")
    result = load_comparison_result(store_root, state["result_id"])
    plan = load_plan(store_root, plan_id)
    check_core_source(plan)
    source = open_plan_source(plan, archive_root)
    supplements = {**review_supplements(plan, source, result, plan_id=plan_id),
                   **(extra_supplements or {})}
    path, error, receipt = _publish(result, state["result_id"], store_root,
                                    Path(reports_root), supplements=supplements,
                                    review_findings=review_findings)
    if path is None:
        raise RuntimeError(f"the review folder was not published: {error}")
    append_ledger(store_root, {
        "event_id": f"funded_comparison_{plan_id[:16]}_review_published_{Path(path).name}",
        "event_type": "review_folder_published", "funded_comparison_plan_id": plan_id,
        "funded_comparison_result_id": state["result_id"], "status": "published",
        "reporting_corrections": [r.get("correction_id")
                                  for r in result.get("reporting_corrections") or []],
        "payload_files_verified": (receipt or {}).get("payloads_verified"),
        "note": "new export version of the same saved result; the economic result is "
                "unchanged",
        "display_time_chicago": chicago_label(to_ns(_now())),
    })
    write_state(state_root, plan_id, review_folder=path, review_error=None,
                phase="review_published")
    return {"review_folder": path, "receipt": receipt}


def publish_comparison_review(*, plan_id: str, store_root: Path, state_root: Path,
                              reports_root: Path) -> dict[str, Any]:
    """Retry publication for a completed, verified result (never recomputed)."""

    from alpha_lab.propsim.funded.runner import read_state

    state = read_state(state_root, plan_id) or {}
    if state.get("status") != "Completed" or not state.get("result_id"):
        raise ValueError("only a completed, verified comparison can be published")
    if state.get("review_folder"):
        return state
    result = load_comparison_result(Path(store_root), state["result_id"])
    path, error, _receipt = _publish(result, state["result_id"], Path(store_root),
                                     Path(reports_root))
    if path:
        append_ledger(Path(store_root), {
            "event_id": f"funded_comparison_{plan_id[:16]}_review_published_{Path(path).name}",
            "event_type": "review_folder_published", "funded_comparison_plan_id": plan_id,
            "funded_comparison_result_id": state["result_id"], "status": "published",
            "note": "review folder published again; the economic result is unchanged",
        })
    return write_state(Path(state_root), plan_id, review_folder=path, review_error=error,
                       phase="review_published" if path else "review_failed")


def run_comparison_plan(*, plan_id: str, store_root: Path, state_root: Path,
                        reports_root: Path, archive_root: Path | None = None,
                        data_root: Path | None = DATA_ROOT, workers: int | None = None,
                        require_approval: bool = True,
                        run_config_fn: Any = None) -> dict[str, Any]:
    """``run_config_fn(task)`` replaces the per-configuration process (tests only)."""

    store_root, state_root = Path(store_root), Path(state_root)
    write_state(state_root, plan_id, status="Running", phase="loading_plan",
                pid=os.getpid(), started_at_utc=_now(), kind="funded_comparison")
    try:
        plan = load_plan(store_root, plan_id)
        check_core_source(plan)
        approval = find_approval(store_root, plan_id)
        if plan.purpose == "historical_comparison" and approval is None and require_approval:
            raise PermissionError("this exact comparison plan has no stored owner approval")
        source = open_plan_source(plan, archive_root)
        names = [c.name for c in plan.configurations]
        append_ledger(store_root, {
            "event_id": f"funded_comparison_{plan_id[:16]}_started",
            "event_type": "run_started", "funded_comparison_plan_id": plan_id,
            "mode": plan.mode, "purpose": plan.purpose, "question": plan.question,
            "configurations": names, "firms": [p.firm_name for p in plan.firm_profiles],
            "size": _size_text(plan),
            "status": "running", "display_time_chicago": chicago_label(to_ns(_now())),
        }, history_source=source.package.root / "ledger" / "events.jsonl")
        write_state(state_root, plan_id, phase="simulating", configurations_total=len(names),
                    configurations_done=0)
        outputs: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        tasks = [{"plan": plan.model_dump(mode="json"), "configuration": name,
                  "archive_root": None if archive_root is None else str(archive_root),
                  "data_root": None if data_root is None else str(data_root)}
                 for name in names]
        labels = {c.name: c.display_name for c in plan.configurations}

        def record(name: str, call) -> None:
            try:
                outputs.append(call())
            except Exception as error:
                failures.append({"configuration": name,
                                 "display_name": labels.get(name, name),
                                 "reason": f"{type(error).__name__}: {error}"})
            write_state(state_root, plan_id, configurations_done=len(outputs) + len(failures),
                        configurations_failed=len(failures))

        if run_config_fn is not None:
            for task in tasks:
                record(task["configuration"], lambda task=task: run_config_fn(task))
        else:
            limit = max(1, min(workers or DEFAULT_WORKERS, len(tasks)))
            with ProcessPoolExecutor(max_workers=limit) as pool:
                futures = {pool.submit(_run_one, task): task["configuration"]
                           for task in tasks}
                for future in as_completed(futures):
                    record(futures[future], future.result)
        write_state(state_root, plan_id, phase="checking_and_saving")
        context = {
            "run_identity": {"funded_comparison_plan_id": plan_id,
                             "engine_version": ENGINE_VERSION,
                             "execution_model_id": plan.execution_model.model_id},
            "funded_comparison_plan_id": plan_id,
            "purpose": plan.purpose,
            "question": plan.question,
            "approval": None if approval is None else {
                "approved_on": approval.payload.approved_on,
                "channel": approval.payload.channel, "scope": approval.payload.scope},
            "source": {"title": source.package.title,
                       "evaluation_first_day": plan.source.evaluation_dates[0],
                       "evaluation_last_day": plan.source.evaluation_dates[-1],
                       "evaluation_days": len(plan.source.evaluation_dates),
                       "warmup_days": len(plan.source.warmup_dates)},
            "owner_decisions": [d.model_dump(mode="json") for d in plan.owner_decisions],
            "limitations": list(plan.limitations),
        }
        settings = {
            **_sizing_settings(plan),
            "size_text": _size_text(plan),
            "processing_clock": plan.processing.model_dump(mode="json"),
            "firm_profiles": [p.model_dump(mode="json") for p in plan.firm_profiles],
            "execution_model": {"id": plan.execution_model.model_id,
                                "description": EXECUTION_MODEL_TEXT,
                                "refused_entry_policy": plan.execution_model.refused_entry_policy,
                                "replacement_policy": plan.execution_model.replacement_policy},
        }
        result = build_comparison_result(
            context=context, outputs=outputs, failures=failures, profiles=plan.firm_profiles,
            trading_days=source.trading_days, start_ns=source.start_ns,
            cutoff_ns=source.cutoff_ns, settings=settings)
        result["validation"] = validate_comparison(result, outputs, plan.firm_profiles,
                                                   source.cutoff_ns)
        # an incomplete configuration is shown as not completed; it does not void the
        # money checks of the completed ones
        result["validation"]["checks"]["every_configuration_completed"] = not failures
        result["validation"]["checks"]["at_least_one_configuration_completed"] = bool(outputs)
        result["validation"]["configurations_not_completed"] = failures
        result["validation"]["passed"] = bool(result["validation"]["passed"] and outputs)
        payload = FundedComparisonResultPayload(
            funded_comparison_plan_id=plan_id,
            funded_comparison_approval_id=(None if approval is None
                                           else approval.funded_comparison_approval_id),
            result_json_sha256=result_sha256(result),
            validation_passed=bool(result["validation"]["passed"]),
            engine_version=ENGINE_VERSION)
        envelope = FundedComparisonResultEnvelope.from_payload(payload)
        result_id = envelope.funded_comparison_result_id
        if not has_envelope(store_root, RESULT_STORE, result_id):
            save_envelope_immutable(
                store_root, RESULT_STORE, envelope,
                extra_files={RESULT_SIDECAR: canonical_json(result).encode("utf-8")})
        saved = load_comparison_result(store_root, result_id)
        write_state(state_root, plan_id, phase="publishing_review_folder",
                    result_id=result_id)
        leaders = {}
        for row in saved["tables"]["pair_results"]:
            if row.get("rank_within_firm") == 1:
                leaders.setdefault(row["firm"], []).append(
                    {"configuration": row["configuration"],
                     "net_cash_earned_usd": row["net_cash_earned_usd"]})
        append_ledger(store_root, {
            "event_id": f"funded_comparison_{plan_id[:16]}_completed",
            "event_type": "run_completed", "funded_comparison_plan_id": plan_id,
            "funded_comparison_result_id": result_id, "mode": plan.mode,
            "status": "completed" if payload.validation_passed
            else "completed_validation_failed",
            "configurations_completed": len(outputs),
            "configurations_not_completed": [f["configuration"] for f in failures],
            "highest_net_cash_by_firm": leaders,
            "display_time_chicago": chicago_label(to_ns(_now())),
        })
        supplements = None
        if payload.validation_passed:
            try:
                supplements = review_supplements(plan, source, saved, plan_id=plan_id)
            except Exception:  # the folder is still published, without the extra evidence
                supplements = None
        review_path, review_error, _receipt = (
            _publish(saved, result_id, store_root, Path(reports_root),
                     supplements=supplements)
            if payload.validation_passed else (None, None, None))
        return write_state(
            state_root, plan_id,
            status=("Failed" if not payload.validation_passed
                    else "Incomplete" if failures else "Completed"),
            phase="completed", result_id=result_id, review_folder=review_path,
            review_error=review_error, completed_at_utc=_now(),
            reason=None if payload.validation_passed else (
                "no configuration completed" if not outputs
                else "internal money checks failed"))
    except Exception as error:
        append_ledger(store_root, {
            "event_id": f"funded_comparison_{plan_id[:16]}_failed_{_now()}",
            "event_type": "run_failed", "funded_comparison_plan_id": plan_id,
            "status": "failed", "reason": f"{type(error).__name__}: {error}",
        })
        write_state(state_root, plan_id, status="Failed", phase="failed",
                    reason=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=8))
        raise


def _size_text(plan) -> str:
    if not is_v2(plan):
        return (f"{plan.quantity} x {INSTRUMENTS[plan.instrument].label} per trade")
    groups: dict[tuple, list[str]] = {}
    for v in plan.variants:
        groups.setdefault((v.quantity, v.instrument, v.exit_policy), []).append(v.name)
    parts = []
    for (quantity, instrument, exit_policy), _names in sorted(groups.items()):
        rule = ("whole-position exits" if exit_policy == "fixed_target_v1"
                else "the half-at-1R scale-out exit")
        parts.append(f"{quantity} x {INSTRUMENTS[instrument].label} per trade for {rule}")
    return "; ".join(parts)


def _sizing_settings(plan) -> dict[str, Any]:
    if not is_v2(plan):
        spec = INSTRUMENTS[plan.instrument]
        return {"instrument": plan.instrument, "instrument_label": spec.label,
                "quantity": plan.quantity, "tick_value_cents": spec.tick_value_cents,
                "cost_per_side_usd": round(plan.cost_per_side_cents / 100, 2)}
    return {
        "sizing_by_configuration": {
            v.name: {"instrument": v.instrument,
                     "instrument_label": INSTRUMENTS[v.instrument].label,
                     "quantity": v.quantity,
                     "tick_value_cents": INSTRUMENTS[v.instrument].tick_value_cents,
                     "cost_per_contract_per_fill_usd": v.cost_per_contract_mills / 1000,
                     "exit_policy": v.exit_policy}
            for v in plan.variants},
        "core_source": plan.core_source.model_dump(mode="json"),
        "base_configuration": plan.base_configuration,
    }
