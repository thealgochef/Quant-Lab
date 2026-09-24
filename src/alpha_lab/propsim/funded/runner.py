"""Worker-side execution of one frozen funded plan (called by the job script).

Reads ONLY the verified plan envelope from the store, re-verifies the strategy
source against the plan's frozen hashes, builds price paths under the plan's
evidence policy, runs both firm instances, checks resumed-state equivalence,
validates the money, saves the immutable result, appends the cumulative
research ledger and publishes the review folder. Progress is written to
``<state_root>/<plan_id>/state.json`` after every phase.
"""

from __future__ import annotations

import json
import os
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_json_sidecar,
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.propsim.funded.campaign import (
    CampaignInputs,
    resumed_equivalence,
    run_campaign,
    validate_inputs,
)
from alpha_lab.propsim.funded.clock import chicago_label, to_ns
from alpha_lab.propsim.funded.plan import (
    FundedPayoutPlanEnvelope,
    FundedPayoutResultEnvelope,
    FundedPayoutResultPayload,
)
from alpha_lab.propsim.funded.price_evidence import DATA_ROOT, build_paths
from alpha_lab.propsim.funded.result import (
    build_result,
    canonical_json,
    result_sha256,
    validate_result,
)
from alpha_lab.propsim.funded.sources import (
    ARCHIVE_ROOT,
    executions_sha256,
    load_minute_bars,
    load_profile_executions,
    load_trading_days,
    open_verified_package,
)

__all__ = [
    "ENGINE_VERSION",
    "PLAN_STORE",
    "RESULT_STORE",
    "RESULT_SIDECAR",
    "run_plan",
    "read_state",
    "write_state",
    "ledger_path",
    "append_ledger",
    "read_ledger",
    "load_result",
    "publish_review",
]

ENGINE_VERSION = "funded_engine_v1"
PLAN_STORE = "funded_payout_plans"
RESULT_STORE = "funded_payout_results"
RESULT_SIDECAR = "result.json"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def state_path(state_root: Path, plan_id: str) -> Path:
    return Path(state_root) / plan_id / "state.json"


def read_state(state_root: Path, plan_id: str) -> dict[str, Any] | None:
    path = state_path(state_root, plan_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(state_root: Path, plan_id: str, **fields: Any) -> dict[str, Any]:
    path = state_path(state_root, plan_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = read_state(state_root, plan_id) or {"plan_id": plan_id, "history": []}
    state.update(fields)
    state["updated_at_utc"] = _now()
    if "phase" in fields:
        state["history"].append({"phase": fields["phase"], "at_utc": state["updated_at_utc"]})
    tmp = path.with_suffix(f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)
    return state


# ── cumulative research ledger (append-only) ──────────────────────────────


def ledger_path(store_root: Path) -> Path:
    return Path(store_root).parent / "funded_payout_research_ledger.jsonl"


def read_ledger(store_root: Path) -> list[dict[str, Any]]:
    path = ledger_path(store_root)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_ledger(store_root: Path, entry: dict[str, Any], *,
                  history_source: Path | None = None) -> None:
    """Append one entry; the first write imports the prior verified history.

    The imported events are copied verbatim from a verified study package's own
    cumulative ledger and labeled as imported history — never rewritten.
    """

    path = ledger_path(store_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_ledger(store_root)
    lines: list[dict[str, Any]] = []
    if not existing and history_source is not None and history_source.is_file():
        for raw in history_source.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                lines.append({"ledger_origin": "imported_verified_history",
                              "imported_from": history_source.parent.parent.parent.name,
                              **json.loads(raw)})
    if any(e.get("event_id") == entry.get("event_id") for e in existing):
        return  # idempotent: a resumed worker never duplicates a ledger event
    entry = {"ledger_origin": "funded_payout_lane", "recorded_at_utc": _now(), **entry}
    lines.append(entry)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line, sort_keys=True) + "\n")


# ── the run ───────────────────────────────────────────────────────────────


def _stop_gap(executions, paths) -> dict[str, int]:
    """Stop exits whose triggering print is worse than the recorded stop fill."""

    count = worse = ticks = 0
    for execution in executions:
        path = paths.get(execution.trade_id)
        if execution.is_warmup or execution.exit_reason != "stop" or path is None:
            continue
        if path.fidelity != "ordered_trade_prints" or not len(path.price_ticks):
            continue
        count += 1
        gap = (execution.stop_ticks - int(path.price_ticks[-1])) * execution.sign
        if gap > 0:
            worse += 1
            ticks += gap
    return {"stop_exits_on_prints": count,
            "stop_exits_with_triggering_print_worse_than_stop": worse,
            "stop_gap_ticks_total_per_contract": ticks}


def load_result(store_root: Path, result_id: str) -> dict[str, Any]:
    envelope = load_verified_envelope(store_root, RESULT_STORE, result_id,
                                      FundedPayoutResultEnvelope)
    result = load_json_sidecar(store_root, RESULT_STORE, result_id, RESULT_SIDECAR)
    if result_sha256(result) != envelope.payload.result_json_sha256:
        raise ValueError("the saved funded result does not match its verified hash")
    return result


def _publish(result: dict[str, Any], result_id: str, store_root: Path,
             reports_root: Path) -> tuple[str | None, str | None]:
    from alpha_lab.agents.data_infra.ifvg.funded_review_package import (
        next_export_version,
        publish_funded_review_folder,
    )

    try:
        published = publish_funded_review_folder(
            result=result, funded_result_id=result_id, reports_root=reports_root,
            ledger_entries=read_ledger(store_root),
            export_version=next_export_version(reports_root, result_id),
        )
    except Exception as error:  # the economic result stays saved and verified
        return None, f"{type(error).__name__}: {error}"
    return str(published.path), None


def publish_review(*, plan_id: str, store_root: Path, state_root: Path,
                   reports_root: Path) -> dict[str, Any]:
    """Retry publication for a completed, verified result (a new export version).

    The economic result is loaded from the store and never recomputed.
    """

    state = read_state(state_root, plan_id) or {}
    if state.get("status") != "Completed" or not state.get("result_id"):
        raise ValueError("only a completed, verified funded result can be published")
    if state.get("review_folder"):
        return state
    result = load_result(Path(store_root), state["result_id"])
    path, error = _publish(result, state["result_id"], Path(store_root), Path(reports_root))
    if path:
        append_ledger(Path(store_root), {
            "event_id": f"funded_{plan_id[:16]}_review_published_{Path(path).name}",
            "event_type": "review_folder_published", "funded_plan_id": plan_id,
            "funded_result_id": state["result_id"], "status": "published",
            "note": "publication retried after an earlier export check failure; the "
                    "economic result is unchanged",
        })
    return write_state(Path(state_root), plan_id, review_folder=path, review_error=error,
                       phase="review_published" if path else "review_failed")


def run_plan(*, plan_id: str, store_root: Path, state_root: Path, reports_root: Path,
             archive_root: Path | None = None, data_root: Path = DATA_ROOT) -> dict[str, Any]:
    store_root, state_root = Path(store_root), Path(state_root)
    archive_root = ARCHIVE_ROOT if archive_root is None else Path(archive_root)
    write_state(state_root, plan_id, status="Running", phase="loading_plan",
                pid=os.getpid(), started_at_utc=_now())
    try:
        plan = load_verified_envelope(store_root, PLAN_STORE, plan_id,
                                      FundedPayoutPlanEnvelope).payload
        source = plan.source
        package = None
        for candidate in archive_root.glob(f"*/{source.package_root_name}"):
            package = open_verified_package(candidate)
        if package is None or package.run_id != source.package_run_id:
            raise ValueError("the plan's verified strategy source package is unavailable")
        if package.manifest_sha256 != source.package_manifest_sha256:
            raise ValueError("the strategy source package changed after the plan was frozen")
        history = package.root / "ledger" / "events.jsonl"
        append_ledger(store_root, {
            "event_id": f"funded_{plan_id[:16]}_started", "event_type": "run_started",
            "funded_plan_id": plan_id, "purpose": plan.purpose,
            "question": plan.question, "status": "running",
            "scope": plan.authorized_scope,
            "display_time_chicago": chicago_label(to_ns(_now())),
        }, history_source=history)
        write_state(state_root, plan_id, phase="verifying_source")
        executions = load_profile_executions(package, source.profile_id)
        if executions_sha256(executions) != source.executions_sha256:
            raise ValueError("strategy executions differ from the frozen plan")
        days, start_ns, cutoff_ns, (first, last) = load_trading_days(package)
        if (first, last) != (source.evaluation_first_day, source.evaluation_last_day):
            raise ValueError("evaluation dates differ from the frozen plan")
        write_state(state_root, plan_id, phase="building_price_paths")
        bars = load_minute_bars(package)
        paths, evidence, files = build_paths(
            executions, bars, policy=plan.price_evidence_policy, data_root=data_root)
        missing = [e for e in evidence if e.fidelity == "missing"]
        if missing:
            raise ValueError(
                f"{len(missing)} execution(s) lack verified ordered trade prints and the plan "
                "requires exact price evidence; first reason: " + missing[0].reason)
        inputs = CampaignInputs(
            executions=executions, paths=paths, trading_days=days, start_ns=start_ns,
            cutoff_ns=cutoff_ns, instrument=plan.instrument, quantity=plan.quantity,
            cost_per_side_cents=plan.cost_per_side_cents, processing=plan.processing,
            profiles=plan.firm_profiles,
        )
        validate_inputs(inputs)
        write_state(state_root, plan_id, phase="simulating")
        instances = run_campaign(inputs)
        write_state(state_root, plan_id, phase="checking_resumed_state")
        resumed = resumed_equivalence(inputs)
        fidelity = Counter(e.fidelity for e in evidence)
        price_evidence = {
            "policy": plan.price_evidence_policy,
            "trades_with_ordered_prints": fidelity.get("ordered_trade_prints", 0),
            "trades_with_minute_approximation": sum(
                v for k, v in fidelity.items() if k.startswith("minute_bars")),
            "trades_where_prints_reach_the_other_exit_first": sum(
                1 for e in evidence if e.ordering_note),
            **_stop_gap(executions, paths),
            "mark_price": "last traded price (exchange trade prints)",
            "per_trade": [asdict(e) for e in evidence],
            "source_files": files,
            "reconciliation": "every one-minute candle of every accepted position rebuilt "
                              "exactly (open, high, low, close, print count) from the prints",
        }
        result = build_result(
            inputs, instances,
            run_identity={"funded_plan_id": plan_id, "engine_version": ENGINE_VERSION,
                          "source_package_run_id": source.package_run_id,
                          "source_profile_id": source.profile_id},
            price_evidence=price_evidence,
            context={
                "funded_plan_id": plan_id, "purpose": plan.purpose,
                "question": plan.question,
                "source": {**source.model_dump(mode="json"), "title": package.title},
                "owner_decisions": [d.model_dump(mode="json") for d in plan.owner_decisions],
                "limitations": list(plan.limitations),
            },
        )
        result["validation"] = validate_result(result, instances, resumed)
        payload = FundedPayoutResultPayload(
            funded_plan_id=plan_id, result_json_sha256=result_sha256(result),
            validation_passed=bool(result["validation"]["passed"]),
            engine_version=ENGINE_VERSION,
        )
        envelope = FundedPayoutResultEnvelope.from_payload(payload)
        result_id = envelope.funded_result_id
        if not has_envelope(store_root, RESULT_STORE, result_id):
            save_envelope_immutable(
                store_root, RESULT_STORE, envelope,
                extra_files={RESULT_SIDECAR: canonical_json(result).encode("utf-8")})
        saved = load_result(store_root, result_id)  # screen and export read this
        write_state(state_root, plan_id, phase="publishing_review_folder",
                    result_id=result_id)
        summaries = saved["summaries"]
        append_ledger(store_root, {
            "event_id": f"funded_{plan_id[:16]}_completed", "event_type": "run_completed",
            "funded_plan_id": plan_id, "funded_result_id": result_id,
            "purpose": plan.purpose, "status": "completed" if payload.validation_passed
            else "completed_validation_failed",
            "outcome": {k: {m: v.get(m) for m in (
                "net_cash_earned_usd", "payouts_received_usd", "largest_single_payout_usd",
                "acquisition_costs_usd", "accounts_lost_before_first_payout",
                "accounts_lost_after_a_payout")} for k, v in summaries.items()},
            "display_time_chicago": chicago_label(to_ns(_now())),
        })
        review_path, review_error = (
            _publish(saved, result_id, store_root, Path(reports_root))
            if payload.validation_passed else (None, None))
        return write_state(
            state_root, plan_id, status="Completed" if payload.validation_passed else "Failed",
            phase="completed", result_id=result_id, review_folder=review_path,
            review_error=review_error, completed_at_utc=_now(),
            reason=None if payload.validation_passed else "internal money checks failed",
        )
    except Exception as error:
        append_ledger(store_root, {
            "event_id": f"funded_{plan_id[:16]}_failed_{_now()}", "event_type": "run_failed",
            "funded_plan_id": plan_id, "status": "failed",
            "reason": f"{type(error).__name__}: {error}",
        })
        write_state(state_root, plan_id, status="Failed", phase="failed",
                    reason=f"{type(error).__name__}: {error}",
                    traceback=traceback.format_exc(limit=8))
        raise
