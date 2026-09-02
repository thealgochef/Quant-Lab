"""S14's regime half — the stratified reports from PERSISTED artifacts only
(R6.1 §6.E / §6.G; D14: S14 performs zero fitting; R6.1-FIX §3.7 / §3.8).

``build_reports(context)`` assembles the :class:`StratificationInputs` of
``regime_stratification_service`` from the run: the exact S10 decision (or
the frozen FEATURE_ELIGIBLE decision of a model-bearing request), the
descriptive OOS-assignment artifact, every gated child's executed-trade
table VERIFIED-LOADED from the immutable ``executed_trade_tables`` store by
its derived id (R6.1-FIX §3.7 — never an in-memory frame; the table id and
hash are bound into every report), the exact persisted account simulation
ids S12/S13 reported, the pooled frontier, and — for the panel grain — a
point-in-time assigner over the persisted panel + fits for historical
no-trade prop events. S14 iterates the charter's child set: a child that is
not gated, or whose table is unavailable, is a TYPED ``children_skipped``
record — never a silent omission — and the reports are ALWAYS built from
the present children (review B-03: a prior attempt's report record is never
returned in place of this attempt's evidence; identical evidence re-mints
identical report ids, which the store reuses). No estimator is fitted here;
every class refusal is recorded per class, never raised.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from ..features.context_bar_panel_materializer import (
    load_context_bar_panel_artifact,
    load_context_bar_panel_frame,
)
from ..search.executed_trade_table import (
    load_executed_trade_table,
    probe_executed_trade_table,
)
from ..search.identities import canonical_contract_sha256
from ..search.store import SidecarLoadError, load_json_sidecar
from .regime_oos_assignment import (
    assign_panel_regimes_to_candidates,
    load_regime_oos_assignment,
)
from .regime_store import load_regime_fit_assignments, load_regime_protocol
from .regime_stratification_service import (
    ChildStratificationInputs,
    StratificationInputs,
    build_regime_stratified_reports,
)
from .regime_stratified_contracts import RegimeStratificationClass
from .regime_stratified_prop import load_account_event_detail_from_json

__all__ = [
    "CHILD_SKIP_REASONS",
    "build_reports",
    "event_detail_loader_for_policy",
    "panel_event_assigner",
]

#: The typed reasons a charter child is recorded under ``children_skipped``.
CHILD_SKIP_REASONS: tuple[str, ...] = (
    "child_not_completed_or_reused",
    "strategy_gates_not_passed",
    "executed_trade_table_unavailable",
)


def event_detail_loader_for_policy(policy_id: str):
    """The D15 reader for the charter's event-detail policy: the v2 Parquet
    partitions when persisted, else the R3 JSON sidecar (historical modes);
    ``None`` from the loader means ``evidence_not_persisted``."""

    if policy_id == "account_event_detail_by_path_parquet_v2":
        from alpha_lab.propsim.event_detail import (  # noqa: PLC0415
            EventDetailUnavailableError,
            load_account_event_detail,
        )

        def _loader(root: Path, account_simulation_id: str) -> Iterator[pd.DataFrame] | None:
            try:
                return load_account_event_detail(Path(root), account_simulation_id)
            except EventDetailUnavailableError:
                return None

        return _loader
    return load_account_event_detail_from_json


def panel_event_assigner(context):
    """Panel grain: a PIT assigner (the normative §6.C rule) for historical
    no-trade events over PERSISTED artifacts only — the descriptive OOS
    assignment artifact, the protocol, the panel frame and every fit's
    assignment sidecar are verified-loaded by the exact ids the run
    recorded; no in-memory run object is consulted (adversarial R6.1 F14).
    """

    regime = context.regime
    request = regime["request"]
    if not request.is_panel:
        return None
    root = Path(context.store_root)
    oos_id = regime["execution"].oos_assignment.regime_oos_assignment_id
    oos_payload = load_regime_oos_assignment(root, oos_id).payload
    if oos_payload.panel_context is None:
        raise ValueError("the persisted OOS assignment carries no panel context")
    protocol = load_regime_protocol(root, oos_payload.resolved_regime_protocol_id)
    panel_envelope = load_context_bar_panel_artifact(
        root, oos_payload.panel_context.context_bar_panel_artifact_id
    )
    panel_frame = load_context_bar_panel_frame(root, panel_envelope)
    # R6.1-FIX (§3.1): every fit's sidecar is verified against the exact ref
    # the persisted OOS artifact bound (sidecar sha256 + schema hash)
    verified_by_id = {
        fit_id: load_regime_fit_assignments(root, fit_id)
        for fit_id in oos_payload.regime_fit_ids
    }
    for ref in oos_payload.regime_fit_assignment_refs:
        verified = verified_by_id[ref.regime_fit_id]
        if verified.ref != ref:
            raise ValueError(
                f"fit {ref.regime_fit_id[:12]}… assignment sidecar does not match the "
                "reference the persisted OOS assignment bound; refusing"
            )
    assignments = pd.concat(
        [verified_by_id[fit_id].frame for fit_id in oos_payload.regime_fit_ids],
        ignore_index=True,
    )
    interval = int(protocol.payload.panel_interval_seconds or 0)

    def _assign(event_ts_utc: pd.Series) -> pd.Series:
        candidates = pd.DataFrame(
            {
                "candidate_id": [f"event_{index:06d}" for index in range(len(event_ts_utc))],
                "as_of_ts_utc": event_ts_utc.to_numpy(),
            }
        )
        assigned = assign_panel_regimes_to_candidates(
            panel_frame,
            assignments,
            candidates,
            protocol=protocol,
            max_staleness_seconds=interval,
        ).set_index("candidate_id")
        ordered = assigned.loc[candidates["candidate_id"]]
        values = ordered["canonical_reporting_cluster_id"].where(ordered["valid"].astype(bool))
        return pd.Series(values.to_numpy(), index=event_ts_utc.index)

    return _assign


def _child_table_evidence(context, core_replay_id: str) -> tuple[Any | None, str | None]:
    """``(evidence, skip_reason)`` for one charter child: the executed-trade
    table this run holds (S02: persisted after a fresh completion / verified
    reproduction, or exact-loaded for a reused child), re-verified here by
    exact id against the store; a corrupt entry propagates typed."""

    held = context.executed_trades_by_child.get(core_replay_id)
    if held is None:
        return None, "executed_trade_table_unavailable"
    table_id = str(held.executed_trade_table_id)
    state = probe_executed_trade_table(context.store_root, table_id)
    if state != "present":
        return None, "executed_trade_table_unavailable"
    loaded = load_executed_trade_table(context.store_root, table_id)
    if loaded.envelope.executed_trade_table_sha256 != held.executed_trade_table_sha256:
        raise SidecarLoadError(
            "sidecar_hash_mismatch",
            f"executed-trade table {table_id[:12]}… no longer hashes to the evidence S02 "
            "recorded for this child",
        )
    return loaded, None


def build_reports(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    spec = context.semantic.payload
    request = spec.regime_study
    regime = context.regime
    charter_payload = context.charter.payload
    cost_policy = charter_payload.cost_policy
    cost = (
        cost_policy.cost_points_round_turn
        if context.wiring.cost_points is None
        else context.wiring.cost_points
    )
    account_simulations = regime.get("account_simulations", {})
    from ..search.orchestrator import _child_evaluation_envelope  # noqa: PLC0415

    # R6.1-FIX §3.7: S14 iterates the CHARTER child set; every child is either
    # a report subject (its persisted executed-trade table verified-loaded by
    # exact id) or a typed ``children_skipped`` record — never omitted silently
    children: dict[str, ChildStratificationInputs] = {}
    skipped: dict[str, str] = {}
    evidence_by_child: dict[str, dict[str, Any]] = {}
    for row in sorted(context.children, key=lambda item: str(item["core_replay_id"])):
        core_replay_id = str(row["core_replay_id"])
        if row["state"] not in ("completed", "reused"):
            skipped[core_replay_id] = "child_not_completed_or_reused"
            continue
        loaded, reason = _child_table_evidence(context, core_replay_id)
        if loaded is None:
            # the most specific typed fact first: a reused child whose table
            # is unavailable could never be gated this run
            skipped[core_replay_id] = str(reason)
            continue
        if core_replay_id not in context.gates_passed:
            skipped[core_replay_id] = "strategy_gates_not_passed"
            continue
        evaluation = _child_evaluation_envelope(core_replay_id, cost_policy)
        evidence_by_child[core_replay_id] = {
            "executed_trade_table_id": loaded.envelope.executed_trade_table_id,
            "executed_trade_table_sha256": loaded.envelope.executed_trade_table_sha256,
            "row_count": int(loaded.envelope.row_count),
        }
        children[core_replay_id] = ChildStratificationInputs(
            core_replay_id=core_replay_id,
            trades=loaded.frame,
            cost_points=float(cost),
            evaluation_config_hash=canonical_contract_sha256(
                {
                    "core_replay_id": core_replay_id,
                    "cost_policy": cost_policy.model_dump(mode="json"),
                }
            ),
            costed_evaluation_id=evaluation.costed_evaluation_id,
            pooled_predictions=None,
            account_simulations=dict(account_simulations.get(core_replay_id, {})),
            executed_trade_table_id=loaded.envelope.executed_trade_table_id,
        )
    for reason in set(skipped.values()):
        if reason not in CHILD_SKIP_REASONS:  # pragma: no cover - closed vocabulary
            raise ValueError(f"unregistered children_skipped reason {reason!r}")
    decision = regime["decision"]
    execution = regime["execution"]
    # review B-03: the verified table evidence of THIS attempt is what S15
    # reloads (executed_trade_tables/<id> by the recorded hash)
    regime["children_evidence"] = {core: dict(evidence_by_child[core]) for core in sorted(children)}
    classes = tuple(
        RegimeStratificationClass(name) for name in request.comparison_classes_requested
    )
    inputs = StratificationInputs(
        root=Path(context.store_root),
        protocol_id=execution.protocol.resolved_regime_protocol_id,
        decision_id=decision.regime_promotion_decision_id,
        owner_decision_artifact_id=request.owner_decision_artifact_id,
        regime_oos_assignment_id=execution.oos_assignment.regime_oos_assignment_id,
        requested_classes=classes,
        children=children,
        frontier_id=context.frontier_id,
        objective_metrics=tuple(charter_payload.objective_policy.pareto_objectives),
        tick_size=float(cost_policy.tick_size),
        event_loader=event_detail_loader_for_policy(
            charter_payload.simulation_protocol.event_detail_persistence_policy_id
        ),
        panel_assigner=panel_event_assigner(context),
        run_scope=str(context.semantic.payload.run_scope.value),
        delivered_by=_delivered_by_s09c(context, regime),
    )
    outcome = build_regime_stratified_reports(inputs)
    regime["stratified_report_ids"] = tuple(outcome.report_ids)
    record = {
        "regime_promotion_decision_id": decision.regime_promotion_decision_id,
        "authority_source": regime.get("authority_source"),
        "regime_oos_assignment_id": execution.oos_assignment.regime_oos_assignment_id,
        "requested_classes": [cls.value for cls in classes],
        "report_ids": list(outcome.report_ids),
        "reports_by_class": {k: list(v) for k, v in outcome.reports_by_class.items()},
        "refusals": dict(outcome.refusals),
        "delivered_by": dict(outcome.delivered_by),
        "children": sorted(children),
        "children_evidence": {core: evidence_by_child[core] for core in sorted(children)},
        "children_skipped": dict(sorted(skipped.items())),
        "fitting_performed": False,
    }
    note = (
        f"; {len(outcome.report_ids)} stratified regime report(s) persisted from persisted "
        f"artifacts only ({len(outcome.refusals)} class refusal(s) recorded; "
        f"{len(outcome.delivered_by)} modeled class(es) delivered by S09c; "
        f"{len(skipped)} child(ren) skipped with a typed reason)"
    )
    return tuple(outcome.report_ids), record, note


_STAGE_S09 = "09_train_models"
_S09_RECORD_SIDECAR = "regime_run.json"


def _delivered_by_s09c(context, regime) -> dict[str, str]:
    """The exact S09c study ids that DELIVERED the modeled classes of this
    run: the in-memory S09c results when this attempt ran them, else the
    run's OWN verified S09 stage record (never a store listing). R6.1-FIX
    §3.8: a corrupt record is a typed failure; only a manifest-proven
    "not produced" record means no delivery."""

    delivered: dict[str, str] = {}
    study = regime.get("controlled_study")
    if study is not None:
        delivered["feature_only"] = str(study.envelope.regime_controlled_study_id)
    cohort = regime.get("cohort_model_study")
    if cohort is not None:
        delivered["cohort_model"] = str(cohort.envelope.regime_cohort_model_study_id)
    if delivered:
        return delivered
    entry = context.state["stages"].get(_STAGE_S09, {})
    stage_result_id = entry.get("stage_result_id")
    if not stage_result_id:
        return delivered
    record = load_json_sidecar(
        context.store_root, "pipeline_stage_results", stage_result_id, _S09_RECORD_SIDECAR
    )
    s09c = ((record or {}).get("S09c") or {}) if record is not None else {}
    for comparison_class, key in (
        ("feature_only", "regime_controlled_study_id"),
        ("cohort_model", "regime_cohort_model_study_id"),
    ):
        if s09c.get(key):
            delivered[comparison_class] = str(s09c[key])
    return delivered
