"""The stratified-report service (R6.1 §6.G; D14 — S14 reports only).

``build_regime_stratified_reports`` reads PERSISTED artifacts only — the
descriptive ``RegimeOosAssignmentArtifact``, the exact promotion decision /
assessment / owner artifact (``resolve_report_gate``), the pooled frontier,
and the account simulations — plus the executed-trade tables the pipeline
already holds (bound into every report by their content hash and the
child's ``core_replay_id``), and persists one immutable report per
(class × subject). Every class is independent: a ``RegimeStatusRefusalError``
is RECORDED per class and never raised; no fit, no model, no selection, no
promotion happens here (test-enforced in the pipeline suite).

The modeled classes (``feature_only`` / ``cohort_model``) are S09c
deliverables: when the run recorded their exact persisted study ids
(``StratificationInputs.delivered_by``) the service verifies each by
exact-id reload and records it under ``delivered_by`` — never as a
refusal; a modeled class requested without a delivered study is a typed
refusal. A ``stratified_prop`` report publishes its D15
``account_event_regime_summary.parquet`` sidecar together with the JSON
detail (both bound by the envelope); a summary over the registered budget
raises ``EventRegimeSummaryBudgetError`` BEFORE anything is published.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pyarrow as pa

from ..features.arrow_tables import arrow_schema_hash
from ..search.orchestrator import SearchFrontierEnvelope
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .regime_assignment_sources import load_regime_oos_assignment_table
from .regime_stratification_gate import RegimeStatusRefusalError, resolve_report_gate
from .regime_stratified_contracts import (
    ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1,
    ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR,
    CLASS_INTERPRETATION,
    REGIME_STRATIFIED_REPORT_STORE,
    STRATIFIED_REPORT_DETAIL_SIDECAR,
    EventRegimeSummaryBudget,
    RegimeAssignmentEvidenceRef,
    RegimeStratificationClass,
    RegimeStratifiedReportEnvelope,
    RegimeStratifiedReportPayload,
)
from .regime_stratified_frontier import build_stratified_frontier_body
from .regime_stratified_prop import (
    AccountEventDetailLoader,
    PanelEventAssigner,
    StratifiedPropResult,
    build_stratified_prop_body,
    load_account_event_detail_from_json,
    read_account_event_regime_summary,
)
from .regime_stratified_strategy import build_cohort_descriptive_body

__all__ = [
    "DEFAULT_RUN_SCOPE",
    "MODELED_DELIVERY_ARTIFACTS",
    "ChildStratificationInputs",
    "StratificationInputs",
    "StratificationOutcome",
    "detail_bytes",
    "persist_regime_stratified_report",
    "load_regime_stratified_report",
    "load_regime_stratified_report_detail",
    "load_regime_stratified_report_summary",
    "build_regime_stratified_reports",
]

DEFAULT_RUN_SCOPE = "full_authorized_development"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
#: The S09c artifact that DELIVERS each modeled class (store, envelope name).
MODELED_DELIVERY_ARTIFACTS: MappingProxyType[str, tuple[str, str]] = MappingProxyType(
    {
        RegimeStratificationClass.FEATURE_ONLY.value: (
            "regime_controlled_studies",
            "RegimeControlledStudy",
        ),
        RegimeStratificationClass.COHORT_MODEL.value: (
            "regime_cohort_model_studies",
            "RegimeCohortModelStudy",
        ),
    }
)


@dataclass(frozen=True)
class ChildStratificationInputs:
    core_replay_id: str
    trades: pd.DataFrame
    cost_points: float
    evaluation_config_hash: str
    costed_evaluation_id: str
    pooled_predictions: pd.DataFrame | None = None
    #: account_simulation_id -> (firm_label, simulation_mode)
    account_simulations: Mapping[str, tuple[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class StratificationInputs:
    root: Path
    protocol_id: str
    decision_id: str
    owner_decision_artifact_id: str | None
    regime_oos_assignment_id: str
    requested_classes: tuple[RegimeStratificationClass, ...]
    children: Mapping[str, ChildStratificationInputs]
    frontier_id: str | None = None
    objective_metrics: tuple[str, ...] = ("net_expectancy_r",)
    tick_size: float = 0.25
    tp_r_multiple: float = 1.0
    event_loader: AccountEventDetailLoader = load_account_event_detail_from_json
    panel_assigner: PanelEventAssigner | None = None
    #: the owner-evidence run scope the report gate verifies under (S6):
    #: the pipeline's run scope (``synthetic_fixture`` only under the
    #: synthetic charter marker, and only in a test namespace)
    run_scope: str = DEFAULT_RUN_SCOPE
    #: modeled classes delivered by S09c — class value -> the EXACT persisted
    #: study id (verified by reload; recorded, never refused)
    delivered_by: Mapping[str, str] = field(default_factory=dict)
    #: the registered D15 budget of the event-regime summary
    summary_budget: EventRegimeSummaryBudget = ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1


@dataclass(frozen=True)
class StratificationOutcome:
    report_ids: tuple[str, ...]
    refusals: Mapping[str, str]
    reports_by_class: Mapping[str, tuple[str, ...]]
    #: modeled classes delivered by S09c (class value -> verified study id)
    delivered_by: Mapping[str, str] = field(default_factory=dict)


def detail_bytes(detail: Mapping[str, Any]) -> bytes:
    return (json.dumps(detail, sort_keys=True, default=str) + "\n").encode("utf-8")


def persist_regime_stratified_report(
    root: Path,
    envelope: RegimeStratifiedReportEnvelope,
    detail: bytes,
    summary: bytes | None = None,
):
    """Publish the report with its detail sidecar and — for a
    ``stratified_prop`` report — the D15 event-regime summary sidecar; the
    bytes must hash to what the envelope binds (else nothing is written)."""

    if hashlib.sha256(detail).hexdigest() != envelope.detail_sha256:
        raise ValueError("report detail bytes do not hash to the envelope's detail_sha256")
    extra_files = {STRATIFIED_REPORT_DETAIL_SIDECAR: detail}
    bound = envelope.account_event_regime_summary_sha256 is not None
    if bound != (summary is not None):
        raise ValueError(
            "a stratified_prop report is published with exactly the event-regime summary "
            "sidecar its envelope binds"
        )
    if summary is not None:
        if hashlib.sha256(summary).hexdigest() != envelope.account_event_regime_summary_sha256:
            raise ValueError(
                "event-regime summary bytes do not hash to the envelope's "
                "account_event_regime_summary_sha256"
            )
        if len(summary) != int(envelope.account_event_regime_summary_bytes):
            raise ValueError("event-regime summary byte count disagrees with the envelope")
        extra_files[ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR] = summary
    return save_or_reuse_envelope(
        Path(root), REGIME_STRATIFIED_REPORT_STORE, envelope, extra_files=extra_files
    )


def load_regime_stratified_report(root: Path, report_id: str) -> RegimeStratifiedReportEnvelope:
    return load_verified_envelope(
        Path(root), REGIME_STRATIFIED_REPORT_STORE, report_id, RegimeStratifiedReportEnvelope
    )


def load_regime_stratified_report_detail(root: Path, report_id: str) -> dict[str, Any]:
    envelope = load_regime_stratified_report(root, report_id)
    raw = load_sidecar_bytes(
        Path(root), REGIME_STRATIFIED_REPORT_STORE, report_id, STRATIFIED_REPORT_DETAIL_SIDECAR
    )
    if hashlib.sha256(raw).hexdigest() != envelope.detail_sha256:
        raise ValueError("stored report detail fails the envelope hash check")
    return json.loads(raw.decode("utf-8"))


def load_regime_stratified_report_summary(root: Path, report_id: str) -> pa.Table:
    """The verified D15 event-regime summary of a ``stratified_prop`` report
    (sha256, byte count, row count and schema hash re-checked against the
    envelope; any other class refuses — it carries none)."""

    envelope = load_regime_stratified_report(root, report_id)
    if envelope.account_event_regime_summary_sha256 is None:
        raise ValueError(
            f"a {envelope.payload.comparison_class.value} report carries no "
            "account_event_regime_summary sidecar"
        )
    raw = load_sidecar_bytes(
        Path(root), REGIME_STRATIFIED_REPORT_STORE, report_id, ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR
    )
    if hashlib.sha256(raw).hexdigest() != envelope.account_event_regime_summary_sha256 or (
        len(raw) != int(envelope.account_event_regime_summary_bytes)
    ):
        raise ValueError("stored event-regime summary fails the envelope hash check")
    table = read_account_event_regime_summary(raw)
    if int(table.num_rows) != int(envelope.account_event_regime_summary_rows):
        raise ValueError("stored event-regime summary row count disagrees with the envelope")
    if arrow_schema_hash(table.schema) != envelope.account_event_regime_summary_schema_hash:
        raise ValueError("stored event-regime summary schema disagrees with the envelope")
    return table


def _envelope(
    payload: RegimeStratifiedReportPayload,
    detail: Mapping[str, Any],
    summary: StratifiedPropResult | None = None,
):
    raw = detail_bytes(detail)
    extras: dict[str, Any] = {"detail_sha256": hashlib.sha256(raw).hexdigest()}
    if summary is not None:
        extras.update(
            account_event_regime_summary_sha256=summary.summary_sha256,
            account_event_regime_summary_bytes=len(summary.summary_bytes),
            account_event_regime_summary_rows=summary.summary_rows,
            account_event_regime_summary_schema_hash=summary.summary_schema_hash,
        )
    envelope = RegimeStratifiedReportEnvelope.from_payload(payload, **extras)
    return envelope, raw


def _verify_delivered(root: Path, comparison_class: RegimeStratificationClass, study_id: str):
    """Exact-id verified reload of the S09c artifact that delivered a modeled
    class (a caller string is never recorded unverified)."""

    if not _HEX64.fullmatch(str(study_id)):
        raise ValueError(
            f"delivered_by[{comparison_class.value}] must name a 64-hex persisted study id"
        )
    if comparison_class is RegimeStratificationClass.FEATURE_ONLY:
        from .regime_controlled_study import load_regime_controlled_study  # noqa: PLC0415

        load_regime_controlled_study(Path(root), str(study_id))
    else:
        from .regime_cohort_model import load_regime_cohort_model_study  # noqa: PLC0415

        load_regime_cohort_model_study(Path(root), str(study_id))
    return str(study_id)


def build_regime_stratified_reports(inputs: StratificationInputs) -> StratificationOutcome:
    """One report per (class × subject); refusals recorded per class."""

    root = Path(inputs.root)
    assignment_envelope, assignment = load_regime_oos_assignment_table(
        root, inputs.regime_oos_assignment_id
    )
    payload = assignment_envelope.payload
    if payload.resolved_regime_protocol_id != inputs.protocol_id:
        raise ValueError("the OOS assignment artifact belongs to another regime protocol")
    fit_ids = tuple(payload.regime_fit_ids)
    evidence = RegimeAssignmentEvidenceRef(
        observation_granularity=payload.observation_granularity,
        regime_fit_ids=fit_ids,
        regime_fold_set_id=payload.regime_fold_set_id,
        fold_schedule_id=payload.fold_schedule_id,
        regime_oos_assignment_id=assignment_envelope.regime_oos_assignment_id,
    )
    report_ids: list[str] = []
    refusals: dict[str, str] = {}
    delivered: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    strategy_bodies: dict[str, Any] = {}
    strategy_results: dict[str, Any] = {}

    def _gate(comparison_class: RegimeStratificationClass):
        return resolve_report_gate(
            root,
            protocol_id=inputs.protocol_id,
            decision_id=inputs.decision_id,
            owner_decision_artifact_id=inputs.owner_decision_artifact_id,
            comparison_class=comparison_class,
            fit_ids=fit_ids,
            run_scope=str(inputs.run_scope),
        )

    def _publish(comparison_class, gate, refs, body, detail, summary=None) -> str:
        report_payload = RegimeStratifiedReportPayload(
            resolved_regime_protocol_id=inputs.protocol_id,
            comparison_class=comparison_class,
            gate=gate,
            assignment_evidence=evidence,
            source_metric_refs=tuple(sorted(set(refs))),
            interpretation=CLASS_INTERPRETATION[comparison_class],
            body=body,
        )
        envelope, raw = _envelope(report_payload, detail, summary)
        persist_regime_stratified_report(
            root, envelope, raw, None if summary is None else summary.summary_bytes
        )
        report_ids.append(envelope.regime_stratified_report_id)
        by_class.setdefault(comparison_class.value, []).append(
            envelope.regime_stratified_report_id
        )
        return envelope.regime_stratified_report_id

    requested = tuple(RegimeStratificationClass(value) for value in inputs.requested_classes)
    wants_strategy = RegimeStratificationClass.COHORT_DESCRIPTIVE in requested
    wants_frontier = RegimeStratificationClass.STRATIFIED_FRONTIER in requested
    # cohort_descriptive strata are the frontier view's input, so they are
    # built whenever either class is requested (persisted only when requested)
    if wants_strategy or wants_frontier:
        try:
            resolved = _gate(RegimeStratificationClass.COHORT_DESCRIPTIVE)
        except RegimeStatusRefusalError as error:
            if wants_strategy:
                refusals[RegimeStratificationClass.COHORT_DESCRIPTIVE.value] = str(error)
            if wants_frontier:
                refusals[RegimeStratificationClass.STRATIFIED_FRONTIER.value] = str(error)
            resolved = None
        if resolved is not None:
            for core_replay_id in sorted(inputs.children):
                child = inputs.children[core_replay_id]
                result = build_cohort_descriptive_body(
                    trades=child.trades,
                    assignment=assignment,
                    core_replay_id=core_replay_id,
                    protocol_id=inputs.protocol_id,
                    fit_ids=fit_ids,
                    cost_points=child.cost_points,
                    evaluation_config_hash=child.evaluation_config_hash,
                    tick_size=inputs.tick_size,
                    tp_r_multiple=inputs.tp_r_multiple,
                    pooled_predictions=child.pooled_predictions,
                )
                strategy_bodies[core_replay_id] = result.body
                strategy_results[core_replay_id] = result
                if wants_strategy:
                    _publish(
                        RegimeStratificationClass.COHORT_DESCRIPTIVE,
                        resolved.gate,
                        (child.costed_evaluation_id, inputs.regime_oos_assignment_id),
                        result.body,
                        result.detail,
                    )
            if wants_frontier:
                if inputs.frontier_id is None:
                    refusals[RegimeStratificationClass.STRATIFIED_FRONTIER.value] = (
                        "stratified_frontier requires the persisted pooled frontier id; "
                        "nothing here promotes"
                    )
                else:
                    frontier = load_verified_envelope(
                        root, "frontiers", inputs.frontier_id, SearchFrontierEnvelope
                    )
                    body = build_stratified_frontier_body(
                        frontier=frontier,
                        strategy_bodies=strategy_bodies,
                        objective_metrics=inputs.objective_metrics,
                    )
                    _publish(
                        RegimeStratificationClass.STRATIFIED_FRONTIER,
                        resolved.gate,
                        (inputs.frontier_id, inputs.regime_oos_assignment_id),
                        body,
                        {"frontier": frontier.model_dump(mode="json")},
                    )
    if RegimeStratificationClass.STRATIFIED_PROP in requested:
        try:
            resolved = _gate(RegimeStratificationClass.STRATIFIED_PROP)
        except RegimeStatusRefusalError as error:
            refusals[RegimeStratificationClass.STRATIFIED_PROP.value] = str(error)
        else:
            from .regime_assignment_sources import regime_for_trades  # noqa: PLC0415

            for core_replay_id in sorted(inputs.children):
                child = inputs.children[core_replay_id]
                if not child.account_simulations:
                    continue
                trade_regimes = (
                    strategy_results[core_replay_id].trade_regimes
                    if core_replay_id in strategy_results
                    else regime_for_trades(child.trades, assignment)
                )
                # the D15 summary is built (and budget-checked) BEFORE the
                # report is published; a budget refusal propagates typed
                result = build_stratified_prop_body(
                    root=root,
                    core_replay_id=core_replay_id,
                    simulations=child.account_simulations,
                    trade_regimes=trade_regimes,
                    evidence=evidence,
                    loader=inputs.event_loader,
                    panel_assigner=inputs.panel_assigner,
                    budget=inputs.summary_budget,
                )
                _publish(
                    RegimeStratificationClass.STRATIFIED_PROP,
                    resolved.gate,
                    (*child.account_simulations, inputs.regime_oos_assignment_id),
                    result.body,
                    result.detail,
                    summary=result,
                )
    for comparison_class in requested:
        if comparison_class.value not in MODELED_DELIVERY_ARTIFACTS:
            continue
        store_name, artifact_name = MODELED_DELIVERY_ARTIFACTS[comparison_class.value]
        study_id = inputs.delivered_by.get(comparison_class.value)
        if study_id is None:
            refusals[comparison_class.value] = (
                f"{comparison_class.value} is a modeled class delivered by the supervised "
                f"regime studies (S09b/S09c) as a persisted {artifact_name} artifact "
                f"({store_name}); this run recorded none for it (not a model-bearing "
                "request, or S09c did not deliver it); nothing here promotes"
            )
            continue
        delivered[comparison_class.value] = _verify_delivered(root, comparison_class, study_id)
    return StratificationOutcome(
        report_ids=tuple(report_ids),
        refusals=dict(refusals),
        reports_by_class={key: tuple(value) for key, value in by_class.items()},
        delivered_by=dict(delivered),
    )
