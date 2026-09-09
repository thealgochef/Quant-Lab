"""The regime study inside the operator pipeline — stage bodies S05–S15
(R6.1 §6.E; D1, D2, D4, D14).

The pipeline runner (``search/pipeline.py``) keeps the 16-stage contract and
calls these helpers from its stage bodies whenever the frozen
``PipelineSemanticSpecPayload.regime_study`` request is present. Every
regime input is a VERIFIED-LOADED artifact (D2): S05 persists the bundle
views (candidate grain) or materializes + persists the context-bar panel
from the verified replay-chart artifact (panel grain) and resolves + persists
the protocol; S06 records the input coverage + the stamped sample-adequacy
floors; S08 persists the fold schedule and the fold-set artifact(s) with the
per-fold adequacy preview; S09a executes the protocol through the executor
(fits, assessment, the descriptive OOS-assignment artifact); S09b/S09c run
only for model-bearing requests; S10 derives the deterministic decisions
(or re-verifies the frozen authority) and persists the diagnostics; S14
builds the stratified reports from persisted artifacts only (zero fitting);
S15 reloads every regime artifact through the verified stores.

Nothing here resolves a "latest" status: descriptive runs derive their own
STRATIFICATION_READY from their own assessment; model-bearing runs carry the
exact frozen promotion / owner / assessment ids in the semantic identity and
re-verify them at S00, S05, S09a, and S10.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from ..features.bundle_feature_view import (
    load_bundle_feature_view,
    save_bundle_feature_view,
)
from ..features.context_bar_panel_materializer import (
    load_context_bar_panel_artifact,
    materialize_context_bar_panel,
    save_context_bar_panel_artifact,
)
from ..fold_schedules import (
    build_candidate_folds_from_schedule,
    build_context_bar_panel_folds,
    derive_fold_schedule,
)
from ..ml.fold_set_artifact import (
    assert_same_fold_schedule,
    build_fold_set_artifact,
    load_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from ..ml.regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    sample_adequacy_minimum,
)
from ..ml.regime_executor import execute_regime_protocol
from ..ml.regime_observation_source import RegimeObservationSourceRef
from ..ml.regime_oos_assignment import load_regime_oos_assignment
from ..ml.regime_sample_adequacy import preview_sample_adequacy
from ..ml.regime_store import (
    load_regime_assessment,
    load_regime_promotion,
    load_regime_protocol,
    persist_regime_protocol,
)
from ..ml.regime_study import (
    RegimeStudyRequest,
    RegimeStudyStagePlanFacts,
    build_s10_decisions,
    regime_evidence_as_of,
    regime_study_block_reason,
    resolve_study_protocol,
    verify_frozen_authority,
)
from .identities import canonical_contract_sha256

__all__ = [
    "PANEL_CHART_SELECTION_POLICY_ID",
    "REGIME_STAGE_VALUES",
    "select_panel_chart",
    "regime_request",
    "regime_run_scope",
    "regime_stage_plan_facts",
    "regime_readiness_reason",
    "regime_wiring_problems",
    "s05_regime_observation",
    "s06_regime_coverage",
    "s08_regime_folds",
    "s09_regime_fit",
    "s10_regime_diagnostics",
    "s14_regime_reports",
    "s15_regime_reload_failures",
    "s15_regime_reload_ok",
    "canonical_json_bytes",
]

#: The declared, identity-bearing chart selection of a panel-grain study
#: (adversarial R6.1 F13): the replay chart of the child with the
#: lexicographically lowest ``core_replay_id`` among the children S04 built
#: charts for — deterministic and enumeration-order-invariant; that child
#: must carry exactly one chart. The selection is recorded in the S05
#: sidecar together with the LOADED artifact id (S1).
PANEL_CHART_SELECTION_POLICY_ID = "panel_chart_lowest_core_replay_id_v1"

#: The stages a regime study touches (capability-scoped readiness).
REGIME_STAGE_VALUES: tuple[str, ...] = (
    "05_materialize_feature_views",
    "06_validate_feature_coverage",
    "08_build_folds",
    "09_train_models",
    "10_generate_predictions_and_diagnostics",
    "14_build_frontier_and_insights",
)
_ET = ZoneInfo("America/New_York")


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, default=str) + "\n").encode("utf-8")


def regime_request(spec) -> RegimeStudyRequest | None:
    return getattr(spec, "regime_study", None)


def regime_run_scope(context) -> str:
    """The owner-evidence run scope: the synthetic marker → ``synthetic_fixture``;
    otherwise the pipeline's own run scope value."""

    if getattr(context, "synthetic", False):
        return "synthetic_fixture"
    return str(context.semantic.payload.run_scope.value)


def regime_stage_plan_facts(spec) -> RegimeStudyStagePlanFacts:
    return RegimeStudyStagePlanFacts(
        feature_bundle_ids=tuple(spec.feature_bundle_ids),
        stage_values=tuple(stage.value for stage in spec.stage_plan),
        label_policy_id=spec.label_policy_id,
        model_protocol_id=spec.model_protocol_id,
    )


def regime_readiness_reason(spec, *, store_root=None, run_scope: str | None = None) -> str | None:
    """The capability-readiness reason of the spec's regime study (None when
    absent or launchable). With a store root the frozen authority of a
    model-bearing request is verified-loaded (fail before any path)."""

    request = regime_request(spec)
    if request is None:
        return None
    return regime_study_block_reason(
        request,
        regime_stage_plan_facts(spec),
        store_root=store_root,
        run_scope=run_scope or str(spec.run_scope.value),
    )


def regime_wiring_problems(spec, wiring) -> list[str]:
    request = regime_request(spec)
    if request is None:
        return []
    problems: list[str] = []
    if request.is_panel and getattr(wiring, "context_bar_source", None) is None:
        problems.append(
            "a panel-grain regime study is planned but no context_bar_source is wired "
            "(the panel is materialized from a VERIFIED replay-chart artifact only)"
        )
    if getattr(wiring, "candidate_view_source", None) is None:
        problems.append("a regime study is planned but no candidate_view_source is wired")
    return problems


# ── S05 ──────────────────────────────────────────────────────────────────────


def select_panel_chart(charts_by_child: Mapping[str, tuple[str, ...]]) -> tuple[str, str]:
    """``(core_replay_id, chart_id)`` under ``PANEL_CHART_SELECTION_POLICY_ID``;
    refuses (typed) when no child carries a chart or the selected child
    carries more than one."""

    if not charts_by_child:
        raise ValueError(
            "a panel-grain regime study requires the replay-chart artifacts of "
            "04_build_or_reuse_replay_charts (none recorded)"
        )
    core_replay_id = min(str(key) for key in charts_by_child)
    chart_ids = tuple(str(chart_id) for chart_id in charts_by_child[core_replay_id])
    if len(chart_ids) != 1:
        raise ValueError(
            f"child {core_replay_id[:12]}… carries {len(chart_ids)} replay-chart ids under "
            f"{PANEL_CHART_SELECTION_POLICY_ID}; the panel source must be exactly one chart"
        )
    return core_replay_id, chart_ids[0]


def _verified_panel_chart(context, chart_id: str):
    """Ask the wired seam for ``chart_id`` and refuse anything but a verified
    artifact of exactly that identity (adversarial R6.1 S1: the seam's
    output identity is never trusted)."""

    source = context.wiring.context_bar_source
    if source is None:
        raise ValueError("no context_bar_source is wired for the panel grain")
    replay = source(chart_id)
    loaded_id = getattr(replay, "artifact_id", None)
    if loaded_id != chart_id:
        raise ValueError(
            f"context_bar_source returned replay-chart artifact {str(loaded_id)[:12]}… for "
            f"the requested chart {chart_id[:12]}…; refusing — the panel binds the chart it "
            "asked for"
        )
    return replay


def s05_regime_observation(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """Persist every bundle view; resolve the regime observation source and
    the protocol; verify the frozen authority of a model-bearing request."""

    spec = context.semantic.payload
    request = regime_request(spec)
    store_root = context.store_root
    persisted_views: dict[str, str] = {}
    for bundle_key, envelope in context.bundle_views.items():
        stored = save_bundle_feature_view(store_root, envelope, context.bundle_frames[bundle_key])
        context.bundle_views[bundle_key] = stored
        persisted_views[bundle_key] = stored.bundle_feature_view_id
    if request is None:
        return (), {"__bundle_views_persisted__": persisted_views}, ""
    regime: dict[str, Any] = {"request": request}
    outputs: list[str] = []
    if request.is_panel:
        source_core_replay_id, chart_id = select_panel_chart(
            dict(context.regime.get("charts_by_child", {}))
        )
        replay = _verified_panel_chart(context, chart_id)
        research_subject = getattr(context.wiring, "research_subject", None)
        scope_kwargs = (
            {
                "research_subject_id": research_subject.subject_id,
                "evaluation_dates": tuple(research_subject.evaluation_dates),
            }
            if research_subject is not None
            else {}
        )
        envelope, panel, validity = materialize_context_bar_panel(
            replay, panel_interval_seconds=int(request.panel_interval_seconds or 0),
            **scope_kwargs,
        )
        if envelope.payload.replay_chart_artifact_id != chart_id:  # pragma: no cover
            raise ValueError("the materialized panel does not bind the selected chart")
        stored_panel, _reused = save_context_bar_panel_artifact(
            store_root, envelope, panel, validity
        )
        regime["panel_envelope"] = stored_panel
        regime["panel_frame"] = panel
        regime["chart_id"] = chart_id
        regime["panel_source_core_replay_id"] = source_core_replay_id
        regime["replay_chart_source_pair_sha256"] = canonical_contract_sha256(
            stored_panel.payload.source_pair.model_dump(mode="json")
        )
        observation = RegimeObservationSourceRef(
            source_kind="context_bar_panel",
            artifact_id=stored_panel.context_bar_panel_artifact_id,
        )
        outputs.append(stored_panel.context_bar_panel_artifact_id)
        protocol = resolve_study_protocol(
            request, observation_source_artifact_id=stored_panel.context_bar_panel_artifact_id
        )
    else:
        view_id = persisted_views.get(request.input_feature_bundle_key)
        if view_id is None:
            raise ValueError(
                f"the regime study bundle {request.input_feature_bundle_key!r} was not "
                "materialized as a bundle view"
            )
        observation = RegimeObservationSourceRef(
            source_kind="bundle_feature_view", artifact_id=view_id
        )
        protocol = resolve_study_protocol(request, observation_source_artifact_id=None)
    persist_regime_protocol(store_root, protocol)
    outputs.append(protocol.resolved_regime_protocol_id)
    regime["observation_ref"] = observation
    regime["protocol"] = protocol
    if request.requires_supervision:
        # D1: the resolved protocol MUST be the frozen decision's protocol
        authority = verify_frozen_authority(
            store_root,
            request,
            expected_protocol_id=protocol.resolved_regime_protocol_id,
            run_scope=regime_run_scope(context),
        )
        regime["authority"] = authority
    context.regime.update(regime)
    record = {
        "__bundle_views_persisted__": persisted_views,
        "__regime_observation__": {
            "observation_granularity": request.observation_granularity.value,
            "source_kind": observation.source_kind,
            "artifact_id": observation.artifact_id,
            "resolved_regime_protocol_id": protocol.resolved_regime_protocol_id,
            "chart_id": regime.get("chart_id"),
            # S1/F13: the LOADED artifact id, the declared selection policy, the
            # selected child and the loaded pair identity (panel grain only)
            "replay_chart_artifact_id": (
                regime["panel_envelope"].payload.replay_chart_artifact_id
                if request.is_panel
                else None
            ),
            "panel_chart_selection_policy_id": (
                PANEL_CHART_SELECTION_POLICY_ID if request.is_panel else None
            ),
            "panel_source_core_replay_id": regime.get("panel_source_core_replay_id"),
            "replay_chart_source_pair_sha256": regime.get("replay_chart_source_pair_sha256"),
            "frozen_authority_verified": request.requires_supervision,
        },
    }
    note = (
        f"; regime observation = {observation.source_kind} {observation.artifact_id[:12]}…, "
        f"protocol {protocol.resolved_regime_protocol_id[:12]}… persisted"
    )
    return tuple(outputs), record, note


# ── S06 ──────────────────────────────────────────────────────────────────────


def _observation_frame(context) -> pd.DataFrame:
    request = context.regime["request"]
    if request.is_panel:
        return context.regime["panel_frame"]
    return context.bundle_frames[request.input_feature_bundle_key]


def s06_regime_coverage(context) -> dict[str, Any] | None:
    """Input coverage of the regime features + the stamped floors (before
    folds exist — the per-fold preview is S08's)."""

    request = regime_request(context.semantic.payload)
    if request is None:
        return None
    frame = _observation_frame(context)
    inputs = list(request.resolved_input_features)
    coverage = {
        name: (float(frame[name].notna().mean()) if len(frame) and name in frame else 0.0)
        for name in inputs
    }
    with_inputs = int(frame[inputs].notna().any(axis=1).sum()) if len(frame) else 0
    grain = ObservationGranularity(request.observation_granularity)
    floors = {
        "minimum_training_observations": sample_adequacy_minimum(grain),
        "minimum_cluster_rows_per_fold": REGIME_PROPOSED_DEFAULTS[
            "minimum_cluster_rows_per_fold"
        ]["value"],
        "minimum_cluster_occupancy_fraction": REGIME_PROPOSED_DEFAULTS[
            "minimum_cluster_occupancy_fraction"
        ]["value"],
        "floor_is_flat_per_grain": True,
        "balanced_cluster_rows_lower_bound": int(request.resolved_cluster_count)
        * int(REGIME_PROPOSED_DEFAULTS["minimum_cluster_rows_per_fold"]["value"]),
    }
    return {
        "observation_granularity": grain.value,
        "rows_total": int(len(frame)),
        "rows_with_inputs": with_inputs,
        "per_feature_nonnull_fraction": coverage,
        "floors": floors,
        "resolved_cluster_count": int(request.resolved_cluster_count),
    }


# ── S08 ──────────────────────────────────────────────────────────────────────


def s08_regime_folds(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """Persist the schedule + the candidate fold-set artifact (labeled when
    S07 ran, label-free otherwise) and, for panel studies, the panel fold set
    under the SAME schedule; emit the per-fold adequacy preview."""

    request = regime_request(context.semantic.payload)
    if request is None:
        return (), {}, ""
    store_root = context.store_root
    view_frame = context.view.frame
    research_subject = getattr(context.wiring, "research_subject", None)
    days = (
        tuple(research_subject.evaluation_dates)
        if research_subject is not None
        else tuple(sorted(set(view_frame["trading_day"].astype(str))))
    )
    schedule = derive_fold_schedule(days)
    persist_fold_schedule(store_root, schedule)
    labeled = context.folds is not None
    if context.folds is None:
        context.folds = build_candidate_folds_from_schedule(
            view_frame, authorized_trading_days=days
        )
    candidate_view_id = context.bundle_views[
        request.input_feature_bundle_key
        if not request.is_panel
        else context.semantic.payload.feature_bundle_ids[0]
    ].bundle_feature_view_id
    candidate_fold_set, definitions = build_fold_set_artifact(
        context.folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=candidate_view_id,
        minimum_train_observations=sample_adequacy_minimum(
            ObservationGranularity.CANDIDATE_STAGE_ROW
        ),
        labeled=labeled,
    )
    persist_fold_set_artifact(store_root, candidate_fold_set, definitions)
    outputs = [schedule.fold_schedule_id, candidate_fold_set.fold_set_artifact_id]
    regime_fold_set = candidate_fold_set
    regime_folds = context.folds
    record: dict[str, Any] = {
        "fold_schedule_id": schedule.fold_schedule_id,
        "candidate_fold_set_artifact_id": candidate_fold_set.fold_set_artifact_id,
        "candidate_fold_set_id": candidate_fold_set.payload.fold_set_id,
        "candidate_folds_labeled": labeled,
        # F7: the schedule's days are the candidate view's OBSERVED trading
        # days (charter allowlist ∩ observed) — the one day source of S08
        "authorized_trading_days": list(days),
        "trading_days_source": (
            "frozen_research_logical_calendar_v1"
            if research_subject is not None else "candidate_view_observed_days"
        ),
        "days_without_labels": (
            sorted(set(days) - set(context.labeled["trading_day"].astype(str)))
            if context.labeled is not None
            else []
        ),
    }
    if request.is_panel:
        panel_frame = context.regime["panel_frame"]
        panel_folds = build_context_bar_panel_folds(panel_frame, authorized_trading_days=days)
        panel_fold_set, panel_definitions = build_fold_set_artifact(
            panel_folds,
            schedule=schedule,
            observation_grain=ObservationGranularity.CONTEXT_BAR_PANEL,
            observation_source_artifact_id=(
                context.regime["panel_envelope"].context_bar_panel_artifact_id
            ),
            minimum_train_observations=sample_adequacy_minimum(
                ObservationGranularity.CONTEXT_BAR_PANEL
            ),
            labeled=False,
        )
        assert_same_fold_schedule(candidate_fold_set.payload, panel_fold_set.payload)
        persist_fold_set_artifact(store_root, panel_fold_set, panel_definitions)
        outputs.append(panel_fold_set.fold_set_artifact_id)
        regime_fold_set = panel_fold_set
        regime_folds = panel_folds
        record["panel_fold_set_artifact_id"] = panel_fold_set.fold_set_artifact_id
        record["panel_fold_set_id"] = panel_fold_set.payload.fold_set_id
    preview = preview_sample_adequacy(
        _observation_frame(context),
        regime_folds,
        observation_granularity=request.observation_granularity,
        resolved_cluster_count=request.resolved_cluster_count,
        resolved_input_features=request.resolved_input_features,
    )
    record["sample_adequacy_preview"] = preview.model_dump(mode="json")
    context.regime.update(
        {
            "schedule": schedule,
            "candidate_fold_set": candidate_fold_set,
            "regime_fold_set": regime_fold_set,
            "regime_folds": regime_folds,
            "days": days,
        }
    )
    valid = sum(1 for fold in regime_folds.folds if fold.valid)
    note = (
        f"; regime schedule {schedule.fold_schedule_id[:12]}… + fold-set artifact(s) persisted "
        f"({valid} valid regime fold(s); expected adequacy {preview.expected_gate_outcome})"
    )
    return tuple(outputs), record, note


# ── S09 ──────────────────────────────────────────────────────────────────────


def s09_regime_fit(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """S09a through the executor; S09b/S09c only for model-bearing requests."""

    request = regime_request(context.semantic.payload)
    if request is None:
        return (), {}, ""
    regime = context.regime
    store_root = context.store_root
    candidate_as_of = None
    if request.is_panel:
        # the PERSISTED primary bundle view (S05) is the candidate as-of source:
        # the executor verified-loads it and stamps the loaded envelope id
        # (adversarial R6.1 F1 — never the in-memory frame)
        primary = context.semantic.payload.feature_bundle_ids[0]
        candidate_as_of = RegimeObservationSourceRef(
            source_kind="bundle_feature_view",
            artifact_id=context.bundle_views[primary].bundle_feature_view_id,
        )
    result = execute_regime_protocol(
        store_root,
        protocol=regime["protocol"],
        observation_source=regime["observation_ref"],
        fold_set_artifact_id=regime["regime_fold_set"].fold_set_artifact_id,
        candidate_as_of_source=candidate_as_of,
        candidate_as_of_stage=request.observation_stage,
        bootstrap_refits=request.bootstrap_refits,
        **(
            {"reuse_completed": True}
            if getattr(context.wiring, "research_subject", None) is not None else {}
        ),
    )
    regime["execution"] = result
    assessment_id = result.regime_capability_assessment_id
    if request.requires_supervision and (
        assessment_id != request.required_capability_assessment_id
    ):
        raise ValueError(
            "S09a reproduced a capability assessment "
            f"{assessment_id[:12]}… that is not the frozen required_capability_assessment_id "
            f"{str(request.required_capability_assessment_id)[:12]}…; a model-bearing run "
            "never proceeds on a different assessment"
        )
    oos_id = result.oos_assignment.regime_oos_assignment_id
    outputs = [assessment_id, *result.regime_fit_ids, oos_id]
    record: dict[str, Any] = {
        "S09a": {
            "resolved_regime_protocol_id": result.protocol.resolved_regime_protocol_id,
            "source_artifact_ids": list(result.source_artifact_ids),
            "observation_matrix_hash": result.observation_matrix_hash,
            "fold_set_artifact_id": result.fold_set_artifact_id,
            "fold_schedule_id": result.fold_schedule_id,
            "regime_fit_ids": list(result.regime_fit_ids),
            # (fits_reused is an ATTEMPT fact — it never enters the semantic
            # stage sidecar, which must be byte-identical across attempts)
            "regime_capability_assessment_id": assessment_id,
            "regime_oos_assignment_id": result.oos_assignment.regime_oos_assignment_id,
            "gates_passed": bool(result.run.assessment.payload.gates_passed),
            "gate_failures": list(result.run.assessment.payload.gate_failures),
        }
    }
    if getattr(result, "regime_execution_request_id", None) is not None:
        outputs.append(result.regime_execution_request_id)
        record["S09a"]["regime_execution_request_id"] = result.regime_execution_request_id
    gate_note = (
        "passed"
        if record["S09a"]["gates_passed"]
        else "FAILED: " + ", ".join(record["S09a"]["gate_failures"])
    )
    reused_count = sum(1 for flag in result.fits_reused if flag)
    note = (
        f"; regime S09a: {len(result.regime_fit_ids)} fit(s) ({reused_count} reused by "
        f"reproduction), assessment {assessment_id[:12]}… (gates {gate_note}), descriptive "
        f"OOS assignment {oos_id[:12]}…"
    )
    if request.requires_supervision:
        supervised_outputs, supervised_record, supervised_note = _s09_supervised(context)
        outputs.extend(supervised_outputs)
        record.update(supervised_record)
        note += supervised_note
    return tuple(outputs), record, note


def _s09_supervised(context) -> tuple[list[str], dict[str, Any], str]:
    """S09b (fold-local regime features) + S09c (controlled regime study /
    cohort model) — model-bearing requests only; every fit for the run
    happens here (S14 performs zero fitting)."""

    from ..ml import regime_supervised_stage  # noqa: PLC0415

    return regime_supervised_stage.run_supervised_substeps(context)


# ── S10 ──────────────────────────────────────────────────────────────────────


def _fallback_as_of(days: tuple[str, ...]) -> str:
    last = datetime.strptime(days[-1], "%Y-%m-%d").date()
    return datetime.combine(last, time(17, 0), tzinfo=_ET).isoformat()


def s10_regime_diagnostics(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """Deterministic decisions for descriptive runs; frozen-authority
    re-verification for model-bearing runs; the diagnostics sidecar."""

    request = regime_request(context.semantic.payload)
    if request is None:
        return (), {}, ""
    regime = context.regime
    result = regime["execution"]
    assessment = result.run.assessment
    # R6.1-FIX: the evidence as-of instant is read from the VERIFIED fit
    # assignment frames (the exact stored bytes), never the in-memory run frame
    decided_at = regime_evidence_as_of(
        result.consulted_assignments, fallback=_fallback_as_of(regime["days"])
    )
    refusals: list[str] = []
    if request.requires_supervision:
        authority = verify_frozen_authority(
            context.store_root,
            request,
            expected_protocol_id=result.protocol.resolved_regime_protocol_id,
            run_scope=regime_run_scope(context),
        )
        decisions = (authority.promotion,)
        authority_source = "frozen_owner_evidence"
    else:
        decisions = build_s10_decisions(
            context.store_root,
            protocol=result.protocol,
            assessment=assessment,
            request=request,
            decided_at=decided_at,
        )
        authority_source = "s10_structural"
    final = decisions[-1]
    regime["decisions"] = decisions
    regime["decision"] = final
    regime["authority_source"] = authority_source
    payload = assessment.payload
    diagnostics: dict[str, Any] = {
        "request": request.model_dump(mode="json"),
        "frozen_authority": {
            "regime_promotion_decision_id": request.regime_promotion_decision_id,
            "owner_decision_artifact_id": request.owner_decision_artifact_id,
            "required_capability_assessment_id": request.required_capability_assessment_id,
        },
        "authority_source": authority_source,
        "resolved_regime_protocol_id": result.protocol.resolved_regime_protocol_id,
        "regime_capability_assessment_id": assessment.regime_capability_assessment_id,
        "regime_fit_ids": list(result.regime_fit_ids),
        "regime_oos_assignment_id": result.oos_assignment.regime_oos_assignment_id,
        "fold_schedule_id": result.fold_schedule_id,
        "regime_fold_set_artifact_id": result.fold_set_artifact_id,
        "candidate_fold_set_artifact_id": regime["candidate_fold_set"].fold_set_artifact_id,
        "gates": {
            "gates_passed": bool(payload.gates_passed),
            "gate_failures": list(payload.gate_failures),
            "coverage_gates_passed": bool(payload.coverage.coverage_gates_passed),
            "oos_assignment_available": bool(payload.oos_assignment_available),
            "oos_assignment_coverage": float(payload.coverage.oos_assignment_coverage),
            "minimum_training_observations_gate": int(
                payload.coverage.minimum_training_observations_gate
            ),
            "minimum_training_observations_observed": int(
                payload.coverage.minimum_training_observations_observed
            ),
        },
        "decisions": [
            {
                "regime_promotion_decision_id": decision.regime_promotion_decision_id,
                "status": decision.payload.status.value,
                "role": decision.payload.role.value,
                "previous_status": decision.payload.previous_status.value,
                "decided_at": decision.payload.decided_at,
            }
            for decision in decisions
        ],
        "final_status": final.payload.status.value,
        "decided_at": decided_at,
        "fold_feature_artifact_id": regime.get("fold_feature_artifact_id"),
        "supervised_study_ids": regime.get("supervised_study_ids", []),
        "paired_deltas": regime.get("paired_deltas", {}),
        "refusals": refusals,
        "sub_steps": ["S09a", *(["S09b", "S09c"] if request.requires_supervision else [])],
    }
    outputs = tuple(decision.regime_promotion_decision_id for decision in decisions)
    note = (
        f"; regime status {final.payload.status.value} ({authority_source}; decided_at "
        f"{decided_at})"
    )
    return outputs, diagnostics, note


# ── S14 ──────────────────────────────────────────────────────────────────────


def s14_regime_reports(context) -> tuple[tuple[str, ...], dict[str, Any], str]:
    """Stratified reports from PERSISTED artifacts only (zero fitting)."""

    request = regime_request(context.semantic.payload)
    if request is None or not request.stratified_reporting_requested:
        return (), {}, ""
    from ..ml import regime_report_stage  # noqa: PLC0415

    return regime_report_stage.build_reports(context)


# ── S15 ──────────────────────────────────────────────────────────────────────


def s15_regime_reload_failures(context) -> dict[str, str]:
    """Every regime artifact the run produced reloads through the verified
    stores; each failure is recorded by ``<store>/<id>`` with its sanitized
    reason (R6.1-FIX §3.8) — the publication gate derives from the record."""

    from .failure import sanitize_failure_message  # noqa: PLC0415

    request = regime_request(context.semantic.payload)
    if request is None:
        return {}
    regime = context.regime
    root = context.store_root
    execution = regime.get("execution")
    checks: list[tuple[str, str, object]] = [
        ("regime_protocols", regime["protocol"].resolved_regime_protocol_id, load_regime_protocol)
    ]
    if execution is not None:
        if getattr(execution, "regime_execution_request_id", None) is not None:
            from ..ml.regime_execution_cache import verify_cached_execution  # noqa: PLC0415

            checks.append(
                ("research_regime_executions", execution.regime_execution_request_id,
                 verify_cached_execution)
            )
        checks.extend(
            [
                (
                    "regime_assessments",
                    execution.regime_capability_assessment_id,
                    load_regime_assessment,
                ),
                (
                    "regime_oos_assignments",
                    execution.oos_assignment.regime_oos_assignment_id,
                    load_regime_oos_assignment,
                ),
                ("fold_sets", execution.fold_set_artifact_id, load_fold_set_artifact),
            ]
        )
    for decision in regime.get("decisions", ()):
        checks.append(
            ("regime_promotions", decision.regime_promotion_decision_id, load_regime_promotion)
        )
    if request.is_panel and "panel_envelope" in regime:
        checks.append(
            (
                "context_bar_panels",
                regime["panel_envelope"].context_bar_panel_artifact_id,
                load_context_bar_panel_artifact,
            )
        )
    elif not request.is_panel and "observation_ref" in regime:
        checks.append(
            (
                "bundle_feature_views",
                regime["observation_ref"].artifact_id,
                load_bundle_feature_view,
            )
        )
    fold_feature_id = regime.get("fold_feature_artifact_id")
    if fold_feature_id is not None:
        from ..ml.regime_fold_features import load_regime_fold_features  # noqa: PLC0415

        checks.append(("regime_fold_features", fold_feature_id, load_regime_fold_features))
    # review B-03: every executed-trade table S14 stratified over must reload
    # AND still hash to the evidence S14 recorded; every stratified report
    # must reload through its store
    if regime.get("children_evidence"):
        from .executed_trade_table import load_executed_trade_table  # noqa: PLC0415

        def _table_loader(expected_sha256: str):
            def _load(root, table_id):
                loaded = load_executed_trade_table(root, table_id)
                if loaded.envelope.executed_trade_table_sha256 != expected_sha256:
                    raise ValueError(
                        f"executed-trade table {table_id[:12]}… no longer hashes to the "
                        "evidence S14 recorded"
                    )
                return loaded

            return _load

        for _core, evidence in sorted(regime["children_evidence"].items()):
            checks.append(
                (
                    "executed_trade_tables",
                    str(evidence["executed_trade_table_id"]),
                    _table_loader(str(evidence["executed_trade_table_sha256"])),
                )
            )
    if regime.get("stratified_report_ids"):
        from ..ml.regime_stratification_service import (  # noqa: PLC0415
            load_regime_stratified_report,
        )

        for report_id in regime["stratified_report_ids"]:
            checks.append(
                ("regime_stratified_reports", str(report_id), load_regime_stratified_report)
            )
    failures: dict[str, str] = {}
    for store_name, artifact_id, loader in checks:
        try:
            loader(root, artifact_id)
        except Exception as error:  # noqa: BLE001 — recorded, typed, never silent
            failures[f"{store_name}/{artifact_id}"] = sanitize_failure_message(str(error))
    return failures


def s15_regime_reload_ok(context) -> bool:
    """Backward-compatible boolean over :func:`s15_regime_reload_failures`."""

    return not s15_regime_reload_failures(context)
