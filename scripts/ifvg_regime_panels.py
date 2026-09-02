"""V1 KMeans regime dashboard panels (R6 + R6.1; FUX §35 R6/R6.1 rows).

Coverage, occupancy, per-fold stability, ASSIGNMENT, and STRATIFICATION
views over EXACT-ID persisted regime artifacts; proposal/default stamps;
the sample-adequacy blocked state; the context-panel grain identity; the
promotion decision (role/status) view with the loaded OWNER DECISION
artifact (provenance, values, effective window) when one is referenced;
the stratified RESULT views of the five ML §5.5 comparison classes by
exact report id (nominal ids, development badge, no counterfactual claim,
typed rows, never a selection input); and the post-V1 algorithms as
visible-disabled planned entries (with the mandatory spectral warning
text). Thin widget layer — registry state from the regime registries,
every artifact read a manifest-verified exact-ID store load (JSON
envelopes and the Arrow assignment frame only — the joblib sidecar is
never unpickled by the UI), every error sanitized. Cluster ids are
NOMINAL: nothing here ranks regimes, and no control can promote, launch,
rank, or retrain anything. R6.1 auto-fills the exact ids of the selected
pipeline run (manifest-verified provider defaults); the text inputs remain
the only way to address an artifact and nothing writes session state.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import (
    PIPELINE_STATE_PREFIX,
    dev_only_badge,
    identity_block,
    render_empty_state,
    sanitize_error,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

__all__ = ["render_regime_lane"]

_REG = f"{PIPELINE_STATE_PREFIX}regime_"
_TIMELINE_ROWS = 50
_DEVELOPMENT_CAPTION = (
    "Development · descriptive research only · no counterfactual claim · "
    "never a selection input · cluster ids are NOMINAL"
)
_RUNGS_CAPTION = (
    "prevalence + logistic + bundle-CatBoost rungs on identical comparison rows "
    "(research-only; the pooled ladder is retained beside every stratum)"
)


def _fmt(value, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _render_algorithm_registry(st_module) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_algorithms import (  # noqa: PLC0415
        REGIME_ALGORITHM_REGISTRY,
    )

    st_module.markdown(
        "**Regime algorithms** — V1 implements `kmeans_v1` only; every "
        "post-V1 entry is visible, pinned, and fail-closed (no fit "
        "implementation is callable, and no planned protocol policy executes)"
    )
    st_module.dataframe(
        [
            {
                "Algorithm": entry.algorithm_key,
                "State": (
                    "implemented (V1)"
                    if entry.implementation_status == "implemented"
                    else f"planned — {entry.planned_release}"
                ),
                "OOS-capable": "yes" if entry.oos_capable else "no",
                "Initialization policy": entry.initialization_policy,
                "Status / reason": entry.refusal_reason or "active",
            }
            for entry in REGIME_ALGORITHM_REGISTRY.values()
        ],
        width="stretch",
        hide_index=True,
    )
    spectral = REGIME_ALGORITHM_REGISTRY["spectral_clustering_train_only_v1"]
    if spectral.mandatory_warning_text:
        st_module.warning(f"**{spectral.mandatory_warning_text}**", icon="⚠️")


def _render_proposal_stamps(st_module) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (  # noqa: PLC0415
        REGIME_PROPOSED_DEFAULTS,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (  # noqa: PLC0415
        STRATIFICATION_REGISTERED_BUDGETS,
    )

    st_module.markdown(
        "**Scientific defaults** — every value below is a "
        "`proposed_protocol_default`: owner ratification is required before "
        "any regime output becomes feature-eligible (nothing is promoted "
        "because fitting succeeded); the `registered_storage_budget` rows are "
        "D15 capacity limits of the report-local event-regime summary, not "
        "research defaults"
    )
    st_module.dataframe(
        [
            {
                "Default": name,
                "Proposed value": str(entry["value"]),
                "Stamp": entry["stamp"],
                "Owner decision": entry["owner_decision"],
                "Ratification required": "yes — before feature-eligible",
            }
            for name, entry in REGIME_PROPOSED_DEFAULTS.items()
        ]
        + [
            {
                "Default": name,
                "Proposed value": str(entry["value"]),
                "Stamp": entry["stamp"],
                "Owner decision": f"{entry['decision']} ({entry['budget_id']})",
                "Ratification required": "no — storage budget (refuses before publication)",
            }
            for name, entry in STRATIFICATION_REGISTERED_BUDGETS.items()
        ],
        width="stretch",
        hide_index=True,
    )


def _render_model_card(st_module, store_root: Path, defaults: Mapping[str, str]) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_assessment,
        load_regime_protocol,
    )

    st_module.markdown("**Regime model card** (exact-ID loads — stores are never listed)")
    if defaults:
        st_module.caption(
            "Exact ids auto-filled from the selected pipeline run's persisted "
            "stage evidence (manifest-verified); edit any input to address "
            "another artifact."
        )
    protocol_id = st_module.text_input(
        "Resolved regime protocol id (64-hex)",
        value=defaults.get("protocol_id", ""),
        key=f"{_REG}protocol_id",
    ).strip()
    assessment_id = st_module.text_input(
        "Capability assessment id (64-hex)",
        value=defaults.get("assessment_id", ""),
        key=f"{_REG}assessment_id",
    ).strip()
    fit_id = st_module.text_input(
        "Regime fit id (64-hex) — assignment / stratification view",
        value=defaults.get("fit_id", ""),
        key=f"{_REG}fit_id",
    ).strip()
    decision_id = st_module.text_input(
        "Promotion decision id (64-hex) — role / status view",
        value=defaults.get("decision_id", ""),
        key=f"{_REG}decision_id",
    ).strip()
    report_id = st_module.text_input(
        "Stratified report id (64-hex) — stratified RESULT view (ML §5.5 classes)",
        value=defaults.get("report_id", ""),
        key=f"{_REG}report_id",
    ).strip()
    if not protocol_id:
        st_module.caption(
            "Paste the exact `resolved_regime_protocol_id` (and optionally "
            "the `regime_capability_assessment_id`, a `regime_fit_id`, a "
            "`regime_promotion_decision_id`, and a `regime_stratified_report_id`) "
            "from a persisted regime run — stores are never listed."
        )
        return
    try:
        protocol = load_regime_protocol(store_root, protocol_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    payload = protocol.payload
    grain = payload.observation_granularity.value
    if payload.panel_interval_seconds is not None:
        grain += (
            f" · interval {payload.panel_interval_seconds}s · panel source "
            f"{payload.panel_source_artifact_id[:12]}… · as-of "
            f"{payload.panel_as_of_policy_id}"
        )
    st_module.dataframe(
        [
            {
                "Field": "Algorithm",
                "Value": f"{payload.algorithm_key} (v{payload.algorithm_version})",
            },
            {"Field": "Observation grain", "Value": grain},
            {"Field": "Observation stage", "Value": payload.observation_stage.value},
            {
                "Field": "Cluster count",
                "Value": (
                    f"{payload.resolved_cluster_count} "
                    f"({payload.cluster_count_policy} — proposed_protocol_default)"
                ),
            },
            {"Field": "Initialization policy", "Value": payload.initialization_policy},
            {"Field": "OOS assignment policy", "Value": payload.out_of_sample_assignment_policy},
            {"Field": "Alignment policy", "Value": payload.cluster_label_alignment_policy},
            {"Field": "Preprocessing", "Value": (
                f"{payload.missingness_policy} · {payload.winsorization_policy} · "
                f"{payload.scaler_policy} (fold-local)"
            )},
            {"Field": "Fit scope", "Value": payload.fit_scope},
            {
                "Field": "Input features",
                "Value": f"{len(payload.resolved_input_features)}: "
                + ", ".join(payload.resolved_input_features),
            },
            {"Field": "Pinned parameters hash", "Value": payload.pinned_parameters_hash},
        ],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Resolved regime protocol id", protocol_id)
    identity_block(
        st_module, "Input feature bundle (resolved id)", payload.input_feature_bundle_ref
    )
    if assessment_id:
        try:
            assessment = load_regime_assessment(store_root, assessment_id)
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            render_empty_state(
                st_module, "artifact_unavailable", detail=sanitize_error(error)
            )
        else:
            result = assessment.payload
            if result.resolved_regime_protocol_id != protocol_id:
                render_empty_state(
                    st_module,
                    "artifact_unavailable",
                    detail="the assessment references a different regime protocol id",
                )
            else:
                _render_assessment(st_module, result, assessment_id)
    if fit_id:
        _render_assignment_view(st_module, store_root, fit_id, protocol_id)
    if decision_id:
        _render_promotion_view(st_module, store_root, decision_id, protocol_id)
    if report_id:
        _render_stratified_report_view(st_module, store_root, report_id, protocol_id)


def _render_assessment(st_module, result, assessment_id: str) -> None:
    coverage = result.coverage
    if not result.gates_passed and "sample_adequacy" in result.gate_failures:
        render_empty_state(
            st_module,
            "insufficient_sample",
            detail=(
                f"minimum training observations per fold: "
                f"{coverage.minimum_training_observations_gate} "
                f"(proposed_protocol_default); observed: "
                f"{coverage.minimum_training_observations_observed} — the fit "
                "stays descriptive; promotion is blocked, k is never shrunk"
            ),
        )
    st_module.markdown("**Coverage**")
    st_module.dataframe(
        [
            {"Field": "Rows total", "Value": str(coverage.rows_total)},
            {"Field": "Rows assigned", "Value": str(coverage.rows_assigned)},
            {
                "Field": "Typed nulls",
                "Value": (
                    "; ".join(
                        f"{reason}: {count}"
                        for reason, count in sorted(coverage.rows_typed_null.items())
                    )
                    or "—"
                ),
            },
            {
                "Field": "OOS assignment coverage",
                "Value": f"{coverage.oos_assignment_coverage:.4f}",
            },
            {
                "Field": "Sample adequacy",
                "Value": (
                    f"observed {coverage.minimum_training_observations_observed} / "
                    f"gate {coverage.minimum_training_observations_gate}"
                ),
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown("**Per-fold coverage** (valid-assignment fraction per fold)")
    st_module.dataframe(
        [
            {"Fold": str(fold_index), "Coverage": f"{share:.4f}"}
            for fold_index, share in sorted(coverage.per_fold_coverage.items())
        ]
        or [{"Fold": "—", "Coverage": "—"}],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown("**Fit identities** (one fold-local fit per valid fold)")
    st_module.dataframe(
        [
            {"#": str(index), "regime_fit_id": fit_id}
            for index, fit_id in enumerate(result.regime_fit_ids)
        ]
        or [{"#": "—", "regime_fit_id": "— (no valid fold)"}],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Capability assessment id", assessment_id)
    identity_block(st_module, "Fold set id", result.fold_set_id)
    st_module.markdown(
        "**Occupancy** (canonical reporting ids are NOMINAL — no ordering "
        "or ranking is implied)"
    )
    st_module.dataframe(
        [
            {"Regime (nominal)": f"regime {cluster}", "Occupancy": f"{share:.4f}"}
            for cluster, share in sorted(result.occupancy.items())
        ]
        or [{"Regime (nominal)": "—", "Occupancy": "—"}],
        width="stretch",
        hide_index=True,
    )
    _render_stability(st_module, result.stability)
    gates_word = "✓ passed" if result.gates_passed else "✕ failed"
    st_module.write(
        f"Capability gates: **{gates_word}**"
        + (
            f" — {', '.join(result.gate_failures)}"
            if result.gate_failures
            else ""
        )
        + " · promotion additionally requires the owner's ratification "
        "reference (P1-3); nothing here can promote."
    )


def _render_stability(st_module, stability) -> None:
    """R6.1 (D10/D11): every valid fold is bootstrapped; the promotion gate
    `minimum_bootstrap_aligned_ami_mean` applies to the protocol-wide
    MINIMUM fold mean; transition semantics are grain-specific."""

    st_module.markdown("**Stability** (reference-fold scalars + the protocol-wide gate)")
    st_module.dataframe(
        [
            {
                "Field": "Bootstrap aligned AMI (mean, reference fold)",
                "Value": _fmt(stability.bootstrap_aligned_ami_mean),
            },
            {
                "Field": "Bootstrap aligned AMI (5th percentile, reference fold)",
                "Value": _fmt(stability.bootstrap_aligned_ami_low),
            },
            {
                "Field": "Bootstrap refits per fold (requested / applied / total cap)",
                "Value": (
                    f"{stability.bootstrap_refits_per_fold_requested} / "
                    f"{stability.bootstrap_refits_per_fold_applied} / "
                    f"{stability.bootstrap_total_refit_cap}"
                ),
            },
            {
                "Field": "Protocol-wide minimum fold AMI mean",
                "Value": _fmt(stability.protocol_min_bootstrap_aligned_ami_mean),
            },
            {
                "Field": "Protocol-wide minimum fold AMI p05 (reported, not gated)",
                "Value": _fmt(stability.protocol_min_bootstrap_aligned_ami_p05),
            },
            {
                "Field": "Gate minimum_bootstrap_aligned_ami_mean (applied)",
                "Value": (
                    f"{stability.minimum_bootstrap_aligned_ami_mean_applied:.4f} · scope "
                    f"{stability.bootstrap_gate_scope} (owner decision 30; "
                    "proposed_protocol_default)"
                ),
            },
            {
                "Field": "Bootstrap fold coverage",
                "Value": _fmt(stability.bootstrap_fold_coverage),
            },
            {
                "Field": "Temporal transitions counted",
                "Value": str(
                    stability.candidate_event_pairs_counted
                    if stability.candidate_event_transition_matrix
                    else stability.temporal_transition_count
                ),
            },
            {
                "Field": "Temporal persistence (OOS timeline)",
                "Value": _fmt(
                    stability.candidate_event_persistence
                    if stability.candidate_event_transition_matrix
                    else stability.temporal_persistence
                ),
            },
            {"Field": "Temporal order policy", "Value": stability.temporal_order_policy},
            {"Field": "Session scheme", "Value": stability.session_scheme_id},
            {
                "Field": "Fold-to-fold recurrence (aligned centroid distance)",
                "Value": _fmt(stability.fold_to_fold_recurrence),
            },
            {"Field": "Alignment space", "Value": stability.alignment_space},
            {
                "Field": "Min centroid separation",
                "Value": _fmt(stability.separation_min_centroid_distance),
            },
            {
                "Field": "Silhouette (descriptive only — never promotes)",
                "Value": _fmt(stability.silhouette_descriptive),
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown("**Per-fold bootstrap stability** (every valid fold; fold-local seeds)")
    st_module.dataframe(
        [
            {
                "Fold": str(fold.fold_index),
                "Refits": str(fold.refit_count),
                "Aligned AMI mean": _fmt(fold.aligned_ami_mean),
                "Aligned AMI p05": _fmt(fold.aligned_ami_p05),
                "Undefined reason": fold.undefined_reason or "—",
            }
            for fold in stability.per_fold_bootstrap_stability
        ]
        or [{"Fold": "—", "Refits": "—", "Aligned AMI mean": "—", "Aligned AMI p05": "—"}],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown("**Per-cluster bootstrap agreement** (nominal ids)")
    st_module.dataframe(
        [
            {"Regime (nominal)": f"regime {cluster}", "Agreement": f"{share:.4f}"}
            for cluster, share in sorted(stability.per_cluster_agreement.items())
        ]
        or [{"Regime (nominal)": "—", "Agreement": "—"}],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown(
        "**Centroid profiles** (top-|z| scaled input features of the reference fold; nominal ids)"
    )
    st_module.dataframe(
        [
            {
                "Regime (nominal)": f"regime {cluster}",
                "Profile": "; ".join(f"{name} {value:+.3f}" for name, value in descriptors),
            }
            for cluster, descriptors in sorted(stability.semantic_descriptors.items())
        ]
        or [{"Regime (nominal)": "—", "Profile": "—"}],
        width="stretch",
        hide_index=True,
    )
    if stability.candidate_event_transition_matrix:
        st_module.markdown(
            "**Candidate-event transition matrix** (OOS test rows by as-of instant "
            "within each fold; trading-day + named-session resets; stamped maximum gap "
            f"{stability.candidate_event_maximum_gap_seconds} s — candidate events "
            "are not a regular time series)"
        )
        _render_matrix(st_module, stability.candidate_event_transition_matrix)
        dropped = dict(stability.candidate_event_pairs_dropped_boundary)
        st_module.caption(
            f"pairs counted {stability.candidate_event_pairs_counted} · dropped by gap "
            f"{stability.candidate_event_pairs_dropped_gap} · dropped at boundaries "
            + (", ".join(f"{k} {v}" for k, v in sorted(dropped.items())) or "none")
            + f" · persistence {_fmt(stability.candidate_event_persistence)}"
        )
    if stability.transition_matrix:
        st_module.markdown(
            "**Regime transition matrix** (OOS test bars, consecutive completed bars "
            "within a trading day)"
        )
        _render_matrix(st_module, stability.transition_matrix)
        st_module.caption(
            f"transitions counted {stability.temporal_transition_count} · persistence "
            f"{_fmt(stability.temporal_persistence)} · dropped by gap "
            f"{stability.panel_pairs_dropped_gap} · dropped at boundaries "
            f"{stability.panel_pairs_dropped_boundary}"
        )


def _render_matrix(st_module, matrix) -> None:
    st_module.dataframe(
        [
            {
                "From \\ To": f"regime {row_index}",
                **{
                    f"regime {column_index}": f"{value:.3f}"
                    for column_index, value in enumerate(row)
                },
            }
            for row_index, row in enumerate(matrix)
        ],
        width="stretch",
        hide_index=True,
    )


def _quantiles(values) -> str:
    import numpy as np  # noqa: PLC0415

    array = np.asarray([v for v in values if v is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return "—"
    p10, p50, p90 = np.quantile(array, [0.1, 0.5, 0.9])
    return f"p10 {p10:.4f} · p50 {p50:.4f} · p90 {p90:.4f}"


def _render_assignment_view(
    st_module, store_root: Path, fit_id: str, protocol_id: str
) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_diagnostics import (  # noqa: PLC0415
        oos_regime_timeline,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_fit_assignments,
    )

    st_module.markdown("**Assignment view** (exact fit id; per-row facts of ONE fold-local fit)")
    try:
        verified = load_regime_fit_assignments(store_root, fit_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    envelope, artifact, assignments = verified.envelope, verified.artifact, verified.frame
    if envelope.payload.resolved_regime_protocol_id != protocol_id:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="the fit references a different regime protocol id",
        )
        return
    st_module.dataframe(
        [
            {"Field": "Fold", "Value": str(artifact.fold_index)},
            {
                "Field": "Fit window",
                "Value": f"{envelope.payload.fit_start or '—'} → {envelope.payload.fit_end or '—'}",
            },
            {"Field": "Training rows", "Value": str(artifact.training_row_count)},
            {"Field": "Inertia", "Value": f"{artifact.inertia_or_loglik:.4f}"},
            {
                "Field": "Fitted parameter payload hash",
                "Value": artifact.fitted_parameter_payload_hash,
            },
            {
                "Field": "Training feature matrix hash",
                "Value": artifact.training_feature_matrix_hash,
            },
            {"Field": "Pipeline sidecar", "Value": artifact.preprocessing_pipeline_ref},
        ],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Regime fit id", fit_id)
    valid = assignments["valid"].astype(bool)
    st_module.markdown("**Coverage by partition**")
    rows = []
    for partition, group in assignments.groupby("partition", sort=True):
        group_valid = group["valid"].astype(bool)
        reasons = group.loc[~group_valid, "missing_reason"].value_counts().to_dict()
        rows.append(
            {
                "Partition": str(partition),
                "Rows": str(len(group)),
                "Valid": str(int(group_valid.sum())),
                "Typed nulls": "; ".join(f"{k}: {v}" for k, v in sorted(reasons.items())) or "—",
                "Assigned distance": _quantiles(group.loc[group_valid, "assigned_distance"]),
                "Margin d2−d1": _quantiles(group.loc[group_valid, "assignment_margin"]),
            }
        )
    st_module.dataframe(rows or [{"Partition": "—"}], width="stretch", hide_index=True)
    st_module.markdown(
        "**Stratification of the assignment frame** (nominal regime × partition; "
        "metric-by-regime stratification of strategy/model/prop results is the "
        "stratified RESULT view below, by exact report id)"
    )
    strata = []
    for (cluster, partition), group in (
        assignments[valid]
        .groupby(["canonical_reporting_cluster_id", "partition"], sort=True)
    ):
        strata.append(
            {
                "Regime (nominal)": f"regime {int(cluster)}",
                "Partition": str(partition),
                "Rows": str(len(group)),
                "Share of partition": (
                    f"{len(group) / max(1, int((assignments['partition'] == partition).sum())):.4f}"
                ),
                "Mean margin": f"{group['assignment_margin'].mean():.4f}",
            }
        )
    st_module.dataframe(
        strata or [{"Regime (nominal)": "—"}], width="stretch", hide_index=True
    )
    timeline = oos_regime_timeline(assignments)
    st_module.markdown(
        f"**Regime timeline** (OOS test rows by as-of timestamp; first {_TIMELINE_ROWS} of "
        f"{len(timeline)})"
    )
    st_module.dataframe(
        [
            {
                "As-of (UTC)": str(row.observation_ts_utc),
                "Row id": str(row.row_id),
                "Regime (nominal)": f"regime {int(row.canonical_reporting_cluster_id)}",
            }
            for row in timeline.head(_TIMELINE_ROWS).itertuples()
        ]
        or [{"As-of (UTC)": "—", "Row id": "—", "Regime (nominal)": "—"}],
        width="stretch",
        hide_index=True,
    )


def _render_promotion_view(
    st_module, store_root: Path, decision_id: str, protocol_id: str
) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_promotion,
    )

    st_module.markdown("**Promotion decision** (role / status — exact decision id)")
    try:
        decision = load_regime_promotion(store_root, decision_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    payload = decision.payload
    if payload.resolved_regime_protocol_id != protocol_id:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="the decision references a different regime protocol id",
        )
        return
    st_module.dataframe(
        [
            {"Field": "Role", "Value": payload.role.value},
            {"Field": "Status", "Value": payload.status.value},
            {"Field": "Previous status", "Value": payload.previous_status.value},
            {
                "Field": "Owner ratification",
                "Value": (
                    "present (verified owner-decision artifact — see below)"
                    if payload.owner_ratification_ref
                    else "absent — feature-eligible and beyond are unreachable"
                ),
            },
            {"Field": "Decided at", "Value": payload.decided_at},
            {"Field": "Capability assessment ref", "Value": payload.capability_assessment_ref},
            {
                "Field": "Previous decision ref",
                "Value": payload.previous_decision_ref or "— (first decision)",
            },
        ],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Promotion decision id", decision_id)
    st_module.caption(
        "A status change never rewrites the protocol or fit identities; "
        "execution-side roles are unrepresentable in V1 (S11 blocked)."
    )
    if payload.owner_ratification_ref:
        _render_owner_decision(st_module, store_root, payload.owner_ratification_ref)


def _render_owner_decision(st_module, store_root: Path, owner_id: str) -> None:
    """R6.1 (§6.F): the referenced owner decision is a VERIFIED store
    artifact — provenance, decisions 25/28/29/30 values, and the effective
    window; a bare reference that does not load renders as unavailable."""

    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (  # noqa: PLC0415
        load_owner_decision,
    )

    st_module.markdown("**Owner decision artifact** (exact id; verified store load)")
    try:
        artifact = load_owner_decision(store_root, owner_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail=(
                "the referenced owner decision is not a verified store artifact "
                f"(a bare 64-hex reference is not evidence): {sanitize_error(error)}"
            ),
        )
        return
    payload = artifact.payload
    st_module.dataframe(
        [
            {"Field": "Decision id", "Value": payload.decision_id},
            {"Field": "Provenance", "Value": payload.provenance},
            {"Field": "Author", "Value": payload.author},
            {"Field": "Approved at", "Value": payload.approved_at},
            {
                "Field": "Effective window",
                "Value": f"{payload.effective_from} → {payload.effective_to or 'open-ended'}",
            },
            {
                "Field": "Authorized transitions",
                "Value": ", ".join(payload.authorized_transitions),
            },
            {"Field": "Decision keys", "Value": ", ".join(payload.decision_keys)},
            {
                "Field": "Reviewed evidence refs",
                "Value": ", ".join(ref[:12] + "…" for ref in payload.reviewed_evidence_refs),
            },
            {"Field": "Supersedes", "Value": payload.supersedes or "— (not a supersession)"},
            {"Field": "Rationale", "Value": payload.rationale},
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.markdown("**Decision values** (25 algorithm · 28 grain · 29 k · 30 gates)")
    st_module.dataframe(
        [
            {
                "Key": key,
                "Value": (
                    "; ".join(f"{k}={v}" for k, v in sorted(dict(value).items()))
                    if isinstance(value, Mapping)
                    else str(value)
                ),
            }
            for key, value in payload.decision_values.items()
        ],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Owner decision artifact id", owner_id)
    identity_block(st_module, "Ratified regime protocol id", payload.resolved_regime_protocol_id)
    identity_block(st_module, "Reviewed capability assessment id", payload.capability_assessment_id)
    if payload.provenance == "synthetic_test_authorization_v1":
        st_module.caption(
            "Synthetic test authorization — lawful in the synthetic_fixture run "
            "scope only; it never authorizes a real study."
        )


def _render_stratified_report_view(
    st_module, store_root: Path, report_id: str, protocol_id: str
) -> None:
    """The stratified RESULT view of one ML §5.5 comparison class (exact
    report id; persisted artifacts only; nothing is rebuilt here)."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (  # noqa: PLC0415
        load_regime_stratified_report,
        load_regime_stratified_report_detail,
    )

    st_module.markdown("**Stratified result view** (exact report id)")
    try:
        envelope = load_regime_stratified_report(store_root, report_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    payload = envelope.payload
    if payload.resolved_regime_protocol_id != protocol_id:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="the report references a different regime protocol id",
        )
        return
    try:
        detail = load_regime_stratified_report_detail(store_root, report_id)
    except Exception as error:  # noqa: BLE001 — the report body still renders
        detail = {}
        st_module.caption(f"detail sidecar unavailable: {sanitize_error(error)}")
    gate = payload.gate
    st_module.caption(
        f"{_DEVELOPMENT_CAPTION} · badge `{payload.development_badge}` · "
        f"counterfactual claim: {payload.counterfactual_claim}"
    )
    st_module.dataframe(
        [
            {"Field": "Comparison class", "Value": payload.comparison_class.value},
            {"Field": "Interpretation", "Value": payload.interpretation},
            {"Field": "Status at report", "Value": gate.status_at_report.value},
            {"Field": "Role at report", "Value": gate.role_at_report.value},
            {"Field": "Minimum status required", "Value": gate.minimum_status_required.value},
            {"Field": "Authority source", "Value": gate.authority_source},
            {
                "Field": "Assignment evidence",
                "Value": (
                    f"{payload.assignment_evidence.observation_granularity.value} · "
                    f"{len(payload.assignment_evidence.regime_fit_ids)} fit(s) · OOS "
                    f"assignment {payload.assignment_evidence.regime_oos_assignment_id[:12]}…"
                ),
            },
            {"Field": "Formula version", "Value": payload.formula_version},
        ],
        width="stretch",
        hide_index=True,
    )
    identity_block(st_module, "Stratified report id", report_id)
    identity_block(st_module, "Gate promotion decision id", gate.regime_promotion_decision_id)
    if gate.owner_decision_artifact_id:
        identity_block(
            st_module, "Gate owner decision artifact id", gate.owner_decision_artifact_id
        )
    body = payload.body
    comparison_class = payload.comparison_class.value
    if comparison_class == "cohort_descriptive":
        _render_cohort_descriptive(st_module, body, detail)
    elif comparison_class == "stratified_frontier":
        _render_stratified_frontier(st_module, body)
    elif comparison_class == "stratified_prop":
        _render_stratified_prop(st_module, body, store_root, report_id)
    else:
        _render_modeled_body(st_module, body, comparison_class)


def _metric(value) -> str:
    return "—" if value is None else f"{value:.4f}"


def _render_cohort_descriptive(st_module, body, detail: Mapping[str, Any]) -> None:
    st_module.markdown(
        "**Cohort descriptive strata** (strategy metrics per nominal regime; the "
        "pooled stratum equals the child's own metrics)"
    )
    thin = [row for row in body.strata if row.typed_state == "insufficient_regime_partition"]
    st_module.dataframe(
        [
            {
                "Stratum": row.key.label,
                "Trades": str(row.executed_trades),
                "Share": f"{row.share_of_trades:.4f}",
                "State": row.typed_state,
                "Net expectancy R": _metric(row.metrics.net_expectancy_r if row.metrics else None),
                "Profit factor": _metric(row.metrics.profit_factor if row.metrics else None),
                "Max drawdown R": _metric(row.metrics.max_drawdown_r if row.metrics else None),
                "Pooled-model skill (brier)": (
                    _metric(dict(row.pooled_model_skill).get("brier"))
                    if row.pooled_model_skill
                    else "—"
                ),
            }
            for row in body.strata
        ],
        width="stretch",
        hide_index=True,
    )
    if thin:
        render_empty_state(
            st_module,
            "insufficient_regime_partition",
            detail=(
                ", ".join(row.key.label for row in thin)
                + f" — below the stamped minimum of {body.minimum_trades_per_regime_stratum} "
                "trades per regime stratum (proposed_protocol_default)"
            ),
        )
    st_module.caption(
        f"coverage {body.trades_regime_covered}/{body.trades_total} trades "
        f"({body.coverage_fraction:.4f}) · works only in regime "
        f"{', '.join(str(c) for c in body.works_only_in_regime) or 'none'} · top-regime "
        f"|net R| share {_metric(body.top_regime_abs_net_r_share)} · unassigned reasons "
        + (
            "; ".join(f"{k}: {v}" for k, v in sorted(body.unassigned_reasons.items()))
            or "none"
        )
    )
    identity_block(st_module, "Core replay id", body.core_replay_id)
    identity_block(st_module, "Executed-trade table sha256", body.executed_trade_table_sha256)


def _render_stratified_frontier(st_module, body) -> None:
    st_module.markdown(
        "**Stratified frontier** (child × nominal regime × metric; `on_frontier` "
        "is read from the POOLED frontier only)"
    )
    st_module.dataframe(
        [
            {
                "Child": cell.core_replay_id[:12] + "…",
                "On frontier (pooled)": (
                    "yes" if body.on_frontier.get(cell.core_replay_id) else "no"
                ),
                "Stratum": cell.key.label,
                "Trades": str(cell.executed_trades),
                "State": cell.typed_state,
                **{
                    metric: _metric(cell.metric_values.get(metric))
                    for metric in body.objective_metrics
                },
            }
            for cell in body.cells
        ]
        or [{"Child": "—"}],
        width="stretch",
        hide_index=True,
    )
    if any(cell.typed_state == "insufficient_regime_partition" for cell in body.cells):
        render_empty_state(
            st_module,
            "insufficient_regime_partition",
            detail="one or more child × regime cells are below the stamped minimum",
        )
    st_module.caption(
        f"frontier role: {body.frontier_role} — never a selection input · children "
        f"without strata: {', '.join(c[:12] for c in body.children_without_strata) or 'none'}"
    )
    identity_block(st_module, "Frontier id", body.frontier_id)


_SUMMARY_ROWS = 200
_SUMMARY_COLUMNS = (
    "account_simulation_id",
    "path_instance_id",
    "regime_stratum",
    "unattributable_reason",
    "event_type",
    "event_count",
    "amount_sum",
    "first_event_ordinal",
    "last_event_ordinal",
)


def _render_event_regime_summary(st_module, body, store_root: Path, report_id: str) -> None:
    """The D15 report-local event-regime summary (verified Parquet sidecar;
    aggregate rows only — never row-oriented event JSON)."""

    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratification_service import (  # noqa: PLC0415
        load_regime_stratified_report_summary,
    )

    st_module.markdown(
        "**Event-regime summary** (D15 report-local ZSTD Parquet — one row per "
        "simulation × path × regime stratum / typed reason × event type, keyed by "
        "the exact assignment evidence)"
    )
    try:
        table = load_regime_stratified_report_summary(store_root, report_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    budget = body.summary_budget
    st_module.caption(
        f"{table.num_rows} summary row(s) under `{budget.budget_id}` "
        f"(≤ {budget.max_summary_rows} rows / {budget.max_published_bytes} bytes; "
        f"registered_storage_budget) · first {min(_SUMMARY_ROWS, table.num_rows)} row(s) shown"
    )
    frame = table.to_pandas().head(_SUMMARY_ROWS)
    rows = [
        {
            "Simulation": str(row["account_simulation_id"])[:12] + "…",
            "Path": str(row["path_instance_id"]),
            "Stratum": str(row["regime_stratum"]),
            "Reason": (
                "—" if row["unattributable_reason"] is None else str(row["unattributable_reason"])
            ),
            "Event type": str(row["event_type"]),
            "Events": str(int(row["event_count"])),
            "Amount sum": "—" if row["amount_sum"] is None else f"{float(row['amount_sum']):.2f}",
            "Ordinals": f"{int(row['first_event_ordinal'])}–{int(row['last_event_ordinal'])}",
        }
        for row in frame[list(_SUMMARY_COLUMNS)].to_dict(orient="records")
    ]
    st_module.dataframe(rows or [{"Simulation": "—"}], width="stretch", hide_index=True)


def _render_stratified_prop(st_module, body, store_root: Path, report_id: str) -> None:
    st_module.markdown(
        "**Stratified prop events** (events attributed to the source trade's "
        "descriptive OOS regime; bootstrap/stress events never consult synthetic "
        "clocks; probabilities are never re-estimated)"
    )
    st_module.dataframe(
        [
            {
                "Firm": row.firm_label,
                "Mode": row.simulation_mode,
                "Stratum": row.key.label,
                "Events": str(row.events_total),
                "By type": "; ".join(
                    f"{k}: {v}" for k, v in sorted(row.event_counts_by_type.items())
                )
                or "—",
                "Payout (trader)": f"{row.payout_trader_amount_sum:.2f}",
                "Fees": f"{row.fee_amount_sum:.2f}",
                "Realized P&L": f"{row.realized_pnl_sum:.2f}",
            }
            for row in body.strata
        ]
        or [{"Firm": "—"}],
        width="stretch",
        hide_index=True,
    )
    st_module.caption(
        f"attribution policy {body.attribution_policy_id} · events attributed "
        f"{body.events_attributed}/{body.events_total} ({body.coverage_fraction:.4f}"
        f"{'; partial coverage' if body.partial_coverage else ''}) · probabilities "
        f"re-estimated: {body.probabilities_reestimated} · unattributable "
        + (
            "; ".join(f"{k}: {v}" for k, v in sorted(body.unattributable_by_reason.items()))
            or "none"
        )
    )
    if body.evidence_not_persisted:
        st_module.caption(
            "evidence_not_persisted (simulations under none_v0): "
            + ", ".join(sid[:12] + "…" for sid in body.evidence_not_persisted)
        )
    identity_block(st_module, "Core replay id", body.core_replay_id)
    _render_event_regime_summary(st_module, body, store_root, report_id)


def _render_modeled_body(st_module, body, comparison_class: str) -> None:
    """feature_only / cohort_model bodies: the generic scalar projection of
    the persisted body (typed rows stay typed; nothing is recomputed)."""

    st_module.markdown(f"**{comparison_class}** — {_RUNGS_CAPTION}")
    dumped = body.model_dump(mode="json") if hasattr(body, "model_dump") else dict(body)
    rows = [
        {"Field": key, "Value": str(value)}
        for key, value in sorted(dumped.items())
        if not isinstance(value, list | dict | tuple)
    ]
    st_module.dataframe(rows or [{"Field": "—", "Value": "—"}], width="stretch", hide_index=True)
    typed = [
        key
        for key, value in dumped.items()
        if isinstance(value, str) and value.startswith("insufficient_regime_partition")
    ]
    if typed:
        render_empty_state(
            st_module,
            "insufficient_regime_partition",
            detail=", ".join(typed),
        )


def render_regime_lane(
    st_module=st, *, roots: Mapping[str, Any], default_ids: Mapping[str, str] | None = None
) -> None:
    """The complete R6/R6.1 regime surface (FUX §35 R6 rows)."""

    dev_only_badge(st_module)
    st_module.caption(
        "V1 KMeans regime lane — fold-local, point-in-time, development "
        "research only. Cluster ids are nominal; regime outputs carry no "
        "feature eligibility, execution role, or promotion without the "
        "owner's ratified decision evidence."
    )
    _render_algorithm_registry(st_module)
    _render_proposal_stamps(st_module)
    _render_model_card(st_module, Path(roots["store_root"]), dict(default_ids or {}))
