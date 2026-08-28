"""V1 KMeans regime dashboard panels (R6; FUX §35 R6 rows).

Coverage, occupancy, stability, ASSIGNMENT, and STRATIFICATION views over
EXACT-ID persisted regime artifacts; proposal/default stamps; the
sample-adequacy blocked state; the context-panel grain identity; the
promotion decision (role/status) view; and the post-V1 algorithms as
visible-disabled planned entries (with the mandatory spectral warning
text). Thin widget layer — registry state from the regime registries,
every artifact read a manifest-verified exact-ID store load (JSON
envelopes and the Arrow assignment frame only — the joblib sidecar is
never unpickled by the UI), every error sanitized. Cluster ids are
NOMINAL: nothing here ranks regimes, and no control can promote, launch,
rank, or retrain anything.

Stratified RESULT views (strategy/model/prop metrics by regime) belong to
the study comparison classes of ML plan §5.5 and land with their studies;
this panel stratifies the ASSIGNMENT frame itself (rows, coverage, and
margins by nominal regime × partition).
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

    st_module.markdown(
        "**Scientific defaults** — every value below is a "
        "`proposed_protocol_default`: owner ratification is required before "
        "any regime output becomes feature-eligible (nothing is promoted "
        "because fitting succeeded)"
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
        ],
        width="stretch",
        hide_index=True,
    )


def _render_model_card(st_module, store_root: Path) -> None:
    from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (  # noqa: PLC0415
        load_regime_assessment,
        load_regime_protocol,
    )

    st_module.markdown("**Regime model card** (exact-ID loads — stores are never listed)")
    protocol_id = st_module.text_input(
        "Resolved regime protocol id (64-hex)", value="", key=f"{_REG}protocol_id"
    ).strip()
    assessment_id = st_module.text_input(
        "Capability assessment id (64-hex)", value="", key=f"{_REG}assessment_id"
    ).strip()
    fit_id = st_module.text_input(
        "Regime fit id (64-hex) — assignment / stratification view",
        value="",
        key=f"{_REG}fit_id",
    ).strip()
    decision_id = st_module.text_input(
        "Promotion decision id (64-hex) — role / status view",
        value="",
        key=f"{_REG}decision_id",
    ).strip()
    if not protocol_id:
        st_module.caption(
            "Paste the exact `resolved_regime_protocol_id` (and optionally "
            "the `regime_capability_assessment_id`, a `regime_fit_id`, and a "
            "`regime_promotion_decision_id`) from a persisted regime run — "
            "stores are never listed."
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
    stability = result.stability
    st_module.markdown("**Stability**")
    st_module.dataframe(
        [
            {
                "Field": "Bootstrap aligned AMI (mean)",
                "Value": _fmt(stability.bootstrap_aligned_ami_mean),
            },
            {
                "Field": "Bootstrap aligned AMI (5th percentile)",
                "Value": _fmt(stability.bootstrap_aligned_ami_low),
            },
            {"Field": "Bootstrap refits", "Value": str(stability.bootstrap_refit_count)},
            {
                "Field": "Temporal persistence (OOS timeline)",
                "Value": _fmt(stability.temporal_persistence),
            },
            {
                "Field": "Temporal transitions counted",
                "Value": str(stability.temporal_transition_count),
            },
            {"Field": "Temporal order policy", "Value": stability.temporal_order_policy},
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
    if stability.transition_matrix:
        st_module.markdown(
            "**Regime transition matrix** (OOS timeline, within-fold; row → column frequency)"
        )
        st_module.dataframe(
            [
                {
                    "From \\ To": f"regime {row_index}",
                    **{
                        f"regime {column_index}": f"{value:.3f}"
                        for column_index, value in enumerate(row)
                    },
                }
                for row_index, row in enumerate(stability.transition_matrix)
            ],
            width="stretch",
            hide_index=True,
        )
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
        envelope, artifact, assignments = load_regime_fit_assignments(store_root, fit_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
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
        "metric-by-regime stratification of strategy/model/prop results lands with "
        "the study comparison classes, not here)"
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
                    "present (OwnerDecisionEvidenceRef content hash)"
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


def render_regime_lane(
    st_module=st, *, roots: Mapping[str, Any]
) -> None:
    """The complete R6 regime surface (FUX §35 R6)."""

    dev_only_badge(st_module)
    st_module.caption(
        "V1 KMeans regime lane — fold-local, point-in-time, development "
        "research only. Cluster ids are nominal; regime outputs carry no "
        "feature eligibility, execution role, or promotion without the "
        "owner's ratified decision evidence."
    )
    _render_algorithm_registry(st_module)
    _render_proposal_stamps(st_module)
    _render_model_card(st_module, Path(roots["store_root"]))
