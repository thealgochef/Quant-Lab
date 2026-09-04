"""MBP-1 order-flow dashboard panels (R5B; FUX §35 R5B rows; UI-3 summary-first).

A readiness SUMMARY first (the activated offline feature block, the
order-flow bundles, the source / coverage readiness, the owner-ratification
state, the controlled-study state and the fixed ``research_only_offline``
boundary — every fact resolved from the SELECTED run's persisted evidence
whenever possible), then the Research details (coverage, missingness, the
exact stage-window drill-down, the Baseline vs Baseline+MBP-1 comparison),
then Advanced diagnostics (the manual exact-id inputs, the registries, the
stamped defaults and the registry hashes). Thin widget layer: registry
state comes from the feature registries, every artifact read is an
exact-ID verified store load (stores are never listed), and every error
surface is sanitized. No control here can launch work, promote a feature,
or reach a live/serving surface.
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
    status_chip_line,
)

from alpha_lab.agents.data_infra.ifvg.presentation.help_registry import help_text
from alpha_lab.agents.data_infra.ifvg.presentation.labels import (
    AvailabilityKind,
    availability_chip,
    availability_for_block_status,
    label_for,
)
from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import (
    evaluate_interval,
    evaluate_metric,
)
from alpha_lab.agents.data_infra.ifvg.presentation.status_vocabulary import UiStatus

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

__all__ = ["render_mbp1_order_flow", "research_only_offline_badge"]

_MBP1 = f"{PIPELINE_STATE_PREFIX}mbp1_"
_BLOCK_KEY = "IFVG_ORDER_FLOW_MBP1_V1"

RESEARCH_ONLY_OFFLINE_TEXT = (
    "research_only_offline — the activated MBP-1 block cannot become a live "
    "model feature, an execution gate, or a Trade-Lab serving feature without "
    "a later Strategy-Core formula/parity contract and a separately approved "
    "sequential model-gated replay (owner decision R-6)."
)


def research_only_offline_badge(st_module) -> None:
    """The persistent R5B boundary label — rendered on every MBP-1 surface."""

    st_module.warning(f"**{RESEARCH_ONLY_OFFLINE_TEXT}**", icon="📊")


def _availability_rows() -> list[dict[str, str]]:
    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        FEATURE_BLOCK_REGISTRY,
        FEATURE_BLOCK_RESOLUTION_REGISTRY,
    )

    rows: list[dict[str, str]] = []
    for key in (_BLOCK_KEY, "IFVG_EXECUTION_LIQUIDITY_V1"):
        definition = FEATURE_BLOCK_REGISTRY[key]
        resolution = FEATURE_BLOCK_RESOLUTION_REGISTRY.get(key)
        rows.append(
            {
                "Block": key,
                "Name": label_for("block", key),
                "Version": str(definition.block_version),
                "Status": definition.status.value,
                "Availability": availability_chip(
                    availability_for_block_status(definition.status)
                ),
                "Resolved id": (
                    resolution.resolved_feature_block_id
                    if resolution is not None
                    else "— (no resolution envelope)"
                ),
                "Boundary": definition.expected_computation_path,
            }
        )
    return rows


def _bundle_rows() -> list[dict[str, str]]:
    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        BlockUnavailableError,
    )
    from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (  # noqa: PLC0415
        FEATURE_BUNDLE_REGISTRY,
        resolve_bundle,
    )

    rows: list[dict[str, str]] = []
    for key, definition in FEATURE_BUNDLE_REGISTRY.items():
        if "ORDER_FLOW" not in key:
            continue
        try:
            envelope = resolve_bundle(key)
        except (BlockUnavailableError, ValueError) as error:
            rows.append(
                {
                    "Bundle": key,
                    "Name": label_for("bundle", key),
                    "Base": definition.base_bundle_key or "—",
                    "State": "blocked",
                    "Resolved id / reason": sanitize_error(error),
                }
            )
        else:
            rows.append(
                {
                    "Bundle": key,
                    "Name": label_for("bundle", key),
                    "Base": definition.base_bundle_key or "—",
                    "State": "available (research-only offline)",
                    "Resolved id / reason": envelope.resolved_feature_bundle_id,
                }
            )
    return rows


def _render_availability(st_module) -> None:
    st_module.markdown("**MBP-1 availability** (activated at R5B — a versioned event)")
    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY,
        PRE_ACTIVATION_RESOLUTION_REGISTRY,
        feature_block_registry_hash,
    )

    st_module.dataframe(_availability_rows(), width="stretch", hide_index=True)
    st_module.dataframe(_bundle_rows(), width="stretch", hide_index=True)
    identity_block(
        st_module,
        "Block-registry hash (activated)",
        feature_block_registry_hash(),
    )
    identity_block(
        st_module,
        "Block-registry hash (pre-activation, R5 planned state)",
        feature_block_registry_hash(
            PRE_ACTIVATION_FEATURE_BLOCK_REGISTRY, PRE_ACTIVATION_RESOLUTION_REGISTRY
        ),
    )


def _render_coverage_policy_stamps(st_module) -> None:
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (  # noqa: PLC0415
        MBP1_PROPOSED_DEFAULTS,
    )

    st_module.markdown(
        "**Coverage-policy defaults (R5B.1)** — every value is a "
        "`proposed_protocol_default` of the owner's R-6 family: unratified for "
        "research; nothing here carries research weight"
    )
    st_module.dataframe(
        [
            {
                "Default": name,
                "Proposed value": str(entry["value"]),
                "Stamp": entry["stamp"],
                "Status": availability_chip(AvailabilityKind.PROPOSED),
                "Owner decision": entry["owner_decision"],
                "Ratification required": "yes — before research use",
            }
            for name, entry in MBP1_PROPOSED_DEFAULTS.items()
        ],
        width="stretch",
        hide_index=True,
    )


def _render_window_registry(st_module) -> None:
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (  # noqa: PLC0415
        R5B_WINDOW_SPECS,
    )

    st_module.markdown(
        "**Registered stage windows** — exact definitions; changing any row "
        "mints a new resolved block identity"
    )
    st_module.dataframe(
        [
            {
                "Window": spec.feature_window_key,
                "From → To": f"{spec.from_stage or '(day start)'} → {spec.to_stage}",
                "Trigger semantics": spec.trigger_semantics.value,
                "Bounds": f"{spec.lower_bound.value} / {spec.upper_bound.value}",
                "Min events": str(spec.minimum_event_count),
                "Cutoff contract": spec.cutoff_policy_id,
                "Features": str(len(spec.feature_names)),
            }
            for spec in R5B_WINDOW_SPECS
        ],
        width="stretch",
        hide_index=True,
    )


def _advanced_inputs(st_module, defaults: Mapping[str, str]) -> dict[str, str]:
    """The manual exact-id inputs (Advanced diagnostics only): auto-resolved
    from the selected run whenever possible; editing addresses another artifact."""

    st_module.caption(
        "Exact ids auto-resolved from the selected run's persisted stage evidence "
        "(manifest-verified) when present; edit any input to address another "
        "artifact — stores are never listed and there is no fuzzy lookup."
    )
    coverage_id = st_module.text_input(
        "Coverage report id (64-hex)",
        value=defaults.get("coverage_report_id", ""),
        key=f"{_MBP1}coverage_id",
        help=help_text("mbp1.coverage_report_id"),
    ).strip()
    artifact_id = st_module.text_input(
        "MBP-1 feature artifact id (64-hex)",
        value=defaults.get("feature_artifact_id", ""),
        key=f"{_MBP1}feature_artifact_id",
        help=help_text("mbp1.feature_artifact_id"),
    ).strip()
    candidate_id = st_module.text_input(
        "Exact candidate id",
        value="",
        key=f"{_MBP1}candidate_id",
        help=help_text("mbp1.candidate_id"),
    ).strip()
    study_id = st_module.text_input(
        "Controlled study id (64-hex)",
        value=defaults.get("controlled_study_id", ""),
        key=f"{_MBP1}study_id",
        help=help_text("mbp1.controlled_study_id"),
    ).strip()
    return {
        "coverage_report_id": coverage_id,
        "feature_artifact_id": artifact_id,
        "candidate_id": candidate_id,
        "controlled_study_id": study_id,
    }


def _render_coverage(st_module, store_root: Path, report_id: str) -> Any:
    """Coverage / missingness evidence by exact id; returns the loaded payload."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage import (  # noqa: PLC0415
        load_mbp1_coverage_report,
    )

    st_module.markdown("**Coverage and missingness evidence** (exact-ID load)")
    if not report_id:
        st_module.caption(
            "Paste the exact `mbp1_coverage_report_id` (S05 lists it among its "
            "output artifact ids) under Advanced diagnostics."
        )
        return None
    try:
        envelope = load_mbp1_coverage_report(store_root, report_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return None
    payload = envelope.payload
    st_module.caption(
        f"Boundary: **{payload.research_boundary}** · candidates: "
        f"{payload.candidate_count}"
    )
    identity_block(st_module, "Source artifact", payload.mbp1_source_artifact_id)
    identity_block(st_module, "Feature artifact", payload.mbp1_feature_artifact_id)
    st_module.markdown(
        "Per-day source coverage — **evidence-based (policy v2)**: coverage "
        "comes from verified partition-scope evidence only; raw sequence "
        "jumps are diagnostics, never gaps"
    )
    st_module.dataframe(
        [
            {
                "Trading day": row.trading_day,
                "Partitions": str(row.partition_count),
                "Events": str(row.row_count),
                "Completeness": row.completeness_status.value,
                "Coverage": (
                    "—"
                    if row.completeness_status.value == "completeness_unknown"
                    else f"{row.coverage_fraction:.4f}"
                ),
                "Declared gaps": str(row.declared_gap_count),
                "Open to partition end": "yes" if row.open_uncertainty_to_partition_end else "no",
                "Dataset condition": row.dataset_condition_status.value,
                "Sequence jumps (diagnostic)": str(row.sequence_positive_jump_count),
                "First event": row.first_ts_utc or "—",
                "Last event": row.last_ts_utc or "—",
            }
            for row in payload.day_rows
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.caption(
        f"Coverage policy: `{payload.coverage_policy_id}` — a day without "
        "partition-scope completeness evidence is `completeness_unknown` and "
        "every window of that day is typed `coverage_evidence_unavailable`."
    )
    st_module.markdown("Per-window validity and typed missing reasons")
    st_module.dataframe(
        [
            {
                "Window": row.feature_window_key,
                "Valid": str(row.valid_count),
                "Invalid": str(row.invalid_count),
                "Reasons": (
                    "; ".join(
                        f"{reason}: {count}"
                        for reason, count in sorted(row.reason_counts.items())
                    )
                    or "—"
                ),
            }
            for row in payload.window_rows
        ],
        width="stretch",
        hide_index=True,
    )
    fractions = dict(payload.per_feature_nonnull_fraction)
    if fractions:
        lowest = sorted(fractions.items(), key=lambda item: item[1])[:10]
        st_module.markdown("Lowest per-feature non-null fractions")
        st_module.dataframe(
            [
                {"Feature": name, "Non-null fraction": f"{fraction:.4f}"}
                for name, fraction in lowest
            ],
            width="stretch",
            hide_index=True,
        )
    return payload


def _render_drilldown(st_module, store_root: Path, artifact_id: str, candidate_id: str) -> None:
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_feature_materializer import (  # noqa: PLC0415
        load_mbp1_feature_artifact,
        load_stage_evidence_frame,
    )

    st_module.markdown(
        "**Exact stage-window drill-down** — one candidate's cutoffs, "
        "admitted counts, and typed reasons"
    )
    if not artifact_id or not candidate_id:
        st_module.caption(
            "Both the exact feature artifact id and the exact candidate id "
            "are required (Advanced diagnostics) — there is no fuzzy or "
            "nearest-match lookup."
        )
        return
    try:
        envelope = load_mbp1_feature_artifact(store_root, artifact_id)
        evidence = load_stage_evidence_frame(store_root, envelope)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return
    scoped = evidence[evidence["candidate_id"].astype(str) == candidate_id]
    if scoped.empty:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail=(
                "the exact candidate id is not in this feature artifact "
                "(no fuzzy fallback exists)"
            ),
        )
        return
    st_module.dataframe(
        [
            {
                "Window": str(row["feature_window_key"]),
                "From → To": f"{row['from_stage'] or '(day start)'} → {row['to_stage']}",
                "Cutoff kind": str(row["cutoff_kind"]) or "—",
                "From ts": str(row["from_ts_utc"]) or "—",
                "To ts": str(row["to_ts_utc"]) or "—",
                "Admitted events": str(row["admitted_event_count"]),
                "Same-ts ambiguous": "yes" if bool(row["same_timestamp_ambiguous"]) else "no",
                "Valid": "✓ valid" if bool(row["valid"]) else "✕ typed null",
                "Missing reason": str(row["missing_reason"]) or "—",
            }
            for _, row in scoped.iterrows()
        ],
        width="stretch",
        hide_index=True,
    )


def _delta_line(delta: Mapping[str, Any] | None) -> str:
    if not delta:
        return (
            "paired Brier delta not evaluable (0 OOS rows — the legitimate "
            "safe-failure shape on verification windows)"
        )
    if not delta.get("available"):
        return (
            "paired Brier delta interval unavailable: "
            f"{delta.get('reason', 'unknown reason')}"
        )

    def _fmt(value: Any) -> str:
        return "—" if value is None else f"{float(value):+.6f}"

    return (
        f"paired Brier delta (challenger − baseline): estimate "
        f"{_fmt(delta.get('estimate'))}, 95% trading-day block-bootstrap CI "
        f"[{_fmt(delta.get('lower'))}, {_fmt(delta.get('upper'))}] "
        "(negative favors the MBP-1 arm; a descriptive research diagnostic, "
        "never a promotion claim)"
    )


def _render_comparison(st_module, store_root: Path, study_id: str) -> Any:
    """The controlled comparison by exact id; returns the loaded payload."""

    from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (  # noqa: PLC0415
        load_controlled_feature_study,
    )

    st_module.markdown("**Baseline vs Baseline+MBP-1 controlled comparison**")
    if not study_id:
        st_module.caption(
            "Paste the exact `controlled_feature_study_id` (S09 lists it "
            "among its output artifact ids when an MBP-1 bundle is trained) "
            "under Advanced diagnostics."
        )
        return None
    try:
        envelope = load_controlled_feature_study(store_root, study_id)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        render_empty_state(st_module, "artifact_unavailable", detail=sanitize_error(error))
        return None
    payload = envelope.payload
    st_module.caption(
        f"Identical rows/labels/folds/model protocol on both arms — "
        f"protocol **{payload.model_protocol_id}**, parity "
        f"**{payload.parity_status}** over {payload.oos_row_count} OOS rows; "
        f"boundary **{payload.research_boundary}**."
    )
    st_module.dataframe(
        [
            {
                "Arm": "Baseline",
                "Bundle": payload.baseline_bundle_key,
                "Resolved bundle id": payload.baseline_resolved_bundle_id,
                "Ladder id": payload.baseline_ladder_id,
                **{
                    key: str(value)
                    for key, value in dict(payload.baseline_summary).items()
                },
            },
            {
                "Arm": "Baseline+MBP-1",
                "Bundle": payload.challenger_bundle_key,
                "Resolved bundle id": payload.challenger_resolved_bundle_id,
                "Ladder id": payload.challenger_ladder_id,
                **{
                    key: str(value)
                    for key, value in dict(payload.challenger_summary).items()
                },
            },
        ],
        width="stretch",
        hide_index=True,
    )
    st_module.write(
        _delta_line(
            dict(payload.paired_brier_delta)
            if payload.paired_brier_delta is not None
            else None
        )
    )
    identity_block(st_module, "MBP-1 feature artifact", payload.mbp1_feature_artifact_id)
    identity_block(st_module, "Label content hash", payload.label_content_hash)
    identity_block(st_module, "Fold set hash", payload.fold_set_hash)
    return payload


def _render_summary(st_module, facts: Mapping[str, Any], defaults: Mapping[str, str]) -> None:
    """UI-3 (plan §7): the readiness summary — every fact from the registries
    or the selected run's persisted evidence; nothing unresolved reads as passing."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        FEATURE_BLOCK_REGISTRY,
        FEATURE_BLOCK_RESOLUTION_REGISTRY,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (  # noqa: PLC0415
        MBP1_PROPOSED_DEFAULTS,
    )

    st_module.markdown(
        "**MBP-1 readiness (summary)** — resolved from the feature registries and the "
        "selected run's persisted evidence; exact ids, registries and stamps are under "
        "Advanced diagnostics."
    )
    block = FEATURE_BLOCK_REGISTRY[_BLOCK_KEY]
    resolution = FEATURE_BLOCK_RESOLUTION_REGISTRY.get(_BLOCK_KEY)
    resolved = (
        f" · resolved id `{resolution.resolved_feature_block_id[:12]}…`"
        if resolution is not None
        else ""
    )
    st_module.markdown(
        f"- Offline feature block: **{label_for('block', _BLOCK_KEY)}** (`{_BLOCK_KEY}` "
        f"v{block.block_version}) — "
        f"{availability_chip(availability_for_block_status(block.status))}{resolved}"
    )
    bundles = _bundle_rows()
    available = [row["Name"] for row in bundles if row["State"].startswith("available")]
    blocked = [row["Name"] for row in bundles if row["State"] == "blocked"]
    st_module.markdown(
        f"- Order-flow bundles: {len(available)} resolvable ({', '.join(available) or 'none'})"
        f" · {len(blocked)} blocked ({', '.join(blocked) or 'none'})"
    )
    coverage = facts.get("coverage")
    if coverage is None:
        status_chip_line(
            st_module,
            UiStatus.UNAVAILABLE,
            "Source / coverage readiness — no coverage evidence resolved from the "
            "selected run (paste an exact id under Advanced diagnostics)",
        )
    else:
        rows = list(coverage.day_rows)
        complete = [row for row in rows if row.completeness_status.value == "evidenced_complete"]
        unknown = [
            row for row in rows if row.completeness_status.value == "completeness_unknown"
        ]
        evidenced = [row.coverage_fraction for row in rows if row not in unknown]
        reading = evaluate_metric(
            "mbp1_day_coverage_fraction",
            min(evidenced) if evidenced else None,
            proposed=True,
        )
        status_chip_line(
            st_module,
            reading.status,
            f"Source / coverage readiness — {len(complete)} of {len(rows)} day(s) "
            f"evidenced_complete, {len(unknown)} completeness_unknown; lowest evidenced "
            f"day coverage {reading.display} — {reading.interpretation}",
        )
    st_module.markdown(
        f"- Owner ratification: {availability_chip(AvailabilityKind.PROPOSED)} — "
        f"{len(MBP1_PROPOSED_DEFAULTS)} coverage-policy defaults are "
        "`proposed_protocol_default` (owner decision R-6); none carries research weight"
    )
    study = facts.get("study")
    if study is None:
        status_chip_line(
            st_module,
            UiStatus.UNAVAILABLE,
            "Controlled study — no Baseline vs Baseline+MBP-1 study resolved from the "
            "selected run",
        )
    else:
        delta = dict(study.paired_brier_delta) if study.paired_brier_delta is not None else {}
        reading = evaluate_interval(
            "paired_brier_delta",
            lower=delta.get("lower"),
            upper=delta.get("upper"),
            available=bool(delta.get("available")),
            reason=str(delta.get("reason") or "not evaluable (0 OOS rows)"),
            estimate=delta.get("estimate"),
        )
        status_chip_line(
            st_module,
            reading.status,
            f"Controlled study — bundle `{study.challenger_bundle_key}` vs "
            f"`{study.baseline_bundle_key}`; parity {study.parity_status} over "
            f"{study.oos_row_count} OOS rows; {reading.interpretation} (a descriptive "
            "research diagnostic, never a promotion claim)",
        )
    st_module.markdown(
        f"- Boundary: {availability_chip(AvailabilityKind.RESEARCH_ONLY_OFFLINE)} — "
        "`research_only_offline` (owner decision R-6); no serving or execution role"
    )
    if not defaults:
        st_module.caption(
            "No pipeline run is selected, so nothing was auto-resolved; the summary "
            "shows the registries only."
        )


def render_mbp1_order_flow(
    st_module=st,
    *,
    roots: Mapping[str, Any],
    default_ids: Mapping[str, str] | None = None,
) -> None:
    """The complete R5B MBP-1 surface (FUX §35): summary first, then the
    Research details (coverage, drill-down, comparison), then Advanced
    diagnostics (manual ids, registries, stamps); persistent
    research_only_offline labeling throughout."""

    dev_only_badge(st_module)
    research_only_offline_badge(st_module)
    store_root = Path(roots["store_root"])
    defaults = dict(default_ids or {})
    summary_slot = st_module.container()
    details = st_module.expander(
        "Research details — coverage, stage-window drill-down, controlled comparison",
        expanded=True,
    )
    with st_module.expander("Advanced diagnostics — MBP-1 exact ids, registries and stamps"):
        ids = _advanced_inputs(st_module, defaults)
        _render_availability(st_module)
        _render_coverage_policy_stamps(st_module)
        _render_window_registry(st_module)
    facts: dict[str, Any] = {}
    with details:
        facts["coverage"] = _render_coverage(st_module, store_root, ids["coverage_report_id"])
        _render_drilldown(
            st_module, store_root, ids["feature_artifact_id"], ids["candidate_id"]
        )
        facts["study"] = _render_comparison(st_module, store_root, ids["controlled_study_id"])
    with summary_slot:
        _render_summary(st_module, facts, defaults)
