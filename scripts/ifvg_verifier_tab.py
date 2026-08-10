"""Streamlit section for the IFVG visual trade verifier.

Rendered by the "Replay / Verifier" sub-tab above the exact-ID inspectors,
sharing one candidate selection (session-state key
``ifvg_context_v1_candidate``).  All evidence, gating, and authorization live
in the pure provider; this module is widgets, caching, and layout only —
no control here ever mentions sealed data or offers destructive actions.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ifvg_verifier_charts import (  # noqa: E402
    VerifierLayers,
    build_setup_figure,
    build_verifier_figure,
    collapse_to_execution_pane,
)

from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (  # noqa: E402
    MissingEvidenceError,
    RangeTooLargeError,
    ReplayAuthorizationError,
    ReplayContext,
    bars_for_pane,
    candidate_evidence,
    chart_range,
    list_candidates,
    open_replay_context,
)
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (  # noqa: E402
    REPLAY_CHART_CATALOG,
    REPLAY_CHART_STORE,
    ArtifactPairRef,
    ReplayChartStoreError,
    VerifierBundleRef,
    find_replay_artifact,
    read_replay_chart_catalog,
)
from alpha_lab.agents.data_infra.ifvg.setup_verifier_provider import (  # noqa: E402
    SETUP_STAGE_ORDER,
    SetupEvidence,
    SetupReplayContext,
    list_setups,
    open_setup_replay_context,
    resolve_setup_selection,
    setup_evidence,
)
from alpha_lab.agents.data_infra.ifvg.visual_review_store import (  # noqa: E402
    REVIEW_TAGS,
    REVIEW_VERDICTS,
    append_review,
    export_csv,
    list_reviews,
)

_STATE_PREFIX = "ifvg_context_v1_"
_CANDIDATE_KEY = f"{_STATE_PREFIX}candidate"
_SETUP_KEY = f"{_STATE_PREFIX}setup"
_WINDOW_PATH = re.compile(r"(?:[A-Za-z]:\\|/)[^\s'\"]+")
_SECRET = re.compile(r"(?i)(token|secret|api[_-]?key)\s*[:=]\s*[^\s,}]+")

_STAGE_ORDER = ("tap", "parent", "lock", "opposing", "inversion", "entry", "resolution")


def _sanitize_error(error: BaseException) -> str:
    value = _WINDOW_PATH.sub("[path redacted]", str(error))
    return _SECRET.sub(r"\1=[redacted]", value)[:800]


def _sanitize_select(st_module, key: str, options: tuple) -> None:
    """Drop a stale selection when the option list no longer contains it."""
    if key in st_module.session_state and st_module.session_state[key] not in options:
        del st_module.session_state[key]


def jump_to_candidate(candidate_id: str) -> None:
    """Load a candidate into the shared verifier selection (report-row jumps)."""
    st.session_state[_CANDIDATE_KEY] = str(candidate_id)


_PENDING_JUMP_KEY = f"{_STATE_PREFIX}pending_jump"


def queue_jump(kind: str, value: str) -> None:
    """Queue an exact-ID jump (candidate_id / decision_id / trade_id).

    Resolution happens inside the verifier section through the provider's
    exact-ID ``resolve_selection`` — never by fallback matching.
    """
    if kind not in ("candidate_id", "decision_id", "trade_id"):
        raise ValueError(f"unsupported jump kind: {kind}")
    st.session_state[_PENDING_JUMP_KEY] = (kind, str(value))


def _apply_pending_jump(st_module, ctx: ReplayContext) -> None:
    pending = st.session_state.pop(_PENDING_JUMP_KEY, None)
    if not pending:
        return
    kind, value = pending
    try:
        from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (
            resolve_selection,
        )

        candidate_id = resolve_selection(ctx, **{kind: value})
    except Exception as error:
        st_module.warning(f"Jump not resolved: {_sanitize_error(error)}")
        return
    st.session_state[_CANDIDATE_KEY] = candidate_id


@st.cache_resource(show_spinner="Opening verified replay context (first load is slow)…")
def _cached_replay_context(
    profile_name: str,
    v2_id: str,
    v2_manifest_hash: str,
    v3_id: str,
    v3_manifest_hash: str,
    replay_artifact_id: str,
    replay_manifest_hash: str,
    _verified_pair,
) -> ReplayContext:
    pair_ref = ArtifactPairRef(
        profile_name=profile_name,
        v2_dataset_id=v2_id,
        v2_manifest_hash=v2_manifest_hash,
        v3_dataset_id=v3_id,
        v3_manifest_hash=v3_manifest_hash,
    )
    ctx = open_replay_context(
        ROOT,
        pair_ref,
        replay_artifact_id=replay_artifact_id,
        verified_pair=_verified_pair,
    )
    stored_hash = ctx.replay.manifest.get("manifest_payload_sha256")
    if stored_hash != replay_manifest_hash:
        raise ReplayChartStoreError("replay-chart manifest hash changed since cataloging")
    return ctx


@st.cache_data(show_spinner=False)
def _cached_candidate_frame(replay_artifact_id: str, _ctx) -> pd.DataFrame:
    return list_candidates(_ctx)


@st.cache_data(show_spinner=False)
def _cached_evidence(
    replay_artifact_id: str,
    candidate_id: str,
    mode: str,
    stage: str | None,
    _ctx,
):
    return candidate_evidence(_ctx, candidate_id, mode=mode, stage=stage)


def _step_candidate(options: list[str], delta: int) -> None:
    current = st.session_state.get(_CANDIDATE_KEY)
    if not options:
        return
    index = (options.index(current) + delta) % len(options) if current in options else 0
    st.session_state[_CANDIDATE_KEY] = options[index]


def _filtered_candidates(st_module, frame: pd.DataFrame) -> pd.DataFrame:
    left, middle, right = st_module.columns([1.6, 1.6, 1.4])
    with left:
        outcome = st_module.multiselect(
            "Outcome / state",
            ("executed", "blocked", "censored", "win", "loss"),
            default=(),
            key=f"{_STATE_PREFIX}verifier_outcome",
        )
        m3_only = st_module.checkbox(
            "M3 qualifying cases only",
            key=f"{_STATE_PREFIX}verifier_m3",
        )
    with middle:
        sessions = st_module.multiselect(
            "Session",
            tuple(sorted(frame["entry_session"].dropna().unique())),
            default=(),
            key=f"{_STATE_PREFIX}verifier_session",
        )
        include_warmup = st_module.checkbox(
            "Include warmup evidence",
            value=False,
            key=f"{_STATE_PREFIX}verifier_warmup",
        )
    with right:
        days = sorted(frame["trading_day"].unique())
        day_from, day_to = st_module.select_slider(
            "Trading-day range",
            options=days,
            value=(days[0], days[-1]),
            key=f"{_STATE_PREFIX}verifier_days",
        )

    filtered = frame[
        (frame["trading_day"] >= day_from) & (frame["trading_day"] <= day_to)
    ]
    if not include_warmup:
        filtered = filtered[~filtered["is_warmup"]]
    if sessions:
        filtered = filtered[filtered["entry_session"].isin(sessions)]
    if m3_only:
        filtered = filtered[filtered["m3_qualifying"]]
    if outcome:
        mask = pd.Series(False, index=filtered.index)
        if "executed" in outcome:
            mask |= filtered["executed"]
        if "blocked" in outcome:
            mask |= filtered["blocked"] & ~filtered["executed"]
        if "censored" in outcome:
            mask |= filtered["censored"]
        if "win" in outcome:
            mask |= filtered["resolution"] == "target"
        if "loss" in outcome:
            mask |= filtered["resolution"] == "stop"
        filtered = filtered[mask]

    # A direct report-row jump must never be filtered away.
    current = st.session_state.get(_CANDIDATE_KEY)
    if current is not None and current not in set(filtered["candidate_id"]):
        jumped = frame[frame["candidate_id"] == current]
        if len(jumped):
            filtered = pd.concat([filtered, jumped]).drop_duplicates("candidate_id")
            filtered = filtered.sort_values(
                ["trading_day", "entry_ts_utc", "candidate_id"], kind="mergesort"
            )
    return filtered.reset_index(drop=True)


def _candidate_label(row: pd.Series) -> str:
    state = "executed" if row["executed"] else ("blocked" if row["blocked"] else "candidate")
    warmup = " · warmup" if row["is_warmup"] else ""
    outcome = f" · {row['resolution']}" if row["resolution"] else ""
    return (
        f"{row['trading_day']} {row['entry_session']} · {state}{outcome}{warmup} · "
        f"{row['candidate_id'][:12]}…"
    )


def _render_side_panel(st_module, ctx: ReplayContext, evidence, row: pd.Series) -> None:
    point_in_time = evidence.mode == "point_in_time"
    with st_module.expander("Identity", expanded=False):
        st_module.json(evidence.identity, expanded=True)
    with st_module.expander("Lineage (exact IDs)", expanded=False):
        for name, value in evidence.lineage.items():
            if value is None:
                continue
            st_module.caption(f"{name}")
            st_module.code(str(value), language=None)
    with st_module.expander("Lifecycle", expanded=False):
        gates = evidence.stage_gates
        rows = []
        for stage_name in _STAGE_ORDER:
            gate = gates.get(stage_name)
            if gate is None:
                rows.append({"stage": stage_name, "ts": "— ungateable", "source": "—"})
                continue
            reached = (
                not point_in_time
                or _STAGE_ORDER.index(stage_name) <= _STAGE_ORDER.index(evidence.stage)
            )
            rows.append(
                {
                    "stage": stage_name if reached else f"{stage_name} (withheld)",
                    "ts": str(gate.ts_utc) if reached else "—",
                    "source": gate.source_kind if reached else "—",
                }
            )
        st_module.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st_module.expander("Geometry", expanded=False):
        st_module.dataframe(
            pd.DataFrame(
                [
                    {
                        "role": zone.role,
                        "tf_s": zone.timeframe_seconds,
                        "direction": zone.direction,
                        "low": zone.gap_low_ticks,
                        "high": zone.gap_high_ticks,
                        "confirmed": str(zone.confirmed_ts_utc),
                    }
                    for zone in evidence.zones
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st_module.caption("Zone fill/invalidation times are not persisted — lifecycle end unknown.")
    with st_module.expander("Execution", expanded=False):
        if evidence.execution is None:
            if row["blocked"]:
                st_module.warning(f"Blocked candidate — {row['block_reasons']}")
            if evidence.counterfactual_labels:
                st_module.caption(
                    "Counterfactual candidate-label paths (not actual execution):"
                )
                st_module.dataframe(
                    pd.DataFrame(list(evidence.counterfactual_labels)),
                    hide_index=True,
                    use_container_width=True,
                )
            elif point_in_time:
                st_module.caption("Outcome evidence withheld before the resolution stage.")
        else:
            execution = evidence.execution
            first, second = st_module.columns(2)
            first.metric("Entry", f"{execution['entry_ticks'] * 0.25:.2f}")
            second.metric("Stop", f"{execution['stop_ticks'] * 0.25:.2f}")
            if "realized_r" in execution:
                third, fourth = st_module.columns(2)
                third.metric("Net R", f"{execution['realized_r']:+.2f}")
                fourth.metric(
                    "Duration", f"{execution['bars_after_entry_to_resolution']} bars"
                )
                st_module.caption(
                    f"MFE {execution['mfe_ticks']} ticks · MAE {execution['mae_ticks']} "
                    "ticks (magnitudes only — timing not persisted)"
                )
            else:
                st_module.caption("Outcome evidence withheld before the resolution stage.")
            if evidence.counterfactual_labels:
                st_module.caption("Counterfactual label paths (kept separate):")
                st_module.dataframe(
                    pd.DataFrame(list(evidence.counterfactual_labels))[
                        ["label_family", "label", "censored", "mfe_r", "mae_r"]
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
    with st_module.expander("Context", expanded=False):
        if len(evidence.structure_stage_summary):
            st_module.caption("MTF alignment per stage (exact Strategy-Core evidence):")
            st_module.dataframe(
                evidence.structure_stage_summary,
                hide_index=True,
                use_container_width=True,
            )
        if len(evidence.displacement):
            st_module.caption("Displacement windows:")
            st_module.dataframe(
                evidence.displacement[
                    [
                        "window_kind",
                        "stage",
                        "start_ts",
                        "end_ts",
                        "valid",
                        "missing_reason",
                        "metrics_path_efficiency_abs",
                        "metrics_setup_net_move_normalized",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
        if len(evidence.pools):
            st_module.caption("Nearest / linked pools:")
            st_module.dataframe(
                evidence.pools[
                    [
                        "pool_pool_id",
                        "pool_pool_type",
                        "pool_lower_bound_ticks",
                        "pool_upper_bound_ticks",
                        "pool_active",
                        "pool_swept",
                        "pool_reclaimed",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
        if len(evidence.sweep_links):
            st_module.caption("Sweep evidence:")
            st_module.dataframe(
                evidence.sweep_links[
                    [
                        "sweep_link_id",
                        "pool_id",
                        "sweep_ts",
                        "sweep_depth_ticks",
                        "qualifies_opposing_leg",
                        "selected",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )
    with st_module.expander(
        "Model probabilities — counterfactual development evidence", expanded=False
    ):
        st_module.caption("Offline OOS development predictions. Not live signals.")
        st_module.dataframe(
            pd.DataFrame(
                [{"tier": tier, **payload} for tier, payload in evidence.model.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if evidence.gating_report.get("ungateable") or evidence.gating_report.get("hidden"):
        with st_module.expander("Gating report", expanded=False):
            ungateable = evidence.gating_report.get("ungateable", ())
            if ungateable:
                st_module.caption("Cannot gate — hidden in point-in-time mode:")
                st_module.dataframe(
                    pd.DataFrame(list(ungateable)),
                    hide_index=True,
                    use_container_width=True,
                )
            hidden = evidence.gating_report.get("hidden", ())
            if hidden:
                st_module.caption(f"{len(hidden)} objects withheld by the stage gate.")
    _render_review_section(st_module, ctx, evidence, row)


def _render_review_section(st_module, ctx: ReplayContext, evidence, row: pd.Series) -> None:
    with st_module.expander("Review (ifvg_visual_review_v1)", expanded=False):
        existing = pd.DataFrame()
        try:
            existing = list_reviews(repo_root=ROOT, candidate_id=evidence.candidate_id)
        except Exception as error:
            st_module.caption(f"Review ledger unreadable: {_sanitize_error(error)}")
        if len(existing):
            st_module.caption(f"{len(existing)} prior review(s) for this candidate:")
            st_module.dataframe(
                existing[["reviewed_at", "reviewer", "overall_verdict", "tags", "notes"]],
                hide_index=True,
                use_container_width=True,
            )
        reviewer = st_module.text_input(
            "Reviewer", key=f"{_STATE_PREFIX}review_reviewer"
        )
        overall = st_module.selectbox(
            "Overall verdict", REVIEW_VERDICTS, key=f"{_STATE_PREFIX}review_overall"
        )
        detail_verdicts: dict[str, str] = {"overall_verdict": overall}
        with st_module.popover("Detail verdicts (optional)"):
            for name in (
                "fvg_geometry_verdict",
                "lifecycle_verdict",
                "entry_verdict",
                "stop_verdict",
                "outcome_verdict",
            ):
                value = st_module.selectbox(
                    name.replace("_", " "),
                    ("—", *REVIEW_VERDICTS),
                    key=f"{_STATE_PREFIX}review_{name}",
                )
                if value != "—":
                    detail_verdicts[name] = value
        tags = st_module.multiselect("Tags", REVIEW_TAGS, key=f"{_STATE_PREFIX}review_tags")
        notes = st_module.text_area("Notes", key=f"{_STATE_PREFIX}review_notes")
        if st_module.button("Save review note", key=f"{_STATE_PREFIX}review_save"):
            try:
                append_review(
                    repo_root=ROOT,
                    replay_chart_artifact_id=ctx.replay.artifact_id,
                    pair_ref=ctx.pair_ref.as_dict(),
                    candidate_id=evidence.candidate_id,
                    decision_id=(
                        str(row["decision_id"]) if pd.notna(row["decision_id"]) else None
                    ),
                    trade_id=str(row["trade_id"]) if pd.notna(row["trade_id"]) else None,
                    reviewer=reviewer,
                    verdicts=detail_verdicts,
                    tags=list(tags),
                    notes=notes,
                )
            except Exception as error:
                st_module.error(f"Review not saved: {_sanitize_error(error)}")
            else:
                st_module.success("Review appended to the ledger.")
        try:
            csv_payload = export_csv(repo_root=ROOT)
        except Exception:
            csv_payload = ""
        if csv_payload:
            st_module.download_button(
                "Download review ledger CSV",
                csv_payload,
                file_name="ifvg_visual_review_v1.csv",
                mime="text/csv",
                key=f"{_STATE_PREFIX}review_csv",
            )


# ── setup-level review mode (additive; candidate mode above is untouched) ────


def _discover_setup_bundle(pair_ref: ArtifactPairRef) -> VerifierBundleRef | None:
    """Newest setup-aware (v2-kind) replay-chart artifact for this exact pair.

    The publication gate keeps the catalog silent about v2 bundles, so
    discovery reads the immutable store manifests directly; every identity in
    the returned ref is re-verified by ``open_setup_replay_context``.
    """
    store = ROOT / REPLAY_CHART_STORE
    if not store.is_dir():
        return None
    wanted = pair_ref.as_dict()
    found: list[VerifierBundleRef] = []
    for directory in sorted(store.iterdir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("artifact_kind") != "ifvg_replay_chart_v2":
            continue
        if manifest.get("source_pair") != wanted:
            continue
        config = manifest.get("effective_config") or {}
        try:
            found.append(
                VerifierBundleRef(
                    profile_name=pair_ref.profile_name,
                    v2_dataset_id=pair_ref.v2_dataset_id,
                    v2_manifest_hash=pair_ref.v2_manifest_hash,
                    v3_dataset_id=pair_ref.v3_dataset_id,
                    v3_manifest_hash=pair_ref.v3_manifest_hash,
                    fsm_audit_artifact_id=str(config.get("fsm_audit_artifact_id")),
                    fsm_audit_manifest_hash=str(config.get("fsm_audit_manifest_hash")),
                    replay_chart_artifact_id=str(directory.name),
                    replay_chart_manifest_hash=str(
                        manifest.get("manifest_payload_sha256")
                    ),
                )
            )
        except Exception:
            continue
    if not found:
        return None
    return sorted(found, key=lambda bundle: bundle.replay_chart_artifact_id)[-1]


@st.cache_resource(show_spinner="Opening verified setup bundle (first load is slow)…")
def _cached_setup_context(
    profile_name: str,
    v2_id: str,
    v2_manifest_hash: str,
    v3_id: str,
    v3_manifest_hash: str,
    fsm_audit_artifact_id: str,
    fsm_audit_manifest_hash: str,
    replay_chart_artifact_id: str,
    replay_chart_manifest_hash: str,
) -> SetupReplayContext:
    bundle = VerifierBundleRef(
        profile_name=profile_name,
        v2_dataset_id=v2_id,
        v2_manifest_hash=v2_manifest_hash,
        v3_dataset_id=v3_id,
        v3_manifest_hash=v3_manifest_hash,
        fsm_audit_artifact_id=fsm_audit_artifact_id,
        fsm_audit_manifest_hash=fsm_audit_manifest_hash,
        replay_chart_artifact_id=replay_chart_artifact_id,
        replay_chart_manifest_hash=replay_chart_manifest_hash,
    )
    return open_setup_replay_context(ROOT, bundle)


@st.cache_data(show_spinner=False)
def _cached_setup_frame(replay_chart_artifact_id: str, _ctx) -> pd.DataFrame:
    return list_setups(_ctx)


@st.cache_data(show_spinner=False)
def _cached_setup_evidence(
    replay_chart_artifact_id: str,
    setup_id: str,
    stage: str | None,
    _ctx,
) -> SetupEvidence:
    return setup_evidence(_ctx, setup_id, stage=stage)


def _step_setup(options: list[str], delta: int) -> None:
    current = st.session_state.get(_SETUP_KEY)
    if not options:
        return
    index = (options.index(current) + delta) % len(options) if current in options else 0
    st.session_state[_SETUP_KEY] = options[index]


def _parse_ts_bound(st_module, raw: str, label: str) -> pd.Timestamp | None:
    if not raw:
        return None
    try:
        stamp = pd.Timestamp(raw)
        return (
            stamp.tz_convert("UTC") if stamp.tzinfo else pd.Timestamp(raw, tz="UTC")
        )
    except (ValueError, TypeError):
        st_module.warning(f"{label} bound is not a valid UTC timestamp — ignored.")
        return None


_TRI_CANDIDATE_LESS = ("all", "candidate-less only", "with candidates")
_TRI_Q40 = ("all", "Q-40 exposed only", "not Q-40 exposed")
_TRI_PARENTLESS = ("all", "parentless only", "never parentless")
_PRESENT_BUT_EMPTY = "0 setups in this artifact (supported by contract; none observed)."


def _filtered_setups(st_module, frame: pd.DataFrame) -> pd.DataFrame:
    left, middle, right = st_module.columns(3)
    with left:
        candidate_less = st_module.selectbox(
            "Candidate-less",
            _TRI_CANDIDATE_LESS,
            key=f"{_STATE_PREFIX}setup_candidate_less",
        )
        reasons = st_module.multiselect(
            "Terminal reason",
            tuple(sorted(frame["terminal_reason"].dropna().astype(str).unique())),
            default=(),
            key=f"{_STATE_PREFIX}setup_terminal_reason",
        )
        phases = st_module.multiselect(
            "Phase at death",
            tuple(sorted(frame["phase_at_death"].dropna().astype(str).unique())),
            default=(),
            key=f"{_STATE_PREFIX}setup_phase",
        )
        # These two flag filters are PRESENT-BUT-EMPTY on current data —
        # rendered always, with an honest zero-count caption, never hidden.
        conflict_count = int(frame["conflict_flag"].sum())
        conflict_only = st_module.checkbox(
            f"Conflict-flagged only ({conflict_count})",
            key=f"{_STATE_PREFIX}setup_conflict",
        )
        if conflict_count == 0:
            st_module.caption(f"Conflict flag: {_PRESENT_BUT_EMPTY}")
        suppression_count = int(frame["structural_suppression_flag"].sum())
        suppression_only = st_module.checkbox(
            f"Structural-suppression only ({suppression_count})",
            key=f"{_STATE_PREFIX}setup_suppression",
        )
        if suppression_count == 0:
            st_module.caption(f"Structural suppression: {_PRESENT_BUT_EMPTY}")
    with middle:
        htf_tfs = st_module.multiselect(
            "HTF timeframe (s)",
            tuple(
                int(value)
                for value in sorted(frame["htf_tf_seconds"].dropna().unique())
            ),
            default=(),
            key=f"{_STATE_PREFIX}setup_htf_tf",
        )
        parent_tfs = st_module.multiselect(
            "Parent timeframe (s)",
            tuple(
                int(value)
                for value in sorted(frame["parent_tf_seconds"].dropna().unique())
            ),
            default=(),
            key=f"{_STATE_PREFIX}setup_parent_tf",
        )
        sessions = st_module.multiselect(
            "Session (doc, at activation)",
            tuple(
                sorted(
                    frame["session_doc_at_activation"].dropna().astype(str).unique()
                )
            ),
            default=(),
            key=f"{_STATE_PREFIX}setup_session",
        )
        q40 = st_module.selectbox(
            "Q-40 exposure", _TRI_Q40, key=f"{_STATE_PREFIX}setup_q40"
        )
        parentless = st_module.selectbox(
            "Parentless (had ≥1 interval)",
            _TRI_PARENTLESS,
            key=f"{_STATE_PREFIX}setup_parentless",
        )
        include_warmup = False
        if "is_warmup" in frame.columns:
            include_warmup = st_module.checkbox(
                "Include warmup setups",
                value=False,
                key=f"{_STATE_PREFIX}setup_warmup",
            )
    with right:
        activation_start = _parse_ts_bound(
            st_module,
            st_module.text_input(
                "Activation from (UTC ISO)",
                key=f"{_STATE_PREFIX}setup_activation_start",
            ),
            "Activation start",
        )
        activation_end = _parse_ts_bound(
            st_module,
            st_module.text_input(
                "Activation to (UTC ISO)",
                key=f"{_STATE_PREFIX}setup_activation_end",
            ),
            "Activation end",
        )
        display_start = _parse_ts_bound(
            st_module,
            st_module.text_input(
                "Display-end from (UTC ISO)",
                key=f"{_STATE_PREFIX}setup_display_end_start",
            ),
            "Display-end start",
        )
        display_end = _parse_ts_bound(
            st_module,
            st_module.text_input(
                "Display-end to (UTC ISO)",
                key=f"{_STATE_PREFIX}setup_display_end_end",
            ),
            "Display-end end",
        )

    filtered = frame
    if candidate_less == "candidate-less only":
        filtered = filtered[filtered["candidate_less"]]
    elif candidate_less == "with candidates":
        filtered = filtered[~filtered["candidate_less"]]
    if reasons:
        filtered = filtered[filtered["terminal_reason"].astype(str).isin(reasons)]
    if phases:
        filtered = filtered[filtered["phase_at_death"].astype(str).isin(phases)]
    if htf_tfs:
        filtered = filtered[filtered["htf_tf_seconds"].isin(htf_tfs)]
    if parent_tfs:
        filtered = filtered[filtered["parent_tf_seconds"].isin(parent_tfs)]
    if sessions:
        filtered = filtered[
            filtered["session_doc_at_activation"].astype(str).isin(sessions)
        ]
    if q40 == "Q-40 exposed only":
        filtered = filtered[filtered["q40_exposed"]]
    elif q40 == "not Q-40 exposed":
        filtered = filtered[~filtered["q40_exposed"]]
    if parentless == "parentless only":
        filtered = filtered[filtered["parentless"]]
    elif parentless == "never parentless":
        filtered = filtered[~filtered["parentless"]]
    if conflict_only:
        filtered = filtered[filtered["conflict_flag"]]
    if suppression_only:
        filtered = filtered[filtered["structural_suppression_flag"]]
    if "is_warmup" in frame.columns and not include_warmup:
        filtered = filtered[~filtered["is_warmup"]]
    activation = pd.to_datetime(filtered["activation_ts_utc"], utc=True)
    if activation_start is not None:
        filtered = filtered[activation >= activation_start]
        activation = activation[activation >= activation_start]
    if activation_end is not None:
        filtered = filtered[activation <= activation_end]
        activation = activation[activation <= activation_end]
    display = pd.to_datetime(filtered["display_end_ts_utc"], utc=True)
    if display_start is not None:
        filtered = filtered[display >= display_start]
        display = display[display >= display_start]
    if display_end is not None:
        filtered = filtered[display <= display_end]
    return filtered.reset_index(drop=True)


def _setup_label(row: pd.Series) -> str:
    day = (
        str(pd.Timestamp(row["activation_ts_utc"]).date())
        if pd.notna(row["activation_ts_utc"])
        else "—"
    )
    kind = (
        "candidate-less"
        if row["candidate_less"]
        else f"{int(row['candidate_count'])} candidate(s)"
    )
    return (
        f"{day} · {row['terminal_reason']} · {row['phase_at_death']} · {kind} · "
        f"{str(row['setup_id'])[:12]}…"
    )


def _setup_candidate_ids(row: pd.Series) -> list[str]:
    try:
        parsed = json.loads(str(row["candidate_ids"]))
    except (ValueError, TypeError):
        return []
    return [str(value) for value in parsed] if isinstance(parsed, list) else []


def _render_setup_panel(st_module, ctx, bundle, evidence, row: pd.Series) -> None:
    candidate_less = bool(row["candidate_less"])
    with st_module.expander("Setup summary", expanded=False):
        first, second = st_module.columns(2)
        first.metric("Terminal reason", str(row["terminal_reason"]))
        second.metric("Phase at death", str(row["phase_at_death"]))
        third, fourth = st_module.columns(2)
        third.metric("Parentless intervals", int(row["parentless_interval_count"]))
        fourth.metric("Candidates", int(row["candidate_count"]))
        st_module.caption(
            f"activation {row['activation_ts_utc']} → terminal "
            f"{row['terminal_ts_utc']} ({row['terminal_reason']})"
        )
        if candidate_less:
            st_module.warning(
                "Candidate-less setup — it never produced an entry candidate "
                "and is NOT executable evidence."
            )
    with st_module.expander("Ordered event log", expanded=False):
        if evidence.events.empty:
            st_module.caption("No events are visible at this stage gate.")
        else:
            columns = [
                name
                for name in (
                    "event_kind",
                    "stage",
                    "ts_utc",
                    "fvg_id",
                    "selected",
                    "drop_reason",
                    "fill_kind",
                    "fill_depth_ticks",
                )
                if name in evidence.events.columns
            ]
            st_module.dataframe(
                evidence.events[columns],
                hide_index=True,
                use_container_width=True,
                height=280,
            )
    if len(evidence.slot_deaths):
        with st_module.expander("Slot-death detail", expanded=False):
            columns = [
                name
                for name in (
                    "death_reason",
                    "phase",
                    "death_ts_utc",
                    "fill_depth_ticks",
                    "prior_reached_ticks",
                    "new_reached_ticks",
                    "parent_clocks",
                    "remaining_window_bars_by_tf",
                    "open_window_timeframes",
                    "setup_terminated",
                )
                if name in evidence.slot_deaths.columns
            ]
            st_module.dataframe(
                evidence.slot_deaths[columns],
                hide_index=True,
                use_container_width=True,
            )
            st_module.caption(
                "Fill depth, prior/new reached extreme, and the window clocks "
                "are the persisted audit evidence — not recomputed."
            )
    # Model tier statuses keep their exact semantics: they exist only for
    # candidates at decision time. A candidate-less setup therefore shows
    # NOTHING model-related.
    if not candidate_less:
        with st_module.expander(
            "Model probabilities — candidate-scoped", expanded=False
        ):
            st_module.caption(
                "Model tier statuses are only defined for candidates at "
                "decision time — open a linked candidate in candidate mode "
                "to see them."
            )
            for candidate_id in _setup_candidate_ids(row):
                st_module.code(candidate_id, language=None)
    _render_setup_review_section(st_module, ctx, bundle, evidence)


_SETUP_DETAIL_VERDICTS = (
    "htf_verdict",
    "parent_verdict",
    "opposing_verdict",
    "inversion_verdict",
    "fill_verdict",
)


def _render_setup_review_section(st_module, ctx, bundle, evidence) -> None:
    with st_module.expander(
        "Review (ifvg_visual_review_v1 — setup level)", expanded=False
    ):
        existing = pd.DataFrame()
        try:
            existing = list_reviews(repo_root=ROOT, setup_id=evidence.setup_id)
        except Exception as error:
            st_module.caption(f"Review ledger unreadable: {_sanitize_error(error)}")
        if len(existing):
            st_module.caption(f"{len(existing)} prior review(s) for this setup:")
            st_module.dataframe(
                existing[["reviewed_at", "reviewer", "overall_verdict", "tags", "notes"]],
                hide_index=True,
                use_container_width=True,
            )
        reviewer = st_module.text_input(
            "Reviewer", key=f"{_STATE_PREFIX}setup_review_reviewer"
        )
        overall = st_module.selectbox(
            "Overall verdict", REVIEW_VERDICTS, key=f"{_STATE_PREFIX}setup_review_overall"
        )
        detail_verdicts: dict[str, str] = {"overall_verdict": overall}
        with st_module.popover("Detail verdicts (optional)"):
            for name in _SETUP_DETAIL_VERDICTS:
                value = st_module.selectbox(
                    name.replace("_", " "),
                    ("—", *REVIEW_VERDICTS),
                    key=f"{_STATE_PREFIX}setup_review_{name}",
                )
                if value != "—":
                    detail_verdicts[name] = value
        tags = st_module.multiselect(
            "Tags", REVIEW_TAGS, key=f"{_STATE_PREFIX}setup_review_tags"
        )
        notes = st_module.text_area("Notes", key=f"{_STATE_PREFIX}setup_review_notes")
        if st_module.button("Save review note", key=f"{_STATE_PREFIX}setup_review_save"):
            try:
                append_review(
                    repo_root=ROOT,
                    replay_chart_artifact_id=bundle.replay_chart_artifact_id,
                    pair_ref=ctx.base.pair_ref.as_dict(),
                    # Setup-scope review: candidate reviews stay in candidate
                    # mode, so this record carries no candidate id ("" for
                    # candidate-less and candidate-full setups alike).
                    candidate_id="",
                    decision_id=None,
                    trade_id=None,
                    reviewer=reviewer,
                    verdicts=detail_verdicts,
                    tags=list(tags),
                    notes=notes,
                    setup_id=evidence.setup_id,
                    fsm_audit_artifact_id=bundle.fsm_audit_artifact_id,
                )
            except Exception as error:
                st_module.error(f"Review not saved: {_sanitize_error(error)}")
            else:
                st_module.success("Review appended to the ledger.")


def _render_setup_section(st_module, pair_ref: ArtifactPairRef) -> None:
    """Setup-level review: every setup, including the candidate-less ones."""
    bundle = _discover_setup_bundle(pair_ref)
    if bundle is None:
        st_module.info(
            "No setup-aware replay-chart (v2) bundle exists for this exact pair "
            "yet. Build the fsm-audit and v2 replay-chart artifacts externally, "
            "then reload."
        )
        return
    try:
        ctx = _cached_setup_context(
            bundle.profile_name,
            bundle.v2_dataset_id,
            bundle.v2_manifest_hash,
            bundle.v3_dataset_id,
            bundle.v3_manifest_hash,
            bundle.fsm_audit_artifact_id,
            bundle.fsm_audit_manifest_hash,
            bundle.replay_chart_artifact_id,
            bundle.replay_chart_manifest_hash,
        )
        setups = _cached_setup_frame(bundle.replay_chart_artifact_id, ctx)
    except Exception as error:
        st_module.error(f"Setup bundle failed verification: {_sanitize_error(error)}")
        return

    filtered = _filtered_setups(st_module, setups)
    if filtered.empty:
        st_module.info("No setups match the current filters.")
        return
    options = list(filtered["setup_id"].astype(str))
    labels = {
        str(row["setup_id"]): _setup_label(row) for _, row in filtered.iterrows()
    }
    _sanitize_select(st_module, _SETUP_KEY, tuple(options))

    nav1, nav2, jump_col, pick = st_module.columns([1, 1, 2, 3])
    nav1.button(
        "◀ Prev setup",
        key=f"{_STATE_PREFIX}setup_prev",
        on_click=_step_setup,
        args=(options, -1),
        use_container_width=True,
    )
    nav2.button(
        "Next setup ▶",
        key=f"{_STATE_PREFIX}setup_next",
        on_click=_step_setup,
        args=(options, 1),
        use_container_width=True,
    )
    with jump_col:
        query = st_module.text_input(
            "Jump to setup (exact ID or unique prefix)",
            key=f"{_STATE_PREFIX}setup_jump",
        )
        if query:
            try:
                resolved = resolve_setup_selection(ctx, query)
            except Exception as error:
                st_module.warning(f"Jump not resolved: {_sanitize_error(error)}")
            else:
                resolved_id = str(resolved["setup_id"])
                if resolved_id in options:
                    st.session_state[_SETUP_KEY] = resolved_id
                else:
                    st_module.warning(
                        "Resolved setup is excluded by the current filters."
                    )
    with pick:
        setup_id = st_module.selectbox(
            "Exact setup ID",
            options,
            format_func=lambda value: labels.get(value, value),
            key=_SETUP_KEY,
        )
    row = filtered[filtered["setup_id"].astype(str) == setup_id].iloc[0]

    stage_col, info_col = st_module.columns([1.4, 3])
    with stage_col:
        stage = st_module.selectbox(
            "Evidence as of stage",
            SETUP_STAGE_ORDER,
            index=len(SETUP_STAGE_ORDER) - 1,
            key=f"{_STATE_PREFIX}setup_stage",
        )
    with info_col:
        if stage == "terminal":
            st_module.caption(
                "Terminal = the setup's full life (all persisted evidence)."
            )
        else:
            st_module.warning(
                f"POINT-IN-TIME — evidence at or before stage '{stage}'; later "
                "events are hidden and counted."
            )
    try:
        evidence = _cached_setup_evidence(
            bundle.replay_chart_artifact_id, setup_id, stage, ctx
        )
    except Exception as error:
        st_module.error(f"Setup evidence unavailable: {_sanitize_error(error)}")
        return
    hidden = int(evidence.gating_report.get("hidden_events") or 0)
    if hidden > 0:
        st_module.warning(
            f"{hidden} of {evidence.gating_report.get('total_events')} events "
            f"hidden by the '{stage}' stage gate."
        )

    start_ts = pd.Timestamp(row["activation_ts_utc"])
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    end_ts = pd.Timestamp(row["display_end_ts_utc"])
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    htf_tf = (
        int(row["htf_tf_seconds"]) if pd.notna(row["htf_tf_seconds"]) else None
    )
    parent_tf = (
        int(row["parent_tf_seconds"]) if pd.notna(row["parent_tf_seconds"]) else None
    )
    bars_by_pane: dict[int, pd.DataFrame] = {}
    for timeframe in sorted({60, *(tf for tf in (parent_tf, htf_tf) if tf)}):
        try:
            bars_by_pane[timeframe] = bars_for_pane(
                ctx.base,
                timeframe_seconds=timeframe,
                start_ts=start_ts,
                end_ts=end_ts,
            )
        except RangeTooLargeError as error:
            bars_by_pane[timeframe] = pd.DataFrame()
            st_module.caption(
                f"{timeframe}s pane omitted — {_sanitize_error(error)}"
            )
        except Exception as error:
            st_module.error(f"Bars unavailable: {_sanitize_error(error)}")
            return

    execution_pane_only = st_module.checkbox(
        "Execution pane only",
        value=True,
        key=f"{_STATE_PREFIX}setup_execution_only",
        help=(
            "Show only the 1m execution pane; parent/HTF panes are hidden."
        ),
    )
    chart_col, panel_col = st_module.columns([4.2, 1.0])
    with chart_col, st_module.container(border=True):
        figure, omissions = build_setup_figure(
            evidence=evidence,
            bars_by_pane=bars_by_pane,
            parent_tf=parent_tf,
            htf_tf=htf_tf,
            range_bounds=(start_ts, end_ts),
        )
        if execution_pane_only:
            figure = collapse_to_execution_pane(figure)
        # hold-left-drag pans; mouse wheel zooms (box-zoom stays in the modebar).
        figure.update_layout(dragmode="pan")
        st_module.plotly_chart(
            figure,
            use_container_width=True,
            key=f"{_STATE_PREFIX}setup_chart",
            config={
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"ifvg_setup_{setup_id[:12]}_{stage}",
                    "scale": 2,
                }
            },
        )
        for line in omissions.summary_lines():
            st_module.caption(f"⚠ {line}")
    with panel_col:
        _render_setup_panel(st_module, ctx, bundle, evidence, row)


def render_verifier_section(st_module, pair, entry: dict) -> str | None:
    """The chart verifier; returns the selected candidate_id (shared selection).

    Returns None when the replay-chart artifact is unavailable — the caller
    then falls back to its own exact-ID selector.
    """
    try:
        pair_ref = ArtifactPairRef(
            profile_name=str(entry["profile_name"]),
            v2_dataset_id=str(entry["v2_artifact_id"]),
            v2_manifest_hash=str(entry["v2_manifest_payload_sha256"]),
            v3_dataset_id=str(entry["v3_artifact_id"]),
            v3_manifest_hash=str(entry["v3_manifest_payload_sha256"]),
        )
    except Exception as error:
        st_module.caption(f"Verifier unavailable: {_sanitize_error(error)}")
        return None
    # Selection mode: candidate mode also serves the decision/trade exact-ID
    # jumps (they resolve into a candidate); setup mode reviews every setup,
    # including the candidate-less ones the candidate verifier cannot show.
    selection_mode = st_module.radio(
        "Selection mode",
        ("candidate", "setup"),
        horizontal=True,
        key=f"{_STATE_PREFIX}selection_mode",
        help=(
            "candidate mode serves candidate / decision / trade exact-ID "
            "review; setup mode reviews every setup, including candidate-less "
            "ones."
        ),
    )
    if selection_mode == "setup":
        _render_setup_section(st_module, pair_ref)
        return None
    try:
        catalog = read_replay_chart_catalog(ROOT / REPLAY_CHART_CATALOG)
        replay_id = find_replay_artifact(catalog, pair_ref)
    except Exception as error:
        st_module.error(f"Replay-chart catalog error: {_sanitize_error(error)}")
        return None
    if replay_id is None:
        st_module.info(
            "No replay-chart artifact exists for this exact artifact pair yet. "
            "Build it externally, then reload:"
        )
        st_module.code(
            f"python scripts/ifvg_build_replay_chart.py build --profile "
            f"{pair_ref.profile_name}",
            language="powershell",
        )
        return None
    try:
        ctx = _cached_replay_context(
            pair_ref.profile_name,
            pair_ref.v2_dataset_id,
            pair_ref.v2_manifest_hash,
            pair_ref.v3_dataset_id,
            pair_ref.v3_manifest_hash,
            replay_id,
            str(catalog[replay_id].get("replay_chart_manifest_payload_sha256")),
            pair,
        )
        candidates = _cached_candidate_frame(replay_id, ctx)
    except Exception as error:
        st_module.error(f"Verifier context failed verification: {_sanitize_error(error)}")
        return None

    _apply_pending_jump(st_module, ctx)
    filtered = _filtered_candidates(st_module, candidates)
    if filtered.empty:
        st_module.info("No candidates match the current filters.")
        return None
    options = list(filtered["candidate_id"])
    labels = {
        row["candidate_id"]: _candidate_label(row) for _, row in filtered.iterrows()
    }
    _sanitize_select(st_module, _CANDIDATE_KEY, tuple(options))
    executed_options = list(filtered.loc[filtered["executed"], "candidate_id"])

    nav1, nav2, nav3, nav4, pick = st_module.columns([1, 1, 1, 1, 3])
    nav1.button(
        "◀ Prev candidate",
        key=f"{_STATE_PREFIX}verifier_prev",
        on_click=_step_candidate,
        args=(options, -1),
        use_container_width=True,
    )
    nav2.button(
        "Next candidate ▶",
        key=f"{_STATE_PREFIX}verifier_next",
        on_click=_step_candidate,
        args=(options, 1),
        use_container_width=True,
    )
    nav3.button(
        "◀ Prev trade",
        key=f"{_STATE_PREFIX}verifier_prev_trade",
        on_click=_step_candidate,
        args=(executed_options, -1),
        use_container_width=True,
        disabled=not executed_options,
    )
    nav4.button(
        "Next trade ▶",
        key=f"{_STATE_PREFIX}verifier_next_trade",
        on_click=_step_candidate,
        args=(executed_options, 1),
        use_container_width=True,
        disabled=not executed_options,
    )
    with pick:
        candidate_id = st_module.selectbox(
            "Exact candidate ID",
            options,
            format_func=lambda value: labels.get(value, value),
            key=_CANDIDATE_KEY,
        )
    row = filtered[filtered["candidate_id"] == candidate_id].iloc[0]

    mode_col, stage_col, range_col, layer_col = st_module.columns([1.2, 2.2, 1.6, 2.0])
    with mode_col:
        mode_label = st_module.radio(
            "Mode",
            ("Full audit", "Point-in-time"),
            key=f"{_STATE_PREFIX}verifier_mode",
        )
    point_in_time = mode_label == "Point-in-time"
    try:
        gates_probe = _cached_evidence(replay_id, candidate_id, "full_audit", None, ctx)
    except Exception as error:
        st_module.error(f"Evidence unavailable: {_sanitize_error(error)}")
        return candidate_id
    gateable = [name for name in _STAGE_ORDER if name in gates_probe.stage_gates]
    stage = None
    with stage_col:
        if point_in_time:
            stage = st_module.select_slider(
                "Evidence as of stage",
                options=gateable,
                value=gateable[-1] if gateable else None,
                key=f"{_STATE_PREFIX}verifier_stage",
            )
        else:
            st_module.caption("Full audit shows all persisted evidence, outcome included.")
    with range_col:
        range_kind = st_module.radio(
            "Range",
            ("Setup", "Trade", "Formation", "Custom"),
            key=f"{_STATE_PREFIX}verifier_range",
        ).lower()
    with layer_col:
        layers = VerifierLayers(
            structure=st_module.checkbox(
                "Structure", key=f"{_STATE_PREFIX}verifier_layer_structure"
            ),
            displacement=st_module.checkbox(
                "Displacement", key=f"{_STATE_PREFIX}verifier_layer_displacement"
            ),
            pools=st_module.checkbox(
                "EQH/EQL pools", key=f"{_STATE_PREFIX}verifier_layer_pools"
            ),
            sessions=st_module.checkbox(
                "Sessions", value=True, key=f"{_STATE_PREFIX}verifier_layer_sessions"
            ),
            zone_projection=st_module.checkbox(
                "Project zones onto 1m",
                value=True,
                key=f"{_STATE_PREFIX}verifier_layer_projection",
            ),
        )
        execution_pane_only = st_module.checkbox(
            "Execution pane only",
            value=True,
            key=f"{_STATE_PREFIX}verifier_layer_execution_only",
            help=(
                "Show only the 1m execution pane; parent/HTF panes are "
                "hidden (their zones stay visible via 1m projection)."
            ),
        )

    custom_bounds = None
    if range_kind == "custom":
        first, second = st_module.columns(2)
        raw_start = first.text_input(
            "Custom start (UTC ISO)", key=f"{_STATE_PREFIX}verifier_custom_start"
        )
        raw_end = second.text_input(
            "Custom end (UTC ISO)", key=f"{_STATE_PREFIX}verifier_custom_end"
        )
        if not raw_start or not raw_end:
            st_module.info("Enter both custom bounds (e.g. 2026-01-13T03:00:00Z).")
            return candidate_id
        try:
            custom_bounds = (
                pd.Timestamp(raw_start).tz_convert("UTC")
                if pd.Timestamp(raw_start).tzinfo
                else pd.Timestamp(raw_start, tz="UTC"),
                pd.Timestamp(raw_end).tz_convert("UTC")
                if pd.Timestamp(raw_end).tzinfo
                else pd.Timestamp(raw_end, tz="UTC"),
            )
        except (ValueError, TypeError) as error:
            st_module.error(f"Custom bounds invalid: {_sanitize_error(error)}")
            return candidate_id

    if point_in_time:
        st_module.warning(
            f"POINT-IN-TIME REPLAY — evidence as of stage '{stage}'. "
            "Later evidence is withheld; ungateable objects are hidden and listed."
        )
    else:
        st_module.info("FULL AUDIT — all persisted evidence including outcome is visible.")

    try:
        evidence = _cached_evidence(
            replay_id,
            candidate_id,
            "point_in_time" if point_in_time else "full_audit",
            stage if point_in_time else None,
            ctx,
        )
        start_ts, end_ts = chart_range(
            ctx, evidence, range_kind, custom_bounds=custom_bounds
        )
        parent_tf = int(evidence.range_row["parent_timeframe_seconds"])
        htf_tf = int(evidence.range_row["htf_timeframe_seconds"])
        bars_by_pane = {
            timeframe: bars_for_pane(
                ctx, timeframe_seconds=timeframe, start_ts=start_ts, end_ts=end_ts
            )
            for timeframe in (60, parent_tf, htf_tf)
        }
    except RangeTooLargeError as error:
        st_module.error(f"Range refused: {_sanitize_error(error)}")
        return candidate_id
    except ReplayAuthorizationError as error:
        st_module.error(f"Not authorized: {_sanitize_error(error)}")
        return candidate_id
    except MissingEvidenceError as error:
        st_module.error(f"Missing evidence: {_sanitize_error(error)}")
        return candidate_id
    except Exception as error:
        st_module.error(f"Verifier failed: {_sanitize_error(error)}")
        return candidate_id

    if range_kind == "formation" and len(bars_by_pane.get(60, ())) == 0:
        st_module.caption(
            "The 1m pane is empty over this formation range (HTF formation predates "
            "the day) — parent/HTF panes carry the formation context."
        )

    chart_col, panel_col = st_module.columns([4.2, 1.0])
    with chart_col, st_module.container(border=True):
        figure, omissions = build_verifier_figure(
            evidence=evidence,
            bars_by_pane=bars_by_pane,
            parent_tf=parent_tf,
            htf_tf=htf_tf,
            range_bounds=(start_ts, end_ts),
            layers=layers,
            blocked_reasons=(row["block_reasons"] or None) if row["blocked"] else None,
        )
        if execution_pane_only:
            figure = collapse_to_execution_pane(figure)
        # hold-left-drag pans; mouse wheel zooms (box-zoom stays in the modebar).
        figure.update_layout(dragmode="pan")
        st_module.plotly_chart(
            figure,
            use_container_width=True,
            key=f"{_STATE_PREFIX}verifier_chart",
            config={
                "scrollZoom": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": (
                        f"ifvg_verifier_{candidate_id[:12]}_"
                        f"{stage if point_in_time else 'full'}"
                    ),
                    "scale": 2,
                }
            },
        )
        for line in omissions.summary_lines():
            st_module.caption(f"⚠ {line}")
        st_module.caption(
            "Crosshair: unified hover spans all panes; a single spike line across "
            "panes is not supported by the charting library."
        )
    with panel_col:
        if row["is_warmup"]:
            st_module.warning(
                "Warmup-origin candidate — not part of post-warmup research or "
                "execution reports."
            )
        _render_side_panel(st_module, ctx, evidence, row)
    return candidate_id
