"""Full Pipeline Run surface (R5; FUX §30, FUX-PIPE-001..006).

Configure / Preview / Launch / Monitor / Resume-Retry / Publish over the
16-stage pipeline runner. Thin widget layer: every derivation is pure
(`study_presentation`), every artifact read goes through the providers or
the exact-ID stores, launches happen ONLY inside the explicit button
handler through the single detached `_spawn_pipeline_job` seam, and every
error surface is sanitized. Session state lives under the dedicated
``ifvg_pipeline_v1_*`` namespace.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from ifvg_ui_common import (
    PIPELINE_STATE_PREFIX,
    cli_escape_hatch,
    dev_only_badge,
    identity_block,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    verification_badge,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

__all__ = ["render_pipeline_run", "PIPELINE_STATE_ROOT"]

#: The mutable pipeline job root (monkeypatched by tests, like the study roots).
PIPELINE_STATE_ROOT = _REPO_ROOT / "data/ifvg_pipeline_jobs"

_PIPE = f"{PIPELINE_STATE_PREFIX}run_"
_PHASE_KEY = f"{PIPELINE_STATE_PREFIX}phase"
_SELECTED_KEY = f"{PIPELINE_STATE_PREFIX}selected_pipeline_id"

_PHASES = ("Configure", "Preview", "Launch", "Monitor", "Resume / Retry", "Publish")

_STRATEGY_PLAN_LABEL = "Strategy pipeline (replays · companions · prop · frontier · publish)"
_FULL_PLAN_LABEL = (
    "Full 16-stage pipeline (adds feature/label/fold/model stages; S11 stays blocked)"
)


def _spawn_pipeline_job(command: list[str]) -> int:
    """The ONE detached launch seam (FUX-PIPE-003; scan-pinned)."""

    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    process = subprocess.Popen(  # noqa: S603 — exact interpreter + repo script
        command, cwd=_REPO_ROOT, creationflags=creation_flags
    )
    return process.pid


def _stage_plan(full: bool):
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        QuantLabPipelineStage as Stage,
    )

    if full:
        return tuple(Stage)
    return (
        Stage.S00_VALIDATE_INPUTS,
        Stage.S01_PREPARE_STRATEGY_PROFILES,
        Stage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
        Stage.S03_BUILD_OR_REUSE_FSM_AUDIT,
        Stage.S04_BUILD_OR_REUSE_REPLAY_CHARTS,
        Stage.S12_RUN_PROP_HISTORICAL_REPLAYS,
        Stage.S13_RUN_BOOTSTRAP_AND_STRESS,
        Stage.S14_BUILD_FRONTIER_AND_INSIGHTS,
        Stage.S15_VERIFY_AND_PUBLISH,
    )


def _logistic_protocol_id() -> str:
    """The one bundle-parametrized model protocol (R5B; the CatBoost fold
    runner is tier-locked in the frozen M0–M3 lane) — always the registered
    constant, never a drifting literal."""

    from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (  # noqa: PLC0415
        LOGISTIC_PROTOCOL_ID,
    )

    return LOGISTIC_PROTOCOL_ID


def _bundle_is_mbp1_bearing(bundle_key: str) -> bool:
    from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (  # noqa: PLC0415
        mbp1_block_keys_in_bundle,
    )
    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        BlockUnavailableError,
    )
    from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (  # noqa: PLC0415
        resolve_bundle,
    )

    try:
        return mbp1_block_keys_in_bundle(resolve_bundle(bundle_key))
    except (BlockUnavailableError, ValueError):
        return False


def _mbp1_default_ids(
    st_module, roots: Mapping[str, Any]
) -> tuple[dict[str, str], str | None]:
    """Exact artifact ids for the MBP-1 panel via the provider layer
    (`mbp1_stage_evidence_defaults`): manifest-verified exact-ID reads;
    absence yields empty inputs, but a store-integrity failure yields a
    sanitized note the panel surfaces (safety review S6)."""

    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        list_pipeline_runs,
        mbp1_stage_evidence_defaults,
    )

    try:
        runs = list_pipeline_runs(PIPELINE_STATE_ROOT)
        pipeline_id = st_module.session_state.get(_SELECTED_KEY) or (
            runs[0].pipeline_semantic_id if runs else None
        )
        return mbp1_stage_evidence_defaults(
            PIPELINE_STATE_ROOT, Path(roots["store_root"]), pipeline_id
        )
    except Exception:  # noqa: BLE001 — defaults are a convenience, never a gate
        return {}, None


def _resolvable_bundles() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """(resolvable bundle keys, ((blocked key, reason), ...)) — planned and
    blocked entries stay VISIBLE with their status, never silently hidden."""

    from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import (  # noqa: PLC0415
        BlockUnavailableError,
    )
    from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import (  # noqa: PLC0415
        FEATURE_BUNDLE_REGISTRY,
        resolve_bundle,
    )

    available: list[str] = []
    blocked: list[tuple[str, str]] = []
    for key in FEATURE_BUNDLE_REGISTRY:
        try:
            resolve_bundle(key)
        except (BlockUnavailableError, ValueError) as error:
            blocked.append((key, sanitize_error(error)))
        else:
            available.append(key)
    return tuple(available), tuple(blocked)


def _model_protocol_rows() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (  # noqa: PLC0415
        MODEL_PROTOCOL_REGISTRY,
        ModelProtocolStatus,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        POST_V1_REGIME_ALGORITHM_KEYS,
    )

    available = tuple(
        key
        for key, entry in MODEL_PROTOCOL_REGISTRY.items()
        if entry.status is ModelProtocolStatus.AVAILABLE
    )
    planned = [
        (key, f"{entry.status.value}: {entry.reason}")
        for key, entry in MODEL_PROTOCOL_REGISTRY.items()
        if entry.status is not ModelProtocolStatus.AVAILABLE
    ]
    planned.extend(
        (key, "post-V1 regime-expansion release; no fit implementation is callable in V1")
        for key in POST_V1_REGIME_ALGORITHM_KEYS
    )
    return available, tuple(planned)


def _active_mode5_draft(st_module, roots: Mapping[str, Any]):
    from alpha_lab.agents.data_infra.ifvg.study_drafts import (  # noqa: PLC0415
        list_drafts,
        load_draft,
    )

    draft_root = Path(roots["draft_root"])
    draft_id = st_module.session_state.get(f"{PIPELINE_STATE_PREFIX}draft_id")
    if draft_id:
        try:
            draft = load_draft(draft_root, str(draft_id))
        except Exception:  # noqa: BLE001 — listing fallback below
            draft = None
        if draft is not None and draft.mode_id == "full_pipeline_run":
            return draft
    for candidate in list_drafts(draft_root):
        if candidate.mode_id == "full_pipeline_run" and not candidate.frozen_search_id:
            return candidate
    return None


def _assemble_pipeline_spec(charter_payload, charter_id: str, *, fields: Mapping[str, Any]):
    from alpha_lab.agents.data_infra.ifvg.data_access import (  # noqa: PLC0415
        allowlist_sha256,
    )
    from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
        canonical_contract_sha256,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        FOLD_PROTOCOL_ID_V1,
        PipelineSemanticSpecPayload,
    )

    full = bool(fields.get("full_plan"))
    verification = (
        charter_payload.date_policy.access_policy_id
        == "verification_fixed_allowlist_max5_v1"
    )
    policy_ids: tuple[str, ...] = ()
    plan = _stage_plan(full)
    if any(stage.value == "12_run_prop_historical_replays" for stage in plan):
        from alpha_lab.agents.data_infra.ifvg.search.executors import (  # noqa: PLC0415
            synthetic_firm_specs,
        )

        policy_ids = tuple(
            sorted(
                spec.policy_set_envelope().account_policy_set_id
                for spec in synthetic_firm_specs()
            )
        )
    return PipelineSemanticSpecPayload(
        run_scope="verification_5d" if verification else "full_authorized_development",
        date_allowlist=charter_payload.date_policy.replay_dates,
        allowlist_hash=allowlist_sha256(charter_payload.date_policy.replay_dates),
        warmup_policy_id=(
            "zero_real_warmup_seed_snapshot_v1"
            if verification
            else "frozen_ten_day_warmup_v1"
        ),
        search_charter_id=charter_id,
        source_artifact_ids=(),
        feature_bundle_ids=(str(fields.get("bundle")),) if full else (),
        label_policy_id=str(fields.get("label_policy")) if full else None,
        fold_protocol_id=FOLD_PROTOCOL_ID_V1 if full else None,
        model_protocol_id=str(fields.get("model_protocol")) if full else None,
        cost_policy_sha256=canonical_contract_sha256(charter_payload.cost_policy),
        account_policy_set_ids=policy_ids,
        portfolio_policy_ids=(),
        simulation_protocol=charter_payload.simulation_protocol,
        software_commits={
            "quant_lab": charter_payload.quant_lab_commit,
            "strategy_core": charter_payload.strategy_core_commit,
        },
        stage_plan=plan,
    )


def _configure_fields(st_module) -> dict[str, Any]:
    available_bundles, blocked_bundles = _resolvable_bundles()
    available_models, planned_models = _model_protocol_rows()
    full_plan = (
        st_module.radio(
            "Selected stage plan (capability-scoped)",
            (_STRATEGY_PLAN_LABEL, _FULL_PLAN_LABEL),
            key=f"{_PIPE}plan",
            help=(
                "A strategy-only plan never waits for MBP-1 activation or "
                "regime fitting; the full plan adds the supervised stages and "
                "renders S11 in its blocked state."
            ),
        )
        == _FULL_PLAN_LABEL
    )
    bundle = None
    label_policy = None
    model_protocol = None
    if full_plan:
        sanitize_select(st_module, f"{_PIPE}bundle", list(available_bundles))
        bundle = st_module.selectbox(
            "Feature bundle (available blocks only)",
            list(available_bundles),
            key=f"{_PIPE}bundle",
        )
        mbp1_selected = bool(bundle) and _bundle_is_mbp1_bearing(str(bundle))
        if mbp1_selected:
            from ifvg_mbp1_panels import research_only_offline_badge  # noqa: PLC0415

            research_only_offline_badge(st_module)
            model_options = [_logistic_protocol_id()]
            st_module.caption(
                "MBP-1-bearing bundles train the controlled Baseline vs "
                "Baseline+MBP-1 study under the logistic protocol on both "
                "arms — the CatBoost fold runner is tier-locked in the "
                "frozen M0–M3 lane."
            )
        else:
            model_options = list(available_models)
        sanitize_select(st_module, f"{_PIPE}model", model_options)
        model_protocol = st_module.selectbox(
            "Model protocol (the ladder always includes the prevalence reference)",
            model_options,
            key=f"{_PIPE}model",
        )
        label_policy = st_module.selectbox(
            "Label policy",
            ["synthetic_fixture_labels_v1"],
            key=f"{_PIPE}labels",
            help=(
                "Verification-scope label derivation; the real label chain "
                "rides the owner-authorized operator run."
            ),
        )
        st_module.caption(
            "Fold protocol: `ifvg_context_walkforward_40_5_5_2_v1` (frozen; "
            "read-only)."
        )
    if blocked_bundles:
        st_module.markdown("**Planned / blocked feature entries** (visible, disabled)")
        st_module.dataframe(
            [
                {"Entry": key, "Status / reason": reason}
                for key, reason in blocked_bundles
            ],
            width="stretch",
            hide_index=True,
        )
    if planned_models:
        st_module.markdown("**Planned / post-V1 model entries** (visible, disabled)")
        st_module.dataframe(
            [
                {"Entry": key, "Status / reason": reason}
                for key, reason in planned_models
            ],
            width="stretch",
            hide_index=True,
        )
    max_workers = st_module.slider(
        "Worker limit (operational — never part of the scientific identity)",
        min_value=1,
        max_value=4,
        value=1,
        key=f"{_PIPE}workers",
    )
    return {
        "full_plan": full_plan,
        "bundle": bundle,
        "label_policy": label_policy,
        "model_protocol": model_protocol,
        "max_workers": int(max_workers),
    }


def _render_configure(st_module, roots: Mapping[str, Any], draft) -> dict[str, Any] | None:
    st_module.subheader("Configure")
    if draft is None:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail=(
                "no Full Pipeline Run draft is open — create one in "
                "New Study → Full Pipeline Run (steps 1–7 configure the "
                "underlying charter)"
            ),
        )
        return None
    baseline = draft.step_payload("baseline")
    validation = draft.step_payload("validation")
    st_module.caption(
        f"Charter half from draft `{draft.draft_id[:12]}…` — baseline "
        f"{baseline.get('baseline_profile_name', '—')}; run scope and dates "
        "come from the Validation step; prop/risk selections ride the charter."
    )
    dates = tuple(validation.get("real_dates") or ())
    if dates:
        st_module.caption(f"Replay dates ({len(dates)}): {', '.join(dates)}")
    return _configure_fields(st_module)


def _render_preview(st_module, roots: Mapping[str, Any], draft, fields) -> None:
    st_module.subheader("Preview")
    if draft is None or fields is None:
        st_module.caption("Complete Configure first.")
        return
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        derive_stage_plan_readiness,
    )
    from alpha_lab.agents.data_infra.ifvg.study_presentation import (  # noqa: PLC0415
        enumerate_child_count,
    )

    try:
        import ifvg_study_wizard as wizard  # noqa: PLC0415

        charter_payload = wizard._assemble_charter(draft, roots)
    except Exception as error:  # noqa: BLE001 — sanitized preview refusal
        st_module.error(f"The draft does not assemble yet: {sanitize_error(error)}")
        return
    from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
        canonical_contract_sha256,
    )

    try:
        spec = _assemble_pipeline_spec(
            charter_payload,
            canonical_contract_sha256(charter_payload),
            fields=fields,
        )
    except Exception as error:  # noqa: BLE001
        st_module.error(f"Pipeline spec refuses to assemble: {sanitize_error(error)}")
        return
    children = enumerate_child_count(
        {
            key: tuple(values)
            for key, values in (
                draft.step_payload("search_space").get("axis_selections") or {}
            ).items()
        }
    )
    specs_count = len(spec.account_policy_set_ids)
    modes = spec.simulation_protocol.modes
    prop_sims = children * max(specs_count, 0) * len(modes)
    rows = [
        ("Resolved dates", f"{len(spec.date_allowlist)} — {', '.join(spec.date_allowlist)}"),
        ("Planned stages", f"{len(spec.stage_plan)} of 16"),
        ("Strategy child count", str(children)),
        ("Feature-view count", str(len(spec.feature_bundle_ids))),
        (
            "Fold / model count",
            "0 valid folds expected on a ≤5-day window (frozen 40/5/5/2 "
            "protocol) — the ladder records the safe-failure shape"
            if spec.run_scope.value == "verification_5d" and fields["full_plan"]
            else ("ladder rungs: 3" if fields["full_plan"] else "not planned"),
        ),
        ("Prop simulation count", str(prop_sims)),
        (
            "Runtime estimate by phase",
            (
                f"replays ≈ {children} children × {len(spec.date_allowlist)} "
                f"days × ~15 s/day; prop ≈ {prop_sims} sims × <1 s; ML stages "
                "in-memory (operational annotation, never identity)"
            ),
        ),
        (
            "Storage estimate by phase",
            "envelopes + references only; per-child alarm at 200 MB",
        ),
        (
            "Verified reuse",
            "resolved at launch: identical replay identities reuse published "
            "artifacts (never re-run)",
        ),
        (
            "New artifacts expected",
            "core replays · memberships · costed evaluations · policy sets · "
            "account simulations · frontier · insights · comparison results · "
            "stage results · pipeline result",
        ),
    ]
    st_module.dataframe(
        [{"Field": name, "Value": value} for name, value in rows],
        width="stretch",
        hide_index=True,
    )
    readiness = derive_stage_plan_readiness(spec)
    st_module.markdown("**Stage-plan readiness (capability-scoped)**")
    st_module.dataframe(
        [
            {
                "Stage": entry.stage.value,
                "State": entry.state,
                "Reason": entry.reason or "",
            }
            for entry in readiness.entries
        ],
        width="stretch",
        hide_index=True,
    )
    if not readiness.launchable:
        st_module.error(
            "This plan references unavailable capabilities and cannot launch."
        )


def _render_launch(st_module, roots: Mapping[str, Any], draft, fields) -> None:
    from alpha_lab.agents.data_infra.ifvg.study_status import (  # noqa: PLC0415
        FULL_SCOPE_ACKNOWLEDGEMENT,
        FULL_SCOPE_WARNING_TEXT,
    )

    st_module.subheader("Launch")
    if draft is None or fields is None:
        st_module.caption("Complete Configure first.")
        return
    validation = draft.step_payload("validation")
    verification = (
        str(validation.get("run_scope") or "verification_5d")
        != "full_authorized_development"
    )
    if verification:
        verification_badge(st_module)
    acknowledged = True
    if not verification:
        st_module.error(FULL_SCOPE_WARNING_TEXT)
        typed = st_module.text_input(
            f"Type '{FULL_SCOPE_ACKNOWLEDGEMENT}' to enable the launch control",
            value="",
            key=f"{_PIPE}ack",
        )
        acknowledged = typed.strip() == FULL_SCOPE_ACKNOWLEDGEMENT
    if st_module.button(
        "Freeze Pipeline Specification and Launch",
        key=f"{_PIPE}launch",
        type="primary",
        disabled=not acknowledged,
    ):
        _freeze_and_launch_pipeline(st_module, roots, draft, fields)


def _freeze_and_launch_pipeline(st_module, roots: Mapping[str, Any], draft, fields) -> None:
    """Freeze the pipeline spec and spawn the detached job — strictly
    inside the Launch button handler (FUX-PIPE-003; the Resume phase's
    retry handler is the one other spawner, both through the single
    scan-pinned `_spawn_pipeline_job` seam)."""

    import ifvg_study_wizard as wizard  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        CharterValidationError,
        SearchCharterEnvelope,
        save_charter,
        validate_charter,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        PipelineSemanticIdentity,
        assert_stage_plan_launchable,
    )
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
        pipeline_entry_key_for_charter,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        save_or_reuse_envelope,
    )
    from alpha_lab.agents.data_infra.ifvg.study_drafts import mark_frozen  # noqa: PLC0415

    store_root = Path(roots["store_root"])
    try:
        charter_payload = wizard._assemble_charter(draft, roots)
        if "unknown" in (
            charter_payload.strategy_core_commit,
            charter_payload.quant_lab_commit,
        ):
            raise CharterValidationError(
                "source-commit provenance could not be resolved; freezing is "
                "refused rather than stamping 'unknown' into the identity"
            )
        validate_charter(
            charter_payload, as_of_utc=datetime.now(UTC).isoformat(timespec="seconds")
        )
        charter = SearchCharterEnvelope.from_payload(charter_payload)
        save_charter(store_root, charter)
        spec = _assemble_pipeline_spec(charter_payload, charter.search_id, fields=fields)
        assert_stage_plan_launchable(spec)
        semantic = PipelineSemanticIdentity.from_payload(spec)
        save_or_reuse_envelope(store_root, "pipeline_specs", semantic)
    except CharterValidationError as error:
        st_module.error(f"Cannot freeze (fail-closed): {sanitize_error(error)}")
        return
    except PermissionError as error:
        st_module.error(f"Launch refused: {sanitize_error(error)}")
        return
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Freeze failed: {sanitize_error(error)}")
        return
    try:
        mark_frozen(Path(roots["draft_root"]), draft, search_id=charter.search_id)
    except Exception as error:  # noqa: BLE001 — draft linkage is non-fatal
        st_module.caption(f"draft could not be marked frozen: {sanitize_error(error)}")
    identity_block(st_module, "Frozen pipeline semantic id", semantic.pipeline_semantic_id)
    entry_key = pipeline_entry_key_for_charter(charter)
    if entry_key is None:
        render_empty_state(
            st_module,
            "runner_executor_planned",
            detail=(
                "full-development pipelines have no registered executor; the "
                "operator full run is a separate, explicitly authorized action"
            ),
        )
        return
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "ifvg_pipeline_job.py"),
        "start",
        "--pipeline-id",
        semantic.pipeline_semantic_id,
        "--store-root",
        str(store_root),
        "--state-root",
        str(PIPELINE_STATE_ROOT),
        "--runner-entry-key",
        entry_key,
        "--max-workers",
        str(fields["max_workers"]),
    ]
    try:
        pid = _spawn_pipeline_job(command)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Detached launch failed: {sanitize_error(error)}")
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_pipeline_job.py start --pipeline-id "
            f"{semantic.pipeline_semantic_id} --runner-entry-key {entry_key}",
            reason="launch the detached pipeline worker manually",
        )
        return
    st_module.success(f"Pipeline launched detached (pid {pid}).")
    st_module.session_state[_SELECTED_KEY] = semantic.pipeline_semantic_id
    st_module.session_state[_PHASE_KEY] = "Monitor"


def _selected_run(st_module, runs):
    options = [run.pipeline_semantic_id for run in runs]
    preselected = st_module.session_state.get(_SELECTED_KEY)
    index = options.index(preselected) if preselected in options else 0
    labels = {
        run.pipeline_semantic_id: (
            f"{run.pipeline_semantic_id[:12]}… · {run.run_scope} · "
            f"attempts {run.attempt_count} · {run.publication_state}"
        )
        for run in runs
    }
    sanitize_select(st_module, f"{_PIPE}selected", options)
    return st_module.selectbox(
        "Pipeline run",
        options,
        index=index,
        format_func=lambda value: labels.get(value, value[:12]),
        key=f"{_PIPE}selected",
    )


def _render_monitor_body(st_module, *, roots: Mapping[str, Any], pipeline_id: str) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        read_pipeline_state,
        request_pipeline_cancel,
    )
    from alpha_lab.agents.data_infra.ifvg.study_presentation import (  # noqa: PLC0415
        derive_pipeline_progress,
        derive_pipeline_stage_rows,
    )

    state = read_pipeline_state(PIPELINE_STATE_ROOT, pipeline_id)
    identity_block(st_module, "pipeline_semantic_id", pipeline_id)
    if state is None:
        render_empty_state(
            st_module,
            "artifact_unavailable",
            detail="the pipeline status file is missing or unreadable",
        )
        cli_escape_hatch(
            st_module,
            f"python scripts/ifvg_pipeline_job.py status --pipeline-id {pipeline_id}",
            reason="the status file could not be read",
        )
        return
    progress = derive_pipeline_progress(state)
    st_module.progress(
        progress.fraction,
        text=(
            f"{progress.terminal_count}/{progress.planned_count} planned stages "
            f"terminal · attempt {progress.attempt_count}"
        ),
    )
    if progress.current_stage_title:
        st_module.caption(f"Current stage: {progress.current_stage_title}")
    # FUX §30.4 fields (adversarial M-3): elapsed / remaining / workers /
    # current child — operational annotations, never identity
    attempts_list = list(state.get("attempts") or ())
    if attempts_list:
        latest = dict(attempts_list[-1])
        started = str(latest.get("started_at") or "")
        ended = latest.get("ended_at")
        elapsed_note = f"started {started}" + (
            f" · ended {ended}" if ended else " · running"
        )
        workers = dict(latest.get("worker_policy") or {}).get("max_workers", 1)
        pending = max(progress.planned_count - progress.terminal_count, 0)
        remaining_note = (
            "0 stages remaining (run terminal)"
            if pending == 0
            else f"{pending} planned stage(s) remaining (estimate only)"
        )
        st_module.caption(
            f"Elapsed: {elapsed_note} · Remaining: {remaining_note} · "
            f"Workers configured: {workers} (children execute sequentially "
            "in R5)"
        )
    running_children = [
        row
        for row in (state.get("children") or ())
        if isinstance(row, dict) and row.get("state") == "running"
    ]
    if running_children:
        st_module.caption(
            "Current child: ordinal "
            f"{running_children[0].get('ordinal')} · "
            f"{str(running_children[0].get('core_replay_id'))[:12]}…"
        )
    warnings_list = [str(item) for item in (state.get("warnings") or ())]
    st_module.caption(
        "Warnings: " + ("; ".join(warnings_list) if warnings_list else "none recorded")
    )
    if progress.failed:
        st_module.error(
            "A stage failed; open Resume / Retry for the sanitized evidence."
        )
    st_module.markdown("**Stages (all 16, canonical order)**")
    st_module.dataframe(
        [row.as_row() for row in derive_pipeline_stage_rows(state)],
        width="stretch",
        hide_index=True,
    )
    attempts = list(state.get("attempts") or ())
    if attempts:
        st_module.markdown("**Execution-attempt history** (operational, never identity)")
        # Arrow-safe typed columns (R5-FIX finding 1's fix pattern): numeric
        # columns stay nullable Int64 — no placeholder strings mixed in
        st_module.dataframe(
            pd.DataFrame(
                {
                    "Attempt": pd.array(
                        [attempt.get("attempt_number") for attempt in attempts],
                        dtype="Int64",
                    ),
                    "Started": pd.array(
                        [attempt.get("started_at") for attempt in attempts],
                        dtype="string",
                    ),
                    "Ended": pd.array(
                        [attempt.get("ended_at") or "—" for attempt in attempts],
                        dtype="string",
                    ),
                    "Workers": pd.array(
                        [
                            dict(attempt.get("worker_policy") or {}).get("max_workers")
                            for attempt in attempts
                        ],
                        dtype="Int64",
                    ),
                    "Retry reason": pd.array(
                        [
                            attempt.get("operational_retry_reason") or "—"
                            for attempt in attempts
                        ],
                        dtype="string",
                    ),
                }
            ),
            width="stretch",
            hide_index=True,
        )
    _render_ladder_panel(st_module, roots, state)
    _render_persisted_comparisons(st_module, roots, state)
    publication = dict(state.get("publication") or {})
    st_module.caption(f"Publication state: {publication.get('state', 'not_prepared')}")
    terminal = progress.planned_count == progress.terminal_count
    if not terminal:
        confirm = st_module.checkbox(
            "I understand the run stops at the NEXT stage boundary and "
            "completed semantic results stay immutable/reusable",
            key=f"{_PIPE}cancel_confirm",
        )
        if st_module.button(
            "Request Safe Cancel", key=f"{_PIPE}cancel", disabled=not confirm
        ):
            try:
                request_pipeline_cancel(PIPELINE_STATE_ROOT, pipeline_id)
            except Exception as error:  # noqa: BLE001 — sanitized surface only
                st_module.error(f"Cancel request failed: {sanitize_error(error)}")
            else:
                st_module.success(
                    "Safe-cancel sentinel written; the runner honors it at the "
                    "next stage boundary."
                )


def _ladder_frame(diagnostics: Mapping[str, Any]) -> pd.DataFrame:
    """The ladder table as an Arrow-safe frame (R5-FIX gate finding 1).

    Every column carries an explicit nullable dtype — the planned GAM row's
    missing numbers are ``pd.NA`` inside Int64/Float64 columns, never a
    placeholder string mixed into a numeric column (the exact shape that
    produced the smoke run's 17 Arrow serialization tracebacks)."""

    rows: list[dict[str, Any]] = []
    for protocol_id, rung in sorted(dict(diagnostics.get("rungs") or {}).items()):
        report = dict(rung.get("prediction_report") or {})
        auc = report.get("auc")
        rows.append(
            {
                "Rung": protocol_id,
                "OOS rows": report.get("count", 0),
                "Brier": report.get("brier_score"),
                "Brier skill": report.get("brier_skill_score"),
                "AUC": (
                    f"{auc:.4f}"
                    if isinstance(auc, (int, float))
                    else report.get("auc_reason", "—")
                ),
                "Status": report.get("status", "—"),
            }
        )
    rows.append(
        {
            "Rung": "ifvg_context_gam_v1",
            "OOS rows": None,
            "Brier": None,
            "Brier skill": None,
            "AUC": "—",
            "Status": "planned: preregistered_basis_penalty_protocol_not_ratified",
        }
    )
    return pd.DataFrame(
        {
            "Rung": pd.array([row["Rung"] for row in rows], dtype="string"),
            "OOS rows": pd.array([row["OOS rows"] for row in rows], dtype="Int64"),
            "Brier": pd.array([row["Brier"] for row in rows], dtype="Float64"),
            "Brier skill": pd.array(
                [row["Brier skill"] for row in rows], dtype="Float64"
            ),
            "AUC": pd.array([row["AUC"] for row in rows], dtype="string"),
            "Status": pd.array([row["Status"] for row in rows], dtype="string"),
        }
    )


def _parity_caption(parity: Mapping[str, Any]) -> str:
    """R5-FIX gate finding 5: zero OOS rows ⇒ the parity claim is NOT
    evaluable — never "held over 0 rows"."""

    count = int(parity.get("oos_row_count", 0) or 0)
    if count:
        return (
            f"Identical-rows parity held over {count} OOS rows "
            "(identical rows/labels/folds across every rung)."
        )
    return (
        "Identical-rows parity not evaluable — 0 OOS rows (no out-of-fold "
        "predictions exist on this window; the rungs still shared identical "
        "rows/labels/folds by construction)."
    )


def _render_ladder_panel(st_module, roots: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    """Supervised-ladder result presentation (§35 R5) from the persisted S10
    diagnostics sidecar — planned/blocked entries and S11 stay truthful."""

    from alpha_lab.agents.data_infra.ifvg.ml.decision_policies import (  # noqa: PLC0415
        S11_BLOCKED_REASON,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        load_ladder_diagnostics,
    )

    stages = dict(state.get("stages") or {})
    s10 = dict(stages.get("10_generate_predictions_and_diagnostics") or {})
    with st_module.expander("Model ladder (supervised)"):
        if not s10.get("in_plan"):
            st_module.caption("Not required by this stage plan.")
            return
        diagnostics = load_ladder_diagnostics(Path(roots["store_root"]), state)
        if diagnostics is None:
            render_empty_state(
                st_module,
                "no_model_result",
                detail="the diagnostics stage has not completed for this run",
            )
            return
        st_module.dataframe(_ladder_frame(diagnostics), width="stretch", hide_index=True)
        st_module.caption(_parity_caption(dict(diagnostics.get("parity") or {})))
        st_module.caption(f"S11 (model-gated replays): BLOCKED — {S11_BLOCKED_REASON}.")


def _render_persisted_comparisons(
    st_module, roots: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    """DEV-R4-16 consumption half (adversarial M-2): the UI reads the
    PERSISTED S14 `ComparisonResultEnvelope`s — exact-ID loads, nothing
    rebuilt at render time."""

    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        load_comparison_results_for_search,
    )

    envelopes = load_comparison_results_for_search(Path(roots["store_root"]), state)
    with st_module.expander("Persisted comparison results (S14)"):
        if not envelopes:
            st_module.caption(
                "No persisted comparison results for this run yet (they "
                "publish at stage 14 when baseline↔challenger lineage "
                "evidence exists)."
            )
            return
        rows = []
        for envelope in envelopes:
            payload = envelope.payload
            for kind, report in sorted(dict(payload.delta_reports).items()):
                report = dict(report)
                rows.append(
                    {
                        "Comparison": envelope.comparison_result_id[:12],
                        "Subject": payload.subject.subject_kind,
                        "Entity": kind,
                        "match_basis": report.get("match_basis", "—"),
                        "Jaccard": report.get("jaccard"),
                        "Status": payload.compatibility.compatibility_status,
                    }
                )
        # Arrow-safe typed columns: Jaccard is nullable Float64, never a
        # float/em-dash mix (R5-FIX finding 1's fix pattern)
        st_module.dataframe(
            pd.DataFrame(
                {
                    "Comparison": pd.array(
                        [row["Comparison"] for row in rows], dtype="string"
                    ),
                    "Subject": pd.array(
                        [row["Subject"] for row in rows], dtype="string"
                    ),
                    "Entity": pd.array([row["Entity"] for row in rows], dtype="string"),
                    "match_basis": pd.array(
                        [row["match_basis"] for row in rows], dtype="string"
                    ),
                    "Jaccard": pd.array(
                        [row["Jaccard"] for row in rows], dtype="Float64"
                    ),
                    "Status": pd.array([row["Status"] for row in rows], dtype="string"),
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st_module.caption(
            "Evidence links (lineage-uniqueness reports): "
            + "; ".join(
                ", ".join(link[:12] for link in envelope.payload.evidence_links)
                for envelope in envelopes
            )
        )


def _render_monitor(st_module, roots: Mapping[str, Any]) -> None:
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        list_pipeline_runs,
    )

    st_module.subheader("Monitor")
    runs = list_pipeline_runs(PIPELINE_STATE_ROOT)
    if not runs:
        render_empty_state(st_module, "pipeline_no_runs")
        cli_escape_hatch(
            st_module,
            "python scripts/ifvg_pipeline_job.py status --pipeline-id <pipeline_semantic_id>",
            reason="inspect a pipeline outside this monitor",
        )
        return
    pipeline_id = _selected_run(st_module, runs)
    st_module.button("Refresh", key=f"{_PIPE}refresh", help="Manual poll fallback")
    fragment = getattr(st_module, "fragment", None)
    if callable(fragment):

        @fragment(run_every="5s")
        def _auto_body() -> None:
            _render_monitor_body(st_module, roots=roots, pipeline_id=pipeline_id)

        _auto_body()
    else:
        _render_monitor_body(st_module, roots=roots, pipeline_id=pipeline_id)


def _render_resume(st_module, roots: Mapping[str, Any]) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
        SearchCharterEnvelope,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        read_pipeline_state,
    )
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (  # noqa: PLC0415
        pipeline_entry_key_for_charter,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        load_verified_envelope,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        list_pipeline_runs,
    )

    st_module.subheader("Resume / Retry")
    runs = list_pipeline_runs(PIPELINE_STATE_ROOT)
    if not runs:
        render_empty_state(st_module, "pipeline_no_runs")
        return
    pipeline_id = _selected_run(st_module, runs)
    state = read_pipeline_state(PIPELINE_STATE_ROOT, pipeline_id)
    if state is None:
        render_empty_state(st_module, "artifact_unavailable")
        return
    failed = [
        (stage_value, entry)
        for stage_value, entry in dict(state.get("stages") or {}).items()
        if isinstance(entry, dict) and entry.get("status") == "failed"
    ]
    for stage_value, entry in failed:
        st_module.error(
            f"{stage_value} failed: {sanitize_error(str(entry.get('explanation')))}"
        )
    st_module.caption(
        "Retrying reuses every verified semantic stage result under the SAME "
        "pipeline semantic id (a new operational attempt). Research-bearing "
        "changes require a new pipeline specification — a new semantic id."
    )
    max_workers = st_module.slider(
        "Worker limit for the retry (a resource clone is visibly operational)",
        min_value=1,
        max_value=4,
        value=1,
        key=f"{_PIPE}retry_workers",
    )
    reason = st_module.text_input(
        "Operational retry reason", value="", key=f"{_PIPE}retry_reason"
    )
    if st_module.button("Resume / Retry (new attempt)", key=f"{_PIPE}retry"):
        try:
            semantic_charter_id = str(state.get("search_charter_id"))
            charter = load_verified_envelope(
                Path(roots["store_root"]),
                "charters",
                semantic_charter_id,
                SearchCharterEnvelope,
            )
            entry_key = pipeline_entry_key_for_charter(charter)
            if entry_key is None:
                render_empty_state(st_module, "runner_executor_planned")
                return
            command = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "ifvg_pipeline_job.py"),
                "resume",
                "--pipeline-id",
                pipeline_id,
                "--store-root",
                str(Path(roots["store_root"])),
                "--state-root",
                str(PIPELINE_STATE_ROOT),
                "--runner-entry-key",
                entry_key,
                "--max-workers",
                str(int(max_workers)),
            ]
            if reason.strip():
                command += ["--retry-reason", reason.strip()]
            pid = _spawn_pipeline_job(command)
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.error(f"Retry launch failed: {sanitize_error(error)}")
            return
        st_module.success(f"Retry launched detached (pid {pid}).")


def _render_publish(st_module, roots: Mapping[str, Any]) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: PLC0415
        PublicationError,
        activate_pipeline_result,
        read_pipeline_state,
        run_publication_gates,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        list_pipeline_runs,
    )

    st_module.subheader("Publish")
    runs = list_pipeline_runs(PIPELINE_STATE_ROOT)
    if not runs:
        render_empty_state(st_module, "pipeline_no_runs")
        return
    pipeline_id = _selected_run(st_module, runs)
    state = read_pipeline_state(PIPELINE_STATE_ROOT, pipeline_id)
    if state is None:
        render_empty_state(st_module, "artifact_unavailable")
        return
    publication = dict(state.get("publication") or {})
    st_module.caption(
        f"State: **{publication.get('state', 'not_prepared')}** — preparation "
        "never publishes; verification-only artifacts can never activate a "
        "research catalog entry."
    )
    # adversarial m-6: the cache is scoped to THIS pipeline id, so another
    # run's gates can never enable this run's activation control
    gate_cache_key = f"{_PIPE}gate_results_{pipeline_id}"
    if st_module.button("Run Publication Gates", key=f"{_PIPE}gates"):
        try:
            gates = run_publication_gates(
                PIPELINE_STATE_ROOT, pipeline_id, store_root=Path(roots["store_root"])
            )
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.error(f"Publication gates failed to run: {sanitize_error(error)}")
        else:
            st_module.session_state[gate_cache_key] = gates
    gates = st_module.session_state.get(gate_cache_key)
    if isinstance(gates, dict) and gates:
        st_module.dataframe(
            [
                {"Gate": gate, "Result": "✓ pass" if passed else "✕ fail"}
                for gate, passed in gates.items()
            ],
            width="stretch",
            hide_index=True,
        )
    verification_scope = state.get("run_scope") == "verification_5d"
    gates_passed = bool(gates) and all(dict(gates or {}).values())
    if verification_scope:
        st_module.caption(
            "Activation is disabled: verification-only artifacts can never "
            "activate a research catalog entry."
        )
    activate_disabled = verification_scope or not gates_passed
    if st_module.button(
        "Publish and Activate Catalog Entry",
        key=f"{_PIPE}activate",
        disabled=activate_disabled,
    ):
        try:
            result_id = activate_pipeline_result(
                PIPELINE_STATE_ROOT, pipeline_id, store_root=Path(roots["store_root"])
            )
        except PublicationError as error:
            st_module.error(f"Activation refused: {sanitize_error(error)}")
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.error(f"Activation failed: {sanitize_error(error)}")
        else:
            st_module.success(f"Catalog entry activated for result {result_id[:12]}…")


def render_pipeline_run(st_module=st, *, roots: Mapping[str, Any], draft=None) -> None:
    """The complete §30 operator workflow (thin; pure logic in src)."""

    dev_only_badge(st_module)
    if draft is None:
        draft = _active_mode5_draft(st_module, roots)
    pending = st_module.session_state.pop(_PHASE_KEY, None)
    if pending in _PHASES:
        st_module.session_state[f"{_PIPE}phase_radio"] = pending
    sanitize_select(st_module, f"{_PIPE}phase_radio", list(_PHASES))
    phase = st_module.radio(
        "Pipeline phase",
        _PHASES,
        horizontal=True,
        key=f"{_PIPE}phase_radio",
        label_visibility="collapsed",
    )
    fields = None
    if phase in ("Configure", "Preview", "Launch"):
        fields = _render_configure(st_module, roots, draft)
    if phase == "Preview":
        _render_preview(st_module, roots, draft, fields)
    elif phase == "Launch":
        _render_launch(st_module, roots, draft, fields)
    elif phase == "Monitor":
        _render_monitor(st_module, roots)
    elif phase == "Resume / Retry":
        _render_resume(st_module, roots)
    elif phase == "Publish":
        _render_publish(st_module, roots)
    with st_module.expander("MBP-1 Order Flow (research-only offline)"):
        from ifvg_mbp1_panels import render_mbp1_order_flow  # noqa: PLC0415

        default_ids, integrity_note = _mbp1_default_ids(st_module, roots)
        if integrity_note:
            st_module.error(integrity_note)
        render_mbp1_order_flow(st_module, roots=roots, default_ids=default_ids)
