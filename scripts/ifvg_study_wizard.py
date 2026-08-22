"""New Study — the eight-step wizard (R4; FUX §§6–15).

``render_new_study`` owns drafts (disk-persisted, autosaved on Next, Save
Draft always visible, exact-step restore, immutable after freeze, Clone as
New Search), the five study modes, the objective/baseline/search-space/
prop/risk/benchmark/validation/review steps, and the freeze-and-launch
handoff. Freezing builds a ``SearchCharterPayload``, runs the fail-closed
``validate_charter``, saves it immutably, marks the draft frozen, and — for
charters with a REGISTERED runner entry — spawns the detached job shim and
routes to Active Runs. Launching happens ONLY inside the explicit button
handler; render, import, AppTest, and page refresh launch nothing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st
from ifvg_ui_common import (
    STATE_PREFIX,
    cli_escape_hatch,
    dev_only_badge,
    identity_block,
    render_empty_state,
    sanitize_error,
    sanitize_select,
    status_badge,
    verification_badge,
)

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    PROFILE_CAPABILITY_REGISTRY,
    ProfileCapabilityStatus,
)
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    SyntheticAuthorizationMarker,
    derive_authorization_requirements,
)
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (
    AXIS_VALUE_REGISTRY_V1,
    SEARCH_AXIS_REGISTRY_V1,
)
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    CharterValidationError,
    CostPolicy,
    DatePolicy,
    ObjectivePolicy,
    ResolvedPropGateThresholds,
    ResolvedRobustnessGateThresholds,
    ResolvedStrategyGateThresholds,
    SearchCharterEnvelope,
    SearchCharterPayload,
    SearchMode,
    SimulationProtocol,
    save_charter,
    validate_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonicalize_section,
    name_free_section_hash,
)
from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
    runner_entry_key_for_charter,
)
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    PROPOSED_VERIFICATION_ALLOWLIST,
)
from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    DraftError,
    StudyDraft,
    clone_draft,
    list_drafts,
    load_draft,
    mark_frozen,
    new_draft,
    save_draft,
)
from alpha_lab.agents.data_infra.ifvg.study_presentation import (
    AXIS_GROUP_ORDER,
    INTERPRETATIONS,
    MAX_ACCOUNTS_PER_FIRM,
    OBJECTIVE_TEMPLATES,
    RESEARCH_QUESTIONS,
    RISK_POLICY_TEMPLATES,
    SESSION_DIRECTION_AXES,
    STUDY_MODES,
    WIZARD_STEP_TITLES,
    axis_renders_widget,
    classification_label,
    computation_path_chip,
    enumerate_child_count,
    estimate_search_work,
    format_duration,
    format_storage,
    grouped_axis_keys,
    mode_compatibility_error,
    validate_baseline_step,
    validate_benchmarks_step,
    validate_objective_step,
    validate_prop_step,
    validate_review_step,
    validate_risk_step,
    validate_search_space_step,
    validate_validation_step,
)
from alpha_lab.agents.data_infra.ifvg.study_status import (
    FULL_SCOPE_ACKNOWLEDGEMENT,
    FULL_SCOPE_WARNING_TEXT,
    StudyStatusKey,
)

__all__ = ["render_new_study"]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DRAFT_KEY = f"{STATE_PREFIX}draft_id"
_ACTIVE_DRAFT_KEY = f"{STATE_PREFIX}wz_active_draft"
_W = f"{STATE_PREFIX}wz_"

_INTERPRETATIONS = INTERPRETATIONS
_SESSION_DIRECTION_AXES = SESSION_DIRECTION_AXES

#: Draft-header widget keys that are already instantiated when the shell
#: decides whether to purge stale step-widget state on a draft switch.
_HEADER_WIDGET_KEYS = frozenset(
    {f"{_W}new_mode", f"{_W}new_draft", f"{_W}open_draft", f"{_W}open_btn"}
)


def _spawn_search_job(command: list[str]) -> int:
    """The ONLY DETACHED process launch (tests monkeypatch it).

    The wizard's one other subprocess use is the read-only
    ``git rev-parse HEAD`` provenance query in :func:`_commit_of`
    (DEV-R4-14); nothing else in the UI lane touches ``subprocess``.
    """

    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "DETACHED_PROCESS", 0
    )
    process = subprocess.Popen(  # noqa: S603 — exact interpreter + repo script
        command, cwd=_REPO_ROOT, creationflags=creation_flags
    )
    return process.pid


# ─────────────────────────────────────────────────────────────────────────────
# Draft header
# ─────────────────────────────────────────────────────────────────────────────


def _draft_header(st_module, roots: Mapping[str, Any]) -> StudyDraft | None:
    draft_root = Path(roots["draft_root"])
    drafts = list_drafts(draft_root)
    open_labels = {
        f"{draft.display_name} · {draft.status} · {draft.draft_id[:8]}": draft
        for draft in drafts
    }
    columns = st_module.columns([2, 2, 1])
    with columns[0]:
        mode_label = st_module.selectbox(
            "Study mode for a new draft",
            [mode.label for mode in STUDY_MODES],
            key=f"{_W}new_mode",
            help="The five study modes (FUX §6).",
        )
        if st_module.button("Start new draft", key=f"{_W}new_draft"):
            mode = next(m for m in STUDY_MODES if m.label == mode_label)
            draft = new_draft(mode.mode_id)
            save_draft(draft_root, draft)
            st_module.session_state[_DRAFT_KEY] = draft.draft_id
    with columns[1]:
        options = ["—", *open_labels]
        sanitize_select(st_module, f"{_W}open_draft", options)
        opened = st_module.selectbox(
            "Open a saved draft", options, key=f"{_W}open_draft"
        )
        if opened != "—" and st_module.button("Open", key=f"{_W}open_btn"):
            st_module.session_state[_DRAFT_KEY] = open_labels[opened].draft_id
    with columns[2]:
        st_module.caption(f"{len(drafts)} saved draft(s)")

    draft_id = st_module.session_state.get(_DRAFT_KEY)
    if not draft_id:
        st_module.info(
            "Start a new draft or open a saved one. Drafts autosave on every "
            "successful Next and can always be saved explicitly."
        )
        return None
    try:
        draft = load_draft(draft_root, str(draft_id))
    except DraftError as error:
        st_module.warning(sanitize_error(error))
        st_module.session_state.pop(_DRAFT_KEY, None)
        return None
    if draft.status == "frozen":
        status_badge(st_module, StudyStatusKey.FROZEN)
        identity_block(st_module, "Frozen search charter id", draft.frozen_search_id or "")
        if st_module.button("Clone as New Search", key=f"{_W}clone_frozen"):
            clone = clone_draft(draft)
            save_draft(draft_root, clone)
            st_module.session_state[_DRAFT_KEY] = clone.draft_id
            st_module.rerun()
        return None
    return draft


# ─────────────────────────────────────────────────────────────────────────────
# Step bodies — each renders widgets seeded from the draft and returns the
# collected field mapping used for validation + persistence.
# ─────────────────────────────────────────────────────────────────────────────


def _step_objective(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("objective")
    mode = next((m for m in STUDY_MODES if m.mode_id == draft.mode_id), STUDY_MODES[0])
    st_module.markdown(f"**Study mode:** {mode.label}")
    st_module.caption(mode.description)
    question_labels = list(RESEARCH_QUESTIONS.values())
    stored_question = RESEARCH_QUESTIONS.get(payload.get("question_id", ""), None)
    question_label = st_module.radio(
        "Research question",
        question_labels,
        index=question_labels.index(stored_question)
        if stored_question in question_labels
        else 0,
        key=f"{_W}question",
    )
    question_id = next(
        key for key, label in RESEARCH_QUESTIONS.items() if label == question_label
    )
    template_labels = [template.label for template in OBJECTIVE_TEMPLATES]
    stored_template = payload.get("template_id", "")
    template_label = st_module.selectbox(
        "Objective template",
        template_labels,
        index=next(
            (
                index
                for index, template in enumerate(OBJECTIVE_TEMPLATES)
                if template.template_id == stored_template
            ),
            0,
        ),
        key=f"{_W}template",
    )
    template = next(t for t in OBJECTIVE_TEMPLATES if t.label == template_label)
    st_module.markdown("**Resolved objective (nothing hidden behind the template):**")
    st_module.table(
        {
            "primary objective": [", ".join(template.primary_objective) or "—"],
            "hard constraints": [", ".join(template.hard_constraints)],
            "tie-breaks": [", ".join(template.tie_breaks) or "—"],
            "status": [template.status],
        }
    )
    st_module.caption(
        "The objective policy contains no hidden weighted score; resolved "
        "thresholds are shown field-by-field in the Benchmarks step."
    )
    custom_objectives: tuple[str, ...] = tuple(payload.get("custom_objectives") or ())
    if template.template_id == "custom":
        from alpha_lab.agents.data_infra.ifvg.search.charter import (  # noqa: PLC0415
            OBJECTIVE_DIRECTIONS,
        )

        custom_objectives = tuple(
            st_module.multiselect(
                "Custom registered objective metrics",
                sorted(OBJECTIVE_DIRECTIONS),
                default=list(custom_objectives),
                key=f"{_W}custom_objectives",
            )
        )
    incompatibility = mode_compatibility_error(
        draft.mode_id, question_id, template.template_id
    )
    if incompatibility:
        st_module.error(incompatibility)
    return {
        "mode_id": draft.mode_id,
        "question_id": question_id,
        "template_id": template.template_id,
        "custom_objectives": custom_objectives,
    }


def _step_baseline(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("baseline")
    names = list(PROFILE_CAPABILITY_REGISTRY)
    stored = payload.get("baseline_profile_name") or names[0]
    selected = st_module.selectbox(
        "Baseline profile",
        names,
        index=names.index(stored) if stored in names else 0,
        key=f"{_W}baseline",
        help=(
            "Only a capability-eligible baseline can anchor a charter. "
            "Generated children are baseline diffs gated by their own "
            "GeneratedProfileCapability — they are never presented as fixed "
            "M0–M3 profiles."
        ),
    )
    capability = PROFILE_CAPABILITY_REGISTRY[selected]
    blocked_reason: str | None = None
    if capability.status is not ProfileCapabilityStatus.RUNNABLE:
        blocked_reason = f"{capability.status.value}: {capability.reason}"
        st_module.error(
            f"This baseline is not runnable — {blocked_reason}. Blocked, "
            "experimental, superseded, or unavailable profiles render their "
            "reason and no launch path exists (FUX §9)."
        )
    card: dict[str, str] = {"Profile": selected, "Status": capability.status.value}
    section_hash = ""
    name_free_hash = ""
    try:
        resolved = resolve_profile_config({"profile_name": selected})
        section = canonicalize_section(resolved.section)
        from strategy_core.strategies.ifvg_smc.section import (  # noqa: PLC0415
            ifvg_profile_hash,
        )

        section_hash = ifvg_profile_hash(section)
        name_free_hash = name_free_section_hash(section)
        direction = []
        if getattr(section, "enable_longs", False):
            direction.append("long")
        if getattr(section, "enable_shorts", False):
            direction.append("short")
        card.update(
            {
                "Entry thesis": ", ".join(
                    getattr(section, "entry_families", ()) or ()
                )
                or str(getattr(section, "entry_family", "—")),
                "Direction": "/".join(direction) or "—",
                "Target / label family": str(getattr(section, "label_family", "—")),
                "FSM concurrency": "one active setup / one active trade",
                "Development date range": (
                    "through 2026-06-10 21:00Z (development cutoff)"
                ),
            }
        )
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Baseline resolution failed: {sanitize_error(error)}")
        blocked_reason = blocked_reason or "baseline section did not resolve"
    st_module.markdown("**Baseline card**")
    st_module.table({key: [value] for key, value in card.items()})
    with st_module.expander("View Technical Identity"):
        identity_block(st_module, "Profile hash (resolved section)", section_hash)
        identity_block(st_module, "Name-free section hash", name_free_hash)
        st_module.caption(
            "Core Replay ID / artifact ids appear per child after launch; "
            "resolved Strategy-Core and Quant-Lab source identities are "
            "computed at freeze time and ride the replay identity. Seed and "
            "date-policy references are shown in the Validation step."
        )
    return {
        "baseline_profile_name": selected,
        "baseline_section_config_hash": section_hash,
        "baseline_blocked_reason": blocked_reason,
    }


def _axis_card(st_module, axis_key: str, payload: Mapping[str, Any]) -> tuple[str, ...]:
    spec = SEARCH_AXIS_REGISTRY_V1[axis_key]
    label = classification_label(spec.classification)
    chip = computation_path_chip(spec)
    st_module.markdown(
        f"**{spec.human_label}**  \n"
        f"`{spec.technical_key}` · {label} · {chip}"
    )
    st_module.caption(spec.description)
    baseline_value = AXIS_VALUE_REGISTRY_V1.get(spec.baseline_value_id)
    baseline_label = (
        baseline_value.human_label if baseline_value is not None else spec.baseline_value_id
    )
    selected: tuple[str, ...] = ()
    if axis_renders_widget(spec):
        options = {
            AXIS_VALUE_REGISTRY_V1[value_id].human_label: value_id
            for value_id in spec.registered_values
            if value_id in AXIS_VALUE_REGISTRY_V1
        }
        stored = tuple(payload.get(axis_key) or ())
        default_labels = [
            label for label, value_id in options.items() if value_id in stored
        ]
        chosen = st_module.multiselect(
            f"Registered search values — baseline: {baseline_label}",
            list(options),
            default=default_labels,
            key=f"{_W}axis_{axis_key}",
            help=(
                "Values come from the typed axis/value registry only; no "
                "free-form overrides exist anywhere in this workspace."
            ),
        )
        selected = tuple(options[label] for label in chosen)
    else:
        st_module.caption(
            f"Value: **{baseline_label}** — no input widget "
            f"({label.lower()}; visible for audit)"
        )
    with st_module.expander("Evidence & authorization"):
        st_module.write(f"Baseline value: `{spec.baseline_value_id}`")
        st_module.write(
            "Registered values: "
            + (", ".join(f"`{value}`" for value in spec.registered_values) or "—")
        )
        rows = []
        for value_id in spec.registered_values:
            value = AXIS_VALUE_REGISTRY_V1.get(value_id)
            if value is None:
                continue
            rows.append(
                {
                    "value": value_id,
                    "capability": str(getattr(value, "capability_status", "—")),
                    "owner ratification": str(
                        getattr(value, "owner_ratification_status", "—")
                    ),
                    "evidence ref": str(
                        getattr(value, "ratification_evidence_ref", None) or "missing"
                    ),
                }
            )
        if rows:
            st_module.dataframe(rows, width="stretch", hide_index=True)
        st_module.write(
            f"Artifact / profile impact: "
            f"{getattr(spec, 'expected_artifact_effect', '—')}; "
            f"full sequential replay required: "
            f"{'yes' if spec.requires_full_sequential_replay else 'no'}"
        )
    return selected


def _step_search_space(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("search_space")
    stored_selections: dict[str, tuple[str, ...]] = {
        key: tuple(values)
        for key, values in (payload.get("axis_selections") or {}).items()
    }
    selections: dict[str, tuple[str, ...]] = {}
    grouped = grouped_axis_keys()
    for group in AXIS_GROUP_ORDER:
        keys = grouped.get(group, ())
        if not keys:
            continue
        with st_module.expander(
            group, expanded=group in ("Staleness", "Parent Handling")
        ):
            for axis_key in keys:
                chosen = _axis_card(st_module, axis_key, stored_selections)
                if chosen:
                    selections[axis_key] = chosen
    session_changed = sorted(set(selections) & _SESSION_DIRECTION_AXES)
    interpretation = payload.get("interpretation") or _INTERPRETATIONS[2]
    if session_changed:
        st_module.markdown("**Session/direction interpretation** (FUX §10.5)")
        interpretation = st_module.radio(
            f"Changed session/direction axes {session_changed} are interpreted as",
            _INTERPRETATIONS,
            index=_INTERPRETATIONS.index(interpretation)
            if interpretation in _INTERPRETATIONS
            else 2,
            key=f"{_W}interpretation",
            help=(
                "Only 'Sequential Strategy Profile' creates an executable "
                "counterfactual child; the other interpretations are "
                "descriptive and never replay."
            ),
        )
    count = enumerate_child_count(selections) if selections else 1
    st_module.caption(
        f"Enumerated child combinations (baseline included per axis at "
        f"launch): ~{count}"
    )
    return {
        "mode_id": draft.mode_id,
        "axis_selections": selections,
        "interpretation": interpretation,
    }


def _contract_cards(
    st_module, roots: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
        load_contract_summaries,
    )

    try:
        return load_contract_summaries(Path(roots["store_root"]))
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.warning(f"Contract listing failed: {sanitize_error(error)}")
        return ()


def _step_prop(st_module, draft: StudyDraft, roots: Mapping[str, Any]) -> dict[str, Any]:
    payload = draft.step_payload("prop_contracts")
    mode = next((m for m in STUDY_MODES if m.mode_id == draft.mode_id), STUDY_MODES[0])
    cards = _contract_cards(st_module, roots)
    if not cards:
        render_empty_state(st_module, "no_verified_firm_contract")
        if not mode.requires_prop_selection:
            st_module.caption(
                "This study mode does not require a prop contract; continue "
                "to Risk Policies."
            )
        return {
            "mode_id": draft.mode_id,
            "selected_contract_ids": (),
            "launchable_contract_ids": (),
        }
    stored = set(payload.get("selected_contract_ids") or ())
    selected: list[str] = []
    launchable: list[str] = []
    for card in cards:
        contract_id = card["firm_contract_id"]
        title = (
            f"{card['firm']} · {card['account_type']} · "
            f"{card['account_size_label']}"
        )
        with st_module.expander(title, expanded=False):
            synthetic = card["verification_status"] == "synthetic_fixture_verified"
            if synthetic:
                st_module.warning(
                    "SYNTHETIC fixture contract — proves the machinery; it "
                    "can never enter a real study.",
                    icon="🧪",
                )
            st_module.table(
                {
                    "contract version": [card["contract_version"]],
                    "effective date": [card["effective_date"]],
                    "verification status": [card["verification_status"]],
                    "drawdown rule": [card["drawdown_rule"]],
                    "daily rule": [card["daily_rule"]],
                    "contract limits": [card["contract_limits"]],
                    "payout rules": [card["payout_rules"]],
                    "fees": [card["fees"]],
                    "post-payout behavior": [card["post_payout"]],
                    "source/evidence status": [card["source_status"]],
                    "min path capabilities": [card["path_capabilities"]],
                }
            )
            with st_module.expander("Audit: evidence bundle & supersession"):
                identity_block(st_module, "firm_contract_id", contract_id)
                identity_block(
                    st_module,
                    "contract_evidence_bundle_id",
                    card.get("contract_evidence_bundle_id", ""),
                )
                st_module.write(
                    f"Supersession: {card.get('supersession', 'not superseded')}"
                )
            st_module.caption(
                "Assumed intrabar path scenarios are NOT part of the firm "
                "contract; they are selected under the simulation policy "
                "(Risk Policies / Validation)."
            )
            eligible = card["launchable"]
            if eligible:
                launchable.append(contract_id)
                checked = st_module.checkbox(
                    "Select this contract",
                    value=contract_id in stored,
                    key=f"{_W}contract_{contract_id[:16]}",
                )
                if checked:
                    selected.append(contract_id)
            else:
                st_module.caption(
                    f"Not launchable: {card.get('launch_block_reason', '—')} "
                    "(no selection control is rendered)"
                )
    return {
        "mode_id": draft.mode_id,
        "selected_contract_ids": tuple(selected),
        "launchable_contract_ids": tuple(launchable),
    }


def _step_risk(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("risk_policies")
    selected_contracts = tuple(
        draft.step_payload("prop_contracts").get("selected_contract_ids") or ()
    )
    st_module.markdown("**Universal Strategy Profile**")
    if selected_contracts:
        tree = "\n".join(
            f"    ├── Firm {contract[:8]}… Account/Risk/Withdrawal/Replacement "
            "Policy Set"
            for contract in selected_contracts
        )
        st_module.code("Universal Strategy Profile\n" + tree, language="text")
    else:
        st_module.caption(
            "No firm contracts selected — strategy-only study; firm-specific "
            "policies never alter strategy identity."
        )
    per_firm: dict[str, dict[str, Any]] = dict(payload.get("per_firm_policies") or {})
    for contract in selected_contracts:
        with st_module.expander(f"Firm {contract[:12]}… policy set", expanded=True):
            stored = per_firm.get(contract, {})
            template = st_module.selectbox(
                "Risk template",
                RISK_POLICY_TEMPLATES,
                index=RISK_POLICY_TEMPLATES.index(stored.get("risk_template"))
                if stored.get("risk_template") in RISK_POLICY_TEMPLATES
                else 0,
                key=f"{_W}risk_{contract[:16]}",
            )
            risk_value = st_module.number_input(
                "Risk parameter (USD or % per the template)",
                min_value=0.0,
                value=float(stored.get("risk_value", 100.0)),
                key=f"{_W}riskv_{contract[:16]}",
                help=(
                    "Every result-changing account policy is visible here "
                    "and enters the resolved simulation identity. UI caps "
                    "follow OWNER_DECISIONS.md without implying research "
                    "ratification."
                ),
            )
            withdrawal = st_module.selectbox(
                "Withdrawal behavior (trader policy — separate from the "
                "firm's permitted payout contract)",
                (
                    "withdraw_at_every_eligibility_v1",
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                ),
                index=0
                if stored.get("withdrawal_policy")
                not in (
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                )
                else (
                    "withdraw_at_every_eligibility_v1",
                    "withdraw_monthly_cadence_v1",
                    "accumulate_no_withdrawal_v1",
                ).index(stored.get("withdrawal_policy")),
                key=f"{_W}wd_{contract[:16]}",
            )
            replacement = st_module.selectbox(
                "Replacement policy (separate scenario; owner decision 12)",
                ("none", "replace_once", "replace_twice"),
                index=("none", "replace_once", "replace_twice").index(
                    stored.get("replacement_policy", "none")
                ),
                key=f"{_W}rep_{contract[:16]}",
            )
            n_accounts = st_module.number_input(
                f"Accounts per firm (max {MAX_ACCOUNTS_PER_FIRM}, owner decision 13)",
                min_value=1,
                max_value=MAX_ACCOUNTS_PER_FIRM,
                value=int(stored.get("n_accounts", 1)),
                key=f"{_W}acct_{contract[:16]}",
            )
            if int(n_accounts) > 1:
                st_module.info(
                    "Copied accounts visibly share the SAME resampled "
                    "market/trade path per simulation path — no per-account "
                    "resampling exists."
                )
            if risk_value <= 0:
                st_module.warning(
                    "A zero risk parameter means every trade is skipped "
                    "(typed skip reason), not silently resized."
                )
            per_firm[contract] = {
                "risk_template": template,
                "risk_value": float(risk_value),
                "withdrawal_policy": withdrawal,
                "replacement_policy": replacement,
                "n_accounts": int(n_accounts),
            }
    per_firm = {
        contract: policy
        for contract, policy in per_firm.items()
        if contract in selected_contracts
    }
    return {
        "selected_contract_ids": selected_contracts,
        "per_firm_policies": per_firm,
    }


_STRATEGY_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("min_executed_trades", "Minimum executed trades", "trades"),
    ("min_independent_days", "Minimum independent trading days", "days"),
    ("min_net_expectancy_r", "Minimum net expectancy", "R"),
    ("min_profit_factor", "Minimum profit factor", "ratio"),
    ("max_drawdown_r", "Maximum drawdown", "R"),
    ("max_time_under_water_days", "Maximum time under water", "days"),
    ("min_session_stability_score", "Minimum session stability", "score 0–1"),
    ("min_time_block_sign_consistency", "Minimum time-block sign consistency", "share"),
    ("max_top_day_pnl_share", "Maximum single-day PnL share", "share"),
    ("max_top_setup_pnl_share", "Maximum single-setup PnL share", "share"),
    ("require_bootstrap_ci_excludes_zero", "Bootstrap CI must exclude zero", "flag"),
)

_PROP_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "minimum_first_payout_probability_60d",
        "Minimum P(first payout within 60 days)",
        "probability",
    ),
    (
        "maximum_breach_probability_90d",
        "Maximum P(breach within 90 days)",
        "probability",
    ),
    (
        "minimum_expected_net_payout_90d",
        "Minimum expected 90-day net payout",
        "USD",
    ),
    ("minimum_p10_net_payout_90d", "Minimum P10 90-day net payout", "USD"),
    (
        "maximum_p90_payout_drought_days",
        "Maximum P90 payout drought",
        "days",
    ),
    (
        "minimum_three_payout_probability",
        "Minimum P(three payouts)",
        "probability",
    ),
)

_ROBUSTNESS_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "maximum_neighbor_expectancy_degradation_r",
        "Maximum neighbor expectancy degradation",
        "R",
    ),
    ("minimum_plateau_width", "Minimum plateau width", "steps"),
    (
        "maximum_worst_firm_breach_probability_90d",
        "Maximum worst-firm breach probability",
        "probability",
    ),
    (
        "minimum_time_block_sign_consistency",
        "Minimum time-block sign consistency",
        "share",
    ),
    (
        "minimum_outer_fold_recurrence",
        "Minimum outer-fold recurrence (schema-reserved)",
        "share",
    ),
)


def _gate_group(
    st_module,
    *,
    title: str,
    rows: tuple[tuple[str, str, str], ...],
    defaults: Mapping[str, Any],
    stored: Mapping[str, Any],
    key_prefix: str,
) -> dict[str, Any]:
    st_module.markdown(f"**{title}**")
    values: dict[str, Any] = {}
    for name, human, unit in rows:
        default = stored.get(name, defaults.get(name))
        columns = st_module.columns([3, 2, 2])
        with columns[0]:
            st_module.write(f"{human}  \n`{name}` · {unit}")
        with columns[1]:
            if isinstance(default, bool):
                value: Any = st_module.checkbox(
                    "required",
                    value=bool(default),
                    key=f"{key_prefix}{name}",
                    label_visibility="visible",
                )
            elif default is None:
                st_module.caption("not required (no threshold)")
                value = None
            else:
                value = st_module.number_input(
                    "resolved value",
                    value=float(default),
                    key=f"{key_prefix}{name}",
                    label_visibility="collapsed",
                )
        with columns[2]:
            st_module.caption(
                "status: proposed_protocol_default"
                if default is not None
                else "status: not required"
            )
        values[name] = value
    return values


def _step_benchmarks(st_module, draft: StudyDraft) -> dict[str, Any]:
    payload = draft.step_payload("benchmarks")
    strategy_defaults = ResolvedStrategyGateThresholds(
        min_session_stability_score=0.5
    ).model_dump()
    prop_defaults = ResolvedPropGateThresholds(
        minimum_first_payout_probability_60d=0.5,
        maximum_breach_probability_90d=0.35,
        minimum_expected_net_payout_90d=None,
        minimum_p10_net_payout_90d=None,
        maximum_p90_payout_drought_days=None,
        minimum_three_payout_probability=None,
    ).model_dump()
    robustness_defaults = ResolvedRobustnessGateThresholds(
        maximum_neighbor_expectancy_degradation_r=None,
        minimum_plateau_width=None,
        maximum_worst_firm_breach_probability_90d=None,
        minimum_time_block_sign_consistency=None,
    ).model_dump()
    strategy = _gate_group(
        st_module,
        title="1 · Underlying Strategy Gate",
        rows=_STRATEGY_GATE_ROWS,
        defaults=strategy_defaults,
        stored=payload.get("strategy_gates") or {},
        key_prefix=f"{_W}sg_",
    )
    prop = _gate_group(
        st_module,
        title="2 · Prop Feasibility Gate",
        rows=_PROP_GATE_ROWS,
        defaults=prop_defaults,
        stored=payload.get("prop_gates") or {},
        key_prefix=f"{_W}pg_",
    )
    robustness = _gate_group(
        st_module,
        title="3 · Robustness Gate",
        rows=_ROBUSTNESS_GATE_ROWS,
        defaults=robustness_defaults,
        stored=payload.get("robustness_gates") or {},
        key_prefix=f"{_W}rg_",
    )
    st_module.info(
        "Why configurations fail here: each gate row explains its own "
        "failure; later gates that were not run display the exact earlier "
        "gate that stopped them (e.g. 'Not run — strategy gate failed'). "
        "Verification fixtures use verification_control_flow_gates_v1, "
        "NEVER these research thresholds. Frontier ranking happens only "
        "after the feasibility gates, and no hidden weighted score exists."
    )
    return {
        "strategy_gates": strategy,
        "prop_gates": prop,
        "robustness_gates": robustness,
    }


def _step_validation(
    st_module, draft: StudyDraft, roots: Mapping[str, Any]
) -> dict[str, Any]:
    payload = draft.step_payload("validation")
    scope_labels = {
        "verification_5d": "Verification Fixture — maximum five authorized real trading days",
        "full_authorized_development": (
            "Full Authorized Development Data — explicit post-acceptance "
            "operator action"
        ),
    }
    stored_scope = payload.get("run_scope", "verification_5d")
    scope_label = st_module.radio(
        "Run scope",
        list(scope_labels.values()),
        index=0 if stored_scope != "full_authorized_development" else 1,
        key=f"{_W}scope",
    )
    run_scope = next(
        key for key, label in scope_labels.items() if label == scope_label
    )
    if run_scope == "verification_5d":
        verification_badge(st_module)
        allowlist = tuple(PROPOSED_VERIFICATION_ALLOWLIST)
        st_module.write(
            "One canonical program-wide allowlist (read-only; owner decision "
            "21/R-5 — the proposed candidate pending coverage sign-off):"
        )
        st_module.code("\n".join(allowlist), language="text")
        st_module.download_button(
            "Download resolved allowlist",
            data=json.dumps({"allowlist": allowlist}, indent=2),
            file_name="verification_allowlist_v1.json",
            key=f"{_W}dl_allowlist",
        )
        st_module.caption(
            "Research interpretation and catalog activation are prohibited "
            "for verification runs."
        )
        real_dates = allowlist
        warmup_dates: tuple[str, ...] = ()
        st_module.caption(
            f"Real-date count: {len(real_dates)} · warmup: 0 real days "
            "(the profile-matching precomputed seed snapshot replaces real "
            "warmup; warmup + evidence ≤ 5 holds)"
        )
        # FUX §14.2: show WHETHER the exact VerificationAuthorizationRef
        # exists — derived from the repository, never asserted.
        from alpha_lab.agents.data_infra.ifvg.study_providers import (  # noqa: PLC0415
            verification_authorization_state,
        )

        try:
            authorization = verification_authorization_state(
                Path(roots["store_root"])
            )
        except Exception as error:  # noqa: BLE001 — sanitized surface only
            st_module.warning(
                f"Authorization lookup failed: {sanitize_error(error)}"
            )
            authorization = None
        if authorization is not None and authorization.exists:
            st_module.success(
                f"VerificationAuthorizationRef: EXISTS — {authorization.detail}"
            )
            for run_id in authorization.verification_run_ids:
                identity_block(st_module, "verification_run_id", run_id)
        elif authorization is not None:
            render_empty_state(
                st_module,
                "verification_authorization_missing",
                detail=authorization.detail,
            )
    else:
        st_module.warning(
            "Full authorized development scope: no full run starts during "
            "tests, import, startup, or page render. Launch requires a "
            "frozen pipeline specification and the typed second "
            "confirmation in Review & Launch."
        )
        raw = st_module.text_area(
            "Authorized development dates (one ISO date per line; the "
            "complete resolved allowlist is downloadable below)",
            value="\n".join(payload.get("real_dates") or ()),
            key=f"{_W}full_dates",
        )
        real_dates = tuple(line.strip() for line in raw.splitlines() if line.strip())
        st_module.download_button(
            "Download resolved date list",
            data=json.dumps({"dates": real_dates}, indent=2),
            file_name="authorized_development_dates.json",
            key=f"{_W}dl_dates",
        )
        raw_warmup = st_module.text_area(
            "Warmup dates (one ISO date per line)",
            value="\n".join(payload.get("warmup_dates") or ()),
            key=f"{_W}warmup_dates",
        )
        warmup_dates = tuple(
            line.strip() for line in raw_warmup.splitlines() if line.strip()
        )
        st_module.caption(
            f"Real-date count: {len(real_dates)} · warmup: {len(warmup_dates)}"
        )
    st_module.write("Protected buffer and sealed boundary (read-only):")
    st_module.code(
        "protected: 2026-06-11 (never constructed, listed, stat-ed, or read)\n"
        "sealed range: per development access policy — counters must stay 0\n"
        "development cutoff: 2026-06-10T21:00:00Z",
        language="text",
    )
    seed = int(
        st_module.number_input(
            "Deterministic seed",
            value=int(payload.get("seed", 7)),
            step=1,
            key=f"{_W}seed",
        )
    )
    workers = int(
        st_module.number_input(
            "Worker limit (operational — never scientific identity)",
            min_value=1,
            max_value=4,
            value=int(payload.get("worker_limit", 4)),
            key=f"{_W}workers",
        )
    )
    selections_for_estimate = {
        key: tuple(values)
        for key, values in (
            draft.step_payload("search_space").get("axis_selections") or {}
        ).items()
    }
    n_children_estimate = enumerate_child_count(selections_for_estimate)
    estimate = estimate_search_work(
        n_children=n_children_estimate,
        n_replay_days=len(real_dates),
        n_firm_policy_combinations=sum(
            int(policy.get("n_accounts", 1))
            for policy in (
                draft.step_payload("risk_policies").get("per_firm_policies")
                or {}
            ).values()
        ),
    )
    st_module.table(
        {
            "search algorithm": ["deterministic_exhaustive_v1"],
            "outer-fold protocol": [
                "schema-reserved (exploratory lane; owner decision 15)"
            ],
            "block-bootstrap protocol": ["single_day · 10,000 paths · seed 42"],
            "stress paths": ["9 registered scenarios × bootstrap"],
            "estimated runtime": [
                format_duration(estimate.estimated_runtime_seconds)
            ],
            "estimated storage": [
                format_storage(estimate.estimated_storage_mb)
            ],
        }
    )
    prop_selected = tuple(
        draft.step_payload("prop_contracts").get("selected_contract_ids") or ()
    )
    selections = draft.step_payload("search_space").get("axis_selections") or {}
    from alpha_lab.agents.data_infra.ifvg.study.computation_path import (  # noqa: PLC0415
        ComputationPath,
    )

    has_search = bool(selections)
    path = ComputationPath(
        full_strategy_replay=has_search,
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=has_search,
        prop_resimulation=bool(prop_selected),
        bootstrap_resimulation=bool(prop_selected),
        reuse_trade_stream_hash=not has_search,
    )
    dimensions = tuple(
        f"strategy_profile.{axis}" for axis in sorted(selections)
    ) + tuple(f"prop_contract.{contract}" for contract in prop_selected)
    requirements = derive_authorization_requirements(
        run_scope, dimensions, path, (), prop_selected
    )
    st_module.markdown("**Authorization requirement checklist** (computation-path-scoped)")
    for requirement in requirements.payload.requirements:
        # No owner-decision evidence artifact is REGISTERED in this
        # repository (the R-2 evidence workflow is unratified), so every
        # requirement derives to unsatisfied; a real launch fails closed
        # at charter validation until the referenced evidence exists.
        st_module.write(
            f"- `{requirement.decision_key}` — {requirement.reason} "
            "(no registered evidence artifact — the launch fails closed "
            "until one exists)"
        )
    if not requirements.payload.requirements:
        st_module.caption(
            "No owner decisions are required for this synthetic/verification "
            "computation path."
        )
    return {
        "run_scope": run_scope,
        "real_dates": real_dates,
        "warmup_dates": warmup_dates,
        "seed": seed,
        "worker_limit": workers,
    }


def _step_review(
    st_module, draft: StudyDraft, roots: Mapping[str, Any]
) -> dict[str, Any]:
    objective = draft.step_payload("objective")
    search_space = draft.step_payload("search_space")
    risk = draft.step_payload("risk_policies")
    validation = draft.step_payload("validation")
    selections = {
        key: tuple(values)
        for key, values in (search_space.get("axis_selections") or {}).items()
    }
    n_children = enumerate_child_count(selections) if selections else 1
    firm_combos = sum(
        int(policy.get("n_accounts", 1))
        for policy in (risk.get("per_firm_policies") or {}).values()
    )
    estimate = estimate_search_work(
        n_children=n_children,
        n_replay_days=len(tuple(validation.get("real_dates") or ())),
        n_firm_policy_combinations=firm_combos,
        n_stress_scenarios=0,
    )
    baseline_step = draft.step_payload("baseline")
    frozen_axes = sorted(
        key for key in SEARCH_AXIS_REGISTRY_V1 if key not in selections
    )
    st_module.markdown("**Resolved charter preview**")
    st_module.table(
        {
            "study mode": [draft.mode_id],
            "objective template": [objective.get("template_id", "—")],
            "baseline": [baseline_step.get("baseline_profile_name", "—")],
            "changed axes": [", ".join(sorted(selections)) or "none (baseline only)"],
            "frozen dimensions": [
                f"{len(frozen_axes)} axes at their baseline values "
                f"(first: {', '.join(frozen_axes[:4])}, …)"
            ],
            "unique strategy profiles": [str(n_children)],
            "firm/risk/account-policy combinations": [str(firm_combos)],
            "run scope": [validation.get("run_scope", "—")],
            "seed": [str(validation.get("seed", "—"))],
        }
    )
    identity_block(
        st_module,
        "Baseline resolved section hash",
        str(baseline_step.get("baseline_section_config_hash") or ""),
    )
    prop_selected_review = tuple(
        draft.step_payload("prop_contracts").get("selected_contract_ids") or ()
    )
    st_module.markdown("**Owner-authorization checklist (resolved for this path)**")
    if validation.get("run_scope") == "verification_5d" and not prop_selected_review:
        st_module.caption(
            "Verification scope: the single blocking item is the owner's "
            "VerificationAuthorizationRef (decisions 21/R-5) — shown with "
            "its live state on the Validation step."
        )
    else:
        st_module.caption(
            "The computation-path-scoped requirement list (with its live "
            "evidence state) is on the Validation step; every unmet "
            "requirement fails the freeze closed."
        )
    st_module.markdown("**Estimated work — expensive vs cheap, separated**")
    st_module.table(
        {
            "Expensive strategy replays": [
                f"{estimate.full_replay_count} full sequential replays "
                f"(~{format_duration(estimate.estimated_runtime_seconds)})"
            ],
            "Verified reuse hits": [
                "resolved at launch (identity-deduped; no source path is "
                "constructed at preview)"
            ],
            "Cheaper downstream prop simulations": [
                f"{estimate.historical_prop_replays} historical/scenario prop "
                f"replays + {estimate.bootstrap_and_stress_simulations} "
                "bootstrap/stress simulations"
            ],
            "Estimated storage": [format_storage(estimate.estimated_storage_mb)],
            "New artifacts expected": [
                "core replays · memberships · costed evaluations · frontier"
            ],
            "Blocked or planned capabilities": [
                "real executors (R5) · MBP-1 bundle (R5B) · regime lane (R6)"
            ],
        }
    )
    typed = ""
    if validation.get("run_scope") == "full_authorized_development":
        st_module.error(FULL_SCOPE_WARNING_TEXT)
        typed = st_module.text_input(
            f"Type '{FULL_SCOPE_ACKNOWLEDGEMENT}' to enable the launch control",
            value="",
            key=f"{_W}ack",
        )
    return {
        "run_scope": validation.get("run_scope"),
        "typed_acknowledgement": typed,
        "required_acknowledgement": FULL_SCOPE_ACKNOWLEDGEMENT,
        "n_children": n_children,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Charter assembly + freeze/launch handlers
# ─────────────────────────────────────────────────────────────────────────────


def _commit_of(path: Path) -> str:
    """Read-only ``git rev-parse HEAD`` provenance query (DEV-R4-14).

    Returns ``"unknown"`` on failure; :func:`_freeze_and_launch` REFUSES to
    freeze a charter whose provenance resolved to ``"unknown"`` — two
    different source states must never share a charter identity through a
    silent fallback.
    """

    try:
        result = subprocess.run(  # noqa: S603, S607 — read-only git query
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 — surfaced as a freeze refusal
        return "unknown"


def _strategy_core_root() -> Path:
    """The Strategy-Core working tree, resolved from the INSTALLED package.

    The editable install points at the actual checkout (case-exact on every
    filesystem); walking up from ``strategy_core.__file__`` to the first
    directory containing ``.git`` avoids hardcoding a sibling folder name.
    """

    try:
        import strategy_core  # noqa: PLC0415

        candidate = Path(strategy_core.__file__).resolve()
        for parent in candidate.parents:
            if (parent / ".git").exists():
                return parent
    except Exception:  # noqa: BLE001 — fall through to the sibling guess
        pass
    return _REPO_ROOT.parent / "strategy-core"


def _assemble_charter(
    draft: StudyDraft, roots: Mapping[str, Any]
) -> SearchCharterPayload:
    baseline = draft.step_payload("baseline")
    search_space = draft.step_payload("search_space")
    prop = draft.step_payload("prop_contracts")
    risk = draft.step_payload("risk_policies")
    benchmarks = draft.step_payload("benchmarks")
    validation = draft.step_payload("validation")
    objective_step = draft.step_payload("objective")
    from alpha_lab.agents.data_infra.ifvg.search.axis_registry import (  # noqa: PLC0415
        AxisClassification,
        registry_sha256,
    )

    selections = {
        key: tuple(values)
        for key, values in (search_space.get("axis_selections") or {}).items()
    }
    axes: dict[str, tuple[str, ...]] = {}
    for axis_key, values in selections.items():
        spec = SEARCH_AXIS_REGISTRY_V1[axis_key]
        merged = tuple(
            dict.fromkeys((spec.baseline_value_id, *values))
        )  # baseline first, deduped, order-stable
        axes[axis_key] = merged
    template = next(
        (
            template
            for template in OBJECTIVE_TEMPLATES
            if template.template_id == objective_step.get("template_id")
        ),
        OBJECTIVE_TEMPLATES[-1],
    )
    if template.template_id == "custom":
        pareto = tuple(objective_step.get("custom_objectives") or ())
        tie_breaks: tuple[str, ...] = ("core_replay_id",)
    else:
        pareto = template.primary_objective
        tie_breaks = (*template.tie_breaks, "core_replay_id")
    prop_selected = tuple(prop.get("selected_contract_ids") or ())
    if not prop_selected:
        # strategy-only studies keep strategy-owned objectives only
        from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import (  # noqa: PLC0415
            StrategyMetrics,
        )

        pareto = tuple(
            metric for metric in pareto if hasattr(StrategyMetrics, metric)
        ) or ("net_expectancy_r",)
        tie_breaks = tuple(
            metric
            for metric in tie_breaks
            if metric == "core_replay_id" or hasattr(StrategyMetrics, metric)
        )
    strategy_gates = {
        **(benchmarks.get("strategy_gates") or {}),
    }
    strategy_gates.setdefault("min_session_stability_score", 0.5)
    prop_gate_values = benchmarks.get("prop_gates") or {}
    robustness_values = benchmarks.get("robustness_gates") or {}
    run_scope = validation.get("run_scope", "verification_5d")
    date_policy = DatePolicy(
        replay_dates=tuple(validation.get("real_dates") or ()),
        warmup_dates=tuple(validation.get("warmup_dates") or ()),
        access_policy_id=(
            "verification_fixed_allowlist_max5_v1"
            if run_scope == "verification_5d"
            else "development_explicit_dates_before_path_v2"
        ),
    )
    withdrawal_ids = tuple(
        sorted(
            {
                str(policy.get("withdrawal_policy"))
                for policy in (risk.get("per_firm_policies") or {}).values()
                if policy.get("withdrawal_policy")
            }
        )
    )
    # Every result-changing account-policy parameter enters the charter
    # identity: the risk-policy ids are CONTENT hashes of the complete
    # per-firm policy dicts (template + value + accounts + replacement), so
    # two charters differing in any parameter can never share an identity.
    # Real AccountPolicySetEnvelope construction is the R5 pipeline's job
    # (DEV-R4-17); the frozen draft retains the resolved parameters as
    # provenance.
    from alpha_lab.agents.data_infra.ifvg.search.identities import (  # noqa: PLC0415
        canonical_contract_sha256,
    )

    risk_ids = tuple(
        sorted(
            canonical_contract_sha256(
                {
                    "firm_contract_id": contract_id,
                    "risk_template": policy.get("risk_template"),
                    "risk_value": policy.get("risk_value"),
                    "n_accounts": policy.get("n_accounts"),
                    "replacement_policy": policy.get("replacement_policy"),
                }
            )
            for contract_id, policy in (
                risk.get("per_firm_policies") or {}
            ).items()
        )
    )
    # No owner evidence bundle exists yet, so every charter carries the typed
    # synthetic marker: in the verification namespace it freezes normally; in
    # the research namespace save_charter REFUSES it (P0-4 namespace
    # confinement) and the wizard surfaces the sanitized refusal — a real
    # charter becomes freezable only when the owner evidence exists
    # (FUX-WIZ-007 fail-closed).
    owner_authorization: Any = SyntheticAuthorizationMarker()
    measured_only = tuple(
        key
        for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
        if spec.classification is AxisClassification.MEASUREMENT_ONLY
    )
    blocked = tuple(
        key
        for key, spec in SEARCH_AXIS_REGISTRY_V1.items()
        if spec.classification is AxisClassification.BLOCKED
    )
    if draft.mode_id in {mode.value for mode in SearchMode}:
        search_mode = SearchMode(draft.mode_id)
    else:
        # Full Pipeline Run drafts (mode 5) freeze the underlying SEARCH
        # charter their pipeline runs over: a config search when axes were
        # selected, the single baseline configuration otherwise
        # (engineering default, DECISIONS_TAKEN R5).
        search_mode = (
            SearchMode.FSM_CONFIG_SEARCH if axes else SearchMode.SINGLE_CONFIGURATION
        )
    return SearchCharterPayload(
        search_mode=search_mode,
        baseline_profile_name=str(baseline.get("baseline_profile_name")),
        baseline_section_config_hash=str(
            baseline.get("baseline_section_config_hash") or ""
        ),
        axes=axes,
        locked_invariants_registry_sha256=registry_sha256(),
        measured_only_fields=measured_only,
        blocked_capabilities=blocked,
        authorized_firm_contract_ids=prop_selected,
        authorized_risk_policy_ids=risk_ids,
        authorized_withdrawal_policy_ids=withdrawal_ids,
        objective_policy=ObjectivePolicy(
            feasibility_gates=ResolvedStrategyGateThresholds(**strategy_gates),
            prop_feasibility_gates=ResolvedPropGateThresholds(**prop_gate_values)
            if prop_gate_values
            else ResolvedPropGateThresholds(
                minimum_first_payout_probability_60d=0.5,
                maximum_breach_probability_90d=0.35,
                minimum_expected_net_payout_90d=None,
                minimum_p10_net_payout_90d=None,
                maximum_p90_payout_drought_days=None,
                minimum_three_payout_probability=None,
            ),
            robustness_gates=ResolvedRobustnessGateThresholds(**robustness_values)
            if robustness_values
            else ResolvedRobustnessGateThresholds(
                maximum_neighbor_expectancy_degradation_r=None,
                minimum_plateau_width=None,
                maximum_worst_firm_breach_probability_90d=None,
                minimum_time_block_sign_consistency=None,
            ),
            pareto_objectives=pareto,
            lexicographic_tie_breaks=tie_breaks,
        ),
        date_policy=date_policy,
        simulation_protocol=SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="historical_calendar_clock_v1",
        ),
        max_child_count=max(
            int(draft.step_payload("review").get("n_children", 1)), 1
        ),
        seed=int(validation.get("seed", 7)),
        cost_policy=CostPolicy(),
        strategy_core_commit=_commit_of(_strategy_core_root()),
        quant_lab_commit=_commit_of(_REPO_ROOT),
        source_artifact_ids=(),
        owner_authorization=owner_authorization,
    )


def _freeze_and_launch(st_module, draft: StudyDraft, roots: Mapping[str, Any]) -> None:
    """The explicit launch-button handler — the only place work starts."""

    from ifvg_study_tab import request_route  # noqa: PLC0415

    draft_root = Path(roots["draft_root"])
    store_root = Path(roots["store_root"])
    state_root = Path(roots["state_root"])
    try:
        payload = _assemble_charter(draft, roots)
        if "unknown" in (payload.strategy_core_commit, payload.quant_lab_commit):
            # DEV-R4-14: two source states must never share a charter
            # identity through a silent provenance fallback.
            raise CharterValidationError(
                "source-commit provenance could not be resolved (git "
                "unavailable?); freezing is refused rather than stamping "
                "'unknown' into the immutable charter"
            )
        validate_charter(
            payload, as_of_utc=datetime.now(UTC).isoformat(timespec="seconds")
        )
        envelope = SearchCharterEnvelope.from_payload(payload)
        save_charter(store_root, envelope)
    except CharterValidationError as error:
        st_module.error(f"Charter cannot freeze (fail-closed): {sanitize_error(error)}")
        return
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Freeze failed: {sanitize_error(error)}")
        return
    mark_frozen(draft_root, draft, search_id=envelope.search_id)
    identity_block(st_module, "Frozen search charter id", envelope.search_id)
    entry_key = runner_entry_key_for_charter(envelope)
    if entry_key is None:
        render_empty_state(
            st_module,
            "runner_executor_planned",
            detail=(
                "this charter carries real (non-synthetic) authorization "
                "requirements; the R5 pipeline registers the executors"
            ),
        )
        cli_escape_hatch(
            st_module,
            (
                "python scripts/ifvg_search_job.py start --search-id "
                f"{envelope.search_id} --runner-entry-key <registered key>"
            ),
            reason="explicit owner action once an executor is registered",
        )
        return
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "ifvg_search_job.py"),
        "start",
        "--search-id",
        envelope.search_id,
        "--store-root",
        str(store_root),
        "--state-root",
        str(state_root),
        "--runner-entry-key",
        entry_key,
    ]
    try:
        pid = _spawn_search_job(command)
    except Exception as error:  # noqa: BLE001 — sanitized surface only
        st_module.error(f"Detached launch failed: {sanitize_error(error)}")
        cli_escape_hatch(
            st_module,
            " ".join(
                [
                    "python",
                    "scripts/ifvg_search_job.py",
                    "start",
                    "--search-id",
                    envelope.search_id,
                    "--runner-entry-key",
                    entry_key,
                ]
            ),
            reason="the detached spawn failed",
        )
        return
    st_module.success(
        f"Search launched detached (pid {pid}). Routing to Active Runs."
    )
    st_module.session_state[f"{STATE_PREFIX}monitor_search_id"] = envelope.search_id
    request_route(st_module, "active_runs")
    st_module.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Wizard shell
# ─────────────────────────────────────────────────────────────────────────────

_STEP_VALIDATORS = {
    0: validate_objective_step,
    1: validate_baseline_step,
    2: validate_search_space_step,
    3: validate_prop_step,
    4: validate_risk_step,
    5: validate_benchmarks_step,
    6: validate_validation_step,
    7: validate_review_step,
}

_STEP_KEY_BY_INDEX = {
    0: "objective",
    1: "baseline",
    2: "search_space",
    3: "prop_contracts",
    4: "risk_policies",
    5: "benchmarks",
    6: "validation",
    7: "review",
}


def render_new_study(st_module=st, *, roots: Mapping[str, Any]) -> None:
    dev_only_badge(st_module)
    draft = _draft_header(st_module, roots)
    if draft is None:
        return
    # Cross-draft widget hygiene (FUX-WIZ-002 exact restore): step widgets
    # use fixed keys, so on a draft SWITCH every not-yet-instantiated
    # step-widget key is purged — draft B can never inherit draft A's
    # mounted widget values (the header widgets already rendered this run
    # and keep their own state).
    if st_module.session_state.get(_ACTIVE_DRAFT_KEY) != draft.draft_id:
        for key in list(st_module.session_state.keys()):
            if key.startswith(_W) and key not in _HEADER_WIDGET_KEYS:
                del st_module.session_state[key]
        st_module.session_state[_ACTIVE_DRAFT_KEY] = draft.draft_id
    draft_root = Path(roots["draft_root"])
    step = min(max(int(draft.step_index), 0), len(WIZARD_STEP_TITLES) - 1)
    st_module.progress(
        (step + 1) / len(WIZARD_STEP_TITLES),
        text=f"Step {step + 1} of {len(WIZARD_STEP_TITLES)}: "
        f"{WIZARD_STEP_TITLES[step]}",
    )
    st_module.caption(" › ".join(
        f"**{title}**" if index == step else title
        for index, title in enumerate(WIZARD_STEP_TITLES)
    ))
    new_name = st_module.text_input(
        "Draft name", value=draft.display_name, key=f"{_W}name"
    )
    if new_name and new_name != draft.display_name:
        draft.display_name = new_name

    if step == 0:
        fields = _step_objective(st_module, draft)
    elif step == 1:
        fields = _step_baseline(st_module, draft)
    elif step == 2:
        fields = _step_search_space(st_module, draft)
    elif step == 3:
        fields = _step_prop(st_module, draft, roots)
    elif step == 4:
        fields = _step_risk(st_module, draft)
    elif step == 5:
        fields = _step_benchmarks(st_module, draft)
    elif step == 6:
        fields = _step_validation(st_module, draft, roots)
    else:
        fields = _step_review(st_module, draft, roots)

    errors = _STEP_VALIDATORS[step](fields)
    for field, message in errors.items():
        st_module.error(f"{field}: {message}")

    columns = st_module.columns([1, 1, 1, 2])
    with columns[0]:
        if st_module.button("Save Draft", key=f"{_W}save"):
            draft.steps[_STEP_KEY_BY_INDEX[step]] = dict(fields)
            save_draft(draft_root, draft)
            st_module.success("Draft saved.")
    with columns[1]:
        if step > 0 and st_module.button("Back", key=f"{_W}back"):
            # Back never discards valid data: persist the visible fields.
            draft.steps[_STEP_KEY_BY_INDEX[step]] = dict(fields)
            draft.step_index = step - 1
            save_draft(draft_root, draft)
            st_module.rerun()
    with columns[2]:
        if step < len(WIZARD_STEP_TITLES) - 1 and st_module.button(
            "Next", key=f"{_W}next", disabled=bool(errors)
        ):
            draft.steps[_STEP_KEY_BY_INDEX[step]] = dict(fields)
            draft.step_index = step + 1
            save_draft(draft_root, draft)  # autosave on Next (FUX-WIZ-002)
            st_module.rerun()
    with columns[3]:
        if step == len(WIZARD_STEP_TITLES) - 1:
            mode = next(
                (m for m in STUDY_MODES if m.mode_id == draft.mode_id),
                STUDY_MODES[0],
            )
            if mode.search_mode is None:
                st_module.caption(
                    "Full Pipeline Run drafts freeze and launch from the "
                    "operator workflow below (Configure / Preview / Launch / "
                    "Monitor / Resume-Retry / Publish)."
                )
            elif st_module.button(
                "Freeze Search Charter and Launch",
                key=f"{_W}freeze",
                disabled=bool(errors),
                type="primary",
            ):
                draft.steps[_STEP_KEY_BY_INDEX[step]] = dict(fields)
                save_draft(draft_root, draft)
                _freeze_and_launch(st_module, draft, roots)
    if step == len(WIZARD_STEP_TITLES) - 1:
        mode = next(
            (m for m in STUDY_MODES if m.mode_id == draft.mode_id),
            STUDY_MODES[0],
        )
        if mode.search_mode is None:
            # R5: the standardized §30 operator workflow replaces the R4-era
            # planned-capability state; the draft's visible fields persist
            # before the surface renders so Launch assembles the same state.
            draft.steps[_STEP_KEY_BY_INDEX[step]] = dict(fields)
            save_draft(draft_root, draft)
            from ifvg_pipeline_tab import render_pipeline_run  # noqa: PLC0415

            render_pipeline_run(st_module, roots=roots, draft=draft)
