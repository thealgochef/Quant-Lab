"""Guided study presentation over the existing draft, validation and launch seams."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import ifvg_study_wizard as w
from ifvg_ui_common import SESSION_DRAFT_KEY, STATE_PREFIX

from alpha_lab.agents.data_infra.ifvg.presentation.workspace import (
    QUESTIONS,
    axis_value_name,
    date_scope,
    human_name,
    profile_name,
)

_DRAFT = f"{STATE_PREFIX}draft_id"
_PREFIX = "ifvg_research_wizard_"


def _pick(st, label, values, stored, key, formatter=lambda value: value):
    values = list(values)
    if st.session_state.get(key) not in values:
        st.session_state.pop(key, None)
    return st.selectbox(
        label,
        values,
        index=values.index(stored) if stored in values else 0,
        format_func=formatter,
        key=key,
    )


def _choose_study(st, roots):
    from ifvg_study_tab import TASK_CARDS, start_draft_from_card, stash_session_draft

    st.header("New study")
    basic = {"Evaluate": "evaluate_one", "Compare": "compare_with_baseline", "Search": "fsm_search"}
    descriptions = {
        "Evaluate": "Measure how one strategy configuration performed.",
        "Compare": "Compare a challenger with the baseline over compatible evidence.",
        "Search": "Explore strategy settings and their consistency.",
    }
    selected = st.radio("What would you like to research?", tuple(basic), horizontal=True)
    st.write(descriptions[selected])
    card_id = basic[selected]
    with st.expander("More study types"):
        more = st.selectbox(
            "Additional research",
            (
                "Use selection above",
                "Feature and model study",
                "Prop feasibility",
                "Strategy across firms",
                "Full workflow",
                "Context feature study",
            ),
        )
        if more != "Use selection above":
            card_id = {
                "Feature and model study": "feature_model_evidence",
                "Prop feasibility": "prop_feasibility",
                "Strategy across firms": "prop_feasibility",
                "Full workflow": "advanced_end_to_end",
                "Context feature study": "context",
            }[more]
    if st.button("Configure study", type="primary"):
        if card_id == "context":
            st.session_state["ifvg_workspace_screen"] = "context"
            st.rerun()
        if card_id == "feature_model_evidence":
            for key in list(st.session_state):
                if key.startswith("ifvg_research_group_"):
                    del st.session_state[key]
            st.session_state.pop("ifvg_research_seed", None)
            st.session_state["ifvg_workspace_screen"] = "research_new"
            st.rerun()
        card = next(card for card in TASK_CARDS if card.card_id == card_id)
        draft = start_draft_from_card(card, roots)
        draft.display_name = (
            f"{more if more != 'Use selection above' else selected} study — {date.today()}"
        )
        if more == "Strategy across firms":
            draft.mode_id = "universal_prop_search"
            draft.steps["objective"]["mode_id"] = draft.mode_id
            draft.steps["objective"]["question_id"] = "one_config_across_firms"
        stash_session_draft(st, draft)
        st.session_state[_DRAFT] = draft.draft_id
        st.rerun()


def render_new_study(st, *, roots):
    draft_id = st.session_state.get(_DRAFT)
    if not draft_id:
        _choose_study(st, roots)
        return
    root = Path(roots["draft_root"])
    draft = w._session_draft(st)
    if draft is None or draft.draft_id != draft_id or w._draft_is_persisted(root, draft):
        try:
            draft = w.load_draft(root, draft_id)
        except Exception:
            st.error(
                "This saved draft could not be read. Return to My studies to choose another study."
            )
            return
    if draft.archived or draft.status == "frozen":
        st.info("This study is saved as history. Return to My studies to restore or clone it.")
        return
    if st.session_state.get(_PREFIX + "active") != draft_id:
        for key in list(st.session_state):
            if key.startswith(
                (
                    _PREFIX,
                    "ifvg_research_pipeline_",
                    "ifvg_research_regime_",
                    "ifvg_research_owner_",
                    "ifvg_research_required_",
                )
            ):
                del st.session_state[key]
        st.session_state[_PREFIX + "active"] = draft_id
    try:
        resolution = w._resolve_for_draft(draft, roots)
    except Exception:
        st.error(
            "Study readiness could not be checked. Your draft is retained; refresh "
            "after the evidence is restored."
        )
        return
    if resolution.resolved is None:
        st.warning(
            "Scope unresolved. Choose the intended research scope before continuing "
            "this legacy draft."
        )
        scope = st.selectbox("Research scope", ("Focused research", "Full development workflow"))
        if st.button("Confirm research scope"):
            purpose = (
                w.RunPurpose.DEVELOPMENT_RESEARCH
                if scope == "Focused research"
                else w.RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
            )
            draft.purpose_annotation = w.RunPurposeAnnotation(
                purpose=purpose,
                derivation="owner_confirmed",
                owner_confirmed=True,
                updated_at=w._utc_now(),
            ).to_dict()
            draft.step_payload("validation")["run_scope"] = w.PURPOSE_RUN_SCOPE[purpose]
            draft.step_payload("validation")["evidence_class"] = "real"
            w.save_draft(root, draft)
            st.rerun()
        return
    if resolution.resolved.purpose is w.RunPurpose.IMPLEMENTATION_VERIFICATION:
        st.warning(
            "This is an implementation verification draft. It is available in the "
            "Developer area when enabled at startup."
        )
        return
    flow = resolution.flow
    if flow is None:
        st.warning(
            "The research question could not be resolved. Choose a compatible "
            "question before running."
        )
        return
    position = w.restore_step_index(
        flow, stored_step_key=draft.current_step_key, legacy_step_index=draft.step_index
    )
    step = flow.steps[position]
    persisted = w._draft_is_persisted(root, draft)
    st.header("Configure study")
    question = QUESTIONS.get(
        draft.steps.get("objective", {}).get("question_id"), "Configure this research study."
    )
    st.caption(f"{question} · {'Saved' if persisted else 'Not saved yet'}")
    st.progress(
        (position + 1) / len(flow.steps),
        text=f"Step {position + 1} of {len(flow.steps)}: {step.title}",
    )
    previous_name = draft.display_name
    draft.display_name = st.text_input(
        "Study name", value=human_name(draft.display_name), key=_PREFIX + "name"
    ).strip()
    fields = render_step(st, draft, step.key, resolution)
    if step.key == "review":
        for setting in ("pipeline_settings", "research_settings"):
            if setting in draft.steps.get("review", {}):
                fields[setting] = draft.steps["review"][setting]
    errors = w._STEP_VALIDATORS_BY_KEY[step.key](fields)
    # Validation remains single-sourced; only human field labels are displayed.
    for field, message in errors.items():
        st.error(_validation_message(step.key, field, message))
    changed = (
        not w._fields_equal(draft.steps.get(step.key), fields)
        or previous_name != draft.display_name
    )
    draft.steps[step.key] = dict(fields)
    draft.current_step_key = step.key
    draft.step_index = position
    name_valid = (
        bool(draft.display_name) and human_name(draft.display_name, "") == draft.display_name
    )
    if not name_valid:
        st.error("Use a descriptive study name without technical identifiers or paths.")
    if persisted and changed and name_valid:
        try:
            w.save_draft(root, draft)
            st.caption("Autosaved")
        except Exception:
            st.error("Changes could not be saved. Keep this page open and try Save draft again.")
    elif not persisted:
        w._stash_session_draft(st, draft)
        try:
            duplicates = w.find_duplicate_drafts(
                w.list_drafts(root),
                mode_id=draft.mode_id,
                question_id=draft.step_payload("objective").get("question_id"),
                purpose=(draft.purpose_annotation or {}).get("purpose"),
                baseline_profile_name=draft.step_payload("baseline").get("baseline_profile_name"),
                exclude_draft_id=draft.draft_id,
            )
        except Exception:
            duplicates = ()
        if duplicates:
            st.info(
                "A saved study already uses this question and configuration. Resume "
                "it from My studies, or save this as a separate study."
            )
    buttons = st.columns(3)
    if buttons[0].button("Save draft", disabled=not name_valid):
        _save(st, root, draft)
    if position and buttons[1].button("Back"):
        draft.current_step_key = flow.steps[position - 1].key
        draft.step_index = position - 1
        _save(st, root, draft)
    if position < len(flow.steps) - 1:
        if buttons[2].button("Next", type="primary", disabled=bool(errors) or not name_valid):
            draft.current_step_key = flow.steps[position + 1].key
            draft.step_index = position + 1
            _save(st, root, draft)
    else:
        if draft.mode_id == "full_pipeline_run":
            from ifvg_research_pipeline import render_pipeline_launch

            render_pipeline_launch(st, draft, resolution, roots, errors=errors)
        elif st.button("Run study", type="primary", disabled=bool(errors) or not name_valid):
            # The original handler revalidates authorization and immutable inputs.
            try:
                w._freeze_and_launch(st, draft, roots)
            except Exception:
                st.error(
                    "The study could not start. Its saved evidence is retained; "
                    "refresh and review readiness before retrying."
                )


def _save(st, root, draft):
    try:
        w.save_draft(root, draft)
        st.session_state.pop(SESSION_DRAFT_KEY, None)
        st.rerun()
    except w.DraftError:
        st.error("The draft could not be saved. Refresh and try again.")


def _validation_message(step, field, message):
    # Preserve useful date/threshold explanations, without raw identifiers.
    text = str(message)
    from alpha_lab.agents.data_infra.ifvg.presentation.workspace import _TECHNICAL

    if not _TECHNICAL.search(text) and not any(
        token in text.lower()
        for token in ("namespace", "unmarked", "schema", "manifest", "runner-entry")
    ):
        return text
    return {
        "objective": "Choose a compatible research question and objective.",
        "baseline": "Choose a configuration that is available for research.",
        "search_space": "Select enough permitted settings for this study's comparison or search.",
        "prop_contracts": (
            "Select the required number of verified firm contracts before continuing."
        ),
        "risk_policies": "Complete the account and risk settings for each selected firm.",
        "benchmarks": (
            "Check the required thresholds; values must follow the displayed units and bounds."
        ),
        "validation": "Choose valid trading days within the permitted research period.",
        "review": (
            "Running is blocked until the study settings and required authorization are ready."
        ),
    }.get(step, "Review the selected settings before continuing.")


def render_step(st, draft, step, resolution):
    payload = draft.step_payload(step)
    key = _PREFIX + step + "_"
    if step == "objective":
        from alpha_lab.agents.data_infra.ifvg.study_presentation import MODE_QUESTION_COMPATIBILITY

        question = _pick(
            st,
            "Research question",
            MODE_QUESTION_COMPATIBILITY[draft.mode_id],
            payload.get("question_id"),
            key + "question",
            lambda value: QUESTIONS.get(value, w.RESEARCH_QUESTIONS[value]),
        )
        templates = {
            t.template_id: t
            for t in w.OBJECTIVE_TEMPLATES
            if not w.mode_compatibility_error(draft.mode_id, question, t.template_id)
        }
        with st.expander("Advanced research objective"):
            template_id = _pick(
                st,
                "Objective",
                templates,
                payload.get("template_id"),
                key + "template",
                lambda value: templates[value].label,
            )
            custom = tuple(payload.get("custom_objectives") or ())
            if template_id == "custom":
                from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import describe
                from alpha_lab.agents.data_infra.ifvg.search.charter import OBJECTIVE_DIRECTIONS

                custom = tuple(
                    st.multiselect(
                        "Metrics to optimize",
                        sorted(OBJECTIVE_DIRECTIONS),
                        default=custom,
                        format_func=lambda value: describe(value).human_name,
                        key=key + "custom",
                    )
                )
        return {
            "mode_id": draft.mode_id,
            "question_id": question,
            "template_id": template_id,
            "custom_objectives": custom,
        }
    if step == "baseline":
        names = list(w.PROFILE_CAPABILITY_REGISTRY)
        selected = _pick(
            st,
            "Configuration",
            names,
            payload.get("baseline_profile_name"),
            key + "profile",
            profile_name,
        )
        capability = w.PROFILE_CAPABILITY_REGISTRY[selected]
        blocked = None
        section_hash = ""
        if capability.status is not w.ProfileCapabilityStatus.RUNNABLE:
            blocked = "The selected configuration is unavailable for this research workflow."
            st.warning(blocked)
        try:
            from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

            section = w.canonicalize_section(
                w.resolve_profile_config({"profile_name": selected}).section
            )
            section_hash = ifvg_profile_hash(section)
            directions = [
                label
                for flag, label in (("enable_longs", "Long"), ("enable_shorts", "Short"))
                if getattr(section, flag, False)
            ]
            st.caption(" / ".join(directions) + " · One active setup and trade at a time")
        except Exception:
            blocked = "The selected configuration could not be verified."
            st.error(blocked)
        return {
            "baseline_profile_name": selected,
            "baseline_section_config_hash": section_hash,
            "baseline_blocked_reason": blocked,
        }
    if step == "search_space":
        selections = {}
        grouped = w.grouped_axis_keys()
        for group in w.AXIS_GROUP_ORDER:
            keys = [
                axis
                for axis in grouped.get(group, ())
                if w.axis_renders_widget(w.SEARCH_AXIS_REGISTRY_V1[axis])
            ]
            if not keys:
                continue
            with st.expander(group, expanded=group == "Staleness"):
                for axis in keys:
                    spec = w.SEARCH_AXIS_REGISTRY_V1[axis]
                    options = [
                        value
                        for value in spec.registered_values
                        if value in w.AXIS_VALUE_REGISTRY_V1
                    ]
                    values = st.multiselect(
                        spec.human_label,
                        options,
                        default=[
                            v
                            for v in (payload.get("axis_selections") or {}).get(axis, ())
                            if v in options
                        ],
                        format_func=axis_value_name,
                        key=key + axis,
                    )
                    if values:
                        selections[axis] = tuple(values)
        interpretation = payload.get("interpretation") or w.INTERPRETATIONS[2]
        if set(selections) & w.SESSION_DIRECTION_AXES:
            interpretation = _pick(
                st,
                "Interpret session and direction changes as",
                w.INTERPRETATIONS,
                interpretation,
                key + "interpretation",
            )
        st.caption(
            f"{w.enumerate_child_count(selections) if selections else 1} "
            "configurations, including the baseline"
        )
        return {
            "mode_id": draft.mode_id,
            "axis_selections": selections,
            "interpretation": interpretation,
        }
    if step == "validation":
        st.write(
            f"Permitted research period: {w.DEVELOPMENT_EVIDENCE_FIRST_DAY} "
            f"to {w.DEVELOPMENT_EVIDENCE_LAST_DAY}."
        )
        raw = st.text_area(
            "Trading days (one date per line)",
            value="\n".join(payload.get("real_dates") or ()),
            placeholder="YYYY-MM-DD",
            key=key + "days",
        )
        days = tuple(line.strip() for line in raw.splitlines() if line.strip())
        st.caption(f"{len(days)} evidence days · {len(w.FROZEN_WARMUP_DATES)} fixed warmup days")
        with st.expander("Advanced research settings"):
            seed = int(
                st.number_input(
                    "Random seed", min_value=0, value=int(payload.get("seed", 7)), key=key + "seed"
                )
            )
            st.caption(
                "Strategy searches are exploratory. Model studies use their saved "
                "walk-forward protocol."
            )
        if not resolution.resolved.freeze_allowed:
            st.warning(
                "Required research authorization is not ready. You can save this "
                "draft and finish the settings; running remains unavailable."
            )
        return {
            "run_scope": resolution.resolved.run_scope,
            "evidence_class": "real",
            "real_dates": days,
            "warmup_dates": w.FROZEN_WARMUP_DATES,
            "seed": seed,
            "worker_limit": 1,
        }
    if step == "benchmarks":
        return _benchmarks(st, draft, resolution, key)
    if step == "prop_contracts":
        return _contracts(st, draft, resolution.roots, key)
    if step == "risk_policies":
        return _risk(st, draft, key)
    return _review(st, draft, resolution, key)


def _benchmarks(st, draft, resolution, key):
    payload = draft.step_payload("benchmarks")
    groups = [
        (
            "strategy_gates",
            "Strategy quality",
            w._STRATEGY_GATE_ROWS,
            w.ResolvedStrategyGateThresholds(min_session_stability_score=0.5).model_dump(),
        )
    ]
    if resolution.flow.includes("prop_contracts"):
        groups.append(
            (
                "prop_gates",
                "Prop feasibility",
                w._PROP_GATE_ROWS,
                w._PROP_GATE_DEFAULTS,
            )
        )
    if resolution.flow.goal in w._ROBUSTNESS_GOALS:
        groups.append(
            (
                "robustness_gates",
                "Consistency",
                w._ROBUSTNESS_GATE_ROWS,
                w._ROBUSTNESS_GATE_DEFAULTS,
            )
        )
    values = {"strategy_gates": {}, "prop_gates": {}, "robustness_gates": {}}
    st.caption(
        "These proposed research thresholds determine which configurations pass. "
        "They are saved with the study."
    )
    for group, title, rows, defaults in groups:
        with st.expander(title, expanded=group == "strategy_gates"):
            for name, label, unit in rows:
                default = payload.get(group, {}).get(name, defaults.get(name))
                if isinstance(default, bool):
                    value = st.checkbox(label, value=default, key=key + name)
                elif default is None:
                    value = None
                    st.caption(label.replace(" (schema-reserved)", "") + ": not required")
                else:
                    value = st.number_input(
                        f"{label} ({unit})", value=float(default), key=key + name
                    )
                values[group][name] = value
    return values


def _contracts(st, draft, roots, key):
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_contract_summaries

    cards = [
        card
        for card in load_contract_summaries(Path(roots["store_root"]))
        if card["verification_status"] != "synthetic_fixture_verified"
    ]
    selected, launchable = [], []
    if not cards:
        st.warning(
            "No verified firm contracts are available. This study needs verified "
            "contract evidence before it can run. Save the draft and return when "
            "that evidence is ready."
        )
    for index, card in enumerate(cards):
        identity = card["firm_contract_id"]
        label = f"{card['firm']} · {card['account_type']} · {card['account_size_label']}"
        with st.expander(label):
            from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
            from alpha_lab.propsim.firm_contracts import PropFirmContractEnvelope

            contract = load_verified_envelope(
                Path(roots["store_root"]), "prop_contracts", identity, PropFirmContractEnvelope
            ).payload
            terms = {
                "Evaluation": contract.evaluation.model_dump(mode="json"),
                "Payout": contract.payout.model_dump(mode="json"),
                "Fees": contract.fees.model_dump(mode="json"),
            }
            st.caption(f"Effective date: {card['effective_date']}")
            st.dataframe(_term_rows(terms), hide_index=True, width="stretch")
            if card["launchable"]:
                launchable.append(identity)
                if st.checkbox(
                    "Use this contract",
                    value=identity
                    in draft.step_payload("prop_contracts").get("selected_contract_ids", ()),
                    key=key + str(index),
                ):
                    selected.append(identity)
            else:
                st.warning("This contract's verification is incomplete. It cannot be used yet.")
    return {
        "mode_id": draft.mode_id,
        "selected_contract_ids": tuple(selected),
        "launchable_contract_ids": tuple(launchable),
        "prop_objective_selected": bool(w._prop_objectives_of(draft)),
        "prop_objectives": w._prop_objectives_of(draft),
    }


def _term_rows(value, prefix=""):
    from ifvg_research_review import words

    rows = []
    if isinstance(value, dict):
        for key, field in value.items():
            if any(token in key for token in ("_id", "hash", "schema", "source", "version")):
                continue
            rows.extend(_term_rows(field, f"{prefix} / {words(key)}".strip(" /")))
    elif isinstance(value, (list, tuple)):
        for index, field in enumerate(value):
            rows.extend(_term_rows(field, f"{prefix} / {index + 1}"))
    elif value is not None:
        rows.append({"Rule": prefix, "Terms": words(str(value))})
    return rows


def _risk(st, draft, key):
    contracts = draft.step_payload("prop_contracts").get("selected_contract_ids", ())
    policies = {}
    for index, contract in enumerate(contracts):
        previous = (
            draft.step_payload("risk_policies").get("per_firm_policies", {}).get(contract, {})
        )
        with st.expander(f"Account policy {index + 1}", expanded=True):
            prefix = key + str(index)
            template = _pick(
                st,
                "Risk method",
                w.RISK_POLICY_TEMPLATES,
                previous.get("risk_template"),
                prefix + "risk",
            )
            risk = st.number_input(
                "Risk amount (dollars or percent for the selected method)",
                min_value=0.0,
                value=float(previous.get("risk_value", 100)),
                key=prefix + "amount",
            )
            withdrawals = {
                "withdraw_at_every_eligibility_v1": "Withdraw whenever eligible",
                "withdraw_monthly_cadence_v1": "Withdraw monthly",
                "accumulate_no_withdrawal_v1": "Accumulate without withdrawals",
            }
            withdrawal = _pick(
                st,
                "Withdrawals",
                withdrawals,
                previous.get("withdrawal_policy"),
                prefix + "withdrawal",
                withdrawals.get,
            )
            replacements = {
                "none": "No replacements",
                "replace_once": "Replace once",
                "replace_twice": "Replace twice",
            }
            replacement = _pick(
                st,
                "Account replacement",
                replacements,
                previous.get("replacement_policy"),
                prefix + "replacement",
                replacements.get,
            )
            accounts = int(
                st.number_input(
                    "Accounts per firm",
                    min_value=1,
                    max_value=w.MAX_ACCOUNTS_PER_FIRM,
                    value=int(previous.get("n_accounts", 1)),
                    key=prefix + "accounts",
                )
            )
            if accounts > 1:
                st.caption("Copied accounts share the same simulated market and trade path.")
            policies[contract] = {
                "risk_template": template,
                "risk_value": risk,
                "withdrawal_policy": withdrawal,
                "replacement_policy": replacement,
                "n_accounts": accounts,
            }
    return {"selected_contract_ids": tuple(contracts), "per_firm_policies": policies}


def _review(st, draft, resolution, key):
    resolved = resolution.resolved
    report = w._satisfiability_for_draft(draft, resolved, resolution.flow)
    validation = draft.step_payload("validation")
    st.write(profile_name(draft.step_payload("baseline").get("baseline_profile_name", "")))
    st.write(date_scope(validation.get("real_dates")))
    st.write(report.configuration_sentence)
    for rule in report.failures:
        st.warning(_validation_message("review", rule.rule_id, rule.detail))
    if not resolved.freeze_allowed:
        st.warning(
            "Running is unavailable until the required authorization and evidence "
            "are ready. Your settings can still be saved."
        )
    acknowledged = resolved.purpose is w.RunPurpose.FULL_AUTHORIZED_DEVELOPMENT
    typed = ""
    if acknowledged:
        st.warning(
            "This runs the full permitted development scope. Review the dates and "
            "research settings before continuing."
        )
        typed = st.text_input(
            f"Type '{w.FULL_SCOPE_ACKNOWLEDGEMENT}' to confirm this scope", key=key + "ack"
        )
    return {
        "run_scope": resolved.run_scope,
        "purpose": resolved.purpose.value,
        "typed_acknowledgement": typed,
        "required_acknowledgement": w.FULL_SCOPE_ACKNOWLEDGEMENT,
        "requires_acknowledgement": acknowledged,
        "n_children": w.enumerate_child_count(w._challenger_selections(draft, resolution.flow)),
        "satisfiable": report.passed,
        "freeze_block_reason": resolved.freeze_block_reason,
    }
