"""Research configuration, recovery and results over the existing pipeline seam."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as _streamlit

_ACTIVITIES = (
    "Check study settings",
    "Prepare configurations",
    "Evaluate strategies",
    "Prepare supporting evidence",
    "Prepare trade charts",
    "Build model features",
    "Check feature coverage",
    "Label outcomes",
    "Build evaluation folds",
    "Train models",
    "Evaluate predictions",
    "Evaluate model-guided strategies",
    "Simulate firm accounts",
    "Evaluate uncertainty and stress",
    "Compare results",
    "Verify final evidence",
)


def activity_label(stage):
    try:
        return _ACTIVITIES[int(str(stage).split("_", 1)[0])]
    except (ValueError, TypeError, IndexError):
        return "Study activity unavailable"


def research_label(value):
    from alpha_lab.agents.data_infra.ifvg.presentation.labels import label_for

    for domain in ("model_protocol", "bundle"):
        try:
            label = label_for(domain, value)
            if label and label != value and "_" not in label:
                return label
        except (ValueError, KeyError):
            pass
    return str(value).replace("ifvg_context_", "").replace("_v1", "").replace("_", " ").capitalize()


_RESEARCH = "ifvg_research_group_"
_BASE_BUNDLES = {
    "B0_CORE": "Core and session",
    "B1_CORE_STRUCTURE": "Core and market structure",
    "B4_CORE_STRUCTURE_LIQUIDITY": "Core, structure and liquidity",
}


def _research_api():
    from alpha_lab.agents.data_infra.ifvg.search import research_runs

    return research_runs


def _research_state_root(roots):
    if roots.get("pipeline_state_root"):
        return Path(roots["pipeline_state_root"])
    from ifvg_pipeline_tab import PIPELINE_STATE_ROOT

    return PIPELINE_STATE_ROOT


def _source_label(row):
    return f"{row.get('label') or 'Saved configuration'} · {row['core_replay_id'][:12]}"


def _inherited_source_rows(subjects):
    """Keep every selected child's own target and cost visible."""
    return [
        {
            "Configuration": _source_label(row),
            "Target (R)": row.get("tp_r_multiple"),
            "Round-trip cost (points)": row.get("cost_points"),
            "Evaluation days": len(row.get("evaluation_dates") or ()),
            "Warmup days": len(row.get("warmup_dates") or ()),
        }
        for row in subjects
    ]


def _default_evaluation_window(subjects):
    """Use the common saved evaluation span; backend verifies exact dates."""
    calendars = [set(row.get("evaluation_dates") or ()) for row in subjects]
    days = sorted(set.intersection(*calendars)) if calendars else []
    if not days:
        return None
    return date.fromisoformat(days[0]), date.fromisoformat(days[-1])


def _research_regime_request(st, store_root, *, enabled, stored, primary_bundle="B0_CORE"):
    if not enabled:
        return None, []
    from ifvg_pipeline_tab import (
        _REGIME_CANDIDATE_INPUT_DEFAULT,
        _panel_bundle_key,
        _panel_numeric_inputs,
        _regime_candidate_numeric_inputs,
    )

    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
        PANEL_AS_OF_POLICY_ID_V1,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
        REGIME_PROPOSED_DEFAULTS,
        sample_adequacy_minimum,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
    from alpha_lab.agents.data_infra.ifvg.study_providers import list_catalogued_envelope_ids

    mode = st.radio(
        "Regime study",
        ("Describe market regimes", "Use approved regimes as model features"),
        index=1
        if tuple((stored or {}).get("comparison_classes_requested") or ()) == ("feature_only",)
        else 0,
        key=_RESEARCH + "regime_mode",
    )
    supervised = mode == "Use approved regimes as model features"
    refs = {}
    blockers = []
    if supervised:
        st.info(
            "This follow-up compares core features with core plus regime features. "
            "Select the exact approved decision, owner authorization and assessment."
        )
        for field, label, family in (
            ("regime_promotion_decision_id", "Approved regime decision", "regime_promotions"),
            ("owner_decision_artifact_id", "Regime owner authorization", "owner_decisions"),
            ("required_capability_assessment_id", "Regime assessment", "regime_assessments"),
        ):
            try:
                entries = list_catalogued_envelope_ids(store_root, family)
            except Exception:
                entries = ()
                blockers.append(f"The saved {label.lower()} catalog could not be verified.")
            names = {identity: f"{name or label} · {identity[:12]}" for identity, name in entries}
            previous = (stored or {}).get(field)
            if previous and previous not in names:
                # Exact linked references remain inspectable; preflight must verify them.
                names[previous] = f"Linked {label.lower()} · {previous[:12]}"
            if not names:
                blockers.append(f"No {label.lower()} is available for this follow-up.")
                continue
            refs[field] = st.selectbox(
                label,
                list(names),
                index=list(names).index(previous) if previous in names else 0,
                format_func=names.get,
                key=_RESEARCH + field,
            )
        if blockers:
            return None, blockers
    linked = (
        supervised
        and bool(stored)
        and (tuple(stored.get("comparison_classes_requested") or ()) == ("feature_only",))
    )
    if linked:
        fields = dict(stored)
        st.info("The linked study retains the exact precursor observation and numerical protocol.")
        with st.expander("Frozen precursor regime settings"):
            st.json({key: value for key, value in fields.items() if key not in refs})
    else:
        grain = st.radio(
            "Regime observations",
            ("Completed 5-minute bars", "Candidate entry decisions"),
            index=1
            if (stored or {}).get("observation_granularity") == "candidate_stage_row"
            else 0,
            key=_RESEARCH + "regime_grain",
        )
        panel = grain == "Completed 5-minute bars"
        fields = dict(stored or {})
        fields.update(
            observation_granularity="context_bar_panel" if panel else "candidate_stage_row",
            panel_interval_seconds=300 if panel else None,
            panel_as_of_policy_id=PANEL_AS_OF_POLICY_ID_V1 if panel else None,
            input_feature_bundle_key=_panel_bundle_key() if panel else primary_bundle,
            resolved_cluster_count=int(REGIME_PROPOSED_DEFAULTS["fixed_cluster_count"]["value"]),
            stratified_reporting_requested=True,
        )
        if panel:
            fields["resolved_input_features"] = _panel_numeric_inputs()
            st.caption("Numeric panel inputs: " + ", ".join(fields["resolved_input_features"]))
        else:
            try:
                options = _regime_candidate_numeric_inputs(primary_bundle)
            except Exception:
                return None, ["The selected candidate feature bundle could not be resolved."]
            default = [
                name
                for name in (stored or {}).get(
                    "resolved_input_features", _REGIME_CANDIDATE_INPUT_DEFAULT
                )
                if name in options
            ]
            fields["resolved_input_features"] = tuple(
                st.multiselect(
                    "Numeric regime input features",
                    options,
                    default=default,
                    key=_RESEARCH + "regime_inputs_" + primary_bundle,
                )
            )
            if not fields["resolved_input_features"]:
                blockers.append("Select at least one numeric candidate regime feature.")
        maximum = int(REGIME_PROPOSED_DEFAULTS["bootstrap_refits_per_fold"]["value"])
        fields["bootstrap_refits"] = int(
            st.number_input(
                "Requested stability refits per fold",
                min_value=1,
                max_value=maximum,
                value=int((stored or {}).get("bootstrap_refits") or maximum),
                key=_RESEARCH + "regime_refits",
            )
        )
        clipping = ("none", "clip_p01_p99_train_fitted_v1")
        fields["winsorization_policy"] = st.selectbox(
            "Regime input clipping",
            clipping,
            index=clipping.index((stored or {}).get("winsorization_policy", "none")),
            format_func=lambda value: (
                "None" if value == "none" else "Training-fold 1st/99th percentiles"
            ),
            key=_RESEARCH + "regime_clipping",
        )
    if blockers:
        return None, blockers
    fields.update(
        comparison_classes_requested=("feature_only",) if supervised else ("cohort_descriptive",),
        supervised_bundle_key="B0_CORE" if supervised else None,
        regime_promotion_decision_id=None,
        owner_decision_artifact_id=None,
        required_capability_assessment_id=None,
    )
    fields.update(refs)
    request = RegimeStudyRequest.model_validate(fields)
    floor = sample_adequacy_minimum(request.observation_granularity)
    st.caption(
        f"KMeans K={request.resolved_cluster_count} uses numeric inputs "
        f"at {request.observation_stage.value}. "
        f"The registered minimum is {floor} training observations per fold. "
        "Preprocessing and fitting use training data only."
    )
    return request.model_dump(mode="json"), []


def research_authorization_statement(preflight):
    """A human-readable approval statement bound to the reviewed exact plan."""
    request = preflight["request"]
    spec = request["study_spec"]
    subjects = ", ".join(request["source_core_replay_ids"])
    return (
        f"I authorize research plan {preflight['plan_id']} for exact source replays {subjects}, "
        f"evaluated from {spec['evaluation_start']} through {spec['evaluation_end']}, "
        f"using {spec['base_bundle_key']}, model protocol {spec['model_protocol_id']}, "
        f"MBP-1 comparison={bool(spec.get('mbp1_comparison'))}, and the regime request "
        "frozen in this plan. Each subject inherits its saved target and cost. "
        "This authorizes the reviewed source/context preparation, feature measurement, "
        "labels, fold-local fitting and research reports. It grants no live execution, "
        "model-gated replay, prop simulation or feature promotion."
    )


def _import_mbp_receipt(roots, row, receipt, *, dry_run):
    from alpha_lab.agents.data_infra.ifvg.search.research_mbp1 import (
        import_research_mbp1_receipt,
    )
    from alpha_lab.agents.data_infra.ifvg.search.research_subject import bind_research_subject

    root = Path(roots["store_root"])
    subject = bind_research_subject(
        root, row["core_replay_id"], evaluation_dates=tuple(row["evaluation_dates"])
    )
    if subject.subject_id != row["subject_id"]:
        raise ValueError("receipt subject differs from the reviewed research source")
    return import_research_mbp1_receipt(
        subject, root, Path(roots["repo_root"]), receipt, dry_run=dry_run
    )


def _render_mbp_receipt_import(st, roots, preflight):
    """Receipt preview is read-only; only the explicit import button persists it."""
    rows = {row["core_replay_id"]: row for row in preflight.get("subjects", ())}
    if not rows or not roots.get("repo_root"):
        return
    with st.expander("Import reviewed MBP-1 completeness evidence"):
        st.caption(
            "A completeness receipt must bind the physical partition, exact source scope and "
            "owner review. Unknown coverage remains unknown until verified evidence supports it."
        )
        selected = st.selectbox(
            "Receipt source configuration",
            list(rows),
            format_func=lambda value: _source_label(rows[value]),
            key=_RESEARCH + "receipt_source",
        )
        upload = st.file_uploader(
            "Reviewed MBP-1 receipt (JSON)", type=["json"], key=_RESEARCH + "receipt_file"
        )
        if upload is None:
            return
        raw = upload.getvalue()
        try:
            receipt = json.loads(raw)
            if not isinstance(receipt, dict):
                raise ValueError("receipt must be a JSON object")
        except (ValueError, UnicodeDecodeError):
            st.error("The uploaded receipt is not a valid JSON object.")
            return
        row = rows[selected]
        receipt_key = _RESEARCH + row["subject_id"] + "_receipt_" + hashlib.sha256(raw).hexdigest()
        st.json(receipt)
        if st.button("Verify receipt", key=receipt_key + "_verify"):
            try:
                st.session_state[receipt_key] = _import_mbp_receipt(
                    roots, row, receipt, dry_run=True
                )
            except Exception as error:
                from ifvg_ui_common import sanitize_error

                st.session_state.pop(receipt_key, None)
                st.error(f"Receipt verification failed: {sanitize_error(error)}")
        preview = st.session_state.get(receipt_key)
        if not preview:
            return
        st.write("Verified prospective evidence references")
        st.json(preview)
        if st.button("Import verified receipt", key=receipt_key + "_import"):
            try:
                saved = _import_mbp_receipt(roots, row, receipt, dry_run=False)
            except Exception as error:
                from ifvg_ui_common import sanitize_error

                st.error(f"Receipt import failed: {sanitize_error(error)}")
                return
            st.session_state[_RESEARCH + "receipt_notice"] = (
                "Reviewed MBP-1 evidence was imported: " + saved["coverage_evidence_id"]
            )
            st.rerun()


def render_research_configuration(st, *, roots, draft=None):
    """Metadata-only review until the explicit authorize-and-start action."""
    store_root = Path(roots["store_root"])
    stored = dict(draft.steps.get("review", {}).get("research_settings", {}) if draft else {})
    stored.update(st.session_state.get("ifvg_research_seed") or {})
    st.header("Feature and model research")
    st.write(
        "Choose exact saved configurations and review the work before authorizing a new study."
    )
    if notice := st.session_state.pop(_RESEARCH + "receipt_notice", None):
        st.success(notice)
    try:
        api = _research_api()
        sources = api.list_source_subjects(store_root)
    except Exception:
        st.error(
            "Saved source evidence could not be verified. Restore it before configuring research."
        )
        return
    if not sources:
        st.info("No saved strategy children are available. Complete a strategy study first.")
        return
    by_id = {row["core_replay_id"]: row for row in sources}
    selected = st.multiselect(
        "Saved strategy configurations",
        list(by_id),
        default=[value for value in stored.get("source_core_replay_ids", ()) if value in by_id],
        format_func=lambda value: _source_label(by_id[value]),
        key=_RESEARCH + "sources",
    )
    if not selected:
        st.info("Select one or more configurations. Each receives its own research results.")
        return
    subjects = [by_id[value] for value in selected]
    st.dataframe(_inherited_source_rows(subjects), hide_index=True, width="stretch")
    st.caption(
        "Targets and costs are inherited from each exact saved child. All candidates are included."
    )
    source_blockers = [
        f"{_source_label(row)}: {reason}" for row in subjects for reason in row.get("blockers", ())
    ]
    for problem in source_blockers:
        st.error(problem)
    if source_blockers:
        return
    with st.expander("Exact source references"):
        st.dataframe(
            [
                {
                    "Core replay": row["core_replay_id"],
                    "Effective section hash": row.get("profile_hash"),
                    "v2 dataset": row.get("v2_dataset_id"),
                }
                for row in subjects
            ],
            hide_index=True,
            width="stretch",
        )
    window = _default_evaluation_window(subjects)
    if window is None:
        st.error("These configurations have no common saved evaluation dates.")
        return
    if st.session_state.get(_RESEARCH + "window_sources") != selected:
        st.session_state.pop(_RESEARCH + "start", None)
        st.session_state.pop(_RESEARCH + "end", None)
        st.session_state[_RESEARCH + "window_sources"] = list(selected)
    display_name = st.text_input(
        "Research study name",
        value=stored.get("display_name")
        or getattr(draft, "display_name", None)
        or "Feature and model study",
        key=_RESEARCH + "name",
    ).strip()
    start, end = st.columns(2)
    evaluation_start = start.date_input(
        "Evaluation starts",
        value=max(
            window[0],
            min(
                window[1],
                date.fromisoformat(stored.get("evaluation_start") or window[0].isoformat()),
            ),
        ),
        min_value=window[0],
        max_value=window[1],
        key=_RESEARCH + "start",
    )
    evaluation_end = end.date_input(
        "Evaluation ends",
        value=max(
            window[0],
            min(
                window[1], date.fromisoformat(stored.get("evaluation_end") or window[1].isoformat())
            ),
        ),
        min_value=window[0],
        max_value=window[1],
        key=_RESEARCH + "end",
    )
    base = st.selectbox(
        "Base feature set",
        list(_BASE_BUNDLES),
        index=list(_BASE_BUNDLES).index(stored.get("base_bundle_key"))
        if stored.get("base_bundle_key") in _BASE_BUNDLES
        else 0,
        format_func=_BASE_BUNDLES.get,
        key=_RESEARCH + "base",
    )
    from ifvg_pipeline_tab import _bundle_model_protocol_ids

    protocols = _bundle_model_protocol_ids()
    model = st.selectbox(
        "Headline model",
        protocols,
        index=protocols.index(stored.get("model_protocol_id"))
        if stored.get("model_protocol_id") in protocols
        else 0,
        format_func=research_label,
        key=_RESEARCH + "model",
    )
    if base == "B4_CORE_STRUCTURE_LIQUIDITY":
        st.session_state[_RESEARCH + "mbp"] = False
    mbp = (
        st.checkbox(
            "Compare with MBP-1 order-flow features",
            value=bool(stored.get("mbp1_comparison")) and base != "B4_CORE_STRUCTURE_LIQUIDITY",
            disabled=base == "B4_CORE_STRUCTURE_LIQUIDITY",
            key=_RESEARCH + "mbp",
        )
        and base != "B4_CORE_STRUCTURE_LIQUIDITY"
    )
    if mbp:
        st.caption(
            "The controlled comparison uses core versus core + order flow"
            if base == "B0_CORE"
            else (
                "The controlled comparison uses core + structure versus core + "
                "structure + order flow"
            )
        )
        st.info(
            "MBP-1 coverage remains unknown where source completeness cannot be "
            "evidenced. Order flow is offline research only."
        )
    regimes = st.checkbox("Include market regime research", value=True, key=_RESEARCH + "regime")
    regime_request, local_blockers = _research_regime_request(
        st, store_root, enabled=regimes, stored=stored.get("regime_study"), primary_bundle=base
    )
    if not display_name:
        local_blockers.append("Enter a research study name.")
    if evaluation_start > evaluation_end:
        local_blockers.append("The evaluation end must follow its start.")
    spec = {
        "display_name": display_name,
        "evaluation_start": evaluation_start.isoformat(),
        "evaluation_end": evaluation_end.isoformat(),
        "base_bundle_key": base,
        "mbp1_comparison": mbp,
        "model_protocol_id": model,
        "regime_study": regime_request,
    }
    if draft is not None:
        draft.step_payload("review")["research_settings"] = {
            **spec,
            "source_core_replay_ids": list(selected),
        }
    for problem in local_blockers:
        st.error(problem)
    if local_blockers:
        return
    try:
        preflight = api.build_research_preflight(store_root, selected, spec)
    except Exception as error:
        from ifvg_ui_common import sanitize_error

        st.error(f"Research preflight could not be verified: {sanitize_error(error)}")
        return
    st.subheader("Review the research plan")
    for warning in preflight.get("warnings", ()):
        st.warning(str(warning))
    for blocker in preflight.get("blockers", ()):
        st.error(str(blocker))
    if mbp:
        _render_mbp_receipt_import(st, roots, preflight)
    cells = preflight.get("cells") or ()
    if cells:
        st.dataframe(
            [
                {
                    "Configuration": row.get("label"),
                    "Research": row.get("lane"),
                    "Outcome cutoff (UTC)": row.get("cutoff_ts_utc"),
                }
                for row in cells
            ],
            hide_index=True,
            width="stretch",
        )
    for activity in preflight.get("planned_work", ()):
        st.write(str(activity))
    for subject in preflight.get("subjects", ()):
        preparation = subject.get("preflight")
        mbp_preparation = subject.get("mbp1_preflight")
        if preparation or mbp_preparation:
            with st.expander("Source preparation evidence · " + _source_label(subject)):
                if preparation:
                    st.json(preparation)
                if mbp_preparation:
                    st.write("MBP-1 source evidence")
                    st.json(mbp_preparation)
    st.caption(
        "Prepare source and context evidence, measure features, derive labels, build folds, "
        "fit models, compare predictions and verify research artifacts."
    )
    with st.expander("Plan identity and activities"):
        st.code(preflight["plan_id"])
        st.write([activity_label(stage) for stage in preflight.get("planned_stages", ())])
    if not preflight.get("ready") or preflight.get("blockers"):
        st.info("Resolve the stated dependencies before authorizing this plan.")
        return
    statement = research_authorization_statement(preflight)
    st.caption(
        "This requires new research authorization; previous strategy-search approval is not reused."
    )
    with st.expander("Exact authorization statement"):
        st.write(statement)
    plan_key = _RESEARCH + preflight["plan_id"]
    author = st.text_input("Reviewer name", key=plan_key + "_author").strip()
    approved = st.checkbox("I authorize this exact research plan", key=plan_key + "_approved")
    if st.button(
        "Authorize and start research",
        type="primary",
        disabled=not author or not approved,
        key=plan_key + "_start",
    ):
        try:
            receipt = api.freeze_research_group(
                store_root, preflight, authorization_statement=statement, author=author
            )
        except Exception as error:
            from ifvg_ui_common import sanitize_error

            st.error(f"Research authorization could not be saved: {sanitize_error(error)}")
            return
        group_id = receipt["group_id"]
        st.session_state["ifvg_workspace_research_group"] = group_id
        st.session_state["ifvg_workspace_screen"] = "research_group"
        try:
            api.launch_research_group(store_root, group_id, _research_state_root(roots))
        except Exception:
            st.session_state[_RESEARCH + "startup_notice"] = (
                "The authorized study was saved, but worker startup was not confirmed. "
                "Resume it after reviewing its current status."
            )
        st.session_state.pop("ifvg_research_seed", None)
        st.rerun()


def render_research_group_cards(st, roots):
    try:
        groups = _research_api().list_research_groups(
            Path(roots["store_root"]), _research_state_root(roots)
        )
    except Exception:
        st.warning("Research study groups could not be verified. Their saved work is retained.")
        return
    if not groups:
        return
    st.subheader("Feature and model studies")
    for group in groups:
        with st.container(border=True):
            st.write(group.get("display_name") or "Feature and model study")
            st.caption(str(group.get("status") or "Saved; progress not recorded"))
            if st.button("Inspect research", key="research_open_" + group["group_id"]):
                st.session_state["ifvg_workspace_research_group"] = group["group_id"]
                st.session_state["ifvg_workspace_screen"] = "research_group"
                st.rerun()


def _research_cell_rows(cells):
    return [
        {
            "Configuration": row.get("label") or row.get("core_replay_id", "")[:12],
            "Research": row.get("lane"),
            "Status": row.get("status") or "Pending",
            "Evidence": row.get("reason") or "",
        }
        for row in cells
    ]


def render_research_group(st, *, roots, group_id):
    store_root, state_root = Path(roots["store_root"]), _research_state_root(roots)
    try:
        api = _research_api()
        group = api.read_research_group(store_root, group_id, state_root)
    except Exception:
        st.error("This research group's saved evidence could not be verified.")
        return
    st.header(group.get("display_name") or "Feature and model study")
    notice = st.session_state.pop(_RESEARCH + "startup_notice", None)
    if notice:
        st.warning(notice)
    st.caption(str(group.get("status") or "Saved; progress not recorded"))
    if group.get("reason"):
        st.warning(str(group["reason"]))
    for warning in group.get("warnings", ()):
        st.warning(str(warning))
    for blocker in group.get("blockers", ()):
        st.error(str(blocker))
    cells = group.get("cells") or []
    if not cells:
        st.info("No worker evidence has been recorded for this group.")
    else:
        st.dataframe(_research_cell_rows(cells), width="stretch", hide_index=True)
    if st.button("Refresh research progress", key=_RESEARCH + "refresh"):
        st.rerun()
    if str(group.get("status", "")).lower() == "running" and st.button(
        "Stop after current work", key=_RESEARCH + "cancel"
    ):
        try:
            api.cancel_research_group(store_root, group_id, state_root)
        except Exception:
            st.error("The stop request could not be saved. Refresh and try again.")
        else:
            st.info("Stop requested. Current work will reach a safe stopping point.")
    if str(group.get("status", "")).lower() not in ("running", "completed") and st.button(
        "Resume authorized research", key=_RESEARCH + "resume"
    ):
        try:
            api.launch_research_group(store_root, group_id, state_root)
        except Exception:
            st.error(
                "Research could not resume. Restore the reported evidence or "
                "authorization dependency."
            )
        else:
            st.info("Resume requested. Refresh to inspect the worker's persisted status.")
    if not cells:
        return
    options = list(range(len(cells)))
    if st.session_state.get(_RESEARCH + "selected_cell") not in options:
        st.session_state.pop(_RESEARCH + "selected_cell", None)
    index = st.selectbox(
        "Research evidence to inspect",
        options,
        format_func=lambda value: (
            f"{cells[value].get('label') or 'Configuration'} · "
            f"{cells[value].get('lane') or 'Research'}"
        ),
        key=_RESEARCH + "selected_cell",
    )
    cell = cells[index]
    pipeline_id = cell.get("pipeline_semantic_id")
    if not pipeline_id:
        st.info(cell.get("reason") or "Source preparation has not produced a pipeline record yet.")
        return
    _render_research_cell_evidence(st, roots, group, cell)


def _research_artifact_refs(state):
    """References come only from this cell's persisted state, including checkpoints."""
    references = set()
    for value in (state.get("stages") or {}).values():
        references.update(value.get("output_artifact_ids") or ())
        references.update(value.get("research_model_input_ids") or ())
    references.update(
        row["executed_trade_table_id"]
        for row in state.get("children") or ()
        if row.get("executed_trade_table_id")
    )
    # State is operational metadata. Restrict path components before exact loads.
    return tuple(
        sorted(
            value
            for value in references
            if isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        )
    )


def _load_research_tables(store_root, state):
    """Verified saved research evidence; never reads raw inputs or fits a model."""
    from alpha_lab.agents.data_infra.ifvg.ml.research_evidence import load_research_inputs
    from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
        load_executed_trade_table,
    )
    from alpha_lab.agents.data_infra.ifvg.search.research_artifacts import (
        ResearchCohortEnvelope,
        load_research_labels,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope

    saved = {"labels": [], "cohorts": [], "trades": [], "model_inputs": [], "errors": []}
    root = Path(store_root)
    for artifact_id in _research_artifact_refs(state):
        for family, key in (
            ("research_labels", "labels"),
            ("research_cohorts", "cohorts"),
            ("executed_trade_tables", "trades"),
            ("research_model_inputs", "model_inputs"),
        ):
            if not (root / family / artifact_id).is_dir():
                continue
            try:
                if key == "labels":
                    envelope, frame = load_research_labels(root, artifact_id)
                    value = {"metadata": envelope.model_dump(mode="json"), "frame": frame}
                elif key == "cohorts":
                    envelope = load_verified_envelope(
                        root, family, artifact_id, ResearchCohortEnvelope
                    )
                    value = {"metadata": envelope.model_dump(mode="json")}
                elif key == "trades":
                    loaded = load_executed_trade_table(root, artifact_id)
                    value = {
                        "metadata": loaded.envelope.model_dump(mode="json"),
                        "frame": loaded.frame,
                    }
                else:
                    value = load_research_inputs(root, artifact_id)
                saved[key].append({"artifact_id": artifact_id, **value})
            except Exception:
                saved["errors"].append(f"Saved {family}/{artifact_id} failed verification.")
    return saved


def _research_review_target(roots, core_id):
    """Find an exact existing search-child chart route using saved metadata only."""
    from alpha_lab.agents.data_infra.ifvg.study_providers import (
        list_search_runs,
        load_search_state,
    )

    if not roots.get("state_root"):
        return None
    for run in list_search_runs(Path(roots["state_root"]), Path(roots["store_root"])):
        if run.archived:
            continue
        state = load_search_state(Path(roots["state_root"]), run.search_id) or {}
        if any(
            row.get("core_replay_id") == core_id and row.get("state") in ("completed", "reused")
            for row in state.get("children") or ()
        ):
            return run.search_id
    return None


def _load_research_charts(store_root, cell, state):
    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
        load_verified_research_replay_chart,
    )
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import PipelineStageResultEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope

    stage = "04_build_or_reuse_replay_charts"
    entry = (state.get("stages") or {}).get(stage) or {}
    if not entry.get("stage_result_id"):
        return []
    result = load_verified_envelope(
        Path(store_root),
        "pipeline_stage_results",
        entry["stage_result_id"],
        PipelineStageResultEnvelope,
    )
    if (
        result.payload.pipeline_semantic_id != cell["pipeline_semantic_id"]
        or result.payload.stage.value != stage
    ):
        raise ValueError("chart stage belongs to another research cell")
    return [
        load_verified_research_replay_chart(
            Path(store_root),
            artifact_id,
            research_subject_id=cell["subject_id"],
            core_replay_id=cell["core_replay_id"],
        )
        for artifact_id in result.payload.output_artifact_ids
    ]


def _research_candidate_figure(chart, source, candidate, label, timeframe):
    import plotly.graph_objects as go

    bars = (
        source.bars_1m.copy()
        if timeframe == 60
        else chart.bars_tf.loc[chart.bars_tf["timeframe_seconds"] == timeframe].copy()
    )
    bars["close_ts_utc"] = pd.to_datetime(bars["close_ts_utc"], utc=True, errors="raise")
    start = candidate.get("setup_start_ts")
    if pd.isna(start):
        start = candidate["entry_anchor_ts"]
    end = pd.Timestamp(candidate["display_end_ts"])
    if label is not None:
        label_end = label.get("resolution_ts_utc")
        if pd.isna(label_end):
            label_end = label.get("label_window_end")
        if pd.notna(label_end):
            end = max(end, pd.Timestamp(label_end))
    cutoff = pd.Timestamp(source.envelope.payload.subject.cutoff_ts_utc)
    bars = bars.loc[
        bars["close_ts_utc"].between(
            pd.Timestamp(start) - pd.Timedelta(minutes=15),
            min(end + pd.Timedelta(minutes=15), cutoff),
        )
    ].sort_values("close_ts_utc")
    figure = go.Figure(
        go.Candlestick(
            x=bars["close_ts_utc"].dt.tz_convert("America/New_York"),
            open=bars["open_ticks"] * 0.25,
            high=bars["high_ticks"] * 0.25,
            low=bars["low_ticks"] * 0.25,
            close=bars["close_ticks"] * 0.25,
            name="Verified source bars",
        )
    )
    if label is not None:
        for name, field, color in (
            ("Research entry", "entry_ticks", "#2563eb"),
            ("Research stop", "stop_ticks", "#dc2626"),
            ("Configured research target", "target_ticks", "#15803d"),
        ):
            if pd.notna(label.get(field)):
                figure.add_hline(
                    y=float(label[field]) * 0.25,
                    line_color=color,
                    line_dash="dot",
                    annotation_text=name,
                )
    figure.update_layout(
        template="plotly_white",
        height=560,
        xaxis_rangeslider_visible=False,
        xaxis_title="America/New_York",
        yaxis_title="Price",
    )
    return figure, bars


def _render_research_chart(st, roots, cell, state):
    st.subheader("Research candidate chart")
    try:
        records = _load_research_charts(roots["store_root"], cell, state)
    except Exception:
        st.error("The selected cell's chart or forward source failed verification.")
        return
    if not records:
        st.info("No verified research chart stage has been recorded yet.")
        return
    if len(records) != 1:
        st.error("This research cell does not have one exact chart source.")
        return
    chart, source = records[0]
    ranges = chart.candidate_ranges
    if ranges.empty:
        st.info("This research window contains no chartable candidates.")
        return
    options = ranges["candidate_id"].astype(str).tolist()
    key = _RESEARCH + cell["pipeline_semantic_id"] + "_chart_candidate"
    pending = st.session_state.pop(_RESEARCH + "pending_chart_candidate", None)
    if pending and pending[0] == cell["pipeline_semantic_id"] and pending[1] in options:
        st.session_state[key] = pending[1]
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    selected = st.selectbox("Exact research candidate", options, key=key)
    candidate = ranges.loc[ranges["candidate_id"].astype(str) == selected].iloc[0]
    saved = _load_research_tables(roots["store_root"], state)
    for error in saved["errors"]:
        st.error(error)
    labels = []
    for item in saved["labels"]:
        if item["metadata"]["payload"]["research_subject_id"] != cell["subject_id"]:
            st.error("The saved research labels belong to another subject.")
            return
        labels.extend(
            item["frame"]
            .loc[item["frame"]["candidate_id"].astype(str) == selected]
            .to_dict("records")
        )
    if len(labels) > 1:
        st.error("This candidate has ambiguous saved research labels.")
        return
    label = labels[0] if labels else None
    if label is None:
        st.info(
            "Configured research labels are not available; the chart shows verified price evidence."
        )
    else:
        st.caption("Annotations use this subject's verified configured-target research labels.")
        st.dataframe(
            [
                {
                    field: label.get(field)
                    for field in (
                        "candidate_id",
                        "reward_r",
                        "label",
                        "censored",
                        "censor_reason",
                        "gross_r",
                        "net_r",
                        "label_policy_id",
                        "cutoff_ts_utc",
                    )
                }
            ],
            hide_index=True,
            width="stretch",
        )
    timeframe = st.selectbox(
        "Research chart interval",
        (60, 300, 900, 3600),
        format_func=lambda value: f"{value // 60} minute bars",
        key=_RESEARCH + "chart_timeframe",
    )
    figure, bars = _research_candidate_figure(chart, source, candidate, label, timeframe)
    if bars.empty:
        st.info("The exact chart range has no bars at this interval.")
        return
    st.plotly_chart(figure, width="stretch", key=_RESEARCH + "candidate_chart")
    with st.expander("Verified chart references and source bars"):
        st.write(
            {
                "chart_id": chart.artifact_id,
                "forward_source_id": source.envelope.research_context_companion_id,
                "subject_id": cell["subject_id"],
            }
        )
        st.dataframe(bars, hide_index=True, width="stretch")


def _render_research_saved_tables(st, roots, cell, state, *, artifacts):
    saved = _load_research_tables(roots["store_root"], state)
    for error in saved["errors"]:
        st.error(error)
    if artifacts:
        st.subheader("Saved research artifacts")
        st.caption("Inputs remain inspectable when fitting is incomplete or no valid folds exist.")
        records = [
            {"Evidence": kind.replace("_", " "), "Exact reference": row["artifact_id"]}
            for kind in ("labels", "cohorts", "trades", "model_inputs")
            for row in saved[kind]
        ]
        if not records:
            st.info("No verified research tables have been recorded for this cell yet.")
        else:
            st.dataframe(records, hide_index=True, width="stretch")
        for item in saved["model_inputs"]:
            with st.expander("Model input evidence · " + item["artifact_id"][:12]):
                st.json(item["request"])
                st.write("Fold definitions")
                st.json(item["fold_definitions"])
                for key, title in (
                    ("candidate_features", "Candidate features"),
                    ("labels", "Configured research labels"),
                    ("fold_assignment", "Fold assignments"),
                ):
                    frame = item[key]
                    st.write(f"{title} · {len(frame)} rows")
                    st.dataframe(frame.head(500), hide_index=True, width="stretch")
                    if len(frame) > 500:
                        st.caption("Showing the first 500 persisted rows.")
        for kind in ("labels", "cohorts", "trades"):
            for item in saved[kind]:
                with st.expander(kind.capitalize() + " identity · " + item["artifact_id"][:12]):
                    st.json(item["metadata"])
        return

    st.subheader("Candidate population and configured research labels")
    for item in saved["cohorts"]:
        payload = item["metadata"]["payload"]
        st.dataframe(
            [
                {"Population": label, "Rows": len(payload.get(key) or ())}
                for key, label in (
                    ("candidate_ids", "Included candidates"),
                    ("executed_trade_ids", "Resolved executions from included candidates"),
                    ("excluded_warmup_candidate_ids", "Excluded warmup candidates"),
                    ("excluded_window_candidate_ids", "Excluded candidates outside the window"),
                    ("cutoff_censored_trade_ids", "Executions censored at the outcome cutoff"),
                    ("excluded_unresolved_trade_ids", "Unresolved executions excluded"),
                )
            ],
            hide_index=True,
            width="stretch",
        )
    if not saved["cohorts"]:
        st.info("The scoped candidate population has not been recorded yet.")
    for item in saved["labels"]:
        frame = item["frame"]
        st.caption(
            "These outcomes use the saved subject's configured R target and cost. "
            "Historical chart labels retain their original version 1 policy."
        )
        st.dataframe(frame.head(500), hide_index=True, width="stretch")
        if len(frame) > 500:
            st.caption(f"Showing 500 of {len(frame)} persisted label rows.")
        if not frame.empty:
            selected = st.selectbox(
                "Research candidate to chart",
                frame["candidate_id"].astype(str).tolist(),
                key=_RESEARCH + item["artifact_id"] + "_candidate",
            )
            if st.button(
                "Inspect configured research chart",
                key=_RESEARCH + item["artifact_id"] + "_open_chart",
            ):
                st.session_state[_RESEARCH + "pending_chart_candidate"] = (
                    cell["pipeline_semantic_id"],
                    selected,
                )
                st.session_state[_RESEARCH + "pending_chart_panel"] = True
                st.rerun()
    if not saved["labels"]:
        st.info("Configured research labels have not been recorded yet.")
    st.subheader("Executed trades")
    for item in saved["trades"]:
        frame = item["frame"]
        st.dataframe(frame.head(500), hide_index=True, width="stretch")
        if len(frame) > 500:
            st.caption(f"Showing 500 of {len(frame)} saved executions.")
        if frame.empty or "trade_id" not in frame:
            continue
        core_id = item["metadata"]["payload"]["core_replay_id"]
        if core_id != cell["core_replay_id"]:
            st.error("The saved execution table belongs to another source configuration.")
            continue
        trade_ids = frame["trade_id"].astype(str).tolist()
        selected = st.selectbox(
            "Exact execution to review", trade_ids, key=_RESEARCH + item["artifact_id"] + "_trade"
        )
        if st.button("Open saved execution chart", key=_RESEARCH + item["artifact_id"] + "_chart"):
            try:
                search_id = _research_review_target(roots, core_id)
            except Exception:
                search_id = None
            if search_id is None:
                st.info("No verified saved search route is available for this exact execution.")
            else:
                st.session_state["ifvg_search_review_pending"] = (search_id, core_id)
                st.session_state["ifvg_search_review_trade_" + core_id] = selected
                st.session_state["ifvg_study_v1_open_review"] = True
                st.rerun()
    if not saved["trades"]:
        st.info("No scoped execution table has been recorded yet.")


def _seed_linked_regime_study(st, group, cell, refs):
    spec = dict(group.get("study_spec") or (group.get("request") or {}).get("study_spec") or {})
    regime = dict(spec.get("regime_study") or {})
    regime.update(
        comparison_classes_requested=["feature_only"],
        supervised_bundle_key="B0_CORE",
        **refs,
    )
    seed = {
        **spec,
        "base_bundle_key": "B0_CORE",
        "mbp1_comparison": False,
        "source_core_replay_ids": [cell["core_replay_id"]],
        "regime_study": regime,
        "display_name": (group.get("display_name") or "Regime study") + " · Regime features",
    }
    for key in list(st.session_state):
        if key.startswith(_RESEARCH):
            del st.session_state[key]
    st.session_state["ifvg_research_seed"] = seed
    st.session_state["ifvg_workspace_screen"] = "research_new"
    st.rerun()


def _render_regime_feature_authorization(st, roots, group, cell):
    """A separate, concrete owner decision; reviewing never creates authority."""
    pipeline_id = cell["pipeline_semantic_id"]
    key = _RESEARCH + pipeline_id + "_eligibility"
    st.subheader("Regime feature eligibility")
    if st.button("Review regime feature eligibility", key=key + "_review"):
        st.session_state[key + "_open"] = True
    if not st.session_state.get(key + "_open"):
        return
    try:
        api = _research_api()
        review = api.read_research_regime_eligibility(
            Path(roots["store_root"]), group["group_id"], pipeline_id
        )
    except Exception:
        st.error("The exact regime eligibility evidence could not be verified.")
        return
    st.caption(
        "Review the saved assessment before approving its use in a separate core versus "
        "core + regime feature study. This decision does not launch that study."
    )
    st.write("Assessment: " + str(review.get("assessment_id") or "Unavailable"))
    st.write("Protocol: " + str(review.get("protocol_id") or "Unavailable"))
    st.dataframe(
        [
            {
                "Requested bootstrap refits": review.get("requested_bootstrap_refits"),
                "Applied bootstrap refits": review.get("applied_bootstrap_refits"),
            }
        ],
        hide_index=True,
        width="stretch",
    )
    for label, field in (
        ("Frozen protocol inputs", "input_features"),
        ("Eligibility gates", "gates"),
        ("Sample requirements", "sample_floors"),
        ("Supporting saved evidence", "evidence"),
    ):
        with st.expander(label, expanded=field in ("gates", "sample_floors")):
            st.json(review.get(field))
    for blocker in review.get("blockers", ()):
        st.error(str(blocker))
    if not review.get("ready") or review.get("blockers") or not review.get("review_id"):
        st.info("The saved assessment does not currently support regime feature approval.")
        return
    review_id = review["review_id"]
    statement = (
        f"I approve regime eligibility review {review_id}, assessment {review['assessment_id']} "
        f"under protocol {review['protocol_id']}, for source {cell['core_replay_id']} in "
        f"research group {group['group_id']} and pipeline {pipeline_id}. I reviewed the "
        "frozen numeric inputs, sample floors, applied bootstrap evidence and eligibility gates. "
        "I authorize feature-only core versus core + regime research eligibility. "
        "A new research plan requires separate authorization before fitting. "
        "This grants no live execution or production promotion."
    )
    with st.expander("Exact regime feature approval"):
        st.write(statement)
    review_key = _RESEARCH + review_id
    author = st.text_input("Regime eligibility reviewer", key=review_key + "_author").strip()
    approved = st.checkbox(
        "I approve feature use for this exact regime assessment", key=review_key + "_approved"
    )
    if st.button(
        "Approve regime feature use",
        disabled=not author or not approved,
        key=review_key + "_approve",
    ):
        try:
            refs = api.promote_research_regime(
                Path(roots["store_root"]),
                group["group_id"],
                pipeline_id,
                author=author,
                approval_statement=statement,
                review_id=review_id,
            )
        except Exception:
            st.error(
                "Regime feature approval was not confirmed. Recheck the saved eligibility evidence."
            )
            return
        _seed_linked_regime_study(
            st,
            group,
            cell,
            {
                field: refs[field]
                for field in (
                    "regime_promotion_decision_id",
                    "owner_decision_artifact_id",
                    "required_capability_assessment_id",
                )
            },
        )


def _render_research_evidence_status(st, store_root, state):
    from alpha_lab.agents.data_infra.ifvg.search.store import load_json_sidecar

    entry = (state.get("stages") or {}).get("09_train_models") or {}
    evidence = None
    verified = False
    if entry.get("stage_result_id"):
        try:
            evidence = load_json_sidecar(
                Path(store_root),
                "pipeline_stage_results",
                entry["stage_result_id"],
                "research_evidence_status.json",
            )
            verified = evidence is not None
        except Exception:
            st.error("The saved research adequacy report failed verification.")
            return
    evidence = evidence or state.get("research_evidence")
    st.subheader("Research evidence status")
    if not evidence:
        st.info("Research adequacy has not been recorded yet.")
        return
    if not verified:
        st.caption(
            "Operational advisory; a verified research adequacy report is not available yet."
        )
    if evidence.get("status") == "insufficient_evidence":
        st.warning("Research evidence is insufficient for the requested comparison.")
    elif evidence.get("status") == "evaluable":
        st.info(
            "Out-of-sample evidence is evaluable. This status does not establish "
            "positive model lift."
        )
    else:
        st.info("Research adequacy remains unresolved.")
    st.dataframe(
        [
            {
                "Out-of-sample rows": evidence.get("oos_rows"),
                "Valid out-of-sample folds": evidence.get("valid_oos_folds"),
                "Regime gates passed": evidence.get("regime_gates_passed"),
            }
        ],
        hide_index=True,
        width="stretch",
    )
    for reason in evidence.get("reasons", ()):
        st.write(str(reason))


def _render_research_source_progress(st, state, core_id):
    for row in state.get("children") or ():
        if row.get("core_replay_id") != core_id:
            continue
        st.subheader("Source and context preparation")
        if row.get("explanation"):
            st.caption(str(row["explanation"]))
        progress = row.get("context_progress") or {}
        completed, total = progress.get("completed_days"), progress.get("total_days")
        if isinstance(completed, int) and isinstance(total, int) and total > 0:
            st.progress(
                min(max(completed / total, 0.0), 1.0),
                text=(
                    f"Context capture: {completed} of {total} days · "
                    f"{progress.get('trading_day') or 'date pending'}"
                ),
            )
        if row.get("state") in ("completed", "reused") and "context_replay_invocations" in row:
            st.write(
                "Saved Core stream reused; exact context companion verified and reused."
                if row["context_replay_invocations"] == 0
                else "Saved Core stream reused; context capture replay completed for this subject."
            )


@_streamlit.fragment(run_every="5s")
def _live_research_source_progress(st, roots, group_id, pipeline_id):
    try:
        group = _research_api().read_research_group(
            Path(roots["store_root"]), group_id, _research_state_root(roots)
        )
        cell = next(row for row in group["cells"] if row["pipeline_semantic_id"] == pipeline_id)
        state = cell.get("state") or {}
    except Exception:
        st.warning("The current source-preparation progress could not be read.")
        return
    entry = (state.get("stages") or {}).get("02_run_or_reuse_sequential_replays") or {}
    if entry.get("status") != "running":
        st.rerun()
    _render_research_source_progress(st, state, cell["core_replay_id"])


def _render_research_cell_evidence(st, roots, group, cell):
    from ifvg_ui_common import PIPELINE_STATE_PREFIX

    from alpha_lab.agents.data_infra.ifvg.search.pipeline import read_pipeline_state
    from alpha_lab.agents.data_infra.ifvg.study_providers import (
        load_ladder_diagnostics,
        mbp1_stage_evidence_defaults,
        regime_stage_evidence_defaults,
    )

    pipeline_id = cell["pipeline_semantic_id"]
    store_root, state_root = Path(roots["store_root"]), _research_state_root(roots)
    state = cell.get("state") or read_pipeline_state(state_root, pipeline_id) or {}
    source_stage = (state.get("stages") or {}).get("02_run_or_reuse_sequential_replays") or {}
    if source_stage.get("status") == "running":
        _live_research_source_progress(st, roots, group["group_id"], pipeline_id)
    else:
        _render_research_source_progress(st, state, cell["core_replay_id"])
    # These readers inspect completed stages even when a later stage failed.
    try:
        diagnostics = load_ladder_diagnostics(store_root, state)
        mbp_ids, mbp_note = mbp1_stage_evidence_defaults(state_root, store_root, pipeline_id)
        regime_ids, regime_note = regime_stage_evidence_defaults(
            state_root, store_root, pipeline_id
        )
    except Exception:
        st.error("Persisted research evidence failed verification. It cannot be interpreted.")
        return
    for note in (mbp_note, regime_note):
        if note:
            st.error(note)
    if str(cell.get("status", "")).lower() not in ("completed", "reused"):
        st.warning(
            "This research cell is incomplete. Available stage artifacts below are "
            "partial evidence."
        )
    _render_research_evidence_status(st, store_root, state)
    stage_rows = [
        {
            "Activity": activity_label(key),
            "Status": value.get("status"),
            "Detail": value.get("explanation"),
        }
        for key, value in (state.get("stages") or {}).items()
        if value.get("in_plan")
    ]
    if stage_rows:
        with st.expander("Research activities"):
            st.dataframe(stage_rows, hide_index=True, width="stretch")
    references = [
        {"Activity": activity_label(key), "Artifact reference": artifact_id}
        for key, value in (state.get("stages") or {}).items()
        for artifact_id in value.get("output_artifact_ids", ())
    ]
    if references:
        with st.expander("Recorded artifact references"):
            st.caption(
                "References include inputs and completed stages from an incomplete run. "
                "Each evidence panel verifies its saved artifacts before reading them."
            )
            st.dataframe(references, hide_index=True, width="stretch")
    headline = (group.get("study_spec") or {}).get("model_protocol_id")
    if headline:
        st.caption(f"Headline review protocol: {research_label(headline)}")
    if diagnostics and diagnostics.get("rungs"):
        st.subheader("Out-of-sample model evidence")
        _model_details(
            st,
            diagnostics["rungs"],
            diagnostics,
            preferred_protocol=headline,
            selection_key=_RESEARCH + pipeline_id + "_model_review",
        )
    # The rich panel widgets must never retain another subject's exact IDs.
    if st.session_state.get(_RESEARCH + "panel_owner") != pipeline_id:
        for key in list(st.session_state):
            if key.startswith((PIPELINE_STATE_PREFIX + "mbp1_", PIPELINE_STATE_PREFIX + "regime_")):
                del st.session_state[key]
        st.session_state[_RESEARCH + "panel_owner"] = pipeline_id
    if st.session_state.pop(_RESEARCH + "pending_chart_panel", False):
        st.session_state[_RESEARCH + "evidence_panel"] = "Research chart"
    panel = st.radio(
        "Supporting research evidence",
        (
            "Overview",
            "Trades and cohorts",
            "Research chart",
            "Order flow",
            "Market regimes",
            "Artifacts",
        ),
        horizontal=True,
        key=_RESEARCH + "evidence_panel",
    )
    if panel == "Order flow":
        from ifvg_mbp1_panels import render_mbp1_order_flow

        render_mbp1_order_flow(st, roots={**roots, "store_root": store_root}, default_ids=mbp_ids)
    elif panel == "Market regimes":
        from ifvg_regime_panels import render_regime_lane

        render_regime_lane(st, roots={**roots, "store_root": store_root}, default_ids=regime_ids)
        _render_regime_feature_authorization(st, roots, group, cell)
    elif panel in ("Trades and cohorts", "Artifacts"):
        _render_research_saved_tables(st, roots, cell, state, artifacts=panel == "Artifacts")
    elif panel == "Research chart":
        _render_research_chart(st, roots, cell, state)
    else:
        st.caption(
            "Inspect coverage, missingness and model comparisons in Order flow; "
            "inspect regime assessments and assignments in Market regimes."
        )
    if (
        regime_ids.get("assessment_id")
        and regime_ids.get("decision_id")
        and st.button("Configure a linked regime feature study", key=_RESEARCH + "followup")
    ):
        _seed_linked_regime_study(
            st,
            group,
            cell,
            {
                "regime_promotion_decision_id": regime_ids["decision_id"],
                "required_capability_assessment_id": regime_ids["assessment_id"],
            },
        )


def render_pipeline_launch(st, draft, resolution, roots, *, errors):
    """Keep strategy workflows separate from exact-source model research."""
    workflow = st.radio(
        "Workflow",
        ("Feature and model research", "Strategy evaluation"),
        key="ifvg_research_workflow_kind",
    )
    if workflow == "Feature and model research":
        render_research_configuration(st, roots=resolution.roots, draft=draft)
        return
    _render_strategy_pipeline_launch(st, draft, resolution, roots, errors=errors)


def _render_strategy_pipeline_launch(st, draft, resolution, roots, *, errors):
    import ifvg_pipeline_tab as pipeline

    fields = {
        "full_plan": False,
        "bundle": None,
        "label_policy": None,
        "model_protocol": None,
        "max_workers": 1,
        "regime_on": False,
    }
    old = draft.steps["review"].get("pipeline_settings")
    draft.steps["review"]["pipeline_settings"] = fields
    if old != fields and draft.display_name:
        from alpha_lab.agents.data_infra.ifvg.study_drafts import save_draft

        save_draft(Path(roots["draft_root"]), draft)
    if st.button("Run workflow", type="primary", disabled=bool(errors)):
        st.session_state.pop(pipeline._SELECTED_KEY, None)
        try:
            pipeline._freeze_and_launch_pipeline(st, roots, draft, fields)
        except Exception:
            st.error(
                "The workflow could not start. Its saved settings are retained; "
                "refresh and check readiness before retrying."
            )
            return
        selected = st.session_state.get(pipeline._SELECTED_KEY)
        if selected:
            st.session_state["ifvg_workspace_selected_study"] = selected
            st.session_state["ifvg_workspace_screen"] = "detail"
            st.rerun()


def _regime_fields(st, bundle, stored, roots):
    import ifvg_pipeline_tab as pipeline
    from ifvg_research_wizard import _pick

    from alpha_lab.agents.data_infra.ifvg.ml.regime_algorithms import REGIME_ALGORITHM_REGISTRY
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import (
        COMPARISON_CLASSES,
        DESCRIPTIVE_CLASSES,
        SUPERVISED_CLASSES,
    )

    enabled = st.checkbox(
        "Include market regime research",
        value=bool(stored.get("regime_on")),
        key="ifvg_research_regime_on",
    )
    if not enabled:
        return {"regime_on": False}
    grains = {
        pipeline._REGIME_GRAIN_CANDIDATE: "Entry opportunities",
        pipeline._REGIME_GRAIN_PANEL_5M: "Completed 5-minute bars",
        pipeline._REGIME_GRAIN_PANEL_15M: "Completed 15-minute bars",
    }
    grain = _pick(
        st,
        "Observation interval",
        grains,
        stored.get("regime_grain"),
        "ifvg_research_regime_grain",
        grains.get,
    )
    algorithms = [
        row.algorithm_key
        for row in REGIME_ALGORITHM_REGISTRY.values()
        if row.implementation_status == "implemented"
    ]
    algorithm = _pick(
        st,
        "Clustering method",
        algorithms,
        stored.get("regime_algorithm"),
        "ifvg_research_regime_algorithm",
        research_label,
    )
    if grain != pipeline._REGIME_GRAIN_CANDIDATE:
        inputs = pipeline._panel_numeric_inputs()
    else:
        from alpha_lab.agents.data_infra.ifvg.context_model import categorical_features_for
        from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle

        names = resolve_bundle(bundle).payload.resolved_feature_names
        options = [name for name in names if name not in categorical_features_for(names)]
        inputs = tuple(
            st.multiselect(
                "Regime features",
                options,
                default=[
                    name
                    for name in stored.get(
                        "regime_inputs", pipeline._REGIME_CANDIDATE_INPUT_DEFAULT
                    )
                    if name in options
                ],
                format_func=research_label,
                key="ifvg_research_regime_inputs",
            )
        )
    refits = int(
        st.number_input(
            "Bootstrap refits per fold",
            min_value=1,
            max_value=50,
            value=int(stored.get("regime_bootstrap_refits", 50)),
            key="ifvg_research_regime_refits",
        )
    )
    classes = tuple(
        st.multiselect(
            "Regime comparisons",
            list(COMPARISON_CLASSES),
            default=stored.get("regime_classes", list(DESCRIPTIVE_CLASSES)),
            format_func=research_label,
            key="ifvg_research_regime_classes",
        )
    )
    authority = dict(stored.get("regime_authority") or {})
    if any(value in SUPERVISED_CLASSES for value in classes):
        st.info(
            "Model comparisons require a prior approved regime assessment. The "
            "assessment remains fixed for this study."
        )
        # Restoration preserves exact saved references. New references are
        # selected from verified catalog evidence, never guessed by recency.
        from alpha_lab.agents.data_infra.ifvg.study_providers import list_catalogued_envelope_ids

        for field, label, store in (
            ("regime_promotion_decision_id", "Approved regime decision", "regime_promotions"),
            ("owner_decision_artifact_id", "Research authorization", "owner_decisions"),
            ("required_capability_assessment_id", "Regime assessment", "regime_assessments"),
        ):
            try:
                options = [
                    identity
                    for identity, _name in list_catalogued_envelope_ids(
                        Path(roots["store_root"]), store
                    )
                ]
            except Exception:
                options = []
            if authority.get(field) and authority[field] not in options:
                options.insert(0, authority[field])
            if options:
                labels = {value: f"{label} {index + 1}" for index, value in enumerate(options)}
                authority[field] = _pick(
                    st, label, options, authority.get(field), "ifvg_research_" + field, labels.get
                )
            else:
                st.warning(
                    f"No verified {label.lower()} is available. This comparison cannot run yet."
                )
    return {
        "regime_on": True,
        "regime_grain": grain,
        "regime_algorithm": algorithm,
        "regime_inputs": inputs,
        "regime_bootstrap_refits": refits,
        "regime_classes": classes,
        "regime_authority": authority,
    }


def launch_existing(st, study, roots):
    import ifvg_pipeline_tab as pipeline
    import ifvg_study_wizard as wizard

    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
        pipeline_entry_key_for_charter,
        resolve_registered_runner_entry,
        runner_entry_key_for_charter,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import load_charter

    try:
        charter = load_charter(study.store_root, study.charter_id)
        if charter is None:
            raise ValueError("Missing charter")
        is_pipeline = study.kind == "pipeline"
        entry = (
            pipeline_entry_key_for_charter(charter)
            if is_pipeline
            else runner_entry_key_for_charter(charter)
        )
        if entry is None:
            st.warning(
                "The required research runner is unavailable. Your saved study can "
                "be resumed after that dependency is ready."
            )
            return
        resolve_registered_runner_entry(entry)
        # Workers revalidate frozen authorization and reuse only verified results.
        script = "ifvg_pipeline_job.py" if is_pipeline else "ifvg_search_job.py"
        key = study.key if is_pipeline else study.charter_id
        state_root = Path(roots["pipeline_state_root" if is_pipeline else "state_root"])
        command = [
            sys.executable,
            str(Path(__file__).with_name(script)),
            "start" if study.status == "Ready" else "resume",
            "--pipeline-id" if is_pipeline else "--search-id",
            key,
            "--store-root",
            str(study.store_root),
            "--state-root",
            str(state_root),
            "--runner-entry-key",
            entry,
        ]
        if is_pipeline:
            command += ["--max-workers", "1"]
        spawn = pipeline._spawn_pipeline_job if is_pipeline else wizard._spawn_search_job
        spawn(command)
    except Exception:
        st.error(
            "The study could not start. Its saved settings are retained. Resolve "
            "the runner or authorization issue, then try again."
        )
        return
    st.info("Run requested. Refresh progress to see the worker's saved status.")


def render_pipeline_results(st, study, roots):
    from ifvg_research_results import load_pipeline_bundle, render_bundle

    from alpha_lab.agents.data_infra.ifvg.study_providers import load_ladder_diagnostics

    try:
        bundle = load_pipeline_bundle(study)
        diagnostics = load_ladder_diagnostics(study.store_root, study.state)
    except Exception:
        st.error(
            "Saved workflow evidence failed verification. Restore it before "
            "interpreting this study."
        )
        return
    rungs = (diagnostics or {}).get("rungs") or {}
    if not rungs:
        render_bundle(st, bundle, study=study, roots=roots)
    if rungs:
        for limitation in bundle.get("limitations", ()):
            st.warning(limitation)
        rows = []
        for key, rung in rungs.items():
            report = rung.get("prediction_report") or {}
            rows.append(
                {
                    "Model": research_label(key),
                    "Out-of-sample rows": report.get("count"),
                    "Brier score": report.get("brier_score"),
                    "Brier skill": report.get("brier_skill_score"),
                    "AUC": report.get("auc")
                    if isinstance(report.get("auc"), (float, int))
                    else None,
                }
            )
        st.subheader("Model comparison")
        st.caption(
            "Out-of-sample model evidence · Lower Brier score and higher Brier "
            "skill are better. Undefined AUC remains unavailable."
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        _model_details(st, rungs, diagnostics)
        if st.checkbox("Show underlying strategy results"):
            render_bundle(st, bundle, study=study, roots=roots)
    _catalog_action(st, study, roots)


def _model_details(st, rungs, diagnostics, *, preferred_protocol=None, selection_key=None):
    from ifvg_lab_charts import build_reliability_figure
    from ifvg_research_context import _human_frame
    from ifvg_research_results import metric_cards

    from alpha_lab.agents.data_infra.ifvg.presentation.metric_registry import evaluate_metric

    protocols = list(rungs)
    if preferred_protocol in rungs:
        protocols.remove(preferred_protocol)
        protocols.insert(0, preferred_protocol)
        st.caption(
            "The frozen headline protocol is first in review order. All saved model "
            "protocols remain available."
        )
    elif preferred_protocol:
        st.info("The selected headline protocol has no saved model report yet.")
    selected = st.selectbox(
        "Model to inspect", protocols, format_func=research_label, key=selection_key
    )
    rung = rungs[selected]
    report = rung.get("prediction_report") or {}
    keys = ("brier_score", "brier_skill_score", "log_loss", "auc")
    metric_cards(st, [evaluate_metric(key, report.get(key)) for key in keys], keys)
    if not report.get("count"):
        st.warning("This model has no out-of-sample predictions. Its performance is inconclusive.")
    calibration = pd.DataFrame(report.get("reliability_bins") or ())
    if not calibration.empty:
        st.plotly_chart(build_reliability_figure(calibration), width="stretch")
        with st.expander("Calibration data table"):
            st.dataframe(_human_frame(calibration), hide_index=True, width="stretch")
    if st.checkbox("Show model research details"):
        st.subheader("Walk-forward folds")
        st.dataframe(
            _human_frame(pd.DataFrame(rung.get("fold_reports") or ())),
            hide_index=True,
            width="stretch",
        )
        st.subheader("Paired model differences")
        deltas = diagnostics.get("paired_deltas") or {}
        rows = []
        for name, value in deltas.items():
            if isinstance(value, dict):
                rows.append({"Comparison": research_label(name), **value})
        if rows:
            st.dataframe(_human_frame(pd.DataFrame(rows)), hide_index=True, width="stretch")
        else:
            st.info("Paired uncertainty evidence is unavailable.")
        st.caption(
            "Feature importance is not included in this saved workflow report. "
            "Context feature studies provide it when available."
        )


def _catalog_action(st, study, roots):
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        activate_pipeline_result,
        publication_state_sha256,
        run_publication_gates,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import resolve_store_namespace

    if study.scope != "research":
        return
    namespace = resolve_store_namespace(study.store_root)
    if namespace.status != "verified":
        return
    state_digest = publication_state_sha256(dict(study.state))
    cache_key = f"ifvg_research_catalog_{study.key}_{namespace.store_namespace_id}_{state_digest}"
    if st.button("Check research catalog eligibility"):
        try:
            st.session_state[cache_key] = run_publication_gates(
                Path(roots["pipeline_state_root"]), study.key, store_root=study.store_root
            )
        except Exception:
            st.error(
                "Eligibility could not be verified. This study has not been added to the catalog."
            )
    gates = st.session_state.get(cache_key)
    if gates and all(value is True for value in gates.values()):
        if st.button("Add to research catalog", type="primary"):
            try:
                activate_pipeline_result(
                    Path(roots["pipeline_state_root"]),
                    study.key,
                    store_root=study.store_root,
                    expected_store_namespace_id=namespace.store_namespace_id,
                )
                st.success("Study added to the research catalog.")
            except Exception:
                st.error("The study is no longer eligible. Refresh and check eligibility again.")
    elif gates:
        st.warning(
            "Required evidence checks have not passed. The study cannot be added to "
            "the research catalog yet."
        )
