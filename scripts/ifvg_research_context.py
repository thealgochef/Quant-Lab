"""Context feature studies with concise results and exact saved-case navigation."""

from __future__ import annotations

import ifvg_lab_tab as lab
import pandas as pd

from alpha_lab.agents.data_infra.ifvg.presentation.workspace import human_name, profile_name


def render_context_study(st, *, run_id=None):
    st.header("Context feature study")
    if run_id:
        try:
            stored = lab.load_context_experiment_run(
                run_id, base_dir=lab.ROOT / lab.CONTEXT_RUN_STORE
            )
        except Exception:
            st.error(
                "Saved model evidence could not be verified. Restore it before "
                "interpreting this study."
            )
            return
        render_context_result(st, stored)
        return
    st.write("Measure whether context features improve predictions of hypothetical entry outcomes.")
    selected = lab._load_selected_pair(st, key="ifvg_research_context_pair")
    if selected is None:
        return
    pair, entry = selected
    view = lab.build_candidate_feature_view(pair)
    from alpha_lab.agents.data_infra.ifvg.presentation.labels import FEATURE_TIER_LABELS

    tier = st.selectbox(
        "Feature tier",
        tuple(lab.ContextFeatureTier),
        format_func=lambda value: FEATURE_TIER_LABELS.get(value.value, value.value),
        key="ifvg_research_context_tier",
    )
    target = st.selectbox(
        "Hypothetical target",
        ("1R", "1.5R", "2R", "Fixed stop and target"),
        key="ifvg_research_context_target",
    )
    cohort = st.selectbox(
        "Research population",
        ("prior_research", "all_development", "exposed_development"),
        format_func=lab.context_cohort_label,
        key="ifvg_research_context_cohort",
    )
    stop, profit = None, None
    if target == "Fixed stop and target":
        stop = int(st.number_input("Stop distance (ticks)", min_value=1, value=16))
        profit = int(st.number_input("Target distance (ticks)", min_value=1, value=16))
    filters = lab.context_cohort_filters(view, cohort)
    frame = lab.apply_observation_filters(view, filters)
    st.caption(f"{len(frame)} entry opportunities · Fixed walk-forward evaluation")
    if tier is lab.ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL:
        st.warning(
            "The 240-minute features are experimental and excluded from primary comparisons."
        )
    if tier is lab.ContextFeatureTier.M3 and lab.m3_cohort_status(frame) != "model_eligible":
        st.warning(
            "This population does not provide enough eligible examples for this "
            "feature tier. Model evidence may remain inconclusive."
        )
    if st.button("Run context study", type="primary"):
        try:
            label = (
                lab.IfvgContextLabelConfig(
                    label_family="fixed_sl_tp", fixed_stop_ticks=stop, fixed_target_ticks=profit
                )
                if stop is not None
                else lab.IfvgContextLabelConfig(reward_r=float(target[:-1]))
            )
            config = lab.IfvgContextExperimentConfig(
                dataset=lab.IfvgContextExperimentDatasetConfig(
                    artifact_pair=pair.reference, profile_name=entry["profile_name"]
                ),
                feature_tier=tier,
                label=label,
                observation_filters=filters,
            )
            with st.spinner("Evaluating context features…"):
                saved = lab.run_and_catalog_context_experiment(
                    pair,
                    config,
                    view_store=lab.ROOT / lab.CONTEXT_VIEW_STORE,
                    run_store=lab.ROOT / lab.CONTEXT_RUN_STORE,
                    catalog_path=lab.ROOT / "data/ifvg_experiments/context_v1_catalog.json",
                )
            st.session_state["ifvg_research_context_saved"] = saved.stored_run.result.run_id
        except Exception:
            st.error(
                "The model study could not finish. Check data readiness and "
                "selected research settings before trying again."
            )
    saved_id = st.session_state.get("ifvg_research_context_saved")
    if saved_id:
        stored = lab.load_context_experiment_run(
            saved_id, base_dir=lab.ROOT / lab.CONTEXT_RUN_STORE
        )
        render_context_result(st, stored)


def _human_frame(frame):
    from ifvg_research_review import words

    table = frame.copy()
    keep = [
        key
        for key in table.columns
        if not any(token in key.lower() for token in ("_id", "hash", "path", "manifest", "source"))
    ]
    table = table[keep]
    # Nested payloads are developer evidence. Research tables contain scalar
    # measurements only, never the repr of a report or scientific identity.
    table = table[
        [
            key
            for key in table
            if not table[key].map(lambda value: isinstance(value, (dict, list, tuple, set))).any()
        ]
    ]
    for key in table.select_dtypes(include=["object", "string"]).columns:
        table[key] = table[key].map(words)
    table.columns = [words(key).capitalize() for key in table.columns]
    return table


def render_context_result(st, stored):
    from ifvg_research_results import metric_cards, reading_table, research_sentence

    from alpha_lab.agents.data_infra.ifvg.presentation.rollups import SECTION_LABELS

    result = stored.result.model_dump(mode="json")
    adapted = lab.adapt_candidate_research_report(result["candidate_research_report"])
    st.subheader("Model outcome")
    st.caption(
        "Hypothetical outcome labels · Actual execution and reviewer judgment are "
        "separate evidence."
    )
    for rollup in lab.decision_summary(result):
        st.write(
            f"**{SECTION_LABELS[rollup.section]}** · {rollup.chip} · "
            f"{research_sentence(rollup.main_reason)}"
        )
    readings = lab.candidate_readings(adapted)
    metric_cards(st, readings.values(), ("brier_score", "brier_skill_score", "log_loss", "auc"))
    for reading in lab._candidate_intervals(adapted):
        st.caption(research_sentence(reading.interpretation))
    reliability = adapted["reliability"]
    if not reliability.empty:
        st.plotly_chart(lab.build_reliability_figure(reliability), width="stretch")
        with st.expander("Calibration data table"):
            st.dataframe(_human_frame(reliability), hide_index=True, width="stretch")
    else:
        st.info("Calibration evidence is unavailable for this study.")
    if st.checkbox("Show model research details", key="ifvg_context_details"):
        st.dataframe(reading_table(readings.values()), hide_index=True, width="stretch")
        for title, frame in (
            ("Walk-forward folds", adapted["folds"]),
            ("Feature importance", lab.importance_top(adapted, n=10)),
            ("Uncertainty", adapted["uncertainty"]),
            (
                "Feature coverage",
                lab.adapt_feature_coverage_report(result["feature_coverage_report"])["features"],
            ),
        ):
            with st.expander(title):
                if frame.empty:
                    st.info("This evidence is not available.")
                else:
                    st.dataframe(_human_frame(frame), hide_index=True, width="stretch")
        actual = lab.execution_readings(
            lab.adapt_actual_execution_report(result["actual_execution_report"])
        )
        st.subheader("Actual execution")
        st.dataframe(reading_table(actual.values()), hide_index=True, width="stretch")
    predictions = stored.predictions
    if not predictions.empty and "candidate_id" in predictions:
        st.subheader("Inspect a supporting opportunity")
        table = _human_frame(predictions)
        st.dataframe(table, hide_index=True, width="stretch", height=250)
        row = (
            int(
                st.number_input("Opportunity row", min_value=1, max_value=len(predictions), value=1)
            )
            - 1
        )
        if st.button("Review selected opportunity", type="primary"):
            from ifvg_ui_common import queue_replay_drilldown

            options = lab._ready_pair_options()
            ref = stored.config.dataset.artifact_pair
            label = next(
                (
                    label
                    for label, entry in options.items()
                    if entry.get("v2_artifact_id") == ref.v2.artifact_id
                    and entry.get("v3_artifact_id") == ref.v3.artifact_id
                    and entry.get("v2_manifest_payload_sha256") == ref.v2.manifest_payload_sha256
                    and entry.get("v3_manifest_payload_sha256") == ref.v3.manifest_payload_sha256
                ),
                None,
            )
            if label is None:
                st.warning(
                    "The exact research data used by this study is unavailable for Trade review."
                )
            elif queue_replay_drilldown(
                st, "candidate_id", str(predictions.iloc[row]["candidate_id"]), pair_label=label
            ):
                st.rerun()
    _compare(st, stored)


def _compare(st, stored):
    with st.expander("Compare saved model studies"):
        entries = lab.list_context_run_catalog(
            catalog_path=lab.ROOT / "data/ifvg_experiments/context_v1_catalog.json"
        )
        ids = [str(row["run_id"]) for row in entries if row.get("run_id") != stored.result.run_id]
        if not ids:
            st.info("Another saved model study is needed for a comparison.")
            return
        labels = {value: f"Model study {index + 1}" for index, value in enumerate(ids)}
        selected = st.selectbox("Compare with", ids, format_func=labels.get)
        try:
            other = lab.load_context_experiment_run(
                selected, base_dir=lab.ROOT / lab.CONTEXT_RUN_STORE
            )
            reconciliation = lab.reconcile_context_runs(stored, other)
        except Exception:
            st.error("The comparison evidence could not be verified.")
            return
        st.caption(
            profile_name(other.config.dataset.profile_name)
            + " · "
            + human_name(str(other.config.feature_tier), "Context features")
        )
        if not reconciliation.compatible_for_metric_delta:
            st.warning(
                "These studies do not have compatible populations or evaluation "
                "settings. Quantitative differences are unavailable."
            )
            return
        st.dataframe(
            _human_frame(lab._metric_delta_frame(stored, other)), hide_index=True, width="stretch"
        )
        paired = lab._metric_deltas(stored, other).get("paired_trading_day_brier_loss_delta")
        if paired:
            st.dataframe(_human_frame(pd.DataFrame([paired])), hide_index=True, width="stretch")
