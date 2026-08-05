"""IFVG Lab: immutable context experiments, exact replay, and data audit."""

from __future__ import annotations

import json
import re
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from alpha_lab.agents.data_infra.ifvg.artifact_io import (  # noqa: E402
    ArtifactVerificationError,
    VerifiedIfvgPair,
    load_verified_ifvg_pair,
    load_verified_label_source_bars,
)
from alpha_lab.agents.data_infra.ifvg.config import (  # noqa: E402
    V2_DATASET_DIR,
    V3_DATASET_DIR,
)
from alpha_lab.agents.data_infra.ifvg.context_contracts import (  # noqa: E402
    ContextRecordTable,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (  # noqa: E402
    PROFILE_CAPABILITY_REGISTRY,
    ArtifactPreparationStatus,
    ContextFeatureTier,
    IfvgContextExperimentConfig,
    IfvgContextExperimentDatasetConfig,
    IfvgContextLabelConfig,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_service import (  # noqa: E402
    context_cohort_filters,
    context_cohort_label,
    describe_observation_cohort,
    run_and_catalog_context_experiment,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (  # noqa: E402
    apply_observation_filters,
    build_candidate_feature_view,
    m3_cohort_status,
)
from alpha_lab.agents.data_infra.ifvg.context_report_adapters import (  # noqa: E402
    adapt_actual_execution_report,
    adapt_candidate_research_report,
    adapt_feature_coverage_report,
    adapt_reconciliation_report,
)
from alpha_lab.agents.data_infra.ifvg.context_run_store import (  # noqa: E402
    CONTEXT_RUN_STORE,
    CONTEXT_VIEW_STORE,
    list_context_run_catalog,
    load_context_experiment_run,
    reconcile_context_runs,
)
from alpha_lab.agents.data_infra.ifvg.context_statistics import (  # noqa: E402
    paired_tier_delta_report,
)
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.experiment import (  # noqa: E402
    list_experiments as list_legacy_experiments,
)
from alpha_lab.agents.data_infra.ifvg.preparation import (  # noqa: E402
    PAIR_CATALOG_PATH,
    PREPARATION_JOB_ROOT,
    read_preparation_state,
)

# Compatibility alias for the pre-restoration AppTest seam. It is used only
# by the caveated read-only legacy panel.
list_experiments = list_legacy_experiments

_STATE_PREFIX = "ifvg_context_v1_"
_WINDOW_PATH = re.compile(r"(?:[A-Za-z]:\\|/)[^\s'\"]+")
_SECRET = re.compile(r"(?i)(token|secret|api[_-]?key)\s*[:=]\s*[^\s,}]+")


def _sanitize_error(error: BaseException) -> str:
    value = _WINDOW_PATH.sub("[path redacted]", str(error))
    return _SECRET.sub(r"\1=[redacted]", value)[:800]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("IFVG catalog/report is unreadable") from error


def _pair_catalog() -> dict[str, dict[str, Any]]:
    path = ROOT / PAIR_CATALOG_PATH
    if not path.exists():
        return {}
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError("IFVG pair catalog must be an object")
    return value


def _ready_pair_options() -> dict[str, dict[str, Any]]:
    result = {}
    for profile, entry in sorted(_pair_catalog().items()):
        if entry.get("preparation_status") != ArtifactPreparationStatus.CONTEXT_READY.value:
            continue
        v2_id = str(entry.get("v2_artifact_id", ""))
        v3_id = str(entry.get("v3_artifact_id", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", v2_id) or not re.fullmatch(
            r"[0-9a-f]{64}", v3_id
        ):
            continue
        label = f"{profile} · v2 {v2_id[:12]} · v3 {v3_id[:12]}"
        result[label] = {"profile_name": profile, **entry}
    return result


@st.cache_resource(show_spinner=False)
def _cached_pair(
    v2_id: str,
    v2_manifest_hash: str,
    v3_id: str,
    v3_manifest_hash: str,
) -> VerifiedIfvgPair:
    pair = load_verified_ifvg_pair(
        v2_root=ROOT / V2_DATASET_DIR,
        v2_artifact_id=v2_id,
        v3_root=ROOT / V3_DATASET_DIR,
        v3_artifact_id=v3_id,
    )
    if pair.v2.reference.manifest_payload_sha256 != v2_manifest_hash:
        raise ArtifactVerificationError("cataloged v2 manifest hash changed")
    if pair.v3.reference.manifest_payload_sha256 != v3_manifest_hash:
        raise ArtifactVerificationError("cataloged v3 manifest hash changed")
    return pair


def _load_selected_pair(st_module, *, key: str) -> tuple[VerifiedIfvgPair, dict] | None:
    options = _ready_pair_options()
    if not options:
        profile = "ifvg_v2_doc_default_fresh_static_1r"
        state = None
        with suppress(Exception):
            state = read_preparation_state(ROOT / PREPARATION_JOB_ROOT / profile)
        st_module.info(
            "No context-ready paired artifact is cataloged. Preparation runs as an "
            "external persisted process; this page never starts source replay."
        )
        if state is None:
            st_module.caption("Preparation status: not_prepared")
        else:
            st_module.caption(
                f"Preparation status: {state.status.value}"
                + (f" · current date {state.current_date}" if state.current_date else "")
                + (f" · error {state.error_code}" if state.error_code else "")
            )
        st_module.markdown("**Start or resume preparation externally**")
        st_module.code(
            "python scripts/ifvg_preparation_job.py start "
            "--profile ifvg_v2_doc_default_fresh_static_1r",
            language="powershell",
        )
        st_module.markdown("**Check persisted preparation status**")
        st_module.code(
            "python scripts/ifvg_preparation_job.py status "
            "--profile ifvg_v2_doc_default_fresh_static_1r",
            language="powershell",
        )
        return None
    selected = st_module.selectbox("Verified artifact pair", tuple(options), key=key)
    entry = options[selected]
    try:
        pair = _cached_pair(
            entry["v2_artifact_id"],
            entry["v2_manifest_payload_sha256"],
            entry["v3_artifact_id"],
            entry["v3_manifest_payload_sha256"],
        )
    except Exception as error:
        st_module.error(f"Artifact verification failed: {_sanitize_error(error)}")
        return None
    return pair, entry


def _render_capability_registry(st_module) -> None:
    rows = [
        {
            "profile": capability.profile_name,
            "status": capability.status.value,
            "reason": capability.reason,
        }
        for capability in PROFILE_CAPABILITY_REGISTRY.values()
    ]
    st_module.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def _legacy_read_only(st_module) -> None:
    with st_module.expander("Legacy results — read-only and not comparable"):
        st_module.warning(
            "Legacy studies use obsolete joins, folds, and persistence. They cannot be "
            "rerun, deleted, promoted, or compared with context-v1 runs."
        )
        try:
            legacy = list_experiments()
        except Exception as error:
            st_module.caption(f"Legacy catalog unavailable: {_sanitize_error(error)}")
            return
        if not legacy:
            st_module.caption("No legacy saved studies.")
            return
        st_module.dataframe(
            pd.DataFrame(
                {
                    "name": item.get("name"),
                    "legacy_id": item.get("experiment_hash"),
                    "classification": "legacy_read_only",
                }
                for item in legacy
            ),
            hide_index=True,
            use_container_width=True,
        )


def _cohort_filters(view, cohort: str) -> dict[str, tuple[str, ...]]:
    days = tuple(sorted(view.frame["trading_day"].dropna().astype(str).unique()))
    if cohort == "Prior research (through Apr 30)":
        days = tuple(day for day in days if day <= "2026-04-30")
    elif cohort == "Exposed development (May 1–Jun 10)":
        days = tuple(day for day in days if "2026-05-01" <= day <= "2026-06-10")
    return {"trading_day": days}


def _config_diff(left: dict[str, Any], right: dict[str, Any]) -> pd.DataFrame:
    def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
        if not isinstance(value, dict):
            return {prefix: value}
        output: dict[str, Any] = {}
        for key, child in sorted(value.items()):
            name = f"{prefix}.{key}" if prefix else str(key)
            output.update(flatten(child, name))
        return output

    left_flat = flatten(left)
    right_flat = flatten(right)

    def display(value: Any) -> str:
        if value is None:
            return "(none)"
        if isinstance(value, list | tuple):
            return json.dumps(list(value), sort_keys=True, default=str)
        return str(value)

    return pd.DataFrame(
        {
            "field": field,
            "left": display(left_flat.get(field)),
            "right": display(right_flat.get(field)),
            "differs": left_flat.get(field) != right_flat.get(field),
        }
        for field in sorted(set(left_flat) | set(right_flat))
    )


def _metric_deltas(left, right) -> dict[str, Any]:
    left_metrics = (
        left.result.candidate_research_report.get("model", {}).get("metrics", {})
    )
    right_metrics = (
        right.result.candidate_research_report.get("model", {}).get("metrics", {})
    )
    scalar = {}
    for name in ("brier_score", "brier_skill_score", "log_loss", "auc"):
        left_value = left_metrics.get(name)
        right_value = right_metrics.get(name)
        scalar[name] = {
            "left": left_value,
            "right": right_value,
            "delta_right_minus_left": (
                None
                if left_value is None or right_value is None
                else float(right_value) - float(left_value)
            ),
        }
    paired = None
    if not left.predictions.empty and not right.predictions.empty:
        left_rows = left.predictions.copy()
        right_rows = right.predictions.copy()
        left_rows["brier_loss"] = (
            pd.to_numeric(left_rows["target"], errors="raise")
            - pd.to_numeric(left_rows["probability"], errors="raise")
        ) ** 2
        right_rows["brier_loss"] = (
            pd.to_numeric(right_rows["target"], errors="raise")
            - pd.to_numeric(right_rows["probability"], errors="raise")
        ) ** 2
        paired = paired_tier_delta_report(
            left_rows,
            right_rows,
            value_column="brier_loss",
        )
    return {
        "direction": "right_minus_left",
        "scalar_metrics": scalar,
        "paired_trading_day_brier_loss_delta": paired,
    }


def _metric_delta_frame(left, right) -> pd.DataFrame:
    payload = _metric_deltas(left, right)
    return pd.DataFrame(
        {"metric": metric, **values}
        for metric, values in payload["scalar_metrics"].items()
    )


def _target_label(config: IfvgContextExperimentConfig) -> str:
    if config.label.label_family == "fixed_sl_tp":
        return f"SL{config.label.fixed_stop_ticks}/TP{config.label.fixed_target_ticks}"
    return f"R{config.label.reward_r:.1f}"


def _history_labels(entries: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in entries:
        run_id = str(item["run_id"])
        try:
            stored = load_context_experiment_run(
                run_id,
                base_dir=ROOT / CONTEXT_RUN_STORE,
            )
            config = stored.config
            name = item.get("display_name") or "Unnamed run"
            cohort = context_cohort_label(
                describe_observation_cohort(config.observation_filters)
            )
            label = (
                f"{name} · {config.feature_tier.value} · {_target_label(config)} · "
                f"{cohort} · {stored.result.status} · {run_id[:12]}"
            )
        except Exception:
            label = (
                f"{item.get('display_name') or 'Unnamed run'} · verification failed · "
                f"{run_id[:12]}"
            )
        labels[label] = run_id
    return labels


def _render_result_json_legacy(st_module, result: dict[str, Any]) -> None:
    st_module.caption(f"Immutable run `{result['run_id']}` · {result['status']}")
    candidate_tab, execution_tab, coverage_tab, audit_tab = st_module.tabs(
        ["Candidate research", "Actual execution", "Feature coverage", "Reconciliation"]
    )
    with candidate_tab:
        st_module.info("Counterfactual candidate labels; actual trade outcomes are excluded.")
        st_module.json(result["candidate_research_report"], expanded=False)
    with execution_tab:
        st_module.info("Only source v2 decisions and executed trades appear here.")
        st_module.json(result["actual_execution_report"], expanded=False)
    with coverage_tab:
        st_module.json(result["feature_coverage_report"], expanded=False)
    with audit_tab:
        st_module.json(result["reconciliation_audit_report"], expanded=False)


def _display_metric(value: Any, *, percent: bool = False) -> str:
    if value is None:
        return "—"
    if percent:
        return f"{float(value):.1%}"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_candidate_report(st_module, report: dict[str, Any]) -> None:
    adapted = adapt_candidate_research_report(report)
    st_module.info("Counterfactual candidate labels; actual trade outcomes are excluded.")
    kpis = adapted["kpis"]
    columns = st_module.columns(3)
    columns[0].metric("Candidates", kpis["candidate_count"])
    columns[1].metric("Resolved", kpis["resolved_candidate_count"])
    columns[2].metric("Censored", kpis["censored_candidate_count"])
    if adapted["status"] != "complete":
        st_module.warning(f"Model status: {adapted['status']}")
    metrics = adapted["metrics"]
    metric_columns = st_module.columns(4)
    metric_columns[0].metric("Brier", _display_metric(metrics["brier_score"]))
    metric_columns[1].metric(
        "Brier skill", _display_metric(metrics["brier_skill_score"])
    )
    metric_columns[2].metric("Log loss", _display_metric(metrics["log_loss"]))
    metric_columns[3].metric(
        "AUC", _display_metric(metrics["auc"]), help=metrics.get("auc_reason")
    )
    label_col, censor_col = st_module.columns(2)
    if not adapted["labels"].empty:
        label_col.markdown("**Candidate resolutions**")
        label_col.bar_chart(adapted["labels"].set_index("label"), height=220)
    if not adapted["censoring"].empty:
        censor_col.markdown("**Censoring reasons**")
        censor_col.bar_chart(adapted["censoring"].set_index("reason"), height=220)
    reliability = adapted["reliability"]
    if not reliability.empty:
        reliability = reliability.dropna(
            subset=["mean_probability", "observed_rate"], how="any"
        )
    if not reliability.empty:
        st_module.markdown("**Reliability**")
        reliability = reliability.copy()
        reliability["ideal"] = reliability["mean_probability"]
        st_module.line_chart(
            reliability.set_index("mean_probability")[["observed_rate", "ideal"]],
            height=280,
        )
    if not adapted["thresholds"].empty:
        st_module.markdown("**Threshold coverage and net R**")
        st_module.bar_chart(
            adapted["thresholds"].set_index("threshold")[
                ["coverage_fraction", "net_r_sum"]
            ],
            height=280,
        )
        st_module.dataframe(
            adapted["thresholds"], hide_index=True, use_container_width=True, height=220
        )
    st_module.markdown("**Walk-forward folds**")
    if adapted["folds"].empty:
        st_module.caption("No valid model folds were produced.")
    else:
        st_module.dataframe(
            adapted["folds"], hide_index=True, use_container_width=True, height=300
        )
    if not adapted["uncertainty"].empty:
        st_module.markdown("**Uncertainty intervals**")
        st_module.dataframe(
            adapted["uncertainty"], hide_index=True, use_container_width=True
        )
    if not adapted["feature_importance"].empty:
        st_module.markdown("**Feature importance (descriptive only)**")
        importance = adapted["feature_importance"].sort_values(
            "permutation_importance_mean", ascending=False
        )
        st_module.bar_chart(
            importance.set_index("feature")[["permutation_importance_mean"]],
            height=320,
        )
        st_module.dataframe(
            importance, hide_index=True, use_container_width=True, height=300
        )
    with st_module.expander("Raw candidate report — audit", expanded=False):
        st_module.json(report, expanded=False)


def _queue_verifier_jump(st_module, kind: str, value: str) -> None:
    """Queue an exact-ID jump for the Replay / Verifier chart (no fallback)."""
    try:
        from ifvg_verifier_tab import queue_jump

        queue_jump(kind, value)
    except Exception as error:
        st_module.caption(f"Verifier jump unavailable: {_sanitize_error(error)}")
    else:
        st_module.toast(
            f"Loaded {kind} {str(value)[:12]}… — open the Replay / Verifier tab."
        )


def _render_execution_report(
    st_module, report: dict[str, Any], *, key_suffix: str = ""
) -> None:
    adapted = adapt_actual_execution_report(report)
    st_module.info(
        "Only source-v2 decisions and executed trades appear here; candidate "
        "counterfactual outcomes are excluded."
    )
    kpis = adapted["kpis"]
    first = st_module.columns(4)
    first[0].metric("Eligible decisions", kpis["eligible_decision_count"])
    first[1].metric("Executed trades", kpis["executed_trade_count"])
    first[2].metric("Resolved trades", kpis["resolved_trade_count"])
    first[3].metric("Win rate", _display_metric(kpis["win_rate"], percent=True))
    second = st_module.columns(4)
    second[0].metric("Total R", _display_metric(kpis["total_realized_r"]))
    second[1].metric("Max drawdown R", _display_metric(kpis["max_drawdown_r"]))
    second[2].metric(
        "Total $ (1 NQ)", _display_metric(kpis["total_realized_dollars"])
    )
    second[3].metric(
        "Max drawdown $", _display_metric(kpis["max_drawdown_dollars"])
    )
    equity = adapted["equity"]
    if equity.empty:
        st_module.caption("No resolved source-v2 trades are available for an equity curve.")
    else:
        chart_col, dollar_col = st_module.columns(2)
        chart_col.markdown("**R equity and drawdown**")
        chart_col.line_chart(equity[["cumulative_r", "drawdown_r"]], height=280)
        dollar_col.markdown("**Dollar equity and drawdown (one NQ contract)**")
        dollar_col.line_chart(
            equity[["cumulative_dollars", "drawdown_dollars"]], height=280
        )
        if "trade_id" in equity.columns:
            st_module.markdown(
                "**Executed trades — select a row to load it in the verifier chart**"
            )
            displayed = equity.reset_index(drop=True)
            selection = st_module.dataframe(
                displayed,
                hide_index=True,
                use_container_width=True,
                height=240,
                on_select="rerun",
                selection_mode="single-row",
                key=f"{_STATE_PREFIX}exec_trades_{key_suffix}",
            )
            selected_rows = getattr(
                getattr(selection, "selection", None), "rows", []
            ) or []
            if selected_rows:
                # The positional index maps through the exact displayed frame.
                trade_id = str(displayed.iloc[selected_rows[0]]["trade_id"])
                _queue_verifier_jump(st_module, "trade_id", trade_id)
    distribution = adapted["r_distribution"]
    if not distribution.empty:
        counts = (
            distribution["realized_r"].round(2).value_counts().sort_index().rename("count")
        )
        st_module.markdown("**Realized R distribution**")
        st_module.bar_chart(counts, height=260)
    if not adapted["uncertainty"].empty:
        st_module.markdown("**Execution uncertainty**")
        st_module.dataframe(
            adapted["uncertainty"], hide_index=True, use_container_width=True
        )
    with st_module.expander("Raw actual-execution report — audit", expanded=False):
        st_module.json(report, expanded=False)


def _render_coverage_report(st_module, report: dict[str, Any]) -> None:
    adapted = adapt_feature_coverage_report(report)
    kpis = adapted["kpis"]
    columns = st_module.columns(4)
    columns[0].metric("Candidates", kpis["candidate_count"])
    columns[1].metric("Features", kpis["feature_count"])
    columns[2].metric("M3 status", kpis["m3_status"] or "—")
    columns[3].metric("240m status", kpis["anchor_240m_status"] or "—")
    features = adapted["features"]
    if features.empty:
        st_module.info("This tier has no reportable features.")
    else:
        chart = features.set_index("feature")[
            ["coverage_fraction", "missing_fraction"]
        ].sort_values("coverage_fraction")
        st_module.markdown("**Feature coverage and missingness**")
        st_module.bar_chart(chart, horizontal=True, height=420)
        st_module.dataframe(
            features, hide_index=True, use_container_width=True, height=360
        )
    if not adapted["flags"].empty:
        st_module.warning("Constant or low-coverage features require interpretation caution.")
        st_module.dataframe(
            adapted["flags"], hide_index=True, use_container_width=True
        )
    pool_col, leg_col = st_module.columns(2)
    pool_col.markdown("**Pool width and separation**")
    pool_col.dataframe(
        adapted["pool_summary"], hide_index=True, use_container_width=True
    )
    leg_col.markdown("**Opposing-leg coverage**")
    leg_col.dataframe(
        adapted["opposing_leg_coverage"], hide_index=True, use_container_width=True
    )
    with st_module.expander("Raw feature-coverage report — audit", expanded=False):
        st_module.json(report, expanded=False)


def _render_reconciliation_report(st_module, report: dict[str, Any]) -> None:
    adapted = adapt_reconciliation_report(report)
    if adapted["passed"]:
        st_module.success("Pair, schemas, hashes, rows, and access evidence reconcile.")
    else:
        st_module.error("Reconciliation did not pass; this run must not be compared.")
    st_module.markdown("**Exact immutable identities**")
    st_module.dataframe(
        adapted["identities"], hide_index=True, use_container_width=True
    )
    gate_col, access_col = st_module.columns(2)
    gate_col.markdown("**Gate cards**")
    gate_col.dataframe(adapted["gates"], hide_index=True, use_container_width=True)
    access_col.markdown("**Access counters**")
    access_col.dataframe(
        adapted["access_counters"], hide_index=True, use_container_width=True
    )
    st_module.markdown("**Context table rows**")
    st_module.dataframe(
        adapted["table_rows"], hide_index=True, use_container_width=True, height=320
    )
    evidence = st_module.columns(2)
    performance = adapted["performance"]
    capacity = adapted["capacity"]
    evidence[0].metric(
        "Replay slowdown",
        _display_metric(performance.get("replay_slowdown_fraction"), percent=True),
    )
    evidence[0].caption(
        "paired p95 "
        + _display_metric(
            performance.get("repeated_run_p95_slowdown_fraction"), percent=True
        )
    )
    evidence[1].metric(
        "Terminal seed bytes", capacity.get("terminal_seed_bytes", "—")
    )
    evidence[1].caption(
        f"largest transition {capacity.get('max_transition_bytes', '—')} bytes"
    )
    with st_module.expander("Raw reconciliation report — audit", expanded=False):
        st_module.json(report, expanded=False)


def _render_result(st_module, result: dict[str, Any]) -> None:
    st_module.caption(f"Immutable run `{result['run_id']}` · {result['status']}")
    candidate_tab, execution_tab, coverage_tab, audit_tab = st_module.tabs(
        ["Candidate research", "Actual execution", "Feature coverage", "Reconciliation"]
    )
    with candidate_tab:
        _render_candidate_report(st_module, result["candidate_research_report"])
    with execution_tab:
        _render_execution_report(
            st_module,
            result["actual_execution_report"],
            key_suffix=str(result["run_id"])[:12],
        )
    with coverage_tab:
        _render_coverage_report(st_module, result["feature_coverage_report"])
    with audit_tab:
        _render_reconciliation_report(st_module, result["reconciliation_audit_report"])


def _run_history(st_module) -> None:
    st_module.markdown("#### Immutable run history")
    try:
        entries = list_context_run_catalog(
            catalog_path=ROOT / "data/ifvg_experiments/context_v1_catalog.json"
        )
    except Exception as error:
        st_module.error(_sanitize_error(error))
        return
    if not entries:
        st_module.caption("No context-v1 runs have been cataloged.")
        return
    labels = _history_labels(entries)
    selected = st_module.selectbox("Saved run", tuple(labels), key=f"{_STATE_PREFIX}history")
    try:
        stored = load_context_experiment_run(
            labels[selected],
            base_dir=ROOT / CONTEXT_RUN_STORE,
        )
    except Exception as error:
        st_module.error(f"Run verification failed: {_sanitize_error(error)}")
        return
    _render_result(st_module, stored.result.model_dump(mode="json"))
    if not stored.predictions.empty and "candidate_id" in stored.predictions.columns:
        with st_module.expander("OOS predictions — select a row to load its candidate"):
            displayed = stored.predictions.reset_index(drop=True)
            selection = st_module.dataframe(
                displayed,
                hide_index=True,
                use_container_width=True,
                height=260,
                on_select="rerun",
                selection_mode="single-row",
                key=f"{_STATE_PREFIX}oos_rows_{stored.result.run_id[:12]}",
            )
            selected_rows = getattr(
                getattr(selection, "selection", None), "rows", []
            ) or []
            if selected_rows:
                candidate_id = str(displayed.iloc[selected_rows[0]]["candidate_id"])
                _queue_verifier_jump(st_module, "candidate_id", candidate_id)
    if len(labels) >= 2:
        compare = st_module.selectbox(
            "Compare with",
            tuple(label for label in labels if label != selected),
            key=f"{_STATE_PREFIX}compare",
        )
        try:
            other = load_context_experiment_run(
                labels[compare],
                base_dir=ROOT / CONTEXT_RUN_STORE,
            )
        except Exception as error:
            st_module.error(f"Comparison run verification failed: {_sanitize_error(error)}")
            return
        reconciliation = reconcile_context_runs(stored, other)
        if reconciliation.compatible_for_metric_delta:
            st_module.success("Runs are compatible for registered metric deltas.")
            st_module.dataframe(
                _metric_delta_frame(stored, other),
                hide_index=True,
                use_container_width=True,
            )
            paired = _metric_deltas(stored, other)[
                "paired_trading_day_brier_loss_delta"
            ]
            if paired is not None:
                with st_module.expander("Paired daily delta details", expanded=False):
                    st_module.json(paired, expanded=False)
        else:
            st_module.warning("Runs are incompatible; quantitative deltas are suppressed.")
        with st_module.expander("Comparison reconciliation audit", expanded=False):
            st_module.json(reconciliation.model_dump(mode="json"), expanded=False)
        st_module.markdown("**Configuration difference**")
        st_module.dataframe(
            _config_diff(
                stored.config.model_dump(mode="json"),
                other.config.model_dump(mode="json"),
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_ifvg_experiments_tab(st_module=st) -> None:
    st_module.subheader("IFVG Lab — Experiments")
    st_module.caption(
        "Deterministic M0/M1/M2/M3 research over exact immutable v2/formula-v2 pairs."
    )
    with st_module.expander("Profile capabilities", expanded=False):
        _render_capability_registry(st_module)
    selected = _load_selected_pair(st_module, key=f"{_STATE_PREFIX}experiment_pair")
    if selected is not None:
        pair, entry = selected
        try:
            view = build_candidate_feature_view(pair)
        except Exception as error:
            st_module.error(f"Experiment inputs are unavailable: {_sanitize_error(error)}")
        else:
            st_module.success(
                f"{entry['preparation_status']} · formula "
                f"{pair.v3.reference.feature_formula_version} · {len(view.frame)} candidates"
            )
            col1, col2, col3 = st_module.columns(3)
            tier = col1.selectbox(
                "Feature tier",
                (
                    ContextFeatureTier.M2.value,
                    ContextFeatureTier.M0.value,
                    ContextFeatureTier.M1_PRIMARY.value,
                    ContextFeatureTier.M3.value,
                    ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL.value,
                ),
                key=f"{_STATE_PREFIX}tier",
            )
            target = col2.selectbox(
                "Counterfactual target",
                ("R1.0", "R1.5", "R2.0", "Fixed SL / TP"),
                key=f"{_STATE_PREFIX}target",
            )
            cohort = col3.selectbox(
                "Observation cohort",
                ("prior_research", "all_development", "exposed_development"),
                format_func=context_cohort_label,
                key=f"{_STATE_PREFIX}cohort",
            )
            fixed_stop_ticks = fixed_target_ticks = None
            if target == "Fixed SL / TP":
                stop_col, target_col = st_module.columns(2)
                fixed_stop_ticks = int(
                    stop_col.number_input(
                        "Fixed stop distance (ticks)",
                        min_value=1,
                        value=16,
                        step=1,
                        key=f"{_STATE_PREFIX}fixed_stop",
                    )
                )
                fixed_target_ticks = int(
                    target_col.number_input(
                        "Fixed target distance (ticks)",
                        min_value=1,
                        value=16,
                        step=1,
                        key=f"{_STATE_PREFIX}fixed_target",
                    )
                )
            cohort_filters = context_cohort_filters(view, cohort)
            cohort_frame = apply_observation_filters(view, cohort_filters)
            selected_m3_status = m3_cohort_status(cohort_frame)
            if ContextFeatureTier(tier) is ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL:
                st_module.warning("240m is experimental and excluded from every primary tier.")
            if (
                ContextFeatureTier(tier) is ContextFeatureTier.M3
                and selected_m3_status != "model_eligible"
            ):
                st_module.warning(selected_m3_status)
            if st_module.button("Run deterministic experiment", key=f"{_STATE_PREFIX}run"):
                try:
                    if target == "Fixed SL / TP":
                        label_config = IfvgContextLabelConfig(
                            label_family="fixed_sl_tp",
                            fixed_stop_ticks=fixed_stop_ticks,
                            fixed_target_ticks=fixed_target_ticks,
                        )
                    else:
                        label_config = IfvgContextLabelConfig(
                            reward_r=float(target.removeprefix("R"))
                        )
                    config = IfvgContextExperimentConfig(
                        dataset=IfvgContextExperimentDatasetConfig(
                            artifact_pair=pair.reference,
                            profile_name=entry["profile_name"],
                        ),
                        feature_tier=ContextFeatureTier(tier),
                        label=label_config,
                        observation_filters=cohort_filters,
                    )
                    with st_module.spinner("Running fixed walk-forward protocol…"):
                        cataloged = run_and_catalog_context_experiment(
                            pair,
                            config,
                            view_store=ROOT / CONTEXT_VIEW_STORE,
                            run_store=ROOT / CONTEXT_RUN_STORE,
                            catalog_path=(
                                ROOT
                                / "data/ifvg_experiments/context_v1_catalog.json"
                            ),
                        )
                    stored = cataloged.stored_run
                    st_module.session_state[f"{_STATE_PREFIX}last_run"] = {
                        "run_id": stored.result.run_id,
                        "manifest_payload_sha256": cataloged.run_manifest_sha256,
                    }
                    if cataloged.reused_run:
                        st_module.info(
                            "The identical immutable run was verified and reused."
                        )
                    else:
                        st_module.success(
                            f"Saved immutable run {stored.result.run_id[:12]}."
                        )
                except Exception as error:
                    st_module.error(f"Experiment failed: {_sanitize_error(error)}")
            last = st_module.session_state.get(f"{_STATE_PREFIX}last_run")
            if isinstance(last, dict):
                try:
                    stored = load_context_experiment_run(
                        str(last["run_id"]),
                        base_dir=ROOT / CONTEXT_RUN_STORE,
                    )
                    if (
                        stored.manifest.get("manifest_payload_sha256")
                        != last.get("manifest_payload_sha256")
                    ):
                        raise ValueError("session run manifest identity changed")
                except Exception as error:
                    st_module.error(f"Run verification failed: {_sanitize_error(error)}")
                else:
                    _render_result(st_module, stored.result.model_dump(mode="json"))
    _run_history(st_module)
    _legacy_read_only(st_module)


def _exact_rows(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return frame.iloc[0:0]
    return frame.loc[frame[column].astype(str) == value].copy()


def render_ifvg_replay_tab(st_module=st) -> None:
    st_module.subheader("IFVG Lab — Replay / Verifier")
    st_module.caption("Candidate selection and every downstream object use exact IDs only.")
    selected = _load_selected_pair(st_module, key=f"{_STATE_PREFIX}replay_pair")
    if selected is None:
        return
    pair, _entry = selected
    links = pair.v3.tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]
    if links.empty:
        st_module.info("The verified pair contains no candidate-stage links.")
        return
    # The visual verifier owns the shared candidate selection when its
    # replay-chart artifact is available; otherwise fall back to the plain
    # exact-ID selector so the inspectors below always work.
    candidate_id = None
    try:
        from ifvg_verifier_tab import render_verifier_section

        candidate_id = render_verifier_section(st_module, pair, _entry)
    except Exception as error:
        st_module.error(f"Visual verifier unavailable: {_sanitize_error(error)}")
    if candidate_id is None:
        candidate_id = st_module.selectbox(
            "Exact candidate ID",
            tuple(sorted(links["candidate_id"].astype(str))),
            key=f"{_STATE_PREFIX}candidate",
        )
    st_module.markdown("**Exact evidence inspectors**")
    link = _exact_rows(links, "candidate_id", candidate_id)
    if len(link) != 1:
        st_module.error("Exact candidate link is missing or duplicated.")
        return
    link_row = link.iloc[0]
    capture_id = str(link_row["context_capture_id"])
    state_id = str(link_row["context_state_id"])
    captures = _exact_rows(
        pair.v3.tables[ContextRecordTable.CONTEXT_CAPTURE],
        "context_capture_id",
        capture_id,
    )
    states = _exact_rows(
        pair.v3.tables[ContextRecordTable.CONTEXT_STATE],
        "context_state_id",
        state_id,
    )
    if len(captures) != 1 or len(states) != 1:
        st_module.error("Exact capture/state reconciliation failed.")
        return
    candidate = _exact_rows(
        pair.v2.tables[RecordTable.ENTRY_CANDIDATE],
        "candidate_id",
        candidate_id,
    )
    decisions = _exact_rows(
        pair.v2.tables[RecordTable.ELIGIBLE_DECISION],
        "candidate_id",
        candidate_id,
    )
    trades = _exact_rows(
        pair.v2.tables[RecordTable.EXECUTED_TRADE],
        "candidate_id",
        candidate_id,
    )
    setup_id = str(link_row["setup_id"])
    lifecycle = pair.v2.tables[RecordTable.SETUP_LIFECYCLE]
    setup_column = (
        "envelope_setup_id" if "envelope_setup_id" in lifecycle else "setup_id"
    )
    lifecycle = _exact_rows(lifecycle, setup_column, setup_id)
    if "trace_ordinal" in lifecycle:
        lifecycle = lifecycle.sort_values("trace_ordinal", kind="mergesort")
    geometry = _exact_rows(
        pair.v2.tables[RecordTable.GEOMETRY_DOSSIER],
        "candidate_id",
        candidate_id,
    )
    summary, context_tab, displacement_tab, pools_tab = st_module.tabs(
        ["Geometry & FSM", "Context snapshot", "Displacement", "Pools & sweeps"]
    )
    with summary:
        st_module.json(
            candidate.to_dict("records")[0] if len(candidate) == 1 else {},
            expanded=False,
        )
        if len(geometry) == 1:
            st_module.markdown("**Exact geometry dossier**")
            st_module.json(geometry.to_dict("records")[0], expanded=False)
        st_module.markdown("**FSM transitions**")
        st_module.dataframe(
            lifecycle,
            hide_index=True,
            use_container_width=True,
        )
        st_module.markdown("**Exact actual linkage**")
        st_module.json(
            {
                "candidate_id": candidate_id,
                "decision_ids": decisions.get(
                    "decision_id", pd.Series(dtype=str)
                ).astype(str).tolist(),
                "trade_ids": trades.get("trade_id", pd.Series(dtype=str)).astype(str).tolist(),
                "geometry_evidence_id": link_row["geometry_evidence_id"],
                "geometry_evidence_cursor": link_row["geometry_evidence_cursor"],
            },
            expanded=False,
        )
        with suppress(Exception):
            bars = load_verified_label_source_bars(pair.v2)
            entry_ts = pd.Timestamp(link_row["feature_as_of_ts"])
            close = pd.to_datetime(bars["close_ts_utc"], utc=True)
            visible = bars.loc[
                (close >= entry_ts - pd.Timedelta(hours=2))
                & (close <= entry_ts + pd.Timedelta(hours=2))
            ].copy()
            if not visible.empty:
                visible = visible.set_index(pd.to_datetime(visible["close_ts_utc"], utc=True))
                st_module.line_chart(visible[["high_ticks", "low_ticks", "close_ticks"]])
    with context_tab:
        st_module.json(
            {
                "link": link.to_dict("records")[0],
                "capture": captures.to_dict("records")[0],
                "state": states.to_dict("records")[0],
            },
            expanded=False,
        )
    capture = captures.iloc[0]
    with displacement_tab:
        ids = capture.get("displacement_window_ids")
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids or ())
        windows = pair.v3.tables[ContextRecordTable.CONTEXT_DISPLACEMENT_WINDOW]
        st_module.dataframe(
            windows.loc[windows["displacement_window_id"].astype(str).isin(map(str, ids))],
            hide_index=True,
            use_container_width=True,
        )
    with pools_tab:
        sweep_ids = capture.get("opposing_leg_sweep_link_ids")
        sweep_ids = (
            sweep_ids.tolist()
            if hasattr(sweep_ids, "tolist")
            else list(sweep_ids or ())
        )
        sweeps = pair.v3.tables[ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK]
        st_module.dataframe(
            sweeps.loc[sweeps["sweep_link_id"].astype(str).isin(map(str, sweep_ids))],
            hide_index=True,
            use_container_width=True,
        )


def render_ifvg_data_audit_tab(st_module=st) -> None:
    st_module.subheader("IFVG Lab — Data & Audit")
    catalog = _pair_catalog()
    if catalog:
        st_module.dataframe(
            pd.DataFrame(
                {"profile": profile, **entry}
                for profile, entry in sorted(catalog.items())
            ),
            hide_index=True,
            use_container_width=True,
        )
    selected = _load_selected_pair(st_module, key=f"{_STATE_PREFIX}audit_pair")
    if selected is None:
        return
    pair, _entry = selected
    v2_tab, v3_tab, access_tab = st_module.tabs(
        ["v2 correctness", "v3 schema & identity", "Access / capacity / validity"]
    )
    with v2_tab:
        for name in (
            "candidate_report.json",
            "decision_report.json",
            "executed_trade_report.json",
            "invariant_audit.json",
            "count_reconciliation.json",
        ):
            st_module.markdown(f"**{name}**")
            st_module.json(pair.v2.reports.get(name, {}), expanded=False)
    with v3_tab:
        st_module.json(
            {
                "v2_reference": pair.v3.manifest.get("accepted_v2_reference"),
                "identity": pair.v3.manifest.get("identity"),
                "arrow_registry": pair.v3.manifest.get("context_arrow_registry"),
            },
            expanded=False,
        )
        st_module.dataframe(
            pd.DataFrame(
                {
                    "table": table.value,
                    "rows": len(pair.v3.tables[table]),
                    "columns": len(pair.v3.tables[table].columns),
                }
                for table in ContextRecordTable
            ),
            hide_index=True,
            use_container_width=True,
        )
    with access_tab:
        for name in (
            "data_access_audit.json",
            "capacity_report.json",
            "performance_report.json",
            "validity_report.json",
            "reconciliation_report.json",
            "identity_report.json",
        ):
            st_module.markdown(f"**{name}**")
            st_module.json(pair.v3.reports.get(name, {}), expanded=False)


# Compatibility seams retained for the existing chart-focused test module.
def _cached_replay_days(*_args, **_kwargs) -> list[str]:
    return []


def _cached_entry_dataset(path: str) -> pd.DataFrame | None:
    candidate = Path(path)
    return pd.read_parquet(candidate) if candidate.is_file() else None


def render_ifvg_lab_tab() -> None:
    tab_experiments, tab_replay, tab_audit = st.tabs(
        ["Experiments", "Replay / Verifier", "Data & Audit"]
    )
    with tab_experiments:
        render_ifvg_experiments_tab(st)
    with tab_replay:
        render_ifvg_replay_tab(st)
    with tab_audit:
        render_ifvg_data_audit_tab(st)
