"""
Dashboard compatibility/export tab for the retained 3-class workflow.

Visualizes walk-forward CatBoost 3-class model performance, feature
importance, confusion matrices, session accuracy, MAE distribution, and
event timelines.

All data loads from saved CSV/JSON artifacts. This is the secondary
compatibility path kept for ML-Trading-Dashboard, not the primary extrema
training workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ═══════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════

_RESULTS_DIR = Path(__file__).resolve().parents[1] / "data" / "experiment" / "training_results"
_DIAG_DIR = _RESULTS_DIR / "diagnostics"
_FEATURE_MATRIX = Path(__file__).resolve().parents[1] / "data" / "experiment" / "feature_matrix.parquet"
_BAR_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "databento" / "NQ"

CLASS_NAMES = {0: "tradeable_reversal", 1: "trap_reversal", 2: "aggressive_blowthrough"}
CLASS_COLORS = {
    "tradeable_reversal": "#26a69a",
    "trap_reversal": "#ef5350",
    "aggressive_blowthrough": "#FFA726",
}


# ═══════════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════════


@st.cache_data
def _load_summary() -> dict | None:
    path = _RESULTS_DIR / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def _load_fold_results() -> pd.DataFrame | None:
    path = _RESULTS_DIR / "fold_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_confusion_matrix() -> pd.DataFrame | None:
    path = _RESULTS_DIR / "confusion_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def _load_feature_stability() -> pd.DataFrame | None:
    path = _RESULTS_DIR / "feature_stability.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_mae_distribution() -> pd.DataFrame | None:
    path = _RESULTS_DIR / "mae_distribution.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_top3_fold_results() -> pd.DataFrame | None:
    path = _DIAG_DIR / "top3_fold_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_top3_confusion_matrix() -> pd.DataFrame | None:
    path = _DIAG_DIR / "top3_confusion_matrix.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


@st.cache_data
def _load_session_accuracy() -> pd.DataFrame | None:
    path = _DIAG_DIR / "session_accuracy.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_feature_distributions() -> pd.DataFrame | None:
    path = _DIAG_DIR / "feature_distributions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def _load_feature_matrix() -> pd.DataFrame | None:
    if not _FEATURE_MATRIX.exists():
        return None
    df = pd.read_parquet(_FEATURE_MATRIX)
    return df[["event_ts", "date", "label", "label_encoded"]].copy()


@st.cache_data
def _load_top3_predictions() -> pd.DataFrame | None:
    path = _RESULTS_DIR / "top3_predictions.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


@st.cache_data
def _load_fold_1m_bars(dates_key: str) -> pd.DataFrame | None:
    """Load and concatenate 1m session bars for given dates.

    dates_key is a comma-separated string of YYYY-MM-DD dates
    (string key for Streamlit cache hashability).
    """
    from datetime import datetime, timedelta

    dates = [d.strip() for d in dates_key.split(",") if d.strip()]
    if not dates:
        return None

    # Expand to include 1 day of padding on each side for context
    all_dates = set(dates)
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")
        all_dates.add((dt - timedelta(days=1)).strftime("%Y-%m-%d"))
        all_dates.add((dt + timedelta(days=1)).strftime("%Y-%m-%d"))

    frames: list[pd.DataFrame] = []
    for d in sorted(all_dates):
        # Prefer session-tagged bars, fall back to plain 1m
        for fname in ("ohlcv_1m_session.parquet", "ohlcv_1m.parquet"):
            p = _BAR_DATA_DIR / d / fname
            if p.exists():
                day_bars = pd.read_parquet(p)
                # Normalize timezone to US/Eastern before concat
                if isinstance(day_bars.index, pd.DatetimeIndex):
                    if day_bars.index.tz is None:
                        day_bars.index = day_bars.index.tz_localize("US/Eastern")
                    elif str(day_bars.index.tz) != "US/Eastern":
                        day_bars.index = day_bars.index.tz_convert("US/Eastern")
                frames.append(day_bars)
                break

    if not frames:
        return None

    bars = pd.concat(frames).sort_index()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.DatetimeIndex(bars.index)
    return bars


# ═══════════════════════════════════════════════════════════════
#  CHART 1: Walk-Forward Performance Overview
# ═══════════════════════════════════════════════════════════════


def _chart_walkforward_performance(
    fold_58: pd.DataFrame,
    fold_t3: pd.DataFrame,
    summary: dict,
) -> go.Figure:
    """Per-fold accuracy bars: 58-feature vs top-3, with threshold lines."""
    folds = fold_58["fold"].values
    x_labels = [f"Fold {f}" for f in folds]

    fig = go.Figure()

    # 58-feature model bars
    fig.add_trace(go.Bar(
        x=x_labels,
        y=fold_58["accuracy"].values,
        name="58-Feature Model",
        marker_color="#7E57C2",
        opacity=0.85,
        hovertemplate="Fold %{x}<br>Accuracy: %{y:.1%}<extra>58-feat</extra>",
    ))

    # Top-3 model bars
    fig.add_trace(go.Bar(
        x=x_labels,
        y=fold_t3["accuracy"].values,
        name="Top-3 Feature Model",
        marker_color="#26a69a",
        opacity=0.85,
        hovertemplate="Fold %{x}<br>Accuracy: %{y:.1%}<extra>Top-3</extra>",
    ))

    # Threshold lines
    fig.add_hline(y=0.40, line_dash="dash", line_color="#FFD700",
                  annotation_text="Accuracy Threshold (40%)",
                  annotation_position="top left")

    # Pooled accuracy annotations
    acc_58 = summary["overall_accuracy"]
    fig.add_hline(y=acc_58, line_dash="dot", line_color="#7E57C2",
                  annotation_text=f"58-feat pooled: {acc_58:.1%}",
                  annotation_position="bottom right")

    # Compute top-3 pooled from folds
    total_correct_t3 = (fold_t3["accuracy"] * fold_t3["n_test"]).sum()
    total_n_t3 = fold_t3["n_test"].sum()
    acc_t3 = total_correct_t3 / total_n_t3 if total_n_t3 > 0 else 0
    fig.add_hline(y=acc_t3, line_dash="dot", line_color="#26a69a",
                  annotation_text=f"Top-3 pooled: {acc_t3:.1%}",
                  annotation_position="top right")

    fig.update_layout(
        title="Walk-Forward Per-Fold Accuracy",
        barmode="group",
        height=400,
        template="plotly_dark",
        yaxis=dict(title="Accuracy", tickformat=".0%", range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 2: Confusion Matrix Heatmaps
# ═══════════════════════════════════════════════════════════════


def _chart_confusion_matrix(cm_df: pd.DataFrame, title: str) -> go.Figure:
    """Heatmap of a 3x3 confusion matrix."""
    labels = ["Reversal", "Trap", "Blowthrough"]
    z = cm_df.values

    # Compute row-normalized percentages for text
    row_sums = z.sum(axis=1, keepdims=True)
    pct = np.where(row_sums > 0, z / row_sums * 100, 0)

    text = [[f"{z[i][j]}<br>({pct[i][j]:.0f}%)" for j in range(3)] for i in range(3)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f"Pred {l}" for l in labels],
        y=[f"Actual {l}" for l in labels],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14),
        colorscale="Tealgrn",
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        height=350,
        template="plotly_dark",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=20, t=50, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 3: Feature Importance
# ═══════════════════════════════════════════════════════════════


def _chart_feature_importance(stability: pd.DataFrame) -> go.Figure:
    """Top 10 features bar chart with fold stability indicators."""
    top10 = stability.head(10).copy()
    top10 = top10.iloc[::-1]  # Horizontal bar: reverse for top at top

    colors = [
        "#26a69a" if row["in_all_folds"] else "#7E57C2"
        for _, row in top10.iterrows()
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top10["feature"],
        x=top10["mean_importance"],
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            array=top10["std_importance"].values,
            color="rgba(255,255,255,0.4)",
        ),
        hovertemplate=(
            "%{y}<br>"
            "Mean importance: %{x:.1f}<br>"
            "In top 10 of %{customdata[0]}/5 folds<br>"
            "Stable across all folds: %{customdata[1]}"
            "<extra></extra>"
        ),
        customdata=list(zip(
            top10["folds_in_top_10"].values,
            ["Yes" if v else "No" for v in top10["in_all_folds"].values],
            strict=True,
        )),
    ))

    fig.update_layout(
        title="Top 10 Features by Mean Importance",
        height=400,
        template="plotly_dark",
        xaxis=dict(title="Mean Importance (CatBoost)"),
        margin=dict(l=200, r=20, t=50, b=40),
        annotations=[dict(
            text="Green = stable across all 5 folds | Purple = appears in fewer folds",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=11, color="gray"),
        )],
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 4: Feature Class Distributions (Box/Violin)
# ═══════════════════════════════════════════════════════════════


def _chart_feature_distributions(feat_dist: pd.DataFrame) -> go.Figure:
    """Box plots for top 3 numeric features split by class.

    Uses the mean/std/median summary from the diagnostic CSV to
    build violin-like shapes.
    """
    top3 = ["int_time_beyond_level", "int_time_within_2pts", "int_absorption_ratio"]
    classes = ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]
    class_short = {"tradeable_reversal": "Reversal", "trap_reversal": "Trap",
                   "aggressive_blowthrough": "Blowthrough"}

    fig = make_subplots(rows=1, cols=3, subplot_titles=[
        f.replace("int_", "") for f in top3
    ])

    for col_idx, feat in enumerate(top3, 1):
        feat_data = feat_dist[feat_dist["feature"] == feat]

        for cls in classes:
            cls_data = feat_data[feat_data["class"] == cls]
            mean_row = cls_data[cls_data["metric"] == "mean"]
            std_row = cls_data[cls_data["metric"] == "std"]
            median_row = cls_data[cls_data["metric"] == "median"]

            if mean_row.empty:
                continue

            mean_val = float(mean_row["value"].iloc[0])
            std_val = float(std_row["value"].iloc[0]) if not std_row.empty else 0
            median_val = float(median_row["value"].iloc[0]) if not median_row.empty else mean_val

            color = CLASS_COLORS[cls]

            # Show as error bars around mean
            fig.add_trace(go.Bar(
                x=[class_short[cls]],
                y=[mean_val],
                name=class_short[cls] if col_idx == 1 else None,
                marker_color=color,
                opacity=0.8,
                error_y=dict(type="data", array=[std_val], color="rgba(255,255,255,0.5)"),
                showlegend=(col_idx == 1),
                legendgroup=cls,
                hovertemplate=(
                    f"{class_short[cls]}<br>"
                    f"Mean: {mean_val:.2f}<br>"
                    f"Std: {std_val:.2f}<br>"
                    f"Median: {median_val:.2f}"
                    "<extra></extra>"
                ),
            ), row=1, col=col_idx)

    fig.update_layout(
        title="Top-3 Feature Distributions by Class (Mean +/- Std)",
        height=400,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.05),
        margin=dict(l=50, r=20, t=80, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 5: Session Accuracy Breakdown
# ═══════════════════════════════════════════════════════════════


def _chart_session_accuracy(session_df: pd.DataFrame) -> go.Figure:
    """Bar chart: accuracy and reversal precision per session."""
    sessions = session_df["session"].values
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=sessions,
        y=session_df["accuracy"].values,
        name="Accuracy",
        marker_color="#26a69a",
        opacity=0.85,
        hovertemplate="%{x}<br>Accuracy: %{y:.1%}<br>N=%{customdata}<extra></extra>",
        customdata=session_df["n"].values,
    ))

    fig.add_trace(go.Bar(
        x=sessions,
        y=session_df["reversal_precision"].values,
        name="Reversal Precision",
        marker_color="#7E57C2",
        opacity=0.85,
        hovertemplate="%{x}<br>Rev Precision: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=sessions,
        y=session_df["reversal_recall"].values,
        name="Reversal Recall",
        marker_color="#FFA726",
        opacity=0.85,
        hovertemplate="%{x}<br>Rev Recall: %{y:.1%}<extra></extra>",
    ))

    # Event counts as text above
    for i, row in session_df.iterrows():
        fig.add_annotation(
            x=row["session"], y=max(row["accuracy"], row["reversal_precision"]) + 0.05,
            text=f"n={int(row['n'])}",
            showarrow=False, font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title="Session Accuracy Breakdown (Top-3 Model)",
        barmode="group",
        height=400,
        template="plotly_dark",
        yaxis=dict(title="Score", tickformat=".0%", range=[0, 1.15]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 6: MAE Distribution
# ═══════════════════════════════════════════════════════════════


def _chart_mae_distribution(mae_df: pd.DataFrame) -> go.Figure:
    """Histogram of MAE for true positive reversals."""
    values = mae_df["mae_pts"].values

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        name="MAE (pts)",
        marker_color="#26a69a",
        opacity=0.8,
        hovertemplate="MAE: %{x:.1f} pts<br>Count: %{y}<extra></extra>",
    ))

    # 37.5pt stop reference
    fig.add_vline(
        x=37.5, line_dash="dash", line_color="#ef5350",
        annotation_text="37.5pt Stop",
        annotation_position="top right",
        annotation_font=dict(color="#ef5350"),
    )

    # Median line
    median_mae = float(np.median(values))
    fig.add_vline(
        x=median_mae, line_dash="dot", line_color="#FFD700",
        annotation_text=f"Median: {median_mae:.1f}pt",
        annotation_position="top left",
        annotation_font=dict(color="#FFD700"),
    )

    below_stop = (values <= 37.5).sum()
    pct_below = below_stop / len(values) * 100 if len(values) > 0 else 0

    fig.update_layout(
        title=f"MAE Distribution — True Positive Reversals ({pct_below:.0f}% within 37.5pt stop)",
        height=350,
        template="plotly_dark",
        xaxis=dict(title="Maximum Adverse Excursion (points)"),
        yaxis=dict(title="Count"),
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  CHART 7: Event Timeline
# ═══════════════════════════════════════════════════════════════


def _chart_event_timeline(
    events: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> go.Figure:
    """Scatter of events over time, colored by class, with fold boundaries."""
    fig = go.Figure()

    # Plot each class
    for cls_id, cls_name in CLASS_NAMES.items():
        mask = events["label_encoded"] == cls_id
        cls_events = events[mask]
        fig.add_trace(go.Scatter(
            x=cls_events["event_ts"],
            y=[cls_name.replace("_", " ").title()] * len(cls_events),
            mode="markers",
            marker=dict(
                size=8,
                color=CLASS_COLORS[cls_name],
                opacity=0.7,
            ),
            name=cls_name.replace("_", " ").title(),
            hovertemplate=(
                "%{x|%Y-%m-%d %H:%M}<br>"
                f"{cls_name.replace('_', ' ').title()}"
                "<extra></extra>"
            ),
        ))

    # Fold boundary lines from test_dates column
    for _, row in fold_results.iterrows():
        test_start = row["test_dates"].split(" -- ")[0].strip()
        test_end = row["test_dates"].split(" -- ")[1].strip()
        fold_num = int(row["fold"])

        # Shaded fold test window
        fig.add_shape(
            type="rect",
            x0=test_start, x1=test_end,
            y0=0, y1=1, yref="paper",
            fillcolor="rgba(126, 87, 194, 0.1)",
            line_width=0,
        )
        # Fold label at midpoint
        mid_date = pd.Timestamp(test_start) + (
            pd.Timestamp(test_end) - pd.Timestamp(test_start)
        ) / 2
        fig.add_annotation(
            x=mid_date, y=1.05, yref="paper",
            text=f"Fold {fold_num}",
            showarrow=False, font=dict(size=10, color="gray"),
        )

    fig.update_layout(
        title="Event Timeline with Walk-Forward Fold Boundaries",
        height=300,
        template="plotly_dark",
        xaxis=dict(title="Date"),
        yaxis=dict(title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=150, r=20, t=60, b=40),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═══════════════════════════════════════════════════════════════


def render_experiment_tab() -> None:
    """Render the dashboard compatibility tab."""
    st.subheader("Dashboard Compatibility Export — 3-Class Results")
    st.caption(
        "Secondary workflow kept to support ML-Trading-Dashboard. "
        "Use the ML Training tab for the primary extrema pipeline."
    )

    # Load all data
    summary = _load_summary()
    fold_58 = _load_fold_results()
    cm_58 = _load_confusion_matrix()
    stability = _load_feature_stability()
    mae_df = _load_mae_distribution()
    fold_t3 = _load_top3_fold_results()
    cm_t3 = _load_top3_confusion_matrix()
    session_df = _load_session_accuracy()
    feat_dist = _load_feature_distributions()
    events = _load_feature_matrix()

    if summary is None or fold_58 is None:
        st.info(
            "No dashboard-compatibility results found. Refresh the retained "
            "3-class pipeline first:\n\n"
            "```bash\npython -m alpha_lab.experiment.training\n"
            "python -m alpha_lab.experiment.diagnostics\n"
            "python scripts/train_dashboard_model.py\n```"
        )
        return

    # ── Verdict Summary Metrics ─────────────────────────────────
    verdict = summary.get("verdict", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        v = verdict.get("Overall Accuracy > 40%", {})
        val = v.get("value", 0)
        passed = v.get("passed", False)
        st.metric("Overall Accuracy", f"{val:.1%}",
                  delta="PASS" if passed else "FAIL",
                  delta_color="normal" if passed else "inverse")
    with c2:
        v = verdict.get("Tradeable Reversal Precision > 50%", {})
        val = v.get("value", 0)
        passed = v.get("passed", False)
        st.metric("Rev. Precision", f"{val:.1%}",
                  delta="PASS" if passed else "FAIL",
                  delta_color="normal" if passed else "inverse")
    with c3:
        v = verdict.get("Blow-through Recall > 60%", {})
        val = v.get("value", 0)
        passed = v.get("passed", False)
        st.metric("BT Recall", f"{val:.1%}",
                  delta="PASS" if passed else "FAIL",
                  delta_color="normal" if passed else "inverse")
    with c4:
        v = verdict.get("Cross-fold Accuracy StdDev < 10%", {})
        val = v.get("value", 0)
        passed = v.get("passed", False)
        st.metric("Cross-fold Std", f"{val:.1%}",
                  delta="PASS" if passed else "FAIL",
                  delta_color="normal" if passed else "inverse")

    passed_count = sum(1 for v in verdict.values() if v.get("passed", False))
    total_checks = len(verdict)
    if passed_count == total_checks:
        st.success(f"Hypothesis SUPPORTED — {passed_count}/{total_checks} thresholds met (58-feature model)")
    else:
        st.warning(f"Hypothesis PARTIALLY SUPPORTED — {passed_count}/{total_checks} thresholds met (58-feature model)")

    st.divider()

    # ── Chart 1: Walk-Forward Performance ───────────────────────
    if fold_t3 is not None:
        st.plotly_chart(
            _chart_walkforward_performance(fold_58, fold_t3, summary),
            use_container_width=True,
        )

        # Side-by-side pooled comparison
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**58-Feature Model (Pooled)**")
            st.markdown(f"- Accuracy: `{summary['overall_accuracy']:.1%}`")
            st.markdown(f"- Rev Precision: `{summary['reversal_precision']:.1%}`")
            st.markdown(f"- BT Recall: `{summary['blowthrough_recall']:.1%}`")
            st.markdown(f"- Cross-fold Std: `{summary['accuracy_variance']:.1%}`")
        with col2:
            # Compute pooled metrics from aggregated confusion matrix
            if cm_t3 is not None:
                cm_vals = cm_t3.values
                t3_acc = float(np.trace(cm_vals)) / cm_vals.sum()
                rev_pred_total = cm_vals[:, 0].sum()
                t3_rev_prec = float(cm_vals[0, 0]) / rev_pred_total if rev_pred_total > 0 else 0
                bt_actual_total = cm_vals[2, :].sum()
                t3_bt_recall = float(cm_vals[2, 2]) / bt_actual_total if bt_actual_total > 0 else 0
            else:
                t3_acc = 0
                t3_rev_prec = 0
                t3_bt_recall = 0
            t3_std = float(fold_t3["accuracy"].std())
            st.markdown("**Top-3 Feature Model (Pooled)**")
            st.markdown(f"- Accuracy: `{t3_acc:.1%}`")
            st.markdown(f"- Rev Precision: `{t3_rev_prec:.1%}`")
            st.markdown(f"- BT Recall: `{t3_bt_recall:.1%}`")
            st.markdown(f"- Cross-fold Std: `{t3_std:.1%}`")
    else:
        st.info("Top-3 model results not available. Run diagnostics to generate.")

    st.divider()

    # ── Chart 2: Confusion Matrices ─────────────────────────────
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        if cm_58 is not None:
            st.plotly_chart(
                _chart_confusion_matrix(cm_58, "58-Feature Model"),
                use_container_width=True,
            )
    with col2:
        if cm_t3 is not None:
            st.plotly_chart(
                _chart_confusion_matrix(cm_t3, "Top-3 Feature Model"),
                use_container_width=True,
            )

    st.divider()

    # ── Chart 3: Feature Importance ─────────────────────────────
    if stability is not None:
        st.plotly_chart(
            _chart_feature_importance(stability),
            use_container_width=True,
        )

    st.divider()

    # ── Chart 4: Feature Class Distributions ────────────────────
    if feat_dist is not None:
        st.plotly_chart(
            _chart_feature_distributions(feat_dist),
            use_container_width=True,
        )

    st.divider()

    # ── Chart 5: Session Accuracy ───────────────────────────────
    if session_df is not None:
        st.plotly_chart(
            _chart_session_accuracy(session_df),
            use_container_width=True,
        )

    st.divider()

    # ── Chart 6: MAE Distribution ───────────────────────────────
    if mae_df is not None:
        st.plotly_chart(
            _chart_mae_distribution(mae_df),
            use_container_width=True,
        )

    st.divider()

    # ── Chart 7: Event Timeline ─────────────────────────────────
    if events is not None and fold_58 is not None:
        st.plotly_chart(
            _chart_event_timeline(events, fold_58),
            use_container_width=True,
        )

    st.divider()

    # ── Chart 8: Trade Visualization ────────────────────────────
    predictions = _load_top3_predictions()
    if predictions is not None and fold_58 is not None:
        _render_trade_visualization(predictions, fold_58)
    else:
        st.info(
            "Trade visualization requires predictions. Generate with:\n\n"
            "```bash\npython -m alpha_lab.experiment.diagnostics --predictions\n```"
        )


# ═══════════════════════════════════════════════════════════════
#  CHART 8: Trade Visualization
# ═══════════════════════════════════════════════════════════════

PRED_COLORS = {
    "tradeable_reversal": "#26a69a",
    "trap_reversal": "#FFA726",
    "aggressive_blowthrough": "#ef5350",
}
PRED_SHORT = {
    "tradeable_reversal": "Reversal",
    "trap_reversal": "Trap",
    "aggressive_blowthrough": "Blowthrough",
}


def _chart_fold_trades(
    bars_1m: pd.DataFrame,
    preds: pd.DataFrame,
    fold_num: int,
    focus_event_ts: pd.Timestamp | None = None,
) -> go.Figure:
    """1-minute candlestick chart with trade prediction markers.

    If focus_event_ts is given, zoom to +/-60 minutes around that event.
    Otherwise show the full fold period.
    """
    # Work with tz-naive timestamps for consistent matching
    bars = bars_1m.copy()
    if isinstance(bars.index, pd.DatetimeIndex) and bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    elif not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.DatetimeIndex(bars.index)

    if focus_event_ts is not None:
        # Focus mode: +/-60 minutes
        center = focus_event_ts.tz_localize(None) if focus_event_ts.tz else focus_event_ts
        window_start = center - pd.Timedelta(minutes=60)
        window_end = center + pd.Timedelta(minutes=60)
        bars = bars[(bars.index >= window_start) & (bars.index <= window_end)]

    if bars.empty:
        fig = go.Figure()
        fig.add_annotation(text="No bar data available for this period", showarrow=False)
        fig.update_layout(height=400, template="plotly_dark")
        return fig

    n = len(bars)
    x_int = list(range(n))

    # Tick labels
    raw_idx = bars.index
    if isinstance(raw_idx, pd.DatetimeIndex):
        tick_labels = raw_idx.strftime("%m/%d %H:%M").tolist()
    else:
        tick_labels = [str(x) for x in raw_idx]

    # Timestamp-to-position map
    ts_to_pos: dict[pd.Timestamp, int] = {}
    for i, ts in enumerate(bars.index):
        ts_to_pos[ts] = i

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
        subplot_titles=(
            f"Fold {fold_num} — Top-3 Model Predictions on 1m NQ Bars",
            "Volume",
        ),
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_int,
        open=bars["open"].values,
        high=bars["high"].values,
        low=bars["low"].values,
        close=bars["close"].values,
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Volume
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(bars["close"], bars["open"], strict=False)
    ]
    fig.add_trace(go.Bar(
        x=x_int, y=bars["volume"].values, name="Volume",
        marker_color=vol_colors, opacity=0.5, showlegend=False,
    ), row=2, col=1)

    # Event markers
    for pred_label in ["tradeable_reversal", "trap_reversal", "aggressive_blowthrough"]:
        mask = preds["predicted_label"] == pred_label
        if not mask.any():
            continue

        subset = preds[mask]
        color = PRED_COLORS[pred_label]
        short_name = PRED_SHORT[pred_label]

        # Separate correct vs wrong
        for is_correct, symbol, suffix in [
            (True, "circle", "Correct"),
            (False, "x", "Wrong"),
        ]:
            sub = subset[subset["correct"] == is_correct]
            if sub.empty:
                continue

            x_positions = []
            y_positions = []
            hover_texts = []

            for _, row in sub.iterrows():
                evt_ts = row["event_ts"]
                if evt_ts.tz is not None:
                    evt_ts = evt_ts.tz_localize(None)

                # Find nearest bar position
                pos = ts_to_pos.get(evt_ts)
                if pos is None:
                    # Search for closest bar within 1 minute
                    diffs = abs(bars.index - evt_ts)
                    min_idx = diffs.argmin()
                    if diffs[min_idx] <= pd.Timedelta(minutes=1):
                        pos = min_idx
                if pos is None:
                    continue

                x_positions.append(pos)
                y_positions.append(float(row["level_price"]))
                hover_texts.append(
                    f"<b>{row['date']} {evt_ts.strftime('%H:%M')}</b><br>"
                    f"Level: {row['level_name']} @ {row['level_price']:.2f}<br>"
                    f"Direction: {row['direction']}<br>"
                    f"Predicted: <b>{PRED_SHORT[row['predicted_label']]}</b><br>"
                    f"Actual: <b>{PRED_SHORT[row['actual_label']]}</b><br>"
                    f"MFE: {row['mfe']:.1f} pts | MAE: {row['mae']:.1f} pts<br>"
                    f"P(Rev): {row['prob_reversal']:.0%} | "
                    f"P(Trap): {row['prob_trap']:.0%} | "
                    f"P(BT): {row['prob_blowthrough']:.0%}"
                )

            if x_positions:
                fig.add_trace(go.Scatter(
                    x=x_positions,
                    y=y_positions,
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=14 if symbol == "circle" else 12,
                        color=color,
                        line=dict(width=2, color="white"),
                        opacity=0.9,
                    ),
                    name=f"{short_name} ({suffix})",
                    hovertext=hover_texts,
                    hoverinfo="text",
                ), row=1, col=1)

    # Level price horizontal lines
    for _, row in preds.iterrows():
        evt_ts = row["event_ts"]
        if evt_ts.tz is not None:
            evt_ts = evt_ts.tz_localize(None)
        pos = ts_to_pos.get(evt_ts)
        if pos is None:
            diffs = abs(bars.index - evt_ts)
            min_idx = diffs.argmin()
            if diffs[min_idx] <= pd.Timedelta(minutes=1):
                pos = min_idx
        if pos is None:
            continue

        x0 = max(0, pos - 30)
        x1 = min(n - 1, pos + 30)
        color = PRED_COLORS[row["predicted_label"]]
        fig.add_shape(
            type="line",
            x0=x0, x1=x1,
            y0=row["level_price"], y1=row["level_price"],
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.5,
            row=1, col=1,
        )

    # Layout
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=20, t=60, b=20),
    )

    # Tick labels
    step = max(1, n // 25)
    t_pos = list(range(0, n, step))
    t_txt = [tick_labels[i] for i in t_pos]
    fig.update_xaxes(tickvals=t_pos, ticktext=t_txt, row=1, col=1)
    fig.update_xaxes(tickvals=t_pos, ticktext=t_txt, row=2, col=1)

    return fig


def _render_trade_visualization(
    predictions: pd.DataFrame,
    fold_results: pd.DataFrame,
) -> None:
    """Render the trade visualization section with fold selector and focus mode."""
    st.subheader("Trade Visualization")

    # Controls
    col_fold, col_filter = st.columns([1, 2])

    with col_fold:
        fold_options = sorted(predictions["fold"].unique())
        fold_labels = [f"Fold {f}" for f in fold_options]
        selected_fold_idx = st.selectbox(
            "Select Fold", range(len(fold_options)),
            format_func=lambda i: fold_labels[i],
            key="trade_viz_fold",
        )
        selected_fold = fold_options[selected_fold_idx]

    with col_filter:
        class_options = ["Reversal", "Trap", "Blowthrough"]
        class_map = {
            "Reversal": "tradeable_reversal",
            "Trap": "trap_reversal",
            "Blowthrough": "aggressive_blowthrough",
        }
        selected_classes = st.multiselect(
            "Filter by Prediction",
            class_options,
            default=class_options,
            key="trade_viz_filter",
        )

    # Filter predictions for this fold
    fold_preds = predictions[predictions["fold"] == selected_fold].copy()
    if selected_classes:
        allowed = [class_map[c] for c in selected_classes]
        fold_preds = fold_preds[fold_preds["predicted_label"].isin(allowed)]

    if fold_preds.empty:
        st.warning("No predictions match the current filter.")
        return

    # Get test dates for this fold from fold_results
    fold_row = fold_results[fold_results["fold"] == selected_fold]
    if not fold_row.empty:
        test_dates_str = fold_row.iloc[0]["test_dates"]
        st.caption(f"Test period: {test_dates_str}")

    # Load 1m bars for the fold's dates
    fold_dates = sorted(fold_preds["date"].unique())
    dates_key = ",".join(fold_dates)
    bars_1m = _load_fold_1m_bars(dates_key)

    if bars_1m is None or bars_1m.empty:
        st.warning(f"No 1-minute bar data found for dates: {', '.join(fold_dates)}")
        return

    # Focus mode: event selector
    focus_ts = None
    focus_mode = st.checkbox("Focus mode (+/-60 min around selected event)", key="focus_mode")

    if focus_mode:
        event_labels = []
        for _, row in fold_preds.iterrows():
            ts = row["event_ts"]
            ts_str = ts.strftime("%m/%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            correct_str = "OK" if row["correct"] else "MISS"
            event_labels.append(
                f"{ts_str} | {row['level_name']} | "
                f"Pred: {PRED_SHORT[row['predicted_label']]} | "
                f"Actual: {PRED_SHORT[row['actual_label']]} | {correct_str}"
            )
        selected_event_idx = st.selectbox(
            "Select Event to Focus",
            range(len(event_labels)),
            format_func=lambda i: event_labels[i],
            key="focus_event",
        )
        focus_ts = fold_preds.iloc[selected_event_idx]["event_ts"]

    # Summary metrics for this fold
    n_total = len(fold_preds)
    n_correct = fold_preds["correct"].sum()
    n_rev = (fold_preds["predicted_label"] == "tradeable_reversal").sum()
    n_trap = (fold_preds["predicted_label"] == "trap_reversal").sum()
    n_bt = (fold_preds["predicted_label"] == "aggressive_blowthrough").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Events", n_total)
    with c2:
        st.metric("Correct", f"{n_correct}/{n_total} ({n_correct/n_total:.0%})" if n_total > 0 else "0")
    with c3:
        st.metric("Pred Reversal", n_rev)
    with c4:
        st.metric("Pred Trap", n_trap)
    with c5:
        st.metric("Pred BT", n_bt)

    # Chart
    fig = _chart_fold_trades(bars_1m, fold_preds, selected_fold, focus_ts)
    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    with st.expander("Event Detail Table", expanded=False):
        display_df = fold_preds[[
            "event_ts", "level_name", "level_price", "direction",
            "predicted_label", "actual_label", "correct",
            "prob_reversal", "prob_trap", "prob_blowthrough",
            "mfe", "mae",
        ]].copy()
        display_df["event_ts"] = display_df["event_ts"].dt.strftime("%Y-%m-%d %H:%M")
        display_df.columns = [
            "Time", "Level", "Price", "Dir",
            "Predicted", "Actual", "Correct",
            "P(Rev)", "P(Trap)", "P(BT)",
            "MFE", "MAE",
        ]
        # Format probabilities
        for col in ["P(Rev)", "P(Trap)", "P(BT)"]:
            display_df[col] = display_df[col].map("{:.0%}".format)
        display_df["MFE"] = display_df["MFE"].map("{:.1f}".format)
        display_df["MAE"] = display_df["MAE"].map("{:.1f}".format)
        for col in ["Predicted", "Actual"]:
            display_df[col] = display_df[col].map(PRED_SHORT)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
