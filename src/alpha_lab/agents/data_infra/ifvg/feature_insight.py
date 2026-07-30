"""Feature-insight section of an IFVG experiment result (plan Part A item 2c).

Model-free univariate scan (quartile buckets over numeric features, per-value
tables over categoricals) + the model importances the engine measured per split
(CatBoost gain and shuffle-one-feature permutation delta-Brier on TEST folds).
Ranks hypotheses only — see :data:`MULTIPLE_COMPARISONS_CAVEAT`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .trade_stats import mean_ci, wilson_ci

__all__ = ["compute_feature_insight", "MULTIPLE_COMPARISONS_CAVEAT"]

MULTIPLE_COMPARISONS_CAVEAT = (
    "~43 features x 4 buckets is a multiple-comparisons field; this panel ranks "
    "hypotheses for targeted experiments, it does not confirm edges; model "
    "importances are noise whenever calibration is flat."
)

#: Boolean causality flags reported as categorical per-value tables.
_BOOL_FLAGS = ("tap_ce_reached", "lock_ce_reached", "inv_sweep_sweep_confirmed")

_MAX_CATEGORY_VALUES = 20


def _bucket_row(sub: pd.DataFrame, label: str) -> dict:
    n = len(sub)
    wins = int((sub["_label"] == "win").sum())
    net = pd.to_numeric(sub["_net_r"], errors="coerce").dropna()
    return {
        "bucket": label,
        "n": n,
        "win_rate": wins / n if n else None,
        "win_rate_ci95": wilson_ci(wins, n),
        "mean_net_r": float(net.mean()) if len(net) else None,
        "net_r_ci95": mean_ci(net),
    }


def _numeric_feature(work: pd.DataFrame, col: str) -> dict | None:
    values = pd.to_numeric(work[col], errors="coerce")
    valid = values.notna()
    if int(valid.sum()) < 8 or values.nunique() < 2:
        return None
    try:
        buckets = pd.qcut(values[valid], q=4, duplicates="drop")
    except ValueError:
        return None
    if buckets.cat.categories.empty:
        return None
    rows = []
    for interval in buckets.cat.categories:
        sub = work.loc[valid].loc[buckets == interval]
        rows.append(_bucket_row(sub, str(interval)))
    nets = [r["mean_net_r"] for r in rows if r["mean_net_r"] is not None]
    spread = (max(nets) - min(nets)) if len(nets) >= 2 else 0.0
    win = (work.loc[valid, "_label"] == "win").astype(int).to_numpy()
    codes = buckets.cat.codes.to_numpy()
    rho = 0.0
    if len(np.unique(codes)) > 1 and len(np.unique(win)) > 1:
        rho_val = spearmanr(codes, win).statistic
        rho = float(rho_val) if np.isfinite(rho_val) else 0.0
    return {
        "feature": col,
        "n_valid": int(valid.sum()),
        "buckets": rows,
        "net_r_spread": spread,
        "spearman_bucket_vs_win": rho,
    }


def _categorical_feature(work: pd.DataFrame, col: str, *, as_bool: bool = False) -> dict:
    if as_bool:
        # Bool flags reach the engine as coerced 0.0/1.0 floats — label them
        # True/False, not "0.0"/"1.0".
        values = work[col].map(lambda v: "none" if pd.isna(v) else str(bool(v)))
    else:
        values = work[col].fillna("none").astype(str)
    rows = []
    for value, _ in values.value_counts().head(_MAX_CATEGORY_VALUES).items():
        sub = work[values == value]
        rows.append(_bucket_row(sub, value))
    return {"feature": col, "values": rows}


def compute_feature_insight(
    work: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    *,
    model_importance: list[dict] | None = None,
    permutation_importance: list[dict] | None = None,
) -> dict:
    """Univariate + model-importance insight over an engine-prepared frame."""
    numeric = []
    for col in numeric_features:
        # Bool flags are reported once, as categorical tables — a quartile scan
        # over a 0/1 column is the same information with worse labels.
        if col not in work.columns or col in _BOOL_FLAGS:
            continue
        entry = _numeric_feature(work, col)
        if entry is not None:
            numeric.append(entry)
    ranked_by_spread = sorted(numeric, key=lambda e: e["net_r_spread"], reverse=True)
    ranked_by_monotonicity = sorted(
        numeric, key=lambda e: abs(e["spearman_bucket_vs_win"]), reverse=True
    )
    cats = [
        _categorical_feature(work, col, as_bool=col in _BOOL_FLAGS)
        for col in [*categorical_features, *_BOOL_FLAGS]
        if col in work.columns
    ]
    return {
        "numeric": numeric,
        "rank_by_net_r_spread": [e["feature"] for e in ranked_by_spread],
        "rank_by_monotonicity": [e["feature"] for e in ranked_by_monotonicity],
        "categorical": cats,
        "model_importance_per_split": model_importance or [],
        "permutation_importance": permutation_importance or [],
        "caveat": MULTIPLE_COMPARISONS_CAVEAT,
    }
