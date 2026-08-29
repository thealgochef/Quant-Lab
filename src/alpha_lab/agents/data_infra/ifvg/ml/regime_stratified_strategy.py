"""``cohort_descriptive`` — strategy metrics by descriptive OOS regime (R6.1 §6.G).

Executed trades are joined EXACTLY (by ``candidate_id``) onto the
descriptive ``RegimeOosAssignmentArtifact``; per stratum (``pooled_all``,
``pooled_regime_covered``, one per canonical reporting id, ``unassigned``)
the SAME ``compute_strategy_metrics`` the search gates use is evaluated over
the stratum's validated executed-trade rows — so ``pooled_all`` IS the
child's own metrics (asserted by test), and a stratum below the stamped
``minimum_trades_per_regime_stratum`` (20) is typed
``insufficient_regime_partition`` rather than reported. Concentration
flags (``works_only_in_regime``, the top regime's share of absolute net R)
and the descriptive restriction of the POOLED model's OOS predictions to
each stratum ride the body. No model is specialized here; no counterfactual
is claimed; nothing is selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..contracts import RecordTable
from ..search.identities import canonical_contract_sha256
from ..search.strategy_metrics import StrategyMetrics, compute_strategy_metrics
from ..trade_stats import _validate_and_normalize_executed_trades
from .regime_assignment_sources import regime_for_trades, stratum_cohorts
from .regime_stratified_contracts import (
    MINIMUM_TRADES_PER_REGIME_STRATUM,
    CohortDescriptiveBody,
    RegimeStratumKey,
    StrategyStratumRow,
    StratumMetrics,
)

__all__ = [
    "CohortDescriptiveResult",
    "executed_trade_table_sha256",
    "stratum_metrics_from",
    "build_cohort_descriptive_body",
]


@dataclass(frozen=True)
class CohortDescriptiveResult:
    body: CohortDescriptiveBody
    detail: dict[str, Any]
    trade_regimes: pd.DataFrame


def executed_trade_table_sha256(trades: pd.DataFrame) -> str:
    """Order-free content hash of the executed-trade table (sorted by trade_id)."""

    if "trade_id" not in trades.columns:
        raise ValueError("executed trades lack the trade_id column")
    ordered = trades.sort_values("trade_id", kind="mergesort").reset_index(drop=True)
    ordered = ordered.loc[:, sorted(ordered.columns)]
    records = []
    for row in ordered.itertuples(index=False):
        record = []
        for value in row:
            if isinstance(value, float | np.floating):
                record.append(None if not np.isfinite(value) else float(value))
            elif isinstance(value, np.integer):
                record.append(int(value))
            elif isinstance(value, np.bool_):
                record.append(bool(value))
            elif value is None or isinstance(value, str | int | bool):
                record.append(value)
            else:
                record.append(str(value))
        records.append(record)
    return canonical_contract_sha256({"columns": list(ordered.columns), "rows": records})


def stratum_metrics_from(metrics: StrategyMetrics) -> StratumMetrics:
    return StratumMetrics(
        executed_trades=metrics.executed_trades,
        independent_days=metrics.independent_days,
        gross_expectancy_r=metrics.gross_expectancy_r,
        net_expectancy_r=metrics.net_expectancy_r,
        profit_factor=metrics.profit_factor,
        max_drawdown_r=metrics.max_drawdown_r,
        time_under_water_days=metrics.time_under_water_days,
        time_block_sign_consistency=metrics.time_block_sign_consistency,
        session_stability_score=metrics.session_stability_score,
        top_day_pnl_share=metrics.top_day_pnl_share,
        top_setup_pnl_share=metrics.top_setup_pnl_share,
        net_expectancy_bootstrap_ci95=metrics.net_expectancy_bootstrap_ci95,
    )


def _pooled_skill(
    predictions: pd.DataFrame | None, candidate_ids: pd.Series
) -> dict[str, float] | None:
    """Descriptive restriction of the POOLED rung's OOS predictions."""

    if predictions is None or predictions.empty:
        return None
    for column in ("candidate_id", "predicted_probability", "binary_target"):
        if column not in predictions.columns:
            raise ValueError(f"pooled predictions lack the {column!r} column")
    keyed = predictions.copy()
    keyed["candidate_id"] = keyed["candidate_id"].astype(str)
    if keyed["candidate_id"].duplicated().any():
        raise ValueError("pooled predictions repeat a candidate_id")
    subset = keyed[keyed["candidate_id"].isin(set(candidate_ids.astype(str)))]
    if subset.empty:
        return {"rows": 0.0}
    probability = pd.to_numeric(subset["predicted_probability"], errors="raise").astype(float)
    target = pd.to_numeric(subset["binary_target"], errors="raise").astype(float)
    clipped = probability.clip(1e-12, 1 - 1e-12)
    brier = float(((probability - target) ** 2).mean())
    log_loss = float(-(target * np.log(clipped) + (1 - target) * np.log(1 - clipped)).mean())
    return {"rows": float(len(subset)), "brier": brier, "log_loss": log_loss}


def build_cohort_descriptive_body(
    *,
    trades: pd.DataFrame,
    assignment: pd.DataFrame,
    core_replay_id: str,
    protocol_id: str,
    fit_ids: tuple[str, ...],
    cost_points: float,
    evaluation_config_hash: str,
    tick_size: float = 0.25,
    tp_r_multiple: float = 1.0,
    pooled_predictions: pd.DataFrame | None = None,
    minimum_trades: int = MINIMUM_TRADES_PER_REGIME_STRATUM,
) -> CohortDescriptiveResult:
    """Strata over the validated executed-trade table under the descriptive
    OOS assignment (module docstring)."""

    ordered = _validate_and_normalize_executed_trades(trades, tick_size=tick_size)
    if ordered.empty:
        raise ValueError("cohort_descriptive requires at least one executed trade")
    joined = regime_for_trades(trades, assignment)
    joined = joined.set_index("trade_id")
    trade_ids = trades["trade_id"].astype(str)
    clusters = joined.loc[trade_ids, "canonical_reporting_cluster_id"].to_numpy()
    valid = joined.loc[trade_ids, "valid"].to_numpy().astype(bool)
    total = int(len(trades))
    covered_mask = valid
    reporting_ids = tuple(sorted({int(value) for value in clusters[valid]}))
    cohorts = stratum_cohorts(protocol_id=protocol_id, fit_ids=fit_ids, cluster_ids=reporting_ids)

    def _metrics(subset: pd.DataFrame) -> StrategyMetrics:
        return compute_strategy_metrics(
            {RecordTable.EXECUTED_TRADE: subset.reset_index(drop=True)},
            cost_points=cost_points,
            evaluation_config_hash=evaluation_config_hash,
            tick_size=tick_size,
            tp_r_multiple=tp_r_multiple,
        )

    def _row(
        key: RegimeStratumKey, subset: pd.DataFrame, *, always_report: bool
    ) -> tuple[StrategyStratumRow, dict[str, Any]]:
        count = int(len(subset))
        cohort = cohorts.get(key.label)
        if count == 0 or (count < minimum_trades and not always_report):
            row = StrategyStratumRow(
                key=key,
                executed_trades=count,
                share_of_trades=count / total,
                typed_state="insufficient_regime_partition",
                metrics=None,
                pooled_model_skill=None,
                cohort_id=cohort.cohort_id if cohort is not None else None,
            )
            return row, {"typed_state": row.typed_state, "executed_trades": count}
        metrics = _metrics(subset)
        skill = _pooled_skill(pooled_predictions, subset["candidate_id"])
        row = StrategyStratumRow(
            key=key,
            executed_trades=count,
            share_of_trades=count / total,
            typed_state="reported",
            metrics=stratum_metrics_from(metrics),
            pooled_model_skill=skill,
            cohort_id=cohort.cohort_id if cohort is not None else None,
        )
        return row, {
            "typed_state": "reported",
            "executed_trades": count,
            "strategy_metrics": metrics.model_dump(mode="json"),
            "trade_ids": sorted(subset["trade_id"].astype(str)),
        }

    strata: list[StrategyStratumRow] = []
    detail: dict[str, Any] = {"strata": {}}
    pooled_row, pooled_detail = _row(
        RegimeStratumKey(stratum="pooled_all"), trades, always_report=True
    )
    strata.append(pooled_row)
    detail["strata"][pooled_row.key.label] = pooled_detail
    covered = trades[covered_mask]
    covered_row, covered_detail = _row(
        RegimeStratumKey(stratum="pooled_regime_covered"), covered, always_report=False
    )
    strata.append(covered_row)
    detail["strata"][covered_row.key.label] = covered_detail
    net_by_regime: dict[int, float] = {}
    for cluster in reporting_ids:
        subset = trades[valid & (clusters == cluster)]
        row, row_detail = _row(
            RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=cluster),
            subset,
            always_report=False,
        )
        strata.append(row)
        detail["strata"][row.key.label] = row_detail
        if row.metrics is not None and row.metrics.net_expectancy_r is not None:
            net_by_regime[cluster] = row.metrics.net_expectancy_r * row.executed_trades
    unassigned = trades[~covered_mask]
    unassigned_row, unassigned_detail = _row(
        RegimeStratumKey(stratum="unassigned"), unassigned, always_report=False
    )
    strata.append(unassigned_row)
    detail["strata"][unassigned_row.key.label] = unassigned_detail

    works_only: tuple[int, ...] = ()
    if len(net_by_regime) >= 2:
        positive = [cluster for cluster, net in net_by_regime.items() if net > 0]
        if len(positive) == 1:
            works_only = (positive[0],)
    abs_total = float(sum(abs(value) for value in net_by_regime.values()))
    top_share = (
        float(max(abs(value) for value in net_by_regime.values()) / abs_total)
        if abs_total > 0
        else None
    )
    reasons = (
        joined.loc[~joined["valid"].astype(bool), "missing_reason"]
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )
    body = CohortDescriptiveBody(
        core_replay_id=core_replay_id,
        executed_trade_table_sha256=executed_trade_table_sha256(trades),
        cost_points=float(cost_points),
        evaluation_config_hash=evaluation_config_hash,
        trades_total=total,
        trades_regime_covered=int(covered_mask.sum()),
        coverage_fraction=float(covered_mask.sum()) / total,
        minimum_trades_per_regime_stratum=int(minimum_trades),
        strata=tuple(strata),
        works_only_in_regime=works_only,
        top_regime_abs_net_r_share=top_share,
        unassigned_reasons={str(k): int(v) for k, v in sorted(reasons.items())},
    )
    detail["cohorts"] = {label: cohort.model_dump(mode="json") for label, cohort in cohorts.items()}
    detail["trade_regimes"] = joined.reset_index().to_dict(orient="records")
    return CohortDescriptiveResult(body=body, detail=detail, trade_regimes=joined.reset_index())
