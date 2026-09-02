"""``cohort_descriptive`` — strategy metrics by descriptive OOS regime (R6.1 §6.G;
R6.1-FIX §3.5 / F-10B).

Executed trades are validated + normalized ONCE
(``_validate_and_normalize_executed_trades``) and that normalized, ordered
frame is the ONLY frame consulted afterwards: it is joined EXACTLY (by
``candidate_id``) onto the descriptive ``RegimeOosAssignmentArtifact``, it
forms every stratum, every ``compute_strategy_metrics`` call and the binding
``executed_trade_table_sha256`` (the raw caller frame never enters a join,
a stratum, a computation or the hash — R6.1-FIX F-10B). Per stratum
(``pooled_all``, ``pooled_regime_covered``, one per canonical reporting id,
``unassigned``) the SAME ``compute_strategy_metrics`` the search gates use is
evaluated — so ``pooled_all`` IS the child's own metrics (asserted by test) —
and a stratum below the stamped ``minimum_trades_per_regime_stratum`` (20) is
typed ``insufficient_regime_partition`` for REPORTABILITY only.

Concentration (R6.1-FIX §3.5, F-09): the raw net-R accounting
(:class:`RegimeNetRAccounting`) sums the per-trade net R of EVERY valid
assigned trade — thin regimes included — so ``top_regime_abs_net_r_share``
and the ``works_only_in_regime`` claim can never understate a thin regime;
the floor governs interval/reportability metrics only, and no statistical
confidence is inferred for a thin stratum. No model is specialized here; no
counterfactual is claimed; nothing is selected.
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
    RegimeNetRAccounting,
    RegimeStratumKey,
    StrategyStratumRow,
    StratumMetrics,
)

__all__ = [
    "CohortDescriptiveResult",
    "executed_trade_table_sha256",
    "normalized_executed_trades",
    "regime_net_r_accounting",
    "stratum_metrics_from",
    "build_cohort_descriptive_body",
]


@dataclass(frozen=True)
class CohortDescriptiveResult:
    body: CohortDescriptiveBody
    detail: dict[str, Any]
    trade_regimes: pd.DataFrame


def normalized_executed_trades(trades: pd.DataFrame, *, tick_size: float = 0.25) -> pd.DataFrame:
    """The validated, normalized executed-trade projection: the normalizer's
    frame with its derived ``_``-prefixed working columns removed, sorted by
    ``trade_id`` (mergesort — deterministic), index reset. This is the ONE
    frame every join / stratum / computation / hash of this module uses."""

    ordered = _validate_and_normalize_executed_trades(trades, tick_size=tick_size)
    columns = [column for column in ordered.columns if not str(column).startswith("_")]
    out = ordered.loc[:, columns].copy()
    out["trade_id"] = out["trade_id"].astype(str)
    return out.sort_values("trade_id", kind="mergesort").reset_index(drop=True)


def executed_trade_table_sha256(trades: pd.DataFrame) -> str:
    """Order-free content hash of an executed-trade table (sorted by trade_id,
    columns sorted). R6.1-FIX F-10B: callers pass the NORMALIZED projection
    (``normalized_executed_trades``), never the raw caller frame."""

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
            elif hasattr(value, "isoformat"):
                record.append(value.isoformat())
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


def regime_net_r_accounting(
    *,
    net_r: pd.Series,
    clusters: np.ndarray,
    valid: np.ndarray,
) -> RegimeNetRAccounting:
    """Plan §3.5 over aligned per-trade vectors (``net_r`` indexed like
    ``clusters`` / ``valid``): raw sums over EVERY valid assigned trade."""

    values = np.asarray(net_r.to_numpy(), dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if len(values) != len(valid) or len(clusters) != len(valid):
        raise ValueError("net R, cluster and validity vectors must align one-to-one")
    net_by_regime: dict[int, float] = {}
    count_by_regime: dict[int, int] = {}
    for value, cluster, ok in zip(values, clusters, valid, strict=True):
        if not ok:
            continue
        key = int(cluster)
        net_by_regime[key] = net_by_regime.get(key, 0.0) + float(value)
        count_by_regime[key] = count_by_regime.get(key, 0) + 1
    unassigned_mask = ~valid
    unassigned_net = float(values[unassigned_mask].sum()) if unassigned_mask.any() else 0.0
    assigned_total = float(sum(net_by_regime.values()))
    mass = float(sum(abs(value) for value in net_by_regime.values()))
    reasons: list[str] = []
    if mass > 0:
        abs_shares: dict[int, float] | None = {
            key: abs(value) / mass for key, value in net_by_regime.items()
        }
        top_share: float | None = max(abs_shares.values())
    else:
        abs_shares = None
        top_share = None
        reasons.append("zero_abs_net_r_mass")
    if assigned_total != 0.0:
        signed: dict[int, float] | None = {
            key: value / assigned_total for key, value in net_by_regime.items()
        }
    else:
        signed = None
        reasons.append("zero_assigned_net_r_total")
    unassigned_count = int(unassigned_mask.sum())
    positive = [key for key, value in net_by_regime.items() if value > 0]
    non_positive = [key for key, value in net_by_regime.items() if value <= 0]
    # plan §3.5 verbatim on the ASSIGNED side; adversarial RA-03: unassigned
    # trades make the claim NULL only when they PREVENT a claim the assigned
    # side supports — a claim the assigned side refutes is FALSE regardless
    # (a single assigned regime is a vacuous truth; ``assigned_regime_count``
    # exposes it)
    assigned_claim = bool(
        assigned_total > 0
        and len(positive) == 1
        and len(non_positive) == len(net_by_regime) - 1
    )
    works: bool | None
    works_id: int | None = None
    works_reason: str | None = None
    if unassigned_count == 0:
        works = assigned_claim
    elif assigned_claim:
        works = None
        works_reason = "incomplete_assignment_accounting"
    else:
        works = False
    if works is True:
        works_id = int(positive[0])
    return RegimeNetRAccounting(
        trade_count_by_regime={int(k): int(v) for k, v in sorted(count_by_regime.items())},
        net_r_by_regime={int(k): float(v) for k, v in sorted(net_by_regime.items())},
        assigned_regime_count=len(net_by_regime),
        assigned_trade_count=int(sum(count_by_regime.values())),
        unassigned_trade_count=unassigned_count,
        unassigned_net_r=unassigned_net,
        assigned_net_r_total=assigned_total,
        abs_net_r_mass=mass,
        abs_net_r_share_by_regime=abs_shares,
        signed_contribution_fraction_by_regime=signed,
        top_regime_abs_net_r_share=top_share,
        works_only_in_regime=works,
        works_only_in_regime_id=works_id,
        works_only_in_regime_reason=works_reason,  # type: ignore[arg-type]
        zero_denominator_reasons=tuple(reasons),  # type: ignore[arg-type]
    )


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
    executed_trade_table_id: str | None = None,
    executed_trade_table_artifact_sha256: str | None = None,
) -> CohortDescriptiveResult:
    """Strata over the validated, normalized executed-trade table under the
    descriptive OOS assignment (module docstring). ``executed_trade_table_id``
    / ``executed_trade_table_artifact_sha256`` name the persisted
    executed-trade table artifact the SERVICE verified ``trades`` against
    (R6.1-FIX §3.7; adversarial RA-01) — both or neither — and ride the body
    beside the normalized-frame hash."""

    ordered = normalized_executed_trades(trades, tick_size=tick_size)
    if ordered.empty:
        raise ValueError("cohort_descriptive requires at least one executed trade")
    joined = regime_for_trades(ordered, assignment)
    joined = joined.set_index("trade_id")
    trade_ids = ordered["trade_id"].astype(str)
    clusters = joined.loc[trade_ids, "canonical_reporting_cluster_id"].to_numpy()
    valid = joined.loc[trade_ids, "valid"].to_numpy().astype(bool)
    total = int(len(ordered))
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
        RegimeStratumKey(stratum="pooled_all"), ordered, always_report=True
    )
    strata.append(pooled_row)
    detail["strata"][pooled_row.key.label] = pooled_detail
    covered = ordered[covered_mask]
    covered_row, covered_detail = _row(
        RegimeStratumKey(stratum="pooled_regime_covered"), covered, always_report=False
    )
    strata.append(covered_row)
    detail["strata"][covered_row.key.label] = covered_detail
    for cluster in reporting_ids:
        subset = ordered[valid & (clusters == cluster)]
        row, row_detail = _row(
            RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=cluster),
            subset,
            always_report=False,
        )
        strata.append(row)
        detail["strata"][row.key.label] = row_detail
    unassigned = ordered[~covered_mask]
    unassigned_row, unassigned_detail = _row(
        RegimeStratumKey(stratum="unassigned"), unassigned, always_report=False
    )
    strata.append(unassigned_row)
    detail["strata"][unassigned_row.key.label] = unassigned_detail

    # R6.1-FIX §3.5: the raw accounting over EVERY valid assigned trade — the
    # per-trade net R vector of the SAME normalized frame the metrics use
    from ..search.strategy_metrics import per_trade_net_r  # noqa: PLC0415

    net = per_trade_net_r(ordered, cost_points=cost_points, tick_size=tick_size)
    net = net.loc[trade_ids.to_numpy()]
    accounting = regime_net_r_accounting(net_r=net, clusters=clusters, valid=valid)
    works_id = accounting.works_only_in_regime_id
    works_only: tuple[int, ...] = (
        (int(works_id),) if accounting.works_only_in_regime is True and works_id is not None else ()
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
        executed_trade_table_sha256=executed_trade_table_sha256(ordered),
        executed_trade_table_id=executed_trade_table_id,
        executed_trade_table_artifact_sha256=executed_trade_table_artifact_sha256,
        cost_points=float(cost_points),
        evaluation_config_hash=evaluation_config_hash,
        trades_total=total,
        trades_regime_covered=int(covered_mask.sum()),
        coverage_fraction=float(covered_mask.sum()) / total,
        minimum_trades_per_regime_stratum=int(minimum_trades),
        strata=tuple(strata),
        works_only_in_regime=works_only,
        top_regime_abs_net_r_share=accounting.top_regime_abs_net_r_share,
        net_r_accounting=accounting,
        unassigned_reasons={str(k): int(v) for k, v in sorted(reasons.items())},
    )
    detail["cohorts"] = {label: cohort.model_dump(mode="json") for label, cohort in cohorts.items()}
    detail["trade_regimes"] = joined.reset_index().to_dict(orient="records")
    detail["net_r_accounting"] = accounting.model_dump(mode="json")
    return CohortDescriptiveResult(body=body, detail=detail, trade_regimes=joined.reset_index())
