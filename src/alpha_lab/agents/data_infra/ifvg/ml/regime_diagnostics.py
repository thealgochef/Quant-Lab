"""Regime stability/diagnostic report builders (ML plan §3/§5.4; R6.1 D10/D11).

Builders only — descriptive facts into the frozen
:class:`RegimeStabilityReport`; nothing here promotes, retrains, or
disables anything.

**Bootstrap stability runs on EVERY valid fold (D10).** For fold ``k`` the
resampler is ``default_rng(7 + k)`` and refit ``i`` uses seed
``1000 + 1000·k + i`` with the REGISTRY-pinned KMeans parameters (minus
``random_state``) — fold 0 therefore reproduces the R6 numbers exactly. Each
refit is Hungarian-aligned (exact ascending-local-id tie-break) onto the
fold's own fit and scored by adjusted mutual information; the reference
fold's values are exported as ``reference_fold_bootstrap_stability`` (and
the R6 scalar aliases), and the promotion gate
``minimum_bootstrap_aligned_ami_mean`` applies to the PROTOCOL-WIDE MINIMUM
fold mean (``bootstrap_gate_scope = protocol_wide_minimum_fold_mean_v1``).
``ami_p05`` is reported, never gated. The stamped budget is 50 refits per
fold under a 400-refit cap (``_bootstrap_plan`` reduces the per-fold count
only beyond 8 valid folds).

**Temporal semantics per grain (D11).** Candidate-stage / decision-row
observations are NOT a regular time series, so their OOS timeline yields a
``candidate_event_transition_matrix``: adjacent OOS candidates within a
fold count as a pair only when they share the trading day AND the named
session (``IFVG_DOC_SESSION_SCHEME`` via ``classify_session``) and their
elapsed time is at most the stamped maximum gap (7 200 s); dropped pairs
are counted by cause. Only the regular 5m/15m panel carries ordinary
``temporal_persistence`` / ``transition_matrix`` semantics: consecutive
completed bars within one trading day (elapsed == interval; a partial or
missing bar breaks the chain). Cross-fold centroid comparisons use the
scaled INPUT-feature coordinates only (``alignment_space``).
"""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from strategy_core.constants import IFVG_DOC_SESSION_SCHEME
from strategy_core.decisions.sessions import classify_session

from ..search.identities import ImmutableMap
from .regime_algorithms import KMEANS_ALGORITHM_KEY, REGIME_ALGORITHM_REGISTRY
from .regime_alignment import (
    align_reporting_labels,
    feature_space_centroids,
    hungarian_lexicographic,
)
from .regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2,
    TEMPORAL_ORDER_POLICY_PANEL_V2,
    FoldBootstrapStability,
    ObservationGranularity,
    RegimeStabilityReport,
)
from .regime_preprocessing import keyed_observations

__all__ = [
    "build_stability_report",
    "oos_regime_timeline",
    "candidate_event_temporal_facts",
    "panel_temporal_facts",
    "bootstrap_plan",
    "bootstrap_fold_stability",
]

SESSION_SCHEME_ID = "ifvg_doc_session_scheme_et_v1"


def _aligned_labels(
    reference_centroids: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    cost = np.linalg.norm(
        centroids[:, None, :] - reference_centroids[None, :, :], axis=2
    )
    columns = hungarian_lexicographic(cost)
    relabel = {local: int(canonical) for local, canonical in enumerate(columns)}
    return np.asarray([relabel[int(label)] for label in labels], dtype=int)


def _normalize_rows(counts: np.ndarray) -> tuple[tuple[float, ...], ...]:
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(
            counts.sum(axis=1, keepdims=True) > 0,
            counts / counts.sum(axis=1, keepdims=True),
            0.0,
        )
    return tuple(tuple(round(float(v), 6) for v in row) for row in normalized)


def _session_facts(ts_utc: pd.Timestamp) -> tuple[str | None, str]:
    info = classify_session(ts_utc.to_pydatetime().astimezone(UTC), IFVG_DOC_SESSION_SCHEME)
    return (info.trading_day.isoformat() if info.trading_day else None), info.session


def oos_regime_timeline(assignments: pd.DataFrame) -> pd.DataFrame:
    """The point-in-time regime timeline: valid OUT-OF-SAMPLE rows with a
    known as-of timestamp, ordered by (fold_index, observation_ts_utc,
    row_id). Columns: fold_index, observation_ts_utc, row_id,
    canonical_reporting_cluster_id, trading_day, session, elapsed_seconds
    (seconds since the previous row of the same fold; NaN for the first)."""

    columns = [
        "fold_index",
        "observation_ts_utc",
        "row_id",
        "canonical_reporting_cluster_id",
        "trading_day",
        "session",
        "elapsed_seconds",
    ]
    if assignments.empty or "observation_ts_utc" not in assignments.columns:
        return pd.DataFrame(columns=columns)
    rows = assignments[
        (assignments["partition"].astype(str) == "test")
        & assignments["valid"].astype(bool)
        & assignments["observation_ts_utc"].notna()
        & assignments["canonical_reporting_cluster_id"].notna()
    ].copy()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    rows["_ts"] = pd.to_datetime(rows["observation_ts_utc"], utc=True)
    rows = rows.sort_values(["fold_index", "_ts", "row_id"], kind="stable")
    facts = [_session_facts(ts) for ts in rows["_ts"]]
    rows["trading_day"] = [day for day, _session in facts]
    rows["session"] = [session for _day, session in facts]
    rows["elapsed_seconds"] = (
        rows.groupby("fold_index", sort=False)["_ts"].diff().dt.total_seconds()
    )
    return rows[columns].reset_index(drop=True)


def candidate_event_temporal_facts(
    assignments: pd.DataFrame, k: int, *, maximum_gap_seconds: int
) -> dict:
    """The CANDIDATE-EVENT transition matrix over the OOS timeline: pairs of
    adjacent OOS candidates WITHIN a fold count only when they share the
    trading day and the named session and are at most ``maximum_gap_seconds``
    apart; dropped pairs are counted by cause."""

    timeline = oos_regime_timeline(assignments)
    counts = np.zeros((k, k), dtype=float)
    same = 0
    counted = 0
    dropped_gap = 0
    dropped_day = 0
    dropped_session = 0
    elapsed_values: list[float] = []
    for _fold_index, group in timeline.groupby("fold_index", sort=True):
        previous = None
        for row in group.itertuples():
            if previous is not None:
                elapsed = float(row.elapsed_seconds)
                if row.trading_day != previous.trading_day:
                    dropped_day += 1
                elif row.session != previous.session:
                    dropped_session += 1
                elif not np.isfinite(elapsed) or elapsed > maximum_gap_seconds:
                    dropped_gap += 1
                else:
                    a = int(previous.canonical_reporting_cluster_id)
                    b = int(row.canonical_reporting_cluster_id)
                    counts[a, b] += 1.0
                    counted += 1
                    same += int(a == b)
                    elapsed_values.append(elapsed)
            previous = row
    summary: dict[str, float] = {}
    if elapsed_values:
        array = np.asarray(elapsed_values, dtype=float)
        summary = {
            "p50": round(float(np.quantile(array, 0.5)), 3),
            "p90": round(float(np.quantile(array, 0.9)), 3),
            "max": round(float(array.max()), 3),
        }
    return {
        "transition_matrix": _normalize_rows(counts) if counted else (),
        "persistence": round(same / counted, 6) if counted else None,
        "pairs_counted": counted,
        "pairs_dropped_gap": dropped_gap,
        "pairs_dropped_boundary": {"trading_day": dropped_day, "session": dropped_session},
        "elapsed_seconds_summary": summary,
    }


def panel_temporal_facts(
    assignments: pd.DataFrame,
    frame: pd.DataFrame,
    k: int,
    *,
    interval_seconds: int,
) -> dict:
    """Ordinary temporal semantics on the REGULAR panel: adjacent OOS bars
    within a fold count when they share the trading day, are exactly one
    interval apart, and the earlier bar is not partial."""

    timeline = oos_regime_timeline(assignments)
    indexed = keyed_observations(frame)
    partial = (
        indexed["is_final_partial"].astype(bool)
        if "is_final_partial" in indexed.columns
        else pd.Series(False, index=indexed.index)
    )
    day_column = (
        indexed["trading_day"].astype(str) if "trading_day" in indexed.columns else None
    )
    counts = np.zeros((k, k), dtype=float)
    same = 0
    counted = 0
    dropped_gap = 0
    dropped_boundary = 0
    for _fold_index, group in timeline.groupby("fold_index", sort=True):
        previous = None
        for row in group.itertuples():
            if previous is not None:
                if day_column is not None:
                    previous_day = day_column.get(str(previous.row_id))
                    current_day = day_column.get(str(row.row_id))
                else:
                    previous_day, current_day = previous.trading_day, row.trading_day
                elapsed = float(row.elapsed_seconds)
                if previous_day != current_day:
                    dropped_boundary += 1
                elif bool(partial.get(str(previous.row_id), False)) or (
                    not np.isfinite(elapsed) or int(round(elapsed)) != int(interval_seconds)
                ):
                    dropped_gap += 1
                else:
                    a = int(previous.canonical_reporting_cluster_id)
                    b = int(row.canonical_reporting_cluster_id)
                    counts[a, b] += 1.0
                    counted += 1
                    same += int(a == b)
            previous = row
    return {
        "transition_matrix": _normalize_rows(counts) if counted else (),
        "persistence": round(same / counted, 6) if counted else None,
        "transition_count": counted,
        "pairs_dropped_gap": dropped_gap,
        "pairs_dropped_boundary": dropped_boundary,
    }


def bootstrap_plan(valid_fold_count: int, *, refits_per_fold: int, total_cap: int) -> int:
    """Applied refits per fold under the total cap (reduced only when the
    requested budget exceeds the cap)."""

    if valid_fold_count <= 0 or refits_per_fold <= 0:
        return 0
    if refits_per_fold * valid_fold_count <= total_cap:
        return int(refits_per_fold)
    return max(1, int(total_cap // valid_fold_count))


def _pinned_kmeans_parameters() -> dict:
    pinned = dict(REGIME_ALGORITHM_REGISTRY[KMEANS_ALGORITHM_KEY].pinned_parameters)
    pinned.pop("random_state", None)
    return pinned


def bootstrap_fold_stability(
    fit, train_matrix: np.ndarray, *, k: int, refits: int
) -> FoldBootstrapStability:
    """One fold's seeded bootstrap: ``default_rng(7 + fold)`` resampling,
    refit seeds ``1000 + 1000·fold + i``, registry-pinned parameters."""

    fold = int(fit.fold_index)
    base_labels = fit.estimator.predict(train_matrix)
    if refits <= 0:
        return FoldBootstrapStability(
            fold_index=fold,
            refit_count=0,
            aligned_ami_mean=None,
            aligned_ami_p05=None,
            per_cluster_agreement={},
            undefined_reason="no_refits_requested",
        )
    if len(train_matrix) <= k or len(set(int(v) for v in base_labels)) < 2:
        return FoldBootstrapStability(
            fold_index=fold,
            refit_count=0,
            aligned_ami_mean=None,
            aligned_ami_p05=None,
            per_cluster_agreement={},
            undefined_reason="insufficient_training_rows_or_clusters",
        )
    rng = np.random.default_rng(7 + fold)
    pinned = _pinned_kmeans_parameters()
    ami_values: list[float] = []
    agreement = np.zeros(k, dtype=float)
    agreement_counts = np.zeros(k, dtype=float)
    for refit_index in range(int(refits)):
        sample = rng.integers(0, len(train_matrix), size=len(train_matrix))
        refit = KMeans(
            n_clusters=k, random_state=1000 + 1000 * fold + refit_index, **pinned
        ).fit(train_matrix[sample])
        refit_labels = _aligned_labels(
            fit.centroids_scaled,
            np.asarray(refit.cluster_centers_, dtype=float),
            refit.predict(train_matrix),
        )
        ami_values.append(float(adjusted_mutual_info_score(base_labels, refit_labels)))
        for cluster in range(k):
            mask = base_labels == cluster
            if mask.any():
                agreement[cluster] += float((refit_labels[mask] == cluster).mean())
                agreement_counts[cluster] += 1.0
    per_cluster = {
        int(cluster): round(float(agreement[cluster] / agreement_counts[cluster]), 6)
        for cluster in range(k)
        if agreement_counts[cluster] > 0
    }
    return FoldBootstrapStability(
        fold_index=fold,
        refit_count=int(refits),
        aligned_ami_mean=round(float(np.mean(ami_values)), 6),
        aligned_ami_p05=round(float(np.quantile(ami_values, 0.05)), 6),
        per_cluster_agreement=ImmutableMap(per_cluster),
        undefined_reason=None,
    )


def _empty_report(ami_gate: float, policy: str) -> RegimeStabilityReport:
    return RegimeStabilityReport(
        bootstrap_refit_count=0,
        bootstrap_refits_per_fold_requested=0,
        bootstrap_total_refit_cap=int(
            REGIME_PROPOSED_DEFAULTS["bootstrap_total_refit_cap"]["value"]
        ),
        bootstrap_refits_per_fold_applied=0,
        bootstrap_aligned_ami_mean=None,
        bootstrap_aligned_ami_low=None,
        per_cluster_agreement={},
        reference_fold_bootstrap_stability=None,
        per_fold_bootstrap_stability=(),
        bootstrap_fold_coverage=None,
        protocol_min_bootstrap_aligned_ami_mean=None,
        protocol_min_bootstrap_aligned_ami_p05=None,
        minimum_bootstrap_aligned_ami_mean_applied=ami_gate,
        temporal_order_policy=policy,
        session_scheme_id=SESSION_SCHEME_ID,
        temporal_persistence=None,
        transition_matrix=(),
        candidate_event_maximum_gap_seconds=int(
            REGIME_PROPOSED_DEFAULTS["candidate_event_maximum_gap_seconds"]["value"]
        ),
        fold_to_fold_recurrence=None,
        separation_min_centroid_distance=None,
        silhouette_descriptive=None,
        semantic_descriptors={},
        stability_gates_passed=False,
        gate_failures=("no_valid_folds",),
    )


def build_stability_report(
    frame: pd.DataFrame,
    protocol,
    fold_fits,
    assignments: pd.DataFrame,
    *,
    bootstrap_refits: int = 50,
) -> RegimeStabilityReport:
    """Per-fold bootstrap AMI + grain-aware temporal facts + recurrence +
    separation + descriptive silhouette."""

    ami_gate = float(REGIME_PROPOSED_DEFAULTS["minimum_bootstrap_aligned_ami_mean"]["value"])
    total_cap = int(REGIME_PROPOSED_DEFAULTS["bootstrap_total_refit_cap"]["value"])
    max_gap = int(REGIME_PROPOSED_DEFAULTS["candidate_event_maximum_gap_seconds"]["value"])
    k = int(protocol.payload.resolved_cluster_count)
    grain = ObservationGranularity(protocol.payload.observation_granularity)
    panel_grain = grain is ObservationGranularity.CONTEXT_BAR_PANEL
    policy = TEMPORAL_ORDER_POLICY_PANEL_V2 if panel_grain else (
        TEMPORAL_ORDER_POLICY_CANDIDATE_EVENT_V2
    )
    if not fold_fits or assignments.empty:
        return _empty_report(ami_gate, policy)

    ordered = sorted(fold_fits, key=lambda fit: fit.fold_index)
    indexed = keyed_observations(frame)
    applied = bootstrap_plan(
        len(ordered), refits_per_fold=int(bootstrap_refits), total_cap=total_cap
    )
    per_fold: list[FoldBootstrapStability] = []
    for fit in ordered:
        # the fold's TRAINING rows only — test rows never enter
        train_matrix = fit.preprocessing.transform(
            indexed.loc[list(fit.preprocessing.training_row_ids)]
        )
        per_fold.append(bootstrap_fold_stability(fit, train_matrix, k=k, refits=applied))
    reference = per_fold[0]
    defined = [entry for entry in per_fold if entry.aligned_ami_mean is not None]
    fold_coverage = round(len(defined) / len(per_fold), 6) if per_fold else None
    protocol_min_mean = (
        min(entry.aligned_ami_mean for entry in defined) if defined else None
    )
    protocol_min_p05 = (
        min(entry.aligned_ami_p05 for entry in defined if entry.aligned_ami_p05 is not None)
        if defined
        else None
    )

    if panel_grain:
        panel = panel_temporal_facts(
            assignments,
            frame,
            k,
            interval_seconds=int(protocol.payload.panel_interval_seconds or 0),
        )
        candidate: dict = {}
    else:
        panel = {}
        candidate = candidate_event_temporal_facts(assignments, k, maximum_gap_seconds=max_gap)

    # fold-to-fold recurrence: mean aligned-centroid distance to the
    # reference in the scaled INPUT-feature coordinates
    reference_fit = ordered[0]
    reference_space = feature_space_centroids(reference_fit)
    recurrence = None
    if len(ordered) > 1:
        distances: list[float] = []
        for fit in ordered[1:]:
            fit_space = feature_space_centroids(fit)
            cost = np.linalg.norm(
                fit_space[:, None, :] - reference_space[None, :, :], axis=2
            )
            columns = hungarian_lexicographic(cost)
            distances.append(
                float(np.mean([cost[local, canonical] for local, canonical in enumerate(columns)]))
            )
        recurrence = round(float(np.mean(distances)), 6) if distances else None

    centroid_gaps = [
        float(np.linalg.norm(reference_space[i] - reference_space[j]))
        for i in range(k)
        for j in range(i + 1, k)
    ]
    separation = round(min(centroid_gaps), 6) if centroid_gaps else None
    reference_matrix = reference_fit.preprocessing.transform(
        indexed.loc[list(reference_fit.preprocessing.training_row_ids)]
    )
    base_labels = reference_fit.estimator.predict(reference_matrix)
    silhouette = None
    if len(set(base_labels)) > 1 and len(reference_matrix) > k:
        silhouette = round(float(silhouette_score(reference_matrix, base_labels)), 6)

    semantic = align_reporting_labels(tuple(fold_fits)).semantic_descriptors

    failures: list[str] = []
    if any(entry.aligned_ami_mean is None for entry in per_fold):
        failures.append("bootstrap_undefined_fold")
    if protocol_min_mean is not None and protocol_min_mean < ami_gate:
        failures.append("bootstrap_aligned_ami_below_minimum")
    return RegimeStabilityReport(
        bootstrap_refit_count=int(reference.refit_count),
        bootstrap_refits_per_fold_requested=int(bootstrap_refits),
        bootstrap_total_refit_cap=total_cap,
        bootstrap_refits_per_fold_applied=int(applied),
        bootstrap_aligned_ami_mean=reference.aligned_ami_mean,
        bootstrap_aligned_ami_low=reference.aligned_ami_p05,
        per_cluster_agreement=reference.per_cluster_agreement,
        reference_fold_bootstrap_stability=reference,
        per_fold_bootstrap_stability=tuple(per_fold),
        bootstrap_fold_coverage=fold_coverage,
        protocol_min_bootstrap_aligned_ami_mean=protocol_min_mean,
        protocol_min_bootstrap_aligned_ami_p05=protocol_min_p05,
        minimum_bootstrap_aligned_ami_mean_applied=ami_gate,
        temporal_order_policy=policy,
        session_scheme_id=SESSION_SCHEME_ID,
        temporal_transition_count=int(panel.get("transition_count", 0)),
        temporal_persistence=panel.get("persistence"),
        transition_matrix=panel.get("transition_matrix", ()),
        panel_pairs_dropped_gap=int(panel.get("pairs_dropped_gap", 0)),
        panel_pairs_dropped_boundary=int(panel.get("pairs_dropped_boundary", 0)),
        candidate_event_transition_matrix=candidate.get("transition_matrix", ()),
        candidate_event_persistence=candidate.get("persistence"),
        candidate_event_pairs_counted=int(candidate.get("pairs_counted", 0)),
        candidate_event_pairs_dropped_gap=int(candidate.get("pairs_dropped_gap", 0)),
        candidate_event_pairs_dropped_boundary=ImmutableMap(
            candidate.get("pairs_dropped_boundary", {})
        ),
        candidate_event_elapsed_seconds_summary=ImmutableMap(
            candidate.get("elapsed_seconds_summary", {})
        ),
        candidate_event_maximum_gap_seconds=max_gap,
        fold_to_fold_recurrence=recurrence,
        separation_min_centroid_distance=separation,
        silhouette_descriptive=silhouette,
        semantic_descriptors=ImmutableMap(
            {int(key): value for key, value in semantic.items()}
        ),
        stability_gates_passed=not failures,
        gate_failures=tuple(failures),
    )
