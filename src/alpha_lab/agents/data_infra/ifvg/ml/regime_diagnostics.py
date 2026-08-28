"""Regime stability/diagnostic report builders (ML plan §3/§5.4).

Builders only — descriptive facts into the frozen
:class:`RegimeStabilityReport`; nothing here promotes, retrains, or
disables anything. Bootstrap stability = seeded KMeans refits on resampled
rows of the REFERENCE fold's training matrix only, Hungarian-aligned
(exact ascending-local-id tie-break) to the fold fit, scored by adjusted
mutual information; internal scores (silhouette) are kept descriptive —
no single internal score can promote (§5.4).

Temporal facts (review F3) are a true point-in-time timeline: they are
computed over OUT-OF-SAMPLE test rows ordered by the observation's as-of
timestamp, within each fold (no cross-fold concatenation, no row-id
ordering, no double counting of overlapping training windows) —
``temporal_order_policy = oos_test_rows_by_observation_ts_within_fold_v1``.
Cross-fold centroid comparisons use the scaled INPUT-feature coordinates
only (``alignment_space = scaled_input_features_v1``, review F9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from ..search.identities import ImmutableMap
from .regime_alignment import (
    align_reporting_labels,
    feature_space_centroids,
    hungarian_lexicographic,
)
from .regime_contracts import REGIME_PROPOSED_DEFAULTS, RegimeStabilityReport
from .regime_preprocessing import keyed_observations

__all__ = [
    "build_stability_report",
    "transition_matrix_from_sequence",
    "oos_regime_timeline",
]


def _aligned_labels(
    reference_centroids: np.ndarray, centroids: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    cost = np.linalg.norm(
        centroids[:, None, :] - reference_centroids[None, :, :], axis=2
    )
    columns = hungarian_lexicographic(cost)
    relabel = {local: int(canonical) for local, canonical in enumerate(columns)}
    return np.asarray([relabel[int(label)] for label in labels], dtype=int)


def transition_matrix_from_sequence(
    sequence: list[int] | np.ndarray, cluster_count: int
) -> tuple[tuple[float, ...], ...]:
    """Row-normalized regime→regime transition frequencies (temporal order)."""

    counts = np.zeros((cluster_count, cluster_count), dtype=float)
    values = [int(v) for v in sequence]
    for previous, current in zip(values[:-1], values[1:], strict=False):
        counts[previous, current] += 1.0
    return _normalize_rows(counts)


def _normalize_rows(counts: np.ndarray) -> tuple[tuple[float, ...], ...]:
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(
            counts.sum(axis=1, keepdims=True) > 0,
            counts / counts.sum(axis=1, keepdims=True),
            0.0,
        )
    return tuple(tuple(round(float(v), 6) for v in row) for row in normalized)


def oos_regime_timeline(assignments: pd.DataFrame) -> pd.DataFrame:
    """The point-in-time regime timeline: valid OUT-OF-SAMPLE rows with a
    known as-of timestamp, ordered by (fold_index, observation_ts_utc,
    row_id). Columns: fold_index, observation_ts_utc, row_id,
    canonical_reporting_cluster_id."""

    if assignments.empty or "observation_ts_utc" not in assignments.columns:
        return pd.DataFrame(
            columns=["fold_index", "observation_ts_utc", "row_id", "canonical_reporting_cluster_id"]
        )
    rows = assignments[
        (assignments["partition"].astype(str) == "test")
        & assignments["valid"].astype(bool)
        & assignments["observation_ts_utc"].notna()
        & assignments["canonical_reporting_cluster_id"].notna()
    ].copy()
    rows["_ts"] = pd.to_datetime(rows["observation_ts_utc"], utc=True)
    rows = rows.sort_values(["fold_index", "_ts", "row_id"], kind="stable")
    return rows[
        ["fold_index", "observation_ts_utc", "row_id", "canonical_reporting_cluster_id"]
    ].reset_index(drop=True)


def _temporal_facts(
    assignments: pd.DataFrame, k: int
) -> tuple[float | None, tuple[tuple[float, ...], ...], int]:
    """(persistence, transition matrix, transition count) over the OOS
    timeline, transitions counted WITHIN each fold only."""

    timeline = oos_regime_timeline(assignments)
    counts = np.zeros((k, k), dtype=float)
    same = 0
    transitions = 0
    for _fold_index, group in timeline.groupby("fold_index", sort=True):
        sequence = [int(v) for v in group["canonical_reporting_cluster_id"].tolist()]
        for previous, current in zip(sequence[:-1], sequence[1:], strict=False):
            counts[previous, current] += 1.0
            transitions += 1
            same += int(previous == current)
    if transitions == 0:
        return None, (), 0
    return round(same / transitions, 6), _normalize_rows(counts), transitions


def build_stability_report(
    frame: pd.DataFrame,
    protocol,
    fold_fits,
    assignments: pd.DataFrame,
    *,
    bootstrap_refits: int = 50,
) -> RegimeStabilityReport:
    """Bootstrap AMI + OOS-timeline persistence/transitions + recurrence + separation."""

    ami_floor = float(
        REGIME_PROPOSED_DEFAULTS["bootstrap_aligned_ami_advisory_floor"]["value"]
    )
    k = int(protocol.payload.resolved_cluster_count)
    if not fold_fits or assignments.empty:
        return RegimeStabilityReport(
            bootstrap_refit_count=0,
            bootstrap_aligned_ami_mean=None,
            bootstrap_aligned_ami_low=None,
            per_cluster_agreement={},
            temporal_persistence=None,
            transition_matrix=(),
            fold_to_fold_recurrence=None,
            separation_min_centroid_distance=None,
            silhouette_descriptive=None,
            semantic_descriptors={},
            stability_gates_passed=False,
            gate_failures=("no_valid_folds",),
        )

    reference = sorted(fold_fits, key=lambda fit: fit.fold_index)[0]
    indexed = keyed_observations(frame)
    # the reference fold's TRAINING rows only — test rows never enter
    train_matrix = reference.preprocessing.transform(
        indexed.loc[list(reference.preprocessing.training_row_ids)]
    )
    base_labels = reference.estimator.predict(train_matrix)

    rng = np.random.default_rng(7)
    pinned = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "tol": 1e-4,
        "algorithm": "lloyd",
    }
    ami_values: list[float] = []
    agreement = np.zeros(k, dtype=float)
    agreement_counts = np.zeros(k, dtype=float)
    for refit_index in range(int(bootstrap_refits)):
        sample = rng.integers(0, len(train_matrix), size=len(train_matrix))
        refit = KMeans(
            n_clusters=k, random_state=1000 + refit_index, **pinned
        ).fit(train_matrix[sample])
        refit_labels = _aligned_labels(
            reference.centroids_scaled,
            np.asarray(refit.cluster_centers_, dtype=float),
            refit.predict(train_matrix),
        )
        ami_values.append(
            float(adjusted_mutual_info_score(base_labels, refit_labels))
        )
        for cluster in range(k):
            mask = base_labels == cluster
            if mask.any():
                agreement[cluster] += float((refit_labels[mask] == cluster).mean())
                agreement_counts[cluster] += 1.0
    ami_mean = float(np.mean(ami_values)) if ami_values else None
    ami_low = float(np.quantile(ami_values, 0.05)) if ami_values else None
    per_cluster = {
        int(cluster): round(float(agreement[cluster] / agreement_counts[cluster]), 6)
        for cluster in range(k)
        if agreement_counts[cluster] > 0
    }

    persistence, transitions, transition_count = _temporal_facts(assignments, k)

    # fold-to-fold recurrence: mean aligned-centroid distance to the
    # reference in the scaled INPUT-feature coordinates
    reference_space = feature_space_centroids(reference)
    recurrence = None
    if len(fold_fits) > 1:
        distances: list[float] = []
        for fit in fold_fits:
            if fit.fold_index == reference.fold_index:
                continue
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
    silhouette = None
    if len(set(base_labels)) > 1 and len(train_matrix) > k:
        silhouette = round(float(silhouette_score(train_matrix, base_labels)), 6)

    semantic = align_reporting_labels(tuple(fold_fits)).semantic_descriptors

    failures: list[str] = []
    if ami_mean is not None and ami_mean < ami_floor:
        failures.append("bootstrap_aligned_ami_below_advisory_floor")
    return RegimeStabilityReport(
        bootstrap_refit_count=int(bootstrap_refits),
        bootstrap_aligned_ami_mean=round(ami_mean, 6) if ami_mean is not None else None,
        bootstrap_aligned_ami_low=round(ami_low, 6) if ami_low is not None else None,
        per_cluster_agreement=ImmutableMap(per_cluster),
        temporal_transition_count=transition_count,
        temporal_persistence=persistence,
        transition_matrix=transitions,
        fold_to_fold_recurrence=recurrence,
        separation_min_centroid_distance=separation,
        silhouette_descriptive=silhouette,
        semantic_descriptors=ImmutableMap(
            {int(key): value for key, value in semantic.items()}
        ),
        stability_gates_passed=not failures,
        gate_failures=tuple(failures),
    )
