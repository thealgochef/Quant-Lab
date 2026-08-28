"""Reporting-only cluster-label alignment (ML plan §5.4; 7B.22-9).

``centroid_min_distance_hungarian_v1``: the reference is the FIRST valid
fold's centroids; every later fold is matched via
``scipy.optimize.linear_sum_assignment`` on the centroid-distance matrix
over the scaled INPUT-feature coordinates (missing-indicator columns are
fold-dependent and never enter an alignment — review F9), with ties broken
EXACTLY by ascending fold-local id: among every optimal assignment the
lexicographically smallest ``(canonical_of_local_0, canonical_of_local_1,
…)`` tuple wins (review F8 — enumerated for k ≤ 8; the plan's k ∈ {3,4,5}).
The output is a pure fold-local → canonical reporting map plus top-|z|
semantic descriptors of the reference centroids. Alignment runs in the
reporting layer only, AFTER predictions exist — it never changes any
assignment's fold-local id (a test proves prediction-hash invariance), and
canonical ids are NOMINAL: the UI never implies ``regime 2 > regime 1``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "RegimeAlignmentMap",
    "hungarian_lexicographic",
    "feature_space_centroids",
    "align_reporting_labels",
]

_ENUMERATION_LIMIT = 8


def hungarian_lexicographic(cost: np.ndarray) -> tuple[int, ...]:
    """Min-cost assignment (rows → columns) with the exact tie-break: among
    all optimal assignments the lexicographically smallest column tuple —
    i.e. ascending row (fold-local id) priority. Returns ``columns`` indexed
    by row. Enumerates optimal assignments for k ≤ 8; beyond that the
    scipy solution stands (documented fallback)."""

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2 or cost.shape[0] != cost.shape[1]:
        raise ValueError("alignment cost matrix must be square (same k in every fold)")
    rows, columns = linear_sum_assignment(cost)
    scipy_solution = tuple(int(columns[list(rows).index(r)]) for r in range(cost.shape[0]))
    optimal = float(cost[rows, columns].sum())
    k = cost.shape[0]
    if k > _ENUMERATION_LIMIT:
        return scipy_solution
    tolerance = 1e-9 * max(1.0, abs(optimal))
    best: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(k)):
        total = float(sum(cost[i, permutation[i]] for i in range(k)))
        if total <= optimal + tolerance and (best is None or permutation < best):
            best = tuple(int(c) for c in permutation)
    return best if best is not None else scipy_solution


def _coordinate_ranks(centroids: np.ndarray) -> tuple[int, ...]:
    """local id → canonical rank under the lexicographic order of the
    centroid's feature-space coordinates (rounded to 1e-9; ties by local id)."""

    keys = [
        (tuple(round(float(v), 9) for v in centroids[local]), local)
        for local in range(len(centroids))
    ]
    order = sorted(range(len(centroids)), key=lambda local: keys[local])
    ranks = [0] * len(centroids)
    for rank, local in enumerate(order):
        ranks[local] = rank
    return tuple(ranks)


def feature_space_centroids(fit) -> np.ndarray:
    """A fit's centroids restricted to the scaled INPUT-feature coordinates
    (the first ``len(features)`` columns — indicator columns excluded)."""

    width = len(fit.preprocessing.features)
    return np.asarray(fit.centroids_scaled, dtype=float)[:, :width]


@dataclass(frozen=True, slots=True)
class RegimeAlignmentMap:
    """fold_index → (fold_local_cluster_id → canonical_reporting_cluster_id)."""

    reference_fold_index: int | None
    alignment_feature_names: tuple[str, ...]
    mapping: dict[int, dict[int, int]]
    semantic_descriptors: dict[int, tuple[tuple[str, float], ...]]

    def canonical_for(self, fold_index: int, fold_local_cluster_id: int) -> int | None:
        fold_map = self.mapping.get(int(fold_index))
        if fold_map is None:
            return None
        return fold_map.get(int(fold_local_cluster_id))


def _descriptors(
    centroid: np.ndarray, feature_names: tuple[str, ...], *, top: int = 5
) -> tuple[tuple[str, float], ...]:
    order = np.argsort(-np.abs(centroid[: len(feature_names)]), kind="stable")[:top]
    return tuple(
        (feature_names[index], round(float(centroid[index]), 6)) for index in order
    )


def align_reporting_labels(fold_fits) -> RegimeAlignmentMap:
    """Hungarian match of every valid fold's centroids onto the reference.

    ``fold_fits`` is the run's tuple of ``RegimeFoldFit``s (each carrying
    ``fold_index``, ``centroids_scaled``, and the fitted preprocessing whose
    ``features`` name the aligned coordinates).
    """

    if not fold_fits:
        return RegimeAlignmentMap(
            reference_fold_index=None,
            alignment_feature_names=(),
            mapping={},
            semantic_descriptors={},
        )
    ordered = sorted(fold_fits, key=lambda fit: fit.fold_index)
    reference = ordered[0]
    names = tuple(reference.preprocessing.features)
    reference_centroids = feature_space_centroids(reference)
    # The reference fold's canonical ids are the ranks of its centroids
    # under the lexicographic order of their feature-space coordinates (ties
    # by local id) — so canonical ids depend on the fitted GEOMETRY only,
    # never on the training row order or the spelling of row ids.
    reference_rank = _coordinate_ranks(reference_centroids)
    mapping: dict[int, dict[int, int]] = {
        reference.fold_index: {
            local: int(reference_rank[local]) for local in range(len(reference_centroids))
        }
    }
    for fit in ordered[1:]:
        if tuple(fit.preprocessing.features) != names:
            raise ValueError(
                "every fold must be fitted on the same input features to align"
            )
        centroids = feature_space_centroids(fit)
        cost = np.linalg.norm(
            centroids[:, None, :] - reference_centroids[None, :, :], axis=2
        )
        columns = hungarian_lexicographic(cost)
        mapping[fit.fold_index] = {
            int(local): int(reference_rank[matched])
            for local, matched in enumerate(columns)
        }
    semantic = {
        int(reference_rank[local]): _descriptors(reference_centroids[local], names)
        for local in range(len(reference_centroids))
    }
    return RegimeAlignmentMap(
        reference_fold_index=reference.fold_index,
        alignment_feature_names=names,
        mapping=mapping,
        semantic_descriptors=semantic,
    )
