"""ML verification fixture 4 (KMeans arm) — OOS assignment fixture
(`ML_REGIME_CONTRACT_PLAN.md` §9.4; the GMM/Nyström arms ship with the
post-V1 regime-expansion release).

Deterministic train/test blob split with INJECTED all-missing rows: proves
deterministic OOS transform+assign for ``kmeans_v1`` and typed-null
preservation (rows survive as ``source_feature_missing``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    KnownClusterFixture,
    known_cluster_fixture,
)


@dataclass(frozen=True, slots=True)
class OosAssignmentFixture:
    base: KnownClusterFixture
    missing_candidate_ids: tuple[str, ...]


def oos_assignment_fixture(k: int = 3, n: int = 600) -> OosAssignmentFixture:
    base = known_cluster_fixture(k=k, n=n)
    frame = base.view.frame.copy()
    # inject all-missing regime inputs on a deterministic subset — the rows
    # must SURVIVE as typed nulls, never vanish (7B.22-10)
    missing_ids = tuple(
        sorted(frame["candidate_id"].astype(str))[:: max(1, n // 12)][:10]
    )
    mask = frame["candidate_id"].astype(str).isin(missing_ids)
    for feature in REGIME_INPUT_FEATURES:
        frame.loc[mask, feature] = np.nan
    view = type(base.view)(
        view_id=base.view.view_id,
        artifact_pair_hash=base.view.artifact_pair_hash,
        feature_registry_hash=base.view.feature_registry_hash,
        frame=frame,
        tier_features=base.view.tier_features,
        m3_status=base.view.m3_status,
    )
    patched = KnownClusterFixture(
        view=view,
        labeled_candidates=base.labeled_candidates,
        trading_days=base.trading_days,
        true_memberships=base.true_memberships,
        regime_input_features=base.regime_input_features,
    )
    return OosAssignmentFixture(base=patched, missing_candidate_ids=missing_ids)


__all__ = ["OosAssignmentFixture", "oos_assignment_fixture"]
