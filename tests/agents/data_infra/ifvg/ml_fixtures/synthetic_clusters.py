"""ML verification fixture 2 — known-cluster fixture
(`ML_REGIME_CONTRACT_PLAN.md` §9.2).

Deterministic (`numpy.random.default_rng(7)`): ``k`` well-separated
Gaussian blobs in two designated numeric features, with KNOWN memberships,
shaped like the real fold-input schema — so `build_context_folds` and the
regime service run end-to-end on it unchanged. Proves KMeans determinism,
occupancy gates, Hungarian alignment stability (permuted fold order →
stable canonical ids), and margins/distances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ContextFeatureTier,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (
    M0_FEATURES,
    CandidateFeatureView,
)

REGIME_INPUT_FEATURES: tuple[str, ...] = (
    "distance_to_htf_ticks",
    "opposing_size_ticks",
)

#: Well-separated blob centers in the two designated features.
CLUSTER_CENTERS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (18.0, 2.0),
    (6.0, 20.0),
    (24.0, 24.0),
    (-14.0, 12.0),
)

_CATEGORICAL_FILL = {
    "direction": ("long", "short"),
    "entry_family": ("fvg_retest", "inversion_retest"),
    "entry_session": ("ny", "asia", "london"),
    "in_engine_session": ("true", "false"),
    "in_doc_session": ("true", "false"),
}


@dataclass(frozen=True, slots=True)
class KnownClusterFixture:
    view: CandidateFeatureView
    labeled_candidates: pd.DataFrame
    trading_days: tuple[str, ...]
    true_memberships: dict[str, int]
    regime_input_features: tuple[str, ...]


def _trading_days(count: int) -> tuple[str, ...]:
    days = pd.bdate_range("2026-01-05", periods=count)
    return tuple(day.strftime("%Y-%m-%d") for day in days)


def known_cluster_fixture(k: int = 3, n: int = 600) -> KnownClusterFixture:
    if not 2 <= k <= len(CLUSTER_CENTERS):
        raise ValueError(f"k must be in [2, {len(CLUSTER_CENTERS)}]")
    rng = np.random.default_rng(7)
    day_count = 55  # 40 train + stepped 5-day test windows under 40/5/5/2
    days = _trading_days(day_count)
    day_assignment = np.sort(rng.integers(0, day_count, size=n))
    memberships = rng.integers(0, k, size=n)
    centers = np.asarray(CLUSTER_CENTERS[:k], dtype=float)
    points = centers[memberships] + rng.normal(0.0, 0.9, size=(n, 2))

    frame_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    true_memberships: dict[str, int] = {}
    for index in range(n):
        day = days[int(day_assignment[index])]
        candidate_id = f"kc_{index:04d}"
        setup_id = f"kcsetup_{index // 2:04d}"
        entry_ts = f"{day}T14:{index % 60:02d}:00Z"
        true_memberships[candidate_id] = int(memberships[index])
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "setup_id": setup_id,
            "trading_day": day,
            "entry_ts_utc": entry_ts,
            "feature_as_of_ts": entry_ts,
            "context_as_of_ts": entry_ts,
            "context_capture_id": "synthetic_capture_v1",
            "context_state_id": f"kcstate_{index:04d}",
            "geometry_evidence_id": f"kcgeom_{index:04d}",
            "geometry_evidence_cursor": index,
            "is_warmup": False,
        }
        for feature in M0_FEATURES:
            if feature == REGIME_INPUT_FEATURES[0]:
                row[feature] = float(points[index, 0])
            elif feature == REGIME_INPUT_FEATURES[1]:
                row[feature] = float(points[index, 1])
            elif feature in _CATEGORICAL_FILL:
                choices = _CATEGORICAL_FILL[feature]
                row[feature] = choices[index % len(choices)]
            else:
                row[feature] = float(5.0 + (index % 7) * 0.5)
        frame_rows.append(row)
        target = index % 2  # outcome-independent of the cluster geometry
        label_rows.append(
            {
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "trading_day": day,
                "entry_ts_utc": entry_ts,
                "resolution_ts_utc": f"{day}T15:{index % 60:02d}:00Z",
                "entry_available": True,
                "resolution_available": True,
                "binary_target": target,
                "gross_r": 1.0 if target else -1.0,
                "net_r": 1.0 if target else -1.0,
            }
        )

    frame = pd.DataFrame(frame_rows)
    tier = ContextFeatureTier.M0
    view = CandidateFeatureView(
        view_id=canonical_contract_sha256(
            {"fixture": "known_cluster_fixture", "k": k, "n": n}
        ),
        artifact_pair_hash=canonical_contract_sha256({"kc_pair": n}),
        feature_registry_hash=canonical_contract_sha256({tier.value: list(M0_FEATURES)}),
        frame=frame,
        tier_features={tier: M0_FEATURES},
        m3_status="synthetic_fixture",
    )
    return KnownClusterFixture(
        view=view,
        labeled_candidates=pd.DataFrame(label_rows),
        trading_days=days,
        true_memberships=true_memberships,
        regime_input_features=REGIME_INPUT_FEATURES,
    )
