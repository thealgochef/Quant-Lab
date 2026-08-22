"""ML verification fixture 1 — class-balanced supervised fixture
(`ML_REGIME_CONTRACT_PLAN.md` §9.1).

Deterministic (`numpy.random.default_rng(7)`): two informative Gaussian
features + two pure-noise features + one informative categorical, EXACT
50/50 classes, and synthetic candidate/setup/trading-day identities shaped
like the real fold-input schema — so the frozen `build_context_folds`
protocol and every ladder rung run end-to-end on it unchanged.

The frame carries the complete M0 registry feature set (the ladder resolves
features through the REAL tier registry, never a fixture-private list);
non-designated features hold benign deterministic values. One informative
feature carries ~5% missing values to exercise the fold-local imputer.
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
from alpha_lab.agents.data_infra.ifvg.context_model import categorical_features_for

INFORMATIVE_FEATURES = ("distance_to_htf_ticks", "opposing_size_ticks")
NOISE_FEATURES = ("entry_fvg_size_ticks", "risk_ticks")
INFORMATIVE_CATEGORICAL = "direction"
MISSING_FEATURE = "distance_to_htf_ticks"

_CATEGORICAL_FILL = {
    "direction": ("long", "short"),
    "entry_family": ("fvg_retest", "inversion_retest"),
    "entry_session": ("ny", "asia", "london"),
    "in_engine_session": ("true", "false"),
    "in_doc_session": ("true", "false"),
}


@dataclass(frozen=True, slots=True)
class SupervisedFixture:
    view: CandidateFeatureView
    labeled_candidates: pd.DataFrame
    trading_days: tuple[str, ...]
    tier: ContextFeatureTier


def _trading_days(count: int) -> tuple[str, ...]:
    days = pd.bdate_range("2026-01-05", periods=count)
    return tuple(day.strftime("%Y-%m-%d") for day in days)


def class_balanced_supervised_fixture(n: int = 400) -> SupervisedFixture:
    if n % 2:
        raise ValueError("the fixture is exactly class-balanced; n must be even")
    rng = np.random.default_rng(7)
    day_count = 55  # 40 train + 3 stepped 5-day test windows under 40/5/5/2
    days = _trading_days(day_count)
    day_assignment = np.sort(rng.integers(0, day_count, size=n))

    categorical_signal = rng.integers(0, 2, size=n)  # 0 = long, 1 = short
    informative_a = rng.normal(0.0, 1.0, size=n)
    informative_b = rng.normal(0.0, 1.0, size=n)
    score = (
        1.4 * informative_a
        - 1.1 * informative_b
        + 0.6 * (categorical_signal * 2 - 1)
        + rng.normal(0.0, 0.35, size=n)
    )
    order = np.argsort(score)
    target = np.zeros(n, dtype=int)
    target[order[n // 2 :]] = 1  # exact 50/50 by construction

    frame_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index in range(n):
        day = days[int(day_assignment[index])]
        candidate_id = f"cand_{index:04d}"
        setup_id = f"setup_{index // 2:04d}"
        entry_ts = f"{day}T14:{index % 60:02d}:00Z"
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "setup_id": setup_id,
            "trading_day": day,
            "entry_ts_utc": entry_ts,
            "feature_as_of_ts": entry_ts,
            "context_as_of_ts": entry_ts,
            "context_capture_id": "synthetic_capture_v1",
            "context_state_id": f"state_{index:04d}",
            "geometry_evidence_id": f"geom_{index:04d}",
            "geometry_evidence_cursor": index,
            "is_warmup": False,
        }
        for feature in M0_FEATURES:
            if feature == INFORMATIVE_FEATURES[0]:
                value: object = float(10.0 + 4.0 * informative_a[index])
                if index % 20 == 3:  # ~5% missing → fold-local imputer path
                    value = np.nan
                row[feature] = value
            elif feature == INFORMATIVE_FEATURES[1]:
                row[feature] = float(6.0 + 3.0 * informative_b[index])
            elif feature == NOISE_FEATURES[0]:
                row[feature] = float(rng.normal(8.0, 2.0))
            elif feature == NOISE_FEATURES[1]:
                row[feature] = float(rng.normal(12.0, 3.0))
            elif feature == INFORMATIVE_CATEGORICAL:
                row[feature] = "short" if categorical_signal[index] else "long"
            elif feature in _CATEGORICAL_FILL:
                choices = _CATEGORICAL_FILL[feature]
                row[feature] = choices[index % len(choices)]
            else:
                row[feature] = float(5.0 + (index % 7) * 0.5)
        frame_rows.append(row)
        gross_r = 1.0 if target[index] else -1.0
        label_rows.append(
            {
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "trading_day": day,
                "entry_ts_utc": entry_ts,
                "resolution_ts_utc": f"{day}T15:{index % 60:02d}:00Z",
                "entry_available": True,
                "resolution_available": True,
                "binary_target": int(target[index]),
                "gross_r": gross_r,
                "net_r": gross_r,
            }
        )

    frame = pd.DataFrame(frame_rows)
    labels = pd.DataFrame(label_rows)
    tier = ContextFeatureTier.M0
    tier_features = {tier: M0_FEATURES}
    feature_registry_hash = canonical_contract_sha256(
        {tier.value: list(M0_FEATURES)}
    )
    view = CandidateFeatureView(
        view_id=canonical_contract_sha256(
            {
                "fixture": "class_balanced_supervised_fixture",
                "n": n,
                "feature_registry_hash": feature_registry_hash,
            }
        ),
        artifact_pair_hash=canonical_contract_sha256({"fixture_artifact_pair": n}),
        feature_registry_hash=feature_registry_hash,
        frame=frame,
        tier_features=tier_features,
        m3_status="synthetic_fixture",
    )
    assert categorical_features_for(M0_FEATURES)  # the registry sees categoricals
    return SupervisedFixture(
        view=view,
        labeled_candidates=labels,
        trading_days=days,
        tier=tier,
    )
