"""R6.1 — the fit-free sample-adequacy preview equals the assessment's
counting (plan §6.C / §9.2)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_sample_adequacy import (
    NON_PREVIEWABLE_GATES,
    preview_sample_adequacy,
    sample_adequacy_floor_facts,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_oos_assignment import (
    oos_assignment_fixture,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id


def _run(fixture):
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0, resolved_input_features=REGIME_INPUT_FEATURES
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    return folds, run


@pytest.mark.parametrize(("n", "expected"), [(600, "pass"), (170, "fail")])
def test_preview_equals_the_assessment_outcome(n, expected):
    fixture = known_cluster_fixture(k=3, n=n)
    folds, run = _run(fixture)
    preview = preview_sample_adequacy(
        fixture.view.frame,
        folds,
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        resolved_cluster_count=3,
        resolved_input_features=REGIME_INPUT_FEATURES,
    )
    coverage = run.assessment.payload.coverage
    assert preview.expected_gate_outcome == expected
    assert preview.observed_minimum_training_rows == (
        coverage.minimum_training_observations_observed
    )
    assert preview.minimum_training_observations == coverage.minimum_training_observations_gate
    assert ("sample_adequacy" in run.assessment.payload.gate_failures) == (expected == "fail")
    by_fold = {fit.fold_index: fit.training_row_count for fit in run.fold_fits}
    for entry in preview.per_fold:
        assert entry.train_rows_with_inputs == by_fold[entry.fold_index]
        assert entry.expected_sample_adequacy == expected
    assert preview.floor_is_flat_per_grain is True
    assert preview.balanced_cluster_rows_lower_bound == 3 * int(
        REGIME_PROPOSED_DEFAULTS["minimum_cluster_rows_per_fold"]["value"]
    )
    assert preview.non_previewable_gates == NON_PREVIEWABLE_GATES


def test_preview_excludes_all_missing_rows_like_the_fit():
    fixture = oos_assignment_fixture(k=3, n=600)
    folds, run = _run(fixture.base)
    preview = preview_sample_adequacy(
        fixture.base.view.frame,
        folds,
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        resolved_cluster_count=3,
        resolved_input_features=REGIME_INPUT_FEATURES,
    )
    by_fold = {fit.fold_index: fit.training_row_count for fit in run.fold_fits}
    assert sum(entry.train_rows_all_missing for entry in preview.per_fold) > 0
    for entry in preview.per_fold:
        assert entry.train_rows_with_inputs == by_fold[entry.fold_index]
        assert entry.train_rows_total == entry.train_rows_with_inputs + entry.train_rows_all_missing


def test_floor_facts_and_refusals():
    facts = sample_adequacy_floor_facts(ObservationGranularity.CONTEXT_BAR_PANEL, 3)
    assert facts["minimum_training_observations"] == 300
    assert facts["floor_is_flat_per_grain"] is True
    assert facts["stamp"] == "proposed_protocol_default"
    fixture = known_cluster_fixture(k=3, n=120)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    with pytest.raises(ValueError, match="lacks regime inputs"):
        preview_sample_adequacy(
            fixture.view.frame.drop(columns=[REGIME_INPUT_FEATURES[0]]),
            folds,
            observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
            resolved_cluster_count=3,
            resolved_input_features=REGIME_INPUT_FEATURES,
        )
    frame = fixture.view.frame.copy()
    frame.loc[:, list(REGIME_INPUT_FEATURES)] = np.nan
    preview = preview_sample_adequacy(
        frame,
        folds,
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        resolved_cluster_count=3,
        resolved_input_features=REGIME_INPUT_FEATURES,
    )
    assert all(entry.train_rows_with_inputs == 0 for entry in preview.per_fold)
    assert preview.expected_gate_outcome in ("fail", "no_valid_folds")
