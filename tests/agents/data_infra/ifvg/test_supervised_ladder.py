"""Supervised-ladder acceptance tests (ML plan §2/§9; acceptance 7B.22-1/2/15).

Fixture 1 drives the complete chain end-to-end: the frozen 40/5/5/2 fold
protocol, the prevalence reference, the logistic rung, and the CatBoost
rung — all on identical rows/folds, with parity asserted before any delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    GAM_PROTOCOL_ID,
    ModelProtocolUnavailableError,
    ProhibitedSelectionError,
    assert_single_frozen_selection,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import (
    DEFAULT_LADDER_PROTOCOLS,
    paired_cell_delta_report,
    run_supervised_ladder,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    class_balanced_supervised_fixture,
)


@pytest.fixture(scope="module")
def fixture():
    return class_balanced_supervised_fixture(n=400)


@pytest.fixture(scope="module")
def folds(fixture):
    return build_context_folds(
        fixture.labeled_candidates,
        authorized_trading_days=fixture.trading_days,
    )


@pytest.fixture(scope="module")
def ladder(fixture, folds):
    return run_supervised_ladder(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        tier=fixture.tier,
    )


def test_fixture_is_exactly_class_balanced(fixture):
    targets = fixture.labeled_candidates["binary_target"]
    assert int(targets.sum()) == len(targets) // 2


def test_folds_exist_and_some_are_valid(folds):
    assert folds.folds, "the 55-day fixture must produce walk-forward folds"
    assert any(fold.valid for fold in folds.folds)


def test_ladder_runs_all_three_default_rungs(ladder):
    assert tuple(r.protocol_id for r in ladder.rungs) == DEFAULT_LADDER_PROTOCOLS


def test_ladder_parity_identical_rows_across_rungs(ladder):
    """7B.22-1: identical oos_row_id sets and identical row keys per rung."""

    assert ladder.parity["identical_rows"] is True
    assert ladder.parity["oos_row_count"] > 0
    reference = ladder.rung("reference_prevalence_v1").predictions
    for rung in ladder.rungs:
        frame = rung.predictions
        assert set(frame["oos_row_id"]) == set(reference["oos_row_id"])
        merged = reference[["oos_row_id", "candidate_id", "target", "training_prevalence"]].merge(
            frame[["oos_row_id", "candidate_id", "target", "training_prevalence"]],
            on="oos_row_id",
            validate="one_to_one",
        )
        assert (merged["candidate_id_x"] == merged["candidate_id_y"]).all()
        assert (merged["target_x"] == merged["target_y"]).all()
        assert np.allclose(merged["training_prevalence_x"], merged["training_prevalence_y"])


def test_prevalence_rung_probability_is_the_fold_training_prevalence(ladder):
    frame = ladder.rung("reference_prevalence_v1").predictions
    assert np.allclose(frame["probability"], frame["training_prevalence"])


def test_oos_row_id_is_model_independent_and_view_scoped(ladder, fixture):
    """The row id hashes {view_id, fold_index, candidate_id} only, so rungs
    pair exactly and a different view can never alias a row."""

    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        canonical_contract_sha256,
    )

    frame = ladder.rung("ifvg_context_logistic_l2_v1").predictions
    sample = frame.iloc[0]
    expected = canonical_contract_sha256(
        {
            "view_id": fixture.view.view_id,
            "fold_index": int(sample["fold_index"]),
            "candidate_id": str(sample["candidate_id"]),
        }
    )
    assert sample["oos_row_id"] == expected


def test_informative_rungs_beat_the_prevalence_reference(ladder):
    """The fixture's informative features are genuinely learnable: both
    fitted rungs must show positive Brier skill vs the fold-local
    prevalence reference (this is a fixture-health check, not a research
    claim)."""

    for protocol_id in ("ifvg_context_logistic_l2_v1", "ifvg_context_catboost_binary_v1"):
        report = ladder.rung(protocol_id).prediction_report
        assert report["status"] == "complete"
        assert report["brier_skill_score"] is not None
        assert report["brier_skill_score"] > 0.0


def test_ladder_reports_carry_reference_brier_from_prevalence_column(ladder):
    report = ladder.rung("ifvg_context_logistic_l2_v1").prediction_report
    assert report["reference_brier_score"] > 0.0


def test_paired_deltas_present_and_bootstrap_shaped(ladder):
    expected_keys = {
        "reference_prevalence_v1__vs__ifvg_context_logistic_l2_v1",
        "reference_prevalence_v1__vs__ifvg_context_catboost_binary_v1",
        "ifvg_context_logistic_l2_v1__vs__ifvg_context_catboost_binary_v1",
    }
    assert set(ladder.paired_deltas) == expected_keys
    for delta in ladder.paired_deltas.values():
        assert delta["repetitions"] == 10_000
        assert delta["seed"] == 7
        assert delta["available"] is True


def test_ladder_is_deterministic_across_runs(fixture, folds, ladder):
    again = run_supervised_ladder(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        tier=fixture.tier,
    )
    for first, second in zip(ladder.rungs, again.rungs, strict=True):
        pd.testing.assert_frame_equal(first.predictions, second.predictions)
        assert first.resolved_protocol_hash == second.resolved_protocol_hash
    assert ladder.ladder_id == again.ladder_id


def test_fold_local_preprocessing_is_structural(fixture, folds):
    """7B.22-2: the fit API receives fold definitions and slices internally —
    a pooled frame cannot be passed, and every preprocessing statistic is
    fitted per fold (proved by refitting fold 0 alone and matching the
    ladder's fold-0 predictions exactly)."""

    from alpha_lab.agents.data_infra.ifvg.context_feature_view import features_for_tier
    from alpha_lab.agents.data_infra.ifvg.ml.logistic_model import (
        run_logistic_fold_models,
    )

    first_valid = next(fold for fold in folds.folds if fold.valid)

    class _OneFold:
        folds = (first_valid,)

    solo = run_logistic_fold_models(
        fixture.view,
        fixture.labeled_candidates,
        _OneFold(),
        features=features_for_tier(fixture.tier),
    )
    full = run_logistic_fold_models(
        fixture.view,
        fixture.labeled_candidates,
        folds,
        features=features_for_tier(fixture.tier),
    )
    fold_rows = full.predictions[
        full.predictions["fold_index"] == first_valid.fold_index
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(solo.predictions, fold_rows)


def test_unknown_protocol_id_is_refused(fixture, folds):
    with pytest.raises(ValueError, match="unknown model protocol id"):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            protocols=("reference_prevalence_v1", "made_up_protocol_v1"),
        )


def test_planned_gam_protocol_refuses_fail_closed(fixture, folds):
    with pytest.raises(ModelProtocolUnavailableError, match="planned"):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            protocols=("reference_prevalence_v1", GAM_PROTOCOL_ID),
        )


def test_missing_prevalence_reference_is_refused(fixture, folds):
    with pytest.raises(ValueError, match="prevalence reference rung is mandatory"):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            protocols=("ifvg_context_logistic_l2_v1",),
        )


def test_planned_calibrator_is_refused(fixture, folds):
    from alpha_lab.agents.data_infra.ifvg.ml.calibration_policies import (
        CalibrationPolicyUnavailableError,
    )

    with pytest.raises(CalibrationPolicyUnavailableError):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            calibration_policy_id="platt_sigmoid_train_fold_v1",
        )


def test_prohibited_selection_enumeration_refused():
    """7B.22-15: >1 threshold/calibrator/cluster-count/feature-subset value
    outside a ratified charter raises ProhibitedSelectionError."""

    with pytest.raises(ProhibitedSelectionError):
        assert_single_frozen_selection("threshold", (0.5, 0.6))
    with pytest.raises(ProhibitedSelectionError):
        assert_single_frozen_selection("calibrator", ("a_v1", "b_v1"))
    # single values and ratified enumerations are lawful
    assert_single_frozen_selection("threshold", (0.5,))
    assert_single_frozen_selection(
        "cluster_count", (3, 4), owner_ratification_ref="owner_ref_1"
    )


def test_paired_cell_delta_requires_identical_oos_ids():
    left = pd.DataFrame(
        {
            "oos_row_id": ["a", "b"],
            "trading_day": ["2026-01-05", "2026-01-06"],
            "value": [1.0, 2.0],
        }
    )
    right_mismatched = pd.DataFrame(
        {
            "oos_row_id": ["a", "c"],
            "trading_day": ["2026-01-05", "2026-01-06"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="identical OOS row IDs"):
        paired_cell_delta_report(left, right_mismatched, value_column="value")


def test_paired_cell_delta_refuses_trading_day_disagreement():
    left = pd.DataFrame(
        {
            "oos_row_id": ["a", "b"],
            "trading_day": ["2026-01-05", "2026-01-06"],
            "value": [1.0, 2.0],
        }
    )
    right = pd.DataFrame(
        {
            "oos_row_id": ["a", "b"],
            "trading_day": ["2026-01-05", "2026-01-07"],
            "value": [1.5, 2.5],
        }
    )
    with pytest.raises(ValueError, match="disagree on trading day"):
        paired_cell_delta_report(left, right, value_column="value")
