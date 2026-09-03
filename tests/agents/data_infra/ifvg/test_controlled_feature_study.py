"""R5B controlled Baseline vs Baseline+MBP-1 study suites (deliverable 10).

Identical rows/labels/folds/model protocol across arms — proven on a
synthetic view with VALID walk-forward folds (60 trading days under the
frozen 40/5/5/2 protocol), a real paired Brier delta, deterministic
identity, immutable persistence, and every fail-closed refusal.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    mbp1_feature_names,
)
from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (
    load_controlled_feature_study,
    load_controlled_study_detail,
    run_controlled_mbp1_study,
    save_controlled_feature_study,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    CATBOOST_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    PREVALENCE_PROTOCOL_ID,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import (
    CATBOOST_BUNDLE_REFUSAL,
    run_supervised_ladder,
)
from tests.agents.ifvg_search.mbp1_fixture import feature_artifact_for_frame
from tests.agents.ifvg_search.pipeline_fixture import build_mini_view


def _sixty_weekdays() -> tuple[str, ...]:
    days: list[str] = []
    cursor = date(2026, 1, 5)
    while len(days) < 60:
        if cursor.weekday() < 5:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return tuple(days)


@pytest.fixture(scope="module")
def study_inputs():
    view, labels = build_mini_view(_sixty_weekdays(), per_day=4)
    folds = build_context_folds(
        labels,
        authorized_trading_days=tuple(sorted(set(labels["trading_day"].astype(str)))),
    )
    assert any(fold.valid for fold in folds.folds), "fixture must have valid folds"
    rng = np.random.default_rng(23)
    mbp1_frame = pd.DataFrame({"candidate_id": view.frame["candidate_id"].astype(str)})
    targets = labels.set_index("candidate_id")["binary_target"]
    for index, name in enumerate(mbp1_feature_names()):
        noise = rng.normal(0.0, 1.0, size=len(mbp1_frame))
        if index < 16:
            # a strong planted signal: the challenger arm must measurably
            # beat the baseline arm, proving the joined MBP-1 columns
            # actually reach the fitted model
            aligned = targets.loc[mbp1_frame["candidate_id"]].to_numpy(dtype=float)
            noise = noise + 2.5 * (aligned - 0.5)
        mbp1_frame[name] = noise
    envelope, full_frame = feature_artifact_for_frame(
        mbp1_frame, trading_day="2026-01-05"
    )
    return view, labels, folds, envelope, full_frame


_LABEL_POLICY = "synthetic_fixture_labels_v1"


def _label_artifact_id(labels):
    """R6.1-FIX §3.6: the exact (policy-bearing, every-consumed-column) label
    artifact id a persisting caller passes."""

    from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import label_artifact_content_id

    return label_artifact_content_id(_LABEL_POLICY, labels)


@pytest.fixture(scope="module")
def study(study_inputs):
    view, labels, folds, envelope, full_frame = study_inputs
    return run_controlled_mbp1_study(
        view,
        labels,
        folds,
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
        label_artifact_id=_label_artifact_id(labels),
        label_policy_id=_LABEL_POLICY,
    )


def test_arms_share_identical_rows_labels_folds_and_protocol(study):
    payload = study.envelope.payload
    assert payload.model_protocol_id == LOGISTIC_PROTOCOL_ID
    assert payload.parity_status == "held"
    assert payload.oos_row_count > 0
    assert payload.baseline_bundle_key == "B0_CORE"
    assert payload.challenger_bundle_key == "B2_CORE_ORDER_FLOW"
    assert payload.research_boundary == "research_only_offline"
    baseline_logistic = study.baseline.rung(LOGISTIC_PROTOCOL_ID).predictions
    challenger_logistic = study.challenger.rung(LOGISTIC_PROTOCOL_ID).predictions
    assert set(baseline_logistic["oos_row_id"]) == set(challenger_logistic["oos_row_id"])
    # the prevalence REFERENCE is numerically identical across arms
    left = study.baseline.rung(PREVALENCE_PROTOCOL_ID).predictions
    right = study.challenger.rung(PREVALENCE_PROTOCOL_ID).predictions
    merged = left[["oos_row_id", "probability"]].merge(
        right[["oos_row_id", "probability"]], on="oos_row_id", validate="one_to_one"
    )
    assert (merged["probability_x"] == merged["probability_y"]).all()


def test_paired_delta_is_computed_with_the_block_bootstrap(study):
    delta = dict(study.envelope.payload.paired_brier_delta)
    assert delta["available"] is True
    assert delta["seed"] == 7
    assert delta["repetitions"] == 10_000
    assert delta["lower"] <= delta["estimate"] <= delta["upper"]
    # the crafted signal rides only the MBP-1 columns: the challenger arm
    # must beat the baseline arm's Brier on these synthetic rows
    assert float(delta["estimate"]) < 0


def test_arm_ladders_are_bundle_parametrized_with_distinct_identities(study):
    assert study.baseline.feature_source["kind"] == "resolved_bundle"
    assert study.challenger.feature_source["kind"] == "resolved_bundle"
    assert (
        study.baseline.feature_source["resolved_feature_bundle_id"]
        == resolve_bundle("B0_CORE").resolved_feature_bundle_id
    )
    assert study.baseline.ladder_id != study.challenger.ladder_id
    assert study.envelope.payload.baseline_ladder_id == study.baseline.ladder_id
    assert study.envelope.payload.challenger_ladder_id == study.challenger.ladder_id


def test_study_identity_is_deterministic(study, study_inputs):
    view, labels, folds, envelope, full_frame = study_inputs
    again = run_controlled_mbp1_study(
        view,
        labels,
        folds,
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        mbp1_features=full_frame,
        mbp1_feature_artifact=envelope,
        label_artifact_id=_label_artifact_id(labels),
        label_policy_id=_LABEL_POLICY,
    )
    assert (
        again.envelope.controlled_feature_study_id
        == study.envelope.controlled_feature_study_id
    )
    assert again.envelope.detail_sha256 == study.envelope.detail_sha256
    # a different pinned evidence artifact (distinct recipe over the same
    # table) is a different study AND a different challenger ladder id —
    # review F2: the evidence rides the ladder identity, so one surfaced
    # ladder id can never cover two evidence artifacts
    metrics_only = full_frame[["candidate_id", *mbp1_feature_names()]]
    other_envelope, other_frame = feature_artifact_for_frame(
        metrics_only, trading_day="2026-01-05", anchor_salt="b"
    )
    other = run_controlled_mbp1_study(
        view,
        labels,
        folds,
        challenger_bundle_key="B2_CORE_ORDER_FLOW",
        mbp1_features=other_frame,
        mbp1_feature_artifact=other_envelope,
        label_artifact_id=_label_artifact_id(labels),
        label_policy_id=_LABEL_POLICY,
    )
    assert (
        other.envelope.controlled_feature_study_id
        != study.envelope.controlled_feature_study_id
    )
    assert (
        other.envelope.payload.challenger_ladder_id
        != study.envelope.payload.challenger_ladder_id
    )
    assert (
        other.envelope.payload.baseline_ladder_id
        == study.envelope.payload.baseline_ladder_id
    )


def test_study_persists_immutably_with_the_detail_sidecar(study, tmp_path):
    root = tmp_path / "store"
    save_controlled_feature_study(root, study)
    reloaded = load_controlled_feature_study(
        root, study.envelope.controlled_feature_study_id
    )
    assert reloaded.model_dump(mode="json") == study.envelope.model_dump(mode="json")
    detail = load_controlled_study_detail(root, reloaded)
    # R6.1 (§6.J): both arms run prevalence + logistic + the bundle-aware CatBoost rung
    assert set(detail["challenger"]["rungs"]) == {
        PREVALENCE_PROTOCOL_ID,
        LOGISTIC_PROTOCOL_ID,
        "ifvg_context_catboost_bundle_v1",
    }
    payload = reloaded.payload
    assert payload.ladder_protocol_ids == (
        PREVALENCE_PROTOCOL_ID,
        LOGISTIC_PROTOCOL_ID,
        "ifvg_context_catboost_bundle_v1",
    )
    assert payload.row_identity_key == "comparison_row_id"
    assert set(payload.paired_brier_deltas) == {
        LOGISTIC_PROTOCOL_ID,
        "ifvg_context_catboost_bundle_v1",
    }
    assert dict(payload.paired_brier_deltas[LOGISTIC_PROTOCOL_ID]) == dict(
        payload.paired_brier_delta
    )
    assert set(payload.challenger_rung_summaries) == set(payload.paired_brier_deltas)
    assert payload.fold_schedule_id is not None and payload.label_artifact_id is not None
    save_controlled_feature_study(root, study)  # verified reuse


def test_non_mbp1_challenger_is_refused(study_inputs):
    view, labels, folds, envelope, full_frame = study_inputs
    with pytest.raises(ValueError, match="no MBP-1 order-flow"):
        run_controlled_mbp1_study(
            view,
            labels,
            folds,
            challenger_bundle_key="B1_CORE_STRUCTURE",
            mbp1_features=full_frame,
            mbp1_feature_artifact=envelope,
        )


def test_catboost_refuses_under_bundle_parametrization(study_inputs):
    view, labels, folds, _envelope, _full_frame = study_inputs
    bundle = resolve_bundle("B0_CORE")
    with pytest.raises(ValueError, match="tier-locked"):
        run_supervised_ladder(
            view,
            labels,
            folds,
            bundle_features=tuple(bundle.payload.resolved_feature_names),
            bundle_ref=bundle.resolved_feature_bundle_id,
            protocols=(PREVALENCE_PROTOCOL_ID, LOGISTIC_PROTOCOL_ID, CATBOOST_PROTOCOL_ID),
        )
    assert "frozen M0-M3 lane" in CATBOOST_BUNDLE_REFUSAL


def test_ladder_feature_parametrization_is_exactly_one_of_tier_or_bundle(study_inputs):
    view, labels, folds, _envelope, _full_frame = study_inputs
    from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
        ContextFeatureTier,
    )

    bundle = resolve_bundle("B0_CORE")
    with pytest.raises(ValueError, match="exactly one feature parametrization"):
        run_supervised_ladder(view, labels, folds)
    with pytest.raises(ValueError, match="exactly one feature parametrization"):
        run_supervised_ladder(
            view,
            labels,
            folds,
            tier=ContextFeatureTier.M0,
            bundle_features=tuple(bundle.payload.resolved_feature_names),
            bundle_ref=bundle.resolved_feature_bundle_id,
        )
    with pytest.raises(ValueError, match="bundle_ref"):
        run_supervised_ladder(
            view,
            labels,
            folds,
            bundle_features=tuple(bundle.payload.resolved_feature_names),
        )
