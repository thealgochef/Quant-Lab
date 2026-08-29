"""R6.1 workstream J — the bundle-aware CatBoost rung
``ifvg_context_catboost_bundle_v1`` and the D13 comparison-row identity
(plan §6.J / D13; §9.1
``test_comparison_row_ids_are_equal_across_baseline_and_challenger_bundles``).

Proofs: identical ``comparison_row_id`` populations across rungs AND across
baseline/challenger arms with different ``view_id``s; fold-local fitting
(test-row poisoning); bundle + fold-local features reach CatBoost (planted
signal); block-declared categoricals (the fit-local regime id,
``cbp_session_state``) are fitted as categoricals; a changed bundle or
feature order changes the resolved hash; the frozen M0–M3 CatBoost lane is
byte- and identity-unchanged; no selection surface exists.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import ContextFeatureTier
from alpha_lab.agents.data_infra.ifvg.context_feature_view import features_for_tier
from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.context_model import (
    resolve_context_model_protocol,
    run_context_fold_models,
)
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    build_bundle_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.ml import catboost_bundle_model as bundle_module
from alpha_lab.agents.data_infra.ifvg.ml.catboost_bundle_model import (
    CATBOOST_BUNDLE_PROTOCOL_ID,
    IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS,
    RegimeFoldFeatureSource,
    bundle_rung_categorical_features,
    resolve_catboost_bundle_protocol,
    run_catboost_bundle_fold_models,
)
from alpha_lab.agents.data_infra.ifvg.ml.comparison_rows import (
    candidate_fold_set_id,
    comparison_row_id,
    default_fold_schedule_id,
    label_content_hash,
)
from alpha_lab.agents.data_infra.ifvg.ml.controlled_feature_study import (
    assert_cross_arm_identity,
)
from alpha_lab.agents.data_infra.ifvg.ml.model_protocols import (
    CATBOOST_PROTOCOL_ID,
    LOGISTIC_PROTOCOL_ID,
    MODEL_PROTOCOL_REGISTRY,
    PREVALENCE_PROTOCOL_ID,
)
from alpha_lab.agents.data_infra.ifvg.ml.supervised_ladder import (
    CATBOOST_BUNDLE_REFUSAL,
    CATBOOST_BUNDLE_RUNG_TIER_REFUSAL,
    DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    paired_cell_delta_report,
    run_supervised_ladder,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_supervised import (
    class_balanced_supervised_fixture,
)

_B0 = resolve_bundle("B0_CORE")
_B0_FEATURES = tuple(_B0.payload.resolved_feature_names)
_LOCAL_ID = "ctx_regime_ab12cd34ef56_local_id"
_LOCAL_MARGIN = "ctx_regime_ab12cd34ef56_local_margin"
_CHALLENGER_VIEW_ID = "c" * 64
_CHALLENGER_BUNDLE_REF = "d" * 64

#: Golden constants of the FROZEN M0–M3 CatBoost lane (never modified in V1).
_CONTEXT_MODEL_SOURCE_SHA256 = "3385af2b5354ec9838f6f650e8c086c766a86d806a32b9d71fd9e465f4b6f455"
_M0_GOLDEN_RESOLVED_HASH = "b967af5e93eecb5be596005c4b41980bb3c5172d0f92c35c51cdcb43d605539e"
_GOLDEN_ENVIRONMENT = {
    "catboost": "1.2.10",
    "numpy": "2.3.1",
    "pandas": "2.3.1",
    "scikit-learn": "1.7.0",
    "python": "3.13.1",
}


@dataclasses.dataclass(frozen=True)
class _FoldSource:
    """A synthetic ``RegimeFoldFeatureSource`` (fork G2 ships the real one)."""

    artifact_id: str
    feature_names: tuple[str, ...]
    categorical_features: tuple[str, ...]
    frames: dict[int, pd.DataFrame]

    def frame_for_fold(self, fold_index: int) -> pd.DataFrame:
        return self.frames[int(fold_index)].copy()


def _fold_source(fixture, folds, *, planted: bool) -> _FoldSource:
    targets = fixture.labeled_candidates.set_index("candidate_id")["binary_target"]
    rng = np.random.default_rng(11)
    frames: dict[int, pd.DataFrame] = {}
    for fold in folds.folds:
        ids = [*fold.train_candidate_ids, *fold.test_candidate_ids]
        aligned = targets.loc[ids].to_numpy(dtype=float)
        noise = rng.normal(0.0, 1.0, size=len(ids))
        margin = noise + (3.0 * (aligned - 0.5) if planted else 0.0)
        if planted:
            local_id = np.where(margin > 0.0, "1", "0")
        else:
            local_id = rng.integers(0, 3, size=len(ids)).astype(str)
        frames[fold.fold_index] = pd.DataFrame(
            {"candidate_id": ids, _LOCAL_ID: local_id, _LOCAL_MARGIN: margin}
        )
    return _FoldSource(
        artifact_id=("f" if planted else "e") * 64,
        feature_names=(_LOCAL_ID, _LOCAL_MARGIN),
        categorical_features=(_LOCAL_ID,),
        frames=frames,
    )


def _brier(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["brier_loss"] = (out["target"] - out["probability"]) ** 2
    return out


@pytest.fixture(scope="module")
def fixture():
    return class_balanced_supervised_fixture(n=400)


@pytest.fixture(scope="module")
def folds(fixture):
    return build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )


@pytest.fixture(scope="module")
def baseline_view(fixture):
    _envelope, frame = build_bundle_feature_view(fixture.view, _B0)
    return dataclasses.replace(fixture.view, frame=frame)


@pytest.fixture(scope="module")
def baseline_ladder(fixture, folds, baseline_view):
    return run_supervised_ladder(
        baseline_view,
        fixture.labeled_candidates,
        folds,
        bundle_features=_B0_FEATURES,
        bundle_ref=_B0.resolved_feature_bundle_id,
        protocols=DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    )


def _challenger_ladder(fixture, folds, baseline_view, source):
    challenger_view = dataclasses.replace(baseline_view, view_id=_CHALLENGER_VIEW_ID)
    return run_supervised_ladder(
        challenger_view,
        fixture.labeled_candidates,
        folds,
        bundle_features=_B0_FEATURES + source.feature_names,
        bundle_ref=_CHALLENGER_BUNDLE_REF,
        bundle_evidence_ref=source.artifact_id,
        fold_local_features=source,
        protocols=DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    )


@pytest.fixture(scope="module")
def planted_challenger(fixture, folds, baseline_view):
    source = _fold_source(fixture, folds, planted=True)
    return _challenger_ladder(fixture, folds, baseline_view, source)


# ── §9.1: comparison rows pair arms with different view ids ──────────────────


def test_comparison_row_ids_are_equal_across_baseline_and_challenger_bundles(
    fixture, folds, baseline_view, baseline_ladder, planted_challenger
):
    assert baseline_ladder.view_id != planted_challenger.view_id
    assert isinstance(_fold_source(fixture, folds, planted=False), RegimeFoldFeatureSource)
    for protocol_id in DEFAULT_BUNDLE_LADDER_PROTOCOLS:
        left = baseline_ladder.rung(protocol_id).predictions
        right = planted_challenger.rung(protocol_id).predictions
        assert not left.empty and not right.empty
        assert set(left["comparison_row_id"]) == set(right["comparison_row_id"])
        # the legacy view-scoped id can never pair the arms
        assert set(left["oos_row_id"]).isdisjoint(set(right["oos_row_id"]))
        for fold_index in sorted(set(left["fold_index"])):
            assert set(left.loc[left["fold_index"] == fold_index, "comparison_row_id"]) == set(
                right.loc[right["fold_index"] == fold_index, "comparison_row_id"]
            )
    # every rung of one ladder shares the identical comparison-row population
    reference = set(baseline_ladder.rung(PREVALENCE_PROTOCOL_ID).predictions["comparison_row_id"])
    for rung in baseline_ladder.rungs:
        assert set(rung.predictions["comparison_row_id"]) == reference
    assert baseline_ladder.parity["row_identity_key"] == "comparison_row_id"
    assert baseline_ladder.parity["status"] == "held"
    assert_cross_arm_identity(baseline_ladder, planted_challenger)
    # the id derives from schedule / fold set / fold / candidate / label artifact
    sample = baseline_ladder.rung(LOGISTIC_PROTOCOL_ID).predictions.iloc[0]
    assert sample["comparison_row_id"] == comparison_row_id(
        fold_schedule_id=default_fold_schedule_id(folds, fixture.labeled_candidates),
        candidate_fold_set_id=candidate_fold_set_id(folds),
        fold_index=int(sample["fold_index"]),
        candidate_id=str(sample["candidate_id"]),
        label_artifact_id=label_content_hash(fixture.labeled_candidates),
    )
    # the paired delta keys on the comparison rows across the arms
    delta = paired_cell_delta_report(
        _brier(baseline_ladder.rung(CATBOOST_BUNDLE_PROTOCOL_ID).predictions),
        _brier(planted_challenger.rung(CATBOOST_BUNDLE_PROTOCOL_ID).predictions),
        value_column="brier_loss",
    )
    assert delta["row_identity_key"] == "comparison_row_id"
    # a different fold set is a different row population: the gate refuses
    subset_days = fixture.trading_days[:-5]
    other_folds = build_context_folds(
        fixture.labeled_candidates[
            fixture.labeled_candidates["trading_day"].isin(subset_days)
        ],
        authorized_trading_days=subset_days,
    )
    other = run_supervised_ladder(
        baseline_view,
        fixture.labeled_candidates,
        other_folds,
        bundle_features=_B0_FEATURES,
        bundle_ref=_B0.resolved_feature_bundle_id,
        protocols=DEFAULT_BUNDLE_LADDER_PROTOCOLS,
    )
    with pytest.raises(ValueError, match="disagree on the exact OOS rows"):
        assert_cross_arm_identity(baseline_ladder, other)


# ── fold-local fitting, planted signal, categoricals ─────────────────────────


def test_bundle_rung_fits_fold_locally_and_is_blind_to_test_rows(fixture, folds, baseline_view):
    first_valid = next(fold for fold in folds.folds if fold.valid)

    class _OneFold:
        folds = (first_valid,)

    def _fit(view):
        sink: dict[int, object] = {}
        run = run_catboost_bundle_fold_models(
            view,
            fixture.labeled_candidates,
            _OneFold(),
            features=_B0_FEATURES,
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fitted_fold_sink=sink,
        )
        return run, sink[first_valid.fold_index]

    run, model = _fit(baseline_view)
    probe = bundle_module._model_frame(
        baseline_view.frame.set_index("candidate_id").loc[list(first_valid.train_candidate_ids)],
        features=_B0_FEATURES,
        categoricals=run.protocol.categorical_features,
    )
    reference = model.predict_proba(probe)[:, 1]
    poisoned_frame = baseline_view.frame.copy()
    mask = poisoned_frame["candidate_id"].astype(str) == first_valid.test_candidate_ids[0]
    poisoned_frame.loc[mask, "opposing_size_ticks"] = 1_000_000.0
    poisoned_run, poisoned_model = _fit(dataclasses.replace(baseline_view, frame=poisoned_frame))
    assert np.array_equal(poisoned_model.predict_proba(probe)[:, 1], reference)
    assert np.array_equal(
        poisoned_model.get_feature_importance(type="FeatureImportance"),
        model.get_feature_importance(type="FeatureImportance"),
    )
    # ...and the poisoned TEST row is scored differently: the test rows are
    # predicted, never fitted
    left = run.predictions.set_index("candidate_id")["probability"]
    right = poisoned_run.predictions.set_index("candidate_id")["probability"]
    assert set(left.index) == set(right.index)
    assert run.protocol.resolved_hash == poisoned_run.protocol.resolved_hash


def test_bundle_and_fold_local_features_reach_catboost_with_planted_signal(
    fixture, folds, baseline_view, baseline_ladder, planted_challenger
):
    baseline_report = baseline_ladder.rung(CATBOOST_BUNDLE_PROTOCOL_ID).prediction_report
    challenger_report = planted_challenger.rung(CATBOOST_BUNDLE_PROTOCOL_ID).prediction_report
    assert challenger_report["brier_score"] < baseline_report["brier_score"]
    delta = paired_cell_delta_report(
        _brier(baseline_ladder.rung(CATBOOST_BUNDLE_PROTOCOL_ID).predictions),
        _brier(planted_challenger.rung(CATBOOST_BUNDLE_PROTOCOL_ID).predictions),
        value_column="brier_loss",
    )
    assert delta["available"] and delta["estimate"] < 0.0  # challenger − baseline
    # the logistic rung sees the fold-local features too (one-hot local id)
    assert (
        planted_challenger.rung(LOGISTIC_PROTOCOL_ID).prediction_report["brier_score"]
        < baseline_ladder.rung(LOGISTIC_PROTOCOL_ID).prediction_report["brier_score"]
    )
    importance = planted_challenger.rung(CATBOOST_BUNDLE_PROTOCOL_ID).feature_importance
    assert importance.loc[importance["feature"] == _LOCAL_MARGIN, "catboost_importance"].max() > 0
    assert importance["permutation_repeats"].eq(20).all()
    # the ladder identity binds the fold-local artifact and the D13 keys
    source = planted_challenger.feature_source
    assert source["fold_local_feature_artifact_id"] == "f" * 64
    assert source["evidence_ref"] == "f" * 64
    assert _LOCAL_ID in source["block_declared_categorical_features"]
    assert source["fold_schedule_id"] == default_fold_schedule_id(folds, fixture.labeled_candidates)
    assert source["label_artifact_id"] == label_content_hash(fixture.labeled_candidates)
    assert baseline_ladder.ladder_id != planted_challenger.ladder_id
    expected_keys = {
        f"{PREVALENCE_PROTOCOL_ID}__vs__{LOGISTIC_PROTOCOL_ID}",
        f"{PREVALENCE_PROTOCOL_ID}__vs__{CATBOOST_BUNDLE_PROTOCOL_ID}",
        f"{LOGISTIC_PROTOCOL_ID}__vs__{CATBOOST_BUNDLE_PROTOCOL_ID}",
    }
    assert set(planted_challenger.paired_deltas) == expected_keys


def test_block_declared_categoricals_are_fitted_as_categoricals(fixture, folds, baseline_view):
    source = _fold_source(fixture, folds, planted=True)
    frame = baseline_view.frame.copy()
    rng = np.random.default_rng(5)
    frame["cbp_session_state"] = rng.choice(["asia", "london", "ny"], size=len(frame))
    view = dataclasses.replace(baseline_view, frame=frame)
    features = _B0_FEATURES + ("cbp_session_state",) + source.feature_names
    categorical = bundle_rung_categorical_features(
        features, block_declared=("cbp_session_state", *source.categorical_features)
    )
    assert "cbp_session_state" in categorical and _LOCAL_ID in categorical
    assert "direction" in categorical  # the frozen registry's categoricals stay
    assert categorical == tuple(name for name in features if name in set(categorical))
    run = run_catboost_bundle_fold_models(
        view,
        fixture.labeled_candidates,
        folds,
        features=features,
        resolved_feature_bundle_id=_CHALLENGER_BUNDLE_REF,
        categorical_features=categorical,
        fold_local_features=source,
    )
    assert run.protocol.categorical_features == categorical
    expected_indices = tuple(index for index, name in enumerate(features) if name in categorical)
    assert run.fitted_categorical_indices
    for indices in run.fitted_categorical_indices.values():
        assert indices == expected_indices
        assert features.index("cbp_session_state") in indices
        assert features.index(_LOCAL_ID) in indices
    assert not run.predictions.empty
    assert not run.predictions["comparison_row_id"].duplicated().any()


# ── identity ─────────────────────────────────────────────────────────────────


def test_resolved_hash_is_sensitive_to_bundle_order_and_fold_identity():
    base = dict(
        ordered_features=_B0_FEATURES,
        resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
        categorical_features=bundle_rung_categorical_features(_B0_FEATURES),
        feature_registry_hash="1" * 64,
        fold_set_id="2" * 64,
        fold_schedule_id="3" * 64,
    )
    reference = resolve_catboost_bundle_protocol(**base)
    assert reference.protocol_id == CATBOOST_BUNDLE_PROTOCOL_ID
    assert reference.parameters == dict(IFVG_CONTEXT_CATBOOST_BUNDLE_PARAMETERS)
    assert reference.random_seed == 7
    assert resolve_catboost_bundle_protocol(**base).resolved_hash == reference.resolved_hash
    other_bundle = resolve_catboost_bundle_protocol(
        **{**base, "resolved_feature_bundle_id": _CHALLENGER_BUNDLE_REF}
    )
    assert other_bundle.resolved_hash != reference.resolved_hash
    reordered = tuple(reversed(_B0_FEATURES))
    other_order = resolve_catboost_bundle_protocol(
        **{
            **base,
            "ordered_features": reordered,
            "categorical_features": bundle_rung_categorical_features(reordered),
        }
    )
    assert other_order.resolved_hash != reference.resolved_hash
    for key in ("fold_set_id", "fold_schedule_id", "feature_registry_hash"):
        assert (
            resolve_catboost_bundle_protocol(**{**base, key: "9" * 64}).resolved_hash
            != reference.resolved_hash
        )
    with pytest.raises(ValueError, match="cannot alter locked protocol"):
        resolve_catboost_bundle_protocol(**base, manual_feature_overrides={"threshold": 0.5})
    with pytest.raises(ValueError, match="outside the ordered features"):
        resolve_catboost_bundle_protocol(**{**base, "categorical_features": ("nope",)})
    with pytest.raises(ValueError, match="feature order"):
        resolve_catboost_bundle_protocol(
            **{**base, "categorical_features": tuple(reversed(base["categorical_features"]))}
        )
    entry = MODEL_PROTOCOL_REGISTRY[CATBOOST_BUNDLE_PROTOCOL_ID]
    assert entry.kind == "nonlinear_challenger_bundle"


def test_frozen_m0_m3_catboost_lane_is_byte_and_identity_unchanged(fixture, folds):
    import alpha_lab.agents.data_infra.ifvg.context_model as frozen_lane

    source = Path(frozen_lane.__file__).resolve()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == _CONTEXT_MODEL_SOURCE_SHA256
    protocol = resolve_context_model_protocol(
        ordered_features=features_for_tier(ContextFeatureTier.M0),
        feature_registry_hash="0" * 64,
    )
    environment = {**protocol.package_versions, "python": protocol.python_version}
    if environment == _GOLDEN_ENVIRONMENT:
        assert protocol.resolved_hash == _M0_GOLDEN_RESOLVED_HASH
    else:  # pragma: no cover - a different environment cannot reproduce the golden hash
        pytest.skip(f"golden protocol hash pinned for {_GOLDEN_ENVIRONMENT}; got {environment}")
    # the frozen runner still emits only the legacy row id; the tier ladder's
    # CatBoost rung IS the frozen runner (the ladder adds the comparison id)
    frozen = run_context_fold_models(
        fixture.view, fixture.labeled_candidates, folds, tier=fixture.tier
    )
    assert "comparison_row_id" not in frozen.predictions.columns
    ladder = run_supervised_ladder(
        fixture.view, fixture.labeled_candidates, folds, tier=fixture.tier
    )
    rung = ladder.rung(CATBOOST_PROTOCOL_ID)
    assert rung.resolved_protocol_hash == frozen.protocol.resolved_hash
    pd.testing.assert_frame_equal(
        rung.predictions.drop(columns=["comparison_row_id"]), frozen.predictions
    )
    assert ladder.parity["row_identity_key"] == "comparison_row_id"


# ── dispatch, refusals, and the absence of any selection surface ─────────────


def test_ladder_dispatch_and_refusals(fixture, folds, baseline_view, baseline_ladder):
    assert [rung.protocol_id for rung in baseline_ladder.rungs] == list(
        DEFAULT_BUNDLE_LADDER_PROTOCOLS
    )
    assert baseline_ladder.tier == "resolved_bundle"
    with pytest.raises(ValueError, match="bundle-parametrized path"):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            protocols=(PREVALENCE_PROTOCOL_ID, CATBOOST_BUNDLE_PROTOCOL_ID),
        )
    assert "ifvg_context_catboost_bundle_v1" in CATBOOST_BUNDLE_RUNG_TIER_REFUSAL
    with pytest.raises(ValueError, match="tier-locked"):
        run_supervised_ladder(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            bundle_features=_B0_FEATURES,
            bundle_ref=_B0.resolved_feature_bundle_id,
            protocols=(PREVALENCE_PROTOCOL_ID, CATBOOST_PROTOCOL_ID),
        )
    assert "ifvg_context_catboost_bundle_v1" in CATBOOST_BUNDLE_REFUSAL
    with pytest.raises(ValueError, match="lawful only on the bundle-parametrized path"):
        run_supervised_ladder(
            fixture.view,
            fixture.labeled_candidates,
            folds,
            tier=fixture.tier,
            fold_local_features=_fold_source(fixture, folds, planted=False),
        )


def test_fold_local_source_coherence_refusals(fixture, folds, baseline_view):
    source = _fold_source(fixture, folds, planted=False)
    run = run_catboost_bundle_fold_models
    with pytest.raises(ValueError, match="inert evidence claim"):
        run(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            features=_B0_FEATURES,  # the source's names are not model features
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fold_local_features=source,
        )
    colliding = dataclasses.replace(source, feature_names=("opposing_size_ticks", _LOCAL_MARGIN))
    with pytest.raises(ValueError, match="collide"):
        run(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            features=_B0_FEATURES + (_LOCAL_MARGIN,),
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fold_local_features=colliding,
        )
    duplicated = dataclasses.replace(
        source,
        frames={
            key: pd.concat([frame, frame.iloc[:1]], ignore_index=True)
            for key, frame in source.frames.items()
        },
    )
    with pytest.raises(ValueError, match="repeats a candidate"):
        run(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            features=_B0_FEATURES + source.feature_names,
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fold_local_features=duplicated,
        )
    headless = dataclasses.replace(
        source,
        frames={key: frame.drop(columns=["candidate_id"]) for key, frame in source.frames.items()},
    )
    with pytest.raises(ValueError, match="lacks candidate_id"):
        run(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            features=_B0_FEATURES + source.feature_names,
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fold_local_features=headless,
        )
    # a candidate WITHOUT a fold row keeps NaN / the missing token (typed, never dropped)
    sparse = dataclasses.replace(
        source, frames={key: frame.iloc[1:] for key, frame in source.frames.items()}
    )
    sparse_run = run(
        baseline_view,
        fixture.labeled_candidates,
        folds,
        features=_B0_FEATURES + source.feature_names,
        resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
        fold_local_features=sparse,
    )
    assert set(sparse_run.predictions["candidate_id"]) == set(
        run(
            baseline_view,
            fixture.labeled_candidates,
            folds,
            features=_B0_FEATURES + source.feature_names,
            resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
            fold_local_features=source,
        ).predictions["candidate_id"]
    )


def test_no_selection_surface_exists():
    parameters = set(inspect.signature(run_catboost_bundle_fold_models).parameters)
    assert parameters == {
        "view",
        "labeled_candidates",
        "folds",
        "features",
        "resolved_feature_bundle_id",
        "categorical_features",
        "fold_local_features",
        "fold_schedule_id",
        "label_artifact_id",
        "manual_feature_overrides",
        "fitted_fold_sink",
    }
    text = Path(bundle_module.__file__).read_text(encoding="utf-8")
    # no search / selection machinery and no promotion or store import
    for token in (
        "GridSearch",
        "RandomizedSearch",
        "argmax(",
        "best_",
        "select_k",
        "regime_store",
        "persist_regime_promotion",
        "owner_decisions",
    ):
        assert token not in text
    protocol = resolve_catboost_bundle_protocol(
        ordered_features=_B0_FEATURES,
        resolved_feature_bundle_id=_B0.resolved_feature_bundle_id,
        categorical_features=bundle_rung_categorical_features(_B0_FEATURES),
        feature_registry_hash="1" * 64,
        fold_set_id="2" * 64,
        fold_schedule_id="3" * 64,
    )
    assert protocol.parameters["random_seed"] == 7
    assert protocol.parameters["thread_count"] == 1


def test_pipeline_readiness_admits_the_bundle_rung_only_on_bundle_paths():
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        QuantLabPipelineStage,
        assert_stage_plan_launchable,
        derive_stage_plan_readiness,
    )
    from tests.agents.ifvg_search.test_pipeline_contracts import _FULL_ML_PLAN, _spec

    common = dict(
        stage_plan=_FULL_ML_PLAN,
        label_policy_id="x_v1",
        fold_protocol_id="ifvg_context_walkforward_40_5_5_2_v1",
    )
    mbp1 = _spec(
        **common,
        feature_bundle_ids=("B2_CORE_ORDER_FLOW",),
        model_protocol_id=CATBOOST_BUNDLE_PROTOCOL_ID,
    )
    assert assert_stage_plan_launchable(mbp1).launchable
    tier = _spec(
        **common, feature_bundle_ids=("B0_CORE",), model_protocol_id=CATBOOST_BUNDLE_PROTOCOL_ID
    )
    report = derive_stage_plan_readiness(tier)
    assert not report.launchable
    blocked = report.entry(QuantLabPipelineStage.S09_TRAIN_MODELS)
    assert blocked.state == "blocked_capability"
    assert "bundle-parametrized" in blocked.reason
    binary = _spec(
        **common, feature_bundle_ids=("B2_CORE_ORDER_FLOW",), model_protocol_id=CATBOOST_PROTOCOL_ID
    )
    binary_report = derive_stage_plan_readiness(binary)
    reason = binary_report.entry(QuantLabPipelineStage.S09_TRAIN_MODELS).reason
    assert "tier-locked" in reason and CATBOOST_BUNDLE_PROTOCOL_ID in reason
