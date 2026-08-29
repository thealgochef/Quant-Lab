"""S09b + S09c — the model-bearing sub-steps of a regime study (R6.1 §6.E /
§6.G; D7, D8, D9, D14).

``run_supervised_substeps(context)`` runs only for a request whose
comparison classes include ``feature_only`` / ``cohort_model``: it builds
and persists the fold-local ``RegimeFoldFeatureArtifact`` (S09b — the ONLY
supervised feature source; fit-local ids and distances per fold; the
descriptive OOS artifact is never consulted), activates the
``IFVG_REGIME_CONTEXT_V1`` block as a pure versioned event bound to the
EXACT frozen FEATURE_ELIGIBLE promotion / owner / assessment ids of the
request (D9), and runs the requested studies on identical
``comparison_row_id`` populations (S09c: the controlled regime study for
``feature_only``, the per-regime cohort model for ``cohort_model``). Every
fit of the run happens here; S10 reads the paired deltas and S14 performs
zero fitting.
"""

from __future__ import annotations

from typing import Any

from ..features.feature_bundles import FEATURE_BUNDLE_REGISTRY
from .regime_block_activation import activate_regime_context_block
from .regime_cohort_model import run_regime_cohort_model_study, save_regime_cohort_model_study
from .regime_controlled_study import (
    run_controlled_regime_study,
    save_regime_controlled_study,
)
from .regime_fold_features import (
    PanelFoldFeatureInputs,
    build_regime_fold_features,
    load_regime_fold_feature_source,
    save_regime_fold_features,
)

__all__ = ["run_supervised_substeps", "REGIME_CHALLENGER_BUNDLE_KEY"]

#: The challenger bundle of a candidate-grain controlled regime study: the
#: request's supervised base bundle + the activated regime block (B7).
REGIME_CHALLENGER_BUNDLE_KEY = "B7_CORE_REGIME"


def _challenger_bundle_for(base_bundle_key: str) -> str:
    """The registered bundle whose base is ``base_bundle_key`` and whose only
    extra block is the regime block — B7 for B0; refused otherwise (no
    other regime-bearing bundle is registered in V1)."""

    for key, definition in FEATURE_BUNDLE_REGISTRY.items():
        if definition.base_bundle_key == base_bundle_key and tuple(
            definition.included_block_keys
        ) == ("IFVG_REGIME_CONTEXT_V1",):
            return key
    raise ValueError(
        f"no registered regime-bearing bundle extends {base_bundle_key!r}; the V1 "
        f"controlled regime study runs {REGIME_CHALLENGER_BUNDLE_KEY} over B0_CORE"
    )


def run_supervised_substeps(context) -> tuple[list[str], dict[str, Any], str]:
    spec = context.semantic.payload
    request = spec.regime_study
    regime = context.regime
    store_root = context.store_root
    execution = regime["execution"]
    if context.labeled is None or context.label_artifact_id is None:
        raise ValueError("S09b/S09c require the derived labels (run 07)")
    if context.folds is None:
        raise ValueError("S09b/S09c require the candidate folds (run 08)")
    # F11: the request's value IS the artifact's value (one vocabulary)
    hard_id_encoding = str(request.hard_id_encoding)
    panel_inputs = None
    if request.is_panel:
        panel_inputs = PanelFoldFeatureInputs(
            panel_frame=regime["panel_frame"],
            panel_artifact=regime["panel_envelope"],
            candidate_as_of_stage=request.observation_stage,
        )
    candidate_view = context.bundle_views[request.supervised_bundle_key]
    candidate_frame = context.bundle_frames[request.supervised_bundle_key]
    # ── S09b: the fold-local regime features (fit k → fold k only) ────────
    fold_feature_envelope, fold_feature_frame = build_regime_fold_features(
        protocol=execution.protocol,
        regime_run=execution.run,
        candidate_fold_set=regime["candidate_fold_set"],
        candidate_folds=context.folds,
        regime_fold_set=regime["regime_fold_set"],
        schedule=regime["schedule"],
        candidate_view_frame=candidate_frame,
        candidate_view_id=candidate_view.bundle_feature_view_id,
        hard_id_encoding=hard_id_encoding,
        panel=panel_inputs,
    )
    save_regime_fold_features(store_root, fold_feature_envelope, fold_feature_frame)
    fold_feature_id = fold_feature_envelope.regime_fold_feature_artifact_id
    regime["fold_feature_artifact_id"] = fold_feature_id
    outputs = [fold_feature_id]
    record: dict[str, Any] = {
        "S09b": {
            "regime_fold_feature_artifact_id": fold_feature_id,
            "row_count": int(len(fold_feature_frame)),
            "hard_id_encoding": hard_id_encoding,
            "candidate_fold_set_artifact_id": regime["candidate_fold_set"].fold_set_artifact_id,
            "regime_fold_set_artifact_id": regime["regime_fold_set"].fold_set_artifact_id,
            "fold_schedule_id": regime["schedule"].fold_schedule_id,
        }
    }
    note = f"; regime S09b: fold-local feature artifact {fold_feature_id[:12]}…"
    # ── D9: the status-gated activation bound to the EXACT frozen authority ──
    from ..search.pipeline_regime import regime_run_scope  # noqa: PLC0415

    activation = activate_regime_context_block(
        store_root,
        resolved_regime_protocol_id=execution.protocol.resolved_regime_protocol_id,
        regime_fold_feature_artifact_id=fold_feature_id,
        regime_promotion_decision_id=str(request.regime_promotion_decision_id),
        owner_decision_artifact_id=str(request.owner_decision_artifact_id),
        run_scope=regime_run_scope(context),
    )
    regime["activation"] = activation
    fold_features = load_regime_fold_feature_source(store_root, fold_feature_id)
    # ── S09c: the controlled study and/or the cohort model ────────────────
    study_ids: list[str] = []
    paired_deltas: dict[str, Any] = {}
    classes = set(request.comparison_classes_requested)
    record["S09c"] = {"activated_block_resolution_id": activation.resolved_feature_block_id}
    if "feature_only" in classes:
        study = run_controlled_regime_study(
            context.view,
            context.labeled,
            context.folds,
            activation=activation,
            challenger_bundle_key=_challenger_bundle_for(request.supervised_bundle_key),
            fold_features=fold_features,
            candidate_fold_set=regime["candidate_fold_set"],
            label_artifact_id=context.label_artifact_id,
        )
        save_regime_controlled_study(store_root, study)
        study_id = study.envelope.regime_controlled_study_id
        study_ids.append(study_id)
        outputs.append(study_id)
        record["S09c"]["regime_controlled_study_id"] = study_id
        record["S09c"]["feature_only"] = {
            "baseline_ladder_id": study.baseline.ladder_id,
            "challenger_ladder_id": study.challenger.ladder_id,
            "paired_deltas": study.envelope.payload.model_dump(mode="json").get(
                "paired_deltas", {}
            ),
        }
        paired_deltas["feature_only"] = record["S09c"]["feature_only"]["paired_deltas"]
        regime["controlled_study"] = study
        note += f"; S09c feature_only study {study_id[:12]}…"
    if "cohort_model" in classes:
        cohort = run_regime_cohort_model_study(
            context.view,
            context.labeled,
            context.folds,
            activation=activation,
            bundle_key=request.supervised_bundle_key,
            fold_features=fold_features,
            candidate_fold_set=regime["candidate_fold_set"],
            label_artifact_id=context.label_artifact_id,
        )
        save_regime_cohort_model_study(store_root, cohort)
        cohort_id = cohort.envelope.regime_cohort_model_study_id
        study_ids.append(cohort_id)
        outputs.append(cohort_id)
        record["S09c"]["regime_cohort_model_study_id"] = cohort_id
        record["S09c"]["cohort_model"] = {
            "pooled_ladder_id": cohort.pooled.ladder_id,
            "specialized_strata": len(cohort.specialized),
        }
        regime["cohort_model_study"] = cohort
        note += f"; S09c cohort_model study {cohort_id[:12]}…"
    regime["supervised_study_ids"] = study_ids
    regime["paired_deltas"] = paired_deltas
    return outputs, record, note
