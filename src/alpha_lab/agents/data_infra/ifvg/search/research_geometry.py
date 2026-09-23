"""Fixed non-MBP geometry ablation through durable registered model ladders."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from ..ml.model_protocols import PREVALENCE_PROTOCOL_ID
from ..ml.supervised_ladder import DEFAULT_BUNDLE_LADDER_PROTOCOLS, paired_cell_delta_report

GEOMETRY_BUNDLES = ("B0_GEOMETRY_CORE_ATR14_V1", "B0_CORE")


def _paired_brier(left, right):
    if left.empty:
        return {"available": False, "reason": "no_oos_predictions", "estimate": None}
    frames = []
    for source in (left, right):
        frame = source.copy()
        frame["brier_loss"] = (frame["target"] - frame["probability"]) ** 2
        frames.append(frame)
    return paired_cell_delta_report(*frames, value_column="brier_loss")


def build_geometry_comparison_report(baseline, challenger):
    """Verify full paired rows and locked numerical protocols before reporting deltas."""
    from ..ml.controlled_feature_study import assert_cross_arm_identity  # noqa: PLC0415

    assert_cross_arm_identity(baseline, challenger)
    checked = (
        "comparison_row_id", "candidate_id", "setup_id", "trading_day", "fold_index",
        "target", "training_prevalence", "gross_r", "net_r",
    )
    reference = baseline.rung(PREVALENCE_PROTOCOL_ID).predictions
    per_protocol = {}
    for protocol in DEFAULT_BUNDLE_LADDER_PROTOCOLS:
        left, right = baseline.rung(protocol), challenger.rung(protocol)
        if not left.predictions.empty:
            pd.testing.assert_frame_equal(
                left.predictions[list(checked)].sort_values("comparison_row_id").reset_index(
                    drop=True
                ),
                right.predictions[list(checked)].sort_values("comparison_row_id").reset_index(
                    drop=True
                ),
                check_exact=True,
                obj="geometry comparison candidate/label/fold/cost identity",
            )
        if protocol != PREVALENCE_PROTOCOL_ID:
            for field in ("parameters", "preprocessing_policy", "package_versions"):
                if getattr(left.resolved_protocol, field) != getattr(
                    right.resolved_protocol, field
                ):
                    raise ValueError(f"geometry arms differ in locked model {field}")
        folds = []
        for original in right.fold_reports:
            index = original["fold_index"]
            selected = right.predictions
            if not selected.empty:
                selected = selected.loc[selected["fold_index"] == index]
            old = left.predictions
            if not old.empty:
                old = old.loc[old["fold_index"] == index]
            prevalence = reference
            if not prevalence.empty:
                prevalence = prevalence.loc[prevalence["fold_index"] == index]
            folds.append({
                **original,
                "brier_delta_vs_b0": _paired_brier(old, selected),
                "brier_delta_vs_prevalence": _paired_brier(prevalence, selected),
                "test_class_counts": {
                    str(value): int((selected["target"] == value).sum())
                    if not selected.empty else 0 for value in (0, 1)
                },
                "test_distinct_setups": int(selected["setup_id"].nunique())
                if not selected.empty else 0,
                "test_entry_days": int(selected["trading_day"].nunique())
                if not selected.empty else 0,
            })
        per_protocol[protocol] = {
            "b0_prediction_report": left.prediction_report,
            "geometry_prediction_report": right.prediction_report,
            "brier_delta_vs_b0": _paired_brier(left.predictions, right.predictions),
            "brier_delta_vs_prevalence": _paired_brier(reference, right.predictions),
            "folds": folds,
        }
    return {
        "schema_version": "ifvg_fixed_geometry_comparison_v1",
        "baseline_bundle_key": "B0_CORE",
        "challenger_bundle_key": GEOMETRY_BUNDLES[0],
        "baseline_ladder_id": baseline.ladder_id,
        "challenger_ladder_id": challenger.ladder_id,
        "delta_convention": "challenger_minus_reference_brier; negative is improvement",
        "row_identity_key": "comparison_row_id",
        "oos_row_count": len(reference),
        "test_class_counts": {
            str(value): int((reference["target"] == value).sum())
            if not reference.empty else 0 for value in (0, 1)
        },
        "test_distinct_setups": int(reference["setup_id"].nunique()) if len(reference) else 0,
        "test_entry_days": int(reference["trading_day"].nunique()) if len(reference) else 0,
        "valid_oos_folds": int(reference["fold_index"].nunique()) if len(reference) else 0,
        "parity_status": "held" if len(reference) else "not_evaluable",
        "uncertainty": "trading_day_block_bootstrap_10000_seed7_percentile95",
        "research_boundary": "exploratory_research_only_offline",
        "per_protocol": per_protocol,
    }


def run_durable_geometry_stage(context):
    """Checkpoint each arm before fitting; completed arm results survive the other's failure."""
    from ..context_model import categorical_features_for  # noqa: PLC0415
    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415
    from ..ml.research_evidence import (  # noqa: PLC0415
        load_research_run,
        save_research_inputs,
        save_research_run,
    )
    from ..ml.supervised_ladder import run_supervised_ladder  # noqa: PLC0415
    from .pipeline_regime import canonical_json_bytes  # noqa: PLC0415
    from .research_artifacts import checkpoint_research_model_input  # noqa: PLC0415

    spec = context.semantic.payload
    if tuple(spec.feature_bundle_ids) != GEOMETRY_BUNDLES or spec.regime_study is not None:
        raise ValueError("geometry requires the exact fixed B0 versus B0+geometry bundle pair")
    if context.folds is None or context.labeled is None or "schedule" not in context.regime:
        raise ValueError("geometry research requires persisted labels and explicit calendar folds")
    expected_ids = set(context.labeled["candidate_id"].astype(str))
    for key in GEOMETRY_BUNDLES:
        frame = context.bundle_frames[key]
        if frame["candidate_id"].duplicated().any() or (
            set(frame["candidate_id"].astype(str)) != expected_ids
        ):
            raise ValueError("geometry arms must retain the exact complete labeled population")

    runs, request_ids, actions, outputs = {}, {}, {}, []
    # Baseline is deliberately persisted first, permitting reuse after a challenger interruption.
    for key in reversed(GEOMETRY_BUNDLES):
        envelope = context.bundle_views[key]
        features = tuple(envelope.payload.resolved_feature_names)
        kwargs = {
            "label_artifact_id": context.label_artifact_id,
            "fold_schedule_id": context.regime["schedule"].fold_schedule_id,
            "bundle_features": features,
            "bundle_ref": resolve_bundle(key).resolved_feature_bundle_id,
            "bundle_evidence_ref": envelope.bundle_feature_view_id,
            "bundle_categorical_features": categorical_features_for(features),
            "protocols": DEFAULT_BUNDLE_LADDER_PROTOCOLS,
        }
        view = replace(context.view, frame=context.bundle_frames[key])
        request_id = save_research_inputs(
            context.store_root, view, context.labeled, context.folds,
            runner="ladder", run_kwargs=kwargs,
        )
        checkpoint_research_model_input(context, request_id)
        run = load_research_run(context.store_root, request_id)
        reused = run is not None
        if run is None:
            run = run_supervised_ladder(view, context.labeled, context.folds, **kwargs)
            save_research_run(context.store_root, request_id, run)
            run = load_research_run(context.store_root, request_id)
        if run is None:  # pragma: no cover
            raise ValueError("geometry arm result could not be verified and reloaded")
        runs[key], request_ids[key] = run, request_id
        actions[key] = "verified and reloaded" if reused else "fitted, persisted and reloaded"
        if request_id not in context.research_model_run_ids:
            context.research_model_run_ids.append(request_id)
        outputs.extend((request_id, run.ladder_id))

    report = build_geometry_comparison_report(runs["B0_CORE"], runs[GEOMETRY_BUNDLES[0]])
    report["request_ids_by_bundle"] = request_ids
    context.stage_sidecars["geometry_comparison.json"] = canonical_json_bytes(report)
    context.stage_sidecars["research_model_runs.json"] = canonical_json_bytes({
        "request_ids": list(context.research_model_run_ids),
        "request_ids_by_bundle": request_ids,
        "input_store": "research_model_inputs", "run_store": "research_model_runs",
    })
    context.ladder = runs[GEOMETRY_BUNDLES[0]]
    return tuple(outputs), (
        f"fixed geometry comparison: {actions}; {report['oos_row_count']} paired OOS rows"
    )
