"""Scoped populations, durable labels and model-stage reuse for real research."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field

from ..contracts import RecordTable
from ..ml.comparison_rows import label_artifact_content_id
from .executed_trade_table import research_trade_cohort_masks
from .identities import SHA256_PATTERN, EnvelopeBase, FrozenContract
from .store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope


class ResearchCohortPayload(FrozenContract):
    schema_version: Literal[1] = 1
    research_subject_id: str = Field(pattern=SHA256_PATTERN)
    candidate_view_id: str = Field(pattern=SHA256_PATTERN)
    candidate_ids: tuple[str, ...]
    executed_trade_ids: tuple[str, ...]
    excluded_warmup_candidate_ids: tuple[str, ...]
    excluded_window_candidate_ids: tuple[str, ...]
    excluded_warmup_trade_ids: tuple[str, ...] = ()
    excluded_window_trade_ids: tuple[str, ...] = ()
    cutoff_censored_trade_ids: tuple[str, ...] = ()
    excluded_unresolved_trade_ids: tuple[str, ...] = ()
    evaluation_dates: tuple[str, ...]


class ResearchCohortEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_cohort_id"
    research_cohort_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchCohortPayload


class ResearchLabelPayload(FrozenContract):
    schema_version: Literal[1] = 1
    research_subject_id: str = Field(pattern=SHA256_PATTERN)
    candidate_view_id: str = Field(pattern=SHA256_PATTERN)
    label_policy_id: str
    label_artifact_id: str = Field(pattern=SHA256_PATTERN)
    labels_sha256: str = Field(pattern=SHA256_PATTERN)
    row_count: int = Field(ge=0)


class ResearchLabelEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_label_id"
    research_label_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchLabelPayload


def save_research_cohort(root, *, subject, view, raw_tables, scoped_trades):
    raw = raw_tables[RecordTable.ENTRY_CANDIDATE]
    if "is_warmup" not in raw or raw["is_warmup"].isna().any():
        raise ValueError("candidate population requires explicit warmup provenance")
    warmup = raw["is_warmup"].astype(bool)
    outside = ~raw["trading_day"].astype(str).isin(subject.evaluation_dates)
    included = set(view.frame["candidate_id"].astype(str))
    expected = set(raw.loc[~warmup & ~outside, "candidate_id"].astype(str))
    if subject.cohort == "all_candidates" and included != expected:
        raise ValueError("research view differs from the exact post-warmup candidate population")
    trades = raw_tables[RecordTable.EXECUTED_TRADE]
    trade_masks = research_trade_cohort_masks(
        trades, candidate_ids=included, cutoff_ts_utc=subject.cutoff_ts_utc,
    )
    if set(scoped_trades["trade_id"].astype(str)) != set(
        trades.loc[trade_masks["included"], "trade_id"].astype(str)
    ):
        raise ValueError("research trade table differs from the exact scoped population")
    envelope = ResearchCohortEnvelope.from_payload(
        ResearchCohortPayload(
            research_subject_id=subject.subject_id,
            candidate_view_id=view.view_id,
            candidate_ids=tuple(sorted(included)),
            executed_trade_ids=tuple(sorted(scoped_trades["trade_id"].astype(str))),
            excluded_warmup_candidate_ids=tuple(
                sorted(raw.loc[warmup, "candidate_id"].astype(str))
            ),
            excluded_window_candidate_ids=tuple(
                sorted(raw.loc[~warmup & outside, "candidate_id"].astype(str))
            ),
            excluded_warmup_trade_ids=tuple(
                sorted(trades.loc[trade_masks["warmup"], "trade_id"].astype(str))
            ),
            excluded_window_trade_ids=tuple(
                sorted(trades.loc[trade_masks["out_of_cohort"], "trade_id"].astype(str))
            ),
            cutoff_censored_trade_ids=tuple(
                sorted(trades.loc[trade_masks["cutoff_censored"], "trade_id"].astype(str))
            ),
            excluded_unresolved_trade_ids=tuple(
                sorted(trades.loc[trade_masks["unresolved"], "trade_id"].astype(str))
            ),
            evaluation_dates=tuple(subject.evaluation_dates),
        )
    )
    save_or_reuse_envelope(Path(root), "research_cohorts", envelope)
    return envelope


def save_research_labels(root, *, subject, view, labels, label_policy_id):
    if labels["candidate_id"].duplicated().any() or labels["candidate_id"].isna().any():
        raise ValueError("research labels require unique non-null candidates")
    if set(labels["candidate_id"].astype(str)) != set(view.frame["candidate_id"].astype(str)):
        raise ValueError("research label derivation must retain the full candidate cohort")
    if not set(labels["trading_day"].astype(str)) <= set(subject.evaluation_dates):
        raise ValueError("labels contain observations outside the research calendar")
    if (
        "is_warmup" not in labels
        or labels["is_warmup"].isna().any()
        or labels["is_warmup"].astype(bool).any()
    ):
        raise ValueError("warmup labels cannot enter research")
    ordered = labels.sort_values("candidate_id", kind="mergesort").reset_index(drop=True)
    raw = ordered.to_parquet(index=False)
    envelope = ResearchLabelEnvelope.from_payload(
        ResearchLabelPayload(
            research_subject_id=subject.subject_id,
            candidate_view_id=view.view_id,
            label_policy_id=label_policy_id,
            label_artifact_id=label_artifact_content_id(label_policy_id, ordered),
            labels_sha256=hashlib.sha256(raw).hexdigest(),
            row_count=len(ordered),
        )
    )
    save_or_reuse_envelope(
        Path(root), "research_labels", envelope, extra_files={"labels.parquet": raw}
    )
    return envelope


def load_research_labels(root, artifact_id):
    envelope = load_verified_envelope(
        Path(root), "research_labels", artifact_id, ResearchLabelEnvelope
    )
    raw = load_sidecar_bytes(Path(root), "research_labels", artifact_id, "labels.parquet")
    if hashlib.sha256(raw).hexdigest() != envelope.payload.labels_sha256:
        raise ValueError("research label bytes differ from their identity")
    frame = pd.read_parquet(BytesIO(raw))
    if (
        len(frame) != envelope.payload.row_count
        or label_artifact_content_id(envelope.payload.label_policy_id, frame)
        != envelope.payload.label_artifact_id
    ):
        raise ValueError("research label content differs from its declared identity")
    return envelope, frame


def checkpoint_research_model_input(context, request_id):
    """Make verified inputs discoverable even if the subsequent fit fails."""
    from .pipeline import QuantLabPipelineStage, _checkpoint  # noqa: PLC0415

    entry = context.state["stages"][QuantLabPipelineStage.S09_TRAIN_MODELS.value]
    ids = entry.setdefault("research_model_input_ids", [])
    if request_id not in ids:
        ids.append(request_id)
    _checkpoint(context)


def run_durable_supervised_stage(context):
    from ..context_model import categorical_features_for  # noqa: PLC0415
    from ..features.feature_bundles import resolve_bundle  # noqa: PLC0415
    from ..ml.controlled_feature_study import (  # noqa: PLC0415
        run_controlled_mbp1_study,
        save_controlled_feature_study,
    )
    from ..ml.research_evidence import (  # noqa: PLC0415
        load_research_run,
        save_research_inputs,
        save_research_run,
    )
    from ..ml.supervised_ladder import (  # noqa: PLC0415
        DEFAULT_BUNDLE_LADDER_PROTOCOLS,
        run_supervised_ladder,
    )
    from .pipeline_regime import canonical_json_bytes  # noqa: PLC0415

    spec = context.semantic.payload
    if context.folds is None or context.labeled is None or "schedule" not in context.regime:
        raise ValueError("real model research requires persisted labels and an explicit calendar")
    primary = spec.feature_bundle_ids[0]
    envelope = context.bundle_views[primary]
    common = {
        "label_artifact_id": context.label_artifact_id,
        "fold_schedule_id": context.regime["schedule"].fold_schedule_id,
    }
    if envelope.payload.mbp1_feature_artifact_id is not None:
        runner = "mbp1"
        evidence = context.mbp1_evidence
        kwargs = {
            **common,
            "label_policy_id": spec.label_policy_id,
            "challenger_bundle_key": primary,
            "mbp1_features": evidence["feature_frame"],
            "mbp1_feature_artifact": evidence["feature_envelope"],
            "headline_protocol_id": spec.model_protocol_id,
        }
        view = context.view
        function = run_controlled_mbp1_study
    else:
        runner = "ladder"
        features = tuple(envelope.payload.resolved_feature_names)
        kwargs = {
            **common,
            "bundle_features": features,
            "bundle_ref": resolve_bundle(primary).resolved_feature_bundle_id,
            "bundle_evidence_ref": envelope.bundle_feature_view_id,
            "bundle_categorical_features": categorical_features_for(features),
            "protocols": DEFAULT_BUNDLE_LADDER_PROTOCOLS,
        }
        view = replace(context.view, frame=context.bundle_frames[primary])
        function = run_supervised_ladder
    request_id = save_research_inputs(
        context.store_root,
        view,
        context.labeled,
        context.folds,
        runner=runner,
        run_kwargs=kwargs,
    )
    checkpoint_research_model_input(context, request_id)
    run = load_research_run(context.store_root, request_id)
    reused = run is not None
    if run is None:
        run = function(view, context.labeled, context.folds, **kwargs)
        save_research_run(context.store_root, request_id, run)
        run = load_research_run(context.store_root, request_id)
    if run is None:  # pragma: no cover - persistence contract
        raise ValueError("completed research result could not be reloaded")
    outputs = []
    if runner == "mbp1":
        save_controlled_feature_study(context.store_root, run)
        context.controlled_study = run
        context.ladder = run.challenger
        outputs.append(run.envelope.controlled_feature_study_id)
        context.stage_sidecars["controlled_feature_study.json"] = canonical_json_bytes(
            run.envelope.model_dump(mode="json")
        )
    else:
        context.ladder = run
    outputs.append(context.ladder.ladder_id)
    outputs.append(request_id)
    context.research_model_run_ids.append(request_id)
    context.stage_sidecars["research_model_runs.json"] = canonical_json_bytes(
        {
            "request_ids": list(context.research_model_run_ids),
            "input_store": "research_model_inputs",
            "run_store": "research_model_runs",
        }
    )
    rows = int(context.ladder.parity.get("oos_row_count", 0))
    action = "verified and reloaded" if reused else "fitted, persisted and reloaded"
    return tuple(
        outputs
    ), f"{runner} {action}; {rows} OOS rows; {'evaluable' if rows else 'insufficient evidence'}"


def research_evidence_status(context):
    """Scientific adequacy is separate from successful stage execution."""
    reasons = []
    ladder = context.ladder
    rows = int(ladder.parity.get("oos_row_count", 0)) if ladder is not None else 0
    valid_folds = (
        min(
            (int(rung.predictions["fold_index"].nunique()) for rung in ladder.rungs),
            default=0,
        )
        if ladder is not None
        else 0
    )
    if rows == 0:
        reasons.append("No matching held-out model predictions were available")
    if valid_folds < 2:
        reasons.append("Fewer than two valid held-out folds; comparison evidence is insufficient")
    mbp_folds = []
    if context.mbp1_evidence:
        from ..features.mbp1_arrow_schemas import mbp1_window_validity_fields  # noqa: PLC0415

        frame = context.mbp1_evidence["feature_frame"]
        flags = list(mbp1_window_validity_fields())
        usable = set(frame.loc[frame[flags].fillna(False).any(axis=1), "candidate_id"].astype(str))
        for fold in context.folds.folds:
            if not fold.valid:
                continue
            train_count = len(usable & set(fold.train_candidate_ids))
            test_count = len(usable & set(fold.test_candidate_ids))
            mbp_folds.append(
                {
                    "fold_index": fold.fold_index,
                    "train_rows_with_valid_windows": train_count,
                    "test_rows_with_valid_windows": test_count,
                }
            )
            if not train_count or not test_count:
                reasons.append(f"MBP fold {fold.fold_index} lacks usable train or test windows")
    regime = context.regime.get("execution")
    gates_passed = None
    if regime is not None:
        assessment = regime.run.assessment.payload
        gates_passed = bool(assessment.gates_passed)
        reasons.extend(f"Regime gate: {reason}" for reason in assessment.gate_failures)
    return {
        "status": "insufficient_evidence" if reasons else "evaluable",
        "oos_rows": rows,
        "valid_oos_folds": valid_folds,
        "regime_gates_passed": gates_passed,
        "mbp_fold_coverage": mbp_folds,
        "reasons": reasons,
        "positive_lift_required": False,
    }


def research_reload_failures(context):
    from ..features.bundle_feature_view import (  # noqa: PLC0415
        load_bundle_feature_view,
        load_bundle_feature_view_frame,
    )
    from ..ml.fold_set_artifact import load_fold_schedule, load_fold_set_artifact  # noqa: PLC0415
    from ..ml.research_evidence import load_research_run  # noqa: PLC0415
    from .executed_trade_table import load_executed_trade_table  # noqa: PLC0415
    from .research_data import load_research_context_companion  # noqa: PLC0415

    root = context.store_root
    checks = {}
    if context.research_label_store_id is None:
        checks["research_labels/missing"] = lambda: (_ for _ in ()).throw(
            ValueError("real labels were not persisted")
        )
    else:
        checks[f"research_labels/{context.research_label_store_id}"] = lambda: load_research_labels(
            root, context.research_label_store_id
        )
    if context.research_cohort_id is not None:
        checks[f"research_cohorts/{context.research_cohort_id}"] = lambda: load_verified_envelope(
            root, "research_cohorts", context.research_cohort_id, ResearchCohortEnvelope
        )
    else:
        checks["research_cohorts/missing"] = lambda: (_ for _ in ()).throw(
            ValueError("scoped research cohort was not persisted")
        )
    source_id = context.wiring.research_preparation.label_source_reference["artifact_id"]
    checks[f"research_context_companions/{source_id}"] = lambda: load_research_context_companion(
        root, source_id
    )
    schedule = context.regime.get("schedule")
    folds = context.regime.get("candidate_fold_set")
    if schedule is not None:
        checks[f"fold_schedules/{schedule.fold_schedule_id}"] = lambda: load_fold_schedule(
            root, schedule.fold_schedule_id
        )
    if folds is not None:
        checks[f"fold_sets/{folds.fold_set_artifact_id}"] = lambda: load_fold_set_artifact(
            root, folds.fold_set_artifact_id
        )
    for evidence in context.executed_trades_by_child.values():
        table_id = evidence.executed_trade_table_id
        checks[f"executed_trade_tables/{table_id}"] = lambda table_id=table_id: (
            load_executed_trade_table(root, table_id)
        )
    for chart_id in context.regime.get("chart_ids", ()):
        checks[f"research_replay_charts/{chart_id}"] = lambda chart_id=chart_id: (
            context.wiring.context_bar_source(chart_id)
        )
    for envelope in context.bundle_views.values():
        view_id = envelope.bundle_feature_view_id

        def verify_view(view_id=view_id):
            loaded = load_bundle_feature_view(root, view_id)
            load_bundle_feature_view_frame(root, loaded)

        checks[f"bundle_feature_views/{view_id}"] = verify_view
    for request_id in context.research_model_run_ids:

        def verify_run(request_id=request_id):
            if load_research_run(root, request_id) is None:
                raise ValueError("research model output is absent")

        checks[f"research_model_runs/{request_id}"] = verify_run
    failures = {}
    for key, check in checks.items():
        try:
            check()
        except (OSError, ValueError, KeyError) as error:
            failures[key] = str(error)
    return failures
