"""Registered real R5–R6 adapter over exact saved strategy children."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ..contracts import RecordTable
from ..fsm_audit_preparation import ChildFsmAuditEnvelope
from ..replay_chart_store import (
    ArtifactPairRef,
    build_replay_chart_artifact,
    load_verified_replay_chart_artifact,
)
from .child_replay import ChildAuditNeutralityEnvelope
from .executed_trade_table import build_research_executed_trade_table
from .gates import evaluate_strategy_gates
from .identities import CoreStrategyReplayIdentity, canonical_contract_sha256
from .orchestrator import (
    ChildSpec,
    SearchChildMembershipEnvelope,
    _child_evaluation_envelope,
    _load_child_evaluation,
    _publish_child_evaluation,
)
from .pipeline import PipelineWiring
from .research_data import ResearchPreparation
from .research_runs import (
    REPO_ROOT,
    RESEARCH_DECISION_ID,
    ResearchRequest,
    ResearchSubjectEnvelope,
    _research_semantic,
    _software_identity,
    load_research_authorization,
)
from .research_subject import preflight_research_subject
from .store import load_verified_envelope, save_or_reuse_envelope
from .strategy_metrics import compute_strategy_metrics


def pipeline_real_research_entry(charter, semantic, *, store_root):
    """Validate exact authority before constructing lazy heavy adapters."""
    root = Path(store_root)
    refs = charter.payload.owner_authorization.decision_refs
    matching = [ref for ref in refs.values() if ref.decision_id == RESEARCH_DECISION_ID]
    if len(matching) != 1:
        raise PermissionError("real research requires its exact research approval")
    approval_id = matching[0].decision_artifact_id
    approval = load_research_authorization(root, approval_id)
    plan = json.loads(approval.payload.plan_json)
    subjects = [
        raw
        for raw in plan["subjects"]
        if canonical_contract_sha256(raw) in semantic.payload.source_artifact_ids
    ]
    if len(subjects) != 1:
        raise PermissionError("each research pipeline must bind one exact approved subject")
    subject_id = canonical_contract_sha256(subjects[0])
    subject = load_verified_envelope(
        root, "research_subjects", subject_id, ResearchSubjectEnvelope
    ).payload
    request = ResearchRequest.model_validate(plan["request"])

    def authorization_check():
        current = load_research_authorization(root, approval_id)
        if current != approval or matching[0] != approval.evidence_ref():
            raise PermissionError("research approval evidence differs from the frozen reference")
        if _software_identity() != plan["software_commits"]:
            raise PermissionError(
                "research software changed after review; create a newly reviewed request"
            )
        expected = _research_semantic(
            subject, approval_id, charter, request, plan["software_commits"]
        )
        if semantic != expected or subject.model_dump(mode="json") != subjects[0]:
            raise PermissionError("pipeline differs from the exact authorized research plan")
        from .charter import SearchCharterEnvelope  # noqa: PLC0415

        original = load_verified_envelope(
            root, "charters", subject.original_search_id, SearchCharterEnvelope
        )
        expected_payload = original.payload.model_copy(
            update={
                "owner_authorization": charter.payload.owner_authorization,
                "source_artifact_ids": (subject.subject_id, approval_id),
                "strategy_core_commit": plan["software_commits"]["strategy_core"],
                "quant_lab_commit": plan["software_commits"]["quant_lab"],
            }
        )
        if charter.payload != expected_payload:
            raise PermissionError("research charter differs from the selected saved strategy study")
        bundle = charter.payload.owner_authorization
        if (
            bundle.requirement_set_id != approval.payload.plan_id
            or dict(bundle.decision_refs) != {"research:exact_plan": approval.evidence_ref()}
            or bundle.store_namespace_id != approval.payload.store_namespace_id
            or bundle.supersession_head_witness != approval.payload.supersession_head_witness
        ):
            raise PermissionError("research charter authority differs from the approved plan")
        preflight_research_subject(subject, root, REPO_ROOT)
        if request.mbp1_comparison:
            from .research_mbp1 import preflight_research_mbp1  # noqa: PLC0415

            frozen = plan.get("mbp1_preflights", {}).get(subject.subject_id)
            actual = preflight_research_mbp1(subject, root, REPO_ROOT)
            if frozen is None or actual != frozen or actual.get("blockers"):
                raise PermissionError("MBP source evidence differs from the reviewed research plan")

    authorization_check()
    preparation = ResearchPreparation(
        subject,
        root,
        REPO_ROOT,
        cost_points=float(charter.payload.cost_policy.cost_points_round_turn),
        mbp1_preflight=plan.get("mbp1_preflights", {}).get(subject.subject_id),
    )
    child = ChildSpec(**subject.child_spec)
    core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)

    def identity_resolver(spec):
        if spec != child:
            raise PermissionError("requested child differs from the frozen research subject")
        return core

    def child_runner(**_kwargs):
        raise PermissionError(
            "research reuses the verified saved Core stream; context capture is explicit in S02"
        )

    def audit_builder(row, _result):
        if row["core_replay_id"] != subject.core_replay_id:
            raise PermissionError("audit requested for another research child")
        report = load_verified_envelope(
            root, "neutrality_reports", subject.neutrality_report_id, ChildAuditNeutralityEnvelope
        )
        if not report.payload.passed:
            raise PermissionError("research source neutrality failed")
        ids = []
        for path in sorted((root / "fsm_audit_companions").glob("*/envelope.json")):
            envelope = load_verified_envelope(
                root, "fsm_audit_companions", path.parent.name, ChildFsmAuditEnvelope
            )
            if envelope.payload.core_replay_id == subject.core_replay_id:
                ids.append(envelope.child_fsm_audit_id)
        if not ids:
            raise PermissionError("selected child has no verified FSM audit companion")
        return (*ids, subject.neutrality_report_id)

    chart_root = root / "research_replay_charts"

    def chart_builder(row, _result):
        if row["core_replay_id"] != subject.core_replay_id:
            raise PermissionError("chart requested for another research child")
        preparation.prepare()
        directory = build_replay_chart_artifact(
            preparation.pair,
            repo_root=REPO_ROOT,
            base_dir=chart_root,
            # Bars are already bound to the original input bundle; no unrelated
            # tick-bar oracle reads are needed to materialize this research chart.
            corroborate=False,
            research_label_source=(root, preparation.label_source_reference["artifact_id"]),
        )
        return (directory.name,)

    def context_bar_source(chart_id):
        preparation.prepare()
        return load_verified_replay_chart_artifact(
            chart_root,
            chart_id,
            expected_pair=ArtifactPairRef.from_verified_pair(preparation.pair),
        )

    return PipelineWiring(
        identity_resolver=identity_resolver,
        child_runner=child_runner,
        cost_points=float(charter.payload.cost_policy.cost_points_round_turn),
        audit_builder=audit_builder,
        chart_builder=chart_builder,
        candidate_view_source=preparation.candidate_view_source,
        label_builder=preparation.label_builder,
        mbp1_evidence_source=preparation.mbp1_evidence_source if request.mbp1_comparison else None,
        context_bar_source=context_bar_source,
        research_subject=subject,
        child_specs_source=lambda: (child,),
        research_preparation=preparation,
        research_authorization_check=authorization_check,
    )


def run_research_source_stage(context):
    """Reuse Core and prepare context, then evaluate a new scoped trade artifact.

    Reuses the same projection, cost metric, gate and lineage primitives as the
    ordinary pipeline. Its distinct inputs prevent legacy cache collisions.
    """
    from .identities import SearchChildMembership  # noqa: PLC0415
    from .lineage import (  # noqa: PLC0415
        build_native_lineage_map,
        persist_lineage_uniqueness,
        serialize_native_lineage_map,
    )
    from .pipeline import (  # noqa: PLC0415
        _checkpoint,
        _persist_and_load_executed_trade_table,
        _record_children,
        _record_executed_trade_evidence,
    )
    from .pipeline_regime import canonical_json_bytes  # noqa: PLC0415
    from .research_artifacts import save_research_cohort  # noqa: PLC0415

    wiring = context.wiring
    subject = wiring.research_subject
    wiring.research_authorization_check()
    if (
        len(context.children) != 1
        or context.children[0]["core_replay_id"] != subject.core_replay_id
    ):
        raise ValueError("real research source stage requires its single exact child")
    row = context.children[0]
    row.update(
        state="running", explanation="Verifying Core reuse and preparing exact context companion"
    )
    _record_children(context)
    _checkpoint(context)

    def progress(completed, total, trading_day):
        row["context_progress"] = {
            "completed_days": completed,
            "total_days": total,
            "trading_day": trading_day,
        }
        _record_children(context)
        _checkpoint(context)

    wiring.research_preparation.progress_callback = progress
    preparation = wiring.research_preparation.prepare()
    core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)
    report = load_verified_envelope(
        context.store_root,
        "neutrality_reports",
        subject.neutrality_report_id,
        ChildAuditNeutralityEnvelope,
    )
    if not report.payload.passed:
        raise PermissionError("selected Core replay failed neutrality")
    table_envelope, table_bytes = build_research_executed_trade_table(
        subject.core_replay_id,
        preparation.v2_tables[RecordTable.EXECUTED_TRADE],
        record_schema_version=core.payload.record_schema_version,
        research_subject_id=subject.subject_id,
        evaluation_dates=subject.evaluation_dates,
        candidate_ids=tuple(preparation.candidate_view.frame["candidate_id"].astype(str)),
        cutoff_ts_utc=subject.cutoff_ts_utc,
    )
    declared = report.payload.audit_disabled_core_table_hashes.get(RecordTable.EXECUTED_TRADE.value)
    if declared != table_envelope.source_core_table_hash:
        raise ValueError("research trade source differs from the verified neutrality table hash")
    stored = _persist_and_load_executed_trade_table(context, table_envelope, table_bytes)
    _record_executed_trade_evidence(context, row, stored.envelope, stored.frame)
    context.neutrality_by_child[subject.core_replay_id] = True
    result = SimpleNamespace(tables=preparation.v2_tables)
    context.tables_by_child[subject.core_replay_id] = result
    cohort = save_research_cohort(
        context.store_root,
        subject=subject,
        view=preparation.candidate_view,
        raw_tables=preparation.v2_tables,
        scoped_trades=stored.frame,
    )
    context.research_cohort_id = cohort.research_cohort_id
    evaluation = _child_evaluation_envelope(
        subject.core_replay_id,
        context.charter.payload.cost_policy,
        research_subject_id=subject.subject_id,
    )
    evaluation_hash = canonical_contract_sha256(
        {
            "core_replay_id": subject.core_replay_id,
            "cost_policy": context.charter.payload.cost_policy.model_dump(mode="json"),
            "research_subject_id": subject.subject_id,
        }
    )
    metrics = _load_child_evaluation(context.store_root, evaluation.costed_evaluation_id)
    if metrics is None:
        metrics = compute_strategy_metrics(
            {RecordTable.EXECUTED_TRADE: stored.frame},
            cost_points=wiring.cost_points,
            evaluation_config_hash=evaluation_hash,
            tick_size=float(context.charter.payload.cost_policy.tick_size),
            tp_r_multiple=float(subject.section_mapping["tp_r_multiple"]),
        )
        _publish_child_evaluation(context.store_root, evaluation, metrics)
    elif metrics.executed_trades and (
        metrics.trade_stats.get("evaluation_config_hash") != evaluation_hash
    ):
        raise ValueError("scoped costed evaluation has inconsistent provenance")
    context.metrics_by_child[subject.core_replay_id] = metrics
    gate = evaluate_strategy_gates(
        metrics, context.charter.payload.objective_policy.feasibility_gates
    )
    if gate.passed:
        context.gates_passed[subject.core_replay_id] = {
            name: getattr(metrics, name)
            for name in context.charter.payload.objective_policy.pareto_objectives
            if hasattr(metrics, name) and getattr(metrics, name) is not None
        }
    row["strategy_gate_passed"] = gate.passed
    row["strategy_gate_reason"] = gate.human_explanation
    row["costed_evaluation_id"] = evaluation.costed_evaluation_id
    row["research_cohort_id"] = cohort.research_cohort_id
    row["replay_invocations"] = int(getattr(preparation, "replay_invocations", 0))
    row["core_replay_invocations"] = 0
    row["context_replay_invocations"] = row["replay_invocations"]
    row["state"] = "reused"
    row["explanation"] = (
        "Verified saved Core stream; exact context companion and scoped research evidence prepared"
    )
    lineage = build_native_lineage_map(preparation.v2_tables, core_replay_id=subject.core_replay_id)
    persist_lineage_uniqueness(context.store_root, lineage.uniqueness_report)
    serialized = serialize_native_lineage_map(lineage)
    context.lineage_sidecars[subject.core_replay_id] = serialized
    context.stage_sidecars[f"lineage_map_{subject.core_replay_id}.json"] = canonical_json_bytes(
        serialized
    )
    context.stage_sidecars["research_cohort.json"] = canonical_json_bytes(
        cohort.model_dump(mode="json")
    )
    membership = SearchChildMembershipEnvelope.from_payload(
        SearchChildMembership(
            parent_search_id=context.charter.search_id,
            child_ordinal=row["ordinal"],
            axis_value_ids=row["axis_value_ids"],
            core_replay_id=subject.core_replay_id,
            comparison_role=row["comparison_role"],
        )
    )
    save_or_reuse_envelope(context.store_root, "memberships", membership)
    _record_children(context)
    _checkpoint(context)
    outputs = (
        subject.core_replay_id,
        preparation.label_source_reference["artifact_id"],
        subject.neutrality_report_id,
        cohort.research_cohort_id,
        stored.executed_trade_table_id,
        evaluation.costed_evaluation_id,
    )
    return (
        outputs,
        f"Exact Core replay reused; {len(stored.frame)} scoped trades; context companion verified",
    )
