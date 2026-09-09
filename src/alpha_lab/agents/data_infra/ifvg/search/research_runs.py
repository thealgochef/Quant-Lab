"""Real research requests, exact authorization and sequential subject groups.

Importing, listing and preflighting never prepare data or fit a model. The
detached worker is the only launch path used by the UI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from ..data_access import allowlist_sha256
from ..manifest import file_sha256, read_repository_state
from ..ml.regime_study import RegimeStudyRequest
from .authorization import OwnerAuthorizationBundle, OwnerDecisionEvidenceRef
from .charter import SearchCharterEnvelope, save_charter
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    strategy_core_source_identity,
)
from .owner_decisions import verify_complete_owner_authority_chain
from .pipeline import (
    FOLD_PROTOCOL_ID_V1,
    PipelineRunScope,
    PipelineSemanticIdentity,
    PipelineSemanticSpecPayload,
    QuantLabPipelineStage,
    assert_stage_plan_launchable,
    read_pipeline_state,
    request_pipeline_cancel,
)
from .research_data import LABEL_POLICY
from .research_subject import ResearchSubject
from .store import load_verified_envelope, save_or_reuse_envelope
from .store_namespace import SupersessionHeadWitness, require_store_namespace

REPO_ROOT = Path(__file__).resolve().parents[6]
RESEARCH_RUNNER_KEY = "pipeline_real_research_v1"
RESEARCH_DECISION_ID = "ifvg_real_research_approval_v1"
REAL_LABEL_POLICY_ID = LABEL_POLICY
RESEARCH_STAGE_PLAN = tuple(
    stage
    for stage in QuantLabPipelineStage
    if stage
    not in (
        QuantLabPipelineStage.S11_RUN_FROZEN_MODEL_GATED_REPLAYS,
        QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS,
        QuantLabPipelineStage.S13_RUN_BOOTSTRAP_AND_STRESS,
    )
)
BASE_BUNDLES = ("B0_CORE", "B1_CORE_STRUCTURE", "B4_CORE_STRUCTURE_LIQUIDITY")
MBP_CHALLENGERS = {
    "B0_CORE": "B2_CORE_ORDER_FLOW",
    "B1_CORE_STRUCTURE": "B3_CORE_STRUCTURE_ORDER_FLOW",
}


class ResearchRequest(FrozenContract):
    schema_version: Literal[1] = 1
    display_name: str = Field(min_length=1)
    evaluation_start: str
    evaluation_end: str
    base_bundle_key: Literal["B0_CORE", "B1_CORE_STRUCTURE", "B4_CORE_STRUCTURE_LIQUIDITY"] = (
        "B0_CORE"
    )
    mbp1_comparison: bool = False
    model_protocol_id: Literal["ifvg_context_logistic_l2_v1", "ifvg_context_catboost_bundle_v1"] = (
        "ifvg_context_catboost_bundle_v1"
    )
    regime_study: RegimeStudyRequest | None = None

    @model_validator(mode="after")
    def _supported(self):
        from datetime import date  # noqa: PLC0415

        start, end = (
            date.fromisoformat(self.evaluation_start),
            date.fromisoformat(self.evaluation_end),
        )
        if end < start:
            raise ValueError("research end precedes its start")
        if self.mbp1_comparison and self.base_bundle_key not in MBP_CHALLENGERS:
            raise ValueError("MBP comparisons support only B0→B2 and B1→B3")
        request = self.regime_study
        if request is not None:
            if set(request.comparison_classes_requested) - {
                "cohort_descriptive",
                "stratified_frontier",
                "feature_only",
            }:
                raise ValueError(
                    "this research preset supports descriptive and feature-only regimes"
                )
            if request.requires_supervision and (
                self.base_bundle_key != "B0_CORE"
                or self.mbp1_comparison
                or request.supervised_bundle_key != "B0_CORE"
            ):
                raise ValueError(
                    "the supervised regime comparison is B0→B7; combined MBP+regime is unavailable"
                )
        return self

    @property
    def feature_bundle_ids(self) -> tuple[str, ...]:
        primary = (
            MBP_CHALLENGERS[self.base_bundle_key] if self.mbp1_comparison else self.base_bundle_key
        )
        keys = [primary]
        if self.mbp1_comparison:
            keys.append(self.base_bundle_key)
        if self.regime_study is not None:
            key = self.regime_study.input_feature_bundle_key
            if key not in keys:
                keys.append(key)
        return tuple(keys)


class ResearchSubjectEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_subject_id"
    research_subject_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchSubject


class ResearchApprovalPayload(FrozenContract):
    decision_id: Literal["ifvg_real_research_approval_v1"] = RESEARCH_DECISION_ID
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    plan_id: str = Field(pattern=SHA256_PATTERN)
    plan_json: str
    author: str = Field(min_length=1)
    approved_at: str
    approval_statement: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bound(self):
        content = json.loads(self.plan_json)
        if canonical_contract_sha256(content) != self.plan_id:
            raise ValueError("research approval does not match its reviewed plan")
        if not self.author.strip() or not self.approval_statement.strip():
            raise ValueError("research approval requires a reviewer and explicit statement")
        if self.supersession_head_witness.store_namespace_id != self.store_namespace_id:
            raise ValueError("research approval witness belongs to another store")
        return self


class ResearchApprovalEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_approval_id"
    research_approval_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchApprovalPayload

    def evidence_ref(self) -> OwnerDecisionEvidenceRef:
        return OwnerDecisionEvidenceRef(
            decision_id=RESEARCH_DECISION_ID,
            decision_artifact_id=self.research_approval_id,
            content_hash=self.research_approval_id,
            author=self.payload.author,
            approved_at=self.payload.approved_at,
            effective_from=self.payload.approved_at,
            reviewed_evidence_refs=(self.payload.plan_id,),
        )


class ResearchGroupPayload(FrozenContract):
    schema_version: Literal[1] = 1
    plan_id: str = Field(pattern=SHA256_PATTERN)
    research_approval_id: str = Field(pattern=SHA256_PATTERN)
    display_name: str
    request_json: str
    cells_json: str


class ResearchGroupEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_group_id"
    research_group_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchGroupPayload


def _software_identity(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    from .research_subject import verify_loaded_core_source  # noqa: PLC0415

    root = Path(repo_root)
    source = root / "src/alpha_lab/agents/data_infra/ifvg"
    files = sorted(source.rglob("*.py"))
    for name in ("ifvg_pipeline_job.py", "ifvg_research_job.py"):
        path = root / "scripts" / name
        if path.is_file():
            files.append(path)
    source_hash = canonical_contract_sha256(
        {str(p.relative_to(root)): file_sha256(p) for p in files}
    )
    core_commit, core_source = strategy_core_source_identity(
        repository_root=root.parent / "Strategy-Core"
    )
    state = read_repository_state("quant-lab", root, source_paths=("src/alpha_lab",))
    return {
        "quant_lab": state.head,
        "strategy_core": core_commit,
        "quant_lab_research_source": source_hash,
        "strategy_core_source": core_source,
        "strategy_core_loaded_source": verify_loaded_core_source(root),
    }


def _subject_row(subject, root: Path) -> dict[str, Any]:
    charter = load_verified_envelope(
        root, "charters", subject.original_search_id, SearchCharterEnvelope
    )
    timeout = subject.section_mapping.get("parent_retest_timeout_1m_bars")
    return {
        "subject_id": subject.subject_id,
        "core_replay_id": subject.core_replay_id,
        "label": "Baseline — no parent timeout" if timeout is None else f"Parent timeout {timeout}",
        "source_search_id": subject.original_search_id,
        "profile_hash": subject.section_config_hash,
        "v2_dataset_id": subject.v2_dataset_id,
        "tp_r_multiple": subject.section_mapping["tp_r_multiple"],
        "cost_points": charter.payload.cost_policy.cost_points_round_turn,
        "replay_dates": list(subject.replay_dates),
        "warmup_dates": list(subject.warmup_dates),
        "evaluation_dates": list(subject.evaluation_dates),
        "cutoff_ts_utc": subject.cutoff_ts_utc,
        "blockers": [],
    }


def list_source_subjects(store_root: Path) -> list[dict[str, Any]]:
    """Return every saved child, including an explicit row for invalid evidence."""
    from .research_subject import bind_research_subject  # noqa: PLC0415

    root = Path(store_root)
    rows = []
    for path in sorted((root / "core_replays").glob("*/envelope.json")):
        try:
            rows.append(_subject_row(bind_research_subject(root, path.parent.name), root))
        except (ValueError, KeyError, OSError, PermissionError) as error:
            rows.append(
                {
                    "core_replay_id": path.parent.name,
                    "label": f"Unavailable child {path.parent.name[:12]}",
                    "blockers": [str(error)],
                    "evaluation_dates": [],
                }
            )
    return rows


def build_research_preflight(
    store_root: Path,
    source_core_replay_ids: list[str] | tuple[str, ...],
    study_spec: dict[str, Any],
) -> dict[str, Any]:
    """Freeze source truth and planned work without preparation, labels or fitting."""
    from .research_subject import bind_research_subject, preflight_research_subject  # noqa: PLC0415

    root = Path(store_root)
    request = ResearchRequest.model_validate(study_spec)
    ids = tuple(source_core_replay_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("select one or more distinct exact strategy children")
    namespace = require_store_namespace(root, expected_class="research")
    chain = verify_complete_owner_authority_chain(root)
    subjects, rows, blockers, warnings = [], [], [], []
    for core_id in ids:
        try:
            full = bind_research_subject(root, core_id)
            days = tuple(
                day
                for day in full.evaluation_dates
                if request.evaluation_start <= day <= request.evaluation_end
            )
            if not days:
                raise ValueError("the selected window has no authorized research dates")
            subject = bind_research_subject(root, core_id, evaluation_dates=days)
            facts = preflight_research_subject(subject, root, REPO_ROOT)
            facts["planned_complete_test_folds"] = max(0, (len(days) - 40) // 5)
            facts["fold_eligibility"] = "Pending real labels, interval purging and class checks"
            facts["fold_calendar_policy"] = {
                "initial_train_days": 40,
                "test_days": 5,
                "step_days": 5,
                "embargo_days": 2,
                "include_zero_candidate_days": True,
            }
            if facts["planned_complete_test_folds"] < 2:
                warnings.append(f"{core_id[:12]}: fewer than two complete test folds are possible")
            row = _subject_row(subject, root)
            row["preflight"] = facts
            row["blockers"] = list(facts.get("blockers", ()))
            blockers.extend(f"{row['label']}: {reason}" for reason in row["blockers"])
            subjects.append(subject.model_dump(mode="json"))
            rows.append(row)
        except (ValueError, KeyError, OSError, PermissionError) as error:
            blockers.append(f"{core_id[:12]}: {error}")
    if request.mbp1_comparison:
        from .research_mbp1 import preflight_research_mbp1  # noqa: PLC0415

        for subject_json, row in zip(subjects, rows, strict=True):
            facts = preflight_research_mbp1(
                ResearchSubject.model_validate(subject_json), root, REPO_ROOT
            )
            row["mbp1_preflight"] = facts
            blockers.extend(f"{row['label']}: {reason}" for reason in facts.get("blockers", ()))
            warnings.extend(str(reason) for reason in facts.get("warnings", ()))
    software = _software_identity()
    plan = {
        "schema_version": 1,
        "request": request.model_dump(mode="json"),
        "subjects": subjects,
        "software_commits": software,
        "store_namespace_id": namespace.store_namespace_id,
        "supersession_head_witness": chain.witness.model_dump(mode="json"),
        "stage_plan": [stage.value for stage in RESEARCH_STAGE_PLAN],
        "mbp1_preflights": {
            row["subject_id"]: row["mbp1_preflight"] for row in rows if "mbp1_preflight" in row
        },
    }
    plan_id = canonical_contract_sha256(plan)
    for subject_json in subjects:
        subject = ResearchSubject.model_validate(subject_json)
        original = load_verified_envelope(
            root, "charters", subject.original_search_id, SearchCharterEnvelope
        )
        try:
            assert_stage_plan_launchable(
                _research_semantic(subject, plan_id, original, request, software).payload,
                store_root=root,
                run_scope="full_authorized_development",
            )
        except (ValueError, PermissionError) as error:
            blockers.append(str(error))
    # Stage readiness for frozen regime authority is checked before approval.
    if request.regime_study is not None and request.regime_study.requires_supervision:
        from ..ml.regime_study import verify_frozen_authority  # noqa: PLC0415

        try:
            verify_frozen_authority(
                root,
                request.regime_study,
                expected_protocol_id=None,
                run_scope="full_authorized_development",
            )
        except (ValueError, PermissionError, OSError, TypeError) as error:
            blockers.append(str(error))
    lane = "R5 + R5B" if request.mbp1_comparison else "R5"
    if request.regime_study is not None:
        lane += (
            " + R6 prediction" if request.regime_study.requires_supervision else " + R6 descriptive"
        )
    return {
        "plan_id": plan_id,
        "plan": plan,
        "request": {
            "source_core_replay_ids": list(ids),
            "study_spec": request.model_dump(mode="json"),
        },
        "subjects": rows,
        "cells": [{**row, "lane": lane} for row in rows],
        "blockers": blockers,
        "warnings": warnings,
        "planned_stages": plan["stage_plan"],
        "ready": not blockers,
        "planned_work": [
            "Verify exact saved replay, audit neutrality and pinned source files",
            "Replay context only when the exact companion is absent; reconcile every Core table",
            "Create scoped features, real outcome labels and logical-calendar folds",
            "Fit or verified-reload the requested models; persist OOS evidence",
        ],
    }


def _research_semantic(subject, approval_id, charter, request, software):
    """One spec constructor shared by preflight, freeze and worker verification."""
    return PipelineSemanticIdentity.from_payload(
        PipelineSemanticSpecPayload(
            run_scope=PipelineRunScope.FULL_AUTHORIZED_DEVELOPMENT,
            date_allowlist=subject.replay_dates,
            allowlist_hash=allowlist_sha256(subject.replay_dates),
            warmup_policy_id="research_scoped_post_warmup_v1",
            search_charter_id=charter.search_id,
            source_artifact_ids=(
                subject.subject_id,
                approval_id,
                subject.core_replay_id,
                subject.v2_dataset_id,
                subject.replay_input_bundle_id,
            ),
            feature_bundle_ids=request.feature_bundle_ids,
            label_policy_id=REAL_LABEL_POLICY_ID,
            fold_protocol_id=FOLD_PROTOCOL_ID_V1,
            model_protocol_id=request.model_protocol_id,
            cost_policy_sha256=canonical_contract_sha256(charter.payload.cost_policy),
            account_policy_set_ids=(),
            portfolio_policy_ids=(),
            simulation_protocol=charter.payload.simulation_protocol,
            software_commits=software,
            stage_plan=RESEARCH_STAGE_PLAN,
            regime_study=request.regime_study,
        )
    )


def freeze_research_group(
    store_root: Path, preflight: dict[str, Any], *, authorization_statement: str, author: str
) -> dict[str, Any]:
    """Persist a reviewed request and new authority; launches nothing."""
    from .research_subject import ResearchSubject  # noqa: PLC0415

    root = Path(store_root)
    fresh = build_research_preflight(root, **preflight["request"])
    if fresh["plan_id"] != preflight["plan_id"]:
        raise ValueError("research inputs changed after review; refresh preflight before approval")
    if not fresh["ready"]:
        raise ValueError("research preflight blocked: " + "; ".join(fresh["blockers"]))
    plan = fresh["plan"]
    request = ResearchRequest.model_validate(plan["request"])
    approval = ResearchApprovalEnvelope.from_payload(
        ResearchApprovalPayload(
            store_namespace_id=plan["store_namespace_id"],
            supersession_head_witness=SupersessionHeadWitness.model_validate(
                plan["supersession_head_witness"]
            ),
            plan_id=fresh["plan_id"],
            plan_json=json.dumps(plan, sort_keys=True),
            author=author.strip(),
            approved_at=datetime.now(UTC).isoformat(),
            approval_statement=authorization_statement.strip(),
        )
    )
    save_or_reuse_envelope(root, "research_approvals", approval)
    cells = []
    for subject_json, row in zip(plan["subjects"], fresh["cells"], strict=True):
        subject = ResearchSubject.model_validate(subject_json)
        subject_envelope = ResearchSubjectEnvelope.from_payload(subject)
        if subject_envelope.research_subject_id != subject.subject_id:
            raise ValueError("research subject serialization changed its identity")
        save_or_reuse_envelope(root, "research_subjects", subject_envelope)
        original = load_verified_envelope(
            root, "charters", subject.original_search_id, SearchCharterEnvelope
        )
        # This charter carries NEW authority. Original strategy approval is kept
        # in the source subject's lineage and never grants model permissions.
        authorization = OwnerAuthorizationBundle(
            requirement_set_id=approval.payload.plan_id,
            decision_refs={"research:exact_plan": approval.evidence_ref()},
            store_namespace_id=approval.payload.store_namespace_id,
            supersession_head_witness=approval.payload.supersession_head_witness,
        )
        charter = SearchCharterEnvelope.from_payload(
            original.payload.model_copy(
                update={
                    "owner_authorization": authorization,
                    "source_artifact_ids": (subject.subject_id, approval.research_approval_id),
                    "strategy_core_commit": plan["software_commits"]["strategy_core"],
                    "quant_lab_commit": plan["software_commits"]["quant_lab"],
                }
            )
        )
        semantic = _research_semantic(
            subject, approval.research_approval_id, charter, request, plan["software_commits"]
        )
        assert_stage_plan_launchable(
            semantic.payload, store_root=root, run_scope="full_authorized_development"
        )
        save_charter(root, charter)
        save_or_reuse_envelope(root, "pipeline_specs", semantic)
        cells.append(
            {
                "subject_id": subject.subject_id,
                "core_replay_id": subject.core_replay_id,
                "label": row["label"],
                "lane": row["lane"],
                "pipeline_semantic_id": semantic.pipeline_semantic_id,
            }
        )
    group = ResearchGroupEnvelope.from_payload(
        ResearchGroupPayload(
            plan_id=fresh["plan_id"],
            research_approval_id=approval.research_approval_id,
            display_name=request.display_name,
            request_json=request.model_dump_json(),
            cells_json=json.dumps(cells, sort_keys=True),
        )
    )
    save_or_reuse_envelope(root, "research_groups", group)
    return {"group_id": group.research_group_id, "cells": cells, "status": "frozen"}


def load_research_authorization(store_root: Path, approval_id: str) -> ResearchApprovalEnvelope:
    root = Path(store_root)
    namespace = require_store_namespace(root, expected_class="research")
    approval = load_verified_envelope(
        root, "research_approvals", approval_id, ResearchApprovalEnvelope
    )
    if approval.payload.store_namespace_id != namespace.store_namespace_id:
        raise PermissionError("research approval belongs to another store")
    verify_complete_owner_authority_chain(
        root, expected_head_witness=approval.payload.supersession_head_witness
    )
    if datetime.fromisoformat(approval.payload.approved_at) > datetime.now(UTC):
        raise PermissionError("research approval is not yet effective")
    return approval


def launch_research_group(store_root: Path, group_id: str, state_root: Path) -> dict[str, Any]:
    """Validate the frozen group and start one hidden sequential worker."""
    from .research_executor import pipeline_real_research_entry  # noqa: PLC0415

    root, state = Path(store_root).resolve(), Path(state_root).resolve()
    group = load_verified_envelope(root, "research_groups", group_id, ResearchGroupEnvelope)
    load_research_authorization(root, group.payload.research_approval_id)
    for cell in json.loads(group.payload.cells_json):
        semantic = load_verified_envelope(
            root, "pipeline_specs", cell["pipeline_semantic_id"], PipelineSemanticIdentity
        )
        charter = load_verified_envelope(
            root, "charters", semantic.payload.search_charter_id, SearchCharterEnvelope
        )
        # The factory verifies every frozen reference, but preparation stays lazy.
        pipeline_real_research_entry(charter, semantic, store_root=root)
    job_dir = state / "groups" / group_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "cancel.requested").unlink(missing_ok=True)
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/ifvg_research_job.py"),
        "worker",
        "--store-root",
        str(root),
        "--state-root",
        str(state),
        "--group-id",
        group_id,
    ]
    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )
    with (job_dir / "job.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )  # noqa: S603
    return {"group_id": group_id, "pid": process.pid, "status": "started"}


def _cell_state(state: dict | None) -> tuple[str, str | None]:
    if state is None:
        return "queued", None
    entries = [row for row in state.get("stages", {}).values() if row.get("in_plan")]
    statuses = {row.get("status") for row in entries}
    for value in ("running", "cancel_requested", "failed", "blocked", "cancelled_at_safe_boundary"):
        if value in statuses:
            reasons = [row.get("explanation", "") for row in entries if row.get("status") == value]
            return value, "; ".join(reason for reason in reasons if reason) or None
    if entries and statuses <= {"completed", "reused"}:
        publication = state.get("publication", {})
        if publication.get("control_flow_gates_passed") is False:
            return "blocked", "Publication verification failed; inspect the preserved artifacts"
        return "completed", None
    return "queued", None


def read_research_group(store_root: Path, group_id: str, state_root: Path) -> dict[str, Any]:
    group = load_verified_envelope(
        Path(store_root), "research_groups", group_id, ResearchGroupEnvelope
    )
    cells = json.loads(group.payload.cells_json)
    for cell in cells:
        state = read_pipeline_state(Path(state_root), cell["pipeline_semantic_id"])
        status, reason = _cell_state(state)
        cell.update(state=state, status=status, reason=reason)
        cell["artifact_ids"] = sorted(
            {
                str(artifact)
                for row in (state or {}).get("stages", {}).values()
                for artifact in row.get("output_artifact_ids", ())
            }
        )
    operational_path = Path(state_root) / "groups" / group_id / "group_state.json"
    operational = (
        json.loads(operational_path.read_text(encoding="utf-8"))
        if operational_path.is_file()
        else {}
    )
    return {
        "group_id": group_id,
        "display_name": group.payload.display_name,
        "status": operational.get("status", "frozen"),
        "reason": operational.get("reason"),
        "cells": cells,
        "study_spec": json.loads(group.payload.request_json),
        "research_approval_id": group.payload.research_approval_id,
    }


def list_research_groups(store_root: Path, state_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((Path(store_root) / "research_groups").glob("*/envelope.json")):
        try:
            rows.append(read_research_group(store_root, path.parent.name, state_root))
        except (OSError, ValueError) as error:
            rows.append(
                {
                    "group_id": path.parent.name,
                    "display_name": "Research evidence unavailable",
                    "status": "failed",
                    "reason": str(error),
                    "cells": [],
                }
            )
    return rows


def cancel_research_group(store_root: Path, group_id: str, state_root: Path) -> dict[str, Any]:
    group = read_research_group(store_root, group_id, state_root)
    directory = Path(state_root) / "groups" / group_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cancel.requested").touch()
    for cell in group["cells"]:
        if cell["status"] in ("running", "cancel_requested"):
            request_pipeline_cancel(Path(state_root), cell["pipeline_semantic_id"])
    return {"group_id": group_id, "status": "cancel_requested"}


def run_research_group(store_root: Path, group_id: str, state_root: Path) -> dict[str, Any]:
    """Worker-only orchestration; each subject uses the registered pipeline."""
    from importlib import import_module  # noqa: PLC0415

    from ..preparation import _write_json_atomic  # noqa: PLC0415
    from .failure import sanitize_failure_message  # noqa: PLC0415
    from .orchestrator import _search_lock  # noqa: PLC0415
    from .pipeline import WorkerPolicy, run_pipeline  # noqa: PLC0415
    from .runner_registry import resolve_registered_runner_entry  # noqa: PLC0415

    root, state = Path(store_root), Path(state_root)
    group = load_verified_envelope(root, "research_groups", group_id, ResearchGroupEnvelope)
    directory = state / "groups" / group_id
    with _search_lock(state / "groups", group_id, stale_lock_seconds=86_400):
        directory.mkdir(parents=True, exist_ok=True)
        sentinel = directory / "cancel.requested"
        record = {
            "group_id": group_id,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(),
        }
        _write_json_atomic(directory / "group_state.json", record)
        try:
            load_research_authorization(root, group.payload.research_approval_id)
            entry = resolve_registered_runner_entry(RESEARCH_RUNNER_KEY)
            module, name = entry.split(":", 1)
            factory = getattr(import_module(module), name)
            cells = json.loads(group.payload.cells_json)
            for cell in cells:
                if sentinel.exists():
                    record["status"] = "cancelled_at_safe_boundary"
                    break
                record["active_pipeline_semantic_id"] = cell["pipeline_semantic_id"]
                _write_json_atomic(directory / "group_state.json", record)
                semantic = load_verified_envelope(
                    root, "pipeline_specs", cell["pipeline_semantic_id"], PipelineSemanticIdentity
                )
                charter = load_verified_envelope(
                    root, "charters", semantic.payload.search_charter_id, SearchCharterEnvelope
                )
                wiring = factory(charter, semantic, store_root=root)
                run_pipeline(
                    semantic,
                    charter,
                    store_root=root,
                    state_root=state,
                    wiring=wiring,
                    worker_policy=WorkerPolicy(
                        max_workers=1, max_tasks_per_child=1, memory_budget_bytes=2 << 30
                    ),
                    operational_retry_reason="research_group_launch_or_resume",
                )
                status, reason = _cell_state(
                    read_pipeline_state(state, cell["pipeline_semantic_id"])
                )
                if status != "completed":
                    record.update(status=status, reason=reason)
                    break
            else:
                record["status"] = "completed"
        except Exception as error:  # noqa: BLE001 - persist worker failure for UI recovery
            record.update(status="failed", reason=sanitize_failure_message(str(error)))
        record["ended_at"] = datetime.now(UTC).isoformat()
        record.pop("active_pipeline_semantic_id", None)
        _write_json_atomic(directory / "group_state.json", record)
    return record


def read_research_regime_eligibility(store_root, group_id, pipeline_id):
    from .research_regimes import read_research_regime_eligibility as read

    return read(store_root, group_id, pipeline_id)


def promote_research_regime(
    store_root, group_id, pipeline_id, *, author, approval_statement, review_id=None
):
    from .research_regimes import promote_research_regime as promote

    return promote(
        store_root,
        group_id,
        pipeline_id,
        author=author,
        approval_statement=approval_statement,
        review_id=review_id,
    )
