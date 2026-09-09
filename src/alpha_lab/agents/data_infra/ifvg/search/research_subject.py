"""Versioned research subjects bound to one verified saved strategy child."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

from ..config import IfvgCaptureConfig
from ..contracts import RecordTable
from ..dataset import table_content_hash
from ..development_access import (
    DEVELOPMENT_CUTOFF_UTC,
    FROZEN_WARMUP_DATES,
    DevelopmentReplayPolicy,
)
from ..manifest import file_sha256
from .charter import SearchCharterEnvelope
from .child_replay import ChildAuditNeutralityEnvelope
from .identities import SHA256_PATTERN, FrozenContract, canonical_contract_sha256
from .review_evidence import load_search_review_evidence
from .store import load_verified_envelope


class ResearchSubject(FrozenContract):
    """The calendar includes zero-candidate days; cohort never pools children."""

    schema_version: Literal[1] = 1
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    v2_dataset_id: str = Field(pattern=SHA256_PATTERN)
    v2_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    replay_input_bundle_id: str = Field(pattern=SHA256_PATTERN)
    section_config_hash: str = Field(pattern=SHA256_PATTERN)
    section_json: str
    core_envelope_json: str
    neutrality_report_id: str = Field(pattern=SHA256_PATTERN)
    original_search_id: str = Field(pattern=SHA256_PATTERN)
    child_spec_json: str
    replay_dates: tuple[str, ...]
    warmup_dates: tuple[str, ...]
    evaluation_dates: tuple[str, ...]
    artifact_provenance_dates: tuple[str, ...]
    cohort: Literal["all_candidates", "eligible_decisions"] = "all_candidates"
    cutoff_ts_utc: str = ""

    @model_validator(mode="after")
    def _exact_scope(self):
        for name in (
            "replay_dates",
            "warmup_dates",
            "evaluation_dates",
            "artifact_provenance_dates",
        ):
            values = getattr(self, name)
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be an ordered unique calendar")
            for value in values:
                date.fromisoformat(value)
        if self.warmup_dates != FROZEN_WARMUP_DATES:
            raise ValueError("research requires the exact saved ten-date warmup")
        if not self.evaluation_dates or set(self.evaluation_dates) & set(self.warmup_dates):
            raise ValueError("evaluation calendar must be nonempty and exclude warmup")
        if not set(self.evaluation_dates) <= set(self.replay_dates):
            raise ValueError("evaluation dates must belong to the exact replay")
        if not set(self.replay_dates) <= set(self.artifact_provenance_dates):
            raise ValueError("artifact provenance must cover the exact replay")
        DevelopmentReplayPolicy(self.replay_dates)
        expected_cutoff = research_study_cutoff(self.evaluation_dates[-1])
        if not self.cutoff_ts_utc:
            object.__setattr__(self, "cutoff_ts_utc", expected_cutoff)
        elif self.cutoff_ts_utc != expected_cutoff:
            raise ValueError("research cutoff must equal the selected study end at 17:00 ET")
        section = IfvgSmcSection.model_validate_json(self.section_json)
        if ifvg_profile_hash(section) != self.section_config_hash:
            raise ValueError("subject section differs from its exact child hash")
        from .identities import CoreStrategyReplayIdentity  # noqa: PLC0415

        core = CoreStrategyReplayIdentity.model_validate_json(self.core_envelope_json)
        if (
            core.core_replay_id != self.core_replay_id
            or core.payload.resolved_section_config_hash != self.section_config_hash
            or core.payload.replay_input_bundle_id != self.replay_input_bundle_id
        ):
            raise ValueError("subject core envelope differs from its binding")
        return self

    @property
    def subject_id(self) -> str:
        return canonical_contract_sha256(self)

    def model_copy(self, *, update=None, deep=False):
        updates = dict(update or {})
        if "evaluation_dates" in updates and "cutoff_ts_utc" not in updates:
            updates["cutoff_ts_utc"] = ""
        return type(self).model_validate({**self.model_dump(), **updates})

    @property
    def section_mapping(self) -> dict:
        return json.loads(self.section_json)

    @property
    def child_spec(self) -> dict:
        return json.loads(self.child_spec_json)


def research_study_cutoff(evaluation_end: str) -> str:
    end = datetime.combine(
        date.fromisoformat(evaluation_end), time(17), ZoneInfo("America/New_York")
    )
    protected = datetime.fromisoformat(DEVELOPMENT_CUTOFF_UTC.replace("Z", "+00:00"))
    return min(end.astimezone(UTC), protected).isoformat().replace("+00:00", "Z")


def _loaded_core_package_root() -> Path:
    import strategy_core  # noqa: PLC0415

    if not strategy_core.__file__:
        raise ValueError("loaded Strategy-Core has no inspectable package source")
    return Path(strategy_core.__file__).resolve().parent


def verify_loaded_core_source(repo_root: Path) -> str:
    """Bind replay imports to checkout source; normalize only text line endings.

    Existing Core artifact identities keep their historical checkout semantics.
    This separate research preflight check verifies the code capture imports.
    """
    checkout = (Path(repo_root).parent / "Strategy-Core/src/strategy_core").resolve()
    loaded = _loaded_core_package_root()

    def source_hashes(root):
        files = sorted(root.rglob("*.py"))
        if not files:
            raise ValueError("Strategy-Core source directory contains no Python modules")
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_text(encoding="utf-8").encode("utf-8")
            ).hexdigest()
            for path in files
        }

    expected = source_hashes(checkout)
    actual = expected if loaded == checkout else source_hashes(loaded)
    mismatched = sorted(
        name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name)
    )
    if mismatched:
        raise ValueError(
            "loaded Strategy-Core differs from the checked source: " + ", ".join(mismatched[:8])
        )
    return canonical_contract_sha256(expected)


def _neutrality_id(root: Path, core_replay_id: str, accepted_tables: dict) -> str:
    matches = []
    for path in sorted((root / "neutrality_reports").glob("*/envelope.json")):
        envelope = load_verified_envelope(
            root, "neutrality_reports", path.parent.name, ChildAuditNeutralityEnvelope
        )
        report = envelope.payload
        if report.core_replay_id == core_replay_id:
            if not report.passed or not report.audit_stamp_referential_integrity:
                raise ValueError("saved child failed audit neutrality")
            if report.mechanism == "dual_drive_ab_v1" and (
                report.tables_equal is not True
                or report.audit_disabled_core_table_hashes != report.audit_enabled_core_table_hashes
            ):
                raise ValueError("saved child neutrality table hashes disagree")
            if report.mechanism == "dual_drive_ab_v1":
                accepted_hashes = {
                    table.value: table_content_hash(table, accepted_tables[table])
                    for table in RecordTable
                }
                if dict(report.audit_enabled_core_table_hashes or {}) != accepted_hashes:
                    raise ValueError("saved child neutrality does not bind the accepted v2 tables")
            matches.append(envelope.neutrality_report_id)
    if not matches:
        raise ValueError("saved child has no verified passing neutrality evidence")
    return matches[0]


def bind_research_subject(
    store_root: Path,
    core_replay_id: str,
    *,
    evaluation_dates: tuple[str, ...] | None = None,
    cohort: Literal["all_candidates", "eligible_decisions"] = "all_candidates",
) -> ResearchSubject:
    """Read saved envelopes/tables only; never replay or infer days from candidates."""
    from .orchestrator import SearchChildMembershipEnvelope  # noqa: PLC0415
    from .strategy_approval import (  # noqa: PLC0415
        StrategySearchApprovalEnvelope,
        charter_intent_hash,
    )

    root = Path(store_root)
    evidence = load_search_review_evidence(root, core_replay_id)
    raw = evidence.dataset.reports["raw_config.json"]
    search_id = raw.get("search_id")
    if not search_id:
        raise ValueError("saved replay lacks an original search charter reference")
    charter = load_verified_envelope(root, "charters", search_id, SearchCharterEnvelope)
    dates = tuple(charter.payload.date_policy.replay_dates)
    input_dates = tuple(
        sorted({r.trading_day for r in evidence.inputs.payload.ordered_day_artifacts})
    )
    if dates != input_dates or tuple(raw["allowlist"]) != dates:
        raise ValueError("saved charter, input bundle, and replay calendars differ")
    section = IfvgSmcSection.model_validate(
        evidence.dataset.reports["effective_config.json"]["section"]
    )
    memberships = []
    for path in sorted((root / "memberships").glob("*/envelope.json")):
        membership = load_verified_envelope(
            root, "memberships", path.parent.name, SearchChildMembershipEnvelope
        ).payload
        if membership.parent_search_id == search_id and membership.core_replay_id == core_replay_id:
            memberships.append(membership)
    if len(memberships) != 1:
        raise ValueError("saved child does not resolve uniquely in its original memberships")
    membership = memberships[0]
    # Frozen saved values, not a new search enumeration against today's registry.
    spec = {
        "ordinal": membership.child_ordinal,
        "axis_value_ids": dict(membership.axis_value_ids),
        "comparison_role": membership.comparison_role,
        "canonical_profile_id": evidence.core.payload.canonical_profile_id,
        "resolved_section_config_hash": evidence.core.payload.resolved_section_config_hash,
        "capability": None,
        "section_overrides": section.model_dump(mode="json"),
    }
    approvals = [
        ref
        for ref in charter.payload.owner_authorization.decision_refs.values()
        if ref.decision_id == "strategy_search_approval_v1"
    ]
    if not approvals:
        raise ValueError("saved child lacks exact source provenance authorization")
    approval = load_verified_envelope(
        root,
        "strategy_search_approvals",
        approvals[0].decision_artifact_id,
        StrategySearchApprovalEnvelope,
    )
    if approval.payload.charter_intent_sha256 != charter_intent_hash(charter.payload):
        raise ValueError("saved source authorization belongs to a different charter")
    warmup = tuple(charter.payload.date_policy.warmup_dates)
    return ResearchSubject(
        **evidence.reference,
        section_config_hash=evidence.core.payload.resolved_section_config_hash,
        section_json=section.model_dump_json(),
        core_envelope_json=evidence.core.model_dump_json(),
        neutrality_report_id=_neutrality_id(root, core_replay_id, evidence.dataset.tables),
        original_search_id=search_id,
        child_spec_json=json.dumps(spec, sort_keys=True),
        replay_dates=dates,
        warmup_dates=warmup,
        evaluation_dates=evaluation_dates or tuple(day for day in dates if day not in warmup),
        artifact_provenance_dates=approval.payload.artifact_provenance_dates,
        cohort=cohort,
    )


def list_research_subjects(store_root: Path) -> tuple[ResearchSubject, ...]:
    """Discover verified saved children, without scanning source data directories."""
    subjects = []
    for path in sorted((Path(store_root) / "core_replays").glob("*/envelope.json")):
        try:
            subjects.append(bind_research_subject(store_root, path.parent.name))
        except (ValueError, KeyError, FileNotFoundError, PermissionError):
            continue
    return tuple(subjects)


def preflight_research_subject(subject: ResearchSubject, store_root: Path, repo_root: Path) -> dict:
    """Verify saved binding and source bytes without decoding or replaying bars."""
    from .identities import (  # noqa: PLC0415
        CoreStrategyReplayIdentity,
        strategy_core_source_identity,
    )
    from .research_data import (  # noqa: PLC0415
        CONTEXT_STORE,
        ResearchContextEnvelope,
        expected_research_context,
    )
    from .store import has_envelope  # noqa: PLC0415

    rebound = bind_research_subject(
        store_root,
        subject.core_replay_id,
        evaluation_dates=subject.evaluation_dates,
        cohort=subject.cohort,
    )
    if rebound != subject:
        raise ValueError("saved research subject changed after binding")
    core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)
    current_commit, current_source = strategy_core_source_identity(
        repository_root=Path(repo_root).parent / "Strategy-Core"
    )
    if (current_commit, current_source) != (
        core.payload.strategy_core_commit,
        core.payload.strategy_core_source_identity,
    ):
        raise ValueError("current Strategy-Core differs from the saved child source")
    loaded_source_hash = verify_loaded_core_source(repo_root)
    evidence = load_search_review_evidence(store_root, subject.core_replay_id)
    cfg = IfvgCaptureConfig(
        section=IfvgSmcSection.model_validate(subject.section_mapping),
        data_dir=Path(repo_root) / "data/databento",
    )
    input_refs = verify_research_input_hashes(subject, evidence, cfg)
    expected_context, _cfg = expected_research_context(subject, repo_root)
    companion_id = expected_context.research_context_companion_id
    context_reusable = has_envelope(store_root, CONTEXT_STORE, companion_id)
    if context_reusable:
        load_verified_envelope(store_root, CONTEXT_STORE, companion_id, ResearchContextEnvelope)
    from .executed_trade_table import research_trade_cohort_masks  # noqa: PLC0415

    candidates = evidence.dataset.tables[RecordTable.ENTRY_CANDIDATE]
    if not candidates["is_warmup"].isin([True, False]).all():
        raise ValueError("saved research warmup stamps must be explicit booleans")
    warmup = candidates["is_warmup"].astype(bool)
    scoped = candidates["trading_day"].astype(str).isin(subject.evaluation_dates) & ~warmup
    if subject.cohort == "eligible_decisions":
        eligible_ids = evidence.dataset.tables[RecordTable.ELIGIBLE_DECISION]["candidate_id"]
        scoped &= candidates["candidate_id"].isin(eligible_ids)
    trades = evidence.dataset.tables[RecordTable.EXECUTED_TRADE]
    trade_masks = research_trade_cohort_masks(
        trades,
        candidate_ids=tuple(candidates.loc[scoped, "candidate_id"].astype(str)),
        cutoff_ts_utc=subject.cutoff_ts_utc,
    )
    counts = {
        "candidate_rows": int(scoped.sum()),
        "total_candidate_rows": len(candidates),
        "warmup_candidate_rows": int(warmup.sum()),
        "excluded_candidate_rows": int((~scoped).sum()),
        "trade_rows": int(trade_masks["included"].sum()),
        "total_trade_rows": len(trades),
        "warmup_trade_rows": int(trade_masks["warmup"].sum()),
        "excluded_trade_rows": int((~trade_masks["included"]).sum()),
        "excluded_window_trade_rows": int(trade_masks["out_of_cohort"].sum()),
        "cutoff_censored_trade_rows": int(trade_masks["cutoff_censored"].sum()),
        "excluded_unresolved_trade_rows": int(trade_masks["unresolved"].sum()),
    }
    return {
        "passed": True,
        "subject_id": subject.subject_id,
        "strategy_core_loaded_source_sha256": loaded_source_hash,
        **counts,
        "input_partition_refs": input_refs,
        "context_companion_id": companion_id,
        "context_companion_reusable": context_reusable,
        "planned_context_replay_invocations": int(not context_reusable),
        "evaluation_dates": list(subject.evaluation_dates),
        "neutrality_report_id": subject.neutrality_report_id,
    }


def verify_research_input_hashes(subject, evidence, cfg) -> list[dict]:
    """Authorize each exact input reference before constructing or hashing a path."""
    policy = DevelopmentReplayPolicy(subject.replay_dates)
    references = []
    for ref in evidence.inputs.payload.ordered_day_artifacts:
        if ref.artifact_kind not in {"bars", "levels"}:
            raise ValueError("unsupported saved research input kind")
        factory = cfg.bars_path if ref.artifact_kind == "bars" else cfg.levels_path
        path = policy.resolve_source_path(ref.trading_day, factory)
        policy.record_file_open(ref.trading_day)
        if path.name != ref.artifact_id or file_sha256(path) != ref.content_sha256:
            raise ValueError("research input differs from the exact saved child reference")
        references.append(
            {
                "trading_day": ref.trading_day,
                "artifact_kind": ref.artifact_kind,
                "artifact_id": ref.artifact_id,
                "sha256": ref.content_sha256,
            }
        )
    policy.assert_zero_forbidden_access()
    return references
