"""Read-only review and explicit recording of one strategy-search approval.

Preparing a review reads cache footers only. Recording saves owner evidence;
neither operation freezes a charter, starts a worker, or runs market data.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pyarrow.parquet as pq
from pydantic import TypeAdapter
from strategy_core.candles._buckets import HTF_ANCHOR_POLICY
from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

from ..config import IfvgCaptureConfig
from ..context_experiment_contracts import PROFILE_CAPABILITY_REGISTRY, ProfileCapabilityStatus
from ..data_access import allowlist_sha256
from ..dataset import _chained_seeds
from ..day_artifacts import _ARTIFACT_SCHEMA_VERSION, _META_KEY, DayArtifacts, DaySeeds
from ..development_access import PERMITTED_DEVELOPMENT_DATES, DevelopmentReplayPolicy
from ..profiles import resolve_profile_config
from ..study.computation_path import ComputationPath
from ..study_providers import list_catalogued_envelope_ids, owner_authorization_bundle_from_store
from .authorization import derive_authorization_requirements
from .axis_registry import (
    SEARCH_AXIS_REGISTRY_V1,
    assert_axes_authorized,
    registry_sha256,
    resolve_axis_overrides,
)
from .catalog import append_catalog_event
from .charter import SearchCharterPayload, validate_charter
from .identities import canonical_contract_sha256, canonicalize_section
from .orchestrator import _locked_invariant_violations
from .owner_decisions import verify_complete_owner_authority_chain
from .runtime_source import research_preparation_approval, strategy_core_repository_root
from .store_namespace import require_store_namespace
from .strategy_approval import (
    DECISION_KEYS,
    STORE,
    StrategySearchApprovalEnvelope,
    StrategySearchApprovalPayload,
    charter_intent,
    charter_intent_hash,
    load_strategy_approval,
    persist_strategy_approval,
)
from .strategy_metrics import StrategyMetrics
from .supersession_chain import current_supersession_head_witness


@dataclass(frozen=True)
class StrategyApprovalReview:
    intent: dict
    intent_sha256: str
    artifact_provenance_dates: tuple[str, ...]
    cache_checked_dates: int
    blockers: tuple[str, ...]
    configuration_rows: tuple[dict, ...]
    requirement_set_id: str
    store_namespace_id: str | None
    evidence_sha256: str


def _check_source_commits(intent):
    """Resolve both recorded commits in their actual installed checkouts."""
    ql_root = Path(__file__).resolve().parents[6]
    for key in ("quant_lab_commit", "strategy_core_commit"):
        if not re.fullmatch(r"[0-9a-f]{40}", intent[key]):
            raise ValueError(f"The {key} source revision is unavailable; refresh the study.")
    sc_root = strategy_core_repository_root(ql_root)
    for key, root in (("quant_lab_commit", ql_root), ("strategy_core_commit", sc_root)):
        commit = intent[key]
        result = subprocess.run(  # noqa: S603, S607 — exact validated read-only git object query
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=root,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode:
            raise ValueError(f"The {key} source revision cannot be verified locally.")


def _typed_intent(fields):
    """Validate the charter's fields without fabricating an authorization."""
    supplied = charter_intent(fields)
    known = set(SearchCharterPayload.model_fields) - {"owner_authorization"}
    if set(supplied) - known:
        raise ValueError("The study includes unrecognized configuration fields.")
    typed = {}
    for name, field in SearchCharterPayload.model_fields.items():
        if name == "owner_authorization":
            continue
        if name not in supplied and field.is_required():
            raise ValueError(f"Complete the study setting {name} before approval.")
        typed[name] = TypeAdapter(field.rebuild_annotation()).validate_python(
            supplied[name] if name in supplied else field.get_default(call_default_factory=True)
        )
    return charter_intent(typed)


def _validate_request(intent, requirement_set):
    if (
        intent["search_mode"] not in {"fsm_config_search", "single_configuration"}
        or not intent["axes"]
    ):
        raise ValueError("This approval is available for strategy configuration searches only.")
    for key in (
        "authorized_firm_contract_ids",
        "authorized_risk_policy_ids",
        "authorized_withdrawal_policy_ids",
        "source_artifact_ids",
    ):
        if intent[key]:
            raise ValueError("This approval covers strategy backtesting only; remove added scopes.")
    if intent["date_policy"]["access_policy_id"] != "development_explicit_dates_before_path_v2":
        raise ValueError("Choose the permitted development dates for this strategy approval.")
    dates = tuple(intent["date_policy"]["replay_dates"])
    DevelopmentReplayPolicy(dates)  # validates warmup/cutoff before any cache path
    expected = derive_authorization_requirements(
        "full_authorized_development",
        tuple(f"strategy_profile.{axis}" for axis in sorted(intent["axes"])),
        ComputationPath(
            full_strategy_replay=True,
            feature_materialization=False,
            label_recomputation=False,
            model_refit=False,
            model_gated_sequential_replay=False,
            cost_recomputation=True,
            prop_resimulation=False,
            bootstrap_resimulation=False,
            reuse_trade_stream_hash=False,
        ),
        (),
        (),
    )
    if requirement_set != expected or {
        r.decision_key for r in requirement_set.payload.requirements
    } != set(DECISION_KEYS):
        raise ValueError("The required approval scope does not match this strategy-only plan.")
    if intent["locked_invariants_registry_sha256"] != registry_sha256():
        raise ValueError("The available settings changed. Refresh and review this study again.")
    baseline_name = intent["baseline_profile_name"]
    capability = PROFILE_CAPABILITY_REGISTRY.get(baseline_name)
    if capability is None or capability.status is not ProfileCapabilityStatus.RUNNABLE:
        raise ValueError("Choose a runnable registered baseline before approval.")
    baseline = resolve_profile_config({"profile_name": baseline_name})
    if baseline.section_config_hash != intent["baseline_section_config_hash"]:
        raise ValueError("The saved baseline configuration changed. Select it again.")
    objective = intent["objective_policy"]
    if any(
        metric != "core_replay_id" and metric not in StrategyMetrics.model_fields
        for metric in (*objective["pareto_objectives"], *objective["lexicographic_tie_breaks"])
    ):
        raise ValueError("Choose strategy objectives for this strategy-only approval.")
    axes = sorted(intent["axes"])
    count = 1
    for axis in axes:
        values = intent["axes"][axis]
        if not values or len(set(values)) != len(values):
            raise ValueError("Each searched setting needs distinct registered values.")
        count *= len(values)
    minimum = 1 if intent["search_mode"] == "single_configuration" else 2
    maximum = 1 if intent["search_mode"] == "single_configuration" else intent["max_child_count"]
    if count < minimum or count > min(maximum, intent["max_child_count"]):
        raise ValueError(
            "The configuration count must fit the search mode and limit; "
            "single_configuration requires exactly one configuration."
        )
    configurations, configs = [], {}
    for combo in product(*(intent["axes"][axis] for axis in axes)):
        selected = dict(zip(axes, combo, strict=True))
        # This inspects availability; it does not ratify any value or authorize a run.
        assert_axes_authorized(selected, require_ratified=False)
        overrides = resolve_axis_overrides(selected)
        resolved = resolve_profile_config(
            {"profile_name": baseline_name, "section_overrides": overrides}
        )
        section = canonicalize_section(resolved.section)
        if not section.runnable:
            raise ValueError("A selected combination does not resolve to a runnable strategy.")
        violations = _locked_invariant_violations(
            section, baseline.section, SEARCH_AXIS_REGISTRY_V1
        )
        if violations:
            raise ValueError(
                "A selected combination changes locked settings: " + ", ".join(violations)
            )
        cfg = IfvgCaptureConfig(section=section)
        configs[cfg.artifacts_tag()] = cfg
        configurations.append(
            {
                "axis_value_ids": selected,
                "section_overrides": overrides,
                "resolved_section_config_hash": ifvg_profile_hash(section),
                "comparison_role": "baseline"
                if all(
                    SEARCH_AXIS_REGISTRY_V1[axis].baseline_value_id == value
                    for axis, value in selected.items()
                )
                else "challenger",
            }
        )
    _check_source_commits(intent)
    return tuple(configurations), tuple(configs.values()), dates


def _provenance_candidates(store_root, dates, preparation_approval=None):
    candidates = {tuple(dates), tuple(PERMITTED_DEVELOPMENT_DATES)}
    if preparation_approval is None:
        preparation_approval = research_preparation_approval(Path(__file__).resolve().parents[6])
    if preparation_approval is not None:
        source_root, approval_id = preparation_approval
        source = load_strategy_approval(Path(source_root), approval_id)
        candidates.add(source.payload.artifact_provenance_dates)
    for artifact_id, _ in list_catalogued_envelope_ids(store_root, STORE):
        try:
            approval = load_strategy_approval(store_root, artifact_id)
        except (OSError, ValueError, PermissionError, RuntimeError):
            continue
        candidates.add(approval.payload.artifact_provenance_dates)
    return {
        allowlist_sha256(candidate): candidate
        for candidate in candidates
        if set(dates) <= set(candidate)
    }


def _cache_review(store_root, configs, dates, preparation_approval=None):
    candidates = _provenance_candidates(store_root, dates, preparation_approval)
    policy = DevelopmentReplayPolicy(dates)
    provenance = None
    stamps = []
    for cfg in configs:
        previous = None
        for day in dates:
            expected = _chained_seeds(previous) or DaySeeds(None, None, None, None)
            paths = [
                policy.resolve_source_path(day, factory)
                for factory in (cfg.bars_path, cfg.levels_path)
            ]
            if any(not path.exists() for path in paths):
                raise ValueError(
                    f"Prepared bars or levels are missing for {day}; prepare data first."
                )
            metadata = []
            for path in paths:
                policy.record_metadata_access(day)
                footer = pq.read_metadata(path)
                metadata.append(json.loads((footer.metadata or {}).get(_META_KEY, b"{}")))
            meta = metadata[0]
            if (
                meta.get("artifact_schema_version") != _ARTIFACT_SCHEMA_VERSION
                or meta.get("artifacts_tag") != cfg.artifacts_tag()
                or meta.get("anchor_policy") != HTF_ANCHOR_POLICY
                or meta.get("source_access_policy") != "explicit_allowlist_before_path_v1"
            ):
                raise ValueError(f"Prepared data for {day} has an incompatible cache stamp.")
            candidate = candidates.get(meta.get("source_allowlist_sha256"))
            if candidate is None:
                raise ValueError(
                    f"The cache creation date scope for {day} is unrecognized; "
                    "restore its reviewed preparation evidence before approval."
                )
            if provenance is not None and candidate != provenance:
                raise ValueError(
                    "The selected prepared data has inconsistent creation date scopes."
                )
            provenance = candidate
            if meta.get("seeds") != expected.meta():
                raise ValueError(
                    f"Prepared data for {day} does not continue the selected day chain."
                )
            if metadata[1] != meta:
                raise ValueError(f"Prepared bars and levels for {day} have different cache stamps.")
            previous = DayArtifacts(
                date_str=day,
                bars=[],
                level_timeline={},
                seeds=expected,
                day_hl=tuple(meta["day_hl"]) if meta.get("day_hl") else None,
                ny_hl=tuple(meta["ny_hl"]) if meta.get("ny_hl") else None,
                reader_warnings=(),
            )
            stamps.append({"day": day, "tag": cfg.artifacts_tag(), "metadata": meta})
    policy.assert_zero_forbidden_access()
    return tuple(provenance or ()), len(dates), stamps


def prepare_strategy_approval_review(
    fields: Mapping, requirement_set, store_root: Path, *, preparation_approval=None
):
    """Inspect exact plan + metadata; never create evidence or run computation."""
    intent = charter_intent(fields)
    blockers = []
    namespace_id, provenance, checked, configurations, stamps, witness = None, (), 0, (), [], None
    try:
        intent = _typed_intent(fields)
        namespace = require_store_namespace(Path(store_root), expected_class="research")
        namespace_id = namespace.store_namespace_id
        verify_complete_owner_authority_chain(Path(store_root))
        witness = current_supersession_head_witness(Path(store_root)).model_dump(mode="json")
        configurations, configs, dates = _validate_request(intent, requirement_set)
        if preparation_approval is None:
            provenance, checked, stamps = _cache_review(Path(store_root), configs, dates)
        else:
            provenance, checked, stamps = _cache_review(
                Path(store_root), configs, dates, preparation_approval
            )
    except Exception as error:  # noqa: BLE001 — read-only readiness returns a concrete blocker
        blockers.append(str(error))
    return StrategyApprovalReview(
        intent=intent,
        intent_sha256=charter_intent_hash(intent),
        artifact_provenance_dates=provenance,
        cache_checked_dates=checked,
        blockers=tuple(blockers),
        configuration_rows=configurations,
        requirement_set_id=requirement_set.requirement_set_id,
        store_namespace_id=namespace_id,
        evidence_sha256=canonical_contract_sha256({"cache_stamps": stamps, "head": witness}),
    )


def record_strategy_approval(
    review: StrategyApprovalReview,
    *,
    current_fields: Mapping,
    requirement_set,
    store_root: Path,
    author: str,
    approval_statement: str,
    approved_at: str,
    preparation_approval=None,
):
    """Record an explicitly confirmed, still-current review; never launch it."""
    if not author.strip() or not approval_statement.strip():
        raise ValueError("Enter the approving owner's name and confirm the displayed approval.")
    current = prepare_strategy_approval_review(
        current_fields,
        requirement_set,
        Path(store_root),
        preparation_approval=preparation_approval,
    )
    if review.blockers or current.blockers:
        raise ValueError("Approval is blocked: " + "; ".join(review.blockers or current.blockers))
    if (
        review.intent_sha256 != current.intent_sha256
        or review.requirement_set_id != current.requirement_set_id
        or review.store_namespace_id != current.store_namespace_id
        or review.evidence_sha256 != current.evidence_sha256
        or review.artifact_provenance_dates != current.artifact_provenance_dates
    ):
        raise ValueError("The study or reviewed evidence changed. Review and confirm it again.")
    envelope = StrategySearchApprovalEnvelope.from_payload(
        StrategySearchApprovalPayload(
            store_namespace_id=current.store_namespace_id,
            requirement_set_id=current.requirement_set_id,
            charter_intent_sha256=current.intent_sha256,
            approved_charter_json=json.dumps(current.intent, sort_keys=True),
            artifact_provenance_dates=current.artifact_provenance_dates,
            author=author.strip(),
            approved_at=approved_at,
            effective_from=approved_at,
            reviewed_evidence_refs=(
                f"strategy-plan:{current.intent_sha256}",
                f"prepared-cache-metadata:{current.evidence_sha256}",
            ),
            approval_statement=approval_statement.strip(),
        )
    )
    saved, _ = persist_strategy_approval(Path(store_root), envelope)
    append_catalog_event(
        Path(store_root),
        kind="display_name",
        artifact_id=saved.strategy_search_approval_id,
        payload={"display_name": f"Strategy approval {current.intent_sha256[:12]}"},
    )
    bundle = owner_authorization_bundle_from_store(
        Path(store_root),
        requirement_set,
        charter_intent_sha256=current.intent_sha256,
    )
    if bundle is None:
        raise PermissionError("The saved approval could not be verified; review readiness again.")
    payload = SearchCharterPayload(**current.intent, owner_authorization=bundle)
    validate_charter(payload, as_of_utc=approved_at, store_root=Path(store_root))
    return saved
