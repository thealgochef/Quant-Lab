"""Read-only strategy explanations from exact study configuration sources.

Only JSON contracts are read. A description verifies its configuration source;
it neither verifies result tables nor performs preparation or a replay.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

from strategy_core.strategies.ifvg_smc.section import ifvg_profile_hash

from ..artifact_io import load_verified_v2_configuration
from ..context_run_store import load_context_experiment_configuration
from ..profiles import resolve_profile_config
from ..search.axis_registry import (
    SEARCH_AXIS_REGISTRY_V1,
    registry_sha256,
    resolve_axis_overrides,
)
from ..search.identities import (
    CoreReplayArtifactReference,
    CoreStrategyReplayIdentity,
    SearchChildMembership,
    canonical_contract_sha256,
    canonicalize_section,
)
from ..search.orchestrator import SearchChildMembershipEnvelope
from ..search.store import load_json_sidecar, load_verified_envelope
from ..study_providers import load_charter, load_search_state
from .flows import flow_for_draft_fields
from .run_purpose import EvidenceClass, RunPurpose, resolve_draft_purpose
from .strategy_rules import RuleDescription, describe_strategy, describe_strategy_variations
from .workspace import StudySummary

__all__ = ["load_study_rules"]

# Reviewed against the installed reducer/section on 2026-09-08. The original
# parent-staleness replay has the same clean Strategy-Core head and source tree.
# These are distinct hash projections: core replay source identity includes
# repository metadata, whereas a v2 dataset records the source tree directly.
# Add historical implementations only after reviewing their executable rules.
_REVIEWED_STRATEGY_COMMIT = "a4e3303179ac6a1088aecaaa3482934cf1aec4d7"
_REVIEWED_STRATEGY_SOURCE_ID = "1b6539f29119f409ab31930d001c1448329aae8763d896fc80d3d91beb4c393c"
_REVIEWED_STRATEGY_TREE = "def4e157bddee7b705a4da0358bc883baecf8935e2f91bb2b55b49374bde3b71"


class _SourceError(ValueError):
    """A configuration cannot be safely attributed to this study."""


def _unavailable(message: str, *, is_preview: bool = False) -> RuleDescription:
    return RuleDescription(
        status="unavailable",
        preview_bullets=(),
        detail_bullets=(),
        issues=(message,),
        is_preview=is_preview,
    )


def _partial(description: RuleDescription, message: str) -> RuleDescription:
    return replace(
        description,
        status="partial" if description.detail_bullets else "unavailable",
        issues=(*description.issues, message),
    )


def _draft_axes(draft) -> dict[str, tuple[str, ...]]:
    validation = draft.step_payload("validation")
    purpose = resolve_draft_purpose(
        draft.purpose_annotation, run_scope=validation.get("run_scope")
    ).purpose
    if purpose is not None:
        evidence = (
            EvidenceClass.SYNTHETIC_FIXTURE
            if purpose is RunPurpose.IMPLEMENTATION_VERIFICATION
            and validation.get("evidence_class") != "real"
            else EvidenceClass.REAL
        )
        flow = flow_for_draft_fields(
            {**draft.step_payload("objective"), "mode_id": draft.mode_id},
            purpose=purpose,
            evidence_class=evidence,
        )
        if not flow.includes("search_space"):
            return {}
    result = {}
    selections = draft.step_payload("search_space").get("axis_selections") or {}
    for axis, values in selections.items():
        spec = SEARCH_AXIS_REGISTRY_V1.get(axis)
        if spec is None:
            raise _SourceError("A configured strategy variation is no longer recognized.")
        challengers = tuple(value for value in values if value != spec.baseline_value_id)
        if challengers:
            result[axis] = tuple(dict.fromkeys((spec.baseline_value_id, *challengers)))
    return result


def _combinations(axes: Mapping[str, Any]) -> list[dict[str, str]]:
    keys = sorted(axes)
    count = 1
    for key in keys:
        values = axes[key]
        if not isinstance(values, (list, tuple)) or not values:
            raise _SourceError("A configured strategy variation is incomplete.")
        count *= len(values)
    if count > 256:
        raise _SourceError("The configured variations exceed the supported study size.")
    return [dict(zip(keys, values, strict=True)) for values in product(*(axes[k] for k in keys))]


def _configuration(base: Mapping[str, Any], selection: Mapping[str, str]) -> dict[str, Any]:
    # No validation/default application to a historical effective section.
    return {**base, **resolve_axis_overrides(selection)}


def _describe_configurations(base, configurations, *, metadata=None) -> RuleDescription:
    def content_key(section):
        return canonical_contract_sha256(
            {
                key: value
                for key, value in section.items()
                if key not in {"profile_name", "qualification_mode"}
            }
        )

    seen = {content_key(base)}
    alternatives = []
    for name, section in configurations:
        key = content_key(section)
        if key not in seen:
            alternatives.append((name, section))
            seen.add(key)
    if not alternatives:
        return describe_strategy(base, semantic_metadata=metadata)
    return describe_strategy_variations(base, alternatives, semantic_metadata=metadata)


def _draft_rules(study: StudySummary) -> RuleDescription:
    draft = study.draft
    name = draft.step_payload("baseline").get("baseline_profile_name") if draft else None
    if not name:
        return RuleDescription(
            status="unconfigured",
            preview_bullets=(),
            detail_bullets=(),
            issues=("Choose a strategy configuration to see its rules.",),
            is_preview=True,
        )
    base = resolve_profile_config({"profile_name": name}).effective_config
    description = describe_strategy(base)
    try:
        combinations = _combinations(_draft_axes(draft))
        configurations = [
            (f"Configuration {index + 1}", _configuration(base, selection))
            for index, selection in enumerate(combinations)
        ]
        description = _describe_configurations(base, configurations)
    except (ValueError, TypeError, KeyError):
        description = _partial(
            description,
            "These are the baseline rules. Some draft variations could not be explained.",
        )
    return replace(description, is_preview=True)


def _study_children(study: StudySummary, roots, charter_id: str) -> tuple[Mapping, ...]:
    state = study.state if study.kind in {"search", "pipeline"} else None
    if state is None and roots.get("state_root"):
        state = load_search_state(Path(roots["state_root"]), charter_id)
    if not state:
        return ()
    if study.kind == "pipeline" and (
        state.get("pipeline_semantic_id", study.key) != study.key
        or state.get("search_charter_id", charter_id) != charter_id
    ):
        raise _SourceError("The saved workflow progress belongs to another study.")
    if state.get("search_id", charter_id) != charter_id:
        raise _SourceError("The saved progress record belongs to another study.")
    children = state.get("children", ())
    if not isinstance(children, (list, tuple)) or any(
        not isinstance(row, Mapping) for row in children
    ):
        raise _SourceError("The study's saved configuration list is unreadable.")
    return tuple(children)


def _saved_child(root: Path, charter, row: Mapping) -> tuple[dict, dict]:
    selection = dict(row["axis_value_ids"])
    axes = charter.payload.axes
    if set(selection) != set(axes) or any(
        value not in axes[key] for key, value in selection.items()
    ):
        raise _SourceError("The selected configuration is outside this study's saved choices.")
    membership = SearchChildMembershipEnvelope.from_payload(
        SearchChildMembership(
            parent_search_id=charter.search_id,
            child_ordinal=row["ordinal"],
            axis_value_ids=selection,
            core_replay_id=row["core_replay_id"],
            comparison_role=row["comparison_role"],
        )
    )
    load_verified_envelope(
        root, "memberships", membership.membership_id, SearchChildMembershipEnvelope
    )
    core = load_verified_envelope(
        root, "core_replays", row["core_replay_id"], CoreStrategyReplayIdentity
    )
    if (
        core.payload.strategy_core_source_identity != _REVIEWED_STRATEGY_SOURCE_ID
        or core.payload.strategy_core_commit != _REVIEWED_STRATEGY_COMMIT
    ):
        raise _SourceError(
            "This result uses a strategy implementation whose rules are not yet supported."
        )
    reference = CoreReplayArtifactReference.model_validate(
        load_json_sidecar(root, "core_replays", core.core_replay_id, "artifact_reference.json")
    )
    if reference.core_replay_id != core.core_replay_id:
        raise _SourceError("The saved configuration reference belongs to another result.")
    if core.payload.record_schema_version != 2 or core.payload.capture_schema_version != 2:
        raise _SourceError("This result uses a strategy version whose rules are not yet supported.")
    section = load_verified_v2_configuration(
        root / "v2_datasets",
        reference.v2_dataset_artifact_id,
        expected_manifest_hash=reference.manifest_payload_sha256,
        expected_profile_hash=core.payload.resolved_section_config_hash,
        supported_strategy_source_trees={_REVIEWED_STRATEGY_TREE},
    )
    if (
        section.get("resolver_policy") != core.payload.resolver_policy
        or section.get("anchor_policy") != core.payload.anchor_policy
    ):
        raise _SourceError("The saved rules disagree with the result's execution settings.")
    return section, {
        "strategy_id": "ifvg_smc",
        "strategy_version": "2",
        "resolver_policy": core.payload.resolver_policy,
    }


def _frozen_rules(study: StudySummary, roots, selected_core_replay_id) -> RuleDescription:
    if study.store_root is None or not study.charter_id:
        raise _SourceError("The frozen strategy settings could not be located.")
    root = Path(study.store_root)
    charter = load_charter(root, study.charter_id)
    if charter is None:
        raise _SourceError("The frozen strategy settings are unavailable.")
    children = _study_children(study, roots, charter.search_id)
    if selected_core_replay_id is not None:
        matching = [row for row in children if row.get("core_replay_id") == selected_core_replay_id]
        if len(matching) != 1:
            raise _SourceError("The selected result could not be linked to this study.")
        section, metadata = _saved_child(root, charter, matching[0])
        return describe_strategy(section, semantic_metadata=metadata)

    baseline_rows = [row for row in children if row.get("comparison_role") == "baseline"]
    if len(baseline_rows) > 1:
        raise _SourceError("The study has conflicting saved baseline references.")
    materialized = {"completed", "reused"}
    baseline_row = baseline_rows[0] if baseline_rows else None
    metadata = None
    if baseline_row and baseline_row.get("state") in materialized:
        base, metadata = _saved_child(root, charter, baseline_row)
    else:
        if charter.payload.strategy_core_commit != _REVIEWED_STRATEGY_COMMIT:
            raise _SourceError(
                "The frozen strategy uses an implementation whose rules are not yet supported."
            )
        resolved = canonicalize_section(
            resolve_profile_config({"profile_name": charter.payload.baseline_profile_name}).section
        )
        if ifvg_profile_hash(resolved) != charter.payload.baseline_section_config_hash:
            raise _SourceError("The baseline settings have changed since this study was frozen.")
        base = resolved.model_dump(mode="json")
    description = describe_strategy(base, semantic_metadata=metadata)
    if not charter.payload.axes:
        return description
    combinations = _combinations(charter.payload.axes)
    current_registry = charter.payload.locked_invariants_registry_sha256 == registry_sha256()
    configurations = []
    try:
        for index, selection in enumerate(combinations):
            matching = [row for row in children if dict(row.get("axis_value_ids", {})) == selection]
            if len(matching) > 1:
                raise _SourceError("The study has conflicting saved variation references.")
            row = matching[0] if matching else None
            if row and row.get("state") in materialized:
                section, _ = _saved_child(root, charter, row)
            elif current_registry:
                # Reconstruct only while the original baseline and registry still match.
                if charter.payload.strategy_core_commit != _REVIEWED_STRATEGY_COMMIT:
                    raise _SourceError("The original strategy implementation is not yet supported.")
                original = canonicalize_section(
                    resolve_profile_config(
                        {"profile_name": charter.payload.baseline_profile_name}
                    ).section
                )
                if ifvg_profile_hash(original) != charter.payload.baseline_section_config_hash:
                    raise _SourceError("The original baseline is no longer available.")
                section = _configuration(original.model_dump(mode="json"), selection)
            else:
                raise _SourceError("The saved variation definitions are no longer available.")
            label = (
                "Baseline"
                if row and row.get("comparison_role") == "baseline"
                else f"Configuration {index + 1}"
            )
            configurations.append((label, section))
        return _describe_configurations(base, configurations, metadata=metadata)
    except (ValueError, TypeError, KeyError, OSError):
        return _partial(
            description,
            "These rules describe the baseline only. Some study variations could not be verified.",
        )


def _context_rules(study: StudySummary, roots) -> RuleDescription:
    config = load_context_experiment_configuration(
        study.key, base_dir=Path(roots["context_run_root"])
    )
    ref = config.dataset.artifact_pair.v2
    root = Path(roots.get("v2_root") or Path(roots["repo_root"]) / "data/ifvg_datasets/v2")
    section = load_verified_v2_configuration(
        root,
        ref.artifact_id,
        expected_manifest_hash=ref.manifest_payload_sha256,
        expected_profile_hash=ref.profile_hash,
        supported_strategy_source_trees={_REVIEWED_STRATEGY_TREE},
    )
    return describe_strategy(section)


def load_study_rules(
    study: StudySummary, roots: Mapping[str, Any], selected_core_replay_id: str | None = None
) -> RuleDescription:
    """Describe saved settings without writing artifacts or reading market data."""
    try:
        if study.kind == "context":
            return _context_rules(study, roots)
        if study.charter_id or (study.draft and study.draft.frozen_search_id):
            if not study.charter_id:
                study = replace(study, charter_id=study.draft.frozen_search_id)
            return _frozen_rules(study, roots, selected_core_replay_id)
        if study.kind == "draft" and selected_core_replay_id is None:
            return _draft_rules(study)
        return _unavailable("The strategy settings for this study are unavailable.")
    except _SourceError as error:
        return _unavailable(str(error), is_preview=study.kind == "draft" and not study.charter_id)
    except (ValueError, TypeError, KeyError, OSError):
        return _unavailable(
            "The saved strategy settings could not be verified. "
            "Other study details remain available.",
            is_preview=study.kind == "draft" and not study.charter_id,
        )
