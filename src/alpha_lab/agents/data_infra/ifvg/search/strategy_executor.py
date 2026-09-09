"""Owner-approved strategy searches over trusted development day artifacts.

Factory construction verifies authority only. Source reads and sequential
replays occur exclusively in the detached worker after the user's Run action.
No verification results, seeds, or synthetic authority are manufactured.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from strategy_core.strategies.ifvg_smc.records import IFVG_RECORD_SCHEMA_VERSION

from ..config import IfvgCaptureConfig
from ..contracts import IFVG_CAPTURE_SCHEMA_VERSION
from ..dataset import _chained_seeds
from ..day_artifacts import DaySeeds, load_day_artifacts
from ..development_access import DevelopmentReplayPolicy
from ..manifest import file_sha256, read_repository_state
from ..profiles import resolve_profile_config
from .axis_registry import AXIS_VALUE_REGISTRY_V1
from .charter import validate_charter
from .child_replay import ArtifactProvenanceReadAdapter, build_child_companions, run_child_replay
from .executors import REPO_ROOT
from .identities import (
    QL_REPLAY_SOURCE_SCOPE,
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
    ReplayAccessAuthorizationRef,
    ReplayDayArtifactRef,
    build_replay_input_bundle,
    canonical_contract_sha256,
    canonicalize_section,
    quant_lab_replay_source_identity,
    strategy_core_source_identity,
)
from .store import save_or_reuse_envelope
from .strategy_approval import DECISION_ID, load_strategy_approval, ratified_registry_for_charter


def search_strategy_development_entry(charter_envelope, *, store_root) -> dict:
    root = Path(store_root)
    payload = charter_envelope.payload
    refs = getattr(payload.owner_authorization, "decision_refs", {})
    if not any(ref.decision_id == DECISION_ID for ref in refs.values()):
        raise PermissionError("an exact saved strategy-search approval is required")
    validate_charter(payload, as_of_utc=datetime.now(UTC).isoformat(), store_root=root)
    ratified_registry_for_charter(payload, root, AXIS_VALUE_REGISTRY_V1)
    approval_ref = next(ref for ref in refs.values() if ref.decision_id == DECISION_ID)
    approval = load_strategy_approval(root, approval_ref.decision_artifact_id)
    dates = tuple(payload.date_policy.replay_dates)
    # This validates the ten-date warmup and the cutoff before any source path.
    DevelopmentReplayPolicy(dates)
    contexts = {}

    def artifact_policy():
        # Cache provenance can name a wider writing allowlist. All actual I/O
        # remains confined to the approved replay dates by the inner policy.
        return ArtifactProvenanceReadAdapter(
            DevelopmentReplayPolicy(dates),
            artifact_provenance_dates=approval.payload.artifact_provenance_dates,
        )

    def identity_resolver(spec):
        key = spec.resolved_section_config_hash
        if key in contexts:
            return contexts[key][3]
        resolved = resolve_profile_config(
            {
                "profile_name": payload.baseline_profile_name,
                "section_overrides": dict(spec.section_overrides),
            }
        )
        section = canonicalize_section(resolved.section)
        resolved = replace(
            resolved,
            section=section,
            section_config_hash=key,
            effective_config=section.model_dump(mode="json"),
        )
        cfg = IfvgCaptureConfig(section=section)
        if cfg.profile_hash != key:
            raise PermissionError("child profile differs from its enumerated identity")
        policy = artifact_policy()
        previous = None
        day_refs = []
        for day in dates:
            expected = _chained_seeds(previous) or DaySeeds(None, None, None, None)
            previous = load_day_artifacts(day, cfg, expected_seeds=expected, access_policy=policy)
            if previous is None:
                raise PermissionError(f"trusted day artifacts are unavailable for {day}")
            for kind, factory in (("bars", cfg.bars_path), ("levels", cfg.levels_path)):
                path = policy.resolve_source_path(day, factory)
                policy.record_file_open(day)
                digest = file_sha256(path)
                day_refs.append(
                    ReplayDayArtifactRef(
                        trading_day=day,
                        artifact_kind=kind,
                        artifact_id=path.name,
                        manifest_payload_sha256=digest,
                        content_sha256=digest,
                    )
                )
        date_id = canonical_contract_sha256(dates)
        bundle = build_replay_input_bundle(
            authorized_date_set_id=date_id,
            source_partitions=(),
            day_artifacts=day_refs,
            source_contract_id="databento_nq_v1",
            source_schema_era_id="cached_day_artifacts_v2",
            access_authorization=ReplayAccessAuthorizationRef(
                access_policy_id=payload.date_policy.access_policy_id,
                authorized_date_set_id=date_id,
                expected_source_inventory_hash=canonical_contract_sha256(
                    [ref.model_dump(mode="json") for ref in day_refs]
                ),
            ),
        )
        sc_commit, sc_source = strategy_core_source_identity(
            repository_root=REPO_ROOT.parent / "Strategy-Core"
        )
        core = CoreStrategyReplayIdentity.from_payload(
            CoreStrategyReplayPayload(
                replay_input_bundle_id=bundle.replay_input_bundle_id,
                quant_lab_replay_source_identity=quant_lab_replay_source_identity(
                    repository_root=REPO_ROOT
                ),
                strategy_core_commit=sc_commit,
                strategy_core_source_identity=sc_source,
                resolved_section_config_hash=key,
                canonical_profile_id=spec.canonical_profile_id,
                warmup_seed_identity=canonical_contract_sha256(
                    {
                        "start": "cold",
                        "warmup_dates": list(payload.date_policy.warmup_dates),
                        "policy": payload.date_policy.access_policy_id,
                    }
                ),
                anchor_policy=section.anchor_policy,
                capture_schema_version=IFVG_CAPTURE_SCHEMA_VERSION,
                record_schema_version=IFVG_RECORD_SCHEMA_VERSION,
            )
        )
        policy.assert_zero_forbidden_access()
        contexts[key] = (resolved, cfg, bundle, core)
        return core

    def child_runner(*, spec, core_replay_id):
        # Recheck approval at the safe child boundary, before replay.
        validate_charter(payload, as_of_utc=datetime.now(UTC).isoformat(), store_root=root)
        core = identity_resolver(spec)
        if core.core_replay_id != core_replay_id:
            raise PermissionError("requested child identity does not match its inputs")
        resolved, cfg, bundle, _core = contexts[spec.resolved_section_config_hash]
        result = run_child_replay(
            dates=dates,
            cfg=cfg,
            resolved_profile=resolved,
            access_policy_factory=artifact_policy,
            core_replay_id=core_replay_id,
            cached_artifacts_only=True,
            dual_drive=True,
        )
        # Input mutation during replay must never publish under an old identity.
        policy = DevelopmentReplayPolicy(dates)
        for ref in bundle.payload.ordered_day_artifacts:
            factory = cfg.bars_path if ref.artifact_kind == "bars" else cfg.levels_path
            path = policy.resolve_source_path(ref.trading_day, factory)
            policy.record_file_open(ref.trading_day)
            if file_sha256(path) != ref.content_sha256:
                raise PermissionError("day artifact changed during replay")
        report = build_child_companions(
            result=result,
            core=core,
            bundle=bundle,
            store_root=root,
            repo_root=REPO_ROOT,
            replay_dates=dates,
            warmup_days=len(payload.date_policy.warmup_dates),
            request_reference=charter_envelope.search_id,
            access_policy_id=payload.date_policy.access_policy_id,
            repository_states=(
                read_repository_state("quant-lab", REPO_ROOT, source_paths=QL_REPLAY_SOURCE_SCOPE),
                read_repository_state(
                    "strategy-core",
                    REPO_ROOT.parent / "Strategy-Core",
                    source_paths=("src/strategy_core",),
                ),
            ),
            authoritative_source_blob="approved_strategy_search_cached_artifacts_v1",
            cost_points=payload.cost_policy.cost_points_round_turn,
        )
        if not report["invariants_passed"]:
            raise PermissionError("child evidence failed its invariant checks")
        save_or_reuse_envelope(root, "replay_input_bundles", bundle)
        save_or_reuse_envelope(
            root,
            "core_replays",
            core,
            extra_files={
                "artifact_reference.json": json.dumps(
                    report["core_replay_artifact_reference"], sort_keys=True
                ).encode()
            },
        )
        return result

    return {"identity_resolver": identity_resolver, "child_runner": child_runner}
