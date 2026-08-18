"""Baseline-capable child replay worker (§3.3) + seed snapshots + neutrality.

Worker flow: resolve profile → construct the validated section → authorize
inputs → run the sequential v2 capture → assert zero forbidden access →
publish atomically → reload and verify, keyed by ``core_replay_id``.

The real five-day baseline vertical slice (TEST_MATRIX §1 Path A) is
implemented here end-to-end but **never executes without an owner-approved
``VerificationAuthorizationRef``** — validation fails before any source path
is constructed. Seed snapshots resolve the R0→R1 mid-chain-start item: the
snapshot carries the entering reducer seed *and* the entering ``DaySeeds``
expectations for the first replayed day (seeds are profile-bound; a
profile/seed mismatch is refused before any source read).
"""

from __future__ import annotations

import io
import pickle
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
from pydantic import Field
from strategy_core.strategies.ifvg_smc.records import IFVG_RECORD_SCHEMA_VERSION
from strategy_core.strategies.ifvg_smc.state import IfvgDaySeed, seed_hash

from ..config import IfvgCaptureConfig
from ..contracts import IFVG_CAPTURE_SCHEMA_VERSION, RecordTable
from ..dataset import ChainStart, V2CaptureResult, build_ifvg_v2_capture, table_content_hash
from ..day_artifacts import DaySeeds
from ..development_access import VerificationReplayPolicy
from ..manifest import file_sha256
from ..profiles import ResolvedProfileConfig, resolve_profile_config
from .authorization import VerificationAuthorizationRef
from .identities import (
    SHA256_PATTERN,
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    ReplayAccessAuthorizationRef,
    ReplayDayArtifactRef,
    ReplaySourcePartitionRef,
    build_replay_input_bundle,
    canonical_contract_sha256,
    quant_lab_replay_source_identity,
    register_identity_pair,
    strategy_core_source_identity,
)
from .store import (
    SEARCH_TEST_STORE_ROOT,
    load_sidecar_bytes,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from .verification import (
    VERIFICATION_POLICY_ID,
    VerificationRunEnvelope,
    VerificationRunValidationError,
    evaluate_control_flow_gates,
    register_program_allowlist,
    validate_verification_run,
    verification_report_stamps,
)

__all__ = [
    "SeedSnapshotPayload",
    "SeedSnapshotEnvelope",
    "DaySeedsRecord",
    "save_seed_snapshot",
    "load_seed_snapshot",
    "SeedSnapshotError",
    "ChildAuditNeutralityReport",
    "build_neutrality_report",
    "ChildReplayResult",
    "run_child_replay",
    "run_baseline_verification_slice",
    "ArtifactProvenanceReadAdapter",
    "CORE_TABLE_NAMES",
]

_SEED_SIDECAR = "seed.pickle"

CORE_TABLE_NAMES: tuple[str, ...] = tuple(table.value for table in RecordTable)


class DaySeedsRecord(FrozenContract):
    """JSON-shaped :class:`DaySeeds` (the FIRST replayed day's expectations)."""

    prev_day: str | None
    prev_full_hl: tuple[int, int] | None
    prev_ny_day: str | None
    prev_ny_hl: tuple[int, int] | None

    @classmethod
    def from_day_seeds(cls, seeds: DaySeeds) -> DaySeedsRecord:
        return cls(
            prev_day=seeds.prev_day.isoformat() if seeds.prev_day else None,
            prev_full_hl=tuple(seeds.prev_full_hl) if seeds.prev_full_hl else None,
            prev_ny_day=seeds.prev_ny_day.isoformat() if seeds.prev_ny_day else None,
            prev_ny_hl=tuple(seeds.prev_ny_hl) if seeds.prev_ny_hl else None,
        )

    def to_day_seeds(self) -> DaySeeds:
        return DaySeeds(
            prev_day=date.fromisoformat(self.prev_day) if self.prev_day else None,
            prev_full_hl=tuple(self.prev_full_hl) if self.prev_full_hl else None,
            prev_ny_day=date.fromisoformat(self.prev_ny_day) if self.prev_ny_day else None,
            prev_ny_hl=tuple(self.prev_ny_hl) if self.prev_ny_hl else None,
        )


class SeedSnapshotPayload(FrozenContract):
    """Identity of one verified, profile-bound mid-chain seed snapshot."""

    profile_name: str
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    seed_schema_version: int
    seed_hash: str
    snapshot_through_day: str
    first_replay_day: str
    entering_day_seeds: DaySeedsRecord
    chain_policy_id: str
    chain_date_count: int = Field(ge=1)
    strategy_core_commit: str


class SeedSnapshotEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "seed_snapshot_id"

    seed_snapshot_id: str = Field(pattern=SHA256_PATTERN)
    payload: SeedSnapshotPayload


class SeedSnapshotError(PermissionError):
    """A seed snapshot failed verification or profile binding."""


def save_seed_snapshot(
    root: Path,
    payload: SeedSnapshotPayload,
    seed: IfvgDaySeed,
) -> SeedSnapshotEnvelope:
    """Persist the snapshot immutably: envelope + hashed seed sidecar."""

    if seed_hash(seed) != payload.seed_hash:
        raise SeedSnapshotError("seed does not hash to the snapshot payload's seed_hash")
    if seed.profile_hash != payload.resolved_section_config_hash:
        raise SeedSnapshotError("seed profile_hash does not match the snapshot payload")
    envelope = SeedSnapshotEnvelope.from_payload(payload)
    save_or_reuse_envelope(
        root,
        "seed_snapshots",
        envelope,
        extra_files={_SEED_SIDECAR: pickle.dumps(seed)},
    )
    return envelope


class _SeedSandboxUnpickler(pickle.Unpickler):
    """Restricted unpickler for seed sidecars: Strategy-Core state graphs only.

    Integrity is already chained (sidecar sha256 → manifest → envelope id),
    but deserialization gadgets are avoided outright by refusing any class
    outside the seed's legitimate type universe.
    """

    _SAFE_BUILTINS = frozenset({"set", "frozenset", "complex", "range", "slice", "bytearray"})

    def find_class(self, module: str, name: str):
        if module.startswith("strategy_core.") or module in (
            "datetime",
            "uuid",
            "collections",
            "enum",
        ):
            return super().find_class(module, name)
        if module == "builtins" and name in self._SAFE_BUILTINS:
            return super().find_class(module, name)
        raise SeedSnapshotError(
            f"seed snapshot pickle references disallowed type {module}.{name}"
        )


def _load_seed_bytes(data: bytes) -> IfvgDaySeed:
    return _SeedSandboxUnpickler(io.BytesIO(data)).load()


def load_seed_snapshot(
    root: Path,
    seed_snapshot_id: str,
    *,
    expected_section_config_hash: str,
) -> tuple[SeedSnapshotEnvelope, ChainStart]:
    """Load + verify a snapshot; refuse a profile mismatch BEFORE any source read."""

    envelope = load_verified_envelope(
        root, "seed_snapshots", seed_snapshot_id, SeedSnapshotEnvelope
    )
    if envelope.payload.resolved_section_config_hash != expected_section_config_hash:
        raise SeedSnapshotError(
            "seed snapshot is bound to a different profile section hash; refused "
            "before any source read (seeds are profile-bound)"
        )
    seed = _load_seed_bytes(
        load_sidecar_bytes(root, "seed_snapshots", seed_snapshot_id, _SEED_SIDECAR)
    )
    if seed_hash(seed) != envelope.payload.seed_hash:
        raise SeedSnapshotError("stored seed no longer hashes to the snapshot payload")
    if seed.profile_hash != expected_section_config_hash:
        raise SeedSnapshotError("stored seed profile_hash mismatch; refused")
    return envelope, ChainStart(
        seed=seed, day_seeds=envelope.payload.entering_day_seeds.to_day_seeds()
    )


class ArtifactProvenanceReadAdapter:
    """Read dev-chain day-artifact caches under a STRICT ≤5-day authorization.

    The existing cache trust stamp pins the *writing* policy's allowlist hash
    (``day_artifacts.write_day_artifacts``), so a verification run cannot load
    those caches with its own 5-day allowlist directly (verified codebase
    fact). This adapter exposes the recorded artifact-provenance allowlist for
    the stamp comparison ONLY, while delegating every date authorization,
    path construction, and audit event to the wrapped
    :class:`VerificationReplayPolicy` — a 6th or off-allowlist date is still
    refused before any path exists, and the runtime audit lives on the strict
    inner policy. Purely additive: no existing loader semantics change.
    """

    _ifvg_development_policy_v2 = True  # admits dev-window artifact reads only;
    # every actual authorization below delegates to the strict inner policy.

    def __init__(
        self,
        inner: VerificationReplayPolicy,
        *,
        artifact_provenance_dates: tuple[str, ...],
    ) -> None:
        ordered = tuple(str(day) for day in artifact_provenance_dates)
        if any(day > "2026-06-10" for day in ordered):
            raise PermissionError(
                "artifact provenance dates may never extend past the development cutoff"
            )
        self.inner = inner
        self.allowlist = frozenset(ordered)
        self.audit = inner.audit

    def authorize_date(self, day: str) -> None:
        self.inner.authorize_date(day)

    def authorize_dates(self, days):
        return self.inner.authorize_dates(days)

    def resolve_source_path(self, day: str, path_factory):
        return self.inner.resolve_source_path(day, path_factory)

    def record_metadata_access(self, day: str) -> None:
        self.inner.record_metadata_access(day)

    def record_file_open(self, day: str, *, rows: int = 0) -> None:
        self.inner.record_file_open(day, rows=rows)

    def record_rows_read(self, day: str, *, rows: int) -> None:
        self.inner.record_rows_read(day, rows=rows)

    def assert_zero_forbidden_access(self) -> None:
        self.inner.assert_zero_forbidden_access()

    def audit_dict(self) -> dict:
        return self.inner.audit_dict()


# ─────────────────────────────────────────────────────────────────────────────
# audit neutrality (§3.3)
# ─────────────────────────────────────────────────────────────────────────────


class ChildAuditNeutralityReport(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    mechanism: Literal["dual_drive_ab_v1", "single_drive_side_channel_v1"]
    audit_disabled_core_table_hashes: ImmutableMap[str, str] | None
    audit_enabled_core_table_hashes: ImmutableMap[str, str] | None
    tables_equal: bool | None
    mechanism_evidence_refs: tuple[str, ...]
    core_trace_content_hash: str = Field(pattern=SHA256_PATTERN)
    audit_stamp_referential_integrity: bool
    passed: bool


def _core_table_hashes(tables: dict[RecordTable, pd.DataFrame]) -> dict[str, str]:
    return {
        table.value: table_content_hash(table, tables.get(table, pd.DataFrame()))
        for table in RecordTable
    }


def _audit_referential_integrity(
    audit_frames: dict[str, pd.DataFrame] | None,
    tables: dict[RecordTable, pd.DataFrame],
) -> bool:
    """Every audit-channel row referencing a core id must reference a real one."""

    if not audit_frames:
        return True
    setups: set[str] = set()
    lifecycle = tables.get(RecordTable.SETUP_LIFECYCLE, pd.DataFrame())
    if not lifecycle.empty and "envelope_setup_id" in lifecycle:
        setups = set(lifecycle["envelope_setup_id"].dropna().astype(str))
    for frame in audit_frames.values():
        if frame is None or frame.empty:
            continue
        for column in ("envelope_setup_id", "setup_id"):
            if column in frame:
                referenced = {
                    value
                    for value in frame[column].dropna().astype(str)
                    if value  # empty string = an unattached audit row, not a ref
                }
                if referenced and setups and not referenced <= setups:
                    return False
    return True


def build_neutrality_report(
    *,
    core_replay_id: str,
    disabled: V2CaptureResult,
    enabled: V2CaptureResult,
    mechanism_evidence_refs: tuple[str, ...] = (),
) -> ChildAuditNeutralityReport:
    """Dual-drive A/B proof: core tables byte-equal with the channel on/off."""

    from ..manifest import canonical_sha256  # noqa: PLC0415

    disabled_hashes = _core_table_hashes(disabled.tables)
    enabled_hashes = _core_table_hashes(enabled.tables)
    tables_equal = disabled_hashes == enabled_hashes
    integrity = _audit_referential_integrity(enabled.audit_frames, enabled.tables)
    return ChildAuditNeutralityReport(
        core_replay_id=core_replay_id,
        mechanism="dual_drive_ab_v1",
        audit_disabled_core_table_hashes=disabled_hashes,
        audit_enabled_core_table_hashes=enabled_hashes,
        tables_equal=tables_equal,
        mechanism_evidence_refs=mechanism_evidence_refs,
        core_trace_content_hash=canonical_sha256(disabled_hashes),
        audit_stamp_referential_integrity=integrity,
        passed=bool(tables_equal and integrity),
    )


# ─────────────────────────────────────────────────────────────────────────────
# child replay worker
# ─────────────────────────────────────────────────────────────────────────────


class ChildReplayResult:
    """One worker run: capture result(s) + neutrality evidence."""

    def __init__(
        self,
        *,
        resolved_profile: ResolvedProfileConfig,
        capture: V2CaptureResult,
        audit_capture: V2CaptureResult | None,
        neutrality: ChildAuditNeutralityReport | None,
        gross_trade_stream_hash: str,
    ) -> None:
        self.resolved_profile = resolved_profile
        self.capture = capture
        self.audit_capture = audit_capture
        self.neutrality = neutrality
        self.gross_trade_stream_hash = gross_trade_stream_hash


def run_child_replay(
    *,
    dates: tuple[str, ...],
    cfg: IfvgCaptureConfig,
    resolved_profile: ResolvedProfileConfig,
    access_policy_factory,
    core_replay_id: str,
    start_after_artifact: ChainStart | None = None,
    cached_artifacts_only: bool = False,
    dual_drive: bool = True,
) -> ChildReplayResult:
    """Sequential per-child replay with a per-child neutrality proof.

    ``access_policy_factory`` builds a FRESH policy per drive so runtime access
    audits stay attempt-scoped (they never enter replay identity, §1.1). Every
    drive asserts zero forbidden access before results are returned.

    The provenance read adapter is READ-ONLY: a rebuild under it would stamp
    the new cache with the wide provenance allowlist hash (write path in
    ``day_artifacts.write_day_artifacts``), so any adapter-backed run must be
    ``cached_artifacts_only=True`` — enforced here, fail-closed.
    """

    if isinstance(access_policy_factory(), ArtifactProvenanceReadAdapter) and (
        not cached_artifacts_only
    ):
        raise PermissionError(
            "the artifact-provenance read adapter is read-only: rebuilding day "
            "artifacts under it would stamp caches with the provenance "
            "allowlist hash; pass cached_artifacts_only=True"
        )
    disabled = build_ifvg_v2_capture(
        list(dates),
        cfg,
        resolved_profile,
        access_policy=access_policy_factory(),
        cached_artifacts_only=cached_artifacts_only,
        start_after_artifact=start_after_artifact,
        audit_capture_mode="disabled",
    )
    audit_capture: V2CaptureResult | None = None
    neutrality: ChildAuditNeutralityReport | None = None
    if dual_drive:
        audit_capture = build_ifvg_v2_capture(
            list(dates),
            cfg,
            resolved_profile,
            access_policy=access_policy_factory(),
            cached_artifacts_only=cached_artifacts_only,
            start_after_artifact=start_after_artifact,
            audit_capture_mode="fsm_audit_v1",
        )
        neutrality = build_neutrality_report(
            core_replay_id=core_replay_id,
            disabled=disabled,
            enabled=audit_capture,
        )
    return ChildReplayResult(
        resolved_profile=resolved_profile,
        capture=disabled,
        audit_capture=audit_capture,
        neutrality=neutrality,
        gross_trade_stream_hash=table_content_hash(
            RecordTable.EXECUTED_TRADE,
            disabled.tables.get(RecordTable.EXECUTED_TRADE, pd.DataFrame()),
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# the real baseline vertical slice (Path A) — blocked until authorized
# ─────────────────────────────────────────────────────────────────────────────


def _assemble_slice_identity(
    *,
    run: VerificationRunEnvelope,
    resolved: ResolvedProfileConfig,
    cfg: IfvgCaptureConfig,
    hashing_policy,
    source_partition_refs: tuple[ReplaySourcePartitionRef, ...],
    source_schema_era_id: str,
    ql_source_identity: str | None,
    sc_identity: tuple[str, str] | None,
    strategy_core_root: Path | None,
    repo_root: Path,
):
    """Content-address the slice's exact inputs and mint its core identity.

    Day-artifact refs hash the actual cached bar/level files through the
    AUTHORIZED policy (path construction + file opens recorded); source
    partition refs are metadata supplied from the accepted manifests (P0-A:
    bundle assembly reuses already-computed hashes — no raw-source reads).
    """

    day_refs: list[ReplayDayArtifactRef] = []
    for day in run.payload.allowlist:
        for kind, factory in (("bars", cfg.bars_path), ("levels", cfg.levels_path)):
            path = hashing_policy.resolve_source_path(day, factory)
            hashing_policy.record_file_open(day)
            digest = file_sha256(path)
            day_refs.append(
                ReplayDayArtifactRef(
                    trading_day=day,
                    artifact_kind=kind,  # type: ignore[arg-type]
                    artifact_id=path.name,
                    manifest_payload_sha256=digest,
                    content_sha256=digest,
                )
            )
    access_authorization = ReplayAccessAuthorizationRef(
        access_policy_id=VERIFICATION_POLICY_ID,
        authorized_date_set_id=run.payload.allowlist_hash,
        expected_source_inventory_hash=canonical_contract_sha256(
            {
                "dates": list(run.payload.allowlist),
                "partitions": [
                    [ref.source_partition_id, ref.content_sha256]
                    for ref in source_partition_refs
                ],
                "policy": VERIFICATION_POLICY_ID,
            }
        ),
    )
    bundle = build_replay_input_bundle(
        authorized_date_set_id=run.payload.allowlist_hash,
        source_partitions=source_partition_refs,
        day_artifacts=day_refs,
        source_contract_id="databento_nq_v1",
        source_schema_era_id=source_schema_era_id,
        access_authorization=access_authorization,
    )
    ql_identity = ql_source_identity or quant_lab_replay_source_identity(
        repository_root=Path(repo_root)
    )
    if sc_identity is None:
        sc_root = strategy_core_root or Path(repo_root).parent / "Strategy-Core"
        sc_identity = strategy_core_source_identity(repository_root=sc_root)
    core_payload = CoreStrategyReplayPayload(
        replay_input_bundle_id=bundle.replay_input_bundle_id,
        quant_lab_replay_source_identity=ql_identity,
        strategy_core_commit=sc_identity[0],
        strategy_core_source_identity=sc_identity[1],
        resolved_section_config_hash=resolved.section_config_hash,
        canonical_profile_id=run.payload.baseline_profile_id,
        warmup_seed_identity=f"seed_snapshot:{run.payload.seed_snapshot_id}",
        anchor_policy=resolved.section.anchor_policy,
        capture_schema_version=IFVG_CAPTURE_SCHEMA_VERSION,
        record_schema_version=IFVG_RECORD_SCHEMA_VERSION,
    )
    return bundle, CoreStrategyReplayIdentity.from_payload(core_payload)


def run_baseline_verification_slice(
    *,
    run: VerificationRunEnvelope,
    authorization: object,
    pipeline_semantic_id: str,
    store_root: Path,
    repo_root: Path,
    data_dir: Path | None = None,
    artifact_provenance_dates: tuple[str, ...] = (),
    source_partition_refs: tuple[ReplaySourcePartitionRef, ...] = (),
    source_schema_era_id: str = "mbp1_era_v1",
    ql_source_identity: str | None = None,
    sc_identity: tuple[str, str] | None = None,
    strategy_core_root: Path | None = None,
    companion_builders=None,
    replay_runner=run_child_replay,
) -> dict:
    """Execute Path A under a real ``VerificationAuthorizationRef`` ONLY.

    Sequence: fail-before-path validation → canonical-namespace binding →
    program-allowlist marker → seed-snapshot load (profile-bound +
    allowlist-continuity refusal) → input-bundle assembly + core-identity
    minting (authorized hashing of the exact cached inputs) → dual-drive
    replay on cached artifacts → neutrality keyed by the REAL core replay id →
    core-envelope publication (save→reload→reuse via the immutable store) →
    control-flow gates → stamped summary. The audit/chart companion builders
    and the exact ``setup_id`` verifier jump are R2 seams (injected via
    ``companion_builders``); their gates evaluate open until they run.
    Research gates are NEVER applied here.
    """

    resolved = resolve_profile_config(
        {"profile_name": run.payload.baseline_profile_id}
    )
    if resolved.section_config_hash != run.payload.baseline_section_config_hash:
        raise SeedSnapshotError(
            "baseline profile section hash does not match the verification run"
        )
    # 1. Fail-before-path: authorization, allowlist, seed, coverage, pipeline.
    validate_verification_run(
        run,
        expected_pipeline_semantic_id=pipeline_semantic_id,
        expected_baseline_profile_id=run.payload.baseline_profile_id,
        expected_baseline_section_config_hash=resolved.section_config_hash,
        expected_seed_snapshot_id=run.payload.seed_snapshot_id,
        authorization=authorization,
    )
    if not isinstance(authorization, VerificationAuthorizationRef):
        raise VerificationRunValidationError(
            "the real verification slice requires a VerificationAuthorizationRef"
        )
    # 2. The output namespace is the Literal search_test/v1 — bind the store
    #    root to it under repo_root so no run can retarget the research store.
    canonical_store = (Path(repo_root) / SEARCH_TEST_STORE_ROOT).resolve()
    if Path(store_root).resolve() != canonical_store:
        raise VerificationRunValidationError(
            "verification output namespace is fixed to search_test/v1; "
            f"got store_root {Path(store_root).resolve()}"
        )
    # 3. One canonical allowlist for the whole program (marker-enforced).
    from .verification import VerificationDataPolicy  # noqa: PLC0415

    policy_record = VerificationDataPolicy.from_allowlist(run.payload.allowlist)
    register_program_allowlist(Path(store_root), policy_record)
    # 4. Seed snapshot (profile-bound; continuity with the allowlist asserted;
    #    refused before any source read).
    snapshot, chain_start = load_seed_snapshot(
        Path(store_root),
        run.payload.seed_snapshot_id,
        expected_section_config_hash=resolved.section_config_hash,
    )
    if snapshot.payload.first_replay_day != run.payload.allowlist[0]:
        raise SeedSnapshotError(
            "seed snapshot is not continuous with the approved allowlist: "
            f"snapshot first_replay_day={snapshot.payload.first_replay_day!r} "
            f"!= allowlist start {run.payload.allowlist[0]!r}"
        )
    # 5. Content-address the exact inputs and mint the core replay identity.
    base_cfg = IfvgCaptureConfig()
    cfg = replace(
        base_cfg,
        section=resolved.section,
        data_dir=Path(data_dir) if data_dir is not None else base_cfg.data_dir,
    )

    def _policy():
        inner = VerificationReplayPolicy(run.payload.allowlist)
        if not artifact_provenance_dates:
            return inner
        return ArtifactProvenanceReadAdapter(
            inner, artifact_provenance_dates=artifact_provenance_dates
        )

    bundle, core = _assemble_slice_identity(
        run=run,
        resolved=resolved,
        cfg=cfg,
        hashing_policy=_policy(),
        source_partition_refs=source_partition_refs,
        source_schema_era_id=source_schema_era_id,
        ql_source_identity=ql_source_identity,
        sc_identity=sc_identity,
        strategy_core_root=strategy_core_root,
        repo_root=Path(repo_root),
    )
    # 6. Dual-drive sequential replay on cached day artifacts only.
    result = replay_runner(
        dates=run.payload.allowlist,
        cfg=cfg,
        resolved_profile=resolved,
        access_policy_factory=_policy,
        core_replay_id=core.core_replay_id,
        start_after_artifact=chain_start,
        cached_artifacts_only=True,
        dual_drive=True,
    )
    for capture in (result.capture, result.audit_capture):
        if capture is not None:
            capture.access_policy.assert_zero_forbidden_access()
    # 7. Publish the replay identity immutably (save → reload → reuse).
    _, bundle_reused = save_or_reuse_envelope(
        Path(store_root), "replay_input_bundles", bundle
    )
    _, core_reused = save_or_reuse_envelope(Path(store_root), "core_replays", core)
    # 8. Companions (R2 machinery, injected): audit/chart artifacts + the
    #    exact verifier link. Their gates stay open until they run.
    companion_report = None
    if companion_builders is not None:
        companion_report = companion_builders(
            result=result,
            run=run,
            core=core,
            bundle=bundle,
            store_root=Path(store_root),
            repo_root=Path(repo_root),
        )
    gates = evaluate_control_flow_gates(
        {
            "replay_completed": True,
            # the v2 invariant audit rides the companion (evaluation) build —
            # honest: open until it runs.
            "invariants_passed": bool(
                companion_report and companion_report.get("invariants_passed")
            ),
            "artifacts_published_and_reloaded": bool(
                bundle_reused is not None
                and core_reused is not None
                and companion_report
                and companion_report.get("published")
            ),
            "neutrality_passed": bool(result.neutrality and result.neutrality.passed),
            "verifier_link_resolves": bool(
                companion_report and companion_report.get("verifier_link_resolves")
            ),
            "zero_forbidden_counters": True,
        }
    )
    return {
        "core_replay_id": core.core_replay_id,
        "replay_input_bundle_id": bundle.replay_input_bundle_id,
        "core_replay_reused": core_reused,
        "replay_input_bundle_reused": bundle_reused,
        "gate_policy": gates.model_dump(mode="json"),
        "neutrality": (
            result.neutrality.model_dump(mode="json") if result.neutrality else None
        ),
        "gross_trade_stream_hash": result.gross_trade_stream_hash,
        "companions": companion_report,
        **verification_report_stamps(allowlist=run.payload.allowlist),
    }


register_identity_pair(
    name="SeedSnapshot",
    envelope_cls=SeedSnapshotEnvelope,
    payload_cls=SeedSnapshotPayload,
    id_field="seed_snapshot_id",
    example_factory=lambda: SeedSnapshotPayload(
        profile_name="ifvg_v2_doc_default_fresh_static_1r",
        resolved_section_config_hash="a" * 64,
        seed_schema_version=1,
        seed_hash="b" * 64,
        snapshot_through_day="2026-06-03",
        first_replay_day="2026-06-04",
        entering_day_seeds=DaySeedsRecord(
            prev_day="2026-06-03",
            prev_full_hl=(100, 50),
            prev_ny_day="2026-06-03",
            prev_ny_hl=(90, 60),
        ),
        chain_policy_id="development_explicit_dates_before_path_v2",
        chain_date_count=131,
        strategy_core_commit="c" * 40,
    ),
)
