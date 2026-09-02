"""Separately authorized seed production (§5.2–§5.5; F-16 / F-21).

No compliant profile-matching seed exists (§5.2): the v2 / replay-chart /
FSM-audit artifacts hold tables and evidence, not a restorable
``IfvgDaySeed`` state graph, and derivation from those tables is not
allowed. The lawful path is ONE separately authorized seed-production
chain replay whose only outputs are the profile-bound seed snapshot, the
access audit and the run receipt. This module authors that path; it never
runs it against real data by itself (the owner's signed
``SeedProductionAuthorizationRef`` is required before any source path is
constructed, and synthetic-provenance authorizations are confined to
``test`` namespaces).

* :class:`SeedProductionReplayPolicy` — the access policy of the chain:
  exactly the ordered STORE-DAY chain (every non-Saturday calendar day
  between its first and last day — the sequence the accepted chain
  replayed; the Sunday file holds the Sunday 18:00 ET open of Monday's
  trading day), every day inside the permitted development window, June 11
  and the sealed range refused BEFORE path construction.
* :class:`SeedProductionAuthorizationPayload` / ``Envelope`` / ``Ref`` —
  binds the store namespace and its supersession-head witness, the
  baseline profile and section hash, the chain (store days + the logical
  trading-day subset), ``snapshot_through_day``, the first intended
  verification day, the access/calendar/chain policy ids, the expected
  source-inventory hash, the Quant-Lab and Strategy-Core identities, the
  seed schema version, ``final_day_exhausts_dataset = False``, the
  permitted and prohibited outputs, and the owner fields. An owner-signed
  authorization must start the chain at the canonical cold start
  (``2026-01-01``); a divergent profile / source / code / date requires a
  new authorization.
* :func:`verify_seed_production_authorization` runs EVERY check before any
  source path exists; :func:`run_seed_production_chain` then replays the
  chain with ``final_day_exhausts_dataset=False``, persists the seed
  snapshot (profile-bound, content-addressed, schema-validated, reload-
  verified), the access audit and the run receipt — and nothing else.
* The unsigned packets (§5.4 B/G, §5.5) carry ``<OWNER_TO_FILL>``
  placeholders that FAIL validation; only the owner creates the refs.

Every report states the exact number of seed-chain days and that this is
a separately authorized preparation action, not part of the ≤5-day
verification evidence footprint.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator
from strategy_core.strategies.ifvg_smc.state import IFVG_SEED_SCHEMA_VERSION, IfvgDaySeed, seed_hash

from ..config import IfvgCaptureConfig
from ..data_access import DataAccessAudit, ExplorationDataPolicy, allowlist_sha256
from ..dataset import build_ifvg_v2_capture
from ..day_artifacts import DayArtifacts, DaySeeds, load_day_artifacts
from ..development_access import (
    PERMITTED_DEVELOPMENT_DATES,
    DevelopmentAccessAudit,
    DevelopmentReplayPolicy,
)
from ..profiles import ResolvedProfileConfig
from .child_replay import (
    ArtifactProvenanceReadAdapter,
    DaySeedsRecord,
    SeedSnapshotEnvelope,
    SeedSnapshotError,
    SeedSnapshotPayload,
    load_seed_snapshot,
    save_seed_snapshot,
)
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    quant_lab_replay_source_identity,
    register_identity_pair,
    strategy_core_source_identity,
)
from .store import SearchStoreError, has_envelope, load_verified_envelope, save_or_reuse_envelope
from .store_namespace import (
    StoreNamespaceEnvelope,
    StoreNamespaceError,
    SupersessionHeadWitness,
    assert_namespace_deployment_coherent,
    load_store_namespace,
    path_looks_like_research_store,
)
from .supersession_chain import assert_head_witness_current, current_supersession_head_witness
from .trading_calendar import (
    CANONICAL_CHAIN_START_DAY,
    PERMITTED_WINDOW_LAST_DAY,
    TRADING_CALENDAR_POLICY_ID,
    assert_consecutive_logical_days,
    is_logical_trading_day,
    store_day_chain,
    trading_day_ref_from_inventory,
)
from .verification import VERIFICATION_POLICY_ID

__all__ = [
    "SEED_PRODUCTION_ACCESS_POLICY_ID",
    "SEED_PRODUCTION_CHAIN_POLICY_ID",
    "SEED_PRODUCTION_AUTHORIZATION_POLICY_ID",
    "SEED_PRODUCTION_AUTHORIZATION_STORE",
    "SEED_PRODUCTION_RUN_STORE",
    "SEED_SNAPSHOT_STORE",
    "PERMITTED_SEED_OUTPUTS",
    "PROHIBITED_SEED_OUTPUTS",
    "SEED_PRODUCTION_FAILURE_REASONS",
    "OWNER_PLACEHOLDER",
    "SeedProductionAuthorizationError",
    "SeedProductionReplayPolicy",
    "SeedProductionAuthorizationPayload",
    "SeedProductionAuthorizationEnvelope",
    "SeedProductionAuthorizationRef",
    "SeedProductionRunPayload",
    "SeedProductionRunEnvelope",
    "SeedProductionRunResult",
    "seed_chain_source_inventory_hash",
    "persist_seed_production_authorization",
    "synthetic_seed_production_authorization",
    "verify_seed_production_authorization",
    "run_seed_production_chain",
    "build_seed_production_packet",
    "render_seed_production_packet_markdown",
    "build_verification_authorization_packet",
    "render_verification_packet_markdown",
    "store_day_chain",
]

SEED_PRODUCTION_ACCESS_POLICY_ID = "seed_production_explicit_chain_v1"
SEED_PRODUCTION_CHAIN_POLICY_ID = "seed_production_store_day_chain_v1"
SEED_PRODUCTION_AUTHORIZATION_POLICY_ID = "seed_production_authorization_v1"
SEED_PRODUCTION_RUN_POLICY_ID = "seed_production_run_v1"
SEED_PRODUCTION_AUTHORIZATION_STORE = "seed_production_authorizations"
SEED_PRODUCTION_RUN_STORE = "seed_production_runs"
SEED_SNAPSHOT_STORE = "seed_snapshots"
CHAIN_DAY_SEMANTICS = "physical_store_day_chain_v1"
OWNER_PLACEHOLDER = "<OWNER_TO_FILL>"
_ACCESS_AUDIT_SIDECAR = "access_audit.json"
_DEFAULT_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"

PERMITTED_SEED_OUTPUTS: tuple[str, ...] = ("seed_snapshot", "access_audit", "run_receipt")
PROHIBITED_SEED_OUTPUTS: tuple[str, ...] = (
    "research_search_metrics",
    "candidate_trade_performance_reports",
    "model_fitting",
    "feature_studies",
    "prop_simulation",
    "frontier_insights",
    "research_catalog_publication",
)
#: The stores a seed-production run may write — everything else is a
#: prohibited output by construction.
SEED_PRODUCTION_STORES_WRITTEN: tuple[str, ...] = (SEED_SNAPSHOT_STORE, SEED_PRODUCTION_RUN_STORE)

SEED_PRODUCTION_FAILURE_REASONS: tuple[str, ...] = (
    "store_namespace_missing",
    "store_namespace_mismatch",
    "synthetic_provenance_confined_to_test_namespaces",
    "namespace_deployment_incoherent",
    "authorization_not_found",
    "authorization_ref_mismatch",
    "supersession_head_witness_mismatch",
    "supersession_head_shorter_than_witness",
    "profile_mismatch",
    "chain_mismatch",
    "source_inventory_mismatch",
    "code_identity_mismatch",
    "seed_schema_mismatch",
    "authorization_not_effective",
    "replay_did_not_produce_a_seed",
    "seed_snapshot_unverified",
    "seed_not_continuous_with_window",
    "run_receipt_unverified",
    "prohibited_output_written",
)


class SeedProductionAuthorizationError(PermissionError):
    """A seed-production authorization / run invariant refused (typed ``reason``)."""

    def __init__(self, reason: str, message: str) -> None:
        if reason not in SEED_PRODUCTION_FAILURE_REASONS:
            raise ValueError(f"unregistered seed-production failure reason {reason!r}")
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _parse_instant(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must carry an explicit UTC offset")
    return parsed


def _assert_store_day_chain(days: tuple[str, ...]) -> None:
    """The structural rule of a seed chain, shared by the policy and the
    authorization payload (message fragments are test-pinned)."""

    if not days:
        raise ValueError("the seed-production chain cannot be empty")
    if days != tuple(sorted(days)) or len(set(days)) != len(days):
        raise ValueError("seed-production chain days must be unique and chronological")
    for day in days:
        if date.fromisoformat(day).weekday() == 5:
            raise ValueError(f"{day} is a Saturday: no store day ever falls on a Saturday")
    forbidden = [day for day in days if day not in PERMITTED_DEVELOPMENT_DATES]
    if forbidden:
        raise PermissionError(
            "seed-production chain days must lie inside the permitted development window; "
            f"refused before path construction: {forbidden}"
        )
    if days != store_day_chain(days[0], days[-1]):
        raise ValueError(
            "the seed-production chain must be the complete store-day chain between its "
            "first and last day (every non-Saturday calendar day; a gap or an extra day is "
            "refused)"
        )


# ── the access policy ────────────────────────────────────────────────────────


class SeedProductionReplayPolicy(DevelopmentReplayPolicy):
    """The fourth trusted policy class: one authorized seed-production chain.

    Authorizes exactly the ordered store-day chain; refuses June 11, the
    sealed range and any off-chain date BEFORE path construction. Like the
    verification policy it bypasses the frozen ten-date warmup demand of
    ``DevelopmentReplayPolicy.__init__`` (the chain may be a prefix of the
    development history) while keeping the development-window trust flag,
    so ``require_fixed_exploration_allowlist`` admits it inside the window.
    """

    _ifvg_development_policy_v2 = True
    _ifvg_seed_production_policy_v1 = True
    policy_id = SEED_PRODUCTION_ACCESS_POLICY_ID

    def __init__(
        self,
        chain_replay_days: Iterable[str],
        *,
        development_audit: DevelopmentAccessAudit | None = None,
    ) -> None:
        ordered = tuple(str(day) for day in chain_replay_days)
        _assert_store_day_chain(ordered)
        ExplorationDataPolicy.__init__(self, audit=DataAccessAudit(), allowlist=frozenset(ordered))
        self.allowed_dates = ordered
        self.development_audit = development_audit or DevelopmentAccessAudit()

    def audit_dict(self) -> dict[str, Any]:
        payload = self.development_audit.as_dict()
        payload["policy"] = self.policy_id
        payload["authorized_date_sha256"] = hashlib.sha256(
            "\n".join(self.allowed_dates).encode("utf-8")
        ).hexdigest()
        payload["chain_replay_day_count"] = len(self.allowed_dates)
        payload["chain_first_day"] = self.allowed_dates[0]
        payload["chain_last_day"] = self.allowed_dates[-1]
        return payload


# ── contracts ────────────────────────────────────────────────────────────────


class SeedProductionAuthorizationPayload(FrozenContract):
    authorization_policy_id: Literal["seed_production_authorization_v1"] = (
        SEED_PRODUCTION_AUTHORIZATION_POLICY_ID
    )
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    baseline_profile_name: str = Field(min_length=1)
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    chain_day_semantics: Literal["physical_store_day_chain_v1"] = CHAIN_DAY_SEMANTICS
    ordered_seed_chain_replay_days: tuple[str, ...] = Field(min_length=1)
    ordered_seed_chain_logical_trading_days: tuple[str, ...]
    snapshot_through_day: str
    first_intended_verification_day: str
    access_policy_id: Literal["seed_production_explicit_chain_v1"] = (
        SEED_PRODUCTION_ACCESS_POLICY_ID
    )
    calendar_policy_id: Literal["cme_globex_18et_weekday_v1"] = TRADING_CALENDAR_POLICY_ID
    chain_policy_id: Literal["seed_production_store_day_chain_v1"] = SEED_PRODUCTION_CHAIN_POLICY_ID
    expected_source_inventory_hash: str = Field(pattern=SHA256_PATTERN)
    quant_lab_source_identity: str = Field(pattern=SHA256_PATTERN)
    strategy_core_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    strategy_core_source_identity: str = Field(pattern=SHA256_PATTERN)
    seed_schema_version: int = Field(ge=1)
    final_day_exhausts_dataset: Literal[False] = False
    permitted_outputs: tuple[str, ...] = PERMITTED_SEED_OUTPUTS
    prohibited_outputs: tuple[str, ...] = PROHIBITED_SEED_OUTPUTS
    provenance: Literal["owner_signed", "synthetic_test_authorization_v1"]
    owner_decision_refs: tuple[str, ...] = Field(min_length=1)
    approved_by: str = Field(min_length=1)
    approved_at: str
    effective_from: str

    @model_validator(mode="after")
    def _well_formed(self):
        for name in ("baseline_profile_name", "approved_by", "approved_at", "effective_from"):
            if OWNER_PLACEHOLDER in str(getattr(self, name)):
                raise ValueError(f"a proposal placeholder in {name} cannot be persisted")
        if any(OWNER_PLACEHOLDER in ref for ref in self.owner_decision_refs):
            raise ValueError("a proposal placeholder in owner_decision_refs cannot be persisted")
        days = self.ordered_seed_chain_replay_days
        try:
            _assert_store_day_chain(days)
        except PermissionError as error:  # a contract refusal, not an access refusal
            raise ValueError(str(error)) from error
        logical = tuple(day for day in days if is_logical_trading_day(day))
        if self.ordered_seed_chain_logical_trading_days != logical:
            raise ValueError(
                "ordered_seed_chain_logical_trading_days must be exactly the logical "
                "trading-day subset of the replay chain"
            )
        if self.snapshot_through_day != days[-1]:
            raise ValueError("snapshot_through_day must be the last replay day of the chain")
        expected_first = (date.fromisoformat(days[-1]) + timedelta(days=1)).isoformat()
        if self.first_intended_verification_day != expected_first or not is_logical_trading_day(
            self.first_intended_verification_day
        ):
            raise ValueError(
                "first_intended_verification_day must be the logical trading day whose "
                "prior physical partition is snapshot_through_day"
            )
        if self.first_intended_verification_day > PERMITTED_WINDOW_LAST_DAY:
            raise ValueError("first_intended_verification_day lies outside the permitted window")
        if self.permitted_outputs != PERMITTED_SEED_OUTPUTS:
            raise ValueError(
                "permitted_outputs are fixed to seed_snapshot, access_audit, run_receipt"
            )
        if self.prohibited_outputs != PROHIBITED_SEED_OUTPUTS:
            raise ValueError("prohibited_outputs must be the registered list")
        if self.seed_schema_version != IFVG_SEED_SCHEMA_VERSION:
            raise ValueError(
                f"seed_schema_version must be the installed {IFVG_SEED_SCHEMA_VERSION}"
            )
        if self.provenance == "owner_signed" and days[0] != CANONICAL_CHAIN_START_DAY:
            raise ValueError(
                "an owner-signed seed chain must start at the canonical chain start "
                f"{CANONICAL_CHAIN_START_DAY} (the accepted chain's cold start)"
            )
        approved = _parse_instant(self.approved_at, field="approved_at")
        effective = _parse_instant(self.effective_from, field="effective_from")
        if effective < approved:
            raise ValueError("effective_from precedes approved_at")
        return self


class SeedProductionAuthorizationRef(FrozenContract):
    seed_production_authorization_id: str = Field(pattern=SHA256_PATTERN)
    approved_by: str = Field(min_length=1)
    approved_at: str
    content_hash: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def _content_hash_is_the_id(self):
        if self.content_hash != self.seed_production_authorization_id:
            raise ValueError("content_hash must equal the authorization's content id")
        return self


class SeedProductionAuthorizationEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "seed_production_authorization_id"

    seed_production_authorization_id: str = Field(pattern=SHA256_PATTERN)
    payload: SeedProductionAuthorizationPayload

    def ref(self) -> SeedProductionAuthorizationRef:
        return SeedProductionAuthorizationRef(
            seed_production_authorization_id=self.seed_production_authorization_id,
            approved_by=self.payload.approved_by,
            approved_at=self.payload.approved_at,
            content_hash=self.seed_production_authorization_id,
        )


class SeedProductionRunPayload(FrozenContract):
    run_policy_id: Literal["seed_production_run_v1"] = SEED_PRODUCTION_RUN_POLICY_ID
    seed_production_authorization_id: str = Field(pattern=SHA256_PATTERN)
    store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    supersession_head_witness: SupersessionHeadWitness
    seed_snapshot_id: str = Field(pattern=SHA256_PATTERN)
    seed_hash: str = Field(pattern=SHA256_PATTERN)
    baseline_profile_name: str
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    ordered_seed_chain_replay_days: tuple[str, ...] = Field(min_length=1)
    chain_replay_day_count: int = Field(ge=1)
    logical_trading_day_count: int = Field(ge=0)
    snapshot_through_day: str
    first_intended_verification_day: str
    access_policy_id: Literal["seed_production_explicit_chain_v1"] = (
        SEED_PRODUCTION_ACCESS_POLICY_ID
    )
    access_audit_sha256: str = Field(pattern=SHA256_PATTERN)
    quant_lab_source_identity: str = Field(pattern=SHA256_PATTERN)
    strategy_core_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    strategy_core_source_identity: str = Field(pattern=SHA256_PATTERN)
    output_namespace: Literal["search_test/v1"] = "search_test/v1"
    verification_evidence_footprint_days: Literal[0] = 0
    separately_authorized_preparation: Literal[True] = True
    stores_written: tuple[str, ...] = SEED_PRODUCTION_STORES_WRITTEN
    provenance: Literal["owner_signed", "synthetic_test_authorization_v1"]

    @model_validator(mode="after")
    def _coherent(self):
        if self.chain_replay_day_count != len(self.ordered_seed_chain_replay_days):
            raise ValueError("chain_replay_day_count must count the replay days")
        if self.stores_written != SEED_PRODUCTION_STORES_WRITTEN:
            raise ValueError(
                "a seed-production run writes exactly the seed snapshot and its receipt"
            )
        return self


class SeedProductionRunEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "seed_production_run_id"

    seed_production_run_id: str = Field(pattern=SHA256_PATTERN)
    payload: SeedProductionRunPayload


@dataclass(frozen=True)
class SeedProductionRunResult:
    authorization: SeedProductionAuthorizationEnvelope
    snapshot: SeedSnapshotEnvelope
    seed: IfvgDaySeed
    receipt: SeedProductionRunEnvelope
    access_audit: dict[str, Any]
    reused: bool


# ── helpers ──────────────────────────────────────────────────────────────────


def seed_chain_source_inventory_hash(
    chain_replay_days: Iterable[str], inventory: Mapping[str, tuple[str, str]]
) -> str:
    """The expected source-inventory hash of a chain: every physical store
    day's ``(kind, content sha256)`` from the ALREADY-AUTHORIZED inventory."""

    days = tuple(str(day) for day in chain_replay_days)
    missing = [day for day in days if day not in inventory]
    if missing:
        raise ValueError(f"the inventory lacks the chain's partitions: {missing[:5]}")
    return canonical_contract_sha256(
        {
            "policy": SEED_PRODUCTION_ACCESS_POLICY_ID,
            "chain_replay_days": list(days),
            "partitions": [[day, inventory[day][0], inventory[day][1]] for day in days],
        }
    )


def _namespace_for_authority(root: Path, *, provenance: str) -> StoreNamespaceEnvelope:
    try:
        namespace = load_store_namespace(root)
    except StoreNamespaceError as error:
        if error.reason == "store_namespace_missing":
            raise SeedProductionAuthorizationError(
                "store_namespace_missing",
                "an unmarked store cannot carry a seed-production authorization",
            ) from error
        raise
    if provenance == "synthetic_test_authorization_v1" and (
        namespace.payload.namespace_class != "test" or path_looks_like_research_store(root)
    ):
        raise SeedProductionAuthorizationError(
            "synthetic_provenance_confined_to_test_namespaces",
            "synthetic_test_authorization_v1 seed-production authorizations are confined to "
            "test namespaces (semantic class AND deployment path)",
        )
    return namespace


def _witness_current(root: Path, witness: SupersessionHeadWitness) -> None:
    try:
        assert_head_witness_current(root, witness)
    except StoreNamespaceError as error:
        reason = (
            error.reason
            if error.reason in SEED_PRODUCTION_FAILURE_REASONS
            else "supersession_head_witness_mismatch"
        )
        raise SeedProductionAuthorizationError(reason, str(error)) from error


def persist_seed_production_authorization(
    root: Path, payload: SeedProductionAuthorizationPayload
) -> SeedProductionAuthorizationEnvelope:
    """Persist (save-or-reuse) an authorization into ITS namespace: the payload's
    namespace id must be this store's, the witness must be current, and
    synthetic provenance is confined to test namespaces."""

    root = Path(root)
    namespace = _namespace_for_authority(root, provenance=payload.provenance)
    if payload.store_namespace_id != namespace.store_namespace_id:
        raise SeedProductionAuthorizationError(
            "store_namespace_mismatch",
            "the authorization names another store namespace",
        )
    _witness_current(root, payload.supersession_head_witness)
    envelope = SeedProductionAuthorizationEnvelope.from_payload(payload)
    stored, _reused = save_or_reuse_envelope(root, SEED_PRODUCTION_AUTHORIZATION_STORE, envelope)
    return stored


def synthetic_seed_production_authorization(
    root: Path,
    *,
    chain_replay_days: Iterable[str],
    first_intended_verification_day: str,
    inventory: Mapping[str, tuple[str, str]],
    quant_lab_source_identity: str,
    strategy_core_commit: str,
    strategy_core_source_identity: str,
    baseline_profile_name: str = _DEFAULT_PROFILE,
    resolved_section_config_hash: str | None = None,
    approved_at: str = "2026-09-01T00:00:00+00:00",
    effective_from: str | None = None,
    owner_decision_refs: tuple[str, ...] = ("21/R-5:synthetic",),
    persist: bool = True,
) -> SeedProductionAuthorizationEnvelope:
    """A SYNTHETIC-provenance authorization (test namespaces only) over an
    explicit chain — the fixture of the synthetic seed-production proof."""

    root = Path(root)
    namespace = _namespace_for_authority(root, provenance="synthetic_test_authorization_v1")
    witness = current_supersession_head_witness(root)
    if resolved_section_config_hash is None:
        from ..profiles import resolve_profile_config  # noqa: PLC0415

        resolved_section_config_hash = resolve_profile_config(
            {"profile_name": baseline_profile_name}
        ).section_config_hash
    days = tuple(str(day) for day in chain_replay_days)
    payload = SeedProductionAuthorizationPayload(
        store_namespace_id=namespace.store_namespace_id,
        supersession_head_witness=witness,
        baseline_profile_name=baseline_profile_name,
        resolved_section_config_hash=resolved_section_config_hash,
        ordered_seed_chain_replay_days=days,
        ordered_seed_chain_logical_trading_days=tuple(
            day for day in days if is_logical_trading_day(day)
        ),
        snapshot_through_day=days[-1],
        first_intended_verification_day=first_intended_verification_day,
        expected_source_inventory_hash=seed_chain_source_inventory_hash(days, inventory),
        quant_lab_source_identity=quant_lab_source_identity,
        strategy_core_commit=strategy_core_commit,
        strategy_core_source_identity=strategy_core_source_identity,
        seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
        provenance="synthetic_test_authorization_v1",
        owner_decision_refs=owner_decision_refs,
        approved_by="synthetic_fixture",
        approved_at=approved_at,
        effective_from=effective_from or approved_at,
    )
    envelope = SeedProductionAuthorizationEnvelope.from_payload(payload)
    if persist:
        return persist_seed_production_authorization(root, payload)
    return envelope


def _load_authorization(root: Path, authorization_id: str) -> SeedProductionAuthorizationEnvelope:
    try:
        return load_verified_envelope(
            Path(root),
            SEED_PRODUCTION_AUTHORIZATION_STORE,
            authorization_id,
            SeedProductionAuthorizationEnvelope,
        )
    except SearchStoreError as error:
        raise SeedProductionAuthorizationError(
            "authorization_not_found",
            f"no verified seed-production authorization {str(authorization_id)[:12]}… exists in "
            "this store (exact-id load; the store is never listed)",
        ) from error


def verify_seed_production_authorization(
    root: Path,
    authorization: str | SeedProductionAuthorizationRef | SeedProductionAuthorizationEnvelope,
    *,
    expected_profile_name: str,
    expected_section_config_hash: str,
    expected_chain_replay_days: Iterable[str],
    expected_source_inventory_hash: str,
    now: str,
    expected_quant_lab_source_identity: str | None = None,
    expected_strategy_core: tuple[str, str] | None = None,
) -> SeedProductionAuthorizationEnvelope:
    """Every check of §5.3 BEFORE any source path: verified store load,
    namespace equality, current head witness, provenance confinement,
    profile / section / chain / inventory / code-identity equality, seed
    schema, effectivity. Typed refusals only."""

    root = Path(root)
    if isinstance(
        authorization, SeedProductionAuthorizationEnvelope | SeedProductionAuthorizationRef
    ):
        authorization_id = authorization.seed_production_authorization_id
    else:
        authorization_id = str(authorization)
    try:
        namespace = load_store_namespace(root)
    except StoreNamespaceError as error:
        if error.reason == "store_namespace_missing":
            raise SeedProductionAuthorizationError(
                "store_namespace_missing", "an unmarked store cannot authorize a seed production"
            ) from error
        raise
    envelope = _load_authorization(root, authorization_id)
    payload = envelope.payload
    if isinstance(authorization, SeedProductionAuthorizationRef) and (
        authorization.content_hash != envelope.seed_production_authorization_id
        or authorization.approved_by != payload.approved_by
        or authorization.approved_at != payload.approved_at
    ):
        raise SeedProductionAuthorizationError(
            "authorization_ref_mismatch",
            "the SeedProductionAuthorizationRef does not agree with the persisted authorization",
        )
    if payload.store_namespace_id != namespace.store_namespace_id:
        raise SeedProductionAuthorizationError(
            "store_namespace_mismatch", "the authorization names another store namespace"
        )
    if payload.provenance == "synthetic_test_authorization_v1" and (
        namespace.payload.namespace_class != "test" or path_looks_like_research_store(root)
    ):
        raise SeedProductionAuthorizationError(
            "synthetic_provenance_confined_to_test_namespaces",
            "a synthetic seed-production authorization is lawful in test namespaces only",
        )
    _witness_current(root, payload.supersession_head_witness)
    if (
        payload.baseline_profile_name != expected_profile_name
        or payload.resolved_section_config_hash != expected_section_config_hash
    ):
        raise SeedProductionAuthorizationError(
            "profile_mismatch",
            "the authorization binds a different baseline profile / section hash",
        )
    if payload.ordered_seed_chain_replay_days != tuple(str(d) for d in expected_chain_replay_days):
        raise SeedProductionAuthorizationError(
            "chain_mismatch", "the authorization binds a different seed chain"
        )
    if payload.expected_source_inventory_hash != expected_source_inventory_hash:
        raise SeedProductionAuthorizationError(
            "source_inventory_mismatch",
            "the current source inventory does not hash to the authorized inventory",
        )
    if expected_quant_lab_source_identity is not None and (
        payload.quant_lab_source_identity != expected_quant_lab_source_identity
    ):
        raise SeedProductionAuthorizationError(
            "code_identity_mismatch", "the Quant-Lab replay source identity diverged"
        )
    if expected_strategy_core is not None and (
        (payload.strategy_core_commit, payload.strategy_core_source_identity)
        != tuple(expected_strategy_core)
    ):
        raise SeedProductionAuthorizationError(
            "code_identity_mismatch", "the Strategy-Core identity diverged"
        )
    if payload.seed_schema_version != IFVG_SEED_SCHEMA_VERSION:
        raise SeedProductionAuthorizationError(
            "seed_schema_mismatch", "the installed seed schema version differs"
        )
    if _parse_instant(payload.effective_from, field="effective_from") > _parse_instant(
        now, field="now"
    ):
        raise SeedProductionAuthorizationError(
            "authorization_not_effective", "the authorization is not yet effective"
        )
    return envelope


def _canonical_utc(value: Any) -> Any:
    """Canonicalize every aware ``datetime`` inside a seed graph to the stdlib
    ``timezone.utc`` (same instant, identical ``isoformat`` → identical
    ``seed_hash``). Day artifacts loaded from Parquet carry ``pytz`` tzinfo
    objects, which the seed sandbox unpickler (Strategy-Core state graphs +
    ``datetime`` only) rightly refuses; the seed a chain produces must not
    depend on which timezone library loaded its bars."""

    if isinstance(value, datetime):
        if value.tzinfo is None or isinstance(value.tzinfo, timezone):
            return value
        return value.astimezone(UTC)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{
                field.name: _canonical_utc(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        )
    if isinstance(value, tuple):
        return tuple(_canonical_utc(item) for item in value)
    if isinstance(value, list):
        return [_canonical_utc(item) for item in value]
    if isinstance(value, dict):
        return {key: _canonical_utc(item) for key, item in value.items()}
    return value


def _entering_day_seeds(last: DayArtifacts) -> DaySeeds:
    """The ``DaySeeds`` the next day's artifact must have been built with,
    given the last replayed day's artifact (the driver's chaining rule: an
    empty last day carries its own expectations through)."""

    previous_day = date.fromisoformat(last.date_str)
    if last.day_hl is None:
        return last.seeds
    return DaySeeds(
        prev_day=previous_day,
        prev_full_hl=last.day_hl,
        prev_ny_day=previous_day if last.ny_hl else last.seeds.prev_ny_day,
        prev_ny_hl=last.ny_hl if last.ny_hl else last.seeds.prev_ny_hl,
    )


def _store_entry_counts(root: Path) -> dict[str, int]:
    """Entry COUNTS per search store directory under ``root`` (a directory
    count, never an id listing) — the RA-07 post-condition baseline."""

    from .store import SEARCH_STORE_NAMES  # noqa: PLC0415

    counts: dict[str, int] = {}
    for name in SEARCH_STORE_NAMES:
        directory = Path(root) / name
        if directory.is_dir():
            counts[name] = sum(1 for entry in directory.iterdir() if entry.is_dir())
    return counts


_SEED_PRODUCTION_PERMITTED_STORES = frozenset({SEED_SNAPSHOT_STORE, SEED_PRODUCTION_RUN_STORE})


def _assert_only_permitted_stores_changed(root: Path, before: Mapping[str, int]) -> None:
    after = _store_entry_counts(root)
    changed = sorted(
        name
        for name in set(before) | set(after)
        if name not in _SEED_PRODUCTION_PERMITTED_STORES
        and before.get(name, 0) != after.get(name, 0)
    )
    if changed:
        raise SeedProductionAuthorizationError(
            "prohibited_output_written",
            "the seed-production run changed store(s) outside its permitted outputs "
            f"({changed}); the preparation action writes only the seed snapshot, the access "
            "audit and the run receipt",
        )


def run_seed_production_chain(
    *,
    root: Path,
    authorization_id: str,
    cfg: IfvgCaptureConfig,
    resolved_profile: ResolvedProfileConfig,
    source_inventory: Mapping[str, tuple[str, str]],
    now: str,
    quant_lab_source_identity: str | None = None,
    strategy_core_identity: tuple[str, str] | None = None,
    repo_root: Path | None = None,
    strategy_core_root: Path | None = None,
    artifact_provenance_dates: tuple[str, ...] = (),
    cached_artifacts_only: bool = True,
) -> SeedProductionRunResult:
    """Run ONE authorized seed-production chain: verify EVERYTHING before any
    source path → replay the chain (``final_day_exhausts_dataset=False``) →
    persist the profile-bound seed snapshot, the access audit and the run
    receipt → reload-verify the snapshot. Nothing else is written."""

    root = Path(root)
    repo = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[6]
    ql_identity = quant_lab_source_identity or quant_lab_replay_source_identity(
        repository_root=repo
    )
    sc_identity = strategy_core_identity or strategy_core_source_identity(
        repository_root=strategy_core_root or repo.parent / "Strategy-Core"
    )
    # the authorization is the source of truth for the chain; the CURRENT
    # inventory must hash to the authorized inventory over that chain
    authorization = _load_authorization(root, authorization_id)
    chain = authorization.payload.ordered_seed_chain_replay_days
    try:
        inventory_hash = seed_chain_source_inventory_hash(chain, source_inventory)
    except ValueError as error:
        raise SeedProductionAuthorizationError("source_inventory_mismatch", str(error)) from error
    profile_name = getattr(resolved_profile.section, "profile_name", _DEFAULT_PROFILE)
    authorization = verify_seed_production_authorization(
        root,
        authorization,
        expected_profile_name=str(profile_name),
        expected_section_config_hash=resolved_profile.section_config_hash,
        expected_chain_replay_days=chain,
        expected_source_inventory_hash=inventory_hash,
        expected_quant_lab_source_identity=ql_identity,
        expected_strategy_core=(sc_identity[0], sc_identity[1]),
        now=now,
    )
    namespace = load_store_namespace(root)
    try:
        assert_namespace_deployment_coherent(root, namespace)
    except StoreNamespaceError as error:
        raise SeedProductionAuthorizationError(
            "namespace_deployment_incoherent", str(error)
        ) from error
    if cfg.profile_hash != resolved_profile.section_config_hash:
        raise SeedProductionAuthorizationError(
            "profile_mismatch", "the capture config section does not match the resolved profile"
        )
    payload = authorization.payload
    inner = SeedProductionReplayPolicy(chain)
    if artifact_provenance_dates:
        if not cached_artifacts_only:
            raise PermissionError(
                "the artifact-provenance read adapter is read-only; pass cached_artifacts_only=True"
            )
        policy: Any = ArtifactProvenanceReadAdapter(
            inner, artifact_provenance_dates=tuple(artifact_provenance_dates)
        )
    else:
        policy = inner
    # adversarial RA-07: the permitted outputs are a POST-CONDITION, not only a
    # construction — every other store directory must hold exactly as many
    # entries after the run as before (counts only; no entry is ever listed
    # by id)
    stores_before = _store_entry_counts(root)
    capture = build_ifvg_v2_capture(
        list(chain),
        cfg,
        resolved_profile,
        access_policy=policy,
        cached_artifacts_only=cached_artifacts_only,
        final_day_exhausts_dataset=False,
        audit_capture_mode="disabled",
    )
    capture.access_policy.assert_zero_forbidden_access()
    seed = capture.end_seed
    if seed is None:
        raise SeedProductionAuthorizationError(
            "replay_did_not_produce_a_seed", "the chain replay returned no end seed"
        )
    if seed.profile_hash != resolved_profile.section_config_hash:
        raise SeedProductionAuthorizationError(
            "profile_mismatch", "the produced seed is bound to another profile section"
        )
    produced_hash = seed_hash(seed)
    seed = _canonical_utc(seed)
    if seed_hash(seed) != produced_hash:
        raise SeedSnapshotError(
            "timezone canonicalization changed the seed hash; refusing to persist"
        )
    last = load_day_artifacts(chain[-1], cfg, expected_seeds=None, access_policy=policy)
    if last is None:
        raise SeedProductionAuthorizationError(
            "replay_did_not_produce_a_seed",
            "the last chain day's artifacts could not be re-read for the entering day seeds",
        )
    policy.assert_zero_forbidden_access()
    snapshot_payload = SeedSnapshotPayload(
        profile_name=payload.baseline_profile_name,
        resolved_section_config_hash=payload.resolved_section_config_hash,
        seed_schema_version=int(seed.schema_version),
        seed_hash=seed_hash(seed),
        snapshot_through_day=payload.snapshot_through_day,
        first_replay_day=payload.first_intended_verification_day,
        entering_day_seeds=DaySeedsRecord.from_day_seeds(_entering_day_seeds(last)),
        chain_policy_id=SEED_PRODUCTION_CHAIN_POLICY_ID,
        chain_date_count=len(chain),
        strategy_core_commit=sc_identity[0],
    )
    snapshot_id = canonical_contract_sha256(snapshot_payload)
    snapshot_existed = has_envelope(root, SEED_SNAPSHOT_STORE, snapshot_id)
    snapshot = save_seed_snapshot(root, snapshot_payload, seed)
    audit = policy.audit_dict()
    audit_bytes = (json.dumps(audit, sort_keys=True) + "\n").encode("utf-8")
    receipt_payload = SeedProductionRunPayload(
        seed_production_authorization_id=authorization.seed_production_authorization_id,
        store_namespace_id=namespace.store_namespace_id,
        supersession_head_witness=payload.supersession_head_witness,
        seed_snapshot_id=snapshot.seed_snapshot_id,
        seed_hash=snapshot_payload.seed_hash,
        baseline_profile_name=payload.baseline_profile_name,
        resolved_section_config_hash=payload.resolved_section_config_hash,
        ordered_seed_chain_replay_days=chain,
        chain_replay_day_count=len(chain),
        logical_trading_day_count=len(payload.ordered_seed_chain_logical_trading_days),
        snapshot_through_day=payload.snapshot_through_day,
        first_intended_verification_day=payload.first_intended_verification_day,
        access_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(),
        quant_lab_source_identity=ql_identity,
        strategy_core_commit=sc_identity[0],
        strategy_core_source_identity=sc_identity[1],
        provenance=payload.provenance,
    )
    receipt, receipt_reused = save_or_reuse_envelope(
        root,
        SEED_PRODUCTION_RUN_STORE,
        SeedProductionRunEnvelope.from_payload(receipt_payload),
        extra_files={_ACCESS_AUDIT_SIDECAR: audit_bytes},
    )
    # reload → verify (profile-bound; the stored seed must hash to the payload)
    reloaded, chain_start = load_seed_snapshot(
        root,
        snapshot.seed_snapshot_id,
        expected_section_config_hash=payload.resolved_section_config_hash,
    )
    if seed_hash(chain_start.seed) != snapshot_payload.seed_hash:
        raise SeedSnapshotError("the reloaded seed does not hash to the produced seed")
    _assert_only_permitted_stores_changed(root, stores_before)
    return SeedProductionRunResult(
        authorization=authorization,
        snapshot=reloaded,
        seed=chain_start.seed,
        receipt=receipt,
        access_audit=audit,
        reused=bool(snapshot_existed and receipt_reused),
    )


# ── unsigned packets ─────────────────────────────────────────────────────────

_PROPOSAL_NOTE = (
    "PROPOSAL — nothing here is an authorization; the placeholders fail validation "
    "until the owner completes the signature workflow"
)


def build_seed_production_packet(
    root: Path,
    *,
    baseline_profile_name: str,
    resolved_section_config_hash: str,
    first_intended_verification_day: str,
    inventory: Mapping[str, tuple[str, str]],
    quant_lab_source_identity: str,
    strategy_core_commit: str,
    strategy_core_source_identity: str,
    owner_decision_refs: tuple[str, ...] = (f"21/R-5:{OWNER_PLACEHOLDER}",),
) -> dict[str, Any]:
    """The UNSIGNED seed-production packet (§5.4 step B): the full canonical
    store-day chain from 2026-01-01 through the physical day before the first
    intended verification day, every binding filled from evidence, and the
    owner fields left as placeholders that fail validation."""

    root = Path(root)
    namespace = _namespace_for_authority(root, provenance="owner_signed")
    witness = current_supersession_head_witness(root)
    if not is_logical_trading_day(first_intended_verification_day):
        raise ValueError(f"{first_intended_verification_day} is not a logical trading day")
    if first_intended_verification_day > PERMITTED_WINDOW_LAST_DAY:
        raise ValueError("the first intended verification day lies outside the permitted window")
    snapshot_through = (
        date.fromisoformat(first_intended_verification_day) - timedelta(days=1)
    ).isoformat()
    chain = store_day_chain(CANONICAL_CHAIN_START_DAY, snapshot_through)
    logical = tuple(day for day in chain if is_logical_trading_day(day))
    payload = {
        "authorization_policy_id": SEED_PRODUCTION_AUTHORIZATION_POLICY_ID,
        "store_namespace_id": namespace.store_namespace_id,
        "supersession_head_witness": witness.model_dump(mode="json"),
        "baseline_profile_name": baseline_profile_name,
        "resolved_section_config_hash": resolved_section_config_hash,
        "chain_day_semantics": CHAIN_DAY_SEMANTICS,
        "ordered_seed_chain_replay_days": list(chain),
        "ordered_seed_chain_logical_trading_days": list(logical),
        "snapshot_through_day": snapshot_through,
        "first_intended_verification_day": first_intended_verification_day,
        "access_policy_id": SEED_PRODUCTION_ACCESS_POLICY_ID,
        "calendar_policy_id": TRADING_CALENDAR_POLICY_ID,
        "chain_policy_id": SEED_PRODUCTION_CHAIN_POLICY_ID,
        "expected_source_inventory_hash": seed_chain_source_inventory_hash(chain, inventory),
        "quant_lab_source_identity": quant_lab_source_identity,
        "strategy_core_commit": strategy_core_commit,
        "strategy_core_source_identity": strategy_core_source_identity,
        "seed_schema_version": IFVG_SEED_SCHEMA_VERSION,
        "final_day_exhausts_dataset": False,
        "permitted_outputs": list(PERMITTED_SEED_OUTPUTS),
        "prohibited_outputs": list(PROHIBITED_SEED_OUTPUTS),
        "provenance": "owner_signed",
        "owner_decision_refs": list(owner_decision_refs),
        "approved_by": OWNER_PLACEHOLDER,
        "approved_at": OWNER_PLACEHOLDER,
        "effective_from": OWNER_PLACEHOLDER,
    }
    return {
        "packet_kind": "seed_production_authorization_packet_v1",
        "payload": payload,
        "seed_chain_replay_day_count": len(chain),
        "logical_trading_day_count": len(logical),
        "separately_authorized_preparation": True,
        "verification_evidence_footprint_days": 0,
        "owner_fields_to_complete": [
            "approved_by",
            "approved_at",
            "effective_from",
            "owner_decision_refs",
        ],
        "_proposal_note": _PROPOSAL_NOTE,
    }


def render_seed_production_packet_markdown(packet: Mapping[str, Any], *, title: str) -> str:
    payload = packet["payload"]
    lines = [
        f"# {title}",
        "",
        "**PROPOSAL — nothing here is an authorization.** The owner's signature workflow",
        "replaces every placeholder; a payload carrying a placeholder cannot be persisted.",
        "",
        "This is a **separately authorized preparation action**, not part of the ≤5-day",
        "verification evidence footprint (footprint days: "
        f"{packet['verification_evidence_footprint_days']}).",
        "",
        "- seed chain replay days (physical store days): "
        f"**{packet['seed_chain_replay_day_count']}** "
        f"({payload['ordered_seed_chain_replay_days'][0]} … {payload['snapshot_through_day']})",
        f"- logical trading days inside the chain: {packet['logical_trading_day_count']}",
        f"- first intended verification day: {payload['first_intended_verification_day']}",
        f"- baseline profile: `{payload['baseline_profile_name']}` / section "
        f"`{payload['resolved_section_config_hash']}`",
        f"- store namespace: `{payload['store_namespace_id']}`; head witness: "
        f"`{payload['supersession_head_witness']['line_count']}` / "
        f"`{payload['supersession_head_witness']['head_sha256']}`",
        f"- expected source-inventory hash: `{payload['expected_source_inventory_hash']}`",
        f"- Quant-Lab identity `{payload['quant_lab_source_identity']}`; Strategy-Core "
        f"`{payload['strategy_core_commit']}` / `{payload['strategy_core_source_identity']}`",
        f"- permitted outputs: {', '.join(payload['permitted_outputs'])}",
        f"- prohibited outputs: {', '.join(payload['prohibited_outputs'])}",
        f"- `final_day_exhausts_dataset` = {payload['final_day_exhausts_dataset']}",
        "",
        f"Owner fields to complete: {', '.join(packet['owner_fields_to_complete'])}.",
        "",
        "```json",
        json.dumps(payload, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


def build_verification_authorization_packet(
    root: Path,
    *,
    seed_snapshot_id: str,
    logical_window: Iterable[str],
    resolved_section_config_hash: str,
    baseline_profile_name: str,
    shortlist_id: str,
    coverage_matrix_artifact_id: str,
    expected_source_inventory_hash: str,
    seed_production_run_id: str | None = None,
    inventory: Mapping[str, tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """The UNSIGNED final verification packet (§5.5): buildable ONLY after a
    verified seed exists (exact-loaded, profile-bound, continuous with the
    window). Every §5.5 item is present; the owner fields are blank. The
    allowlist hash is computed locally for DISPLAY only — nothing is
    registered."""

    root = Path(root)
    namespace = load_store_namespace(root)
    witness = current_supersession_head_witness(root)
    window = assert_consecutive_logical_days(logical_window, max_days=5)
    if window[-1] > PERMITTED_WINDOW_LAST_DAY:
        raise ValueError("the verification window lies outside the permitted window")
    try:
        snapshot, _chain_start = load_seed_snapshot(
            root, seed_snapshot_id, expected_section_config_hash=resolved_section_config_hash
        )
    except (SearchStoreError, SeedSnapshotError) as error:
        raise SeedProductionAuthorizationError(
            "seed_snapshot_unverified",
            f"no verified profile-bound seed snapshot {str(seed_snapshot_id)[:12]}… exists "
            f"({error})",
        ) from error
    if (
        snapshot.payload.first_replay_day != window[0]
        or snapshot.payload.profile_name != baseline_profile_name
    ):
        raise SeedProductionAuthorizationError(
            "seed_not_continuous_with_window",
            "the seed snapshot's first replay day / profile do not match the window",
        )
    seed_provenance: dict[str, Any] = {
        "provenance": "unknown_no_run_receipt_supplied",
        "seed_production_run_id": None,
        "seed_production_authorization_id": None,
    }
    if seed_production_run_id is not None:
        try:
            receipt = load_verified_envelope(
                root, SEED_PRODUCTION_RUN_STORE, seed_production_run_id, SeedProductionRunEnvelope
            )
        except SearchStoreError as error:
            raise SeedProductionAuthorizationError(
                "run_receipt_unverified", f"no verified seed-production run receipt ({error})"
            ) from error
        if receipt.payload.seed_snapshot_id != snapshot.seed_snapshot_id:
            raise SeedProductionAuthorizationError(
                "run_receipt_unverified", "the run receipt names another seed snapshot"
            )
        seed_provenance = {
            "provenance": receipt.payload.provenance,
            "seed_production_run_id": receipt.seed_production_run_id,
            "seed_production_authorization_id": receipt.payload.seed_production_authorization_id,
        }
    refs = []
    if inventory is not None:
        for day in window:
            ref = trading_day_ref_from_inventory(day, inventory)
            refs.append(None if ref is None else ref.model_dump(mode="json"))
    display_hash = allowlist_sha256(window)
    stamps = {
        "verification_only": True,
        "not_for_research_interpretation": True,
        "full_pipeline_not_run": True,
    }
    ref_draft = {
        "verification_policy_id": VERIFICATION_POLICY_ID,
        "approved_allowlist_hash": display_hash,
        "coverage_matrix_artifact_id": coverage_matrix_artifact_id,
        "seed_snapshot_id": snapshot.seed_snapshot_id,
        "store_namespace_id": namespace.store_namespace_id,
        "supersession_head_witness": witness.model_dump(mode="json"),
        "approved_by": OWNER_PLACEHOLDER,
        "approved_at": OWNER_PLACEHOLDER,
        "content_hash": OWNER_PLACEHOLDER,
    }
    return {
        "packet_kind": "verification_authorization_packet_v1",
        "corrected_logical_allowlist": list(window),
        "corrected_logical_allowlist_hash_display_only": display_hash,
        "trading_day_refs": refs,
        "shortlist_id": shortlist_id,
        "coverage_matrix_artifact_id": coverage_matrix_artifact_id,
        "baseline_profile": {
            "profile_name": baseline_profile_name,
            "resolved_section_config_hash": resolved_section_config_hash,
        },
        "seed": {
            "seed_snapshot_id": snapshot.seed_snapshot_id,
            "seed_hash": snapshot.payload.seed_hash,
            "seed_schema_version": snapshot.payload.seed_schema_version,
            "snapshot_through_day": snapshot.payload.snapshot_through_day,
            "first_replay_day": snapshot.payload.first_replay_day,
            "chain_policy_id": snapshot.payload.chain_policy_id,
            "chain_date_count": snapshot.payload.chain_date_count,
            "strategy_core_commit": snapshot.payload.strategy_core_commit,
            **seed_provenance,
        },
        "verification_replay_policy_id": VERIFICATION_POLICY_ID,
        "store_namespace_id": namespace.store_namespace_id,
        "supersession_head_witness": witness.model_dump(mode="json"),
        "pipeline_semantic_id": "<PENDING: frozen pipeline specification>",
        "expected_replay_input_bundle_source_inventory_hash": expected_source_inventory_hash,
        "zero_protected_sealed_requirements": True,
        "stamps": stamps,
        "verification_authorization_ref": ref_draft,
        "owner_fields_to_complete": ["approved_by", "approved_at", "content_hash"],
        "_proposal_note": _PROPOSAL_NOTE,
    }


def render_verification_packet_markdown(packet: Mapping[str, Any], *, title: str) -> str:
    seed = packet["seed"]
    lines = [
        f"# {title}",
        "",
        "**PROPOSAL — nothing here is an authorization.** Only the owner may create the",
        "final `VerificationAuthorizationRef`; the placeholders below fail validation.",
        "",
        f"- corrected logical allowlist: {' '.join(packet['corrected_logical_allowlist'])} "
        f"(hash, display only: `{packet['corrected_logical_allowlist_hash_display_only']}`)",
        f"- shortlist id `{packet['shortlist_id']}`; coverage matrix "
        f"`{packet['coverage_matrix_artifact_id']}`",
        f"- baseline profile `{packet['baseline_profile']['profile_name']}` / section "
        f"`{packet['baseline_profile']['resolved_section_config_hash']}`",
        f"- seed snapshot `{seed['seed_snapshot_id']}` (seed hash `{seed['seed_hash']}`, schema "
        f"{seed['seed_schema_version']}, through {seed['snapshot_through_day']}, first replay day "
        f"{seed['first_replay_day']}, chain policy `{seed['chain_policy_id']}`, "
        f"{seed['chain_date_count']} chain days, provenance `{seed['provenance']}`)",
        f"- verification replay policy `{packet['verification_replay_policy_id']}`",
        f"- store namespace `{packet['store_namespace_id']}`; head witness "
        f"`{packet['supersession_head_witness']['line_count']}` / "
        f"`{packet['supersession_head_witness']['head_sha256']}`",
        f"- pipeline / verification semantic identity: {packet['pipeline_semantic_id']}",
        "- expected ReplayInputBundle source-inventory hash "
        f"`{packet['expected_replay_input_bundle_source_inventory_hash']}`",
        f"- zero protected / sealed requirements: {packet['zero_protected_sealed_requirements']}",
        "- stamps: " + ", ".join(f"{k}={v}" for k, v in packet["stamps"].items()),
        "",
        f"Owner fields to complete: {', '.join(packet['owner_fields_to_complete'])}.",
        "",
        "```json",
        json.dumps(packet["verification_authorization_ref"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)


# ── identity registry ────────────────────────────────────────────────────────


def _example_authorization() -> SeedProductionAuthorizationPayload:
    chain = store_day_chain("2026-01-01", "2026-01-04")
    return SeedProductionAuthorizationPayload(
        store_namespace_id="a" * 64,
        supersession_head_witness=SupersessionHeadWitness(
            store_namespace_id="a" * 64, line_count=0, head_sha256="b" * 64
        ),
        baseline_profile_name=_DEFAULT_PROFILE,
        resolved_section_config_hash="c" * 64,
        ordered_seed_chain_replay_days=chain,
        ordered_seed_chain_logical_trading_days=tuple(
            day for day in chain if is_logical_trading_day(day)
        ),
        snapshot_through_day=chain[-1],
        first_intended_verification_day="2026-01-05",
        expected_source_inventory_hash="d" * 64,
        quant_lab_source_identity="e" * 64,
        strategy_core_commit="f" * 40,
        strategy_core_source_identity="f" * 64,
        seed_schema_version=IFVG_SEED_SCHEMA_VERSION,
        provenance="owner_signed",
        owner_decision_refs=("21/R-5:example",),
        approved_by="example",
        approved_at="2026-09-01T00:00:00+00:00",
        effective_from="2026-09-01T00:00:00+00:00",
    )


def _example_run() -> SeedProductionRunPayload:
    chain = store_day_chain("2026-01-01", "2026-01-04")
    return SeedProductionRunPayload(
        seed_production_authorization_id="a" * 64,
        store_namespace_id="a" * 64,
        supersession_head_witness=SupersessionHeadWitness(
            store_namespace_id="a" * 64, line_count=0, head_sha256="b" * 64
        ),
        seed_snapshot_id="c" * 64,
        seed_hash="d" * 64,
        baseline_profile_name=_DEFAULT_PROFILE,
        resolved_section_config_hash="e" * 64,
        ordered_seed_chain_replay_days=chain,
        chain_replay_day_count=len(chain),
        logical_trading_day_count=2,
        snapshot_through_day=chain[-1],
        first_intended_verification_day="2026-01-05",
        access_audit_sha256="1" * 64,
        quant_lab_source_identity="2" * 64,
        strategy_core_commit="f" * 40,
        strategy_core_source_identity="3" * 64,
        provenance="owner_signed",
    )


register_identity_pair(
    name="SeedProductionAuthorization",
    envelope_cls=SeedProductionAuthorizationEnvelope,
    payload_cls=SeedProductionAuthorizationPayload,
    id_field="seed_production_authorization_id",
    example_factory=_example_authorization,
)
register_identity_pair(
    name="SeedProductionRun",
    envelope_cls=SeedProductionRunEnvelope,
    payload_cls=SeedProductionRunPayload,
    id_field="seed_production_run_id",
    example_factory=_example_run,
)
