"""Decomposed replay identity stack for the FSM configuration search lane.

Implements ``CONTRACTS_AND_SCHEMAS.md`` §0–§1 of
``ifvg_prop_robust_config_search_v1``:

* the universal non-self-referential **Payload/Envelope** convention plus the
  identity-projection registry the audit test enumerates (§0.1);
* the **deep-immutability** helpers — :class:`ImmutableMap` and
  :func:`deep_freeze` (§0.3);
* the content-addressed :class:`ReplayInputBundlePayload` with exact physical
  source partitions, the deterministic preflight
  :class:`ReplayAccessAuthorizationRef`, and the runtime
  :class:`ReplayExecutionAccessAudit` that never enters identity (§1.1);
* :class:`CoreStrategyReplayIdentity` — the reusable scientific replay
  identity, decoupled from study membership, cost, prop, audit, chart, and
  resource concerns (§1.2);
* canonical, study-independent semantic profile naming (§1.3);
* :class:`GeneratedProfileCapability` — the capability contract for generated
  child profiles; the fixed ``PROFILE_CAPABILITY_REGISTRY`` gates baselines
  only (§1.4);
* membership, companion, and costed-evaluation identities (§1.5).
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from pydantic_core import core_schema
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

from ..context_experiment_contracts import (
    PROFILE_CAPABILITY_REGISTRY,
    ProfileCapabilityStatus,
    canonical_contract_sha256,
)
from ..manifest import read_repository_state

__all__ = [
    "FrozenContract",
    "ImmutableMap",
    "deep_freeze",
    "canonical_contract_sha256",
    "RegisteredIdentityPair",
    "ID_PRODUCING_CONTRACTS",
    "register_identity_pair",
    "registered_identity_pairs",
    "EnvelopeBase",
    "SHA256_PATTERN",
    "ReplaySourcePartitionRef",
    "ReplayDayArtifactRef",
    "ReplayAccessAuthorizationRef",
    "ReplayInputBundlePayload",
    "ReplayInputBundleEnvelope",
    "ReplayExecutionAccessAudit",
    "build_replay_input_bundle",
    "CoreStrategyReplayPayload",
    "CoreStrategyReplayIdentity",
    "RESOLVER_POLICY_V1",
    "name_free_section_hash",
    "canonical_profile_id_for",
    "canonicalize_section",
    "is_generated_profile_id",
    "QL_REPLAY_SOURCE_SCOPE",
    "quant_lab_replay_source_identity",
    "strategy_core_source_identity",
    "ResolvedSearchProfileRef",
    "GeneratedProfileCapability",
    "evaluate_generated_profile_capability",
    "SearchChildMembership",
    "CoreReplayArtifactReference",
    "FsmAuditArtifactIdentity",
    "ReplayChartArtifactIdentity",
    "CostedEvaluationIdentity",
]

SHA256_PATTERN = r"^[0-9a-f]{64}$"

RESOLVER_POLICY_V1 = "next_1m_bar_stop_first_v1"

_GENERATED_PROFILE_PREFIX = "ifvg_search_profile_"


# ─────────────────────────────────────────────────────────────────────────────
# §0.3 deep immutability
# ─────────────────────────────────────────────────────────────────────────────


class ImmutableMap(Mapping):
    """Read-only, canonically sorted mapping wrapper (deep-copied on build).

    Serializes as a tuple of ``(key, value)`` records sorted by the canonical
    JSON form of the key, so identity hashing over it is deterministic and a
    caller can never mutate shared nested state after identity calculation.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping | Iterable[tuple[Any, Any]] = ()) -> None:
        if isinstance(data, Mapping):  # ImmutableMap is itself a Mapping
            items = list(data.items())
        else:
            items = [(key, value) for key, value in data]
        materialized = {}
        for key, value in items:
            if key in materialized:
                raise ValueError(f"ImmutableMap received duplicate key {key!r}")
            materialized[key] = copy.deepcopy(value)
        object.__setattr__(
            self,
            "_data",
            dict(sorted(materialized.items(), key=lambda kv: _canonical_key(kv[0]))),
        )

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"ImmutableMap({self._data!r})"

    def __eq__(self, other) -> bool:
        if isinstance(other, ImmutableMap):
            return self._data == other._data
        if isinstance(other, Mapping):
            return self._data == dict(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple((key, _try_hashable(value)) for key, value in self._data.items()))

    @classmethod
    def __class_getitem__(cls, item):
        # Preserve the typed alias so pydantic can validate keys and values.
        from typing import _GenericAlias  # noqa: PLC0415

        return _GenericAlias(cls, item if isinstance(item, tuple) else (item,))

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        args = get_args(source_type)
        if args and len(args) == 2:
            keys_schema = handler.generate_schema(args[0])
            values_schema = handler.generate_schema(args[1])
        else:
            keys_schema = core_schema.any_schema()
            values_schema = core_schema.any_schema()
        dict_schema = core_schema.dict_schema(keys_schema, values_schema)

        def _coerce(value):
            if isinstance(value, ImmutableMap):
                return dict(value)
            if isinstance(value, Mapping):
                return dict(value)
            if isinstance(value, (list, tuple)):
                materialized: dict = {}
                for key, item in value:
                    if key in materialized:  # same fail-closed rule as __init__
                        raise ValueError(
                            f"ImmutableMap received duplicate key {key!r}"
                        )
                    materialized[key] = item
                return materialized
            return value

        def _wrap(value):
            return cls(value)

        def _serialize(value: ImmutableMap, serializer, info):
            dumped = serializer(dict(value))
            return tuple(
                (key, dumped[key])
                for key in sorted(dumped, key=_canonical_key)
            )

        return core_schema.no_info_after_validator_function(
            _wrap,
            core_schema.no_info_before_validator_function(_coerce, dict_schema),
            serialization=core_schema.wrap_serializer_function_ser_schema(
                _serialize, schema=dict_schema, info_arg=True
            ),
        )


def _canonical_key(key: Any) -> str:
    return json.dumps(key, sort_keys=True, default=str)


def _try_hashable(value: Any):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def deep_freeze(value: Any) -> Any:
    """Freeze an ``Any``-typed payload into canonical immutable structures.

    dicts become canonically sorted tuples of ``(key, value)`` pairs, lists
    and tuples become tuples, sets are refused (no canonical order), scalars
    pass through. The result serializes natively and cannot be mutated after
    identity calculation.
    """

    if isinstance(value, ImmutableMap):
        return tuple((key, deep_freeze(item)) for key, item in value.items())
    if isinstance(value, Mapping):
        return tuple(
            (key, deep_freeze(item))
            for key, item in sorted(value.items(), key=lambda kv: _canonical_key(kv[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise ValueError("sets have no canonical order; use a sorted tuple")
    return value


# ─────────────────────────────────────────────────────────────────────────────
# §0.1 Payload/Envelope convention + identity-projection registry
# ─────────────────────────────────────────────────────────────────────────────


class FrozenContract(BaseModel):
    """Boundary-spec base: frozen pydantic v2, unknown fields forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EnvelopeBase(FrozenContract):
    """An identity envelope: ``<id_field> = canonical_contract_sha256(payload)``.

    Subclasses declare ``_ID_FIELD``, the id field itself, and a typed
    ``payload`` field. The id is validated on every construction, so a stored
    envelope whose payload was tampered with fails closed on reload.
    """

    _ID_FIELD: ClassVar[str] = "id"

    @model_validator(mode="after")
    def _envelope_id_hashes_payload(self):
        expected = canonical_contract_sha256(self.payload)
        actual = getattr(self, type(self)._ID_FIELD)
        if actual != expected:
            raise ValueError(
                f"{type(self).__name__}.{type(self)._ID_FIELD} does not hash its payload"
            )
        return self

    @classmethod
    def from_payload(cls, payload, **extra):
        return cls(
            **{cls._ID_FIELD: canonical_contract_sha256(payload)},
            payload=payload,
            **extra,
        )


#: Field names that may never appear on a hashed identity payload (§0.1):
#: derived self-ids, display metadata, annotations, or execution-attempt facts.
_FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "display_name",
        "display_metadata",
        "annotation",
        "annotations",
        "notes",
        "attempt_number",
        "operational_retry_reason",
        "host_environment",
        "started_at",
        "ended_at",
        "writer_pid",
    }
)


@dataclass(frozen=True)
class RegisteredIdentityPair:
    """One ID-producing contract for the identity-projection audit."""

    name: str
    envelope_cls: type
    payload_cls: type
    id_field: str
    example_factory: Callable[[], Any] | None = None
    #: post-materialization facts the ENVELOPE may carry beyond (id, payload)
    extra_envelope_fields: tuple[str, ...] = ()


ID_PRODUCING_CONTRACTS: list[RegisteredIdentityPair] = []


def register_identity_pair(
    *,
    name: str,
    envelope_cls: type,
    payload_cls: type,
    id_field: str,
    example_factory: Callable[[], Any] | None = None,
    extra_envelope_fields: tuple[str, ...] = (),
) -> None:
    if any(entry.name == name for entry in ID_PRODUCING_CONTRACTS):
        raise ValueError(f"identity pair {name!r} registered twice")
    payload_fields = set(payload_cls.model_fields)
    if id_field in payload_fields:
        raise ValueError(f"{name}: payload contains its own derived id {id_field!r}")
    forbidden = sorted(payload_fields & _FORBIDDEN_PAYLOAD_FIELDS)
    if forbidden:
        raise ValueError(f"{name}: payload carries non-semantic fields {forbidden}")
    envelope_fields = set(envelope_cls.model_fields)
    expected = {id_field, "payload", *extra_envelope_fields}
    if envelope_fields != expected:
        raise ValueError(
            f"{name}: envelope fields {sorted(envelope_fields)} != declared {sorted(expected)}"
        )
    ID_PRODUCING_CONTRACTS.append(
        RegisteredIdentityPair(
            name=name,
            envelope_cls=envelope_cls,
            payload_cls=payload_cls,
            id_field=id_field,
            example_factory=example_factory,
            extra_envelope_fields=extra_envelope_fields,
        )
    )


def registered_identity_pairs() -> tuple[RegisteredIdentityPair, ...]:
    """Import every contract-bearing lane module, then list the registry."""

    from alpha_lab.propsim import (  # noqa: F401, PLC0415
        account,
        contract_evidence,
        firm_contracts,
        risk,
        simulation,
        trade_path,
        withdrawal,
    )

    from .. import fsm_audit_preparation  # noqa: F401, PLC0415
    from ..features import (  # noqa: F401, PLC0415
        bundle_feature_view,
        feature_blocks,
        feature_bundles,
        mbp1_coverage,
        mbp1_coverage_diagnostic,
        mbp1_coverage_evidence,
        mbp1_feature_materializer,
        mbp1_source_artifact,
    )
    from ..ml import (  # noqa: F401, PLC0415
        controlled_feature_study,
        decision_policies,
        regime_contracts,
    )
    from ..study import cohort, comparison_contracts, contrasts, study_cell  # noqa: F401, PLC0415
    from . import (  # noqa: F401, PLC0415
        authorization,
        charter,
        child_replay,
        lineage,
        orchestrator,
        pipeline,
        verification,
    )

    return tuple(ID_PRODUCING_CONTRACTS)


# ─────────────────────────────────────────────────────────────────────────────
# §1.1 ReplayInputBundle — content-addressing the complete replay input
# ─────────────────────────────────────────────────────────────────────────────


class ReplaySourcePartitionRef(FrozenContract):
    """One exact physical source partition contributing to one trading day."""

    source_partition_id: str
    source_partition_utc_date: str
    source_manifest_id: str | None
    relative_logical_partition_key: str
    trading_day: str
    instrument: str
    contract_symbol: str | None
    source_kind: Literal["mbp1", "trades", "legacy_verified_replay_source"]
    source_schema_id: str
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)


class ReplayDayArtifactRef(FrozenContract):
    trading_day: str
    artifact_kind: Literal["bars", "levels", "calendar", "other_required_input"]
    artifact_id: str
    manifest_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    content_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ReplayAccessAuthorizationRef(FrozenContract):
    """Deterministic preflight authorization — MAY enter replay identity."""

    access_policy_id: str
    authorized_date_set_id: str
    expected_source_inventory_hash: str = Field(pattern=SHA256_PATTERN)


def _partition_sort_key(ref: ReplaySourcePartitionRef) -> tuple[str, str, str, str, str]:
    return (
        ref.trading_day,
        ref.source_partition_utc_date,
        ref.relative_logical_partition_key,
        ref.source_kind,
        ref.instrument,
    )


def _day_artifact_sort_key(ref: ReplayDayArtifactRef) -> tuple[str, str]:
    return (ref.trading_day, ref.artifact_kind)


class ReplayInputBundlePayload(FrozenContract):
    """The exact bytes a replay consumed (= ReplayInputContentIdentity)."""

    authorized_date_set_id: str
    ordered_source_partitions: tuple[ReplaySourcePartitionRef, ...]
    ordered_day_artifacts: tuple[ReplayDayArtifactRef, ...]
    source_contract_id: str
    source_schema_era_id: str
    access_authorization: ReplayAccessAuthorizationRef

    @model_validator(mode="after")
    def _canonically_ordered(self):
        partition_keys = [_partition_sort_key(ref) for ref in self.ordered_source_partitions]
        if len(set(partition_keys)) != len(partition_keys):
            raise ValueError(
                "replay input bundle has source partitions with identical canonical "
                "keys — the ordering is not canonicalizable; refuse instead of guessing"
            )
        if partition_keys != sorted(partition_keys):
            raise ValueError("source partitions are not in canonical order")
        artifact_keys = [_day_artifact_sort_key(ref) for ref in self.ordered_day_artifacts]
        if len(set(artifact_keys)) != len(artifact_keys):
            raise ValueError(
                "replay input bundle has day artifacts with identical canonical keys"
            )
        if artifact_keys != sorted(artifact_keys):
            raise ValueError("day artifacts are not in canonical order")
        for ref in self.ordered_source_partitions:
            for value in (ref.source_partition_id, ref.relative_logical_partition_key):
                if value.startswith(("/", "\\")) or (len(value) > 1 and value[1] == ":"):
                    raise ValueError("absolute paths may not enter a replay input bundle")
        return self


class ReplayInputBundleEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "replay_input_bundle_id"

    replay_input_bundle_id: str = Field(pattern=SHA256_PATTERN)
    payload: ReplayInputBundlePayload


def build_replay_input_bundle(
    *,
    authorized_date_set_id: str,
    source_partitions: Iterable[ReplaySourcePartitionRef],
    day_artifacts: Iterable[ReplayDayArtifactRef],
    source_contract_id: str,
    source_schema_era_id: str,
    access_authorization: ReplayAccessAuthorizationRef,
) -> ReplayInputBundleEnvelope:
    """Normalize reorder-only inputs to canonical order, then mint the bundle."""

    payload = ReplayInputBundlePayload(
        authorized_date_set_id=authorized_date_set_id,
        ordered_source_partitions=tuple(
            sorted(source_partitions, key=_partition_sort_key)
        ),
        ordered_day_artifacts=tuple(sorted(day_artifacts, key=_day_artifact_sort_key)),
        source_contract_id=source_contract_id,
        source_schema_era_id=source_schema_era_id,
        access_authorization=access_authorization,
    )
    return ReplayInputBundleEnvelope.from_payload(payload)


class ReplayExecutionAccessAudit(FrozenContract):
    """RUNTIME access evidence for ONE attempt — never enters replay identity."""

    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    execution_attempt_ref: str
    event_chain_sha256: str = Field(pattern=SHA256_PATTERN)
    counters: ImmutableMap[str, int]

    def assert_zero_protected(self) -> None:
        violations = {key: value for key, value in self.counters.items() if value}
        if violations:
            raise AssertionError(f"protected source access counters non-zero: {violations}")


# ─────────────────────────────────────────────────────────────────────────────
# §1.3 canonical semantic profile naming
# ─────────────────────────────────────────────────────────────────────────────


def name_free_section_hash(section: IfvgSmcSection) -> str:
    """SHA-256 of the section content with ``profile_name`` removed.

    Mirrors ``ifvg_profile_hash`` serialization exactly (sorted compact JSON of
    ``model_dump(mode="json")``) so the only difference is the name removal.
    """

    payload = section.model_dump(mode="json")
    payload.pop("profile_name")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def canonical_profile_id_for(section: IfvgSmcSection) -> str:
    """Study-independent canonical profile id for a generated child section."""

    return f"{_GENERATED_PROFILE_PREFIX}{name_free_section_hash(section)[:16]}"


def is_generated_profile_id(profile_id: str) -> bool:
    return profile_id.startswith(_GENERATED_PROFILE_PREFIX)


@lru_cache(maxsize=1)
def _registered_baseline_name_free_hashes() -> dict[str, str]:
    """name-free section hash → registered profile name, for every baseline."""

    from ..profiles import resolve_profile_config  # noqa: PLC0415

    hashes: dict[str, str] = {}
    for profile_name in PROFILE_CAPABILITY_REGISTRY:
        resolved = resolve_profile_config({"profile_name": profile_name})
        hashes[name_free_section_hash(resolved.section)] = profile_name
    return hashes


def canonicalize_section(section: IfvgSmcSection) -> IfvgSmcSection:
    """Re-validate the section under its canonical, content-derived name.

    The name-free content is compared against EVERY registered baseline (not
    just the one whose name the section happens to carry): content identical
    to a registered baseline adopts that baseline's registered name; any
    other content gets the canonical study-independent
    ``ifvg_search_profile_<hash16>`` id. Two different names for identical
    semantics, and one baseline name for different semantics, are both
    impossible (CS §1.3).
    """

    content_hash = name_free_section_hash(section)
    registered_name = _registered_baseline_name_free_hashes().get(content_hash)
    if registered_name is not None:
        if section.profile_name == registered_name:
            return section
        return IfvgSmcSection.model_validate(
            {**section.model_dump(mode="json"), "profile_name": registered_name}
        )
    canonical = canonical_profile_id_for(section)
    if section.profile_name == canonical:
        return section
    return IfvgSmcSection.model_validate(
        {**section.model_dump(mode="json"), "profile_name": canonical}
    )


# ─────────────────────────────────────────────────────────────────────────────
# §1.2 CoreStrategyReplayIdentity + scoped source identities
# ─────────────────────────────────────────────────────────────────────────────

#: The Quant-Lab code surface that controls profile resolution, the capture
#: driver, the dataset chain, table partitioning, seed chaining, and resolver
#: invocation. Changing any of it changes every downstream core replay id.
QL_REPLAY_SOURCE_SCOPE: tuple[str, ...] = (
    "src/alpha_lab/agents/data_infra/ifvg/capture_driver.py",
    "src/alpha_lab/agents/data_infra/ifvg/config.py",
    "src/alpha_lab/agents/data_infra/ifvg/contracts.py",
    "src/alpha_lab/agents/data_infra/ifvg/data_access.py",
    "src/alpha_lab/agents/data_infra/ifvg/dataset.py",
    "src/alpha_lab/agents/data_infra/ifvg/day_artifacts.py",
    "src/alpha_lab/agents/data_infra/ifvg/development_access.py",
    "src/alpha_lab/agents/data_infra/ifvg/entry_dataset.py",
    "src/alpha_lab/agents/data_infra/ifvg/profiles.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/child_replay.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/identities.py",
)


def quant_lab_replay_source_identity(
    *,
    repository_root: Path | None = None,
    source_paths: tuple[str, ...] = QL_REPLAY_SOURCE_SCOPE,
) -> str:
    """Scoped QL replay-source identity (commit + dirty evidence + tree hash)."""

    root = repository_root or Path(__file__).resolve().parents[6]
    state = read_repository_state("quant-lab-replay-scope", root, source_paths=source_paths)
    return canonical_contract_sha256(state.identity_payload())


def strategy_core_source_identity(*, repository_root: Path) -> tuple[str, str]:
    """(commit, scoped source identity) for the installed Strategy-Core."""

    state = read_repository_state(
        "strategy-core", repository_root, source_paths=("src/strategy_core",)
    )
    return state.head, canonical_contract_sha256(state.identity_payload())


class CoreStrategyReplayPayload(FrozenContract):
    """Replay-defining fields ONLY (test-enforced §1.2 identity rule)."""

    replay_input_bundle_id: str = Field(pattern=SHA256_PATTERN)
    quant_lab_replay_source_identity: str = Field(pattern=SHA256_PATTERN)
    strategy_core_commit: str
    strategy_core_source_identity: str = Field(pattern=SHA256_PATTERN)
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    canonical_profile_id: str
    warmup_seed_identity: str
    resolver_policy: Literal["next_1m_bar_stop_first_v1"] = RESOLVER_POLICY_V1
    anchor_policy: str
    capture_schema_version: int
    record_schema_version: int


class CoreStrategyReplayIdentity(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "core_replay_id"

    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    payload: CoreStrategyReplayPayload


# ─────────────────────────────────────────────────────────────────────────────
# §1.4 GeneratedProfileCapability
# ─────────────────────────────────────────────────────────────────────────────


class ResolvedSearchProfileRef(FrozenContract):
    canonical_profile_id: str
    baseline_profile_id: str
    resolved_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    axis_value_ids: ImmutableMap[str, str]
    owner_authorization_id: str
    profile_capability_id: str


class GeneratedProfileCapability(FrozenContract):
    capability_id: str
    status: Literal[
        "generated_runnable",
        "blocked_owner_decision",
        "blocked_invalid_section",
        "blocked_invariant_failure",
        "blocked_base_profile",
    ]
    baseline_capability_ref: str
    registry_hash: str = Field(pattern=SHA256_PATTERN)
    authorization_ref: str
    validation_report_ref: str | None
    reason: str | None

    @model_validator(mode="after")
    def _blocked_requires_reason(self):
        if self.status != "generated_runnable" and not self.reason:
            raise ValueError("a blocked generated profile requires a reason")
        return self


def evaluate_generated_profile_capability(
    *,
    baseline_profile_id: str,
    axis_value_ids: Mapping[str, str],
    registry_hash: str,
    authorization_ref: str,
    authorization_state: Literal["authorized", "missing", "not_required"],
    section: IfvgSmcSection | None,
    section_error: str | None = None,
    invariant_violations: tuple[str, ...] = (),
) -> GeneratedProfileCapability:
    """Fail-closed capability for one generated child profile.

    Eligibility (§1.4): runnable baseline + registered values authorized for
    the run scope + intact locked invariants + a valid resolved section under
    deterministic canonical naming. ``invariant_violations`` names any
    LOCKED_INVARIANT section field the caller found changed vs the resolved
    baseline — non-empty blocks the child (`blocked_invariant_failure`).
    Generated children are never inserted into the fixed registry and never
    fail merely for being absent from it.
    """

    frozen_values = ImmutableMap(axis_value_ids)
    base = PROFILE_CAPABILITY_REGISTRY.get(baseline_profile_id)
    payload = {
        "baseline_profile_id": baseline_profile_id,
        "axis_value_ids": deep_freeze(dict(frozen_values)),
        "registry_hash": registry_hash,
        "authorization_ref": authorization_ref,
    }
    capability_id = canonical_contract_sha256(payload)

    def _blocked(status: str, reason: str) -> GeneratedProfileCapability:
        return GeneratedProfileCapability(
            capability_id=capability_id,
            status=status,  # type: ignore[arg-type]
            baseline_capability_ref=baseline_profile_id,
            registry_hash=registry_hash,
            authorization_ref=authorization_ref,
            validation_report_ref=None,
            reason=reason,
        )

    if base is None or base.status is not ProfileCapabilityStatus.RUNNABLE:
        reason = (
            f"baseline {baseline_profile_id!r} is not a runnable registered baseline"
            if base is None
            else f"baseline is {base.status.value}: {base.reason}"
        )
        return _blocked("blocked_base_profile", reason)
    if authorization_state == "missing":
        return _blocked(
            "blocked_owner_decision",
            "one or more axis values lack owner authorization for this run scope",
        )
    if section is None:
        return _blocked(
            "blocked_invalid_section",
            section_error or "resolved section failed validation",
        )
    if invariant_violations:
        return _blocked(
            "blocked_invariant_failure",
            "locked correctness invariants changed vs the resolved baseline: "
            + ", ".join(sorted(invariant_violations)),
        )
    expected = canonical_profile_id_for(section)
    if section.profile_name != expected:
        return _blocked(
            "blocked_invalid_section",
            f"section name {section.profile_name!r} is not the canonical id {expected!r}",
        )
    return GeneratedProfileCapability(
        capability_id=capability_id,
        status="generated_runnable",
        baseline_capability_ref=baseline_profile_id,
        registry_hash=registry_hash,
        authorization_ref=authorization_ref,
        validation_report_ref=None,
        reason=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# §1.5 membership, companions, costed evaluation
# ─────────────────────────────────────────────────────────────────────────────


class SearchChildMembership(FrozenContract):
    """Parent-study linkage; never part of the core replay identity."""

    parent_search_id: str = Field(pattern=SHA256_PATTERN)
    child_ordinal: int = Field(ge=0)
    axis_value_ids: ImmutableMap[str, str]
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    comparison_role: Literal["baseline", "challenger", "neighbor"]


class CoreReplayArtifactReference(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    v2_dataset_artifact_id: str = Field(pattern=SHA256_PATTERN)
    manifest_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    gross_trade_stream_hash: str = Field(pattern=SHA256_PATTERN)


class FsmAuditArtifactIdentity(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    audit_schema_version: int
    audit_contract_fingerprint: str
    neutrality_mechanism_id: str


class ReplayChartArtifactIdentity(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    replay_chart_schema_version: int
    range_policy_id: str
    stage_gating_policy_id: str


class CostedEvaluationIdentity(FrozenContract):
    core_replay_id: str = Field(pattern=SHA256_PATTERN)
    cost_policy_sha256: str = Field(pattern=SHA256_PATTERN)


# ─────────────────────────────────────────────────────────────────────────────
# registry entries + audit example factories
# ─────────────────────────────────────────────────────────────────────────────


def _example_partition() -> ReplaySourcePartitionRef:
    return ReplaySourcePartitionRef(
        source_partition_id="databento/NQ/2026-06-04/mbp1",
        source_partition_utc_date="2026-06-04",
        source_manifest_id=None,
        relative_logical_partition_key="day_utc_date/mbp1",
        trading_day="2026-06-04",
        instrument="NQ",
        contract_symbol="NQM6",
        source_kind="mbp1",
        source_schema_id="databento_mbp1_parquet_v1",
        content_sha256="a" * 64,
        byte_size=128,
        row_count=64,
    )


def _example_input_bundle_payload() -> ReplayInputBundlePayload:
    return ReplayInputBundlePayload(
        authorized_date_set_id="example_date_set_v1",
        ordered_source_partitions=(_example_partition(),),
        ordered_day_artifacts=(
            ReplayDayArtifactRef(
                trading_day="2026-06-04",
                artifact_kind="bars",
                artifact_id="ifvg_tbars_deadbeef",
                manifest_payload_sha256="b" * 64,
                content_sha256="c" * 64,
            ),
        ),
        source_contract_id="databento_nq_v1",
        source_schema_era_id="mbp1_era_v1",
        access_authorization=ReplayAccessAuthorizationRef(
            access_policy_id="verification_fixed_allowlist_max5_v1",
            authorized_date_set_id="example_date_set_v1",
            expected_source_inventory_hash="d" * 64,
        ),
    )


def _example_core_replay_payload() -> CoreStrategyReplayPayload:
    return CoreStrategyReplayPayload(
        replay_input_bundle_id=canonical_contract_sha256(_example_input_bundle_payload()),
        quant_lab_replay_source_identity="e" * 64,
        strategy_core_commit="f" * 40,
        strategy_core_source_identity="f" * 64,
        resolved_section_config_hash="0" * 64,
        canonical_profile_id="ifvg_v2_doc_default_fresh_static_1r",
        warmup_seed_identity="cold_start_v1",
        anchor_policy="trading_day_18et_elapsed_v1",
        capture_schema_version=2,
        record_schema_version=2,
    )


register_identity_pair(
    name="ReplayInputBundle",
    envelope_cls=ReplayInputBundleEnvelope,
    payload_cls=ReplayInputBundlePayload,
    id_field="replay_input_bundle_id",
    example_factory=_example_input_bundle_payload,
)
register_identity_pair(
    name="CoreStrategyReplay",
    envelope_cls=CoreStrategyReplayIdentity,
    payload_cls=CoreStrategyReplayPayload,
    id_field="core_replay_id",
    example_factory=_example_core_replay_payload,
)
