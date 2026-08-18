"""Profile-independent opportunity lineage (CONTRACTS_AND_SCHEMAS.md §4).

Native record ids embed the profile hash (``make_setup_id(profile_hash, …)``,
verified), so two profiles never share native ids even for the same market
opportunity. Cross-profile population deltas therefore run on this layer:
lineage payloads are profile-independent, point-in-time, and source-derived
(Strategy-Core FVG ids are deterministic ``{seconds}s:{direction}:{c_bar_id}``
— bar identity + gap geometry, no profile content). Native→lineage must be
one-to-one; collisions are recorded and the entity kind becomes
``not_comparable`` — never deduped, kept-first, or fuzz-matched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd
from pydantic import Field

from ..contracts import RecordTable
from .identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
    register_identity_pair,
)

__all__ = [
    "SetupLineagePayload",
    "CandidateLineagePayload",
    "DecisionLineagePayload",
    "TradeLineagePayload",
    "LineageCollisionRecord",
    "LineageUniquenessReport",
    "LineageUniquenessEnvelope",
    "NativeLineageMap",
    "build_native_lineage_map",
    "persist_lineage_uniqueness",
    "derive_lineage_validity",
    "parse_fvg_id",
    "LINEAGE_ENTITY_KINDS",
    "ENTRY_POLICY_ID",
]

LINEAGE_ENTITY_KINDS: tuple[str, ...] = ("setup", "candidate", "decision", "trade")

ENTRY_POLICY_ID = "confirmation_close_next1m_stop_first_v1"


class SetupLineagePayload(FrozenContract):
    direction: str
    htf_timeframe_seconds: int
    htf_fvg_id: str
    activation_cursor: str


class CandidateLineagePayload(FrozenContract):
    setup_lineage_id: str
    entry_family: str
    entry_trigger_cursor: str
    entry_fvg_id: str | None
    decision_as_of_cursor: str


class DecisionLineagePayload(FrozenContract):
    candidate_lineage_id: str
    decision_kind: str
    decision_cursor: str


class TradeLineagePayload(FrozenContract):
    decision_lineage_id: str
    entry_cursor: str
    entry_policy_id: str


class LineageCollisionRecord(FrozenContract):
    core_replay_id: str
    entity_kind: str
    lineage_key: str
    native_ids: tuple[str, ...]


class LineageUniquenessReport(FrozenContract):
    core_replay_id: str
    # kind -> "one_to_one" | "collisions_present" (immutable wrapper, CS §0.3)
    per_entity_kind: ImmutableMap[str, str]
    collisions: tuple[LineageCollisionRecord, ...] = Field(default=())


class LineageUniquenessEnvelope(EnvelopeBase):
    """Immutable-store envelope: the uniqueness report ACCOMPANIES every
    replay used in a cross-profile comparison (CS §4 / DT §4.3 P1-E)."""

    _ID_FIELD: ClassVar[str] = "lineage_report_id"

    lineage_report_id: str = Field(pattern=SHA256_PATTERN)
    payload: LineageUniquenessReport


def persist_lineage_uniqueness(store_root, report: LineageUniquenessReport):
    """Publish the per-replay uniqueness report (collision records included)
    immutably into the ``lineage_reports`` store. Returns (envelope, reused)."""

    from .store import save_or_reuse_envelope  # noqa: PLC0415

    return save_or_reuse_envelope(
        store_root, "lineage_reports", LineageUniquenessEnvelope.from_payload(report)
    )


#: Axis technical keys whose change alters FVG detection or lineage-key
#: geometry — cross-profile lineage keys are NOT comparable across them.
_LINEAGE_BREAKING_KEYS: frozenset[str] = frozenset(
    {
        "min_gap_ticks_capture",
        "swing_strength_bars",
        "swing_pool_max",
        "htf_timeframes",
        "parent_timeframes",
        "entry_families",
        "entry_family",
    }
)


def derive_lineage_validity(
    changed_axis_keys: tuple[str, ...], axis_specs=None
) -> tuple[bool, str | None]:
    """(lineage_valid, reason) for a changed-axis set (DT §4.3 rules).

    An axis that changes capture artifacts (TF sets / tick geometry) or the
    entry-trigger meaning invalidates cross-profile lineage keys; the caller
    passes the verdict into ``build_lineage_population_delta`` — never a
    hand-asserted boolean.
    """

    if axis_specs is None:
        from .axis_registry import SEARCH_AXIS_REGISTRY_V1  # noqa: PLC0415

        axis_specs = SEARCH_AXIS_REGISTRY_V1
    breaking: list[str] = []
    for key in changed_axis_keys:
        spec = axis_specs.get(key)
        if key in _LINEAGE_BREAKING_KEYS or (
            spec is not None and spec.changes_capture_artifacts
        ):
            breaking.append(key)
    if breaking:
        return False, (
            "changed axes alter the lineage key derivation (gap detection / "
            f"capture geometry / entry-trigger meaning): {sorted(breaking)}"
        )
    return True, None


def parse_fvg_id(fvg_id: str) -> tuple[int, str]:
    """(timeframe_seconds, direction) from the deterministic SC fvg id.

    Format: ``{seconds}s:{direction}:{c_bar_id}`` — refuse anything else
    rather than guessing.
    """

    parts = str(fvg_id).split(":", 2)
    if len(parts) != 3 or not parts[0].endswith("s"):
        raise ValueError(f"unparseable fvg id {fvg_id!r}")
    return int(parts[0][:-1]), parts[1]


@dataclass(frozen=True)
class NativeLineageMap:
    """Native-id → lineage-id projection for ONE replay.

    ``incomplete_kinds`` names entity kinds whose exact lineage key could not
    be constructed for every native row (reason recorded); population deltas
    over those kinds are DISABLED (`not_comparable`), never inferred.
    """

    core_replay_id: str
    native_to_lineage: Mapping[str, Mapping[str, str]]
    lineage_payloads: Mapping[str, Mapping[str, FrozenContract]]
    uniqueness_report: LineageUniquenessReport
    incomplete_kinds: Mapping[str, str] = field(default_factory=dict)

    def lineage_keys(self, kind: str) -> frozenset[str]:
        return frozenset(self.native_to_lineage.get(kind, {}).values())

    def usable(self, kind: str) -> bool:
        return (
            kind not in self.incomplete_kinds
            and self.uniqueness_report.per_entity_kind.get(kind) == "one_to_one"
        )


def _iso(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _setup_sources(
    tables: Mapping[RecordTable, pd.DataFrame],
    audit_tables: Mapping[str, pd.DataFrame] | None,
) -> tuple[dict[str, dict], dict[str, str], set[str]]:
    """(setup_id → {htf_fvg_id, direction}) + activation cursors + all setups."""

    lifecycle = tables.get(RecordTable.SETUP_LIFECYCLE, pd.DataFrame())
    candidates = tables.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame())
    identity: dict[str, dict] = {}
    activation: dict[str, str] = {}
    all_setups: set[str] = set()
    if not lifecycle.empty:
        all_setups |= set(lifecycle["envelope_setup_id"].dropna().astype(str))
        activated = lifecycle.loc[lifecycle["transition"] == "setup_activated"]
        for row in activated.to_dict("records"):
            setup_id = str(row["envelope_setup_id"])
            cursor = row.get("event_cursor")
            if cursor is not None and not pd.isna(cursor):
                activation[setup_id] = str(cursor)
    if not candidates.empty and "envelope_setup_id" in candidates:
        for row in candidates.to_dict("records"):
            setup_id = str(row["envelope_setup_id"])
            all_setups.add(setup_id)
            fvg = row.get("htf_fvg_id")
            if fvg is not None and not pd.isna(fvg) and str(fvg):
                identity.setdefault(setup_id, {"htf_fvg_id": str(fvg)})
    if audit_tables:
        for name in ("ifvg_audit_htf_tap", "htf_tap"):
            taps = audit_tables.get(name)
            if taps is None or taps.empty:
                continue
            setup_col = (
                "envelope_setup_id" if "envelope_setup_id" in taps else "setup_id"
            )
            fvg_col = "fvg_fvg_id" if "fvg_fvg_id" in taps else "htf_fvg_id"
            if setup_col not in taps or fvg_col not in taps:
                continue
            for row in taps.to_dict("records"):
                setup_id = row.get(setup_col)
                fvg = row.get(fvg_col)
                if (
                    setup_id is not None
                    and not pd.isna(setup_id)
                    and str(setup_id)  # dropped taps carry an empty setup id
                    and fvg is not None
                    and not pd.isna(fvg)
                    and str(fvg)
                ):
                    identity.setdefault(str(setup_id), {"htf_fvg_id": str(fvg)})
    return identity, activation, all_setups


def build_native_lineage_map(
    tables: Mapping[RecordTable, pd.DataFrame],
    *,
    core_replay_id: str,
    audit_tables: Mapping[str, pd.DataFrame] | None = None,
) -> NativeLineageMap:
    """Project every native row onto its profile-independent lineage key.

    Sources: entry candidates (carry ``htf_fvg_id``), lifecycle activation
    cursors, and — for candidate-less setups — the optional audit companion's
    tap rows. A kind with any unconstructable key is marked incomplete
    (reason recorded); a lineage key claimed by more than one native id is a
    persisted collision. Neither is ever repaired silently.
    """

    native_to_lineage: dict[str, dict[str, str]] = {kind: {} for kind in LINEAGE_ENTITY_KINDS}
    payloads: dict[str, dict[str, FrozenContract]] = {kind: {} for kind in LINEAGE_ENTITY_KINDS}
    incomplete: dict[str, str] = {}
    collisions: list[LineageCollisionRecord] = []

    identity, activation, all_setups = _setup_sources(tables, audit_tables)

    # ── setups ───────────────────────────────────────────────────────────────
    setup_lineage_by_native: dict[str, str] = {}
    missing_identity = sorted(all_setups - set(identity))
    missing_activation = sorted(set(identity) - set(activation))
    if missing_identity:
        incomplete["setup"] = (
            f"{len(missing_identity)} setup(s) lack a source-derived HTF fvg "
            "identity (no candidate row and no audit tap evidence)"
        )
    elif missing_activation:
        incomplete["setup"] = (
            f"{len(missing_activation)} setup(s) lack an activation cursor"
        )
    for setup_id in sorted(identity):
        cursor = activation.get(setup_id)
        if cursor is None:
            continue
        fvg_id = identity[setup_id]["htf_fvg_id"]
        timeframe, direction = parse_fvg_id(fvg_id)
        payload = SetupLineagePayload(
            direction=direction,
            htf_timeframe_seconds=timeframe,
            htf_fvg_id=fvg_id,
            activation_cursor=cursor,
        )
        lineage_id = canonical_contract_sha256(payload)
        setup_lineage_by_native[setup_id] = lineage_id
        native_to_lineage["setup"][setup_id] = lineage_id
        payloads["setup"][lineage_id] = payload

    # ── candidates ───────────────────────────────────────────────────────────
    candidates = tables.get(RecordTable.ENTRY_CANDIDATE, pd.DataFrame())
    candidate_lineage_by_native: dict[str, str] = {}
    if not candidates.empty:
        for row in candidates.to_dict("records"):
            native_id = str(row["candidate_id"])
            setup_native = str(row.get("envelope_setup_id"))
            setup_lineage = setup_lineage_by_native.get(setup_native)
            trigger = row.get("trigger_cursor")
            as_of = _iso(row.get("envelope_ts_utc"))
            if setup_lineage is None or trigger is None or pd.isna(trigger) or as_of is None:
                incomplete.setdefault(
                    "candidate",
                    "candidate row(s) lack a setup lineage or exact trigger cursor",
                )
                continue
            # the candidate's own (entry) FVG rides the fvg_* projection columns
            entry_fvg = row.get("fvg_fvg_id")
            payload = CandidateLineagePayload(
                setup_lineage_id=setup_lineage,
                entry_family=str(row.get("entry_family")),
                entry_trigger_cursor=str(trigger),
                entry_fvg_id=(
                    str(entry_fvg)
                    if entry_fvg is not None and not pd.isna(entry_fvg) and str(entry_fvg)
                    else None
                ),
                decision_as_of_cursor=as_of,
            )
            lineage_id = canonical_contract_sha256(payload)
            candidate_lineage_by_native[native_id] = lineage_id
            native_to_lineage["candidate"][native_id] = lineage_id
            payloads["candidate"][lineage_id] = payload

    # ── decisions ────────────────────────────────────────────────────────────
    decisions = tables.get(RecordTable.ELIGIBLE_DECISION, pd.DataFrame())
    decision_lineage_by_native: dict[str, str] = {}
    if not decisions.empty:
        for row in decisions.to_dict("records"):
            native_id = str(row["decision_id"])
            candidate_lineage = candidate_lineage_by_native.get(
                str(row.get("candidate_id"))
            )
            cursor = _iso(row.get("envelope_ts_utc"))
            if candidate_lineage is None or cursor is None:
                incomplete.setdefault(
                    "decision", "decision row(s) lack a candidate lineage or cursor"
                )
                continue
            selected = row.get("selected")
            drop_reason = row.get("drop_reason")
            if selected is not None and not pd.isna(selected) and bool(selected):
                kind = "selected"
            elif drop_reason is not None and not pd.isna(drop_reason) and str(drop_reason):
                kind = f"dropped:{drop_reason}"
            else:
                kind = "eligible"
            payload = DecisionLineagePayload(
                candidate_lineage_id=candidate_lineage,
                decision_kind=kind,
                decision_cursor=cursor,
            )
            lineage_id = canonical_contract_sha256(payload)
            decision_lineage_by_native[native_id] = lineage_id
            native_to_lineage["decision"][native_id] = lineage_id
            payloads["decision"][lineage_id] = payload

    # ── trades ───────────────────────────────────────────────────────────────
    trades = tables.get(RecordTable.EXECUTED_TRADE, pd.DataFrame())
    if not trades.empty:
        for row in trades.to_dict("records"):
            native_id = str(row["trade_id"])
            decision_lineage = decision_lineage_by_native.get(str(row.get("decision_id")))
            entry_cursor = _iso(row.get("entry_ts_utc"))
            if decision_lineage is None or entry_cursor is None:
                incomplete.setdefault(
                    "trade", "trade row(s) lack a decision lineage or entry cursor"
                )
                continue
            payload = TradeLineagePayload(
                decision_lineage_id=decision_lineage,
                entry_cursor=entry_cursor,
                entry_policy_id=ENTRY_POLICY_ID,
            )
            lineage_id = canonical_contract_sha256(payload)
            native_to_lineage["trade"][native_id] = lineage_id
            payloads["trade"][lineage_id] = payload

    # ── one-to-one uniqueness (never dedupe / keep-first / fuzzy) ────────────
    per_kind: dict[str, str] = {}
    for kind in LINEAGE_ENTITY_KINDS:
        mapping = native_to_lineage[kind]
        reverse: dict[str, list[str]] = {}
        for native_id, lineage_id in mapping.items():
            reverse.setdefault(lineage_id, []).append(native_id)
        kind_collisions = {
            lineage_id: sorted(native_ids)
            for lineage_id, native_ids in reverse.items()
            if len(native_ids) > 1
        }
        for lineage_id, native_ids in sorted(kind_collisions.items()):
            collisions.append(
                LineageCollisionRecord(
                    core_replay_id=core_replay_id,
                    entity_kind=kind,
                    lineage_key=lineage_id,
                    native_ids=tuple(native_ids),
                )
            )
        per_kind[kind] = "collisions_present" if kind_collisions else "one_to_one"

    report = LineageUniquenessReport(
        core_replay_id=core_replay_id,
        per_entity_kind=per_kind,
        collisions=tuple(collisions),
    )
    return NativeLineageMap(
        core_replay_id=core_replay_id,
        native_to_lineage={kind: dict(v) for kind, v in native_to_lineage.items()},
        lineage_payloads={kind: dict(v) for kind, v in payloads.items()},
        uniqueness_report=report,
        incomplete_kinds=dict(incomplete),
    )


register_identity_pair(
    name="LineageUniqueness",
    envelope_cls=LineageUniquenessEnvelope,
    payload_cls=LineageUniquenessReport,
    id_field="lineage_report_id",
    example_factory=lambda: LineageUniquenessReport(
        core_replay_id="a" * 64,
        per_entity_kind={"setup": "one_to_one"},
        collisions=(),
    ),
)
