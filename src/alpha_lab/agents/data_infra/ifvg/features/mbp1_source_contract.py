"""MBP-1 source contract, ordering key, and stage-cutoff semantics.

Implements CONTRACTS_AND_SCHEMAS.md §10 and DELTA_TAXONOMY.md §6.2: the
complete deterministic total order ``(ts_event, ts_recv, sequence,
source_ordinal)``; :class:`StageEvidenceCutoff` objects (never an artificial
``+inf`` bound); per-feature :class:`WindowTriggerSemantics`; typed missing
reasons including ``same_timestamp_order_unavailable``. MBP-1 is the maximum
order-flow depth for every new contract — the deep-book guard regex enforces
it across every feature/bundle/control/registry namespace, with the single
opaque replay-provenance literal ``legacy_verified_replay_source`` exempt (and
unqueryable from this layer).

Contracts only in R1/R5; the numerical materializer ships in R5B
(research-only offline — never a live, serving, or gate feature).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ..search.identities import FrozenContract, ImmutableMap

__all__ = [
    "StageCutoffKind",
    "WindowTriggerSemantics",
    "IntervalBound",
    "StageEvidenceCutoff",
    "Mbp1FeatureWindowSpec",
    "Mbp1SourceContract",
    "MBP1_MISSING_REASONS",
    "MBP1_STAGES",
    "MBP1_SNAPSHOT_METRICS",
    "MBP1_TRANSITION_METRICS",
    "MBP1_FORMULA_VERSION",
    "MBP1_MATERIALIZER_VERSION",
    "MIN_DAY_COVERAGE_FRACTION",
    "R5B_WINDOW_SPECS",
    "mbp1_feature_names",
    "DEEP_BOOK_IDENTIFIER_REGEX",
    "DEEP_BOOK_EXEMPT_LITERALS",
    "assert_no_deep_book_identifiers",
]

#: The R5B activation's formula/materializer identities. Any change to the
#: registered formulas or the materializer semantics bumps these and mints a
#: new resolved block id (DELTA_TAXONOMY.md §6.2).
MBP1_FORMULA_VERSION = "ifvg_order_flow_mbp1_formula_v1"
MBP1_MATERIALIZER_VERSION = "mbp1_feature_materializer_v1"

#: Engineering default (DECISIONS_TAKEN R5B): a day whose sequence-gap-
#: adjusted coverage falls below this fraction marks every window of that
#: day ``coverage_below_threshold``. Unratified for research, like every
#: engineering default.
MIN_DAY_COVERAGE_FRACTION = 0.95

#: Deeper-than-MBP-1 identifier guard (acceptance §7A.19.13). Strengthened
#: beyond the plan's literal ``mbp[\W_]?(10|\d{2,})`` (safety review S2): any
#: ``mbp<N>`` token whose depth is not exactly 1 — mbp2…mbp9 included — is
#: unrepresentable, matching the §9 boundary statement ("MBP-1 is the
#: maximum") rather than only its MBP-10 example. The plan's mandated
#: pattern is a strict subset of this one.
DEEP_BOOK_IDENTIFIER_REGEX = re.compile(r"mbp[\W_]?(?!1(?!\d))\d+", re.IGNORECASE)

#: The single exemption: opaque legacy replay provenance (V3 P0-3). It is a
#: provenance literal, not a queryable feature/control identifier.
DEEP_BOOK_EXEMPT_LITERALS = ("legacy_verified_replay_source",)


def assert_no_deep_book_identifiers(names: Iterable[str]) -> None:
    offenders = [
        name
        for name in names
        if name not in DEEP_BOOK_EXEMPT_LITERALS and DEEP_BOOK_IDENTIFIER_REGEX.search(name)
    ]
    if offenders:
        raise ValueError(
            f"deeper-than-MBP-1 identifiers are unrepresentable here: {sorted(offenders)}"
        )


class StageCutoffKind(StrEnum):
    EXACT_SOURCE_ORDER_KEY = "exact_source_order_key"
    TIMESTAMP_EXCLUSIVE = "timestamp_exclusive"
    COMPLETED_BAR_BOUNDARY = "completed_bar_boundary"
    AMBIGUOUS_SAME_TIMESTAMP = "ambiguous_same_timestamp"


class WindowTriggerSemantics(StrEnum):
    PRE_TRIGGER_EXCLUSIVE = "pre_trigger_exclusive"
    POST_TRIGGER_INCLUSIVE = "post_trigger_inclusive"
    COMPLETED_BAR_AS_OF = "completed_bar_as_of"


class IntervalBound(StrEnum):
    OPEN = "open"
    CLOSED = "closed"


class StageEvidenceCutoff(FrozenContract):
    """The exact evidence boundary of one lifecycle stage — never ``+inf``."""

    stage_id: str
    stage_as_of_ts_utc: str
    cutoff_kind: StageCutoffKind
    exact_source_order_key: tuple[str, str, int, int] | None
    completed_bar_close_ts_utc: str | None
    same_timestamp_policy_id: str
    source_evidence_ref: str | None

    @model_validator(mode="after")
    def _kind_fields_agree(self):
        if (
            self.cutoff_kind is StageCutoffKind.EXACT_SOURCE_ORDER_KEY
            and self.exact_source_order_key is None
        ):
            raise ValueError("exact cutoff requires the stage-triggering event key")
        if (
            self.cutoff_kind is StageCutoffKind.COMPLETED_BAR_BOUNDARY
            and self.completed_bar_close_ts_utc is None
        ):
            raise ValueError("completed-bar cutoff requires the bar close ts")
        if self.exact_source_order_key is not None:
            # review F6: BOTH timestamp elements of the exact key are decimal
            # nanosecond strings — an artificial infinity in either position
            # is unrepresentable (the withdrawn +inf rule can never resurface)
            for element in self.exact_source_order_key[:2]:
                if "inf" in element.lower() or not element.lstrip("-").isdigit():
                    raise ValueError(
                        "an artificial +inf (or non-numeric) cutoff bound is "
                        "unrepresentable"
                    )
        return self


class Mbp1FeatureWindowSpec(FrozenContract):
    feature_window_key: str
    feature_names: tuple[str, ...]
    from_stage: str | None
    to_stage: str
    lower_bound: IntervalBound
    upper_bound: IntervalBound
    trigger_semantics: WindowTriggerSemantics
    cutoff_policy_id: str
    minimum_event_count: int = Field(ge=0)
    missingness_policy_id: str

    @model_validator(mode="after")
    def _names_shallow(self):
        assert_no_deep_book_identifiers(self.feature_names)
        assert_no_deep_book_identifiers((self.feature_window_key,))
        return self


MBP1_MISSING_REASONS: tuple[str, ...] = (
    "no_mbp1_partition",
    "sequence_gap",
    "coverage_below_threshold",
    "stage_outside_coverage",
    "instrument_roll_boundary",
    "same_timestamp_order_unavailable",
    "minimum_event_count_not_met",
)

MBP1_STAGES: tuple[str, ...] = ("htf_tap", "parent_lock", "opposing", "inversion", "entry")

MBP1_SNAPSHOT_METRICS: tuple[str, ...] = (
    "spread_ticks",
    "queue_imbalance",
    "order_count_imbalance",
    "microprice_offset_ticks",
    "bid_sz",
    "ask_sz",
    "bid_ct",
    "ask_ct",
)

MBP1_TRANSITION_METRICS: tuple[str, ...] = (
    "ofi_sum",
    "aggressive_buy_frac",
    "aggressive_sell_frac",
    "depletion_events",
    "replenishment_events",
    "absorption_score",
    "event_count",
    "quote_intensity",
    "trade_intensity",
)


def mbp1_feature_names() -> tuple[str, ...]:
    """Stage snapshots + stage-transition aggregates (DT §6.2 naming)."""

    snapshots = tuple(
        f"ofl_snap_{stage}_{metric}"
        for stage in MBP1_STAGES
        for metric in MBP1_SNAPSHOT_METRICS
    )
    transitions = tuple(
        f"ofl_win_{from_stage}_{to_stage}_{metric}"
        for from_stage, to_stage in zip(MBP1_STAGES[:-1], MBP1_STAGES[1:], strict=True)
        for metric in MBP1_TRANSITION_METRICS
    )
    return (*snapshots, *transitions)


def _snapshot_specs() -> tuple[Mbp1FeatureWindowSpec, ...]:
    return tuple(
        Mbp1FeatureWindowSpec(
            feature_window_key=f"ofl_snap_{stage}",
            feature_names=tuple(
                f"ofl_snap_{stage}_{metric}" for metric in MBP1_SNAPSHOT_METRICS
            ),
            from_stage=None,
            to_stage=stage,
            lower_bound=IntervalBound.OPEN,
            upper_bound=IntervalBound.CLOSED,
            trigger_semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE,
            cutoff_policy_id="stage_evidence_cutoff_v2",
            minimum_event_count=1,
            missingness_policy_id="typed_null_preserve_row_v1",
        )
        for stage in MBP1_STAGES
    )


def _transition_specs() -> tuple[Mbp1FeatureWindowSpec, ...]:
    return tuple(
        Mbp1FeatureWindowSpec(
            feature_window_key=f"ofl_win_{from_stage}_{to_stage}",
            feature_names=tuple(
                f"ofl_win_{from_stage}_{to_stage}_{metric}"
                for metric in MBP1_TRANSITION_METRICS
            ),
            from_stage=from_stage,
            to_stage=to_stage,
            lower_bound=IntervalBound.OPEN,
            upper_bound=IntervalBound.CLOSED,
            trigger_semantics=WindowTriggerSemantics.POST_TRIGGER_INCLUSIVE,
            cutoff_policy_id="stage_evidence_cutoff_v2",
            minimum_event_count=1,
            missingness_policy_id="typed_null_preserve_row_v1",
        )
        for from_stage, to_stage in zip(MBP1_STAGES[:-1], MBP1_STAGES[1:], strict=True)
    )


#: The frozen initial R5B window registry (DT §6.2): a stage snapshot is the
#: top-of-book state when the stage became observable; a transition aggregate
#: covers ``(from_stage, to_stage]``. Changing any of this mints a new
#: ``FeatureBlockResolutionPayload`` and resolved block id.
R5B_WINDOW_SPECS: tuple[Mbp1FeatureWindowSpec, ...] = (
    *_snapshot_specs(),
    *_transition_specs(),
)


class Mbp1SourceContract(FrozenContract):
    order_flow_depth_policy: Literal["mbp1_only_v1"] = "mbp1_only_v1"
    vendor: Literal["databento"] = "databento"
    schema_name: Literal["mbp-1"] = Field(default="mbp-1", alias="schema")
    instrument: str
    contract_roll_policy_id: str
    event_time_field: Literal["ts_event"] = "ts_event"
    receive_time_field: Literal["ts_recv"] = "ts_recv"
    sequence_field: Literal["sequence"] = "sequence"
    sequence_ordering_policy: Literal["ts_event_ts_recv_sequence_source_ordinal_v1"] = (
        "ts_event_ts_recv_sequence_source_ordinal_v1"
    )
    stage_cutoff_contract: Literal["stage_evidence_cutoff_v2"] = "stage_evidence_cutoff_v2"
    feature_window_specs: tuple[Mbp1FeatureWindowSpec, ...]
    source_partitions: str = "data/databento/NQ/<date>/mbp1.parquet"
    coverage_policy: ImmutableMap[str, float]
    gap_semantics: str = "sequence_gap_marks_interval_invalid_v1"
    trades_derivation: Literal["trades_from_mbp1_d_p_17"] = "trades_from_mbp1_d_p_17"

    model_config = FrozenContract.model_config | {"populate_by_name": True}
