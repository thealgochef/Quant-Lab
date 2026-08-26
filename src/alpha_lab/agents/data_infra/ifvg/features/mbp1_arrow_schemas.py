"""Exact MBP-1 Arrow schemas and schema hashes (R5B deliverable 2).

Four pinned schemas govern the offline MBP-1 lane:

* the raw Databento ``mbp-1`` parquet source contract (``ts_recv`` retained —
  the R5B materializer decodes it directly from parquet; no Strategy-Core
  change),
* the normalized event working schema the materializer computes over
  (prices already converted to ticks under the pinned scale policy, the
  deterministic ``source_ordinal`` final tie-break appended),
* the materialized candidate feature table (76 metric columns + per-window
  validity/missing-reason evidence), and
* the per-candidate stage-window evidence table (exact cutoffs, admitted
  counts, typed reasons).

Every schema hash is the canonical hash of the ordered ``(name, type)``
pairs, so any field or type change mints a new resolved block identity
(DELTA_TAXONOMY.md §6.2). MBP-1 is the maximum representable order-flow
depth — the deep-book guard runs over every field name at import.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import pyarrow as pa

from ..search.identities import canonical_contract_sha256
from .mbp1_source_contract import (
    MBP1_STAGES,
    assert_no_deep_book_identifiers,
    mbp1_feature_names,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

__all__ = [
    "PRICE_SCALE_POLICY_ID",
    "DATABENTO_PRICE_SCALE",
    "INSTRUMENT_TICK_SIZES",
    "MBP1_SOURCE_EVENT_SCHEMA",
    "MBP1_NORMALIZED_EVENT_SCHEMA",
    "MBP1_FEATURE_TABLE_SCHEMA",
    "MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA",
    "MBP1_SOURCE_EVENT_SCHEMA_HASH",
    "MBP1_NORMALIZED_EVENT_SCHEMA_HASH",
    "MBP1_FEATURE_TABLE_SCHEMA_HASH",
    "MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH",
    "mbp1_window_validity_fields",
    "mbp1_window_missing_reason_fields",
    "mbp1_window_keys",
    "arrow_schema_hash",
    "assert_schema_names_match",
]

#: Databento fixed-precision price decoding: int64 at 1e-9 dollars.
PRICE_SCALE_POLICY_ID = "databento_fixed_price_1e9_v1"
DATABENTO_PRICE_SCALE = 1e-9

#: Pinned tick sizes for the instruments this lane may represent. The
#: normalizer refuses unknown instruments rather than guessing a scale.
INSTRUMENT_TICK_SIZES: MappingProxyType[str, float] = MappingProxyType(
    {"NQ": 0.25, "ES": 0.25}
)

#: Raw Databento ``mbp-1`` parquet source contract. ``ts_recv`` is present in
#: the raw parquet (verified in planning; the current SC reader discards it —
#: this lane reads it directly). Unsigned vendor variants of the integer
#: columns are accepted by widening to the pinned signed type
#: (``widen_to_int64_v1``); names must match exactly.
MBP1_SOURCE_EVENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("ts_recv", pa.int64()),
        pa.field("ts_event", pa.int64()),
        pa.field("rtype", pa.int64()),
        pa.field("publisher_id", pa.int64()),
        pa.field("instrument_id", pa.int64()),
        pa.field("action", pa.large_string()),
        pa.field("side", pa.large_string()),
        pa.field("depth", pa.int64()),
        pa.field("price", pa.int64()),
        pa.field("size", pa.int64()),
        pa.field("flags", pa.int64()),
        pa.field("ts_in_delta", pa.int64()),
        pa.field("sequence", pa.int64()),
        pa.field("bid_px_00", pa.int64()),
        pa.field("ask_px_00", pa.int64()),
        pa.field("bid_sz_00", pa.int64()),
        pa.field("ask_sz_00", pa.int64()),
        pa.field("bid_ct_00", pa.int64()),
        pa.field("ask_ct_00", pa.int64()),
        pa.field("symbol", pa.large_string()),
    ]
)

#: The normalized working schema the materializer computes over. Prices are
#: converted to ticks under ``PRICE_SCALE_POLICY_ID`` + the pinned tick size;
#: ``source_ordinal`` is the deterministic row ordinal within one source
#: partition — the final tie-break of the complete four-part order key.
MBP1_NORMALIZED_EVENT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("ts_event", pa.int64()),
        pa.field("ts_recv", pa.int64()),
        pa.field("sequence", pa.int64()),
        pa.field("source_ordinal", pa.int64()),
        pa.field("action", pa.large_string()),
        pa.field("side", pa.large_string()),
        pa.field("size", pa.int64()),
        pa.field("bid_px_ticks", pa.float64()),
        pa.field("ask_px_ticks", pa.float64()),
        pa.field("bid_sz", pa.int64()),
        pa.field("ask_sz", pa.int64()),
        pa.field("bid_ct", pa.int64()),
        pa.field("ask_ct", pa.int64()),
        pa.field("instrument_id", pa.int64()),
        pa.field("symbol", pa.large_string()),
        pa.field("trading_day", pa.large_string()),
    ]
)


def mbp1_window_keys() -> tuple[str, ...]:
    """The nine registered window keys (5 snapshots + 4 transitions)."""

    snapshots = tuple(f"ofl_snap_{stage}" for stage in MBP1_STAGES)
    transitions = tuple(
        f"ofl_win_{from_stage}_{to_stage}"
        for from_stage, to_stage in zip(MBP1_STAGES[:-1], MBP1_STAGES[1:], strict=True)
    )
    return (*snapshots, *transitions)


def mbp1_window_validity_fields() -> tuple[str, ...]:
    """Per-window validity evidence columns (never model features)."""

    return tuple(f"{key}_valid" for key in mbp1_window_keys())


def mbp1_window_missing_reason_fields() -> tuple[str, ...]:
    """Per-window typed missing-reason columns (never model features)."""

    return tuple(f"{key}_missing_reason" for key in mbp1_window_keys())


#: The materialized feature table: exact join key + evidence identity, then
#: the 76 metric columns, then per-window validity/missing-reason evidence.
MBP1_FEATURE_TABLE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("candidate_id", pa.large_string()),
        pa.field("setup_id", pa.large_string()),
        pa.field("trading_day", pa.large_string()),
        *(pa.field(name, pa.float64()) for name in mbp1_feature_names()),
        *(pa.field(name, pa.bool_()) for name in mbp1_window_validity_fields()),
        *(
            pa.field(name, pa.large_string())
            for name in mbp1_window_missing_reason_fields()
        ),
    ]
)

#: Per-candidate stage-window evidence: the exact cutoff facts the dashboard
#: drill-down renders (never an artificial ``+inf`` bound anywhere).
MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("candidate_id", pa.large_string()),
        pa.field("trading_day", pa.large_string()),
        pa.field("feature_window_key", pa.large_string()),
        pa.field("from_stage", pa.large_string()),
        pa.field("to_stage", pa.large_string()),
        pa.field("trigger_semantics", pa.large_string()),
        pa.field("cutoff_kind", pa.large_string()),
        pa.field("from_ts_utc", pa.large_string()),
        pa.field("to_ts_utc", pa.large_string()),
        pa.field("admitted_event_count", pa.int64()),
        pa.field("same_timestamp_ambiguous", pa.bool_()),
        pa.field("valid", pa.bool_()),
        pa.field("missing_reason", pa.large_string()),
    ]
)


def arrow_schema_hash(schema: pa.Schema) -> str:
    """Canonical hash of the ordered ``(name, type)`` pairs of one schema."""

    return canonical_contract_sha256(
        {"fields": [[field.name, str(field.type)] for field in schema]}
    )


def assert_schema_names_match(schema: pa.Schema, actual_names: list[str] | tuple[str, ...]) -> None:
    """Every contract field must exist in the actual source, exactly by name."""

    missing = [field.name for field in schema if field.name not in set(actual_names)]
    if missing:
        raise ValueError(
            f"source does not satisfy the pinned MBP-1 schema; missing fields: {missing}"
        )


MBP1_SOURCE_EVENT_SCHEMA_HASH = arrow_schema_hash(MBP1_SOURCE_EVENT_SCHEMA)
MBP1_NORMALIZED_EVENT_SCHEMA_HASH = arrow_schema_hash(MBP1_NORMALIZED_EVENT_SCHEMA)
MBP1_FEATURE_TABLE_SCHEMA_HASH = arrow_schema_hash(MBP1_FEATURE_TABLE_SCHEMA)
MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA_HASH = arrow_schema_hash(
    MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA
)

# ── import-time guards: MBP-1 is the maximum representable depth ─────────────
for _schema in (
    MBP1_SOURCE_EVENT_SCHEMA,
    MBP1_NORMALIZED_EVENT_SCHEMA,
    MBP1_FEATURE_TABLE_SCHEMA,
    MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA,
):
    assert_no_deep_book_identifiers(field.name for field in _schema)
