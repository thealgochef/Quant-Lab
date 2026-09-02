"""``stratified_prop`` — prop-account events by descriptive OOS regime (R6.1 §6.G; D15).

Attribution policy ``source_trade_then_pit_v1``: an event that carries a
``source_trade_id`` takes THAT trade's descriptive OOS regime (the exact
trade join of ``regime_assignment_sources``); a historical no-trade event on
the PANEL grain is assigned point-in-time by its ``event_ts_utc`` through
the normative panel rule over the persisted fit assignments; bootstrap /
stress events are attributed ONLY via their source trade — their synthetic
2020 clocks are never consulted. Everything else is typed unattributable.
Counts / shares by regime × event type per firm/mode, payout and fee sums,
per-regime realized P&L, coverage, and the unattributable bucket ride the
body; ``probabilities_reestimated=False`` always.

Event detail enters ONLY through a loader seam (``AccountEventDetailLoader``)
yielding the documented partition frames (D15 columns) or ``None`` when the
simulation was persisted under ``none_v0`` — such simulations are typed
``evidence_not_persisted``, never silently dropped. The default loader reads
the historical ``account_events.json`` sidecar of the ``account_simulations``
store (present for historical modes only) and projects it onto the same
columns (the event's own ``trading_day`` and the D15 ``EVENT_TYPE_PRECEDENCE``
— never a UTC date prefix or a local rank table); the D15 v2 Parquet reader
plugs into the same seam.

HARDENING-BACKEND §4.4 (F-17) — external aggregation. Every partition is
attributed as it streams past and its group rows are written as a TYPED
intermediate Parquet partition into an attempt-local temp directory
(:data:`INTERMEDIATE_SUMMARY_SCHEMA`); no event and no group row is retained
in Python across partitions. The exact aggregation across partitions, the
exact unique-path counts, the cross-partition path-repetition refusal, the
exact row-budget count and the canonical ordering (:func:`summary_order_sql`
— the ORDER BY form of :func:`_summary_sort_key`) run in DuckDB under an
explicit ``memory_limit`` (:data:`SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES`),
one thread (deterministic floating-point accumulation) and a spill
``temp_directory`` under the attempt directory. The final rows stream in
canonical order into a row-group-aligned ZSTD Parquet writer whose bytes
equal ``pq.write_table(table, row_group_size=65_536)`` of the same table
(:func:`summary_parquet_bytes`), so the published bytes and their hash are
independent of the streaming path. The attempt directory is removed on
success and on every refusal; immutable outputs are never touched.

D15 report-local summary (plan §6.G): a ``stratified_prop`` report carries a
bounded ZSTD-Parquet ``account_event_regime_summary.parquet`` sidecar — one
row per (simulation, path, regime stratum / typed unattributable reason,
event type) with counts, amount sums and first/last event ordinals /
instants, keyed by the exact ``RegimeAssignmentEvidenceRef`` — built under
the registered ``EventRegimeSummaryBudget``; a summary over the row or byte
budget refuses (``EventRegimeSummaryBudgetError``) BEFORE anything is
published. The JSON detail sidecar keeps aggregate facts only.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from alpha_lab.propsim.event_detail import EVENT_TYPE_PRECEDENCE

from ..features.arrow_tables import arrow_schema_hash, bytes_sha256
from ..search.store import SearchStoreError, load_sidecar_bytes, load_verified_envelope
from .regime_stratified_contracts import (
    ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1,
    ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR,
    EventRegimeSummaryBudget,
    PropStratumRow,
    RegimeAssignmentEvidenceRef,
    RegimeStratumKey,
    StratifiedPropBody,
)

__all__ = [
    "EVENT_DETAIL_COLUMNS",
    "HISTORICAL_CLOCK_POLICY_ID",
    "SYNTHETIC_CLOCK_POLICY_ID",
    "ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_VERSION",
    "ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA",
    "ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH",
    "SUMMARY_ROW_GROUP_SIZE",
    "SUMMARY_ROW_KEYS",
    "INTERMEDIATE_SUMMARY_SCHEMA",
    "SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES",
    "EventRegimeSummaryBudgetError",
    "AccountEventDetailLoader",
    "PanelEventAssigner",
    "StratifiedPropResult",
    "load_account_event_detail_from_json",
    "event_detail_frame",
    "summary_parquet_bytes",
    "summary_order_sql",
    "read_account_event_regime_summary",
    "build_stratified_prop_body",
]

#: The D15 event-detail partition columns (plan §6.G).
EVENT_DETAIL_COLUMNS: tuple[str, ...] = (
    "path_instance_id",
    "path_block_id",
    "account_id",
    "event_id",
    "event_ts_utc",
    "trading_day",
    "clock_policy_id",
    "event_precedence",
    "event_ordinal",
    "event_type",
    "account_phase",
    "source_trade_id",
    "source_candidate_id",
    "amount",
)

HISTORICAL_CLOCK_POLICY_ID = "historical_calendar_clock_v1"
SYNTHETIC_CLOCK_POLICY_ID = "synthetic_path_clock_v1"
_ACCOUNT_SIMULATION_STORE = "account_simulations"
_EVENTS_SIDECAR = "account_events.json"
_NO_REASON = ""
#: The summary writer's row-group size (unchanged since R6.1 D15).
SUMMARY_ROW_GROUP_SIZE = 65_536
#: The explicit DuckDB memory limit of the external aggregation; beyond it
#: the engine spills into the attempt-local temp directory.
SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
_AGGREGATION_THREADS = 1

#: ``loader(root, account_simulation_id) -> Iterator[DataFrame] | None``
AccountEventDetailLoader = Callable[[Path, str], "Iterator[pd.DataFrame] | None"]
#: ``assigner(event_ts_utc: Series[str]) -> Series[int | None]`` (panel PIT)
PanelEventAssigner = Callable[[pd.Series], pd.Series]


class EventRegimeSummaryBudgetError(ValueError):
    """The event-regime summary would exceed its registered budget — refused
    BEFORE publication (no partial artifact ever exists)."""


ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_VERSION = 1
#: One row per (simulation, path, regime stratum / typed reason, event type);
#: the leading columns key every row to the exact assignment evidence.
ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("core_replay_id", pa.string(), nullable=False),
        pa.field("regime_oos_assignment_id", pa.string(), nullable=False),
        pa.field("regime_fold_set_id", pa.string(), nullable=False),
        pa.field("fold_schedule_id", pa.string(), nullable=False),
        pa.field("observation_granularity", pa.string(), nullable=False),
        pa.field("regime_fit_ids_sha256", pa.string(), nullable=False),
        pa.field("account_simulation_id", pa.string(), nullable=False),
        pa.field("firm_label", pa.string(), nullable=False),
        pa.field("simulation_mode", pa.string(), nullable=False),
        pa.field("clock_policy_id", pa.string(), nullable=False),
        pa.field("path_instance_id", pa.string(), nullable=False),
        pa.field("regime_stratum", pa.string(), nullable=False),
        pa.field("canonical_reporting_cluster_id", pa.int64(), nullable=True),
        pa.field("unattributable_reason", pa.string(), nullable=True),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("event_precedence", pa.int32(), nullable=False),
        pa.field("event_count", pa.int64(), nullable=False),
        pa.field("amount_sum", pa.float64(), nullable=True),
        pa.field("first_event_ordinal", pa.int64(), nullable=False),
        pa.field("last_event_ordinal", pa.int64(), nullable=False),
        pa.field("first_event_ts_utc", pa.string(), nullable=False),
        pa.field("last_event_ts_utc", pa.string(), nullable=False),
    ]
)
ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH = arrow_schema_hash(ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA)
#: The grouping key of one summary row (within a simulation).
SUMMARY_ROW_KEYS: tuple[str, ...] = (
    "path_instance_id",
    "clock_policy_id",
    "regime_stratum",
    "unattributable_reason",
    "event_type",
)
#: The typed intermediate partition the external aggregation consumes: one
#: row per (partition, summary key) with that partition's partial aggregates.
INTERMEDIATE_SUMMARY_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("partition_ordinal", pa.int64(), nullable=False),
        pa.field("path_instance_id", pa.string(), nullable=False),
        pa.field("clock_policy_id", pa.string(), nullable=False),
        pa.field("regime_stratum", pa.string(), nullable=False),
        pa.field("canonical_reporting_cluster_id", pa.int64(), nullable=True),
        pa.field("unattributable_reason", pa.string(), nullable=False),
        pa.field("event_type", pa.string(), nullable=False),
        pa.field("event_precedence", pa.int32(), nullable=False),
        pa.field("event_count", pa.int64(), nullable=False),
        pa.field("amount_count", pa.int64(), nullable=False),
        pa.field("amount_total", pa.float64(), nullable=False),
        pa.field("first_event_ordinal", pa.int64(), nullable=False),
        pa.field("last_event_ordinal", pa.int64(), nullable=False),
        pa.field("first_event_ts_utc", pa.string(), nullable=False),
        pa.field("last_event_ts_utc", pa.string(), nullable=False),
    ]
)
_AMOUNT_TYPES: tuple[str, ...] = ("payout", "fee", "equity_update")


@dataclass(frozen=True)
class StratifiedPropResult:
    """The report body, the aggregate JSON detail, and the D15 summary BYTES
    (already checked against the budget). The Arrow table is parsed lazily
    from the bytes (``summary``) — the builder never holds a second copy."""

    body: StratifiedPropBody
    detail: dict[str, Any]
    summary_bytes: bytes
    summary_sha256: str
    summary_schema_hash: str
    summary_rows: int

    @property
    def summary(self) -> pa.Table:
        return read_account_event_regime_summary(self.summary_bytes)


def _amount(event_type: str, payload: Mapping[str, Any]) -> float:
    if event_type == "fee":
        return float(payload.get("amount", 0.0))
    if event_type == "payout":
        return float(payload.get("trader_amount", 0.0))
    if event_type == "equity_update":
        return float(payload.get("realized_delta", 0.0))
    if event_type == "replacement":
        return float(payload.get("reset_fee", 0.0))
    return 0.0


def _precedence(event_type: str) -> int:
    rank = EVENT_TYPE_PRECEDENCE.get(event_type)
    if rank is None:
        raise ValueError(
            f"event type {event_type!r} is not a registered D15 event type "
            f"({', '.join(sorted(EVENT_TYPE_PRECEDENCE))})"
        )
    return int(rank)


def _trading_day(record: Mapping[str, Any], *, clock_policy_id: str) -> str | None:
    value = record.get("trading_day")
    if value is None or str(value) == "":
        if clock_policy_id == HISTORICAL_CLOCK_POLICY_ID:
            raise ValueError(
                f"historical event {str(record.get('event_id'))[:12]}… carries no trading_day "
                "(PropAccountEventEnvelope.trading_day); a UTC date prefix is never substituted"
            )
        return None
    return str(value)


def event_detail_frame(
    records: Iterable[Mapping[str, Any]], *, clock_policy_id: str
) -> pd.DataFrame:
    """Project ``PropAccountEventEnvelope`` records onto the D15 columns —
    the record's own ``trading_day`` (required under the historical clock;
    null under a synthetic clock, which is never consulted) and the D15
    ``EVENT_TYPE_PRECEDENCE`` (an unregistered event type refuses)."""

    rows = []
    for record in records:
        event_type = str(record["event_type"])
        ts = str(record["event_ts_utc"])
        rows.append(
            {
                "path_instance_id": str(record["path_instance_id"]),
                "path_block_id": 0,
                "account_id": str(record["account_id"]),
                "event_id": str(record["event_id"]),
                "event_ts_utc": ts,
                "trading_day": _trading_day(record, clock_policy_id=clock_policy_id),
                "clock_policy_id": clock_policy_id,
                "event_precedence": _precedence(event_type),
                "event_ordinal": int(record["event_ordinal"]),
                "event_type": event_type,
                "account_phase": str(record["account_phase"]),
                "source_trade_id": record.get("source_trade_id"),
                "source_candidate_id": record.get("source_candidate_id"),
                "amount": _amount(event_type, record.get("payload") or {}),
            }
        )
    return pd.DataFrame(rows, columns=list(EVENT_DETAIL_COLUMNS))


def load_account_event_detail_from_json(
    root: Path, account_simulation_id: str
) -> Iterator[pd.DataFrame] | None:
    """The R3-era historical event stream (``account_events.json``) as ONE
    detail frame; ``None`` when the simulation carries no event detail."""

    from alpha_lab.propsim.simulation import AccountSimulationEnvelope  # noqa: PLC0415

    envelope = load_verified_envelope(
        Path(root), _ACCOUNT_SIMULATION_STORE, account_simulation_id, AccountSimulationEnvelope
    )
    try:
        raw = load_sidecar_bytes(
            Path(root), _ACCOUNT_SIMULATION_STORE, account_simulation_id, _EVENTS_SIDECAR
        )
    except SearchStoreError:
        return None
    records = json.loads(raw.decode("utf-8"))
    clock = HISTORICAL_CLOCK_POLICY_ID
    if not envelope.payload.simulation_mode.startswith("historical"):
        clock = SYNTHETIC_CLOCK_POLICY_ID
    return iter((event_detail_frame(records, clock_policy_id=clock),))


def _validate_detail(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(EVENT_DETAIL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"event detail lacks columns {missing}")
    if frame["event_id"].astype(str).duplicated().any():
        raise ValueError("event detail repeats an event_id within one partition")
    return frame


def _attribute_partition(
    events: pd.DataFrame,
    *,
    regime_of_trade: Mapping[str, int | None],
    historical: bool,
    panel_assigner: PanelEventAssigner | None,
) -> tuple[pd.Series, pd.Series]:
    """Per event: the attributed regime (or None) and the typed unattributable
    reason (or None) — vectorized; the panel PIT seam is consulted ONLY for
    historical no-trade events."""

    source = events["source_trade_id"].astype(object)
    is_na = source.isna()
    text = source.where(~is_na, "").astype(str)
    has_source = (~is_na) & (~text.isin(("", "nan", "None", "<NA>")))
    known = has_source & text.isin(list(regime_of_trade))
    cluster = text.map(dict(regime_of_trade))
    regime = pd.Series([None] * len(events), index=events.index, dtype=object)
    reason = pd.Series([None] * len(events), index=events.index, dtype=object)
    hit = known & cluster.notna()
    if bool(hit.any()):
        regime[hit] = [int(value) for value in cluster[hit]]
    reason[known & ~hit] = "source_trade_unassigned"
    reason[has_source & ~known] = "source_trade_not_in_regime_evidence"
    no_source = ~has_source
    if historical and panel_assigner is not None and bool(no_source.any()):
        pit = pd.Series(panel_assigner(events.loc[no_source, "event_ts_utc"].astype(str)))
        present = pit.index.intersection(events.index[no_source])
        values = pit.loc[present]
        ok = values.notna().to_numpy()
        if bool(ok.any()):
            regime.loc[present[ok]] = [int(value) for value in values[ok]]
        reason.loc[present[~ok]] = "panel_pit_unassigned"
    remaining = no_source & regime.isna() & reason.isna()
    reason[remaining] = "no_source_trade" if historical else "no_source_trade_synthetic_clock"
    return regime, reason


def _partition_groups(events: pd.DataFrame, regime: pd.Series, reason: pd.Series) -> pd.DataFrame:
    """The partition's summary rows (one per ``SUMMARY_ROW_KEYS`` group)."""

    frame = pd.DataFrame(
        {
            "path_instance_id": events["path_instance_id"].astype(str).to_numpy(),
            "clock_policy_id": events["clock_policy_id"].astype(str).to_numpy(),
            "regime_stratum": [
                "unassigned" if value is None else f"regime:{int(value)}" for value in regime
            ],
            "unattributable_reason": [
                _NO_REASON if value is None else str(value) for value in reason
            ],
            "event_type": events["event_type"].astype(str).to_numpy(),
            "event_precedence": pd.to_numeric(events["event_precedence"], errors="raise")
            .astype(int)
            .to_numpy(),
            "event_ordinal": pd.to_numeric(events["event_ordinal"], errors="raise")
            .astype(int)
            .to_numpy(),
            "event_ts_utc": events["event_ts_utc"].astype(str).to_numpy(),
            "amount": pd.to_numeric(events["amount"], errors="raise").astype(float).to_numpy(),
        }
    )
    if frame.empty:
        return frame.iloc[0:0]
    return (
        frame.groupby(list(SUMMARY_ROW_KEYS), sort=True)
        .agg(
            event_count=("event_ordinal", "size"),
            amount_count=("amount", "count"),
            amount_total=("amount", "sum"),
            event_precedence=("event_precedence", "first"),
            first_event_ordinal=("event_ordinal", "min"),
            last_event_ordinal=("event_ordinal", "max"),
            first_event_ts_utc=("event_ts_utc", "min"),
            last_event_ts_utc=("event_ts_utc", "max"),
        )
        .reset_index()
    )


def _cluster_of(stratum: str) -> int | None:
    return None if stratum == "unassigned" else int(stratum.split(":", 1)[1])


def _write_intermediate_partition(
    groups: pd.DataFrame, *, partition_ordinal: int, path: Path
) -> int:
    """Write one partition's group rows under the TYPED intermediate schema
    (a vectorized frame → Arrow conversion; no per-row Python loop)."""

    strata = groups["regime_stratum"].astype(str)
    cluster = pd.to_numeric(
        strata.where(strata != "unassigned").str.split(":", n=1).str[1], errors="raise"
    ).astype("Int64")
    frame = pd.DataFrame(
        {
            "partition_ordinal": pd.Series(
                [int(partition_ordinal)] * len(groups), dtype="int64", index=groups.index
            ),
            "path_instance_id": groups["path_instance_id"].astype(str),
            "clock_policy_id": groups["clock_policy_id"].astype(str),
            "regime_stratum": strata,
            "canonical_reporting_cluster_id": cluster,
            "unattributable_reason": groups["unattributable_reason"].astype(str),
            "event_type": groups["event_type"].astype(str),
            "event_precedence": groups["event_precedence"].astype("int32"),
            "event_count": groups["event_count"].astype("int64"),
            "amount_count": groups["amount_count"].astype("int64"),
            "amount_total": groups["amount_total"].astype("float64"),
            "first_event_ordinal": groups["first_event_ordinal"].astype("int64"),
            "last_event_ordinal": groups["last_event_ordinal"].astype("int64"),
            "first_event_ts_utc": groups["first_event_ts_utc"].astype(str),
            "last_event_ts_utc": groups["last_event_ts_utc"].astype(str),
        }
    )
    table = pa.Table.from_pandas(
        frame, schema=INTERMEDIATE_SUMMARY_SCHEMA, preserve_index=False
    ).replace_schema_metadata(None)
    pq.write_table(table, path, compression="zstd", row_group_size=SUMMARY_ROW_GROUP_SIZE)
    return int(table.num_rows)


def _stratum_row(
    key: RegimeStratumKey,
    counts: Mapping[str, int],
    amounts: Mapping[str, float],
    *,
    firm_label: str,
    mode: str,
    simulation_id: str,
    simulation_total: int,
) -> PropStratumRow:
    return PropStratumRow(
        firm_label=firm_label,
        simulation_mode=mode,
        account_simulation_id=simulation_id,
        key=key,
        event_counts_by_type={k: int(v) for k, v in sorted(counts.items())},
        event_share_by_type={
            k: (int(v) / simulation_total if simulation_total else 0.0)
            for k, v in sorted(counts.items())
        },
        payout_trader_amount_sum=float(amounts.get("payout", 0.0)),
        fee_amount_sum=float(amounts.get("fee", 0.0)),
        realized_pnl_sum=float(amounts.get("equity_update", 0.0)),
        events_total=int(sum(counts.values())),
    )


def _summary_sort_key(row: Mapping[str, Any]) -> tuple:
    """The canonical order of the summary rows (the reference form; the
    external aggregation sorts with :func:`summary_order_sql`)."""

    cluster = row["canonical_reporting_cluster_id"]
    return (
        str(row["account_simulation_id"]),
        str(row["path_instance_id"]),
        (1, 0) if cluster is None else (0, int(cluster)),
        str(row["unattributable_reason"] or ""),
        str(row["event_type"]),
        str(row["clock_policy_id"]),
    )


def summary_order_sql() -> str:
    """The ORDER BY clause equal to :func:`_summary_sort_key` WITHIN one
    simulation (simulations are processed in sorted id order): path, then
    assigned clusters before the unassigned bucket, cluster id, reason
    (the empty string for none), event type, clock policy — binary
    (code-point) collation, as Python compares these ASCII strings."""

    return (
        "ORDER BY path_instance_id, "
        "CASE WHEN canonical_reporting_cluster_id IS NULL THEN 1 ELSE 0 END, "
        "COALESCE(canonical_reporting_cluster_id, 0), "
        "unattributable_reason, event_type, clock_policy_id"
    )


def summary_parquet_bytes(table: pa.Table) -> bytes:
    """Deterministic ZSTD Parquet bytes of a summary table (the D15 writer's
    settings; no pandas metadata)."""

    if arrow_schema_hash(table.schema) != ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH:
        raise ValueError("the table is not an account_event_regime_summary table")
    sink = io.BytesIO()
    pq.write_table(
        table.replace_schema_metadata(None),
        sink,
        compression="zstd",
        row_group_size=SUMMARY_ROW_GROUP_SIZE,
    )
    return sink.getvalue()


def read_account_event_regime_summary(data: bytes) -> pa.Table:
    """Parse summary bytes and refuse any schema drift (names / types)."""

    table = pq.read_table(io.BytesIO(data))
    if arrow_schema_hash(table.schema) != ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH:
        raise ValueError("stored event-regime summary does not carry the registered schema")
    return table


class _RowGroupAlignedWriter:
    """A streaming Parquet writer whose row groups are EXACTLY the groups
    ``pq.write_table(table, row_group_size=N)`` would cut from the whole
    table — so the streamed bytes equal the single-write bytes."""

    def __init__(self, path: Path, schema: pa.Schema, row_group_size: int) -> None:
        self._writer = pq.ParquetWriter(path, schema, compression="zstd")
        self._schema = schema
        self._size = int(row_group_size)
        self._buffer: list[pa.Table] = []
        self._buffered = 0
        self.rows_written = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        self._buffer.append(table)
        self._buffered += table.num_rows
        while self._buffered >= self._size:
            combined = pa.concat_tables(self._buffer).combine_chunks()
            head = combined.slice(0, self._size)
            self._writer.write_table(head, row_group_size=self._size)
            self.rows_written += head.num_rows
            rest = combined.slice(self._size)
            self._buffer = [rest] if rest.num_rows else []
            self._buffered = rest.num_rows

    def close(self) -> None:
        if self._buffered:
            combined = pa.concat_tables(self._buffer).combine_chunks()
            self._writer.write_table(combined, row_group_size=self._size)
            self.rows_written += combined.num_rows
            self._buffer = []
            self._buffered = 0
        self._writer.close()


def _row_budget_refusal(
    core_replay_id: str, budget: EventRegimeSummaryBudget
) -> EventRegimeSummaryBudgetError:
    return EventRegimeSummaryBudgetError(
        f"the account_event_regime_summary of child {core_replay_id[:12]}… would exceed the "
        f"registered row budget ({budget.max_summary_rows} rows, {budget.budget_id}); refused "
        "before publication"
    )


def _sql_path(path: Path) -> str:
    return str(Path(path).as_posix()).replace("'", "''")


def _open_aggregation_connection(temp_root: Path):
    """A DuckDB connection under the explicit memory limit, ONE thread (a
    deterministic accumulation order) and the attempt-local spill directory."""

    connection = duckdb.connect(database=":memory:")
    spill = Path(temp_root) / "duckdb_tmp"
    spill.mkdir(parents=True, exist_ok=True)
    connection.execute(
        f"SET memory_limit='{SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES // (1024 * 1024)}MiB'"
    )
    connection.execute(f"SET temp_directory='{_sql_path(spill)}'")
    connection.execute(f"SET threads={_AGGREGATION_THREADS}")
    connection.execute("SET preserve_insertion_order=true")
    return connection


_AGGREGATE_SQL = (
    "SELECT path_instance_id, clock_policy_id, regime_stratum, canonical_reporting_cluster_id, "
    "unattributable_reason, event_type, "
    "MIN(event_precedence) AS event_precedence, "
    "SUM(event_count)::BIGINT AS event_count, "
    "SUM(amount_count)::BIGINT AS amount_count, "
    "SUM(amount_total)::DOUBLE AS amount_total, "
    "MIN(first_event_ordinal) AS first_event_ordinal, "
    "MAX(last_event_ordinal) AS last_event_ordinal, "
    "MIN(first_event_ts_utc) AS first_event_ts_utc, "
    "MAX(last_event_ts_utc) AS last_event_ts_utc "
    "FROM read_parquet([{files}]) "
    "GROUP BY path_instance_id, clock_policy_id, regime_stratum, canonical_reporting_cluster_id, "
    "unattributable_reason, event_type"
)


def _summary_batch(
    batch: pa.RecordBatch,
    *,
    core_replay_id: str,
    evidence: RegimeAssignmentEvidenceRef,
    fit_ids_sha256: str,
    simulation_id: str,
    firm_label: str,
    mode: str,
) -> pa.Table:
    """One aggregated batch (already in canonical order) as a summary table
    under the registered schema."""

    rows = batch.num_rows
    amount_count = batch.column("amount_count").to_pylist()
    amount_total = batch.column("amount_total").to_pylist()
    reasons = batch.column("unattributable_reason").to_pylist()
    columns: dict[str, Any] = {
        "core_replay_id": pa.array([core_replay_id] * rows, pa.string()),
        "regime_oos_assignment_id": pa.array(
            [evidence.regime_oos_assignment_id] * rows, pa.string()
        ),
        "regime_fold_set_id": pa.array([evidence.regime_fold_set_id] * rows, pa.string()),
        "fold_schedule_id": pa.array([evidence.fold_schedule_id] * rows, pa.string()),
        "observation_granularity": pa.array(
            [str(evidence.observation_granularity.value)] * rows, pa.string()
        ),
        "regime_fit_ids_sha256": pa.array([fit_ids_sha256] * rows, pa.string()),
        "account_simulation_id": pa.array([simulation_id] * rows, pa.string()),
        "firm_label": pa.array([firm_label] * rows, pa.string()),
        "simulation_mode": pa.array([mode] * rows, pa.string()),
        "clock_policy_id": batch.column("clock_policy_id").cast(pa.string()),
        "path_instance_id": batch.column("path_instance_id").cast(pa.string()),
        "regime_stratum": batch.column("regime_stratum").cast(pa.string()),
        "canonical_reporting_cluster_id": batch.column("canonical_reporting_cluster_id").cast(
            pa.int64()
        ),
        "unattributable_reason": pa.array(
            [None if value == _NO_REASON else str(value) for value in reasons], pa.string()
        ),
        "event_type": batch.column("event_type").cast(pa.string()),
        "event_precedence": batch.column("event_precedence").cast(pa.int32()),
        "event_count": batch.column("event_count").cast(pa.int64()),
        "amount_sum": pa.array(
            [
                float(total) if int(count) > 0 else None
                for total, count in zip(amount_total, amount_count, strict=True)
            ],
            pa.float64(),
        ),
        "first_event_ordinal": batch.column("first_event_ordinal").cast(pa.int64()),
        "last_event_ordinal": batch.column("last_event_ordinal").cast(pa.int64()),
        "first_event_ts_utc": batch.column("first_event_ts_utc").cast(pa.string()),
        "last_event_ts_utc": batch.column("last_event_ts_utc").cast(pa.string()),
    }
    return pa.Table.from_pydict(
        {name: columns[name] for name in ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA.names},
        schema=ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA,
    )


def _empty_summary_table() -> pa.Table:
    return pa.Table.from_pydict(
        {field.name: [] for field in ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA},
        schema=ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA,
    )


def _strata_for(
    per_cluster_type: list[tuple[int | None, str, int, float]],
    *,
    firm_label: str,
    mode: str,
    simulation_id: str,
    simulation_total: int,
) -> list[PropStratumRow]:
    """The pooled / covered / per-regime / unassigned strata from the SMALL
    (cluster, event type) aggregate the engine returned."""

    common = dict(
        firm_label=firm_label, mode=mode, simulation_id=simulation_id,
        simulation_total=simulation_total,
    )

    def _rows(selected) -> tuple[dict[str, int], dict[str, float]]:
        counts: dict[str, int] = {}
        amounts: dict[str, float] = {}
        for _cluster, event_type, count, amount in selected:
            counts[event_type] = counts.get(event_type, 0) + int(count)
            if event_type in _AMOUNT_TYPES:
                amounts[event_type] = amounts.get(event_type, 0.0) + float(amount)
        return counts, amounts

    covered = [entry for entry in per_cluster_type if entry[0] is not None]
    unassigned = [entry for entry in per_cluster_type if entry[0] is None]
    strata = [
        _stratum_row(RegimeStratumKey(stratum="pooled_all"), *_rows(per_cluster_type), **common),
        _stratum_row(
            RegimeStratumKey(stratum="pooled_regime_covered"), *_rows(covered), **common
        ),
    ]
    for cluster in sorted({int(entry[0]) for entry in covered}):
        strata.append(
            _stratum_row(
                RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=cluster),
                *_rows([entry for entry in covered if int(entry[0]) == cluster]),
                **common,
            )
        )
    strata.append(
        _stratum_row(RegimeStratumKey(stratum="unassigned"), *_rows(unassigned), **common)
    )
    return strata


def build_stratified_prop_body(
    *,
    root: Path,
    core_replay_id: str,
    simulations: Mapping[str, tuple[str, str]],
    trade_regimes: pd.DataFrame,
    evidence: RegimeAssignmentEvidenceRef,
    loader: AccountEventDetailLoader = load_account_event_detail_from_json,
    panel_assigner: PanelEventAssigner | None = None,
    budget: EventRegimeSummaryBudget = ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1,
) -> StratifiedPropResult:
    """``simulations`` maps ``account_simulation_id -> (firm_label, mode)``;
    ``trade_regimes`` is the exact trade join (``regime_for_trades``);
    ``evidence`` keys every summary row. Streams the partitions into typed
    intermediate partitions, aggregates them EXTERNALLY (DuckDB; explicit
    memory limit, spill directory, one thread) and refuses (typed) before
    publication when the summary exceeds ``budget``; the attempt-local temp
    directory never survives."""

    from alpha_lab.propsim.simulation import AccountSimulationEnvelope  # noqa: PLC0415

    if "trade_id" not in trade_regimes.columns:
        raise ValueError("trade_regimes lacks the trade_id column")
    regimes = trade_regimes.copy()
    regimes["trade_id"] = regimes["trade_id"].astype(str)
    if regimes["trade_id"].duplicated().any():
        raise ValueError("trade_regimes repeats a trade_id")
    regime_of_trade: dict[str, int | None] = {
        str(row.trade_id): (
            int(row.canonical_reporting_cluster_id)
            if bool(row.valid) and pd.notna(row.canonical_reporting_cluster_id)
            else None
        )
        for row in regimes.itertuples(index=False)
    }
    fit_ids_sha256 = bytes_sha256("\n".join(evidence.regime_fit_ids).encode("utf-8"))
    strata: list[PropStratumRow] = []
    detail: dict[str, Any] = {
        "summary": {
            "sidecar": ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR,
            "schema_version": ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_VERSION,
            "schema_hash": ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH,
            "row_keys": list(SUMMARY_ROW_KEYS),
            "budget": budget.model_dump(mode="json"),
            "aggregation": {
                "engine": "duckdb_external_v1",
                "memory_limit_bytes": int(SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES),
                "threads": _AGGREGATION_THREADS,
                "row_group_size": SUMMARY_ROW_GROUP_SIZE,
            },
        },
        "simulations": {},
    }
    events_total = 0
    events_attributed = 0
    summary_rows_total = 0
    unattributable: dict[str, int] = {}
    not_persisted: list[str] = []
    temp_root = Path(tempfile.mkdtemp(prefix="ifvg_event_regime_summary_"))
    connection = None
    writer: _RowGroupAlignedWriter | None = None
    summary_path = temp_root / ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR
    try:
        for index, simulation_id in enumerate(sorted(simulations)):
            firm_label, mode = simulations[simulation_id]
            envelope = load_verified_envelope(
                Path(root), _ACCOUNT_SIMULATION_STORE, simulation_id, AccountSimulationEnvelope
            )
            if envelope.payload.simulation_mode != mode:
                raise ValueError(
                    f"account simulation {simulation_id[:12]}… is a "
                    f"{envelope.payload.simulation_mode} run, not {mode}"
                )
            frames = loader(Path(root), simulation_id)
            if frames is None:
                not_persisted.append(simulation_id)
                detail["simulations"][simulation_id] = {"evidence_not_persisted": True}
                continue
            historical = mode.startswith("historical")
            simulation_dir = temp_root / f"sim_{index:05d}"
            simulation_dir.mkdir(parents=True)
            clocks: set[str] = set()
            partitions = 0
            intermediate: list[Path] = []
            for frame in frames:
                events = _validate_detail(frame)
                clocks |= set(events["clock_policy_id"].astype(str))
                regime, reason = _attribute_partition(
                    events,
                    regime_of_trade=regime_of_trade,
                    historical=historical,
                    panel_assigner=panel_assigner,
                )
                groups = _partition_groups(events, regime, reason)
                path = simulation_dir / f"part_{partitions:06d}.parquet"
                if _write_intermediate_partition(groups, partition_ordinal=partitions, path=path):
                    intermediate.append(path)
                partitions += 1
            if intermediate:
                if connection is None:
                    connection = _open_aggregation_connection(temp_root)
                files = ", ".join(f"'{_sql_path(path)}'" for path in intermediate)
                source = f"read_parquet([{files}])"
                repeated = connection.execute(
                    f"SELECT path_instance_id FROM {source} GROUP BY path_instance_id "
                    "HAVING COUNT(DISTINCT partition_ordinal) > 1 LIMIT 1"
                ).fetchone()
                if repeated is not None:
                    raise ValueError("event detail repeats a path_instance_id across partitions")
                connection.execute(
                    "CREATE OR REPLACE TEMPORARY TABLE simulation_summary AS "
                    + _AGGREGATE_SQL.format(files=files)
                )
                count_rows, count_events, attributed, paths = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(event_count), 0), "
                    "COALESCE(SUM(CASE WHEN canonical_reporting_cluster_id IS NULL THEN 0 "
                    "ELSE event_count END), 0), COUNT(DISTINCT path_instance_id) "
                    "FROM simulation_summary"
                ).fetchone()
                count_rows, count_events, attributed, paths = (
                    int(count_rows), int(count_events), int(attributed), int(paths)
                )
                if summary_rows_total + count_rows > budget.max_summary_rows:
                    raise _row_budget_refusal(core_replay_id, budget)
                reasons = {
                    str(label): int(total)
                    for label, total in connection.execute(
                        "SELECT unattributable_reason, SUM(event_count) FROM simulation_summary "
                        "WHERE canonical_reporting_cluster_id IS NULL "
                        "GROUP BY unattributable_reason ORDER BY unattributable_reason"
                    ).fetchall()
                }
                per_cluster_type = [
                    (None if cluster is None else int(cluster), str(event_type), int(count),
                     float(amount))
                    for cluster, event_type, count, amount in connection.execute(
                        "SELECT canonical_reporting_cluster_id, event_type, SUM(event_count), "
                        "SUM(CASE WHEN amount_count > 0 THEN amount_total ELSE 0.0 END) "
                        "FROM simulation_summary GROUP BY canonical_reporting_cluster_id, "
                        "event_type ORDER BY canonical_reporting_cluster_id NULLS LAST, event_type"
                    ).fetchall()
                ]
                if writer is None:
                    writer = _RowGroupAlignedWriter(
                        summary_path, ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA, SUMMARY_ROW_GROUP_SIZE
                    )
                reader = connection.execute(
                    f"SELECT * FROM simulation_summary {summary_order_sql()}"
                ).fetch_record_batch(SUMMARY_ROW_GROUP_SIZE)
                for batch in reader:
                    writer.write(
                        _summary_batch(
                            batch,
                            core_replay_id=core_replay_id,
                            evidence=evidence,
                            fit_ids_sha256=fit_ids_sha256,
                            simulation_id=simulation_id,
                            firm_label=firm_label,
                            mode=mode,
                        )
                    )
                connection.execute("DROP TABLE simulation_summary")
            else:
                count_rows = count_events = attributed = paths = 0
                reasons = {}
                per_cluster_type = []
            shutil.rmtree(simulation_dir, ignore_errors=True)
            events_total += count_events
            events_attributed += attributed
            summary_rows_total += count_rows
            for label, reason_count in reasons.items():
                unattributable[label] = unattributable.get(label, 0) + reason_count
            strata.extend(
                _strata_for(
                    per_cluster_type,
                    firm_label=firm_label,
                    mode=mode,
                    simulation_id=simulation_id,
                    simulation_total=count_events,
                )
            )
            detail["simulations"][simulation_id] = {
                "firm_label": firm_label,
                "simulation_mode": mode,
                "clock_policy_ids": sorted(clocks),
                "partitions": partitions,
                "paths": paths,
                "events_total": count_events,
                "events_attributed": attributed,
                "unattributable_by_reason": {k: int(v) for k, v in sorted(reasons.items())},
                "summary_rows": count_rows,
            }
        if summary_rows_total > budget.max_summary_rows:
            raise _row_budget_refusal(core_replay_id, budget)
        if writer is not None:
            writer.close()
            writer = None
            if summary_path.stat().st_size > budget.max_published_bytes:
                raise EventRegimeSummaryBudgetError(
                    f"the account_event_regime_summary of child {core_replay_id[:12]}… "
                    f"serializes to {summary_path.stat().st_size} bytes, over the registered "
                    f"byte budget ({budget.max_published_bytes}, {budget.budget_id}); refused "
                    "before publication"
                )
            data = summary_path.read_bytes()
        else:
            data = summary_parquet_bytes(_empty_summary_table())
            if len(data) > budget.max_published_bytes:
                raise EventRegimeSummaryBudgetError(
                    f"the account_event_regime_summary of child {core_replay_id[:12]}… "
                    f"serializes to {len(data)} bytes, over the registered byte budget "
                    f"({budget.max_published_bytes}, {budget.budget_id}); refused before "
                    "publication"
                )
    finally:
        if writer is not None:
            writer.close()
        if connection is not None:
            connection.close()
        shutil.rmtree(temp_root, ignore_errors=True)
    detail["summary"]["rows"] = int(summary_rows_total)
    detail["summary"]["bytes"] = len(data)
    body = StratifiedPropBody(
        core_replay_id=core_replay_id,
        account_simulation_ids=tuple(sorted(simulations)),
        strata=tuple(strata),
        events_total=events_total,
        events_attributed=events_attributed,
        coverage_fraction=(events_attributed / events_total) if events_total else 0.0,
        partial_coverage=events_attributed < events_total,
        unattributable_by_reason={k: int(v) for k, v in sorted(unattributable.items())},
        evidence_not_persisted=tuple(sorted(not_persisted)),
        summary_budget=budget,
        summary_rows=int(summary_rows_total),
    )
    return StratifiedPropResult(
        body=body,
        detail=detail,
        summary_bytes=data,
        summary_sha256=bytes_sha256(data),
        summary_schema_hash=ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH,
        summary_rows=int(summary_rows_total),
    )
