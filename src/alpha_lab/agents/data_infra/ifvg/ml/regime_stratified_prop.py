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
plugs into the same seam. Every partition is attributed and AGGREGATED as it
streams past: no event is retained across partitions and no row-oriented
event JSON is ever written (memory is bounded by one partition).

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
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "SUMMARY_ROW_KEYS",
    "EventRegimeSummaryBudgetError",
    "AccountEventDetailLoader",
    "PanelEventAssigner",
    "StratifiedPropResult",
    "load_account_event_detail_from_json",
    "event_detail_frame",
    "summary_parquet_bytes",
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


@dataclass(frozen=True)
class StratifiedPropResult:
    """The report body, the aggregate JSON detail, and the D15 summary table
    with its serialized bytes (already checked against the budget)."""

    body: StratifiedPropBody
    detail: dict[str, Any]
    summary: pa.Table
    summary_bytes: bytes
    summary_sha256: str
    summary_schema_hash: str

    @property
    def summary_rows(self) -> int:
        return int(self.summary.num_rows)


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


def _merge_group(target: dict[str, Any], row: Mapping[str, Any]) -> None:
    target["event_count"] += int(row["event_count"])
    if int(row["amount_count"]) > 0:
        target["amount_sum"] = float(target["amount_sum"] or 0.0) + float(row["amount_total"])
    target["first_event_ordinal"] = min(
        target["first_event_ordinal"], int(row["first_event_ordinal"])
    )
    target["last_event_ordinal"] = max(target["last_event_ordinal"], int(row["last_event_ordinal"]))
    target["first_event_ts_utc"] = min(target["first_event_ts_utc"], str(row["first_event_ts_utc"]))
    target["last_event_ts_utc"] = max(target["last_event_ts_utc"], str(row["last_event_ts_utc"]))


def _cluster_of(stratum: str) -> int | None:
    return None if stratum == "unassigned" else int(stratum.split(":", 1)[1])


def _stratum_row(
    key: RegimeStratumKey,
    groups: Iterable[Mapping[str, Any]],
    *,
    firm_label: str,
    mode: str,
    simulation_id: str,
    simulation_total: int,
) -> PropStratumRow:
    counts: dict[str, int] = {}
    amounts = {"payout": 0.0, "fee": 0.0, "equity_update": 0.0}
    for group in groups:
        event_type = str(group["event_type"])
        counts[event_type] = counts.get(event_type, 0) + int(group["event_count"])
        if event_type in amounts and group["amount_sum"] is not None:
            amounts[event_type] += float(group["amount_sum"])
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
        payout_trader_amount_sum=float(amounts["payout"]),
        fee_amount_sum=float(amounts["fee"]),
        realized_pnl_sum=float(amounts["equity_update"]),
        events_total=int(sum(counts.values())),
    )


def _summary_sort_key(row: Mapping[str, Any]) -> tuple:
    cluster = row["canonical_reporting_cluster_id"]
    return (
        str(row["account_simulation_id"]),
        str(row["path_instance_id"]),
        (1, 0) if cluster is None else (0, int(cluster)),
        str(row["unattributable_reason"] or ""),
        str(row["event_type"]),
        str(row["clock_policy_id"]),
    )


def _summary_table(
    rows: list[dict[str, Any]], *, core_replay_id: str, evidence: RegimeAssignmentEvidenceRef
) -> pa.Table:
    ordered = sorted(rows, key=_summary_sort_key)
    fit_ids_sha256 = bytes_sha256("\n".join(evidence.regime_fit_ids).encode("utf-8"))
    columns: dict[str, list[Any]] = {
        field.name: [] for field in ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA
    }
    for row in ordered:
        columns["core_replay_id"].append(core_replay_id)
        columns["regime_oos_assignment_id"].append(evidence.regime_oos_assignment_id)
        columns["regime_fold_set_id"].append(evidence.regime_fold_set_id)
        columns["fold_schedule_id"].append(evidence.fold_schedule_id)
        columns["observation_granularity"].append(str(evidence.observation_granularity.value))
        columns["regime_fit_ids_sha256"].append(fit_ids_sha256)
        columns["account_simulation_id"].append(str(row["account_simulation_id"]))
        columns["firm_label"].append(str(row["firm_label"]))
        columns["simulation_mode"].append(str(row["simulation_mode"]))
        columns["clock_policy_id"].append(str(row["clock_policy_id"]))
        columns["path_instance_id"].append(str(row["path_instance_id"]))
        columns["regime_stratum"].append(str(row["regime_stratum"]))
        columns["canonical_reporting_cluster_id"].append(row["canonical_reporting_cluster_id"])
        columns["unattributable_reason"].append(row["unattributable_reason"])
        columns["event_type"].append(str(row["event_type"]))
        columns["event_precedence"].append(int(row["event_precedence"]))
        columns["event_count"].append(int(row["event_count"]))
        columns["amount_sum"].append(
            None if row["amount_sum"] is None else float(row["amount_sum"])
        )
        columns["first_event_ordinal"].append(int(row["first_event_ordinal"]))
        columns["last_event_ordinal"].append(int(row["last_event_ordinal"]))
        columns["first_event_ts_utc"].append(str(row["first_event_ts_utc"]))
        columns["last_event_ts_utc"].append(str(row["last_event_ts_utc"]))
    return pa.Table.from_pydict(columns, schema=ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA)


def summary_parquet_bytes(table: pa.Table) -> bytes:
    """Deterministic ZSTD Parquet bytes of a summary table (the D15 writer's
    settings; no pandas metadata)."""

    if arrow_schema_hash(table.schema) != ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH:
        raise ValueError("the table is not an account_event_regime_summary table")
    sink = io.BytesIO()
    pq.write_table(
        table.replace_schema_metadata(None), sink, compression="zstd", row_group_size=65_536
    )
    return sink.getvalue()


def read_account_event_regime_summary(data: bytes) -> pa.Table:
    """Parse summary bytes and refuse any schema drift (names / types)."""

    table = pq.read_table(io.BytesIO(data))
    if arrow_schema_hash(table.schema) != ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH:
        raise ValueError("stored event-regime summary does not carry the registered schema")
    return table


def _new_group(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path_instance_id": str(row["path_instance_id"]),
        "clock_policy_id": str(row["clock_policy_id"]),
        "regime_stratum": str(row["regime_stratum"]),
        "canonical_reporting_cluster_id": _cluster_of(str(row["regime_stratum"])),
        "unattributable_reason": (
            None
            if row["unattributable_reason"] == _NO_REASON
            else str(row["unattributable_reason"])
        ),
        "event_type": str(row["event_type"]),
        "event_precedence": int(row["event_precedence"]),
        "event_count": int(row["event_count"]),
        "amount_sum": (float(row["amount_total"]) if int(row["amount_count"]) > 0 else None),
        "first_event_ordinal": int(row["first_event_ordinal"]),
        "last_event_ordinal": int(row["last_event_ordinal"]),
        "first_event_ts_utc": str(row["first_event_ts_utc"]),
        "last_event_ts_utc": str(row["last_event_ts_utc"]),
    }


def _row_budget_refusal(
    core_replay_id: str, budget: EventRegimeSummaryBudget
) -> EventRegimeSummaryBudgetError:
    return EventRegimeSummaryBudgetError(
        f"the account_event_regime_summary of child {core_replay_id[:12]}… would exceed the "
        f"registered row budget ({budget.max_summary_rows} rows, {budget.budget_id}); refused "
        "before publication"
    )


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
    ``evidence`` keys every summary row. Streams the partitions; refuses
    (typed) before publication when the summary exceeds ``budget``."""

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
    strata: list[PropStratumRow] = []
    detail: dict[str, Any] = {
        "summary": {
            "sidecar": ACCOUNT_EVENT_REGIME_SUMMARY_SIDECAR,
            "schema_version": ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_VERSION,
            "schema_hash": ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH,
            "row_keys": list(SUMMARY_ROW_KEYS),
            "budget": budget.model_dump(mode="json"),
        },
        "simulations": {},
    }
    summary_rows: list[dict[str, Any]] = []
    events_total = 0
    events_attributed = 0
    unattributable: dict[str, int] = {}
    not_persisted: list[str] = []
    for simulation_id in sorted(simulations):
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
        groups: dict[tuple, dict[str, Any]] = {}
        seen_paths: set[str] = set()
        clocks: set[str] = set()
        partitions = 0
        for frame in frames:
            partitions += 1
            events = _validate_detail(frame)
            paths = set(events["path_instance_id"].astype(str))
            if paths & seen_paths:
                raise ValueError("event detail repeats a path_instance_id across partitions")
            seen_paths |= paths
            clocks |= set(events["clock_policy_id"].astype(str))
            regime, reason = _attribute_partition(
                events,
                regime_of_trade=regime_of_trade,
                historical=historical,
                panel_assigner=panel_assigner,
            )
            for row in _partition_groups(events, regime, reason).to_dict(orient="records"):
                key = tuple(row[name] for name in SUMMARY_ROW_KEYS)
                target = groups.get(key)
                if target is None:
                    groups[key] = _new_group(row)
                else:
                    _merge_group(target, row)
            if len(summary_rows) + len(groups) > budget.max_summary_rows:
                raise _row_budget_refusal(core_replay_id, budget)
        rows = list(groups.values())
        count = int(sum(row["event_count"] for row in rows))
        attributed = int(
            sum(
                row["event_count"]
                for row in rows
                if row["canonical_reporting_cluster_id"] is not None
            )
        )
        reasons: dict[str, int] = {}
        for row in rows:
            if row["canonical_reporting_cluster_id"] is None:
                label = str(row["unattributable_reason"])
                reasons[label] = reasons.get(label, 0) + int(row["event_count"])
        events_total += count
        events_attributed += attributed
        for label, reason_count in reasons.items():
            unattributable[label] = unattributable.get(label, 0) + reason_count
        common: dict[str, Any] = {
            "firm_label": firm_label,
            "mode": mode,
            "simulation_id": simulation_id,
            "simulation_total": count,
        }
        covered = [row for row in rows if row["canonical_reporting_cluster_id"] is not None]
        strata.append(_stratum_row(RegimeStratumKey(stratum="pooled_all"), rows, **common))
        strata.append(
            _stratum_row(RegimeStratumKey(stratum="pooled_regime_covered"), covered, **common)
        )
        for cluster in sorted({int(row["canonical_reporting_cluster_id"]) for row in covered}):
            strata.append(
                _stratum_row(
                    RegimeStratumKey(stratum="regime", canonical_reporting_cluster_id=cluster),
                    [row for row in covered if row["canonical_reporting_cluster_id"] == cluster],
                    **common,
                )
            )
        strata.append(
            _stratum_row(
                RegimeStratumKey(stratum="unassigned"),
                [row for row in rows if row["canonical_reporting_cluster_id"] is None],
                **common,
            )
        )
        for row in rows:
            summary_rows.append(
                {
                    **row,
                    "account_simulation_id": simulation_id,
                    "firm_label": firm_label,
                    "simulation_mode": mode,
                }
            )
        detail["simulations"][simulation_id] = {
            "firm_label": firm_label,
            "simulation_mode": mode,
            "clock_policy_ids": sorted(clocks),
            "partitions": partitions,
            "paths": len(seen_paths),
            "events_total": count,
            "events_attributed": attributed,
            "unattributable_by_reason": {k: int(v) for k, v in sorted(reasons.items())},
            "summary_rows": len(rows),
        }
    if len(summary_rows) > budget.max_summary_rows:
        raise _row_budget_refusal(core_replay_id, budget)
    table = _summary_table(summary_rows, core_replay_id=core_replay_id, evidence=evidence)
    data = summary_parquet_bytes(table)
    if len(data) > budget.max_published_bytes:
        raise EventRegimeSummaryBudgetError(
            f"the account_event_regime_summary of child {core_replay_id[:12]}… serializes to "
            f"{len(data)} bytes, over the registered byte budget ({budget.max_published_bytes}, "
            f"{budget.budget_id}); refused before publication"
        )
    detail["summary"]["rows"] = int(table.num_rows)
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
        summary_rows=int(table.num_rows),
    )
    return StratifiedPropResult(
        body=body,
        detail=detail,
        summary=table,
        summary_bytes=data,
        summary_sha256=bytes_sha256(data),
        summary_schema_hash=ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA_HASH,
    )
