"""Mutable preparation provenance beside immutable day artifacts.

Contract selection uses the canonical reader's instrument-only prescan for the
logical day's primary partition and its exact window row groups. It never
decodes prices, constructs events, or rebuilds bars when backfilling a catalog.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import date
from pathlib import Path

from strategy_core.data.databento_parquet import DatabentoParquetSource

from .data_access import ExplorationDataPolicy


def read_preparation_instrument(day: str, catalog_paths: tuple[Path, ...]) -> int | None:
    """Read a day entry; absent evidence remains absent and conflicting stores fail."""
    values: list[int | None] = []
    for path in catalog_paths:
        if not Path(path).exists():
            continue
        catalog = json.loads(Path(path).read_text(encoding="utf-8"))
        entry = catalog.get("days", {}).get(day)
        if entry is None:
            continue
        value = entry.get("selected_instrument_id")
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("preparation catalog instrument id must be an integer or null")
        values.append(value)
    if values and any(value != values[0] for value in values):
        raise ValueError("preparation catalogs disagree on the selected instrument")
    return values[0] if values else None


def record_preparation_instrument(
    day: str, *, data_dir: Path, symbol: str, catalog_path: Path,
    access_policy: ExplorationDataPolicy,
    source: DatabentoParquetSource | None = None,
) -> dict:
    """Record Core's selected contract without modifying bars, seeds or datasets."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    access_policy.authorize_date(day)
    if source is None:
        source = DatabentoParquetSource.for_trading_day(
            Path(data_dir) / symbol, date.fromisoformat(day), requested_symbol=symbol,
            allowed_source_dates=frozenset(date.fromisoformat(d) for d in access_policy.allowlist),
        )
    index = next(i for i, path in enumerate(source.paths) if path.parent.name == day)
    path = source.paths[index]
    access_policy.resolve_source_path(day, lambda _day: path)
    access_policy.record_metadata_access(day)
    access_policy.record_file_open(day)
    parquet = pq.ParquetFile(path)
    selected = [name for name in ("action", "instrument_id", "raw_symbol", "symbol")
                if name in parquet.schema_arrow.names]
    start, end = source._window_for(index)
    groups = source._window_row_groups(parquet, start, end)
    instrument_id = source._front_month_instrument_id(parquet, selected, groups)
    if "instrument_id" in selected:
        index_rows = (parquet.metadata.num_rows if groups is None else
                      sum(parquet.metadata.row_group(i).num_rows for i in groups))
        access_policy.record_rows_read(day, rows=index_rows)
    raw_symbol = None
    symbol_columns = [name for name in ("raw_symbol", "symbol") if name in selected]
    if instrument_id is not None and symbol_columns:
        columns = ["instrument_id", *symbol_columns]
        table = (parquet.read(columns=columns) if groups is None else
                 parquet.read_row_groups(groups, columns=columns))
        access_policy.record_rows_read(day, rows=table.num_rows)
        table = table.filter(
            pc.fill_null(pc.equal(table["instrument_id"], instrument_id), False)
        )
        # Imported MBP files can use ``symbol`` for the actual raw contract.
        # Match Core's nonempty raw_symbol -> symbol convention, but do not
        # mistake a root/continuous symbol or spread for a raw contract name.
        symbols = pc.cast(table[symbol_columns[0]], pa.string())
        if symbol_columns == ["raw_symbol", "symbol"]:
            symbols = pc.if_else(
                pc.fill_null(pc.not_equal(symbols, ""), False),
                symbols,
                pc.cast(table["symbol"], pa.string()),
            )
        symbols = symbols.drop_null()
        distinct = set(pc.unique(symbols).to_pylist())
        if len(distinct) == 1 and re.fullmatch(
            rf"{re.escape(symbol)}[FGHJKMNQUVXZ]\d{{1,4}}", next(iter(distinct))
        ):
            raw_symbol = next(iter(distinct))
    entry = {"selected_instrument_id": int(instrument_id) if instrument_id is not None else None,
             "raw_symbol": raw_symbol}
    catalog_path = Path(catalog_path).resolve()
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")) if catalog_path.exists() else {}
    days = catalog.setdefault("days", {})
    previous = days.get(day, {})
    if "selected_instrument_id" in previous:
        same_instrument = previous["selected_instrument_id"] == entry["selected_instrument_id"]
        same_symbol = previous.get("raw_symbol") == entry["raw_symbol"]
        symbol_enrichment = previous.get("raw_symbol") is None and entry["raw_symbol"] is not None
        if not same_instrument or not (same_symbol or symbol_enrichment):
            raise ValueError("preparation instrument evidence changed for an existing day")
    days[day] = {**previous, **entry}
    catalog["days"] = dict(sorted(days.items()))
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = catalog_path.with_name(f".{catalog_path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(json.dumps(catalog, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, catalog_path)
    return entry
