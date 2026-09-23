"""One logical day's read-only, chunked DBN/parquet conversion audit.

This is evidence, never an import or a completeness receipt. Full associated
physical files are compared, including rows outside the logical session; no
other DBN dates are opened. Every output uses exclusive creation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
import zipfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from alpha_lab.agents.data_infra.ifvg.development_access import PERMITTED_DEVELOPMENT_DATES
from alpha_lab.agents.data_infra.ifvg.features.mbp1_arrow_schemas import (
    DATABENTO_PRICE_SCALE,
    MBP1_SOURCE_EVENT_SCHEMA,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_coverage_evidence import (
    authorized_session_span_ns,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    ORDER_KEY_COLUMNS,
    normalize_mbp1_events,
)
from alpha_lab.agents.data_infra.ifvg.search.mbp1_vendor_conditions import (
    load_archived_dataset_conditions,
)

AUDIT_VERSION = "mbp1_dbn_parquet_exact_order_snapshot_audit_v1"
CHUNK_ROWS = 200_000
BOOK_COLUMNS = ("bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00")


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, default=str)
        output.write("\n")


def canonical_frame(frame):
    if frame.index.name == "ts_recv":
        frame = frame.reset_index()
    return frame.reset_index(drop=True)


class ParquetRows:
    """Consume any requested number of rows without materializing a file."""

    def __init__(self, path):
        self.batches = iter(pq.ParquetFile(path).iter_batches(batch_size=CHUNK_ROWS))
        self.pending = pd.DataFrame()

    def take(self, count):
        pieces = []
        while count:
            if self.pending.empty:
                try:
                    self.pending = canonical_frame(next(self.batches).to_pandas())
                except StopIteration as exc:
                    raise AssertionError("parquet ended before reconstructed DBN") from exc
            size = min(count, len(self.pending))
            pieces.append(self.pending.iloc[:size])
            self.pending = self.pending.iloc[size:]
            count -= size
        return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

    def assert_exhausted(self):
        if not self.pending.empty or next(self.batches, None) is not None:
            raise AssertionError("parquet contains rows absent from reconstructed DBN")


def assert_exact_rows(actual, expected):
    """Check every scalar and its ordinal; nulls agree, dtypes may serialize differently."""
    if set(actual.columns) != set(expected.columns):
        raise AssertionError("conversion column sets differ")
    pd.testing.assert_frame_equal(
        actual[expected.columns].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
        check_index_type=False,
    )


def stream_stats(frame, totals, previous, midnight, lower, upper):
    totals["kept_rows"] += len(frame)
    ts = frame.ts_event.astype("int64").to_numpy()
    recv = frame.ts_recv.astype("int64").to_numpy()
    for name, values in (("ts_event", ts), ("ts_recv", recv)):
        before = previous.get(name)
        totals[f"{name}_adjacent_reversals"] += int((np.diff(values) < 0).sum())
        totals[f"{name}_adjacent_equalities"] += int((np.diff(values) == 0).sum())
        if before is not None:
            totals[f"{name}_adjacent_reversals"] += int(values[0] < before)
            totals[f"{name}_adjacent_equalities"] += int(values[0] == before)
        previous[name] = int(values[-1])
    flags = frame["flags"].to_numpy()
    totals["snapshot_rows"] += int(((flags & 32) != 0).sum())
    totals["maybe_bad_book_rows"] += int(((flags & 4) != 0).sum())
    totals["bad_receive_timestamp_rows"] += int(((flags & 8) != 0).sum())
    totals["event_time_before_physical_midnight"] += int((ts < midnight).sum())
    totals["event_time_after_or_at_next_midnight"] += int(
        (ts >= midnight + 86_400_000_000_000).sum()
    )
    totals["logical_rows_before_instrument_selection"] += int(((ts >= lower) & (ts < upper)).sum())
    totals["unmapped_symbol_rows"] += int(frame.symbol.isna().sum())


def audit_partition(repo, archive, expected, out, logical_day, session, last_states):
    day, member = expected["source_date"], expected["archive_member"]
    parquet_path = repo / expected["parquet_relative_path"]
    before = parquet_path.stat()
    parquet_sha = sha_file(parquet_path)
    if parquet_sha != expected["parquet_sha256"]:
        raise ValueError(f"original parquet differs from preserved acceptance inventory: {day}")
    extracted = out / member
    if not extracted.exists():
        with archive.open(member) as source, extracted.open("xb") as target:
            while chunk := source.read(8 * 1024 * 1024):
                target.write(chunk)
    compressed_sha = sha_file(extracted)
    if compressed_sha != expected["archive_member_sha256"]:
        raise ValueError(f"compressed DBN differs from preserved acceptance inventory: {day}")
    store = db.DBNStore.from_file(extracted)
    midnight = pd.Timestamp(day, tz="UTC").value
    lower, upper = max(session[0], midnight), min(session[1], midnight + 86_400_000_000_000)
    reader = ParquetRows(parquet_path)
    prior_physical_states = dict(last_states)
    totals, previous = Counter(), {}
    flag_counts, instrument_counts, trade_counts = Counter(), Counter(), Counter()
    logical_instrument_counts = Counter()
    snapshots, sample_parts = [], []
    sample_count = 0
    digest = hashlib.sha256()
    started = time.monotonic()
    for chunk_number, original in enumerate(store.to_df(count=CHUNK_ROWS), 1):
        raw = canonical_frame(original)
        totals["decoded_dbn_rows"] += len(raw)
        spreads = raw.symbol.str.contains("-", na=False)
        totals["filtered_spread_rows"] += int(spreads.sum())
        frame = raw.loc[~spreads].reset_index(drop=True)
        if frame.empty:
            continue
        stored = reader.take(len(frame))
        assert_exact_rows(frame, stored)
        # Diagnostic hash only; exact scalar assertion above establishes equality.
        digest.update(
            pd.util.hash_pandas_object(frame[stored.columns], index=False).to_numpy().tobytes()
        )
        stream_stats(frame, totals, previous, midnight, lower, upper)
        flag_counts.update(map(int, frame["flags"]))
        instrument_counts.update(map(int, frame.instrument_id))
        ts = frame.ts_event.astype("int64")
        logical = frame.loc[ts.ge(lower) & ts.lt(upper)]
        logical_instrument_counts.update(map(int, logical.instrument_id))
        trade_counts.update(map(int, logical.loc[logical.action.eq("T"), "instrument_id"]))
        # Include all snapshot rows; there are normally only a few opening records.
        boundary = frame.loc[(frame["flags"].astype("int64") & 32).ne(0)]
        for _, row in boundary.iterrows():
            record = row.to_dict()
            record["physical_source_date"] = day
            prior = prior_physical_states.get(int(row.instrument_id))
            record["prior_physical_last_record"] = prior
            record["same_prior_event_time_and_sequence"] = bool(
                prior and row.ts_event == prior["ts_event"] and row.sequence == prior["sequence"]
            )
            record["same_prior_book_state"] = bool(
                prior
                and all(
                    (pd.isna(row[key]) and pd.isna(prior[key])) or row[key] == prior[key]
                    for key in BOOK_COLUMNS
                )
            )
            record["clipped_by_current_physical_event_time_policy"] = bool(
                row.ts_event.value < lower or row.ts_event.value >= upper
            )
            snapshots.append(record)
        # A bounded real-record normalization probe; full-file ordering is checked
        # independently by the exact streaming comparison and adjacent diagnostics.
        if sample_count < 10_000 and len(logical):
            sample = logical.head(10_000 - sample_count)
            sample_parts.append(sample)
            sample_count += len(sample)
        for instrument_id, tail in (
            frame.groupby("instrument_id", sort=False).tail(1).set_index("instrument_id").iterrows()
        ):
            last_states[int(instrument_id)] = tail.to_dict()
        if chunk_number % 10 == 0:
            print(
                f"{day}: {totals['decoded_dbn_rows']:,} DBN rows compared "
                f"in {time.monotonic() - started:.1f}s",
                flush=True,
            )
    reader.assert_exhausted()
    after = parquet_path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("source parquet changed while auditing")
    if totals["kept_rows"] != expected["parquet_rows"]:
        raise AssertionError("preserved footer and exact conversion row counts differ")
    # Strategy-Core's verified selector breaks tied trade counts toward larger IDs.
    selected = (
        max(trade_counts, key=lambda value: (trade_counts[value], value)) if trade_counts else None
    )
    probe = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    probe = probe.loc[probe.instrument_id.eq(selected)].copy() if len(probe) else probe
    normalized_probe = None
    if len(probe):
        raw_probe = probe[[field.name for field in MBP1_SOURCE_EVENT_SCHEMA]].copy()
        for column in ("price", "bid_px_00", "ask_px_00"):
            raw_probe[column] = raw_probe[column] / DATABENTO_PRICE_SCALE
        normalized_probe = normalize_mbp1_events(
            raw_probe, instrument="NQ", trading_day=logical_day
        )
        keys = normalized_probe[list(ORDER_KEY_COLUMNS)]
        if not keys.equals(
            keys.sort_values(list(ORDER_KEY_COLUMNS), kind="stable").reset_index(drop=True)
        ):
            raise AssertionError("normalizer does not preserve the pinned total order")
        if set(normalized_probe.source_ordinal) != set(range(len(raw_probe))):
            raise AssertionError("normalizer lost source ordinals")
        normalized_probe.to_parquet(out / f"{day}_normalized_order_probe.parquet", index=False)
        probe.to_parquet(out / f"{day}_raw_order_probe.parquet", index=False)
    result = {
        "source_date": day,
        "parquet_path": str(parquet_path),
        "parquet_sha256": parquet_sha,
        "compressed_dbn_member": member,
        "compressed_dbn_sha256": compressed_sha,
        "dbn_metadata": str(store.metadata),
        "exact_all_scalar_and_original_row_order_comparison": "passed",
        "columns_compared": list(pq.read_schema(parquet_path).names),
        "decoded_retained_row_hash_sha256": digest.hexdigest(),
        "counts": dict(totals),
        "flags_counts": dict(flag_counts),
        "instrument_counts": dict(instrument_counts),
        "logical_trade_counts_by_instrument": dict(trade_counts),
        "logical_rows_by_instrument": dict(logical_instrument_counts),
        "dominant_trade_instrument_probe": selected,
        "logical_selected_rows": logical_instrument_counts[selected],
        "normalization_probe_rows": 0 if normalized_probe is None else len(normalized_probe),
        "snapshot_records": snapshots,
        "elapsed_seconds": time.monotonic() - started,
        "source_completeness_status": "unknown",
    }
    write_json(out / f"{day}_conversion_audit.json", result)
    print(f"{day}: EXACT EQUALITY {totals['kept_rows']:,} retained rows", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-day", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    physical = (
        (date.fromisoformat(args.logical_day) - timedelta(days=1)).isoformat(),
        args.logical_day,
    )
    if any(day not in PERMITTED_DEVELOPMENT_DATES for day in physical):
        raise PermissionError("one-day audit exceeds permitted source dates")
    expected = {
        row["source_date"]: row for row in map(json.loads, args.inventory.read_text().splitlines())
    }
    if any(day not in expected for day in physical):
        raise PermissionError("one-day audit lacks preserved scoped source hashes")
    conditions, warnings = load_archived_dataset_conditions(repo, physical)
    if any(day not in conditions or conditions[day].condition != "available" for day in physical):
        raise ValueError("this bounded audit requires both physical dates to be non-degraded")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    if (output / "audit_summary.json").exists():
        raise FileExistsError("audit output already completed; choose a new output directory")
    session = authorized_session_span_ns(args.logical_day)
    archive_before = args.archive.stat()
    with zipfile.ZipFile(args.archive) as archive:
        partitions, last_states = [], {}
        for day in physical:
            partitions.append(
                audit_partition(
                    repo, archive, expected[day], output, args.logical_day, session, last_states
                )
            )
    archive_after = args.archive.stat()
    if (archive_before.st_size, archive_before.st_mtime_ns) != (
        archive_after.st_size,
        archive_after.st_mtime_ns,
    ):
        raise ValueError("original vendor archive changed while auditing")
    summary = {
        "audit_version": AUDIT_VERSION,
        "logical_day": args.logical_day,
        "physical_dates": physical,
        "session_start_utc": str(pd.Timestamp(session[0], tz="UTC")),
        "session_end_exclusive_utc": str(pd.Timestamp(session[1], tz="UTC")),
        "script_sha256": sha_file(__file__),
        "inventory_sha256": sha_file(args.inventory),
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("databento", "databento-dbn", "pandas", "pyarrow", "numpy")
            },
        },
        "chunk_rows": CHUNK_ROWS,
        "conversion": (
            "DBNStore.to_df defaults; exclude symbol containing '-'; "
            "no sorting, deduplication or front-month filter"
        ),
        "physical_file_conversion_equivalence": "passed_every_scalar_and_retained_ordinal",
        "positive_source_completeness": "unknown",
        "completeness_receipt_created": False,
        "vendor_conditions": {
            day: item.model_dump(mode="json") for day, item in conditions.items()
        },
        "vendor_warnings": warnings,
        "full_associated_physical_rows_including_outside_logical_session": True,
        "dbn_dates_opened": physical,
        "original_files_preserved": True,
        "partitions": partitions,
        "counts": {
            key: sum(item["counts"].get(key, 0) for item in partitions)
            for key in sorted({key for item in partitions for key in item["counts"]})
        },
    }
    write_json(output / "audit_summary.json", summary)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "audit_version",
                    "logical_day",
                    "counts",
                    "physical_file_conversion_equivalence",
                    "positive_source_completeness",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
