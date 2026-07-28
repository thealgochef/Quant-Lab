"""INGEST P4a OVERLAP IDENTITY: original mbp10.parquet vs converted mbp1.parquet.

Both files drain through ``DatabentoParquetSource`` separately (single-file
drains, full-file window, ``front_month_only=True`` — the serving posture),
then compare with the W3a proof pattern: count, order, type, every field,
ns-exact timestamps — three projections, all streaming:

1. TRADE stream (D-P-17 classifies mbp-1 action='T' rows like mbp-10):
   exact event-by-event identity, no tolerance.
2. QUOTE stream (L1-TOB-deduped): exact event-by-event identity, no tolerance.
3. MERGED stream: exact interleave, EXCEPT that an ns-tied cluster split by a
   reader batch boundary (per-batch sort, W3a contract item 3 — the two files
   have different row counts, so 65,536-row boundaries land at different
   events) may interleave trade-vs-quote differently; such a cluster is
   accepted IFF both sides are exact permutations (multiset-equal on every
   field) and is REPORTED. First observed on 2026-02-12 where the converted
   file's batch boundary at row 3,407,872 = 52*65,536 splits the 14:51:17
   fill+cancel pair (venue gives the trade and its book update THE SAME
   sequence number) — underlying rows byte-identical in both files.

Any other divergence: the first differing event is printed with both values
and the run exits 1 (STOP-the-window per the work order). Warnings are
compared by full dict MINUS 'source' (the source is the file name, which
differs by construction); zero warnings are expected on store days.

Writes INGEST_IDENTITY.log at the QL root. Re-runnable:
    python scratch_ingest_identity.py [days...]
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from strategy_core.data.databento_parquet import DatabentoParquetSource
from strategy_core.data.events import DataQualityWarning
from strategy_core.types import Quote, Trade

STORE = Path(r"C:/Users/gonza/Documents/Claude-Quant-Lab/data/databento/NQ")
DEFAULT_DAYS = ("2026-01-12", "2026-02-12", "2026-02-20")
LOG = Path(__file__).resolve().parent / "INGEST_IDENTITY.log"


def fields(event):
    if isinstance(event, DataQualityWarning):
        payload = event.to_dict()
        payload.pop("source", None)  # file-name-only; differs by construction
        return ("warning", payload)
    if isinstance(event, Trade):
        return ("trade", event.event_ts_utc, event.price_ticks, event.size, event.side)
    return (
        "quote",
        event.event_ts_utc,
        event.bid_price_ticks,
        event.ask_price_ticks,
        event.bid_size,
        event.ask_size,
    )


def source_for(day: str, filename: str, schema: str) -> DatabentoParquetSource:
    path = STORE / day / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return DatabentoParquetSource(
        paths=(path,), requested_symbol="NQ", schema=schema, front_month_only=True
    )


def ts_key(event):
    ts = event.event_ts_utc
    return ts.value if isinstance(ts, pd.Timestamp) else ts


def _next(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


def compare_stream(out, kind: str, left_iter, right_iter) -> tuple[bool, int]:
    """Exact event-by-event identity of one projection (type, every field,
    ns-exact ts), streaming. The per-type order is (ts, seq)-deterministic
    across batch boundaries, so no tolerance is applied here."""
    sentinel = object()
    n = 0
    for index, (old, new) in enumerate(zip_longest(left_iter, right_iter, fillvalue=sentinel)):
        if old is sentinel or new is sentinel:
            out(f"  DIVERGENCE ({kind}) at event {index}: stream length mismatch")
            out(f"    mbp10: {'<exhausted>' if old is sentinel else repr(old)}")
            out(f"    mbp1:  {'<exhausted>' if new is sentinel else repr(new)}")
            return False, n
        if fields(old) != fields(new):
            out(f"  DIVERGENCE ({kind}) at event {index}:")
            out(f"    mbp10: {old!r}")
            out(f"    mbp1:  {new!r}")
            return False, n
        if isinstance(old, Trade | Quote):
            old_ts, new_ts = old.event_ts_utc, new.event_ts_utc
            if isinstance(old_ts, pd.Timestamp) != isinstance(new_ts, pd.Timestamp) or (
                isinstance(old_ts, pd.Timestamp) and old_ts.value != new_ts.value
            ):
                out(f"  DIVERGENCE ({kind}) at event {index}: ts ns mismatch")
                out(f"    mbp10: {old_ts!r}")
                out(f"    mbp1:  {new_ts!r}")
                return False, n
        n += 1
    return True, n


def compare_merged(out, left_iter, right_iter) -> tuple[bool, int]:
    """Merged-stream comparison with the reader's documented order granularity:
    per-batch sort (W3a contract item 3) means two DIFFERENTLY-SIZED physical
    files can interleave trade-vs-quote differently inside a single ns-tied
    cluster when a batch boundary splits it. Such a cluster is accepted IFF the
    two sides are exact permutations of each other (multiset-equal on every
    field); anything else is a hard divergence. Streaming; returns
    (ok, n_reordered)."""
    reordered = 0
    left_head, right_head = _next(left_iter), _next(right_iter)
    while left_head is not None or right_head is not None:
        if (
            left_head is not None
            and right_head is not None
            and fields(left_head) == fields(right_head)
        ):
            left_head, right_head = _next(left_iter), _next(right_iter)
            continue
        if left_head is None or right_head is None:
            out("  MERGED DIVERGENCE: stream length mismatch")
            out(f"    mbp10: {left_head!r}")
            out(f"    mbp1:  {right_head!r}")
            return False, reordered
        if isinstance(left_head, DataQualityWarning) or isinstance(right_head, DataQualityWarning):
            out("  MERGED DIVERGENCE at a warning event:")
            out(f"    mbp10: {left_head!r}")
            out(f"    mbp1:  {right_head!r}")
            return False, reordered
        ts = ts_key(left_head)
        if ts != ts_key(right_head):
            out("  MERGED DIVERGENCE: heads at different ts")
            out(f"    mbp10: {left_head!r}")
            out(f"    mbp1:  {right_head!r}")
            return False, reordered
        def _market_at(head, at_ts):
            return (
                head is not None
                and not isinstance(head, DataQualityWarning)
                and ts_key(head) == at_ts
            )

        cluster_left, cluster_right = [left_head], [right_head]
        left_head = _next(left_iter)
        while _market_at(left_head, ts):
            cluster_left.append(left_head)
            left_head = _next(left_iter)
        right_head = _next(right_iter)
        while _market_at(right_head, ts):
            cluster_right.append(right_head)
            right_head = _next(right_iter)
        left_multiset = sorted(map(repr, map(fields, cluster_left)))
        right_multiset = sorted(map(repr, map(fields, cluster_right)))
        if left_multiset != right_multiset:
            out(f"  MERGED DIVERGENCE at ns-cluster ts={cluster_left[0].event_ts_utc!r}:")
            out(f"    mbp10 cluster ({len(cluster_left)} events): {cluster_left!r}")
            out(f"    mbp1  cluster ({len(cluster_right)} events): {cluster_right!r}")
            return False, reordered
        reordered += 1
        out(
            f"  reordered ns-cluster #{reordered}: ts={cluster_left[0].event_ts_utc!r} "
            f"({len(cluster_left)} events, permutation-verified)"
        )
    return True, reordered


def compare_day(out, day: str) -> dict:
    def drain(kinds=None):
        src10 = source_for(day, "mbp10.parquet", "mbp-10")
        src1 = source_for(day, "mbp1.parquet", "mbp-1")
        if kinds is None:
            return src10.events(), src1.events()
        return (
            (e for e in src10.events() if isinstance(e, kinds)),
            (e for e in src1.events() if isinstance(e, kinds)),
        )

    trades_ok, n_trades = compare_stream(out, "trades", *drain(Trade))
    quotes_ok, n_quotes = (False, 0)
    warns_ok, n_warns = (False, 0)
    merged_ok, reordered = (False, 0)
    if trades_ok:
        quotes_ok, n_quotes = compare_stream(out, "quotes", *drain(Quote))
    if quotes_ok:
        warns_ok, n_warns = compare_stream(out, "warnings", *drain(DataQualityWarning))
    if warns_ok:
        merged_ok, reordered = compare_merged(out, *drain())
    return {
        "identical": trades_ok and quotes_ok and warns_ok and merged_ok,
        "events": n_trades + n_quotes + n_warns,
        "reordered": reordered,
        "trade": n_trades,
        "quote": n_quotes,
        "warning": n_warns,
    }


def main(days: list[str]) -> int:
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    out("INGEST P4a OVERLAP IDENTITY — original mbp10.parquet vs converted mbp1.parquet")
    out(f"run: {datetime.now().isoformat(timespec='seconds')}")
    out(
        f"env: python {sys.version.split()[0]}  pyarrow {pa.__version__}  "
        f"numpy {np.__version__}  pandas {pd.__version__}"
    )
    out(f"store: {STORE}")
    out(
        "posture: single-file drains, full-file window, front_month_only=True; "
        "compare = type + every field + ns-exact ts, W3a pattern; "
        "expected = exact identity (trades via D-P-17 + L1-TOB-deduped quotes)"
    )
    out("")

    failed = False
    for day in days:
        out(f"== {day} " + "=" * 60)
        raw10 = pq.ParquetFile(STORE / day / "mbp10.parquet").metadata.num_rows
        raw1 = pq.ParquetFile(STORE / day / "mbp1.parquet").metadata.num_rows
        out(f"  raw rows: mbp10={raw10:,}  mbp1={raw1:,}")
        t0 = time.perf_counter()
        result = compare_day(out, day)
        elapsed = time.perf_counter() - t0
        if not result["identical"]:
            out("  RESULT: NOT IDENTICAL — STOP THE WINDOW (see divergence above)")
            failed = True
            break
        out(
            f"  RESULT: IDENTICAL — {result['events']:,} events "
            f"(trades {result['trade']:,} / quotes {result['quote']:,} / "
            f"warnings {result['warning']:,}); trade/quote/warning streams "
            f"exact event-by-event; merged interleave exact except "
            f"{result['reordered']} permutation-verified ns-tied cluster(s) "
            f"(reader per-batch sort granularity, W3a contract item 3) "
            f"in {elapsed:.1f}s"
        )
        out("")

    out("OVERALL: " + ("DIVERGED — see above" if failed else f"IDENTICAL on all {len(days)} days"))
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nlog written: {LOG}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or list(DEFAULT_DAYS)))
