"""INGEST: convert a Databento MBP-1 batch download into per-day store files.

Writes ``data/databento/NQ/{YYYY-MM-DD}/mbp1.parquet`` using the exact
``process_batch_download.py`` transform core (``DBNStore.from_file`` ->
``to_df()`` library defaults -> write-time calendar-spread filter ->
``df.to_parquet``), so converted days carry the store fingerprint: pretty
float-dollar prices, tz-aware ns timestamps, mapped ``symbol`` column,
``ts_recv`` as the pandas index. Additions over that script:

- input is the portal zip OR a directory of already-extracted ``.dbn.zst``
- ``--workers N`` (default 2) parallel day conversion (process pool)
- per-day progress line ``day | rows | seconds | status`` appended to INGEST.log
- per-day sanity gates: non-empty after filtering, ``ts_recv`` within the
  file's UTC calendar day, and a written-schema pin against the store layout
  (level-00-only projection of the mbp10 store schema)
- failed days are reported and conversion continues; exit code 1 if any failed

The batch job is NQ.FUT parent symbology -> the files carry NQ outrights
(front + back months) plus calendar spreads; spreads are dropped at write
time, outrights are all kept — front-month selection stays a read-time
concern, exactly like the existing mbp10 store era built by
``process_batch_download.py``.

Usage:
    python scripts/ingest_databento_batch.py <zip-or-dir> [--workers 2]
        [--start-date 2026-01-11] [--end-date 2026-03-02]
        [--log INGEST.log]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

# ── Config (mirrors process_batch_download.py) ──────────────────────
SYMBOL = "NQ"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "databento"
DATE_RE = re.compile(r"glbx-mdp3-(\d{4})(\d{2})(\d{2})\.(mbp-\d+|trades)")
LOW_ROW_WARN = 1000

#: Store fingerprint pin: the level-00 projection of the mbp10 store schema
#: (column order included; ts_recv is the pandas index, physically last).
EXPECTED_MBP1_SCHEMA: tuple[tuple[str, str], ...] = (
    ("ts_event", "timestamp[ns, tz=UTC]"),
    ("rtype", "uint8"),
    ("publisher_id", "uint16"),
    ("instrument_id", "uint32"),
    ("action", "string"),
    ("side", "string"),
    ("depth", "uint8"),
    ("price", "double"),
    ("size", "uint32"),
    ("flags", "uint8"),
    ("ts_in_delta", "int32"),
    ("sequence", "uint32"),
    ("bid_px_00", "double"),
    ("ask_px_00", "double"),
    ("bid_sz_00", "uint32"),
    ("ask_sz_00", "uint32"),
    ("bid_ct_00", "uint32"),
    ("ask_ct_00", "uint32"),
    ("symbol", "string"),
    ("ts_recv", "timestamp[ns, tz=UTC]"),
)


@dataclass(frozen=True)
class DayJob:
    member: str  # zip member name or absolute file path (dir mode)
    date_str: str  # YYYY-MM-DD from the FILENAME (store day-dir convention)
    schema_raw: str  # e.g. "mbp-1"
    out_path: Path


@dataclass(frozen=True)
class DayResult:
    date_str: str
    status: str  # OK | EXISTS | FAILED
    rows: int
    seconds: float
    message: str = ""
    out_mb: float = 0.0


def plan_jobs(
    names: list[str],
    *,
    out_root: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[DayJob]:
    """Map ``.dbn.zst`` names to per-day jobs; date + schema from the filename."""
    jobs: list[DayJob] = []
    for name in sorted(names):
        if not name.endswith(".dbn.zst"):
            continue
        m = DATE_RE.search(name)
        if not m:
            continue
        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        schema_raw = m.group(4)
        filename = schema_raw.replace("-", "") + ".parquet"  # "mbp-1" -> "mbp1.parquet"
        jobs.append(DayJob(name, date_str, schema_raw, out_root / date_str / filename))
    return jobs


def target_exists_and_loads(path: Path) -> bool:
    """Resume predicate: the target exists AND opens as a non-empty parquet."""
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).metadata.num_rows > 0
    except Exception:
        return False


def filter_spreads(df):
    """Write-time calendar-spread drop — the process_batch_download.py rule."""
    if "symbol" not in df.columns:
        return df, 0
    before = len(df)
    df = df[~df["symbol"].str.contains("-", na=False)]
    return df, before - len(df)


def sanity_problems(df, date_str: str) -> tuple[list[str], list[str]]:
    """(fatal, warnings). Fatal: empty frame; ts_recv outside the UTC calendar
    day (the batch files are split by ts_recv at UTC midnight — ts_event may
    legitimately reach back via snapshot seed rows). Warnings mirror
    process_batch_download.py's non-fatal checks."""
    fatal: list[str] = []
    warns: list[str] = []
    if df.empty:
        return ["0 rows after filtering"], warns
    day_start = datetime.fromisoformat(date_str).replace(tzinfo=UTC)
    day_end = day_start + timedelta(days=1)
    ts_min, ts_max = df.index.min(), df.index.max()
    if ts_min < day_start or ts_max >= day_end:
        fatal.append(f"ts_recv outside day window: [{ts_min}, {ts_max}]")
    if len(df) < LOW_ROW_WARN:
        warns.append(f"low row count ({len(df):,})")
    if "price" in df.columns:
        n_bad = int(((df["price"] <= 0) | (df["price"] > 100_000)).sum())
        if n_bad:
            warns.append(f"{n_bad:,} invalid prices (<=0 or >100k)")
    return fatal, warns


def schema_problems(path: Path) -> list[str]:
    """Pin the written file against the store fingerprint (names+types, order)."""
    import pyarrow.parquet as pq

    written = [(f.name, str(f.type)) for f in pq.read_schema(path)]
    if written != list(EXPECTED_MBP1_SCHEMA):
        expected = dict(EXPECTED_MBP1_SCHEMA)
        got = dict(written)
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        wrong = sorted(
            f"{name}: {got[name]} != {expected[name]}"
            for name in set(got) & set(expected)
            if got[name] != expected[name]
        )
        order = [] if sorted(written) == sorted(EXPECTED_MBP1_SCHEMA) else ["column order differs"]
        return [
            "; ".join(
                filter(
                    None,
                    [
                        f"missing={missing}" if missing else "",
                        f"extra={extra}" if extra else "",
                        f"type={wrong}" if wrong else "",
                        *order,
                    ],
                )
            )
        ]
    return []


def convert_one(input_path: str, is_zip: bool, job: DayJob) -> DayResult:
    """Worker: one day file -> mbp1.parquet with the store-fingerprint core."""
    import databento as db

    started = time.perf_counter()
    tmp_path: Path | None = None
    try:
        if is_zip:
            tmp_dir = DATA_DIR / "_processing_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / job.member.split("/")[-1]
            with zipfile.ZipFile(input_path) as zf, zf.open(job.member) as src, open(
                tmp_path, "wb"
            ) as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
            dbn_path = tmp_path
        else:
            dbn_path = Path(job.member)

        store = db.DBNStore.from_file(str(dbn_path))
        df = store.to_df()  # library defaults: pretty px/ts, mapped symbols
        del store
        gc.collect()

        df, n_spreads = filter_spreads(df)
        fatal, warns = sanity_problems(df, job.date_str)
        if fatal:
            return DayResult(
                job.date_str, "FAILED", len(df), time.perf_counter() - started,
                "sanity: " + "; ".join(fatal),
            )

        job.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(job.out_path)
        rows = len(df)
        del df
        gc.collect()

        pin = schema_problems(job.out_path)
        if pin:
            job.out_path.unlink(missing_ok=True)  # never leave an off-fingerprint day
            return DayResult(
                job.date_str, "FAILED", rows, time.perf_counter() - started,
                "schema pin: " + "; ".join(pin),
            )

        notes = []
        if n_spreads:
            notes.append(f"filtered {n_spreads:,} spread rows")
        notes.extend(warns)
        return DayResult(
            job.date_str, "OK", rows, time.perf_counter() - started,
            "; ".join(notes), job.out_path.stat().st_size / 1024 / 1024,
        )
    except Exception as exc:  # failed days are reported and conversion continues
        return DayResult(
            job.date_str, "FAILED", 0, time.perf_counter() - started,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                os.remove(tmp_path)


def format_log_line(result: DayResult) -> str:
    line = (
        f"{result.date_str} | rows={result.rows:,} | {result.seconds:.1f}s | {result.status}"
    )
    if result.status == "OK":
        line += f" ({result.out_mb:.1f} MB)"
    if result.message:
        line += f" | {result.message}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="portal batch zip OR directory of .dbn.zst files")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-date", default=None, help="inclusive YYYY-MM-DD filter")
    parser.add_argument("--end-date", default=None, help="inclusive YYYY-MM-DD filter")
    parser.add_argument(
        "--log", type=Path, default=Path(__file__).resolve().parent.parent / "INGEST.log"
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}")
        return 2
    is_zip = args.input.is_file()
    if is_zip:
        with zipfile.ZipFile(args.input) as zf:
            names = zf.namelist()
    else:
        names = [str(p) for p in args.input.glob("*.dbn.zst")]

    out_root = DATA_DIR / SYMBOL
    jobs = plan_jobs(
        names, out_root=out_root, start_date=args.start_date, end_date=args.end_date
    )
    if not jobs:
        print("no matching .dbn.zst day files in input")
        return 2

    pending = [job for job in jobs if not target_exists_and_loads(job.out_path)]
    skipped_existing = len(jobs) - len(pending)
    started = time.perf_counter()

    def log(text: str) -> None:
        print(text, flush=True)
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")

    log(
        f"=== ingest_databento_batch {datetime.now(UTC).isoformat(timespec='seconds')} | "
        f"input={args.input} | days={len(jobs)} (skip-existing={skipped_existing}, "
        f"convert={len(pending)}) | workers={args.workers} ==="
    )

    results: list[DayResult] = []
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(convert_one, str(args.input), is_zip, job): job for job in pending
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log(format_log_line(result))

    converted = sum(1 for r in results if r.status == "OK")
    failed = sorted(r.date_str for r in results if r.status == "FAILED")
    wall = time.perf_counter() - started
    log(
        f"=== done | converted={converted} skipped-existing={skipped_existing} "
        f"failed={len(failed)} | wall={wall:.0f}s ==="
    )
    if failed:
        log(f"=== failed days: {', '.join(failed)} ===")

    tmp_dir = DATA_DIR / "_processing_tmp"
    if tmp_dir.exists():
        with contextlib.suppress(Exception):
            for leftover in tmp_dir.iterdir():
                os.remove(leftover)
            tmp_dir.rmdir()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
