"""
Process Databento batch download zip into per-day Parquet files.

Extracts .dbn.zst files from the portal download zip, converts each to
a Parquet file, and places it in the expected directory structure:

    data/databento/NQ/{YYYY-MM-DD}/mbp10.parquet

Filters out calendar spread symbols (e.g. "NQH6-NQH7"), keeping only
the primary front-month contract.

Usage:
    python scripts/process_batch_download.py <path_to_zip>
    python scripts/process_batch_download.py  # auto-finds in Downloads
"""

from __future__ import annotations

import gc
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import databento as db

# ── Config ──────────────────────────────────────────────────────────
SYMBOL = "NQ"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "databento"
# Regex to extract date from filenames like "glbx-mdp3-20251121.mbp-10.dbn.zst"
DATE_RE = re.compile(r"glbx-mdp3-(\d{4})(\d{2})(\d{2})\.(mbp-\d+|trades)")


def find_zip_in_downloads() -> Path | None:
    """Auto-find the Databento zip in the Downloads folder."""
    downloads = Path.home() / "Downloads"
    for f in sorted(downloads.glob("GLBX-*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        return f
    return None


def process_zip(zip_path: Path) -> None:
    """Extract and convert all .dbn.zst files from the zip."""
    print(f"Processing: {zip_path}")
    print(f"Output dir: {DATA_DIR / SYMBOL}")
    print()

    zf = zipfile.ZipFile(zip_path, "r")
    dbn_files = [n for n in zf.namelist() if n.endswith(".dbn.zst")]
    print(f"Found {len(dbn_files)} .dbn.zst files to process")
    print()

    processed = 0
    skipped = 0
    errors = 0

    for i, name in enumerate(sorted(dbn_files), 1):
        # Extract date from filename
        m = DATE_RE.search(name)
        if not m:
            print(f"  [{i}/{len(dbn_files)}] SKIP (no date match): {name}")
            skipped += 1
            continue

        year, month, day = m.group(1), m.group(2), m.group(3)
        schema_raw = m.group(4)  # e.g. "mbp-10", "mbp-1", "trades"
        date_str = f"{year}-{month}-{day}"

        # Schema -> filename mapping
        schema_filename = schema_raw.replace("-", "") + ".parquet"  # "mbp-10" -> "mbp10.parquet"

        out_dir = DATA_DIR / SYMBOL / date_str
        out_path = out_dir / schema_filename

        if out_path.exists():
            size_mb = out_path.stat().st_size / 1024 / 1024
            print(f"  [{i}/{len(dbn_files)}] EXISTS ({size_mb:.1f} MB): {date_str}/{schema_filename}")
            skipped += 1
            continue

        info = zf.getinfo(name)
        size_mb = info.file_size / 1024 / 1024
        print(f"  [{i}/{len(dbn_files)}] Processing {date_str} ({size_mb:.1f} MB compressed)...", end="", flush=True)

        try:
            # Use a persistent temp dir to avoid Windows file locking issues
            tmp_dir = DATA_DIR / "_processing_tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmp_dir / name.split("/")[-1]

            # Extract .dbn.zst from zip
            with zf.open(name) as src, open(tmp_path, "wb") as dst:
                while True:
                    chunk = src.read(8 * 1024 * 1024)  # 8MB chunks
                    if not chunk:
                        break
                    dst.write(chunk)

            # Convert using databento's DBNStore
            store = db.DBNStore.from_file(str(tmp_path))
            df = store.to_df()
            del store  # Release file handle
            gc.collect()

            if df.empty:
                print(" EMPTY (0 rows)")
                skipped += 1
                _safe_remove(tmp_path)
                continue

            # Filter out calendar spreads (e.g. "NQH6-NQH7")
            # Keep only single-leg contracts like "NQH6", "NQZ5"
            if "symbol" in df.columns:
                before = len(df)
                df = df[~df["symbol"].str.contains("-", na=False)]
                filtered = before - len(df)
                if filtered > 0:
                    print(f" (filtered {filtered:,} spread rows)", end="", flush=True)

            if df.empty:
                print(" EMPTY after filtering")
                skipped += 1
                _safe_remove(tmp_path)
                continue

            # Save as Parquet
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path)
            del df
            gc.collect()

            out_mb = out_path.stat().st_size / 1024 / 1024
            print(f" OK ({out_mb:.1f} MB)")
            processed += 1

            # Clean up temp file
            _safe_remove(tmp_path)

        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1

    print()
    print(f"Done! Processed: {processed}, Skipped: {skipped}, Errors: {errors}")

    # Clean up temp dir
    tmp_dir = DATA_DIR / "_processing_tmp"
    if tmp_dir.exists():
        try:
            for f in tmp_dir.iterdir():
                _safe_remove(f)
            tmp_dir.rmdir()
        except Exception:
            pass

    # Show summary of available dates
    if (DATA_DIR / SYMBOL).exists():
        dates = sorted(
            d.name for d in (DATA_DIR / SYMBOL).iterdir()
            if d.is_dir() and any(d.glob("*.parquet"))
        )
        print(f"\nAvailable dates for {SYMBOL}: {len(dates)}")
        if dates:
            print(f"  Range: {dates[0]} to {dates[-1]}")


def _safe_remove(path: Path) -> None:
    """Remove a file, ignoring errors (Windows file locking)."""
    try:
        os.remove(path)
    except Exception:
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        zip_path = Path(sys.argv[1])
    else:
        zip_path = find_zip_in_downloads()
        if zip_path is None:
            print("No GLBX-*.zip found in Downloads. Pass path as argument.")
            sys.exit(1)

    if not zip_path.exists():
        print(f"File not found: {zip_path}")
        sys.exit(1)

    process_zip(zip_path)
