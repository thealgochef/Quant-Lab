"""Inspect cached Databento OHLCV data to diagnose rendering issues."""
import pandas as pd
from pathlib import Path

# Find cached files - search recursively
data_dir = Path("data/databento/NQ")
if not data_dir.exists():
    print(f"Data directory not found: {data_dir}")
    exit(1)

ohlcv_files = list(data_dir.rglob("ohlcv_*.parquet"))
if not ohlcv_files:
    print("No cached OHLCV files found")
    exit(1)

# Inspect the most recent file
latest = max(ohlcv_files, key=lambda p: p.stat().st_mtime)
print(f"Inspecting: {latest.name}\n")

df = pd.read_parquet(latest)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 10 rows:")
print(df.head(10))

# Check OHLC relationships
print(f"\n=== OHLC Validation ===")
print(f"Rows where open==close==high==low: {((df['open'] == df['close']) & (df['close'] == df['high']) & (df['high'] == df['low'])).sum()}")
print(f"Mean candle range (high-low): {(df['high'] - df['low']).mean():.4f}")
print(f"Median candle range: {(df['high'] - df['low']).median():.4f}")
print(f"Min candle range: {(df['high'] - df['low']).min():.4f}")
print(f"Max candle range: {(df['high'] - df['low']).max():.4f}")

# Sample a few bars
print(f"\n=== Sample bars (last 5) ===")
for idx in range(max(0, len(df)-5), len(df)):
    row = df.iloc[idx]
    print(f"[{idx}] O={row['open']:.2f} H={row['high']:.2f} L={row['low']:.2f} C={row['close']:.2f} | Range={(row['high']-row['low']):.2f}")
