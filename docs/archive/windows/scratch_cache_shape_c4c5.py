"""CACHE_SHAPE_RECON PART C — C4/C5 size arithmetic from the recorded run counts.

Reads the staged result JSONs (counts produced by scratch_cache_shape_timing.py)
and computes:
  C4: estimated on-disk size of one day's DRAINED stream serialized to parquet
      with the dtypes the reader produces. ASSUMPTION (verbatim): raw fixed-width
      columns, no parquet encoding/dictionary/compression.
      Trade columns (strategy_core/types.py:58-70; reader decode widths in
      data/databento_parquet.py):
        event_ts_utc  int64 ns  8 B  (ts_sort = raw ns .view("i8").astype(int64),
                                      databento_parquet.py:618, ns branch :620-629)
        price_ticks   int64     8 B  (_grid_ticks -> astype(np.int64), :684;
                                      prealloc :900)
        size          int64     8 B  (_strict_size -> _int64_values cast pa.int64,
                                      :655-661, :695-707; prealloc :901)
        side          1 B assumed    (reader emits python str|None 'A'/'B'/'N',
                                      :848-858, :985-994; fixed-width 1-byte code
                                      ASSUMED for the estimate)
        -> 25 B/trade
      Quote columns (types.py:73-87):
        event_ts_utc    int64 ns 8 B (:618-629)
        bid_price_ticks int64    8 B (_grid_ticks :934, astype :684; prealloc :910)
        ask_price_ticks int64    8 B (_grid_ticks :935; prealloc :911)
        bid_size        int64    8 B (_optional_size astype(np.int64) :727;
                                      prealloc :912)
        ask_size        int64    8 B (:939-941; prealloc :913)
        -> 40 B/quote
  C5: same arithmetic for a bars-only artifact (Bar fields,
      strategy_core/types.py:90-112), per-bar widths:
        timeframe_ticks int64 8; trading_day date32 4 (days-since-epoch ASSUMED);
        bar_index int64 8; bar_id raw string bytes at the day's OBSERVED total
        length (1 B/char ASSUMED, no encoding); open_ts_utc int64 8;
        close_ts_utc int64 8; open/high/low/close_ticks int64 4x8=32;
        volume int64 8; trade_count int64 8; is_complete bool 1; is_partial
        bool 1; close_reason raw string bytes (observed 'complete' = 8 B/bar on
        both days; 1 B/char ASSUMED).
        fixed portion = 8+4+8+8+8+32+8+8+1+1 = 86 B/bar (+ bar_id + close_reason).
"""

import json
from pathlib import Path

_SCRATCH = Path(r"C:\Users\gonza\Documents\Claude-Quant-Lab\_scratch_timing")

TRADE_ROW_B = 8 + 8 + 8 + 1          # ts, price_ticks, size, side(1B assumed)
QUOTE_ROW_B = 8 + 8 + 8 + 8 + 8      # ts, bid_ticks, ask_ticks, bid_size, ask_size
BAR_FIXED_B = 8 + 4 + 8 + 8 + 8 + 32 + 8 + 8 + 1 + 1  # everything but bar_id/close_reason

out = {}
for day in ("2025-12-08", "2026-03-02"):
    v = json.loads((_SCRATCH / f"result_staged_{day}.json").read_text("utf-8"))["volumes"]
    n_tr = v["drained_trades"]
    n_qt = v["drained_quotes_after_tob_dedup"]
    n_bars = v["completed_bars_147t"]
    bar_id_chars = json.loads(
        (_SCRATCH / f"result_staged_{day}.json").read_text("utf-8")
    )["stages"][
        "(b) runtime fold LUMPED (bar fold + level/session + per-bar zone rebuild"
        " + touch detect; re-pays decode; incl. compact-array capture)"
    ]["bar_id_total_chars"]
    close_reason_b = 8 * n_bars  # observed: every completed bar close_reason='complete'
    trades_b = n_tr * TRADE_ROW_B
    quotes_b = n_qt * QUOTE_ROW_B
    bars_b = n_bars * BAR_FIXED_B + bar_id_chars + close_reason_b
    out[day] = {
        "c4_trades_rows": n_tr,
        "c4_trades_bytes": trades_b,
        "c4_quotes_rows": n_qt,
        "c4_quotes_bytes": quotes_b,
        "c4_total_bytes": trades_b + quotes_b,
        "c4_total_MiB": round((trades_b + quotes_b) / 2**20, 1),
        "c4_vs_raw_day_file_bytes": v["files"][-1]["bytes"],
        "c5_bars": n_bars,
        "c5_bar_id_total_chars": bar_id_chars,
        "c5_close_reason_bytes": close_reason_b,
        "c5_total_bytes": bars_b,
        "c5_total_KiB": round(bars_b / 1024, 1),
        "c5_mean_bytes_per_bar": round(bars_b / n_bars, 2),
    }

print(json.dumps(out, indent=2))
(_SCRATCH / "c4c5.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
