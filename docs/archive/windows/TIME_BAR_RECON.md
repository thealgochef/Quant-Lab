# TIME_BAR_RECON â€” adding TIME bar construction alongside the tick path

Read-only static recon. Facts only: what IS, not what should be. No design, no proposals,
no diffs, no recommendations. Every claim carries a `path:line` citation or a quoted grep
result; anything not obtainable under the read-only prohibitions is answered with the
literal string `NOT OBTAINABLE READ-ONLY`.

## Observed tips

All three tips MATCH the last-verified values stated in the task. All file:line citations
below are against these tips.

| Repo | Path | `git rev-parse HEAD` | Expected | Match |
|---|---|---|---|---|
| SC | `C:\Users\gonza\Documents\Strategy-core` | `4740ecdd460caf4ed2c8038fc37c23b2b1d17788` | `4740ecd` | YES |
| TL | `C:\Users\gonza\Documents\Trade-Lab` | `7a911c0ce0ae5d9e2c97339b7d897377a37a3647` | `7a911c0` | YES |
| QL | `C:\Users\gonza\Documents\Claude-Quant-Lab` | `27cdfd6630898c7b31559fbe57f009701f8ed931` | `27cdfd6` | YES |

`git status --porcelain=v2 --branch` (branch lines; all working-tree entries in all three
repos are untracked `?` records â€” **zero tracked files modified or staged in any repo**):

```
=== C:\Users\gonza\Documents\Strategy-core ===
# branch.oid 4740ecdd460caf4ed2c8038fc37c23b2b1d17788
# branch.head platform-refactor
# branch.upstream origin/platform-refactor
# branch.ab +0 -0

=== C:\Users\gonza\Documents\Trade-Lab ===
# branch.oid 7a911c0ce0ae5d9e2c97339b7d897377a37a3647
# branch.head platform-refactor
# branch.upstream origin/platform-refactor
# branch.ab +0 -0

=== C:\Users\gonza\Documents\Claude-Quant-Lab ===
# branch.oid 27cdfd6630898c7b31559fbe57f009701f8ed931
# branch.head platform-refactor
# branch.upstream origin/platform-refactor
# branch.ab +0 -0
```

Tracked-dirty check, run separately (`git status --porcelain` filtered to non-`??` lines):

```
C:\Users\gonza\Documents\Strategy-core     tracked-dirty: NONE
C:\Users\gonza\Documents\Trade-Lab         tracked-dirty: NONE
C:\Users\gonza\Documents\Claude-Quant-Lab  tracked-dirty: NONE
```

Excluded from the sweep by construction: `C:\Users\gonza\Documents\Strategy-Core-verify`
and `C:\Users\gonza\Documents\Trade-Lab-verify` (separate verify clones â€” not cited).

## Method

Three concurrent read-only agents, one per part (A / B / C). Static file reads and ripgrep
only. No build, no training, no replay, no test execution (not even `pytest --collect-only`).
No data read under any NQ day directory dated 2026-06-12..2026-07-10. The single write of
this task is this file, untracked, at the QL repo root.

---


---

## PART A â€” the `Bar` type and its serialization surface

Scope: SC = `C:\Users\gonza\Documents\Strategy-core`, TL = `C:\Users\gonza\Documents\Trade-Lab`, QL = `C:\Users\gonza\Documents\Claude-Quant-Lab`. Static read + grep only; no file in any repo was modified and no test was executed.

---

### A1 â€” the `Bar` dataclass

**Declaration:** `SC/src/strategy_core/types.py:90-91` (decorator on :90, `class Bar:` on :91).

SC/src/strategy_core/types.py:90-121
```python
@dataclass(frozen=True, slots=True)
class Bar:
    """An aggregated tick/time/volume bar. Prices in integer ticks.

    Field layout mirrors Trade-Lab's ``Candle`` so the promoted candle builders
    can emit this type unchanged.
    """

    timeframe_ticks: int
    trading_day: date
    bar_index: int
    bar_id: str
    open_ts_utc: datetime
    close_ts_utc: datetime
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int
    volume: int
    trade_count: int
    is_complete: bool
    is_partial: bool
    close_reason: CloseReason | None = None

    def high_points(self, tick_size: float) -> float:
        return self.high_ticks * tick_size

    def low_points(self, tick_size: float) -> float:
        return self.low_ticks * tick_size

    def close_points(self, tick_size: float) -> float:
        return self.close_ticks * tick_size
```

**Fields, in declaration order** (all from `SC/src/strategy_core/types.py:98-112`):

| # | field | line | annotation | default |
|---|-------|------|-----------|---------|
| 1 | `timeframe_ticks` | :98 | `int` | â€” (required) |
| 2 | `trading_day` | :99 | `date` | â€” |
| 3 | `bar_index` | :100 | `int` | â€” |
| 4 | `bar_id` | :101 | `str` | â€” |
| 5 | `open_ts_utc` | :102 | `datetime` | â€” |
| 6 | `close_ts_utc` | :103 | `datetime` | â€” |
| 7 | `open_ticks` | :104 | `int` | â€” |
| 8 | `high_ticks` | :105 | `int` | â€” |
| 9 | `low_ticks` | :106 | `int` | â€” |
| 10 | `close_ticks` | :107 | `int` | â€” |
| 11 | `volume` | :108 | `int` | â€” |
| 12 | `trade_count` | :109 | `int` | â€” |
| 13 | `is_complete` | :110 | `bool` | â€” |
| 14 | `is_partial` | :111 | `bool` | â€” |
| 15 | `close_reason` | :112 | `CloseReason \| None` | `None` |

Fifteen fields; exactly one (`close_reason`) has a default.

**Frozen / slots:** both. `SC/src/strategy_core/types.py:90` â€” `@dataclass(frozen=True, slots=True)`.

**Custom `__eq__` / `__hash__` / `__post_init__`:** NONE. The class body (`SC/src/strategy_core/types.py:92-121`) contains only the docstring, the 15 field declarations, and three plain helper methods `high_points` (:114), `low_points` (:117), `close_points` (:120). `frozen=True` therefore yields the dataclass-generated field-tuple `__eq__` and `__hash__`.

Supporting type used by field 15 â€” `SC/src/strategy_core/types.py:51-55`:
```python
class CloseReason(StrEnum):
    """Why a bar closed. Mirrors Trade-Lab ``CandleCloseReason`` exactly."""

    COMPLETE = "complete"
    END_OF_DAY = "end_of_day"
```
There is no `INTERVAL`/time-close member.

`Bar` is exported from the package root: `SC/src/strategy_core/__init__.py:138` (`"Bar",` in `__all__`).

**A parallel, structurally identical type lives in TL** â€” `TL/backend/src/trade_lab/domain/candles.py:21-37`, `@dataclass(frozen=True, slots=True) class Candle`, same 15 fields in the same order, with `close_reason: CandleCloseReason | None = None` (:37). It is a DTO/display mirror; TL builds no bars (`TL/backend/src/trade_lab/domain/candles.py:1-8`).

---

### A2 â€” `BarKind`

**Declaration:** `SC/src/strategy_core/strategies/protocols.py:61-65`.

SC/src/strategy_core/strategies/protocols.py:61-65
```python
class BarKind(StrEnum):
    """How a bar closes. ``StrEnum`` so ``BarKind.TICK == "tick"`` (PLAN Â§2.1(3))."""

    TICK = "tick"  # close on trade_count == size      (CURRENT engine)
    TIME = "time"  # close on wall-clock interval edge  (NEW close trigger, Phase F)
```

Members: `TICK = "tick"` (:64), `TIME = "time"` (:65). Two members, no others.

**Every reference, all three trees** (grep `BarKind` across SC, TL, QL):

*SC â€” executable code:*

| path:line | text | classification |
|---|---|---|
| `SC/src/strategy_core/strategies/protocols.py:23` | `* ``BarSpec`` / ``BarKind`` â€” a strategy *declares* the tick AND time barsâ€¦` | module docstring (non-executable prose) |
| `SC/src/strategy_core/strategies/protocols.py:46` | `"BarKind",` inside `__all__` | export declaration (string literal) |
| `SC/src/strategy_core/strategies/protocols.py:61` | `class BarKind(StrEnum):` | **declaration** |
| `SC/src/strategy_core/strategies/protocols.py:62` | class docstring | doc |
| `SC/src/strategy_core/strategies/protocols.py:64` | `TICK = "tick"` | member declaration |
| `SC/src/strategy_core/strategies/protocols.py:65` | `TIME = "time"` | member declaration |
| `SC/src/strategy_core/strategies/protocols.py:79` | `kind: BarKind` (field of `BarSpec`) | type annotation |
| `SC/src/strategy_core/strategies/touch_reversal/plugin.py:60` | `BarKind,` in the import list | import |
| `SC/src/strategy_core/strategies/touch_reversal/plugin.py:249` | `return (BarSpec(kind=BarKind.TICK, size=_DECISION_TIMEFRAME, label=_DECISION_BAR_LABEL),)` | **construction** â€” the ONLY value-level use of a `BarKind` member anywhere |

*SC â€” docs / captured context (non-code):* `SC/docs/PLATFORM_REFACTOR_PLAN.md:266`, `:272`; `SC/b2_context.md:256`, `:279`, `:294`, `:295`, `:312`, `:747`, `:913`.

*TL:* the only hits are in the untracked recon document `TL/PLUGIN_SDK_RECON.md` (lines 54, 55, 175, 264, 266, 336, 544, 554). **No TL source, test, or frontend file references `BarKind`.**

*QL:* `"No matches found"` â€” grep for `BarKind` over the whole `Claude-Quant-Lab` tree returned nothing. Zero references.

**Comparison sites: none.** There is no `== BarKind`, `!= BarKind`, `in (BarKindâ€¦)`, or `match` on a kind anywhere in the three trees. Furthermore the field is never *read*: grep for `\.kind\b` over `SC/src` returns `"No matches found"`, so `BarSpec.kind` is write-only â€” the value set at `plugin.py:249` is never consumed. The registry validates only the container type, not the kind: `SC/src/strategy_core/strategies/registry.py:70` â€” `if not isinstance(bars, tuple) or not bars or not all(isinstance(b, BarSpec) for b in bars):`. Bar routing keys off the *label*, not the kind: `SC/src/strategy_core/runtime/context.py:49-52`, `def _label_to_timeframe(label: str) -> int | None:` â€¦ `re.match(r"\d+", label)`.

**`BarKind.TIME` is therefore dead/unused at runtime today** â€” declared (`protocols.py:65`), never constructed, never compared, never read.

**Explicit answer:** No â€” `BarKind` is never passed to `Bar` at all (`Bar` has no kind field, `SC/src/strategy_core/types.py:98-112`), and the only `BarKind` value ever constructed anywhere in the three trees is `BarKind.TICK` at `SC/src/strategy_core/strategies/touch_reversal/plugin.py:249`, so no code path constructs a `Bar` (or a `BarSpec`) with a TIME kind today.

Corroborating engine-side constraint: the tick engine rejects non-positive timeframes and has no wall-clock trigger â€” `SC/src/strategy_core/candles/streaming.py:107-108`, `if not timeframes or any(size <= 0 for size in timeframes): raise ValueError("tick timeframes must be positive")`; the only close triggers are `trade_count == timeframe` (`streaming.py:180`) and trading-day rollover (`streaming.py:143-144`).

---

### A3 â€” `bar_id`

**Construction â€” the single canonical formatter.**

SC/src/strategy_core/candles/_ids.py:19-27
```python
def make_bar_id(timeframe_ticks: int, trading_day: date, bar_index: int) -> str:
    """Format the canonical bar id ``f"{tf}t:{trading_day}:{bar_index}"``.

    Exactly reproduces ``make_bar_id`` (``candles.py:195-196``): the timeframe in
    ticks followed by a ``t`` literal, the ISO trading day, and the per-day
    bar index, colon-separated. ``date.isoformat()`` is used for the day so the
    string is identical across the batch and streaming paths.
    """
    return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"
```

The exact expression (SC/src/strategy_core/candles/_ids.py:27):
```python
    return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"
```

**A byte-identical duplicate lives in TL** (DTO-side, kept for live importers):

TL/backend/src/trade_lab/domain/candles.py:40-41
```python
def make_bar_id(timeframe_ticks: int, trading_day: date, bar_index: int) -> str:
    return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"
```

**Call sites that stamp a `Bar.bar_id`:**
- `SC/src/strategy_core/candles/streaming.py:153` â€” `bar_id=make_bar_id(timeframe, trading_day, bar_index),` (streaming builder)
- `SC/src/strategy_core/candles/batch.py:164` â€” `bar_id=make_bar_id(timeframe, td, bi),` (batch builder)
- `QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:132` â€” `bar_id=make_bar_id(0, trading_day, i),` (research frame â†’ engine `Bar`; note `timeframe_ticks=0` at :129, so these ids look like `0t:YYYY-MM-DD:i`)
- `SC/validation/phase4b_validate.py:147`, `:202`; `SC/validation/decision_diff_harness.py:210` (validation harnesses)
- Copy-through (not re-derivation): `SC/src/strategy_core/candles/streaming.py:59` (`_MutableCandle.freeze`), `TL/backend/src/trade_lab/services/strategy_core_service.py:315`, `TL/backend/src/trade_lab/api/dto.py:307`.

**Parsers of `bar_id` â€” NONE.**

Grep for `bar_id.split(`, `bar_id.rsplit(`, `bar_id.partition(`, `bar_id[`, and the JS equivalents (`barId.split/slice/substring/match/startsWith/indexOf`, `barId[`) across all three trees returned `"No matches found"` in every repo. No `re.match`/`re.search` is applied to a bar id anywhere.

The id is consumed only as an opaque scalar:
- `TL/frontend/src/domain/normalize.ts:66` â€” `barId: typeof dto.bar_id === 'string' ? dto.bar_id : null,` (type check only, no parse).
- `TL/frontend/src/chart/viewModels.ts:41` â€” `export const barKey = (bar: MarketBar): string => bar.barId ?? `${bar.timeframe}:${bar.tradingDay}:${bar.openTimeUtc}`;` (used whole as a map key; the fallback *builds* a substitute string, it does not decompose the id).
- `QL/scratch_cache_shape_timing.py:345` / `:357` â€” `sum(len(b.bar_id) for b in day_bars)` (byte-length measurement only).
- `SC/validation/test_duckdb_streaming_parity.py:116` â€” `assert b.bar_id == make_bar_id(b.timeframe_ticks, b.trading_day, b.bar_index)` â€” re-*derives* and compares, does not parse.

The nearest thing to a parser in the codebase parses a **`BarSpec` label**, not a bar id: `SC/src/strategy_core/runtime/context.py:49-52`, `_label_to_timeframe("147t") -> 147` via `re.match(r"\d+", label)`. It is never fed a `bar_id`.

**Component extraction summary: no site anywhere extracts any component (timeframe, trading day, or index) back out of a `bar_id`.**

---

### A4 â€” frozen digest / golden fixtures

#### The only frozen digest fixtures in any of the three repos

Glob `**/_fixtures/**` over SC returns exactly two files; grep for `golden|frozen|FROZEN|snapshot_sha` over SC finds no other fixture artefacts (remaining hits are docs, `*_DIFF.txt` capture files, and `@dataclass(frozen=True)` decorators).

| # | fixture file | object family digested |
|---|---|---|
| 1 | `SC/validation/_fixtures/b3_regression/golive.json` | per-day **`RuntimeUpdate` stream** (which contains bars, levels, zones, touches, feed status, quote) + final **`RuntimeSnapshot`** |
| 2 | `SC/validation/_fixtures/b3_regression/multiday.json` | same, over a 9-day reset-bracketed chain |

SC/validation/_fixtures/b3_regression/golive.json:1-19
```json
{
  "days": {
    "2025-07-07": {
      "day": "2025-07-07",
      "n_trades": 306103,
      "seq_sha256": "b0de7a306097c81c20567e65ce973c5c5d636c6a4716e840857a0f7f465728e2",
      "snapshot_sha256": "e067f4aeeeff74399b3d081f27b94216ffac16aa51550762863b995bd1761488",
      "touches": 5
    },
    "2025-07-15": {
      "day": "2025-07-15",
      "n_trades": 339997,
      "seq_sha256": "03e4be3b995172ef222314a09009e76999869ea7df21719e998b8f376ba01d85",
      "snapshot_sha256": "f36a836ee61a7fca5548b973e8e144764a8fe7c2a1bd01a7bade2ee1b552deca",
      "touches": 5
    }
  },
  "seeding": "golive_synthetic_inside_range_pdh_pdl"
}
```
`SC/validation/_fixtures/b3_regression/multiday.json:1-15` has the same per-day key shape plus a `"skip"` flag.

Consumers: `SC/validation/test_b3_golive_plugin_regression.py:38` and `SC/validation/test_b3_multiday_reset_plugin_regression.py:30`, both asserting on `_DIGEST_KEYS = ("n_trades", "touches", "seq_sha256", "snapshot_sha256")`.

#### The exact serialization call that produces the digested bytes

SC/validation/_b3_regression_util.py:43-44
```python
def _canon(d: dict) -> bytes:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode()
```

SC/validation/_b3_regression_util.py:62-68
```python
    def add(self, update) -> None:
        self._h.update(_canon(update.to_dict()))
        self.n += 1
        self.touches += len(update.touches)

    def hexdigest(self) -> str:
        return self._h.hexdigest()
```

SC/validation/_b3_regression_util.py:71-72
```python
def snapshot_sha256(snapshot) -> str:
    return hashlib.sha256(_canon(snapshot.to_dict())).hexdigest()
```

So the bytes are `json.dumps(<RuntimeUpdate|RuntimeSnapshot>.to_dict(), sort_keys=True, separators=(",",":"), default=str).encode()`.

#### Tracing from the `Bar` object to those bytes

`RuntimeUpdate.to_dict()` carries bars through an **explicit, hand-written field map** â€” not `asdict`, not `__dict__`, not `astuple`, not `model_dump`:

SC/src/strategy_core/runtime/state.py:36-53
```python
def _bar(bar: Bar) -> dict[str, Any]:
    return {
        "timeframe_ticks": bar.timeframe_ticks,
        "trading_day": bar.trading_day.isoformat(),
        "bar_index": bar.bar_index,
        "bar_id": bar.bar_id,
        "open_ts_utc": bar.open_ts_utc.isoformat(),
        "close_ts_utc": bar.close_ts_utc.isoformat(),
        "open_ticks": bar.open_ticks,
        "high_ticks": bar.high_ticks,
        "low_ticks": bar.low_ticks,
        "close_ticks": bar.close_ticks,
        "volume": bar.volume,
        "trade_count": bar.trade_count,
        "is_complete": bar.is_complete,
        "is_partial": bar.is_partial,
        "close_reason": None if bar.close_reason is None else bar.close_reason.value,
    }
```

SC/src/strategy_core/runtime/state.py:137-147
```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_status": None if self.feed_status is None else self.feed_status.to_dict(),
            "warnings": [_warning(item) for item in self.warnings],
            "current_bars": [_bar(item) for item in self.current_bars],
            "closed_bars": [_bar(item) for item in self.closed_bars],
            "levels": [_level(item) for item in self.levels],
            "zones": [_zone(item) for item in self.zones],
            "touches": [_touch(item) for item in self.touches],
            "last_quote": _quote(self.last_quote),
        }
```

SC/src/strategy_core/runtime/state.py:164-177 (`RuntimeSnapshot.to_dict`) likewise routes bars through the same `_bar` helper:
```python
    def to_dict(self) -> dict[str, Any]:
        return {
            "current_bars": [_bar(item) for item in self.current_bars],
            "recent_closed_bars": [_bar(item) for item in self.recent_closed_bars],
            ...
```

**No reflective step exists anywhere on this path.** Grep for `asdict|astuple|model_dump|__dict__|dataclasses\.fields|\.fields\(` over all `*.py` in SC returns exactly four hits, none of them on the digest path:
- `SC/tests/test_candle_parity.py:21` `from dataclasses import astuple`
- `SC/tests/test_candle_parity.py:119-121` â€” `assert astuple(s) == astuple(b)` (streaming-vs-batch equality, **no frozen hash**; both sides are freshly computed)
- `SC/tests/test_touch_reversal_plugin.py:202` â€” `section.model_dump()` (a pydantic *section config*, not a `Bar`)

#### DIRECT ANSWER

**No â€” adding a new field to `Bar` would not change any frozen digest in any of the three repos, unless `_bar()` is also edited.**

The settling code path: the bytes hashed at `SC/validation/_b3_regression_util.py:63` and `:72` come from `_canon(... .to_dict())`, and every `Bar` reaching `to_dict()` is flattened by the literal 15-key dict at `SC/src/strategy_core/runtime/state.py:37-53`. That dict enumerates field names one by one; a 16th field on the dataclass produces no 16th key. The `default=str` fallback at `_b3_regression_util.py:44` never sees a `Bar` (bars are already dicts by then), so it cannot re-introduce the field either. Nothing on the path invokes `dataclasses.asdict`, `astuple`, `__dict__`, or `model_dump` on a `Bar` â€” there is no such call in SC at all.

Corollary: the two other hash surfaces in the trees are also insensitive to `Bar`'s shape.
- `QL/src/alpha_lab/agents/data_infra/ml/config.py:408-424` â€” `dataset_config_hash()` hashes `f"mode=â€¦" + json.dumps(self.extrema.model_dump(), sort_keys=True) + â€¦ + f"|platform_version={PLATFORM_VERSION}"`, i.e. **config models and module constants only** (`BAR_PRICE_SOURCE`, `LABEL_ENTRY_REFERENCE`, `PLATFORM_VERSION` â€” `SC/src/strategy_core/constants.py:36` for `BAR_PRICE_SOURCE = "trade_price"`). No `Bar` object participates. Final line, `config.py:424`: `return hashlib.sha256(payload.encode()).hexdigest()[:8]`.
- TL/QL model-bundle checksums (`TL/backend/src/trade_lab/services/model_registry.py:142` `CHECKSUM_FILE = "model.cbm.sha256"`, digest loop at `:555-562`) hash the raw bytes of a CatBoost model file â€” unrelated to `Bar`.

There is **no** golden/snapshot JSON of a serialized bar payload in TL or QL: `TL/backend/tests/test_api_contract.py:621-624` asserts individual scalar values inline (`snapshot["payload"]["current_bars"][0]["bar_index"] == 1`, `â€¦["bar_id"] == "2t:2026-01-05:1"`), not a whole-payload hash or key-set.

*Caveat (scope, not a hedge):* the frozen digests DO depend on `_bar()`'s output. If a new `Bar` field were added AND `_bar()` extended to emit it, both `seq_sha256` and `snapshot_sha256` in both fixtures would change. Whether those two regression tests currently reproduce their fixtures on this machine is `NOT OBTAINABLE READ-ONLY` â€” they require replaying real store days (2025-07-07 â€¦ 2025-07-22), which is test/pipeline execution and forbidden here; running `SC/validation/test_b3_golive_plugin_regression.py` and `â€¦multidayâ€¦` would settle it.

---

### A5 â€” every site where a `Bar` (or a collection of `Bar`s) is serialized

Sweep of SC, TL, QL for parquet writes, JSON dumps, DTO/pydantic models, websocket payloads, and cache writes (pickle/feather/npz).

**SC**

| path:line | mechanism | notes |
|---|---|---|
| `SC/src/strategy_core/runtime/state.py:36-53` (`_bar`) | hand-written `dict[str, Any]` projection | THE single SC barâ†’dict encoder; 15 explicit keys, `date`/`datetime` via `.isoformat()`, enum via `.value` |
| `SC/src/strategy_core/runtime/state.py:141-142` | `RuntimeUpdate.to_dict()` â†’ `[_bar(item) â€¦]` for `current_bars` / `closed_bars` | |
| `SC/src/strategy_core/runtime/state.py:166-167` | `RuntimeSnapshot.to_dict()` â†’ `[_bar(item) â€¦]` for `current_bars` / `recent_closed_bars` | |
| `SC/validation/_b3_regression_util.py:44` | `json.dumps(..., sort_keys=True, separators=(",",":"), default=str).encode()` | the bytes hashed for the frozen fixtures (A4) |
| `SC/validation/phase4b_validate.py:249` | manual tuple projection `return (b.timeframe_ticks, b.trading_day, b.bar_index, b.bar_id, b.open_ts_utc, b.close_ts_utc, â€¦)` | in-memory comparison key, not persisted |
| `SC/tests/test_candle_parity.py:121` | `dataclasses.astuple(s) == astuple(b)` | reflective, but comparison-only â€” nothing is written or hashed |

No parquet/pickle/feather/npz write of a `Bar` exists in SC: grep `to_parquet|pq\.write|pickle|feather|savez|joblib` over SC `*.py` yields only test-fixture writes of raw Databento rows (`SC/tests/test_prior_day.py:78`, `:155`; `SC/tests/test_databento_parquet_source.py:13`; `SC/tests/test_databento_parquet_day_mode.py:17`) â€” input parquet, not bars.

**TL**

| path:line | mechanism | notes |
|---|---|---|
| `TL/backend/src/trade_lab/services/strategy_core_service.py:310-327` (`_bar_to_trade_lab`) | field-by-field construction of the TL `Candle` dataclass from an SC `Bar` | adapter seam; `close_reason` remapped at `:326` via `_close_reason_to_trade_lab` (`:330-333`) |
| `TL/backend/src/trade_lab/api/dto.py:50-65` | **pydantic** `class BarDTO(ApiModel)` â€” 15 fields, `close_reason: str \| None` | `ApiModel` sets `model_config = ConfigDict(extra="forbid", populate_by_name=True)` (`dto.py:46-47`) |
| `TL/backend/src/trade_lab/api/dto.py:302-319` (`bar_to_dto`) | explicit `Candle` â†’ `BarDTO` field map | |
| `TL/backend/src/trade_lab/api/dto.py:534-535` (`bars_payload`) | `{"bars": [bar_to_dto(bar).model_dump(mode="json") for bar in bars]}` | pydantic `model_dump` â€” **on the DTO, not on `Bar`** |
| `TL/backend/src/trade_lab/api/dto.py:544-545` | `current_bars=[bar_to_dto(bar) â€¦]`, `recent_closed_bars=[bar_to_dto(bar) â€¦]` into the snapshot DTO | |
| `TL/backend/src/trade_lab/api/dto.py:601`, `:610` | `payload.model_dump(mode="json", by_alias=True)` / `envelope.model_dump(mode="json", by_alias=True)` | envelope wrapping |
| `TL/backend/src/trade_lab/services/broadcaster.py:127` | **websocket** â€” `self.envelope_bytes("market.bar.updated", bars_payload(update.current_bars))` | |
| `TL/backend/src/trade_lab/services/broadcaster.py:131` | **websocket** â€” `self.envelope_bytes("market.bar.closed", bars_payload(update.closed_bars))` | |
| `TL/backend/src/trade_lab/api/serialization.py:29-32` (`dumps_bytes`) | `orjson.dumps(payload, default=_default, option=orjson.OPT_SORT_KEYS)`, falling back to `json.dumps(payload, default=_default, sort_keys=True, separators=(",",":")).encode()` | the wire bytes for every envelope, bars included |

Frontend side is **de**serialization only: `TL/frontend/src/domain/normalize.ts:66` maps `bar_id`â†’`barId`; `TL/frontend/src/domain/models.ts:23` declares `barId: string | null`. `JSON.stringify` in `TL/frontend/src` is used for request bodies and leak-assertions (`api/client.ts:110`, `:136`), never to emit a bar.

TL persists no bars: grep `to_parquet|pq\.write|pickle|feather|savez|joblib` over `TL/backend` `*.py` hits only test fixtures writing raw Databento rows (`test_historical_parquet_adapter.py:25`, `test_performance.py:386`, `test_replay_seed.py:76`, `test_replay_catalog.py:21`) and the SC reader import (`adapters/historical_parquet.py:20`). The journal writes outcome/execution payloads, not bars â€” `TL/backend/src/trade_lab/services/journal.py:98` `line = json.dumps(payload, default=_json_default, separators=(",",":"))`, whose only bar-adjacent key is a scalar count at `:71` (`"bars_to_resolution": outcome.bars_to_resolution`).

**QL**

| path:line | mechanism | notes |
|---|---|---|
| `QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:127-145` (`bars_et_to_engine`) | constructs SC `Bar`s from a pandas frame (the reverse direction â€” deserialization into `Bar`) | `bar_id=make_bar_id(0, trading_day, i)` at `:132` |
| `QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:656`, `:672`, `:748` | `day_bars: list[Bar]` accumulated in memory and passed to `resolve_honest_outcome` | **never serialized**; the emitted rows (`rows: list[dict]` at `:744`) are touch/feature/label records |
| `QL/src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:65-78` (`_write_day_cache`) | **parquet cache write** â€” `pa.Table.from_pandas(frame, preserve_index=False)` â€¦ `pq.write_table(table, cache_path)` (`:73`, `:78`), plus custom schema metadata `_SEED_META_KEY = b"ml_utility_prev_full_hl"` (`:54`, stamped at `:75`) | the frame is the touch-row utility dataset, NOT bars â€” no `Bar` field reaches this file |
| `QL/scratch_cache_shape_timing.py:345`, `:357` | `sum(len(b.bar_id) for b in day_bars)` written into a scratch JSON | measurement of bar-id byte length only; scratch/untracked |

QL's `OHLCVBar` (`QL/src/alpha_lab/dashboard/db/models.py:69`, constructed at `dashboard/pipeline/tick_bar_builder.py:86`, `:116`; `price_buffer.py:206`, `:242`, `:259`, `:303`; `databento_client.py:193`; `dashboard/api/server.py:179`) is a **different, unrelated SQLAlchemy type** â€” the dashboard's own Decimal-priced bar. It is not `strategy_core.types.Bar` and shares no fields with it beyond OHLCV names.

**Net:** the `Bar` object itself is serialized in exactly three shapes â€” (1) the SC `_bar()` dict â†’ JSON bytes for digests, (2) the TL `Candle` â†’ `BarDTO` â†’ `model_dump(mode="json")` â†’ orjson websocket/HTTP bytes, and (3) nothing else. `Bar` is never written to parquet, pickle, feather, npz, or any on-disk cache in any of the three repos.

---

## PART B â€” consumers of `timeframe_ticks` and of bar-type STRINGS

Scope: SC = `C:\Users\gonza\Documents\Strategy-core`, TL = `C:\Users\gonza\Documents\Trade-Lab`, QL = `C:\Users\gonza\Documents\Claude-Quant-Lab`. Verify clones excluded. Read-only: static reads + ripgrep only; no file in any repo was modified, staged, or committed; no build/test/replay was executed.

Convention: `.md` / `.txt` recon docs and `*_DIFF.txt` patch dumps in the repo roots were excluded from the sweep tables (they are prose, not executed code). `.py`/`.ts`/`.tsx`/`.json` were all swept.

---

### B1

#### B1.1 â€” `.timeframe_ticks` ATTRIBUTE / dict-key READ sites

**SC â€” 12 read sites**

| # | path:line | what the code DOES with the value |
|---|---|---|
| 1 | SC/src/strategy_core/candles/streaming.py:56 | reads `self.timeframe_ticks` off the mutable accumulator and **passes it onward** verbatim into the frozen `Bar(...)` |
| 2 | SC/src/strategy_core/runtime/context.py:80 | **integer comparison** `bar.timeframe_ticks == tf` where `tf` came from parsing a BarSpec label; filters closed bars |
| 3 | SC/src/strategy_core/runtime/context.py:87 | **integer comparison** `tf is None or bar.timeframe_ticks == tf`; selects the forming bar |
| 4 | SC/src/strategy_core/runtime/state.py:38 | **dict-key write for wire serialization** â€” `{"timeframe_ticks": bar.timeframe_ticks, ...}` (the `_bar` JSON projection) |
| 5 | SC/src/strategy_core/runtime/state.py:354 | **integer comparison** `bar.timeframe_ticks != self.decision_timeframe: continue` â€” the decision-bar gate |
| 6 | SC/src/strategy_core/decisions/streaming.py:307 | **integer comparison** `bar.timeframe_ticks != self._forward_timeframe_ticks: return ()` â€” the streaming honest resolver's forward-bar gate |
| 7 | SC/tests/test_candle_parity.py:107 | used as the first element of a **sort/identity key tuple** `(bar.timeframe_ticks, bar.trading_day.isoformat(), bar.bar_index)` |
| 8 | SC/tests/test_candle_parity.py:168 | **integer comparison used as a filter for a trade-count sum**: `sum(b.trade_count for b in streaming if b.timeframe_ticks == TIMEFRAMES[0])` |
| 9 | SC/validation/phase4b_validate.py:249 | first element of a **field-identity tuple** used for byte-comparison of two Bar lists |
| 10 | SC/validation/test_d1_streaming_vs_batch_parity.py:118 | **integer comparison** `if bar.timeframe_ticks == FORWARD_TF` |
| 11 | SC/validation/test_duckdb_streaming_parity.py:116 | **passed onward** into `make_bar_id(b.timeframe_ticks, b.trading_day, b.bar_index)` and string-compared to `b.bar_id` |
| 12 | SC/src/strategy_core/candles/_ids.py:27 | **string formatting** â€” the parameter named `timeframe_ticks` is interpolated with a hard `t` suffix (see quote in B3) |

**TL â€” 11 read sites** (9 backend/frontend production, 2 tests)

| # | path:line | what the code DOES with the value |
|---|---|---|
| 1 | TL/backend/src/trade_lab/services/runtime.py:849 | **used as a dict key** â€” `kept_by_timeframe.get(bar.timeframe_ticks, 0)` (per-timeframe seed-bar retention) |
| 2 | TL/backend/src/trade_lab/services/runtime.py:852 | **used as a dict key** â€” `kept_by_timeframe[bar.timeframe_ticks] = kept + 1` |
| 3 | TL/backend/src/trade_lab/services/strategy_core_service.py:194 | **membership test** `bar.timeframe_ticks in self._display_timeframes` (current bars) |
| 4 | TL/â€¦/strategy_core_service.py:199 | **membership test** (closed bars) |
| 5 | TL/â€¦/strategy_core_service.py:222 | **membership test** (snapshot current bars) |
| 6 | TL/â€¦/strategy_core_service.py:227 | **membership test** (snapshot recent closed bars) |
| 7 | TL/â€¦/strategy_core_service.py:312 | **passed onward** â€” `timeframe_ticks=bar.timeframe_ticks` into the TL `Candle` DTO |
| 8 | TL/backend/src/trade_lab/api/dto.py:304 | **passed onward** â€” `timeframe_ticks=bar.timeframe_ticks` into `BarDTO` |
| 9 | TL/frontend/src/domain/normalize.ts:63 | **dict-key read + rename** â€” `timeframe: dto.timeframe_ticks` maps the wire field onto `MarketBar.timeframe` |
| 10 | TL/backend/tests/test_runtime_replay_broadcaster.py:538 | **integer comparison** filter |
| 11 | TL/backend/tests/test_strategy_core_service.py:53 | **integer comparison** assertion `first.current_bars[0].timeframe_ticks == 2` |

**QL â€” 2 read sites**

| # | path:line | what the code DOES with the value |
|---|---|---|
| 1 | QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:671 | **integer comparison** `if bar.timeframe_ticks == tick_count:` â€” selects the day's decision-grid bars |
| 2 | QL/scratch_cache_shape_timing.py:336 | same comparison, in a repo-root scratch script |

#### B1.2 â€” `timeframe_ticks=` KEYWORD / declaration sites (not reads)

| path:line | form |
|---|---|
| SC/src/strategy_core/types.py:98 | field declaration `timeframe_ticks: int` on `Bar` |
| SC/src/strategy_core/candles/streaming.py:39 | field declaration on `_MutableCandle` |
| SC/src/strategy_core/candles/streaming.py:56 | kwarg into `Bar(...)` |
| SC/src/strategy_core/candles/streaming.py:150 | kwarg `timeframe_ticks=timeframe` into `_MutableCandle(...)` |
| SC/src/strategy_core/candles/batch.py:161 | kwarg `timeframe_ticks=timeframe` into `Bar(...)` |
| SC/src/strategy_core/candles/_ids.py:19 | parameter name on `make_bar_id` |
| SC/validation/phase4b_validate.py:146, :202 Â· SC/validation/decision_diff_harness.py:207 | kwarg into `Bar(...)` |
| SC/tests/test_outcomes.py:53 Â· test_honest_entry.py:59 | `timeframe_ticks=0` (decision layer ignores it) |
| SC/tests/test_touch.py:39 | `timeframe_ticks=100` |
| SC/tests/test_streaming_resolver.py:57 | `timeframe_ticks=147` |
| SC/tests/test_touch_reversal_plugin.py:115-128 | helper param, also **assigned to `trade_count=timeframe_ticks`** at :128 |
| TL/backend/src/trade_lab/domain/candles.py:23 | field declaration on `Candle` |
| TL/backend/src/trade_lab/domain/candles.py:40 | parameter name on TL's own `make_bar_id` |
| TL/backend/src/trade_lab/api/dto.py:51 | pydantic field `timeframe_ticks: int` on `BarDTO` (no constraint) |
| TL/frontend/src/realtime/types.ts:20 | `timeframe_ticks: number;` on the wire BarDTO type |
| TL/backend/tests/test_execution_tracker.py:68, test_execution_surface.py:62, test_live_databento.py:1059 | kwargs |
| TL/frontend/src/domain/normalize.test.ts:257 Â· realtime/client.test.ts:63 Â· components/ChartWorkspace.test.tsx:41 | test fixtures |
| QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:129 | `timeframe_ticks=0,  # unused by the decision layer` |

Related-but-distinct names (`decision_timeframe_ticks`, `forward_timeframe_ticks`) are covered in B4/B6.

#### B1.3 â€” FLAGGED sites (observed properties, not recommendations)

These are stated as facts about what the code does today. Each would be factually wrong if the integer denoted **minutes** rather than **ticks**.

**(a) Sites that numerically equate the value to a COUNT OF TRADES.**

SC/src/strategy_core/candles/streaming.py:180 â€” this site compares an accumulated trade counter for equality against the timeframe number, and closes the bar on equality:
```python
            if candle.trade_count == timeframe:
                completed.append(candle.freeze(complete=True, reason=CloseReason.COMPLETE))
```
SC/src/strategy_core/candles/streaming.py:164 â€” this site treats the value `1` as "one trade print", emitting the bar COMPLETE on its seeding trade:
```python
                if timeframe == 1:
                    completed.append(new_candle.freeze(complete=True, reason=CloseReason.COMPLETE))
```
SC/src/strategy_core/candles/batch.py:124 â€” this site **integer-divides a per-trading-day running trade counter by the value** to assign bar indices:
```python
        work["bar_index"] = (day_groups.cumcount() // timeframe).to_numpy()
```
SC/src/strategy_core/candles/batch.py:175-177 â€” this site derives completeness by comparing the aggregated `trade_count` against the value:
```python
                is_complete=tc == timeframe,
                is_partial=tc != timeframe,
                close_reason=CloseReason.COMPLETE if tc == timeframe else CloseReason.END_OF_DAY,
```
SC/tests/test_candle_parity.py:168 â€” this site **sums trade counts over bars filtered by the value** and asserts the total against a trade count.

SC/tests/test_touch_reversal_plugin.py:128 â€” this site **assigns the timeframe value directly into `trade_count`** when constructing a fixture Bar.

**(b) Sites that append a literal `t` to the number when formatting.**

SC/src/strategy_core/candles/_ids.py:27, SC/src/strategy_core/decisions/streaming.py:202, SC/src/strategy_core/strategies/touch_reversal/plugin.py:89, SC/src/strategy_core/strategies/touch_reversal/section.py:187, TL/backend/src/trade_lab/domain/candles.py:41, TL/backend/src/trade_lab/services/model_registry.py:122, TL/frontend/src/components/ChartWorkspace.tsx:44 and :55, TL/frontend/src/components/TraderStrip.tsx:105, QL/src/alpha_lab/dashboard/pipeline/tick_bar_builder.py:61 and :65 â€” all quoted verbatim in B3. Each of these produces a string whose suffix asserts "ticks" about the number it wraps.

**(c) Sites that parse a leading integer out of a label and compare it to `timeframe_ticks`.**

SC/src/strategy_core/runtime/context.py:49-52 + :80/:87 â€” this site parses the LEADING digits of a BarSpec label with `re.match(r"\d+", label)` and then integer-compares that to `bar.timeframe_ticks`:
```python
def _label_to_timeframe(label: str) -> int | None:
    """Parse the leading integer from a BarSpec label (e.g. ``"147t"`` -> 147)."""
    match = re.match(r"\d+", label)
    return int(match.group(0)) if match else None
```
Observed property: the regex has no suffix anchor, so a label `"1m"` yields `1`, which this code then integer-compares against `bar.timeframe_ticks`, matching a 1-tick bar.

**(d) Sites that constrain the value to a fixed tick-count enum.**

TL/frontend/src/domain/models.ts:1 â€” `export type Timeframe = 147 | 987 | 2000;` â€” a closed literal union of three tick counts.
TL/frontend/src/chart/viewModels.ts:33-35 â€” `const SUPPORTED_TIMEFRAMES: Timeframe[] = [147, 987, 2000];` with `isSupportedTimeframe` gating; TL/frontend/src/realtime/client.ts:374-375, :405 drop any bar whose timeframe is not in that list.
TL/backend/src/trade_lab/config.py:64 â€” `tick_timeframes: tuple[int, ...] = (147, 987, 2000)`.

**(e) Sites that treat "smallest number" as "finest/decision bar".**

SC/src/strategy_core/runtime/state.py:222 â€” `self.decision_timeframe = decision_timeframe or min(timeframes)`.
TL/backend/src/trade_lab/services/strategy_core_service.py:109 â€” `decision_timeframe=min(self._display_timeframes)`.
TL/backend/src/trade_lab/api/app.py:278 â€” `decision_timeframe_ticks=min(settings.tick_timeframes)`.
TL/backend/scripts/w3b/headless_replay.py:144 â€” `decision_timeframe_ticks=min(settings.tick_timeframes)`.
TL/frontend/src/components/ChartWorkspace.tsx:30 â€” comment "the decision timeframe is the smallest served timeframe".
Observed property: for tick bars a smaller number is a shorter bar; the same `min()` applied to a minutes-valued set also picks the shortest bar, so this family of sites is ordering-consistent under either unit. Recorded here for completeness, not as a defect.

**(f) The only validation SC applies to a timeframe value.**

SC/src/strategy_core/candles/streaming.py:106-110:
```python
        # candles.py:111-112 -- positive timeframes only.
        if not timeframes or any(size <= 0 for size in timeframes):
            raise ValueError("tick timeframes must be positive")
        # candles.py:113 -- dedup + ascending order so iteration is deterministic.
        self.timeframes = tuple(sorted(set(timeframes)))
```
Observed property: positivity + dedup + sort only. No upper bound, no tick-vs-minute discriminator.

---

### B2 â€” `parse_bar_type` and every other bar-type string parser

#### B2.1 â€” TL: the canonical `parse_bar_type` â€” CONFIRMED, pattern is `^(\d+)t$`

**Confirmed.** TL pins exactly `r"^(\d+)t$"`. Verbatim:

TL/backend/src/trade_lab/services/inference/resolution_adapter.py:23
```python
_BAR_TYPE_RE = re.compile(r"^(\d+)t$")
```

TL/backend/src/trade_lab/services/inference/resolution_adapter.py:35-41 (FULL BODY)
```python
def parse_bar_type(bar_type: str) -> int:
    """Map a contract ``bar_type`` like ``147t`` to its tick count ``147``."""

    match = _BAR_TYPE_RE.match(bar_type.strip().lower())
    if match is None:
        raise ValueError(f"unsupported forward_bar_type {bar_type!r}; expected '<n>t'")
    return int(match.group(1))
```

Observed acceptance set: after `.strip().lower()`, ONLY a string of one-or-more ASCII digits followed by a single literal `t`, fully anchored at both ends. `"1m"`, `"1H"`, `"147"`, `"147 t"`, `"147ticks"` all raise `ValueError`.

Module docstring context, TL/â€¦/resolution_adapter.py:8-10:
```
``parse_bar_type`` relocated from the retired ``outcome_tracker`` module
(regex + ValueError semantics intact); the resolver build consumes it to map a
contract ``forward_bar_type`` onto the runtime's tick timeframes.
```

**Every caller of `parse_bar_type`:**

| path:line | call |
|---|---|
| TL/backend/src/trade_lab/services/model_registry.py:41 | `from trade_lab.services.inference.resolution_adapter import parse_bar_type` (import) |
| TL/backend/src/trade_lab/services/model_registry.py:116 | `bar_type_ticks = parse_bar_type(section.touch_rule.bar_type)` â€” inside `serving_compatibility_error`, wrapped in `try/except ValueError` which returns the message as the activation-refusal string |
| TL/backend/src/trade_lab/services/runtime.py:41 | import |
| TL/backend/src/trade_lab/services/runtime.py:457 | `forward_timeframe_ticks=parse_bar_type(policy.forward_bar_type)` â€” feeds `StreamingHonestResolver` |

Call site 1 verbatim, TL/backend/src/trade_lab/services/model_registry.py:115-123:
```python
    try:
        bar_type_ticks = parse_bar_type(section.touch_rule.bar_type)
    except ValueError as exc:
        return str(exc)
    if bar_type_ticks != capabilities.decision_timeframe_ticks:
        return (
            f"contract touch_rule.bar_type {section.touch_rule.bar_type!r} does not match "
            f"the runtime decision timeframe {capabilities.decision_timeframe_ticks}t"
        )
```

Call site 2 verbatim, TL/backend/src/trade_lab/services/runtime.py:456-465:
```python
        return StreamingHonestResolver(
            forward_timeframe_ticks=parse_bar_type(policy.forward_bar_type),
            tick_size=contract.tick_size,
            tp_points=policy.tp_points,
            sl_points=policy.sl_points,
            trap_mfe_min=policy.trap_mfe_min,
            decision_offset_minutes=policy.decision_offset_minutes,
            trade_price_at=lambda ts_utc: self.strategy_core_service.trade_price_at(ts_utc),
            available_timeframes=self._tick_timeframes,
        )
```

#### B2.2 â€” SC: NO `parse_bar_type` function exists

A tree grep for `parse_bar_type|def parse_bar` across SC/src returned only two docstring mentions, both in SC/src/strategy_core/strategies/touch_reversal/section.py:171-176, which reference TL's function by name. SC has no bar-type string parser of its own â€” it treats `touch_rule.bar_type` / `label_policy.forward_bar_type` as opaque length-bounded strings (B4).

SC's ONLY stringâ†’timeframe decoder is the label parser already quoted in B1.3(c), SC/src/strategy_core/runtime/context.py:49-52 â€” regex `re.match(r"\d+", label)`, leading-digits only, no suffix check, returns `int | None` (never raises).

#### B2.3 â€” QL: TWO ad-hoc suffix parsers (not regex, not shared)

QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:631-636 (FULL BODY of the parse):
```python
    bar_type = str(config.bar_type)
    if not bar_type.endswith("t"):
        raise ValueError(
            f"stream labeling requires a tick-count bar_type, got {bar_type!r}"
        )
    tick_count = int(bar_type[:-1])
```
Acceptance: any string ending in `t` whose prefix `int()` accepts. No anchor, no digit class â€” `"-5t"` and `"  12t"` parse; `"1m"` raises `ValueError`; a non-numeric prefix raises `ValueError` from `int()` rather than the explicit message. Result is fed to `StrategyRuntime(timeframes=(tick_count,), ...)` at :646 and integer-compared at :671.

QL/src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:311-350 (the three-branch dispatcher â€” the ONLY place in any of the three repos that recognizes a TIME bar-type string):
```python
        bar_type = util_cfg.bar_type
        if bar_type == "1m":
            # Check for cached session bars first
            cached = data_dir / symbol / date_str / "ohlcv_1m_session.parquet"
            if cached.exists():
                df = pd.read_parquet(cached)
                if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                return df
            df = store.build_bars_from_ticks(
                symbol,
                start_utc,
                end_utc,
                bar_size="1 minute",
            )
        elif bar_type.endswith("t"):
            tick_count = int(bar_type[:-1])
            ...
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=tick_count,
                price_source="trade",
            )
        else:
            logger.warning("Unknown bar_type: %s, falling back to 987t", bar_type)
            df = store.build_tick_bars(
                symbol,
                start_utc,
                end_utc,
                tick_count=987,
                price_source="trade",
```
Acceptance: exact string `"1m"` (special-cased, routed to a DuckDB `bar_size="1 minute"` build), else any `*t` suffix (tick path), else a **warn-and-fall-back to 987 ticks** â€” the only non-raising unknown-bar-type handler found in the three repos.

#### B2.4 â€” Frontend / TypeScript

A sweep of TL/frontend/src for `timeframe|bar_type|barType` across `*.ts`/`*.tsx` found **no bar-type string parser**. The frontend carries the timeframe as a NUMBER end-to-end (`timeframe_ticks: number` at TL/frontend/src/realtime/types.ts:20 â†’ `timeframe: dto.timeframe_ticks` at TL/frontend/src/domain/normalize.ts:63) and only ever BUILDS `${n}t` strings for display (B3). There is no `.ts`/`.tsx` code that reads a `"147t"`-shaped string back into a number.

QL/src/alpha_lab/agents/data_infra/ml/labeling.py:75 and :149 build keys `f"label_{t}t"` where `t` iterates `config.rebound_thresholds` â€” this is a rebound-threshold-in-ticks label column name, NOT a bar-type string, and nothing parses it back.

#### B2.5 â€” No JSON Schema files

```
$ find {Strategy-core,Trade-Lab,Claude-Quant-Lab} -name "*.schema.json" -not -path "*/node_modules/*" -not -path "*/.git/*"
=== Strategy-core ===
=== Trade-Lab ===
=== Claude-Quant-Lab ===
```
No `*.schema.json` exists in any of the three trees, so there is no json-schema `pattern` constraint on any bar-type field anywhere.

---

### B3 â€” Sites that BUILD a bar-type string

**Python**

| path:line | exact format expression | note |
|---|---|---|
| SC/src/strategy_core/candles/_ids.py:27 | `return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"` | the canonical `bar_id`; `t` is a hard literal |
| SC/src/strategy_core/strategies/touch_reversal/plugin.py:89 | `_DECISION_BAR_LABEL = f"{DEFAULT_TICK_COUNT}t"` | module-level constant, becomes the `BarSpec.label` |
| SC/src/strategy_core/strategies/touch_reversal/section.py:187 | `bar_type=f"{DEFAULT_TICK_COUNT}t",` | the contract-default `touch_rule.bar_type` |
| SC/src/strategy_core/decisions/streaming.py:202 | `f"forward timeframe {forward_timeframe_ticks}t is not among the runtime's "` | error message only |
| TL/backend/src/trade_lab/domain/candles.py:41 | `return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"` | TL's own copy of the bar-id formatter |
| TL/backend/src/trade_lab/services/model_registry.py:122 | `f"the runtime decision timeframe {capabilities.decision_timeframe_ticks}t"` | activation-refusal message |
| TL/backend/scripts/w3b/parity.py:458-459 | `f"rep_px={t.representative_price} -> {expected_ticks}t "` / `f"!= serv {s.level_price_ticks}t",` | price-in-ticks diagnostic, NOT a bar type |
| QL/src/alpha_lab/dashboard/pipeline/tick_bar_builder.py:61 | `f"{tc}t": _Accumulator(tick_count=tc) for tc in tick_counts` | bar-type string used as a **dict key** for the accumulator map |
| QL/src/alpha_lab/dashboard/pipeline/tick_bar_builder.py:65 | `{f"{tc}t": [] for tc in tick_counts}` | same key space for the completed-bar store |
| QL/src/alpha_lab/agents/data_infra/ml/labeling.py:75, :149 | `f"label_{t}t"` / `labels[f"label_{threshold}t"]` | rebound-threshold column name, NOT a bar type |
| QL/src/alpha_lab/agents/data_infra/ml/dashboard_utility_builder.py:343 | `logger.warning("Unknown bar_type: %s, falling back to 987t", bar_type)` | literal `987t` in a log string |

Note: SC/src/strategy_core/strategies/touch_reversal/section.py:172 contains the same expression inside a docstring (`f"{DEFAULT_TICK_COUNT}t"` = `"147t"`), not executed.

**TypeScript / TSX**

| path:line | exact template expression |
|---|---|
| TL/frontend/src/components/TraderStrip.tsx:105 | `` value={`${selectedTimeframe}t`} `` |
| TL/frontend/src/components/ChartWorkspace.tsx:44 | `` {timeframe}t{timeframe === decisionTimeframe ? ' Â· decision' : ''} `` (JSX text, produces `147t Â· decision`) |
| TL/frontend/src/components/ChartWorkspace.tsx:55 | `` emptyTitle={feedReady ? `Awaiting ${selectedTimeframe}t tick bars` : 'Runtime snapshot idle'} `` |
| TL/frontend/src/realtime/client.test.ts:66 | `` bar_id: overrides.bar_id ?? `${timeframe}t:${tradingDay}:${barIndex}` `` (test fixture mirroring the backend `bar_id` format) |

Not a bar type but matched the sweep: TL/frontend/src/intel/viewModels.ts:48 â€” `` `${sign}${Math.abs(points).toFixed(2)} pts (${sign}${Math.abs(distanceTicks)}t)` `` â€” a price-distance-in-ticks display string.

Observed property across all of B3: every producer emits `<int>t` unconditionally. No producer in any repo emits a minute/hour-suffixed bar-type string; the only place a non-`t` bar-type string is even *recognized* is the QL `"1m"` literal at dashboard_utility_builder.py:312.

---

### B4 â€” Contract schema fields carrying bar-type strings

#### B4.1 â€” `touch_rule.bar_type` (SC, pydantic)

SC/src/strategy_core/contract/schema.py:140-148
```python
class TouchRule(_ContractModel):
    """First-touch detection rule. Ported from ``strategy_contract.py:73-79``."""

    type: str = Field(min_length=1, max_length=32)
    bar_type: str = Field(min_length=1, max_length=16)
    zone_proximity_pts: float = Field(ge=0.0)
    zone_representative_price: str = Field(min_length=1, max_length=64)
    scope: str = Field(min_length=1, max_length=64)
    direction_from_side: dict[str, str]
```
Declared type: `str`. Constraints: `min_length=1, max_length=16` ONLY. **No regex/pattern, no `Literal`, no `Enum`, no field/model validator.** The full validator inventory for the contract package is:
```
$ rg "field_validator|model_validator|@validator|pattern=|Literal\[" SC/src/strategy_core/contract
schema.py:42:from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
schema.py:185:    barrier_mode: Literal["fixed_points", "r_relative"] = "fixed_points"
schema.py:247:    @model_validator(mode="before")
schema.py:255:    @model_validator(mode="after")
```
Both `model_validator`s belong to `ClassMap` (SC/src/strategy_core/contract/schema.py:247-262: `_coerce_string_keys` and `_classes_are_contiguous_from_zero`) and neither touches a bar-type field. The only `Literal` in the whole contract package restricts `barrier_mode`, not a bar type. Verbatim, SC/src/strategy_core/contract/schema.py:247-262:
```python
    @model_validator(mode="before")
    @classmethod
    def _coerce_string_keys(cls, value: Any) -> Any:
        # strategy.json encodes class indices as JSON object keys (strings).
        if isinstance(value, dict) and "mapping" not in value:
            return {"mapping": value}
        return value

    @model_validator(mode="after")
    def _classes_are_contiguous_from_zero(self) -> ClassMap:
        keys = sorted(self.mapping)
        if not keys or keys != list(range(len(keys))):
            raise ValueError("class_map indices must be contiguous and start at 0")
        if len(set(self.mapping.values())) != len(self.mapping):
            raise ValueError("class_map labels must be unique")
        return self
```

Default value producer: SC/src/strategy_core/strategies/touch_reversal/section.py:185-194 sets `bar_type=f"{DEFAULT_TICK_COUNT}t"`. Its docstring at :171-176 records the intent:
```
    E3 ledger fix: ``touch_rule.bar_type`` defaults to the canonical production bar
    literal (``f"{DEFAULT_TICK_COUNT}t"`` = ``"147t"``, the same form Trade-Lab's
    ``parse_bar_type`` accepts), not the pre-E3 ``"tick"`` placeholder. The old
    ``label_policy.forward_bar_type="tick"`` landmine died with ``label_policy``'s
    move to the ENVELOPE (the section carries no label policy; the emitter sources
    ``forward_bar_type`` per-run).
```
Per-run override: QL/src/alpha_lab/agents/data_infra/ml/strategy_contract.py:92-96 â€” `base.touch_rule.model_copy(update={"bar_type": du.bar_type})`.

The only VALUE enforcement on `touch_rule.bar_type` anywhere is at TL activation (B2.1 call site 1): the string must satisfy `^(\d+)t$` AND its integer must equal `capabilities.decision_timeframe_ticks`.

#### B4.2 â€” `label_policy.forward_bar_type` (SC, pydantic)

SC/src/strategy_core/contract/schema.py:178-193
```python
    resolution: str = Field(min_length=1, max_length=32)
    #: Contract v3 (E3): how the tp/sl/trap thresholds are interpreted â€”
    #: "fixed_points" (absolute points off the entry, today's only implemented
    #: barrier, ``FixedPointsBarrier``) or "r_relative" (thresholds expressed in
    #: R-multiples; declared for forward compatibility, no producer emits it yet).
    #: Closes the Phase-D named debt that the Barrier abstraction had no contract
    #: field to bind to.
    barrier_mode: Literal["fixed_points", "r_relative"] = "fixed_points"
    entry_reference: str = Field(min_length=1, max_length=64)
    decision_offset_minutes: int = Field(gt=0, le=1440)
    tp_points: float = Field(gt=0.0)
    sl_points: float = Field(gt=0.0)
    trap_mfe_min: float = Field(ge=0.0)
    forward_bar_type: str = Field(min_length=1, max_length=16)
    forward_cutoff: str = Field(min_length=1, max_length=64)
    no_resolution_dropped: bool
```
Declared type: `str`. Constraints: `min_length=1, max_length=16` ONLY. **No pattern, no Literal/Enum, no validator.** Value enforcement is entirely at TL/backend/src/trade_lab/services/runtime.py:457 (`parse_bar_type`, i.e. `^(\d+)t$`), plus the resolver's own membership guard (B6).

Emitter: QL/src/alpha_lab/agents/data_infra/ml/strategy_contract.py:226 â€” `"forward_bar_type": du.bar_type,` (the SAME `du.bar_type` value that fills `touch_rule.bar_type`).

Source of `du.bar_type`, QL/src/alpha_lab/agents/data_infra/ml/config.py:333-336:
```python
    bar_type: str = Field(
        default="147t",
        description="Bar type for touch detection and MFE/MAE: '147t', '987t', '2000t', or '1m'",
    )
```
Declared type `str`, default `"147t"`, **no regex/Literal/validator** â€” the enumeration `'147t', '987t', '2000t', or '1m'` lives in the free-text `description` only and is not enforced. Observed property: this is the single field in the three repos whose declared vocabulary includes a TIME bar-type string (`"1m"`), and the only consumer that honors it is QL/â€¦/dashboard_utility_builder.py:312.

#### B4.3 â€” `BarSpec` (SC dataclass) and `required_bars()`

SC/src/strategy_core/strategies/protocols.py:61-81 (full declaration, both models):
```python
class BarKind(StrEnum):
    """How a bar closes. ``StrEnum`` so ``BarKind.TICK == "tick"`` (PLAN Â§2.1(3))."""

    TICK = "tick"  # close on trade_count == size      (CURRENT engine)
    TIME = "time"  # close on wall-clock interval edge  (NEW close trigger, Phase F)


@dataclass(frozen=True, slots=True)
class BarSpec:
    """A bar a strategy declares it needs (PLAN Â§2.1(3)).

    ``size`` is a tick count for ``TICK`` or interval *seconds* for ``TIME``;
    ``label`` is a stable id (e.g. ``"147t"``, ``"1m"``, ``"1H"``) the platform routes
    closed bars by. Archetype 1 declares a single ``TICK`` spec; archetype 2 declares a
    mix of ``TIME`` specs (1m..4H) â€” which the engine cannot yet build (decision 9.4,
    Phase F).
    """

    kind: BarKind
    size: int
    label: str
```
`BarSpec` is a plain frozen dataclass, NOT pydantic: `kind: BarKind` (StrEnum, two members), `size: int` (unconstrained), `label: str` (unconstrained, no pattern). `BarKind.TIME` is DECLARED but the docstring states the engine cannot build those bars. `BarSpec.size` carries different units per `kind` (ticks vs interval seconds) with no runtime discriminator beyond `kind`.

`required_bars()` protocol declaration, SC/src/strategy_core/strategies/protocols.py:276-284:
```python
    def required_bars() -> tuple[BarSpec, ...]:
```
(`@staticmethod`, followed by `decision_bar_label() -> str` at :283 â€” "The BarSpec label that drives single-bar decisions.")

Sole implementation, SC/src/strategy_core/strategies/touch_reversal/plugin.py:246-253:
```python
    @staticmethod
    def required_bars() -> tuple[BarSpec, ...]:
        # One TICK BarSpec at the decision timeframe (R3).
        return (BarSpec(kind=BarKind.TICK, size=_DECISION_TIMEFRAME, label=_DECISION_BAR_LABEL),)

    @staticmethod
    def decision_bar_label() -> str:
        return _DECISION_BAR_LABEL
```
with SC/src/strategy_core/strategies/touch_reversal/plugin.py:83-89:
```python
# R3: the decision timeframe is the engine's default smallest tick-count. The runtime
# resolves ``decision_timeframe = decision_timeframe or min(timeframes)``
# with default timeframes (147, 987, 2000), i.e. DEFAULT_TICK_COUNT. Held as a constant
# because required_bars()/decision_bar_label() are @staticmethod per Â§2.2; a configurable
# decision tf is a wiring concern deferred to Phase B.
_DECISION_TIMEFRAME = DEFAULT_TICK_COUNT
_DECISION_BAR_LABEL = f"{DEFAULT_TICK_COUNT}t"
```

Registry-time validation of `required_bars()`, SC/src/strategy_core/strategies/registry.py:63-72:
```python
    # Â§9.1: required_bars() must return a non-empty tuple of BarSpec.
    try:
        bars = plugin_cls.required_bars()
    except Exception as exc:
        ...
            f"{plugin_cls.__name__}.required_bars() raised at register time: {exc!r}"
    if not isinstance(bars, tuple) or not bars or not all(isinstance(b, BarSpec) for b in bars):
        ...
            f"{plugin_cls.__name__}.required_bars() must return a non-empty tuple of BarSpec"
```
Observed property: the registry checks TYPE and non-emptiness only. It does not inspect `kind`, `size`, or `label`, and does not check that the declared bars are buildable by the wired `CandleEngine`.

Callers of `required_bars()` (whole sweep, all three repos): SC/tests/test_plugin_cross_bar_suppression.py:51, SC/tests/test_touch_reversal_plugin.py:150 and :182, SC/src/strategy_core/strategies/registry.py:65. Greps for `required_bars|BarSpec|decision_bar_label` over TL (excluding `*.md`) and over QL (excluding `*.md`) both returned **"No matches found"** â€” neither TL nor QL reads the declared-bars surface at all.

#### B4.4 â€” TypeScript mirrors of these fields

None. A `*.ts`/`*.tsx` sweep for `bar_type|barType` across TL/frontend/src returned no hits â€” the contract's bar-type fields are never sent to or typed in the frontend. The frontend's only timeframe surface is numeric: `supported_tick_timeframes: number[]` (TL/frontend/src/api/types.ts:53), `timeframe_ticks: number` (TL/frontend/src/realtime/types.ts:20), and `export type Timeframe = 147 | 987 | 2000;` (TL/frontend/src/domain/models.ts:1) â€” a **TS literal union, i.e. the tightest bar-type-value restriction found anywhere in the three repos, and it lives in the frontend, not the contract**. `MarketBar.timeframe` widens it back out: `timeframe: Timeframe | number;` (TL/frontend/src/domain/models.ts:20).

TL's backend `BarDTO.timeframe_ticks` (TL/backend/src/trade_lab/api/dto.py:51) is a bare `int` with no constraint, under `model_config = ConfigDict(extra="forbid", populate_by_name=True)` (TL/backend/src/trade_lab/api/dto.py:46-47).

TL defines no contract schema of its own â€” it imports SC's: `from strategy_core import PLATFORM_VERSION, ContractError, StrategyContract, load_strategy_contract` (TL/backend/src/trade_lab/services/model_registry.py:30).

---

### B5 â€” Every `CandleEngine(...)` construction site

Greps for `CandleEngine\(` returned **no matches in TL** and **no matches in QL**. All construction sites are in SC.

| # | path:line | exact `timeframes` argument |
|---|---|---|
| 1 | SC/src/strategy_core/runtime/state.py:221 | `CandleEngine(timeframes, scheme=scheme)` â€” positional; `timeframes` is the `StrategyRuntime.__init__` parameter |
| 2 | SC/src/strategy_core/runtime/state.py:256 | `CandleEngine(self.candles.timeframes, scheme=self.scheme)` â€” positional, re-uses the previous engine's normalized tuple (the `reset()` path) |
| 3 | SC/tests/test_candle_parity.py:97 | `CandleEngine(timeframes, scheme=scheme)` â€” positional test parameter |
| 4 | SC/validation/test_production_pair_parity.py:108 | `CandleEngine(timeframes=(TICK_COUNT,), scheme=RESEARCH_SESSION_SCHEME)` |
| 5 | SC/validation/test_duckdb_streaming_parity.py:105 | `CandleEngine(timeframes=(TICK_COUNT,), scheme=RESEARCH_SESSION_SCHEME)` |
| 6 | SC/validation/phase4d_liveorder.py:86 | `CandleEngine(timeframes=(TICK_COUNT,), scheme=RESEARCH_SESSION_SCHEME)` |
| 7 | SC/validation/decision_diff_harness.py:236 | `CandleEngine(timeframes=(TICK_COUNT,), scheme=RESEARCH_SESSION_SCHEME)` |

Constructor signature and its default, SC/src/strategy_core/candles/streaming.py:100-113:
```python
    def __init__(
        self,
        timeframes: tuple[int, ...] = (147, 987, 2000),
        *,
        scheme: SessionScheme = RESEARCH_SESSION_SCHEME,
    ) -> None:
        # candles.py:111-112 -- positive timeframes only.
        if not timeframes or any(size <= 0 for size in timeframes):
            raise ValueError("tick timeframes must be positive")
        # candles.py:113 -- dedup + ascending order so iteration is deterministic.
        self.timeframes = tuple(sorted(set(timeframes)))
        self._scheme = scheme
        self._current: dict[int, _MutableCandle] = {}
        self._next_index: dict[tuple[int, date], int] = {}
```

The two indirect wiring paths that determine site #1's argument (TL and QL never touch `CandleEngine` directly â€” they pass `timeframes` to `StrategyRuntime`):

TL/backend/src/trade_lab/services/strategy_core_service.py:99-109
```python
        self._display_timeframes = tuple(sorted(set(tick_timeframes)))
        self._runtime = StrategyRuntime(
            requested_symbol=requested_symbol,
            timeframes=self._display_timeframes,
            ...
            decision_timeframe=min(self._display_timeframes),
```
fed from TL/backend/src/trade_lab/config.py:64 (`tick_timeframes: tuple[int, ...] = (147, 987, 2000)`) via TL/backend/src/trade_lab/api/app.py:251.

QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:646
```python
    runtime = StrategyRuntime(timeframes=(tick_count,), requested_symbol=symbol)
```
where `tick_count` came from the `bar_type[:-1]` parse at :636.

---

### B6 â€” `detect_touches` and the honest resolver: do they branch on `timeframe_ticks`?

#### B6.1 â€” `detect_touches` (batch AND streaming are the SAME function) â€” DOES NOT branch on `timeframe_ticks`

There is exactly ONE `detect_touches` implementation: SC/src/strategy_core/decisions/touch.py:55-117. The streaming path calls that same function with a one-element tuple. I read the complete function body (SC/src/strategy_core/decisions/touch.py:55-117) and its helper `is_touch` (SC/src/strategy_core/decisions/touch.py:42-52). Verbatim, SC/src/strategy_core/decisions/touch.py:88-117:
```python
    touches: list[Touch] = []

    for bar in bars:
        low_points = bar.low_ticks * tick_size
        high_points = bar.high_ticks * tick_size

        for zone in zones:
            if zone.touched:
                continue

            # v3 look-ahead guard: the level is not yet available at this bar's close
            # -- skip WITHOUT consuming first-touch so a later (post-availability) bar
            # can record the real return-to-level touch.
            if zone.available_from is not None and bar.close_ts_utc < zone.available_from:
                continue

            if is_touch(low_points, high_points, zone.representative_price):
                zone.touched = True
                direction = direction_from_side[zone.side]
                touches.append(
                    Touch(
                        bar_ts_utc=bar.close_ts_utc,
                        representative_price=zone.representative_price,
                        direction=direction,
                        level_type=zone.names[0],
                        trading_day=trading_day,
                    )
                )

    return touches
```
**Explicitly: `detect_touches` contains NO reference to `timeframe_ticks`.** The only `Bar` fields it reads are `low_ticks`, `high_ticks`, and `close_ts_utc`. Its three branches are: `zone.touched`, the `available_from` availability guard, and `is_touch`.

The GATING happens one layer up, in the callers, not inside `detect_touches`:

- Batch/day callers pass a whole day's bar list with no timeframe filter â€” SC/validation/decision_diff_harness.py:418 `touches = detect_touches(day_bars, zones_for_touch, tick_size=TICK_SIZE, trading_day=D)`; SC/validation/phase4b_validate.py:328, :338.
- Streaming caller passes ONE bar â€” SC/src/strategy_core/strategies/touch_reversal/plugin.py:285-287:
```python
        touches = detect_touches(
            (bar,), zones, tick_size=ctx.tick_size, trading_day=bar.trading_day
        )
```
and the plugin explicitly refuses to re-gate, SC/src/strategy_core/strategies/touch_reversal/plugin.py:266-271:
```python
        # The PLATFORM gates which bars reach here (D-B2i). The runtime's loop only calls
        # on_bar_closed for bars whose timeframe == the runtime's decision_timeframe, so
        # this method does NOT re-gate on its own declared decision bar
        # (_DECISION_TIMEFRAME): re-gating on a hardcoded timeframe would break
        # byte-identity whenever the runtime's decision_timeframe differs from it (e.g. a
        # tf=2 acceptance harness). It processes the decision bar it is handed: ...
```
- The ONE upstream `timeframe_ticks` branch feeding the streaming touch path lives in the runtime loop, SC/src/strategy_core/runtime/state.py:353-359:
```python
        for bar in candle_update.completed:
            if bar.timeframe_ticks != self.decision_timeframe:
                continue
            # The plugin owns the cross-bar first-touch dedup (S-B3a); its touches flow
            # back VERBATIM â€” no re-derivation, re-keying, or filtering here.
            step = self._plugin.on_bar_closed(bar, self._ctx)
            touches.extend(step.touches)
```

#### B6.2 â€” Honest resolver, BATCH implementation (`resolve_honest_outcome`) â€” DOES NOT branch on `timeframe_ticks`

SC/src/strategy_core/decisions/honest_entry.py:76-176. I read the complete function. It has exactly four branches â€” flatten (`:142`), cutoff (`:144`), no_fill (`:150`), no_forward (`:160`) â€” plus one comprehension filter. **No reference to `timeframe_ticks` anywhere in the file.** The forward-window selection is purely time-based, SC/src/strategy_core/decisions/honest_entry.py:153-159:
```python
    # (d) forward window: bars whose close is strictly AFTER the decision AND strictly
    # BEFORE the RTH cutoff (matches ``index > decision_ts_et & index < rth_cutoff``).
    forward = [
        bar
        for bar in day_bars
        if bar.close_ts_utc > decision_ts_utc and bar.close_ts_utc < rth_cutoff_et
    ]
```
Its docstring makes the agnosticism explicit, SC/src/strategy_core/decisions/honest_entry.py:102-107:
```
        day_bars: The FULL day's engine bars (the forward window is sliced from these
            by close-instant; the touch bar and pre-decision bars are excluded by the
            strict ``> decision_ts`` bound). Bar representation is agnostic â€” only
            ``close_ts_utc`` and the high/low ticks (read by ``resolve_outcome``) are
            used.
```
Observed property: `resolve_honest_outcome` accepts whatever `day_bars` its caller hands it. The caller decides the grid â€” e.g. QL/src/alpha_lab/agents/data_infra/ml/engine_decision.py:670-672 filters `if bar.timeframe_ticks == tick_count:` when accumulating `day_bars`.

#### B6.3 â€” Honest resolver, STREAMING implementation (`StreamingHonestResolver`) â€” YES, it branches on `timeframe_ticks`

SC/src/strategy_core/decisions/streaming.py:173-360. There is exactly ONE such branch in the class, and one construction-time membership check. Both quoted verbatim.

Branch 1 â€” the per-bar gate, SC/src/strategy_core/decisions/streaming.py:298-308:
```python
    def on_bar(self, bar: Bar) -> tuple[StreamResolution | StreamDrop, ...]:
        """Advance every open setup with one just-closed bar.

        Bars whose ``timeframe_ticks`` differ from the forward timeframe are
        ignored. A bar closing at/after a setup's cutoff finalizes that setup
        (no_forward / no_resolution) WITHOUT contributing its range â€” the batch
        forward window is strictly ``close < rth_cutoff`` (``honest_entry.py:154``),
        so a cutoff-straddling bar's extremes never count.
        """
        if bar.timeframe_ticks != self._forward_timeframe_ticks:
            return ()
```

Branch 2 â€” the construction-time membership check, SC/src/strategy_core/decisions/streaming.py:198-206:
```python
        if available_timeframes is not None and forward_timeframe_ticks not in tuple(
            available_timeframes
        ):
            raise ValueError(
                f"forward timeframe {forward_timeframe_ticks}t is not among the runtime's "
                f"configured timeframes {tuple(available_timeframes)}; no bar would ever "
                "advance the resolver (fail loud at activation, not silently never-resolve)"
            )
        self._forward_timeframe_ticks = int(forward_timeframe_ticks)
```
Rationale recorded in the class docstring, SC/src/strategy_core/decisions/streaming.py:176-181:
```
    One resolver instance serves one runtime/contract; ``reset()`` clears all open
    setups (runtime reset / model hot-swap). Construction fails loud when the
    forward timeframe is not among the runtime's configured timeframes â€” a
    mismatch would otherwise mean NO bar ever matches and every setup silently
    rides to its cutoff (the D-window recon's silent-never-resolve hole).
```
Accessor, SC/src/strategy_core/decisions/streaming.py:218-220:
```python
    @property
    def forward_timeframe_ticks(self) -> int:
        return self._forward_timeframe_ticks
```
The rest of `on_bar` (SC/src/strategy_core/decisions/streaming.py:310-360) branches only on `bar.close_ts_utc` vs `setup.rth_cutoff` / `setup.decision_ts_utc`, on `setup.direction`, and on `classify_mae_first` returning `None` â€” no further `timeframe_ticks` reads.

The value that reaches branch 2 is produced by TL's `parse_bar_type` at TL/backend/src/trade_lab/services/runtime.py:457, and `available_timeframes` is TL's `self._tick_timeframes` (TL/backend/src/trade_lab/services/runtime.py:464). Observed property: the constructor's membership check is `int not in tuple[int]` â€” a forward bar type whose integer is not in the runtime's configured tick-count tuple raises at activation; the `t` suffix in the error message at `:202` is a hard literal.

#### B6.4 â€” Summary

| implementation | path:line | branches on `timeframe_ticks`? |
|---|---|---|
| `detect_touches` (single impl, serves batch + streaming) | SC/src/strategy_core/decisions/touch.py:55-117 | **NO** â€” read in full; zero references |
| `is_touch` | SC/src/strategy_core/decisions/touch.py:42-52 | **NO** â€” three float args only |
| honest resolver, batch: `resolve_honest_outcome` | SC/src/strategy_core/decisions/honest_entry.py:76-176 | **NO** â€” read in full; zero references |
| honest resolver, streaming: `StreamingHonestResolver.on_bar` | SC/src/strategy_core/decisions/streaming.py:307 | **YES** â€” one equality gate |
| honest resolver, streaming: `StreamingHonestResolver.__init__` | SC/src/strategy_core/decisions/streaming.py:198-206 | **YES** â€” one membership check, raises |
| runtime touch loop (the gate feeding `detect_touches`) | SC/src/strategy_core/runtime/state.py:354 | **YES** â€” one inequality `continue` |
| `TouchReversalPlugin.on_bar_closed` | SC/src/strategy_core/strategies/touch_reversal/plugin.py:265-295 | **NO** â€” deliberately does not re-gate (comment at :266-271) |

---

## PART C â€” the tick-bar pattern that a time-bar path would mirror

Scope: read-only static analysis of SC = `C:\Users\gonza\Documents\Strategy-core`, TL = `C:\Users\gonza\Documents\Trade-Lab`, QL = `C:\Users\gonza\Documents\Claude-Quant-Lab` (all on `platform-refactor`). No file modified, no test executed, no NQ day data read. Facts only.

Files at the centre of this section:

- SC `tests/test_candle_parity.py` (180 lines, read in full)
- SC `src/strategy_core/candles/batch.py` (193 lines, read in full)
- SC `src/strategy_core/candles/streaming.py` (216 lines, read in full)
- SC `src/strategy_core/candles/_ids.py`, `src/strategy_core/candles/__init__.py`
- SC `src/strategy_core/decisions/sessions.py`, `src/strategy_core/constants.py`

---

### C1 â€” `tests/test_candle_parity.py`

Located at SC `tests/test_candle_parity.py` (the only file of that name in any of the three trees; grep for `test_candle_parity` also hits only doc/diff text elsewhere).

It contains three tests: `test_batch_streaming_parity_research_scheme` (SC/tests/test_candle_parity.py:124), `test_batch_streaming_parity_ct_scheme_closed_window` (SC/tests/test_candle_parity.py:148), `test_empty_frame_returns_no_bars` (SC/tests/test_candle_parity.py:175).

#### How it drives the STREAMING engine (calls, in order)

SC/tests/test_candle_parity.py:92-103

```python
def _streaming_bars(
    trades: list[Trade], timeframes: tuple[int, ...], scheme: SessionScheme
) -> list[Bar]:
    """All bars from the streaming engine: completed bars in stream order plus the
    final ``finalize_trading_day`` END_OF_DAY partials."""
    engine = CandleEngine(timeframes, scheme=scheme)
    bars: list[Bar] = []
    for trade in trades:
        update = engine.process_trade(trade)
        bars.extend(update.completed)
    bars.extend(engine.finalize_trading_day())
    return bars
```

Order: (1) `CandleEngine(timeframes, scheme=scheme)` construction; (2) one `engine.process_trade(trade)` per trade in stream order, collecting only `update.completed` (the `update.current` snapshot is ignored); (3) exactly one terminal `engine.finalize_trading_day()`. Timeframes passed are `TIMEFRAMES = (3, 5)` (SC/tests/test_candle_parity.py:39).

#### How it drives the BATCH builder (function, inputs)

SC/tests/test_candle_parity.py:131-133 (research scheme) and :160-162 (CT scheme)

```python
    batch = build_tick_bars_from_frame(
        frame, TIMEFRAMES, scheme=RESEARCH_SESSION_SCHEME, tick_size=DEFAULT_TICK_SIZE
    )
```

The `frame` input is built from the SAME `Trade` objects â€” SC/tests/test_candle_parity.py:77-89

```python
def _trades_to_frame(trades: list[Trade], *, tick_size: float) -> pd.DataFrame:
    """Render the trade stream as the batch builder's input DataFrame.

    ``price`` is the points value (ticks * tick_size); the builder rounds it back
    to integer ticks, mirroring how the streaming engine already carries ticks.
    """
    return pd.DataFrame(
        {
            "ts_event": [t.event_ts_utc for t in trades],
            "price": [t.price_ticks * tick_size for t in trades],
            "size": [t.size for t in trades],
        }
    )
```

So both sides get the identical event sequence; the only transformation is ticks -> points -> (round) ticks.

#### What it asserts equal â€” field-by-field via `dataclasses.astuple`, after a canonical sort

SC/tests/test_candle_parity.py:106-121 (verbatim)

```python
def _sort_key(bar: Bar) -> tuple[int, str, int]:
    return (bar.timeframe_ticks, bar.trading_day.isoformat(), bar.bar_index)


def _assert_bars_equal(streaming: list[Bar], batch: list[Bar]) -> None:
    """Both lists sorted by (timeframe, trading_day, bar_index), then every field
    of every bar compared exactly."""
    assert len(streaming) == len(batch), (
        f"bar count differs: streaming={len(streaming)} batch={len(batch)}"
    )
    s_sorted = sorted(streaming, key=_sort_key)
    b_sorted = sorted(batch, key=_sort_key)
    for s, b in zip(s_sorted, b_sorted, strict=True):
        # astuple compares EVERY field (ohlc ticks, volume, trade_count,
        # is_complete, is_partial, close_reason, bar_id, timestamps, ...).
        assert astuple(s) == astuple(b), f"bar mismatch:\n streaming={s}\n    batch={b}"
```

Not a digest and not a per-field enumeration: it is whole-object tuple equality (`astuple`) per bar, plus an equal-length precondition. Note it compares sorted lists, so emission ORDER is not itself asserted â€” only the `(timeframe, trading_day, bar_index)`-keyed content.

Additional sanity assertions in the two parity tests â€” SC/tests/test_candle_parity.py:137-145

```python
    # Sanity: the stream really does straddle the 18:00 ET boundary -> two trading
    # days -> at least one END_OF_DAY bar per timeframe (the first day's partial).
    trading_days = {b.trading_day for b in streaming}
    assert len(trading_days) == 2, (
        f"expected the stream to span 2 ET trading days, got {trading_days}"
    )
    end_of_day = [b for b in streaming if b.close_reason and b.close_reason.value == "end_of_day"]
    assert end_of_day, "expected at least one END_OF_DAY bar from the day rollover"
    assert all(b.is_partial and not b.is_complete for b in end_of_day)
```

SC/tests/test_candle_parity.py:166-172

```python
    # Sanity: trades land in the 16:00-18:00 CT closed window (21:00-23:00 UTC), so
    # the dropped trades mean fewer total trades aggregated than the raw stream.
    total_trade_count = sum(b.trade_count for b in streaming if b.timeframe_ticks == TIMEFRAMES[0])
    assert total_trade_count < len(trades), (
        "expected some trades dropped in the CT closed window; "
        f"aggregated {total_trade_count} of {len(trades)}"
    )
```

SC/tests/test_candle_parity.py:175-179 (empty-input case)

```python
def test_empty_frame_returns_no_bars() -> None:
    """Empty input yields ``[]`` (seed.py:56-57 empty-case), matching a stream with
    no trades through the streaming engine."""
    empty = pd.DataFrame({"ts_event": [], "price": [], "size": []})
    assert build_tick_bars_from_frame(empty, TIMEFRAMES) == []
```

#### Fixture data â€” SYNTHETIC, generated in-test, no file

SC/tests/test_candle_parity.py:42-74

```python
def _synthetic_trades() -> list[Trade]:
    """A deterministic trade stream spanning the 18:00 ET trading-day boundary.
    ...
    """
    base = datetime(2025, 6, 2, 13, 30, 0, tzinfo=UTC)
    trades: list[Trade] = []
    price_ticks = 80_000  # e.g. 20000.00 points at 0.25 tick size
    for i in range(300):
        # Deterministic zig-zag: rises 7, dips 3, rises 2, repeats. Gives real
        # intrabar highs/lows that differ from open and close.
        step = (7, -3, 2, -5, 4)[i % 5]
        price_ticks += step
        trades.append(
            Trade(
                event_ts_utc=base + timedelta(seconds=150 * i),
                price_ticks=price_ticks,
                size=1 + (i % 4),  # 1..4, so volume != trade_count
            )
        )
    return trades
```

300 trades, one every 150 s from `2025-06-02 13:30:00Z`, fixed integer-tick zig-zag, `size = 1 + (i % 4)` so `volume != trade_count`. No parquet/CSV, no store path, no `now()`, no RNG. Timeframes `(3, 5)` are chosen small "so a few hundred trades fill several complete bars per day plus a trailing partial (END_OF_DAY) on each day" (SC/tests/test_candle_parity.py:37-39). The stream is engineered to cross 22:00 UTC (18:00 ET rollover), the 21:00-23:00 UTC CT closed window and 23:00 UTC (18:00 CT rollover) â€” SC/tests/test_candle_parity.py:43-58.

#### Does it call `finalize_trading_day()`?

YES â€” exactly once per streaming run, as the terminal call in `_streaming_bars`. SC/tests/test_candle_parity.py:102

```python
    bars.extend(engine.finalize_trading_day())
```

#### Related (not the parity test, but the same pattern against real data)

SC `validation/test_duckdb_streaming_parity.py` drives the same engine against DuckDB-bucketed trades â€” SC/validation/test_duckdb_streaming_parity.py:105 and :111

```python
    eng = CandleEngine(timeframes=(TICK_COUNT,), scheme=RESEARCH_SESSION_SCHEME)
...
    bars.extend(eng.finalize_trading_day())  # day's trailing partial -> END_OF_DAY
```

Its module docstring names the gate/supporting split: SC/validation/test_duckdb_streaming_parity.py:14-17 â€” "This is the SUPPORTING test ... The GATE (research vs Trade-Lab's real WIRE order) is `test_production_pair_parity.py`." That harness reads the local Databento store and skips if absent; it was NOT executed here.

---

### C2 â€” `build_tick_bars_from_frame`

Single definition: SC `src/strategy_core/candles/batch.py:38`. (Grep across all three trees finds no other definition; TL's original was retired â€” see TL/backend/src/trade_lab/domain/candles.py:1-9, "W2 P1e retired the last TL-local bar construction (the `services/seed.py` Chicago display seed) ... NO local bar building remains in production", and `backend/src/trade_lab/services/seed.py` no longer exists.)

#### Full signature

SC/src/strategy_core/candles/batch.py:38-44

```python
def build_tick_bars_from_frame(
    frame: "pd.DataFrame",
    timeframes: tuple[int, ...],
    *,
    scheme: SessionScheme = RESEARCH_SESSION_SCHEME,
    tick_size: float = DEFAULT_TICK_SIZE,
) -> list[Bar]:
```

Required frame columns: `ts_event` (UTC), `price` (tick-aligned points), `size` (SC/src/strategy_core/candles/batch.py:47-48).

#### The bucketing expression (how trades are grouped into bars)

SC/src/strategy_core/candles/batch.py:118-138

```python
    # seed.py:88 -- cumcount is taken per trading day so bar_index restarts daily.
    day_groups = work.groupby("trading_day", sort=True)
    bars: list[Bar] = []
    for timeframe in sorted(set(timeframes)):
        # seed.py:91 -- floor-divide the per-day running count to bucket trades
        # into fixed N-trade bars.
        work["bar_index"] = (day_groups.cumcount() // timeframe).to_numpy()
        agg = (
            work.groupby(["trading_day", "bar_index"], sort=True)
            .agg(
                open_ts=("ts_event", "first"),
                close_ts=("ts_event", "last"),
                open_ticks=("price_ticks", "first"),
                close_ticks=("price_ticks", "last"),
                high_ticks=("price_ticks", "max"),
                low_ticks=("price_ticks", "min"),
                volume=("size", "sum"),
                trade_count=("price_ticks", "size"),
            )
            .reset_index()
        )
```

The bucket key is COUNT-based, not time-based: `day_groups.cumcount() // timeframe`. Ordering that makes this well-defined â€” SC/src/strategy_core/candles/batch.py:114-116

```python
    work = work.sort_values("ts_event", kind="stable").reset_index(drop=True)
    if work.empty:
        return []
```

Price quantization feeding the aggregation â€” SC/src/strategy_core/candles/batch.py:86-88

```python
    # seed.py:67 -- price (points) -> integer ticks via round-half-to-even rint,
    # exactly matching the streaming Trade's already-integer price_ticks.
    price_ticks = np.rint(frame["price"].to_numpy(dtype="float64") / tick_size).astype("int64")
```

#### How `trading_day` is assigned per row

SC/src/strategy_core/candles/batch.py:98-113

```python
    # seed.py:77-81 -- localize to the scheme timezone, derive seconds-of-day and
    # the trading day (roll into the next calendar day at/after the boundary).
    local = work["ts_event"].dt.tz_convert(scheme.timezone)
    sod = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    cal_date = local.dt.tz_localize(None).dt.floor("D")
    boundary_sod = _seconds_of(scheme.trading_day_boundary)
    roll = (sod >= boundary_sod).astype("int64")
    work["trading_day"] = cal_date + pd.to_timedelta(roll, unit="D")

    # seed.py:82-84 -- drop the scheme's closed window (half-open [start, end));
    # None (research scheme) drops nothing.
    if scheme.closed_window is not None:
        closed_start_sod = _seconds_of(scheme.closed_window[0])
        closed_end_sod = _seconds_of(scheme.closed_window[1])
        closed = (sod >= closed_start_sod) & (sod < closed_end_sod)
        work = work[~closed]
```

Helper â€” SC/src/strategy_core/candles/batch.py:32-35

```python
def _seconds_of(t: time) -> int:
    """Seconds-of-day for a local wall-clock ``time`` (mirrors the streaming
    classifier's minute-aligned boundaries; matches ``seed.py:78`` sod math)."""
    return t.hour * 3600 + t.minute * 60 + t.second
```

Note: this is the vectorized twin of `trading_day_for` (C3) â€” same `>=` boundary rule, same closed-window half-open drop, but computed in pandas rather than via `zoneinfo`/`classify_session`. `trading_day` is materialized as a midnight-floored naive Timestamp and only converted to a `date` at emit time (`agg["trading_day"].dt.date`, batch.py:147).

#### How `bar_index` is assigned

SC/src/strategy_core/candles/batch.py:124

```python
        work["bar_index"] = (day_groups.cumcount() // timeframe).to_numpy()
```

`day_groups` is `work.groupby("trading_day", sort=True)` (batch.py:119) computed ONCE before the timeframe loop, so the running count restarts at 0 for each trading day and `bar_index` is `floor(per-day ordinal / timeframe)` â€” 0-based, per `(trading_day, timeframe)`, matching the streaming allocator.

Emitted `bar_id` â€” SC/src/strategy_core/candles/batch.py:164 `bar_id=make_bar_id(timeframe, td, bi)`, shared with streaming via SC/src/strategy_core/candles/_ids.py:27 `return f"{timeframe_ticks}t:{trading_day.isoformat()}:{bar_index}"`.

#### Trailing partial: EMITTED (never discarded), labelled END_OF_DAY

There is no drop of the last group. Every `(trading_day, bar_index)` group becomes a `Bar`; completeness is decided per group by comparing its trade count to the timeframe â€” SC/src/strategy_core/candles/batch.py:172-177

```python
                # seed.py:109 -- complete iff it filled exactly N trades; the day's
                # trailing partial is END_OF_DAY, matching finalize_trading_day.
                is_complete=tc == timeframe,
                is_partial=tc != timeframe,
                close_reason=CloseReason.COMPLETE if tc == timeframe else CloseReason.END_OF_DAY,
```

Because the cumcount is per trading day, the "trailing partial" arises once per `(trading_day, timeframe)` â€” which is exactly what the streaming side produces via the mid-stream rollover freeze plus the terminal `finalize_trading_day()`. The full emit block (vectorized, one comprehension) is SC/src/strategy_core/candles/batch.py:147-192; its comment at :139-146 states the vectorization is byte-identical to the earlier per-row `.itertuples()` build, including nanosecond truncation in `dt.to_pydatetime`.

---

### C3 â€” `CandleEngine.process_trade`, `trading_day_for`, the boundary constant

`CandleEngine` is defined at SC/src/strategy_core/candles/streaming.py:86; `process_trade` at SC/src/strategy_core/candles/streaming.py:115.

#### Trading-day rollover branch

SC/src/strategy_core/candles/streaming.py:139-145

```python
        for timeframe in self.timeframes:
            candle = current.get(timeframe)
            # candles.py:135-137 -- trading-day rollover: freeze the open bar as
            # END_OF_DAY (incomplete) and reopen on the new day.
            if candle is not None and candle.trading_day != trading_day:
                completed.append(candle.freeze(complete=False, reason=CloseReason.END_OF_DAY))
                candle = None
```

The prior gate that produces `trading_day` (and the closed-window skip) â€” SC/src/strategy_core/candles/streaming.py:129-132

```python
        # candles.py:124-126 -- trades with no trading day (closed window) are skipped.
        trading_day = trading_day_for(trade.event_ts_utc, self._scheme)
        if trading_day is None:
            return self.snapshot_update(())
```

#### Emit-on-fill branch

SC/src/strategy_core/candles/streaming.py:169-183

```python
            # candles.py:161-168 -- accumulate into the open bar. High/low use
            # strict comparisons so equal prices leave them untouched.
            candle.close_ts_utc = event_ts_utc
            candle.close_ticks = price_ticks
            if price_ticks > candle.high_ticks:
                candle.high_ticks = price_ticks
            if price_ticks < candle.low_ticks:
                candle.low_ticks = price_ticks
            candle.volume += size
            candle.trade_count += 1
            # candles.py:169-171 -- exact fill closes the bar COMPLETE and clears it.
            if candle.trade_count == timeframe:
                completed.append(candle.freeze(complete=True, reason=CloseReason.COMPLETE))
                del current[timeframe]
        return self.snapshot_update(tuple(completed))
```

The companion "open a fresh bar" branch, including the `timeframe == 1` emit-immediately special case â€” SC/src/strategy_core/candles/streaming.py:146-168

```python
            if candle is None:
                # candles.py:138-159 -- open a fresh bar seeded by this trade.
                bar_index = self._allocate_bar_index(timeframe, trading_day)
                new_candle = _MutableCandle(
                    timeframe_ticks=timeframe,
                    trading_day=trading_day,
                    bar_index=bar_index,
                    bar_id=make_bar_id(timeframe, trading_day, bar_index),
                    open_ts_utc=event_ts_utc,
                    close_ts_utc=event_ts_utc,
                    open_ticks=price_ticks,
                    high_ticks=price_ticks,
                    low_ticks=price_ticks,
                    close_ticks=price_ticks,
                    volume=size,
                    trade_count=1,
                )
                # candles.py:154-159 -- 1-tick bars close on their seeding trade.
                if timeframe == 1:
                    completed.append(new_candle.freeze(complete=True, reason=CloseReason.COMPLETE))
                else:
                    current[timeframe] = new_candle
                continue
```

Per-day index allocator â€” SC/src/strategy_core/candles/streaming.py:208-216

```python
    def _allocate_bar_index(self, timeframe: int, trading_day: date) -> int:
        """Hand out a monotonic per-``(timeframe, trading_day)`` index from 0.

        Exact port of ``CandleEngine._allocate_bar_index`` (``candles.py:188-192``).
        """
        key = (timeframe, trading_day)
        bar_index = self._next_index.get(key, 0)
        self._next_index[key] = bar_index + 1
        return bar_index
```

Terminal flush (the streaming counterpart of the batch trailing partial) â€” SC/src/strategy_core/candles/streaming.py:194-206

```python
    def finalize_trading_day(self) -> tuple[Bar, ...]:
        """Explicitly close every incomplete bar at its last trade as END_OF_DAY.

        Exact port of ``CandleEngine.finalize_trading_day`` (``candles.py:178-186``):
        each open bar is frozen incomplete with ``END_OF_DAY`` and the current map
        is cleared. Returns the freshly-closed bars in current-dict (ascending
        timeframe) order.
        """
        completed = tuple(
            c.freeze(complete=False, reason=CloseReason.END_OF_DAY) for c in self._current.values()
        )
        self._current.clear()
        return completed
```

#### `trading_day_for` in full

SC/src/strategy_core/decisions/sessions.py:112-122

```python
def trading_day_for(
    ts_utc: datetime,
    scheme: SessionScheme = RESEARCH_SESSION_SCHEME,
) -> date | None:
    """Return the trading day a UTC timestamp belongs to, or ``None`` if closed.

    The trading-day primitive the candle builders share. Identical rule to
    ``classify_session`` step 3; returns ``None`` for timestamps inside the
    scheme's ``closed_window`` (which have no trading day).
    """
    return classify_session(ts_utc, scheme).trading_day
```

It delegates; the actual boundary comparison lives in `classify_session` â€” SC/src/strategy_core/decisions/sessions.py:90-109

```python
    local = _to_local(ts_utc, scheme)
    local_time = local.time()

    if scheme.closed_window is not None:
        closed_start, closed_end = scheme.closed_window
        if closed_start <= local_time < closed_end:
            return SessionInfo(trading_day=None, session=_CLOSED_SESSION, local_ts=local)

    if local_time >= scheme.trading_day_boundary:
        trading_day = local.date() + timedelta(days=1)
    else:
        trading_day = local.date()

    session = _NO_SESSION
    for name, window in scheme.sessions.items():
        if window.contains(local_time):
            session = name
            break

    return SessionInfo(trading_day=trading_day, session=session, local_ts=local)
```

and the UTC->local conversion â€” SC/src/strategy_core/decisions/sessions.py:58-68

```python
def _to_local(ts_utc: datetime, scheme: SessionScheme) -> datetime:
    """Convert a tz-aware UTC timestamp into the scheme's local timezone.
    ...
    """
    if ts_utc.tzinfo is None:
        raise ValueError("ts_utc must be timezone-aware (UTC); got a naive datetime")
    return ts_utc.astimezone(ZoneInfo(scheme.timezone))
```

#### The day-boundary constant

SC/src/strategy_core/constants.py:143-145

```python
# â”€â”€ Sessions (ET) â€” canonical dashboard_utility scheme (v3 re-clock) â”€â”€â”€â”€â”€â”€â”€â”€â”€
SESSION_TIMEZONE = "US/Eastern"
TRADING_DAY_BOUNDARY = time(18, 0)  # CME 6pm ET rollover (UNCHANGED in v3)
```

Carried onto the scheme the engines are parameterized by â€” SC/src/strategy_core/constants.py:168-177

```python
RESEARCH_SESSION_SCHEME = SessionScheme(
    timezone=SESSION_TIMEZONE,
    trading_day_boundary=TRADING_DAY_BOUNDARY,
    sessions={
        "asia": SessionWindow(time(19, 0), time(2, 45), crosses_midnight=True),
        "london": SessionWindow(time(3, 0), time(8, 0)),
        "ny": SessionWindow(time(9, 0), RTH_END),
    },
    closed_window=None,  # research drops nothing; bars span the full 18:00->18:00 ET day
)
```

and the non-canonical CT scheme used by the second parity test â€” SC/src/strategy_core/constants.py:183-192

```python
TRADE_LAB_CT_SESSION_SCHEME = SessionScheme(
    timezone="America/Chicago",
    trading_day_boundary=time(18, 0),
    sessions={
        "asia": SessionWindow(time(18, 0), time(2, 0), crosses_midnight=True),
        "london": SessionWindow(time(2, 0), time(8, 0)),
        "ny": SessionWindow(time(8, 0), time(16, 0)),
    },
    closed_window=(time(16, 0), time(18, 0)),
)
```

Provenance note for the `>=` boundary â€” SC/src/strategy_core/decisions/sessions.py:13-20 records that the research builder never named the rule, and grounds it in `strategy_contract.py:53` `_TRADING_DAY_BOUNDARY = time(18, 0)` and TL `domain/sessions.py:44` `local_time >= time(18, 0)`.

---

### C4 â€” Does any time-bucketing / resampling of trades already exist?

Answer: **YES â€” but none of it is in Strategy-Core.** SC has zero time-bucketing of trades; TL has zero (backend and frontend); QL has several independent time-bar paths (DuckDB `time_bucket` and pandas `.resample`), none of which is wired to the SC candle engine or covered by a streaming/batch parity test.

Patterns run (ripgrep, all three trees, plus TS-specific sweeps):

```
resample
pd\.Grouper
Grouper
date_trunc
time_bucket
dt\.floor
\.floor\(
\.dt\.round
groupby\(   /  group_by\(  /  GROUP BY
Math\.floor\([^)]*(time|ts|timestamp)   |  bucket | intervalMs | barSeconds | timeframeSeconds   (TS/TSX)
startOf\( | truncate\(                                                          (TS/TSX)
```

`date_trunc` returned zero hits in all three trees.

#### SC â€” no time bucketing of trades

| Hit | What it produces |
| --- | --- |
| SC/src/strategy_core/candles/batch.py:102 `cal_date = local.dt.tz_localize(None).dt.floor("D")` | Calendar-DAY floor used only to derive the trading-day label; not a bar bucket. |
| SC/src/strategy_core/candles/batch.py:119, :126 `groupby("trading_day")`, `groupby(["trading_day","bar_index"])` | The tick-count bucketing of C2 â€” grouped by a DAY label plus a count-derived index, never by a time bucket. |
| SC/src/strategy_core/data/databento_parquet.py:682 `valid = np.isfinite(ticks) & (ticks == np.floor(ticks))` | Numeric integer-grid validation of prices; nothing to do with time. |
| SC/validation/phase4b_validate.py:122 `cal_date = local.dt.tz_localize(None).dt.floor("D")` | Same trading-day derivation inside a validation harness. |
| SC/W2_SC_DIFF.txt:1393 | Diff text of the same line; not code. |

Grep for `min_bars|1min|minute_bar|time_bar|TimeBar` over `SC/src` returned no matches â€” there is no time-bar type, constant or code path in the engine.

#### TL â€” none

`resample|time_bucket|Grouper|dt.floor` over `TL/backend`: **no matches.** `groupby(|group_by(|GROUP BY` (case-insensitive) over `TL/backend/src`: **no matches.**

Frontend TS sweep found only non-time-bar hits:

| Hit | What it produces |
| --- | --- |
| TL/frontend/src/chart/viewModels.ts:39 `const toTimestamp = (iso: string): UTCTimestamp => Math.floor(new Date(iso).getTime() / 1000)` | ISO string -> whole-second epoch for lightweight-charts; unit conversion, not bucketing. |
| TL/frontend/src/chart/viewModels.ts:43-49 `chartTimestamp` (`base = Math.floor(Date.parse(\`${bar.tradingDay}T00:00:00Z\`) / 1000); return (base + bar.barIndex)`) | Synthesizes a SYNTHETIC 1-second-per-bar x-axis coordinate from `tradingDay + barIndex` so tick bars plot evenly; it consumes an existing bar, it does not aggregate trades. |
| TL/frontend/src/performance/viewModels.ts:163-167, api/types.ts:256 | "buckets" keyed by close-reason / anomaly category, not time. |
| TL/frontend/src/strip/viewModels.ts:168 `const hourKey = Math.floor(epochMs / 3_600_000)` | Hour key for a display/strip grouping of already-built items (not trade aggregation). |
| TL/frontend/src/{execution,intel,strip,performance}/viewModels.ts (`Math.floor(ms/1000)`, `/3600`, `/60`, `/12`) | Duration and index formatting. |

Confirming TL owns no bar construction at all â€” TL/backend/src/trade_lab/domain/candles.py:1-9

```python
"""Tick-bar DTO/display types. Bar AGGREGATION lives in Strategy-Core (D2).

The local candle-engine shadow implementation was deleted in D2: authoritative
live/replay bars are produced by Strategy-Core's streaming engine and mapped into
these compatibility types at the adapter seam (``strategy_core_service``). W2 P1e
retired the last TL-local bar construction (the ``services/seed.py`` Chicago
display seed) â€” live warm-up bars now come from the engine via the trading-day
replay, so NO local bar building remains in production.
"""
```

(TL/BASELINE_REPORT.md:658 states the same as prose: "**No resampler in Strategy-Core.** There is no minute/hour resampling code in the engine; bars are tick-count aggregations only (batch.py / streaming.py)." That file is an untracked report in the TL tree, cited only as corroboration.)

#### QL â€” several time-bucketing paths (all pre-existing, none SC-connected)

| Hit | What it produces |
| --- | --- |
| QL/src/alpha_lab/agents/data_infra/tick_store.py:416-508 `build_bars_from_ticks(..., bar_size="5 minutes")`, `time_bucket(INTERVAL '{bar_size}', ts_event) AS bar_time` at :467 and :488 | DuckDB SQL time bars: OHLCV per fixed wall-clock interval, from top-of-book mid when `bid_px_00`/`ask_px_00` exist (:465-483) else raw trade `price` (:486-500); returns a DataFrame indexed by `bar_time`. No trading-day label, no `bar_index`, no partial/complete flags. |
| QL/src/alpha_lab/agents/data_infra/aggregation.py:17-27 `_RESAMPLE_RULES` (`1min`,`3min`,`5min`,`10min`,`15min`,`30min`,`1h`,`4h`) | Timeframe -> pandas resample-rule table. |
| QL/src/alpha_lab/agents/data_infra/aggregation.py:132-146 `bars_1m[...].resample(rule).agg({open:first, high:max, low:min, close:last, volume:sum}).dropna(subset=["open"])` | Resamples 1m OHLCV up to 3m..4H. Empty buckets are dropped via `dropna`. |
| QL/src/alpha_lab/agents/data_infra/aggregation.py:149-177 `_aggregate_daily` (`groupby("_td")`) | Daily bars grouped by a trading-date string taken from `session_id` (regex extract) or falling back to `index.date` â€” the docstring at :151-153 says "A CME trading day runs 18:00 ET to 17:00 ET next day", but the fallback path groups by CALENDAR date. |
| QL/src/alpha_lab/agents/data_infra/aggregation.py:30-90 `aggregate_tick_bars` (`"group": [i // tick_count for i in range(n)]`, :70) | A SEPARATE, older tick-bar aggregator â€” count bucketing like SC's, but with different trailing-partial semantics: :84-86 "Drop partial final chunk if less than 50 % full" (`if len(agg) > 1 and agg.iloc[-1]["tick_count"] < tick_count * 0.5`) and no trading-day reset. This differs from SC's always-emit END_OF_DAY partial. |
| QL/src/alpha_lab/experiment/features.py:190-196 `time_bucket(INTERVAL '5 minutes', ts_event) AS bucket ... GROUP BY bucket` | 5-minute close series feeding a stdev-of-returns volatility feature (`vol_buckets` -> `vol_returns` -> `vol_full`). |
| QL/src/alpha_lab/experiment/features.py:206-213 `time_bucket(INTERVAL '1 minute', ts_event) AS bucket ... GROUP BY bucket` | 1-minute close series feeding the recent-window volatility feature. |
| QL/test-chart/nq_candlestick.py:75-94 `df.resample("5min").agg({...})` | Standalone chart script: 1m -> 5m OHLCV. |
| QL/scripts/dashboard.py:263 (docstring) | "builds 1m bars day-by-day for memory efficiency, then resamples to all standard timeframes" â€” orchestrates `aggregation.py` above. |
| QL/src/alpha_lab/agents/data_infra/providers/polygon.py:155-158; providers/databento.py:355 | Provider comments/errors: fetch 1m and let the caller resample. |
| QL/src/alpha_lab/propsim/bootstrap.py:3; PROPSIM_BASELINE.md:167; model_evaluator.py:396 | STATISTICAL resampling (bootstrap with replacement over trading days / block bootstrap) â€” not time bucketing. Listed to disambiguate the grep hits. |
| QL/docs/DECISIONS.md:100-101 | Decision record: fetch 1m from Polygon and resample locally; daily uses session-aware grouping. |
| QL/dashboard-ui/src (TS) | Only `setInterval` polling (TradingView.tsx:117, ObservationPanel.tsx:24) and `Math.floor(Date.parse(...)/1000)` for marker times (TradeOverlays.tsx:74). No aggregation. |

Cross-cutting fact: none of the QL time-bar paths emits `strategy_core.types.Bar`, none carries `trading_day`/`bar_index`/`bar_id`/`is_complete`/`close_reason`, and no test in any tree asserts parity between a QL time-bar builder and a streaming engine (the only candle parity tests are SC/tests/test_candle_parity.py, SC/validation/test_duckdb_streaming_parity.py and SC/validation/test_production_pair_parity.py, all tick-bar).

---

### C5 â€” Where the streaming/batch PARITY CONTRACT is stated

It is stated in FOUR module docstrings (package, streaming, batch, test) plus the bar-id module. All quoted verbatim.

**Package docstring** â€” SC/src/strategy_core/candles/__init__.py:1-7

```python
"""Shared candle layer: a vectorized batch builder and a streaming builder.

Both produce identical :class:`strategy_core.types.Bar` sequences from the same
trades under the same :class:`strategy_core.types.SessionScheme`; the parity test
(`tests/test_candle_parity.py`) locks them together. Promoted from Trade-Lab's
``build_tick_bars_from_frame`` (batch) and ``CandleEngine`` (streaming).
"""
```

**Streaming engine docstring** â€” SC/src/strategy_core/candles/streaming.py:1-16

```python
"""Streaming, event-at-a-time tick-bar builder.

Promoted from Trade-Lab's ``CandleEngine`` /  ``_MutableCandle``
(``Trade-Lab/backend/src/trade_lab/domain/candles.py:35-196``) and generalized so
the trading-day calendar is supplied by a :class:`~strategy_core.types.SessionScheme`
instead of the hardcoded Chicago ``SessionClassifier``. The bar *mechanics* --
per-timeframe current bar, freeze-and-restart on a trading-day change as
``END_OF_DAY`` (incomplete), accumulate high/low/close/volume/trade_count, freeze
``COMPLETE`` when ``trade_count == timeframe``, and the ``timeframe == 1`` emit-now
special case -- are reproduced exactly. The batch builder in ``candles/batch.py``
produces an identical :class:`~strategy_core.types.Bar` sequence; the parity test
(``tests/test_candle_parity.py``) locks the two paths together.

Emits :class:`strategy_core.types.Bar` (not Trade-Lab's ``Candle``) and uses
:class:`strategy_core.types.CloseReason`.
"""
```

**Batch builder docstring** â€” SC/src/strategy_core/candles/batch.py:1-15

```python
"""Vectorized (pandas + numpy) tick-bar builder over a trades DataFrame.

Promoted from Trade-Lab's ``build_tick_bars_from_frame``
(``Trade-Lab/backend/src/trade_lab/services/seed.py:42-131``) and generalized so the
trading-day calendar / closed window come from a
:class:`~strategy_core.types.SessionScheme` instead of the hardcoded Chicago
16:00/18:00 boundaries. It is the vectorized equivalent of feeding the same trades
through :class:`strategy_core.candles.streaming.CandleEngine`; the parity test
(``tests/test_candle_parity.py``) locks the two paths together.

This is the ONLY engine module permitted to import pandas (see package rules): the
batch path is the research/warm-up fast path, run off the event loop. Emits
:class:`strategy_core.types.Bar` (not Trade-Lab's ``Candle``) with
:class:`strategy_core.types.CloseReason`.
"""
```

The batch function docstring also restates the ordering guarantee â€” SC/src/strategy_core/candles/batch.py:64-66

```python
    Returns bars over timeframes in ascending order; for a fixed timeframe the
    bars are in ``(trading_day, bar_index)`` order, identical to the streaming
    builder's emission for the same trades.
```

**Parity test docstring â€” the fullest statement of WHY** â€” SC/tests/test_candle_parity.py:1-17

```python
"""Parity lock: the batch and streaming candle builders must agree exactly.

The streaming :class:`~strategy_core.candles.streaming.CandleEngine` (event-at-a-time,
the live/replay path) and the vectorized
:func:`~strategy_core.candles.batch.build_tick_bars_from_frame` (research / warm-up
path) are two implementations of the same N-trade tick bar. Any drift between them
means a strategy trained on the batch bars would execute on subtly different live
bars -- the exact failure this package exists to prevent. This test feeds the SAME
deterministic synthetic trade stream through both and asserts the resulting
:class:`~strategy_core.types.Bar` lists are field-for-field identical.

All timestamps are fixed, timezone-aware UTC datetimes (never ``now()``/random) so the
test is fully deterministic. The stream deliberately spans the 18:00 ET trading-day
boundary so the day-rollover ``END_OF_DAY`` behavior is exercised, and a second case
runs under ``TRADE_LAB_CT_SESSION_SCHEME`` so the closed-window drop path is shown to
be parity-safe too.
"""
```

**Shared bar-id module â€” the join-key half of the contract** â€” SC/src/strategy_core/candles/_ids.py:1-10

```python
"""Shared deterministic bar-id construction for both candle builders.

A bar id is the join key the rest of the stack uses to reconcile a streaming
bar with its batch-built twin, so the two builders MUST format it identically.
Single-sourced here and imported by both ``candles/batch.py`` and
``candles/streaming.py``.

Ported verbatim from Trade-Lab's ``make_bar_id``:
``Trade-Lab/backend/src/trade_lab/domain/candles.py:195-196``.
"""
```

**Extension of the contract to the research store** â€” SC/validation/test_duckdb_streaming_parity.py:1-5

```python
"""Standing parity test: research DuckDB batch bars == engine STREAMING bars, BYTE-FOR-BYTE.

Why this pair: research trains on DuckDB-built bars; Trade-Lab serves on streaming-built
bars. Zero-drift requires the two to be identical on the SAME spec.
```

---

#### Compliance note

No file in SC, TL or QL was modified, staged or committed; only read-only tooling (`Read`, `Grep`, `head`/`sed`/`ls`, `find`) was used. No build, training, replay or test was executed â€” not even `pytest --collect-only`. No data under any NQ day directory was read. The only write is this file in the scratchpad.
