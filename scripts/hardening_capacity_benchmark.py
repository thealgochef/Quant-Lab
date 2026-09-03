"""HARDENING_CAPACITY_POLICY_V1 — the §4.4 capacity benchmarks (F-17).

Synthetic data only; no research simulation, no real source, everything
under a temporary directory. Two benchmarks:

* **B1 — event-detail writer**: the production
  ``alpha_lab.propsim.event_detail.build_account_event_detail`` fed by a
  GENERATOR of ``(path_record, walk_result)`` pairs (never materialized),
  published through the immutable store as a sidecar producer, then read
  back through the production reader ``load_account_event_detail``.
* **B2 — regime-stratified event summary**: the production
  ``build_stratified_prop_body`` over lazily generated event-detail
  partitions (the loader seam), i.e. the external DuckDB aggregation path.

Each (benchmark, size, mode) runs in a FRESH subprocess. ``mode=rss`` records
total process RSS with a native monitor — Windows ``GetProcessMemoryInfo``
(``PeakWorkingSetSize`` / ``WorkingSetSize``) through ``ctypes``, POSIX
``resource.getrusage`` — as the peak increase over a baseline taken after
the imports and the fixture setup; ``mode=tracemalloc`` records the Python
allocation peak separately (tracemalloc inflates RSS, so it never taints the
RSS numbers). Available RAM at start comes from ``GlobalMemoryStatusEx`` /
``sysconf``. ``psutil`` is not required.

Gates (plan §4.4, ``HARDENING_CAPACITY_POLICY_V1``): minimum available RAM
at start ≥ 8 GiB; 1M-row peak RSS increase ≤ 1.5 GiB; 1M-row Python
allocation peak ≤ 512 MiB (B1) / ≤ 768 MiB (B2); 1M-row wall time ≤ 300 s;
per-row RSS slope 500k→1M ≤ 1.25 × max(slope 250k→500k, 64 B/row);
projection at the registered maximum (10,000,000 detail rows; 5,000,000
summary rows) ≤ min(6 GiB, 50 % of the minimum measured available RAM);
serialized artifact ≤ 2 GiB at 10M detail rows / ≤ 256 MiB at 5M summary
rows; byte-identical output hash on a repeated run. Measured and
extrapolated values are reported in separate tables.

    python scripts/hardening_capacity_benchmark.py --out-dir <evidence dir>
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

POLICY_ID = "HARDENING_CAPACITY_POLICY_V1"
DEFAULT_SIZES = (250_000, 500_000, 1_000_000)
GIB = 1024**3
MIB = 1024**2
B1_EVENTS_PER_PATH = 200
#: HARDENING-BACKEND-FIX §9.3: the B1 input shapes — ``normal`` (200 events per
#: path, the registered policy's benchmark shape), ``skewed`` (five paths carry
#: 90 % of the rows) and ``dense`` (250 paths — ONE path block — whose rows
#: exceed one partition many times); every shape must obey the hard
#: resident-row bound (max partition rows ≤ max_rows_per_partition).
B1_SHAPES = ("normal", "skewed", "dense")
B1_EXTRA_SHAPES = ("skewed", "dense")
B1_SKEWED_WHALES = 5
B1_DENSE_PATHS = 250
B2_EVENTS_PER_PATH = 4
B2_SIMULATIONS = 4
B2_PARTITION_ROWS = 50_000
GATES = {
    "B1": {
        "registered_max_rows": 10_000_000,
        "artifact_bytes_max": 2 * GIB,
        "tracemalloc_peak_max": 512 * MIB,
    },
    "B2": {
        "registered_max_rows": 5_000_000,
        "artifact_bytes_max": 256 * MIB,
        "tracemalloc_peak_max": 768 * MIB,
    },
}
MIN_AVAILABLE_RAM = 8 * GIB
PEAK_RSS_INCREASE_MAX = int(1.5 * GIB)
WALL_MAX_SECONDS = 300.0
PROJECTION_ABS_MAX = 6 * GIB
SLOPE_FLOOR_BYTES_PER_ROW = 64.0
SLOPE_GROWTH_MAX = 1.25
_EXTRA_KEYS = (
    "paths",
    "partitions",
    "summary_rows",
    "events_total",
    "build_seconds",
    "read_seconds",
    "max_partition_rows",
    "max_resident_rows_observed",
    "max_rows_per_partition",
)


# ── native memory readers (no psutil) ───────────────────────────────────────


def _win32_memory_counters() -> tuple[int, int]:
    """(working set, peak working set) of this process in bytes."""

    from ctypes import wintypes  # noqa: PLC0415

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = ()
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    psapi.GetProcessMemoryInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    )
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(ProcessMemoryCounters)
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    ):
        raise OSError(ctypes.get_last_error(), "GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _win32_available_ram() -> tuple[int, int]:
    """(available physical bytes, total physical bytes)."""

    from ctypes import wintypes  # noqa: PLC0415

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    kernel32.GlobalMemoryStatusEx.argtypes = (ctypes.POINTER(MemoryStatusEx),)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    return int(status.ullAvailPhys), int(status.ullTotalPhys)


def memory_counters() -> tuple[int, int]:
    """(current RSS, lifetime peak RSS) in bytes via the native reader."""

    if sys.platform == "win32":
        return _win32_memory_counters()
    import resource  # noqa: PLC0415

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    peak = peak if sys.platform == "darwin" else peak * 1024
    current = peak
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
    return current, peak


def available_ram() -> tuple[int, int]:
    if sys.platform == "win32":
        return _win32_available_ram()
    page = os.sysconf("SC_PAGE_SIZE")
    return int(os.sysconf("SC_AVPHYS_PAGES") * page), int(os.sysconf("SC_PHYS_PAGES") * page)


# ── B1: the event-detail writer over a generator ────────────────────────────


class _Phase:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


class _Payload:
    __slots__ = ("realized_delta", "amount", "trader_amount")

    def __init__(self, realized_delta=None, amount=None, trader_amount=None) -> None:
        self.realized_delta = realized_delta
        self.amount = amount
        self.trader_amount = trader_amount


class _Event:
    __slots__ = (
        "path_instance_id",
        "event_ordinal",
        "event_type",
        "event_ts_utc",
        "account_id",
        "account_ordinal",
        "event_id",
        "trading_day",
        "account_phase",
        "source_trade_id",
        "source_candidate_id",
        "payload",
    )


class _Walk:
    __slots__ = ("events",)

    def __init__(self, events) -> None:
        self.events = events


class _Record:
    __slots__ = ("path_instance_id", "draw_ordinal")

    def __init__(self, path_instance_id: str, draw_ordinal: int) -> None:
        self.path_instance_id = path_instance_id
        self.draw_ordinal = draw_ordinal


_B1_TYPES = ("equity_update", "fee", "payout", "daily_halt", "equity_update", "equity_update")
_FUNDED = _Phase("funded")


def _b1_path_plan(rows: int, shape: str) -> list[int]:
    """Events per path in draw order for one shape (sums to ``rows``)."""

    if shape == "normal":
        paths = rows // B1_EVENTS_PER_PATH
        remainder = rows - paths * B1_EVENTS_PER_PATH
        return [B1_EVENTS_PER_PATH] * paths + ([remainder] if remainder else [])
    if shape == "dense":
        per_path = rows // B1_DENSE_PATHS
        remainder = rows - per_path * B1_DENSE_PATHS
        plan = [per_path] * B1_DENSE_PATHS
        for index in range(remainder):
            plan[index] += 1
        return [count for count in plan if count]
    if shape == "skewed":
        whale = (rows * 9) // (10 * B1_SKEWED_WHALES)  # 90 % of the rows in five paths
        plan = [whale] * B1_SKEWED_WHALES
        remaining = rows - whale * B1_SKEWED_WHALES
        full = remaining // B1_EVENTS_PER_PATH
        remainder = remaining - full * B1_EVENTS_PER_PATH
        plan += [B1_EVENTS_PER_PATH] * full + ([remainder] if remainder else [])
        return plan
    raise ValueError(f"unregistered B1 shape {shape!r}; registered: {B1_SHAPES}")


def _b1_walks(rows: int, simulation_id: str, shape: str = "normal"):
    """Lazily yield ``(record, walk)`` pairs for exactly ``rows`` events."""

    for draw, count in enumerate(_b1_path_plan(rows, shape)):
        path_id = f"path-{draw:05d}-{simulation_id[:16]}"
        events = []
        for ordinal in range(count):
            event = _Event()
            event.path_instance_id = path_id
            event.event_ordinal = ordinal
            event_type = _B1_TYPES[ordinal % len(_B1_TYPES)]
            event.event_type = event_type
            day = 6 + (ordinal // 12) % 20
            event.event_ts_utc = (
                f"2020-01-{day:02d}T{10 + ordinal % 8:02d}:{ordinal % 60:02d}:00+00:00"
            )
            event.trading_day = f"2020-01-{day:02d}"
            event.account_id = f"{path_id}:acct-0"
            event.account_ordinal = 0
            event.event_id = hashlib.sha256(f"{path_id}:{ordinal}".encode()).hexdigest()
            event.account_phase = _FUNDED
            if event_type in ("equity_update", "payout"):
                event.source_trade_id = f"trade-{(draw * 7 + ordinal) % 1000:04d}"
                event.source_candidate_id = f"cand-{(draw * 7 + ordinal) % 1000:04d}"
            else:
                event.source_trade_id = None
                event.source_candidate_id = None
            amount = float(((draw * 31 + ordinal * 17) % 200) - 100) / 4.0
            if event_type == "equity_update":
                event.payload = _Payload(realized_delta=amount)
            elif event_type == "fee":
                event.payload = _Payload(amount=abs(amount))
            elif event_type == "payout":
                event.payload = _Payload(trader_amount=abs(amount))
            else:
                event.payload = _Payload()
            events.append(event)
        yield _Record(path_id, draw), _Walk(tuple(events))


def _b1_path_count(rows: int, shape: str = "normal") -> int:
    return len(_b1_path_plan(rows, shape))


def _v2_simulation_envelope(seed: int):
    from alpha_lab.propsim.event_detail import (  # noqa: PLC0415
        EVENT_DETAIL_POLICY_PARQUET_V2,
        event_detail_identity_fields,
    )
    from alpha_lab.propsim.simulation import (  # noqa: PLC0415
        AccountSimulationEnvelope,
        AccountSimulationPayload,
    )

    payload = AccountSimulationPayload(
        core_replay_id="a" * 64,
        gross_trade_stream_hash="b" * 64,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id="d" * 64,
        trade_path_bundle_manifest_sha256="e" * 64,
        path_capability_report_id="f" * 64,
        account_policy_set_id="1" * 64,
        simulation_mode="day_block_bootstrap",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id="day_block_bootstrap_h90_v1",
        stress_scenario_id=None,
        seed=seed,
        n_paths=1,
        **event_detail_identity_fields(EVENT_DETAIL_POLICY_PARQUET_V2),
    )
    return AccountSimulationEnvelope.from_payload(payload)


def run_b1(rows: int, workdir: Path, shape: str = "normal") -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        save_envelope_immutable,
    )
    from alpha_lab.propsim.calendar import BOOTSTRAP_CLOCK_POLICY  # noqa: PLC0415
    from alpha_lab.propsim.event_detail import (  # noqa: PLC0415
        EVENT_DETAIL_BUDGET_V2,
        build_account_event_detail,
        load_account_event_detail,
        load_account_event_detail_manifest,
    )

    envelope = _v2_simulation_envelope(seed=rows)
    simulation_id = envelope.account_simulation_id
    store_root = workdir / "store"
    bundle_facts: dict = {}

    def producer(directory: Path):
        from alpha_lab.propsim import event_detail as event_detail_module  # noqa: PLC0415

        # review RB-04: an INDEPENDENT observation of the writer's resident batch
        # at its Parquet seam (never the manifest's own claim): every flushed
        # table's row count is the number of rows resident right before the flush
        observed = {"max_rows": 0}
        real_write = event_detail_module._write_parquet

        def _observing_write(table, path):
            observed["max_rows"] = max(observed["max_rows"], int(table.num_rows))
            real_write(table, path)

        event_detail_module._write_parquet = _observing_write
        try:
            bundle = build_account_event_detail(
                _b1_walks(rows, simulation_id, shape),
                clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
                budget=EVENT_DETAIL_BUDGET_V2,
                event_order_policy_id="prop_account_event_order_v1",
                directory=directory,
                total_rows=rows,
                path_count=_b1_path_count(rows, shape),
            )
        finally:
            event_detail_module._write_parquet = real_write
        bundle_facts["partitions"] = bundle.partition_count
        bundle_facts["max_partition_rows"] = bundle.max_partition_rows
        bundle_facts["max_resident_rows_observed"] = observed["max_rows"]
        bundle_facts["total_bytes"] = bundle.total_bytes
        bundle_facts["manifest_sha256"] = bundle.manifest_sidecar.sha256
        return bundle.produced()

    started = time.perf_counter()
    save_envelope_immutable(store_root, "account_simulations", envelope, sidecar_producer=producer)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    rows_read = 0
    digest = hashlib.sha256()
    for frame in load_account_event_detail(store_root, simulation_id):
        rows_read += len(frame)
    manifest = load_account_event_detail_manifest(store_root, simulation_id)
    for entry in manifest["partitions"]:
        digest.update(entry["sha256"].encode("ascii"))
    read_seconds = time.perf_counter() - started
    if rows_read != rows:
        raise RuntimeError(f"B1 read back {rows_read} rows, expected {rows}")
    return {
        "rows": rows,
        "shape": shape,
        "paths": _b1_path_count(rows, shape),
        "partitions": bundle_facts["partitions"],
        "max_partition_rows": int(bundle_facts["max_partition_rows"]),
        "max_resident_rows_observed": int(bundle_facts["max_resident_rows_observed"]),
        "max_rows_per_partition": int(EVENT_DETAIL_BUDGET_V2.max_rows_per_partition),
        "partition_bound_policy_id": manifest["partition_bound_policy_id"],
        "artifact_bytes": int(bundle_facts["total_bytes"]),
        "output_hash": digest.hexdigest(),
        "manifest_sha256": bundle_facts["manifest_sha256"],
        "build_seconds": build_seconds,
        "read_seconds": read_seconds,
        "wall_seconds": build_seconds + read_seconds,
        "uniqueness_check": manifest["event_id_uniqueness"]["artifact_check"],
    }


# ── B2: the stratified event summary over lazy partitions ───────────────────


def _b2_partition_frames(rows: int, simulation_index: int):
    """Lazily yield D15 partition frames (``B2_PARTITION_ROWS`` events each)
    built vectorized — a path lives in one partition; every (path, type) is
    a distinct summary key, so summary rows ≈ input rows."""

    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop import (  # noqa: PLC0415
        EVENT_DETAIL_COLUMNS,
    )
    from alpha_lab.propsim.event_detail import EVENT_TYPE_PRECEDENCE  # noqa: PLC0415

    types = np.array(["equity_update", "fee", "payout", "daily_halt"])
    precedence = np.array([EVENT_TYPE_PRECEDENCE[t] for t in types], dtype="int32")
    produced = 0
    block = 0
    while produced < rows:
        count = min(B2_PARTITION_ROWS, rows - produced)
        ordinal = np.arange(produced, produced + count, dtype="int64")
        path_index = ordinal // B2_EVENTS_PER_PATH
        event_index = ordinal % B2_EVENTS_PER_PATH
        type_index = (path_index + event_index) % len(types)
        event_types = types[type_index]
        trade_index = (path_index * 7 + event_index) % 1100
        has_trade = np.isin(event_types, ("equity_update", "payout")) & (event_index % 3 != 2)
        path_ids = np.char.add(
            np.char.add(f"path-{simulation_index}-", path_index.astype(str)), "-x"
        )
        trades = np.where(has_trade, np.char.add("trade-", trade_index.astype(str)), None)
        hours = 10 + (event_index % 8)
        days = np.char.zfill((6 + event_index % 5).astype(str), 2)
        stamps = np.char.add(
            np.char.add(
                np.char.add(
                    np.char.add(np.char.add("2020-01-", days), "T"),
                    np.char.zfill(hours.astype(str), 2),
                ),
                np.char.add(":", np.char.zfill((path_index % 60).astype(str), 2)),
            ),
            ":00+00:00",
        )
        amounts = (((path_index * 31 + event_index * 17) % 200) - 100).astype("float64") / 4.0
        frame = pd.DataFrame(
            {
                "path_instance_id": path_ids,
                "path_block_id": np.full(count, block, dtype="int64"),
                "account_id": np.char.add(path_ids, ":acct-0"),
                "event_id": np.char.zfill(
                    np.char.add(f"{simulation_index}", ordinal.astype(str)), 64
                ),
                "event_ts_utc": stamps,
                "trading_day": np.char.add("2020-01-", days),
                "clock_policy_id": np.full(count, "synthetic_path_clock_v1"),
                "event_precedence": precedence[type_index],
                "event_ordinal": ordinal,
                "event_type": event_types,
                "account_phase": np.full(count, "funded"),
                "source_trade_id": trades,
                "source_candidate_id": np.where(
                    has_trade, np.char.add("cand-", trade_index.astype(str)), None
                ),
                "amount": amounts,
            },
            columns=list(EVENT_DETAIL_COLUMNS),
        )
        produced += count
        block += 1
        yield frame


def run_b2(rows: int, workdir: Path) -> dict:
    import pandas as pd  # noqa: PLC0415

    from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (  # noqa: PLC0415
        ObservationGranularity,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (  # noqa: PLC0415
        RegimeAssignmentEvidenceRef,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop import (  # noqa: PLC0415
        build_stratified_prop_body,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: PLC0415
        save_or_reuse_envelope,
    )

    store_root = workdir / "store"
    simulations: dict[str, tuple[str, str]] = {}
    rows_by_simulation: dict[str, tuple[int, int]] = {}
    per_simulation = rows // B2_SIMULATIONS
    for index in range(B2_SIMULATIONS):
        envelope = _v2_simulation_envelope(seed=1000 + index)
        save_or_reuse_envelope(store_root, "account_simulations", envelope)
        simulation_id = envelope.account_simulation_id
        simulations[simulation_id] = (f"firm_{index}", "day_block_bootstrap")
        share = per_simulation if index < B2_SIMULATIONS - 1 else rows - per_simulation * (
            B2_SIMULATIONS - 1
        )
        rows_by_simulation[simulation_id] = (index, share)

    def loader(_root, simulation_id):
        index, share = rows_by_simulation[simulation_id]
        return _b2_partition_frames(share, index)

    trade_regimes = pd.DataFrame(
        {
            "trade_id": [f"trade-{index}" for index in range(1000)],
            "valid": [index % 5 != 4 for index in range(1000)],
            "canonical_reporting_cluster_id": [
                None if index % 5 == 4 else index % 3 for index in range(1000)
            ],
        }
    )
    evidence = RegimeAssignmentEvidenceRef(
        observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
        regime_fit_ids=("1" * 64, "2" * 64),
        regime_fold_set_id="3" * 64,
        fold_schedule_id="4" * 64,
        regime_oos_assignment_id="5" * 64,
        assignment_table_sha256="6" * 64,
        assignment_schema_hash="7" * 64,
    )
    started = time.perf_counter()
    result = build_stratified_prop_body(
        root=store_root,
        core_replay_id="a" * 64,
        simulations=simulations,
        trade_regimes=trade_regimes,
        evidence=evidence,
        loader=loader,
    )
    wall = time.perf_counter() - started
    return {
        "rows": rows,
        "summary_rows": int(result.summary_rows),
        "artifact_bytes": len(result.summary_bytes),
        "output_hash": result.summary_sha256,
        "events_total": int(result.body.events_total),
        "wall_seconds": wall,
    }


# ── worker / parent ─────────────────────────────────────────────────────────


def _worker(
    benchmark: str, rows: int, mode: str, workdir: Path, shape: str = "normal"
) -> dict:
    import gc  # noqa: PLC0415

    # import everything the run touches BEFORE the baseline
    import duckdb  # noqa: F401, PLC0415
    import pandas  # noqa: F401, PLC0415
    import pyarrow  # noqa: F401, PLC0415

    import alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop  # noqa: F401, PLC0415
    import alpha_lab.propsim.event_detail  # noqa: F401, PLC0415
    import alpha_lab.propsim.simulation  # noqa: F401, PLC0415

    available, total = available_ram()
    gc.collect()
    baseline_rss, baseline_peak = memory_counters()
    if mode == "tracemalloc":
        tracemalloc.start()
    facts = run_b1(rows, workdir, shape) if benchmark == "B1" else run_b2(rows, workdir)
    python_peak = None
    if mode == "tracemalloc":
        python_peak = int(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()
    final_rss, final_peak = memory_counters()
    facts.update(
        {
            "benchmark": benchmark,
            "mode": mode,
            "available_ram_bytes_at_start": available,
            "total_ram_bytes": total,
            "baseline_rss_bytes": baseline_rss,
            "baseline_peak_rss_bytes": baseline_peak,
            "final_rss_bytes": final_rss,
            "final_peak_rss_bytes": final_peak,
            "peak_rss_increase_bytes": max(0, final_peak - baseline_rss),
            "python_allocation_peak_bytes": python_peak,
        }
    )
    return facts


def _spawn(benchmark: str, rows: int, mode: str, shape: str = "normal") -> dict:
    workdir = Path(tempfile.mkdtemp(prefix=f"hardening_capacity_{benchmark}_{rows}_{shape}_"))
    try:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            benchmark,
            "--rows",
            str(rows),
            "--mode",
            mode,
            "--workdir",
            str(workdir),
            "--shape",
            shape,
        ]
        started = time.perf_counter()
        completed = subprocess.run(  # noqa: S603 - the exact local interpreter/script only
            command, capture_output=True, text=True, cwd=ROOT, check=False
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"worker {benchmark} rows={rows} mode={mode} failed:\n{completed.stderr[-4000:]}"
            )
        line = [text for text in completed.stdout.splitlines() if text.startswith("{")][-1]
        facts = json.loads(line)
        facts["subprocess_wall_seconds"] = elapsed
        facts["command"] = " ".join(command)
        return facts
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _gib(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.3f} GiB"


def _mib(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value / MIB:.1f} MiB"


def _evaluate(benchmark: str, sizes: tuple[int, ...], runs: dict) -> dict:
    """Gates over the FIRST rss run of each size (the repeat proves
    determinism; the tracemalloc run supplies the Python peak)."""

    rss = {size: runs[(size, "rss", 1)] for size in sizes}
    repeat = {size: runs[(size, "rss", 2)] for size in sizes}
    traced = {size: runs[(size, "tracemalloc", 1)] for size in sizes}
    gates = GATES[benchmark]
    largest = sizes[-1]
    every_run = [*rss.values(), *repeat.values(), *traced.values()]
    min_available = min(facts["available_ram_bytes_at_start"] for facts in every_run)
    increase = {size: rss[size]["peak_rss_increase_bytes"] for size in sizes}
    slope_lo = (increase[sizes[1]] - increase[sizes[0]]) / (sizes[1] - sizes[0])
    slope_hi = (increase[sizes[2]] - increase[sizes[1]]) / (sizes[2] - sizes[1])
    slope_gate = SLOPE_GROWTH_MAX * max(slope_lo, SLOPE_FLOOR_BYTES_PER_ROW)
    projection_slope = max(slope_hi, 0.0)
    registered_max = gates["registered_max_rows"]
    projected_rss = increase[largest] + projection_slope * (registered_max - largest)
    projection_limit = min(PROJECTION_ABS_MAX, 0.5 * min_available)
    artifact_1m = rss[largest]["artifact_bytes"]
    projected_artifact = artifact_1m * registered_max / largest
    determinism = all(rss[size]["output_hash"] == repeat[size]["output_hash"] for size in sizes)
    python_peak_1m = traced[largest]["python_allocation_peak_bytes"]
    wall_1m = rss[largest]["wall_seconds"]
    tracemalloc_max = gates["tracemalloc_peak_max"]
    artifact_max = gates["artifact_bytes_max"]
    gate_rows = [
        (
            "minimum available RAM at start ≥ 8 GiB",
            min_available >= MIN_AVAILABLE_RAM,
            _gib(min_available),
            _gib(MIN_AVAILABLE_RAM),
        ),
        (
            "1M-row peak RSS increase ≤ 1.5 GiB",
            increase[largest] <= PEAK_RSS_INCREASE_MAX,
            _gib(increase[largest]),
            _gib(PEAK_RSS_INCREASE_MAX),
        ),
        (
            f"1M-row Python allocation peak ≤ {tracemalloc_max // MIB} MiB",
            python_peak_1m <= tracemalloc_max,
            _mib(python_peak_1m),
            _mib(tracemalloc_max),
        ),
        (
            "1M-row wall time ≤ 300 s",
            wall_1m <= WALL_MAX_SECONDS,
            f"{wall_1m:.1f} s",
            f"{WALL_MAX_SECONDS:.0f} s",
        ),
        (
            "growth 500k→1M: slope ≤ 1.25 × max(slope 250k→500k, 64 B/row)",
            slope_hi <= slope_gate,
            f"{slope_hi:.1f} B/row",
            f"{slope_gate:.1f} B/row",
        ),
        (
            f"projection at the registered maximum ({registered_max:,} rows) ≤ "
            "min(6 GiB, 50 % of min available RAM)",
            projected_rss <= projection_limit,
            _gib(projected_rss),
            _gib(projection_limit),
        ),
        (
            f"serialized artifact ≤ {_gib(artifact_max)} at {registered_max:,} rows",
            projected_artifact <= artifact_max,
            _gib(projected_artifact),
            _gib(artifact_max),
        ),
        (
            "determinism: byte-identical output hash on repeat (every size)",
            determinism,
            "identical" if determinism else "DIFFERENT",
            "identical",
        ),
    ]
    return {
        "benchmark": benchmark,
        "sizes": list(sizes),
        "measured": {
            str(size): {
                "peak_rss_increase_bytes": increase[size],
                "peak_rss_increase_bytes_repeat": repeat[size]["peak_rss_increase_bytes"],
                "python_allocation_peak_bytes": traced[size]["python_allocation_peak_bytes"],
                "wall_seconds": rss[size]["wall_seconds"],
                "wall_seconds_repeat": repeat[size]["wall_seconds"],
                "artifact_bytes": rss[size]["artifact_bytes"],
                "output_hash": rss[size]["output_hash"],
                "output_hash_repeat": repeat[size]["output_hash"],
                "available_ram_bytes_at_start": rss[size]["available_ram_bytes_at_start"],
                "baseline_rss_bytes": rss[size]["baseline_rss_bytes"],
                "final_peak_rss_bytes": rss[size]["final_peak_rss_bytes"],
                **{key: rss[size][key] for key in _EXTRA_KEYS if key in rss[size]},
            }
            for size in sizes
        },
        "extrapolated": {
            "slope_250k_500k_bytes_per_row": slope_lo,
            "slope_500k_1m_bytes_per_row": slope_hi,
            "slope_gate_bytes_per_row": slope_gate,
            "projection_slope_bytes_per_row": projection_slope,
            "registered_max_rows": registered_max,
            "projected_peak_rss_increase_bytes_at_max": projected_rss,
            "projection_limit_bytes": projection_limit,
            "projected_artifact_bytes_at_max": projected_artifact,
            "artifact_limit_bytes": gates["artifact_bytes_max"],
        },
        "gates": [
            {"gate": name, "passed": bool(passed), "measured": measured, "limit": limit}
            for name, passed, measured, limit in gate_rows
        ],
        "passed": all(passed for _name, passed, _m, _l in gate_rows),
        "min_available_ram_bytes": min_available,
    }


def _evaluate_b1_shapes(largest: int, normal_runs: dict, shape_runs: dict) -> dict:
    """HARDENING-BACKEND-FIX §9.3: the hard resident-row bound must hold for
    EVERY lawful shape (a shape-conditional projection is not sufficient),
    and the skewed / dense 1M-row runs must meet the same absolute gates."""

    bound_rows = []
    for (size, mode, repeat), facts in normal_runs.items():
        bound_rows.append(("normal", size, mode, repeat, facts))
    for (shape, mode, repeat), facts in shape_runs.items():
        bound_rows.append((shape, largest, mode, repeat, facts))
    bound_ok = all(
        facts["max_resident_rows_observed"] <= facts["max_rows_per_partition"]
        for *_k, facts in bound_rows
    )
    bound_measured = max(facts["max_resident_rows_observed"] for *_k, facts in bound_rows)
    bound_limit = bound_rows[0][-1]["max_rows_per_partition"]
    gate_rows = [
        (
            "maximum resident writer batch (observed at the writer's Parquet seam) ≤ "
            "max_rows_per_partition (every shape, every size)",
            bound_ok,
            f"{bound_measured:,} rows",
            f"{bound_limit:,} rows",
        )
    ]
    shapes: dict[str, dict] = {}
    for shape in B1_EXTRA_SHAPES:
        rss = shape_runs[(shape, "rss", 1)]
        repeat = shape_runs[(shape, "rss", 2)]
        traced = shape_runs[(shape, "tracemalloc", 1)]
        determinism = rss["output_hash"] == repeat["output_hash"]
        rows = [
            (
                f"{shape}: 1M-row peak RSS increase ≤ 1.5 GiB",
                rss["peak_rss_increase_bytes"] <= PEAK_RSS_INCREASE_MAX,
                _gib(rss["peak_rss_increase_bytes"]),
                _gib(PEAK_RSS_INCREASE_MAX),
            ),
            (
                f"{shape}: 1M-row Python allocation peak ≤ 512 MiB",
                traced["python_allocation_peak_bytes"] <= GATES["B1"]["tracemalloc_peak_max"],
                _mib(traced["python_allocation_peak_bytes"]),
                _mib(GATES["B1"]["tracemalloc_peak_max"]),
            ),
            (
                f"{shape}: 1M-row wall time ≤ 300 s",
                rss["wall_seconds"] <= WALL_MAX_SECONDS,
                f"{rss['wall_seconds']:.1f} s",
                f"{WALL_MAX_SECONDS:.0f} s",
            ),
            (
                f"{shape}: byte-identical output hash on repeat",
                determinism,
                "identical" if determinism else "DIFFERENT",
                "identical",
            ),
            (
                f"{shape}: maximum resident writer batch (observed) ≤ max_rows_per_partition",
                rss["max_resident_rows_observed"] <= rss["max_rows_per_partition"],
                f"{rss['max_resident_rows_observed']:,} rows",
                f"{rss['max_rows_per_partition']:,} rows",
            ),
        ]
        gate_rows.extend(rows)
        shapes[shape] = {
            "rows": largest,
            "paths": rss["paths"],
            "partitions": rss["partitions"],
            "max_partition_rows": rss["max_partition_rows"],
            "max_resident_rows_observed": rss["max_resident_rows_observed"],
            "peak_rss_increase_bytes": rss["peak_rss_increase_bytes"],
            "peak_rss_increase_bytes_repeat": repeat["peak_rss_increase_bytes"],
            "python_allocation_peak_bytes": traced["python_allocation_peak_bytes"],
            "wall_seconds": rss["wall_seconds"],
            "wall_seconds_repeat": repeat["wall_seconds"],
            "artifact_bytes": rss["artifact_bytes"],
            "output_hash": rss["output_hash"],
            "output_hash_repeat": repeat["output_hash"],
        }
    return {
        "shapes": shapes,
        "gates": [
            {"gate": name, "passed": bool(passed), "measured": measured, "limit": limit}
            for name, passed, measured, limit in gate_rows
        ],
        "passed": all(passed for _name, passed, _m, _l in gate_rows),
    }


def _markdown(report: dict) -> str:
    lines = [
        "# CAPACITY_BENCHMARKS — HARDENING_CAPACITY_POLICY_V1 (plan §4.4, F-17)",
        "",
        f"Run {report['started_at']} → {report['finished_at']} on {report['host']['platform']} "
        f"({report['host']['processor']}, {report['host']['logical_cpus']} logical CPUs, "
        f"{_gib(report['host']['total_ram_bytes'])} RAM); Python {report['host']['python']}, "
        f"pyarrow {report['host']['pyarrow']}, duckdb {report['host']['duckdb']}, "
        f"pandas {report['host']['pandas']}. Synthetic data only; every run in a fresh "
        "subprocess under a temporary directory; RSS from the native process monitor "
        f"({report['host']['rss_monitor']}); `tracemalloc` in a SEPARATE run (supplementary).",
        "",
        "Command:",
        "",
        "```text",
        report["command"],
        "```",
        "",
    ]
    for benchmark in [name for name in ("B1", "B2") if name in report["benchmarks"]]:
        section = report["benchmarks"][benchmark]
        title = (
            "B1 — event-detail writer (production writer over a generator + store publish + "
            "production reader)"
            if benchmark == "B1"
            else "B2 — regime-stratified event summary (production builder; external DuckDB "
            "aggregation)"
        )
        lines += [f"## {title}", "", f"**Overall: {'PASS' if section['passed'] else 'FAIL'}**", ""]
        lines += [
            "### Measured",
            "",
            "| input rows | peak RSS increase | repeat | Python alloc peak | wall | repeat wall "
            "| artifact bytes | output hash (first 12) | repeat hash | available RAM at start |",
            "|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
        ]
        for size in section["sizes"]:
            row = section["measured"][str(size)]
            lines.append(
                f"| {size:,} | {_gib(row['peak_rss_increase_bytes'])} | "
                f"{_gib(row['peak_rss_increase_bytes_repeat'])} | "
                f"{_mib(row['python_allocation_peak_bytes'])} | {row['wall_seconds']:.1f} s | "
                f"{row['wall_seconds_repeat']:.1f} s | {row['artifact_bytes']:,} | "
                f"`{row['output_hash'][:12]}` | `{row['output_hash_repeat'][:12]}` | "
                f"{_gib(row['available_ram_bytes_at_start'])} |"
            )
        first_row = section["measured"][str(section["sizes"][0])]
        extra_keys = [key for key in _EXTRA_KEYS if key in first_row]
        if extra_keys:
            lines += [
                "",
                "| input rows | " + " | ".join(extra_keys) + " |",
                "|---:|" + "---:|" * len(extra_keys),
            ]
            for size in section["sizes"]:
                row = section["measured"][str(size)]
                cells = [
                    f"{row[key]:.1f}" if isinstance(row[key], float) else f"{row[key]:,}"
                    for key in extra_keys
                ]
                lines.append(f"| {size:,} | " + " | ".join(cells) + " |")
        ex = section["extrapolated"]
        lines += [
            "",
            "### Extrapolated (never a pass on its own)",
            "",
            "| quantity | value |",
            "|---|---:|",
            f"| per-row RSS slope 250k→500k | {ex['slope_250k_500k_bytes_per_row']:.1f} B/row |",
            f"| per-row RSS slope 500k→1M | {ex['slope_500k_1m_bytes_per_row']:.1f} B/row |",
            "| slope gate (1.25 × max(slope 250k→500k, 64 B/row)) | "
            f"{ex['slope_gate_bytes_per_row']:.1f} B/row |",
            "| projection slope used (max(slope 500k→1M, 0)) | "
            f"{ex['projection_slope_bytes_per_row']:.1f} B/row |",
            f"| projected peak RSS increase at {ex['registered_max_rows']:,} rows | "
            f"{_gib(ex['projected_peak_rss_increase_bytes_at_max'])} |",
            "| projection limit min(6 GiB, 50 % of min available RAM) | "
            f"{_gib(ex['projection_limit_bytes'])} |",
            f"| projected serialized artifact at {ex['registered_max_rows']:,} rows | "
            f"{_gib(ex['projected_artifact_bytes_at_max'])} |",
            f"| artifact limit | {_gib(ex['artifact_limit_bytes'])} |",
            "",
            "### Gates",
            "",
            "| gate | measured | limit | result |",
            "|---|---:|---:|---|",
        ]
        for gate in section["gates"]:
            lines.append(
                f"| {gate['gate']} | {gate['measured']} | {gate['limit']} | "
                f"{'PASS' if gate['passed'] else 'FAIL'} |"
            )
        lines.append("")
        shapes = section.get("shape_evaluation")
        if shapes:
            lines += [
                "### Worst-lawful-shape proof (HARDENING-BACKEND-FIX §9.3; 1M rows)",
                "",
                "| shape | paths | partitions | max partition rows | peak RSS increase | repeat "
                "| Python alloc peak | wall | artifact bytes | output hash (first 12) "
                "| repeat hash |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
            ]
            for shape, row in shapes["shapes"].items():
                lines.append(
                    f"| {shape} | {row['paths']:,} | {row['partitions']:,} | "
                    f"{row['max_partition_rows']:,} | {_gib(row['peak_rss_increase_bytes'])} | "
                    f"{_gib(row['peak_rss_increase_bytes_repeat'])} | "
                    f"{_mib(row['python_allocation_peak_bytes'])} | {row['wall_seconds']:.1f} s | "
                    f"{row['artifact_bytes']:,} | `{row['output_hash'][:12]}` | "
                    f"`{row['output_hash_repeat'][:12]}` |"
                )
            lines += ["", "| gate | measured | limit | result |", "|---|---:|---:|---|"]
            for gate in shapes["gates"]:
                lines.append(
                    f"| {gate['gate']} | {gate['measured']} | {gate['limit']} | "
                    f"{'PASS' if gate['passed'] else 'FAIL'} |"
                )
            lines.append("")
    lines += [
        "## Reading",
        "",
        "- The measured columns are what this machine did; the extrapolated table is a "
        "linear projection from the measured 1M-row point with the measured 500k→1M slope "
        "and is reported separately (plan §4.4: hardening never passes on extrapolation "
        "alone — the implementation must be block/partition bounded AND the projection must "
        "satisfy every gate).",
        "- B1 counts rows = events; a walk yields 200 events per path (5,000 paths at 1M rows, "
        "20 path blocks of 250 paths); the writer's input is a GENERATOR, its uniqueness "
        "proof is the external DuckDB distinct check, and the reader re-verifies every "
        "partition.",
        "- B2 counts rows = input events; each (path, event type) is a distinct summary key, "
        "so summary rows ≈ input rows (the registered maximum of 5,000,000 summary rows "
        "projects from the 1M-row point).",
        "- `peak RSS increase` = PeakWorkingSetSize after the run − WorkingSetSize baseline "
        "after imports and fixture setup (the lifetime peak cannot be reset on Windows, so "
        "the number is conservative).",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=("B1", "B2"), default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--mode", choices=("rss", "tracemalloc"), default="rss")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--shape", choices=B1_SHAPES, default="normal")
    parser.add_argument(
        "--shapes",
        default=",".join(B1_EXTRA_SHAPES),
        help="the extra B1 shapes proven at the largest size (HARDENING-BACKEND-FIX 9.3)",
    )
    parser.add_argument("--sizes", default=",".join(str(size) for size in DEFAULT_SIZES))
    parser.add_argument("--benchmarks", default="B1,B2")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    if args.worker:
        facts = _worker(
            args.worker, int(args.rows), args.mode, Path(args.workdir), args.shape
        )
        print(json.dumps(facts, sort_keys=True))
        return 0
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if len(sizes) != 3:
        raise SystemExit("exactly three sizes are required (250k / 500k / 1M by the policy)")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    benchmarks: dict[str, dict] = {}
    raw: list[dict] = []
    for benchmark in args.benchmarks.split(","):
        runs: dict[tuple[int, str, int], dict] = {}
        for size in sizes:
            for mode, repeat in (("rss", 1), ("rss", 2), ("tracemalloc", 1)):
                facts = _spawn(benchmark, size, mode)
                facts["repeat"] = repeat
                runs[(size, mode, repeat)] = facts
                raw.append(facts)
                print(
                    f"{benchmark} rows={size:,} mode={mode} repeat={repeat}: "
                    f"peak RSS +{_gib(facts['peak_rss_increase_bytes'])}, "
                    f"wall {facts['wall_seconds']:.1f} s, hash {facts['output_hash'][:12]}",
                    flush=True,
                )
        benchmarks[benchmark] = _evaluate(benchmark, sizes, runs)
        if benchmark == "B1" and args.shapes:
            shape_runs: dict[tuple[str, str, int], dict] = {}
            for shape in [s for s in args.shapes.split(",") if s]:
                for mode, repeat in (("rss", 1), ("rss", 2), ("tracemalloc", 1)):
                    facts = _spawn(benchmark, sizes[-1], mode, shape)
                    facts["repeat"] = repeat
                    shape_runs[(shape, mode, repeat)] = facts
                    raw.append(facts)
                    print(
                        f"{benchmark} shape={shape} rows={sizes[-1]:,} mode={mode} "
                        f"repeat={repeat}: peak RSS +{_gib(facts['peak_rss_increase_bytes'])}, "
                        f"max partition rows {facts['max_partition_rows']:,}, "
                        f"wall {facts['wall_seconds']:.1f} s, hash {facts['output_hash'][:12]}",
                        flush=True,
                    )
            evaluation = _evaluate_b1_shapes(sizes[-1], runs, shape_runs)
            benchmarks[benchmark]["shape_evaluation"] = evaluation
            benchmarks[benchmark]["passed"] = bool(
                benchmarks[benchmark]["passed"] and evaluation["passed"]
            )
    import duckdb  # noqa: PLC0415
    import pandas  # noqa: PLC0415
    import pyarrow  # noqa: PLC0415

    report = {
        "policy_id": POLICY_ID,
        "partition_bound_policy_id": "event_detail_partition_row_bound_v2",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": "python scripts/hardening_capacity_benchmark.py "
        + " ".join(argv if argv is not None else sys.argv[1:]),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "total_ram_bytes": available_ram()[1],
            "python": platform.python_version(),
            "pyarrow": pyarrow.__version__,
            "duckdb": duckdb.__version__,
            "pandas": pandas.__version__,
            "rss_monitor": (
                "Win32 GetProcessMemoryInfo (PeakWorkingSetSize / WorkingSetSize) via ctypes"
                if sys.platform == "win32"
                else "POSIX resource.getrusage ru_maxrss (+ /proc/self/status VmRSS)"
            ),
        },
        "benchmarks": benchmarks,
        "raw_runs": raw,
        "passed": all(section["passed"] for section in benchmarks.values()),
    }
    (out_dir / "CAPACITY_BENCHMARKS.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "CAPACITY_BENCHMARKS.md").write_text(_markdown(report), encoding="utf-8")
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'} -> {out_dir}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
