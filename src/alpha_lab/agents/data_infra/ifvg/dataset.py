"""Legacy capture reader plus the partitioned IFVG v2 replay chain.

The old heterogeneous union is retained strictly for read-only v1
reproduction.  New replay uses :func:`build_ifvg_v2_capture`, keeps the
Strategy-Core trace transient, and returns typed append-only tables.

* a day's capture parquet is consumed only if its stamped ENTERING seed hash
  equals the hash of the seed the chain is actually carrying into that day
  (and its stored end-seed pickle re-hashes to the stamped end hash) — a
  broken/stale middle link therefore invalidates every later day, never
  silently (the ``prev_full_hl`` lesson generalized);
* artifacts are chain-verified too: each day's artifact must have been built
  with the PRIOR day's outputs as seeds (day H/L + NY H/L); a mismatch
  rebuilds the artifact in place;
* the first ``cfg.warmup_days`` chain days are flagged ``is_warmup`` and every
  row carries ``days_of_htf_history`` (training filters are a choice, not a
  rebuild — census: the 4H registry stabilizes days 8-10).
"""

from __future__ import annotations

import json
import pickle
import platform
import statistics
from datetime import date
from time import perf_counter_ns
from typing import TYPE_CHECKING

import pandas as pd
from strategy_core.candles.exchange_calendar import (
    DEFAULT_CONTEXT_SOURCE_COVERAGE,
    ContextSourceCoverage,
)
from strategy_core.strategies.ifvg_smc.replay import (
    ContextReplayTape,
    IfvgContextDayResult,
    run_day,
)
from strategy_core.strategies.ifvg_smc.state import IfvgDaySeed, seed_hash
from strategy_core.structures.context import canonical_json

from .capture_driver import (
    ContextCaptureDayResult,
    capture_single_date,
    capture_single_date_with_context,
    normalize_context_days,
)
from .config import IfvgCaptureConfig, IfvgV3CaptureConfig
from .context_contracts import (
    ContextRecordTable,
    validate_context_foreign_keys,
    validate_context_primary_keys,
    validate_context_table_identity,
)
from .contracts import (
    RecordTable,
    partition_capture_tables,
    stamp_table_contract,
    validate_foreign_keys,
    validate_primary_keys,
    validate_table_identity,
)
from .data_access import (
    ExplorationDataPolicy,
    require_fixed_exploration_allowlist,
)
from .day_artifacts import (
    DayArtifacts,
    DaySeeds,
    build_day_artifacts,
    levels_for_from_frame,
    load_day_artifacts,
    seeds_for_day,
    write_day_artifacts,
)
from .entry_dataset import build_candidate_labels_from_tables

if TYPE_CHECKING:
    from .profiles import ResolvedProfileConfig

__all__ = [
    "assemble_fsm_audit_tables",
    "build_ifvg_capture",
    "build_ifvg_fsm_audit_v1",
    "build_ifvg_v2_capture",
    "CaptureChainResult",
    "ChainStart",
    "FsmAuditBuildResult",
    "V2CaptureResult",
    "V3CaptureResult",
    "build_ifvg_v3_capture",
    "load_accepted_v2_tables",
    "reconcile_v3_core_to_accepted_v2",
    "table_content_hash",
    "CAPTURE_UNION_SCHEMA",
    "LEGACY_CAPTURE_UNION_SCHEMA",
    "conform_capture_frame",
]

_CAP_META_KEY = b"ifvg_capture_meta"

#: The FULL flattened v1-record union (column -> pandas dtype), frozen from the
#: pandas-2 concat of the capture chain. Per-day frames are a union of per-kind
#: record columns, so on any given day most columns are absent or all-NA;
#: pandas 2 excluded such entries from concat result-dtype inference, pandas 3
#: does not — so the union dtype is DECLARED here and every per-day frame is
#: conformed to it before concat (inference-free under either pandas). A column
#: outside this schema means the record shape drifted: bump
#: ``IFVG_RECORD_SCHEMA_VERSION`` and re-freeze; never widen silently.
LEGACY_CAPTURE_UNION_SCHEMA: dict[str, str] = {
    "kind": "object",
    "entering_seed_hash": "object",
    "envelope_schema_version": "int64",
    "envelope_strategy_id": "object",
    "envelope_strategy_version": "object",
    "envelope_profile_hash": "object",
    "envelope_trading_day": "object",
    "envelope_ts_utc": "datetime64[ns, UTC]",
    "envelope_setup_id": "object",
    "htf_tf_seconds": "float64",
    "fvg_fvg_id": "object",
    "fvg_timeframe_seconds": "float64",
    "fvg_direction": "object",
    "fvg_gap_low_ticks": "float64",
    "fvg_gap_high_ticks": "float64",
    "fvg_size_ticks": "float64",
    "fvg_a_bar_id": "object",
    "fvg_c_bar_id": "object",
    "fvg_a_open_ts_utc": "datetime64[ns, UTC]",
    "fvg_confirmed_ts_utc": "datetime64[ns, UTC]",
    "fvg_trading_day": "object",
    "direction": "object",
    "penetration_ticks": "float64",
    "ce_reached": "object",
    "htf_age_seconds": "float64",
    "remaining_fraction": "float64",
    "registry_live_count": "float64",
    "rank": "float64",
    "conflicted": "object",
    "nearest_level_kind": "object",
    "nearest_level_distance_ticks": "float64",
    "session_engine": "object",
    "session_doc": "object",
    "selected": "object",
    "drop_reason": "object",
    "resolution": "object",
    "entry_family": "object",
    "entry_ticks": "float64",
    "stop_ticks": "float64",
    "tp_ticks": "float64",
    "mfe_ticks": "float64",
    "mae_ticks": "float64",
    "bars_in_trade": "float64",
    "tap_ts_utc": "datetime64[ns, UTC]",
    "parent_confirmed_ts_utc": "datetime64[ns, UTC]",
    "lock_ts_utc": "datetime64[ns, UTC]",
    "armed_ts_utc": "datetime64[ns, UTC]",
    "inversion_ts_utc": "datetime64[ns, UTC]",
    "entry_ts_utc": "datetime64[ns, UTC]",
    "htf_fvg_id": "object",
    "parent_fvg_id": "object",
    "opposing_fvg_id": "object",
    "is_warmup": "bool",
    "days_of_htf_history": "int64",
    "parent_tf_seconds": "float64",
    "distance_to_htf_ticks": "float64",
    "elapsed_1m_bars_since_tap": "float64",
    "confirmed_after": "object",
    "fully_formed_after": "object",
    "elapsed_1m_bars_since_selection": "float64",
    "distance_to_parent_ticks": "float64",
    "elapsed_1m_bars_since_lock": "float64",
    "close_through_margin_ticks": "float64",
    "bars_armed_to_inversion": "float64",
    "opposing_size_ticks": "float64",
    "sweep_sweep_confirmed": "object",
    "sweep_swept_kinds": "object",
    "sweep_max_penetration_ticks": "float64",
    "sweep_nearest_unswept_distance_ticks": "float64",
    "sweep_sweep_ts_utc": "datetime64[ns, UTC]",
    "sweep_leg_extreme_ticks": "float64",
    "entry_fvg": "float64",
    "entry_fvg_fvg_id": "object",
    "entry_fvg_timeframe_seconds": "float64",
    "entry_fvg_direction": "object",
    "entry_fvg_gap_low_ticks": "float64",
    "entry_fvg_gap_high_ticks": "float64",
    "entry_fvg_size_ticks": "float64",
    "entry_fvg_a_bar_id": "object",
    "entry_fvg_c_bar_id": "object",
    "entry_fvg_a_open_ts_utc": "datetime64[ns, UTC]",
    "entry_fvg_confirmed_ts_utc": "datetime64[ns, UTC]",
    "entry_fvg_trading_day": "object",
    "risk_ticks": "float64",
    "bars_since_inversion": "float64",
    "entry_to_parent_ticks": "float64",
    "in_engine_session": "object",
    "in_doc_session": "object",
}

# Compatibility spelling for v1 readers/tests.  V2 never conforms to this
# wide union.
CAPTURE_UNION_SCHEMA = LEGACY_CAPTURE_UNION_SCHEMA

#: dtypes that cannot represent NA — a per-day frame missing one of these
#: columns is malformed, not sparse (they are stamped on every row).
_NON_NULLABLE = ("bool", "int64")


def conform_capture_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex/astype one per-day capture frame to ``CAPTURE_UNION_SCHEMA``.

    Absent and all-NA columns get a typed-empty column (their inferred dtype is
    pure noise — the day simply never emitted the field); populated columns are
    cast, so a genuine value/dtype conflict still raises."""
    unknown = [c for c in frame.columns if c not in CAPTURE_UNION_SCHEMA]
    if unknown:
        raise ValueError(f"capture frame columns outside CAPTURE_UNION_SCHEMA: {unknown}")
    out: dict[str, pd.Series] = {}
    for col, dtype in CAPTURE_UNION_SCHEMA.items():
        present = col in frame.columns
        if present and (str(frame[col].dtype) == dtype or not frame[col].isna().all()):
            series = frame[col]
            if str(series.dtype) != dtype:
                series = series.astype(dtype)
        else:
            if dtype in _NON_NULLABLE and present is False:
                raise ValueError(f"non-nullable column {col!r} missing from capture frame")
            if dtype in _NON_NULLABLE:
                series = frame[col].astype(dtype)
            else:
                series = pd.Series(index=frame.index, dtype=dtype)
        out[col] = series
    return pd.DataFrame(out, index=frame.index)


class CaptureChainResult:
    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []
        self.day_funnels: dict[str, dict[str, int]] = {}
        self.rebuilt_days: list[str] = []
        self.cached_days: list[str] = []

    def frame(self) -> pd.DataFrame:
        real = [conform_capture_frame(f) for f in self.frames if len(f)]
        if not real:
            return pd.DataFrame()
        out = pd.concat(real, ignore_index=True)
        drift = {
            c: (str(out[c].dtype), CAPTURE_UNION_SCHEMA[c])
            for c in out.columns
            if str(out[c].dtype) != CAPTURE_UNION_SCHEMA[c]
        }
        if drift:
            raise ValueError(f"capture concat dtypes drifted from schema (got, want): {drift}")
        return out


class ChainStart:
    """QL-only mid-chain start for one sequential v2 replay.

    ``seed`` is the reducer/registry state entering the first replayed day and
    ``day_seeds`` the entering :class:`DaySeeds` expectations that day's
    artifact must have been built with (both produced by the prior day of the
    SAME profile's chain — seeds are profile-bound). This is the documented
    ``start_after_artifact`` fallback of ``ifvg_prop_robust_config_search_v1``
    R0→R1; it involves no Strategy-Core change.
    """

    def __init__(self, *, seed: IfvgDaySeed, day_seeds: DaySeeds) -> None:
        self.seed = seed
        self.day_seeds = day_seeds


class V2CaptureResult:
    """In-memory result of one deterministic allowlisted replay."""

    def __init__(
        self,
        *,
        tables: dict[RecordTable, pd.DataFrame],
        day_funnels: dict[str, dict[str, int]],
        rebuilt_days: list[str],
        cached_artifact_days: list[str],
        bars_by_day: dict[str, tuple],
        access_policy: ExplorationDataPolicy,
        audit_frames: dict[str, pd.DataFrame] | None = None,
        end_seed: IfvgDaySeed | None = None,
        trace_audit_rows: pd.DataFrame | None = None,
    ) -> None:
        self.tables = tables
        self.day_funnels = day_funnels
        self.rebuilt_days = rebuilt_days
        self.cached_artifact_days = cached_artifact_days
        self.bars_by_day = bars_by_day
        self.access_policy = access_policy
        self.audit_frames = audit_frames
        self.end_seed = end_seed
        # Core-trace rows carrying the trace-cut audit kinds (with their global
        # trace_ordinal already assigned) — retained ONLY under an audit
        # capture mode so the per-child fsm-audit companion can be assembled
        # without a second heavyweight replay. Never populated for the plain
        # (audit-disabled) capture path.
        self.trace_audit_rows = trace_audit_rows

    @property
    def candidates(self) -> pd.DataFrame:
        return self.tables[RecordTable.ENTRY_CANDIDATE]

    @property
    def decisions(self) -> pd.DataFrame:
        return self.tables[RecordTable.ELIGIBLE_DECISION]

    @property
    def trades(self) -> pd.DataFrame:
        return self.tables[RecordTable.EXECUTED_TRADE]


def _write_capture(
    frame: pd.DataFrame,
    cfg: IfvgCaptureConfig,
    date_str: str,
    *,
    entering_hash: str | None,
    end_hash: str,
    funnel: dict[str, int],
) -> None:
    del frame, cfg, date_str, entering_hash, end_hash, funnel
    raise PermissionError(
        "heterogeneous IFVG capture caches are frozen v1 artifacts; "
        "v2 persists typed tables through the immutable manifest writer"
    )


def _load_capture(
    cfg: IfvgCaptureConfig, date_str: str, *, expected_entering: str | None
) -> tuple[pd.DataFrame, dict[str, int], IfvgDaySeed] | None:
    import pyarrow.parquet as pq

    cpath = cfg.capture_path(date_str)
    spath = cfg.seed_path(date_str)
    if not cpath.exists() or not spath.exists():
        return None
    try:
        md = pq.read_metadata(cpath).metadata or {}
        meta = json.loads(md.get(_CAP_META_KEY, b"{}"))
    except Exception:
        return None
    if meta.get("entering_seed_hash") != expected_entering:
        return None
    try:
        with open(spath, "rb") as fh:
            end_seed = pickle.load(fh)
    except Exception:
        return None
    if seed_hash(end_seed) != meta.get("end_seed_hash"):
        return None
    return pd.read_parquet(cpath), dict(meta.get("funnel", {})), end_seed


def _chained_seeds(prev: DayArtifacts | None) -> DaySeeds | None:
    """The artifact seeds day N MUST have been built with, given day N-1's
    artifact (None = no expectation, first chain day)."""
    if prev is None:
        return None
    prev_day = date.fromisoformat(prev.date_str)
    if prev.day_hl is None:
        # Empty prior day: expectations carry through from ITS seeds.
        return prev.seeds
    return DaySeeds(
        prev_day=prev_day,
        prev_full_hl=prev.day_hl,
        prev_ny_day=prev_day if prev.ny_hl else prev.seeds.prev_ny_day,
        prev_ny_hl=prev.ny_hl if prev.ny_hl else prev.seeds.prev_ny_hl,
    )


def build_ifvg_capture(
    dates: list[str],
    cfg: IfvgCaptureConfig,
    progress_fn=None,
    *,
    cached_only: bool = False,
) -> CaptureChainResult:
    """``cached_only=True`` is the READ-ONLY rebuild mode: every day must pass
    chain trust verification from the existing per-day caches, or the chain
    raises — it never re-drives the reducer and never (re)writes an artifact,
    capture parquet, or seed pickle."""
    if cfg.identity_lane != "legacy_v1":
        raise RuntimeError(
            "build_ifvg_capture is the frozen v1 heterogeneous reader; "
            "use build_ifvg_v2_capture for repaired replay"
        )
    if not cached_only:
        raise PermissionError(
            "legacy IFVG artifacts are read-only; pass cached_only=True"
        )
    result = CaptureChainResult()
    seed: IfvgDaySeed | None = None
    prev_artifacts: DayArtifacts | None = None

    for chain_idx, date_str in enumerate(dates):
        entering_hash = seed_hash(seed) if seed is not None else None
        cached = _load_capture(cfg, date_str, expected_entering=entering_hash)
        expected_seeds = _chained_seeds(prev_artifacts)
        artifacts = load_day_artifacts(date_str, cfg, expected_seeds=expected_seeds)
        if cached is not None and artifacts is not None:
            frame, funnel, seed = cached
            result.cached_days.append(date_str)
        elif cached_only:
            raise RuntimeError(
                f"cached-only chain: {date_str} failed trust verification "
                f"(capture cached={cached is not None}, artifacts ok={artifacts is not None})"
                " — refusing to re-drive the reducer or rewrite any artifact"
            )
        else:
            if artifacts is None:
                build_seeds = (
                    expected_seeds if expected_seeds is not None else seeds_for_day(date_str, cfg)
                )
                artifacts = build_day_artifacts(date_str, cfg, build_seeds)
                write_day_artifacts(artifacts, cfg)
                result.rebuilt_days.append(date_str)
            day = capture_single_date(date_str, cfg, artifacts=artifacts, seed=seed)
            seed = day.end_seed
            frame, funnel = day.rows, day.funnel
            end_hash = seed_hash(seed)
            _write_capture(
                frame,
                cfg,
                date_str,
                entering_hash=entering_hash,
                end_hash=end_hash,
                funnel=funnel,
            )
            with open(cfg.seed_path(date_str), "wb") as fh:
                pickle.dump(seed, fh)
        if len(frame):
            frame = frame.copy()
            frame["is_warmup"] = chain_idx < cfg.warmup_days
            frame["days_of_htf_history"] = chain_idx
        result.frames.append(frame)
        result.day_funnels[date_str] = funnel
        prev_artifacts = artifacts
        if progress_fn is not None:
            progress_fn(chain_idx + 1, len(dates), date_str)
    return result


def build_ifvg_v2_capture(
    dates: list[str] | tuple[str, ...],
    cfg: IfvgCaptureConfig,
    resolved_profile: ResolvedProfileConfig,
    *,
    access_policy: ExplorationDataPolicy | None = None,
    cached_artifacts_only: bool = False,
    progress_fn=None,
    start_after_artifact: ChainStart | None = None,
    audit_capture_mode: str = "disabled",
    final_day_exhausts_dataset: bool = True,
) -> V2CaptureResult:
    """Replay an explicit allowlisted chain into typed v2 tables.

    Authorization of the complete date list occurs before any config path is
    constructed.  The first date is a deliberate cold start unless
    ``start_after_artifact`` provides a verified profile-matching mid-chain
    seed (the ``ifvg_prop_robust_config_search_v1`` R0→R1 fallback); a
    profile/seed mismatch is refused BEFORE any source path exists.  The last
    day finalizes any open execution as ``open_unresolved`` without P&L.
    """
    if cfg.identity_lane != "v2":
        raise PermissionError("v2 replay cannot use a legacy capture identity")
    if resolved_profile.section_config_hash != cfg.profile_hash:
        raise ValueError(
            "resolved profile hash does not match the capture section"
        )
    if (
        start_after_artifact is not None
        and start_after_artifact.seed.profile_hash != cfg.profile_hash
    ):
        raise PermissionError(
            "start seed profile_hash does not match the capture section; "
            "refused before any source read (seeds are profile-bound)"
        )
    policy = access_policy or ExplorationDataPolicy()
    require_fixed_exploration_allowlist(policy)
    chain_dates = policy.authorize_dates(dates)
    if not chain_dates:
        raise ValueError("IFVG v2 replay requires at least one allowlisted date")

    cold = DaySeeds(
        prev_day=None,
        prev_full_hl=None,
        prev_ny_day=None,
        prev_ny_hl=None,
    )
    first_expected = (
        cold if start_after_artifact is None else start_after_artifact.day_seeds
    )
    seed: IfvgDaySeed | None = (
        None if start_after_artifact is None else start_after_artifact.seed
    )
    previous_artifacts: DayArtifacts | None = None
    trace_frames: list[pd.DataFrame] = []
    audit_frames: dict[str, pd.DataFrame] = {}
    day_funnels: dict[str, dict[str, int]] = {}
    rebuilt_days: list[str] = []
    cached_days: list[str] = []
    bars_by_day: dict[str, tuple] = {}
    core_trace_offset = 0

    for chain_index, date_str in enumerate(chain_dates):
        expected_seeds = _chained_seeds(previous_artifacts) or first_expected
        artifacts = load_day_artifacts(
            date_str,
            cfg,
            expected_seeds=expected_seeds,
            access_policy=policy,
        )
        if artifacts is None:
            if cached_artifacts_only:
                raise RuntimeError(
                    f"cached-artifacts-only v2 replay: {date_str} is missing "
                    "or failed seed trust"
                )
            artifacts = build_day_artifacts(
                date_str,
                cfg,
                expected_seeds,
                access_policy=policy,
            )
            write_day_artifacts(artifacts, cfg, access_policy=policy)
            rebuilt_days.append(date_str)
        else:
            cached_days.append(date_str)

        day_result = capture_single_date(
            date_str,
            cfg,
            artifacts=artifacts,
            seed=seed,
            dataset_exhausted=(
                final_day_exhausts_dataset and chain_index == len(chain_dates) - 1
            ),
            audit_capture_mode=audit_capture_mode,
        )
        seed = day_result.end_seed
        if day_result.audit_rows is not None:
            audit_frame = day_result.audit_rows.copy()
            if not audit_frame.empty:
                # identical stamps to the fsm-audit builder path, so the
                # per-child companion assembly validates the same contracts
                audit_frame["is_warmup"] = chain_index < cfg.warmup_days
                audit_frame["days_of_htf_history"] = chain_index
                audit_frame["evaluation_config_hash"] = (
                    resolved_profile.evaluation_config_hash
                )
                audit_frame["core_trace_ordinal_before_global"] = (
                    audit_frame["stamp_core_trace_ordinal_before"].astype(int)
                    + core_trace_offset
                )
                audit_frame["core_trace_ordinal_after_global"] = (
                    audit_frame["stamp_core_trace_ordinal_after"].astype(int)
                    + core_trace_offset
                )
            audit_frames[date_str] = audit_frame
        frame = day_result.rows.copy()
        if not frame.empty:
            frame["is_warmup"] = chain_index < cfg.warmup_days
            frame["days_of_htf_history"] = chain_index
            frame["evaluation_config_hash"] = (
                resolved_profile.evaluation_config_hash
            )
        trace_frames.append(frame)
        core_trace_offset += len(frame)
        day_funnels[date_str] = day_result.funnel
        bars_by_day[date_str] = tuple(artifacts.bars)
        previous_artifacts = artifacts
        if progress_fn is not None:
            progress_fn(chain_index + 1, len(chain_dates), date_str)
    trace = (
        pd.concat(trace_frames, ignore_index=True, sort=False)
        if any(not frame.empty for frame in trace_frames)
        else pd.DataFrame()
    )
    if not trace.empty:
        trace["trace_ordinal"] = range(len(trace))
    tables = partition_capture_tables(trace)
    for table, frame in tuple(tables.items()):
        stamped = frame.copy()
        stamped["evaluation_config_hash"] = (
            resolved_profile.evaluation_config_hash
        )
        tables[table] = stamped

    labels = build_candidate_labels_from_tables(
        tables[RecordTable.ENTRY_CANDIDATE],
        bars_by_day=bars_by_day,
        tick_size=cfg.tick_size,
        resolved_profile=resolved_profile,
    )
    labels = stamp_table_contract(RecordTable.CANDIDATE_LABEL, labels)
    tables[RecordTable.CANDIDATE_LABEL] = labels

    for table, frame in tables.items():
        validate_primary_keys(table, frame)
        validate_table_identity(table, frame)
    validate_foreign_keys(tables)
    policy.assert_zero_forbidden_access()
    trace_audit_rows: pd.DataFrame | None = None
    if audit_capture_mode != "disabled":
        from .audit_contracts import TRACE_AUDIT_KIND_BY_TABLE  # noqa: PLC0415

        trace_audit_rows = (
            trace.loc[
                trace["kind"].isin(tuple(TRACE_AUDIT_KIND_BY_TABLE.values()))
            ].reset_index(drop=True)
            if not trace.empty and "kind" in trace.columns
            else pd.DataFrame()
        )
    return V2CaptureResult(
        tables=tables,
        day_funnels=day_funnels,
        rebuilt_days=rebuilt_days,
        cached_artifact_days=cached_days,
        bars_by_day=bars_by_day,
        access_policy=policy,
        audit_frames=audit_frames if audit_capture_mode != "disabled" else None,
        end_seed=seed,
        trace_audit_rows=trace_audit_rows,
    )


class FsmAuditBuildResult:
    """One-replay FSM-audit build: regenerated v2 tables (parity-gated against
    the accepted final-review dataset) + the audit companion tables."""

    def __init__(
        self,
        *,
        audit_tables: dict,
        core_tables: dict[RecordTable, pd.DataFrame],
        parity_report: dict,
        reconciliation_report: dict,
        day_funnels: dict[str, dict[str, int]],
        rebuilt_days: list[str],
        cached_artifact_days: list[str],
        access_policy: ExplorationDataPolicy,
        capacity_report: dict,
    ) -> None:
        self.audit_tables = audit_tables
        self.core_tables = core_tables
        self.parity_report = parity_report
        self.reconciliation_report = reconciliation_report
        self.day_funnels = day_funnels
        self.rebuilt_days = rebuilt_days
        self.cached_artifact_days = cached_artifact_days
        self.access_policy = access_policy
        self.capacity_report = capacity_report


def _derive_parentless_intervals(
    steps: pd.DataFrame,
    deaths: pd.DataFrame,
    windows: pd.DataFrame,
    locks: pd.DataFrame,
) -> pd.DataFrame:
    """Group consecutive parentless step rows (reducer step ordinals are
    chain-continuous, and every counted bar emits exactly one row, so an
    interval is a maximal ordinal-contiguous run per setup)."""
    from .manifest import canonical_sha256

    if steps.empty:
        return pd.DataFrame()
    selected = (
        windows.loc[windows["event_kind"] == "parent_selected"]
        if not windows.empty
        else pd.DataFrame()
    )
    terminal = (
        deaths.loc[deaths["setup_terminated"].astype(bool)]
        if not deaths.empty
        else pd.DataFrame()
    )
    lock_setups = (
        set(locks["envelope_setup_id"].astype(str)) if not locks.empty else set()
    )
    rows: list[dict] = []
    ordered = steps.sort_values("stamp_source_step_ordinal", kind="mergesort")
    for setup_id, group in ordered.groupby("setup_id", sort=True):
        ordinals = group["stamp_source_step_ordinal"].astype(int).tolist()
        records = group.to_dict("records")
        segment_start = 0
        for index in range(1, len(ordinals) + 1):
            if index < len(ordinals) and ordinals[index] == ordinals[index - 1] + 1:
                continue
            first, last = records[segment_start], records[index - 1]
            last_ordinal = int(last["stamp_source_step_ordinal"])
            end_reason = "chain_end"
            successor = None
            if not selected.empty:
                match = selected.loc[
                    (selected["setup_id"].astype(str) == str(setup_id))
                    & (selected["stamp_source_step_ordinal"].astype(int) == last_ordinal + 1)
                ]
                if len(match):
                    end_reason = "successor_parent_selected"
                    successor = str(match.iloc[0]["parent_fvg_id"])
            terminal_reason = None
            if not terminal.empty:
                mine = terminal.loc[terminal["setup_id"].astype(str) == str(setup_id)]
                if len(mine):
                    terminal_reason = str(mine.iloc[0]["death_reason"])
                    death_ordinal = int(mine.iloc[0]["stamp_source_step_ordinal"])
                    if end_reason == "chain_end" and death_ordinal in (
                        last_ordinal,
                        last_ordinal + 1,
                    ):
                        end_reason = {
                            "expired_parent_search": "windows_expired",
                            "invalidated_htf_filled": "htf_filled",
                            "dataset_exhaustion_pre_entry": "dataset_exhausted",
                        }.get(terminal_reason, terminal_reason)
            first_day = str(first["envelope_trading_day"])
            last_day = str(last["envelope_trading_day"])
            rows.append(
                {
                    "interval_id": canonical_sha256(
                        {
                            "setup_id": str(setup_id),
                            "first_step_ordinal": int(first["stamp_source_step_ordinal"]),
                            "last_step_ordinal": last_ordinal,
                        }
                    ),
                    "setup_id": str(setup_id),
                    "first_counted_bar_id": first["bar_bar_id"],
                    "last_counted_bar_id": last["bar_bar_id"],
                    "first_counted_cursor": first["bar_cursor"],
                    "last_counted_cursor": last["bar_cursor"],
                    "first_step_ordinal": int(first["stamp_source_step_ordinal"]),
                    "last_step_ordinal": last_ordinal,
                    "bars_count": index - segment_start,
                    "start_ts_utc": first["envelope_ts_utc"],
                    "end_ts_utc": last["envelope_ts_utc"],
                    "end_reason": end_reason,
                    "open_timeframes_at_start": first["open_window_timeframes"],
                    "open_timeframes_at_end": last["open_window_timeframes"],
                    "successor_parent_fvg_id": successor,
                    "eventual_lock": str(setup_id) in lock_setups,
                    "eventual_terminal_reason": terminal_reason,
                    "first_source_date": first_day,
                    "last_source_date": last_day,
                    "crosses_day_boundary": first_day != last_day,
                    "audit_schema_version": 1,
                }
            )
            segment_start = index
    return pd.DataFrame(rows)


def assemble_fsm_audit_tables(
    *,
    trace: pd.DataFrame,
    audit_frames: list[pd.DataFrame],
    day_funnels: dict[str, dict[str, int]],
    chain_dates,
    warmup_days: int,
) -> dict:
    """Cut the shared core trace + audit-channel frames into the typed audit
    tables (validated + link-checked). Factored out of the builder so contract
    tests can drive it with synthetic chains."""
    from .audit_contracts import (
        CHANNEL_AUDIT_KIND_BY_TABLE,
        TRACE_AUDIT_KIND_BY_TABLE,
        AuditTable,
        validate_audit_links,
        validate_audit_table,
    )
    from .audit_contracts import (
        REQUIRED_COLUMNS as _AUDIT_REQUIRED,
    )

    def _trim_audit_frame(frame: pd.DataFrame, audit_table) -> pd.DataFrame:
        # Drop the irrelevant all-NA union columns but ALWAYS keep (or
        # recreate) the contract's required columns — nullable required
        # columns may legitimately be all-NA.
        required = _AUDIT_REQUIRED[audit_table]
        keep = [
            column
            for column in frame.columns
            if column in required or frame[column].notna().any()
        ]
        out = frame.loc[:, keep].copy()
        for column in required:
            if column not in out.columns:
                out[column] = pd.NA
        return out

    audit_tables: dict = {}
    for audit_table, kind in TRACE_AUDIT_KIND_BY_TABLE.items():
        frame = (
            trace.loc[trace["kind"] == kind].copy()
            if not trace.empty and "kind" in trace.columns
            else pd.DataFrame()
        )
        if not frame.empty:
            frame = frame.drop(columns=["kind"], errors="ignore")
            frame = _trim_audit_frame(frame, audit_table)
        audit_tables[audit_table] = frame.reset_index(drop=True)

    audit_all = (
        pd.concat(
            [frame for frame in audit_frames if not frame.empty],
            ignore_index=True,
            sort=False,
        )
        if any(not frame.empty for frame in audit_frames)
        else pd.DataFrame()
    )
    if not audit_all.empty:
        audit_all["audit_trace_ordinal"] = range(len(audit_all))
    for audit_table, kind in CHANNEL_AUDIT_KIND_BY_TABLE.items():
        frame = (
            audit_all.loc[audit_all["kind"] == kind].copy()
            if not audit_all.empty
            else pd.DataFrame()
        )
        if not frame.empty:
            frame = frame.drop(columns=["kind", "bar", "fvg"], errors="ignore")
            frame = _trim_audit_frame(frame, audit_table)
        audit_tables[audit_table] = frame.reset_index(drop=True)

    audit_tables[AuditTable.PARENTLESS_INTERVAL] = _derive_parentless_intervals(
        audit_tables[AuditTable.PARENTLESS_STEP],
        audit_tables[AuditTable.SLOT_DEATH],
        audit_tables[AuditTable.PARENT_WINDOW],
        audit_tables[AuditTable.PARENT_LOCK],
    )
    chain_order = list(chain_dates)
    funnel_rows = [
        {
            "source_date": date_str,
            "counter": counter,
            "value": int(value),
            "is_warmup": chain_order.index(date_str) < warmup_days,
            "audit_schema_version": 1,
        }
        for date_str, counters in sorted(day_funnels.items())
        for counter, value in sorted(counters.items())
    ]
    audit_tables[AuditTable.DAY_FUNNEL] = pd.DataFrame(funnel_rows)

    for audit_table in AuditTable:
        frame = audit_tables.get(audit_table, pd.DataFrame())
        if not frame.empty:
            validate_audit_table(audit_table, frame)
    validate_audit_links(audit_tables)
    return audit_tables


def build_ifvg_fsm_audit_v1(
    dates: list[str] | tuple[str, ...],
    cfg: IfvgCaptureConfig,
    resolved_profile: ResolvedProfileConfig,
    *,
    accepted_v2_exploration_dir,
    access_policy: ExplorationDataPolicy | None = None,
    cached_artifacts_only: bool = False,
    progress_fn=None,
) -> FsmAuditBuildResult:
    """One audit-enabled replay of the authorized chain.

    Clone of the :func:`build_ifvg_v2_capture` chain loop that additionally
    retains the FSM audit channel. HARD GATE: the 7 core tables regenerated by
    this very replay must content-hash-match the accepted final-review v2
    dataset (``FSM_AUDIT_ACCEPTED_V2_*`` pins) — any mismatch raises and no
    audit artifact may be saved.
    """
    from .audit_contracts import (
        CHANNEL_AUDIT_KIND_BY_TABLE,
        reconcile_funnel_to_audit,
    )
    from .config import (
        FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    )

    if cfg.identity_lane != "v2":
        raise PermissionError("fsm audit replay cannot use a legacy capture identity")
    if resolved_profile.section_config_hash != cfg.profile_hash:
        raise ValueError("resolved profile hash does not match the capture section")
    policy = access_policy or ExplorationDataPolicy()
    require_fixed_exploration_allowlist(policy)
    chain_dates = policy.authorize_dates(dates)
    if not chain_dates:
        raise ValueError("IFVG fsm audit replay requires at least one allowlisted date")

    cold = DaySeeds(prev_day=None, prev_full_hl=None, prev_ny_day=None, prev_ny_hl=None)
    seed: IfvgDaySeed | None = None
    previous_artifacts: DayArtifacts | None = None
    trace_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []
    core_day_offsets: list[int] = []
    day_funnels: dict[str, dict[str, int]] = {}
    rebuilt_days: list[str] = []
    cached_days: list[str] = []
    bars_by_day: dict[str, tuple] = {}
    core_offset = 0

    for chain_index, date_str in enumerate(chain_dates):
        expected_seeds = _chained_seeds(previous_artifacts) or cold
        artifacts = load_day_artifacts(
            date_str, cfg, expected_seeds=expected_seeds, access_policy=policy
        )
        if artifacts is None:
            if cached_artifacts_only:
                raise RuntimeError(
                    f"cached-artifacts-only fsm audit replay: {date_str} is "
                    "missing or failed seed trust"
                )
            artifacts = build_day_artifacts(
                date_str, cfg, expected_seeds, access_policy=policy
            )
            write_day_artifacts(artifacts, cfg, access_policy=policy)
            rebuilt_days.append(date_str)
        else:
            cached_days.append(date_str)

        day_result = capture_single_date(
            date_str,
            cfg,
            artifacts=artifacts,
            seed=seed,
            dataset_exhausted=chain_index == len(chain_dates) - 1,
            audit_capture_mode="fsm_audit_v1",
        )
        seed = day_result.end_seed
        frame = day_result.rows.copy()
        if not frame.empty:
            frame["is_warmup"] = chain_index < cfg.warmup_days
            frame["days_of_htf_history"] = chain_index
            frame["evaluation_config_hash"] = resolved_profile.evaluation_config_hash
        trace_frames.append(frame)
        core_day_offsets.append(core_offset)
        core_offset += len(frame)
        audit_frame = (
            day_result.audit_rows.copy()
            if day_result.audit_rows is not None
            else pd.DataFrame()
        )
        if not audit_frame.empty:
            audit_frame["is_warmup"] = chain_index < cfg.warmup_days
            audit_frame["days_of_htf_history"] = chain_index
            audit_frame["evaluation_config_hash"] = (
                resolved_profile.evaluation_config_hash
            )
            audit_frame["core_trace_ordinal_before_global"] = (
                audit_frame["stamp_core_trace_ordinal_before"].astype(int)
                + core_day_offsets[chain_index]
            )
            audit_frame["core_trace_ordinal_after_global"] = (
                audit_frame["stamp_core_trace_ordinal_after"].astype(int)
                + core_day_offsets[chain_index]
            )
        audit_frames.append(audit_frame)
        day_funnels[date_str] = day_result.funnel
        bars_by_day[date_str] = tuple(artifacts.bars)
        previous_artifacts = artifacts
        if progress_fn is not None:
            progress_fn(chain_index + 1, len(chain_dates), date_str)

    trace = (
        pd.concat(trace_frames, ignore_index=True, sort=False)
        if any(not frame.empty for frame in trace_frames)
        else pd.DataFrame()
    )
    if not trace.empty:
        trace["trace_ordinal"] = range(len(trace))
    tables = partition_capture_tables(trace)
    for table, frame in tuple(tables.items()):
        stamped = frame.copy()
        stamped["evaluation_config_hash"] = resolved_profile.evaluation_config_hash
        tables[table] = stamped
    labels = build_candidate_labels_from_tables(
        tables[RecordTable.ENTRY_CANDIDATE],
        bars_by_day=bars_by_day,
        tick_size=cfg.tick_size,
        resolved_profile=resolved_profile,
    )
    labels = stamp_table_contract(RecordTable.CANDIDATE_LABEL, labels)
    tables[RecordTable.CANDIDATE_LABEL] = labels
    for table, frame in tables.items():
        validate_primary_keys(table, frame)
        validate_table_identity(table, frame)
    validate_foreign_keys(tables)

    # HARD GATE: exact identity vs the accepted final-review v2 dataset.
    accepted = load_accepted_v2_tables(
        accepted_v2_exploration_dir,
        expected_dataset_id=FSM_AUDIT_ACCEPTED_V2_DATASET_ID,
        expected_manifest_payload_sha256=FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256,
    )
    parity_report = reconcile_v3_core_to_accepted_v2(tables, accepted)

    audit_tables = assemble_fsm_audit_tables(
        trace=trace,
        audit_frames=audit_frames,
        day_funnels=day_funnels,
        chain_dates=chain_dates,
        warmup_days=cfg.warmup_days,
    )
    reconciliation_report = reconcile_funnel_to_audit(day_funnels, audit_tables)

    audit_row_counts = {
        table.value: int(len(frame)) for table, frame in audit_tables.items()
    }
    channel_tables = list(CHANNEL_AUDIT_KIND_BY_TABLE)
    per_day_rows: dict[str, int] = {}
    for table in channel_tables:
        frame = audit_tables[table]
        if frame.empty:
            continue
        for day, count in frame["envelope_trading_day"].astype(str).value_counts().items():
            per_day_rows[day] = per_day_rows.get(day, 0) + int(count)
    capacity_report = {
        "rows_by_table": audit_row_counts,
        "total_audit_channel_rows": int(sum(audit_row_counts[t.value] for t in channel_tables)),
        "max_audit_rows_per_day": max(per_day_rows.values()) if per_day_rows else 0,
        "days": len(chain_dates),
    }

    policy.assert_zero_forbidden_access()
    return FsmAuditBuildResult(
        audit_tables=audit_tables,
        core_tables=tables,
        parity_report=parity_report,
        reconciliation_report=reconciliation_report,
        day_funnels=day_funnels,
        rebuilt_days=rebuilt_days,
        cached_artifact_days=cached_days,
        access_policy=policy,
        capacity_report=capacity_report,
    )


class V3CaptureResult:
    """One-replay generation-3 result; only ``context_tables`` are persistable."""

    def __init__(
        self,
        *,
        context_tables: dict[ContextRecordTable, pd.DataFrame],
        core_parity_tables: dict[RecordTable, pd.DataFrame],
        baseline_reconciliation: dict,
        day_funnels: dict[str, dict[str, int]],
        rebuilt_days: list[str],
        cached_artifact_days: list[str],
        bars_by_day: dict[str, tuple],
        access_policy: ExplorationDataPolicy,
        capacity_metrics: dict,
        performance_measurements: dict,
        diagnostic_timings: dict | None = None,
    ) -> None:
        self.context_tables = context_tables
        # Transient control output.  The v3 saver deliberately has no parameter
        # for these tables, preventing v2 evidence duplication.
        self.core_parity_tables = core_parity_tables
        self.baseline_reconciliation = baseline_reconciliation
        self.day_funnels = day_funnels
        self.rebuilt_days = rebuilt_days
        self.cached_artifact_days = cached_artifact_days
        self.bars_by_day = bars_by_day
        self.access_policy = access_policy
        self.capacity_metrics = capacity_metrics
        self.performance_measurements = performance_measurements
        self.diagnostic_timings = diagnostic_timings or {}

    @property
    def captures(self) -> pd.DataFrame:
        return self.context_tables[ContextRecordTable.CONTEXT_CAPTURE]

    @property
    def candidate_links(self) -> pd.DataFrame:
        return self.context_tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]

    @property
    def decision_links(self) -> pd.DataFrame:
        return self.context_tables[ContextRecordTable.DECISION_CONTEXT_LINK]

    @property
    def trade_links(self) -> pd.DataFrame:
        return self.context_tables[ContextRecordTable.TRADE_CONTEXT_LINK]


def load_accepted_v2_tables(
    exploration_dir,
    *,
    expected_dataset_id: str,
    expected_manifest_payload_sha256: str,
) -> dict[RecordTable, pd.DataFrame]:
    """Load the immutable accepted control after verifying both pinned identities."""

    from pathlib import Path

    root = Path(exploration_dir).resolve()
    if root.parent.name != expected_dataset_id:
        raise ValueError("accepted v2 path does not match the pinned dataset ID")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != expected_dataset_id:
        raise ValueError("accepted v2 manifest dataset ID mismatch")
    if manifest.get("manifest_payload_sha256") != expected_manifest_payload_sha256:
        raise ValueError("accepted v2 manifest payload hash mismatch")
    return {
        table: pd.read_parquet(root / f"{table.value}.parquet")
        for table in RecordTable
    }


def _canonical_cell(value):
    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def table_content_hash(table: RecordTable, frame: pd.DataFrame) -> str:
    """Order-insensitive, dtype-normalized content hash of one typed v2 table.

    Public per ``ifvg_prop_robust_config_search_v1`` R1 (promotion of the
    former private helper): the search lane uses it for gross trade-stream
    hashes and audit-neutrality table comparisons.
    """

    from .manifest import canonical_sha256

    key_by_table = {
        RecordTable.SETUP_LIFECYCLE: "lifecycle_event_id",
        RecordTable.ENTRY_CANDIDATE: "candidate_id",
        RecordTable.CANDIDATE_LABEL: "candidate_label_id",
        RecordTable.ELIGIBLE_DECISION: "decision_id",
        RecordTable.EXECUTED_TRADE: "trade_id",
        RecordTable.GEOMETRY_DOSSIER: "candidate_id",
        RecordTable.QUARANTINE: "quarantine_id",
    }
    columns = sorted(frame.columns)
    ordered = frame.loc[:, columns]
    key = key_by_table[table]
    if key in ordered and not ordered.empty:
        ordered = ordered.sort_values(key, kind="mergesort")
    rows = [
        {column: _canonical_cell(value) for column, value in row.items()}
        for row in ordered.to_dict("records")
    ]
    return canonical_sha256({"columns": columns, "rows": rows})


#: Backward-compatible private alias (pre-promotion callers).
_table_content_hash = table_content_hash


def reconcile_v3_core_to_accepted_v2(
    derived: dict[RecordTable, pd.DataFrame],
    accepted: dict[RecordTable, pd.DataFrame],
) -> dict:
    """Require table membership and serialized values to match the accepted v2 set."""

    table_reports: dict[str, dict] = {}
    for table in RecordTable:
        derived_frame = derived.get(table, pd.DataFrame())
        accepted_frame = accepted.get(table, pd.DataFrame())
        derived_hash = _table_content_hash(table, derived_frame)
        accepted_hash = _table_content_hash(table, accepted_frame)
        table_reports[table.value] = {
            "derived_rows": int(len(derived_frame)),
            "accepted_rows": int(len(accepted_frame)),
            "derived_content_sha256": derived_hash,
            "accepted_content_sha256": accepted_hash,
            "matched": derived_hash == accepted_hash,
        }
    passed = all(item["matched"] for item in table_reports.values())
    report = {"passed": passed, "tables": table_reports}
    if not passed:
        mismatches = [name for name, item in table_reports.items() if not item["matched"]]
        raise ValueError(
            "IFVG v3 core replay differs from accepted v2 tables: "
            f"{mismatches}"
        )
    return report


def _nearest_rank_p99_ms(values: tuple[int, ...]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, (99 * len(ordered) + 99) // 100 - 1))
    return ordered[index] / 1_000_000


def _performance_source_dates(dates: tuple[str, ...]) -> tuple[str, ...]:
    """Freeze the benchmark denominator to the original 26-date source."""

    return dates[:26]


def _timed_preloaded_replay(
    dates: tuple[str, ...],
    artifacts_by_day: dict[str, DayArtifacts],
    cfg: IfvgV3CaptureConfig,
    *,
    context_enabled: bool,
    context_source_coverage: ContextSourceCoverage,
    strategy_core_commit: str,
    strategy_core_source_tree_hash: str,
    context_replay_tape: ContextReplayTape | None = None,
) -> tuple[float, float, float]:
    core_seed: IfvgDaySeed | None = None
    context_seed = None
    observer_p99: list[float] = []
    multi_p99: list[float] = []
    started = perf_counter_ns()
    for index, day in enumerate(dates):
        artifacts = artifacts_by_day[day]
        bars_by_tf: dict[int, list] = {}
        for bar in artifacts.bars:
            bars_by_tf.setdefault(bar.timeframe_ticks, []).append(bar)
        result = run_day(
            bars_by_tf,
            section=cfg.core.section,
            seed=core_seed,
            trading_day=date.fromisoformat(day),
            tick_size=cfg.core.tick_size,
            levels_for=levels_for_from_frame(artifacts.level_timeline),
            dataset_exhausted=index == len(dates) - 1,
            context_config=cfg.context if context_enabled else None,
            context_seed=context_seed if context_enabled else None,
            context_source_coverage=context_source_coverage,
            context_symbol=cfg.core.symbol,
            strategy_core_commit=strategy_core_commit,
            strategy_core_source_tree_hash=strategy_core_source_tree_hash,
            context_replay_tape=(
                context_replay_tape if context_enabled else None
            ),
        )
        core_seed = result.end_seed
        if context_enabled:
            if not isinstance(result, IfvgContextDayResult):
                raise TypeError("context benchmark did not return context telemetry")
            context_seed = result.end_context_seed
            observer_p99.append(
                _nearest_rank_p99_ms(result.performance_trace.observer_step_ns)
            )
            multi_p99.append(
                _nearest_rank_p99_ms(
                    result.performance_trace.multi_timeframe_callback_ns
                )
            )
    elapsed = (perf_counter_ns() - started) / 1_000_000_000
    return (
        elapsed,
        max(observer_p99, default=0.0),
        max(multi_p99, default=0.0),
    )


def _performance_benchmark(
    dates: tuple[str, ...],
    artifacts_by_day: dict[str, DayArtifacts],
    cfg: IfvgV3CaptureConfig,
    *,
    context_source_coverage: ContextSourceCoverage,
    strategy_core_commit: str,
    strategy_core_source_tree_hash: str,
) -> dict[str, object]:
    context_replay_tape = ContextReplayTape()

    def execute(context_enabled: bool) -> tuple[float, float, float]:
        return _timed_preloaded_replay(
            dates,
            artifacts_by_day,
            cfg,
            context_enabled=context_enabled,
            context_source_coverage=context_source_coverage,
            strategy_core_commit=strategy_core_commit,
            strategy_core_source_tree_hash=strategy_core_source_tree_hash,
            context_replay_tape=(
                context_replay_tape if context_enabled else None
            ),
        )

    for warmup in range(2):
        order = (False, True) if warmup % 2 == 0 else (True, False)
        for enabled in order:
            execute(enabled)

    baseline_seconds: list[float] = []
    context_seconds: list[float] = []
    observer_p99: list[float] = []
    multi_p99: list[float] = []
    paired_slowdown: list[float] = []
    for repetition in range(10):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        pair: dict[bool, tuple[float, float, float]] = {}
        for enabled in order:
            pair[enabled] = execute(enabled)
        baseline_seconds.append(pair[False][0])
        context_seconds.append(pair[True][0])
        observer_p99.append(pair[True][1])
        multi_p99.append(pair[True][2])
        paired_slowdown.append(pair[True][0] / pair[False][0] - 1.0)

    def coefficient_of_variation(values: list[float]) -> float | None:
        mean = statistics.fmean(values)
        return statistics.pstdev(values) / mean if mean else None

    def quantile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, int(probability * len(ordered) + 0.999999999) - 1),
        )
        return ordered[index]

    return {
        "disabled_replay_seconds": statistics.median(baseline_seconds),
        "enabled_replay_seconds": statistics.median(context_seconds),
        "completed_1m_step_p99_ms": max(observer_p99),
        "multi_timeframe_callback_p99_ms": max(multi_p99),
        "repeated_run_p95_slowdown_fraction": quantile(paired_slowdown, 0.95),
        "measurement_policy": "preloaded_two_warmup_ten_alternating_pairs_v2",
        "warmup_pairs": 2,
        "measured_pairs": 10,
        "source_date_count": len(dates),
        "baseline_seconds": baseline_seconds,
        "context_seconds": context_seconds,
        "paired_slowdown_fractions": paired_slowdown,
        "baseline_p95_seconds": quantile(baseline_seconds, 0.95),
        "context_p95_seconds": quantile(context_seconds, 0.95),
        "observer_p99_ms_by_run": observer_p99,
        "multi_timeframe_p99_ms_by_run": multi_p99,
        "baseline_coefficient_of_variation": coefficient_of_variation(
            baseline_seconds
        ),
        "context_coefficient_of_variation": coefficient_of_variation(context_seconds),
        "machine": platform.platform(),
        "python_version": platform.python_version(),
    }


def build_ifvg_v3_capture(
    dates: list[str] | tuple[str, ...],
    cfg: IfvgV3CaptureConfig,
    resolved_profile: ResolvedProfileConfig,
    *,
    strategy_core_commit: str,
    strategy_core_source_tree_hash: str,
    access_policy: ExplorationDataPolicy | None = None,
    cached_artifacts_only: bool = False,
    accepted_v2_tables: dict[RecordTable, pd.DataFrame] | None = None,
    context_source_coverage: ContextSourceCoverage = DEFAULT_CONTEXT_SOURCE_COVERAGE,
    measure_performance: bool = False,
    progress_fn=None,
) -> V3CaptureResult:
    """Replay once into unchanged transient v2 controls plus normalized context."""

    core_cfg = cfg.core
    if resolved_profile.section_config_hash != core_cfg.profile_hash:
        raise ValueError("resolved profile hash does not match the v3 core section")
    policy = access_policy or ExplorationDataPolicy()
    require_fixed_exploration_allowlist(policy)
    chain_dates = policy.authorize_dates(dates)
    if not chain_dates:
        raise ValueError("IFVG v3 replay requires at least one allowlisted date")

    cold = DaySeeds(
        prev_day=None,
        prev_full_hl=None,
        prev_ny_day=None,
        prev_ny_hl=None,
    )
    core_seed: IfvgDaySeed | None = None
    context_seed = None
    previous_artifacts: DayArtifacts | None = None
    trace_frames: list[pd.DataFrame] = []
    context_days: list[ContextCaptureDayResult] = []
    day_funnels: dict[str, dict[str, int]] = {}
    rebuilt_days: list[str] = []
    cached_days: list[str] = []
    bars_by_day: dict[str, tuple] = {}
    artifacts_by_day: dict[str, DayArtifacts] = {}
    source_loading_ns: list[int] = []

    replay_started_ns = perf_counter_ns()
    for chain_index, date_str in enumerate(chain_dates):
        expected_seeds = _chained_seeds(previous_artifacts) or cold
        source_load_started_ns = perf_counter_ns()
        artifacts = load_day_artifacts(
            date_str,
            core_cfg,
            expected_seeds=expected_seeds,
            access_policy=policy,
        )
        source_loading_ns.append(perf_counter_ns() - source_load_started_ns)
        if artifacts is None:
            if cached_artifacts_only:
                raise RuntimeError(
                    f"cached-artifacts-only v3 replay: {date_str} is missing "
                    "or failed seed trust"
                )
            artifacts = build_day_artifacts(
                date_str,
                core_cfg,
                expected_seeds,
                access_policy=policy,
            )
            write_day_artifacts(artifacts, core_cfg, access_policy=policy)
            rebuilt_days.append(date_str)
        else:
            cached_days.append(date_str)

        day_result = capture_single_date_with_context(
            date_str,
            core_cfg,
            artifacts=artifacts,
            seed=core_seed,
            context_config=cfg.context,
            context_seed=context_seed,
            strategy_core_commit=strategy_core_commit,
            strategy_core_source_tree_hash=strategy_core_source_tree_hash,
            context_source_coverage=context_source_coverage,
            dataset_exhausted=chain_index == len(chain_dates) - 1,
        )
        core_seed = day_result.end_seed
        context_seed = day_result.end_context_seed
        frame = day_result.core_rows.copy()
        if not frame.empty:
            frame["is_warmup"] = chain_index < core_cfg.warmup_days
            frame["days_of_htf_history"] = chain_index
            frame["evaluation_config_hash"] = resolved_profile.evaluation_config_hash
        trace_frames.append(frame)
        context_days.append(day_result)
        day_funnels[date_str] = day_result.funnel
        bars_by_day[date_str] = tuple(artifacts.bars)
        artifacts_by_day[date_str] = artifacts
        previous_artifacts = artifacts
        if progress_fn is not None:
            progress_fn(chain_index + 1, len(chain_dates), date_str)
    trace = (
        pd.concat(trace_frames, ignore_index=True, sort=False)
        if any(not frame.empty for frame in trace_frames)
        else pd.DataFrame()
    )
    if not trace.empty:
        trace["trace_ordinal"] = range(len(trace))
    core_tables = partition_capture_tables(trace)
    for table, frame in tuple(core_tables.items()):
        stamped = frame.copy()
        stamped["evaluation_config_hash"] = resolved_profile.evaluation_config_hash
        core_tables[table] = stamped
    core_tables[RecordTable.CANDIDATE_LABEL] = stamp_table_contract(
        RecordTable.CANDIDATE_LABEL,
        build_candidate_labels_from_tables(
            core_tables[RecordTable.ENTRY_CANDIDATE],
            bars_by_day=bars_by_day,
            tick_size=core_cfg.tick_size,
            resolved_profile=resolved_profile,
        ),
    )
    for table, frame in core_tables.items():
        validate_primary_keys(table, frame)
        validate_table_identity(table, frame)
    validate_foreign_keys(core_tables)

    accepted = accepted_v2_tables or load_accepted_v2_tables(
        cfg.accepted_v2_exploration_dir,
        expected_dataset_id=cfg.accepted_v2_dataset_id,
        expected_manifest_payload_sha256=cfg.accepted_v2_manifest_sha256,
    )
    baseline = reconcile_v3_core_to_accepted_v2(core_tables, accepted)
    # End the fixed-source replay clock after the shared v2 core table, label,
    # and accepted-baseline parity work. Context-only normalization below is not
    # part of the disabled control and is deliberately excluded.
    replay_elapsed_ns = perf_counter_ns() - replay_started_ns
    normalization_started_ns = perf_counter_ns()
    context_tables = normalize_context_days(
        context_days,
        warmup_days=core_cfg.warmup_days,
    )
    normalization_ns = perf_counter_ns() - normalization_started_ns
    for table, frame in context_tables.items():
        validate_context_primary_keys(table, frame)
        validate_context_table_identity(table, frame)
    validate_context_foreign_keys(context_tables, core_tables=core_tables)
    policy.assert_zero_forbidden_access()
    if context_seed is None:
        raise RuntimeError("IFVG v3 replay ended without a context seed")
    terminal_seed_bytes = len(canonical_json(context_seed).encode("utf-8"))
    max_transition_bytes = max(
        (
            event.serialized_size()
            for day in context_days
            for event in day.context_events
        ),
        default=0,
    )

    observer_ns = [
        value
        for day in context_days
        for value in day.performance_trace.observer_step_ns
    ]
    multi_ns = [
        value
        for day in context_days
        for value in day.performance_trace.multi_timeframe_callback_ns
    ]
    advance_ns = [
        value
        for day in context_days
        for value in day.performance_trace.observer_advance_ns
    ]
    capture_ns = [
        value
        for day in context_days
        for value in day.performance_trace.event_capture_ns
    ]
    seed_restore_ns = [
        value
        for day in context_days
        for value in day.performance_trace.seed_restore_ns
    ]
    seed_snapshot_ns = [
        value
        for day in context_days
        for value in day.performance_trace.seed_snapshot_ns
    ]

    enabled_ns = replay_elapsed_ns
    disabled_estimate_ns = max(0, enabled_ns - sum(observer_ns))
    if measure_performance:
        performance_dates = _performance_source_dates(tuple(chain_dates))
        performance_measurements = _performance_benchmark(
            performance_dates,
            {day: artifacts_by_day[day] for day in performance_dates},
            cfg,
            context_source_coverage=context_source_coverage,
            strategy_core_commit=strategy_core_commit,
            strategy_core_source_tree_hash=strategy_core_source_tree_hash,
        )
    else:
        performance_measurements = {
            "disabled_replay_seconds": disabled_estimate_ns / 1_000_000_000,
            "enabled_replay_seconds": enabled_ns / 1_000_000_000,
            "completed_1m_step_p99_ms": _nearest_rank_p99_ms(tuple(observer_ns)),
            "multi_timeframe_callback_p99_ms": _nearest_rank_p99_ms(tuple(multi_ns)),
            "repeated_run_p95_slowdown_fraction": None,
            "measurement_policy": "diagnostic_single_pass_observer_subtraction_v1",
        }
    return V3CaptureResult(
        context_tables=context_tables,
        core_parity_tables=core_tables,
        baseline_reconciliation=baseline,
        day_funnels=day_funnels,
        rebuilt_days=rebuilt_days,
        cached_artifact_days=cached_days,
        bars_by_day=bars_by_day,
        access_policy=policy,
        capacity_metrics={
            "terminal_state_bytes": terminal_seed_bytes,
            "terminal_seed_bytes": terminal_seed_bytes,
            "max_transition_bytes": max_transition_bytes,
        },
        performance_measurements=performance_measurements,
        diagnostic_timings={
            "scope": "single_capture_pass_excluded_from_slowdown_denominator_v1",
            "source_loading_seconds": sum(source_loading_ns) / 1_000_000_000,
            "normalization_seconds": normalization_ns / 1_000_000_000,
            "observer_advance_seconds": sum(advance_ns) / 1_000_000_000,
            "event_capture_seconds": sum(capture_ns) / 1_000_000_000,
            "seed_restore_seconds": sum(seed_restore_ns) / 1_000_000_000,
            "seed_snapshot_seconds": sum(seed_snapshot_ns) / 1_000_000_000,
            "source_loading_calls": len(source_loading_ns),
            "observer_advance_calls": len(advance_ns),
            "event_capture_calls": len(capture_ns),
            "seed_restore_calls": len(seed_restore_ns),
            "seed_snapshot_calls": len(seed_snapshot_ns),
        },
    )
