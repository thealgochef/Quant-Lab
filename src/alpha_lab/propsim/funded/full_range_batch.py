"""Exact six-row IFSM batch over Task B's registered inputs.

This adapter changes orchestration and observational reporting only. The existing
Core driver, pair reducer, account ledger and ordered-print walker own decisions
and money. Plans/approvals/results use the established immutable comparison stores.
Each configuration checkpoints its three independent streams after every day.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.agents.data_infra.ifvg.search.task_b import TaskBExecutionScope
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, ProcessingClockPolicy, to_ns
from alpha_lab.propsim.funded.comparison_plan import (
    PLAN_STORE,
    RESULT_STORE,
    CoreSourceRef,
    ExecutionModelRef,
    FundedComparisonResultEnvelope,
    FundedComparisonResultPayload,
)
from alpha_lab.propsim.funded.comparison_run import PrintStats, pair_id_for
from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_one_day
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.plan import OwnerDecisionRef
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS, FundedFirmProfile
from alpha_lab.propsim.funded.result import canonical_json, result_sha256
from alpha_lab.propsim.funded.runner import append_ledger, read_state, write_state
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

ENGINE_VERSION = "ifsm_correct_config_full_range_batch_v1"
PLAN_SCHEMA = "ifsm_correct_config_full_range_plan_v1"
ANNOTATION = {
    "menthorq_context_version": "menthorq_eod_v1",
    "regime_gate_policy": "off",
    "nearest_support_gex1_block": False,
    "regime_unknown_policy": "allow",
    "nearest_support_universe": "all_19",
}
CONFIGURATION_NAMES = (
    "S1-T1-H14-P1-L-SO",
    "S0-T1-H14-P1-L-SO",
    "S0_D80_W1_P1",
    "S1_D80_W1_P1",
    "S0_D160_W1_P1",
    "S1_D160_W1_P1",
)
REPO_ROOT = Path(__file__).resolve().parents[4]


class FullRangeSourceRef(FrozenContract):
    kind: Literal["verified_task_b_registered_inputs"] = "verified_task_b_registered_inputs"
    task_b_plan_id: str = Field(pattern=SHA256_PATTERN)
    task_b_plan_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    task_b_store_namespace_id: str = Field(pattern=SHA256_PATTERN)
    title: str
    warmup_dates: tuple[str, ...]
    evaluation_dates: tuple[str, ...]
    cutoff_utc: str


class FullRangeConfiguration(FrozenContract):
    batch_id: str
    name: str
    display_name: str
    historical_section_json: str
    historical_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    effective_section_json: str
    effective_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    instrument: Literal["mini", "micro"]
    quantity: int = Field(gt=0)
    cost_per_contract_mills: int = Field(ge=0)
    axis_value_ids: tuple[tuple[str, str], ...] = ()

    @property
    def resolved_section_config_hash(self) -> str:
        return self.effective_section_config_hash

    @property
    def exit_policy(self) -> str:
        return json.loads(self.effective_section_json)["exit_policy"]


class FullRangeBatchPlanPayload(FrozenContract):
    plan_schema: Literal["ifsm_correct_config_full_range_plan_v1"] = PLAN_SCHEMA
    mode: Literal["single_account_configuration_comparison"] = (
        "single_account_configuration_comparison"
    )
    purpose: Literal["historical_comparison"] = "historical_comparison"
    question: str
    source: FullRangeSourceRef
    configurations: tuple[FullRangeConfiguration, ...] = Field(min_length=6, max_length=6)
    task_b_store_root: str
    task_b_plan_id: str = Field(pattern=SHA256_PATTERN)
    task_b_scope: TaskBExecutionScope
    prepared_registration_ids: ImmutableMap[str, str]
    core_root: str
    core_source: CoreSourceRef
    runtime_source_file_sha256: ImmutableMap[str, str]
    request_file: str
    request_file_sha256: str = Field(pattern=SHA256_PATTERN)
    owner_evidence_file_sha256: ImmutableMap[str, str]
    calendar_sha256: str = Field(pattern=SHA256_PATTERN)
    firm_profiles: tuple[FundedFirmProfile, ...] = tuple(FIRM_PROFILES.values())
    processing: ProcessingClockPolicy = TWO_BUSINESS_DAYS_FED_1600
    execution_model: ExecutionModelRef = ExecutionModelRef()
    owner_decisions: tuple[OwnerDecisionRef, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _exact(self):
        if tuple(row.name for row in self.configurations) != CONFIGURATION_NAMES:
            raise ValueError("the batch contains exactly C01-C06 in the requested order")
        if tuple(row.batch_id for row in self.configurations) != tuple(
            f"C{index:02d}" for index in range(1, 7)
        ):
            raise ValueError("the batch row IDs must be C01-C06")
        if self.source.task_b_plan_id != self.task_b_plan_id:
            raise ValueError("source and input plan identities disagree")
        if tuple(self.firm_profiles) != tuple(FIRM_PROFILES.values()):
            raise ValueError("the batch preserves both saved firm profiles")
        if canonical_contract_sha256(self.processing) != canonical_contract_sha256(
            TWO_BUSINESS_DAYS_FED_1600
        ):
            raise ValueError("the batch preserves the saved processing calendar")
        dates = self.source.warmup_dates + self.source.evaluation_dates
        if (
            len(self.source.warmup_dates) != 10
            or len(self.source.evaluation_dates) != 253
            or dates != tuple(sorted(set(dates)))
            or dates[0] != "2025-06-02"
            or dates[10] != "2025-06-16"
            or dates[-1] != "2026-06-10"
            or self.source.cutoff_utc != "2026-06-10T21:00:00Z"
        ):
            raise ValueError("the batch requires Task B's complete 253+10 date scope")
        from alpha_lab.agents.data_infra.ifvg.prepared_store import TASK_B_WARMUP_DATES

        if self.source.warmup_dates != TASK_B_WARMUP_DATES:
            raise ValueError("the batch warmup membership differs from Task B")
        for index, row in enumerate(self.configurations):
            expected = ("micro", 10, 514) if index < 2 else ("mini", 1, 5140)
            if (row.instrument, row.quantity, row.cost_per_contract_mills) != expected:
                raise ValueError("the batch size or exact per-contract costs changed")
        return self


class FullRangeBatchPlanEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_plan_id"
    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: FullRangeBatchPlanPayload


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def resolve_effective_section(mapping: dict) -> tuple[dict, str]:
    """Strict current-schema validation and canonical naming, with no dropped fields."""
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    from alpha_lab.agents.data_infra.ifvg.search.identities import canonicalize_section

    section = canonicalize_section(IfvgSmcSection.model_validate(mapping))
    return section.model_dump(mode="json"), ifvg_profile_hash(section)


def batch_calendar_sha256(section, evaluation_dates) -> str:
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar

    if isinstance(section, str):
        section = IfvgSmcSection.model_validate_json(section)
    elif isinstance(section, dict):
        section = IfvgSmcSection.model_validate(section)

    schedule, _start, _cutoff = _calendar(section, tuple(evaluation_dates))
    return canonical_contract_sha256([asdict(day) for day in schedule])


def _assert_request(plan, sections, request):
    if (
        request["strategy_configurations"] != 6
        or request["funded_result_rows"] != 12
        or request["batch_count"] != 1
        or request["configs"][0]["batch_id"] != "C01"
    ):
        raise PermissionError("the frozen request no longer specifies the one six-row batch")
    for row, section, expected in zip(
        plan.configurations, sections, request["configs"], strict=True
    ):
        values = section.model_dump(mode="json")
        if row.name != expected["saved_configuration_name"] or row.batch_id != expected["batch_id"]:
            raise PermissionError("worker configuration membership differs from the request")
        for field, value in request["common_expected_effective_values"].items():
            if values.get(field) != value or field not in values:
                raise PermissionError(f"{row.batch_id}: worker field {field} differs")
        for field, value in ANNOTATION.items():
            if values.get(field) != value:
                raise PermissionError(f"{row.batch_id}: passive annotation field {field} differs")
        for field in (
            "entry_schedule_policy",
            "opposing_parent_distance_ticks_max",
            "exit_policy",
            "tp_r_multiple",
        ):
            if values[field] != expected[field]:
                raise PermissionError(f"{row.batch_id}: worker field {field} differs")
        if row.historical_section_config_hash != expected["reference"]["historical_section_hash"]:
            raise PermissionError("historical reference identity differs from the request")
        if (
            row.instrument != expected["instrument"]
            or row.quantity != expected["quantity"]
            or row.cost_per_contract_mills / 1000 != expected["cost_per_contract_per_fill_usd"]
        ):
            raise PermissionError("worker size or fees differ from the request")


def validate_full_range_plan(plan: FullRangeBatchPlanPayload, *, verify_inputs: bool = True):
    """Verify runtime, exact worker payload, old input authority and named receipts.

    Returns sections/config/policy/calendar/start/cutoff. No candle or raw-market
    observations are decoded here; optional metadata verification hashes named caches.
    """
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
    from alpha_lab.agents.data_infra.ifvg.prepared_store import (
        PreparedStoreReplayPolicy,
        load_prepared_store,
    )
    from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import require_store_namespace
    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
    from alpha_lab.propsim.funded.core_identity import core_source_identity

    actual = core_source_identity()
    if (
        Path(actual["root"]).resolve() != Path(plan.core_root).resolve()
        or actual["base_commit"] != plan.core_source.base_commit
        or actual["patch_sha256"] != plan.core_source.patch_sha256
    ):
        raise PermissionError("the worker imported a different frozen compatible Core")
    for path, expected in (
        *plan.runtime_source_file_sha256.items(),
        *plan.owner_evidence_file_sha256.items(),
        (plan.request_file, plan.request_file_sha256),
    ):
        if _sha(path) != expected:
            raise PermissionError(f"frozen source/owner evidence changed: {path}")
    sections = tuple(
        IfvgSmcSection.model_validate_json(row.effective_section_json)
        for row in plan.configurations
    )
    for row, section in zip(plan.configurations, sections, strict=True):
        if (
            section.model_dump(mode="json") != json.loads(row.effective_section_json)
            or ifvg_profile_hash(section) != row.effective_section_config_hash
        ):
            raise PermissionError(f"{row.batch_id}: full worker payload or identity changed")
        original = json.loads(row.historical_section_json)
        from alpha_lab.agents.data_infra.ifvg.profiles import saved_section_matches_profile_hash

        if not saved_section_matches_profile_hash(original, row.historical_section_config_hash):
            raise PermissionError(f"{row.batch_id}: preserved historical section does not verify")
        # Names and annotation are allowed translations; all other saved values survive.
        for field, value in original.items():
            if (
                field not in ("profile_name", *ANNOTATION)
                and section.model_dump(mode="json").get(field) != value
            ):
                raise PermissionError(f"{row.batch_id}: historical semantic field {field} changed")
    request = json.loads(Path(plan.request_file).read_text(encoding="utf-8"))
    _assert_request(plan, sections, request)
    namespace = require_store_namespace(Path(plan.task_b_store_root), expected_class="research")
    if namespace.store_namespace_id != plan.source.task_b_store_namespace_id:
        raise PermissionError("Task B input namespace differs from the frozen plan")
    charter = load_verified_envelope(
        Path(plan.task_b_store_root), "charters", plan.task_b_plan_id, SearchCharterEnvelope
    )
    manifest = Path(plan.task_b_store_root) / "charters" / plan.task_b_plan_id / "manifest.json"
    if _sha(manifest) != plan.source.task_b_plan_manifest_sha256:
        raise PermissionError("Task B source charter manifest changed")
    if (
        charter.payload.task_b_execution != plan.task_b_scope
        or tuple(charter.payload.date_policy.replay_dates)
        != plan.source.warmup_dates + plan.source.evaluation_dates
    ):
        raise PermissionError("registered inputs or date membership differ from completed Task B")
    paths = tuple(Path(path) for path in plan.task_b_scope.prepared_store_registry_paths)
    registrations = tuple(load_prepared_store(path) for path in paths)
    ids = {str(reg.path): reg.registration_id for reg in registrations}
    if ids != dict(plan.prepared_registration_ids):
        raise PermissionError("prepared store registrations changed")
    catalogs = {
        str(Path(reg.definition["catalog_path"]).resolve()): _sha(reg.definition["catalog_path"])
        for reg in registrations
    }
    if catalogs != dict(plan.task_b_scope.preparation_catalog_sha256):
        raise PermissionError("source-selected contract catalog changed")
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    policy = PreparedStoreReplayPolicy(dates, registry_paths=paths)
    raw_roots = {str(Path(reg.definition["raw_data_dir"]).resolve()) for reg in registrations}
    if len(raw_roots) != 1:
        raise PermissionError("registered input segments use different raw source roots")
    cfg = IfvgCaptureConfig(
        section=sections[0],
        data_dir=Path(next(iter(raw_roots))),
        prepared_store_registry_paths=paths,
        preparation_catalog_paths=tuple(Path(path) for path in catalogs),
    )
    calendars = [_calendar(section, plan.source.evaluation_dates) for section in sections]
    if any(value != calendars[0] for value in calendars[1:]):
        raise PermissionError("the six workers disagree on the existing closing calendar")
    schedule, start_ns, cutoff_ns = calendars[0]
    if canonical_contract_sha256(
        [asdict(day) for day in schedule]
    ) != plan.calendar_sha256 or cutoff_ns != to_ns(plan.source.cutoff_utc):
        raise PermissionError("the closing calendar or exact cutoff changed")
    if verify_inputs:
        policy.verify_registered_metadata(cfg)
    policy.assert_zero_forbidden_access()
    return sections, cfg, policy, schedule, start_ns, cutoff_ns


def _plain(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def entry_context(provider, section, bar, signal, tick_size):
    """Observe the actual entry of this stream; this function returns no admission input."""
    from strategy_core.decisions.sessions import classify_session
    from strategy_core.strategies.ifvg_smc.menthorq_levels import (
        derive_menthorq_values,
        evaluate_menthorq_entry_gates,
    )
    from strategy_core.strategies.ifvg_smc.replay import _runtime_scheme

    ts = bar.availability_ts_utc
    snapshot = provider.snapshot(ts)
    price = signal.entry_ticks * tick_size
    derived = derive_menthorq_values(
        snapshot,
        price,
        ts,
        bar_open_points=bar.open_ticks * tick_size,
        prior_cash_close_points=provider.prior_cash_close_for(ts),
        nearest_support_universe=section.nearest_support_universe,
    )
    gates = evaluate_menthorq_entry_gates(
        snapshot,
        price,
        ts,
        regime_gate_policy=section.regime_gate_policy,
        regime_unknown_policy=section.regime_unknown_policy,
        nearest_support_gex1_block=section.nearest_support_gex1_block,
        nearest_support_universe=section.nearest_support_universe,
        enable_shorts=section.enable_shorts,
    )
    scheme = _runtime_scheme(section.session_scheme)
    out = {
        **{field.name: getattr(snapshot, field.name) for field in fields(snapshot)},
        **{field.name: getattr(derived, field.name) for field in fields(derived)},
        "entry_session": classify_session(ts, scheme).session,
        "context_asof_ts": ts.isoformat(),
        "availability_ts_utc": ts.isoformat(),
        "entry_price_points": price,
        "nearest_support_universe": section.nearest_support_universe,
        "gate_status": gates.gate_status,
        "regime_gate_blocked": gates.regime_gate_blocked,
        "nearest_support_gate_blocked": gates.nearest_support_gate_blocked,
        "availability_window_status": (
            "available"
            if snapshot.context_available
            else "outside_availability"
            if snapshot.unavailable_reason in {"before_0600", "after_1700"}
            else "missing_context"
        ),
    }
    return _plain(out)


class ObservedCoreDriver(CoreStrategyDriver):
    """Core driver with entry-context and existing funnel observation only."""

    def __init__(self, section, *, tick_size, menthorq_provider):
        super().__init__(section, tick_size=tick_size, menthorq_provider=menthorq_provider)
        self.entry_contexts = {}
        self.day_entries = 0
        self.day_refused = 0
        self.day_open_position = False
        self.last_day_evidence = {}
        self.core_exit_annotations = {}

    def begin_day(self, bars_by_tf, levels_for):
        self.day_entries = self.day_refused = 0
        self.day_open_position = False
        return super().begin_day(bars_by_tf, levels_for)

    def step(self, bar, gate):
        out = super().step(bar, gate)
        self._observe_exits(out.core_exits)
        self.day_refused += len(out.refused)
        self.day_open_position = self.day_open_position or self.in_position
        if out.entry is not None:
            self.day_entries += 1
            self.entry_contexts[out.entry.trade_id] = entry_context(
                self.menthorq_provider, self.section, bar, out.entry, self.tick_size
            )
        return out

    def end_day(self, trading_day, *, dataset_exhausted=False):
        self.last_day_evidence = {
            "entries": self.day_entries,
            "refused_entries": self.day_refused,
            "open_position_status": "observed_open" if self.day_open_position else "flat_all_day",
            "open_position_at_day_end": self.in_position,
            "stage_counters": self._orch._reducer.funnel_counters(),
        }
        records = super().end_day(trading_day, dataset_exhausted=dataset_exhausted)
        self._observe_exits(records)
        return records

    def _observe_exits(self, records):
        for record in records:
            self.core_exit_annotations[str(record.trade_id)] = {
                "scale_out_ts_utc": _plain(getattr(record, "scale_out_ts_utc", None)),
                "scale_out_fraction": getattr(record, "scale_out_fraction", None),
                "exit_policy": getattr(record, "exit_policy", self.section.exit_policy),
            }

    def checkpoint(self):
        return {
            **super().checkpoint(),
            "entry_contexts": self.entry_contexts,
            "breakeven_cleared": getattr(self, "breakeven_cleared", 0),
            "core_exit_annotations": self.core_exit_annotations,
        }

    def restore(self, state):
        super().restore(state)
        self.entry_contexts = state["entry_contexts"]
        self.breakeven_cleared = state.get("breakeven_cleared", 0)
        self.core_exit_annotations = state.get("core_exit_annotations", {})


def _provider_state(provider):
    return {
        "cash_closes": {day.isoformat(): value for day, value in provider._cash_closes.items()},
        "selected_instruments": {
            day.isoformat(): value for day, value in provider._selected_instruments.items()
        },
    }


def _restore_provider(provider, state):
    provider._cash_closes = {
        date.fromisoformat(day): value for day, value in state["cash_closes"].items()
    }
    provider._selected_instruments = {
        date.fromisoformat(day): value for day, value in state["selected_instruments"].items()
    }


def save_checkpoint(path: Path, payload: dict):
    """Atomic latest-day checkpoint, bound to its exact bytes and frozen dispatch."""
    body = canonical_json(payload)
    envelope = canonical_json(
        {"payload": payload, "sha256": hashlib.sha256(body.encode()).hexdigest()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(envelope)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path, *, dispatch_sha256: str, dates: tuple[str, ...]):
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if hashlib.sha256(canonical_json(payload).encode()).hexdigest() != envelope["sha256"]:
        raise PermissionError("batch checkpoint bytes failed verification")
    done = tuple(payload["completed_dates"])
    if payload["dispatch_sha256"] != dispatch_sha256 or done != dates[: len(done)]:
        raise PermissionError("batch checkpoint belongs to a different plan, worker or date prefix")
    return payload


def _dispatch(plan_id, approval_id, plan, row):
    return {
        "plan_id": plan_id,
        "approval_id": approval_id,
        "worker": row.model_dump(mode="json"),
        "source": plan.source.model_dump(mode="json"),
        "core_root": plan.core_root,
        "core_source": plan.core_source.model_dump(mode="json"),
        "runtime_source_file_sha256": dict(plan.runtime_source_file_sha256),
        "scope": plan.task_b_scope.model_dump(mode="json"),
        "engine_version": ENGINE_VERSION,
    }


def _roundtrip(value):
    return json.loads(canonical_json(value))


def _assert_terminal_result_binding(saved, envelope, *, plan_id, approval_id, plan):
    """A mutable job-state pointer cannot substitute another valid saved result."""
    payload = envelope.payload
    batch = saved["full_range_batch"]
    if (
        payload.funded_comparison_plan_id != plan_id
        or payload.funded_comparison_approval_id != approval_id
        or payload.engine_version != ENGINE_VERSION
        or saved["funded_comparison_plan_id"] != plan_id
        or batch["plan"] != plan.model_dump(mode="json")
    ):
        raise PermissionError("terminal result belongs to a different frozen plan or approval")
    expected = {row.batch_id: row for row in plan.configurations}
    completed = []
    for dispatch in batch["worker_dispatches"]:
        batch_id = dispatch["worker"]["batch_id"]
        if batch_id not in expected or dispatch != _dispatch(
            plan_id, approval_id, plan, expected[batch_id]
        ):
            raise PermissionError("terminal result worker dispatch differs from the frozen plan")
        completed.append(batch_id)
    failed = []
    for failure in batch["failed_configurations"]:
        batch_id = failure["batch_id"]
        if (
            batch_id not in expected
            or failure["configuration"] != expected[batch_id].name
            or failure["display_name"] != expected[batch_id].display_name
            or not failure["reason"]
        ):
            raise PermissionError("terminal failure row differs from the frozen plan")
        failed.append(batch_id)
    membership = completed + failed
    if len(membership) != 6 or set(membership) != set(expected):
        raise PermissionError("terminal result does not retain exactly the six dispatched rows")
    statuses = {
        (row.name, firm.firm_key): "Completed" if row.batch_id in completed else "Not completed"
        for row in plan.configurations
        for firm in plan.firm_profiles
    }
    rows = saved["tables"]["pair_results"]
    if (
        len(rows) != 12
        or {(row["configuration"], row["firm_key"]): row["status"] for row in rows} != statuses
    ):
        raise PermissionError("terminal result does not retain the twelve approved funded rows")
    return "Completed" if not failed and payload.validation_passed else "Incomplete"


def _verify_saved_print_receipts(cfg, policy, receipts):
    """Guard named source paths before reading any saved execution receipt."""
    from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _PrintSourceReceipts

    allowed = frozenset(
        day for registration in policy.registrations for day in registration.source_dates
    )
    observed = _PrintSourceReceipts()
    for item in receipts:
        path = Path(item["path"]).resolve()
        if (
            path.parent.name not in allowed
            or path.parent.name > "2026-06-10"
            or path.parent.parent != (cfg.data_dir / cfg.symbol).resolve()
            or path.name not in {"mbp10.parquet", "mbp1.parquet", "trades.parquet"}
            or path.parent.name != item["physical_date"]
            or path.name != item["filename"]
        ):
            raise PermissionError("saved print receipt escaped registered named input paths")
        observed.observe(path)
        if observed.files[str(path)] != item:
            raise PermissionError("funded print receipt changed since the saved execution")
    return observed


def run_stream_days(
    runs,
    providers,
    days,
    prints_factory,
    *,
    checkpoint_path,
    dispatch_sha256,
    all_dates,
    daily_activity,
    coverage_for_day,
    stats_state,
    on_day=None,
):
    """Streaming execution seam shared by production and deterministic fixture tests."""
    completed = list(all_dates[: stats_state.get("completed_count", 0)])
    for day in days:
        index = len(completed)
        if day.trading_day != all_dates[index]:
            raise PermissionError("worker decoded a day outside the frozen chronological order")
        prints = prints_factory(day)
        for run in runs:
            before_cash = len(run.ledger.payout_events) if run.ledger is not None else 0
            before_strategy = len(run.strategy_trades)
            before_funded = len(run.ledger.trades) if run.ledger is not None else 0
            run_one_day(run, day, prints)
            for trade in run.strategy_trades[before_strategy:]:
                trade.update(run.driver.core_exit_annotations.get(trade["trade_id"], {}))
                context = run.driver.entry_contexts.get(trade["trade_id"])
                if context is not None:
                    trade["entry_context"] = context
            if run.ledger is not None:
                for trade in run.ledger.trades[before_funded:]:
                    context = run.driver.entry_contexts.get(trade["strategy_trade_id"])
                    if context is not None:
                        trade["entry_context"] = context
            if day.is_evaluation:
                evidence = dict(run.driver.last_day_evidence)
                evidence.update(coverage_for_day(day.trading_day))
                evidence.update(
                    {
                        "stream": "strategy" if run.ledger is None else "funded",
                        "firm_key": None if run.ledger is None else run.ledger.profile.firm_key,
                        "evaluation_date": day.trading_day,
                        "processing_status": "not_applicable"
                        if run.ledger is None
                        else run.ledger.current.status,
                        "account_status_at_end": None
                        if run.ledger is None
                        else run.ledger.current.status,
                        "payout_events_this_day": 0
                        if run.ledger is None
                        else len(run.ledger.payout_events) - before_cash,
                    }
                )
                daily_activity.append(evidence)
        release = getattr(prints, "release", None)
        if release is not None:
            release()
        completed.append(day.trading_day)
        stats_state["completed_count"] = len(completed)
        payload = {
            "schema": "ifsm_full_range_day_checkpoint_v1",
            "dispatch_sha256": dispatch_sha256,
            "completed_dates": completed,
            "runs": [_roundtrip(run.to_state()) for run in runs],
            "providers": [_provider_state(provider) for provider in providers],
            "daily_activity": daily_activity,
            "stats": stats_state,
        }
        save_checkpoint(checkpoint_path, payload)
        if on_day is not None:
            on_day(index, day)
    return completed


def run_full_range_configuration(*, plan_id, approval_id, plan, row, work_root, on_day=None):
    """Run/resume one row's ordinary stream and two independent funded streams."""
    from dataclasses import replace

    from alpha_lab.agents.data_infra.ifvg.day_artifacts import levels_for_from_frame
    from alpha_lab.agents.data_infra.ifvg.menthorq_levels import load_menthorq_levels
    from alpha_lab.agents.data_infra.ifvg.prepared_store import load_registered_day_artifacts
    from alpha_lab.propsim.funded.comparison_describe import describe_section

    started = time.monotonic()
    sections, base_cfg, policy, schedule, start_ns, cutoff_ns = validate_full_range_plan(
        plan, verify_inputs=False
    )
    ordinal = tuple(item.batch_id for item in plan.configurations).index(row.batch_id)
    section = sections[ordinal]
    cfg = replace(base_cfg, section=section)
    dispatch = _dispatch(plan_id, approval_id, plan, row)
    dispatch_sha = canonical_contract_sha256(dispatch)
    work_root = Path(work_root) / row.batch_id
    checkpoint_path = work_root / "checkpoint.json"
    dates = plan.source.warmup_dates + plan.source.evaluation_dates
    checkpoint = load_checkpoint(checkpoint_path, dispatch_sha256=dispatch_sha, dates=dates)
    providers = [
        load_menthorq_levels(preparation_catalog_paths=cfg.preparation_catalog_paths)
        for _ in range(3)
    ]
    if any(
        dict(provider.source_file_sha256) != dict(plan.task_b_scope.menthorq_source_file_sha256)
        for provider in providers
    ):
        raise PermissionError("passive context sources changed after plan approval")
    drivers = [
        ObservedCoreDriver(section, tick_size=cfg.tick_size, menthorq_provider=provider)
        for provider in providers
    ]
    runs = [PairRun("reference", drivers[0], None)]
    for profile, driver in zip(plan.firm_profiles, drivers[1:], strict=True):
        pair_id = pair_id_for(row.name, profile.firm_key)
        ledger = PairLedger(
            pair_id=pair_id,
            configuration=row.name,
            profile=profile,
            processing=plan.processing,
            quantity=row.quantity,
            tick_value_cents=INSTRUMENTS[row.instrument].tick_value_cents,
            cost_per_side_cents=0,
            cost_per_contract_mills=row.cost_per_contract_mills,
            trading_days=schedule,
            start_ns=start_ns,
            cutoff_ns=cutoff_ns,
            scale_out=row.exit_policy != "fixed_target_v1",
        )
        runs.append(PairRun(pair_id, driver, ledger))
    daily_activity, saved_stats = [], {"completed_count": 0}
    if checkpoint is not None:
        for run, state, provider, provider_state in zip(
            runs, checkpoint["runs"], providers, checkpoint["providers"], strict=True
        ):
            if run.pair_id != state["pair_id"]:
                raise PermissionError("checkpoint stream membership differs from dispatch")
            run.driver.restore(state["driver"])
            if run.ledger is not None:
                run.ledger.restore(state["ledger"])
            run.strategy_trades = state["strategy_trades"]
            run.warmup_trades, run.also_blocked = state["warmup_trades"], state["also_blocked"]
            _restore_provider(provider, provider_state)
        daily_activity, saved_stats = checkpoint["daily_activity"], checkpoint["stats"]
    allowed = frozenset(
        day for registration in policy.registrations for day in registration.source_dates
    )
    # Resume verifies the named print-source receipts before restored state is used.
    print_sources = _verify_saved_print_receipts(
        cfg, policy, saved_stats.get("print_source_receipts", [])
    )

    def authorize_print_day(day):
        if day.isoformat() not in allowed or day.isoformat() > "2026-06-10":
            raise PermissionError("funded prints lie outside registered source dates")

    stats = PrintStats(
        cfg.data_dir / cfg.symbol,
        authorize_source_day=authorize_print_day,
        source_file_observer=print_sources.observe,
    )
    stats.minutes_checked = saved_stats.get("minutes_checked", 0)
    stats.minutes_matched = saved_stats.get("minutes_matched", 0)
    stats.files = {item["file"]: item for item in saved_stats.get("files", [])}
    stats.missing = set(saved_stats.get("missing_utc_days", []))
    evaluation = set(plan.source.evaluation_dates)
    by_day = {day.trading_day: day for day in schedule}
    coverage = {}

    def days():
        for index, day in enumerate(
            dates[saved_stats["completed_count"] :], start=saved_stats["completed_count"]
        ):
            artifacts = load_registered_day_artifacts(day, cfg, access_policy=policy)
            for provider in providers:
                provider.register_day_artifacts(artifacts, cfg.tick_size)
            bars = {}
            for bar in artifacts.bars:
                bars.setdefault(bar.timeframe_ticks, []).append(bar)
            selected = providers[0]._selected_instruments.get(date.fromisoformat(day))
            coverage[day] = {
                "source_status": "available" if bars.get(60) else "unavailable",
                "source_selected_instrument_id": selected,
                "roll_flag": providers[0].roll_flag_for(date.fromisoformat(day)),
            }
            yield DayInput(
                day,
                bars,
                levels_for_from_frame(artifacts.level_timeline),
                day in evaluation,
                by_day.get(day),
                index == len(dates) - 1,
            )
        policy.assert_zero_forbidden_access()

    def prints_factory(day):
        factory = stats.factory(day)
        original_release = factory.release

        def release():
            original_release()
            saved_stats.update(
                {
                    "minutes_checked": stats.minutes_checked,
                    "minutes_matched": stats.minutes_matched,
                    "files": list(stats.files.values()),
                    "missing_utc_days": sorted(stats.missing),
                    "print_source_receipts": list(print_sources.files.values()),
                }
            )

        factory.release = release
        return factory

    completed = run_stream_days(
        runs,
        providers,
        days(),
        prints_factory,
        checkpoint_path=checkpoint_path,
        dispatch_sha256=dispatch_sha,
        all_dates=dates,
        daily_activity=daily_activity,
        coverage_for_day=coverage.__getitem__,
        stats_state=saved_stats,
        on_day=on_day,
    )
    if tuple(completed) != dates:
        raise PermissionError("configuration did not complete the identical full date scope")
    for run in runs[1:]:
        run.ledger.finish()
    print_sources.recheck_stats()
    pairs = {
        run.ledger.profile.firm_key: {
            "pair_id": run.pair_id,
            "ledger": _roundtrip(run.ledger.snapshot()),
            "strategy_trades": run.strategy_trades,
            "entry_candidates_also_blocked_by_strategy": run.also_blocked,
            "forced_flat": run.driver.forced_flat,
            "discarded_refused_setups": run.driver.discarded_setups,
            "final_seed_hash": run.driver.seed_hash(),
            "trades_matching_reference_entries": None,
            "trades_not_in_reference": None,
        }
        for run in runs[1:]
    }
    output = {
        "configuration": row.name,
        "batch_id": row.batch_id,
        "display_name": row.display_name,
        "axes": dict(row.axis_value_ids),
        "settings_plain": describe_section(section),
        "section_config_hash": cfg.profile_hash,
        "exit_policy": row.exit_policy,
        "sizing": {
            "instrument": row.instrument,
            "quantity": row.quantity,
            "instrument_label": INSTRUMENTS[row.instrument].label,
            "tick_value_cents": INSTRUMENTS[row.instrument].tick_value_cents,
            "cost_per_contract_mills": row.cost_per_contract_mills,
        },
        "strategy_trades_no_account": runs[0].strategy_trades,
        "pairs": pairs,
        "daily_activity": daily_activity,
        "dispatch": dispatch,
        "dispatch_sha256": dispatch_sha,
        "completed_dates": completed,
        "reference": {
            "equivalent": None,
            "compared_with": "No historical equality claim for this new full continuation; "
            "matched-input engineering checks establish driver preservation.",
            "saved_study_trades": None,
            "replayed_trades": len(runs[0].strategy_trades),
            "warmup_trades": runs[0].warmup_trades,
        },
        "resumed": {},
        "prints": {
            "minutes_checked": stats.minutes_checked,
            "minutes_rebuilt_exactly": stats.minutes_matched,
            "files": list(stats.files.values()),
            "missing_utc_days": sorted(stats.missing),
        },
        "print_source_receipts": list(print_sources.files.values()),
        "access_audit": policy.audit_dict(),
        "seconds": round(time.monotonic() - started, 1),
    }
    save_checkpoint(
        work_root / "output.json",
        {"dispatch_sha256": dispatch_sha, "completed_dates": completed, "output": output},
    )
    return output


def _run_worker(task):
    plan = FullRangeBatchPlanPayload.model_validate(task["plan"])
    row = next(row for row in plan.configurations if row.batch_id == task["batch_id"])
    return run_full_range_configuration(
        plan_id=task["plan_id"],
        approval_id=task["approval_id"],
        plan=plan,
        row=row,
        work_root=Path(task["work_root"]),
    )


def run_full_range_batch(
    *, plan_id: str, store_root: Path, state_root: Path, reports_root: Path, workers: int = 2
):
    """Execute only the approved frozen six-row plan; reuse exact finished row outputs."""
    from alpha_lab.propsim.funded.comparison_result import (
        build_comparison_result,
        validate_comparison,
    )
    from alpha_lab.propsim.funded.comparison_runner import (
        RESULT_SIDECAR,
        find_approval,
        load_comparison_result,
    )
    from alpha_lab.propsim.funded.full_range_reporting import attach_full_range_reports

    store_root, state_root = Path(store_root), Path(state_root)
    plan = load_verified_envelope(
        store_root, PLAN_STORE, plan_id, FullRangeBatchPlanEnvelope
    ).payload
    sections, cfg, policy, schedule, start_ns, cutoff_ns = validate_full_range_plan(plan)
    approval = find_approval(store_root, plan_id)
    if approval is None:
        raise PermissionError("this exact final batch plan has no new owner approval")
    prior = read_state(state_root, plan_id) or {}
    if prior.get("status") in {"Completed", "Incomplete"} and prior.get("result_id"):
        saved = load_comparison_result(store_root, prior["result_id"])
        saved_envelope = load_verified_envelope(
            store_root, RESULT_STORE, prior["result_id"], FundedComparisonResultEnvelope
        )
        terminal_status = _assert_terminal_result_binding(
            saved,
            saved_envelope,
            plan_id=plan_id,
            approval_id=approval.funded_comparison_approval_id,
            plan=plan,
        )
        if prior["status"] != terminal_status:
            raise PermissionError("mutable job status disagrees with the verified terminal result")
        _verify_saved_print_receipts(
            cfg, policy, saved["full_range_batch"]["print_source_receipts"]
        )
        return prior
    approval_id = approval.funded_comparison_approval_id
    work_root = state_root / plan_id / "workers"
    write_state(
        state_root,
        plan_id,
        status="Running",
        kind="funded_comparison",
        phase="simulating_full_range",
        pid=os.getpid(),
        configurations_total=6,
        configurations_done=0,
        started_at_utc=datetime.now(UTC).isoformat(),
    )
    outputs, failures, tasks = [], [], []
    for row in plan.configurations:
        dispatch = _dispatch(plan_id, approval_id, plan, row)
        completed = load_checkpoint(
            work_root / row.batch_id / "output.json",
            dispatch_sha256=canonical_contract_sha256(dispatch),
            dates=plan.source.warmup_dates + plan.source.evaluation_dates,
        )
        if completed is not None:
            if (
                tuple(completed["completed_dates"])
                != plan.source.warmup_dates + plan.source.evaluation_dates
            ):
                raise PermissionError("saved row output covers a different date scope")
            output = completed["output"]
            if (
                output["dispatch"] != dispatch
                or output["dispatch_sha256"] != canonical_contract_sha256(dispatch)
                or output["batch_id"] != row.batch_id
                or output["configuration"] != row.name
            ):
                raise PermissionError("saved row output differs from the frozen worker dispatch")
            _verify_saved_print_receipts(cfg, policy, output["print_source_receipts"])
            outputs.append(output)
        else:
            tasks.append(
                {
                    "plan_id": plan_id,
                    "approval_id": approval_id,
                    "plan": plan.model_dump(mode="json"),
                    "batch_id": row.batch_id,
                    "work_root": str(work_root),
                }
            )
    with ProcessPoolExecutor(max_workers=max(1, min(workers, 6))) as pool:
        futures = {pool.submit(_run_worker, task): task["batch_id"] for task in tasks}
        for future in as_completed(futures):
            batch_id = futures[future]
            try:
                outputs.append(future.result())
            except Exception as error:
                row = next(row for row in plan.configurations if row.batch_id == batch_id)
                failures.append(
                    {
                        "configuration": row.name,
                        "batch_id": row.batch_id,
                        "display_name": row.display_name,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
            write_state(
                state_root,
                plan_id,
                configurations_done=len(outputs) + len(failures),
                configurations_failed=len(failures),
            )
    outputs.sort(key=lambda output: output["batch_id"])
    context = {
        "funded_comparison_plan_id": plan_id,
        "purpose": plan.purpose,
        "question": plan.question,
        "run_identity": {
            "funded_comparison_plan_id": plan_id,
            "engine_version": ENGINE_VERSION,
            "core_source": plan.core_source.model_dump(mode="json"),
        },
        "approval": {
            "approved_on": approval.payload.approved_on,
            "channel": approval.payload.channel,
            "scope": approval.payload.scope,
        },
        "source": {
            "title": plan.source.title,
            "evaluation_first_day": plan.source.evaluation_dates[0],
            "evaluation_last_day": plan.source.evaluation_dates[-1],
            "evaluation_days": 253,
            "warmup_days": 10,
        },
        "owner_decisions": [decision.model_dump(mode="json") for decision in plan.owner_decisions],
        "limitations": list(plan.limitations),
    }
    settings = {
        "size_text": "C01/C02: ten micros; C03-C06: one mini",
        "sizing_by_configuration": {
            row.name: {
                "instrument": row.instrument,
                "quantity": row.quantity,
                "tick_value_cents": INSTRUMENTS[row.instrument].tick_value_cents,
                "cost_per_contract_per_fill_usd": row.cost_per_contract_mills / 1000,
                "exit_policy": row.exit_policy,
            }
            for row in plan.configurations
        },
        "firm_profiles": [profile.model_dump(mode="json") for profile in plan.firm_profiles],
        "processing_clock": TWO_BUSINESS_DAYS_FED_1600.model_dump(mode="json"),
        "execution_model": plan.execution_model.model_dump(mode="json"),
        "core_source": plan.core_source.model_dump(mode="json"),
    }
    result = build_comparison_result(
        context=context,
        outputs=outputs,
        failures=failures,
        profiles=plan.firm_profiles,
        trading_days=schedule,
        start_ns=start_ns,
        cutoff_ns=cutoff_ns,
        settings=settings,
        resume_check_requested=False,
    )
    result["validation"] = validate_comparison(
        result,
        outputs,
        plan.firm_profiles,
        cutoff_ns,
        require_resume_check=False,
        require_reference_check=False,
    )
    result["full_range_batch"] = {
        "plan": plan.model_dump(mode="json"),
        "input_metadata_access_audit": policy.audit_dict(),
        "worker_dispatches": [output["dispatch"] for output in outputs],
        "print_source_receipts": list(
            {
                item["path"]: item for output in outputs for item in output["print_source_receipts"]
            }.values()
        ),
        "failed_configurations": failures,
    }
    result = attach_full_range_reports(
        result,
        outputs,
        failures,
        configuration_names=CONFIGURATION_NAMES,
        evaluation_dates=plan.source.evaluation_dates,
        warmup_dates=plan.source.warmup_dates,
        cutoff_utc=plan.source.cutoff_utc,
    )
    result["validation"]["checks"].update(
        {
            "exactly_six_configurations": len(outputs) + len(failures) == 6,
            "exactly_twelve_funded_rows": len(result["tables"]["pair_results"]) == 12,
            "complete_date_membership": all(
                tuple(output["completed_dates"])
                == plan.source.warmup_dates + plan.source.evaluation_dates
                for output in outputs
            ),
            "every_configuration_completed": not failures,
        }
    )
    result["validation"]["configurations_not_completed"] = failures
    result["validation"]["passed"] = bool(
        result["validation"]["passed"]
        and result["validation"]["checks"]["exactly_six_configurations"]
        and result["validation"]["checks"]["exactly_twelve_funded_rows"]
        and result["validation"]["checks"]["complete_date_membership"]
        and result["full_range_reporting"]["validation"]["passed"]
    )
    payload = FundedComparisonResultPayload(
        funded_comparison_plan_id=plan_id,
        funded_comparison_approval_id=approval_id,
        result_json_sha256=result_sha256(result),
        validation_passed=result["validation"]["passed"],
        engine_version=ENGINE_VERSION,
    )
    envelope = FundedComparisonResultEnvelope.from_payload(payload)
    result_id = envelope.funded_comparison_result_id
    if not has_envelope(store_root, RESULT_STORE, result_id):
        save_envelope_immutable(
            store_root,
            RESULT_STORE,
            envelope,
            extra_files={RESULT_SIDECAR: canonical_json(result).encode()},
        )
    load_comparison_result(store_root, result_id)
    append_ledger(
        store_root,
        {
            "event_id": f"full_range_{plan_id}_completed",
            "event_type": "run_completed",
            "funded_comparison_plan_id": plan_id,
            "funded_comparison_result_id": result_id,
            "mode": plan.mode,
            "status": "completed" if not failures else "incomplete",
            "configurations_completed": len(outputs),
            "configurations_not_completed": failures,
        },
    )
    # Root/export adapter publishes the verified saved result with task supplements.
    return write_state(
        state_root,
        plan_id,
        result_id=result_id,
        status="Completed" if not failures and payload.validation_passed else "Incomplete",
        phase="completed_unpublished",
        configurations_done=6,
        configurations_failed=len(failures),
        completed_at_utc=datetime.now(UTC).isoformat(),
        review_folder=None,
        review_error=None,
    )
