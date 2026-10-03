"""Exact Task B stages using the normal sequential replay and funded ledger.

No feature, label, fold, fit, prediction, bootstrap, frontier or insight stage
is present. The thirteen named configurations each get their own firm state.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import ClassVar, Literal
from zoneinfo import ZoneInfo

from pydantic import Field

from ..contracts import RecordTable
from ..data_access import allowlist_sha256
from .identities import (
    SHA256_PATTERN,
    CoreReplayArtifactReference,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
)
from .pipeline import (
    PipelineRunScope,
    PipelineSemanticIdentity,
    PipelineSemanticSpecPayload,
    PipelineWiring,
    QuantLabPipelineStage,
)
from .store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .task_b import TASK_B_WARMUP_POLICY, validate_task_b_request

TASK_B_STAGES = (
    QuantLabPipelineStage.S00_VALIDATE_INPUTS,
    QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
    QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    QuantLabPipelineStage.S03_BUILD_OR_REUSE_FSM_AUDIT,
    QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS,
    QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS,
    QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH,
)


def _source_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class _PrintSourceReceipts:
    """Bind only the named files opened by the ordered-print reader."""

    def __init__(self):
        self.files = {}

    def observe(self, path):
        path = Path(path).resolve()
        day = path.parent.name
        date.fromisoformat(day)
        if day > "2026-06-10":
            raise PermissionError("funded print source is beyond the authorized cutoff")
        stat = path.stat()
        stamp = (stat.st_size, stat.st_mtime_ns)
        prior = self.files.get(str(path))
        if prior is not None:
            if (prior["size_bytes"], prior["mtime_ns"]) != stamp:
                raise PermissionError("funded print source changed during the account replays")
            return
        checksum = _source_sha256(path)
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != stamp:
            raise PermissionError("funded print source changed while its receipt was built")
        self.files[str(path)] = {
            "path": str(path), "physical_date": day, "filename": path.name,
            "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": checksum,
        }

    def recheck_stats(self):
        for path in self.files:
            self.observe(path)


_VERIFIED_PRINT_SOURCE_HASHES: dict[tuple[str, int, int], str] = {}


def _verify_print_source_receipts(context, files):
    from ..prepared_store import load_prepared_store

    scope = context.charter.payload.task_b_execution
    registrations = tuple(load_prepared_store(path) for path in scope.prepared_store_registry_paths)
    allowed = {day for registration in registrations for day in registration.source_dates}
    roots = {
        (Path(registration.definition["raw_data_dir"])
         / registration.definition["symbol"]).resolve()
        for registration in registrations
    }
    receipts = json.loads(files["print_source_receipts.json"])
    for item in receipts:
        path = Path(item["path"]).resolve()
        day = item["physical_date"]
        if (
            day not in allowed or day > "2026-06-10" or path.parent.name != day
            or path.parent.parent not in roots or path.name != item["filename"]
            or path.name not in {"mbp10.parquet", "mbp1.parquet", "trades.parquet"}
        ):
            raise PermissionError("saved print receipt is outside registered named source paths")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size_bytes"], item["mtime_ns"]):
            raise PermissionError("saved funded print source size or modification stamp changed")
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        actual = _VERIFIED_PRINT_SOURCE_HASHES.get(key)
        if actual is None:
            actual = _source_sha256(path)
            after = path.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise PermissionError("saved funded print source changed during verification")
            _VERIFIED_PRINT_SOURCE_HASHES[key] = actual
        if actual != item["sha256"]:
            raise PermissionError("saved funded print source bytes differ from their receipt")


def task_b_semantic(charter) -> PipelineSemanticIdentity:
    validate_task_b_request(charter.payload)
    scope = charter.payload.task_b_execution
    if scope is None:
        raise PermissionError("the Task B pipeline requires its exact task scope")
    return PipelineSemanticIdentity.from_payload(PipelineSemanticSpecPayload(
        run_scope=PipelineRunScope.FULL_AUTHORIZED_DEVELOPMENT,
        date_allowlist=charter.payload.date_policy.replay_dates,
        allowlist_hash=allowlist_sha256(charter.payload.date_policy.replay_dates),
        warmup_policy_id=TASK_B_WARMUP_POLICY, search_charter_id=charter.search_id,
        source_artifact_ids=(), feature_bundle_ids=(), label_policy_id=None,
        fold_protocol_id=None, model_protocol_id=None,
        cost_policy_sha256=canonical_contract_sha256(charter.payload.cost_policy),
        account_policy_set_ids=tuple(sorted(scope.funded_profile_sha256.values())),
        portfolio_policy_ids=(), simulation_protocol=charter.payload.simulation_protocol,
        software_commits={
            "quant_lab": charter.payload.quant_lab_commit,
            "strategy_core": charter.payload.strategy_core_commit,
        }, stage_plan=TASK_B_STAGES,
    ))


class TaskBArtifactPayload(FrozenContract):
    kind: Literal["funded_accounts", "study_tables"]
    search_id: str = Field(pattern=SHA256_PATTERN)
    core_replay_ids: tuple[str, ...]
    evaluation_dates: tuple[str, ...]
    files: ImmutableMap[str, str]
    funded_profile_sha256: ImmutableMap[str, str]
    processing_policy_sha256: str = Field(pattern=SHA256_PATTERN)


class TaskBArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "task_b_artifact_id"
    task_b_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: TaskBArtifactPayload


def _inputs(context):
    rows = sorted(context.children, key=lambda row: row["ordinal"])
    if len(rows) != 13 or any(row["state"] not in {"completed", "reused"} for row in rows):
        raise PermissionError("Task B requires thirteen completed exact strategy children")
    return rows


def _artifact_payload(context, kind, files):
    scope = context.charter.payload.task_b_execution
    return TaskBArtifactPayload(
        kind=kind, search_id=context.charter.search_id,
        core_replay_ids=tuple(row["core_replay_id"] for row in _inputs(context)),
        evaluation_dates=tuple(context.charter.payload.date_policy.replay_dates[10:]),
        files={name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
        funded_profile_sha256=scope.funded_profile_sha256,
        processing_policy_sha256=scope.processing_policy_sha256,
    )


def _save(context, kind, files):
    envelope = TaskBArtifactEnvelope.from_payload(_artifact_payload(context, kind, files))
    saved, _ = save_or_reuse_envelope(
        context.store_root, "task_b_artifacts", envelope, extra_files=files
    )
    index = context.store_root / "task_b_runs" / context.charter.search_id / (kind + ".json")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"artifact_id": saved.task_b_artifact_id}), encoding="utf-8")
    return saved


def _load(context, kind):
    index = context.store_root / "task_b_runs" / context.charter.search_id / (kind + ".json")
    if not index.exists():
        return None
    artifact_id = json.loads(index.read_text(encoding="utf-8"))["artifact_id"]
    envelope = load_verified_envelope(
        context.store_root, "task_b_artifacts", artifact_id, TaskBArtifactEnvelope
    )
    files = {
        name: load_sidecar_bytes(context.store_root, "task_b_artifacts", artifact_id, name)
        for name in envelope.payload.files
    }
    if envelope.payload != _artifact_payload(context, kind, files):
        raise PermissionError("saved Task B evidence differs from this exact plan")
    if kind == "funded_accounts":
        _verify_print_source_receipts(context, files)
    return envelope, files


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _study_run_identity(context, row, spec, cfg, ref, core, bundle, policy):
    return {
        "search_id": context.charter.search_id,
        "pipeline_semantic_id": context.semantic.pipeline_semantic_id,
        "core_replay_id": row["core_replay_id"],
        "v2_dataset_artifact_id": ref.v2_dataset_artifact_id,
        "resolved_section_config_hash": spec.resolved_section_config_hash,
        "replay_dates": list(policy.dates),
        "evaluation_dates": list(policy.evaluation_dates),
        "warmup_dates": list(policy.warmup_dates),
        "strategy_core_commit": core.payload.strategy_core_commit,
        "strategy_core_source_identity": core.payload.strategy_core_source_identity,
        "quant_lab_commit": context.charter.payload.quant_lab_commit,
        "quant_lab_replay_source_identity": core.payload.quant_lab_replay_source_identity,
        "replay_input_bundle_id": bundle.replay_input_bundle_id,
    }


def _provider(cfg, scope):
    from ..menthorq_levels import load_menthorq_levels
    from ..prepared_store import load_prepared_store

    catalogs = tuple(
        Path(load_prepared_store(path).definition["catalog_path"])
        for path in cfg.prepared_store_registry_paths
    )
    provider = load_menthorq_levels(preparation_catalog_paths=catalogs)
    if dict(provider.source_file_sha256) != dict(scope.menthorq_source_file_sha256):
        raise PermissionError("Task B provider source hashes differ from its exact approved inputs")
    return provider


def _calendar(section, evaluation_dates):
    from strategy_core.strategies.ifvg_smc.daily_close import configured_daily_close_calendar

    from alpha_lab.propsim.funded.campaign import TradingDay
    from alpha_lab.propsim.funded.clock import to_ns

    calendar = configured_daily_close_calendar(
        section.daily_close_timezone, section.daily_close_time, section.daily_close_buffer_minutes
    )
    boundary = time.fromisoformat(section.session_scheme.trading_day_boundary)
    logical_zone = ZoneInfo(section.session_scheme.timezone)
    schedule = []
    for day in evaluation_dates:
        logical_date = date.fromisoformat(day)
        opened = datetime.combine(logical_date - timedelta(days=1), boundary, logical_zone)
        ended = datetime.combine(logical_date, boundary, logical_zone)
        first_local = opened.astimezone(calendar.zone).date()
        last_local = ended.astimezone(calendar.zone).date()
        closures = tuple(
            closure for offset in range((last_local - first_local).days + 1)
            for closure in calendar.closures_on(first_local + timedelta(days=offset))
            if opened <= closure.market_close_ts_utc < ended
        )
        if len(closures) != 1:
            raise PermissionError(
                f"Task B logical day {day} must own one existing market-close transition"
            )
        closure = closures[0]
        if not opened <= closure.deadline_ts_utc < ended:
            raise PermissionError("Task B mandatory deadline lies outside its logical day")
        schedule.append(TradingDay(
            trading_day=day, day_end_ns=to_ns(closure.market_close_ts_utc),
            reopen_ns=to_ns(closure.reopen_ts_utc), deadline_ns=to_ns(closure.deadline_ts_utc),
        ))
    last = [row for row in schedule if row.trading_day == evaluation_dates[-1]]
    if len(last) != 1:
        raise PermissionError("Task B cutoff must resolve to one existing scheduled market close")
    first = date.fromisoformat(evaluation_dates[0]) - timedelta(days=1)
    start = datetime.combine(first, boundary, logical_zone)
    return tuple(schedule), to_ns(start), last[0].day_end_ns


def _funded(context, contexts, artifact_policy):
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.comparison_describe import describe_section
    from alpha_lab.propsim.funded.comparison_result import (
        build_comparison_result,
        validate_comparison,
    )
    from alpha_lab.propsim.funded.comparison_run import PrintStats, pair_id_for
    from alpha_lab.propsim.funded.comparison_source import core_trade_key
    from alpha_lab.propsim.funded.pair_engine import DayInput, PairRun, run_days
    from alpha_lab.propsim.funded.pair_ledger import PairLedger
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES, INSTRUMENTS
    from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver

    from ..day_artifacts import levels_for_from_frame
    from ..prepared_store import load_registered_day_artifacts
    from .saved_strategy_result import load_saved_strategy_result

    prior = _load(context, "funded_accounts")
    if prior:
        outputs = json.loads(prior[1]["funded_runs.json"])
        result = json.loads(prior[1]["comparison_result.json"])
        validation = validate_comparison(
            result, outputs, tuple(FIRM_PROFILES.values()),
            cutoff_ns=result["task_b_cutoff_ns"], require_resume_check=False,
        )
        if not validation["passed"] or validation != result["validation"]:
            raise PermissionError(
                "saved funded account evidence failed canonical rule verification"
            )
        return (prior[0].task_b_artifact_id,), "Verified funded account results reused"
    evaluation_dates = tuple(context.charter.payload.date_policy.replay_dates[10:])
    outputs = []
    shared_calendar = None
    print_sources = _PrintSourceReceipts()
    for row, config in zip(
        _inputs(context), context.charter.payload.explicit_configurations, strict=True
    ):
        resolved, cfg, _bundle, _core = contexts[
            next(spec.resolved_section_config_hash for spec in context.specs
                 if spec.ordinal == row["ordinal"])
        ]
        section = resolved.section
        schedule, start_ns, cutoff_ns = _calendar(section, evaluation_dates)
        if shared_calendar is None:
            shared_calendar = (schedule, start_ns, cutoff_ns)
        elif shared_calendar != (schedule, start_ns, cutoff_ns):
            raise PermissionError(
                "Task B configurations disagree on their existing holding calendar"
            )
        by_day = {day.trading_day: day for day in schedule}
        providers = [_provider(cfg, context.charter.payload.task_b_execution)
                     for _ in range(len(FIRM_PROFILES) + 1)]
        active = section.menthorq_context_version is not None
        drivers = [CoreStrategyDriver(section, tick_size=cfg.tick_size,
                    menthorq_provider=provider if active else None) for provider in providers]
        reference = PairRun("reference", drivers[0], None)
        runs = [PairRun(pair_id_for(config.name, firm.firm_key), driver, PairLedger(
            pair_id=pair_id_for(config.name, firm.firm_key), configuration=config.name,
            profile=firm, processing=TWO_BUSINESS_DAYS_FED_1600,
            quantity=1, tick_value_cents=INSTRUMENTS["mini"].tick_value_cents,
            cost_per_side_cents=514, trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns,
        )) for driver, firm in zip(drivers[1:], FIRM_PROFILES.values(), strict=True)]
        policy = artifact_policy()

        def days(policy=policy, cfg=cfg, providers=providers, by_day=by_day):
            for index, day in enumerate(policy.dates):
                artifacts = load_registered_day_artifacts(day, cfg, access_policy=policy)
                for provider in providers:
                    provider.register_day_artifacts(artifacts, cfg.tick_size)
                bars = {}
                for bar in artifacts.bars:
                    bars.setdefault(bar.timeframe_ticks, []).append(bar)
                yield DayInput(
                    day, bars, levels_for_from_frame(artifacts.level_timeline),
                    day in evaluation_dates, by_day.get(day), index == len(policy.dates)-1
                )
            policy.assert_zero_forbidden_access()

        allowed = {d for registration in policy.registrations for d in registration.source_dates}

        def authorize_print_day(day, allowed=frozenset(allowed)):
            if day.isoformat() not in allowed or day.isoformat() > "2026-06-10":
                raise PermissionError("funded print evidence lies outside registered source dates")

        stats = PrintStats(
            cfg.data_dir / cfg.symbol, authorize_source_day=authorize_print_day,
            source_file_observer=print_sources.observe,
        )
        run_days([reference, *runs], days(), stats.factory)
        for run in runs:
            run.ledger.finish()
        saved = load_saved_strategy_result(context.store_root, row["core_replay_id"])
        expected_frame = saved.tables[RecordTable.EXECUTED_TRADE]
        expected_frame = expected_frame.loc[~expected_frame["is_warmup"].astype(bool)]
        expected = sorted(core_trade_key(record) for record in expected_frame.to_dict("records"))
        actual = sorted(core_trade_key(record) for record in reference.strategy_trades
                        if not record["is_warmup"])
        if expected != actual:
            raise AssertionError(
                f"{config.name}: funded reference differs from the exact saved strategy child"
            )
        reference_keys = {key[:3] for key in actual}
        outputs.append({
            "configuration": config.name, "display_name": config.name,
            "axes": dict(config.axis_value_ids), "settings_plain": describe_section(section),
            "section_config_hash": cfg.profile_hash,
            "exit_policy": getattr(section, "exit_policy", "fixed_target_v1"),
            "sizing": {"instrument": "mini", "instrument_label": INSTRUMENTS["mini"].label,
                       "quantity": 1, "tick_value_cents": 500, "cost_per_contract_mills": 5140},
            "strategy_trades_no_account": reference.strategy_trades,
            "reference": {"equivalent": True, "saved_study_trades": len(expected),
                          "replayed_trades": len(actual)},
            "resumed": {}, "pairs": {
            run.ledger.profile.firm_key: {
                "pair_id": run.pair_id, "ledger": run.ledger.snapshot(),
                "strategy_trades": run.strategy_trades, "final_seed_hash": run.driver.seed_hash(),
                "forced_flat": run.driver.forced_flat,
                "discarded_refused_setups": run.driver.discarded_setups,
                "entry_candidates_also_blocked_by_strategy": run.also_blocked,
                "trades_not_in_reference": sum(
                    (datetime.fromtimestamp(t["entry_ns"] // 1_000_000_000, tz=UTC).isoformat(),
                     t["entry_ticks"], t["stop_ticks"]) not in reference_keys
                    for t in run.ledger.trades
                ),
            } for run in runs
        }, "prints": {
            "minutes_checked": stats.minutes_checked,
            "minutes_rebuilt_exactly": stats.minutes_matched,
            "missing_utc_days": sorted(stats.missing), "files": list(stats.files.values()),
        }})
        print_sources.recheck_stats()
        validate_task_b_request(context.charter.payload, verify_files=True)
    schedule, start_ns, cutoff_ns = shared_calendar
    result = build_comparison_result(
        context={"search_id": context.charter.search_id, "study_name": "menthorq_level_rules_b1",
                 "task_b_cutoff_ns": cutoff_ns},
        outputs=outputs, failures=[], profiles=tuple(FIRM_PROFILES.values()),
        trading_days=schedule, start_ns=start_ns, cutoff_ns=cutoff_ns,
        settings={"instrument": "mini", "quantity": 1, "tick_value_cents": 500,
                  "cost_per_side_cents": 514,
                  "processing_policy": TWO_BUSINESS_DAYS_FED_1600.model_dump(mode="json")},
        rank_results=False, resume_check_requested=False,
    )
    result["validation"] = validate_comparison(
        result, outputs, tuple(FIRM_PROFILES.values()), cutoff_ns, require_resume_check=False
    )
    if not result["validation"]["passed"]:
        raise AssertionError(
            "Task B funded accounts failed canonical independent rule verification"
        )
    saved = _save(context, "funded_accounts", {
        "funded_runs.json": _json_bytes(outputs), "comparison_result.json": _json_bytes(result),
        "print_source_receipts.json": _json_bytes(
            sorted(print_sources.files.values(), key=lambda item: item["path"])
        ),
    })
    return (saved.task_b_artifact_id,), (
        "Thirteen configurations replayed with independent funded firm accounts"
    )


def _reports(context, contexts, artifact_policy):
    from ..artifact_io import load_verified_v2_artifact
    from ..menthorq_study_reporting import StudyConfigurationInput, write_study_reports
    from ..prepared_store import load_registered_day_artifacts

    funded = _load(context, "funded_accounts")
    if funded is None:
        raise PermissionError("Task B table reports require verified funded account evidence")
    cash = json.loads(funded[1]["comparison_result.json"])
    inputs = []
    for row, config in zip(
        _inputs(context), context.charter.payload.explicit_configurations, strict=True
    ):
        spec = next(spec for spec in context.specs if spec.ordinal == row["ordinal"])
        resolved, cfg, _bundle, _core = contexts[spec.resolved_section_config_hash]
        ref = CoreReplayArtifactReference.model_validate_json(load_sidecar_bytes(
            context.store_root, "core_replays", row["core_replay_id"], "artifact_reference.json"
        ))
        artifact = load_verified_v2_artifact(
            context.store_root / "v2_datasets", ref.v2_dataset_artifact_id
        )
        provider = _provider(cfg, context.charter.payload.task_b_execution)
        policy = artifact_policy()
        for day in policy.dates:
            provider.register_day_artifacts(
                load_registered_day_artifacts(day, cfg, access_policy=policy), cfg.tick_size
            )
        policy.assert_zero_forbidden_access()
        inputs.append(StudyConfigurationInput(
            configuration=config.name, tables=artifact.tables, provider=provider,
            section=resolved.section, evaluation_days=tuple(policy.evaluation_dates),
            funded_result=cash,
            cost_points=context.charter.payload.cost_policy.cost_points_round_turn,
            tick_size=cfg.tick_size,
            run_identity=_study_run_identity(
                context, row, spec, cfg, ref, _core, _bundle, policy
            ),
        ))
    work = context.store_root / "task_b_runs" / context.charter.search_id / "tables"
    paths = write_study_reports(work, inputs)
    review_root = work.parent / "review_exports"
    review_paths = {}
    for name, path in paths.items():
        if name.endswith("/context_export"):
            target = review_root / (name + ".csv")
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
            review_paths[name] = str(target)
    (review_root / "review_paths.json").write_bytes(_json_bytes(review_paths))
    relative = {
        str(path.relative_to(work)).replace("\\", "/"): path
        for name, path in paths.items() if not name.endswith("/context_export")
    }
    files = {name.replace("/", "__"): path.read_bytes() for name, path in relative.items()}
    if len(files) != len(relative):
        raise ValueError("Task B report paths collide after store-sidecar flattening")
    files["report_paths.json"] = _json_bytes({name: name.replace("/", "__") for name in relative})
    saved = _save(context, "study_tables", files)
    return (saved.task_b_artifact_id,), "Task B factual tables saved and reconciled"


def _verify_review_exports(context, contexts, artifact_policy):
    """Read review headers/counts only; exports never enter replay or ML identity."""
    from ..artifact_io import load_verified_v2_artifact

    review_root = context.store_root / "task_b_runs" / context.charter.search_id / "review_exports"
    for row, config in zip(
        _inputs(context), context.charter.payload.explicit_configurations, strict=True
    ):
        spec = next(spec for spec in context.specs if spec.ordinal == row["ordinal"])
        resolved, cfg, bundle, core = contexts[spec.resolved_section_config_hash]
        ref = CoreReplayArtifactReference.model_validate_json(load_sidecar_bytes(
            context.store_root, "core_replays", row["core_replay_id"], "artifact_reference.json"
        ))
        artifact = load_verified_v2_artifact(
            context.store_root / "v2_datasets", ref.v2_dataset_artifact_id
        )
        provider = _provider(cfg, context.charter.payload.task_b_execution)
        identity = _study_run_identity(
            context, row, spec, cfg, ref, core, bundle, artifact_policy()
        )
        path = review_root / config.name / "context_export.csv"
        with path.open("r", encoding="utf-8", newline="") as stream:
            first = stream.readline()
            if not first.startswith("# "):
                raise PermissionError("Task B review export lacks its actual run header")
            header = json.loads(first[2:])
            rows = csv.reader(stream)
            next(rows)
            count = sum(1 for _ in rows)
        expected_header = {
            "configuration": config.name, "run_identity": identity,
            "menthorq_context_version": resolved.section.menthorq_context_version,
            "report_only": True, "archival": False,
            "source_file_sha256": dict(provider.source_file_sha256),
            "schema_version": provider.schema_version,
            "formula_version": provider.formula_version,
        }
        if header != expected_header or count != len(artifact.tables[RecordTable.ENTRY_CANDIDATE]):
            raise PermissionError("Task B review header, source hashes or row count do not match")


def pipeline_task_b_entry(charter, semantic, *, store_root):
    from ..fsm_audit_preparation import ChildFsmAuditEnvelope
    from ..manifest import read_repository_state
    from .charter import validate_charter
    from .child_replay import ChildAuditNeutralityEnvelope
    from .executors import REPO_ROOT
    from .runtime_source import strategy_core_repository_root
    from .strategy_executor import search_strategy_development_entry

    root = Path(store_root)

    def authority():
        validate_task_b_request(charter.payload, verify_files=True)
        validate_charter(charter.payload, as_of_utc=datetime.now(UTC).isoformat(), store_root=root)
        if semantic != task_b_semantic(charter):
            raise PermissionError("Task B pipeline differs from its exact no-fit plan")
        sources = (
            ("quant_lab", REPO_ROOT, ("src/alpha_lab", "scripts"),
             charter.payload.quant_lab_commit),
            ("strategy_core", strategy_core_repository_root(REPO_ROOT),
             ("src/strategy_core",), charter.payload.strategy_core_commit),
        )
        for name, repository, paths, approved_commit in sources:
            state = read_repository_state(name, repository, source_paths=paths)
            if state.head != approved_commit or state.dirty_status_sha256 != hashlib.sha256(
                b""
            ).hexdigest():
                raise PermissionError(
                    "Task B runtime source differs from the approved committed plan"
                )

    authority()
    strategy = search_strategy_development_entry(charter, store_root=root)
    contexts, policy = strategy["task_b_contexts"], strategy["task_b_artifact_policy"]

    def audit(row, _result):
        report = json.loads(load_sidecar_bytes(
            root, "core_replays", row["core_replay_id"], "task_b_companions.json"
        ))
        load_verified_envelope(
            root, "fsm_audit_companions", report["child_fsm_audit_id"], ChildFsmAuditEnvelope
        )
        neutrality = load_verified_envelope(
            root, "neutrality_reports", report["neutrality_report_id"], ChildAuditNeutralityEnvelope
        )
        if not neutrality.payload.passed or not report["verifier_link_resolves"]:
            raise PermissionError("Task B audit neutrality or exact drill-target validation failed")
        return report["child_fsm_audit_id"], report["neutrality_report_id"]

    def verify(context):
        authority()
        for kind in ("funded_accounts", "study_tables"):
            if _load(context, kind) is None:
                return {kind: "Task B evidence is missing"}
        _verify_review_exports(context, contexts, policy)
        return {}

    return PipelineWiring(
        identity_resolver=strategy["identity_resolver"], child_runner=strategy["child_runner"],
        cost_points=charter.payload.cost_policy.cost_points_round_turn, audit_builder=audit,
        research_authorization_check=authority,
        funded_account_builder=lambda context: _funded(context, contexts, policy),
        study_report_builder=lambda context: _reports(context, contexts, policy),
        study_report_verifier=verify,
    )
