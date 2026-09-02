"""HARDENING-BACKEND §4.4 (F-17) — the regime-stratified event summary is
aggregated EXTERNALLY.

The verified event-detail partitions stream past once; each partition's
group rows are written as a typed intermediate Parquet partition into an
attempt-local temp directory; the exact aggregation, the exact unique-path
counts and the cross-partition path-repetition refusal run in DuckDB under an
explicit memory limit and a spill directory; the final rows are
canonically sorted before hashing; the temp directory never survives; no
per-event or per-row Python structure grows with the artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from alpha_lab.agents.data_infra.ifvg.ml import regime_stratified_prop as prop_module
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_contracts import (
    ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1,
    EventRegimeSummaryBudget,
    RegimeAssignmentEvidenceRef,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_stratified_prop import (
    ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA,
    SUMMARY_ROW_KEYS,
    EventRegimeSummaryBudgetError,
    StratifiedPropResult,
    build_stratified_prop_body,
    event_detail_frame,
    read_account_event_regime_summary,
    summary_parquet_bytes,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
from alpha_lab.propsim.event_detail import EVENT_TYPE_PRECEDENCE
from alpha_lab.propsim.simulation import AccountSimulationEnvelope, AccountSimulationPayload

_CORE = "a" * 64
_EVIDENCE = RegimeAssignmentEvidenceRef(
    observation_granularity=ObservationGranularity.CANDIDATE_STAGE_ROW,
    regime_fit_ids=("1" * 64, "2" * 64),
    regime_fold_set_id="3" * 64,
    fold_schedule_id="4" * 64,
    regime_oos_assignment_id="5" * 64,
    assignment_table_sha256="6" * 64,
    assignment_schema_hash="7" * 64,
)
_TYPES = ("equity_update", "fee", "payout", "daily_halt")


def _simulation(root: Path, *, mode: str, seed: int) -> str:
    payload = AccountSimulationPayload(
        core_replay_id=_CORE,
        gross_trade_stream_hash="1" * 64,
        costed_evaluation_id="2" * 64,
        trade_path_bundle_id="3" * 64,
        trade_path_bundle_manifest_sha256="4" * 64,
        path_capability_report_id="5" * 64,
        account_policy_set_id="6" * 64,
        simulation_mode=mode,
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id="day_block_bootstrap_h90_v1" if mode.startswith("day") else None,
        stress_scenario_id=None,
        seed=seed,
        n_paths=1,
    )
    envelope = AccountSimulationEnvelope.from_payload(payload)
    save_or_reuse_envelope(root, "account_simulations", envelope)
    return envelope.account_simulation_id


def _trade_regimes() -> pd.DataFrame:
    rows = []
    for index in range(12):
        rows.append(
            {
                "trade_id": f"trade-{index:03d}",
                "valid": index % 4 != 3,
                "canonical_reporting_cluster_id": None if index % 4 == 3 else index % 3,
            }
        )
    return pd.DataFrame(rows)


def _record(ordinal: int, path: str, *, ts: str, event_type: str, trade: str | None, amount):
    payload = {}
    if event_type == "equity_update":
        payload = {"realized_delta": amount}
    elif event_type == "fee":
        payload = {"fee_kind": "monthly", "amount": amount}
    elif event_type == "payout":
        payload = {"trader_amount": amount}
    return {
        "event_id": hashlib.sha256(f"{path}:{ordinal}:{ts}:{event_type}".encode()).hexdigest(),
        "event_ts_utc": ts,
        "trading_day": ts[:10],
        "event_ordinal": ordinal,
        "path_instance_id": path,
        "account_id": "acct-1",
        "account_ordinal": 0,
        "firm_contract_id": "7" * 64,
        "account_phase": "funded",
        "source_trade_id": trade,
        "source_decision_id": None,
        "source_candidate_id": None,
        "source_setup_id": None,
        "source_path_event_id": None,
        "event_type": event_type,
        "event_order_policy_id": "test",
        "payload": payload,
    }


def _partitions(*, paths: int, events_per_path: int, partitions: int, path_offset: int = 0):
    """``partitions`` frames; a path lives inside ONE frame; keys repeat
    across partitions (the merge across partitions is exercised)."""

    frames: list[list[dict]] = [[] for _ in range(partitions)]
    ordinal = 0
    for path_index in range(paths):
        path = f"path-{path_index + path_offset:05d}"
        target = frames[path_index % partitions]
        for event_index in range(events_per_path):
            ordinal += 1
            event_type = _TYPES[(path_index + event_index) % len(_TYPES)]
            trade = (
                f"trade-{(path_index * 7 + event_index) % 13:03d}"
                if event_type in ("equity_update", "payout") and event_index % 3 != 2
                else None
            )
            hour = 10 + (event_index % 8)
            ts = f"2020-01-{6 + (event_index % 5):02d}T{hour:02d}:{path_index % 60:02d}:00+00:00"
            target.append(
                _record(
                    ordinal,
                    path,
                    ts=ts,
                    event_type=event_type,
                    trade=trade,
                    amount=float(((path_index * 31 + event_index * 17) % 200) - 100) / 4.0,
                )
            )
    return frames


def _loader_for(frames_by_simulation: dict[str, list[list[dict]]]):
    def _loader(_root, simulation_id):
        frames = frames_by_simulation.get(simulation_id)
        if frames is None:
            return None
        return (
            event_detail_frame(records, clock_policy_id="synthetic_path_clock_v1")
            for records in frames
        )

    return _loader


def _reference_summary(frames_by_simulation, trade_regimes) -> pd.DataFrame:
    """The in-memory reference: attribute every event, group over ALL events
    of a simulation, then sort by the documented key."""

    regime_of_trade = {
        str(row.trade_id): (
            int(row.canonical_reporting_cluster_id)
            if bool(row.valid) and pd.notna(row.canonical_reporting_cluster_id)
            else None
        )
        for row in trade_regimes.itertuples(index=False)
    }
    rows = []
    for simulation_id, frames in frames_by_simulation.items():
        events = pd.concat(
            [event_detail_frame(f, clock_policy_id="synthetic_path_clock_v1") for f in frames],
            ignore_index=True,
        )
        for record in events.to_dict(orient="records"):
            trade = record["source_trade_id"]
            if trade is None:
                regime, reason = None, "no_source_trade_synthetic_clock"
            elif trade not in regime_of_trade:
                regime, reason = None, "source_trade_not_in_regime_evidence"
            elif regime_of_trade[trade] is None:
                regime, reason = None, "source_trade_unassigned"
            else:
                regime, reason = regime_of_trade[trade], None
            rows.append(
                {
                    "account_simulation_id": simulation_id,
                    "path_instance_id": record["path_instance_id"],
                    "clock_policy_id": record["clock_policy_id"],
                    "regime_stratum": "unassigned" if regime is None else f"regime:{regime}",
                    "canonical_reporting_cluster_id": regime,
                    "unattributable_reason": reason,
                    "event_type": record["event_type"],
                    "event_precedence": int(record["event_precedence"]),
                    "event_ordinal": int(record["event_ordinal"]),
                    "event_ts_utc": record["event_ts_utc"],
                    "amount": record["amount"],
                }
            )
    frame = pd.DataFrame(rows)
    grouped = (
        frame.groupby(
            [
                "account_simulation_id",
                "path_instance_id",
                "clock_policy_id",
                "regime_stratum",
                "unattributable_reason",
                "event_type",
            ],
            dropna=False,
            sort=False,
        )
        .agg(
            canonical_reporting_cluster_id=("canonical_reporting_cluster_id", "first"),
            event_precedence=("event_precedence", "first"),
            event_count=("event_ordinal", "size"),
            amount_count=("amount", "count"),
            amount_total=("amount", "sum"),
            first_event_ordinal=("event_ordinal", "min"),
            last_event_ordinal=("event_ordinal", "max"),
            first_event_ts_utc=("event_ts_utc", "min"),
            last_event_ts_utc=("event_ts_utc", "max"),
        )
        .reset_index()
    )
    grouped["amount_sum"] = [
        float(total) if int(count) > 0 else None
        for total, count in zip(grouped["amount_total"], grouped["amount_count"], strict=True)
    ]
    grouped["unattributable_reason"] = [
        None if (isinstance(value, float) and pd.isna(value)) else value
        for value in grouped["unattributable_reason"]
    ]
    ordered = sorted(
        grouped.to_dict(orient="records"),
        key=lambda r: (
            r["account_simulation_id"],
            r["path_instance_id"],
            (1, 0)
            if r["canonical_reporting_cluster_id"] is None
            or pd.isna(r["canonical_reporting_cluster_id"])
            else (0, int(r["canonical_reporting_cluster_id"])),
            r["unattributable_reason"] or "",
            r["event_type"],
            r["clock_policy_id"],
        ),
    )
    return pd.DataFrame(ordered)


def test_external_aggregation_equals_the_in_memory_reference_and_is_deterministic(tmp_path):
    root = tmp_path / "store"
    sim_a = _simulation(root, mode="day_block_bootstrap", seed=1)
    sim_b = _simulation(root, mode="day_block_bootstrap", seed=2)
    frames = {
        sim_a: _partitions(paths=9, events_per_path=7, partitions=3),
        sim_b: _partitions(paths=5, events_per_path=4, partitions=2, path_offset=100),
    }
    trade_regimes = _trade_regimes()
    simulations = {
        sim_a: ("firm_a", "day_block_bootstrap"),
        sim_b: ("firm_b", "day_block_bootstrap"),
    }
    result = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations=simulations,
        trade_regimes=trade_regimes,
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
    )
    observed = result.summary.to_pandas()
    reference = _reference_summary(frames, trade_regimes)
    assert len(observed) == len(reference) == result.summary_rows == result.body.summary_rows
    assert list(observed["account_simulation_id"]) == list(reference["account_simulation_id"])
    assert list(observed["path_instance_id"]) == list(reference["path_instance_id"])
    assert list(observed["regime_stratum"]) == list(reference["regime_stratum"])
    assert list(observed["event_type"]) == list(reference["event_type"])
    assert [None if pd.isna(v) else v for v in observed["unattributable_reason"]] == list(
        reference["unattributable_reason"]
    )
    assert observed["event_count"].tolist() == reference["event_count"].astype(int).tolist()
    assert observed["first_event_ordinal"].tolist() == (
        reference["first_event_ordinal"].astype(int).tolist()
    )
    assert observed["last_event_ordinal"].tolist() == (
        reference["last_event_ordinal"].astype(int).tolist()
    )
    assert observed["first_event_ts_utc"].tolist() == reference["first_event_ts_utc"].tolist()
    assert observed["last_event_ts_utc"].tolist() == reference["last_event_ts_utc"].tolist()
    for got, want in zip(observed["amount_sum"], reference["amount_sum"], strict=True):
        if want is None:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(want, abs=1e-9)
    assert observed["event_precedence"].tolist() == [
        EVENT_TYPE_PRECEDENCE[t] for t in observed["event_type"]
    ]
    # body facts agree with the external counts
    body = result.body
    assert body.events_total == 9 * 7 + 5 * 4
    assert body.events_attributed == int(reference.loc[
        reference["canonical_reporting_cluster_id"].notna(), "event_count"
    ].sum())
    facts = result.detail["simulations"][sim_a]
    assert facts["partitions"] == 3 and facts["paths"] == 9 and facts["events_total"] == 63
    assert facts["summary_rows"] == int((reference["account_simulation_id"] == sim_a).sum())
    # deterministic bytes on repeat; the parsed bytes ARE the summary
    again = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations=simulations,
        trade_regimes=trade_regimes,
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
    )
    assert again.summary_bytes == result.summary_bytes and again.body == body
    assert read_account_event_regime_summary(result.summary_bytes).equals(result.summary)
    assert result.summary.schema.equals(ACCOUNT_EVENT_REGIME_SUMMARY_SCHEMA)
    # the summary table is parsed lazily from the bytes — no second in-memory copy
    assert isinstance(StratifiedPropResult.summary, property)
    assert list(SUMMARY_ROW_KEYS) == result.detail["summary"]["row_keys"]


def test_external_bytes_equal_the_single_write_parquet_of_the_table(tmp_path):
    """The row-group layout of the streamed writer equals ``pq.write_table``
    with ``row_group_size=65_536`` — even past one row group."""

    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=3)
    frames = {sim: _partitions(paths=17_000, events_per_path=4, partitions=4)}
    result = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations={sim: ("firm_a", "day_block_bootstrap")},
        trade_regimes=_trade_regimes(),
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
    )
    assert result.summary_rows > 65_536
    assert summary_parquet_bytes(result.summary) == result.summary_bytes


def test_a_path_repeated_across_partitions_is_refused_externally(tmp_path):
    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=4)
    first, second = _partitions(paths=4, events_per_path=3, partitions=2)
    repeated = [dict(record, event_ordinal=record["event_ordinal"] + 1000) for record in first[:3]]
    second = [*second, *repeated]
    for record in second[-3:]:
        record["event_id"] = hashlib.sha256(f"dup:{record['event_ordinal']}".encode()).hexdigest()
    with pytest.raises(ValueError, match="repeats a path_instance_id across partitions"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE,
            simulations={sim: ("firm_a", "day_block_bootstrap")},
            trade_regimes=_trade_regimes(),
            evidence=_EVIDENCE,
            loader=_loader_for({sim: [first, second]}),
        )


def test_row_budget_is_an_exact_external_count_before_publication(tmp_path):
    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=5)
    # the exact external count (never a per-partition upper bound) decides
    frames = {sim: _partitions(paths=6, events_per_path=6, partitions=3)}
    reference = _reference_summary(frames, _trade_regimes())
    exact = len(reference)
    fits = EventRegimeSummaryBudget(
        budget_id="fits_exactly", max_summary_rows=exact, max_published_bytes=50_000_000
    )
    result = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations={sim: ("firm_a", "day_block_bootstrap")},
        trade_regimes=_trade_regimes(),
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
        budget=fits,
    )
    assert result.summary_rows == exact
    too_small = EventRegimeSummaryBudget(
        budget_id="one_short", max_summary_rows=exact - 1, max_published_bytes=50_000_000
    )
    with pytest.raises(EventRegimeSummaryBudgetError, match="row budget"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE,
            simulations={sim: ("firm_a", "day_block_bootstrap")},
            trade_regimes=_trade_regimes(),
            evidence=_EVIDENCE,
            loader=_loader_for(frames),
            budget=too_small,
        )


def test_attempt_temp_directory_is_cleaned_on_success_and_on_refusal(tmp_path, monkeypatch):
    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=6)
    frames = {sim: _partitions(paths=4, events_per_path=3, partitions=2)}
    created: list[Path] = []
    real_mkdtemp = prop_module.tempfile.mkdtemp

    def _record(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(prop_module.tempfile, "mkdtemp", _record)
    build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations={sim: ("firm_a", "day_block_bootstrap")},
        trade_regimes=_trade_regimes(),
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
    )
    assert created and not any(path.exists() for path in created)
    created.clear()
    tiny = EventRegimeSummaryBudget(budget_id="tiny", max_summary_rows=1, max_published_bytes=10)
    with pytest.raises(EventRegimeSummaryBudgetError):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE,
            simulations={sim: ("firm_a", "day_block_bootstrap")},
            trade_regimes=_trade_regimes(),
            evidence=_EVIDENCE,
            loader=_loader_for(frames),
            budget=tiny,
        )
    assert created and not any(path.exists() for path in created)
    # nothing was published for the refused build
    assert not (root / "regime_stratified_reports").exists()


def test_duckdb_aggregation_runs_under_an_explicit_memory_limit_and_spill_directory(
    tmp_path, monkeypatch
):
    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=7)
    frames = {sim: _partitions(paths=4, events_per_path=3, partitions=2)}
    observed: list[dict] = []
    real_open = prop_module._open_aggregation_connection

    def _open(temp_root):
        connection = real_open(temp_root)
        settings = connection.execute(
            "SELECT current_setting('memory_limit'), current_setting('temp_directory'), "
            "current_setting('threads')"
        ).fetchone()
        observed.append(
            {
                "memory_limit": settings[0],
                "temp_directory": settings[1],
                "threads": int(settings[2]),
                "temp_root": Path(temp_root),
            }
        )
        return connection

    monkeypatch.setattr(prop_module, "_open_aggregation_connection", _open)
    build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations={sim: ("firm_a", "day_block_bootstrap")},
        trade_regimes=_trade_regimes(),
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
    )
    assert len(observed) == 1
    facts = observed[0]
    expected_mib = prop_module.SUMMARY_AGGREGATION_MEMORY_LIMIT_BYTES // (1024 * 1024)
    reported = facts["memory_limit"].replace(" ", "").upper()
    unit = reported.lstrip("0123456789.")
    value = float(reported[: len(reported) - len(unit)])
    assert unit == "MIB" and value == pytest.approx(expected_mib, abs=0.5)
    assert Path(facts["temp_directory"]).resolve().is_relative_to(facts["temp_root"].resolve())
    assert facts["threads"] == 1
    assert not facts["temp_root"].exists()
    assert json.dumps(ACCOUNT_EVENT_REGIME_SUMMARY_BUDGET_V1.model_dump(mode="json"))
    assert isinstance(pa.schema([]), pa.Schema)


def test_byte_budget_refuses_before_publication_with_a_generous_row_budget(
    tmp_path, monkeypatch
):
    """B-07: the byte budget is checked on the WRITTEN summary file (and on the
    empty-table bytes) before any publication — independently of the row
    budget, which is generous here."""

    root = tmp_path / "store"
    sim = _simulation(root, mode="day_block_bootstrap", seed=9)
    frames = {sim: _partitions(paths=6, events_per_path=6, partitions=3)}
    created: list[Path] = []
    real_mkdtemp = prop_module.tempfile.mkdtemp

    def _record(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(prop_module.tempfile, "mkdtemp", _record)
    tiny_bytes = EventRegimeSummaryBudget(
        budget_id="tiny_bytes", max_summary_rows=1_000_000, max_published_bytes=64
    )
    with pytest.raises(EventRegimeSummaryBudgetError, match="byte budget"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE,
            simulations={sim: ("firm_a", "day_block_bootstrap")},
            trade_regimes=_trade_regimes(),
            evidence=_EVIDENCE,
            loader=_loader_for(frames),
            budget=tiny_bytes,
        )
    assert created and not any(path.exists() for path in created)
    assert not (root / "regime_stratified_reports").exists()
    # the EMPTY-table branch (no event detail frames at all) checks its bytes too
    created.clear()
    with pytest.raises(EventRegimeSummaryBudgetError, match="byte budget"):
        build_stratified_prop_body(
            root=root,
            core_replay_id=_CORE,
            simulations={sim: ("firm_a", "day_block_bootstrap")},
            trade_regimes=_trade_regimes(),
            evidence=_EVIDENCE,
            loader=_loader_for({sim: []}),
            budget=tiny_bytes,
        )
    assert created and not any(path.exists() for path in created)
    # a budget that admits the bytes passes with the same generous row budget
    generous = EventRegimeSummaryBudget(
        budget_id="generous", max_summary_rows=1_000_000, max_published_bytes=50_000_000
    )
    result = build_stratified_prop_body(
        root=root,
        core_replay_id=_CORE,
        simulations={sim: ("firm_a", "day_block_bootstrap")},
        trade_regimes=_trade_regimes(),
        evidence=_EVIDENCE,
        loader=_loader_for(frames),
        budget=generous,
    )
    assert len(result.summary_bytes) > 64
