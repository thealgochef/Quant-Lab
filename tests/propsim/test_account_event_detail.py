"""R6.1 D15 — versioned, compressed, bounded prop-event detail (plan §6.G;
§9.1 ``test_prop_event_detail_capacity_and_exact_order``; §9.2
``tests/propsim/test_account_event_detail.py``).

The representation policy / storage policy / schema version / budget enter
the account- and portfolio-simulation identities; v2 publishes ZSTD Parquet
path blocks with exact event timestamps, trading days, clock policy, and
the total-order fields; every partition and the detail manifest carry
sha256 / bytes / rows / schema; a budget overrun fails before publication;
uniqueness is refused; ``none_v0`` artifacts are never widened. R6.1 safety
review S12: the writer is a store sidecar PRODUCER — one path block in
memory at a time, partitions streamed into the store's temporary
publication directory, byte-identical to the in-memory Parquet form; the
store re-hashes every produced file, refuses bookkeeping disagreements and
stray files, and verifies a producer's bytes on reuse.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from alpha_lab.agents.data_infra.ifvg.search.charter import SimulationProtocol
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    ProducedSidecar,
    SearchStoreError,
    envelope_destination,
    has_envelope,
    load_sidecar_bytes,
    save_or_reuse_envelope,
    write_produced_sidecar,
)
from alpha_lab.propsim.account import AccountPolicySetPayload, AccountTrade
from alpha_lab.propsim.calendar import BOOTSTRAP_CLOCK_POLICY, HISTORICAL_CLOCK_POLICY
from alpha_lab.propsim.event_detail import (
    EVENT_DETAIL_BUDGET_V1,
    EVENT_DETAIL_MANIFEST_SIDECAR,
    EVENT_DETAIL_POLICY_NONE,
    EVENT_DETAIL_POLICY_PARQUET_V2,
    EVENT_DETAIL_SCHEMA,
    EVENT_DETAIL_SCHEMA_HASH,
    EVENT_DETAIL_SCHEMA_VERSION_V2,
    EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2,
    EventDetailBudget,
    EventDetailBudgetError,
    EventDetailIntegrityError,
    EventDetailUnavailableError,
    account_event_detail_producer,
    build_account_event_detail,
    event_detail_identity_fields,
    load_account_event_detail,
    load_account_event_detail_manifest,
    partition_sidecar_name,
)
from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM
from alpha_lab.propsim.prop_metrics import build_payout_reliability_vector
from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
from alpha_lab.propsim.search_bridge import make_prop_simulator, persist_account_simulation
from alpha_lab.propsim.simulation import (
    AccountSimulationPayload,
    PortfolioLeg,
    PortfolioSimulationPayload,
    run_account_simulation,
)
from alpha_lab.propsim.trade_path import (
    build_closed_trade_artifact,
    build_trade_path_bundle,
    evaluate_path_capabilities,
)
from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY

_FIRM_ID = canonical_contract_sha256(SYNTHETIC_FIXTURE_FIRM)
_POLICY_SET = AccountPolicySetPayload(
    firm_contract_id=_FIRM_ID,
    risk_policy_id=canonical_contract_sha256(FIXED_ONE_NQ_RISK_POLICY),
    withdrawal_policy_id=canonical_contract_sha256(REQUEST_MAX_AT_ELIGIBILITY),
    replacement_policy="none",
    max_replacements=0,
    clock_policy_id="historical_calendar_clock_v1",
)
_CORE = "a" * 64
_STREAM = "b" * 64
_STORE = "account_simulations"


def _trade(day: str, points: float, slot: int = 0) -> AccountTrade:
    return AccountTrade(
        day=date.fromisoformat(day),
        entry_ts_utc=f"{day}T1{4 + slot}:00:00+00:00",
        resolution_ts_utc=f"{day}T1{4 + slot}:05:00+00:00",
        points=points,
        risk_points=10.0,
        mfe_pts=None,
        mae_pts=None,
        trade_id=f"trade-{day}-{slot}",
        candidate_id=f"cand-{day}-{slot}",
    )


_DAY_BLOCKS = [
    (date(2026, 1, 13), [_trade("2026-01-13", 40.0), _trade("2026-01-13", -5.0, 1)]),
    (date(2026, 1, 14), [_trade("2026-01-14", -20.0)]),
    (date(2026, 1, 15), [_trade("2026-01-15", 30.0)]),
]


def _evidence():
    trades = [trade for _day, block in _DAY_BLOCKS for trade in block]
    artifacts = tuple(
        build_closed_trade_artifact(
            core_replay_id=_CORE,
            trade_id=trade.trade_id,
            entry_ts_utc=trade.entry_ts_utc,
            resolution_ts_utc=trade.resolution_ts_utc,
            entry_price=20_000.0,
            exit_price=20_000.0 + trade.points,
        )
        for trade in trades
    )
    bundle = build_trade_path_bundle(
        artifacts,
        gross_trade_stream_hash=_STREAM,
        ordered_trade_ids=tuple(trade.trade_id for trade in trades),
    )
    report = evaluate_path_capabilities(
        bundle, SYNTHETIC_FIXTURE_FIRM.rule_path_requirements, artifacts=artifacts
    )
    return artifacts, bundle, report


_ARTIFACTS, _BUNDLE, _REPORT = _evidence()


def _payload(policy: str = EVENT_DETAIL_POLICY_PARQUET_V2, **overrides) -> AccountSimulationPayload:
    base = dict(
        core_replay_id=_CORE,
        gross_trade_stream_hash=_STREAM,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id=_BUNDLE.trade_path_bundle_id,
        trade_path_bundle_manifest_sha256=_BUNDLE.manifest_payload_sha256,
        path_capability_report_id=canonical_contract_sha256(_REPORT),
        account_policy_set_id=canonical_contract_sha256(_POLICY_SET),
        simulation_mode="historical_closed_trade",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id=None,
        stress_scenario_id=None,
        seed=42,
        n_paths=1,
    )
    base.update(event_detail_identity_fields(policy))
    base.update(overrides)
    return AccountSimulationPayload(**base)


def _run(payload: AccountSimulationPayload):
    return run_account_simulation(
        payload,
        firm=SYNTHETIC_FIXTURE_FIRM,
        risk_policy=FIXED_ONE_NQ_RISK_POLICY,
        withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
        policy_set=_POLICY_SET,
        day_blocks=_DAY_BLOCKS,
        bundle=_BUNDLE,
        artifacts=_ARTIFACTS,
        capability_report=_REPORT,
    )


def _bootstrap_payload(n_paths: int, policy: str = EVENT_DETAIL_POLICY_PARQUET_V2, **overrides):
    return _payload(
        policy,
        simulation_mode="day_block_bootstrap",
        bootstrap_protocol_id="day_block_bootstrap_h3_v1",
        n_paths=n_paths,
        **overrides,
    )


def _persist(root, run):
    vector = build_payout_reliability_vector(run.walk_results)
    persist_account_simulation(root, run, vector=vector, risk_policy_label="fixture")
    return run.envelope.account_simulation_id


def _store_manifest(root, simulation_id) -> dict:
    return json.loads(
        (envelope_destination(root, _STORE, simulation_id) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )


def _fresh_dir(base: Path, name: str) -> Path:
    directory = base / name
    directory.mkdir(parents=True)
    return directory


def _build(run, directory: Path, *, clock=BOOTSTRAP_CLOCK_POLICY, budget=EVENT_DETAIL_BUDGET_V1):
    return build_account_event_detail(
        run.walk_results,
        run.path_records,
        clock_policy_id=clock.policy_id,
        budget=budget,
        event_order_policy_id="prop_account_event_order_v1",
        directory=directory,
    )


def _no_scratch_residue(root) -> bool:
    return not any(name.startswith(".") for name in os.listdir(root / _STORE))


# ── identity ─────────────────────────────────────────────────────────────────


def test_policy_representation_and_budget_enter_the_simulation_identities():
    none = _payload(EVENT_DETAIL_POLICY_NONE)
    v2 = _payload(EVENT_DETAIL_POLICY_PARQUET_V2)
    assert canonical_contract_sha256(none) != canonical_contract_sha256(v2)
    smaller = _payload(
        EVENT_DETAIL_POLICY_PARQUET_V2,
        event_detail_budget=EventDetailBudget(
            budget_id="test_budget",
            max_event_detail_rows=10,
            max_published_bytes=10,
            path_block_size=2,
        ),
    )
    assert canonical_contract_sha256(smaller) != canonical_contract_sha256(v2)
    assert v2.event_detail_budget == EVENT_DETAIL_BUDGET_V1
    assert v2.event_detail_storage_policy_id == EVENT_DETAIL_STORAGE_ZSTD_PARQUET_V2
    assert v2.event_detail_schema_version == EVENT_DETAIL_SCHEMA_VERSION_V2
    # the four fields are ONE coherent identity — partial combinations refuse
    with pytest.raises(ValueError, match="requires the storage policy"):
        _payload(EVENT_DETAIL_POLICY_PARQUET_V2, event_detail_storage_policy_id="none")
    with pytest.raises(ValueError, match="schema_version"):
        _payload(EVENT_DETAIL_POLICY_PARQUET_V2, event_detail_schema_version=0)
    with pytest.raises(ValueError, match="requires a registered budget"):
        _payload(EVENT_DETAIL_POLICY_PARQUET_V2, event_detail_budget=None)
    with pytest.raises(ValueError, match="none_v0 carries no storage policy"):
        _payload(EVENT_DETAIL_POLICY_NONE, event_detail_budget=EVENT_DETAIL_BUDGET_V1)
    with pytest.raises(ValueError, match="unregistered event_detail_persistence_policy_id"):
        event_detail_identity_fields("account_event_detail_json_v9")
    # the portfolio identity carries the same fields
    portfolio_base = dict(
        core_replay_id=_CORE,
        gross_trade_stream_hash=_STREAM,
        costed_evaluation_id="c" * 64,
        trade_path_bundle_id="d" * 64,
        trade_path_bundle_manifest_sha256="e" * 64,
        path_capability_report_id="f" * 64,
        portfolio_policy_id="2" * 64,
        resolved_legs=(
            PortfolioLeg(leg_id="leg-1", account_policy_set_id="1" * 64, n_accounts=2),
        ),
        simulation_mode="historical_closed_trade",
        intrabar_scenario_policy_id=None,
        bootstrap_protocol_id=None,
        stress_scenario_id=None,
        seed=42,
        n_paths=1,
    )
    portfolio_none = PortfolioSimulationPayload(**portfolio_base)
    portfolio_v2 = PortfolioSimulationPayload(
        **portfolio_base, **event_detail_identity_fields(EVENT_DETAIL_POLICY_PARQUET_V2)
    )
    assert portfolio_none.event_detail_persistence_policy_id == EVENT_DETAIL_POLICY_NONE
    assert canonical_contract_sha256(portfolio_none) != canonical_contract_sha256(portfolio_v2)
    with pytest.raises(ValueError, match="requires the storage policy"):
        PortfolioSimulationPayload(
            **{
                **portfolio_base,
                "event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_PARQUET_V2,
            }
        )


def test_simulation_protocol_carries_the_policy_with_the_r3_default():
    protocol = SimulationProtocol(
        modes=("historical_closed_trade",),
        stress_scenario_ids=(),
        trade_path_capability_policy_id="path_capability_policy_v1",
        clock_policy_id="simulated_clock_v1",
    )
    assert protocol.event_detail_persistence_policy_id == EVENT_DETAIL_POLICY_NONE
    v2 = protocol.model_copy(
        update={"event_detail_persistence_policy_id": EVENT_DETAIL_POLICY_PARQUET_V2}
    )
    assert canonical_contract_sha256(v2) != canonical_contract_sha256(protocol)
    with pytest.raises(ValueError):
        SimulationProtocol(
            modes=("historical_closed_trade",),
            stress_scenario_ids=(),
            trade_path_capability_policy_id="path_capability_policy_v1",
            clock_policy_id="simulated_clock_v1",
            event_detail_persistence_policy_id="json_v9",
        )


# ── partitions, manifest, exact order ────────────────────────────────────────


def test_partitions_and_manifest_carry_sha256_bytes_rows_schema(tmp_path):
    root = tmp_path / "store"
    run = _run(_bootstrap_payload(600))
    simulation_id = _persist(root, run)
    detail = load_account_event_detail_manifest(root, simulation_id)
    assert detail["event_detail_persistence_policy_id"] == EVENT_DETAIL_POLICY_PARQUET_V2
    assert detail["schema_hash"] == EVENT_DETAIL_SCHEMA_HASH
    assert detail["compression"] == "zstd"
    assert detail["path_count"] == 600
    assert [entry["path_block_id"] for entry in detail["partitions"]] == [0, 1, 2]
    assert detail["partitions"][0]["path_ordinal_min"] == 0
    assert detail["partitions"][0]["path_ordinal_max"] == 249
    assert detail["partitions"][2]["path_ordinal_max"] == 599
    store_manifest = _store_manifest(root, simulation_id)
    listed = {entry["path"]: entry for entry in store_manifest["artifacts"]}
    assert EVENT_DETAIL_MANIFEST_SIDECAR in listed
    total_rows = 0
    for entry in detail["partitions"]:
        stored = listed[entry["name"]]
        assert stored["sha256"] == entry["sha256"]
        assert stored["bytes"] == entry["bytes"]
        assert entry["rows"] > 0
        assert entry["schema_hash"] == EVENT_DETAIL_SCHEMA_HASH
        total_rows += entry["rows"]
    assert total_rows == detail["total_rows"] == sum(len(r.events) for r in run.walk_results)
    assert detail["total_bytes"] == sum(entry["bytes"] for entry in detail["partitions"])
    assert detail["clock_policy_id"] == BOOTSTRAP_CLOCK_POLICY.policy_id
    assert detail["event_order_policy_id"] == "prop_account_event_order_v1"
    summary = json.loads(load_sidecar_bytes(root, _STORE, simulation_id, "walk_summary.json"))
    assert summary["event_detail_persistence_policy_id"] == EVENT_DETAIL_POLICY_PARQUET_V2
    assert summary["event_detail_partition_count"] == 3
    assert summary["event_detail_rows"] == total_rows
    # bootstrap simulations carry NO historical account_events.json (audit JSON is historical-only)
    assert "account_events.json" not in listed
    # the reader yields the partitions in block order with the contract schema
    frames = list(load_account_event_detail(root, simulation_id))
    assert [int(frame["path_block_id"].iloc[0]) for frame in frames] == [0, 1, 2]
    assert list(frames[0].columns) == [field.name for field in EVENT_DETAIL_SCHEMA]
    assert sum(len(frame) for frame in frames) == total_rows


def test_prop_event_detail_capacity_and_exact_order(tmp_path):
    """§9.1: a high path count stays within the registered budget (or fails
    before publish); timestamp / clock / order fields reproduce the walk's
    total chronology exactly."""

    root = tmp_path / "store"
    run = _run(_bootstrap_payload(1200))
    simulation_id = _persist(root, run)
    detail = load_account_event_detail_manifest(root, simulation_id)
    assert detail["total_rows"] <= EVENT_DETAIL_BUDGET_V1.max_event_detail_rows
    assert detail["total_bytes"] <= EVENT_DETAIL_BUDGET_V1.max_published_bytes
    assert len(detail["partitions"]) == 5  # 1200 / 250 → blocks 0..4
    table = pd.concat(list(load_account_event_detail(root, simulation_id)), ignore_index=True)
    # the walk's emission order IS the total order (path_ordinal, event_ordinal)
    expected = [
        (int(record.draw_ordinal), int(event.event_ordinal), event.event_id)
        for record, result in zip(run.path_records, run.walk_results, strict=True)
        for event in result.events
    ]
    expected.sort()
    observed = list(
        zip(
            table["path_ordinal"].astype(int),
            table["event_ordinal"].astype(int),
            table["event_id"],
            strict=True,
        )
    )
    assert observed == expected
    assert table["event_id"].is_unique
    assert not table.duplicated(["path_instance_id", "event_ordinal"]).any()
    assert (table["path_block_id"] == table["path_ordinal"] // 250).all()
    # exact timestamps: the verbatim ISO string and its UTC nanosecond value agree
    parsed = pd.to_datetime(table["event_ts_utc"], utc=True).astype("int64")
    assert (parsed.to_numpy() == table["event_ts_ns"].to_numpy()).all()
    # the synthetic clock is DESCRIPTIVE: every row names the bootstrap clock
    # policy and its trading day is the synthetic day the walk played
    assert (table["clock_policy_id"] == BOOTSTRAP_CLOCK_POLICY.policy_id).all()
    assert table["trading_day"].str.startswith("2020-01").all()
    # attribution is through the source trade / candidate, never the clock
    fee_or_halt = table["event_type"].isin(("fee", "daily_halt"))
    assert table.loc[~fee_or_halt, "source_trade_id"].notna().any()
    assert table.loc[table["source_trade_id"].notna(), "source_candidate_id"].str.startswith(
        "cand-"
    ).all()
    assert set(table["event_precedence"].unique()) <= set(range(8))
    # a tiny ROW budget fails at preflight — nothing is published
    tiny_rows = _bootstrap_payload(
        16,
        event_detail_budget=EventDetailBudget(
            budget_id="tiny_rows",
            max_event_detail_rows=5,
            max_published_bytes=EVENT_DETAIL_BUDGET_V1.max_published_bytes,
            path_block_size=250,
        ),
    )
    tiny_run = _run(tiny_rows)
    with pytest.raises(EventDetailBudgetError, match="preflight"):
        _persist(root, tiny_run)
    assert not has_envelope(root, _STORE, tiny_run.envelope.account_simulation_id)
    # a tiny BYTE budget fails while streaming — nothing is published either
    tiny_bytes = _bootstrap_payload(
        16,
        event_detail_budget=EventDetailBudget(
            budget_id="tiny_bytes",
            max_event_detail_rows=EVENT_DETAIL_BUDGET_V1.max_event_detail_rows,
            max_published_bytes=64,
            path_block_size=4,
        ),
    )
    tiny_bytes_run = _run(tiny_bytes)
    with pytest.raises(EventDetailBudgetError, match="streaming overrun"):
        _persist(root, tiny_bytes_run)
    assert not has_envelope(root, _STORE, tiny_bytes_run.envelope.account_simulation_id)
    assert not any(
        name.startswith(".") for name in os.listdir(root / _STORE)
    ), "no temporary / partial artifact survives a refused publication"


def test_historical_rows_carry_the_played_trading_day_and_match_the_audit_json(tmp_path):
    root = tmp_path / "store"
    run = _run(_payload())
    simulation_id = _persist(root, run)
    frames = list(load_account_event_detail(root, simulation_id))
    assert len(frames) == 1
    table = frames[0]
    assert (table["clock_policy_id"] == HISTORICAL_CLOCK_POLICY.policy_id).all()
    assert (table["path_ordinal"] == 0).all()
    played = table["trading_day"].dropna().unique().tolist()
    assert set(played) <= {"2026-01-13", "2026-01-14", "2026-01-15"}
    # only an account-open fee emitted before any day may lack a trading day
    assert table["trading_day"].notna().sum() >= len(table) - 1
    # the audit JSON (historical modes) and the Parquet detail describe the SAME stream
    audit = json.loads(load_sidecar_bytes(root, _STORE, simulation_id, "account_events.json"))
    assert [event["event_id"] for event in sorted(audit, key=lambda e: e["event_ordinal"])] == (
        table["event_id"].tolist()
    )
    assert all("trading_day" in event for event in audit)
    # amounts follow the registered projection (fee → amount; equity_update → realized delta)
    fee_rows = table[table["event_type"] == "fee"]
    if len(fee_rows):
        assert fee_rows["amount"].notna().all()
    halts = table[table["event_type"].isin(("phase_transition", "daily_halt"))]
    assert halts["amount"].isna().all()


# ── uniqueness ───────────────────────────────────────────────────────────────


def test_uniqueness_and_order_violations_are_refused(tmp_path):
    run = _run(_bootstrap_payload(3))
    budget = EVENT_DETAIL_BUDGET_V1
    # a duplicated path record (same draw ordinal twice)
    records = run.path_records
    with pytest.raises(EventDetailIntegrityError, match="repeat a draw ordinal"):
        build_account_event_detail(
            run.walk_results[:2],
            (records[0], records[0]),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=budget,
            event_order_policy_id="prop_account_event_order_v1",
            directory=_fresh_dir(tmp_path, "dup_ordinal"),
        )
    # a duplicated event inside one path (same event id AND ordinal)
    first = run.walk_results[0]
    doubled = dataclasses.replace(first, events=first.events + first.events[:1])
    with pytest.raises(EventDetailIntegrityError, match="not strictly increasing"):
        build_account_event_detail(
            (doubled,),
            (records[0],),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=budget,
            event_order_policy_id="prop_account_event_order_v1",
            directory=_fresh_dir(tmp_path, "doubled"),
        )
    # an event id repeated across paths
    second = run.walk_results[1]
    forged = second.events[0].model_copy(update={"event_id": first.events[0].event_id})
    replaced = dataclasses.replace(second, events=(forged,) + second.events[1:])
    with pytest.raises(EventDetailIntegrityError, match="duplicate event id"):
        build_account_event_detail(
            (first, replaced),
            records[:2],
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=budget,
            event_order_policy_id="prop_account_event_order_v1",
            directory=_fresh_dir(tmp_path, "forged_same_block"),
        )
    # an event id repeated ACROSS path blocks (block 0 vs block 1 under a
    # 1-path block size) is caught by the whole-artifact index — also before
    # any publication
    tiny_blocks = EventDetailBudget(
        budget_id="one_path_blocks",
        max_event_detail_rows=budget.max_event_detail_rows,
        max_published_bytes=budget.max_published_bytes,
        path_block_size=1,
    )
    with pytest.raises(EventDetailIntegrityError, match="duplicate event id across path blocks"):
        build_account_event_detail(
            (first, replaced),
            records[:2],
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=tiny_blocks,
            event_order_policy_id="prop_account_event_order_v1",
            directory=_fresh_dir(tmp_path, "forged_across_blocks"),
        )
    # walk results and path records must pair one-to-one
    with pytest.raises(EventDetailIntegrityError, match="disagree in length"):
        build_account_event_detail(
            run.walk_results,
            records[:1],
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=budget,
            event_order_policy_id="prop_account_event_order_v1",
            directory=_fresh_dir(tmp_path, "length"),
        )
    # the publication directory must exist (the store hands the writer its own)
    with pytest.raises(ValueError, match="existing publication directory"):
        _build(run, tmp_path / "missing")


# ── none_v0 artifacts are never widened; the reader verifies ────────────────


def test_none_v0_artifacts_are_never_widened(tmp_path):
    root = tmp_path / "store"
    run = _run(_payload(EVENT_DETAIL_POLICY_NONE))
    simulation_id = _persist(root, run)
    listed = {entry["path"] for entry in _store_manifest(root, simulation_id)["artifacts"]}
    assert not any(name.startswith("event_detail_") for name in listed)
    assert "account_events.json" in listed  # the R3–R6 audit JSON is unchanged
    with pytest.raises(EventDetailUnavailableError, match="evidence_not_persisted"):
        list(load_account_event_detail(root, simulation_id))
    # re-publishing the SAME identity with detail sidecars is refused by the
    # immutable store (no widening); the id itself cannot carry v2 (the
    # policy is inside the identity, so v2 mints a different id)
    producer = account_event_detail_producer(
        run.walk_results,
        run.path_records,
        clock_policy_id=HISTORICAL_CLOCK_POLICY.policy_id,
        budget=EVENT_DETAIL_BUDGET_V1,
        event_order_policy_id="prop_account_event_order_v1",
    )
    with pytest.raises(SearchStoreError, match="DIFFERENT sidecar content"):
        save_or_reuse_envelope(root, _STORE, run.envelope, sidecar_producer=producer)
    assert {entry["path"] for entry in _store_manifest(root, simulation_id)["artifacts"]} == listed
    assert _no_scratch_residue(root), "the reuse-verification scratch directory never survives"
    v2_id = _persist(root, _run(_payload(EVENT_DETAIL_POLICY_PARQUET_V2)))
    assert v2_id != simulation_id
    assert len(list(load_account_event_detail(root, v2_id))) == 1


def test_reader_verifies_bytes_rows_and_schema_before_yielding(tmp_path):
    root = tmp_path / "store"
    run = _run(_bootstrap_payload(300))
    simulation_id = _persist(root, run)
    # idempotent re-publication of the identical detail reuses the entry
    _persist(root, run)
    # relocation: the whole store moves and still verifies
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    assert len(list(load_account_event_detail(moved, simulation_id))) == 2
    # a tampered partition fails closed at the store-manifest hash
    directory = envelope_destination(root, _STORE, simulation_id)
    partition = directory / partition_sidecar_name(1)
    original = partition.read_bytes()
    partition.write_bytes(original[:-8] + b"\x00" * 8)
    with pytest.raises(SearchStoreError):
        list(load_account_event_detail(root, simulation_id))
    partition.write_bytes(original)
    assert len(list(load_account_event_detail(root, simulation_id))) == 2
    # a tampered detail manifest fails closed too (its bytes are store-hashed)
    manifest_path = directory / EVENT_DETAIL_MANIFEST_SIDECAR
    kept = manifest_path.read_bytes()
    detail = json.loads(kept.decode("utf-8"))
    detail["partitions"][0]["rows"] += 1
    manifest_path.write_text(json.dumps(detail, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SearchStoreError):
        list(load_account_event_detail(root, simulation_id))
    manifest_path.write_bytes(kept)


# ── the search bridge persists v2 detail for every mode ─────────────────────


def test_search_bridge_persists_v2_detail_for_every_mode(tmp_path):
    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
    from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS, make_resolved_trades_frame
    from tests.agents.ifvg_search.pipeline_fixture import firm_simulation_specs

    frame = make_resolved_trades_frame(SYNTHETIC_DAYS)
    result = SimpleNamespace(
        tables={RecordTable.EXECUTED_TRADE: frame},
        gross_trade_stream_hash=canonical_contract_sha256({"bridge": "detail"}),
    )
    outcome = SimpleNamespace()
    outcome.core_replay_id = "a" * 64
    store_root = tmp_path / "store"
    with pytest.raises(ValueError, match="unregistered event_detail_persistence_policy_id"):
        make_prop_simulator(
            firm_simulation_specs(),
            tick_size=0.25,
            costed_evaluation_id_for=lambda core_id: canonical_contract_sha256({"c": core_id}),
            event_detail_persistence_policy_id="json_v9",
        )
    simulator = make_prop_simulator(
        firm_simulation_specs(),
        tick_size=0.25,
        costed_evaluation_id_for=lambda core_id: canonical_contract_sha256({"c": core_id}),
        simulation_modes=("historical_closed_trade", "day_block_bootstrap"),
        bootstrap_protocol_id="day_block_bootstrap_h90_v1",
        n_paths=16,
        store_root=store_root,
        event_detail_persistence_policy_id=EVENT_DETAIL_POLICY_PARQUET_V2,
    )
    vectors = simulator(outcome=outcome, result=result)
    assert len(vectors) == 2
    simulation_ids = [name for name in os.listdir(store_root / _STORE) if len(name) == 64]
    assert len(simulation_ids) == 2
    modes = set()
    for simulation_id in simulation_ids:
        detail = load_account_event_detail_manifest(store_root, simulation_id)
        summary = json.loads(
            load_sidecar_bytes(store_root, _STORE, simulation_id, "walk_summary.json")
        )
        modes.add(summary["simulation_mode"])
        frames = list(load_account_event_detail(store_root, simulation_id))
        assert frames and sum(len(f) for f in frames) == detail["total_rows"]
        if summary["simulation_mode"] == "historical_closed_trade":
            audit = json.loads(
                load_sidecar_bytes(store_root, _STORE, simulation_id, "account_events.json")
            )
            assert {e["event_id"] for e in audit} == set(frames[0]["event_id"])
            assert detail["clock_policy_id"] == HISTORICAL_CLOCK_POLICY.policy_id
        else:
            assert detail["path_count"] == 16
            assert detail["clock_policy_id"] == BOOTSTRAP_CLOCK_POLICY.policy_id
    assert modes == {"historical_closed_trade", "day_block_bootstrap"}
    # the R3 default (none_v0) is byte-preserved: no detail sidecars, no policy churn
    plain_root = tmp_path / "plain"
    plain = make_prop_simulator(
        firm_simulation_specs(),
        tick_size=0.25,
        costed_evaluation_id_for=lambda core_id: canonical_contract_sha256({"c": core_id}),
        store_root=plain_root,
    )
    plain(outcome=outcome, result=result)
    for simulation_id in (name for name in os.listdir(plain_root / _STORE) if len(name) == 64):
        manifest = _store_manifest(plain_root, simulation_id)
        listed = {entry["path"] for entry in manifest["artifacts"]}
        assert not any(name.startswith("event_detail_") for name in listed)
        with pytest.raises(EventDetailUnavailableError):
            list(load_account_event_detail(plain_root, simulation_id))


# ── S12: the writer is a streaming store producer ────────────────────────────


def test_streaming_writer_is_deterministic_and_reproduces_the_in_memory_parquet(tmp_path):
    run = _run(_bootstrap_payload(600))
    first = _build(run, _fresh_dir(tmp_path, "first"))
    second = _build(run, _fresh_dir(tmp_path, "second"))
    assert first.manifest == second.manifest
    assert [dataclasses.astuple(r) for r in first.produced()] == [
        dataclasses.astuple(r) for r in second.produced()
    ]
    assert first.partition_count == 3 and first.total_rows == first.manifest["total_rows"]
    # the records the store lists are exactly the bytes on disk (streamed hash)
    for record in first.produced():
        data = (first.directory / record.name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == record.sha256 and len(data) == record.bytes
        assert data == (second.directory / record.name).read_bytes()
    assert {entry["name"] for entry in first.manifest["partitions"]} == {
        r.name for r in first.partitions
    }
    assert json.loads((first.directory / EVENT_DETAIL_MANIFEST_SIDECAR).read_bytes()) == (
        first.manifest
    )
    # every partition file IS the in-memory Parquet form of its table (same
    # writer settings): the streaming path changed the memory profile only
    for record in first.partitions:
        path = first.directory / record.name
        table = pq.read_table(path)
        sink = io.BytesIO()
        pq.write_table(table, sink, compression="zstd", row_group_size=65_536)
        assert sink.getvalue() == path.read_bytes()
        assert table.schema.equals(EVENT_DETAIL_SCHEMA)
    # nothing but the declared files exists in the directory
    assert set(os.listdir(first.directory)) == {r.name for r in first.produced()}


def test_writer_streams_one_path_block_at_a_time_and_refuses_early(tmp_path, monkeypatch):
    from alpha_lab.propsim import event_detail as module

    run = _run(_bootstrap_payload(600))
    written: list[tuple[int, int]] = []
    real_write = module._write_parquet

    def _spy(table, path):
        # the block handed to the writer never exceeds ONE path block of rows
        block_ids = set(table.column("path_block_id").to_pylist())
        assert len(block_ids) == 1
        written.append((block_ids.pop(), table.num_rows))
        real_write(table, path)

    monkeypatch.setattr(module, "_write_parquet", _spy)
    bundle = _build(run, _fresh_dir(tmp_path, "blocks"))
    assert [block for block, _rows in written] == [0, 1, 2]
    assert sum(rows for _block, rows in written) == bundle.total_rows
    assert max(rows for _block, rows in written) < bundle.total_rows
    # a byte overrun at block k refuses right there: later blocks are never built
    written.clear()
    small = EventDetailBudget(
        budget_id="stop_at_block_1",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V1.max_event_detail_rows,
        max_published_bytes=bundle.manifest["partitions"][0]["bytes"] + 1,
        path_block_size=250,
    )
    directory = _fresh_dir(tmp_path, "overrun")
    with pytest.raises(EventDetailBudgetError, match="at path block 1"):
        _build(run, directory, budget=small)
    assert [block for block, _rows in written] == [0, 1]
    assert sorted(os.listdir(directory)) == [partition_sidecar_name(0), partition_sidecar_name(1)]
    assert not (directory / EVENT_DETAIL_MANIFEST_SIDECAR).exists()


def test_store_refuses_producer_bookkeeping_lies_stray_and_reserved_files(tmp_path):
    root = tmp_path / "store"
    run = _run(_bootstrap_payload(4))
    simulation_id = run.envelope.account_simulation_id

    def _honest(directory):
        return (write_produced_sidecar(directory, "detail.bin", b"honest bytes"),)

    def _wrong_hash(directory):
        record = _honest(directory)[0]
        return (dataclasses.replace(record, sha256="0" * 64),)

    def _wrong_size(directory):
        record = _honest(directory)[0]
        return (dataclasses.replace(record, bytes=record.bytes + 1),)

    def _ghost(directory):
        return (ProducedSidecar(name="ghost.bin", sha256="0" * 64, bytes=1),)

    def _stray(directory):
        (directory / "undeclared.bin").write_bytes(b"stray")
        return _honest(directory)

    def _reserved(directory):
        # claims the extra_files sidecar's name (the store refuses the claim
        # before it even looks for a file)
        return (ProducedSidecar(name="walk_summary.json", sha256="0" * 64, bytes=2),)

    def _bad_name(directory):
        return (ProducedSidecar(name="../escape.bin", sha256="0" * 64, bytes=1),)

    def _not_records(directory):
        return ("detail.bin",)

    cases = [
        (_wrong_hash, "disagrees with the bytes on disk"),
        (_wrong_size, "disagrees with the bytes on disk"),
        (_ghost, "wrote no such file"),
        (_stray, "undeclared files"),
        (_reserved, "reserved name"),
        (_bad_name, "invalid sidecar file name"),
        (_not_records, "must return ProducedSidecar"),
    ]
    for producer, reason in cases:
        with pytest.raises(SearchStoreError, match=reason):
            save_or_reuse_envelope(
                root,
                _STORE,
                run.envelope,
                extra_files={"walk_summary.json": b"{}"},
                sidecar_producer=producer,
            )
        assert not has_envelope(root, _STORE, simulation_id), reason
        assert _no_scratch_residue(root), reason
    # the honest producer publishes; its file is manifest-listed like extra_files
    save_or_reuse_envelope(
        root,
        _STORE,
        run.envelope,
        extra_files={"walk_summary.json": b"{}"},
        sidecar_producer=_honest,
    )
    listed = {entry["path"]: entry for entry in _store_manifest(root, simulation_id)["artifacts"]}
    assert listed["detail.bin"]["sha256"] == hashlib.sha256(b"honest bytes").hexdigest()
    assert listed["detail.bin"]["bytes"] == len(b"honest bytes")
    assert load_sidecar_bytes(root, _STORE, simulation_id, "detail.bin") == b"honest bytes"
    assert [entry["path"] for entry in _store_manifest(root, simulation_id)["artifacts"]] == [
        "envelope.json",
        "detail.bin",
        "walk_summary.json",
    ]


def test_reuse_with_a_producer_verifies_the_produced_bytes(tmp_path):
    root = tmp_path / "store"
    run = _run(_bootstrap_payload(300))
    simulation_id = _persist(root, run)
    before = _store_manifest(root, simulation_id)
    # the identical producer reuses the entry (verified byte for byte) and
    # leaves no scratch directory behind
    _persist(root, run)
    assert _store_manifest(root, simulation_id) == before
    assert _no_scratch_residue(root)

    # a producer whose bytes differ under the same identity fails closed
    def _different_summary(directory):
        return (write_produced_sidecar(directory, "walk_summary.json", b"{\"forged\": true}\n"),)

    with pytest.raises(SearchStoreError, match="DIFFERENT sidecar content for 'walk_summary.json'"):
        save_or_reuse_envelope(root, _STORE, run.envelope, sidecar_producer=_different_summary)
    assert _store_manifest(root, simulation_id) == before
    assert _no_scratch_residue(root)

    # a producer declaring a sidecar the stored entry never had fails closed too
    def _extra(directory):
        return (write_produced_sidecar(directory, "never_stored.bin", b"x"),)

    with pytest.raises(SearchStoreError, match="DIFFERENT sidecar content for 'never_stored.bin'"):
        save_or_reuse_envelope(root, _STORE, run.envelope, sidecar_producer=_extra)
    assert _no_scratch_residue(root)
    # the bridge's own summary carries the counts the streaming writer produced
    summary = json.loads(load_sidecar_bytes(root, _STORE, simulation_id, "walk_summary.json"))
    detail = load_account_event_detail_manifest(root, simulation_id)
    assert summary["event_detail_partition_count"] == len(detail["partitions"]) == 2
    assert summary["event_detail_rows"] == detail["total_rows"]
    assert summary["event_detail_bytes"] == detail["total_bytes"]
