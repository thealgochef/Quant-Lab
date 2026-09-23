"""Saved metric recovery verifies one execution table and never replays bars."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.dataset import table_content_hash
from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256
from alpha_lab.agents.data_infra.ifvg.profiles import resolve_profile_config
from alpha_lab.agents.data_infra.ifvg.search import saved_strategy_result, strategy_executor
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    _example_core_replay_payload,
)
from alpha_lab.agents.data_infra.ifvg.search.saved_strategy_result import (
    load_saved_strategy_result,
    make_saved_strategy_result_loader,
)
from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import compute_strategy_metrics
from tests.agents.ifvg_search.conftest import make_resolved_trades_frame


def _bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def _fixture(root, *, fault=None, empty=False, generated_profile_alias=False):
    section = resolve_profile_config().effective_config
    canonical_profile_id = section["profile_name"]
    if generated_profile_alias:
        section["profile_name"] = "ifvg_search_profile_5ce78b888ea9214b"
    section_hash = canonical_sha256(section)
    core = CoreStrategyReplayIdentity.from_payload(
        _example_core_replay_payload().model_copy(
            update={
                "resolved_section_config_hash": section_hash,
                "canonical_profile_id": canonical_profile_id,
            }
        )
    )
    tables = make_resolved_trades_frame(
        ("2026-01-12", "2026-01-13", "2026-01-14"), section_config_hash=section_hash
    )
    tables["is_warmup"] = tables.trading_day.eq("2026-01-12")
    if empty:
        tables = tables.head(0)
    if fault == "duplicate_trade":
        tables.loc[1, "trade_id"] = tables.loc[0, "trade_id"]
    if fault == "trade_profile":
        tables["envelope_section_config_hash"] = "e" * 64
    identity = {
        "resolved_profile_hash": "b" * 64 if fault == "dataset_profile" else section_hash,
        "dataset_schema_version": 2,
    }
    dataset_id = canonical_sha256(identity)
    exploration = root / "v2_datasets" / dataset_id / "exploration"
    exploration.mkdir(parents=True)
    artifacts = []
    config = {**section, "tp_r_multiple": 2.0} if fault == "section_hash" else section
    for name, value in (
        (
            "raw_config.json",
            {"core_replay_id": "b" * 64 if fault == "raw_core" else core.core_replay_id},
        ),
        ("effective_config.json", {"section": config}),
        ("invariant_audit.json", {"passed": fault != "invariant"}),
    ):
        path = exploration / name
        data = _bytes(value)
        path.write_bytes(data)
        artifacts.append(
            {
                "path": "exploration/" + name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    trade_path = exploration / "executed_trade.parquet"
    tables.to_parquet(trade_path, index=False)
    artifacts.append(
        {
            "path": "exploration/executed_trade.parquet",
            "bytes": trade_path.stat().st_size,
            "sha256": hashlib.sha256(trade_path.read_bytes()).hexdigest(),
            "rows": len(tables) + (1 if fault == "row_count" else 0),
        }
    )
    # Deliberately absent: selective recovery must not open/check full-history
    # tables or the optional source-bar artifact, even when the manifest names them.
    for name in ("setup_lifecycle_event", "geometry_evidence", "label_source_bars"):
        artifacts.append(
            {
                "path": f"exploration/{name}.parquet",
                "bytes": 1000000000,
                "sha256": "a" * 64,
                "rows": 10000000,
            }
        )
    if fault == "duplicate_path":
        artifacts.append(artifacts[0])
    manifest = {
        "dataset_id": dataset_id,
        "manifest_schema_version": 2,
        "identity": identity,
        "artifacts": artifacts,
    }
    manifest_hash = canonical_sha256(manifest)
    manifest["manifest_payload_sha256"] = manifest_hash
    (exploration / "manifest.json").write_bytes(_bytes(manifest))
    ref = {
        "core_replay_id": "b" * 64 if fault == "reference_core" else core.core_replay_id,
        "v2_dataset_artifact_id": dataset_id,
        "manifest_payload_sha256": "b" * 64 if fault == "reference_manifest" else manifest_hash,
        "gross_trade_stream_hash": "b" * 64
        if fault == "trade_stream"
        else table_content_hash(RecordTable.EXECUTED_TRADE, tables),
    }
    save_envelope_immutable(
        root, "core_replays", core, extra_files={"artifact_reference.json": _bytes(ref)}
    )
    if fault == "trade_bytes":
        with trade_path.open("ab") as handle:
            handle.write(b"changed")
    if fault == "config_bytes":
        (exploration / "effective_config.json").write_text("{}", encoding="utf-8")
    if fault == "manifest_bytes":
        manifest["identity"]["changed"] = True
        (exploration / "manifest.json").write_bytes(_bytes(manifest))
    return core, section, tables, trade_path


def _inventory(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("cost", [0.514, 1.028])
def test_recovered_metrics_equal_original_with_warmup_and_exact_costs(tmp_path, monkeypatch, cost):
    core, section, trades, trade_path = _fixture(tmp_path)
    before = _inventory(tmp_path)
    calls = []
    original_read = pd.read_parquet

    def bounded_read(path, *args, **kwargs):
        assert Path(path) == trade_path
        calls.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(saved_strategy_result.pd, "read_parquet", bounded_read)
    loaded = load_saved_strategy_result(tmp_path, core.core_replay_id)
    assert list(loaded.tables) == [RecordTable.EXECUTED_TRADE]
    assert len(calls) == 1
    pd.testing.assert_frame_equal(loaded.tables[RecordTable.EXECUTED_TRADE], trades)
    assert loaded.effective_section == section
    assert loaded.core_replay_id == core.core_replay_id
    kwargs = {
        "cost_points": cost,
        "evaluation_config_hash": "a" * 64,
        "tp_r_multiple": section["tp_r_multiple"],
    }
    actual = compute_strategy_metrics(loaded.tables, **kwargs)
    expected = compute_strategy_metrics({RecordTable.EXECUTED_TRADE: trades}, **kwargs)
    assert actual == expected
    assert actual.executed_trades == 8
    assert actual.net_expectancy_r == pytest.approx(0.5 - cost / 10.0)
    assert _inventory(tmp_path) == before


def test_genuine_empty_execution_table_is_preserved(tmp_path):
    core, _, _, _ = _fixture(tmp_path, empty=True)
    loaded = load_saved_strategy_result(tmp_path, core.core_replay_id)
    assert loaded.tables[RecordTable.EXECUTED_TRADE].empty
    actual = compute_strategy_metrics(
        loaded.tables, cost_points=0.514, evaluation_config_hash="a" * 64
    )
    assert actual.executed_trades == 0
    assert actual.net_expectancy_r is None


@pytest.mark.parametrize(
    "fault, message",
    [
        ("reference_core", "reference belongs"),
        ("reference_manifest", "manifest differs"),
        ("dataset_profile", "profile differs"),
        ("raw_core", "dataset belongs"),
        ("section_hash", "effective section identity"),
        ("invariant", "invariant audit"),
        ("row_count", "row count"),
        ("trade_stream", "execution stream"),
        ("trade_bytes", "artifact changed"),
        ("config_bytes", "artifact changed"),
        ("manifest_bytes", "manifest payload hash"),
        ("duplicate_path", "duplicate artifact paths"),
        ("duplicate_trade", "not unique"),
        ("trade_profile", "another section"),
    ],
)
def test_corrupt_or_mismatched_saved_evidence_fails_closed(tmp_path, fault, message):
    core, _, _, _ = _fixture(tmp_path, fault=fault)
    with pytest.raises(ValueError, match=message):
        load_saved_strategy_result(tmp_path, core.core_replay_id)


def test_missing_saved_evidence_cannot_become_zero_trades(tmp_path):
    core, _, _, trade_path = _fixture(tmp_path)
    trade_path.unlink()
    with pytest.raises(ValueError, match="artifact is missing"):
        load_saved_strategy_result(tmp_path, core.core_replay_id)


def test_factory_loader_never_resolves_inputs_and_checks_child_spec(tmp_path, monkeypatch):
    core, section, _, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        strategy_executor, "load_day_artifacts", lambda *a, **kw: pytest.fail("source bars opened")
    )
    monkeypatch.setattr(
        strategy_executor, "run_child_replay", lambda *a, **kw: pytest.fail("replay started")
    )
    loader = make_saved_strategy_result_loader(tmp_path)
    spec = SimpleNamespace(
        resolved_section_config_hash=core.payload.resolved_section_config_hash,
        canonical_profile_id=section["profile_name"],
    )
    loaded = loader(spec=spec, core_replay_id=core.core_replay_id)
    assert loaded.core_replay_id == core.core_replay_id
    spec.resolved_section_config_hash = "b" * 64
    with pytest.raises(PermissionError, match="requested child section"):
        loader(spec=spec, core_replay_id=core.core_replay_id)
    spec.resolved_section_config_hash = core.payload.resolved_section_config_hash
    spec.canonical_profile_id = "different_profile"
    with pytest.raises(PermissionError, match="canonical profile"):
        loader(spec=spec, core_replay_id=core.core_replay_id)


def test_baseline_core_profile_identity_may_differ_from_generated_section_alias(tmp_path):
    core, section, _, _ = _fixture(tmp_path, generated_profile_alias=True)
    assert core.payload.canonical_profile_id != section["profile_name"]
    spec = SimpleNamespace(
        resolved_section_config_hash=core.payload.resolved_section_config_hash,
        canonical_profile_id=core.payload.canonical_profile_id,
    )
    loaded = make_saved_strategy_result_loader(tmp_path)(
        spec=spec, core_replay_id=core.core_replay_id
    )
    assert loaded.canonical_profile_id == core.payload.canonical_profile_id
    assert loaded.effective_section == section
    # The generated alias cannot stand in for the Core identity in a spec.
    spec.canonical_profile_id = section["profile_name"]
    with pytest.raises(PermissionError, match="canonical profile"):
        make_saved_strategy_result_loader(tmp_path)(spec=spec, core_replay_id=core.core_replay_id)
