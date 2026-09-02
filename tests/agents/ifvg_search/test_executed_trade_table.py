"""R6.1-FIX §3.7 (F-06) — the immutable, content-addressed executed-trade
table artifact: an exact ordered Arrow projection (never inferred from a
frame), derived from the core replay identity so every consumer exact-loads
it without a listing; identical bytes reuse, different bytes under one id
fail closed, tampering fails closed; the costed evaluation computed from
the persisted projection equals the one computed from the raw table.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pyarrow as pa
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.dataset import table_content_hash
from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256
from alpha_lab.agents.data_infra.ifvg.search.executed_trade_table import (
    EXECUTED_TRADE_TABLE_PROJECTION_ID,
    EXECUTED_TRADE_TABLE_SCHEMA_HASH,
    EXECUTED_TRADE_TABLE_SCHEMA_V1,
    EXECUTED_TRADE_TABLE_SIDECAR,
    EXECUTED_TRADE_TABLE_STORE,
    ExecutedTradeTableEnvelope,
    ExecutedTradeTablePayload,
    build_executed_trade_table,
    executed_trade_table_bytes,
    executed_trade_table_id_for,
    load_executed_trade_table,
    probe_executed_trade_table,
    project_executed_trades,
    save_executed_trade_table,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
    registered_identity_pairs,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SEARCH_STORE_NAMES,
    SearchStoreError,
    SidecarLoadError,
    envelope_destination,
)
from alpha_lab.agents.data_infra.ifvg.search.strategy_metrics import compute_strategy_metrics
from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS, make_resolved_trades_frame

_CORE = "a" * 64


def test_projection_schema_is_exact_and_never_inferred() -> None:
    names = tuple(EXECUTED_TRADE_TABLE_SCHEMA_V1.names)
    assert len(names) == len(set(names)) == 42
    assert names[:5] == (
        "record_table",
        "record_schema_version",
        "capture_schema_version",
        "dataset_schema_version",
        "trade_schema_version",
    )
    assert names[-3:] == ("realized_ticks", "mfe_ticks", "mae_ticks")
    assert EXECUTED_TRADE_TABLE_SCHEMA_V1.field("trading_day").type == pa.date32()
    assert EXECUTED_TRADE_TABLE_SCHEMA_V1.field("entry_ts_utc").type == pa.timestamp("ns", "UTC")
    assert EXECUTED_TRADE_TABLE_SCHEMA_V1.field("risk_ticks").type == pa.float64()
    assert EXECUTED_TRADE_TABLE_PROJECTION_ID == "core_executed_trade_exact_v1"
    assert EXECUTED_TRADE_TABLE_STORE in SEARCH_STORE_NAMES
    assert any(pair.name == "ExecutedTradeTable" for pair in registered_identity_pairs())
    raw = make_resolved_trades_frame(SYNTHETIC_DAYS)
    projected = project_executed_trades(raw)
    assert list(projected.columns) == list(names)
    assert list(projected["trade_id"]) == sorted(raw["trade_id"].astype(str))
    # a frame lacking a projection column is refused (nothing is inferred or filled)
    with pytest.raises(ValueError, match="entry_session"):
        project_executed_trades(raw.drop(columns=["entry_session"]))
    # extra columns of the wide v2 union are dropped, never carried
    widened = raw.assign(geometry_entry_ticks=1.0)
    assert list(project_executed_trades(widened).columns) == list(names)


def test_table_identity_derives_from_the_core_replay_and_binds_the_bytes(tmp_path) -> None:
    raw = make_resolved_trades_frame(SYNTHETIC_DAYS)
    envelope, table_bytes = build_executed_trade_table(_CORE, raw, record_schema_version=2)
    payload = envelope.payload
    assert isinstance(payload, ExecutedTradeTablePayload)
    assert payload.core_replay_id == _CORE
    assert payload.source_record_schema_version == 2
    assert payload.executed_trade_arrow_schema_hash == EXECUTED_TRADE_TABLE_SCHEMA_HASH
    assert payload.source_table_name == "executed_trade"
    assert envelope.executed_trade_table_id == executed_trade_table_id_for(
        _CORE, record_schema_version=2
    )
    assert envelope.executed_trade_table_id == canonical_contract_sha256(payload)
    assert envelope.executed_trade_table_sha256 == hashlib.sha256(table_bytes).hexdigest()
    assert envelope.source_core_table_hash == table_content_hash(RecordTable.EXECUTED_TRADE, raw)
    assert envelope.row_count == len(raw) and envelope.byte_size == len(table_bytes)
    # the record schema version must be the table's own
    with pytest.raises(ValueError, match="record_schema_version"):
        build_executed_trade_table(_CORE, raw, record_schema_version=3)
    # save → exact load by the DERIVED id (no listing) → identical bytes reuse
    root = tmp_path / "store"
    _stored, reused = save_executed_trade_table(root, envelope, table_bytes)
    assert reused is False
    assert probe_executed_trade_table(root, envelope.executed_trade_table_id) == "present"
    assert probe_executed_trade_table(root, "0" * 64) == "absent"
    loaded = load_executed_trade_table(root, envelope.executed_trade_table_id)
    assert loaded.envelope.model_dump(mode="json") == envelope.model_dump(mode="json")
    assert loaded.table_bytes == table_bytes
    assert list(loaded.frame.columns) == list(EXECUTED_TRADE_TABLE_SCHEMA_V1.names)
    _again, reused = save_executed_trade_table(root, envelope, table_bytes)
    assert reused is True
    # different bytes under one semantic id fail closed
    other_raw = raw.copy()
    other_raw.loc[0, "realized_ticks"] = 8
    other_raw.loc[0, "resolution"] = "target"
    other_raw.loc[0, "mfe_ticks"] = 40
    other_raw.loc[0, "mae_ticks"] = -8
    other_envelope, other_bytes = build_executed_trade_table(
        _CORE, other_raw, record_schema_version=2
    )
    assert other_envelope.executed_trade_table_id == envelope.executed_trade_table_id
    assert other_envelope.executed_trade_table_sha256 != envelope.executed_trade_table_sha256
    with pytest.raises(SearchStoreError, match="DIFFERENT"):
        save_executed_trade_table(root, other_envelope, other_bytes)
    # a tampered sidecar fails closed at load AND at the probe (corrupt ≠ absent)
    sidecar = (
        envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, envelope.executed_trade_table_id)
        / EXECUTED_TRADE_TABLE_SIDECAR
    )
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-4] + b"\x00" * 4)
        with pytest.raises(SearchStoreError):
            load_executed_trade_table(root, envelope.executed_trade_table_id)
        with pytest.raises(SearchStoreError):
            probe_executed_trade_table(root, envelope.executed_trade_table_id)
    finally:
        sidecar.write_bytes(original)
    # the envelope's own binding is re-checked on load
    forged = ExecutedTradeTableEnvelope.from_payload(
        payload,
        executed_trade_table_sha256="1" * 64,
        source_core_table_hash=envelope.source_core_table_hash,
        row_count=envelope.row_count,
        byte_size=envelope.byte_size,
    )
    with pytest.raises(ValueError, match="hash"):
        save_executed_trade_table(tmp_path / "forged", forged, table_bytes)


def _v2_capture_typed(raw: pd.DataFrame) -> pd.DataFrame:
    """The synthetic fixture re-typed the way the real v2 capture types the
    table: float64 ticks, ``date`` trading days, tz-aware timestamps."""

    typed = raw.copy()
    for column in (
        "entry_ticks",
        "stop_ticks",
        "target_ticks",
        "risk_ticks",
        "bars_after_entry_to_resolution",
        "realized_ticks",
        "mfe_ticks",
        "mae_ticks",
    ):
        typed[column] = typed[column].astype("float64")
    typed["trading_day"] = pd.to_datetime(typed["trading_day"]).dt.date
    for column in ("entry_ts_utc", "resolution_ts_utc"):
        typed[column] = pd.to_datetime(typed[column], utc=True)
    return typed


def _evaluation_bytes(frame: pd.DataFrame) -> bytes:
    metrics = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: frame}, cost_points=0.5, evaluation_config_hash="e" * 64
    )
    return (json.dumps(metrics.model_dump(mode="json"), sort_keys=True) + "\n").encode("utf-8")


@pytest.mark.parametrize("shape", ["int_typed_fixture", "v2_capture_typed"])
def test_costed_evaluation_from_the_persisted_projection_equals_the_raw_evaluation(
    tmp_path, shape
) -> None:
    """Reviews B-01 / B-10: the evaluation computed from the persisted
    projection equals the raw one as a VALUE, and its serialized bytes are
    identical whichever verified path produced the frame (the projection
    loaded, loaded again, or projected in memory) — the ONE canonical
    evaluation frame the pipeline publishes from. The int-typed synthetic
    fixture serializes differently from its own projection (the root of
    B-01); the real capture's typing IS the projection's typing."""

    raw = make_resolved_trades_frame(SYNTHETIC_DAYS)
    if shape == "v2_capture_typed":
        raw = _v2_capture_typed(raw)
    envelope, table_bytes = build_executed_trade_table(_CORE, raw, record_schema_version=2)
    root = tmp_path / "store"
    save_executed_trade_table(root, envelope, table_bytes)
    loaded = load_executed_trade_table(root, envelope.executed_trade_table_id)
    from_raw = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: raw}, cost_points=0.5, evaluation_config_hash="e" * 64
    )
    from_table = compute_strategy_metrics(
        {RecordTable.EXECUTED_TRADE: loaded.frame},
        cost_points=0.5,
        evaluation_config_hash="e" * 64,
    )
    assert from_table.model_dump(mode="json") == from_raw.model_dump(mode="json")
    again = load_executed_trade_table(root, envelope.executed_trade_table_id)
    canonical = _evaluation_bytes(loaded.frame)
    assert _evaluation_bytes(again.frame) == canonical
    assert _evaluation_bytes(project_executed_trades(raw)) == canonical
    # both typings of the same values project to identical bytes
    assert table_bytes == executed_trade_table_bytes(
        project_executed_trades(_v2_capture_typed(make_resolved_trades_frame(SYNTHETIC_DAYS)))
    )
    if shape == "v2_capture_typed":
        assert _evaluation_bytes(raw) == canonical
    else:
        assert _evaluation_bytes(raw) != canonical


def test_probe_types_corrupt_entries_by_the_store_reason_never_absent(tmp_path) -> None:
    """Reviews B-02 / B-08: an entry directory without its manifest is typed
    corruption (never "absent"); a malformed envelope is ``malformed_sidecar``
    — the reason comes from the store's detection point, not from message
    text; only a non-existent entry is absent."""

    raw = make_resolved_trades_frame(SYNTHETIC_DAYS)
    envelope, table_bytes = build_executed_trade_table(_CORE, raw, record_schema_version=2)
    root = tmp_path / "store"
    save_executed_trade_table(root, envelope, table_bytes)
    table_id = envelope.executed_trade_table_id
    directory = envelope_destination(root, EXECUTED_TRADE_TABLE_STORE, table_id)
    manifest_path = directory / "manifest.json"
    envelope_path = directory / "envelope.json"
    original_manifest = manifest_path.read_bytes()
    original_envelope = envelope_path.read_bytes()
    try:
        manifest_path.unlink()
        with pytest.raises(SidecarLoadError) as info:
            probe_executed_trade_table(root, table_id)
        assert info.value.reason == "manifest_missing_for_existing_entry"
    finally:
        manifest_path.write_bytes(original_manifest)
    try:
        garbage = b"{not json\n"
        envelope_path.write_bytes(garbage)
        manifest = json.loads(original_manifest.decode("utf-8"))
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "envelope.json":
                artifact["sha256"] = hashlib.sha256(garbage).hexdigest()
                artifact["bytes"] = len(garbage)
        core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
        manifest["manifest_payload_sha256"] = canonical_sha256(core)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SidecarLoadError) as info:
            probe_executed_trade_table(root, table_id)
        assert info.value.reason == "malformed_sidecar"
    finally:
        envelope_path.write_bytes(original_envelope)
        manifest_path.write_bytes(original_manifest)
    assert probe_executed_trade_table(root, table_id) == "present"
    assert probe_executed_trade_table(root, "0" * 64) == "absent"
