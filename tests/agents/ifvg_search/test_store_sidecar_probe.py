"""R6.1-FIX §3.8 (F-07) — the typed sidecar probe/load contract of the store:
``sidecar_not_produced_for_path`` is the ONLY lawful optional absence; every
other state (declared-but-missing file, sidecar hash mismatch, manifest hash
mismatch, envelope identity mismatch, malformed content, unexpected I/O
error) is a typed failure that propagates."""

from __future__ import annotations

import hashlib
import json

import pytest

from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    CoreStrategyReplayPayload,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SIDECAR_NOT_PRODUCED_FOR_PATH,
    SearchStoreError,
    SidecarLoadError,
    envelope_destination,
    has_envelope,
    has_sidecar,
    load_json_sidecar,
    load_optional_sidecar_bytes,
    load_sidecar_bytes,
    load_verified_envelope,
    probe_sidecar,
    save_envelope_immutable,
)


def _envelope() -> CoreStrategyReplayIdentity:
    return CoreStrategyReplayIdentity.from_payload(
        CoreStrategyReplayPayload(
            replay_input_bundle_id="1" * 64,
            quant_lab_replay_source_identity="e" * 64,
            strategy_core_commit="f" * 40,
            strategy_core_source_identity="f" * 64,
            resolved_section_config_hash="2" * 64,
            canonical_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            warmup_seed_identity="synthetic_cold_start_v1",
            anchor_policy="trading_day_18et_elapsed_v1",
            capture_schema_version=2,
            record_schema_version=2,
        )
    )


@pytest.fixture
def entry(tmp_path):
    root = tmp_path / "store"
    envelope = _envelope()
    save_envelope_immutable(
        root,
        "core_replays",
        envelope,
        extra_files={"vectors.json": b'{"a": 1}\n'},
    )
    return root, envelope.core_replay_id


def test_lawful_absence_is_the_only_optional_state(entry) -> None:
    root, envelope_id = entry
    assert probe_sidecar(root, "core_replays", envelope_id, "vectors.json") == "present"
    assert probe_sidecar(root, "core_replays", envelope_id, "other.json") == (
        SIDECAR_NOT_PRODUCED_FOR_PATH
    )
    assert has_sidecar(root, "core_replays", envelope_id, "vectors.json") is True
    assert has_sidecar(root, "core_replays", envelope_id, "other.json") is False
    assert load_optional_sidecar_bytes(root, "core_replays", envelope_id, "other.json") is None
    assert load_optional_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json") == (
        b'{"a": 1}\n'
    )
    assert load_json_sidecar(root, "core_replays", envelope_id, "vectors.json") == {"a": 1}


def test_every_corruption_state_is_a_typed_failure(entry) -> None:
    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    sidecar = directory / "vectors.json"
    manifest_path = directory / "manifest.json"
    original_sidecar = sidecar.read_bytes()
    original_manifest = manifest_path.read_bytes()

    def _expect(reason: str) -> None:
        with pytest.raises(SidecarLoadError) as info:
            load_optional_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json")
        assert info.value.reason == reason
        with pytest.raises(SidecarLoadError):
            has_sidecar(root, "core_replays", envelope_id, "vectors.json")

    try:
        sidecar.write_bytes(b'{"a": 2}\n')
        _expect("sidecar_hash_mismatch")
        sidecar.unlink()
        _expect("sidecar_missing_but_manifest_declares_it")
    finally:
        sidecar.write_bytes(original_sidecar)
    try:
        manifest = json.loads(original_manifest.decode("utf-8"))
        manifest["artifacts"].append({"path": "ghost.bin", "sha256": "0" * 64, "bytes": 0})
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _expect("manifest_hash_mismatch")
        manifest = json.loads(original_manifest.decode("utf-8"))
        manifest["envelope_id"] = "0" * 64
        core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
        from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256

        manifest["manifest_payload_sha256"] = canonical_sha256(core)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        _expect("envelope_identity_mismatch")
        manifest_path.write_text("{not json", encoding="utf-8")
        _expect("malformed_manifest")
    finally:
        manifest_path.write_bytes(original_manifest)
    # a present sidecar whose bytes are not JSON is malformed for the JSON reader
    try:
        sidecar.write_bytes(b"{not json\n")
        manifest = json.loads(original_manifest.decode("utf-8"))
        import hashlib

        for artifact in manifest["artifacts"]:
            if artifact["path"] == "vectors.json":
                artifact["sha256"] = hashlib.sha256(b"{not json\n").hexdigest()
                artifact["bytes"] = len(b"{not json\n")
        core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
        from alpha_lab.agents.data_infra.ifvg.manifest import canonical_sha256

        manifest["manifest_payload_sha256"] = canonical_sha256(core)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SidecarLoadError) as info:
            load_json_sidecar(root, "core_replays", envelope_id, "vectors.json")
        assert info.value.reason == "malformed_sidecar"
    finally:
        sidecar.write_bytes(original_sidecar)
        manifest_path.write_bytes(original_manifest)
    # a missing store entry is never "not produced"
    with pytest.raises(SidecarLoadError) as info:
        load_optional_sidecar_bytes(root, "core_replays", "0" * 64, "vectors.json")
    assert info.value.reason == "store_entry_missing"


# ── adversarial round (reviews B-02 / B-08) ──────────────────────────────────


def test_a_manifest_less_entry_is_corrupt_never_absent(entry) -> None:
    """Review B-02: an entry directory without its manifest is a typed
    corruption state on EVERY read path (probe, has_sidecar, optional bytes,
    has_envelope, verified envelope, sidecar bytes); only a non-existent
    entry directory is ``store_entry_missing`` / ``has_envelope`` False."""

    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    probes = (
        lambda: probe_sidecar(root, "core_replays", envelope_id, "vectors.json"),
        lambda: has_sidecar(root, "core_replays", envelope_id, "vectors.json"),
        lambda: load_optional_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
        lambda: has_envelope(root, "core_replays", envelope_id),
        lambda: load_verified_envelope(
            root, "core_replays", envelope_id, CoreStrategyReplayIdentity
        ),
        lambda: load_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
    )
    try:
        manifest_path.unlink()
        for probe in probes:
            with pytest.raises(SidecarLoadError) as info:
                probe()
            assert info.value.reason == "manifest_missing_for_existing_entry"
    finally:
        manifest_path.write_bytes(original)
    assert has_envelope(root, "core_replays", envelope_id) is True
    assert has_envelope(root, "core_replays", "0" * 64) is False
    with pytest.raises(SidecarLoadError) as info:
        load_verified_envelope(root, "core_replays", "0" * 64, CoreStrategyReplayIdentity)
    assert info.value.reason == "store_entry_missing"
    assert "missing search-store entry" in str(info.value)


def test_verified_envelope_load_raises_the_typed_reason_at_detection(entry) -> None:
    """Review B-08: the store raises the reason where it detects it — a
    caller never infers it from message text. The legacy message substrings
    survive inside the typed message."""

    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    envelope_path = directory / "envelope.json"
    manifest_path = directory / "manifest.json"
    sidecar = directory / "vectors.json"
    original_envelope = envelope_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    original_sidecar = sidecar.read_bytes()

    def _load():
        return load_verified_envelope(root, "core_replays", envelope_id, CoreStrategyReplayIdentity)

    try:
        sidecar.unlink()
        with pytest.raises(SidecarLoadError) as info:
            _load()
        assert info.value.reason == "sidecar_missing_but_manifest_declares_it"
        assert "failed verification" in str(info.value)
        sidecar.write_bytes(b'{"a": 3}\n')
        with pytest.raises(SidecarLoadError) as info:
            _load()
        assert info.value.reason == "sidecar_hash_mismatch"
        assert "failed verification" in str(info.value)
    finally:
        sidecar.write_bytes(original_sidecar)
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
            _load()
        assert info.value.reason == "malformed_sidecar"
    finally:
        envelope_path.write_bytes(original_envelope)
        manifest_path.write_bytes(original_manifest)
    try:
        manifest_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(SidecarLoadError) as info:
            _load()
        assert info.value.reason == "malformed_manifest"
        manifest = json.loads(original_manifest.decode("utf-8"))
        manifest["envelope_id"] = "0" * 64
        core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
        manifest["manifest_payload_sha256"] = canonical_sha256(core)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(SidecarLoadError) as info:
            _load()
        assert info.value.reason == "envelope_identity_mismatch"
        assert "manifest identity mismatch" in str(info.value)
    finally:
        manifest_path.write_bytes(original_manifest)
    assert _load().core_replay_id == envelope_id
    # every existing caller keeps its contract: the typed error IS a store error
    assert issubclass(SidecarLoadError, SearchStoreError)
