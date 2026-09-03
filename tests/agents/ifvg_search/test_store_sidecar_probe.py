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
    save_or_reuse_envelope,
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


# ── HARDENING-BACKEND-FIX §7.1 — the central manifest-entry validator ─────────


def _rewrite_manifest(directory, mutate) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    core = {k: v for k, v in manifest.items() if k != "manifest_payload_sha256"}
    manifest["manifest_payload_sha256"] = canonical_sha256(core)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


_VECTORS_SHA = hashlib.sha256(b'{"a": 1}\n').hexdigest()
_MALFORMED_ENTRIES = {
    "traversal": {"path": "../escape.json", "sha256": "0" * 64, "bytes": 0},
    "absolute_posix": {"path": "/etc/passwd", "sha256": "0" * 64, "bytes": 0},
    "absolute_windows": {"path": "\\\\server\\share\\x.json", "sha256": "0" * 64, "bytes": 0},
    "drive_qualified": {"path": "C:evil.json", "sha256": "0" * 64, "bytes": 0},
    "dot": {"path": ".", "sha256": "0" * 64, "bytes": 0},
    "dotdot": {"path": "..", "sha256": "0" * 64, "bytes": 0},
    "empty": {"path": "", "sha256": "0" * 64, "bytes": 0},
    "separator": {"path": "sub/other.json", "sha256": "0" * 64, "bytes": 0},
    "backslash": {"path": "sub\\other.json", "sha256": "0" * 64, "bytes": 0},
    "reserved_manifest": {"path": "manifest.json", "sha256": "0" * 64, "bytes": 0},
    "hidden": {"path": ".hidden", "sha256": "0" * 64, "bytes": 0},
    "bad_hash": {"path": "other.json", "sha256": "not-a-hash", "bytes": 0},
    "short_hash": {"path": "other.json", "sha256": "0" * 63, "bytes": 0},
    "uppercase_hash": {"path": "other.json", "sha256": "A" * 64, "bytes": 0},
    "negative_bytes": {"path": "other.json", "sha256": "0" * 64, "bytes": -1},
    "bool_bytes": {"path": "other.json", "sha256": "0" * 64, "bytes": True},
    "string_bytes": {"path": "other.json", "sha256": "0" * 64, "bytes": "9"},
    "non_mapping": ["other.json", "0" * 64, 0],
    "scalar_entry": "other.json",
    "missing_keys": {"path": "other.json"},
    "non_string_path": {"path": 7, "sha256": "0" * 64, "bytes": 0},
    "duplicate_path": {"path": "vectors.json", "sha256": _VECTORS_SHA, "bytes": 9},
    "case_collision": {"path": "VECTORS.json", "sha256": _VECTORS_SHA, "bytes": 9},
}


@pytest.mark.parametrize("label", sorted(_MALFORMED_ENTRIES))
def test_malformed_manifest_entries_fail_typed_on_every_read_path(entry, label) -> None:
    """HB-FIX-08: traversal, absolute / drive-qualified paths, dot and empty
    paths, separators, reserved names, invalid hashes and byte counts,
    duplicate / normalized-collision paths and non-mapping entries are the
    typed ``malformed_manifest`` on every probe / load path — never an
    ``AttributeError`` / ``KeyError``, never silent absence."""

    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    bad_entry = _MALFORMED_ENTRIES[label]
    _rewrite_manifest(directory, lambda manifest: manifest["artifacts"].append(bad_entry))
    probes = (
        lambda: probe_sidecar(root, "core_replays", envelope_id, "vectors.json"),
        lambda: has_sidecar(root, "core_replays", envelope_id, "vectors.json"),
        lambda: load_optional_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
        lambda: load_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
        lambda: load_json_sidecar(root, "core_replays", envelope_id, "vectors.json"),
        lambda: load_verified_envelope(
            root, "core_replays", envelope_id, CoreStrategyReplayIdentity
        ),
        # the idempotent-reuse path reads the manifest too
        lambda: save_or_reuse_envelope(
            root, "core_replays", _envelope(), extra_files={"vectors.json": b'{"a": 1}\n'}
        ),
    )
    try:
        for probe in probes:
            with pytest.raises(SidecarLoadError) as info:
                probe()
            assert info.value.reason == "malformed_manifest", (label, str(info.value))
    finally:
        manifest_path.write_bytes(original)
    assert probe_sidecar(root, "core_replays", envelope_id, "vectors.json") == "present"


def test_a_manifest_without_the_envelope_entry_is_malformed(entry) -> None:
    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()
    _rewrite_manifest(
        directory,
        lambda manifest: manifest.update(
            artifacts=[a for a in manifest["artifacts"] if a["path"] != "envelope.json"]
        ),
    )
    try:
        with pytest.raises(SidecarLoadError) as info:
            load_verified_envelope(root, "core_replays", envelope_id, CoreStrategyReplayIdentity)
        assert info.value.reason == "malformed_manifest"
        _rewrite_manifest(directory, lambda manifest: manifest.update(artifacts={"a": 1}))
        with pytest.raises(SidecarLoadError) as info:
            probe_sidecar(root, "core_replays", envelope_id, "vectors.json")
        assert info.value.reason == "malformed_manifest"
    finally:
        manifest_path.write_bytes(original)


def test_a_declared_byte_count_that_disagrees_with_the_file_is_a_hash_mismatch(entry) -> None:
    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    manifest_path = directory / "manifest.json"
    original = manifest_path.read_bytes()

    def _shrink(manifest):
        for artifact in manifest["artifacts"]:
            if artifact["path"] == "vectors.json":
                artifact["bytes"] = 1

    _rewrite_manifest(directory, _shrink)
    try:
        for probe in (
            lambda: probe_sidecar(root, "core_replays", envelope_id, "vectors.json"),
            lambda: load_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
            lambda: load_verified_envelope(
                root, "core_replays", envelope_id, CoreStrategyReplayIdentity
            ),
        ):
            with pytest.raises(SidecarLoadError) as info:
                probe()
            assert info.value.reason == "sidecar_hash_mismatch"
    finally:
        manifest_path.write_bytes(original)


def test_symlink_escape_is_refused_before_the_file_is_opened(entry, tmp_path) -> None:
    """HB-FIX-08: a declared sidecar that is a symbolic link to a file OUTSIDE
    the artifact directory is refused as ``sidecar_path_escape`` even when
    the target's bytes would hash correctly."""

    import os

    root, envelope_id = entry
    directory = envelope_destination(root, "core_replays", envelope_id)
    sidecar = directory / "vectors.json"
    original = sidecar.read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_bytes(original)  # identical bytes: only the escape can refuse it
    sidecar.unlink()
    try:
        os.symlink(outside, sidecar)
    except (OSError, NotImplementedError) as error:
        sidecar.write_bytes(original)
        pytest.skip(f"symbolic links are unavailable on this host: {error}")
    try:
        for probe in (
            lambda: probe_sidecar(root, "core_replays", envelope_id, "vectors.json"),
            lambda: has_sidecar(root, "core_replays", envelope_id, "vectors.json"),
            lambda: load_sidecar_bytes(root, "core_replays", envelope_id, "vectors.json"),
            lambda: load_verified_envelope(
                root, "core_replays", envelope_id, CoreStrategyReplayIdentity
            ),
        ):
            with pytest.raises(SidecarLoadError) as info:
                probe()
            assert info.value.reason == "sidecar_path_escape"
    finally:
        sidecar.unlink()
        sidecar.write_bytes(original)
    assert probe_sidecar(root, "core_replays", envelope_id, "vectors.json") == "present"


def test_invalid_store_locators_are_distinguished_from_absence(entry) -> None:
    root, envelope_id = entry
    for store, key in (
        ("core_replays", "not-an-id"),
        ("core_replays", "../" + "0" * 61),
        ("no_such_store", envelope_id),
        ("core_replays", ""),
    ):
        with pytest.raises(SidecarLoadError) as info:
            load_optional_sidecar_bytes(root, store, key, "vectors.json")
        assert info.value.reason == "invalid_store_locator", (store, key)
        with pytest.raises(SidecarLoadError) as info:
            load_verified_envelope(root, store, key, CoreStrategyReplayIdentity)
        assert info.value.reason == "invalid_store_locator", (store, key)
    with pytest.raises(SidecarLoadError) as info:
        load_optional_sidecar_bytes(root, "core_replays", "0" * 64, "vectors.json")
    assert info.value.reason == "store_entry_missing"
    # a sidecar NAME is held to the same whitelist (a store error, not a probe result)
    for bad in ("../x", "a/b", "C:evil", ".hidden", "", "manifest.json"):
        with pytest.raises(SearchStoreError):
            probe_sidecar(root, "core_replays", envelope_id, bad)
