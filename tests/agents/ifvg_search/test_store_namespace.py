"""HARDENING-BACKEND §4.1 (F-11) — the semantic store namespace.

Authority follows the VERIFIED ``STORE_NAMESPACE.json`` envelope, never the
path: relocation preserves the id, an unmarked store fails closed, the class
is an explicit operator statement, the pathname heuristic is defense in
depth only, and a missing supersession head is corruption.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import (
    OwnerDecisionArtifactEnvelope,
    OwnerDecisionRefusalError,
    assert_run_scope_lawful_for_root,
    load_owner_decision,
    persist_owner_decision,
)
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    STORE_NAMESPACE_FILE,
    StoreNamespaceError,
    assert_namespace_deployment_coherent,
    genesis_id_for,
    initialize_store_namespace,
    initialize_test_namespace,
    load_store_namespace,
    namespace_class_of,
    path_looks_like_research_store,
    require_store_namespace,
    supersession_head_path,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import (
    current_supersession_head_witness,
)


def test_initialization_is_explicit_idempotent_and_immutable(tmp_path: Path) -> None:
    root = tmp_path / "store"
    assert namespace_class_of(root) is None
    with pytest.raises(StoreNamespaceError) as missing:
        require_store_namespace(root)
    assert missing.value.reason == "store_namespace_missing"
    first = initialize_store_namespace(root, namespace_class="test")
    assert first.payload.namespace_class == "test"
    assert first.payload.authority_genesis_id == genesis_id_for(
        first.payload.store_instance_id, "test"
    )
    # identical replay reuses; a different class is a typed divergence
    again = initialize_store_namespace(root, namespace_class="test")
    assert again.store_namespace_id == first.store_namespace_id
    with pytest.raises(StoreNamespaceError) as divergent:
        initialize_store_namespace(root, namespace_class="research")
    assert divergent.value.reason == "store_namespace_divergent"
    with pytest.raises(StoreNamespaceError) as other_instance:
        initialize_store_namespace(root, namespace_class="test", store_instance_id="1" * 32)
    assert other_instance.value.reason == "store_namespace_divergent"
    with pytest.raises(ValueError, match="namespace_class must be one of"):
        initialize_store_namespace(tmp_path / "bad", namespace_class="prod")
    # the genesis head exists from the start
    witness = current_supersession_head_witness(root)
    assert witness.line_count == 0
    assert witness.head_sha256 == first.payload.authority_genesis_id


def test_namespace_file_is_verified_on_every_load(tmp_path: Path) -> None:
    root = tmp_path / "store"
    namespace = initialize_test_namespace(root)
    path = root / STORE_NAMESPACE_FILE
    original = path.read_text(encoding="utf-8")
    document = json.loads(original)
    # rewriting the class (with a matching genesis) breaks the envelope id
    document["payload"]["namespace_class"] = "research"
    document["payload"]["authority_genesis_id"] = genesis_id_for(
        document["payload"]["store_instance_id"], "research"
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(StoreNamespaceError) as tampered:
        load_store_namespace(root)
    assert tampered.value.reason == "store_namespace_identity_mismatch"
    with pytest.raises(StoreNamespaceError):
        namespace_class_of(root)  # corrupt is never "unmarked"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreNamespaceError) as malformed:
        load_store_namespace(root)
    assert malformed.value.reason == "store_namespace_malformed"
    path.write_text(original, encoding="utf-8")
    assert load_store_namespace(root).store_namespace_id == namespace.store_namespace_id
    with pytest.raises(StoreNamespaceError) as wrong_class:
        require_store_namespace(root, expected_class="research")
    assert wrong_class.value.reason == "store_namespace_class_mismatch"


def test_identity_is_not_a_path_hash_and_relocation_preserves_it(tmp_path: Path) -> None:
    instance = "ab" * 16
    left = initialize_store_namespace(tmp_path / "left", namespace_class="test",
                                      store_instance_id=instance)
    right = initialize_store_namespace(tmp_path / "somewhere" / "else", namespace_class="test",
                                       store_instance_id=instance)
    assert left.store_namespace_id == right.store_namespace_id
    moved = tmp_path / "moved" / "search_test" / "v1"
    shutil.copytree(tmp_path / "left", moved)
    assert load_store_namespace(moved).store_namespace_id == left.store_namespace_id
    assert current_supersession_head_witness(moved) == current_supersession_head_witness(
        tmp_path / "left"
    )


def test_path_heuristic_is_defense_in_depth_never_authority(tmp_path: Path) -> None:
    research_looking = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    plain = tmp_path / "plain"
    assert path_looks_like_research_store(research_looking)
    assert not path_looks_like_research_store(plain)
    # an UNMARKED research-looking path has no authority either way
    assert namespace_class_of(research_looking) is None
    # a test namespace deployed under a research-looking path is incoherent
    namespace = initialize_test_namespace(research_looking)
    with pytest.raises(StoreNamespaceError) as incoherent:
        assert_namespace_deployment_coherent(research_looking, namespace)
    assert incoherent.value.reason == "store_namespace_deployment_incoherent"
    # ...and a research namespace under a plain path is coherent (the path
    # never defines the class)
    research = initialize_store_namespace(plain, namespace_class="research")
    assert_namespace_deployment_coherent(plain, research)
    assert namespace_class_of(plain) == "research"


def test_synthetic_scope_follows_the_namespace_class(tmp_path: Path) -> None:
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    assert_run_scope_lawful_for_root(unmarked, "full_authorized_development")
    # an unmarked plain root: the scope unlocks nothing (no owner artifact
    # can exist there) and is not refused for structural writes
    assert_run_scope_lawful_for_root(unmarked, "synthetic_fixture")
    research = tmp_path / "research"
    initialize_store_namespace(research, namespace_class="research")
    with pytest.raises(OwnerDecisionRefusalError, match="research namespace refuses"):
        assert_run_scope_lawful_for_root(research, "synthetic_fixture")
    assert_run_scope_lawful_for_root(research, "full_authorized_development")
    test_root = tmp_path / "test"
    initialize_test_namespace(test_root)
    assert_run_scope_lawful_for_root(test_root, "synthetic_fixture")
    # defense in depth: a research-looking path refuses the scope whatever
    # the marking
    looks_research = tmp_path / "search" / "v1"
    looks_research.mkdir(parents=True)
    with pytest.raises(OwnerDecisionRefusalError, match="defense in depth"):
        assert_run_scope_lawful_for_root(looks_research, "synthetic_fixture")


def test_unmarked_store_cannot_carry_owner_authority(tmp_path: Path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.owner_decisions import _example_payload

    root = tmp_path / "unmarked"
    envelope = OwnerDecisionArtifactEnvelope.from_payload(_example_payload())
    with pytest.raises(OwnerDecisionRefusalError) as refused:
        persist_owner_decision(root, envelope, recorded_at="2026-09-01T00:00:00+00:00")
    assert getattr(refused.value, "reason", None) == "store_namespace_missing"
    with pytest.raises(OwnerDecisionRefusalError):
        load_owner_decision(root, envelope.owner_decision_artifact_id)
    # a marked store refuses an artifact that names ANOTHER namespace
    marked = tmp_path / "marked"
    initialize_test_namespace(marked)
    with pytest.raises(OwnerDecisionRefusalError, match="another store namespace"):
        persist_owner_decision(marked, envelope, recorded_at="2026-09-01T00:00:00+00:00")


def test_missing_head_is_corruption_not_no_supersessions(tmp_path: Path) -> None:
    root = tmp_path / "store"
    initialize_test_namespace(root)
    supersession_head_path(root).unlink()
    with pytest.raises(StoreNamespaceError) as missing:
        current_supersession_head_witness(root)
    assert missing.value.reason == "supersession_head_missing"
    # re-initialization never re-creates a head for a marked store
    with pytest.raises(StoreNamespaceError) as still_missing:
        initialize_test_namespace(root)
    assert still_missing.value.reason == "supersession_head_missing"


def test_interrupted_initialization_recovers_only_from_a_genesis_head(tmp_path: Path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        write_supersession_head_atomic,
    )

    root = tmp_path / "store"
    # a crash after the genesis head and before the namespace file: the
    # retry rewrites the genesis head for the new instance
    write_supersession_head_atomic(
        root, store_namespace_id="0" * 64, record_id=None, line_count=0, head_sha256="1" * 64
    )
    namespace = initialize_test_namespace(root)
    assert current_supersession_head_witness(root).head_sha256 == (
        namespace.payload.authority_genesis_id
    )
    # a head that names records without a namespace file is corruption
    other = tmp_path / "corrupt"
    write_supersession_head_atomic(
        other, store_namespace_id="0" * 64, record_id="2" * 64, line_count=1,
        head_sha256="3" * 64,
    )
    with pytest.raises(StoreNamespaceError) as corrupt:
        initialize_test_namespace(other)
    assert corrupt.value.reason == "supersession_head_malformed"


def test_cli_shows_intent_then_initializes_and_refuses_divergence(tmp_path: Path, capsys) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ifvg_store_namespace", Path("scripts/ifvg_store_namespace.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    root = tmp_path / "cli_store"
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "test"]) == 0
    intent = json.loads(capsys.readouterr().out)
    assert intent["status"] == "intent" and intent["currently_marked_as"] is None
    assert namespace_class_of(root) is None  # nothing happened without --confirm
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "test",
                        "--confirm"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["status"] == "initialized" and initialized["namespace_class"] == "test"
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "research",
                        "--confirm"]) == 2
    refused = json.loads(capsys.readouterr().out)
    assert refused["reason"] == "store_namespace_divergent"
    assert module.main(["show", "--store-root", str(root)]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["supersession_head_witness"]["line_count"] == 0
    # a test namespace under a research-looking path is refused by the CLI
    looks = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    assert module.main(["init", "--store-root", str(looks), "--namespace-class", "test",
                        "--confirm"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "store_namespace_deployment_incoherent"
    )
    assert module.main(["show", "--store-root", str(tmp_path / "nothing")]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "store_namespace_missing"
