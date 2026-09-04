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
    # HARDENING-BACKEND-FIX.1 §2: the FIRST initialization of a real store
    # requires an explicit instance id — the initializer never generates one
    with pytest.raises(StoreNamespaceError) as no_instance:
        initialize_store_namespace(root, namespace_class="test")
    assert no_instance.value.reason == "store_instance_id_required"
    assert namespace_class_of(root) is None
    assert not (root / STORE_NAMESPACE_FILE).exists()
    assert not supersession_head_path(root).exists()
    instance = "0f" * 16
    first = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert first.payload.namespace_class == "test"
    assert first.payload.store_instance_id == instance
    assert first.payload.authority_genesis_id == genesis_id_for(
        first.payload.store_instance_id, "test"
    )
    # identical replay reuses (explicit or class-only); a different class is a
    # typed divergence
    again = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert again.store_namespace_id == first.store_namespace_id
    class_only = initialize_store_namespace(root, namespace_class="test")
    assert class_only.store_namespace_id == first.store_namespace_id
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
    research = initialize_store_namespace(
        plain, namespace_class="research", store_instance_id="a7" * 16
    )
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
    initialize_store_namespace(research, namespace_class="research", store_instance_id="a8" * 16)
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
    # re-initialization never re-creates a head for a marked store unless the
    # request provably reproduces the SAME initialization (HARDENING-BACKEND-FIX
    # §4.2: a class-only request cannot, so the store stays incomplete)
    with pytest.raises(StoreNamespaceError) as still_missing:
        initialize_test_namespace(root)
    assert still_missing.value.reason == "incomplete_store_namespace_initialization"
    assert not supersession_head_path(root).exists()


def test_interrupted_initialization_recovers_only_from_a_genesis_head(tmp_path: Path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        write_supersession_head_atomic,
    )

    root = tmp_path / "store"
    # an orphan genesis head that matches NO deterministic initialization is
    # never overwritten by a retry (HARDENING-BACKEND-FIX §4.2: recovery is
    # exact or refused)
    write_supersession_head_atomic(
        root, store_namespace_id="0" * 64, record_id=None, line_count=0, head_sha256="1" * 64
    )
    orphan_bytes = supersession_head_path(root).read_bytes()
    with pytest.raises(StoreNamespaceError) as unmatched:
        initialize_test_namespace(root)
    assert unmatched.value.reason == "incomplete_store_namespace_initialization"
    with pytest.raises(StoreNamespaceError) as unmatched_instance:
        initialize_store_namespace(root, namespace_class="test", store_instance_id="2" * 32)
    assert unmatched_instance.value.reason == "incomplete_store_namespace_initialization"
    assert supersession_head_path(root).read_bytes() == orphan_bytes
    assert not (root / STORE_NAMESPACE_FILE).exists()
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
    assert intent["store_instance_id_required"] is True  # HARDENING-BACKEND-FIX.1 §2
    assert namespace_class_of(root) is None  # nothing happened without --confirm
    # the first initialization without an explicit instance id is refused untouched
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "test",
                        "--confirm"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "store_instance_id_required"
    assert namespace_class_of(root) is None
    instance = "c1" * 16
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "test",
                        "--store-instance-id", instance, "--confirm"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["status"] == "initialized" and initialized["namespace_class"] == "test"
    assert initialized["store_instance_id"] == instance
    # an identical replay (class-only, or the explicit id) is idempotent
    assert module.main(["init", "--store-root", str(root), "--namespace-class", "test",
                        "--confirm"]) == 0
    replayed = json.loads(capsys.readouterr().out)
    assert replayed["store_namespace_id"] == initialized["store_namespace_id"]
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
                        "--store-instance-id", "c2" * 16, "--confirm"]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == (
        "store_namespace_deployment_incoherent"
    )
    assert module.main(["show", "--store-root", str(tmp_path / "nothing")]) == 2
    assert json.loads(capsys.readouterr().out)["reason"] == "store_namespace_missing"


# ── HARDENING-BACKEND-FIX §4.2 — atomic, recoverable initialization ──────────


def _genesis_pair_is_coherent(root: Path, expected_id: str) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import read_supersession_head

    envelope = load_store_namespace(root)
    assert envelope.store_namespace_id == expected_id
    head = read_supersession_head(root, store_namespace_id=envelope.store_namespace_id)
    assert head["line_count"] == 0 and head["record_id"] is None
    assert head["head_sha256"] == envelope.payload.authority_genesis_id
    assert current_supersession_head_witness(root).head_sha256 == (
        envelope.payload.authority_genesis_id
    )
    assert not list(root.glob(".STORE_NAMESPACE.json.tmp-*"))
    assert not list((root / "owner_decisions").glob(".SUPERSESSIONS.head.tmp-*"))


def test_concurrent_identical_initializers_publish_one_coherent_pair(tmp_path: Path) -> None:
    import threading

    root = tmp_path / "store"
    instance = "ab" * 16
    start = threading.Barrier(6, timeout=10)
    results: list[str] = []
    errors: list[str] = []

    def _init() -> None:
        start.wait()
        try:
            results.append(
                initialize_store_namespace(
                    root, namespace_class="test", store_instance_id=instance
                ).store_namespace_id
            )
        except BaseException as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=_init) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert len(results) == 6 and len(set(results)) == 1
    _genesis_pair_is_coherent(root, results[0])
    # HARDENING-BACKEND-FIX.1 §2: class-only requests (no explicit instance)
    # against an UNMARKED store are ALL refused — the initializer never mints
    # an instance id — and nothing is published
    other = tmp_path / "class_only"
    start2 = threading.Barrier(4, timeout=10)
    reasons: list[str] = []

    def _init_class_only() -> None:
        start2.wait()
        try:
            initialize_store_namespace(other, namespace_class="test")
        except StoreNamespaceError as error:
            reasons.append(error.reason)
        except BaseException as error:  # noqa: BLE001
            errors.append(repr(error))

    threads = [threading.Thread(target=_init_class_only) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert errors == []
    assert reasons == ["store_instance_id_required"] * 4
    assert namespace_class_of(other) is None
    assert not (other / STORE_NAMESPACE_FILE).exists()
    assert not supersession_head_path(other).exists()
    assert not list(other.glob(".STORE_NAMESPACE.json.tmp-*"))


def test_concurrent_divergent_initializers_one_wins_other_refuses(tmp_path: Path) -> None:
    import threading

    root = tmp_path / "store"
    start = threading.Barrier(2, timeout=10)
    outcomes: dict[str, object] = {}

    def _init(name: str, namespace_class: str, instance: str) -> None:
        start.wait()
        try:
            outcomes[name] = initialize_store_namespace(
                root, namespace_class=namespace_class, store_instance_id=instance
            )
        except StoreNamespaceError as error:
            outcomes[name] = error.reason
        except BaseException as error:  # noqa: BLE001
            outcomes[name] = repr(error)

    threads = [
        threading.Thread(target=_init, args=("test", "test", "1" * 32)),
        threading.Thread(target=_init, args=("research", "research", "2" * 32)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    winners = [name for name, value in outcomes.items() if not isinstance(value, str)]
    losers = [name for name, value in outcomes.items() if isinstance(value, str)]
    assert len(winners) == 1 and len(losers) == 1, outcomes
    assert outcomes[losers[0]] == "store_namespace_divergent"
    winner = outcomes[winners[0]]
    _genesis_pair_is_coherent(root, winner.store_namespace_id)
    assert load_store_namespace(root).payload.namespace_class == winners[0]


def test_crash_after_genesis_before_namespace_recovers_exactly(tmp_path: Path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import store_namespace as namespace_module

    root = tmp_path / "store"
    instance = "cd" * 16
    real_write = namespace_module._atomic_write_text

    def _crash_on_namespace(path: Path, text: str) -> None:
        if path.name == STORE_NAMESPACE_FILE:
            raise RuntimeError("simulated crash after the genesis head, before the namespace")
        real_write(path, text)

    monkeypatch.setattr(namespace_module, "_atomic_write_text", _crash_on_namespace)
    with pytest.raises(RuntimeError, match="simulated crash"):
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    monkeypatch.setattr(namespace_module, "_atomic_write_text", real_write)
    assert supersession_head_path(root).exists()
    assert not (root / STORE_NAMESPACE_FILE).exists()
    head_bytes = supersession_head_path(root).read_bytes()
    # a different class, a different instance, or no instance cannot complete it
    for kwargs in (
        {"namespace_class": "research", "store_instance_id": instance},
        {"namespace_class": "test", "store_instance_id": "ef" * 16},
        {"namespace_class": "test"},
    ):
        with pytest.raises(StoreNamespaceError) as refused:
            initialize_store_namespace(root, **kwargs)
        assert refused.value.reason == "incomplete_store_namespace_initialization"
        assert supersession_head_path(root).read_bytes() == head_bytes
        assert not (root / STORE_NAMESPACE_FILE).exists()
    assert namespace_class_of(root) is None  # still unmarked: no authority
    # the identical request recovers EXACTLY: same id as an uninterrupted
    # initialization with the same class + instance, same head bytes
    recovered = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    reference = initialize_store_namespace(
        tmp_path / "reference", namespace_class="test", store_instance_id=instance
    )
    assert recovered.store_namespace_id == reference.store_namespace_id
    assert supersession_head_path(root).read_bytes() == head_bytes
    assert (root / STORE_NAMESPACE_FILE).read_bytes() == (
        tmp_path / "reference" / STORE_NAMESPACE_FILE
    ).read_bytes()
    _genesis_pair_is_coherent(root, recovered.store_namespace_id)


def test_crash_after_namespace_before_genesis_recovers_exactly(tmp_path: Path, monkeypatch):
    from alpha_lab.agents.data_infra.ifvg.search import store_namespace as namespace_module
    from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession

    root = tmp_path / "store"
    instance = "12" * 16
    # publish the namespace and crash before the head (the write order is
    # head-then-namespace; the crash state is the same either way)
    real_write = namespace_module._atomic_write_text

    def _namespace_only(path: Path, text: str) -> None:
        if path.name == STORE_NAMESPACE_FILE:
            real_write(path, text)
            raise RuntimeError("simulated crash after the namespace, before the genesis head")
        # skip the head: it is what the crash loses
        return None

    monkeypatch.setattr(namespace_module, "_atomic_write_text", _namespace_only)
    with pytest.raises(RuntimeError, match="simulated crash"):
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    monkeypatch.setattr(namespace_module, "_atomic_write_text", real_write)
    assert (root / STORE_NAMESPACE_FILE).exists()
    assert not supersession_head_path(root).exists()
    namespace_bytes = (root / STORE_NAMESPACE_FILE).read_bytes()
    # the marked-but-headless store carries no head witness (corruption)...
    with pytest.raises(StoreNamespaceError) as missing:
        current_supersession_head_witness(root)
    assert missing.value.reason == "supersession_head_missing"
    # ...and only the identical request may complete it
    for kwargs in (
        {"namespace_class": "research", "store_instance_id": instance},
        {"namespace_class": "test", "store_instance_id": "34" * 16},
        {"namespace_class": "test"},
    ):
        with pytest.raises(StoreNamespaceError) as refused:
            initialize_store_namespace(root, **kwargs)
        assert refused.value.reason == "incomplete_store_namespace_initialization"
        assert not supersession_head_path(root).exists()
        assert (root / STORE_NAMESPACE_FILE).read_bytes() == namespace_bytes
    recovered = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    reference = initialize_store_namespace(
        tmp_path / "reference", namespace_class="test", store_instance_id=instance
    )
    assert recovered.store_namespace_id == reference.store_namespace_id
    assert (root / STORE_NAMESPACE_FILE).read_bytes() == namespace_bytes
    assert supersession_head_path(root).read_bytes() == supersession_head_path(
        tmp_path / "reference"
    ).read_bytes()
    _genesis_pair_is_coherent(root, recovered.store_namespace_id)
    # a headless store that already holds supersession RECORDS is never
    # "recovered" to genesis (that would roll the chain back)
    publish_supersession(
        root,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="one record",
        effective_at="2026-09-01T00:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    supersession_head_path(root).unlink()
    with pytest.raises(StoreNamespaceError) as rolled:
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert rolled.value.reason == "incomplete_store_namespace_initialization"
    assert not supersession_head_path(root).exists()


def test_namespace_head_mismatch_never_returns_success(tmp_path: Path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
        write_supersession_head_atomic,
    )

    root = tmp_path / "store"
    instance = "56" * 16
    namespace = initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    namespace_bytes = (root / STORE_NAMESPACE_FILE).read_bytes()
    # (i) a genesis head of ANOTHER namespace beside this envelope
    write_supersession_head_atomic(
        root, store_namespace_id="6" * 64, record_id=None, line_count=0, head_sha256="6" * 64
    )
    foreign_bytes = supersession_head_path(root).read_bytes()
    for kwargs in (
        {"namespace_class": "test", "store_instance_id": instance},
        {"namespace_class": "test"},
        {"namespace_class": "research", "store_instance_id": instance},
    ):
        with pytest.raises(StoreNamespaceError) as refused:
            initialize_store_namespace(root, **kwargs)
        assert refused.value.reason == "supersession_head_namespace_mismatch"
        # neither object was selected as authoritative and rewritten
        assert supersession_head_path(root).read_bytes() == foreign_bytes
        assert (root / STORE_NAMESPACE_FILE).read_bytes() == namespace_bytes
    # (ii) the right namespace id but a genesis anchor that is not the
    # authority genesis of this namespace
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace.store_namespace_id,
        record_id=None,
        line_count=0,
        head_sha256="9" * 64,
    )
    with pytest.raises(StoreNamespaceError) as anchor:
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert anchor.value.reason == "supersession_head_malformed"
    # (iii) a malformed head file beside a valid envelope
    supersession_head_path(root).write_text("{not json", encoding="utf-8")
    with pytest.raises(StoreNamespaceError) as malformed:
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert malformed.value.reason == "supersession_head_malformed"
    # (iv) a tampered envelope beside a valid head is never overwritten
    write_supersession_head_atomic(
        root,
        store_namespace_id=namespace.store_namespace_id,
        record_id=None,
        line_count=0,
        head_sha256=namespace.payload.authority_genesis_id,
    )
    document = json.loads(namespace_bytes)
    document["store_namespace_id"] = "0" * 64
    (root / STORE_NAMESPACE_FILE).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(StoreNamespaceError) as tampered:
        initialize_store_namespace(root, namespace_class="test", store_instance_id=instance)
    assert tampered.value.reason == "store_namespace_identity_mismatch"
    assert json.loads((root / STORE_NAMESPACE_FILE).read_text(encoding="utf-8")) == document


def test_relocated_store_retains_namespace_identity(tmp_path: Path) -> None:
    instance = "78" * 16
    origin = tmp_path / "origin" / "store"
    namespace = initialize_store_namespace(
        origin, namespace_class="test", store_instance_id=instance
    )
    text = (origin / STORE_NAMESPACE_FILE).read_text(encoding="utf-8")
    assert str(origin) not in text and str(origin.resolve()) not in text
    moved = tmp_path / "elsewhere" / "deep" / "search_test" / "v1"
    shutil.copytree(origin, moved)
    assert load_store_namespace(moved).store_namespace_id == namespace.store_namespace_id
    assert current_supersession_head_witness(moved) == current_supersession_head_witness(origin)
    # the identical request against the relocated store is idempotent reuse
    # and rewrites nothing
    before = (moved / STORE_NAMESPACE_FILE).read_bytes(), supersession_head_path(moved).read_bytes()
    again = initialize_store_namespace(moved, namespace_class="test", store_instance_id=instance)
    assert again.store_namespace_id == namespace.store_namespace_id
    assert (
        (moved / STORE_NAMESPACE_FILE).read_bytes(),
        supersession_head_path(moved).read_bytes(),
    ) == before
