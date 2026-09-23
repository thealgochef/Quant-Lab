"""Compatibility identity and authority regressions using temporary metadata only."""

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.search import (
    research_executor,
    research_runs,
    research_subject,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    CoreStrategyReplayIdentity,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.research_compatibility import (
    COMPATIBILITY_STORE,
    ResearchCoreCompatibilityProof,
)
from alpha_lab.agents.data_infra.ifvg.search.research_data import (
    CONTEXT_STORE,
    ResearchContextCompatiblePayload,
    ResearchContextEnvelope,
    ResearchContextPayload,
    load_research_context_companion,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    load_verified_envelope,
    save_or_reuse_envelope,
)
from tests.agents.ifvg_search.test_real_research_launch import frozen
from tests.agents.ifvg_search.test_real_research_launch import sources as launch_sources
from tests.agents.ifvg_search.test_research_subject_data import (
    compatibility_fixture,
    subject_fixture,
)


@pytest.fixture(name="sources")
def metadata_sources(tmp_path, monkeypatch):
    return launch_sources.__wrapped__(tmp_path, monkeypatch)


def legacy_context_record(subject):
    # Historical schema: no compatibility field, including no null placeholder.
    payload = {
        "schema_version": 1,
        "subject": subject.model_dump(mode="json"),
        "producer_source_hash": "1" * 64,
        "feature_schema_hash": "2" * 64,
        "context_config_hash": "3" * 64,
        "context_config_json": '{"fixture": true}',
        "neutrality_policy": "all_accepted_core_tables_exact_v1",
    }
    return {
        "research_context_companion_id": canonical_contract_sha256(payload),
        "payload": payload,
    }


def altered_proof(subject):
    original = compatibility_fixture(subject)
    return ResearchCoreCompatibilityProof.from_payload(
        original.payload.model_copy(update={"runtime_environment_json": '{"fixture": false}'})
    )


def test_legacy_context_record_keeps_identity_and_needs_no_proof_store(tmp_path):
    subject = subject_fixture()
    historical = legacy_context_record(subject)
    envelope = ResearchContextEnvelope.model_validate_json(json.dumps(historical))
    assert type(envelope.payload) is ResearchContextPayload
    assert envelope.model_dump(mode="json") == historical
    assert envelope.payload.subject == subject
    save_or_reuse_envelope(tmp_path, CONTEXT_STORE, envelope)
    reloaded = load_verified_envelope(
        tmp_path, CONTEXT_STORE, historical["research_context_companion_id"],
        ResearchContextEnvelope,
    )
    assert reloaded.model_dump(mode="json") == historical
    assert not (tmp_path / COMPATIBILITY_STORE).exists()


def test_new_context_binds_proof_without_rewriting_saved_source_identity():
    subject = subject_fixture()
    legacy = legacy_context_record(subject)
    proof = compatibility_fixture(subject)
    compatible = ResearchContextEnvelope.from_payload(
        ResearchContextCompatiblePayload.model_validate(
            {
                **legacy["payload"],
                "schema_version": 2,
                "core_compatibility_proof_id": proof.proof_id,
            }
        )
    )
    reloaded = ResearchContextEnvelope.model_validate_json(compatible.model_dump_json())
    assert type(reloaded.payload) is ResearchContextCompatiblePayload
    assert reloaded.research_context_companion_id != legacy["research_context_companion_id"]
    assert reloaded.payload.subject.model_dump(mode="json") == legacy["payload"]["subject"]
    assert reloaded.payload.subject.subject_id == subject.subject_id
    original_core = CoreStrategyReplayIdentity.model_validate_json(subject.core_envelope_json)
    assert original_core.core_replay_id == subject.core_replay_id
    assert reloaded.payload.subject.core_envelope_json == subject.core_envelope_json
    changed = ResearchContextEnvelope.from_payload(
        reloaded.payload.model_copy(
            update={"core_compatibility_proof_id": altered_proof(subject).proof_id}
        )
    )
    assert changed.research_context_companion_id != reloaded.research_context_companion_id


@pytest.mark.parametrize("problem", ["different_core", "missing_sidecar", "different_sidecar"])
def test_context_loader_rejects_invalid_proof_binding_before_loading_tables(tmp_path, problem):
    subject = subject_fixture()
    proof = compatibility_fixture(subject)
    if problem == "different_core":
        proof = ResearchCoreCompatibilityProof.from_payload(
            proof.payload.model_copy(update={"core_replay_id": "f" * 64})
        )
    save_or_reuse_envelope(tmp_path, COMPATIBILITY_STORE, proof)
    envelope = ResearchContextEnvelope.from_payload(
        ResearchContextCompatiblePayload.model_validate(
            {
                **legacy_context_record(subject)["payload"],
                "schema_version": 2,
                "core_compatibility_proof_id": proof.proof_id,
            }
        )
    )
    sidecars = {}
    if problem != "missing_sidecar":
        recorded = altered_proof(subject) if problem == "different_sidecar" else proof
        sidecars["core_compatibility.json"] = recorded.model_dump_json().encode()
    save_or_reuse_envelope(tmp_path, CONTEXT_STORE, envelope, extra_files=sidecars)
    messages = {
        "different_core": "compatibility proof differs from its exact source subject",
        "missing_sidecar": "core_compatibility.json",
        "different_sidecar": "compatibility evidence differs from the verified proof",
    }
    with pytest.raises(ValueError, match=messages[problem]):
        load_research_context_companion(tmp_path, envelope.research_context_companion_id)


@pytest.mark.parametrize("proof_present", [False, True])
def test_worker_rejects_missing_or_changed_fresh_proof_before_adapter_construction(
    tmp_path, sources, monkeypatch, proof_present
):
    _, _, semantic, charter = frozen(tmp_path, sources)
    facts = {"passed": True}
    if proof_present:
        facts["core_compatibility_proof"] = altered_proof(sources).model_dump(mode="json")
    monkeypatch.setattr(research_executor, "preflight_research_subject", lambda *args: facts)
    monkeypatch.setattr(
        research_executor, "ResearchPreparation",
        lambda *args, **kwargs: pytest.fail("changed evidence reached adapter construction"),
    )
    with pytest.raises(PermissionError, match="compatibility evidence differs"):
        research_executor.pipeline_real_research_entry(charter, semantic, store_root=tmp_path)


def test_worker_rechecks_proof_after_factory_before_source_stage(tmp_path, sources, monkeypatch):
    _, _, semantic, charter = frozen(tmp_path, sources)
    wiring = research_executor.pipeline_real_research_entry(charter, semantic, store_root=tmp_path)
    assert wiring.research_preparation.replay_invocations == 0
    monkeypatch.setattr(
        research_executor, "preflight_research_subject",
        lambda *args: {"core_compatibility_proof": altered_proof(sources).model_dump(mode="json")},
    )
    with pytest.raises(PermissionError, match="compatibility evidence differs"):
        wiring.research_authorization_check()
    assert wiring.research_preparation.replay_invocations == 0


def test_worker_rejects_tampered_persisted_proof_even_when_fresh_proof_matches(
    tmp_path, sources, monkeypatch
):
    _, _, semantic, charter = frozen(tmp_path, sources)
    proof = compatibility_fixture(sources)
    path = tmp_path / COMPATIBILITY_STORE / proof.proof_id / "envelope.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["payload"]["runtime_environment_json"] = '{"fixture": false}'
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(
        research_executor, "ResearchPreparation",
        lambda *args, **kwargs: pytest.fail("corrupt proof reached adapter construction"),
    )
    with pytest.raises(SearchStoreError):
        research_executor.pipeline_real_research_entry(charter, semantic, store_root=tmp_path)


def test_changed_proof_after_preflight_requires_review_before_any_approval(
    tmp_path, sources, monkeypatch
):
    preflight = research_runs.build_research_preflight(
        tmp_path, [sources.core_replay_id],
        {
            "display_name": "Compatibility review fixture",
            "evaluation_start": "2026-01-13",
            "evaluation_end": "2026-01-14",
        },
    )
    monkeypatch.setattr(
        research_subject, "preflight_research_subject",
        lambda *args: {"core_compatibility_proof": altered_proof(sources).model_dump(mode="json")},
    )
    with pytest.raises(ValueError, match="changed after review"):
        research_runs.freeze_research_group(
            tmp_path, preflight, author="Fixture reviewer", authorization_statement="Reviewed"
        )
    assert not (tmp_path / "research_approvals").exists()
    assert not (tmp_path / COMPATIBILITY_STORE).exists()
