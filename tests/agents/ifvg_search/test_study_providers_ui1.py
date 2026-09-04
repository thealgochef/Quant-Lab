"""UI-1 backend-contract handshake (plan §9 Phase 1, §10):

* the UI consumes the VERIFIED ``StoreNamespaceEnvelope`` — a local path
  never defines authority;
* typed authorization readiness distinguishes missing, stale head, wrong
  namespace, wrong profile, wrong source, store unmarked / corrupt and ready;
* run listings carry the artifact's own store, namespace class and scope.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.presentation.run_purpose import (
    AUTHORIZATION_READINESS_STATUSES,
)
from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    VERIFICATION_FIXTURE_DECISION_KEY,
    OwnerAuthorizationBundle,
    derive_authorization_requirements,
    validate_owner_authorization,
)
from alpha_lab.agents.data_infra.ifvg.search.catalog import append_catalog_event
from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import (
    initialize_test_namespace,
    load_store_namespace,
)
from alpha_lab.agents.data_infra.ifvg.search.supersession_chain import publish_supersession
from alpha_lab.agents.data_infra.ifvg.search.verification import (
    VerificationRunEnvelope,
    VerificationRunPayload,
)
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    artifact_scope_for_charter,
    list_pipeline_runs,
    list_search_runs,
    locate_charter_store,
    owner_authorization_readiness,
    resolve_store_namespace,
    verification_authorization_readiness,
    verification_owner_bundle,
)
from tests.agents.ifvg_search.namespace_fixture import (
    dummy_witness,
    verification_authorization_ref,
)
from tests.agents.ifvg_search.study_ui_fixture import build_completed_search

_ALLOWLIST = ("2026-06-04", "2026-06-05")
_PROFILE = "ifvg_v2_doc_default_fresh_static_1r"
_SECTION = "f" * 64


def _persist_verification_run(root: Path, *, ref, allowlist=_ALLOWLIST, profile=_PROFILE):
    envelope = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=ref,
            allowlist=tuple(allowlist),
            allowlist_hash=allowlist_sha256(tuple(allowlist)),
            seed_snapshot_id=ref.seed_snapshot_id,
            baseline_profile_id=profile,
            baseline_section_config_hash=_SECTION,
            coverage_matrix_artifact_id=ref.coverage_matrix_artifact_id,
        )
    )
    save_envelope_immutable(root, "verification_runs", envelope)
    append_catalog_event(
        root,
        kind="display_name",
        artifact_id=envelope.verification_run_id,
        payload={"display_name": "verification run"},
    )
    return envelope


def _bound_ref(root: Path, *, allowlist=_ALLOWLIST):
    return verification_authorization_ref(
        root,
        approved_allowlist_hash=allowlist_sha256(tuple(allowlist)),
        coverage_matrix_artifact_id="c" * 64,
        seed_snapshot_id="d" * 64,
    )


@pytest.fixture(scope="module")
def completed_search(tmp_path_factory) -> dict:
    return build_completed_search(tmp_path_factory.mktemp("providers_ui1"))


def test_ui_uses_verified_store_namespace_contract(tmp_path) -> None:
    research_looking = tmp_path / "data" / "ifvg_datasets" / "search" / "v1"
    research_looking.mkdir(parents=True)
    state = resolve_store_namespace(research_looking, expected_class="research")
    assert state.status == "unmarked" and state.store_namespace_id is None
    marked = tmp_path / "marked"
    initialize_test_namespace(marked)
    envelope = load_store_namespace(marked)
    verified = resolve_store_namespace(marked, expected_class="test")
    assert verified.status == "verified"
    assert verified.store_namespace_id == envelope.store_namespace_id


def test_ui_authorization_states_match_backend_contracts(tmp_path) -> None:
    """Every typed readiness the providers derive is a registered status and
    each failure keeps the backend's own reason."""

    # unmarked store: no semantic authority at all
    unmarked = verification_authorization_readiness(tmp_path / "unmarked")
    assert unmarked.status == "store_unmarked"

    root = tmp_path / "store"
    initialize_test_namespace(root)
    missing = verification_authorization_readiness(root)
    assert missing.status == "missing"
    assert "21/R-5" in missing.detail
    assert missing.store_namespace_id == load_store_namespace(root).store_namespace_id

    # a ref bound to THIS store's namespace and current head → ready
    ref = _bound_ref(root)
    run = _persist_verification_run(root, ref=ref)
    ready = verification_authorization_readiness(
        root,
        baseline_profile_name=_PROFILE,
        baseline_section_config_hash=_SECTION,
        allowlist=_ALLOWLIST,
    )
    assert ready.status == "ready"
    assert ready.evidence_ids == (run.verification_run_id,)

    # wrong profile / wrong source are distinguished from ready
    wrong_profile = verification_authorization_readiness(
        root, baseline_profile_name="ifvg_v2_other_profile"
    )
    assert wrong_profile.status == "wrong_profile"
    wrong_source = verification_authorization_readiness(
        root, allowlist=("2026-06-08", "2026-06-09")
    )
    assert wrong_source.status == "wrong_source"

    # the bundle a REAL verification charter carries validates against the
    # verification requirement set and binds the store
    requirement_set = derive_authorization_requirements("verification_5d", (), None, (), ())
    bundle = verification_owner_bundle(root, ready, requirement_set)
    assert isinstance(bundle, OwnerAuthorizationBundle)
    assert VERIFICATION_FIXTURE_DECISION_KEY in bundle.decision_refs
    validate_owner_authorization(
        bundle, requirement_set, as_of_utc="2026-09-04T00:00:00Z", store_root=root
    )
    assert verification_owner_bundle(root, missing, requirement_set) is None

    # the head moves after signing → stale head, never ready
    publish_supersession(
        root,
        superseded_decision_id="a" * 64,
        replacement_decision_id="b" * 64,
        reason="moved after signing",
        effective_at="2026-08-18T01:00:00+00:00",
        owner_evidence_ref="b" * 64,
    )
    stale = verification_authorization_readiness(root)
    assert stale.status in ("stale_head", "wrong_head", "superseded")
    assert stale.status != "ready"

    # a ref naming ANOTHER namespace → wrong_namespace
    other = tmp_path / "other"
    initialize_test_namespace(other)
    foreign = verification_authorization_ref(
        None,
        approved_allowlist_hash=allowlist_sha256(_ALLOWLIST),
        coverage_matrix_artifact_id="c" * 64,
        seed_snapshot_id="d" * 64,
    )
    assert foreign.supersession_head_witness == dummy_witness()
    _persist_verification_run(other, ref=foreign)
    assert verification_authorization_readiness(other).status == "wrong_namespace"

    # a corrupt namespace envelope → store_corrupt
    (other / "STORE_NAMESPACE.json").write_text("{not json", encoding="utf-8")
    assert verification_authorization_readiness(other).status == "store_corrupt"

    for readiness in (unmarked, missing, ready, wrong_profile, wrong_source, stale):
        assert readiness.status in AUTHORIZATION_READINESS_STATUSES


def test_owner_authorization_readiness_names_missing_decisions(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.study.computation_path import ComputationPath

    path = ComputationPath(
        full_strategy_replay=True,
        feature_materialization=False,
        label_recomputation=False,
        model_refit=False,
        model_gated_sequential_replay=False,
        cost_recomputation=True,
        prop_resimulation=False,
        bootstrap_resimulation=False,
        reuse_trade_stream_hash=False,
    )
    requirement_set = derive_authorization_requirements(
        "full_authorized_development",
        ("strategy_profile.parent_retest_timeout_1m_bars",),
        path,
        (),
        (),
    )
    assert requirement_set.payload.requirements  # the search path needs decisions
    unmarked = owner_authorization_readiness(tmp_path / "research", requirement_set)
    assert unmarked.status == "store_unmarked"
    root = tmp_path / "marked_test"
    initialize_test_namespace(root)
    # a test namespace is not the research class the purpose requires
    assert owner_authorization_readiness(root, requirement_set).status == "wrong_namespace"
    missing = owner_authorization_readiness(root, requirement_set, expected_class="test")
    assert missing.status == "missing"
    assert set(missing.missing_decision_keys) == {
        requirement.decision_key for requirement in requirement_set.payload.requirements
    }


def test_run_listings_carry_the_artifact_store_namespace_and_scope(
    completed_search, tmp_path
) -> None:
    runs = list_search_runs(completed_search["state_root"], completed_search["store_root"])
    assert len(runs) == 1
    run = runs[0]
    assert run.store_root == str(completed_search["store_root"])
    assert run.namespace_class is None  # an unmarked tmp store is never guessed
    assert run.verification_only is True  # a synthetic-marker charter
    scope = artifact_scope_for_charter(completed_search["charter"])
    assert scope.evidence_class == "synthetic_fixture"
    assert scope.run_scope == "synthetic_fixture"
    assert scope.verification_only is True
    # the charter is located across the KNOWN roots by exact id; an empty
    # decoy root never claims it
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    assert locate_charter_store(run.search_id, (decoy, completed_search["store_root"])) == (
        completed_search["store_root"]
    )
    assert locate_charter_store(run.search_id, (decoy,)) is None
    listed = list_search_runs(
        completed_search["state_root"], decoy, store_roots=(completed_search["store_root"],)
    )
    assert listed[0].store_root == str(completed_search["store_root"])
    assert listed[0].display_name == "Synthetic 2x2 study"  # annotations from ITS store
    assert list_pipeline_runs(tmp_path / "no_pipeline_state") == ()
