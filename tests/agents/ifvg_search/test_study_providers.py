"""UI data providers: exact-ID loads, catalog-driven listing, and the
lineage-uniqueness gate on cross-profile deltas (CS §12; R2→R4 obligation)."""

from __future__ import annotations

import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import ImmutableMap
from alpha_lab.agents.data_infra.ifvg.search.lineage import (
    LineageUniquenessReport,
    NativeLineageMap,
)
from alpha_lab.agents.data_infra.ifvg.search.store import has_envelope
from alpha_lab.agents.data_infra.ifvg.study_providers import (
    costed_evaluation_id_for,
    list_catalogued_envelope_ids,
    list_search_runs,
    load_account_simulation_events,
    load_charter,
    load_child_metrics,
    load_contract_summaries,
    load_frontier_for_state,
    load_prop_vectors,
    load_search_state,
    prepare_cross_profile_deltas,
)
from tests.agents.ifvg_search.study_ui_fixture import build_completed_search


@pytest.fixture(scope="module")
def completed_search(tmp_path_factory) -> dict:
    return build_completed_search(tmp_path_factory.mktemp("study_ui"))


def test_run_listing_reads_state_root_and_catalog_names(completed_search) -> None:
    runs = list_search_runs(
        completed_search["state_root"], completed_search["store_root"]
    )
    assert [run.search_id for run in runs] == [completed_search["search_id"]]
    run = runs[0]
    assert run.display_name == "Synthetic 2x2 study"  # catalog annotation
    assert run.phase == "search_complete"
    assert run.child_count == 4
    assert not run.archived


def test_charter_state_and_frontier_load_by_exact_pointer(completed_search) -> None:
    store_root = completed_search["store_root"]
    state = load_search_state(
        completed_search["state_root"], completed_search["search_id"]
    )
    assert state is not None and state["phase"] == "search_complete"
    assert "frontier_id" in (state.get("phase_notes") or {})
    charter = load_charter(store_root, completed_search["search_id"])
    assert charter is not None
    frontier = load_frontier_for_state(store_root, state)
    assert frontier is not None
    assert (
        frontier.payload.frontier.development_exploratory_representative_id
        == completed_search["representative"]
    )
    assert load_frontier_for_state(store_root, None) is None
    assert load_frontier_for_state(store_root, {"phase_notes": {}}) is None


def test_child_metrics_load_through_the_published_costed_evaluation(
    completed_search,
) -> None:
    store_root = completed_search["store_root"]
    charter = completed_search["charter"]
    representative = completed_search["representative"]
    evaluation_id = costed_evaluation_id_for(
        representative, charter.payload.cost_policy
    )
    assert has_envelope(store_root, "costed_evaluations", evaluation_id)
    metrics = load_child_metrics(
        store_root, representative, charter.payload.cost_policy
    )
    assert metrics is not None and metrics.executed_trades > 0
    assert (
        load_child_metrics(store_root, "0" * 64, charter.payload.cost_policy)
        is None
    )


def test_catalogued_listing_never_lists_the_store_root(completed_search) -> None:
    store_root = completed_search["store_root"]
    listed = list_catalogued_envelope_ids(store_root, "account_simulations")
    assert [entry[0] for entry in listed] == [completed_search["simulation_id"]]
    # a catalogued id whose artifact is absent never appears
    from alpha_lab.agents.data_infra.ifvg.search.catalog import (
        append_catalog_event,
    )

    append_catalog_event(
        store_root,
        kind="display_name",
        artifact_id="9" * 64,
        payload={"display_name": "ghost"},
    )
    listed = list_catalogued_envelope_ids(store_root, "account_simulations")
    assert [entry[0] for entry in listed] == [completed_search["simulation_id"]]


def test_account_simulation_events_load_in_envelope_order(completed_search) -> None:
    loaded = load_account_simulation_events(
        completed_search["store_root"], completed_search["simulation_id"]
    )
    assert loaded is not None
    summary, events = loaded
    assert summary["simulation_mode"] == "historical_closed_trade"
    assert [event["event_ordinal"] for event in events] == [0, 1, 2, 3, 4]
    assert events[2]["event_type"] == "payout"
    assert events[2]["source_trade_id"] == "trade-0001"
    assert (
        load_account_simulation_events(completed_search["store_root"], "0" * 64)
        is None
    )


def test_prop_vectors_group_by_core_replay_id(completed_search) -> None:
    vectors = load_prop_vectors(completed_search["store_root"])
    assert set(vectors) == {completed_search["representative"]}
    label, summary = next(iter(vectors[completed_search["representative"]].items()))
    assert label == "synthetic_fixture_firm · fixed_one_nq"
    assert summary["payout_reliability_vector"]["expected_net_payout_90d"] == 1850.0
    assert summary["account_simulation_id"] == completed_search["simulation_id"]


def test_contract_summaries_render_truthful_synthetic_cards(completed_search) -> None:
    cards = load_contract_summaries(completed_search["store_root"])
    assert len(cards) == 1
    card = cards[0]
    assert card["firm_contract_id"] == completed_search["contract_id"]
    assert card["verification_status"] == "synthetic_fixture_verified"
    assert card["synthetic"] is True
    assert card["launchable"] is True  # selectable; REAL use still fails closed
    assert card["supersession"] == "not superseded"
    for field in (
        "drawdown_rule",
        "daily_rule",
        "contract_limits",
        "payout_rules",
        "fees",
        "post_payout",
        "path_capabilities",
    ):
        assert card[field], field


def test_unverified_intermediate_contracts_are_not_launchable(
    completed_search, tmp_path
) -> None:
    """FUX §11 (adversarial F6): first_party_evidence_compiled /
    owner_reviewed cards render NO launchable checkbox — only the two
    VERIFIED ladder endpoints are selectable."""

    from alpha_lab.agents.data_infra.ifvg.search.catalog import (
        append_catalog_event,
    )
    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        canonical_contract_sha256,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        save_envelope_immutable,
    )
    from alpha_lab.propsim.firm_contracts import (
        SYNTHETIC_FIXTURE_FIRM,
        PropFirmContractEnvelope,
    )

    store_root = tmp_path / "contracts_store"
    for status in (
        "first_party_evidence_compiled",
        "owner_reviewed",
        "superseded",
        "first_party_verified",
    ):
        payload = SYNTHETIC_FIXTURE_FIRM.model_copy(
            update={"verification_status": status}
        )
        envelope = PropFirmContractEnvelope(
            firm_contract_id=canonical_contract_sha256(payload),
            payload=payload,
            contract_evidence_bundle_id="e" * 64,
        )
        save_envelope_immutable(store_root, "prop_contracts", envelope)
        append_catalog_event(
            store_root,
            kind="display_name",
            artifact_id=envelope.firm_contract_id,
            payload={"display_name": status},
        )
    cards = {
        card["verification_status"]: card
        for card in load_contract_summaries(store_root)
    }
    assert cards["first_party_evidence_compiled"]["launchable"] is False
    assert "owner review" in cards["first_party_evidence_compiled"][
        "launch_block_reason"
    ]
    assert cards["owner_reviewed"]["launchable"] is False
    assert cards["superseded"]["launchable"] is False
    assert cards["superseded"]["launch_block_reason"] == "contract is superseded"
    assert cards["first_party_verified"]["launchable"] is True
    assert cards["first_party_verified"]["launch_block_reason"] is None


def test_verification_authorization_state_is_derived(
    completed_search, tmp_path
) -> None:
    """FUX §14.2 (adversarial F3): the UI shows WHETHER the exact ref
    exists — missing without a persisted verification run, present with one."""

    from alpha_lab.agents.data_infra.ifvg.search.authorization import (
        VerificationAuthorizationRef,
    )
    from alpha_lab.agents.data_infra.ifvg.search.catalog import (
        append_catalog_event,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        save_envelope_immutable,
    )
    from alpha_lab.agents.data_infra.ifvg.search.verification import (
        VerificationRunEnvelope,
        VerificationRunPayload,
    )
    from alpha_lab.agents.data_infra.ifvg.study_providers import (
        verification_authorization_state,
    )

    missing = verification_authorization_state(completed_search["store_root"])
    assert missing.exists is False
    assert "21/R-5" in missing.detail

    store_root = tmp_path / "auth_store"
    envelope = VerificationRunEnvelope.from_payload(
        VerificationRunPayload(
            pipeline_semantic_id="a" * 64,
            verification_authorization=VerificationAuthorizationRef(
                verification_policy_id="verification_fixed_allowlist_max5_v1",
                approved_allowlist_hash="b" * 64,
                coverage_matrix_artifact_id="c" * 64,
                seed_snapshot_id="d" * 64,
                approved_by="owner",
                approved_at="2026-08-21T00:00:00+00:00",
                content_hash="e" * 64,
            ),
            allowlist=("2026-06-04",),
            allowlist_hash="b" * 64,
            seed_snapshot_id="d" * 64,
            baseline_profile_id="ifvg_v2_doc_default_fresh_static_1r",
            baseline_section_config_hash="f" * 64,
            coverage_matrix_artifact_id="c" * 64,
        )
    )
    save_envelope_immutable(store_root, "verification_runs", envelope)
    append_catalog_event(
        store_root,
        kind="display_name",
        artifact_id=envelope.verification_run_id,
        payload={"display_name": "authorized run"},
    )
    present = verification_authorization_state(store_root)
    assert present.exists is True
    assert envelope.verification_run_id in present.verification_run_ids


def _lineage_map(core_replay_id: str, keys: dict[str, dict[str, str]]):
    per_kind = {kind: "one_to_one" for kind in keys}
    report = LineageUniquenessReport(
        core_replay_id=core_replay_id,
        per_entity_kind=ImmutableMap(per_kind),
        collisions=(),
    )
    return NativeLineageMap(
        core_replay_id=core_replay_id,
        native_to_lineage=keys,
        lineage_payloads={kind: {} for kind in keys},
        uniqueness_report=report,
        incomplete_kinds={},
    )


def test_cross_profile_deltas_persist_uniqueness_reports_first(tmp_path) -> None:
    """The R2→R4 obligation: no cross-profile delta without BOTH persisted
    lineage-uniqueness reports; validity derives from the registry."""

    baseline = _lineage_map(
        "a" * 64, {"setup": {"n1": "L1", "n2": "L2"}, "trade": {"t1": "T1"}}
    )
    challenger = _lineage_map(
        "b" * 64, {"setup": {"m1": "L1", "m3": "L3"}, "trade": {"u1": "T1"}}
    )
    deltas = prepare_cross_profile_deltas(
        tmp_path,
        baseline_map=baseline,
        challenger_map=challenger,
        changed_axis_keys=("parent_retest_timeout_1m_bars",),
        entity_kinds=("setup", "trade"),
    )
    assert deltas.lineage_valid is True
    assert has_envelope(
        tmp_path, "lineage_reports", deltas.baseline_lineage_report_id
    )
    assert has_envelope(
        tmp_path, "lineage_reports", deltas.challenger_lineage_report_id
    )
    setup_report = deltas.reports["setup"]
    assert setup_report.match_basis == "profile_independent_lineage_exact"
    assert set(setup_report.common_keys) == {"L1"}
    assert set(setup_report.added_keys) == {"L3"}
    assert set(setup_report.removed_keys) == {"L2"}


def test_lineage_breaking_axis_disables_membership_claims(tmp_path) -> None:
    baseline = _lineage_map("a" * 64, {"setup": {"n1": "L1"}})
    challenger = _lineage_map("b" * 64, {"setup": {"m1": "L1"}})
    deltas = prepare_cross_profile_deltas(
        tmp_path,
        baseline_map=baseline,
        challenger_map=challenger,
        changed_axis_keys=("min_gap_ticks_capture",),
        entity_kinds=("setup",),
    )
    assert deltas.lineage_valid is False
    assert deltas.lineage_invalid_reason
    report = deltas.reports["setup"]
    assert report.match_basis == "not_comparable"
    assert not report.common_keys and not report.added_keys
