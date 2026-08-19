"""The PHASED R3 gate end-to-end: compile → contract → simulate → gates.

ONE synthetic contract fixture is compiled from synthetic source documents
(reaching, and capped at, ``synthetic_fixture_verified``), then simulated end
to end on the R1 baseline stream — the same synthetic resolved
EXECUTED_TRADE fixture the R1 vertical slice and R2 orchestration E2E run on
— through the PRODUCTION chain: adapters → trade-path artifacts → bundle →
capability report → account walk → payout reliability vector →
``evaluate_prop_gates`` → the orchestrator frontier. No piece is fabricated;
every identity the simulation pins is built from the objects consumed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.charter import (
    ResolvedPropGateThresholds,
)
from alpha_lab.agents.data_infra.ifvg.search.gates import evaluate_prop_gates
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import (
    enumerate_children,
    read_search_state,
    run_search,
)
from alpha_lab.propsim.contract_evidence import (
    REQUIRED_EVIDENCE_FIELDS,
    ContractStatusError,
    PropContractDraft,
    PropContractSourceDocument,
    PropRuleEvidence,
    advance_verification_status,
    compile_contract_draft,
)
from alpha_lab.propsim.firm_contracts import (
    SYNTHETIC_FIXTURE_FIRM,
    PropFirmContractEnvelope,
)
from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY
from alpha_lab.propsim.search_bridge import FirmSimulationSpec, make_prop_simulator
from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY
from tests.agents.ifvg_search.conftest import (
    SYNTHETIC_DAYS,
    make_resolved_trades_frame,
)
from tests.agents.ifvg_search.test_orchestrator import (
    _charter,
    _identity_resolver,
    _runner_factory,
)

_TICK_SIZE = 0.25


def _synthetic_documents() -> tuple[PropContractSourceDocument, ...]:
    return (
        PropContractSourceDocument(
            source_document_id="doc-rules-v1",
            content_sha256="1" * 64,
            provenance_url="synthetic://fixture/contract-v1/rules",
            retrieved_at_utc="2026-01-02T00:00:00+00:00",
            effective_from="2026-01-01",
            effective_to=None,
            document_kind="official_rules",
        ),
    )


def _compiled_synthetic_contract():
    """Compile the synthetic fixture firm from its synthetic evidence."""

    rows = tuple(
        PropRuleEvidence(
            field_path=field_path,
            source_document_id="doc-rules-v1",
            locator=f"synthetic:{field_path}",
            normalized_value_json="null",
            reviewer=None,
            conflict_status="none",
        )
        for field_path in REQUIRED_EVIDENCE_FIELDS
    )
    draft = PropContractDraft(
        contract_payload=SYNTHETIC_FIXTURE_FIRM, evidence_rows=rows
    )
    documents = _synthetic_documents()
    compilation = compile_contract_draft(draft, documents)
    assert compilation.passed and compilation.field_coverage_complete
    status = advance_verification_status(
        "synthetic_fixture_verified",
        "synthetic_fixture_verified",
        documents=documents,
        compilation=compilation,
    )
    assert status == "synthetic_fixture_verified"
    # the synthetic cap holds inside the same end-to-end chain
    with pytest.raises(ContractStatusError, match="never support"):
        advance_verification_status(
            "synthetic_fixture_verified",
            "first_party_evidence_compiled",
            documents=documents,
            compilation=compilation,
        )
    envelope = PropFirmContractEnvelope(
        firm_contract_id=canonical_contract_sha256(SYNTHETIC_FIXTURE_FIRM),
        payload=SYNTHETIC_FIXTURE_FIRM,
        contract_evidence_bundle_id=compilation.draft_hash,
    )
    return envelope, compilation


def _simulator():
    envelope, _compilation = _compiled_synthetic_contract()
    spec = FirmSimulationSpec(
        label="synthetic_fixture_firm",
        firm=envelope.payload,
        risk_policy=FIXED_ONE_NQ_RISK_POLICY,
        withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
    )
    return make_prop_simulator(
        (spec,),
        tick_size=_TICK_SIZE,
        costed_evaluation_id_for=lambda core_id: canonical_contract_sha256(
            {"costed_for": core_id}
        ),
    )


def test_synthetic_contract_compiles_and_simulates_end_to_end() -> None:
    """The gate sentence, verbatim: compiled and simulated end to end on the
    R1 baseline stream, through the production chain."""

    simulator = _simulator()
    frame = make_resolved_trades_frame(SYNTHETIC_DAYS)
    outcome = SimpleNamespace(core_replay_id="a" * 64)
    result = SimpleNamespace(tables={RecordTable.EXECUTED_TRADE: frame})
    vectors = simulator(outcome=outcome, result=result)
    assert set(vectors) == {"synthetic_fixture_firm"}
    vector = vectors["synthetic_fixture_firm"]
    # the vector is a real measurement of the walked stream, not a stub
    assert 0.0 <= vector.breach_probability_90d <= 1.0
    assert 0.0 <= vector.first_payout_probability_60d <= 1.0
    # and it feeds the search gates with all six thresholds covered
    report = evaluate_prop_gates(
        vector,
        ResolvedPropGateThresholds(
            minimum_first_payout_probability_60d=0.0,
            maximum_breach_probability_90d=1.0,
            minimum_expected_net_payout_90d=None,
            minimum_p10_net_payout_90d=None,
            maximum_p90_payout_drought_days=None,
            minimum_three_payout_probability=None,
        ),
    )
    assert len(report.checks) == 6
    assert report.passed
    # determinism: the same stream simulates to the identical vector
    assert simulator(outcome=outcome, result=result) == vectors
    # a reused child without rebuilt tables is an explicit refusal, never an
    # implicit re-read
    with pytest.raises(ValueError, match="REUSED"):
        simulator(outcome=outcome, result=None)


def test_production_simulator_wires_the_frontier_through_run_search(tmp_path) -> None:
    """PHASED R3 'frontier wired to prop metrics' — through REAL simulations."""

    charter = _charter(
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
        lexicographic_tie_breaks=("core_replay_id",),
        # thresholds a 3-day synthetic stream can honestly satisfy: the test
        # proves the WIRING (vector → gates → worst-firm merge → frontier),
        # not any profitability claim about the fixture
        prop_feasibility_gates=ResolvedPropGateThresholds(
            minimum_first_payout_probability_60d=0.0,
            maximum_breach_probability_90d=1.0,
            minimum_expected_net_payout_90d=None,
            minimum_p10_net_payout_90d=None,
            maximum_p90_payout_drought_days=None,
            minimum_three_payout_probability=None,
        ),
    )
    specs = enumerate_children(
        charter,
        identity_resolver=lambda spec: _identity_resolver(spec).core_replay_id,
    )
    baseline = next(s for s in specs if s.comparison_role == "baseline")
    passing = {baseline.resolved_section_config_hash}
    result = run_search(
        charter,
        store_root=tmp_path / "store",
        state_root=tmp_path / "state",
        identity_resolver=_identity_resolver,
        child_runner=_runner_factory([], passing),
        prop_simulator=_simulator(),
        stale_lock_seconds=3600.0,
    )
    assert result.phase == "search_complete"
    passing_children = [
        outcome
        for outcome in result.children
        if outcome.gate_report is not None and outcome.gate_report.passed
    ]
    assert len(passing_children) == 1
    assert result.frontier is not None
    assert (
        result.frontier.development_exploratory_representative_id
        == passing_children[0].core_replay_id
    )
    champions = dict(result.frontier.per_objective_champions)
    assert champions["expected_net_payout_90d"] == passing_children[0].core_replay_id
    state = read_search_state(tmp_path / "state", charter.search_id)
    assert "prop_simulations" not in state["phase_notes"]
