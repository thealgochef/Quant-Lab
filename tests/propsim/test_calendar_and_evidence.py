"""DayCountBasis/clock semantics (P0-14) + the contract-evidence ladder (P0-17/P1-A)."""

from __future__ import annotations

from datetime import date

import pytest

from alpha_lab.agents.data_infra.ifvg.search.authorization import (
    OwnerDecisionEvidenceRef,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    canonical_contract_sha256,
)
from alpha_lab.propsim.calendar import (
    BOOTSTRAP_CLOCK_POLICY,
    HISTORICAL_CLOCK_POLICY,
    DayCountBasis,
    DurationRule,
    FirmCalendarPolicy,
    SimulatedClock,
    UnsupportedCalendarRuleError,
)
from alpha_lab.propsim.contract_evidence import (
    REQUIRED_EVIDENCE_FIELDS,
    ContractStatusError,
    PropContractDraft,
    PropContractReviewDecision,
    PropContractSourceDocument,
    PropRuleEvidence,
    advance_verification_status,
    compile_contract_draft,
)
from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM


def _walk_days(clock: SimulatedClock, spec: list[tuple[str, bool]]) -> None:
    for day, winning in spec:
        clock.advance(date.fromisoformat(day), winning=winning)


def test_bases_advance_correctly_through_a_weekend_and_holiday() -> None:
    calendar = FirmCalendarPolicy(
        policy_id="test_calendar", holidays=("2026-01-19",)  # the Monday holiday
    )
    clock = SimulatedClock(policy=HISTORICAL_CLOCK_POLICY, calendar=calendar)
    _walk_days(
        clock,
        [
            ("2026-01-15", True),   # Thu
            ("2026-01-16", False),  # Fri
            ("2026-01-20", True),   # Tue (Mon 19th = holiday)
            ("2026-01-21", True),   # Wed
        ],
    )
    assert clock.elapsed_since(0, DayCountBasis.TRADING_DAY) == 3
    assert clock.elapsed_since(0, DayCountBasis.WINNING_DAY) == 2
    assert clock.elapsed_since(0, DayCountBasis.CALENDAR_DAY) == 6
    # Fri 16 + Tue 20 + Wed 21 are business days after Thu 15; Mon 19 is not
    assert clock.elapsed_since(0, DayCountBasis.BUSINESS_DAY) == 3
    assert clock.total(DayCountBasis.WINNING_DAY) == 3
    assert clock.satisfied(DurationRule(count=3, basis=DayCountBasis.TRADING_DAY))
    assert not clock.satisfied(DurationRule(count=4, basis=DayCountBasis.TRADING_DAY))


def test_calendar_month_rule_under_bootstrap_fails_closed() -> None:
    """P0-14: a calendar-month fee under day-block bootstrap is unsupported."""

    clock = SimulatedClock(policy=BOOTSTRAP_CLOCK_POLICY)
    _walk_days(clock, [("2020-01-06", True), ("2020-01-07", False)])
    assert clock.elapsed_since(0, DayCountBasis.TRADING_DAY) == 1
    with pytest.raises(UnsupportedCalendarRuleError, match="calendar_month"):
        clock.elapsed_since(0, DayCountBasis.CALENDAR_MONTH)
    with pytest.raises(UnsupportedCalendarRuleError, match="business_day"):
        clock.satisfied(DurationRule(count=1, basis=DayCountBasis.BUSINESS_DAY))


def test_clock_only_advances_forward() -> None:
    clock = SimulatedClock(policy=HISTORICAL_CLOCK_POLICY)
    clock.advance(date(2026, 1, 15), winning=True)
    with pytest.raises(ValueError, match="only advances forward"):
        clock.advance(date(2026, 1, 15), winning=False)


# ── contract evidence ────────────────────────────────────────────────────────


def _documents(*, synthetic: bool) -> tuple[PropContractSourceDocument, ...]:
    scheme = "synthetic://fixture" if synthetic else "https://firm.example"
    return (
        PropContractSourceDocument(
            source_document_id="doc-1",
            content_sha256="1" * 64,
            provenance_url=f"{scheme}/rules",
            retrieved_at_utc="2026-08-18T00:00:00Z",
            effective_from="2026-01-01",
            effective_to=None,
            document_kind="official_rules",
        ),
    )


def _draft(conflict: bool = False, missing: bool = False) -> PropContractDraft:
    rows = []
    fields = list(REQUIRED_EVIDENCE_FIELDS)
    if missing:
        fields = fields[:-1]
    for index, field_path in enumerate(fields):
        rows.append(
            PropRuleEvidence(
                field_path=field_path,
                source_document_id="doc-1",
                locator=f"section-{index}",
                normalized_value_json="null",
                reviewer=None,
                conflict_status=(
                    "conflicting_sources" if conflict and index == 0 else "none"
                ),
            )
        )
    return PropContractDraft(
        contract_payload=SYNTHETIC_FIXTURE_FIRM, evidence_rows=tuple(rows)
    )


def test_compilation_requires_full_field_coverage_and_no_conflicts() -> None:
    documents = _documents(synthetic=True)
    passing = compile_contract_draft(_draft(), documents)
    assert passing.passed and passing.field_coverage_complete
    conflicted = compile_contract_draft(_draft(conflict=True), documents)
    assert not conflicted.passed
    assert any("conflicting_sources" in item for item in conflicted.unresolved_conflicts)
    incomplete = compile_contract_draft(_draft(missing=True), documents)
    assert not incomplete.passed and not incomplete.field_coverage_complete
    orphaned = compile_contract_draft(_draft(), ())
    assert not orphaned.passed
    assert any(item.startswith("orphan_source:") for item in orphaned.unresolved_conflicts)


def _review(draft_hash: str) -> PropContractReviewDecision:
    return PropContractReviewDecision(
        draft_hash=draft_hash,
        owner_decision_ref=OwnerDecisionEvidenceRef(
            decision_id="D-005",
            decision_artifact_id="artifact-1",
            content_hash="2" * 64,
            author="owner",
            approved_at="2026-08-18T00:00:00Z",
            effective_from="2026-08-18",
            reviewed_evidence_refs=("doc-1",),
        ),
        verdict="approved",
    )


def test_synthetic_evidence_can_never_reach_first_party_verified() -> None:
    """P1-A: the ladder ordering + the synthetic cap, with no override."""

    synthetic_docs = _documents(synthetic=True)
    compilation = compile_contract_draft(_draft(), synthetic_docs)
    assert (
        advance_verification_status(
            "synthetic_fixture_verified",
            "synthetic_fixture_verified",
            documents=synthetic_docs,
        )
        == "synthetic_fixture_verified"
    )
    for target in (
        "first_party_evidence_compiled",
        "owner_reviewed",
        "first_party_verified",
    ):
        with pytest.raises(ContractStatusError, match="synthetic"):
            advance_verification_status(
                "synthetic_fixture_verified",
                target,
                documents=synthetic_docs,
                compilation=compilation,
                review=_review(compilation.draft_hash),
            )

    first_party_docs = _documents(synthetic=False)
    compilation = compile_contract_draft(_draft(), first_party_docs)
    status = advance_verification_status(
        "first_party_evidence_compiled",
        "first_party_verified",
        documents=first_party_docs,
        compilation=compilation,
        review=_review(compilation.draft_hash),
    )
    assert status == "first_party_verified"
    # an unresolved conflict blocks progression
    conflicted = compile_contract_draft(_draft(conflict=True), first_party_docs)
    with pytest.raises(ContractStatusError, match="PASSED"):
        advance_verification_status(
            "synthetic_fixture_verified",
            "first_party_evidence_compiled",
            documents=first_party_docs,
            compilation=conflicted,
        )
    # the ladder is one-way, and superseded is terminal
    with pytest.raises(ContractStatusError, match="one-way"):
        advance_verification_status(
            "owner_reviewed",
            "first_party_evidence_compiled",
            documents=first_party_docs,
            compilation=compilation,
        )
    assert (
        advance_verification_status(
            "first_party_verified", "superseded", documents=first_party_docs
        )
        == "superseded"
    )
    with pytest.raises(ContractStatusError, match="never advance"):
        advance_verification_status(
            "superseded", "owner_reviewed", documents=first_party_docs
        )
    # a review for a DIFFERENT draft is refused
    with pytest.raises(ContractStatusError, match="DIFFERENT draft"):
        advance_verification_status(
            "first_party_evidence_compiled",
            "owner_reviewed",
            documents=first_party_docs,
            compilation=compilation,
            review=_review(canonical_contract_sha256({"other": "draft"})),
        )


def test_adverse_first_never_appears_in_firm_phase_rules() -> None:
    """§3.10 firm/scenario separation: ordering lives ONLY in scenario policies."""

    from pathlib import Path

    import alpha_lab.propsim.firm_contracts as firm_module

    source = Path(firm_module.__file__).read_text(encoding="utf-8")
    assert "adverse_first" not in source
    assert "favorable_first" not in source
    from alpha_lab.propsim.firm_contracts import PhaseRules

    assert "intrabar" not in " ".join(PhaseRules.model_fields)

def test_supersession_retires_the_prior_contract_id() -> None:
    """P0-17: supersession retires the old id for NEW work, with effective_at."""

    from alpha_lab.propsim.contract_evidence import (
        PropContractSupersession,
        SupersededContractError,
        assert_contract_not_superseded,
    )

    supersession = PropContractSupersession(
        prior_firm_contract_id="1" * 64,
        replacement_firm_contract_id="2" * 64,
        reason="rules revision v2",
        effective_at="2026-01-20T00:00:00+00:00",
    )
    with pytest.raises(SupersededContractError, match="superseded"):
        assert_contract_not_superseded("1" * 64, (supersession,))
    # the replacement id (and any unrelated id) remains usable
    assert_contract_not_superseded("2" * 64, (supersession,))
    assert_contract_not_superseded("3" * 64, (supersession,))


def test_status_advancement_binds_the_compilations_document_set() -> None:
    """The synthetic cap cannot be laundered by presenting a different (or
    empty) document set than the compilation actually cited."""

    from alpha_lab.propsim.contract_evidence import (
        ContractStatusError,
        advance_verification_status,
        compile_contract_draft,
    )

    documents = _documents(synthetic=True)
    compilation = compile_contract_draft(_draft(), documents)
    assert compilation.passed
    with pytest.raises(ContractStatusError, match="not.*presented"):
        advance_verification_status(
            "synthetic_fixture_verified",
            "first_party_evidence_compiled",
            documents=(),
            compilation=compilation,
        )
