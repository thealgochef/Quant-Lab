"""HARDENING-BACKEND Phase 3 §5.1 — the logical-window coverage shortlist.

Rebuilt from already-authorized evidence only (typed tables + the audit
funnel's day set + the accepted inventory); ranked lexicographically in the
exact §5.1 order with no hidden score; never registers or hashes a
permanent allowlist; never selects for the owner.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
    logical_trading_days,
    store_day_chain,
)
from alpha_lab.agents.data_infra.ifvg.search.verification_window import (
    JUNE_PROPOSAL_WINDOW,
    LOGICAL_WINDOW_RANKING_ORDER,
    R1_STORE_DAY_CANDIDATE,
    VerificationWindowShortlist,
    build_logical_day_coverage,
    build_verification_window_shortlist,
    rank_logical_windows,
    render_shortlist_markdown,
)


def _inventory(days: tuple[str, ...]) -> dict[str, tuple[str, str]]:
    return {day: ("mbp1", f"{index:064x}") for index, day in enumerate(days, start=1)}


def _frame(rows: dict[str, int], *, with_setups: bool = False) -> pd.DataFrame:
    records = []
    ordinal = 0
    for day, count in rows.items():
        for _ in range(count):
            ordinal += 1
            record = {"envelope_trading_day": day, "envelope_setup_id": f"setup-{ordinal:04d}"}
            if with_setups:
                record["setup_id"] = f"setup-{ordinal:04d}"
                record["status"] = "resolved"
            records.append(record)
    return pd.DataFrame(records)


@pytest.fixture()
def synthetic_evidence():
    # an inventory covering the whole store-day chain 2026-01-01 … 2026-02-20
    inventory = _inventory(store_day_chain("2026-01-01", "2026-02-20"))
    lifecycle = _frame(
        {
            "2026-02-05": 10,
            "2026-02-06": 30,
            "2026-02-09": 40,
            "2026-02-10": 20,
            "2026-02-11": 25,
            "2026-02-12": 5,
            "2026-02-13": 5,
            "2026-01-20": 3,
        }
    )
    candidates = _frame(
        {"2026-02-06": 6, "2026-02-09": 10, "2026-02-10": 4, "2026-02-11": 6, "2026-02-12": 1}
    )
    decisions = _frame(
        {"2026-02-05": 1, "2026-02-06": 2, "2026-02-09": 3, "2026-02-10": 1, "2026-02-11": 2}
    )
    trades = _frame(
        {"2026-02-05": 1, "2026-02-06": 2, "2026-02-09": 3, "2026-02-10": 1, "2026-02-11": 2},
        with_setups=True,
    )
    tables = {
        RecordTable.SETUP_LIFECYCLE: lifecycle,
        RecordTable.ENTRY_CANDIDATE: candidates,
        RecordTable.ELIGIBLE_DECISION: decisions,
        RecordTable.EXECUTED_TRADE: trades,
        RecordTable.CANDIDATE_LABEL: candidates.copy(),
    }
    funnel_days = frozenset(logical_trading_days("2026-01-02", "2026-02-20")) - {"2026-02-13"}
    return {"inventory": inventory, "tables": tables, "funnel_days": funnel_days}


def test_ranking_order_is_the_plan_order_with_no_hidden_score() -> None:
    assert LOGICAL_WINDOW_RANKING_ORDER == (
        "required_source_and_replay_coverage",
        "lifecycle_path_classes_represented",
        "audit_day_coverage",
        "executed_trades",
        "decisions",
        "candidates",
        "exact_verifier_targets",
        "mbp1_scope_evidence",
        "panel_control_flow_coverage",
        "distance_from_protected_boundary",
    )


def test_day_coverage_is_computed_on_logical_days_with_both_partitions(synthetic_evidence) -> None:
    coverage = build_logical_day_coverage(
        logical_days=logical_trading_days("2026-02-05", "2026-02-13"),
        inventory=synthetic_evidence["inventory"],
        tables=synthetic_evidence["tables"],
        funnel_days=synthetic_evidence["funnel_days"],
    )
    by_day = {row.logical_trading_day: row for row in coverage}
    monday = by_day["2026-02-09"]
    assert monday.source_partitions_present
    assert [r.physical_utc_date for r in monday.source_partition_refs_flat] == [
        "2026-02-08",
        "2026-02-09",
    ]
    assert monday.setup_lifecycle_rows == 40
    assert monday.executed_trade_rows == 3
    assert monday.exact_verifier_targets == 3  # resolved trades whose setup resolves
    assert monday.audit_day_covered
    assert not by_day["2026-02-13"].audit_day_covered
    assert monday.mbp1_scope_evidence == "not_evaluated"
    # the Sunday partition date is not a coverage row (not a logical day)
    assert "2026-02-08" not in by_day
    # a day whose partition is missing from the inventory is not hash-addressable
    partial = dict(synthetic_evidence["inventory"])
    partial.pop("2026-02-08")
    coverage_partial = build_logical_day_coverage(
        logical_days=("2026-02-09",),
        inventory=partial,
        tables=synthetic_evidence["tables"],
        funnel_days=synthetic_evidence["funnel_days"],
    )
    assert not coverage_partial[0].source_partitions_present
    assert coverage_partial[0].source_partition_refs == ()


def test_windows_rank_lexicographically_and_carry_hard_constraints(synthetic_evidence) -> None:
    coverage = build_logical_day_coverage(
        logical_days=logical_trading_days("2026-01-02", "2026-02-20"),
        inventory=synthetic_evidence["inventory"],
        tables=synthetic_evidence["tables"],
        funnel_days=synthetic_evidence["funnel_days"],
    )
    ranked = rank_logical_windows(
        coverage, window_length=5, inventory=synthetic_evidence["inventory"]
    )
    assert ranked[0].days == ("2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10", "2026-02-11")
    top = ranked[0]
    assert top.eligible
    assert top.hard_constraints["consecutive_logical_days_1_to_5"]
    assert top.hard_constraints["all_source_partitions_present_and_hash_addressable"]
    assert top.hard_constraints["seed_chain_producible_through_prior_store_day"]
    assert top.hard_constraints["at_least_one_exact_verifier_target"]
    assert top.hard_constraints["june_11_and_sealed_excluded"]
    assert top.rank_key[0] == 5  # every day source-covered
    assert top.rank_key[1] == 4  # all four lifecycle path classes
    assert top.rank_key[3] == 9  # executed trades
    assert top.seed_chain_replay_days == store_day_chain("2026-01-01", "2026-02-04")
    assert top.seed_chain_replay_day_count == len(top.seed_chain_replay_days)
    # the window 02-09…02-13 has a day without audit coverage → ranks below
    later = next(w for w in ranked if w.days[0] == "2026-02-09")
    assert later.rank_key[2] == 4
    # a window without any trade is ineligible (no exact verifier target)
    quiet = next(w for w in ranked if w.days[0] == "2026-01-02")
    assert not quiet.eligible
    assert not quiet.hard_constraints["at_least_one_exact_verifier_target"]
    # descending lexicographic order holds along the ranking
    keys = [w.rank_key for w in ranked]
    assert keys == sorted(keys, reverse=True)


def test_shortlist_contains_the_required_entries_and_never_registers(
    synthetic_evidence, tmp_path
) -> None:
    coverage = build_logical_day_coverage(
        logical_days=logical_trading_days("2026-01-02", "2026-06-10"),
        inventory=_inventory(store_day_chain("2026-01-01", "2026-06-10")),
        tables=synthetic_evidence["tables"],
        funnel_days=frozenset(logical_trading_days("2026-01-02", "2026-06-10")),
    )
    shortlist = build_verification_window_shortlist(
        coverage,
        inventory=_inventory(store_day_chain("2026-01-01", "2026-06-10")),
        evidence_source_dataset_id="1" * 64,
        evidence_source_manifest_sha256="2" * 64,
        audit_artifact_id="3" * 64,
    )
    assert isinstance(shortlist, VerificationWindowShortlist)
    assert shortlist.owner_selection == "NOT PERFORMED"
    assert shortlist.register_program_allowlist_called is False
    assert shortlist.no_raw_source_reads is True
    labels = {entry.label: entry for entry in shortlist.entries}
    assert labels["june_proposal"].window.days == JUNE_PROPOSAL_WINDOW
    assert (
        labels["highest_lifecycle_candidate_decision_trade_coverage"].window.days[0] == "2026-02-05"
    )
    assert "highest_complete_audit_day_coverage" in labels
    # the R1 store-day candidate is provisional: it names a Sunday partition
    assert R1_STORE_DAY_CANDIDATE == (
        "2026-02-06",
        "2026-02-08",
        "2026-02-09",
        "2026-02-10",
        "2026-02-11",
    )
    assert shortlist.r1_store_day_candidate.status == "provisional_ineligible_as_stated"
    assert shortlist.r1_store_day_candidate.non_trading_day_ids == ("2026-02-08",)
    assert shortlist.r1_store_day_candidate.corrected_logical_window == (
        "2026-02-06",
        "2026-02-09",
        "2026-02-10",
        "2026-02-11",
        "2026-02-12",
    )
    assert len(shortlist.shortlist_id) == 64
    # every shortlisted window carries the exact trading-day refs (both partitions)
    for entry in shortlist.entries:
        assert len(entry.window.trading_day_refs) == len(entry.window.days)
        for ref in entry.window.trading_day_refs:
            assert len(ref.ordered_source_partition_refs) == 2
    document = json.loads(shortlist.model_dump_json())
    assert document["owner_selection"] == "NOT PERFORMED"
    markdown = render_shortlist_markdown(shortlist)
    assert "NOT PERFORMED" in markdown and "register_program_allowlist" in markdown
    # no allowlist marker was written anywhere
    assert not list(tmp_path.rglob("VERIFICATION_ALLOWLIST_MARKER.json"))
