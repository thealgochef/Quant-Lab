"""Population + funnel delta suites (DT §4.3/§4.4; TEST_MATRIX §3.2 rows 3–4)."""

from __future__ import annotations

import pandas as pd

from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable
from alpha_lab.agents.data_infra.ifvg.study.funnel_delta import build_funnel_delta
from alpha_lab.agents.data_infra.ifvg.study.population_delta import (
    DIVERGENCE_REASONS,
    build_native_population_delta,
)


def _lifecycle(rows: list[tuple[str, str, str, str | None]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "envelope_setup_id": [row[0] for row in rows],
            "envelope_trading_day": [row[1] for row in rows],
            "event_cursor": [row[2] for row in rows],
            "reason": [row[3] for row in rows],
        }
    )


def test_exact_set_relationships_and_first_divergence() -> None:
    baseline = {
        RecordTable.SETUP_LIFECYCLE: _lifecycle(
            [
                ("setup-A", "2026-01-13", "c-010", None),
                ("setup-B", "2026-01-14", "c-020", "expired_parent_search"),
            ]
        )
    }
    challenger = {
        RecordTable.SETUP_LIFECYCLE: _lifecycle(
            [
                ("setup-A", "2026-01-13", "c-010", None),
                ("setup-C", "2026-01-13", "c-015", None),
            ]
        )
    }
    report = build_native_population_delta("setup", baseline, challenger)
    assert report.match_basis == "native_id_exact"
    assert report.common_keys == ("setup-A",)
    assert report.added_keys == ("setup-C",)
    assert report.removed_keys == ("setup-B",)
    assert report.jaccard == 1 / 3
    divergence = report.first_divergence
    assert divergence is not None
    # earliest (day, cursor): setup-C added on 01-13 precedes setup-B removed on 01-14
    assert divergence.entity_key == "setup-C"
    assert divergence.side == "added"
    assert divergence.trading_day == "2026-01-13"
    assert divergence.challenger_cursor == "c-015"
    assert divergence.baseline_cursor is None
    assert divergence.divergence_reason in DIVERGENCE_REASONS


def test_divergence_reason_maps_only_registered_terminals() -> None:
    baseline = {
        RecordTable.SETUP_LIFECYCLE: _lifecycle(
            [("setup-B", "2026-01-13", "c-020", "expired_parent_search")]
        )
    }
    challenger = {RecordTable.SETUP_LIFECYCLE: _lifecycle([])}
    report = build_native_population_delta("setup", baseline, challenger)
    assert report.first_divergence.divergence_reason == "earlier_stale_parent_expiry"

    unmapped = {
        RecordTable.SETUP_LIFECYCLE: _lifecycle(
            [("setup-B", "2026-01-13", "c-020", "some_novel_reason")]
        )
    }
    report = build_native_population_delta("setup", unmapped, challenger)
    assert report.first_divergence.divergence_reason == "unattributed"  # never guessed


def test_identical_populations_are_jaccard_one_with_no_divergence() -> None:
    tables = {
        RecordTable.SETUP_LIFECYCLE: _lifecycle([("setup-A", "2026-01-13", "c-1", None)])
    }
    report = build_native_population_delta("setup", tables, tables)
    assert report.jaccard == 1.0
    assert report.first_divergence is None


def test_funnel_delta_reports_union_vocabulary_and_conversions() -> None:
    baseline = {
        "2026-01-13": {"htf_taps": 10, "setups_born": 4, "parents_locked": 2},
        "2026-01-14": {"htf_taps": 6, "setups_born": 2, "baseline_only_counter": 1},
    }
    challenger = {
        "2026-01-13": {"htf_taps": 12, "setups_born": 3, "parents_locked": 3},
        "2026-01-14": {"htf_taps": 5, "setups_born": 2, "challenger_only_counter": 7},
    }
    report = build_funnel_delta(
        baseline,
        challenger,
        baseline_terminal_reasons={"expired_parent_search": 2},
        challenger_terminal_reasons={"expired_parent_search": 1, "session_gate_reset": 1},
    )
    totals = {row.counter: row for row in report.totals}
    assert totals["htf_taps"].baseline == 16
    assert totals["htf_taps"].challenger == 17
    assert totals["htf_taps"].delta == 1
    # one-sided counters are reported zero-filled and flagged, never dropped
    assert totals["baseline_only_counter"].missing_from == "challenger"
    assert totals["challenger_only_counter"].missing_from == "baseline"
    assert set(report.vocabulary_mismatches) == {
        "baseline_only_counter",
        "challenger_only_counter",
    }
    # per-day breakdown covers the union of days
    assert set(report.per_day) == {"2026-01-13", "2026-01-14"}
    # conditional conversions computed per side when both counters exist
    tap_to_setup = report.conversions["tap_to_setup"]
    assert tap_to_setup[0] == 6 / 16
    assert tap_to_setup[1] == 5 / 17
    setup_to_lock = report.conversions["setup_to_lock"]
    assert setup_to_lock[0] == 2 / 6
    assert setup_to_lock[1] == 3 / 5
    # absent-denominator conversions are typed None, not fabricated
    assert report.conversions["lock_to_armed"] == (None, None)
    terminal = {row.counter: row for row in report.terminal_reason_deltas}
    assert terminal["expired_parent_search"].delta == -1
    assert terminal["session_gate_reset"].missing_from == "baseline"
