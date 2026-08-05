"""Generation-3 context schemas remain separate and exact-ID joined."""

from __future__ import annotations

import json

import pytest
from strategy_core.strategies.ifvg_smc.context_config import FEATURE_FORMULA_VERSION

from alpha_lab.agents.data_infra.ifvg.artifact_io import (
    ArtifactVerificationError,
    validate_exact_context_links,
)
from alpha_lab.agents.data_infra.ifvg.context_contracts import (
    ContextRecordTable,
    validate_context_foreign_keys,
    validate_context_primary_keys,
    validate_context_table_identity,
)
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

from .ifvg_v3_fixtures import context_fixture


def test_v3_tables_do_not_widen_v2_enum_or_generic_capture_lane() -> None:
    assert [table.value for table in RecordTable] == [
        "setup_lifecycle_event",
        "entry_candidate",
        "candidate_label",
        "eligible_decision",
        "executed_trade",
        "geometry_dossier",
        "quarantine",
    ]
    assert len(ContextRecordTable) == 12
    v2_names = {table.value for table in RecordTable}
    context_names = {table.value for table in ContextRecordTable}
    assert not (v2_names & context_names)


def test_all_normalized_tables_validate_and_have_formula_identity() -> None:
    _day, tables, core, _emissions = context_fixture()
    assert set(tables) == set(ContextRecordTable)
    for table, frame in tables.items():
        validate_context_primary_keys(table, frame)
        validate_context_table_identity(table, frame)
    validate_context_foreign_keys(tables, core_tables=core)
    captures = tables[ContextRecordTable.CONTEXT_CAPTURE]
    assert set(captures["feature_set_version"]) == {"ifvg_context_v1"}
    assert set(captures["feature_formula_version"]) == {FEATURE_FORMULA_VERSION}


def test_candidate_decision_and_trade_links_are_one_to_one_and_frozen() -> None:
    _day, tables, core, _emissions = context_fixture()
    candidates = tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]
    decisions = tables[ContextRecordTable.DECISION_CONTEXT_LINK]
    trades = tables[ContextRecordTable.TRADE_CONTEXT_LINK]
    assert len(candidates) == 2  # eligible plus independently captured blocked family
    assert len(decisions) == len(trades) == 1
    decision = decisions.iloc[0]
    trade = trades.iloc[0]
    assert trade["candidate_id"] == decision["candidate_id"]
    assert trade["decision_id"] == decision["decision_id"]
    assert trade["decision_context_capture_id"] == decision["context_capture_id"]
    assert trade["frozen_from_capture_id"] == decision["context_capture_id"]
    validate_context_foreign_keys(tables, core_tables=core)


def test_candidate_link_uses_exact_v2_trigger_geometry_not_capture_record_id() -> None:
    _day, tables, core, _emissions = context_fixture()
    candidates = tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]
    captures = tables[ContextRecordTable.CONTEXT_CAPTURE].set_index(
        "context_capture_id", verify_integrity=True
    )
    for link in candidates.to_dict("records"):
        capture = captures.loc[link["context_capture_id"]]
        assert link["geometry_evidence_id"] != capture["evidence_id"]
        assert link["geometry_evidence_cursor"] == capture["evidence_cursor"]
    validate_exact_context_links(tables, core_tables=core)

    tampered = dict(tables)
    tampered_links = candidates.copy()
    tampered_links.loc[0, "geometry_evidence_id"] = "nearest-time-fallback"
    tampered[ContextRecordTable.CANDIDATE_CONTEXT_LINK] = tampered_links
    with pytest.raises(ArtifactVerificationError, match="trigger_evidence_id"):
        validate_exact_context_links(tampered, core_tables=core)


def test_setup_only_nearest_and_keep_last_fallbacks_fail_validation() -> None:
    _day, tables, core, _emissions = context_fixture()
    crossed = dict(tables)
    decisions = tables[ContextRecordTable.DECISION_CONTEXT_LINK].copy()
    wrong_candidate = next(
        item
        for item in tables[ContextRecordTable.CANDIDATE_CONTEXT_LINK]["candidate_id"]
        if item != decisions.loc[0, "candidate_id"]
    )
    decisions.loc[0, "candidate_id"] = wrong_candidate
    crossed[ContextRecordTable.DECISION_CONTEXT_LINK] = decisions
    with pytest.raises(ValueError, match="non-exact|does not match"):
        validate_context_foreign_keys(crossed, core_tables=core)

    missing_exact = dict(tables)
    links = decisions.drop(columns=["candidate_context_capture_id"])
    missing_exact[ContextRecordTable.DECISION_CONTEXT_LINK] = links
    with pytest.raises(ValueError, match="candidate_context_capture_id"):
        validate_context_foreign_keys(missing_exact)


def test_context_tables_contain_no_outcome_or_post_entry_feature_sources() -> None:
    _day, tables, _core, _emissions = context_fixture()
    forbidden = ("mfe", "mae", "realized", "future_return", "outcome", "resolution")
    for table, frame in tables.items():
        payload = json.dumps(
            {"columns": list(frame.columns), "rows": frame.to_dict("records")},
            default=str,
        ).lower()
        assert not any(token in payload for token in forbidden), table.value
