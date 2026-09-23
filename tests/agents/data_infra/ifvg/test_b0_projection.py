"""Exact real-record B0 regression, clock, replacement and leakage guards."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.artifact_io import ArtifactVerificationError
from alpha_lab.agents.data_infra.ifvg.audit_contracts import AuditTable
from alpha_lab.agents.data_infra.ifvg.b0_projection import (
    B0_PROJECTION_VERSION,
    STRUCTURAL_ENTRY_FEATURES,
    project_b0_candidates,
    validate_b0_feature_mapping,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (
    M0_FEATURES,
    build_candidate_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.context_run_store import (
    load_candidate_feature_view_frame,
    save_candidate_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.contracts import RecordTable

from .b0_fixture import real_b0_fixture, real_b0_view


def _project(candidates, source, bars):
    return project_b0_candidates(
        candidates, source, decision_bars=bars, advertised_features=M0_FEATURES,
    )


def test_five_real_candidates_match_original_core_stage_values_and_geometry():
    candidates, source, bars, payload = real_b0_fixture()
    before = candidates.copy(deep=True)
    projected, evidence = _project(candidates, source, bars)
    pd.testing.assert_frame_equal(candidates, before)
    assert evidence["projection_version"] == B0_PROJECTION_VERSION
    for row in projected.to_dict("records"):
        for name, expected in payload["expected_values"][row["candidate_id"]].items():
            assert row[name] == expected
    assert all(row["all_formula_clock_ordinal_checks_passed"]
               for row in evidence["candidate_evidence"])
    for field in M0_FEATURES:
        assert evidence["feature_mapping"][field]["unmapped_count"] == 0
    assert set(projected.entry_family) == {"ifvg_retest", "fresh_fvg_continuation"}
    for name in STRUCTURAL_ENTRY_FEATURES:
        assert projected.loc[projected.entry_family.eq("ifvg_retest"), name].isna().all()
        continuation = projected.entry_family.eq("fresh_fvg_continuation")
        assert projected.loc[continuation, name].notna().all()


def test_real_parent_clocks_count_original_completed_parent_bars():
    candidates, source, bars, payload = real_b0_fixture()
    projected, _ = _project(candidates, source, bars)
    parent_bars = pd.DataFrame(payload["parent_completed_bars"])
    # Core counts every delivered closed parent bar; its terminal partial bar
    # also appears in tf_bar_close_counts. Filtering is_complete changes the clock.
    assert len(parent_bars.loc[~parent_bars.is_complete]) == 1
    counts = parent_bars.groupby("candidate_id").size()
    for row in projected.itertuples():
        assert row.elapsed_parent_bars_since_tap == counts[row.candidate_id]
    assert set(projected.parent_tf_seconds) >= {300, 900, 1800}


def test_real_sparse_sequence_is_not_wall_clock_minutes():
    view = real_b0_view()
    expected = {
        "93f9128c-cdcb-5bca-9a9e-5febc557204c": (233, 293),
        "5e898e6d-e7ac-59f6-9397-b36bde9fc2e9": (103, 104),
    }
    for candidate_id, (steps, minutes) in expected.items():
        row = view.frame.set_index("candidate_id").loc[candidate_id]
        evidence = next(item for item in view.b0_projection_evidence["candidate_evidence"]
                        if item["candidate_id"] == candidate_id)
        selection = pd.Timestamp(evidence["stage_available_ts_utc"][AuditTable.PARENT_CANDIDATE])
        wall_minutes = (selection - row.geometry_tap_bar_logical_close_ts_utc).total_seconds() / 60
        assert wall_minutes == minutes
        assert row.elapsed_1m_bars_since_tap == steps


def test_real_parent_and_opposing_replacements_use_final_selected_events():
    candidates, source, bars, _ = real_b0_fixture()
    _, evidence = _project(candidates, source, bars)
    replacements = {AuditTable.PARENT_CANDIDATE: 0, AuditTable.OPPOSING: 0}
    for item in evidence["candidate_evidence"]:
        for table, next_stage in ((AuditTable.PARENT_CANDIDATE, AuditTable.PARENT_LOCK),
                                  (AuditTable.OPPOSING, AuditTable.INVERSION)):
            rows = source.tables[table]
            chosen = rows.loc[
                rows.envelope_setup_id.eq(item["setup_id"]) & rows.selected.eq(True)
                & rows.trace_ordinal.lt(item["stage_trace_ordinals"][next_stage])
            ]
            assert chosen.trace_ordinal.max() == item["stage_trace_ordinals"][table]
            replacements[table] += len(chosen) > 1
    assert all(replacements.values())


@pytest.mark.parametrize("field", ["elapsed_1m_bars_since_tap", "elapsed_parent_bars_since_tap"])
def test_changed_stage_ordinal_or_clock_fails_parity(field):
    candidates, source, bars, _ = real_b0_fixture()
    parents = source.tables[AuditTable.PARENT_CANDIDATE].copy()
    parents[field] += 1
    source = replace(source, tables={**source.tables, AuditTable.PARENT_CANDIDATE: parents})
    with pytest.raises(ArtifactVerificationError, match=field):
        _project(candidates, source, bars)


def test_missing_stage_mapping_cannot_be_treated_as_structural_null():
    candidates, source, bars, _ = real_b0_fixture()
    parents = source.tables[AuditTable.PARENT_CANDIDATE].drop(columns="distance_to_htf_ticks")
    source = replace(source, tables={**source.tables, AuditTable.PARENT_CANDIDATE: parents})
    with pytest.raises(ArtifactVerificationError, match="unmapped advertised.*distance_to_htf"):
        _project(candidates, source, bars)


def test_stage_emission_after_entry_is_not_decision_time_evidence():
    candidates, source, bars, _ = real_b0_fixture()
    inversions = source.tables[AuditTable.INVERSION].copy()
    inversions["envelope_ts_utc"] += pd.Timedelta(days=400)
    source = replace(source, tables={**source.tables, AuditTable.INVERSION: inversions})
    with pytest.raises(ArtifactVerificationError, match="unavailable at decision time"):
        _project(candidates, source, bars)


def test_matching_but_future_dated_geometry_and_stage_fvg_are_rejected():
    candidates, source, bars, _ = real_b0_fixture()
    candidates["geometry_parent_confirmed_ts_utc"] += pd.Timedelta(days=400)
    parents = source.tables[AuditTable.PARENT_CANDIDATE].copy()
    parents["fvg_confirmed_ts_utc"] += pd.Timedelta(days=400)
    source = replace(source, tables={**source.tables, AuditTable.PARENT_CANDIDATE: parents})
    with pytest.raises(ArtifactVerificationError, match="unavailable at stage decision time"):
        _project(candidates, source, bars)


def test_underlying_inversion_ohlc_drift_is_rejected():
    candidates, source, bars, _ = real_b0_fixture()
    bars.loc[bars.bar_id.isin(candidates.geometry_inversion_bar_bar_id), "close_ticks"] += 1
    with pytest.raises(ArtifactVerificationError, match="inversion decision bar close"):
        _project(candidates, source, bars)


def test_fold_empty_structural_feature_is_mapped_but_unmapped_required_feature_fails():
    frame = real_b0_view().frame
    retests = frame.loc[frame.entry_family.eq("ifvg_retest")].copy()
    coverage = validate_b0_feature_mapping(retests, M0_FEATURES)
    assert coverage["entry_fvg_size_ticks"]["non_null_count"] == 0
    assert coverage["entry_fvg_size_ticks"]["structural_null_count"] == len(retests)
    retests["parent_tf_seconds"] = None
    with pytest.raises(ArtifactVerificationError, match="unmapped advertised B0 feature parent_tf"):
        validate_b0_feature_mapping(retests, M0_FEATURES)


def test_missing_advertised_column_and_fabricated_retest_geometry_fail():
    frame = real_b0_view().frame
    with pytest.raises(ArtifactVerificationError, match="unmapped advertised B0 field: risk_ticks"):
        validate_b0_feature_mapping(frame.drop(columns="risk_ticks"), M0_FEATURES)
    frame.loc[frame.entry_family.eq("ifvg_retest"), "entry_fvg_size_ticks"] = 0
    with pytest.raises(ArtifactVerificationError, match="fabricates an entry FVG"):
        validate_b0_feature_mapping(frame, M0_FEATURES)


def test_fresh_view_requires_exact_selected_stage_source_and_accepted_identity():
    candidates, source, bars, _ = real_b0_fixture()
    pair = SimpleNamespace(
        v2=SimpleNamespace(tables={RecordTable.ENTRY_CANDIDATE: candidates},
                           reference=SimpleNamespace(artifact_id="different-dataset",
                                                     manifest_payload_sha256="different-manifest")),
        v3=SimpleNamespace(tables={}),
    )
    with pytest.raises(ArtifactVerificationError, match="exact selected Core stage audit source"):
        build_candidate_feature_view(pair)
    with pytest.raises(ArtifactVerificationError, match="not bound to the accepted v2"):
        build_candidate_feature_view(pair, b0_source=source, decision_bars=bars)


def test_repaired_view_persists_projection_evidence_in_content_identity(tmp_path):
    view = real_b0_view()
    view.frame["context_capture_id"] = "fixture-capture-" + view.frame["candidate_id"]
    view = replace(view, view_id=canonical_contract_sha256({
        "artifact_pair_hash": view.artifact_pair_hash,
        "feature_registry_hash": view.feature_registry_hash,
        "candidate_ids": view.frame.candidate_id.tolist(),
        "candidate_link_ids": view.frame.context_capture_id.tolist(),
        "b0_projection_evidence_hash": view.b0_projection_evidence["projection_evidence_hash"],
    }))
    save_candidate_feature_view(view, base_dir=tmp_path)
    loaded, manifest = load_candidate_feature_view_frame(view.view_id, base_dir=tmp_path)
    assert manifest["b0_projection_evidence"] == view.b0_projection_evidence
    pd.testing.assert_frame_equal(loaded[list(M0_FEATURES)], view.frame[list(M0_FEATURES)])
