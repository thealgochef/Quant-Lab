"""Real B0 record extracts exercise the fail-closed S06 pipeline boundary."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.b0_projection import (
    B0_PROJECTION_VERSION,
    STAGE_FEATURES,
    STRUCTURAL_ENTRY_FEATURES,
)
from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import canonical_contract_sha256
from alpha_lab.agents.data_infra.ifvg.context_feature_view import M0_FEATURES
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    _stage_s06_coverage,
    _validated_research_b0_projection,
)
from tests.agents.data_infra.ifvg.b0_fixture import real_b0_view


@pytest.fixture
def context():
    view = real_b0_view()
    return SimpleNamespace(
        view=view,
        wiring=SimpleNamespace(
            research_subject=SimpleNamespace(
                core_replay_id=view.b0_projection_evidence["source"]["core_replay_id"],
            )
        ),
        bundle_views={
            "B0_CORE": SimpleNamespace(
                payload=SimpleNamespace(
                    resolved_feature_names=M0_FEATURES,
                )
            )
        },
        bundle_frames={"B0_CORE": view.frame[["candidate_id", *M0_FEATURES]].copy()},
        semantic=SimpleNamespace(payload=SimpleNamespace(regime_study=None)),
        stage_sidecars={},
    )


def _mutate_proof(context, mutate, *, rehash=True):
    evidence = deepcopy(context.view.b0_projection_evidence)
    mutate(evidence)
    if rehash:
        evidence.pop("projection_evidence_hash")
        evidence["projection_evidence_hash"] = canonical_contract_sha256(evidence)
    context.view = replace(context.view, b0_projection_evidence=evidence)


def test_s06_accepts_real_structural_nulls_and_persists_mapping_evidence(context):
    outputs, note = _stage_s06_coverage(context)
    assert outputs == () and "feature coverage" in note
    report = json.loads(context.stage_sidecars["feature_coverage.json"])["B0_CORE"]
    saved = json.loads(context.stage_sidecars["b0_projection_evidence.json"])
    assert report["b0_projection_version"] == B0_PROJECTION_VERSION
    assert report["b0_projection_evidence_hash"] == saved["evidence"]["projection_evidence_hash"]
    assert saved["scoped_candidate_count"] == len(context.view.frame) == 5
    retests = int(context.view.frame["entry_family"].eq("ifvg_retest").sum())
    assert 0 < retests < len(context.view.frame)
    for name in STRUCTURAL_ENTRY_FEATURES:
        mapping = report["b0_feature_mapping"][name]
        assert mapping["mapping_status"] == "mapped"
        assert mapping["structural_null_count"] == retests
        assert mapping["unmapped_count"] == 0
        assert report["per_feature_nonnull_fraction"][name] == (5 - retests) / 5
    for names in STAGE_FEATURES.values():
        for name in names:
            assert report["per_feature_nonnull_fraction"][name] == 1.0


@pytest.mark.parametrize("historical_partial", [False, True])
def test_s06_rejects_missing_evidence_including_historical_partial_views(
    context, historical_partial
):
    frame = context.view.frame.copy()
    if historical_partial:
        for names in STAGE_FEATURES.values():
            frame[list(names)] = np.nan
    context.view = replace(context.view, frame=frame, b0_projection_evidence=None)
    with pytest.raises(ValueError, match="require verified projection evidence"):
        _stage_s06_coverage(context)
    assert context.stage_sidecars == {}


def test_s06_rejects_altered_repaired_feature_before_publishing_coverage(context):
    frame = context.view.frame.copy()
    frame.loc[frame.index[0], "elapsed_1m_bars_since_tap"] += 1
    context.view = replace(context.view, frame=frame)
    with pytest.raises(ValueError, match="differs from source evidence"):
        _stage_s06_coverage(context)
    assert context.stage_sidecars == {}


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("hash", "evidence hash mismatch"),
        ("omitted_field", "omits advertised source fields"),
        ("unverified", "lacks verified geometry/event/clock evidence"),
        ("missing_candidate", "lacks verified geometry/event/clock evidence"),
        ("wrong_source", "source differs from the research subject"),
        ("decision_time", "decision timestamp differs from candidate"),
    ],
)
def test_s06_rejects_invalid_evidence_even_when_internally_rehashed(context, mutation, reason):
    def change(evidence):
        first = evidence["candidate_evidence"][0]
        if mutation == "hash":
            evidence["projection_evidence_hash"] = "0" * 64
        elif mutation == "omitted_field":
            first["projected_values"].pop("elapsed_1m_bars_since_lock")
        elif mutation == "unverified":
            first["all_formula_clock_ordinal_checks_passed"] = False
        elif mutation == "missing_candidate":
            evidence["candidate_evidence"].pop(0)
        elif mutation == "wrong_source":
            evidence["source"]["core_replay_id"] = "f" * 64
        else:
            first["entry_ts_utc"] = "2000-01-01T00:00:00+00:00"

    _mutate_proof(context, change, rehash=mutation != "hash")
    with pytest.raises(ValueError, match=reason):
        _validated_research_b0_projection(context)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("value", "B0 persisted bundle projection"),
        ("missing_column", "bundle omits advertised B0 features"),
        ("omitted_advertised_name", "does not advertise the complete B0 schema"),
    ],
)
def test_s06_rejects_bundle_value_and_schema_drift(context, mutation, reason):
    if mutation == "value":
        frame = context.bundle_frames["B0_CORE"]
        frame.loc[frame.index[0], "parent_tf_seconds"] += 60
    elif mutation == "missing_column":
        context.bundle_frames["B0_CORE"] = context.bundle_frames["B0_CORE"].drop(
            columns=["parent_tf_seconds"]
        )
    else:
        context.bundle_views["B0_CORE"].payload.resolved_feature_names = tuple(
            name for name in M0_FEATURES if name != "parent_tf_seconds"
        )
    with pytest.raises((ValueError, AssertionError), match=reason):
        _stage_s06_coverage(context)
    assert context.stage_sidecars == {}


@pytest.mark.parametrize(
    "family,reason",
    [
        ("fresh_fvg_continuation", "non-structural nulls"),
        ("ifvg_retest", "fabricates an entry FVG"),
    ],
)
def test_s06_distinguishes_missing_continuation_geometry_from_legitimate_retest_null(
    context, family, reason
):
    frame = context.view.frame.copy()
    mask = frame["entry_family"].eq(family)
    frame.loc[mask, "entry_fvg_size_ticks"] = np.nan if family == "fresh_fvg_continuation" else 4.0
    context.view = replace(context.view, frame=frame)
    with pytest.raises(ValueError, match=reason):
        _validated_research_b0_projection(context)
