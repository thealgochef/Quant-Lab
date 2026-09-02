"""R6.1-FIX workstream A — verified assignment evidence (plan §3.1–§3.4;
findings F-01 / F-02 / F-03 / F-04 / F-05).

The persisted per-fit assignment sidecar has an ENFORCED Arrow schema and
row invariants; the descriptive OOS-assignment artifact and the fold-local
feature artifact are built from VERIFIED stored bytes only and bind every
fit's sidecar hash + schema hash into their identities; a fit that already
exists is reused only when the candidate assignment bytes equal the stored
bytes byte-for-byte; a candidate without its stage anchor is preserved with
the typed reason ``candidate_as_of_missing``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
    PANEL_ASSIGNMENT_MISSING_REASONS,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_blocks import AvailabilityStage
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.fold_schedules import derive_fold_schedule
from alpha_lab.agents.data_infra.ifvg.ml import regime_executor as executor_module
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import (
    FIT_ASSIGNMENT_SCHEMA,
    FIT_ASSIGNMENT_SCHEMA_HASH,
    REGIME_ASSIGNMENT_MISSING_REASONS,
    FitAssignmentRef,
    ObservationGranularity,
    RegimeAssignmentColumns,
    fit_assignment_table_bytes,
    validate_assignment_rows,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_executor import execute_regime_protocol
from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
    OOS_ASSIGNMENT_SCHEMA_HASH,
    RegimeOosAssignmentPayload,
    assign_panel_regimes_to_candidates,
    build_regime_oos_assignment_artifact,
    candidate_as_of_frame,
    candidate_as_of_source_hash,
    consulted_assignments_hash,
    load_regime_oos_assignment,
    load_regime_oos_assignment_frame,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_service import (
    resolve_kmeans_protocol,
    run_regime_protocol,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_store import (
    ASSIGNMENTS_SIDECAR,
    REGIME_FIT_STORE,
    VerifiedFitAssignments,
    load_regime_fit_assignments,
    persist_regime_fit,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    envelope_destination,
    load_sidecar_bytes,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    REGIME_INPUT_FEATURES,
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
    persisted_candidate_source,
    persisted_panel_source,
)

_B0 = resolve_bundle("B0_CORE").resolved_feature_bundle_id


@pytest.fixture(scope="module")
def candidate_run():
    fixture = known_cluster_fixture(k=3, n=600)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    protocol = resolve_kmeans_protocol(
        input_feature_bundle_ref=_B0, resolved_input_features=REGIME_INPUT_FEATURES
    )
    run = run_regime_protocol(
        fixture.view.frame,
        folds,
        protocol,
        source_artifact_ids=(fixture.view.view_id,),
        bootstrap_refits=2,
    )
    return fixture, folds, protocol, run


def _fold_assignments(run, fold_fit) -> pd.DataFrame:
    return run.assignments[run.assignments["fold_index"] == fold_fit.fold_index]


def _persist_all(root: Path, candidate_run) -> dict[int, VerifiedFitAssignments]:
    fixture, _folds, _protocol, run = candidate_run
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            _fold_assignments(run, fold_fit),
            observation_frame=fixture.view.frame,
        )
    return {
        fit.fold_index: load_regime_fit_assignments(root, fit.fit_envelope.regime_fit_id)
        for fit in run.fold_fits
    }


# ── §3.4 the enforced fit-assignment schema and row invariants (F-05) ────────


def test_fit_assignment_sidecar_schema_is_enforced_on_save_and_load(candidate_run, tmp_path):
    fixture, _folds, _protocol, run = candidate_run
    assert tuple(FIT_ASSIGNMENT_SCHEMA.names) == tuple(RegimeAssignmentColumns)
    assert hashlib.sha256(
        "\n".join(f"{f.name}:{f.type}" for f in FIT_ASSIGNMENT_SCHEMA).encode("utf-8")
    ).hexdigest() == FIT_ASSIGNMENT_SCHEMA_HASH
    root = tmp_path / "store"
    fold_fit = run.fold_fits[0]
    frame = _fold_assignments(run, fold_fit)
    persist_regime_fit(root, fold_fit, frame, observation_frame=fixture.view.frame)
    fit_id = fold_fit.fit_envelope.regime_fit_id
    verified = load_regime_fit_assignments(root, fit_id)
    assert isinstance(verified, VerifiedFitAssignments)
    assert verified.regime_fit_id == fit_id == verified.envelope.regime_fit_id
    assert verified.artifact.regime_fit_id == fit_id
    stored = load_sidecar_bytes(root, REGIME_FIT_STORE, fit_id, ASSIGNMENTS_SIDECAR)
    # the stored bytes ARE the enforced-schema serialization of the frame
    assert stored == fit_assignment_table_bytes(frame)
    assert verified.assignments_sidecar_sha256 == hashlib.sha256(stored).hexdigest()
    assert verified.assignment_schema_hash == FIT_ASSIGNMENT_SCHEMA_HASH
    assert list(verified.frame.columns) == list(RegimeAssignmentColumns)
    assert len(verified.frame) == len(frame)
    # the verified frame satisfies the row invariants of the fit sidecar
    validate_assignment_rows(verified.frame, cluster_count=3, kind="fit")
    # a frame lacking a VALUE column is refused at persist (no inferred schema)
    with pytest.raises(ValueError, match="distances"):
        persist_regime_fit(
            tmp_path / "other",
            fold_fit,
            frame.drop(columns=["distances"]),
            observation_frame=fixture.view.frame,
        )
    assert not (tmp_path / "other").exists()
    # a tampered sidecar fails closed at the verified load
    sidecar = envelope_destination(root, REGIME_FIT_STORE, fit_id) / ASSIGNMENTS_SIDECAR
    original = sidecar.read_bytes()
    try:
        sidecar.write_bytes(original[:-4] + b"\x00" * 4)
        with pytest.raises(SearchStoreError):
            load_regime_fit_assignments(root, fit_id)
    finally:
        sidecar.write_bytes(original)


def _valid_row(**overrides) -> dict:
    row = {
        "resolved_regime_protocol_id": "a" * 64,
        "regime_fit_id": "b" * 64,
        "row_id": "cand-1",
        "fold_index": 0,
        "partition": "test",
        "observation_ts_utc": "2026-01-05T14:00:00+00:00",
        "fold_local_cluster_id": 1,
        "canonical_reporting_cluster_id": 2,
        "distances": [0.5, 0.1, 0.9],
        "assigned_distance": 0.1,
        "assignment_margin": 0.4,
        "assignment_entropy": np.nan,
        "log_density": np.nan,
        "outlier_score": np.nan,
        "valid": True,
        "missing_reason": None,
    }
    row.update(overrides)
    return row


def _invalid_row(**overrides) -> dict:
    row = _valid_row(
        fold_local_cluster_id=None,
        canonical_reporting_cluster_id=None,
        distances=None,
        assigned_distance=np.nan,
        assignment_margin=np.nan,
        valid=False,
        missing_reason="source_feature_missing",
    )
    row.update(overrides)
    return row


def _frame(*rows: dict) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(RegimeAssignmentColumns))


def test_valid_descriptive_rows_carry_every_value_column():
    """A ``valid=true`` row must carry the complete value set; an invalid row
    carries only linkage + one registered reason (plan §3.4)."""

    validate_assignment_rows(_frame(_valid_row(), _invalid_row(row_id="cand-2")), cluster_count=3)
    refusals = {
        "distances": _valid_row(distances=None),
        "assigned_distance": _valid_row(assigned_distance=np.nan),
        "assignment_margin": _valid_row(assignment_margin=np.nan),
        "canonical_reporting_cluster_id": _valid_row(canonical_reporting_cluster_id=None),
        "length": _valid_row(distances=[0.5, 0.1]),
        "finite": _valid_row(distances=[0.5, np.inf, 0.9]),
        "local": _valid_row(fold_local_cluster_id=3),
        "partition": _valid_row(partition="train"),
        "fit id": _valid_row(regime_fit_id=""),
        "missing_reason": _valid_row(missing_reason="coverage_gap"),
        "valid": _valid_row(valid=None),
    }
    for label, row in refusals.items():
        with pytest.raises(ValueError, match="valid|descriptive|partition|fit|reason|distance"):
            validate_assignment_rows(_frame(row), cluster_count=3, kind="descriptive")
        assert label
    # an invalid row may not carry assignment outputs, and needs a registered reason
    with pytest.raises(ValueError, match="invalid"):
        validate_assignment_rows(_frame(_invalid_row(assignment_margin=0.2)), cluster_count=3)
    with pytest.raises(ValueError, match="invalid"):
        validate_assignment_rows(_frame(_invalid_row(fold_local_cluster_id=0)), cluster_count=3)
    with pytest.raises(ValueError, match="registered"):
        validate_assignment_rows(
            _frame(_invalid_row(missing_reason="made_up_reason")), cluster_count=3
        )
    with pytest.raises(ValueError, match="registered"):
        validate_assignment_rows(_frame(_invalid_row(missing_reason=None)), cluster_count=3)
    assert "source_feature_missing" in REGIME_ASSIGNMENT_MISSING_REASONS


def test_model_facing_rows_use_the_separate_schema():
    """The model-facing kind admits train rows and a null canonical id (it is
    reporting-only there) but still requires the complete numeric outputs."""

    validate_assignment_rows(
        _frame(_valid_row(partition="train", canonical_reporting_cluster_id=None)),
        cluster_count=3,
        kind="model_facing",
    )
    with pytest.raises(ValueError, match="distance"):
        validate_assignment_rows(
            _frame(_valid_row(partition="train", distances=None)),
            cluster_count=3,
            kind="model_facing",
        )
    with pytest.raises(ValueError, match="partition"):
        validate_assignment_rows(
            _frame(_valid_row(partition="holdout")), cluster_count=3, kind="model_facing"
        )
    # the fit kind: train + test partitions, canonical id required for valid rows
    validate_assignment_rows(_frame(_valid_row(partition="train")), cluster_count=3, kind="fit")
    with pytest.raises(ValueError, match="canonical"):
        validate_assignment_rows(
            _frame(_valid_row(canonical_reporting_cluster_id=None)), cluster_count=3, kind="fit"
        )
    with pytest.raises(ValueError, match="kind"):
        validate_assignment_rows(_frame(_valid_row()), cluster_count=3, kind="other")


# ── §3.1 byte-for-byte reuse (F-02) ──────────────────────────────────────────


def test_reuse_refuses_a_fit_whose_stored_assignments_differ_in_values(candidate_run, tmp_path):
    fixture, _folds, _protocol, run = candidate_run
    root = tmp_path / "store"
    fold_fit = run.fold_fits[0]
    frame = _fold_assignments(run, fold_fit)
    persist_regime_fit(root, fold_fit, frame, observation_frame=fixture.view.frame)
    # identical bytes → verified REUSE
    _artifact, reused = persist_regime_fit(
        root, fold_fit, frame, observation_frame=fixture.view.frame, return_reuse=True
    )
    assert reused is True
    # same row set, same labels, DIFFERENT value bytes (a diagnostic column
    # the row invariants do not constrain) → the byte-for-byte reuse refuses
    perturbed = frame.copy()
    valid_index = perturbed.index[perturbed["valid"].astype(bool)][0]
    perturbed.loc[valid_index, "outlier_score"] = 0.123456
    with pytest.raises(ValueError, match="assignment bytes|byte"):
        persist_regime_fit(root, fold_fit, perturbed, observation_frame=fixture.view.frame)
    # an arithmetically inconsistent value (a margin that is no longer the
    # runner-up minus the minimum) is refused even earlier, by the row
    # invariants (adversarial RA-06) — never reaching the store
    inconsistent = frame.copy()
    inconsistent.loc[valid_index, "assignment_margin"] = (
        float(inconsistent.loc[valid_index, "assignment_margin"]) + 1.0
    )
    with pytest.raises(ValueError, match="margin"):
        persist_regime_fit(root, fold_fit, inconsistent, observation_frame=fixture.view.frame)
    # the stored bytes were never rewritten
    stored = load_sidecar_bytes(
        root, REGIME_FIT_STORE, fold_fit.fit_envelope.regime_fit_id, ASSIGNMENTS_SIDECAR
    )
    assert stored == fit_assignment_table_bytes(frame)


# ── §3.1 descriptive identity binds every fit sidecar (F-01) ─────────────────


def _descriptive_frame(run, fixture, verified):
    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        candidate_fold_oos_assignment,
        consulted_assignment_frame,
    )

    ids = tuple(fixture.view.frame["candidate_id"].astype(str))
    return candidate_fold_oos_assignment(consulted_assignment_frame(verified), ids)


def test_oos_assignment_identity_binds_every_fit_sidecar_hash(candidate_run, tmp_path):
    fixture, _folds, protocol, run = candidate_run
    root = tmp_path / "store"
    verified = _persist_all(root, candidate_run)
    frame = _descriptive_frame(run, fixture, verified)
    schedule = derive_fold_schedule(fixture.trading_days)
    as_of = candidate_as_of_frame(fixture.view.frame, stage=AvailabilityStage.ENTRY_DECISION)
    envelope, table = build_regime_oos_assignment_artifact(
        frame,
        protocol=protocol,
        verified_fit_assignments=verified,
        regime_fold_set_id=run.fold_set_id,
        fold_schedule_id=schedule.fold_schedule_id,
        candidate_as_of=as_of,
        candidate_as_of_source_ref=f"bundle_feature_view:{fixture.view.view_id}",
        candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
    )
    payload = envelope.payload
    refs = payload.regime_fit_assignment_refs
    assert refs == tuple(sorted(refs, key=lambda ref: ref.regime_fit_id))
    assert payload.regime_fit_ids == tuple(ref.regime_fit_id for ref in refs)
    by_id = {v.regime_fit_id: v for v in verified.values()}
    for ref in refs:
        assert ref.assignments_sidecar_sha256 == by_id[ref.regime_fit_id].assignments_sidecar_sha256
        assert ref.assignment_schema_hash == FIT_ASSIGNMENT_SCHEMA_HASH
    assert payload.assignment_schema_hash == OOS_ASSIGNMENT_SCHEMA_HASH
    # the projection must equal the refs' ordered ids
    with pytest.raises(ValueError, match="projection|regime_fit_ids"):
        RegimeOosAssignmentPayload(
            **{**payload.model_dump(), "regime_fit_ids": tuple(reversed(payload.regime_fit_ids))}
        )
    # a ref with a different sidecar hash mints a different identity
    other_refs = tuple(
        FitAssignmentRef(
            regime_fit_id=ref.regime_fit_id,
            assignments_sidecar_sha256=("0" * 64 if index == 0 else ref.assignments_sidecar_sha256),
            assignment_schema_hash=ref.assignment_schema_hash,
        )
        for index, ref in enumerate(refs)
    )
    other = RegimeOosAssignmentPayload(
        **{**payload.model_dump(), "regime_fit_assignment_refs": other_refs}
    )
    assert hashlib.sha256(b"x") and other != payload
    from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256

    assert canonical_contract_sha256(other) != envelope.regime_oos_assignment_id
    # the consulted hash covers EVERY value column that can change the output
    consulted = pd.concat([v.frame for v in verified.values()], ignore_index=True)
    base = consulted_assignments_hash(consulted)
    assert base == payload.consulted_assignments_hash
    valid_index = consulted.index[consulted["valid"].astype(bool)][0]
    for column, value in (
        ("canonical_reporting_cluster_id", 99),
        ("assignment_margin", 123.0),
        ("assigned_distance", 123.0),
        ("distances", [9.0, 9.0, 9.0]),
    ):
        changed = consulted.copy()
        changed.at[valid_index, column] = value
        assert consulted_assignments_hash(changed) != base, column
    invalid_index = consulted.index[~consulted["valid"].astype(bool)]
    if len(invalid_index):
        changed = consulted.copy()
        changed.loc[invalid_index[0], "missing_reason"] = "coverage_gap"
        assert consulted_assignments_hash(changed) != base
    # the builder refuses anything but verified fit assignments
    with pytest.raises(TypeError, match="VerifiedFitAssignments"):
        build_regime_oos_assignment_artifact(
            frame,
            protocol=protocol,
            verified_fit_assignments={0: run.assignments},
            regime_fold_set_id=run.fold_set_id,
            fold_schedule_id=schedule.fold_schedule_id,
            candidate_as_of=as_of,
            candidate_as_of_source_ref=f"bundle_feature_view:{fixture.view.view_id}",
            candidate_as_of_stage=AvailabilityStage.ENTRY_DECISION,
        )
    # persisted → reloaded frames satisfy the descriptive invariants
    from alpha_lab.agents.data_infra.ifvg.ml.regime_oos_assignment import (
        save_regime_oos_assignment,
    )

    save_regime_oos_assignment(root, envelope, table)
    stored = load_regime_oos_assignment(root, envelope.regime_oos_assignment_id)
    stored_frame = load_regime_oos_assignment_frame(root, stored)
    validate_assignment_rows(stored_frame, cluster_count=3, kind="descriptive")
    valid = stored_frame[stored_frame["valid"].astype(bool)]
    assert valid["canonical_reporting_cluster_id"].notna().all()
    assert valid["distances"].map(len).eq(3).all()


# ── §3.1 the executor derives from stored bytes, never the memory object (F-02) ─


def test_executor_builds_the_descriptive_artifact_from_verified_fit_bytes(tmp_path, monkeypatch):
    root = tmp_path / "store"
    source = persisted_candidate_source(root)
    captured: dict[str, object] = {}
    real_run = executor_module.run_regime_protocol_from_source

    def _capture(*args, **kwargs):
        run = real_run(*args, **kwargs)
        captured["run"] = run
        return run

    real_persist_assessment = executor_module.persist_regime_assessment

    def _mutate_then_persist(root_, assessment):
        # the fits are already persisted here; poison the IN-MEMORY frame the
        # old executor consumed — the artifact must not see any of it
        run = captured["run"]
        run.assignments["assignment_margin"] = 999.0
        run.assignments["canonical_reporting_cluster_id"] = 0
        return real_persist_assessment(root_, assessment)

    monkeypatch.setattr(executor_module, "run_regime_protocol_from_source", _capture)
    monkeypatch.setattr(executor_module, "persist_regime_assessment", _mutate_then_persist)
    result = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=2,
    )
    assert captured["run"] is result.run
    frame = load_regime_oos_assignment_frame(root, result.oos_assignment)
    valid = frame[frame["valid"].astype(bool)]
    assert len(valid) > 0
    assert not (valid["assignment_margin"] == 999.0).any()
    # every value in the artifact traces to the STORED sidecar bytes
    stored = pd.concat(
        [
            load_regime_fit_assignments(root, fit_id).frame
            for fit_id in result.regime_fit_ids
        ],
        ignore_index=True,
    )
    stored_valid = stored[(stored["partition"] == "test") & stored["valid"].astype(bool)]
    stored_margin = stored_valid.set_index(stored_valid["row_id"].astype(str))["assignment_margin"]
    assert np.allclose(
        valid.set_index("candidate_id")["assignment_margin"].loc[stored_margin.index].to_numpy(),
        stored_margin.to_numpy(),
    )
    assert set(result.verified_fit_assignments) == {
        fit.fold_index for fit in result.run.fold_fits
    }
    for verified in result.verified_fit_assignments.values():
        assert isinstance(verified, VerifiedFitAssignments)
    refs = result.oos_assignment.payload.regime_fit_assignment_refs
    assert {ref.regime_fit_id for ref in refs} == set(result.regime_fit_ids)
    assert result.oos_assignment.payload.consulted_assignments_hash == (
        consulted_assignments_hash(stored)
    )


# ── §3.3 candidate as-of policy (F-04) ───────────────────────────────────────


_PANEL_ID = "b" * 64
_PANEL_INPUTS = ("cbp_realized_range_12", "cbp_realized_volatility_12")
_BP0 = resolve_bundle("BP0_CONTEXT_BAR_PANEL").resolved_feature_bundle_id


def _panel_protocol():
    return resolve_kmeans_protocol(
        input_feature_bundle_ref=_BP0,
        resolved_input_features=_PANEL_INPUTS,
        observation_granularity=ObservationGranularity.CONTEXT_BAR_PANEL,
        panel_interval_seconds=300,
        panel_source_artifact_id=_PANEL_ID,
        panel_as_of_policy_id="completed_bars_last_at_or_before_v1",
    )


def _panel(day: str, *, interval: int = 300, bars: int = 16) -> pd.DataFrame:
    base = pd.Timestamp(f"{day}T14:00:00Z")
    return pd.DataFrame(
        {
            "row_id": [f"{day}:{index}" for index in range(bars)],
            "trading_day": day,
            "bar_close_ts_utc": [
                (base + pd.Timedelta(seconds=interval * (index + 1))).isoformat()
                for index in range(bars)
            ],
            "cbp_valid": [index >= 12 for index in range(bars)],
            "cbp_missing_reason": [
                None if index >= 12 else "insufficient_trading_day_lookback"
                for index in range(bars)
            ],
        }
    )


def _panel_assignments(panel: pd.DataFrame) -> pd.DataFrame:
    valid = panel[panel["cbp_valid"].astype(bool)]
    return pd.DataFrame(
        {
            "row_id": valid["row_id"].astype(str),
            "fold_index": 0,
            "partition": "test",
            "regime_fit_id": "0" * 64,
            "fold_local_cluster_id": [index % 3 for index in range(len(valid))],
            "canonical_reporting_cluster_id": [index % 3 for index in range(len(valid))],
            "distances": [[0.1, 0.5, 0.9]] * len(valid),
            "assigned_distance": 0.1,
            "assignment_margin": 0.4,
            "valid": True,
            "missing_reason": None,
        }
    )


def test_a_candidate_without_the_stage_anchor_is_typed_not_refused_descriptive():
    day = "2026-01-13"
    panel = _panel(day)
    candidates = pd.DataFrame(
        {
            "candidate_id": ["anchored", "unanchored"],
            "as_of_ts_utc": [f"{day}T15:07:00Z", None],
        }
    )
    out = assign_panel_regimes_to_candidates(
        panel, _panel_assignments(panel), candidates, protocol=_panel_protocol()
    ).set_index("candidate_id")
    assert bool(out.loc["anchored", "valid"])
    assert not bool(out.loc["unanchored", "valid"])
    assert out.loc["unanchored", "missing_reason"] == "candidate_as_of_missing"
    assert "candidate_as_of_missing" in PANEL_ASSIGNMENT_MISSING_REASONS
    assert out.loc["unanchored", "panel_row_id"] is None or pd.isna(
        out.loc["unanchored", "panel_row_id"]
    )
    # the source hash represents the null deterministically
    assert candidate_as_of_source_hash(candidates) == candidate_as_of_source_hash(
        candidates.copy()
    )
    # an unparseable NON-null instant is still a hard error
    with pytest.raises(ValueError, match="parse"):
        assign_panel_regimes_to_candidates(
            panel,
            _panel_assignments(panel),
            pd.DataFrame({"candidate_id": ["x"], "as_of_ts_utc": ["not-a-time"]}),
            protocol=_panel_protocol(),
        )


def test_a_candidate_without_the_stage_anchor_is_typed_not_refused_fold_features(tmp_path):
    from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
        build_fold_set_artifact,
        persist_fold_set_artifact,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_fold_features import (
        REGIME_FOLD_FEATURE_MISSING_REASONS,
        PanelFoldFeatureInputs,
        build_regime_fold_features,
    )

    root = tmp_path / "panel"
    panel = persisted_panel_source(root)
    fixture = known_cluster_fixture(k=3, n=600)
    folds = build_context_folds(
        fixture.labeled_candidates, authorized_trading_days=fixture.trading_days
    )
    candidate_fold_set, definitions = build_fold_set_artifact(
        folds,
        schedule=panel.schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=fixture.view.view_id,
        minimum_train_observations=30,
        labeled=True,
    )
    persist_fold_set_artifact(root, candidate_fold_set, definitions)
    run = run_regime_protocol(
        panel.panel_frame,
        panel.folds,
        panel.protocol,
        source_artifact_ids=(panel.panel_envelope.context_bar_panel_artifact_id,),
        bootstrap_refits=2,
    )
    verified: dict[int, VerifiedFitAssignments] = {}
    for fold_fit in run.fold_fits:
        persist_regime_fit(
            root,
            fold_fit,
            run.assignments[run.assignments["fold_index"] == fold_fit.fold_index],
            observation_frame=panel.panel_frame,
        )
        verified[fold_fit.fold_index] = load_regime_fit_assignments(
            root, fold_fit.fit_envelope.regime_fit_id
        )
    valid_fold = next(fold for fold in folds.folds if fold.valid and fold.fold_index in verified)
    victim = str(valid_fold.test_candidate_ids[0])
    view_frame = fixture.view.frame.copy()
    view_frame.loc[view_frame["candidate_id"] == victim, "entry_ts_utc"] = None
    envelope, frame = build_regime_fold_features(
        protocol=panel.protocol,
        regime_run=run,
        fit_assignments=verified,
        candidate_fold_set=candidate_fold_set,
        candidate_folds=folds,
        regime_fold_set=panel.fold_set_envelope,
        schedule=panel.schedule,
        candidate_view_frame=view_frame,
        candidate_view_id=fixture.view.view_id,
        panel=PanelFoldFeatureInputs(
            panel_frame=panel.panel_frame, panel_artifact=panel.panel_envelope
        ),
    )
    columns = envelope.payload.columns
    row = frame[(frame["fold_index"] == valid_fold.fold_index) & (frame["candidate_id"] == victim)]
    assert len(row) == 1
    assert not bool(row.iloc[0][columns.valid])
    assert row.iloc[0][columns.missing_reason] == "candidate_as_of_missing"
    assert "candidate_as_of_missing" in REGIME_FOLD_FEATURE_MISSING_REASONS
    # the population is preserved: every train/test member of the fold has a row
    rows = frame[frame["fold_index"] == valid_fold.fold_index]
    assert len(rows) == len(valid_fold.train_candidate_ids) + len(valid_fold.test_candidate_ids)


# ── adversarial round RA-06: invalid-row linkage + valid-row self-consistency ─


def test_invalid_rows_retain_linkage_and_valid_rows_are_self_consistent():
    """RA-06: an invalid row keeps its reconciliation linkage (a non-null key;
    a non-null partition / fit id / fold index must be lawful) and a valid row
    is arithmetically self-consistent (`assigned_distance == distances[local]
    == min(distances)`, `assignment_margin == second − first ≥ 0`)."""

    # the reviewer's frames
    with pytest.raises(ValueError, match="assigned distance"):
        validate_assignment_rows(
            _frame(
                _valid_row(
                    distances=[1.0, 2.0, 3.0],
                    fold_local_cluster_id=2,
                    assigned_distance=99.0,
                    assignment_margin=-5.0,
                )
            ),
            cluster_count=3,
            kind="fit",
        )
    with pytest.raises(ValueError, match="margin"):
        validate_assignment_rows(
            _frame(
                _valid_row(
                    distances=[1.0, 2.0, 3.0],
                    fold_local_cluster_id=0,
                    assigned_distance=1.0,
                    assignment_margin=-5.0,
                )
            ),
            cluster_count=3,
            kind="fit",
        )
    # the local id must be the argmin of the distance vector
    with pytest.raises(ValueError, match="assigned distance"):
        validate_assignment_rows(
            _frame(
                _valid_row(
                    distances=[0.5, 0.1, 0.9],
                    fold_local_cluster_id=0,
                    assigned_distance=0.5,
                    assignment_margin=0.4,
                )
            ),
            cluster_count=3,
            kind="fit",
        )
    # the kernel's own arithmetic passes (margin = second − first)
    validate_assignment_rows(
        _frame(
            _valid_row(
                distances=[0.5, 0.1, 0.9],
                fold_local_cluster_id=1,
                assigned_distance=0.1,
                assignment_margin=0.4,
            )
        ),
        cluster_count=3,
        kind="fit",
    )
    # invalid rows: the `_typed()` shape (null fit / fold / partition) is lawful …
    validate_assignment_rows(
        _frame(_invalid_row(regime_fit_id=None, fold_index=None, partition=None)),
        cluster_count=3,
        kind="fit",
    )
    # … but an unlawful non-null partition, a malformed fit id, a negative
    # fold index or a null key are refused
    with pytest.raises(ValueError, match="partition"):
        validate_assignment_rows(_frame(_invalid_row(partition="x")), cluster_count=3, kind="fit")
    with pytest.raises(ValueError, match="fit id"):
        validate_assignment_rows(
            _frame(_invalid_row(regime_fit_id="zzz")), cluster_count=3, kind="fit"
        )
    with pytest.raises(ValueError, match="fold_index"):
        validate_assignment_rows(_frame(_invalid_row(fold_index=-1)), cluster_count=3, kind="fit")
    with pytest.raises(ValueError, match="row_id"):
        validate_assignment_rows(_frame(_invalid_row(row_id=None)), cluster_count=3, kind="fit")
    with pytest.raises(ValueError, match="row_id"):
        validate_assignment_rows(_frame(_valid_row(row_id="")), cluster_count=3, kind="fit")
    # the descriptive kind keys on candidate_id
    descriptive = pd.DataFrame(
        [
            {
                "candidate_id": None,
                "regime_fit_id": None,
                "fold_index": None,
                "partition": None,
                "panel_row_id": None,
                "fold_local_cluster_id": None,
                "canonical_reporting_cluster_id": None,
                "distances": None,
                "assigned_distance": np.nan,
                "assignment_margin": np.nan,
                "valid": False,
                "missing_reason": "coverage_gap",
                "elapsed_seconds_since_bar_close": np.nan,
            }
        ]
    )
    with pytest.raises(ValueError, match="candidate_id"):
        validate_assignment_rows(descriptive, cluster_count=3, kind="descriptive")
    validate_assignment_rows(
        descriptive.assign(candidate_id=["cand-9"]), cluster_count=3, kind="descriptive"
    )


def test_loader_refuses_a_sidecar_of_another_protocol(candidate_run, tmp_path, monkeypatch):
    """RA-06 (c): the verified load re-checks the sidecar's protocol column
    against the envelope, not only the fit id and the fold index."""

    from alpha_lab.agents.data_infra.ifvg.ml import regime_store as store_module

    fixture, _folds, _protocol, run = candidate_run
    root = tmp_path / "store"
    fold_fit = run.fold_fits[0]
    persist_regime_fit(
        root, fold_fit, _fold_assignments(run, fold_fit), observation_frame=fixture.view.frame
    )
    fit_id = fold_fit.fit_envelope.regime_fit_id
    real = store_module._assignments_from_bytes

    def _swapped(data: bytes) -> pd.DataFrame:
        frame = real(data)
        frame["resolved_regime_protocol_id"] = "f" * 64
        return frame

    monkeypatch.setattr(store_module, "_assignments_from_bytes", _swapped)
    with pytest.raises(ValueError, match="protocol"):
        load_regime_fit_assignments(root, fit_id)


# ── adversarial round RA-07: strict as-of identity + the recorded stage ──────


def test_candidate_as_of_hash_refuses_unparseable_anchors_and_records_the_stage(
    candidate_run, tmp_path
):
    """RA-07: the as-of source hash routes through the same parser as the
    assignment rule — an unparseable NON-null anchor is a hard error (never
    silently a null), a true null is represented deterministically — and the
    payload records which stage's anchor the hashed instants came from, so
    the same output bytes under another stage never share an identity."""

    fixture, _folds, protocol, run = candidate_run
    root = tmp_path / "store"
    verified = _persist_all(root, candidate_run)
    frame = _descriptive_frame(run, fixture, verified)
    schedule = derive_fold_schedule(fixture.trading_days)
    as_of = candidate_as_of_frame(fixture.view.frame, stage=AvailabilityStage.ENTRY_DECISION)
    garbage = as_of.copy()
    garbage.loc[garbage.index[-1], "as_of_ts_utc"] = "not-a-time"
    with pytest.raises(ValueError, match="parse"):
        candidate_as_of_source_hash(garbage)
    nulled = as_of.copy()
    nulled.loc[nulled.index[0], "as_of_ts_utc"] = None
    assert candidate_as_of_source_hash(nulled) == candidate_as_of_source_hash(nulled.copy())
    assert candidate_as_of_source_hash(nulled) != candidate_as_of_source_hash(as_of)

    def _build(candidate_as_of, stage):
        return build_regime_oos_assignment_artifact(
            frame,
            protocol=protocol,
            verified_fit_assignments=verified,
            regime_fold_set_id=run.fold_set_id,
            fold_schedule_id=schedule.fold_schedule_id,
            candidate_as_of=candidate_as_of,
            candidate_as_of_source_ref=f"bundle_feature_view:{fixture.view.view_id}",
            candidate_as_of_stage=stage,
        )

    with pytest.raises(ValueError, match="parse"):
        _build(garbage, AvailabilityStage.ENTRY_DECISION)
    entry, _ = _build(as_of, AvailabilityStage.ENTRY_DECISION)
    assert entry.payload.candidate_as_of_stage is AvailabilityStage.ENTRY_DECISION
    tap, _ = _build(as_of, AvailabilityStage.HTF_TAP)
    assert tap.payload.candidate_as_of_stage is AvailabilityStage.HTF_TAP
    assert tap.regime_oos_assignment_id != entry.regime_oos_assignment_id
    # the payload refuses a stage that is not an AvailabilityStage
    with pytest.raises(ValueError):
        RegimeOosAssignmentPayload(
            **{**entry.payload.model_dump(), "candidate_as_of_stage": "not_a_stage"}
        )


def test_executor_refuses_an_unparseable_anchor_and_types_a_null_one(tmp_path, monkeypatch):
    """RA-07 at the executor: the candidate grain hashes the anchors of the
    requested stage through the strict parser (a garbage anchor halts the
    execution; a null anchor is a typed, deterministic null) and the
    artifact records that stage."""

    from alpha_lab.agents.data_infra.ifvg.ml import regime_oos_assignment as oos_module

    root = tmp_path / "store"
    source = persisted_candidate_source(root)
    real = executor_module.candidate_as_of_frame

    def _poisoned(frame, *, stage):
        out = real(frame, stage=stage).copy()
        out.loc[out.index[-1], "as_of_ts_utc"] = "not-a-time"
        return out

    monkeypatch.setattr(executor_module, "candidate_as_of_frame", _poisoned)
    with pytest.raises(ValueError, match="parse"):
        execute_regime_protocol(
            root,
            protocol=source.protocol,
            observation_source=source.source_ref,
            fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
            bootstrap_refits=2,
        )

    def _nulled(frame, *, stage):
        out = real(frame, stage=stage).copy()
        out.loc[out.index[0], "as_of_ts_utc"] = None
        return out

    monkeypatch.setattr(executor_module, "candidate_as_of_frame", _nulled)
    result = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=2,
    )
    payload = result.oos_assignment.payload
    assert payload.candidate_as_of_stage is AvailabilityStage.ENTRY_DECISION
    monkeypatch.setattr(executor_module, "candidate_as_of_frame", real)
    clean = execute_regime_protocol(
        root,
        protocol=source.protocol,
        observation_source=source.source_ref,
        fold_set_artifact_id=source.fold_set_envelope.fold_set_artifact_id,
        bootstrap_refits=2,
    )
    # the null anchor is represented in the identity; the clean run differs
    assert clean.oos_assignment.regime_oos_assignment_id != (
        result.oos_assignment.regime_oos_assignment_id
    )
    assert clean.oos_assignment.payload.candidate_as_of_stage is AvailabilityStage.ENTRY_DECISION
    assert oos_module.OOS_ASSIGNMENT_FORMULA_VERSION == "regime_oos_assignment_v2"
