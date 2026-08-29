"""R6.1 workstream B — fold schedules, panel-native / label-free candidate
folds, and the persisted fold-set artifact (plan §6.B, §9.1
``test_same_schedule_different_fold_sets_across_grains``, §9.2)."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_folds import build_context_folds
from alpha_lab.agents.data_infra.ifvg.fold_schedules import (
    BOUNDARY_POLICY_ID_V1,
    FOLD_PROTOCOL_ID_V1,
    PURGE_POLICY_ID_V1,
    FoldSchedulePayload,
    build_candidate_folds_from_schedule,
    build_context_bar_panel_folds,
    derive_fold_schedule,
)
from alpha_lab.agents.data_infra.ifvg.ml.fold_set_artifact import (
    FOLD_SET_STORE,
    assert_same_fold_schedule,
    build_fold_set_artifact,
    fold_set_id,
    load_fold_set_artifact,
    persist_fold_schedule,
    persist_fold_set_artifact,
)
from alpha_lab.agents.data_infra.ifvg.ml.regime_contracts import ObservationGranularity
from alpha_lab.agents.data_infra.ifvg.search.pipeline import FOLD_PROTOCOL_ID_V1 as PIPELINE_ID
from alpha_lab.agents.data_infra.ifvg.search.store import (
    SearchStoreError,
    envelope_destination,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
    known_cluster_fixture,
)
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
    synthetic_label_source_1m,
    synthetic_trading_days,
)


@pytest.fixture(scope="module")
def cluster_fixture():
    return known_cluster_fixture(k=3, n=600)


@pytest.fixture(scope="module")
def panel_frame():
    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
        compute_context_bar_panel_features,
    )
    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import resample_label_bars

    days = synthetic_trading_days(55)
    bars = resample_label_bars(synthetic_label_source_1m(days), 300)
    panel, _validity = compute_context_bar_panel_features(bars, interval_seconds=300)
    return days, panel


def test_schedule_windows_equal_the_labeled_builder_day_for_day(cluster_fixture):
    schedule = derive_fold_schedule(cluster_fixture.trading_days)
    labeled = build_context_folds(
        cluster_fixture.labeled_candidates,
        authorized_trading_days=cluster_fixture.trading_days,
    )
    assert [(w.fold_index, w.train_days, w.test_days) for w in schedule.payload.windows] == [
        (f.fold_index, f.train_days, f.test_days) for f in labeled.folds
    ]
    assert schedule.payload.fold_protocol_id == FOLD_PROTOCOL_ID_V1 == PIPELINE_ID
    assert schedule.payload.purge_policy_id == PURGE_POLICY_ID_V1
    assert schedule.payload.boundary_policy_id == BOUNDARY_POLICY_ID_V1
    # expanding train, stepped test
    assert schedule.payload.windows[1].train_days == cluster_fixture.trading_days[:45]


def test_schedule_identity_moves_with_days_step_embargo_and_policies(cluster_fixture):
    days = cluster_fixture.trading_days
    base = derive_fold_schedule(days)
    assert derive_fold_schedule(days).fold_schedule_id == base.fold_schedule_id
    fewer = derive_fold_schedule(days[:-5])
    assert fewer.fold_schedule_id != base.fold_schedule_id
    payload = base.payload.model_dump(mode="json")
    for field, value in (
        ("embargo_days", 3),
        ("step_days", 10),
        ("train_days", 30),
        ("purge_policy_id", "some_other_policy"),
        ("fold_protocol_id", "ifvg_context_walkforward_30_5_5_2_v1"),
    ):
        with pytest.raises(ValueError):
            FoldSchedulePayload.model_validate({**payload, field: value})
    # windows that do not derive from the days are refused
    with pytest.raises(ValueError, match="do not derive"):
        FoldSchedulePayload.model_validate({**payload, "windows": payload["windows"][:-1]})
    with pytest.raises(ValueError, match="unique and chronological"):
        derive_fold_schedule(tuple(reversed(days)))


def test_label_free_candidate_folds_purge_by_interval_or_entry(cluster_fixture):
    frame = cluster_fixture.labeled_candidates.drop(columns=["binary_target", "gross_r", "net_r"])
    folds = build_candidate_folds_from_schedule(
        frame, authorized_trading_days=cluster_fixture.trading_days
    )
    assert [f.valid for f in folds.folds] == [True, True, True]
    assert all(f.training_prevalence is None for f in folds.folds)
    labeled = build_context_folds(
        cluster_fixture.labeled_candidates,
        authorized_trading_days=cluster_fixture.trading_days,
    )
    for free, lab in zip(folds.folds, labeled.folds, strict=True):
        assert (free.train_days, free.test_days) == (lab.train_days, lab.test_days)
        assert free.embargoed_candidate_ids == lab.embargoed_candidate_ids
        assert free.excluded_boundary_setup_ids == lab.excluded_boundary_setup_ids
    # a candidate whose resolution reaches into the test window is purged;
    # without a resolution column the entry instant alone decides
    poisoned = frame.copy()
    first_test_entry = pd.Timestamp(
        cluster_fixture.labeled_candidates.set_index("candidate_id").loc[
            folds.folds[0].test_candidate_ids[0], "entry_ts_utc"
        ]
    )
    victim = folds.folds[0].train_candidate_ids[0]
    poisoned.loc[poisoned["candidate_id"] == victim, "resolution_ts_utc"] = (
        first_test_entry + pd.Timedelta(hours=1)
    ).isoformat()
    purged = build_candidate_folds_from_schedule(
        poisoned, authorized_trading_days=cluster_fixture.trading_days
    )
    assert victim in purged.folds[0].purged_candidate_ids
    assert victim not in purged.folds[0].train_candidate_ids
    entry_only = build_candidate_folds_from_schedule(
        frame.drop(columns=["resolution_ts_utc"]),
        authorized_trading_days=cluster_fixture.trading_days,
    )
    assert entry_only.folds[0].purged_candidate_ids == ()
    # floors and refusals
    tiny = build_candidate_folds_from_schedule(
        frame, authorized_trading_days=cluster_fixture.trading_days, minimum_train_rows=10_000
    )
    assert {f.invalid_reason for f in tiny.folds} == {"insufficient_train_candidates"}
    with pytest.raises(ValueError, match="missing columns"):
        build_candidate_folds_from_schedule(
            frame.drop(columns=["setup_id"]),
            authorized_trading_days=cluster_fixture.trading_days,
        )
    with pytest.raises(ValueError, match="unauthorized trading day"):
        build_candidate_folds_from_schedule(
            frame, authorized_trading_days=cluster_fixture.trading_days[:10]
        )


def test_panel_folds_embargo_purge_floor_and_refusals(panel_frame):
    days, panel = panel_frame
    folds = build_context_bar_panel_folds(panel, authorized_trading_days=days)
    assert [f.valid for f in folds.folds] == [True, True, True]
    first = folds.folds[0]
    assert first.train_days == days[:40] and first.test_days == days[40:45]
    embargo_days = set(days[38:40])
    by_row = panel.set_index("row_id")
    assert {by_row.loc[r, "trading_day"] for r in first.embargoed_candidate_ids} == embargo_days
    assert not any(
        by_row.loc[r, "trading_day"] in embargo_days for r in first.train_candidate_ids
    )
    assert first.purged_candidate_ids == ()  # training days precede test days
    assert first.training_prevalence is None
    assert folds.assignment["partition"].isin(["train", "test"]).all()
    # the stamped floor (300 valid training bars) decides validity
    thin = build_context_bar_panel_folds(
        panel, authorized_trading_days=days, minimum_train_rows=1_000_000
    )
    assert {f.invalid_reason for f in thin.folds} == {"insufficient_train_rows"}
    assert thin.status == "insufficient_train_rows"
    with pytest.raises(ValueError, match="missing columns"):
        build_context_bar_panel_folds(
            panel.drop(columns=["bar_open_ts_utc"]), authorized_trading_days=days
        )
    with pytest.raises(ValueError, match="unique non-null row ids"):
        build_context_bar_panel_folds(
            pd.concat([panel, panel.head(1)]), authorized_trading_days=days
        )
    # a malformed bar that spans into the test window is purged (the guard)
    malformed = panel.copy()
    victim = first.train_candidate_ids[0]
    malformed.loc[malformed["row_id"] == victim, "bar_close_ts_utc"] = (
        pd.Timestamp(panel.set_index("row_id").loc[first.test_candidate_ids[0], "bar_close_ts_utc"])
        + pd.Timedelta(seconds=1)
    ).isoformat()
    purged = build_context_bar_panel_folds(malformed, authorized_trading_days=days)
    assert victim in purged.folds[0].purged_candidate_ids


def test_same_schedule_different_fold_sets_across_grains(cluster_fixture, panel_frame):
    """§9.1: panel and candidate fold sets over the SAME days share the
    schedule identity and per-fold windows but never a fold_set_id; the
    cross-grain rule accepts them and refuses a different schedule."""

    days, panel = panel_frame
    labeled = build_context_folds(
        cluster_fixture.labeled_candidates, authorized_trading_days=days
    )
    panel_folds = build_context_bar_panel_folds(panel, authorized_trading_days=days)
    schedule = derive_fold_schedule(days)
    candidate_artifact, _ = build_fold_set_artifact(
        labeled,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id="a" * 64,
        minimum_train_observations=150,
        labeled=True,
    )
    panel_artifact, _ = build_fold_set_artifact(
        panel_folds,
        schedule=schedule,
        observation_grain=ObservationGranularity.CONTEXT_BAR_PANEL,
        observation_source_artifact_id="b" * 64,
        minimum_train_observations=300,
        labeled=False,
    )
    assert candidate_artifact.payload.fold_schedule_id == panel_artifact.payload.fold_schedule_id
    assert candidate_artifact.payload.fold_set_id != panel_artifact.payload.fold_set_id
    assert candidate_artifact.payload.observation_key == "candidate_id"
    assert panel_artifact.payload.observation_key == "row_id"
    assert_same_fold_schedule(candidate_artifact.payload, panel_artifact.payload)
    other_days = days[:-5]
    other = build_context_folds(
        cluster_fixture.labeled_candidates[
            cluster_fixture.labeled_candidates["trading_day"].isin(other_days)
        ],
        authorized_trading_days=other_days,
    )
    other_artifact, _ = build_fold_set_artifact(
        other,
        schedule=derive_fold_schedule(other_days),
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id="a" * 64,
        minimum_train_observations=150,
        labeled=True,
    )
    with pytest.raises(ValueError, match="different fold schedules"):
        assert_same_fold_schedule(panel_artifact.payload, other_artifact.payload)
    # a fold set built under another schedule cannot be wrapped as this one
    with pytest.raises(ValueError, match="schedule"):
        build_fold_set_artifact(
            other,
            schedule=schedule,
            observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
            observation_source_artifact_id="a" * 64,
            minimum_train_observations=150,
            labeled=True,
        )
    # the observation key must agree with the grain
    with pytest.raises(ValueError, match="keyed by"):
        type(panel_artifact.payload).model_validate(
            {**panel_artifact.payload.model_dump(mode="json"), "observation_key": "candidate_id"}
        )


def test_legacy_fold_set_hash_three_way_equality(cluster_fixture):
    """No existing identity moves: the service, ladder, and controlled-study
    hashes all delegate to ``fold_set_id``."""

    from alpha_lab.agents.data_infra.ifvg.ml import (
        controlled_feature_study,
        regime_service,
        supervised_ladder,
    )

    labeled = build_context_folds(
        cluster_fixture.labeled_candidates,
        authorized_trading_days=cluster_fixture.trading_days,
    )
    expected = fold_set_id(labeled)
    assert regime_service._fold_set_id(labeled) == expected
    assert controlled_feature_study._fold_hash(labeled) == expected
    assert supervised_ladder._fold_set_hash(labeled) == expected


def test_fold_set_artifact_persist_load_tamper_relocation(tmp_path, cluster_fixture):
    labeled = build_context_folds(
        cluster_fixture.labeled_candidates,
        authorized_trading_days=cluster_fixture.trading_days,
    )
    schedule = derive_fold_schedule(cluster_fixture.trading_days)
    envelope, definitions = build_fold_set_artifact(
        labeled,
        schedule=schedule,
        observation_grain=ObservationGranularity.CANDIDATE_STAGE_ROW,
        observation_source_artifact_id=cluster_fixture.view.view_id,
        minimum_train_observations=150,
        labeled=True,
    )
    root = tmp_path / "store"
    persist_fold_set_artifact(root, envelope, definitions)
    with pytest.raises(SearchStoreError, match="missing search-store entry"):
        # the fold set requires its schedule in the store
        load_fold_set_artifact(root, envelope.fold_set_artifact_id)
    persist_fold_schedule(root, schedule)
    persist_fold_set_artifact(root, envelope, definitions)  # verified reuse
    reloaded, rebuilt = load_fold_set_artifact(root, envelope.fold_set_artifact_id)
    assert reloaded.model_dump(mode="json") == envelope.model_dump(mode="json")
    assert fold_set_id(rebuilt) == envelope.payload.fold_set_id
    assert [f.model_dump(mode="json") for f in rebuilt.folds] == [
        f.model_dump(mode="json") for f in labeled.folds
    ]
    assert set(rebuilt.assignment["candidate_id"]) == set(labeled.assignment["candidate_id"])
    # relocation: the artifact directory moves wholesale and still verifies
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    again, _ = load_fold_set_artifact(moved, envelope.fold_set_artifact_id)
    assert again.fold_set_artifact_id == envelope.fold_set_artifact_id
    # tampering the sidecar fails closed
    directory = envelope_destination(root, FOLD_SET_STORE, envelope.fold_set_artifact_id)
    sidecar = directory / "fold_definitions.json"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload[0]["train_candidate_ids"] = payload[0]["train_candidate_ids"][:-1]
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SearchStoreError):
        load_fold_set_artifact(root, envelope.fold_set_artifact_id)
    # wrong definitions cannot be persisted under the envelope
    with pytest.raises(ValueError, match="do not hash"):
        persist_fold_set_artifact(tmp_path / "other", envelope, b"[]\n")
