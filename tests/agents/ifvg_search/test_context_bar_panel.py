"""R6.1 workstream A — the context-bar panel materializer (plan §6.A, §9.1
``test_incomplete_source_bar_invalidates_entire_panel_window``, §9.2)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
    CONTEXT_BAR_PANEL_FEATURES,
    LOOKBACK_BARS,
    MINIMUM_SOURCE_BARS,
    PANEL_INTERVALS_SECONDS_V1,
    STD_DDOF,
)
from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_materializer import (
    CONTEXT_BAR_PANEL_STORE,
    PANEL_SIDECAR,
    ContextBarPanelArtifactPayload,
    compute_context_bar_panel_features,
    load_context_bar_panel_artifact,
    load_context_bar_panel_frame,
    load_context_bar_panel_validity,
    load_verified_context_bar_panel,
    materialize_context_bar_panel,
    save_context_bar_panel_artifact,
    verify_context_bar_panel_frame,
)
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
    ReplayChartStoreError,
    load_verified_replay_chart_artifact,
    resample_label_bars,
)
from alpha_lab.agents.data_infra.ifvg.search.store import SearchStoreError, envelope_destination
from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
    planted_regime_for_bar,
    synthetic_label_source_1m,
    synthetic_pair_ref,
    synthetic_trading_days,
    write_synthetic_replay_chart_artifact,
)

DAYS = synthetic_trading_days(55)


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    base = tmp_path_factory.mktemp("replay_chart")
    bars = synthetic_label_source_1m(
        DAYS,
        early_close_day=DAYS[5],
        early_close_minute=8 * 60 + 403,  # not a 5m multiple → partial last bar
        drop_window=(DAYS[3], 8 * 60 + 70, 8 * 60 + 72),
    )
    pair = synthetic_pair_ref()
    artifact_id = write_synthetic_replay_chart_artifact(base, bars, pair)
    return load_verified_replay_chart_artifact(base, artifact_id, expected_pair=pair), base, pair


@pytest.fixture(scope="module")
def materialized(replay):
    artifact, _base, _pair = replay
    return materialize_context_bar_panel(artifact, panel_interval_seconds=300)


def test_verified_artifact_only_input_and_bars_rehash(replay, tmp_path):
    artifact, base, pair = replay
    with pytest.raises(TypeError, match="VerifiedReplayChartArtifact"):
        materialize_context_bar_panel(artifact.bars_tf, panel_interval_seconds=300)  # type: ignore[arg-type]
    # tampering the on-disk bars refuses even when the in-memory frame is intact
    copied = tmp_path / "copy"
    shutil.copytree(base, copied)
    tampered = load_verified_replay_chart_artifact(copied, artifact.artifact_id, expected_pair=pair)
    (copied / artifact.artifact_id / "bars_tf.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="byte size|do not hash"):
        materialize_context_bar_panel(tampered, panel_interval_seconds=300)
    with pytest.raises(ValueError, match="not owner-registered"):
        materialize_context_bar_panel(artifact, panel_interval_seconds=600)


def test_partial_final_bar_excluded_and_counted(materialized, replay):
    envelope, panel, _validity = materialized
    artifact = replay[0]
    bars_5m = artifact.bars_tf[artifact.bars_tf["timeframe_seconds"] == 300]
    partial = bars_5m[bars_5m["is_final_partial"].astype(bool)]
    assert len(partial) == 1 and partial["trading_day"].iloc[0] == DAYS[5]
    assert envelope.excluded_partial_bar_count == 1
    assert partial["bar_id"].iloc[0] not in set(panel["row_id"])
    assert envelope.row_count == len(panel) == len(bars_5m) - 1


def test_warmup_reset_and_lookback_gap_typing(materialized):
    _envelope, panel, _validity = materialized
    for day in (DAYS[0], DAYS[1]):
        rows = panel[panel["trading_day"] == day].sort_values("bar_index")
        assert list(rows["cbp_missing_reason"].head(LOOKBACK_BARS)) == (
            ["insufficient_trading_day_lookback"] * LOOKBACK_BARS
        )
        assert rows["cbp_valid"].iloc[LOOKBACK_BARS]
        assert rows[[*CONTEXT_BAR_PANEL_FEATURES]].head(LOOKBACK_BARS).isna().all().all()
    # a gap in the completed bars is typed, never bridged
    bars = resample_label_bars(synthetic_label_source_1m(DAYS[:2]), 300)
    gapped = bars[~((bars["trading_day"] == DAYS[0]) & (bars["bar_index"] == 110))]
    frame, _ = compute_context_bar_panel_features(gapped, interval_seconds=300)
    day0 = frame[frame["trading_day"] == DAYS[0]].set_index("bar_index")
    assert day0.loc[111, "cbp_missing_reason"] == "lookback_window_gap"
    assert day0.loc[122, "cbp_missing_reason"] == "lookback_window_gap"
    assert day0.loc[123, "cbp_valid"]
    assert day0.loc[109, "cbp_valid"]


def test_incomplete_source_bar_invalidates_entire_panel_window(materialized):
    """§9.1: any observed/expected 1m mismatch in the 13-bar window nulls every
    feature and persists the offending bar evidence."""

    _envelope, panel, validity = materialized
    day = panel[panel["trading_day"] == DAYS[3]].set_index("bar_index")
    incomplete_index = (8 * 60 + 70) // 5  # the dropped minutes 70..71 of 5m bar 110
    assert day.loc[incomplete_index, "observed_1m_count"] == 3
    affected = list(range(incomplete_index, incomplete_index + MINIMUM_SOURCE_BARS))
    for index in affected:
        assert day.loc[index, "cbp_missing_reason"] == "source_bar_incomplete"
        assert not day.loc[index, "cbp_valid"]
        assert day.loc[index, list(CONTEXT_BAR_PANEL_FEATURES)].isna().all()
    assert day.loc[incomplete_index + MINIMUM_SOURCE_BARS, "cbp_valid"]
    assert day.loc[incomplete_index - 1, "cbp_valid"]
    offending = validity[validity["row_id"].str.startswith(f"300s:{DAYS[3]}:")]
    assert len(offending) == MINIMUM_SOURCE_BARS
    assert set(offending["offending_bar_id"]) == {f"300s:{DAYS[3]}:{incomplete_index}"}
    assert set(offending["observed_1m_count"]) == {3} and set(offending["expected_1m_count"]) == {5}
    assert set(offending["reason"]) == {"source_bar_incomplete"}


def test_formula_nan_keeps_row_valid_and_ddof_zero_pinned():
    bars = resample_label_bars(synthetic_label_source_1m(DAYS[:1]), 300)
    flat = bars.copy()
    # a flat prior window: zero range and zero volume dispersion → formula NaN
    prior = (flat["bar_index"] >= 100) & (flat["bar_index"] < 112)
    for column in ("open_ticks", "high_ticks", "low_ticks", "close_ticks"):
        flat.loc[prior, column] = 20_000
    flat.loc[prior, "volume"] = 50
    frame, _ = compute_context_bar_panel_features(flat, interval_seconds=300)
    row = frame[frame["bar_index"] == 112].iloc[0]
    assert row["cbp_valid"] and row["cbp_missing_reason"] is None
    assert np.isnan(row["cbp_range_compression_ratio_12"])
    assert np.isnan(row["cbp_volume_intensity_zscore_12"])
    assert row["cbp_session_state"] in ("asia", "london", "ny", "none", "closed")
    # ddof 0 against a hand computation on an untouched window
    normal, _ = compute_context_bar_panel_features(bars, interval_seconds=300)
    target = normal[normal["bar_index"] == 130].iloc[0]
    window = bars[(bars["bar_index"] >= 118) & (bars["bar_index"] <= 130)].sort_values("bar_index")
    closes = window["close_ticks"].to_numpy(dtype=float)
    deltas = np.diff(closes)
    assert target["cbp_realized_volatility_12"] == pytest.approx(np.std(deltas, ddof=STD_DDOF))
    assert target["cbp_realized_volatility_12"] != pytest.approx(np.std(deltas, ddof=1))
    ranges = (window["high_ticks"] - window["low_ticks"]).to_numpy(dtype=float)
    assert target["cbp_realized_range_12"] == pytest.approx(ranges[1:].mean())
    assert target["cbp_range_compression_ratio_12"] == pytest.approx(
        ranges[-1] / ranges[:-1].mean()
    )
    assert target["cbp_path_efficiency_12"] == pytest.approx(
        abs(closes[-1] - closes[0]) / np.abs(deltas).sum()
    )
    volumes = window["volume"].to_numpy(dtype=float)
    assert target["cbp_volume_intensity_zscore_12"] == pytest.approx(
        (volumes[-1] - volumes[:-1].mean()) / np.std(volumes[:-1], ddof=0)
    )


def test_session_state_and_position_are_dst_aware():
    days = ("2026-03-06", "2026-03-09")  # Friday before / Monday after the DST switch
    bars = resample_label_bars(
        synthetic_label_source_1m(days, start_minute=0, end_minute=23 * 60), 300
    )
    frame, _ = compute_context_bar_panel_features(bars, interval_seconds=300)
    valid = frame[frame["cbp_valid"]]
    for day in days:
        rows = valid[valid["trading_day"] == day].set_index("bar_index")
        # the NY session (08:00–14:00 ET) starts at 08:00 ET whatever the UTC offset
        ny = rows[rows["cbp_session_state"] == "ny"]
        assert ny["cbp_bar_position_in_session"].min() == pytest.approx(300 / (6 * 3600))
        assert ny["cbp_bar_position_in_session"].max() == pytest.approx(1.0)
        first_ny_close = pd.Timestamp(ny["bar_close_ts_utc"].iloc[0]).tz_convert("America/New_York")
        assert first_ny_close.strftime("%H:%M") == "08:05"
        gap = rows[rows["cbp_session_state"] == "none"]
        assert gap["cbp_bar_position_in_session"].isna().all()
        assert (rows["cbp_bar_position_in_session"].dropna().between(0.0, 1.0)).all()
    # an early close leaves the value < 1 (no future knowledge)
    early = resample_label_bars(
        synthetic_label_source_1m(
            days[:1],
            start_minute=0,
            end_minute=23 * 60,
            early_close_day=days[0],
            early_close_minute=17 * 60,
        ),
        300,
    )
    early_frame, _ = compute_context_bar_panel_features(early, interval_seconds=300)
    ny_early = early_frame[(early_frame["cbp_session_state"] == "ny") & early_frame["cbp_valid"]]
    assert ny_early["cbp_bar_position_in_session"].max() < 1.0


def test_features_are_point_in_time_by_construction():
    bars = resample_label_bars(synthetic_label_source_1m(DAYS[:2]), 300)
    base, _ = compute_context_bar_panel_features(bars, interval_seconds=300)
    perturbed = bars.copy()
    later = perturbed["bar_index"] > 120
    perturbed.loc[later, ["high_ticks", "close_ticks"]] += 500
    perturbed.loc[later, "volume"] *= 3
    after, _ = compute_context_bar_panel_features(perturbed, interval_seconds=300)
    prefix = base["bar_index"] <= 120
    pd.testing.assert_frame_equal(
        base[prefix].reset_index(drop=True), after[prefix].reset_index(drop=True)
    )
    # row-order invariance
    shuffled, _ = compute_context_bar_panel_features(
        bars.sample(frac=1.0, random_state=3), interval_seconds=300
    )
    pd.testing.assert_frame_equal(base, shuffled)


def test_planted_regimes_separate_in_the_features(materialized):
    _envelope, panel, _ = materialized
    valid = panel[panel["cbp_valid"]].copy()
    minutes = valid["bar_index"] * 5 + 4  # the bar's last minute
    valid["planted"] = [
        planted_regime_for_bar(int(m), start_minute=8 * 60, end_minute=20 * 60) for m in minutes
    ]
    means = valid.groupby("planted")["cbp_realized_range_12"].mean()
    assert means[0] < means[1] < means[2]


def test_identity_moves_with_interval_pair_and_replay(replay, materialized, tmp_path):
    artifact, base, pair = replay
    envelope, panel, validity = materialized
    fifteen, _, _ = materialize_context_bar_panel(artifact, panel_interval_seconds=900)
    assert fifteen.context_bar_panel_artifact_id != envelope.context_bar_panel_artifact_id
    assert fifteen.payload.panel_interval_seconds == 900
    other_pair = synthetic_pair_ref("other")
    other_base = tmp_path / "other"
    other_id = write_synthetic_replay_chart_artifact(
        other_base, synthetic_label_source_1m(DAYS), other_pair
    )
    other = load_verified_replay_chart_artifact(other_base, other_id, expected_pair=other_pair)
    other_envelope, _, _ = materialize_context_bar_panel(other, panel_interval_seconds=300)
    assert other_envelope.context_bar_panel_artifact_id != envelope.context_bar_panel_artifact_id
    assert (
        other_envelope.payload.replay_chart_artifact_id
        != envelope.payload.replay_chart_artifact_id
    )
    # determinism
    again, _, _ = materialize_context_bar_panel(artifact, panel_interval_seconds=300)
    assert again.model_dump(mode="json") == envelope.model_dump(mode="json")
    assert set(PANEL_INTERVALS_SECONDS_V1) == {300, 900}


def test_save_reload_reuse_tamper_relocation(tmp_path, materialized):
    envelope, panel, validity = materialized
    root = tmp_path / "store"
    save_context_bar_panel_artifact(root, envelope, panel, validity)
    save_context_bar_panel_artifact(root, envelope, panel, validity)  # verified reuse
    stored = load_context_bar_panel_artifact(root, envelope.context_bar_panel_artifact_id)
    frame = load_context_bar_panel_frame(root, stored)
    verify_context_bar_panel_frame(stored, frame)
    assert len(frame) == envelope.row_count
    stored_validity = load_context_bar_panel_validity(root, stored)
    assert len(stored_validity) == envelope.validity_row_count
    _again, verified = load_verified_context_bar_panel(root, envelope.context_bar_panel_artifact_id)
    pd.testing.assert_frame_equal(verified, frame)
    with pytest.raises(ValueError, match="does not hash"):
        verify_context_bar_panel_frame(stored, frame.iloc[:-1])
    with pytest.raises(ValueError, match="does not hash"):
        save_context_bar_panel_artifact(tmp_path / "x", envelope, panel.iloc[:-1], validity)
    moved = tmp_path / "moved"
    shutil.copytree(root, moved)
    _, relocated = load_verified_context_bar_panel(moved, envelope.context_bar_panel_artifact_id)
    pd.testing.assert_frame_equal(relocated, frame)
    sidecar = (
        envelope_destination(root, CONTEXT_BAR_PANEL_STORE, envelope.context_bar_panel_artifact_id)
        / PANEL_SIDECAR
    )
    sidecar.write_bytes(sidecar.read_bytes() + b"\x00")
    with pytest.raises(SearchStoreError):
        load_context_bar_panel_frame(root, stored)


def test_payload_refuses_unregistered_interval_and_wrong_source_bars(materialized):
    envelope, _, _ = materialized
    dumped = envelope.payload.model_dump(mode="json")
    with pytest.raises(ValueError, match="not owner-registered"):
        ContextBarPanelArtifactPayload.model_validate({**dumped, "panel_interval_seconds": 600})
    with pytest.raises(ValueError, match="current bar"):
        ContextBarPanelArtifactPayload.model_validate({**dumped, "minimum_source_bars": 12})
    with pytest.raises(ValueError, match="resample rule"):
        ContextBarPanelArtifactPayload.model_validate({**dumped, "resample_rule_id": "other"})
    assert dumped["std_ddof"] == 0 and dumped["minimum_source_bars"] == 13
    assert dumped["intensity_source_field"] == "volume"


def test_replay_chart_fixture_is_a_real_verified_artifact(replay):
    artifact, base, pair = replay
    manifest = json.loads((base / artifact.artifact_id / "manifest.json").read_text("utf-8"))
    assert manifest["artifact_kind"] == "ifvg_replay_chart_v1"
    with pytest.raises(ReplayChartStoreError, match="different v2/v3 pair"):
        load_verified_replay_chart_artifact(
            base, artifact.artifact_id, expected_pair=synthetic_pair_ref("x")
        )
