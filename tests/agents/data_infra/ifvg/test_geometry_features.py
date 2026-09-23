"""Geometry formulas, causal clocks, source guards and immutable persistence."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.context_feature_view import _IDENTITY_COLUMNS
from alpha_lab.agents.data_infra.ifvg.features import geometry_features as geometry
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    BundleFeatureViewEnvelope,
    BundleFeatureViewPayload,
    load_bundle_feature_view,
    load_bundle_feature_view_frame,
    resolve_available_bundle_view,
    save_bundle_feature_view,
)
from alpha_lab.agents.data_infra.ifvg.features.feature_bundles import resolve_bundle
from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256

from .b0_fixture import real_b0_view


def _bars(count=24, day="2026-02-11", start="2026-02-11T06:03:00Z"):
    return pd.DataFrame({
        "bar_id": [f"60s:{day}:{index}" for index in range(count)],
        "trading_day": [day] * count, "timeframe_ticks": [60] * count,
        "availability_ts_utc": pd.date_range(start, periods=count, freq="min"),
        "open_ticks": np.arange(count) + 100, "high_ticks": np.arange(count) + 102,
        "low_ticks": np.arange(count) + 100, "close_ticks": np.arange(count) + 101,
        "is_complete": [True] * count, "is_partial": [False] * count,
    })


def _inputs(monkeypatch):
    original = real_b0_view()
    candidate = original.frame.iloc[[0]].copy()
    row = candidate.iloc[0]
    bars = _bars()
    for component in ("open", "high", "low", "close"):
        bars.loc[len(bars) - 1, f"{component}_ticks"] = int(
            row[f"geometry_entry_bar_{component}_ticks"]
        )
    bars.loc[len(bars) - 1, "bar_id"] = row["geometry_entry_bar_bar_id"]
    candidate["trading_day"] = "2026-02-11"
    for column in _IDENTITY_COLUMNS:
        if column not in candidate:
            candidate[column] = False if column == "is_warmup" else "fixture"
    view = replace(original, view_id="a" * 64, artifact_pair_hash="b" * 64,
                   feature_registry_hash="e" * 64, frame=candidate)
    data = bars.to_parquet(index=False)
    monkeypatch.setattr(geometry, "_verify_context_binding", lambda _view, _ref: data)
    reference = {
        "artifact_id": "c" * 64, "manifest_payload_sha256": "d" * 64,
        "sha256": geometry.bytes_sha256(data), "path": "fixture-only",
    }
    return view, bars, reference


def test_wilder_seed_recurrence_and_causal_future_mutation():
    bars = _bars()
    bars.loc[20, ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] = [200, 202, 200, 201]
    result = geometry.compute_wilder_atr20(bars)
    assert result.decision_atr20_ticks.iloc[:19].isna().all()
    assert result.decision_atr20_ticks.iloc[19] == 2
    # Previous close=120, TR21=max(2,82,80)=82; Wilder recursion, not SMA.
    assert result.decision_atr20_ticks.iloc[20] == pytest.approx((19 * 2 + 82) / 20)
    future = bars.copy()
    future.loc[21:, ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] += 50000
    changed = geometry.compute_wilder_atr20(future)
    pd.testing.assert_series_equal(result.decision_atr20_ticks[:21],
                                   changed.decision_atr20_ticks[:21])


def test_partial_bars_gaps_and_trading_day_reset():
    bars = _bars()
    bars.loc[20, "is_partial"] = True
    bars.loc[20, ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] += 50000
    # The absent wall-clock minute is never synthesized; next observed gap enters TR.
    bars.loc[21:, "availability_ts_utc"] += pd.Timedelta(minutes=60)
    second = _bars(2, day="2026-02-12", start="2026-02-12T06:00:00Z")
    result = geometry.compute_wilder_atr20(pd.concat([bars, second], ignore_index=True))
    assert result.iloc[20].atr_missing_reason == "decision_bar_incomplete"
    assert result.iloc[21].atr_completed_bar_count == 21
    assert result.iloc[21].decision_atr20_ticks == pytest.approx((38 + 3) / 20)
    assert result.iloc[-1].atr_completed_bar_count == 2
    assert np.isnan(result.iloc[-1].decision_atr20_ticks)
    pd.testing.assert_frame_equal(result, geometry.compute_wilder_atr20(
        pd.concat([bars, second], ignore_index=True).sample(frac=1, random_state=1)
    ))


def test_three_ratios_and_persisted_bundle_reload(monkeypatch, tmp_path):
    view, bars, reference = _inputs(monkeypatch)
    before = view.frame.copy(deep=True)
    artifact, features = geometry.materialize_geometry_features(view, bars,
                                                               source_reference=reference)
    row = features.iloc[0]
    assert row[geometry.GEOMETRY_FEATURES[0]] == pytest.approx(
        row.close_through_margin_ticks / row.decision_atr20_ticks)
    assert row[geometry.GEOMETRY_FEATURES[1]] == pytest.approx(
        row.distance_to_htf_ticks / row.decision_atr20_ticks)
    assert row[geometry.GEOMETRY_FEATURES[2]] == pytest.approx(
        row.geometry_opposing_size_ticks / row.geometry_parent_size_ticks)
    geometry.save_geometry_features(tmp_path, artifact, features)
    loaded, reloaded = geometry.load_geometry_features(
        tmp_path, artifact.geometry_feature_artifact_id,
    )
    assert loaded == artifact
    pd.testing.assert_frame_equal(features, reloaded)
    envelope, frame = resolve_available_bundle_view(
        view, geometry.GEOMETRY_BUNDLE_KEY, geometry_features=features,
        geometry_feature_artifact=artifact,
    )
    saved = save_bundle_feature_view(tmp_path, envelope, frame)
    assert load_bundle_feature_view(tmp_path, saved.bundle_feature_view_id) == saved
    pd.testing.assert_frame_equal(frame, load_bundle_feature_view_frame(tmp_path, saved))
    pd.testing.assert_frame_equal(view.frame, before)
    base = resolve_bundle("B0_CORE").payload.resolved_feature_names
    assert resolve_bundle(geometry.GEOMETRY_BUNDLE_KEY).payload.resolved_feature_names == (
        *base, *geometry.GEOMETRY_FEATURES,
    )


@pytest.mark.parametrize("change,match", [
    ("bar_values", "differ from verified"), ("entry_link", "exact entry decision bar is absent"),
    ("entry_clock", "entry time disagrees"), ("selected_parent", "selected FVG linkage"),
    ("numerator", "numerator disagrees"), ("source_hash", "source file hash"),
    ("b0_evidence", "projection evidence hash"),
])
def test_refuses_source_event_and_value_drift(monkeypatch, change, match):
    view, bars, reference = _inputs(monkeypatch)
    if change == "bar_values":
        bars.loc[0, "high_ticks"] += 1
    elif change == "entry_link":
        view.frame["geometry_entry_bar_bar_id"] = "another-bar"
    elif change == "entry_clock":
        view.frame["entry_ts_utc"] += pd.Timedelta(minutes=1)
    elif change == "selected_parent":
        view.frame["geometry_parent_fvg_id"] = "another-parent"
    elif change == "numerator":
        view.frame["close_through_margin_ticks"] += 1
    elif change == "source_hash":
        reference["sha256"] = "0" * 64
    else:
        view.b0_projection_evidence["candidate_count"] += 1
    with pytest.raises(ValueError, match=match):
        geometry.materialize_geometry_features(view, bars, source_reference=reference)


def test_zero_volatility_and_missing_lookback_are_typed_nulls():
    bars = _bars()
    bars[["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] = 100
    result = geometry.compute_wilder_atr20(bars)
    assert result.iloc[18].atr_missing_reason == "insufficient_same_day_completed_bars"
    assert result.iloc[19].atr_missing_reason == "zero_atr_denominator"
    assert result.iloc[19].decision_atr20_ticks == 0


def test_candidate_zero_denominators_keep_rows_and_distinct_reasons(monkeypatch):
    view, bars, reference = _inputs(monkeypatch)
    bars[["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] = 100
    for component in ("open", "high", "low", "close"):
        view.frame[f"geometry_entry_bar_{component}_ticks"] = 100
    view.frame["geometry_parent_size_ticks"] = 0
    view.frame["geometry_parent_gap_high_ticks"] = view.frame["geometry_parent_gap_low_ticks"]
    data = bars.to_parquet(index=False)
    monkeypatch.setattr(geometry, "_verify_context_binding", lambda _view, _ref: data)
    reference["sha256"] = geometry.bytes_sha256(data)
    _artifact, frame = geometry.materialize_geometry_features(view, bars,
                                                             source_reference=reference)
    assert len(frame) == len(view.frame) == 1
    assert frame[list(geometry.GEOMETRY_FEATURES)].isna().all().all()
    assert frame.iloc[0].atr_missing_reason == "zero_atr_denominator"
    assert frame.iloc[0].parent_size_missing_reason == "zero_parent_size_denominator"


def test_geometry_evidence_tamper_and_population_mismatch_refuse(monkeypatch):
    view, bars, reference = _inputs(monkeypatch)
    artifact, frame = geometry.materialize_geometry_features(view, bars,
                                                            source_reference=reference)
    changed = frame.copy()
    changed.loc[0, geometry.GEOMETRY_FEATURES[0]] += 1
    with pytest.raises(ValueError, match="feature frame hash mismatch"):
        geometry.verify_geometry_feature_frame(artifact, changed)
    with pytest.raises(ValueError, match="population mismatch"):
        geometry.verify_geometry_feature_frame(artifact, frame.iloc[:0])
    with pytest.raises(ValueError, match="another candidate view"):
        resolve_available_bundle_view(
            replace(view, view_id="9" * 64), geometry.GEOMETRY_BUNDLE_KEY,
            geometry_features=frame, geometry_feature_artifact=artifact,
        )


def test_another_child_context_rejected_even_with_identical_bars(monkeypatch, tmp_path):
    from alpha_lab.agents.data_infra.ifvg.search.research_data import CONTEXT_STORE

    view = real_b0_view()
    artifact_id = "c" * 64
    source = tmp_path / CONTEXT_STORE / artifact_id / "forward_bars.parquet"
    context = SimpleNamespace(payload=SimpleNamespace(subject=SimpleNamespace(
        core_replay_id="other-child", v2_dataset_id="other-v2", v2_manifest_hash="other-manifest",
    )))
    monkeypatch.setattr(geometry, "load_verified_envelope", lambda *args: context)
    with pytest.raises(ValueError, match="another child/configuration"):
        geometry._verify_context_binding(view, {"path": str(source), "artifact_id": artifact_id})


def test_legacy_bundle_identity_is_unchanged():
    legacy = dict(view_id="a" * 64, feature_bundle_key="B0_CORE",
                  resolved_feature_bundle_id="b" * 64, feature_registry_hash="c" * 64,
                  candidate_count=1, resolved_feature_names=("direction",),
                  mbp1_feature_artifact_id=None, regime_fold_feature_artifact_id=None)
    expected = canonical_contract_sha256(legacy)
    envelope = BundleFeatureViewEnvelope.from_payload(BundleFeatureViewPayload(**legacy))
    assert envelope.bundle_feature_view_id == expected
    assert "geometry_feature_artifact_id" not in envelope.payload.model_dump(mode="json")
