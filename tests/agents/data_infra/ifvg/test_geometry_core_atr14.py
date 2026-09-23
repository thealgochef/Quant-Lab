"""Actual Core ATR14 convention, source-chain carry and versioned geometry."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features import geometry_core_atr14 as geometry
from alpha_lab.agents.data_infra.ifvg.features import geometry_features as historical
from alpha_lab.agents.data_infra.ifvg.features.bundle_feature_view import (
    resolve_available_bundle_view,
)

from .test_geometry_features import _bars as _old_bars
from .test_geometry_features import _inputs as _old_inputs


def _with_core_fields(bars):
    bars = bars.copy()
    bars["kind"] = "time"
    bars["close_ts_utc"] = bars.availability_ts_utc
    bars["logical_close_ts_utc"] = bars.availability_ts_utc
    bars["open_ts_utc"] = bars.availability_ts_utc - pd.Timedelta(minutes=1)
    bars["logical_open_ts_utc"] = bars.open_ts_utc
    bars["bar_index"] = np.arange(len(bars))
    bars["volume"] = 0
    bars["trade_count"] = 0
    bars["close_reason"] = None
    return bars


def _bars(*args, **kwargs):
    return _with_core_fields(_old_bars(*args, **kwargs))


def _inputs(monkeypatch):
    """Synthetic formula source, real selected geometry; no source-check bypass in production."""
    from alpha_lab.agents.data_infra.ifvg.search import store

    view, bars, reference = _old_inputs(monkeypatch)
    bars = _with_core_fields(bars)
    data = bars.to_parquet(index=False)
    reference["sha256"] = historical.bytes_sha256(data)
    monkeypatch.setattr(historical, "_verify_context_binding", lambda _view, _ref: data)
    original_loader = store.load_verified_envelope
    context = SimpleNamespace(payload=SimpleNamespace(context_config_json=json.dumps({
        "atr_scale_period": 14, "require_complete_time_bars": True,
    })))

    def loader(root, name, artifact_id, envelope):
        if name == "research_context_companions":
            return context
        return original_loader(root, name, artifact_id, envelope)

    monkeypatch.setattr(store, "load_verified_envelope", loader)
    return view, bars, reference


def _independent_rolling(bars):
    complete = bars.loc[bars.is_complete].sort_values("availability_ts_utc")
    previous = complete.close_ticks.shift(1)
    ranges = pd.concat([complete.high_ticks - complete.low_ticks,
                        (complete.high_ticks - previous).abs(),
                        (complete.low_ticks - previous).abs()], axis=1).max(axis=1)
    return pd.Series(ranges.rolling(14, min_periods=14).mean().to_numpy(), index=complete.bar_id)


def test_actual_core_helper_matches_independent_trailing14_across_days_and_gaps():
    first = _bars(24)
    second = _bars(24, day="2026-02-12", start="2026-02-12T06:03:00Z")
    second[["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] += 100
    second.loc[3, "is_complete"] = False
    second.loc[3, "is_partial"] = True
    bars = pd.concat([first, second], ignore_index=True)
    actual = geometry.compute_core_atr14(bars)
    expected = actual.bar_id.map(_independent_rolling(bars))
    np.testing.assert_allclose(actual.decision_atr14_ticks, expected, equal_nan=True)
    assert actual.decision_atr14_ticks.iloc[:13].isna().all()
    assert actual.decision_atr14_ticks.iloc[13] == 2
    assert actual.decision_atr14_ticks.iloc[24] > 2
    assert actual.atr_completed_bar_count.iloc[24] == 25
    assert actual.atr_missing_reason.iloc[27] == "decision_bar_incomplete"
    assert actual.atr_completed_bar_count.iloc[27] == 27
    assert actual.atr_completed_bar_count.iloc[28] == 28


def test_future_values_do_not_change_entry_scale_and_wilder_is_not_used():
    bars = _bars()
    bars.loc[20, ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] = [200, 202, 200, 201]
    actual = geometry.compute_core_atr14(bars)
    assert actual.decision_atr14_ticks.iloc[20] == pytest.approx((13 * 2 + 82) / 14)
    changed = bars.copy()
    changed.loc[21:, ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] += 10000
    future = geometry.compute_core_atr14(changed)
    pd.testing.assert_series_equal(
        actual.decision_atr14_ticks[:21], future.decision_atr14_ticks[:21],
    )


def test_complete_time_acceptance_exactly_matches_core_and_bad_adapter_clock_refuses():
    bars = _bars()
    bars.loc[13, "is_partial"] = True
    # Core's exact scale predicate is TIME and is_complete; partial is not a second filter.
    assert geometry.compute_core_atr14(bars).decision_atr14_ticks.iloc[13] == 2
    bad_kind = bars.copy()
    bad_kind.loc[0, "kind"] = "tick"
    with pytest.raises(ValueError, match="non-TIME"):
        geometry.compute_core_atr14(bad_kind)
    bars.loc[0, "logical_close_ts_utc"] += pd.Timedelta(seconds=1)
    with pytest.raises(ValueError, match="authoritative availability"):
        geometry.compute_core_atr14(bars)


def test_new_bundle_ratios_reload_and_historical_reader_compatibility(monkeypatch, tmp_path):
    view, bars, reference = _inputs(monkeypatch)
    artifact, frame = geometry.materialize_geometry_features(view, bars, source_reference=reference)
    independent = _independent_rolling(bars)
    expected = independent.loc[frame.iloc[0].atr_as_of_bar_id]
    assert frame.iloc[0].decision_atr14_ticks == expected
    assert frame.iloc[0][geometry.GEOMETRY_FEATURES[0]] == pytest.approx(
        frame.iloc[0].close_through_margin_ticks / expected)
    assert frame.iloc[0][geometry.GEOMETRY_FEATURES[1]] == pytest.approx(
        frame.iloc[0].distance_to_htf_ticks / expected)
    geometry.save_geometry_features(tmp_path, artifact, frame)
    loaded, reloaded = historical.load_geometry_features(
        tmp_path, artifact.geometry_feature_artifact_id,
    )
    assert loaded == artifact
    pd.testing.assert_frame_equal(frame, reloaded)
    envelope, bundle = resolve_available_bundle_view(
        view, geometry.GEOMETRY_BUNDLE_KEY, geometry_features=frame,
        geometry_feature_artifact=artifact,
    )
    assert envelope.payload.geometry_feature_artifact_id == artifact.geometry_feature_artifact_id
    assert tuple(envelope.payload.resolved_feature_names[-3:]) == geometry.GEOMETRY_FEATURES
    assert "decision_atr14_ticks" not in bundle
    old, old_frame = historical.materialize_geometry_features(
        view, bars, source_reference=reference,
    )
    assert old.geometry_feature_artifact_id != artifact.geometry_feature_artifact_id
    historical.save_geometry_features(tmp_path, old, old_frame)
    assert geometry.load_geometry_features(tmp_path, old.geometry_feature_artifact_id)[0] == old
    with pytest.raises(ValueError, match="three frozen protocol additions"):
        resolve_available_bundle_view(view, geometry.GEOMETRY_BUNDLE_KEY,
                                      geometry_features=old_frame, geometry_feature_artifact=old)


def test_zero_scale_preserves_rows_without_fabricated_denominator(monkeypatch):
    view, bars, reference = _inputs(monkeypatch)
    bars[["open_ticks", "high_ticks", "low_ticks", "close_ticks"]] = 100
    for component in ("open", "high", "low", "close"):
        view.frame[f"geometry_entry_bar_{component}_ticks"] = 100
    data = bars.to_parquet(index=False)
    reference["sha256"] = historical.bytes_sha256(data)
    monkeypatch.setattr(historical, "_verify_context_binding", lambda _view, _ref: data)
    _artifact, frame = geometry.materialize_geometry_features(view, bars,
                                                             source_reference=reference)
    assert len(frame) == 1 and frame.iloc[0].atr_missing_reason == "zero_atr_denominator"
    assert frame[list(geometry.GEOMETRY_FEATURES[:2])].isna().all().all()
    assert frame[geometry.GEOMETRY_FEATURES[2]].notna().all()
    bad = bars.copy()
    bad.loc[0, "kind"] = "tick"
    with pytest.raises(ValueError, match="non-TIME"):
        geometry.materialize_geometry_features(replace(view), bad, source_reference=reference)
