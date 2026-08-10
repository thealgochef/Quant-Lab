"""Tests for the ``ifvg_replay_chart_v1`` store (resampler, identity, catalog).

Synthetic frames cover the pure resampler and identity/catalog contracts.
Oracle-equality tests run against the real verified pair + engine tbars and
skip cleanly on machines without the data stores.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.config import IfvgCaptureConfig
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
    REPLAY_TIMEFRAMES_SECONDS,
    ArtifactPairRef,
    ReplayChartStoreError,
    find_replay_artifact,
    replay_chart_effective_config,
    replay_chart_identity,
    resample_label_bars,
    trading_day_open_utc,
)

_REPO = Path(__file__).resolve().parents[2]
_ACTIVE_V2 = (
    _REPO
    / "data/ifvg_datasets/v2"
    / "143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7"
    / "exploration"
)
_LABEL_BARS = _ACTIVE_V2 / "label_source_1m.parquet"
_HAS_REAL_DATA = _LABEL_BARS.is_file()

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


def _pair_ref(**overrides: str) -> ArtifactPairRef:
    fields = {
        "profile_name": "ifvg_v2_doc_default_fresh_static_1r",
        "v2_dataset_id": _SHA_A,
        "v2_manifest_hash": _SHA_B,
        "v3_dataset_id": _SHA_C,
        "v3_manifest_hash": _SHA_D,
    }
    fields.update(overrides)
    return ArtifactPairRef(**fields)


def _synthetic_day(day: str, minutes: int) -> pd.DataFrame:
    """Aligned 1m bars for one trading day starting at its 18:00 ET open."""
    open_ts = trading_day_open_utc(day)
    rows = []
    for index in range(minutes):
        close = open_ts + pd.Timedelta(seconds=60 * (index + 1))
        rows.append(
            {
                "source_date": day,
                "bar_id": f"60s:{day}:{index}",
                "close_ts_utc": close,
                "open_ticks": 100 + index,
                "high_ticks": 110 + index,
                "low_ticks": 90 + index,
                "close_ticks": 105 + index,
                "volume": 10,
                "trade_count": 3,
            }
        )
    return pd.DataFrame(rows)


class TestResampler:
    def test_elapsed_window_aggregation_is_exact(self) -> None:
        bars = _synthetic_day("2026-01-07", minutes=10)
        frame = resample_label_bars(bars, 300)
        assert list(frame["bar_id"]) == ["300s:2026-01-07:0", "300s:2026-01-07:1"]
        first = frame.iloc[0]
        assert first["open_ticks"] == 100
        assert first["high_ticks"] == 114
        assert first["low_ticks"] == 90
        assert first["close_ticks"] == 109
        assert first["volume"] == 50
        assert first["trade_count"] == 15
        assert first["observed_1m_count"] == 5
        assert first["expected_1m_count"] == 5
        assert not first["is_final_partial"]

    def test_windows_follow_elapsed_time_not_bar_count(self) -> None:
        """A missing minute must NOT shift later bars into earlier windows."""
        bars = _synthetic_day("2026-01-07", minutes=10)
        bars = bars[bars["bar_id"] != "60s:2026-01-07:2"].reset_index(drop=True)
        frame = resample_label_bars(bars, 300)
        first = frame.iloc[0]
        assert first["observed_1m_count"] == 4
        # Naive index//k grouping would pull bar index 5 into window 0.
        second = frame.iloc[1]
        assert second["bar_id"] == "300s:2026-01-07:1"
        assert second["open_ticks"] == 105
        assert second["observed_1m_count"] == 5

    def test_final_partial_window_is_flagged_and_clipped(self) -> None:
        bars = _synthetic_day("2026-01-07", minutes=7)
        frame = resample_label_bars(bars, 300)
        final = frame.iloc[-1]
        assert final["is_final_partial"]
        assert final["observed_1m_count"] == 2
        assert final["logical_close_ts_utc"] == pd.to_datetime(
            bars["close_ts_utc"]
        ).max()
        assert not frame.iloc[0]["is_final_partial"]

    def test_resampler_is_deterministic(self) -> None:
        bars = _synthetic_day("2026-03-09", minutes=61)  # crosses the DST change
        once = resample_label_bars(bars, 3600)
        twice = resample_label_bars(bars, 3600)
        pd.testing.assert_frame_equal(once, twice)

    def test_rejects_unsupported_timeframe(self) -> None:
        bars = _synthetic_day("2026-01-07", minutes=5)
        with pytest.raises(ReplayChartStoreError, match="unsupported"):
            resample_label_bars(bars, 61)

    def test_rejects_misaligned_bars(self) -> None:
        bars = _synthetic_day("2026-01-07", minutes=5)
        bars.loc[2, "close_ts_utc"] = bars.loc[2, "close_ts_utc"] + pd.Timedelta(seconds=30)
        with pytest.raises(ReplayChartStoreError, match="aligned"):
            resample_label_bars(bars, 300)


@pytest.mark.skipif(not _HAS_REAL_DATA, reason="verified v2 artifact not on this machine")
class TestResamplerOracleEquality:
    """Regression net for the ratified ``trading_day_18et_elapsed_v1`` rule."""

    # Normal day / early-close day (naive index//k grouping fails) / last day.
    DAYS = ("2026-01-07", "2026-04-03", "2026-06-10")

    @pytest.fixture(scope="class")
    def label_bars(self) -> pd.DataFrame:
        return pd.read_parquet(_LABEL_BARS)

    @pytest.mark.parametrize("day", DAYS)
    def test_resample_matches_engine_tbars(self, label_bars: pd.DataFrame, day: str) -> None:
        config = IfvgCaptureConfig(data_dir=_REPO / "data/databento")
        tbars_path = config.bars_path(day)
        if not tbars_path.is_file():
            pytest.skip(f"engine tbars oracle missing for {day}")
        oracle = pd.read_parquet(tbars_path)
        day_bars = label_bars[label_bars["bar_id"].str.contains(f":{day}:")]
        for timeframe in REPLAY_TIMEFRAMES_SECONDS:
            ours = resample_label_bars(day_bars, timeframe).set_index("bar_id")
            theirs = oracle[oracle["timeframe_ticks"] == timeframe].set_index("bar_id")
            assert sorted(ours.index) == sorted(theirs.index), (day, timeframe)
            columns = ["open_ticks", "high_ticks", "low_ticks", "close_ticks"]
            aligned = theirs.loc[ours.index, columns]
            assert (aligned.to_numpy() == ours[columns].to_numpy()).all(), (day, timeframe)


class TestIdentity:
    def test_identity_changes_when_any_policy_field_changes(self) -> None:
        base = replay_chart_effective_config(_pair_ref(), label_source_sha256=_SHA_E)
        base_id = replay_chart_identity(base)
        for field, value in (
            ("resample_rule", "different_rule"),
            ("development_cutoff_utc", "2026-06-11T00:00:00Z"),
            ("candidate_range_policy", "candidate_range_v2"),
            ("stage_gating_policy", "other_policy"),
            ("partial_bar_policy", "other_partial"),
            ("bar_schema_version", 2),
            ("anchor_240m_status", "ratified"),
            ("range_pad_1m_bars", 30),
            ("source_label_table_sha256", _SHA_F),
            ("timeframes_seconds", [300]),
        ):
            changed = dict(base)
            changed[field] = value
            assert replay_chart_identity(changed) != base_id, field

    def test_identity_changes_with_pair(self) -> None:
        one = replay_chart_effective_config(_pair_ref(), label_source_sha256=_SHA_E)
        other = replay_chart_effective_config(
            _pair_ref(v2_dataset_id=_SHA_F), label_source_sha256=_SHA_E
        )
        assert replay_chart_identity(one) != replay_chart_identity(other)

    def test_corroboration_evidence_is_outside_identity(self) -> None:
        config = replay_chart_effective_config(_pair_ref(), label_source_sha256=_SHA_E)
        assert not any("corrobor" in key for key in config)
        assert not any("tbars" in key for key in config)


class TestPairRefAndCatalog:
    def test_pair_ref_requires_full_hashes(self) -> None:
        with pytest.raises(ReplayChartStoreError, match="SHA-256"):
            _pair_ref(v2_dataset_id="143b510f")

    def test_find_replay_artifact_matches_all_five_fields(self) -> None:
        pair = _pair_ref()
        catalog = {
            _SHA_E: {**pair.as_dict(), "replay_chart_manifest_payload_sha256": _SHA_F},
        }
        assert find_replay_artifact(catalog, pair) == _SHA_E

    def test_same_profile_different_pair_cannot_cross_resolve(self) -> None:
        stored = _pair_ref()
        newer = _pair_ref(v2_dataset_id=_SHA_F, v2_manifest_hash=_SHA_E)
        catalog = {
            _SHA_E: {**stored.as_dict(), "replay_chart_manifest_payload_sha256": _SHA_F},
        }
        assert find_replay_artifact(catalog, newer) is None

    def test_duplicate_pair_entries_are_refused(self) -> None:
        pair = _pair_ref()
        catalog = {
            _SHA_E: {**pair.as_dict(), "replay_chart_manifest_payload_sha256": _SHA_F},
            _SHA_F: {**pair.as_dict(), "replay_chart_manifest_payload_sha256": _SHA_E},
        }
        with pytest.raises(ReplayChartStoreError, match="duplicate"):
            find_replay_artifact(catalog, pair)


def test_v1_finder_ignores_v2_bundle_entries_for_the_same_pair() -> None:
    """Regression: after the publication gate writes the setup-aware v2 bundle
    entry, the SAME pair exists twice in the catalog. The v1 finder must
    resolve only the v1 artifact (candidate mode broke with a duplicate-pair
    error when it matched both)."""
    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
        ArtifactPairRef,
        find_replay_artifact,
        find_replay_artifact_v2,
    )

    pair = ArtifactPairRef(
        profile_name="p",
        v2_dataset_id="1" * 64,
        v2_manifest_hash="2" * 64,
        v3_dataset_id="3" * 64,
        v3_manifest_hash="4" * 64,
    )
    catalog = {
        "a" * 64: {
            **pair.as_dict(),
            "replay_chart_manifest_payload_sha256": "5" * 64,
        },
        "b" * 64: {
            **pair.as_dict(),
            "artifact_kind": "ifvg_replay_chart_v2",
            "fsm_audit_artifact_id": "6" * 64,
            "fsm_audit_manifest_hash": "7" * 64,
            "replay_chart_manifest_payload_sha256": "8" * 64,
        },
    }
    assert find_replay_artifact(catalog, pair) == "a" * 64
    assert (
        find_replay_artifact_v2(catalog, pair, fsm_audit_artifact_id="6" * 64)
        == "b" * 64
    )
