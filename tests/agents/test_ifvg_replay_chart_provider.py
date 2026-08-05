"""Golden + contract tests for the replay-chart provider.

Golden tests compare provider outputs against INDEPENDENT bare
``pd.read_parquet`` reads of the source artifacts — no provider code in the
expected side.  They run against the real verified pair and skip cleanly on
machines without the data stores.  Pure gate-ordering contracts run
everywhere.
"""

from __future__ import annotations

import dataclasses
import shutil
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.replay_chart_provider import (
    FsmStage,
    MissingEvidenceError,
    RangeTooLargeError,
    ReplayAuthorizationError,
    StageGate,
    _gate_visible,
    _GateKeys,
    bars_for_pane,
    candidate_evidence,
    chart_range,
    list_candidates,
    open_replay_context,
    resolve_selection,
)
from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
    ArtifactPairRef,
    ReplayChartStoreError,
    load_verified_replay_chart_artifact,
    trading_day_open_utc,
)

_REPO = Path(__file__).resolve().parents[2]
_V2_ID = "143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7"
_V3_ID = "09ef35d0c37ca4c2b688ec88c57b2881f02d52f11cc99c4daf0f74b5c46245c3"
_PAIR = ArtifactPairRef(
    profile_name="ifvg_v2_doc_default_fresh_static_1r",
    v2_dataset_id=_V2_ID,
    v2_manifest_hash="b089dfadf44253b7071882cc9577feeacd97e532f4fd7086fe2fcbac39e60c22",
    v3_dataset_id=_V3_ID,
    v3_manifest_hash="20e0873ebb4de06125b8de8979d9eedaf8ecc55300109d7e6bad086840d85c0a",
)
_V2_DIR = _REPO / "data/ifvg_datasets/v2" / _V2_ID / "exploration"
_M3_EXECUTED = "44d23fe5-4b99-554a-96e0-e8a6ad3fb4ce"
_HAS_REAL_DATA = (_V2_DIR / "label_source_1m.parquet").is_file()

pytestmark_real = pytest.mark.skipif(
    not _HAS_REAL_DATA, reason="verified pair not on this machine"
)


def _gate(ts: str, cursor: str | None = None, ordinal: int | None = None) -> StageGate:
    return StageGate(
        stage=FsmStage.ENTRY,
        ts_utc=pd.Timestamp(ts),
        event_cursor=cursor,
        trace_ordinal=ordinal,
        source_event_id="test",
        source_kind="test",
    )


class TestGateOrderingContract:
    """stage_gate_ordinal_cursor_ts_v1: ordinal, then cursor, then timestamp."""

    def test_trace_ordinal_wins_over_later_timestamp(self) -> None:
        gate = _gate("2026-01-07T10:00:00Z", ordinal=10)
        keys = _GateKeys(ts=pd.Timestamp("2026-01-07T11:00:00Z"), cursor=None, ordinal=9)
        assert _gate_visible(gate, keys) is True

    def test_cursor_orders_same_minute_events(self) -> None:
        gate = _gate("2026-01-07T10:00:00Z", cursor="2026-01-07T10:00:00+00:00|60|x")
        earlier = _GateKeys(
            ts=pd.Timestamp("2026-01-07T10:00:00Z"),
            cursor="2026-01-07T09:59:00+00:00|60|x",
            ordinal=None,
        )
        later = _GateKeys(
            ts=pd.Timestamp("2026-01-07T10:00:00Z"),
            cursor="2026-01-07T10:01:00+00:00|60|x",
            ordinal=None,
        )
        assert _gate_visible(gate, earlier) is True
        assert _gate_visible(gate, later) is False

    def test_pure_timestamp_tie_is_indeterminate(self) -> None:
        gate = _gate("2026-01-07T10:00:00Z")
        tie = _GateKeys(ts=pd.Timestamp("2026-01-07T10:00:00Z"), cursor=None, ordinal=None)
        assert _gate_visible(gate, tie) is None

    def test_strict_timestamp_ordering_still_resolves(self) -> None:
        gate = _gate("2026-01-07T10:00:00Z")
        before = _GateKeys(ts=pd.Timestamp("2026-01-07T09:59:00Z"), cursor=None, ordinal=None)
        after = _GateKeys(ts=pd.Timestamp("2026-01-07T10:01:00Z"), cursor=None, ordinal=None)
        assert _gate_visible(gate, before) is True
        assert _gate_visible(gate, after) is False

    def test_missing_keys_are_indeterminate(self) -> None:
        gate = _gate("2026-01-07T10:00:00Z")
        assert _gate_visible(gate, _GateKeys(ts=None, cursor=None, ordinal=None)) is None


@pytestmark_real
class TestProviderGolden:
    @pytest.fixture(scope="class")
    def ctx(self):
        return open_replay_context(_REPO, _PAIR)

    @pytest.fixture(scope="class")
    def candidates(self, ctx) -> pd.DataFrame:
        return list_candidates(ctx)

    @pytest.fixture(scope="class")
    def raw(self) -> dict[str, pd.DataFrame]:
        return {
            name: pd.read_parquet(_V2_DIR / f"{name}.parquet")
            for name in (
                "label_source_1m",
                "geometry_dossier",
                "setup_lifecycle_event",
                "executed_trade",
                "entry_candidate",
            )
        }

    def _pinned(self, candidates: pd.DataFrame) -> dict[str, str]:
        executed = candidates[candidates["executed"] & ~candidates["is_warmup"]]
        wins = executed[executed["resolution"] == "target"].sort_values("trade_id")
        losses = executed[executed["resolution"] == "stop"].sort_values("trade_id")
        blocked = candidates[candidates["blocked"] & ~candidates["executed"]]
        return {
            "executed_win": wins.iloc[0]["candidate_id"],
            "executed_loss": losses.iloc[0]["candidate_id"],
            "blocked": blocked.iloc[0]["candidate_id"],
            "m3_executed": _M3_EXECUTED,
        }

    def test_cohort_shape(self, candidates: pd.DataFrame) -> None:
        assert len(candidates) == 132
        assert int(candidates["is_warmup"].sum()) == 7
        assert int(candidates["executed"].sum()) == 33
        assert int((candidates["executed"] & ~candidates["is_warmup"]).sum()) == 30
        assert set(candidates.loc[candidates["m3_qualifying"], "candidate_id"]) >= {
            _M3_EXECUTED
        }

    def test_every_candidate_resolves_by_exact_id(self, ctx, candidates) -> None:
        for candidate_id in candidates["candidate_id"]:
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            assert evidence.candidate_id == candidate_id
            # Retest-family candidates carry no fresh entry-FVG geometry block.
            roles = {zone.role for zone in evidence.zones}
            assert {"htf", "parent", "opposing"} <= roles
            assert len(evidence.zones) in (3, 4)

    def test_zone_bounds_match_dossier(self, ctx, candidates, raw) -> None:
        dossiers = raw["geometry_dossier"].set_index("candidate_id")
        for candidate_id in self._pinned(candidates).values():
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            row = dossiers.loc[candidate_id]
            for zone in evidence.zones:
                prefix = f"geometry_{zone.role}_"
                assert zone.gap_low_ticks == int(row[f"{prefix}gap_low_ticks"])
                assert zone.gap_high_ticks == int(row[f"{prefix}gap_high_ticks"])
                assert zone.confirmed_ts_utc == pd.Timestamp(row[f"{prefix}confirmed_ts_utc"])
                assert zone.a_bar_id == str(row[f"{prefix}a_bar_id"])

    def test_lifecycle_matches_source_events(self, ctx, candidates, raw) -> None:
        lifecycle = raw["setup_lifecycle_event"]
        pinned = self._pinned(candidates)
        evidence = candidate_evidence(ctx, pinned["executed_win"], mode="full_audit")
        setup_id = evidence.lineage["setup_id"]
        expected = lifecycle[lifecycle["setup_id"].astype(str) == setup_id].sort_values(
            "trace_ordinal"
        )
        assert len(evidence.lifecycle) == len(expected)
        assert list(evidence.lifecycle["transition"]) == list(expected["transition"])
        assert list(evidence.lifecycle["trace_ordinal"]) == list(expected["trace_ordinal"])
        assert list(pd.to_datetime(evidence.lifecycle["envelope_ts_utc"], utc=True)) == list(
            pd.to_datetime(expected["envelope_ts_utc"], utc=True)
        )

    def test_execution_matches_executed_trade_row(self, ctx, candidates, raw) -> None:
        trades = raw["executed_trade"].set_index("candidate_id")
        for key in ("executed_win", "executed_loss", "m3_executed"):
            candidate_id = self._pinned(candidates)[key]
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            row = trades.loc[candidate_id]
            assert evidence.execution is not None
            assert evidence.execution["entry_ticks"] == int(row["entry_ticks"])
            assert evidence.execution["stop_ticks"] == int(row["stop_ticks"])
            assert evidence.execution["target_ticks"] == int(row["target_ticks"])
            assert evidence.execution["resolution"] == str(row["resolution"])
            assert evidence.execution["resolution_ts_utc"] == pd.Timestamp(
                row["resolution_ts_utc"]
            )
            assert evidence.execution["realized_r"] == float(row["realized_r"])

    def test_candle_ohlc_matches_label_source(self, ctx, candidates, raw) -> None:
        labels = raw["label_source_1m"].set_index("bar_id")
        pinned = self._pinned(candidates)
        evidence = candidate_evidence(ctx, pinned["executed_win"], mode="full_audit")
        start, end = chart_range(ctx, evidence, "setup")
        bars = bars_for_pane(ctx, timeframe_seconds=60, start_ts=start, end_ts=end)
        assert len(bars) > 0
        for row in bars.head(4).to_dict("records"):
            expected = labels.loc[row["bar_id"]]
            for column in ("open_ticks", "high_ticks", "low_ticks", "close_ticks"):
                assert row[column] == expected[column], (row["bar_id"], column)

    def test_replay_bar_matches_independent_aggregation(self, ctx, raw) -> None:
        labels = raw["label_source_1m"]
        day = "2026-01-07"
        day_open = trading_day_open_utc(day)
        day_bars = labels[labels["bar_id"].astype(str).str.contains(f":{day}:")].copy()
        day_bars["close_ts_utc"] = pd.to_datetime(day_bars["close_ts_utc"], utc=True)
        for timeframe in (300, 14400):
            window = (
                (day_bars["close_ts_utc"] - pd.Timedelta(seconds=60) - day_open)
                .dt.total_seconds()
                .floordiv(timeframe)
                .astype(int)
            )
            source = day_bars[window == 1].sort_values("close_ts_utc")
            pane = bars_for_pane(
                ctx,
                timeframe_seconds=timeframe,
                start_ts=day_open + pd.Timedelta(seconds=timeframe),
                end_ts=day_open + pd.Timedelta(seconds=2 * timeframe - 1),
            )
            target = pane[pane["bar_id"] == f"{timeframe}s:{day}:1"].iloc[0]
            assert target["open_ticks"] == source.iloc[0]["open_ticks"]
            assert target["high_ticks"] == source["high_ticks"].max()
            assert target["low_ticks"] == source["low_ticks"].min()
            assert target["close_ticks"] == source.iloc[-1]["close_ticks"]

    def test_model_values_match_run_predictions(self, ctx, candidates) -> None:
        available = None
        for candidate_id in candidates["candidate_id"]:
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            hits = {
                tier: payload
                for tier, payload in evidence.model.items()
                if payload.get("status") == "prediction_available"
            }
            if hits:
                available = (candidate_id, hits)
                break
        assert available is not None, "no candidate has any OOS prediction"
        candidate_id, hits = available
        for tier, payload in hits.items():
            run = ctx.tier_runs[tier]
            expected = run.predictions[
                run.predictions["candidate_id"].astype(str) == candidate_id
            ].iloc[0]
            assert payload["probability"] == float(expected["probability"])
            assert payload["fold_index"] == int(expected["fold_index"])
            assert payload["note"].startswith("counterfactual development evidence")


@pytestmark_real
class TestProviderContracts:
    @pytest.fixture(scope="class")
    def ctx(self):
        return open_replay_context(_REPO, _PAIR)

    @pytest.fixture(scope="class")
    def candidates(self, ctx) -> pd.DataFrame:
        return list_candidates(ctx)

    @pytest.fixture(scope="class")
    def executed_id(self, candidates) -> str:
        return candidates[candidates["executed"] & ~candidates["is_warmup"]].iloc[0][
            "candidate_id"
        ]

    def test_chart_opens_with_zero_model_runs(self, ctx, executed_id) -> None:
        bare = dataclasses.replace(ctx, tier_runs={tier: None for tier in ctx.tier_runs})
        evidence = candidate_evidence(bare, executed_id, mode="full_audit")
        assert len(evidence.zones) == 4
        assert all(
            payload["status"] == "run_not_available" for payload in evidence.model.values()
        )

    def test_chart_opens_with_partial_tiers(self, ctx, executed_id) -> None:
        partial = dataclasses.replace(
            ctx,
            tier_runs={
                tier: (run if tier == "M0" else None) for tier, run in ctx.tier_runs.items()
            },
        )
        evidence = candidate_evidence(partial, executed_id, mode="full_audit")
        statuses = {tier: payload["status"] for tier, payload in evidence.model.items()}
        assert statuses["M1_PRIMARY"] == "run_not_available"
        assert statuses["M0"] != "run_not_available"

    def test_selection_requires_exactly_one_exact_id(self, ctx, executed_id) -> None:
        with pytest.raises(MissingEvidenceError):
            resolve_selection(ctx)
        with pytest.raises(MissingEvidenceError):
            resolve_selection(ctx, candidate_id=executed_id, trade_id="x")
        with pytest.raises(MissingEvidenceError):
            resolve_selection(ctx, candidate_id="not-a-candidate")

    def test_outcome_hidden_before_resolution(self, ctx, executed_id) -> None:
        at_entry = candidate_evidence(
            ctx, executed_id, mode="point_in_time", stage=FsmStage.ENTRY
        )
        assert at_entry.execution is not None
        assert "resolution" not in at_entry.execution
        assert "mfe_ticks" not in at_entry.execution
        assert "realized_r" not in at_entry.execution
        assert at_entry.counterfactual_labels == ()
        at_resolution = candidate_evidence(
            ctx, executed_id, mode="point_in_time", stage=FsmStage.RESOLUTION
        )
        assert "resolution" in at_resolution.execution
        assert len(at_resolution.counterfactual_labels) == 3

    def test_model_no_earlier_than_entry(self, ctx, candidates) -> None:
        with_prediction = None
        for candidate_id in candidates["candidate_id"]:
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            if any(
                payload.get("status") == "prediction_available"
                for payload in evidence.model.values()
            ):
                with_prediction = candidate_id
                break
        assert with_prediction is not None
        at_lock = candidate_evidence(
            ctx, with_prediction, mode="point_in_time", stage=FsmStage.LOCK
        )
        assert all(
            payload["status"] in ("hidden_until_entry", "candidate_not_in_oos_folds")
            for payload in at_lock.model.values()
        )
        at_entry = candidate_evidence(
            ctx, with_prediction, mode="point_in_time", stage=FsmStage.ENTRY
        )
        assert any(
            payload["status"] == "prediction_available" for payload in at_entry.model.values()
        )

    def test_pit_lock_hides_later_stage_evidence(self, ctx, executed_id) -> None:
        evidence = candidate_evidence(
            ctx, executed_id, mode="point_in_time", stage=FsmStage.LOCK
        )
        gate = evidence.stage_gates["lock"]
        assert not any(zone.role in ("opposing", "entry_fvg") for zone in evidence.zones)
        assert (evidence.lifecycle["trace_ordinal"].astype(int) <= gate.trace_ordinal).all()
        assert "inversion" not in evidence.transition_bars
        assert set(evidence.structure_stage_summary.get("stage", pd.Series())) <= {
            "tap",
            "lock",
        }
        for row in evidence.pool_members.to_dict("records"):
            assert pd.Timestamp(row["swing_confirmation_ts"]) <= gate.ts_utc

    def test_executed_and_counterfactual_paths_stay_separate(self, ctx, candidates) -> None:
        blocked = candidates[candidates["blocked"] & ~candidates["executed"]].iloc[0][
            "candidate_id"
        ]
        evidence = candidate_evidence(ctx, blocked, mode="full_audit")
        assert evidence.execution is None
        assert len(evidence.counterfactual_labels) == 3
        assert evidence.range_row["display_end_source"] != "trade_resolution"

    def test_censored_candidate_ends_at_authorized_cutoff(self, ctx, candidates) -> None:
        censored = candidates[candidates["censored"] & ~candidates["executed"]]
        if censored.empty:
            pytest.skip("no censored non-executed candidate in this pair")
        evidence = candidate_evidence(
            ctx, censored.iloc[0]["candidate_id"], mode="full_audit"
        )
        row = evidence.range_row
        if row["display_end_source"] == "authorized_cutoff_censor":
            assert pd.Timestamp(row["display_end_ts"]) == pd.Timestamp(
                "2026-06-10T21:00:00Z"
            )

    def test_range_too_large_fails_gracefully(self, ctx) -> None:
        with pytest.raises(RangeTooLargeError, match="narrow"):
            bars_for_pane(
                ctx,
                timeframe_seconds=60,
                start_ts=pd.Timestamp("2026-01-05T00:00:00Z"),
                end_ts=pd.Timestamp("2026-03-01T00:00:00Z"),
            )

    def test_protected_and_sealed_ranges_are_refused(self, ctx) -> None:
        for start, end in (
            ("2026-06-11T00:00:00Z", "2026-06-11T23:00:00Z"),
            ("2026-06-12T00:00:00Z", "2026-06-13T00:00:00Z"),
            ("2026-06-10T00:00:00Z", "2026-06-11T04:00:00Z"),
        ):
            with pytest.raises(ReplayAuthorizationError):
                bars_for_pane(
                    ctx,
                    timeframe_seconds=60,
                    start_ts=pd.Timestamp(start),
                    end_ts=pd.Timestamp(end),
                )

    def test_tampered_replay_artifact_is_refused(self, ctx, tmp_path) -> None:
        source = ctx.replay.directory
        copy_root = tmp_path / "store"
        copy_dir = copy_root / ctx.replay.artifact_id
        shutil.copytree(source, copy_dir)
        bars_path = copy_dir / "bars_tf.parquet"
        payload = bytearray(bars_path.read_bytes())
        payload[len(payload) // 2] ^= 0xFF
        bars_path.write_bytes(bytes(payload))
        with pytest.raises(ReplayChartStoreError):
            load_verified_replay_chart_artifact(
                copy_root, ctx.replay.artifact_id, expected_pair=ctx.pair_ref
            )

    def test_pit_requires_a_stage_and_gates_exist(self, ctx, executed_id) -> None:
        with pytest.raises(MissingEvidenceError):
            candidate_evidence(ctx, executed_id, mode="point_in_time")
        evidence = candidate_evidence(ctx, executed_id, mode="full_audit")
        assert set(evidence.stage_gates) == {stage.value for stage in FsmStage}

    def test_all_executed_render_entry_stop_target_resolution(self, ctx, candidates) -> None:
        executed = candidates[candidates["executed"] & ~candidates["is_warmup"]]
        assert len(executed) == 30
        for candidate_id in executed["candidate_id"]:
            evidence = candidate_evidence(ctx, candidate_id, mode="full_audit")
            assert evidence.execution is not None
            for key in ("entry_ticks", "stop_ticks", "target_ticks", "resolution_ts_utc"):
                assert evidence.execution.get(key) is not None
