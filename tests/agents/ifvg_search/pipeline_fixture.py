"""Shared synthetic pipeline fixture (R5): one frozen 2×2 charter + a fully
wired 16-stage verification-scope pipeline over the R2 synthetic replay
machinery, the real feature/label/fold/ladder chain (M0 view; the frozen
40/5/5/2 protocol legitimately yields zero valid folds on three days), and
the REAL propsim bridge with the synthetic fixture firm.

Used by the pipeline E2E, the job-shim contract tests, and the AppTests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.agents.data_infra.ifvg.context_experiment_contracts import (
    ContextFeatureTier,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.context_feature_view import (
    M0_FEATURES,
    CandidateFeatureView,
)
from alpha_lab.agents.data_infra.ifvg.data_access import allowlist_sha256
from alpha_lab.agents.data_infra.ifvg.search.charter import SimulationProtocol
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
    FOLD_PROTOCOL_ID_V1,
    PipelineSemanticIdentity,
    PipelineSemanticSpecPayload,
    PipelineWiring,
    QuantLabPipelineStage,
    WorkerPolicy,
)
from alpha_lab.propsim.search_bridge import FirmSimulationSpec
from tests.agents.ifvg_search.conftest import SYNTHETIC_DAYS
from tests.agents.ifvg_search.test_orchestrator import (
    _charter,
    _identity_resolver,
)

LABEL_POLICY_ID = "synthetic_fixture_labels_v1"

FULL_STAGE_PLAN: tuple[QuantLabPipelineStage, ...] = tuple(QuantLabPipelineStage)

STRATEGY_ONLY_STAGE_PLAN: tuple[QuantLabPipelineStage, ...] = (
    QuantLabPipelineStage.S00_VALIDATE_INPUTS,
    QuantLabPipelineStage.S01_PREPARE_STRATEGY_PROFILES,
    QuantLabPipelineStage.S02_RUN_OR_REUSE_SEQUENTIAL_REPLAYS,
    QuantLabPipelineStage.S03_BUILD_OR_REUSE_FSM_AUDIT,
    QuantLabPipelineStage.S04_BUILD_OR_REUSE_REPLAY_CHARTS,
    QuantLabPipelineStage.S12_RUN_PROP_HISTORICAL_REPLAYS,
    QuantLabPipelineStage.S13_RUN_BOOTSTRAP_AND_STRESS,
    QuantLabPipelineStage.S14_BUILD_FRONTIER_AND_INSIGHTS,
    QuantLabPipelineStage.S15_VERIFY_AND_PUBLISH,
)

_CATEGORICAL_FILL = {
    "direction": ("long", "short"),
    "entry_family": ("fvg_retest", "inversion_retest"),
    "entry_session": ("ny", "asia", "london"),
    "in_engine_session": ("true", "false"),
    "in_doc_session": ("true", "false"),
}


def build_mini_view(days: tuple[str, ...] = SYNTHETIC_DAYS, *, per_day: int = 8):
    """A tiny M0-complete candidate view + labels over the synthetic days."""

    rng = np.random.default_rng(7)
    frame_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    index = 0
    for day in days:
        for _ in range(per_day):
            candidate_id = f"pcand_{index:04d}"
            setup_id = f"psetup_{index // 2:04d}"
            entry_ts = f"{day}T14:{index % 60:02d}:00Z"
            row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "setup_id": setup_id,
                "trading_day": day,
                "entry_ts_utc": entry_ts,
                "feature_as_of_ts": entry_ts,
                "context_as_of_ts": entry_ts,
                "context_capture_id": "synthetic_capture_v1",
                "context_state_id": f"pstate_{index:04d}",
                "geometry_evidence_id": f"pgeom_{index:04d}",
                "geometry_evidence_cursor": index,
                "is_warmup": False,
            }
            for feature in M0_FEATURES:
                if feature in _CATEGORICAL_FILL:
                    choices = _CATEGORICAL_FILL[feature]
                    row[feature] = choices[index % len(choices)]
                else:
                    row[feature] = float(rng.normal(10.0, 2.0))
            frame_rows.append(row)
            target = index % 2
            label_rows.append(
                {
                    "candidate_id": candidate_id,
                    "setup_id": setup_id,
                    "trading_day": day,
                    "entry_ts_utc": entry_ts,
                    "resolution_ts_utc": f"{day}T15:{index % 60:02d}:00Z",
                    "entry_available": True,
                    "resolution_available": True,
                    "binary_target": target,
                    "gross_r": 1.0 if target else -1.0,
                    "net_r": 1.0 if target else -1.0,
                }
            )
            index += 1
    frame = pd.DataFrame(frame_rows)
    labels = pd.DataFrame(label_rows)
    tier = ContextFeatureTier.M0
    feature_registry_hash = canonical_contract_sha256({tier.value: list(M0_FEATURES)})
    view = CandidateFeatureView(
        view_id=canonical_contract_sha256(
            {"fixture": "pipeline_mini_view", "days": list(days), "per_day": per_day}
        ),
        artifact_pair_hash=canonical_contract_sha256({"pipeline_pair": len(days)}),
        feature_registry_hash=feature_registry_hash,
        frame=frame,
        tier_features={tier: M0_FEATURES},
        m3_status="synthetic_fixture",
    )
    return view, labels


def firm_simulation_specs() -> tuple[FirmSimulationSpec, ...]:
    from alpha_lab.agents.data_infra.ifvg.search.executors import (
        synthetic_firm_specs,
    )

    return synthetic_firm_specs()


def _companion_builder(kind: str):
    def _builder(row, result) -> tuple[str, ...]:
        return (
            canonical_contract_sha256(
                {"companion": kind, "core_replay_id": row["core_replay_id"]}
            ),
        )

    return _builder


def build_mbp1_evidence(view) -> tuple[Any, dict[str, Any], Any]:
    """Synthetic MBP-1 evidence for the mini view (R5B pipeline path).

    Per synthetic day: a clean 30-second event stream from 13:45:01Z to
    15:00:01Z (offset one second from the minute anchors, so no boundary
    tie exists) plus exact per-candidate stage anchors derived from each
    candidate's entry timestamp.
    """

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
        build_mbp1_source_artifact,
        normalize_mbp1_events,
    )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
        MIN_DAY_COVERAGE_FRACTION,
        R5B_WINDOW_SPECS,
        Mbp1SourceContract,
    )
    from tests.agents.ifvg_search.mbp1_fixture import raw_event

    events_by_day: dict[str, Any] = {}
    days = tuple(sorted(set(view.frame["trading_day"].astype(str))))
    for day in days:
        base = int(pd.Timestamp(f"{day}T13:45:01Z").value)
        rows = [
            raw_event(
                ts_event=base + i * 30 * 1_000_000_000,
                sequence=1000 + i,
                bid_ticks=20000 + (i % 3),
                ask_ticks=20001 + (i % 3),
                bid_sz=4 + (i % 4),
                ask_sz=6 - (i % 3),
                action="T" if i % 5 == 0 else "A",
                side="B" if i % 10 == 0 else ("A" if i % 5 == 0 else "N"),
                size=1 + (i % 2),
            )
            for i in range(151)  # 13:45:01 … 15:00:01
        ]
        events_by_day[day] = normalize_mbp1_events(
            pd.DataFrame(rows), instrument="NQ", trading_day=day
        )
    anchor_rows = []
    for _, row in view.frame.iterrows():
        entry = pd.Timestamp(row["entry_ts_utc"])
        anchor_rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "setup_id": str(row["setup_id"]),
                "trading_day": str(row["trading_day"]),
                "tap_ts_utc": (entry - pd.Timedelta(minutes=8)).isoformat(),
                "lock_ts_utc": (entry - pd.Timedelta(minutes=6)).isoformat(),
                "armed_ts_utc": (entry - pd.Timedelta(minutes=4)).isoformat(),
                "inversion_ts_utc": (entry - pd.Timedelta(minutes=2)).isoformat(),
                "entry_ts_utc": entry.isoformat(),
            }
        )
    anchors = pd.DataFrame(anchor_rows)
    contract = Mbp1SourceContract(
        instrument="NQ",
        contract_roll_policy_id="front_month_open_interest_roll_v1",
        feature_window_specs=R5B_WINDOW_SPECS,
        coverage_policy={"min_day_coverage_fraction": MIN_DAY_COVERAGE_FRACTION},
    )
    from tests.agents.ifvg_search.mbp1_fixture import default_partition_evidence

    source_envelope, _event_bytes = build_mbp1_source_artifact(
        events_by_day,
        contract=contract,
        authorized_date_set_id="synthetic_fixture_days_v1",
        events_stored=True,
        # R5B.1: coverage policy v2 — synthetic partition-scope evidence
        # (positive completeness compiled under synthetic provenance; lawful
        # only under the synthetic marker, which this fixture carries)
        coverage_evidence=default_partition_evidence(events_by_day),
    )
    return source_envelope, events_by_day, anchors


REGIME_STUDY_SHAPES: tuple[str, ...] = (
    "candidate",
    "panel",
    "candidate_supervised",
    "panel_supervised",
)


def build_regime_study_request(
    shape: str,
    *,
    authority: tuple[str, str, str] | None = None,
    bootstrap_refits: int = 3,
    hard_id_encoding: str = "none",
):
    """The frozen regime-study request of one fixture shape (R6.1 §6.E)."""

    from alpha_lab.agents.data_infra.ifvg.features.context_bar_panel_contract import (
        PANEL_AS_OF_POLICY_ID_V1,
    )
    from alpha_lab.agents.data_infra.ifvg.ml.regime_study import RegimeStudyRequest
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
        REGIME_INPUT_FEATURES,
    )
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_observation_source import (
        PANEL_NUMERIC_FEATURES,
    )

    if shape not in REGIME_STUDY_SHAPES:
        raise ValueError(f"unknown regime study shape {shape!r}")
    supervised = shape.endswith("_supervised")
    panel = shape.startswith("panel")
    classes: tuple[str, ...] = ("cohort_descriptive", "stratified_prop", "stratified_frontier")
    if supervised:
        classes = (*classes, "feature_only", "cohort_model")
    grain_fields: dict[str, Any] = (
        {
            "observation_granularity": "context_bar_panel",
            "panel_interval_seconds": 300,
            "panel_as_of_policy_id": PANEL_AS_OF_POLICY_ID_V1,
            "input_feature_bundle_key": "BP0_CONTEXT_BAR_PANEL",
            "resolved_input_features": PANEL_NUMERIC_FEATURES,
        }
        if panel
        else {
            "input_feature_bundle_key": "B0_CORE",
            "resolved_input_features": REGIME_INPUT_FEATURES,
        }
    )
    authority_fields: dict[str, Any] = {}
    if supervised:
        if authority is None:
            raise ValueError("a supervised regime study shape requires the frozen authority ids")
        decision_id, owner_id, assessment_id = authority
        authority_fields = {
            "supervised_bundle_key": "B0_CORE",
            "regime_promotion_decision_id": decision_id,
            "owner_decision_artifact_id": owner_id,
            "required_capability_assessment_id": assessment_id,
        }
    encoding_fields: dict[str, Any] = (
        {"hard_id_encoding": hard_id_encoding} if hard_id_encoding != "none" else {}
    )
    return RegimeStudyRequest(
        **grain_fields,
        bootstrap_refits=bootstrap_refits,
        stratified_reporting_requested=True,
        comparison_classes_requested=classes,
        **encoding_fields,
        **authority_fields,
    )


def build_pipeline_fixture(
    tmp_root: Path,
    *,
    stage_plan: tuple[QuantLabPipelineStage, ...] = FULL_STAGE_PLAN,
    bootstrap_n_paths: int = 32,
    label_builder=None,
    mbp1: bool = False,
    regime_study: str | None = None,
    regime_days: int = 55,
    regime_authority: tuple[str, str, str] | None = None,
    regime_bootstrap_refits: int = 3,
    regime_hard_id_encoding: str = "none",
) -> dict[str, Any]:
    """Everything one `run_pipeline` invocation needs, on temporary roots.

    ``mbp1=True`` pins the B2 order-flow bundle + the logistic protocol and
    wires the synthetic MBP-1 evidence seam — the R5B controlled-study
    pipeline path.

    ``regime_study`` (R6.1) selects one of ``REGIME_STUDY_SHAPES``: the
    candidate view becomes the 55-day known-cluster fixture (fixture 2) under
    the three-day synthetic allowlist (a recorded fixture-shape deviation —
    verification scope + synthetic marker), the panel shapes add the
    ``BP0_CONTEXT_BAR_PANEL`` bundle and wire a synthetic VERIFIED
    replay-chart artifact as the ``context_bar_source``, and the supervised
    shapes pin the logistic protocol and freeze ``regime_authority`` =
    (promotion decision id, owner artifact id, assessment id).
    """

    tmp_root = Path(tmp_root)
    if regime_study is not None and mbp1:
        raise ValueError("the regime fixture shapes do not combine with the MBP-1 shape")
    regime_request = (
        build_regime_study_request(
            regime_study,
            authority=regime_authority,
            bootstrap_refits=regime_bootstrap_refits,
            hard_id_encoding=regime_hard_id_encoding,
        )
        if regime_study is not None
        else None
    )
    simulation_protocol = SimulationProtocol(
        modes=("historical_closed_trade", "day_block_bootstrap"),
        bootstrap_n_paths=bootstrap_n_paths,
        bootstrap_seed=42,
        stress_scenario_ids=(),
        trade_path_capability_policy_id="path_capability_policy_v1",
        clock_policy_id="simulated_clock_v1",
    )
    charter = _charter(
        simulation_protocol=simulation_protocol,
        pareto_objectives=("net_expectancy_r", "expected_net_payout_90d"),
    )
    specs = firm_simulation_specs()
    policy_ids = tuple(
        sorted(spec.policy_set_envelope().account_policy_set_id for spec in specs)
    )
    payload = PipelineSemanticSpecPayload(
        run_scope="verification_5d",
        date_allowlist=SYNTHETIC_DAYS,
        allowlist_hash=allowlist_sha256(SYNTHETIC_DAYS),
        warmup_policy_id="zero_real_warmup_seed_snapshot_v1",
        search_charter_id=charter.search_id,
        source_artifact_ids=(),
        feature_bundle_ids=_fixture_bundle_ids(mbp1=mbp1, regime_request=regime_request),
        label_policy_id=LABEL_POLICY_ID,
        fold_protocol_id=FOLD_PROTOCOL_ID_V1,
        model_protocol_id=_fixture_model_protocol(mbp1=mbp1, regime_request=regime_request),
        cost_policy_sha256=canonical_contract_sha256(charter.payload.cost_policy),
        account_policy_set_ids=policy_ids,
        portfolio_policy_ids=(),
        simulation_protocol=simulation_protocol,
        software_commits={"quant_lab": "b" * 40, "strategy_core": "a" * 40},
        stage_plan=stage_plan,
        regime_study=regime_request,
    )
    semantic = PipelineSemanticIdentity.from_payload(payload)
    regime_fixture = None
    context_bar_source = None
    chart_builder = _companion_builder("replay_chart")
    if regime_request is not None:
        from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_clusters import (
            known_cluster_fixture,
        )

        regime_fixture = known_cluster_fixture(k=3, n=600)
        assert len(regime_fixture.trading_days) == regime_days
        view, labels = regime_fixture.view, regime_fixture.labeled_candidates
        if regime_request.is_panel:
            chart_builder, context_bar_source = _synthetic_replay_chart_seam(
                tmp_root / "replay_chart", regime_fixture.trading_days
            )
    else:
        view, labels = build_mini_view()

    def _default_label_builder(_view) -> tuple[str, pd.DataFrame]:
        return LABEL_POLICY_ID, labels.copy()

    observed: list = []

    def _runner(*, spec, core_replay_id):
        # every child passes the (relaxed) strategy gates in this fixture
        from alpha_lab.agents.data_infra.ifvg.contracts import (  # noqa: PLC0415
            RecordTable,
        )
        from tests.agents.ifvg_search.conftest import (  # noqa: PLC0415
            make_resolved_trades_frame,
        )

        observed.append((spec.ordinal, core_replay_id))
        return SimpleNamespace(
            tables={
                RecordTable.EXECUTED_TRADE: make_resolved_trades_frame(SYNTHETIC_DAYS)
            },
            gross_trade_stream_hash=canonical_contract_sha256(
                {"child": core_replay_id}
            ),
        )

    mbp1_evidence = build_mbp1_evidence(view) if mbp1 else None
    wiring = PipelineWiring(
        identity_resolver=_identity_resolver,
        child_runner=_runner,
        audit_builder=_companion_builder("fsm_audit"),
        chart_builder=chart_builder,
        candidate_view_source=lambda: view,
        label_builder=label_builder or _default_label_builder,
        firm_specs=specs,
        mbp1_evidence_source=(lambda: mbp1_evidence) if mbp1 else None,
        context_bar_source=context_bar_source,
    )
    return {
        "regime_request": regime_request,
        "regime_fixture": regime_fixture,
        "charter": charter,
        "semantic": semantic,
        "wiring": wiring,
        "store_root": tmp_root / "store",
        "state_root": tmp_root / "state",
        "worker_policy": WorkerPolicy(
            max_workers=1, max_tasks_per_child=1, memory_budget_bytes=1 << 30
        ),
        "observed": observed,
        "view": view,
        "labels": labels,
        "policy_ids": policy_ids,
    }


def _fixture_bundle_ids(*, mbp1: bool, regime_request) -> tuple[str, ...]:
    if mbp1:
        return ("B2_CORE_ORDER_FLOW",)
    if regime_request is not None and regime_request.is_panel:
        return ("B0_CORE", "BP0_CONTEXT_BAR_PANEL")
    return ("B0_CORE",)


def _fixture_model_protocol(*, mbp1: bool, regime_request) -> str | None:
    if mbp1:
        return "ifvg_context_logistic_l2_v1"
    if regime_request is not None:
        # descriptive studies run S09a only; supervised shapes pin the fast
        # bundle-parametrized logistic protocol for the ladder
        return "ifvg_context_logistic_l2_v1" if regime_request.requires_supervision else None
    return "ifvg_context_catboost_binary_v1"


def _synthetic_replay_chart_seam(base_dir: Path, days: tuple[str, ...]):
    """The panel seam of the fixture (adversarial R6.1 S1): ``(chart_builder,
    context_bar_source)``. The S04 chart builder WRITES a real-shaped
    ``ifvg_replay_chart_v1`` artifact (the production resampler + identity
    functions) under ``base_dir`` and returns its content identity — one
    chart per child, all children sharing the pair's bars and therefore one
    artifact id; the source loads EXACTLY the chart it is asked for through
    the real verified loader (a request for an unknown id fails closed)."""

    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
        load_verified_replay_chart_artifact,
    )
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
        synthetic_pair_ref,
        write_synthetic_replay_chart_artifact,
    )

    pair = synthetic_pair_ref(seed="pipeline-panel")
    bars = regime_structured_bars_1m(days)
    written: list[str] = []
    requested: list[str] = []

    def _chart_builder(row, result) -> tuple[str, ...]:
        artifact_id = write_synthetic_replay_chart_artifact(base_dir, bars, pair)
        written.append(artifact_id)
        return (artifact_id,)

    def _source(chart_id: str):
        requested.append(str(chart_id))
        return load_verified_replay_chart_artifact(base_dir, str(chart_id), expected_pair=pair)

    _source.requested = requested  # type: ignore[attr-defined]
    _source.written = written  # type: ignore[attr-defined]
    _source.pair = pair  # type: ignore[attr-defined]
    _source.base_dir = base_dir  # type: ignore[attr-defined]
    return _chart_builder, _source


#: (high/low spread scale in ticks, drift ticks per minute, noise ticks per
#: minute, volume scale) of the three regimes planted PER TRADING DAY.
REGIME_DAY_SCALES: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 0.0, 0.8, 20.0),  # quiet chop: tiny ranges, no drift
    (5.0, 2.0, 5.0, 60.0),  # normal: mid ranges, mild drift
    (14.0, 8.0, 12.0, 180.0),  # wide, trending: large ranges, strong drift
)


def regime_structured_bars_1m(days: tuple[str, ...]) -> pd.DataFrame:
    """Aligned 1m bars (02:00 → 14:00 ET, the panel fixture's span) whose
    regime is planted PER TRADING DAY (``day index mod 3`` → the three
    ``REGIME_DAY_SCALES`` rows; the trend direction alternates by day). The
    12-bar panel windows never cross a day, so every window sits inside ONE
    regime and the k=3 clusters are well occupied (≈⅓ each) with a passing
    capability assessment — the shape the model-bearing PANEL study needs
    (FEATURE_ELIGIBLE is lawful only over a fully passing assessment)."""

    import numpy as np  # noqa: PLC0415

    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (  # noqa: PLC0415
        trading_day_open_utc,
    )

    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    price = 20_000.0
    for day_index, day in enumerate(days):
        spread, drift, noise, volume_scale = REGIME_DAY_SCALES[day_index % 3]
        direction = 1.0 if day_index % 2 == 0 else -1.0
        day_open = trading_day_open_utc(day)
        for minute in range(8 * 60, 20 * 60):
            step = float(rng.normal(direction * drift, noise))
            open_ticks = price
            close_ticks = price + step
            high_ticks = max(open_ticks, close_ticks) + abs(rng.normal(0.0, spread))
            low_ticks = min(open_ticks, close_ticks) - abs(rng.normal(0.0, spread))
            price = close_ticks
            close_ts = day_open + pd.Timedelta(minutes=minute + 1)
            rows.append(
                {
                    "source_date": day,
                    "bar_id": f"60s:{day}:{minute}",
                    "close_ts_utc": close_ts.isoformat(),
                    "open_ticks": int(round(open_ticks)),
                    "high_ticks": int(round(high_ticks)),
                    "low_ticks": int(round(low_ticks)),
                    "close_ticks": int(round(close_ticks)),
                    "volume": int(max(1, round(rng.normal(volume_scale, volume_scale * 0.15)))),
                    "trade_count": int(max(1, round(volume_scale * 0.4))),
                }
            )
    return pd.DataFrame(rows)


def foreign_context_bar_source(base_dir: Path, days: tuple[str, ...]):
    """A seam that IGNORES the requested chart id and serves another verified
    artifact (a different synthetic pair) — the S1 probe shape; S05 must
    refuse it."""

    from alpha_lab.agents.data_infra.ifvg.replay_chart_store import (
        load_verified_replay_chart_artifact,
    )
    from tests.agents.data_infra.ifvg.ml_fixtures.synthetic_context_panel import (
        synthetic_label_source_1m,
        synthetic_pair_ref,
        write_synthetic_replay_chart_artifact,
    )

    pair = synthetic_pair_ref(seed="foreign-pair")
    artifact_id = write_synthetic_replay_chart_artifact(
        base_dir, synthetic_label_source_1m(days), pair
    )

    def _source(chart_id: str):
        return load_verified_replay_chart_artifact(base_dir, artifact_id, expected_pair=pair)

    _source.artifact_id = artifact_id  # type: ignore[attr-defined]
    return _source
