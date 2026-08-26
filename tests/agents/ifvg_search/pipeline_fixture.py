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
    source_envelope, _event_bytes = build_mbp1_source_artifact(
        events_by_day,
        contract=contract,
        authorized_date_set_id="synthetic_fixture_days_v1",
        events_stored=True,
    )
    return source_envelope, events_by_day, anchors


def build_pipeline_fixture(
    tmp_root: Path,
    *,
    stage_plan: tuple[QuantLabPipelineStage, ...] = FULL_STAGE_PLAN,
    bootstrap_n_paths: int = 32,
    label_builder=None,
    mbp1: bool = False,
) -> dict[str, Any]:
    """Everything one `run_pipeline` invocation needs, on temporary roots.

    ``mbp1=True`` pins the B2 order-flow bundle + the logistic protocol and
    wires the synthetic MBP-1 evidence seam — the R5B controlled-study
    pipeline path.
    """

    tmp_root = Path(tmp_root)
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
        feature_bundle_ids=("B2_CORE_ORDER_FLOW",) if mbp1 else ("B0_CORE",),
        label_policy_id=LABEL_POLICY_ID,
        fold_protocol_id=FOLD_PROTOCOL_ID_V1,
        model_protocol_id=(
            "ifvg_context_logistic_l2_v1" if mbp1 else "ifvg_context_catboost_binary_v1"
        ),
        cost_policy_sha256=canonical_contract_sha256(charter.payload.cost_policy),
        account_policy_set_ids=policy_ids,
        portfolio_policy_ids=(),
        simulation_protocol=simulation_protocol,
        software_commits={"quant_lab": "b" * 40, "strategy_core": "a" * 40},
        stage_plan=stage_plan,
    )
    semantic = PipelineSemanticIdentity.from_payload(payload)
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
        chart_builder=_companion_builder("replay_chart"),
        candidate_view_source=lambda: view,
        label_builder=label_builder or _default_label_builder,
        firm_specs=specs,
        mbp1_evidence_source=(lambda: mbp1_evidence) if mbp1 else None,
    )
    return {
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
