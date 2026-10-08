"""Small synthetic checks for the MFFU worker seams; no market inputs."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from alpha_lab.agents.data_infra.ifvg.search.identities import canonical_contract_sha256
from alpha_lab.propsim.funded import mffu_batch_run as batch
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, to_ns
from alpha_lab.propsim.funded.full_range_batch import load_checkpoint, save_checkpoint
from alpha_lab.propsim.funded.pair_ledger import PairLedger
from alpha_lab.propsim.funded.position_walk import MinuteObservations
from alpha_lab.propsim.funded.profiles import FIRM_PROFILES
from alpha_lab.propsim.funded.strategy_driver import CoreStrategyDriver, EntrySignal, StepOutcome


@dataclass(frozen=True)
class _PolicyDecision:
    event: str = "entry_admission"
    action: str = "admit"
    reasons: tuple[str, ...] = ()


class _Context:
    def __init__(self, sign: str):
        self.sign = sign

    def snapshot(self, ts):
        return {
            "policy_id": "synthetic_nominal_eod", "decision_time_utc": ts.isoformat(),
            "gamma": {"sign": self.sign, "status": "selected" if self.sign != "unknown"
                      else "stale", "report_date": "2025-06-13"},
            "levels": {"status": "selected", "report_date": "2025-06-13", "items": {}},
        }

    def policy_context_fields(self, ts):
        return {
            "decision_ts_utc": ts,
            "total_net_gex": 1.0 if self.sign == "positive" else -1.0
            if self.sign == "negative" else None,
            "gamma_status": "available" if self.sign != "unknown" else "stale",
            "gamma_eligible_from_utc": datetime(2025, 6, 13, 22, tzinfo=UTC),
        }


@pytest.mark.parametrize(
    ("quantity_policy", "gamma_sign", "expected"),
    [("Q10", "positive", 10), ("Q6", "negative", 6),
     ("QG", "positive", 6), ("QG", "negative", 10),
     ("QG", "unknown", 10)],
)
def test_driver_freezes_entry_quantity_from_entry_asof_context(
    monkeypatch, quantity_policy, gamma_sign, expected,
):
    pytest.importorskip("strategy_core.strategies.ifvg_smc.ifsm_policy_context")
    from strategy_core.decisions import sessions
    from strategy_core.strategies.ifvg_smc import replay

    from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver

    monkeypatch.setattr(sessions, "classify_session",
                        lambda *_args: SimpleNamespace(session="ny"))
    monkeypatch.setattr(replay, "_runtime_scheme", lambda _scheme: object())
    signal = EntrySignal("trade-1", "long", 80_000, 79_900, 80_100, "1m")
    emission = SimpleNamespace(kind="ifsm_policy_decision", record=_PolicyDecision())
    monkeypatch.setattr(CoreStrategyDriver, "step", lambda *_args, **_kwargs:
                        StepOutcome(signal, (), (), emissions=(emission,)))
    section = SimpleNamespace(session_scheme=None, exit_policy="fixed_target_v1")
    driver = MffuCoreDriver(section, tick_size=.25, context_index=_Context(gamma_sign),
                            quantity_policy=quantity_policy)
    driver._orch = SimpleNamespace(_reducer=SimpleNamespace(active_trade_count=1))
    bar = SimpleNamespace(availability_ts_utc=datetime(2025, 6, 16, 15, tzinfo=UTC),
                          trading_day=date(2025, 6, 16))
    out = driver.step(bar, None)
    assert out.entry == signal
    assert driver.entry_quantities[signal.trade_id] == expected
    assert driver.entry_contexts[signal.trade_id]["selected_quantity"] == expected
    assert driver.policy_decisions[0]["action"] == "admit"
    driver._orch = None
    restored = MffuCoreDriver(section, tick_size=.25, context_index=_Context(gamma_sign),
                              quantity_policy=quantity_policy)
    restored.restore(driver.checkpoint())
    assert restored.entry_quantities == driver.entry_quantities
    assert restored.entry_contexts == driver.entry_contexts


def test_target_override_bridge_uses_the_funded_observation_time():
    pytest.importorskip("strategy_core.strategies.ifvg_smc.ifsm_policy_context")
    from strategy_core.strategies.ifvg_smc.ifsm_policy_context import IfsmPolicyContext

    from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver

    section = SimpleNamespace(session_scheme=None, exit_policy="gamma_conditional_1r_v1")
    index = _Context("positive")
    driver = MffuCoreDriver(section, tick_size=.25, context_index=index)
    instant = datetime(2025, 6, 16, 15, tzinfo=UTC)
    observed = {
        "asof": index.snapshot(instant),
        "receipt": IfsmPolicyContext(**index.policy_context_fields(instant)).receipt(),
    }
    override = driver.target_override_from({
        "decision_ns": int(instant.timestamp() * 1_000_000_000), "action": "whole",
        "context": observed,
    })
    assert override.action == "whole"
    assert override.context.decision_ts_utc == instant
    assert override.context.gamma_sign == "positive"
    with pytest.raises(ValueError, match="differs"):
        driver.target_override_from({
            "decision_ns": int(instant.timestamp() * 1_000_000_000), "action": "partial",
            "context": observed,
        })
    with pytest.raises(ValueError, match="target context differs"):
        driver.target_override_from({
            "decision_ns": int(instant.timestamp() * 1_000_000_000), "action": "whole",
            "context": {**observed, "receipt": {**observed["receipt"],
                                                "gamma_source_id": "wrong-source"}},
        })


def test_target_selector_and_core_override_respect_nominal_release_nanosecond():
    pytest.importorskip("strategy_core.strategies.ifvg_smc.ifsm_policy_context")
    from strategy_core.strategies.ifvg_smc.ifsm_policy_context import IfsmPolicyContext

    from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver

    release = datetime(2025, 6, 16, 22, tzinfo=UTC)

    class ReleaseContext(_Context):
        def snapshot(self, ts):
            return _Context("positive" if ts >= release else "unknown").snapshot(ts)

        def policy_context_fields(self, ts):
            selected = ts >= release
            return {
                "decision_ts_utc": ts,
                "total_net_gex": 1.0 if selected else None,
                "gamma_status": "available" if selected else "not_yet_eligible",
                "gamma_eligible_from_utc": release if selected else None,
            }

    driver = MffuCoreDriver(
        SimpleNamespace(session_scheme=None, exit_policy="gamma_conditional_1r_v1"),
        tick_size=.25, context_index=ReleaseContext("unknown"),
    )
    select = batch._target_selector(driver)
    release_ns = to_ns(release)
    before = select(release_ns - 1)
    after = select(release_ns + 1)
    assert (before.action, after.action) == ("partial", "whole")
    for ns, decision in ((release_ns - 1, before), (release_ns + 1, after)):
        override = driver.target_override_from({
            "decision_ns": ns, "action": decision.action, "context": decision.context,
        })
        assert override.action == decision.action
        assert decision.context["receipt"] == IfsmPolicyContext(
            **driver.context_index.policy_context_fields(override.context.decision_ts_utc)
        ).receipt()


def test_mffu_runtime_accepts_six_micro_per_fill_rounding_and_rejects_drift():
    valid = SimpleNamespace(
        variant_id="MCB001", possible_quantities=(10,),
        cost_per_contract_mills=514, fee_rounding_policy=batch.FEE_ROUNDING_POLICY_ID,
        exit_policy="scale_out_half_breakeven_hold_to_close_v1",
    )
    six = SimpleNamespace(**{
        **vars(valid), "variant_id": "MCB002", "possible_quantities": (6, 10),
    })
    batch._validate_fee_postings(SimpleNamespace(
        fee_rounding_policy=batch.FEE_ROUNDING_POLICY_ID, variants=(valid, six),
    ))
    changed = SimpleNamespace(**{**vars(six), "fee_rounding_policy": "other"})
    with pytest.raises(PermissionError, match="MCB002: fee rounding policy changed"):
        batch._validate_fee_postings(SimpleNamespace(
            fee_rounding_policy=batch.FEE_ROUNDING_POLICY_ID,
            variants=(valid, changed),
        ))


def test_spawned_worker_accepts_preimported_frozen_core_only(monkeypatch, tmp_path):
    source = tmp_path / "core" / "src"
    package = source / "strategy_core"
    package.mkdir(parents=True)
    entry = package / "__init__.py"
    entry.write_text("", encoding="utf-8")
    monkeypatch.setattr(batch.sys, "path", list(batch.sys.path))
    monkeypatch.setitem(batch.sys.modules, "strategy_core",
                        SimpleNamespace(__file__=str(entry)))
    batch._worker_init_core(str(tmp_path / "core"))
    assert batch.sys.path[0] == str(source.resolve())
    monkeypatch.setitem(batch.sys.modules, "strategy_core",
                        SimpleNamespace(__file__=str(tmp_path / "foreign.py")))
    with pytest.raises(PermissionError, match="outside the frozen source"):
        batch._worker_init_core(str(tmp_path / "core"))


def test_driver_daily_policy_counts_exclude_structural_candidate_blocks(monkeypatch):
    pytest.importorskip("strategy_core.strategies.ifvg_smc.ifsm_policy_context")
    from alpha_lab.propsim.funded.mffu_batch_driver import MffuCoreDriver

    emission = SimpleNamespace(
        kind="ifsm_policy_decision",
        record=_PolicyDecision(
            action="reject", reasons=("already_in_trade", "ifsm_early_positive",
                                      "daily_execution_cap", "retest_trigger_unratified"),
        ),
    )
    monkeypatch.setattr(CoreStrategyDriver, "step", lambda *_args, **_kwargs:
                        StepOutcome(None, (), (), emissions=(emission,)))
    driver = MffuCoreDriver(
        SimpleNamespace(session_scheme=None, exit_policy="fixed_target_v1"),
        tick_size=.25, context_index=_Context("positive"),
    )
    driver._orch = SimpleNamespace(_reducer=SimpleNamespace(active_trade_count=0))
    bar = SimpleNamespace(trading_day=date(2025, 6, 16))
    driver.step(bar, None)
    assert dict(driver.day_policy_reasons) == {
        "ifsm_early_positive": 1, "daily_execution_cap": 1,
    }
    assert driver.policy_decisions[0]["reasons"] == [
        "already_in_trade", "ifsm_early_positive", "daily_execution_cap",
        "retest_trigger_unratified",
    ]


@pytest.mark.parametrize("quantity,half,entry_fee,exit_fee", [
    (10, 5, 514, 257),
    (6, 3, 308, 154),
])
def test_mffu_funded_entry_and_two_half_exits(
    quantity, half, entry_fee, exit_fee,
):
    ledger = PairLedger(
        pair_id="MCB001|myfundedfutures", configuration="MCB001",
        profile=FIRM_PROFILES["myfundedfutures"],
        processing=TWO_BUSINESS_DAYS_FED_1600,
        quantity=quantity, tick_value_cents=50, cost_per_side_cents=0,
        cost_per_contract_mills=514, trading_days=(), start_ns=0,
        cutoff_ns=10_000_000_000, scale_out=True,
    )
    ledger.start()
    ledger.open(
        ts_ns=1_000_000_000, trade_ref="ten-micro", direction="long",
        entry_ticks=80_000, stop_ticks=79_900, target_ticks=80_100,
        trading_day="2025-06-16", strategy={"trade_id": "ten-micro"},
        quantity=quantity,
    )
    assert ledger.position.quantity == quantity
    assert ledger.position.entry_cost_cents == entry_fee

    def minute(ns, price):
        return MinuteObservations(
            open_ns=ns, close_ns=ns + 60, close_ticks=price,
            ts_ns=np.array([ns + 1], dtype=np.int64),
            price_ticks=np.array([price], dtype=np.int64),
            continuous=np.array([False]), fidelity="ordered_trade_prints",
        )

    assert ledger.on_minute(minute(2_000_000_000, 80_100), deadline_minute=False,
                            trading_day="2025-06-16") is None
    assert ledger.position.remaining_quantity == half
    exit_ = ledger.on_minute(minute(3_000_000_000, 80_000), deadline_minute=False,
                             trading_day="2025-06-16")
    assert exit_ is not None and exit_.kind == "breakeven_stop"
    trade = ledger.trades[0]
    assert (trade["quantity"], trade["scale_out_quantity"],
            trade["final_exit_quantity"]) == (quantity, half, half)
    assert trade["costs_cents"] == entry_fee + exit_fee + exit_fee


def test_dispatch_and_completed_child_checkpoint_reject_mutated_identity(tmp_path):
    class Source:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"calendar": "frozen"}

    class Variant:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"variant_id": "MCB001", "quantity": 10}

    class Core:
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"source_tree": "frozen"}

    plan = SimpleNamespace(core_root="C:/task/core", core_source=Core(),
                           runtime_source_file_sha256={"worker.py": "a" * 64},
                           context_archive_sha256="b" * 64, source=Source(),
                           task_b_scope=Source())
    dispatch = batch._dispatch("c" * 64, "d" * 64, plan, Variant())
    digest = canonical_contract_sha256(dispatch)
    path = tmp_path / "workers" / "MCB001" / "output.json"
    save_checkpoint(path, {"schema": batch.FINAL_SCHEMA, "dispatch_sha256": digest,
                           "completed_dates": ["2025-06-13", "2025-06-16"],
                           "output": {"dispatch": dispatch, "batch_id": "MCB001"}})
    loaded = load_checkpoint(path, dispatch_sha256=digest,
                             dates=("2025-06-13", "2025-06-16"))
    assert loaded["output"]["dispatch"] == dispatch
    with pytest.raises(PermissionError, match="different plan, worker or date prefix"):
        load_checkpoint(path, dispatch_sha256=canonical_contract_sha256({
            **dispatch, "context_archive_sha256": "e" * 64,
        }), dates=("2025-06-13", "2025-06-16"))
    with pytest.raises(PermissionError, match="different plan, worker or date prefix"):
        load_checkpoint(path, dispatch_sha256=digest,
                        dates=("2025-06-13", "2025-06-17"))


def test_terminal_result_pointer_binds_approval_and_every_worker():
    class Contract:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_dump(self, *, mode):
            assert mode == "json"
            return dict(self.__dict__)

    plan_id, approval_id = "a" * 64, "b" * 64
    rows = (Contract(variant_id="MCB001"), Contract(variant_id="MCB002"))
    plan = Contract(
        variants=rows, core_root="C:/task/core", core_source=Contract(source="bound"),
        runtime_source_file_sha256={"worker.py": "c" * 64},
        context_archive_sha256="d" * 64, source=Contract(scope="bound"),
        task_b_scope=Contract(task="bound"),
    )
    envelope = Contract(payload=Contract(
        funded_comparison_plan_id=plan_id,
        funded_comparison_approval_id=approval_id,
        engine_version=batch.ENGINE_VERSION, validation_passed=True,
    ))
    saved = {
        "funded_comparison_plan_id": plan_id, "validation": {"passed": True},
        "mffu_batch": {
            "schema": batch.RESULT_SCHEMA, "plan": plan.model_dump(mode="json"),
            "approval_id": approval_id,
            "worker_dispatches": [batch._dispatch(plan_id, approval_id, plan, row)
                                  for row in rows],
            "dispositions": [{"variant_id": row.variant_id, "status": "newly_completed"}
                             for row in rows],
        },
    }
    batch._verify_terminal_result_binding(
        saved, envelope, plan_id=plan_id, approval_id=approval_id, plan=plan,
    )
    for changed in (
        {"approval_id": "e" * 64},
        {"worker_dispatches": saved["mffu_batch"]["worker_dispatches"][:1]},
        {"dispositions": saved["mffu_batch"]["dispositions"][:1]},
    ):
        with pytest.raises(PermissionError, match="terminal MFFU result"):
            batch._verify_terminal_result_binding(
                {**saved, "mffu_batch": {**saved["mffu_batch"], **changed}},
                envelope, plan_id=plan_id, approval_id=approval_id, plan=plan,
            )
    with pytest.raises(PermissionError, match="terminal MFFU result"):
        batch._verify_terminal_result_binding(
            saved, Contract(payload=Contract(**{
                **vars(envelope.payload), "funded_comparison_approval_id": "e" * 64,
            })), plan_id=plan_id, approval_id=approval_id, plan=plan,
        )


def test_day_checkpoint_rejects_internal_count_or_schema_drift():
    dates = ("2025-06-13", "2025-06-16")
    checkpoint = {
        "schema": batch.CHECKPOINT_SCHEMA,
        "completed_dates": [dates[0]],
        "stats": {"completed_count": 1},
        "runs": [{"pair_id": "reference"}, {"pair_id": "funded"}],
        "daily_activity": [],
    }
    batch._verify_day_checkpoint(checkpoint, dates, (dates[1],))
    for changed in (
        {"schema": "old_schema"},
        {"completed_dates": [dates[1]]},
        {"stats": {"completed_count": 2}},
        {"runs": [{"pair_id": "reference"}]},
    ):
        with pytest.raises(PermissionError, match="checkpoint day state"):
            batch._verify_day_checkpoint({**checkpoint, **changed}, dates, (dates[1],))
    full = {
        **checkpoint, "completed_dates": list(dates), "stats": {"completed_count": 2},
        "daily_activity": [
            {"evaluation_date": dates[1], "stream": "strategy"},
            {"evaluation_date": dates[1], "stream": "funded"},
        ],
    }
    batch._verify_day_checkpoint(full, dates, (dates[1],))
    with pytest.raises(PermissionError, match="checkpoint day state"):
        batch._verify_day_checkpoint(
            {**full, "daily_activity": full["daily_activity"][:1]}, dates, (dates[1],),
        )


def test_managed_batch_preserves_all_failed_rows_as_unavailable(monkeypatch, tmp_path):
    """A worker failure must remain visible without manufacturing zero cash."""
    from alpha_lab.propsim.funded import mffu_batch_analysis, mffu_batch_reuse
    from alpha_lab.propsim.funded.clock import to_ns
    from alpha_lab.propsim.funded.mffu_batch_plan import HANDOFF_ROOT

    class Contract:
        def __init__(self, **fields):
            self.__dict__.update(fields)

        def model_dump(self, *, mode):
            assert mode == "json"
            return dict(self.__dict__)

    class Variant(Contract):
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"variant_id": self.variant_id, "quantity_policy": "Q10"}

    class Plan(Contract):
        def model_dump(self, *, mode):
            assert mode == "json"
            return {"plan": "synthetic_failed_rows_v1"}

    rows = tuple(Variant(variant_id=f"MCB{i:03d}", name=f"MCB{i:03d}",
                         instrument="micro", quantity=10, possible_quantities=(10,),
                         quantity_policy="Q10", cost_per_contract_mills=514,
                         fee_rounding_policy=batch.FEE_ROUNDING_POLICY_ID,
                         exit_policy="scale_out_half_breakeven_hold_to_close_v1")
                 for i in range(1, 65))
    source = Contract(warmup_dates=("2025-06-13",),
                      evaluation_dates=("2025-06-16", "2025-06-17"),
                      cutoff_utc="2025-06-18T21:00:00Z", title="synthetic")
    pairs_bytes = b"left_variant_id,right_variant_id,axis,pair_type,notes\n"
    handoff = tmp_path / "handoff.zip"
    with zipfile.ZipFile(handoff, "w") as package:
        package.writestr(HANDOFF_ROOT + "COMPARISON_PAIRS.csv", pairs_bytes)
    plan = Plan(
        variants=rows, source=source, purpose="historical_comparison",
        question="synthetic failed child preservation", core_root=str(tmp_path / "core"),
        core_source=Contract(root="synthetic"), runtime_source_file_sha256={},
        context_archive_sha256="a" * 64, task_b_scope=Contract(scope="synthetic"),
        firm_profiles=(FIRM_PROFILES["myfundedfutures"],),
        processing=TWO_BUSINESS_DAYS_FED_1600,
        execution_model=Contract(kind="synthetic"), owner_decisions=(),
        limitations=(), handoff_zip=str(handoff),
        source_member_sha256={"COMPARISON_PAIRS.csv": hashlib.sha256(pairs_bytes).hexdigest()},
        mode="single_account_configuration_comparison",
    )
    approval = Contract(
        funded_comparison_approval_id="b" * 64,
        payload=Contract(approved_on="2025-06-01", channel="owner", scope="synthetic"),
    )
    monkeypatch.setattr(batch, "load_approved_mffu_batch_plan",
                        lambda *_args: (Contract(payload=plan), approval))
    monkeypatch.setattr(batch, "validate_mffu_runtime", lambda *_args: (
        (), Contract(), Contract(audit_dict=lambda: {}), (),
        to_ns("2025-06-13T00:00:00Z"), to_ns(source.cutoff_utc),
    ))
    monkeypatch.setattr(batch, "_load_context", lambda _plan: object())
    monkeypatch.setattr(mffu_batch_reuse, "try_reuse_partial", lambda *_args: None)
    monkeypatch.setattr(mffu_batch_analysis, "analyze_mffu_batch", lambda *_args, **_kwargs:
                        {"status": "unavailable_failed_rows"})

    class FailedFuture:
        def result(self):
            raise RuntimeError("synthetic worker failure")

    class Pool:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, _fn, _task):
            return FailedFuture()

    monkeypatch.setattr(batch, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(batch, "as_completed", list)
    state = batch.run_mffu_batch(
        plan_id="c" * 64, store_root=tmp_path / "store",
        state_root=tmp_path / "state", workers=2,
    )
    assert state["status"] == "Incomplete"
    assert state["configurations_failed"] == 64
    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result

    saved = load_comparison_result(tmp_path / "store", state["result_id"])
    assert len(saved["tables"]["pair_results"]) == 64
    assert len(saved["mffu_batch"]["dispositions"]) == 64
    assert all(row["status"] == "Not completed" for row in saved["tables"]["pair_results"])
    assert all("net_cash_earned_usd" not in row for row in saved["tables"]["pair_results"])
    assert all(row["status"] == "failed" for row in saved["mffu_batch"]["dispositions"])
