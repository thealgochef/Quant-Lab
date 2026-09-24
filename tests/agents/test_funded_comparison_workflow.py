"""Funded configuration comparison through the NORMAL workspace configurator.

Uses the real verified daily-close study's approved configurations (read only;
skipped when that archive is absent) and temporary stores. The per-configuration
replay is replaced by deterministic synthetic account runs, so this proves the
configurator -> frozen plan -> approval -> worker -> result wiring for several
distinct configurations; real replays are covered by the engine tests and the
engineering run recorded in TASKS.md.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.propsim.funded.comparison_source import discover_comparison_sources  # noqa: E402
from tests.propsim.funded.pair_builders import (  # noqa: E402
    BASE,
    SyntheticDay,
    flat,
    run_pair,
    weekdays,
)

S0 = "enabled_entry_sessions.asia-london-ny"
S3 = "enabled_entry_sessions.morning_chicago_0700_1030_v1"
PICK = {
    "enabled_entry_sessions": [S0, S3],
    "opposing_parent_distance_ticks_max": ["opposing_parent_distance_ticks_max.160"],
    "opposing_min_gap_ticks": ["opposing_min_gap_ticks.1"],
    "parent_timeframes": ["parent_timeframes.3m-5m-10m-15m-30m"],
}


@pytest.fixture
def env(tmp_path, monkeypatch):
    if not any("S0_D160_W1_P0" in s.by_name for s in discover_comparison_sources()):
        pytest.skip("the verified daily-close study archive is not available locally")
    import ifvg_funded_comparison_study
    import ifvg_workspace

    roots = {
        "repo_root": tmp_path, "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "funded_state_root": tmp_path / "funded_jobs",
        "funded_comparison_state_root": tmp_path / "comparison_jobs",
        "reports_root": tmp_path / "reports",
    }
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    spawned: list[list[str]] = []

    def spawn(cmd):
        from alpha_lab.propsim.funded.runner import write_state

        spawned.append(cmd)
        write_state(Path(cmd[cmd.index("--state-root") + 1]),
                    cmd[cmd.index("--plan-id") + 1], status="Running", phase="queued")
        return 1

    monkeypatch.setattr(ifvg_funded_comparison_study, "_spawn", spawn)
    return {"roots": roots, "spawned": spawned}


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def _fake_configuration_run(requests: list):
    days = weekdays(date(2026, 3, 2), 6)

    def run(task):
        from alpha_lab.propsim.funded.comparison_plan import FundedComparisonPlanPayload

        plan = FundedComparisonPlanPayload.model_validate(task["plan"])
        name = task["configuration"]
        requests.append((name, plan.quantity, [p.firm_key for p in plan.firm_profiles]))
        gain = 700 if name.startswith("S0") else 60  # distinct outcomes per configuration
        specs = [SyntheticDay(d, [[BASE], [BASE + gain // 2, BASE + gain], *flat(4)],
                              {0: ("long", BASE - 100, BASE + gain)}) for d in days]
        pairs, resumed = {}, {}
        for profile in plan.firm_profiles:
            pair_id = f"{name}|{profile.firm_key}"
            run_, ledger, _ = run_pair(specs, firm_key=profile.firm_key, pair_id=pair_id,
                                       quantity=plan.quantity)
            pairs[profile.firm_key] = {
                "pair_id": pair_id, "ledger": json.loads(json.dumps(ledger.snapshot())),
                "forced_flat": run_.driver.forced_flat, "trades_not_in_reference": 0}
            resumed[profile.firm_key] = {"identical": True}
        return {"configuration": name, "display_name": f"Synthetic {name}", "axes": {},
                "settings_plain": [{"setting": "Entry hours", "value": name}],
                "reference": {"equivalent": True, "saved_study_trades": 0,
                              "replayed_trades": 0},
                "resumed": resumed,
                "prints": {"minutes_checked": 0, "minutes_rebuilt_exactly": 0,
                           "missing_utc_days": [], "files": []},
                "pairs": pairs}

    return run


def test_two_configurations_save_reopen_approve_and_reach_the_worker(env):
    roots = env["roots"]
    at = AppTest.from_function(_app, default_timeout=120).run()
    at.button(key="ifvg_workspace_new").click().run()
    at.selectbox[0].select("Funded configuration comparison").run()
    next(b for b in at.button if b.label == "Configure study").click().run()
    assert not at.exception, at.exception
    # the draft starts with every approved value selected (32 configurations)
    assert "32 configurations" in " ".join(str(m.value) for m in at.markdown)
    for axis, values in PICK.items():
        at.multiselect(key=f"ifvg_fcmp_axis_{axis}").set_value(values).run()
    at.number_input(key="ifvg_fcmp_quantity").set_value(2).run()
    assert not at.exception, at.exception
    (saved,) = roots["draft_root"].glob("*/draft.json")
    steps = json.loads(saved.read_text())["steps"]
    assert steps["search_space"]["axis_selections"]["enabled_entry_sessions"] == [S0, S3]
    assert steps["review"]["funded_comparison"]["quantity"] == 2
    text = " ".join(str(m.value) for m in at.markdown)
    assert "2 configurations" in text and "4 separate results" in text
    # an unsupported size is refused, never clipped
    at.number_input(key="ifvg_fcmp_quantity").set_value(4).run()
    assert any("MyFundedFutures" in e.value for e in at.error)
    at.number_input(key="ifvg_fcmp_quantity").set_value(2).run()
    # reopen in a fresh session: the saved selection comes back
    fresh = AppTest.from_function(_app, default_timeout=120).run()
    next(b for b in fresh.button if b.label == "Continue").click().run()
    assert fresh.multiselect(key="ifvg_fcmp_axis_enabled_entry_sessions").value == [S0, S3]
    assert fresh.number_input(key="ifvg_fcmp_quantity").value == 2
    # no run without the owner's approval of this exact plan
    assert fresh.button(key="ifvg_fcmp_run").disabled
    assert fresh.button(key="ifvg_fcmp_approve").disabled
    fresh.checkbox(key="ifvg_fcmp_agree").check().run()
    fresh.button(key="ifvg_fcmp_approve").click().run()
    assert not fresh.button(key="ifvg_fcmp_run").disabled
    fresh.button(key="ifvg_fcmp_run").click().run()
    assert not fresh.exception, fresh.exception
    (command,) = env["spawned"]
    plan_id = command[command.index("--plan-id") + 1]
    store = Path(command[command.index("--store-root") + 1])
    state_root = Path(command[command.index("--state-root") + 1])

    from alpha_lab.propsim.funded.comparison_runner import (
        find_approval,
        load_comparison_result,
        run_comparison_plan,
    )

    assert find_approval(store, plan_id).payload.channel == "study_screen"
    requests: list = []
    final = run_comparison_plan(plan_id=plan_id, store_root=store, state_root=state_root,
                                reports_root=roots["reports_root"],
                                run_config_fn=_fake_configuration_run(requests))
    assert final["status"] == "Completed", final
    assert sorted(r[0] for r in requests) == ["S0_D160_W1_P0", "S3_D160_W1_P0"]
    assert {r[1] for r in requests} == {2}  # the draft's size reached every configuration
    assert all(r[2] == ["takeprofittrader", "myfundedfutures"] for r in requests)
    result = load_comparison_result(store, final["result_id"])
    assert result["validation"]["passed"], result["validation"]
    pairs = {(row["configuration"], row["firm"]) for row in result["tables"]["pair_results"]}
    assert pairs == {("S0_D160_W1_P0", "TakeProfitTrader"), ("S0_D160_W1_P0", "MyFundedFutures"),
                     ("S3_D160_W1_P0", "TakeProfitTrader"), ("S3_D160_W1_P0", "MyFundedFutures")}
    by_pair = result["summaries_cents"]
    assert (by_pair["S0_D160_W1_P0|takeprofittrader"]["net_cash_earned_cents"]
            != by_pair["S3_D160_W1_P0|takeprofittrader"]["net_cash_earned_cents"])
    # every pair bought exactly one account in these runs: no five-copy multiplication
    assert {s["accounts_purchased"] for s in by_pair.values()} == {1}

    # the automatic review folder carries the exact bindings and the saved calendar
    first = Path(final["review_folder"])
    bindings = json.loads((first / "configuration_bindings.json").read_text(encoding="utf-8"))
    assert [c["configuration"] for c in bindings["configurations"]] == [
        "S0_D160_W1_P0", "S3_D160_W1_P0"]
    assert all(c["re_resolved_now_to_the_same_hash"] and c["axis_value_ids"]
               and c["effective_section"]["enabled_entry_sessions"]
               for c in bindings["configurations"])
    assert (first / "trading_calendar.csv").is_file()
    # a new export version of the same saved result, verified against its own manifest
    from alpha_lab.propsim.funded.comparison_runner import republish_comparison_review

    again = republish_comparison_review(
        plan_id=plan_id, store_root=store, state_root=state_root,
        reports_root=roots["reports_root"], review_findings="# Findings\n\nNone blocking.")
    second = Path(again["review_folder"])
    assert second.name.endswith("_export_v2") and again["receipt"]["passed"]
    assert again["receipt"]["payload_files_listed"] == len(
        [p for p in second.rglob("*") if p.is_file()]) - 1
    assert (second / "REVIEW_FINDINGS.md").is_file()
    assert load_comparison_result(store, final["result_id"]) == result  # never recomputed


def test_worker_refuses_a_plan_without_owner_approval(env, tmp_path):
    from alpha_lab.propsim.funded.comparison_runner import run_comparison_plan
    from alpha_lab.propsim.funded.comparison_study import (
        build_comparison_plan,
        resolve_selection,
        save_plan,
    )

    source = next(s for s in discover_comparison_sources() if "S0_D160_W1_P0" in s.by_name)
    configurations, _ = resolve_selection(source, PICK)
    envelope = build_comparison_plan(source, configurations,
                                     firm_keys=["takeprofittrader"], instrument="mini",
                                     quantity=1, cost_per_side_cents=514)
    store = tmp_path / "store2"
    plan_id = save_plan(store, envelope)
    with pytest.raises(PermissionError):
        run_comparison_plan(plan_id=plan_id, store_root=store, state_root=tmp_path / "s",
                            reports_root=tmp_path / "r",
                            run_config_fn=lambda task: pytest.fail("must not run"))
    state = json.loads((tmp_path / "s" / plan_id / "state.json").read_text())
    assert state["status"] == "Failed"


def test_unapproved_combinations_are_listed_not_dropped(env):
    from alpha_lab.propsim.funded.comparison_study import resolve_selection

    source = next(s for s in discover_comparison_sources() if "S0_D160_W1_P0" in s.by_name)
    chosen, unavailable = resolve_selection(
        source, {**PICK, "opposing_parent_distance_ticks_max": [
            "opposing_parent_distance_ticks_max.160", "opposing_parent_distance_ticks_max.40"]})
    assert len(chosen) == 2 and len(unavailable) == 2


FAKE_CORE = {"root": "research-core", "base_commit": "7c7111e398c083cf8e966e2e0c5aac8a41cc12c0",
             "branch": "funded-scale-out-exit", "patch_sha256": "1" * 64}


def test_variation_plan_saves_reopens_and_launches_only_on_its_frozen_core(
        env, monkeypatch, tmp_path):
    import ifvg_funded_comparison_job as job

    from alpha_lab.propsim.funded import core_identity, research_core_sources
    from alpha_lab.propsim.funded.comparison_runner import is_v2, load_plan

    monkeypatch.setattr(core_identity, "core_source_identity", lambda: dict(FAKE_CORE))
    roots = env["roots"]
    at = AppTest.from_function(_app, default_timeout=180).run()
    at.button(key="ifvg_workspace_new").click().run()
    at.selectbox[0].select("Funded configuration comparison").run()
    next(b for b in at.button if b.label == "Configure study").click().run()
    at.radio(key="ifvg_fcmp_plan_kind").set_value("variations").run()
    at.selectbox(key="ifvg_fcmp_var_base").set_value("S0_D160_W1_P0").run()
    at.multiselect(key="ifvg_fcmp_var_tp_r_multiple").set_value(
        ["tp_r_multiple.1.0", "tp_r_multiple.2.0"]).run()
    at.number_input(key="ifvg_fcmp_var_whole_q").set_value(2).run()
    assert not at.exception, at.exception
    (saved,) = roots["draft_root"].glob("*/draft.json")
    settings = json.loads(saved.read_text())["steps"]["review"]["funded_comparison"]
    assert settings["plan_kind"] == "variations"
    assert settings["variation"]["base"] == "S0_D160_W1_P0"
    assert settings["variation"]["selections"]["tp_r_multiple"] == [
        "tp_r_multiple.1.0", "tp_r_multiple.2.0"]
    assert settings["variation"]["whole_quantity"] == 2
    # reopen in a fresh session: the variation choices come back
    fresh = AppTest.from_function(_app, default_timeout=180).run()
    next(b for b in fresh.button if b.label == "Continue").click().run()
    assert fresh.radio(key="ifvg_fcmp_plan_kind").value == "variations"
    assert fresh.multiselect(key="ifvg_fcmp_var_tp_r_multiple").value == [
        "tp_r_multiple.1.0", "tp_r_multiple.2.0"]
    text = " ".join(str(m.value) for m in fresh.markdown)
    assert "2 configurations around" in text and "4 separate results" in text
    assert any("Strategy-Core branch funded-scale-out-exit (7c7111e" in str(c.value)
               for c in fresh.caption)
    fresh.checkbox(key="ifvg_fcmp_agree").check().run()
    fresh.button(key="ifvg_fcmp_approve").click().run()
    fresh.button(key="ifvg_fcmp_run").click().run()
    assert not fresh.exception, fresh.exception
    (command,) = env["spawned"]
    plan_id = command[command.index("--plan-id") + 1]
    store = Path(command[command.index("--store-root") + 1])
    plan = load_plan(store, plan_id)
    assert is_v2(plan) and plan.core_source.patch_sha256 == "1" * 64
    assert [v.quantity for v in plan.variants] == [2, 2]
    # the worker starts only on a checkout with exactly the frozen source
    assert job.worker_environment(store, plan_id) == (None, None)
    checkout = tmp_path / "strategy-core-research"
    (checkout / "src" / "strategy_core").mkdir(parents=True)
    (checkout / "src" / "strategy_core" / "__init__.py").write_text("")
    monkeypatch.setenv("IFSM_RESEARCH_CORE", str(checkout))
    monkeypatch.setattr(research_core_sources, "source_identity_at",
                        lambda path: dict(FAKE_CORE) if Path(path) == checkout.resolve()
                        else {"base_commit": "x", "patch_sha256": "y"})
    worker_env, core = job.worker_environment(store, plan_id)
    assert core == checkout.resolve()
    assert str(checkout.resolve() / "src") in worker_env["PYTHONPATH"]
