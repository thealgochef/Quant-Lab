"""Funded payout study through the NORMAL workspace: save, reopen, freeze,
launch, worker propagation, verified result, screen and review folder.

Uses a SYNTHETIC verified package and synthetic trade prints in pytest's
temporary directory. It proves the workflow wiring, not a historical result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.propsim.funded import sources  # noqa: E402
from tests.propsim.funded.fake_package import RUN_ID, build_fake_package  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    import ifvg_funded_study
    import ifvg_workspace

    archive, data = tmp_path / "archive", tmp_path / "prints"
    build_fake_package(archive, data)
    monkeypatch.setattr(sources, "ARCHIVE_ROOT", archive)
    roots = {
        "repo_root": tmp_path, "store_root": tmp_path / "store",
        "store_roots": {"research": tmp_path / "store"},
        "draft_root": tmp_path / "drafts", "state_root": tmp_path / "jobs",
        "pipeline_state_root": tmp_path / "pipelines",
        "funded_state_root": tmp_path / "funded_jobs",
        "reports_root": tmp_path / "reports",
    }
    monkeypatch.setattr(ifvg_workspace, "_TEST_ROOTS", roots, raising=False)
    authorized = {**ifvg_funded_study.PILOT_AUTHORIZATION["settings"],
                  "package_run_id": RUN_ID, "profile_id": "P1", "quantity": 2}
    monkeypatch.setattr(ifvg_funded_study, "PILOT_AUTHORIZATION",
                        {**ifvg_funded_study.PILOT_AUTHORIZATION, "settings": authorized})
    monkeypatch.setattr(ifvg_funded_study, "DEFAULT_SETTINGS",
                        {**ifvg_funded_study.DEFAULT_SETTINGS, "package_run_id": RUN_ID,
                         "profile_id": "P1"})
    spawned: list[list[str]] = []

    def spawn(cmd):
        # the real job script records its queued state before the worker runs
        from alpha_lab.propsim.funded.runner import write_state

        spawned.append(cmd)
        write_state(Path(cmd[cmd.index("--state-root") + 1]),
                    cmd[cmd.index("--plan-id") + 1], status="Running", phase="queued")
        return 1

    monkeypatch.setattr(ifvg_funded_study, "_spawn", spawn)
    return {"roots": roots, "archive": archive, "data": data, "spawned": spawned}


def _app():
    import ifvg_workspace
    import streamlit as st

    ifvg_workspace.render_workspace(st, roots=ifvg_workspace._TEST_ROOTS)


def _open_funded(at):
    at.button(key="ifvg_workspace_new").click().run()
    at.selectbox[0].select("Funded five-account operation (earlier budgeted mode)").run()
    next(b for b in at.button if b.label == "Configure study").click().run()
    assert not at.exception, at.exception


def test_settings_save_reopen_reach_the_worker_and_publish(env):
    roots = env["roots"]
    at = AppTest.from_function(_app, default_timeout=60).run()
    _open_funded(at)
    # unauthorized size first: saved, but the run button stays disabled
    at.number_input(key="ifvg_funded_quantity").set_value(3).run()
    run_button = at.button(key="ifvg_funded_run")
    assert run_button.disabled
    saved = list(roots["draft_root"].glob("*/draft.json"))
    assert len(saved) == 1
    stored = json.loads(saved[0].read_text())["steps"]["review"]["funded_settings"]
    assert stored["quantity"] == 3 and stored["processing"] == "two_business_days"
    # an unsupported size is refused, never clipped
    at.number_input(key="ifvg_funded_quantity").set_value(4).run()
    assert any("MyFundedFutures" in e.value for e in at.error)
    # reopen from My studies in a fresh session: the saved value comes back
    at.number_input(key="ifvg_funded_quantity").set_value(2).run()
    fresh = AppTest.from_function(_app, default_timeout=60).run()
    draft_button = next(b for b in fresh.button if b.label == "Continue")
    draft_button.click().run()
    assert fresh.number_input(key="ifvg_funded_quantity").value == 2
    # the authorized plan launches exactly one worker for the frozen plan
    assert not fresh.button(key="ifvg_funded_run").disabled
    fresh.button(key="ifvg_funded_run").click().run()
    assert not fresh.exception, fresh.exception
    (command,) = env["spawned"]
    plan_id = command[command.index("--plan-id") + 1]
    store = command[command.index("--store-root") + 1]
    state = command[command.index("--state-root") + 1]
    reports = command[command.index("--reports-root") + 1]
    # run that worker's code path in-process against the frozen plan
    from alpha_lab.propsim.funded.runner import load_result, read_ledger, run_plan

    final = run_plan(plan_id=plan_id, store_root=Path(store), state_root=Path(state),
                     reports_root=Path(reports), archive_root=env["archive"],
                     data_root=env["data"])
    assert final["status"] == "Completed", final
    result = load_result(Path(store), final["result_id"])
    assert result["settings"]["quantity"] == 2  # the draft value reached the worker
    assert result["validation"]["passed"]
    assert result["price_evidence"]["trades_with_ordered_prints"] == 3
    tpt = result["summaries_cents"]["takeprofittrader"]
    # A: +560 ticks x $5 x 2 minis - 2 x $10.28 = 5,579.44 -> 3,479.44 gross each.
    # After the Jan 15 receipt, C (+100 ticks, 2 minis, less $20.56) leaves
    # $3,079.44 -> a second $979.44 request, still processing at the cutoff.
    assert tpt["gross_withdrawals_requested_cents"] == 5 * (347_944 + 97_944)
    assert tpt["payouts_processing_at_cutoff"] == 5
    assert tpt["growth_purchases"] == 1  # the posted receipts funded one block
    assert {t["quantity"] for t in result["tables"]["trades"]} == {2}
    trade_ids = {t["trade_id"] for t in result["tables"]["trades"]}
    assert trade_ids == {"A", "C"}  # B arrived during payout processing
    folder = Path(final["review_folder"])
    assert folder.is_dir() and (folder / "trades.csv").is_file()
    assert not list(folder.rglob("*.py"))
    ledger = read_ledger(Path(store))
    assert ledger[0]["ledger_origin"] == "imported_verified_history"
    assert [e["event_type"] for e in ledger[1:]] == ["run_started", "run_completed"]
    # the completed study is listed and its verified result renders
    listed = AppTest.from_function(_app, default_timeout=60).run()
    open_button = next(b for b in listed.button if b.label == "View results")
    open_button.click().run()
    assert not listed.exception, listed.exception
    text = " ".join(str(m.value) for m in listed.markdown)
    assert "TakeProfitTrader" in text and "MyFundedFutures" in text


def test_worker_refuses_a_changed_source_package(env, tmp_path):
    import ifvg_funded_study

    from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable
    from alpha_lab.propsim.funded.runner import PLAN_STORE, run_plan

    settings = {**ifvg_funded_study.DEFAULT_SETTINGS, "quantity": 2}
    envelope = ifvg_funded_study.build_plan(settings)
    store = tmp_path / "store2"
    save_envelope_immutable(store, PLAN_STORE, envelope)
    trades = next(env["archive"].rglob("trades.csv"))
    trades.write_text(trades.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_plan(plan_id=envelope.funded_plan_id, store_root=store,
                 state_root=tmp_path / "s", reports_root=tmp_path / "r",
                 archive_root=env["archive"], data_root=env["data"])
    state = json.loads((tmp_path / "s" / envelope.funded_plan_id / "state.json").read_text())
    assert state["status"] == "Failed"
