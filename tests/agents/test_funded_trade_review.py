"""Funded comparison trades in the existing Trade review (repair R3).

Synthetic saved result (tests/propsim/funded/comparison_fixture.py) plus one
synthetic half-exit row; the saved-result and plan loaders are replaced so no
store is read. The strategy evidence is reported unavailable here, which also
covers the honest missing-evidence path.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import (  # noqa: E402
    funded_trade_facts,
    pair_trades,
    review_keys,
    strategy_trade_links,
)
from tests.propsim.funded.comparison_fixture import comparison_fixture_result  # noqa: E402

RESULT_ID = "c" * 64
PLAN_ID = "abababababababababababababababababababababababababababababababab"


def _with_half_exit(result):
    result = copy.deepcopy(result)
    base = next(t for t in result["tables"]["trades"]
                if t["pair_id"] == "S0_D160|takeprofittrader" and t["seq"] == 21)
    base.update(quantity=10, scale_out_ns=1769550053641855957, scale_out_ticks=base[
        "target_ticks"], scale_out_quantity=5, final_exit_quantity=5,
        final_stop_ticks=base["entry_ticks"], exit_kind="breakeven_stop",
        exit_ticks=base["entry_ticks"])
    return result


@pytest.fixture
def result():
    return _with_half_exit(comparison_fixture_result())


def test_pair_trades_are_one_pair_in_recorded_time_order(result):
    trades = pair_trades(result, "S0_D160", "takeprofittrader")
    assert [t["seq"] for t in trades] == [4, 15, 21]
    assert {t["pair_id"] for t in trades} == {"S0_D160|takeprofittrader"}


def test_half_exit_facts_carry_the_exact_partial_fill_and_moved_stop(result):
    trade = next(t for t in pair_trades(result, "S0_D160", "takeprofittrader")
                 if t["seq"] == 21)
    facts = dict(funded_trade_facts(result, trade, instrument="micro").rows)
    assert facts["Initial quantity"] == "10 x Micro E-mini Nasdaq-100 (MNQ)"
    assert facts["Half exit (partial fill)"].endswith(
        "January 27, 2026 3:40:53.641855957 PM CST")
    assert facts["Remaining quantity"] == "5"
    assert facts["Stop after the half exit"].startswith(
        f"{trade['entry_ticks'] * 0.25:,.2f}")
    labels = [m.label for m in funded_trade_facts(result, trade).markers]
    assert labels == ["Entry", "Half exit", "Final exit"]
    stops = funded_trade_facts(result, trade).stops
    assert [s.label for s in stops] == ["Initial stop", "Moved stop"]
    assert stops[0].end == stops[1].start == 1769550053641855957  # the exact half-fill instant


def test_account_failure_shows_the_limit_check_and_the_replacement(result):
    trade = next(t for t in pair_trades(result, "S0_D160", "takeprofittrader")
                 if t["account_failed"])
    facts = dict(funded_trade_facts(result, trade, instrument="mini").rows)
    assert "Account liquidation" in facts and "Final exit" not in facts
    assert "the loss limit" in facts["Loss limit check"]
    assert facts["Replacement"].startswith("Account 2 started")


def test_same_strategy_trade_at_both_firms_never_shares_a_review(result):
    tpt = next(t for t in result["tables"]["trades"] if t["pair_id"] == "S0_D160|takeprofittrader"
               and t["seq"] == 4)
    mff = next(t for t in result["tables"]["trades"] if t["pair_id"] == "S0_D160|myfundedfutures"
               and t["seq"] == 4)
    # the same strategy trade entered at the same instant in both firms' paths
    mff.update(strategy_trade_id=tpt["strategy_trade_id"], trade_ref=tpt["trade_ref"],
               entry_utc=tpt["entry_utc"])
    assert tpt["strategy_trade_id"] == mff["strategy_trade_id"]
    assert tpt["entry_utc"] == mff["entry_utc"]
    keys = {review_keys(RESULT_ID, PLAN_ID, t)[1] for t in (tpt, mff)}
    other_result = review_keys("d" * 64, PLAN_ID, tpt)[1]
    assert len(keys) == 2 and other_result not in keys
    assert all(k.startswith("funded:") for k in keys)  # never a strategy candidate id


def test_geometry_only_for_configurations_of_the_verified_study():
    member = SimpleNamespace(core_replay_id="core-member")
    cache = SimpleNamespace(core_replay_id="core-cache")
    source = SimpleNamespace(by_name={"S0_D80_W1_P1": member, "S1_D80_W1_P1": cache})
    plan = SimpleNamespace(variants=(
        SimpleNamespace(name="S0_D80_W1_P1", in_verified_study=True,
                        cache_configuration="S0_D80_W1_P1"),
        SimpleNamespace(name="S0-T1-H1-P1-L-SO", in_verified_study=False,
                        cache_configuration="S1_D80_W1_P1")))
    assert strategy_trade_links(plan, source, "S0_D80_W1_P1") == ("core-member", "core-member")
    assert strategy_trade_links(plan, source, "S0-T1-H1-P1-L-SO") == (None, "core-cache")


# ── the real Trade review screen (AppTest) ───────────────────────────────────


def _app():
    import ifvg_search_review
    import streamlit as st

    ifvg_search_review.render_trade_review(st, ifvg_search_review._TEST_ROOTS)


@pytest.fixture
def screen(monkeypatch, tmp_path, result):
    import ifvg_funded_trade_review
    import ifvg_search_review

    from alpha_lab.propsim.funded import comparison_runner

    study = SimpleNamespace(key=PLAN_ID, kind="funded_comparison", archived=False,
                            status="Completed", name="Funded variation study",
                            state={"result_id": RESULT_ID, "status": "Completed"})
    roots = {"state_root": tmp_path / "jobs", "store_root": tmp_path / "store",
             "repo_root": tmp_path}
    monkeypatch.setattr(ifvg_search_review, "_TEST_ROOTS", roots, raising=False)
    monkeypatch.setattr(ifvg_search_review, "list_search_runs", lambda *a: [])
    monkeypatch.setattr(ifvg_search_review, "load_studies", lambda roots: ([study], []))
    monkeypatch.setattr(ifvg_funded_trade_review, "_result", lambda store, rid: result)
    monkeypatch.setattr(ifvg_funded_trade_review, "_strategy_source", lambda store, pid: None)
    monkeypatch.setattr(comparison_runner, "load_plan",
                        lambda store, pid: SimpleNamespace(instrument="mini", variants=()))
    monkeypatch.setattr(comparison_runner, "load_comparison_result", lambda store, rid: result)
    return {"roots": roots, "study": study}


def _run(state: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app, default_timeout=60)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _text(at) -> str:
    return "\n".join(str(e.value) for kind in ("markdown", "caption", "warning", "info",
                                               "error") for e in at.get(kind))


def test_funded_comparisons_are_offered_when_no_saved_search_exists(screen):
    at = _run({"ifvg_review_source": "Study executions"})
    assert not at.exception, at.exception
    assert "No saved searches are available for review." not in _text(at)
    study = at.selectbox(key="ifvg_search_review_search")
    assert study.value == "funded:" + PLAN_ID
    assert study.format_func(study.value) == "Funded variation study (funded accounts)"
    facts = at.table[0].value.iloc[:, 0].to_dict()
    assert facts["Firm and account"] == "TakeProfitTrader — account 1"
    assert "chart is unavailable" in _text(at)  # honest missing-evidence message


def test_results_link_opens_the_exact_configuration_and_firm(screen):
    prefix = f"ifvg_funded_review_{RESULT_ID[:16]}"
    at = _run({"ifvg_funded_review_pending": {"plan_id": PLAN_ID, "result_id": RESULT_ID,
                                              "configuration": "S0_D160",
                                              "firm_key": "myfundedfutures"}})
    assert not at.exception, at.exception
    assert at.selectbox(key=f"{prefix}_configuration").value == "S0_D160"
    assert at.radio(key=f"{prefix}_firm").value == "myfundedfutures"
    facts = at.table[0].value.iloc[:, 0].to_dict()
    assert facts["Firm and account"].startswith("MyFundedFutures")


def test_zero_trade_and_not_completed_pairs_say_so(screen, result):
    prefix = f"ifvg_funded_review_{RESULT_ID[:16]}"
    at = _run({"ifvg_review_source": "Study executions",
               f"{prefix}_configuration": "S3_D80", f"{prefix}_firm": "takeprofittrader"})
    assert "No trades were taken by this configuration with this firm." in _text(at)
    result["summaries_cents"]["S3_D80|takeprofittrader"]["status"] = "Not completed"
    at = _run({"ifvg_review_source": "Study executions",
               f"{prefix}_configuration": "S3_D80", f"{prefix}_firm": "takeprofittrader"})
    assert "did not complete with this firm" in _text(at)


def test_a_funded_review_is_saved_under_its_own_keys(screen):
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    at = _run({"ifvg_review_source": "Study executions"})
    reviewer = next(t for t in at.text_input if t.label == "Reviewer")
    reviewer.input("Test reviewer").run()
    verdict = next(s for s in at.selectbox if s.label == "Overall verdict")
    verdict.select_index(1).run()
    next(b for b in at.button if "Save" in b.label).click().run()
    assert not at.exception, at.exception
    saved = list_reviews(repo_root=screen["roots"]["repo_root"])
    assert len(saved) == 1
    row = saved.iloc[0]
    assert row["replay_chart_artifact_id"] == f"funded_comparison_result:{RESULT_ID}"
    assert row["candidate_id"].startswith(f"funded:{RESULT_ID}:S0_D160|takeprofittrader#")
    assert row["pair_ref"]["firm_key"] == "takeprofittrader"
    assert pd.isna(row["decision_id"]) or row["decision_id"] is None


@pytest.mark.parametrize(("status", "label", "notice"), [
    ("Incomplete", "Funded variation study (funded accounts; some configurations did not "
                   "complete)", "only the completed ones have trades here"),
    ("Failed", "Funded variation study (funded accounts; failed its money checks, review "
               "only)", "shown for review only"),
])
def test_results_shown_on_the_results_page_are_offered_with_their_status(
        screen, status, label, notice):
    """Review finding: the results page shows these results and their review button."""

    screen["study"].status = status
    at = _run({"ifvg_funded_review_pending": {"plan_id": PLAN_ID, "result_id": RESULT_ID,
                                              "configuration": "S0_D160",
                                              "firm_key": "takeprofittrader"}})
    assert not at.exception, at.exception
    study = at.selectbox(key="ifvg_search_review_search")
    assert study.value == "funded:" + PLAN_ID and study.format_func(study.value) == label
    assert notice in _text(at)


def test_a_study_that_cannot_be_offered_never_opens_another(screen, monkeypatch):
    """Review finding: no silent fallback to the first listed study."""

    import ifvg_search_review

    other = SimpleNamespace(search_id="d" * 64, archived=False)
    monkeypatch.setattr(ifvg_search_review, "list_search_runs", lambda *a: [other])
    screen["study"].archived = True
    at = _run({"ifvg_funded_review_pending": {"plan_id": PLAN_ID, "result_id": RESULT_ID,
                                              "configuration": "S0_D160",
                                              "firm_key": "takeprofittrader"}})
    assert not at.exception, at.exception
    assert "No other study was opened in its place." in _text(at)
    assert not [w for w in at.selectbox if w.key == "ifvg_search_review_search"]


def test_the_results_account_opens_in_review_and_back_restores_it(screen):
    prefix = f"ifvg_funded_review_{RESULT_ID[:16]}"
    pair = "S0_D160|takeprofittrader"
    at = _run({"ifvg_funded_review_pending": {"plan_id": PLAN_ID, "result_id": RESULT_ID,
                                              "configuration": "S0_D160",
                                              "firm_key": "takeprofittrader",
                                              "account_number": 2}})
    assert not at.exception, at.exception
    assert at.selectbox(key=f"{prefix}_{pair}_account").value == 2
    facts = at.table[0].value.iloc[:, 0].to_dict()
    assert facts["Firm and account"] == "TakeProfitTrader — account 2"
    at.button(key=f"{prefix}_back").click().run()
    context = at.session_state["funded_comparison_v1_selected_context"][RESULT_ID]
    assert context == {"configuration": "S0_D160", "firm_key": "takeprofittrader",
                       "account": {"pair": pair, "number": 2}}
    # an account of another configuration or firm is never applied
    at = _run({"ifvg_funded_review_pending": {"plan_id": PLAN_ID, "result_id": RESULT_ID,
                                              "configuration": "S3_D80",
                                              "firm_key": "myfundedfutures",
                                              "account_number": 2}})
    assert at.selectbox(key=f"{prefix}_S3_D80|myfundedfutures_account").value == "all"

