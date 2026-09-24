"""Funded configuration-comparison results screen (SYNTHETIC fixture only).

The result rendered here is a hand-made engineering fixture built through the
real comparison result builder. Passing these tests proves the presenter and
screen handle every state; it is not evidence that a historical study ran.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.funded_comparison import (  # noqa: E402
    COMPARISON_HEADLINE_KEYS,
    FUTURE_NOTE,
    comparison_headline_figures,
    completed_configurations,
    plain_reason,
    present_comparison,
    present_pair_detail,
)
from tests.propsim.funded.comparison_fixture import comparison_fixture_result  # noqa: E402

_AMPM = re.compile(r"^\w+ \d{1,2}, \d{4} \d{1,2}:\d{2} (AM|PM) C[DS]T$")
_FORBIDDEN = [
    re.compile(r"\b[0-9a-f]{32,}\b"),  # hashes
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # uuids
    re.compile(r"\|(takeprofittrader|myfundedfutures)"),  # pair ids
    re.compile(r"#\d+-payout"),  # request ids
    re.compile(r"\bS\d+_D\d+\b"),  # configuration keys
    re.compile(r"_cents\b|_utc\b|_usd\b|_ns\b"),
    re.compile(r"funded_comparison_result_v1|single_account_configuration_comparison"),
    re.compile(r"[A-Za-z]:\\|\\\\|/Users/|/home/|\.py\b|\.json\b|\.parquet\b"),
    re.compile(r"ValueError|Traceback"),
]


@pytest.fixture(scope="module")
def result():
    return comparison_fixture_result()


def _clean(text: str) -> list[str]:
    return [p.pattern for p in _FORBIDDEN if p.search(text)]


# ── presenter ────────────────────────────────────────────────────────────


def test_headline_figures_are_the_exact_saved_cents(result):
    figures = comparison_headline_figures(result)
    assert len(figures) == 6  # 3 configurations x 2 firms
    for key, summary in result["summaries_cents"].items():
        if summary["status"] == "Completed":
            assert figures[key] == {"status": "Completed",
                                    **{k: summary[k] for k in COMPARISON_HEADLINE_KEYS}}
        else:
            assert figures[key] == {"status": "Not completed"}  # never zeros


def test_firm_tables_are_separate_ranked_and_keep_every_configuration(result):
    view = present_comparison(result)
    assert [t.firm for t in view.firm_tables] == ["TakeProfitTrader", "MyFundedFutures"]
    tpt, mff = view.firm_tables
    assert all(r.firm_key == "takeprofittrader" for r in tpt.rows)
    assert all(r.firm_key == "myfundedfutures" for r in mff.rows)
    assert [r.rank_text for r in tpt.rows] == ["1", "2", "—"]
    assert tpt.rows[0].label.startswith("Alpha") and mff.rows[0].label.startswith("Beta")
    top = tpt.rows[0].display()
    assert top["Net cash earned"] == "$2,203.78"
    assert top["Payouts received after the split"] == "$2,407.78 (1 payout)"
    assert top["Account costs"] == "$204.00"
    assert top["Lost after a payout"] == "1" and top["Lost before a payout"] == "0"
    # zero-payout configuration stays in the table, unfavorable figures shown plainly
    zero = tpt.rows[1].display()
    assert zero["Net cash earned"] == "-$102.00"
    assert zero["Largest payout"] == "No payout received"
    # not completed: labeled with the reason, never shown as a zero
    gamma = tpt.rows[2]
    assert not gamma.completed and set(gamma.display().values()) >= {"Not completed"}
    assert "$0.00" not in " ".join(gamma.display().values())
    assert gamma.reason == "Not completed: Recorded prices for 2026-02-03 are unavailable."


def test_header_status_and_price_evidence(result):
    view = present_comparison(result)
    assert view.question.startswith("Which tested strategy configuration")
    assert view.period_text == ("January 12, 2026 5:00 PM CST to March 6, 2026 4:00 PM CST "
                                "(Chicago time)")
    assert view.firms_text.startswith("TakeProfitTrader and MyFundedFutures")
    assert view.size_text == "1 E-mini Nasdaq-100 (NQ) contract per trade"
    assert view.configurations_text == "2 of 3 configurations completed"
    status = " ".join(s.text for s in view.status)
    assert "Simulated historical result" in status
    assert "Engineering sample" in status
    assert "220 of 240 position minutes used recorded exchange trades" in status
    assert "20 used a labeled one-minute approximation" in status
    assert "did not complete" in status
    assert view.future_note == FUTURE_NOTE and view.limitations


def test_status_for_approved_and_failed_results(result):
    variant = dict(result)
    variant["purpose"] = "historical_comparison"
    variant["approval"] = {"approved_on": "2026-09-23", "channel": "chat", "scope": "x"}
    variant["validation"] = {"passed": False, "checks": {}}
    view = present_comparison(variant)
    texts = [s.text for s in view.status]
    assert any("approved by the owner on September 23, 2026" in t for t in texts)
    assert not any("Engineering sample" in t for t in texts)
    assert any(s.level == "error" for s in view.status)


def test_detail_payout_timing_months_accounts_and_time_split(result):
    detail = present_pair_detail(result, "S0_D160", "takeprofittrader")
    assert detail.completed and detail.rank_text == "1"
    assert ("Higher-timeframe gap charts", "one-hour and four-hour") in detail.settings
    facts = {f.label: f.value for f in detail.facts}
    assert facts["First payout received"] == "January 15, 2026 4:00 PM CST"
    assert facts["Largest unrecovered account spending"].startswith("$102.00 (reached")
    assert facts["Money pending at the cutoff"] == "None"
    assert facts["Stop exits filled worse than the stop"] == "1 of 1 ($20.00 worse in total)"
    # every month, including zero months
    assert [m.label for m in detail.months] == [
        "January 2026 (partial month)", "February 2026", "March 2026 (partial month)"]
    assert detail.months[1].display()["Payouts received"] == "$0.00"
    # replacement history
    first, second = detail.accounts
    assert first.payout_outcome == "Lost after receiving a payout"
    assert first.failure_reason == "Open-position equity reached the loss limit"
    assert second.replaces == "Replaces account 1" and second.payout_outcome == (
        "Still open at the end")
    assert first.payouts[0].display()["Received after the split"] == "$2,407.78"
    assert _AMPM.match(first.created) and _AMPM.match(first.lost_when)
    # time split separates payout pauses from missing signals
    reasons = {t.reason: t.hours_text for t in detail.time_split}
    assert reasons["Payout processing — paused until the payment arrives"] == "48.0 hours"
    assert "Ready, market open, no entry" in reasons
    assert {f.label: f.value for f in detail.refused}[
        "Entries refused during payout processing"] == "1"
    # trades: Chicago AM/PM, points, words, stop worse than the stop
    stop = detail.trades[2]
    assert _AMPM.match(stop.entry_time) and stop.close_reason == "Protective stop hit"
    assert stop.entry_price == "25,000.00" and stop.exit_price == "24,979.00"
    assert stop.stop_worse == "Yes, 1.00 points worse"
    assert stop.price_source == "Recorded exchange trades"
    assert detail.trades[0].stop_worse == "—"
    # cumulative cash goes to the cutoff
    assert detail.cash[-1].net_cash_usd == 2203.78


def test_detail_zero_payout_pending_and_approximation(result):
    alpha_mff = present_pair_detail(result, "S0_D160", "myfundedfutures")
    assert "No payouts were received in this period." in alpha_mff.notices
    assert alpha_mff.accounts[0].payout_outcome == "Lost before any payout"
    assert alpha_mff.trades[0].price_source == "One-minute approximation for 3 of 40 minutes"
    assert alpha_mff.trades[0].approximate
    beta_mff = present_pair_detail(result, "S3_D80", "myfundedfutures")
    pending = {f.label: f.value for f in beta_mff.facts}["Money pending at the cutoff"]
    assert pending == "$800.75 after the split ($889.72 gross, 1 request still processing)"
    assert beta_mff.accounts[0].payouts[0].state == "Processing, not received at the end"
    beta_tpt = present_pair_detail(result, "S3_D80", "takeprofittrader")
    assert beta_tpt.trades == () and "No trades were taken in this period." in beta_tpt.notices


def test_not_completed_detail_and_plain_reason(result):
    detail = present_pair_detail(result, "S9_D40", "takeprofittrader")
    assert not detail.completed and detail.months == () and detail.trades == ()
    assert "not a zero result" in detail.notices[0]
    assert plain_reason("KeyError: missing C:\\data\\x.parquet file") == (
        "Missing (internal location omitted) file.")
    assert completed_configurations(result) == [
        ("S0_D160", "Alpha · original three windows · distance 160 ticks (40 points)"),
        ("S3_D80", "Beta · New York morning only · distance 80 ticks (20 points)")]
    with pytest.raises(KeyError):
        present_pair_detail(result, "S0_D160", "unknown_firm")


def test_presenter_text_has_no_internal_identities(result):
    view = present_comparison(result)
    parts = [view.question, view.period_text, view.firms_text, view.size_text,
             view.evidence_text, *(s.text for s in view.status), *view.limitations]
    for table in view.firm_tables:
        parts += [table.note, *(r.reason for r in table.rows)]
        parts += [v for r in table.rows for v in r.display().values()]
    for configuration, _ in completed_configurations(result):
        for firm in ("takeprofittrader", "myfundedfutures"):
            d = present_pair_detail(result, configuration, firm)
            parts += [f.value for f in (*d.headline, *d.facts)]
            parts += [v for t in d.trades for v in t.display().values()]
            parts += [v for a in d.accounts for v in a.display().values()]
            parts += [v for a in d.accounts for p in a.payouts for v in p.display().values()]
            parts += [p.what for a in d.accounts for p in a.balance_path]
    assert _clean("\n".join(parts)) == []


# ── Streamlit screen ─────────────────────────────────────────────────────


def _comparison_app():
    import ifvg_funded_comparison_results
    import streamlit as st

    ifvg_funded_comparison_results.render_funded_comparison_results(
        st, ifvg_funded_comparison_results._TEST_RESULT)


def _run(monkeypatch, result):
    import ifvg_funded_comparison_results
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_funded_comparison_results, "_TEST_RESULT", result, raising=False)
    at = AppTest.from_function(_comparison_app, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _visible_text(at) -> str:
    values = []
    for kind in ("markdown", "caption", "warning", "info", "error", "success", "subheader"):
        values.extend(str(item.value) for item in at.get(kind))
    for widget in (*at.selectbox, *at.radio):
        values.append(str(widget.label))
        values.extend(str(option) for option in widget.options)
    for metric in at.metric:
        values.extend([metric.label, str(metric.value), str(metric.proto.help)])
    for table in at.dataframe:
        values.append(table.value.to_csv(index=False))
    for table in at.table:
        values.append(table.value.to_csv())
    for tab in at.tabs:
        values.append(str(tab.label))
    return "\n".join(values).replace("\\$", "$")


def test_screen_renders_comparison_with_separate_firm_tabs(monkeypatch, result):
    at = _run(monkeypatch, result)
    assert [t.label for t in at.tabs] == ["TakeProfitTrader", "MyFundedFutures"]
    tpt = at.tabs[0].dataframe[0].value
    mff = at.tabs[1].dataframe[0].value
    # the settings that differ between configurations are separate readable columns
    assert list(tpt.columns) == [
        "Rank", "Net cash earned", "Entry hours", "Largest opposing distance",
        "Payouts received after the split", "Account costs", "Largest payout",
        "Accounts purchased", "Lost before a payout", "Lost after a payout"]
    assert tpt["Net cash earned"].tolist() == ["$2,203.78", "-$102.00", "Not completed"]
    assert mff["Net cash earned"].tolist() == ["-$125.00", "-$250.00", "Not completed"]
    assert mff["Entry hours"].tolist()[:2] == ["New York morning only",
                                               "Original three windows"]
    assert mff["Largest opposing distance"].tolist()[0] == "80 ticks (20 points)"
    assert mff["Entry hours"].tolist()[2].startswith("Gamma")  # not completed: its name
    text = _visible_text(at)
    assert "Recorded prices for 2026-02-03 are unavailable." in text
    assert FUTURE_NOTE in text
    assert "220 of 240 position minutes used recorded exchange trades" in text
    assert "Engineering sample" in text
    assert "One historical path." in text  # material limitation visible


def test_screen_never_sums_configurations_or_firms(monkeypatch, result):
    text = _visible_text(_run(monkeypatch, result))
    # TakeProfitTrader configurations, MyFundedFutures configurations, and each
    # configuration across firms — none of these sums may appear anywhere.
    for total in ("$2,101.78", "-$375.00", "$1,953.78", "-$227.00"):
        assert total not in text
    assert not re.search(r"\b(grand total|portfolio total|combined)\b", text, re.IGNORECASE)


def test_screen_text_has_no_hashes_ids_or_paths(monkeypatch, result):
    at = _run(monkeypatch, result)
    assert _clean(_visible_text(at)) == []
    at.selectbox(key="funded_comparison_v1_configuration").set_value("S3_D80").run()
    at.radio(key="funded_comparison_v1_firm").set_value("myfundedfutures").run()
    assert not at.exception
    assert _clean(_visible_text(at)) == []


def test_screen_detail_selection_shows_the_selected_pair(monkeypatch, result):
    at = _run(monkeypatch, result)
    text = _visible_text(at)
    assert "rank 1 for this firm" in text and "Alpha" in text
    assert "Lost after receiving a payout" in text
    assert "Yes, 1.00 points worse" in text
    months = next(df.value for df in at.dataframe if "Month" in df.value.columns)
    assert months["Month"].tolist() == [
        "January 2026 (partial month)", "February 2026", "March 2026 (partial month)"]
    at.selectbox(key="funded_comparison_v1_configuration").set_value("S3_D80").run()
    at.radio(key="funded_comparison_v1_firm").set_value("myfundedfutures").run()
    text = _visible_text(at)
    assert "$800.75 after the split ($889.72 gross, 1 request still processing)" in text
    assert "Processing, not received at the end" in text
    at.radio(key="funded_comparison_v1_firm").set_value("takeprofittrader").run()
    text = _visible_text(at)
    assert "No trades were taken in this period." in text
    assert not at.exception


def test_screen_with_no_completed_configuration(monkeypatch, result):
    variant = dict(result)
    variant["tables"] = {**result["tables"], "configurations": []}
    at = _run(monkeypatch, variant)
    assert "No configuration completed, so there is no detail to show." in _visible_text(at)
