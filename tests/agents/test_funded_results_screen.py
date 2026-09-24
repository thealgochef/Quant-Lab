"""Completed funded-payout results screen (SYNTHETIC sample results only).

The results rendered here are hand-made engineering samples. Passing these
tests proves the screen renders every state; it is not evidence that a
historical simulation completed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from alpha_lab.agents.data_infra.ifvg.presentation.funded_results import (  # noqa: E402
    HEADLINE_KEYS,
    HEADLINE_SENTENCE,
    format_points,
    format_usd,
    headline_figures,
    present_funded_result,
)
from tests.propsim.funded.builders import synthetic_sample_result  # noqa: E402
from tests.propsim.funded.sample_variants import all_failed_no_payout_result  # noqa: E402


@pytest.fixture(scope="module")
def sample():
    return synthetic_sample_result()


@pytest.fixture(scope="module")
def failed_sample():
    return all_failed_no_payout_result()


def _funded_app():
    import ifvg_funded_results
    import streamlit as st

    ifvg_funded_results.render_funded_results(st, ifvg_funded_results._TEST_RESULT)


def _run(monkeypatch, result):
    import ifvg_funded_results
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(ifvg_funded_results, "_TEST_RESULT", result, raising=False)
    at = AppTest.from_function(_funded_app, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def _visible_text(at) -> str:
    values = []
    for kind in ("markdown", "caption", "warning", "info", "error", "success", "subheader"):
        values.extend(str(item.value) for item in at.get(kind))
    for widget in at.selectbox:
        values.extend(str(option) for option in widget.options)
    for metric in at.metric:
        values.extend([metric.label, str(metric.value), str(metric.proto.help)])
    for table in at.dataframe:
        values.append(table.value.to_csv(index=False))
    for tab in at.tabs:
        values.append(str(tab.label))
    return "\n".join(values).replace("\\$", "$")


_HASH = re.compile(r"\b[0-9a-f]{64}\b")
_PATH = re.compile(r"[A-Za-z]:\\|\\\\|/Users/|/home/|\.py\b|\.json\b|\.parquet\b")


# ── presenter ────────────────────────────────────────────────────────────


def test_presenter_reads_exact_cents_and_formats(sample):
    view = present_funded_result(sample)
    assert view.headline == HEADLINE_SENTENCE
    assert view.is_sample and view.validation_passed
    figures = headline_figures(view)
    for key, summary in sample["summaries_cents"].items():
        assert figures[key] == {k: summary[k] for k in HEADLINE_KEYS}
    assert [f.firm for f in view.firms] == ["TakeProfitTrader", "MyFundedFutures"]
    assert format_usd(-10_200) == "-$102.00"
    assert format_usd(3_244_450) == "$32,444.50"
    assert format_points(100_001) == "25,000.25"
    assert "Engineering sample — not a historical result" in " ".join(
        s.text for s in view.status)
    # A processing account shows the Chicago date/time it resumes.
    processing = [a for f in view.firms for a in f.accounts if a.status_key == "processing"]
    assert processing and all(
        re.search(r"trading paused until \w+ \d{1,2}, \d{4} \d{1,2}:\d{2} (AM|PM)", a.status)
        for a in processing)
    # Trades carry 12-hour Chicago times and prices in points.
    trade = view.firms[0].trades[0]
    assert re.fullmatch(r"\w+ \d{1,2}, \d{4} \d{1,2}:\d{2} (AM|PM) C[DS]T", trade.entry_time)
    assert trade.entry_price == "25,000.00"
    assert trade.close_reason == "Profit target reached"


def test_presenter_approximation_status_is_visible(sample):
    variant = dict(sample)
    variant["purpose"] = "pilot_validation"
    variant["price_evidence"] = {**sample["price_evidence"],
                                 "policy": "ordered_trade_prints_with_labeled_minute_fallback",
                                 "trades_with_minute_approximation": 3,
                                 "approximate_failures": 1}
    view = present_funded_result(variant)
    text = " ".join(s.text for s in view.status)
    assert "Approximation" in text and "not a historical result" not in text


def test_presenter_failed_validation_is_an_error(sample):
    variant = dict(sample)
    variant["validation"] = {"passed": False, "checks": {}}
    view = present_funded_result(variant)
    assert any(s.level == "error" for s in view.status)


# ── rendered screen ──────────────────────────────────────────────────────


def test_screen_renders_sample_with_cards_limitations_and_status(monkeypatch, sample):
    at = _run(monkeypatch, sample)
    text = _visible_text(at)
    assert HEADLINE_SENTENCE in text
    assert "TakeProfitTrader" in text and "MyFundedFutures" in text
    labels = [m.label for m in at.metric]
    for label in ("Net cash earned after all account costs",
                  "Payouts received after the firm's share",
                  "Largest single received payout", "Account costs", "Accounts lost"):
        assert labels.count(label) == 2, label
    values = {m.label: [] for m in at.metric}
    for m in at.metric:
        values[m.label].append(m.value)
    assert values["Net cash earned after all account costs"] == ["$32,444.50", "$36,243.75"]
    assert "Engineering sample — not a historical result" in text
    assert "Simulated result" in text
    for limitation in sample["limitations"]:
        assert limitation in text
    assert "What this result cannot tell you" in text
    # Pending cash is shown and explicitly not received.
    assert "still processing at the end" in text
    assert "NOT received cash" in text
    # No grand total across firms.
    assert "combined" not in text.lower() and "grand total" not in text.lower()
    assert not _HASH.search(text), "raw hash visible"
    assert not _PATH.search(text), "file path visible"
    assert not at.json and not at.code


def test_account_and_trade_selection(monkeypatch, sample):
    at = _run(monkeypatch, sample)
    key = "funded_results_v1_account_takeprofittrader"
    box = at.selectbox(key=key)
    lost = next(o for o in box.options if "Account lost" in o)
    box.select(_option_value(sample, lost)).run()
    assert not at.exception
    text = _visible_text(at)
    assert "Reason: Open-position equity reached the loss limit" in text
    assert "It was lost before any payout." in text
    trade_box = at.selectbox(key="funded_results_v1_trade_takeprofittrader")
    assert len(trade_box.options) == 2
    trade_box.select(1).run()
    assert not at.exception
    text = _visible_text(at)
    assert "Account loss limit reached — position closed and account lost" in text
    assert "Initial risk (entry to stop)" in text
    assert "Ordered exchange trade prints" in text
    # Switching the account resets the trade choice without errors.
    at.selectbox(key=key).select("takeprofittrader-001").run()
    assert not at.exception
    assert at.selectbox(key="funded_results_v1_trade_takeprofittrader").value == 0
    assert not _HASH.search(_visible_text(at))


def _option_value(result, label):
    view = present_funded_result(result)
    for account in view.firms[0].accounts:
        if f"{account.label} — {account.status}" == label:
            return account.account_id
    raise AssertionError(label)


def test_all_failed_no_payout_state_is_explicit(monkeypatch, failed_sample):
    at = _run(monkeypatch, failed_sample)
    text = _visible_text(at)
    assert text.count("No payouts were received in this period.") == 2
    assert "Every account was lost by the end of the period." in text
    assert "Account costs exceeded received payouts" in text
    largest = [m.value for m in at.metric if m.label == "Largest single received payout"]
    assert largest == ["No payout received", "No payout received"]
    lost = [m.value for m in at.metric if m.label == "Accounts lost"]
    assert lost == ["5 before first payout · 0 after a payout"] * 2
    assert not _HASH.search(text) and not _PATH.search(text)
    box = at.selectbox(key="funded_results_v1_account_myfundedfutures")
    box.select("myfundedfutures-003").run()
    assert not at.exception
    assert "This account received no payouts." in _visible_text(at)
