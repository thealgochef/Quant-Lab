"""Analytical corrections A6, A7, A8 and A10 in Trade review (September 25, 2026).

A6 setup identity, A7 point in time including the early-January case, A8
independent reviewer judgments, A10 the published approximated-minute companion.

Synthetic cases use small one-minute bar frames (at most 60 rows), synthetic
setup packages and published folders built in pytest's ``tmp_path``, and review
ledgers in ``tmp_path`` only — never the real ``data/ifvg_visual_review`` ledger.
Reference tests read the saved funded result, its verified strategy package and
the published review folder through the existing readers (read only) and are
skipped where they are absent. Nothing here launches, approves or saves a study.

For the evidence file, every synthetic regression case is a module-level constant
(``SETUP_IDENTITY_CASES``, ``POINT_IN_TIME_CASES``, ``REVIEW_JUDGMENT_CASES``,
``MINUTE_COMPANION_CASES``, ``FOLDER_CASES``) with ``id``, ``inputs`` and
``expected``; ``setup_identity_actual``, ``point_in_time_actual``,
``minute_companion_actual`` and ``review_judgment_actual`` recompute the actual
outputs. ``REFERENCE_EARLY_JANUARY`` and ``REFERENCE_MINUTE_LINK`` record the
real-study checks.
"""

from __future__ import annotations

import contextlib
import copy
import csv
import hashlib
import html
import io
import json
import re
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab import minute_companion as mc
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_chart as rc
from alpha_lab.agents.data_infra.ifvg.presentation.lab import review_panels as rp
from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import (
    Gap,
    SetupRecord,
    entry_match_key,
    find_setup_record,
    load_setup_record_source,
)

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
PLAN_ID = "78406b15cdd9c1363b7cf626f7e45d48e58f6aa4e634a25dc00a3e57bdb28336"
LEADER = "S1-T1-H14-P1-L-SO"
TPT, MFF = "takeprofittrader", "myfundedfutures"
HAVE_STUDY = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()
PUBLISHED = REPO / "reports/funded_comparison/funded_comparison_5fa65149843484b1_export_v4"


def _utc(text: str) -> pd.Timestamp:
    stamp = pd.Timestamp(text)
    return stamp.tz_convert("UTC") if stamp.tzinfo else stamp.tz_localize("UTC")


def _markup_text(markup: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", str(markup)))


# ══ A6 — setup identity ═══════════════════════════════════════════════════

_ENTRY = "2026-04-13 00:07:00+00:00"
_FAMILY = "fresh_fvg_continuation"
_MEMBERS = {
    "S1_D80_W1_P1": {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
                     "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m"},
    "S1_D80_W1_P0": {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
                     "parent_timeframes": "parent_timeframes.3m-5m-10m-15m-30m"},
    "S0_D80_W1_P1": {"enabled_entry_sessions": "enabled_entry_sessions.asia-london-ny",
                     "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m"},
}
_VARIANT_IDS = {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
                "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m",
                "exit_policy": "exit_policy.scale_out_half_breakeven_hold_to_close_v1"}
_FUNDED_TRADE = {"strategy_trade_id": "variant-trade", "direction": "long",
                 "entry_utc": "2026-04-13T00:07:00Z", "entry_ticks": 99884, "stop_ticks": 99799,
                 "target_ticks": 99969}


def _package_row(profile: str, trade_id: str, *, htf_low: int = 97806, htf_high: int = 100347,
                 tap: str = "2026-04-12 22:18:00+00:00",
                 parent_confirmed: str = "2026-04-13 00:00:00+00:00",
                 inversion: str = "2026-04-13 00:05:00+00:00") -> dict:
    """One saved execution of a synthetic verified package (same entry key and stop)."""

    inv = pd.Timestamp(inversion)
    return {
        "profile": profile, "trade_id": trade_id, "setup_id": f"setup-{trade_id}",
        "is_warmup": False, "direction": "LONG", "entry_family": _FAMILY,
        "entry_ts_utc": _ENTRY, "entry_ticks": 99884, "stop_ticks": 99799, "target_ticks": 99969,
        "resolution": "target", "entry_match_key": entry_match_key(_ENTRY, "LONG", _FAMILY, 99884),
        "exact_execution_key": "x",
        "geometry_htf_timeframe_seconds": 14400, "geometry_htf_direction": "bullish",
        "geometry_htf_gap_low_ticks": htf_low, "geometry_htf_gap_high_ticks": htf_high,
        "geometry_htf_size_ticks": htf_high - htf_low,
        "geometry_htf_a_open_ts_utc": "2026-04-07 18:00:00+00:00",
        "geometry_htf_confirmed_ts_utc": "2026-04-08 06:00:00+00:00",
        "geometry_parent_timeframe_seconds": 300, "geometry_parent_direction": "bullish",
        "geometry_parent_gap_low_ticks": 99749, "geometry_parent_gap_high_ticks": 99866,
        "geometry_parent_size_ticks": 117,
        "geometry_parent_a_open_ts_utc": "2026-04-12 23:45:00+00:00",
        "geometry_parent_confirmed_ts_utc": parent_confirmed,
        "geometry_opposing_timeframe_seconds": 60, "geometry_opposing_direction": "bearish",
        "geometry_opposing_gap_low_ticks": 99847, "geometry_opposing_gap_high_ticks": 99850,
        "geometry_opposing_size_ticks": 3,
        "geometry_opposing_a_open_ts_utc": "2026-04-13 00:00:00+00:00",
        "geometry_opposing_confirmed_ts_utc": "2026-04-13 00:03:00+00:00",
        "geometry_tap_bar_logical_open_ts_utc": tap,
        "geometry_tap_bar_logical_close_ts_utc": str(pd.Timestamp(tap) + pd.Timedelta(minutes=1)),
        "geometry_inversion_bar_logical_open_ts_utc": str(inv),
        "geometry_inversion_bar_logical_close_ts_utc": str(inv + pd.Timedelta(minutes=1)),
    }


def _package(root: Path, rows: list[dict]) -> Path:
    (root / "data").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "data/trades.csv", index=False)
    (root / "run_context.json").write_text(json.dumps({"frozen_batch": {"members": [
        {"name": name, "spec": {"axis_value_ids": ids}} for name, ids in _MEMBERS.items()]}}),
        encoding="utf-8")
    (root / "configs.json").write_text(json.dumps({
        name: {"section": {"entry_family": _FAMILY}} for name in _MEMBERS}), encoding="utf-8")
    files = [{"path": rel, "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
             for rel in ("data/trades.csv", "run_context.json", "configs.json")]
    (root / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return root


#: A6 regression cases: two saved executions share the funded trade's entry key and stop
SETUP_IDENTITY_CASES = [
    {"id": "same_entry_different_zone",
     "inputs": {"configuration": LEADER, "in_verified_study": False,
                "funded_trade": _FUNDED_TRADE, "axis_value_ids": _VARIANT_IDS,
                "rows": [{"profile": "S1_D80_W1_P1", "trade_id": "p", "htf_low": 97806},
                         {"profile": "S0_D80_W1_P1", "trade_id": "q", "htf_low": 97000}]},
     "expected": {"link": "same_execution", "identity_established": False,
                  "configuration": "S1_D80_W1_P1", "htf_low_ticks": 97806,
                  "same_choice_in_reversed_row_order": True,
                  "step_label": "Higher-timeframe gap is valid",
                  "moments_equal_no_record": True}},
    {"id": "same_entry_different_prior_history",
     "inputs": {"configuration": LEADER, "in_verified_study": False,
                "funded_trade": _FUNDED_TRADE, "axis_value_ids": _VARIANT_IDS,
                "rows": [{"profile": "S1_D80_W1_P0", "trade_id": "r",
                          "tap": "2026-04-12 21:02:00+00:00",
                          "parent_confirmed": "2026-04-12 23:55:00+00:00",
                          "inversion": "2026-04-13 00:04:00+00:00"},
                         {"profile": "S0_D80_W1_P1", "trade_id": "s",
                          "tap": "2026-04-12 22:18:00+00:00",
                          "parent_confirmed": "2026-04-13 00:00:00+00:00",
                          "inversion": "2026-04-13 00:05:00+00:00"}]},
     # both differ from the funded configuration in two settings: the name breaks the tie
     "expected": {"link": "same_execution", "identity_established": False,
                  "configuration": "S0_D80_W1_P1", "htf_low_ticks": 97806,
                  "same_choice_in_reversed_row_order": True,
                  "step_label": "Higher-timeframe gap is valid",
                  "moments_equal_no_record": True}},
    {"id": "exact_member_match",
     "inputs": {"configuration": "S1_D80_W1_P1", "in_verified_study": True,
                "funded_trade": {**_FUNDED_TRADE, "strategy_trade_id": "own"},
                "axis_value_ids": _MEMBERS["S1_D80_W1_P1"],
                "rows": [{"profile": "S1_D80_W1_P1", "trade_id": "own", "htf_low": 97806},
                         {"profile": "S0_D80_W1_P1", "trade_id": "other", "htf_low": 97000}]},
     "expected": {"link": "exact", "identity_established": True,
                  "configuration": "S1_D80_W1_P1", "htf_low_ticks": 97806,
                  "same_choice_in_reversed_row_order": True,
                  "step_label": "Four-hour gap is valid",
                  "moments_equal_no_record": False}},
]


def _identity_record(case: dict, root: Path, *, reverse: bool = False) -> SetupRecord | None:
    rows = [_package_row(r["profile"], r["trade_id"], htf_low=r.get("htf_low", 97806),
                         **{k: r[k] for k in ("tap", "parent_confirmed", "inversion") if k in r})
            for r in case["inputs"]["rows"]]
    source = load_setup_record_source(_package(root, rows[::-1] if reverse else rows))
    inputs = case["inputs"]
    return find_setup_record(inputs["funded_trade"], configuration=inputs["configuration"],
                             source=source, axis_value_ids=inputs["axis_value_ids"],
                             in_verified_study=inputs["in_verified_study"])


def setup_identity_actual(case: dict, root: Path) -> dict:
    """The actual A6 outputs of one :data:`SETUP_IDENTITY_CASES` case (package in ``root``)."""

    record = _identity_record(case, root / "forward")
    reverse = _identity_record(case, root / "reverse", reverse=True)
    view = rp.TradeView.from_row(ROW)
    return {"link": record.link, "identity_established": record.identity_established,
            "configuration": record.configuration, "htf_low_ticks": record.htf.low_ticks,
            "same_choice_in_reversed_row_order": reverse == record,
            "step_label": rp.step_label("htf", record),
            "moments_equal_no_record": rp.moments(view, record) == rp.moments(view, None)}


@pytest.mark.parametrize("case", SETUP_IDENTITY_CASES, ids=[c["id"] for c in SETUP_IDENTITY_CASES])
def test_a6_setup_identity_cases(case, tmp_path):
    """A6: matching entry time, direction, entry price and stop gives related context only.

    Same entry with a different zone, or with a different prior history, links a
    related record (identity not established) chosen deterministically — the same
    whatever the package's row order; only the configuration's own record by trade
    id establishes identity.
    """

    assert setup_identity_actual(case, tmp_path) == case["expected"]


def test_a6_ties_on_the_same_configuration_are_broken_by_trade_id(tmp_path):
    """A6: two records of one configuration with the same entry key and stop (different prior
    history) always give the same related record, whatever their order in the package."""

    rows = [_package_row("S1_D80_W1_P1", "b", tap="2026-04-12 21:00:00+00:00"),
            _package_row("S1_D80_W1_P1", "a", tap="2026-04-12 22:18:00+00:00")]
    chosen = []
    for name, order in (("one", rows), ("two", rows[::-1])):
        source = load_setup_record_source(_package(tmp_path / name, order))
        chosen.append(find_setup_record(_FUNDED_TRADE, configuration=LEADER, source=source,
                                        axis_value_ids=_VARIANT_IDS))
    assert chosen[0] == chosen[1] and chosen[0].trade_id == "a"
    assert not chosen[0].identity_established


# ── the trade and a hand-built record for the presentation checks ─────────

ROW = {
    "seq": 380, "account_number": 6, "direction": "long", "quantity": 10,
    "entry_utc": "2026-04-13T00:07:00Z", "entry_ticks": 99884, "stop_ticks": 99799,
    "target_ticks": 99969, "scale_out_ns": 1776039027251840803, "scale_out_ticks": 99969,
    "scale_out_quantity": 5, "final_exit_quantity": 5, "final_stop_ticks": 99884,
    "exit_utc": "2026-04-13T20:55:00Z", "exit_ticks": 102305, "exit_kind": "scheduled_close",
    "net_pnl_usd": 6254.72, "costs_usd": 10.28, "balance_before_usd": 1821.10,
    "balance_after_usd": 8075.82, "initial_risk_usd": 425.0, "trading_day": "2026-04-13",
    "minutes_on_prints": 1247, "minutes_approximated": 1, "approximate_exit": False,
    "account_failed": False, "pair_id": f"CFG|{TPT}", "trade_ref": "ref-380",
}


def _record(link: str, configuration: str, *, entry: str = "2026-04-13T00:07:00Z",
            tap: str = "2026-04-12T22:18:00Z", parent_first: str = "2026-04-12T23:45:00Z",
            parent_confirmed: str = "2026-04-13T00:00:00Z",
            opposing_confirmed: str = "2026-04-13T00:03:00Z",
            inversion: str = "2026-04-13T00:05:00Z", entry_ticks: int = 99884,
            stop_ticks: int = 99799) -> SetupRecord:
    """A saved setup (four-hour, five-minute parent, one-minute opposing gap), built by hand."""

    minute = pd.Timedelta(minutes=1)
    gaps = {
        "htf": Gap("htf", 14400, "bullish", 97806, 100347, 2541, _utc("2026-01-05T18:00:00Z"),
                   _utc("2026-01-06T06:00:00Z")),
        "parent": Gap("parent", 300, "bullish", entry_ticks - 135, entry_ticks - 18, 117,
                      _utc(parent_first), _utc(parent_confirmed)),
        "opposing": Gap("opposing", 60, "bearish", entry_ticks - 37, entry_ticks - 34, 3,
                        _utc(parent_confirmed), _utc(opposing_confirmed)),
    }
    bars = {"tap_bar": (_utc(tap), _utc(tap) + minute),
            "inversion_bar": (_utc(inversion), _utc(inversion) + minute)}
    return SetupRecord(link=link, configuration=configuration, trade_id="t", setup_id="s",
                       gaps=gaps, bars=bars, entry_utc=_utc(entry), entry_ticks=entry_ticks,
                       stop_ticks=stop_ticks, target_ticks=None)


def _bars(start: str, count: int, *, base: float, trading_day: str) -> pd.DataFrame:
    opens = pd.date_range(start, periods=count, freq="1min", tz="UTC")
    price = [base + (i % 7) * 0.5 + i * 0.25 for i in range(count)]
    return pd.DataFrame({
        "trading_day": trading_day, "logical_open_ts_utc": opens,
        "logical_close_ts_utc": opens + pd.Timedelta(minutes=1), "open": price,
        "high": [p + 1.5 for p in price], "low": [p - 1.5 for p in price],
        "close": [p + 0.75 for p in price]})


def _text(lines) -> list[str]:
    return [f"{line.label} | {re.sub('<[^>]+>', '', str(line.text))}" for line in lines]


class _FakeSt:
    """Enough of ``st`` for the cards: containers are plain context managers."""

    def container(self, **_kwargs):
        import contextlib

        return contextlib.nullcontext()


def test_a6_related_record_is_labelled_context_and_keeps_every_execution_fact(monkeypatch):
    """A6: a related record is labelled context everywhere, never names the first review step
    or adds point-in-time moments, and leaves the recorded fills, stops, exit, result and
    balances (and every chart trace) exactly as without a record."""

    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    view = rp.TradeView.from_row(ROW)
    related = _record("same_execution", "S1_D80_W1_P1")
    own = _record("exact", "CFG")
    assert not related.identity_established and own.identity_established
    assert related.source_sentence == (
        "Related context from configuration S1_D80_W1_P1's saved setup record for the same "
        "entry time, direction, entry price and stop, and target; exact setup identity for this "
        "configuration is not established.")
    assert own.source_sentence == "Zones come from this configuration's own saved setup record."
    assert rp.gap_legend(related) == "Related four-hour gap (context)"
    assert rp.gap_legend(own) == "Four-hour gap"
    assert rp.setup_card_title(related) == "Related setup context · 1-minute candles"
    assert rp.setup_card_title(own) == rp.setup_card_title(None) == "The setup · 1-minute candles"
    assert rp.formed_card_title(related) == "Related setup context"
    assert rp.formed_card_title(own) == "How the setup formed"
    assert rp.related_source_line(related) == (
        "From configuration S1_D80_W1_P1's record; not this configuration's own formation "
        "history.")
    assert rp.related_source_line(own) is None and rp.related_source_line(None) is None
    assert rp.step_label("htf", related) == "Higher-timeframe gap is valid"
    assert rp.step_label("htf", own) == "Four-hour gap is valid"
    assert rp.moments(view, related) == rp.moments(view, None)
    assert len(rp.moments(view, own)) == len(rp.moments(view, None)) + 4  # own setup steps
    # this configuration's distance limit is never claimed for another configuration's gap
    assert "limit" not in " ".join(s.text for s in rp.setup_steps(related, view, distance_cap=80))
    assert "inside the 80-tick limit" in " ".join(
        s.text for s in rp.setup_steps(own, view, distance_cap=80))
    # execution facts: identical with no record and with a related record
    assert _text(rp.recorded_lines(view, instrument="micro")) == [
        "7:07 PM | Bought 10 micros at 24,971.00", "Initial stop | 24,949.75 · risk $425.00",
        "7:10 PM | Sold 5 at 24,992.25 · target reached, stop on the rest moved to entry",
        "Apr 13, 3:55 PM | Sold 5 at 25,576.25 · scheduled daily close",
        "Result | +$6,254.72 after $10.28 costs", "Account 6 balance | $1,821.10 → $8,075.82"]
    bars = _bars("2026-04-12T23:40:00Z", 60, base=24_960.0, trading_day="2026-04-13")
    with_related = rc.whole_trade_figure(bars, view, related, size_minutes=10)
    without = rc.whole_trade_figure(bars, view, None, size_minutes=10)
    assert with_related.data == without.data  # fills, stops, target, candles and result dots
    zone = next(a.text for a in with_related.layout.annotations if "gap" in (a.text or ""))
    assert zone.startswith("<b>Related four-hour gap (context) ")
    result = [a.text for a in with_related.layout.annotations if "Trade result" in (a.text or "")]
    assert result == [a.text for a in without.layout.annotations
                      if "Trade result" in (a.text or "")]
    # the page's cards
    shown: list[str] = []
    monkeypatch.setattr(ifvg_lab_ui, "show",
                        lambda markup, st_module=None: shown.append(str(markup)))
    study = types.SimpleNamespace(plan=object())
    review._whole_trade_card(_FakeSt(), None, view, related, related.source_sentence,
                             moment=None, size=10, zones=True, stops=True, times=True,
                             close_clock=None, instrument=None, key="t")
    review._setup_card(_FakeSt(), None, view, related, moment=None, key="t")
    review._formed_card(_FakeSt(), study, "cfg", view, related, moment=None)
    text = _markup_text("\n".join(shown))
    for words in ("Related four-hour gap (context)", "Related setup context · 1-minute candles",
                  "From configuration S1_D80_W1_P1's record; not this configuration's own "
                  "formation history.", "exact setup identity for this configuration is not "
                  "established."):
        assert words in text, words
    assert "How the setup formed" not in text and "The setup · 1-minute candles" not in text


# ══ A7 — point in time, including the early-January case ══════════════════

_PARTS = ["All open-market hours · Long only · Half at 1R", "1R", "TakeProfitTrader"]


def _trade(seq: int, account: int, entry: str, **extra) -> dict:
    """A later/earlier trade of the pair: only what the hidden views read."""

    entry_at = _utc(entry)
    return {"seq": seq, "account_number": account, "direction": "long", "quantity": 1,
            "entry_utc": entry, "entry_ticks": 100000, "stop_ticks": 99960,
            "target_ticks": 100040, "exit_utc": (entry_at + pd.Timedelta(minutes=30)).isoformat(),
            "exit_ticks": 100040, "exit_kind": "target", "net_pnl_usd": 195.0,
            "trading_day": "", "pair_id": f"CFG|{TPT}", **extra}


_JANUARY_TRADE = {
    "seq": 4, "account_number": 1, "direction": "long", "quantity": 1,
    "entry_utc": "2026-01-13T04:31:00Z", "entry_ticks": 100000, "stop_ticks": 99962,
    "target_ticks": 100038, "exit_utc": "2026-01-13T04:41:57.695030175Z", "exit_ticks": 99962,
    "exit_kind": "stop", "final_exit_quantity": 1, "final_stop_ticks": 99962,
    "net_pnl_usd": -190.28, "costs_usd": 0.28, "balance_before_usd": 0.0,
    "balance_after_usd": -190.28, "initial_risk_usd": 190.0, "trading_day": "2026-01-13",
    "minutes_on_prints": 11, "minutes_approximated": 0, "approximate_exit": False,
    "account_failed": False, "pair_id": f"CFG|{TPT}", "trade_ref": "ref-4",
}

#: A7 regression cases: the hidden view at the cursor never changes when later records change
POINT_IN_TIME_CASES = [
    {"id": "early_january_first_trade_only_account_1",
     "inputs": {
         "reviewed_seq": 4,
         "trades": [_JANUARY_TRADE,
                    _trade(9, 1, "2026-01-13T23:02:00Z"),
                    _trade(20, 2, "2026-01-16T15:00:00Z"),
                    _trade(25, 3, "2026-01-22T15:00:00Z")],
         "accounts_opened": {1: "2026-01-12T23:00:00Z", 2: "2026-01-15T19:56:07.305287043Z",
                             3: "2026-01-21T19:04:07.036586151Z"},
         "bars": {"start": "2026-01-13T04:10:00Z", "count": 40, "base": 24_990.0,
                  "trading_day": "2026-01-13"},
         "record": {"link": "same_execution", "configuration": "S1_D80_W1_P1",
                    "entry": "2026-01-13T04:31:00Z", "tap": "2026-01-13T03:40:00Z",
                    "parent_first": "2026-01-13T04:15:00Z",
                    "parent_confirmed": "2026-01-13T04:25:00Z",
                    "opposing_confirmed": "2026-01-13T04:28:00Z",
                    "inversion": "2026-01-13T04:29:00Z", "entry_ticks": 100000,
                    "stop_ticks": 99962},
     },
     "expected": {"cursor": "2026-01-13T04:35:00+00:00", "cursor_label": "10:35 PM",
                  "known_accounts": [1], "known_trades": [4],
                  "context_line_end": "Account 1 · trade 1 at this firm",
                  "trade_labels": ["Jan 12, 10:31 PM · result hidden"],
                  "chart_window": ["Jan 12, 5:00 PM", "Jan 13, 4:00 PM"],
                  "half_fill_hidden": None, "every_variant_identical": True,
                  "differing": {}}},
    {"id": "april_half_exit_after_the_cursor",
     "inputs": {
         "reviewed_seq": 380,
         "trades": [_trade(100, 5, "2026-03-10T15:00:00Z"),
                    _trade(200, 6, "2026-04-01T14:00:00Z"),
                    {**ROW, "account_number": 6},
                    _trade(399, 6, "2026-04-16T19:01:00Z"),
                    _trade(450, 7, "2026-05-01T14:00:00Z")],
         "accounts_opened": {5: "2026-02-11T16:27:15.569565547Z",
                             6: "2026-03-17T18:21:05.574327279Z", 7: "2026-04-20T15:00:00Z"},
         "bars": {"start": "2026-04-12T23:50:00Z", "count": 40, "base": 24_960.0,
                  "trading_day": "2026-04-13"},
         "record": {"link": "exact", "configuration": "CFG"},
     },
     "expected": {"cursor": "2026-04-13T00:10:00+00:00", "cursor_label": "7:10 PM",
                  "known_accounts": [5, 6], "known_trades": [100, 200, 380],
                  "context_line_end": "Account 6 · trade 3 at this firm",
                  "trade_labels": ["Mar 10, 10:00 AM · result hidden",
                                   "Apr 1, 9:00 AM · result hidden",
                                   "Apr 12, 7:07 PM · result hidden"],
                  "chart_window": ["Apr 12, 5:00 PM", "Apr 13, 4:00 PM"],
                  "half_fill_hidden": "2026-04-13T00:10:27.251840803+00:00",
                  "every_variant_identical": True, "differing": {}}},
]


def _world(case: dict) -> dict:
    inputs = case["inputs"]
    spec = inputs["bars"]
    return {"trades": copy.deepcopy(inputs["trades"]),
            "opened": {int(a): _utc(t) for a, t in inputs["accounts_opened"].items()},
            "bars": _bars(spec["start"], spec["count"], base=spec["base"],
                          trading_day=spec["trading_day"]),
            "tables": {}}


def _cursor(case: dict) -> pd.Timestamp:
    row = next(t for t in case["inputs"]["trades"] if t["seq"] == case["inputs"]["reviewed_seq"])
    return rp.default_moment(rp.TradeView.from_row(row))


def _reviewed(world: dict, case: dict) -> dict:
    return next(t for t in world["trades"] if t["seq"] == case["inputs"]["reviewed_seq"])


def _later_bars(world: dict, cursor: pd.Timestamp, change) -> dict:
    bars = world["bars"]
    later = bars["logical_close_ts_utc"] > cursor
    world["bars"] = change(bars, later)
    return world


def _changed_bars(bars, later):
    out = bars.copy()
    out.loc[later, ["high", "close"]] += 200.0
    out.loc[later, "low"] -= 150.0
    return out


def _appended_bars(bars, _later):
    last = bars["logical_open_ts_utc"].max()
    extra = _bars((last + pd.Timedelta(minutes=1)).isoformat(), 20,
                  base=float(bars["close"].iloc[-1]) + 300.0,
                  trading_day=str(bars["trading_day"].iloc[0]))
    return pd.concat([bars, extra], ignore_index=True)


def _later_trade_list(world: dict, cursor: pd.Timestamp, keep) -> dict:
    world["trades"] = [t for t in world["trades"] if _utc(t["entry_utc"]) <= cursor or keep(t)]
    return world


def _renumber(world: dict, cursor: pd.Timestamp) -> dict:
    later = {a for a, at in world["opened"].items() if at > cursor}
    world["opened"] = {(a + 10 if a in later else a): at for a, at in world["opened"].items()}
    for trade in world["trades"]:
        if trade["account_number"] in later:
            trade["account_number"] += 10
    return world


def _fail_after_cursor(world: dict, case: dict, cursor: pd.Timestamp) -> dict:
    row = _reviewed(world, case)
    exit_at = cursor + pd.Timedelta(minutes=47)
    row.update(account_failed=True, exit_kind="account_failure", exit_utc=exit_at.isoformat(),
               exit_ticks=row["stop_ticks"] - 40, net_pnl_usd=-2_210.0, scale_out_ns=None,
               scale_out_ticks=None, scale_out_quantity=None, balance_after_usd=-1_310.0)
    new = max(world["opened"]) + 1
    world["opened"][new] = exit_at + pd.Timedelta(hours=3)
    world["tables"] = {
        "rule_boundary_evidence": [{"pair_id": row["pair_id"], "check": "account_failure",
                                    "account_number": row["account_number"],
                                    "trade_ref": row.get("trade_ref"),
                                    "ts_utc": exit_at.isoformat(), "equity_usd": -2_000.0,
                                    "floor_usd": -2_000.0, "comparator": "at_or_below"}],
        "account_events": [{"pair_id": row["pair_id"], "event": "created",
                            "account_number": new, "replaces": row["account_number"],
                            "ts_utc": (exit_at + pd.Timedelta(hours=3)).isoformat()}]}
    return world


def _variants(case: dict) -> dict:
    """Future records changed in every way the hidden view must not notice."""

    cursor = _cursor(case)

    def reviewed(update):
        def apply(world):
            _reviewed(world, case).update(update(_reviewed(world, case)))
            return world
        return apply

    def later_exit(row):
        at = _utc(row["exit_utc"]) + pd.Timedelta(hours=2)
        return {"exit_utc": at.isoformat(), "exit_ticks": row["entry_ticks"] + 300,
                "exit_kind": "target", "net_pnl_usd": 1_234.56}

    def other_half(row):
        if row.get("scale_out_ns") is not None:  # April: no half exit at all
            return {"scale_out_ns": None, "scale_out_ticks": None, "scale_out_quantity": None,
                    "final_exit_quantity": row["quantity"]}
        at = cursor + pd.Timedelta(minutes=2, seconds=13)  # January: a half exit appears
        return {"scale_out_ns": int(at.value), "scale_out_ticks": row["target_ticks"],
                "scale_out_quantity": 1, "final_stop_ticks": row["entry_ticks"],
                "exit_utc": (at + pd.Timedelta(hours=1)).isoformat()}

    extra_account = max(case["inputs"]["accounts_opened"]) + 3
    return {
        "bars_after_cursor_changed": lambda w: _later_bars(w, cursor, _changed_bars),
        "bars_appended": lambda w: _later_bars(w, cursor, _appended_bars),
        "bars_truncated_at_cursor": lambda w: _later_bars(w, cursor, lambda b, later: b[~later]),
        "exit_changed": reviewed(later_exit),
        "half_exit_changed": reviewed(other_half),
        "result_and_balance_changed": reviewed(lambda r: {
            "net_pnl_usd": 99_999.99, "balance_after_usd": 123_456.78, "costs_usd": 42.0}),
        "account_failed_after_cursor": lambda w: _fail_after_cursor(w, case, cursor),
        "later_trades_added": lambda w: {**w, "trades": w["trades"] + [
            _trade(900, int(_reviewed(w, case)["account_number"]),
                   (cursor + pd.Timedelta(days=3)).isoformat())]},
        "later_trades_removed": lambda w: _later_trade_list(w, cursor, lambda t: False),
        "later_accounts_added": lambda w: {
            **w, "opened": {**w["opened"], extra_account: cursor + pd.Timedelta(days=5)},
            "trades": w["trades"] + [_trade(950, extra_account,
                                            (cursor + pd.Timedelta(days=6)).isoformat())]},
        "later_accounts_removed": lambda w: {
            **w, "opened": {a: at for a, at in w["opened"].items() if at <= cursor},
            "trades": [t for t in w["trades"] if t["account_number"] in
                       {a for a, at in w["opened"].items() if at <= cursor}]},
        "later_accounts_renumbered": lambda w: _renumber(w, cursor),
    }


def _case_record(case: dict) -> SetupRecord:
    spec = dict(case["inputs"]["record"])
    return _record(spec.pop("link"), spec.pop("configuration"), **spec)


def hidden_view(case: dict, world: dict) -> dict:
    """Everything Trade review shows at the cursor (point in time), for one world of records."""

    cursor = _cursor(case)
    record = _case_record(case)
    views = [rp.TradeView.from_row(t) for t in world["trades"]]
    view = next(v for v in views if v.seq == case["inputs"]["reviewed_seq"])
    row = _reviewed(world, case)
    day = rc.day_minutes(world["bars"], view.trading_day)
    order = [v.seq for v in views]
    key = rp.setup_key(record, view, moment=cursor)
    known = rp.known_trades(views, cursor, view.seq)
    accounts = sorted({v.account for v in views if v.account is not None})
    return {
        "whole_trade_figure": rc.whole_trade_figure(day, view, record, moment=cursor).to_json(),
        "setup_figure": rc.setup_figure(day, view, record, moment=cursor, key=key)[0].to_json(),
        "chart_window": rc.chart_window(day, view, moment=cursor),
        "setup_window": rc.setup_window(view, record, moment=cursor),
        "recorded_lines": [(line.label, str(line.text), line.at)
                           for line in rp.recorded_lines(view, instrument="micro",
                                                         moment=cursor)],
        "failure_lines": rp.failure_lines(world["tables"], row, moment=cursor),
        "setup_steps": [(s.number, s.text, s.at, s.time_text, s.bold)
                        for s in rp.setup_steps(record, view, distance_cap=80, moment=cursor)],
        "setup_key": key,
        "moments": [(m.at, m.label) for m in rp.moments(view, record)],
        "default_moment": rp.default_moment(view),
        "step_label": rp.step_label("htf", record),
        "whole_trade_caption": rc.whole_trade_caption(view, moment=cursor),
        "known_accounts": rp.known_accounts(accounts, world["opened"], cursor, view.account,
                                            trades=views),
        "known_trades": [v.seq for v in known],
        "context_line": rp.context_line(_PARTS, view.account, order.index(view.seq) + 1,
                                        len(order), point_in_time=True),
        "trade_labels": [rp.trade_label(v, hide_result=True, account=True) for v in known],
    }


def point_in_time_actual(case: dict) -> dict:
    """The actual A7 outputs of one :data:`POINT_IN_TIME_CASES` case."""

    base = hidden_view(case, _world(case))
    differing = {}
    for name, change in _variants(case).items():
        changed = hidden_view(case, change(_world(case)))
        differing[name] = sorted(k for k in base if changed[k] != base[k])
    view = rp.TradeView.from_row(next(t for t in case["inputs"]["trades"]
                                      if t["seq"] == case["inputs"]["reviewed_seq"]))
    cursor = _cursor(case)
    half = None
    if view.half_utc is not None:
        texts = [base["whole_trade_figure"], base["setup_figure"], base["whole_trade_caption"],
                 json.dumps([list(map(str, x)) for x in base["recorded_lines"]]),
                 json.dumps([list(map(str, x)) for x in base["setup_key"]])]
        wall = rc._wall(view.half_utc).strftime("%H:%M:%S")
        assert view.half_utc > cursor
        assert not any(word in t for t in texts
                       for word in ("Half", "half came out", "Sold 5", wall))
        half = view.half_utc.isoformat()
    return {"cursor": cursor.isoformat(), "cursor_label": rp.clock(cursor),
            "known_accounts": base["known_accounts"], "known_trades": base["known_trades"],
            "context_line_end": " · ".join(base["context_line"].rsplit(" · ", 2)[-2:]),
            "trade_labels": base["trade_labels"],
            "chart_window": [rp.short(t) for t in base["chart_window"]],
            "half_fill_hidden": half,
            "every_variant_identical": not any(differing.values()),
            "differing": {k: v for k, v in differing.items() if v}}


@pytest.mark.parametrize("case", POINT_IN_TIME_CASES, ids=[c["id"] for c in POINT_IN_TIME_CASES])
def test_a7_future_records_never_change_the_hidden_view(case):
    """A7: at the cursor, changing later bars (changed, appended, truncated), the trade's exit,
    half exit, result and balance, a later loss-limit failure, later trades and later accounts
    (added, removed, renumbered) leaves every hidden item identical — figures, bounds, recorded
    lines, setup steps and key, moments, default moment, caption, known accounts and trades,
    context line and trade labels. Early January: only Account 1 exists at the cursor."""

    actual = point_in_time_actual(case)
    assert actual == case["expected"]  # "differing" names any item that changed


def test_a7_the_april_half_fill_instant_is_kept_exactly():
    """A7: the recorded half-exit fill (7:10:27.251840803 PM) is never moved: it stays after the
    7:10:00 PM cursor, so the half is hidden there and shown from its own instant on."""

    view = rp.TradeView.from_row(ROW)
    assert view.half_utc == pd.Timestamp(1776039027251840803, unit="ns", tz="UTC")
    assert view.half_utc.value == ROW["scale_out_ns"]
    cursor = rp.default_moment(view)
    assert cursor == _utc("2026-04-13T00:10:00Z") < view.half_utc
    assert not any("Sold 5" in t for t in _text(rp.recorded_lines(view, moment=cursor)))
    assert any("Sold 5" in t for t in _text(rp.recorded_lines(view, moment=view.half_utc)))


def test_a7_the_old_bar_window_would_have_leaked_later_bars():
    """A7: in Full history the window still follows the stored bars (so appending later bars
    moves it); point in time uses the scheduled 5:00 PM → 4:00 PM day whatever the bars."""

    case = POINT_IN_TIME_CASES[1]
    world = _world(case)
    view = rp.TradeView.from_row(ROW)
    cursor = _cursor(case)
    day = rc.day_minutes(world["bars"], view.trading_day)
    longer = rc.day_minutes(_appended_bars(world["bars"], None), view.trading_day)
    assert rc.chart_window(day, view) != rc.chart_window(longer, view)  # full history
    assert rc.chart_window(day, view, moment=cursor) == rc.chart_window(longer, view,
                                                                         moment=cursor)
    assert rc.chart_window(day, view, moment=cursor) == rc.trading_window(None, "2026-04-13")
    assert rc.scheduled_window(rp.TradeView.from_row({**ROW, "trading_day": ""})) == (
        rc.scheduled_window(view))


def test_a7_a_later_loss_limit_and_its_replacement_stay_hidden():
    """A7: a loss-limit check and a replacement account after the moment are left out."""

    case = POINT_IN_TIME_CASES[0]
    cursor = _cursor(case)
    world = _fail_after_cursor(_world(case), case, cursor)
    row = _reviewed(world, case)
    assert rp.failure_lines(world["tables"], row, moment=cursor) == []
    after = rp.failure_lines(world["tables"], row,
                             moment=_utc(row["exit_utc"]) + pd.Timedelta(minutes=1))
    assert [label for label, _ in after] == ["Loss limit"]
    assert [label for label, _ in rp.failure_lines(world["tables"], row)] == [
        "Loss limit", "Replacement"]


_OPENED = {1: "2026-01-12T23:00:00Z", 2: "2026-01-15T19:56:07Z", 3: "2026-01-21T19:04:07Z"}
_LISTED_TRADES = [(4, 1, "2026-01-13T04:31:00Z"), (18, 1, "2026-01-15T08:22:00Z"),
                  (38, 2, "2026-01-16T09:03:00Z")]

#: A7 account-listing cases: the Account picker lists only accounts with a trade by the moment
ACCOUNT_LISTING_CASES = [
    {"id": "account_opened_minutes_before_the_moment_without_a_trade",
     "inputs": {"accounts": [1, 2, 3], "opened": _OPENED, "trades": _LISTED_TRADES,
                "moment": "2026-01-15T20:00:00Z", "current": 1},  # Jan 15, 2:00 PM Chicago
     "expected": {"with_trades": [1], "opening_time_only": [1, 2]}},
    {"id": "account_with_a_trade_by_the_moment",
     "inputs": {"accounts": [1, 2, 3], "opened": _OPENED, "trades": _LISTED_TRADES,
                "moment": "2026-01-16T10:00:00Z", "current": 2},
     "expected": {"with_trades": [1, 2], "opening_time_only": [1, 2]}},
    {"id": "the_reviewed_trades_account_is_always_kept",
     "inputs": {"accounts": [1, 2, 3], "opened": _OPENED, "trades": _LISTED_TRADES,
                "moment": "2026-01-12T23:30:00Z", "current": 1},  # before any entry
     "expected": {"with_trades": [1], "opening_time_only": [1]}},
]


def account_listing_actual(case: dict) -> dict:
    inputs = case["inputs"]
    views = [rp.TradeView.from_row(_trade(seq, account, entry))
             for seq, account, entry in inputs["trades"]]
    opened = {int(a): _utc(t) for a, t in inputs["opened"].items()}
    moment = _utc(inputs["moment"])
    return {"with_trades": rp.known_accounts(inputs["accounts"], opened, moment,
                                             inputs["current"], trades=views),
            "opening_time_only": rp.known_accounts(inputs["accounts"], opened, moment,
                                                   inputs["current"])}


@pytest.mark.parametrize("case", ACCOUNT_LISTING_CASES,
                         ids=[c["id"] for c in ACCOUNT_LISTING_CASES])
def test_a7_an_account_is_listed_only_with_a_trade_by_the_moment(case):
    """A7: in point in time the Account picker lists an account only when it had a trade
    entered by the moment (or holds the reviewed trade), so choosing it always opens a trade;
    the opening-time-only rule (kept for other callers) would list an empty account."""

    assert account_listing_actual(case) == case["expected"]


def test_a6_no_gap_swatch_without_a_zone_and_related_hover_is_qualified(monkeypatch):
    """A6: the whole-trade legend shows no gap swatch when no zone is saved; the setup chart's
    opposing-gap hover names a related record's gap as related context."""

    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    assert rp.gap_legend(None) is None
    no_htf = _record("same_execution", "S1_D80_W1_P1")
    no_htf = SetupRecord(**{**no_htf.__dict__, "gaps": {k: v for k, v in no_htf.gaps.items()
                                                         if k != "htf"}})
    assert rp.gap_legend(no_htf) is None
    shown: list[str] = []
    monkeypatch.setattr(ifvg_lab_ui, "show",
                        lambda markup, st_module=None: shown.append(str(markup)))
    view = rp.TradeView.from_row(ROW)
    review._whole_trade_card(_FakeSt(), None, view, None, rp.no_record_sentence(), moment=None,
                             size=10, zones=True, stops=True, times=True, close_clock=None,
                             instrument=None, key="t")
    text = _markup_text("\n".join(shown))
    assert "Four-hour gap" not in text and "gap (context)" not in text
    assert "Up candle" in text and "In the trade" in text
    assert "Setup zones weren't recorded for this study" in text
    # a thin opposing gap is drawn as a line with a hover naming it
    bars = _bars("2026-04-12T23:40:00Z", 60, base=24_900.0, trading_day="2026-04-13")
    bars["high"] += 60.0  # a wide price range: the three-tick gap is drawn as a line
    for link, words in (("same_execution", "Related opposing gap 24,961.75–24,962.50 (context)"),
                        ("exact", "Opposing gap 24,961.75–24,962.50")):
        record = _record(link, "S1_D80_W1_P1")
        fig, _ = rc.setup_figure(bars, view, record, key=rp.setup_key(record, view))
        hovers = [t.text[0] for t in fig.data if getattr(t, "text", None)
                  and "pposing gap" in str(t.text[0])]
        assert hovers == [words], hovers


# ── the real study: the first leader trade (early January) ────────────────

#: A7 real-study check (skipped where the saved study is absent)
REFERENCE_EARLY_JANUARY = {
    "id": "leader_first_trade_takeprofittrader",
    "inputs": {"result_id": RESULT_ID, "configuration": LEADER, "firm": "TakeProfitTrader",
               "trade": "first trade of the pair (seq 4)"},
    "expected": {"entry": "Jan 12, 10:31 PM", "trading_day": "2026-01-13",
                 "default_moment": "Jan 12, 10:35 PM", "account": 1, "known_accounts": [1],
                 "known_trades": [4], "all_accounts": [1, 2, 3, 4, 5, 6], "trades": 114,
                 "context_line_end": "Account 1 · trade 1 at this firm"},
}


def _real_study():
    if not HAVE_STUDY:
        pytest.skip("the saved funded variation study is not on this computer")
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
        ordered_trades,
    )

    study = open_funded_study(STORE, RESULT_ID)
    return study, list(ordered_trades(study, LEADER, TPT))


def reference_early_january_actual() -> dict:
    """The actual values of :data:`REFERENCE_EARLY_JANUARY` (same keys; needs the saved study)."""

    import ifvg_lab_trade_review as review

    study, rows = _real_study()
    views = [rp.TradeView.from_row(r) for r in rows]
    first = views[0]
    moment = rp.default_moment(first)
    accounts = sorted({v.account for v in views})
    opened = review._accounts_opened(study, LEADER, TPT)
    line = rp.context_line(_PARTS, first.account, 1, len(views), point_in_time=True)
    return {"entry": rp.short(first.entry_utc), "trading_day": first.trading_day,
            "default_moment": rp.short(moment), "account": first.account,
            "known_accounts": rp.known_accounts(accounts, opened, moment, first.account,
                                                trades=views),
            "known_trades": [v.seq for v in rp.known_trades(views, moment, first.seq)],
            "all_accounts": accounts, "trades": len(views),
            "context_line_end": " · ".join(line.rsplit(" · ", 2)[-2:])}


def test_a7_real_first_leader_trade_knows_only_account_1():
    """A7: at the first leader trade's default moment (the first five-minute mark after the
    January 12, 10:31 PM Chicago entry, trading day January 13) only Account 1 exists, only
    that trade is listed, and the context line has no total."""

    actual = reference_early_january_actual()
    assert actual == REFERENCE_EARLY_JANUARY["expected"]
    assert "of 114" not in actual["context_line_end"]


# ── the page (AppTest) ────────────────────────────────────────────────────


def _app():
    import ifvg_lab_trade_review as page
    import streamlit as st

    page.render_trade_review_page(st, page._TEST_ROOTS)


_TARGET = {"result_id": RESULT_ID, "store_root": str(STORE), "app": "ifsm",
           "study_key": PLAN_ID, "plan_id": PLAN_ID,
           "name": "Funded variation study — 64 configurations", "status": "Completed"}


@contextlib.contextmanager
def page_environment(repo_root: Path):
    """Trade review on the saved result, headless, with ``repo_root`` as its (isolated) repo:
    reviews and published folders are read and written there only. Restores everything."""

    import ifvg_lab_trade_review as page
    import ifvg_lab_ui

    def fake_clickable(markup, *, key, st_module=None):  # components v2 needs a browser
        import streamlit as st

        (st_module or st).html(str(markup))
        return None

    saved = (ifvg_lab_ui.clickable, page._funded_targets, getattr(page, "_TEST_ROOTS", None))
    ifvg_lab_ui.clickable = fake_clickable
    page._funded_targets = lambda st_module, roots: [dict(_TARGET)]
    page._TEST_ROOTS = {"store_root": STORE, "repo_root": Path(repo_root),
                        "state_root": Path(repo_root) / "jobs",
                        "draft_root": Path(repo_root) / "drafts"}
    try:
        yield
    finally:
        ifvg_lab_ui.clickable, page._funded_targets = saved[0], saved[1]
        if saved[2] is None:
            del page._TEST_ROOTS
        else:
            page._TEST_ROOTS = saved[2]


@pytest.fixture
def page(tmp_path):
    if not HAVE_STUDY:
        pytest.skip("the saved funded variation study is not on this computer")
    with page_environment(tmp_path):
        yield {"target": dict(_TARGET), "ledger": tmp_path}


def _run(state: dict):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app, default_timeout=90)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def _link(seq: int, account: int, *, point: bool, firm: str = TPT) -> dict:
    return {"ifvg_lab_v1_review_target": {
        "source": "Funded trades", "result_id": RESULT_ID, "configuration": LEADER,
        "firm_key": firm, "account_number": account, "trade_seq": seq, "point_in_time": point,
        "back_tab": "Trades"}}


def _page_text(at) -> str:
    parts = []
    for kind in ("markdown", "caption", "warning", "info", "error", "html"):
        try:
            parts += [str(getattr(e, "body", None) or getattr(e, "value", ""))
                      for e in at.get(kind)]
        except Exception:
            continue
    return html.unescape("\n".join(parts))


_SCOPE = RESULT_ID[:16]


_CONTEXT_LINE = re.compile(r"[^\n<>]*· Account \d+ · trade \d+(?: of \d+)? at this firm")


def _context_line(at) -> str:
    """The context line above the charts (``… · Account 6 · trade 77 at this firm``)."""

    found = _CONTEXT_LINE.search(_page_text(at))
    return found.group(0).strip() if found else ""


def _pickers(at) -> dict:
    trade = at.selectbox(key=f"ifvg_lab_v1_review_trade_{_SCOPE}")
    account = at.selectbox(key=f"ifvg_lab_v1_review_account_{_SCOPE}")
    moment = at.selectbox(key=f"ifvg_lab_v1_review_moment_{_SCOPE}_{LEADER}_{TPT}_{trade.value}")
    return {"accounts": list(account.options), "account": account.value,
            "trades": list(trade.options), "trade": trade.value,
            "moments": list(moment.options), "moment": moment.value,
            "context": _context_line(at)}


def _helps(at) -> list[str]:
    widgets = [*at.selectbox, *at.button, *at.radio, *at.checkbox, *at.text_input,
               *at.text_area]
    return [str(getattr(w, "help", "") or "") for w in widgets]


def test_a7_page_first_trade_in_point_in_time_shows_only_account_1(page):
    """A7: opening the first leader trade in point in time offers "All accounts" and
    "Account 1" only, one trade, no total, an enabled "›", and no count or result in any
    help text."""

    at = _run(_link(4, 1, point=True))
    assert not at.exception, at.exception
    shown = _pickers(at)
    assert shown["accounts"] == ["All accounts", "Account 1"]
    assert shown["trades"] == ["Jan 12, 10:31 PM · result hidden"]
    assert shown["moment"] == "2026-01-13T04:35:00+00:00"
    assert shown["context"].endswith("Account 1 · trade 1 at this firm")
    text = _page_text(at)
    assert "of 114" not in text and "-$190.28" not in text and "−$190.28" not in text
    assert at.button(key="ifvg_lab_v1_review_next").disabled is False
    for text in _helps(at):
        assert "114" not in text and "$" not in text, text
        assert not re.search(r"\bAccount [2-9]\b|\d+ (trades|accounts)\b", text), text
    at.selectbox(key=f"ifvg_lab_v1_review_account_{_SCOPE}").set_value("all").run()
    assert not at.exception, at.exception
    assert _pickers(at)["trades"] == ["Jan 12, 10:31 PM · result hidden"]


def test_a7_page_full_history_first_leaks_nothing_into_point_in_time(page):
    """A7: opening Full history first and then switching to point in time gives the same
    options, selections and text as opening point in time directly."""

    direct = _run(_link(4, 1, point=True))
    assert not direct.exception, direct.exception
    via = _run(_link(4, 1, point=False))
    assert not via.exception, via.exception
    accounts = list(via.selectbox(key=f"ifvg_lab_v1_review_account_{_SCOPE}").options)
    assert len(accounts) == 7  # Full history: every account
    via.radio(key="ifvg_lab_v1_review_mode").set_value("Point in time").run()
    assert not via.exception, via.exception
    assert _pickers(via) == _pickers(direct)
    assert _page_text(via) == _page_text(direct)
    assert _helps(via) == _helps(direct)


def test_a7_page_next_is_a_deliberate_step_forward_in_time(page):
    """A7: "›" opens the next trade at its own, later moment (a deliberate step in time); the
    earlier cursor's options were never widened by it."""

    at = _run(_link(4, 1, point=True))
    first = _pickers(at)
    at.button(key="ifvg_lab_v1_review_next").click().run()
    assert not at.exception, at.exception
    second = _pickers(at)
    assert second["trade"] == 9 and pd.Timestamp(second["moment"]) > pd.Timestamp(first["moment"])
    assert second["moment"] == "2026-01-13T23:05:00+00:00"  # Jan 13, 5:05 PM Chicago
    assert second["accounts"] == ["All accounts", "Account 1"]  # Account 2 opened Jan 15
    assert second["context"].endswith("Account 1 · trade 2 at this firm")


# ── page regression cases (review follow-up; the saved result, isolated repo root) ──

_ACCOUNT_KEY = f"ifvg_lab_v1_review_account_{_SCOPE}"
_TRADE_KEY = f"ifvg_lab_v1_review_trade_{_SCOPE}"
_CONTEXT_KEY = "funded_comparison_v1_selected_context"
_JANUARY_LABELS = ["Jan 12, 10:31 PM · result hidden", "Jan 13, 5:02 PM · result hidden",
                   "Jan 13, 6:54 PM · result hidden", "Jan 14, 1:35 AM · result hidden",
                   "Jan 15, 2:22 AM · result hidden"]
_FIRST_TRADE_VIEW = {"mode": "Point in time", "accounts": ["All accounts", "Account 1"],
                     "trades": ["Jan 12, 10:31 PM · result hidden"], "trade": 4,
                     "moment": "2026-01-13T04:35:00+00:00",
                     "context_end": "Account 1 · trade 1 at this firm", "note": None,
                     "exception": False}
_EVERY_ACCOUNT = ["All accounts", *[f"Account {n}" for n in range(1, 7)]]
_END_OF_STUDY = ("There is no later recorded trade for this configuration at this firm: the "
                 "clock moved to the end of the study, so everything is shown.")
_STEP_ANSWERS = {"htf": "Agree", "Parent gap and retest are right": "Disagree",
                 "Entry and stop are right": "Agree", "Exit handling is right": "Unclear"}

#: page-level regression cases on the saved result (AppTest, isolated repo root); the actual
#: (:func:`page_case_actual`) returns every expected key plus the other observed values
PAGE_CASES = [
    {"id": "april_trade_then_account_1",
     "inputs": {"link": {"seq": 380, "account": 6, "point": True},
                "actions": [["account", 1]]},
     "expected": dict(_FIRST_TRADE_VIEW)},
    {"id": "april_trade_then_shared_account_1",
     "inputs": {"link": {"seq": 380, "account": 6, "point": True},
                "actions": [["shared_account", 1]]},
     "expected": dict(_FIRST_TRADE_VIEW)},
    {"id": "april_trade_then_an_earlier_trade_from_all_accounts",
     "inputs": {"link": {"seq": 380, "account": 6, "point": True},
                "actions": [["account", "all"], ["trade", 4]]},
     "expected": {**_FIRST_TRADE_VIEW}},
    {"id": "trade_18_at_2_pm_lists_no_account_without_a_trade",
     "inputs": {"link": {"seq": 18, "account": 1, "point": True},
                "actions": [["moment", "2026-01-15T20:00:00+00:00"], ["account", "all"]]},
     "expected": {"mode": "Point in time", "accounts": ["All accounts", "Account 1"],
                  "trades": _JANUARY_LABELS, "trade": 18,
                  "moment": "2026-01-15T20:00:00+00:00",
                  "context_end": "Account 1 · trade 5 at this firm", "note": None,
                  "exception": False}},
    {"id": "last_trade_next_moves_the_clock_to_the_end_of_the_study",
     "inputs": {"link": {"seq": 582, "account": 6, "point": True}, "actions": [["next"]]},
     "expected": {"mode": "Full history", "accounts": _EVERY_ACCOUNT, "trade": 582,
                  "trade_count": 56, "moment": None,
                  "context_end": "Account 6 · trade 114 of 114 at this firm",
                  "note": _END_OF_STUDY, "exception": False}},
    {"id": "last_trade_save_and_next_moves_the_clock_to_the_end_of_the_study",
     "inputs": {"link": {"seq": 582, "account": 6, "point": True},
                "actions": [["save_next", {"overall": "correct"}]]},
     "expected": {"mode": "Full history", "accounts": _EVERY_ACCOUNT, "trade": 582,
                  "trade_count": 56, "moment": None,
                  "context_end": "Account 6 · trade 114 of 114 at this firm",
                  "note": _END_OF_STUDY, "saved": 1, "exception": False}},
    {"id": "no_link_point_in_time_opens_the_plans_first_configuration",
     "inputs": {"link": None, "state": {"ifvg_lab_v1_review_mode_value": "Point in time"},
                "actions": []},
     "expected": {"mode": "Point in time", "configuration": "S0-T1-H1-P1-L-FX", "trade": 4,
                  "accounts": ["All accounts", "Account 1"], "exception": False}},
    {"id": "no_link_full_history_opens_the_ranking_leader",
     "inputs": {"link": None, "state": {"ifvg_lab_v1_review_mode_value": "Full history"},
                "actions": []},
     "expected": {"mode": "Full history", "configuration": LEADER, "exception": False}},
    {"id": "related_record_disables_the_formation_steps",
     "inputs": {"link": {"seq": 380, "account": 6, "point": False},
                "actions": [["save", {"overall": "correct", "steps": _STEP_ANSWERS}]]},
     "expected": {"disabled_steps": ["Higher-timeframe gap is valid",
                                     "Parent gap and retest are right"],
                  "disabled_help": [
                      "Needs this configuration's own setup record. The zones shown are related "
                      "context from another configuration, so this step can't be judged here "
                      "and nothing is saved for it."] * 2,
                  "saved_verdicts": [{"overall_verdict": "correct", "htf_verdict": None,
                                      "parent_verdict": None, "entry_verdict": "correct",
                                      "stop_verdict": "correct",
                                      "outcome_verdict": "insufficient_evidence"}],
                  "exception": False}},
]


def _act(at, action: list) -> None:
    """One owner action on the page (a picker, "›", a save)."""

    kind, *args = action
    if kind == "account":
        at.selectbox(key=_ACCOUNT_KEY).set_value(args[0]).run()
    elif kind == "trade":
        at.selectbox(key=_TRADE_KEY).set_value(args[0]).run()
    elif kind == "moment":
        seq = at.selectbox(key=_TRADE_KEY).value
        at.selectbox(key=f"ifvg_lab_v1_review_moment_{_SCOPE}_{LEADER}_{TPT}_{seq}").set_value(
            args[0]).run()
    elif kind == "shared_account":  # another screen chose this account for the pair
        contexts = {k: dict(v) for k, v in dict(at.session_state[_CONTEXT_KEY]).items()}
        contexts.setdefault(RESULT_ID, {})["account"] = {"pair": f"{LEADER}|{TPT}",
                                                         "number": args[0]}
        at.session_state[_CONTEXT_KEY] = contexts
        at.run()
    elif kind == "next":
        at.button(key="ifvg_lab_v1_review_next").click().run()
    elif kind in ("save", "save_next"):
        form = args[0]
        next(r for r in at.radio if r.label == "Overall").set_value(form["overall"]).run()
        at.text_input(key="ifvg_lab_v1_review_reviewer").input("Test reviewer").run()
        for step, choice in (form.get("steps") or {}).items():
            # set even a disabled control: the page must still save nothing for it
            next(s for s in at.selectbox
                 if (s.key or "").endswith(f"step_{step[:12]}")).set_value(choice)
        label = "Save review" if kind == "save" else "Save and next trade"
        next(b for b in at.button if b.label == label).click().run()
    else:
        raise ValueError(kind)


def _observe(at, repo_root: Path) -> dict:
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    out: dict = {"exception": bool(at.exception)}
    if at.exception:
        out["error"] = str(at.exception)
        return out
    text = _page_text(at)
    trade = at.selectbox(key=_TRADE_KEY)
    mode = at.radio(key="ifvg_lab_v1_review_mode").value
    moment_key = f"ifvg_lab_v1_review_moment_{_SCOPE}_"
    moments = [s for s in at.selectbox if (s.key or "").startswith(moment_key)]
    context = _context_line(at)
    steps = [s for s in at.selectbox if "_step_" in (s.key or "")]
    saved = list_reviews(repo_root=Path(repo_root))
    fields = ("overall_verdict", "htf_verdict", "parent_verdict", "entry_verdict",
              "stop_verdict", "outcome_verdict")
    out.update({
        "mode": mode,
        "configuration": at.selectbox(key=f"ifvg_lab_v1_review_config_{_SCOPE}").value,
        "accounts": list(at.selectbox(key=_ACCOUNT_KEY).options),
        "trades": list(trade.options), "trade_count": len(trade.options), "trade": trade.value,
        "moment": moments[0].value if moments else None,
        "context_end": " · ".join(context.rsplit(" · ", 2)[-2:]),
        "note": _END_OF_STUDY if _END_OF_STUDY in text else None,
        "disabled_steps": [s.label for s in steps if s.disabled],
        "disabled_help": [s.help for s in steps if s.disabled],
        "saved": len(saved),
        "saved_verdicts": [{f: (r.get(f) if isinstance(r.get(f), str) else None)
                            for f in fields} for _, r in saved.iterrows()],
    })
    return out


def page_case_actual(case: dict, repo_root: Path) -> dict:
    """The actual page values of one :data:`PAGE_CASES` case (reviews go to ``repo_root``)."""

    with page_environment(repo_root):
        state = dict(case["inputs"].get("state") or {})
        link = case["inputs"].get("link")
        if link:
            state.update(_link(link["seq"], link["account"], point=link["point"]))
        at = _run(state)
        for action in case["inputs"]["actions"]:
            if at.exception:
                break
            _act(at, action)
        return _observe(at, repo_root)


@pytest.mark.parametrize("case", PAGE_CASES, ids=[c["id"] for c in PAGE_CASES])
def test_a7_page_regression_cases(case, tmp_path):
    """A7/A6 (review follow-up): in point in time, choosing another account (on the page or
    through the shared selection) or another trade rebuilds every picker from THAT trade's own
    moment; an account with no trade by the moment is never listed (trade 18 at Jan 15,
    2:00 PM); "›" or "Save and next trade" at the last trade moves the clock to the study's end
    and says so; without a link point in time opens the plan's first configuration (Full
    history the ranking leader); a related record disables the gap and parent steps and
    nothing is saved for them."""

    if not HAVE_STUDY:
        pytest.skip("the saved funded variation study is not on this computer")
    actual = page_case_actual(case, tmp_path)
    assert _expected_part(actual, case["expected"]) == case["expected"], actual.get("error")


def test_a7_page_an_empty_account_shows_a_note_instead_of_crashing(page, monkeypatch):
    """A7: even if an account without a trade by the moment were listed (the old opening-time
    rule, forced here), choosing it shows a note and never crashes."""

    real = rp.known_accounts
    monkeypatch.setattr(rp, "known_accounts",
                        lambda accounts, opened, moment, current, *, trades=None:
                        real(accounts, opened, moment, current))
    at = _run(_link(18, 1, point=True))
    _act(at, ["moment", "2026-01-15T20:00:00+00:00"])
    assert list(at.selectbox(key=_ACCOUNT_KEY).options) == [
        "All accounts", "Account 1", "Account 2"]
    _act(at, ["account", 2])
    assert not at.exception, at.exception
    assert "Account 2 has no trade entered by 2:00 PM. Choose another account." in (
        _page_text(at))
    _act(at, ["account", 1])
    assert not at.exception, at.exception
    assert at.selectbox(key=_TRADE_KEY).value == 18


# ══ A8 — independent reviewer judgments ═══════════════════════════════════

_JUDGED = {"pair_id": f"CFG-A|{TPT}", "configuration": "CFG-A", "firm_key": TPT,
           "account_number": 6, "seq": 380, "strategy_trade_id": "trade-a"}
_ALL_NOT_REVIEWED = {step: "Not reviewed" for step, _ in rp.STEPS}

#: A8 regression cases (isolated tmp ledgers only)
REVIEW_JUDGMENT_CASES = [
    {"id": "mixed_judgment_round_trip",
     "inputs": {"result_id": "r" * 64, "plan_id": "p" * 64, "trade": _JUDGED,
                "seed": {"verdicts": {"overall_verdict": "correct", "entry_verdict": "correct",
                                      "stop_verdict": "incorrect"}, "notes": "old"},
                "saves": [{"kind": "notes_only", "overall": "correct",
                           "steps": _ALL_NOT_REVIEWED, "notes": "new note"},
                          {"kind": "unchanged", "overall": "correct", "steps": {},
                           "notes": ""}]},
     "expected": {"entry_stop_saved": [["correct", "incorrect"], [None, None], [None, None]],
                  "notes_saved": ["old", "new note", ""],
                  "earlier_reviews_steps": ["—", "—", "Entry correct, Stop incorrect"],
                  "original_line_unchanged": True, "incorrect_shown": 1}},
    {"id": "combined_step_not_reviewed_writes_nothing",
     "inputs": {"overall": "correct", "steps": {"Entry and stop are right": "Not reviewed"}},
     "expected": {"verdicts": {"overall_verdict": "correct"}}},
    {"id": "combined_step_agree_writes_both",
     "inputs": {"overall": "correct", "steps": {"Entry and stop are right": "Agree"}},
     "expected": {"verdicts": {"overall_verdict": "correct", "entry_verdict": "correct",
                               "stop_verdict": "correct"}}},
    {"id": "namespaces",
     "inputs": {"saved_for": {"result_id": "r" * 64, **_JUDGED},
                "listed_for": {"another firm": {"firm_key": MFF, "pair_id": f"CFG-A|{MFF}"},
                               "another account": {"account_number": 7},
                               "another trade": {"strategy_trade_id": "trade-b"},
                               "another configuration": {"configuration": "CFG-B",
                                                         "pair_id": f"CFG-B|{TPT}"},
                               "another result": {"result_id": "s" * 64}}},
     "expected": {"same case": 1, "another firm": 0, "another account": 0, "another trade": 0,
                  "another configuration": 0, "another result": 0}},
    # A6 on the form: the formation steps can't be judged against another configuration's setup
    {"id": "related_record_disables_the_formation_steps",
     "inputs": {"overall": "correct", "record_link": "same_execution",
                "steps": {step: "Agree" for step, _ in rp.STEPS}},
     "expected": {"disabled": ["htf", "Parent gap and retest are right"],
                  "verdicts": {"overall_verdict": "correct", "entry_verdict": "correct",
                               "stop_verdict": "correct", "outcome_verdict": "correct"}}},
    {"id": "own_record_keeps_every_step",
     "inputs": {"overall": "correct", "record_link": "exact",
                "steps": {step: "Agree" for step, _ in rp.STEPS}},
     "expected": {"disabled": [],
                  "verdicts": {"overall_verdict": "correct", "htf_verdict": "correct",
                               "parent_verdict": "correct", "entry_verdict": "correct",
                               "stop_verdict": "correct", "outcome_verdict": "correct"}}},
    {"id": "no_record_keeps_every_step",
     "inputs": {"overall": "correct", "record_link": None,
                "steps": {"htf": "Disagree", "Parent gap and retest are right": "Unclear"}},
     "expected": {"disabled": [],
                  "verdicts": {"overall_verdict": "correct", "htf_verdict": "incorrect",
                               "parent_verdict": "insufficient_evidence"}}},
]


def _earlier_rows(repo_root: Path, case_key: str, chart_key: str,
                  moment=None) -> tuple[str, list[list[str]]]:
    """The page's "Earlier reviews" block, as text and table rows (``show`` restored after)."""

    import ifvg_lab_trade_review as review
    import ifvg_lab_ui

    from alpha_lab.agents.data_infra.ifvg.visual_review_store import list_reviews

    shown: list[str] = []
    real = ifvg_lab_ui.show
    ifvg_lab_ui.show = lambda markup, st_module=None: shown.append(str(markup))
    try:
        review._earlier_reviews(types.SimpleNamespace(), list_reviews, repo_root, case_key,
                                chart_key, moment=moment, exit_utc=None)
    finally:
        ifvg_lab_ui.show = real
    markup = "\n".join(shown)
    rows = [[_markup_text(cell).strip() for cell in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", markup, re.S)]
    return _markup_text(markup), [r for r in rows if r]


def review_judgment_actual(case: dict, repo_root: Path) -> dict:
    """The actual A8 outputs of one :data:`REVIEW_JUDGMENT_CASES` case (ledger in repo_root)."""

    from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import review_keys
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import (
        VISUAL_REVIEW_LEDGER,
        append_review,
        list_reviews,
    )

    inputs = case["inputs"]
    if "record_link" in inputs:
        link = inputs["record_link"]
        record = None if link is None else _record(link, "S1_D80_W1_P1")
        disabled = rp.disabled_steps(record)
        return {"disabled": list(disabled),
                "verdicts": rp.review_verdicts(inputs["overall"], inputs["steps"],
                                               disabled=disabled)}
    if case["id"].startswith("combined_step"):
        return {"verdicts": rp.review_verdicts(inputs["overall"], inputs["steps"])}

    def save(result_id, trade, verdicts, notes):
        chart, key, pair = review_keys(result_id, "p" * 64, trade)
        append_review(repo_root=repo_root, replay_chart_artifact_id=chart, pair_ref=pair,
                      candidate_id=key, decision_id=None, trade_id=trade["strategy_trade_id"],
                      reviewer="Test reviewer", verdicts=verdicts, tags=[], notes=notes)
        return chart, key

    if case["id"] == "namespaces":
        saved = dict(inputs["saved_for"])
        result_id = saved.pop("result_id")
        chart, key = save(result_id, saved, {"overall_verdict": "correct"}, "")
        out = {"same case": len(list_reviews(repo_root=repo_root, candidate_id=key,
                                             replay_chart_artifact_id=chart))}
        for name, change in inputs["listed_for"].items():
            other = {**saved, **change}
            other_result = other.pop("result_id", result_id)
            other_chart, other_key, _ = review_keys(other_result, "p" * 64, other)
            out[name] = len(list_reviews(repo_root=repo_root, candidate_id=other_key,
                                         replay_chart_artifact_id=other_chart))
        return out
    seed = inputs["seed"]
    chart, key = save(inputs["result_id"], inputs["trade"], seed["verdicts"], seed["notes"])
    ledger = repo_root / VISUAL_REVIEW_LEDGER
    original = ledger.read_bytes()
    for step in inputs["saves"]:
        save(inputs["result_id"], inputs["trade"],
             rp.review_verdicts(step["overall"], step["steps"]), step["notes"])
    saved = list_reviews(repo_root=repo_root, candidate_id=key, replay_chart_artifact_id=chart)

    def value(v):
        return v if isinstance(v, str) else None

    out = {"entry_stop_saved": [[value(r["entry_verdict"]), value(r["stop_verdict"])]
                                for _, r in saved.iterrows()],
           "notes_saved": [str(r["notes"]) for _, r in saved.iterrows()],
           "original_line_unchanged": ledger.read_bytes().startswith(original)
           and ledger.read_bytes().splitlines()[0] == original.splitlines()[0]}
    text, rows = _earlier_rows(repo_root, key, chart)
    out["earlier_reviews_steps"] = [r[3] for r in rows]
    out["incorrect_shown"] = text.lower().count("incorrect")
    return out


@pytest.mark.parametrize("case", REVIEW_JUDGMENT_CASES,
                         ids=[c["id"] for c in REVIEW_JUDGMENT_CASES])
def test_a8_reviewer_judgments_stay_independent(case, tmp_path):
    """A8: a saved mixed judgment (entry correct, stop incorrect) is listed as saved; notes-only
    and unchanged saves append records with unknown entry and stop (never "incorrect") and
    leave the original ledger line byte-identical; the combined step writes both only on an
    explicit choice; a review is never listed for another firm, account, trade,
    configuration or result."""

    actual = review_judgment_actual(case, tmp_path)
    assert actual == case["expected"]


def test_a8_page_preselects_nothing_and_keeps_the_mixed_judgment(page):
    """A8: with a mixed earlier judgment in an isolated ledger, the page preselects nothing,
    lists "Entry correct, Stop incorrect", and a notes-only save and an unchanged save through
    the page append records with unknown entry and stop, leaving the first line unchanged."""

    import ifvg_lab_trade_review as review

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study
    from alpha_lab.agents.data_infra.ifvg.visual_review_store import (
        VISUAL_REVIEW_LEDGER,
        append_review,
        list_reviews,
    )

    study = open_funded_study(STORE, RESULT_ID)
    row = next(t for t in study.result["tables"]["trades"]
               if t["pair_id"] == f"{LEADER}|{TPT}" and t["seq"] == 380)
    chart, key, pair = review._case(page["target"], study, row)
    append_review(repo_root=page["ledger"], replay_chart_artifact_id=chart, pair_ref=pair,
                  candidate_id=key, decision_id=None, trade_id=row["strategy_trade_id"],
                  reviewer="Earlier reviewer",
                  verdicts={"overall_verdict": "correct", "entry_verdict": "correct",
                            "stop_verdict": "incorrect"}, tags=[], notes="old")
    ledger = page["ledger"] / VISUAL_REVIEW_LEDGER
    original = ledger.read_bytes()
    at = _run(_link(380, 6, point=False))
    assert not at.exception, at.exception
    assert next(r for r in at.radio if r.label == "Overall").value is None
    steps = [s for s in at.selectbox if s.label in {rp.step_label(k, None) for k, _ in rp.STEPS}]
    assert len(steps) == 4 and all(s.value == "Not reviewed" for s in steps)
    assert "Entry correct, Stop incorrect" in _page_text(at)
    # a notes-only save through the page
    next(r for r in at.radio if r.label == "Overall").set_value("correct").run()
    at.text_input(key="ifvg_lab_v1_review_reviewer").input("Test reviewer").run()
    next(t for t in at.text_area if t.label == "Notes").input("new note").run()
    next(b for b in at.button if b.label == "Save review").click().run()
    assert not at.exception, at.exception
    # an unchanged save (overall only) in a fresh session
    again = _run(_link(380, 6, point=False))
    next(r for r in again.radio if r.label == "Overall").set_value("correct").run()
    again.text_input(key="ifvg_lab_v1_review_reviewer").input("Test reviewer").run()
    next(b for b in again.button if b.label == "Save review").click().run()
    assert not again.exception, again.exception
    saved = list_reviews(repo_root=page["ledger"], candidate_id=key,
                         replay_chart_artifact_id=chart)
    assert len(saved) == 3
    assert ledger.read_bytes().splitlines()[0] == original.splitlines()[0]
    values = [[v if isinstance(v, str) else None for v in (r["entry_verdict"], r["stop_verdict"])]
              for _, r in saved.iterrows()]
    assert values == [["correct", "incorrect"], [None, None], [None, None]]
    assert [r["notes"] for _, r in saved.iterrows()] == ["old", "new note", ""]
    text = _page_text(again)
    assert text.count("Entry correct, Stop incorrect") == 1
    assert text.lower().count("incorrect") == 1
    # never listed for the same trade at the other firm (its own account there)
    other = next(t for t in study.result["tables"]["trades"]
                 if t["pair_id"] == f"{LEADER}|{MFF}"
                 and t["strategy_trade_id"] == row["strategy_trade_id"])
    other_chart, other_key, _ = review._case(page["target"], study, other)
    assert list_reviews(repo_root=page["ledger"], candidate_id=other_key,
                        replay_chart_artifact_id=other_chart).empty


def test_a8_earlier_reviews_are_hidden_in_point_in_time_even_after_the_exit(tmp_path):
    """A8/A7: earlier reviews were written with full knowledge, so point in time never lists
    them (even at a moment after the exit); Full history lists each one as saved."""

    case = REVIEW_JUDGMENT_CASES[0]
    review_judgment_actual(case, tmp_path)
    from alpha_lab.agents.data_infra.ifvg.presentation.funded_trade_review import review_keys

    chart, key, _ = review_keys(case["inputs"]["result_id"], "p" * 64, case["inputs"]["trade"])
    text, rows = _earlier_rows(tmp_path, key, chart, moment=_utc("2026-04-14T00:00:00Z"))
    assert rows == [] and "3 earlier reviews of this trade hidden in point in time" in text
    _text_full, rows = _earlier_rows(tmp_path, key, chart)
    assert len(rows) == 3


def test_a6_disabled_formation_steps_save_nothing_whatever_the_control_holds():
    """A6: with a related record the gap and parent steps are disabled; even an "Agree" left in
    their controls writes nothing, while entry, stop and the exit are still saved; the
    combined entry-and-stop step still writes both only on an explicit choice."""

    related = _record("same_execution", "S1_D80_W1_P1")
    disabled = rp.disabled_steps(related)
    assert disabled == rp.OWN_RECORD_STEPS == ("htf", "Parent gap and retest are right")
    assert rp.disabled_steps(_record("exact", "CFG")) == () == rp.disabled_steps(None)
    verdicts = rp.review_verdicts("correct", {"htf": "Agree",
                                              "Parent gap and retest are right": "Disagree",
                                              "Entry and stop are right": "Not reviewed",
                                              "Exit handling is right": "Agree"},
                                  disabled=disabled)
    assert verdicts == {"overall_verdict": "correct", "outcome_verdict": "correct"}
    assert rp.OWN_RECORD_HELP == (
        "Needs this configuration's own setup record. The zones shown are related context from "
        "another configuration, so this step can't be judged here and nothing is saved for it.")


# ══ A10 — the published approximated-minute companion ═════════════════════

_WHY = ("2 trades share the minute's first exchange timestamp (one matching event, prices "
        "25,124.00, 25,124.75); the file lists 25,124.75 first, the candle opened at 25,124.00. "
        "High, low, last trade and trade count agree")
_WHAT = ("No stop, target, break-even stop or loss limit lies inside the minute's price range, "
         "so no ordering inside it can change the half exit, the final exit or account "
         "survival; it only moves the open-position mark between its high and low")


def _companion_row(*, configuration: str = "CFG-A", firm: str = "TakeProfitTrader",
                   account: int = 6, entry: str = "April 12, 2026 07:07 PM CDT",
                   exit_: str = "April 13, 2026 03:55 PM CDT", kind: str = "scheduled_close",
                   net: str = "6,254.72", minute_utc: str = "2026-04-13T07:03:00+00:00",
                   minute: str = "April 13, 2026 2:03:00 AM CDT") -> dict:
    return {"configuration": configuration, "firm": firm, "account_number": str(account),
            "trade_entry_chicago": entry, "trade_exit_chicago": exit_, "trade_exit_kind": kind,
            "trade_net_result_usd": net, "minute_open_utc": minute_utc, "minute_chicago": minute,
            "why_recorded_trades_did_not_rebuild_the_candle": _WHY,
            "what_the_minute_can_decide": _WHAT,
            "ordering_can_change_final_exit_or_result": "False"}


_T1 = {**ROW, "seq": 1}
_T2 = {**ROW, "seq": 2, "entry_utc": "2026-04-16T19:01:00Z",
       "exit_utc": "2026-04-16T20:02:31.5Z", "exit_kind": "breakeven_stop",
       "net_pnl_usd": 297.22, "scale_out_ns": None, "scale_out_ticks": None,
       "trading_day": "2026-04-16", "minutes_approximated": 1}
_T3 = {**ROW, "seq": 3, "entry_utc": "2026-04-20T14:00:00Z", "exit_utc": "2026-04-20T14:30:00Z",
       "exit_kind": "target", "net_pnl_usd": 399.72, "minutes_approximated": 0,
       "trading_day": "2026-04-20"}
_R1 = _companion_row()
_R2 = _companion_row(entry="April 16, 2026 02:01 PM CDT", exit_="April 16, 2026 03:02 PM CDT",
                     kind="breakeven_stop", net="297.22",
                     minute_utc="2026-04-16T19:46:00+00:00",
                     minute="April 16, 2026 2:46:00 PM CDT")
_R_OTHER_FIRM = _companion_row(firm="MyFundedFutures", account=4)
_R_OTHER_CONFIG = _companion_row(configuration="CFG-B")
_BASE_ROWS = [_R1, _R2, _R_OTHER_FIRM, _R_OTHER_CONFIG]
_SOURCE = {"folder": "funded_comparison_rrrrrrrrrrrrrrrr_export_v4", "version": 4,
           "result_id": "r" * 64, "sha256": "0" * 64}

#: A10 regression cases: (companion rows, trade, the pair's trades) → link status
MINUTE_COMPANION_CASES = [
    {"id": "linked_first_trade",
     "inputs": {"rows": _BASE_ROWS, "trade": _T1, "pair_trades": [_T1, _T2, _T3]},
     "expected": {"status": "linked", "minutes": ["2026-04-13T07:03:00+00:00"], "reason": "",
                  "evidence": (
                      "1,247 minutes on recorded exchange trades, 1 approximated from the "
                      "one-minute bar: April 13, 2:03 AM — 2 trades share the minute's first "
                      "exchange timestamp; the file lists 25,124.75 first, the candle opened at "
                      f"25,124.00. {_WHAT}. Source: the published review folder (export 4 of "
                      "this result), approximated_minutes.csv, hash-checked. The exit was "
                      "priced from recorded trades.")}},
    {"id": "linked_second_trade",
     "inputs": {"rows": _BASE_ROWS, "trade": _T2, "pair_trades": [_T1, _T2, _T3]},
     "expected": {"status": "linked", "minutes": ["2026-04-16T19:46:00+00:00"], "reason": ""}},
    {"id": "trade_row_without_the_minute_field_still_links",
     "inputs": {"rows": _BASE_ROWS, "trade": {**_T1, "minutes_approximated": None},
                "pair_trades": [{**_T1, "minutes_approximated": None}, _T2]},
     "expected": {"status": "linked", "minutes": ["2026-04-13T07:03:00+00:00"], "reason": ""}},
    {"id": "duplicate_identity_in_the_pair",
     "inputs": {"rows": _BASE_ROWS, "trade": _T1,
                "pair_trades": [_T1, {**_T1, "seq": 9}, _T2]},
     "expected": {"status": "conflict", "minutes": ["2026-04-13T07:03:00+00:00"],
                  "reason": "another trade of this configuration at this firm has the same "
                            "account, entry and exit minutes, exit and result",
                  "evidence": (
                      "1,247 minutes on recorded exchange trades, 1 approximated from the "
                      "one-minute bar; the published minute record doesn't match this trade "
                      "exactly (another trade of this configuration at this firm has the same "
                      "account, entry and exit minutes, exit and result), so it isn't shown. "
                      "The exit was priced from recorded trades.")}},
    {"id": "minute_outside_the_trade",
     "inputs": {"rows": [_companion_row(minute_utc="2026-04-12T23:00:00+00:00",
                                        minute="April 12, 2026 6:00:00 PM CDT"), _R2],
                "trade": _T1, "pair_trades": [_T1, _T2]},
     "expected": {"status": "conflict", "minutes": ["2026-04-12T23:00:00+00:00"],
                  "reason": "a listed minute falls outside this trade's entry-to-exit time"}},
    {"id": "count_mismatch",
     "inputs": {"rows": _BASE_ROWS, "trade": {**_T1, "minutes_approximated": 2},
                "pair_trades": [{**_T1, "minutes_approximated": 2}, _T2]},
     "expected": {"status": "conflict", "minutes": ["2026-04-13T07:03:00+00:00"],
                  "reason": "it lists 1 minute where this trade recorded 2"}},
    {"id": "approximated_minute_without_a_row",
     "inputs": {"rows": [_R2, _R_OTHER_FIRM, _R_OTHER_CONFIG], "trade": _T1,
                "pair_trades": [_T1, _T2]},
     "expected": {"status": "conflict", "minutes": [],
                  "reason": "it lists no minute for this trade"}},
    {"id": "minute_listed_twice",
     "inputs": {"rows": [_R1, _R1], "trade": {**_T1, "minutes_approximated": 2},
                "pair_trades": [{**_T1, "minutes_approximated": 2}]},
     "expected": {"status": "conflict",
                  "minutes": ["2026-04-13T07:03:00+00:00", "2026-04-13T07:03:00+00:00"],
                  "reason": "a minute is listed twice for this trade"}},
    {"id": "minute_chicago_and_utc_disagree",
     "inputs": {"rows": [_companion_row(minute="April 13, 2026 2:04:00 AM CDT")], "trade": _T1,
                "pair_trades": [_T1]},
     "expected": {"status": "conflict", "minutes": ["2026-04-13T07:03:00+00:00"],
                  "reason": "a listed minute's Chicago time and UTC time disagree"}},
    {"id": "no_approximated_minute_and_no_row",
     "inputs": {"rows": _BASE_ROWS, "trade": _T3, "pair_trades": [_T1, _T2, _T3]},
     "expected": {"status": "none", "minutes": [], "reason": "",
                  "evidence": "1,247 minutes on recorded exchange trades. The exit was priced "
                              "from recorded trades."}},
    # a row with this trade's account, entry and exit minutes but another result or exit kind
    # disagrees with the trade: a conflict, never "no minute"
    {"id": "same_times_different_result",
     "inputs": {"rows": [_companion_row(net="6,254.70")], "trade": _T1, "pair_trades": [_T1]},
     "expected": {"status": "conflict", "minutes": [],
                  "reason": "it lists a minute for this account, entry and exit time with a "
                            "different exit or result",
                  "evidence": (
                      "1,247 minutes on recorded exchange trades, 1 approximated from the "
                      "one-minute bar; the published minute record doesn't match this trade "
                      "exactly (it lists a minute for this account, entry and exit time with a "
                      "different exit or result), so it isn't shown. The exit was priced from "
                      "recorded trades.")}},
    {"id": "same_times_different_exit_kind_and_no_minute_recorded",
     "inputs": {"rows": [_companion_row(kind="target")],
                "trade": {**_T1, "minutes_approximated": 0},
                "pair_trades": [{**_T1, "minutes_approximated": 0}]},
     "expected": {"status": "conflict", "minutes": [],
                  "reason": "it lists a minute for this account, entry and exit time with a "
                            "different exit or result",
                  "evidence": (
                      "1,247 minutes on recorded exchange trades. The published minute record "
                      "doesn't match this trade exactly (it lists a minute for this account, "
                      "entry and exit time with a different exit or result), so it isn't shown. "
                      "The exit was priced from recorded trades.")}},
    {"id": "same_times_row_of_another_recorded_trade",
     "inputs": {"rows": [_R1],
                "trade": {**_T1, "seq": 9, "net_pnl_usd": 1.0, "minutes_approximated": 0},
                "pair_trades": [_T1, {**_T1, "seq": 9, "net_pnl_usd": 1.0,
                                      "minutes_approximated": 0}]},
     "expected": {"status": "none", "minutes": [], "reason": ""}},
]


def minute_companion_actual(case: dict) -> dict:
    """The actual A10 outputs of one :data:`MINUTE_COMPANION_CASES` case (every expected key,
    plus ``evidence`` for every case)."""

    inputs = case["inputs"]
    trade = rp.TradeView.from_row(inputs["trade"])
    link = mc.link_trade(inputs["rows"], trade=trade,
                         pair_trades=[rp.TradeView.from_row(t) for t in inputs["pair_trades"]],
                         configuration="CFG-A", firm_name="TakeProfitTrader",
                         source=mc.CompanionSource(**_SOURCE))
    return {"status": link.status, "minutes": [m.isoformat() for m in link.minutes],
            "reason": link.reason, "evidence": rp.evidence_note(trade, link)}


def _expected_part(actual: dict, expected: dict) -> dict:
    return {key: actual.get(key, "<missing>") for key in expected}


@pytest.mark.parametrize("case", MINUTE_COMPANION_CASES,
                         ids=[c["id"] for c in MINUTE_COMPANION_CASES])
def test_a10_minute_companion_links_only_an_exact_unique_trade(case):
    """A10: a companion row links to a trade only by configuration, firm, account, entry and
    exit minutes (the Chicago texts parsed to instants), exit kind and result; duplicate
    identity, a minute outside the trade, a count mismatch, a repeated minute, disagreeing
    minute times, or a row with this trade's times but another exit or result are conflicts;
    no approximated minute and no row is "none"."""

    actual = minute_companion_actual(case)
    assert _expected_part(actual, case["expected"]) == case["expected"]


#: A10 wording cases: a trade row without its approximated-minute count never reads as zero
EVIDENCE_WORDING_CASES = [
    {"id": "count_not_recorded_and_companion_unavailable",
     "inputs": {"minutes_approximated": None, "status": "unavailable"},
     "expected": {"evidence": (
         "1,247 minutes on recorded exchange trades; the number of approximated minutes isn't "
         "recorded for this trade, and no hash-checked published minute record is available "
         "for this result. The exit was priced from recorded trades.")}},
    {"id": "count_not_recorded_and_companion_lists_none",
     "inputs": {"minutes_approximated": None, "status": "none"},
     "expected": {"evidence": (
         "1,247 minutes on recorded exchange trades; the number of approximated minutes isn't "
         "recorded for this trade, and the hash-checked published minute record lists none for "
         "it. The exit was priced from recorded trades.")}},
    {"id": "count_not_recorded_and_no_companion_checked",
     "inputs": {"minutes_approximated": None, "status": None},
     "expected": {"evidence": (
         "1,247 minutes on recorded exchange trades; the number of approximated minutes isn't "
         "recorded for this trade. The exit was priced from recorded trades.")}},
    {"id": "one_recorded_and_companion_unavailable",
     "inputs": {"minutes_approximated": 1, "status": "unavailable"},
     "expected": {"evidence": (
         "1,247 minutes on recorded exchange trades, 1 approximated from the one-minute bar; no "
         "hash-checked published minute record is available for this result. The exit was "
         "priced from recorded trades.")}},
    {"id": "zero_recorded_and_companion_unavailable",
     "inputs": {"minutes_approximated": 0, "status": "unavailable"},
     "expected": {"evidence": (
         "1,247 minutes on recorded exchange trades. The exit was priced from recorded "
         "trades.")}},
]


def evidence_wording_actual(case: dict) -> dict:
    inputs = case["inputs"]
    view = rp.TradeView.from_row({**ROW, "minutes_approximated": inputs["minutes_approximated"]})
    link = None if inputs["status"] is None else mc.MinuteLink(inputs["status"], reason="r")
    return {"evidence": rp.evidence_note(view, link)}


@pytest.mark.parametrize("case", EVIDENCE_WORDING_CASES,
                         ids=[c["id"] for c in EVIDENCE_WORDING_CASES])
def test_a10_a_missing_approximated_count_never_reads_as_zero(case):
    """A10: when the trade row lacks its approximated-minute count, the evidence line says the
    count isn't recorded (and whether a hash-checked record exists), never implying zero."""

    assert evidence_wording_actual(case) == case["expected"]


def test_a10_times_are_parsed_never_reformatted():
    """A10: the companion's Chicago texts are parsed to instants (zone abbreviation checked);
    trade instants are floored to the minute for the comparison and never changed."""

    assert mc.parse_chicago_text("April 12, 2026 07:07 PM CDT") == _utc("2026-04-13T00:07:00Z")
    assert mc.parse_chicago_text("April 13, 2026 2:03:00 AM CDT") == _utc("2026-04-13T07:03:00Z")
    assert mc.parse_chicago_text("November 1, 2026 1:30 AM CDT") == _utc("2026-11-01T06:30:00Z")
    assert mc.parse_chicago_text("November 1, 2026 1:30 AM CST") == _utc("2026-11-01T07:30:00Z")
    for bad in ("April 13, 2026 2:03:00 AM CST", "March 8, 2026 2:30 AM CST",
                "2026-04-13 02:03", ""):
        assert mc.parse_chicago_text(bad) is None, bad
    view = rp.TradeView.from_row(_T2)
    identity = mc.trade_identity(view)
    assert identity[2] == _utc("2026-04-16T20:02:00Z")  # 3:02:31.5 PM → the 3:02 PM minute
    assert view.exit_utc == _utc("2026-04-16T20:02:31.5Z")  # the recorded instant is kept


def _publish(parent: Path, name: str, result_id: str, rows: list[dict], *,
             sha: str | None = None, zip_only: bool = False) -> None:
    import zipfile

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    manifest = {"funded_comparison_result_id": result_id, "export_version": 4,
                "published_at_utc": "2026-09-23T19:34:33Z",
                "files": [{"path": mc.COMPANION_FILE, "bytes": len(data),
                           "sha256": sha or hashlib.sha256(data).hexdigest()}]}
    if zip_only:
        with zipfile.ZipFile(parent / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}/{mc.COMPANION_FILE}", data)
            archive.writestr(f"{name}/run_manifest.json", json.dumps(manifest))
        return
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / mc.COMPANION_FILE).write_bytes(data)
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


_RID = "b" * 64
_PREFIX = f"funded_comparison_{_RID[:16]}_export_v"

#: A10 source-binding cases: which published folder may supply the minute record
FOLDER_CASES = [
    {"id": "this_results_folder", "inputs": {"folders": [
        {"name": f"{_PREFIX}4", "result_id": _RID}]},
     "expected": {"status": "linked", "version": 4, "result_id": _RID}},
    {"id": "only_another_results_folder", "inputs": {"folders": [
        {"name": f"{_PREFIX}4", "result_id": "c" * 64}]},
     "expected": {"status": "unavailable", "version": None,
                  "reason": "no published review folder names this result"}},
    {"id": "hash_mismatch", "inputs": {"folders": [
        {"name": f"{_PREFIX}4", "result_id": _RID, "sha": "0" * 64}]},
     "expected": {"status": "unavailable", "version": 4,
                  "reason": "approximated_minutes.csv is missing or failed its hash check"}},
    {"id": "zip_only_folder", "inputs": {"folders": [
        {"name": f"{_PREFIX}4", "result_id": _RID, "zip_only": True}]},
     "expected": {"status": "linked", "version": 4, "result_id": _RID}},
    {"id": "no_folder", "inputs": {"folders": []},
     "expected": {"status": "unavailable", "version": None,
                  "reason": "no published review folder names this result"}},
]


def folder_case_actual(case: dict, repo_root: Path) -> dict:
    import ifvg_lab_detail_settings as settings

    parent = repo_root / "reports/funded_comparison"
    parent.mkdir(parents=True, exist_ok=True)
    for folder in case["inputs"]["folders"]:
        _publish(parent, folder["name"], folder["result_id"], [_R1, _R2],
                 sha=folder.get("sha"), zip_only=folder.get("zip_only", False))
    before = sorted(p.as_posix() for p in repo_root.rglob("*"))
    review = settings.latest_review_folder(repo_root, _RID)
    link = mc.link_from_review_folder(review, result_id=_RID,
                                      trade=rp.TradeView.from_row(_T1),
                                      pair_trades=[rp.TradeView.from_row(_T1)],
                                      configuration="CFG-A", firm_name="TakeProfitTrader")
    assert sorted(p.as_posix() for p in repo_root.rglob("*")) == before  # read only
    out = {"status": link.status,
           "version": link.source.version if link.source is not None else None}
    if link.status == "linked":
        out["result_id"] = link.source.result_id
        data = review.read(mc.COMPANION_FILE)
        assert link.source.sha256 == hashlib.sha256(data).hexdigest()
    else:
        out["reason"] = link.reason
    return out


@pytest.mark.parametrize("case", FOLDER_CASES, ids=[c["id"] for c in FOLDER_CASES])
def test_a10_only_this_results_hash_checked_folder_is_used(case, tmp_path):
    """A10: the minute record comes only from the latest published folder whose manifest
    names this exact result, and only when the file matches the manifest's SHA-256;
    another result's folder, a hash mismatch or no folder is "unavailable"."""

    assert folder_case_actual(case, tmp_path) == case["expected"]


def test_a10_unavailable_and_unchecked_sentences():
    """A10: without a hash-checked record the evidence line says so; with no link at all the
    saved row's own wording is kept."""

    view = rp.TradeView.from_row(ROW)
    unavailable = mc.MinuteLink("unavailable", reason="no published review folder names this "
                                                      "result")
    assert rp.evidence_note(view, unavailable) == (
        "1,247 minutes on recorded exchange trades, 1 approximated from the one-minute bar; no "
        "hash-checked published minute record is available for this result. The exit was "
        "priced from recorded trades.")
    none = rp.TradeView.from_row({**ROW, "minutes_approximated": 0})
    assert rp.evidence_note(none, unavailable) == (
        "1,247 minutes on recorded exchange trades. The exit was priced from recorded trades.")
    assert "isn't in this study's export" in rp.evidence_note(view)
    missing = rp.TradeView.from_row({**ROW, "minutes_on_prints": None,
                                     "minutes_approximated": None})
    assert rp.evidence_note(missing, unavailable) == (
        "Price evidence for this trade is not in this study's export.")


# ── the published companion of the saved result (read only) ───────────────

#: A10 real-study check (skipped where the saved study or its published folder is absent)
REFERENCE_MINUTE_LINK = {
    "id": "leader_april_trade_takeprofittrader_account_6",
    "inputs": {"result_id": RESULT_ID, "configuration": LEADER, "firm": "TakeProfitTrader",
               "account": 6, "trade_seq": 380, "entry": "April 12, 2026 7:07 PM Chicago",
               "file": "approximated_minutes.csv"},
    "expected": {"status": "linked", "rows": 1, "minute_open_utc": "2026-04-13T07:03:00+00:00",
                 "minute_chicago": "April 13, 2026, 2:03 AM",
                 "ordering_can_change_final_exit_or_result": "False",
                 "folder": "funded_comparison_5fa65149843484b1_export_v4", "export_version": 4,
                 "result_id": RESULT_ID,
                 "sha256": "6201ef47e5d052772d00958b34462b7f481c3f32f72fb4c706aa172235694867",
                 "other_firm_account_4_links_the_same_minute": True},
}


def reference_minute_link_actual() -> dict:
    """The actual values of :data:`REFERENCE_MINUTE_LINK` (same keys, plus the evidence line;
    needs the saved study and its published review folder)."""

    import ifvg_lab_detail_settings as settings

    from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    study, rows = _real_study()
    folder = settings.latest_review_folder(REPO, RESULT_ID)
    if folder is None:
        pytest.skip("the published review folder is not on this computer")
    views = [rp.TradeView.from_row(r) for r in rows]
    view = next(v for v in views if v.seq == 380)
    link = mc.link_from_review_folder(folder, result_id=RESULT_ID, trade=view, pair_trades=views,
                                      configuration=LEADER, firm_name="TakeProfitTrader")
    other_views = [rp.TradeView.from_row(r) for r in ordered_trades(study, LEADER, MFF)]
    other = next(v for v in other_views if v.entry_utc == view.entry_utc)
    other_link = mc.link_from_review_folder(folder, result_id=RESULT_ID, trade=other,
                                            pair_trades=other_views, configuration=LEADER,
                                            firm_name="MyFundedFutures")
    source = link.source
    return {
        "status": link.status, "rows": len(link.rows),
        "minute_open_utc": link.minutes[0].isoformat() if link.minutes else None,
        "minute_chicago": fmt.chicago_long(link.minutes[0]) if link.minutes else None,
        "ordering_can_change_final_exit_or_result": (
            link.rows[0].get("ordering_can_change_final_exit_or_result") if link.rows else None),
        "folder": source.folder if source else None,
        "export_version": source.version if source else None,
        "result_id": source.result_id if source else None,
        "sha256": source.sha256 if source else None,
        "file_sha256_now": hashlib.sha256(folder.read(mc.COMPANION_FILE) or b"").hexdigest(),
        "other_firm_account_4_links_the_same_minute": (
            other.account == 4 and other_link.status == "linked"
            and other_link.minutes == link.minutes),
        "reason": link.reason, "evidence": rp.evidence_note(view, link),
    }


def test_a10_real_leader_april_trade_links_the_published_minute():
    """A10: the leader's April trade at TakeProfitTrader (Account 6) links exactly one published
    minute, April 13, 2026, 2:03 AM Chicago, whose ordering can't change the exit or result;
    the source is export 4 of the full result id with the manifest's SHA-256."""

    expected = REFERENCE_MINUTE_LINK["expected"]
    actual = reference_minute_link_actual()
    assert _expected_part(actual, expected) == expected, actual["reason"]
    assert actual["file_sha256_now"] == expected["sha256"]
    evidence = actual["evidence"]
    assert "1 approximated from the one-minute bar: April 13, 2:03 AM — " in evidence
    assert evidence.endswith("Source: the published review folder (export 4 of this result), "
                             "approximated_minutes.csv, hash-checked. The exit was priced from "
                             "recorded trades.")


def test_a10_page_shows_the_linked_minute_from_an_isolated_copy(page):
    """A10: Trade review shows the linked minute from a hash-checked copy of the published
    folder (isolated repo root), keeps it hidden in point in time before the exit, and says
    "unavailable" when no folder names the result."""

    at = _run(_link(380, 6, point=False))
    assert not at.exception, at.exception
    assert ("1 approximated from the one-minute bar; no hash-checked published minute record "
            "is available for this result.") in _page_text(at)
    if not (PUBLISHED / "run_manifest.json").is_file():
        pytest.skip("the published review folder is not on this computer")
    copy_to = page["ledger"] / "reports/funded_comparison" / PUBLISHED.name
    copy_to.mkdir(parents=True)
    for name in ("run_manifest.json", mc.COMPANION_FILE):
        (copy_to / name).write_bytes((PUBLISHED / name).read_bytes())
    at = _run(_link(380, 6, point=False))
    assert not at.exception, at.exception
    text = _page_text(at)
    assert ("1,247 minutes on recorded exchange trades, 1 approximated from the one-minute bar: "
            "April 13, 2:03 AM — ") in text
    assert ("Source: the published review folder (export 4 of this result), "
            "approximated_minutes.csv, hash-checked.") in text
    hidden = _run(_link(380, 6, point=True))
    assert not hidden.exception, hidden.exception
    text = _page_text(hidden)
    assert "April 13, 2:03 AM" not in text
    assert "Price evidence for the whole trade is hidden with the exit" in text


def test_a10_a_conflicting_record_never_reads_an_unrecorded_count_as_zero():
    """A10 (re-review): with a conflicting minute record and no recorded count, the sentence
    says the count isn't recorded instead of reading it as zero."""

    view = rp.TradeView.from_row({**ROW, "minutes_on_prints": 1247, "minutes_approximated": None})
    link = mc.MinuteLink("conflict", (), "a minute is listed twice for this trade", {}, None)
    text = rp.evidence_note(view, link)
    assert "the number of approximated minutes isn't recorded for this trade" in text
    assert "doesn't match this trade exactly (a minute is listed twice for this trade)" in text
    zero = rp.evidence_note(rp.TradeView.from_row({**ROW, "minutes_on_prints": 1247,
                                                    "minutes_approximated": 0}), link)
    assert "isn't recorded" not in zero and text != zero
