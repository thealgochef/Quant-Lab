"""Regression tests for the review findings fixed in the shared redesign code.

Synthetic records only (no saved study needed): drop-the-largest-payout ties and
wording, the deflated Sharpe's N, undefined moments for a configuration without
trades, the 5:00 PM open on daylight-saving Sundays, unique configuration names,
and deep links that can't crash the workspace.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

CAL = ["2026-01-13", "2026-01-14", "2026-01-15", "2026-01-16"]


def _summary(config: str, firm: str, *, net: int, largest: int, rank: int) -> dict:
    return {"configuration": config, "firm_key": firm, "firm": firm, "status": "Completed",
            "net_cash_earned_cents": net, "largest_single_payout_cents": largest,
            "payouts_received_count": 1, "rank_within_firm": rank}


def _study(summaries: dict, trades: list[dict] | None = None):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result

    plan = SimpleNamespace(source=SimpleNamespace(evaluation_dates=CAL, warmup_dates=[]))
    result = {"summaries_cents": summaries, "tables": {"trades": trades or []}}
    return study_from_result(result, result_id="r" * 16, plan=plan)


def test_drop_largest_tie_keeps_the_leader_at_rank_one():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        drop_largest_payout,
    )

    study = _study({"A|t": _summary("A", "t", net=3_000_000, largest=500_000, rank=1),
                    "B|t": _summary("B", "t", net=2_900_000, largest=400_000, rank=2)})
    check = drop_largest_payout(study, "t")
    assert check.leader_after_cents == check.runner_up_after_cents == 2_500_000
    assert check.passed is True


def test_both_firms_only_when_the_same_configuration_leads_at_both():
    from ifvg_lab_funded import drop_scope

    same = {"t": SimpleNamespace(passed=True, leader="A"),
            "m": SimpleNamespace(passed=True, leader="A")}
    different = {"t": SimpleNamespace(passed=True, leader="A"),
                 "m": SimpleNamespace(passed=True, leader="B")}
    assert drop_scope(same, "t", "TakeProfitTrader") == "at both firms"
    assert drop_scope(different, "t", "TakeProfitTrader") == "at TakeProfitTrader"


def _trade(config: str, day: str, pnl: float, seq: int) -> dict:
    return {"pair_id": f"{config}|t", "configuration": config, "firm_key": "t",
            "trading_day": day, "net_pnl_usd": pnl, "entry_utc": f"{day}T15:00:00Z", "seq": seq}


def test_deflated_sharpe_counts_every_completed_configuration():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import deflated_sharpe

    trades = [_trade("A", CAL[0], 500, 1), _trade("A", CAL[1], -100, 2),
              _trade("A", CAL[2], 300, 3), _trade("B", CAL[0], 50, 4),
              _trade("B", CAL[3], -20, 5)]
    study = _study({"A|t": _summary("A", "t", net=1, largest=1, rank=1),
                    "B|t": _summary("B", "t", net=1, largest=1, rank=2),
                    "C|t": _summary("C", "t", net=1, largest=1, rank=3)}, trades)
    result = deflated_sharpe(study, "A", "t")
    assert result.tested == 3  # C took no trades and is still one of the three compared
    no_trades = deflated_sharpe(study, "C", "t")
    assert no_trades.daily_sharpe is None and no_trades.skew is None
    assert no_trades.above_zero is None and no_trades.deflated is None


def test_moments_are_undefined_when_no_day_varies():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import (
        sharpe_confidence,
    )

    flat = sharpe_confidence([0.0] * 107)
    assert (flat.daily_sharpe, flat.skew, flat.kurtosis, flat.above_zero) == (
        None, None, None, None)


@pytest.mark.parametrize(("entry", "hours"), [
    ("2026-03-08T22:30:00Z", 0.5),  # spring forward Sunday, 5:30 PM CDT
    ("2026-11-01T23:30:00Z", 0.5),  # fall back Sunday, 5:30 PM CST
    ("2026-04-13T00:07:00Z", 2 + 7 / 60),
])
def test_time_of_day_counts_from_the_wall_clock_open(entry, hours):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.market import _entry_x

    minutes = pd.DataFrame({"close": [1.0],
                            "logical_close_ts_utc": [pd.Timestamp("2026-01-01", tz="UTC")]})
    value = _entry_x(minutes, np.array([0]), {"entry_utc": entry, "entry_ticks": 4}, "time")
    assert value == pytest.approx(hours)


def test_study_names_are_unique_when_hidden_settings_vary():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import study_names

    base = [("Entry hours", "All open-market hours"), ("Direction", "Long only"),
            ("Profit target", "equal to the initial risk (1 to 1)"),
            ("Higher-timeframe gap charts", "one-hour"),
            ("Supporting (parent) charts", "one-minute, five-minute")]
    names = study_names({
        "D80": [*base, ("Largest distance from the parent gap to the opposing gap",
                        "80 ticks (20 points)")],
        "D160": [*base, ("Largest distance from the parent gap to the opposing gap",
                         "160 ticks (40 points)")],
        "twin_a": base, "twin_b": base})
    fulls = [name.full for name in names.values()]
    assert len(set(fulls)) == len(fulls)
    assert "opposing distance 160 ticks" in names["D160"].line2
    assert names["twin_a"].line2.endswith("twin_a")


class _FakeStreamlit:
    def __init__(self, params: dict):
        self.session_state: dict = {}
        self.query_params = params


@pytest.mark.parametrize("params", [
    {"view": "review", "result": "a" * 64, "account": "x", "trade": "1.5"},
    {"view": "detail", "result": "../../etc", "firm": "t"},
    {"view": "funded", "result": "a" * 64, "app": "elsewhere"},
])
def test_malformed_links_never_crash_and_change_nothing(params, tmp_path):
    import ifvg_lab_nav as nav

    fake = _FakeStreamlit(params)
    roots = {"repo_root": tmp_path, "draft_root": tmp_path / "data/ifvg_study_drafts",
             "store_root": tmp_path / "data/ifvg_datasets/search/v1",
             "state_root": tmp_path / "s", "pipeline_state_root": tmp_path / "p"}
    nav.apply_deep_link(roots, fake)
    state = fake.session_state
    target = state.get(nav.FUNDED_TARGET)
    if "../" in params["result"]:
        assert target is None and state[nav.SCREEN] == "list"
        assert "could not be opened" in state[nav.LINK_NOTE]
    else:
        # an unknown saved result opens read-only with its status unconfirmed
        assert target["status"] is None and target["name"] == ""
        review = state.get(nav.REVIEW_TARGET)
        if review:
            assert review["account_number"] is None and review["trade_seq"] is None
    assert not list(tmp_path.rglob("*.json"))  # nothing was written
