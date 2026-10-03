"""Setup records for funded trades (TASK.md section 4.1; rules 6–8).

Synthetic packages check the linking rules and the manifest check; the
reference test reads the saved funded variation study and its verified strategy
package on this computer (read only) and is skipped where they are absent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.presentation.lab.setup_records import (
    NO_RECORD,
    SetupRecordError,
    entry_match_key,
    find_setup_record,
    load_setup_record_source,
    settings_distance,
)

REPO = Path(__file__).resolve().parents[3]
STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT = "takeprofittrader"

ENTRY = "2026-04-13 00:07:00+00:00"


def _row(profile: str, trade_id: str, *, target: int = 99969, stop: int = 99799,
         entry: int = 99884, when: str = ENTRY, htf_low: int = 97806) -> dict:
    return {
        "profile": profile, "trade_id": trade_id, "setup_id": f"setup-{trade_id}",
        "is_warmup": False, "direction": "LONG", "entry_family": "fresh_fvg_continuation",
        "entry_ts_utc": when, "entry_ticks": entry, "stop_ticks": stop, "target_ticks": target,
        "resolution": "target",
        "entry_match_key": entry_match_key(when, "LONG", "fresh_fvg_continuation", entry),
        "exact_execution_key": "x",
        "geometry_htf_timeframe_seconds": 14400, "geometry_htf_direction": "bullish",
        "geometry_htf_gap_low_ticks": htf_low, "geometry_htf_gap_high_ticks": 100347,
        "geometry_htf_size_ticks": 100347 - htf_low,
        "geometry_htf_a_open_ts_utc": "2026-04-07 18:00:00+00:00",
        "geometry_htf_confirmed_ts_utc": "2026-04-08 06:00:00+00:00",
        "geometry_parent_timeframe_seconds": 300, "geometry_parent_direction": "bullish",
        "geometry_parent_gap_low_ticks": 99749, "geometry_parent_gap_high_ticks": 99866,
        "geometry_parent_size_ticks": 117,
        "geometry_parent_a_open_ts_utc": "2026-04-12 23:45:00+00:00",
        "geometry_parent_confirmed_ts_utc": "2026-04-13 00:00:00+00:00",
        "geometry_opposing_timeframe_seconds": 60, "geometry_opposing_direction": "bearish",
        "geometry_opposing_gap_low_ticks": 99847, "geometry_opposing_gap_high_ticks": 99850,
        "geometry_opposing_size_ticks": 3,
        "geometry_opposing_a_open_ts_utc": "2026-04-13 00:00:00+00:00",
        "geometry_opposing_confirmed_ts_utc": "2026-04-13 00:03:00+00:00",
        "geometry_tap_bar_logical_open_ts_utc": "2026-04-12 22:18:00+00:00",
        "geometry_tap_bar_logical_close_ts_utc": "2026-04-12 22:19:00+00:00",
        "geometry_inversion_bar_logical_open_ts_utc": "2026-04-13 00:05:00+00:00",
        "geometry_inversion_bar_logical_close_ts_utc": "2026-04-13 00:06:00+00:00",
    }


MEMBERS = {
    "S1_D80_W1_P1": {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
                     "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m"},
    "S1_D80_W1_P0": {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
                     "parent_timeframes": "parent_timeframes.3m-5m-10m-15m-30m"},
    "S0_D80_W1_P1": {"enabled_entry_sessions": "enabled_entry_sessions.asia-london-ny",
                     "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m"},
}
VARIANT_IDS = {"enabled_entry_sessions": "enabled_entry_sessions.all_open_market_v1",
               "parent_timeframes": "parent_timeframes.1m-3m-5m-10m-15m-30m",
               "exit_policy": "exit_policy.scale_out_half_breakeven_hold_to_close_v1"}
TRADE = {"strategy_trade_id": "funded-variant-id", "direction": "long",
         "entry_utc": "2026-04-13T00:07:00Z", "entry_ticks": 99884, "stop_ticks": 99799,
         "target_ticks": 99969}


def _package(root: Path, rows: list[dict]) -> Path:
    (root / "data").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(root / "data/trades.csv", index=False)
    (root / "run_context.json").write_text(json.dumps({"frozen_batch": {"members": [
        {"name": name, "spec": {"axis_value_ids": ids}} for name, ids in MEMBERS.items()]}}),
        encoding="utf-8")
    (root / "configs.json").write_text(json.dumps({
        name: {"section": {"entry_family": "fresh_fvg_continuation"}} for name in MEMBERS}),
        encoding="utf-8")
    files = [{"path": rel, "sha256": hashlib.sha256((root / rel).read_bytes()).hexdigest()}
             for rel in ("data/trades.csv", "run_context.json", "configs.json")]
    (root / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return root


def test_entry_match_key_is_the_packages_definition():
    values = ["2026-04-13T00:07:00+00:00", "LONG", "fresh_fvg_continuation", 99884]
    expected = hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()
    assert entry_match_key("2026-04-13T00:07:00Z", "LONG", "fresh_fvg_continuation",
                           99884) == expected
    assert entry_match_key(pd.Timestamp(ENTRY), "LONG", "fresh_fvg_continuation",
                           99884.0) == expected


def test_a_tampered_package_file_is_refused(tmp_path):
    root = _package(tmp_path / "pkg", [_row("S1_D80_W1_P1", "a")])
    (root / "data/trades.csv").write_text("profile\nchanged\n", encoding="utf-8")
    with pytest.raises(SetupRecordError):
        load_setup_record_source(root)


def test_a_member_configuration_uses_its_own_record_by_trade_id(tmp_path):
    source = load_setup_record_source(_package(tmp_path / "pkg", [
        _row("S1_D80_W1_P1", "member-trade"), _row("S0_D80_W1_P1", "other", htf_low=97000)]))
    trade = {**TRADE, "strategy_trade_id": "member-trade"}
    record = find_setup_record(trade, configuration="S1_D80_W1_P1", source=source,
                               axis_value_ids=MEMBERS["S1_D80_W1_P1"], in_verified_study=True)
    assert record.link == "exact" and record.configuration == "S1_D80_W1_P1"
    # correction A6: only the configuration's own record establishes the setup's identity
    assert record.identity_established
    assert record.source_sentence == "Zones come from this configuration's own saved setup record."


def test_a_variant_links_the_closest_configurations_record_of_the_same_execution(tmp_path):
    source = load_setup_record_source(_package(tmp_path / "pkg", [
        _row("S0_D80_W1_P1", "a", htf_low=97000), _row("S1_D80_W1_P0", "b", htf_low=97100),
        _row("S1_D80_W1_P1", "c")]))
    record = find_setup_record(TRADE, configuration=LEADER, source=source,
                               axis_value_ids=VARIANT_IDS)
    assert record.link == "same_execution" and record.configuration == "S1_D80_W1_P1"
    assert record.same_target and record.htf.low_ticks == 97806
    # correction A6: matching entry, stop and target is related context, not setup identity
    assert not record.identity_established
    assert record.source_sentence == (
        "Related context from configuration S1_D80_W1_P1's saved setup record for the same "
        "entry time, direction, entry price and stop, and target; exact setup identity for this "
        "configuration is not established.")


def test_a_record_with_the_same_target_is_preferred_and_a_different_one_is_said(tmp_path):
    source = load_setup_record_source(_package(tmp_path / "pkg", [
        _row("S1_D80_W1_P1", "far-target", target=100054),
        _row("S0_D80_W1_P1", "same-target")]))
    record = find_setup_record(TRADE, configuration=LEADER, source=source,
                               axis_value_ids=VARIANT_IDS)
    assert record.configuration == "S0_D80_W1_P1" and record.same_target
    only_other = load_setup_record_source(_package(tmp_path / "pkg2", [
        _row("S1_D80_W1_P1", "far-target", target=100054)]))
    record = find_setup_record(TRADE, configuration=LEADER, source=only_other,
                               axis_value_ids=VARIANT_IDS)
    assert not record.same_target
    # correction A6: related context, and its differing target is said
    assert record.source_sentence == (
        "Related context from configuration S1_D80_W1_P1's saved setup record for the same "
        "entry time, direction, entry price and stop (its target differs); exact setup identity "
        "for this configuration is not established.")


@pytest.mark.parametrize("change", [
    {"stop_ticks": 99800}, {"entry_ticks": 99885}, {"direction": "short"},
    {"entry_utc": "2026-04-13T00:08:00Z"}])
def test_no_record_without_the_same_execution(tmp_path, change):
    source = load_setup_record_source(_package(tmp_path / "pkg", [_row("S1_D80_W1_P1", "c")]))
    assert find_setup_record({**TRADE, **change}, configuration=LEADER, source=source,
                             axis_value_ids=VARIANT_IDS) is None
    assert NO_RECORD == "Setup zones weren't recorded for this study"


def test_settings_distance_uses_the_baseline_for_unnamed_settings():
    baseline = {"htf_timeframes": "htf_timeframes.1H-4H"}
    ids = {"htf_timeframes": "htf_timeframes.1H-4H", "exit_policy": "exit_policy.fixed_target_v1"}
    assert settings_distance(ids, {}, baseline) == 0
    assert settings_distance({**ids, "htf_timeframes": "htf_timeframes.1H"}, {}, baseline) == 1


# ── the reference trade (CALCULATIONS.md "Trade review") ──────────────────


@pytest.fixture(scope="module")
def reference():
    if not (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file():
        pytest.skip("the saved funded variation study is not on this computer")
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import market
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
        open_funded_study,
        ordered_trades,
    )

    study = open_funded_study(STORE, RESULT_ID)
    root = market.study_package_root(study.plan)
    if root is None:
        pytest.skip("the verified strategy package is not on this computer")
    variant = next(v for v in study.plan.variants if v.name == LEADER)
    row = next(t for t in ordered_trades(study, LEADER, TPT) if t["seq"] == 380)
    record = find_setup_record(row, configuration=LEADER, source=load_setup_record_source(root),
                               axis_value_ids=dict(variant.axis_value_ids),
                               in_verified_study=variant.in_verified_study,
                               base_configuration=study.plan.base_configuration)
    return study, root, row, record


def test_reference_setup_record(reference):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt

    _study, _root, row, record = reference
    assert row["account_number"] == 6 and row["entry_ticks"] == 99884
    assert record is not None and record.link == "same_execution"
    assert record.configuration == "S1_D80_W1_P1" and record.same_target
    assert not record.identity_established  # correction A6: related context only
    points = lambda t: fmt.points(t, from_ticks=True)  # noqa: E731
    assert (points(record.htf.low_ticks), points(record.htf.high_ticks)) == (
        "24,451.50", "25,086.75")
    assert fmt.chicago_long(record.htf.confirmed_utc) == "April 8, 2026, 1:00 AM"
    assert fmt.chicago_long(record.bar_open("tap_bar")) == "April 12, 2026, 5:18 PM"
    assert record.parent.timeframe_seconds == 300
    assert (points(record.parent.low_ticks), points(record.parent.high_ticks)) == (
        "24,937.25", "24,966.50")
    assert fmt.chicago_clock(record.parent.confirmed_utc) == "7:00 PM"
    assert (points(record.opposing.low_ticks), points(record.opposing.high_ticks)) == (
        "24,961.75", "24,962.50")
    assert fmt.chicago_clock(record.opposing.confirmed_utc) == "7:03 PM"
    assert fmt.chicago_clock(record.bar_open("inversion_bar")) == "7:05 PM"
    assert fmt.chicago_clock(record.entry_utc) == "7:07 PM"
    assert points(record.stop_ticks) == "24,949.75"


def test_member_configurations_link_every_trade_exactly(reference):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import ordered_trades

    study, root, _row, _record = reference
    source = load_setup_record_source(root)
    for variant in (v for v in study.plan.variants if v.in_verified_study):
        trades = ordered_trades(study, variant.name, TPT)
        records = [find_setup_record(t, configuration=variant.name, source=source,
                                     axis_value_ids=dict(variant.axis_value_ids),
                                     in_verified_study=True) for t in trades]
        assert trades and all(r is not None and r.link == "exact" for r in records)
        assert all(r.identity_established for r in records)  # correction A6
