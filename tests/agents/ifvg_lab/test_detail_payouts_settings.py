"""Configuration detail tabs "Payouts and accounts" (mock 04) and "Settings and evidence" (mock 08).

Pure helpers run on hand-made records and on the SYNTHETIC comparison fixture
(an engineering fixture, not evidence that a study ran). Reference checks read
the saved funded variation study ``5fa65149843484b1`` on this computer (read
only) and are skipped where it is absent. Nothing here writes a store; the one
review-folder test builds its folders under pytest's ``tmp_path``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

import ifvg_lab_detail_payouts as payouts  # noqa: E402
import ifvg_lab_detail_settings as settings  # noqa: E402

STORE = REPO / "data/ifsm_ui_replication/search/v1"
RESULT_ID = "5fa65149843484b143b64701a20aa063fb1e2da34708db8b4d6a2a2acbf4d09b"
LEADER = "S1-T1-H14-P1-L-SO"
TPT, MFF = "takeprofittrader", "myfundedfutures"
SAVED = (STORE / "funded_comparison_results" / RESULT_ID / "result.json").is_file()
needs_saved = pytest.mark.skipif(not SAVED, reason="the saved funded variation study is not on "
                                                   "this computer")

_FORBIDDEN = [
    re.compile(r"\b[0-9a-f]{32,}\b"),  # hashes
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # uuids
    re.compile(r"\|(takeprofittrader|myfundedfutures)"),  # pair ids
    re.compile(r"_cents\b|_utc\b|_usd\b|_ns\b|_v1\b"),  # field names and policy keys
    re.compile(r"[A-Za-z]:\\|/Users/|\.py\b|\.parquet\b"),
    re.compile(r"Traceback|Error\b"),
    re.compile(r"\bCS?T\b|\bCDT\b"),  # the zone is named once, on the rail
]

TPT_PROFILE = {"firm_key": TPT, "threshold_update": "intraday_peak_equity",
               "floor_lock_cents": 0}
MFF_PROFILE = {"firm_key": MFF, "threshold_update": "session_close_balance",
               "floor_lock_cents": 10000}


def _clean(text: str) -> list[str]:
    return [p.pattern for p in _FORBIDDEN if p.search(text)]


# ── pure helpers ──────────────────────────────────────────────────────────


def test_month_labels_and_ticks():
    assert payouts.month_label("2026-01", True) == "January (partial)"
    assert payouts.month_label("2026-02", False) == "February"
    assert payouts.month_label("2025-12", False, with_year=True) == "December 2025"
    ticks = payouts.money_ticks([-2000.0, -663.9])
    assert ticks[0] == (-2000, "−$2k") and ticks[-1] == (0, "$0")
    assert all(not label.startswith("$-") for _, label in ticks)


def test_full_range_settings_reads_saved_effective_gap_rule_before_review_publication():
    from types import SimpleNamespace

    study = SimpleNamespace(plan=SimpleNamespace(
        plan_schema="ifsm_correct_config_full_range_plan_v1",
        configurations=[SimpleNamespace(
            name="C01", effective_section_json=json.dumps({
                "htf_gap_invalidation_policy": "own_timeframe_close_v1",
            }),
        )],
    ))
    # No exported bindings are needed to display the frozen worker's setting.
    assert settings.gap_rule_text(study, "C01") == "A candle on its own chart closes through it"
    assert settings.gap_rule_text(study, "missing") is None


def test_month_ticks_start_at_the_first_point_then_each_month():
    import pandas as pd

    stamps = [pd.Timestamp("2026-01-12 17:00"), pd.Timestamp("2026-03-20 16:00")]
    labels = [label for _, label in payouts.month_ticks(stamps)]
    assert labels == ["January", "February", "March"]


def test_loss_limit_words_come_from_the_firm_rules():
    assert payouts.loss_limit_words(TPT_PROFILE) == "Loss limit (trails the account's high point)"
    assert payouts.loss_limit_words(MFF_PROFILE) == (
        "Loss limit (rises with each day's closing balance, stops at +$100.00)")
    assert payouts.loss_limit_words(None) == "Loss limit"


def _journey(**extra):
    base = {"account_number": 1, "created_utc": "2026-01-12T23:00:00Z",
            "failed_utc": "2026-01-15T19:56:07.305287043Z", "final_balance_usd": -663.9,
            "final_floor_usd": -661.33, "payouts_received": 0, "received_usd": 0.0,
            "largest_payout_usd": 0.0, "replaces_account_number": None,
            "failure_reason": "open-position equity reached the loss limit",
            "status_at_end": "Account lost"}
    base.update(extra)
    return base


def _trade(**extra):
    base = {"trade_ref": "a", "net_pnl_usd": 477.22, "balance_before_usd": -1141.12,
            "balance_after_usd": -663.9, "floor_before_usd": -1955.14,
            "floor_after_usd": -661.33, "scale_out_ns": 1, "scale_out_quantity": 5,
            "account_failed": True, "exit_utc": "2026-01-15T19:56:07Z"}
    base.update(extra)
    return base


def test_account_story_when_the_limit_trailed_up_during_the_last_trade():
    events = [{"status_after": "failed", "trade_ref": "a"}]
    story = payouts.account_story(_journey(), [_trade()], events,
                                  {"comparator": "at_or_below"}, TPT_PROFILE)
    assert story.headline == ("Lost before any payout: final balance −$663.90 against a loss "
                              "limit of −$661.33.")
    assert story.detail == (
        "Its last trade finished +$477.22. During it the account's high point rose, the limit "
        "trailed up behind it, and the pullback on the remaining half touched the new limit.")
    assert story.started == "Started January 12, 2026, 5:00 PM, the first account."


def test_account_story_without_a_limit_move_uses_the_starting_cushion():
    trade = _trade(net_pnl_usd=-125.28, balance_before_usd=211.92, floor_before_usd=100.0,
                   floor_after_usd=100.0, scale_out_ns=None, scale_out_quantity=0)
    story = payouts.account_story(
        _journey(payouts_received=2, final_balance_usd=86.64, final_floor_usd=100.0,
                 replaces_account_number=2),
        [trade], [{"status_after": "failed", "trade_ref": "a"}], {"comparator": "below"},
        MFF_PROFILE)
    assert story.headline.startswith("Lost after receiving a payout: final balance $86.64")
    assert story.detail == ("Its last trade finished −$125.28. It started $111.92 above the "
                            "limit, and the open loss went below it.")
    assert story.started.endswith("replacing Account 2.")


def test_account_story_other_reasons_and_live_accounts():
    trade = _trade(net_pnl_usd=-10.28, floor_after_usd=-1955.14)
    story = payouts.account_story(_journey(failure_reason="the entry cost reached the loss "
                                                          "limit"), [trade], [], None,
                                  TPT_PROFILE)
    assert story.detail == ("Its last trade finished −$10.28. The entry cost reached the loss "
                            "limit.")
    live = payouts.account_story(
        _journey(failed_utc=None, final_balance_usd=2100.0, final_floor_usd=0.0,
                 payouts_received=11, received_usd=28747.46, largest_payout_usd=5610.66,
                 status_at_end="Ready to trade"), [], [], None, TPT_PROFILE)
    assert live.headline == ("Still open at the end (ready to trade): balance $2,100.00 against "
                             "a loss limit of $0.00.")
    assert live.detail == ("It received 11 payouts, $28,747.46 after the split; the largest was "
                           "$5,610.66.")


def test_account_path_by_trade():
    created = {"balance_usd": 0.0, "floor_usd": -2000.0, "ts_utc": "2026-01-12T23:00:00Z"}
    trades = [_trade(balance_after_usd=-190.28, floor_after_usd=-1955.14, net_pnl_usd=-190.28),
              _trade()]
    path = payouts.account_path(created, trades)
    assert [p.label for p in path] == ["Start", "Trade 1", "Trade 2"]
    assert [p.balance for p in path] == [0.0, -190.28, -663.9]
    assert [p.limit for p in path] == [-2000.0, -1955.14, -661.33]


def test_money_spans_escape_and_keep_amounts_whole():
    html = str(payouts.money_spans("final −$663.90 <b> +$477.22"))
    assert "&lt;b&gt;" in html
    assert html.count('white-space:nowrap">') == 2 and ">−$663.90<" in html


def test_chart_list_text():
    assert settings.chart_list_text("one-hour and four-hour") == "1-hour and 4-hour"
    assert settings.chart_list_text("one-hour") == "1-hour"
    assert settings.chart_list_text(
        "one-minute, three-minute, five-minute, ten-minute, fifteen-minute and thirty-minute"
    ) == "1, 3, 5, 10, 15 and 30 minutes"
    assert settings.chart_list_text("weekly") == "Weekly"  # unknown: shown as saved


def test_decisions_are_shown_in_full_and_assumptions_flagged():
    result = {"owner_decisions": [
        {"decided_on": "2026-09-22", "subject": "S", "decision": "x" * 900,
         "status": "assumption"},
        {"decided_on": "2026-09-23", "subject": "T", "decision": "y", "status":
            "owner_confirmed"}]}
    rows = settings.decision_rows(result)
    assert rows[0] == ("Sep 22", "S", "x" * 900, "Assumption, not from a published source", True)
    assert rows[1][3] == "Confirmed by you" and rows[1][4] is False
    html = str(settings.decisions_card(settings.SettingsView((), (), (), (), tuple(rows), None)))
    assert "x" * 900 in html  # never truncated


_NAME_CELL = re.compile(r'<div class="lab-cell-main">(.*?)</div>'
                        r'(?:<div class="lab-cell-sub">(.*?)</div>)?')


def _metric(configuration: str, net_r: float) -> dict:
    return {"configuration": configuration, "trades": 10, "long_trades": 10, "short_trades": 0,
            "win_rate_pct": 50.0, "net_r_after_costs": net_r, "expectancy_r_per_trade": 0.1,
            "profit_factor": 1.2, "max_drawdown_r": 2.0, "longest_trading_days_under_water": 3}


def test_strategy_table_never_shows_two_configurations_with_one_name():
    """Review fix: configurations that differ only in the opposing-gap settings got the same
    two lines (the other saved result, 92f5a08d…)."""

    from types import SimpleNamespace

    base = [("Entry hours", "Original three windows (8:30 AM to 11:00 AM, …)"),
            ("Direction", "Long only"), ("Profit target", "1 to 1 (equal to the initial risk)"),
            ("Exit rule", "The whole position at the target or stop"),
            ("Higher-timeframe gap charts", "one-hour and four-hour"),
            ("Supporting (parent) charts", "three-minute")]
    settings_by = {
        f"S0_D{d}_W{w}_P1": base + [
            ("Largest distance from the parent gap to the opposing gap", f"{d} ticks"),
            ("Smallest opposing gap", f"{w} ticks")]
        for d in (80, 160) for w in (1, 4)}
    study = SimpleNamespace(
        configurations=list(settings_by), settings=lambda key: list(settings_by[key]),
        result={"tables": {"strategy_metrics": [_metric(k, 4.0 - i)
                                                for i, k in enumerate(settings_by)]}})
    cells = _NAME_CELL.findall(str(settings.strategy_table(study, "S0_D80_W1_P1")))
    assert len(cells) == 4
    assert len(set(cells)) == 4, cells  # every row names its configuration apart
    assert all("opposing distance" in sub and "minimum" in sub for _, sub in cells)


STORE_OTHER = "92f5a08d3b81ae83eddde0525a04220ea9e1e9eb849d1118303d5de7efd743f4"


@pytest.mark.skipif(not (STORE / "funded_comparison_results" / STORE_OTHER / "result.json")
                    .is_file(), reason="the other saved funded result is not on this computer")
def test_strategy_table_on_the_other_saved_result():
    """92f5a08d… saved no strategy measures: the table says so (no rows to confuse); its
    names, as the table would build them, are all different."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import study_names

    other = open_funded_study(STORE, STORE_OTHER)
    html = str(settings.strategy_table(other, other.configurations[0]))
    if (other.result.get("tables") or {}).get("strategy_metrics"):
        cells = _NAME_CELL.findall(html)
        assert len(set(cells)) == len(cells) > 1
    else:
        assert "Strategy measures without accounts are not in this study" in html
    names = study_names({k: other.settings(k) for k in other.configurations})
    assert len({n.full for n in names.values()}) == len(other.configurations) == 32


# ── synthetic fixture: every pair renders without internal names ──────────


@pytest.fixture(scope="module")
def fixture_study():
    sys.path.insert(0, str(REPO))
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import study_from_result
    from tests.propsim.funded.comparison_fixture import comparison_fixture_result

    return study_from_result(comparison_fixture_result(), result_id="a" * 64)


def test_fixture_views_render_plain_text(fixture_study, tmp_path):
    study = fixture_study
    for summary in study.result["summaries_cents"].values():
        cfg, firm = summary["configuration"], summary["firm_key"]
        view = settings.build_settings_view(study, cfg, firm, summary["firm"], tmp_path)
        html = "".join(str(x) for x in (settings.top_cards(view), settings.limitations_card(view),
                                        settings.decisions_card(view),
                                        settings.about_card(study.result)))
        assert not _clean(html), _clean(html)
        assert view.review is None  # no published folder under tmp_path
        if summary.get("status") != "Completed":
            continue
        pview = payouts.build_payouts_view(study, cfg, firm)
        html = "".join(str(x) for x in (payouts.top_cards(pview), payouts.months_card(pview),
                                        payouts.account_table(pview)))
        html += "".join(a.story.headline + a.story.detail + str(payouts.payout_table(a))
                        for a in pview.accounts)
        assert not _clean(html), _clean(html)
        assert len(pview.accounts) == int(summary["accounts_purchased"])


def test_fixture_verification_without_approval_or_folder(fixture_study):
    items = settings.verification_items(fixture_study.result, "S0_D160", TPT,
                                        "TakeProfitTrader", None)
    kind, text = items[-1]
    assert kind == "warn"
    assert ("No approval is recorded" in text or "Engineering sample" in text), text
    result = dict(fixture_study.result, purpose="historical_comparison",
                  approval={"approved_on": "2026-09-23"})
    kind, text = settings.verification_items(result, "S0_D160", TPT, "TakeProfitTrader",
                                             None)[-1]
    assert (kind, text) == ("check", "Approved by you September 23, 2026.")


def test_unperformed_equality_checks_remain_distinct_from_failed_checks():
    evidence = {"no_account_replay_equals_saved_study": None,
                "resumed_run_identical": None, "resume_check_requested": False}
    result = {"tables": {"execution_evidence": [evidence]}}
    items = settings.verification_items(result, "synthetic", TPT, "TakeProfitTrader", None)
    texts = " ".join(text for _, text in items)
    assert "Historical no-account equality was not checked" in texts
    assert "Production resume equality was not checked" in texts
    assert "0 of 1" not in texts
    evidence.update(no_account_replay_equals_saved_study=False,
                    resumed_run_identical=False, resume_check_requested=True)
    texts = " ".join(text for _, text in settings.verification_items(
        result, "synthetic", TPT, "TakeProfitTrader", None))
    assert texts.count("0 of 1") == 2
    assert "was not checked" not in texts


def test_full_range_ordinary_table_uses_exact_saved_parent_charts():
    from types import SimpleNamespace

    parents = ["1m", "3m", "5m", "10m", "15m", "30m"]
    plan = SimpleNamespace(
        plan_schema="ifsm_correct_config_full_range_plan_v1",
        configurations=[SimpleNamespace(
            name="C03", effective_section_json=json.dumps({"parent_timeframes": parents}))],
    )
    study = SimpleNamespace(
        configurations=["C03"],
        settings=lambda key: [("Supporting (parent) charts", "one-minute, three-minute")],
        plan=plan,
        result={"tables": {"strategy_metrics": [_metric("C03", 1.0)]}},
    )
    html = str(settings.strategy_table(study, "C03"))
    assert "1, 3, 5, 10, 15, 30-minute parents" in html
    assert "1- and 3-minute parents" not in html


# ── published review folder (tmp_path only) ───────────────────────────────


def _publish(parent: Path, name: str, result_id: str, rows: list[dict[str, str]],
             *, zip_only: bool = False) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    manifest = {"funded_comparison_result_id": result_id, "export_version": 1,
                "published_at_utc": "2026-09-23T19:34:33Z",
                "files": [{"path": "approximated_minutes.csv", "bytes": len(data),
                           "sha256": hashlib.sha256(data).hexdigest()}]}
    if zip_only:
        with zipfile.ZipFile(parent / f"{name}.zip", "w") as archive:
            archive.writestr(f"{name}/approximated_minutes.csv", data)
            archive.writestr(f"{name}/run_manifest.json", json.dumps(manifest))
        return
    folder = parent / name
    folder.mkdir(parents=True)
    (folder / "approximated_minutes.csv").write_bytes(data)
    (folder / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _row(could: bool) -> dict[str, str]:
    row = {c: "False" for c in settings._APPROX_EFFECT_COLUMNS}
    row["ordering_can_change_account_survival"] = str(could)
    return row


def test_latest_review_folder_picks_this_results_highest_version(tmp_path):
    parent = tmp_path / "reports/funded_comparison"
    parent.mkdir(parents=True)
    rid = "b" * 64
    prefix = f"funded_comparison_{rid[:16]}_export_v"
    _publish(parent, f"{prefix}1", rid, [_row(False), _row(False)])
    _publish(parent, f"{prefix}2", rid, [_row(True), _row(False)], zip_only=True)
    _publish(parent, f"{prefix}3", "c" * 64, [_row(False)])  # another result, same prefix
    before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    review = settings.latest_review_folder(tmp_path, rid)
    assert review is not None and review.version == 2 and review.folder is None
    assert settings.approximation_effects(review) == (2, 1)
    older = settings.ReviewFolder(f"{prefix}1", 1, parent / f"{prefix}1", None, None,
                                  review.files | {"approximated_minutes.csv": "0" * 64})
    assert settings.approximation_effects(older) is None  # a hash mismatch is not trusted
    in_memory = settings.latest_review_folder(tmp_path, rid)
    assert zipfile.ZipFile(io.BytesIO(settings.folder_zip_bytes(in_memory))).namelist()
    folder_only = settings.ReviewFolder(f"{prefix}1", 1, parent / f"{prefix}1", None, None, {})
    names = zipfile.ZipFile(io.BytesIO(settings.folder_zip_bytes(folder_only))).namelist()
    assert f"{prefix}1/approximated_minutes.csv" in names
    assert sorted(p.as_posix() for p in tmp_path.rglob("*")) == before  # read only
    assert settings.latest_review_folder(tmp_path, "d" * 64) is None


# ── publish the review folder again (tmp_path stores; publishing replaced) ──


PLAN = "e" * 64


def _run_state(repo: Path, app: str = "ifsm", plan_id: str = PLAN, **fields) -> dict:
    """A funded comparison run state under ``repo`` (tmp_path), as the worker writes it."""

    from ifvg_funded_comparison_study import comparison_state_root
    from ifvg_lab_nav import app_roots

    roots = app_roots(repo)[app]
    folder = comparison_state_root(roots) / plan_id
    folder.mkdir(parents=True, exist_ok=True)
    state = {"plan_id": plan_id, "status": "Completed", "result_id": RESULT_ID,
             "review_folder": None, "review_error": None, **fields}
    (folder / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return roots


def _target(repo: Path, app: str = "ifsm", **extra) -> dict:
    from ifvg_lab_nav import app_roots

    return {"result_id": RESULT_ID, "store_root": str(app_roots(repo)[app]["store_root"]),
            "app": app, "status": "Completed", "name": "Funded variation study", **extra}


def test_publish_again_is_offered_only_when_the_run_state_records_a_review_error(tmp_path):
    from types import SimpleNamespace

    study = SimpleNamespace(result_id=RESULT_ID, plan_id=PLAN)
    roots = _run_state(tmp_path, review_folder="reports/x")  # published: nothing to offer
    target = _target(tmp_path, plan_id=PLAN, study_key=PLAN)
    assert settings.publish_again_route(target, study, roots) is None
    _run_state(tmp_path)  # completed, no error recorded
    assert settings.publish_again_route(target, study, roots) is None
    _run_state(tmp_path, review_error="OSError: disk full")
    route = settings.publish_again_route(target, study, roots)
    assert route == settings.PublishAgain(
        plan_id=PLAN, store_root=Path(target["store_root"]),
        state_root=tmp_path / "data/ifsm_ui_replication/funded_comparison_jobs",
        reports_root=tmp_path / "reports")
    # a deep link carries no plan id: the verified result's own plan id finds the state
    link = _target(tmp_path, study_key=RESULT_ID)
    assert settings.publish_again_route(link, study, roots) == route
    # a state of another result is never used
    other = SimpleNamespace(result_id="f" * 64, plan_id=PLAN)
    assert settings.publish_again_route(dict(link, result_id="f" * 64), other, roots) is None
    # the other application's result, opened from this one, uses that application's state
    from ifvg_lab_nav import app_roots

    main = app_roots(tmp_path)["main"]
    assert settings.publish_again_route(target, study, main) == route
    assert settings.publish_again_route(_target(tmp_path, app="main", plan_id=PLAN), study,
                                        main) is None
    assert settings.publish_again_route(dict(target, plan_id="../x"),
                                        SimpleNamespace(result_id=RESULT_ID, plan_id=None),
                                        roots) is None


def _publish_page():
    """AppTest page: the publish-again block of Settings and evidence for one result."""

    from types import SimpleNamespace

    import ifvg_lab_detail_settings as settings
    import streamlit as st

    ctx = SimpleNamespace(target=st.session_state["target"], study=st.session_state["study"],
                          roots=st.session_state["roots"], result_id=st.session_state["rid"],
                          store_root=st.session_state["target"]["store_root"])
    settings.publish_again(st, ctx)
    st.html("<p>page end</p>")


def _publish_app(tmp_path: Path):
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_publish_page, default_timeout=60)
    at.session_state["target"] = _target(tmp_path, plan_id=PLAN)
    at.session_state["study"] = SimpleNamespace(result_id=RESULT_ID, plan_id=PLAN)
    at.session_state["roots"] = _run_state(tmp_path)
    at.session_state["rid"] = RESULT_ID
    return at


def test_publish_again_button_calls_the_earlier_publish_path(tmp_path, monkeypatch):
    from alpha_lab.propsim.funded import comparison_runner

    calls: list[dict] = []
    monkeypatch.setattr(comparison_runner, "publish_comparison_review",
                        lambda **kwargs: calls.append(kwargs) or {})
    at = _publish_app(tmp_path).run()
    assert not at.exception, at.exception
    assert not [b for b in at.button if b.label == settings.PUBLISH_LABEL]  # no error: none
    _run_state(tmp_path, review_error="OSError: disk full")
    at.run()
    assert not at.exception, at.exception
    body = "\n".join(str(x.proto.body) for x in at.get("html"))
    assert settings.PUBLISH_WARNING in body and not _clean(body)
    button = next(b for b in at.button if b.label == settings.PUBLISH_LABEL)
    assert button.help and not calls
    button.click().run()
    assert not at.exception, at.exception
    assert calls == [{"plan_id": PLAN,
                      "store_root": tmp_path / "data/ifsm_ui_replication/search/v1",
                      "state_root": tmp_path / "data/ifsm_ui_replication/funded_comparison_jobs",
                      "reports_root": tmp_path / "reports"}]
    assert all(REPO not in Path(v).parents for v in calls[0].values() if isinstance(v, Path))


def test_publish_again_refusal_is_shown_not_raised(tmp_path, monkeypatch):
    from alpha_lab.propsim.funded import comparison_runner

    def refuse(**_kwargs):
        raise ValueError("only a completed, verified comparison can be published")

    monkeypatch.setattr(comparison_runner, "publish_comparison_review", refuse)
    at = _publish_app(tmp_path)
    _run_state(tmp_path, status="Incomplete", review_error="OSError: disk full")
    at.run()
    next(b for b in at.button if b.label == settings.PUBLISH_LABEL).click().run()
    assert not at.exception, at.exception
    assert [e.value for e in at.error] == [
        "The review folder was not published: only a completed, verified comparison can be "
        "published"]


# ── reference: the saved funded variation study (read only) ───────────────


@pytest.fixture(scope="module")
def study():
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import open_funded_study

    return open_funded_study(STORE, RESULT_ID)


@needs_saved
def test_reference_payouts_and_accounts(study):
    view = payouts.build_payouts_view(study, LEADER, TPT)
    hours = {p.key: p.hours for p in view.parts}
    assert hours["trading"] == 268.0 and hours["processing"] == 864.0
    assert view.refused_processing == 42 and view.refused_protection == 0
    assert view.waiting == "none · assumed instant"
    timing = {label: str(value) for label, value in view.timing}
    assert timing["First payout received"] == "February 26, 2026, 4:00 PM"
    assert re.sub("<[^>]+>", "", timing["Stop exits filled worse than the stop"]) == (
        "23 of 96 · $150.00 in total")
    assert "$510.00" in timing["Most account spending not yet paid back"]
    assert "February 11, 10:27 AM" in timing["Most account spending not yet paid back"]
    first = view.accounts[0]
    assert first.story.headline == ("Lost before any payout: final balance −$663.90 against a "
                                    "loss limit of −$661.33.")
    assert first.story.detail == (
        "Its last trade finished +$477.22. During it the account's high point rose, the limit "
        "trailed up behind it, and the pullback on the remaining half touched the new limit.")
    assert [p.label for p in first.path][-1] == "Trade 5"
    assert first.path[-1].balance == -663.9 and first.path[-1].limit == -661.33
    assert [a.lost for a in view.accounts] == [True] * 5 + [False]
    assert view.months[-1].cumulative_net_usd == 30781.88
    assert (view.received_cents, view.costs_cents, view.net_cents) == (3139388, 61200, 3078188)
    assert view.limit_words == "Loss limit (trails the account's high point)"


@needs_saved
def test_reference_every_pair_explains_every_lost_account(study):
    for cfg in study.configurations:
        for firm in (TPT, MFF):
            view = payouts.build_payouts_view(study, cfg, firm)
            for account in view.accounts:
                assert account.story.detail and "None" not in account.story.detail
                if account.lost:
                    assert account.story.headline.startswith("Lost ")


@needs_saved
def test_reference_settings_and_evidence(study):
    view = settings.build_settings_view(study, LEADER, TPT, "TakeProfitTrader", REPO)
    rows = {label: str(value) for label, value in view.settings}
    assert rows["Big gap charts"] == "1-hour and 4-hour"
    assert rows["When a big gap stops counting"] == "A candle on its own chart closes through it"
    assert rows["Supporting charts"] == "1, 3, 5, 10, 15 and 30 minutes"
    assert rows["Position size and cost"] == "10 micros per trade · $0.514 per micro per fill"
    assert rows["Traded product"] == "Micro E-mini Nasdaq-100"
    assert rows["Strategy engine"].startswith("Research version with the half exit")
    texts = [text for _, text in view.verification]
    assert texts[0].startswith("515,388 of 515,438 position minutes used recorded exchange "
                               "trades. 50 used a labeled one-minute approximation, 2 of them "
                               "in this configuration")
    assert "Receipts, account costs and balances reconcile for all 128 results." in texts
    assert any(t.endswith("64 of 64.") and "switched off" in t for t in texts)
    assert "Stopping halfway and resuming gave identical results: 64 of 64." in texts
    assert "Approved by you September 23, 2026." in texts
    assert [kind for kind, text in view.verification if "micro positions" in text] == ["warn"]
    stop = view.corrections[0]
    assert stop.title == "Stop exits filled worse than the stop"
    assert stop.this_configuration == ("This configuration at TakeProfitTrader: 23 exits, "
                                       "$150.00 (was $225.00).")
    assert len(view.limitations) == len(study.result["limitations"])
    assert len(view.decisions) == len(study.result["owner_decisions"])
    assert sum(1 for d in view.decisions if d[4]) == 1
    html = str(settings.top_cards(view)) + str(settings.decisions_card(view))
    assert not _clean(html), _clean(html)


# ── the running tabs (AppTest; read only) ─────────────────────────────────


def _detail_app():
    """One detail tab rendered with the shell's DetailContext (the shell's clickable header
    is a components-v2 element AppTest cannot mount, so the tab is called directly)."""

    import importlib
    from pathlib import Path

    import streamlit as st
    from ifvg_lab_cache import ranking
    from ifvg_lab_funded import DetailContext
    from ifvg_lab_nav import app_roots, funded_context
    from ifvg_lab_ui import funded_study

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import configuration_name

    roots = app_roots(Path(st.session_state["repo"]))["ifsm"]
    rid, cfg, firm = st.session_state["rid"], st.session_state["cfg"], "takeprofittrader"
    store = str(roots["store_root"])
    study = funded_study(store, rid)
    row = next(r for r in ranking(store, rid, firm) if r.configuration == cfg)
    target = {"result_id": rid, "store_root": store, "app": "ifsm", "status": "Completed",
              "name": "Funded variation study", "study_key": rid}
    if st.session_state.get("roots_repo"):  # run states and reports under tmp_path
        roots = app_roots(Path(st.session_state["roots_repo"]))["ifsm"]
    ctx = DetailContext(target=target, study=study, store_root=store, result_id=rid,
                        configuration=cfg, firm_key=firm, firm=dict(study.firms)[firm],
                        name=configuration_name(study.settings(cfg), cfg), row=row,
                        roots=roots, context=funded_context(rid))
    module = importlib.import_module(st.session_state["module"])
    if hasattr(module, "header_right"):
        module.header_right(st, ctx)
    module.render(st, ctx)


def _app(tab: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_detail_app, default_timeout=120)
    at.session_state["repo"] = str(REPO)
    at.session_state["rid"] = RESULT_ID
    at.session_state["cfg"] = LEADER
    at.session_state["module"] = tab
    return at


@needs_saved
def test_app_payouts_tab_remembers_the_account_per_pair():
    at = _app("ifvg_lab_detail_payouts")
    started = time.perf_counter()
    at.run()
    first = time.perf_counter() - started
    assert not at.exception, at.exception
    picker = next(w for w in at.selectbox if str(w.key).startswith("ifvg_lab_v1_account_"))
    assert picker.label == "Show one account" and picker.value == 1
    picker.set_value(6).run()
    assert not at.exception, at.exception
    contexts = at.session_state["funded_comparison_v1_selected_context"]
    assert contexts[RESULT_ID]["account"] == {"pair": f"{LEADER}|{TPT}", "number": 6}
    started = time.perf_counter()
    at.run()
    cached = time.perf_counter() - started
    assert cached < 5, (first, cached)
    html = "\n".join(str(h.proto.body) for h in at.get("html"))
    assert "Where the account time went" in html and "3,574 hours in all" in html
    assert "Account 6 payouts" in html
    # leave the tab and come back: the account is restored for this pair only
    at.session_state["module"] = "ifvg_lab_detail_settings"
    at.run()
    at.session_state["module"] = "ifvg_lab_detail_payouts"
    at.run()
    picker = next(w for w in at.selectbox if str(w.key).startswith("ifvg_lab_v1_account_"))
    assert picker.value == 6
    # another configuration never inherits it
    at.session_state["cfg"] = "S0-T1-H1-P1-L-SO"
    at.run()
    assert not at.exception, at.exception
    picker = next(w for w in at.selectbox if str(w.key).startswith("ifvg_lab_v1_account_"))
    assert picker.value == 1


@needs_saved
def test_app_settings_tab_renders_with_the_download():
    at = _app("ifvg_lab_detail_settings").run()
    assert not at.exception, at.exception
    html = "\n".join(str(h.proto.body) for h in at.get("html"))
    for title in ("Configuration settings", "Verification", "Reporting corrections · 2",
                  "What this result can", "Your decisions and assumptions used",
                  "Strategy measures without accounts"):
        assert title in html, title
    downloads = at.get("download_button")
    assert len(downloads) == 1
    assert not [b for b in at.button if b.label == settings.PUBLISH_LABEL]


@needs_saved
def test_app_settings_tab_offers_publish_again_after_a_failed_export(tmp_path, monkeypatch):
    """The saved study's Settings tab with its run state (and reports) under tmp_path."""

    from alpha_lab.propsim.funded import comparison_runner

    calls: list[dict] = []
    monkeypatch.setattr(comparison_runner, "publish_comparison_review",
                        lambda **kwargs: calls.append(kwargs) or {})
    plan_id = "78406b15cdd9c1363b7cf626f7e45d48e58f6aa4e634a25dc00a3e57bdb28336"
    _run_state(tmp_path, plan_id=plan_id, review_error="OSError: disk full")
    at = _app("ifvg_lab_detail_settings")
    at.session_state["roots_repo"] = str(tmp_path)
    at.run()
    assert not at.exception, at.exception
    html = "\n".join(str(h.proto.body) for h in at.get("html"))
    assert settings.PUBLISH_WARNING in html and "Configuration settings" in html
    assert [b.disabled for b in at.button if b.label == settings.DOWNLOAD_LABEL] == [True]
    next(b for b in at.button if b.label == settings.PUBLISH_LABEL).click().run()
    assert not at.exception, at.exception
    assert calls == [{"plan_id": plan_id, "store_root": STORE,
                      "state_root": tmp_path / "data/ifsm_ui_replication/funded_comparison_jobs",
                      "reports_root": tmp_path / "reports"}]


# ── theme: CSS variables in the HTML, the active palette in the figures ───


_COLOR_LITERAL = re.compile(r"#[0-9A-Fa-f]{6}\b|rgba?\(")


def _first_completed(study) -> tuple[str, str, str]:
    summary = next(s for s in study.result["summaries_cents"].values()
                   if s.get("status") == "Completed")
    return summary["configuration"], summary["firm_key"], summary["firm"]


def test_payouts_html_names_palette_variables_and_the_view_holds_no_color(fixture_study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    cfg, firm, _name = _first_completed(fixture_study)
    view = payouts.build_payouts_view(fixture_study, cfg, firm)
    html = "".join(str(x) for x in (payouts.top_cards(view), payouts.months_card(view),
                                    payouts.account_table(view), payouts.fact_notes_panel(view),
                                    *(payouts.payout_table(a) for a in view.accounts)))
    found = _COLOR_LITERAL.search(html)
    assert found is None, found.group(0)
    assert "background:var(--lab-ink)" in html and "var(--lab-light-rule)" in html
    # the cached view (``_cached_view``) carries palette KEYS, never a theme's color value
    assert [p.color_key for p in view.parts] == ["ink", "blue_line", "blue", "control_border",
                                                  "light_rule"]
    assert all(p.color_key in theme.COLORS for p in view.parts)
    source = Path(payouts.__file__).read_text(encoding="utf-8")
    found = _COLOR_LITERAL.search(source)
    assert found is None, found.group(0)


def test_payouts_figures_take_the_dark_palette_when_the_page_is_dark(fixture_study):
    from alpha_lab.agents.data_infra.ifvg.presentation.lab import theme

    cfg, firm, _name = _first_completed(fixture_study)
    view = payouts.build_payouts_view(fixture_study, cfg, firm)
    account = view.accounts[0]
    light = payouts.cash_figure(view)
    assert [t.line.color for t in light.data] == [theme.COLORS[k]
                                                  for k in ("blue", "ink", "orange")]
    theme.set_theme_resolver(lambda: "dark")
    try:
        cash = payouts.cash_figure(view)
        balance = payouts.account_figure(account, view.limit_words)
    finally:
        theme.set_theme_resolver(None)
    dark = theme.DARK_COLORS
    assert [t.line.color for t in cash.data] == [dark["blue"], dark["ink"], dark["orange"]]
    assert cash.layout.plot_bgcolor == dark["chart_ground"]
    assert balance.data[0].line.color == dark["orange"]
    assert balance.data[1].line.color == dark["ink"]
    if account.lost:
        assert balance.data[2].marker.color == dark["orange"]
    assert balance.layout.annotations[0].font.color == dark["orange"]
    assert payouts.cash_figure(view).data[0].line.color == theme.COLORS["blue"]  # light again


def test_settings_html_names_palette_variables_not_literals(fixture_study, tmp_path):
    cfg, firm, name = _first_completed(fixture_study)
    view = settings.build_settings_view(fixture_study, cfg, firm, name, tmp_path)
    assumption = settings.SettingsView(
        (), (), (), (), (("Sep 22", "S", "x", "Assumption, not from a published source", True),),
        None)
    html = "".join(str(x) for x in (settings.top_cards(view), settings.limitations_card(view),
                                    settings.decisions_card(view),
                                    settings.decisions_card(assumption),
                                    settings.strategy_table(fixture_study, cfg),
                                    settings.mark("warn"), settings.mark("check")))
    found = _COLOR_LITERAL.search(html)
    assert found is None, found.group(0)
    assert "color:var(--lab-orange-dark)" in html  # the assumption status
    assert "border:2px solid var(--lab-orange)" in html  # the caution mark
    assert 'style="color:var(--lab-blue);' in html  # the check mark
    assert "border-top:1px solid var(--lab-light-rule)" in html
    source = Path(settings.__file__).read_text(encoding="utf-8")
    found = _COLOR_LITERAL.search(source)
    assert found is None, found.group(0)


def test_summary_gate_marks_name_palette_variables_not_literals():
    from types import SimpleNamespace

    import ifvg_lab_detail_summary as summary

    gates = [SimpleNamespace(gate="Trades", required="30 or more", value="96", note=None,
                             needs_decision=False, passed=True),
             SimpleNamespace(gate="Profit factor", required="1.2 or more", value="1.1",
                             note="funded trades, for reference", needs_decision=True,
                             passed=False),
             SimpleNamespace(gate="Sharpe", required="—", value=None, note=None,
                             needs_decision=False, passed=None)]
    html = str(summary._gates_card(SimpleNamespace(gates=gates)))
    assert '<b style="color:var(--lab-blue)">Pass</b>' in html
    assert '<b style="color:var(--lab-orange)">Fail</b>' in html
    found = _COLOR_LITERAL.search(html)
    assert found is None, found.group(0)
    source = Path(summary.__file__).read_text(encoding="utf-8")
    found = _COLOR_LITERAL.search(source)
    assert found is None, found.group(0)
