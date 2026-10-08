"""Cached, read-only computations shared by the redesigned funded screens.

Everything is keyed by (store root, result id, firm, configuration, settings,
seed, path count) so a screen opens quickly the second time (rule 9, rule 15).
Conditional resampling results with a firm's ledger rules are also bound to the
model id, sampling method, trade slots, cutoff and a digest of the saved firm
terms and processing clock (correction A3), so a figure is never reused for a
different model or different account assumptions. The saved result is opened
once per process (``ifvg_lab_ui.funded_study``); nothing here writes a store or
recomputes a stored money figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import funded_measures as fm
from alpha_lab.agents.data_infra.ifvg.presentation.lab import market, resampling
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import (
    daily_results,
    ordered_trades,
    package_gate_thresholds,
)

__all__ = [
    "EarlyLossRace",
    "SummaryBundle",
    "default_firm_race",
    "early_loss_race",
    "firm_race_key",
    "firm_race_results",
    "configuration_names",
    "firm_loss_limit",
    "firm_terms",
    "pair_findings",
    "race_trigger",
    "index_minutes",
    "ranking",
    "saved_race_binding",
    "summary_bundle",
    "trade_values",
]


def _study(store_root: str, result_id: str):
    from ifvg_lab_ui import funded_study

    return funded_study(store_root, result_id)


@st.cache_data(show_spinner="Ranking the configurations…", max_entries=16)
def ranking(store_root: str, result_id: str, firm_key: str) -> list[fm.FundedRow]:
    return fm.ranking(_study(store_root, result_id), firm_key)


@st.cache_data(show_spinner=False, max_entries=16)
def configuration_names(store_root: str, result_id: str) -> dict[str, Any]:
    """Unique readable names for every configuration of one saved result."""

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.names import study_names

    study = _study(store_root, result_id)
    return study_names({key: study.settings(key) for key in study.configurations})


@st.cache_data(show_spinner=False, max_entries=16)
def drop_largest(store_root: str, result_id: str, firm_key: str) -> fm.DropLargestCheck:
    return fm.drop_largest_payout(_study(store_root, result_id), firm_key)


def trade_values(store_root: str, result_id: str, configuration: str,
                 firm_key: str) -> list[float]:
    study = _study(store_root, result_id)
    return [float(t.get("net_pnl_usd") or 0.0)
            for t in ordered_trades(study, configuration, firm_key)]


class _UnavailableError(Exception):
    """Raised inside a cached function so an unavailable package is never cached."""


@st.cache_resource(show_spinner="Reading the stored one-minute E-mini bars…", max_entries=4)
def _minutes(store_root: str, result_id: str, source_signature: tuple = ()):
    study = _study(store_root, result_id)
    minutes = market.load_study_index_minutes(study.plan, cutoff_utc=(
        study.result.get("period") or {}).get("cutoff_utc"))
    if minutes is None:
        raise _UnavailableError  # not cached: found on a later open once the package is back
    return minutes


def index_minutes(store_root: str, result_id: str):
    """One-minute E-mini bars of the study's verified package, or None (read only)."""

    try:
        study = _study(store_root, result_id)
        signature = ()
        if (getattr(getattr(study.plan, "source", None), "kind", None)
                == "verified_task_b_registered_inputs"):
            from alpha_lab.agents.data_infra.ifvg.presentation.lab.registered_market import (
                cache_signature,
            )

            signature = cache_signature(study.plan)
        return _minutes(store_root, result_id, signature)
    except _UnavailableError:
        return None


@st.cache_data(show_spinner=False, max_entries=8)
def _gate_thresholds(store_root: str, result_id: str) -> dict[str, Any]:
    study = _study(store_root, result_id)
    gates = package_gate_thresholds(market.study_package_root(study.plan))
    if gates is None:
        raise _UnavailableError
    return gates


def gate_thresholds(store_root: str, result_id: str) -> dict[str, Any] | None:
    try:
        return _gate_thresholds(store_root, result_id)
    except _UnavailableError:
        return None


def firm_terms(study, firm_key: str) -> dict[str, Any] | None:
    """The firm's frozen simulation terms saved with the result (never retyped)."""

    for profile in (study.result.get("settings") or {}).get("firm_profiles") or []:
        if profile.get("firm_key") == firm_key:
            return profile
    return None


def firm_loss_limit(study, firm_key: str) -> float | None:
    """The firm's loss allowance from the result's frozen firm terms (dollars)."""

    terms = firm_terms(study, firm_key) or {}
    allowance = terms.get("loss_allowance_cents")
    return int(allowance) / 100 if allowance else None


def race_trigger(study, firm_key: str) -> float | None:
    """The fixed diagnostic's default upper boundary: retained cushion + minimum request.

    Dollars, from the saved terms; used as a fixed closed-profit boundary only.
    """

    terms = firm_terms(study, firm_key) or {}
    cushion, minimum = terms.get("retained_cushion_cents"), terms.get("minimum_gross_request_cents")
    return (int(cushion) + int(minimum)) / 100 if cushion is not None and minimum else None


@st.cache_resource(show_spinner=False)
def firm_race_results() -> dict[tuple, Any]:
    """Conditional resampling results run on request, kept for this process (never saved).

    Keyed by :func:`firm_race_key`.
    """

    return {}


def firm_race_key(store_root: str, result_id: str, configuration: str, firm_key: str, *,
                  method: str, seed: int, paths: int, slots: int, cutoff_ns: int,
                  terms_digest: str) -> tuple:
    """The one key of a conditional resampling result (correction A3).

    (store root, result id, configuration, firm, model id, sampling method, seed,
    paths, trade slots, cutoff, terms digest). The fixed diagnostic's typed
    boundaries are not part of it: they never change the ledger run.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.firm_race import MODEL_ID

    return (str(store_root), str(result_id), str(configuration), str(firm_key), MODEL_ID,
            str(method), int(seed), int(paths), int(slots), int(cutoff_ns), str(terms_digest))


def saved_race_binding(study, configuration: str, firm_key: str) -> tuple[int, int, str] | None:
    """(trade slots, cutoff instant, terms digest) from the saved result, or None.

    The same saved inputs the ledger rules are built from: the pair's recorded
    trades, the result's period cutoff, the frozen firm terms and processing clock.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant
    from alpha_lab.agents.data_infra.ifvg.presentation.lab.firm_race import terms_digest

    terms = firm_terms(study, firm_key)
    clock = (study.result.get("settings") or {}).get("processing_clock")
    cutoff = utc_instant((study.result.get("period") or {}).get("cutoff_utc"))
    if terms is None or not clock or cutoff is None:
        return None
    return (len(ordered_trades(study, configuration, firm_key)), int(cutoff.value),
            terms_digest(dict(terms), dict(clock)))


def default_firm_race(store_root: str, result_id: str, configuration: str, firm_key: str):
    """The conditional resampling result at the default draw, if one was run; else None.

    The key is rebuilt from the saved result exactly as the Risk tab builds it; a
    result stored under another model id, other terms or another clock, method,
    seed, path count, slot count or cutoff is never returned.
    """

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.firm_race import (
        DEFAULT_FULL_PATHS,
        MODEL_ID,
    )

    store = firm_race_results()
    if not store:
        return None
    try:
        binding = saved_race_binding(_study(store_root, result_id), configuration, firm_key)
    except Exception:  # an unreadable result has no conditional figure
        return None
    if binding is None:
        return None
    slots, cutoff_ns, digest = binding
    key = firm_race_key(store_root, result_id, configuration, firm_key, method="blocks",
                        seed=fm.DEFAULT_SEED, paths=DEFAULT_FULL_PATHS, slots=slots,
                        cutoff_ns=cutoff_ns, terms_digest=digest)
    found = store.get(key)
    if found is None or getattr(found, "model_id", None) != MODEL_ID:
        return None
    return found


@dataclass(frozen=True)
class EarlyLossRace:
    """Which model the early-losses finding quotes (corrections A2, A3).

    ``basis`` "firm": conditional resampling with the firm's ledger rules (its
    ``model_id`` and ``detail``, e.g. "1,000 paths, to June 10, 2026"); "flat": the
    fixed closed-profit boundary diagnostic (``boundaries`` = (lower, upper),
    ``detail`` its path count).
    """

    died_share: float | None
    basis: str
    detail: str | None = None
    boundaries: tuple[float, float] | None = None
    model_id: str | None = None


def early_loss_race(bundle: SummaryBundle, store_root: str, result_id: str, configuration: str,
                    firm_key: str, firm: str) -> EarlyLossRace:
    """The conditional figure when it exists at the default draw; else the fixed diagnostic.

    Never runs anything (F2): the conditional result exists only after the owner
    ran it under Risk and simulation in this process.
    """

    race = bundle.race
    boundaries = (race.loss_limit, race.trigger) if race is not None else None
    full = default_firm_race(store_root, result_id, configuration, firm_key)
    if full is not None:
        to = (f", to {fmt.date_long(full.horizon_last_day)}"
              if getattr(full, "horizon_last_day", None) else "")
        return EarlyLossRace(full.died_share, "firm", f"{full.paths:,} paths{to}", boundaries,
                             full.model_id)
    return EarlyLossRace(race.died_share if race is not None else None, "flat",
                         f"{race.paths:,} resampled paths" if race is not None else None,
                         boundaries)


@dataclass(frozen=True)
class SummaryBundle:
    """Everything the Summary tab and the leader checks show for one pair."""

    row: fm.FundedRow
    ranges: fm.BootstrapRange | None
    sharpe: fm.SharpeConfidence | None
    concentration: fm.Concentration
    #: whole trades held to the daily deadline (kept; the findings use ``held_legs``)
    held_count: int
    held_net: float
    total_profit: float
    funded_profit_factor: float | None
    #: the fixed closed-profit boundary diagnostic at the saved defaults
    race: resampling.PayoutRace | None
    tie: market.IndexTie | None
    hold: market.BuyAndHold | None
    gates: list[fm.GateRow] = field(default_factory=list)
    loss_limit: float | None = None
    lost_first_accounts: int = 0
    trades_approximated: int = 0
    strategy_trades: int | None = None
    trading_days: int = 0
    #: the deadline trades' first halves and remainders (correction A9)
    held_legs: fm.HeldLegs | None = None
    #: days in the straight-line fit of daily results on the E-mini change
    beta_days: int | None = None


def summary_bundle(store_root: str, result_id: str, configuration: str,
                   firm_key: str) -> SummaryBundle:
    """Cached per pair and per whether the study's verified package is available now."""

    available = (index_minutes(store_root, result_id) is not None,
                 gate_thresholds(store_root, result_id) is not None)
    return _summary_bundle(store_root, result_id, configuration, firm_key, available)


@st.cache_data(show_spinner="Working out the measures…", max_entries=32)
def _summary_bundle(store_root: str, result_id: str, configuration: str, firm_key: str,
                    available: tuple[bool, bool]) -> SummaryBundle:
    study = _study(store_root, result_id)
    row = fm.funded_row(study, configuration, firm_key)
    trades = ordered_trades(study, configuration, firm_key)
    values = [float(t.get("net_pnl_usd") or 0.0) for t in trades]
    wins = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v <= 0)
    daily = daily_results(study, configuration, firm_key) if study.calendar else []
    minutes = index_minutes(store_root, result_id)
    tie = hold = None
    if minutes is not None and daily:
        tie = market.index_tie(daily, market.daily_closes(minutes))
        hold = market.buy_and_hold(minutes, study.calendar)
    loss_limit = firm_loss_limit(study, firm_key)
    trigger = race_trigger(study, firm_key)
    race = (resampling.payout_race(values, loss_limit=-loss_limit, trigger=trigger,
                                   seed=fm.DEFAULT_SEED)
            if values and loss_limit and trigger else None)
    conc = fm.concentration(study, configuration, firm_key)
    held_count, held_net = fm.held_to_close(study, configuration, firm_key)
    journeys = sorted(study.rows("account_journeys", configuration, firm_key),
                      key=lambda j: int(j.get("account_number") or 0))
    first_lost = 0
    for journey in journeys:
        if journey.get("failed_utc") and not int(journey.get("payouts_received") or 0):
            first_lost += 1
        else:
            break
    days_with_trade = len({t.get("trading_day") for t in trades})
    thresholds = gate_thresholds(store_root, result_id)
    measures = study.strategy_measures(configuration) or {}
    summary = study.summary(configuration, firm_key) or {}
    return SummaryBundle(
        row=row,
        ranges=fm.bootstrap_mean_ranges(values) if values else None,
        sharpe=fm.deflated_sharpe(study, configuration, firm_key) if daily else None,
        concentration=conc, held_count=held_count, held_net=held_net,
        total_profit=round(sum(values), 2),
        funded_profit_factor=(wins / losses) if losses > 0 else None,
        race=race, tie=tie, hold=hold,
        gates=fm.quality_gates(study, configuration, thresholds,
                               funded_days_with_trade=days_with_trade,
                               funded_best_day_share=conc.best_day_share),
        loss_limit=loss_limit, lost_first_accounts=first_lost,
        trades_approximated=int(summary.get("trades_with_approximated_minutes") or 0),
        strategy_trades=measures.get("trades"), trading_days=len(study.calendar),
        held_legs=fm.held_to_deadline_legs(study, configuration, firm_key),
        beta_days=tie.days if tie is not None else None,
    )


def pair_findings(bundle: SummaryBundle, store_root: str, result_id: str, configuration: str,
                  firm_key: str, firm: str) -> list[fm.Finding]:
    """One pair's findings, built the same way for the Summary tab and the overview.

    Both screens call this, so they quote the same model, legs and fit and count
    the same findings (never runs a simulation).
    """

    race = early_loss_race(bundle, store_root, result_id, configuration, firm_key, firm)
    held = bundle.held_legs
    tie = bundle.tie
    return fm.findings(
        largest_account_share=bundle.concentration.largest_account_share,
        five_largest_share=bundle.concentration.five_largest_trades_share,
        died_first_share=race.died_share, race_basis=race.basis, race_detail=race.detail,
        boundaries=race.boundaries, firm=firm,
        held=held, held_count=bundle.held_count, held_net=bundle.held_net,
        half_exit=held.half_exit if held is not None else False,
        total_profit=bundle.total_profit,
        beta_r2=tie.r_squared if tie is not None else None,
        beta_days=bundle.beta_days)
