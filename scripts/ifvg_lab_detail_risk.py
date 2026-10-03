"""Configuration detail · Risk and simulation (mocks 05, 05b).

Every chart here resamples ONE configuration's own recorded funded trades at ONE
firm WITH REPLACEMENT (blocks of 10 consecutive trades, or one trade at a time):
a path can repeat some trades and leave out others, so its total differs from
the recorded total. The charts describe the recorded trades under that sampling
model; they are never a forecast (``CALCULATIONS.md`` "Risk and simulation";
analytical corrections A2–A5, September 25, 2026).

- the fixed closed-profit boundary diagnostic (lower and upper boundary inputs,
  defaults read from the firm's frozen terms in the saved result): which fixed
  cumulative closed-profit level a path crosses first — not the firm's account
  rules; then, on request, conditional resampling of the recorded trades with
  the firm's ledger rules (:mod:`...lab.firm_race`), with its limitations always
  shown;
- the resampled equity fan with the "Keep streaks together" / "Draw single
  trades" switch and the comparison of both methods with what happened;
- where the paths end up and the longest losing streak (they follow the switch);
- the sampled closed-profit drawdown from a previous high (no withdrawals, no
  account floor — not the share of accounts that fail).

Rule 9: a fixed default seed (``funded_measures.DEFAULT_SEED``) and the path
count are shown; "Run again" draws a new seed, says which, and offers the way
back. Results are cached by study, configuration, firm, settings, seed and
path count. Nothing here writes a store.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Sequence
from typing import Any

import numpy as np
import streamlit as st

from alpha_lab.agents.data_infra.ifvg.presentation.lab import firm_race as fr
from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab import resampling as rs
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import DEFAULT_SEED
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import (
    FONT_MONO,
    FONT_SANS,
    css_var,
    palette,
    rgba,
    style_chart,
)

__all__ = [
    "DEFAULT_PATHS",
    "PATH_CHOICES",
    "header_right",
    "new_seed",
    "parse_money",
    "render",
    "seed_note",
]

PREFIX = "ifvg_lab_v1_"
#: tab-specific styles (injected once per run, style-only so it is never sanitized away)
RISK_CSS = (
    ".lab-risk-table .lab-table td, .lab-risk-table .lab-table th { padding: 9px 10px; }"
    ".lab-risk-table.compact .lab-table td, .lab-risk-table.compact .lab-table th "
    "{ font-size: 13px; padding: 9px 8px; }"
    ".lab-risk-table.compact .lab-table td.num { white-space: nowrap; }"
)
PATH_CHOICES = (2_000, 5_000, 20_000)
DEFAULT_PATHS = 20_000
BIN_WIDTH = 5_000.0
CHECKPOINTS = (10, 20, 40, 100)
#: palette keys; the values are read from the active theme when a figure is built
STILL_GOING = "light_rule"
SAMPLE_LINE = "sample_line"
SAMPLE_LINE_ALPHA = 0.55  # the mock's sample-line gray at 55%
ZERO_LINE = "zero_line"
PERCENT_TICKS = [0, 25, 50, 75, 100]
PERCENT_TEXT = [f"{v}%" for v in PERCENT_TICKS]


# ── small pure helpers (tested) ───────────────────────────────────────────


def parse_money(text: Any) -> float | None:
    """A dollar amount typed by the owner: ``−$2,000``, ``-2000``, ``+$2,600``, ``2600``."""

    if text is None:
        return None
    cleaned = (str(text).strip().replace("−", "-").replace("–", "-").replace("$", "")
               .replace(",", "").replace(" ", ""))
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def new_seed(previous: int | None = None) -> int:
    """A fresh random seed for "Run again" (never the fixed one, never the last one)."""

    while True:
        seed = 10_000 + secrets.randbelow(90_000)
        if seed not in (previous, DEFAULT_SEED):
            return seed


def seed_note(seed: int) -> str | None:
    """What the screen says after "Run again" (None for the fixed draw)."""

    if seed == DEFAULT_SEED:
        return None
    return (f"Run again used a new random draw (seed {seed}). The fixed draw returns when "
            "you choose Back to the fixed draw.")


def trades_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}" if float(value).is_integer() else f"{value:.1f}"


def possessive(name: str) -> str:
    """``TakeProfitTrader's``, ``MyFundedFutures'``."""

    return f"{name}'" if name.endswith("s") else f"{name}'s"


def pct(share: float | None) -> str:
    return fmt.percent(share)


def explanation(count: int) -> str:
    # correction A5: resampling with replacement, not reordering
    return (f"Every chart here resamples this configuration's own {count:,} recorded funded "
            "trades with replacement — in blocks of 10 consecutive trades, or one trade at a "
            "time — thousands of times. A path can repeat some trades and leave out others, "
            "so its total differs from the recorded total. These charts describe the recorded "
            "trades under that sampling model; they are not a forecast.")


#: the sampling switch's help (correction A5)
METHOD_HELP = ("Keep streaks together draws blocks of 10 consecutive recorded trades with "
               "replacement, so clusters of losses stay together; Draw single trades draws "
               "one recorded trade at a time with replacement.")


def race_defaults(terms: dict[str, Any] | None) -> tuple[float | None, float | None]:
    """(lower, upper) fixed closed-profit boundaries in dollars from the firm's frozen terms.

    Lower = −(loss allowance); upper = retained cushion + minimum request. They are
    used as fixed closed-profit boundaries only, not as the firm's floor or payout
    (correction A2).
    """

    if not terms:
        return None, None
    allowance = terms.get("loss_allowance_cents")
    cushion = terms.get("retained_cushion_cents")
    minimum = terms.get("minimum_gross_request_cents")
    loss = -int(allowance) / 100 if allowance else None
    trigger = (int(cushion) + int(minimum)) / 100 if cushion is not None and minimum else None
    return loss, trigger


def spread_labels(values: Sequence[float], gap: float) -> list[float]:
    """Label heights for values at one edge, pushed apart so none are closer than ``gap``."""

    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = [0.0] * len(values)
    last = -math.inf
    for i in order:
        y = max(values[i], last + gap)
        placed[i] = y
        last = y
    # recentre so the group sits around the original values
    shift = (sum(placed) - sum(values)) / max(1, len(values))
    return [y - shift for y in placed]


def money_ticks(lo: float, hi: float, *, target: int = 5) -> tuple[list[float], list[str]]:
    """Round money ticks and their ``$20k`` / ``−$5k`` labels."""

    span = max(1.0, hi - lo)
    step = next((s for s in (500, 1_000, 2_000, 2_500, 5_000, 10_000, 20_000, 25_000, 50_000,
                              100_000, 200_000, 250_000, 500_000)
                 if span / s <= target), 1_000_000)
    first = math.ceil(lo / step) * step
    ticks = [float(v) for v in np.arange(first, hi + step * 0.001, step)]
    return ticks, [fmt.money_short(v) for v in ticks]


def end_bins(end_values: Sequence[float], width: float = BIN_WIDTH
             ) -> list[tuple[float, float]]:
    """(left edge, share of paths) of end values in ``width`` bins aligned on $0."""

    ends = np.asarray(end_values, dtype=float)
    if ends.size == 0:
        return []
    lo = math.floor(ends.min() / width) * width
    hi = math.ceil(ends.max() / width) * width
    if hi <= lo:
        hi = lo + width
    edges = np.arange(lo, hi + width * 0.5, width)
    counts, _ = np.histogram(ends, bins=edges)
    return [(float(edges[i]), float(counts[i] / ends.size)) for i in range(len(counts))]


def methods_reading(fans: dict[str, rs.EquityFan]) -> str:
    """One sentence comparing the two methods.

    Close = the bad case, typical, good case and bad-case drop of the two methods each
    differ by at most 10% of the typical end value (a scale that stays meaningful when
    a bad case sits near $0).
    """

    blocks, shuffle = fans.get("blocks"), fans.get("shuffle")
    if blocks is None or shuffle is None:
        return ""
    pairs = [(blocks.bad_end, shuffle.bad_end), (blocks.typical_end, shuffle.typical_end),
             (blocks.good_end, shuffle.good_end),
             (blocks.bad_case_drawdown, shuffle.bad_case_drawdown)]
    scale = max(abs(blocks.typical_end), abs(shuffle.typical_end), 1.0)
    close = all(abs(a - b) <= 0.10 * scale for a, b in pairs)
    if close:
        return ("The two methods land close together, so how losses cluster doesn't change the "
                "picture much here. When they split apart on a future configuration, trust "
                "\"keep streaks together.\"")
    return (f"The two methods split apart: keeping streaks together gives a bad case of "
            f"{fmt.money_whole(blocks.bad_end)} against {fmt.money_whole(shuffle.bad_end)} when "
            "single trades are drawn, so how losses cluster matters for this configuration. "
            "Trust \"keep streaks together.\"")


# correction A5 (supersedes F3's rule): a percentile is reported descriptively under its
# sampling model, with ties counting half; it is never classified as luck.


def end_caption(fan: rs.EquityFan, actual: float) -> str:
    """Where the recorded trading profit sits among the resampled paths' ends."""

    above = rs.share_below_ties_half(fan.end_values, actual)
    return (f"In trading profit, the recorded result ({fmt.money_whole(actual, signed=True)}) "
            f"is above {pct(above)} of the {fan.paths:,} resampled paths' ends (ties count "
            "half). Each path draws the recorded trades with replacement "
            f"({rs.sampling_words(fan.method, fan.trades)}), so path totals vary; a pure "
            "reordering of the same trades would always end at the recorded total. The "
            "percentile describes where the recorded result sits under this sampling model; "
            "it does not measure luck.")


def horizon_text(race: fr.FullRace) -> str:
    """``the study's 114 trade slots, January 13 – June 10, 2026`` (the race's own inputs)."""

    text = f"the study's {race.slots:,} trade slots"
    if race.horizon_first_day and race.horizon_last_day:
        text += f", {fmt.date_range(race.horizon_first_day, race.horizon_last_day)}"
    return text


def firm_standing(race: fr.FullRace, actual_net_cents: int | None) -> str | None:
    """Where the recorded net cash sits among the conditional paths (ties count half)."""

    if actual_net_cents is None:
        return None
    above = race.share_below(actual_net_cents / 100)
    if above is None:
        return None
    words = rs.sampling_words(race.method, race.slots)
    how = f"in {words}" if words.startswith("blocks") else words
    return (f"Under this conditional model, the recorded net cash "
            f"({fmt.money_cents(actual_net_cents)}) is above {pct(above)} of the "
            f"{race.paths:,} resampled paths' net cash (ties count half). Horizon: "
            f"{horizon_text(race)}; paths draw recorded trades with replacement {how}. The "
            "percentile describes where the recorded result sits; it does not measure luck.")


def streak_caption(dist: rs.StreakDistribution, actual: int) -> str:
    typical = int(round(dist.typical))
    tail = int(math.ceil(dist.bad))
    first = f"Typical worst run is {fmt.count(typical, 'loss', 'losses')}"
    if actual == typical:
        first += ", which is exactly what happened."
    else:
        first += f"; the actual order's worst run was {fmt.count(actual, 'loss', 'losses')}."
    return f"{first} Plan for {tail} or more one time in twenty (orange)."


# ── cached computations (rule 9: study, configuration, firm, settings, seed, paths) ──


def _values(store_root: str, result_id: str, configuration: str, firm_key: str) -> list[float]:
    from ifvg_lab_cache import trade_values

    return trade_values(store_root, result_id, configuration, firm_key)


@st.cache_data(show_spinner="Resampling the trades…", max_entries=32)
def fixed_race(store_root: str, result_id: str, configuration: str, firm_key: str,
               loss_limit: float, trigger: float, seed: int, paths: int) -> rs.PayoutRace:
    values = _values(store_root, result_id, configuration, firm_key)
    return rs.payout_race(values, loss_limit=loss_limit, trigger=trigger, paths=paths,
                          seed=seed, method="blocks")


@st.cache_data(show_spinner="Resampling the trades…", max_entries=32)
def fans(store_root: str, result_id: str, configuration: str, firm_key: str, seed: int,
         paths: int) -> dict[str, rs.EquityFan]:
    values = _values(store_root, result_id, configuration, firm_key)
    return {method: rs.equity_fan(values, method=method, paths=paths, seed=seed)
            for method in rs.METHODS}


@st.cache_data(show_spinner="Measuring falls from a previous high…", max_entries=32)
def growth(store_root: str, result_id: str, configuration: str, firm_key: str, limit: float,
           seed: int, paths: int) -> rs.DrawdownGrowth:
    values = _values(store_root, result_id, configuration, firm_key)
    return rs.drawdown_growth(values, paths=paths, seed=seed, method="blocks", limit=limit,
                              checkpoints=CHECKPOINTS)


class _PackageUnavailableError(Exception):
    """The verified strategy package (or its bars) can't be used now; ``reason`` says why.

    Raised inside the cached function and caught by :func:`_full_inputs`: Streamlit
    never caches an exception, so a package restored later is found on the next run.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@st.cache_resource(show_spinner=False, max_entries=16)
def _cached_full_inputs(store_root: str, result_id: str, configuration: str, firm_key: str):
    from ifvg_lab_cache import index_minutes
    from ifvg_lab_ui import funded_study

    study = funded_study(store_root, result_id)
    try:
        source = fr.open_source(study.plan)
    except Exception as error:  # an unverifiable package is never used
        raise _PackageUnavailableError(f"the study's strategy package failed its check "
                                  f"({type(error).__name__})") from error
    if source is None:
        raise _PackageUnavailableError("the study's verified strategy package is not on this "
                                  "computer")
    frame = index_minutes(store_root, result_id)
    if frame is None:
        raise _PackageUnavailableError("the study's stored one-minute E-mini bars can't be read "
                                  "right now")
    minutes = fr.MinuteIndex.from_frame(frame)
    try:
        rules, slots, shapes = fr.build_inputs(study, configuration, firm_key, source=source,
                                               minutes=minutes)
    except ValueError as error:  # a fact of the saved result: kept
        return None, str(error)
    validation = fr.check_original(study, configuration, firm_key, rules, slots, shapes)
    return (rules, slots, shapes, validation), None


def _full_inputs(store_root: str, result_id: str, configuration: str, firm_key: str):
    """(rules, slots, shapes, validation) or (None, reason) — read only, once per process.

    A package that is missing or unreadable is looked for again on the next run.
    """

    try:
        return _cached_full_inputs(store_root, result_id, configuration, firm_key)
    except _PackageUnavailableError as missing:
        return None, missing.reason


def _full_results() -> dict[tuple, fr.FullRace]:
    """Conditional resampling results kept for this process (``ifvg_lab_cache.firm_race_key``).

    Shared with the Summary's early-losses finding (``ifvg_lab_cache``).
    """

    from ifvg_lab_cache import firm_race_results

    return firm_race_results()


def ledger_key(ctx, rules: fr.FirmRules, slots: Sequence[fr.Slot], *, seed: int,
               paths: int, method: str = "blocks") -> tuple | None:
    """The cache key of this pair's conditional run (correction A3); None without terms.

    Built from the run's own slots and cutoff and the digest of the saved terms and
    processing clock the rules came from. The fixed diagnostic's typed boundaries
    are not an input: they never change the ledger run.
    """

    from ifvg_lab_cache import firm_race_key, saved_race_binding

    binding = saved_race_binding(ctx.study, ctx.configuration, ctx.firm_key)
    if binding is None:
        return None
    saved_slots, saved_cutoff, digest = binding
    if saved_slots != len(slots) or saved_cutoff != int(rules.cutoff_ns):
        return None  # the run's inputs don't match the saved result: never cached or quoted
    return firm_race_key(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key,
                         method=method, seed=seed, paths=paths, slots=saved_slots,
                         cutoff_ns=saved_cutoff, terms_digest=digest)


# ── charts ────────────────────────────────────────────────────────────────


def _go():
    import plotly.graph_objects as go

    return go


def race_figure(race: rs.PayoutRace):
    """The fixed closed-profit boundary diagnostic: which boundary each path crossed first."""

    go = _go()
    x = list(range(len(race.paid_by_trade)))
    paid = [v * 100 for v in race.paid_by_trade]
    died = [v * 100 for v in race.died_by_trade]
    going = [max(0.0, 100 - p - d) for p, d in zip(paid, died, strict=True)]
    c = palette()
    fig = go.Figure()
    # correction A2: boundary crossings of closed profit, not payouts or account losses
    for name, ys, color, word in (("Upper boundary first", paid, c["blue"],
                                   "upper boundary first"),
                                  ("Neither yet", going, c[STILL_GOING], "neither yet"),
                                  ("Lower boundary first", died, c["orange"],
                                   "lower boundary first")):
        fig.add_trace(go.Scatter(
            x=x, y=ys, name=name, mode="lines", stackgroup="race", fillcolor=color,
            line={"width": 0, "color": color},
            hovertemplate=f"After %{{x}} trades: %{{y:.0f}}% {word}<extra></extra>"))
    last = len(x) - 1
    for at in (10, 20):
        if at > last:
            continue
        fig.add_shape(type="line", x0=at, x1=at, y0=0, y1=100,
                      line={"color": c["ink"], "width": 1.5, "dash": "dash"})
        middle = (paid[at] + (100 - died[at])) / 2
        fig.add_annotation(
            x=at, y=min(85.0, max(15.0, middle)), xanchor="left", yanchor="middle", xshift=8,
            text=(f"<b>After {at} trades:</b><br>{paid[at]:.0f}% upper · "
                  f"{died[at]:.0f}% lower"),
            showarrow=False, align="left", bgcolor=c["panel"], bordercolor=c["panel"],
            borderpad=6, font={"family": FONT_SANS, "size": 13, "color": c["body"]})
    if paid[last] > 10:
        fig.add_annotation(x=last - 0.6, y=paid[last] / 2, xanchor="right", showarrow=False,
                           text="<b>Upper first</b>",
                           font={"family": FONT_SANS, "size": 15, "color": c["on_ink"]})
    if died[last] > 8:
        fig.add_annotation(x=last - 0.6, y=100 - died[last] / 2, xanchor="right",
                           showarrow=False, text="<b>Lower first</b>",
                           font={"family": FONT_SANS, "size": 15, "color": c["on_ink"]})
    early = min(3, last)
    if going[early] > 12:
        fig.add_annotation(x=early, y=(paid[early] + 100 - died[early]) / 2, xanchor="left",
                           showarrow=False, text="<b>Neither yet</b>",
                           font={"family": FONT_SANS, "size": 14, "color": c["body_2"]})
    style_chart(fig, height=380, x_title="Resampled trades",
                y_title="Share of resampled paths", money_axis=None)
    fig.update_xaxes(range=[0, last], dtick=5, showgrid=False)
    fig.update_yaxes(range=[0, 100], tickvals=PERCENT_TICKS, ticktext=PERCENT_TEXT,
                     gridcolor=rgba("panel", 0.35))
    fig.update_layout(plot_bgcolor=c[STILL_GOING], hovermode="x unified")
    return fig


def race_caption(race: rs.PayoutRace) -> str:
    upper = fmt.money_whole(race.trigger, signed=True)
    lower = fmt.money_whole(race.loss_limit)
    text = (f"Of {race.paths:,} resampled paths of up to {race.max_trades} trades (blocks of "
            f"{rs.BLOCK_SIZE} recorded trades drawn with replacement), {pct(race.paid_share)} "
            f"reached cumulative closed profit of {upper} before {lower} and "
            f"{pct(race.died_share)} fell to {lower} first")
    if race.still_going_share >= 0.005:
        text += f"; {pct(race.still_going_share)} reached neither"
    return text + (". This diagnostic counts closed trade results against fixed boundaries "
                   "only: it leaves out the firm's trailing and locking floor, losses inside "
                   "open trades, payout requests, processing and receipts, so it is not a "
                   "payout or account-failure model.")


def _x_ticks(count: int) -> list[int]:
    step = 20 if count > 60 else 10 if count > 20 else 5
    ticks = list(range(0, count, step))
    if ticks and count - ticks[-1] < step * 0.4 and len(ticks) > 1:
        ticks = ticks[:-1]
    return [*ticks, count]


def fan_figure(fan: rs.EquityFan, actual: Sequence[float]):
    go = _go()
    x = list(range(fan.trades + 1))
    p5, p25, p50, p75, p95 = (list(fan.bands[p]) for p in (5, 25, 50, 75, 95))
    c = palette()
    sample_color = rgba(SAMPLE_LINE, SAMPLE_LINE_ALPHA)
    fig = go.Figure()
    for upper, lower, color in ((p95, p5, c["blue_band"]), (p75, p25, c["blue_mid"])):
        edge = {"width": 0, "color": "rgba(0,0,0,0)"}  # band edges: never drawn
        fig.add_trace(go.Scatter(x=x, y=upper, mode="lines", line=edge,
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=lower, mode="lines", line=edge,
                                 fill="tonexty", fillcolor=color, hoverinfo="skip",
                                 showlegend=False))
    for sample in fan.samples:
        fig.add_trace(go.Scatter(x=x, y=list(sample), mode="lines",
                                 line={"color": sample_color, "width": 1}, hoverinfo="skip",
                                 showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=p50, mode="lines", line={"color": c["blue"], "width": 3}, showlegend=False,
        customdata=[fmt.money_whole(v) for v in p50],
        hovertemplate="Trade %{x}: typical %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(range(len(actual))), y=list(actual), mode="lines",
        line={"color": c["ink"], "width": 2.5, "dash": "dash"}, showlegend=False,
        customdata=[fmt.money_whole(v, signed=True) for v in actual],
        hovertemplate="Trade %{x}: what happened %{customdata}<extra></extra>"))
    fig.add_shape(type="line", x0=0, x1=fan.trades, y0=0, y1=0,
                  line={"color": c[ZERO_LINE], "width": 1.5})
    lo = min(min(p5), min(actual), min(min(s) for s in fan.samples) if fan.samples else 0, 0)
    hi = max(max(p95), max(actual), max(max(s) for s in fan.samples) if fan.samples else 0)
    pad = (hi - lo) * 0.06
    ends = [p95[-1], p75[-1], p50[-1], p25[-1], p5[-1]]
    words = ["good", "", "typical", "", "bad"]
    heights = spread_labels(ends, (hi - lo) * 0.06)
    for value, height, word in zip(ends, heights, words, strict=True):
        typical = word == "typical"
        fig.add_annotation(
            x=fan.trades, y=height, xanchor="left", xshift=10, showarrow=False,
            text=(f"<b>{fmt.money_whole(value)} {word}</b>" if typical
                  else f"{fmt.money_whole(value)} {word}".strip()),
            font={"family": FONT_MONO, "size": 13,
                  "color": c["blue_dark"] if typical else c["body_2"]})
    end = actual[-1]
    fig.add_annotation(
        x=fan.trades, y=lo, xanchor="right", yanchor="bottom", xshift=-6, yshift=6,
        showarrow=False, text=f"What happened: {fmt.money_whole(end, signed=True)}",
        bgcolor=c["ink"], bordercolor=c["ink"], borderpad=5,
        font={"family": FONT_SANS, "size": 13, "color": c["on_ink"]})
    style_chart(fig, height=440,
                x_title="Trade number · profit shown is trading profit across accounts, "
                        "before payouts", y_title="Trading profit so far", money_axis=None)
    ticks, labels = money_ticks(lo - pad, hi + pad)
    fig.update_yaxes(range=[lo - pad, hi + pad], tickvals=ticks, ticktext=labels)
    fig.update_xaxes(range=[0, fan.trades], tickvals=_x_ticks(fan.trades), showgrid=False)
    fig.update_layout(margin={"l": 8, "r": 150, "t": 12, "b": 8})
    return fig


def fan_legend() -> h.Markup:
    item = ('<span style="display:flex;align-items:center;gap:6px">'
            '<span style="{style}"></span>{label}</span>')
    parts = [
        item.format(style=f"width:18px;height:12px;background:{css_var('blue_band')}",
                    label="9 in 10 paths"),
        item.format(style=f"width:18px;height:12px;background:{css_var('blue_mid')}",
                    label="Middle half"),
        item.format(style=f"width:18px;height:3px;background:{css_var('blue')}",
                    label="Typical path"),
        item.format(style=f"width:18px;height:0;border-top:1px solid {css_var('sample_line')}",
                    label="A few sample paths"),
        item.format(style=f"width:18px;height:0;border-top:3px dashed {css_var('ink')}",
                    label="What actually happened"),
    ]
    return h.Markup('<div class="lab" style="display:flex;gap:18px;font-size:13px;'
                    f'color:{css_var("body_2")};flex-wrap:wrap">{"".join(parts)}</div>')


def fan_title(paths: int, trades: int) -> str:
    return f"{paths:,} resampled paths of {trades:,} trades"


def fan_caption(fan: rs.EquityFan) -> str:
    return (f"Half of the {fan.paths:,} resampled paths end between "
            f"{fmt.money_whole(fan.bands[25][-1])} and {fmt.money_whole(fan.bands[75][-1])} "
            f"after {fan.trades:,} trades; one in twenty ends below "
            f"{fmt.money_whole(fan.bad_end)}.")


def comparison_table(fan_by: dict[str, rs.EquityFan], actual_end: float,
                     actual_drawdown: float | None, trades: int) -> h.Markup:
    blocks, shuffle = fan_by["blocks"], fan_by["shuffle"]
    columns = [h.Column("what", f"After {trades:,} trades", width="34%"),
               h.Column("blocks", rs.METHODS["blocks"], "right"),
               h.Column("shuffle", rs.METHODS["shuffle"], "right"),
               h.Column("actual", "What happened", "right")]
    bold = h.Markup(f"<b>{h.esc(fmt.money_whole(actual_end))}</b>")
    rows = [
        h.Row({"what": "Bad case (1 in 20 did worse)", "blocks": fmt.money_whole(blocks.bad_end),
               "shuffle": fmt.money_whole(shuffle.bad_end), "actual": ""}),
        h.Row({"what": "Typical", "blocks": fmt.money_whole(blocks.typical_end),
               "shuffle": fmt.money_whole(shuffle.typical_end), "actual": bold}),
        h.Row({"what": "Good case (1 in 20 did better)",
               "blocks": fmt.money_whole(blocks.good_end),
               "shuffle": fmt.money_whole(shuffle.good_end), "actual": ""}),
        h.Row({"what": "Worst drop from a high, bad case",
               "blocks": fmt.money_whole(blocks.bad_case_drawdown),
               "shuffle": fmt.money_whole(shuffle.bad_case_drawdown),
               "actual": fmt.money_whole(actual_drawdown)}),
        h.Row({"what": "Finished below $0",
               "blocks": fmt.percent(blocks.below_zero_share, decimals=1),
               "shuffle": fmt.percent(shuffle.below_zero_share, decimals=1), "actual": ""}),
    ]
    table = h.table(columns, rows, plain=True, wrap=False)
    return h.Markup(f'<div class="lab lab-risk-table">{table}</div>')


def _marker(fig, x: float, color: str, dash: str | None) -> None:
    fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper",
                  line={"color": color, "width": 2, **({"dash": dash} if dash else {})})


def end_figure(fan: rs.EquityFan, actual: float):
    go = _go()
    bins = end_bins(fan.end_values)
    lefts = [b[0] for b in bins]
    shares = [b[1] * 100 for b in bins]
    c = palette()
    colors = [c["orange"] if left < 0 else c["blue_mid"] for left in lefts]
    fig = go.Figure(go.Bar(
        x=[left + BIN_WIDTH / 2 for left in lefts], y=shares, width=BIN_WIDTH * 0.84,
        marker={"color": colors, "line": {"width": 0}},
        customdata=[f"{fmt.money_whole(left)} to {fmt.money_whole(left + BIN_WIDTH)}"
                    for left in lefts],
        hovertemplate="%{customdata}: %{y:.1f}% of paths<extra></extra>"))
    marks = [(fan.bad_end, c["orange"], "dash", f"bad {fmt.money_short(fan.bad_end)}",
              c["orange_dark"], 1.13, "center", False),
             (fan.typical_end, c["blue"], None, f"typical {fmt.money_short(fan.typical_end)}",
              c["blue_dark"], 1.13, "left", True),
             (actual, c["ink"], "dash", f"actual {fmt.money_short(actual)}", c["ink"], 1.03,
              "right", False),
             (fan.good_end, c["blue"], "dash", f"good {fmt.money_short(fan.good_end)}",
              c["blue_dark"], 1.03, "center", False)]
    for x, color, dash, text, font_color, y, anchor, strong in marks:
        _marker(fig, x, color, dash)
        fig.add_annotation(x=x, y=y, yref="paper", xanchor=anchor, yanchor="bottom",
                           showarrow=False, text=f"<b>{text}</b>" if strong else text,
                           xshift={"left": 4, "right": -4}.get(anchor, 0),
                           font={"family": FONT_MONO, "size": 12, "color": font_color})
    style_chart(fig, height=320, x_title=f"Trading profit after {fan.trades:,} trades · share "
                "of paths", money_axis=None)
    lo = min(lefts) if lefts else 0
    hi = (max(lefts) + BIN_WIDTH) if lefts else BIN_WIDTH
    ticks, labels = money_ticks(lo, hi, target=6)
    fig.update_xaxes(tickvals=ticks, ticktext=labels, showgrid=False, range=[lo, hi])
    fig.update_yaxes(showticklabels=False, showgrid=False, rangemode="tozero")
    fig.update_layout(margin={"l": 8, "r": 16, "t": 56, "b": 8}, bargap=0.1)
    return fig


def streak_figure(dist: rs.StreakDistribution):
    go = _go()
    lengths = list(range(min(dist.shares), max(dist.shares) + 1))
    shares = [dist.shares.get(n, 0.0) * 100 for n in lengths]
    typical = int(round(dist.typical))
    tail = int(math.ceil(dist.bad))
    c = palette()
    colors = [c["blue"] if n == typical else c["orange"] if n >= tail else c["blue_mid"]
              for n in lengths]
    labels = [(f"<b>{s:.0f}%</b>" if n == typical else f"{s:.0f}%") if s >= 0.5 else ""
              for n, s in zip(lengths, shares, strict=True)]
    fig = go.Figure(go.Bar(
        x=lengths, y=shares, marker={"color": colors, "line": {"width": 0}}, text=labels,
        textposition="outside", cliponaxis=False,
        textfont={"family": FONT_MONO, "size": 11, "color": c["body_2"]},
        hovertemplate="Longest run of %{x} losses: %{y:.1f}% of paths<extra></extra>"))
    style_chart(fig, height=320, x_title="Losses in a row · share of paths", money_axis=None)
    fig.update_xaxes(tickvals=lengths, showgrid=False)
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     range=[0, max(shares) * 1.18 if shares else 1])
    fig.update_layout(margin={"l": 8, "r": 16, "t": 24, "b": 8}, bargap=0.22)
    return fig


def growth_figures(g: rs.DrawdownGrowth):
    """Sampled closed-profit drawdown from a previous high (correction A2)."""

    go = _go()
    x = list(range(len(g.typical)))
    top = max(max(g.worse_1_in_20), g.limit) * 1.15
    pal = palette()
    fall = fmt.money_whole(g.limit)
    left = go.Figure()
    left.add_shape(type="rect", x0=0, x1=x[-1], y0=g.limit, y1=top, layer="below",
                   fillcolor=pal["orange_light"], opacity=0.6, line={"width": 0})
    left.add_shape(type="line", x0=0, x1=x[-1], y0=g.limit, y1=g.limit,
                   line={"color": pal["orange"], "width": 2})
    left.add_annotation(x=1, y=g.limit, xanchor="left", yanchor="bottom", showarrow=False,
                        text=f"<b>{fall} fall</b>",
                        font={"family": FONT_SANS, "size": 12, "color": pal["orange_dark"]})
    lines = ((g.worse_1_in_20, "1 in 20 worse", pal["orange"], 2, "dash", pal["orange_dark"]),
             (g.worse_1_in_4, "1 in 4 worse", pal["blue_line"], 2, None, pal["body_2"]),
             (g.typical, "Typical", pal["blue"], 3, None, pal["blue_dark"]))
    ends = [series[-1] for series, *_ in lines]
    heights = spread_labels(ends, top * 0.07)
    for (series, name, color, width, dash, text_color), height in zip(lines, heights,
                                                                       strict=True):
        left.add_trace(go.Scatter(
            x=x, y=list(series), mode="lines", name=name, showlegend=False,
            line={"color": color, "width": width, **({"dash": dash} if dash else {})},
            customdata=[fmt.money_whole(v) for v in series],
            hovertemplate=f"After %{{x}} trades: {name.lower()} %{{customdata}}<extra></extra>"))
        strong = name == "Typical"
        text = fmt.money_whole(series[-1])
        left.add_annotation(x=x[-1], y=height, xanchor="left", xshift=8, showarrow=False,
                            text=f"<b>{text}</b>" if strong else text,
                            font={"family": FONT_MONO, "size": 12, "color": text_color})
    style_chart(left, height=330, x_title="Resampled trades",
                y_title="Largest fall so far", money_axis=None)
    ticks, labels = money_ticks(0, top)
    left.update_yaxes(range=[0, top], tickvals=ticks, ticktext=labels)
    left.update_xaxes(range=[0, x[-1]], showgrid=False)
    left.update_layout(margin={"l": 8, "r": 70, "t": 12, "b": 8})

    right = go.Figure()
    touched = [v * 100 for v in g.touched_share]
    right.add_trace(go.Scatter(
        x=x, y=touched, mode="lines", fill="tozeroy", fillcolor=pal["orange_light"],
        line={"color": pal["orange"], "width": 3}, showlegend=False,
        hovertemplate=(f"By trade %{{x}}: %{{y:.0f}}% of paths have fallen {fall} from a "
                       "previous high<extra></extra>")))
    points = [c for c in CHECKPOINTS if c < len(touched)]
    right.add_trace(go.Scatter(
        x=points, y=[touched[c] for c in points], mode="markers", showlegend=False,
        marker={"color": pal["ink"], "size": 10}, hoverinfo="skip"))
    for index, c in enumerate(points):
        text = f"{touched[c]:.0f}% by trade {c}" if index == 0 else f"{touched[c]:.0f}% by {c}"
        last = c == points[-1] and c == x[-1]
        right.add_annotation(x=c, y=touched[c], showarrow=False,
                             xanchor="right" if last else "left",
                             yanchor="top", xshift=-8 if last else 8, yshift=-6,
                             text=text, font={"family": FONT_MONO, "size": 12,
                                              "color": pal["ink"]})
    style_chart(right, height=330, x_title="Resampled trades",
                y_title="Share of sampled paths", money_axis=None)
    right.update_yaxes(range=[0, 100], tickvals=PERCENT_TICKS, ticktext=PERCENT_TEXT)
    right.update_xaxes(range=[0, x[-1]], showgrid=False)
    right.update_layout(margin={"l": 8, "r": 16, "t": 12, "b": 8})
    return left, right


def growth_legend() -> h.Markup:
    return h.Markup(
        f'<div class="lab" style="display:flex;gap:16px;font-size:13px;color:{css_var("body_2")}">'
        '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
        f'height:3px;background:{css_var("blue")}"></span>Typical</span>'
        '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
        f'height:2px;background:{css_var("blue_line")}"></span>1 in 4 worse</span>'
        '<span style="display:flex;align-items:center;gap:6px"><span style="width:18px;'
        f'height:0;border-top:2px dashed {css_var("orange")}"></span>1 in 20 worse</span></div>')


def growth_table(g: rs.DrawdownGrowth) -> h.Markup:
    points = [c for c in CHECKPOINTS if c < len(g.typical)]
    columns = [h.Column("what", "After…", width="210px")]
    columns += [h.Column(f"c{c}", fmt.count(c, "trade"), "right") for c in points]
    worst = {"what": "Largest fall, typical · 1 in 20"}
    profit = {"what": "Closed profit, 5th · median · 95th percentile"}
    touched = {"what": h.Markup(f'<b style="color:{css_var("orange_dark")}">Fell '
                                f"{h.esc(fmt.money_whole(g.limit))} from a previous high</b>")}
    for c in points:
        worst[f"c{c}"] = (f"{fmt.money_whole(g.typical[c])} · "
                          f"{fmt.money_whole(g.worse_1_in_20[c])}")
        bad, typical, good = g.profit_at.get(c, (None, None, None))
        # one money format across the row: thousands with one decimal (fix F10)
        profit[f"c{c}"] = (f"{fmt.money_k(bad)} · {fmt.money_k(typical)} · "
                           f"{fmt.money_k(good)}")
        touched[f"c{c}"] = h.Markup(f'<b style="color:{css_var("orange_dark")}">'
                                    f"{h.esc(pct(g.touched_share[c]))}</b>")
    table = h.table(columns, [h.Row(worst), h.Row(profit), h.Row(touched)], plain=True,
                    wrap=False)
    return h.Markup(f'<div class="lab lab-risk-table compact">{table}</div>')


def lock_text(terms: dict[str, Any] | None) -> str | None:
    """``at $0`` / ``at +$100`` from the saved floor lock (never a literal)."""

    if not terms or terms.get("floor_lock_cents") is None:
        return None
    return f"at {fmt.money_whole(int(terms['floor_lock_cents']) / 100, signed=True)}"


def growth_sentence(g: rs.DrawdownGrowth, firm_name: str, lock: str | None) -> str:
    """What the sampled closed-profit drawdown shows, and what it is not (correction A2).

    It is not the share of funded accounts that fail, and it gives no reason to
    believe early withdrawals prevent these falls.
    """

    last = min(CHECKPOINTS[-1], len(g.touched_share) - 1)
    fall = fmt.money_whole(g.limit)
    floor = (f"{possessive(firm_name)} floor can lock ({lock})" if lock
             else f"{possessive(firm_name)} floor can lock")
    return (f"By {last} resampled trades, {pct(g.touched_share[last])} of paths have had "
            f"their cumulative closed profit fall at least {fall} below a previous high. This "
            "counts closed trade results with no withdrawals and no account floor, so it is not "
            f"the share of funded accounts that fail: {floor}, so a {fall} fall from a higher "
            "peak need not end an account, and a loss inside an open trade can end an account "
            "even when the closed result recovers.")


# ── HTML pieces ───────────────────────────────────────────────────────────


def stat_block(label: str, value: str, *, border: str, color: str, size: int = 30,
               sub: str | None = None) -> h.Markup:
    sub_html = (f'<div style="font-size:12px;color:{css_var("muted")}">{h.esc(sub)}</div>'
                if sub else "")
    return h.Markup(
        f'<div class="lab" style="padding:10px 16px;border-left:4px solid {border};'
        'display:flex;flex-direction:column;gap:2px">'
        f'<div style="font-size:13px;color:{css_var("muted")}">{h.esc(label)}</div>'
        f'<div class="lab-mono" style="font-size:{size}px;font-weight:500;color:{color}">'
        f"{h.esc(value)}</div>{sub_html}</div>")


def soft_box(label: str, value: Any, *, detail: str | None = None,
             placeholder: bool = False) -> h.Markup:
    style = (f"font-size:16px;color:{css_var('muted')}" if placeholder
             else f"font-size:22px;color:{css_var('ink')};font-weight:500")
    more = (f'<div style="font-size:13px;line-height:1.45;color:{css_var("body_2")}">'
            f"{h.esc(detail)}</div>" if detail else "")
    return h.Markup(
        f'<div class="lab" style="background:{css_var("soft_panel")};border-radius:8px;'
        'padding:12px 14px;display:flex;flex-direction:column;gap:4px">'
        f'<div style="font-size:13px;color:{css_var("muted")}">{h.esc(label)}</div>'
        f'<div class="lab-mono" style="{style}">{h.esc(value)}</div>{more}</div>')


def title_row(title: str, right: str | None = None, *, size: int = 24) -> h.Markup:
    right_html = (f'<div style="font-size:13px;color:{css_var("muted")}">{h.esc(right)}</div>'
                  if right else "")
    return h.Markup(
        '<div class="lab" style="display:flex;justify-content:space-between;'
        'align-items:baseline;gap:16px;flex-wrap:wrap">'
        f'<div class="lab-card-title" style="font-size:{size}px">{h.esc(title)}</div>'
        f"{right_html}</div>")


def line(text: str, *, color: str | None = None) -> h.Markup:
    color = color or css_var("body")
    return h.Markup(f'<div class="lab" style="font-size:14px;line-height:1.5;color:{color}">'
                    f"{h.esc(text)}</div>")


# ── conditional resampling with the firm's ledger rules (corrections A3, A4) ──


PAYOUTS_LABEL = "Average payouts among accounts that failed within the tested horizon"
CASH_LABEL = "Pooled net cash per purchased account"
#: the comparison table's rows, in order (correction A4)
COMPARISON_ROWS = ("Upper endpoint first", "Lower endpoint first",
                   "Neither by the end of the horizon", "Horizon",
                   "Typical trades to each endpoint", "Typical days from purchase",
                   "Net cash over the study's dates, 5th · median · 95th percentile")


def ledger_heading(firm: str) -> str:
    return f"Conditional resampling of recorded trades with {possessive(firm)} ledger rules"


def limitations_line() -> str:
    """Shown before and after a run (correction A3): what the model does not represent."""

    return ("Conditional on the recorded trades, not an exact fresh-account model at either "
            "firm: each path redraws this configuration's recorded funded trades (already "
            "shaped by the historical accounts' entry selection and skipped opportunities) into "
            "the study's fixed trade slots; trades cut short when an account was lost keep their "
            "shortened results; and each trade is compressed to its entry, lowest, highest and "
            "exit points, so a reversal between them is not represented.")


def run_label(firm: str) -> str:
    return f"Run conditional resampling with {possessive(firm)} rules"


def run_help(firm: str) -> str:
    return (f"Put resampled recorded trades through the funded account ledger with "
            f"{possessive(firm)} saved floor, payout, processing and replacement rules. It takes "
            "several seconds and is kept for this draw.")


def not_run_text(firm: str) -> str:
    return f"Not run yet · choose {run_label(firm)}"


def beyond_chart(slots: int) -> str:
    """What the ledger run adds past the chart's draw ("" when every trade is shared)."""

    shared = fr.shared_trades(slots)
    if slots <= shared:
        return ""
    return (f" (their first {shared:,} trades; the other {slots - shared:,} trade slots "
            "continue with a further fixed draw)")


def dollars(cents: int) -> str:
    """``$102`` for whole dollars, ``$0.51`` otherwise (exact cents)."""

    return fmt.money_whole(cents / 100) if cents % 100 == 0 else fmt.money_cents(cents)


def cutoff_day(race: fr.FullRace) -> str:
    return fmt.date_long(race.horizon_last_day) if race.horizon_last_day else "study's"


def payouts_value(race: fr.FullRace) -> str:
    if race.payouts_before_death is None:
        return "No account failed"
    return f"{race.payouts_before_death:.2f} payouts"


def payouts_detail(race: fr.FullRace) -> str:
    """Numerator, denominator, open accounts and unresolved requests (correction A4)."""

    still_open = race.open_accounts
    pending = race.unresolved_requests
    processing = ("no payout request was still processing" if not pending else
                  f"{fmt.count(pending, 'payout request')} totalling "
                  f"{fmt.money_cents(race.unresolved_trader_cents)} after the split "
                  f"{'was' if pending == 1 else 'were'} still processing — not counted as "
                  "received")
    return (f"{fmt.count(race.payouts_by_failed, 'payout')} received by the "
            f"{fmt.count(race.failed_accounts, 'account')} that failed before the "
            f"{cutoff_day(race)} cutoff, across {fmt.count(race.paths, 'resampled path')}. "
            "Not a lifetime expectation. At the cutoff "
            f"{fmt.count(still_open, 'account')} {'was' if still_open == 1 else 'were'} still "
            f"open (they had received {fmt.count(race.payouts_by_open, 'payout')}), and "
            f"{processing}.")


def cash_detail(race: fr.FullRace, rules: fr.FirmRules) -> str:
    """Which money is received and which costs are deducted (correction A4)."""

    share = rules.profile.trader_share_pct
    price = dollars(rules.profile.acquisition_cost_cents)
    bought = f"{race.accounts_bought:,}"
    return (f"All paths' payouts received after the {share}% split "
            f"({fmt.money_cents(race.received_cents_total)}) minus all account purchases "
            f"({fmt.money_cents(race.costs_cents_total)}: {bought} accounts at {price}), divided "
            f"by all {bought} accounts purchased. Requested-but-unreceived payouts and money "
            "left in open accounts count as $0 received. This pools every path; it is not an "
            "average of each path's own ratio.")


#: what :func:`firm_race.check_original` compares, in the words the Risk tab uses
_CHECKED_NAMES = {"net_cash_cents": "net cash", "received_cents": "payouts received",
                  "payouts": "payout count", "accounts": "accounts bought",
                  "costs_cents": "account costs"}
_NOT_CHECKED = ("It doesn't compare fill times, prices or quantities, which account took each "
                "trade, payout or failure times, or which setup each trade came from.")


def _checked_value(key: str, value: int | None) -> str:
    return fmt.money_cents(value) if key.endswith("_cents") else f"{int(value or 0):,}"


def validation_text(v: fr.Validation) -> str:
    """The historical-order check, naming exactly what it compares (review follow-up).

    Every figure is the saved value the check compared (``v.saved``).
    """

    if v.exact:
        figures = [f"{name} ({_checked_value(key, v.saved[key])})"
                   for key, name in _CHECKED_NAMES.items()]
        return (f"Checked first: the saved order, replayed through the same simulator, matches "
                f"the saved result's {', '.join(figures[:-1])} and {figures[-1]} exactly, and "
                f"gives the same number of trades ({v.trades:,}), each with the same net result "
                f"and account-loss flag in the saved order. {_NOT_CHECKED} This historical-order "
                "check confirms the adapter reproduces the saved order's money and account "
                "losses; it doesn't show that other sampled orders, or a fresh account running "
                "the full strategy, would be modelled exactly.")
    parts = [f"{_CHECKED_NAMES.get(key, key)} {_checked_value(key, v.saved[key])} saved vs "
             f"{_checked_value(key, v.replayed[key])} replayed"
             for key in v.differences]
    if v.replayed_trades is not None and v.replayed_trades != v.trades:
        trades = (f"the replay produced {v.replayed_trades:,} trades against {v.trades:,} "
                  "saved, so the trades were not compared one by one")
    else:
        trades = (f"{v.trades_matching:,} of {v.trades:,} trades have the same net result and "
                  "account-loss flag")
    return ("The saved order, replayed through the simulator, does not reproduce the saved "
            f"result ({'; '.join(parts) or 'per-trade results differ'}; {trades}).")


def approximation_text(rules: fr.FirmRules) -> str:
    """Why the compressed trade path is conditional at either firm (correction A3)."""

    firm = rules.firm_name
    if rules.profile.threshold_update == "intraday_peak_equity":
        lock = fmt.money_whole(rules.profile.floor_lock_cents / 100, signed=True)
        return (f"For {firm}, whose floor trails the highest equity inside a trade, the "
                "compressed path can miss or create an account loss: a trade marked $0 → "
                "−$1,000 → +$2,500 → −$100 → +$4,000 → +$3,000 fails once the floor has locked "
                f"at {lock}, and one marked $0 → −$1,000 → +$2,500 → +$2,000 → +$4,000 → "
                "+$3,000 survives, yet both compress to the same four points ($0, −$1,000, "
                "+$4,000, +$3,000).")
    return (f"For {firm}, whose floor moves only at the session close, a trade's stored lowest "
            "point is enough to decide a loss inside that trade; the recorded-trade selection, "
            "fixed slots and shortened results still apply, so these results are conditional "
            "too, not exact.")


def clock_words(processing: Any) -> str:
    """``two-business-day`` / ``48-hour`` from the saved processing clock."""

    basis = str(getattr(processing, "basis", "") or "")
    return {"two_business_days": "two-business-day",
            "elapsed_48_hours": "48-hour"}.get(basis, basis.replace("_", " ") or "saved")


def _days(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def full_comparison(race: fr.FullRace, fixed: rs.PayoutRace | None,
                    actual_net_cents: int | None, rules: fr.FirmRules) -> h.Markup:
    """The fixed diagnostic beside the conditional ledger model (corrections A2–A5).

    Different endpoints and horizons, stated in the rows and the footer.
    """

    firm = rules.firm_name
    terms = rules.profile
    columns = [h.Column("what", "", width="34%"),
               h.Column("fixed", "Fixed boundaries (diagnostic)", "right"),
               h.Column("full", f"{possessive(firm)} ledger rules (conditional)", "right")]
    if fixed is None:
        fixed_cells = ("—",) * 5
    else:
        upper = fmt.money_whole(fixed.trigger, signed=True)
        lower = fmt.money_whole(fixed.loss_limit)
        fixed_cells = (f"{pct(fixed.paid_share)} reached {upper}",
                       f"{pct(fixed.died_share)} fell to {lower}",
                       pct(fixed.still_going_share),
                       f"Up to {fixed.max_trades:,} resampled trades per path",
                       f"{trades_text(fixed.typical_to_payout)} to the upper · "
                       f"{trades_text(fixed.typical_to_limit)} to the lower")
    horizon = horizon_text(race)
    rows = [
        h.Row({"what": COMPARISON_ROWS[0], "fixed": fixed_cells[0],
               "full": f"{pct(race.paid_share)} received a first payout"}),
        h.Row({"what": COMPARISON_ROWS[1], "fixed": fixed_cells[1],
               "full": f"{pct(race.died_share)} failed before any payout was received"}),
        h.Row({"what": COMPARISON_ROWS[2], "fixed": fixed_cells[2],
               "full": pct(race.still_going_share)}),
        h.Row({"what": COMPARISON_ROWS[3], "fixed": fixed_cells[3],
               "full": horizon[0].upper() + horizon[1:]}),
        h.Row({"what": COMPARISON_ROWS[4], "fixed": fixed_cells[4],
               "full": f"{trades_text(race.median_trades_to_eligibility)} to eligibility · "
                       f"{trades_text(race.median_trades_to_request)} to the request · "
                       f"{trades_text(race.median_trades_to_receipt)} to receipt · "
                       f"{trades_text(race.median_trades_to_failure)} to failure before a payout"}),
        h.Row({"what": COMPARISON_ROWS[5], "fixed": h.placeholder("Not measured"),
               "full": f"{_days(race.median_days_to_eligibility)} to eligibility · "
                       f"{_days(race.median_days_to_request)} to the request · "
                       f"{_days(race.median_days_to_receipt)} to receipt · "
                       f"{_days(race.median_days_to_failure)} to failure before a payout"}),
        h.Row({"what": h.Markup(f"{h.esc(COMPARISON_ROWS[6])}"
                                f'<div class="lab-cell-sub">Recorded: '
                                f"{h.esc(fmt.money_cents(actual_net_cents))}</div>"),
               "fixed": h.placeholder("Not modelled"),
               "full": f"{fmt.money_short(race.net_cash_bad)} · "
                       f"{fmt.money_short(race.net_cash_typical)} · "
                       f"{fmt.money_short(race.net_cash_good)}"}),
    ]
    paths_note = (f"Fixed boundaries: {fixed.paths:,} paths. " if fixed is not None else "")
    seconds = int(round(race.seconds))
    foot = (f"{paths_note}Ledger rules: the first {race.paths:,} of the same resampled "
            f"draws{beyond_chart(race.slots)}, seed {race.seed}, "
            f"{fmt.count(seconds, 'second')}. "
            "The two use different endpoints and horizons, so their shares are not directly "
            "comparable. Eligibility: realized balance at least the "
            f"{dollars(terms.retained_cushion_cents)} cushion plus the "
            f"{dollars(terms.minimum_gross_request_cents)} minimum right after a closed trade, "
            "with the account alive and flat; request: at that trading day's end; receipt: "
            f"after the saved {clock_words(rules.processing)} processing clock. No "
            "requested-but-unreceived amount counts as cash received.")
    table = h.table(columns, rows, plain=True, wrap=False)
    standing = firm_standing(race, actual_net_cents)
    standing_html = (f'<div class="lab" style="font-size:14px;line-height:1.5;'
                     f'color:{css_var("body")}">{h.esc(standing)}</div>' if standing else "")
    return h.Markup(f'<div class="lab lab-risk-table">{table}</div>{standing_html}'
                    f'<div class="lab-line">{h.esc(foot)}</div>')


def full_notes(race: fr.FullRace, rules: fr.FirmRules) -> list[str]:
    terms = rules.profile
    return [
        (f"Each resampled path keeps the study's own trade slots — every trade's entry, exit "
         f"and trading day — and puts a resampled recorded trade's result in each. "
         f"{possessive(rules.firm_name)}"
         f" rules then apply as in the study: the {dollars(terms.loss_allowance_cents)} "
         "loss allowance and its floor, the payout request at the end of the day once "
         f"{dollars(terms.retained_cushion_cents + terms.minimum_gross_request_cents)} "
         "is reached, entries refused while finished for the day or while a payout is "
         "processing, the processing clock, and a replacement bought at "
         f"{dollars(terms.acquisition_cost_cents)} when an account is lost. The boundaries "
         "typed above don't change this run."),
        ("Received a first payout = the first account's first payout was received before it "
         "failed; failed before any payout = it failed first. Trades to an endpoint count that "
         "account's trades closed by then. The eligibility, request and receipt figures are "
         "medians over the paths whose first account reached that event by the cutoff; the "
         "failure figure is the median over the paths whose first account failed before any "
         "payout was received."),
        ("A trade that ended because an account was lost keeps its recorded, shortened result, "
         "as in every other chart here."),
        approximation_text(rules),
    ]


# ── the tab ───────────────────────────────────────────────────────────────


def _pair(ctx) -> str:
    return f"{ctx.configuration}|{ctx.firm_key}"


def current_seed(ctx) -> int:
    return int((ctx.context.get("risk_seeds") or {}).get(_pair(ctx), DEFAULT_SEED))


def current_paths(ctx) -> int:
    paths = ctx.context.get("risk_paths", DEFAULT_PATHS)
    return paths if paths in PATH_CHOICES else DEFAULT_PATHS


def header_right(st_module, ctx) -> None:
    """Paths picker and "Run again" (mock 05), in place of the firm switch."""

    from ifvg_lab_ui import show

    label_col, select_col, button_col = st_module.columns(
        [0.5, 1.05, 1.15], vertical_alignment="center", gap="small")
    with label_col:
        show(h.Markup(f'<div class="lab" style="font-size:13px;color:{css_var("muted")};'
                      'text-align:right">Paths</div>'), st_module)
    with select_col:
        paths = st_module.selectbox(
            "Paths", PATH_CHOICES, index=PATH_CHOICES.index(current_paths(ctx)),
            key=f"{PREFIX}risk_paths", format_func=lambda v: f"{v:,}",
            label_visibility="collapsed",
            help="How many resampled paths every chart on this tab draws. More paths give "
                 "steadier figures; the default is 20,000.")
    ctx.context["risk_paths"] = paths
    with button_col:
        if st_module.button("Run again", key=f"{PREFIX}risk_run_again", type="primary",
                            width="stretch",
                            help="Draw a new random set of resampled paths with a new seed. The "
                                 "screen says which seed it used, and Back to the fixed draw "
                                 "returns to the default."):
            seeds = ctx.context.setdefault("risk_seeds", {})
            seeds[_pair(ctx)] = new_seed(seeds.get(_pair(ctx)))


def _seed_line(st_module, ctx, seed: int, paths: int) -> None:
    from ifvg_lab_ui import show

    note = seed_note(seed)
    if note is None:
        show(h.Markup(f'<div class="lab-line">Fixed draw: seed {DEFAULT_SEED}, {paths:,} paths '
                      "per chart. The same draw returns on every reload.</div>"), st_module)
        return
    text_col, button_col = st_module.columns([4, 1], vertical_alignment="center")
    with text_col:
        show(h.note(f"{note} {paths:,} paths per chart.", "blue"), st_module)
    with button_col:
        if st_module.button("Back to the fixed draw", key=f"{PREFIX}risk_fixed_draw",
                            width="stretch",
                            help=f"Return to the fixed default seed ({DEFAULT_SEED})."):
            (ctx.context.get("risk_seeds") or {}).pop(_pair(ctx), None)
            st_module.rerun()


#: the fixed-boundary card's title and right-hand text (correction A2)
RACE_TITLE = "Fixed closed-profit boundaries: which is crossed first?"
RACE_RIGHT = "Diagnostic · closed trade results only · not the firm's account rules"


def lower_help(firm: str, terms: dict[str, Any] | None) -> str:
    text = ("A resampled path stops when its cumulative closed profit falls to this amount.")
    allowance = (terms or {}).get("loss_allowance_cents")
    if allowance:
        text += (f" The default is {possessive(firm)} saved "
                 f"{fmt.money_whole(int(allowance) / 100)} loss allowance, used here as a fixed "
                 "closed-profit boundary, not as the firm's floor.")
    return text


def upper_help(firm: str, terms: dict[str, Any] | None) -> str:
    text = "A resampled path stops when its cumulative closed profit reaches this amount."
    cushion = (terms or {}).get("retained_cushion_cents")
    minimum = (terms or {}).get("minimum_gross_request_cents")
    if cushion is not None and minimum:
        text += (f" The default is {possessive(firm)} saved "
                 f"{fmt.money_whole(int(cushion) / 100)} cushion plus its "
                 f"{fmt.money_whole(int(minimum) / 100)} minimum request; reaching it here is "
                 "not a payout.")
    else:
        text += " Reaching it here is not a payout."
    return text


def _race_card(st_module, ctx, values: list[float], seed: int, paths: int) -> rs.PayoutRace | None:
    from ifvg_lab_ui import plot, show

    terms = fr.firm_terms(ctx.study, ctx.firm_key)
    loss_default, trigger_default = race_defaults(terms)
    slug = _pair(ctx).replace("|", "_")
    race = None
    with st_module.container(key="ifvg_lab_card_risk_race"):
        show(title_row(RACE_TITLE, RACE_RIGHT), st_module)
        cols = st_module.columns([1, 1, 1.15, 1.15, 1.15], vertical_alignment="bottom")
        problems: list[str] = []
        with cols[0]:
            loss_text = st_module.text_input(
                "Lower boundary", value=fmt.money_whole(loss_default) if loss_default else "",
                key=f"{PREFIX}risk_loss_{slug}", help=lower_help(ctx.firm, terms))
        with cols[1]:
            trigger_text = st_module.text_input(
                "Upper boundary",
                value=fmt.money_whole(trigger_default, signed=True) if trigger_default else "",
                key=f"{PREFIX}risk_trigger_{slug}", help=upper_help(ctx.firm, terms))
        loss = parse_money(loss_text)
        trigger = parse_money(trigger_text)
        if loss is None or loss == 0:
            problems.append("Enter the lower boundary as a dollar amount such as −$2,000.")
            loss = None
        else:
            loss = -abs(loss)
        if trigger is None or trigger <= 0:
            problems.append("Enter the upper boundary as a dollar amount above $0, such as "
                            "+$2,600.")
            trigger = None
        if loss is not None and trigger is not None:
            race = fixed_race(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key,
                              float(loss), float(trigger), seed, paths)
        upper = (fmt.money_whole(trigger, signed=True) if trigger is not None
                 else "the upper boundary")
        lower = fmt.money_whole(loss) if loss is not None else "the lower boundary"
        with cols[2]:
            show(stat_block(f"Reached {upper} first",
                            pct(race.paid_share) if race else "—", border=css_var("blue"),
                            color=css_var("blue_dark")), st_module)
        with cols[3]:
            show(stat_block(f"Fell to {lower} first",
                            pct(race.died_share) if race else "—", border=css_var("orange"),
                            color=css_var("orange_dark")), st_module)
        with cols[4]:
            show(stat_block("Typical trades to cross",
                            (f"{trades_text(race.typical_to_payout)} · "
                             f"{trades_text(race.typical_to_limit)}") if race else "—",
                            border=css_var("control_border"), color=css_var("ink"), size=22,
                            sub="to the upper · to the lower"), st_module)
        if problems:
            show(h.note(" ".join(problems), "orange"), st_module)
        if terms is None:
            show(h.note(f"The saved result doesn't record {possessive(ctx.firm)} loss "
                        "allowance and payout cushion, so there are no default boundaries; type "
                        "both values."),
                 st_module)
        if race is not None:
            plot(race_figure(race), key=f"risk_race_{slug}", st_module=st_module)
            show(h.Markup(f'<div class="lab-line">{h.esc(race_caption(race))}</div>'),
                 st_module)
        _full_version(st_module, ctx, race, seed)
    return race


def _full_version(st_module, ctx, fixed: rs.PayoutRace | None, seed: int) -> None:
    """Conditional resampling with the firm's ledger rules (corrections A3, A4)."""

    from ifvg_lab_ui import show

    firm = ctx.firm
    # always visible, before and after a run (correction A3)
    show(h.Markup('<div class="lab" style="font-size:18px;font-weight:600;margin-top:10px">'
                  f"{h.esc(ledger_heading(firm))}</div>"), st_module)
    show(h.Markup(f'<div class="lab-line">{h.esc(limitations_line())}</div>'), st_module)
    boxes = st_module.empty()
    inputs, reason = _full_inputs(ctx.store_root, ctx.result_id, ctx.configuration,
                                  ctx.firm_key)
    slug = _pair(ctx).replace("|", "_")
    if inputs is None:
        with boxes.container():
            show(h.grid([soft_box(PAYOUTS_LABEL, "Not available here", placeholder=True),
                         soft_box(CASH_LABEL, "Not available here", placeholder=True)], 2),
                 st_module)
        show(h.note(f"Conditional resampling with {possessive(firm)} rules can't run here: "
                    f"{reason}. It needs the study's trading calendar from that package; the "
                    "fixed-boundary diagnostic above is unaffected."), st_module)
        return
    rules, slots, shapes, validation = inputs
    run_col, paths_col, note_col = st_module.columns([1.35, 0.75, 2.3],
                                                     vertical_alignment="center")
    with paths_col:
        full_paths = st_module.selectbox(
            "Paths for this run", fr.FULL_PATH_CHOICES,
            index=fr.FULL_PATH_CHOICES.index(ctx.context.get("risk_full_paths",
                                                             fr.DEFAULT_FULL_PATHS)),
            key=f"{PREFIX}risk_full_paths", format_func=lambda v: f"{v:,} paths",
            label_visibility="collapsed",
            help="How many of the resampled paths to put through the funded account ledger. It "
                 "is slower than the chart, so it uses fewer paths (1,000 by default).")
    ctx.context["risk_full_paths"] = full_paths
    # model, method, seed, paths, slots, cutoff and saved terms; never the typed boundaries
    key = ledger_key(ctx, rules, slots, seed=seed, paths=full_paths)
    runnable = validation.exact and key is not None
    store = _full_results()
    with run_col:
        clicked = st_module.button(
            run_label(firm), key=f"{PREFIX}risk_full_run_{slug}", type="secondary",
            width="stretch", disabled=not runnable, help=run_help(firm))
    note_slot = note_col.empty()
    if validation.exact and key is None:  # say why the button is off (re-review)
        show(h.note("Conditional resampling isn't run for this configuration: the saved "
                    "result's trade slots or cutoff don't match this run's inputs, so a result "
                    "couldn't be tied to the saved study."), st_module)
    if clicked and runnable and key not in store:
        bar = st_module.progress(0.0, text=f"Running {full_paths:,} resampled paths with "
                                           f"{possessive(firm)} rules…")
        def progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                bar.progress(done / total,
                             text=f"Running resampled paths with {possessive(firm)} rules… "
                                  f"{done:,} of {total:,}")

        store[key] = fr.full_race(rules, slots, shapes, paths=full_paths, seed=seed,
                                  method="blocks", progress=progress, validation=validation)
        bar.empty()
    result = store.get(key) if key is not None else None
    if result is not None and getattr(result, "model_id", None) != fr.MODEL_ID:
        result = None  # never show another model's figure under this one's name
    with note_slot.container():
        beyond = beyond_chart(len(slots))
        if result is None:
            seconds = max(1, round(full_paths * 0.012))
            note = (f"Runs the first {full_paths:,} of the chart's resampled paths{beyond} "
                    f"through the funded account ledger, about {seconds} seconds. Kept "
                    "afterwards for this draw.")
        else:
            note = (f"Ran the first {full_paths:,} of the chart's resampled paths{beyond} "
                    f"through the funded account ledger in {result.seconds:.0f} seconds; kept "
                    "for this draw.")
        show(h.Markup(f'<div class="lab-line">{h.esc(note)}</div>'), st_module)
    summary = ctx.study.summary(ctx.configuration, ctx.firm_key) or {}
    actual_net = summary.get("net_cash_earned_cents")
    with boxes.container():
        if not validation.exact:
            unmatched = "Not shown: the saved order did not replay exactly"
            show(h.grid([soft_box(PAYOUTS_LABEL, unmatched, placeholder=True),
                         soft_box(CASH_LABEL, unmatched, placeholder=True)], 2), st_module)
        elif result is None:
            waiting = not_run_text(firm)
            show(h.grid([soft_box(PAYOUTS_LABEL, waiting, placeholder=True),
                         soft_box(CASH_LABEL, waiting, placeholder=True)], 2), st_module)
        else:
            show(h.grid([
                soft_box(PAYOUTS_LABEL, payouts_value(result), detail=payouts_detail(result)),
                soft_box(CASH_LABEL, fmt.money_whole(result.cash_per_account),
                         detail=cash_detail(result, rules)),
            ], 2), st_module)
    if not validation.exact:
        show(h.alert("Conditional resampling with the firm's ledger rules isn't shown for this "
                     "configuration.", validation_text(validation)), st_module)
        return
    show(h.Markup(f'<div class="lab-line">{h.esc(validation_text(validation))}'
                  "</div>"), st_module)
    if result is None:
        return
    show(full_comparison(result, fixed, actual_net, rules), st_module)
    show(h.Markup('<div class="lab" style="display:flex;flex-direction:column;gap:6px">'
                  + "".join(f'<div class="lab-line">{h.esc(text)}</div>'
                            for text in full_notes(result, rules)) + "</div>"), st_module)


def _fan_card(st_module, ctx, fan_by: dict[str, rs.EquityFan], actual: list[float],
              trades: int, paths: int) -> str:
    from ifvg_lab_ui import plot, show, switch

    with st_module.container(key="ifvg_lab_card_risk_fan"):
        title_col, switch_col = st_module.columns([2.2, 1], vertical_alignment="center")
        with title_col:
            show(title_row(fan_title(paths, trades)), st_module)
        with switch_col:
            method = switch("Sampling method", list(rs.METHODS), key="risk_method",
                            value=ctx.context.get("risk_method", "blocks"),
                            format_func=rs.METHODS.get, help=METHOD_HELP, st_module=st_module)
        ctx.context["risk_method"] = method
        if trades <= rs.BLOCK_SIZE:
            show(h.note(f"With {fmt.count(trades, 'trade')} there are no streaks of "
                        f"{rs.BLOCK_SIZE} to keep, so both methods draw single trades."),
                 st_module)
        fan = fan_by[method]
        show(fan_legend(), st_module)
        plot(fan_figure(fan, actual), key=f"risk_fan_{method}_{_pair(ctx)}", st_module=st_module)
        show(h.Markup(f'<div class="lab-line">{h.esc(fan_caption(fan))}</div>'), st_module)
        show(comparison_table(fan_by, actual[-1], ctx.row.worst_drawdown, trades), st_module)
        show(line(methods_reading(fan_by)), st_module)
    return method


def _distribution_cards(st_module, ctx, fan: rs.EquityFan, actual: list[float],
                        values: list[float]) -> None:
    from ifvg_lab_ui import plot, show

    method_word = rs.METHODS[fan.method]
    left, right = st_module.columns(2, gap="small")
    with left, st_module.container(key="ifvg_lab_card_risk_ends"):
        show(title_row(f"Where {len(values):,} trades end up", method_word, size=22), st_module)
        plot(end_figure(fan, actual[-1]), key=f"risk_ends_{fan.method}_{_pair(ctx)}",
             st_module=st_module)
        show(line(end_caption(fan, actual[-1])), st_module)
    with right, st_module.container(key="ifvg_lab_card_risk_streaks"):
        dist = rs.streak_distribution(fan.streaks)
        observed = int(rs.longest_losing_runs(np.asarray([values], dtype=float))[0])
        show(title_row("Longest losing streak", method_word, size=22), st_module)
        plot(streak_figure(dist), key=f"risk_streaks_{fan.method}_{_pair(ctx)}",
             st_module=st_module)
        show(line(streak_caption(dist, observed)), st_module)


#: the drawdown card's title and right-hand text (correction A2)
GROWTH_TITLE = "Sampled closed-profit drawdown from a previous high"
GROWTH_RIGHT = "No withdrawals · no account floor · streaks kept together"


def _growth_card(st_module, ctx, g: rs.DrawdownGrowth, lock: str | None) -> None:
    from ifvg_lab_ui import plot, show

    fall = fmt.money_whole(g.limit)
    with st_module.container(key="ifvg_lab_card_risk_growth"):
        show(title_row(GROWTH_TITLE, GROWTH_RIGHT), st_module)
        left_fig, right_fig = growth_figures(g)
        left, right = st_module.columns(2, gap="large")
        last = len(g.typical) - 1
        with left:
            show(h.Markup('<div class="lab" style="font-size:15px;font-weight:600">Largest fall '
                          "so far from a previous closed-profit high</div>"), st_module)
            plot(left_fig, key=f"risk_growth_drop_{_pair(ctx)}", st_module=st_module)
            show(h.Markup(
                f'<div class="lab-line">By trade {last}, the typical largest fall is '
                f"{h.esc(fmt.money_whole(g.typical[last]))}; one path in twenty passes "
                f"{h.esc(fmt.money_whole(g.worse_1_in_20[last]))}.</div>"), st_module)
        with right:
            show(h.Markup('<div class="lab" style="font-size:15px;font-weight:600">Share of '
                          f"sampled paths whose closed profit has fallen {h.esc(fall)} from a "
                          "previous high</div>"), st_module)
            plot(right_fig, key=f"risk_growth_touch_{_pair(ctx)}", st_module=st_module)
            points = [c for c in CHECKPOINTS if c <= last]
            first, final = points[0], points[-1]
            show(h.Markup(
                f'<div class="lab-line">{h.esc(pct(g.touched_share[first]))} of paths have '
                f"fallen {h.esc(fall)} from a previous high by trade {first} and "
                f"{h.esc(pct(g.touched_share[final]))} by trade {final}.</div>"), st_module)
        show(growth_legend(), st_module)
        show(growth_table(g), st_module)
        show(line(growth_sentence(g, ctx.firm, lock)), st_module)


def render(st_module, ctx) -> None:
    from ifvg_lab_ui import show

    from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_data import trade_path

    st_module.html(f"<style>{RISK_CSS}</style>")
    values = _values(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key)
    show(h.Markup('<div class="lab" style="font-size:14px;line-height:1.5;'
                  f'color:{css_var("body_2")};margin-top:-8px">{h.esc(explanation(len(values)))}'
                  "</div>"), st_module)
    if not values:
        show(h.note(f"This configuration has no funded trades at {ctx.firm}, so there is "
                    "nothing to resample."), st_module)
        return
    seed = current_seed(ctx)
    paths = current_paths(ctx)
    _seed_line(st_module, ctx, seed, paths)
    _race_card(st_module, ctx, values, seed, paths)
    actual = trade_path(ctx.study, ctx.configuration, ctx.firm_key)
    fan_by = fans(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key, seed, paths)
    method = _fan_card(st_module, ctx, fan_by, actual, len(values), paths)
    _distribution_cards(st_module, ctx, fan_by[method], actual, values)
    terms = fr.firm_terms(ctx.study, ctx.firm_key) or {}
    limit = (int(terms["loss_allowance_cents"]) / 100 if terms.get("loss_allowance_cents")
             else None)
    if limit is None:
        show(h.note(f"The saved result doesn't record {possessive(ctx.firm)} loss allowance, "
                    "so the closed-profit drawdown chart has no fall size to count."),
             st_module)
        return
    g = growth(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key, limit, seed,
               paths)
    _growth_card(st_module, ctx, g, lock_text(terms))
