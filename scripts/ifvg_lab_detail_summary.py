"""Configuration detail · Summary (mocks 03, 03b).

Headline tiles, the four-part verdict, findings with next steps, key measures
(with the working 68/90/95% range switch), concentration and quality gates —
all for ONE configuration at ONE firm, computed as CALCULATIONS.md defines.
"""

from __future__ import annotations

from alpha_lab.agents.data_infra.ifvg.presentation.lab import format as fmt
from alpha_lab.agents.data_infra.ifvg.presentation.lab import html as h
from alpha_lab.agents.data_infra.ifvg.presentation.lab.funded_measures import verdict
from alpha_lab.agents.data_infra.ifvg.presentation.lab.theme import css_var

_SEVERITY_TONE = {"High": "orange", "Medium": "orange", "Info": "neutral"}


def _tiles(ctx) -> h.Markup:
    row = ctx.row
    return h.grid([
        h.tile("Net cash earned", fmt.money_cents(row.net_cash_cents), big=True),
        h.tile("Received after the split", fmt.money_cents(row.received_cents),
               fmt.count(row.payouts or 0, "payout")),
        h.tile("Account costs", fmt.money_cents(row.costs_cents),
               fmt.count(row.accounts or 0, "account")),
        h.tile("Largest payout", fmt.money_cents(row.largest_payout_cents,
                                                 missing="No payout received")),
        h.tile("Median payout", fmt.money_cents(row.median_payout_cents,
                                                missing="No payout received")),
    ], 5)


def _verdict_parts(ctx, bundle) -> list:
    result = ctx.study.result
    evidence = result.get("price_evidence") or {}
    checked = int(evidence.get("position_minutes_checked") or 0)
    exact = int(evidence.get("position_minutes_rebuilt_exactly_from_prints") or 0)
    approximated = int(evidence.get("position_minutes_approximated") or 0)
    reconciles = bool((result.get("validation") or {}).get("passed"))
    missing_days = list(evidence.get("missing_print_days") or [])
    integrity_ok = (reconciles and checked == exact + approximated and not missing_days
                    and checked > 0)
    integrity = (f"{exact:,} of {checked:,} position minutes on recorded trades. "
                 + ("Money reconciles. " if reconciles else "Money does not reconcile. ")
                 + f"{fmt.count(bundle.trades_approximated, 'trade')} for this configuration "
                 f"used the one-minute approximation.")
    ranges = bundle.ranges
    low95, high95 = ranges.ranges[95] if ranges else (None, None)
    deflated = bundle.sharpe.deflated if bundle.sharpe else None
    tested = bundle.sharpe.tested if bundle.sharpe else None
    edge = (f"Result per trade: 95% range {fmt.money_whole(low95)} to "
            f"{fmt.money_whole(high95)}, {'above' if (low95 or 0) > 0 else 'not above'} zero."
            if ranges else "No trades to resample.")
    if deflated is not None:
        edge += f" Deflated Sharpe ratio for {tested} configurations compared: {deflated:.2f}."
    else:  # unavailable is not a failure (correction A11): say which check wasn't made
        edge += (" The deflated Sharpe ratio isn't available for this comparison (for "
                 "example fewer than two completed configurations, or daily results that don't "
                 "vary), so that check wasn't made.")
    limit = bundle.loss_limit
    row = ctx.row
    # correction A2: a closed-profit drawdown across accounts, not an account's floor
    worst = fmt.money_whole(row.worst_drawdown)
    account = (f"Worst closed-profit drawdown of the trade path across accounts {worst} "
               f"(each account's loss allowance: {fmt.money_whole(limit)})." if limit else
               f"Worst closed-profit drawdown of the trade path across accounts {worst}.")
    lost = row.lost_before_payout or 0
    if lost and bundle.lost_first_accounts == lost:
        account += f" The first {fmt.count(lost, 'account')} " \
                   f"{'was' if lost == 1 else 'were'} lost before any payout."
    elif lost:
        account += f" {fmt.count(lost, 'account')} {'was' if lost == 1 else 'were'} lost " \
                   "before any payout."
    else:
        account += " No account was lost before its first payout."
    days = bundle.trading_days
    sample = (f"One {days}-day window. {fmt.count(row.trades or 0, 'funded trade')}"
              + (f", {fmt.count(int(bundle.strategy_trades), 'strategy trade')}"
                 if bundle.strategy_trades is not None else "")
              + ". June 11 onward not yet tested.")
    return verdict(integrity_ok=integrity_ok, integrity_text=integrity, range95_low=low95,
                   deflated=deflated, edge_text=edge, worst_drawdown=row.worst_drawdown,
                   loss_limit=limit, lost_before_payout=lost, account_text=account,
                   trading_days=days, has_unseen_result=False, sample_text=sample)


def _findings_card(ctx, bundle) -> h.Markup:
    from ifvg_lab_cache import pair_findings

    # the conditional ledger figure when it was run at the default draw (kept in
    # memory), else the fixed diagnostic; never run here. The overview's finding
    # count uses the same helper.
    items = pair_findings(bundle, ctx.store_root, ctx.result_id, ctx.configuration,
                          ctx.firm_key, ctx.firm)
    rows = [h.finding_row(f.severity, _SEVERITY_TONE.get(f.severity, "neutral"), f.title,
                          f.text, f.next_step) for f in items]
    if not rows:
        rows = [h.note("None of the default findings fire for this configuration.")]
    return h.card(h.Markup("".join(rows)), title=f"Findings · {len(items)}")


def _range_text(bundle, level: int) -> str:
    if not bundle.ranges:
        return "No trades to resample."
    low, high = bundle.ranges.ranges[level]
    return f"{level}% range: {fmt.money_whole(low)} to {fmt.money_whole(high)}"


def sharpe_note(sharpe) -> str | None:
    """How both Sharpe scores are measured (correction A5); None without the moments."""

    if not sharpe or sharpe.skew is None or sharpe.kurtosis is None:
        return None
    lead = (f"How the Sharpe scores are measured: both use this configuration's "
            f"{sharpe.days:,} daily funded results (days without a trade count as $0), treat "
            f"the days as independent draws with the observed skew ({sharpe.skew:.2f}) and "
            f"kurtosis ({sharpe.kurtosis:.2f}), and report a normal-approximation confidence "
            "score (Φ of a z-score, like one minus a one-sided p-value) comparing the observed "
            "daily Sharpe ratio with a benchmark")
    if sharpe.tested and sharpe.benchmark_annualized is not None and sharpe.deflated is not None:
        n = sharpe.tested
        return (f"{lead} — 0 for the probabilistic Sharpe ratio, and the expected best of the "
                f"{n} configurations compared in this study at this firm "
                f"({sharpe.benchmark_annualized:.2f} annualized) for the deflated one. Neither "
                "is a probability that the edge is genuine or that payouts will continue, and "
                f"the {n}-configuration adjustment doesn't account for research done before "
                "this study.")
    return (f"{lead} of 0 (the probabilistic Sharpe ratio). It is not a probability that the "
            "edge is genuine or that payouts will continue.")


def _measure_tiles(ctx, bundle) -> h.Markup:
    row = ctx.row
    sharpe = bundle.sharpe
    # correction A5: a probabilistic Sharpe ratio, not a "chance the true Sharpe is above zero"
    chance = (h.placeholder("Not enough days") if not sharpe or sharpe.above_zero is None
              else ("> 0.99" if sharpe.above_zero > 0.99 else f"{sharpe.above_zero:.2f}"))
    deflated = (f"Deflated for {sharpe.tested} configurations compared: {sharpe.deflated:.2f}"
                if sharpe and sharpe.deflated is not None else "Deflation not available")
    measures = ctx.study.strategy_measures(ctx.configuration) or {}
    pf_strategy = measures.get("profit_factor")
    skew_text = "Not enough days"
    tail = ""
    if sharpe and sharpe.skew is not None and sharpe.kurtosis is not None:
        skew_text = f"{sharpe.skew:.2f} · {sharpe.kurtosis:.2f}"
        tail = ("A long right tail of big winning days" if sharpe.skew > 1
                else "A long left tail of big losing days" if sharpe.skew < -1
                else "Winning and losing days are fairly balanced")
    tie = bundle.tie
    hold = bundle.hold
    hold_caption = "Not available"
    if hold and hold.result is not None:
        # one entry instant for the profit and the fall (correction A10)
        entered = fmt.chicago_long(hold.entry_utc).replace(", 2026", "", 1)
        hold_caption = (f"Bought at the {entered} open; first fell "
                        f"{fmt.money_whole(hold.limit)} from its high "
                        f"{fmt.chicago_long(hold.breach_utc).replace(', 2026', '', 1)}"
                        if hold.breach_utc is not None
                        else f"Bought at the {entered} open; never fell "
                             f"{fmt.money_whole(hold.limit)} from its high")
    top = h.grid([
        h.tile("Sharpe", fmt.number(row.sharpe), f"Sortino {fmt.number(row.sortino)}"),
        h.tile("Probabilistic Sharpe ratio", chance, deflated),
    ], 2)
    bottom = h.grid([
        h.tile("Profit factor", fmt.number(bundle.funded_profit_factor),
               f"{fmt.number(pf_strategy)} without accounts" if pf_strategy is not None
               else "Without accounts: not in export"),
        h.tile("Skew · tail weight", skew_text, tail),
        h.tile("Tie to the index (beta)",
               fmt.number(tie.beta) if tie and tie.beta is not None else fmt.MISSING,
               # correction A5: an observation about daily co-movement, not independence
               f"R² {tie.r_squared:.2f}: a straight-line fit explains "
               f"{fmt.percent(tie.r_squared)} of daily variation"
               if tie and tie.r_squared is not None else "Stored index bars not available"),
        h.tile("Buy and hold one E-mini, same dates",
               fmt.money_whole(hold.result, signed=True) if hold and hold.result is not None
               else fmt.MISSING, hold_caption),
    ], 4)
    return top, bottom


def _gates_card(bundle) -> h.Markup:
    columns = [h.Column("gate", "Gate", width="36%"),
               h.Column("required", "Required", "right", width="22%"),
               h.Column("value", "This result", "right", width="28%"),
               h.Column("status", "", "right", width="14%")]
    rows = []
    for gate in bundle.gates:
        value = h.placeholder("Not in export") if gate.value is None else gate.value
        if gate.note:
            value = h.Markup(f"{h.esc(value)}<div class=\"lab-cell-sub\">{h.esc(gate.note)}</div>")
        if gate.passed is True:
            status = h.Markup(f'<b style="color:{css_var("blue")}">Pass</b>')
        elif gate.passed is False:
            status = h.Markup(f'<b style="color:{css_var("orange")}">Fail</b>')
        else:
            status = ""
        required = gate.required
        if gate.needs_decision:
            required = h.Markup(f"{h.esc(gate.required)}<div class=\"lab-cell-sub\">saved · "
                                "awaiting your decision</div>")
        rows.append(h.Row({"gate": gate.gate, "required": required, "value": value,
                           "status": status}))
    note = ("Evaluated on the strategy replay without accounts, against the strategy study's "
            "saved gate settings. A gate the replay did not store shows \"Not in export\"; "
            "the funded-trade figure under it is for reference only and is not counted.")
    return h.card(h.Markup(str(h.table(columns, rows, plain=True, wrap=False))
                           + f'<div class="lab-line">{h.esc(note)}</div>'),
                  title="Quality gates")


def _concentration_card(bundle) -> h.Markup:
    c = bundle.concentration
    month = fmt.date_long(f"{c.best_month}-01").split(" ")[0] if c.best_month else None
    rows = [
        ("Largest account's share of cash received", c.largest_account_share),
        ("Five largest trades' share of trading profit", c.five_largest_trades_share),
        (f"Best month's share of net cash{f' ({month})' if month else ''}", c.best_month_share),
        ("Largest payout's share of cash received", c.largest_payout_share),
        ("Best day's share of profit", c.best_day_share),
    ]
    body = h.bars([(label, share, fmt.percent(share) if share is not None else "—")
                   for label, share in rows])
    note = ("Best day's share = the best day's result ÷ the sum of all winning days. "
            "Bars turn orange at 50% and above.")
    return h.card(h.Markup(f"{body}<div class=\"lab-line\">{h.esc(note)}</div>"),
                  title="How concentrated is the result?")


def render(st_module, ctx) -> None:
    from ifvg_lab_cache import summary_bundle
    from ifvg_lab_ui import show, switch

    bundle = summary_bundle(ctx.store_root, ctx.result_id, ctx.configuration, ctx.firm_key)
    show(_tiles(ctx), st_module)
    show(h.section_title("Verdict"), st_module)
    show(h.grid([h.verdict_card(p.part, p.status, p.tone, p.text)
                 for p in _verdict_parts(ctx, bundle)], 4), st_module)
    show(_findings_card(ctx, bundle), st_module)
    show(h.section_title("Key measures"), st_module)
    top, bottom = _measure_tiles(ctx, bundle)
    left, right = st_module.columns([1, 1], gap="small")
    with left, st_module.container(key="ifvg_lab_card_average"):
        label_col, switch_col = st_module.columns([1.4, 1], vertical_alignment="center")
        with label_col:
            show(h.Markup('<div class="lab-tile-label">Average result per trade</div>'),
                 st_module)
        with switch_col:
            level = switch("Range shown", [68, 90, 95], key="range_level",
                           value=ctx.context.get("range_level", 95),
                           format_func=lambda v: f"{v}%",
                           help="How wide a range of resampled averages to show: 68%, "
                                "90% or 95% of 20,000 resamples of this configuration's "
                                "own trades.", st_module=st_module)
            ctx.context["range_level"] = level
        mean = bundle.ranges.mean if bundle.ranges else None
        show(h.Markup(f'<div class="lab-tile-value big">{h.esc(fmt.money(mean))}</div>'
                      f'<div class="lab-tile-caption">{h.esc(_range_text(bundle, level))}'
                      "</div>"), st_module)
    with right:
        show(top, st_module)
    show(bottom, st_module)
    note = sharpe_note(bundle.sharpe)
    if note:
        show(h.Markup(f'<div class="lab-line">{h.esc(note)}</div>'), st_module)
    show(h.grid([_concentration_card(bundle), _gates_card(bundle)], 2, gap=16), st_module)
    if bundle.ranges:
        show(h.Markup(f'<div class="lab-line">Ranges resample this configuration\'s '
                      f"{bundle.ranges.count} funded trades {bundle.ranges.paths:,} times with a "
                      "fixed seed; they describe these trades, not a forecast.</div>"),
             st_module)
