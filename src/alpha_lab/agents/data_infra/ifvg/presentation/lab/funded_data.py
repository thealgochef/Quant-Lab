"""Read-only access to one saved funded configuration comparison.

``open_funded_study`` loads the ONE verified result (hash-checked, with the
recorded summary-only reporting corrections applied on load — the saved bytes
are never changed) and its verified plan, which carries the study's own
trading calendar. Nothing here recomputes a money figure: headline cash comes
from ``summaries_cents``; per-trade series are the saved trade rows.

Every series belongs to ONE configuration at ONE firm. Nothing here adds
configurations or firms together.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_lab.agents.data_infra.ifvg.presentation.chicago_time import utc_instant

__all__ = [
    "COMPLETED",
    "FundedStudy",
    "daily_results",
    "firm_name",
    "open_funded_study",
    "ordered_trades",
    "package_gate_thresholds",
    "pair_key",
    "study_from_result",
    "trade_path",
]

COMPLETED = "Completed"


def pair_key(configuration: str, firm_key: str) -> str:
    """The result's internal pair identifier (never displayed)."""

    return f"{configuration}|{firm_key}"


def _instant_ns(value: Any) -> int:
    stamp = utc_instant(value) if value else None
    return -1 if stamp is None else int(stamp.value)


@dataclass(frozen=True)
class FundedStudy:
    """One saved funded comparison result, indexed for the screens."""

    result_id: str
    plan_id: str | None
    store_root: Path | None
    result: dict[str, Any] = field(repr=False)
    #: the study's evaluation trading days (``YYYY-MM-DD``, by closing date)
    calendar: tuple[str, ...] = ()
    warmup: tuple[str, ...] = ()
    #: (firm key, firm name) in the saved profile order
    firms: tuple[tuple[str, str], ...] = ()
    #: configuration keys in saved order
    configurations: tuple[str, ...] = ()
    #: pair key → trade rows in execution order (entry instant, then saved sequence)
    trades_by_pair: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict,
                                                                   repr=False)
    #: plan fields the screens read (source title, settings); ``None`` if not loaded
    plan: Any = field(default=None, repr=False)

    def summary(self, configuration: str, firm_key: str) -> dict[str, Any] | None:
        return (self.result.get("summaries_cents") or {}).get(pair_key(configuration, firm_key))

    def summary_usd(self, configuration: str, firm_key: str) -> dict[str, Any] | None:
        return (self.result.get("summaries") or {}).get(pair_key(configuration, firm_key))

    def completed_at(self, firm_key: str) -> list[dict[str, Any]]:
        """Completed pair summaries at one firm, in saved rank order."""

        rows = [s for s in (self.result.get("summaries_cents") or {}).values()
                if s.get("firm_key") == firm_key and s.get("status") == COMPLETED]
        rows.sort(key=lambda s: (s.get("rank_within_firm") is None,
                                 s.get("rank_within_firm") or 0, str(s.get("configuration"))))
        return rows

    def strategy_measures(self, configuration: str) -> dict[str, Any] | None:
        for row in (self.result.get("tables") or {}).get("strategy_metrics") or []:
            if row.get("configuration") == configuration:
                return row
        return None

    def configuration_row(self, configuration: str) -> dict[str, Any] | None:
        for row in (self.result.get("tables") or {}).get("configurations") or []:
            if row.get("configuration") == configuration:
                return row
        return None

    def settings(self, configuration: str) -> list[tuple[str, str]]:
        row = self.configuration_row(configuration) or {}
        return [(str(i.get("setting", "")), str(i.get("value", "")))
                for i in row.get("settings") or []]

    def rows(self, table: str, configuration: str, firm_key: str) -> list[dict[str, Any]]:
        key = pair_key(configuration, firm_key)
        return [r for r in (self.result.get("tables") or {}).get(table) or []
                if r.get("pair_id") == key]


def firm_name(study: FundedStudy, firm_key: str) -> str:
    return dict(study.firms).get(firm_key, firm_key)


def _firm_order(result: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for profile in (result.get("settings") or {}).get("firm_profiles") or []:
        key = profile.get("firm_key")
        if key and key not in seen:
            seen.add(key)
            order.append((key, str(profile.get("firm_name") or key)))
    for summary in (result.get("summaries_cents") or {}).values():
        key = summary.get("firm_key")
        if key and key not in seen:
            seen.add(key)
            order.append((key, str(summary.get("firm") or key)))
    return tuple(order)


def study_from_result(result: dict[str, Any], *, result_id: str, plan: Any = None,
                      store_root: Path | None = None) -> FundedStudy:
    """Index a loaded result (and optionally its plan payload) for the screens."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in (result.get("tables") or {}).get("trades") or []:
        grouped[str(row.get("pair_id"))].append(row)
    trades = {key: tuple(sorted(rows, key=lambda t: (_instant_ns(t.get("entry_utc")),
                                                     t.get("seq") or 0)))
              for key, rows in grouped.items()}
    source = getattr(plan, "source", None)
    calendar = tuple(getattr(source, "evaluation_dates", None) or ())
    warmup = tuple(getattr(source, "warmup_dates", None) or ())
    return FundedStudy(
        result_id=result_id,
        plan_id=result.get("funded_comparison_plan_id"),
        store_root=store_root,
        result=result,
        calendar=calendar,
        warmup=warmup,
        firms=_firm_order(result),
        configurations=tuple(str(c.get("configuration"))
                             for c in (result.get("tables") or {}).get("configurations") or []),
        trades_by_pair=trades,
        plan=plan,
    )


def open_funded_study(store_root: Path, result_id: str) -> FundedStudy:
    """Load the verified result and its verified plan (read only)."""

    from alpha_lab.propsim.funded.comparison_runner import load_comparison_result, load_plan

    result = load_comparison_result(Path(store_root), result_id)
    plan = None
    plan_id = result.get("funded_comparison_plan_id")
    if plan_id:
        try:
            plan = load_plan(Path(store_root), str(plan_id))
        except Exception:  # the result stays readable; the calendar is then unavailable
            plan = None
    return study_from_result(result, result_id=result_id, plan=plan, store_root=Path(store_root))


def package_gate_thresholds(package_root: Path | None) -> dict[str, Any] | None:
    """The strategy study's saved gate thresholds (``run_context.json``), read only.

    ``frozen_batch.gate_policy.feasibility_gates`` of the verified package the
    funded plan is bound to (its run id and manifest hash were checked when the
    package was located). ``None`` when the package or the record is absent.
    """

    import json

    if package_root is None:
        return None
    path = Path(package_root) / "run_context.json"
    if not path.is_file():
        return None
    try:
        context = json.loads(path.read_text(encoding="utf-8"))
        gates = context["frozen_batch"]["gate_policy"]["feasibility_gates"]
    except (KeyError, TypeError, ValueError):
        return None
    return dict(gates) if isinstance(gates, dict) else None


def ordered_trades(study: FundedStudy, configuration: str,
                   firm_key: str) -> tuple[dict[str, Any], ...]:
    """All funded trades of one configuration at one firm, across its accounts, in order."""

    return study.trades_by_pair.get(pair_key(configuration, firm_key), ())


def daily_results(study: FundedStudy, configuration: str,
                  firm_key: str) -> list[tuple[str, float]]:
    """Net result per trading day over the study calendar, 0 on days without a trade.

    A trade belongs to its saved ``trading_day`` (the trading day runs from the
    5:00 PM reopen to the 4:00 PM close and is named by its closing date).
    Raises if the calendar is unavailable or a trade falls outside it.
    """

    if not study.calendar:
        raise ValueError("this study's trading calendar is not available")
    totals = dict.fromkeys(study.calendar, 0.0)
    for trade in ordered_trades(study, configuration, firm_key):
        day = str(trade.get("trading_day"))
        if day not in totals:
            raise ValueError(f"a trade on {day} is outside the study's trading calendar")
        totals[day] += float(trade.get("net_pnl_usd") or 0.0)
    return [(day, round(value, 2)) for day, value in totals.items()]


def trade_path(study: FundedStudy, configuration: str, firm_key: str) -> list[float]:
    """Cumulative net result by trade, starting at $0 before the first trade."""

    path = [0.0]
    for trade in ordered_trades(study, configuration, firm_key):
        path.append(round(path[-1] + float(trade.get("net_pnl_usd") or 0.0), 2))
    return path
