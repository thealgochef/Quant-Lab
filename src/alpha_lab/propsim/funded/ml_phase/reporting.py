"""Read-only projection of completed ML operations through existing funded reporting."""

from __future__ import annotations

import copy
import json
from collections import Counter
from types import SimpleNamespace

from alpha_lab.agents.data_infra.ifvg.presentation.lab.mffu_lenses import build_rows
from alpha_lab.propsim.funded.comparison_result import (
    build_comparison_result,
    validate_comparison,
)

from .protocol import digest


def operation_description(cell):
    if cell is None:
        return "Matched control: enter baseline setups; five off at 1R, hold five to BE/close"
    job = ("Reject entry only when expected net R < 0" if cell["job"] == "ENTRY" else
           "Five off at 1R; close five on next print only when incremental HOLD R < 0")
    return f"{cell['reference']} · {cell['feature_set']} · {cell['model']} · {job}"


def economic_report(*, operations, contracts, calendar, profiles, identity):
    cells = {c["cell_id"]: c for c in contracts["MODEL_MATRIX"]["cells"]}
    outputs, coverage = [], []
    for reference, payload in operations.items():
        for name, state in payload["streams"].items():
            ledger = copy.deepcopy(state["ledger"])
            saved_pair = ledger["pair_id"]
            pair_id = f"{name}|myfundedfutures"
            ledger["pair_id"] = pair_id
            for table in ("cash_ledger", "payout_events", "account_events", "trades",
                          "boundary_evidence"):
                for row in ledger[table]:
                    row["saved_pair_id"] = row["pair_id"]
                    row["pair_id"] = pair_id
            cell = cells.get(name)
            decision_rows = [*state["entry_rows"].values(), *state["checkpoint_rows"].values()]
            selected = [r for r in decision_rows if cell and r["job"] == cell["job"]]
            coverage.append({"operation": name, "saved_pair_id": saved_pair,
                "report_pair_id": pair_id, "reference": reference,
                "decisions": len(selected),
                "actions": dict(Counter(r["prediction"]["action"] for r in selected)),
                "reasons": dict(Counter(r["prediction"].get("reason") or "available"
                                        for r in selected)),
                "new_opportunities": sum(bool(r["prediction"].get("new_opportunity"))
                                         for r in selected)})
            stats = payload["stats"]
            outputs.append({"configuration": name, "display_name": operation_description(cell),
                "axes": {"reference": reference},
                "settings_plain": [{"setting": "ML policy", "value": operation_description(cell)}],
                "exit_policy": "scale_out_half_breakeven_hold_to_close_v1",
                "sizing": {"instrument_label": "micros (NQ tape proxy)", "quantity": 10,
                           "tick_value_cents": 50, "cost_per_contract_mills": 514},
                "reference": {"equivalent": None, "saved_study_trades": None,
                              "replayed_trades": len(state["driver"]["ml_entries"])},
                "resumed": {}, "prints": {**stats,
                    "minutes_rebuilt_exactly": stats["minutes_matched"]},
                "pairs": {"myfundedfutures": {"pair_id": pair_id, "ledger": ledger,
                    "forced_flat": sum(t["account_failed"] for t in ledger["trades"]),
                    "trades_not_in_reference": None}}})
    sample = next(iter(operations.values()))
    result = build_comparison_result(context={"ml_phase": identity}, outputs=outputs,
        failures=[], profiles=profiles, trading_days=calendar, start_ns=sample["start_ns"],
        cutoff_ns=sample["cutoff_ns"], settings={"tick_value_cents": 50},
        resume_check_requested=False)
    validation = validate_comparison(result, outputs, profiles, sample["cutoff_ns"],
                                    require_resume_check=False, require_reference_check=False)
    result_id = digest(result)
    study = SavedPhaseStudy(result, result_id, contracts)
    lenses = build_rows(study)
    by_name = {r["configuration_id"]: r for r in lenses}
    for row in lenses:
        ref = row["reference"]
        control = by_name[f"NO_ML_{ref}"]
        row["matched_control"] = control["configuration_id"]
        row["net_cash_delta_cents"] = (
            row["net_received_cash_cents"] - control["net_received_cash_cents"])
    return {"result_id": result_id, "economic_result": result, "lenses": lenses,
            "operation_coverage": coverage, "financial_validation": validation}


class SavedPhaseStudy:
    def __init__(self, result, result_id, contracts):
        self.result, self.result_id = result, result_id
        self.calendar = contracts["DATE_AND_FOLD_PLAN"]["scored_dates"]
        cells = {c["cell_id"]: c for c in contracts["MODEL_MATRIX"]["cells"]}
        variants = []
        for summary in result["summaries_cents"].values():
            name = summary["configuration"]
            cell = cells.get(name)
            ref = cell["reference"] if cell else name.removeprefix("NO_ML_")
            variants.append(SimpleNamespace(name=name, intent_json=json.dumps({
                "reference": ref, "job": cell["job"] if cell else "CONTROL",
                "feature_set": cell["feature_set"] if cell else "NONE",
                "model": cell["model"] if cell else "NONE",
                "description": operation_description(cell)})))
        self.plan = SimpleNamespace(source=SimpleNamespace(
            cutoff_utc=result["period"]["cutoff_utc"]), configurations=variants)
        self.trades_by_pair = {f"{v.name}|myfundedfutures": self.rows(
            "trades", v.name, "myfundedfutures") for v in variants}

    def rows(self, table, configuration, firm):
        return [r for r in self.result["tables"][table]
                if r["configuration"] == configuration and r["firm_key"] == firm]

    def summary(self, configuration, firm):
        return self.result["summaries_cents"].get(f"{configuration}|{firm}")
