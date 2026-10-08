"""Saved-prediction diagnostics; no fitting, policy choice or account replay."""

from __future__ import annotations

from collections import Counter

import numpy as np


def paired_diagnostics(benchmark, dates):
    """Paired squared-loss contrasts using identical five-date bootstrap draws."""
    rng = np.random.default_rng(7)
    n = len(dates)
    starts = rng.integers(0, n - 4, size=(2000, (n + 4) // 5))
    samples = (starts[..., None] + np.arange(5)).reshape(2000, -1)[:, :n]
    date_index = {day: i for i, day in enumerate(dates)}
    cells = benchmark["cells"]
    indexed = {c["cell_id"]: {r["row_id"]: r for r in benchmark["predictions"]
                              if r["cell_id"] == c["cell_id"]} for c in cells}
    contrasts = []
    for left in cells:
        for right in cells:
            if (left["reference"], left["job"]) != (right["reference"], right["job"]):
                continue
            feature = left["model"] == right["model"] and (
                left["feature_set"], right["feature_set"]) in {("F1", "F0"), ("F2", "F1")}
            model = left["feature_set"] == right["feature_set"] and (
                left["model"], right["model"]) == ("CATBOOST", "RIDGE")
            if not (feature or model):
                continue
            a, b = indexed[left["cell_id"]], indexed[right["cell_id"]]
            sums, counts = np.zeros(n), np.zeros(n)
            for key in sorted(a.keys() & b.keys()):
                x, y = a[key], b[key]
                if any(r[k] is None for r in (x, y) for k in ("prediction", "label")):
                    continue
                if x["label"] != y["label"] or x["trading_day"] != y["trading_day"]:
                    raise ValueError("paired predictions disagree on observed outcome")
                i = date_index[x["trading_day"]]
                sums[i] += (x["prediction"] - x["label"]) ** 2 - (
                    y["prediction"] - y["label"]) ** 2
                counts[i] += 1
            numerator, denominator = sums[samples].sum(axis=1), counts[samples].sum(axis=1)
            boot = numerator[denominator > 0] / denominator[denominator > 0]
            contrasts.append({"left_cell": left["cell_id"], "right_cell": right["cell_id"],
                "paired_rows": int(counts.sum()), "evaluated_dates_including_empty": n,
                "mse_difference_left_minus_right": float(sums.sum() / counts.sum())
                    if counts.sum() else None,
                "development_95pct_interval": np.quantile(boot, [.025, .975]).tolist()
                    if len(boot) else None,
                "seed": 7, "repetitions": 2000, "block_dates": 5,
                "interpretation": "Negative favors left; development uncertainty only"})
    return contrasts


def coverage(datasets, definitions):
    output = []
    for reference, jobs in datasets.items():
        for job, rows in jobs.items():
            output.append({"reference": reference, "job": job, "rows": len(rows),
                "dates": len({r["trading_day"] for r in rows}),
                "label_status": dict(Counter(r["label_status"] for r in rows)),
                "missing_features": {d["name"]: sum(r.get(d["name"]) is None for r in rows)
                                     for d in definitions if job in d["jobs"]}})
    return output


def largest_continuations(datasets, benchmark):
    """Retain every continuation, ranked posthoc, with both jobs' signed decisions."""
    scored = set(r["trading_day"] for r in benchmark["predictions"])
    output = []
    for reference, jobs in datasets.items():
        rows = [r for r in jobs["CONTINUATION"]
                if r["label_status"] == "exact" and r["trading_day"] in scored]
        def remainder_value(row):
            c = row["label_components"]
            realized = c["partial_gross_cents"] - c["entry_cost_cents"] - c["partial_cost_cents"]
            return (c["hold_balance_cents"] - realized) / c["initial_risk_cents"]

        rows.sort(key=lambda r: (-remainder_value(r), r["row_id"]))
        for rank, row in enumerate(rows, 1):
            decisions = [p for p in benchmark["predictions"]
                         if p["episode_id"] == row["episode_id"]]
            output.append({"reference": reference, "rank_baseline_remainder_value": rank,
                "row_id": row["row_id"], "episode_id": row["episode_id"],
                "incremental_hold_net_r": row["label"],
                "baseline_remainder_net_r": remainder_value(row),
                "decisions": [{"cell_id": p["cell_id"], "prediction": p["prediction"],
                               "negative": p["action"] == "change"} for p in decisions],
                "interpretation": "Retrospective diagnosis; never an input or exclusion"})
    return output


def occupancy_evidence(operations):
    """Actual new entry while the matched control still owns the displaced position."""
    proofs = []
    for reference, payload in operations.items():
        control = payload["streams"][f"NO_ML_{reference}"]
        baseline_keys = {(r["candidate"], r["decision_ns"])
                         for r in control["entry_rows"].values()}
        for name, state in payload["streams"].items():
            if name.startswith("NO_ML_"):
                continue
            changed = [r for r in (*state["entry_rows"].values(),
                                  *state["checkpoint_rows"].values())
                       if r["prediction"]["action"] == "change"]
            for row in state["entry_rows"].values():
                if (row["candidate"], row["decision_ns"]) in baseline_keys:
                    continue
                for trade in control["ledger"]["trades"]:
                    if not trade["entry_ns"] < row["decision_ns"] < trade["exit_ns"]:
                        continue
                    old = control["driver"]["ml_entries"][trade["trade_ref"]]
                    prior = next((r for r in changed if r["candidate"] == old["candidate_id"]
                                  and r["decision_ns"] < row["decision_ns"]), None)
                    if prior is not None:
                        proofs.append({"operation": name, "reference": reference,
                            "changed_prior_decision": prior,
                            "new_actual_entry_decision": row,
                            "matched_control_occupied_trade": trade,
                            "matched_control_candidate": old["candidate_id"],
                            "new_in_matched_control": False,
                            "interpretation": "Earlier learned action frees the policy slot; "
                                              "matched control is still in its old position"})
    return proofs
