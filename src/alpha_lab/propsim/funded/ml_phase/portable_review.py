"""Recheck delivered labels, all predictions and ledger money without market history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .benchmark import OnlineScorer, frame_from_rows
from .models import RegressionFit, select_fold
from .protocol import digest


def verify(report, model_root):
    if digest({k: v for k, v in report.items() if k != "report_id"}) != report["report_id"]:
        raise PermissionError("report identity mismatch")
    datasets, benchmark, contracts = report["datasets"], report["benchmark"], report["contracts"]
    labels = 0
    for jobs in datasets.values():
        for job, rows in jobs.items():
            assert len({r["row_id"] for r in rows}) == len(rows)
            for row in rows:
                assert row["label_start_ns"] <= row["label_end_ns"] <= row["label_available_ns"]
                assert all(p["known_at_ns"] <= row["decision_ns"]
                           for p in row["feature_provenance"].values())
                if row["label_status"] != "exact":
                    assert row["label"] is None
                    continue
                c = row["label_components"]
                net = c["gross_pnl_cents"] - c["entry_cost_cents"] - c["partial_cost_cents"] - c[
                    "exit_cost_cents"]
                assert net == c["net_pnl_cents"]
                numerator = net if job == "ENTRY" else c["hold_balance_cents"] - c[
                    "close_balance_cents"]
                np.testing.assert_allclose(row["label"], numerator / c["initial_risk_cents"],
                                           rtol=0, atol=1e-14)
                labels += 1
    assert digest(datasets) == benchmark["identity"]["datasets_sha256"]
    scored = 0
    for cell in benchmark["cells"]:
        scorer = OnlineScorer(cell, benchmark, model_root, contracts)
        rows = {r["row_id"]: r for r in datasets[cell["reference"]][cell["job"]]}
        for record in [r for r in benchmark["fits"] if r["cell_id"] == cell["cell_id"]]:
            if record["model_id"] is None:
                continue
            model = RegressionFit.load(model_root / record["fit_path"])
            state = model.state
            train, _, _ = select_fold(frame_from_rows(list(rows.values())),
                state["identity"]["fold"], **state["identity"]["boundaries"])
            assert train.row_id.tolist() == state["train_row_ids"]
            assert digest(train.label.tolist()) == state["train_labels_sha256"]
        for prediction in benchmark["predictions"]:
            if prediction["cell_id"] != cell["cell_id"]:
                continue
            actual = scorer(cell["job"], rows[prediction["row_id"]])
            assert actual["action"] == prediction["action"]
            if actual["score"] is None:
                assert prediction["prediction"] is None
            else:
                np.testing.assert_allclose(actual["score"], prediction["prediction"],
                                           rtol=1e-12, atol=1e-12)
            scored += 1
    operations, online_decisions = 0, 0
    cells = {c["cell_id"]: c for c in benchmark["cells"]}
    for payload in report["operations"].values():
        for name, stream in payload["streams"].items():
            if name in cells:
                scorer = OnlineScorer(cells[name], benchmark, model_root, contracts)
                for row in (*stream["entry_rows"].values(), *stream["checkpoint_rows"].values()):
                    actual = row["prediction"]
                    if row.get("checkpoint_fidelity") not in (None, "ordered_trade_prints"):
                        assert actual["score"] is None
                        assert actual["action"] == "baseline_unavailable"
                        assert actual["reason"] == "approximate_checkpoint"
                        continue
                    predicted = scorer(row["job"], row)
                    for key in ("action", "model_id", "fold_id", "training_cutoff_ns",
                                "activation_ns", "new_opportunity"):
                        assert predicted.get(key) == actual.get(key)
                    if predicted["score"] is None:
                        assert actual["score"] is None
                    else:
                        np.testing.assert_allclose(predicted["score"], actual["score"],
                                                   rtol=1e-12, atol=1e-12)
                    online_decisions += 1
            ledger = stream["ledger"]
            costs = sum(r["amount_cents"] for r in ledger["cash_ledger"]
                        if r["kind"] == "account_purchase")
            receipts = sum(r["amount_cents"] for r in ledger["cash_ledger"]
                           if r["kind"] == "payout_received")
            assert costs == 12500 * len(ledger["accounts"]) == ledger["costs"]
            assert receipts == ledger["receipts"]
            summary = report["economics"]["economic_result"]["summaries_cents"][
                f"{name}|myfundedfutures"]
            assert summary["net_cash_earned_cents"] == receipts - costs
            assert ledger["finished"] and ledger["position"] is None
            assert ledger["accounts"][0]["created_ns"] == payload["start_ns"]
            operations += 1
    assert operations == 26 and len(benchmark["cells"]) == 24
    return {"passed": True, "exact_labels_checked": labels, "predictions_replayed": scored,
            "actual_policy_decisions_replayed": online_decisions,
            "operations_reconciled": operations, "raw_market_history_accessed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    report = json.loads((args.folder / "report.json").read_text(encoding="utf-8"))
    print(json.dumps(verify(report, args.folder / "models"), indent=2))


if __name__ == "__main__":
    main()
