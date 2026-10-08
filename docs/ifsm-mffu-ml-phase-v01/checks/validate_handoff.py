"""Check this handoff's contracts and synthetic examples only. No model fit or market read."""
from __future__ import annotations
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def input_path(relative: str) -> Path:
    """Resolve preserved binary inputs relocated out of the code checkout."""
    local = ROOT / relative
    if local.exists() or local.suffix.lower() != ".zip":
        return local
    archived = (
        ROOT.parents[2] / "Claude-Quant-Lab-Research-Artifacts" / ROOT.name
        / "repository-evidence" / relative
    )
    return archived

def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def fee(q: int) -> Decimal:
    if q <= 0:
        raise ValueError("quantity must be positive")
    return (Decimal(q) * Decimal("0.514")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def verify() -> dict:
    manifest = load("references/INPUT_MANIFEST.json")
    for row in manifest["files"]:
        p = input_path(row["path"])
        b = p.read_bytes()
        assert len(b) == row["bytes"], row["path"]
        assert hashlib.sha256(b).hexdigest() == row["sha256"], row["path"]
        if p.suffix == ".zip":
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None, p.name
    matrix = load("contracts/MODEL_MATRIX.json")
    cells = matrix["cells"]
    assert len(cells) == 24
    assert len({c["cell_id"] for c in cells}) == 24
    assert len({(c["reference"], c["job"], c["feature_set"], c["model"]) for c in cells}) == 24
    assert all(c["threshold"] == 0 and not c["combined_entry_and_exit_policy"] for c in cells)
    assert len(matrix["funded_controls"]) == 2
    scope = load("references/INHERITED_DATE_SCOPE.json")
    schedule = load("contracts/DATE_AND_FOLD_PLAN.json")
    days = scope["evaluation_dates"]
    assert len(days) == len(set(days)) == 253
    assert len(scope["warmup_dates"]) == 10
    assert sorted(days) == days
    assert schedule["original_evaluation_dates"] == days
    assert len(schedule["folds"]) == 9
    observed_test = []
    for n, fold in enumerate(schedule["folds"]):
        s = 82 + 20 * n
        end = min(s + 20, len(days))
        assert fold["train_candidate_dates"] == days[:s-2]
        assert fold["separation_dates"] == days[s-2:s]
        assert fold["test_dates"] == days[s:end]
        assert not (set(fold["train_candidate_dates"]) & set(fold["test_dates"]))
        observed_test += fold["test_dates"]
    assert observed_test == schedule["scored_dates"] == days[82:]
    assert len(observed_test) == len(set(observed_test)) == 171
    assert observed_test[0] == "2025-10-09" and observed_test[-1] == "2026-06-10"
    assert not (set(days) & set(scope["known_missing_dates"]))
    definitions = load("contracts/FEATURE_DEFINITIONS.json")
    lists = load("contracts/FEATURE_LISTS.json")
    byname = {f["name"]: f for f in definitions["rows"]}
    assert len(byname) == len(definitions["rows"])
    for job in ["ENTRY", "CONTINUATION"]:
        for group in ["F0", "F1", "F2"]:
            expected = [f["name"] for f in definitions["rows"]
                        if job in f["jobs"] and f["group"] in definitions["feature_groups"][group]]
            assert lists[job][group] == expected
            assert len(expected) == len(set(expected))
        assert set(lists[job]["F0"]) < set(lists[job]["F1"]) < set(lists[job]["F2"])
        assert "frozen_max_distance_points" not in lists[job]["F0"]
    original = load("references/B8DB_PLAN_ENVELOPE.json")
    assert original["payload"]["source"]["evaluation_dates"] == days
    for name in ["MCB062","MCB025"]:
        copied = load(f"references/{name}_SAVED_VARIANT.json")
        found = next(v for v in original["payload"]["variants"] if v["batch_id"] == name)
        assert copied == found
        section = load(f"references/{name}_EFFECTIVE_SECTION.json")
        assert section == json.loads(found["effective_section_json"])
        assert copied["quantity"] == 10 and copied["instrument"] == "micro"
        assert section["entry_family"] == "fresh_fvg_continuation"
        assert section["max_executed_trades_per_day"] is None
    x = load("checks/CONTRACT_EXAMPLES.json")
    for f in x["fee_cases"]:
        assert fee(f["quantity"]) == Decimal(f["expected_posted_usd"])
    assert fee(10)+fee(5)+fee(5) == Decimal("10.28")
    assert fee(6)+fee(3)+fee(3) == Decimal("6.16")
    e = x["entry_case"]
    E,S,V = Decimal(e["entry"]),Decimal(e["stop"]),Decimal(e["micro_point_value"])
    R = abs(E-S)*e["quantity"]*V
    gross = ((Decimal(e["first_exit_price"])-E)*e["first_exit_quantity"] +
             (Decimal(e["final_exit_price"])-E)*e["final_exit_quantity"])*V
    costs = fee(e["quantity"])+fee(e["first_exit_quantity"])+fee(e["final_exit_quantity"])
    assert R == Decimal(e["expected_initial_risk_usd"])
    assert gross == Decimal(e["expected_gross_usd"])
    assert costs == Decimal(e["expected_fees_usd"])
    assert gross-costs == Decimal(e["expected_net_usd"])
    assert (gross-costs)/R == Decimal(e["expected_net_r"])
    c = x["continuation_case"]
    for p,delta,r in zip(c["hold_terminal_prices"],c["expected_delta_usd"],c["expected_delta_r"]):
        change = (Decimal(p)-Decimal(c["close_on_strictly_next_print"]))*c["remaining_quantity"]*Decimal(c["micro_point_value"])
        assert change == Decimal(delta)
        assert change/R == Decimal(r)
    for a in x["action_cases"]:
        outcome = "baseline_unavailable" if a["value"] is None else "change" if Decimal(a["value"])<0 else "baseline"
        assert outcome == a["action"]
    p=x["precision_case"]
    assert p["event_ns"]//1000 == p["receipt_us"]
    assert p["event_ns"]%1000 == p["representation_remainder_ns"]
    events=x["event_order_case"]
    first=events[0]
    later=[e for e in events if (e["ns"],e["event_ordinal"])>(first["ns"],first["event_ordinal"])]
    assert later[0]["event_ordinal"]==x["expected_close_event_ordinal"]
    # Only authored relative Markdown links: historical source documents can refer to their original project.
    for md in list(ROOT.glob("*.md"))+[ROOT/"references/INDEX.md"]:
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)",md.read_text(encoding="utf-8")):
            if target.startswith(("http:","https:","#")):
                continue
            p=(md.parent/target.split("#")[0]).resolve()
            if not p.exists() and p.suffix.lower() == ".zip" and p.is_relative_to(ROOT):
                p = input_path(str(p.relative_to(ROOT)))
            assert p.exists(), f"broken link: {md.name}: {target}"
    return {"status":"passed","scope":"handoff manifest, copied references, matrix, schedule, feature partitions, synthetic costs/labels/events, and authored links only",
            "model_cells":24,"funded_controls":2,"folds":9,"scored_dates":171,
            "input_files_verified":len(manifest["files"]),"application_tests_run":False,
            "models_fitted":0,"financial_operations_executed":0}

if __name__ == "__main__":
    try:
        print(json.dumps(verify(),indent=2))
    except Exception as exc:
        print(f"HANDOFF CHECK FAILED: {exc}", file=sys.stderr)
        raise
