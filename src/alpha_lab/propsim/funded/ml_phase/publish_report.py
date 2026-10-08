"""Build the one saved report from verified terminal records; never launch work."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.task_b_execution import _calendar
from alpha_lab.propsim.funded.mffu_batch_plan import MffuBatchPlanEnvelope

from .benchmark import save_json
from .diagnostics import coverage, largest_continuations, occupancy_evidence, paired_diagnostics
from .protocol import digest
from .reporting import economic_report
from .runtime import sha_file
from .saved import load_completed


def build_saved_report(*, store: Path, plan_id: str, work: Path, destination: Path):
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection

    saved = load_completed(store=store, plan_id=plan_id, work=work)
    plan = saved["plan"]
    inherited = MffuBatchPlanEnvelope.model_validate_json(
        Path(plan.payload.reference_plan_path).read_text(encoding="utf-8")).payload
    section = IfvgSmcSection.model_validate_json(inherited.variants[24].effective_section_json)
    contracts, benchmark = saved["contracts"], saved["benchmark"]
    calendar, _, _ = _calendar(section, contracts["DATE_AND_FOLD_PLAN"]["scored_dates"])
    identity = {"plan_id": plan_id,
        "approval_id": saved["approval"].funded_comparison_approval_id,
        "label_source_plan_id": saved["datasets"]["MCB062"]["ENTRY"][0]["source"],
        "source_files": saved["source_files"],
        "original_result_id": plan.payload.reference_result_id,
        "benchmark_identity": benchmark["identity"],
        "reporting_source_files": {str(path): sha_file(path) for path in (
            Path(__file__), Path(__file__).with_name("reporting.py"),
            Path(__file__).with_name("diagnostics.py"), Path(__file__).with_name("saved.py"))},
        "reporting_schema": "ifsm_ml_phase_report_v1",
        "created_utc": datetime.now(UTC).isoformat()}
    economics = economic_report(operations=saved["operations"], contracts=contracts,
        calendar=calendar, profiles=inherited.firm_profiles, identity=identity)
    if not economics["financial_validation"]["passed"]:
        raise PermissionError("financial reconciliation did not pass")
    fit_states = [json.loads((work / "benchmark" / f["fit_path"] / "fit.json").read_text())
                  for f in benchmark["fits"] if f["model_id"]]
    contrasts = paired_diagnostics(benchmark, contracts["DATE_AND_FOLD_PLAN"]["scored_dates"])
    lines = [
        "This is a development-exposed historical study, with chronological predictions "
        "and fresh matched October accounts. It does not establish live performance.",
    ]
    for job, question in (("ENTRY", "Entry value"), ("CONTINUATION", "Continuation value")):
        ids = {c["cell_id"] for c in benchmark["cells"] if c["job"] == job}
        metrics = [r for r in benchmark["pooled_metrics"] if r["cell_id"] in ids]
        skills = [r["skill"] for r in metrics if r["skill"] is not None]
        lines.append(f"**{question}:** {sum(v > 0 for v in skills)} of {len(skills)} cells "
            f"improved squared error versus their training-only mean. Pooled skill ranges "
            f"from {min(skills):.4f} to {max(skills):.4f}.")
    policy_rows = [r for r in economics["lenses"] if r["job"] != "CONTROL"]
    lines.append(f"**Modeled cash:** {sum(r['net_cash_delta_cents'] > 0 for r in policy_rows)} "
        "of 24 learned policies earned more received cash after account costs than their own "
        "matched no-ML control. All alternatives remain separate operations.")
    for ref in ("MCB062", "MCB025"):
        control = next(r for r in economics["lenses"] if r["configuration_id"] == f"NO_ML_{ref}")
        lines.append(f"{ref} matched control: ${control['net_received_cash_cents']/100:,.2f} "
            f"net cash across 171 evaluated dates; {control['accounts_bought']} accounts bought.")
    lines.append("**Feature contribution:** inspect the paired F1−F0 and F2−F1 loss differences "
        "below. Negative differences favor the added bundle. Five-date block intervals preserve "
        "empty dates; they describe this reused development sample, not forecast cash uncertainty.")
    lines.append("Each fitted model assumes zero historical training latency at its activation. "
        "Seven rows in each label population have approximate price evidence and remain excluded "
        "from primary fitting. Missing optional values retain train-only handling; unavailable "
        "decisions preserve the baseline. No production settings were promoted.")
    operations = {ref: {**{k: value[k] for k in ("completed_dates", "seed_hash", "start_ns",
        "cutoff_ns", "stats", "access_audit", "elapsed_seconds")},
        "streams": {name: {**{k: state[k] for k in ("entry_rows", "checkpoint_rows", "ledger",
                     "day_records")}, "driver": {"ml_entries": state["driver"]["ml_entries"]}}
                    for name, state in value["streams"].items()}}
        for ref, value in saved["operations"].items()}
    body = {"identity": identity, "contracts": contracts, "benchmark": benchmark,
        "datasets": saved["datasets"], "fit_states": fit_states, "economics": economics,
        "operations": operations, "seeds": {r: {k: seed[k] for k in (
            "seed_hash", "first_scored_date", "no_account", "no_position", "completed_dates")}
            for r, seed in saved["seeds"].items()},
        "coverage": coverage(saved["datasets"], contracts["FEATURE_DEFINITIONS"]["rows"]),
        "paired_diagnostics": contrasts,
        "largest_continuations": largest_continuations(saved["datasets"], benchmark),
        "changed_occupancy_evidence": occupancy_evidence(saved["operations"]),
        "summary_markdown": "\n\n".join(lines)}
    report = {**body, "report_id": digest(body)}
    if destination.exists():
        raise FileExistsError("never overwrite a saved report")
    save_json(destination, report)
    return report
