"""Verified completed phase readers. These never start or resume an execution."""

from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
from alpha_lab.propsim.funded.comparison_runner import find_approval
from alpha_lab.propsim.funded.full_range_batch import load_checkpoint

from .benchmark import OnlineScorer
from .plan import MlPhasePlanEnvelope, verify_phase_plan
from .protocol import digest
from .runtime import sha_file


def load_completed(*, store: Path, plan_id: str, work: Path):
    plan = load_verified_envelope(store, PLAN_STORE, plan_id, MlPhasePlanEnvelope)
    contracts = verify_phase_plan(plan.payload, imported=False)
    approval = find_approval(store, plan_id)
    if approval is None:
        raise PermissionError("completed phase has no exact-plan approval")
    reuse_path = Path(plan.payload.runtime_root) / "shadow_reuse.json"
    reuse = json.loads(reuse_path.read_text(encoding="utf-8")) if reuse_path.exists() else {
        "plan_id": plan_id, "work_root": str(work)}
    shadow_plan = load_verified_envelope(store, PLAN_STORE, reuse["plan_id"], MlPhasePlanEnvelope)
    verify_phase_plan(shadow_plan.payload, imported=False)
    shadow_approval = find_approval(store, reuse["plan_id"])
    if shadow_approval is None:
        raise PermissionError("label source has no approval")
    benchmark_path = work / "benchmark/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["identity"]["plan_id"] != plan_id:
        raise PermissionError("benchmark plan differs")
    original = contracts["DATE_AND_FOLD_PLAN"]
    all_dates = tuple(original["original_warmup_dates"] + original["original_evaluation_dates"])
    scored = tuple(original["scored_dates"])
    shadows, operations, seeds, files = {}, {}, {}, {str(benchmark_path): sha_file(benchmark_path)}
    for reference in ("MCB062", "MCB025"):
        shadow_dispatch = digest({"plan_id": reuse["plan_id"],
            "approval_id": shadow_approval.funded_comparison_approval_id,
            "reference": reference, "kind": "fixed_shadow"})
        folder = Path(reuse["work_root"]) / "shadows" / reference
        shadows[reference] = load_checkpoint(folder / "output.json",
            dispatch_sha256=shadow_dispatch, dates=all_dates)
        seeds[reference] = load_checkpoint(folder / "scored_seed.json",
            dispatch_sha256=shadow_dispatch, dates=all_dates)
        if not shadows[reference] or tuple(shadows[reference]["completed_dates"]) != all_dates:
            raise PermissionError("full shadow result is required")
        if not seeds[reference] or not seeds[reference]["no_position"]:
            raise PermissionError("verified flat scored seed is required")
        dispatch = digest({"plan_id": plan_id,
            "approval_id": approval.funded_comparison_approval_id,
            "reference": reference, "kind": "matched_policies",
            "seed_hash": seeds[reference]["seed_hash"], "benchmark": digest(benchmark)})
        path = work / "operations" / reference / "output.json"
        operation = load_checkpoint(path, dispatch_sha256=dispatch, dates=scored)
        if not operation or tuple(operation["completed_dates"]) != scored:
            raise PermissionError("all 171 operation dates must be completed")
        expected = {c["cell_id"] for c in contracts["MODEL_MATRIX"]["cells"]
                    if c["reference"] == reference} | {f"NO_ML_{reference}"}
        if set(operation["streams"]) != expected:
            raise PermissionError("operation membership differs from declared matrix")
        if not all(s["ledger"]["finished"] for s in operation["streams"].values()):
            raise PermissionError("operation ledger is not finalized")
        operations[reference] = operation
        for file in (path, folder / "output.json", folder / "scored_seed.json"):
            files[str(file)] = sha_file(file)
    datasets = {r: s["stream"]["datasets"] for r, s in shadows.items()}
    if digest(datasets) != benchmark["identity"]["datasets_sha256"]:
        raise PermissionError("benchmark datasets changed")
    for cell in contracts["MODEL_MATRIX"]["cells"]:
        OnlineScorer(cell, benchmark, work / "benchmark", contracts)
    return {"plan": plan, "approval": approval, "contracts": contracts,
            "benchmark": benchmark, "datasets": datasets, "shadows": shadows,
            "operations": operations, "seeds": seeds, "source_files": files}
