"""Concrete plan preparation. Approval remains the existing separate owner API."""

from __future__ import annotations

import importlib
import io
import json
import shutil
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .plan import MlPhasePlanEnvelope, MlPhasePlanPayload, save_phase_plan
from .protocol import load_contracts
from .runtime import sha_file


def _copy_worker_sources(repo: Path, runtime: Path) -> dict[str, str]:
    bound = {}
    for name in ("ifvg_ml_phase_job.py", "research_workspace.py"):
        target = runtime / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / "scripts" / name, target)
        bound[str(target)] = sha_file(target)
    return bound


def prepare_plan(*, repo: Path, task_root: Path, revision: str, owner_request: str,
                 shadow_revision: str | None = None):
    repo, task_root = repo.resolve(strict=True), task_root.resolve(strict=True)
    runtime = task_root / f"execution-{revision}"
    core = task_root / "core"
    if runtime.exists():
        raise FileExistsError("never overwrite a frozen phase runtime")
    if not (core / "overlay.json").is_file():
        raise ValueError("prepare and test the separate research Core first")
    packet = repo / "docs/ifsm-mffu-ml-phase-v01"
    contracts = load_contracts(packet)
    bound = {}
    for path in sorted((repo / "src").rglob("*")):
        if not path.is_file() or path.suffix not in {".py", ".typed"}:
            continue
        target = runtime / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        bound[str(target)] = sha_file(target)
    bound.update(_copy_worker_sources(repo, runtime))
    if shadow_revision:
        from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope
        from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
        from alpha_lab.propsim.funded.comparison_runner import find_approval

        from .plan import verify_phase_plan

        prior = json.loads((task_root / f"prepared-{shadow_revision}.json").read_text())
        old = load_verified_envelope(task_root / "study_store", PLAN_STORE,
                                     prior["plan_id"], MlPhasePlanEnvelope)
        verify_phase_plan(old.payload, imported=False)
        approval = find_approval(task_root / "study_store", prior["plan_id"])
        if approval is None:
            raise PermissionError("shadow reuse requires its own exact source approval")
        # Only prediction validation and orchestration changed; every derivation file is equal.
        permitted = {"benchmark.py", "freeze.py", "runner.py", "diagnostics.py"}
        for path in (runtime / "src").rglob("*.py"):
            relative = path.relative_to(runtime)
            if path.parent.name == "ml_phase" and path.name in permitted:
                continue
            if sha_file(path) != sha_file(Path(old.payload.runtime_root) / relative):
                raise PermissionError(f"shadow derivation changed: {relative}")
        files = {}
        for ref in ("MCB062", "MCB025"):
            for name in ("output.json", "scored_seed.json"):
                path = Path(prior["work_root"]) / "shadows" / ref / name
                files[str(path)] = sha_file(path)
                bound[str(path)] = files[str(path)]
        receipt = runtime / "shadow_reuse.json"
        receipt.write_text(json.dumps({"plan_id": prior["plan_id"],
            "work_root": prior["work_root"], "files": files,
            "reason": "Prediction guards only; shadow derivation source byte-identical"},
            indent=2), encoding="utf-8")
        bound[str(receipt)] = sha_file(receipt)
    overlay = json.loads((core / "overlay.json").read_text(encoding="utf-8"))
    for relative, expected in overlay["runtime_files"].items():
        actual = sha_file(core / relative)
        if actual != expected:
            raise PermissionError("research Core differs from its tested overlay")
        bound[str(core / relative)] = actual
    bound[str(core / "overlay.json")] = sha_file(core / "overlay.json")
    for path in sorted((packet / "contracts").glob("*.json")):
        bound[str(path)] = sha_file(path)
    reference = task_root.parent / "ifsm-mffu-repair-integration-v01/study_store"
    bindings = contracts["REFERENCE_BINDINGS"]
    plan = reference / "funded_comparison_plans" / bindings["reference_plan_id"] / "envelope.json"
    result = (
        reference / "funded_comparison_results" / bindings["economic_result_id"] / "result.json"
    )
    libraries, imports = {}, {}
    for name in ("numpy", "pandas", "pyarrow", "sklearn", "scipy", "catboost", "pydantic"):
        module = importlib.import_module(name)
        libraries[name] = module.__version__
        imports[name] = str(Path(module.__file__).resolve())
        bound[imports[name]] = sha_file(Path(imports[name]))
    import numpy as np

    configuration = io.StringIO()
    with redirect_stdout(configuration):
        np.show_config()
    environment = {
        "python": sys.version,
        "executable": sys.executable,
        "libraries": libraries,
        "imports": imports,
        "numpy_configuration": configuration.getvalue(),
    }
    envelope = MlPhasePlanEnvelope.from_payload(
        MlPhasePlanPayload(
            contracts_json=json.dumps(contracts, sort_keys=True),
            reference_plan_path=str(plan),
            reference_plan_sha256=sha_file(plan),
            reference_result_id=bindings["economic_result_id"],
            reference_result_path=str(result),
            reference_result_sha256=sha_file(result),
            core_root=str(core),
            runtime_root=str(runtime),
            bound_files=tuple(sorted(bound.items())),
            environment_json=json.dumps(environment, sort_keys=True),
            owner_request=owner_request,
        )
    )
    plan_id = save_phase_plan(task_root / "study_store", envelope)
    receipt = {
        "plan_id": plan_id,
        "runtime_root": str(runtime),
        "core_root": str(core),
        "store_root": str(task_root / "study_store"),
        "work_root": str(task_root / f"run-{revision}"),
        "approval_id": None,
        "bound_files": len(bound),
        "status": "saved_plan_requires_owner_approval",
    }
    (task_root / f"prepared-{revision}.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt
