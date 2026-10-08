"""Prepare the narrowly instrumented research Core without editing its source.

The new callback is after every existing admission guard, before the eligible
decision/fill. Its disabled path is byte-for-byte the original reducer logic.
The adapter is installed only on the phase driver's reducer instance.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

REDUCER = "src/strategy_core/strategies/ifvg_smc/reducer.py"
ANCHOR = """        if self._cfg.ifsm_context_policy_version is not None:
            quota_before = self._executions_by_day.get(bar.trading_day, 0)
"""
HOOK = """        # Optional phase-01 research callback: all ordinary guards have passed.
        research_decision = getattr(self, "_research_entry_decision", None)
        if not blocks and research_decision is not None:
            refusal = research_decision(
                setup=s, geometry=geometry, bar=bar, candidate_id=candidate_id,
            )
            if refusal is not None:
                if refusal != "ml_entry_negative_return":
                    raise ValueError("unsupported research admission action")
                blocks.append(refusal)

"""
REFUSAL = """                    "ifsm_overhead_gex", "daily_execution_cap",
                } for reason in blocks)"""
REFUSAL_WITH_ML = """                    "ifsm_overhead_gex", "daily_execution_cap",
                    "ml_entry_negative_return",
                } for reason in blocks)"""


def sha_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def prepare_research_core(source: Path, target: Path) -> dict:
    """Copy code only; never copy .git, market data, or a dependency tree."""
    source, target = source.resolve(strict=True), target.resolve()
    if target.exists() or target == source or target.is_relative_to(source):
        raise ValueError("research source destination must be new and separate")
    files = sorted(
        p for p in (source / "src").rglob("*") if p.is_file() and p.suffix in {".py", ".typed"}
    )
    original = (source / REDUCER).read_text(encoding="utf-8")
    if original.count(ANCHOR) != 1 or original.count(REFUSAL) != 1:
        raise ValueError("corrected reducer does not expose the reviewed insertion points")
    patched = original.replace(ANCHOR, HOOK + ANCHOR).replace(REFUSAL, REFUSAL_WITH_ML)
    compile(patched, REDUCER, "exec")
    for path in files:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    (target / REDUCER).write_text(patched, encoding="utf-8", newline="\n")
    receipt = {
        "schema": "ifsm_ml_research_core_overlay_v1",
        "source_root": str(source),
        "target_root": str(target),
        "changed_file": REDUCER,
        "source_files": {p.relative_to(source).as_posix(): sha_file(p) for p in files},
        "runtime_files": {
            p.relative_to(source).as_posix(): sha_file(target / p.relative_to(source))
            for p in files
        },
    }
    (target / "overlay.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
