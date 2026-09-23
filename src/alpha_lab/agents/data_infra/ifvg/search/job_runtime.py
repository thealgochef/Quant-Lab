"""Version job-control repairs separately from immutable strategy replay sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ..preparation import _write_json_atomic
from .identities import canonical_contract_sha256, quant_lab_replay_source_identity

JOB_RUNTIME_SOURCE_PATHS = (
    "scripts/ifvg_search_job.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/job_runtime.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/orchestrator.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/saved_strategy_result.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/strategy_metrics.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/gates.py",
    "src/alpha_lab/agents/data_infra/ifvg/trade_stats.py",
    "src/alpha_lab/agents/data_infra/ifvg/artifact_io.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/store.py",
)


def record_job_runtime(repository_root: Path, state_root: Path, search_id: str) -> dict:
    """Publish the exact execution-control source without rewriting replay IDs."""
    payload = {
        "policy_id": "bounded_saved_result_resume_v1",
        "quant_lab_replay_source_identity": quant_lab_replay_source_identity(
            repository_root=repository_root
        ),
        "source_files": {
            path: hashlib.sha256((repository_root / path).read_bytes()).hexdigest()
            for path in JOB_RUNTIME_SOURCE_PATHS
        },
    }
    receipt = {"runtime_id": canonical_contract_sha256(payload), "payload": payload}
    destination = state_root / search_id / "job_runtimes" / f"{receipt['runtime_id']}.json"
    if destination.exists():
        if json.loads(destination.read_text(encoding="utf-8")) != receipt:
            raise RuntimeError("Saved job-runtime receipt does not match its identity")
    else:
        _write_json_atomic(destination, receipt)
    return receipt


def guard_resume_identities(resolver: Callable, previous_state: Mapping[str, Any] | None):
    """Refuse to rerun saved configurations under changed source/data identities.

    Fresh resolution still executes every existing input and authorization check.
    A mismatch needs a separate study; it cannot silently replace saved progress.
    """
    expected = {}
    for child in (previous_state or {}).get("children", []):
        if child.get("core_replay_id") and child.get("state") != "blocked":
            key = tuple(sorted(child["axis_value_ids"].items()))
            expected[key] = child["core_replay_id"]

    def resolve(spec):
        identity = resolver(spec)
        prior = expected.get(tuple(sorted(spec.axis_value_ids.items())))
        current = identity if isinstance(identity, str) else identity.core_replay_id
        if prior is not None and current != prior:
            raise RuntimeError(
                "Resume refused: this configuration's source, data, or replay identity "
                "differs from the saved study. Saved progress is preserved; use the "
                "original runtime and inputs or create a separate study."
            )
        return identity

    return resolve
