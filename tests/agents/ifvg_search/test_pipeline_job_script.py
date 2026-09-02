"""Detached pipeline job-shim contract tests (R5; FUX-PIPE-003 backend half).

Mirrors ``test_search_job_script.py``: importing the shim launches nothing;
``status`` prints ``null`` for unknown ids; ``cancel`` writes the sentinel;
the worker executes only registry-resolved entries end-to-end; ``start``
spawns exactly one detached worker process.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO / "scripts"))

import ifvg_pipeline_job as job  # noqa: E402

from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter  # noqa: E402
from alpha_lab.agents.data_infra.ifvg.search.pipeline import (  # noqa: E402
    PipelineWiring,
    StageStatus,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (  # noqa: E402
    save_or_reuse_envelope,
)
from tests.agents.ifvg_search.pipeline_fixture import (  # noqa: E402
    build_pipeline_fixture,
)

_SOURCE = Path(job.__file__).read_text(encoding="utf-8")


def synthetic_pipeline_entry(charter, semantic, *, store_root) -> PipelineWiring:
    """The registered synthetic pipeline wiring (registry key
    ``pipeline_synthetic_fixture_v1``): deterministic fixture wiring for ANY
    synthetic-marker charter — a development-checkout convenience exactly
    like the R2 search fixture entry (DEV-R4-5). Real charters are refused;
    they resolve to the baseline-verification executor instead."""

    from alpha_lab.agents.data_infra.ifvg.search.authorization import (
        SyntheticAuthorizationMarker,
    )

    if not isinstance(
        charter.payload.owner_authorization, SyntheticAuthorizationMarker
    ):
        raise PermissionError(
            "the synthetic pipeline entry serves synthetic-marker charters only"
        )
    # R6.1: the fixture SHAPE derives from the frozen regime study request
    # (grain → panel seam; supervised classes → the frozen authority ids)
    request = getattr(semantic.payload, "regime_study", None)
    if request is None:
        fixture = build_pipeline_fixture(Path(store_root).parent)
    else:
        shape = ("panel" if request.is_panel else "candidate") + (
            "_supervised" if request.requires_supervision else ""
        )
        authority = (
            (
                str(request.regime_promotion_decision_id),
                str(request.owner_decision_artifact_id),
                str(request.required_capability_assessment_id),
            )
            if request.requires_supervision
            else None
        )
        fixture = build_pipeline_fixture(
            Path(store_root).parent, regime_study=shape, regime_authority=authority
        )
    return fixture["wiring"]


def test_import_launches_nothing() -> None:
    """AST proof: every launch/run call sits inside a function body (same
    scan as the search shim); exactly one detached Popen seam exists."""

    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if "Popen" in name or "run_pipeline" in name:
                assert getattr(node, "col_offset", 0) > 0
    assert "if __name__" in _SOURCE
    assert _SOURCE.count("subprocess.Popen") == 1  # the one detached seam


def test_status_prints_null_for_unknown_pipelines(tmp_path, capsys) -> None:
    code = job.main(
        [
            "status",
            "--pipeline-id",
            "c" * 64,
            "--state-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out) is None


def test_invalid_pipeline_id_is_refused(tmp_path) -> None:
    with pytest.raises(SystemExit, match="64-hex"):
        job.main(["status", "--pipeline-id", "..\\evil", "--state-root", str(tmp_path)])


def test_cancel_writes_the_sentinel(tmp_path, capsys) -> None:
    pipeline_id = "d" * 64
    code = job.main(
        ["cancel", "--pipeline-id", pipeline_id, "--state-root", str(tmp_path)]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "cancellation_requested"
    assert (tmp_path / pipeline_id / "cancel.requested").exists()


def test_worker_refuses_raw_unregistered_entries(tmp_path) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.runner_registry import (
        RunnerEntryError,
    )

    with pytest.raises(RunnerEntryError, match="not registered"):
        job.main(
            [
                "worker",
                "--pipeline-id",
                "e" * 64,
                "--state-root",
                str(tmp_path),
                "--runner-entry",
                "os:system",
            ]
        )


def test_worker_runs_the_registered_synthetic_pipeline(tmp_path, capsys) -> None:
    fixture = build_pipeline_fixture(tmp_path)
    store_root = fixture["store_root"]
    save_charter(store_root, fixture["charter"])
    save_or_reuse_envelope(store_root, "pipeline_specs", fixture["semantic"])
    code = job.main(
        [
            "worker",
            "--pipeline-id",
            fixture["semantic"].pipeline_semantic_id,
            "--store-root",
            str(store_root),
            "--state-root",
            str(fixture["state_root"]),
            "--runner-entry-key",
            "pipeline_synthetic_fixture_v1",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    statuses = payload["stage_statuses"]
    assert statuses["11_run_frozen_model_gated_replays"] == StageStatus.BLOCKED.value
    assert statuses["15_verify_and_publish"] == StageStatus.COMPLETED.value
    # HARDENING-BACKEND section 4.6: the attempt receipt states the sequential truth
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import read_pipeline_state

    state = read_pipeline_state(fixture["state_root"], fixture["semantic"].pipeline_semantic_id)
    assert state["attempts"][-1]["effective_workers"] == 1
    assert state["attempts"][-1]["execution_mode"] == "sequential_children_v1"
    # publication gates via the CLI fallback, then verification-scope
    # activation refusal — the verify-then-activate boundary end to end
    gates_code = job.main(
        [
            "publish-gates",
            "--pipeline-id",
            fixture["semantic"].pipeline_semantic_id,
            "--store-root",
            str(store_root),
            "--state-root",
            str(fixture["state_root"]),
        ]
    )
    assert gates_code == 0
    activate_code = job.main(
        [
            "activate",
            "--pipeline-id",
            fixture["semantic"].pipeline_semantic_id,
            "--store-root",
            str(store_root),
            "--state-root",
            str(fixture["state_root"]),
        ]
    )
    refusal = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert activate_code == 1
    assert refusal["status"] == "refused"
    assert "never activate" in refusal["reason"]


def test_start_spawns_exactly_one_detached_worker(tmp_path, capsys) -> None:
    spawned: list[list[str]] = []

    class _Process:
        pid = 4242

    original = subprocess.Popen
    try:
        subprocess.Popen = lambda command, **kwargs: (  # type: ignore[assignment]
            spawned.append(list(command)) or _Process()
        )
        code = job.main(
            [
                "start",
                "--pipeline-id",
                "f" * 64,
                "--state-root",
                str(tmp_path),
                "--runner-entry-key",
                "pipeline_synthetic_fixture_v1",
            ]
        )
    finally:
        subprocess.Popen = original  # type: ignore[assignment]
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "started", "pid": 4242, "pipeline_id": "f" * 64}
    assert len(spawned) == 1
    command = spawned[0]
    assert command[0] == sys.executable
    assert command[2] == "worker"
    assert "--runner-entry-key" in command


# ── HARDENING-BACKEND section 4.6 (F-20): sequential-execution truth at the shim ──


@pytest.mark.parametrize("command", ["start", "resume"])
def test_start_and_resume_refuse_parallelism_before_spawning(tmp_path, capsys, command) -> None:
    """A ``--max-workers`` above one is refused BEFORE any subprocess spawn or
    job directory creation, with the typed reason and exit code 2."""

    spawned: list[list[str]] = []
    original = subprocess.Popen
    try:
        subprocess.Popen = lambda cmd, **kwargs: spawned.append(list(cmd))  # type: ignore[assignment]
        code = job.main(
            [
                command,
                "--pipeline-id",
                "f" * 64,
                "--state-root",
                str(tmp_path / "state"),
                "--runner-entry-key",
                "pipeline_synthetic_fixture_v1",
                "--max-workers",
                "2",
            ]
        )
    finally:
        subprocess.Popen = original  # type: ignore[assignment]
    assert code == 2
    refusal = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert refusal["status"] == "refused"
    assert refusal["reason"] == "unsupported_worker_parallelism_v1"
    assert refusal["requested_workers"] == 2
    assert refusal["supported_child_workers"] == 1
    assert spawned == []
    assert not (tmp_path / "state").exists()


def test_worker_refuses_parallelism_before_any_store_access(tmp_path, capsys) -> None:
    """The worker refuses ``--max-workers 2`` before loading the spec or the
    charter: a nonexistent store root is never touched."""

    code = job.main(
        [
            "worker",
            "--pipeline-id",
            "f" * 64,
            "--store-root",
            str(tmp_path / "no-such-store"),
            "--state-root",
            str(tmp_path / "state"),
            "--runner-entry-key",
            "pipeline_synthetic_fixture_v1",
            "--max-workers",
            "2",
        ]
    )
    assert code == 2
    refusal = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert refusal["reason"] == "unsupported_worker_parallelism_v1"
    assert not (tmp_path / "no-such-store").exists()
    assert not (tmp_path / "state").exists()
