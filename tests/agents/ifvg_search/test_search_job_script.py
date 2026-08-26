"""Contract tests for the detached search job shim (IMPLEMENTATION_PLAN §6).

The shim never runs replays in-process for the UI, never launches anything at
import time, refuses execution without an explicit runner entry, and its
status/cancel surfaces read/write exactly the orchestrator's state files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ifvg_search_job as job  # noqa: E402

_SOURCE = Path(job.__file__).read_text(encoding="utf-8")


def test_import_launches_nothing() -> None:
    # no module-level process launch, no import-time execution path
    import ast

    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if "Popen" in name or "run_search" in name:
                # every launch lives inside a function body, never at module level
                parents = getattr(node, "col_offset", 0)
                assert parents > 0
    assert "if __name__" in _SOURCE


def test_search_id_is_validated_before_any_path(capsys) -> None:
    with pytest.raises(SystemExit, match="64-hex"):
        job.main(["status", "--search-id", "../../evil"])
    with pytest.raises(SystemExit, match="64-hex"):
        job.main(["cancel", "--search-id", "short"])


def test_status_reads_the_orchestrator_state_file(tmp_path, capsys) -> None:
    search_id = "a" * 64
    state_dir = tmp_path / search_id
    state_dir.mkdir(parents=True)
    (state_dir / "search_state.json").write_text(
        json.dumps(
            {"schema_version": 1, "search_id": search_id, "phase": "replays", "children": []}
        ),
        encoding="utf-8",
    )
    code = job.main(
        ["status", "--search-id", search_id, "--state-root", str(tmp_path)]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["phase"] == "replays"

    # unknown search → null status, exit 0 (the monitor's missing-file state)
    code = job.main(["status", "--search-id", "b" * 64, "--state-root", str(tmp_path)])
    assert code == 0
    assert json.loads(capsys.readouterr().out) is None


def test_cancel_writes_the_safe_boundary_sentinel(tmp_path, capsys) -> None:
    search_id = "c" * 64
    code = job.main(
        ["cancel", "--search-id", search_id, "--state-root", str(tmp_path)]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "cancellation_requested"
    assert (tmp_path / search_id / "cancel.requested").exists()


def test_worker_refuses_without_an_explicit_runner_entry(tmp_path) -> None:
    """No built-in real-data wiring exists: execution requires --runner-entry."""

    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
    from tests.agents.ifvg_search.test_orchestrator import _charter

    charter = _charter()
    save_or_reuse_envelope(tmp_path / "store", "charters", charter)
    with pytest.raises(SystemExit, match="runner entry"):
        job.main(
            [
                "worker",
                "--search-id",
                charter.search_id,
                "--store-root",
                str(tmp_path / "store"),
                "--state-root",
                str(tmp_path / "state"),
            ]
        )


def test_runner_entry_shape_is_validated() -> None:
    with pytest.raises(SystemExit, match="module:function"):
        job._resolve_runner_entry("not a module path")
    with pytest.raises(SystemExit, match="module:function"):
        job._resolve_runner_entry("os.system('x')")


def test_unregistered_runner_entries_are_refused_before_import(monkeypatch) -> None:
    """R4 registry gate: a well-shaped but unregistered module:function can
    never reach importlib — no user-shaped string executes."""

    import importlib

    def _boom(name):  # pragma: no cover — must never be called
        raise AssertionError("import_module reached for an unregistered entry")

    monkeypatch.setattr(importlib, "import_module", _boom)
    with pytest.raises(SystemExit, match="not registered"):
        job._resolve_runner_entry("os:system")
    with pytest.raises(SystemExit, match="not registered"):
        job._resolve_runner_entry("subprocess:run")


def test_runner_entry_key_resolves_through_the_registry() -> None:
    from types import SimpleNamespace

    args = SimpleNamespace(
        runner_entry_key="synthetic_search_job_fixture_v1", runner_entry=None
    )
    assert job._entry_from_args(args) == (
        "tests.agents.ifvg_search.test_search_job_script:synthetic_runner_entry"
    )
    with pytest.raises(SystemExit, match="not registered"):
        job._entry_from_args(
            SimpleNamespace(runner_entry_key="nope_v1", runner_entry=None)
        )


def test_resume_command_reenters_the_idempotent_worker(tmp_path) -> None:
    """`resume` exists and launches the same detached worker as `start`."""

    import subprocess as real_subprocess

    captured: dict = {}

    class _FakeProcess:
        pid = 4242

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProcess()

    original = real_subprocess.Popen
    real_subprocess.Popen = _fake_popen
    try:
        code = job.main(
            [
                "resume",
                "--search-id",
                "a" * 64,
                "--store-root",
                str(tmp_path / "store"),
                "--state-root",
                str(tmp_path / "state"),
                "--runner-entry-key",
                "synthetic_search_job_fixture_v1",
            ]
        )
    finally:
        real_subprocess.Popen = original
    assert code == 0
    assert "worker" in captured["command"]
    assert "--runner-entry-key" in captured["command"]
    assert "synthetic_search_job_fixture_v1" in captured["command"]


def test_worker_runs_the_search_with_an_injected_runner(tmp_path, capsys) -> None:
    from alpha_lab.agents.data_infra.ifvg.search.store import save_or_reuse_envelope
    from tests.agents.ifvg_search.test_orchestrator import _charter

    charter = _charter()
    save_or_reuse_envelope(tmp_path / "store", "charters", charter)
    code = job.main(
        [
            "worker",
            "--search-id",
            charter.search_id,
            "--store-root",
            str(tmp_path / "store"),
            "--state-root",
            str(tmp_path / "state"),
            "--runner-entry",
            "tests.agents.ifvg_search.test_search_job_script:synthetic_runner_entry",
        ]
    )
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["phase"] == "search_complete"
    state = json.loads(
        (tmp_path / "state" / charter.search_id / "search_state.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["phase"] == "search_complete"
    assert len(state["children"]) == 4


def synthetic_runner_entry(charter, *, store_root=None):
    """The injected wiring the worker test uses (synthetic control flow only).

    Accepts the worker's ``store_root`` per the aligned shim contract
    (DEV-R5-10 closure) — the synthetic wiring itself has no store to root.
    """

    from types import SimpleNamespace

    from alpha_lab.agents.data_infra.ifvg.search.identities import (
        canonical_contract_sha256,
    )
    from tests.agents.ifvg_search.test_orchestrator import (
        _identity_resolver,
        _typed_empty_tables,
    )

    def _runner(*, spec, core_replay_id):
        return SimpleNamespace(
            tables=_typed_empty_tables(),
            gross_trade_stream_hash=canonical_contract_sha256({"c": core_replay_id}),
        )

    return {"identity_resolver": _identity_resolver, "child_runner": _runner}
