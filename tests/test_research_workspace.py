"""Research work stays external before any worker can import or write."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location(
    "research_workspace", SCRIPTS / "research_workspace.py",
)
assert _SPEC is not None and _SPEC.loader is not None
workspace = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workspace)


def test_new_workspace_and_no_overwrite(tmp_path):
    repo = tmp_path / "Claude-Quant-Lab"
    repo.mkdir()
    destination = workspace.create_workspace("research-example-20261008", repo_root=repo)
    assert destination.parent == tmp_path / "Claude-Quant-Lab-Research-Artifacts"
    assert {p.name for p in destination.iterdir()} == {
        "inputs", "study_store", "work", "source", "validation", "README.md",
    }
    saved = destination / "work" / "previous-evidence.txt"
    saved.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        workspace.create_workspace("research-example-20261008", repo_root=repo)
    assert saved.read_text(encoding="utf-8") == "preserve me"
    assert "no study has been approved or run" in (destination / "README.md").read_text()
    assert not list(repo.iterdir())


@pytest.mark.parametrize("task_id", [
    "", "../escape", "a/b", r"a\b", "Upper", "two--hyphens", "trailing-", "con", "lpt1",
    "a" * 65,
])
def test_invalid_task_id_creates_nothing(tmp_path, task_id):
    with pytest.raises(ValueError, match="task ID"):
        workspace.create_workspace(task_id, repo_root=tmp_path / "repo")
    assert not list(tmp_path.iterdir())


def test_external_paths_resolve_before_check(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    for path in (repo, repo / "new" / "work", tmp_path / "outside" / ".." / "repo" / "work"):
        with pytest.raises(ValueError, match="outside the repository"):
            workspace.require_external_work_paths(repo, {"state": path})
    sibling = tmp_path / "repo-artifacts" / "task"
    assert workspace.require_external_work_paths(repo, {"work": sibling}) == {"work": sibling}


def test_external_link_back_into_repository_is_rejected(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    link = tmp_path / "linked-work"
    try:
        link.symlink_to(repo, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable on this host: {error}")
    with pytest.raises(ValueError, match="outside the repository"):
        workspace.require_external_work_paths(repo, {"work": link / "new-state"})


@pytest.mark.parametrize("marker_kind", ["directory", "worktree_file"])
def test_frozen_worker_rejects_original_git_checkout(tmp_path, marker_kind):
    original = tmp_path / "original-checkout"
    original.mkdir()
    marker = original / ".git"
    if marker_kind == "directory":
        marker.mkdir()
    else:
        marker.write_text("gitdir: ../main/.git/worktrees/original\n", encoding="utf-8")
    frozen = tmp_path / "research-artifacts" / "frozen-source"
    frozen.mkdir(parents=True)
    target = original / "reports" / "new-work"
    with pytest.raises(ValueError, match="outside the repository"):
        workspace.require_external_work_paths(frozen, {"work": target})
    assert not target.exists()
    external = tmp_path / "research-artifacts" / "working-store"
    assert workspace.require_external_work_paths(frozen, {"work": external}) == {"work": external}


@pytest.mark.parametrize(("script", "base_args", "guarded_flags"), [
    ("ifvg_full_range_batch_job.py", ["start", "--plan-id", "unused"],
     ["--store-root", "--state-root"]),
    ("ifvg_mffu_batch_job.py", ["prepare"],
     ["--store-root", "--state-root", "--staging-root"]),
    ("ifvg_ml_phase_job.py", ["--plan", "unused", "--stage", "fit"],
     ["--store", "--work"]),
])
def test_worker_paths_rejected_before_imports_or_writes(tmp_path, script, base_args, guarded_flags):
    repo = SCRIPTS.parent
    if script == "ifvg_ml_phase_job.py":
        fixed = {"--core": tmp_path / "missing-core", "--runtime": tmp_path / "missing-runtime"}
    else:
        fixed = {"--core-root": tmp_path / "missing-core"}
        if script == "ifvg_full_range_batch_job.py":
            fixed["--reports-root"] = repo / "reports" / "unused-test-report"
    for invalid_flag in guarded_flags:
        paths = {**fixed, **{flag: tmp_path / flag[2:] for flag in guarded_flags}}
        paths[invalid_flag] = repo / "reports" / "unused-test-work"
        args = [value for flag, path in paths.items() for value in (flag, str(path))]
        result = subprocess.run(
            [sys.executable, str(SCRIPTS / script), *base_args, *args],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2, result.stderr
        assert f"{invalid_flag} must be outside the repository" in result.stderr
        assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_status_can_read_historical_repository_store(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mffu_job", SCRIPTS / "ifvg_mffu_batch_job.py")
    assert spec is not None and spec.loader is not None
    job = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job)

    class ReachedReadOnlySourceSetupError(Exception):
        pass

    def source_setup(_):
        raise ReachedReadOnlySourceSetupError

    monkeypatch.setattr(job, "_source_paths", source_setup)
    monkeypatch.setattr(sys, "argv", [
        "ifvg_mffu_batch_job.py", "status", "--core-root", str(tmp_path),
        "--store-root", str(SCRIPTS.parent / "data"),
        "--state-root", str(SCRIPTS.parent / "reports"), "--plan-id", "historical",
    ])
    with pytest.raises(ReachedReadOnlySourceSetupError):
        job.main()


def test_frozen_ml_worker_binds_its_workspace_helper(tmp_path):
    from alpha_lab.propsim.funded.ml_phase.freeze import _copy_worker_sources
    from alpha_lab.propsim.funded.ml_phase.runtime import sha_file

    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("ifvg_ml_phase_job.py", "research_workspace.py"):
        (scripts / name).write_text(f"# synthetic {name}\n", encoding="utf-8")
    runtime = tmp_path / "runtime"
    bound = _copy_worker_sources(repo, runtime)
    assert set(bound) == {str(runtime / "scripts" / name) for name in (
        "ifvg_ml_phase_job.py", "research_workspace.py",
    )}
    for filename, expected in bound.items():
        assert expected == sha_file(Path(filename)) == sha_file(scripts / Path(filename).name)
