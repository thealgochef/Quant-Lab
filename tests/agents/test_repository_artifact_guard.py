"""Exercise repository artifact policy against real, isolated Git indexes."""

import subprocess
from pathlib import Path

import pytest
from scripts import check_tracked_artifacts as guard
from scripts import install_repo_hooks as hooks


@pytest.mark.parametrize("path", [
    "handoff.zip", "docs/task/INPUT.ZIP", "docs/task/source.tar.gz", "docs/run/data.csv",
    "docs/run/model.cbm", "docs/run/data.parquet", "models/example/metadata.json",
    "handoff/source/a.py", "data/ifvg_search_jobs/job.json", "ifvg_search_trades/trades.json",
    "docs/task/staging/config.json", "docs/task/extracted/README.md",
    "docs/task/source-review/README.md", "docs/task/src/package/core.py",
    "docs/task/tests/test_core.py", "other/Strategy-Core/README.md", "nested/.git/config",
    "dashboard-ui/package.json", "nested/RePoRtS/a.md", "node_modules/pkg/package.json",
    "docs/task/screenshots/a.png", "docs/task/evidence/after.png",
    "docs/task/evidence/nested/before.webp", "docs/task/mocks/images/screen.jpg",
])
def test_generated_paths_are_rejected(path):
    assert guard.path_violation(path)


@pytest.mark.parametrize("path", [
    "docs/task/contracts/plan.json", "docs/task/references/source.json",
    "docs/task/checks/validate_contract.py", "docs/architecture.svg", "docs/diagram.png",
    "docs/archive/windows/REPORT.md", "src/alpha_lab/agents/validation/tests/risk.py",
    "tests/agents/test_source_archive.py", "tests/agents/fixtures/synthetic.csv",
    "tests/agents/fixtures/source.zip", "tests/agents/fixtures/src/example.py",
    "research/core/ifsm-daily-close.bundle", "research/core/current.json",
    "docs/task/evidence/FINDINGS.md", "docs/authored/assets/diagram.png",
])
def test_code_contracts_and_small_reusable_fixtures_are_allowed(path):
    assert guard.path_violation(path) is None


def run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    run_git(tmp_path, "init", "-q")
    run_git(tmp_path, "config", "user.email", "artifact-tests@example.invalid")
    run_git(tmp_path, "config", "user.name", "Artifact guard test")
    # Do not execute real user/global hooks when committing synthetic fixtures.
    run_git(tmp_path, "config", "core.hooksPath", ".test-disabled-hooks")
    return tmp_path


def stage(root: Path, path: str, contents: bytes = b"synthetic") -> Path:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)
    run_git(root, "add", "-f", "--", path)
    return target


def test_staged_check_handles_initial_commit_spaces_and_deletions(repository):
    stage(repository, "docs/source contract.md")
    stage(repository, "docs/bad archive.zip")
    assert guard.index_violations(repository, staged=True) == [
        "docs/bad archive.zip: generated archive, model, data or database file"
    ]
    run_git(repository, "commit", "-qm", "synthetic baseline")
    run_git(repository, "rm", "--", "docs/bad archive.zip")
    stage(repository, "docs/new contract.md")
    assert guard.index_violations(repository, staged=True) == []
    assert guard.index_violations(repository) == []


def test_staged_check_uses_index_size_not_unstaged_worktree(repository, monkeypatch):
    monkeypatch.setattr(guard, "MAX_BLOB_BYTES", 1024)
    target = stage(repository, "docs/diagram.png", b"\0" * 1025)
    target.write_bytes(b"small unstaged replacement")
    assert "exceeds" in guard.index_violations(repository, staged=True)[0]
    run_git(repository, "add", "--", "docs/diagram.png")
    target.write_bytes(b"\0" * 1025)
    assert guard.index_violations(repository, staged=True) == []


def test_full_check_includes_unchanged_tracked_artifacts(repository):
    stage(repository, "docs/old.zip")
    run_git(repository, "commit", "-qm", "synthetic old artifact")
    stage(repository, "docs/current.md")
    assert guard.index_violations(repository, staged=True) == []
    assert "docs/old.zip" in guard.index_violations(repository)[0]


def test_size_limit_also_applies_to_fixture_archives(repository, monkeypatch):
    monkeypatch.setattr(guard, "MAX_BLOB_BYTES", 16)
    stage(repository, "tests/fixtures/example.zip", b"0" * 17)
    assert "exceeds" in guard.index_violations(repository)[0]


def test_embedded_git_repository_is_rejected(repository):
    stage(repository, "README.md")
    run_git(repository, "commit", "-qm", "synthetic baseline")
    commit = run_git(repository, "rev-parse", "HEAD")
    run_git(repository, "update-index", "--add", "--cacheinfo", f"160000,{commit},copy")
    assert guard.index_violations(repository, staged=True) == [
        "copy: embedded Git repository/submodule is not allowed"
    ]


def test_ignore_policy_keeps_archives_local_and_fixtures_addable(repository):
    (repository / ".gitignore").write_bytes((guard.ROOT / ".gitignore").read_bytes())
    for path in [
        "handoff.zip", "docs/task/source.zip", "handoff/source.py", "docs/task/src/a.py",
        "docs/task/screenshots/a.png", "docs/task/evidence/b.png", "docs/task/mocks/images/c.png",
        "docs/task/evidence/nested/d.png", "docs/task/mocks/images/nested/e.png",
    ]:
        result = subprocess.run(["git", "check-ignore", "--", path], cwd=repository)
        assert result.returncode == 0
    for path in [
        "tests/example/fixtures/a.zip", "research/core/ifsm-daily-close.bundle",
        "docs/diagram.png", "docs/task/evidence/FINDINGS.md",
    ]:
        result = subprocess.run(["git", "check-ignore", "--", path], cwd=repository)
        assert result.returncode == 1


def test_hook_installer_retains_custom_hooks(repository):
    with pytest.raises(SystemExit, match="Existing core.hooksPath retained"):
        hooks.install(repository)
    assert run_git(repository, "config", "--get", "core.hooksPath") == ".test-disabled-hooks"


def test_hook_installer_is_idempotent_and_retains_existing_hook(repository):
    run_git(repository, "config", "--unset", "core.hooksPath")
    existing = repository / ".git" / "hooks" / "pre-commit"
    existing.write_text("# existing local hook\n")
    with pytest.raises(SystemExit, match="Existing Git hooks retained"):
        hooks.install(repository)
    existing.unlink()
    hook = repository / ".githooks" / "pre-commit"
    hook.parent.mkdir()
    hook.write_bytes((guard.ROOT / ".githooks" / "pre-commit").read_bytes())
    hooks.install(repository)
    hooks.install(repository)
    assert run_git(repository, "config", "--get", "core.hooksPath") == ".githooks"


def test_hook_installer_does_not_disable_existing_pre_push_hook(repository):
    run_git(repository, "config", "--unset", "core.hooksPath")
    existing = repository / ".git" / "hooks" / "pre-push"
    existing.write_text("# existing local hook\n")
    with pytest.raises(SystemExit, match="Existing Git hooks retained"):
        hooks.install(repository)
    assert existing.read_text() == "# existing local hook\n"


def test_installed_hook_blocks_forced_archive_add_and_allows_code_commit(repository):
    run_git(repository, "config", "--unset", "core.hooksPath")
    stage(repository, ".githooks/pre-commit", (guard.ROOT / ".githooks/pre-commit").read_bytes())
    stage(repository, "scripts/check_tracked_artifacts.py", Path(guard.__file__).read_bytes())
    hooks.install(repository)
    stage(repository, "docs/accidental.zip")
    attempted = subprocess.run(
        ["git", "commit", "-qm", "blocked archive"],
        cwd=repository, capture_output=True, text=True,
    )
    assert attempted.returncode != 0
    assert "docs/accidental.zip" in attempted.stdout + attempted.stderr
    run_git(repository, "rm", "--cached", "--", "docs/accidental.zip")
    run_git(repository, "commit", "-qm", "valid source")
