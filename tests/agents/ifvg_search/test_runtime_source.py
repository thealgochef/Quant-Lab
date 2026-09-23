"""Process-local Core provenance never changes packages or another checkout."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.search import runtime_source

_PIN = "1" * 40


def _project(repo, pin=_PIN):
    repo.mkdir(exist_ok=True)
    (repo / "pyproject.toml").write_text(
        '[project]\ndependencies = ["strategy-core @ '
        f'git+https://github.com/thealgochef/Strategy-Core.git@{pin}"]\n',
        encoding="utf-8",
    )


def _managed_path(repo, pin=_PIN):
    return repo.parent / "Claude-Quant-Lab-Research-Artifacts" / "ifsm-research-core" / pin


def _vcs_metadata(monkeypatch, *, installed_commit=_PIN, checkout_commit=_PIN):
    monkeypatch.setattr(runtime_source, "_git_head", lambda _root: checkout_commit)
    metadata = json.dumps({"vcs_info": {"commit_id": installed_commit}})
    monkeypatch.setattr(
        runtime_source, "distribution",
        lambda _name: SimpleNamespace(read_text=lambda _name: metadata),
    )


def _module_tree(package, *, newline="\n"):
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(f"VERSION = 1{newline}".encode())
    (package / "engine.py").write_bytes(f"def run():{newline}    return 1{newline}".encode())
    (package / "py.typed").write_text("", encoding="utf-8")


def test_imported_frozen_checkout_takes_precedence_over_sibling(tmp_path, monkeypatch):
    checkout = tmp_path / "frozen_core"
    package = checkout / "src" / "strategy_core"
    _module_tree(package)
    (checkout / ".git").write_text("gitdir: a-worktree-git-directory", encoding="utf-8")
    monkeypatch.setattr(
        runtime_source, "strategy_core", SimpleNamespace(__file__=package / "__init__.py")
    )
    repo = tmp_path / "quant_lab"
    _project(repo)
    managed = _managed_path(repo)
    managed.mkdir(parents=True)  # Broken managed state cannot replace an explicit import.
    assert runtime_source.strategy_core_repository_root(repo) == checkout


@pytest.fixture
def installed_pair(tmp_path, monkeypatch):
    installed = tmp_path / "site-packages" / "strategy_core"
    sibling = tmp_path / "Strategy-Core"
    source = sibling / "src" / "strategy_core"
    _module_tree(installed, newline="\r\n")
    _module_tree(source)
    (sibling / ".git").mkdir()
    monkeypatch.setattr(
        runtime_source, "strategy_core", SimpleNamespace(__file__=installed / "__init__.py")
    )
    repo = tmp_path / "quant_lab"
    _project(repo)
    return repo, installed, sibling, source


@pytest.fixture
def managed_pair(installed_pair, monkeypatch):
    repo, installed, sibling, sibling_source = installed_pair
    managed = _managed_path(repo)
    source = managed / "src" / "strategy_core"
    _module_tree(source)
    (managed / ".git").mkdir()
    _vcs_metadata(monkeypatch)
    return repo, installed, managed, source, sibling, sibling_source


def test_installed_source_equivalence_accepts_only_newline_representation(installed_pair):
    repo, _installed, sibling, _source = installed_pair
    assert runtime_source.strategy_core_repository_root(repo) == sibling


@pytest.mark.parametrize("change", ["different", "extra", "missing"])
def test_installed_mismatch_refuses_before_any_replay(installed_pair, change):
    repo, installed, _sibling, _source = installed_pair
    if change == "different":
        (installed / "engine.py").write_text("def run(): return 2", encoding="utf-8")
    elif change == "extra":
        (installed / "extra.py").write_text("RUN = 2", encoding="utf-8")
    else:
        (installed / "engine.py").unlink()
    with pytest.raises(RuntimeError, match="differs from its source checkout"):
        runtime_source.strategy_core_repository_root(repo)


def test_managed_current_pin_wins_over_stale_sibling(managed_pair):
    repo, _installed, managed, _source, _sibling, sibling_source = managed_pair
    (sibling_source / "engine.py").write_text("def run(): return -1", encoding="utf-8")
    assert runtime_source.strategy_core_repository_root(repo) == managed


@pytest.mark.parametrize("change", ["source", "missing_git", "missing_module"])
def test_corrupt_managed_checkout_never_falls_back_to_matching_sibling(managed_pair, change):
    repo, _installed, managed, source, _sibling, _sibling_source = managed_pair
    if change == "source":
        (source / "engine.py").write_text("def run(): return -1", encoding="utf-8")
    elif change == "missing_git":
        (managed / ".git").rmdir()
    else:
        (source / "engine.py").unlink()
    with pytest.raises(RuntimeError, match="prepare_ifsm_research_core.py"):
        runtime_source.strategy_core_repository_root(repo)


def test_managed_checkout_cannot_mask_changed_installed_code(managed_pair):
    repo, installed, _managed, _source, _sibling, _sibling_source = managed_pair
    (installed / "engine.py").write_text("def run(): return 99", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from its source checkout"):
        runtime_source.strategy_core_repository_root(repo)


@pytest.mark.parametrize(
    ("installed_commit", "checkout_commit", "error"),
    [
        ("2" * 40, _PIN, "pin differs from source checkout HEAD"),
        (_PIN, "2" * 40, "HEAD differs from the Quant-Lab dependency pin"),
        (None, _PIN, "lacks immutable VCS commit provenance"),
    ],
)
def test_managed_commit_proof_is_required_even_when_source_matches(
    managed_pair, monkeypatch, installed_commit, checkout_commit, error,
):
    repo, _installed, _managed, _source, _sibling, _sibling_source = managed_pair
    _vcs_metadata(monkeypatch, installed_commit=installed_commit, checkout_commit=checkout_commit)
    with pytest.raises(RuntimeError, match=error):
        runtime_source.strategy_core_repository_root(repo)


def test_invalid_current_pin_does_not_fall_back_to_sibling(installed_pair):
    repo, _installed, _sibling, _source = installed_pair
    _project(repo, pin="main")
    with pytest.raises(RuntimeError, match="one exact Git dependency pin"):
        runtime_source.strategy_core_repository_root(repo)


def test_unavailable_source_does_not_fall_back_to_outer_repository(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "strategy_core"
    _module_tree(package)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        runtime_source, "strategy_core", SimpleNamespace(__file__=package / "__init__.py")
    )
    repo = tmp_path / "quant_lab"
    _project(repo)
    with pytest.raises(RuntimeError, match="source checkout is unavailable"):
        runtime_source.strategy_core_repository_root(repo)


def test_preparation_reference_is_fixed_and_only_enabled_for_dedicated_ui(tmp_path, monkeypatch):
    monkeypatch.delenv("IFSM_RESEARCH_UI", raising=False)
    assert runtime_source.research_preparation_approval(tmp_path) is None
    monkeypatch.setenv("IFSM_RESEARCH_UI", "true")
    assert runtime_source.research_preparation_approval(tmp_path) is None
    monkeypatch.setenv("IFSM_RESEARCH_UI", "1")
    root, identity = runtime_source.research_preparation_approval(tmp_path)
    assert root == Path(tmp_path) / "data" / "ifvg_datasets" / "search" / "v1"
    assert identity == "18abdc711313c108a23c123134c713eafd093c7eda39f545ab9cb94cae0043e7"
