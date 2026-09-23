"""Process-local Core provenance never changes packages or another checkout."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.search import runtime_source


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
    assert runtime_source.strategy_core_repository_root(tmp_path / "quant_lab") == checkout


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
    return tmp_path / "quant_lab", installed, sibling, source


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


def test_unavailable_source_does_not_fall_back_to_outer_repository(tmp_path, monkeypatch):
    package = tmp_path / "site-packages" / "strategy_core"
    _module_tree(package)
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(
        runtime_source, "strategy_core", SimpleNamespace(__file__=package / "__init__.py")
    )
    with pytest.raises(RuntimeError, match="source checkout is unavailable"):
        runtime_source.strategy_core_repository_root(tmp_path / "quant_lab")


def test_preparation_reference_is_fixed_and_only_enabled_for_dedicated_ui(tmp_path, monkeypatch):
    monkeypatch.delenv("IFSM_RESEARCH_UI", raising=False)
    assert runtime_source.research_preparation_approval(tmp_path) is None
    monkeypatch.setenv("IFSM_RESEARCH_UI", "true")
    assert runtime_source.research_preparation_approval(tmp_path) is None
    monkeypatch.setenv("IFSM_RESEARCH_UI", "1")
    root, identity = runtime_source.research_preparation_approval(tmp_path)
    assert root == Path(tmp_path) / "data" / "ifvg_datasets" / "search" / "v1"
    assert identity == "18abdc711313c108a23c123134c713eafd093c7eda39f545ab9cb94cae0043e7"
