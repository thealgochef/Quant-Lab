"""Study input preparation reuse, with temporary paths and synthetic artifacts only."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.day_artifacts import DayArtifacts, DaySeeds
from alpha_lab.agents.data_infra.ifvg.search import strategy_executor as executor
from alpha_lab.agents.data_infra.ifvg.search.charter import SearchCharterEnvelope
from alpha_lab.agents.data_infra.ifvg.search.orchestrator import enumerate_children
from tests.agents.ifvg_search.test_strategy_approval import approved  # noqa: F401


@pytest.fixture
def inputs(approved, monkeypatch):  # noqa: F811
    root, payload, _, _ = approved
    charter = SearchCharterEnvelope.from_payload(payload)
    state = SimpleNamespace(
        root=root / "synthetic_inputs",
        tag="",
        deny_day=None,
        fail_day=None,
        mutate_on_load=False,
        loaded=[],
        hashed=[],
        authorized=[],
        digests={},
        source_calls=[],
    )
    config_type = executor.IfvgCaptureConfig
    original_tag = config_type.artifacts_tag
    resolve_path = executor.DevelopmentReplayPolicy.resolve_source_path

    def config(**kwargs):
        return replace(config_type(**kwargs), data_dir=state.root)

    def authorize(policy, day, factory):
        if day == state.deny_day:
            raise PermissionError("fixture authorization revoked")
        path = resolve_path(policy, day, factory)
        assert path.is_relative_to(state.root)
        state.authorized.append((day, path))
        return path

    def digest(path):
        assert path.is_relative_to(state.root)
        state.hashed.append(path)
        return state.digests.get(path, "d" * 64)

    def artifacts(day, cfg, *, expected_seeds, access_policy):
        access_policy.authorize_date(day)
        state.loaded.append(day)
        index = payload.date_policy.replay_dates.index(day)
        previous = date.fromisoformat(payload.date_policy.replay_dates[index - 1])
        assert expected_seeds == (
            DaySeeds(previous, (100, 80), previous, (95, 85))
            if index
            else DaySeeds(None, None, None, None)
        )
        if day == state.fail_day:
            return None
        if state.mutate_on_load:
            state.digests[cfg.bars_path(day)] = "e" * 64
        return DayArtifacts(day, [], {}, expected_seeds, (100, 80), (95, 85), ())

    def source_identity(**kwargs):
        state.source_calls.append(kwargs)
        return "a" * 40, "a" * 64

    monkeypatch.setattr(executor, "IfvgCaptureConfig", config)
    monkeypatch.setattr(config_type, "artifacts_tag", lambda cfg: original_tag(cfg) + state.tag)
    monkeypatch.setattr(executor.DevelopmentReplayPolicy, "resolve_source_path", authorize)
    monkeypatch.setattr(executor, "load_day_artifacts", artifacts)
    monkeypatch.setattr(executor, "file_sha256", digest)
    monkeypatch.setattr(executor, "strategy_core_repository_root", lambda _: root / "core")
    monkeypatch.setattr(executor, "strategy_core_source_identity", source_identity)
    monkeypatch.setattr(executor, "quant_lab_replay_source_identity", lambda **_: "b" * 64)
    monkeypatch.setattr(
        executor, "run_child_replay", lambda **_: pytest.fail("no replay belongs in this test")
    )
    state.days = tuple(payload.date_policy.replay_dates)
    state.specs = enumerate_children(
        charter, store_root=root, identity_resolver=lambda spec: spec.resolved_section_config_hash
    )
    state.factory = lambda: executor.search_strategy_development_entry(charter, store_root=root)
    state.resolve = state.factory()["identity_resolver"]
    return state


def test_four_strategy_children_validate_one_chain_and_keep_fresh_hashes(inputs):
    identities = []
    count = len(inputs.days)
    for index, spec in enumerate(inputs.specs):
        before = len(inputs.hashed)
        identities.append(inputs.resolve(spec))
        assert len(inputs.hashed) - before == count * (4 if index == 0 else 2)
    assert inputs.loaded == list(inputs.days)
    assert len({item.core_replay_id for item in identities}) == 4
    assert len({item.payload.replay_input_bundle_id for item in identities}) == 1
    assert len(inputs.source_calls) == 4
    assert {day for day, _ in inputs.authorized} == set(inputs.days)


@pytest.mark.parametrize("kind", ["bars", "levels"])
def test_changed_cached_input_refuses_before_reusing_trust(inputs, kind):
    inputs.resolve(inputs.specs[0])
    marker = "ifvg_tbars_" if kind == "bars" else "ifvg_levels_"
    path = next(path for path in inputs.hashed if path.name.startswith(marker))
    inputs.digests[path] = "e" * 64
    with pytest.raises(PermissionError, match="day artifact changed during study preparation"):
        inputs.resolve(inputs.specs[1])
    assert inputs.loaded == list(inputs.days)


def test_cached_inputs_still_require_path_authorization(inputs):
    inputs.resolve(inputs.specs[0])
    before = len(inputs.hashed)
    inputs.deny_day = inputs.days[0]
    with pytest.raises(PermissionError, match="fixture authorization revoked"):
        inputs.resolve(inputs.specs[1])
    assert len(inputs.hashed) == before


@pytest.mark.parametrize("changed", ["tag", "root"])
def test_artifact_signature_or_location_changes_force_fresh_validation(inputs, changed):
    inputs.resolve(inputs.specs[0])
    if changed == "tag":
        inputs.tag = "_different"
    else:
        inputs.root = inputs.root / "relocated"
    inputs.resolve(inputs.specs[1])
    assert inputs.loaded == list(inputs.days) * 2


def test_failed_chain_is_not_cached(inputs):
    inputs.fail_day = inputs.days[0]
    with pytest.raises(PermissionError, match="trusted day artifacts are unavailable"):
        inputs.resolve(inputs.specs[0])
    inputs.fail_day = None
    inputs.resolve(inputs.specs[0])
    assert inputs.loaded == [inputs.days[0], *inputs.days]


def test_mutation_during_validation_refuses_and_does_not_cache(inputs):
    inputs.mutate_on_load = True
    with pytest.raises(PermissionError, match="day artifact changed during study preparation"):
        inputs.resolve(inputs.specs[0])
    inputs.mutate_on_load = False
    inputs.resolve(inputs.specs[0])
    assert inputs.loaded == list(inputs.days) * 2


def test_fresh_worker_revalidates_inputs_and_preserves_identity(inputs):
    original = inputs.resolve(inputs.specs[0])
    resumed = inputs.factory()["identity_resolver"](inputs.specs[0])
    assert resumed.core_replay_id == original.core_replay_id
    assert inputs.loaded == list(inputs.days) * 2
