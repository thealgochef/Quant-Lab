"""Saved progress cannot silently turn into new replay work after source drift."""

from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.search.job_runtime import guard_resume_identities


@pytest.mark.parametrize("state", ["completed", "reused", "failed", "cancelled_at_safe_boundary"])
def test_resume_requires_original_identity_for_every_saved_configuration(state):
    spec = SimpleNamespace(axis_value_ids={"timeout": "timeout.240"})
    saved = {"children": [{"axis_value_ids": spec.axis_value_ids,
                           "core_replay_id": "a" * 64, "state": state}]}
    calls = []

    def resolve(child):
        calls.append(child)
        return SimpleNamespace(core_replay_id="a" * 64)

    guarded = guard_resume_identities(resolve, saved)
    assert guarded(spec).core_replay_id == "a" * 64
    assert calls == [spec]  # Fresh input/approval checks are never bypassed.
    changed = guard_resume_identities(lambda _: SimpleNamespace(core_replay_id="b" * 64), saved)
    with pytest.raises(RuntimeError, match="Saved progress is preserved"):
        changed(spec)
    assert saved["children"][0]["core_replay_id"] == "a" * 64


def test_empty_state_and_synthetic_identity_remain_supported():
    spec = SimpleNamespace(axis_value_ids={"timeout": "timeout.240"})
    assert guard_resume_identities(lambda _: "a" * 64, None)(spec) == "a" * 64


def test_resume_guard_does_not_mask_input_verification_failure():
    def resolve(_):
        raise PermissionError("Input bundle changed")

    with pytest.raises(PermissionError, match="Input bundle changed"):
        guard_resume_identities(resolve, None)(SimpleNamespace(axis_value_ids={}))
