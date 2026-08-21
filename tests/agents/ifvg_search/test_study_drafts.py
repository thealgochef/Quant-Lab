"""Draft persistence: autosave, exact restore, freeze immutability, clone
(FUX-WIZ-002/003; FUX §29 draft policy)."""

from __future__ import annotations

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    STEP_KEYS,
    DraftError,
    DraftFrozenError,
    DraftNotFoundError,
    clone_draft,
    discard_draft,
    list_drafts,
    load_draft,
    mark_frozen,
    new_draft,
    save_draft,
)


def _clock():
    ticks = iter(range(1000))

    def _now() -> str:
        return f"2026-08-21T00:00:{next(ticks):02d}+00:00"

    return _now


def test_wizard_steps_are_the_eight_step_keys() -> None:
    assert STEP_KEYS == (
        "objective",
        "baseline",
        "search_space",
        "prop_contracts",
        "risk_policies",
        "benchmarks",
        "validation",
        "review",
    )


def test_save_load_restores_the_exact_step_and_fields(tmp_path) -> None:
    now = _clock()
    draft = new_draft("fsm_config_search", now_fn=now)
    draft.step_index = 2
    draft.step_payload("objective")["template_id"] = "payout_reliability"
    draft.step_payload("search_space")["axis_selections"] = {
        "parent_retest_timeout_1m_bars": ["parent_retest_timeout_1m_bars.240"]
    }
    save_draft(tmp_path, draft, now_fn=now)
    restored = load_draft(tmp_path, draft.draft_id)
    assert restored.step_index == 2
    assert restored.mode_id == "fsm_config_search"
    assert restored.steps["objective"]["template_id"] == "payout_reliability"
    assert restored.steps["search_space"]["axis_selections"] == {
        "parent_retest_timeout_1m_bars": ["parent_retest_timeout_1m_bars.240"]
    }
    with pytest.raises(DraftError, match="unknown wizard step"):
        restored.step_payload("nope")


def test_writes_are_atomic_and_listing_orders_by_update(tmp_path) -> None:
    now = _clock()
    first = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, first, now_fn=now)
    second = new_draft("prop_benchmark", now_fn=now)
    save_draft(tmp_path, second, now_fn=now)
    save_draft(tmp_path, first, now_fn=now)  # first becomes newest
    listed = list_drafts(tmp_path)
    assert [draft.draft_id for draft in listed] == [
        first.draft_id,
        second.draft_id,
    ]
    assert not list(tmp_path.glob("*/.draft.json.tmp-*"))  # no torn temp files
    # an unreadable draft never breaks the listing
    broken = tmp_path / "brokendraft"
    broken.mkdir()
    (broken / "draft.json").write_text("{not json", encoding="utf-8")
    assert len(list_drafts(tmp_path)) == 2


def test_frozen_drafts_refuse_mutation_and_discard(tmp_path) -> None:
    now = _clock()
    draft = new_draft("fsm_config_search", now_fn=now)
    save_draft(tmp_path, draft, now_fn=now)
    mark_frozen(tmp_path, draft, search_id="f" * 64, now_fn=now)
    stored = json.loads(
        (tmp_path / draft.draft_id / "draft.json").read_text(encoding="utf-8")
    )
    assert stored["status"] == "frozen"
    assert stored["frozen_search_id"] == "f" * 64
    with pytest.raises(DraftFrozenError, match="never a plain save"):
        save_draft(tmp_path, draft, now_fn=now)  # in-memory frozen object
    with pytest.raises(DraftFrozenError, match="cannot be discarded"):
        discard_draft(tmp_path, draft.draft_id)
    with pytest.raises(DraftFrozenError, match="already frozen"):
        mark_frozen(tmp_path, draft, search_id="e" * 64, now_fn=now)
    # the STORED frozen status also refuses a racing save from a stale
    # in-memory copy that still believes it is a draft (TOCTOU closure)
    stale = new_draft("fsm_config_search", now_fn=now)
    stale.draft_id = draft.draft_id
    with pytest.raises(DraftFrozenError, match="Clone as New Search"):
        save_draft(tmp_path, stale, now_fn=now)
    # and a second freezer racing the same id refuses on the stored record
    racer = new_draft("fsm_config_search", now_fn=now)
    racer.draft_id = draft.draft_id
    with pytest.raises(DraftFrozenError, match="already frozen"):
        mark_frozen(tmp_path, racer, search_id="d" * 64, now_fn=now)
    # a draft object cannot smuggle a frozen status through plain save either
    fresh = new_draft("fsm_config_search", now_fn=now)
    fresh.status = "frozen"
    with pytest.raises(DraftFrozenError, match="mark_frozen"):
        save_draft(tmp_path, fresh, now_fn=now)


def test_clone_deep_copies_and_leaves_the_original_untouched(tmp_path) -> None:
    now = _clock()
    draft = new_draft("universal_prop_search", now_fn=now)
    draft.step_payload("risk_policies")["per_firm_policies"] = {
        "c" * 64: {"risk_template": "Fixed Dollar", "n_accounts": 2}
    }
    save_draft(tmp_path, draft, now_fn=now)
    mark_frozen(tmp_path, draft, search_id="a" * 64, now_fn=now)
    clone = clone_draft(draft, now_fn=now)
    assert clone.draft_id != draft.draft_id
    assert clone.status == "draft"
    assert clone.frozen_search_id is None
    assert clone.cloned_from == draft.draft_id
    clone.steps["risk_policies"]["per_firm_policies"]["c" * 64][
        "n_accounts"
    ] = 1
    assert (
        draft.steps["risk_policies"]["per_firm_policies"]["c" * 64]["n_accounts"]
        == 2
    )  # deep copy — the frozen original is untouched (FUX-WIZ-003)
    save_draft(tmp_path, clone, now_fn=now)
    frozen_on_disk = load_draft(tmp_path, draft.draft_id)
    assert frozen_on_disk.status == "frozen"


def test_discard_removes_only_mutable_drafts(tmp_path) -> None:
    now = _clock()
    draft = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, draft, now_fn=now)
    discard_draft(tmp_path, draft.draft_id)
    with pytest.raises(DraftNotFoundError):
        load_draft(tmp_path, draft.draft_id)
    for hostile in ("../escape", "C:x", "CON", "a" * 31, "A" * 32, ""):
        with pytest.raises(DraftError, match="32-hex identifier"):
            load_draft(tmp_path, hostile)
    # a hand-edited file whose embedded id disagrees with its directory is
    # refused rather than trusted (no foreign-path save can follow)
    import json as _json

    victim = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, victim, now_fn=now)
    record = tmp_path / victim.draft_id / "draft.json"
    payload = _json.loads(record.read_text(encoding="utf-8"))
    payload["draft_id"] = "f" * 32
    record.write_text(_json.dumps(payload), encoding="utf-8")
    with pytest.raises(DraftError, match="does not match its directory"):
        load_draft(tmp_path, victim.draft_id)
