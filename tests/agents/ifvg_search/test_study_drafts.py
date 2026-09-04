"""Draft persistence: autosave, exact restore, freeze immutability, clone
(FUX-WIZ-002/003; FUX §29 draft policy) — and the UI-2 lifecycle (owner Q2):
archive / restore as the normal reversible actions, permanent delete only for
never-frozen archived drafts with the exact typed name, the retired
``discard_draft``, the one-time bulk archive of the empty untitled drafts,
duplicate detection and the proposed draft name."""

from __future__ import annotations

import json

import pytest

from alpha_lab.agents.data_infra.ifvg.study_drafts import (
    DRAFT_SCHEMA_VERSION,
    STEP_KEYS,
    DraftError,
    DraftFrozenError,
    DraftNotFoundError,
    archive_draft,
    bulk_archive_empty_untitled_drafts,
    clone_draft,
    delete_draft_permanently,
    discard_draft,
    find_duplicate_drafts,
    is_empty_untitled_draft,
    list_drafts,
    load_draft,
    mark_frozen,
    new_draft,
    proposed_draft_name,
    restore_draft,
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
    draft.current_step_key = "search_space"
    draft.step_payload("objective")["template_id"] = "payout_reliability"
    draft.step_payload("search_space")["axis_selections"] = {
        "parent_retest_timeout_1m_bars": ["parent_retest_timeout_1m_bars.240"]
    }
    save_draft(tmp_path, draft, now_fn=now)
    restored = load_draft(tmp_path, draft.draft_id)
    assert restored.step_index == 2
    assert restored.current_step_key == "search_space"
    assert restored.mode_id == "fsm_config_search"
    assert restored.schema_version == DRAFT_SCHEMA_VERSION == 2
    assert restored.archived is False and restored.archived_at_utc is None
    assert restored.steps["objective"]["template_id"] == "payout_reliability"
    assert restored.steps["search_space"]["axis_selections"] == {
        "parent_retest_timeout_1m_bars": ["parent_retest_timeout_1m_bars.240"]
    }
    with pytest.raises(DraftError, match="unknown wizard step"):
        restored.step_payload("nope")


def test_legacy_schema_one_records_load_unchanged(tmp_path) -> None:
    """A v1 draft file (no archive / step-key fields) loads with the UI-2
    defaults; nothing is rewritten until the next save."""

    now = _clock()
    draft = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, draft, now_fn=now)
    path = tmp_path / draft.draft_id / "draft.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for key in ("archived", "archived_at_utc", "current_step_key"):
        record.pop(key, None)
    record["schema_version"] = 1
    path.write_text(json.dumps(record), encoding="utf-8")
    loaded = load_draft(tmp_path, draft.draft_id)
    assert loaded.archived is False
    assert loaded.current_step_key is None
    assert loaded.schema_version == 1
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


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


def test_frozen_drafts_refuse_mutation_archive_and_delete(tmp_path) -> None:
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
    with pytest.raises(DraftFrozenError, match="historical provenance"):
        archive_draft(tmp_path, draft.draft_id, now_fn=now)
    with pytest.raises(DraftFrozenError, match="never deletable"):
        delete_draft_permanently(tmp_path, draft.draft_id, confirm_name=draft.display_name)
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
    # the frozen record is still on disk — nothing above deleted it
    assert load_draft(tmp_path, draft.draft_id).status == "frozen"


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
    assert clone.archived is False  # a clone of an archived draft is live
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


def test_draft_ids_are_validated_and_hand_edited_identities_refused(tmp_path) -> None:
    now = _clock()
    for hostile in ("../escape", "C:x", "CON", "a" * 31, "A" * 32, ""):
        with pytest.raises(DraftError, match="32-hex identifier"):
            load_draft(tmp_path, hostile)
    # a hand-edited file whose embedded id disagrees with its directory is
    # refused rather than trusted (no foreign-path save can follow)
    victim = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, victim, now_fn=now)
    record = tmp_path / victim.draft_id / "draft.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["draft_id"] = "f" * 32
    record.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DraftError, match="does not match its directory"):
        load_draft(tmp_path, victim.draft_id)


# ── UI-2 lifecycle (owner Q2) ────────────────────────────────────────────────


def test_archive_hides_restores_and_never_deletes(tmp_path) -> None:
    now = _clock()
    draft = new_draft("fsm_config_search", display_name="Archive me", now_fn=now)
    draft.step_payload("objective")["template_id"] = "strategy_quality_only"
    save_draft(tmp_path, draft, now_fn=now)
    archived = archive_draft(tmp_path, draft.draft_id, now_fn=now)
    assert archived.archived is True and archived.archived_at_utc
    assert archived.steps["objective"]["template_id"] == "strategy_quality_only"  # kept
    assert [d.draft_id for d in list_drafts(tmp_path)] == []  # hidden by default
    assert [d.draft_id for d in list_drafts(tmp_path, include_archived=True)] == [
        draft.draft_id
    ]
    assert (tmp_path / draft.draft_id / "draft.json").exists()  # archive is not delete
    with pytest.raises(DraftError, match="already archived"):
        archive_draft(tmp_path, draft.draft_id, now_fn=now)
    restored = restore_draft(tmp_path, draft.draft_id, now_fn=now)
    assert restored.archived is False and restored.archived_at_utc is None
    assert [d.draft_id for d in list_drafts(tmp_path)] == [draft.draft_id]
    with pytest.raises(DraftError, match="not archived"):
        restore_draft(tmp_path, draft.draft_id, now_fn=now)
    with pytest.raises(DraftNotFoundError):
        archive_draft(tmp_path, "0" * 32, now_fn=now)


def test_permanent_delete_requires_archived_never_frozen_and_the_exact_name(tmp_path) -> None:
    now = _clock()
    draft = new_draft("single_configuration", display_name="Delete me", now_fn=now)
    save_draft(tmp_path, draft, now_fn=now)
    # a live (unarchived) draft cannot be deleted — archive is the normal action
    with pytest.raises(DraftError, match="archive it first"):
        delete_draft_permanently(tmp_path, draft.draft_id, confirm_name="Delete me")
    archive_draft(tmp_path, draft.draft_id, now_fn=now)
    # the exact display name is required (case and whitespace exact)
    for wrong in ("", "delete me", "Delete me ", "Delete", "Delete me!"):
        with pytest.raises(DraftError, match="exact draft name"):
            delete_draft_permanently(tmp_path, draft.draft_id, confirm_name=wrong)
    assert (tmp_path / draft.draft_id / "draft.json").exists()
    delete_draft_permanently(tmp_path, draft.draft_id, confirm_name="Delete me")
    assert not (tmp_path / draft.draft_id).exists()
    with pytest.raises(DraftNotFoundError):
        load_draft(tmp_path, draft.draft_id)
    # a draft that was ever frozen is provenance and never deletable, even
    # if its record were hand-edited to look archived and unfrozen
    frozen = new_draft("single_configuration", display_name="Frozen", now_fn=now)
    save_draft(tmp_path, frozen, now_fn=now)
    mark_frozen(tmp_path, frozen, search_id="a" * 64, now_fn=now)
    with pytest.raises(DraftFrozenError, match="never deletable"):
        delete_draft_permanently(tmp_path, frozen.draft_id, confirm_name="Frozen")
    record = tmp_path / frozen.draft_id / "draft.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["status"] = "draft"
    payload["archived"] = True
    record.write_text(json.dumps(payload), encoding="utf-8")  # frozen_search_id remains
    with pytest.raises(DraftFrozenError, match="never deletable"):
        delete_draft_permanently(tmp_path, frozen.draft_id, confirm_name="Frozen")
    assert record.exists()


def test_discard_draft_is_retired_and_deletes_nothing(tmp_path) -> None:
    now = _clock()
    draft = new_draft("single_configuration", now_fn=now)
    save_draft(tmp_path, draft, now_fn=now)
    with pytest.raises(DraftError, match="retired"):
        discard_draft(tmp_path, draft.draft_id)
    assert load_draft(tmp_path, draft.draft_id).status == "draft"


def test_bulk_archive_of_empty_untitled_drafts_preserves_everything_else(tmp_path) -> None:
    """Owner Q2: a one-time bulk archive for the existing empty step-0
    'Untitled study' files — never a delete; named, advanced, frozen or
    already-archived drafts are untouched."""

    now = _clock()
    empties = [new_draft(mode, now_fn=now) for mode in ("fsm_config_search", "prop_benchmark")]
    for draft in empties:
        save_draft(tmp_path, draft, now_fn=now)
        assert is_empty_untitled_draft(draft)
    named = new_draft("fsm_config_search", display_name="test_run (2026-09-01)", now_fn=now)
    save_draft(tmp_path, named, now_fn=now)
    advanced = new_draft("fsm_config_search", now_fn=now)  # untitled but with content
    advanced.step_index = 3
    advanced.step_payload("objective")["template_id"] = "custom"
    save_draft(tmp_path, advanced, now_fn=now)
    annotated = new_draft("fsm_config_search", now_fn=now)  # untitled but purposeful
    annotated.purpose_annotation = {"purpose": "development_research"}
    save_draft(tmp_path, annotated, now_fn=now)
    frozen = new_draft("fsm_config_search", now_fn=now)
    save_draft(tmp_path, frozen, now_fn=now)
    mark_frozen(tmp_path, frozen, search_id="b" * 64, now_fn=now)
    already = new_draft("fsm_config_search", now_fn=now)
    save_draft(tmp_path, already, now_fn=now)
    archive_draft(tmp_path, already.draft_id, now_fn=now)
    for draft in (named, advanced, annotated):
        assert not is_empty_untitled_draft(draft)
    archived_ids = bulk_archive_empty_untitled_drafts(tmp_path, now_fn=now)
    assert sorted(archived_ids) == sorted(draft.draft_id for draft in empties)
    live = {draft.draft_id for draft in list_drafts(tmp_path)}
    # frozen provenance is listed (never archived by the bulk action)
    assert live == {named.draft_id, advanced.draft_id, annotated.draft_id, frozen.draft_id}
    everything = {draft.draft_id for draft in list_drafts(tmp_path, include_archived=True)}
    assert everything == live | {draft.draft_id for draft in empties} | {already.draft_id}
    assert all((tmp_path / draft.draft_id / "draft.json").exists() for draft in empties)
    assert load_draft(tmp_path, frozen.draft_id).status == "frozen"
    # idempotent: a second pass archives nothing
    assert bulk_archive_empty_untitled_drafts(tmp_path, now_fn=now) == ()


def test_duplicate_detection_and_the_proposed_name(tmp_path) -> None:
    now = _clock()
    existing = new_draft("single_configuration", display_name="Compare A", now_fn=now)
    existing.purpose_annotation = {"purpose": "development_research"}
    existing.steps["objective"] = {"question_id": "compare_one_with_baseline"}
    existing.steps["baseline"] = {"baseline_profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
    save_draft(tmp_path, existing, now_fn=now)
    other = new_draft("fsm_config_search", display_name="Search B", now_fn=now)
    other.purpose_annotation = {"purpose": "development_research"}
    other.steps["objective"] = {"question_id": "find_robust_fsm"}
    save_draft(tmp_path, other, now_fn=now)
    drafts = list_drafts(tmp_path)
    duplicates = find_duplicate_drafts(
        drafts,
        mode_id="single_configuration",
        question_id="compare_one_with_baseline",
        purpose="development_research",
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
    )
    assert [draft.draft_id for draft in duplicates] == [existing.draft_id]
    assert not find_duplicate_drafts(
        drafts,
        mode_id="single_configuration",
        question_id="compare_one_with_baseline",
        purpose="implementation_verification",
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
    )
    # the draft itself is never its own duplicate
    assert not find_duplicate_drafts(
        drafts,
        mode_id="single_configuration",
        question_id="compare_one_with_baseline",
        purpose="development_research",
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        exclude_draft_id=existing.draft_id,
    )
    assert proposed_draft_name(
        "Compare one configuration with the baseline",
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        day="2026-09-04",
    ) == "Compare one configuration with the baseline — doc default fresh static 1r — 2026-09-04"
    assert proposed_draft_name(
        "Search FSM parameters", baseline_profile_name=None, day="2026-09-04"
    ) == ("Search FSM parameters — 2026-09-04")
