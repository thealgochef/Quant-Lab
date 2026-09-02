"""HARDENING-BACKEND §4.4 (F-17) — the event-detail writer is stream-bounded.

The writer consumes an ITERABLE of ``(path_record, walk_result)`` pairs in
draw-ordinal order exactly once (a generator is never materialized into an
all-path event list), holds one path block of rows at a time, keeps no
whole-artifact event-id index (the ``np.concatenate`` spike is gone), and
proves event-id uniqueness by the canonical key (unique path ids × strictly
increasing per-path ordinals, the projection ``AccountWalk._emit`` hashes)
plus a disk-backed DuckDB distinct check over the written partitions under
an explicit memory limit and an attempt-local temp directory — the check
that catches a FORGED duplicate id across path blocks. The search bridge
feeds the writer through a generator over the simulation run.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

import pytest

from alpha_lab.propsim import event_detail as event_detail_module
from alpha_lab.propsim import search_bridge as search_bridge_module
from alpha_lab.propsim.calendar import BOOTSTRAP_CLOCK_POLICY
from alpha_lab.propsim.event_detail import (
    EVENT_DETAIL_BUDGET_V1,
    EVENT_DETAIL_MANIFEST_SIDECAR,
    EventDetailBudget,
    EventDetailBudgetError,
    EventDetailIntegrityError,
    build_account_event_detail,
)
from alpha_lab.propsim.prop_metrics import build_payout_reliability_vector
from alpha_lab.propsim.search_bridge import persist_account_simulation
from tests.propsim.test_account_event_detail import _bootstrap_payload, _run

_ORDER = "prop_account_event_order_v1"


class _OneShot:
    """An iterable that can be iterated exactly once and counts the pulls."""

    def __init__(self, pairs):
        self._pairs = pairs
        self.iterations = 0
        self.pulled = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("the writer iterated the walk source twice")
        for pair in self._pairs:
            self.pulled += 1
            yield pair


def _pairs(run):
    order = sorted(range(len(run.path_records)), key=lambda i: run.path_records[i].draw_ordinal)
    return [(run.path_records[i], run.walk_results[i]) for i in order]


def _rows(run) -> int:
    return sum(len(result.events) for result in run.walk_results)


def _fresh(base: Path, name: str) -> Path:
    directory = base / name
    directory.mkdir(parents=True)
    return directory


def _stream_build(run, directory, *, budget=EVENT_DETAIL_BUDGET_V1, **overrides):
    kwargs = dict(
        clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
        budget=budget,
        event_order_policy_id=_ORDER,
        directory=directory,
        total_rows=_rows(run),
        path_count=len(run.path_records),
    )
    kwargs.update(overrides)
    source = kwargs.pop("source", None)
    return build_account_event_detail(source if source is not None else iter(_pairs(run)), **kwargs)


def test_iterable_pairs_are_consumed_exactly_once_and_never_materialized(tmp_path):
    run = _run(_bootstrap_payload(600))
    source = _OneShot(_pairs(run))
    bundle = _stream_build(run, _fresh(tmp_path, "stream"), source=source)
    assert source.iterations == 1 and source.pulled == 600
    assert bundle.partition_count == 3 and bundle.total_rows == _rows(run)
    # the writer keeps NO whole-artifact id index (the R6.1 np.concatenate spike)
    assert not hasattr(event_detail_module, "_event_id_index")
    assert not hasattr(event_detail_module, "_assert_unique")


def test_streaming_form_reproduces_the_sequence_form_byte_for_byte(tmp_path):
    run = _run(_bootstrap_payload(600))
    legacy = build_account_event_detail(
        run.walk_results,
        run.path_records,
        clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
        budget=EVENT_DETAIL_BUDGET_V1,
        event_order_policy_id=_ORDER,
        directory=_fresh(tmp_path, "legacy"),
    )
    streamed = _stream_build(run, _fresh(tmp_path, "streamed"))
    assert legacy.manifest == streamed.manifest
    for record in legacy.produced():
        assert (legacy.directory / record.name).read_bytes() == (
            streamed.directory / record.name
        ).read_bytes()
    uniqueness = streamed.manifest["event_id_uniqueness"]
    assert uniqueness["canonical_key"] == ["path_instance_id", "event_ordinal"]
    assert uniqueness["artifact_check"] == event_detail_module.EVENT_ID_UNIQUENESS_CHECK_V1


def test_iterable_form_requires_and_verifies_the_preflight_counts(tmp_path):
    run = _run(_bootstrap_payload(8))
    with pytest.raises(ValueError, match="total_rows"):
        build_account_event_detail(
            iter(_pairs(run)),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=EVENT_DETAIL_BUDGET_V1,
            event_order_policy_id=_ORDER,
            directory=_fresh(tmp_path, "no_counts"),
        )
    with pytest.raises(EventDetailIntegrityError, match="declared total_rows"):
        _stream_build(run, _fresh(tmp_path, "wrong_rows"), total_rows=_rows(run) + 1)
    with pytest.raises(EventDetailIntegrityError, match="declared path_count"):
        _stream_build(run, _fresh(tmp_path, "wrong_paths"), path_count=7)
    tiny = EventDetailBudget(
        budget_id="tiny_rows",
        max_event_detail_rows=5,
        max_published_bytes=EVENT_DETAIL_BUDGET_V1.max_published_bytes,
        path_block_size=250,
    )
    # the declared count is the preflight; nothing is written
    directory = _fresh(tmp_path, "preflight")
    with pytest.raises(EventDetailBudgetError, match="preflight"):
        _stream_build(run, directory, budget=tiny)
    assert os.listdir(directory) == []
    # an under-declared count streams past the budget: refused at the block, no manifest
    directory = _fresh(tmp_path, "row_overrun")
    with pytest.raises(EventDetailBudgetError, match="row overrun"):
        _stream_build(run, directory, budget=tiny, total_rows=5)
    assert not (directory / EVENT_DETAIL_MANIFEST_SIDECAR).exists()


def test_iterable_form_requires_draw_ordinal_order(tmp_path):
    run = _run(_bootstrap_payload(4))
    pairs = _pairs(run)
    shuffled = [pairs[1], pairs[0], *pairs[2:]]
    with pytest.raises(EventDetailIntegrityError, match="draw-ordinal order"):
        _stream_build(run, _fresh(tmp_path, "unordered"), source=iter(shuffled))
    doubled = [pairs[0], pairs[0], *pairs[1:]]
    with pytest.raises(EventDetailIntegrityError, match="draw-ordinal order"):
        _stream_build(run, _fresh(tmp_path, "doubled"), source=iter(doubled), path_count=5)


def test_external_check_catches_a_forged_duplicate_across_blocks_without_a_global_index(
    tmp_path, monkeypatch
):
    run = _run(_bootstrap_payload(3))
    first, second = run.walk_results[0], run.walk_results[1]
    forged = second.events[0].model_copy(update={"event_id": first.events[0].event_id})
    replaced = dataclasses.replace(second, events=(forged,) + second.events[1:])
    one_path_blocks = EventDetailBudget(
        budget_id="one_path_blocks",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V1.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V1.max_published_bytes,
        path_block_size=1,
    )
    calls: list[dict] = []
    real_check = event_detail_module._external_uniqueness_check

    def _spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real_check(*args, **kwargs)

    monkeypatch.setattr(event_detail_module, "_external_uniqueness_check", _spy)
    records = run.path_records
    with pytest.raises(EventDetailIntegrityError, match="duplicate event id across path blocks"):
        build_account_event_detail(
            (first, replaced),
            records[:2],
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=one_path_blocks,
            event_order_policy_id=_ORDER,
            directory=_fresh(tmp_path, "forged"),
        )
    assert len(calls) == 1
    # with the external check disabled NOTHING in memory catches the cross-block
    # forgery — the proof lives on disk, never in a whole-artifact index
    monkeypatch.setattr(
        event_detail_module, "_external_uniqueness_check", lambda *a, **k: {"checked": False}
    )
    bundle = build_account_event_detail(
        (first, replaced),
        records[:2],
        clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
        budget=one_path_blocks,
        event_order_policy_id=_ORDER,
        directory=_fresh(tmp_path, "unchecked"),
    )
    assert bundle.partition_count == 2


def test_external_check_refuses_a_path_id_reused_under_another_ordinal(tmp_path):
    run = _run(_bootstrap_payload(3))
    records = run.path_records
    shared_path = records[0].path_instance_id
    reused = records[1].model_copy(update={"path_instance_id": shared_path})
    second = run.walk_results[1]
    second = dataclasses.replace(
        second,
        events=tuple(
            event.model_copy(update={"path_instance_id": shared_path}) for event in second.events
        ),
    )
    one_path_blocks = EventDetailBudget(
        budget_id="one_path_blocks",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V1.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V1.max_published_bytes,
        path_block_size=1,
    )
    pairs = [(records[0], run.walk_results[0]), (reused, second)]
    with pytest.raises(EventDetailIntegrityError, match="one-to-one"):
        build_account_event_detail(
            iter(pairs),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=one_path_blocks,
            event_order_policy_id=_ORDER,
            directory=_fresh(tmp_path, "reused_path"),
            total_rows=len(run.walk_results[0].events) + len(second.events),
            path_count=2,
        )


def test_external_check_temp_directory_is_attempt_local_and_cleaned(tmp_path, monkeypatch):
    run = _run(_bootstrap_payload(6))
    created: list[Path] = []
    real_mkdtemp = event_detail_module.tempfile.mkdtemp

    def _record(*args, **kwargs):
        path = real_mkdtemp(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(event_detail_module.tempfile, "mkdtemp", _record)
    directory = _fresh(tmp_path, "cleanup")
    bundle = _stream_build(run, directory)
    assert created and not any(path.exists() for path in created)
    # the publication directory holds ONLY the declared files (no scratch)
    assert set(os.listdir(directory)) == {record.name for record in bundle.produced()}
    created.clear()
    first, second = run.walk_results[0], run.walk_results[1]
    forged = second.events[0].model_copy(update={"event_id": first.events[0].event_id})
    replaced = dataclasses.replace(second, events=(forged,) + second.events[1:])
    one_path_blocks = EventDetailBudget(
        budget_id="one_path_blocks",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V1.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V1.max_published_bytes,
        path_block_size=1,
    )
    with pytest.raises(EventDetailIntegrityError):
        build_account_event_detail(
            (first, replaced),
            run.path_records[:2],
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=one_path_blocks,
            event_order_policy_id=_ORDER,
            directory=_fresh(tmp_path, "cleanup_refused"),
        )
    assert created and not any(path.exists() for path in created)


def test_search_bridge_streams_pairs_and_builds_no_second_all_path_list(tmp_path, monkeypatch):
    run = _run(_bootstrap_payload(20))
    seen: dict = {}
    real_build = search_bridge_module.build_account_event_detail

    def _spy(walks, path_records=None, **kwargs):
        seen["is_sequence"] = isinstance(walks, list | tuple)
        seen["path_records"] = path_records
        seen["total_rows"] = kwargs.get("total_rows")
        seen["path_count"] = kwargs.get("path_count")
        return real_build(walks, path_records, **kwargs)

    monkeypatch.setattr(search_bridge_module, "build_account_event_detail", _spy)
    root = tmp_path / "store"
    persist_account_simulation(
        root,
        run,
        vector=build_payout_reliability_vector(run.walk_results),
        risk_policy_label="fixture",
    )
    assert seen["is_sequence"] is False and seen["path_records"] is None
    assert seen["total_rows"] == _rows(run) and seen["path_count"] == 20
    manifest = json.loads(
        (
            tmp_path / "store" / "account_simulations" / run.envelope.account_simulation_id
            / EVENT_DETAIL_MANIFEST_SIDECAR
        ).read_text(encoding="utf-8")
    )
    assert manifest["path_count"] == 20 and manifest["total_rows"] == _rows(run)
    shutil.rmtree(root, ignore_errors=True)


def test_iterable_form_refuses_a_path_count_overrun_while_streaming(tmp_path):
    """B-08: an ordered stream LONGER than the declared ``path_count`` is refused
    at the first extra path — before the final count check, before any flush of
    the trailing block and before any manifest exists."""

    run = _run(_bootstrap_payload(4))
    pairs = _pairs(run)
    directory = _fresh(tmp_path, "path_overrun")
    with pytest.raises(EventDetailIntegrityError, match="exceeded while streaming"):
        _stream_build(
            run,
            directory,
            source=iter(pairs),
            path_count=len(pairs) - 1,
            total_rows=_rows(run),
        )
    # four paths share one path block: the overrun fired before the block
    # was flushed, so the publication directory holds no partition and no
    # detail manifest
    assert sorted(os.listdir(directory)) == []
