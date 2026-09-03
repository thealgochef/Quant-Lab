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

import pyarrow.parquet as pq
import pytest

from alpha_lab.propsim import event_detail as event_detail_module
from alpha_lab.propsim import search_bridge as search_bridge_module
from alpha_lab.propsim.calendar import BOOTSTRAP_CLOCK_POLICY
from alpha_lab.propsim.event_detail import (
    EVENT_DETAIL_BUDGET_V2,
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


def _stream_build(run, directory, *, budget=EVENT_DETAIL_BUDGET_V2, **overrides):
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
        budget=EVENT_DETAIL_BUDGET_V2,
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
            budget=EVENT_DETAIL_BUDGET_V2,
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
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=250,
        max_rows_per_partition=EVENT_DETAIL_BUDGET_V2.max_rows_per_partition,
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
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V2.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=1,
        max_rows_per_partition=EVENT_DETAIL_BUDGET_V2.max_rows_per_partition,
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
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V2.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=1,
        max_rows_per_partition=EVENT_DETAIL_BUDGET_V2.max_rows_per_partition,
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
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V2.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=1,
        max_rows_per_partition=EVENT_DETAIL_BUDGET_V2.max_rows_per_partition,
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


# ── HARDENING-BACKEND-FIX §9 — the hard per-partition resident-row bound ─────


def _bounded(rows_per_partition: int, *, path_block_size: int = 250) -> EventDetailBudget:
    return EventDetailBudget(
        budget_id=f"bounded_{rows_per_partition}",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V2.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=path_block_size,
        max_rows_per_partition=rows_per_partition,
    )


def _stretch(result, *, length: int):
    """A walk result whose event stream is ``length`` events long (the first
    event repeated with strictly increasing ordinals and unique ids)."""

    import hashlib

    base = result.events[0]
    events = tuple(
        base.model_copy(
            update={
                "event_ordinal": ordinal,
                "event_id": hashlib.sha256(
                    f"{base.path_instance_id}:{ordinal}".encode()
                ).hexdigest(),
            }
        )
        for ordinal in range(length)
    )
    return dataclasses.replace(result, events=events)


def _event_keys(pairs) -> list[tuple[int, int, str]]:
    keys = []
    for record, result in pairs:
        for event in result.events:
            keys.append((int(record.draw_ordinal), int(event.event_ordinal), str(event.event_id)))
    return keys


def _persisted_frames(tmp_path, run):
    root = tmp_path / "store"
    persist_account_simulation(
        root,
        run,
        vector=build_payout_reliability_vector(run.walk_results),
        risk_policy_label="fixture",
    )
    from alpha_lab.propsim.event_detail import (
        load_account_event_detail,
        load_account_event_detail_manifest,
    )

    simulation_id = run.envelope.account_simulation_id
    frames = list(load_account_event_detail(root, simulation_id))
    return root, load_account_event_detail_manifest(root, simulation_id), frames


def test_v1_budget_is_loadable_but_refused_by_the_row_bounded_writer(tmp_path):
    from alpha_lab.propsim.event_detail import EVENT_DETAIL_BUDGET_V1, event_detail_identity_fields

    run = _run(_bootstrap_payload(4))
    assert EVENT_DETAIL_BUDGET_V1.max_rows_per_partition is None
    # the pre-bound budget serializes exactly as before (no new key), so every
    # identity minted under it is preserved; the V2 budget carries the bound
    assert "max_rows_per_partition" not in EVENT_DETAIL_BUDGET_V1.model_dump(mode="json")
    assert EVENT_DETAIL_BUDGET_V2.model_dump(mode="json")["max_rows_per_partition"] == 50_000
    assert (
        EVENT_DETAIL_BUDGET_V2.max_event_detail_rows == EVENT_DETAIL_BUDGET_V1.max_event_detail_rows
    )
    assert EVENT_DETAIL_BUDGET_V2.max_published_bytes == EVENT_DETAIL_BUDGET_V1.max_published_bytes
    assert event_detail_identity_fields("account_event_detail_by_path_parquet_v2")[
        "event_detail_budget"
    ] == EVENT_DETAIL_BUDGET_V2
    directory = _fresh(tmp_path, "v1")
    with pytest.raises(EventDetailBudgetError, match="max_rows_per_partition"):
        _stream_build(run, directory, budget=EVENT_DETAIL_BUDGET_V1)
    assert os.listdir(directory) == []


def test_one_path_larger_than_one_partition_is_split_and_reconstructed_exactly(tmp_path):
    """HB-FIX-11: a single path longer than the bound is split across
    partitions of the same path block in strict event order; the reader
    reconstructs the exact stream; no partition exceeds the bound."""

    run = _run(_bootstrap_payload(3))
    pairs = _pairs(run)
    pairs[1] = (pairs[1][0], _stretch(pairs[1][1], length=23))
    budget = _bounded(7)
    total = sum(len(result.events) for _record, result in pairs)
    directory = _fresh(tmp_path, "split")
    bundle = build_account_event_detail(
        iter(pairs),
        clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
        budget=budget,
        event_order_policy_id=_ORDER,
        directory=directory,
        total_rows=total,
        path_count=3,
    )
    assert bundle.max_partition_rows <= 7
    assert bundle.partition_count >= 4  # 3 paths, one of 23 events, bound 7
    entries = bundle.manifest["partitions"]
    assert all(entry["rows"] <= 7 for entry in entries)
    assert bundle.manifest["max_partition_rows_written"] == bundle.max_partition_rows
    assert bundle.manifest["max_rows_per_partition"] == 7
    assert bundle.manifest["partition_bound_policy_id"] == "event_detail_partition_row_bound_v2"
    # every partition of the one path block is keyed by its ordinal within the block
    assert [entry["path_block_id"] for entry in entries] == [0] * len(entries)
    assert [entry["partition_ordinal"] for entry in entries] == list(range(len(entries)))
    assert [entry["name"] for entry in entries] == [
        f"event_detail_block_000000_{index:03d}.parquet" for index in range(len(entries))
    ]
    # the long path spans several partitions — split mid-path, never dropped
    spanning = [entry for entry in entries if entry["path_ordinal_min"] == 1]
    assert len(spanning) >= 3
    keys = [tuple(entry["first_event_key"]) for entry in entries]
    assert keys == sorted(keys)
    # exact reconstruction from the written partitions, in partition order
    read_back = []
    for entry in entries:
        table = pq.read_table(directory / entry["name"])
        for path_ordinal, event_ordinal, event_id in zip(
            table["path_ordinal"].to_pylist(),
            table["event_ordinal"].to_pylist(),
            table["event_id"].to_pylist(),
            strict=True,
        ):
            read_back.append((int(path_ordinal), int(event_ordinal), str(event_id)))
    assert read_back == _event_keys(pairs)


def test_one_path_block_and_skewed_shapes_obey_the_bound_and_reconstruct(tmp_path):
    """HB-FIX-11: a path block (250 paths) whose rows exceed the bound many
    times, and a highly skewed shape (one whale path beside tiny paths), both
    write partitions ≤ the bound and read back as the exact stream through
    the store and the production reader."""

    from alpha_lab.propsim.event_detail import load_account_event_detail

    run = _run(_bootstrap_payload(12))
    # (a) one block of 12 paths under a 5-row bound: many partitions, one block
    budget = _bounded(5)
    dense = _run(_bootstrap_payload(12, event_detail_budget=budget))
    root, manifest, frames = _persisted_frames(tmp_path / "dense", dense)
    entries = manifest["partitions"]
    assert {entry["path_block_id"] for entry in entries} == {0}
    assert max(entry["rows"] for entry in entries) <= 5
    assert len(entries) >= _rows(dense) // 5
    assert manifest["max_partition_rows_written"] <= 5
    reconstructed = [
        (int(p), int(e), str(i))
        for frame in frames
        for p, e, i in zip(
            frame["path_ordinal"], frame["event_ordinal"], frame["event_id"], strict=True
        )
    ]
    assert reconstructed == _event_keys(_pairs(dense))
    # (b) skewed: one whale path of 41 events beside 11 tiny paths, bound 6
    pairs = _pairs(run)
    pairs[4] = (pairs[4][0], _stretch(pairs[4][1], length=41))
    skewed_budget = _bounded(6)
    total = sum(len(result.events) for _record, result in pairs)
    directory = _fresh(tmp_path, "skewed")
    bundle = build_account_event_detail(
        iter(pairs),
        clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
        budget=skewed_budget,
        event_order_policy_id=_ORDER,
        directory=directory,
        total_rows=total,
        path_count=12,
    )
    assert bundle.max_partition_rows <= 6
    whale = [e for e in bundle.manifest["partitions"] if e["path_ordinal_min"] == 4]
    assert len(whale) >= 6
    read_back = []
    for entry in bundle.manifest["partitions"]:
        table = pq.read_table(directory / entry["name"])
        read_back.extend(
            (int(p), int(e), str(i))
            for p, e, i in zip(
                table["path_ordinal"].to_pylist(),
                table["event_ordinal"].to_pylist(),
                table["event_id"].to_pylist(),
                strict=True,
            )
        )
    assert read_back == _event_keys(pairs)
    # the reader refuses a partition that exceeds the manifest's declared bound
    manifest_rows = manifest["partitions"][0]["rows"]
    assert manifest_rows <= manifest["max_rows_per_partition"]
    assert list(load_account_event_detail(root, dense.envelope.account_simulation_id))


def test_partitioning_is_independent_of_iterator_chunking_and_repeat_is_byte_identical(tmp_path):
    """HB-FIX-11: the partition boundaries, names and manifest depend only on
    the event stream and the bound — never on how the iterator was chunked —
    and a repeat produces byte-identical partitions."""

    import itertools

    run = _run(_bootstrap_payload(9))
    pairs = _pairs(run)
    pairs[2] = (pairs[2][0], _stretch(pairs[2][1], length=17))
    budget = _bounded(4, path_block_size=3)
    total = sum(len(result.events) for _record, result in pairs)

    def _build(source, name):
        directory = _fresh(tmp_path, name)
        return build_account_event_detail(
            source,
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=budget,
            event_order_policy_id=_ORDER,
            directory=directory,
            total_rows=total,
            path_count=9,
        )

    whole = _build(iter(pairs), "whole")
    chunked = _build(itertools.chain(iter(pairs[:1]), iter(pairs[1:5]), iter(pairs[5:])), "chunked")
    one_by_one = _build((pair for pair in pairs), "generator")
    for other in (chunked, one_by_one):
        assert other.manifest == whole.manifest
        assert other.produced() == whole.produced()
        for record in whole.produced():
            assert (whole.directory / record.name).read_bytes() == (
                other.directory / record.name
            ).read_bytes()
    # three path blocks of three paths; the long path splits inside block 0
    blocks = sorted({entry["path_block_id"] for entry in whole.manifest["partitions"]})
    assert blocks == [0, 1, 2]
    assert all(entry["rows"] <= 4 for entry in whole.manifest["partitions"])
    for block in blocks:
        ordinals = [
            entry["partition_ordinal"]
            for entry in whole.manifest["partitions"]
            if entry["path_block_id"] == block
        ]
        assert ordinals == list(range(len(ordinals)))


def test_total_row_refusal_publishes_nothing_and_cleans_every_partition(tmp_path):
    """HB-FIX-11: the total row budget still fails before publication under
    the bounded writer, nothing is truncated, and a refused build leaves no
    partition file behind (direct directory and store publication alike)."""

    from alpha_lab.agents.data_infra.ifvg.search.store import (
        has_envelope,
        save_envelope_immutable,
    )

    run = _run(_bootstrap_payload(6))
    pairs = _pairs(run)
    total = _rows(run)
    capped = EventDetailBudget(
        budget_id="capped_total",
        max_event_detail_rows=total - 1,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=2,
        max_rows_per_partition=3,
    )
    # the preflight refuses the declared count outright
    directory = _fresh(tmp_path, "preflight")
    with pytest.raises(EventDetailBudgetError, match="preflight"):
        _stream_build(run, directory, budget=capped)
    assert os.listdir(directory) == []
    # an under-declared count streams past the budget: refused at the row,
    # after several partitions were already written — all of them removed
    directory = _fresh(tmp_path, "overrun")
    with pytest.raises(EventDetailBudgetError, match="row overrun"):
        _stream_build(run, directory, budget=capped, total_rows=total - 1, source=iter(pairs))
    assert os.listdir(directory) == []
    # through the store: the publication is discarded and no entry exists
    root = tmp_path / "store"
    envelope = run.envelope

    def _producer(publication_directory):
        bundle = build_account_event_detail(
            iter(pairs),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=capped,
            event_order_policy_id=_ORDER,
            directory=publication_directory,
            total_rows=total - 1,
            path_count=6,
        )
        return bundle.produced()

    with pytest.raises(EventDetailBudgetError, match="row overrun"):
        save_envelope_immutable(root, "account_simulations", envelope, sidecar_producer=_producer)
    assert not has_envelope(root, "account_simulations", envelope.account_simulation_id)
    store_dir = root / "account_simulations"
    assert not store_dir.exists() or not any(
        name.startswith(".") for name in os.listdir(store_dir)
    )


# ── HARDENING-BACKEND-FIX focused review round (RB-03, RB-04) ────────────────


def _tiny_budget() -> EventDetailBudget:
    return EventDetailBudget(
        budget_id="tiny_partitions",
        max_event_detail_rows=EVENT_DETAIL_BUDGET_V2.max_event_detail_rows,
        max_published_bytes=EVENT_DETAIL_BUDGET_V2.max_published_bytes,
        path_block_size=2,
        max_rows_per_partition=3,
    )


def test_cleanup_covers_a_partial_write_and_the_manifest_stage(tmp_path, monkeypatch):
    """Review RB-03: a partition file left half-written by an I/O failure and
    a refusal at the detail-manifest stage both leave the caller-owned
    directory clean; a pre-existing (foreign) manifest is never removed."""

    run = _run(_bootstrap_payload(6))
    real_write = event_detail_module._write_parquet
    calls = {"n": 0}

    def _partial_then_fail(table, path):
        calls["n"] += 1
        if calls["n"] == 3:
            path.write_bytes(b"PAR1 partial (simulated ENOSPC)")
            raise OSError(28, "no space left on device (simulated)")
        real_write(table, path)

    monkeypatch.setattr(event_detail_module, "_write_parquet", _partial_then_fail)
    directory = _fresh(tmp_path, "partial")
    with pytest.raises(OSError, match="simulated"):
        _stream_build(run, directory, budget=_tiny_budget())
    assert calls["n"] == 3
    assert os.listdir(directory) == []  # the two complete AND the partial partition removed
    monkeypatch.undo()
    # a refusal at the manifest stage removes every partition, never the foreign manifest
    directory = _fresh(tmp_path, "manifest")
    foreign = directory / event_detail_module.EVENT_DETAIL_MANIFEST_SIDECAR
    foreign.write_bytes(b"{}\n")
    with pytest.raises(
        event_detail_module.EventDetailIntegrityError, match="manifest already exists"
    ):
        _stream_build(run, directory, budget=_tiny_budget())
    assert os.listdir(directory) == [event_detail_module.EVENT_DETAIL_MANIFEST_SIDECAR]
    assert foreign.read_bytes() == b"{}\n"
    # the same inputs build cleanly afterwards (nothing stale is left over)
    directory = _fresh(tmp_path, "clean")
    bundle = _stream_build(run, directory, budget=_tiny_budget())
    assert bundle.max_partition_rows <= 3 and bundle.partition_count >= 3


def test_reader_binds_the_partition_bound_to_the_identity_bound_budget(tmp_path, monkeypatch):
    """Review RB-04: the bound the reader proves is the simulation identity's
    budget bound (the manifest budget proven equal to the payload's); a detail
    manifest whose loose top-level bound disagrees is refused before any
    partition is yielded."""

    from alpha_lab.agents.data_infra.ifvg.search.store import save_envelope_immutable

    run = _run(_bootstrap_payload(4))
    pairs = _pairs(run)
    root = tmp_path / "store"

    def _producer(publication_directory):
        return build_account_event_detail(
            iter(pairs),
            clock_policy_id=BOOTSTRAP_CLOCK_POLICY.policy_id,
            budget=EVENT_DETAIL_BUDGET_V2,
            event_order_policy_id=_ORDER,
            directory=publication_directory,
            total_rows=_rows(run),
            path_count=4,
        ).produced()

    save_envelope_immutable(root, "account_simulations", run.envelope, sidecar_producer=_producer)
    simulation_id = run.envelope.account_simulation_id
    manifest = event_detail_module.load_account_event_detail_manifest(root, simulation_id)
    assert manifest["max_rows_per_partition"] == manifest["budget"]["max_rows_per_partition"]
    frames = list(event_detail_module.load_account_event_detail(root, simulation_id))
    assert sum(len(frame) for frame in frames) == _rows(run)
    real_loader = event_detail_module.load_account_event_detail_manifest

    def _loosened(store_root, account_simulation_id):
        loose = real_loader(store_root, account_simulation_id)
        loose["max_rows_per_partition"] = int(loose["max_rows_per_partition"]) * 1000
        return loose

    monkeypatch.setattr(event_detail_module, "load_account_event_detail_manifest", _loosened)
    with pytest.raises(
        event_detail_module.EventDetailIntegrityError, match="disagrees with the identity-bound"
    ):
        list(event_detail_module.load_account_event_detail(root, simulation_id))
