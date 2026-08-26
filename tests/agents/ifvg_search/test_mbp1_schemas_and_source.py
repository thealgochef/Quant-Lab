"""R5B suites: exact Arrow schemas + the immutable MBP-1 source artifact.

TEST_MATRIX rows: schema hashes as identity inputs; deep-book guard over
every field name; legacy source-kind scoping re-run at R5B; content
addressing/identity sensitivity; immutable save/reload/reuse; the
no-``+inf`` and no-nearest-time source scans over the features package.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.features import (
    mbp1_arrow_schemas as schemas,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
    ORDER_KEY_COLUMNS,
    load_mbp1_source_artifact,
    load_partition_events,
    normalize_mbp1_events,
    read_mbp1_partition_frame,
    save_mbp1_source_artifact,
)
from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    Mbp1SourceContract,
    assert_no_deep_book_identifiers,
)
from tests.agents.ifvg_search.mbp1_fixture import (
    FIXTURE_DAY,
    build_fixture_source,
    default_day_events,
    ns_at,
    raw_event,
    synthetic_contract,
)

_FEATURES_DIR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "alpha_lab"
    / "agents"
    / "data_infra"
    / "ifvg"
    / "features"
)


# ── schemas ──────────────────────────────────────────────────────────────────


def test_source_schema_retains_ts_recv_and_the_full_databento_shape() -> None:
    names = [field.name for field in schemas.MBP1_SOURCE_EVENT_SCHEMA]
    for required in ("ts_recv", "ts_event", "sequence", "bid_px_00", "ask_px_00"):
        assert required in names
    assert names.index("ts_recv") == 0  # databento column order starts at ts_recv


def test_schema_hashes_are_stable_and_field_sensitive() -> None:
    import pyarrow as pa

    assert schemas.arrow_schema_hash(schemas.MBP1_SOURCE_EVENT_SCHEMA) == (
        schemas.MBP1_SOURCE_EVENT_SCHEMA_HASH
    )
    changed = pa.schema(
        [*schemas.MBP1_SOURCE_EVENT_SCHEMA, pa.field("extra", pa.int64())]
    )
    assert schemas.arrow_schema_hash(changed) != schemas.MBP1_SOURCE_EVENT_SCHEMA_HASH
    retyped = pa.schema(
        [
            pa.field(field.name, pa.float64() if field.name == "sequence" else field.type)
            for field in schemas.MBP1_SOURCE_EVENT_SCHEMA
        ]
    )
    assert schemas.arrow_schema_hash(retyped) != schemas.MBP1_SOURCE_EVENT_SCHEMA_HASH


def test_every_schema_field_passes_the_deep_book_guard() -> None:
    for schema in (
        schemas.MBP1_SOURCE_EVENT_SCHEMA,
        schemas.MBP1_NORMALIZED_EVENT_SCHEMA,
        schemas.MBP1_FEATURE_TABLE_SCHEMA,
        schemas.MBP1_STAGE_WINDOW_EVIDENCE_SCHEMA,
    ):
        assert_no_deep_book_identifiers(field.name for field in schema)


def test_feature_table_schema_carries_per_window_evidence_columns() -> None:
    names = {field.name for field in schemas.MBP1_FEATURE_TABLE_SCHEMA}
    assert set(schemas.mbp1_window_validity_fields()) <= names
    assert set(schemas.mbp1_window_missing_reason_fields()) <= names
    assert len(schemas.mbp1_window_keys()) == 9


# ── normalization and coverage ───────────────────────────────────────────────


def test_normalization_assigns_source_ordinal_before_sorting() -> None:
    rows = [
        raw_event(ts_event=ns_at(10), sequence=5),
        raw_event(ts_event=ns_at(5), sequence=4),  # out of order in the source
    ]
    normalized = normalize_mbp1_events(
        pd.DataFrame(rows), instrument="NQ", trading_day=FIXTURE_DAY
    )
    # sorted by the four-part key, but ordinals reflect SOURCE row order
    assert list(normalized["ts_event"]) == sorted(normalized["ts_event"])
    assert list(normalized["source_ordinal"]) == [1, 0]
    assert tuple(ORDER_KEY_COLUMNS) == ("ts_event", "ts_recv", "sequence", "source_ordinal")


def test_unknown_instrument_refuses_instead_of_guessing_scale() -> None:
    with pytest.raises(ValueError, match="no pinned tick size"):
        normalize_mbp1_events(
            pd.DataFrame([raw_event(ts_event=ns_at(1), sequence=1)]),
            instrument="ZB",
            trading_day=FIXTURE_DAY,
        )


def test_sequence_gaps_mark_intervals_and_resets_do_not() -> None:
    rows = [
        raw_event(ts_event=ns_at(0), sequence=100),
        raw_event(ts_event=ns_at(10), sequence=101),
        raw_event(ts_event=ns_at(20), sequence=105),  # gap 101→105
        raw_event(ts_event=ns_at(30), sequence=3),  # vendor reset — NOT a gap
        raw_event(ts_event=ns_at(40), sequence=4),
    ]
    envelope, _bytes, _events = build_fixture_source(
        {
            FIXTURE_DAY: normalize_mbp1_events(
                pd.DataFrame(rows), instrument="NQ", trading_day=FIXTURE_DAY
            )
        }
    )
    coverage = envelope.payload.ordered_partitions[0]
    assert coverage.sequence_gap_intervals == ((ns_at(10), ns_at(20)),)
    assert 0.0 < coverage.coverage_fraction < 1.0


# ── identity sensitivity, save/reload/reuse ──────────────────────────────────


def test_source_artifact_identity_is_content_addressed(tmp_path) -> None:
    envelope, event_bytes, events = build_fixture_source()
    again, _, _ = build_fixture_source()
    assert envelope.mbp1_source_artifact_id == again.mbp1_source_artifact_id

    changed_rows = default_day_events()
    changed_rows.loc[0, "bid_sz"] = 999
    changed, _, _ = build_fixture_source({FIXTURE_DAY: changed_rows})
    assert changed.mbp1_source_artifact_id != envelope.mbp1_source_artifact_id

    root = tmp_path / "store"
    save_mbp1_source_artifact(root, envelope, event_bytes)
    reloaded = load_mbp1_source_artifact(root, envelope.mbp1_source_artifact_id)
    assert reloaded.model_dump(mode="json") == envelope.model_dump(mode="json")
    events_back = load_partition_events(root, reloaded, FIXTURE_DAY)
    pd.testing.assert_frame_equal(events_back, events[FIXTURE_DAY])
    # idempotent second save = verified reuse, never a duplicate
    save_mbp1_source_artifact(root, envelope, event_bytes)


def test_tampered_event_sidecar_fails_closed(tmp_path) -> None:
    envelope, event_bytes, _events = build_fixture_source()
    root = tmp_path / "store"
    save_mbp1_source_artifact(root, envelope, event_bytes)
    stored = (
        root
        / "mbp1_source_artifacts"
        / envelope.mbp1_source_artifact_id
        / f"events_{FIXTURE_DAY.replace('-', '')}.arrow"
    )
    stored.write_bytes(b"tampered")
    with pytest.raises(Exception, match="verification|hash"):
        load_partition_events(root, envelope, FIXTURE_DAY)


def test_events_stored_artifact_requires_exactly_the_covered_days(tmp_path) -> None:
    envelope, event_bytes, _events = build_fixture_source()
    with pytest.raises(ValueError, match="exactly the covered days"):
        save_mbp1_source_artifact(tmp_path / "s", envelope, {})


# ── access safety: authorize-before-path + legacy scoping ────────────────────


def test_real_partition_read_refuses_without_a_policy() -> None:
    calls: list[str] = []

    def _factory(day: str) -> Path:  # pragma: no cover - must never run
        calls.append(day)
        return Path("nonexistent") / day / "mbp1.parquet"

    with pytest.raises(PermissionError, match="authorize-before-path"):
        read_mbp1_partition_frame(
            "2026-01-13", access_policy=None, path_factory=_factory, instrument="NQ"
        )
    assert calls == []  # fail-before-path: the factory was never invoked


def test_denied_day_refuses_before_any_path_construction(tmp_path) -> None:
    class _DenyingPolicy:
        def resolve_source_path(self, day, path_factory):
            raise PermissionError("source date is outside the verification allowlist")

    calls: list[str] = []

    def _factory(day: str) -> Path:  # pragma: no cover - must never run
        calls.append(day)
        return tmp_path / day / "mbp1.parquet"

    with pytest.raises(PermissionError, match="outside the verification allowlist"):
        read_mbp1_partition_frame(
            "2026-06-04",
            access_policy=_DenyingPolicy(),
            path_factory=_factory,
            instrument="NQ",
        )
    assert calls == []


def test_legacy_replay_provenance_is_unrepresentable_in_the_source_contract() -> None:
    with pytest.raises(Exception):  # noqa: B017 — pydantic Literal refusal
        Mbp1SourceContract.model_validate(
            {
                **synthetic_contract().model_dump(mode="json", by_alias=True),
                "schema": "mbp-10",
            }
        )
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
        _require_no_legacy_provenance,
    )

    with pytest.raises(PermissionError, match="cannot enter MBP-1 feature"):
        _require_no_legacy_provenance("legacy_verified_replay_source")


# ── source scans ─────────────────────────────────────────────────────────────


def _features_sources() -> dict[str, str]:
    """The scan surface: the features package PLUS every R5B module that can
    construct cutoffs or consume the evidence seam (review F6 widened the
    scope beyond the features package alone)."""

    ifvg_root = _FEATURES_DIR.parent
    extra = (
        ifvg_root / "search" / "pipeline.py",
        ifvg_root / "ml" / "supervised_ladder.py",
        ifvg_root / "ml" / "controlled_feature_study.py",
    )
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in (*sorted(_FEATURES_DIR.glob("*.py")), *extra)
    }


def test_no_plus_inf_cutoff_construction_exists() -> None:
    """TEST_MATRIX §3.8: the withdrawn '(stage_ts, +inf, +inf, +inf)' rule
    can never resurface — no infinite bound is constructed anywhere in the
    features package."""

    forbidden = re.compile(
        r"float\(\s*[\"']\+?inf|math\.inf|np\.inf|numpy\.inf|[\"']\+inf[\"']"
    )
    for name, source in _features_sources().items():
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not forbidden.search(stripped), (name, stripped)


def test_no_nearest_time_or_row_order_join_fallback_exists() -> None:
    """R5B deliverable 7: exact joins only — no as-of/nearest merges and no
    positional concatenation joins anywhere in the features package."""

    forbidden = re.compile(r"merge_asof|reindex_like|searchsorted|direction\s*=\s*[\"']nearest")
    for name, source in _features_sources().items():
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not forbidden.search(stripped), (name, stripped)


# ── review-round additions (findings S1/S2/S3/F5/F6/F7) ──────────────────────


def test_storage_mode_is_operational_never_identity() -> None:
    """Review F7: byte-identical evidence has ONE artifact id whether its
    event bytes are stored (synthetic) or referenced by hash (real)."""

    events = {FIXTURE_DAY: default_day_events()}
    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
        build_mbp1_source_artifact,
    )

    stored, _bytes = build_mbp1_source_artifact(
        events,
        contract=synthetic_contract(),
        authorized_date_set_id="synthetic_fixture_days_v1",
        events_stored=True,
    )
    referenced, _empty = build_mbp1_source_artifact(
        events,
        contract=synthetic_contract(),
        authorized_date_set_id="synthetic_fixture_days_v1",
        events_stored=False,
    )
    assert stored.mbp1_source_artifact_id == referenced.mbp1_source_artifact_id
    assert stored.events_stored is True
    assert referenced.events_stored is False


def test_saving_bytes_that_contradict_coverage_refuses(tmp_path) -> None:
    """Review F1: a sidecar whose bytes do not hash to the coverage row's
    content_sha256 refuses BEFORE any store write."""

    envelope, event_bytes, _events = build_fixture_source()
    poisoned = dict(event_bytes)
    poisoned[FIXTURE_DAY] = event_bytes[FIXTURE_DAY] + b"x"
    with pytest.raises(ValueError, match="coverage content_sha256"):
        save_mbp1_source_artifact(tmp_path / "s", envelope, poisoned)


def test_deeper_book_parquet_is_refused_and_reads_are_column_projected(
    tmp_path,
) -> None:
    """Safety review S1: an mbp-10-shaped file (every pinned column plus
    deeper book levels) is REFUSED before any row decodes; a lawful file is
    read through the pinned column projection only."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    class _PermissivePolicy:
        def resolve_source_path(self, day, path_factory):
            return path_factory(day)

    lawful = pd.DataFrame([raw_event(ts_event=ns_at(1), sequence=1)])
    deeper = lawful.assign(bid_px_01=1, ask_sz_01=2)
    lawful_path = tmp_path / "2026-01-13" / "mbp1.parquet"
    deeper_path = tmp_path / "2026-01-14" / "mbp1.parquet"
    lawful_path.parent.mkdir(parents=True)
    deeper_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pandas(lawful, preserve_index=False), lawful_path)
    pq.write_table(pa.Table.from_pandas(deeper, preserve_index=False), deeper_path)

    frame = read_mbp1_partition_frame(
        "2026-01-13",
        access_policy=_PermissivePolicy(),
        path_factory=lambda day: tmp_path / day / "mbp1.parquet",
        instrument="NQ",
    )
    assert len(frame) == 1
    with pytest.raises(PermissionError, match="book levels beyond MBP-1"):
        read_mbp1_partition_frame(
            "2026-01-14",
            access_policy=_PermissivePolicy(),
            path_factory=lambda day: tmp_path / day / "mbp1.parquet",
            instrument="NQ",
        )


def test_real_builder_refuses_no_policy_and_empty_days(tmp_path) -> None:
    """Safety review S3: the top-of-function refusals — never a degenerate
    zero-partition 'real' artifact."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_artifact import (
        build_mbp1_source_artifact_from_paths,
    )

    with pytest.raises(PermissionError, match="requires an access policy"):
        build_mbp1_source_artifact_from_paths(
            (),
            access_policy=None,
            path_factory=lambda day: tmp_path / day,
            contract=synthetic_contract(),
            authorized_date_set_id="x",
        )

    class _NeverPolicy:
        def resolve_source_path(self, day, path_factory):  # pragma: no cover
            raise AssertionError("must not be reached for empty days")

    with pytest.raises(ValueError, match="at least one day"):
        build_mbp1_source_artifact_from_paths(
            (),
            access_policy=_NeverPolicy(),
            path_factory=lambda day: tmp_path / day,
            contract=synthetic_contract(),
            authorized_date_set_id="x",
        )


def test_deep_book_regex_covers_every_depth_beyond_one() -> None:
    r"""Safety review S2: the guard is the §9 boundary ('MBP-1 is the
    maximum'), not only its MBP-10 example — mbp2…mbp9 are unrepresentable,
    while mbp1 (and the mbp-1 schema literal) stay lawful. The plan's
    literal `mbp[\W_]?(10|\d{2,})` remains a strict subset."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
        DEEP_BOOK_IDENTIFIER_REGEX,
    )

    for lawful in ("mbp1", "mbp-1", "mbp1_only_v1", "IFVG_ORDER_FLOW_MBP1_V1"):
        assert DEEP_BOOK_IDENTIFIER_REGEX.search(lawful) is None, lawful
    for unlawful in ("mbp5_queue", "mbp2", "MBP_7_x", "mbp10", "mbp-25"):
        assert DEEP_BOOK_IDENTIFIER_REGEX.search(unlawful), unlawful
    with pytest.raises(ValueError, match="unrepresentable"):
        assert_no_deep_book_identifiers(("mbp5_imbalance",))


def test_exact_key_refuses_inf_in_either_timestamp_element() -> None:
    """Review F6: both timestamp elements of the exact key are decimal
    nanosecond strings — infinity (or any non-numeric) in either position
    is unrepresentable."""

    from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
        StageEvidenceCutoff,
    )

    for key in (("+inf", "0", 0, 0), ("123", "+inf", 0, 0), ("123", "later", 0, 0)):
        with pytest.raises(ValueError, match="unrepresentable"):
            StageEvidenceCutoff(
                stage_id="entry",
                stage_as_of_ts_utc="2026-01-13T14:06:00Z",
                cutoff_kind="exact_source_order_key",
                exact_source_order_key=key,
                completed_bar_close_ts_utc=None,
                same_timestamp_policy_id="exact_key_total_order_v1",
                source_evidence_ref=None,
            )


def test_features_package_has_no_source_kind_ingress() -> None:
    """Review F5: the legacy-provenance seal is STRUCTURAL — no public
    callable in the features package accepts a `source_kind` parameter, so
    the opaque literal has no channel into materialization at all; the
    documented refusal function is its executable statement."""

    import importlib
    import inspect

    modules = [
        importlib.import_module(
            f"alpha_lab.agents.data_infra.ifvg.features.{path.stem}"
        )
        for path in sorted(_FEATURES_DIR.glob("*.py"))
        if path.stem != "__init__"
    ]
    for module in modules:
        for name in getattr(module, "__all__", ()):  # the public surface
            symbol = getattr(module, name)
            if not callable(symbol) or inspect.isclass(symbol):
                continue
            parameters = inspect.signature(symbol).parameters
            assert "source_kind" not in parameters, (module.__name__, name)
