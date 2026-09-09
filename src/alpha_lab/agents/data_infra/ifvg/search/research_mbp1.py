"""Real MBP-1 evidence adapter using existing normalization/coverage semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from strategy_core.data.databento_parquet import DatabentoParquetSource

from ..contracts import RecordTable
from ..development_access import (
    DEVELOPMENT_CUTOFF_UTC,
    PERMITTED_DEVELOPMENT_DATES,
    DevelopmentReplayPolicy,
)
from ..features.mbp1_arrow_schemas import (
    DATABENTO_PRICE_SCALE,
    MBP1_SOURCE_EVENT_SCHEMA,
    assert_schema_names_match,
)
from ..features.mbp1_coverage_evidence import (
    MBP1_COMPLETENESS_REPORT_STORE,
    MBP1_GAP_MANIFEST_STORE,
    Mbp1CompletenessCompilationReportEnvelope,
    Mbp1CompletenessCompilationReportPayload,
    Mbp1DeclaredGapInterval,
    Mbp1EvidenceProvenance,
    Mbp1EvidenceScope,
    Mbp1EvidenceScopeLevel,
    authorized_session_span_ns,
    compile_mbp1_partition_gap_manifest,
    compute_partition_coverage,
    load_verified_partition_evidence,
    save_mbp1_partition_gap_manifest,
)
from ..features.mbp1_source_artifact import (
    Mbp1SourceArtifactEnvelope,
    _clipped_spans,
    _partition_coverage,
    build_mbp1_source_artifact,
    normalize_mbp1_events,
)
from ..features.mbp1_source_contract import R5B_WINDOW_SPECS, Mbp1SourceContract
from ..features.mbp1_stage_windows import stage_anchor_frame_from_candidates
from ..manifest import file_sha256
from .identities import canonical_contract_sha256
from .store import save_or_reuse_envelope

ROLL_POLICY = "strategy_core_dominant_trade_count_authorized_partition_span_v1"


class BoundedDayEvents(Mapping):
    """Lazy normalized events with at most one resident logical day."""

    def __init__(self, days, loader):
        self.days = tuple(days)
        if self.days != tuple(sorted(set(self.days))):
            raise ValueError("lazy MBP-1 days must be ordered and unique")
        self.loader = loader
        self._cached_day = None
        self._cached_frame = None
        self.load_count = 0
        self.max_resident_days = 0

    def __len__(self):
        return len(self.days)

    def __iter__(self):
        return iter(self.days)

    def __getitem__(self, day):
        if day not in self.days:
            raise KeyError(day)
        if day != self._cached_day:
            self.release()
            self._cached_frame = self.loader(day)
            self._cached_day = day
            self.load_count += 1
            self.max_resident_days = max(self.max_resident_days, len(self.resident_days))
        return self._cached_frame

    @property
    def resident_days(self):
        return () if self._cached_day is None else (self._cached_day,)

    def release(self):
        self._cached_day = None
        self._cached_frame = None


def _verify_physical_freshness(subject, repo_root, day, policy, frozen_refs):
    for physical in _physical_dates(subject, day):
        path = policy.resolve_source_path(
            physical, lambda value: Path(repo_root) / "data/databento/NQ" / value / "mbp1.parquet"
        )
        expected = frozen_refs[physical].get("sha256")
        if path.is_file():
            policy.record_file_open(physical)
            actual = file_sha256(path)
        else:
            actual = None
        if actual != expected:
            raise ValueError("MBP-1 physical source changed after approved preflight")


def research_mbp1_contract() -> Mbp1SourceContract:
    """Describe the Core selector actually called below, not an inferred OI roll."""
    return Mbp1SourceContract(
        instrument="NQ",
        contract_roll_policy_id=ROLL_POLICY,
        feature_window_specs=R5B_WINDOW_SPECS,
        coverage_policy={"min_day_coverage_fraction": 0.95},
    )


def _physical_dates(subject, day):
    if day not in subject.evaluation_dates:
        raise PermissionError("MBP-1 logical day is outside the exact subject evaluation calendar")
    previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    dates = (previous, day)
    if any(value not in PERMITTED_DEVELOPMENT_DATES for value in dates):
        raise PermissionError("MBP-1 physical source date is outside permitted development")
    return dates


def research_mbp1_physical_dates(subject):
    return tuple(
        sorted(
            {value for day in subject.evaluation_dates for value in _physical_dates(subject, day)}
        )
    )


def _source_policy(subject):
    return DevelopmentReplayPolicy(
        tuple(sorted(set(subject.warmup_dates) | set(research_mbp1_physical_dates(subject))))
    )


def _discover_evidence(store_root, subject, partition_refs):
    by_date = {item["source_date"]: item for item in partition_refs if item.get("sha256")}
    found, warnings = {}, []
    for path in sorted((Path(store_root) / MBP1_GAP_MANIFEST_STORE).glob("*/envelope.json")):
        evidence = load_verified_partition_evidence(store_root, manifest_id=path.parent.name)
        scope = evidence.scope
        if scope.utc_date not in by_date or scope.schema_name != "mbp-1":
            continue
        if evidence.provenance is not Mbp1EvidenceProvenance.OWNER_REVIEWED:
            continue
        report = evidence.completeness_report
        ref = by_date[scope.utc_date]
        if report is None or tuple(report.payload.verified_partition_refs) != (ref["sha256"],):
            continue
        if scope.symbol not in (None, "NQ") or scope.dataset != "GLBX.MDP3":
            continue
        if scope.scope_level not in (
            Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION,
            Mbp1EvidenceScopeLevel.INSTRUMENT_PARTITION,
        ):
            continue
        midnight = pd.Timestamp(scope.utc_date, tz="UTC").value
        if (
            not midnight
            <= scope.partition_expected_start_ts
            < scope.partition_expected_end_ts
            <= midnight + pd.Timedelta(days=1).value
        ):
            warnings.append(f"MBP-1 evidence {path.parent.name} is not one physical UTC partition")
            continue
        found.setdefault(scope.utc_date, []).append(evidence)
    for day, items in found.items():
        if len(items) != 1:
            raise ValueError(f"ambiguous MBP-1 coverage evidence for physical partition {day}")
    mapping = {
        day: tuple(e for physical in _physical_dates(subject, day) for e in found.get(physical, ()))
        for day in subject.evaluation_dates
    }
    return mapping, warnings


def preflight_research_mbp1(subject, store_root, repo_root) -> dict:
    """Schema/hash/receipt preflight; event rows are not decoded."""
    policy = _source_policy(subject)
    blockers, warnings = [], []
    physical = research_mbp1_physical_dates(subject)
    refs = []
    available = 0
    for day in physical:
        path = policy.resolve_source_path(
            day, lambda value: Path(repo_root) / "data/databento/NQ" / value / "mbp1.parquet"
        )
        if not path.is_file():
            refs.append({"source_date": day, "sha256": None})
            warnings.append(f"MBP-1 physical partition absent: {day}; affected coverage is unknown")
            continue
        policy.record_metadata_access(day)
        try:
            schema = pq.read_schema(path)
            assert_schema_names_match(MBP1_SOURCE_EVENT_SCHEMA, schema.names)
            policy.record_file_open(day)
            refs.append(
                {
                    "source_date": day,
                    "sha256": file_sha256(path),
                    "row_count": pq.read_metadata(path).num_rows,
                    "relative_path": f"data/databento/NQ/{day}/mbp1.parquet",
                }
            )
            available += 1
        except (ValueError, OSError, pa.ArrowException) as exc:
            blockers.append(f"MBP-1 schema unavailable for {day}: {exc}")
    if available == 0:
        blockers.append("No MBP-1 source partitions are available for the subject calendar")
    warnings.append("No completeness is inferred from file presence or sequence continuity")
    evidence, evidence_warnings = _discover_evidence(store_root, subject, refs)
    warnings.extend(evidence_warnings)
    ids = sorted(
        {
            item.gap_manifest.mbp1_partition_gap_manifest_id
            for items in evidence.values()
            for item in items
        }
    )
    by_date = {item["source_date"]: item for item in refs}
    potential_days = []
    for day, items in evidence.items():
        if any(
            by_date[item.scope.utc_date].get("row_count", 0) > 0
            and compute_partition_coverage(
                item,
                trading_day=day,
                partition_content_refs=(by_date[item.scope.utc_date]["sha256"],),
            ).coverage_fraction
            >= 0.95
            for item in items
        ):
            potential_days.append(day)
    if not potential_days:
        blockers.append(
            "R5B requires an owner-reviewed MBP-1 extraction/coverage receipt bound to "
            "the current physical partition SHA256; import it with import_research_mbp1_receipt"
        )
    policy.assert_zero_forbidden_access()
    contract = research_mbp1_contract()
    return {
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "partition_count": len(physical),
        "available_partition_count": available,
        "physical_source_dates": list(physical),
        "input_partition_refs": refs,
        "coverage_evidence_ids": ids,
        "evidence_refs": ids,
        "evidenced_logical_day_count": sum(bool(items) for items in evidence.values()),
        "potentially_usable_logical_days": potential_days,
        "mbp1_source_contract_id": canonical_contract_sha256(contract),
        "source_contract": contract.model_dump(mode="json"),
    }


def import_research_mbp1_receipt(
    subject, store_root, repo_root, receipt: dict, *, dry_run=False
) -> dict:
    """Compile a supplied extraction declaration and explicit owner review.

    ``receipt`` contains ``source_document`` (schema_version=1, scope using the
    existing Mbp1EvidenceScope fields, partition_sha256, complete_outside_intervals,
    intervals, extraction_description), and ``owner_review`` (subject_id,
    source_document_sha256, partition_sha256, approve_completeness, reviewed_by,
    reviewed_at). The exact documents are retained; file presence cannot create
    either declaration or approval. No market-event rows are decoded.
    """
    if not isinstance(receipt, dict) or not all(
        isinstance(receipt.get(key), dict) for key in ("source_document", "owner_review")
    ):
        raise ValueError("receipt requires source_document and owner_review objects")
    document, review = receipt["source_document"], receipt["owner_review"]
    if document.get("schema_version") != 1 or not document.get("extraction_description"):
        raise ValueError("receipt requires a versioned explicit extraction declaration")

    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    source_bytes, review_bytes = encoded(document), encoded(review)
    document_sha = hashlib.sha256(source_bytes).hexdigest()
    owner_sha = hashlib.sha256(review_bytes).hexdigest()
    scope = Mbp1EvidenceScope.model_validate(document["scope"])
    if scope.utc_date not in research_mbp1_physical_dates(subject):
        raise PermissionError("receipt physical partition is outside the subject source scope")
    if scope.dataset != "GLBX.MDP3" or scope.symbol not in (None, "NQ"):
        raise ValueError("receipt does not describe this NQ source")
    if scope.scope_level not in (
        Mbp1EvidenceScopeLevel.PUBLISHER_PARTITION,
        Mbp1EvidenceScopeLevel.INSTRUMENT_PARTITION,
    ):
        raise ValueError("this adapter requires publisher or instrument physical partition scope")
    midnight = pd.Timestamp(scope.utc_date, tz="UTC").value
    if (
        not midnight
        <= scope.partition_expected_start_ts
        < scope.partition_expected_end_ts
        <= midnight + pd.Timedelta(days=1).value
    ):
        raise ValueError("receipt scope must describe one physical UTC partition")
    if (
        document.get("complete_outside_intervals") is not True
        or review.get("approve_completeness") is not True
    ):
        raise ValueError("positive coverage requires explicit extraction and owner declarations")
    if not review.get("reviewed_by") or not review.get("reviewed_at"):
        raise ValueError("receipt requires an attributable, dated owner review")
    if (
        review.get("subject_id") != subject.subject_id
        or review.get("source_document_sha256") != document_sha
    ):
        raise ValueError("owner review does not bind this subject and extraction document")
    policy = _source_policy(subject)
    path = policy.resolve_source_path(
        scope.utc_date, lambda day: Path(repo_root) / "data/databento/NQ" / day / "mbp1.parquet"
    )
    policy.record_file_open(scope.utc_date)
    partition_sha = file_sha256(path)
    if (
        document.get("partition_sha256") != partition_sha
        or review.get("partition_sha256") != partition_sha
    ):
        raise ValueError("receipt partition hash differs from the actual physical source")
    report = Mbp1CompletenessCompilationReportEnvelope.from_payload(
        Mbp1CompletenessCompilationReportPayload(
            scope=scope,
            source_inventory_id=document_sha,
            verified_partition_refs=(partition_sha,),
            evidence_refs=(document_sha,),
            owner_review_decision_id=owner_sha,
            provenance="owner_reviewed",
            compiled_intervals=tuple(
                Mbp1DeclaredGapInterval.model_validate(value)
                for value in document.get("intervals", ())
            ),
            positive_completeness_authorized=True,
            compiled_at=review["reviewed_at"],
        )
    )
    manifest = compile_mbp1_partition_gap_manifest(
        report,
        source_document_sha256=document_sha,
        declared_by=review["reviewed_by"],
        declared_at=review["reviewed_at"],
    )
    if not dry_run:
        save_or_reuse_envelope(
            Path(store_root),
            MBP1_COMPLETENESS_REPORT_STORE,
            report,
            extra_files={"source_document.json": source_bytes, "owner_review.json": review_bytes},
        )
        save_mbp1_partition_gap_manifest(Path(store_root), manifest)
    policy.assert_zero_forbidden_access()
    return {
        "coverage_evidence_id": manifest.mbp1_partition_gap_manifest_id,
        "completeness_report_id": report.mbp1_completeness_compilation_report_id,
        "partition_sha256": partition_sha,
    }


def read_research_mbp1_day(subject, repo_root, day, policy):
    """Compose authorized UTC partitions, scale exported prices, then normalize.

    The unchanged Core dominant-instrument selector receives only the exact
    authorized interval. Post-cutoff rows cannot influence contract selection.
    Source decisions and hashes remain attached to the composed evidence.
    """
    _physical_dates(subject, day)
    policy.authorize_date(day)
    start_ns, end_ns = authorized_session_span_ns(day)
    end_ns = min(end_ns, pd.Timestamp(DEVELOPMENT_CUTOFF_UTC).value)
    frames, refs = [], []
    for physical_day in _physical_dates(subject, day):
        path = policy.resolve_source_path(
            physical_day,
            lambda value: Path(repo_root) / "data/databento/NQ" / value / "mbp1.parquet",
        )
        if not path.is_file():
            continue
        midnight = pd.Timestamp(physical_day, tz="UTC")
        lower = max(start_ns, midnight.value)
        upper = min(end_ns, (midnight + pd.Timedelta(days=1)).value)
        if lower >= upper:
            continue
        parquet = pq.ParquetFile(path)
        assert_schema_names_match(MBP1_SOURCE_EVENT_SCHEMA, parquet.schema_arrow.names)
        lower_ts, upper_ts = pd.Timestamp(lower, tz="UTC"), pd.Timestamp(upper, tz="UTC")
        timestamp_type = parquet.schema_arrow.field("ts_event").type
        lower_value = lower_ts.to_pydatetime() if pa.types.is_timestamp(timestamp_type) else lower
        upper_value = upper_ts.to_pydatetime() if pa.types.is_timestamp(timestamp_type) else upper
        columns = [field.name for field in MBP1_SOURCE_EVENT_SCHEMA]
        bounded = pq.read_table(
            path,
            columns=columns,
            filters=[("ts_event", ">=", lower_value), ("ts_event", "<", upper_value)],
        )

        class SelectorInput:
            def __init__(self, table):
                self.table = table

            def read(self, *, columns):
                return self.table.select(columns)

        core_source = DatabentoParquetSource(paths=(path,), requested_symbol="NQ", schema="mbp-1")
        selected_id = core_source._front_month_instrument_id(
            SelectorInput(bounded),
            parquet.schema_arrow.names,
            None,
        )
        raw = bounded.to_pandas()
        policy.record_file_open(physical_day, rows=len(raw))
        ts = pd.to_numeric(raw["ts_event"], errors="raise").astype("int64")
        mask = (
            ts.ge(lower) & ts.lt(upper) & ~raw["symbol"].astype(str).str.contains("-", regex=False)
        )
        if selected_id is not None:
            mask &= raw["instrument_id"].eq(selected_id)
        elif len(raw):
            raise ValueError("Core cannot establish a dominant traded instrument for MBP-1")
        raw = raw.loc[mask].copy()
        for column in ("price", "bid_px_00", "ask_px_00"):
            dtype = parquet.schema_arrow.field(column).type
            if pa.types.is_floating(dtype):
                raw[column] = pd.to_numeric(raw[column], errors="raise") / DATABENTO_PRICE_SCALE
            elif not pa.types.is_integer(dtype):
                raise ValueError(f"unsupported MBP-1 price representation: {column}")
        frames.append(raw)
        refs.append(
            {
                "source_date": physical_day,
                "sha256": file_sha256(path),
                "instrument_id": selected_id,
                "start_ns": lower,
                "end_ns": upper,
            }
        )
    if frames:
        raw = pd.concat(frames, ignore_index=True)
    else:
        raw = pd.DataFrame(
            {field.name: pd.Series(dtype="object") for field in MBP1_SOURCE_EVENT_SCHEMA}
        )
    normalized = normalize_mbp1_events(raw, instrument="NQ", trading_day=day)
    normalized.attrs["source_file_sha256"] = canonical_contract_sha256(refs)
    normalized.attrs["source_partition_refs"] = refs
    return normalized


def build_research_mbp1_evidence(
    preparation,
    *,
    contract: Mbp1SourceContract,
    coverage_evidence=None,
):
    """Read exact subject days, preserving unknown completeness as unknown.

    The caller supplies its explicit roll/source contract and verified coverage
    records. Sequence gaps and timestamps are never promoted to completeness.
    """
    preparation.prepare()
    subject = preparation.subject
    if contract.instrument != "NQ":
        raise ValueError("this saved-child source adapter requires the exact NQ instrument")
    if contract.contract_roll_policy_id != ROLL_POLICY:
        raise ValueError("MBP-1 contract does not describe the actual Core instrument selector")
    frozen = getattr(preparation, "mbp1_preflight", None)
    current = preflight_research_mbp1(subject, preparation.store_root, preparation.repo_root)
    if not current["passed"]:
        raise ValueError("; ".join(current["blockers"]))
    if frozen is not None and any(
        current[key] != frozen.get(key)
        for key in (
            "input_partition_refs",
            "coverage_evidence_ids",
            "physical_source_dates",
            "mbp1_source_contract_id",
        )
    ):
        raise ValueError("MBP-1 source or coverage evidence changed after research approval")
    if canonical_contract_sha256(contract) != current["mbp1_source_contract_id"]:
        raise ValueError("MBP-1 source contract differs from the frozen canonical adapter")
    discovered, _warnings = _discover_evidence(
        preparation.store_root, subject, current["input_partition_refs"]
    )
    if coverage_evidence is not None and coverage_evidence != discovered:
        raise ValueError("explicit MBP-1 evidence differs from verified source discovery")
    coverage_evidence = discovered
    if any(
        evidence.provenance is Mbp1EvidenceProvenance.SYNTHETIC_FIXTURE
        for items in (coverage_evidence or {}).values()
        for evidence in items
    ):
        raise PermissionError("synthetic completeness cannot certify real research MBP-1")
    if not set(coverage_evidence or {}) <= set(subject.evaluation_dates):
        raise ValueError("MBP-1 coverage evidence escapes the subject evaluation calendar")
    policy = _source_policy(subject)
    frozen_refs = {item["source_date"]: item for item in current["input_partition_refs"]}

    def load_day(day):
        _verify_physical_freshness(subject, preparation.repo_root, day, policy, frozen_refs)
        frame = read_research_mbp1_day(subject, preparation.repo_root, day, policy)
        _verify_physical_freshness(subject, preparation.repo_root, day, policy, frozen_refs)
        policy.assert_zero_forbidden_access()
        return frame

    events = BoundedDayEvents(subject.evaluation_dates, load_day)
    # Reuse the coverage kernel with the physical raw file each receipt certifies.
    # The ordinary one-file adapter binds a single raw hash plus normalized hash;
    # these two-UTC-file days retain normalized content identity independently.
    partitions = []
    source = None
    for day in events:
        day_events = events[day]
        source, _bytes = build_mbp1_source_artifact(
            {day: day_events},
            contract=contract,
            authorized_date_set_id=canonical_contract_sha256(subject.evaluation_dates),
            events_stored=False,
            retain_event_bytes=False,
        )
        original = source.payload.ordered_partitions[0]
        day_evidence = tuple(coverage_evidence.get(day, ()))
        if not day_evidence:
            partitions.append(original)
            del day_events
            continue
        _clipped_spans(day_evidence, trading_day=day)
        refs = {item["source_date"]: item for item in day_events.attrs["source_partition_refs"]}
        for evidence in day_evidence:
            ref = refs[evidence.scope.utc_date]
            if (
                evidence.scope.instrument_id is not None
                and evidence.scope.instrument_id != ref["instrument_id"]
            ):
                raise ValueError(
                    "receipt instrument differs from the actual Core-selected contract"
                )
            scope = evidence.scope
            physical_mask = day_events["ts_event"].ge(
                scope.partition_expected_start_ts
            ) & day_events["ts_event"].lt(scope.partition_expected_end_ts)
            if not day_events.loc[physical_mask, "publisher_id"].eq(scope.publisher_id).all():
                raise ValueError("receipt publisher differs from the normalized physical source")
            partitions.append(
                _partition_coverage(
                    day_events,
                    evidence,
                    trading_day=day,
                    content_sha256=original.content_sha256,
                    source_content_refs=(ref["sha256"],),
                )
            )
        del day_events
    if source is None:
        raise ValueError("research MBP-1 requires a nonempty logical calendar")
    source = Mbp1SourceArtifactEnvelope.from_payload(
        source.payload.model_copy(update={"ordered_partitions": tuple(partitions)}),
        events_stored=False,
    )
    events.release()
    after = preflight_research_mbp1(subject, preparation.store_root, preparation.repo_root)
    if any(after[key] != current[key] for key in ("input_partition_refs", "coverage_evidence_ids")):
        raise ValueError("MBP-1 source changed while evidence was being prepared")
    scoped_ids = set(preparation.candidate_view.frame["candidate_id"].astype(str))
    candidates = preparation.v2_tables[RecordTable.ENTRY_CANDIDATE]
    candidates = candidates.loc[candidates["candidate_id"].astype(str).isin(scoped_ids)].copy()
    # Exact immutable geometry timestamps, not nearest-stage or chronological joins.
    for destination, source_column in {
        "tap_ts_utc": "geometry_tap_bar_logical_close_ts_utc",
        "lock_ts_utc": "geometry_lock_bar_logical_close_ts_utc",
        "armed_ts_utc": "geometry_opposing_confirmed_ts_utc",
        "inversion_ts_utc": "geometry_inversion_bar_logical_close_ts_utc",
        "entry_ts_utc": "geometry_entry_bar_logical_close_ts_utc",
    }.items():
        if source_column in candidates:
            candidates[destination] = candidates[source_column]
    anchors = stage_anchor_frame_from_candidates(candidates)
    policy.assert_zero_forbidden_access()
    preparation.mbp1_evidence = {
        "source_envelope": source,
        "events_by_day": events,
        "anchors": anchors,
        "access_audit": policy.audit_dict(),
    }
    return source, events, anchors
