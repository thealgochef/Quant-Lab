"""Explicit registered cache segments for Task B's separately prepared history.

Raw source roots, artifact destinations, source allowlists and artifact seed
lineage are independent. Registration inspects named caches without changing
them; replay selects one registered owner per date and verifies its saved bytes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
from strategy_core.candles._buckets import HTF_ANCHOR_POLICY
from strategy_core.data.databento_parquet import DatabentoParquetSource

from .data_access import DataAccessAudit, ExplorationDataPolicy, allowlist_sha256
from .day_artifacts import (
    _ARTIFACT_SCHEMA_VERSION,
    _META_KEY,
    DayArtifacts,
    DaySeeds,
    build_day_artifacts,
    load_day_artifacts,
    write_day_artifacts,
)
from .manifest import file_sha256

if TYPE_CHECKING:
    from .config import IfvgCaptureConfig

TASK_B_FIRST_PREPARATION_DAY = "2025-06-02"
TASK_B_LAST_PREPARATION_DAY = "2026-01-01"
TASK_B_LAST_PERMITTED_DAY = "2026-06-10"
TASK_B_MISSING_SOURCE_DATES = ("2025-07-08", "2025-11-14", "2025-11-20")
TASK_B_WARMUP_DATES = tuple(
    (date(2025, 6, 2) + timedelta(days=offset)).isoformat()
    for offset in range(12)
    if (date(2025, 6, 2) + timedelta(days=offset)).weekday() < 5
)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any, *, immutable: bool = False) -> None:
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if immutable and path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(
                "registered preparation evidence already exists with different content"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _ordered_dates(values: Iterable[str], *, allow_empty: bool = False) -> tuple[str, ...]:
    days = tuple(str(value) for value in values)
    if not days and not allow_empty:
        raise ValueError("a registered date scope cannot be empty")
    if days != tuple(sorted(days)) or len(days) != len(set(days)):
        raise ValueError("registered dates must be unique and chronological")
    for day in days:
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError("registered dates must use ISO calendar dates")
        if not TASK_B_FIRST_PREPARATION_DAY <= day <= TASK_B_LAST_PERMITTED_DAY:
            raise PermissionError(
                "registered preparation dates are outside Task B's permitted window"
            )
    return days


def _validate_definition(definition: Mapping[str, Any]) -> None:
    if definition.get("schema_version") != 1:
        raise ValueError("unsupported prepared-store registration schema")
    source = _ordered_dates(definition["source_dates"])
    preparation = _ordered_dates(definition["preparation_dates"])
    owned = _ordered_dates(definition["owned_dates"])
    if not set(owned) <= set(preparation) <= set(source):
        raise ValueError(
            "prepared-store ownership must be within its preparation and source scopes"
        )
    if definition["source_allowlist_sha256"] != allowlist_sha256(source):
        raise ValueError("prepared-store source scope hash differs from its dates")
    if definition["symbol"] != "NQ":
        raise ValueError("Task B prepared stores require the registered NQ source")
    if definition["mode"] not in {"separate_task_b_preparation", "preserved_existing"}:
        raise ValueError("unrecognized prepared-store mode")
    raw = Path(definition["raw_data_dir"])
    cache = Path(definition["artifact_cache_dir"])
    if not raw.is_absolute() or not cache.is_absolute():
        raise ValueError("prepared-store roots must be absolute")
    if definition["mode"] == "separate_task_b_preparation":
        if cache.resolve().is_relative_to(raw.resolve()) or raw.resolve().is_relative_to(
            cache.resolve()
        ):
            raise PermissionError(
                "Task B preparation destination must be separate from the raw/2026 store"
            )
        if any(day > TASK_B_LAST_PREPARATION_DAY for day in source):
            raise PermissionError(
                "the separate Task B preparation source scope ends January 1, 2026"
            )
        if any(date.fromisoformat(day).weekday() >= 5 for day in preparation):
            raise PermissionError("Task B logical preparation uses the supplied weekday inventory")
    authorization = Path(definition["authorization_file"])
    if file_sha256(authorization) != definition["authorization_file_sha256"]:
        raise ValueError("prepared-store owner authorization evidence changed")


def _definition(
    cfg: IfvgCaptureConfig,
    *,
    preparation_dates: tuple[str, ...],
    source_dates: tuple[str, ...],
    owned_dates: tuple[str, ...],
    artifact_cache_dir: Path,
    catalog_path: Path,
    authorization_file: Path,
    mode: str,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "mode": mode,
        "raw_data_dir": str(Path(cfg.data_dir).resolve()),
        "artifact_cache_dir": str(Path(artifact_cache_dir).resolve()),
        "symbol": cfg.symbol,
        "artifacts_tag": cfg.artifacts_tag(),
        "source_dates": list(source_dates),
        "preparation_dates": list(preparation_dates),
        "owned_dates": list(owned_dates),
        "source_allowlist_sha256": allowlist_sha256(source_dates),
        "catalog_path": str(Path(catalog_path).resolve()),
        "authorization_file": str(Path(authorization_file).resolve()),
        "authorization_file_sha256": file_sha256(Path(authorization_file)),
        "initial_day_seeds": DaySeeds(None, None, None, None).meta(),
    }
    _validate_definition(result)
    return result


def _seeds_from_meta(meta: Mapping[str, Any]) -> DaySeeds:
    return DaySeeds(
        date.fromisoformat(meta["prev_day"]) if meta.get("prev_day") else None,
        tuple(meta["prev_full_hl"]) if meta.get("prev_full_hl") else None,
        date.fromisoformat(meta["prev_ny_day"]) if meta.get("prev_ny_day") else None,
        tuple(meta["prev_ny_hl"]) if meta.get("prev_ny_hl") else None,
    )


def _next_seeds(day: str, metadata: Mapping[str, Any]) -> DaySeeds:
    entering = _seeds_from_meta(metadata["seeds"])
    if not metadata.get("day_hl"):
        return entering
    return DaySeeds(
        date.fromisoformat(day),
        tuple(metadata["day_hl"]),
        date.fromisoformat(day) if metadata.get("ny_hl") else entering.prev_ny_day,
        tuple(metadata["ny_hl"]) if metadata.get("ny_hl") else entering.prev_ny_hl,
    )


def _inspect_day(
    day: str, cfg: IfvgCaptureConfig, definition: Mapping[str, Any], expected_seeds: DaySeeds
) -> dict[str, Any]:
    footers, hashes = [], {}
    for kind, path in (("bars", cfg.bars_path(day)), ("levels", cfg.levels_path(day))):
        metadata = json.loads((pq.read_metadata(path).metadata or {}).get(_META_KEY, b"{}"))
        footers.append(metadata)
        hashes[f"{kind}_sha256"] = file_sha256(path)
    metadata = footers[0]
    if footers[1] != metadata:
        raise ValueError(f"prepared bars and levels have different cache stamps for {day}")
    required = {
        "artifact_schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifacts_tag": cfg.artifacts_tag(),
        "anchor_policy": HTF_ANCHOR_POLICY,
        "source_access_policy": "explicit_allowlist_before_path_v1",
        "source_allowlist_sha256": definition["source_allowlist_sha256"],
        "seeds": expected_seeds.meta(),
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError(f"prepared artifact provenance or seed lineage differs for {day}")
    return {**hashes, "metadata": metadata}


@dataclass(frozen=True)
class PreparedStoreRegistration:
    path: Path
    registration_id: str
    definition: Mapping[str, Any]
    days: Mapping[str, Mapping[str, Any]]

    @property
    def owned_dates(self) -> tuple[str, ...]:
        return tuple(self.definition["owned_dates"])

    @property
    def source_dates(self) -> tuple[str, ...]:
        return tuple(self.definition["source_dates"])

    def cfg_for(self, cfg: IfvgCaptureConfig) -> IfvgCaptureConfig:
        if cfg.artifacts_tag() != self.definition["artifacts_tag"]:
            raise ValueError("registered store uses a different artifact tag")
        if Path(cfg.data_dir).resolve() != Path(self.definition["raw_data_dir"]):
            raise ValueError("registered store uses a different raw source root")
        return replace(
            cfg,
            artifact_cache_dir=Path(self.definition["artifact_cache_dir"]),
            prepared_store_registry_paths=(),
        )


def _save_registration(
    path: Path, definition: Mapping[str, Any], days: Mapping[str, Mapping[str, Any]]
) -> PreparedStoreRegistration:
    if set(definition["preparation_dates"]) != set(days):
        raise ValueError("registered store needs a receipt for every preparation date")
    body = {"definition": dict(definition), "days": dict(days)}
    _write_json(path, {**body, "registration_id": _canonical_sha256(body)}, immutable=True)
    return load_prepared_store(path)


def _validate_receipts(definition: Mapping[str, Any], days: Mapping[str, Any]) -> None:
    expected = _seeds_from_meta(definition["initial_day_seeds"])
    if expected != DaySeeds(None, None, None, None):
        raise ValueError("registered preparation segments require their recorded cold start")
    for day in definition["preparation_dates"]:
        metadata = days[day]["metadata"]
        if (
            metadata.get("seeds") != expected.meta()
            or metadata.get("source_allowlist_sha256") != definition["source_allowlist_sha256"]
            or metadata.get("artifacts_tag") != definition["artifacts_tag"]
        ):
            raise ValueError(f"registered preparation seed or source lineage differs for {day}")
        expected = _next_seeds(day, metadata)


def load_prepared_store(path: Path) -> PreparedStoreRegistration:
    path = Path(path).resolve()
    stat = path.stat()
    return _load_prepared_store_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _load_prepared_store_cached(path: str, _mtime_ns: int, _size: int) -> PreparedStoreRegistration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {"definition": payload["definition"], "days": payload["days"]}
    if payload["registration_id"] != _canonical_sha256(body):
        raise ValueError("prepared-store registration hash does not match its content")
    _validate_definition(payload["definition"])
    if set(payload["definition"]["preparation_dates"]) != set(payload["days"]):
        raise ValueError("prepared-store registration has incomplete day evidence")
    _validate_receipts(payload["definition"], payload["days"])
    return PreparedStoreRegistration(
        Path(path), payload["registration_id"], payload["definition"], payload["days"]
    )


def _owner_for(day: str, registry_paths: tuple[Path, ...]) -> PreparedStoreRegistration:
    _ordered_dates((day,))  # cutoff before any date-derived path
    owners = [
        registration
        for registration in (load_prepared_store(path) for path in registry_paths)
        if day in registration.owned_dates
    ]
    if len(owners) != 1:
        raise ValueError(f"prepared date {day} must have exactly one registered store owner")
    return owners[0]


def prepared_artifact_day_dir(day: str, cfg: IfvgCaptureConfig) -> Path:
    owner = _owner_for(day, cfg.prepared_store_registry_paths)
    if owner.definition["artifacts_tag"] != cfg.artifacts_tag():
        raise ValueError("prepared cache tag differs from the registered store")
    return Path(owner.definition["artifact_cache_dir"]) / cfg.symbol / day


class PreparedStoreSourcePolicy(ExplorationDataPolicy):
    """Preparation authorization tied to one bounded, separate store definition."""

    def __init__(self, definition: Mapping[str, Any]) -> None:
        _validate_definition(definition)
        self.definition = dict(definition)
        super().__init__(allowlist=frozenset(definition["source_dates"]))

    def validate_registration_scope(self) -> None:
        _validate_definition(self.definition)
        if self.allowlist != frozenset(self.definition["source_dates"]):
            raise PermissionError("registered source policy differs from its preparation receipt")


class PreparedStoreReplayPolicy(ExplorationDataPolicy):
    """Task B's exact weekday warmup and declared cross-store replay dates."""

    def __init__(
        self,
        dates: Iterable[str],
        *,
        registry_paths: tuple[Path, ...],
        warmup_dates: tuple[str, ...] = TASK_B_WARMUP_DATES,
    ) -> None:
        self.replay_dates = self.dates = _ordered_dates(dates)
        if warmup_dates != TASK_B_WARMUP_DATES or self.dates[:10] != TASK_B_WARMUP_DATES:
            raise ValueError("Task B replay requires the exact June 2–13 ten-weekday warmup")
        self.warmup_dates = warmup_dates
        self.evaluation_dates = self.dates[10:]
        if self.evaluation_dates and self.evaluation_dates[0] != "2025-06-16":
            raise ValueError("Task B's first evaluable date must be June 16, 2025")
        self.registry_paths = tuple(Path(path).resolve() for path in registry_paths)
        self.registrations = tuple(load_prepared_store(path) for path in self.registry_paths)
        self._owners = {day: _owner_for(day, self.registry_paths) for day in self.dates}
        super().__init__(audit=DataAccessAudit(), allowlist=frozenset(self.dates))

    validate_date = ExplorationDataPolicy.authorize_date

    def validate_registration_scope(self) -> None:
        if self.allowlist != frozenset(self.dates):
            raise PermissionError("registered replay source scope changed")
        for registration in self.registrations:
            _validate_definition(registration.definition)
            current = load_prepared_store(registration.path)
            if current.registration_id != registration.registration_id:
                raise ValueError("prepared-store registration changed during replay")

    def for_day(self, day: str) -> _StoreArtifactPolicy:
        self.authorize_date(day)
        return _StoreArtifactPolicy(self, self._owners[day])

    def verify_registered_metadata(
        self, cfg: IfvgCaptureConfig
    ) -> tuple[tuple[str, ...], list[dict]]:
        """Read-only approval evidence: exact bytes/footers, without decoding bars."""
        self.validate_registration_scope()
        stamps = []
        for day in self.dates:
            owner = self._owners[day]
            segment_cfg = owner.cfg_for(cfg)
            receipt = owner.days[day]
            inspected = _inspect_day(
                day,
                segment_cfg,
                owner.definition,
                _seeds_from_meta(receipt["metadata"]["seeds"]),
            )
            if inspected != receipt:
                raise ValueError(f"registered prepared evidence changed for {day}")
            # This inspection opens named caches only. Record metadata plus the
            # file hashing reads, without manufacturing decoded row counts.
            for _kind in ("bars", "levels"):
                self.record_metadata_access(day)
                self.record_file_open(day)
            stamps.append(
                {
                    "day": day,
                    "tag": cfg.artifacts_tag(),
                    "registration_id": owner.registration_id,
                    "metadata": receipt["metadata"],
                    "bars_sha256": receipt["bars_sha256"],
                    "levels_sha256": receipt["levels_sha256"],
                }
            )
        provenance = tuple(
            sorted({day for owner in self.registrations for day in owner.source_dates})
        )
        return provenance, stamps

    def audit_dict(self) -> dict:
        payload = super().audit_dict()
        payload["policy"] = "ifsm_task_b_registered_prepared_stores_v1"
        payload["prepared_store_registration_ids"] = [r.registration_id for r in self.registrations]
        return payload


class _StoreArtifactPolicy:
    """One writing allowlist for footer matching; all I/O uses the replay scope."""

    def __init__(self, inner: PreparedStoreReplayPolicy, owner: PreparedStoreRegistration) -> None:
        self.inner, self.owner = inner, owner
        self.allowlist = frozenset(owner.source_dates)
        self.audit = inner.audit

    def validate_registration_scope(self) -> None:
        self.inner.validate_registration_scope()
        if self.allowlist != frozenset(self.owner.source_dates):
            raise PermissionError("registered artifact writing scope changed")

    def authorize_date(self, day: str) -> None:
        self.inner.authorize_date(day)
        if day not in self.owner.owned_dates:
            raise PermissionError("artifact date is outside its registered store ownership")

    def resolve_source_path(self, day: str, factory: Callable[[str], Path]) -> Path:
        self.authorize_date(day)
        return self.inner.resolve_source_path(day, factory)

    def record_metadata_access(self, day: str) -> None:
        self.authorize_date(day)
        self.inner.record_metadata_access(day)

    def record_file_open(self, day: str, *, rows: int = 0) -> None:
        self.authorize_date(day)
        self.inner.record_file_open(day, rows=rows)

    def record_rows_read(self, day: str, *, rows: int) -> None:
        self.authorize_date(day)
        self.inner.record_rows_read(day, rows=rows)

    def assert_zero_forbidden_access(self) -> None:
        self.inner.assert_zero_forbidden_access()


def is_registered_prepared_policy(policy: object) -> bool:
    if not isinstance(
        policy, PreparedStoreSourcePolicy | PreparedStoreReplayPolicy | _StoreArtifactPolicy
    ):
        return False
    policy.validate_registration_scope()
    return True


def registered_day_seeds(day: str, cfg: IfvgCaptureConfig) -> DaySeeds:
    owner = _owner_for(day, cfg.prepared_store_registry_paths)
    return _seeds_from_meta(owner.days[day]["metadata"]["seeds"])


def load_registered_day_artifacts(
    day: str, cfg: IfvgCaptureConfig, *, access_policy: PreparedStoreReplayPolicy
) -> DayArtifacts:
    access_policy.authorize_date(day)
    owner = _owner_for(day, cfg.prepared_store_registry_paths)
    policy = access_policy.for_day(day)
    if policy.owner.registration_id != owner.registration_id:
        raise PermissionError("capture configuration differs from its registered replay source")
    segment_cfg = owner.cfg_for(cfg)
    receipt = owner.days[day]
    for kind, factory in (("bars", segment_cfg.bars_path), ("levels", segment_cfg.levels_path)):
        path = policy.resolve_source_path(day, factory)
        policy.record_file_open(day)
        if file_sha256(path) != receipt[f"{kind}_sha256"]:
            raise ValueError(f"registered {kind} bytes changed for {day}")
    artifacts = load_day_artifacts(
        day,
        segment_cfg,
        expected_seeds=_seeds_from_meta(receipt["metadata"]["seeds"]),
        access_policy=policy,
    )
    if artifacts is None:
        raise ValueError(f"registered cache metadata or seed lineage changed for {day}")
    return artifacts


def register_existing_prepared_store(
    *,
    cfg: IfvgCaptureConfig,
    store_root: Path,
    preparation_dates: tuple[str, ...],
    source_dates: tuple[str, ...],
    owned_dates: tuple[str, ...],
    authorization_file: Path,
    catalog_path: Path,
) -> PreparedStoreRegistration:
    """Bind existing cache bytes and the entire original seed chain without writes."""
    if cfg.prepared_store_registry_paths:
        raise ValueError("existing store registration requires an explicit single artifact root")
    root = cfg.artifact_cache_dir if cfg.artifact_cache_dir is not None else cfg.data_dir
    definition = _definition(
        cfg,
        preparation_dates=_ordered_dates(preparation_dates),
        source_dates=_ordered_dates(source_dates),
        owned_dates=_ordered_dates(owned_dates),
        artifact_cache_dir=Path(root),
        catalog_path=catalog_path,
        authorization_file=authorization_file,
        mode="preserved_existing",
    )
    expected, receipts = DaySeeds(None, None, None, None), {}
    for day in preparation_dates:
        receipt = _inspect_day(day, cfg, definition, expected)
        receipts[day] = receipt
        expected = _next_seeds(day, receipt["metadata"])
    return _save_registration(Path(store_root) / "prepared_store.json", definition, receipts)


def task_b_preparation_coverage(coverage_csv: Path) -> tuple[dict[str, str], ...]:
    with Path(coverage_csv).open(encoding="utf-8-sig", newline="") as handle:
        rows = tuple(
            row
            for row in csv.DictReader(handle)
            if TASK_B_FIRST_PREPARATION_DAY <= row["date"] <= TASK_B_LAST_PREPARATION_DAY
        )
    expected = tuple(
        (date(2025, 6, 2) + timedelta(days=offset)).isoformat()
        for offset in range((date(2026, 1, 1) - date(2025, 6, 2)).days + 1)
        if (date(2025, 6, 2) + timedelta(days=offset)).weekday() < 5
    )
    if tuple(row["date"] for row in rows) != expected:
        raise ValueError("Task B coverage must contain the exact 154 weekday inventory")
    if any(row["symbol"] != "NQ" or row["raw_files"] not in {"", "mbp10.parquet"} for row in rows):
        raise ValueError("Task B coverage differs from the named NQ MBP-10 source inventory")
    if tuple(row["date"] for row in rows if not row["raw_files"]) != TASK_B_MISSING_SOURCE_DATES:
        raise ValueError("Task B coverage differs from the three recorded missing weekdays")
    return rows


def _hash_day_sources(
    day: str,
    cfg: IfvgCaptureConfig,
    policy: PreparedStoreSourcePolicy,
    digest_cache: dict[str, tuple[int, int, str]],
) -> list[dict[str, Any]]:
    """Hash the canonical reader's current/prior physical files, without a drain."""
    policy.resolve_source_path(day, cfg.day_dir)
    prior = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
    if prior in policy.allowlist:
        policy.resolve_source_path(prior, cfg.day_dir)
    source = DatabentoParquetSource.for_trading_day(
        Path(cfg.data_dir) / cfg.symbol,
        date.fromisoformat(day),
        requested_symbol=cfg.symbol,
        allowed_source_dates=frozenset(date.fromisoformat(value) for value in policy.allowlist),
    )
    result = []
    for index, path in enumerate(source.paths):
        physical_day = path.parent.name
        policy.resolve_source_path(physical_day, lambda _day, source_path=path: source_path)
        policy.record_metadata_access(physical_day)
        stat = path.stat()
        previous = digest_cache.get(str(path))
        if previous is None or previous[:2] != (stat.st_size, stat.st_mtime_ns):
            policy.record_file_open(physical_day)
            previous = (stat.st_size, stat.st_mtime_ns, file_sha256(path))
            digest_cache[str(path)] = previous
        start, end = source._window_for(index)
        result.append(
            {
                "physical_date": physical_day,
                "filename": path.name,
                "sha256": previous[2],
                "window_start_utc": start.isoformat() if start else None,
                "window_end_utc": end.isoformat() if end else None,
            }
        )
    return result


def prepare_task_b_store(
    *,
    coverage_csv: Path,
    cfg: IfvgCaptureConfig,
    store_root: Path,
    authorization_file: Path,
    on_day: Callable[[dict[str, Any]], None] | None = None,
) -> PreparedStoreRegistration:
    """Run/resume the exact unattended 2025 weekday chain in a separate destination.

    Checkpoints are written only after both cache files pass their saved stamp
    and seed checks. Missing source weekdays receive report entries and never
    enter the builder. Current/prior physical partitions follow the canonical
    reader within the authorized June 2–January 1 window; only weekday logical
    days are prepared. Existing complete day receipts are verified on resume.
    """
    rows = task_b_preparation_coverage(coverage_csv)
    root = Path(store_root).resolve()
    source_dates = tuple(
        (date(2025, 6, 2) + timedelta(days=offset)).isoformat()
        for offset in range((date(2026, 1, 1) - date(2025, 6, 2)).days + 1)
    )
    present_dates = tuple(row["date"] for row in rows if row["raw_files"])
    catalog = root / "preparation_catalog.json"
    cfg = replace(
        cfg,
        artifact_cache_dir=root / "artifacts",
        prepared_store_registry_paths=(),
        preparation_catalog_paths=(catalog,),
    )
    definition = _definition(
        cfg,
        preparation_dates=present_dates,
        source_dates=source_dates,
        owned_dates=present_dates,
        artifact_cache_dir=cfg.artifact_cache_dir,
        catalog_path=catalog,
        authorization_file=authorization_file,
        mode="separate_task_b_preparation",
    )
    definition["coverage_csv_sha256"] = file_sha256(Path(coverage_csv))
    _write_json(root / "preparation_definition.json", definition, immutable=True)
    policy = PreparedStoreSourcePolicy(definition)
    checkpoint = root / "preparation_days.json"
    reports = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    expected, receipts = DaySeeds(None, None, None, None), {}
    digest_cache: dict[str, tuple[int, int, str]] = {}
    run_start = perf_counter()
    for row in rows:
        day, started = row["date"], perf_counter()
        if not row["raw_files"]:
            report = {
                "date": day,
                "status": "missing",
                "missing_reason": "no_source_partition",
                "selected_instrument_id": None,
                "raw_symbol": None,
                "seconds": 0.0,
                "bar_count": 0,
                "partial_bar_count": 0,
                "warmup": day in TASK_B_WARMUP_DATES,
            }
        elif day in reports:
            report = reports[day]
            receipt = _inspect_day(day, cfg, definition, expected)
            if receipt != report["artifact_receipt"]:
                raise ValueError(f"prepared artifact changed since its checkpoint for {day}")
            if _hash_day_sources(day, cfg, policy, digest_cache) != report["source_files"]:
                raise ValueError(f"raw source changed since its preparation checkpoint for {day}")
            receipts[day] = receipt
            expected = _next_seeds(day, receipt["metadata"])
        else:
            source_path = policy.resolve_source_path(
                day, lambda value: cfg.day_dir(value) / "mbp10.parquet"
            )
            policy.record_metadata_access(day)
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"the supplied present Task B source partition is absent: {day}"
                )
            artifacts = build_day_artifacts(day, cfg, expected, access_policy=policy)
            write_day_artifacts(artifacts, cfg, access_policy=policy)
            receipt = _inspect_day(day, cfg, definition, expected)
            receipts[day] = receipt
            expected = _next_seeds(day, receipt["metadata"])
            source_files = _hash_day_sources(day, cfg, policy, digest_cache)
            source_sha256 = next(
                item["sha256"] for item in source_files if item["physical_date"] == day
            )
            entry = json.loads(catalog.read_text(encoding="utf-8"))["days"][day]
            report = {
                "date": day,
                "status": "prepared",
                "missing_reason": None,
                "selected_instrument_id": entry["selected_instrument_id"],
                "raw_symbol": entry.get("raw_symbol"),
                "seconds": perf_counter() - started,
                "bar_count": len(artifacts.bars),
                "partial_bar_count": sum(bar.is_partial for bar in artifacts.bars),
                "reader_warnings": list(artifacts.reader_warnings),
                "warmup": day in TASK_B_WARMUP_DATES,
                "source_sha256": source_sha256,
                "source_files": source_files,
                "artifact_receipt": receipt,
            }
        reports[day] = report
        _write_json(checkpoint, reports)
        if on_day is not None:
            on_day(report)
    policy.assert_zero_forbidden_access()
    _write_json(
        root / "preparation_run.json",
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "process_seconds": perf_counter() - run_start,
            "day_seconds": sum(report["seconds"] for report in reports.values()),
            "prepared_days": len(receipts),
            "missing_days": len(rows) - len(receipts),
            "source_access_audit": policy.audit_dict(),
        },
    )
    return _save_registration(root / "prepared_store.json", definition, receipts)
