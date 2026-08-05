"""IFVG capture configuration and cache identities.

The default configuration is the executable IFVG v2 document profile.  The
old warmed v1 cache identities remain available only through
``legacy_ifvg_capture_config``; they are never silently reused by v2.

Identity discipline:

* ``artifacts_tag`` identifies profile-independent bars and level timelines.
  It includes the platform version, named anchor, session scheme, timeframes,
  tick size, and artifact pipeline version.
* ``capture_tag`` identifies the sequential reducer stream.  It additionally
  includes the resolved profile hash and record/capture schema versions.
* cache trust is separate from identity: entering and exiting state hashes are
  checked for every day in a capture chain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path

from strategy_core import PLATFORM_VERSION
from strategy_core.candles._buckets import HTF_ANCHOR_POLICY
from strategy_core.constants import (
    DEFAULT_TICK_SIZE,
    IFVG_DOC_SESSION_SCHEME,
    RESEARCH_SESSION_SCHEME,
)
from strategy_core.strategies.ifvg_smc.context_config import (
    ContextFeatureConfig,
    context_config_hash,
    feature_schema_hash,
)
from strategy_core.strategies.ifvg_smc.records import IFVG_RECORD_SCHEMA_VERSION
from strategy_core.strategies.ifvg_smc.section import (
    IFVG_STRATEGY_ID,
    IFVG_STRATEGY_VERSION,
    IfvgSmcSection,
    default_ifvg_smc_section,
    ifvg_profile_hash,
    legacy_ifvg_smc_section,
)
from strategy_core.strategies.touch_reversal.section import (
    _contract_scheme_from_runtime,
)
from strategy_core.types import SessionScheme, SessionWindow

from .contracts import IFVG_CAPTURE_SCHEMA_VERSION

__all__ = [
    "IfvgCaptureConfig",
    "DEFAULT_DATA_DIR",
    "V2_DATASET_DIR",
    "SEALED_HOLDOUT_START",
    "LEGACY_DEFAULT_ARTIFACTS_TAG",
    "LEGACY_DEFAULT_CAPTURE_TAG",
    "legacy_ifvg_capture_config",
    "custom_session_capture_config",
    "IfvgV3CaptureConfig",
    "V3_DATASET_DIR",
    "ACCEPTED_V2_DATASET_ID",
    "ACCEPTED_V2_MANIFEST_SHA256",
    "FSM_AUDIT_ACCEPTED_V2_DATASET_ID",
    "FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256",
    "FSM_AUDIT_DATASET_DIR",
]

DEFAULT_DATA_DIR = Path("data/databento")
V2_DATASET_DIR = Path("data/ifvg_datasets/v2")
V3_DATASET_DIR = Path("data/ifvg_datasets/v3")

ACCEPTED_V2_DATASET_ID = (
    "49902280956e2f6448799b16018e65b197387c58c6ae6109d17b5a76386c9a1a"
)
ACCEPTED_V2_MANIFEST_SHA256 = (
    "e635d064225444f02e2f11277d01912ee96e87bf24618aaa363cc86d53af7c5f"
)

# FSM-audit lane pins — the FINAL-REVIEW accepted v2 dataset (per
# reports/ifvg_final_review/2314066…/IFVG_LAB_FINAL_VERIFICATION.md), which is
# NOT the same lineage as ``ACCEPTED_V2_DATASET_ID`` (the v3-baseline pin
# above). The discrepancy is deliberate and registered in
# docs/ifvg/IFVG_FSM_AUDITABILITY_OPEN_DECISIONS.md: the audit lane pins the
# final-review identity as its own constants and never reuses or moves the v3
# lane's pin. The exact-parity gate verifies both ids against the on-disk
# manifest before any audit artifact is saved.
FSM_AUDIT_ACCEPTED_V2_DATASET_ID = (
    "143b510f8a73896072f44e08f331ef5156e85eb8e5124d25bdf441c4fb6b2ac7"
)
FSM_AUDIT_ACCEPTED_V2_MANIFEST_SHA256 = (
    "b089dfadf44253b7071882cc9577feeacd97e532f4fd7086fe2fcbac39e60c22"
)
FSM_AUDIT_DATASET_DIR = Path("data/ifvg_datasets/fsm_audit/v1")

# Retained for v1 readers only.  Repaired-v2 data access denies this entire
# range before path construction; no repair workflow may invoke sealed
# evaluation.
SEALED_HOLDOUT_START = "2026-06-12"

_ARTIFACTS_PIPELINE_TAG = "ifvg_day_artifacts_v2"
_CAPTURE_PIPELINE_TAG = "ifvg_capture_pipeline_v2"

# Literal read-only reproduction identities.  Strategy-Core v2 versions and
# hashes must never move or overwrite the warmed v1 files under these names.
LEGACY_DEFAULT_ARTIFACTS_TAG = "466b5fe8e7952ecd"
LEGACY_DEFAULT_CAPTURE_TAG = "2a40b18e0b273ee0"


def _short_hash(payload: str, n: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def _windows_signature(scheme: SessionScheme) -> str:
    return ",".join(
        f"{name}:{window.start.isoformat()}-{window.end.isoformat()}:"
        f"{int(window.crosses_midnight)}"
        for name, window in sorted(scheme.sessions.items())
    )


@dataclass(frozen=True)
class IfvgCaptureConfig:
    symbol: str = "NQ"
    data_dir: Path = DEFAULT_DATA_DIR
    tick_size: float = DEFAULT_TICK_SIZE
    section: IfvgSmcSection = field(default_factory=default_ifvg_smc_section)
    # First ten available chain days are registry warmup only.
    warmup_days: int = 10
    session_scheme: SessionScheme = field(default=IFVG_DOC_SESSION_SCHEME)
    # ``legacy_v1`` can locate old files but cannot enter any v2 writer.
    identity_lane: str = "v2"

    def __post_init__(self) -> None:
        if self.identity_lane not in {"v2", "legacy_v1"}:
            raise ValueError("identity_lane must be 'v2' or 'legacy_v1'")
        if self.identity_lane == "legacy_v1" and self.section.execution_enabled:
            raise ValueError("legacy_v1 capture configuration must be non-executable")

    @property
    def profile_hash(self) -> str:
        return ifvg_profile_hash(self.section)

    def timeframes_seconds(self) -> tuple[int, ...]:
        return self.section.timeframe_seconds()

    def artifacts_tag(self) -> str:
        if self.identity_lane == "legacy_v1":
            return LEGACY_DEFAULT_ARTIFACTS_TAG
        scheme = self.session_scheme
        parts = [
            _ARTIFACTS_PIPELINE_TAG,
            PLATFORM_VERSION,
            HTF_ANCHOR_POLICY,
            scheme.timezone,
            scheme.trading_day_boundary.isoformat(),
            ",".join(sorted(scheme.sessions)),
            ",".join(str(seconds) for seconds in self.timeframes_seconds()),
            repr(self.tick_size),
        ]
        if _windows_signature(scheme) != _windows_signature(IFVG_DOC_SESSION_SCHEME):
            parts.append(_windows_signature(scheme))
        return _short_hash("|".join(parts))

    def capture_tag(self) -> str:
        if self.identity_lane == "legacy_v1":
            return LEGACY_DEFAULT_CAPTURE_TAG
        payload = "|".join(
            (
                _CAPTURE_PIPELINE_TAG,
                PLATFORM_VERSION,
                IFVG_STRATEGY_ID,
                IFVG_STRATEGY_VERSION,
                str(IFVG_RECORD_SCHEMA_VERSION),
                str(IFVG_CAPTURE_SCHEMA_VERSION),
                self.profile_hash,
                self.artifacts_tag(),
            )
        )
        return _short_hash(payload)

    def day_dir(self, date_str: str) -> Path:
        return Path(self.data_dir) / self.symbol / date_str

    def bars_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_tbars_{self.artifacts_tag()}.parquet"

    def levels_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_levels_{self.artifacts_tag()}.parquet"

    def capture_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_capture_{self.capture_tag()}.parquet"

    def seed_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_seed_{self.capture_tag()}.pkl"


@dataclass(frozen=True)
class IfvgV3CaptureConfig:
    """Generation-3 adapter; v2 strategy/profile identity stays untouched."""

    core: IfvgCaptureConfig = field(default_factory=IfvgCaptureConfig)
    context: ContextFeatureConfig = field(default_factory=ContextFeatureConfig)
    accepted_v2_dataset_id: str = ACCEPTED_V2_DATASET_ID
    accepted_v2_manifest_sha256: str = ACCEPTED_V2_MANIFEST_SHA256

    def __post_init__(self) -> None:
        if self.core.identity_lane != "v2":
            raise ValueError("IFVG v3 requires the repaired v2 core identity lane")
        if self.core.warmup_days != 10:
            raise ValueError("IFVG v3 freezes the first ten available dates as warmup")
        if len(self.accepted_v2_dataset_id) != 64:
            raise ValueError("accepted v2 dataset ID must be a full SHA-256")
        if len(self.accepted_v2_manifest_sha256) != 64:
            raise ValueError("accepted v2 manifest hash must be a full SHA-256")

    @property
    def feature_schema_hash(self) -> str:
        return feature_schema_hash(self.context)

    @property
    def context_config_hash(self) -> str:
        return context_config_hash(self.context)

    @property
    def accepted_v2_exploration_dir(self) -> Path:
        return V2_DATASET_DIR / self.accepted_v2_dataset_id / "exploration"


def legacy_ifvg_capture_config(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    symbol: str = "NQ",
) -> IfvgCaptureConfig:
    """Return the read-only v1 candidate-stream configuration."""
    return IfvgCaptureConfig(
        symbol=symbol,
        data_dir=data_dir,
        section=legacy_ifvg_smc_section(),
        session_scheme=RESEARCH_SESSION_SCHEME,
        identity_lane="legacy_v1",
    )


_CUSTOM_SESSION_NAMES = ("asia", "london", "ny")


def custom_session_capture_config(
    windows: dict[str, tuple[time, time]],
    base: IfvgCaptureConfig | None = None,
) -> IfvgCaptureConfig:
    """Build a v2 custom-session profile with both identity axes updated."""
    base = base or IfvgCaptureConfig()
    if base.identity_lane != "v2":
        raise ValueError("custom session recapture is unavailable for legacy_v1")
    if sorted(windows) != sorted(_CUSTOM_SESSION_NAMES):
        raise ValueError(
            f"custom session windows must define exactly {_CUSTOM_SESSION_NAMES}, "
            f"got {sorted(windows)}"
        )
    base_scheme = base.session_scheme
    scheme = SessionScheme(
        timezone=base_scheme.timezone,
        trading_day_boundary=base_scheme.trading_day_boundary,
        sessions={
            name: SessionWindow(
                start=start,
                end=end,
                crosses_midnight=start > end,
            )
            for name, (start, end) in sorted(windows.items())
        },
        closed_window=base_scheme.closed_window,
    )
    section = base.section.model_copy(
        update={"session_scheme": _contract_scheme_from_runtime(scheme)}
    )
    return replace(base, section=section, session_scheme=scheme)
