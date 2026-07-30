"""IFVG capture configuration + the three cache identities.

Hash discipline (the ``dataset_config_hash`` idiom, extended):

* ``artifacts_tag`` — Phase-A identity. Profile-INDEPENDENT so reducer/profile
  iteration never invalidates the expensive drains. Folds the SC platform
  version, the session scheme identity, the timeframe set, and the pipeline
  tag.
* ``capture_tag`` — Phase-C identity: the full profile hash + strategy version
  + record schema + the artifacts tag.
* per-day TRUST is a separate axis from identity: an artifact/capture file is
  consumed only if its stamped ENTERING seeds match (``prev_full_hl`` +
  ``prev_ny_hl`` for artifacts; the ``IfvgDaySeed`` hash for captures).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from datetime import time
from pathlib import Path

from strategy_core import PLATFORM_VERSION
from strategy_core.constants import DEFAULT_TICK_SIZE, RESEARCH_SESSION_SCHEME
from strategy_core.strategies.ifvg_smc.records import IFVG_RECORD_SCHEMA_VERSION
from strategy_core.strategies.ifvg_smc.section import (
    IFVG_STRATEGY_ID,
    IFVG_STRATEGY_VERSION,
    IfvgSmcSection,
    default_ifvg_smc_section,
    ifvg_profile_hash,
)

# The ratified runtime->contract scheme adapter. It is module-internal to the
# touch section but is the exact function SC's own ifvg section.py uses to
# build its contract scheme (single-source; SC stays unmodified).
from strategy_core.strategies.touch_reversal.section import _contract_scheme_from_runtime
from strategy_core.types import SessionScheme, SessionWindow

__all__ = [
    "IfvgCaptureConfig",
    "DEFAULT_DATA_DIR",
    "SEALED_HOLDOUT_START",
    "custom_session_capture_config",
]

DEFAULT_DATA_DIR = Path("data/databento")

#: 2026-06-12..2026-07-10 stays SEALED this window (census precedent): captured
#: and funnel-counted, but excluded from label expectancies and ALL gate
#: training/evaluation — the future one-shot validation range.
SEALED_HOLDOUT_START = "2026-06-12"

_ARTIFACTS_PIPELINE_TAG = "ifvg_day_artifacts_v1"
_CAPTURE_PIPELINE_TAG = "ifvg_capture_pipeline_v1"


def _short_hash(payload: str, n: int = 16) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


def _windows_signature(scheme: SessionScheme) -> str:
    """Deterministic per-window times signature: ``name:start-end:xmid`` per
    sorted name. Folded into ``artifacts_tag`` ONLY for non-default schemes so
    the canonical tag (and the 624 warmed files) stays byte-identical."""
    return ",".join(
        f"{name}:{w.start.isoformat()}-{w.end.isoformat()}:{int(w.crosses_midnight)}"
        for name, w in sorted(scheme.sessions.items())
    )


@dataclass(frozen=True)
class IfvgCaptureConfig:
    symbol: str = "NQ"
    data_dir: Path = DEFAULT_DATA_DIR
    tick_size: float = DEFAULT_TICK_SIZE
    section: IfvgSmcSection = field(default_factory=default_ifvg_smc_section)
    #: First N chain days feed registries only; their rows carry is_warmup=True
    #: (census: the 4H registry stabilizes days 8-10).
    warmup_days: int = 10
    #: Runtime scheme Phase A is built with (bars trading-day slicing, level
    #: session ranges, NY-extremes stamping). MUST stay consistent with
    #: ``section.session_scheme`` — build customs via
    #: :func:`custom_session_capture_config` so both tags roll together.
    session_scheme: SessionScheme = field(default=RESEARCH_SESSION_SCHEME)

    @property
    def profile_hash(self) -> str:
        return ifvg_profile_hash(self.section)

    def timeframes_seconds(self) -> tuple[int, ...]:
        return self.section.timeframe_seconds()

    def artifacts_tag(self) -> str:
        scheme = self.session_scheme
        parts = [
            _ARTIFACTS_PIPELINE_TAG,
            PLATFORM_VERSION,
            scheme.timezone,
            scheme.trading_day_boundary.isoformat(),
            ",".join(sorted(scheme.sessions)),
            ",".join(str(s) for s in self.timeframes_seconds()),
            repr(self.tick_size),
        ]
        # Session-window TIMES fold in conditionally (tag-collision fix): the
        # default payload stays byte-identical, so the canonical tag never moves.
        if _windows_signature(scheme) != _windows_signature(RESEARCH_SESSION_SCHEME):
            parts.append(_windows_signature(scheme))
        payload = "|".join(parts)
        return _short_hash(payload)

    def capture_tag(self) -> str:
        payload = "|".join(
            (
                _CAPTURE_PIPELINE_TAG,
                PLATFORM_VERSION,
                IFVG_STRATEGY_ID,
                IFVG_STRATEGY_VERSION,
                str(IFVG_RECORD_SCHEMA_VERSION),
                self.profile_hash,
                self.artifacts_tag(),
            )
        )
        return _short_hash(payload)

    # ── per-day file locations ────────────────────────────────────────────────
    def day_dir(self, date_str: str) -> Path:
        return Path(self.data_dir) / self.symbol / date_str

    def bars_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_tbars_{self.artifacts_tag()}.parquet"

    def levels_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_levels_{self.artifacts_tag()}.parquet"

    def capture_path(self, date_str: str) -> Path:
        return self.day_dir(date_str) / f"ifvg_capture_{self.capture_tag()}.parquet"

    def seed_path(self, date_str: str) -> Path:
        """The IfvgDaySeed (pickle) written at the END of ``date_str``. Trust is
        the SC ``seed_hash`` stamped in the capture parquet, never the pickle
        bytes; the capture_tag in the name invalidates stale shapes."""
        return self.day_dir(date_str) / f"ifvg_seed_{self.capture_tag()}.pkl"


_CUSTOM_SESSION_NAMES = ("asia", "london", "ny")


def custom_session_capture_config(
    windows: dict[str, tuple[time, time]],
    base: IfvgCaptureConfig | None = None,
) -> IfvgCaptureConfig:
    """A capture config whose session windows carry custom TIMES (names fixed:
    asia/london/ny; timezone + trading_day_boundary fixed to the base scheme's).

    ``crosses_midnight`` is derived per window (``start > end``). Both identity
    axes roll together: the runtime ``session_scheme`` moves ``artifacts_tag``
    (conditional windows signature) and the section's contract scheme moves
    ``profile_hash`` -> ``capture_tag``.
    """
    base = base or IfvgCaptureConfig()
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
            name: SessionWindow(start=start, end=end, crosses_midnight=start > end)
            for name, (start, end) in sorted(windows.items())
        },
        closed_window=base_scheme.closed_window,
    )
    section = base.section.model_copy(
        update={"session_scheme": _contract_scheme_from_runtime(scheme)}
    )
    return replace(base, section=section, session_scheme=scheme)
