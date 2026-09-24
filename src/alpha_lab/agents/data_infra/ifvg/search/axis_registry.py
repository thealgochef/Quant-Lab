"""Typed search-axis and axis-value registry (CONTRACTS_AND_SCHEMAS.md §2).

Every ``IfvgSmcSection`` field except ``profile_name`` (owned by canonical
naming) is classified here; the classification is seeded from the *actual*
doc-default baseline section so registry truth can never drift from code
truth. Values are typed and individually ratifiable; blocked and inert axes
fail closed before enumeration; the UI never exposes raw ``section_overrides``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import field_validator, model_validator
from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

from ..presentation.axis_values import format_axis_value
from .identities import (
    FrozenContract,
    canonical_contract_sha256,
    deep_freeze,
)

__all__ = [
    "AxisClassification",
    "DimensionValueSpec",
    "RegisteredAxisValue",
    "CompositeAxisValue",
    "SearchAxisSpec",
    "SEARCH_AXIS_REGISTRY_V1",
    "AXIS_VALUE_REGISTRY_V1",
    "registry_sha256",
    "assert_axes_authorized",
    "AxisAuthorizationError",
    "resolve_axis_overrides",
]


class AxisClassification(StrEnum):
    LOCKED_INVARIANT = "locked_correctness_invariant"
    THESIS_DEFINING = "strategy_thesis_defining"
    APPROVED_SEARCH_AXIS = "approved_search_axis"
    RISK_POLICY_AXIS = "risk_policy_axis"
    MEASUREMENT_ONLY = "measurement_only"
    BLOCKED = "blocked"
    EXPERIMENTAL = "experimental"


class DimensionValueSpec(FrozenContract):
    value_schema_id: str
    json_schema_hash: str


class RegisteredAxisValue(FrozenContract):
    value_id: str
    axis_technical_key: str
    payload: Any
    human_label: str
    capability_status: Literal[
        "available",
        "blocked_reducer_hardcoded",
        "blocked_inert_field",
        "blocked_legacy_field",
        "blocked_pending_owner_policy_review",
        "planned",
    ]
    owner_ratification_status: Literal["ratified", "pending", "not_required"]
    ratification_evidence_ref: str | None
    dependencies: tuple[str, ...] = ()
    incompatibilities: tuple[str, ...] = ()
    expected_replay_effect: str

    @field_validator("payload")
    @classmethod
    def _freeze_payload(cls, value: Any) -> Any:
        return deep_freeze(value)


class CompositeAxisValue(FrozenContract):
    """A dependent field group selected atomically as one named policy."""

    value_id: str
    human_label: str
    member_values: tuple[tuple[str, Any], ...]
    capability_status: Literal[
        "available",
        "blocked_reducer_hardcoded",
        "blocked_inert_field",
        "blocked_legacy_field",
        "blocked_pending_owner_policy_review",
        "planned",
    ]
    owner_ratification_status: Literal["ratified", "pending", "not_required"]
    ratification_evidence_ref: str | None
    dependencies: tuple[str, ...] = ()
    incompatibilities: tuple[str, ...] = ()
    expected_replay_effect: str

    @field_validator("member_values", mode="before")
    @classmethod
    def _freeze_members(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            return deep_freeze(value)
        return deep_freeze(dict(value))

    def member_mapping(self) -> dict[str, Any]:
        return dict(self.member_values)


class SearchAxisSpec(FrozenContract):
    technical_key: str
    human_label: str
    description: str
    classification: AxisClassification
    value_spec: DimensionValueSpec
    registered_values: tuple[str, ...]
    baseline_value_id: str
    requires_full_sequential_replay: bool
    changes_capture_artifacts: bool
    dependencies: tuple[str, ...] = ()
    expected_artifact_effect: str

    @model_validator(mode="after")
    def _baseline_registered(self):
        if self.registered_values and self.baseline_value_id not in self.registered_values:
            raise ValueError(
                f"axis {self.technical_key}: baseline value is not registered"
            )
        return self


class AxisAuthorizationError(PermissionError):
    """Raised when an axis value cannot enter enumeration (fail-closed)."""


def _schema(value_schema_id: str, description: dict) -> DimensionValueSpec:
    return DimensionValueSpec(
        value_schema_id=value_schema_id,
        json_schema_hash=canonical_contract_sha256(description),
    )


_NULLABLE_POS_INT = _schema(
    "nullable_positive_int_v1", {"type": ["integer", "null"], "exclusiveMinimum": 0}
)
_POS_INT = _schema("positive_int_v1", {"type": "integer", "exclusiveMinimum": 0})
_NONNEG_INT = _schema("nonnegative_int_v1", {"type": "integer", "minimum": 0})
_POS_FLOAT = _schema("positive_float_v1", {"type": "number", "exclusiveMinimum": 0})
_BOOL = _schema("bool_v1", {"type": "boolean"})
_STR = _schema("string_literal_v1", {"type": "string"})
_STR_TUPLE = _schema(
    "string_tuple_v1", {"type": "array", "items": {"type": "string"}}
)
_RECORD = _schema("record_v1", {"type": "object"})
_COMPOSITE = _schema("composite_group_v1", {"type": "object", "composite": True})


_BASELINE_SECTION = default_ifvg_smc_section()
_B = _BASELINE_SECTION.model_dump(mode="json")


def _vid(key: str, token: str) -> str:
    return f"{key}.{token}"


def _token(value: Any) -> str:
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, (list, tuple)):
        return "-".join(str(item) for item in value) or "empty"
    return str(value).replace(" ", "_")


_AXIS_SPECS: dict[str, SearchAxisSpec] = {}
_VALUE_REGISTRY: dict[str, RegisteredAxisValue | CompositeAxisValue] = {}


def _register_axis(
    key: str,
    *,
    label: str,
    description: str,
    classification: AxisClassification,
    value_spec: DimensionValueSpec,
    replay: bool,
    artifacts: bool = False,
    artifact_effect: str = "none — day bar/level artifacts unchanged",
    dependencies: tuple[str, ...] = (),
    baseline_payload: Any = "__from_section__",
    baseline_capability: str = "available",
    baseline_ratification: str = "not_required",
    baseline_reason: str | None = None,
    baseline_token: str | None = None,
    extra_values: tuple[RegisteredAxisValue | CompositeAxisValue, ...] = (),
) -> None:
    if baseline_payload == "__from_section__":
        baseline_payload = _B[key]
    baseline_id = _vid(key, baseline_token or _token(baseline_payload))
    # Uniform ratification posture (owner decision 2): EVERY value of a
    # searchable axis — including the accepted doc-default baseline — needs
    # value-level ratification before entering a REAL charter. Non-searchable
    # classifications keep not_required (they can never be selected at all).
    if (
        classification is AxisClassification.APPROVED_SEARCH_AXIS
        and baseline_capability == "available"
        and baseline_ratification == "not_required"
    ):
        baseline_ratification = "pending"
    baseline = RegisteredAxisValue(
        value_id=baseline_id,
        axis_technical_key=key,
        payload=baseline_payload,
        human_label=f"{label} — default: {format_axis_value(key, baseline_payload)}",
        capability_status=baseline_capability,  # type: ignore[arg-type]
        owner_ratification_status=baseline_ratification,  # type: ignore[arg-type]
        ratification_evidence_ref=baseline_reason,
        expected_replay_effect="baseline behavior (no change)",
    )
    values = (baseline, *extra_values)
    for value in values:
        if value.value_id in _VALUE_REGISTRY:
            raise ValueError(f"duplicate axis value id {value.value_id}")
        _VALUE_REGISTRY[value.value_id] = value
    _AXIS_SPECS[key] = SearchAxisSpec(
        technical_key=key,
        human_label=label,
        description=description,
        classification=classification,
        value_spec=value_spec,
        registered_values=tuple(value.value_id for value in values),
        baseline_value_id=baseline_id,
        requires_full_sequential_replay=replay,
        changes_capture_artifacts=artifacts,
        dependencies=dependencies,
        expected_artifact_effect=artifact_effect,
    )


def _locked(key: str, label: str, why: str) -> None:
    _register_axis(
        key,
        label=label,
        description=why,
        classification=AxisClassification.LOCKED_INVARIANT,
        value_spec=_RECORD if isinstance(_B[key], (dict, list)) else _STR,
        replay=False,
        baseline_token="baseline",
    )


def _thesis(key: str, label: str, why: str) -> None:
    _register_axis(
        key,
        label=label,
        description=why,
        classification=AxisClassification.THESIS_DEFINING,
        value_spec=_RECORD if isinstance(_B[key], (dict, list)) else _STR,
        replay=True,
        baseline_token="baseline",
    )


def _blocked(key: str, label: str, capability: str, why: str, spec: DimensionValueSpec) -> None:
    _register_axis(
        key,
        label=label,
        description=why,
        classification=AxisClassification.BLOCKED,
        value_spec=spec,
        replay=True,
        baseline_capability=capability,
        baseline_ratification="not_required",
        baseline_token="baseline",
    )


def _search_values(
    key: str, payloads: tuple[Any, ...], *, effect: str, evidence: str | None = None,
    readable_labels: bool = False,
) -> tuple[RegisteredAxisValue, ...]:
    return tuple(
        RegisteredAxisValue(
            value_id=_vid(key, _token(payload)),
            axis_technical_key=key,
            payload=payload,
            human_label=(
                format_axis_value(key, payload)
                if readable_labels else f"{key} = {_token(payload)}"
            ),
            capability_status="available",
            owner_ratification_status="pending",
            ratification_evidence_ref=evidence,
            expected_replay_effect=effect,
        )
        for payload in payloads
    )


# ── locked correctness invariants (no widget; visible for audit) ─────────────
_locked("runnable", "Profile runnable flag", "capability escalation is forbidden")
_locked("execution_enabled", "Execution enabled", "capability escalation is forbidden")
_locked("non_runnable_reason", "Non-runnable reason", "capability metadata, not a knob")
_locked("qualification_mode", "Qualification mode", "resolver forces custom_profile on overrides")
_locked("causality_parent", "Parent causality", "locked causality triple (PIT correctness)")
_locked("causality_opposing", "Opposing causality", "locked causality triple (PIT correctness)")
_locked("causality_entry", "Entry causality", "locked causality triple (PIT correctness)")
_locked("anchor_policy", "HTF anchor policy", "trading_day_18et_elapsed_v1 is the as-built anchor")

# ── thesis-defining, locked in v1 ────────────────────────────────────────────
_thesis("session_scheme", "Session scheme", "thesis-locked in v1 (stamp + gating clock)")
_register_axis(
    "doc_sessions",
    label="Doc session windows",
    description=(
        "Entry-window definitions supplied by the selected session preset; "
        "only registered Enabled entry sessions presets may replace them."
    ),
    classification=AxisClassification.MEASUREMENT_ONLY,
    value_spec=_RECORD,
    replay=True,
    baseline_token="baseline",
    dependencies=("enabled_entry_sessions",),
)
_thesis("entry_families", "Declared entry families", "thesis-locked in v1")
_thesis("entry_family", "Entry family", "fresh_fvg_continuation is the executable thesis")
_thesis("label_family", "Label family", "candidate label semantics are thesis-locked")
_thesis("enable_longs", "Long side enabled", "long-only thesis in v1")

# ── searchable clocks ────────────────────────────────────────────────────────
# These optional fields exist in the isolated repaired IFSM runtime. The normal
# installed Core may predate them: never advertise a setting it cannot execute.
if "setup_timeout_1m_bars" in _B:
    _register_axis(
        "setup_timeout_1m_bars",
        label="Whole-setup lifetime (1m bars)",
        description=(
            "Maximum processed one-minute bars from the original HTF activation "
            "while waiting for entry, across S1–S4. Parent replacement does not "
            "restart this clock. Expiry occurs on the first bar beyond the limit; "
            "an open position keeps its existing exit rules. None = no lifetime limit."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_NULLABLE_POS_INT,
        replay=True,
        extra_values=_search_values(
            "setup_timeout_1m_bars", (180, 240),
            effect="bounds pre-entry occupation from activation without changing open trades",
            evidence=(
                "Completed IFSM frequency studies, 2026-09-09; exploratory evidence "
                "does not promote a configuration or authorize another study."
            ),
            readable_labels=True,
        ),
    )

_register_axis(
    "parent_retest_timeout_1m_bars",
    label="Parent retest staleness timeout (1m bars)",
    description=(
        "How many 1m bars S1 may wait for a retest after selecting the current "
        "parent. Replacing the parent restarts the clock. "
        "None = no timeout for this wait."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
    baseline_ratification="pending",
    extra_values=_search_values(
        "parent_retest_timeout_1m_bars",
        (60, 90, 120),
        effect="stale setups release the single FSM slot earlier",
    ) + _search_values(
        "parent_retest_timeout_1m_bars",
        (240, 360, 480),
        effect="stale setups release the single FSM slot earlier",
        evidence=(
            "240/480 exist as prepared or implemented variants (implementation "
            "evidence only — no owner decision ratifies them as search values)"
        ),
    ),
)
_register_axis(
    "opposing_timeout_1m_bars",
    label="Opposing-FVG timeout (1m bars)",
    description=(
        "Maximum one-minute bars to wait for an opposing FVG after the parent "
        "retest locks the setup (S2). None = no timeout for this wait."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
    extra_values=_search_values(
        "opposing_timeout_1m_bars",
        (60, 90, 120),
        effect="setups waiting too long for an opposing FVG expire and release the FSM slot",
    ),
)
_register_axis(
    "inversion_timeout_1m_bars",
    label="Inversion timeout (1m bars)",
    description="S3→S4 wait bound; None = unbounded doc-default.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
)
_register_axis(
    "post_inversion_expiry_1m_bars_max",
    label="Post-inversion entry expiry (1m bars)",
    description="Maximum bars after inversion during which entries may trigger.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
)
_register_axis(
    "parent_reaction_window_parent_bars",
    label="Parent reaction window (parent bars)",
    description="Reaction clock in each parent's own timeframe.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
    extra_values=_search_values(
        "parent_reaction_window_parent_bars", (20,),
        effect="shortens admission of newly formed parents on each parent timeframe clock",
        evidence="Completed ParentReaction20 study, 2026-09-09; exploratory, not approval.",
        readable_labels=True,
    ),
)

if "parent_replacement_policy" in _B:
    _register_axis(
        "parent_replacement_policy",
        label="Provisional parent replacement",
        description=(
            "Before lock, rank new eligible parents by timeframe, confirmation time "
            "and FVG ID. The preservation alternative keeps an eligible selected "
            "parent and its retest clock; it creates no fallback list. Physical or "
            "structural clearing still permits only eligible new arrivals."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_STR,
        replay=True,
        extra_values=_search_values(
            "parent_replacement_policy", ("preserve_selected",),
            effect="keeps the provisional incumbent instead of taking higher-ranked arrivals",
            evidence=(
                "Completed PreserveSelected study, 2026-09-09; exploratory and "
                "not an approved configuration."
            ),
            readable_labels=True,
        ),
    )

if "parent_retest_depth_policy" in _B:
    _register_axis(
        "parent_retest_depth_policy",
        label="First parent retest depth admission",
        description=(
            "After physical invalidation and expiry checks, the strict alternative "
            "rejects and clears a provisional parent when its first later touch "
            "reaches half its width (CE): 2 × penetration >= width. A shallower "
            "touch locks immediately. The activation clocks continue; rejected "
            "parents are never reused and eligible new arrivals may still be selected."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_STR,
        replay=True,
        extra_values=_search_values(
            "parent_retest_depth_policy", ("strictly_before_ce",),
            effect="admits only a first live retouch strictly before the existing CE midpoint",
            evidence=(
                "Completed ShallowFirstRetest study, 2026-09-09; exploratory and "
                "not an approved configuration."
            ),
            readable_labels=True,
        ),
    )

# ── searchable distances ─────────────────────────────────────────────────────
_register_axis(
    "parent_htf_distance_ticks_max",
    label="Parent↔HTF distance cap (ticks)",
    description="Locality bound between parent FVG and the tapped HTF zone.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NONNEG_INT,
    replay=True,
    extra_values=_search_values(
        "parent_htf_distance_ticks_max", (160,),
        effect="admits parent intervals up to 160 ticks from the selected HTF interval",
        evidence="Completed ParentDistance160 study, 2026-09-09; exploratory, not approval.",
        readable_labels=True,
    ),
)
_register_axis(
    "opposing_parent_distance_ticks_max",
    label="Opposing↔parent distance cap (ticks)",
    description="Locality bound between the opposing FVG and the locked parent.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NONNEG_INT,
    replay=True,
    extra_values=_search_values(
        "opposing_parent_distance_ticks_max", (160,),
        effect="admits opposing intervals up to 160 ticks from the locked parent interval",
        evidence="Completed OpposingDistance160 study, 2026-09-09; exploratory, not approval.",
        readable_labels=True,
    ),
)
_register_axis(
    "entry_parent_distance_ticks_max",
    label="Entry↔parent distance cap (ticks)",
    description=(
        "Dependent member of the entry_near_parent composite group — never "
        "selectable standalone (MEASUREMENT_ONLY enforces it structurally)."
    ),
    classification=AxisClassification.MEASUREMENT_ONLY,
    value_spec=_NULLABLE_POS_INT.model_copy(),
    replay=True,
    dependencies=("entry_near_parent",),
)
_register_axis(
    "entry_near_parent",
    label="Entry-near-parent gate",
    description="Composite gate: entry must trigger within a distance of the parent.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_COMPOSITE,
    replay=True,
)
_composite_near_40 = CompositeAxisValue(
    value_id="entry_near_parent.within_40",
    human_label="Entry within 40 ticks of parent",
    member_values={"entry_near_parent": True, "entry_parent_distance_ticks_max": 40},
    capability_status="available",
    owner_ratification_status="pending",
    ratification_evidence_ref=None,
    dependencies=("entry_near_parent", "entry_parent_distance_ticks_max"),
    expected_replay_effect="entries far from the parent are suppressed",
)
_VALUE_REGISTRY[_composite_near_40.value_id] = _composite_near_40
_AXIS_SPECS["entry_near_parent"] = _AXIS_SPECS["entry_near_parent"].model_copy(
    update={
        "registered_values": (
            *_AXIS_SPECS["entry_near_parent"].registered_values,
            _composite_near_40.value_id,
        )
    }
)

# ── searchable retention / capture geometry / trade geometry ─────────────────
_register_axis(
    "htf_registry_max_age_days",
    label="HTF registry max age (days)",
    description=(
        "Maximum HTF-zone age, measured as the calendar-day difference between "
        "trading-day dates, including weekends. Zones at the limit remain; "
        "older zones are removed."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
    extra_values=_search_values(
        "htf_registry_max_age_days",
        (1, 2, 3, 4, 5),
        effect="older HTF zones leave the registry earlier, changing available setups",
    ),
)
_register_axis(
    "ltf_registry_max_live",
    label="LTF registry max live",
    description="Retention bound for LTF zones.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
)
_register_axis(
    "htf_selection_max_per_timeframe",
    label="HTF selection cap per timeframe",
    description=(
        "Newest live HTF zones admitted per timeframe before tap filtering, across "
        "both directions. Conflict and direction rules follow admission. One active "
        "setup and one active position remain the execution limits."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
    extra_values=_search_values(
        "htf_selection_max_per_timeframe",
        (2,),
        effect="admits the second ranked live HTF zone per timeframe before tap filtering",
    ),
)
if "htf_direction_selection_policy" in _B:
    _register_axis(
        "htf_direction_selection_policy",
        label="HTF direction selection order",
        description=(
            "Mixed-direction ranking preserves the baseline. The research alternative "
            "filters disabled directions before per-timeframe admission and ranking; "
            "all physical zones remain tracked. Requires exact strategy-search approval."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_STR,
        replay=True,
        extra_values=_search_values(
            "htf_direction_selection_policy", ("enabled_before_rank_v1",),
            effect=("enabled-direction zones enter ranking before the selection cap "
                    "and conflict checks"),
        ),
    )
if "htf_gap_invalidation_policy" in _B:
    _register_axis(
        "htf_gap_invalidation_policy",
        label="One-hour / four-hour gap invalidation",
        description=(
            "A one-hour gap waits for a completed one-hour candle. A four-hour gap "
            "waits for a completed four-hour candle. Wicks and smaller-chart closes "
            "do not invalidate these gaps. Smaller-pattern rules and protective stops "
            "are unchanged. Other setup cancellation and eligibility rules still apply."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_STR,
        replay=True,
        artifact_effect=(
            "policy-specific eligibility, replay identity and day seeds; "
            "full chronological replay required"
        ),
        extra_values=_search_values(
            "htf_gap_invalidation_policy", ("own_timeframe_close_v1",),
            effect=(
                "one-hour and four-hour starting gaps remain price-valid through wick "
                "traversals until a later finalized candle on their own timeframe "
                "closes strictly beyond the far boundary"
            ),
            readable_labels=True,
        ),
    )
_register_axis(
    "min_gap_ticks_capture",
    label="Minimum FVG gap (ticks)",
    description="FVG capture threshold — changes gap detection, hence lineage validity.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
)
if "opposing_min_gap_ticks" in _B:
    _register_axis(
        "opposing_min_gap_ticks",
        label="One-minute opposing-pattern minimum gap",
        description=(
            "Retain the separately configured minimum for one-minute opposing patterns. "
            "This inherited sequence-supply setting does not change starting-gap validity."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_NULLABLE_POS_INT,
        replay=True,
        artifacts=True,
        artifact_effect="one-minute capture threshold and source artifact identity",
        extra_values=_search_values(
            "opposing_min_gap_ticks", (1, 4),
            effect="changes only the opposing-pattern minimum; other gap roles retain their floor",
            readable_labels=True,
        ),
    )
_register_axis(
    "swing_strength_bars",
    label="Swing strength (bars)",
    description="Stop-anchor swing detection strength.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
)
_register_axis(
    "swing_pool_max",
    label="Swing pool size",
    description="Stop-anchor swing pool retention.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_INT,
    replay=True,
)
_register_axis(
    "sl_buffer_ticks",
    label="Stop buffer (ticks)",
    description="Stop distance beyond the anchored swing extreme.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NONNEG_INT,
    replay=True,
)
_register_axis(
    "tp_r_multiple",
    label="Target R multiple",
    description="Static take-profit as a multiple of risk.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_POS_FLOAT,
    replay=True,
    extra_values=_search_values(
        "tp_r_multiple", (2.0, 3.0),
        effect="the fixed target moves to 2 or 3 times the initial risk",
        readable_labels=True,
    ),
)
if "exit_policy" in _B:  # Strategy-Core builds with the scale-out exit
    _register_axis(
        "exit_policy",
        label="Exit rule",
        description=("How an open trade exits: the whole position at the target, or half "
                     "at the target with the rest held at break-even to the daily close."),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_STR,
        replay=True,
        dependencies=("holding_policy", "tp_r_multiple"),
        extra_values=_search_values(
            "exit_policy", ("scale_out_half_breakeven_hold_to_close_v1",),
            effect=("half exits at the target level, the rest moves to break-even and is "
                    "held to that stop or the scheduled daily close"),
            readable_labels=True,
        ),
    )

# ── searchable session policy / caps / sides ─────────────────────────────────
_entry_schedule_presets = ()
if "entry_schedule_policy" in _B:
    for _field, _label in (
        ("entry_schedule_policy", "Entry schedule policy"),
        ("entry_schedule_timezone", "Entry schedule timezone"),
        ("entry_schedule_windows", "Entry schedule windows"),
    ):
        _register_axis(
            _field, label=_label,
            description="Resolved atomically by the selected entry-session preset.",
            classification=AxisClassification.MEASUREMENT_ONLY,
            value_spec=_RECORD if _field == "entry_schedule_windows" else _STR,
            replay=True, baseline_token="baseline", dependencies=("enabled_entry_sessions",),
        )
    _entry_schedule_presets = tuple(
        CompositeAxisValue(
            value_id=f"enabled_entry_sessions.{token}",
            human_label=label,
            member_values={
                "enabled_entry_sessions": tuple(_B["enabled_entry_sessions"]),
                "entry_schedule_policy": policy,
                "entry_schedule_timezone": "America/Chicago",
                "entry_schedule_windows": windows,
            },
            capability_status="available", owner_ratification_status="pending",
            ratification_evidence_ref=None,
            dependencies=("enabled_entry_sessions", "entry_schedule_policy",
                          "entry_schedule_timezone", "entry_schedule_windows"),
            expected_replay_effect=(
                "Restricts actual new entries using a separate Chicago clock; structural "
                "sessions, market context and position management remain unchanged."
            ),
        )
        for token, label, policy, windows in (
            ("all_open_market_v1", "All open-market hours - Chicago time",
             "all_open_market_v1", ()),
            ("daytime_chicago_0700_1555_v1", "Daytime - 7:00 AM to 3:55 PM Chicago time",
             "explicit_windows_v1", (("07:00", "15:55"),)),
            ("morning_chicago_0700_1030_v1", "Morning - 7:00 AM to 10:30 AM Chicago time",
             "explicit_windows_v1", (("07:00", "10:30"),)),
        )
    )
if "holding_policy" in _B:
    for _field, _label in (
        ("daily_close_timezone", "Daily-close timezone"),
        ("daily_close_time", "Daily-close clock"),
        ("daily_close_buffer_minutes", "Scheduled-closure lead in minutes"),
    ):
        _register_axis(
            _field, label=_label,
            description="Frozen by the mandatory daily-close holding preset.",
            classification=AxisClassification.MEASUREMENT_ONLY,
            value_spec=_NONNEG_INT if _field == "daily_close_buffer_minutes" else _STR,
            replay=True, baseline_token="baseline", dependencies=("holding_policy",),
        )
    _register_axis(
        "holding_policy", label="Position holding rule",
        description=(
            "The research deadline is 3:55 PM Chicago time: a five-minute buffer before "
            "the owner's 4:00 PM boundary. Scheduled shortened days close earlier with "
            "the same lead. Entries stay locked until the actual market reopening. "
            "The historical holding behavior remains available for reproduction."
        ),
        classification=AxisClassification.APPROVED_SEARCH_AXIS,
        value_spec=_COMPOSITE, replay=True,
        extra_values=(CompositeAxisValue(
            value_id="holding_policy.scheduled_daily_close_v1",
            human_label="Mandatory daily close - 3:55 PM Chicago time (five-minute buffer)",
            member_values={
                "holding_policy": "scheduled_daily_close_v1",
                "daily_close_timezone": "America/Chicago",
                "daily_close_time": "15:55",
                "daily_close_buffer_minutes": 5,
            },
            capability_status="available", owner_ratification_status="pending",
            ratification_evidence_ref=None,
            dependencies=("holding_policy", "daily_close_timezone", "daily_close_time",
                          "daily_close_buffer_minutes"),
            expected_replay_effect=(
                "Actually closes positions before prohibited closures, records priced time "
                "exits, and blocks entries until scheduled reopening; full replay required."
            ),
        ),),
    )
_register_axis(
    "enabled_entry_sessions",
    label="Enabled entry sessions",
    description=(
        "Choose when new entries may execute. Starts are included and ends excluded. "
        "The corrected morning is 7:00 AM to 10:30 AM Chicago time. Entry windows "
        "do not liquidate positions; the separate holding rule governs their exit."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_COMPOSITE,
    replay=True,
    extra_values=(
        *(
            RegisteredAxisValue(
                value_id=f"enabled_entry_sessions.{name}",
                axis_technical_key="enabled_entry_sessions",
                payload=(name,),
                human_label=f"{label} only ({chicago} Chicago time)",
                capability_status="available",
                owner_ratification_status="pending",
                ratification_evidence_ref=None,
                expected_replay_effect=f"new entries are restricted to the {label} doc session",
            )
            for name, label, chicago in (
                ("asia", "Asia", "3:00 PM–12:45 AM"),
                ("london", "London", "1:00 AM–6:00 AM"),
                ("ny", "New York", "7:00 AM–1:00 PM"),
            )
        ),
        CompositeAxisValue(
            value_id="enabled_entry_sessions.ny_0700_1030",
            human_label="Legacy morning - 6:00 AM to 9:30 AM Chicago time (historical)",
            member_values={
                "enabled_entry_sessions": ("ny_0700_1030",),
                # A single window prevents the overlapping standard London/NY
                # stamps from shadowing this preset during replay.
                "doc_sessions": {"ny_0700_1030": ("07:00", "10:30")},
            },
            capability_status="available",
            owner_ratification_status="pending",
            ratification_evidence_ref=None,
            dependencies=("enabled_entry_sessions", "doc_sessions"),
            expected_replay_effect=(
                "new entries only at confirmation closes in [07:00, 10:30) "
                "America/New_York; market session clocks and open-trade exits unchanged"
            ),
        ),
        *_entry_schedule_presets,
    ),
)
_register_axis(
    "outside_session_policy",
    label="Outside-session policy",
    description="What a waiting setup does outside enabled sessions.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_STR,
    replay=True,
)
_register_axis(
    "max_executed_trades_per_day",
    label="Max executed trades per day",
    description=(
        "Maximum trades opened per trading day across all sessions; resets at "
        "18:00 ET. Counts executions, not candidates. None = no daily trade cap."
    ),
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_NULLABLE_POS_INT,
    replay=True,
    extra_values=_search_values(
        "max_executed_trades_per_day",
        (1, 2, 3),
        effect="additional trade entries are suppressed after reaching the daily execution cap",
    ),
)
_register_axis(
    "enable_shorts",
    label="Short side enabled",
    description="Canonical shorts remain blocked until ratified; schema-ready.",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_BOOL,
    replay=True,
    extra_values=(
        RegisteredAxisValue(
            value_id="enable_shorts.true",
            axis_technical_key="enable_shorts",
            payload=True,
            human_label="Enable canonical shorts",
            capability_status="available",
            owner_ratification_status="pending",
            ratification_evidence_ref=None,
            expected_replay_effect="adds the short side of the FSM thesis",
        ),
    ),
)

# ── searchable but expensive timeframe sets (new capture artifacts) ──────────
_register_axis(
    "htf_timeframes",
    label="HTF timeframe set",
    description="Changes the day bar/level artifact chain (expensive rebuild).",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_STR_TUPLE,
    replay=True,
    artifacts=True,
    artifact_effect="new artifacts_tag — full sequential day-artifact rebuild",
    extra_values=_search_values(
        "htf_timeframes", (("1H",),),
        effect="setups start only from one-hour gaps (four-hour gaps are not used)",
        readable_labels=True,
    ),
)
_register_axis(
    "parent_timeframes",
    label="Parent timeframe set",
    description="Changes the day bar/level artifact chain (expensive rebuild).",
    classification=AxisClassification.APPROVED_SEARCH_AXIS,
    value_spec=_STR_TUPLE,
    replay=True,
    artifacts=True,
    artifact_effect="new artifacts_tag — full sequential day-artifact rebuild",
    extra_values=(
        *_search_values(
            "parent_timeframes", (("1m", "3m", "5m", "10m", "15m", "30m"),),
            effect=("adds fresh one-minute supporting parents while retaining all "
                    "original parent charts"),
            readable_labels=True,
        ),
        *_search_values(
            "parent_timeframes", (("1m", "5m", "10m", "15m", "30m"),),
            effect="one-minute supporting parents added and three-minute parents removed",
            readable_labels=True,
        ),
    ),
)

# ── blocked axes (fail closed before enumeration) ────────────────────────────
_blocked(
    "break_even_enabled",
    "Break-even management",
    "blocked_inert_field",
    "in the section hash but never threaded into IfvgReducerConfig — varying it "
    "burns a full replay for zero behavioral delta",
    _BOOL,
)
_blocked(
    "legacy_candidate_row_limit",
    "Legacy candidate row limit",
    "blocked_inert_field",
    "inert legacy field; never threaded into the reducer",
    _NULLABLE_POS_INT,
)
_blocked(
    "parent_reaction_window_1m_bars_max",
    "Legacy global parent reaction window (1m)",
    "blocked_legacy_field",
    "legacy read-only field; v2 uses parent-bar reaction clocks",
    _NULLABLE_POS_INT,
)
_blocked(
    "retest_trigger",
    "Retest trigger",
    "blocked_reducer_hardcoded",
    "ifvg_retest execution is unconditionally blocked (retest_trigger_unratified)",
    _STR,
)
_blocked(
    "resolver_policy",
    "Same-bar resolver policy",
    "blocked_reducer_hardcoded",
    "field exists but the reducer never branches on it (stop-before-target fixed)",
    _STR,
)
_blocked(
    "parent_full_fill_invalidation",
    "Parent full-fill invalidation",
    "blocked_pending_owner_policy_review",
    "the Boolean cannot distinguish pre-lock vs locked fills, phase, wick-vs-body "
    "traversal, overshoot, thesis destruction, or replacement (FSM-audit D-1…D-5)",
    _BOOL,
)
_blocked(
    "parent_structural_invalidation",
    "Parent structural invalidation",
    "blocked_pending_owner_policy_review",
    "same open owner policy review as parent_full_fill_invalidation",
    _BOOL,
)

SEARCH_AXIS_REGISTRY_V1: Mapping[str, SearchAxisSpec] = MappingProxyType(dict(_AXIS_SPECS))
AXIS_VALUE_REGISTRY_V1: Mapping[str, RegisteredAxisValue | CompositeAxisValue] = (
    MappingProxyType(dict(_VALUE_REGISTRY))
)


def registry_sha256(
    axes: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
    values: Mapping[str, RegisteredAxisValue | CompositeAxisValue] = AXIS_VALUE_REGISTRY_V1,
) -> str:
    return canonical_contract_sha256(
        {
            "axes": {key: spec.model_dump(mode="json") for key, spec in sorted(axes.items())},
            "values": {
                key: value.model_dump(mode="json") for key, value in sorted(values.items())
            },
        }
    )


def assert_axes_authorized(
    axis_value_ids: Mapping[str, str],
    registry: Mapping[str, RegisteredAxisValue | CompositeAxisValue] = AXIS_VALUE_REGISTRY_V1,
    *,
    axes: Mapping[str, SearchAxisSpec] = SEARCH_AXIS_REGISTRY_V1,
    require_ratified: bool = True,
) -> None:
    """Fail-closed, value-level authorization check before enumeration.

    ``require_ratified=True`` (real scope) refuses ``pending`` values;
    synthetic charters pass ``require_ratified=False`` under the typed
    synthetic marker, which implies no owner authorization.
    """

    for axis_key, value_id in sorted(axis_value_ids.items()):
        spec = axes.get(axis_key)
        if spec is None:
            raise AxisAuthorizationError(f"unknown search axis {axis_key!r}")
        if spec.classification not in (
            AxisClassification.APPROVED_SEARCH_AXIS,
            AxisClassification.RISK_POLICY_AXIS,
            AxisClassification.EXPERIMENTAL,
        ):
            raise AxisAuthorizationError(
                f"axis {axis_key!r} is {spec.classification.value}; it is not searchable"
            )
        value = registry.get(value_id)
        if value is None:
            raise AxisAuthorizationError(
                f"axis value {value_id!r} is not registered; raw overrides are not accepted"
            )
        if value_id not in spec.registered_values:
            raise AxisAuthorizationError(
                f"axis value {value_id!r} is not registered for axis {axis_key!r}"
            )
        if value.capability_status != "available":
            raise AxisAuthorizationError(
                f"axis value {value_id!r} is {value.capability_status}"
            )
        if require_ratified and value.owner_ratification_status == "pending":
            raise AxisAuthorizationError(
                f"axis value {value_id!r} awaits owner ratification and cannot enter "
                "a real charter"
            )
        # a value bound to a DIFFERENT axis can never authorize under this key
        if isinstance(value, CompositeAxisValue):
            if axis_key not in value.dependencies:
                raise AxisAuthorizationError(
                    f"composite value {value_id!r} does not belong to axis {axis_key!r}"
                )
        elif value.axis_technical_key != axis_key:
            raise AxisAuthorizationError(
                f"axis value {value_id!r} belongs to {value.axis_technical_key!r}, "
                f"not {axis_key!r}"
            )
    # declared incompatibilities between the SELECTED values are enforced
    selected_ids = set(axis_value_ids.values())
    selected_axes = set(axis_value_ids.keys())
    for value_id in sorted(selected_ids):
        value = registry.get(value_id)
        if value is None:
            continue  # already refused above
        clash = set(value.incompatibilities) & (selected_ids | selected_axes)
        if clash:
            raise AxisAuthorizationError(
                f"axis value {value_id!r} is incompatible with {sorted(clash)}"
            )


def resolve_axis_overrides(
    axis_value_ids: Mapping[str, str],
    registry: Mapping[str, RegisteredAxisValue | CompositeAxisValue] = AXIS_VALUE_REGISTRY_V1,
) -> dict[str, Any]:
    """Expand registered value ids into exact ``IfvgSmcSection`` overrides.

    Standalone: it re-verifies that every value id belongs to the axis key it
    is presented under (a corrupted mapping cannot silently expand another
    axis's payload), even when ``assert_axes_authorized`` already ran.
    """

    overrides: dict[str, Any] = {}

    def _assign(field: str, payload: Any) -> None:
        if field in overrides and overrides[field] != payload:
            raise AxisAuthorizationError(
                f"conflicting overrides for section field {field!r}"
            )
        overrides[field] = payload

    for axis_key, value_id in sorted(axis_value_ids.items()):
        value = registry.get(value_id)
        if value is None:
            raise AxisAuthorizationError(f"axis value {value_id!r} is not registered")
        if isinstance(value, CompositeAxisValue):
            if axis_key not in value.dependencies:
                raise AxisAuthorizationError(
                    f"composite value {value_id!r} does not belong to axis {axis_key!r}"
                )
            for field, payload in value.member_values:
                # This field is a sequence of clock pairs, not a frozen mapping.
                # Preserve its declared shape instead of treating start clocks as keys.
                restored = (
                    [list(window) for window in payload]
                    if field == "entry_schedule_windows" else _thaw(payload)
                )
                _assign(field, restored)
        else:
            if value.axis_technical_key != axis_key:
                raise AxisAuthorizationError(
                    f"axis value {value_id!r} belongs to "
                    f"{value.axis_technical_key!r}, not {axis_key!r}"
                )
            _assign(value.axis_technical_key, _thaw(value.payload))
    return overrides


def _thaw(value: Any) -> Any:
    """Deep-frozen payloads back to plain JSON shapes for section validation."""

    if isinstance(value, tuple) and all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in value
    ) and value:
        return {key: _thaw(item) for key, item in value}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
