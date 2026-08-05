"""Exact candidate-stage feature view for formula-v2 IFVG experiments."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from .artifact_io import ArtifactVerificationError, VerifiedIfvgPair
from .context_contracts import ContextRecordTable
from .context_experiment_contracts import (
    CandidateStageContextLink,
    ContextFeatureTier,
    canonical_contract_sha256,
)
from .contracts import RecordTable

__all__ = [
    "M0_FEATURES",
    "STRUCTURE_PRIMITIVES",
    "DISPLACEMENT_PRIMITIVES",
    "M3_POOL_PRIMITIVES",
    "M3_SWEEP_PRIMITIVES",
    "CandidateFeatureView",
    "build_candidate_feature_view",
    "features_for_tier",
    "m3_cohort_status",
    "apply_observation_filters",
]

_PRIMARY_TFS = (60, 180, 300, 600, 900, 1800, 3600)
_EXPERIMENTAL_TFS = (*_PRIMARY_TFS, 14400)
_WINDOW_KINDS = (
    "parent_reaction",
    "counter_leg",
    "inversion_response",
    "post_inversion",
)

M0_FEATURES = (
    "direction",
    "entry_family",
    "entry_session",
    "parent_tf_seconds",
    "distance_to_htf_ticks",
    "elapsed_parent_bars_since_tap",
    "elapsed_1m_bars_since_tap",
    "elapsed_1m_bars_since_selection",
    "distance_to_parent_ticks",
    "elapsed_1m_bars_since_lock",
    "close_through_margin_ticks",
    "bars_armed_to_inversion",
    "opposing_size_ticks",
    "bars_since_inversion",
    "entry_to_parent_ticks",
    "in_engine_session",
    "in_doc_session",
    "entry_fvg_size_ticks",
    "entry_fvg_gap_low_ticks",
    "entry_fvg_gap_high_ticks",
    "geometry_htf_size_ticks",
    "geometry_parent_size_ticks",
    "geometry_opposing_size_ticks",
    "geometry_manipulation_swing_ticks",
    "geometry_sl_buffer_ticks",
    "geometry_entry_ticks",
    "geometry_stop_ticks",
    "risk_ticks",
)

STRUCTURE_PRIMITIVES = (
    "high_relationship",
    "low_relationship",
    "swing_sequence_state",
    "structure_direction",
    "last_break_type",
    "last_break_direction",
    "bars_since_break",
    "minutes_since_break",
    "state_age_bars",
    "structure_alignment",
    "break_alignment",
    "last_confirmed_swing_high_ticks",
    "last_confirmed_swing_low_ticks",
    "valid",
    "source_available",
    "missing_reason",
)

DISPLACEMENT_PRIMITIVES = (
    "valid",
    "source_available",
    "missing_reason",
    "metrics_observed_bar_count",
    "metrics_expected_eligible_bar_count",
    "metrics_missing_bar_count",
    "metrics_wall_elapsed_minutes",
    "metrics_eligible_elapsed_minutes",
    "metrics_range_ticks_mean",
    "metrics_body_ticks_sum",
    "metrics_true_range_ticks_mean",
    "metrics_body_fraction_mean",
    "metrics_directional_wick_fraction_mean",
    "metrics_opposing_wick_fraction_mean",
    "metrics_expected_close_location_mean",
    "metrics_setup_close_location_mean",
    "metrics_overlap_fraction_mean",
    "metrics_directional_bar_fraction",
    "metrics_opposing_bar_fraction",
    "metrics_max_consecutive_directional_bars",
    "metrics_directional_body_ticks_sum",
    "metrics_opposing_body_ticks_sum",
    "metrics_expected_close_progress_ticks_mean",
    "metrics_setup_close_progress_ticks_mean",
    "metrics_expected_net_move_ticks",
    "metrics_setup_net_move_ticks",
    "metrics_expected_net_move_normalized",
    "metrics_setup_net_move_normalized",
    "metrics_expected_velocity_normalized",
    "metrics_setup_velocity_normalized",
    "metrics_path_efficiency_abs",
    "metrics_expected_path_efficiency_signed",
    "metrics_setup_path_efficiency_signed",
    "metrics_max_pullback_ticks",
    "metrics_max_pullback_fraction",
    "metrics_directional_fvg_count",
    "metrics_opposing_fvg_count",
    "metrics_directional_fvg_width_sum_ticks",
    "metrics_opposing_fvg_width_sum_ticks",
    "metrics_directional_fvg_width_sum_normalized",
    "metrics_directional_gap_density",
)

M3_POOL_PRIMITIVES = (
    "containing_pool_count",
    "nearest_context_nearest_eqh_distance_ticks",
    "nearest_context_nearest_eqh_swing_count",
    "nearest_context_nearest_eqh_age_minutes",
    "nearest_context_nearest_eqh_span_minutes",
    "nearest_context_nearest_eqh_width_ticks",
    "nearest_context_nearest_eqh_creation_separation_ticks",
    "nearest_context_nearest_eql_distance_ticks",
    "nearest_context_nearest_eql_swing_count",
    "nearest_context_nearest_eql_age_minutes",
    "nearest_context_nearest_eql_span_minutes",
    "nearest_context_nearest_eql_width_ticks",
    "nearest_context_nearest_eql_creation_separation_ticks",
    "nearest_context_nearest_thesis_opposing_distance_ticks",
    "nearest_context_nearest_thesis_opposing_swing_count",
    "nearest_context_nearest_thesis_opposing_age_minutes",
    "nearest_context_nearest_thesis_opposing_width_ticks",
)

M3_SWEEP_PRIMITIVES = (
    "qualifies_opposing_leg",
    "pool_type",
    "source_timeframe_seconds",
    "sweep_depth_ticks",
    "sweep_depth_normalized",
    "reclaimed_after_sweep",
    "reclaim_latency_bars",
    "reclaim_close_distance_ticks",
    "reclaim_close_distance_normalized",
    "distance_at_lock_ticks",
    "distance_at_lock_normalized",
)


def _structure_features(timeframes: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        f"ctx_structure_{seconds}s_{field}"
        for seconds in timeframes
        for field in STRUCTURE_PRIMITIVES
    )


_M1_PRIMARY = (*M0_FEATURES, *_structure_features(_PRIMARY_TFS))
_M1_EXPERIMENTAL = (*M0_FEATURES, *_structure_features(_EXPERIMENTAL_TFS))
_M2 = (
    *_M1_PRIMARY,
    *(
        f"ctx_displacement_{kind}_{field}"
        for kind in _WINDOW_KINDS
        for field in DISPLACEMENT_PRIMITIVES
    ),
)
_M3 = (
    *_M2,
    *(f"ctx_pool_{field}" for field in M3_POOL_PRIMITIVES),
    *(f"ctx_sweep_{field}" for field in M3_SWEEP_PRIMITIVES),
    "ctx_sweep_qualifying_link_count",
)

TIER_FEATURE_REGISTRY = MappingProxyType(
    {
        ContextFeatureTier.M0: M0_FEATURES,
        ContextFeatureTier.M1_PRIMARY: _M1_PRIMARY,
        ContextFeatureTier.M1_PLUS_240_EXPERIMENTAL: _M1_EXPERIMENTAL,
        ContextFeatureTier.M2: _M2,
        ContextFeatureTier.M3: _M3,
    }
)

_IDENTITY_COLUMNS = (
    "candidate_id",
    "setup_id",
    "trading_day",
    "entry_ts_utc",
    "feature_as_of_ts",
    "context_as_of_ts",
    "context_capture_id",
    "context_state_id",
    "geometry_evidence_id",
    "geometry_evidence_cursor",
    "is_warmup",
)


@dataclass(frozen=True, slots=True)
class CandidateFeatureView:
    view_id: str
    artifact_pair_hash: str
    feature_registry_hash: str
    frame: pd.DataFrame
    tier_features: dict[ContextFeatureTier, tuple[str, ...]]
    m3_status: str

    def frame_for_tier(self, tier: ContextFeatureTier) -> pd.DataFrame:
        columns = [*_IDENTITY_COLUMNS, *self.tier_features[tier]]
        return self.frame.loc[:, columns].copy()


def features_for_tier(tier: ContextFeatureTier | str) -> tuple[str, ...]:
    return TIER_FEATURE_REGISTRY[ContextFeatureTier(tier)]


def m3_cohort_status(frame: pd.DataFrame) -> str:
    qualifying = pd.to_numeric(
        frame["ctx_sweep_qualifying_link_count"], errors="coerce"
    ).fillna(0)
    return (
        "model_eligible"
        if qualifying.gt(0).any() and qualifying.eq(0).any()
        else "descriptive_only_no_positive_qualification_coverage"
    )


def _one(frame: pd.DataFrame, key: str, label: str) -> pd.DataFrame:
    if key not in frame:
        raise ArtifactVerificationError(f"{label} lacks exact key {key}")
    if frame[key].isna().any() or frame[key].duplicated().any():
        raise ArtifactVerificationError(f"{label} key {key} is null or duplicated")
    return frame.set_index(key, verify_integrity=True, drop=False)


def _nullable(row: pd.Series, column: str) -> Any:
    return row[column] if column in row.index else None


def _first_present(row: pd.Series, *columns: str) -> Any:
    for column in columns:
        value = _nullable(row, column)
        if value is None:
            continue
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if hasattr(missing, "item"):
            missing = missing.item()
        if isinstance(missing, bool) and missing:
            continue
        return value
    return None


def _items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return tuple(value)
    raise ArtifactVerificationError("context ID collection is not a list")


def _candidate_link(row: pd.Series) -> CandidateStageContextLink:
    return CandidateStageContextLink(
        candidate_id=str(row["candidate_id"]),
        setup_id=str(row["setup_id"]),
        stage=str(row["stage"]),
        geometry_evidence_id=str(row["geometry_evidence_id"]),
        geometry_evidence_cursor=str(row["geometry_evidence_cursor"]),
        context_capture_id=str(row["context_capture_id"]),
        context_state_id=str(row["context_state_id"]),
        context_as_of_ts=pd.Timestamp(row["context_as_of_ts"]).to_pydatetime(),
        feature_as_of_ts=pd.Timestamp(row["feature_as_of_ts"]).to_pydatetime(),
    )


def _structure_values(
    capture: pd.Series,
    structures: pd.DataFrame,
) -> dict[str, Any]:
    result = {
        name: None
        for name in _structure_features(_EXPERIMENTAL_TFS)
    }
    state_ids = []
    # The local state is always exact; the MTF state IDs are stored on context_state.
    state = capture.attrs.get("context_state")
    if state is not None:
        local = _nullable(state, "local_structure_state_id")
        if pd.notna(local):
            state_ids.append(str(local))
        state_ids.extend(str(item) for item in _items(_nullable(state, "mtf_state_ids")))
    seen_tf: set[int] = set()
    for state_id in state_ids:
        if state_id not in structures.index:
            raise ArtifactVerificationError("context state references a missing structure state")
        row = structures.loc[state_id]
        seconds = int(row["source_timeframe_seconds"])
        if seconds in seen_tf:
            raise ArtifactVerificationError("candidate has duplicate structure timeframe evidence")
        seen_tf.add(seconds)
        if seconds not in _EXPERIMENTAL_TFS:
            raise ArtifactVerificationError("candidate has an unregistered structure timeframe")
        for field in STRUCTURE_PRIMITIVES:
            result[f"ctx_structure_{seconds}s_{field}"] = _nullable(row, field)
    return result


def _displacement_values(
    capture: pd.Series,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    result = {
        f"ctx_displacement_{kind}_{field}": None
        for kind in _WINDOW_KINDS
        for field in DISPLACEMENT_PRIMITIVES
    }
    seen: set[str] = set()
    for window_id in _items(_nullable(capture, "displacement_window_ids")):
        if str(window_id) not in windows.index:
            raise ArtifactVerificationError("capture references a missing displacement window")
        row = windows.loc[str(window_id)]
        kind = str(row["window_kind"])
        if kind not in _WINDOW_KINDS:
            raise ArtifactVerificationError("capture contains an unregistered displacement window")
        if kind in seen:
            raise ArtifactVerificationError("capture has duplicate displacement window kind")
        seen.add(kind)
        for field in DISPLACEMENT_PRIMITIVES:
            result[f"ctx_displacement_{kind}_{field}"] = _nullable(row, field)
    return result


def _m3_values(
    capture: pd.Series,
    state: pd.Series,
    sweeps: pd.DataFrame,
) -> dict[str, Any]:
    result = {
        **{f"ctx_pool_{field}": _nullable(state, field) for field in M3_POOL_PRIMITIVES},
        **{f"ctx_sweep_{field}": None for field in M3_SWEEP_PRIMITIVES},
    }
    link_ids = tuple(
        str(item)
        for item in _items(_nullable(capture, "opposing_leg_sweep_link_ids"))
    )
    selected = _nullable(capture, "selected_opposing_leg_sweep_link_id")
    if pd.notna(selected) and str(selected) not in link_ids:
        raise ArtifactVerificationError("selected sweep link is not in capture evidence")
    qualifying = 0
    for link_id in link_ids:
        if link_id not in sweeps.index:
            raise ArtifactVerificationError("capture references a missing sweep link")
        qualifying += int(bool(sweeps.loc[link_id]["qualifies_opposing_leg"]))
    result["ctx_sweep_qualifying_link_count"] = qualifying
    if pd.notna(selected):
        row = sweeps.loc[str(selected)]
        if not bool(row["qualifies_opposing_leg"]):
            raise ArtifactVerificationError("selected opposing sweep link is not qualifying")
        for field in M3_SWEEP_PRIMITIVES:
            result[f"ctx_sweep_{field}"] = _nullable(row, field)
    return result


def build_candidate_feature_view(pair: VerifiedIfvgPair) -> CandidateFeatureView:
    core = pair.v2.tables
    context = pair.v3.tables
    candidates = _one(core[RecordTable.ENTRY_CANDIDATE], "candidate_id", "v2 candidates")
    links = _one(
        context[ContextRecordTable.CANDIDATE_CONTEXT_LINK],
        "candidate_id",
        "candidate context links",
    )
    captures = _one(
        context[ContextRecordTable.CONTEXT_CAPTURE],
        "context_capture_id",
        "context captures",
    )
    states = _one(
        context[ContextRecordTable.CONTEXT_STATE],
        "context_state_id",
        "context states",
    )
    structures = _one(
        context[ContextRecordTable.CONTEXT_STRUCTURE_STATE],
        "structure_state_id",
        "structure states",
    )
    windows = _one(
        context[ContextRecordTable.CONTEXT_DISPLACEMENT_WINDOW],
        "displacement_window_id",
        "displacement windows",
    )
    sweeps = _one(
        context[ContextRecordTable.EQUAL_LEVEL_SWEEP_LINK],
        "sweep_link_id",
        "sweep links",
    )
    if set(candidates.index.astype(str)) != set(links.index.astype(str)):
        raise ArtifactVerificationError("candidate view requires complete exact link coverage")

    rows: list[dict[str, Any]] = []
    for candidate_id in sorted(candidates.index.astype(str)):
        candidate = candidates.loc[candidate_id]
        link_row = links.loc[candidate_id]
        link = _candidate_link(link_row)
        if link.stage != "entry_candidate":
            raise ArtifactVerificationError("candidate link is not at entry_candidate stage")
        if link.context_capture_id not in captures.index:
            raise ArtifactVerificationError("candidate link has no exact context capture")
        capture = captures.loc[link.context_capture_id].copy()
        if link.context_state_id not in states.index:
            raise ArtifactVerificationError("candidate link has no exact context state")
        state = states.loc[link.context_state_id]
        capture.attrs["context_state"] = state
        entry_ts = _first_present(candidate, "entry_ts_utc", "envelope_ts_utc")
        if entry_ts is None:
            entry_ts = link.feature_as_of_ts
        entry_ts = pd.Timestamp(entry_ts)
        if entry_ts.tzinfo is None:
            raise ArtifactVerificationError(
                f"candidate {candidate_id} entry timestamp is not timezone-aware"
            )
        entry_ts = entry_ts.tz_convert("UTC")
        if entry_ts != pd.Timestamp(link.feature_as_of_ts).tz_convert("UTC"):
            raise ArtifactVerificationError(
                f"candidate {candidate_id} entry time differs from exact feature as-of"
            )
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "setup_id": link.setup_id,
            "trading_day": str(
                _first_present(candidate, "trading_day", "envelope_trading_day")
            ),
            "entry_ts_utc": entry_ts,
            "feature_as_of_ts": link.feature_as_of_ts,
            "context_as_of_ts": link.context_as_of_ts,
            "context_capture_id": link.context_capture_id,
            "context_state_id": link.context_state_id,
            "geometry_evidence_id": link.geometry_evidence_id,
            "geometry_evidence_cursor": link.geometry_evidence_cursor,
            "is_warmup": bool(_nullable(candidate, "is_warmup")),
        }
        row.update({field: _nullable(candidate, field) for field in M0_FEATURES})
        row.update(_structure_values(capture, structures))
        row.update(_displacement_values(capture, windows))
        row.update(_m3_values(capture, state, sweeps))
        rows.append(row)

    columns = [*_IDENTITY_COLUMNS, *dict.fromkeys(_M3 + _M1_EXPERIMENTAL)]
    frame = pd.DataFrame(rows, columns=columns)
    if frame["candidate_id"].duplicated().any():
        raise ArtifactVerificationError("candidate view contains duplicate candidates")
    m3_status = m3_cohort_status(frame)
    artifact_pair_hash = canonical_contract_sha256(pair.reference)
    feature_registry_hash = canonical_contract_sha256(
        {tier.value: list(features) for tier, features in TIER_FEATURE_REGISTRY.items()}
    )
    view_payload = {
        "artifact_pair_hash": artifact_pair_hash,
        "feature_registry_hash": feature_registry_hash,
        "candidate_ids": frame["candidate_id"].tolist(),
        "candidate_link_ids": frame["context_capture_id"].tolist(),
    }
    return CandidateFeatureView(
        view_id=canonical_contract_sha256(view_payload),
        artifact_pair_hash=artifact_pair_hash,
        feature_registry_hash=feature_registry_hash,
        frame=frame,
        tier_features=dict(TIER_FEATURE_REGISTRY),
        m3_status=m3_status,
    )


def apply_observation_filters(
    view: CandidateFeatureView,
    filters: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    forbidden = {
        "profile_name",
        "entry_session_definition",
        "enable_shorts",
        "trigger",
        "entry_family",
        "session_scheme",
    }
    invalid = sorted(forbidden & set(filters))
    if invalid:
        raise ValueError(
            "strategy-construction controls require a newly prepared profile: "
            f"{invalid}"
        )
    result = view.frame.copy()
    for column, allowed in sorted(filters.items()):
        if column not in result:
            raise ValueError(f"unknown observation filter {column!r}")
        result = result[result[column].astype(str).isin(tuple(map(str, allowed)))]
    return result.reset_index(drop=True)
