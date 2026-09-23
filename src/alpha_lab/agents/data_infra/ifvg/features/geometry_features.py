"""Source-bound, offline geometry additions; frozen Wilder ATR(20), ticks.

The original B0 view remains immutable. This artifact adds exactly three ratios,
while retaining the denominator and selected-event measurements as audit columns.
"""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from pydantic import Field

from ..b0_projection import B0_PROJECTION_VERSION
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .arrow_tables import bytes_sha256, frame_from_arrow_bytes, frame_to_arrow_bytes

GEOMETRY_BLOCK_KEY = "IFVG_GEOMETRY_ATR20_V1"
GEOMETRY_BUNDLE_KEY = "B0_GEOMETRY_ATR20_V1"
GEOMETRY_FORMULA_VERSION = "ifvg_geometry_wilder_atr20_1m_v1"
GEOMETRY_FEATURES = (
    "geo_inversion_clearance_atr20",
    "geo_parent_htf_distance_atr20",
    "geo_opposing_parent_size_ratio",
)
GEOMETRY_STORE = "geometry_feature_artifacts"
GEOMETRY_FRAME_SIDECAR = "geometry_features.arrow"
ATR_PERIOD = 20
GEOMETRY_FORMULAS = {
    GEOMETRY_FEATURES[0]: "close_through_margin_ticks / decision_atr20_ticks",
    GEOMETRY_FEATURES[1]: "distance_to_htf_ticks / decision_atr20_ticks",
    GEOMETRY_FEATURES[2]: "geometry_opposing_size_ticks / geometry_parent_size_ticks",
    "volatility": (
        "Completed observed 60s bars, same saved trading_day (18ET reset); named sessions "
        "do not reset. First TR=H-L; subsequent TR=max(H-L,abs(H-prevC),abs(L-prevC)). "
        "Seed arithmetic mean first20 TR; subsequent ATR=(19*previousATR+TR)/20. "
        "No fabricated missing minutes; first19 complete bars null; partial bars excluded. "
        "All input prices and ATR are ticks; ratios dimensionless. Zero denominator null."
    ),
}
_BAR_COLUMNS = (
    "bar_id", "trading_day", "timeframe_ticks", "availability_ts_utc",
    "open_ticks", "high_ticks", "low_ticks", "close_ticks", "is_complete", "is_partial",
)
_INPUTS = (
    "close_through_margin_ticks", "distance_to_htf_ticks",
    "geometry_opposing_size_ticks", "geometry_parent_size_ticks",
)


class GeometryFeaturePayload(FrozenContract):
    view_id: str = Field(pattern=SHA256_PATTERN)
    b0_projection_evidence_hash: str = Field(pattern=SHA256_PATTERN)
    source_artifact_id: str = Field(pattern=SHA256_PATTERN)
    source_manifest_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    source_file_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bar_table_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_table_sha256: str = Field(pattern=SHA256_PATTERN)
    formula_version: str = GEOMETRY_FORMULA_VERSION
    formula_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_names: tuple[str, ...] = GEOMETRY_FEATURES
    candidate_count: int = Field(ge=0)


class GeometryFeatureArtifactEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "geometry_feature_artifact_id"
    geometry_feature_artifact_id: str = Field(pattern=SHA256_PATTERN)
    payload: GeometryFeaturePayload


def _normalized_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_BAR_COLUMNS) - set(bars))
    if missing:
        raise ValueError(f"geometry source bars lack authoritative fields: {missing}")
    frame = bars.loc[:, list(_BAR_COLUMNS)].copy()
    frame["trading_day"] = frame["trading_day"].astype(str)
    frame["availability_ts_utc"] = pd.to_datetime(frame["availability_ts_utc"], utc=True)
    if frame[list(_BAR_COLUMNS)].isna().any().any():
        raise ValueError("geometry source bar fields cannot be missing")
    if frame.bar_id.duplicated().any() or frame.availability_ts_utc.duplicated().any():
        raise ValueError("geometry source bars duplicate an ID or decision boundary")
    if not frame.timeframe_ticks.eq(60).all():
        raise ValueError("geometry volatility requires only 60s bars")
    for name in ("is_complete", "is_partial"):
        if not frame[name].isin([True, False]).all():
            raise ValueError("geometry source completeness must be explicit")
        frame[name] = frame[name].astype(bool)
    for name in ("open_ticks", "high_ticks", "low_ticks", "close_ticks"):
        values = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(values).all() or not values.eq(np.floor(values)).all():
            raise ValueError("geometry source prices must be finite integer ticks")
        frame[name] = values.astype("int64")
    if ((frame.high_ticks < frame.low_ticks)
            | (frame.open_ticks < frame.low_ticks) | (frame.open_ticks > frame.high_ticks)
            | (frame.close_ticks < frame.low_ticks) | (frame.close_ticks > frame.high_ticks)).any():
        raise ValueError("geometry source OHLC is inconsistent")
    return frame.sort_values(["trading_day", "availability_ts_utc", "bar_id"]).reset_index(
        drop=True
    )


def compute_wilder_atr20(bars: pd.DataFrame) -> pd.DataFrame:
    """Pure, causal observed-bar computation; retain partial rows with null ATR."""
    frame = _normalized_bars(bars)
    frame["true_range_ticks"] = np.nan
    frame["decision_atr20_ticks"] = np.nan
    frame["atr_completed_bar_count"] = 0
    frame["atr_missing_reason"] = "insufficient_same_day_completed_bars"
    for _day, group in frame.groupby("trading_day", sort=False):
        previous_close, atr, seed = None, None, []
        count = 0
        for index, row in group.iterrows():
            if not row.is_complete or row.is_partial:
                frame.at[index, "atr_completed_bar_count"] = count
                frame.at[index, "atr_missing_reason"] = "decision_bar_incomplete"
                continue
            count += 1
            tr = float(row.high_ticks - row.low_ticks)
            if previous_close is not None:
                tr = max(tr, abs(row.high_ticks - previous_close),
                         abs(row.low_ticks - previous_close))
            previous_close = float(row.close_ticks)
            frame.at[index, "true_range_ticks"] = tr
            frame.at[index, "atr_completed_bar_count"] = count
            if atr is None:
                seed.append(tr)
                if count == ATR_PERIOD:
                    atr = float(np.mean(seed))
            else:
                atr = ((ATR_PERIOD - 1) * atr + tr) / ATR_PERIOD
            if atr is not None:
                frame.at[index, "decision_atr20_ticks"] = atr
                frame.at[index, "atr_missing_reason"] = "zero_atr_denominator" if atr == 0 else ""
    return frame


def _verify_context_binding(view, source_reference: dict) -> bytes:
    """Read the verified child-bound envelope, never trust a caller's source ID."""
    from ..context_experiment_contracts import (  # noqa: PLC0415
        ArtifactReference,
        PairedIfvgArtifactReference,
    )
    from ..contracts import RecordTable  # noqa: PLC0415
    from ..search.research_data import CONTEXT_STORE, ResearchContextEnvelope  # noqa: PLC0415
    from ..search.review_evidence import load_search_review_evidence  # noqa: PLC0415

    path = Path(source_reference["path"]).resolve()
    artifact_id = source_reference["artifact_id"]
    if (path.name != "forward_bars.parquet" or path.parent.name != artifact_id
            or path.parent.parent.name != CONTEXT_STORE):
        raise ValueError("geometry source path is not its claimed context artifact")
    root = path.parent.parent.parent
    context = load_verified_envelope(root, CONTEXT_STORE, artifact_id, ResearchContextEnvelope)
    subject = context.payload.subject
    source = view.b0_projection_evidence["source"]
    if (subject.core_replay_id != source["core_replay_id"]
            or subject.v2_dataset_id != source["v2_dataset_id"]
            or subject.v2_manifest_hash != source["v2_manifest_payload_sha256"]):
        raise ValueError("geometry context belongs to another child/configuration")
    manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
    if manifest["manifest_payload_sha256"] != source_reference["manifest_payload_sha256"]:
        raise ValueError("geometry context manifest reference mismatch")
    core = load_search_review_evidence(root, subject.core_replay_id)
    accepted = core.dataset.tables[RecordTable.ENTRY_CANDIDATE].set_index("candidate_id")
    columns = [
        f"geometry_{geometry}_{field}"
        for geometry in ("parent", "opposing")
        for field in ("fvg_id", "size_ticks", "gap_low_ticks", "gap_high_ticks")
    ]
    columns += [f"geometry_entry_bar_{component}_ticks"
                for component in ("open", "high", "low", "close")]
    columns += ["geometry_entry_bar_bar_id"]
    for candidate in view.frame.to_dict("records"):
        original = accepted.loc[candidate["candidate_id"]]
        if any(candidate[name] != original[name] for name in columns):
            raise ValueError("geometry measurements differ from the accepted child candidate")
    pair = PairedIfvgArtifactReference(
        v2=core.dataset.reference,
        v3=ArtifactReference(
            artifact_id=artifact_id,
            manifest_payload_sha256=source_reference["manifest_payload_sha256"],
            artifact_kind="v3", dataset_schema_version=4, profile_hash=subject.section_config_hash,
            feature_formula_version=json.loads(context.payload.context_config_json)[
                "feature_formula_version"
            ],
        ),
    )
    if canonical_contract_sha256(pair) != view.artifact_pair_hash:
        raise ValueError("geometry context does not match the exact candidate artifact pair")
    return load_sidecar_bytes(root, CONTEXT_STORE, artifact_id, "forward_bars.parquet")


def materialize_geometry_features(view, bars_1m: pd.DataFrame, *, source_reference: dict):
    """Historical ATR20 prefit protocol; retained for exact artifact compatibility."""
    return _materialize_geometry_features(
        view, bars_1m, source_reference=source_reference,
        feature_names=GEOMETRY_FEATURES, formula_version=GEOMETRY_FORMULA_VERSION,
        formula_contract=GEOMETRY_FORMULAS, volatility_field="decision_atr20_ticks",
        compute_volatility=compute_wilder_atr20, normalize_bars=_normalized_bars,
    )


def _materialize_geometry_features(
    view, bars_1m: pd.DataFrame, *, source_reference: dict, feature_names: tuple[str, ...],
    formula_version: str, formula_contract: dict, volatility_field: str,
    compute_volatility, normalize_bars,
):
    """Verify saved source bytes and B0 selected events before computing additions."""
    evidence = view.b0_projection_evidence or {}
    if evidence.get("projection_version") != B0_PROJECTION_VERSION:
        raise ValueError("geometry additions require repaired selected-stage B0")
    projection_hash = evidence.get("projection_evidence_hash")
    if projection_hash != canonical_contract_sha256({
        key: value for key, value in evidence.items() if key != "projection_evidence_hash"
    }):
        raise ValueError("geometry B0 projection evidence hash mismatch")
    data = _verify_context_binding(view, source_reference)
    if bytes_sha256(data) != source_reference["sha256"]:
        raise ValueError("geometry source file hash mismatch")
    bars = normalize_bars(bars_1m)
    source_bars = normalize_bars(pd.read_parquet(BytesIO(data)))
    bar_bytes = frame_to_arrow_bytes(bars)
    if bar_bytes != frame_to_arrow_bytes(source_bars):
        raise ValueError("geometry source bars differ from verified source file")
    atr = compute_volatility(bars_1m).set_index("bar_id", drop=False)
    candidates = view.frame
    if candidates.candidate_id.duplicated().any():
        raise ValueError("geometry view repeats candidate IDs")
    selected = {row["candidate_id"]: row for row in evidence["candidate_evidence"]}
    rows = []
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        stage = selected.get(candidate_id)
        if stage is None or not stage["all_formula_clock_ordinal_checks_passed"]:
            raise ValueError("geometry candidate lacks verified selected-event linkage")
        decision_ts = pd.Timestamp(candidate["entry_ts_utc"])
        if decision_ts != pd.Timestamp(stage["entry_ts_utc"]):
            raise ValueError("geometry candidate entry time disagrees with selected evidence")
        if decision_ts != pd.Timestamp(candidate["feature_as_of_ts"]):
            raise ValueError("geometry candidate feature as-of is not exact entry decision")
        bar_id = str(candidate["geometry_entry_bar_bar_id"])
        if bar_id not in atr.index:
            raise ValueError("geometry candidate exact entry decision bar is absent")
        bar = atr.loc[bar_id]
        if (str(candidate["trading_day"]) != bar.trading_day
                or decision_ts != bar.availability_ts_utc):
            raise ValueError("geometry candidate decision bar has wrong day or availability")
        for component in ("open", "high", "low", "close"):
            if candidate[f"geometry_entry_bar_{component}_ticks"] != bar[f"{component}_ticks"]:
                raise ValueError("geometry entry decision OHLC disagrees with source")
        for geometry in ("parent", "opposing"):
            if candidate[f"geometry_{geometry}_fvg_id"] != stage[f"selected_{geometry}_fvg_id"]:
                raise ValueError("geometry selected FVG linkage disagrees with B0")
            width = candidate[f"geometry_{geometry}_gap_high_ticks"] - candidate[
                f"geometry_{geometry}_gap_low_ticks"]
            if candidate[f"geometry_{geometry}_size_ticks"] != width or width < 0:
                raise ValueError("geometry selected gap width disagrees with bounds")
        for name in ("close_through_margin_ticks", "distance_to_htf_ticks"):
            if candidate[name] != stage["projected_values"][name]:
                raise ValueError("geometry numerator disagrees with selected B0 event")
        values = {name: float(candidate[name]) for name in _INPUTS}
        if not all(np.isfinite(value) and value >= 0 for value in values.values()):
            raise ValueError("geometry measurements must be finite nonnegative ticks")
        if values["geometry_opposing_size_ticks"] != stage["projected_values"][
            "opposing_size_ticks"
        ]:
            raise ValueError("geometry opposing width disagrees with exact inversion")
        denominator = float(bar[volatility_field])
        reason = str(bar.atr_missing_reason)
        parent = values["geometry_parent_size_ticks"]
        row = {
            "candidate_id": candidate_id, "entry_family": candidate["entry_family"],
            "trading_day": str(candidate["trading_day"]), "entry_ts_utc": decision_ts.isoformat(),
            **values, volatility_field: denominator,
            "atr_completed_bar_count": int(bar.atr_completed_bar_count),
            "atr_as_of_bar_id": bar_id, "atr_as_of_ts_utc": bar.availability_ts_utc.isoformat(),
            "atr_missing_reason": reason,
            "parent_size_missing_reason": "zero_parent_size_denominator" if parent == 0 else "",
            "selected_parent_fvg_id": stage["selected_parent_fvg_id"],
            "selected_opposing_fvg_id": stage["selected_opposing_fvg_id"],
            feature_names[0]: values[_INPUTS[0]] / denominator if not reason else np.nan,
            feature_names[1]: values[_INPUTS[1]] / denominator if not reason else np.nan,
            feature_names[2]: values[_INPUTS[2]] / parent if parent > 0 else np.nan,
        }
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    payload = GeometryFeaturePayload(
        view_id=view.view_id, b0_projection_evidence_hash=projection_hash,
        source_artifact_id=source_reference["artifact_id"],
        source_manifest_payload_sha256=source_reference["manifest_payload_sha256"],
        source_file_sha256=source_reference["sha256"],
        source_bar_table_sha256=bytes_sha256(bar_bytes),
        feature_table_sha256=bytes_sha256(frame_to_arrow_bytes(frame)),
        formula_contract_sha256=canonical_contract_sha256(formula_contract),
        formula_version=formula_version, feature_names=feature_names,
        candidate_count=len(frame),
    )
    return GeometryFeatureArtifactEnvelope.from_payload(payload), frame


def verify_geometry_feature_frame(envelope, frame: pd.DataFrame) -> None:
    names, contract = geometry_protocol_contract(envelope.payload.formula_version)
    if (envelope.payload.feature_names != names
            or envelope.payload.formula_contract_sha256 != canonical_contract_sha256(contract)):
        raise ValueError("geometry feature formula contract mismatch")
    if len(frame) != envelope.payload.candidate_count or frame.candidate_id.duplicated().any():
        raise ValueError("geometry feature population mismatch")
    if bytes_sha256(frame_to_arrow_bytes(frame)) != envelope.payload.feature_table_sha256:
        raise ValueError("geometry feature frame hash mismatch")


def save_geometry_features(root: Path, envelope, frame: pd.DataFrame):
    verify_geometry_feature_frame(envelope, frame)
    _names, contract = geometry_protocol_contract(envelope.payload.formula_version)
    saved, _ = save_or_reuse_envelope(
        root, GEOMETRY_STORE, envelope,
        extra_files={GEOMETRY_FRAME_SIDECAR: frame_to_arrow_bytes(frame),
                     "formula_contract.json": json.dumps(contract, indent=2).encode()},
    )
    return saved


def load_geometry_features(root: Path, artifact_id: str):
    envelope = load_verified_envelope(
        root, GEOMETRY_STORE, artifact_id, GeometryFeatureArtifactEnvelope,
    )
    frame = frame_from_arrow_bytes(load_sidecar_bytes(root, GEOMETRY_STORE, artifact_id,
                                                     GEOMETRY_FRAME_SIDECAR))
    verify_geometry_feature_frame(envelope, frame)
    return envelope, frame


def geometry_protocol_contract(formula_version: str):
    """Readers preserve both prefit ATR20 and the corrected Core ATR14 protocol."""
    if formula_version == GEOMETRY_FORMULA_VERSION:
        return GEOMETRY_FEATURES, GEOMETRY_FORMULAS
    from . import geometry_core_atr14 as core_protocol  # noqa: PLC0415

    if formula_version == core_protocol.GEOMETRY_FORMULA_VERSION:
        return core_protocol.GEOMETRY_FEATURES, core_protocol.GEOMETRY_FORMULAS
    raise ValueError("geometry feature formula contract mismatch")


register_identity_pair(
    name="GeometryFeatureArtifact", envelope_cls=GeometryFeatureArtifactEnvelope,
    payload_cls=GeometryFeaturePayload, id_field="geometry_feature_artifact_id",
    example_factory=lambda: GeometryFeaturePayload(
        view_id="a" * 64, b0_projection_evidence_hash="b" * 64,
        source_artifact_id="c" * 64, source_manifest_payload_sha256="d" * 64,
        source_file_sha256="e" * 64, source_bar_table_sha256="f" * 64,
        feature_table_sha256=hashlib.sha256(b"example").hexdigest(),
        formula_contract_sha256=canonical_contract_sha256(GEOMETRY_FORMULAS), candidate_count=1,
    ),
)
