"""Regime fit/assign service (ML_REGIME_CONTRACT_PLAN §3–§5; V1 KMeans lane).

Fold-local everything: per VALID fold, the fixed preprocessing pipeline and
the pinned ``kmeans_v1`` estimator are fitted on training rows only, then
train AND test rows are transformed + assigned deterministically
(distances to every centroid, assigned distance, margin d2−d1). Rows are
PRESERVED: missing sources, invalid folds, and uncovered rows become typed
nulls (7B.22-10). The sample-adequacy gate and the occupancy/stability
gates are evaluated into a ``RegimeCapabilityAssessment`` — failure blocks
PROMOTION, never silently shrinks k (Amendment P1-2/P1-3).

Fail-closed inputs (reviews F1/F2/S3/S4): before any preprocessing or fit,
the protocol must be EXECUTABLE by the registry (planned algorithms AND
planned policy values refuse), its ``input_feature_bundle_ref`` must
resolve to a registered AVAILABLE bundle that CONTAINS every input
feature, the leakage validator must pass with registry-derived
availability stages, the verified ``source_artifact_ids`` must be
supplied, and the observation frame must be uniquely keyed.

``CONTEXT_BAR_PANEL`` grain (Amendment P1-B): the model may fit on a
completed point-in-time context-bar panel and assign the frozen current
regime to candidate stages at their as-of instants — strictly the last
COMPLETED bar at or before each candidate's as-of timestamp, taken only
from OUT-OF-SAMPLE (test-partition) assignments so no fit that saw the
candidate's future is ever consulted; a candidate before the first
completed bar, or on a bar with no OOS assignment, is a typed
``coverage_gap``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ..context_folds import ContextFoldSet, IfvgContextFoldDefinition
from ..search.identities import SHA256_PATTERN, ImmutableMap, canonical_contract_sha256
from .fold_set_artifact import fold_set_id as _legacy_fold_set_id
from .regime_algorithms import (
    KMEANS_ALGORITHM_KEY,
    assert_protocol_executable,
    pinned_parameters_hash,
    resolve_regime_algorithm_entry,
)
from .regime_contracts import (
    REGIME_PROPOSED_DEFAULTS,
    ObservationGranularity,
    RegimeCapabilityAssessment,
    RegimeCapabilityAssessmentEnvelope,
    RegimeCoverageReport,
    RegimeFitEnvelope,
    RegimeFitPayload,
    RegimeLeakageError,
    RegimeProtocolEnvelope,
    RegimeProtocolPayload,
    assert_no_regime_leakage,
    sample_adequacy_minimum,
)
from .regime_preprocessing import (
    FittedRegimePreprocessing,
    fit_regime_preprocessing,
    keyed_observations,
    observation_ts_column,
)

#: R6.1 safety review S9: ``threadpoolctl`` is a scikit-learn dependency the
#: repo does not declare itself, so the single-thread kernel imports it
#: LAZILY (a missing module fails the regime run with this exact reason,
#: never the import of the whole regime lane).
THREADPOOLCTL_MISSING_MESSAGE = (
    "the regime kernel requires 'threadpoolctl' (a scikit-learn dependency) to run under "
    "threadpool_limits(1) for byte-reproducible assignment tables; it is not importable in "
    "this environment"
)

__all__ = [
    "THREADPOOLCTL_MISSING_MESSAGE",
    "resolved_bundle_feature_names",
    "assert_inputs_permitted",
    "resolve_kmeans_protocol",
    "RegimeFoldFit",
    "RegimeProtocolRun",
    "run_regime_protocol",
    "assign_panel_regimes_to_candidates",
    "PANEL_ASSIGNMENT_COLUMNS",
]

_SHA256_RE = re.compile(SHA256_PATTERN)


def resolved_bundle_feature_names(input_feature_bundle_ref: str) -> tuple[str, ...]:
    """The feature names of the registered AVAILABLE bundle whose resolved
    identity is ``input_feature_bundle_ref`` — an unknown/unresolvable
    reference is refused (a caller-provided string is not evidence)."""

    from ..features.feature_blocks import BlockUnavailableError  # noqa: PLC0415
    from ..features.feature_bundles import (  # noqa: PLC0415
        FEATURE_BUNDLE_REGISTRY,
        resolve_bundle,
    )

    for key in FEATURE_BUNDLE_REGISTRY:
        try:
            envelope = resolve_bundle(key)
        except BlockUnavailableError:
            continue
        if envelope.resolved_feature_bundle_id == input_feature_bundle_ref:
            return tuple(envelope.payload.resolved_feature_names)
    raise ValueError(
        "input_feature_bundle_ref does not resolve to any registered AVAILABLE "
        "feature bundle; regime inputs must come from a verified bundle identity"
    )


def assert_inputs_permitted(
    payload: RegimeProtocolPayload,
    *,
    feature_stage_for: dict | None = None,
) -> None:
    """§5.2: every input is a member of the referenced bundle AND passes the
    leakage validator (registry-derived availability stages by default)."""

    names = resolved_bundle_feature_names(payload.input_feature_bundle_ref)
    outside = sorted(set(payload.resolved_input_features) - set(names))
    if outside:
        raise RegimeLeakageError(
            f"regime inputs outside the referenced feature bundle: {outside}"
        )
    assert_grain_bundle_coherent(payload)
    assert_no_regime_leakage(
        payload.resolved_input_features,
        observation_stage=payload.observation_stage,
        feature_stage_for=feature_stage_for,
    )


def _bundle_block_join_keys(input_feature_bundle_ref: str) -> dict[str, tuple[str, ...]]:
    """block key → join keys for every resolved block of the referenced bundle."""

    from ..features.feature_blocks import (  # noqa: PLC0415
        FEATURE_BLOCK_RESOLUTION_REGISTRY,
        BlockUnavailableError,
    )
    from ..features.feature_bundles import (  # noqa: PLC0415
        FEATURE_BUNDLE_REGISTRY,
        resolve_bundle,
    )

    by_id = {
        envelope.resolved_feature_block_id: envelope
        for envelope in FEATURE_BLOCK_RESOLUTION_REGISTRY.values()
    }
    for key in FEATURE_BUNDLE_REGISTRY:
        try:
            bundle = resolve_bundle(key)
        except BlockUnavailableError:
            continue
        if bundle.resolved_feature_bundle_id == input_feature_bundle_ref:
            return {
                by_id[block_id].payload.feature_block_key: tuple(by_id[block_id].payload.join_keys)
                for block_id in bundle.payload.resolved_block_ids
                if block_id in by_id
            }
    raise ValueError("input_feature_bundle_ref does not resolve to a registered bundle")


def assert_grain_bundle_coherent(payload: RegimeProtocolPayload) -> None:
    """R6.1 (§6.A): grain / bundle-key coherence — every block of the input
    bundle must join on ``row_id`` for the panel grain and on
    ``candidate_id`` otherwise (a panel protocol over candidate-keyed
    blocks, or a candidate protocol over the panel block, is a leakage of
    grain semantics and refuses)."""

    grain = ObservationGranularity(payload.observation_granularity)
    expected = (
        ("row_id",) if grain is ObservationGranularity.CONTEXT_BAR_PANEL else ("candidate_id",)
    )
    incoherent = sorted(
        block_key
        for block_key, keys in _bundle_block_join_keys(payload.input_feature_bundle_ref).items()
        if tuple(keys) != expected
    )
    if incoherent:
        raise RegimeLeakageError(
            f"grain {grain.value} requires every input-bundle block to join on "
            f"{expected}; blocks {incoherent} do not (grain/bundle-key coherence)"
        )


def resolve_kmeans_protocol(
    *,
    input_feature_bundle_ref: str,
    resolved_input_features: tuple[str, ...],
    observation_granularity: ObservationGranularity = (
        ObservationGranularity.CANDIDATE_STAGE_ROW
    ),
    observation_stage=None,
    panel_interval_seconds: int | None = None,
    panel_source_artifact_id: str | None = None,
    panel_as_of_policy_id: str | None = None,
    resolved_cluster_count: int = 3,
    winsorization_policy: str = "none",
    feature_stage_for: dict | None = None,
) -> RegimeProtocolEnvelope:
    """One concrete, hashed ``kmeans_v1`` protocol (bundle- and
    leakage-checked; pinned parameters hashed into the identity)."""

    from ..features.feature_blocks import AvailabilityStage  # noqa: PLC0415

    stage = observation_stage or AvailabilityStage.ENTRY_DECISION
    entry = resolve_regime_algorithm_entry(KMEANS_ALGORITHM_KEY)
    import sklearn  # noqa: PLC0415

    payload = RegimeProtocolPayload(
        algorithm_key=KMEANS_ALGORITHM_KEY,
        algorithm_version=entry.algorithm_version,
        pinned_parameters_hash=pinned_parameters_hash(entry),
        input_feature_bundle_ref=input_feature_bundle_ref,
        resolved_input_features=tuple(resolved_input_features),
        observation_granularity=observation_granularity,
        panel_interval_seconds=panel_interval_seconds,
        panel_source_artifact_id=panel_source_artifact_id,
        panel_as_of_policy_id=panel_as_of_policy_id,
        observation_stage=stage,
        missingness_policy="median_impute_with_indicator_v1",
        winsorization_policy=winsorization_policy,
        scaler_policy="standard_scaler_v1",
        dimensionality_reduction_policy="none",
        kernel_or_affinity_policy=None,
        cluster_count_policy="fixed_k",
        resolved_cluster_count=resolved_cluster_count,
        initialization_policy=entry.initialization_policy,
        out_of_sample_assignment_policy="centroid_predict_v1",
        cluster_label_alignment_policy="centroid_min_distance_hungarian_v1",
        fit_scope="per_training_fold",
        software_versions={
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        formula_version="regime_lane_v1",
    )
    assert_inputs_permitted(payload, feature_stage_for=feature_stage_for)
    return RegimeProtocolEnvelope.from_payload(payload)


@dataclass(frozen=True, slots=True)
class RegimeFoldFit:
    """One fold's complete fit: identity, preprocessing, estimator, facts."""

    fit_envelope: RegimeFitEnvelope
    fold_index: int
    preprocessing: FittedRegimePreprocessing
    estimator: KMeans
    centroids_scaled: np.ndarray
    inertia: float
    training_row_count: int


@dataclass(frozen=True, slots=True)
class RegimeProtocolRun:
    protocol: RegimeProtocolEnvelope
    fold_fits: tuple[RegimeFoldFit, ...]
    assignments: pd.DataFrame
    fold_set_id: str
    assessment: RegimeCapabilityAssessmentEnvelope = field(hash=False)


def _training_row_ids_hash(train_ids: tuple[str, ...]) -> str:
    return canonical_contract_sha256({"training_row_ids": sorted(train_ids)})


def _fold_set_id(folds: ContextFoldSet) -> str:
    """The ONE legacy row-population hash (R6.1: delegated to
    ``fold_set_artifact.fold_set_id`` — a three-way equality test pins that
    no existing identity moved)."""

    return _legacy_fold_set_id(folds)


def _observation_ts(indexed: pd.DataFrame, ts_column: str | None) -> dict[str, str | None]:
    """row id → ISO-8601 UTC as-of timestamp (None when absent)."""

    if ts_column is None:
        return {}
    stamps = pd.to_datetime(indexed[ts_column], utc=True, errors="coerce")
    return {
        str(row_id): (None if pd.isna(stamp) else stamp.isoformat())
        for row_id, stamp in zip(indexed.index, stamps, strict=True)
    }


def _null_assignment_row(
    protocol_id: str,
    fit_id: str | None,
    row_id: str,
    fold_index: int,
    partition: str,
    reason: str,
    observation_ts: str | None,
) -> dict:
    return {
        "resolved_regime_protocol_id": protocol_id,
        "regime_fit_id": fit_id or "",
        "row_id": row_id,
        "fold_index": fold_index,
        "partition": partition,
        "observation_ts_utc": observation_ts,
        "fold_local_cluster_id": None,
        "canonical_reporting_cluster_id": None,
        "distances": None,
        "assigned_distance": np.nan,
        "assignment_margin": np.nan,
        "assignment_entropy": np.nan,
        "log_density": np.nan,
        "outlier_score": np.nan,
        "valid": False,
        "missing_reason": reason,
    }


def _fit_one_fold(
    frame: pd.DataFrame,
    indexed: pd.DataFrame,
    observation_ts: dict[str, str | None],
    protocol: RegimeProtocolEnvelope,
    fold: IfvgContextFoldDefinition,
    *,
    source_artifact_ids: tuple[str, ...],
) -> tuple[RegimeFoldFit, list[dict]]:
    payload = protocol.payload
    # the executability refusal comes BEFORE any preprocessing fit (review S9)
    entry = assert_protocol_executable(payload)
    preprocessing = fit_regime_preprocessing(
        frame,
        payload.resolved_input_features,
        fold,
        winsorization_policy=payload.winsorization_policy,
    )
    pinned = dict(entry.pinned_parameters)
    estimator = KMeans(n_clusters=payload.resolved_cluster_count, **pinned)
    train_ids = list(preprocessing.training_row_ids)
    train_matrix = preprocessing.transform(indexed.loc[train_ids])
    estimator.fit(train_matrix)
    centroids = np.asarray(estimator.cluster_centers_, dtype=float)

    fit_payload = RegimeFitPayload(
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        source_artifact_ids=source_artifact_ids,
        fold_index=fold.fold_index,
        fit_start=fold.train_days[0] if fold.train_days else None,
        fit_end=fold.train_days[-1] if fold.train_days else None,
        training_row_ids_hash=_training_row_ids_hash(preprocessing.training_row_ids),
        training_feature_matrix_hash=preprocessing.training_feature_matrix_hash,
    )
    fit_envelope = RegimeFitEnvelope.from_payload(fit_payload)
    fold_fit = RegimeFoldFit(
        fit_envelope=fit_envelope,
        fold_index=fold.fold_index,
        preprocessing=preprocessing,
        estimator=estimator,
        centroids_scaled=centroids,
        inertia=float(estimator.inertia_),
        training_row_count=len(train_ids),
    )

    rows: list[dict] = []
    features = list(payload.resolved_input_features)
    protocol_id = protocol.resolved_regime_protocol_id
    fit_id = fit_envelope.regime_fit_id
    partitions = (
        ("train", tuple(sorted(str(cid) for cid in fold.train_candidate_ids))),
        ("test", tuple(sorted(str(cid) for cid in fold.test_candidate_ids))),
    )
    for partition, ids in partitions:
        if not ids:
            continue
        raw = indexed.loc[list(ids), features].to_numpy(dtype=float)
        present_mask = ~np.isnan(raw).all(axis=1)
        present_ids = [row_id for row_id, ok in zip(ids, present_mask, strict=True) if ok]
        for row_id, ok in zip(ids, present_mask, strict=True):
            if not ok:
                rows.append(
                    _null_assignment_row(
                        protocol_id,
                        fit_id,
                        row_id,
                        fold.fold_index,
                        partition,
                        "source_feature_missing",
                        observation_ts.get(row_id),
                    )
                )
        if not present_ids:
            continue
        transformed = preprocessing.transform(indexed.loc[present_ids])
        # distances to EVERY fold-local centroid in the scaled space
        distance_matrix = np.linalg.norm(
            transformed[:, None, :] - centroids[None, :, :], axis=2
        )
        order = np.argsort(distance_matrix, axis=1, kind="stable")
        for position, row_id in enumerate(present_ids):
            distances = distance_matrix[position]
            first, second = int(order[position, 0]), int(order[position, 1])
            rows.append(
                {
                    "resolved_regime_protocol_id": protocol_id,
                    "regime_fit_id": fit_id,
                    "row_id": row_id,
                    "fold_index": fold.fold_index,
                    "partition": partition,
                    "observation_ts_utc": observation_ts.get(row_id),
                    "fold_local_cluster_id": first,
                    "canonical_reporting_cluster_id": None,
                    "distances": [float(v) for v in distances],
                    "assigned_distance": float(distances[first]),
                    "assignment_margin": float(distances[second] - distances[first]),
                    "assignment_entropy": np.nan,
                    "log_density": np.nan,
                    "outlier_score": np.nan,
                    "valid": True,
                    "missing_reason": None,
                }
            )
    return fold_fit, rows


def _validated_source_ids(source_artifact_ids: tuple[str, ...]) -> tuple[str, ...]:
    ids = tuple(str(ref) for ref in source_artifact_ids)
    if not ids or any(not _SHA256_RE.fullmatch(ref) for ref in ids):
        raise ValueError(
            "run_regime_protocol requires at least one verified 64-hex "
            "source_artifact_id (the observation view / panel artifact)"
        )
    return ids


def run_regime_protocol(
    frame: pd.DataFrame,
    folds: ContextFoldSet,
    protocol: RegimeProtocolEnvelope,
    *,
    source_artifact_ids: tuple[str, ...],
    bootstrap_refits: int = 50,
) -> RegimeProtocolRun:
    """The complete fold-local run under a SINGLE BLAS/OpenMP thread.

    R6.1: multithreaded BLAS / OpenMP reductions make centroid distances
    non-reproducible at the ULP level between otherwise identical runs, which
    would fork the persisted assignment tables (and therefore every artifact
    that hashes them) without any scientific change. The kernel is therefore
    executed under ``threadpool_limits(limits=1)`` — an execution-environment
    control, not a protocol field: fit identities, labels, and the R6 golden
    fit id are unchanged, and a double run is byte-identical.
    """

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415 — see S9 below
    except ImportError as error:  # pragma: no cover - exercised by monkeypatched import
        raise RuntimeError(THREADPOOLCTL_MISSING_MESSAGE) from error
    with threadpool_limits(limits=1):
        return _run_regime_protocol_single_thread(
            frame,
            folds,
            protocol,
            source_artifact_ids=source_artifact_ids,
            bootstrap_refits=bootstrap_refits,
        )


def _run_regime_protocol_single_thread(
    frame: pd.DataFrame,
    folds: ContextFoldSet,
    protocol: RegimeProtocolEnvelope,
    *,
    source_artifact_ids: tuple[str, ...],
    bootstrap_refits: int = 50,
) -> RegimeProtocolRun:
    """The complete fold-local run: fits, assignments, alignment, gates.

    ``frame`` is the observation frame (candidate view rows for the
    candidate-stage grain; a completed-bar panel for the panel grain) keyed
    by ``candidate_id`` (or ``row_id``) and carrying an as-of timestamp
    column; ``source_artifact_ids`` are the verified identities of that
    evidence (they enter every fit identity).
    """

    from .regime_diagnostics import build_stability_report  # noqa: PLC0415

    payload = protocol.payload
    assert_protocol_executable(payload)  # planned algorithm/policy → refused first
    assert_inputs_permitted(payload)
    source_ids = _validated_source_ids(tuple(source_artifact_ids))
    if (
        payload.observation_granularity is ObservationGranularity.CONTEXT_BAR_PANEL
        and payload.panel_source_artifact_id not in source_ids
    ):
        # R6.1 (D2/§6.D): the panel protocol's pinned source must BE one of
        # the verified observation sources — a frame from elsewhere refuses
        raise ValueError(
            "panel protocol pins panel_source_artifact_id "
            f"{str(payload.panel_source_artifact_id)[:12]}… but the observation "
            "source ids do not include it; the panel frame is not this protocol's"
        )
    missing = sorted(set(payload.resolved_input_features) - set(frame.columns))
    if missing:
        raise ValueError(f"observation frame lacks regime inputs: {missing}")
    indexed = keyed_observations(frame)
    ts_column = observation_ts_column(frame)
    if ts_column is None:
        raise ValueError(
            "observation frame requires an as-of timestamp column "
            "(entry_ts_utc / feature_as_of_ts / bar_close_ts_utc)"
        )
    observation_ts = _observation_ts(indexed, ts_column)

    fold_fits: list[RegimeFoldFit] = []
    rows: list[dict] = []
    for fold in folds.folds:
        if not fold.valid:
            for partition, ids in (
                ("train", sorted(str(cid) for cid in fold.train_candidate_ids)),
                ("test", sorted(str(cid) for cid in fold.test_candidate_ids)),
            ):
                rows.extend(
                    _null_assignment_row(
                        protocol.resolved_regime_protocol_id,
                        None,
                        row_id,
                        fold.fold_index,
                        partition,
                        "fold_invalid",
                        observation_ts.get(row_id),
                    )
                    for row_id in ids
                )
            continue
        fold_fit, fold_rows = _fit_one_fold(
            frame,
            indexed,
            observation_ts,
            protocol,
            fold,
            source_artifact_ids=source_ids,
        )
        fold_fits.append(fold_fit)
        rows.extend(fold_rows)

    assignments = pd.DataFrame(rows)
    if not assignments.empty:
        from .regime_alignment import align_reporting_labels  # noqa: PLC0415

        alignment = align_reporting_labels(tuple(fold_fits))
        assignments["canonical_reporting_cluster_id"] = [
            alignment.canonical_for(int(fold_index), local)
            if valid and local is not None
            else None
            for fold_index, local, valid in zip(
                assignments["fold_index"],
                assignments["fold_local_cluster_id"],
                assignments["valid"],
                strict=True,
            )
        ]

    fold_set_id = _fold_set_id(folds)
    assessment = _assess_capability(
        protocol,
        tuple(fold_fits),
        assignments,
        fold_set_id=fold_set_id,
        frame=frame,
        bootstrap_refits=bootstrap_refits,
        build_stability_report=build_stability_report,
    )
    return RegimeProtocolRun(
        protocol=protocol,
        fold_fits=tuple(fold_fits),
        assignments=assignments,
        fold_set_id=fold_set_id,
        assessment=assessment,
    )


def _assess_capability(
    protocol: RegimeProtocolEnvelope,
    fold_fits: tuple[RegimeFoldFit, ...],
    assignments: pd.DataFrame,
    *,
    fold_set_id: str,
    frame: pd.DataFrame,
    bootstrap_refits: int,
    build_stability_report,
) -> RegimeCapabilityAssessmentEnvelope:
    payload = protocol.payload
    minimum_rows = sample_adequacy_minimum(payload.observation_granularity)
    occupancy_gate = float(
        REGIME_PROPOSED_DEFAULTS["minimum_cluster_occupancy_fraction"]["value"]
    )
    cluster_rows_gate = int(
        REGIME_PROPOSED_DEFAULTS["minimum_cluster_rows_per_fold"]["value"]
    )
    total = int(len(assignments))
    valid_mask = assignments["valid"].astype(bool) if total else pd.Series(dtype=bool)
    assigned = int(valid_mask.sum()) if total else 0
    typed_null = (
        assignments.loc[~valid_mask, "missing_reason"].value_counts().to_dict()
        if total
        else {}
    )
    per_fold = {}
    if total:
        for fold_index, group in assignments.groupby("fold_index"):
            per_fold[int(fold_index)] = float(group["valid"].astype(bool).mean())
    test_rows = (
        assignments[(assignments["partition"] == "test")] if total else pd.DataFrame()
    )
    oos_coverage = (
        float(test_rows["valid"].astype(bool).mean()) if len(test_rows) else 0.0
    )
    train_valid = (
        assignments[(assignments["partition"] == "train") & valid_mask]
        if total
        else pd.DataFrame()
    )
    occupancy: dict[int, float] = {}
    per_fold_canonical: dict[str, float] = {}
    if len(train_valid):
        counts = train_valid["canonical_reporting_cluster_id"].value_counts()
        occupancy = {
            int(cluster): float(count / len(train_valid))
            for cluster, count in sorted(counts.items())
        }
        # R6.1: occupancy per (fold, canonical id) — the same key space the
        # rows-per-fold gate uses, so both gates are reviewable together
        for fold_index, fold_group in train_valid.groupby("fold_index", sort=True):
            fold_counts = fold_group["canonical_reporting_cluster_id"].value_counts()
            for cluster, count in sorted(fold_counts.items()):
                per_fold_canonical[f"{int(fold_index)}:{int(cluster)}"] = float(
                    count / len(fold_group)
                )

    gate_failures: list[str] = []
    observed_min = min((f.training_row_count for f in fold_fits), default=0)
    if not fold_fits:
        gate_failures.append("no_valid_folds")
    if fold_fits and observed_min < minimum_rows:
        gate_failures.append("sample_adequacy")
    if fold_fits and len(train_valid):
        per_fold_cluster_rows = (
            train_valid.groupby(["fold_index", "fold_local_cluster_id"]).size()
        )
        if occupancy and min(occupancy.values()) < occupancy_gate:
            gate_failures.append("minimum_cluster_occupancy")
        if (per_fold_cluster_rows < cluster_rows_gate).any():
            gate_failures.append("minimum_cluster_rows_per_fold")
        expected = {
            (fit.fold_index, cluster)
            for fit in fold_fits
            for cluster in range(payload.resolved_cluster_count)
        }
        if set(per_fold_cluster_rows.index) != expected:
            gate_failures.append("empty_cluster_in_fold")
    if oos_coverage <= 0.0:
        gate_failures.append("no_oos_assignment_coverage")

    coverage_kinds = ("no_valid_folds", "sample_adequacy", "no_oos_assignment_coverage")
    coverage = RegimeCoverageReport(
        rows_total=total,
        rows_assigned=assigned,
        rows_typed_null={str(k): int(v) for k, v in sorted(typed_null.items())},
        per_fold_coverage=per_fold,
        oos_assignment_coverage=oos_coverage,
        per_cluster_occupancy=occupancy,
        per_fold_canonical_occupancy=per_fold_canonical,
        minimum_training_observations_gate=minimum_rows,
        minimum_training_observations_observed=observed_min,
        coverage_gates_passed=not any(f in coverage_kinds for f in gate_failures),
        gate_failures=tuple(f for f in gate_failures if f in coverage_kinds),
    )
    stability = build_stability_report(
        frame,
        protocol,
        fold_fits,
        assignments,
        bootstrap_refits=bootstrap_refits,
    )
    assessment = RegimeCapabilityAssessment(
        resolved_regime_protocol_id=protocol.resolved_regime_protocol_id,
        regime_fit_ids=tuple(fit.fit_envelope.regime_fit_id for fit in fold_fits),
        fold_set_id=fold_set_id,
        coverage=coverage,
        occupancy=ImmutableMap(occupancy),
        stability=stability,
        oos_assignment_available=oos_coverage > 0.0,
        minimum_cluster_occupancy_gate=occupancy_gate,
        minimum_cluster_rows_gate=cluster_rows_gate,
        minimum_assignment_confidence_gate=None,
        gates_passed=not gate_failures and stability.stability_gates_passed,
        gate_failures=tuple(
            (*gate_failures, *stability.gate_failures)
            if gate_failures or not stability.stability_gates_passed
            else ()
        ),
    )
    return RegimeCapabilityAssessmentEnvelope.from_payload(assessment)


#: The panel→candidate assignment output columns (R6.1: the descriptive
#: OOS-assignment artifact's columns — ``regime_oos_assignment``); the PIT
#: rule itself lives in ``regime_oos_assignment.assign_panel_regimes_to_candidates``.
from .regime_oos_assignment import (  # noqa: E402
    OOS_ASSIGNMENT_COLUMNS as PANEL_ASSIGNMENT_COLUMNS,
)
from .regime_oos_assignment import (  # noqa: E402
    assign_panel_regimes_to_candidates,
)
