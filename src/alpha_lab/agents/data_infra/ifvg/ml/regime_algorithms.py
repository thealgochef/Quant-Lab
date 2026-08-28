"""Regime algorithm registry (ML_REGIME_CONTRACT_PLAN §4; V1 = KMeans only).

Every registered entry ships in V1 with its pinned parameter payload and
fail-closed planned/blocked behavior; **only ``kmeans_v1`` has a callable
fit implementation in V1** (revision P1-1; V3 P1-6). A fit request for a
planned algorithm is refused with the correct status and reason (P1-C) —
no direct spectral or Nyström fit is callable anywhere in V1. The
direct-spectral restrictions (train-fold-only affinity, observation and
memory caps, no test-partition assignments, forced
``BLOCKED_NO_OOS_ASSIGNMENT``) are recorded on the registry entry and bind
the future regime-expansion implementation.

Protocol POLICIES fail closed too (safety review S3): each entry names the
policy values it can execute (``executable_policies``); a protocol that
declares any other value — ``inner_train_only_selection``, PCA, a kernel,
``none_training_only`` … — is refused by ``assert_protocol_executable``
BEFORE any preprocessing or fit, with the planned reason. The registry's
pinned parameters enter every protocol identity as
``pinned_parameters_hash`` and are re-verified at run time, so a registry
edit can never execute under an old protocol id.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

from pydantic import Field

from ..search.identities import FrozenContract, ImmutableMap, canonical_contract_sha256
from .regime_contracts import (
    INITIALIZATION_POLICIES,
    RegimeProtocolPayload,
    RegimeStatus,
)

__all__ = [
    "RegimeAlgorithmEntry",
    "REGIME_ALGORITHM_REGISTRY",
    "KMEANS_ALGORITHM_KEY",
    "POST_V1_REGIME_EXPANSION_KEYS",
    "PROTOCOL_POLICY_FIELDS",
    "RegimeAlgorithmUnavailableError",
    "resolve_regime_algorithm_entry",
    "assert_regime_algorithm_fittable",
    "pinned_parameters_hash",
    "assert_protocol_executable",
]

KMEANS_ALGORITHM_KEY = "kmeans_v1"

#: The post-V1 regime-expansion algorithm keys — registered, pinned, and
#: refusal-tested in V1; implemented in the expansion release only.
POST_V1_REGIME_EXPANSION_KEYS: tuple[str, ...] = (
    "minibatch_kmeans_v1",
    "gaussian_mixture_v1",
    "spectral_clustering_train_only_v1",
    "nystrom_kmeans_v1",
    "surrogate_assignment_logistic_v1",
)

#: The protocol fields whose values are execution POLICIES (every one is
#: hashed into the protocol identity and must be executable by the entry).
PROTOCOL_POLICY_FIELDS: tuple[str, ...] = (
    "cluster_count_policy",
    "dimensionality_reduction_policy",
    "kernel_or_affinity_policy",
    "out_of_sample_assignment_policy",
    "missingness_policy",
    "winsorization_policy",
    "scaler_policy",
    "cluster_label_alignment_policy",
    "fit_scope",
)


class RegimeAlgorithmEntry(FrozenContract):
    algorithm_key: str
    implementation_status: Literal["implemented", "planned"]
    planned_release: str | None
    oos_capable: bool
    pinned_parameters: ImmutableMap[str, Any]
    constraints: ImmutableMap[str, Any]
    outputs: tuple[str, ...]
    initialization_policy: str
    #: policy field → the values this entry can EXECUTE (``None`` values are
    #: spelled "none"); empty for planned entries — nothing executes.
    executable_policies: ImmutableMap[str, tuple[str, ...]]
    default_status_on_fit: RegimeStatus | None
    refusal_reason: str | None
    mandatory_warning_text: str | None = None
    algorithm_version: str = Field(default="1")


_KMEANS_EXECUTABLE_POLICIES: dict[str, tuple[str, ...]] = {
    "cluster_count_policy": ("fixed_k",),
    "dimensionality_reduction_policy": ("none",),
    "kernel_or_affinity_policy": ("none",),
    "out_of_sample_assignment_policy": ("centroid_predict_v1",),
    "missingness_policy": ("median_impute_with_indicator_v1",),
    "winsorization_policy": ("none", "clip_p01_p99_train_fitted_v1"),
    "scaler_policy": ("standard_scaler_v1",),
    "cluster_label_alignment_policy": ("centroid_min_distance_hungarian_v1",),
    "fit_scope": ("per_training_fold",),
}


REGIME_ALGORITHM_REGISTRY: MappingProxyType[str, RegimeAlgorithmEntry] = MappingProxyType(
    {
        KMEANS_ALGORITHM_KEY: RegimeAlgorithmEntry(
            algorithm_key=KMEANS_ALGORITHM_KEY,
            implementation_status="implemented",
            planned_release=None,
            oos_capable=True,
            pinned_parameters={
                "init": "k-means++",
                "n_init": 10,
                "max_iter": 300,
                "tol": 1e-4,
                "random_state": 7,
                "algorithm": "lloyd",
            },
            constraints={},
            outputs=(
                "fold_local_cluster_id",
                "distance_to_every_centroid",
                "assigned_distance",
                "assignment_margin_d2_minus_d1",
            ),
            initialization_policy="k-means++_n_init_10_v1",
            executable_policies=_KMEANS_EXECUTABLE_POLICIES,
            default_status_on_fit=None,
            refusal_reason=None,
        ),
        "minibatch_kmeans_v1": RegimeAlgorithmEntry(
            algorithm_key="minibatch_kmeans_v1",
            implementation_status="planned",
            planned_release="post_v1_regime_expansion",
            oos_capable=True,
            pinned_parameters={
                "batch_size": 1024,
                "n_init": 10,
                "max_no_improvement": 100,
                "random_state": 7,
                "reassignment_ratio": 0.01,
            },
            constraints={"use_only_when_rows_gt": 100_000},
            outputs=("fold_local_cluster_id", "assigned_distance"),
            initialization_policy="minibatch_k-means++_n_init_10_v1",
            executable_policies={},
            default_status_on_fit=None,
            refusal_reason=(
                "the post-V1 regime-expansion release implements the "
                "deterministic mini-batch protocol; no fit implementation is "
                "callable in V1"
            ),
        ),
        "gaussian_mixture_v1": RegimeAlgorithmEntry(
            algorithm_key="gaussian_mixture_v1",
            implementation_status="planned",
            planned_release="post_v1_regime_expansion",
            oos_capable=True,
            pinned_parameters={
                "covariance_type": "diag",
                "reg_covar": 1e-6,
                "max_iter": 200,
                "n_init": 5,
                "init_params": "kmeans",
                "random_state": 7,
                "tol": 1e-3,
            },
            constraints={"covariance_policy": "diag_only"},
            outputs=("component_id", "probabilities", "entropy", "log_density"),
            initialization_policy="gmm_kmeans_init_n_init_5_v1",
            executable_policies={},
            default_status_on_fit=None,
            refusal_reason=(
                "the post-V1 regime-expansion release implements the "
                "bounded diag-covariance mixture; no fit implementation is "
                "callable in V1"
            ),
        ),
        "spectral_clustering_train_only_v1": RegimeAlgorithmEntry(
            algorithm_key="spectral_clustering_train_only_v1",
            implementation_status="planned",
            planned_release="post_v1_regime_expansion",
            oos_capable=False,
            pinned_parameters={
                "affinity": "nearest_neighbors",
                "n_neighbors": 15,
                "eigen_solver": "arpack",
                "assign_labels": "kmeans",
                "random_state": 7,
                "eigen_tol": "auto",
            },
            constraints={
                "max_training_observations": 20_000,
                "max_affinity_memory_bytes": 512 * 2**20,
                "min_observation_grain": "candidate_stage_row_or_completed_context_bar",
                "train_fold_only_affinity": True,
            },
            outputs=("training_only_cluster_id",),
            initialization_policy="spectral_kmeans_label_assignment_v1",
            executable_policies={},
            default_status_on_fit=RegimeStatus.BLOCKED_NO_OOS_ASSIGNMENT,
            refusal_reason=(
                "training-only spectral diagnostics arrive with the "
                "post-V1 regime-expansion release; no fit implementation is "
                "callable in V1, and its results can never enter a predictive "
                "bundle without a registered OOS assignment policy"
            ),
            mandatory_warning_text=(
                "Training-only exploratory clustering — no OOS assignment; "
                "cannot enter a predictive feature bundle"
            ),
        ),
        "nystrom_kmeans_v1": RegimeAlgorithmEntry(
            algorithm_key="nystrom_kmeans_v1",
            implementation_status="planned",
            planned_release="post_v1_regime_expansion",
            oos_capable=True,
            pinned_parameters={
                "kernel": "rbf",
                "gamma": "<spec_field>",
                "n_components": "<spec_field>",
                "nystroem_random_state": 7,
                # nested config as an immutable pair tuple (CS §0.3 — nothing
                # inside a registry entry is mutable in place)
                "kmeans": (
                    ("algorithm", "lloyd"),
                    ("max_iter", 300),
                    ("n_init", 10),
                    ("random_state", 7),
                ),
            },
            constraints={
                "identity_fields": (
                    "kernel",
                    "gamma",
                    "n_components",
                    "nystroem_random_state",
                    "kmeans_config",
                ),
            },
            outputs=("fold_local_cluster_id", "assigned_distance", "assignment_margin"),
            initialization_policy="nystroem_rbf_then_k-means++_n_init_10_v1",
            executable_policies={},
            default_status_on_fit=None,
            refusal_reason=(
                "the production-compatible spectral approximation "
                "(fold-fitted scaler → Nyström map → KMeans) arrives with the "
                "post-V1 regime-expansion release; no fit implementation is "
                "callable in V1"
            ),
        ),
        "surrogate_assignment_logistic_v1": RegimeAlgorithmEntry(
            algorithm_key="surrogate_assignment_logistic_v1",
            implementation_status="planned",
            planned_release="post_v1_regime_expansion",
            oos_capable=True,
            pinned_parameters={},
            constraints={
                "requires_explicit_owner_authorization": True,
                "secondary_to": "nystrom_kmeans_v1",
                "persist": (
                    "surrogate_algorithm",
                    "training_data_hash",
                    "training_accuracy",
                    "per_class_uncertainty",
                    "identity",
                ),
            },
            outputs=("surrogate_cluster_id",),
            initialization_policy="surrogate_logistic_v1",
            executable_policies={},
            default_status_on_fit=None,
            refusal_reason=(
                "the owner-authorized surrogate assignment arrives "
                "with the post-V1 regime-expansion release; no fit "
                "implementation is callable in V1"
            ),
        ),
    }
)


class RegimeAlgorithmUnavailableError(PermissionError):
    """A planned regime algorithm or policy was requested for fitting (P1-C)."""


def resolve_regime_algorithm_entry(algorithm_key: str) -> RegimeAlgorithmEntry:
    entry = REGIME_ALGORITHM_REGISTRY.get(algorithm_key)
    if entry is None:
        raise ValueError(
            f"unknown regime algorithm {algorithm_key!r}; registered keys: "
            f"{sorted(REGIME_ALGORITHM_REGISTRY)}"
        )
    return entry


def assert_regime_algorithm_fittable(algorithm_key: str) -> RegimeAlgorithmEntry:
    """Fail closed on planned algorithms with the registered status/reason."""

    entry = resolve_regime_algorithm_entry(algorithm_key)
    if entry.implementation_status != "implemented":
        raise RegimeAlgorithmUnavailableError(
            f"regime algorithm {algorithm_key!r} is "
            f"{entry.implementation_status}: {entry.refusal_reason}"
        )
    return entry


def pinned_parameters_hash(entry: RegimeAlgorithmEntry) -> str:
    """The canonical hash of an entry's pinned parameters (protocol identity)."""

    return canonical_contract_sha256(
        {"algorithm_key": entry.algorithm_key, "pinned": dict(entry.pinned_parameters)}
    )


def assert_protocol_executable(payload: RegimeProtocolPayload) -> RegimeAlgorithmEntry:
    """Refuse — before any preprocessing or fit — every protocol the
    registry cannot execute exactly: a planned algorithm, a drifted
    algorithm version or pinned-parameter hash, an unregistered
    initialization policy, or ANY policy value outside the entry's
    executable set (planned policies never silently run as something else)."""

    entry = assert_regime_algorithm_fittable(payload.algorithm_key)
    if payload.algorithm_version != entry.algorithm_version:
        raise RegimeAlgorithmUnavailableError(
            f"protocol algorithm_version {payload.algorithm_version!r} is not the "
            f"registered {entry.algorithm_version!r} for {entry.algorithm_key}"
        )
    if payload.pinned_parameters_hash != pinned_parameters_hash(entry):
        raise RegimeAlgorithmUnavailableError(
            "protocol pinned_parameters_hash does not match the registry's "
            f"pinned parameters for {entry.algorithm_key}; a registry edit "
            "never executes under an old protocol identity"
        )
    if payload.initialization_policy != entry.initialization_policy:
        raise RegimeAlgorithmUnavailableError(
            f"initialization_policy {payload.initialization_policy!r} is not "
            f"executable by {entry.algorithm_key} (registered: "
            f"{entry.initialization_policy!r})"
        )
    if payload.initialization_policy not in INITIALIZATION_POLICIES:
        raise RegimeAlgorithmUnavailableError("unregistered initialization policy")
    executable = dict(entry.executable_policies)
    for field in PROTOCOL_POLICY_FIELDS:
        value = getattr(payload, field)
        spelled = "none" if value is None else str(value)
        allowed = executable.get(field, ())
        if spelled not in allowed:
            raise RegimeAlgorithmUnavailableError(
                f"protocol policy {field}={spelled!r} is planned — not executable "
                f"by {entry.algorithm_key} in V1 (executable: {allowed}); the "
                "protocol is refused before any preprocessing or fit"
            )
    return entry


# V1 boundary invariants: exactly one implemented algorithm; the expansion
# keys are all registered and all planned (kickoff §9 ML/regime); every
# planned entry executes NOTHING; every initialization policy is registered.
_implemented = [
    key
    for key, entry in REGIME_ALGORITHM_REGISTRY.items()
    if entry.implementation_status == "implemented"
]
if _implemented != [KMEANS_ALGORITHM_KEY]:
    raise AssertionError("V1 implements kmeans_v1 only")
if set(POST_V1_REGIME_EXPANSION_KEYS) - set(REGIME_ALGORITHM_REGISTRY):
    raise AssertionError("every post-V1 expansion key must be registered (fail-closed)")
for _key, _entry in REGIME_ALGORITHM_REGISTRY.items():
    if _entry.implementation_status == "planned" and dict(_entry.executable_policies):
        raise AssertionError(f"planned entry {_key} must execute no policy")
    if _entry.initialization_policy not in INITIALIZATION_POLICIES:
        raise AssertionError(f"entry {_key} names an unregistered initialization policy")
