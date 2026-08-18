"""Deterministic computation-path derivation (DELTA_TAXONOMY.md §3.2).

Pure OR-fold over each changed dimension's ``requires_*`` flags plus the
closure rules; no I/O. The dashboard can show the computation path before
launch because it is materialized at spec-build time.
"""

from __future__ import annotations

from ..search.identities import FrozenContract

__all__ = ["ComputationPath", "derive_computation_path"]


class ComputationPath(FrozenContract):
    full_strategy_replay: bool
    feature_materialization: bool
    label_recomputation: bool
    model_refit: bool
    model_gated_sequential_replay: bool
    cost_recomputation: bool
    prop_resimulation: bool
    bootstrap_resimulation: bool
    reuse_trade_stream_hash: bool


def derive_computation_path(changed) -> ComputationPath:
    """OR-fold the changed dimensions' flags, then apply the closure rules.

    Closure (§3.2): full replay ⇒ cost + prop + bootstrap resimulation;
    a model-gated sequential replay produces a new trade stream ⇒ cost + prop
    + bootstrap; the stream hash is reusable only when nothing above
    invalidates it.
    """

    full_replay = any(spec.requires_full_strategy_replay for spec in changed)
    features = any(spec.requires_feature_materialization for spec in changed)
    labels = any(spec.requires_label_recomputation for spec in changed)
    refit = any(spec.requires_model_refit for spec in changed)
    gated = any(spec.requires_model_gated_sequential_replay for spec in changed)
    cost = any(spec.requires_cost_recomputation for spec in changed)
    prop = any(spec.requires_prop_resimulation for spec in changed)
    bootstrap = any(spec.requires_bootstrap_resimulation for spec in changed)

    if full_replay or gated:
        cost = True
        prop = True
        bootstrap = True
    return ComputationPath(
        full_strategy_replay=full_replay,
        feature_materialization=features,
        label_recomputation=labels,
        model_refit=refit,
        model_gated_sequential_replay=gated,
        cost_recomputation=cost,
        prop_resimulation=prop,
        bootstrap_resimulation=bootstrap,
        reuse_trade_stream_hash=not (full_replay or gated),
    )
