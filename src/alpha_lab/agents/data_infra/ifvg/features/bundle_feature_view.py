"""Bundle-scoped feature views over the AVAILABLE blocks (PHASED R5/R5B).

`resolve_bundle` is the only gate onto a view: a bundle containing any
non-available block (regime/key-level/execution-liquidity until their
releases) refuses with the block's status and reason BEFORE any frame is
touched. Since the R5B activation, MBP-1-bearing bundles resolve — their
``ofl_*`` columns come from an immutable, exactly-joined MBP-1 feature
artifact (`join_mbp1_features`: one-to-one on ``candidate_id``, typed nulls,
no nearest-time or row-order fallback), and the resulting view identity
binds the immutable candidate view, the exact resolved bundle, AND the MBP-1
feature artifact that supplied the joined evidence. The activated block is
research-only offline (owner decision R-6).
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import Field

from ..context_experiment_contracts import ContextFeatureTier
from ..context_feature_view import _IDENTITY_COLUMNS, CandidateFeatureView
from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)
from ..search.store import load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope
from .arrow_tables import bytes_sha256, frame_from_arrow_bytes, frame_to_arrow_bytes
from .feature_bundles import FROZEN_TIER_BUNDLES, resolve_bundle
from .mbp1_source_contract import assert_no_deep_book_identifiers, mbp1_feature_names

__all__ = [
    "BUNDLE_FEATURE_VIEW_STORE",
    "BUNDLE_VIEW_FRAME_SIDECAR",
    "BundleFeatureViewPayload",
    "BundleFeatureViewEnvelope",
    "build_bundle_feature_view",
    "resolve_available_bundle_view",
    "frozen_tier_for_bundle",
    "mbp1_block_keys_in_bundle",
    "bundle_categorical_features",
    "bundle_view_frame_bytes",
    "verify_bundle_feature_view_frame",
    "save_bundle_feature_view",
    "load_bundle_feature_view",
    "load_bundle_feature_view_frame",
]

BUNDLE_FEATURE_VIEW_STORE = "bundle_feature_views"
BUNDLE_VIEW_FRAME_SIDECAR = "bundle_feature_view.arrow"


class BundleFeatureViewPayload(FrozenContract):
    """One immutable candidate view scoped to one resolved bundle."""

    view_id: str = Field(pattern=SHA256_PATTERN)
    feature_bundle_key: str
    resolved_feature_bundle_id: str = Field(pattern=SHA256_PATTERN)
    feature_registry_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=0)
    resolved_feature_names: tuple[str, ...]
    #: Exact binding of joined MBP-1 evidence (None for bundles without the
    #: order-flow block): the same view+bundle over different MBP-1 feature
    #: artifacts must never share one identity.
    mbp1_feature_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    #: R6.1 (D9): exact binding of the fold-local regime feature artifact a
    #: regime-bearing bundle view joined (None otherwise). Pre-acceptance
    #: identity evolution — every bundle_feature_view_id moves (recorded).
    regime_fold_feature_artifact_id: str | None = Field(default=None, pattern=SHA256_PATTERN)


class BundleFeatureViewEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "bundle_feature_view_id"

    bundle_feature_view_id: str = Field(pattern=SHA256_PATTERN)
    payload: BundleFeatureViewPayload
    #: R6.1 (D): post-materialization fact binding the persisted frame bytes
    #: (None on a view envelope that was never persisted with its frame).
    frame_table_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


def mbp1_block_keys_in_bundle(bundle_envelope) -> bool:
    """Whether the resolved bundle carries MBP-1 order-flow features."""

    metric_set = set(mbp1_feature_names())
    return any(
        name in metric_set for name in bundle_envelope.payload.resolved_feature_names
    )


def build_bundle_feature_view(
    view: CandidateFeatureView,
    bundle_envelope,
    *,
    mbp1_features: pd.DataFrame | None = None,
    mbp1_feature_artifact=None,
) -> tuple[BundleFeatureViewEnvelope, pd.DataFrame]:
    """Scope the candidate view's frame to the bundle's resolved features.

    v2/v3 features must exist as view columns; ``ofl_*`` features come from
    the supplied MBP-1 feature frame through the exact one-to-one join.
    A resolved MBP-1-bearing bundle requires BOTH the materialized frame
    and its ``Mbp1FeatureArtifactEnvelope`` — and the frame is VERIFIED to
    hash to that envelope's ``feature_table_sha256`` before the id is
    pinned (review F1: the binding is rehashed, never caller-asserted).
    Every view candidate must have a row in the evidence (typed-null VALUES
    are lawful; a missing ROW is a cohort misalignment and refuses —
    review F9). No other missing column is ever filled, and no
    deeper-than-MBP-1 identifier can appear.
    """

    names = tuple(bundle_envelope.payload.resolved_feature_names)
    assert_no_deep_book_identifiers(names)
    frame_source = view.frame
    missing = sorted(set(names) - set(frame_source.columns))
    mbp1_needed = tuple(name for name in missing if name in set(mbp1_feature_names()))
    truly_missing = sorted(set(missing) - set(mbp1_needed))
    if truly_missing:
        raise ValueError(
            f"bundle features are missing from the immutable candidate view: {truly_missing}"
        )
    mbp1_artifact_id: str | None = None
    if mbp1_needed:
        if mbp1_features is None or mbp1_feature_artifact is None:
            raise ValueError(
                "the bundle resolves MBP-1 order-flow features; the exact "
                "materialized MBP-1 feature frame AND its artifact envelope "
                "are required (research-only offline evidence — never inferred)"
            )
        from .mbp1_feature_join import join_mbp1_features  # noqa: PLC0415
        from .mbp1_feature_materializer import (  # noqa: PLC0415
            verify_mbp1_feature_frame,
        )

        verify_mbp1_feature_frame(mbp1_feature_artifact, mbp1_features)
        uncovered = sorted(
            set(view.frame["candidate_id"].astype(str))
            - set(mbp1_features["candidate_id"].astype(str))
        )
        if uncovered:
            raise ValueError(
                f"{len(uncovered)} view candidate(s) have no row in the "
                "MBP-1 feature artifact — a missing ROW is a cohort "
                "misalignment (typed-null VALUES carry the registered "
                "reasons; absent rows carry none) and is refused"
            )
        frame_source = join_mbp1_features(
            view.frame, mbp1_features, feature_columns=mbp1_needed
        )
        mbp1_artifact_id = mbp1_feature_artifact.mbp1_feature_artifact_id
    elif mbp1_feature_artifact is not None:
        raise ValueError(
            "an MBP-1 feature artifact was supplied but the bundle resolves "
            "no order-flow features; refusing an inert evidence claim"
        )
    frame = frame_source.loc[:, [*_IDENTITY_COLUMNS, *names]].copy()
    payload = BundleFeatureViewPayload(
        view_id=view.view_id,
        feature_bundle_key=bundle_envelope.payload.feature_bundle_key,
        resolved_feature_bundle_id=bundle_envelope.resolved_feature_bundle_id,
        feature_registry_hash=view.feature_registry_hash,
        candidate_count=int(len(frame)),
        resolved_feature_names=names,
        mbp1_feature_artifact_id=mbp1_artifact_id,
    )
    return BundleFeatureViewEnvelope.from_payload(payload), frame


def resolve_available_bundle_view(
    view: CandidateFeatureView,
    feature_bundle_key: str,
    *,
    allow_experimental: bool = False,
    mbp1_features: pd.DataFrame | None = None,
    mbp1_feature_artifact=None,
) -> tuple[BundleFeatureViewEnvelope, pd.DataFrame]:
    """Resolve-then-scope in one step; planned blocks refuse before any frame."""

    bundle_envelope = resolve_bundle(
        feature_bundle_key, allow_experimental=allow_experimental
    )
    return build_bundle_feature_view(
        view,
        bundle_envelope,
        mbp1_features=mbp1_features,
        mbp1_feature_artifact=mbp1_feature_artifact,
    )


def bundle_categorical_features(
    names: tuple[str, ...], bundle_envelope
) -> tuple[str, ...]:
    """R6.1 (D8): the categorical features of a bundle-parametrized model =
    the frozen M0–M3 registry's categoricals ∪ every block-declared
    categorical of the bundle's resolved blocks (e.g. ``cbp_session_state``,
    the fold-local regime id) — in feature order."""

    from ..context_model import categorical_features_for  # noqa: PLC0415
    from .feature_blocks import FEATURE_BLOCK_RESOLUTION_REGISTRY  # noqa: PLC0415

    declared: set[str] = set(categorical_features_for(names))
    resolved_ids = set(bundle_envelope.payload.resolved_block_ids)
    for envelope in FEATURE_BLOCK_RESOLUTION_REGISTRY.values():
        if envelope.resolved_feature_block_id in resolved_ids:
            declared.update(envelope.payload.categorical_features)
    return tuple(name for name in names if name in declared)


def bundle_view_frame_bytes(frame: pd.DataFrame) -> bytes:
    """Deterministic Arrow bytes of a bundle-view frame (pandas metadata
    stripped; object columns carried as strings/nulls)."""

    return frame_to_arrow_bytes(frame.reset_index(drop=True))


def verify_bundle_feature_view_frame(
    envelope: BundleFeatureViewEnvelope, frame: pd.DataFrame
) -> None:
    """The frame IS the persisted view's table, or refuse (rehash)."""

    if envelope.frame_table_sha256 is None:
        raise ValueError("the bundle view envelope was never persisted with its frame")
    if bytes_sha256(bundle_view_frame_bytes(frame)) != envelope.frame_table_sha256:
        raise ValueError(
            "the supplied bundle-view frame does not hash to the persisted "
            "frame_table_sha256 — the binding is verified, never asserted"
        )


def save_bundle_feature_view(
    root: Path, envelope: BundleFeatureViewEnvelope, frame: pd.DataFrame
) -> BundleFeatureViewEnvelope:
    """Persist the view envelope + its frame (save-or-reuse). The returned
    envelope carries ``frame_table_sha256``; the identity is unchanged."""

    data = bundle_view_frame_bytes(frame)
    if len(frame) != envelope.payload.candidate_count:
        raise ValueError("bundle-view frame row count disagrees with the payload")
    bound = BundleFeatureViewEnvelope(
        bundle_feature_view_id=envelope.bundle_feature_view_id,
        payload=envelope.payload,
        frame_table_sha256=bytes_sha256(data),
    )
    stored, _reused = save_or_reuse_envelope(
        Path(root),
        BUNDLE_FEATURE_VIEW_STORE,
        bound,
        extra_files={BUNDLE_VIEW_FRAME_SIDECAR: data},
    )
    return stored


def load_bundle_feature_view(root: Path, view_id: str) -> BundleFeatureViewEnvelope:
    return load_verified_envelope(
        Path(root), BUNDLE_FEATURE_VIEW_STORE, view_id, BundleFeatureViewEnvelope
    )


def load_bundle_feature_view_frame(
    root: Path, envelope: BundleFeatureViewEnvelope
) -> pd.DataFrame:
    data = load_sidecar_bytes(
        Path(root),
        BUNDLE_FEATURE_VIEW_STORE,
        envelope.bundle_feature_view_id,
        BUNDLE_VIEW_FRAME_SIDECAR,
    )
    if envelope.frame_table_sha256 is None or bytes_sha256(data) != envelope.frame_table_sha256:
        raise ValueError("stored bundle-view frame fails the envelope hash check")
    frame = frame_from_arrow_bytes(data)
    if len(frame) != envelope.payload.candidate_count:
        raise ValueError("stored bundle-view frame row count disagrees with the payload")
    return frame


def frozen_tier_for_bundle(
    resolved_feature_names: tuple[str, ...],
) -> ContextFeatureTier | None:
    """The frozen tier whose feature SET the bundle resolves to, if any.

    The tier-frozen supervised ladder runs on the tier registry; a bundle
    whose resolved set matches a frozen tier exactly (order-independent)
    maps onto that tier. An MBP-1-bearing bundle never matches a tier — its
    ladder wiring is the R5B bundle-parametrized path.
    """

    from .feature_bundles import _TIER_TO_FROZEN  # noqa: PLC0415

    target = set(resolved_feature_names)
    for tier, frozen_key in _TIER_TO_FROZEN.items():
        if set(FROZEN_TIER_BUNDLES[frozen_key]) == target:
            return tier
    return None


def _example_bundle_view_payload() -> BundleFeatureViewPayload:
    return BundleFeatureViewPayload(
        view_id="a" * 64,
        feature_bundle_key="B0_CORE",
        resolved_feature_bundle_id="b" * 64,
        feature_registry_hash="c" * 64,
        candidate_count=1,
        resolved_feature_names=("direction",),
    )


register_identity_pair(
    name="BundleFeatureView",
    envelope_cls=BundleFeatureViewEnvelope,
    payload_cls=BundleFeatureViewPayload,
    id_field="bundle_feature_view_id",
    example_factory=_example_bundle_view_payload,
    extra_envelope_fields=("frame_table_sha256",),
)
