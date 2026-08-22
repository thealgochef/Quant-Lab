"""Bundle-scoped feature views over the AVAILABLE blocks (PHASED R5).

`resolve_bundle` is the only gate onto a view: a bundle containing any
non-available block (`IFVG_ORDER_FLOW_MBP1_V1` until R5B, regime/key-level/
execution-liquidity until their releases) refuses with the block's status
and reason BEFORE any frame is touched, so no baseline-vs-MBP-1 study is
constructible in R5. The resulting view identity binds the immutable
candidate view to the exact resolved bundle.
"""

from __future__ import annotations

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
from .feature_bundles import FROZEN_TIER_BUNDLES, resolve_bundle
from .mbp1_source_contract import assert_no_deep_book_identifiers

__all__ = [
    "BundleFeatureViewPayload",
    "BundleFeatureViewEnvelope",
    "build_bundle_feature_view",
    "resolve_available_bundle_view",
    "frozen_tier_for_bundle",
]


class BundleFeatureViewPayload(FrozenContract):
    """One immutable candidate view scoped to one resolved bundle."""

    view_id: str = Field(pattern=SHA256_PATTERN)
    feature_bundle_key: str
    resolved_feature_bundle_id: str = Field(pattern=SHA256_PATTERN)
    feature_registry_hash: str = Field(pattern=SHA256_PATTERN)
    candidate_count: int = Field(ge=0)
    resolved_feature_names: tuple[str, ...]


class BundleFeatureViewEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "bundle_feature_view_id"

    bundle_feature_view_id: str = Field(pattern=SHA256_PATTERN)
    payload: BundleFeatureViewPayload


def build_bundle_feature_view(
    view: CandidateFeatureView,
    bundle_envelope,
) -> tuple[BundleFeatureViewEnvelope, pd.DataFrame]:
    """Scope the candidate view's frame to the bundle's resolved features.

    Every resolved feature must exist as a view column (fail-closed — a
    bundle can never silently widen or narrow against the immutable view),
    and no deeper-than-MBP-1 identifier can appear (defense in depth on top
    of the registry guards).
    """

    names = tuple(bundle_envelope.payload.resolved_feature_names)
    assert_no_deep_book_identifiers(names)
    missing = sorted(set(names) - set(view.frame.columns))
    if missing:
        raise ValueError(
            f"bundle features are missing from the immutable candidate view: {missing}"
        )
    frame = view.frame.loc[:, [*_IDENTITY_COLUMNS, *names]].copy()
    payload = BundleFeatureViewPayload(
        view_id=view.view_id,
        feature_bundle_key=bundle_envelope.payload.feature_bundle_key,
        resolved_feature_bundle_id=bundle_envelope.resolved_feature_bundle_id,
        feature_registry_hash=view.feature_registry_hash,
        candidate_count=int(len(frame)),
        resolved_feature_names=names,
    )
    return BundleFeatureViewEnvelope.from_payload(payload), frame


def resolve_available_bundle_view(
    view: CandidateFeatureView,
    feature_bundle_key: str,
    *,
    allow_experimental: bool = False,
) -> tuple[BundleFeatureViewEnvelope, pd.DataFrame]:
    """Resolve-then-scope in one step; planned blocks refuse before any frame."""

    bundle_envelope = resolve_bundle(
        feature_bundle_key, allow_experimental=allow_experimental
    )
    return build_bundle_feature_view(view, bundle_envelope)


def frozen_tier_for_bundle(
    resolved_feature_names: tuple[str, ...],
) -> ContextFeatureTier | None:
    """The frozen tier whose feature SET the bundle resolves to, if any.

    The R5 supervised ladder runs on the tier registry; a bundle whose
    resolved set matches a frozen tier exactly (order-independent) maps onto
    that tier — anything else has no ladder wiring in R5 and the caller must
    fail closed rather than improvise a feature list.
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
)
