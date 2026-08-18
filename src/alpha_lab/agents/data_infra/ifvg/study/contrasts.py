"""Declared main effects and interactions (DT §5; brief §7A.9).

Only charter-declared contrasts evaluate — a ``contrast_id`` absent from the
frozen charter's declared list is refused (post-hoc contrasts are
structurally impossible). Unbalanced grids without a registered adjustment
refuse (`UnbalancedDesignError`); raw child results are always retained
(contrasts reference cells, never replace them); wording is observational.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field

from ..search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    register_identity_pair,
)

#: The registered uncertainty protocol for paired contrast deltas: the seed-7
#: pair-level bootstrap (10,000 resamples, 95% interval). Deterministic; the
#: same seed the study-lane bootstrap protocol id pins.
CONTRAST_BOOTSTRAP_PROTOCOL_ID = "paired_cell_bootstrap_10000_seed7_v1"
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 7

__all__ = [
    "InteractionEvaluationUnavailableError",
    "DeclaredContrastPayload",
    "DeclaredContrastEnvelope",
    "ContrastResult",
    "UnbalancedDesignError",
    "PostHocContrastError",
    "evaluate_declared_contrast",
]


class UnbalancedDesignError(ValueError):
    """The axis grid is not fully crossed over the conditioning slice."""


class PostHocContrastError(PermissionError):
    """The contrast is not declared in the frozen charter."""


class InteractionEvaluationUnavailableError(ValueError):
    """Interaction evaluation is not yet wired (declared contracts stand).

    Two-axis interactions remain fully DECLARABLE (payload + identity are
    complete); their evaluation lands with the comparison surfaces (R4,
    §7A.19.11). Nothing silently computes in the meantime.
    """


class DeclaredContrastPayload(FrozenContract):
    charter_id: str = Field(pattern=SHA256_PATTERN)
    axis_dimension_ids: tuple[str, ...]  # 1 = main effect, 2 = two-way interaction
    conditioning: ImmutableMap[str, str]
    cell_ids: tuple[str, ...]  # the exact compatible child set — the denominator
    effect_kind: Literal[
        "main_effect",
        "interaction",
        "conditional_effect",
        "neighbor_effect",
        "firm_specific_effect",
        "regime_specific_effect",
    ]


class DeclaredContrastEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "contrast_id"

    contrast_id: str = Field(pattern=SHA256_PATTERN)
    payload: DeclaredContrastPayload


class ContrastResult(FrozenContract):
    contrast_id: str = Field(pattern=SHA256_PATTERN)
    matched_pairs: tuple[tuple[str, str], ...]
    effect_estimate: ImmutableMap[str, Any]
    denominator: int
    causal_language_permitted: Literal[False] = False


def _paired_bootstrap_ci(deltas: list[float]) -> tuple[float, float] | None:
    """Seed-7 pair-resampling 95% CI; typed None below two pairs (no fuzzing)."""

    if len(deltas) < 2:
        return None
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    indices = rng.integers(0, len(values), size=(_BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def evaluate_declared_contrast(
    contrast: DeclaredContrastEnvelope,
    *,
    charter,
    cell_axis_values: Mapping[str, Mapping[str, str]],
    cell_metrics: Mapping[str, Mapping[str, float]],
    metrics: tuple[str, ...],
) -> ContrastResult:
    """Paired axis-value deltas over the declared, fully crossed child grid.

    ``cell_axis_values`` maps cell_id → {axis_dimension_id: value_id};
    ``cell_metrics`` maps cell_id → {metric: value}. Pairing is exact: for a
    main effect over axis A, cells matching on every OTHER axis (and the
    conditioning slice) pair across A's two extreme registered values; a hole
    in the grid is an :class:`UnbalancedDesignError`, never an imputation.
    """

    if contrast.payload.charter_id != charter.search_id:
        raise PostHocContrastError(
            "contrast is bound to a different charter; declarations are "
            "anchored in the frozen charter identity"
        )
    if contrast.contrast_id not in charter.payload.declared_contrast_ids:
        raise PostHocContrastError(
            "contrast is not declared in the frozen charter's "
            "declared_contrast_ids; post-hoc contrasts are refused"
        )
    payload = contrast.payload
    if len(payload.axis_dimension_ids) != 1:
        raise InteractionEvaluationUnavailableError(
            "two-axis interaction evaluation lands with the comparison "
            "surfaces (R4); the declaration remains valid and unevaluated"
        )
    axis = payload.axis_dimension_ids[0]
    cells = [cell for cell in payload.cell_ids if cell in cell_axis_values]
    if set(cells) != set(payload.cell_ids):
        missing = sorted(set(payload.cell_ids) - set(cells))
        raise UnbalancedDesignError(f"declared cells lack axis values: {missing}")
    for cell in cells:
        for key, value in payload.conditioning.items():
            if cell_axis_values[cell].get(key) != value:
                raise UnbalancedDesignError(
                    f"cell {cell[:12]}… is outside the conditioning slice"
                )
    values = sorted({cell_axis_values[cell][axis] for cell in cells})
    if len(values) != 2:
        raise UnbalancedDesignError(
            f"main effect requires exactly two registered values on {axis!r}; "
            f"got {values}"
        )
    other_axes = sorted(
        {key for cell in cells for key in cell_axis_values[cell] if key != axis}
    )
    strata: dict[tuple[str, ...], dict[str, str]] = {}
    for cell in cells:
        stratum = tuple(cell_axis_values[cell].get(key, "") for key in other_axes)
        position = strata.setdefault(stratum, {})
        value = cell_axis_values[cell][axis]
        if value in position and position[value] != cell:
            raise UnbalancedDesignError(
                f"two declared cells occupy one grid position (stratum "
                f"{stratum!r}, value {value!r}) — the pairing is ambiguous"
            )
        position[value] = cell
    pairs: list[tuple[str, str]] = []
    for stratum, by_value in sorted(strata.items()):
        if set(by_value) != set(values):
            raise UnbalancedDesignError(
                f"axis grid is not fully crossed at stratum {stratum!r} "
                "and no adjustment method is registered"
            )
        pairs.append((by_value[values[0]], by_value[values[1]]))
    estimate: dict[str, Any] = {}
    for metric in metrics:
        deltas = []
        for left, right in pairs:
            left_value = cell_metrics.get(left, {}).get(metric)
            right_value = cell_metrics.get(right, {}).get(metric)
            if left_value is None or right_value is None:
                raise UnbalancedDesignError(
                    f"metric {metric!r} is missing for a paired cell"
                )
            deltas.append(right_value - left_value)
        estimate[metric] = ImmutableMap(
            {
                "paired_mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "paired_deltas": tuple(deltas),
                "n_pairs": len(deltas),
                "bootstrap_ci95": _paired_bootstrap_ci(deltas),
                "bootstrap_protocol_id": CONTRAST_BOOTSTRAP_PROTOCOL_ID,
                "wording": "observational paired difference; no causal claim",
            }
        )
    return ContrastResult(
        contrast_id=contrast.contrast_id,
        matched_pairs=tuple(pairs),
        effect_estimate=estimate,
        denominator=len(cells),
    )


register_identity_pair(
    name="DeclaredContrast",
    envelope_cls=DeclaredContrastEnvelope,
    payload_cls=DeclaredContrastPayload,
    id_field="contrast_id",
    example_factory=lambda: DeclaredContrastPayload(
        charter_id="a" * 64,
        axis_dimension_ids=("strategy_profile.parent_retest_timeout_1m_bars",),
        conditioning={},
        cell_ids=("b" * 64, "c" * 64),
        effect_kind="main_effect",
    ),
)
