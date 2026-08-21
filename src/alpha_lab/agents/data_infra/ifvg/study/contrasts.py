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
    if len(payload.axis_dimension_ids) == 1:
        return _evaluate_main_effect(
            contrast,
            cell_axis_values=cell_axis_values,
            cell_metrics=cell_metrics,
            metrics=metrics,
        )
    if len(payload.axis_dimension_ids) == 2:
        return _evaluate_interaction(
            contrast,
            cell_axis_values=cell_axis_values,
            cell_metrics=cell_metrics,
            metrics=metrics,
        )
    raise UnbalancedDesignError(
        "only main effects (1 axis) and two-way interactions (2 axes) are "
        "declarable (DT §5); higher-order designs have no registered "
        "adjustment method"
    )


def _declared_cells(
    payload: DeclaredContrastPayload,
    cell_axis_values: Mapping[str, Mapping[str, str]],
) -> list[str]:
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
    return cells


def _two_values(
    axis: str,
    cells: list[str],
    cell_axis_values: Mapping[str, Mapping[str, str]],
    *,
    design: str,
) -> list[str]:
    values = sorted({cell_axis_values[cell][axis] for cell in cells})
    if len(values) != 2:
        raise UnbalancedDesignError(
            f"{design} requires exactly two registered values on {axis!r}; "
            f"got {values}"
        )
    return values


def _metric_value(
    cell_metrics: Mapping[str, Mapping[str, float]], cell: str, metric: str
) -> float:
    value = cell_metrics.get(cell, {}).get(metric)
    if value is None:
        raise UnbalancedDesignError(f"metric {metric!r} is missing for a paired cell")
    return value


def _estimate_entry(deltas: list[float], *, n_key: str, wording: str) -> ImmutableMap:
    return ImmutableMap(
        {
            "paired_mean_delta": sum(deltas) / len(deltas) if deltas else None,
            "paired_deltas": tuple(deltas),
            n_key: len(deltas),
            "bootstrap_ci95": _paired_bootstrap_ci(deltas),
            "bootstrap_protocol_id": CONTRAST_BOOTSTRAP_PROTOCOL_ID,
            "wording": wording,
        }
    )


def _evaluate_main_effect(
    contrast: DeclaredContrastEnvelope,
    *,
    cell_axis_values: Mapping[str, Mapping[str, str]],
    cell_metrics: Mapping[str, Mapping[str, float]],
    metrics: tuple[str, ...],
) -> ContrastResult:
    payload = contrast.payload
    axis = payload.axis_dimension_ids[0]
    cells = _declared_cells(payload, cell_axis_values)
    values = _two_values(axis, cells, cell_axis_values, design="main effect")
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
        deltas = [
            _metric_value(cell_metrics, right, metric)
            - _metric_value(cell_metrics, left, metric)
            for left, right in pairs
        ]
        estimate[metric] = _estimate_entry(
            deltas,
            n_key="n_pairs",
            wording="observational paired difference; no causal claim",
        )
    return ContrastResult(
        contrast_id=contrast.contrast_id,
        matched_pairs=tuple(pairs),
        effect_estimate=estimate,
        denominator=len(cells),
    )


def _evaluate_interaction(
    contrast: DeclaredContrastEnvelope,
    *,
    cell_axis_values: Mapping[str, Mapping[str, str]],
    cell_metrics: Mapping[str, Mapping[str, float]],
    metrics: tuple[str, ...],
) -> ContrastResult:
    """Balanced two-way interaction over the fully crossed 2×2 grid (R4).

    Per stratum of every OTHER axis, the interaction delta is the
    difference-of-differences with deterministic lexicographic orientation
    (DEV-R2-6): ``(m[a1,b1] − m[a0,b1]) − (m[a1,b0] − m[a0,b0])`` where
    ``a0 < a1`` and ``b0 < b1`` sort by value token. Any hole, duplicate
    grid position, or missing metric refuses with
    :class:`UnbalancedDesignError` — never an imputation.
    ``matched_pairs`` records the two oriented A-pairs per stratum
    (the ``b0`` pair first), so the double difference is reconstructible.
    """

    payload = contrast.payload
    axis_a, axis_b = payload.axis_dimension_ids
    cells = _declared_cells(payload, cell_axis_values)
    values_a = _two_values(axis_a, cells, cell_axis_values, design="interaction")
    values_b = _two_values(axis_b, cells, cell_axis_values, design="interaction")
    other_axes = sorted(
        {
            key
            for cell in cells
            for key in cell_axis_values[cell]
            if key not in (axis_a, axis_b)
        }
    )
    strata: dict[tuple[str, ...], dict[tuple[str, str], str]] = {}
    for cell in cells:
        stratum = tuple(cell_axis_values[cell].get(key, "") for key in other_axes)
        grid = strata.setdefault(stratum, {})
        position = (cell_axis_values[cell][axis_a], cell_axis_values[cell][axis_b])
        if position in grid and grid[position] != cell:
            raise UnbalancedDesignError(
                f"two declared cells occupy one grid position (stratum "
                f"{stratum!r}, values {position!r}) — the design is ambiguous"
            )
        grid[position] = cell
    quartets: list[dict[str, str]] = []
    pairs: list[tuple[str, str]] = []
    required = [(a, b) for b in values_b for a in values_a]
    for stratum, grid in sorted(strata.items()):
        if set(grid) != set(required):
            missing = sorted(set(required) - set(grid))
            raise UnbalancedDesignError(
                f"interaction grid is not fully crossed at stratum "
                f"{stratum!r} (missing positions {missing}) and no "
                "adjustment method is registered"
            )
        quartet = {
            "a0b0": grid[(values_a[0], values_b[0])],
            "a1b0": grid[(values_a[1], values_b[0])],
            "a0b1": grid[(values_a[0], values_b[1])],
            "a1b1": grid[(values_a[1], values_b[1])],
        }
        quartets.append(quartet)
        pairs.append((quartet["a0b0"], quartet["a1b0"]))
        pairs.append((quartet["a0b1"], quartet["a1b1"]))
    estimate: dict[str, Any] = {}
    for metric in metrics:
        deltas = []
        for quartet in quartets:
            low_b = _metric_value(
                cell_metrics, quartet["a1b0"], metric
            ) - _metric_value(cell_metrics, quartet["a0b0"], metric)
            high_b = _metric_value(
                cell_metrics, quartet["a1b1"], metric
            ) - _metric_value(cell_metrics, quartet["a0b1"], metric)
            deltas.append(high_b - low_b)
        estimate[metric] = _estimate_entry(
            deltas,
            n_key="n_quartets",
            wording=(
                "observational paired difference-of-differences; "
                "no causal claim"
            ),
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
