"""Declared contrasts: charter-gated, balanced, paired, observational (DT §5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from alpha_lab.agents.data_infra.ifvg.study.contrasts import (
    CONTRAST_BOOTSTRAP_PROTOCOL_ID,
    DeclaredContrastEnvelope,
    DeclaredContrastPayload,
    PostHocContrastError,
    UnbalancedDesignError,
    evaluate_declared_contrast,
)


def _charter_for(*contrast_ids: str):
    """A duck-typed frozen-charter view: search_id + declared ids only."""

    return SimpleNamespace(
        search_id="9" * 64,
        payload=SimpleNamespace(declared_contrast_ids=tuple(contrast_ids)),
    )

_AXIS = "strategy_profile.parent_retest_timeout_1m_bars"
_OTHER = "strategy_profile.entry_near_parent"

_CELLS = {
    "a" * 64: {_AXIS: "none", _OTHER: "false"},
    "b" * 64: {_AXIS: "240", _OTHER: "false"},
    "c" * 64: {_AXIS: "none", _OTHER: "within_40"},
    "d" * 64: {_AXIS: "240", _OTHER: "within_40"},
}

_METRICS = {
    "a" * 64: {"net_expectancy_r": 0.10},
    "b" * 64: {"net_expectancy_r": 0.16},
    "c" * 64: {"net_expectancy_r": 0.05},
    "d" * 64: {"net_expectancy_r": 0.13},
}


def _contrast(cells=None, conditioning=None) -> DeclaredContrastEnvelope:
    return DeclaredContrastEnvelope.from_payload(
        DeclaredContrastPayload(
            charter_id="9" * 64,
            axis_dimension_ids=(_AXIS,),
            conditioning=conditioning or {},
            cell_ids=tuple(cells or _CELLS),
            effect_kind="main_effect",
        )
    )


def test_paired_main_effect_with_seed7_bootstrap() -> None:
    contrast = _contrast()
    result = evaluate_declared_contrast(
        contrast,
        charter=_charter_for(contrast.contrast_id),
        cell_axis_values=_CELLS,
        cell_metrics=_METRICS,
        metrics=("net_expectancy_r",),
    )
    assert result.denominator == 4
    assert len(result.matched_pairs) == 2  # one pair per stratum of the other axis
    # pair orientation is the deterministic lexicographic value sort:
    # "240" < "none", so each pair is (240-cell, none-cell) and the delta is
    # none − 240 — the recorded pairs disambiguate the direction exactly
    assert result.matched_pairs == (("b" * 64, "a" * 64), ("d" * 64, "c" * 64))
    estimate = result.effect_estimate["net_expectancy_r"]
    assert estimate["n_pairs"] == 2
    assert estimate["paired_mean_delta"] == pytest.approx(-(0.06 + 0.08) / 2)
    assert estimate["bootstrap_protocol_id"] == CONTRAST_BOOTSTRAP_PROTOCOL_ID
    low, high = estimate["bootstrap_ci95"]
    assert low <= estimate["paired_mean_delta"] <= high
    assert result.causal_language_permitted is False
    assert "no causal claim" in estimate["wording"]
    # determinism: the seed-7 protocol reproduces the identical interval
    rerun = evaluate_declared_contrast(
        contrast,
        charter=_charter_for(contrast.contrast_id),
        cell_axis_values=_CELLS,
        cell_metrics=_METRICS,
        metrics=("net_expectancy_r",),
    )
    assert rerun.effect_estimate["net_expectancy_r"]["bootstrap_ci95"] == (low, high)


def test_post_hoc_contrast_is_refused() -> None:
    contrast = _contrast()
    with pytest.raises(PostHocContrastError, match="post-hoc"):
        evaluate_declared_contrast(
            contrast,
            charter=_charter_for("0" * 64),  # the charter declared something else
            cell_axis_values=_CELLS,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )
    # a contrast bound to a DIFFERENT charter refuses even when its id is listed
    foreign = SimpleNamespace(
        search_id="1" * 64,
        payload=SimpleNamespace(declared_contrast_ids=(contrast.contrast_id,)),
    )
    with pytest.raises(PostHocContrastError, match="different charter"):
        evaluate_declared_contrast(
            contrast,
            charter=foreign,
            cell_axis_values=_CELLS,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )


def _interaction(cells=None) -> DeclaredContrastEnvelope:
    return DeclaredContrastEnvelope.from_payload(
        DeclaredContrastPayload(
            charter_id="9" * 64,
            axis_dimension_ids=(_AXIS, _OTHER),
            conditioning={},
            cell_ids=tuple(cells or _CELLS),
            effect_kind="interaction",
        )
    )


def test_balanced_two_way_interaction_evaluates_at_r4() -> None:
    """DEV-R2-5 closure (§7A.19.11): the declared 2×2 interaction computes
    the deterministic lexicographic difference-of-differences."""

    interaction = _interaction()
    result = evaluate_declared_contrast(
        interaction,
        charter=_charter_for(interaction.contrast_id),
        cell_axis_values=_CELLS,
        cell_metrics=_METRICS,
        metrics=("net_expectancy_r",),
    )
    assert result.denominator == 4
    # one quartet → the two oriented A-pairs, low-B ("false") pair first
    assert result.matched_pairs == (("b" * 64, "a" * 64), ("d" * 64, "c" * 64))
    estimate = result.effect_estimate["net_expectancy_r"]
    assert estimate["n_quartets"] == 1
    # ("240" < "none", "false" < "within_40"):
    # (0.05 − 0.13) − (0.10 − 0.16) = −0.02
    assert estimate["paired_mean_delta"] == pytest.approx(-0.02)
    assert estimate["bootstrap_ci95"] is None  # typed None below two quartets
    assert estimate["bootstrap_protocol_id"] == CONTRAST_BOOTSTRAP_PROTOCOL_ID
    assert "no causal claim" in estimate["wording"]
    assert result.causal_language_permitted is False


def test_stratified_interaction_pairs_and_seed7_determinism() -> None:
    """Two strata of a third axis → two quartets, four recorded pairs, and a
    deterministic seed-7 bootstrap interval over the interaction deltas."""

    third = "strategy_profile.enable_shorts"
    cells = {}
    metrics = {}
    values = {
        ("none", "false"): 0.10,
        ("240", "false"): 0.16,
        ("none", "within_40"): 0.05,
        ("240", "within_40"): 0.13,
    }
    for index, stratum in enumerate(("false", "true")):
        for offset, ((a, b), value) in enumerate(sorted(values.items())):
            cell = f"{index}{offset}" * 32
            cells[cell] = {_AXIS: a, _OTHER: b, third: stratum}
            metrics[cell] = {"net_expectancy_r": value + 0.01 * index}
    interaction = _interaction(cells=cells)
    result = evaluate_declared_contrast(
        interaction,
        charter=_charter_for(interaction.contrast_id),
        cell_axis_values=cells,
        cell_metrics=metrics,
        metrics=("net_expectancy_r",),
    )
    estimate = result.effect_estimate["net_expectancy_r"]
    assert estimate["n_quartets"] == 2
    assert len(result.matched_pairs) == 4
    # the constant per-stratum shift cancels in the double difference
    assert estimate["paired_mean_delta"] == pytest.approx(-0.02)
    first = estimate["bootstrap_ci95"]
    rerun = evaluate_declared_contrast(
        interaction,
        charter=_charter_for(interaction.contrast_id),
        cell_axis_values=cells,
        cell_metrics=metrics,
        metrics=("net_expectancy_r",),
    )
    assert rerun.effect_estimate["net_expectancy_r"]["bootstrap_ci95"] == first


def test_interaction_grid_hole_refuses_never_imputes() -> None:
    cells = {key: value for key, value in _CELLS.items() if key != "d" * 64}
    interaction = _interaction(cells=cells)
    with pytest.raises(UnbalancedDesignError, match="not fully crossed"):
        evaluate_declared_contrast(
            interaction,
            charter=_charter_for(interaction.contrast_id),
            cell_axis_values=cells,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )


def test_higher_order_designs_are_refused() -> None:
    three_way = DeclaredContrastEnvelope.from_payload(
        DeclaredContrastPayload(
            charter_id="9" * 64,
            axis_dimension_ids=(_AXIS, _OTHER, "strategy_profile.enable_shorts"),
            conditioning={},
            cell_ids=tuple(_CELLS),
            effect_kind="interaction",
        )
    )
    with pytest.raises(UnbalancedDesignError, match="two-way interactions"):
        evaluate_declared_contrast(
            three_way,
            charter=_charter_for(three_way.contrast_id),
            cell_axis_values=_CELLS,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )


def test_duplicate_grid_position_is_ambiguous_and_refused() -> None:
    cells = dict(_CELLS)
    cells["e" * 64] = dict(cells["a" * 64])  # a second cell on the same position
    contrast = _contrast(cells=cells)
    with pytest.raises(UnbalancedDesignError, match="one grid position"):
        evaluate_declared_contrast(
            contrast,
            charter=_charter_for(contrast.contrast_id),
            cell_axis_values=cells,
            cell_metrics={**_METRICS, "e" * 64: {"net_expectancy_r": 0.2}},
            metrics=("net_expectancy_r",),
        )


def test_declared_contrast_ids_are_inside_the_charter_identity() -> None:
    """F3: declarations are frozen — adding one mints a NEW search_id."""

    from tests.agents.ifvg_search.test_orchestrator import _charter

    base = _charter()
    amended = type(base).from_payload(
        base.payload.model_copy(update={"declared_contrast_ids": ("f" * 64,)})
    )
    assert amended.search_id != base.search_id


def test_grid_hole_is_unbalanced_never_imputed() -> None:
    cells = {key: value for key, value in _CELLS.items() if key != "d" * 64}
    contrast = _contrast(cells=cells)
    with pytest.raises(UnbalancedDesignError, match="not fully crossed"):
        evaluate_declared_contrast(
            contrast,
            charter=_charter_for(contrast.contrast_id),
            cell_axis_values=cells,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )


def test_conditioning_slice_membership_is_enforced() -> None:
    contrast = _contrast(conditioning={_OTHER: "false"})
    with pytest.raises(UnbalancedDesignError, match="outside the conditioning slice"):
        evaluate_declared_contrast(
            contrast,
            charter=_charter_for(contrast.contrast_id),
            cell_axis_values=_CELLS,
            cell_metrics=_METRICS,
            metrics=("net_expectancy_r",),
        )


def test_missing_paired_metric_refuses() -> None:
    metrics = {key: dict(value) for key, value in _METRICS.items()}
    del metrics["d" * 64]["net_expectancy_r"]
    contrast = _contrast()
    with pytest.raises(UnbalancedDesignError, match="missing for a paired cell"):
        evaluate_declared_contrast(
            contrast,
            charter=_charter_for(contrast.contrast_id),
            cell_axis_values=_CELLS,
            cell_metrics=metrics,
            metrics=("net_expectancy_r",),
        )


def test_single_pair_reports_typed_none_ci() -> None:
    cells = {
        "a" * 64: {_AXIS: "none"},
        "b" * 64: {_AXIS: "240"},
    }
    contrast = _contrast(cells=cells)
    result = evaluate_declared_contrast(
        contrast,
        charter=_charter_for(contrast.contrast_id),
        cell_axis_values=cells,
        cell_metrics=_METRICS,
        metrics=("net_expectancy_r",),
    )
    estimate = result.effect_estimate["net_expectancy_r"]
    assert estimate["n_pairs"] == 1
    assert estimate["bootstrap_ci95"] is None  # typed None below two pairs
