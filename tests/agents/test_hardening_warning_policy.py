"""HARDENING-BACKEND §4.5 (F-18) — the warning policy.

Project-owned warnings are fixed with explicit, schema-aligned typed frames:
``concat_schema_aligned`` reproduces the PRE-deprecation pandas result (all-NA
entries excluded from dtype determination) without emitting the
``FutureWarning`` and never drops an all-null column. The registered pytest
policy runs the suite under ``filterwarnings = error`` with exactly ONE
narrowly scoped third-party rule.
"""

from __future__ import annotations

import tomllib
import warnings
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alpha_lab.agents.data_infra.ifvg.dataset import concat_schema_aligned

_REPO = Path(__file__).resolve().parents[2]


def _old_concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """The pre-hardening call (the deprecated all-NA exclusion), silenced
    ONLY here to produce the reference result the new helper must equal."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(
            [frame for frame in frames if not frame.empty], ignore_index=True, sort=False
        )


def _assert_no_warning_and_equal(frames: list[pd.DataFrame]) -> pd.DataFrame:
    reference = _old_concat(frames)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = concat_schema_aligned(frames)
    assert list(result.columns) == list(reference.columns)
    assert result.dtypes.to_dict() == reference.dtypes.to_dict()
    assert_frame_equal(result, reference)
    return result


def test_all_na_object_column_against_typed_columns_keeps_the_old_dtypes():
    """The classic warning shape: an all-None object column in one frame and
    a typed (float / bool / datetime) column in another — the old result
    dtype (from the non-NA entries) is reproduced without a warning."""

    stamp = pd.Timestamp("2026-01-13T14:00:00Z")
    first = pd.DataFrame(
        {
            "trace": [1, 2],
            "amount": [None, None],
            "flag": [None, None],
            "at": [None, None],
        }
    )
    second = pd.DataFrame(
        {
            "trace": [3],
            "amount": [1.5],
            "flag": [True],
            "at": [stamp],
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pd.concat([first, second], ignore_index=True, sort=False)
    assert any(issubclass(w.category, FutureWarning) for w in caught), (
        "the fixture must reproduce the deprecated all-NA concat shape"
    )
    result = _assert_no_warning_and_equal([first, second])
    assert result["amount"].dtype == np.dtype("float64")
    assert result["at"].dtype == pd.DatetimeTZDtype("ns", "UTC")
    assert result["amount"].isna().tolist() == [True, True, False]


def test_all_null_columns_are_never_dropped_and_missing_columns_align():
    """A column that is all-NA in EVERY frame survives with its dtype; a
    column absent from one frame is aligned as typed NA (never dropped)."""

    first = pd.DataFrame({"a": [1, 2], "always_null": [None, None], "only_first": [1.0, 2.0]})
    second = pd.DataFrame({"a": [3], "always_null": [None], "only_second": ["x"]})
    result = _assert_no_warning_and_equal([first, second])
    assert "always_null" in result.columns
    assert result["always_null"].isna().all()
    assert list(result.columns) == ["a", "always_null", "only_first", "only_second"]
    assert result["only_first"].isna().tolist() == [False, False, True]
    assert result["only_second"].isna().tolist() == [True, True, False]


def test_empty_plus_nonempty_and_all_empty_inputs():
    """Genuinely EMPTY frames are excluded before the concat (the existing
    behavior); an all-empty input yields an empty frame."""

    empty = pd.DataFrame()
    typed = pd.DataFrame({"a": [1], "b": ["s"]})
    result = _assert_no_warning_and_equal([empty, typed, empty])
    assert len(result) == 1
    assert concat_schema_aligned([empty, pd.DataFrame()]).empty


def test_all_nan_float_column_against_object_strings():
    """An all-NaN float64 entry against an object (string) entry: the old
    result is object with NaN preserved as missing."""

    first = pd.DataFrame({"k": [1, 2], "label": [np.nan, np.nan]})
    second = pd.DataFrame({"k": [3, 4], "label": ["x", "y"]})
    result = _assert_no_warning_and_equal([first, second])
    assert result["label"].dtype == np.dtype("O")
    assert result["label"].isna().tolist() == [True, True, False, False]


def test_int_and_bool_columns_against_all_na_entries_follow_the_old_value_concat():
    """int64 / bool non-NA entries plus an all-NA PRESENT entry: the old path
    excluded the entry from dtype determination but concatenated its raw
    values, so an all-None object entry yields object (ints + None kept
    exactly), an all-NaN float entry yields float64, and a bool column with
    an all-None entry yields object — all reproduced without a warning."""

    first = pd.DataFrame({"k": ["a", "b"], "count": [None, None], "flag": [None, None]})
    second = pd.DataFrame({"k": ["c"], "count": [7], "flag": [True]})
    result = _assert_no_warning_and_equal([first, second])
    assert result["count"].dtype == np.dtype("O")
    assert result["count"].tolist() == [None, None, 7]
    assert result["flag"].dtype == np.dtype("O")
    third = pd.DataFrame({"k": ["a", "b"], "count": [np.nan, np.nan]})
    fourth = pd.DataFrame({"k": ["c"], "count": [7]})
    result = _assert_no_warning_and_equal([third, fourth])
    assert result["count"].dtype == np.dtype("float64")
    assert result["count"].tolist()[2] == 7.0
    # an ABSENT int column widens to float64 (pandas' ensure_dtype_can_hold_na)
    fifth = pd.DataFrame({"k": ["z"]})
    result = _assert_no_warning_and_equal([fourth, fifth])
    assert result["count"].dtype == np.dtype("float64")
    assert result["count"].isna().tolist() == [False, True]


def test_real_audit_frames_concat_is_byte_identical_to_the_old_path():
    """The production shape (§4.5): the per-day audit-channel frames of the
    deterministic synthetic FSM chain concatenate to the exact frame the
    deprecated call produced (columns, dtypes, values) — and no warning."""

    from strategy_core.strategies.ifvg_smc.replay import run_day
    from strategy_core.strategies.ifvg_smc.section import default_ifvg_smc_section

    from alpha_lab.agents.data_infra.ifvg.capture_driver import flatten_audit_emissions
    from tests.agents.test_ifvg_fsm_audit_contracts import _DAY0, _three_days

    section = default_ifvg_smc_section()
    seed = None
    audit_frames: list[pd.DataFrame] = []
    days = _three_days()
    core_offset = 0
    for day_idx, by_tf in enumerate(days):
        result = run_day(
            by_tf,
            section=section,
            seed=seed,
            trading_day=_DAY0 + timedelta(days=day_idx),
            dataset_exhausted=day_idx == len(days) - 1,
            audit_capture_mode="fsm_audit_v1",
        )
        seed = result.end_seed
        audit_frame = flatten_audit_emissions(result.audit_emissions, entering_seed_hash=None)
        if not audit_frame.empty:
            audit_frame["is_warmup"] = day_idx < 1
            audit_frame["days_of_htf_history"] = day_idx
            audit_frame["evaluation_config_hash"] = "test-eval-hash"
            audit_frame["core_trace_ordinal_before_global"] = (
                audit_frame["stamp_core_trace_ordinal_before"].astype(int) + core_offset
            )
            audit_frame["core_trace_ordinal_after_global"] = (
                audit_frame["stamp_core_trace_ordinal_after"].astype(int) + core_offset
            )
        core_offset += len(result.emissions)
        audit_frames.append(audit_frame)
    assert any(not frame.empty for frame in audit_frames)
    _assert_no_warning_and_equal(audit_frames)


def test_registered_pytest_warning_policy_is_error_with_one_exact_third_party_rule():
    """The suite runs under ``filterwarnings = error``; the ONLY ignore rule is
    the narrowly scoped sklearn / SciPy L-BFGS-B deprecation (exact message,
    category and module) — no broad DeprecationWarning suppression."""

    config = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    rules = config["tool"]["pytest"]["ini_options"]["filterwarnings"]
    assert rules[0] == "error"
    ignores = [rule for rule in rules[1:] if rule.startswith("ignore")]
    assert len(ignores) == 1 and len(rules) == 2
    action, message, category, module = ignores[0].split(":")[:4]
    assert action == "ignore"
    assert category == "DeprecationWarning"
    assert module == r"sklearn\.linear_model\._logistic"
    assert "L-BFGS-B" in message and "disp" in message and "iprint" in message
    assert "SciPy 1\\.18\\.0" in message
    # the baseline record agrees with the registered rule
    baseline = (
        _REPO
        / "QL-FSM-PROP-SEARCH-DASHBOARD"
        / "implementation-progress"
        / "HARDENING-BACKEND"
        / "WARNING_BASELINE.json"
    )
    if baseline.exists():  # evidence folders are untracked; guard the clone case
        import json

        record = json.loads(baseline.read_text(encoding="utf-8"))
        assert record["pytest_filterwarnings"] == rules
        assert record["project_owned_warnings_remaining"] == 0
        assert len(record["third_party_rules"]) == 1


def test_typed_row_append_replaces_loc_enlargement_without_a_warning():
    """The test-side fix (:260): appending a row that carries NaT / None
    through a typed one-row frame reproduces the old ``loc`` enlargement
    result (object ``binary_target``, NaT resolution) without the warning."""

    from tests.agents.test_ifvg_context_experiment_engine import (
        _fold_frame,
        append_censored_row,
    )

    days = tuple(
        (datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=index)).date().isoformat()
        for index in range(45)
    )
    frame = _fold_frame(days)
    censored = frame.loc[40].copy()
    censored["candidate_id"] = "censored-test-sibling"
    censored["resolution_ts_utc"] = pd.NaT
    censored["resolution_available"] = False
    censored["binary_target"] = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        reference = frame.copy()
        reference.loc[len(reference)] = censored
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = append_censored_row(frame, censored)
    assert result.dtypes.to_dict() == reference.dtypes.to_dict()
    assert_frame_equal(result, reference)
    assert result["binary_target"].dtype == np.dtype("O")
    assert result.iloc[-1]["binary_target"] is None
    assert pd.isna(result.iloc[-1]["resolution_ts_utc"])
    assert isinstance(date.today(), date)  # keep the datetime import honest for ruff


@pytest.mark.parametrize("value", [2, 4, 8])
def test_worker_policy_accepts_only_one_worker_typed(value):
    """§4.6 (F-20): the V1 executor is sequential; a WorkerPolicy above one
    worker is refused with the typed reason BEFORE any job exists."""

    from pydantic import ValidationError

    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        EXECUTION_MODE_V1,
        SUPPORTED_CHILD_WORKERS,
        UNSUPPORTED_WORKER_PARALLELISM_REASON,
        UnsupportedWorkerParallelismError,
        WorkerPolicy,
        assert_supported_worker_parallelism,
        worker_parallelism_refusal,
    )

    assert SUPPORTED_CHILD_WORKERS == 1
    assert EXECUTION_MODE_V1 == "sequential_children_v1"
    assert UNSUPPORTED_WORKER_PARALLELISM_REASON == "unsupported_worker_parallelism_v1"
    with pytest.raises(UnsupportedWorkerParallelismError) as typed:
        assert_supported_worker_parallelism(value)
    assert typed.value.reason == UNSUPPORTED_WORKER_PARALLELISM_REASON
    assert typed.value.requested_workers == value
    with pytest.raises(ValidationError) as refused:
        WorkerPolicy(max_workers=value, max_tasks_per_child=1, memory_budget_bytes=0)
    refusal = worker_parallelism_refusal(refused.value)
    assert refusal is not None and refusal.reason == UNSUPPORTED_WORKER_PARALLELISM_REASON
    accepted = WorkerPolicy(max_workers=1, max_tasks_per_child=1, memory_budget_bytes=0)
    assert accepted.max_workers == 1
