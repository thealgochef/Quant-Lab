"""IFVG dataset assembly guards (the IFVG-FIX window regression net).

F1 — a fully-all-NaN column in the emitted entry dataset means a pivot rename
or join key collided (the ``parent_fvg_id`` lesson); the builder must raise,
never emit. F2 — the per-day capture union must be dtype-explicit so pandas-3
concat (which stops excluding all-NA frames from result-dtype determination)
cannot change the schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.agents.data_infra.ifvg.dataset import (
    CAPTURE_UNION_SCHEMA,
    CaptureChainResult,
    conform_capture_frame,
)
from alpha_lab.agents.data_infra.ifvg.entry_dataset import assert_no_all_nan_columns


def test_all_nan_guard_raises_with_column_names() -> None:
    frame = pd.DataFrame(
        {
            "good_int": [1, 2],
            "good_partial": [np.nan, 3.0],
            "dead_float": [np.nan, np.nan],
            "dead_object": [None, None],
        }
    )
    with pytest.raises(ValueError, match=r"dead_float.*dead_object"):
        assert_no_all_nan_columns(frame)


def test_all_nan_guard_passes_clean_and_empty_frames() -> None:
    assert_no_all_nan_columns(pd.DataFrame({"a": [1.0], "b": ["x"]}))
    assert_no_all_nan_columns(pd.DataFrame())


def _day_frame(rows: list[dict]) -> pd.DataFrame:
    """A synthetic per-day capture frame the way ``flatten_emissions`` builds
    one: DataFrame-from-row-dicts, columns = union of the keys present."""
    stamped = [
        {"kind": "htf_tap", "envelope_schema_version": 1, "is_warmup": False,
         "days_of_htf_history": 3, **row}
        for row in rows
    ]
    return pd.DataFrame(stamped)


def test_capture_union_dtypes_do_not_depend_on_pandas_concat_inference() -> None:
    # The pandas-2 vs pandas-3 divergence: a float-schema column that is all-NA
    # (hence OBJECT-inferred from None values) in one day frame but populated
    # float in another. pandas 2 excluded the all-NA frame from result-dtype
    # determination; pandas 3 does not, and plain concat would come out object.
    day_a = _day_frame(
        [{"nearest_level_distance_ticks": None}, {"nearest_level_distance_ticks": None}]
    )
    day_b = _day_frame(
        [{"nearest_level_distance_ticks": 12.0}, {"nearest_level_distance_ticks": 3.0}]
    )
    assert day_a["nearest_level_distance_ticks"].dtype == object  # the hazard, expressed

    chain = CaptureChainResult()
    chain.frames.extend([day_a, day_b])
    out = chain.frame()  # the real concat code path

    assert len(out) == 4
    assert list(out.columns) == list(CAPTURE_UNION_SCHEMA)
    got = {c: str(t) for c, t in out.dtypes.items()}
    assert got == CAPTURE_UNION_SCHEMA
    assert out["nearest_level_distance_ticks"].dtype == "float64"


def test_conform_rejects_columns_outside_the_declared_schema() -> None:
    frame = _day_frame([{"a_brand_new_field": 1.0}])
    with pytest.raises(ValueError, match="a_brand_new_field"):
        conform_capture_frame(frame)
