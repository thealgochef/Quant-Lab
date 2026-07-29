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
