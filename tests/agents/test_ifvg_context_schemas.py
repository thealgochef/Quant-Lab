"""Formula-v2 context tables have stable, exact Arrow schemas."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_lab.agents.data_infra.ifvg.context_contracts import ContextRecordTable
from alpha_lab.agents.data_infra.ifvg.context_schemas import (
    IFVG_CONTEXT_ARROW_REGISTRY,
    IFVG_CONTEXT_ARROW_REGISTRY_HASH,
    context_arrow_schema,
    context_arrow_schema_hash,
    context_table_from_frame,
    context_table_from_rows,
    validate_context_arrow_table,
)


def _value(kind: pa.DataType):
    if pa.types.is_string(kind):
        return "x"
    if pa.types.is_int64(kind):
        return 1
    if pa.types.is_float64(kind):
        return 1.25
    if pa.types.is_boolean(kind):
        return True
    if pa.types.is_timestamp(kind):
        return datetime(2026, 1, 13, 14, 30, tzinfo=UTC)
    if pa.types.is_list(kind):
        return ["a", "b"]
    raise AssertionError(f"unhandled test Arrow type {kind}")


@pytest.mark.parametrize("table", list(ContextRecordTable))
def test_empty_all_null_and_populated_tables_share_exact_schema(table, tmp_path) -> None:
    schema = context_arrow_schema(table)
    empty = context_table_from_frame(table, pd.DataFrame(columns=schema.names))
    all_null = context_table_from_rows(
        table,
        [{field.name: None for field in schema}],
    )
    populated = context_table_from_rows(
        table,
        [{field.name: _value(field.type) for field in schema}],
    )
    for label, arrow in (("empty", empty), ("null", all_null), ("full", populated)):
        validate_context_arrow_table(table, arrow)
        path = tmp_path / f"{table.value}-{label}.parquet"
        pq.write_table(arrow, path)
        assert pq.read_schema(path).equals(schema, check_metadata=True)


@pytest.mark.parametrize("table", list(ContextRecordTable))
def test_context_schema_rejects_missing_and_extra_columns(table) -> None:
    schema = context_arrow_schema(table)
    complete = {field.name: None for field in schema}
    missing = dict(complete)
    missing.pop(schema.names[0])
    with pytest.raises(ValueError, match="missing columns"):
        context_table_from_rows(table, [missing])
    with pytest.raises(ValueError, match="undeclared columns"):
        context_table_from_rows(table, [{**complete, "surprise": 1}])


def test_registry_has_exactly_twelve_hashed_tables() -> None:
    assert tuple(IFVG_CONTEXT_ARROW_REGISTRY) == tuple(
        table.value for table in ContextRecordTable
    )
    assert len(IFVG_CONTEXT_ARROW_REGISTRY_HASH) == 64
    assert len({context_arrow_schema_hash(table) for table in ContextRecordTable}) == 12

