"""Deterministic Arrow IPC table bytes (R6.1 — shared by the context-bar
panel and the regime-assignment artifacts).

The MBP-1 lane keeps its own private helpers; this module is the
lane-neutral equivalent: pandas metadata is STRIPPED from the schema so the
persisted bytes depend on the values, the declared schema, and nothing about
the producing pandas version. Every sidecar written through here is hashed
by its exact bytes and re-verified against the envelope on reload.
"""

from __future__ import annotations

import hashlib
from io import BytesIO

import pandas as pd
import pyarrow as pa
import pyarrow.ipc

__all__ = [
    "frame_to_arrow_bytes",
    "frame_from_arrow_bytes",
    "arrow_schema_hash",
    "bytes_sha256",
]


def frame_to_arrow_bytes(frame: pd.DataFrame, schema: pa.Schema | None = None) -> bytes:
    """Arrow IPC file bytes of ``frame`` under ``schema`` (pandas metadata
    removed; column order is the schema's when one is given)."""

    if schema is not None:
        table = pa.Table.from_pandas(
            frame.loc[:, list(schema.names)], schema=schema, preserve_index=False
        )
    else:
        table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(None)
    sink = BytesIO()
    with pyarrow.ipc.new_file(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue()


def frame_from_arrow_bytes(data: bytes) -> pd.DataFrame:
    with pyarrow.ipc.open_file(BytesIO(data)) as reader:
        return reader.read_all().to_pandas()


def arrow_schema_hash(schema: pa.Schema) -> str:
    """Canonical hash of a schema's (name, type) pairs."""

    canonical = "\n".join(f"{field.name}:{field.type}" for field in schema)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
