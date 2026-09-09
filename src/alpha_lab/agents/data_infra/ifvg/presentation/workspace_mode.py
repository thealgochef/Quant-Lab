"""Startup-only presentation boundary. This never supplies execution authority."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Imported once by the server, not evaluated from session state or query parameters.
DEVELOPER_MODE = os.environ.get("QUANT_LAB_DEVELOPER_MODE") == "1"
_DEVELOPER_AREA: ContextVar[bool] = ContextVar("ifvg_developer_area", default=False)


def technical_details_enabled() -> bool:
    return DEVELOPER_MODE and _DEVELOPER_AREA.get()


@contextmanager
def developer_area() -> Iterator[None]:
    if not DEVELOPER_MODE:
        raise PermissionError("Developer presentation was not enabled at startup")
    token = _DEVELOPER_AREA.set(True)
    try:
        yield
    finally:
        _DEVELOPER_AREA.reset(token)
