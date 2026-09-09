"""Presentation context for retained Developer diagnostics acceptance tests."""

from contextvars import ContextVar

import pytest


@pytest.fixture
def developer_presentation():
    """UI-1–UI-3 technical surfaces now run only in the Developer area.

    AppTest executes in its own thread, so the test's ContextVar default
    models the startup-enabled Developer route across that thread boundary.
    Normal-workspace tests deliberately do not request this fixture.
    """
    from alpha_lab.agents.data_infra.ifvg.presentation import workspace_mode

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(workspace_mode, "DEVELOPER_MODE", True)
        patch.setattr(
            workspace_mode, "_DEVELOPER_AREA", ContextVar("test_developer_area", default=True)
        )
        yield
