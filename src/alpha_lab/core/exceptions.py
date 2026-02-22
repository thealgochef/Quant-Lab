"""Custom exception hierarchy for the Alpha Signal Research Lab."""

from __future__ import annotations


class AlphaLabError(Exception):
    """Base exception for all Alpha Lab errors."""


class ConfigurationError(AlphaLabError):
    """Invalid or missing configuration."""


class DataProviderError(AlphaLabError):
    """Data provider connection or fetch failure."""


class DataQualityError(AlphaLabError):
    """Data fails quality validation checks."""


class SignalComputationError(AlphaLabError):
    """Signal detector computation failure."""


class ValidationFirewallError(AlphaLabError):
    """Violation of the SIG-001 <-> VAL-001 firewall rules."""


class AgentCommunicationError(AlphaLabError):
    """Message bus routing or delivery failure."""


class ExecutionConstraintError(AlphaLabError):
    """Prop firm or risk constraint violation."""


class SchemaValidationError(AlphaLabError):
    """Message envelope or payload schema mismatch."""
