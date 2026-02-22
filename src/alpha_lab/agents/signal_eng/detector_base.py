"""
Abstract base class for signal detectors with auto-registration.

Any concrete subclass that defines `detector_id` automatically registers
itself in the SignalDetectorRegistry via __init_subclass__.

Usage:
    class EmaConfluence(SignalDetector):
        detector_id = "ema_confluence"
        category = "ema_confluence"
        tier = SignalTier.CORE
        timeframes = [Timeframe.M5, Timeframe.M15, Timeframe.H1]

        def compute(self, data: DataBundle) -> list[SignalVector]:
            ...

    # EmaConfluence is now in SignalDetectorRegistry automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from alpha_lab.core.contracts import DataBundle, SignalVector
from alpha_lab.core.enums import SignalTier


class SignalDetectorRegistry:
    """Global registry of all signal detector classes, populated by __init_subclass__."""

    _detectors: ClassVar[dict[str, type[SignalDetector]]] = {}

    @classmethod
    def get(cls, detector_id: str) -> type[SignalDetector]:
        """Get a detector class by ID."""
        return cls._detectors[detector_id]

    @classmethod
    def get_all(cls) -> dict[str, type[SignalDetector]]:
        """Get all registered detector classes."""
        return dict(cls._detectors)

    @classmethod
    def get_by_tier(cls, tier: SignalTier) -> dict[str, type[SignalDetector]]:
        """Get all detector classes for a specific tier."""
        return {k: v for k, v in cls._detectors.items() if v.tier == tier}

    @classmethod
    def list_ids(cls) -> list[str]:
        """List all registered detector IDs."""
        return sorted(cls._detectors.keys())

    @classmethod
    def count(cls) -> int:
        """Return number of registered detectors."""
        return len(cls._detectors)


class SignalDetector(ABC):
    """
    Abstract base for all signal detectors.

    Subclasses MUST define as class variables:
        detector_id: str       — unique identifier (e.g., 'ema_confluence')
        category: str          — signal category name
        tier: SignalTier       — which tier (CORE, ICT_STRUCTURAL, COMPOSITE)
        timeframes: list[str]  — which timeframes this detector operates on

    Subclasses MUST implement:
        compute(data: DataBundle) -> list[SignalVector]
    """

    detector_id: ClassVar[str]
    category: ClassVar[str]
    tier: ClassVar[SignalTier]
    timeframes: ClassVar[list[str]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "detector_id" in cls.__dict__:
            SignalDetectorRegistry._detectors[cls.detector_id] = cls

    @abstractmethod
    def compute(self, data: DataBundle) -> list[SignalVector]:
        """
        Compute signal vectors from a DataBundle.

        Must produce signals that are:
        - Point-in-time (NO look-ahead bias)
        - Direction normalized to [-1, 0, +1]
        - Strength normalized to [0, 1]
        - Tagged with formation bar index and timeframe

        Returns empty list if no signal generated.
        """

    def validate_inputs(self, data: DataBundle) -> bool:
        """
        Check if this detector has the data it needs.

        Override for detectors with specific data requirements
        (e.g., tick data needed, multiple timeframes needed).
        """
        return True
