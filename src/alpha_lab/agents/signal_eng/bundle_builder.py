"""
SignalBundle assembler — runs all registered detectors and packages results.

Orchestrates the execution of all signal detectors against a DataBundle
and produces a complete SignalBundle for handoff to VAL-001.
"""

from __future__ import annotations

from alpha_lab.agents.signal_eng.detector_base import SignalDetector
from alpha_lab.core.contracts import DataBundle, SignalBundle, SignalVector


def build_signal_bundle(
    data: DataBundle,
    detector_ids: list[str] | None = None,
) -> SignalBundle:
    """
    Run detectors and assemble a SignalBundle.

    Args:
        data: Clean DataBundle from DATA-001
        detector_ids: Optional list of specific detectors to run.
                      If None, runs all registered detectors.

    Returns:
        SignalBundle containing all signal vectors
    """
    raise NotImplementedError


def run_single_detector(
    detector_cls: type[SignalDetector], data: DataBundle
) -> list[SignalVector]:
    """
    Instantiate and run a single detector.

    Handles input validation and error wrapping.
    """
    raise NotImplementedError
