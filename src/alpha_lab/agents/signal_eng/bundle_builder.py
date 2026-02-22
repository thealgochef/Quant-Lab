"""
SignalBundle assembler — runs all registered detectors and packages results.

Orchestrates the execution of all signal detectors against a DataBundle
and produces a complete SignalBundle for handoff to VAL-001.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from alpha_lab.agents.signal_eng.detector_base import (
    SignalDetector,
    SignalDetectorRegistry,
)
from alpha_lab.core.contracts import DataBundle, SignalBundle, SignalVector
from alpha_lab.core.exceptions import SignalComputationError

logger = logging.getLogger(__name__)


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

    Raises:
        SignalComputationError: If no signals could be generated at all
    """
    all_detectors = SignalDetectorRegistry.get_all()

    if detector_ids is not None:
        detectors_to_run = {
            did: all_detectors[did]
            for did in detector_ids
            if did in all_detectors
        }
    else:
        detectors_to_run = all_detectors

    all_signals: list[SignalVector] = []
    timeframes_seen: set[str] = set()
    errors: list[str] = []

    for detector_id, detector_cls in sorted(detectors_to_run.items()):
        try:
            signals = run_single_detector(detector_cls, data)
            all_signals.extend(signals)
            for sv in signals:
                timeframes_seen.add(sv.timeframe)
            if signals:
                logger.info(
                    "Detector %s produced %d signals", detector_id, len(signals)
                )
        except NotImplementedError:
            # Stub detectors (Tier 2/3) — skip silently
            logger.debug("Detector %s not yet implemented, skipping", detector_id)
        except Exception as exc:
            errors.append(f"{detector_id}: {exc}")
            logger.warning("Detector %s failed: %s", detector_id, exc)

    if not all_signals and not errors:
        logger.warning("No signals generated from any detector")

    if not all_signals and errors:
        msg = f"All detectors failed: {'; '.join(errors)}"
        raise SignalComputationError(msg)

    return SignalBundle(
        instrument=data.instrument,
        signals=all_signals,
        composite_scores={},
        timeframes_covered=sorted(timeframes_seen),
        total_signals=len(all_signals),
        generation_timestamp=datetime.now(UTC).isoformat(),
    )


def run_single_detector(
    detector_cls: type[SignalDetector], data: DataBundle
) -> list[SignalVector]:
    """
    Instantiate and run a single detector.

    Handles input validation and error wrapping.

    Args:
        detector_cls: The detector class to instantiate and run
        data: DataBundle to compute signals from

    Returns:
        List of SignalVector (empty if validation fails)

    Raises:
        NotImplementedError: If detector is a stub
        SignalComputationError: If computation fails unexpectedly
    """
    detector = detector_cls()

    if not detector.validate_inputs(data):
        logger.debug(
            "Detector %s: input validation failed, skipping",
            detector.detector_id,
        )
        return []

    try:
        return detector.compute(data)
    except NotImplementedError:
        raise
    except Exception as exc:
        msg = f"Detector {detector.detector_id} computation failed: {exc}"
        raise SignalComputationError(msg) from exc
