"""Calibration-policy registry (`ML_REGIME_CONTRACT_PLAN.md` §6; brief §7B.16).

Raw probability diagnostics remain mandatory regardless of calibrator; in
V1 they are also the ONLY executable policy. Platt/isotonic are registered
planned entries whose future fit services must accept train-fold rows only
(acceptance 7B.22-14) — requesting them now refuses fail-closed with the
registered reason.
"""

from __future__ import annotations

from enum import StrEnum
from types import MappingProxyType

from ..context_experiment_contracts import IFVG_CONTEXT_CALIBRATION_POLICY_ID
from ..search.identities import FrozenContract
from ..study.study_cell import REGISTERED_CALIBRATION_POLICY_IDS

__all__ = [
    "CalibrationPolicyStatus",
    "CalibrationPolicyEntry",
    "CALIBRATION_POLICY_REGISTRY",
    "CalibrationPolicyUnavailableError",
    "resolve_calibration_policy_entry",
    "assert_calibration_policy_executable",
]


class CalibrationPolicyStatus(StrEnum):
    AVAILABLE = "available"
    PLANNED = "planned"


class CalibrationPolicyEntry(FrozenContract):
    calibration_policy_id: str
    status: CalibrationPolicyStatus
    reason: str | None = None
    #: binding constraint on the FUTURE implementation, recorded now
    fit_scope: str = "train_fold_rows_only"


CALIBRATION_POLICY_REGISTRY: MappingProxyType[str, CalibrationPolicyEntry] = (
    MappingProxyType(
        {
            IFVG_CONTEXT_CALIBRATION_POLICY_ID: CalibrationPolicyEntry(
                calibration_policy_id=IFVG_CONTEXT_CALIBRATION_POLICY_ID,
                status=CalibrationPolicyStatus.AVAILABLE,
            ),
            "platt_sigmoid_train_fold_v1": CalibrationPolicyEntry(
                calibration_policy_id="platt_sigmoid_train_fold_v1",
                status=CalibrationPolicyStatus.PLANNED,
                reason=(
                    "fit inside each training fold on a recorded calibration "
                    "subset; never fit on pooled OOS and score the same rows — "
                    "implementation is post-R5"
                ),
            ),
            "isotonic_train_fold_v1": CalibrationPolicyEntry(
                calibration_policy_id="isotonic_train_fold_v1",
                status=CalibrationPolicyStatus.PLANNED,
                reason=(
                    "requires calibration-subset support n>=200 per fold — "
                    "implementation is post-R5"
                ),
            ),
        }
    )
)

if tuple(CALIBRATION_POLICY_REGISTRY) != REGISTERED_CALIBRATION_POLICY_IDS:
    raise AssertionError(
        "the calibration-policy registry must cover exactly the study-cell "
        "registered calibration policy ids, in order"
    )


class CalibrationPolicyUnavailableError(PermissionError):
    """A registered-but-planned calibration policy was requested."""


def resolve_calibration_policy_entry(calibration_policy_id: str) -> CalibrationPolicyEntry:
    entry = CALIBRATION_POLICY_REGISTRY.get(calibration_policy_id)
    if entry is None:
        raise ValueError(
            f"unknown calibration policy id {calibration_policy_id!r}; registered: "
            f"{sorted(CALIBRATION_POLICY_REGISTRY)}"
        )
    return entry


def assert_calibration_policy_executable(
    calibration_policy_id: str,
) -> CalibrationPolicyEntry:
    entry = resolve_calibration_policy_entry(calibration_policy_id)
    if entry.status is not CalibrationPolicyStatus.AVAILABLE:
        raise CalibrationPolicyUnavailableError(
            f"calibration policy {calibration_policy_id!r} is "
            f"{entry.status.value}: {entry.reason}"
        )
    return entry
