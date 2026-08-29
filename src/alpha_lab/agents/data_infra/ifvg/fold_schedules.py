"""Fold schedules — an identity of their own (R6.1 D3; owner correction 3).

A :class:`FoldScheduleEnvelope` hashes the grain-agnostic walk-forward
schedule: the fold protocol id, the authorized trading days, the ordered
train/test day windows, the step, the embargo, and the grain-agnostic purge
and boundary POLICY ids. The schedule is derived EXACTLY the way the frozen
labeled builder (``context_folds.build_context_folds``) walks the days —
expanding train ``days[:i]``, test ``days[i:i+5]``, ``i`` from 40 in steps
of 5 — so a candidate fold set and a context-bar panel fold set over the
same days derive the SAME ``fold_schedule_id`` while their row populations
(and therefore their ``fold_set_id``s) differ. Cross-grain studies compare
schedule identities and per-fold windows, never fold-set identities.

Two grain-specific fold BUILDERS live here:

* :func:`build_context_bar_panel_folds` — rows keyed by ``row_id`` (one row
  per completed context bar); purge = training bars whose ``[open, close]``
  span overlaps the test window (a malformed-timestamp guard — normally
  empty, since training days precede test days); embargo = the last two
  training days; duplicate-OOS guard; validity ``insufficient_train_rows`` /
  ``no_test_rows``; ``training_prevalence=None`` (no labels exist on a panel).
* :func:`build_candidate_folds_from_schedule` — the LABEL-FREE candidate
  builder for descriptive regime studies (no ``binary_target``, no
  class-coverage check): the setup-boundary exclusion of the labeled builder,
  purge by ``[entry, resolution]`` overlap when a resolution timestamp is
  present (else by the entry instant), the same embargo.

``build_context_folds`` stays byte-identical and remains the labeled builder
for every supervised path; a pinned test proves its windows equal the
schedule's day for day.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import pandas as pd
from pydantic import Field, model_validator

from .context_experiment_contracts import IfvgContextFoldDefinition
from .context_folds import ContextFoldSet
from .search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    register_identity_pair,
)

__all__ = [
    "FOLD_PROTOCOL_ID_V1",
    "PURGE_POLICY_ID_V1",
    "BOUNDARY_POLICY_ID_V1",
    "FoldWindow",
    "FoldSchedulePayload",
    "FoldScheduleEnvelope",
    "schedule_windows",
    "derive_fold_schedule",
    "build_context_bar_panel_folds",
    "build_candidate_folds_from_schedule",
]

#: The frozen walk-forward protocol (matches the study-cell / pipeline Literal).
FOLD_PROTOCOL_ID_V1 = "ifvg_context_walkforward_40_5_5_2_v1"
#: Grain-agnostic purge policy: a training observation whose own interval
#: overlaps the test interval is purged (the interval is grain-specific —
#: label interval for candidates, bar span for panels — and a fold-set fact).
PURGE_POLICY_ID_V1 = "observation_interval_overlaps_test_interval_v1"
#: Windows never cross the 18:00 ET trading-day boundary.
BOUNDARY_POLICY_ID_V1 = "trading_day_18et_boundary_v1"

_PROTOCOL_PARAMETERS: dict[str, dict[str, int]] = {
    FOLD_PROTOCOL_ID_V1: {"train_days": 40, "test_days": 5, "step_days": 5, "embargo_days": 2}
}


class FoldWindow(FrozenContract):
    fold_index: int = Field(ge=0)
    train_days: tuple[str, ...] = Field(min_length=1)
    test_days: tuple[str, ...] = Field(min_length=1)


class FoldSchedulePayload(FrozenContract):
    """The grain-agnostic schedule: protocol, days, windows, policies."""

    fold_protocol_id: str
    authorized_trading_days: tuple[str, ...] = Field(min_length=1)
    windows: tuple[FoldWindow, ...]
    train_days: int = Field(ge=1)
    test_days: int = Field(ge=1)
    step_days: int = Field(ge=1)
    embargo_days: int = Field(ge=0)
    purge_policy_id: Literal["observation_interval_overlaps_test_interval_v1"] = (
        PURGE_POLICY_ID_V1
    )
    boundary_policy_id: Literal["trading_day_18et_boundary_v1"] = BOUNDARY_POLICY_ID_V1

    @model_validator(mode="after")
    def _windows_derive_from_the_days(self):
        days = self.authorized_trading_days
        if days != tuple(sorted(days)) or len(days) != len(set(days)):
            raise ValueError("authorized trading days must be unique and chronological")
        parameters = _PROTOCOL_PARAMETERS.get(self.fold_protocol_id)
        if parameters is None:
            raise ValueError(f"unregistered fold protocol {self.fold_protocol_id!r}")
        for name, value in parameters.items():
            if getattr(self, name) != value:
                raise ValueError(
                    f"{self.fold_protocol_id} fixes {name}={value}; got {getattr(self, name)}"
                )
        expected = schedule_windows(
            days,
            train_days=self.train_days,
            test_days=self.test_days,
            step_days=self.step_days,
        )
        if tuple(self.windows) != expected:
            raise ValueError(
                "schedule windows do not derive from the authorized days under "
                "the protocol (expanding train, stepped test)"
            )
        return self


class FoldScheduleEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "fold_schedule_id"

    fold_schedule_id: str = Field(pattern=SHA256_PATTERN)
    payload: FoldSchedulePayload


def schedule_windows(
    days: tuple[str, ...], *, train_days: int, test_days: int, step_days: int
) -> tuple[FoldWindow, ...]:
    """Expanding-train / stepped-test windows — identical to the walk in
    ``context_folds.build_context_folds`` (``days[:i]`` / ``days[i:i+test]``)."""

    windows: list[FoldWindow] = []
    fold_index = 0
    test_start = train_days
    while test_start + test_days <= len(days):
        windows.append(
            FoldWindow(
                fold_index=fold_index,
                train_days=tuple(days[:test_start]),
                test_days=tuple(days[test_start : test_start + test_days]),
            )
        )
        fold_index += 1
        test_start += step_days
    return tuple(windows)


def derive_fold_schedule(
    authorized_trading_days: tuple[str, ...], *, protocol: str = FOLD_PROTOCOL_ID_V1
) -> FoldScheduleEnvelope:
    """The pure schedule derivation for one authorized day set."""

    parameters = _PROTOCOL_PARAMETERS.get(protocol)
    if parameters is None:
        raise ValueError(f"unregistered fold protocol {protocol!r}")
    days = tuple(str(day) for day in authorized_trading_days)
    payload = FoldSchedulePayload(
        fold_protocol_id=protocol,
        authorized_trading_days=days,
        windows=schedule_windows(
            days,
            train_days=parameters["train_days"],
            test_days=parameters["test_days"],
            step_days=parameters["step_days"],
        ),
        **parameters,
    )
    return FoldScheduleEnvelope.from_payload(payload)


def _sorted_ids(frame: pd.DataFrame, key: str) -> tuple[str, ...]:
    return tuple(sorted(frame[key].astype(str)))


def _assignment_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=(
            "fold_index",
            "candidate_id",
            "partition",
            "fold_valid",
            "fold_invalid_reason",
        ),
    )


def build_context_bar_panel_folds(
    panel_frame: pd.DataFrame,
    *,
    authorized_trading_days: tuple[str, ...],
    minimum_train_rows: int | None = None,
    embargo_days: int = 2,
) -> ContextFoldSet:
    """Panel-native folds keyed by ``row_id`` under the frozen schedule.

    ``minimum_train_rows`` defaults to the stamped panel sample-adequacy floor
    (``sample_adequacy_minimum(CONTEXT_BAR_PANEL)`` = 300). The training-row
    count that decides validity is the number of training bars whose
    features are VALID (``cbp_valid``) when the column is present — the same
    population the regime fit will actually train on (all-missing rows never
    enter a fit) — so a fold's validity agrees with the assessment's
    sample-adequacy gate.
    """

    required = {"row_id", "trading_day", "bar_open_ts_utc", "bar_close_ts_utc"}
    missing = sorted(required - set(panel_frame.columns))
    if missing:
        raise ValueError(f"panel fold input is missing columns {missing}")
    if minimum_train_rows is None:
        from .ml.regime_contracts import (  # noqa: PLC0415
            ObservationGranularity,
            sample_adequacy_minimum,
        )

        minimum_train_rows = sample_adequacy_minimum(ObservationGranularity.CONTEXT_BAR_PANEL)
    schedule = derive_fold_schedule(authorized_trading_days)
    days = schedule.payload.authorized_trading_days
    frame = panel_frame.copy()
    if frame["row_id"].isna().any() or frame["row_id"].astype(str).duplicated().any():
        raise ValueError("panel fold input requires unique non-null row ids")
    if not set(frame["trading_day"].astype(str)).issubset(days):
        raise ValueError("panel fold input contains an unauthorized trading day")
    frame["_open"] = pd.to_datetime(frame["bar_open_ts_utc"], utc=True, errors="raise")
    frame["_close"] = pd.to_datetime(frame["bar_close_ts_utc"], utc=True, errors="raise")
    if (frame["_close"] < frame["_open"]).any():
        raise ValueError("a panel bar closes before it opens")
    valid_column = frame["cbp_valid"].astype(bool) if "cbp_valid" in frame else None

    folds: list[IfvgContextFoldDefinition] = []
    assignment_rows: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    for window in schedule.payload.windows:
        train = frame[frame["trading_day"].astype(str).isin(window.train_days)].copy()
        test = frame[frame["trading_day"].astype(str).isin(window.test_days)].copy()
        purged_ids: tuple[str, ...] = ()
        if not test.empty:
            test_start = test["_open"].min()
            test_end = test["_close"].max()
            overlap = (train["_open"] <= test_end) & (train["_close"] >= test_start)
            purged_ids = _sorted_ids(train.loc[overlap], "row_id")
            train = train.loc[~overlap].copy()
        embargo_values = window.train_days[-embargo_days:] if embargo_days else ()
        embargo = train["trading_day"].astype(str).isin(embargo_values)
        embargoed_ids = _sorted_ids(train.loc[embargo], "row_id")
        train = train.loc[~embargo].copy()
        duplicate = set(test["row_id"].astype(str)) & seen_test_ids
        if duplicate:
            raise ValueError(f"panel row appears in multiple OOS folds: {sorted(duplicate)[:3]}")
        seen_test_ids.update(test["row_id"].astype(str))
        train_rows_with_inputs = (
            int(valid_column.loc[train.index].sum()) if valid_column is not None else len(train)
        )
        reason: str | None = None
        if train_rows_with_inputs < minimum_train_rows:
            reason = "insufficient_train_rows"
        elif test.empty:
            reason = "no_test_rows"
        definition = IfvgContextFoldDefinition(
            fold_index=window.fold_index,
            train_days=window.train_days,
            test_days=window.test_days,
            train_candidate_ids=_sorted_ids(train, "row_id"),
            test_candidate_ids=_sorted_ids(test, "row_id"),
            excluded_boundary_setup_ids=(),
            purged_candidate_ids=purged_ids,
            embargoed_candidate_ids=embargoed_ids,
            valid=reason is None,
            invalid_reason=reason,
            training_prevalence=None,
        )
        folds.append(definition)
        for partition, subset in (("train", train), ("test", test)):
            assignment_rows.extend(
                {
                    "fold_index": window.fold_index,
                    "candidate_id": row_id,
                    "partition": partition,
                    "fold_valid": definition.valid,
                    "fold_invalid_reason": definition.invalid_reason,
                }
                for row_id in _sorted_ids(subset, "row_id")
            )
    valid_folds = sum(fold.valid for fold in folds)
    return ContextFoldSet(
        folds=tuple(folds),
        assignment=_assignment_frame(assignment_rows),
        status="ready" if valid_folds else "insufficient_train_rows",
    )


def build_candidate_folds_from_schedule(
    candidate_frame: pd.DataFrame,
    *,
    authorized_trading_days: tuple[str, ...],
    minimum_train_rows: int = 150,
    embargo_days: int = 2,
) -> ContextFoldSet:
    """The LABEL-FREE candidate fold builder (descriptive regime studies).

    Requires ``candidate_id``, ``setup_id``, ``trading_day``, ``entry_ts_utc``;
    an optional ``resolution_ts_utc`` widens each candidate's purge interval
    from the entry instant to ``[entry, resolution]``. No ``binary_target`` is
    consulted and no class-coverage check exists. The setup-boundary rule is
    the labeled builder's: a setup with candidates on both sides of the
    train/test boundary is excluded from both partitions.
    """

    required = {"candidate_id", "setup_id", "trading_day", "entry_ts_utc"}
    missing = sorted(required - set(candidate_frame.columns))
    if missing:
        raise ValueError(f"candidate fold input is missing columns {missing}")
    schedule = derive_fold_schedule(authorized_trading_days)
    days = schedule.payload.authorized_trading_days
    frame = candidate_frame.copy()
    if frame["candidate_id"].isna().any() or frame["candidate_id"].duplicated().any():
        raise ValueError("fold input requires unique non-null candidate IDs")
    if not set(frame["trading_day"].astype(str)).issubset(days):
        raise ValueError("fold input contains an unauthorized trading day")
    frame["_entry"] = pd.to_datetime(frame["entry_ts_utc"], utc=True, errors="coerce")
    if frame["_entry"].isna().any():
        raise ValueError("every candidate requires a parseable entry timestamp")
    if "resolution_ts_utc" in frame.columns:
        resolution = pd.to_datetime(frame["resolution_ts_utc"], utc=True, errors="coerce")
        frame["_resolution"] = resolution.where(resolution.notna(), frame["_entry"])
    else:
        frame["_resolution"] = frame["_entry"]
    if (frame["_resolution"] < frame["_entry"]).any():
        raise ValueError("candidate interval resolves before entry")

    folds: list[IfvgContextFoldDefinition] = []
    assignment_rows: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    for window in schedule.payload.windows:
        raw_train = frame[frame["trading_day"].astype(str).isin(window.train_days)]
        raw_test = frame[frame["trading_day"].astype(str).isin(window.test_days)]
        boundary = set(raw_train["setup_id"].astype(str)) & set(raw_test["setup_id"].astype(str))
        train = raw_train[~raw_train["setup_id"].astype(str).isin(boundary)].copy()
        test = raw_test[~raw_test["setup_id"].astype(str).isin(boundary)].copy()
        purged_ids: tuple[str, ...] = ()
        if not test.empty:
            test_start = test["_entry"].min()
            test_end = test["_resolution"].max()
            overlap = (train["_entry"] <= test_end) & (train["_resolution"] >= test_start)
            purged_ids = _sorted_ids(train.loc[overlap], "candidate_id")
            train = train.loc[~overlap].copy()
        embargo_values = window.train_days[-embargo_days:] if embargo_days else ()
        embargo = train["trading_day"].astype(str).isin(embargo_values)
        embargoed_ids = _sorted_ids(train.loc[embargo], "candidate_id")
        train = train.loc[~embargo].copy()
        duplicate = set(test["candidate_id"].astype(str)) & seen_test_ids
        if duplicate:
            raise ValueError(
                f"candidate appears in multiple OOS folds: {sorted(duplicate)[:3]}"
            )
        seen_test_ids.update(test["candidate_id"].astype(str))
        reason: str | None = None
        if len(train) < minimum_train_rows:
            reason = "insufficient_train_candidates"
        elif test.empty:
            reason = "no_test_candidates"
        definition = IfvgContextFoldDefinition(
            fold_index=window.fold_index,
            train_days=window.train_days,
            test_days=window.test_days,
            train_candidate_ids=_sorted_ids(train, "candidate_id"),
            test_candidate_ids=_sorted_ids(test, "candidate_id"),
            excluded_boundary_setup_ids=tuple(sorted(boundary)),
            purged_candidate_ids=purged_ids,
            embargoed_candidate_ids=embargoed_ids,
            valid=reason is None,
            invalid_reason=reason,
            training_prevalence=None,
        )
        folds.append(definition)
        for partition, subset in (("train", train), ("test", test)):
            assignment_rows.extend(
                {
                    "fold_index": window.fold_index,
                    "candidate_id": candidate_id,
                    "partition": partition,
                    "fold_valid": definition.valid,
                    "fold_invalid_reason": definition.invalid_reason,
                }
                for candidate_id in _sorted_ids(subset, "candidate_id")
            )
    valid_folds = sum(fold.valid for fold in folds)
    return ContextFoldSet(
        folds=tuple(folds),
        assignment=_assignment_frame(assignment_rows),
        status="ready" if valid_folds else "insufficient_train_candidates",
    )


def _example_schedule() -> FoldSchedulePayload:
    days = tuple(day.strftime("%Y-%m-%d") for day in pd.bdate_range("2026-01-05", periods=45))
    return derive_fold_schedule(days).payload


register_identity_pair(
    name="FoldSchedule",
    envelope_cls=FoldScheduleEnvelope,
    payload_cls=FoldSchedulePayload,
    id_field="fold_schedule_id",
    example_factory=_example_schedule,
)
