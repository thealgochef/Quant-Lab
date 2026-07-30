"""IFVG experiment engine (plan Part A): headless core behind the gate CLI/UI.

``run_ifvg_experiment(config)`` refactors ``scripts/run_ifvg_gate_experiment.py``
into an importable engine returning JSON-safe sections: applied config, feature
matrix, expectancy blocks, per-split OOS + pooled + deduplicated OOS, coverage
sweep, calibration + Brier, the full 2b trade-statistics contract
(:mod:`.trade_stats`) and the 2c feature-insight panel
(:mod:`.feature_insight`).

SEALED GUARD (non-configurable): the normal engine path hard-clamps
``trading_day >= SEALED_HOLDOUT_START`` out of every computation — only the
excluded COUNT is reported. :func:`run_sealed_validation` is the single code
path allowed to touch sealed rows; it writes files only (ledgered, sequenced)
and never prints sealed statistics.

Parity contract: under the default config with the model block on, the engine
reproduces the gate script's numbers exactly (same feature exclusion, fillna,
CatBoost params/seed, day splits with 2-day purge, qcut calibration, Brier).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from strategy_core.strategies.ifvg_smc.labels import resolve_ifvg_outcome
from strategy_core.types import Bar, Direction

from .config import SEALED_HOLDOUT_START, IfvgCaptureConfig
from .day_artifacts import load_day_artifacts
from .entry_dataset import NQ_COST_POINTS_ROUND_TURN
from .feature_insight import MULTIPLE_COMPARISONS_CAVEAT, compute_feature_insight
from .funnel_report import DOC_DEFAULT_CAPS, DOC_DEFAULT_FLOORS, doc_default_pass
from .trade_stats import compute_trade_stats, mean_ci, wilson_ci

__all__ = [
    "IfvgFilterConfig",
    "SessionWindow",
    "IfvgDocDefaultsConfig",
    "IfvgScoringConfig",
    "IfvgSlTpConfig",
    "IfvgModelConfig",
    "IfvgExperimentConfig",
    "run_ifvg_experiment",
    "dedup_pooled_oos",
    "stamp_custom_sessions",
    "recover_override_stop_ticks",
    "json_safe",
    "EXPERIMENTS_DIR",
    "save_experiment",
    "list_experiments",
    "load_experiment",
    "delete_experiment",
    "run_sealed_validation",
    "sealed_ledger_count",
    "NO_PATH_DEPENDENCE_CAVEAT",
]

EXPERIMENTS_DIR = Path("data/ifvg_experiments")

R_FAMILY_MULTIPLE = {"r10": 1.0, "r15": 1.5, "r20": 2.0}

NO_PATH_DEPENDENCE_CAVEAT = (
    "SL/TP override is a label-layer recompute over the same logged candidate "
    "stream: no path-dependence re-simulation (slot occupancy and subsequent-"
    "setup effects unchanged); risk is re-derived from the overridden stop."
)

_HHMM_RE = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_FAMILIES = ("fresh_fvg_continuation", "ifvg_retest")


# ── configuration ─────────────────────────────────────────────────────────────


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class IfvgFilterConfig(_Frozen):
    """Row filters applied over the entry dataset (pre-seal rows only)."""

    sessions_engine: tuple[str, ...] | None = None
    sessions_doc: tuple[str, ...] | None = None
    direction: Literal["long", "short", "both"] = "both"
    families: tuple[str, ...] = _FAMILIES
    include_warmup: bool = False
    day_start: str | None = None
    day_end: str | None = None
    #: Deterministic first-N by entry time over the logged candidates.
    max_trades_per_day: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_days(self) -> IfvgFilterConfig:
        for value in (self.day_start, self.day_end):
            if value is not None and not _DAY_RE.match(value):
                raise ValueError(f"day bounds must be YYYY-MM-DD, got {value!r}")
        return self


class SessionWindow(_Frozen):
    """One custom ET session window; ``start > end`` crosses midnight."""

    start: str
    end: str

    @model_validator(mode="after")
    def _check_hhmm(self) -> SessionWindow:
        for value in (self.start, self.end):
            if not _HHMM_RE.match(value):
                raise ValueError(f"session times must be HH:MM, got {value!r}")
        return self


class IfvgDocDefaultsConfig(_Frozen):
    """doc_default_pass knobs; defaults mirror the funnel_report hardcoded values."""

    apply: bool = False
    tap_fvg_size_min: float = DOC_DEFAULT_FLOORS["tap_fvg_size_ticks"]
    parent_fvg_size_min: float = DOC_DEFAULT_FLOORS["parent_fvg_size_ticks"]
    opp_fvg_size_min: float = DOC_DEFAULT_FLOORS["opp_fvg_size_ticks"]
    parent_distance_max: float = DOC_DEFAULT_CAPS["parent_distance_to_htf_ticks"]
    opp_distance_max: float = DOC_DEFAULT_CAPS["opp_distance_to_parent_ticks"]
    bars_since_inversion_max: float = DOC_DEFAULT_CAPS["bars_since_inversion"]

    def floors(self) -> dict[str, float]:
        return {
            "tap_fvg_size_ticks": self.tap_fvg_size_min,
            "parent_fvg_size_ticks": self.parent_fvg_size_min,
            "opp_fvg_size_ticks": self.opp_fvg_size_min,
        }

    def caps(self) -> dict[str, float]:
        return {
            "parent_distance_to_htf_ticks": self.parent_distance_max,
            "opp_distance_to_parent_ticks": self.opp_distance_max,
            "bars_since_inversion": self.bars_since_inversion_max,
        }


class IfvgScoringConfig(_Frozen):
    r_family: Literal["r10", "r15", "r20"] = "r10"
    cost_points: float = Field(default=NQ_COST_POINTS_ROUND_TURN, ge=0)


class IfvgSlTpConfig(_Frozen):
    """SL/TP overrides -> label-kernel recompute (entry stays the logged close)."""

    tp_mode: Literal["r_multiple", "fixed_points"] | None = None
    tp_value: float | None = None
    sl_mode: Literal["logged", "fixed_points", "swing_buffer_ticks"] = "logged"
    sl_value: float | None = None

    @model_validator(mode="after")
    def _check(self) -> IfvgSlTpConfig:
        if self.tp_mode is not None and (self.tp_value is None or self.tp_value <= 0):
            raise ValueError("tp_mode requires tp_value > 0")
        if self.sl_mode != "logged" and (self.sl_value is None or self.sl_value <= 0):
            raise ValueError(f"sl_mode={self.sl_mode!r} requires sl_value > 0")
        return self


class IfvgModelConfig(_Frozen):
    """CatBoost gate block; ``model=None`` on the experiment = expectancy-only."""

    iterations: int = Field(default=200, ge=1)
    depth: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=0.08, gt=0)
    seed: int = 7
    split_fracs: tuple[float, ...] = (0.5, 0.65, 0.8)
    purge_days: int = Field(default=2, ge=0)
    thresholds: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)


class IfvgExperimentConfig(_Frozen):
    """Frozen, fully-resolved experiment spec; name/note live in save meta only."""

    filters: IfvgFilterConfig = Field(default_factory=IfvgFilterConfig)
    #: Custom ET session windows; when set, each trade is re-stamped offline
    #: from entry_ts_utc (filter semantics only — level construction is baked
    #: into the capture and goes through the Part C re-capture job).
    custom_sessions: dict[str, SessionWindow] | None = None
    #: Filter on the custom stamps ("none" admits unstamped rows).
    sessions_custom: tuple[str, ...] | None = None
    doc_defaults: IfvgDocDefaultsConfig = Field(default_factory=IfvgDocDefaultsConfig)
    scoring: IfvgScoringConfig = Field(default_factory=IfvgScoringConfig)
    sl_tp: IfvgSlTpConfig | None = None
    model: IfvgModelConfig | None = None

    @model_validator(mode="after")
    def _check_custom(self) -> IfvgExperimentConfig:
        if self.sessions_custom is not None:
            if not self.custom_sessions:
                raise ValueError("sessions_custom requires custom_sessions")
            unknown = set(self.sessions_custom) - set(self.custom_sessions) - {"none"}
            if unknown:
                raise ValueError(f"sessions_custom names not defined: {sorted(unknown)}")
        return self

    def experiment_hash(self, capture_tag: str) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(f"{payload}|{capture_tag}".encode()).hexdigest()[:8]


# ── JSON safety ───────────────────────────────────────────────────────────────


def json_safe(obj):
    """Recursively convert numpy/pandas scalars, timestamps, intervals and
    non-finite floats into JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | set):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Interval):
        return str(obj)
    if isinstance(obj, pd.Timestamp | datetime | date):
        return obj.isoformat()
    if isinstance(obj, str | int):
        return obj
    return str(obj)


# ── feature selection (parity with the gate script) ───────────────────────────

_EXCLUDE_PREFIXES = ("label_", "realized_", "bars_to_res_", "mfe_r", "mae_r")
_IDENTITY = {
    "setup_id",
    "trading_day",
    "entry_ts_utc",
    "selected",
    "drop_reason",
    "is_warmup",
    "days_of_htf_history",
    "profile_hash",
    "strategy_version",
    "label_window_end",
    "entry_slippage_next_open_pts",
    "entry_ticks",
    "stop_ticks",
    "parent_fvg_id",  # string identifier (IFVG-FIX F1), not a measurement
    "entry_model_final",  # provenance stamp (IFVG-FIX F5), redundant with entry_family
    # engine-added bookkeeping columns (never features):
    "row_id",
    "session_custom",
    "target",
    "p_win",
    "split",
    "bin",
}
_CATEGORICAL = (
    "entry_family",
    "direction",
    "session_engine",
    "session_doc",
    "tap_nearest_level_kind",
    "inv_sweep_swept_kinds",
)


def _feature_columns(ds: pd.DataFrame) -> list[str]:
    cols = []
    for col in ds.columns:
        if col in _IDENTITY or col.startswith("_"):
            continue
        if any(col.startswith(p) for p in _EXCLUDE_PREFIXES):
            continue
        # Raw timestamps are a monotone train-earlier-than-test signal under
        # expanding splits — exclude every datetime64 column BY DTYPE (F3).
        if pd.api.types.is_datetime64_any_dtype(ds[col]):
            continue
        cols.append(col)
    return cols


# ── custom sessions ───────────────────────────────────────────────────────────


def stamp_custom_sessions(
    frame: pd.DataFrame, custom_sessions: dict[str, SessionWindow]
) -> pd.Series:
    """Re-stamp each row's session from ``entry_ts_utc`` in America/New_York
    against the custom windows ([start, end) half-open; start > end crosses
    midnight; first matching window in dict order wins; no match = "none")."""
    et = pd.to_datetime(frame["entry_ts_utc"], utc=True).dt.tz_convert("America/New_York")
    minutes = et.dt.hour * 60 + et.dt.minute

    def _mins(hhmm: str) -> int:
        h, m = hhmm.split(":")
        return int(h) * 60 + int(m)

    stamp = pd.Series("none", index=frame.index, dtype=object)
    unassigned = pd.Series(True, index=frame.index)
    for name, window in custom_sessions.items():
        start, end = _mins(window.start), _mins(window.end)
        if start < end:
            mask = (minutes >= start) & (minutes < end)
        elif start > end:  # crosses midnight
            mask = (minutes >= start) | (minutes < end)
        else:  # start == end: empty window
            mask = pd.Series(False, index=frame.index)
        take = mask & unassigned
        stamp[take] = name
        unassigned &= ~take
    return stamp


# ── SL/TP override recompute ──────────────────────────────────────────────────


def recover_override_stop_ticks(
    *,
    direction: str,
    entry_ticks: int,
    logged_stop_ticks: int,
    sl_tp: IfvgSlTpConfig,
    tick_size: float,
) -> int:
    """New stop in ticks. Swing recovery inverts the reducer's 1-tick buffer:
    LONG swing_low = stop+1, new stop = swing_low - N; SHORT swing_high =
    stop-1, new stop = swing_high + N."""
    if sl_tp.sl_mode == "logged":
        return logged_stop_ticks
    is_long = direction == "LONG"
    if sl_tp.sl_mode == "fixed_points":
        offset = int(round(sl_tp.sl_value / tick_size))
        return entry_ticks - offset if is_long else entry_ticks + offset
    buffer_ticks = int(round(sl_tp.sl_value))
    if is_long:
        return (logged_stop_ticks + 1) - buffer_ticks
    return (logged_stop_ticks - 1) + buffer_ticks


def _default_bars_loader(capture_cfg: IfvgCaptureConfig) -> Callable[[str], list[Bar]]:
    cache: dict[str, list[Bar]] = {}

    def _load(day: str) -> list[Bar]:
        if day not in cache:
            artifacts = load_day_artifacts(day, capture_cfg, expected_seeds=None)
            cache[day] = (
                [b for b in artifacts.bars if b.timeframe_ticks == 60] if artifacts else []
            )
        return cache[day]

    return _load


def _apply_sl_tp(
    work: pd.DataFrame,
    config: IfvgExperimentConfig,
    *,
    tick_size: float,
    bars_loader: Callable[[str], list[Bar]],
) -> tuple[pd.DataFrame, dict]:
    """Recompute outcomes through the shared SC kernel under overridden SL/TP.

    Entry stays the logged ``entry_ticks``; rows whose overridden risk falls
    below 1 tick are dropped (counted), as are rows with no forward bars.
    """
    sl_tp = config.sl_tp
    base_r = R_FAMILY_MULTIPLE[config.scoring.r_family]
    cost = config.scoring.cost_points
    kept, recs = [], []
    risk_dropped = no_forward_dropped = 0
    for idx, row in work.iterrows():
        entry_ticks = int(row["entry_ticks"])
        logged_stop = int(row["stop_ticks"])
        direction = str(row["direction"])
        new_stop = recover_override_stop_ticks(
            direction=direction,
            entry_ticks=entry_ticks,
            logged_stop_ticks=logged_stop,
            sl_tp=sl_tp,
            tick_size=tick_size,
        )
        is_long = direction == "LONG"
        risk_ticks = entry_ticks - new_stop if is_long else new_stop - entry_ticks
        if risk_ticks < 1:
            risk_dropped += 1
            continue
        bars = bars_loader(str(row["trading_day"]))
        entry_ts = row["entry_ts_utc"]
        forward = [b for b in bars if b.close_ts_utc > entry_ts]
        if not forward:
            no_forward_dropped += 1
            continue
        risk_pts = risk_ticks * tick_size
        if sl_tp.tp_mode == "r_multiple":
            r_multiple = float(sl_tp.tp_value)
        elif sl_tp.tp_mode == "fixed_points":
            r_multiple = float(sl_tp.tp_value) / risk_pts
        else:
            r_multiple = base_r
        outcome = resolve_ifvg_outcome(
            entry_ticks=entry_ticks,
            stop_ticks=new_stop,
            direction=Direction.LONG if is_long else Direction.SHORT,
            forward_bars_1m=forward,
            tick_size=tick_size,
            r_multiple=r_multiple,
        )
        sign = 1 if is_long else -1
        if outcome.label == "win":
            realized = r_multiple * risk_pts
        elif outcome.label == "loss":
            realized = -risk_pts
        else:
            realized = sign * (forward[-1].close_ticks - entry_ticks) * tick_size
        kept.append(idx)
        recs.append(
            {
                "stop_ticks": new_stop,
                "risk_ticks": risk_ticks,
                "risk_points": risk_pts,
                "label_window_end": forward[-1].close_ts_utc,
                "_label": outcome.label,
                "_bars_to_res": outcome.bars_to_resolution,
                "_realized_pts": realized,
                "_net_r": (realized - cost) / risk_pts,
                "_mfe_r": outcome.mfe_r,
                "_mae_r": outcome.mae_r,
                "_tp_r": r_multiple,
            }
        )
    out = work.loc[kept].copy()
    if recs:
        overrides = pd.DataFrame(recs, index=kept)
        for col in overrides.columns:
            out[col] = overrides[col]
    else:  # every row dropped — still emit the working-column schema
        out["_label"] = pd.Series(dtype=object)
        for col in ("_bars_to_res", "_realized_pts", "_net_r", "_mfe_r", "_mae_r", "_tp_r"):
            out[col] = pd.Series(dtype=float)
    info = {
        "sl_tp_risk_dropped": risk_dropped,
        "sl_tp_no_forward_dropped": no_forward_dropped,
    }
    return out, info


# ── row preparation (shared by the normal path and sealed validation) ─────────


def _prepare_frame(
    frame: pd.DataFrame,
    config: IfvgExperimentConfig,
    *,
    tick_size: float,
    bars_loader: Callable[[str], list[Bar]] | None,
) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    """Filters + working columns + dtype conversion. NO sealed clamp here —
    callers own sealing (the normal path clamps BEFORE calling this)."""
    f = config.filters
    tag = config.scoring.r_family
    label_col = f"label_{tag}"
    info: dict = {"rows_in": int(len(frame))}

    mask = frame[label_col].fillna("no_forward") != "no_forward"
    if not f.include_warmup:
        mask &= ~frame["is_warmup"].astype(bool)
    if f.day_start is not None:
        mask &= frame["trading_day"] >= f.day_start
    if f.day_end is not None:
        mask &= frame["trading_day"] <= f.day_end
    mask &= frame["entry_family"].isin(f.families)
    if f.direction != "both":
        mask &= frame["direction"] == f.direction.upper()
    if f.sessions_engine is not None:
        mask &= frame["session_engine"].fillna("none").isin(f.sessions_engine)
    if f.sessions_doc is not None:
        mask &= frame["session_doc"].fillna("none").isin(f.sessions_doc)
    work = frame.loc[mask].copy()

    if config.custom_sessions:
        work["session_custom"] = stamp_custom_sessions(work, config.custom_sessions)
        if config.sessions_custom is not None:
            work = work[work["session_custom"].isin(config.sessions_custom)].copy()
    if config.doc_defaults.apply:
        passing = doc_default_pass(
            work, floors=config.doc_defaults.floors(), caps=config.doc_defaults.caps()
        )
        work = work.loc[passing].copy()
    if f.max_trades_per_day is not None:
        work = (
            work.sort_values("entry_ts_utc", kind="mergesort")
            .groupby("trading_day", sort=False)
            .head(f.max_trades_per_day)
        )
    info["rows_after_filters"] = int(len(work))

    if config.sl_tp is not None:
        if bars_loader is None:
            raise ValueError("sl_tp overrides need a bars loader (capture_cfg or injected)")
        work, sl_info = _apply_sl_tp(
            work, config, tick_size=tick_size, bars_loader=bars_loader
        )
        info.update(sl_info)
    else:
        cost = config.scoring.cost_points
        work["_label"] = work[label_col]
        work["_bars_to_res"] = pd.to_numeric(
            work.get(f"bars_to_res_{tag}", np.nan), errors="coerce"
        )
        work["_realized_pts"] = pd.to_numeric(work[f"realized_pts_{tag}"], errors="coerce")
        work["_net_r"] = (work["_realized_pts"] - cost) / pd.to_numeric(
            work["risk_points"], errors="coerce"
        )
        work["_mfe_r"] = pd.to_numeric(work.get("mfe_r", np.nan), errors="coerce")
        work["_mae_r"] = pd.to_numeric(work.get("mae_r", np.nan), errors="coerce")
        work["_tp_r"] = R_FAMILY_MULTIPLE[tag]
    work["_risk_points"] = pd.to_numeric(work["risk_points"], errors="coerce")

    features = _feature_columns(work)
    cats = [c for c in _CATEGORICAL if c in features]
    for col in cats:
        work[col] = work[col].fillna("none").astype(str)
    for col in [c for c in features if c not in cats]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work["target"] = (work["_label"] == "win").astype(int)
    work = work.sort_values("entry_ts_utc").reset_index(drop=True)
    info["rows_final"] = int(len(work))
    return work, features, cats, info


# ── expectancy + model sections ───────────────────────────────────────────────


def _stat_entry(sub: pd.DataFrame) -> dict:
    n = len(sub)
    wins = int((sub["_label"] == "win").sum())
    net = pd.to_numeric(sub["_net_r"], errors="coerce").dropna()
    return {
        "n": n,
        "win_rate": (wins / n) if n else None,
        "win_rate_ci95": wilson_ci(wins, n),
        "mean_net_r": float(net.mean()) if len(net) else None,
        "net_r_ci95": mean_ci(net),
    }


def _expectancy_section(work: pd.DataFrame) -> dict:
    out = {"overall": _stat_entry(work), "per_family": {}, "per_session_engine": {}}
    for family, group in work.groupby("entry_family"):
        out["per_family"][str(family)] = _stat_entry(group)
    for session, group in work.groupby("session_engine"):
        out["per_session_engine"][str(session)] = _stat_entry(group)
    out["per_session_doc"] = {
        str(s): _stat_entry(g) for s, g in work.groupby("session_doc")
    }
    if "session_custom" in work.columns:
        out["per_session_custom"] = {
            str(s): _stat_entry(g) for s, g in work.groupby("session_custom")
        }
    out["per_session_engine_family"] = {
        f"{s}|{f}": _stat_entry(g)
        for (s, f), g in work.groupby(["session_engine", "entry_family"])
    }
    return out


def dedup_pooled_oos(oos: pd.DataFrame) -> pd.DataFrame:
    """Pooled OOS rows appear once per covering split (expanding windows
    overlap); deduplicate on the stable ``row_id``, keeping the prediction from
    the LAST split (largest train window)."""
    return oos.sort_values("split", kind="mergesort").drop_duplicates("row_id", keep="last")


def _oos_stats(sub: pd.DataFrame) -> tuple[int, float, float]:
    if sub.empty:
        return 0, float("nan"), float("nan")
    wins = float((sub["_label"] == "win").mean())
    return len(sub), wins, float(pd.to_numeric(sub["_net_r"], errors="coerce").mean())


def _run_model(
    work: pd.DataFrame,
    config: IfvgExperimentConfig,
    features: list[str],
    cats: list[str],
) -> tuple[dict, pd.DataFrame | None, list[dict], list[dict]]:
    mc = config.model
    days = sorted(work["trading_day"].unique())
    section: dict = {"n_rows": int(len(work)), "n_days": len(days)}
    if len(work) < 40 or len(days) < 12:
        section["skipped_reason"] = (
            f"too few rows/days for the gate (rows {len(work)}, days {len(days)})"
        )
        return section, None, [], []

    from catboost import CatBoostClassifier, Pool

    splits: list[dict] = []
    folds: list[pd.DataFrame] = []
    importance: list[dict] = []
    permutation_raw: dict[str, list[float]] = {c: [] for c in features}
    for split_idx, frac in enumerate(mc.split_fracs):
        k = int(len(days) * frac)
        train_days = set(days[:k])
        test_days = set(days[k + mc.purge_days :])
        train = work[work["trading_day"].isin(train_days)]
        test = work[work["trading_day"].isin(test_days)]
        if len(train) < 30 or len(test) < 10:
            splits.append(
                {"frac": frac, "train_n": len(train), "test_n": len(test), "skipped": True}
            )
            continue
        model = CatBoostClassifier(
            iterations=mc.iterations,
            depth=mc.depth,
            learning_rate=mc.learning_rate,
            loss_function="Logloss",
            random_seed=mc.seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(train[features].fillna(-1), train["target"], cat_features=cats))
        test_x = test[features].fillna(-1)
        proba = model.predict_proba(Pool(test_x, cat_features=cats))[:, 1]
        fold = test.copy()
        fold["p_win"] = proba
        fold["split"] = frac
        folds.append(fold)
        n, wr, net = _oos_stats(fold)
        splits.append(
            {
                "frac": frac,
                "train_n": len(train),
                "test_n": n,
                "win_rate": wr,
                "mean_net_r": net,
                "skipped": False,
            }
        )
        importance.append(
            {
                "frac": frac,
                "importances": dict(
                    zip(features, [float(v) for v in model.feature_importances_], strict=True)
                ),
            }
        )
        target = test["target"].to_numpy()
        base_brier = float(np.mean((proba - target) ** 2))
        for feat_idx, col in enumerate(features):
            rng = np.random.default_rng([mc.seed, split_idx, feat_idx])
            shuffled = test_x.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            p2 = model.predict_proba(Pool(shuffled, cat_features=cats))[:, 1]
            permutation_raw[col].append(float(np.mean((p2 - target) ** 2)) - base_brier)

    section["splits"] = splits
    if not folds:
        section["skipped_reason"] = "no viable splits"
        return section, None, importance, []
    oos = pd.concat(folds, ignore_index=True)

    n, wr, net = _oos_stats(oos)
    section["pooled"] = {"n": n, "win_rate": wr, "mean_net_r": net}
    # Deliberately the UNMODIFIED doc defaults (not config.doc_defaults overrides):
    # a fixed baseline keeps this number comparable across every run.
    dd = oos[doc_default_pass(oos)]
    n, wr, net = _oos_stats(dd)
    section["doc_defaults_baseline"] = {"n": n, "win_rate": wr, "mean_net_r": net}

    dedup = dedup_pooled_oos(oos)
    n, wr, net = _oos_stats(dedup)
    section["dedup"] = {
        "n": n,
        "win_rate": wr,
        "mean_net_r": net,
        "brier": float(((dedup["p_win"] - dedup["target"]) ** 2).mean()),
        "note": "one prediction per row_id, kept from the last (largest-train) split",
    }

    coverage = []
    for thr in mc.thresholds:
        sub = oos[oos["p_win"] >= thr]
        n, wr, net = _oos_stats(sub)
        coverage.append(
            {
                "thr": thr,
                "n": n,
                "coverage": n / max(1, len(oos)),
                "win_rate": wr,
                "mean_net_r": net,
            }
        )
    section["coverage"] = coverage

    # qcut bins kept OUT of the oos frame (Interval dtype is not parquet-safe).
    bins = pd.qcut(oos["p_win"], q=min(4, oos["p_win"].nunique()), duplicates="drop")
    calibration = [
        {
            "bin": str(interval),
            "n": len(group),
            "mean_p": float(group["p_win"].mean()),
            "actual": float(group["target"].mean()),
        }
        for interval, group in oos.groupby(bins, observed=True)
    ]
    section["calibration"] = calibration
    brier = float(((oos["p_win"] - oos["target"]) ** 2).mean())
    base_rate = float(oos["target"].mean())
    section["brier"] = brier
    section["base_rate"] = base_rate
    section["ref_brier"] = base_rate * (1.0 - base_rate)

    section["per_session"] = [
        {"session": str(s), **dict(zip(("n", "win_rate", "mean_net_r"), _oos_stats(g),
                                       strict=True))}
        for s, g in oos.groupby("session_engine")
    ]
    section["per_family"] = [
        {"family": str(f), **dict(zip(("n", "win_rate", "mean_net_r"), _oos_stats(g),
                                      strict=True))}
        for f, g in oos.groupby("entry_family")
    ]
    if "session_custom" in oos.columns:
        section["per_session_custom"] = [
            {"session": str(s), **dict(zip(("n", "win_rate", "mean_net_r"), _oos_stats(g),
                                           strict=True))}
            for s, g in oos.groupby("session_custom")
        ]

    permutation = sorted(
        (
            {
                "feature": col,
                "mean_delta_brier": float(np.mean(deltas)),
                "per_split": [float(d) for d in deltas],
            }
            for col, deltas in permutation_raw.items()
            if deltas
        ),
        key=lambda e: e["mean_delta_brier"],
        reverse=True,
    )
    return section, oos, importance, permutation


# ── the engine ────────────────────────────────────────────────────────────────


def _default_dataset_path(capture_cfg: IfvgCaptureConfig) -> Path:
    return (
        Path(capture_cfg.data_dir)
        / capture_cfg.symbol
        / f"ifvg_entry_dataset_{capture_cfg.capture_tag()}.parquet"
    )


def run_ifvg_experiment(
    config: IfvgExperimentConfig,
    *,
    dataset: pd.DataFrame | None = None,
    dataset_path: Path | None = None,
    capture_cfg: IfvgCaptureConfig | None = None,
    bars_loader: Callable[[str], list[Bar]] | None = None,
) -> dict:
    """Run one experiment over pre-seal rows only; returns JSON-safe sections.

    The raw pooled-OOS frame rides along under the PRIVATE ``"_oos_frame"`` key
    (a DataFrame; stripped by :func:`save_experiment` / :func:`json_safe`).
    """
    capture_cfg = capture_cfg or IfvgCaptureConfig()
    capture_tag = capture_cfg.capture_tag()
    if dataset is None:
        dataset = pd.read_parquet(dataset_path or _default_dataset_path(capture_cfg))
    frame = dataset.reset_index(drop=True).copy()
    frame["row_id"] = frame.index

    # SEALED GUARD — non-configurable; only the COUNT of clamped rows escapes.
    sealed_mask = frame["trading_day"].astype(str) >= SEALED_HOLDOUT_START
    sealed_excluded = int(sealed_mask.sum())
    pre_seal = frame.loc[~sealed_mask]

    if bars_loader is None and config.sl_tp is not None:
        bars_loader = _default_bars_loader(capture_cfg)
    work, features, cats, info = _prepare_frame(
        pre_seal, config, tick_size=capture_cfg.tick_size, bars_loader=bars_loader
    )

    caveats = [
        "Candidate pooling: non-selected family rows are label-resolved hypotheticals, "
        "not walked executions.",
        MULTIPLE_COMPARISONS_CAVEAT,
    ]
    if config.sl_tp is not None:
        caveats.insert(0, NO_PATH_DEPENDENCE_CAVEAT)

    model_section: dict | None = None
    oos = None
    importance: list[dict] = []
    permutation: list[dict] = []
    if config.model is not None:
        model_section, oos, importance, permutation = _run_model(work, config, features, cats)

    numeric_features = [c for c in features if c not in cats]
    result: dict = {
        "meta": {
            "engine": "ifvg_experiment_v1",
            "experiment_hash": config.experiment_hash(capture_tag),
            "capture_tag": capture_tag,
            "created_utc": datetime.now(UTC).isoformat(),
            "sealed": {
                "sealed_start": SEALED_HOLDOUT_START,
                "sealed_rows_excluded": sealed_excluded,
            },
            "counts": info,
            "caveats": caveats,
        },
        "config": config.model_dump(mode="json"),
        "features": {"all": features, "categorical": cats, "n": len(features)},
        "expectancy": _expectancy_section(work),
        "trade_stats": compute_trade_stats(work, cost_points=config.scoring.cost_points),
        "feature_insight": compute_feature_insight(
            work,
            numeric_features,
            cats,
            model_importance=importance,
            permutation_importance=permutation,
        ),
        "model": model_section,
    }
    if oos is not None:
        result["_oos_frame"] = oos
    return result


# ── persistence ───────────────────────────────────────────────────────────────


def _run_dir(ref: str | Path, base_dir: Path | None) -> Path:
    path = Path(ref)
    if path.is_dir() and (path / "config.json").exists():
        return path
    return Path(base_dir or EXPERIMENTS_DIR) / str(ref)


def save_experiment(
    config: IfvgExperimentConfig,
    result: dict,
    *,
    name: str | None = None,
    note: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Persist ``config.json`` (meta + fully-resolved config) + ``result.json``
    (+ ``oos.parquet`` when the model ran) under ``<base>/<experiment_hash>/``."""
    run_hash = result["meta"]["experiment_hash"]
    run_dir = Path(base_dir or EXPERIMENTS_DIR) / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    created = result["meta"]["created_utc"]
    payload = {
        "meta": {
            "name": name or f"{str(created)[:10]}-{run_hash}",
            "note": note,
            "created_utc": created,
        },
        "config": config.model_dump(mode="json"),
        "experiment_hash": run_hash,
        "capture_tag": result["meta"]["capture_tag"],
    }
    (run_dir / "config.json").write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    clean = {k: v for k, v in result.items() if not k.startswith("_")}
    (run_dir / "result.json").write_text(
        json.dumps(json_safe(clean), indent=2, sort_keys=True), encoding="utf-8"
    )
    oos = result.get("_oos_frame")
    if oos is not None:
        oos.to_parquet(run_dir / "oos.parquet", index=False)
    return run_dir


def sealed_ledger_count(base_dir: Path | None = None) -> int:
    """GLOBAL sealed-evaluation count across ALL configs.

    Holdout degradation is per-look, not per-config (later configs are chosen
    with knowledge of earlier sealed results), so this single number is the
    trust meter for every sealed result."""
    ledger = Path(base_dir or EXPERIMENTS_DIR) / "sealed_ledger.jsonl"
    if not ledger.exists():
        return 0
    return sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())


def _sealed_history(base: Path, run_hash: str) -> list[int]:
    ledger = base / "sealed_ledger.jsonl"
    if not ledger.exists():
        return []
    seqs = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("config_hash") == run_hash:
            seqs.append(int(entry["seq"]))
    return seqs


def list_experiments(base_dir: Path | None = None) -> list[dict]:
    """Saved-run summaries (hash, name, created, key stats, sealed history)."""
    base = Path(base_dir or EXPERIMENTS_DIR)
    if not base.exists():
        return []
    summaries = []
    for run_dir in sorted(p for p in base.iterdir() if (p / "config.json").exists()):
        cfg_payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        stats: dict = {}
        result_path = run_dir / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            ts = result.get("trade_stats", {})
            stats = {
                "n_trades": ts.get("n"),
                "net_r": ts.get("equity", {}).get("r", {}).get("net"),
                "profit_factor": (
                    ts.get("pnl_usd", {}).get("all", {}).get("profit_factor")
                ),
            }
        run_hash = cfg_payload.get("experiment_hash", run_dir.name)
        summaries.append(
            {
                "experiment_hash": run_hash,
                "name": cfg_payload.get("meta", {}).get("name"),
                "created_utc": cfg_payload.get("meta", {}).get("created_utc"),
                "capture_tag": cfg_payload.get("capture_tag"),
                "stats": stats,
                "sealed_validations": _sealed_history(base, run_hash),
            }
        )
    return summaries


def load_experiment(ref: str | Path, base_dir: Path | None = None) -> dict:
    """Load a saved run: reconstructed config + meta + result (+ oos frame)."""
    run_dir = _run_dir(ref, base_dir)
    payload = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    result_path = run_dir / "result.json"
    oos_path = run_dir / "oos.parquet"
    return {
        "run_dir": run_dir,
        "meta": payload.get("meta", {}),
        "capture_tag": payload.get("capture_tag"),
        "experiment_hash": payload.get("experiment_hash"),
        "config": IfvgExperimentConfig(**payload["config"]),
        "result": (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        ),
        "oos": pd.read_parquet(oos_path) if oos_path.exists() else None,
    }


def delete_experiment(ref: str | Path, base_dir: Path | None = None) -> None:
    """Delete one saved run directory (the sealed ledger is append-only and
    never touched). Runs with sealed validations are REFUSED — their result
    artifacts are pointed at by ledger entries and must survive."""
    base = Path(base_dir or EXPERIMENTS_DIR)
    run_dir = _run_dir(ref, base_dir)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return
    run_hash = json.loads(config_path.read_text(encoding="utf-8")).get("experiment_hash")
    if run_hash and _sealed_history(base, run_hash):
        raise ValueError(
            f"run {run_hash} has sealed validations on the ledger; refusing to delete"
        )
    shutil.rmtree(run_dir)


# ── sealed validation (the ONLY code path allowed to touch sealed rows) ───────


def run_sealed_validation(
    ref: str | Path,
    *,
    dataset: pd.DataFrame | None = None,
    dataset_path: Path | None = None,
    capture_cfg: IfvgCaptureConfig | None = None,
    bars_loader: Callable[[str], list[Bar]] | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Apply a SAVED run's exact configuration to the sealed holdout.

    Retrains deterministically on pre-seal rows per the saved config (or runs
    expectancy-only), scores sealed rows only, appends the append-only global
    ledger and writes ``sealed_result_{seq}.json`` into the run dir (one file
    per ledgered look). Files only — sealed statistics are NEVER printed to
    stdout. Returns the result path.
    """
    base = Path(base_dir or EXPERIMENTS_DIR)
    loaded = load_experiment(ref, base)
    config: IfvgExperimentConfig = loaded["config"]
    run_dir: Path = loaded["run_dir"]

    capture_cfg = capture_cfg or IfvgCaptureConfig()
    if dataset is None:
        dataset = pd.read_parquet(dataset_path or _default_dataset_path(capture_cfg))
    frame = dataset.reset_index(drop=True).copy()
    frame["row_id"] = frame.index
    sealed_mask = frame["trading_day"].astype(str) >= SEALED_HOLDOUT_START
    if bars_loader is None and config.sl_tp is not None:
        bars_loader = _default_bars_loader(capture_cfg)
    pre_work, features, cats, _ = _prepare_frame(
        frame.loc[~sealed_mask], config, tick_size=capture_cfg.tick_size,
        bars_loader=bars_loader,
    )
    sealed_work, _, _, sealed_info = _prepare_frame(
        frame.loc[sealed_mask], config, tick_size=capture_cfg.tick_size,
        bars_loader=bars_loader,
    )

    model_section: dict | None = None
    if config.model is not None and len(pre_work) >= 40 and len(sealed_work) > 0:
        from catboost import CatBoostClassifier, Pool

        mc = config.model
        model = CatBoostClassifier(
            iterations=mc.iterations,
            depth=mc.depth,
            learning_rate=mc.learning_rate,
            loss_function="Logloss",
            random_seed=mc.seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(pre_work[features].fillna(-1), pre_work["target"], cat_features=cats))
        proba = model.predict_proba(
            Pool(sealed_work[features].fillna(-1), cat_features=cats)
        )[:, 1]
        sealed_work = sealed_work.copy()
        sealed_work["p_win"] = proba
        coverage = []
        for thr in mc.thresholds:
            sub = sealed_work[sealed_work["p_win"] >= thr]
            n, wr, net = _oos_stats(sub)
            coverage.append(
                {
                    "thr": thr,
                    "n": n,
                    "coverage": n / max(1, len(sealed_work)),
                    "win_rate": wr,
                    "mean_net_r": net,
                }
            )
        target = sealed_work["target"].to_numpy()
        model_section = {
            "trained_on_pre_seal_rows": int(len(pre_work)),
            "coverage": coverage,
            "brier": float(np.mean((proba - target) ** 2)),
            "base_rate": float(np.mean(target)),
        }

    # Sequence = ledger line count + 1. Single-process assumption: the dashboard
    # and CLI are the only writers and never run sealed validations concurrently;
    # no file lock is taken.
    ledger = base / "sealed_ledger.jsonl"
    seq = 1
    if ledger.exists():
        seq = (
            sum(1 for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip())
            + 1
        )
    # Per-look artifact: each ledgered sealed evaluation keeps its own file.
    result_path = run_dir / f"sealed_result_{seq}.json"
    sealed_result = {
        "sealed_validation": True,
        "sequence": seq,
        "experiment_hash": loaded["experiment_hash"],
        "capture_tag": loaded["capture_tag"],
        "created_utc": datetime.now(UTC).isoformat(),
        "sealed_start": SEALED_HOLDOUT_START,
        "counts": sealed_info,
        "expectancy": _expectancy_section(sealed_work),
        "trade_stats": compute_trade_stats(
            sealed_work, cost_points=config.scoring.cost_points
        ),
        "model": model_section,
    }
    result_path.write_text(
        json.dumps(json_safe(sealed_result), indent=2, sort_keys=True), encoding="utf-8"
    )
    base.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "config_hash": loaded["experiment_hash"],
                    "timestamp_utc": sealed_result["created_utc"],
                    "seq": seq,
                    "result_path": str(result_path),
                }
            )
            + "\n"
        )
    return result_path
