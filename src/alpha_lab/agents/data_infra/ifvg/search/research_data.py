"""Lazy, exact-child context companions and shared-Core research outcomes.

Construction is read-only. ``prepare`` is the explicit worker boundary which
may replay the approved source and publish a new companion. Original v2
tables and historical labels are never written or relabeled in place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from datetime import date
from enum import Enum
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import ClassVar, Literal

import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field
from strategy_core.strategies.ifvg_smc.context_config import ContextFeatureConfig
from strategy_core.strategies.ifvg_smc.labels import resolve_ifvg_outcome
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection
from strategy_core.types import Bar, BarKind, CloseReason, Direction

from ..artifact_io import VerifiedIfvgArtifact, VerifiedIfvgPair, validate_exact_context_links
from ..config import IfvgCaptureConfig, IfvgV3CaptureConfig
from ..context_contracts import (
    ContextRecordTable,
    validate_context_foreign_keys,
    validate_context_primary_keys,
    validate_context_table_identity,
)
from ..context_experiment_contracts import ArtifactReference, PairedIfvgArtifactReference
from ..context_feature_view import build_candidate_feature_view, m3_cohort_status
from ..context_schemas import context_frame_from_table, context_table_from_frame
from ..contracts import RecordTable
from ..dataset import build_ifvg_v3_capture
from ..development_access import DEVELOPMENT_CUTOFF_UTC, DevelopmentReplayPolicy
from ..manifest import source_tree_hash
from ..profiles import ResolvedProfileConfig
from .child_replay import ArtifactProvenanceReadAdapter
from .identities import (
    SHA256_PATTERN,
    CoreStrategyReplayIdentity,
    EnvelopeBase,
    FrozenContract,
    canonical_contract_sha256,
)
from .research_subject import (
    ResearchSubject,
    preflight_research_subject,
    verify_research_input_hashes,
)
from .review_evidence import load_search_review_evidence
from .store import has_envelope, load_sidecar_bytes, load_verified_envelope, save_or_reuse_envelope

CONTEXT_STORE = "research_context_companions"
LABEL_POLICY = "ifvg_configured_r_resolution_or_study_cutoff_core_v1"


class ResearchContextPayload(FrozenContract):
    schema_version: Literal[1] = 1
    subject: ResearchSubject
    producer_source_hash: str = Field(pattern=SHA256_PATTERN)
    feature_schema_hash: str = Field(pattern=SHA256_PATTERN)
    context_config_hash: str = Field(pattern=SHA256_PATTERN)
    context_config_json: str
    neutrality_policy: Literal["all_accepted_core_tables_exact_v1"] = (
        "all_accepted_core_tables_exact_v1"
    )


class ResearchContextEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "research_context_companion_id"
    research_context_companion_id: str = Field(pattern=SHA256_PATTERN)
    payload: ResearchContextPayload


@dataclass(frozen=True)
class LoadedResearchContext:
    envelope: ResearchContextEnvelope
    context_tables: dict
    bars_1m: pd.DataFrame
    directory: Path
    manifest_payload_sha256: str
    forward_bars_sha256: str


def expected_research_context(subject: ResearchSubject, repo_root: Path):
    """Pure configuration/source identity; never decode bars or run a replay."""
    cfg = IfvgV3CaptureConfig(
        core=IfvgCaptureConfig(
            section=IfvgSmcSection.model_validate(subject.section_mapping),
            data_dir=Path(repo_root) / "data/databento",
        ),
        context=ContextFeatureConfig(),
        accepted_v2_dataset_id=subject.v2_dataset_id,
        accepted_v2_manifest_sha256=subject.v2_manifest_hash,
    )
    producer = source_tree_hash(
        Path(repo_root),
        (
            "src/alpha_lab/agents/data_infra/ifvg/search/research_data.py",
            "src/alpha_lab/agents/data_infra/ifvg/search/research_subject.py",
            "src/alpha_lab/agents/data_infra/ifvg/capture_driver.py",
            "src/alpha_lab/agents/data_infra/ifvg/dataset.py",
            "src/alpha_lab/agents/data_infra/ifvg/context_feature_view.py",
        ),
    )
    return ResearchContextEnvelope.from_payload(
        ResearchContextPayload(
            subject=subject,
            producer_source_hash=producer,
            feature_schema_hash=cfg.feature_schema_hash,
            context_config_hash=cfg.context_config_hash,
            context_config_json=json.dumps(asdict(cfg.context), sort_keys=True),
        )
    ), cfg


def bars_to_frame(bars) -> pd.DataFrame:
    rows = []
    for bar in bars:
        row = {
            key: value.value if isinstance(value, Enum) else value
            for key, value in asdict(bar).items()
        }
        row["trading_day"] = bar.trading_day.isoformat()
        row["availability_ts_utc"] = bar.availability_ts_utc
        # Chart and fold windows consume the logical completed-bar boundary.
        row["close_ts_utc"] = bar.availability_ts_utc
        rows.append(row)
    return pd.DataFrame(rows)


def frame_to_bars(frame: pd.DataFrame) -> tuple[Bar, ...]:
    """Serialization adapter only; all barrier decisions remain in Core."""

    def timestamp(row, name):
        value = row.get(name)
        return None if value is None or pd.isna(value) else pd.Timestamp(value).to_pydatetime()

    bars = []
    for row in frame.to_dict("records"):
        bars.append(
            Bar(
                timeframe_ticks=int(row.get("timeframe_ticks", 60)),
                trading_day=date.fromisoformat(str(row["trading_day"])[:10]),
                bar_index=int(row.get("bar_index", len(bars))),
                bar_id=str(row["bar_id"]),
                open_ts_utc=timestamp(row, "open_ts_utc") or timestamp(row, "close_ts_utc"),
                close_ts_utc=timestamp(row, "close_ts_utc"),
                open_ticks=int(row["open_ticks"]),
                high_ticks=int(row["high_ticks"]),
                low_ticks=int(row["low_ticks"]),
                close_ticks=int(row["close_ticks"]),
                volume=int(row.get("volume", 0)),
                trade_count=int(row.get("trade_count", 0)),
                is_complete=bool(row.get("is_complete", True)),
                is_partial=bool(row.get("is_partial", False)),
                close_reason=(
                    CloseReason(row["close_reason"]) if pd.notna(row.get("close_reason")) else None
                ),
                kind=BarKind(row.get("kind", "time")),
                logical_open_ts_utc=timestamp(row, "logical_open_ts_utc"),
                logical_close_ts_utc=timestamp(row, "logical_close_ts_utc"),
            )
        )
    return tuple(bars)


def _validate_forward_bars(frame: pd.DataFrame, subject: ResearchSubject) -> None:
    required = {
        "bar_id",
        "trading_day",
        "close_ts_utc",
        "open_ticks",
        "high_ticks",
        "low_ticks",
        "close_ticks",
    }
    if required - set(frame):
        raise ValueError("research forward bars lack exact Core evidence")
    if frame["bar_id"].isna().any() or frame["bar_id"].duplicated().any():
        raise ValueError("research forward bars require unique exact bar IDs")
    if not set(frame["trading_day"].astype(str)) <= set(subject.replay_dates):
        raise ValueError("research forward bars escape the subject calendar")
    times = pd.to_datetime(frame["close_ts_utc"], utc=True, errors="raise")
    if (times >= pd.Timestamp(subject.cutoff_ts_utc)).any():
        raise ValueError("research forward bars cross the protected cutoff")
    if "timeframe_ticks" in frame and not frame["timeframe_ticks"].eq(60).all():
        raise ValueError("research forward bars must use the exact one-minute Core stream")
    if "availability_ts_utc" in frame and not times.equals(
        pd.to_datetime(frame["availability_ts_utc"], utc=True, errors="raise")
    ):
        raise ValueError("research forward bars disagree with logical availability")


def load_research_context_companion(store_root: Path, companion_id: str) -> LoadedResearchContext:
    root = Path(store_root)
    envelope = load_verified_envelope(root, CONTEXT_STORE, companion_id, ResearchContextEnvelope)
    tables = {}
    for table in ContextRecordTable:
        data = load_sidecar_bytes(root, CONTEXT_STORE, companion_id, f"{table.value}.parquet")
        tables[table] = context_frame_from_table(pq.read_table(BytesIO(data)))
        validate_context_primary_keys(table, tables[table])
        validate_context_table_identity(table, tables[table])
    validate_context_foreign_keys(tables)
    report = json.loads(load_sidecar_bytes(root, CONTEXT_STORE, companion_id, "neutrality.json"))
    if report.get("passed") is not True or set(report.get("tables", {})) != {
        table.value for table in RecordTable
    }:
        raise ValueError("research context companion lacks exact accepted-table neutrality")
    if any(
        value.get("matched") is not True
        or value.get("derived_content_sha256") != value.get("accepted_content_sha256")
        or value.get("derived_rows") != value.get("accepted_rows")
        for value in report["tables"].values()
    ):
        raise ValueError("research context neutrality contains a mismatched table")
    data = load_sidecar_bytes(root, CONTEXT_STORE, companion_id, "forward_bars.parquet")
    bars = pd.read_parquet(BytesIO(data))
    _validate_forward_bars(bars, envelope.payload.subject)
    directory = root / CONTEXT_STORE / companion_id
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    return LoadedResearchContext(
        envelope,
        tables,
        bars,
        directory,
        manifest["manifest_payload_sha256"],
        hashlib.sha256(data).hexdigest(),
    )


def build_configured_r_labels(
    candidates: pd.DataFrame,
    bars_1m: pd.DataFrame,
    *,
    r_multiple: float,
    tick_size: float = 0.25,
    cost_points: float = 0.0,
    cutoff_ts_utc: str = DEVELOPMENT_CUTOFF_UTC,
    label_source_id: str,
) -> tuple[str, pd.DataFrame]:
    """Actual configured target, shared Core resolver, cross-day forward window.

    Geometry-unavailable and cutoff-censored candidates remain explicit rows;
    they receive no binary target or manufactured realized P&L.
    """
    if (
        not all(isfinite(value) for value in (r_multiple, tick_size, cost_points))
        or r_multiple <= 0
        or tick_size <= 0
        or cost_points < 0
    ):
        raise ValueError("label target/tick size must be positive and costs nonnegative")
    if candidates["candidate_id"].isna().any() or candidates["candidate_id"].duplicated().any():
        raise ValueError("research labels require unique candidate IDs")
    cutoff = pd.Timestamp(cutoff_ts_utc)
    if cutoff.tzinfo is None:
        raise ValueError("label cutoff must be timezone aware")
    all_bars = tuple(
        sorted(frame_to_bars(bars_1m), key=lambda bar: (bar.availability_ts_utc, bar.bar_id))
    )
    by_id = {bar.bar_id: bar for bar in all_bars}
    if len(by_id) != len(all_bars):
        raise ValueError("duplicate label source bar IDs")
    policy_id = canonical_contract_sha256(
        {
            "policy": LABEL_POLICY,
            "r_multiple": r_multiple,
            "tick_size": tick_size,
            "cost_points": cost_points,
            "cutoff": cutoff.isoformat(),
            "source": label_source_id,
        }
    )
    rows = []
    for candidate in candidates.to_dict("records"):
        out = dict(candidate)
        out.update(
            {
                "label_policy_id": LABEL_POLICY,
                "label_policy_sha256": policy_id,
                "reward_r": r_multiple,
                "entry_available": False,
                "resolution_available": False,
                "label": "censored",
                "binary_target": None,
                "gross_r": None,
                "net_r": None,
                "censored": True,
                "censor_reason": "geometry_unavailable",
                "resolution_ts_utc": pd.NaT,
                "label_window_end": cutoff,
                "cutoff_ts_utc": cutoff,
                "resolution_bar_id": None,
            }
        )
        entry_id = candidate.get("geometry_entry_bar_bar_id", candidate.get("entry_bar_id"))
        entry_bar = by_id.get(str(entry_id))
        entry = candidate.get("entry_ticks", candidate.get("geometry_entry_ticks"))
        stop = candidate.get("proposed_stop_ticks", candidate.get("geometry_stop_ticks"))
        if entry_bar is None or pd.isna(entry) or pd.isna(stop):
            rows.append(out)
            continue
        direction = Direction(str(candidate["direction"]).upper())
        if not all(isfinite(float(value)) and float(value).is_integer() for value in (entry, stop)):
            raise ValueError("candidate entry and stop must be exact integer ticks")
        entry, stop = int(entry), int(stop)
        risk = entry - stop if direction is Direction.LONG else stop - entry
        if risk <= 0:
            out["censor_reason"] = "invalid_candidate_geometry"
            rows.append(out)
            continue
        if pd.Timestamp(entry_bar.availability_ts_utc) >= cutoff:
            raise ValueError("candidate entry reaches the protected cutoff")
        if pd.notna(candidate.get("entry_ts_utc")) and (
            pd.Timestamp(candidate["entry_ts_utc"]) != pd.Timestamp(entry_bar.availability_ts_utc)
        ):
            raise ValueError("candidate timestamp disagrees with exact geometry entry bar")
        forward = tuple(
            bar
            for bar in all_bars
            if entry_bar.availability_ts_utc < bar.availability_ts_utc < cutoff
        )
        outcome = resolve_ifvg_outcome(
            entry_ticks=entry,
            stop_ticks=stop,
            direction=direction,
            entry_bar=entry_bar,
            forward_bars_1m=forward,
            tick_size=tick_size,
            r_multiple=r_multiple,
        )
        resolved = outcome.resolution_bar_id is not None
        resolution_ts = by_id[outcome.resolution_bar_id].availability_ts_utc if resolved else pd.NaT
        gross = (
            abs(outcome.target_ticks - entry) / risk
            if outcome.label == "win"
            else -1.0
            if outcome.label == "loss"
            else None
        )
        out.update(
            {
                "entry_ticks": entry,
                "stop_ticks": stop,
                "target_ticks": outcome.target_ticks,
                "entry_ts_utc": entry_bar.availability_ts_utc,
                "entry_available": True,
                "resolution_available": resolved,
                "label": outcome.label,
                "binary_target": int(outcome.label == "win") if resolved else None,
                "gross_r": gross,
                "net_r": gross - cost_points / (risk * tick_size) if resolved else None,
                "risk_ticks": risk,
                "censored": not resolved,
                "censor_reason": None if resolved else "censored_study_cutoff",
                "resolution_bar_id": outcome.resolution_bar_id,
                "resolution_ts_utc": resolution_ts,
                "label_window_end": resolution_ts if resolved else cutoff,
                "bars_after_entry_to_resolution": outcome.bars_after_entry_to_resolution,
                "mfe_r": outcome.mfe_r,
                "mae_r": outcome.mae_r,
            }
        )
        rows.append(out)
    return LABEL_POLICY, pd.DataFrame(rows)


def scope_research_candidate_view(full, candidates, eligible_decisions, subject):
    """Retain emitted counterfactuals and exact block evidence in one child cohort."""
    if not candidates["is_warmup"].isin([True, False]).all():
        raise ValueError("saved candidate warmup stamps must be explicit booleans")
    extra = [
        column for column in candidates if column not in full.frame or column == "candidate_id"
    ]
    frame = full.frame.merge(
        candidates[extra], on="candidate_id", how="left", validate="one_to_one"
    )
    eligible = set(eligible_decisions["candidate_id"].astype(str))
    frame["was_eligible_decision"] = frame["candidate_id"].astype(str).isin(eligible)
    frame = frame.loc[
        frame["trading_day"].astype(str).isin(subject.evaluation_dates)
        & ~frame["is_warmup"].astype(bool)
    ].copy()
    if subject.cohort == "eligible_decisions":
        frame = frame.loc[frame["was_eligible_decision"]].copy()
    return replace(
        full,
        frame=frame.reset_index(drop=True),
        view_id=canonical_contract_sha256(
            {"source_view": full.view_id, "subject": subject.subject_id}
        ),
        m3_status=m3_cohort_status(frame),
    )


class ResearchPreparation:
    def __init__(
        self,
        subject: ResearchSubject,
        store_root: Path,
        repo_root: Path,
        *,
        output_root: Path | None = None,
        cost_points: float = 0.0,
        mbp1_contract=None,
        mbp1_coverage_evidence=None,
        mbp1_preflight=None,
        progress_callback=None,
    ):
        self.subject = subject
        self.store_root = Path(store_root)
        self.repo_root = Path(repo_root)
        self.output_root = Path(output_root) if output_root else self.store_root
        self.cost_points = cost_points
        self._loaded = None
        self._candidate_view = None
        self.mbp1_evidence = {}
        self.mbp1_contract = mbp1_contract
        self.mbp1_coverage_evidence = mbp1_coverage_evidence
        self.mbp1_preflight = mbp1_preflight
        self.progress_callback = progress_callback
        self.replay_invocations = 0
        self.context_reused = False

    def _input_hashes(self, evidence, cfg):
        return verify_research_input_hashes(self.subject, evidence, cfg)

    def prepare(self):
        if self._candidate_view is not None:
            return self
        preflight_research_subject(self.subject, self.store_root, self.repo_root)
        evidence = load_search_review_evidence(self.store_root, self.subject.core_replay_id)
        envelope, v3_cfg = expected_research_context(self.subject, self.repo_root)
        cfg, context = v3_cfg.core, v3_cfg.context
        section = cfg.section
        companion_id = envelope.research_context_companion_id
        self._input_hashes(evidence, cfg)
        if not has_envelope(self.output_root, CONTEXT_STORE, companion_id):
            core = CoreStrategyReplayIdentity.model_validate_json(self.subject.core_envelope_json)
            effective = evidence.dataset.reports["effective_config.json"]
            resolved = ResolvedProfileConfig(
                raw_ui_config={},
                effective_config=self.subject.section_mapping,
                evaluator_config=effective["evaluator"],
                diagnostics={},
                section=section,
                section_config_hash=self.subject.section_config_hash,
                evaluation_config_hash=evidence.dataset.manifest["identity"][
                    "evaluation_config_hash"
                ],
            )
            policy = ArtifactProvenanceReadAdapter(
                DevelopmentReplayPolicy(self.subject.replay_dates),
                artifact_provenance_dates=self.subject.artifact_provenance_dates,
            )
            self.replay_invocations += 1
            capture = build_ifvg_v3_capture(
                self.subject.replay_dates,
                v3_cfg,
                resolved,
                access_policy=policy,
                cached_artifacts_only=True,
                accepted_v2_tables=evidence.dataset.tables,
                strategy_core_commit=core.payload.strategy_core_commit,
                strategy_core_source_tree_hash=core.payload.strategy_core_source_identity,
                measure_performance=False,
                progress_fn=self.progress_callback,
            )
            self._input_hashes(evidence, cfg)
            sidecars = {}
            for table, frame in capture.context_tables.items():
                stream = BytesIO()
                pq.write_table(context_table_from_frame(table, frame), stream)
                sidecars[f"{table.value}.parquet"] = stream.getvalue()
            bars = bars_to_frame(
                bar
                for day in self.subject.replay_dates
                for bar in capture.bars_by_day[day]
                if bar.timeframe_ticks == 60
                and bar.availability_ts_utc < pd.Timestamp(self.subject.cutoff_ts_utc)
            )
            _validate_forward_bars(bars, self.subject)
            sidecars["forward_bars.parquet"] = bars.to_parquet(index=False)
            sidecars["neutrality.json"] = json.dumps(
                capture.baseline_reconciliation, sort_keys=True
            ).encode()
            save_or_reuse_envelope(self.output_root, CONTEXT_STORE, envelope, extra_files=sidecars)
        else:
            self.context_reused = True
        self._loaded = load_research_context_companion(self.output_root, companion_id)
        self.v2_tables = evidence.dataset.tables
        self.context_tables = self._loaded.context_tables
        reference = ArtifactReference(
            artifact_id=companion_id,
            manifest_payload_sha256=self._loaded.manifest_payload_sha256,
            artifact_kind="v3",
            dataset_schema_version=4,
            profile_hash=self.subject.section_config_hash,
            feature_formula_version=context.feature_formula_version,
        )
        context_artifact = VerifiedIfvgArtifact(
            reference, self._loaded.directory, {}, self.context_tables, {}
        )
        self.pair = VerifiedIfvgPair(
            PairedIfvgArtifactReference(v2=evidence.dataset.reference, v3=reference),
            evidence.dataset,
            context_artifact,
        )
        validate_exact_context_links(self.context_tables, core_tables=self.v2_tables)
        full = build_candidate_feature_view(self.pair)
        self._candidate_view = scope_research_candidate_view(
            full,
            self.v2_tables[RecordTable.ENTRY_CANDIDATE],
            self.v2_tables[RecordTable.ELIGIBLE_DECISION],
            self.subject,
        )
        return self

    @property
    def candidate_view(self):
        self.prepare()
        return self._candidate_view

    def candidate_view_source(self):
        return self.candidate_view

    @property
    def bars_1m(self):
        self.prepare()
        return self._loaded.bars_1m.copy()

    @property
    def label_source_reference(self):
        self.prepare()
        return {
            "artifact_id": self._loaded.envelope.research_context_companion_id,
            "manifest_payload_sha256": self._loaded.manifest_payload_sha256,
            "sha256": self._loaded.forward_bars_sha256,
            "path": str(self._loaded.directory / "forward_bars.parquet"),
        }

    def label_builder(self, view):
        self.prepare()
        if view.view_id != self._candidate_view.view_id:
            raise ValueError("label builder received a view from another subject")
        return build_configured_r_labels(
            view.frame,
            self.bars_1m,
            r_multiple=float(self.subject.section_mapping["tp_r_multiple"]),
            cost_points=self.cost_points,
            cutoff_ts_utc=self.subject.cutoff_ts_utc,
            label_source_id=self._loaded.envelope.research_context_companion_id,
        )

    def mbp1_evidence_source(self):
        from .research_mbp1 import (  # noqa: PLC0415
            build_research_mbp1_evidence,
            research_mbp1_contract,
        )

        if self.mbp1_evidence:
            return (
                self.mbp1_evidence["source_envelope"],
                self.mbp1_evidence["events_by_day"],
                self.mbp1_evidence["anchors"],
            )
        return build_research_mbp1_evidence(
            self,
            contract=self.mbp1_contract or research_mbp1_contract(),
            coverage_evidence=self.mbp1_coverage_evidence,
        )
