"""The finite, owner-specified IFSM Task B plan (decisions 20 and 21).

This module records explicit rows, rather than expanding their values into a
Cartesian search. It grants no permission and never reads market partitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .identities import SHA256_PATTERN, FrozenContract, ImmutableMap

STUDY_NAME = "menthorq_level_rules_b1"
TASK_B_WARMUP_POLICY = "ifsm_task_b_ten_weekdays_v1"
TASK_B_WARMUP_DATES = (
    "2025-06-02", "2025-06-03", "2025-06-04", "2025-06-05", "2025-06-06",
    "2025-06-09", "2025-06-10", "2025-06-11", "2025-06-12", "2025-06-13",
)
TASK_B_FUNDED_DECISION_KEY = "funded:exact_account_plan"


def task_b_funded_dimensions(payload) -> tuple[str, ...]:
    values = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    scope = values.get("task_b_execution")
    if not scope:
        return ()
    return tuple("funded_account." + digest for digest in sorted(
        dict(scope["funded_profile_sha256"]).values()
    ))


def task_b_requirement_set(payload):
    from ..study.computation_path import ComputationPath
    from .authorization import derive_authorization_requirements

    values = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    return derive_authorization_requirements(
        "full_authorized_development",
        tuple(f"strategy_profile.{axis}" for axis in sorted(dict(values["axes"])))
        + task_b_funded_dimensions(values),
        ComputationPath(
            full_strategy_replay=True, feature_materialization=False, label_recomputation=False,
            model_refit=False, model_gated_sequential_replay=False, cost_recomputation=True,
            prop_resimulation=True, bootstrap_resimulation=False, reuse_trade_stream_hash=False,
        ), (), (),
    )


class ExplicitConfiguration(FrozenContract):
    name: str = Field(min_length=1)
    axis_value_ids: ImmutableMap[str, str]


class TaskBEvidenceRef(FrozenContract):
    label: Literal["TASK_B.md", "owner_decisions_20_21"]
    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)


class TaskBExecutionScope(FrozenContract):
    policy_id: Literal["ifsm_task_b_decisions20_21_v1"] = "ifsm_task_b_decisions20_21_v1"
    study_name: Literal["menthorq_level_rules_b1"] = STUDY_NAME
    prepared_store_registry_paths: tuple[str, ...] = Field(min_length=2, max_length=2)
    artifact_provenance_dates: tuple[str, ...]
    owner_evidence: tuple[TaskBEvidenceRef, ...] = Field(min_length=2, max_length=2)
    menthorq_source_file_sha256: ImmutableMap[str, str]
    preparation_catalog_sha256: ImmutableMap[str, str]
    funded_profile_sha256: ImmutableMap[str, str]
    processing_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    funded_instrument: Literal["mini"] = "mini"
    funded_quantity: Literal[1] = 1
    cash_attribution: Literal["configuration_firm_only_cash_event_month"] = (
        "configuration_firm_only_cash_event_month"
    )
    roll_policy: Literal["report_all_days_no_exclusions"] = "report_all_days_no_exclusions"

    @model_validator(mode="after")
    def _bounded(self):
        if {ref.label for ref in self.owner_evidence} != {"TASK_B.md", "owner_decisions_20_21"}:
            raise ValueError("Task B binds TASK_B.md and owner decisions 20/21")
        if set(self.funded_profile_sha256) != {"takeprofittrader", "myfundedfutures"}:
            raise ValueError("Task B uses every existing funded firm configuration")
        for digest in self.funded_profile_sha256.values():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Task B funded profiles require content hashes")
        if len(set(self.prepared_store_registry_paths)) != 2:
            raise ValueError("Task B requires separate 2025 and 2026 store registrations")
        if len(self.menthorq_source_file_sha256) != 2 or len(self.preparation_catalog_sha256) != 2:
            raise ValueError("Task B binds its two original MenthorQ sources and two catalogs")
        for digest in (*self.menthorq_source_file_sha256.values(),
                       *self.preparation_catalog_sha256.values()):
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("Task B original runtime sources require content hashes")
        if (
            tuple(sorted(set(self.artifact_provenance_dates))) != self.artifact_provenance_dates
            or any(day > "2026-06-10" for day in self.artifact_provenance_dates)
        ):
            raise ValueError("Task B registered source dates must be unique and before the cutoff")
        return self


def task_b_configurations() -> tuple[ExplicitConfiguration, ...]:
    base = {
        "holding_policy": "holding_policy.scheduled_daily_close_v1",
        "menthorq_context_version": "menthorq_context_version.none",
        "regime_gate_policy": "regime_gate_policy.off",
        "regime_unknown_policy": "regime_unknown_policy.allow",
        "nearest_support_gex1_block": "nearest_support_gex1_block.false",
        "nearest_support_universe": "nearest_support_universe.all_19",
        "enabled_entry_sessions": "enabled_entry_sessions.asia-london-ny",
    }
    recipes = (
        ("baseline", None, None, False, None),
        ("context_on", None, None, False, None),
        ("regime_positive", "positive_only", None, False, None),
        ("regime_negative", "negative_only", None, False, None),
        ("gex1_block", None, None, True, None),
        ("slot_s1", None, "slot_s1_0830_1000", False, None),
        ("slot_s2", None, "slot_s2_1000_1200", False, None),
        ("slot_s3", None, "slot_s3_1200_1330", False, None),
        ("slot_s4", None, "slot_s4_1330_1510", False, None),
        ("midsession", None, "midsession_1000_1330", False, None),
        ("negative_s1", "negative_only", "slot_s1_0830_1000", False, None),
        ("positive_mid_gex1", "positive_only", "midsession_1000_1330", True, None),
        ("ny_only", None, None, False, "ny"),
    )
    rows = []
    for name, regime, slot, block, session in recipes:
        selected = dict(base)
        if name not in {"baseline", "ny_only"}:
            selected["menthorq_context_version"] = "menthorq_context_version.eod_v1"
        if regime:
            selected["regime_gate_policy"] = "regime_gate_policy." + regime
        if block:
            selected["nearest_support_gex1_block"] = "nearest_support_gex1_block.true"
            selected["nearest_support_universe"] = "nearest_support_universe.studied_8"
        if slot or session:
            selected["enabled_entry_sessions"] = "enabled_entry_sessions." + (slot or session)
        rows.append(ExplicitConfiguration(name=name, axis_value_ids=selected))
    return tuple(rows)


def explicit_configuration_rows(payload) -> tuple[tuple[str | None, dict[str, str]], ...]:
    """The shared review/UI/runner enumerator; legacy requests keep product order."""
    from itertools import product

    values = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    explicit = values.get("explicit_configurations", ())
    if explicit:
        return tuple((row["name"], dict(row["axis_value_ids"])) for row in explicit)
    axes = sorted(dict(values["axes"]).items())
    return tuple(
        (None, dict(zip((key for key, _ in axes), combo, strict=True)))
        for combo in product(*(ids for _, ids in axes))
    )


def validate_task_b_request(payload, *, verify_files: bool = False) -> None:
    from .identities import canonical_contract_sha256
    values = payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload
    scope_raw = values.get("task_b_execution")
    explicit = values.get("explicit_configurations", ())
    if not scope_raw:
        if explicit:
            raise ValueError("explicit configuration rows require exact Task B authority")
        if values.get("date_policy", {}).get("warmup_policy_id"):
            raise ValueError("Task B weekday warmup requires exact Task B authority")
        return
    scope = TaskBExecutionScope.model_validate(scope_raw)
    rows = tuple(ExplicitConfiguration.model_validate(row) for row in explicit)
    if rows != task_b_configurations() or values["max_child_count"] != 13:
        raise ValueError("Task B plan must contain exactly the thirteen owner-specified rows")
    expected_axes = {
        axis: tuple(dict.fromkeys(row.axis_value_ids[axis] for row in rows))
        for axis in rows[0].axis_value_ids
    }
    if {key: tuple(ids) for key, ids in dict(values["axes"]).items()} != expected_axes:
        raise ValueError("Task B aggregate axes differ from its explicit configurations")
    policy = values["date_policy"]
    dates = tuple(policy["replay_dates"])
    if (
        policy.get("warmup_policy_id") != TASK_B_WARMUP_POLICY
        or tuple(policy["warmup_dates"]) != TASK_B_WARMUP_DATES
        or dates[:10] != TASK_B_WARMUP_DATES
        or not dates[10:] or dates[10] != "2025-06-16" or dates[-1] != "2026-06-10"
        or any(day < "2025-06-16" or day > "2026-06-10" for day in dates[10:])
    ):
        raise ValueError("Task B dates differ from its exact weekday warmup and evaluation range")
    if values["baseline_profile_name"] != "ifvg_v2_doc_default_fresh_static_1r":
        raise ValueError("Task B requires the owner-specified baseline profile")
    if tuple(values["simulation_protocol"]["modes"]) != ("historical_closed_trade",):
        raise ValueError(
            "Task B permits its single historical funded replay and no resampling modes"
        )
    if not set(dates) <= set(scope.artifact_provenance_dates):
        raise ValueError("Task B registered source provenance must cover its replay dates")
    if values["cost_policy"] != {
        "cost_points_round_turn": 0.514, "dollars_per_point": 20.0, "tick_size": 0.25
    }:
        raise ValueError("Task B preserves the existing NQ round-trip cost policy")
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    if dict(scope.funded_profile_sha256) != {
        key: canonical_contract_sha256(profile) for key, profile in FIRM_PROFILES.items()
    } or scope.processing_policy_sha256 != canonical_contract_sha256(TWO_BUSINESS_DAYS_FED_1600):
        raise ValueError("Task B funded economics differ from the existing harness configuration")
    if verify_files:
        from ..manifest import file_sha256
        from ..menthorq_levels import load_menthorq_levels
        from ..prepared_store import load_prepared_store
        from ..research_period import EXTENDED_HISTORY_CLOSURES
        from .trading_calendar import logical_trading_days

        for ref in scope.owner_evidence:
            if file_sha256(Path(ref.path)) != ref.sha256:
                raise PermissionError("Task B owner authorization evidence changed")
        registrations = tuple(
            load_prepared_store(Path(path)) for path in scope.prepared_store_registry_paths
        )
        catalogs = {
            str(Path(registration.definition["catalog_path"]).resolve()): file_sha256(
                Path(registration.definition["catalog_path"])
            ) for registration in registrations
        }
        if catalogs != dict(scope.preparation_catalog_sha256):
            raise PermissionError("Task B preparation catalog contents changed after review")
        actual_sources = dict(load_menthorq_levels().source_file_sha256)
        if actual_sources != dict(scope.menthorq_source_file_sha256):
            raise PermissionError("Task B original MenthorQ source contents changed after review")
        provenance = tuple(sorted({
            day for registration in registrations for day in registration.source_dates
        }))
        if provenance != scope.artifact_provenance_dates:
            raise PermissionError("Task B registered source provenance changed")
        logical = set(logical_trading_days("2025-06-16", "2026-06-10")) - set(
            EXTENDED_HISTORY_CLOSURES
        )
        owned = tuple(sorted({
            day for registration in registrations for day in registration.owned_dates
            if day in TASK_B_WARMUP_DATES or day in logical
        }))
        if owned != dates:
            raise PermissionError(
                "Task B replay dates differ from the existing registered logical calendar"
            )


def task_b_axes() -> dict[str, tuple[str, ...]]:
    rows = task_b_configurations()
    return {
        axis: tuple(dict.fromkeys(row.axis_value_ids[axis] for row in rows))
        for axis in rows[0].axis_value_ids
    }


def task_b_request_fields(
    template: dict, *, replay_dates: tuple[str, ...], registry_paths: tuple[Path, ...],
    owner_evidence: tuple[TaskBEvidenceRef, ...],
    artifact_provenance_dates: tuple[str, ...] | None = None,
    menthorq_source_file_sha256: dict[str, str] | None = None,
    preparation_catalog_sha256: dict[str, str] | None = None,
) -> dict:
    """Bind the finite table into existing wizard terms without new economics."""
    from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600
    from alpha_lab.propsim.funded.profiles import FIRM_PROFILES

    from ..profiles import resolve_profile_config
    from .axis_registry import registry_sha256
    from .identities import canonical_contract_sha256
    from .strategy_approval import charter_intent

    fields = charter_intent(template)
    if menthorq_source_file_sha256 is None:
        from ..menthorq_levels import load_menthorq_levels

        menthorq_source_file_sha256 = dict(load_menthorq_levels().source_file_sha256)
    if preparation_catalog_sha256 is None:
        from ..manifest import file_sha256
        from ..prepared_store import load_prepared_store

        catalogs = tuple(Path(load_prepared_store(path).definition["catalog_path"])
                         for path in registry_paths)
        preparation_catalog_sha256 = {str(path.resolve()): file_sha256(path) for path in catalogs}
    if artifact_provenance_dates is None:
        from ..prepared_store import load_prepared_store

        artifact_provenance_dates = tuple(sorted({
            day for path in registry_paths for day in load_prepared_store(path).source_dates
        }))
    fields.update(
        search_mode="fsm_config_search",
        baseline_profile_name="ifvg_v2_doc_default_fresh_static_1r",
        baseline_section_config_hash=resolve_profile_config(
            {"profile_name": "ifvg_v2_doc_default_fresh_static_1r"}
        ).section_config_hash,
        axes=task_b_axes(), locked_invariants_registry_sha256=registry_sha256(),
        max_child_count=13,
        explicit_configurations=task_b_configurations(),
        task_b_execution=TaskBExecutionScope(
            prepared_store_registry_paths=tuple(str(p.resolve()) for p in registry_paths),
            artifact_provenance_dates=artifact_provenance_dates,
            owner_evidence=owner_evidence,
            menthorq_source_file_sha256=menthorq_source_file_sha256,
            preparation_catalog_sha256=preparation_catalog_sha256,
            funded_profile_sha256={
                key: canonical_contract_sha256(profile) for key, profile in FIRM_PROFILES.items()
            },
            processing_policy_sha256=canonical_contract_sha256(TWO_BUSINESS_DAYS_FED_1600),
        ),
        date_policy={
            "replay_dates": replay_dates, "warmup_dates": TASK_B_WARMUP_DATES,
            "development_cutoff_utc": "2026-06-10T21:00:00Z",
            "access_policy_id": "development_explicit_dates_before_path_v2",
            "warmup_policy_id": TASK_B_WARMUP_POLICY,
        },
    )
    return charter_intent(fields)
