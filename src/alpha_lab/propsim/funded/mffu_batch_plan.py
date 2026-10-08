"""Frozen 64-intent MyFundedFutures batch plan.

The handoff matrix is a declaration of research intent, not a worker plan.  This
module resolves every row against the *verified repaired* six-configuration
plan, validates the final Strategy-Core section, and saves one content-addressed
plan in the existing funded-comparison plan store.  It never records approval or
starts a financial worker.  Workers must call :func:`verify_mffu_batch_plan`
before dispatch and on resume; a source, context table, or handoff mutation
invalidates the plan/checkpoint binding.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections import Counter
from itertools import product
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field, model_validator

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
    ImmutableMap,
    canonical_contract_sha256,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    has_envelope,
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.agents.data_infra.ifvg.search.task_b import TaskBExecutionScope
from alpha_lab.propsim.funded.clock import TWO_BUSINESS_DAYS_FED_1600, ProcessingClockPolicy
from alpha_lab.propsim.funded.comparison_plan import (
    PLAN_STORE,
    RESULT_STORE,
    CoreSourceRef,
    ExecutionModelRef,
    FundedComparisonResultEnvelope,
)
from alpha_lab.propsim.funded.full_range_batch import (
    FullRangeBatchPlanEnvelope,
    FullRangeSourceRef,
)
from alpha_lab.propsim.funded.plan import OwnerDecisionRef
from alpha_lab.propsim.funded.position_walk import FEE_ROUNDING_POLICY_ID
from alpha_lab.propsim.funded.profiles import MYFUNDEDFUTURES_PROFILE, FundedFirmProfile
from alpha_lab.propsim.funded.result import result_sha256

PLAN_SCHEMA = "ifsm_mffu_context_64_batch_plan_v1"
HANDOFF_ROOT = "docs/ifsm-mffu-context-batch-v01/"
REFERENCE_RESULT_ID = "7278632babf01b084c43ddb6df77332f15d8b16702c7082390553b856053d4ad"
CONTEXT_ROOT = "MenthorQ_Research_Data_v02/"
CONTEXT_TABLES = (
    "data/canonical/end_of_day/metrics/total_gamma_by_report_date.csv",
    "data/canonical/end_of_day/levels/level_sets.csv",
    "data/canonical/end_of_day/levels/gamma_long.csv",
    "data/canonical/end_of_day/levels/gamma_prices_wide.csv",
    "data/research/end_of_day/nominal_schedule_index.csv",
)
_SOURCE_MEMBERS = (
    "CONFIGURATION_MATRIX.json",
    "POLICY_DEFINITIONS.json",
    "REFERENCE_SCOPE.json",
    "DATA_INTEGRATION.md",
    "STUDY_SPEC.md",
    "VALIDATION_AND_OUTPUTS.md",
    "COMPARISON_PAIRS.csv",
)
_ENTRY_POLICIES = {
    "F0": "off",
    "FE": "skip_early_positive_v1",
    "FL": "skip_positive_london_v1",
    "FEL": "skip_early_positive_and_london_v1",
}
_EXIT_POLICIES = {
    "XP": "scale_out_half_breakeven_hold_to_close_v1",
    "XF": "fixed_target_v1",
    "XG": "gamma_conditional_1r_v1",
    "XE": "early_positive_whole_1r_v1",
}
_GEOMETRY_POLICIES = {
    "G0": "fixed_v1",
    "G05": "implied_move_005_v1",
    "G075": "implied_move_0075_v1",
    "G10": "implied_move_010_v1",
}
_OVERHEAD_POLICIES = {
    "O0": "off",
    "O8": "nearest_studied_8_above_target_gex_gt_300000_v1",
}
_FAMILY_COUNTS = {"core_48": 48, "sizing": 4, "overhead": 4, "geometry": 6,
                  "state_exit": 2}
REPO_ROOT = Path(__file__).resolve().parents[4]
REQUIRED_RUNTIME_SOURCES = (
    # Plan resolution, registered input decoding, and source-bound approvals.
    "src/alpha_lab/agents/data_infra/ifvg/config.py",
    "src/alpha_lab/agents/data_infra/ifvg/context_contracts.py",
    "src/alpha_lab/agents/data_infra/ifvg/context_experiment_contracts.py",
    "src/alpha_lab/agents/data_infra/ifvg/context_schemas.py",
    "src/alpha_lab/agents/data_infra/ifvg/contracts.py",
    "src/alpha_lab/agents/data_infra/ifvg/data_access.py",
    "src/alpha_lab/agents/data_infra/ifvg/day_artifacts.py",
    "src/alpha_lab/agents/data_infra/ifvg/development_access.py",
    "src/alpha_lab/agents/data_infra/ifvg/manifest.py",
    "src/alpha_lab/agents/data_infra/ifvg/prepared_store.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/authorization.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/axis_registry.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/charter.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/entry_activity.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/file_mutex.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/identities.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/store.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/store_namespace.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/task_b.py",
    "src/alpha_lab/agents/data_infra/ifvg/search/task_b_execution.py",
    # Worker dispatch, print selection, account lifecycle, and result creation.
    "src/alpha_lab/propsim/funded/mffu_batch_plan.py",
    "src/alpha_lab/propsim/funded/full_range_batch.py",
    "src/alpha_lab/propsim/funded/campaign.py",
    "src/alpha_lab/propsim/funded/comparison_describe.py",
    "src/alpha_lab/propsim/funded/comparison_plan.py",
    "src/alpha_lab/propsim/funded/comparison_run.py",
    "src/alpha_lab/propsim/funded/comparison_source.py",
    "src/alpha_lab/propsim/funded/comparison_study.py",
    "src/alpha_lab/propsim/funded/core_identity.py",
    "src/alpha_lab/propsim/funded/profiles.py",
    "src/alpha_lab/propsim/funded/clock.py",
    "src/alpha_lab/propsim/funded/instance.py",
    "src/alpha_lab/propsim/funded/paths.py",
    "src/alpha_lab/propsim/funded/print_minutes.py",
    "src/alpha_lab/propsim/funded/price_evidence.py",
    "src/alpha_lab/propsim/funded/positions.py",
    "src/alpha_lab/propsim/funded/result.py",
    "src/alpha_lab/propsim/funded/strategy_driver.py",
    "src/alpha_lab/propsim/funded/pair_engine.py",
    "src/alpha_lab/propsim/funded/pair_ledger.py",
    "src/alpha_lab/propsim/funded/position_walk.py",
    "src/alpha_lab/propsim/funded/mffu_batch_driver.py",
    "src/alpha_lab/propsim/funded/mffu_batch_run.py",
    "src/alpha_lab/propsim/funded/mffu_batch_reuse.py",
    "src/alpha_lab/propsim/funded/mffu_batch_analysis.py",
    "src/alpha_lab/propsim/funded/mffu_batch_review.py",
    "src/alpha_lab/propsim/funded/full_range_reporting.py",
    "src/alpha_lab/propsim/funded/comparison_result.py",
    "src/alpha_lab/propsim/funded/comparison_runner.py",
    "src/alpha_lab/propsim/funded/runner.py",
    "src/alpha_lab/propsim/funded/sources.py",
    "src/alpha_lab/propsim/funded/plan.py",
    "src/alpha_lab/agents/data_infra/ifvg/funded_comparison_review.py",
    "src/alpha_lab/agents/data_infra/ifvg/menthorq_levels.py",
    "src/alpha_lab/agents/data_infra/ifvg/menthorq_asof.py",
    "scripts/ifvg_mffu_batch_job.py",
)


def _runtime_source_paths(extra_files: tuple[Path, ...]) -> tuple[str, ...]:
    """Bind the concrete task replay and financial code, plus caller sources."""
    paths = {str(Path(path).resolve()) for path in extra_files}
    paths.update(str((REPO_ROOT / relative).resolve())
                 for relative in REQUIRED_RUNTIME_SOURCES)
    return tuple(sorted(paths))


def _runtime_source_root(bound_sources) -> str:
    """Validate a frozen source root independently of the dashboard reader."""
    runtime_roots = set()
    for relative in REQUIRED_RUNTIME_SOURCES:
        suffix = "/" + relative
        paths = [Path(path).as_posix() for path in bound_sources
                 if Path(path).as_posix().endswith(suffix)]
        if len(paths) != 1:
            raise ValueError(f"required economic source is not bound: {relative}")
        runtime_roots.add(paths[0][:-len(suffix)])
    if len(runtime_roots) != 1:
        raise ValueError("required economic sources must share one frozen runtime root")
    return runtime_roots.pop()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _context_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: _sha_bytes(archive.read(CONTEXT_ROOT + name))
            for name in CONTEXT_TABLES
        }


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class MffuVariantRef(FrozenContract):
    """One declared intent and the complete section sent to its future worker."""

    variant_id: str
    batch_id: str
    name: str
    display_name: str
    family: str
    intent_json: str
    intent_sha256: str = Field(pattern=SHA256_PATTERN)
    base_reference: str
    base_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    effective_section_json: str
    effective_section_config_hash: str = Field(pattern=SHA256_PATTERN)
    effective_behavior_hash: str = Field(pattern=SHA256_PATTERN)
    quantity_policy: Literal["Q10", "Q6", "QG"]
    quantity: int
    possible_quantities: tuple[int, ...]
    instrument: Literal["micro"] = "micro"
    cost_per_contract_mills: Literal[514] = 514
    fee_rounding_policy: Literal["per_fill_total_round_half_up_cent_v1"] = (
        FEE_ROUNDING_POLICY_ID
    )
    firm_key: Literal["myfundedfutures"] = "myfundedfutures"

    @property
    def exit_policy(self) -> str:
        return str(json.loads(self.effective_section_json)["exit_policy"])

    @model_validator(mode="after")
    def _exact(self):
        if self.variant_id != self.batch_id or self.variant_id != self.name:
            raise ValueError("one intent must retain one stable worker identity")
        if _sha_bytes(self.intent_json.encode("utf-8")) != self.intent_sha256:
            raise ValueError("intent hash differs from the persisted row")
        if self.possible_quantities != {"Q10": (10,), "Q6": (6,), "QG": (6, 10)}[
            self.quantity_policy
        ]:
            raise ValueError("quantity options differ from the declared sizing rule")
        if any(quantity % 2 for quantity in self.possible_quantities):
            raise ValueError("a half exit requires an even whole-contract quantity")
        if self.quantity != max(self.possible_quantities):
            raise ValueError("static quantity records the maximum possible entry size")
        return self


class MffuBatchPlanPayload(FrozenContract):
    plan_schema: Literal["ifsm_mffu_context_64_batch_plan_v1"] = PLAN_SCHEMA
    mode: Literal["single_account_configuration_comparison"] = (
        "single_account_configuration_comparison"
    )
    purpose: Literal["historical_comparison"] = "historical_comparison"
    question: str = ("Which combination of IFSM participation, schedule, market context "
                     "and profit-taking increases MyFundedFutures received cash after all "
                     "account costs, and which components contribute?")
    reference_result_id: str = Field(pattern=SHA256_PATTERN)
    reference_plan_id: str = Field(pattern=SHA256_PATTERN)
    reference_plan_envelope_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_store_root: str
    source: FullRangeSourceRef
    calendar_sha256: str = Field(pattern=SHA256_PATTERN)
    task_b_store_root: str
    task_b_plan_id: str = Field(pattern=SHA256_PATTERN)
    task_b_scope: TaskBExecutionScope
    prepared_registration_ids: ImmutableMap[str, str]
    context_archive_path: str
    context_archive_sha256: str = Field(pattern=SHA256_PATTERN)
    context_policy_version: Literal["mq_eod_asof_nominal_2200_chicago_v01"] = (
        "mq_eod_asof_nominal_2200_chicago_v01"
    )
    fee_rounding_policy: Literal["per_fill_total_round_half_up_cent_v1"] = (
        FEE_ROUNDING_POLICY_ID
    )
    handoff_zip: str
    handoff_zip_sha256: str = Field(pattern=SHA256_PATTERN)
    source_member_sha256: ImmutableMap[str, str]
    input_archive_sha256: ImmutableMap[str, str]
    context_table_sha256: ImmutableMap[str, str]
    runtime_source_file_sha256: ImmutableMap[str, str]
    core_root: str
    core_source: CoreSourceRef
    variants: tuple[MffuVariantRef, ...] = Field(min_length=64, max_length=64)
    firm_profiles: tuple[FundedFirmProfile, ...] = (MYFUNDEDFUTURES_PROFILE,)
    processing: ProcessingClockPolicy = TWO_BUSINESS_DAYS_FED_1600
    execution_model: ExecutionModelRef = ExecutionModelRef()
    owner_decisions: tuple[OwnerDecisionRef, ...]
    limitations: tuple[str, ...]

    @model_validator(mode="after")
    def _exact(self):
        if tuple(v.variant_id for v in self.variants) != tuple(
            f"MCB{index:03d}" for index in range(1, 65)
        ):
            raise ValueError("the frozen plan must preserve all 64 intents in matrix order")
        if dict(Counter(v.family for v in self.variants)) != _FAMILY_COUNTS:
            raise ValueError("matrix family membership changed")
        hashes = [v.effective_behavior_hash for v in self.variants]
        if len(set(hashes)) != 64:
            raise ValueError("distinct declared behaviors collapsed into one identity")
        if any(v.fee_rounding_policy != self.fee_rounding_policy for v in self.variants):
            raise ValueError("variant fee policy differs from the frozen batch")
        if self.reference_result_id != REFERENCE_RESULT_ID:
            raise ValueError("the source is not the completed repaired reference")
        if len(self.source.warmup_dates) != 10 or len(self.source.evaluation_dates) != 253:
            raise ValueError("the plan does not use the exact 10+253 date scope")
        if self.source.cutoff_utc != "2026-06-10T21:00:00Z":
            raise ValueError("the historical terminal cutoff changed")
        if self.source.task_b_plan_id != self.task_b_plan_id:
            raise ValueError("Task B source and bound execution scope disagree")
        if tuple(self.firm_profiles) != (MYFUNDEDFUTURES_PROFILE,):
            raise ValueError("this batch includes only the saved MyFundedFutures profile")
        if canonical_contract_sha256(self.processing) != canonical_contract_sha256(
            TWO_BUSINESS_DAYS_FED_1600
        ):
            raise ValueError("the payout processing clock changed")
        _runtime_source_root(self.runtime_source_file_sha256)
        return self

    @property
    def configurations(self) -> tuple[MffuVariantRef, ...]:
        """Match the existing comparison worker's configuration access seam."""
        return self.variants


class MffuBatchPlanEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_plan_id"

    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: MffuBatchPlanPayload


def _read_handoff(handoff_zip: Path) -> tuple[dict, dict, dict, dict[str, str],
                                               dict[str, str], dict[str, str]]:
    """Verify all packaged bytes before interpreting the exact matrix members."""
    with zipfile.ZipFile(handoff_zip) as archive:
        manifest = json.loads(archive.read(HANDOFF_ROOT + "PACKAGE_MANIFEST.json"))
        if manifest.get("schema") != "ifsm_mffu_context_batch_handoff_manifest_v1":
            raise ValueError("unexpected handoff manifest")
        listed = manifest["files"]
        actual = {name.removeprefix(HANDOFF_ROOT) for name in archive.namelist()
                  if name.startswith(HANDOFF_ROOT)
                  and name != HANDOFF_ROOT + "PACKAGE_MANIFEST.json"}
        if actual != set(listed):
            raise ValueError("handoff members differ from its manifest")
        for name, receipt in listed.items():
            body = archive.read(HANDOFF_ROOT + name)
            if len(body) != receipt["bytes"] or _sha_bytes(body) != receipt["sha256"]:
                raise ValueError(f"handoff payload changed: {name}")
        members = {name: listed[name]["sha256"] for name in _SOURCE_MEMBERS}
        matrix = json.loads(archive.read(HANDOFF_ROOT + "CONFIGURATION_MATRIX.json"))
        policies = json.loads(archive.read(HANDOFF_ROOT + "POLICY_DEFINITIONS.json"))
        scope = json.loads(archive.read(HANDOFF_ROOT + "REFERENCE_SCOPE.json"))
        archives = json.loads(archive.read(HANDOFF_ROOT + "INPUT_ARCHIVES.json"))
        input_hashes = {row["path"]: row["sha256"] for row in archives["archives"]}
        if any(listed[name]["sha256"] != digest
               for name, digest in input_hashes.items()):
            raise ValueError("input archive receipts disagree with the package manifest")
        context_archive = archive.read(HANDOFF_ROOT + "inputs/MenthorQ_Research_Data_v02.zip")
        with zipfile.ZipFile(io.BytesIO(context_archive)) as context:
            table_hashes = {
                name: _sha_bytes(context.read(CONTEXT_ROOT + name))
                for name in CONTEXT_TABLES
            }
    return matrix, policies, scope, members, input_hashes, table_hashes


def _verify_reference_context_member(handoff_zip: Path, scope: dict) -> None:
    with zipfile.ZipFile(handoff_zip) as handoff:
        archive_bytes = handoff.read(
            HANDOFF_ROOT + "references/funded_comparison_7278632babf01b08_export_v1(1).zip"
        )
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as reference:
        matches = [name for name in reference.namelist() if name.endswith("/run_context.json")]
        if len(matches) != 1 or _sha_bytes(reference.read(matches[0])) != scope[
            "source_member_run_context_sha256"
        ]:
            raise ValueError("the delivered repaired run context identity differs")


def _validate_matrix(matrix: dict, policies: dict) -> tuple[dict, ...]:
    if (matrix.get("schema") != "ifsm_64_candidate_intent_v1"
            or matrix.get("count") != 64 or matrix.get("core_count") != 48
            or matrix.get("targeted_count") != 16):
        raise ValueError("unexpected 64-intent matrix version or count")
    if policies.get("schema") != "ifsm_mffu_context_research_policy_intent_v1":
        raise ValueError("unexpected policy definition version")
    if (policies["day_cap"]["U"] is not None
            or policies["day_cap"]["D1"] != 1
            or policies["data"]["batch_adapter"]
            != "mq_eod_asof_nominal_2200_chicago_v01"
            or policies["data"]["nominal_publication_local"] != "22:00"
            or policies["constants"]["firm"] != "myfundedfutures"
            or policies["constants"]["micro_cost_per_contract_per_fill_usd"] != "0.514"
            or policies["overhead"]["O8"]["threshold_vendor_units"] != 300000
            or policies["overhead"]["O8"]["threshold_comparator"] != "strictly_greater"
            or policies["geometry"]["G05"]["implied_move_fraction"] != "0.05"
            or policies["geometry"]["G075"]["implied_move_fraction"] != "0.075"
            or policies["geometry"]["G10"]["implied_move_fraction"] != "0.10"
            or policies["sizing"]["Q6"]["positive"] != 6
            or policies["sizing"]["QG"]["positive"] != 6
            or policies["sizing"]["QG"]["negative"] != 10):
        raise ValueError("policy definitions differ from the implemented resolution")
    rows = tuple(matrix["variants"])
    if len(rows) != 64 or len({_json(row) for row in rows}) != 64:
        raise ValueError("matrix rows are absent or duplicated")
    for index, row in enumerate(rows, start=1):
        if row["variant_id"] != f"MCB{index:03d}" or row["firm"] != "myfundedfutures":
            raise ValueError("matrix order, identity or firm changed")
        for name, values in (
            ("schedule", {"S0", "S1"}), ("daily_cap", {"U", "D1"}),
            ("entry_context", set(_ENTRY_POLICIES)), ("exit", set(_EXIT_POLICIES)),
            ("sizing", {"Q10", "Q6", "QG"}),
            ("overhead", set(_OVERHEAD_POLICIES)),
            ("geometry", set(_GEOMETRY_POLICIES)),
        ):
            if row[name] not in values:
                raise ValueError(f"unknown declared {name}: {row[name]}")
        if row["base_reference"] != policies["schedule"][row["schedule"]][
            "base_configuration"
        ]:
            raise ValueError("matrix row does not use its declared schedule base")
    core = rows[:48]
    expected = set(product(("S0", "S1"), ("U", "D1"), ("F0", "FE", "FL", "FEL"),
                           ("XP", "XF", "XG")))
    observed = {(r["schedule"], r["daily_cap"], r["entry_context"], r["exit"])
                for r in core}
    if observed != expected or any(
        r["family"] != "core_48" or (r["sizing"], r["overhead"], r["geometry"])
        != ("Q10", "O0", "G0") for r in core
    ):
        raise ValueError("the 48 core rows are not the declared full cross-product")
    if dict(Counter(r["family"] for r in rows)) != _FAMILY_COUNTS:
        raise ValueError("the targeted family counts changed")
    targeted = {
        "sizing": {(s, "U", "F0", "XP", q, "O0", "G0")
                   for s, q in product(("S0", "S1"), ("Q6", "QG"))},
        "overhead": {(s, cap, "F0", "XP", "Q10", "O8", "G0")
                     for s, cap in product(("S0", "S1"), ("U", "D1"))},
        "geometry": {(s, "U", "F0", "XP", "Q10", "O0", g)
                     for s, g in product(("S0", "S1"), ("G05", "G075", "G10"))},
        "state_exit": {(s, "U", "F0", "XE", "Q10", "O0", "G0")
                       for s in ("S0", "S1")},
    }
    for family, expected_rows in targeted.items():
        actual_rows = {
            tuple(row[key] for key in ("schedule", "daily_cap", "entry_context", "exit",
                                       "sizing", "overhead", "geometry"))
            for row in rows if row["family"] == family
        }
        if actual_rows != expected_rows:
            raise ValueError(f"{family} rows differ from the predeclared matrix")
    return rows


def _resolve_section(section: dict) -> tuple[dict, str]:
    """Require the final imported Core to accept and hash every new setting."""
    from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, ifvg_profile_hash

    from alpha_lab.agents.data_infra.ifvg.search.identities import canonicalize_section

    exact = canonicalize_section(IfvgSmcSection.model_validate(section))
    values = exact.model_dump(mode="json")
    for name in section:
        if name not in values:
            raise ValueError(f"Core silently dropped {name}")
    return values, ifvg_profile_hash(exact)


def _variant_from_intent(row: dict, base: dict) -> MffuVariantRef:
    section = json.loads(base["effective_section_json"])
    section["max_executed_trades_per_day"] = 1 if row["daily_cap"] == "D1" else None
    section["entry_context_policy"] = _ENTRY_POLICIES[row["entry_context"]]
    section["overhead_policy"] = _OVERHEAD_POLICIES[row["overhead"]]
    section["opposing_distance_policy"] = _GEOMETRY_POLICIES[row["geometry"]]
    section["exit_policy"] = _EXIT_POLICIES[row["exit"]]
    # The v02 selection contract also governs the F0/U/XP controls.  It makes
    # the cap's refused-setup semantics explicit while neutral v1 historical
    # profiles retain their original identity outside this new plan.
    section["ifsm_context_policy_version"] = "mq_eod_asof_nominal_2200_chicago_v01"
    resolved, section_hash = _resolve_section(section)
    for name, value in section.items():
        if name != "profile_name" and resolved.get(name) != value:
            raise ValueError(f"final Core changed declared {name} during resolution")
    quantities = {"Q10": (10,), "Q6": (6,), "QG": (6, 10)}[row["sizing"]]
    behavior = {
        "section_hash": section_hash,
        "schedule": row["schedule"],
        "daily_cap": row["daily_cap"],
        "entry_context": row["entry_context"],
        "exit": row["exit"],
        "sizing": row["sizing"],
        "overhead": row["overhead"],
        "geometry": row["geometry"],
        "instrument": "micro",
        "possible_quantities": quantities,
        "cost_per_contract_mills": 514,
        "fee_rounding_policy": FEE_ROUNDING_POLICY_ID,
        "firm": "myfundedfutures",
    }
    intent_json = _json(row)
    return MffuVariantRef(
        variant_id=row["variant_id"], batch_id=row["variant_id"],
        name=row["variant_id"], display_name=row["variant_id"],
        family=row["family"], intent_json=intent_json,
        intent_sha256=_sha_bytes(intent_json.encode("utf-8")),
        base_reference=row["base_reference"],
        base_section_config_hash=base["effective_section_config_hash"],
        effective_section_json=_json(resolved), effective_section_config_hash=section_hash,
        effective_behavior_hash=canonical_contract_sha256(behavior),
        quantity_policy=row["sizing"], quantity=max(quantities),
        possible_quantities=quantities,
    )


def build_mffu_batch_plan(
    *, handoff_zip: Path, reference_store_root: Path, context_archive_path: Path,
    runtime_source_files: tuple[Path, ...], core_root: Path,
) -> MffuBatchPlanEnvelope:
    """Resolve and freeze the *final* source; this creates no approval or run.

    ``REQUIRED_RUNTIME_SOURCES`` binds the task's registered-input, replay,
    account, and result paths. ``runtime_source_files`` adds caller sources.
    Freeze only after Core, context, worker, and Lab serialization are final.
    """
    handoff_zip = Path(handoff_zip).resolve()
    reference_store_root = Path(reference_store_root).resolve()
    context_archive_path = Path(context_archive_path).resolve()
    core_root = Path(core_root).resolve()
    matrix, policies, scope, members, archives, tables = _read_handoff(handoff_zip)
    rows = _validate_matrix(matrix, policies)
    if scope["source_result_id"] != REFERENCE_RESULT_ID:
        raise ValueError("handoff references a different result")
    if (scope["source_archive_sha256"]
            != archives["references/funded_comparison_7278632babf01b08_export_v1(1).zip"]):
        raise ValueError("reference archive identity differs from the handoff")
    _verify_reference_context_member(handoff_zip, scope)
    stored_profile = MYFUNDEDFUTURES_PROFILE.model_dump(mode="json")
    if any(stored_profile.get(key) != value for key, value in scope["firm_profile"].items()):
        raise ValueError("the saved MyFundedFutures profile differs from the handoff")
    plan_id = scope["source_plan_id_reference_only"]
    plan = load_verified_envelope(
        reference_store_root, PLAN_STORE, plan_id, FullRangeBatchPlanEnvelope
    )
    if (scope["calendar_hash_reference"] != plan.payload.calendar_sha256
            or scope["processing_clock"] != plan.payload.processing.model_dump(mode="json")
            or scope["execution_model_reference"]
            != plan.payload.execution_model.model_dump(mode="json")):
        raise ValueError("saved calendar, processing or execution model differs")
    result = load_verified_envelope(
        reference_store_root, RESULT_STORE, REFERENCE_RESULT_ID,
        FundedComparisonResultEnvelope,
    )
    result_path = reference_store_root / RESULT_STORE / REFERENCE_RESULT_ID / "result.json"
    saved_result = json.loads(result_path.read_text(encoding="utf-8"))
    if (result_sha256(saved_result) != result.payload.result_json_sha256
            or result.payload.funded_comparison_plan_id != plan_id
            or saved_result.get("configurations_completed") != 6
            or saved_result.get("configurations_requested") != 6
            or saved_result["run_identity"]["core_source"]["patch_sha256"]
            != plan.payload.core_source.patch_sha256):
        raise ValueError("the repaired reference result is incomplete or unverified")
    source = plan.payload.source
    if (tuple(scope["evaluation_dates"]) != source.evaluation_dates
            or tuple(scope["warmup_dates"]) != source.warmup_dates
            or scope["cutoff_utc"] != source.cutoff_utc
            or plan.payload.calendar_sha256 != saved_result["full_range_batch"]["plan"][
                "calendar_sha256"]):
        raise ValueError("the source scope or repaired calendar differs")
    bases = {row.name: row.model_dump(mode="json") for row in plan.payload.configurations}
    variants = tuple(_variant_from_intent(row, bases[row["base_reference"]]) for row in rows)
    context_archive_hash = _sha_file(context_archive_path)
    if context_archive_hash != archives["inputs/MenthorQ_Research_Data_v02.zip"]:
        raise ValueError("registered MenthorQ archive differs from the immutable handoff")
    context_hashes = _context_hashes(context_archive_path)
    if context_hashes != tables:
        raise ValueError("registered MenthorQ tables differ from the immutable handoff")
    runtime_hashes = {
        path: _sha_file(path)
        for path in _runtime_source_paths(runtime_source_files)
    }
    from alpha_lab.propsim.funded.core_identity import core_source_identity

    imported_core = core_source_identity()
    if Path(imported_core["root"]).resolve() != core_root:
        raise PermissionError("the imported Core root differs from the final task source")
    payload = MffuBatchPlanPayload(
        reference_result_id=REFERENCE_RESULT_ID, reference_plan_id=plan_id,
        reference_plan_envelope_sha256=_sha_file(
            reference_store_root / PLAN_STORE / plan_id / "envelope.json"
        ),
        reference_store_root=str(reference_store_root),
        source=source, calendar_sha256=plan.payload.calendar_sha256,
        task_b_store_root=plan.payload.task_b_store_root,
        task_b_plan_id=plan.payload.task_b_plan_id,
        task_b_scope=plan.payload.task_b_scope,
        prepared_registration_ids=plan.payload.prepared_registration_ids,
        context_archive_path=str(context_archive_path),
        context_archive_sha256=context_archive_hash,
        handoff_zip=str(handoff_zip), handoff_zip_sha256=_sha_file(handoff_zip),
        source_member_sha256=ImmutableMap(members),
        input_archive_sha256=ImmutableMap(archives),
        context_table_sha256=ImmutableMap(context_hashes),
        runtime_source_file_sha256=ImmutableMap(runtime_hashes),
        core_root=str(core_root),
        core_source=CoreSourceRef(
            base_commit=imported_core["base_commit"],
            branch=imported_core["branch"],
            patch_sha256=imported_core["patch_sha256"],
            description="Final task-owned IFSM MFFU context batch source",
        ),
        variants=variants,
        owner_decisions=plan.payload.owner_decisions,
        limitations=(*plan.payload.limitations,
                     "Each execution fill posts its $0.514-per-micro fee after "
                     "multiplying by the actual filled quantity, rounded to the "
                     "nearest cent with ROUND_HALF_UP. A six-micro entry and two "
                     "three-micro exits post $3.08 + $1.54 + $1.54 = $6.16.",
                     "MenthorQ EOD selection uses a nominal 10:00 PM Chicago release "
                     "convention; historical endpoint publication times were not measured.",
                     "The 186 in-scope supplied EOD level sets use the vendor NQ1! "
                     "continuous-front-month price coordinate. Task B uses source-selected "
                     "NQ contracts; exact vendor-versus-Task B roll parity was not "
                     "independently verified. No futures-basis adjustment is applied."),
    )
    return MffuBatchPlanEnvelope.from_payload(payload)


def verify_mffu_batch_plan(plan: MffuBatchPlanPayload) -> None:
    """Fail closed if any final source/input or any resolved intent changes."""
    if _sha_file(plan.handoff_zip) != plan.handoff_zip_sha256:
        raise PermissionError("the frozen handoff archive changed")
    if _sha_file(plan.context_archive_path) != plan.context_archive_sha256:
        raise PermissionError("the registered MenthorQ archive changed")
    if _context_hashes(Path(plan.context_archive_path)) != dict(plan.context_table_sha256):
        raise PermissionError("the registered MenthorQ table content changed")
    for path, expected in plan.runtime_source_file_sha256.items():
        if _sha_file(path) != expected:
            raise PermissionError(f"frozen source changed: {path}")
    from alpha_lab.propsim.funded.core_identity import core_source_identity

    actual = core_source_identity()
    if (Path(actual["root"]).resolve() != Path(plan.core_root).resolve()
            or actual["base_commit"] != plan.core_source.base_commit
            or actual["patch_sha256"] != plan.core_source.patch_sha256):
        raise PermissionError("the imported Core differs from the frozen batch source")
    reference_store = Path(plan.reference_store_root)
    reference_path = reference_store / PLAN_STORE / plan.reference_plan_id / "envelope.json"
    if _sha_file(reference_path) != plan.reference_plan_envelope_sha256:
        raise PermissionError("the repaired reference plan envelope changed")
    reference = load_verified_envelope(
        reference_store, PLAN_STORE, plan.reference_plan_id, FullRangeBatchPlanEnvelope
    )
    result = load_verified_envelope(
        reference_store, RESULT_STORE, plan.reference_result_id,
        FundedComparisonResultEnvelope,
    )
    saved_result = json.loads((reference_store / RESULT_STORE / plan.reference_result_id /
                               "result.json").read_text(encoding="utf-8"))
    if (result_sha256(saved_result) != result.payload.result_json_sha256
            or result.payload.funded_comparison_plan_id != plan.reference_plan_id
            or saved_result.get("configurations_completed") != 6):
        raise PermissionError("the repaired reference result changed")
    if (plan.source != reference.payload.source
            or plan.calendar_sha256 != reference.payload.calendar_sha256
            or plan.task_b_plan_id != reference.payload.task_b_plan_id
            or plan.task_b_scope != reference.payload.task_b_scope
            or plan.prepared_registration_ids != reference.payload.prepared_registration_ids):
        raise PermissionError("the original date, calendar or task source changed")
    matrix, policies, scope, members, archives, tables = _read_handoff(Path(plan.handoff_zip))
    if (members != dict(plan.source_member_sha256)
            or archives != dict(plan.input_archive_sha256)
            or scope["source_result_id"] != plan.reference_result_id):
        raise PermissionError("handoff source identities changed")
    _verify_reference_context_member(Path(plan.handoff_zip), scope)
    if tables != dict(plan.context_table_sha256):
        raise PermissionError("registered context table set changed")
    rows = _validate_matrix(matrix, policies)
    bases = {row.name: row.model_dump(mode="json")
             for row in reference.payload.configurations}
    expected = tuple(_variant_from_intent(row, bases[row["base_reference"]]) for row in rows)
    if expected != plan.variants:
        raise PermissionError("matrix resolution differs from the frozen worker configurations")


def save_mffu_batch_plan(store_root: Path, envelope: MffuBatchPlanEnvelope) -> str:
    """Use the established immutable plan store; approval remains a separate action."""
    verify_mffu_batch_plan(envelope.payload)
    plan_id = envelope.funded_comparison_plan_id
    if not has_envelope(store_root, PLAN_STORE, plan_id):
        save_envelope_immutable(store_root, PLAN_STORE, envelope)
    else:
        load_verified_envelope(store_root, PLAN_STORE, plan_id, MffuBatchPlanEnvelope)
    return plan_id


def load_approved_mffu_batch_plan(store_root: Path, plan_id: str):
    """Return an exact approved plan for worker dispatch or raise before replay."""
    from alpha_lab.propsim.funded.comparison_runner import find_approval

    envelope = load_verified_envelope(store_root, PLAN_STORE, plan_id, MffuBatchPlanEnvelope)
    verify_mffu_batch_plan(envelope.payload)
    approval = find_approval(store_root, plan_id)
    if approval is None:
        raise PermissionError("the exact final 64-intent plan has no owner approval")
    return envelope, approval
