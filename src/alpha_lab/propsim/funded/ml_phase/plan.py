"""Source-bound phase plans using the established immutable approval store."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field

from alpha_lab.agents.data_infra.ifvg.search.identities import (
    SHA256_PATTERN,
    EnvelopeBase,
    FrozenContract,
)
from alpha_lab.agents.data_infra.ifvg.search.store import (
    load_verified_envelope,
    save_envelope_immutable,
)
from alpha_lab.propsim.funded.comparison_plan import PLAN_STORE
from alpha_lab.propsim.funded.comparison_runner import find_approval

from .protocol import validate_contracts
from .runtime import sha_file


class MlPhasePlanPayload(FrozenContract):
    plan_schema: Literal["ifsm_ml_phase_plan_v1"] = "ifsm_ml_phase_plan_v1"
    purpose: Literal["historical_development_only"] = "historical_development_only"
    contracts_json: str
    reference_plan_path: str
    reference_plan_sha256: str = Field(pattern=SHA256_PATTERN)
    reference_result_id: str = Field(pattern=SHA256_PATTERN)
    reference_result_path: str
    reference_result_sha256: str = Field(pattern=SHA256_PATTERN)
    core_root: str
    runtime_root: str
    bound_files: tuple[tuple[str, str], ...]
    environment_json: str
    owner_request: str


class MlPhasePlanEnvelope(EnvelopeBase):
    _ID_FIELD: ClassVar[str] = "funded_comparison_plan_id"
    funded_comparison_plan_id: str = Field(pattern=SHA256_PATTERN)
    payload: MlPhasePlanPayload


def verify_phase_plan(plan: MlPhasePlanPayload, *, imported: bool = True) -> dict:
    contracts = json.loads(plan.contracts_json)
    validate_contracts(contracts)
    for path, expected in (
        *plan.bound_files,
        (plan.reference_plan_path, plan.reference_plan_sha256),
        (plan.reference_result_path, plan.reference_result_sha256),
    ):
        if sha_file(Path(path)) != expected:
            raise PermissionError(f"frozen ML phase source changed: {path}")
    if imported:
        import strategy_core

        from . import protocol

        core = Path(strategy_core.__file__).resolve()
        quant = Path(protocol.__file__).resolve()
        if not core.is_relative_to(Path(plan.core_root).resolve() / "src"):
            raise PermissionError("phase imported a different Core source")
        if not quant.is_relative_to(Path(plan.runtime_root).resolve() / "src"):
            raise PermissionError("phase imported a different Quant source")
        env = json.loads(plan.environment_json)
        if sys.version != env["python"]:
            raise PermissionError("phase Python version changed")
        for name, version in env["libraries"].items():
            if importlib.import_module(name).__version__ != version:
                raise PermissionError(f"phase dependency version changed: {name}")
    return contracts


def save_phase_plan(store: Path, envelope: MlPhasePlanEnvelope) -> str:
    verify_phase_plan(envelope.payload, imported=False)
    save_envelope_immutable(store, PLAN_STORE, envelope)
    return envelope.funded_comparison_plan_id


def load_approved_phase_plan(store: Path, plan_id: str):
    envelope = load_verified_envelope(store, PLAN_STORE, plan_id, MlPhasePlanEnvelope)
    approval = find_approval(store, plan_id)
    if approval is None:
        raise PermissionError("the exact ML phase plan has no stored owner approval")
    verify_phase_plan(envelope.payload)
    return envelope, approval
