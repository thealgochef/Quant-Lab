"""Production runner-entry factories (R5; closes DEV-R4-5's registry gap).

The R2/R4 job shims execute only registry-resolved entries. Until R5 the
registry held the synthetic fixture wiring alone; this module registers the
REAL executors:

* ``search_baseline_verification_v1`` — run_search wiring whose child
  runner executes the REAL Path-A machinery (`run_child_replay` under the
  `VerificationReplayPolicy`, dual-drive, cached artifacts only) for the
  BASELINE child and refuses every other child: real multi-child replays
  are synthetic-only by the two-path verification design (seeds are
  profile-bound), and nothing here can widen that.
* ``pipeline_baseline_verification_v1`` — the same real seams wrapped as
  `PipelineWiring` for the 16-stage pipeline job.

Both factories FAIL BEFORE PATH CONSTRUCTION: without a persisted
`VerificationRunEnvelope` (whose payload binds a real, owner-approved
`VerificationAuthorizationRef`) the factory raises `PermissionError` at
construction — no config, policy, or source path is built. Registering the
executors therefore unblocks the LAUNCH SURFACE, never the data: execution
stays blocked exactly as R1's acceptance is blocked, until the owner's
fixture authorization exists.
"""

from __future__ import annotations

from pathlib import Path

from ..config import IfvgCaptureConfig
from ..development_access import VerificationReplayPolicy
from ..profiles import resolve_profile_config
from .child_replay import run_child_replay
from .store import load_verified_envelope
from .verification import VerificationRunEnvelope

__all__ = [
    "REPO_ROOT",
    "synthetic_firm_specs",
    "real_verification_context",
    "search_baseline_verification_entry",
    "pipeline_baseline_verification_entry",
]

REPO_ROOT = Path(__file__).resolve().parents[6]


def _verification_run_envelope(store_root: Path) -> VerificationRunEnvelope:
    """The persisted verification run for this store — fail-before-path.

    Discovery goes through the provider (catalog/persisted envelopes only,
    stores never listed); absence refuses BEFORE any config, policy, or
    source path is constructed.
    """

    from ..study_providers import verification_authorization_state  # noqa: PLC0415

    state = verification_authorization_state(Path(store_root))
    if not state.exists or not state.verification_run_ids:
        raise PermissionError(
            "real verification execution is blocked: no persisted "
            "VerificationRunEnvelope exists for this store — the owner's "
            "VerificationAuthorizationRef (decisions 21/R-5) has not been "
            "granted (fail-before-path)"
        )
    return load_verified_envelope(
        Path(store_root),
        "verification_runs",
        state.verification_run_ids[0],
        VerificationRunEnvelope,
    )


def real_verification_context(charter_envelope, *, store_root: Path) -> dict:
    """Shared fail-closed preflight for both real executors."""

    from .authorization import SyntheticAuthorizationMarker  # noqa: PLC0415

    payload = charter_envelope.payload
    if isinstance(payload.owner_authorization, SyntheticAuthorizationMarker):
        raise PermissionError(
            "the real verification executors never run synthetic-marker "
            "charters; use the synthetic fixture entry"
        )
    if payload.date_policy.access_policy_id != "verification_fixed_allowlist_max5_v1":
        raise PermissionError(
            "the real verification executors serve the ≤5-day verification "
            "fixture only; full development execution is a separate, "
            "owner-authorized operator action"
        )
    run = _verification_run_envelope(Path(store_root))
    if tuple(run.payload.allowlist) != tuple(payload.date_policy.replay_dates):
        raise PermissionError(
            "the persisted verification run's allowlist does not match the "
            "charter's date policy (one canonical allowlist, V3 P0-7)"
        )
    if run.payload.baseline_profile_id != payload.baseline_profile_name:
        raise PermissionError(
            "the persisted verification run authorizes a different baseline "
            "profile than this charter"
        )
    resolved = resolve_profile_config({"profile_name": payload.baseline_profile_name})
    cfg = IfvgCaptureConfig(section=resolved.section)
    return {
        "run": run,
        "resolved_profile": resolved,
        "cfg": cfg,
        "allowlist": tuple(run.payload.allowlist),
    }


def _baseline_child_runner(context: dict):
    allowlist = context["allowlist"]

    def _runner(*, spec, core_replay_id: str):
        if spec.comparison_role != "baseline":
            raise PermissionError(
                "real multi-child replays are synthetic-only by the two-path "
                "verification design (seeds are profile-bound); only the "
                "baseline vertical slice may replay real data"
            )
        return run_child_replay(
            dates=allowlist,
            cfg=context["cfg"],
            resolved_profile=context["resolved_profile"],
            access_policy_factory=lambda: VerificationReplayPolicy(allowlist),
            core_replay_id=core_replay_id,
            cached_artifacts_only=True,
            dual_drive=True,
        )

    return _runner


def _slice_identity_resolver(context: dict, store_root: Path):
    """Baseline identity via the Path-A assembly (authorized hashing)."""

    from .child_replay import _assemble_slice_identity  # noqa: PLC0415

    def _resolver(spec):
        if spec.comparison_role != "baseline":
            raise PermissionError(
                "real verification identities exist for the baseline child "
                "only; multi-child identities are synthetic-only"
            )
        policy = VerificationReplayPolicy(context["allowlist"])
        _bundle, identity = _assemble_slice_identity(
            run=context["run"],
            resolved=context["resolved_profile"],
            cfg=context["cfg"],
            hashing_policy=policy,
            source_partition_refs=(),
            source_schema_era_id="mbp1_era_v1",
            ql_source_identity=None,
            sc_identity=None,
            strategy_core_root=None,
            repo_root=REPO_ROOT,
        )
        return identity

    return _resolver


def search_baseline_verification_entry(charter_envelope, *, store_root=None) -> dict:
    """`ifvg_search_job` wiring for the real baseline verification slice."""

    root = Path(store_root) if store_root else REPO_ROOT / "data/ifvg_datasets/search_test/v1"
    context = real_verification_context(charter_envelope, store_root=root)
    return {
        "identity_resolver": _slice_identity_resolver(context, root),
        "child_runner": _baseline_child_runner(context),
    }


def pipeline_baseline_verification_entry(
    charter_envelope, semantic=None, *, store_root=None
):
    """`ifvg_pipeline_job` wiring for the real baseline verification pipeline."""

    from .pipeline import PipelineWiring  # noqa: PLC0415

    root = Path(store_root) if store_root else REPO_ROOT / "data/ifvg_datasets/search_test/v1"
    context = real_verification_context(charter_envelope, store_root=root)
    return PipelineWiring(
        identity_resolver=_slice_identity_resolver(context, root),
        child_runner=_baseline_child_runner(context),
        verification_run=context["run"],
        verification_authorization=context["run"].payload.verification_authorization,
    )


def synthetic_firm_specs():
    """The ONE canonical synthetic firm-simulation spec set.

    Shared by the synthetic pipeline wiring (tests) and the UI's
    account-policy-set pinning, so the semantic spec's
    ``account_policy_set_ids`` and the wiring's persisted envelopes can
    never drift. Synthetic contracts stay ``synthetic_fixture_verified``;
    nothing here touches a real firm.
    """

    from alpha_lab.propsim.firm_contracts import SYNTHETIC_FIXTURE_FIRM  # noqa: PLC0415
    from alpha_lab.propsim.risk import FIXED_ONE_NQ_RISK_POLICY  # noqa: PLC0415
    from alpha_lab.propsim.search_bridge import FirmSimulationSpec  # noqa: PLC0415
    from alpha_lab.propsim.withdrawal import REQUEST_MAX_AT_ELIGIBILITY  # noqa: PLC0415

    return (
        FirmSimulationSpec(
            label="synthetic_fixture_firm",
            firm=SYNTHETIC_FIXTURE_FIRM,
            risk_policy=FIXED_ONE_NQ_RISK_POLICY,
            withdrawal_policy=REQUEST_MAX_AT_ELIGIBILITY,
        ),
    )
