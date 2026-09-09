"""Stage the three shared docs as HEAD + HARDENING-BACKEND lane transforms ONLY.

Same mechanism as ``../R6.1-FIX/stage_shared_docs.py`` (user hunks never enter
the release commit; ``--apply-worktree`` post-commit replays the same
transforms so the surviving diff is the user's pre-existing hunks only).
HEAD for HARDENING-BACKEND is the R6.1-FIX commit (``0c8d528``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
HARDENING-BACKEND additions (Phase 2 backend hardening + Phase 3 / Phase 4 contract
authoring of `QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/`
`R6.1-FIX-HARDENING-READINESS-PLAN/IMPLEMENTATION_PLAN.md`
revision 3; findings F-11 F-12 F-13 F-16 F-17 F-18 F-20 F-21 F-22; no owner action taken,
no real seed replay, no real ≤5-day run):

- **Semantic store namespace (F-11)** (`ifvg/search/store_namespace.py`,
  `scripts/ifvg_store_namespace.py`): research-versus-test authority is the store's
  immutable, re-verified `STORE_NAMESPACE.json` envelope (`namespace_class`, a stable
  `store_instance_id` that is never a path hash, the supersession genesis anchor) — never a
  pathname; every owner decision, supersession record, authorization bundle, verification
  and seed-production authorization binds the `store_namespace_id`; an unmarked store has no
  authority; the one-time explicit `init` migration states the class; the old pathname
  heuristic survives as a deployment defense-in-depth check only.
- **Immutable supersession chain + head witnesses (F-12)** (`ifvg/search/supersession_chain.py`;
  store `owner_decision_supersessions`): one content-addressed record per replacement, a
  mandatory head (`owner_decisions/SUPERSESSIONS.head`) whose digest commits to the whole chain
  from the genesis anchor, four-step atomic publication under the lock (an orphan record has no
  authority; the head only ever names a verified record), idempotent identical replay, refused
  divergent replay; every real charter / authorization records the current
  `{store_namespace_id, line_count, head_sha256}` witness and a missing, shorter or different
  current head is refused (local rollback detection, not cryptographic authenticity).
- **Liveness-aware owner-decision lock (F-13)** (`ifvg/search/owner_decision_lock.py`): pid,
  process-start token, random token, host, heartbeat; reclaimed only when the heartbeat timed
  out AND the holder is demonstrably dead (no such pid / exited / PID reuse); a live holder,
  another host or a malformed body is never reclaimed; the writer re-verifies its token before
  publication and a lost lock aborts; release unlinks only its own token.
- **Capacity (F-17)** (`propsim/event_detail.py`, `propsim/search_bridge.py`,
  `ifvg/ml/regime_stratified_prop.py`, `scripts/hardening_capacity_benchmark.py`): the
  event-detail writer streams an iterable of walk pairs (never materialized), keeps no
  whole-artifact id index (canonical-key argument + an unconditional disk-backed DuckDB
  distinct check over the written partitions), and the regime-stratified event summary
  aggregates exactly through DuckDB over intermediate Parquet partitions under an explicit
  memory limit and attempt-local temp directory with canonical ordering; the
  `HARDENING_CAPACITY_POLICY_V1` benchmark (native RSS) passed every §4.4 gate at 250k/500k/1M
  rows (`CAPACITY_BENCHMARKS.md`).
- **Warning policy (F-18)**: `pyproject.toml` runs the suite under `filterwarnings = error`
  with ONE exact third-party rule (the scikit-learn 1.7 / SciPy 1.16 L-BFGS-B deprecation);
  `dataset.concat_schema_aligned` replaces the deprecated concat with explicit dtypes (frozen
  bytes unchanged; all-null columns never dropped); project-owned warnings = 0.
- **Sequential execution truth (F-20)** (`ifvg/search/pipeline.py`, `scripts/ifvg_pipeline_job.py`):
  `SUPPORTED_CHILD_WORKERS = 1`, `execution_mode = sequential_children_v1`; `WorkerPolicy`
  refuses `max_workers != 1` with the typed reason `unsupported_worker_parallelism_v1` (never
  coerced), the job shim refuses before job creation, and every attempt receipt persists
  `effective_workers=1` / `execution_mode`.
- **Phase 3 contracts (F-16 / F-21 / F-22)** (`ifvg/search/trading_calendar.py`,
  `verification_window.py`, `seed_production.py`; `scripts/ifvg_verification_window_shortlist.py`,
  `scripts/ifvg_seed_production.py`; stores `seed_production_authorizations`,
  `seed_production_runs`): a logical trading day is the Strategy-Core trading-day id whose
  stream is `[td−1 18:00 ET, td 18:00 ET)` over the physical partitions `(td−1, td)`
  (`cme_globex_18et_weekday_v1`; physical Sunday partition dates are not trading days); the
  coverage shortlist was rebuilt from already-authorized evidence on consecutive logical days
  under the plan's lexicographic ranking (no owner selection, no allowlist registration — the
  June proposal is INELIGIBLE as stated: no exact verifier target); the separately authorized
  seed-production lane (`SeedProductionReplayPolicy`, authorization / run contracts that bind
  the namespace + head witness, profile, store-day chain, source-inventory hash and code
  identities; permitted outputs = seed snapshot + access audit + run receipt) is proven
  synthetically; the owner packets are unsigned and their placeholders fail validation.
- **Phase 4 authoring** (`ifvg/search/bounded_verification.py`,
  `scripts/ifvg_bounded_verification.py`; stores `r1_baseline_gate_reports`,
  `bounded_release_control_flow_reports`): the typed §6.1
  preflight (namespace, witness, real authorization, 1–5 consecutive logical days, physical
  mapping, program allowlist, seed) before any path; the immutable `R1BaselineGateReport`
  (both attempts' six gates + the audit-mode digests + the eight "also prove" proofs) and the
  release-specific `BoundedReleaseControlFlowReport` (eight components typed from the persisted
  pipeline state — never research evidence); the runner refuses `fail_before_path` without the
  owner's persisted authorization (proven against the real, run-less verification store).
- **Unchanged (golden-tested)**: `resolved_regime_protocol_id`, the R6 golden `regime_fit_id`,
  the frozen M0 CatBoost hash, `core_replay_id`, `account_simulation_id`,
  `feature_block_registry_hash`, the `B0_CORE` bundle id. Re-minted (synthetic only):
  owner-decision artifact ids (`store_namespace_id`), verification-run ids and charter ids that
  carry a real bundle (namespace + witness), execution-attempt receipts.
"""

README_OLD = """immutable `executed_trade_tables` store consumed by S02/S14, the typed
sidecar probe with fail-closed prior-stage recoveries, and the
`PipelineWiringError` / MBP-1 scope-equality / enum-copy corrections)
implementations are complete;"""
README_NEW = """immutable `executed_trade_tables` store consumed by S02/S14, the typed
sidecar probe with fail-closed prior-stage recoveries, and the
`PipelineWiringError` / MBP-1 scope-equality / enum-copy corrections), and
HARDENING-BACKEND (the semantic store namespace, the immutable supersession
record chain with head witnesses, the liveness-aware owner-decision lock,
the streaming event-detail writer and the external DuckDB event-regime
summary under measured capacity gates, the warnings-as-errors policy, the
sequential-execution truth of the V1 executor, the logical trading-day
calendar with the rebuilt verification-window shortlist, the seed-production
authorization/run contracts, and the bounded-verification preflight and
reports)
implementations are complete;"""

YAML_OLD = (
    '    V1_hardening: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    HARDENING_BACKEND: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'phases: "2 (backend hardening), 3 (contract authoring), 4 (code authoring)", '
    'new_stores: "owner_decision_supersessions, seed_production_authorizations, '
    "seed_production_runs, r1_baseline_gate_reports, bounded_release_control_flow_reports\", "
    'contracts: "StoreNamespace, SupersessionHeadWitness, OwnerDecisionSupersession, '
    "OwnerDecisionLock, TradingCalendar cme_globex_18et_weekday_v1, VerificationWindowShortlist, "
    "SeedProductionAuthorization, SeedProductionRun, BoundedVerificationPreflight, "
    "R1BaselineGateReport, BoundedReleaseControlFlowReport\", "
    'execution_mode: "sequential_children_v1", supported_child_workers: 1, '
    'warning_policy: "hardening_warning_policy_v1 (filterwarnings=error; '
    "one exact third-party rule; project_owned_warnings=0)\", "
    'capacity_policy: "HARDENING_CAPACITY_POLICY_V1 (B1/B2 gates passed at 250k/500k/1M rows)", '
    'findings_closed: "F-11 F-12 F-13 F-17 F-18 F-20; F-16 F-21 F-22 contracts authored", '
    'owner_actions_performed: "none (window not selected; no SeedProductionAuthorizationRef; '
    "no VerificationAuthorizationRef; no real seed replay; no real verification run)\"}\n"
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)

YAML_ALLOWLIST_OLD = (
    '    allowlist_status: "PROPOSAL pending coverage-evidenced owner sign-off; '
    'one canonical allowlist program-wide"\n'
)
YAML_ALLOWLIST_NEW = (
    '    allowlist_status: "PROPOSAL pending coverage-evidenced owner sign-off; '
    "one canonical allowlist program-wide; HARDENING-BACKEND finding: the June proposal is "
    "INELIGIBLE under the plan's exact-verifier-target constraint (no executed trade in the "
    "accepted evidence) — the owner selects from the rebuilt LOGICAL-day shortlist "
    "(implementation-progress/HARDENING-BACKEND/VERIFICATION_WINDOW_SHORTLIST.md); "
    "the allowlist is a tuple of consecutive logical trading days, "
    "never physical partition dates\"\n"
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW), (YAML_ALLOWLIST_OLD, YAML_ALLOWLIST_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], input=data, capture_output=True, check=True)
    return result.stdout


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit(f"transform anchor missing at HEAD: {old[:60]!r}")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
    dry_run = "--dry-run" in sys.argv[1:]
    for path, transforms in TRANSFORMS.items():
        if apply_worktree:
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the HARDENING-BACKEND lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        if dry_run:
            print(f"dry-run {path}: {len(head)} -> {len(staged)} chars")
            continue
        blob = (
            _git(
                "hash-object",
                "-w",
                "--stdin",
                "--path",
                path,
                data=staged.encode("utf-8"),
            )
            .decode("ascii")
            .strip()
        )
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + HARDENING-BACKEND lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
