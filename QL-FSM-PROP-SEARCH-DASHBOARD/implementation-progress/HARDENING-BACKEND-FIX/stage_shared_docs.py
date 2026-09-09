"""Stage the three shared docs as HEAD + HARDENING-BACKEND-FIX lane transforms ONLY.

Same mechanism as ``../HARDENING-BACKEND/stage_shared_docs.py`` and
``../R6.1-FIX/stage_shared_docs.py`` (user hunks never enter the release commit;
``--apply-worktree`` post-commit replays the same transforms so the surviving
worktree diff is the user's pre-existing hunks only).
HEAD for HARDENING-BACKEND-FIX is the HARDENING-BACKEND commit (``e56f937``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
HARDENING-BACKEND-FIX additions (the compact backend correction of
`QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/HARDENING-BACKEND-FIX/IMPLEMENTATION_PLAN.md`
— ten corrections, nothing else; no owner action taken, no seed production, no real ≤5-day
run; `backend_dev_complete_for_ui = true`, acceptance still transitively blocked by R1):

- **Token-safe stale-lock reclamation** (`ifvg/search/file_mutex.py`, `owner_decision_lock.py`):
  reclamation runs under a private standard-library cross-process mutex (`msvcrt` byte range /
  `fcntl.flock`; never unlinked; carries no authority); the stale body is re-read and re-evaluated
  under the mutex and unlinked ONLY when byte-identical to the dead holder observed; a persistent
  read failure is the typed `lock_read_failed` (never absence); `release()` raises
  `lock_release_failed` when it cannot verify its own lock; a failed body write removes the partial
  exclusive file.
- **Atomic, recoverable namespace initialization** (`ifvg/search/store_namespace.py`): the
  envelope and the genesis head are published as one pair through temporary files and
  verified-loaded together under a one-time init mutex; a half-initialized store is recovered only
  by the identical request (class + explicit instance id, no supersession records) and is otherwise
  the typed `incomplete_store_namespace_initialization`; conflicting bytes are never overwritten.
- **Public source-kind boundary** (`ifvg/search/trading_calendar.py`): the public `SourceKind` is
  exactly `mbp1` / `trades` / `legacy_verified_replay_source`; a historical physical partition
  resolves to the opaque legacy value through the private physical-file resolver (an internal
  `PhysicalSourceDescriptor` keeps the truthful file name, era, hash and partition key and implies
  nothing beyond replay bytes); every inventory, window ref and the seed inventory hash refuse the
  physical stem.
- **Exact regime provenance and native validation** (`ifvg/ml/regime_contracts.py`,
  `regime_oos_assignment.py`, `regime_fold_features.py`, `regime_assignment_sources.py`): the
  descriptive OOS assignment keeps three-way semantics (valid; invalid with its applicable fit /
  fold / partition and the fit's typed reason; `no_oos_assignment` only for a candidate with no
  OOS test row) on the candidate grain, the panel PIT rule, the fold-feature spine and the
  executed-trade projection; every assignment / fold-feature table is validated natively before
  any conversion (actual booleans, integral values, no numeric strings / infinity / sentinel
  identifiers, no `errors="coerce"`); the OOS payload binds the registered schema hash, the saver
  and the loader decode the Arrow bytes and prove schema / count / uniqueness / row invariants, and
  the candidate-as-of and assignment sets are exactly equal.
- **Fail-closed manifests; exact label and executed-trade evidence** (`ifvg/search/store.py`,
  `ifvg/ml/comparison_rows.py`, the three study runners, `regime_stratification_service.py`): one
  central manifest-entry validator is shared by every probe / load / reuse path (bare relative file
  names, lowercase 64-hex digests, non-negative byte counts, no duplicates / reserved names, the
  envelope entry exactly once; artifacts resolved and compared before opening —
  `sidecar_path_escape`; `invalid_store_locator` distinguished from absence); duplicate label
  candidates are refused before hashing; the pipeline's persisting study seams prove the label
  artifact derives exactly from the registered policy; the persisting stratification service
  requires every child's exact `executed_trade_table_id`.
- **Central seed canonicalization** (`ifvg/search/child_replay.py::save_seed_snapshot`): the one
  seam rebuilds every aware datetime (pytz / zoneinfo / fixed offsets) under the stdlib UTC
  tzinfo, leaves naive datetimes unchanged and rebuilds containers, so the same instants under any
  representation mint the same seed hash, snapshot id and sidecar bytes; the seed-production runner
  delegates to it.
- **Bounded event-detail partition** (`propsim/event_detail.py`, `propsim/search_bridge.py`,
  `scripts/hardening_capacity_benchmark.py`): `EVENT_DETAIL_BUDGET_V2` registers
  `max_rows_per_partition = 50,000` (the benchmark's measured row-group size; no ceiling lowered);
  the writer flushes at the bound even inside one path, partitions are keyed
  `(path_block_id, partition_ordinal)` with first / last event keys, rows, bytes, digest and schema
  hash, a refused build leaves no partition behind, the reader proves the bound and the total
  order; the V1 budget stays loadable but is refused by the writer; the benchmark's `normal` /
  `skewed` / `dense` shapes and the resident-batch gate passed at 250k / 500k / 1M rows.
- **Complete authority-chain proof** (`ifvg/search/owner_decisions.py::verify_complete_owner_authority_chain`):
  the ONE proof every real authority seam runs (charter freeze / load, pipeline launch, activation,
  executors, verification run, MBP-1 diagnostic, the seed-production and bounded-verification
  authorizations, the regime chain loader) — every record verified from the genesis anchor, every
  superseded and replacement decision verified-loaded, lawful transitions only, and the signed
  witness equal to the verified current head (`supersession_decision_unverifiable`,
  `supersession_transition_unlawful`, `supersession_chain_divergent`).
- **Unchanged (golden-tested)**: the Strategy-Core pin, the fixed M0–M3 lane, the R6.1-FIX goldens
  (`B0_CORE` bundle id, candidate protocol id, `core_replay_id`, `account_simulation_id`,
  `feature_block_registry_hash`), R5B formulas, model protocol parameters, KMeans fit identities,
  prop-firm rule contracts, the S11 blocked reason, `order_flow_depth_policy="mbp1_only_v1"`.
  Re-minted (synthetic only): inventories / windows / seed authorizations that serialized the
  physical stem, seeds created from non-UTC representations, simulations under the default (V2)
  event-detail budget, regime OOS / fold artifacts whose invalid rows previously collapsed.
"""

README_OLD = """authorization/run contracts, and the bounded-verification preflight and
reports)
implementations are complete;"""
README_NEW = """authorization/run contracts, and the bounded-verification preflight and
reports), and HARDENING-BACKEND-FIX (the compact backend correction: token-safe
stale-lock reclamation, atomic recoverable namespace initialization, the public
source-kind boundary, exact regime provenance with native validation, fail-closed
manifests with exact label / executed-trade evidence, central seed
canonicalization, the bounded event-detail partition, and the complete
authority-chain proof at every real seam)
implementations are complete;"""

YAML_OLD = (
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    HARDENING_BACKEND_FIX: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    "backend_dev_complete_for_ui: true, ui_implementation_may_begin: true, "
    "real_verification_run_completed: false, full_authorized_development_run_completed: false, "
    'corrections: "token-safe stale-lock reclamation under a private cross-process mutex; '
    "atomic recoverable namespace initialization (envelope + genesis head as one verified pair); "
    "public source kind mbp1 / trades / legacy_verified_replay_source (the physical stem never "
    "public); exact invalid OOS / fold-feature provenance with native validation before "
    "conversion, the registered OOS schema hash and exact candidate sets at save and load; one "
    "central manifest-entry validator (traversal, symlink escape, duplicates, reserved names, "
    "malformed entries fail typed); exact label and executed-trade artifact identities for every "
    "persisted study; save_seed_snapshot as the one seed canonicalization seam; "
    "event_detail_budget_v2 (max_rows_per_partition 50000); "
    'verify_complete_owner_authority_chain at every real authority seam", '
    'capacity_policy: "HARDENING_CAPACITY_POLICY_V1 + event_detail_partition_row_bound_v2 '
    "(B1 normal / skewed / dense and B2 gates passed at 250k/500k/1M rows; maximum resident "
    'writer batch 50000 rows)", '
    'owner_actions_performed: "none (window not selected; no SeedProductionAuthorizationRef; '
    'no VerificationAuthorizationRef; no real seed replay; no real verification run)"}\n'
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
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
                elif new not in updated:
                    raise SystemExit(f"worktree anchor missing: {path} {old[:60]!r}")
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the HARDENING-BACKEND-FIX lane transforms")
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
        print(f"staged {path} = HEAD + HARDENING-BACKEND-FIX lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
