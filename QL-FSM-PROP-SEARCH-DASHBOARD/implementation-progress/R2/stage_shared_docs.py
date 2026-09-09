"""Stage the three shared docs as HEAD + R2 lane transforms ONLY.

The user's pre-existing uncommitted hunks never enter the release commit
(kickoff §1.4): each staged blob is rebuilt from the HEAD content plus exactly
the transforms the R2 working-tree edit applied, hashed with
``git hash-object -w`` and planted via ``git update-index --cacheinfo``.
Run from the repo root AFTER ``git add`` of the normal (non-shared) files and
BEFORE ``git commit``. Idempotent.
"""

from __future__ import annotations

import subprocess
import sys

ARCH_APPEND = """
R2 additions (multi-child search, lineage, deltas, verifier integration):

- **Parent orchestrator** (`search/orchestrator.py`): deterministic
  enumeration over registered axis values deduped on the resolved replay
  identity (within and across studies), `GeneratedProfileCapability` enforced
  BEFORE any replay, O_EXCL per-search lock with checkpoint heartbeats +
  stale-orphan break (resume-after-kill), atomic `search_state.json`
  checkpoints per child transition, `cancel.requested` honored at child
  boundaries only, store-identity verified reuse. Launch shim:
  `scripts/ifvg_search_job.py` (start/status/cancel; worker refuses execution
  without an explicit runner entry — real executors land with the R5
  pipeline).
- **Profile-independent lineage** (`search/lineage.py`): setup → candidate →
  decision → trade lineage payloads derived from source-stable evidence
  (deterministic SC fvg ids + cursors); native→lineage one-to-one enforced
  with persisted `LineageCollisionRecord`s; collisions or incomplete keys
  disable the population (never deduped or fuzz-matched).
- **Exact deltas** (`study/population_delta.py` · `funnel_delta.py`): every
  population delta declares its `match_basis` (`native_id_exact` same-profile,
  `profile_independent_lineage_exact` cross-profile, else `not_comparable`
  with a reason); funnel deltas run over the union counter vocabulary with
  typed conversions and terminal-reason deltas.
- **Gates / frontier / robustness / insights** (`search/gates.py` ·
  `strategy_metrics.py` · `frontier.py` · `robustness.py` · `insights.py`):
  all eleven charter thresholds evaluated with human explanations (incl. the
  bootstrap-CI-excludes-zero gate over the trading-day cluster bootstrap);
  deterministic O(n²) dominance with a persisted lexicographic tie-break trace
  selecting a `Development Exploratory Representative`; ±1-step neighbor
  degradation + plateau widths + knife-edge warnings; seven fixed insight
  categories with exact `EvidenceRef`s, match-basis-aware suppression, and a
  structural forbidden-wording refusal.
- **Declared contrasts** (`study/contrasts.py`): charter-declared only
  (post-hoc refused), fully-crossed paired main effects with the seed-7
  pair bootstrap CI, observational wording enforced by type.
- **Companion wiring (DEV-R1-6 closed)**: `build_child_fsm_audit` +
  `publish_child_fsm_audit` (`fsm_audit_preparation.py`) assemble the
  per-child audit companion from the audit-enabled drive's retained trace and
  channel rows, gated by the per-child `ChildAuditNeutralityReport` and the
  exact funnel⇔audit reconciliation — parity-EXEMPT by design (the accepted
  doc-default parity gate is untouched and remains doc-default-only);
  `build_slice_companions` (`search/child_replay.py`) closes the vertical
  slice's audit/publication/verifier-link gates (v2 tables via the existing
  heavyweight saver; neutrality + audit companions into the immutable search
  stores; exact drill-target resolution proven, vacuous-zero-targets recorded
  honestly).
- **Exact-setup drill-through**: `resolve_selection` gains the `setup_id`
  exact-ID kind (unique-candidate resolution; zero/multi-candidate setups
  refuse toward setup mode — never a policy pick); `queue_jump("setup_id", …)`
  routes into the setup verifier's own exact resolver before the mode radio
  instantiates.
"""

README_OLD = """`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-042, D-043 (D-041,
D-044, D-045 reserved for R3/R5-R6/R4). R1 implementation is complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""
README_NEW = """`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-042, D-043 (D-041,
D-044, D-045 reserved for R3/R5-R6/R4). R1 and R2 (multi-child search,
lineage, exact deltas, verifier integration, `scripts/ifvg_search_job.py`)
implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""

YAML_OLD = (
    '    R2_multi_child_search: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R2_multi_child_search: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], input=data, capture_output=True, check=True
    )
    return result.stdout


def main() -> int:
    for path, transforms in TRANSFORMS.items():
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = head
        for old, new in transforms:
            if old is None:
                staged = staged.rstrip("\n") + "\n" + new
            else:
                if old not in staged:
                    raise SystemExit(f"{path}: transform anchor missing at HEAD")
                staged = staged.replace(old, new)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R2 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
