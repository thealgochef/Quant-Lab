"""Stage the three shared docs as HEAD + R5B.1 lane transforms ONLY.

Same mechanism as ``../R6/stage_shared_docs.py`` (user hunks never enter
the release commit; ``--apply-worktree`` post-commit replays the same
transforms so the surviving diff is the user's pre-existing hunks only).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
R5B.1 correction (MBP-1 source-coverage policy v2 — owner planning decision
Q1, 2026-08-28; plan-review correction 4; final closure #2/#7):

- **The R5B rule "positive raw venue sequence jump > 1 = source gap" is
  WITHDRAWN and unrepresentable** (`Mbp1SourceContract.gap_semantics` is the
  Literal `mbp1_source_coverage_declared_evidence_v2`;
  `sequence_jump_semantics = sequence_jump_diagnostic_only_v2`). Databento's
  `sequence` is the venue's original channel sequence and `mbp-1` emits only
  top-of-book-changing events, so symbol-level continuity is never implied:
  raw sequence jumps / resets and `ts_recv` spacing are DIAGNOSTICS
  (`Mbp1SequenceJumpDiagnostics`, `Mbp1TsRecvGapDiagnostics`) — never an
  interval, never a missing reason, never a reduced coverage.
- **Evidence-based coverage** (`ifvg/features/mbp1_coverage_evidence.py`):
  every claim carries an explicit `Mbp1EvidenceScope` (dataset / publisher /
  channel / instrument partition, physical partition key, UTC date, the
  VERIFIED expected partition span). Accepted evidence kinds: a
  partition-scope declared gap manifest (`Mbp1PartitionGapManifest`), the
  vendor `F_MAYBE_BAD_BOOK` flag (DBN bit 4 — a CHANNEL-gap warning), and
  the dataset-condition record. Positive completeness exists ONLY through
  `compile_mbp1_partition_gap_manifest` over a verified
  `Mbp1CompletenessCompilationReport` (source inventory + owner review) —
  never a bare boolean. Dataset conditions map to
  `vendor_no_known_dataset_issue` / `vendor_dataset_degraded` /
  `vendor_dataset_pending` / `vendor_dataset_missing` /
  `vendor_condition_unavailable`; they can downgrade a scope, never prove
  a partition complete.
- **`F_MAYBE_BAD_BOOK` semantics**: the uncertainty starts at a
  manifest-declared start, else the last TRUSTED in-scope event (no
  bad-book / bad-`ts_recv` flag), else the partition's expected start —
  never automatically the detection row; it closes ONLY at a documented
  recovery boundary (manifest-declared end, documented vendor recovery
  event, documented snapshot recovery, owner-approved boundary) and
  otherwise runs to the partition end (`open_uncertainty_to_partition_end`,
  fail closed) — the next unflagged row never closes it. The interval is
  channel-scoped only behind a VERIFIED publisher/channel map; otherwise it
  expands conservatively to the publisher/physical partition and is never
  narrowed to the flagged instrument.
- **Coverage calculation (D12)**: per physical partition, denominator =
  `intersection([partition_expected_start_ts, partition_expected_end_ts],
  authorized_session_span)` (18:00 ET previous day → 17:00 ET, DST-aware);
  intervals are unioned, merged once, and clipped to that span; `coverage =
  1 − union_gap_ns / physical_expected_span_ns`; physical partition spans
  are half-open and must be pairwise disjoint after clipping (overlaps
  refuse); multiple UTC partitions of one trading day are measured
  separately and duration-weighted (the weakest partition status wins);
  head/tail gaps count only when declared; an empty span is unknown; an
  in-session event outside every declared span refuses the build; session
  sub-spans no partition covers are uncovered intervals whose windows type
  `coverage_evidence_unavailable`. A positive completeness claim must name
  the partition content it certifies (report `verified_partition_refs` ∩
  partition content hashes), and a provenance label without a manifest is
  unrepresentable. `completeness_status ∈ {evidenced_complete,
  declared_gaps, completeness_unknown}`; a partition without
  partition-scope evidence — or under a downgrading dataset condition — is
  `completeness_unknown` and every window of that day is typed
  `coverage_evidence_unavailable`. A window intersecting a merged verified
  interval is typed `declared_source_gap` (never widened, never imputed).
- **Real read seam**: `read_mbp1_partition_frame` clips the UTC-date file to
  the trading day's authorized session span and to `DEVELOPMENT_CUTOFF_UTC`
  BEFORE normalization or hashing (the protected 18:00 ET tail of the last
  exposed day is structurally excluded); the clipped counts and the raw
  file's sha256 ride the partition row. The previous UTC file's evening
  portion is not composed in R5B.1 (recorded deviation).
- **Re-minted identities**: the normalized event schema retains
  `publisher_id` + `flags` (new schema hash → new source artifact ids);
  `IFVG_ORDER_FLOW_MBP1_V1` is RE-RESOLVED as a second versioned event
  (`with_reresolved_block`: block v3, `ifvg_order_flow_mbp1_formula_v2` /
  `mbp1_feature_materializer_v2`, new resolved block id, new registry hash,
  new B2/B3 bundle ids; the R5B activation payload keeps the historical v1
  versions and the R5B state stays exported as `PRE_R5B1_*`); feature
  artifacts, coverage reports (`coverage_policy_id`), and controlled-study
  ids all move. Synthetic coverage evidence (`provenance =
  synthetic_fixture`) is lawful only under the synthetic marker — the
  pipeline's S05 and the real source builder refuse it.
- **Bounded real-data diagnostic** (`ifvg/features/mbp1_coverage_diagnostic.py`,
  `scripts/ifvg_mbp1_coverage_diagnostic.py`): characterizes per-partition
  row/flag counts, sequence-jump and `ts_recv`-gap distributions, the
  clipped-row counts, and — only through store-verified partition manifests
  supplied by `--evidence-json` — the open-interval facts, which manifests /
  reports / condition records were loaded, and the policy-v2 completeness
  status (an evidence-less partition reports `open_uncertainty_to_partition_end
  = None`, never a misleading `False`); writes an immutable
  `Mbp1CoverageDiagnosticReport` into the verification namespace; runs only
  under the R1 real-slice gate — a `VerificationReplayPolicy` over exactly
  the requested days, a persisted verified `VerificationRunEnvelope` +
  `VerificationAuthorizationRef` binding the allowlist hash and a verified
  coverage-matrix artifact, the one canonical program allowlist, and a
  store root ending in `search_test/v1` (fail-before-path — the real run is
  an owner action); never infers completeness from sequence continuity.
  Real MBP-1 research and R5B acceptance stay blocked until policy v2
  passes on the real fixture.
"""

README_OLD = """`scripts/ifvg_mbp1_panels.py`), and R6 (the V1 KMeans regime lane:"""
README_NEW = """`scripts/ifvg_mbp1_panels.py`; R5B.1 replaced the withdrawn
sequence-jump gap rule with the evidence-based coverage policy v2 —
`ifvg/features/mbp1_coverage_{evidence,diagnostic}.py`, block re-resolution
v3, `scripts/ifvg_mbp1_coverage_diagnostic.py`), and R6 (the V1 KMeans regime lane:"""

YAML_OLD = (
    '    R5B_mbp1_offline_activation: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    R5B_mbp1_offline_activation: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
    '    R5B_1_mbp1_coverage_correction: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'coverage_policy: "mbp1_source_coverage_declared_evidence_v2", '
    'sequence_jump_rule: "withdrawn (diagnostic only)"}\n'
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


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit("transform anchor missing at HEAD")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
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
            print(f"worktree {path} updated with the R5B.1 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R5B.1 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
