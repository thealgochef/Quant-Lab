# R4 — Pre-implementation baseline

Recorded 2026-08-21 before any R4 change.

## Repository state

- **Branch:** `feature/ifvg-prop-robust-config-search-v1`
- **HEAD:** `a23893c44d4a68a5525685a842c7d289499cdaa2` (`R3: add prop lifecycle
  contracts and synthetic account engine`)
- **Working tree:** carries the SAME pre-existing user-owned modifications the
  R1 baseline recorded (`ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`,
  `docs/README.md`, `docs/pipeline_state.yaml` + the untracked research
  artifacts/dirs listed in `../R1/PRE_IMPLEMENTATION_BASELINE.md`). The
  authoritative byte-record of the user hunks remains
  `../R1/PRE_EXISTING_DIFF.patch`; R4 release commits stage those three shared
  docs as HEAD + lane transforms only (same mechanism as
  `../R3/stage_shared_docs.py`).
- **Untracked:** `QL-FSM-PROP-SEARCH-DASHBOARD/` (this progress tree) plus the
  pre-existing user research artifacts; none enter release commits except the
  planned lane files.

## Toolchain

- Python 3.13.1
- streamlit 1.54.0 installed (the plan's `>=1.41` floor is already satisfied
  at runtime; the R4 `pyproject.toml` change declares the floor)
- Strategy-Core: installed per the R1 baseline (editable local working tree,
  pin enforced by CI); **not modified by this lane** (read-only per kickoff §1.5)

## Test/lint baseline (pre-R4)

- Full repo at R3 commit `a23893c`: **1372 passed** (R3 `GATE_SUMMARY.md`,
  final row) — re-confirmed at R4 start: see `TEST_RESULTS.md` baseline row.
- `ruff check` over the covered paths: clean at R3 close.
- `git diff --check`: clean at R3 close.

## Authority for this release

- `FINAL-IMPLEMENTATION-PLAN-DOCS/PHASED_DELIVERY.md` — Release 4 scope/gate.
- `FINAL-IMPLEMENTATION-PLAN-DOCS/FRONTEND_UX_CONTRACT.md` — complete
  normative UI/UX authority (read in full before any UI file was touched).
- `FINAL-IMPLEMENTATION-PLAN-DOCS/TEST_MATRIX.md` §3.11 — R4 AppTest/scan rows.
- `FINAL-IMPLEMENTATION-PLAN-DOCS/CONTRACTS_AND_SCHEMAS.md` §13 — presentation
  contracts (`study_status.py` / `study_presentation.py`).
- Carried obligations INTO R4 (from `../R2/DEVIATIONS.md` +
  `../R3/GATE_SUMMARY.md` open-blockers §3):
  1. comparison surfaces call `persist_lineage_uniqueness` before surfacing
     any cross-profile delta;
  2. the job shim's `--runner-entry` is registry-gated (the UI never passes
     user-shaped strings);
  3. declared two-axis interaction contrasts become evaluable
     (DEV-R2-5; DT §5 / §7A.19.11) — balanced designs only, refusals retained;
  4. R3 carries forward: scenario-mode bridging at the production seam and
     real artifact-store manifests (DEV-R3-9/11) — R4 touches these only where
     the UI stores require it.

## Standing blockers (unchanged)

1. Owner fixture authorization (decisions 21 + R-5) blocks R1 acceptance and
   transitively every later release's acceptance — R4 included. R4 targets
   `implementation_status = complete`,
   `acceptance_status = transitively_blocked_by_R1`.
2. Interactive browser/keyboard/viewport evidence beyond AppTest smoke is a
   hardening gate; whatever cannot be produced in this session's environment
   is recorded OPEN, never claimed passed.
