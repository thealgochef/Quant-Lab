# IFVG FSM Auditability Repair — Implementation Verification

Status: implementation complete; publication fields in §7 are recorded at the
Phase 6.5 gate. Behavior-neutral by construction and by measurement: NO
strategy rule/profile/label change, NO training, NO ablation, existing
artifacts immutable, sealed/June-11 untouched, Trade-Lab unmodified.

## 1. Starting state (captured BEFORE any edits, 2026-08-04)

Quant-Lab HEAD `bd6825924d02f76c108e81ef1d4edb27b032d55a` on `platform-refactor`.
The full `git status --porcelain` inventory (93 entries of pre-existing dirt
from the prior IFVG windows — none of it belongs to this task) is preserved
verbatim at `reports/ifvg_fsm_audit/QL_DIRTY_STATE_BEFORE_EDITS.txt`.
Strategy-Core was clean at `f16d27d07f68820b3a5958eafd66cce68f6a7f3c`.

## 2. Strategy-Core changes (committed + pushed BEFORE the QL pin edit)

- `527f3dded814f0bd822866c25fb2f8177bd63972` — opt-in FSM audit channel
  (`audit_capture_mode = disabled | fsm_audit_v1`): enriched `FvgFillEvent`
  evidence fields (not snapshotted), six audit record kinds carrying the
  `AuditStamp` cross-channel ordering contract, per-step-drained reducer
  buffer with enforced drain discipline, day-local ordinal counters that make
  chained and continuous drives stamp identically, `IfvgDayResult.audit_emissions`
  defaulted field. 24 new tests incl. the D-1/D-2 characterizations, the
  parentless-step semantics suite, drain-boundary enforcement, the seed-shape
  freeze test, and pre-change `seed_hash` golden digests
  (`fcd5cdf7…`, `2887bf35…`, `1bfebf0c…`) proven identical with the channel
  on, off, and absent.
- `a4e3303179ac6a1088aecaaa3482934cf1aec4d7` — hot-path repair after the
  declared disabled-mode regression gate FAILED its first measurement
  (7.63% median vs the 5% ceiling): prior penetration is now derived lazily
  at event construction. Re-measured +1.97% median / −2.60% p95 — PASSED
  (`reports/ifvg_fsm_audit/IFVG_FSM_PERFORMANCE_GATE.json`).
- Full Strategy-Core suite: 225 passed; ruff clean. QL `pyproject.toml` pins
  `@a4e3303…`; the installed distribution's `direct_url.json` records the
  same commit (verified).

## 3. Quant-Lab audit lane

New modules: `audit_contracts.py` (13-table contract, nullability, PKs,
funnel⇔events reconciliation), `fsm_audit_io.py` (verified loader),
`fsm_audit_parity.py` (deep exact-parity diagnostics), `fsm_audit_preparation.py`
(lock-guarded persisted job), `setup_verifier_provider.py` (bundle open,
`list_setups`, stage-gated `setup_evidence`). Extended: `capture_driver.py`
(`CaptureDayResult.audit_rows`, `flatten_audit_emissions`), `dataset.py`
(`build_ifvg_fsm_audit_v1` + `assemble_fsm_audit_tables` +
`_derive_parentless_intervals`), `manifest.py` (`FsmAuditIdentity`,
`save_fsm_audit_immutable`), `config.py` (FSM-audit pins `143b510f…`/
`b089dfad…` — the pin discrepancy vs `ACCEPTED_V2_DATASET_ID` is deliberate
and registered), `replay_chart_store.py` (v2 setup-aware artifact +
`VerifierBundleRef`), `visual_review_store.py` (additive setup fields),
`scripts/prepare_ifvg_fsm_audit.py`, `scripts/ifvg_fsm_audit_reports.py`.

## 4. Gates — all measured, all passed

| Gate | Result |
|---|---|
| Exact v2 parity (7 tables, content hashes, PK/FK, row hashes) vs accepted `143b510f…` | **PASSED** — byte-exact; expected funnel 215/132/33/30 confirmed |
| Funnel ⇔ audit events reconciliation (all mapped counters + per-day parentless interval checksum) | **EXACT** |
| Cross-channel total ordering (no (step, substep, ordinal) ties; per-day audit_seq strictly monotone) | asserted at build AND at provider load |
| Per-step drain boundary (undrained buffer at next bar) | hard error, tested |
| Seed identity | goldens byte-identical; seed-shape freeze test pins every snapshotted dataclass |
| Disabled-mode regression (declared ≤5% median / ≤10% p95) | 1.97% / −2.60% — PASSED (after one real failure + fix, §2) |
| 215/215 setups selectable with evidence; 170 candidate-less renderable | PASSED (`test_ifvg_setup_verifier_provider.py`) |
| Zero protected/sealed access | all counters zero on every run |

Validation-lineage artifacts (dirty-tree identities, superseded by the
publication run): fsm-audit `8f87271f…`, replay-chart v2 `8897da5c…`.
Key measured facts: 251 slot deaths = 215 terminal + 36 provisional;
101,066 fill events; 18,863 parentless steps grouped into 251 intervals
(checksum exact); 132 entry-causality records ↔ 132 candidates; 92
htf-fill terminal deaths; 83 parent-fill terminal deaths; 13 S4 + 15 S3
deaths (the review-queue strata).

## 5. Reports

`reports/ifvg_fsm_audit/`: `IFVG_FSM_AUDITABILITY_PARITY_REPORT.json`,
`IFVG_FSM_FUNNEL_REPORT.{md,json}` (the three previously-unknown quantities —
cross-setup opportunity cost, session impact, ranked-fallback benefit — are
now reported from persisted evidence, DESCRIPTIVE ONLY),
`IFVG_FSM_EVIDENCE_COVERAGE_REPORT.{md,json}` (every contract drop reason
observed or provably zero), `IFVG_FSM_RULE_CONFLICT_REPORT.md` (D-1..D-5
characterized, all OPEN), `IFVG_SETUP_VISUAL_REVIEW_SAMPLE.csv`
(deterministic 12-cohort queue incl. all 13 S4 and all 15 S3 deaths),
`IFVG_FSM_PERFORMANCE_GATE.json`.

## 6. Out of scope — enforced

No rule/threshold/session/timeout/direction/entry-family/label change; no
ranked fallback; no conflict-rule fix; no ablation; no training; no
Trade-Lab writes; no sealed access. The D-1..D-5 register holds every
characterized-but-unresolved decision. Browser viewport/keyboard QA is
recorded as OPEN (no approved browser session).

## 7. Publication record (filled at the Phase 6.5 gate)

- QL source commit: `PENDING`
- Final fsm-audit artifact id / manifest hash: `PENDING`
- Final replay-chart v2 artifact id / manifest hash: `PENDING`
- Catalog update commit: `PENDING`
- Remaining pre-existing dirt not belonging to this task: `PENDING`
