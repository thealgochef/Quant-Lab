# IFVG Setup-Level Verifier — Verification

Status: provider layer verified against real immutable artifacts; UI layer
verification summarized in §3.

## 1. Provider (`setup_verifier_provider.py`) — proven on real artifacts

`tests/agents/test_ifvg_setup_verifier_provider.py` (runs whenever a local
fsm-audit + replay-chart-v2 bundle exists; skips cleanly otherwise):

- **All 215 setups selectable**, each with non-empty contract-ordered
  evidence and a terminal event; 170 candidate-less setups included — the
  cohort the candidate verifier could never display.
- Death cohorts match the audit exactly: 83 `invalidated_parent_filled`,
  33 `slot_freed`, 13 S4, 15 S3.
- **PIT stage gating**: the activation gate hides all later evidence
  (hidden + visible = full log); no killing fill event leaks through an
  earlier gate. Gates are ORDER POINTS of canonical stage events, not
  bucket filters — fills after the gate are hidden.
- **Bundle identity isolation**: a mismatched fsm-audit manifest hash
  refuses to open.
- Parentless interval reconciliation is visible per setup
  (`sum(bars_count) == step rows`).
- The cross-channel total ordering is re-asserted at open time
  (`_assert_total_order`) and per-evidence-log; a same-minute ambiguity is a
  hard failure.
- Present-but-empty filters (`conflict_flag`, `structural_suppression_flag`)
  carry provably-zero counts on this artifact and are surfaced, never hidden.

## 2. Review ledger

`ifvg_visual_review_v1` extended additively: `setup_id` +
`fsm_audit_artifact_id` keys (a setup-level review must pin both;
candidate-less reviews pass an empty candidate id), new verdict fields
(`htf_verdict`, `parent_verdict`, `opposing_verdict`, `inversion_verdict`,
`fill_verdict`) and setup-level tags. Existing rows remain readable;
append-only immutability and CSV export re-verified by the existing store
tests plus the extended suite.

## 3. Tab + charts (setup mode)

Setup selection mode with the §10.2 filter set (candidate-less, terminal
reason, phase-at-death, HTF/parent timeframe, session, Q-40, parentless,
conflict/suppression — the last two present-but-empty with an honest zero
caption), separate activation/display-end range fields, stage selector over
the extended ladder, gating-report surfacing, and the styling grammar rule
that a candidate-less setup never looks executable. Candidate/decision/trade
modes preserved unchanged (their tests must stay green). See the UI test
files for the executable specification.

## 4. Open items

- Browser viewport/keyboard QA: OPEN (no approved browser session).
- The deterministic review queue is
  `reports/ifvg_fsm_audit/IFVG_SETUP_VISUAL_REVIEW_SAMPLE.csv` (12 cohorts,
  incl. all 13 S4 + all 15 S3 deaths; suppression/conflict cohorts are
  explicit zero-row entries in the coverage report).
