# HARDENING-BACKEND — Warning Policy Report (plan §4.5, F-18)

Policy id `hardening_warning_policy_v1` (machine-readable record:
`WARNING_BASELINE.json`). Written by WS-C on 2026-09-02 against the
R6.1-FIX baseline (`0c8d528`): 435 warnings per full-suite run = 429
third-party + 6 project-owned (`../R6.1-FIX/_final_pytest.txt`).

## Outcome

| Item | Before (R6.1-FIX) | After (HARDENING-BACKEND) |
|---|---|---|
| Suite warning mode | default (warnings reported, never fatal) | `filterwarnings = ["error", <one exact rule>]` — every warning not matching the one registered rule FAILS the test |
| Project-owned warnings | 6 (5 × `dataset.py:813` pandas concat `FutureWarning`; 1 × `tests/agents/test_ifvg_context_experiment_engine.py:260` `FutureWarning`) | **0** — both fixed at the source with explicit schemas / dtypes |
| Third-party warnings | 429 × sklearn / SciPy L-BFGS-B `disp` / `iprint` `DeprecationWarning` | one narrowly scoped ignore rule (exact message, category, module, package version, reason, owner, expiry) — nothing broader |
| Broad `DeprecationWarning` suppression | none | none |

## Project-owned fix 1 — `dataset.py` audit-channel concat (`assemble_fsm_audit_tables`)

**What emitted it.** The per-day audit-channel frames of a chain are
row-concatenated into `audit_all`. Different event kinds populate different
columns, so a day's frame routinely carries all-NA columns; pandas ≥ 2.1
deprecated *excluding all-NA entries from the result-dtype determination* and
warns at every such concat (`FutureWarning: The behavior of DataFrame
concatenation with empty or all-NA entries is deprecated …`).

**What the code does now.** `concat_schema_aligned(frames)` (new, exported
from `alpha_lab.agents.data_infra.ifvg.dataset`) states the schema instead of
relying on inference:

1. genuinely EMPTY frames are excluded (the pre-existing behavior);
2. the column order is the first-seen union (`sort=False` semantics);
3. a column's dtype is the common dtype of its NON-all-NA entries;
4. a column ABSENT from some frame widens that dtype exactly as pandas'
   `ensure_dtype_can_hold_na` does (numpy int/uint → float64, bool → object;
   everything else holds NA natively) and is aligned as typed NA;
5. an all-NA PRESENT entry joining a numpy int / uint / bool column takes the
   common type the deprecated path produced when it concatenated the raw
   values (int + all-NaN float → float64; an all-None object entry, a bool
   target, or a datetime entry → object) — the ints and the `None`s are kept
   exactly;
6. every other all-NA entry is cast to the column dtype (NaN / NaT / None);
7. the frames, now dtype-identical per column, are concatenated —
   no all-NA inference remains, so no warning fires.

**Why frozen bytes are unchanged.** The helper reproduces the
PRE-deprecation result — the same columns, the same dtypes, the same values —
and never drops an all-null column (the contract's nullable required columns
may legitimately be all-NA and are preserved; `_trim_audit_frame` still
decides what a typed audit table keeps). Proof:

- `tests/agents/test_hardening_warning_policy.py::test_real_audit_frames_concat_is_byte_identical_to_the_old_path`
  builds the deterministic three-day synthetic FSM chain's audit frames (the
  same fixture `test_ifvg_fsm_audit_contracts.py` uses) and asserts the new
  result equals the old call (computed with the deprecation silenced ONLY in
  the test) — `assert_frame_equal` with dtype checking — while
  `warnings.simplefilter("error")` proves no warning is emitted;
- the synthetic shape tests cover every branch: all-None object vs
  float / bool / tz-aware datetime; an all-null column in every frame; a column
  absent from one frame; all-NaN float vs object strings; int and bool columns
  vs all-None and all-NaN entries; empty + non-empty and all-empty inputs;
- `tests/agents/test_ifvg_fsm_audit_contracts.py`,
  `tests/agents/ifvg_search/test_child_audit_companion.py`,
  `tests/agents/test_ifvg_dataset.py`,
  `tests/agents/test_ifvg_v2_replay_reconciliation.py` and the golden
  identities (`tests/agents/ifvg_search/test_r61_fix_goldens.py`) pass
  unchanged under warnings-as-errors (`_ws_C_pytest_core.txt`).

## Project-owned fix 2 — `tests/agents/test_ifvg_context_experiment_engine.py:260`

`frame.loc[len(frame)] = censored` appended a row carrying `NaT` and `None`
through pandas' internal concat (the same deprecated all-NA path). The test
module now appends the row through `append_censored_row(frame, row)`: a
TYPED one-row frame (`binary_target` becomes `object` because an int column
receives `None`; `resolution_ts_utc` keeps its tz-aware dtype with `NaT`)
concatenated with the frame whose affected columns are widened explicitly.
`test_typed_row_append_replaces_loc_enlargement_without_a_warning` asserts the
result equals the old `loc` enlargement (dtypes and values) and emits no
warning.

## The one third-party rule

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error",
    "ignore:scipy\\.optimize\\x3a The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1\\.18\\.0\\.:DeprecationWarning:sklearn\\.linear_model\\._logistic",
]
```

| Field | Value |
|---|---|
| Exact message | `scipy.optimize: The \`disp\` and \`iprint\` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.` (the regex escapes every `.`; the colon after `scipy.optimize` is the hex escape `\x3a` because pytest splits an ini rule on `:`) |
| Category | `DeprecationWarning` |
| Module | `sklearn\.linear_model\._logistic` (issued at `sklearn/linear_model/_logistic.py:456`, `optimize.minimize(..., method="L-BFGS-B", options={"disp", "iprint", …})`) |
| Package / version | scikit-learn 1.7.0 (triggered by SciPy 1.16.0's deprecation) |
| Reason | every lbfgs logistic fit (the R5 supervised ladder, the R6/R6.1 regime studies, pipeline S09/S10) passes through this third-party call; the repository cannot change scikit-learn's solver call |
| Owner | WS-C / HARDENING-BACKEND (F-18) |
| Expiry / removal | remove when the installed scikit-learn no longer passes `disp` / `iprint` (≥ 1.8 expected) or when SciPy ≥ 1.18.0 is installed (the options are gone); a different message, category, or module is NOT suppressed |

Measured baseline of the rule (exact text captured before the policy):

```text
python -m pytest tests/agents/data_infra/ifvg/test_logistic_model.py -q -p no:cacheprovider -W default
→ 8 passed, 11 warnings
  sklearn\linear_model\_logistic.py:456: DeprecationWarning: scipy.optimize: The `disp` and `iprint`
  options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.
python -c "import sklearn, scipy, pandas; print(...)"  → sklearn 1.7.0 scipy 1.16.0 pandas 2.3.1
```

`tests/agents/test_hardening_warning_policy.py::test_registered_pytest_warning_policy_is_error_with_one_exact_third_party_rule`
pins the policy: `filterwarnings[0] == "error"`, exactly one `ignore` rule,
category `DeprecationWarning`, module `sklearn\.linear_model\._logistic`, the
message names `L-BFGS-B`, `disp`, `iprint` and `SciPy 1\.18\.0`, and
`WARNING_BASELINE.json` records the identical rule list.

## Suites run under the policy (warnings as errors)

See `_ws_C_pytest_core.txt` (117 passed, 2 failed), `_ws_C_pytest_ifvg_search.txt`
(484 passed, 9 failed, 5 errors), `_ws_C_pytest_data_infra_ifvg.txt` (212
passed, 5 failed), `_ws_C_pytest_propsim_ifvg_misc.txt` (539 passed, 1
failed); `_ws_C_pytest.txt` indexes them. **Not one failure is a warning**:
every failure is a concurrent WS-A contract change (the namespace / witness
fields and `validate_verification_run(store_root=)`) or a WS-B red-first
module still in progress at run time — the attribution per test is in
`_ws_C_NOTES.md`. No pytest "warnings summary" appears in any run: under
`error` mode a surviving warning would have failed its test. Test modules
that use `pytest.warns` (`tests/agents/test_ml_pipeline.py`, the
`evaluate_walk_forward` deprecation) and `warnings.simplefilter("error")`
(`tests/agents/ifvg_search/test_mbp1_scope_equality.py`) were run and pass:
`pytest.warns` captures the expected warning inside its own context, so the
error filter never sees it. The release gate (main agent) runs the FULL
repository suite twice under this policy (as-is and with provider keys
cleared); WS-C did not run the full suite.
