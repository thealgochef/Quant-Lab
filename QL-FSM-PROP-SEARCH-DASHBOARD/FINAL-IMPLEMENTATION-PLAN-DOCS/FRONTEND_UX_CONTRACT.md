# Frontend and User-Experience Contract — Quant-Lab FSM / Prop Search Workspace

**Contract name:** `ifvg_prop_robust_config_search_frontend_v1`  
**Parent contract:** `ifvg_prop_robust_config_search_v1`  
**Status:** implementation contract with the user-authorized focused-workspace amendment below; desktop flows reviewed, full live execution acceptance remains open  
**Source basis:** planning brief V4 §10.1–§10.30 plus the original codebase-specific frontend design in `IMPLEMENTATION_PLAN.md`, reconciled with the final replay-identity, authorization, prop-fidelity, MBP-1, model/regime, and release contracts.
**UI-1 amendment (2026-09-04):** the owner-approved UI/UX redesign plan (`../implementation-progress/UI-UX-REDESIGN-PLAN/IMPLEMENTATION_PLAN.md`, revision 2) is the controlling amendment for the sections it names. UI-1 (Phase 1) amends §3.2, §7, §8.1, §14, §15, §30, §31, §35 and §36 below in the same change as its code; §5.1 (Summary / Research details / Technical identity & audit), the goal-conditional flows of §7 and the state-driven lifecycle of §30 are amended by UI-3, UI-2 and UI-6 respectively when their code lands. Where an amended paragraph and an original paragraph conflict, the amended paragraph applies.

---

## Focused-workspace presentation amendment — 2026-09-07

This amendment controls normal IFVG presentation and the app shell where it
conflicts with the original sections below, including navigation (§3), disclosure
and identities (§5), study authoring (§7–§15), progress/results/comparisons
(§16–§24), Replay, lifecycle (§30), health/evidence states (§31), and acceptance
(§35–§36). It implements the focused-workspace amendment to the UI/UX redesign
plan. It changes no backend authorization, strategy, evaluation, scientific schema,
identity, immutable result, or store contract.

1. The app registers IFVG Lab (default), ML Training, Dashboard Compatibility, and
   Strategy Analysis using Streamlit's page router. Only the selected page executes;
   workflow controls render within their owning workspace.
2. IFVG has two normal destinations: **My studies** and **Trade review**. My studies
   contains new/resumed study authoring, progress, results and history. Optional
   research detail is distinct from Developer information.
3. **Developer** is registered only when `QUANT_LAB_DEVELOPER_MODE=1` was set at
   startup. No UI or query parameter can enable it. Developer route execution is
   additionally required to instantiate technical renderers, even inside collapsed
   panels. The retained Verification Center and technical tools live there. This
   presentation switch grants no execution authorization.
4. Normal screens, tooltips, tables and research exports show human research fields;
   IDs, hashes, raw JSON, filesystem paths, commands and internal stage codes remain
   in Developer. Exact full identities continue to govern provider joins and links.
5. The default study list excludes artifact-derived verification scope. Mutable
   names/purpose annotations cannot turn frozen verification into research.
   Unresolved legacy scope stays unresolved. Corrupt/missing evidence is visible as
   unavailable and cannot establish success.
6. **Evaluate**, **Compare** and **Search** are primary study choices. **More study
   types** preserves feature/model, prop, cross-firm, full-workflow and context
   research. The wizard presents only applicable steps and one study/save summary.
   The original validators and authorization checks govern freezing and execution.
7. Research results lead with the question's verdict, limitations, up to four
   relevant metrics and a primary comparison/chart. Strategy defaults are net
   expectancy, profit factor, maximum drawdown and trade count. Model/prop metrics
   follow their research questions. Required missing evidence remains a limitation;
   inapplicable prop panels do not render. Quantitative deltas require compatibility.
8. Lifecycle actions follow persisted state. Running progress alone never implies
   completion. Run, resume, safe cancellation, clone, rename, archive, restore and
   catalog addition use existing supported seams. Catalog addition requires fresh
   eligibility and namespace/state validation. Opening/switching pages starts no work.
9. Replay returns a typed **candidate**, **setup**, **empty**, or **unavailable**
   selection. The parent must not render a second selector/inspector. Unresolved
   exact links and empty filters cannot fall through to unrelated evidence. Setup
   review supports candidate-less cases. Actual execution, hypothetical labels,
   model probability and reviewer judgment remain separate. **Save Review** is the
   only review persistence action. Charts have accessible table alternatives.
10. One persistent **Exploratory research** label replaces repeated development
    warnings in normal IFVG. Relevant uncertainty, scope, readiness and failures
    remain visible. Evaluated health reports preserve explicit fail/unknown states;
    a missing verdict or policy-enforced zero is never sufficient for a pass.

The original diagnostic labels and exact technical-identity requirements below
continue to apply within Developer, subject to the preserved UI-1–UI-3 amendments.
The user subsequently narrowed acceptance to desktop layout and actual actions;
mobile responsiveness is excluded. Current browser evidence covers desktop study,
result, review and Developer flows; live execution lifecycle acceptance remains open
where required evidence or authority is absent. See the
[implementation report](../implementation-progress/UI-UX-REDESIGN-PLAN/FOCUSED_WORKSPACE/COMPLETION_REPORT.md)
for actual checks and remaining acceptance; historical screenshots are insufficient.
Filtered selections must update both the displayed case and its evidence. Save
Review must show one current persistence status. Model-result verdicts identify
their evidence section; replay chart text remains readable in a dark app theme.

---

## 1. Authority, scope, and non-regression rule

This document is the complete normative frontend specification for the new Quant-Lab research workspace. It restores the full interaction and usability detail that was compressed during the backend-contract amendment passes.

The implementation agent must treat every **must**, quoted visible label, route, state, fallback, and acceptance row in this document as binding.

This contract does not change the final backend architecture. It applies the final terminology and boundaries:

- fixed M0–M3 remains an unchanged, separate lane;
- every strategy-changing configuration receives a sequential replay or exact verified reuse;
- real scope requires computation-path-scoped owner authorization;
- `VerificationAuthorizationRef` is required before the real five-day slice constructs a source path;
- development selections are displayed as **Development Exploratory Representative**;
- every population comparison displays `match_basis` and disables unsupported commonality claims;
- assumed 1-minute paths are labeled **scenario** or **approximation**, never exact historical chronology;
- MBP-1 is the maximum new order-flow depth; opaque legacy replay provenance is not exposed as a feature, control, model input, or live source;
- R5 is MBP-1 contract readiness; R5B is the offline/research-only MBP-1 activation;
- S11 remains blocked under the exact authoritative reason until model-gated sequential semantics are separately ratified;
- no Trade-Lab serving, activation, order, or deployment control is added.

Where this document names a presentation or interaction while a backend document defines the data contract, both apply. A UI convenience may never weaken a backend fail-closed rule.

---

## 2. Product principles

The workspace must be:

1. **Guided rather than raw-config driven.** The user chooses registered objectives, axes, policies, and bundles. No arbitrary `section_overrides`, free-form contract JSON, hidden model search, or unregistered feature list is exposed.
2. **Progressively disclosed.** Summary answers come first; Analyst and Audit detail remain available without crowding the default view.
3. **Exact and auditable.** Every meaningful result can drill into the exact child, setup, candidate, decision, trade, account event, model fold, artifact, or manifest that supports it.
4. **Immutable after freeze.** Frozen charters, pipeline specifications, completed children, and research results cannot be edited or deleted from the workspace. The user clones to create a changed study.
5. **Truthfully scoped.** Candidate research, actual strategy execution, historical/scenario prop replay, bootstrap simulation, and stress simulation are never mixed or ambiguously titled.
6. **Fail-closed and explanatory.** A blocked capability renders the reason and no launch control. A skipped downstream stage explains which earlier gate prevented it.
7. **Accessible without relying on color or pointer input.** Every visual interaction has a keyboard-operable or standard-widget equivalent.
8. **Responsive and performance-bounded.** Large studies use indexed pagination, bounded chart layers, and summary-first rendering rather than loading complete artifacts into Streamlit memory.

---

## 3. Information architecture

### 3.1 Existing IFVG Lab shell

Retain the existing top-level IFVG Lab tabs in their asserted order:

```text
Experiments
Replay / Verifier
Data & Audit
```

The new workspace is added **inside Experiments**. Do not revive the deleted React prototype. The implementation remains pure Streamlit + Plotly.

### 3.2 Experiments sub-navigation

Use a session-state-backed horizontal radio, not nested `st.tabs`. **UI-1 amendment** — the radio is:

```text
Start | Verify Implementation | New Study | Active Runs | Results | History | Context Research
```

Requirements:

- `Start` is the task-oriented entry: nine cards (Verify the current implementation · Review strategy setups and trades · Evaluate one configuration · Compare one configuration with the baseline · Search FSM parameters · Evaluate feature and model evidence · Test prop-firm feasibility · Run an advanced end-to-end study · Inspect system health and audit evidence); each card derives the run purpose, the study family, the stage plan and the semantic namespace class, shows the store's verified `store_namespace_id` read-only, what will run, the owner approvals needed and a live availability chip (implemented and selectable / implemented but owner-unratified / research-only offline / blocked by a missing dependency / planned post-V1 / available on another tab).
- `Verify Implementation` is the Verification Center (UI-2, plan §5.3; §14.4 below): one state-driven surface — purpose / readiness → the logical trading-day fixture (logical days and physical partitions displayed separately) → the seed-production packet, the registration receipt, the exact seed job command and the verified seed → the final verification packet and the owner's completed `VerificationAuthorizationRef` → review / run (the charter frozen from the signed reference, the registered run, the §6.1 preflight and the exact bounded-run command) → the monitor of the resolved seed / verification stages. It renders no research gates, prop / risk / model controls or Publish route, has no spawn seam, never signs, never produces a seed and never registers the program allowlist.
- **No mutable namespace selector exists on any route** (owner Q1). The semantic namespace derives from the draft's run purpose — Implementation Verification → the `test` store (`search_test/v1`); Development Research and Full Authorized Development → the `research` store (`search/v1`) — and is displayed read-only with the store's VERIFIED `store_namespace_id`. A local filesystem path is an operational location only and never defines authority. Results, History and Active Runs read each run's OWN store (the charter located by exact id across the deployed stores) and derive the verification badge from the artifact.
- `Context Research` delegates to the existing M0–M3 experiment panel verbatim.
- Only the selected sub-surface executes; hidden panels do not poll jobs or build charts.
- Programmatic navigation supports:
  - a Start card → New Study (a draft carrying the card-selected purpose) or Verify Implementation;
  - wizard freeze/launch → Active Runs (only after the worker's persisted state exists);
  - Active Runs row → Results;
  - funnel/result delta → exact verifier;
  - History clone → New Study.
- Existing asserted top-level tab labels remain unchanged.
- New session-state prefixes are:

```text
ifvg_study_v1_*
ifvg_pipeline_v1_*
```

- Existing `ifvg_context_v1_*` keys are written only by the explicitly registered cross-lane interactions.

### 3.3 Why a radio is required

A horizontal radio is normative because it:

- executes only the selected surface;
- avoids hidden `st.fragment` polling;
- avoids wasted chart construction;
- supports programmatic route changes;
- preserves existing top-level tab assertions;
- remains keyboard reachable.

---

## 4. Frontend module and ownership map

Pure, testable presentation logic belongs under `src/`; Streamlit files remain thin widget layers.

| File | Required public surface and responsibility |
|---|---|
| `scripts/ifvg_ui_common.py` | `sanitize_error`, `sanitize_select`, `display_metric`, `status_badge`, `dev_only_badge`, `disclosure_level`, `identity_block`, `cli_escape_hatch`, `render_empty_state`, `paginate_controls`, `result_scope_caption`, `queue_replay_drilldown`; shared responsive and accessibility helpers |
| `scripts/ifvg_study_tab.py` | `render_ifvg_study_tab(st_module=st, context_research=None)`; Experiments sub-navigation router; only selected route executes |
| `scripts/ifvg_study_wizard.py` | `render_new_study`; complete eight-step wizard, draft persistence, validation, clone, freeze, launch handoff |
| `scripts/ifvg_active_runs_tab.py` | `render_active_runs` and plain-callable `_render_monitor_body`; status polling, funnel, table, details, safe cancel |
| `scripts/ifvg_results_tab.py` | `render_results`, `render_history`; search selection, overview, frontier/sensitivity/explorer routing, immutable history |
| `scripts/ifvg_results_compare.py` | `render_comparison`, `render_insight_panel`, `render_account_timeline`; exact evidence/drill-down actions |
| `scripts/ifvg_results_charts.py` | pure builders: funnel, funnel delta, frontier, sensitivity heatmap, firm matrix, survival, payout distribution, account timeline; no Streamlit state inside builders |
| `scripts/ifvg_pipeline_tab.py` | `render_pipeline_run`; Configure/Preview/Launch/Monitor/Resume-Retry/Publish workflow |
| `scripts/ifvg_search_job.py` | detached `start/status/cancel/resume` search CLI shim; atomic JSON status; Windows `CREATE_NO_WINDOW` pattern where applicable |
| `scripts/ifvg_pipeline_job.py` | detached pipeline CLI shim with the same status/checkpoint contracts |
| `src/alpha_lab/agents/data_infra/ifvg/study_status.py` | internal status keys, visible labels, glyphs, accessible text, forbidden wording |
| `src/alpha_lab/agents/data_infra/ifvg/study_presentation.py` | pure validators for wizard steps, human configuration names, column presets, fixed empty-state copy, funnel/pagination/estimate formatting, responsive presentation decisions |

Required modifications:

```text
scripts/ifvg_lab_tab.py          # delegates the Experiments branch
scripts/ifvg_verifier_tab.py     # exact queue_jump support including setup_id
pyproject.toml                   # Streamlit >=1.41
.github/workflows/ci.yml         # Ruff includes covered scripts
```

CI must lint the new/modified UI scripts. Pure logic must be unit-testable without Streamlit AppTest.

### 4.1 Required Streamlit capability fallbacks

The supported pin provides `st.fragment(run_every)`, `st.plotly_chart(on_select)`, `st.dialog`, and pinned/sticky dataframe columns. Each capability still has a designed fallback:

| Primary capability | Required fallback |
|---|---|
| `st.fragment(run_every="5s")` | visible manual Refresh control using the same plain body function |
| Plotly `on_select` | always-present selectbox or table-row selector that drives the same selected ID |
| `st.dialog` | inline confirmation form with the same exact warning and typed acknowledgement |
| pinned/sticky table columns | horizontally scrollable table plus repeated selected-config identity/details above the table |
| chart interaction | keyboard-operable button/selectbox/table twin |
| large chart | data-table view and bounded layer/omission report |

A fallback must preserve semantics, not merely suppress an exception.

---

## 5. Shared presentation contracts

### 5.1 Disclosure levels

Every major screen supports three detail levels (**UI-3 amendment** — plan §6.6; the original progressive-disclosure intent is preserved):

```text
Summary
Research details
Technical identity & audit
```

- **Summary:** plain-language answers, the section roll-ups (§5.6), principal cards with status chips, top blockers.
- **Research details:** charts and their table twins, distributions, sensitivity, comparisons, model/regime diagnostics.
- **Technical identity & audit:** complete identities, manifests, contract versions, source/fold/seed references, raw reports, missingness and omission reports.

The selected level persists within the workspace session under the unchanged persisted vocabulary (`summary` / `analyst` / `audit`); the R4 labels `Analyst` / `Audit` are superseded by the tier names above. `ifvg_ui_common.detail_levels` renders the selector and the R4 `disclosure_level` seam delegates to it.

### 5.2 Result-scope labels

Every metric block and chart displays one explicit scope label:

```text
Candidate Research
Actual Executed Strategy
Prop Historical Closed-Trade Replay
Prop 1m Scenario / Approximation
Prop Ordered-Event Replay
Bootstrap Simulation
Stress Simulation
```

Gross, costed, and net values are separately labeled. A candidate metric must never be captioned as an executed-trade result.

### 5.3 Development status

Development-only pages display a persistent, non-dismissible badge. The selected development configuration is shown as:

```text
Development Exploratory Representative
```

Do not display `Best`, `Winner`, `Production Ready`, `Validated`, `Live Ready`, or equivalent wording without a separately authorized status contract.

### 5.4 Status semantics

Internal status keys and visible text must support:

```text
Draft
Frozen
Queued
Running
Replay Failed
Strategy Rejected
Prop Rejected
Robust Finalist
Development Exploratory Representative
Superseded
Blocked
```

Every status is rendered as:

```text
glyph + visible word/phrase + high-contrast color
```

Color is never the sole carrier of meaning.

### 5.5 Identity display

- Full SHA-256 identities are copyable through `st.code` or an equivalent copy-safe block.
- Short IDs are display-only and never accepted as authoritative input.
- Long IDs wrap or truncate visually without losing the copyable full value.
- Display names are mutable catalog annotations; they never replace technical identity.

### 5.6 Metric metadata, section roll-ups, helper text and glossary (UI-3)

- **Metric metadata registry** (`ifvg/presentation/metric_registry.py`, plan §6.2): one `MetricSpec` per displayed technical key — human name, definition, formula or persisted source, unit, directionality (from `OBJECTIVE_DIRECTIONS` wherever the metric is a charter objective) and the reference the value is read against. References exist only where the code registers them: the selected resolved gate (strategy / prop / robustness thresholds), the prevalence-reference Brier, the 0 skill boundary, the 0.5 chance line of AUC (a direction only, never a band), the calibration targets 1 / 0 (the distance only), the sample-adequacy minimums the contracts stamp, the persisted report `limits`, the measured access counters (0 PASS, nonzero FAIL) versus the policy-enforced `protected_*` zeros (informational), and a 95 % interval that crosses zero (INCONCLUSIVE). Missing, non-numeric or unevaluated evidence is UNAVAILABLE — never PASS. Screens call `describe(technical_key)` and `evaluate_metric` / `evaluate_interval` / `evaluate_gate_flag`; a metric card renders human name · value · status chip · interpretation (reference, sample, caveat) · the "why" help.
- **Section roll-ups** (`rollups.py`, plan §6.3): one status, one sentence, the main reason and the next thing to inspect per section (Data integrity, Strategy quality, Probability skill, Calibration, Stability, Prop feasibility, Robustness, Authorization readiness, Capacity and performance) under the fixed rule FAIL → BLOCKED → INCONCLUSIVE → WARNING (cautions or proposed thresholds) → PASS → INFORMATIONAL (no gate applies) → UNAVAILABLE (no evidence).
- **Helper text and glossary** (`help_registry.py`, plan §6.4): a `HelpEntry` per control id (what it changes, why, default, whether it mints a new identity, the work it requires, owner approval, availability) rendered by `help_text`; `help_for_metric` renders a registry row; the glossary defines FVG, IFVG, FSM, HTF, LTF, OOS, PIT, MBP-1, R multiple, Brier, Brier skill, AUC, EQH/EQL, MFE/MAE, AMI and Q-40 (`glossary_expander`). The source scan requires every widget call of every UI script to carry `help=` or to be a registered, justified, LIVE `HELP_EXEMPTIONS` entry (self-explanatory navigation only); every collapsed label carries an accessible name of at least two words plus help.
- **Human labels and availability chips** (`labels.py`, plan §6.5): human names for profiles, bundles, blocks, objectives, model protocols, regime algorithms, stamps, statuses, roles, comparison classes, feature tiers and verdicts — the technical key is always recoverable and never replaced; the availability chips make implemented / planned / proposed / ratified / research-only-offline / blocked / experimental / superseded states distinct by glyph and word.

---

## 6. New Study modes

Offer exactly these five modes:

```text
Single Configuration
FSM Configuration Search
Prop Benchmark
Universal Prop Search
Full Pipeline Run
```

### 6.1 Single Configuration

Evaluate one exact frozen strategy profile.

### 6.2 FSM Configuration Search

Search a small, registered, owner-authorized set of strategy configurations. Every occupancy/order-changing child uses a full sequential replay or verified reuse.

### 6.3 Prop Benchmark

Apply selected prop contracts, account/risk/withdrawal/replacement policies, and simulation protocols to one frozen executed-strategy stream.

### 6.4 Universal Prop Search

Evaluate one strategy configuration across multiple firms with per-firm account policy sets while preserving one universal strategy profile.

### 6.5 Full Pipeline Run

Configure the standardized pipeline under one of:

```text
Verification Fixture — maximum five authorized real trading days
Full Authorized Development Data — explicit post-acceptance operator action
```

The distinction is prominent before any control. Verification is functional proof only. Full authorized development is never launched by tests, import, startup, catalog load, or implementation verification.

---

## 7. Wizard-wide behavior

The New Study page uses an eight-step wizard:

```text
1. Objective
2. Baseline
3. Strategy Search Space
4. Prop Contracts
5. Risk Policies
6. Benchmarks
7. Validation
8. Review & Launch
```

**UI-1 amendment (run purpose; plan §5.6):** a goal card is fixed at the top of every step showing the draft's purpose, the derived run scope, the semantic namespace class and the store's verified `store_namespace_id` (or its typed non-verified state), the evidence class (synthetic fixture / real), the authorization class the actual computation path requires with its TYPED readiness, the stage-plan policy, publication availability and the result label. `Start new draft` (and every Start card) records the selected purpose as a mutable, presentation-only `RunPurposeAnnotation` on the draft; a legacy draft whose run scope does not derive one unambiguous purpose is `purpose_unresolved` and cannot freeze or launch until the owner confirms the purpose in the goal card. The annotation never enters a charter or pipeline identity and cannot bypass the `RunScope`, namespace, stage-plan or authorization validators.

**UI-2 amendment (goal-derived flows, plan §5.4; drafts, owner Q2):** the fixed eight-step sequence is superseded by the GOAL-DERIVED flow (`presentation/flows.py`) — Evaluate one configuration → Goal · Configuration · Strategy gates · Validation · Review; Compare with the baseline → Goal · Baseline · Challenger · Strategy gates · Validation · Review; FSM search → Goal · Baseline · Search axes · Strategy gates · Validation · Review; Prop feasibility → Goal · Strategy source · Firm contracts (≥ 1 verified) · Risk / payout policies · Prop benchmarks · Validation · Review; Universal → Goal · Baseline · Strategy search · Firm contracts (≥ 2) · Policies · Universal gates · Validation · Review; Feature / model evidence → Goal · Cohort anchor · Strategy gates · Validation · Review & Configure; the advanced end-to-end study → every step; the REAL verification slice → Goal · Exact baseline · Validation · Review (a synthetic fixture keeps its study family's flow). Skipped steps are listed on the goal card with their reason, never rendered empty, and contribute nothing to satisfiability or charter assembly; a selected prop objective keeps the contract step in a strategy goal's flow (§11). Drafts: `Start new draft` and every Start card create a SESSION draft — no file is written until the first explicit Save Draft or the first valid Next; afterwards every change autosaves with a visible `Saved` / `Autosaved` / `Not saved yet` chip; the draft name is required before the first persistence (default `<goal> — <baseline short name> — <date>`); an identical persisted goal / baseline draft raises a warning, never a block; the draft records its flow step key for exact restore; an archived draft cannot be edited until restored from History (§29). The global rules below read "step index" as the position in the flow.

Global interaction rules:

- the step position within the goal-derived flow (persisted with its step key);
- `st.progress` plus a visible breadcrumb/step heading;
- validation-gated Next and Back;
- Back never discards valid data;
- invalid fields show actionable, sanitized messages adjacent to the field;
- drafts are persisted on disk in the mutable draft namespace from the first Save Draft or the first valid Next (session-only before);
- autosave occurs on every change of a persisted draft and on every successful Next transition;
- **Save Draft** remains visible on every step;
- History lists drafts separately from immutable frozen runs;
- reopening a draft restores the exact step and fields;
- `Clone as New Search` deep-copies any frozen charter or completed search into a new mutable draft;
- the original frozen object remains immutable;
- launching occurs only inside the explicit launch-button handler;
- no render, import, AppTest, or page refresh launches a process.

---

## 8. Wizard step 1 — Objective

### 8.1 Research question

The user first selects one of (**UI-1 amendment**, owner Q4 — Evaluate is a separate task from Compare):

```text
Evaluate one configuration
Compare one configuration with the baseline
Find a robust FSM configuration
Test repeat-payout feasibility
Find one strategy configuration across multiple firms
```

`Evaluate one configuration` resolves exactly one profile (no comparison or delta claim); `Compare one configuration with the baseline` resolves exactly one baseline plus exactly one challenger configuration (a challenger may carry several registered differences as ONE configuration); both use the single-configuration study family internally.

### 8.2 Objective templates

Offer:

```text
Payout Reliability
Maximum Expected Payout
Low Breach / Long Account Life
Balanced Prop Performance
Strategy Quality Only
Custom
```

Every template visibly resolves and exposes:

```text
primary objective
hard constraints
tie-breaks
```

Resolved thresholds may not be hidden behind the template name. The objective policy contains no hidden weighted score.

### 8.3 Mode compatibility

The page validates that the selected question/template is compatible with the chosen study mode. Incompatible combinations are explained before Next is enabled.

---

## 9. Wizard step 2 — Baseline

Show a human-readable baseline card containing:

```text
Profile
Entry thesis
Direction
Target / label family
FSM concurrency
Development date range
Status
```

Provide a `View Technical Identity` expander with:

```text
Copy Profile Hash
Copy Core Replay ID / Artifact IDs where available
Resolved Strategy-Core and Quant-Lab source identities
Section/config hash
Seed and date-policy references
```

The user selects only a capability-eligible baseline. Blocked, experimental, superseded, or unavailable profiles render a reason and no launch path.

Generated children are not presented as fixed M0–M3 profiles; they are displayed as baseline diffs with their generated-profile capability state.

---

## 10. Wizard step 3 — Strategy Search Space

### 10.1 Axis groups

Organize cards by market meaning:

```text
Staleness
Parent Handling
HTF Selection
Causality & Locality
Entry Timing
Session Policy
Risk Admissibility
```

### 10.2 Card contents

Every card shows:

```text
human label
technical key
baseline value
registered search values
classification
market meaning
existing evidence
capability / authorization status
artifact and profile impact
```

### 10.3 Visible classifications

Use visible labels:

```text
Locked Invariant
Search Axis
Measured Only
Blocked
Experimental
```

Locked and blocked axes render **no input widget**. Their values and reasons remain visible for audit.

### 10.4 Computation-path chips

Every editable or filterable dimension displays one of:

```text
Requires New Sequential Replay
Analysis Filter
Feature Rematerialization
Model Refit
Prop Resimulation
```

The primary user-facing distinction required everywhere is:

```text
Analysis Filter
Requires New Sequential Replay
```

### 10.5 Exact registered values only

- No free-form `section_overrides` control exists.
- No raw JSON/YAML editor exists.
- Values come from the typed axis/value registry.
- Owner-ratification and generated-profile capability are shown before launch.
- Parent-fill axes remain visible as blocked with `blocked_pending_owner_policy_review`; no widget is exposed.
- Session/direction changes invoke the interpretation selector:

```text
Descriptive Slice
Specialized Model
Sequential Strategy Profile
```

Only `Sequential Strategy Profile` creates an executable counterfactual child.

### 10.6 Evidence expander

Each card has an evidence expander showing applicable audit counts, current baseline behavior, known caveats, and the exact owner-decision reference or missing decision.

---

## 11. Wizard step 4 — Prop Contracts

Each selected firm/account card displays:

```text
firm
account type / phase scope
account size label
contract version
effective date
verification status
drawdown rule
daily rule
contract limits
payout rules
fees
post-payout behavior
source/evidence status
minimum path capabilities required
```

Rules:

- `synthetic_fixture_verified` is visibly synthetic and cannot be mistaken for a real contract.
- A real study requires `first_party_verified` plus the required owner evidence.
- Stale contracts render a warning and cannot launch under a policy that forbids stale evidence.
- Unverified/blocked contracts render no launchable checkbox.
- Contract cards show the exact evidence bundle and supersession state in Audit disclosure.
- Assumed path scenarios are not part of the firm contract; they are selected later under the simulation policy.
- **UI-2 (plan §7 / F-04):** when the selected objective (template or custom) carries a prop metric, this step stays in the flow whatever the goal; at least one `first_party_verified` contract is mandatory and its absence blocks Next / Freeze with the explicit contract-workflow action (owner decisions 5 / 6). The selected objective is never removed or rewritten. Strategy-only goals skip this step and Risk Policies with the visible reason "no prop objective".

---

## 12. Wizard step 5 — Risk Policies

Support templates:

```text
Fixed Dollar
Percent of Starting Buffer
Percent of Current Buffer
NQ/MNQ Adaptive
Custom Contract Count
```

Display the hierarchy:

```text
Universal Strategy Profile
    ├── Firm A Account/Risk/Withdrawal/Replacement Policy Set
    ├── Firm B Account/Risk/Withdrawal/Replacement Policy Set
    └── Firm C Account/Risk/Withdrawal/Replacement Policy Set
```

Requirements:

- firm-specific policies do not alter strategy identity;
- every result-changing account policy is visible and enters the resolved simulation identity;
- UI caps and defaults follow `OWNER_DECISIONS.md` without implying research ratification;
- invalid minimum-contract/buffer combinations explain whether a trade will be skipped or the policy is blocked;
- withdrawal behavior is edited separately from the firm’s permitted payout contract;
- copied accounts visibly share the same resampled market/trade path per simulation path.

---

## 13. Wizard step 6 — Benchmarks

Show three clearly separated groups in this order:

```text
Underlying Strategy Gate
Prop Feasibility Gate
Robustness Gate
```

Every threshold displays:

```text
technical metric
human explanation
resolved value
unit
status: owner-ratified | proposed_protocol_default | not required
```

The UI must explain:

- why a configuration failed;
- which later gates were not run because an earlier gate failed;
- that verification fixtures use `verification_control_flow_gates_v1`, not research thresholds;
- that frontier/ranking occurs only after feasibility gates;
- that no hidden weighted score is used.

---

## 14. Wizard step 7 — Validation

### 14.1 Fields

Display:

```text
run scope: verification_5d | full_authorized_development
authorized development dates
real-date count
warmup
protected buffer
sealed boundary
search algorithm
outer-fold protocol
block-bootstrap protocol
stress paths
seed
estimated runtime
estimated storage
worker limit
authorization requirement checklist
```

Protected and sealed values are read-only.

**UI-1 amendment (§14 as a whole):** the run scope is DERIVED from the goal card's purpose and displayed read-only — the "Run scope" radio is gone. Implementation Verification chooses the evidence class on this step: a synthetic fixture (the typed `SyntheticAuthorizationMarker`; proves the machinery, never evidence) or the real ≤5-day slice (the owner's validated `VerificationAuthorizationRef`). The readiness of the exact `VerificationAuthorizationRef` is TYPED and rendered as such (`ready` / `missing` / `stale_head` / `wrong_head` / `superseded` / `wrong_namespace` / `wrong_profile` / `wrong_source` / `not_effective` / `store_unmarked` / `store_corrupt` / `store_incoherent` / `unavailable`), never collapsed into a Boolean. Development scopes show the frozen ten-date warmup prefix read-only and validate every evidence date field-by-field against the backend logical-day calendar (weekends, registered closures, the protected buffer and the sealed range refuse at the field). No worker control exists: the step states `Execution mode: sequential_children_v1 — effective workers: 1`.

### 14.2 Verification scope

For `verification_5d`:

- real-date count must be `<= 5`;
- warmup + evidence must be `<= 5`;
- the one canonical program-wide allowlist is displayed read-only;
- a downloadable resolved allowlist is available;
- a non-dismissible badge says:

```text
VERIFICATION ONLY — not research evidence
```

- research interpretation and catalog activation are prohibited;
- the UI shows whether the exact `VerificationAuthorizationRef` exists;
- absent authorization blocks the real run before source-path construction but does not block code-authoring/synthetic states.

### 14.3 Full authorized scope

For `full_authorized_development`:

- the user explicitly selects the scope;
- the UI displays every authorized date or provides the complete downloadable resolved allowlist;
- the owner-authorization requirements for the selected stage plan are shown;
- no full run starts during tests, import, startup, or page render;
- launch requires a frozen pipeline specification and the second confirmation in Step 8.

### 14.4 Verification Center (UI-2)

The `Verify Implementation` route is the Verification Center (plan §5.3). Its six sections render every TYPED backend state — never a Boolean "present" — and the center has no spawn seam:

1. **Purpose & readiness** — a sticky card: semantic store namespace, logical-day fixture, physical partition coverage, seed-production authorization, seed job, verified seed, final verification authorization (`VerificationAuthorizationRef`), bounded-run readiness; `Execution mode: sequential_children_v1 — effective workers: 1`; the 21/R-5 requirement; the synthetic-fixture draft entry (a machinery proof, never evidence).
2. **Fixture** — the shortlist document (`logical_window_shortlist_v1`) re-validated through its contract (missing / corrupt are typed states with the exact rebuild command); the required entries and the ranking trace; the owner records ONE eligible window as PROVISIONAL (the document keeps `owner_selection = NOT PERFORMED`; nothing registers an allowlist); the LOGICAL trading days (session bounds, coverage counts) and their PHYSICAL partitions (td−1 `prev_utc_date`, td `utc_date`, source kind, content sha256 from the accepted inventory) are separate tables; the protected buffer and the sealed range are read-only exclusions; the coverage matrix id derives from the shortlist rows.
3. **Seed** — the UNSIGNED seed-production packet (the canonical store-day chain from 2026-01-01 through the day before the window; permitted and prohibited outputs; owner placeholders that fail validation) prepared in-app or by the CLI; the owner's registration through `scripts/ifvg_seed_production.py register-authorization` (external; its `--receipt-out` receipt is picked up on refresh); the persisted authorization verified through the backend (`verified` with the accepted inventory and the code identities; `verified_envelope` when the inventory is unavailable — explicitly not the full check; every refusal keeps its typed reason); the exact seed job command (`run … --receipt-out …`) is shown ONLY for a verified authorization; the verified seed through the exact-id, profile-bound loader (continuity with the window; the run receipt's provenance and zero verification-evidence footprint).
4. **Final authorization** — the UNSIGNED final packet, buildable only after a verified seed; the owner's completed reference is read from the named file and validated typed (file missing / unsigned / malformed / wrong namespace / stale head / superseded / seed mismatch / allowlist mismatch / valid) and never persisted by the center.
5. **Review & run** — the exact baseline, allowlist and partitions, seed, dual drive, release-control checks, artifacts, execution mode; with a valid reference the action `Freeze the verification charter and register the run` freezes the exact-baseline charter whose owner bundle derives from the SIGNED reference's own content hash (never a run id, never a synthetic marker), persists the pipeline spec and the `VerificationRunEnvelope` (verified reuse on repeat); with a registered run the §6.1 preflight runs (typed; refusals named) and, ONLY when it passes, the exact `scripts/ifvg_bounded_verification.py run` command is shown. Nothing launches from this page.
6. **Monitor** — the seed stage and the stages of the resolved verification plan ONLY (stages outside the plan are never listed), the attempt's execution mode, access counters when persisted (never assumed zero), the evidence folder's report flags, the exact-verifier note. No Publish route exists.

---

## 15. Wizard step 8 — Review & Launch

Show the complete resolved charter and estimated work, including:

```text
study mode and objective
baseline and resolved technical identities
changed and frozen dimensions
owner-authorization checklist
unique strategy profiles
full sequential replay count
verified reuse hits
firm/risk/account-policy combinations
historical/scenario prop replay count
bootstrap/stress simulation count
estimated runtime and storage
new artifacts expected
blocked or planned capabilities
```

Clearly distinguish:

```text
Expensive strategy replays
Cheaper downstream prop simulations
```

**UI-1 amendment (satisfiability, readiness, honest launch):** a `Charter satisfiability` card (PASS / FAIL per named rule — FSM search ≥ 1 challenger; Evaluate exactly one profile; Compare exactly one challenger configuration; a selected prop objective requires a verified firm contract and is never rewritten; Prop Benchmark ≥ 1 firm; Universal ≥ 2 firms; real verification excludes research gates and is the exact baseline; a synthetic fixture is confined to Implementation Verification; every objective registered) precedes the primary action and a FAIL disables it; the typed authorization readiness of the actual path is shown and the action stays disabled while it is not `ready` (the typed acknowledgement alone never enables a Full Authorized Development freeze). The service validator enforces the identity-bearing subset of these rules again at freeze.

Primary action:

```text
Freeze Search Charter and Launch
```

After freeze:

- the charter is immutable, saved into the PURPOSE's store, and its purpose is recorded as a mutable catalog annotation;
- the draft remains only as historical provenance or is marked frozen;
- research-bearing changes require `Clone as New Search`;
- the handler resolves the REGISTERED executor before any spawn (an unregistered key is the typed `runner_unavailable` state — nothing is spawned, nothing is reported as launched), spawns the detached job, and reports the launch as started ONLY after the worker's persisted state exists (otherwise the typed `launch_not_started` state with the job-log location and the CLI fallback); routing to Active Runs happens only on a started launch.

Full authorized scope requires a second confirmation with the exact warning:

```text
This will run the full authorized development pipeline.
It is not an implementation verification run.
```

The user must type the required study/pipeline name or acknowledgement phrase before the launch control is enabled.

---

## 16. Active Runs

### 16.1 Polling model

Use `st.fragment(run_every="5s")` around a plain, AppTest-callable body function that reads atomic JSON status files. Provide a manual Refresh fallback.

### 16.2 Parent progress

Show a phase checklist and counts for:

```text
Profiles Generated
Replays Completed
Strategy-Gate Passes
Prop-Feasible Configs
Robust Finalists
```

### 16.3 Clickable funnel

Render five keyboard-operable buttons:

```text
Generated · N
Replay Valid · N
Strategy Pass · N
Prop Feasible · N
Robust · N
```

A Plotly funnel may accompany the buttons but is not the sole interaction. Clicking a stage filters the child table.

### 16.4 Child table

Use an indexed, paged table with columns:

```text
Config
Replay
Strategy Gate
Prop Simulation
Robustness
Status
Human Explanation
```

Stages not run because of an earlier failure display:

```text
Not run — strategy gate failed
```

or the exact applicable earlier-gate reason.

### 16.5 Row detail and actions

The selected child shows:

- human configuration name and changed parameters;
- core replay/membership/companion identities;
- sanitized failure text;
- gate evidence;
- artifact references;
- `Open in Results`;
- `Open Replay` / exact verifier action;
- attempt history where relevant.

### 16.6 Safe cancel

`Request Safe Cancel` requires confirmation and writes the sentinel. Cancellation occurs only at a declared safe boundary. Completed children remain immutable and reusable.

### 16.7 Missing status fallback

When a status file is missing or unreadable:

- show a sanitized state, not a traceback/path;
- provide the CLI escape hatch with the exact command form and no secret/local-path disclosure beyond the approved project-relative command;
- preserve the frozen study identity.

---

## 17. Results common frame

Every Results view contains:

- search/run picker by full identity with a human display name;
- `Summary | Analyst | Audit` disclosure control;
- persistent development-only badge;
- explicit result-scope caption;
- gross/costed/net selector or labels where applicable;
- current selected configuration control;
- exact evidence and technical-identity access.

All figures are pure builders. They honor existing `LAYER_BUDGETS`, produce an `OmissionReport` when data is truncated, and use the repository display-timezone helper.

**UI-3 amendment (§17):** the common frame's disclosure control is the three-tier `detail_levels` selector (§5.1); the Overview is followed by a **Selected configuration** block — the Strategy quality and Prop feasibility roll-ups (§5.6) and the configuration's registry metric cards read against the charter's RESOLVED gates (the strategy metrics; the prop vector as the worst value across the simulated firms, D-#18) with the proposed-threshold caveat; the heatmap and firm-matrix metric pickers caption the picked metric's registry definition and direction; the explorer carries a **Column guide** describing every preset column from the registry (schema-reserved columns say so).

---

## 18. Results overview

The first view answers:

1. Did any configuration pass?
2. Which is the Development Exploratory Representative?
3. Which maximizes expected payout?
4. Which maximizes payout reliability?
5. Which minimizes breach risk?
6. Why do they differ?

Summary cards:

```text
Development Exploratory Representative
Highest Expected Payout
Highest Payout Reliability
Lowest Breach Risk
```

These are ranking-dimension titles, not a declaration of a universally “best” strategy.

If none pass, render exactly:

```text
No configuration passed all benchmarks.
```

Then show dominant failure reasons and the number stopped at each gate.

---

## 19. Payout-reliability frontier

Primary chart:

```text
x-axis: expected 90-day net payout
y-axis: probability of at least one payout per 30 days
color: 90-day breach probability
size: expected account lifetime
```

Requirements:

- selecting a point updates the complete page;
- an always-present configuration selectbox is the accessible/AppTest twin;
- hover/detail includes firm/account-policy context and uncertainty/evidence scope;
- infeasible/blocked points are not silently plotted as feasible;
- selected configuration remains synchronized with the explorer and comparison surfaces.

---

## 20. Parameter-sensitivity heatmap

Provide row-axis, column-axis, and metric pickers.

Supported metric families include:

```text
net E[R]
trade count
maximum drawdown
expected payout
breach probability
payout reliability
```

Every cell has a non-color glyph class:

```text
stable plateau      ◼
knife-edge point    ▲
failed region       ✕
insufficient data   ·
blocked cell        ⊘
```

A table twin exposes the same cell status, value, sample count, and evidence reason.

---

## 21. Firm compatibility matrix

- Rows: strategy configurations.
- Columns: firms/account contracts.
- Metric toggle:

```text
P(3 payouts before breach)
expected payout
breach probability
first-payout probability
```

Universal versus firm-specific configurations must be visually obvious. Cells show missing/blocked/unsupported reasons rather than zero.

---

## 22. Account survival curves

For a selected strategy configuration, plot account-survival probability over trading days with one line per firm/account policy.

Use:

- step lines;
- distinct dash patterns in addition to color;
- a table twin for key survival horizons;
- explicit simulation mode/path-capability caption.

---

## 23. Payout distributions

Horizon selector:

```text
30-day
60-day
90-day
lifetime
```

Always display:

```text
mean
median
10th percentile
90th percentile
```

The 10th percentile is listed first in the lower-tail summary and is visually prominent. The chart and table state the simulation mode, path fidelity/capabilities, fees, withdrawals, replacements, and net/gross basis.

---

## 24. Configuration explorer

Provide presets:

### Strategy

```text
trade count
net E[R]
realized payoff ratio
profit factor
max DD R
trade frequency
setup occupancy
```

### Prop

```text
firm
risk policy
first payout
3 payouts
90-day survival
expected payout
Q10 payout
fees
replacement cost
```

### Robustness

```text
outer folds passed
stress tests passed
neighbor stability
worst-firm result
concentration warning
```

Sticky/pinned columns:

```text
rank
config name
status
changed parameters
```

Requirements:

- server-side sorting and indexed pagination;
- no complete child artifact load into Streamlit memory;
- long IDs remain copyable in row detail;
- human names derive from the baseline diff;
- only changed fields are displayed first;
- technical full diff remains available in Audit disclosure.

Example display names:

```text
60-Bar Parent Timeout
Same-Session Parent Expiry
60-Bar Timeout + Close-Aware S4 Fill
```

Names are examples of the naming algorithm, not authorization of blocked axes.

---

## 25. Baseline and arbitrary comparison

Render the dimension-diff ribbon before metrics. It displays:

```text
delta type
changed dimensions
frozen dimensions
computation path
compatibility
cohort identity
uncertainty method
development status
match_basis
```

Then provide four panels:

```text
Parameter Diff
Funnel Delta
Strategy Delta
Prop Delta
```

Funnel deltas include, where comparable:

```text
activations
locks
opposing selections
inversions
executed trades
setup-slot occupancy
terminal reasons
```

Rules:

- incompatible pairs render the parameter diff only plus an explicit incompatibility notice;
- `not_comparable` populations never claim common/added/removed membership;
- every delta row lists exact affected setup/trade/account-event IDs;
- every affected ID can open the exact verifier or account timeline;
- changing the reference configuration recomputes presentation from the stored compatible comparison contract, not an ad hoc nearest match.

---

## 26. Deterministic insight panel

Render persisted deterministic insight text verbatim under seven fixed categories:

```text
What Changed
Edge Effect
Prop Effect
Robustness
Concentration Warning
Evidence Quality
Recommended Inspection
```

Each insight includes evidence buttons and exact `EvidenceRef`s. Wording is causally neutral and shows sample size, concentration, compatibility, `match_basis`, and path-evidence limitations. An AI-written summary is out of scope and can never replace this panel.

---

## 27. Exact drill-down

From any eligible result, provide:

```text
Open Changed Setups
Open Additional Trades
Open Removed Trades
Open Breach-Causing Path
Open Payout Event
```

`queue_replay_drilldown` writes the existing verifier pair key plus an exact `queue_jump` supporting `setup_id`, candidate, decision, or trade identity as defined by the verifier contract.

Rules:

- no setup/time, nearest-time, row-order, keep-last, or fuzzy fallback;
- unresolved IDs terminate in a sanitized warning;
- no local filesystem path is shown;
- cross-tab navigation uses the existing session-state pattern and a visible toast/instruction when the top-level tab cannot be switched programmatically.

**UI-2 amendment (reviewer verdicts; owner Q3 / F-06):** the verifier's review form opens `Unreviewed` — a UI state only, never a ledger value: opening a case creates no ledger row and preselects no verdict. The labels are Correct → `correct`, Incorrect → `incorrect`, Unclear → `insufficient_evidence`, Needs investigation → `questionable`, Not applicable → `not_applicable` (the one additive `ifvg_visual_review_v1` key), each with its definition beside the control; widgets are keyed by the case so a verdict never carries across candidates or setups; nothing persists without the explicit `Save Review` (disabled until a verdict and a reviewer exist) and the persistence state is visible (`Unsaved` / `Saved`); existing v1 rows render with their labels and preselect nothing.

---

## 28. Account timeline

For exact historical/scenario account replay, show:

```text
balance
trailing threshold
daily-loss threshold
payout eligibility windows
payouts
fees
breaches
replacement events
phase transitions
```

Visual semantics:

- balance, trailing threshold, and daily-loss threshold are separate lines;
- payout-eligibility intervals are shaded;
- markers:

```text
payout       ▽
fee          ◇
breach       ✕
replacement  □
```

- marker shapes/labels supplement color;
- a synchronized event table is always available;
- clicking a marker or table row opens the linked trade/setup/path event;
- event ordering follows `PropAccountEventEnvelope`, not chart timestamp sorting alone.

---

## 29. History

History contains separate sections for:

```text
Drafts
Frozen / Running Studies
Completed Studies
Superseded Studies
Legacy Read-Only Results
```

Requirements:

- drafts may be opened, cloned or ARCHIVED — the normal, reversible action (archived drafts are hidden from the default listing and restorable from the archived view); permanent deletion exists only in the archived / advanced view, only for drafts never frozen and never launched, only with the exact typed draft name; frozen charters, attempts and runs are never deletable (the catalog archive flag only); the R4-era empty untitled step-0 drafts are archived by the one-time `Archive empty untitled drafts` action, never deleted (UI-2, owner Q2);
- the purpose / store-namespace-class filters are READ-ONLY listing filters: they narrow History and never change where a charter freezes or launches (UI-2);
- frozen/completed research artifacts have no delete or overwrite control;
- display names and notes are mutable annotations only;
- duplicate semantic configurations show verified reuse rather than a duplicate run;
- `Clone as New Search` creates a new draft;
- legacy results retain their original reports and mandatory caveats, with no rerun/modification/promotion controls;
- comparison actions are enabled only when the compatibility contract permits them.

---

## 30. Full Pipeline Run UI

The Full Pipeline Run surface is a standardized operator workflow with these phases:

```text
Configure
Preview
Launch
Monitor
Resume / Retry
Publish
```

**UI-1 amendment (§30 as a whole):** the pipeline surface derives its store from the draft's purpose (never from a selector); Configure and Resume / Retry carry no worker control and state `Execution mode: sequential_children_v1 — effective workers: 1` (HARDENING-BACKEND §4.6; the backend refuses any other value before a job exists); Launch shows the purpose, run scope, namespace class + verified id, evidence class and typed readiness, applies the satisfiability rules of §15, resolves the registered executor before any spawn and reports the launch only after the worker's persisted state exists; Monitor, Resume / Retry and Publish act on the selected run's OWN store (its charter located by exact id) and show the artifact-derived scope. The phase radio remains until UI-6 lands the state-driven lifecycle (plan §5.5).

### 30.1 Configure

Select:

```text
run scope
baseline profile or parent search
authorized date range / allowlist
feature bundle
label policy
fold protocol
model protocol
cost policy
prop contracts
account/risk/withdrawal/replacement policy sets
simulation protocol
worker/resource limits
selected stage plan
```

Rules:

- `verification_5d` always shows the mandatory non-dismissible verification badge;
- dates and protected/sealed boundaries are read-only;
- planned/blocked ML or feature entries are visible but disabled with status/reason;
- spectral/Nyström entries state that numerical fitting is post-V1;
- MBP-1 bundle is planned/refused before R5B and available as `research_only_offline` after R5B;
- capability-scoped stage plans do not require unused subsystems.

### 30.2 Preview

Show:

```text
resolved dates and count
planned stages
strategy child count
feature-view count
fold/model count
prop simulation count
runtime estimate by phase
storage estimate by phase
verified reuse opportunities
new artifacts expected
capability/authorization blockers
```

Estimates are operational annotations, not scientific identity.

### 30.3 Launch

Primary action:

```text
Freeze Pipeline Specification and Launch
```

Launch is executed only inside the explicit button handler. Full scope requires the exact second confirmation from §15.

### 30.4 Monitor

Render:

```text
overall progress
current phase
current date / child / fold
elapsed time
estimated remaining time
worker utilization
checkpoint status
reused stages
warnings
safe-cancel control
pipeline_semantic_id
execution-attempt history
```

Render all 16 pipeline stages with glyph + word states, including `reused`, `blocked`, `failed`, and `not required`.

**UI-3 amendment (§30.4):** the supervised-ladder table keeps AUC numeric (a nullable Float64) and carries the `auc_reason` token in its own **AUC reason** column, names the rung's model protocol, and captions the registry definitions of its columns. The MBP-1 and Regime Lane panels beneath the phases are summary-first: a readiness summary resolved from the registries and the SELECTED run's persisted evidence (the activated offline block and the order-flow bundles, the source / coverage readiness, the owner-ratification state, the controlled-study state and the fixed `research_only_offline` boundary; the active algorithm, the observation grain, the panel / fit / assignment coverage, the role / promotion status and why a model-bearing path is blocked or eligible, the planned post-V1 algorithms visible but disabled, the stamped defaults), then Research details, then **Advanced diagnostics** holding the manual exact-id inputs, the full registries, the stamps and the registry hashes. Nothing unresolved reads as passing; neither panel implies live serving or execution gating.

### 30.5 Resume and Retry

Allow:

```text
resume from verified checkpoint
retry a failed operational stage
clone with a changed resource policy
open sanitized logs and failure evidence
```

Rules:

- retrying the same semantic pipeline produces a new execution-attempt identity and reuses verified semantic outputs;
- changing a research-bearing field creates a new pipeline semantic identity;
- a resource-policy clone is visibly operational, not a new scientific configuration;
- completed immutable stages are never overwritten.

### 30.6 Publish

The state remains:

```text
prepared_not_published
```

until the user runs:

```text
Run Publication Gates
```

The UI shows the complete checklist. Only when every required gate passes does it enable:

```text
Publish and Activate Catalog Entry
```

Publication and activation are separate from preparation. Verification-only artifacts can never activate a research catalog entry.

**UI-1 amendment (namespace-bound publication; plan F-03):** the gates record the verified `store_namespace_id` of the store they ran under and the digest of the state they ran over; the UI's gate cache is keyed by (pipeline id, that namespace id, that digest); activation passes the namespace it believes it acts under and the backend refuses an activation under another namespace, over a changed state (a later attempt), or for an unmarked / corrupt store; the Publish surface shows the run's store namespace state read-only and disables activation while it is not verified.

---

## 31. Empty, blocked, and failure states

The UI intentionally renders all of these states:

```text
no configurations pass
no verified firm contract
blocked search axis
insufficient sample
child replay failed
prop simulation not run because strategy gate failed
no model result
artifact unavailable
protected-range refusal
verification authorization missing
feature block planned / unavailable
regime algorithm planned / unavailable
lineage not comparable
browser QA unavailable
```

**UI-1 amendment (plan §6.7):** the following states are additive and distinct — a "no runs", "not selected" or "not applicable" situation never renders as `artifact unavailable`, and a missing artifact is distinguished from a corrupt one:

```text
no runs (search / pipeline)
not selected
not applicable
not configured
interrupted — resume available
artifact missing
artifact failed verification (corrupt)
legacy read-only result
superseded
run purpose unresolved
no registered executor is available in this process (runner unavailable)
launch requested — no persisted state yet (launch not started)
owner authorization is not ready (typed readiness)
store namespace not verified
seed production not authorized
```

**UI-2 amendment:** the Verification Center and draft-lifecycle states are additive and distinct:

```text
verification window shortlist unavailable
no provisional verification window selected
no verified profile-matching seed
final verification authorization not signed
bounded-run preflight refused
this draft is archived
draft not saved yet (session only)
```

Each state includes:

- a concise human explanation;
- the owning gate/capability;
- the next valid action, where one exists;
- an Audit disclosure link to sanitized evidence;
- no raw traceback, secret, or local filesystem path.

Exact required no-pass text:

```text
No configuration passed all benchmarks.
```

---

## 32. Accessibility and responsiveness

### 32.1 Keyboard and screen-reader behavior

- Every form control has a visible label and useful help text.
- Navigation, wizard Next/Back, funnel filters, table selection twins, configuration selection, safe cancel, and verifier actions are keyboard operable.
- Charts never provide the only route to select a configuration or event.
- Statuses and heatmap classes use text/glyph semantics in addition to color.
- Focus order follows visual order.
- Dialogs have an inline-form fallback.
- Empty/failure messages are headings or clearly associated with their surface.
- Copyable IDs have descriptive labels.
- **UI-3:** every non-obvious control carries registry helper text (`help=`); the only exemptions are the registered self-explanatory navigation controls; every collapsed label has an accessible name of at least two words; every metric, roll-up and status is readable by glyph + word without its technical key (§5.6).

### 32.2 Required viewport QA

Interactive and screenshot QA is mandatory at:

```text
1440×900
1024×768
768×1024
390×844
```

### 32.3 Responsive behavior

- Desktop may use side-by-side controls and charts.
- Tablet stacks secondary panels while preserving the selected configuration and disclosure state.
- Mobile uses a single-column layout, collapsible technical identity, preset-based tables, and no unusable horizontal control row.
- Wide tables remain horizontally scrollable with selected-row details repeated outside the table.
- Large charts offer a table fallback and do not overflow the viewport.
- Long configuration names and hashes wrap/truncate visually without clipping action controls.

### 32.4 Visual evidence

Hardening stores screenshots for every required viewport and representative state:

```text
New Study step 3
Active Runs running state
Results overview with passing configs
Results no-pass state
Baseline comparison
Account timeline
Pipeline blocked verification state
Pipeline monitor
one mobile empty/failure state
```

Keyboard-only navigation evidence and any failed/unavailable browser backend are recorded. An unavailable browser backend leaves the gate open; it is never called passed.

---

## 33. Performance, pagination, and honest truncation

- Only the selected sub-surface executes.
- Active Runs reads atomic status summaries, not complete child artifacts.
- Child/configuration tables use indexed server-side sort and pagination.
- Results initial load does not materialize every child’s full evidence.
- Chart builders honor `LAYER_BUDGETS` and emit an `OmissionReport` describing omitted layers/rows.
- Results initial-load, page-filter, chart-render, and drill-down budgets in `TEST_MATRIX.md` remain binding.
- `st.fragment` does not rebuild unrelated charts.
- Caches are keyed by complete artifact/run IDs and manifest hashes; short/display IDs never key a cache.
- Cache entries are refused/invalidated on identity or manifest mismatch.

---

## 34. Error sanitization and safety

- All user-facing exceptions pass through the shared sanitizer.
- No local absolute path, secret, raw stack trace, connector token, or internal exception object is rendered.
- Blocked/protected/sealed requests fail before path construction and render the policy reason.
- The UI contains no `allow_sealed`, unlock, recapture, promote, order, live activation, or Trade-Lab serving control.
- Verification/full-scope controls never auto-trigger work.
- CLI escape-hatch commands are explicit owner actions and never execute merely because rendered.

---

## 35. Release ownership

### R4 — core trader workspace

R4 implements and accepts:

- information architecture and routing;
- shared UI primitives and presentation contracts;
- complete New Study wizard and drafts;
- Active Runs monitor;
- Results, History, comparison, insight, exact drill-down, and account timeline over R1–R3 fixtures;
- baseline accessibility/AppTest coverage.

### R5 — pipeline and capability readiness

R5 implements:

- Full Pipeline Configure/Preview/Launch/Monitor/Resume-Retry/Publish UX;
- supervised-ladder result presentation;
- planned/blocked MBP-1 and post-V1 regime states;
- capability-scoped stage-plan behavior;
- verification-only E2E UI control flow.

### R5B — MBP-1 activation UX

R5B adds:

- MBP-1 availability and coverage panels;
- bundle selection after versioned activation;
- Baseline versus Baseline+MBP-1 comparison;
- missingness/coverage evidence;
- exact stage-window drill-down;
- persistent `research_only_offline` labeling.

### R6 — V1 KMeans regime UX

R6 adds:

- KMeans regime coverage, occupancy, stability, assignment, and stratification views;
- proposal/default stamps;
- sample-adequacy blocked states;
- context-panel grain identity;
- post-V1 algorithms visible only as planned/disabled entries.

### Hardening

Hardening closes all viewport, keyboard, screenshot, responsive, empty/failure, long-ID, wide-table, performance, and exact-drill-down gates in this contract.

### UI-1 — semantic purpose, namespace and authorization truth (2026-09-04)

UI-1 implements (plan Phase 1): the presentation-only run purpose with the derived scope / namespace / evidence / authorization class; the removal of the namespace selector; the `Start` and `Verify Implementation` routes; charter satisfiability before freeze (no silent objective rewrite); authorization by the actual computation path with typed readiness; the honest launch outcome; namespace- and state-bound publication; the sequential-V1 runtime truth; backend-derived development date validation; direction-aware heatmap / firm-matrix colorscales; reconciliation banners derived from evaluated gates; the additive §31 states; artifact-derived verification badges.

### UI-2 — the Verification Center, goal-derived flows, safe drafts and explicit reviews (2026-09-04)

UI-2 implements (plan Phase 2): the complete Verification Center (§14.4) with no spawn seam; the goal-derived conditional flows with visible skip reasons and the prop objective that blocks instead of disappearing (§§7, 11); the session-only draft lifecycle with archive / restore / typed permanent delete, the bulk archive of the empty untitled drafts and the read-only History filters (§§7, 29); the explicit reviewer verdicts (§27); the seed CLI seams `register-authorization` / `--receipt-out`; the additive §31 states. UI-3 … UI-6 own the metric / help system, the Replay / Verifier redesign, Data & Audit, and the state-driven lifecycle with the mandatory browser acceptance.

### UI-3 — metric and help system; research results; MBP-1 / regime presentation (2026-09-04)

UI-3 implements (plan Phase 3): the metric metadata registry, the deterministic section roll-ups, the helper-text registry with the glossary and the explicit live exemptions, the human-label registry with the distinct availability chips (§5.6); the three detail levels on every major screen (§5.1); the decision-summary-first Context Research presentation (the four roll-ups, the sample-adequacy card, registry metric cards with their references, the named reliability diagonal, coverage and net R on separate axes, fold validity chips, intervals that cross zero as inconclusive, the top-N importance with fold stability, run-compatibility reasons in words, raw JSON only under Technical identity & audit — F-07 full, F-09); the Results presentation (§17); the ladder frame and the summary-first MBP-1 / regime panels (§30.4); help on every widget of every UI script (F-11 part). UI-4 … UI-6 own the Replay / Verifier redesign, Data & Audit, and the state-driven lifecycle with the mandatory browser acceptance.

---

## 36. Normative frontend acceptance matrix

| ID | Surface | Required proof |
|---|---|---|
| FUX-IA-001 | IFVG Lab | top-level order remains `Experiments`, `Replay / Verifier`, `Data & Audit` |
| FUX-IA-002 | Experiments | horizontal sub-nav contains New Study / Active Runs / Results / History / Context Research; only selected route executes |
| FUX-IA-003 | Context Research | the M0–M3 computation lane delegated unchanged; the presentation follows §§5.1 and 5.6 (UI-3) |
| FUX-MOD-001 | Architecture | pure logic in `src`, thin scripts, scripts included in Ruff |
| FUX-WIZ-001 | Wizard shell | eight steps, progress/breadcrumb, validation-gated Next/Back |
| FUX-WIZ-002 | Drafts | autosave on Next, always-visible Save Draft, restore exact step, History listing |
| FUX-WIZ-003 | Clone | frozen/completed study clones to new mutable draft; original unchanged |
| FUX-WIZ-004 | Objective | five modes, four questions, six templates, all thresholds/tie-breaks visible |
| FUX-WIZ-005 | Baseline | human card plus full copyable technical identity |
| FUX-WIZ-006 | Search space | exact axis groups/card fields; locked/blocked have no widgets; computation-path chip and evidence expander |
| FUX-WIZ-007 | Authorization | real scope fails closed on missing required evidence; unrelated decisions not demanded |
| FUX-WIZ-008 | Prop | full contract cards; synthetic/stale/unverified states truthful and blocked as required |
| FUX-WIZ-009 | Risk | universal strategy with per-firm account-policy hierarchy |
| FUX-WIZ-010 | Benchmarks | three ordered gate groups; skip reasons visible |
| FUX-WIZ-011 | Validation | read-only boundary/date fields; canonical allowlist; verification badge; full-scope distinction |
| FUX-WIZ-012 | Review | replay-vs-prop work preview, reuse hits, exact freeze/launch semantics, typed full-scope confirmation |
| FUX-MON-001 | Active Runs | five-second fragment plus manual fallback, phase checklist, keyboard funnel buttons |
| FUX-MON-002 | Child table | exact columns, pagination, human reasons, skipped-stage copy |
| FUX-MON-003 | Child detail | sanitized evidence, identities, results/replay actions, attempt history |
| FUX-MON-004 | Safe cancel | confirmed sentinel; safe boundary; completed children immutable |
| FUX-RES-001 | Common frame | search picker, disclosure, dev badge, scope and gross/cost/net labels |
| FUX-RES-002 | Overview | exact four cards and exact no-pass sentence plus dominant failures |
| FUX-RES-003 | Frontier | exact axes/encodings; selection sync; accessible selectbox twin |
| FUX-RES-004 | Heatmap | axis/metric controls; glyph classes; table twin; no color-only meaning |
| FUX-RES-005 | Firm/survival/payout | metric toggles, dash/step semantics, horizons, P10 prominence |
| FUX-RES-006 | Explorer | exact presets, sticky columns, indexed sort/pagination, baseline-diff names |
| FUX-RES-007 | Comparison | dimension ribbon, four panels, incompatibility suppression, exact affected IDs |
| FUX-RES-008 | Insights | seven categories, exact evidence, causally neutral text |
| FUX-RES-009 | Timeline | required lines/windows/markers/table, total event order, linked drill-down |
| FUX-DRILL-001 | Verifier | exact IDs only; no time/setup/fuzzy fallback; sanitized unresolved state |
| FUX-HIST-001 | History | drafts and immutable run classes separated; no delete/overwrite for frozen evidence; clone supported |
| FUX-PIPE-001 | Configure | complete grouped fields, capability-scoped stages, disabled planned entries |
| FUX-PIPE-002 | Preview | exact counts, phase estimates, reuse and new-artifact disclosure |
| FUX-PIPE-003 | Launch | button-handler only; exact second confirmation; no automatic full run |
| FUX-PIPE-004 | Monitor | exact progress fields, 16 stages, semantic ID and attempt history |
| FUX-PIPE-005 | Resume/Retry | checkpoint reuse, operational retry identity, research change creates new semantic ID |
| FUX-PIPE-006 | Publish | prepared_not_published; gates first; explicit separate activation; verification cannot activate |
| FUX-STATE-001 | Empty/failure | every state in §31 rendered intentionally and sanitized |
| FUX-A11Y-001 | Keyboard | all primary paths complete without pointer-only chart interactions |
| FUX-A11Y-002 | Semantics | status and heatmap meaning never color-only; visible labels/help |
| FUX-A11Y-003 | Viewports | 1440×900, 1024×768, 768×1024, 390×844 screenshots and responsive assertions |
| FUX-A11Y-004 | Fallbacks | fragment/on-select/dialog/pinned-column fallbacks preserve behavior |
| FUX-PERF-001 | Performance | indexed pagination, bounded layers, omission reports, existing UI budgets pass |
| FUX-SAFE-001 | Safety | no raw path/traceback/secret; no sealed/live/order/promotion controls |
| FUX-LABEL-001 | Truthfulness | Development Exploratory Representative; match_basis; scenario/approximation; explicit result scopes |
| FUX-UI1-001 | Purpose | one presentation-only purpose derives run scope, namespace class, evidence class, authorization class and publication availability; the annotation round-trips and never enters an identity; ambiguous legacy drafts are `purpose_unresolved` and cannot freeze |
| FUX-UI1-002 | Namespace | no namespace radio on any route; the `store_namespace_id` comes only from the verified envelope; a local path never defines authority; listings and badges derive from the artifact's own store |
| FUX-UI1-003 | Satisfiability | zero-axis search, baseline-vs-itself, Compare without exactly one challenger, prop objective without a verified contract, one-firm Universal, and verification with research gates fail BEFORE freeze with their reason; the selected objective is never rewritten |
| FUX-UI1-004 | Authorization | a fully synthetic fixture carries the typed marker; real verification refuses it; research scopes require the computation-path-scoped owner bundle bound to the store's verified namespace and current head; readiness is typed (missing / stale head / superseded / wrong namespace / wrong head / wrong profile / wrong source / store unmarked / corrupt / ready) |
| FUX-UI1-005 | Honest launch | an unregistered runner is refused before any spawn; a launch is reported as started only after the worker's persisted state exists |
| FUX-UI1-006 | Publication | gates and activation are bound to the same verified store namespace and state digest; cross-namespace or stale-state activation refuses |
| FUX-UI1-007 | Runtime truth | no worker control above one; `sequential_children_v1 · effective workers 1` stated on Validation, Configure, Resume and Monitor |
| FUX-UI1-008 | Direction | minimize metrics (drawdown, breach probability) use the reversed colorscale with the direction labelled |
| FUX-UI1-009 | Evidence truth | the reconciliation banner is green only for evaluated passing gates; unevaluated / default-only evidence renders UNAVAILABLE, never PASS |
| FUX-UI1-010 | States | no-runs / not-selected / not-applicable never render as artifact unavailable; missing and corrupt artifacts are distinct |
| FUX-UI2-001 | Verification Center | no research gate, prop / risk / model control or Publish route; no spawn seam; the sticky readiness card renders every typed state |
| FUX-UI2-002 | Fixture | logical trading days and physical partitions displayed separately; the owner's selection is provisional; `owner_selection` stays NOT PERFORMED and no allowlist is registered |
| FUX-UI2-003 | Seed | no seed job command without a verified `SeedProductionAuthorizationRef`; the packet is unsigned; receipts are picked up by exact id and verified |
| FUX-UI2-004 | Final authorization | the final packet cannot exist before a verified profile-matching seed; the completed reference validates typed and is never persisted by the center |
| FUX-UI2-005 | Review / run | no bounded-run command without a validated final reference, a charter frozen from the SIGNED reference, a registered run and a passed preflight; the monitor lists only the planned stages |
| FUX-UI2-006 | Flows | skipped steps carry visible reasons and contribute nothing; a selected prop objective blocks at the contract step instead of disappearing |
| FUX-UI2-007 | Drafts | session-only until Save / the first valid Next; Saved / Autosaved / Not-saved chip; archive / restore / typed delete for never-frozen drafts only; bulk archive; read-only filters |
| FUX-UI2-008 | Review | opens Unreviewed; nothing persists without Save Review; owner-approved labels map onto the preserved keys; `not_applicable` additive; v1 rows valid |
| FUX-UI3-001 | Metric registry | every displayed metric key (Results pickers and explorer columns, wizard gate rows, ladder columns, Context Research keys, Data & Audit report fields) resolves to a registry spec; directionality follows `OBJECTIVE_DIRECTIONS`; every reference is a registered code source; no invented threshold |
| FUX-UI3-002 | Metric status | gate PASS / FAIL by direction; boundaries strict; the 0.5 AUC line and the calibration targets are informational; adequacy minima are inconclusive below; limits and measured zeros pass or fail; policy zeros informational; intervals crossing zero inconclusive; missing / unevaluated evidence never PASS |
| FUX-UI3-003 | Roll-ups | the deterministic FAIL → BLOCKED → INCONCLUSIVE → WARNING → PASS → INFORMATIONAL order; one sentence, main reason and inspect-next per section |
| FUX-UI3-004 | Helper coverage | every widget of every UI script carries `help=` or a registered, justified, live exemption; every `help_text` id exists; every collapsed label has an accessible name; the glossary defines the sixteen terms |
| FUX-UI3-005 | Context Research | the decision summary (Data integrity, Probability skill, Calibration, Stability) leads; sample adequacy against registered minimums; coverage and net R on separate axes; the reliability diagonal named; raw JSON only under Technical identity & audit; incompatibility reasons in words; the M0–M3 computation untouched |
| FUX-UI3-006 | Results | the Selected configuration roll-ups and registry metric cards against the charter's resolved gates; the prop vector as the worst firm; the column guide; the three detail levels |
| FUX-UI3-007 | Panels | the MBP-1 and regime panels lead with a summary resolved from the selected run; manual exact ids, registries and stamps live under Advanced diagnostics only; proposed / ratified / offline / planned / implemented / blocked states are distinct chips; nothing unresolved reads as passing; the ladder keeps AUC numeric with a separate reason column |

---

## 37. Source-requirement retention crosswalk

This table confirms that every original frontend-requirement section remains authoritative here.

| Planning brief section | Authoritative section in this contract |
|---|---|
| §10.1 top-level information architecture | §§3–4 |
| §10.2 New Study modes | §6 |
| §10.3 objective-first wizard | §§7–8 |
| §10.4 objective templates | §8.2 |
| §10.5 baseline card | §9 |
| §10.6 search-space editor | §10 |
| §10.7 combination preview | §15 |
| §10.8 prop contract cards | §11 |
| §10.9 risk-policy editor | §12 |
| §10.10 benchmark editor | §13 |
| §10.11 validation screen | §14 |
| §10.12 freeze and launch | §15 |
| §10.12A full-pipeline UI | §30 |
| §10.13 Active Runs | §16 |
| §10.14 results overview | §18 |
| §10.15 payout-reliability frontier | §19 |
| §10.16 sensitivity heatmap | §20 |
| §10.17 firm matrix | §21 |
| §10.18 survival curves | §22 |
| §10.19 payout distributions | §23 |
| §10.20 configuration explorer | §24 |
| §10.21 human-readable names | §24 |
| §10.22 baseline comparison | §25 |
| §10.23 deterministic insight panel | §26 |
| §10.24 exact drill-down | §27 |
| §10.25 account timeline | §28 |
| §10.26 progressive disclosure | §5.1 |
| §10.27 status vocabulary | §5.4 |
| §10.28 UX safeguards | §§2, 5.2–5.3, 10.4 |
| §10.29 empty/blocked/failure states | §31 |
| §10.30 accessibility/responsiveness | §32 |

---

## 38. Final frontend acceptance rule

The frontend release is not accepted merely because pages render or AppTests are green.

Acceptance requires:

1. every applicable `FUX-*` row passes;
2. the UI contract is reviewed adversarially against screenshots and code, not only tests;
3. exact verifier and account-event links resolve;
4. required viewport and keyboard evidence exists;
5. browser-backend unavailability remains an open gate rather than being waived;
6. no original §10.1–§10.30 requirement is omitted or silently replaced by a generic summary;
7. the fixed M0–M3 lane remains unchanged;
8. the implementation retains truthful development, authorization, path-fidelity, and capability labels.
