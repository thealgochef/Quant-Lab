# IFVG Lab UI/UX Redesign — Implementation Plan

## Controlling focused-workspace amendment — 2026-09-07

The user-authorized implementation task expands the presentation scope to **IFVG
plus the app shell**, with research/results first, guided configuration with
advanced settings, and **Developer mode enabled only at startup**. This amendment
supersedes conflicting routes, visible labels, always-visible identity/help panels,
and Replay fallback requirements in revision 2 below. The revision-2 review and
phase reports remain historical evidence, not proof of the current appearance.

The current shell uses `st.navigation` / `st.Page` for IFVG Lab (default), ML
Training, Dashboard Compatibility, and Strategy Analysis. The former shared
pipeline controls belong only to Strategy Analysis. Within IFVG, **My studies**
combines Start, Active Runs, Results and History through selected detail views;
**Trade review** is the second destination. Only the selected route executes.

`QUANT_LAB_DEVELOPER_MODE=1`, captured at startup, registers a separate Developer
page. Technical rendering also requires that page's execution context. No UI
toggle or query override exists. Verification Center, original research diagnostic
tools, data/system health, identities, hashes, manifests, JSON, commands, traces and
technical help are confined there. This flag conveys no execution authority.

Normal study summaries use human names, questions, date ranges, evidence scope,
persisted status, and supported next actions. Exact identities remain internal.
Synthetic/implementation verification is excluded; frozen charter evidence controls
scope and unresolved legacy records remain unresolved. Normal research settings
preserve all registered workflows and validators. More study types and research
details reveal complexity only when requested. Strategy-only results have no empty
prop cards. Model and prop studies use applicable metrics; compatibility, metric
direction, uncertainty and missing-evidence limitations remain mandatory.

Replay now returns a typed candidate/setup/empty/unavailable selection. No lower
duplicate inspector, candidate fallback, or nearest-match navigation is allowed.
Chart/table alternatives and explicit Save Review remain required; actual
execution, hypothetical labels, model probabilities and reviewer judgment are
separate evidence. Unknown health is never a pass. Failure copy must describe the
effect on research and available recovery; technical exceptions belong in Developer.

Delivery is organized into these four increments, carrying forward UI-4–UI-6:

| Increment | Implemented surface | Remaining acceptance |
|---|---|---|
| 1 | Selected-page app shell, startup Developer boundary, My studies | Desktop flow review recorded |
| 2 | Goal-conditional guided configuration and concise strategy/model/prop results | Desktop flow review recorded; execution prerequisites remain |
| 3 | Typed Replay selection, chart-centered review and data alternatives | Desktop navigation, keyboard replay, save/export reviewed |
| 4 | Persisted-state lifecycle/recovery, structured health, regression coverage | Health and draft actions reviewed; live execution lifecycle remains open |

Completed UI-1–UI-3 backend and presentation contracts remain in force: no authority
bypass, no scientific schema/identity/store changes, no final-model evaluation
substituted for OOS evidence, no inferred success from progress, and no implicit
review persistence. Retained technical AppTests run under explicit Developer
presentation; normal-screen integration tests separately enforce this boundary.

The user's subsequent clarification makes acceptance **desktop layout and actual
screen/action flows**; mobile responsiveness is excluded. The browser retry captured
current desktop evidence at 1440×900 and 1280×720, including saved-study actions,
results, review, keyboard replay, exports and Developer health. Full live execution
lifecycle acceptance remains open where authority/contracts/labels are unavailable.
Historical screenshots and automated tests do not replace missing live evidence.
Exact coverage and limits are in
[FOCUSED_WORKSPACE/COMPLETION_REPORT.md](FOCUSED_WORKSPACE/COMPLETION_REPORT.md) and
the [desktop flow report](../../../reports/ifvg_browser_acceptance/20260907/DESKTOP_FLOW_REPORT.md).

## Historical revision-2 baseline

**Scope:** the IFVG Lab UI added or modified by R4–R6.1 (Experiments: New Study, Configure/Preview/Launch/Monitor/Resume/Publish, Active Runs, Results, History, Context Research; Replay / Verifier; Data & Audit) and its shared presentation helpers, providers and AppTests.
**Prepared:** 2026-09-01 · **Revised:** 2026-09-01 · **Status:** PLAN — revision 2, pending owner approval. Not an authorization. Planning only; no code, tests, configuration, artifacts, catalogs, data, owner decisions or authorizations were modified.
**Supersession:** revision 2 replaces revision 1 in full. Do not implement from revision 1.
**Authority and scope separation:** subordinate to `..\..\FINAL-IMPLEMENTATION-PLAN-DOCS\`. This plan changes UI/UX only; the backend hardening plan remains separate. After those backend contracts land, the UI consumes their namespace, date, seed, authorization and sequential-worker interfaces. Any material interface mismatch must be reconciled rather than duplicated in Streamlit.
**Frontend-contract amendment:** after owner approval, this revision is the controlling amendment for the explicitly named changes to `FRONTEND_UX_CONTRACT.md` §§3.2, 5.1, 7 and 30. The implementation must update that contract and the repository decision/architecture/state documents in the same phase as the corresponding code.

---

## 1. Executive assessment

**What works.** The workspace shell is sound: an Experiments sub-navigation radio that executes only the selected route (`ifvg_study_tab.render_ifvg_study_tab`), a typed status vocabulary with glyph + word + help (`study_status.STATUS_PRESENTATIONS`, `PIPELINE_STAGE_STATUS_PRESENTATIONS`, `HEATMAP_GLYPHS`), fifteen registered empty/blocked states (`EMPTY_STATE_PRESENTATIONS`), sanitized errors everywhere (`ifvg_ui_common.sanitize_error`), exact-ID-only loading, copyable identities (`identity_block`), a disciplined results tab with selectbox/table twins for every chart, a frozen-authority check before anything persists in the pipeline tab (R6.1 S7), and source scans that forbid promotion/ranking/sealed controls (`test_ifvg_study_scans.py`). Nothing in the UI launches work outside an explicit button handler.

**What makes the UI unsafe or confusing.** (1) Run purpose has no single authority: an "Artifact namespace" radio on every study route and a "Run scope" radio on wizard step 7 are independent, every wizard charter carries the synthetic authorization marker, so the Research namespace can never freeze (refused after eight steps as a generic "Freeze failed"), while Verification + "Full Authorized Development" freezes a synthetic charter whose authorization branch is dead code. (2) The wizard reports a successful detached launch that cannot start: the synthetic runner key exists only in the test conftest, so the worker exits before writing state and Active Runs then shows "artifact unavailable". (3) Publish caches gate results per pipeline id but not per store, so gates run under one namespace can enable activation into the other. (4) Contradictory drafts freeze silently (zero-axis FSM search; baseline compared with itself; a prop objective silently rewritten to `net_expectancy_r` when no contract is selected while prop gates still enter the charter). (5) The heatmap and firm matrix paint minimize metrics (drawdown, breach probability) green when high. (6) The reviewer verdict defaults to `correct` and persists on Save. (7) Context Research shows a green "reconcile" banner backed by a constant `True`. (8) Only 15 of 134 widgets carry help text, no glossary exists, disclosure levels gate content on one screen only, the Replay/Verifier has no case card and renders evidence twice, and Data & Audit is a raw catalog frame plus eleven collapsed JSON dumps.

**Must change before owner verification (Phases 1–2).** The run-purpose authority and derived semantic store namespace; typed authorization readiness; honest launch outcome; namespace-safe publication; charter satisfiability before freeze; the complete seed-production and verification-authorization path in the Verification Center; the review default; the false reconciliation/access-success presentation; the inverted colorscales; and removal of the ineffective worker control.

**Follow after the semantic/verification corrections (Phases 3–6).** The full metric metadata registry and section roll-ups, Context Research and results presentation, the Replay/Verifier redesign, the Data & Audit health summary, MBP-1/regime panel simplification, History/lifecycle polish, and final responsive/accessibility acceptance do not block Phase-1/2 coding, but they must pass before the redesigned UI is accepted and before the owner relies on it for the real verification run.

**Implementation order.** Backend hardening establishes namespace, authorization, date, seed and execution truth. UI Phase 1 verifies those interfaces; it does not reimplement them. All UI phases and browser/accessibility gates complete before owner seed production and real verification. UI acceptance uses synthetic/test stores only.

---

## 2. Current code and route map

| Route / screen | Presentation | Read model / providers | Source contract | Session keys | Mutating actions | Tests |
|---|---|---|---|---|---|---|
| `scripts/dashboard.py:1398` → `ifvg_lab_tab.render_ifvg_lab_tab` (:1103) | three `st.tabs`: Experiments / Replay-Verifier / Data & Audit | — | — | `ifvg_context_v1_*` | none | `test_ifvg_study_tab.py::test_top_level_shell_order_is_unchanged` |
| Experiments → `ifvg_study_tab.render_ifvg_study_tab` (:87) | horizontal radio (label collapsed) + **"Artifact namespace" radio** (:112) → `workspace_roots` (:69; constant `STATE_ROOT`) | `SEARCH_STORE_ROOT` / `SEARCH_TEST_STORE_ROOT` | P0-4 namespace rule (`search/charter.py:498–519`) | `ifvg_study_v1_route`, `_namespace`, `_pending_route` | none | `test_ifvg_study_tab.py` (6) |
| New Study → `ifvg_study_wizard.render_new_study` (:1542) | `_draft_header` (:158; "Start new draft" writes a file), 8 fixed steps `_step_objective` … `_step_review`, Back/Next/Save, `_freeze_and_launch` (:1424) | `study_drafts` (`new_draft` "Untitled study (date)", `save_draft`, `clone_draft`, `mark_frozen`, `discard_draft`), `study_presentation` (`STUDY_MODES`, `RESEARCH_QUESTIONS`, `MODE_QUESTION_COMPATIBILITY`, `OBJECTIVE_TEMPLATES`, `validate_*_step`, `enumerate_child_count`, `estimate_search_work`), `study_providers.verification_authorization_state` | `SearchCharterPayload`, `validate_charter` (:313), `save_charter`, `derive_authorization_requirements`, `runner_entry_key_for_charter` | `ifvg_study_v1_w_*`, `_draft_id` | save/clone/freeze (`save_charter`), spawn `ifvg_search_job.py` | `test_ifvg_study_wizard.py` (21) |
| Full Pipeline Run → `ifvg_pipeline_tab.render_pipeline_run` (:1781) | phase radio (collapsed) Configure/Preview/Launch/Monitor/Resume-Retry/Publish, all always clickable; `_render_configure` :710, `_render_preview` :736, `_render_launch` :911 → `_freeze_and_launch_pipeline` :962, `_render_monitor_body` :1093, `_render_resume` :1619, `_render_publish` :1707; MBP-1 and Regime Lane expanders | `study_providers.list_pipeline_runs`, `load_ladder_diagnostics`, `load_regime_*`, `mbp1_stage_evidence_defaults` | `PipelineSemanticSpecPayload`, `derive_stage_plan_readiness`, `run_publication_gates`, `activate_pipeline_result` | `ifvg_pipeline_v1_*` | freeze spec, spawn `ifvg_pipeline_job.py`, cancel sentinel, gates, **activation (catalog event)** | `test_ifvg_pipeline_tab.py` (60) |
| Pipeline read-model boundary → `study_providers.py` | namespace/store discovery, search/pipeline listings, authorization state, artifact and diagnostic loads | `list_search_runs`, `list_pipeline_runs`, `verification_authorization_state`, `load_*` | current filesystem roots and persisted contracts | cache keys vary by caller | none | provider tests distributed across wizard/pipeline/results suites |
| MBP-1 panel → `ifvg_mbp1_panels.py` | availability, activated/planned bundles, policy defaults, stage-window registry, exact-ID diagnostic inputs | `mbp1_stage_evidence_defaults`, MBP-1 stores/providers | R5B/R5B.1 contracts; `research_only_offline` boundary | pipeline keys | none in normal display; exact-ID diagnostics | pipeline/AppTests and R5B/R5B.1 tests |
| Regime panel → `ifvg_regime_panels.py` | algorithm registry, proposed defaults, model/fit/capability/promotion/stratified-report exact-ID inputs | `load_regime_*` | R6/R6.1 contracts and promotion ladder | pipeline keys | none in normal display; exact-ID diagnostics | pipeline/AppTests and R6/R6.1 tests |
| Active Runs → `ifvg_active_runs_tab.render_active_runs` (:66) | monitor body, funnel buttons, child table, safe cancel | `list_search_runs`, `load_search_state`, `funnel_counts`, `child_row_presentation` | search_state.json | `ifvg_study_v1_monitor_*` | cancel sentinel | `test_ifvg_active_runs_tab.py` (6) |
| Results → `ifvg_results_tab.render_results` (:222) | `disclosure_level` (the only screen using it), overview, frontier, heatmap, firm views, explorer, audit block | `load_results_bundle`, `load_frontier_for_state`, `load_child_metrics`, `load_prop_vectors` | frontier / costed evaluation / prop vectors | `ifvg_study_v1_results_*` | none | `test_ifvg_results_tab.py` (13) |
| History → `render_history` (:961) | drafts (Open/Clone/**Discard behind bare `confirm`**), frozen/running/completed/superseded runs (Rename) | `list_drafts`, `list_search_runs`, `append_catalog_event` | catalog events | — | discard draft (hard delete), rename | `test_history_separates_sections_and_keeps_frozen_immutable` |
| Context Research → `ifvg_lab_tab.render_ifvg_experiments_tab` (:733) | capability registry, pair selector, tier/target/cohort, run, `_render_result` → `_render_candidate_report` :384, `_render_execution_report` :477, `_render_coverage_report` :548, `_render_reconciliation_report` :586, `_run_history` :649 | `context_report_adapters`, `context_reporting`, `context_run_store` | M0–M3 lane (frozen) | `ifvg_context_v1_*` | runs the M0–M3 experiment (existing, unchanged lane) | `test_ifvg_lab_tab.py` (19) |
| Replay / Verifier → `render_ifvg_replay_tab` (:879) → `ifvg_verifier_tab.render_verifier_section` (:1174) | selection mode radio (candidate / setup), filters `_filtered_candidates` :200, prev/next candidate + trade, Mode (Full audit / Point-in-time) + stage slider, Range, six overlay checkboxes, chart (`ifvg_verifier_charts.build_verifier_figure` / `build_setup_figure`), side accordions `_render_side_panel` :279 (Identity, Lineage, Lifecycle, Geometry, Execution, Context, Model, Gating, Review), **lower duplicate inspectors** (`ifvg_lab_tab.py:955–1027`) | `setup_verifier_provider`, `replay_chart_provider`, `replay_chart_store`, `visual_review_store` | replay-chart artifact, verified pair | `ifvg_context_v1_verifier_*`, `_review_*` | `append_review` (append-only ledger `ifvg_visual_review_v1`) | `test_ifvg_verifier_tab.py` (7), `test_ifvg_setup_verifier_tab.py` (8), `test_ifvg_verifier_charts.py` |
| Data & Audit → `render_ifvg_data_audit_tab` (:1030) | catalog `st.dataframe` (5 of 7 columns 64-hex), pair selector, three tabs of collapsed `st.json` (5 v2 reports; v3 identity + table counts; 6 v3 reports) | `_pair_catalog`, `VerifiedIfvgPair.v2.reports / v3.reports` | `ifvg/reporting.py` report schemas (validity/reconciliation/capacity/performance/identity), `data_access.DataAccessAudit` | — | none | `test_tab_smoke_apptest` |
| Shared | `ifvg_ui_common` (`status_badge`, `dev_only_badge`, `verification_badge`, `disclosure_level`, `identity_block`, `render_empty_state`, `paginate_controls`, `result_scope_caption`, `queue_replay_drilldown`); `ifvg_results_charts` builders; `ifvg_lab_charts` (`build_coverage_figure`, `build_calibration_figure`, `build_replay_figure` exported but **unused**) | `study_status`, `study_presentation` | — | — | — | `test_ifvg_study_scans.py` (14) |

Helper-text coverage today (widgets / with `help=`): wizard 23/5, pipeline tab 21/5, verifier 47/3, lab tab 9/1, results 11/0, active runs 5/1, regime panels 5/0, MBP-1 panels 4/0, compare 4/1, study tab 2/1. Collapsed labels: `ifvg_study_tab.py:100`, `ifvg_pipeline_tab.py:1796`, `ifvg_results_tab.py:1066`, `ifvg_study_wizard.py:799`. `column_config`: none across ~76 dataframes. Disclosure levels: `ifvg_results_tab.py:254` only.

---

## 3. Confirmed findings

| ID | Sev | Screen | Confirmed behavior | Consequence | Phase |
|---|---|---|---|---|---|
| F-01 | P0 | New Study, study routes | `_assemble_charter` hard-codes `owner_authorization = SyntheticAuthorizationMarker()` (`ifvg_study_wizard.py:1348`); `save_charter` refuses synthetic charters under `search/v1` (`charter.py:511–519`); `validate_charter` returns at `if synthetic` (`:410`) so the scope→`derive_authorization_requirements` branch (`:413–427`) is dead for every UI charter; the namespace radio (`ifvg_study_tab.py:112`) and the scope radio (`ifvg_study_wizard.py:869–890`) are independent; `_effective_run_scope` returns `synthetic_fixture` while the persisted spec says `verification_5d` / `full_authorized_development` (`ifvg_pipeline_tab.py:946–959`); the namespace radio does not scope listings (`workspace_roots` returns a constant `STATE_ROOT`, `:48,81`; `list_search_runs` uses `store_root` for annotations only), yet the empty-state copy says "under this namespace" and the verification badge keys on the radio, not the artifact (`ifvg_results_tab.py:258`). | Research namespace can never freeze and fails after eight steps as "Freeze failed"; Verification + Full Authorized Development freezes a synthetic charter labelled full-scope; badges and listings misdescribe scope. | 1 |
| F-02 | P0 | New Study → Active Runs | `runner_entry_key_for_charter` returns `synthetic_search_job_fixture_v1` for every UI charter (`runner_registry.py:164–165`); that key lives only in `_DEVELOPMENT_ENTRIES`, populated by `tests/agents/ifvg_search/conftest.py:32`; the detached worker exits at `ifvg_search_job.py:97–99` before writing state; the UI prints "Search launched detached (pid …)" and routes to Active Runs (`:1507–1512`), which renders `artifact_unavailable` (`ifvg_active_runs_tab.py:69–74`). The `runner_executor_planned` branch (`:1456–1473`) is unreachable. | Every wizard launch is reported as success and dead-ends; no test covers it (all tests monkeypatch `_spawn_search_job`). | 1 |
| F-03 | P0 | Publish | Gate cache key `f"{_PIPE}gate_results_{pipeline_id}"` (`ifvg_pipeline_tab.py:1736`) is store-blind; `list_pipeline_runs(PIPELINE_STATE_ROOT)` is namespace-blind (`:1719`); `activate_pipeline_result(..., store_root=roots["store_root"])` (`:1770`) writes the catalog event into whichever store the radio currently selects; the only refusal keys on `state["run_scope"] == verification_5d` (`pipeline.py:2587`), not on authorization kind or store. | Gates passed under one namespace can activate a research catalog entry for a synthetic full-scope run after flipping the radio. | 1 |
| F-04 | P0 | New Study (satisfiability) | No minimum-axis rule (`validate_search_space_step`, `validate_charter:369–372` bounds upward only): `fsm_config_search` with zero axes freezes as a one-child search; `single_configuration` + "compare with baseline" compares the baseline with itself; a prop-bearing template with zero contracts is silently rewritten to `net_expectancy_r` (`ifvg_study_wizard.py:1276–1284`) while step-6 prop gates still enter the charter (`:1384–1393`); `universal_prop_search` passes with one firm (`charter.py:378–381`). | Contradictory intent freezes into immutable charters; the declared objective and the persisted objective differ without notice. | 1 |
| F-05 | P0 | Results heatmap, firm matrix | `colorscale="RdYlGn"` (`ifvg_results_charts.py:267,326`) for metrics registered `minimize` (`max_drawdown_r`, `breach_probability_90d`; `HEATMAP_METRICS`/`FIRM_MATRIX_METRICS`, `study_presentation.py:1180–1197`; `OBJECTIVE_DIRECTIONS`, `charter.py:121–`); the frontier already uses `RdYlGn_r` (`:195`). Glyphs encode data adequacy, not goodness. | Worst drawdown / highest breach probability renders green. | 1 |
| F-06 | P0 | Replay / Verifier review | "Overall verdict" selectbox directly over `REVIEW_VERDICTS` whose first element is `correct` (`ifvg_verifier_tab.py:472–474`, `:946–948`; `visual_review_store.py:33`); `append_review` requires only presence (`:109–110`); no unsaved/saved state; widgets keep the last verdict across candidates. | A reviewer who saves notes writes an affirmative verdict into the append-only ledger they never chose. | 2 |
| F-07 | P0 | Context Research reconciliation; access counters | `build_context_reconciliation_audit_report` returns `"passed": True` unconditionally (`context_reporting.py:295`) → `st.success("… reconcile.")` (`ifvg_lab_tab.py:588–589`); the evaluated gates (`reporting.py:482,536,573,596,644`) appear only as an uncolored "Gate cards" frame; `data_access_audit.json` carries no `passed`, so its row renders `passed=None, violation_count=0` (`context_report_adapters.py:188–194`); `protected_*` counters are compile-time zeros (`data_access.py:90–93`). | Unknown / not-evaluated evidence is presented green or as if evaluated. Minimal truthfulness correction is Phase 1; full metric/roll-up presentation is Phase 3. | 1, 3 |
| F-08 | P1 | New Study step 7/8, Full Pipeline Run | `DatePolicy` requires the frozen ten-date warmup prefix for the development policy (`charter.py:224–227`) but the wizard offers two unlabelled free-text date areas (`ifvg_study_wizard.py:949–969`) with no per-field validation (`validate_validation_step` bounds only verification); the pydantic error surfaces at freeze as "Freeze failed"; the acknowledgement is typed twice (`:1174–1178`, `ifvg_pipeline_tab.py:931–936`); a verification run traverses all eight steps (prop, risk, benchmarks) and the research gates are stored in the verification charter. | Full Authorized Development is un-assemblable from the UI; verification is buried in an inert research wizard. | 1–2 |
| F-09 | P1 | Context Research, ladder panel | Brier / Brier skill / AUC rendered as bare metrics (`ifvg_lab_tab.py:396–403`; `_ladder_frame` `ifvg_pipeline_tab.py:1243–1276` puts the `auc_reason` token in the AUC column) while `prevalence`, `mean_probability`, `reference_brier_score` exist (`context_statistics.py:149–153`) and are dropped; coverage fraction (0–1) and `net_r_sum` share one `st.bar_chart` axis (`:424–431`) while `build_coverage_figure` (two-row subplot) is dead code. | Skill metrics are uninterpretable; coverage is an unreadable sliver. | 3 |
| F-10 | P1 | Replay / Verifier | No persistent case card (facts live in the selectbox option string `_candidate_label` :269–276); nine identical collapsed expanders; `render_ifvg_replay_tab` renders a second evidence set below (`ifvg_lab_tab.py:955–1027`) duplicating Geometry/Lifecycle/Lineage/Context as raw JSON; setup mode returns `None` (`:1209`) so the caller falls through to an "Exact candidate ID" selectbox over all candidates and inspects an unrelated candidate (`ifvg_lab_tab.py:900–905`); evidence boundary (Mode/stage), range and overlays share one row; trade vs candidate navigation unexplained; light chart (`paper_bgcolor="white"`, `ifvg_verifier_charts.py:871`) in a dark shell; legend effectively empty (`showlegend=False` on most traces); no table twin of the chart. | Trader/researcher cannot read a case at a glance; auditors see duplicated, sometimes unrelated, evidence. | 4 |
| F-11 | P1 | All screens | Help on 15 of 134 widgets; no glossary (FVG, IFVG, FSM, HTF, LTF, OOS, PIT, MBP-1, R, Brier, AUC, EQH/EQL, MFE/MAE, AMI, Q-40 undefined); four collapsed labels incl. both navigation radios; zero `column_config`; `disclosure_level` used only in Results; `artifact_unavailable` rendered for empty lists (`ifvg_active_runs_tab.py:69`, `ifvg_results_tab.py:231`, `ifvg_pipeline_tab.py:712`) although `pipeline_no_runs` exists; viewports evidenced at 1440×1100 only; hashes clip mid-value in R6.1 screenshots 12/13. | Owner must read internal keys; states misreport; accessibility rows unproven. | 3, 6 |
| F-12 | P2 | New Study, History, Data & Audit | Stale copy: `STUDY_MODES["full_pipeline_run"].description` "lands with R5" (`study_presentation.py:332–339`), "Blocked or planned capabilities: real executors (R5) · MBP-1 bundle (R5B) · regime lane (R6)" (`ifvg_study_wizard.py:1166–1168`), `runner_executor_planned` detail "the R5 pipeline registers the executors" (`:1460–1463`); the count caption `~{count}` (`:481–484`) reads as "-1" (no expression yields −1; `enumerate_child_count` returns `prod(sizes)` or 1); "Start new draft" writes `Untitled study (date)` immediately (12 such step-0 files on disk today); History Discard is a hard delete behind a bare `confirm` checkbox (`ifvg_results_tab.py:991–1002`); Data & Audit is a raw catalog frame + eleven collapsed `st.json` blocks with no derived verdict although every report carries `passed`/`observed`/`limits` keys. | Misleading capability claims, clutter, and an audit surface without a summary. | 2, 5, 6 |
| F-13 | P0 | Cross-plan namespace, authorization, verification and runtime integration | The current UI treats filesystem roots and Boolean "authorization present" checks as sufficient authority, does not represent the separately authorized seed-production chain required before the final verification authorization, and exposes a worker control although the accepted backend V1 executor is sequential. The backend hardening plan replaces path-derived authority with `StoreNamespaceEnvelope/store_namespace_id`, distinguishes typed authorization failure modes, separates logical trading days from physical partitions, requires `SeedProductionAuthorizationRef` before seed creation and a later `VerificationAuthorizationRef`, and accepts only effective worker count 1. | Implementing the redesign directly against the old path/Boolean/control model would create a second authority and make the UI stale as soon as backend hardening lands. | 1–2 |

Refuted hypotheses (evidence): a `−1` child count cannot be produced (`study_presentation.py:715–724`, `:1270`; `study_providers.py:142`); the wizard has no free step-jumping (linear Back/Next, `disabled=bool(errors)`); Publish already disables verification activation with a stated reason (`ifvg_pipeline_tab.py:1756–1763`); `discard_draft` refuses frozen drafts (`study_drafts.py:243–254`); `ifvg_mbp1_panels.py:129` "pre-activation, R5 planned state" is a deliberate before/after identity label; the frontier colorscale is correct; no chart-only selection route exists outside the verifier chart.

---

## 4. Open questions and resolved ambiguities

Asked in one batch on 2026-09-01; answered by the owner (binding for this plan):

| # | Question | Owner decision (summary of the verbatim answer) |
|---|---|---|
| Q1 | Run-purpose model vs the two contract scopes and the namespace radio | **Presentation-only purpose; derive namespace.** `RunScope` values unchanged. Implementation Verification → `verification_5d`, `search_test/v1`, exact baseline only, verification gates only, no research interpretation or publication. Development Research → `full_authorized_development`, `search/v1`, a restricted question-derived stage plan, publication unavailable by default and separately gated. Full Authorized Development → `full_authorized_development`, `search/v1`, the full owner-authorized stage plan, publication a separate eligibility/action gate. Remove the namespace radio from New Study, Configure and Publish; namespace derived from purpose and displayed read-only; Results/History may offer read-only purpose/store filters that never affect where a charter freezes or launches. |
| Q2 | Draft lifecycle | **Archive first; delete never-frozen only.** "Start new draft" creates an in-memory/session draft and writes no file; first persistence on explicit Save Draft or the first valid Next; afterwards autosave with visible Saved/Saving/Unsaved. Archive is the normal reversible action; archived drafts hidden from default History and restorable. Permanent Delete only for drafts never frozen and never launched, only from an Archived/Advanced view, with typed confirmation of the exact draft name. Frozen charters, attempts and runs are never deletable (catalog archive flag only). Provide a one-time migration or bulk-archive for the existing empty step-0 "Untitled study" files; never silently delete them. |
| Q3 | Review verdict vocabulary | **Keep ledger keys `ifvg_visual_review_v1`; preserve rows.** "Unreviewed" is an unsaved UI state only; opening a case creates no ledger row and preselects no verdict. Labels: Correct→`correct`, Incorrect→`incorrect`, Unclear→`insufficient_evidence` ("evidence incomplete, ambiguous or insufficient to judge correctness"), Needs investigation→`questionable` ("a possible problem or inconsistency that warrants follow-up, not yet asserting incorrectness"), Not applicable→new additive key `not_applicable` ("the review question does not apply to this case"). Persist only on explicit Save Review; show Unsaved/Saving/Saved; helper text per verdict. If `not_applicable` cannot be added additively under the current validator, stop and ask before a v2 schema. |
| Q4 | Baseline-alone goal vs compare | **Two separate task cards.** "Evaluate one configuration": exactly one selected frozen profile (baseline or another eligible profile as anchor), no comparison or delta claim; rule `resolved_profile_count == 1`, `challenger_count == 0`. "Compare one configuration with the baseline": exactly one baseline + exactly one challenger; compatible populations/policies/evidence before deltas; the UI describes what changed; rule `resolved_profile_count == 2`, `challenger_count == 1`. Both use the single-configuration study family internally. FSM Configuration Search stays separate (≥1 axis, ≥2 profiles). Implementation Verification is separate and never becomes a development-research "Evaluate one configuration" result. |

Revision-2 clarifications, without changing Q1–Q4:

- `RunPurpose` is non-semantic; mutable `RunPurposeAnnotation` preserves it across reload. Ambiguous legacy drafts become `purpose_unresolved` and cannot freeze.
- Selected prop objectives are never rewritten; missing verified contracts block the path.
- One challenger means one resolved configuration, which may contain several registered differences.
- Missing browser/keyboard/viewport evidence leaves UI acceptance open.
- Namespace, seed-production, final verification authorization and worker truth come only from the separate backend contracts.

Proposed UX defaults: task-card wording; the `Verify Implementation` route; Full Authorized Development disabled until typed readiness is `ready`; fail-closed degenerate-charter validation; sticky work estimates; color tokens; and contextual help plus a searchable glossary.

---

## 5. Target information architecture

### 5.1 Before / after route map

```text
BEFORE                                                AFTER
IFVG Lab                                              IFVG Lab
├── Experiments                                       ├── Experiments
│   [Artifact namespace radio on every route]         │   ├── Start            ← task cards ("What are you trying to do?")
│   ├── New Study (8 fixed steps, 5 modes)            │   ├── Verify Implementation  ← Verification Center (dedicated)
│   │   └── mode 5 → Configure/Preview/Launch/…       │   ├── New Study (goal-derived conditional flows)
│   ├── Active Runs                                   │   │   └── advanced end-to-end study → Configure / Review / Launch
│   ├── Results (disclosure levels)                   │   ├── Active Runs (state-driven actions)
│   ├── History (drafts + runs)                       │   ├── Results (Summary / Research / Audit on every block)
│   └── Context Research (M0–M3, unchanged lane)      │   ├── History (drafts, archived, runs; purpose/store filters read-only)
├── Replay / Verifier                                 │   └── Context Research (same lane; decision summary first)
│   └── side accordions + lower duplicate inspectors  ├── Replay / Verifier
└── Data & Audit                                      │   └── case card · chart · research details · advanced evidence
    └── catalog frame + 11 st.json                    └── Data & Audit
                                                          └── health summary · catalog · structured reports · raw JSON (audit)
```

Top-level `st.tabs` labels stay `Experiments`, `Replay / Verifier`, `Data & Audit` (asserted by `test_top_level_shell_order_is_unchanged`). The Experiments radio gains `Start` and `Verify Implementation` (`StudyWorkspaceRoute` + `ROUTE_LABELS` additive; existing route ids unchanged so programmatic `request_route` callers keep working).

### 5.2 Task-oriented entry (`Start`)

Nine cards; each derives purpose, study family, stage plan, and semantic namespace (owner Q1/Q4):

| Card | Derives | Notes |
|---|---|---|
| Verify the current implementation | purpose = Implementation Verification → Verification Center | never a research result |
| Review strategy setups and trades | Replay / Verifier (exact pair + case) | no run |
| Evaluate one configuration | Development Research · single_configuration · `challenger_count == 0` | descriptive strategy/model/prop evaluation |
| Compare one configuration with the baseline | Development Research · single_configuration · exactly one challenger | deltas only when compatible |
| Search FSM parameters | Development Research · fsm_config_search · ≥1 axis, ≥2 profiles | recommended preset: parent-retest timeout {unbounded, 240, 360, 480} |
| Evaluate feature and model evidence | Development Research · pipeline with S05–S10 only (bundle/ladder/optional regime) | S11 blocked by contract |
| Test prop-firm feasibility | Development Research · Prop Benchmark or Universal Prop Search | Prop Benchmark requires ≥1 `first_party_verified` firm; Universal requires ≥2 |
| Run an advanced end-to-end study | Full Authorized Development · full stage plan | visible-disabled until typed computation-path authorization readiness is `ready` (proposed UX default) |
| Inspect system health and audit evidence | Data & Audit | — |

### 5.3 Verification Center (`Verify Implementation`)

One state-driven Verification Center, with no research steps:

1. **Purpose/readiness:** `verification_5d`, verified test `StoreNamespaceEnvelope`, verification-only gates and badge.
2. **Fixture:** ranked logical trading-day windows, session bounds, exact physical partitions, coverage/ranking evidence and protected/sealed exclusions; owner explicitly selects the provisional window.
3. **Seed:** matching-seed status; unsigned/signed `SeedProductionAuthorizationRef`; exact dates/sources and prohibited outputs; explicit seed job; verified seed ID, profile match, provenance and access audit.
4. **Final authorization:** unsigned/signed `VerificationAuthorizationRef`, available only after a verified seed; typed missing/stale/superseded/wrong-namespace/head/profile/source states.
5. **Review/run:** exact baseline, allowlist/partitions, seed, dual drive, release-control checks, artifacts, `sequential_children_v1`, workers=1; launch only after typed preflight passes.
6. **Monitor/inspect:** only resolved seed/verification stages, gate evidence, verifier links and access counters; no Publish.

The center consumes the backend namespace/date/seed/authorization/readiness contracts and never uses a synthetic marker for real verification. External signing or job actions expose exact commands, packet paths and Refresh behavior. Prop, risk, research benchmarks, feature/model selection and publication are absent.

### 5.4 Conditional research flows (New Study)

Steps are derived from the card; skipped steps are listed with a reason ("Prop Contracts — skipped: no prop objective"), never rendered empty:

| Goal | Steps |
|---|---|
| Evaluate one configuration | Goal → Configuration (anchor profile) → Strategy gates (display) → Review |
| Compare with baseline | Goal → Baseline → Challenger (one resolved configuration; one or more explicitly registered differences allowed) → Strategy gates → Review |
| FSM search | Goal → Baseline → Search axes → Strategy gates → Review |
| Prop feasibility | Goal → Strategy source → Firm contracts (≥1 verified) → Risk/payout policies → Prop benchmarks → Review |
| Universal multi-firm | Goal → Strategy search → ≥2 firm contracts → policies → universal gates → Review |
| Feature/model study | Goal → Cohort → Feature bundles → Model ladder → Optional regime analysis → Review |
| Advanced end-to-end | Goal → all applicable steps → Configure → Review → Launch (typed acknowledgement once) |

### 5.5 State-driven run lifecycle

`RunLifecycleState` (pure, `study_presentation.py`) derived from draft/charter/state/publication:

| State | Available primary actions | Hidden / disabled |
|---|---|---|
| draft | Configure, Review, Save, Archive | Launch, Monitor, Publish |
| ready and authorized | Launch (with second confirmation for full scope) | Publish |
| running | Monitor, Cancel (safe boundary) | Configure edits, Publish |
| failed / interrupted | Resume / Retry (operational identity), Monitor | Publish |
| completed research, publication-eligible | Finalize / Add to Research Catalog (verify-then-activate), Results | — |
| completed verification | Results (verification-only), Monitor | **no Publish route** |
| archived / superseded | Results (read-only), Restore (drafts) | everything else |

The phase radio becomes a state-driven set of enabled actions; unavailable phases render as disabled chips with the reason, never as clickable navigation.

### 5.6 Presentation-only purpose persistence

`RunPurpose` is a UI/workflow classification, not a scientific identity field. Persist a mutable `RunPurposeAnnotation{purpose, derivation, owner_confirmed, updated_at}` alongside draft metadata and as a catalog annotation referencing a frozen charter/pipeline. New drafts record the card-selected purpose immediately in session state and on first persistence. On reload:

1. use the stored annotation when present;
2. otherwise derive only when run scope + stage plan + authorization kind produce one unambiguous purpose;
3. otherwise render `purpose_unresolved`, require explicit owner confirmation, and block freeze/launch.

Changing the annotation may alter available UI actions and labels but never changes an immutable charter or pipeline identity. It cannot be used to bypass the actual `RunScope`, `StoreNamespaceEnvelope`, stage-plan, or authorization validators.

---

## 6. Shared UX / presentation contracts (all pure, under `src/…/ifvg/presentation/`)

**6.1 Status vocabulary** — `presentation/status_vocabulary.py`: `UiStatus` = PASS, FAIL, BLOCKED, WARNING, INCONCLUSIVE, INFORMATIONAL, NOT_APPLICABLE, NOT_SELECTED, UNAVAILABLE, CORRUPT, IN_PROGRESS, COMPLETE, SUPERSEDED; `StatusSpec{label, glyph, color_token, meaning, blocks_next_action: bool}`; `status_from_gate(value, gate)`, `status_from_reference(...)`, `status_for_evidence(kind)`. Existing `StudyStatusKey`, pipeline stage statuses, heatmap classes and empty-state keys MAP onto `UiStatus` (additive adapters; the old enums stay as persisted/contract vocabulary). Color tokens: green PASS, red FAIL, amber WARNING/INCONCLUSIVE/proposed, blue INFORMATIONAL, gray NOT_APPLICABLE/NOT_SELECTED/UNAVAILABLE/no comparator, purple research-only/experimental; every chip = glyph + word; color never alone.

**6.2 Metric metadata registry** — `presentation/metric_registry.py`: `MetricSpec{technical_key, human_name, definition, formula_or_source, unit, directionality ∈ {higher_better, lower_better, target, descriptive}, reference: ReferenceSpec | None, gate_source: symbol path | None}`; `evaluate_metric(spec, value, *, gate=None, reference=None, sample=None) -> MetricReading{status, interpretation, caveat, evidence_ref}`. Directionality comes from `OBJECTIVE_DIRECTIONS` where registered; references from existing code only (Brier → `reference_brier_score`/prevalence; Brier skill → 0 boundary; AUC → 0.5 direction only, no bands; net E[R]/PF/drawdown/TUW → the selected resolved gate; sample counts → adequacy minimums; CI crossing zero → INCONCLUSIVE; access counters → 0 PASS, nonzero FAIL, missing → UNAVAILABLE never PASS; capacity/performance → observed vs `limits`; calibration slope/intercept → target 1/0 with distance, no good/bad). Registry rows cover the Q1 inventory (§3 of Reviewer B): Context Research candidate/execution/coverage/reconciliation keys, results overview/frontier/heatmap/explorer/firm/survival/payout keys, wizard gate rows, ladder columns, regime model-card gates, verifier execution metrics, Data & Audit report fields. One source; screens call `describe(technical_key)`.

**6.3 Section roll-ups** — `presentation/rollups.py`: `SectionRollup{status, sentence, main_reason, inspect_next}` with the deterministic rule FAIL if any blocking child fails → BLOCKED if required evidence/authorization missing → INCONCLUSIVE if no failure but evidence insufficient/undefined → WARNING if usable with cautions or proposed thresholds → PASS if every required metric passes → INFORMATIONAL if no gate applies. Sections: Data integrity, Strategy quality, Probability skill, Calibration, Stability, Prop feasibility, Robustness, Authorization readiness, Capacity and performance.

**6.4 Helper text and glossary** — `presentation/help_registry.py`: `HelpEntry{what_it_changes, why, default, changes_identity: bool, requires: {replay, refit, resimulation}, owner_approval: bool, availability}` keyed by control id; `GLOSSARY` (FVG, IFVG, FSM, HTF, LTF, OOS, PIT, MBP-1, R multiple, Brier, Brier skill, AUC, EQH/EQL, MFE/MAE, AMI, Q-40). Every metric, section roll-up, non-obvious input/selector/toggle, consequential action, acronym and status gets visible helper text or keyboard-accessible tooltip content. A small explicit `HELP_EXEMPTIONS` registry may exempt self-explanatory controls such as Back; every exemption records the control id and rationale. Source scans fail on an unregistered missing helper, inaccessible tooltip, or collapsed label without an accessible name—they do not force noisy help onto trivial controls.

**6.5 Human-label registry** — `presentation/labels.py`: profile display names, bundle keys (`B5_CORE_STRUCTURE_ORDER_FLOW_REGIME` → "Core + structure + order flow + regime"), objective keys (`payout_probability_per_rolling_30d` → "Payout probability per rolling 30 days"), stamps (`proposed_protocol_default` → "Proposed default — owner ratification required"), verdict labels (Q3). Technical key always available via reveal.

**6.6 Technical-detail disclosure** — `ifvg_ui_common.detail_levels(st, key)` renders Summary → Research details → Technical identity & audit on every major screen (extends `disclosure_level`); `identity_reveal(st, label, value)` = human label + `st.code` full id + copy; raw JSON only inside the Audit level.

**6.7 Empty/error states** — `EmptyStateKey` gains `NO_RUNS`, `NOT_SELECTED`, `NOT_APPLICABLE`, `NOT_CONFIGURED`, `RESUME_AVAILABLE`, `ARTIFACT_MISSING`, `ARTIFACT_CORRUPT` (split from `ARTIFACT_UNAVAILABLE`), `LEGACY_READ_ONLY`, `SUPERSEDED`; each with heading, meaning, owning gate, next action, audit disclosure. A contract test asserts no `no_runs`/`not_selected`/`not_applicable` situation calls `artifact_unavailable`.

---

## 7. Screen-by-screen target design

**Start.** Title + one-sentence purpose; nine task cards (§5.2); each card shows purpose badge, resolved namespace class and `store_namespace_id` (read-only), what will run, owner approvals needed, availability status chip (implemented and selectable / implemented but owner-unratified / research-only offline / blocked by missing dependency / planned post-V1). No namespace radio anywhere on New Study/Configure/Publish (owner Q1). Local filesystem paths may appear only under Technical identity as operational locations and never define authority.

**Verification Center.** §5.3; sticky readiness card for semantic store namespace, logical-day fixture, physical partition coverage, seed-production authorization, seed job, verified seed, final verification authorization and bounded-run readiness. It renders every typed backend state rather than a Boolean "present" check. Real launch accepts only a validated `VerificationAuthorizationRef`; `SyntheticAuthorizationMarker` is confined to fully synthetic fixtures. The center exposes exact external commands and refresh actions when signing or seed production remains outside Streamlit. It shows only active seed/verification stages, labels all results verification-only, and has no Publish route.

**New Study.** Goal card fixed at the top with the derived purpose, semantic namespace, stage plan and skipped steps; steps conditional (§5.4); `RunPurposeAnnotation` persists the presentation purpose without entering semantic identity; "Start new draft" is session-only until Save/first valid Next (owner Q2) with a Saved/Saving/Unsaved chip; draft name required before first persistence (default proposal "<goal> — <baseline short name> — <date>"); duplicate detection warns when an identical unsaved goal/baseline draft exists. Existing drafts without a uniquely derivable purpose show `purpose_unresolved` and cannot freeze until the owner confirms it. **Strategy Search Space:** only applicable, searchable, non-blocked axes with ≥1 challenger shown; preset "Parent-retest timeout: unbounded baseline, 240, 360, 480"; locked/measured-only/blocked/legacy/evidence/impact tables under Advanced Audit; sticky calculation summary (selected axes, configurations "baseline + N challengers", full sequential replays, new artifact families, estimated runtime/storage from `estimate_search_work`, required owner approvals) — the `~{count}` caption is replaced by "N configurations: baseline + (N−1) challengers" / "Baseline only: 1 configuration" / "No challenger selected". For Compare, exactly one challenger configuration is required, but that challenger may contain one or more explicitly registered changes; the Review page lists every difference. **Prop Contracts / Risk:** when a prop objective is selected at least one `first_party_verified` contract is mandatory (Universal Prop Search requires at least two); absence blocks Next/Freeze with an explicit contract-workflow action. The UI never removes or rewrites a selected objective. Strategy-only goals skip prop/risk steps because the selected goal never requested prop metrics. **Benchmarks:** active gates only, "Add another gate" for inactive; percentages, plain inequalities, rationale, proposed/ratified badges; never shown on verification paths. **Review & Launch:** `CharterSatisfiabilityReport` card (PASS/FAIL per rule; FAIL disables Freeze); typed authorization readiness, not a Boolean; single typed acknowledgement for full scope; honest launch outcome (see Phase 1).

**Active Runs.** Distinct `NO_RUNS` state with next action; state-driven actions (§5.5); child table with `column_config` and repeated selected-row detail; CLI fallback unchanged.

**Results.** Every block: scope label, roll-up card, metrics via the registry (human name, value, status chip, reference, sample, caveat, "why" popover), Research details (charts with twins), Audit (ids, manifests, raw). Heatmap/firm matrix colorscale direction from `OBJECTIVE_DIRECTIONS` (`RdYlGn_r` for minimize); colorbar titled with direction ("lower is better").

**MBP-1 and regime panels.** These become normal summary/readiness surfaces rather than registry dumps. MBP-1 first shows: available offline feature block, source/coverage readiness, owner-ratification state, active bundle, current controlled-study state, and the fixed `research_only_offline` boundary; manual coverage/artifact/candidate/study ID fields move to Advanced diagnostics and auto-resolve from the selected run whenever possible. Regime first shows: active algorithm (`kmeans_v1`), panel/fit/assignment coverage, role/promotion status, OOS capability and why a model-bearing path is blocked or eligible; planned post-V1 algorithms remain visible but disabled. Scientific defaults display proposed/ratified status and helper text. Full registries, hashes and exact IDs remain under Technical identity & audit. Neither panel implies live serving or execution gating. Runtime controls state `Execution mode: Sequential in V1; effective workers: 1`; no operative worker slider remains.

**History.** Columns: name, study type, current step, updated, status, purpose/store (read-only filter), actions (Open, Clone, Archive, Restore; Delete only in the Archived/Advanced view with typed exact name and only for never-frozen/never-launched drafts); immutable runs: Results, Clone as New Search, Rename, Archive flag; one-time "Archive empty untitled drafts" action for the existing files (owner Q2).

**Context Research.** Decision summary first (roll-ups: Data integrity, Probability skill, Calibration, Stability; candidate vs actual-execution scope labels); sample adequacy card; Brier/Brier skill/log loss/AUC with references (prevalence, reference Brier, 0 skill boundary, 0.5 AUC direction) and helper text; reliability chart with the diagonal named; coverage and net R on separate axes (reuse `build_coverage_figure`); threshold coverage table; fold validity chips; uncertainty with CI-crossing-zero → INCONCLUSIVE; feature importance top-N with fold stability; run compatibility reasons in words first; the reconcile banner derived from the evaluated gate `passed` flags and `UNAVAILABLE` when a report carries no `passed`; direct exact links from candidate/result rows into the verifier (`queue_replay_drilldown`).

**Replay / Verifier.** Persistent case card (date/session, state, executed/blocked/censored, entry family, HTF/parent/opposing/entry timeframes, actual result, counterfactual labels, block reason, review status) with four visibly distinct panels: Actual execution · Counterfactual label · Model probability · Reviewer judgment. Controls grouped: Find a case (filters, candidate/trade navigation with an explanation "trades are the executed subset"), Evidence boundary (Full audit vs Point-in-time with helper text and a stage scrubber tap→parent→lock→opposing→inversion→entry→resolution listing hidden evidence), Chart range, Overlays. Chart: dark-theme template, readable axes, explicit zone legend with consistent role colors, human lifecycle markers with hover, lifecycle timeline strip, local time with UTC detail, tick/price terminology, no overlapping labels, Q-40/240m warning kept, responsive width, keyboard-operable range/stage controls, and a table twin of every plotted layer (Advanced). Side: Summary panel → Research details → Advanced exact evidence (Identity, Lineage, raw JSON, wide frames). The duplicate lower inspectors in `render_ifvg_replay_tab` are removed; setup mode no longer falls through to an unrelated candidate. Review: initial Unreviewed, five labels (owner Q3), definitions, Unsaved/Saving/Saved, Save Review explicit, notes visibly separated from immutable evidence.

**Data & Audit.** Overview health summary (Artifact integrity, v2/v3 pairing, PK/FK/as-of validity, reconciliation, protected/sealed access, capacity, performance, preparation status, experimental/superseded) as cards {status, observed, limit/rule, meaning, next action} from the existing report keys (`validity_report`, `reconciliation_report`, `identity_report`, `capacity_report.observed/limits`, `performance_report`, `data_access_audit`); access: zero → PASS, nonzero → FAIL, missing/not evaluated → UNAVAILABLE (never green); catalog with human profile names and statuses (runnable / blocked / analysis only / legacy read-only / superseded) first, ids/hashes under Technical details with copy/download; structured summaries for every report (candidate, decision, executed-trade, invariant audit, count reconciliation, v3 schema/identity, access, validity, capacity, performance) with raw JSON in Audit; capacity/performance show observed vs limit and utilisation %.

---

## 8. Exact implementation changes

| File | Symbol / component | Change | Reason | Depends on | Compatibility boundary |
|---|---|---|---|---|---|
| `src/…/ifvg/presentation/run_purpose.py` (new) | `RunPurpose`, `RunPurposeAnnotation`, `PurposeResolutionStatus`, `ResolvedPurpose{purpose, run_scope, namespace_class, store_namespace_id, allowed_stage_plan, allowed_objectives, authorization_readiness, publication_available, result_label}`, `resolve_purpose(...)` | one user-facing purpose authority; persist purpose as mutable annotation; resolve namespace from verified backend envelope; ambiguous legacy purpose → `purpose_unresolved` | F-01, F-13, Q1 | backend `StoreNamespaceEnvelope`, `RunScope`, typed authorization readiness | purpose/annotation never enters charter or pipeline semantic identity |
| `scripts/ifvg_study_tab.py` | `render_ifvg_study_tab`, workspace resolution | remove namespace radio; add `Start` and `Verify Implementation`; resolve local operational root from a verified `store_namespace_id`; visible route label | F-01, F-11, F-13 | run-purpose/provider adapter | `ROUTE_LABELS` additive; local path never defines authority |
| `src/…/ifvg/study_status.py` | `StudyWorkspaceRoute`, `EmptyStateKey`, `EMPTY_STATE_PRESENTATIONS` | additive routes/states (§6.7), including `PURPOSE_UNRESOLVED`, seed-production and typed authorization states | F-11, F-13 | — | persisted status enums remain readable through adapters |
| `src/…/ifvg/study_providers.py` | namespace, listing and authorization read models | exact-load/verify `StoreNamespaceEnvelope`; resolve `store_namespace_id → operational root`; list search/pipeline results by semantic namespace; expose typed `AuthorizationReadiness`, fixture shortlist, `VerificationTradingDayRef`s, seed-production packet/job/seed state and final verification state | F-01, F-03, F-13 | separate backend hardening contracts | read-only adapters; no duplicate authorization or date logic |
| `src/…/ifvg/presentation/charter_satisfiability.py` (new) | `CharterSatisfiabilityReport{rules, passed}`, `evaluate_charter_satisfiability(...)` | rules of §5.4/Q4: FSM search ≥1 axis/≥2 profiles; Evaluate exactly 1 profile; Compare exactly 1 baseline + 1 challenger configuration (any registered diff set); prop objective requires verified contract; Universal requires ≥2 firms; verification excludes research gates | F-04 | current contracts and resolved computation path | pure presentation report; service validator remains authority |
| `src/…/ifvg/search/charter.py` | `validate_charter` | enforce identity-bearing satisfiability rules before save/CLI execution; never silently rewrite objective | F-04 | — | fail-closed tightening; no identity fields added |
| `scripts/ifvg_study_wizard.py` | `_assemble_charter` | authorization by actual path: fully synthetic fixture → `SyntheticAuthorizationMarker`; real `verification_5d` → validated `VerificationAuthorizationRef`; Development Research / Full Authorized Development → validated computation-path-scoped `OwnerAuthorizationBundle`; selected prop objective is never stripped; date policy derives from backend logical-day contract | F-01, F-04, F-08, F-13 | run purpose, provider readiness | charter shape unchanged unless backend accepted contract already changes it |
| `scripts/ifvg_study_wizard.py` | `_freeze_and_launch` | resolve registered runner before spawn; report success only after persisted state appears; typed `RUNNER_UNAVAILABLE` / `LAUNCH_NOT_STARTED` with exact safe fallback | F-02 | runner registry, state provider | no new spawn site |
| `scripts/ifvg_study_wizard.py` | `render_new_study`, `_draft_header`, step bodies | session-only draft until Save/valid Next; purpose annotation; conditional steps; saved-state chip; sticky calculation summary; compare-diff summary; no `~{count}`; stale milestone copy removed | Q2, F-08, F-12 | `presentation/flows.py` | existing draft content preserved through migration |
| `src/…/ifvg/presentation/flows.py` (new) | `StudyFlow{goal, steps, skipped}`, `flow_for_goal` | conditional flows §5.4 | F-08 | run purpose | — |
| `src/…/ifvg/study_drafts.py` | draft annotation/lifecycle | add purpose annotation, `archived`, archive/restore; permanent delete only when never frozen/never launched; exact-name confirmation; migration/bulk archive for empty untitled step-0 drafts; remove or hard-deprecate `discard_draft` in Phase 2 | Q2, F-12 | — | schema bump additive; frozen/run evidence non-deletable |
| `scripts/ifvg_results_tab.py` | `render_history`, `render_results` | lifecycle actions; purpose/namespace read-only filters; artifact-scope badge from artifact envelope; metrics/help/status; distinct no-run/missing/corrupt states | F-01, F-11, Q2 | provider, registries | immutable results remain immutable |
| `scripts/ifvg_pipeline_tab.py` | `render_pipeline_run`, `_render_publish`, `_effective_run_scope`, runtime controls | state-driven actions replace phase radio; gate cache keyed by `(pipeline_id, store_namespace_id, state_sha256)`; activation bound to same verified namespace/head/authorization; worker slider removed/disabled, display sequential V1/effective workers 1; typed readiness reasons | F-03, F-08, F-13 | backend namespace/attempt contracts | no pipeline semantic identity change |
| `src/…/ifvg/search/pipeline.py` | `activate_pipeline_result` | validate same `store_namespace_id`, current supersession-head witness and eligible non-synthetic authorization used by publication gates; refuse verification results | F-03, F-13 | backend hardening contracts | validation/read-model addition; no semantic id change |
| `scripts/ifvg_verification_center.py` (new) | `render_verification_center` | complete §5.3 flow: fixture shortlist/mapping, seed-production packet/signature/job/verified seed, final authorization packet/signature, review/run, filtered monitor/results; exact external commands + refresh where actions remain outside Streamlit | F-08, F-13 | provider adapters, existing launch seams | no synthetic marker on real verification; no research/publication controls |
| `scripts/ifvg_results_charts.py` | heatmap and firm matrix builders | colorscale from `OBJECTIVE_DIRECTIONS`; reversed for minimize; direction-labelled colorbar | F-05 | — | pure builders |
| `scripts/ifvg_verifier_tab.py` | review renderers | Unreviewed sentinel, explicit verdict, corrected Q3 label mapping, definitions, unsaved/saving/saved state | F-06 | review store | existing rows/keys valid |
| `src/…/ifvg/visual_review_store.py` | `REVIEW_VERDICTS` | additive `not_applicable` only if current v1 validator supports it; otherwise stop before schema change | Q3 | — | append-only ledger preserved |
| `src/…/ifvg/context_reporting.py`, `context_report_adapters.py` | reconciliation/access presentation inputs | Phase-1 truthfulness patch: derive `passed/evaluated` from actual gate evidence; unevaluated/default-only evidence never PASS; expose prevalence/reference fields. Phase 3 adds full presentation metadata | F-07, F-09 | — | report JSON additive |
| `scripts/ifvg_lab_tab.py` | Context Research, replay wrapper, Data & Audit | metric/roll-up/help wiring; separate coverage/net-R chart; remove duplicate verifier inspectors and setup fallthrough; health summary and structured reports | F-07, F-09, F-10, F-12 | presentation registry | M0–M3 computation untouched |
| `scripts/ifvg_verifier_tab.py`, `scripts/ifvg_verifier_charts.py` | verifier UI/figures | case card, grouped controls, PIT stage scrubber, dark chart, legend/timeline, table twins, three detail levels | F-10 | labels/help | exact provider and joins unchanged |
| `scripts/ifvg_mbp1_panels.py` | MBP-1 configuration/readiness/audit panels | summary-first readiness, helper text, ratification/offline status; auto-resolve IDs from selected run; manual IDs and registries only in Advanced diagnostics | F-11, F-13 | metric/help/label registries | remains `research_only_offline`; no serving/execution role |
| `scripts/ifvg_regime_panels.py` | regime configuration/readiness/audit panels | summary-first KMeans/coverage/role/promotion status; planned algorithms disabled; manual IDs and full registries in Advanced | F-11 | metric/help/label registries | no change to regime fits/promotion contracts |
| `scripts/ifvg_ui_common.py` | shared UI primitives | detail levels, identity reveal, status/metric/roll-up cards, helper text, saved-state chip, accessible tooltip/focus behavior | §6 | presentation package | additive |
| `src/…/ifvg/presentation/{status_vocabulary,metric_registry,rollups,help_registry,labels}.py` (new) | §6 | centralized presentation metadata and deterministic interpretation | F-09…F-12 | existing metric/gate contracts | pure, unit-tested |
| `FINAL-IMPLEMENTATION-PLAN-DOCS/FRONTEND_UX_CONTRACT.md` and repo docs | named authority sections; D-045/next decision; architecture/readme/pipeline state | formally record Start/Verification routes, conditional flows, state-driven lifecycle, detail levels, purpose annotation, namespace/authorization integration, sequential V1 runtime truth | authority consistency | owner-approved revision 2 | docs-in-same-change; no backend plan merged |
| tests (see §10) | unit, AppTest, scans, browser evidence | cover semantic correctness, workflows, exact links, accessibility and unchanged boundaries | — | isolated synthetic/test stores | no real seed production or five-day verification during UI acceptance |

Boundary summary: pure presentation metadata → `src/…/ifvg/presentation/`; UI state/routing → study/pipeline/verification scripts; read-model adapters and semantic namespace resolution → `study_providers`; charter satisfiability → presentation report plus authoritative `search/charter.validate_charter`; backend namespace/authorization/date/seed semantics are consumed from the separate hardening implementation and are never copied into Streamlit; shared charts remain pure; all test stores are isolated temporary or frozen synthetic stores.

---

## 9. Phased implementation sequence

**Prerequisite and separation.** The backend R6.1-FIX/hardening implementation remains a separate workstream. Before UI Phase 1 closes, reconcile this branch against the actual final backend symbols for `StoreNamespaceEnvelope`, supersession-head witnessing, `SeedProductionAuthorizationRef`, `VerificationTradingDayRef`, `VerificationAuthorizationRef`, authorization readiness, execution attempts and effective workers. Do not copy those contracts into the UI lane. A mismatch that changes authority or workflow requires a concise owner question before proceeding.

Each phase is tests-first and ends in one release-scoped commit. Evidence lives under `..\UI-UX-REDESIGN-PLAN\<phase>\` with TEST_RESULTS/JUnit, Ruff, `git diff --check`, FILES_TOUCHED, DEVIATIONS, and browser evidence where applicable. No push/merge. The UI phase may be accepted independently when its own gates pass; the broader program remains `transitively_blocked_by_R1` until the separately authorized real verification passes.

**Phase 1 — Semantic correctness, backend-contract handshake and truthfulness (F-01…F-05, F-07 minimum, F-08/F-13 part).** Files: backend-interface adapters in `study_providers.py`; `presentation/run_purpose.py`, `presentation/charter_satisfiability.py`, `presentation/status_vocabulary.py`; `search/charter.py`; `ifvg_study_tab.py`; `ifvg_study_wizard.py` (`_assemble_charter`, `_freeze_and_launch`, date policy); `ifvg_pipeline_tab.py` (publication/runtime scope); `search/pipeline.py`; `context_reporting.py`/`context_report_adapters.py` minimal truthfulness patch; `ifvg_results_charts.py`; and the corresponding `FRONTEND_UX_CONTRACT.md`, decisions/architecture/readme/pipeline-state updates. Tests first:

- purpose resolves `RunScope`, namespace class, verified `store_namespace_id`, stage plan and publication availability;
- `RunPurposeAnnotation` round-trips; ambiguous legacy drafts become `purpose_unresolved` and cannot freeze;
- real `verification_5d` refuses `SyntheticAuthorizationMarker`; fully synthetic fixtures require it;
- typed authorization readiness distinguishes missing, stale, superseded, wrong namespace/head, wrong profile/source and ready;
- namespace radio is absent; local path does not define authority;
- contradictory drafts fail before freeze: zero-axis search, baseline-vs-itself, Compare without exactly one challenger, prop objective without verified contract, one-firm Universal, verification with research gates;
- challenger configuration may contain multiple registered differences and every difference is summarized;
- launch refuses an unregistered runner before spawn and reports success only after state exists;
- publication gates/activation are bound to the same `store_namespace_id`, state hash, head witness and eligible authorization;
- worker values above one are not offered and backend refusal is rendered truthfully;
- minimize metrics use reversed colorscales;
- reconciliation/access success derives only from evaluated evidence; missing/default-only evidence is never green.

Acceptance: one source of truth for purpose/scope/namespace; no path can freeze using the wrong authorization class; no silent objective rewrite; launch never claims false success; cross-namespace activation is impossible; worker UI is truthful; known false-PASS banners are removed. Migration: drop the namespace session key; migrate/derive purpose only when unambiguous; otherwise require confirmation. Commit: `UI-1: semantic purpose, namespace and authorization truth; satisfiability; honest launch and evidence`.

**Phase 2 — Verification Center, conditional New Study, drafts and review (F-06, F-08, F-12/F-13 workflow).** Files: `ifvg_verification_center.py`, `presentation/flows.py`, `ifvg_study_wizard.py`, `study_drafts.py`, `study_providers.py`, `ifvg_results_tab.render_history`, `ifvg_verifier_tab`, `visual_review_store`. Build the complete UI sequence for logical-day shortlist → seed-production packet/ref/job/verified seed → final verification packet/ref → review/run → filtered monitor/results. When signing or job execution is external, provide exact command/packet path/refresh semantics. Remove/hard-deprecate unsafe `discard_draft` in this phase.

Tests first:

- Verification Center never renders research gates, prop/risk/model controls or Publish;
- logical trading days and physical partitions are displayed separately;
- seed production cannot launch without a validated `SeedProductionAuthorizationRef`;
- final verification packet cannot exist before a verified profile-matching seed;
- real verification cannot launch without a validated final `VerificationAuthorizationRef`;
- monitor lists only stages in the resolved seed/verification plan;
- conditional flows skip irrelevant steps with visible reasons;
- selected prop objective blocks rather than disappearing when no contract is available;
- session-only draft writes no file until Save/valid Next; purpose annotation persists;
- archive/restore/delete rules and one-time bulk archive preserve evidence;
- review opens Unreviewed, persists nothing without explicit Save, uses owner-approved label mappings, and keeps existing v1 rows valid.

UI implementation acceptance uses isolated synthetic/test namespaces, signed test fixtures and synthetic job states. It does **not** produce a real seed, sign owner authorization, or execute the five-day run. Acceptance proves the owner can traverse the complete workflow and see exact next actions once real backend packets exist. Commit: `UI-2: complete Verification Center, goal-derived flows, safe drafts and explicit reviews`.

**Phase 3 — Shared metric/helper/status system; Context Research, Results, MBP-1 and regime panels (F-07 full, F-09, F-11 part).** Files: `presentation/{metric_registry,rollups,help_registry,labels}.py`, `ifvg_ui_common.py`, `ifvg_lab_tab.py`, `context_reporting.py`, `context_report_adapters.py`, `ifvg_lab_charts.py`, `ifvg_results_tab.py`, `ifvg_pipeline_tab._ladder_frame`, `ifvg_mbp1_panels.py`, `ifvg_regime_panels.py`. Tests first: registry coverage of displayed metric keys; metric status rules; deterministic roll-ups; no unknown evidence shown PASS; coverage and net R use separate scales; MBP-1/regime summary fields resolve from selected run while manual IDs remain Advanced-only; proposed/ratified/offline/planned statuses are distinct; helper/glossary coverage with explicit `HELP_EXEMPTIONS`; accessible names for every collapsed/hidden label. Acceptance: every consequential metric/control/status is interpretable; no arbitrary thresholds; no registry dump or exact-ID paste field dominates normal workflow; MBP-1 remains offline-only and regime roles are honest. Commit: `UI-3: metric and help system; research results; MBP-1/regime presentation`.

**Phase 4 — Replay / Verifier redesign (F-10).** Files: `ifvg_verifier_tab.py`, `ifvg_verifier_charts.py`, `ifvg_lab_tab.render_ifvg_replay_tab`. Tests first: case card separates actual/counterfactual/model/reviewer evidence; controls grouped by purpose; PIT scrubber names hidden evidence; setup mode never falls through; duplicate lower inspectors removed; dark chart has legend/timeline/table twins; exact deep links resolve. Acceptance: a trader can understand a case before opening audit details; raw exact evidence remains available; exact-ID semantics unchanged. Commit: `UI-4: Replay / Verifier case summary, evidence boundary, chart and audit tiers`.

**Phase 5 — Data & Audit redesign (F-12 part).** Files: `ifvg_lab_tab.render_ifvg_data_audit_tab`, new `ifvg_audit_panels.py`, report-field registry rows. Tests first: health cards derive from actual report fields; access unknown/unavailable is not PASS; capacity/performance shows observed, limit and utilization; human catalog status precedes IDs; raw JSON remains available. Acceptance: owner can assess system health without reading JSON while auditors retain exact reports. Commit: `UI-5: Data & Audit health overview and structured reports`.

**Phase 6 — State-driven lifecycle, Active Runs/History polish and mandatory responsive/accessibility acceptance (F-11 rest).** Files: `ifvg_active_runs_tab.py`, `ifvg_results_tab.py`, `ifvg_pipeline_tab.py`, table `column_config`, `hardening_ui_smoke_app.py`, browser manifest/focus evidence. Tests first: lifecycle actions follow state; unavailable navigation is not clickable; no-runs/not-selected/not-applicable never use artifact-unavailable; tables have readable columns and selected-row detail; keyboard focus order and tooltip access; charts have table equivalents.

Browser evidence is mandatory at 1440×900, 1024×768, 768×1024 and 390×844 for populated, empty, blocked, inconclusive, corrupt, failed, running, completed, verification-only, research-only and legacy/superseded states. Every required state/viewport must be captured and pass. Browser/session unavailability leaves UI acceptance **open**; it may be documented as code-complete but must never be called accepted. Commit: `UI-6: state-driven lifecycle, responsive layout and accessibility acceptance`.

---

## 10. Test and evidence matrix

| Area | Test / evidence | Fixture | Assertion | Failure meaning |
|---|---|---|---|---|
| Backend contract handshake | `test_ui_uses_verified_store_namespace_contract`, `test_ui_authorization_states_match_backend_contracts`, source scan for duplicate contract definitions | isolated backend-contract fixtures | UI imports/adapts final backend namespace/date/seed/auth/attempt types; no duplicate authority in scripts/presentation | UI and backend plans diverged |
| Purpose / namespace persistence | `test_run_purpose_resolves_scope_namespace_and_locks_options`, `test_purpose_annotation_round_trips`, `test_ambiguous_legacy_purpose_blocks_freeze`, `test_namespace_radio_is_gone` | three purposes + legacy drafts | semantic namespace id, scope, steps, objectives, stage plan, publication and label deterministic; purpose annotation non-semantic | two sources of truth or silent legacy inference |
| Charter satisfiability | `test_contradictory_drafts_fail_before_freeze[…]`, `test_validate_charter_refuses_degenerate_charters`, `test_one_challenger_may_have_multiple_registered_diffs` | synthetic drafts per §5.4 | named failure before freeze; service validator raises; no objective rewrite | contradictory intent persists |
| Authorization class | `test_real_verification_refuses_synthetic_marker`, `test_fully_synthetic_fixture_requires_synthetic_marker`, `test_authorization_readiness_typed_failures` | synthetic/test envelopes | exact authorization class required; missing/stale/superseded/wrong namespace/head/profile/source distinguished | authorization weakened or misleading |
| Verification Center | `test_verification_center_never_renders_research_steps_or_publish`, `test_logical_days_and_physical_partitions_are_distinct`, `test_seed_production_requires_signed_ref`, `test_final_verification_ref_requires_verified_seed`, `test_verification_launch_requires_final_ref`, `test_verification_monitor_filters_active_stages` | isolated test namespace; fake signed refs/job states | complete two-stage owner flow, no real source reads, no research controls | owner workflow incomplete or unsafe |
| Launch blocking | `test_launch_refuses_unregistered_runner_before_spawn`, `test_launch_reports_started_only_after_state_exists` | unregistered key; slow/failing worker | typed state, no spawn; "launched" only with persisted state | false success |
| Publication / namespace | `test_publish_gate_cache_is_namespace_state_and_head_bound`, `test_activation_refuses_synthetic_verification_and_namespace_mismatch` | gates in namespace A, activation in B | refusal with human reason | cross-namespace activation |
| Runtime truth | `test_worker_control_is_absent_or_fixed_to_one`, backend refusal fixture | V1 sequential attempt | no operative >1 worker control; receipt shown as sequential/effective 1 | UI advertises nonexistent parallelism |
| State mapping | `test_empty_states_never_use_artifact_unavailable_for_no_runs`, `test_lifecycle_actions_follow_state` | every §6.7 state | exact state and allowed actions | misreported state |
| Reconciliation/access truth | `test_reconcile_banner_derives_from_evaluated_gates`, `test_unknown_access_evidence_is_never_pass` | pass/fail/unevaluated reports | green only for evaluated pass | false assurance |
| Metric status | `test_metric_status_rules[…]`, `test_metric_registry_covers_every_displayed_key` | registry fixtures | deterministic status/reference/sample/caveat; no arbitrary AUC/calibration bands | arbitrary or missing interpretation |
| Helper coverage | `test_help_registry_covers_required_controls`, `test_help_exemptions_are_explicit`, `test_glossary_defines_listed_acronyms`, accessible-name scan | source/AppTest | every consequential element has help; exemptions justified; no inaccessible collapsed label | unexplained or inaccessible control |
| Review default | `test_review_default_is_unreviewed_and_nothing_persists_without_save`, verdict mapping tests | verifier AppTest | no ledger row until Save Review; v1 rows render; additive key only | false verdict persisted |
| Exact deep links | exact candidate/decision/trade/setup jump tests | immutable/synthetic pair | exact resolution or sanitized unresolved state; no fuzzy path | exact-ID guarantee weakened |
| MBP-1 / regime panels | summary/readiness and Advanced-only ID tests | synthetic R5B/R6 artifacts | normal panel human-readable; planned/offline/unratified states distinct; manual IDs not primary | registry console remains user workflow |
| Forbidden controls | existing scans extended to new scripts | source scan | no sealed/unlock/recapture/live/order/Trade-Lab serving controls; no new spawn site | safety regression |
| Responsive / accessibility | screenshot manifest × four viewports × required states; `focus_order.json`; keyboard/tooltip/table-equivalent evidence | `hardening_ui_smoke_app.py` and isolated test stores | all required captures and interactions pass; browser unavailable = gate open/fail, never pass | UX acceptance unproven |
| Evidence / immutability | Ruff, `git diff --check`, JUnit, zero unexpected tracebacks/console errors, isolated temporary-store before/after inventory, M0–M3 and propsim goldens | isolated roots | only intended UI/docs/test files change; no real replay/seed/model/prop run | scope or immutability regression |

The UI implementation suite must not sign owner artifacts, launch seed production against real sources, or execute the real ≤5-day verification. Those are post-implementation owner actions governed by the separate backend plan.

---

## 11. Acceptance criteria

UI implementation acceptance proves, using isolated synthetic/test stores and typed test authorization envelopes, that an owner without source knowledge can:

- understand every task card, purpose, namespace, status and workflow from visible helper text;
- configure only satisfiable studies, with contradictions refused before freeze and no silent objective rewrite;
- distinguish Evaluate, Compare, FSM Search, Prop Benchmark, Universal Prop Search, Development Research, Full Authorized Development and Implementation Verification;
- use the Verification Center to review logical trading days and physical partitions, prepare/refresh seed-production authorization, monitor seed status, review the concrete seed, prepare/refresh final verification authorization, and reach the bounded Launch action—without seeing unrelated research or publication controls;
- see why any action is blocked in one sentence plus exact Advanced evidence;
- interpret every displayed metric and section roll-up (name, definition, direction, reference/gate, status, sample, caveat) without reading a technical key;
- distinguish PASS, FAIL, BLOCKED, WARNING, INCONCLUSIVE, INFORMATIONAL, NOT APPLICABLE, UNAVAILABLE and CORRUPT by glyph and word, never by color alone;
- review an exact setup/trade from a persistent case card, lifecycle timeline and chart; distinguish actual execution, counterfactual labels, model probabilities and reviewer judgment;
- inspect artifact, access, validity, capacity and performance health without opening raw JSON while retaining exact IDs/hashes/manifests under Technical identity & audit;
- use MBP-1 and regime panels as readable readiness/research surfaces rather than manual-ID registry consoles;
- operate all primary paths by keyboard at 1440×900, 1024×768, 768×1024 and 390×844, with every required browser state captured and passed.

The actual seed-production authorization, seed-producing replay, final owner `VerificationAuthorizationRef`, and real ≤5-day verification are not UI implementation acceptance actions. They occur afterwards under the separate backend plan.

---

## 12. Explicit non-goals and preserved guarantees

Not changed: Strategy-Core; strategy behavior, labels, feature formulas or prop rules; M0–M3 identities and Context Research computation; immutable scientific artifacts/stores; exact-ID loading (no fuzzy, nearest-time, keep-last or row-order joins); automatic model/feature/threshold/cluster selection; protected/sealed access; S11; MBP-1's `research_only_offline` boundary; live serving, orders or production activation; Trade-Lab; the legacy ML Training Workbench and unrelated product areas; `RunScope` values; existing `ifvg_visual_review_v1` rows/keys except the explicitly additive `not_applicable` value if validator-compatible; charter/pipeline semantic identities except fail-closed validation; or any backend contract owned by the separate R6.1 hardening plan.

This UI plan does not implement `StoreNamespaceEnvelope`, owner-decision supersession, seed-production policy/contracts, logical-day mapping, final verification authorization, or sequential executor semantics. It consumes their verified interfaces and must stop on an authority mismatch.

Upon approval, update `FRONTEND_UX_CONTRACT.md` rather than leaving contradictory authority text:

- §3.2: Experiments navigation gains Start and Verify Implementation and loses the mutable namespace selector;
- §5.1: all major screens use Summary / Research details / Technical identity & audit, preserving the original progressive-disclosure intent;
- §7: the fixed eight-step wizard is superseded by goal-conditional flows and a dedicated Verification Center;
- §30: Configure/Preview/Launch/Monitor/Resume/Publish becomes a state-driven lifecycle; verification has no Publish route;
- add presentation-only `RunPurposeAnnotation`, semantic namespace display, typed authorization readiness, helper/metric metadata and mandatory browser acceptance.

Record the decision in `docs/DECISIONS.md` and update `ARCHITECTURE.md`, `docs/README.md` and `docs/pipeline_state.yaml` in the corresponding phase. Every guarantee in §1 "What works" remains preserved.
