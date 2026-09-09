# Architecture Map — Current State of Claude-Quant-Lab and Strategy-Core

**Document type:** Supporting document to `IMPLEMENTATION_PLAN.md` (brief §18 item 2)
**Status:** Plan-only. Describes the codebase as inspected on 2026-08-17. No code was changed while producing this map.
**Repos inspected:**
- Claude-Quant-Lab (`QL`) — `C:\Users\gonza\Documents\Claude-Quant-Lab`, package `alpha_lab` (distribution `alpha-signal-lab`)
- Strategy-Core (`SC`) — `C:\Users\gonza\Documents\Strategy-Core`, package `strategy_core`, HEAD `a4e3303179ac6a1088aecaaa3482934cf1aec4d7` (branch `platform-refactor`), pinned by QL as a **non-editable VCS wheel**

All paths below are repo-relative unless prefixed `SC/` or `QL/`.

---

## 1. Repository roles and integration

- **Strategy-Core** is a *library only* (no console entry points, no scripts beyond one seam check). It owns the deterministic sequential IFVG FSM, the vectorized Databento reader, time bars, sessions, context features, and label kernels. `PLATFORM_VERSION = "strategy_core_platform_v1"` (the old `ENGINE_VERSION` v1→v3 lineage is preserved in the `strategy_core/__init__.py` docstring); `CONTRACT_VERSION = "trade_lab_contract_v3"`; per-plugin `IFVG_STRATEGY_VERSION = "2"`.
- **Claude-Quant-Lab** owns research artifacts, immutable stores, the experiment framework, cost/label/statistics layers, the prop simulator, all scripts, and the Streamlit dashboard.
- QL depends on SC via `pyproject.toml`: `strategy-core @ git+https://github.com/thealgochef/Strategy-Core.git@a4e3303…`. Editing `SC/src` has **no effect** on QL until commit → pin bump → reinstall. `QL/src/alpha_lab/agents/data_infra/ifvg/verification.py::_strategy_core_repository_root` **raises if the installed pin ≠ the sibling SC checkout HEAD**.
- QL's own package needs `pythonpath = ["src"]` (pytest ini) / `sys.path.insert(0, ROOT/"src")` bootstrap in every script.
- CI (`.github/workflows/ci.yml`): Python 3.13, `pip install -e ".[dev]"`, `ruff check src tests`, `pytest`. **`scripts/` (the entire UI) is outside CI lint/type coverage today.**

---

## 2. Claude-Quant-Lab layout

```
src/alpha_lab/
  agents/data_infra/ifvg/     ← 41 modules, ~19.5k LOC — the IFVG research lane (the heart of this plan)
  agents/data_infra/ml/       ← touch_reversal ML lane (dashboard-utility caches, D-036)
  agents/{execution,monitoring,orchestrator,signal_eng,validation}/   ← legacy multi-agent scaffold
  experiment/                 ← legacy
  propsim/                    ← prop-firm evaluation simulator (D-038 window) — see §9
scripts/                      ← 38 CLIs + all Streamlit UI modules
tests/                        ← 66 pytest files (tests/agents/, tests/core/, tests/integration/, tests/propsim/)
config/                       ← instruments.yaml, prop_firms.yaml (superseded by propsim presets), settings.yaml
data/ifvg_datasets/           ← immutable artifact stores (v2, v3, fsm_audit/v1, replay_chart/v1, context_views, catalogs)
data/ifvg_experiments/        ← context_v1 run store + legacy 8-hex experiment dirs + sealed_ledger.jsonl
data/ifvg_preparation_jobs/   ← mutable job state roots
docs/                         ← DECISIONS.md (D-001…D-038), ARCHITECTURE.md, pipeline_state.yaml, docs/ifvg/*
```

Key deps: pandas ≥2.1, pyarrow ≥18, pydantic ≥2.5, duckdb ≥1.0, streamlit ≥1.38 (installed 1.54.0), plotly ≥5.18, catboost ≥1.2 (installed 1.2.10), scikit-learn ≥1.4 (installed 1.7.0 — KMeans/MiniBatchKMeans/GaussianMixture/SpectralClustering/Nystroem all importable).

---

## 3. Immutable artifact-store discipline (`ifvg/manifest.py`)

The canonical save protocol every new store in this plan clones:

1. Refuse unless the required gate reports pass (v2: `invariant_audit.passed`; v3: validity/coverage/reconciliation/capacity/identity/performance all pass; fsm-audit: parity + reconciliation + contract fingerprint).
2. `dataset_id = canonical_sha256(identity.payload())` — full 64-hex SHA-256; directory names are never mutable labels.
3. `if destination.exists(): raise FileExistsError` — overwrite refusal.
4. Write into a sibling tmp dir; validate every table (PK / identity / FK) + count reconciliation.
5. Manifest = `{**core, "manifest_payload_sha256": canonical_sha256(core)}` with per-file sha256/bytes/rowcounts/Arrow-schema hashes.
6. `os.replace(tmp, destination)` — the rename **is** the publication point; `shutil.rmtree(tmp)` on failure.

Helpers: `canonical_sha256(value)`, `file_sha256(path)`, `source_tree_hash(root, relative_paths)`, `RepositoryState`/`read_repository_state(name, root, source_paths)` (HEAD + dirty-status sha + source-tree hash; **a dirty tree changes every downstream identity**). `_reject_performance_payload` bans outcome-bearing keys from measurement-only lanes.

| Identity | ID fn | Saver | On-disk root |
|---|---|---|---|
| `DatasetIdentity` | `dataset_id_for` | `save_v2_dataset_immutable` | `data/ifvg_datasets/v2/<id>/exploration/` |
| `V3DatasetIdentity` | `v3_dataset_id_for` | `save_v3_dataset_immutable` | `data/ifvg_datasets/v3/<id>/exploration/` |
| `FsmAuditIdentity` | `fsm_audit_dataset_id_for` | `save_fsm_audit_immutable` | `data/ifvg_datasets/fsm_audit/v1/<id>/exploration/` |

Verified reloaders: `ifvg/artifact_io.py` — `resolve_artifact_directory` (exact-ID only, never lists the catalog root, escape-guarded), `load_verified_v2/v3_artifact`, `load_verified_ifvg_pair`, `load_verified_label_source_bars` (hard cutoff assert at 2026-06-10T21:00:00Z). Lightweight store to clone: `ifvg/context_run_store.py` (`_atomic_directory`/`_publish`) plus the save→reload→assert-equality pattern in `context_experiment_service._persist_and_verify_{view,run}`.

Content-hash helper for tables: `ifvg/dataset.py::_table_content_hash(table, frame)` + `_canonical_cell` (currently private — this plan promotes it).

---

## 4. IFVG research artifacts

- **v2 core** (`ifvg/contracts.py`, schema version 2): `RecordTable` StrEnum — `setup_lifecycle_event`, `entry_candidate`, `candidate_label`, `eligible_decision`, `executed_trade`, `geometry_dossier`, `quarantine` — with PK/identity/FK validators and `count_reconciliation`. Built by `ifvg/dataset.py::build_ifvg_v2_capture(dates, cfg, resolved_profile, *, access_policy, cached_artifacts_only, progress_fn)` over the per-day driver `ifvg/capture_driver.py::capture_single_date`.
- **`ifvg_fsm_audit_v1`** (`ifvg/audit_contracts.py`, schema v1): 13 `AuditTable`s (6 trace-derived, 5 audit-channel, parentless-interval derived, day_funnel), `audit_contract_fingerprint()`, `reconcile_funnel_to_audit`. Parity gate `ifvg/fsm_audit_parity.py::compare_v2_exact` pins the accepted doc-default v2 dataset (`FSM_AUDIT_ACCEPTED_V2_DATASET_ID = 143b510f…`). On-disk artifact `7e55ee89…`.
- **`ifvg_replay_chart_v1`** (`ifvg/replay_chart_store.py`): resample rule `trading_day_18et_elapsed_v1`, timeframes (180…14400 s), stage-gating policy `stage_gate_ordinal_cursor_ts_v2`, `AUTHORIZED_CUTOFF_UTC = "2026-06-10T21:00:00Z"`. Presentation providers: `replay_chart_provider.py` (PIT candidate evidence by stage), `setup_verifier_provider.py`.
- **Context/"formula v2" (v3)** artifacts: `context_contracts.py`, `context_schemas.py` (Arrow registry hashes); feature-formula authority is SC `strategies/ifvg_smc/context_config.py` (`FEATURE_FORMULA_VERSION = "ifvg_context_formula_v2"`, `context_config_hash`, `feature_schema_hash`).
- **Funnel reports**: `ifvg/funnel_report.py` (doc-default floors/caps, `IFVG_FUNNEL.md`/`IFVG_LABELS.md`); FSM-lane reports via `scripts/ifvg_fsm_audit_reports.py` → `reports/ifvg_fsm_audit/*` (JSON + Markdown always emitted as matched pairs).
- **Review ledger**: `ifvg/visual_review_store.py` — append-only JSONL + CSV export.

---

## 5. The M0–M3 context experiment lane (the contract template for every new lane)

All in `src/alpha_lab/agents/data_infra/ifvg/`:

| Module | Load-bearing symbols |
|---|---|
| `context_experiment_contracts.py` | Frozen pydantic v2 (`frozen=True, extra="forbid"`); `canonical_contract_sha256`; `context_run_identity(...)` — **run id folds in the exact normalized OOS prediction stream**; `ProfileCapabilityStatus` (RUNNABLE/BLOCKED/ANALYSIS_ONLY/LEGACY_READ_ONLY) + `PROFILE_CAPABILITY_REGISTRY` (fail-closed capability pattern); `ContextFeatureTier` (M0/M1/M1+240/M2/M3); `IfvgContextExperimentConfig` (Literal-locked: train 40 / test 5 / step 5 / embargo 2 / min-train 30 / bootstrap 10 000 / seed 7); `IfvgContextFoldDefinition`; `IfvgContextFeatureIdentity`; `IfvgContextRunReconciliation` (the paired-delta compatibility gate); model protocol `ifvg_context_catboost_binary_v1` (pinned params incl. `random_seed 7, thread_count 1`), calibration `raw_probability_diagnostics_v1`, threshold grid (0.40/0.50/0.60/0.70) |
| `context_folds.py` | `build_context_folds` — expanding walk-forward with setup-grouped boundary exclusion, interval purge, 2-day embargo, duplicate-OOS assertion, per-fold `training_prevalence` |
| `context_model.py` | `resolve_context_model_protocol` (resolved hash includes catboost/numpy/pandas/sklearn versions + python), `run_context_fold_models`, `MISSING_CATEGORY = "__MISSING__"`, `CATEGORICAL_FEATURE_REGISTRY`, `_permutation_importance` |
| `context_statistics.py` | `binary_prediction_report` (Brier, **reference Brier from per-row training prevalence**, Brier skill, log loss, AUC + single-class reason, calibration intercept/slope, reliability bins, threshold coverage/net-R); `block_bootstrap_interval` (10 000 reps, seed 7, setup-cluster and trading-day blocks); `paired_tier_delta_report` (identical-OOS-ids paired deltas); `candidate_uncertainty_report`; `feature_importance_report` |
| `context_feature_view.py` | `TIER_FEATURE_REGISTRY` (cumulative M0⊂M1⊂M2⊂M3 ladder), `build_candidate_feature_view(pair)` → content-hashed `view_id`, `apply_observation_filters`, `m3_cohort_status` |
| `context_labels.py` | `derive_context_candidate_labels` — label identity includes a hash of the bar evidence actually consumed |
| `context_reporting.py` / `context_experiment_service.py` | `execute_context_experiment`, `run_and_catalog_context_experiment`, `_persist_and_verify_*`, `CONTEXT_COHORT_REGISTRY` |

Identity chain: config hash → view_id → label_derivation_id → resolved model-protocol hash → run_id (with OOS stream). `save_context_experiment_run` recomputes the run id and refuses on mismatch.

---

## 6. Prior search infrastructure (reference, not directly reusable for FSM deltas)

- `scratch_ifvg_search.py` (962 LOC, **untracked**, repo root): the 42/100-eval overnight search orchestrator — budget cap, phase/day-window policy, mkdir-lock ledger mutex, atomic ledger writes (tmp + `os.replace`), `_sealed_precheck/_sealed_postcheck` (`sealed_ledger_count() == 0` on both sides of every run), subcommands `recon|run|status|declare-shortlist|validate|export-*`.
- `IFVG_SEARCH_LEDGER.json` (meta + 45 entries; every entry stamps `sealed_ledger_count: 0`), `ifvg_search_runs/`, `ifvg_search_trades/*.csv` (a propsim-ready trade-stream shape: setup_id, trading_day, direction, sessions, risk_points, net_r, mfe_r/mae_r, …).
- **Caveat:** it searches *downstream filters/scoring/model* over one already-captured trace via the legacy `ifvg/experiment.py::run_ifvg_experiment`, which (a) hard-refuses the repaired v2 identity lane and (b) would violate the brief's §9.5.9 ("never derive strategy execution by filtering an existing candidate table") if reused for FSM deltas. Only its ledger/budget/lock idioms are imitated. The v2-safe single-config evaluator is `run_ifvg_v2_evaluation`.

---

## 7. Job / orchestration primitives

| Primitive | Location | Behavior |
|---|---|---|
| Resumable checkpointed job | `ifvg/preparation.py::run_resumable_preparation_job` | Checkpoints only at authorized **date boundaries**; on-disk checkpoint must be a prefix of requested dates; atomic `state.json` before/after each date; honors a `cancel.requested` sentinel; `FAILED` with `error_code` on exception |
| Job state | `PreparationJobState{profile_name, status, completed_dates, current_date, v2_artifact_id, v3_artifact_id, error_code}` + `read_preparation_state` | Roots under `data/ifvg_preparation_jobs/<profile>/` |
| Exclusive lock | `ifvg/preparation.py::_profile_lock` | `os.open(..., O_CREAT|O_EXCL)`, pid recorded, unlink in `finally` |
| Atomic JSON | `_write_json_atomic` | tmp + `os.replace` |
| Detached process launch | `scripts/ifvg_preparation_job.py` (`start|status|cancel`), `scripts/ifvg_recapture_job.py` (`write_job_status`/`read_job_status`) | `subprocess.Popen` + `CREATE_NO_WINDOW`; JSON status file; **no UI reads these today** |
| Process pool | `scripts/w3_cache_warmer.py` | `ProcessPoolExecutor(max_workers, max_tasks_per_child)`, spawn-safe top-level worker, deferred heavy imports, skip-if-loads resumability, verify-after-write self-heal, peak-RSS reporting, `_suggest_workers(peak_gb)` |
| Sequential-only warmer | `scripts/ifvg_artifact_warmer.py` | **Hard-rejects `--workers != 1`** — IFVG day artifacts are seed-chained |

---

## 8. Data access & safety (`ifvg/development_access.py`, `ifvg/data_access.py`)

- `DEVELOPMENT_CUTOFF_UTC = "2026-06-10T21:00:00Z"`; `FROZEN_WARMUP_DATES` (exactly 10, 2026-01-01…01-12); `PRIOR_RESEARCH_DATES` (01-13…04-30); `EXPOSED_DEVELOPMENT_DATES` (05-01…06-10); `PERMITTED_DEVELOPMENT_DATES` = the union (calendar dates including weekends; the trading-day subset is what has source files).
- `SourceDateClass`: 2026-06-11 → `PROTECTED_BUFFER`; ≥ 2026-06-12 → `SEALED`.
- `DevelopmentDataAccess`: **the date is authorized before the path is constructed**; the parent directory is never listed; every operation appends to an event chain whose SHA-256 lands in `data_access_audit.json` inside every immutable artifact; `assert_zero_protected()`.
- `DevelopmentReplayPolicy` (policy id `development_explicit_dates_before_path_v2`): requires the frozen ten-date warmup as an exact prefix.
- **Gate relevant to this plan:** `data_access.require_fixed_exploration_allowlist` admits only (a) the fixed 26-day January repair allowlist or (b) a `DevelopmentReplayPolicy`. A ≤5-day verification run is **not expressible today** — the plan adds a third trusted policy class (`VerificationReplayPolicy`, see `CONTRACTS_AND_SCHEMAS.md` §6).
- Sealed look counter: `ifvg/experiment.py::sealed_ledger_count()` over `data/ifvg_experiments/sealed_ledger.jsonl` (global, per-look).

---

## 9. Prop simulator — `src/alpha_lab/propsim/` (exists; evaluation-phase only)

| File | Contents |
|---|---|
| `models.py` | `TradePath(day, entry_ts, points_optimistic, points_conservative, mfe_pts, mae_pts, resolution)`; `Ruleset(starting_balance, profit_target, trail_amount, trail_style, trail_locks_at_start, dll_amount, dll_hard, consistency_pct, min_days, max_eval_days, point_value)`; `WalkResult`. Trading day = 18:00 ET roll, dated by entry |
| `engine.py` | Current legacy `EvaluationWalk` — `BREACH_MODES = ("realized_only", "unrealized_adverse_first")`; the target lifecycle contract removes scenario ordering from firm rules and retains this only as legacy/current-state evidence; trail styles `eod_floor_realtime_breach | intraday_peak_trail | static_floor`; barrier semantics on the adverse leg; soft DLL force-closes at the level and halts the day; PASS at EOD with min-days + consistency; expiry on day max+1; `group_by_day`, `walk_days` |
| `bootstrap.py` | `run_bootstrap` — **day-level block-bootstrap Monte Carlo**, n=10 000, seed 42, one seeded `np.random.default_rng`, Wilson CIs |
| `presets.py` | `PRESETS` = `topstep_50k`, `apex_50k_eod`, `apex_50k_intraday`, `tpt_50k_test` — **presets are data** (two carry ⚠ verify-at-dashboard flags) |
| `loaders.py` | Trade-Lab executions dir / prediction journal / OOS-parquet loaders; dedup by physical fill signature; graceful degradation to realized-only with a stated reason. **No loader exists for IFVG `EXECUTED_TRADE` tables or `ifvg_search_trades/*.csv`** |
| `report.py` | `build_report` — full (fill column × breach mode) matrix, per-cell historical + bootstrap; `format_human` |
| `cli.py`/`__main__.py` | `python -m alpha_lab.propsim …` |

**Missing vs the brief** (added by this plan): funded-phase state machine, payout eligibility/caps/splits/waiting periods, fees (activation/recurring/reset), breach/replacement events + account timelines, risk-policy contracts, portfolio topology on one common path, stress scenarios, the payout-reliability vector, multi-config × multi-firm sweeps, immutable persistence of sim results.

**Nothing exists anywhere** for: Pareto/non-dominated frontier, parameter-sensitivity generation, neighbor-stability/knife-edge analysis, a search charter contract, a tracked parent/child orchestrator.

---

## 10. Dashboard / UI layer (pure Streamlit + Plotly)

`dashboard-ui/` is **dead** (orphaned `node_modules` only; the React prototype was deleted in the 2026-07 cleanup). There is no FastAPI server in use.

- Entry: `streamlit run scripts/dashboard.py` → 8 top-level `st.tabs`; `🧪 IFVG Lab` → `scripts/ifvg_lab_tab.py::render_ifvg_lab_tab()` → `st.tabs(["Experiments", "Replay / Verifier", "Data & Audit"])` (labels asserted in `tests/agents/test_ifvg_lab_tab.py:425-429`).
- Conventions: `scripts/<name>_tab.py` (widgets/caching/layout, `render_*_tab(st_module=st)` injection seam) + `scripts/<name>_charts.py` (pure `__all__`-exported Plotly builders); pure providers/stores in `src/`; `_STATE_PREFIX = "ifvg_context_v1_"` session-state namespacing; caches keyed by full artifact IDs + manifest hashes, hash re-verified after load, `_`-prefixed unhashed context args; `_sanitize_error` (path/secret scrubbing — currently duplicated in two files); CLI escape hatch (`st.info` + `st.code` with the exact command) when an artifact is not ready.
- Verifier drill-down: `scripts/ifvg_verifier_tab.py::queue_jump(kind, value)` (kinds `candidate_id|decision_id|trade_id`, `ValueError` otherwise) + `_apply_pending_jump` via `replay_chart_provider.resolve_selection`; `render_verifier_section(st_module, pair, entry)`; selectable-dataframe→jump recipe (`st.dataframe(on_select="rerun", selection_mode="single-row")` → `_queue_verifier_jump` → `st.toast`); candidate mode + setup mode + PIT stage slider (`tap/parent/lock/opposing/inversion/entry/resolution`).
- Charts: `ifvg_verifier_charts.py::build_verifier_figure` (3-pane 1m/parent/HTF), `LAYER_BUDGETS` + `OmissionReport` (honest per-layer trace truncation with "N omitted" captions), `to_display_timezone` (America/New_York). Orphaned-but-tested builders in `ifvg_lab_charts.py`: `build_equity_figure`, `build_r_histogram_figure`, `build_calibration_figure`, `build_coverage_figure`.
- Results-panel skeleton: `ifvg_lab_tab._render_result` (KPI metric rows, PASS/FAIL `st.metric` badges, raw-audit `st.expander` + `st.json`); run-vs-run comparison with a compatibility gate (`_run_history`, `_config_diff`, `_metric_delta_frame`).
- Testing: Streamlit AppTest harness (`test_ifvg_lab_tab.py:406-427` — monkeypatch module-level loader seams, re-import inside `_app()`, `AppTest.from_function`, `assert not at.exception`); source-introspection contract tests (`test_ifvg_verifier_tab.py` — forbidden button labels `delete|sealed|recapture|promote|unlock`, `allow_sealed` banned from source).
- **Gaps:** no wizard/stepper, no live job monitor (no `st.fragment`/auto-refresh/polling anywhere), no in-UI launcher for long jobs, no frontier/heatmap-of-configs views, no markdown-report viewer, no pagination, no deep links, no theming.

**Target frontend authority:** `FRONTEND_UX_CONTRACT.md` is the complete target-state UI contract. It preserves the existing IFVG Lab shell and exact verifier while adding the decomposed `ifvg_ui_common` / study router / eight-step wizard / Active Runs / Results+History / comparison+timeline / pure chart builders / pipeline UI modules. It normatively defines field-level wizard controls, immutable draft/freeze/clone behavior, monitor columns and fallbacks, all result charts and table twins, pipeline retry/publication semantics, exact drill-down, empty/blocked/failure states, and keyboard/responsive/screenshot acceptance at 1440×900, 1024×768, 768×1024, and 390×844. The new UI scripts are brought under Ruff; pure presentation logic moves to `src` for unit coverage.

---

## 11. Strategy-Core: profile, FSM, replay, reader

### 11.1 Profile = `IfvgSmcSection` (`SC/src/strategy_core/strategies/ifvg_smc/section.py`)

A pydantic model (`extra="forbid"`); `ifvg_profile_hash(section)` (sha256 of the sorted JSON dump) is the profile identity, stamped as `section_config_hash` on every record envelope. Five named builders; only `ifvg_v2_doc_default_fresh_static_1r` is `runnable`. Full field inventory and searchability classification: see `CONTRACTS_AND_SCHEMAS.md` §1.2.

**Traps confirmed:** `break_even_enabled` and `legacy_candidate_row_limit` are in the section (and the hash) but are **never threaded into `IfvgReducerConfig`** — varying them burns full replays for zero behavioral delta. `parent_reaction_window_1m_bars_max` is a legacy read-only field.

**Identity-critical fact (verified; drives the revision's identity decomposition):** `profile_name` is a section field (section.py:97) and `ifvg_profile_hash` hashes the **full** `model_dump` including it (section.py:324–331); that hash is embedded in every record ID (`make_setup_id(profile_hash, htf_fvg_id, tap_cursor)`, records.py:76–77) and checked against the seed at replay time (replay.py:362 — **seeds are profile-bound**). Consequences: a parent-study-specific profile name would give the same strategy different record IDs per study (hence the canonical `ifvg_search_profile_<hash16>` naming rule), one baseline seed cannot warm-start a changed-section child (hence the single-baseline verification vertical slice), and two profiles never share native record IDs (hence the profile-independent opportunity-lineage layer for cross-profile deltas).

### 11.2 FSM (`reducer.py`, 2 798 LOC)

Single `_setup` slot → one-active-setup/one-active-trade is structural (`active_setup_count = int(self._setup is not None)`; asserted ≤ 1). Phases S0 (idle) → S1 (HTF tapped, parent search) → S2 (parent locked) → S3 (opposing armed) → S4 (inverted, watching entries) → S5 (in trade). While occupied, counterfactual streams still emit (`slot_occupied` taps, `already_in_trade` candidate blocks). Session gating uses `doc_sessions`/`session_doc` (the `session_scheme` is stamp-only for gating). No RNG anywhere; every ordering has an explicit total key. Trading-day roll (18:00 ET) is **not** a trade exit.

Hardcoded (not reachable by config; SC change + pin bump required to search them): entry fill = confirmation-bar close; stop-before-target same-bar resolution (`resolver_policy` field exists but is never branched on); stop anchor = running swing extreme ± buffer; opposing/entry gaps are 1m-only; `ifvg_retest` family unconditionally blocked (`retest_trigger_unratified`); inversion = body close through the far boundary only; HTF tap conflict = hard drop; parent/HTF ranking fixed (higher TF, then newer); no intraday flatten for IFVG.

### 11.3 Replay (`replay.py`)

`run_day(bars_by_tf, *, section, seed, trading_day, tick_size, levels_for, dataset_exhausted, context_*, audit_capture_mode) → IfvgDayResult | IfvgContextDayResult`. **Multi-day sequentiality is the caller's loop** carrying `end_seed`; `DayOrchestrator` raises if the seed's `profile_hash` differs from the section's. The **cache-trust theorem** (`SC/tests/test_ifvg_replay_parity.py`): chained per-day `run_day` is emission-identical to one continuous drive — this licenses per-day chunking/resume. The audit channel (`audit_capture_mode="fsm_audit_v1"`) is opt-in, **outside the profile hash**, and byte-identical for core emissions on/off.

### 11.4 Reader (`data/databento_parquet.py`)

`DatabentoParquetSource.for_trading_day` composes the `[prev 18:00 ET, 18:00 ET)` window across UTC-midnight partitions, DST-aware, era-boundary-aware (mbp10 → mbp1 schema eras). Vectorized decode (×12.5–13 vs the old path; 2.87–3.36 µs/event, byte-identical proof in `SC/docs/archive/windows/W3A_READER_PROOF.log`). Trades-from-MBP-1 per D-P-17 (`action ∈ {T, TRADE}` rows on `is_tob` schemas). The reader is ~98 % of a day's replay cost. **Note (verified):** the current reader intentionally decodes and discards `ts_recv` (databento_parquet.py:42) — the raw parquet retains it, so the planned MBP-1 feature materializer can decode the full `(ts_event, ts_recv, sequence, source_ordinal)` ordering key without any SC change.

### 11.5 QL↔SC driver stack and cost anatomy

```
run_day (SC)
 └─ QL capture_driver.capture_single_date[_with_context]
     └─ QL dataset.build_ifvg_v2_capture / build_ifvg_fsm_audit_v1 / build_ifvg_v3_capture
         └─ QL preparation.run_resumable_preparation_job (persisted pairs)
```

Cache layers: per-day bar/level artifacts `ifvg_tbars_/ifvg_levels_<artifacts_tag>.parquet` (profile-independent except TF set/tick/scheme — `IfvgCaptureConfig.artifacts_tag()`); per-day capture/seed caches are **legacy-v1 only** (v2 replays always re-drive the reducer); `capture_tag()` includes the profile hash.

Measured costs (from repo logs/reports): v2 replay on **cached** artifacts ≈ **15 s/day** (138-day chain < ~35 min including audit + parity + immutable save); day-artifact rebuild (new TF set) ≈ 40 s/day, strictly sequential (seed-chained); audit channel adds ≈ +2 % when disabled-path, more when enabled on micro-fixtures. **Correct parallel axis for a config search: one process per child config; days sequential within a child.**

### 11.6 The proven child-config plug point

`QL/src/alpha_lab/agents/data_infra/ifvg/profiles.py::resolve_profile_config({"profile_name": <runnable base>, "section_overrides": {...}})` → re-validated section, `qualification_mode` forced to `custom_profile`, escalation of `runnable/execution_enabled/non_runnable_reason` forbidden, evaluator knobs frozen (bootstrap 10 000 / CI 0.95). Working end-to-end template: `QL/scripts/run_ifvg_tf_variant.py` (overrides → `dataclasses.replace(IfvgCaptureConfig(), section=…)` → `build_ifvg_v2_capture` → `run_ifvg_v2_evaluation` → timed JSON report). `QL/scripts/prepare_ifvg_tf_variant_retest480_pair.py` already flips `parent_retest_timeout_1m_bars = 480` (D-6, ratified 2026-08-11).

---

## 12. Conventions the new lane must follow

- Boundary specs: frozen pydantic v2, `extra="forbid"` (D-001). Internal value objects: `@dataclass(frozen=True, slots=True)`. Enums: `StrEnum`. Registries: `MappingProxyType`.
- Hashing: `manifest.canonical_sha256` / `canonical_contract_sha256` (same algorithm) — do not invent a third.
- Versioning: module-level schema-version ints + string policy ids ending `_v1`/`_v2`.
- Decisions: `docs/DECISIONS.md`, one `## D-0NN:` section each; **latest is D-038; this plan reserves D-039…D-045**.
- Docs-in-same-change rule: `ARCHITECTURE.md` + `docs/README.md` + `docs/pipeline_state.yaml` must be updated with any semantics change.
- Reports: JSON machine artifact + Markdown human companion, matched base names.
- Tests: synthetic verified-pair fixture pattern (`tests/agents/ifvg_v3_fixtures.py::context_fixture`); AppTest + source-introspection contract tests for UI.

---

## 13. Gap analysis summary (current state → brief requirements)

| Brief requirement | Current state | Verdict |
|---|---|---|
| Immutable identities, overwrite refusal, reuse (§9.23, §16.9) | Fully solved by `manifest.py` protocol | **Reuse** |
| Concurrency-safe mutable catalog | `update_context_run_catalog` has **no locking** (verified) — concurrent publishers can lose writes | **New** (lock-guarded append-only event log + rebuildable index) |
| Content-addressed replay inputs | Per-file source hashing exists (`data_access.hash_allowlisted_source_files` → `permitted_source_hashes` in existing manifest identities) and day artifacts carry stamps/manifests — but the per-day bar/level caches are **mutable files on disk**, and `artifacts_tag()`/date hashes identify only the *requested* configuration | **New** (`ReplayInputBundle` composes the existing hashes into the core replay identity — no duplicate hashing; Amendment P0-A) |
| Generated-profile gating | `PROFILE_CAPABILITY_REGISTRY` is keyed by the five fixed profile names — a generated `ifvg_search_profile_<hash16>` can never be (and must never need to be) a key in it | **New** (`GeneratedProfileCapability`: fixed registry gates baselines; generated children gated by their own contract; Amendment P0-D) |
| Cross-profile population matching | Native record IDs embed the profile hash — impossible across profiles | **New** (profile-independent lineage layer, match-basis contract) |
| Intratrade path chronology for prop rules | Executed-trade rows carry realized + MFE/MAE magnitudes only; 1m day artifacts exist for path reconstruction | **New** (trade-path fidelity contracts; unordered 1m OHLC evidence plus registered assumed intrabar scenarios in v1) |
| Child FSM profiles + full sequential replay (§9.4–9.5) | Plug point + single-child template exist; no enumerator/orchestrator | **Extend** |
| Search charter / axis registry (§9.2–9.3) | Nothing typed; scratch ledger only | **New** |
| Strategy gates (§9.7) | `compute_trade_stats` core exists; TUW/stability/concentration missing | **Extend** |
| Prop contracts + funded/payout/fees/replacement (§9.9–9.12) | Evaluation-phase engine only | **Extend (major)** |
| Risk policies (§9.11) | None | **New** |
| Portfolio one-common-path (§9.13) | None | **New** |
| Bootstrap (§9.14) | Day-block MC exists (n=10k, seed 42, Wilson) | **Reuse** |
| Stress scenarios (§9.14) | None | **New** |
| Pareto frontier / robustness / knife-edge (§9.17–9.19) | None | **New** |
| Failure attribution + deterministic insights (§9.21–9.22) | None | **New** |
| ≤5-day verification policy (§9.6A) | Blocked by fixed-allowlist gate | **New policy class** |
| Full-pipeline contract + operator UI (§9.6B, §10.12A) | Nothing | **New** |
| Study-cell/dimension/delta contracts (§7A) | `IfvgContextRunReconciliation` is the seed pattern | **New (generalize)** |
| Feature blocks/bundles + MBP-1 contract (§7A.11–12) | Cumulative tier enum only | **New (graph, tiers frozen as bundles)** |
| Supervised ladder (§7B.2) | Prevalence + CatBoost exist; logistic rung missing | **Extend** |
| Regime lane (§7B.4–7B.15) | Nothing (sklearn available) | **New** |
| Wizard / monitor / results / pipeline UI (§10) | Strong primitives; no wizard/monitor/frontier views | **New tabs on reused primitives** |


---

## 12. Final V4 target-boundary clarifications

The implementation plan now distinguishes the following target contracts explicitly:

- one-trade `TradePathArtifactEnvelope` versus whole-stream `TradePathBundleEnvelope`;
- account-level versus portfolio-level simulation identities;
- firm-rule path capabilities versus simulation scenario policy;
- stable feature-block registry definitions versus content-derived block resolutions;
- real verification authorization versus generic owner authorization;
- role-free numerical regime fits versus separately ratified promotion decisions.

These are target-plan contracts, not claims about existing repository symbols. Current code remains as mapped above until the corresponding release implements and verifies each seam.
