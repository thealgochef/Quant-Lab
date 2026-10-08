# Research storage and run index

Updated October 8, 2026.

`Claude-Quant-Lab` is the code repository. Its sibling
`Claude-Quant-Lab-Research-Artifacts` is the local research evidence library,
not a second application checkout. Keeping large stores and exact historical
sources there prevents research jobs from filling the source tree. Both folders
are intentional; this index gives them one navigable organization.

## Where files belong

| Location | Contents | Git policy |
|---|---|---|
| `src/`, `scripts/`, `tests/` | Application code, maintained tools, reusable small fixtures | Versioned |
| `docs/`, `research/core/` | Contracts, decisions, architecture, exact Core pin metadata | Versioned; no delivery ZIPs or captured screenshots |
| `../Claude-Quant-Lab-Research-Artifacts/<task-id>/` | Research inputs, working stores, source snapshots, logs, validation and archived evidence | Outside this repository |
| `reports/` | Final local review deliveries still used by saved-result/download catalogs | Ignored; never commit |
| `models/` | Local fitted model bundles and their metadata | Ignored; existing files retained |
| `data/` | Existing market inputs, application stores, saved catalogs and user state | Ignored; do not recursively inspect or clean |
| `ledger/` | Existing local immutable research events | Ignored; retain |

Existing immutable stores contain absolute paths and content hashes. Do not
rename their folders to make them prettier, rewrite old identities, or resume
them with current source. Human-readable labels belong in this index. Copies of
source under an external task may be required to reproduce its approved run.

## Recent completed research

Paths in the following table are relative to the sibling artifact library.
Short IDs are navigation labels; exact identities remain in verified envelopes.

| Research label | Folder | Status and retrieval |
|---|---|---|
| MenthorQ level context — Task A1 | `ifsm-task-a1-20261003/` | Original context implementation and verification; relocated checkout handoff is `deliveries/repository-handoff/` |
| MenthorQ level-rule study — Task B | `ifsm-task-b-20261003/` | Completed study and prepared-store bindings; `study_store/`, `deliveries/repository-handoff/`; combined original delivery is `inputs/task-a1-task-b-handoff.zip` |
| Six-configuration full-range comparison | `ifsm-correct-config-full-range-v01/` | Corrected result `7278632babf01b08…`; `repair_v02/result_validation.json` records completion; original interrupted attempt retained |
| MFFU 64-intent context batch — original | `ifsm-mffu-context-batch-20261007/` | Completed original result `7a8cb062d3fc31bc…`; `study_store/`; original source and result remain unchanged |
| MFFU lifecycle repair — current baseline | `ifsm-mffu-repair-integration-v01/` | Completed corrected result `b8db8427cb226061…`; `study_store/`, exact `core/` and `runtime-v03/` |
| MFFU reporting follow-up | `ifsm-b8db-reporting-followup-v01/` | Reporting-v6 corrections for the saved baseline; no economic replay |
| MFFU ML phase 01 | `ifsm-mffu-ml-phase-v01/` | Completed report `5afb61d81dcf…`; `report-v01.json` and `final-delivery-status.json`; 24 cells, 216 folds, 26 funded operations |

ML `run-v01`/`execution-v01` contain shadow derivations; v02 is a preserved
intermediate fit; v03 owns the final approved benchmark and policy operations.
Keep every version because the final report links its exact dependencies. An
old `prepared-v03.json` saying approval was required is a historical receipt;
`freeze-v03.json` and final verified results record the later execution.

The active gamma input remains the original archive at
`ifsm-mffu-context-batch-20261007/handoff/docs/ifsm-mffu-context-batch-v01/inputs/MenthorQ_Research_Data_v02.zip`.
Its location and hashes were not changed by cleanup.

The normal dashboard continues to discover saved results through its existing
catalogs under `data/ifvg_datasets/search/v1/catalog/`. Final active downloads
remain under `reports/funded_comparison/`, `reports/funded_payout/`,
`reports/ifsm_mffu_context_batch/`, `reports/ifsm_mffu_repair_integration/`,
`reports/ifsm_b8db_reporting_followup/`, and `reports/ifsm_ml_phase_v01/`.

## Historical work and source archives

The artifact library's top-level `README.md` labels every existing task folder.
September UI work is grouped by its existing `ifvg-dashboard-*`,
`ifvg-redesign-*`, `ifvg-analytical-corrections-*`, `ifvg-setup-timing-*` and
`ifvg-theme-fix-*` names. `ifsm-research-core/`, `research-core-sources/` and
`strategy-core-scale-out-exit/` retain exact source checkouts. `cleanup-*`,
`merge-*` and `premerge*` are historical maintenance evidence, not new studies.

## October 8 relocations

All moves preserve file bytes. The complete old-path/new-path/size/SHA-256
record is `../Claude-Quant-Lab-Research-Artifacts/cleanup-20261008/relocations.json`.
No raw market data, live result store, model bundle or prior approval was deleted.

| Former checkout location | Current external task location |
|---|---|
| `handoff/task_a1/`, `handoff/task_b/` | Respective Task A1/B `deliveries/repository-handoff/` |
| Root `handoff.zip` | Task B `inputs/task-a1-task-b-handoff.zip` |
| `docs/IFSM_*v01.zip` | `ifsm-correct-config-full-range-v01/inputs/` |
| `docs/ifvg-dashboard-redesign.zip` | `ifvg-dashboard-redesign-20260924/inputs/` |
| Binary files under `docs/ifvg-dashboard-redesign/` | `ifvg-dashboard-redesign-20260924/repository-evidence/`, same inner path |
| Binary files under `docs/ifvg-dashboard-repairs/` | `ifvg-dashboard-repairs-20260923/repository-evidence/`, same inner path |
| Binary files under `docs/ifvg-redesign-fixes/` | `ifvg-redesign-fixes-20260924/repository-evidence/`, same inner path |
| Input/review ZIPs under `docs/ifsm-mffu-ml-phase-v01/` | `ifsm-mffu-ml-phase-v01/repository-evidence/`, same inner path |
| Root `ifvg_search_runs/`, `ifvg_search_trades/`, scratch search script and generated search Markdown | `legacy-ifvg-search/repository-outputs/`, same filename |
| Old `reports/IFVG_*` and `reports/ifvg_dashboard_repairs/` | `archived-reports/repository-deliveries-20261008/`, same filename |
| Obsolete `dashboard-ui/node_modules/` | `retired-dashboard-prototype/dependencies/node_modules/` |
| Root `catboost_info/` | `legacy-ml-training/catboost_info/` |

Historical evidence documents and package manifests retain their original
filenames/references. Resolve archived relative screenshot paths with the table
above, or the complete relocation record. Small narrative contracts stay in Git.
The ML handoff validator resolves relocated binary inputs while checking their
original hashes. Cleanup does not rewrite Git history or shrink existing Git
objects; it removes generated artifacts from the current versioned tree.

## Starting new work without recreating clutter

Run these commands from the repository root:

```powershell
python scripts/install_repo_hooks.py
python scripts/research_workspace.py ifsm-example-study-20261008
python scripts/check_tracked_artifacts.py
```

The scaffolder creates `inputs/`, `study_store/`, `work/`, `source/`,
`validation/`, and a task README outside the repository. It refuses to overwrite
an existing task. Use a descriptive lowercase hyphenated task ID, including a
date or version. Record the exact plan/result IDs and current status in its README.

The MFFU, ML-phase and full-range worker CLIs reject working stores, state,
work and staging paths within the checkout before execution. Existing read-only
status access remains available. Resolved ancestor Git markers also protect the
original checkout when a newly frozen worker runs from an external source copy.
Research authorization gates still apply; historical frozen scripts are unchanged.

The installed pre-commit hook checks staged Git blobs, including force-added
files. CI checks the full tracked tree. They reject local stores, reports,
archives, models, copied source trees, generated screenshots and files over
5 MiB; small reusable fixtures and the explicit historical Core bundle are
deliberate exceptions. New clones must install the hook. These controls prevent
the known routes into Git; a manually bypassed hook or an unrelated script can
still create local files. Never use `git clean -X` to maintain this project.
