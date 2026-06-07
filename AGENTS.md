# Alpha Signal Research Lab - Agent Notes

Updated: 2026-06-04.

## Start points

- This repo is a Python ML training workbench for NQ/ES futures plus local dashboard surfaces.
- Current canonical architecture doc: `ARCHITECTURE.md`.
- Docs index / stale-doc classification: `docs/README.md`.
- Streamlit ML tab workflow: `docs/ML_TRAINING_WORKBENCH.md`.
- Session experiment CLI: `scripts/run_dashboard_session_experiment.py`.
- Strategy-Core v3 cross-repo matrix: `../Strategy-Core/V3_COMPATIBILITY_MATRIX.md`.
- For model-training changes, start with `scripts/ml_training_tab.py` and `src/alpha_lab/agents/data_infra/ml/`, not the older generic multi-agent scaffold.

## Commands

Use the active local environment when present:

```bash
cd /root/trading-algos/Quant-Lab
. .venv/bin/activate
python -m pytest tests/agents/test_strategy_contract_nodrift.py tests/agents/test_strategy_contract_repoint.py -q
python -m ruff check src tests scripts
streamlit run scripts/dashboard.py
```

Project metadata requires Python `>=3.14`; prefer the pinned `.python-version` (`3.14.5`) via `uv`. The previous Windows Python 3.13 path may exist on AlgoChef's workstation, but do not hardcode it in new docs or scripts.

React dashboard commands run from `dashboard-ui/`:

```bash
npm install      # if node_modules is missing
npm run dev
npm run build
npm test
```

FastAPI dashboard backend:

```bash
cd /root/trading-algos/Quant-Lab
. .venv/bin/activate
PYTHONPATH=src python -m alpha_lab.dashboard.api
```

Python 3.14 base installs are Databento-first. The legacy Rithmic adapter is not
installed in the base dependency set because current `async-rithmic` pins protobuf
`<5`, which conflicts with the Python 3.14-compatible protobuf stack used by
Streamlit.

## Data flow

- Databento batch import: `python scripts/process_batch_download.py <zip>`.
- Imported tick files are expected at `data/databento/{symbol}/{YYYY-MM-DD}/mbp10.parquet`, `mbp1.parquet`, or `trades.parquet`.
- The Streamlit ML tab reads local parquet only; it does not call the Databento API.
- Per-date ML caches are mode-specific:
  - Extrema: `ml_features_{config_hash}.parquet`
  - Dashboard utility: `ml_utility_{config_hash}.parquet`
- The dashboard-utility hash includes Strategy-Core engine/source semantics, so v1/v2/v3 structural changes do not reuse stale caches.

## Current Strategy-Core v3 contract

Dashboard-utility mode is the production-aligned research path and is single-sourced to Strategy-Core v3:

- `engine_version`: `strategy_core_engine_v3`
- Bars/features: trade-price/trade-print, NQ 0.25 grid, default `147t` touch bars.
- Sessions: ET `asia` 19:00→02:45, `london` 03:00→08:00, `ny` 09:00→17:00; 18:00 ET trading-day boundary.
- Levels: PDH/PDL from the full prior trading day; Asia/London levels from current-day sessions.
- Touch guard: level `available_from` is enforced; pre-availability self-touches do not consume zones.
- Label entry: realistic decision-time price at `touch_close + 5m`; not level price at touch.
- Cutoffs: flatten/no-new-decision at 16:40 ET; forward cutoff 17:00 ET.
- Gate: `tradeable_reversal`, `eligible_session="ny"`, confidence `0.70`.
- Research session experiments: train/evaluate all sessions by default, production-gate NY by default; scope is persisted into `evaluation.json`, `metadata.json`, and `strategy.json` as audit metadata.

Trade-Lab is not v3-compatible yet; do not claim runtime readiness until Trade-Lab is repointed and end-to-end parity is proven.

## Evaluation rules

- Quality gates must use concatenated out-of-sample walk-forward fold predictions, not predictions from the final refit model.
- RFECV runs once before the walk-forward loop; the selected features must be reused by every fold model and the final saved model.
- Label purging removes training rows whose forward labeling window crosses into the test period.
- The saved runtime model is refit on all labeled rows after evaluation.
- Session experiment filters are applied after broad dataset generation: fold training, OOS metrics, and final refit use the configured training/evaluation scope.
- Preserve CatBoost native missing-value handling; do not blanket `fillna(0.0)` unless intentionally matching a separate runtime approximation path.

## Runtime/output artifacts

A Streamlit save writes:

```text
models/{model_name}/model.cbm
models/{model_name}/metadata.json
models/{model_name}/evaluation.json
models/{model_name}/strategy.json
models/{model_name}/oos_predictions.parquet  # when OOS rows are available
```

Treat `models/`, `catboost_info/`, `*.cbm`, cached `*.parquet`/`*.csv`, local imported data, and scratch chart HTML as generated local outputs unless a task explicitly says otherwise.

The old `data/models/dashboard_3feature_v1.cbm` exporter is retained compatibility tooling, not automatically a v3 bundle. Canonical bundle location/checksum verification is deferred until the incoming local zip is available.

## Documentation maintenance

When code behavior changes, update docs in the same change. At minimum:

- Strategy/session/feature/label semantics -> `ARCHITECTURE.md`, `docs/ML_TRAINING_WORKBENCH.md`, `docs/pipeline_state.yaml`, and Strategy-Core docs/matrix.
- UI/workflow changes -> `docs/ML_TRAINING_WORKBENCH.md`.
- Model-bundle/cache output changes -> `ARCHITECTURE.md`, `docs/README.md`, and this file.
