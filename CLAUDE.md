# Alpha Signal Research Lab — Agent Handoff & Context Guide

## What This Project Is

A **multi-agent quantitative trading research system** for discovering and validating profitable trading signals in **NQ/ES futures markets**, designed for prop firm environments (Apex Trader Funding, Topstep).

The system has **6 agents** communicating through typed JSON message envelopes with a **validation firewall** between Signal Engineering and Statistical Validation.

## Current State: SCAFFOLD COMPLETE, NO BUSINESS LOGIC YET

All 92 files are scaffolded. The project **installs, lints clean, and passes 77 tests**. But every business-logic function currently raises `NotImplementedError`. The tests validate the architecture (contracts serialize, bus routes messages, detectors auto-register, agents instantiate, config loads).

### Verification Commands
```bash
# Use Python 3.13 (pip installs to 3.13, not 3.11)
"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/ -v
ruff check src/ tests/
```

## Architecture Quick Reference

### 6 Agents
| ID | Module Path | Role |
|---|---|---|
| ORCH-001 | `agents/orchestrator/` | Research Director — sequences phases, resolves conflicts |
| DATA-001 | `agents/data_infra/` | Market Data Engineer — tick aggregation, bar construction, sessions |
| SIG-001 | `agents/signal_eng/` | Alpha Researcher — 20 signal detectors across 3 tiers |
| VAL-001 | `agents/validation/` | Quant Reviewer — blind statistical testing, DEPLOY/REFINE/REJECT |
| EXEC-001 | `agents/execution/` | Risk Manager — cost modeling, prop firm constraints, Monte Carlo |
| MON-001 | `agents/monitoring/` | Production Ops — live metrics, alerts, regime detection |

### Critical Design Rules
1. **Validation Firewall**: SIG-001 and VAL-001 NEVER share implementation details. VAL-001 receives opaque signal vectors (direction + strength arrays only). Feedback is limited to metric name + value + threshold. See `agents/validation/firewall.py`.
2. **All contracts in one file**: `core/contracts.py` is the single source of truth for all Pydantic models. Never split these across agent directories.
3. **Signal detectors auto-register**: Any `SignalDetector` subclass with `detector_id` in its `__dict__` auto-registers via `__init_subclass__`. Import `agents/signal_eng/detectors` to populate the registry.
4. **Message bus dedup**: Messages are deduplicated by `request_id + message_type + sender`. The bus logs everything for orchestrator visibility.
5. **Conflict resolution hierarchy**: VAL-001 > SIG-001 (stats override intuition), EXEC-001 > VAL-001 (unprofitable = worthless).

### 20 Signal Detectors (all stubs)
- **Tier 1 (Core)**: ema_confluence, kama_regime, vwap_deviation
- **Tier 2 (ICT)**: liquidity_sweeps, fair_value_gaps, ifvg, market_structure, killzone_timing, pd_levels_poi, tick_microstructure
- **Tier 3 (Composite)**: multi_tf_confluence, ema_vwap_interaction, displacement, order_blocks, volume_profile, scalp_entry, sweep_fvg_combo, ema_reclaim, session_gap, adaptive_regime

### Key Files to Know
| File | What It Does |
|---|---|
| `core/enums.py` | All enumerations (AgentID, Timeframe, SignalTier, Verdict, AlertLevel, etc.) |
| `core/contracts.py` | All 13 Pydantic interface models (DataBundle, SignalVector, SignalBundle, ValidationReport, etc.) |
| `core/message.py` | MessageEnvelope + MessageBus (dedup, audit log, routing) |
| `core/agent_base.py` | BaseAgent ABC (state machine, send/ack/nack/escalate) |
| `core/config.py` | YAML config loader → typed Settings object |
| `agents/signal_eng/detector_base.py` | SignalDetector ABC + SignalDetectorRegistry (auto-registration) |
| `agents/orchestrator/pipeline.py` | Pipeline state machine (INIT → PHASE_1_2 → ... → DEPLOYED) |
| `agents/validation/firewall.py` | ValidationTest ABC + signal stripping + verdict assembly |
| `config/instruments.yaml` | NQ/ES contract specs (tick size, commissions, slippage) |
| `config/prop_firms.yaml` | Apex/Topstep constraint profiles (DD limits, consistency rules) |
| `config/validation_thresholds.yaml` | IC, hit rate, Sharpe, decay thresholds for signal approval |

## Implementation Roadmap — What to Build Next

The architecture spec defines a 9-phase pipeline. We are at **Phase 0 (scaffold done)** and need to start **Phase 1-2 (Data + Signals)**.

### Phase 1: DATA-001 Implementation
**Goal**: Clean, session-tagged OHLCV bars at all 11 timeframes.

Files to implement (in order):
1. **`agents/data_infra/providers/base.py`** — Already has the ABC. Need to create a concrete provider for the chosen data vendor.
2. **`agents/data_infra/aggregation.py`** — `aggregate_tick_bars()` and `aggregate_time_bars()`. These construct 987/2000-tick bars and 1m→1D time bars from raw ticks.
3. **`agents/data_infra/sessions.py`** — `tag_sessions()` and `classify_killzone()`. Tags every bar with session_id (e.g., 'NQ_2026-02-21_RTH'), session_type (RTH/GLOBEX), and killzone (LONDON/NEW_YORK/ASIA/NONE).
4. **`agents/data_infra/quality.py`** — `run_quality_checks()`. Validates: no gaps >2min during RTH, volume >0, OHLC constraints, timestamp monotonicity, cross-TF consistency.
5. **`agents/data_infra/agent.py`** — Wire `build_data_bundle()` to use provider + aggregation + sessions + quality, produce a complete DataBundle.

**Data source**: User selected "Data vendor API" but hasn't specified which vendor yet. Ask before implementing the provider.

### Phase 2: SIG-001 Implementation (Tier 1 First)
**Goal**: Implement the 3 Tier 1 (Core) detectors first since all Tier 3 composites depend on them.

1. **`detectors/tier1/ema_confluence.py`** — Compute 13/48/200 EMAs, detect alignment, crossover velocity, spread. Normalize to [-1, +1] direction + [0, 1] strength.
2. **`detectors/tier1/kama_regime.py`** — Kaufman's Adaptive MA: slope, price-KAMA divergence, adaptive smoothing.
3. **`detectors/tier1/vwap_deviation.py`** — Session-anchored VWAP, std dev bands, VWAP slope.
4. **`agents/signal_eng/bundle_builder.py`** — Wire `build_signal_bundle()` to iterate registered detectors, run each, collect SignalVectors.

Then Tier 2 (ICT Structural), then Tier 3 (Composites which combine Tier 1+2).

### Phase 3-4: VAL-001 Implementation
After signals exist, implement the 6 validation tests:
1. `validation/tests/ic_testing.py` — Spearman IC, rolling IC, t-stat
2. `validation/tests/hit_rate.py` — Directional accuracy, long/short split
3. `validation/tests/risk_adjusted.py` — Sharpe, Sortino, max DD, profit factor
4. `validation/tests/decay_analysis.py` — Exponential decay fit, half-life
5. `validation/tests/orthogonality.py` — Factor correlations, incremental R²
6. `validation/tests/robustness.py` — Subsample, instrument, regime stability

Then wire `firewall.py` to run all tests and produce SignalVerdicts.

### Phase 5+: EXEC-001 and MON-001
Implement after validated signals exist. Cost modeling, prop firm checks, Monte Carlo, then live monitoring.

## Testing Approach

Tests exist for all core infrastructure. As you implement each function:
- The existing tests validate contracts, bus routing, agent lifecycle, config loading, and detector registration
- Add new tests for the actual business logic as you implement each function
- Test naming convention: `test_<module>.py` in the corresponding `tests/` subdirectory
- Run with: `"/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe" -m pytest tests/ -v`

## Python Environment Note

Two Python versions exist on this machine:
- **Python 3.11** at `/c/Users/gonza/AppData/Local/Programs/Python/Python311/python`
- **Python 3.13** at `/c/Users/gonza/AppData/Local/Programs/Python/Python313/python.exe`

`pip` installs to **3.13**. All dependencies (pydantic, pandas, numpy, scipy, pytest, ruff) are on 3.13. Always use the 3.13 path for running tests.

## Architecture Spec Location

The full architecture spec (6 agents, 12 handoff protocols, 15 message types, validation firewall rules, error handling matrix, bootstrap sequence) was provided in conversation. Key details are captured in the agent docstrings and this file. If you need the exact spec text, check the system prompts in each agent's `.py` file — they reference the relevant spec sections.
