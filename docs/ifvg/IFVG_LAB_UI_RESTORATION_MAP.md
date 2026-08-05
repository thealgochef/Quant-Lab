# IFVG Lab UI Restoration Map

Status: implementation map  
Reference shell: Quant-Lab commit `bd6825924d02f76c108e81ef1d4edb27b032d55a`

## Router

`render_ifvg_lab_tab()` restores a three-tab IFVG Lab. The old component structure and visual
primitives may be reused, but all loaders, selectors, experiment calls, persistence, comparison,
and replay linkage use the new immutable contracts. Experiments is the default tab.

## Tabs

### Experiments

- Select a verified paired artifact and profile by full identity.
- Show capability/preparation status and blocking reason before controls.
- Configure observation cohort, counterfactual label, and registered feature tier.
- Launch the deterministic context experiment engine; no adaptive search.
- Render candidate research separately from actual execution outcomes.
- Show immutable history and compatibility-aware comparison.
- On incompatible runs, show a configuration diff without metric deltas.
- Show legacy results in a separate read-only section with mandatory caveats.

### Replay / Verifier

- Select an exact candidate ID from a verified artifact.
- Resolve exact geometry/cursor, decision, and trade links; no setup/time fallback.
- Render only pre-protected-boundary candles.
- Display geometry, FSM transitions, context captures, displacement windows, pools, sweeps,
  decisions, and trades while keeping counterfactual and actual outcomes distinct.
- Surface link/provenance errors in sanitized form without local paths.

### Data & Audit

- Preserve the current v2 audit panels.
- Add v3 schema/identity, pairing, parity, access, capacity, performance, validity, and preparation
  reports.
- Show protected-buffer and sealed access counters explicitly; every value must be event-derived.
- Preparation is an external persisted process; Streamlit does not own replay jobs.

## Reuse and rewrite

| Area | Treatment |
|---|---|
| chart primitives | reuse |
| calibration and coverage visuals | reuse with new result contract |
| feature insight presentation | reuse, descriptive only |
| history/config diff layout | reuse with immutable IDs |
| current audit panels | reuse and extend |
| selectors/loaders | rewrite for catalog and verified full IDs |
| run calls/persistence | rewrite for context engine and immutable store |
| comparison | rewrite for strict compatibility |
| replay | rewrite for exact links |
| mutable deletion | remove |
| sealed-data controls | remove |
| broad directory listing | remove |
| legacy recapture/session task | remove |
| setup/time fallback matching | remove |
| `doc_defaults.apply` | remove |

## UI state and cache

Use a new session-state namespace. Cache keys include complete artifact/run IDs and manifest
hashes. Short IDs are display-only. Cache entries never survive an identity mismatch.

All user-facing exceptions pass through path/secret sanitization. Empty, blocked, failed,
preparing, superseded, and no-valid-fold states are intentional rendered states, not tracebacks.

## Acceptance map

| Behavior | Verification |
|---|---|
| Experiments is first/default | AppTest tab order and heading assertion |
| capability reasons visible | runnable/blocked fixture tests |
| exact candidate replay | cursor/link adversarial fixture |
| reports remain separated | candidate/execution outcome sentinel test |
| audit contains v2 and v3 | AppTest section assertions |
| no sealed/protected controls | text/widget absence assertions |
| immutable run UX | duplicate-refusal and no-delete assertions |
| legacy is read-only | widget/action absence and caveat assertion |
| accessibility/responsiveness | keyboard, viewport, empty/failure, visual checks |

## Implemented UI status

`render_ifvg_lab_tab()` now renders the three tabs in the locked order with Experiments first.
The UI validates full lowercase SHA-256 artifact IDs, exposes capability and preparation reasons,
supports R1.0/R1.5/R2.0 plus fixed SL/TP label controls, shows cohort-specific M3 limitations,
and separates candidate research from source-v2 execution reporting. Compatible comparison
requires the complete reconciliation identity and uses paired trading-day deltas; incompatible
runs show only a configuration diff.

Replay selection is exact by candidate ID and renders geometry evidence, FSM lifecycle,
candidate/decision/trade links, context state, displacement, pools, and sweep evidence without
setup/time fallback. Data & Audit retains v2 panels and adds verified v3 identity/schema/pairing,
access, capacity, performance, validity, and preparation reports. Mutable delete, recapture,
sealed-data, and promotion controls are absent.

`tests/agents/test_ifvg_lab_tab.py` passes all 15 tests, including AppTest tab order, default
surface, exact-ID behavior, and forbidden-control inventory. A local headless Streamlit server
also returned HTTP 200. Interactive viewport, keyboard, and visual inspection remains open
because the in-app browser was unavailable during final verification; this open high-severity
gate blocks UI activation alongside the critical performance gate.
