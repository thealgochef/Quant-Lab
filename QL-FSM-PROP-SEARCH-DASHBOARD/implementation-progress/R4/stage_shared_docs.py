"""Stage the three shared docs as HEAD + R4 lane transforms ONLY.

The user's pre-existing uncommitted hunks never enter the release commit
(kickoff §1.4): each staged blob is rebuilt from the HEAD content plus exactly
the transforms the R4 working-tree edit applied, hashed with
``git hash-object -w`` and planted via ``git update-index --cacheinfo``.
Run from the repo root AFTER ``git add`` of the normal (non-shared) files and
BEFORE ``git commit``. Idempotent. (Same mechanism as ``../R3/stage_shared_docs.py``.)
"""

from __future__ import annotations

import subprocess
import sys

ARCH_APPEND = """
R4 additions (trader workspace UI — the guided study surface over R1–R3;
`FRONTEND_UX_CONTRACT.md` is the normative authority):

- **Workspace routing** (`scripts/ifvg_study_tab.py` + the Experiments
  delegation in `ifvg_lab_tab.py`): a session-state-backed horizontal radio
  (`New Study | Active Runs | Results | History | Context Research`,
  namespace `ifvg_study_v1_*`); only the selected route executes; the
  M0–M3 Context Research panel delegates verbatim; a namespace selector
  switches between the research (`search/v1`) and verification
  (`search_test/v1`) stores with truthful badging.
- **Presentation contracts** (`ifvg/study_status.py` ·
  `ifvg/study_presentation.py`): CS §13 status/scope/empty-state registries
  with the exact required copy (`Development Exploratory Representative`,
  the verification badge, the no-pass sentence, `Not run — strategy gate
  failed`); pure wizard validators; funnel/stage derivations pinned by
  contract test to the orchestrator's exact explanation sentinels;
  baseline-diff display names; work estimates as operational annotations.
- **Drafts** (`ifvg/study_drafts.py`, `data/ifvg_study_drafts/`): the one
  deliberately mutable authoring surface — atomic JSON, autosave on Next,
  exact-step restore, deep-copy Clone as New Search; `mark_frozen` is the
  single permitted final write and every later mutation/discard refuses.
- **UI providers** (`ifvg/study_providers.py`): exact-ID manifest-verified
  loads only; run enumeration from the mutable job root + catalog event
  log (immutable store roots are never listed; the orchestrator records
  the frontier envelope id as a state-file phase note as the locator);
  `prepare_cross_profile_deltas` is the ONLY cross-profile delta
  constructor and persists BOTH lineage-uniqueness reports first.
- **Wizard** (`scripts/ifvg_study_wizard.py`): five modes × eight steps;
  registered axis cards grouped by market meaning with NO widget for
  locked/blocked/measured axes and no raw override editor anywhere;
  computation-path chips; truthful synthetic contract cards; per-firm
  account/risk/withdrawal/replacement policy sets under one universal
  strategy profile; three ordered benchmark gate groups
  (`proposed_protocol_default` stamps); read-only verification allowlist +
  the exact typed full-scope confirmation; freeze → `validate_charter` →
  immutable charter save → detached launch ONLY in the button handler via
  the registry-gated job shim.
- **Monitor** (`scripts/ifvg_active_runs_tab.py`): `st.fragment(5s)` over a
  plain AppTest-callable body + manual Refresh; phase checklist; five
  keyboard funnel buttons (the Plotly funnel accompanies, never replaces);
  the exact child table with skipped-stage reasons; sanitized detail;
  confirmed safe-cancel sentinel; CLI escape hatch on missing status.
- **Results/History/Compare** (`scripts/ifvg_results_tab.py` ·
  `ifvg_results_compare.py` · `ifvg_results_charts.py`): overview cards +
  exact no-pass copy; frontier with an always-present selectbox twin;
  heatmap glyph classes (◼ ▲ ✕ · ⊘) + table twin; firm matrix / survival
  (step+dash) / payout distributions (P10-first); indexed explorer with
  identity-first columns; dimension-diff ribbon with `match_basis` and
  four panels (membership claims disabled for non-comparable pairs, never
  fuzzed); seven deterministic insight categories with exact
  `EvidenceRef` actions; the account timeline in `event_ordinal` order
  with FUX marker shapes and exact-id drill-down; all figures budgeted
  with honest `OmissionReport`s.
- **Execution gating** (`ifvg/search/runner_registry.py` + the shim): the
  job shim's `--runner-entry` is registry-gated — the UI passes registered
  KEYS only, raw strings are refused before any import, and the only
  registered entry before the R5 executors is the synthetic fixture
  wiring, so a real charter's launch renders capability-blocked; `resume`
  re-enters the idempotent worker. Declared two-axis interaction
  contrasts now evaluate on balanced grids (`study/contrasts.py`,
  difference-of-differences, seed-7 bootstrap; refusals retained).
"""

README_OLD = """`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-041, D-042, D-043
(D-044, D-045 reserved for R5-R6/R4). R1, R2 (multi-child search, lineage,
exact deltas, verifier integration, `scripts/ifvg_search_job.py`), and R3
(prop lifecycle: fidelity-first trade paths, typed calendars, field-level
contract evidence, the full account walk, portfolio/stress/simulation
identities, and prop-gate/worst-firm frontier wiring — `alpha_lab.propsim`
lifecycle modules + `tests/propsim/`) implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""
README_NEW = """`tests/agents/ifvg_search/`. Decisions D-039 through D-043 + D-045's
trader-workspace half (D-044 and D-045's pipeline half reserved for
R5–R6). R1, R2 (multi-child search, lineage, exact deltas, verifier
integration, `scripts/ifvg_search_job.py`), R3 (prop lifecycle:
fidelity-first trade paths, typed calendars, field-level contract
evidence, the full account walk, portfolio/stress/simulation identities,
and prop-gate/worst-firm frontier wiring — `alpha_lab.propsim` lifecycle
modules + `tests/propsim/`), and R4 (trader workspace UI: the Experiments
sub-navigation, eight-step five-mode wizard with disk drafts, registry-
gated launch, Active Runs monitor, Results/History/comparison/insights/
account-timeline surfaces — `scripts/ifvg_study_*.py`,
`ifvg_active_runs_tab.py`, `ifvg_results_*.py`, `ifvg_ui_common.py` +
`ifvg/study_{status,presentation,drafts,providers}.py`) implementations
are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""

YAML_OLD = (
    '    R4_trader_ui: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R4_trader_ui: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *args], input=data, capture_output=True, check=True
    )
    return result.stdout


def main() -> int:
    for path, transforms in TRANSFORMS.items():
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = head
        for old, new in transforms:
            if old is None:
                staged = staged.rstrip("\n") + "\n" + new
            else:
                if old not in staged:
                    raise SystemExit(f"{path}: transform anchor missing at HEAD")
                staged = staged.replace(old, new)
        blob = _git(
            "hash-object", "-w", "--stdin", "--path", path,
            data=staged.encode("utf-8"),
        ).decode("ascii").strip()
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + R4 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
