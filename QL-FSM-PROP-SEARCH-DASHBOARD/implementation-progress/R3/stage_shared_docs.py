"""Stage the three shared docs as HEAD + R3 lane transforms ONLY.

The user's pre-existing uncommitted hunks never enter the release commit
(kickoff §1.4): each staged blob is rebuilt from the HEAD content plus exactly
the transforms the R3 working-tree edit applied, hashed with
``git hash-object -w`` and planted via ``git update-index --cacheinfo``.
Run from the repo root AFTER ``git add`` of the normal (non-shared) files and
BEFORE ``git commit``. Idempotent.
"""

from __future__ import annotations

import subprocess
import sys

ARCH_APPEND = """
R3 additions (prop lifecycle — fidelity contracts first, then the synthetic
account engine; `alpha_lab.propsim` lifecycle modules, additive beside the
untouched evaluation-only walker):

- **Trade-path fidelity** (`propsim/trade_path.py`): typed path evidence and
  bundles with payload/envelope identities; a 1m bar's
  `observed_intrabar_order` can only be `"unknown"`; assumed intrabar paths
  are registered SCENARIO policies (`bar_adverse_extreme_first_v1` /
  `bar_favorable_extreme_first_v1` — two identities and, on order-sensitive
  trades, two different results); rule support is capability-based
  (`PropRulePathRequirement`: required path capabilities + accepted fidelity
  classes, never enum ordering) and fails closed via `PathCapabilityReport`.
- **Typed calendars** (`propsim/calendar.py`): `DayCountBasis`/`DurationRule`
  under a `SimulatedClockPolicy`; a basis the active clock cannot represent
  (calendar-month recurring fee under day-block bootstrap without a synthetic
  calendar) fails closed with `UnsupportedCalendarRuleError`.
- **Firm contracts + contract evidence** (`propsim/firm_contracts.py` ·
  `contract_evidence.py`): permitted rules/thresholds/observation policies +
  the rule→capability matrix; adverse/favorable ordering never inside
  `PhaseRules` (scenario policies only); source documents → per-field
  evidence → compilation → owner review → supersession on a one-way status
  ladder where synthetic evidence can NEVER reach `first_party_verified`.
- **Account walk** (`propsim/account.py`): full lifecycle (evaluation →
  funded → payouts/fees/replacement) as ONE strictly ordered
  `PropAccountEventEnvelope` stream (`prop_account_event_order_v1`, global
  `event_ordinal`, exact source trade/decision/candidate/setup/path links);
  the evaluation-only walker (`engine.py`) stays untouched, compatibility
  proven by the one-contract `AccountWalk`≡`EvaluationWalk` parity fixture.
- **Risk sizing + withdrawal behavior** (`propsim/risk.py` ·
  `withdrawal.py`): every sizing family with typed skip reasons (never
  silently forced to one contract); trader withdrawal choices are a separate
  identity from the firm contract and both enter every simulation identity.
- **Adapters + stream hash** (`propsim/adapters.py`): v2 executed-trade →
  account-trade tick→point/cost mapping plus the stable ORDER-SENSITIVE
  gross stream hash pinning the exact resolved trade sequence.
- **Portfolio / stress / simulation identities** (`propsim/portfolio.py` ·
  `stress.py` · `simulation.py`): copied accounts replay ONE common
  correlated market path per draw (no per-account resampling path exists);
  nine seeded deterministic stress scenarios ride the simulation identity;
  account/portfolio simulation identities carry every result-changing policy
  (constructor-surface audit: no result-changing constructor-only kwarg);
  duplicate bootstrap index sequences are legal with a unique
  `path_instance_id` + stored `sampled_index_sequence_hash` per draw.
- **Prop metrics + search wiring** (`propsim/prop_metrics.py` ·
  `search/gates.py` · `search/orchestrator.py`): lower-tail-first
  `PayoutReliabilityVector`; `evaluate_prop_gates` fail-closed rows (None
  thresholds are explicit not-required passes; missing observations fail);
  the orchestrator `prop_simulator` seam applies the conservative ALL-legs
  feasibility rule, merges each prop objective into the frontier as the
  WORST value across the child's simulations, defers prop-owned pareto
  objectives past the strategy stage only when a simulator is wired, and
  records explicit skip notes otherwise.
"""

README_OLD = """`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-042, D-043 (D-041,
D-044, D-045 reserved for R3/R5-R6/R4). R1 and R2 (multi-child search,
lineage, exact deltas, verifier integration, `scripts/ifvg_search_job.py`)
implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""
README_NEW = """`tests/agents/ifvg_search/`. Decisions D-039, D-040, D-041, D-042, D-043
(D-044, D-045 reserved for R5-R6/R4). R1, R2 (multi-child search, lineage,
exact deltas, verifier integration, `scripts/ifvg_search_job.py`), and R3
(prop lifecycle: fidelity-first trade paths, typed calendars, field-level
contract evidence, the full account walk, portfolio/stress/simulation
identities, and prop-gate/worst-firm frontier wiring — `alpha_lab.propsim`
lifecycle modules + `tests/propsim/`) implementations are complete;
**R1 acceptance is blocked pending the owner-approved verification fixture**"""

YAML_OLD = (
    '    R3_prop_lifecycle: {implementation_status: "not_started", '
    'acceptance_status: "transitively_blocked_by_R1"}'
)
YAML_NEW = (
    '    R3_prop_lifecycle: {implementation_status: "complete", '
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
        print(f"staged {path} = HEAD + R3 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
