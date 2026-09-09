"""Stage the three shared docs as HEAD + UI-3 lane transforms ONLY.

Same mechanism as ``../UI-2/stage_shared_docs.py`` (user hunks never enter the
release commit; ``--apply-worktree`` post-commit replays the same transforms so
the surviving worktree diff is the user's pre-existing hunks only). HEAD for
UI-3 is the UI-2 release head (``eb85d09``).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCH_APPEND = """
UI-3 additions (Phase 3 of the UI/UX redesign over the UI-2 release head; UI-4 … UI-6 follow;
acceptance still transitively blocked by R1):

- **Presentation package** (`ifvg/presentation/`): `metric_registry` — one `MetricSpec` per
  displayed technical key (human name, definition, persisted source, unit, directionality from
  `OBJECTIVE_DIRECTIONS` where registered, and the reference the value is read against — the
  selected resolved gate, the prevalence-reference Brier, the 0 skill boundary, the 0.5 chance
  line as a direction only, the calibration targets as a distance only, the stamped
  sample-adequacy minimums, the persisted report limits, the measured access counters versus
  the policy-enforced `protected_*` zeros, and intervals that cross zero) with
  `evaluate_metric` / `evaluate_interval` / `evaluate_gate_flag` — missing or unevaluated
  evidence is UNAVAILABLE, never PASS; `rollups` — the deterministic FAIL → BLOCKED →
  INCONCLUSIVE → WARNING → PASS → INFORMATIONAL section roll-ups (one sentence, main reason,
  inspect-next); `help_registry` — a `HelpEntry` per control id rendered by `help_text`, the
  glossary of the plan's sixteen terms and the explicit live `HELP_EXEMPTIONS`; `labels` — the
  human-label registry (profiles, bundles, blocks, objectives, model protocols, regime
  algorithms, stamps, statuses, roles, classes, tiers, verdicts) and the distinct availability
  chips; `context_research` and `results_presentation` — the pure reading assemblies of the
  Context Research and Results screens.
- **Shared primitives** (`scripts/ifvg_ui_common.py`): `detail_levels` (Summary / Research
  details / Technical identity & audit over the unchanged persisted vocabulary; `disclosure_level`
  delegates), `identity_reveal`, `status_chip_line`, `metric_card`, `rollup_card`,
  `glossary_expander`.
- **Context Research** (`scripts/ifvg_lab_tab.py`): the decision summary first (Data integrity,
  Probability skill, Calibration, Stability), the sample-adequacy card, registry metric cards
  with their references, the named reliability diagonal, coverage and net R on separate axes
  (`ifvg_lab_charts.build_coverage_figure` over the adapter's threshold rows;
  `build_reliability_figure`), fold validity chips, intervals, the top-N importance with fold
  stability, run-compatibility reasons in words, raw JSON only under Technical identity & audit;
  the adapters expose the reference / calibration / fold / access fields the frozen statistics
  already persist — `context_reporting.py` and the M0–M3 computation are untouched.
- **Results** (`scripts/ifvg_results_tab.py`): the Selected configuration block (Strategy
  quality and Prop feasibility roll-ups; registry metric cards against the charter's resolved
  gates; the prop vector as the worst firm), registry captions on the metric pickers, the
  explorer column guide.
- **Pipeline surface**: the ladder frame keeps AUC numeric with a separate `AUC reason` column;
  the MBP-1 (`ifvg_mbp1_panels.py`) and regime (`ifvg_regime_panels.py`) panels are summary-first
  — readiness resolved from the selected run, then Research details, then Advanced diagnostics
  holding the manual exact-id inputs, registries and stamps.
- **Help everywhere**: every widget of every UI script carries registry help or is a registered
  navigation exemption; the source scan (`tests/agents/test_ifvg_help_scans.py`) enforces it.
- **Unchanged**: Strategy-Core, the fixed M0–M3 lane, every immutable artifact identity, the
  exact-ID loading rule, S11, MBP-1's `research_only_offline` boundary, the backend contracts.
"""

README_OLD = """receipt seams — `ifvg/presentation/{flows,review_vocabulary}.py`,
`scripts/ifvg_verification_center.py`)
implementations are complete;"""
README_NEW = """receipt seams — `ifvg/presentation/{flows,review_vocabulary}.py`,
`scripts/ifvg_verification_center.py`),
and UI-3 (Phase 3: the metric metadata registry, the deterministic
section roll-ups, the helper-text registry with the glossary and explicit
exemptions, the human-label registry, the three detail levels, the
decision-summary-first Context Research and Results presentation and the
summary-first MBP-1 / regime panels —
`ifvg/presentation/{metric_registry,rollups,help_registry,labels,context_research,results_presentation}.py`)
implementations are complete;"""

YAML_OLD = (
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)
YAML_NEW = (
    '    UI_3_metric_help_system_research_results_panels: {implementation_status: "complete", '
    'acceptance_status: "transitively_blocked_by_R1", '
    'ui_phase: "3 of 6 (UI-UX-REDESIGN-PLAN revision 2)", '
    'presentation_package: "ifvg/presentation/{metric_registry,rollups,help_registry,labels,'
    'context_research,results_presentation}.py", '
    'surfaces: "ifvg_ui_common detail levels / metric and roll-up cards / glossary; Context '
    "Research decision summary first; Results selected-configuration roll-ups and column guide; "
    'ladder AUC reason column; summary-first MBP-1 and regime panels", '
    'help_coverage: "every widget of every UI script (registry help or a registered navigation '
    'exemption; scanned)", '
    'findings_closed: "F-07(full) F-09 F-11(part)", '
    'browser_acceptance: "open (UI-6)", '
    'owner_actions_performed: "none (no signing, no seed production, no real verification run)"}\n'
    '    V1_hardening: {implementation_status: "superseded_by_HARDENING_BACKEND", '
    'acceptance_status: "transitively_blocked_by_R1"}\n'
)

TRANSFORMS: dict[str, list[tuple[str | None, str]]] = {
    "ARCHITECTURE.md": [(None, ARCH_APPEND)],
    "docs/README.md": [(README_OLD, README_NEW)],
    "docs/pipeline_state.yaml": [(YAML_OLD, YAML_NEW)],
}


def _git(*args: str, data: bytes | None = None) -> bytes:
    result = subprocess.run(["git", *args], input=data, capture_output=True, check=True)
    return result.stdout


def _transformed(head: str, transforms: list[tuple[str | None, str]]) -> str:
    staged = head
    for old, new in transforms:
        if old is None:
            staged = staged.rstrip("\n") + "\n" + new
        else:
            if old not in staged:
                raise SystemExit(f"transform anchor missing at HEAD: {old[:60]!r}")
            staged = staged.replace(old, new)
    return staged


def main() -> int:
    apply_worktree = "--apply-worktree" in sys.argv[1:]
    dry_run = "--dry-run" in sys.argv[1:]
    for path, transforms in TRANSFORMS.items():
        if apply_worktree:
            worktree = Path(path).read_text(encoding="utf-8")
            updated = worktree
            for old, new in transforms:
                if old is None:
                    if new not in updated:
                        updated = updated.rstrip("\n") + "\n" + new
                elif old in updated:
                    updated = updated.replace(old, new)
                elif new not in updated:
                    raise SystemExit(f"worktree anchor missing: {path} {old[:60]!r}")
            Path(path).write_text(updated, encoding="utf-8", newline="\n")
            print(f"worktree {path} updated with the UI-3 lane transforms")
            continue
        head = _git("show", f"HEAD:{path}").decode("utf-8")
        staged = _transformed(head, transforms)
        if dry_run:
            print(f"dry-run {path}: {len(head)} -> {len(staged)} chars")
            continue
        blob = (
            _git(
                "hash-object",
                "-w",
                "--stdin",
                "--path",
                path,
                data=staged.encode("utf-8"),
            )
            .decode("ascii")
            .strip()
        )
        _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}")
        print(f"staged {path} = HEAD + UI-3 lane transforms ({blob[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
