"""Launch the IFSM study UI with Quant-Lab's current verified Core pin.

The launcher selects the same commit as the ordinary dependency. It does not
install packages or rewrite historical studies. Starting this UI initializes
its separate store, never a replay.

``--research-core PATH`` is an explicit, labeled exception for version-2
funded variation plans: a research checkout whose base commit is the pinned
commit, with uncommitted research changes (for example the half-exit rule). The
receipt prints its exact commit and uncommitted-change hash. It is never chosen
implicitly and does not change the pin.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ifsm_research_runtime import CORE_COMMIT as CORE_COMMIT
from ifsm_research_runtime import CORE_SOURCE as CORE_SOURCE
from ifsm_research_runtime import ROOT, select_core

CORE = select_core()
WORKSPACE = ROOT / "data/ifsm_ui_replication"


def environment(core: Path | None = None) -> dict[str, str]:
    core = core or CORE
    if not (core / "src/strategy_core/__init__.py").is_file():
        raise ValueError(
            f"The verified IFSM study engine is missing: {core}. "
            "Run python scripts/prepare_ifsm_research_core.py first."
        )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, (ROOT / "src", ROOT / "scripts", core / "src")))
    env["IFSM_RESEARCH_CORE"] = str(core)
    env["IFSM_RESEARCH_UI"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def verify_runtime(env: dict[str, str]) -> dict:
    probe = """
import json
from pathlib import Path
import strategy_core, alpha_lab
from alpha_lab.agents.data_infra.ifvg.search.identities import strategy_core_source_identity
from alpha_lab.agents.data_infra.ifvg.ifsm_replication import catalog, recipe_axis_values
from strategy_core.strategies.ifvg_smc.section import IfvgSmcSection, default_ifvg_smc_section
import run_ifsm_research_ui as launcher
def require(condition, message):
    if not condition:
        raise ValueError(message)
require(Path(strategy_core.__file__).resolve().is_relative_to(launcher.CORE),
        'Strategy-Core import is outside the selected research checkout')
require(Path(alpha_lab.__file__).resolve().is_relative_to(launcher.ROOT / 'src'),
        'Quant-Lab import is outside the current source checkout')
import os
research = os.environ.get('IFSM_RESEARCH_CORE_MODE') == 'research_branch'
if research:
    from alpha_lab.propsim.funded.core_identity import source_identity_at
    funded = source_identity_at(launcher.CORE)
    commit, identity = funded['base_commit'], funded['patch_sha256']
    require(commit == launcher.CORE_COMMIT,
            'the research Core must be based on the pinned commit')
else:
    commit, identity = strategy_core_source_identity(repository_root=launcher.CORE)
    require((commit, identity) == (launcher.CORE_COMMIT, launcher.CORE_SOURCE),
            'Core commit or source identity differs from the current pinned runtime')
recipes = catalog()['recipes']
for recipe in recipes:
    recipe_axis_values(recipe)
section = default_ifvg_smc_section()
require(section.holding_policy == 'legacy_unrestricted_v1', 'Holding default changed')
require(section.entry_schedule_policy == 'legacy_doc_sessions_v1', 'Entry default changed')
require(section.htf_gap_invalidation_policy == 'execution_wick_full_fill_v1',
        'Gap invalidation default changed')
IfvgSmcSection.model_validate({**section.model_dump(),
 'htf_gap_invalidation_policy':'own_timeframe_close_v1'})
from alpha_lab.agents.data_infra.ifvg.search.axis_registry import resolve_axis_overrides
for preset in ('all_open_market_v1', 'daytime_chicago_0700_1555_v1',
               'morning_chicago_0700_1030_v1'):
    overrides = resolve_axis_overrides({
        'enabled_entry_sessions':'enabled_entry_sessions.' + preset,
        'holding_policy':'holding_policy.scheduled_daily_close_v1'})
    IfvgSmcSection.model_validate({**section.model_dump(), **overrides})
print(json.dumps({'strategy_core_import':strategy_core.__file__,
 'alpha_lab_import':alpha_lab.__file__, 'core_commit':commit,
 'core_source_identity':identity, 'exact_recipes_verified':len(recipes),
 'core_mode':'explicit research branch (uncommitted changes on the pinned commit)'
             if research else 'pinned',
 'daily_close_time_chicago':'15:55', 'daily_close_buffer_minutes':5,
 'entry_schedule_presets_verified':3,
 'gap_invalidation_default':'execution_wick_full_fill_v1',
 'gap_invalidation_choices':['execution_wick_full_fill_v1','own_timeframe_close_v1']}))
"""
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, env=env,
        capture_output=True, text=True, check=True, timeout=30,
    )
    return json.loads(result.stdout)


def initialize_workspace(env: dict[str, str]) -> None:
    # Record the explicit recoverable namespace requests BEFORE initialization.
    plan = {
        "schema_version": 1,
        "research": {"path": str(WORKSPACE / "search/v1"),
                     "class": "research", "instance_id": "e0c48bd1db2b4b5bb62d7ae36604c17b"},
        "test": {"path": str(WORKSPACE / "search_test/v1"),
                 "class": "test", "instance_id": "8459d1e10c1848ccbe0169045842201f"},
    }
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    path = WORKSPACE / "namespace_plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Existing replication namespace plan differs; preserve it.")
    else:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, indent=2) + "\n")
    command = """
import json
from pathlib import Path
from run_ifsm_research_ui import WORKSPACE
from alpha_lab.agents.data_infra.ifvg.search.store_namespace import initialize_store_namespace
plan=json.loads((WORKSPACE/'namespace_plan.json').read_text(encoding='utf-8'))
for name in ('research','test'):
    item=plan[name]
    initialize_store_namespace(Path(item['path']), namespace_class=item['class'],
                               store_instance_id=item['instance_id'])
"""
    subprocess.run([sys.executable, "-c", command], cwd=ROOT, env=env, check=True, timeout=30)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--core", type=Path, help="Exact prepared research checkout")
    parser.add_argument(
        "--research-core", type=Path,
        help="Explicit research checkout on the pinned commit with uncommitted changes "
             "(version-2 funded variation plans only; not a pin change)")
    parser.add_argument(
        "--check", action="store_true", help="Read-only imports/configuration check"
    )
    args = parser.parse_args(argv)
    if args.core and args.research_core:
        parser.error("choose --core or --research-core, not both")
    env = environment(select_core(args.research_core or args.core))
    if args.research_core:
        env["IFSM_RESEARCH_CORE_MODE"] = "research_branch"
    receipt = verify_runtime(env)
    print(json.dumps(receipt), flush=True)
    if args.check:
        return 0
    initialize_workspace(env)
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "scripts/ifsm_research_ui.py"),
         "--server.port", str(args.port), "--server.address", "127.0.0.1",
         "--server.headless", "true", "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
