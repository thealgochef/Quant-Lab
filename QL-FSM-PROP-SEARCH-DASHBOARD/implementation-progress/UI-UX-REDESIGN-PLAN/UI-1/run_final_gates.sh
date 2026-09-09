#!/usr/bin/env bash
# UI-1 final gate: the COMPLETE suite as ONE invocation (warnings as errors per
# pyproject), detached from the tool harness (see the release convention).
cd /c/Users/gonza/Documents/Claude-Quant-Lab || exit 1
EV="QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/UI-UX-REDESIGN-PLAN/UI-1"
export PY_COLORS=0
echo "full_suite_started $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$EV/gates.progress"
{
  echo "# UI-1 full suite — start $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# command: python -m pytest -q -p no:cacheprovider --junitxml=$EV/junit_full.xml"
  echo "# POLYGON_API_KEY present: $([ -n "$POLYGON_API_KEY" ] && echo yes || echo no); DATABENTO_API_KEY present: $([ -n "$DATABENTO_API_KEY" ] && echo yes || echo no)"
  python -m pytest -q -p no:cacheprovider --junitxml="$EV/junit_full.xml" 2>&1 | grep -av 'missing ScriptRunContext'
  echo "exit=${PIPESTATUS[0]}"
  echo "# finished $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$EV/_final_pytest.txt" 2>&1
echo "full_suite_finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$EV/gates.progress"
touch "$EV/gates.done"
