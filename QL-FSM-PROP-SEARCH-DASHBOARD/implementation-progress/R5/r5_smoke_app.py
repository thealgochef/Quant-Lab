"""Live browser-smoke harness for the R5 Full Pipeline Run surface.

Shell-only launcher (never part of the repo's importable surface): builds
one COMPLETED synthetic 16-stage pipeline run under a CONTENT-ADDRESSED
scratch directory, points the pipeline tab's roots at it, and serves the
real `render_pipeline_run` so a browser can walk Configure / Preview /
Monitor / Resume-Retry / Publish over genuine persisted artifacts.

R5-FIX (gate finding 4): the old harness reused a bare ``prepared.ok``
marker forever — stale evidence could survive code changes. The scratch
root is now keyed by sha256 over (git commit, a source-tree digest of the
exercised surfaces, the fixture's ``pipeline_semantic_id``); any change to
the commit, the sources, or the fixture semantics lands in a FRESH keyed
directory, other keys are cleared on startup, and reuse additionally
requires the persisted evidence to VALIDATE (every planned stage terminal
in the state file + the pipeline result envelope reloads through the
verifying store). ``smoke_manifest.json`` in the keyed directory records
the key inputs and result artifact ids for the evidence bundle.

Run from the repo root:

    python -m streamlit run \
        QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R5/r5_smoke_app.py \
        --server.port 8598 --server.headless true

Everything is written under %TEMP%/ifvg_r5_smoke/<key>; the repo's data/
tree is untouched.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
for entry in (str(_REPO / "src"), str(_REPO / "scripts"), str(_REPO)):
    if entry not in sys.path:
        sys.path.insert(0, str(entry))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="R5 pipeline smoke", layout="wide")

_SCRATCH_BASE = Path(tempfile.gettempdir()) / "ifvg_r5_smoke"

#: the surfaces this smoke actually exercises — any byte change here forces
#: a fresh scratch key (finding 4: the marker is tied to the sources)
_DIGEST_GLOBS = (
    ("src/alpha_lab/agents/data_infra/ifvg", "**/*.py"),
    ("src/alpha_lab/propsim", "**/*.py"),
    ("scripts", "ifvg_*.py"),
    ("tests/agents/ifvg_search", "*.py"),
)


def _git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _source_tree_digest() -> str:
    entries: list[tuple[str, str]] = []
    for base, pattern in _DIGEST_GLOBS:
        root = _REPO / base
        for path in sorted(root.glob(pattern)):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries.append((path.relative_to(_REPO).as_posix(), digest))
    entries.append(
        ("r5_smoke_app.py", hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    )
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prepared_evidence_validates(fixture: dict, scratch: Path) -> dict | None:
    """The persisted run evidence, iff it VALIDATES; else None (finding 4:
    reuse is evidence-tied — never a bare marker file)."""

    from alpha_lab.agents.data_infra.ifvg.search.pipeline import (
        PipelineResultEnvelope,
        read_pipeline_state,
    )
    from alpha_lab.agents.data_infra.ifvg.search.store import load_verified_envelope

    semantic = fixture["semantic"]
    state = read_pipeline_state(
        fixture["state_root"], semantic.pipeline_semantic_id
    )
    if state is None:
        return None
    terminal = {"completed", "reused", "blocked"}
    stages = state.get("stages") or {}
    planned = [
        name for name, entry in stages.items() if (entry or {}).get("in_plan")
    ]
    if not planned or any(
        (stages[name] or {}).get("status") not in terminal for name in planned
    ):
        return None
    publication = state.get("publication") or {}
    result_id = publication.get("pipeline_result_id")
    if not result_id:
        return None
    try:
        load_verified_envelope(
            fixture["store_root"], "search_results", result_id, PipelineResultEnvelope
        )
    except Exception:
        return None
    return {"state": state, "pipeline_result_id": result_id}


@st.cache_resource
def _prepared_run() -> dict:
    from alpha_lab.agents.data_infra.ifvg.search.charter import save_charter
    from alpha_lab.agents.data_infra.ifvg.search.pipeline import run_pipeline
    from alpha_lab.agents.data_infra.ifvg.search.store import (
        save_or_reuse_envelope,
    )
    from tests.agents.ifvg_search.pipeline_fixture import build_pipeline_fixture

    commit = _git_commit()
    tree_digest = _source_tree_digest()
    # the semantic id depends only on the fixture construction — build it
    # once against a probe root to derive the content key, then bind the
    # real fixture to the keyed scratch directory
    probe = build_pipeline_fixture(_SCRATCH_BASE / "_probe")
    pipeline_semantic_id = probe["semantic"].pipeline_semantic_id
    key_material = {
        "commit": commit,
        "source_tree_digest": tree_digest,
        "pipeline_semantic_id": pipeline_semantic_id,
    }
    key = hashlib.sha256(
        json.dumps(key_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    _SCRATCH_BASE.mkdir(parents=True, exist_ok=True)
    # clear every OTHER key's scratch (stale evidence never lingers)
    for stale in _SCRATCH_BASE.iterdir():
        if stale.name != key:
            shutil.rmtree(stale, ignore_errors=True)

    scratch = _SCRATCH_BASE / key
    scratch.mkdir(parents=True, exist_ok=True)
    fixture = build_pipeline_fixture(scratch)
    assert fixture["semantic"].pipeline_semantic_id == pipeline_semantic_id

    evidence = _prepared_evidence_validates(fixture, scratch)
    reused = evidence is not None
    if evidence is None:
        save_charter(fixture["store_root"], fixture["charter"])
        save_or_reuse_envelope(
            fixture["store_root"], "pipeline_specs", fixture["semantic"]
        )
        run_pipeline(
            fixture["semantic"],
            fixture["charter"],
            store_root=fixture["store_root"],
            state_root=fixture["state_root"],
            wiring=fixture["wiring"],
            worker_policy=fixture["worker_policy"],
        )
        evidence = _prepared_evidence_validates(fixture, scratch)
        if evidence is None:
            raise RuntimeError(
                "the freshly built pipeline run does not validate — refusing "
                "to serve unverified smoke evidence"
            )

    manifest = {
        **key_material,
        "scratch_key": key,
        "scratch_root": str(scratch),
        "reused_validated_evidence": reused,
        "pipeline_result_id": evidence["pipeline_result_id"],
        "prepared_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "streamlit_version": st.__version__,
    }
    (scratch / "smoke_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**fixture, "manifest": manifest}


fixture = _prepared_run()

import ifvg_pipeline_tab as pipeline_tab  # noqa: E402

pipeline_tab.PIPELINE_STATE_ROOT = fixture["state_root"]
roots = {
    "store_root": fixture["store_root"],
    "state_root": fixture["state_root"],
    "draft_root": Path(fixture["manifest"]["scratch_root"]) / "drafts",
    "namespace": "verification",
}
roots["draft_root"].mkdir(parents=True, exist_ok=True)

manifest = fixture["manifest"]
st.title("R5 smoke — Full Pipeline Run over a completed synthetic run")
st.caption(
    f"commit `{manifest['commit'][:12]}` · tree "
    f"`{manifest['source_tree_digest'][:12]}` · pipeline "
    f"`{manifest['pipeline_semantic_id'][:12]}` · scratch key "
    f"`{manifest['scratch_key']}` · prepared {manifest['prepared_at_utc']}"
)
pipeline_tab.render_pipeline_run(st, roots=roots)
