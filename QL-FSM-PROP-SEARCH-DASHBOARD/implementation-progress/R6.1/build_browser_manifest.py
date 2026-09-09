"""Write the R6.1 browser-smoke MANIFEST.json (schema v2) from the captured evidence.

Run from the repo root AFTER the screenshots are in ``browser-smoke/`` and the
smoke server log has been copied there:

    python QL-FSM-PROP-SEARCH-DASHBOARD/implementation-progress/R6.1/build_browser_manifest.py \
        <scratch_root>/smoke_manifest.json browser-smoke/_screenshots.json

``_screenshots.json`` maps screenshot file name -> description. Everything
else (sha256 / bytes of every bound file, the committed-tree digest over the
harness digest globs at the bound commit, the worktree-dirty-in-scope check,
the server-log counts, the bound evidence ids) is computed here from the
repository and the harness manifest; ``verify_browser_manifest.py`` then
recomputes it all independently.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FOLDER = _HERE / "browser-smoke"
_MARKER = "[r61-smoke]"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    harness_manifest = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    descriptions = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    verifier = _load("verify_browser_manifest")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=True, text=True
    ).stdout.strip()
    commit = harness_manifest["commit"]
    if commit != head:
        raise SystemExit(f"harness commit {commit[:12]} is not HEAD {head[:12]}")
    # the harness digest globs, in the validator's [prefix, pattern] form
    smoke_src = (_HERE / "r61_smoke_app.py").read_text(encoding="utf-8")
    start = smoke_src.index("_DIGEST_GLOBS = (")
    end = smoke_src.index(")\n\n", start)
    globs = [list(item) for item in eval(smoke_src[start + len("_DIGEST_GLOBS = ") : end + 1])]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *[g[0] for g in globs]],
        capture_output=True, check=True, text=True,
    ).stdout
    dirty_in_scope = [
        line for line in dirty.splitlines()
        if any(verifier._matches(line[3:].strip()[len(g[0].rstrip("/")) + 1:], g[1])
               for g in globs if line[3:].strip().startswith(g[0].rstrip("/") + "/"))
    ]
    files: dict[str, dict[str, object]] = {}
    for path in sorted(_FOLDER.iterdir()):
        if path.name in {"MANIFEST.json", "_screenshots.json"} or not path.is_file():
            continue
        data = path.read_bytes()
        files[path.name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    screenshots = {name: descriptions[name] for name in sorted(descriptions)}
    for name in screenshots:
        if name not in files:
            raise SystemExit(f"screenshot {name} is not in the folder")
    log_text = (_FOLDER / "_smoke_server.log").read_text(encoding="utf-8", errors="replace")
    markers = [line for line in log_text.splitlines() if _MARKER in line]
    runs = harness_manifest["runs"]
    bound: dict[str, object] = {
        "pipeline_semantic_id": runs["candidate"]["pipeline_semantic_id"],
        "candidate": {k: v for k, v in runs["candidate"].items() if isinstance(v, str) and len(v) == 64},
        "panel": {k: v for k, v in runs["panel"].items() if isinstance(v, str) and len(v) == 64},
        "supervised": {k: v for k, v in runs["supervised"].items() if isinstance(v, str) and len(v) == 64},
        "supervised_frozen_authority": dict(runs["supervised"].get("frozen_authority", {})),
    }
    manifest = {
        "manifest_schema_version": 2,
        "release": "R6.1",
        "commit": commit,
        "commit_is_head_at_capture": commit == head,
        "worktree_dirty_in_scope": bool(dirty_in_scope),
        "digest_globs": globs,
        "scoped_tree_digest": verifier.committed_tree_digest(commit, globs),
        "harness_source_tree_digest": harness_manifest["source_tree_digest"],
        "scratch_key": harness_manifest["scratch_key"],
        "scratch_root": harness_manifest["scratch_root"],
        "prepared_at_utc": harness_manifest["prepared_at_utc"],
        "streamlit_version": harness_manifest["streamlit_version"],
        "files": files,
        "screenshots": screenshots,
        "server_log": "_smoke_server.log",
        "expected_marker_count": len(markers),
        "server_log_counts": {
            "tracebacks": log_text.count("Traceback"),
            "deprecation_warnings": log_text.count("DeprecationWarning"),
        },
        "bound_evidence": bound,
        "captured_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capture_method": "Chrome (claude-in-chrome extension) against the headless Streamlit "
                          "server on :8611; viewport 1440x1100; no launch / promote / rank / "
                          "retrain control exists on the served surface",
    }
    (_FOLDER / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"MANIFEST.json written: {len(files)} files, {len(screenshots)} screenshots, "
          f"{len(markers)} marker lines, dirty_in_scope={bool(dirty_in_scope)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
