"""Enable the repository's artifact guard for this checkout without replacing hooks."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def install(root: Path) -> None:
    configured = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"], cwd=root, capture_output=True, text=True,
    )
    if configured.returncode not in (0, 1):
        configured.check_returncode()
    if configured.stdout.strip() not in ("", ".githooks"):
        raise SystemExit(
            "Existing core.hooksPath retained. Add `python scripts/check_tracked_artifacts.py "
            "--staged` to your current pre-commit hook."
        )
    if not configured.stdout.strip():
        hook_path = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=root, capture_output=True, text=True, check=True,
        ).stdout.strip()
        existing = Path(hook_path)
        if not existing.is_absolute():
            existing = root / existing
        installed = (
            [path for path in existing.iterdir() if path.is_file() and path.suffix != ".sample"]
            if existing.is_dir() else []
        )
        if installed:
            raise SystemExit(
                "Existing Git hooks retained. Add `python scripts/check_tracked_artifacts.py "
                "--staged` to the existing pre-commit hook, preserving your other hooks."
            )
    hook = root / ".githooks" / "pre-commit"
    if not hook.is_file():
        raise SystemExit(f"Missing versioned hook: {hook}")
    hook.chmod(hook.stat().st_mode | 0o111)
    subprocess.run(
        ["git", "config", "--local", "core.hooksPath", ".githooks"], cwd=root, check=True,
    )
    print("Repository artifact guard enabled. CI also checks the entire tracked tree.")


if __name__ == "__main__":
    install(ROOT)
