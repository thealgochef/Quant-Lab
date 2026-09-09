"""Unambiguous parent/Replay selection contract, including setup-only evidence."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReplaySelection:
    kind: Literal["candidate", "setup", "empty", "unavailable"]
    selected_id: str | None = None
    evidence_available: bool = True

    def __post_init__(self) -> None:
        if (self.kind in ("candidate", "setup")) != bool(self.selected_id):
            raise ValueError("A selected candidate or setup requires its exact identity")
