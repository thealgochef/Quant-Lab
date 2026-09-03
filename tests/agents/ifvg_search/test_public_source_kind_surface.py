"""HARDENING-BACKEND-FIX §5 (HB-FIX-03) — the focused public-surface scan.

The public source kind is exactly ``mbp1`` / ``trades`` /
``legacy_verified_replay_source``. The historical physical file stem may
appear ONLY in the private physical-file resolver of ``trading_calendar.py``
(and in historical test fixtures); it is prohibited from every serialized
public contract, registry example, study cell, feature bundle, live
contract and model feature name of the IFVG lane, the prop simulator and
the IFVG scripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

import pytest

from alpha_lab.agents.data_infra.ifvg.features.mbp1_source_contract import (
    DEEP_BOOK_EXEMPT_LITERALS,
    DEEP_BOOK_IDENTIFIER_REGEX,
)
from alpha_lab.agents.data_infra.ifvg.search.identities import (
    ReplaySourcePartitionRef,
    registered_identity_pairs,
)
from alpha_lab.agents.data_infra.ifvg.search.trading_calendar import (
    LEGACY_VERIFIED_REPLAY_SOURCE,
    PUBLIC_SOURCE_KINDS,
    SourceKind,
    SourcePartitionRef,
)

REPO = Path(__file__).resolve().parents[3]
#: a quoted literal naming the historical physical stem (any separator style)
_PHYSICAL_STEM_LITERAL = re.compile(r"""["']mbp[-_ ]?10["']""", re.IGNORECASE)
#: the public contract surface that is scanned
_SCANNED_TREES = (
    "src/alpha_lab/agents/data_infra/ifvg",
    "src/alpha_lab/propsim",
)
_SCANNED_SCRIPT_PREFIX = "ifvg_"
#: the ONLY permitted location: the private physical-file resolver
_RESOLVER = "src/alpha_lab/agents/data_infra/ifvg/search/trading_calendar.py"


def _scanned_files():
    for tree in _SCANNED_TREES:
        for path in sorted((REPO / tree).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path
    for path in sorted((REPO / "scripts").glob(f"{_SCANNED_SCRIPT_PREFIX}*.py")):
        yield path


def test_public_source_kind_is_exactly_the_registered_triple() -> None:
    assert set(get_args(SourceKind)) == set(PUBLIC_SOURCE_KINDS)
    assert set(PUBLIC_SOURCE_KINDS) == {"mbp1", "trades", LEGACY_VERIFIED_REPLAY_SOURCE}
    replay_kind = ReplaySourcePartitionRef.model_fields["source_kind"].annotation
    assert set(get_args(replay_kind)) == set(PUBLIC_SOURCE_KINDS)
    for contract in (SourcePartitionRef, ReplaySourcePartitionRef):
        with pytest.raises(ValueError):
            contract.model_validate(
                {
                    **{
                        name: ("a" * 64 if "sha256" in name else "x")
                        for name in contract.model_fields
                    },
                    "source_kind": "mbp10",
                }
            )


def test_public_surface_permits_the_physical_stem_only_in_the_private_resolver() -> None:
    offenders: list[str] = []
    resolver_hits: list[str] = []
    for path in _scanned_files():
        relative = path.relative_to(REPO).as_posix()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not _PHYSICAL_STEM_LITERAL.search(line):
                continue
            if relative == _RESOLVER:
                resolver_hits.append(line.strip())
            else:
                offenders.append(f"{relative}:{number}: {line.strip()}")
    assert offenders == [], "the physical stem leaked into the public surface"
    # inside the resolver the literal is confined to the stem table, where it
    # maps to the public legacy kind
    assert resolver_hits, "the private resolver must name the physical stem exactly once"
    assert all("LEGACY_VERIFIED_REPLAY_SOURCE" in hit for hit in resolver_hits), resolver_hits
    assert len(resolver_hits) == 1


def test_registered_contract_examples_serialize_no_physical_stem() -> None:
    scanned = 0
    for pair in registered_identity_pairs():
        if pair.example_factory is None:
            continue
        example = pair.example_factory()
        document = json.dumps(example.model_dump(mode="json"), sort_keys=True, default=str)
        assert not re.search(r"mbp[-_ ]?10", document, re.IGNORECASE), pair.name
        scanned += 1
    assert scanned > 0


def test_legacy_literal_is_the_only_deep_book_exemption() -> None:
    assert DEEP_BOOK_EXEMPT_LITERALS == (LEGACY_VERIFIED_REPLAY_SOURCE,)
    assert DEEP_BOOK_IDENTIFIER_REGEX.search(LEGACY_VERIFIED_REPLAY_SOURCE) is None
    for kind in PUBLIC_SOURCE_KINDS:
        assert DEEP_BOOK_IDENTIFIER_REGEX.search(kind) is None, kind
