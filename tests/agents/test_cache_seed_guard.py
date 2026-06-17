"""Focused guard test: a per-day utility cache must record the PDH/PDL seed it
was built with, and an existing cache may be trusted ONLY when its stamped seed
matches the seed entering that day. This is the regression guard for the
2026-02-12 stale-cache bug (a standalone timing build wrote a prev_full_hl=None
cache — missing PDH/PDL touches — that the seeded build then silently skipped).
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _SEED_META_KEY,
    _cache_seed_matches,
    _write_day_cache,
)

_FRAME = pd.DataFrame({"date": ["2026-02-12"], "representative_price": [25058.25]})
_SEED = (25465.25, 25058.25)


def test_seeded_cache_matches_same_seed_only(tmp_path):
    path = tmp_path / "ml_utility_x.parquet"
    _write_day_cache(_FRAME, path, _SEED)
    # stamp is present and round-trips
    md = pq.read_metadata(path).metadata
    assert _SEED_META_KEY in md
    assert _cache_seed_matches(path, _SEED) is True
    # a different seed or no-seed must NOT be trusted
    assert _cache_seed_matches(path, (25465.25, 25000.0)) is False
    assert _cache_seed_matches(path, None) is False
    # content survives the stamped write
    assert pd.read_parquet(path)["representative_price"].iloc[0] == 25058.25


def test_none_seed_cache_trusted_only_when_no_prior_day(tmp_path):
    path = tmp_path / "ml_utility_none.parquet"
    _write_day_cache(_FRAME, path, None)
    # first window day (no prior) → expected seed None → trusted
    assert _cache_seed_matches(path, None) is True
    # but a day that HAS a prior window day (seed expected) → NOT trusted → rebuild
    assert _cache_seed_matches(path, _SEED) is False


def test_legacy_unstamped_cache_not_trusted_when_seed_expected(tmp_path):
    # A cache written the old way (plain to_parquet, no seed stamp) — the exact
    # shape of the stale 2026-02-12 file.
    path = tmp_path / "ml_utility_legacy.parquet"
    _FRAME.to_parquet(path, index=False)
    assert _SEED_META_KEY not in (pq.read_metadata(path).metadata or {})
    # unstamped is trusted only when no seed is expected
    assert _cache_seed_matches(path, None) is True
    assert _cache_seed_matches(path, _SEED) is False


def test_missing_or_corrupt_file_not_trusted(tmp_path):
    missing = tmp_path / "nope.parquet"
    assert _cache_seed_matches(missing, _SEED) is False
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"not a parquet file")
    assert _cache_seed_matches(corrupt, _SEED) is False
