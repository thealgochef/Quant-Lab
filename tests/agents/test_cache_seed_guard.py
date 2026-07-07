"""Focused guard test: a per-day utility cache must record the PDH/PDL seed it
was built with, and an existing cache may be trusted ONLY when its stamped seed
matches the seed entering that day. This is the regression guard for the
2026-02-12 stale-cache bug (a standalone timing build wrote a prev_full_hl=None
cache — missing PDH/PDL touches — that the seeded build then silently skipped).
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from alpha_lab.agents.data_infra.ml import dashboard_utility_builder as dub
from alpha_lab.agents.data_infra.ml.config import MLPipelineConfig
from alpha_lab.agents.data_infra.ml.dashboard_utility_builder import (
    _SEED_META_KEY,
    _cache_seed_matches,
    _write_day_cache,
    build_utility_dataset,
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


def test_builder_stamps_entering_seed_and_second_run_cache_hits(tmp_path, monkeypatch):
    """SEED P3 (SEED_PARITY_RECON §3(d)): the builder must stamp each day-D cache with
    the seed ENTERING D — the value the :161 trust check compares against on the next
    run and the warmer's convention. Pre-fix the builder stamped the post-update carry
    (day D's OWN H/L), so every builder-written cache self-invalidated on re-run."""
    config = MLPipelineConfig()
    symbol = config.instrument
    cache_tag = config.dataset_config_hash()
    dates = ["2026-01-05", "2026-01-06"]
    own_hl = {"2026-01-05": (100.0, 90.0), "2026-01-06": (110.0, 95.0)}
    build_calls: list[str] = []

    def fake_process(date_str, data_dir, sym, util_cfg, prev_full_hl):
        build_calls.append(date_str)
        return pd.DataFrame({"date": [date_str], "representative_price": [100.0]})

    def fake_hl(data_dir, sym, date_str, util_cfg, prev_full_hl):
        return own_hl[date_str]

    monkeypatch.setattr(dub, "_process_single_date", fake_process)
    monkeypatch.setattr(dub, "_get_session_hl_for_date", fake_hl)

    first = build_utility_dataset(dates, tmp_path, config)
    assert build_calls == dates
    assert len(first) == 2

    # Stamps carry the seed each day was BUILT with: None for the first window day,
    # day-1's own H/L for day 2 — NOT each day's own H/L.
    stamp = {
        d: pq.read_metadata(
            tmp_path / symbol / d / f"ml_utility_{cache_tag}.parquet"
        ).metadata[_SEED_META_KEY]
        for d in dates
    }
    assert stamp["2026-01-05"] == b"none"
    assert stamp["2026-01-06"] == b"100.0,90.0"

    # Second run over the same window: both days take the cache-hit path — no rebuild.
    second = build_utility_dataset(dates, tmp_path, config)
    assert build_calls == dates  # unchanged: _process_single_date never ran again
    assert len(second) == 2
