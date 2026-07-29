"""IFVG capture research harness (IFVG window).

QL is the DRIVER only — every strategy semantic lives in ``strategy_core``:
bars come from ``candles/time_batch.py``, levels from
``runtime/levels.StrategyLevelState``, the FSM from
``strategies/ifvg_smc/replay.run_day``, labels from the shared MAE-first
kernel. This package orchestrates the store replay in three phases:

* Phase A (``day_artifacts``) — parallel, profile-independent, one reader drain
  per day: 8-TF time bars + the level timeline, cached per day with seed
  stamps.
* Phase C (``capture_driver``/``dataset``) — sequential seeded ``run_day``
  chain over the cached artifacts (seconds per day), per-day capture parquets
  trusted by (profile_hash, entering seed_hash).
* Labels/features/reports — offline composition over the capture rows and the
  cached 1m bars.
"""
