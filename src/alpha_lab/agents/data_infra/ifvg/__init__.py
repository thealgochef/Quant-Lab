"""IFVG v2 replay plus separate v3 deterministic-context evidence.

Strategy-Core owns bars, FVGs, the sequential FSM, IDs, geometry, and trade
resolution. Quant-Lab authorizes sources, drives the fixed nonsealed chain,
adds counterfactual candidate labels, validates foreign keys/invariants, and
writes content-addressed immutable datasets.

The v1 wide candidate stream remains a read-only reproduction lane. It cannot
produce eligible decisions, executed trades, or performance statistics.

The v3 context lane references the accepted v2 artifact and persists only new
normalized context/link tables. It is measurement-only and invokes no training
or evaluation.
"""
