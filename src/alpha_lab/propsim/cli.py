"""CLI for the prop-firm evaluation walker.

Usage (spec form)::

    python -m alpha_lab.propsim --source executions <dir> [--journal <dir>] \
        --preset topstep_50k --column conservative --n 10000 --seed 42 --json out.json
    python -m alpha_lab.propsim --oos <parquet> --preset topstep_50k ...
    python -m alpha_lab.propsim --source journal <dir> --tp-points 15 --sl-points 15 ...

For ``--oos`` the TP/SL barrier points resolve from (in order) the
``--tp-points``/``--sl-points`` flags, then a ``strategy.json`` next to the
parquet (``label_policy.tp_points``/``sl_points``); ``--source journal``
requires the flags explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from alpha_lab.propsim.engine import FILL_COLUMNS
from alpha_lab.propsim.loaders import (
    LoadedTrades,
    load_executions_trades,
    load_journal_trades,
    load_oos_trades,
)
from alpha_lab.propsim.presets import PRESETS, ruleset_from_preset
from alpha_lab.propsim.report import build_report, format_human

_SOURCE_KINDS = ("executions", "journal")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m alpha_lab.propsim",
        description="Prop-firm evaluation walker: pass-probability from equity paths.",
    )
    parser.add_argument(
        "--source",
        nargs=2,
        metavar=("KIND", "DIR"),
        help=f"trade source: KIND in {{{', '.join(_SOURCE_KINDS)}}} + its directory",
    )
    parser.add_argument("--journal", type=Path, default=None,
                        help="journal dir joined for MFE/MAE (executions source)")
    parser.add_argument("--oos", type=Path, default=None,
                        help="a bundle's oos_predictions.parquet")
    parser.add_argument("--preset", default="topstep_50k", choices=sorted(PRESETS),
                        help="ruleset preset (default: topstep_50k)")
    parser.add_argument("--column", choices=[*FILL_COLUMNS, "both"], default="both",
                        help="fill column(s) to evaluate (default: both)")
    parser.add_argument("--n", type=int, default=10_000, dest="n_runs",
                        help="bootstrap runs (default: 10000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-days", type=int, default=1_000,
                        help="runaway guard per bootstrap run (default: 1000)")
    parser.add_argument("--json", type=Path, default=None, dest="json_out",
                        help="write the full JSON payload here")
    parser.add_argument("--tp-points", type=float, default=None)
    parser.add_argument("--sl-points", type=float, default=None)
    parser.add_argument("--tick-size", type=float, default=0.25,
                        help="tick size for the journal conservative fill model")
    parser.add_argument("--gate-column", default=None,
                        help="OOS boolean gate column (default: all labeled rows)")
    return parser


def _resolve_tp_sl(args: argparse.Namespace) -> tuple[float, float]:
    if args.tp_points is not None and args.sl_points is not None:
        return float(args.tp_points), float(args.sl_points)
    if args.oos is not None:
        contract = args.oos.parent / "strategy.json"
        if contract.is_file():
            policy = json.loads(contract.read_text(encoding="utf-8")).get(
                "label_policy", {}
            )
            tp = args.tp_points if args.tp_points is not None else policy.get("tp_points")
            sl = args.sl_points if args.sl_points is not None else policy.get("sl_points")
            if tp is not None and sl is not None:
                return float(tp), float(sl)
    msg = (
        "TP/SL barrier points unresolved: pass --tp-points/--sl-points "
        "(or point --oos at a bundle whose strategy.json carries label_policy)"
    )
    raise ValueError(msg)


def _load(args: argparse.Namespace) -> LoadedTrades:
    if (args.source is None) == (args.oos is None):
        msg = "exactly one trade source required: --source KIND DIR | --oos PARQUET"
        raise ValueError(msg)
    if args.oos is not None:
        tp, sl = _resolve_tp_sl(args)
        return load_oos_trades(
            args.oos, tp_points=tp, sl_points=sl, gate_column=args.gate_column
        )
    kind, path = args.source
    if kind not in _SOURCE_KINDS:
        msg = f"unknown --source kind {kind!r}; expected one of {', '.join(_SOURCE_KINDS)}"
        raise ValueError(msg)
    directory = Path(path)
    if kind == "executions":
        return load_executions_trades(directory, journal_dir=args.journal)
    tp, sl = _resolve_tp_sl(args)
    return load_journal_trades(
        directory, tp_points=tp, sl_points=sl, tick_size=args.tick_size
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        loaded = _load(args)
        ruleset = ruleset_from_preset(args.preset)
        columns = FILL_COLUMNS if args.column == "both" else (args.column,)
        report = build_report(
            loaded,
            args.preset,
            ruleset,
            columns=columns,
            n_runs=args.n_runs,
            seed=args.seed,
            max_days=args.max_days,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"propsim: {exc}", file=sys.stderr)
        return 2
    print(format_human(report))
    if args.json_out is not None:
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=False), encoding="utf-8"
        )
        print(f"\nJSON written: {args.json_out}")
    return 0
