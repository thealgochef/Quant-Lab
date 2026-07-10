"""CLI: argument contract, JSON output, degradation surfacing, exit codes."""

from __future__ import annotations

import json

import pandas as pd

from alpha_lab.propsim.cli import main


def _write_oos(tmp_path, with_outcome_columns: bool = True):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-11-21 09:31:00", "2025-11-21 10:02:00", "2025-11-24 09:45:00"]
            ).tz_localize("US/Eastern"),
            "session": ["ny", "ny", "ny"],
            "label": ["tradeable_reversal", "trap_reversal", "tradeable_reversal"],
        }
    )
    if with_outcome_columns:
        frame["max_mfe_pts"] = [17.5, 4.0, 16.0]
        frame["max_mae_pts"] = [2.25, 15.0, 5.0]
        frame["entry_price"] = [15000.0, 15010.0, 15020.0]
        frame["resolution_type"] = ["tp_hit", "sl_hit", "tp_hit"]
    path = tmp_path / "oos_predictions.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_cli_oos_run_writes_json(tmp_path, capsys):
    oos = _write_oos(tmp_path)
    out = tmp_path / "out.json"
    code = main(
        [
            "--oos", str(oos),
            "--preset", "topstep_50k",
            "--column", "both",
            "--tp-points", "15", "--sl-points", "15",
            "--n", "50", "--seed", "42", "--max-days", "50",
            "--json", str(out),
        ]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "topstep_50k" in stdout
    assert "unrealized_adverse_first" in stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["preset"] == "topstep_50k"
    assert payload["monte_carlo"] == {"n_runs": 50, "seed": 42, "max_days": 50}
    assert set(payload["matrix"]) == {"optimistic", "conservative"}
    for column_cells in payload["matrix"].values():
        assert set(column_cells) == {"realized_only", "unrealized_adverse_first"}
        for cell in column_cells.values():
            assert cell["historical"] is not None
            assert cell["bootstrap"] is not None


def test_cli_strategy_json_sidecar_resolves_tp_sl(tmp_path, capsys):
    oos = _write_oos(tmp_path)
    (tmp_path / "strategy.json").write_text(
        json.dumps({"label_policy": {"tp_points": 15.0, "sl_points": 15.0}}),
        encoding="utf-8",
    )
    code = main(["--oos", str(oos), "--n", "10", "--max-days", "20"])
    assert code == 0
    assert "tp_hit -> +15.0" in capsys.readouterr().out


def test_cli_pre_p1_parquet_states_the_degradation(tmp_path, capsys):
    oos = _write_oos(tmp_path, with_outcome_columns=False)
    code = main(
        ["--oos", str(oos), "--tp-points", "15", "--sl-points", "15",
         "--n", "10", "--max-days", "20"]
    )
    assert code == 0
    stdout = capsys.readouterr().out
    assert "DEGRADED->realized" in stdout
    assert "pre-P1" in stdout


def test_cli_requires_exactly_one_source(tmp_path, capsys):
    assert main(["--preset", "topstep_50k"]) == 2
    oos = _write_oos(tmp_path)
    assert main(
        ["--oos", str(oos), "--source", "executions", str(tmp_path)]
    ) == 2
    err = capsys.readouterr().err
    assert "exactly one trade source" in err


def test_cli_journal_source_requires_tp_sl(tmp_path, capsys):
    journal = tmp_path / "journal"
    journal.mkdir()
    assert main(["--source", "journal", str(journal)]) == 2
    assert "TP/SL barrier points unresolved" in capsys.readouterr().err


def test_cli_unknown_source_kind(tmp_path, capsys):
    assert main(["--source", "carrier-pigeon", str(tmp_path)]) == 2
    assert "unknown --source kind" in capsys.readouterr().err
