"""CLI for the canonical report: 0=PASS, 1=FAIL, 2=INCONCLUSIVE or input error.

Adapted from the author's pre-existing local gate-check CLI; provenance and
review decisions are recorded in WORK_LOG.md. No separate gate logic lives here.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

from .core import litigious_report


def _table(argument):
    """Read CSV/text with an optional header, or an inline numeric list."""
    path = Path(argument)
    try:
        exists = path.is_file()
    except OSError:  # Long inline arrays need not be valid filesystem paths.
        exists = False
    if exists:
        if path.suffix.lower() not in (".csv", ".txt"):
            raise ValueError("use a .csv or .txt file; no Parquet engine is required")
        text = path.read_text(encoding="utf-8-sig")
        rows = list(csv.reader(io.StringIO(text), strict=True))
        if not rows:
            return None, []
        if any(not row or len(row) != len(rows[0]) for row in rows):
            raise ValueError("CSV has empty rows or inconsistent column counts")
        # These are missing/nonfinite observations, never implicit header names.
        reserved = {"", "na", "n/a", "nan", "+nan", "-nan", "null", "none", "<na>",
                    "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
        if any(cell.strip().casefold() in reserved for cell in rows[0]):
            raise ValueError("CSV first row contains a missing or nonfinite value")
        try:
            [float(cell) for cell in rows[0]]
        except ValueError:
            header = [cell.strip() for cell in rows[0]]
            if any(not cell for cell in header) or len(set(header)) != len(header):
                raise ValueError("CSV header names must be nonempty and unique")
            return header, rows[1:]
        return None, rows
    # A malformed token is an error, not an observation to drop.
    tokens = argument.replace("\n", ",").split(",") if "," in argument or "\n" in argument else argument.split()
    if not tokens or any(not token.strip() for token in tokens):
        raise ValueError("provide a readable CSV/text file or a nonempty numeric list")
    try:
        values = [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError("provide a readable CSV/text file or a numeric list") from exc
    return None, [[value] for value in values]


def _returns(argument, column=None):
    header, rows = _table(argument)
    if column is not None:
        if header is None or column not in header:
            raise ValueError(f"returns column {column!r} does not exist")
        index = header.index(column)
    else:
        width = len(header) if header is not None else (len(rows[0]) if rows else 1)
        preferred = [] if header is None else [
            i for i, name in enumerate(header)
            if name.lower() in {"return", "returns", "ret", "rets", "pnl", "r", "net", "net_return"}
        ]
        if len(preferred) == 1:
            index = preferred[0]
        elif width == 1:
            index = 0
        else:
            raise ValueError("ambiguous returns columns; supply a named CSV column with --col")
    try:
        return np.array([float(row[index]) for row in rows], dtype=float)
    except (ValueError, IndexError) as exc:
        raise ValueError("returns column contains a missing or nonnumeric value") from exc


def _matrix(argument):
    header, rows = _table(argument)
    width = len(header) if header is not None else (len(rows[0]) if rows else 0)
    try:
        return np.array(rows, dtype=float).reshape(len(rows), width)
    except ValueError as exc:
        raise ValueError("returns_matrix must contain only numeric trial columns") from exc


def _pairs(argument, *, gate=False):
    if argument is None:
        return None
    result = {}
    for item in argument.split(","):
        if item.count("=") != 1:
            raise ValueError("expected comma-separated key=value pairs")
        key, value = (part.strip() for part in item.split("="))
        if not key or not value or key in result:
            raise ValueError("key=value pairs must have unique nonempty keys and values")
        try:
            result[key] = int(value) if gate and key == "min_trades" else float(value)
        except ValueError as exc:
            raise ValueError(f"invalid numeric value for {key}") from exc
    return result


def _format(value):
    return "n/a" if value is None else f"{value:.6g}"


def _render(report):
    record = report.to_dict()
    gate = record["gate_thresholds"]
    lines = [
        "MONOLITH GATE-CHECK | research diagnostics",
        f"Observations: {record['n_obs']}",
        f"Annualised Sharpe: {_format(record['sharpe'])}",
        f"PSR (supplied benchmark): {_format(record['psr'])}",
        f"Search trials: {record['n_trials']}",
        f"Expected maximum annual Sharpe: {_format(record['expected_max_sharpe_annual'])}",
        f"Deflated Sharpe: {_format(record['deflated_sharpe'])} | gate >= {gate['deflated_sharpe']:g}",
        f"PBO: {_format(record['pbo'])} | gate <= {gate['pbo']:g}",
        f"Independent trades: {_format(record['trade_count'])} | gate >= {gate['min_trades']}",
        f"Evidence complete: {record['gate_ready']}",
        f"VERDICT: {record['gate_status']}",
    ]
    lines.extend(f"  - {reason}" for reason in record["gate_reasons"])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="gate-check",
        description="Compute PSR/DSR/PBO diagnostics and an evidence-complete research gate.",
    )
    parser.add_argument("returns", help="CSV/text file or inline comma-separated per-period returns")
    parser.add_argument("--col", help="explicit returns column in a CSV header")
    parser.add_argument("--ppy", type=int, default=252, help="periods per year (default 252)")
    parser.add_argument("--n-trials", type=int, default=1, help="full declared search count; never reduced")
    parser.add_argument("--trial-sharpes", help="annualised Sharpes, inline or single-column CSV/text")
    parser.add_argument("--trial-sr-std", type=float, help="per-period std of trial Sharpes; overrides the sample")
    parser.add_argument("--matrix", help="CSV/text T x M returns matrix; all columns must be numeric trials")
    parser.add_argument("--pbo-splits", type=int, default=16, help="even CSCV block count (default 16)")
    parser.add_argument("--scenarios", help="annual scenario Sharpes: base=1.1,cost3x=0.4")
    parser.add_argument("--trades", type=int, help="nonnegative independent trade count (never inferred)")
    parser.add_argument("--benchmark", type=float, default=0.0, help="per-period PSR/MinTRL benchmark, not DSR")
    parser.add_argument("--gate", help="overrides: deflated_sharpe=0.5,pbo=0.2,min_trades=50")
    parser.add_argument("--json", action="store_true", help="strict JSON; undefined statistics become null")
    args = parser.parse_args(argv)
    try:
        report = litigious_report(
            _returns(args.returns, args.col),
            periods_per_year=args.ppy,
            n_trials=args.n_trials,
            trial_sharpes=_returns(args.trial_sharpes) if args.trial_sharpes is not None else None,
            trial_sr_std=args.trial_sr_std,
            returns_matrix=_matrix(args.matrix) if args.matrix is not None else None,
            pbo_splits=args.pbo_splits,
            scenario_sharpes=_pairs(args.scenarios),
            sr_benchmark=args.benchmark,
            trade_count=args.trades,
            gate=_pairs(args.gate, gate=True),
        )
    except (ValueError, OSError, csv.Error, OverflowError) as exc:
        if args.json:
            print(json.dumps({"gate_status": "ERROR", "verdict": "ERROR", "gate_pass": False,
                              "gate_ready": False, "error": str(exc)}, allow_nan=False))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        out = report.to_dict()
        out["verdict"] = report.gate_status
        print(json.dumps(out, indent=2, allow_nan=False))
    else:
        print(_render(report))
    return {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}[report.gate_status]


if __name__ == "__main__":
    raise SystemExit(main())
