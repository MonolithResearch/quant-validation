"""Run the public module entry point against real inline and CSV inputs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
INLINE = ",".join([".01", ".03"] * 8)


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "monolith_validation.cli", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
    )


def record(result):
    def reject_non_json(token):
        pytest.fail(f"non-standard JSON token: {token}")
    return json.loads(result.stdout, parse_constant=reject_non_json)


@pytest.fixture
def matrix_file(tmp_path):
    path = tmp_path / "trials.csv"
    np.savetxt(path, np.tile([[.01, -.01], [.03, .01]], (8, 1)), delimiter=",")
    return path


def complete_args(matrix_file):
    return [INLINE, "--matrix", matrix_file, "--pbo-splits", "4",
            "--trial-sr-std", ".1", "--trades", "80"]


def test_cli_missing_evidence_json_exit_status():
    result = run_cli(INLINE, "--json")
    assert result.returncode == 2
    out = record(result)
    assert out["gate_status"] == out["verdict"] == "INCONCLUSIVE"
    assert out["gate_ready"] is out["gate_pass"] is False
    assert out["pbo"] is None
    assert len(out["missing_reasons"]) == 2
    assert not result.stderr


def test_cli_complete_pass_and_fail_match_api_status(matrix_file):
    passed = run_cli(*complete_args(matrix_file), "--json")
    assert passed.returncode == 0
    assert record(passed)["gate_status"] == "PASS"
    failed = run_cli(*complete_args(matrix_file), "--gate", "min_trades=100", "--json")
    assert failed.returncode == 1
    out = record(failed)
    assert out["gate_status"] == "FAIL"
    assert out["gate_thresholds"]["min_trades"] == 100
    assert any("trades" in reason for reason in out["gate_reasons"])


def test_cli_text_displays_configured_thresholds(matrix_file):
    result = run_cli(*complete_args(matrix_file), "--gate", "min_trades=100,pbo=0.1")
    assert result.returncode == 1
    assert "VERDICT: FAIL" in result.stdout
    assert ">= 100" in result.stdout
    assert "<= 0.1" in result.stdout


def test_headerless_csv_keeps_first_return(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text(".01\n.03\n.02\n", encoding="utf-8")
    result = run_cli(path, "--json")
    assert result.returncode == 2
    assert record(result)["n_obs"] == 3


def test_named_csv_column_and_preferred_column(tmp_path):
    path = tmp_path / "returns.csv"
    path.write_text("date,returns,other\na,.01,1\nb,.03,2\nc,.02,3\n", encoding="utf-8")
    auto = run_cli(path, "--json")
    selected = run_cli(path, "--col", "returns", "--json")
    assert record(auto) == record(selected)
    assert record(auto)["n_obs"] == 3
    missing = run_cli(path, "--col", "typo", "--json")
    assert missing.returncode == 2
    assert record(missing)["gate_status"] == "ERROR"
    assert "typo" in record(missing)["error"]


def test_ambiguous_csv_requires_column(tmp_path):
    path = tmp_path / "ambiguous.csv"
    path.write_text("one,two\n.01,.02\n.02,.01\n.03,.02\n", encoding="utf-8")
    result = run_cli(path, "--json")
    assert result.returncode == 2
    assert "--col" in record(result)["error"]


@pytest.mark.parametrize("args", [
    ["--n-trials", "0"], ["--trial-sr-std", "nan"],
    ["--trial-sr-std", "-1"], ["--trades", "-1"],
    ["--ppy", "0"], ["--benchmark", "inf"],
    ["--trial-sharpes", "1,nan,2"],
    ["--gate", "pbo=nan"], ["--gate", "pbo=1.1"],
    ["--gate", "min_trades=1.5"], ["--gate", "typo=.5"],
    ["--gate", "pbo"], ["--gate", "pbo=.1,pbo=.2"],
    ["--scenarios", "base=nan"], ["--scenarios", "base=.1,bad"],
    ["--pbo-splits", "3"],
])
def test_cli_bad_numeric_and_configuration_input_is_structured_error(args):
    result = run_cli(INLINE, *args, "--json")
    assert result.returncode == 2
    out = record(result)
    assert out["gate_status"] == "ERROR"
    assert out["gate_pass"] is False
    assert out["error"]
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("returns", ["1,nan,2", "1,inf,2", "1,,2", "not-a-file.csv"])
def test_cli_unreadable_or_malformed_returns_do_not_certify(returns):
    result = run_cli(returns, "--json")
    assert result.returncode == 2
    assert record(result)["gate_status"] == "ERROR"


def test_cli_short_sample_outputs_standard_json_nulls():
    result = run_cli(".1", "--json")
    assert result.returncode == 2
    out = record(result)
    assert out["gate_status"] == "INCONCLUSIVE"
    assert out["psr"] is None
    assert out["deflated_sharpe"] is None


@pytest.mark.parametrize("option,value", [("--n-trials", "1.5"), ("--trades", "true")])
def test_cli_argparse_errors_have_usage_and_no_traceback(option, value):
    result = run_cli(INLINE, option, value)
    assert result.returncode == 2
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_help_runs_without_install():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "--pbo-splits" in result.stdout


def test_cli_negative_inline_returns_after_separator():
    result = run_cli("--json", "--", "-0.01,0.02,0.03")
    assert result.returncode == 2
    assert record(result)["n_obs"] == 3


def test_csv_blank_value_cannot_be_silently_dropped(tmp_path):
    path = tmp_path / "missing.csv"
    path.write_text("returns,other\n.01,1\n,2\n.03,3\n", encoding="utf-8")
    result = run_cli(path, "--json")
    assert result.returncode == 2
    assert record(result)["gate_status"] == "ERROR"


def test_trial_sample_with_larger_declaration_uses_api_count():
    result = run_cli(INLINE, "--n-trials", "500", "--trial-sharpes", "0,2", "--json")
    assert result.returncode == 2
    out = record(result)
    assert out["n_trials"] == 500
    assert out["expected_max_sharpe_annual"] > 4


@pytest.mark.parametrize("marker", ["NA", "NaN", "null", "", "N/A", "None", "<NA>"])
@pytest.mark.parametrize("field", ["returns", "matrix", "trial_sharpes"])
def test_first_row_missing_markers_are_not_headers(tmp_path, matrix_file, marker, field):
    path = tmp_path / "missing_first.csv"
    row = f"{marker},.02" if field == "matrix" else marker
    body = ".01,-.01\n.03,.01\n" * 40 if field == "matrix" else ".01\n.03\n" * 40
    path.write_text(row + "\n" + body, encoding="utf-8")
    args = complete_args(matrix_file)
    if field == "returns":
        args[0] = path
    elif field == "matrix":
        args[2] = path
    else:
        args.extend(["--trial-sharpes", path])
    result = run_cli(*args, "--json")
    assert result.returncode == 2, result.stdout
    out = record(result)
    assert out["gate_status"] == out["verdict"] == "ERROR"
    assert out["gate_pass"] is out["gate_ready"] is False
    assert out["error"]
    assert not result.stderr


@pytest.mark.parametrize("field", ["returns", "matrix", "trial_sharpes"])
def test_csv_reader_errors_return_structured_json(tmp_path, matrix_file, field):
    path = tmp_path / "oversized.csv"
    path.write_text("x" * 140_000 + "\n.01\n.03\n", encoding="utf-8")
    args = complete_args(matrix_file)
    if field == "returns":
        args[0] = path
    elif field == "matrix":
        args[2] = path
    else:
        args.extend(["--trial-sharpes", path])
    result = run_cli(*args, "--json")
    assert result.returncode == 2, result.stderr
    out = record(result)
    assert out["gate_status"] == out["verdict"] == "ERROR"
    assert out["gate_pass"] is out["gate_ready"] is False
    assert out["error"]
    assert not result.stderr


@pytest.mark.parametrize("option", ["--ppy", "--n-trials", "--trades", "--pbo-splits", "--gate"])
def test_huge_integer_arguments_return_structured_json(matrix_file, option):
    value = str(10**309)
    if option == "--gate":
        value = "min_trades=" + value
    result = run_cli(*complete_args(matrix_file), option, value, "--json")
    assert result.returncode == 2, result.stdout + result.stderr
    out = record(result)
    assert out["gate_status"] == out["verdict"] == "ERROR"
    assert out["gate_pass"] is out["gate_ready"] is False
    assert out["error"]
    assert not result.stderr


def test_unterminated_csv_quote_is_an_input_error(tmp_path, matrix_file):
    path = tmp_path / "unterminated.csv"
    path.write_text(".01\n.03\n" * 39 + '.01\n".03', encoding="utf-8")
    result = run_cli(path, *complete_args(matrix_file)[1:], "--json")
    assert result.returncode == 2, result.stdout
    assert record(result)["gate_status"] == "ERROR"
    assert not result.stderr


def test_named_matrix_and_trial_headers_preserve_complete_pass(tmp_path):
    returns = tmp_path / "returns.csv"
    matrix = tmp_path / "matrix.csv"
    sharpes = tmp_path / "sharpes.csv"
    returns.write_text("strategy_return\n" + ".01\n.03\n" * 40, encoding="utf-8")
    matrix.write_text("strategy_a,strategy_b\n" + ".01,-.01\n.03,.01\n" * 40, encoding="utf-8")
    sharpes.write_text("annual_sharpe\n0\n2\n", encoding="utf-8")
    result = run_cli(returns, "--matrix", matrix, "--pbo-splits", "4", "--trial-sharpes", sharpes, "--trades", "80", "--json")
    assert result.returncode == 0, result.stdout + result.stderr
    assert record(result)["gate_status"] == "PASS"
    assert record(result)["n_obs"] == 80
    assert record(result)["n_trials"] == 2
