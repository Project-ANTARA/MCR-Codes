"""Regression guards for scheduled workflow entrypoints and result handling."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from compare_scenarios import load_scenario_data, main as compare_main

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("validation_report", ROOT / "scripts/generate_validation_report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


@pytest.mark.parametrize("layout", ["flat", "nested", "legacy"])
def test_comparison_respects_custom_results_directory(tmp_path, layout):
    scenario = "leop_24h"
    folder = tmp_path if layout == "flat" else tmp_path / scenario
    folder.mkdir(exist_ok=True)
    prefix = "" if layout == "legacy" else f"{scenario}_"
    pd.DataFrame({"time_sec": [0, 2], "mode": ["SAFE", "SAFE"]}).to_csv(
        folder / f"{prefix}ANTARA_Science_telemetry.csv", index=False)
    data = load_scenario_data(scenario, tmp_path)
    assert data["telemetry"]["time_sec"].tolist() == [0, 2]


def test_comparison_requires_all_scenarios(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["compare", "--results-dir", str(tmp_path),
                                      "--output-dir", str(tmp_path / "figures"), "--require-all"])
    with pytest.raises(SystemExit) as exc:
        compare_main()
    assert exc.value.code == 2


def test_comparison_rejects_partial_preset(tmp_path, monkeypatch):
    pd.DataFrame({"time_sec": [0, 3600], "mode": ["SAFE", "SAFE"]}).to_csv(
        tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["compare", "--results-dir", str(tmp_path),
                                      "--output-dir", str(tmp_path / "figures"), "--require-full-duration"])
    with pytest.raises(SystemExit) as exc:
        compare_main()
    assert exc.value.code == 2


@pytest.mark.parametrize("script,suffix", list(report.RESULT_NAMES.items()))
def test_report_reads_actual_json_filenames(tmp_path, script, suffix):
    (tmp_path / f"validation_results_{suffix}.json").write_text('{"overall_pass": true}')
    assert report.load_json_result(script, tmp_path)["overall_pass"] is True


def test_report_preserves_failed_subprocess_status(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        assert kwargs["env"]["ANTARA_VALIDATION_OUTPUT_DIR"] == str(tmp_path)
        return subprocess.CompletedProcess(args, 1, "diagnostic", "failure")
    monkeypatch.setattr(report.subprocess, "run", failed)
    assert report.run_test("test_igrf_field.py", tmp_path)["status"] == "FAIL"


def test_report_preserves_timeout_logs(tmp_path, monkeypatch):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args, 300, output=b"partial", stderr=b"details")
    monkeypatch.setattr(report.subprocess, "run", timed_out)
    result = report.run_test("test_igrf_field.py", tmp_path)
    assert (result["status"], result["stdout"], result["stderr"]) == ("TIMEOUT", "partial", "details")


def test_report_rejects_stale_results_before_running(tmp_path, monkeypatch):
    stale = tmp_path / "validation_results_igrf.json"
    stale.write_text('{"overall_pass": true}')
    monkeypatch.setattr(sys, "argv", ["report", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(report, "run_test", lambda *args: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        report.main()
    assert exc.value.code == 2
    assert json.loads(stale.read_text())["overall_pass"] is True


def test_report_fails_when_success_has_no_quantitative_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["report", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(report, "run_test", lambda *args: {"script": args[0], "status": "PASS"})
    assert report.main() is False
    assert "Missing or failing quantitative results" in (tmp_path / "VALIDATION_REPORT.md").read_text()


def test_igrf_validation_rejects_wrong_vector_and_fallback(tmp_path, monkeypatch):
    path = ROOT / "tests/validation/test_igrf_field.py"
    spec = importlib.util.spec_from_file_location("igrf_validation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Keep the magnitude identical but reverse NED: the vector check must fail.
    import numpy as np
    monkeypatch.setenv("ANTARA_VALIDATION_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(module, "reference_field_ned", lambda *args: np.array([1., 2., 3.]))
    monkeypatch.setattr(module.IGRFModel, "get_ned_field_tesla", lambda *args: -np.array([1., 2., 3.])*1e-9)
    assert module.main() is False
    monkeypatch.setattr(module.IGRFModel, "__init__", lambda self, year: setattr(self, "_available", False))
    with pytest.raises(RuntimeError, match="dipole fallback"):
        module.main()
