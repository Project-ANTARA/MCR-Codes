"""Regression guards for scheduled workflow entrypoints and result handling."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from simulations.ditl.compare_scenarios import (load_scenario_data, main as compare_main,
                               compute_mode_duty_cycle, compute_saturation_holds,
                               compute_eclipse_fraction, compute_science_metrics,
                               compute_battery_stats, compute_attitude_error_per_mode,
                               stream_scenario_metrics, scenario_telemetry_path)

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
    metrics, rows = stream_scenario_metrics(scenario_telemetry_path(scenario, tmp_path), scenario, chunksize=1)
    assert rows == 2 and metrics["duration_hours"] == 2 / 3600


@pytest.mark.parametrize("event", ["mode_transitions", "saa_passes", "gs_contacts"])
def test_empty_optional_event_csv_is_valid(tmp_path, event):
    pd.DataFrame({"time_sec": [0, 2], "mode": ["SAFE", "SAFE"]}).to_csv(
        tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
    pd.DataFrame().to_csv(tmp_path / f"leop_24h_ANTARA_Science_{event}.csv", index=False)
    assert load_scenario_data("leop_24h", tmp_path)[event].empty


@pytest.mark.parametrize("mode", [None, "", "TYPO_MODE"])
def test_invalid_mode_cannot_disappear_from_duty_cycle(mode):
    frame = pd.DataFrame({"time_sec": [0., 2., 22.], "mode": ["SAFE", mode, "SAFE"]})
    with pytest.raises(ValueError, match="mode"):
        compute_mode_duty_cycle(frame, 22 / 3600)


def test_boolean_strings_are_not_truthy_saturation_holds():
    frame = pd.DataFrame({"saturation_hold_active": ["False", "True", "False", "True"]})
    assert compute_saturation_holds(frame) == 2


@pytest.mark.parametrize("bad", [None, "unknown", 2])
def test_invalid_saturation_flag_rejected(bad):
    with pytest.raises(ValueError, match="saturation_hold_active"):
        compute_saturation_holds(pd.DataFrame({"saturation_hold_active": [False, bad]}))


def test_string_eclipse_flags_use_duration_weights():
    frame = pd.DataFrame({"time_sec": [0., 2., 22.], "in_eclipse": ["False", "True", "False"]})
    assert compute_eclipse_fraction(frame) == pytest.approx(100 * 20 / 22)


def test_empty_telemetry_is_not_an_optional_event(tmp_path):
    pd.DataFrame().to_csv(tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
    with pytest.raises(pd.errors.EmptyDataError):
        load_scenario_data("leop_24h", tmp_path)


def test_malformed_event_csv_is_not_suppressed(tmp_path):
    path = tmp_path / "leop_24h_ANTARA_Science_saa_passes.csv"
    path.write_text('a,b\n1,2\n3,4,5\n')
    with pytest.raises(pd.errors.ParserError):
        # Provide telemetry to select the flat layout.
        pd.DataFrame({"time_sec": [0, 2], "mode": ["SAFE", "SAFE"]}).to_csv(
            tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
        load_scenario_data("leop_24h", tmp_path)


@pytest.mark.parametrize("flags", [[False, True, False, True], [0, 1, 0, 1], ["0", "1", "0", "1"]])
def test_supported_boolean_encodings_agree(flags):
    assert compute_saturation_holds(pd.DataFrame({"saturation_hold_active": flags})) == 2


def test_valid_mode_duties_conserve_time():
    frame = pd.DataFrame({"time_sec": [0, 2, 22], "mode": ["SAFE", "DETECTION", "RELAY"]})
    duty = compute_mode_duty_cycle(frame, 22 / 3600)
    assert sum(duty.values()) == pytest.approx(100.)
    assert duty["RELAY"] == 0.
    assert duty["DETECTION"] == pytest.approx(100 * 20 / 22)


def test_absent_science_counters_are_unknown_not_zero():
    result = compute_science_metrics(pd.DataFrame({"mode": ["SAFE"]}), pd.DataFrame())
    assert all(pd.isna(value) for value in result.values())


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., "broken"])
@pytest.mark.parametrize("column", ["science_data_mb", "downlinked_mb"])
def test_invalid_science_counter_rejected(column, bad):
    frame = pd.DataFrame({"science_data_mb": [1., 2.], "downlinked_mb": [0., 1.]})
    frame[column] = frame[column].astype(object)
    frame.loc[1, column] = bad
    with pytest.raises(ValueError, match=column):
        compute_science_metrics(frame, pd.DataFrame())


def test_science_totals_include_terminal_flush_and_zero_is_not_efficiency():
    frame = pd.DataFrame({"science_data_mb": [1., 0., 2.], "downlinked_mb": [0., 1., 0.],
                          "interval_duration_sec": [2., 2., 0.]})
    result = compute_science_metrics(frame, pd.DataFrame())
    assert result["science_generated_mb"] == 3.
    assert result["science_downlinked_mb"] == 1.
    assert result["downlink_efficiency_pct"] == pytest.approx(100 / 3)
    zero = compute_science_metrics(frame.assign(science_data_mb=0., downlinked_mb=0.), pd.DataFrame())
    assert zero["science_generated_mb"] == zero["science_downlinked_mb"] == 0.
    assert pd.isna(zero["downlink_efficiency_pct"])


def test_missing_downlink_does_not_fabricate_efficiency():
    result = compute_science_metrics(pd.DataFrame({"science_data_mb": ["1", "2"]}), pd.DataFrame())
    assert result["science_generated_mb"] == 3.
    assert pd.isna(result["science_downlinked_mb"])
    assert pd.isna(result["downlink_efficiency_pct"])


def test_science_counter_overflow_rejected():
    with pytest.raises(FloatingPointError):
        compute_science_metrics(pd.DataFrame({"science_data_mb": [1e308, 1e308]}), pd.DataFrame())


def test_large_equal_byte_totals_have_finite_efficiency():
    result = compute_science_metrics(pd.DataFrame({"science_data_mb": [1e308], "downlinked_mb": [1e308]}), pd.DataFrame())
    assert result["downlink_efficiency_pct"] == 100.


@pytest.mark.parametrize("chunksize", [1, 2, 3, 20])
@pytest.mark.parametrize("legacy", [False, True])
def test_streamed_comparison_matches_full_frame(tmp_path, chunksize, legacy):
    frame = pd.DataFrame({
        "time_sec": [0., 2., 22., 22.25, 24.],
        "interval_duration_sec": [2., 20., .25, 1.75, 0.],
        "mode": ["SAFE", "DETECTION", "DETECTION", "RELAY", "SAFE"],
        "battery_wh": [10., 20., float("nan"), 30., 40.],
        "attitude_error_deg": [1., 2., 4., 3., 0.],
        "saturation_hold_active": [False, True, True, False, True],
        "in_eclipse": [False, True, False, True, False],
        "science_data_mb": [0., 1., 2., 0., 3.], "downlinked_mb": [0., 0., 0., 2., 0.],
        "science_counts_per_bin": ["[100, 200]"] * 5,
    })
    if legacy:
        frame = frame.drop(columns="interval_duration_sec")
    path = tmp_path / "telemetry.csv"
    frame.to_csv(path, index=False)
    result, rows = stream_scenario_metrics(path, "test", chunksize=chunksize)
    assert rows == len(frame)
    assert result["duration_hours"] == 24 / 3600
    assert result["saturation_hold_count"] == compute_saturation_holds(frame)
    for mode, value in compute_mode_duty_cycle(frame, 24 / 3600).items():
        assert result[f"duty_{mode}_pct"] == pytest.approx(value)
    for key, value in compute_battery_stats(frame).items():
        assert result[f"batt_{key}"] == pytest.approx(value)
    for key, value in compute_science_metrics(frame, pd.DataFrame()).items():
        assert result[key] == pytest.approx(value)
    for mode, value in compute_attitude_error_per_mode(frame).items():
        assert result[f"att_err_{mode}_deg"] == pytest.approx(value)
    assert result["eclipse_fraction_pct"] == pytest.approx(compute_eclipse_fraction(frame))


def test_streamed_comparison_reads_only_bounded_selected_columns(tmp_path, monkeypatch):
    path = tmp_path / "telemetry.csv"
    pd.DataFrame({"time_sec": [0, 2, 4], "mode": ["SAFE"] * 3}).to_csv(path, index=False)
    original = pd.read_csv
    calls = []
    def checked(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["chunksize"] == 1
        assert kwargs["usecols"]("battery_wh")
        assert not kwargs["usecols"]("science_counts_per_bin")
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", checked)
    result, rows = stream_scenario_metrics(path, "test", chunksize=1)
    assert rows == 3 and len(calls) == 1
    assert pd.isna(result["science_generated_mb"])


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_streaming_rejects_cross_chunk_time_corruption(tmp_path, chunksize):
    path = tmp_path / "telemetry.csv"
    pd.DataFrame({"time_sec": [0, 2, 4], "interval_duration_sec": [2, 1, 0],
                  "mode": ["SAFE"] * 3}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        stream_scenario_metrics(path, "test", chunksize=chunksize)


def test_streaming_full_duration_check_and_no_guessed_duration(tmp_path):
    path = tmp_path / "telemetry.csv"
    pd.DataFrame({"time_sec": [0, 86400], "mode": ["SAFE"] * 2}).to_csv(path, index=False)
    assert stream_scenario_metrics(path, "leop_24h", chunksize=1, require_full_duration=True)[0]["duration_hours"] == 24.
    with pytest.raises(ValueError, match="8760"):
        stream_scenario_metrics(path, "leop_1yr", chunksize=1, require_full_duration=True)
    pd.DataFrame({"interval_duration_sec": [2, 0], "mode": ["SAFE"] * 2}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="time_sec"):
        stream_scenario_metrics(path, "test")


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_bounded_comparison_rejects_unfinished_final_interval(tmp_path, chunksize):
    path = tmp_path / "unfinished.csv"
    pd.DataFrame({"time_sec": [0., 2.], "interval_duration_sec": [2., 2.],
                  "mode": ["SAFE", "DETECTION"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="terminal"):
        stream_scenario_metrics(path, "leop_24h", chunksize=chunksize)


def test_unfinished_comparison_does_not_publish_artifacts(tmp_path, monkeypatch, capsys):
    pd.DataFrame({"time_sec": [0., 2.], "interval_duration_sec": [2., 2.],
                  "mode": ["SAFE", "DETECTION"]}).to_csv(
        tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
    out = tmp_path / "comparison"
    monkeypatch.setattr(sys, "argv", ["compare", "--results-dir", str(tmp_path), "--output-dir", str(out)])
    with pytest.raises(SystemExit) as exc:
        compare_main()
    assert exc.value.code == 2
    assert "terminal" in capsys.readouterr().err
    assert not (out / "scenario_comparison.csv").exists()


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
