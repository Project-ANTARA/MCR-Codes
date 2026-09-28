"""Reject corrupt time accounting instead of publishing plausible summaries."""
import sys

import numpy as np
import pandas as pd
import pytest

from simulations.ditl.compare_scenarios import main as compare_main
from simulations.helper.telemetry_metrics import interval_weights, streaming_summary, weighted_mean, plotting_sample


def telemetry():
    return pd.DataFrame({
        "time_sec": [0., 2., 22.], "interval_duration_sec": [2., 20., 0.],
        "state_time_sec": [2., 22., 22.], "mode": ["SAFE"] * 3,
        "battery_wh": [10., 20., 30.],
        "wheel_mom_x": [0.] * 3, "wheel_mom_y": [0.] * 3,
        "wheel_mom_z": [0.] * 3,
    })


@pytest.mark.parametrize("times", [[0., np.nan, 22.], [0., 2., np.inf],
                                   [0., 0., 22.], [2., 0., 22.]])
def test_invalid_legacy_timestamps(times):
    with pytest.raises(ValueError, match="timestamps"):
        interval_weights(pd.DataFrame({"time_sec": times}))


def test_legacy_and_fractional_event_intervals_remain_supported():
    times = np.array([0., 1.23456789, 3.23456789, 4.])
    legacy = pd.DataFrame({"time_sec": times})
    weights = interval_weights(legacy)
    current = legacy.assign(interval_duration_sec=weights, state_time_sec=times + weights)
    np.testing.assert_array_equal(interval_weights(current), weights)
    assert weights.sum() == 4.


@pytest.mark.parametrize("column,value", [("interval_duration_sec", 3.),
                                         ("state_time_sec", 3.),
                                         ("state_time_sec", np.nan)])
def test_inconsistent_metadata(column, value):
    frame = telemetry()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        interval_weights(frame)


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_valid_summary_independent_of_chunk_boundaries(tmp_path, chunksize):
    path = tmp_path / "telemetry.csv"
    telemetry().to_csv(path, index=False)
    summary = streaming_summary(path, chunksize)
    assert summary["duration_sec"] == 22.
    assert summary["metrics"]["battery_wh"]["mean"] == pytest.approx(420 / 22)


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_corruption_at_chunk_boundary(tmp_path, chunksize):
    frame = telemetry()
    frame.loc[1, "interval_duration_sec"] = 19.
    frame.loc[1, "state_time_sec"] = 21.
    path = tmp_path / "telemetry.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="disagree|discontinuous"):
        streaming_summary(path, chunksize)


def test_truncated_summary_rejected(tmp_path):
    path = tmp_path / "telemetry.csv"
    telemetry().iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="terminal"):
        streaming_summary(path, chunksize=1)


def test_annual_timestamp_does_not_relax_absolute_tolerance():
    frame = telemetry()
    frame["time_sec"] += 31_000_000
    frame["state_time_sec"] += 31_000_000
    frame.loc[0, "state_time_sec"] += .01
    with pytest.raises(ValueError, match="state times"):
        interval_weights(frame)


def test_complete_preset_with_false_interval_accounting_rejected(tmp_path, monkeypatch, capsys):
    pd.DataFrame({"time_sec": [0., 86400.], "interval_duration_sec": [2., 0.],
                  "mode": ["SAFE", "SAFE"]}).to_csv(
        tmp_path / "leop_24h_ANTARA_Science_telemetry.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["compare", "--results-dir", str(tmp_path),
                       "--output-dir", str(tmp_path / "out"), "--require-full-duration"])
    with pytest.raises(SystemExit) as exc:
        compare_main()
    assert exc.value.code == 2
    assert "interval durations disagree" in capsys.readouterr().err
    assert not (tmp_path / "out/scenario_comparison.csv").exists()


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_measurement_rejected_by_both_reporting_paths(tmp_path, value):
    frame = telemetry()
    frame.loc[1, "battery_wh"] = value
    with pytest.raises(ValueError, match="Infinite"):
        weighted_mean(frame.battery_wh, frame.interval_duration_sec)
    path = tmp_path / "infinite.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Infinite.*battery_wh"):
        streaming_summary(path, chunksize=1)


@pytest.mark.parametrize("weights", [[1., -1.], [1., np.nan], [1., np.inf]])
def test_invalid_mean_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        weighted_mean([1., 2.], pd.Series(weights))


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_missing_measurements_have_explicit_coverage(tmp_path, chunksize):
    frame = telemetry()
    frame.loc[1, "battery_wh"] = np.nan
    frame["unavailable_scalar"] = np.nan
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    summary = streaming_summary(path, chunksize)
    battery = summary["metrics"]["battery_wh"]
    assert battery == {"mean": 10., "min": 10., "max": 30.,
                       "valid_duration_sec": 2., "missing_duration_sec": 20., "missing_rows": 1}
    assert summary["metrics"]["unavailable_scalar"] == {
        "mean": None, "min": None, "max": None, "valid_duration_sec": 0.,
        "missing_duration_sec": 22., "missing_rows": 3}
    assert weighted_mean(frame.battery_wh, frame.interval_duration_sec) == battery["mean"]


def test_normalized_mean_avoids_finite_product_overflow(tmp_path):
    frame = telemetry()
    frame["battery_wh"] = 1e308
    assert weighted_mean(frame.battery_wh, frame.interval_duration_sec) == pytest.approx(1e308)
    path = tmp_path / "large.csv"
    frame.to_csv(path, index=False)
    assert streaming_summary(path, chunksize=1)["metrics"]["battery_wh"]["mean"] == pytest.approx(1e308)


@pytest.mark.parametrize("chunksize", [1, 2, 20])
@pytest.mark.parametrize("bad_row", [0, 1, 2])
def test_malformed_scalar_rejected_independent_of_chunks(tmp_path, chunksize, bad_row):
    frame = telemetry()
    frame["battery_wh"] = frame.battery_wh.astype(object)
    frame.loc[bad_row, "battery_wh"] = "broken"
    path = tmp_path / "malformed.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="battery_wh"):
        streaming_summary(path, chunksize)


@pytest.mark.parametrize("chunksize", [1, 2, 20])
def test_metadata_and_nullable_boolean_schema_is_chunk_independent(tmp_path, chunksize):
    frame = telemetry()
    frame["science_cell_id"] = [None, "123", "abc"]
    frame["science_event_type"] = [None, "measurement", None]
    frame["science_counts_per_bin"] = [None, "[1, 2]", "[]"]
    frame["science_saa_state_at_sample"] = [None, True, None]
    path = tmp_path / "metadata.csv"
    frame.to_csv(path, index=False)
    metrics = streaming_summary(path, chunksize)["metrics"]
    assert "science_cell_id" not in metrics
    assert "science_event_type" not in metrics
    assert "science_counts_per_bin" not in metrics
    assert metrics["science_saa_state_at_sample"] == {
        "mean": 1., "min": 1., "max": 1., "valid_duration_sec": 20.,
        "missing_duration_sec": 2., "missing_rows": 2}


@pytest.mark.parametrize("mode", [None, "", "TYPO_MODE"])
def test_streaming_mode_coverage_cannot_silently_drop_time(tmp_path, mode):
    frame = telemetry()
    frame.loc[1, "mode"] = mode
    path = tmp_path / "mode.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="mode"):
        streaming_summary(path, chunksize=1)


def test_invalid_boolean_scalar_rejected(tmp_path):
    frame = telemetry()
    frame["in_eclipse"] = [0, 2, 0]
    path = tmp_path / "boolean.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="in_eclipse"):
        streaming_summary(path, chunksize=1)


@pytest.mark.parametrize("rows,limit", [(101, 10), (43, 7), (11, 10), (10, 10), (3, 10)])
def test_plot_sample_spans_complete_file_uniformly(tmp_path, rows, limit):
    frame = pd.DataFrame({"time_sec": np.arange(rows), "note": ["multiline\nvalue"] * rows})
    path = tmp_path / "plot.csv"
    frame.to_csv(path, index=False)
    sample, sampled = plotting_sample(path, limit)
    expected = np.linspace(0, rows - 1, min(limit, rows), dtype=int)
    assert sample.time_sec.tolist() == expected.tolist()
    assert sample.note.tolist() == ["multiline\nvalue"] * len(expected)
    assert sampled == (rows > limit)
    assert sample.equals(plotting_sample(path, limit)[0])


@pytest.mark.parametrize("limit", [0, 1, -1, 2.5, True])
def test_plot_sample_rejects_invalid_limit(tmp_path, limit):
    path = tmp_path / "plot.csv"
    pd.DataFrame({"time_sec": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="limit"):
        plotting_sample(path, limit)


@pytest.mark.parametrize("rows", [0, 1])
def test_plot_sample_preserves_small_file_schema(tmp_path, rows):
    frame = pd.DataFrame({"time_sec": np.arange(rows), "value": np.arange(rows)})
    path = tmp_path / "small.csv"
    frame.to_csv(path, index=False)
    sample, sampled = plotting_sample(path, 2)
    assert list(sample.columns) == list(frame.columns)
    assert len(sample) == rows
    assert not sampled


def test_plot_sampling_uses_two_bounded_csv_passes(tmp_path, monkeypatch):
    path = tmp_path / "plot.csv"
    pd.DataFrame({"time_sec": range(43), "value": range(43)}).to_csv(path, index=False)
    original = pd.read_csv
    calls = []
    def checked(*args, **kwargs):
        assert kwargs.get("chunksize") == 7
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", checked)
    sample, sampled = plotting_sample(path, 7)
    assert len(calls) == 2 and calls[0]["usecols"] == [0]
    assert sampled and sample.time_sec.tolist() == [0, 7, 14, 21, 28, 35, 42]
