"""Exercise real process serialization, including the spawn start method."""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing
import pickle
from pathlib import Path
import subprocess
import sys
import io
import os
from types import SimpleNamespace

import pytest

import numpy as np

import leop_detumble_mc as mc
from antara_models import MissionConfig


def test_spawn_matches_serial(monkeypatch):
    config = MissionConfig()
    serial = mc.run_monte_carlo(4, config, timeout_sec=2., seed=42, n_workers=1)
    monkeypatch.setattr(mc, "ProcessPoolExecutor", partial(
        ProcessPoolExecutor, mp_context=multiprocessing.get_context("spawn")))
    parallel = mc.run_monte_carlo(4, config, timeout_sec=2., seed=42, n_workers=2)
    assert len(parallel) == len(serial) == 4
    for first, second in zip(serial, parallel):
        np.testing.assert_array_equal(first.rate0_deg_sec, second.rate0_deg_sec)
        np.testing.assert_array_equal(first.accel0_deg_sec2, second.accel0_deg_sec2)
        assert first.success == second.success
        assert first.time_to_complete_sec == second.time_to_complete_sec
        assert first.peak_wheel_momentum_nms == second.peak_wheel_momentum_nms


def test_cli_pickle_loads_in_another_process(tmp_path):
    output = tmp_path / "trials.pkl"
    process = subprocess.run([
        sys.executable, str(Path(mc.__file__)), "--n-trials", "3",
        "--workers", "2", "--timeout-sec", "1", "--output", str(output),
    ], check=True, capture_output=True, text=True)
    assert '100.0%' in process.stderr and '3/3' in process.stderr
    assert 'Elapsed=' in process.stderr and 'ETA~00:00:00' in process.stderr
    assert '\x1b' not in process.stderr  # Redirected logs are plain text.
    assert 'Baseline:' in process.stdout and 'Timed out:' in process.stdout
    with output.open("rb") as stream:
        results = pickle.load(stream)
    assert len(results) == 3
    assert all(isinstance(result, mc.DetumbleTrialResult) for result in results)


@pytest.mark.parametrize('interactive', [False, True])
def test_progress_counts_eta_and_terminal_cleanup(monkeypatch, interactive):
    stream = io.StringIO()
    monkeypatch.setattr(stream, 'isatty', lambda: interactive)
    # Replace only this module's bindings, not stdlib objects shared by pytest.
    original_sys, original_shutil, original_time = mc.sys, mc.shutil, mc.time
    terminal_size = original_shutil.get_terminal_size
    monotonic = original_time.monotonic
    stderr = original_sys.stderr
    monkeypatch.setattr(mc, 'sys', SimpleNamespace(stderr=stream))
    monkeypatch.setattr(mc, 'shutil', SimpleNamespace(
        get_terminal_size=lambda fallback=(80, 24): os.terminal_size((200, 24))))
    clock = [0.]
    monkeypatch.setattr(mc, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    assert original_shutil.get_terminal_size is terminal_size
    assert original_time.monotonic is monotonic
    assert original_sys.stderr is stderr
    assert terminal_size(fallback=(80, 24)).columns > 0
    progress = mc._TrialProgress(3, True)
    progress.trial_update(0., 'BDOT', 9.)
    progress.record(SimpleNamespace(success=True, hit_rate_cap=False))
    clock[0] = 10.
    progress.record(SimpleNamespace(success=False, hit_rate_cap=True))
    assert 'ETA~00:00:10' in stream.getvalue()
    progress.record(SimpleNamespace(success=True, hit_rate_cap=False))
    progress.close()
    output = stream.getvalue()
    assert 'Baseline: sim=0.0s BDOT |w|=9.000deg/s' in output
    assert '100.0% 3/3 | OK=2 Timeout=1 Cap=1' in output
    assert output.endswith('\n')
    assert ('\r\033[2K' in output) is interactive


def test_progress_clears_wrapped_terminal_lines(monkeypatch):
    stream = io.StringIO()
    monkeypatch.setattr(stream, 'isatty', lambda: True)
    monkeypatch.setattr(mc, 'sys', SimpleNamespace(stderr=stream))
    monkeypatch.setattr(mc, 'shutil', SimpleNamespace(
        get_terminal_size=lambda fallback=(80, 24): os.terminal_size((40, 24))))
    progress = mc._TrialProgress(1, True)
    progress.render(force=True)
    assert progress.line_count > 1
    progress.record(SimpleNamespace(success=True, hit_rate_cap=False))
    progress.close()
    assert '\033[1A\r\033[2K' in stream.getvalue()


def test_no_progress_single_baseline(tmp_path):
    process = subprocess.run([
        sys.executable, str(Path(mc.__file__)), '--n-trials', '1',
        '--timeout-sec', '1', '--no-progress', '--output', str(tmp_path / 'quiet.pkl'),
    ], check=True, capture_output=True, text=True)
    assert 'Detumble [' not in process.stderr
    assert 'Baseline:' in process.stdout and 'Total trials: 1' in process.stdout
