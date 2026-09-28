#!/usr/bin/env python3
"""LEOP Detumble Monte Carlo Capability Envelope.

Runs synthetic detumble trials using only ADCSModel and MissionConfig
(no Orekit, no full simulator). Measures capability envelope for
tip-off rate combinations. Acceleration is derived from torque, not sampled.
"""

from __future__ import annotations

# Support direct execution and qualified simulations.* imports.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import math
import pickle
import shutil
import sys
import textwrap
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Import from antara_models (Orekit-free)
from simulations.helper.antara_models import ADCSModel, MissionConfig


@dataclass
class DetumbleTrialResult:
    """Result of a single detumble trial."""
    rate0_deg_sec: np.ndarray
    accel0_deg_sec2: np.ndarray
    rate0_mag: float
    accel0_mag: float
    success: bool
    hit_rate_cap: bool
    time_to_handover_sec: float
    time_to_complete_sec: float
    peak_wheel_momentum_nms: float
    failure_reason: str = ""


# Expose rate cap from ADCSModel._integrate_body_rates
RATE_CAP_DEG_SEC = 120.0  # max(120.0, 10.0 * config.max_slew_rate_deg_sec)


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _TrialProgress:
    """Parent-only progress display; workers never write to the terminal."""

    def __init__(self, total: int, enabled: bool):
        self.total, self.enabled = total, enabled
        self.stream = sys.stderr
        self.interactive = self.stream.isatty()
        self.started = time.monotonic()
        self.batch_started = self.started
        self.last_update = float('-inf')
        self.done = self.successes = self.capped = 0
        self.line_open = False
        self.line_count = 1

    def render(self, detail: str = "", force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_update < (0.25 if self.interactive else 5.0):
            return
        self.last_update = now
        fraction = self.done / self.total
        filled = int(16 * fraction)
        rate = (self.done - 1) / max(now - self.batch_started, 1e-9) if self.done > 1 else 0.
        eta = _duration((self.total - self.done) / rate) if rate else '--:--:--'
        if self.done == self.total:
            eta = '00:00:00'
        text = (f"Detumble [{'#' * filled}{'-' * (16 - filled)}] {100*fraction:5.1f}% "
                f"{self.done}/{self.total} | OK={self.successes} "
                f"Failed={self.done-self.successes} Cap={self.capped} | "
                f"Elapsed={_duration(now-self.started)} ETA~{eta}")
        if detail:
            text += f" | {detail}"
        prefix = ''
        if self.interactive:
            prefix = '\r\033[2K'
            if self.line_open:
                prefix += '\033[1A\r\033[2K' * (self.line_count - 1)
            width = max(1, shutil.get_terminal_size().columns - 1)
            lines = textwrap.wrap(text, width=width)
            self.line_count = len(lines)
            text = '\n'.join(lines)
        print(prefix + text, file=self.stream, end='' if self.interactive else '\n', flush=True)
        self.line_open = self.interactive

    def record(self, result: DetumbleTrialResult) -> None:
        self.done += 1
        self.successes += int(result.success)
        self.capped += int(result.hit_rate_cap)
        if self.done == 1:
            self.batch_started = time.monotonic()
        self.render(force=self.done == self.total)

    def trial_update(self, t_sec: float, phase: str, rate: float) -> None:
        label = 'Baseline' if self.done == 0 else 'Trial'
        self.render(f"{label}: sim={t_sec:.1f}s {phase} |w|={rate:.3f}deg/s")

    def close(self) -> None:
        if self.line_open:
            print(file=self.stream, flush=True)
            self.line_open = False


def rotation_matrix_z(angle: float) -> np.ndarray:
    """Return 3x3 rotation matrix about Z axis."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def synthetic_b_body(
    adcs: ADCSModel,
    t_sec: float,
    b_magnitude_t: float,
    orbital_period_sec: float,
) -> np.ndarray:
    """Return synthetic B-field vector in body frame.

    Rotates a fixed-magnitude inertial B-vector at orbital rate,
    projects into current body frame.
    """
    omega_orb = 2.0 * math.pi / orbital_period_sec
    # Inertial B in dipole frame (e.g., along +Z inertial)
    b_inertial = np.array([0.0, 0.0, b_magnitude_t], dtype=float)
    # Rotate by orbital motion
    angle = omega_orb * t_sec
    rot = rotation_matrix_z(angle)
    b_inertial_rotated = rot @ b_inertial
    # Project to current body frame
    return adcs.rotate_inertial_vector_to_body(adcs.current_quaternion, b_inertial_rotated)


def run_single_trial(
    rate0: np.ndarray,
    accel0: np.ndarray,
    config: MissionConfig,
    dt_sec: float = 0.5,
    timeout_sec: float = 3600.0,
    progress_callback: Optional[Callable[[float, str, float], None]] = None,
) -> DetumbleTrialResult:
    """Run a single detumble trial with given initial conditions.

    Args:
        rate0: Initial body rates [deg/s] (x, y, z)
        accel0: Legacy vector argument; must be zero (acceleration follows torque)
        config: MissionConfig for ADCSModel
        dt_sec: Timestep for detumble loop
        timeout_sec: Maximum simulation time before giving up
        progress_callback: Optional parent-side progress observer (no physics changes)

    Returns:
        DetumbleTrialResult with trial metrics
    """
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be finite and positive")
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        raise ValueError("timeout_sec must be finite and positive")
    rate0 = np.asarray(rate0, dtype=float)
    accel0 = np.asarray(accel0, dtype=float)
    if rate0.shape != (3,) or accel0.shape != (3,) or not np.all(np.isfinite([rate0, accel0])):
        raise ValueError("Initial vectors must contain three finite components")
    if np.any(accel0 != 0.0):
        raise ValueError("Initial acceleration is not an independent rigid-body state; supply zero. A disturbance requires a torque/time model.")
    adcs = ADCSModel(config)
    adcs.body_rate_deg_sec = rate0.copy()
    adcs.body_accel_deg_sec2 = accel0.copy()

    rate0_mag = float(np.linalg.norm(rate0))
    accel0_mag = float(np.linalg.norm(accel0))

    hit_rate_cap = False
    time_to_handover_sec = 0.0
    time_to_complete_sec = 0.0
    peak_wheel_momentum_nms = 0.0
    handover_done = False
    failure_reason = ""

    # B-field magnitude for synthetic model
    b_mag = config.earth_B_field_T

    # Orbital period from config (approximate for 600km SSO)
    orbital_period_sec = 97.5 * 60.0

    t_sec = 0.0
    while t_sec < timeout_sec:
        if progress_callback is not None:
            progress_callback(t_sec, adcs.detumble_phase, float(np.linalg.norm(adcs.body_rate_deg_sec)))
        # Check rate cap (from _integrate_body_rates)
        if np.any(np.abs(adcs.body_rate_deg_sec) >= RATE_CAP_DEG_SEC - 1e-3):
            hit_rate_cap = True

        # Track peak wheel momentum
        wheel_mag = float(np.linalg.norm(adcs.wheel_momentum_nms))
        if wheel_mag > peak_wheel_momentum_nms:
            peak_wheel_momentum_nms = wheel_mag

        # Track handover (BDOT -> WHEEL_DAMP)
        if not handover_done and adcs.detumble_phase == "WHEEL_DAMP":
            handover_done = True
            time_to_handover_sec = t_sec

        # Check completion
        if adcs.detumble_complete:
            time_to_complete_sec = t_sec
            break

        # Step detumble physics
        b_body = synthetic_b_body(adcs, t_sec, b_mag, orbital_period_sec)
        interval = min(dt_sec, timeout_sec - t_sec)
        try:
            wheel_torque = adcs.step_detumble(b_body, interval)
            adcs.update_wheel_momentum(wheel_torque, interval, unload_scale=0.0, b_projection_scale=0.0)
        except FloatingPointError as exc:
            hit_rate_cap = True
            failure_reason = str(exc)
            break
        peak_wheel_momentum_nms = max(peak_wheel_momentum_nms, float(np.linalg.norm(adcs.wheel_momentum_nms)))
        t_sec += interval
        if np.any(np.abs(adcs.wheel_momentum_nms) >= config.wheel_max_momentum_nms):
            failure_reason = "Wheel momentum capacity exceeded"
            break
        if adcs.detumble_complete:
            time_to_complete_sec = t_sec
            break

    success = adcs.detumble_complete and not failure_reason and t_sec <= timeout_sec
    if not adcs.detumble_complete:
        time_to_complete_sec = float('inf')

    return DetumbleTrialResult(
        rate0_deg_sec=rate0,
        accel0_deg_sec2=accel0,
        rate0_mag=rate0_mag,
        accel0_mag=accel0_mag,
        success=success,
        hit_rate_cap=hit_rate_cap,
        time_to_handover_sec=time_to_handover_sec,
        time_to_complete_sec=time_to_complete_sec if success else float('inf'),
        peak_wheel_momentum_nms=peak_wheel_momentum_nms,
        failure_reason=failure_reason,
    )


def run_monte_carlo(
    n_trials: int,
    config: MissionConfig,
    dt_sec: float = 0.5,
    timeout_sec: Optional[float] = None,
    seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> list[DetumbleTrialResult]:
    """Run Monte Carlo sweep of detumble trials.

    Args:
        n_trials: Number of trials to run
        config: MissionConfig for ADCSModel
        dt_sec: Timestep for detumble loop
        timeout_sec: Maximum simulation time (default: 3x baseline)
        seed: RNG seed
        n_workers: Number of parallel workers (default: CPU count)
        show_progress: Display progress on stderr (throttled lines when redirected)

    Returns:
        List of DetumbleTrialResult
    """
    if n_trials < 1:
        raise ValueError("n_trials must include at least the baseline trial")
    if not math.isfinite(dt_sec) or dt_sec <= 0.0:
        raise ValueError("dt_sec must be finite and positive")
    if timeout_sec is not None and (not math.isfinite(timeout_sec) or timeout_sec <= 0.0):
        raise ValueError("timeout_sec must be finite and positive")
    if n_workers is not None and n_workers < 1:
        raise ValueError("n_workers must be positive")

    rng = np.random.default_rng(seed)

    # Sample rate0 magnitude ~Uniform(0, 60) deg/s, direction uniform on sphere
    # Acceleration is derived from torque, not independently sampled.
    tasks = []
    for i in range(n_trials):
        rate0_mag = rng.uniform(0.0, 60.0)

        # Uniform direction on sphere
        phi = rng.uniform(0.0, 2.0 * math.pi)
        costheta = rng.uniform(-1.0, 1.0)
        theta = math.acos(costheta)
        rate0_dir = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=float)

        rate0 = rate0_dir * rate0_mag
        # Angular acceleration follows applied torque; it is not a sampled state.
        accel0 = np.zeros(3)
        tasks.append((i, rate0, accel0))

    # Run baseline trial first to determine timeout
    print("Running baseline trial at default tip-off rates...", flush=True)
    baseline_rate0 = np.array([
        config.initial_body_rate_deg_sec_x,
        config.initial_body_rate_deg_sec_y,
        config.initial_body_rate_deg_sec_z,
    ], dtype=float)
    baseline_accel0 = np.zeros(3)

    progress = _TrialProgress(n_trials, show_progress)
    print(f"dt={dt_sec:g} s | workers={n_workers or 'auto (CPU count)'} | "
          f"trials={n_trials} (includes baseline)", flush=True)
    print(f"Baseline rates={baseline_rate0} deg/s | accelerations={baseline_accel0} deg/s^2", flush=True)
    try:
        baseline_result = run_single_trial(
            baseline_rate0, baseline_accel0, config, dt_sec, timeout_sec or 10000.0,
            progress.trial_update if show_progress else None,
        )
    finally:
        progress.close()
    baseline_time = baseline_result.time_to_complete_sec
    if timeout_sec is None:
        timeout_sec = 3.0 * max(baseline_time, 1000.0) if math.isfinite(baseline_time) else 10000.0
    print(f"Baseline: {'PASS' if baseline_result.success else 'FAILED'} | "
          f"time_to_complete={baseline_time:.1f} s | trial timeout={timeout_sec:.1f} s", flush=True)

    # Run remaining trials
    results = [baseline_result]
    remaining_tasks = [
        (index, rate, accel, config, dt_sec, timeout_sec)
        for index, rate, accel in tasks[1:]
    ]

    try:
        progress.record(baseline_result)
        if n_workers == 1:
            for task in remaining_tasks:
                result = run_single_trial(*task[1:], progress_callback=progress.trial_update if show_progress else None)
                results.append(result)
                progress.record(result)
        elif remaining_tasks:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(worker, task): task[0] for task in remaining_tasks}
                pending = set(futures)
                ordered_results = {}
                while pending:
                    completed, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
                    for future in completed:
                        result = future.result()
                        ordered_results[futures[future]] = result
                        progress.record(result)
                    if pending:
                        progress.render("Workers running")
                results.extend(ordered_results[index] for index in sorted(ordered_results))
    finally:
        progress.close()

    return results


def worker(task):
    """Worker function for ProcessPoolExecutor."""
    return run_single_trial(task[1], task[2], task[3], task[4], task[5])


if __name__ == "__main__":
    import argparse
    # Serialize results with the importable module name, not __main__, so the
    # analysis command can load CLI-generated pickles in a fresh process.
    from simulations.detumble.leop_detumble_mc import run_monte_carlo

    parser = argparse.ArgumentParser(description="LEOP Detumble Monte Carlo")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of MC trials")
    parser.add_argument("--dt-sec", type=float, default=0.5, help="Timestep (s)")
    parser.add_argument("--timeout-sec", type=float, default=None, help="Timeout (s), default 3x baseline")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--output", type=Path, default=Path("detumble_mc_results.pkl"), help="Output pickle path")
    parser.add_argument("--no-progress", action="store_true", help="Hide progress display (keep setup and final summary)")

    args = parser.parse_args()

    config = MissionConfig()
    print(f"Running {args.n_trials} trials with seed={args.seed}...")

    results = run_monte_carlo(
        n_trials=args.n_trials,
        config=config,
        dt_sec=args.dt_sec,
        timeout_sec=args.timeout_sec,
        seed=args.seed,
        n_workers=args.workers,
        show_progress=not args.no_progress,
    )

    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    # Quick summary
    success_count = sum(1 for r in results if r.success)
    print(f"\n=== Results ===")
    print(f"Total trials: {len(results)}")
    print(f"Successful: {success_count} ({100.0*success_count/len(results):.1f}%)")
    print(f"Timed out: {sum(not r.success and not r.failure_reason for r in results)}")
    print(f"Capacity/numerical failures: {sum(bool(r.failure_reason) for r in results)}")
    print(f"Hit rate cap: {sum(1 for r in results if r.hit_rate_cap)} trials")
    if success_count > 0:
        success_results = [r for r in results if r.success]
        print(f"Avg time_to_complete: {np.mean([r.time_to_complete_sec for r in success_results]):.1f} s")
        print(f"Avg time_to_handover: {np.mean([r.time_to_handover_sec for r in success_results]):.1f} s")
        print(f"Peak wheel momentum (max): {max(r.peak_wheel_momentum_nms for r in results):.6f} Nms")

    print(f"\nSaved results to {args.output}")
