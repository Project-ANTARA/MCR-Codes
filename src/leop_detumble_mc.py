#!/usr/bin/env python3
"""LEOP Detumble Monte Carlo Capability Envelope.

Runs synthetic detumble trials using only ADCSModel and MissionConfig
(no Orekit, no full simulator). Measures capability envelope for
tip-off rate/accel combinations.
"""

from __future__ import annotations

import math
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Import from antara_models (Orekit-free)
from antara_models import ADCSModel, MissionConfig, _as_flat_vector


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


# Expose rate cap from ADCSModel._integrate_body_rates
RATE_CAP_DEG_SEC = 120.0  # max(120.0, 10.0 * config.max_slew_rate_deg_sec)


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
) -> DetumbleTrialResult:
    """Run a single detumble trial with given initial conditions.

    Args:
        rate0: Initial body rates [deg/s] (x, y, z)
        accel0: Initial body accelerations [deg/s^2] (x, y, z)
        config: MissionConfig for ADCSModel
        dt_sec: Timestep for detumble loop
        timeout_sec: Maximum simulation time before giving up

    Returns:
        DetumbleTrialResult with trial metrics
    """
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

    # B-field magnitude for synthetic model
    b_mag = config.earth_B_field_T

    # Orbital period from config (approximate for 600km SSO)
    orbital_period_sec = 97.5 * 60.0

    t_sec = 0.0
    while t_sec < timeout_sec:
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
        adcs.step_detumble(b_body, dt_sec)

        t_sec += dt_sec

    success = adcs.detumble_complete and t_sec < timeout_sec
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
    )


def run_monte_carlo(
    n_trials: int,
    config: MissionConfig,
    dt_sec: float = 0.5,
    timeout_sec: Optional[float] = None,
    seed: Optional[int] = None,
    n_workers: Optional[int] = None,
) -> list[DetumbleTrialResult]:
    """Run Monte Carlo sweep of detumble trials.

    Args:
        n_trials: Number of trials to run
        config: MissionConfig for ADCSModel
        dt_sec: Timestep for detumble loop
        timeout_sec: Maximum simulation time (default: 3x baseline)
        seed: RNG seed
        n_workers: Number of parallel workers (default: CPU count)

    Returns:
        List of DetumbleTrialResult
    """
    if timeout_sec is None:
        # Will be set after baseline trial
        pass

    rng = np.random.default_rng(seed)

    def worker(args):
        trial_idx, rate0, accel0 = args
        return run_single_trial(rate0, accel0, config, dt_sec, timeout_sec)

    # Sample rate0 magnitude ~Uniform(0, 60) deg/s, direction uniform on sphere
    # Sample accel0 magnitude ~Uniform(0, 1.0) deg/s^2, direction uniform on sphere
    # (Assumption: accel bound pending real deployer tip-off data)
    tasks = []
    for i in range(n_trials):
        rate0_mag = rng.uniform(0.0, 60.0)
        accel0_mag = rng.uniform(0.0, 1.0)

        # Uniform direction on sphere
        phi = rng.uniform(0.0, 2.0 * math.pi)
        costheta = rng.uniform(-1.0, 1.0)
        theta = math.acos(costheta)
        rate0_dir = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=float)
        accel0_dir = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=float)

        rate0 = rate0_dir * rate0_mag
        accel0 = accel0_dir * accel0_mag
        tasks.append((i, rate0, accel0))

    # Run baseline trial first to determine timeout
    print(f"Running baseline trial at default tip-off rates...")
    baseline_rate0 = np.array([
        config.initial_body_rate_deg_sec_x,
        config.initial_body_rate_deg_sec_y,
        config.initial_body_rate_deg_sec_z,
    ], dtype=float)
    baseline_accel0 = np.array([
        config.initial_body_accel_deg_sec2_x,
        config.initial_body_accel_deg_sec2_y,
        config.initial_body_accel_deg_sec2_z,
    ], dtype=float)

    baseline_result = run_single_trial(
        baseline_rate0, baseline_accel0, config, dt_sec, timeout_sec or 10000.0
    )
    baseline_time = baseline_result.time_to_complete_sec
    if timeout_sec is None:
        timeout_sec = 3.0 * max(baseline_time, 1000.0)
        print(f"Baseline time_to_complete = {baseline_time:.1f} s, timeout = {timeout_sec:.1f} s")

    # Run remaining trials
    results = [baseline_result]
    remaining_tasks = tasks[1:] if len(tasks) > 1 else []

    if n_workers == 1:
        for task in remaining_tasks:
            results.append(worker(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(worker, task): task for task in remaining_tasks}
            for future in as_completed(futures):
                results.append(future.result())

    return results


def worker(task):
    """Worker function for ProcessPoolExecutor."""
    return run_single_trial(task[1], task[2], task[3], task[4], task[5])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LEOP Detumble Monte Carlo")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of MC trials")
    parser.add_argument("--dt-sec", type=float, default=0.5, help="Timestep (s)")
    parser.add_argument("--timeout-sec", type=float, default=None, help="Timeout (s), default 3x baseline")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--workers", type=int, default=None, help="Parallel workers")
    parser.add_argument("--output", type=Path, default=Path("detumble_mc_results.pkl"), help="Output pickle path")

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
    )

    # Save results
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)

    # Quick summary
    success_count = sum(1 for r in results if r.success)
    print(f"\n=== Results ===")
    print(f"Total trials: {len(results)}")
    print(f"Successful: {success_count} ({100.0*success_count/len(results):.1f}%)")
    if success_count > 0:
        success_results = [r for r in results if r.success]
        print(f"Avg time_to_complete: {np.mean([r.time_to_complete_sec for r in success_results]):.1f} s")
        print(f"Avg time_to_handover: {np.mean([r.time_to_handover_sec for r in success_results]):.1f} s")
        print(f"Peak wheel momentum (max): {max(r.peak_wheel_momentum_nms for r in results):.6f} Nms")
        print(f"Hit rate cap: {sum(1 for r in results if r.hit_rate_cap)} trials")

    print(f"\nSaved results to {args.output}")