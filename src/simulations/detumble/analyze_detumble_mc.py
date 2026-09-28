#!/usr/bin/env python3
"""Analyze LEOP Detumble Monte Carlo results.

Produces:
- 2D success-probability heatmap over (rate0_mag, accel0_mag) bins
- Logistic regression boundary (50%/95% success contours)
- time_to_complete_sec vs rate0_mag scatter with trend
- Per-axis breakdown given inertia_tensor = (0.016, 0.016, 0.004)
- Tip-off margin vs fitted contours
"""

from __future__ import annotations

# Support direct execution and qualified simulations.* imports.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


import argparse
from simulations.helper.serialization import load_trusted_pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from simulations.helper.antara_models import MissionConfig


def logistic_2d(x, a, b, c, d):
    """2D logistic function: success ~ 1 / (1 + exp(-(a*rate + b*accel + c*rate*accel + d)))"""
    rate, accel = x
    return 1.0 / (1.0 + np.exp(-(a * rate + b * accel + c * rate * accel + d)))


def load_results(path: Path) -> list:
    """Load pickle results."""
    with open(path, 'rb') as f:
        return load_trusted_pickle(f)


def create_dataframe(results: list) -> pd.DataFrame:
    """Convert results to DataFrame for analysis."""
    rows = []
    for r in results:
        rows.append({
            'rate0_x': r.rate0_deg_sec[0],
            'rate0_y': r.rate0_deg_sec[1],
            'rate0_z': r.rate0_deg_sec[2],
            'rate0_mag': r.rate0_mag,
            'accel0_x': r.accel0_deg_sec2[0],
            'accel0_y': r.accel0_deg_sec2[1],
            'accel0_z': r.accel0_deg_sec2[2],
            'accel0_mag': r.accel0_mag,
            'success': r.success,
            'hit_rate_cap': r.hit_rate_cap,
            'time_to_handover_sec': r.time_to_handover_sec,
            'time_to_complete_sec': r.time_to_complete_sec,
            'peak_wheel_momentum_nms': r.peak_wheel_momentum_nms,
            'failure_reason': getattr(r, 'failure_reason', ''),
        })
    return pd.DataFrame(rows)


def plot_success_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create 2D success-probability heatmap over (rate0_mag, accel0_mag)."""
    # Bin the data
    rate_bins = np.linspace(0, 60, 31)
    accel_bins = np.linspace(0, 1.0, 21)

    # Compute success rate per bin
    df_binned = df.copy()
    df_binned['rate_bin'] = pd.cut(df_binned['rate0_mag'], rate_bins)
    df_binned['accel_bin'] = pd.cut(df_binned['accel0_mag'], accel_bins)

    success_rates = df_binned.groupby(['rate_bin', 'accel_bin'], observed=False)['success'].agg(['mean', 'count']).reset_index()

    # Create meshgrid for pcolormesh
    rate_centers = (rate_bins[:-1] + rate_bins[1:]) / 2
    accel_centers = (accel_bins[:-1] + accel_bins[1:]) / 2
    success_grid = np.full((len(accel_centers), len(rate_centers)), np.nan)

    for _, row in success_rates.iterrows():
        rate_idx = np.searchsorted(rate_centers, row['rate_bin'].mid)
        accel_idx = np.searchsorted(accel_centers, row['accel_bin'].mid)
        if 0 <= rate_idx < len(rate_centers) and 0 <= accel_idx < len(accel_centers):
            success_grid[accel_idx, rate_idx] = row['mean']

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.pcolormesh(rate_bins, accel_bins, success_grid, cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
    plt.colorbar(im, ax=ax, label='Success Probability')
    ax.set_xlabel('Initial Rate Magnitude (deg/s)')
    ax.set_ylabel('Initial Accel Magnitude (deg/s²)')
    ax.set_title('Detumble Success Probability Heatmap')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.0)

    # Overlay current config tip-off
    config = MissionConfig()
    tipoff_rate = float(np.linalg.norm([
        config.initial_body_rate_deg_sec_x,
        config.initial_body_rate_deg_sec_y,
        config.initial_body_rate_deg_sec_z,
    ]))
    tipoff_accel = float(np.linalg.norm([
        config.initial_body_accel_deg_sec2_x,
        config.initial_body_accel_deg_sec2_y,
        config.initial_body_accel_deg_sec2_z,
    ]))
    ax.plot(tipoff_rate, tipoff_accel, 'w*', markersize=15, markeredgecolor='k', label=f'Current config ({tipoff_rate:.1f} deg/s, {tipoff_accel:.3f} deg/s²)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'success_heatmap.png', dpi=300)
    plt.close()
    print(f"Saved success heatmap to {output_dir / 'success_heatmap.png'}")


def fit_logistic_regression(df: pd.DataFrame) -> tuple:
    """Fit 2D logistic regression to success data."""
    if df['accel0_mag'].nunique() < 2 or df['success'].nunique() < 2:
        print("Skipping 2D fit: acceleration or outcome has no variation.")
        return None, None
    # Prepare data
    X = np.vstack([df['rate0_mag'].values, df['accel0_mag'].values])
    y = df['success'].astype(float).values

    # Initial guess: a, b, c, d
    p0 = [-0.1, -5.0, 0.0, 2.0]

    try:
        popt, pcov = curve_fit(logistic_2d, X, y, p0=p0, maxfev=5000)
        return popt, pcov
    except Exception as e:
        print(f"Logistic fit failed: {e}")
        return None, None


def plot_contours(df: pd.DataFrame, popt: np.ndarray, output_dir: Path):
    """Plot 50% and 95% success contours from logistic fit."""
    if popt is None:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create contour grid
    rate_grid = np.linspace(0, 60, 200)
    accel_grid = np.linspace(0, 1.0, 200)
    R, A = np.meshgrid(rate_grid, accel_grid)
    P = logistic_2d((R, A), *popt)

    # Scatter raw data
    colors = np.where(df['success'], 'green', 'red')
    ax.scatter(df['rate0_mag'], df['accel0_mag'], c=colors, alpha=0.3, s=10)

    # Contours
    ax.contour(R, A, P, levels=[0.5], colors='blue', linewidths=2.5, label='50% contour')
    ax.contour(R, A, P, levels=[0.95], colors='darkblue', linewidths=2.5, linestyles='--', label='95% contour')

    # Current config
    config = MissionConfig()
    tipoff_rate = float(np.linalg.norm([
        config.initial_body_rate_deg_sec_x,
        config.initial_body_rate_deg_sec_y,
        config.initial_body_rate_deg_sec_z,
    ]))
    tipoff_accel = float(np.linalg.norm([
        config.initial_body_accel_deg_sec2_x,
        config.initial_body_accel_deg_sec2_y,
        config.initial_body_accel_deg_sec2_z,
    ]))
    ax.plot(tipoff_rate, tipoff_accel, 'w*', markersize=18, markeredgecolor='k', markeredgewidth=2, label=f'Current config ({tipoff_rate:.1f} deg/s)')

    ax.set_xlabel('Initial Rate Magnitude (deg/s)')
    ax.set_ylabel('Initial Accel Magnitude (deg/s²)')
    ax.set_title('Logistic Regression Success Contours')
    ax.legend()
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    # Compute margin at median and 90th percentile accel
    for pct in [50, 90]:
        accel_pct = np.percentile(df['accel0_mag'], pct)
        # Find rate where P(success) = 0.5 at this accel
        # Solve: a*rate + b*accel + c*rate*accel + d = 0
        a, b, c, d = popt
        rate_50 = -(b * accel_pct + d) / (a + c * accel_pct)
        margin = rate_50 - tipoff_rate
        print(f"  {pct}th percentile accel ({accel_pct:.3f} deg/s²): 50% success rate at {rate_50:.1f} deg/s, margin = {margin:.1f} deg/s")

    plt.tight_layout()
    plt.savefig(output_dir / 'success_contours.png', dpi=300)
    plt.close()
    print(f"Saved contours to {output_dir / 'success_contours.png'}")


def plot_time_to_complete(df: pd.DataFrame, output_dir: Path):
    """Scatter plot of time_to_complete_sec vs rate0_mag for successes."""
    success_df = df[df['success']].copy()
    if len(success_df) < 2 or success_df['rate0_mag'].nunique() < 2:
        print("Skipping completion trend: fewer than two distinct successful rates.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter with trend
    ax1.scatter(success_df['rate0_mag'], success_df['time_to_complete_sec'], alpha=0.5, s=10)
    # Trend line
    z = np.polyfit(success_df['rate0_mag'], success_df['time_to_complete_sec'], 1)
    p = np.poly1d(z)
    rate_range = np.linspace(0, 60, 100)
    ax1.plot(rate_range, p(rate_range), 'r-', linewidth=2, label=f'Trend: {z[0]:.2f}*rate + {z[1]:.1f}')
    ax1.set_xlabel('Initial Rate Magnitude (deg/s)')
    ax1.set_ylabel('Time to Complete (s)')
    ax1.set_title('Time to Complete vs Initial Rate (Successes)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Handover time
    ax2.scatter(success_df['rate0_mag'], success_df['time_to_handover_sec'], alpha=0.5, s=10, color='orange')
    z2 = np.polyfit(success_df['rate0_mag'], success_df['time_to_handover_sec'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(rate_range, p2(rate_range), 'r-', linewidth=2, label=f'Trend: {z2[0]:.2f}*rate + {z2[1]:.1f}')
    ax2.set_xlabel('Initial Rate Magnitude (deg/s)')
    ax2.set_ylabel('Time to Handover (s)')
    ax2.set_title('Time to BDOT->WHEEL_DAMP Handover vs Initial Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'time_to_complete.png', dpi=300)
    plt.close()
    print(f"Saved time plot to {output_dir / 'time_to_complete.png'}")


def plot_per_axis_breakdown(df: pd.DataFrame, output_dir: Path):
    """Per-axis breakdown given inertia_tensor = (0.016, 0.016, 0.004)."""
    success_df = df[df['success']].copy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)

    axes_list = [
        ('rate0_x', 'rate0_y', 'rate0_z'),
        ('accel0_x', 'accel0_y', 'accel0_z'),
    ]
    titles = ['Initial Rate (deg/s)', 'Initial Accel (deg/s²)']

    for row, (ax_row, cols) in enumerate(zip(axes, axes_list)):
        for col_idx, col in enumerate(cols):
            ax = ax_row[col_idx]
            success_vals = success_df[col].values
            fail_vals = df[~df['success']][col].values
            if len(success_vals):
                ax.hist(success_vals, bins=30, alpha=0.7, label='Success', color='green', density=True)
            if len(fail_vals):
                ax.hist(fail_vals, bins=30, alpha=0.7, label='Failure', color='red', density=True)
            ax.set_xlabel(titles[row])
            ax.set_ylabel('Density')
            ax.set_title(f'{col}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # Add inertia tensor note
    fig.suptitle('Per-Axis Breakdown (Inertia: Ixx=0.016, Iyy=0.016, Izz=0.004 kg·m²)', fontsize=12)

    plt.savefig(output_dir / 'per_axis_breakdown.png', dpi=300)
    plt.close()
    print(f"Saved per-axis breakdown to {output_dir / 'per_axis_breakdown.png'}")


def plot_wheel_momentum(df: pd.DataFrame, output_dir: Path):
    """Plot peak wheel momentum vs rate/accel."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(df['rate0_mag'], df['accel0_mag'], c=df['peak_wheel_momentum_nms'],
                         cmap='viridis', s=30, alpha=0.7, edgecolors='none')
    plt.colorbar(scatter, ax=ax, label='Peak Wheel Momentum (Nms)')
    ax.set_xlabel('Initial Rate Magnitude (deg/s)')
    ax.set_ylabel('Initial Accel Magnitude (deg/s²)')
    ax.set_title('Peak Wheel Momentum')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'peak_wheel_momentum.png', dpi=300)
    plt.close()
    print(f"Saved wheel momentum to {output_dir / 'peak_wheel_momentum.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze LEOP Detumble MC results")
    parser.add_argument('input', type=Path, help='Input pickle file from leop_detumble_mc.py')
    parser.add_argument('--output-dir', type=Path, default=Path('detumble_analysis'),
                        help='Output directory for plots and CSV')
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    df = create_dataframe(results)

    # Save summary CSV
    summary = df.copy()
    summary['success'] = summary['success'].astype(int)
    summary.to_csv(output_dir / 'detumble_mc_summary.csv', index=False)
    print(f"Saved summary CSV to {output_dir / 'detumble_mc_summary.csv'}")

    # Overall stats
    total = len(df)
    success = df['success'].sum()
    print(f"\n=== Overall Stats ===")
    print(f"Total trials: {total}")
    print(f"Successful: {success} ({100.0*success/total:.1f}%)")
    if success > 0:
        success_df = df[df['success']]
        print(f"  Avg time_to_complete: {success_df['time_to_complete_sec'].mean():.1f} s")
        print(f"  Avg time_to_handover: {success_df['time_to_handover_sec'].mean():.1f} s")
        print(f"  Max peak wheel momentum: {df['peak_wheel_momentum_nms'].max():.6f} Nms")

    # Current config tip-off
    config = MissionConfig()
    tipoff_rate = float(np.linalg.norm([
        config.initial_body_rate_deg_sec_x,
        config.initial_body_rate_deg_sec_y,
        config.initial_body_rate_deg_sec_z,
    ]))
    print(f"\nCurrent config tip-off rate: {tipoff_rate:.1f} deg/s (acceleration is torque-derived)")

    # Generate all plots
    print("\nGenerating plots...")
    if df['accel0_mag'].eq(0).all():
        print("Rate-only experiment: acceleration is derived from torque, not swept.")
        bins = pd.cut(df['rate0_mag'], np.linspace(0, 60, 31), include_lowest=True)
        rates = df.groupby(bins, observed=True)['success'].mean()
        fig, ax = plt.subplots()
        ax.plot([interval.mid for interval in rates.index], rates.to_numpy(), 'o-')
        ax.set(xlabel='Initial rate magnitude (deg/s)', ylabel='Success fraction', ylim=(0, 1))
        fig.savefig(output_dir / 'success_by_rate.png', dpi=150)
        plt.close(fig)
    else:
        plot_success_heatmap(df, output_dir)
    popt, pcov = fit_logistic_regression(df)
    if popt is not None:
        plot_contours(df, popt, output_dir)
    plot_time_to_complete(df, output_dir)
    plot_per_axis_breakdown(df, output_dir)
    plot_wheel_momentum(df, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
