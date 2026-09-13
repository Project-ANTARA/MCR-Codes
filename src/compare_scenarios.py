#!/usr/bin/env python3
"""Compare outputs from the three ANTARA Science scenarios.

Loads telemetry, mode transitions, SAA passes, and GS contacts CSVs
from leop_24h, no_leop_24h, and leop_1yr scenarios and produces:
- scenario_comparison.csv with key metrics
- 3-panel comparison figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paths import results_dir


SCENARIOS = ["leop_24h", "no_leop_24h", "leop_1yr"]
RESULTS_BASE = results_dir("science")


def load_scenario_data(scenario: str, results_base: Path = RESULTS_BASE) -> dict:
    """Load all CSVs for a given scenario."""
    base = Path(results_base)
    if not (base / f"{scenario}_ANTARA_Science_telemetry.csv").exists():
        base = base / scenario  # Legacy nested output layout.
    data = {}

    # Telemetry
    telem_path = base / f"{scenario}_ANTARA_Science_telemetry.csv"
    if telem_path.exists():
        data["telemetry"] = pd.read_csv(telem_path)
    else:
        # Try without prefix (legacy)
        telem_path = base / "ANTARA_Science_telemetry.csv"
        if telem_path.exists():
            data["telemetry"] = pd.read_csv(telem_path)
        else:
            data["telemetry"] = pd.DataFrame()

    # Mode transitions
    mode_path = base / f"{scenario}_ANTARA_Science_mode_transitions.csv"
    if mode_path.exists():
        data["mode_transitions"] = pd.read_csv(mode_path)
    else:
        data["mode_transitions"] = pd.DataFrame()

    # SAA passes
    saa_path = base / f"{scenario}_ANTARA_Science_saa_passes.csv"
    if saa_path.exists():
        data["saa_passes"] = pd.read_csv(saa_path)
    else:
        data["saa_passes"] = pd.DataFrame()

    # GS contacts
    gs_path = base / f"{scenario}_ANTARA_Science_gs_contacts.csv"
    if gs_path.exists():
        data["gs_contacts"] = pd.read_csv(gs_path)
    else:
        data["gs_contacts"] = pd.DataFrame()

    return data


def compute_mode_duty_cycle(telemetry: pd.DataFrame, total_sim_hours: float) -> dict:
    """Compute per-mode duty cycle percentages."""
    if telemetry.empty or "mode" not in telemetry.columns:
        return {}

    mode_counts = telemetry["mode"].value_counts()
    total_steps = len(telemetry)
    duty = {}
    for mode in [
        "COLD_AND_DARK", "DETUMBLE", "SAFE", "SLEWING",
        "SUN_POINTING", "DETECTION", "RELAY"
    ]:
        count = mode_counts.get(mode, 0)
        duty[mode] = 100.0 * count / total_steps if total_steps > 0 else 0.0
    return duty


def compute_battery_stats(telemetry: pd.DataFrame) -> dict:
    """Compute min/avg/max battery Wh."""
    if telemetry.empty or "battery_wh" not in telemetry.columns:
        return {"min": np.nan, "avg": np.nan, "max": np.nan}
    return {
        "min": float(telemetry["battery_wh"].min()),
        "avg": float(telemetry["battery_wh"].mean()),
        "max": float(telemetry["battery_wh"].max()),
    }


def compute_saturation_holds(telemetry: pd.DataFrame) -> int:
    """Count saturation-hold rising edges."""
    if telemetry.empty or "saturation_hold_active" not in telemetry.columns:
        return 0
    holds = telemetry["saturation_hold_active"].astype(bool)
    rising = holds & ~holds.shift(1, fill_value=False)
    return int(rising.sum())


def compute_science_metrics(telemetry: pd.DataFrame, mode_transitions: pd.DataFrame) -> dict:
    """Compute total science MB generated and downlinked, downlink efficiency."""
    result = {
        "science_generated_mb": 0.0,
        "science_downlinked_mb": 0.0,
        "downlink_efficiency_pct": 0.0,
    }
    if telemetry.empty:
        return result

    if "science_data_mb" in telemetry.columns:
        result["science_generated_mb"] = float(telemetry["science_data_mb"].sum())
    if "downlinked_mb" in telemetry.columns:
        result["science_downlinked_mb"] = float(telemetry["downlinked_mb"].sum())

    if result["science_generated_mb"] > 0:
        result["downlink_efficiency_pct"] = 100.0 * result["science_downlinked_mb"] / result["science_generated_mb"]

    return result


def compute_eclipse_fraction(telemetry: pd.DataFrame) -> float:
    """Compute eclipse fraction percentage."""
    if telemetry.empty or "in_eclipse" not in telemetry.columns:
        return np.nan
    return 100.0 * float(telemetry["in_eclipse"].mean())


def compute_attitude_error_per_mode(telemetry: pd.DataFrame) -> dict:
    """Compute average attitude_error_deg per mode."""
    if telemetry.empty or "attitude_error_deg" not in telemetry.columns:
        return {}
    return telemetry.groupby("mode")["attitude_error_deg"].mean().to_dict()


def main():
    parser = argparse.ArgumentParser(description="Compare ANTARA Science scenarios")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_BASE,
                        help="Base results directory (default: results/science)")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory for comparison files")
    parser.add_argument("--require-all", action="store_true",
                        help="Fail if any expected scenario telemetry is missing")
    parser.add_argument("--require-full-duration", action="store_true",
                        help="Require complete 24h/24h/8760h preset telemetry")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for scenario in SCENARIOS:
        print(f"\n=== Loading {scenario} ===")
        data = load_scenario_data(scenario, results_dir)

        telemetry = data["telemetry"]
        mode_transitions = data["mode_transitions"]
        saa_passes = data["saa_passes"]
        gs_contacts = data["gs_contacts"]

        if telemetry.empty:
            if args.require_all or args.require_full_duration:
                parser.error(f"Missing telemetry for {scenario} under {results_dir}")
            print(f"  WARNING: No telemetry data found for {scenario}")
            continue

        # Determine simulation duration from telemetry
        if "time_sec" in telemetry.columns:
            total_sim_hours = (telemetry["time_sec"].max() - telemetry["time_sec"].min()) / 3600.0
        else:
            if args.require_full_duration:
                parser.error(f"{scenario}: telemetry has no time_sec column")
            total_sim_hours = 24.0  # fallback

        if args.require_full_duration:
            expected = 8760.0 if scenario == "leop_1yr" else 24.0
            times = telemetry["time_sec"].to_numpy(dtype=float)
            if (not np.isfinite(times).all() or len(times) < 2
                    or not (np.diff(times) > 0).all()
                    or not np.isclose(times[0], 0.0, rtol=0, atol=1e-6)
                    or not np.isclose(times[-1], expected * 3600, rtol=0, atol=1e-6)):
                parser.error(f"{scenario}: expected complete {expected:g}h telemetry, got {total_sim_hours:g}h")

        print(f"  Simulation duration: {total_sim_hours:.2f} hours")
        print(f"  Telemetry rows: {len(telemetry)}")

        # Compute metrics
        metrics = {"scenario": scenario, "duration_hours": total_sim_hours}

        # Mode duty cycles
        duty = compute_mode_duty_cycle(telemetry, total_sim_hours)
        for mode, pct in duty.items():
            metrics[f"duty_{mode}_pct"] = pct

        # Battery stats
        batt = compute_battery_stats(telemetry)
        metrics.update({f"batt_{k}": v for k, v in batt.items()})

        # Saturation holds
        metrics["saturation_hold_count"] = compute_saturation_holds(telemetry)

        # Science metrics
        sci = compute_science_metrics(telemetry, pd.DataFrame())
        metrics.update(sci)

        # Eclipse fraction
        metrics["eclipse_fraction_pct"] = compute_eclipse_fraction(telemetry)

        # Attitude error per mode
        att_err = compute_attitude_error_per_mode(telemetry)
        for mode, err in att_err.items():
            metrics[f"att_err_{mode}_deg"] = err

        all_metrics.append(metrics)
        print(f"  Computed metrics for {scenario}")

    if not all_metrics:
        print("ERROR: No valid scenario data found")
        sys.exit(1)

    # Create comparison DataFrame
    df = pd.DataFrame(all_metrics)
    csv_path = output_dir / "scenario_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV: {csv_path}")

    # Create 3-panel comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Panel 1: Battery Wh over time (or min/avg/max bar)
    ax1 = axes[0]
    x = np.arange(len(all_metrics))
    width = 0.25
    ax1.bar(x - width, [m["batt_min"] for m in all_metrics], width, label="Min", color="red", alpha=0.7)
    ax1.bar(x, [m["batt_avg"] for m in all_metrics], width, label="Avg", color="orange", alpha=0.7)
    ax1.bar(x + width, [m["batt_max"] for m in all_metrics], width, label="Max", color="green", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m["scenario"] for m in all_metrics])
    ax1.set_ylabel("Battery (Wh)")
    ax1.set_title("Battery Min / Avg / Max")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Mode duty cycle stacked bar
    ax2 = axes[1]
    modes = ["COLD_AND_DARK", "DETUMBLE", "SAFE", "SLEWING", "SUN_POINTING", "DETECTION", "RELAY"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(modes)))
    bottom = np.zeros(len(all_metrics))
    for i, mode in enumerate(modes):
        vals = [m.get(f"duty_{mode}_pct", 0) for m in all_metrics]
        ax2.bar(x, vals, bottom=bottom, label=mode, color=colors[i], alpha=0.8)
        bottom += np.array(vals)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m["scenario"] for m in all_metrics])
    ax2.set_ylabel("Duty Cycle (%)")
    ax2.set_title("Mode Duty Cycle")
    ax2.legend(loc="upper right", fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Science & Downlink
    ax3 = axes[2]
    ax3.bar(x - 0.2, [m["science_generated_mb"] for m in all_metrics], 0.4, label="Science Generated (MB)", color="red", alpha=0.7)
    ax3.bar(x + 0.2, [m["science_downlinked_mb"] for m in all_metrics], 0.4, label="Science Downlinked (MB)", color="blue", alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m["scenario"] for m in all_metrics])
    ax3.set_ylabel("Data (MB)")
    ax3.set_title("Science Generated vs Downlinked")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.suptitle("ANTARA Science Scenario Comparison", fontsize=14)
    fig_path = output_dir / "scenario_comparison.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved comparison figure: {fig_path}")

    print("\n=== Comparison Complete ===")
    print(f"CSV: {csv_path}")
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()
