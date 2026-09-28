#!/usr/bin/env python3
"""Compare outputs from the three ANTARA Science scenarios.

Streams scalar telemetry from leop_24h, no_leop_24h, and leop_1yr
scenarios and produces:
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
from telemetry_metrics import (interval_weights, weighted_mean, validated_modes,
                               boolean_measurements, SCIENCE_MODES)


SCENARIOS = ["leop_24h", "no_leop_24h", "leop_1yr"]
RESULTS_BASE = results_dir("science")
COMPARISON_COLUMNS = frozenset({
    "time_sec", "state_time_sec", "interval_duration_sec", "mode", "battery_wh",
    "saturation_hold_active", "science_data_mb", "downlinked_mb", "in_eclipse",
    "attitude_error_deg",
})


def scenario_telemetry_path(scenario, results_base):
    base = Path(results_base)
    for path in (base / f"{scenario}_ANTARA_Science_telemetry.csv",
                 base / scenario / f"{scenario}_ANTARA_Science_telemetry.csv",
                 base / scenario / "ANTARA_Science_telemetry.csv"):
        if path.exists():
            return path
    return None


def _comparison_intervals(path, chunksize):
    """One-row lookahead preserves legacy duration inference at chunk edges."""
    pending = None
    with pd.read_csv(path, chunksize=chunksize,
                     usecols=lambda name: name in COMPARISON_COLUMNS) as reader:
        for chunk in reader:
            if chunk.empty:
                continue
            frame = pd.concat([pending, chunk], ignore_index=True) if pending is not None else chunk.reset_index(drop=True)
            if "time_sec" not in frame:
                raise ValueError("Comparison requires time_sec; duration cannot be guessed")
            weights = interval_weights(frame)
            validated_modes(frame)
            pending = frame.iloc[-1:].copy()
            if len(frame) > 1:
                completed = frame.iloc[:-1].copy()
                completed["interval_duration_sec"] = weights.iloc[:-1]
                yield completed
    if pending is not None:
        pending["interval_duration_sec"] = interval_weights(pending)
        if pending["interval_duration_sec"].iloc[0] != 0.:
            raise ValueError("Comparison requires a zero-duration terminal observation; telemetry is unfinished")
        yield pending


def stream_scenario_metrics(path, scenario, *, chunksize=10000, require_full_duration=False):
    """Compute comparison scalars in O(chunksize) memory, without event tables."""
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer")
    mode_seconds = dict.fromkeys(SCIENCE_MODES, 0.)
    means, mean_seconds = {}, {}
    byte_totals = {}
    battery_min, battery_max = np.inf, -np.inf
    elapsed, rows, holds_count = 0., 0, 0
    first_time = last_time = final_duration = None
    previous_hold = False

    def add_mean(key, values, weights):
        values = pd.to_numeric(values, errors="raise").astype(float)
        mean = weighted_mean(values, weights)  # Also validates infinities.
        seconds = float(weights[values.notna()].sum())
        old_seconds = mean_seconds.get(key, 0.)
        mean_seconds[key] = old_seconds + seconds
        if seconds:
            means[key] = weighted_mean([means.get(key, 0.), mean], pd.Series([old_seconds, seconds]))

    for frame in _comparison_intervals(path, chunksize):
        weights = frame["interval_duration_sec"]
        if first_time is None:
            first_time = float(frame.time_sec.iloc[0])
        last_time = float(frame.time_sec.iloc[-1])
        final_duration = float(weights.iloc[-1])
        rows += len(frame)
        elapsed += float(weights.sum())
        for mode, seconds in weights.groupby(frame["mode"]).sum().items():
            mode_seconds[mode] += float(seconds)
        if "battery_wh" in frame:
            values = pd.to_numeric(frame.battery_wh, errors="raise").astype(float)
            add_mean("batt_avg", values, weights)
            if values.notna().any():
                battery_min = min(battery_min, float(values.min()))
                battery_max = max(battery_max, float(values.max()))
        if "in_eclipse" in frame:
            add_mean("eclipse", boolean_measurements(frame.in_eclipse, "in_eclipse"), weights)
        if "attitude_error_deg" in frame:
            for mode, group in frame.groupby("mode"):
                add_mean(f"att_err_{mode}_deg", group.attitude_error_deg, weights.loc[group.index])
        if "saturation_hold_active" in frame:
            holds = boolean_measurements(frame.saturation_hold_active, "saturation_hold_active").astype(bool)
            holds_count += int((holds & ~holds.shift(1, fill_value=previous_hold)).sum())
            previous_hold = bool(holds.iloc[-1])
        counters = compute_science_metrics(frame, pd.DataFrame())
        for key in ("science_generated_mb", "science_downlinked_mb"):
            if np.isfinite(counters[key]):
                byte_totals[key] = byte_totals.get(key, 0.) + counters[key]
                if not np.isfinite(byte_totals[key]):
                    raise FloatingPointError(f"Byte total overflow: {key}")

    if not rows:
        raise ValueError("Empty telemetry")
    if require_full_duration:
        expected = 8760. if scenario == "leop_1yr" else 24.
        if (rows < 2 or final_duration != 0. or
                not np.isclose(first_time, 0., rtol=0., atol=1e-6) or
                not np.isclose(last_time, expected * 3600, rtol=0., atol=1e-6)):
            raise ValueError(f"expected complete {expected:g}h telemetry")
    metrics = {"scenario": scenario, "duration_hours": (last_time - first_time) / 3600,
               "batt_min": battery_min if battery_min != np.inf else np.nan,
               "batt_max": battery_max if battery_max != -np.inf else np.nan,
               "batt_avg": means.get("batt_avg", np.nan),
               "eclipse_fraction_pct": 100 * means.get("eclipse", np.nan),
               "saturation_hold_count": holds_count}
    metrics.update({f"duty_{mode}_pct": 100 * seconds / elapsed if elapsed else 0.
                    for mode, seconds in mode_seconds.items()})
    metrics.update({key: means.get(key, np.nan) for key in mean_seconds if key.startswith("att_err_")})
    counter_frame = pd.DataFrame({column: [byte_totals[key]] for column, key in
                                 (("science_data_mb", "science_generated_mb"),
                                  ("downlinked_mb", "science_downlinked_mb")) if key in byte_totals})
    metrics.update(compute_science_metrics(counter_frame, pd.DataFrame()))
    return metrics, rows


def _read_optional_event_csv(path):
    """An absent/headerless event log means no events, not failed telemetry."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_scenario_data(scenario: str, results_base: Path = RESULTS_BASE) -> dict:
    """Legacy full-frame API for small callers; the CLI uses streaming instead."""
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
    data["mode_transitions"] = _read_optional_event_csv(mode_path)

    # SAA passes
    saa_path = base / f"{scenario}_ANTARA_Science_saa_passes.csv"
    data["saa_passes"] = _read_optional_event_csv(saa_path)

    # GS contacts
    gs_path = base / f"{scenario}_ANTARA_Science_gs_contacts.csv"
    data["gs_contacts"] = _read_optional_event_csv(gs_path)

    return data


def compute_mode_duty_cycle(telemetry: pd.DataFrame, total_sim_hours: float) -> dict:
    """Compute per-mode duty cycle percentages."""
    if telemetry.empty:
        return {}

    weights = interval_weights(telemetry)
    mode_counts = weights.groupby(validated_modes(telemetry)).sum()
    total_steps = weights.sum()
    duty = {}
    for mode in SCIENCE_MODES:
        count = mode_counts.get(mode, 0)
        duty[mode] = 100.0 * count / total_steps if total_steps > 0 else 0.0
    return duty


def compute_battery_stats(telemetry: pd.DataFrame) -> dict:
    """Compute min/avg/max battery Wh."""
    if telemetry.empty or "battery_wh" not in telemetry.columns:
        return {"min": np.nan, "avg": np.nan, "max": np.nan}
    return {
        "min": float(telemetry["battery_wh"].min()),
        "avg": weighted_mean(telemetry["battery_wh"], interval_weights(telemetry)),
        "max": float(telemetry["battery_wh"].max()),
    }


def compute_saturation_holds(telemetry: pd.DataFrame) -> int:
    """Count saturation-hold rising edges."""
    if telemetry.empty or "saturation_hold_active" not in telemetry.columns:
        return 0
    holds = boolean_measurements(telemetry["saturation_hold_active"], "saturation_hold_active").astype(bool)
    rising = holds & ~holds.shift(1, fill_value=False)
    return int(rising.sum())


def compute_science_metrics(telemetry: pd.DataFrame, mode_transitions: pd.DataFrame) -> dict:
    """Sum complete byte counters, including zero-duration terminal flushes.

    Missing columns are unavailable, not zero. A present but incomplete/invalid
    counter cannot provide a trustworthy total. No duration weighting applies
    to these interval byte increments.
    """
    result = {
        "science_generated_mb": np.nan,
        "science_downlinked_mb": np.nan,
        "downlink_efficiency_pct": np.nan,
    }
    if telemetry.empty:
        return result

    for column, key in (("science_data_mb", "science_generated_mb"),
                        ("downlinked_mb", "science_downlinked_mb")):
        if column not in telemetry:
            continue
        try:
            values = pd.to_numeric(telemetry[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid byte counter {column}") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid or missing byte counter {column}")
        with np.errstate(over="raise", invalid="raise"):
            result[key] = float(values.sum())

    if result["science_generated_mb"] > 0 and np.isfinite(result["science_downlinked_mb"]):
        result["downlink_efficiency_pct"] = (result["science_downlinked_mb"] / result["science_generated_mb"]) * 100.0
        if not np.isfinite(result["downlink_efficiency_pct"]):
            raise FloatingPointError("Downlink efficiency overflow")

    return result


def compute_eclipse_fraction(telemetry: pd.DataFrame) -> float:
    """Compute eclipse fraction percentage."""
    if telemetry.empty or "in_eclipse" not in telemetry.columns:
        return np.nan
    eclipse = boolean_measurements(telemetry["in_eclipse"], "in_eclipse")
    return 100.0 * weighted_mean(eclipse, interval_weights(telemetry))


def compute_attitude_error_per_mode(telemetry: pd.DataFrame) -> dict:
    """Compute average attitude_error_deg per mode."""
    if telemetry.empty or "attitude_error_deg" not in telemetry.columns:
        return {}
    weights = interval_weights(telemetry)
    return {mode: weighted_mean(group["attitude_error_deg"], weights.loc[group.index])
            for mode, group in telemetry.groupby("mode")}


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
        path = scenario_telemetry_path(scenario, results_dir)
        if path is None:
            if args.require_all or args.require_full_duration:
                parser.error(f"Missing telemetry for {scenario} under {results_dir}")
            print(f"  WARNING: No telemetry data found for {scenario}")
            continue

        try:
            metrics, rows = stream_scenario_metrics(path, scenario, require_full_duration=args.require_full_duration)
        except (ValueError, FloatingPointError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            parser.error(f"{scenario}: invalid telemetry: {exc}")
        print(f"  Simulation duration: {metrics['duration_hours']:.2f} hours")
        print(f"  Telemetry rows: {rows}")

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
