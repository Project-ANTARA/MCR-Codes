# Scenario Presets and Output Comparison

Source: [presets](../src/scenarios.py), [Science CLI](../src/ANTARA_Science_Sim.py), and [comparison tool](../src/compare_scenarios.py). Reviewed 2026-09-12.

## Presets

`build_scenario(name)` returns `(MissionConfig, skip_leop)`. The presets are `leop_24h` (24 hours, startup), `no_leop_24h` (24 hours, skip startup), and `leop_1yr` (8760 hours, startup). Each sets its matching output tag and a six-hour telemetry flush interval. Configuration imports come from `antara_models`, not the running simulator.

Explicit CLI values override presets, which override global defaults. Supplying `--step-sec` selects fixed stepping unless `--coarse-step-sec` is also supplied. See the [Science guide](ANTARA_Science_Sim.md) for all switches.

```bash
python src/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
```

Science writes flat paths such as `results/science/leop_24h_ANTARA_Science_telemetry.csv`. Reusing a tag overwrites files. Configure a separate `output_tag` programmatically for independent experiments; there is no output-tag CLI switch.

## Verify actual duration

```python
import pandas as pd

df = pd.read_csv("results/science/leop_24h_ANTARA_Science_telemetry.csv", low_memory=False)
assert not df.empty and "mode" in df
print("Rows:", len(df), "Modes:", df["mode"].unique())
print("Hours:", (df.time_sec.iloc[-1] - df.time_sec.iloc[0]) / 3600)
print("Intervals:", sorted(df.time_sec.diff().dropna().unique()))
```

The one-hour fixed-step checks produced 1,801 rows at two-second intervals. The later 60-hour annual-preset check produced 17,758 rows at two-/20-second intervals. Neither establishes a successful full-year run. Checkpoint switches exist, but the current `run()` resets its loop clock to zero; exact resume chronology remains unverified.

## Comparison tool limitations

As of 2026-09-13, `compare_scenarios.py` reads Science's flat prefixed CSVs and honors `--results-dir`. Legacy nested directories and unprefixed nested telemetry remain supported.

It writes `scenario_comparison.csv` and `scenario_comparison.png` to `--output-dir`. Use `--require-all` to reject missing scenarios; nightly additionally uses `--require-full-duration` to require complete, monotonic 24h/24h/8760h telemetry starting at zero. Without the latter flag, bounded studies can still be compared. Sample-count duty cycles remain biased for multirate telemetry; derive duration-weighted metrics independently before making mission claims.

For current recovery evidence, see the [runtime report](SIMULATION_RUNTIME_FIX_REPORT.md).
