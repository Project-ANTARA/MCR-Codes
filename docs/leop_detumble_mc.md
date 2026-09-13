# Detumble Monte Carlo and Analysis

Source: [runner](../src/leop_detumble_mc.py) and [analysis](../src/analyze_detumble_mc.py). Reviewed 2026-09-12.

## Run locally

Run from the repository root. The runner uses NumPy and shared `ADCSModel`/`MissionConfig`; it does not require Orekit, SpacePy, or a JVM. The analysis additionally needs pandas, Matplotlib, and SciPy, available in the documented `orekit_env` setup.

```bash
conda activate orekit_env
# Short dispatch check, not a scientific capability study
python src/leop_detumble_mc.py --n-trials 8 --workers 2 --timeout-sec 2 --output /tmp/detumble_dispatch.pkl

# Normal study; choose a new filename to preserve previous results
python src/leop_detumble_mc.py --n-trials 5000 --workers 4 --output detumble_mc_results.pkl
python src/analyze_detumble_mc.py detumble_mc_results.pkl --output-dir detumble_analysis
```

## Options and terminal information

| Option | Default | Meaning |
| --- | --- | --- |
| `--n-trials` | 100 | Total records, including the baseline; minimum 1 |
| `--workers` | CPU count | Process-pool size; 1 selects serial execution |
| `--dt-sec` | 0.5 | Simulated seconds per physics step |
| `--timeout-sec` | Derived | Normally `3 * max(baseline completion, 1000)` simulated seconds |
| `--seed` | 42 | Reproducible sampling |
| `--output` | `detumble_mc_results.pkl` | Pickle destination; existing file is overwritten |
| `--no-progress` | Off | Hide progress, retaining setup and summary output |

Progress goes to stderr. Interactive terminals show a refreshed bar with completed/total trials, percentage, successes (`OK`), timeouts, rate-cap hits (`Cap`), elapsed wall time, and approximate ETA. Baseline and serial runs also show simulated seconds, `BDOT`/`WHEEL_DAMP` phase, and body-rate magnitude. Parallel workers do not print; the parent refreshes while awaiting results. Redirected logs get plain-text updates at roughly five-second intervals plus completion. ETA is unavailable until a random trial finishes and may vary substantially with trial duration. Rate-cap hits can overlap success or timeout counts.

```bash
python src/leop_detumble_mc.py --n-trials 100 --workers 4 --no-progress --output quiet_trials.pkl
```

## Results and limits

Results retain seeded trial order, regardless of process completion order. Module-level workers and importable `DetumbleTrialResult` classes support spawn and fresh-process pickle loading. Load only trusted pickles; pickle deserialization can execute code. Output parent directories must already exist.

The analysis writes `detumble_mc_summary.csv` and success heatmap/contour, completion-time, per-axis, and wheel-momentum plots. A fitted contour may be unavailable when samples do not support a fit. Do not treat a tiny or deliberately time-limited dispatch test as a hardware capability estimate.

Physics limitations remain: synthetic field geometry rather than propagated IGRF, sampled acceleration parallel to the sampled rate direction, and no full mission/EPS/FSM coupling. The baseline has a 10,000-second cap when no timeout is supplied. If it fails, the current derived timeout can become infinite; use an explicit finite timeout for such configurations. Progress does not change trial physics or fix these modeling limitations.

## Verification

```bash
PYTHONPATH=src python -m pytest tests/test_detumble_multiprocessing.py -v
```

Tests cover serial/spawn equivalence, portable CLI pickles, progress counters/ETA, narrow terminals, redirected output, and quiet single-baseline runs. A normal eight-trial progress run completed with a 1009-second baseline, 3027-second trial timeout, and three successful trials. See [historical repair evidence](SIMULATION_RUNTIME_FIX_REPORT.md) for earlier campaign scope.
