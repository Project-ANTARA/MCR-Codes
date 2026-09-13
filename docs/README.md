# ANTARA Mission Analysis — Source Code Documentation

Maintained module guides for `src/`, reviewed on 2026-09-12. The operational entrypoints, supporting studies, and shared models have different configurations and validation scope; a Science fix does not automatically update another simulator. Guides cover:
- File overview & role
- Dependencies
- Exports/Public API (constants, functions, classes, methods)
- Internal logic walkthrough
- Usage examples
- Notes/TODOs

---

## Table of Contents

### Core Simulation Entry Points

| File | Description |
|------|-------------|
| [`ANTARA_Master.md`](ANTARA_Master.md) | **High-fidelity simulation** — 7-state FSM, numerical Orekit propagation, ADCS/EPS/SAA/GS/payload/comms models, 9-panel PDR plots, comprehensive telemetry |
| [`ANTARA_MCR.md`](ANTARA_MCR.md) | **Monte Carlo RAAN trade** — 365-day × 7 RAANs × 2 MC runs, randomized parameters, 7 output CSVs + statistics |
| [`ANTARA_avg.md`](ANTARA_avg.md) | **Monte Carlo battery sizing** — 365-day × 3 MC runs × battery cases (25 Wh), simplified power/battery model |
| [`ANTARA_conops.md`](ANTARA_conops.md) | **Fast CONOPS timeline** — 30-day Eckstein-Hechler analytical propagation, 60s step, eclipse/SAA/GS log |
| [`ANTARA_Science_Sim.md`](ANTARA_Science_Sim.md) | **Science payload simulation** — ppigrf/IGRF-14, configured proton-flux baseline (live AP9 unavailable), detector physics, dual-GS, CLI |
| [`leop_detumble_mc.md`](leop_detumble_mc.md) | **Detumble Monte Carlo and analysis** — progress/ETA, workers, seeded output, pickle safety, analysis limits |
| [`scenarios.md`](scenarios.md) | **Scenario presets and comparison** — CLI precedence, output layout, current comparison limitations |

### Shared Models

| File | Description |
|------|-------------|
| [`antara_models.md`](antara_models.md) | **Orekit-free core models** — `MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `_as_flat_vector` (zero Java deps, for lightweight MC/validation) |

### Specialized Studies

| File | Description |
|------|-------------|
| [`ANTARA_FINAL_POWERGEN.md`](ANTARA_FINAL_POWERGEN.md) | **RAAN power optimization** — 24h × 25 samples (0° and 360° both included), fixed panel + 2 wings |
| [`ANTARA_drag_finalconfig.md`](ANTARA_drag_finalconfig.md) | **Orbital lifetime (final config)** — Altitude sweep 500–600km, solar max, tumbling factor 1.5 |
| [`ANTARA_drag_study.md`](ANTARA_drag_study.md) | **Orbital lifetime (trade study)** — 4 configurations, altitude sweep, tumbling factor 1.0 |
| [`TradeCase.md`](TradeCase.md) | **Battery trade analysis** — Analytical DoD vs RAAN, SOC simulation, cycle life margin |
| [`ANTARA_pointing_budget.md`](ANTARA_pointing_budget.md) | **GS pointing budget** — 1-day Keplerian, nadir-pointing, off-boresight angle during passes |

### Infrastructure

| File | Description |
|------|-------------|
| [`paths.md`](paths.md) | **Path management & Orekit data init** — Central I/O routing, auto-download support |
| [`antara_models.md`](antara_models.md) | **Orekit-free shared models** — `MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `_as_flat_vector` (numpy only, no JVM) |
| [`ThermalSimulator.md`](ThermalSimulator.md) | **Thermal simulator** — 7-node lumped capacitance, RK4, bounding cases (HOT/COLD), beta targeting, latitude-dependent Earth IR/albedo |

---

## Quick Reference: Simulation Commands

```bash
# Fast verification (seconds)
python src/ANTARA_conops.py

# Full high-fidelity (minutes)
python src/ANTARA_Master.py

# Monte Carlo RAAN trade (hours)
python src/ANTARA_MCR.py

# Battery sizing MC (hours)
python src/ANTARA_avg.py

# Science smoke: scenario structure, explicit one-hour duration and fixed step
python src/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots

# Thermal analysis
python src/ThermalSimulator.py --case HOT --target-beta-deg 90

# Detumble: bounded worker count, terminal progress enabled by default
python src/leop_detumble_mc.py --n-trials 100 --workers 4 --output detumble_mc_results.pkl
python src/analyze_detumble_mc.py detumble_mc_results.pkl --output-dir detumble_analysis

# Trade studies (seconds)
python src/ANTARA_FINAL_POWERGEN.py
python src/ANTARA_drag_finalconfig.py
python src/ANTARA_drag_study.py
python src/TradeCase.py
python src/ANTARA_pointing_budget.py
```

---

## Common Patterns

| Pattern | Description |
|---------|-------------|
| `orekit.initVM()` | Initialize JVM before Java-backed API use; initialization placement varies by entrypoint |
| `init_orekit_data()` | Initializes Orekit data from `orekit-data/` or `orekit-data.zip` |
| `results_path("subdir", "file.csv")` | Output path: `results/subdir/file.csv` (auto-creates dirs) |
| `figures_path("subdir", "plot.png")` | Figure path: `figures/subdir/plot.png` (auto-creates dirs) |
| `MissionConfig` dataclass | Put new configurable numeric parameters in the owning module's config; standalone scripts also use module constants |
| `BODY_AXIS_VECTORS` | Canonical body axes: +X=RAM, +Y=nadir/relay, +Z=sun-panel |
| Quaternion convention | Scalar-first [w,x,y,z], body-to-inertial |
| `from antara_models import ...` | Lightweight shared models without Orekit/JVM; Science still shadows shared EPS/FSM/exception imports with local definitions |

---

## Directory Structure

```
src/
├── ANTARA_Master.py          # Main high-fidelity simulation
├── ANTARA_MCR.py             # Monte Carlo RAAN trade
├── ANTARA_avg.py             # Battery sizing Monte Carlo
├── ANTARA_conops.py          # Fast CONOPS (analytical)
├── ANTARA_Science_Sim.py     # Science payload + proton flux
├── ANTARA_FINAL_POWERGEN.py  # RAAN power optimization
├── ANTARA_drag_finalconfig.py# Lifetime (final config)
├── ANTARA_drag_study.py      # Lifetime (trade study)
├── TradeCase.py              # Battery DoD/cycle life trade
├── ANTARA_pointing_budget.py # GS pointing analysis
├── ThermalSimulator.py       # Thermal lumped capacitance
├── paths.py                  # Shared path utilities
├── leop_detumble_mc.py       # LEOP detumble Monte Carlo
├── analyze_detumble_mc.py    # Detumble MC analysis & plots
├── compare_scenarios.py      # Scenario comparison
├── plot_fsm_diagram.py       # FSM diagram generator
├── fsm_spec.py               # FSM decision specification
├── scenarios.py              # Scenario builder
└── antara_models.py          # Orekit-free shared models

docs/
├── README.md                 # This file
├── ANTARA_Master.md
├── ANTARA_MCR.md
├── ANTARA_avg.md
├── ANTARA_conops.md
├── ANTARA_Science_Sim.md
├── ANTARA_FINAL_POWERGEN.md
├── ANTARA_drag_finalconfig.md
├── ANTARA_drag_study.md
├── TradeCase.md
├── ANTARA_pointing_budget.md
├── paths.md
├── ThermalSimulator.md
├── leop_detumble_mc.md
├── scenarios.md
├── SIMULATION_RUNTIME_FIX_REPORT.md
└── antara_models.md

results/                      # Gitignored — generated CSVs
├── master/
├── mcr/
├── avg/
├── conops/
├── science/
├── thermal/
├── tradecase/
└── pointing/

figures/                      # Gitignored — generated plots
├── master/
├── mcr/
├── science/
├── thermal/
├── tradecase/
└── pointing/
```

---

## Dependencies

Use the [repository setup guide](../README.md#quick-start). Locally activate Conda `orekit_env`; CI uses Micromamba `antara` with Python 3.11. All simulation backends must resolve from the Orekit interpreter.

```bash
conda activate orekit_env
conda install -c conda-forge ppigrf=2.1.0
python -m pip install spacepy==0.7.0
python scripts/verify_physics_dependencies.py
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -v
```

Orekit comes from conda-forge. NumPy, pandas, SciPy, Shapely, and plotting dependencies are listed in the root setup instructions. Science guards plotting imports; some other entrypoints import plotting libraries directly.

`ppigrf` supplies IGRF-14 through `igrf(lon, lat, altitude_km, date)`. Missing ppigrf selects a dipole fallback. SpacePy's IRBEM interface supplies AE8/AP8, not live AP9; ANTARA currently uses its configured AP9 baseline after query failure. This baseline enters primary science flux. Backend verification executes AP8 only and must not be described as an AP9 validation.

## Current verification and reports

Documentation-refresh verification on 2026-09-12: the combined physics/attitude, detumble multiprocessing/progress, and shared FSM command above passed **31 tests**. Local Markdown links and code-fence pairing were checked. Supporting annual/thermal/trade studies and hosted CI were not rerun.

The [physics/attitude repair report](../OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md) records 14 passing targeted tests and both one-hour scenarios producing 1,801 rows at two-second intervals. Numerical geometry uses normalized cross/dot `atan2` with unchanged `1e-6°` alignment tolerance and negative tests. Remote CI execution and full mission fidelity are separate from those local results.

The [runtime repair report](SIMULATION_RUNTIME_FIX_REPORT.md) records the later **60-hour annual-preset run**, including saturation timeout recovery through SAFE and return to SUN_POINTING. The 5° normal tracking limit remains enforced; saturation hold validates frozen attitude instead. The full 8,760-hour campaign has not been verified. The [detumble guide](leop_detumble_mc.md) describes the newest progress display, redirected logging, and regression commands.

Explicit CLI arguments override scenario presets. An explicit `--step-sec` selects fixed stepping unless `--coarse-step-sec` is also supplied. See the [Science guide](ANTARA_Science_Sim.md) for the actual CLI and output names, and [paths guide](paths.md) for root-level Orekit data discovery.

Older root validation and CI investigation reports retain their original captures and are marked historical; do not treat their old warnings as current package-installation guidance.

The [2026-09-13 scheduled verification report](SCHEDULED_VERIFICATION_FIX_REPORT.md) records nightly/weekly concurrency repairs, validation reference corrections, local results, and remaining full-year runtime limits. New validation runs write under `results/validation/`, not the historical root report.

---

For contributor conventions, see [AGENTS.md](../AGENTS.md). For specific subsystem behavior, consult its guide and current source.
