# ANTARA Mission Analysis — Agent Instructions

## Project Overview
Python-based CubeSat mission simulation using Orekit (Java astrodynamics library). Simulates orbital dynamics, power systems, ADCS, thermal, and science operations for the ANTARA 3U CubeSat.

## Key Entry Points
Run from repository root with `python src/<script>.py`:

| Script | Purpose |
|--------|---------|
| `ANTARA_Master.py` | High-fidelity simulation with full FSM, ADCS, EPS, payload models |
| `ANTARA_MCR.py` | Monte Carlo RAAN trade study (365-day, multiple RAANs) |
| `ANTARA_avg.py` | Monte Carlo battery sizing study (25 Wh case) |
| `ANTARA_conops.py` | Fast analytical propagation for CONOPS timeline (Eckstein-Hechler) |
| `ANTARA_Science_Sim.py` | Science payload simulation with proton flux modelling |

## Environment Setup
```bash
# Required: Java 11+ for Orekit VM
# Python deps (install via pip):
pip install orekit numpy pandas matplotlib cartopy shapely
# Optional: pyIGRF, spacepy for enhanced science models
```

## Orekit Data (Required)
Place Orekit data at one of:
- `data/orekit/orekit-data/orekit-data-main/` (extracted)
- `data/orekit-data.zip` (auto-downloaded if missing via `init_orekit_data(download_if_missing=True)`)

Scripts auto-initialize via `from paths import init_orekit_data; init_orekit_data()`.

## Directory Structure
```
src/          Active simulation entrypoints + paths.py
data/         Orekit datasets + reference CSV inputs
results/      Generated CSV/TXT outputs (gitignored)
figures/      Generated plots (gitignored)
notebooks/    Jupyter analysis notebooks
archive/      Legacy code variants (MATLAB, old power models)
tmp/          Scratch space
```

## Running Simulations
```bash
# Single master simulation (~12 hr sim, 2s step)
python src/ANTARA_Master.py

# Monte Carlo RAAN trade (7 RAANs × 2 runs × 365 days)
python src/ANTARA_MCR.py

# Battery sizing Monte Carlo (3 runs × 365 days)
python src/ANTARA_avg.py

# Fast CONOPS timeline (30 days, 60s step)
python src/ANTARA_conops.py
```

## Output Locations
Results written to `results/<script_subdir>/` via `paths.py` helpers:
- `results_path("mcr", "file.csv")` → `results/mcr/file.csv`
- `figures_path("thermal", "plot.png")` → `figures/thermal/plot.png`

## Critical Conventions
- **All scripts must call `orekit.initVM()` before any Orekit imports**
- **Quaternion convention: scalar-first [w, x, y, z], body-to-inertial**
- **Body frame (ICD Rev B): +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel**
- **Time base: AbsoluteDate in UTC, simulation steps in seconds**
- **Config dataclasses hold ALL numeric constants — no magic numbers in logic**

## Testing / Verification
No formal test suite. Verify by:
1. Run `python src/ANTARA_conops.py` (fastest, ~seconds)
2. Check `results/conops/ANTARA_conops_log.csv` exists with expected columns
3. Run `python src/ANTARA_Master.py` for full validation (longer)

## Common Gotchas
- **Orekit VM**: Must call `orekit.initVM()` exactly once at module top-level
- **Java classpath**: Orekit JARs resolved via `orekit` Python wrapper (uses Maven)
- **Matplotlib/cartopy**: Optional for plotting; scripts guard with try/except
- **Large outputs**: Monte Carlo runs generate many CSVs — monitor disk space
- **paths.py**: All file I/O routes through `results_dir()`, `figures_dir()`, etc.

## Configuration Knobs
Edit `MissionConfig` dataclass in each script for:
- Orbit parameters (altitude, inclination, RAAN, etc.)
- Spacecraft mass, drag/SRP coefficients
- ADCS torque/momentum limits, slew rates
- Battery capacity, solar panel area/efficiency
- Power loads per subsystem
- Ground station locations, elevation masks
- SAA polygon vertices