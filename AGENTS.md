# ANTARA Mission Analysis — Agent Instructions

## Project Overview
Python-based CubeSat mission simulation using Orekit (Java astrodynamics library). Simulates orbital dynamics, power systems, ADCS, thermal, and science operations for the ANTARA 3U CubeSat.

## Key Entry Points
Run from repository root with `python src/<script>.py`:

| Script | Purpose | Duration | Step |
|--------|---------|----------|------|
| `ANTARA_Master.py` | High-fidelity simulation with full FSM, ADCS, EPS, payload models | 12h default | 2s |
| `ANTARA_MCR.py` | Monte Carlo RAAN trade study (365-day, 7 RAANs × 2 runs) | ~hours | 300s |
| `ANTARA_avg.py` | Monte Carlo battery sizing study (365-day, 3 runs × battery cases) | ~hours | 300s |
| `ANTARA_conops.py` | Fast analytical propagation for CONOPS timeline (Eckstein-Hechler) | 30 days | 60s |
| `ANTARA_Science_Sim.py` | Science payload simulation with proton flux modelling (CLI configurable) | 24h default | 2s |
| `ANTARA_FINAL_POWERGEN.py` | RAAN power optimization (24h × 24 RAANs at equinox) | ~minutes | 30s |
| `ANTARA_drag_finalconfig.py` | Orbital lifetime (final config, altitude sweep 500-600km) | ~seconds | 1 day |
| `ANTARA_drag_study.py` | Orbital lifetime trade study (4 configs, altitude sweep) | ~seconds | 1 day |
| `TradeCase.py` | Analytical battery DoD/cycle life trade (static eclipse table) | ~seconds | 10s |
| `ANTARA_pointing_budget.py` | GS pointing budget (1-day Keplerian, nadir-pointing) | ~seconds | 10s |
| `ThermalSimulator.py` | Thermal lumped-capacitance (7-node, RK4, bounding cases) | ~minutes | 2s |
| `leop_detumble_mc.py` | LEOP detumble Monte Carlo capability envelope (2000-5000 trials) | ~minutes | - |
| `analyze_detumble_mc.py` | Analyze detumble MC results (heatmaps, contours, per-axis) | ~seconds | - |
| `compare_scenarios.py` | Compare scenario outputs (telemetry, mode transitions, SAA passes) | ~seconds | - |
| `plot_fsm_diagram.py` | Generate FSM diagrams (Graphviz PNG/SVG + Mermaid) | ~seconds | - |

## Environment Setup
```bash
# Required: Java 11+ for Orekit VM
# Python deps:
pip install orekit numpy pandas matplotlib cartopy shapely scipy skyfield pytest graphviz
# Optional: pyIGRF, spacepy for enhanced science models (ANTARA_Science_Sim.py)
```

## Orekit Data (Required)
Place Orekit data at one of:
- `orekit-data/orekit-data-main/` (extracted, at repo root)
- `orekit-data.zip` (at repo root, auto-downloaded if missing via `init_orekit_data(download_if_missing=True)`)

Scripts initialize via `from paths import init_orekit_data; init_orekit_data()` (called at runtime, not import time).

## Directory Structure
```
src/          Active simulation entrypoints + paths.py + shared utils
data/         Orekit datasets + reference CSV inputs (gitignored)
results/      Generated CSV/TXT outputs (gitignored)
figures/      Generated plots (gitignored)
notebooks/    Jupyter analysis notebooks
archive/      Legacy code variants (MATLAB, old power models)
tmp/          Scratch space
docs/         Auto-generated documentation (FSM diagrams, VALIDATION_REPORT.md)
tests/        Unit and validation tests
scripts/      Utility scripts (generate_validation_report.py)
.github/      CI/CD workflows
```

## Running Simulations
```bash
# Fast verification (seconds) — START HERE
python src/ANTARA_conops.py

# Full high-fidelity (minutes)
python src/ANTARA_Master.py

# Monte Carlo RAAN trade (hours)
python src/ANTARA_MCR.py

# Battery sizing MC (hours)
python src/ANTARA_avg.py

# Science payload — 3 predefined scenarios (minutes-hours, CLI options)
python src/ANTARA_Science_Sim.py --scenario leop_24h        # 24h with LEOP detumble
python src/ANTARA_Science_Sim.py --scenario no_leop_24h     # 24h steady-state
python src/ANTARA_Science_Sim.py --scenario leop_1yr        # 1 year, multi-rate stepping
python src/ANTARA_Science_Sim.py --scenario leop_1yr --coarse-step-sec 30.0
python src/ANTARA_Science_Sim.py --scenario leop_24h --no-plots  # faster, no matplotlib

# LEOP Detumble Monte Carlo (Stage 2)
python src/leop_detumble_mc.py --n-trials 2000 --output detumble_mc_results.pkl
python src/analyze_detumble_mc.py detumble_mc_results.pkl --output-dir detumble_analysis

# Thermal analysis (minutes)
python src/ThermalSimulator.py --case HOT --target-beta-deg 90

# FSM diagrams (requires graphviz system package)
python src/plot_fsm_diagram.py --output-dir docs --format both

# Trade studies (seconds)
python src/ANTARA_FINAL_POWERGEN.py
python src/ANTARA_drag_finalconfig.py
python src/ANTARA_drag_study.py
python src/TradeCase.py
python src/ANTARA_pointing_budget.py
```

## Output Locations
Results written to `results/<script_subdir>/` via `paths.py` helpers:
- `results_path("mcr", "file.csv")` → `results/mcr/file.csv`
- `figures_path("thermal", "plot.png")` → `figures/thermal/plot.png`

Subdirectories: `master/`, `mcr/`, `avg/`, `conops/`, `science/`, `thermal/`, `tradecase/`, `pointing/`

Science scenarios write to `results/science/<tag>_*.csv` and `figures/science/<tag>_*.png`:
- `<tag>_ANTARA_Science_telemetry.csv` - per-timestep telemetry (80+ columns)
- `<tag>_ANTARA_Science_saa_passes.csv` - SAA entry/exit events with flux integrals
- `<tag>_ANTARA_Science_gs_contacts.csv` - Ground station contacts (India + Svalbard)
- `<tag>_ANTARA_Science_mode_transitions.csv` - FSM transitions with reasons
- `<tag>_ANTARA_Science_slew_events.csv` - Slew angles, durations
- `<tag>_ANTARA_Science_detumble_events.csv` - Startup detumble logs
- `<tag>_ANTARA_Science_proton_passes.csv` - Per-SAA-pass proton science summary

## Critical Conventions
- **`orekit.initVM()` must be called before Orekit Java imports**. Most scripts call it at module top-level; `ANTARA_Master.py` and `ANTARA_Science_Sim.py` call it inside `_init_orekit()` method (works because imports are lazy).
- **Quaternion convention: scalar-first [w, x, y, z], body-to-inertial**
- **Body frame (ICD Rev B): +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel**
- **Time base: AbsoluteDate in UTC, simulation steps in seconds**
- **Config dataclasses hold ALL numeric constants — no magic numbers in logic**
- **`paths.py` is the single I/O routing layer** — all file paths go through `results_path()`/`figures_path()`

## Testing / Verification
No formal unit test suite for simulation logic. Verify by:
1. Run `python src/ANTARA_conops.py` (fastest, ~seconds)
2. Check `results/conops/ANTARA_conops_log.csv` exists with expected columns
3. Run `python src/ANTARA_Master.py` for full validation (longer)

### Validation Suite (Stage 4)
Six validation tests against real-world references:
| Test | Subsystem | Reference | Tolerance |
|------|-----------|-----------|-----------|
| `test_orbit_propagation.py` | Orbit | Skyfield/SGP4, SSO precession | Period ±1s, Precession ±5% |
| `test_igrf_field.py` | IGRF B-field | NOAA IGRF-13 calculator | Field magnitude ±5% |
| `test_adcs_detumble_time.py` | ADCS Detumble | B-dot τ ≈ I/(k·B²) | Order-of-magnitude |
| `test_eps_power_budget.py` | EPS | BOL cell × packing factor | ±2% |
| `test_saa_flux_model.py` | SAA/Flux | AP9 quiet-time at L~1.5 | Factor of 2 |
| `test_comms_link_budget.py` | Comms | Friis link budget | Heuristic within 30% at 45° |

Run all and generate report:
```bash
python scripts/generate_validation_report.py
# Output: VALIDATION_REPORT.md + validation_results_*.json
```

### FSM Drift Guard Test (Stage 3)
```bash
python -m pytest tests/test_fsm_spec_matches_code.py -v
```

## CI/CD Pipeline (Stage 5)
| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Smoke Tests** | Every push/PR | Syntax, imports, basic imports |
| **Unit Tests** | Every push/PR | FSM drift guard, syntax checks |
| **Scenario Smoke** | Every push/PR | 1-hour sim for leop_24h, no_leop_24h |
| **FSM Diagram** | Every push/PR | Generates PNG/SVG/Mermaid diagrams |
| **Nightly Full** | Daily 2 AM UTC | Full scenarios, MC (5000 trials), validation |
| **Weekly Validation** | Sundays 3 AM | Full validation suite + report |

Artifacts retained 30-90 days: `fsm-diagrams`, `nightly-results`, `validation-report`, `validation-data`

## Common Gotchas
- **Orekit VM**: Must call `orekit.initVM()` exactly once — redundant calls are harmless but wasteful
- **Java classpath**: Orekit JARs resolved via `orekit` Python wrapper (uses Maven) — first run downloads ~100MB
- **Matplotlib/cartopy**: Optional for plotting; `ANTARA_Science_Sim.py` and `ThermalSimulator.py` guard with try/except; `ANTARA_Master.py` imports directly (will fail if missing)
- **Large outputs**: Monte Carlo runs generate many CSVs — monitor disk space (`results/mcr/` can be 100MB+)
- **Orekit data**: Scripts call `init_orekit_data()` at runtime — ensure data exists before running
- **No shared library**: Each script duplicates propagation/power/SAA/GS logic — changes must be replicated across files
- **Orekit wrapper**: NOT on PyPI — must build from source (see requirements.txt for instructions)

## Configuration Knobs
Edit `MissionConfig` dataclass in each script for:
- Orbit parameters (altitude, inclination, RAAN, etc.)
- Spacecraft mass, drag/SRP coefficients
- ADCS torque/momentum limits, slew rates
- Battery capacity, solar panel area/efficiency
- Power loads per subsystem
- Ground station locations, elevation masks
- SAA polygon vertices

Key knobs in `ANTARA_Science_Sim.py`:
```python
altitude_m = 600_000.0
inclination_deg = 97.6
raan_deg = 15.0
max_slew_rate_deg_sec = 5.0
wheel_max_torque_nm = 0.001
wheel_max_momentum_nms = 0.01317
battery_capacity_wh = 25.0
solar_panel_efficiency = 0.193515
solar_panel_area_m2 = 0.085
downlink_rate_mbps = 0.0192
gs_elevation_mask_deg = 15.0
coarse_step_sec = 20.0  # multi-rate stepping (leop_1yr)
```

## Key Differences Between Scripts
| Feature | Master | MCR | Avg | ConOps | Science | Thermal |
|---------|--------|-----|-----|--------|---------|---------|
| Propagator | Numerical (DP853) | Numerical (DP853) | Numerical (DP853) | Analytical (Eckstein-Hechler) | Numerical (DP853) | Numerical (DP853) |
| ADCS | Full model (wheels, mag, detumble) | None (nadir) | None (nadir) | None | Full (LVLH-fixed) | None (LVLH/tumbling) |
| EPS | Detailed (taper, loads, heater) | Simple (fixed load) | Simple (fixed load) | None | Detailed | Synthetic duty-cycle |
| FSM | 7-state | None | None | None | 7-state (variant) | None |
| SAA | Shapely polygon | Lat/lon box | Lat/lon box | Lat/lon box | Shapely + distance | None |
| GS | Elevation + TopocentricFrame | Lat/lon box | Lat/lon box | Lat/lon box | Dual GS (India + Svalbard) | None |
| Outputs | 6 CSVs + 9-panel plot | 7 CSVs + stats | 3 CSVs + summary | 1 CSV | 7 CSVs + 3 plots | 2 CSVs + 4 plots |

## Architecture Notes
- **No shared simulation core**: Each script is standalone with duplicated Orekit setup, propagation loop, and subsystem models
- **`paths.py` is the only true shared module** — provides path helpers and Orekit data init
- **MissionConfig pattern**: Each script defines its own dataclass with overlapping but different parameters
- **Two propagator types**: Numerical (DP853) for fidelity; Analytical (Eckstein-Hechler/Keplerian) for speed
- **SAA model divergence**: `Master`/`Science` use Shapely polygon with entry/exit events; `MCR`/`Avg`/`ConOps` use simple lat/lon bounding boxes
- **GS model divergence**: `Master`/`Science`/`Pointing` use Orekit elevation; others use lat/lon boxes

## Contributing
1. All numeric constants in `MissionConfig` - no magic numbers
2. `orekit.initVM()` must be called before any `org.orekit.*` imports
3. Quaternion convention: scalar-first [w,x,y,z], body-to-inertial
4. Body frame (ICD Rev B): +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel
5. Run `python -m py_compile src/*.py` before committing