# ANTARA Mission Analysis Workspace

This workspace contains the ANTARA 3U CubeSat mission simulation codebase, organized for active development, validation, and analysis.

## Layout

```
src/          Active simulation entrypoints and shared utilities
data/         Orekit datasets and reference inputs (gitignored)
results/      Generated CSV/TXT outputs from simulations (gitignored)
figures/      Generated plots and images (gitignored)
notebooks/    Jupyter analysis notebooks
archive/      Legacy code variants (MATLAB, old power models)
tmp/          Scratch space
docs/         Generated documentation (FSM diagrams, VALIDATION_REPORT.md)
tests/        Unit and validation tests
scripts/      Utility scripts (validation report generator)
.github/      CI/CD workflows
```

## Quick Start

### Prerequisites
- **Java 11+** (required for Orekit JVM)
- **Python 3.11+**
- **Orekit data** (auto-downloaded on first run, or place at `data/orekit-data.zip`)

```bash
# Install Python dependencies
pip install orekit numpy pandas matplotlib cartopy shapely scipy skyfield pytest
```

### Fast Verification (seconds)
```bash
# Fast CONOPS timeline (analytical propagation)
python src/ANTARA_conops.py

# Syntax/import check all modules
python -c "
import orekit; orekit.initVM()
from ANTARA_Science_Sim import MissionConfig, ANTARAScienceSimulator
from ANTARA_Master import MissionConfig, ANTARASimulator
print('All imports OK')
"

# Lightweight Orekit-free imports (for MC/validation scripts)
python -c "
import sys; sys.path.insert(0, 'src')
from antara_models import MissionConfig, ADCSModel, SpacecraftFSM
print('Orekit-free imports OK')
"
```

---

## Simulation Scenarios (Stage 1)

The Science Simulator supports three predefined scenarios via `--scenario`:

| Scenario | Duration | LEOP | Output Tag | Use Case |
|----------|----------|------|------------|----------|
| `leop_24h` | 24 hours | ✅ Full detumble | `leop_24h` | Full mission with startup |
| `no_leop_24h` | 24 hours | ❌ Starts in SUN_POINTING | `no_leop_24h` | Operational steady-state |
| `leop_1yr` | 8760 hours (1 year) | ✅ Full detumble | `leop_1yr` | Full mission with multi-rate stepping |

### Running Scenarios

```bash
# 24-hour with LEOP (full detumble)
python src/ANTARA_Science_Sim.py --scenario leop_24h

# 24-hour without LEOP (starts in SUN_POINTING)
python src/ANTARA_Science_Sim.py --scenario no_leop_24h

# 1-year mission with multi-rate stepping (coarse=20s default)
python src/ANTARA_Science_Sim.py --scenario leop_1yr

# Override coarse step for leop_1yr
python src/ANTARA_Science_Sim.py --scenario leop_1yr --coarse-step-sec 30.0

# Disable plotting (faster, no matplotlib/cartopy needed)
python src/ANTARA_Science_Sim.py --scenario leop_24h --no-plots
```

### Outputs
Each scenario writes to `results/science/<tag>_*.csv` and `figures/science/<tag>_*.png`:
- `<tag>_ANTARA_Science_telemetry.csv` - per-timestep telemetry (80+ columns)
- `<tag>_ANTARA_Science_saa_passes.csv` - SAA entry/exit events with flux integrals
- `<tag>_ANTARA_Science_gs_contacts.csv` - Ground station contacts (India + Svalbard)
- `<tag>_ANTARA_Science_mode_transitions.csv` - FSM transitions with reasons
- `<tag>_ANTARA_Science_slew_events.csv` - Slew angles, durations
- `<tag>_ANTARA_Science_detumble_events.csv` - Startup detumble logs
- `<tag>_ANTARA_Science_proton_passes.csv` - Per-SAA-pass proton science summary
- Plots: mission overview, proton analysis, attitude diagnostics

---

## LEOP Detumble Monte Carlo (Stage 2)

Capability envelope analysis for tip-off rate/acceleration combinations.

```bash
# Run Monte Carlo (2000 trials default, parallelized)
python src/leop_detumble_mc.py --n-trials 2000 --output detumble_mc_results.pkl

# Analyze results (heatmaps, contours, per-axis breakdown)
python src/analyze_detumble_mc.py detumble_mc_results.pkl --output-dir detumble_analysis
```

**Outputs:**
- `detumble_mc_results.pkl` - raw trial data
- `detumble_analysis/success_heatmap.png` - 2D success probability over (rate, accel)
- `detumble_analysis/success_contours.png` - 50%/95% logistic contours with margin
- `detumble_analysis/time_to_complete.png` - completion time vs rate with trend
- `detumble_analysis/per_axis_breakdown.png` - x/y/z histograms (Ixx=Iyy≠Izz)
- `detumble_analysis/peak_wheel_momentum.png` - momentum map
- `detumble_analysis/detumble_mc_summary.csv` - full trial data

---

## FSM Decision Specification & Diagrams (Stage 3)

The FSM logic is documented in `src/fsm_spec.py` with exact branch ordering matching `SpacecraftFSM.evaluate()`.

### Generate Diagrams

```bash
# Generate FSM diagrams (requires graphviz)
sudo apt-get install graphviz  # or brew install graphviz
python src/plot_fsm_diagram.py --output-dir docs --format both
```

**Outputs:**
- `docs/fsm_diagram.png` / `.svg` - Graphviz rendered state diagram
- `docs/fsm_diagram.mmd` - Mermaid stateDiagram-v2 (renderable in GitHub, VS Code, Obsidian)

### Drift Guard Test

Ensures code matches spec:

```bash
python -m pytest tests/test_fsm_spec_matches_code.py -v
```

---

## Subsystem Validation (Stage 4)

Six validation tests against real-world references:

| Test | Subsystem | Reference | Tolerance |
|------|-----------|-----------|-----------|
| `test_orbit_propagation.py` | Orbit | Skyfield/SGP4, closed-form SSO precession | Period ±1s, Precession ±5% |
| `test_igrf_field.py` | IGRF B-field | NOAA IGRF-13 calculator | Field magnitude ±5% |
| `test_adcs_detumble_time.py` | ADCS Detumble | B-dot τ ≈ I/(k·B²) | Order-of-magnitude |
| `test_eps_power_budget.py` | EPS | BOL cell × packing factor | ±2% |
| `test_saa_flux_model.py` | SAA/Flux | AP9 quiet-time at L~1.5 | Factor of 2 |
| `test_comms_link_budget.py` | Comms | Friis link budget | Heuristic within 30% at 45° |

### Run Validation

```bash
# Run individual test
python tests/validation/test_orbit_propagation.py

# Run all and generate report
python scripts/generate_validation_report.py

# Output: VALIDATION_REPORT.md + validation_results_*.json
```

---

## Legacy / Supporting Studies

| Script | Purpose |
|--------|---------|
| `ANTARA_Master.py` | High-fidelity 12h simulation (full FSM, ADCS, EPS, payload) |
| `ANTARA_MCR.py` | Monte Carlo RAAN trade (365 days × 7 RAANs) |
| `ANTARA_avg.py` | Battery sizing MC (365 days × 3 runs) |
| `ANTARA_conops.py` | Fast CONOPS (30 days, analytical) |
| `ANTARA_FINAL_POWERGEN.py` | RAAN power optimization (24h × 24 RAANs) |
| `ANTARA_drag_finalconfig.py` | Lifetime sweep (500-600km, final config) |
| `ANTARA_drag_study.py` | Lifetime trade (4 configs) |
| `TradeCase.py` | Analytical battery DoD/cycle life |
| `ANTARA_pointing_budget.py` | GS pointing budget (1 day Keplerian) |
| `ThermalSimulator.py` | Thermal 7-node lumped capacitance (RK4) |
| `antara_models.py` | Orekit-free shared models (MissionConfig, ADCSModel, SpacecraftFSM) |

---

## CI/CD Pipeline (Stage 5)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **Smoke Tests** | Every push/PR | Syntax, imports, basic imports |
| **Unit Tests** | Every push/PR | FSM drift guard, syntax checks |
| **Scenario Smoke** | Every push/PR | 1-hour sim for leop_24h, no_leop_24h |
| **FSM Diagram** | Every push/PR | Generates PNG/SVG/Mermaid diagrams |
| **Nightly Full** | Daily 2 AM UTC | Full scenarios, MC (5000 trials), validation |
| **Weekly Validation** | Sundays 3 AM | Full validation suite + report |

### Artifacts (retained 30-90 days)
- `fsm-diagrams` - FSM diagrams (PNG/SVG/Mermaid)
- `nightly-results` - Full scenario outputs, MC analysis, VALIDATION_REPORT.md
- `nightly-fsm-diagrams` - Updated FSM diagrams
- `validation-report` - Weekly VALIDATION_REPORT.md
- `validation-data` - Raw validation JSON outputs

---

## Orekit Data

Orekit requires ephemeris/atmosphere data. The simulator auto-downloads on first run:

```bash
# Auto-download (first run)
python src/ANTARA_Science_Sim.py --duration-hours 0.1

# Or manual download
wget -O data/orekit-data.zip "https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip"
unzip data/orekit-data.zip -d data/
```

Expected locations (auto-detected):
- `data/orekit/orekit-data/orekit-data-main/` (extracted)
- `data/orekit-data.zip` (zip)

---

## Configuration

All numeric parameters live in `MissionConfig` dataclasses (one per script). Key knobs:

```python
# Orbit
altitude_m = 600_000.0
inclination_deg = 97.6
raan_deg = 15.0

# ADCS
max_slew_rate_deg_sec = 5.0
wheel_max_torque_nm = 0.001
wheel_max_momentum_nms = 0.01317

# EPS
battery_capacity_wh = 25.0
solar_panel_efficiency = 0.193515  # BOL × packing
solar_panel_area_m2 = 0.085

# Comms
downlink_rate_mbps = 0.0192  # 19.2 kbps
gs_elevation_mask_deg = 15.0

# Multi-rate (leop_1yr)
coarse_step_sec = 20.0  # used when not in SLEWING/DETECTION/RELAY
```

---

## Validation Report

The weekly `VALIDATION_REPORT.md` contains:
- Summary table (subsystem, status, reference, tolerance)
- Per-test stdout/stderr captures
- Quantitative results (period error, precession ratio, B-dot τ, efficiency error, flux ratios, link budget comparison)

---

## Directory Structure Reference

```
ANTARA_CODES-main/
├── src/
│   ├── ANTARA_Science_Sim.py      # Main science simulator (24h-1yr)
│   ├── ANTARA_Master.py           # High-fidelity PDR simulator
│   ├── ANTARA_MCR.py              # Monte Carlo RAAN trade
│   ├── ANTARA_avg.py              # Battery sizing MC
│   ├── ANTARA_conops.py           # Fast CONOPS (analytical)
│   ├── ANTARA_FINAL_POWERGEN.py   # RAAN power optimization
│   ├── ANTARA_drag_finalconfig.py # Lifetime (final config)
│   ├── ANTARA_drag_study.py       # Lifetime trade (4 configs)
│   ├── TradeCase.py               # Analytical battery trade
│   ├── ANTARA_pointing_budget.py  # GS pointing budget
│   ├── ThermalSimulator.py        # 7-node thermal (RK4)
│   ├── leop_detumble_mc.py        # Detumble MC capability
│   ├── analyze_detumble_mc.py     # MC analysis & plots
│   ├── compare_scenarios.py       # Scenario comparison
│   ├── scenarios.py               # Scenario builder
│   ├── fsm_spec.py                # FSM decision spec
│   ├── plot_fsm_diagram.py        # FSM diagram generator
│   ├── paths.py                   # I/O routing + Orekit init
│   └── antara_models.py           # Orekit-free shared models
├── tests/
│   ├── test_fsm_spec_matches_code.py
│   └── validation/
│       ├── test_orbit_propagation.py
│       ├── test_igrf_field.py
│       ├── test_adcs_detumble_time.py
│       ├── test_eps_power_budget.py
│       ├── test_saa_flux_model.py
│       └── test_comms_link_budget.py
├── scripts/
│   └── generate_validation_report.py
├── .github/workflows/ci.yml
├── docs/                          # Generated (FSM diagrams, etc.)
├── results/                       # Generated (gitignored)
├── figures/                       # Generated (gitignored)
├── data/                          # Orekit data (gitignored)
├── notebooks/
├── archive/                       # Legacy code
└── tmp/                           # Scratch
```

---

## Contributing

1. All numeric constants in `MissionConfig` - no magic numbers
2. `orekit.initVM()` must be first call in every script
3. Quaternion convention: scalar-first [w,x,y,z], body-to-inertial
4. Body frame (ICD Rev B): +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel
5. Run `python -m py_compile src/*.py` before committing

---

*See `docs/fsm_diagram.mmd` for FSM state machine, `VALIDATION_REPORT.md` for validation status.*