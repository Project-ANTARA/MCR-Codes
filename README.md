# ANTARA Mission Analysis Workspace

This workspace contains the ANTARA 3U CubeSat mission simulation codebase, organized for active development, validation, and analysis.

**Science Sim is the authoritative DITL simulator.** Master is a superseded
attempt, retained for historical reference—not a second active DITL target.
Implementations and their guides live together under `src/simulations/`.
Flat `src/` files have been moved, not copied; shared utilities live in
`src/simulations/helper/`. Old flat commands and imports are no longer supported. See [simulation layout](docs/SIMULATION_LAYOUT.md).

## Current status

September 25 audit repairs change ADCS motion/torque accounting, payload flux response, detumble trial semantics, duration-weighted reporting, and supporting-study integration/energy accounting. See [changes, verification and limits](docs/AUDIT_REPAIRS_2026-09-25.md). Headless Science writes an exact scalar `*_ANTARA_Science_summary.json` without loading the full telemetry into memory. Historical results below retain their original scope; do not compare old payload counts or detumble capability estimates as though model semantics were unchanged.

Documentation reviewed against the working tree on 2026-09-12. Science's annual preset has been verified through **60 hours**, not a full year: 17,758 telemetry rows, successful saturation → SAFE → SUN_POINTING recovery, and a maximum settled, non-held SUN_POINTING error of 4.999973° against the unchanged 5° limit. See the [runtime repair evidence](src/simulations/ditl/SIMULATION_RUNTIME_FIX_REPORT.md).

The pre-reorganization combined regression suite passed **345 tests** locally.
See [short operational evidence](src/simulations/ditl/SHORT_OPERATIONAL_VERIFICATION.md) and
[reorganization verification](docs/SIMULATION_LAYOUT.md) for their distinct scopes.

Start with the [documentation index](docs/README.md), [Science guide](src/simulations/ditl/ANTARA_Science_Sim.md), and [detumble Monte Carlo guide](src/simulations/detumble/leop_detumble_mc.md). Local test results do not certify all subsystem physics or a hosted CI run. Live AP9 remains unavailable through the current interface.

## Layout

```
src/simulations/ Implementations and guides grouped by simulation family;
                 shared helper/ utilities and superseded legacy/ Master
data/         Orekit datasets and reference inputs (gitignored)
results/      Generated CSV/TXT outputs from simulations (gitignored)
figures/      Generated plots and images (gitignored)
notebooks/    Jupyter analysis notebooks
archive/      Legacy code variants (MATLAB, old power models)
tmp/          Scratch space
docs/         Repository-wide documentation index, reports and runner operations
tests/        Unit and validation tests
scripts/      Backend verification and validation report generator
.github/      CI/CD workflows
```

## Quick Start

### Prerequisites
- **Java 11+** (required for Orekit JVM)
- **Python 3.11 in CI**; local physics fixes were verified with Python 3.10.21
- **Orekit data** at a repository-root location listed below; prepare it before simulation

```bash
# Use the local environment containing Orekit; Micromamba is not required locally
conda activate orekit_env
conda install -c conda-forge orekit numpy pandas matplotlib cartopy shapely scipy skyfield pytest pip ppigrf=2.1.0
python -m pip install spacepy==0.7.0 graphviz
python scripts/verify_physics_dependencies.py
```

### Fast Verification (seconds)
```bash
# Fast CONOPS timeline (analytical propagation)
python src/simulations/conops/ANTARA_conops.py

# Import check for the active DITL simulator
PYTHONPATH=src python -c "
import orekit; orekit.initVM()
from simulations.ditl.ANTARA_Science_Sim import MissionConfig, ANTARAScienceSimulator
print('All imports OK')
"

# Lightweight Orekit-free imports (for MC/validation scripts)
python -c "
import sys; sys.path.insert(0, 'src')
from simulations.helper.antara_models import MissionConfig, ADCSModel, SpacecraftFSM
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
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h

# 24-hour without LEOP (starts in SUN_POINTING)
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario no_leop_24h

# 1-year mission with multi-rate stepping (coarse=20s default)
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_1yr

# Override coarse step for leop_1yr
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_1yr --coarse-step-sec 30.0

# Disable plot generation
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h --no-plots

# CI-equivalent one-hour runs with fixed two-second steps
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots
```

Explicit CLI values override scenario presets, which override global defaults. An explicit `--step-sec` also sets the coarse step unless `--coarse-step-sec` is supplied separately. Preset-only runs retain multirate stepping. `--no-plots`, RNG, and periphery-merge controls also apply with scenarios.

The verified one-hour runs each produced 1,801 telemetry rows (both endpoints included), spanning `time_sec=0` to `3600` with two-second gaps. Wheel-capacity events can shorten a requested step (`wheel_event_split`); the horizon can also clip the final interval. Reusing an output tag replaces that scenario's outputs; preserve earlier results before rerunning.

The [September 25 execution/payload repair](src/simulations/ditl/EXECUTION_PAYLOAD_REPAIR.md) unifies stepped and continuous runs and removes extra endpoint integration. The [September 28 operational checks](src/simulations/ditl/SHORT_OPERATIONAL_VERIFICATION.md) follow the [wheel-event repair](WHEEL_EVENT_REPAIR.md) and advance atomic checkpoints to **version 7**: nominal-frame and slew-energy corrections make older physics histories incompatible. Keep the original CSV alongside its checkpoint for separate-file resume. Telemetry exposes interval duration and end-state time, with a zero-duration terminal observation. Verification uses short runs, not annual simulations.

### Physics backends and limitations

The simulator imports **`ppigrf==2.1.0`**, not `pyIGRF`. Its IGRF-14 backend takes altitude in kilometers and returns East/North/Up in nT; ANTARA converts to North/East/Down in Tesla. Missing ppigrf selects a dipole fallback with a warning.

When `earth_b_field_model="igrf"` is selected, Science queries at the actual mission date and rotates the Earth-fixed field into the inertial frame using Orekit. Standalone NED queries preserve decimal-year fractions. The September 2026 [scientific-integrity repair record](src/simulations/ditl/SCIENTIFIC_INTEGRITY_REPAIR.md) tracks these corrections, dual-station consistency, invalid detumble-state rejection, and the unresolved audit backlog. Passing these regressions is not full-year scientific validation.

**SpacePy 0.7.0 / IRBEM does not enable live AP9.** The current AP9 query uses an unsupported interface and retains the configured flux baseline, warning once when queried. `get_AEP8` supplies AE8/AP8; the verification script exercises AP8 only. This baseline feeds the primary proton-flux calculation, so it is not merely a plotting limitation. Keep Orekit, ppigrf, and SpacePy in the same environment; do not install physics packages only into base Conda or with `--user`.

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

Process-pool tasks use a module-level worker with explicit configuration and timestep arguments, including on spawn-based platforms. Results retain seeded trial order across serial and parallel runs; the requested count includes the baseline trial. Use `--workers` to bound local CPU usage. See [runtime repair verification](src/simulations/ditl/SIMULATION_RUNTIME_FIX_REPORT.md).

The terminal progress bar shows completed/total trials, successes, timeouts, rate-cap hits, elapsed wall time, and an approximate ETA based on completed random trials. Baseline and serial execution also report simulated seconds, detumble phase, and body-rate magnitude. Parallel progress is reported only by the parent process. Redirected output uses occasional plain-text lines on stderr; `--no-progress` hides the display but retains setup information and the final summary. A timeout is a trial outcome, not a worker error; worker exceptions still stop the run.

```bash
# Run Monte Carlo with progress (count includes baseline; use a new output path)
python src/simulations/detumble/leop_detumble_mc.py --n-trials 5000 --workers 4 --output detumble_mc_results.pkl

# Analyze results (heatmaps, contours, per-axis breakdown)
python src/simulations/detumble/analyze_detumble_mc.py detumble_mc_results.pkl --output-dir detumble_analysis
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

`src/simulations/helper/fsm_spec.py` describes the **shared** FSM in `src/simulations/helper/antara_models.py`. Science defines a separate local FSM, including saturation-hold priority during SLEWING; the shared drift guard and generated diagrams do not certify that local implementation. Its recovery path has separate runtime regressions.

### Generate Diagrams

```bash
# Generate FSM diagrams (requires graphviz)
sudo apt-get install graphviz  # or brew install graphviz
python src/simulations/ditl/plot_fsm_diagram.py --output-dir src/simulations/ditl --format both
```

**Outputs:**
- `src/simulations/ditl/fsm_diagram.png` / `.svg` - Graphviz rendered state diagram
- `src/simulations/ditl/fsm_diagram.mmd` - Mermaid stateDiagram-v2 (renderable in GitHub, VS Code, Obsidian)

### Drift Guard Test

Ensures code matches spec:

```bash
python -m pytest tests/test_fsm_spec_matches_code.py -v
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py -v
PYTHONPATH=src python -m pytest tests/test_scientific_integrity.py -v
```

The regressions cover startup geometry, negative alignments, IGRF units, CLI precedence, multirate accounting, saturation recovery, spawn-safe Monte Carlo serialization, and terminal progress. Numerical alignment uses normalized cross/dot `atan2` with unchanged `1e-6°` startup tolerance; this is not a flight pointing requirement. The [dependency repair report](OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md) retains its earlier 14-test evidence; [runtime verification](src/simulations/ditl/SIMULATION_RUNTIME_FIX_REPORT.md) records subsequent checks.

---

## Subsystem Validation (Stage 4)

Six standalone subsystem checks against the reference values and heuristics embedded in their source. These are separate from the targeted regression suite; a passing baseline-flux comparison is not live AP9 validation:

| Test | Subsystem | Reference | Tolerance |
|------|-----------|-----------|-----------|
| `test_orbit_propagation.py` | Orbit | Numerical two-body crossing, closed-form J2 precession | Period ±1s, Precession ±5% |
| `test_igrf_field.py` | IGRF B-field | Independent IRBEM IGRF-13; 2025 endpoint caveat | Magnitude and NED vector ±5% |
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

# Output: results/validation/VALIDATION_REPORT.md, JSON files and full logs
# For subsequent runs, preserve old evidence and choose a fresh directory:
python scripts/generate_validation_report.py --output-dir results/validation-rerun
```

---

## Legacy / Supporting Studies

| Script | Purpose |
|--------|---------|
| `ANTARA_Master.py` | Legacy/superseded DITL attempt; not an active repair target |
| `ANTARA_MCR.py` | Monte Carlo RAAN trade (365 days × 7 RAANs) |
| `ANTARA_avg.py` | Battery sizing MC (365 days × 3 runs) |
| `ANTARA_conops.py` | Fast CONOPS (30 days, analytical) |
| `ANTARA_FINAL_POWERGEN.py` | RAAN power optimization (24h × 25 samples, including equivalent 0°/360° endpoints) |
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
| **Unit Tests** | Every push/PR | Shared FSM drift guard, physics/attitude and detumble multiprocessing/progress regressions |
| **Scenario Smoke** | Every push/PR | Both scenarios: 1 hour, fixed 2-second steps, telemetry bounds checked |
| **FSM Diagram** | Every push/PR | Generates PNG/SVG/Mermaid diagrams |
| **Nightly Full** | Daily 2 AM UTC | Full scenarios, MC (5000 trials), validation |
| **Weekly Validation** | Sundays 3 AM | Full validation suite + report |

Nightly, weekly, and push runs use separate concurrency groups. Once this workflow
is published on the default branch, manual CLI triggers are available:

```bash
gh workflow run ci.yml --ref main -f suite=weekly
gh workflow run ci.yml --ref main -f suite=nightly
```

Manual triggers launch real Actions jobs. Nightly retains all 8,760 simulated
hours and now targets the local `antara-nightly` runner, with a 20-hour annual
process budget and a 23-hour job budget. The local service was installed and
verified online on 2026-09-13. Service-account physics setup and full-year
completion remain **unverified**. See [local runner status and operation](docs/LOCAL_NIGHTLY_RUNNER.md).
See [scheduled verification repairs and limits](docs/SCHEDULED_VERIFICATION_FIX_REPORT.md).

All five Orekit jobs use Micromamba environment `antara`, conda-forge `ppigrf=2.1.0`, and activated `python -m pip install spacepy==0.7.0`. Backend verification checks `/micromamba/envs/antara/` in the interpreter path and executes real IGRF/IRBEM queries. These workflow changes were checked locally; the repair report does not claim a hosted CI run.

### Artifacts (retained 30-90 days)
- `fsm-diagrams` - FSM diagrams (PNG/SVG/Mermaid)
- `nightly-results` - Full scenario outputs, MC analysis, VALIDATION_REPORT.md
- `nightly-fsm-diagrams` - Checked-in FSM diagrams (not regenerated by nightly)
- `validation-report` - Weekly VALIDATION_REPORT.md
- `validation-data` - Raw validation JSON outputs

---

## Orekit Data

Orekit requires ephemeris/atmosphere data. Run this from the repository root to prepare the root-level archive used by CI and local simulations:

```bash
python -c 'import orekit; orekit.initVM(); from orekit.pyhelpers import download_orekit_data_curdir; download_orekit_data_curdir()'
```

Expected locations (auto-detected):
- `orekit-data/orekit-data-main/` (extracted, repository root)
- `orekit-data.zip` (repository root)

`src/simulations/helper/paths.py` currently searches only these two locations. Its optional `download_if_missing=True` branch writes to `data/orekit-data.zip` for that invocation, but subsequent default calls do not search that location. Science calls the default initializer and therefore needs data prepared beforehand. See [path behavior](src/simulations/helper/paths.md).

---

## Configuration

Science's `MissionConfig` lives in `src/simulations/helper/antara_models.py`; standalone studies define their own configurations. Edit the defining module, not a re-export. Science also retains local `EPSModel` and `SpacecraftFSM` definitions distinct from the shared models; inspect the actual consumer before synchronizing changes. Key knobs:

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
coarse_step_sec = 20.0  # outside startup, active modes, eclipse edges, saturation holds
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
│   ├── simulations/              # Implementations; see docs/SIMULATION_LAYOUT.md
│   ├── ANTARA_Master.py           # Legacy/superseded DITL (unchanged)
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
│   ├── test_physics_attitude.py
│   ├── test_detumble_multiprocessing.py
│   └── validation/
│       ├── test_orbit_propagation.py
│       ├── test_igrf_field.py
│       ├── test_adcs_detumble_time.py
│       ├── test_eps_power_budget.py
│       ├── test_saa_flux_model.py
│       └── test_comms_link_budget.py
├── scripts/
│   ├── verify_physics_dependencies.py
│   └── generate_validation_report.py
├── .github/workflows/ci.yml
├── docs/                          # Maintained guides and generated FSM diagrams
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
2. Initialize the Orekit JVM before using Java-backed APIs; preserve each entrypoint's initialization order
3. Quaternion convention: scalar-first [w,x,y,z], body-to-inertial
4. Body frame (ICD Rev B): +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel
5. Run `python -m compileall -q src` and the layout parity tests to check both full source layouts

---

See [module documentation](docs/README.md), [contributor instructions](AGENTS.md), and the [physics/attitude repair report](OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md). `VALIDATION_REPORT.md` is a dated captured run, not proof that the current revision passes every subsystem validation.
