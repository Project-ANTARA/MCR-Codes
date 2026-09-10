# ANTARA Mission Analysis — Source Code Documentation

Auto-generated documentation for all Python source files in `src/`. Each file has a corresponding `.md` document with:
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
| [`ANTARA_Science_Sim.md`](ANTARA_Science_Sim.md) | **Science payload simulation** — Proton flux (AP9/IGRF), detector physics, event-driven histograms, dual-GS, CLI |

### Shared Models

| File | Description |
|------|-------------|
| [`antara_models.md`](antara_models.md) | **Orekit-free core models** — `MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `_as_flat_vector` (zero Java deps, for lightweight MC/validation) |

### Specialized Studies

| File | Description |
|------|-------------|
| [`ANTARA_FINAL_POWERGEN.md`](ANTARA_FINAL_POWERGEN.md) | **RAAN power optimization** — 24h × 24 RAANs (15° steps), fixed zenith + 2 deployed wings, nadir-pointing |
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

# Science payload (minutes-hours)
python src/ANTARA_Science_Sim.py --duration-hours 24

# Thermal analysis
python src/ThermalSimulator.py --case HOT --target-beta-deg 90

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
| `orekit.initVM()` | **Must be first** — called at module top-level in every script |
| `init_orekit_data()` | Initializes Orekit data from `orekit-data/` or `orekit-data.zip` |
| `results_path("subdir", "file.csv")` | Output path: `results/subdir/file.csv` (auto-creates dirs) |
| `figures_path("subdir", "plot.png")` | Figure path: `figures/subdir/plot.png` (auto-creates dirs) |
| `MissionConfig` dataclass | **All numeric constants** — no magic numbers in logic |
| `BODY_AXIS_VECTORS` | Canonical body axes: +X=RAM, +Y=nadir/relay, +Z=sun-panel |
| Quaternion convention | Scalar-first [w,x,y,z], body-to-inertial |
| `from antara_models import ...` | **Lightweight imports** — `MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `_as_flat_vector`, `BODY_AXIS_VECTORS` (no Orekit/JVM) |

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

**Required**:
- Java 11+ (for Orekit JVM)
- Python: `orekit`, `numpy`, `pandas`

**Optional** (for plotting):
- `matplotlib`, `cartopy`

**Optional** (for enhanced science):
- `pyIGRF` (`ppigrf`) — IGRF magnetic field
- `spacepy` (`irbempy`) — AP9/AE9 radiation belts

---

*Generated by automated documentation script. For questions about specific modules, see individual `.md` files.*