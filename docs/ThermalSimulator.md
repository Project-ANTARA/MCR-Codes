# ThermalSimulator.py

## Current scope (2026-09-12)

Source: [ThermalSimulator.py](../src/ThermalSimulator.py). Thermal has its own `ThermalMissionConfig`, attitude generation, and synthetic power schedule. It does not consume Science's saturation/SAFE telemetry automatically. Its CLI is independent: use its documented thermal options, not Science scenario or detumble progress switches. Use distinct output copies for HOT/COLD studies because default filenames are shared.

Setup and navigation: [repository README](../README.md) · [documentation index](README.md). Supporting study results were not rerun as part of this documentation review.

## File Overview
Lumped-capacitance thermal simulator for the ANTARA 3U CubeSat. Propagates orbit with Orekit (numerical, Dormand-Prince 8/5/3) and integrates a 7-node thermal network (6 faces + 1 internal bus) using RK4 with configurable sub-steps. Supports bounding-case analysis (HOT/COLD/NOMINAL), beta-angle targeting via RAAN search, latitude-dependent Earth albedo/IR, solar-cell electrical conversion derating, face-to-face conduction, and tumbling attitude worst-case. Produces telemetry CSV, worst-case summary CSV, and 4-panel plot suite.

## Dependencies
- **Standard Library**: `argparse`, `math`, `dataclasses`, `typing`
- **External**: `numpy`, `pandas`, `orekit` (Python wrapper)
- **Optional External**: `matplotlib` (guarded with try/except)
- **Internal**: `paths` (`init_orekit_data`, `results_path`, `figures_path`) — with local fallback if import fails
- **Orekit/Java Modules**: Numerical propagation, forces (gravity 10×10, drag NRLMSISE-00/DTM2000, SRP with occultation), frames, orbits, time, bodies, atmosphere

## Exports / Public API

### Module Constants
| Constant | Type | Description |
|----------|------|-------------|
| `BODY_AXIS_VECTORS` | `Dict[str, np.ndarray]` | Canonical body axes (+X, -X, +Y, -Y, +Z, -Z) |
| `FACE_ORDER` | `List[str]` | Ordered face list for consistent iteration |
| `FACE_ADJACENCY` | `Dict[str, List[str]]` | Structural adjacency for face-to-face conduction |

### Helper Functions

#### `_as_flat_vector(vector) -> np.ndarray`
Normalizes any vector-like input to flat float array (3,). Same as other modules.

#### `_default_face_properties() -> Dict[str, FaceThermalProperties]`
Builds default 6-face thermal properties for 3U CubeSat (10×10×34 cm, long axis = X).
- End caps (+X, -X): 0.010 m²
- Long faces (+Y, -Y, +Z, -Z): 0.034 m²
- Properties: absorptivity, emissivity, cell efficiency, mass, specific heat, conductance to internal

#### `build_bounding_case_config(case: str, base_config: Optional[ThermalMissionConfig]) -> ThermalMissionConfig`
Factory for HOT/COLD/NOMINAL bounding cases:
- **NOMINAL**: Returns base config unchanged
- **HOT**: Perihelion (Jan 3), EOL optics (+0.05 albedo), +0.5W bus dissipation
- **COLD**: Aphelion (Jul 4), BOL optics (-0.05 albedo), -0.5W bus, relay/detection OFF

### Classes

#### `FaceThermalProperties` (dataclass)
Per-face thermal-optical and lumped-capacitance properties.
**Fields**: `label`, `normal_body` (np.ndarray), `area_m2`, `solar_absorptivity`, `solar_absorptivity_eol_delta`, `ir_emissivity`, `cell_efficiency`, `mass_kg`, `specific_heat_j_kgk`, `conductance_to_internal_w_per_k`
**Properties**:
- `thermal_capacitance_j_per_k` = mass × specific_heat
- `effective_solar_absorptivity(mission_life_condition)` — adds EOL delta if "EOL"

#### `ThermalMissionConfig` (dataclass)
**All parameters in one dataclass — no magic numbers.**
Key groups:
- **Simulation**: `sim_duration_hours=24.0`, `step_sec=2.0`, `thermal_substeps_per_step=10`, `progress_report_interval_sec=600.0`
- **Orbit**: `altitude_m=600_000`, `inclination_deg=97.6`, `eccentricity=0.001`, `raan_deg=15.0`
- **Spacecraft**: `mass_kg=3.0`, `drag_coeff=2.2`, `cross_section_m2=0.01`, `srp_coeff=1.8`
- **Integrator**: `min/max_integrator_step_sec`, position/velocity tolerances
- **Thermal Environment**: `solar_constant_w_m2=1361.0`, `deep_space_temp_k=2.7`, `stefan_boltzmann`
- **Earth Albedo/IR**: `use_latitude_dependent_environment=True`, equator/pole values for both
- **Mission Life**: `mission_life_condition="BOL"` ("BOL" or "EOL")
- **Internal Node**: `internal_mass_kg=1.8`, `internal_specific_heat_j_kgk=800.0`
- **Duty-Cycle Power**: `baseline_bus=1.5W`, `charging_bonus=0.4W`, `eclipse_heater=0.8W`, `relay_pass=3.0W/420s`, `detection_window=1.2W/900s`
- **Face-to-Face**: `face_to_face_conductance_w_per_k=0.05`
- **Attitude**: `attitude_mode="LVLH"` ("LVLH" or "TUMBLING"), tumble rate/axis
- **Integration**: `initial_temperature_k=293.15`, `applied_margin_c=10.0`
- **Validation**: `__post_init__` checks all bounds and enum values

#### `OrbitState` (dataclass)
Propagated state per timestep: `absolute_date`, `time_sec`, `lat/lon/alt`, `in_eclipse`, `beta_angle_deg`, `sun_distance_au`, inertial position/velocity.

#### `ThermalSimulator`
Main simulation runner.

**Methods**:
- `__init__(config)` — Initialize Orekit, propagator, face properties, thermal state, Earth view-factor table
- `_init_orekit_frames()` — Time scales, frames, Earth, Sun
- `_build_initial_orbit(raan_deg)` — Keplerian orbit at epoch
- `_build_atmosphere()` — NRLMSISE-00 → DTM2000 fallback chain
- `_build_srp_model()` — SRP with Earth occultation
- `_init_propagator()` — Numerical propagator with gravity (10×10), drag, SRP
- `_compute_beta_angle_deg()` — Beta angle from orbit normal and Sun direction
- `_is_in_eclipse_geometry()` — Conical shadow test (penumbra-aware)
- `_build_orbit_state()` — Convert Orekit state to OrbitState
- `solve_raan_for_target_beta_deg(target_beta_deg, coarse_step_deg=2.0) -> float` — RAAN search for desired beta angle at epoch
- `_build_lvlh_face_normals()` — LVLH frame: +Y=nadir, +Z=sun-side orbit normal, +X=RAM
- `_build_tumbling_face_normals()` — Rigid tumble about fixed inertial axis
- `_build_earth_view_factor_table()` — Numerically integrated flat-plate-to-sphere view factor F(φ) table (0.5° resolution, 200×200 samples)
- `_earth_view_factor(cos_theta_to_nadir)` — Interpolated view factor lookup
- `_albedo_for_latitude()` / `_earth_ir_for_latitude()` — Latitude-dependent environment (quadratic interpolation)
- `_compute_face_fluxes()` — Per-face: solar heat/electrical, albedo, IR, Earth view factor
- `_internal_dissipation_w()` — Synthetic duty-cycle: baseline + eclipse/charging + relay pass + detection window
- `_thermal_derivatives()` — 7-node ODE: absorbed - radiated + conduction (internal + face-to-face)
- `_rk4_thermal_step()` — RK4 integration with constant fluxes per orbit step
- `run(write_outputs=True, verbose=True) -> pd.DataFrame` — Main loop: propagate + thermal sub-steps + logging
- `run_convergence_check(duration_hours=2.0)` — Halve step/double substeps, compare worst-case temps
- `_write_worst_case_summary_csv()` — Per-node hot/cold temps with ±10°C margin
- `_plot_results()` — 4 figures: temperatures, radiation components, orbit context, worst-case bar chart
- `_print_summary()` — Console summary table

### CLI Arguments (`_parse_args()`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--duration-hours` | 24.0 | Simulation duration |
| `--step-sec` | 2.0 | Orbital propagation step (s) |
| `--thermal-substeps` | 10 | RK4 sub-steps per orbit step |
| `--case` | NOMINAL | NOMINAL / HOT / COLD |
| `--attitude-mode` | LVLH | LVLH / TUMBLING |
| `--target-beta-deg` | None | Solve RAAN for target beta angle |
| `--internal-dissipation-w` | None | Override baseline bus dissipation |
| `--convergence-check` | False | Run 2h convergence check before full run |
| `--convergence-only` | False | Only run convergence check |

## Internal Logic

### Thermal Network (7-Node Lumped Capacitance)
**Nodes**: 6 faces (+X, -X, +Y, -Y, +Z, -Z) + INTERNAL (bus)

**Per-Face ODE**:
```
dT_face/dt = (q_solar_heat + q_albedo + q_IR - q_radiated + q_cond_internal + q_cond_neighbors) / C_face
```
- `q_radiated = εσA(T⁴ - T_space⁴)`
- `q_cond_internal = G_internal(T_internal - T_face)`
- `q_cond_neighbors = Σ G_ff(T_neighbor - T_face)` over 4 adjacent faces

**Internal Node ODE**:
```
dT_internal/dt = (P_dissipation - Σ q_cond_internal) / C_internal
```

**Integration**: RK4 with `thermal_substeps_per_step` sub-steps per orbit propagation step. Fluxes held constant across sub-steps; internal dissipation re-evaluated at each RK4 stage.

### Face Fluxes
**Solar**: `q_solar = S × α_eff × A × max(0, n̂·ŝ)` (0 in eclipse)
- Electrical: `q_elec = q_solar × cell_efficiency` (subtracted from heat)
- Heat: `q_heat = q_solar × (1 - cell_efficiency)`

**Albedo**: `q_albedo = A_albedo × S × F_earth × α_eff × A × cos(θ_subsat_sun) × (1 - cell_efficiency)`
- Only when sunlit AND subsatellite point sunlit (`cos_subsat_sun > 0`)

**Earth IR**: `q_IR = EIR(φ) × F_earth × ε × A`

**Earth View Factor**: Numerically integrated table F(φ) where φ = angle between face normal and nadir. More accurate than analytic approximation, handles horizon grazing.

### Bounding Cases
| Parameter | HOT | COLD |
|-----------|-----|------|
| Epoch | Jan 3 (perihelion) | Jul 4 (aphelion) |
| Optics | EOL (α + 0.05 albedo) | BOL (α - 0.05 albedo) |
| Bus dissipation | +0.5W | -0.5W (min 0) |
| Relay/Detection | ON | OFF |

### Attitude Modes
- **LVLH**: Continuous LVLH frame solved every step (+X=RAM, +Y=nadir, +Z=sun-side orbit normal)
- **TUMBLING**: Fixed inertial tumble axis at `tumble_rate_deg_s` — worst-case pre-detumble bound

### Beta-Angle Targeting
`solve_raan_for_target_beta_deg()` scans RAAN 0–360° at epoch using analytic orbit (no propagation), finds RAAN yielding target beta angle. Useful for HOT (β≈90° continuous sun) or COLD (β≈0° max eclipse) case setup.

## Usage Example
```bash
# Nominal 24h LVLH run
python src/ThermalSimulator.py --duration-hours 24 --case NOMINAL --attitude-mode LVLH

# HOT bounding case with beta targeting (continuous sun)
python src/ThermalSimulator.py --case HOT --target-beta-deg 90

# COLD bounding case with tumbling attitude
python src/ThermalSimulator.py --case COLD --attitude-mode TUMBLING

# Convergence check (2h at two resolutions)
python src/ThermalSimulator.py --convergence-check

# Only convergence check
python src/ThermalSimulator.py --convergence-only
```

## Output Files
**CSV** (in `results/thermal/`):
- `ANTARA_Thermal_telemetry.csv` — Per-step: time, orbit state, per-face temps (K/°C), fluxes (W), view factors, internal dissipation
- `ANTARA_Thermal_worst_case_summary.csv` — Per-node: hot/cold temps, times, ±margin temps

**Plots** (in `figures/thermal/`, requires matplotlib):
- `ANTARA_Thermal_temperatures.png` — 6 face + internal temps vs time (eclipse shaded)
- `ANTARA_Thermal_radiation_components.png` — 3-panel: solar heat, albedo, Earth-IR per face
- `ANTARA_Thermal_orbit_context.png` — 3-panel: beta angle, altitude, internal dissipation
- `ANTARA_Thermal_worst_case_summary.png` — Bar chart: hot/cold per node with margin markers

## Notes / TODOs
- **Local paths fallback**: If `paths` import fails, defines local `results_path`/`figures_path`/`init_orekit_data` — allows standalone execution
- **No ADCS/FSM coupling**: Uses synthetic duty-cycle for internal dissipation — "swap `_internal_dissipation_w()` to pull real mode schedule for mission-grade accuracy" (per comment)
- **Face properties**: "representative CubeSat engineering estimates... NOT measured ANTARA hardware data" (TBD refine with CAD/FEA)
- **Eclipse flag**: Thermal uses own `_is_in_eclipse_geometry()` (conical shadow), not Orekit detector
- **Solar distance**: Scales solar constant by true Sun-Earth distance from Orekit ephemeris (captures perihelion/aphelion)
- **Convergence check**: Compares coarse (2s/10 substeps) vs fine (1s/20 substeps) over 2h, reports max ΔT
- **RAAN targeting**: Solves at epoch only; beta angle evolves over mission — not a full mission beta optimization
- **Face-to-face conduction**: Added as "suggestion #5" — 0.05 W/K between adjacent faces
- **Latitude-dependent environment**: "suggestion #6" — quadratic interpolation between equator/pole values
