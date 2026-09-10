# ANTARA_Science_Sim.py

## File Overview
Science payload simulation for the ANTARA mission with detailed proton flux modeling. Extends the high-fidelity simulation framework with: IGRF magnetic field model (via pyIGRF), AP9/AE9 radiation belt environment (via spacepy.irbempy), multi-layer detector physics (geometric factor, energy-dependent efficiency, dead time, dark counts), event-driven histogram packetization, and SAA flux depth scaling. Includes full ADCS, EPS, FSM, telemetry, and plotting (9-panel suite + science-specific plots). Configurable via CLI arguments.

**Core models** (`MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `IllegalFSMTransitionError`, `_as_flat_vector`, `BODY_AXIS_VECTORS`) are now imported from the shared **`antara_models.py`** module (Orekit-free, numpy-only). This enables lightweight Monte Carlo scripts (`leop_detumble_mc.py`, validation tests) to reuse the same models without initializing the Orekit JVM.

## Dependencies
- **Standard Library**: `math`, `argparse`, `json`, `os`, `pickle`, `time`, `dataclasses`, `enum`, `pathlib`, `typing`, `datetime`
- **External**: `numpy`, `pandas`, `orekit`, `shapely`
- **Optional External** (guarded with try/except): `matplotlib`, `cartopy`, `pyIGRF` (`ppigrf`), `spacepy.irbempy`
- **Internal**: `paths` (`init_orekit_data`, `results_path`, `figures_path`), `antara_models` (`MissionConfig`, `ADCSModel`, `SpacecraftFSM`, `SunSubmode`, `IllegalFSMTransitionError`, `_as_flat_vector`, `BODY_AXIS_VECTORS`)
- **Orekit/Java Modules**: Extensive (propagation, forces, frames, orbits, time, bodies, atmosphere, events)

## Exports / Public API

### Module Constants
| Constant | Type | Description |
|----------|------|-------------|
| `BODY_AXIS_VECTORS` | `Dict[str, np.ndarray]` | Canonical body axes (re-exported from `antara_models`) |
| `_MATPLOTLIB_AVAILABLE` | `bool` | Matplotlib import status |
| `_CARTOPY_AVAILABLE` | `bool` | Cartopy import status |
| `_PYIGRF_AVAILABLE` | `bool` | pyIGRF import status |
| `_SPACEPY_AVAILABLE` | `bool` | spacepy import status |

### Helper Functions

#### `_as_flat_vector(vector) -> np.ndarray`
Normalizes any vector-like input to flat float array (3,). **Re-exported from `antara_models`** — this version additionally handles Orekit `Vector3D` objects.

#### `compute_detector_geometric_factor_cm2sr(theta_max_deg: float) -> float`
Computes geometric factor G = 2π(1 - cos(θ_max)) for a conical FOV.

### Classes

#### `SunSubmode(str, Enum)`
Power submodes within SUN_POINTING: CHARGING, PASSIVE, ECLIPSE, NONE. **Re-exported from `antara_models`**.

#### `MissionConfig` (dataclass)
**All mission parameters** — **re-exported from `antara_models`** with identical definition. Validation in `__post_init__` catches config errors early.

#### `OrbitState` / `AttitudeCommand` (dataclasses)
Same structure as Master. Uses Orekit `AbsoluteDate` and `Vector3D`.

#### `ADCSModel`
Attitude determination & control with quaternion dynamics, B-dot, wheel damping, saturation. **Re-exported from `antara_models`** with bug fixes:
- Bug 11 fix: explicit re-normalization after quaternion integration step
- Bug fix in `wheel_damp_control`: per-axis inertia-scaled damping (k_damp = 0.4/dt)
- `_enforce_quaternion_continuity` tracks corrections by call site

#### `EPSModel`
Similar to Master with `payload_coin_logic_power_w` added to loads.

#### `SAAModel`
Shapely polygon, entry/exit detection, data accumulation.

#### `GroundStationModel`
Supports two ground stations (GS_INDIA + optional GS2_SVALBARD).

#### `IGRFModel`
**NEW** — Wrapper for pyIGRF magnetic field calculations.
- `get_field(lat_deg, lon_deg, alt_km, year) -> Tuple[float, float, float]` — Returns (B_north, B_east, B_down) in nT
- Falls back to dipole model if pyIGRF unavailable

#### `ProtonEnvironmentModel`
**NEW** — Proton flux modeling with AP9 baseline + SAA depth + asymmetries.
- `get_flux(lat_deg, lon_deg, alt_km, saa_distance_frac, storm_flag) -> np.ndarray[5]` — Differential flux per energy channel
- Flux = AP9_baseline × SAA_depth_factor × EW_asymmetry × storm_multiplier
- SAA depth factor: linear interpolation between min/max based on normalized distance to SAA polygon
- EW asymmetry: sinusoidal modulation by longitude

#### `DetectorLayerSpec` (dataclass)
Detector layer specification: `name`, `thickness_um`, `material`, `threshold_mev`, `geometric_factor_cm2sr`, `energy_range_mev`

#### `PayloadModel`
**NEW** — Detailed proton detector simulation.
- **Attributes**: `config`, `detector_layers` (list of `DetectorLayerSpec`), `histogram_buffers` (per-channel counts), `total_counts`, `livetime_fraction`, `packets_generated`, `data_volume_mb`
- **Methods**:
  - `step(dt_sec, flux_per_channel, orbit_state, mode, battery_wh) -> Tuple[float, float]` — Accumulate counts, handle dead time, return (counts_this_step, data_mb_generated)
  - `flush_histogram(orbit_state, mode) -> Dict` — Generate packet: header + 64 bins × 4 bytes per channel
  - `compute_livetime(dt_sec, flux_per_channel) -> float` — Paralyzable dead time model

#### `CommunicationsModel`
Extended with dual-GS support and event-driven histogram downlink.

#### `IllegalFSMTransitionError(RuntimeError)`
Custom exception for FSM transition violations. **Re-exported from `antara_models`**.

#### `SpacecraftFSM`
7-state finite state machine with debounced fault logic. **Re-exported from `antara_models`** with additional `SAFE` exit condition: `battery_wh >= safe_exit_battery_fraction × capacity`, `fault_debounce_samples` / `recovery_debounce_samples` for hysteresis, `target_mode` tracking for slew coordination.

#### `TelemetryLogger`
Extended telemetry columns including science-specific fields (flux, counts, livetime, packet info, IGRF B-field).

#### `ANTARAScienceSimulator`
Top-level runner with science-specific logic.
**Key Additional Methods**:
- `_init_igrf()` — Initialize IGRF model
- `_init_proton_environment()` — Initialize proton model
- `_init_payload()` — Initialize detector layers (5 layers, 5 energy channels)
- `_compute_saa_distance_frac(lat, lon) -> float` — Normalized distance to SAA polygon (0=center, 1=boundary)
- `_update_science_payload(...)` — Step detector model, accumulate histograms
- `_flush_science_packet(...)` — Event-driven packet flush on SAA exit or interval timeout
- `_plot_science_results(...)` — Science-specific plots: proton flux map, detector counts, histogram timeline

**CLI Arguments** (`_parse_args()`):
- `--duration-hours` (default 24)
- `--step-sec` (default 2.0)
- `--raan-deg` (default 15.0)
- `--battery-wh` (default 25.0)
- `--output-dir` (default from paths)
- `--no-plots` (disable plotting)
- `--save-pickle` (save telemetry DataFrames as pickle)

## Internal Logic

### Simulation Loop (similar to Master with science extensions)
1. Initialize IGRF, proton environment, payload detector
2. Per timestep:
   - Propagate orbit, build OrbitState
   - Compute SAA distance fraction for flux scaling
   - Run FSM evaluation (with SAFE exit on battery recovery)
   - Update attitude guidance (LVLH-fixed frame in all operational modes)
   - Step ADCS (detumble or operational)
   - Compute power (EPS), step battery
   - **Science payload step**: Get proton flux for current position → accumulate counts in histogram buffers with dead time
   - **Event-driven flush**: On SAA exit or 60s interval timeout → generate packet, add to storage
   - GS visibility (dual station), downlink if RELAY mode
   - Log telemetry (extended columns)
   - Physics sanity checks
   - Progress reporting

### Proton Flux Model
```
flux[channel] = ap9_baseline_flux[channel]
              × saa_depth_factor(distance_to_saa)
              × (1 + ew_asymmetry_amplitude × sin(longitude))
              × (storm_time_flux_multiplier if storm_flag else 1.0)
```
- `saa_depth_factor`: linear from `saa_depth_flux_min` (boundary) to `saa_depth_flux_max` (center)

### Detector Physics
- **Geometric factor**: G = 2π(1 - cos(θ_max)) per layer
- **Count rate**: flux × G × efficiency × ΔE (energy bin width)
- **Dead time**: Paralyzable model — `livetime = exp(-rate × dead_time)`
- **Dark counts**: Added per channel at `detector_dark_count_rate_hz`
- **Coincidence**: Optional anti-coincidence logic between layers

### Packetization
- **Event-driven**: Flush on SAA pass closure (primary) or 60s timeout (safety)
- **Packet size**: 5 channels × 64 bins × 4 bytes + 16 byte header = 1,296 bytes
- **Downlink**: 19.2 kbps → ~2.4 KB/s, can downlink ~1 packet per second

## Usage Example
```bash
# Run from repository root
python src/ANTARA_Science_Sim.py --duration-hours 24 --step-sec 2.0

# With custom RAAN and battery
python src/ANTARA_Science_Sim.py --raan-deg 60 --battery-wh 30

# Disable plots (faster, no matplotlib/cartopy needed)
python src/ANTARA_Science_Sim.py --no-plots

# Save DataFrames as pickle for later analysis
python src/ANTARA_Science_Sim.py --save-pickle
```

## Output Files (in `results/science/` and `figures/science/`)
- `science_telemetry.csv` — Per-step telemetry with science columns
- `science_packets.csv` — Generated histogram packets
- `science_saa_passes.csv` — SAA passes with flux integrals
- `science_gs_contacts.csv` — Dual GS contacts
- `science_mode_transitions.csv` — FSM transitions
- `science_slew_events.csv` — Slew events
- `science_detumble_events.csv` — Detumble events
- `ANTARA_Science_mission_overview.png` — 9-panel PDR plot
- `ANTARA_Science_proton_analysis.png` — Proton flux map, detector counts
- `ANTARA_Science_attitude_diagnostics.png` — Pointing error diagnostics

## Notes / TODOs
- **Massive file**: 6247 lines — significant duplication with `ANTARA_Master.py` (ADCS, EPS, FSM, telemetry, plotting)
- **Optional dependencies**: pyIGRF and spacepy provide enhanced physics but simulation runs with fallbacks (dipole B-field, no AP9)
- **LVLH-fixed attitude**: Unlike Master, all operational modes (SUN_POINTING, DETECTION, RELAY) hold continuous LVLH frame — no active tracking
- **Event-driven packets**: Novel approach tying readout to spatial cell crossings rather than fixed timer
- **CLI configurable**: More flexible than hardcoded Master/MCR scripts
- **Bug fixes**: Includes fixes for quaternion continuity (Bug 11) and wheel damping instability
- **Validation**: Comprehensive `__post_init__` in MissionConfig catches config errors early
- **Dual GS**: Svalbard station optional (disabled by default) for high-latitude contacts