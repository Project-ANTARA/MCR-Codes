# ANTARA_MCR.py

## File Overview
Monte Carlo RAAN trade study for the ANTARA mission. Runs 365-day numerical propagation for 7 RAAN values (0°–180° in 30° steps) × 2 Monte Carlo runs each, with randomized altitude, drag coefficient, SRP coefficient, battery efficiency, and solar tracking error. Produces 7 output CSVs covering power, battery, CONOPS, orbit classification, drag lifetime, and a master trade matrix, plus statistical summaries with 99% confidence intervals.

## Dependencies
- **Standard Library**: `os`, `csv`, `math`, `time`, `random`, `collections.Counter`
- **External**: `numpy`, `pandas`, `orekit` (Python wrapper)
- **Internal**: `paths` (`init_orekit_data`, `results_dir`)
- **Orekit/Java Modules**: `AbsoluteDate`, `TimeScalesFactory`, `FramesFactory`, `KeplerianOrbit`, `PositionAngleType`, `Constants`, `IERSConventions`, `OneAxisEllipsoid`, `CelestialBodyFactory`, `NadirPointing`, `NumericalPropagator`, `HolmesFeatherstoneAttractionModel`, `ThirdBodyAttraction`, `GravityFieldFactory`, `DragForce`, `IsotropicDrag`, `DTM2000`, `CssiSpaceWeatherData`, `SolarRadiationPressure`, `IsotropicRadiationSingleCoefficient`, `EclipseDetector`, `SpacecraftState`, `Vector3D`, `DormandPrince853Integrator`

## Exports / Public API
This script has no exported functions/classes — it runs as a standalone Monte Carlo campaign when executed.

### Configuration Constants (Module-Level)
| Constant | Value | Description |
|----------|-------|-------------|
| `MISSION_DURATION_DAYS` | 365 | Simulation duration per run |
| `STEP_SIZE` | 300.0 | Propagation step (seconds) |
| `MONTE_CARLO_RUNS` | 2 | Runs per RAAN value |
| `BATTERY_CAPACITY` | 25.0 | Battery capacity (Wh) |
| `RAAN_VALUES` | [0, 30, 60, 90, 120, 150, 180] | RAAN sweep values (deg) |
| `ALTITUDE` | 600000.0 | Nominal altitude (m) |
| `INCLINATION` | 97.4 | Orbit inclination (deg) |
| `ECC` | 0.001 | Eccentricity |
| `MASS` | 4.0 | Spacecraft mass (kg) |
| `CD_BASE` | 2.2 | Baseline drag coefficient |
| `CR_BASE` | 1.2 | Baseline SRP coefficient |
| `FIXED_PANEL_AREA` | 0.030 | Body-mounted panel area (m²) |
| `DEPLOYED_PANEL_AREA` | 0.030 | Per deployed panel area (m²) |
| `NUM_DEPLOYED_PANELS` | 2 | Number of deployed panels |
| `TOTAL_AREA` | 0.076799 | Total cross-section for drag/SRP (m²) |
| `SOLAR_CONSTANT` | 1361.0 | Solar constant (W/m²) |
| `CELL_EFFICIENCY` | 0.28 | Solar cell efficiency (BOL) |
| `SYSTEM_EFF` | 0.513 | End-to-end power system efficiency |
| `LOAD_POWER` | 8.0 | Constant spacecraft load (W) |
| `TRACKING_BIAS` | 2.0 | Mean solar tracking error (deg) |
| `ALTITUDE_SIGMA` | 5000.0 | Altitude randomization sigma (m) |
| `CD_SIGMA` | 0.15 | Drag coefficient sigma |
| `CR_SIGMA` | 0.10 | SRP coefficient sigma |
| `BATTERY_EFF_SIGMA` | 0.03 | Battery efficiency sigma |
| `REENTRY_ALTITUDE_KM` | 300.0 | Reentry termination altitude |
| `Z_99` | 2.576 | 99% confidence interval z-score |

### Helper Functions

#### `safe_normalize(v: Vector3D) -> Vector3D`
Returns unit vector; handles zero-norm case.

#### `check_saa(lat_deg: float, lon_deg: float) -> bool`
SAA bounding box: lat [-50°, 0°], lon [-90°, 40°].

#### `check_gs_bits_goa(lat_deg: float, lon_deg: float) -> bool`
Goa GS bounding box: lat [0°, 20°], lon [60°, 80°].

#### `compute_statistics(data: List[float], metric_name: str) -> Optional[Dict]`
Computes mean, std, variance, min, max, 99% CI for a metric across RAAN values.
Returns dict with keys: Metric, Mean, StdDev, Variance, Min, Max, CI99_Low, CI99_High.

#### `calculate_power(state, eclipse, tracking_sigma, degradation_factor) -> Tuple[float, float, float]`
Computes instantaneous power generation:
- **Fixed panel**: `SOLAR_CONSTANT × FIXED_PANEL_AREA × cos(θ) × CELL_EFF × SYS_EFF × degradation` (θ = sun incidence on -Z body axis)
- **Deployed panels**: `SOLAR_CONSTANT × DEPLOYED_PANEL_AREA × NUM_DEPLOYED × tracking_loss × CELL_EFF × SYS_EFF × degradation`
  - `tracking_loss = cos(radians(random.gauss(TRACKING_BIAS, tracking_sigma)))`
- Returns: `(total_power_W, incidence_angle_deg, tracking_loss)`

## Internal Logic

### Main Loop Structure
```
for each RAAN in RAAN_VALUES:
    for each Monte Carlo run (2):
        Randomize: altitude (±5km), CD (±0.15), CR (±0.10), battery_eff (±0.03), tracking_sigma (uniform 3-8°)
        Build numerical propagator (Dormand-Prince 8/5/3, 5×5 gravity, DTM2000, drag, SRP, Sun/Moon 3-body)
        Initialize battery at full capacity
        For each 300s step over 365 days:
            Propagate orbit
            Compute lat/lon/altitude
            Detect orbit boundary (ascending node: lat sign change - → +)
            Classify orbit type: QUIET / ONLY_SAA / ONLY_GS / MIXED
            Check eclipse (umbra detector)
            Compute power (with 5%/year degradation)
            Update battery (net power × dt × battery_eff, with capacity fade: 0.005% per cycle)
            Track SOC, DoD, min SOC, throughput, cycle count
            Check SAA/GS bounding boxes
            Write per-step rows to 3 CSVs (power, battery, conops)
            Progress print every 200 global steps
        On reentry (alt ≤ 300km) or end: record lifetime, decay rate
        Aggregate per-run metrics (avg power, avg SOC, min SOC, cycles, lifetime)
    Aggregate per-RAAN statistics (mean across MC runs)
    Write master trade matrix row
```

### Battery Model
- **Initial**: `battery_energy = BATTERY_CAPACITY` (full)
- **Per step**: `delta_energy = (power_generated - LOAD_POWER) × (STEP_SIZE/3600) × battery_eff`
- **Capacity fade**: `capacity_remaining = BATTERY_CAPACITY × (1 - 0.00005 × cycle_count)`
- **Cycle count**: `throughput_Wh / (2 × BATTERY_CAPACITY)`
- **Clamp**: `battery_energy ∈ [0, capacity_remaining]`
- **Brownout**: `SOC < 0.20`

### Orbit Classification
At each ascending node crossing (lat crosses 0° northbound):
- `QUIET`: no SAA, no GS in previous orbit
- `ONLY_SAA`: SAA count > 0, GS count = 0
- `ONLY_GS`: SAA count = 0, GS count > 0
- `MIXED`: both > 0

### Output Files (in `results/mcr/`)
| File | Description |
|------|-------------|
| `power_analysis.csv` | Per-step: RAAN, run, time_days, orbit#, power_W, margin_W, incidence_deg, tracking_loss, eclipse, energy_Wh |
| `battery_analysis.csv` | Per-step: RAAN, run, time_days, orbit#, SOC, DoD, energy_Wh, capacity_Wh, cycles, throughput_Wh, brownout |
| `conops_analysis.csv` | Per-step: RAAN, run, time_days, orbit#, lat, lon, eclipse, SAA, GS_visible |
| `orbit_classification.csv` | Per-orbit: RAAN, run, orbit#, type, SAA_count, GS_count, period_min |
| `drag_lifetime_analysis.csv` | Per-run: RAAN, lifetime_days, lifetime_years, decay_km_day, final_alt_km |
| `raan_master_trade_matrix.csv` | Per-RAAN: RAAN, avg_power, avg_soc, min_soc, battery_cycles, lifetime_years, quiet/only_gs/only_saa/mixed orbit counts |
| `mission_statistics.csv` | 99% CI statistics across RAANs for: avg power, avg SOC, min SOC, battery cycles, lifetime years |

## Usage Example
```bash
# Run from repository root (takes hours)
python src/ANTARA_MCR.py
```
Output: `results/mcr/*.csv` (7 files)

## Notes / TODOs
- **Runtime**: ~hours (7 RAANs × 2 runs × 365 days @ 300s = ~2M propagation steps)
- **No elevation mask**: GS visibility uses simple lat/lon box, not actual elevation angle
- **SAA simplification**: Rectangular bounding box, not flux-based contour
- **Attitude**: Fixed nadir-pointing throughout (no ADCS model, no slews)
- **Degradation**: Linear 5%/year solar array degradation applied uniformly
- **Battery fade**: Simplified 0.005% capacity loss per equivalent full cycle
- **Progress tracking**: Global step counter with ETA estimation
- **File handles**: All 6 CSVs opened at start, closed at end — risk of data loss if crash mid-run
- **Statistics**: 99% CI computed across RAAN means (n=7), not across MC runs
- **Fixed orbit period**: Orbit classification writes hardcoded 97.0 min period