# ANTARA_avg.py

## File Overview
Monte Carlo battery sizing study for the ANTARA mission. Runs 365-day numerical propagation for a single RAAN (75°) with 3 Monte Carlo runs, testing battery capacity cases (currently only 25 Wh). Randomizes altitude, drag coefficient, SRP coefficient, battery efficiency, and solar tracking error. Produces per-run timeseries CSVs and statistical summaries with 99% confidence intervals.

## Dependencies
- **Standard Library**: `os`, `csv`, `math`, `time`, `random`, `statistics`
- **External**: `numpy`, `pandas`, `orekit` (Python wrapper)
- **Internal**: `paths` (`init_orekit_data`, `results_dir`)
- **Orekit/Java Modules**: Same as `ANTARA_MCR.py` (numerical propagation with DTM2000, drag, SRP, 3-body, eclipse detector)

## Exports / Public API
This script has no exported functions/classes — it runs as a standalone Monte Carlo campaign when executed.

### Configuration Constants (Module-Level)
| Constant | Value | Description |
|----------|-------|-------------|
| `MISSION_DURATION_DAYS` | 365 | Simulation duration per run |
| `STEP_SIZE` | 300.0 | Propagation step (seconds) |
| `MONTE_CARLO_RUNS` | 3 | Runs per battery case |
| `BATTERY_CASES` | [25.0] | Battery capacities to test (Wh) — 30 Wh commented in header but not in list |
| `ALTITUDE` | 600000.0 | Nominal altitude (m) |
| `INCLINATION` | 97.4 | Orbit inclination (deg) |
| `ECC` | 0.001 | Eccentricity |
| `RAAN` | 75.0 | Fixed RAAN (deg) |
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
| `TRACKING_ERROR_SIGMA` | 5.0 | Unused (tracking_sigma randomized per run) |
| `TRACKING_BIAS` | 2.0 | Mean solar tracking error (deg) |
| `ALTITUDE_SIGMA` | 5000.0 | Altitude randomization sigma (m) |
| `CD_SIGMA` | 0.15 | Drag coefficient sigma |
| `CR_SIGMA` | 0.10 | SRP coefficient sigma |
| `BATTERY_EFF_SIGMA` | 0.03 | Battery efficiency sigma |
| `Z_99` | 2.576 | 99% confidence interval z-score |

### Helper Functions

#### `safe_normalize(v: Vector3D) -> Vector3D`
Returns unit vector; handles zero-norm case.

#### `check_saa(lat_deg: float, lon_deg: float) -> bool`
SAA bounding box: lat [-50°, 0°], lon [-90°, 40°].

#### `check_gs_bits_goa(lat_deg: float, lon_deg: float) -> bool`
Goa GS bounding box: lat [0°, 20°], lon [60°, 80°].

#### `compute_statistics(data: List[float], metric_name: str) -> Dict`
Computes mean, std, variance, min, max, 99% CI for a metric across MC runs.
Returns dict with keys: Metric, Mean, StdDev, Variance, Min, Max, CI99_Low, CI99_High.

#### `calculate_power(state, eclipse, tracking_sigma) -> float`
Computes instantaneous power generation (simplified vs MCR):
- **Fixed panel**: `SOLAR_CONSTANT × FIXED_PANEL_AREA × cos(θ) × CELL_EFF × SYS_EFF` (no degradation factor)
- **Deployed panels**: `SOLAR_CONSTANT × DEPLOYED_PANEL_AREA × NUM_DEPLOYED × tracking_loss × CELL_EFF × SYS_EFF`
  - `tracking_loss = cos(radians(random.gauss(TRACKING_BIAS, tracking_sigma)))`
- Returns `total_power` (W); returns 0.0 if eclipse

## Internal Logic

### Main Loop Structure
```
for each battery_capacity in BATTERY_CASES:
    Open timeseries CSV for this battery case
    For each Monte Carlo run (3):
        Randomize: altitude (±5km), CD (±0.15), CR (±0.10), battery_eff (±0.03), tracking_sigma (uniform 3-8°)
        Build numerical propagator (Dormand-Prince 8/5/3, 5×5 gravity, DTM2000, drag, SRP, Sun/Moon 3-body)
        Initialize battery at full capacity (battery_capacity)
        For each 300s step over 365 days:
            Propagate orbit
            Compute lat/lon/altitude
            Check eclipse (umbra detector)
            Compute power (calculate_power)
            Update battery: net_power × dt_hr × battery_eff, clamp [0, battery_capacity]
            Track SOC, min SOC, cumulative power/SOC
            Check SAA/GS bounding boxes
            Write per-step row to timeseries CSV
            Progress print every 500 global steps
        Compute run metrics: avg_power, avg_soc, eclipse_fraction, min_soc, lifetime_estimate
    Compute 99% CI statistics across MC runs for 5 metrics
    Write statistics CSV for this battery case
    Accumulate into all_summary_rows
Write combined mission_summary.csv across all battery cases
```

### Battery Model
- **Initial**: `battery_energy = battery_capacity` (full)
- **Per step**: `delta_energy = (power_generated - LOAD_POWER) × (STEP_SIZE/3600) × battery_eff`
- **Clamp**: `battery_energy ∈ [0, battery_capacity]` (no capacity fade model unlike MCR)
- **SOC**: `battery_energy / battery_capacity`

### Lifetime Estimate
Crude linear approximation: `lifetime_years = (final_altitude_km - 450.0) / 25.0`
- Not a true drag lifetime; just altitude-based heuristic

### Output Files (in `results/avg/`)
| File | Description |
|------|-------------|
| `mission_timeseries_25Wh.csv` | Per-step: run_id, time_days, altitude_km, lat, lon, eclipse, power_W, SOC, battery_Wh, GS_visible, SAA |
| `statistics_25Wh.csv` | 99% CI stats for: avg power, eclipse fraction, min SOC, avg SOC, lifetime estimate |
| `mission_summary.csv` | Combined summary across all battery cases (Battery_Wh + metric stats) |

## Usage Example
```bash
# Run from repository root
python src/ANTARA_avg.py
```
Output: `results/avg/mission_timeseries_25Wh.csv`, `results/avg/statistics_25Wh.csv`, `results/avg/mission_summary.csv`

## Notes / TODOs
- **Battery cases**: Header mentions 25 Wh + 30 Wh but `BATTERY_CASES = [25.0]` only — add 30.0 to test both
- **No degradation**: Unlike MCR, no solar array degradation factor applied
- **No capacity fade**: Battery capacity constant (no cycle-based degradation)
- **No orbit classification**: Unlike MCR, doesn't classify orbits by SAA/GS type
- **No drag lifetime**: No reentry detection or decay rate calculation
- **Fixed RAAN**: Single RAAN (75°) vs MCR's 7-value sweep
- **Simplified power model**: No degradation factor, no incidence angle returned
- **Progress tracking**: Global step counter with ETA (every 500 steps)
- **Statistics**: 99% CI computed across MC runs (n=3) — very small sample
- **File handles**: Timeseries CSV opened per battery case with `with` block (safe)
- **Duplicate code**: Significant overlap with `ANTARA_MCR.py` — could share common propagation/power logic