# ANTARA_FINAL_POWERGEN.py

## Current scope (2026-09-12)

Source: [ANTARA_FINAL_POWERGEN.py](../src/ANTARA_FINAL_POWERGEN.py). This standalone panel trade uses its own `NadirPointing` provider, fixed -Z panel, and independent wing tracking assumptions. Do not apply Science's +Y relay mapping or `MissionConfig` values to it. The sweep includes both 0° and 360°: 25 samples representing 24 unique RAAN directions, with random tracking realizations.

Setup and navigation: [repository README](../README.md) · [documentation index](README.md). Supporting study results were not rerun as part of this documentation review.

## File Overview
RAAN optimization analysis for solar power generation. Runs short (24-hour) numerical propagation for RAAN values 0°–360° in 15° steps, computing average/peak power and eclipse fraction for a 3U CubeSat with 1 fixed zenith panel (-Z) + 2 deployed wings. Identifies optimal RAAN for maximum average power at equinox (March 20).

## Dependencies
- **Standard Library**: `os`, `math`, `random`
- **External**: `orekit` (Python wrapper)
- **Internal**: `paths` (`init_orekit_data`)
- **Orekit/Java Modules**: Numerical propagation (Dormand-Prince 8/5/3), forces (Holmes-Featherstone 5×5, Harris-Priester drag, SRP, Sun/Moon 3-body), nadir-pointing attitude, eclipse detector

## Exports / Public API

### Module Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `SIM_DURATION` | 86400.0 (24h) | Simulation duration (seconds) |
| `STEP_SIZE` | 30.0 | Propagation step (seconds) |
| `panel_areas` | Dict | Fixed panel areas: only -Z = 0.021 m² |
| `DEPLOYED_PANEL_AREA` | 0.018065 | Per deployed panel area (m²) |
| `NUM_DEPLOYED_PANELS` | 2 | Number of deployed wings |
| `SOLAR_CONSTANT` | 1361.0 | Solar constant (W/m²) |
| `CELL_EFFICIENCY` | 0.30 | Solar cell efficiency (BOL) |
| `SYSTEM_EFF` | 0.72675 | System efficiency = 0.85 × 0.90 × 0.95 |
| `TRACKING_ERROR_SIGMA` | 5.0 | Tracking error std dev (deg) |
| `TRACKING_BIAS` | 2.0 | Tracking error mean bias (deg) |
| `ALTITUDE` | 600000.0 | Orbit altitude (m) |
| `INC` | 97.4 | Inclination (deg) |

### Helper Functions

#### `safe_normalize(v: Vector3D) -> Vector3D`
Manual Vector3D normalization (avoids zero-norm crash).

#### `calculate_power(state, eclipse) -> float`
Instantaneous power generation:
- **Fixed panel (-Z)**: `SOLAR_CONSTANT × 0.021 × cos(θ) × CELL_EFF × SYS_EFF` (if cos > 0)
- **Deployed wings (2×)**: `SOLAR_CONSTANT × 0.018065 × 2 × tracking_loss × CELL_EFF × SYS_EFF`
  - `tracking_loss = cos(radians(random.gauss(TRACKING_BIAS, TRACKING_ERROR_SIGMA)))`
- Returns 0.0 in eclipse

#### `run_simulation(test_raan: float) -> Tuple[float, float, float]`
Runs 24h propagation for given RAAN.
- **Epoch**: March 20, 2026 (equinox)
- **Orbit**: 600km, 97.4°, ecc=0.001, specified RAAN
- **Forces**: 5×5 gravity, Harris-Priester drag (Cd=2.2, A=0.03), SRP (Cr=1.2, A=0.03), Sun/Moon 3-body
- **Attitude**: Nadir-pointing (fixed)
- **Returns**: `(avg_power_W, max_power_W, eclipse_fraction_percent)`

## Internal Logic

### Main Loop
```
for RAAN in range(0, 361, 15):
    avg_p, max_p, ecl_frac = run_simulation(RAAN)
    print formatted row
    track best RAAN by max avg_power
```

### Power Model Details
- **Fixed panel**: Body-mounted on -Z in this study's attitude frame. Power uses the positive Sun projection on that axis; this is not Science's panel/relay frame implementation.
- **Deployed wings**: Assumed to track Sun with Gaussian pointing error (2° bias, 5° σ). No cosine loss from body attitude — wings actively track.
- **System efficiency chain**: 0.85 (MPPT) × 0.90 (harness) × 0.95 (thermal/aging) = 0.72675

### Eclipse
Uses Orekit `EclipseDetector` with umbra only.

## Usage Example
```bash
# Run from repository root
python src/ANTARA_FINAL_POWERGEN.py
```
Output: Console table with RAAN vs Avg Power / Max Power / Eclipse %, plus optimal RAAN.

## Notes / TODOs
- **Short duration**: 24h at equinox — not representative of full-year beta angle variation
- **Fixed attitude**: Nadir-pointing only; no Sun-tracking body attitude (deployed wings track independently)
- **Harris-Priester atmosphere**: Older model vs DTM2000/NRLMSISE-00 used elsewhere
- **EIGEN5C gravity**: Different from WGS84 used in other scripts
- **No battery/SOC model**: Pure power generation analysis
- **Random tracking error**: Different error realization per step (not per orbit) — may overestimate variability
- **No output files**: Results only printed to console
- **Hardcoded equinox**: March 20, 2026 — not configurable
- **Panel areas**: -Z fixed (0.021 m²), 2 deployed (0.018065 m² each) — specific hardware config
