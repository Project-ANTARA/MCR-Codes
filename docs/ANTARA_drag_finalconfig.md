# ANTARA_drag_finalconfig.py

## File Overview
Orbital lifetime analysis for the ANTARA 3U CubeSat final configuration. Computes drag-limited lifetime vs. initial altitude (500–600 km) for the "1 fixed + 2 deployed" solar array configuration (0.076799 m²) under solar maximum conditions with tumbling-averaged cross-section. Uses Orekit numerical propagation with DTM2000 atmosphere (solar max epoch) and RK4 integration at 1-day steps.

## Dependencies
- **Standard Library**: `math`
- **External**: `orekit` (Python wrapper)
- **Internal**: `paths` (`init_orekit_data`)
- **Orekit/Java Modules**: DTM2000 atmosphere (CSSI space weather), drag force, Holmes-Featherstone gravity (2×0), RK4 integrator, Keplerian orbit

## Exports / Public API

### Module Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `MU` | `Constants.WGS84_EARTH_MU` | Earth gravitational parameter |
| `RE` | `Constants.WGS84_EARTH_EQUATORIAL_RADIUS` | Earth equatorial radius |
| `MASS` | 4.0 kg | Spacecraft mass |
| `CD` | 2.2 | Drag coefficient |
| `TUMBLING_FACTOR` | 1.5 | Tumbling-averaged area multiplier |
| `END_ALT` | 450000.0 m | Reentry termination altitude |
| `areas` | `{"1 fixed + 2 deployed": 0.076799}` | Configuration cross-sections |

### Functions

#### `lifetime(area: float, start_alt_m: float) -> float`
Computes orbital lifetime (seconds) for given cross-section and initial altitude.
- **Epoch**: Jan 1, 2000, 12:00 UTC (solar max for Cycle 23 proxy)
- **Orbit**: Circular (ecc=1e-4), 98° inclination, specified altitude
- **Atmosphere**: DTM2000 with CSSI space weather (solar max at epoch)
- **Drag**: `A_eff = TUMBLING_FACTOR × area`, `CD = 2.2`
- **Gravity**: 2×0 zonal harmonics only
- **Integrator**: Classical RK4, fixed 86400s (1 day) step
- **Termination**: Altitude < 450 km or 30-year cap
- **Returns**: Duration in seconds from start date

### Main Execution
```
for altitude_km in range(500, 601, 10):
    T = lifetime(0.076799, altitude_km × 1000)
    print altitude_km, T in years
```

## Internal Logic

### Lifetime Calculation
1. Initialize orbit at `start_alt_m` with solar max epoch (2000-01-01)
2. Build propagator with drag (tumbling-averaged area) + 2×0 gravity
3. Propagate in 1-day steps using RK4
4. Check altitude after each step
5. Return elapsed time when altitude < 450 km
6. Cap at 30 years (10,950 days)

### Tumbling Factor
`TUMBLING_FACTOR = 1.5` multiplies the geometric cross-section to approximate worst-case tumbling drag (average projected area over random tumble). For a 3U CubeSat, this represents the effective area seen by the flow during uncontrolled rotation.

## Usage Example
```bash
# Run from repository root
python src/ANTARA_drag_finalconfig.py
```
Output: Console table of Altitude (km) vs Lifetime (years) for the single configuration.

## Notes / TODOs
- **Solar max epoch**: Uses year 2000 (Cycle 23 peak) as proxy — not Cycle 25 actual peak (~2024-2025)
- **Single configuration**: Only "1 fixed + 2 deployed" area tested
- **RK4 integrator**: Fixed 1-day step — coarse but fast for lifetime trends
- **2×0 gravity**: Only J2 zonal harmonic — no tesseral, no 3-body, no SRP
- **No attitude model**: Tumbling factor approximates attitude-averaged drag
- **CSSI space weather**: Uses default supported names (F10.7/Ap proxies for year 2000)
- **Output**: Console only — no CSV/file export
- **30-year cap**: Arbitrary limit for very high altitudes
- **Altitude range**: 500–600 km in 10 km steps — could extend lower/higher