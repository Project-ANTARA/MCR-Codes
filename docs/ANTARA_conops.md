# ANTARA_conops.py

## Current scope (2026-09-12)

Source: [ANTARA_conops.py](../src/ANTARA_conops.py). This script executes its analytical timeline at module scope; importing it can start a run and write output. Run it from the repository root after the [Orekit setup](../README.md#orekit-data). It uses bounding-box visibility rather than Science's full attitude/recovery logic, so successful CONOPS execution is only a basic environment check.

Setup and navigation: [repository README](../README.md) · [documentation index](README.md). Supporting study results were not rerun as part of this documentation review.

## File Overview
Fast analytical propagation script for CONOPS (Concept of Operations) timeline generation. Uses Orekit's Eckstein-Hechler analytical propagator (J2-perturbed) to simulate 30 days of orbit at 60-second steps, producing a CSV log of orbital position, eclipse status, SAA crossings, and ground station visibility. This is the quickest simulation entry point for verification.

## Dependencies
- **Standard Library**: `math`, `csv`
- **External**: `orekit` (VM init + Orekit Java API via `orekit` Python wrapper)
- **Internal**: `paths` (`init_orekit_data`, `results_path`)
- **Orekit Modules**: `FramesFactory`, `OneAxisEllipsoid`, `CelestialBodyFactory`, `AbsoluteDate`, `TimeScalesFactory`, `KeplerianOrbit`, `PositionAngleType`, `Constants`, `IERSConventions`, `EcksteinHechlerPropagator`, `EclipseDetector`

## Exports / Public API
The script defines the helpers below and runs its simulation at module scope; it is not a side-effect-free library import.

### Configuration Constants (Module-Level)
| Constant | Value | Description |
|----------|-------|-------------|
| `start_date_py` | `AbsoluteDate(2023, 1, 1, 12, 0, 0.0, UTC)` | Simulation epoch |
| `duration` | `2592000.0` (30 days) | Total simulation time in seconds |
| `step` | `60.0` | Propagation step size in seconds |
| `output_filename` | `results/conops/ANTARA_conops_log.csv` | Output CSV path |

### Helper Functions

#### `check_saa(lat_deg: float, lon_deg: float) -> bool`
Returns `True` if the given lat/lon falls within the South Atlantic Anomaly bounding box.
- **SAA Region**: Lat [-50°, 0°], Lon [-90°, 40°]
- **Note**: Simplified rectangular approximation; not a precise flux contour

#### `check_gs_bits_goa(lat_deg: float, lon_deg: float) -> bool`
Returns `True` if the given lat/lon falls within the Goa ground station visibility bounding box.
- **GS Region**: Lat [0°, 20°], Lon [60°, 80°]
- **Note**: Simplified rectangular approximation; no elevation mask applied

## Internal Logic
1. **Initialization**: Calls `orekit.initVM()` (must be first), then `init_orekit_data()` to configure Orekit data path
2. **Frames**: Sets up EME2000 inertial frame and ITRF2010 Earth-fixed frame with WGS84 ellipsoid
3. **Orbit**: Defines a 7000 km semi-major axis (~622 km altitude), 98° inclination, near-circular orbit with RAAN=90°
4. **Propagator**: Uses `EcksteinHechlerPropagator` — analytical J2-perturbed propagator (fast, no numerical integration)
5. **Detectors**: Creates umbra-only `EclipseDetector` for sunlight/eclipse determination
6. **Simulation Loop** (30 days × 60s = 43,200 steps):
   - Propagate to current time
   - Convert inertial position to lat/lon via Earth body transform
   - Detect ascending node crossing (lat sign change - → +) for orbit counting
   - Evaluate eclipse detector `g()` function (negative = in umbra)
   - Check SAA and GS bounding boxes
   - Write row to CSV
7. **Output**: CSV with columns: Time(s), Latitude, Longitude, Orbit#, Eclipse, In SAA, GS(Goa)

## Usage Example
```bash
# Run from repository root
python src/ANTARA_conops.py
```
Expected output: `results/conops/ANTARA_conops_log.csv` with ~43,200 rows.

## Notes / TODOs
- **No elevation mask**: Ground station visibility uses simple lat/lon bounding box, not actual elevation angle calculation
- **Simplified SAA**: Rectangular approximation; real SAA is irregular and altitude-dependent
- **Analytical propagator**: Eckstein-Hechler only models J2 perturbation — no drag, SRP, or third-body effects. Suitable for short-term CONOPS but not long-term lifetime analysis
- **Fixed orbit parameters**: Altitude, inclination, RAAN hardcoded; modify module constants to change
- **Progress print**: Every 600s (10 minutes sim time) prints to stdout
- **Verification script**: This is the recommended first-run test per AGENTS.md
