# ANTARA_pointing_budget.py

## Current scope (2026-09-12)

Source: [ANTARA_pointing_budget.py](ANTARA_pointing_budget.py). The code explicitly uses `Vector3D.PLUS_K` (+Z) as the antenna and Orekit `NadirPointing`. This is a standalone convention, not Science's +Y relay convention. Do not interpret its plot as validation of Science's moving-LOS acquisition or saturation recovery. See the [Science attitude guide](../ditl/ANTARA_Science_Sim.md#attitude-geometry) for that model.

Setup and navigation: [repository README](../../../README.md) · [documentation index](../../../docs/README.md). Supporting study results were not rerun as part of this documentation review.

## File Overview
Ground station contact pointing budget analysis for ANTARA. Propagates 1-day Keplerian orbit (analytical, no perturbations) with nadir-pointing attitude, computes off-boresight angle between +Z body axis (antenna boresight) and ground station LOS vector during passes above 10° elevation mask. Outputs pass statistics and off-boresight angle plot.

## Dependencies
- **Standard Library**: `math`
- **External**: `matplotlib`, `numpy`, `orekit` (Python wrapper)
- **Internal**: `paths` (`init_orekit_data`, `figures_path`)
- **Orekit/Java Modules**: Keplerian propagator, nadir-pointing attitude, topocentric frame, EGM96 gravity

## Exports / Public API
This script has no exported functions/classes — runs as standalone analysis.

### Configuration Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `simulation_duration` | 86400.0 s (1 day) | Simulation length |
| `step` | 10.0 s | Propagation step |
| `start_date` | 2026-01-01 12:00 UTC | Epoch |
| `altitude` | 600000.0 m | Orbit altitude |
| `inclination` | 97.4° | SSO inclination |
| `RAAN` | 0.0° | Fixed RAAN |
| `station` | 15°N, 74°E, 30m | Goa ground station |
| `elevation_mask` | 10.0° | Minimum elevation for contact |
| `boresight` | +Z body axis | Antenna pointing direction |

### Internal Logic

#### Simulation Loop (1 day, 10s steps = 8,640 steps)
1. Propagate Keplerian orbit (analytical, no drag/J2 evolution)
2. Compute GS elevation via `TopocentricFrame.getElevation()`
3. **If elevation > 10°**:
   - Get GS position in inertial frame
   - Compute LOS vector: `GS_inertial - SAT_inertial`
   - Transform LOS to body frame via attitude rotation
   - Compute off-boresight angle: `angle(+Z_body, LOS_body)`
   - Track contact start/end, accumulate durations
   - Track maximum off-boresight angle
4. **Else**: Close contact if was in contact

#### Outputs
- Console: Number of passes, total access time (min), avg pass duration (min), max off-boresight angle (deg)
- Plot: `figures/pointing/boresight1.png` — Off-boresight angle vs time (minutes) during passes

## Usage Example
```bash
python src/simulations/pointing/ANTARA_pointing_budget.py
```

## Notes / TODOs
- **Analytical propagator**: `KeplerianPropagator` — no J2, drag, SRP → orbit doesn't evolve, RAAN/argument of perigee fixed
- **Fixed RAAN=0°**: Single RAAN case — no sweep
- **Attitude provider**: Uses Orekit `NadirPointing` and transforms the LOS with its attitude rotation; do not substitute Science's custom LVLH axis mapping
- **Single GS**: Goa only (15°N, 74°E)
- **No link budget**: Only pointing angle — no SNR, data rate, margin analysis
- **Body frame**: +Z antenna is specific to this study and differs from Science's +Y relay antenna
- **Short duration**: 1 day only — no seasonal/annual variation
- **Plot only during contact**: `times`/`angles` only appended when elevation > mask
