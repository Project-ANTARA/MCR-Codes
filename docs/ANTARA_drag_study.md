# ANTARA_drag_study.py

## File Overview
Orbital lifetime trade study comparing 4 solar array configurations for the ANTARA 3U CubeSat. Computes drag-limited lifetime vs. initial altitude (500–600 km) for each configuration under solar maximum conditions. Uses `TUMBLING_FACTOR = 1.0` (no tumbling multiplier, unlike `ANTARA_drag_finalconfig.py`).

## Dependencies
Identical to `ANTARA_drag_finalconfig.py`.

## Exports / Public API

### Module Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `MASS` | 4.0 kg | Spacecraft mass |
| `CD` | 2.2 | Drag coefficient |
| `TUMBLING_FACTOR` | 1.0 | **No tumbling multiplier** (vs 1.5 in finalconfig) |
| `END_ALT` | 450000.0 m | Reentry termination altitude |
| `areas` | Dict | 4 configurations with cross-sections |

### Configurations Tested
| Name | Cross-Section (m²) | Description |
|------|-------------------|-------------|
| "4 fixed panels" | 0.030 | All body-mounted |
| "1 fixed + 1 deployed" | 0.060 | 1 body + 1 deployable |
| "1 fixed + 2 deployed" | 0.090 | **Final config** (but area differs from finalconfig's 0.076799) |
| "4 deployed only" | 0.150 | All deployable |

### Functions

#### `lifetime(area: float, start_alt_m: float) -> float`
Same as `ANTARA_drag_finalconfig.py` but with `TUMBLING_FACTOR = 1.0`.

### Main Execution
```
for each config in areas:
    for altitude_km in 500..600 step 10:
        T = lifetime(area, altitude_km × 1000)
        print altitude_km, T in years
```

## Internal Logic
Same as `ANTARA_drag_finalconfig.py` but:
- **TUMBLING_FACTOR = 1.0** (not 1.5)
- **4 configurations** vs 1
- **"1 fixed + 2 deployed" = 0.090 m²** vs 0.076799 m² in finalconfig

## Usage Example
```bash
python src/ANTARA_drag_study.py
```
Output: Console tables for each configuration.

## Notes / TODOs
- **Area discrepancy**: "1 fixed + 2 deployed" = 0.090 m² here vs 0.076799 m² in `ANTARA_drag_finalconfig.py` — which is correct?
- **TUMBLING_FACTOR = 1.0**: Comment says "Tumbling OFF(max drag area)" but factor=1 means no tumbling multiplier applied
- **Same solar max epoch**: Year 2000 (Cycle 23)
- **No tumbling**: Unlike finalconfig, this study assumes controlled attitude (factor=1)
- **Use case**: This appears to be an earlier trade study; finalconfig represents the selected configuration
- **Console only**: No file output