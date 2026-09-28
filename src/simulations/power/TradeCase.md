# TradeCase.py

## Current scope (2026-09-12)

Source: [TradeCase.py](TradeCase.py). This script performs analytical calculations but still calls `init_orekit_data()` at module scope. Use the [repository environment and data setup](../../../README.md#quick-start); importing it can write figures/CSVs. Its hardcoded eclipse and cycle-life tables are assumptions, not refreshed vendor evidence or outputs from the latest Science recovery run.

Setup and navigation: [repository README](../../../README.md) · [documentation index](../../../docs/README.md). Supporting study results were not rerun as part of this documentation review.

## File Overview
Analytical battery trade study for ANTARA mission. Computes Depth of Discharge (DoD) vs. RAAN for multiple battery sizes using pre-computed eclipse fractions, simulates SOC over 10 orbits for worst-case RAAN, and evaluates cycle life margin. No orbital propagation — uses static eclipse table and simple energy balance.

## Dependencies
- **Standard Library**: None beyond built-ins
- **External**: `matplotlib`, `pandas`, `numpy`
- **Internal**: `paths` (`init_orekit_data`, `results_path`, `figures_path`)
- **Orekit**: Only for `init_orekit_data()` call (not used for computation)

## Exports / Public API

### Module Constants
| Constant | Value | Description |
|----------|-------|-------------|
| `LOAD_POWER` | 8.0 W | Constant spacecraft load |
| `ORBIT_PERIOD_MIN` | 97.0 min | Orbital period |
| `battery_sizes` | [10, 15, 20, 25, 30] Wh | Battery capacities tested |
| `raan_table` | Dict[int, float] | Eclipse fraction (%) per RAAN (from Orekit analysis) |
| `cycle_life_estimates` | Dict[int, int] | Estimated cycle life per battery size |

### Eclipse Table (`raan_table`)
Eclipse fraction (%) at each RAAN (0°–345°, 15° steps). Key features:
- **Eclipse-free zone**: RAAN 75°–105° and 255°–285° (0% eclipse)
- **Max eclipse**: ~36.5% at RAAN 0°, 165°, 180°, 195°
- **Source**: "from Orekit eclipse analysis" (pre-computed)

### Functions

#### `simulate_soc(battery_wh: float, eclipse_pct: float, orbits: int = 10) -> List[float]`
Simple SOC simulation over N orbits.
- **Timestep**: 10 seconds
- **Eclipse**: `Pgen = 0` for `eclipse_duration = eclipse_pct × orbit_period`
- **Sunlight**: `Pgen = LOAD_POWER + 1.3 W` (recharge margin)
- **SOC update**: `SOC += (Pnet × dt/3600) / battery_wh`, clamped [0, 1]
- **Returns**: List of SOC values per timestep

### Computation Logic

#### DoD Trade Computation
```
for each battery_size:
    for each RAAN:
        eclipse_fraction = raan_table[RAAN] / 100
        eclipse_time_hr = eclipse_fraction × (97/60)
        eclipse_energy_Wh = LOAD_POWER × eclipse_time_hr
        DoD = eclipse_energy_Wh / battery_wh
```
- Assumes **no generation during eclipse**, **full load throughout**
- DoD = energy deficit during eclipse / battery capacity

#### SOC Simulation (Worst RAAN)
- Uses RAAN with max eclipse (36.5% → RAAN 180°)
- 10 orbits × 97 min = ~16 hours
- 10s timestep → ~5,820 steps
- Sunlight recharge: +1.3W net above load

#### Cycle Life Margin
- Required cycles: 6,200 (mission requirement)
- Estimated cycle life per battery (from vendor datasheet):
  - 10 Wh: 1,500 cycles
  - 15 Wh: 4,000 cycles
  - 20 Wh: 6,000 cycles
  - 25 Wh: 8,000 cycles
  - 30 Wh: 10,000 cycles
- Margin = estimated_life - 6,200

## Output Files
**CSV** (in `results/tradecase/`):
- `battery_DoD_trade.csv` — RAAN, Battery_Wh, DoD (all combinations)
- `battery_lifetime_summary.csv` — Battery_Wh, Max_DoD, Cycle_Margin

**Plots** (in `figures/tradecase/`):
- `DoD_envelope.png` — DoD vs RAAN curves for each battery size
- `SOC_simulation.png` — SOC vs time step for 25 Wh, worst RAAN (36.5% eclipse)

## Usage Example
```bash
python src/simulations/power/TradeCase.py
```
Output: Console summary table + CSV files + plots.

## Notes / TODOs
- **Analytical only**: No orbital dynamics, no beta angle variation, no seasonal effects
- **Static eclipse table**: Single year/single epoch — doesn't capture annual beta angle evolution
- **Simplified power model**: No solar array sizing, no charging efficiency, no battery taper
- **Cycle life estimates**: Hardcoded vendor numbers — not derived from DoD-depth relationship
- **Worst RAAN**: Uses 36.5% eclipse (RAAN 180°) but table shows 36.5% at 165°, 180°, 195°
- **Orbit period**: Fixed 97 min — no altitude/RAAN dependence
- **Recharge margin**: +1.3W in sunlight is arbitrary
- **SOC simulation**: Only runs for 25 Wh battery (hardcoded)
- **Output directory**: Uses `tradecase` subdirectory in results/figures
