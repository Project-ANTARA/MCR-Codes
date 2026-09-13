# ANTARA_Science_Sim.py

## Overview

Science mission simulator with Orekit numerical propagation, ADCS, EPS, FSM, dual ground-station support, proton environment and detector models, CSV telemetry, and optional plots. Run from the repository root in the environment containing Orekit.

`MissionConfig`, `ADCSModel`, and body-axis definitions come from [antara_models.py](antara_models.md). The file also defines its own `EPSModel`, `SpacecraftFSM`, and `IllegalFSMTransitionError`, shadowing those shared imports. Do not assume the shared FSM drift guard alone verifies the simulator's local FSM.

## Environment and physics backends

Use the [root setup instructions](../README.md#quick-start), then:

```bash
conda activate orekit_env
python scripts/verify_physics_dependencies.py
```

Core dependencies include Orekit, NumPy, pandas, and Shapely. Plot imports are guarded. The verified physics packages are `ppigrf==2.1.0` and `spacepy==0.7.0`; CI installs both into Micromamba environment `antara`. Local Micromamba is not required.

### IGRF

`IGRFModel(epoch_year)` exposes:

- `get_ned_field_tesla(lat_deg, lon_deg, alt_km)`: shape `(3,)`, North/East/Down in Tesla.
- `get_inertial_field_tesla(lat_deg, lon_deg, alt_km, position_inertial)`: converts the field for pitch-angle calculations.

The backend is `ppigrf.igrf(lon, lat, altitude_km, date)`, returning East/North/Up arrays in nT. Version 2.1.0 defaults to IGRF-14 coefficients. The historical `_PYIGRF_AVAILABLE` flag describes **ppigrf** availability; it does not imply a `pyIGRF` import. Missing ppigrf warns and falls back to a centered dipole.

Altitude must remain in kilometers at the backend boundary. The existing epoch mapping uses March 1 of the configured integer year, and the inertial conversion is an approximate static rotation rather than a full Earth-fixed/time-dependent frame transform.

### Proton environment and AP9 limitation

`ProtonEnvironmentModel(config, saa_polygon)` provides `get_ap9_flux(lat_deg, lon_deg, alt_km)` and `compute_flux(...)`. The latter combines the baseline with SAA depth, magnetic pitch angle, east-west asymmetry, storm scaling, and detector response.

SpacePy's `get_AEP8` supports AE8/AP8, **not AP9**. The current AP9 call also uses an invalid `irbem.Coords` lookup and unsupported keyword arguments. It returns the configured `ap9_baseline_flux` after failure and emits one explanatory warning per model instance. Missing SpacePy also selects that baseline. This baseline feeds primary simulated flux, not just comparison plots.

`scripts/verify_physics_dependencies.py` verifies an actual IRBEM/AP8 query; it does not validate live AP9. The SpacePy `ae9ap9` module reads model output files rather than running AP9. Do not interpret successful SpacePy imports as AP9 model availability.

## CLI and scenario precedence

Run `python src/ANTARA_Science_Sim.py --help` for supported options.

| Option | Behavior |
| --- | --- |
| `--scenario` | `leop_24h`, `no_leop_24h`, or `leop_1yr` preset |
| `--duration-hours` | Explicit duration overrides preset; 24 hours without a preset |
| `--step-sec` | Explicit fine step; also selects the coarse step unless supplied separately |
| `--coarse-step-sec` | Explicit coarse step for multirate operation; preset/global default 20 seconds |
| `--rng-seed` | Explicit RNG seed overrides configuration default |
| `--no-periphery-merge` | Disable adaptive periphery bin merging |
| `--no-plots` | Skip plot generation, including with a scenario |
| `--checkpoint-every-hours` | Request periodic checkpointing |
| `--resume-from` | Load a saved checkpoint |

Precedence is **explicit CLI > scenario preset > global defaults**. `_parse_args(argv=None)` distinguishes omitted numeric values with `None`; `_config_from_args(args)` returns the validated configuration and LEOP-skip flag. Scenarios import `MissionConfig` directly from `antara_models` to avoid importing the running simulator twice.

Preset durations are 24, 24, and 8760 hours respectively. `no_leop_24h` starts in SUN_POINTING; the other presets retain startup sequencing. Preset-only runs retain multirate behavior. To request fixed two-second stepping, supply `--step-sec 2` without a separate coarse step.

```bash
# CI-equivalent fixed-step runs
python src/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots

# Annual preset with explicit multirate controls
python src/ANTARA_Science_Sim.py --scenario leop_1yr --step-sec 2 --coarse-step-sec 30 --no-plots
```

RAAN and battery capacity are configuration fields, not `--raan-deg` or `--battery-wh` CLI options. There are no `--output-dir` or `--save-pickle` options. Configure `output_tag` through `MissionConfig` when constructing runs programmatically. Only resume trusted checkpoint files.

Checkpoint CLI switches exist, but `run()` currently initializes its loop clock to zero; exact continuation has not been verified. Do not rely on checkpoint/resume to preserve annual-run chronology without testing it. See [scenario and output guidance](scenarios.md). The detumble progress bar belongs to the separate [Monte Carlo runner](leop_detumble_mc.md); Science retains its periodic `[Progress]` lines.

## Attitude geometry

Quaternions are scalar-first `[w, x, y, z]`, mapping body to inertial. Body axes remain +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel.

Constructor `_run_geometric_self_tests()` checks identity rotation, 90-degree yaw, detection +X/+Z, relay +Y/+Z, inertial/body vector round trip, and quaternion/DCM round trip. `_assert_axis_alignment()` uses normalized vectors and:

```python
degrees(atan2(norm(cross(a, b)), clip(dot(a, b), -1, 1)))
```

The alignment bound remains `1e-6` degrees. It tests numerical frame construction, not spacecraft pointing performance. Valid built-in alignment residuals were at most `1.81e-14` degrees; negative tests at 0.01, 0.1, and 1 degree still raise assertions. See the [repair report](../OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md) for full precision evidence.

## Output and verification

### Runtime tracking and multirate accounting

RELAY acquisition refreshes the actual ground-station LOS during SLEWING and operational tracking. The estimated slew timer alone cannot end acquisition: the attitude must converge and settle. Moving-target settling counts consecutive time within the configured pointing accuracy, rather than restarting the settling clock for each new target sample. Slew speed is preserved across updates within an ongoing maneuver. Pointing validity and beamwidth assertions remain enabled at their existing limits.

SUN_POINTING checks both the achieved attitude and the retained target against the current RAM/nadir geometry. A stale target outside the 5-degree hard limit forces a command refresh, including when the requested command is unchanged. This prevents the 7.5-degree hold deadband from allowing convergence to an invalid target. The hard limit, finite slew dynamics, and settling requirements are unchanged; this is a tracking fix, not a tolerance increase.

Wheel saturation intentionally freezes inertial attitude while unloading. It cannot complete a pending slew; the original objective must be reacquired after release. During hold, diagnostics check rotation validity, telemetry consistency, and alignment with the frozen target rather than nominal RAM/LOS tracking. Nominal limits apply again after release, and science/transmission are gated off throughout hold. Frozen SLEWING uses the pending mode's pointing budget and evaluates the existing saturation-timeout SAFE policy before its completion timer. Recovery thresholds are unchanged.

Startup (including DETUMBLE), SLEWING, DETECTION, and RELAY use the fine cadence. Other modes may use the coarse interval. Each iteration uses the same selected interval for dynamics, EPS, payload, mode-time accumulation, and the simulation clock; EPS is advanced once. This corrects the former 20-second clock/2-second detumble mismatch and duplicate battery update.

The earlier 14-test result below records the dependency repair. The current regression suite additionally covers moving-target settling, slew completion, timestep accounting, and multiprocessing; see [runtime repair verification](SIMULATION_RUNTIME_FIX_REPORT.md).

Paths are routed through [paths.py](paths.md). Prepare root-level Orekit data before constructing the simulator. Scenario outputs use `results/science/<tag>_ANTARA_Science_<name>.csv`, with names:

- `telemetry`, `saa_passes`, `gs_contacts`, `mode_transitions`
- `slew_events`, `detumble_events`, `rejected_transitions`, `proton_passes`

Without an output tag, the prefix is simply `ANTARA_Science_`. Plots use `figures/science/` and are skipped by `--no-plots`. Reusing a tag overwrites earlier outputs.

Telemetry uses elapsed `time_sec`; derive absolute timestamps from the configured start date. The verified one-hour runs produced 1,801 rows from 0 through 3600 seconds inclusive, at exactly two-second intervals. Neither entered DETECTION, so they establish startup/run completion rather than SAA science fidelity.

```bash
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -v
python -m py_compile src/*.py
```

The original dependency repair passed 14 tests. Subsequent verification completed 60 hours of the annual preset: 17,758 rows, two-/20-second cadence, saturation → SAFE → SUN_POINTING recovery, and worst settled, non-held pointing error 4.999973° below 5°. See [runtime evidence](SIMULATION_RUNTIME_FIX_REPORT.md). Full-year and subsystem validation are separate; old generated reports must be read with their date and environment. Use `pd.read_csv(path, low_memory=False)` for the mixed-type optional science columns.
