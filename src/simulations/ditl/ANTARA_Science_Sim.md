# ANTARA_Science_Sim.py

## Overview

Science mission simulator with Orekit numerical propagation, ADCS, EPS, FSM, dual ground-station support, proton environment and detector models, CSV telemetry, and optional plots. Run from the repository root in the environment containing Orekit.

`MissionConfig`, `ADCSModel`, and body-axis definitions come from [antara_models.py](../helper/antara_models.md). The file defines its own `EPSModel`, `SpacecraftFSM`, and `IllegalFSMTransitionError`; these are distinct from the shared models and are not imported from them. Do not assume the shared FSM drift guard alone verifies the simulator's local FSM.

## Environment and physics backends

Use the [root setup instructions](../../../README.md#quick-start), then:

```bash
conda activate orekit_env
python scripts/verify_physics_dependencies.py
```

Core dependencies include Orekit, NumPy, pandas, and Shapely. Plot imports are guarded. The verified physics packages are `ppigrf==2.1.0` and `spacepy==0.7.0`; CI installs both into Micromamba environment `antara`. Local Micromamba is not required.

### IGRF

`IGRFModel(epoch_year)` exposes:

- `get_ned_field_tesla(lat_deg, lon_deg, alt_km, date=None)`: shape `(3,)`, North/East/Down in Tesla. An explicit date is UTC; otherwise the configured decimal year is used.
- `get_inertial_field_tesla(lat_deg, lon_deg, alt_km, absolute_date, earth_frame, inertial_frame)`: queries at the mission epoch and rotates the field into the requested inertial frame.

The backend is `ppigrf.igrf(lon, lat, altitude_km, date)`, returning East/North/Up arrays in nT. Version 2.1.0 defaults to IGRF-14 coefficients. The historical `_PYIGRF_AVAILABLE` flag describes **ppigrf** availability; it does not imply a `pyIGRF` import. Missing ppigrf warns and falls back to a centered dipole.

Altitude remains in kilometers at the backend boundary. Decimal years preserve the fraction of the actual calendar year, including leap years: `2000.25` means April 1 at 12:00 UTC. Runtime inertial queries use the current `OrbitState.absolute_date`, not a frozen configured epoch. The NED basis first produces an Earth-fixed vector; Orekit's rotation at that same date then maps it into the simulator's inertial frame. No translation is applied to a magnetic-field vector. The dipole fallback follows the same rotation. The separate configured `earth_b_field_model="dipole"` branch is unchanged.

### Ground-station consistency

Visibility, relay LOS, off-boresight angle, link elevation and downlink credit use one selection policy: highest elevation among stations above their own masks, with the primary winning ties. Contact histories retain each station's own elevation and visibility, including downlink credit at AOS. No visible station means no active link. Station selection is an epoch-local query, so look-ahead propagation cannot replace the current target. Existing finite slew and settlement checks still apply; there is no instantaneous attitude switch.

### Invalid detumble dynamics

The shared ADCS rejects non-finite body rates, fields, torques and invalid timesteps/inertias. Integration overflow raises instead of zeroing rates and falsely reporting successful detumbling. Valid rigid-body integration and existing finite-rate caps are unchanged. This does not yet repair the Monte Carlo acceleration-input or wheel-momentum accounting issues identified in the audit; see the [repair backlog](SCIENTIFIC_INTEGRITY_REPAIR.md).

### Proton environment and AP9 limitation

Histogram acquisition and packet serialization are distinct: accepted counts may
accumulate for several steps before `science_data_mb` becomes positive. SAA exit
and horizon flushes are included in the SAA pass's packet-volume total. The
zero-science diagnostic tests accepted counts, not packet bytes. September 26
regressions cover both closure paths and restart with pending counts.

`ProtonEnvironmentModel(config, saa_polygon)` provides `get_ap9_flux(lat_deg, lon_deg, alt_km)` and `compute_flux(...)`. The latter combines the baseline with SAA depth, magnetic pitch angle, east-west asymmetry, storm scaling, and detector response.

SpacePy's `get_AEP8` supports AE8/AP8, **not AP9**. The legacy `get_ap9_flux()` accessor now explicitly returns `ap9_baseline_flux` and warns once; it does not attempt an unsupported backend call. The environment spectrum feeds detector counts through linear energy interpolation onto detector bin centers. The achieved +X payload boresight supplies the directional look vector. The `ap9_flux` output key remains a compatibility name for baseline flux, not independent AP9 measurements. See [September 25 repairs](../../../docs/AUDIT_REPAIRS_2026-09-25.md).

`scripts/verify_physics_dependencies.py` verifies an actual IRBEM/AP8 query; it does not validate live AP9. The SpacePy `ae9ap9` module reads model output files rather than running AP9. Do not interpret successful SpacePy imports as AP9 model availability.

## CLI and scenario precedence

Run `python src/simulations/ditl/ANTARA_Science_Sim.py --help` for supported options.

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
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots

# Annual preset with explicit multirate controls
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_1yr --step-sec 2 --coarse-step-sec 30 --no-plots
```

RAAN and battery capacity are configuration fields, not `--raan-deg` or `--battery-wh` CLI options. There are no `--output-dir` or `--save-pickle` options. Configure `output_tag` through `MissionConfig` when constructing runs programmatically. Only resume trusted checkpoint files.

`run()` and `run_steps(n)` share the production integration loop and continue from the saved clock. Checkpoints are written synchronously at completed intervals, on **simulated-time** deadlines, under `results/checkpoints/`. Writes are atomic. Current **version-10** snapshots retain pending commands, settlement, bounded active-pass statistics, payload counters/buffers, event histories and RNG state; Orekit objects are reconstructed. Version-7/8/9 snapshots are migrated on load. Completed Science events use append-only `results/science/*.jsonl` archives: retain these files at their recorded paths. Checkpoints store a byte-prefix hash, not the complete archive; resume verifies and forks that prefix. All event CSV exports are chunked. Only one open slew and one open detumble record remain mutable; completion archives them without changing indices. Event figures sample at most 5,000 records per stream and are not exact statistics. Version-6 and older snapshots are rejected: September 28 nominal-frame and slew-energy corrections invalidate prior physics histories. Retain the original telemetry CSV: the checkpoint verifies and copies its saved byte prefix into a separate resume-tagged file, leaving the original untouched. Do not combine `--resume-from` with scenario/duration/step/seed overrides; it uses the saved configuration (`--no-plots` is allowed).

Older unversioned pickle and JSON/NPZ checkpoints lack required state and are rejected explicitly. Never load untrusted pickle files. Short uninterrupted/segmented/resumed comparisons are tested; annual resume accuracy, cost and memory use are not established. See [execution verification](EXECUTION_PAYLOAD_REPAIR.md) and [scenario guidance](scenarios.md). Science retains its periodic `[Progress]` lines.

## Attitude geometry

Quaternions are scalar-first `[w, x, y, z]`, mapping body to inertial. Body axes remain +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel.

Constructor `_run_geometric_self_tests()` checks identity rotation, 90-degree yaw, detection +X/+Z, relay +Y/+Z, inertial/body vector round trip, and quaternion/DCM round trip. `_assert_axis_alignment()` uses normalized vectors and:

```python
degrees(atan2(norm(cross(a, b)), clip(dot(a, b), -1, 1)))
```

The alignment bound remains `1e-6` degrees. It tests numerical frame construction, not spacecraft pointing performance. Valid built-in alignment residuals were at most `1.81e-14` degrees; negative tests at 0.01, 0.1, and 1 degree still raise assertions. See the [repair report](../../../OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md) for full precision evidence.

## Output and verification

### Runtime tracking and multirate accounting

RELAY acquisition refreshes the actual ground-station LOS during SLEWING and operational tracking. The estimated slew timer alone cannot end acquisition: the attitude must converge and settle. Moving-target settling counts consecutive time within the configured pointing accuracy, rather than restarting the settling clock for each new target sample. Slew speed is preserved across updates within an ongoing maneuver. Pointing validity and beamwidth assertions remain enabled at their existing limits.

SUN_POINTING checks both the achieved attitude and the retained target against the current RAM/nadir geometry. A stale target outside the 5-degree hard limit forces a command refresh, including when the requested command is unchanged. This prevents the 7.5-degree hold deadband from allowing convergence to an invalid target. The hard limit, finite slew dynamics, and settling requirements are unchanged; this is a tracking fix, not a tolerance increase.

Wheel saturation intentionally freezes inertial attitude while unloading. It cannot complete a pending slew; the original objective must be reacquired after release. During hold, diagnostics check rotation validity, telemetry consistency, and alignment with the frozen target rather than nominal RAM/LOS tracking. Nominal limits apply again after release, and science/transmission are gated off throughout hold. Frozen SLEWING uses the pending mode's pointing budget and evaluates the existing saturation-timeout SAFE policy before its completion timer. Recovery thresholds are unchanged.

Startup (including DETUMBLE), SLEWING, DETECTION, and RELAY use the fine cadence. Other modes may use the coarse interval. Each iteration uses the same selected interval for dynamics, EPS, payload, mode-time accumulation, and the simulation clock; EPS is advanced once. This corrects the former 20-second clock/2-second detumble mismatch and duplicate battery update.

Slew intervals now stop at the first predicted wheel-capacity crossing. A disposable ADCS preview determines the boundary; real dynamics advance once, and wheel drive/unloading are integrated per phase. `wheel_event_split=True` marks shortened intervals; `state_time_sec` gives their endpoint. Thus even fixed stepping specifies a maximum, with shorter wheel-event or horizon intervals permitted. Existing saturation hold/recovery rules remain unchanged. See [verification and model limits](../../../WHEEL_EVENT_REPAIR.md).

The earlier 14-test result below records the dependency repair. The current regression suite additionally covers moving-target settling, slew completion, timestep accounting, and multiprocessing; see [runtime repair verification](SIMULATION_RUNTIME_FIX_REPORT.md).

Paths are routed through [paths.py](../helper/paths.md). Prepare root-level Orekit data before constructing the simulator. Scenario outputs use `results/science/<tag>_ANTARA_Science_<name>.csv`, with names:

- `telemetry`, `saa_passes`, `gs_contacts`, `mode_transitions`
- `slew_events`, `detumble_events`, `rejected_transitions`, `proton_passes`

Without an output tag, the prefix is simply `ANTARA_Science_`. Plots use `figures/science/` and are skipped by `--no-plots`. Reusing a tag overwrites earlier outputs.

Telemetry uses elapsed `time_sec`; derive absolute timestamps from the configured start date. The verified one-hour runs produced 1,801 rows from 0 through 3600 seconds inclusive, at exactly two-second intervals. Neither entered DETECTION, so they establish startup/run completion rather than SAA science fidelity.

Since the September 25 repair, interval rows explicitly carry `interval_duration_sec` and `state_time_sec`: geometry/mode are sampled at `time_sec`, while integrated subsystem state is at the interval end (`state_time_sec`). The last row is an observation with `terminal_snapshot=True` and zero duration, not another integration step. Ignore it for sample-based detector statistics and weight interval statistics by duration. Any final pending histogram flush is included in that row's data/storage totals. The CSV is not rewritten during finalization, preserving checkpoint prefix integrity.

Reporting now validates timestamp monotonicity, duration/gap agreement and end-state times, including continuity across streaming chunks. A truncated file lacking its zero-duration final observation cannot produce a completed streaming summary. These checks detect inconsistent accounting, not incorrect physical calibration; see [telemetry integrity](TELEMETRY_INTEGRITY_REPAIR.md).

Numeric scalar summaries reject infinite values. Missing samples are excluded from the mean, with `valid_duration_sec`, `missing_duration_sec` and `missing_rows` reported per metric. All-missing metrics have null mean/min/max. Terminal observations can affect extrema and missing-row counts but carry no duration weight. Check coverage before treating an average as representative of the complete run.

Streaming summaries explicitly exclude the declared textual/vector fields and strictly parse the remaining scalar columns. Malformed scalar text raises an error naming the column instead of disappearing from statistics. Numeric-looking cell IDs remain metadata; nullable boolean measurements retain their duration coverage regardless of CSV chunk boundaries. New text/vector telemetry fields must be registered in `SCIENCE_NONSCALAR_COLUMNS` in `telemetry_metrics.py`.

Display decimation selects up to 5,000 globally spaced row indices, retaining the
first and last row. Two bounded CSV passes prevent the former bias toward recent
chunks. This spacing is by row, not elapsed time; short events and extrema may
still be absent from a figure. Use complete telemetry and summary metrics for
scientific conclusions. Sampling requires a completed, unchanged input file.

Payload precision estimates use each archived bin's accumulated expected count (`1/sqrt(max(N_bin, 1))`), not a common pooled count. The floor represents a conservative reporting convention for empty/low-count bins, not an exact low-count confidence interval. Periphery merging now precedes both mission-histogram and packet accumulation. Counts remain deterministic expected counts; neither stochastic detector realization nor absolute radiation calibration is established by these fixes.

```bash
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -v
python -m py_compile src/*.py
```

The original dependency repair passed 14 tests. Subsequent verification completed 60 hours of the annual preset: 17,758 rows, two-/20-second cadence, saturation → SAFE → SUN_POINTING recovery, and worst settled, non-held pointing error 4.999973° below 5°. See [runtime evidence](SIMULATION_RUNTIME_FIX_REPORT.md). Full-year and subsystem validation are separate; old generated reports must be read with their date and environment. Use `pd.read_csv(path, low_memory=False)` for the mixed-type optional science columns.
