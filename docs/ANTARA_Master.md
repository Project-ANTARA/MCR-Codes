# ANTARA_Master.py

## File Overview
High-fidelity simulation entry point for the ANTARA 3U CubeSat mission. Implements a full numerical propagation loop with Orekit (Dormand-Prince 8/5/3 integrator), a 7-state spacecraft FSM (COLD_AND_DARK, DETUMBLE, SAFE, SLEWING, SUN_POINTING, DETECTION, RELAY), detailed ADCS model with reaction wheels and magnetorquers, EPS model with battery taper charging, SAA polygon detection, ground station elevation-based visibility, science payload data generation, and communications downlink. Produces comprehensive telemetry CSVs and a 9-panel PDR plot suite.

## Dependencies
- **Standard Library**: `math`, `dataclasses`, `enum`, `typing` (Dict, List, Optional, Tuple)
- **External**: `numpy`, `pandas`, `matplotlib`, `cartopy`, `shapely`, `orekit` (Python wrapper for Orekit Java library)
- **Internal**: `paths` (`init_orekit_data`, `results_path`, `figures_path`)
- **Orekit/Java Modules**: Extensive use of `org.hipparchus`, `org.orekit.*` for propagation, forces, frames, orbits, events, time, bodies, attitudes

## Exports / Public API

### Module Constants
| Constant | Type | Description |
|----------|------|-------------|
| `BODY_AXIS_VECTORS` | `Dict[str, np.ndarray]` | Canonical body axes in body frame: +X, -X, +Y, -Y, +Z, -Z |

### Helper Functions

#### `_as_flat_vector(vector) -> np.ndarray`
Normalizes any vector-like input (Orekit Vector3D, list, np.array of shape (3,), (3,1), (1,3)) to flat float array of shape (3,).
- **Raises**: `ValueError` if result isn't shape (3,)

### Classes

#### `SunSubmode(str, Enum)`
Power-state submodes within SUN_POINTING FSM mode.
- `CHARGING` — Sunlit, battery gaining charge
- `PASSIVE` — Sunlit, battery full (coasting)
- `ECLIPSE` — In umbra, solar generation = 0
- `NONE` — Current mode is not SUN_POINTING

#### `MissionConfig` (dataclass)
**All mission parameters. No magic numbers in logic.**
Key groups:
- **Simulation**: `sim_duration_hours=12.0`, `step_sec=2.0`, `progress_report_interval_sec=180.0`
- **Orbit**: `altitude_m=600_000.0`, `inclination_deg=97.6`, `eccentricity=0.001`, `raan_deg=60.0`
- **Spacecraft**: `mass_kg=3.0`, `drag_coeff=2.2`, `cross_section_m2=0.01`, `srp_coeff=1.8`
- **ADCS**: `max_slew_rate_deg_sec=5.0`, `wheel_max_torque_nm=0.001`, `wheel_max_momentum_nms=0.01317`, `pointing_accuracy_deg=0.4`, `settling_time_sec=8.0`
- **EPS**: `battery_capacity_wh=25.0`, `initial_battery_wh=15.0`, `solar_panel_efficiency=0.193515`, `solar_panel_area_m2=0.085`, `eps_charge_efficiency=0.88`, `solar_temp_derating=0.88`
- **Thresholds**: `low_power_enter_wh=10.0`, `low_power_exit_wh=17.5`, `critical_power_wh=7.5`, `saa_detection_min_power_wh=10.0`, `min_relaybattery_capacity_wh=10.0`
- **Loads**: `obc_power_w=0.8`, `eps_power_w=1.2`, `payload_power_w=10.0`, `radio_tx_power_w=12.0`
- **Storage**: `onboard_storage_mb=32_000.0`, `initial_storage_mb=314.0`, `storage_high_watermark_mb=24_000.0`
- **Comms**: `downlink_rate_mbps=0.0096`, `gs_elevation_mask_deg=15.0`, GS at 15°N, 74°E, 70m
- **Science**: `science_data_rate_mbps=0.001152`
- **Timing/Hysteresis**: 50+ parameters for startup, detumble, slew, saturation, dwell times, deadbands
- **B-field**: `earth_b_field_model="dipole"`, `magnetorquer_dipole_Am2=0.2`, `earth_B_field_T=30.0e-6`
- **Integrator**: `min_integrator_step_sec=0.1`, `max_integrator_step_sec=120.0`, position/velocity tolerances
- **Validation**: `__post_init__` enforces eccentricity/RAAN/inclination bounds, EPS threshold ordering, tip-off rate sanity

#### `OrbitState` (dataclass)
Container for propagated state per timestep.
Fields: `absolute_date`, `time_sec`, `latitude_deg`, `longitude_deg`, `altitude_km`, `in_eclipse`, `beta_angle_deg`, `sat_position_earth_frame`, `sat_position_inertial`, `sat_velocity_inertial`

#### `AttitudeCommand` (dataclass)
Mode-dependent target attitude with metadata.
Fields: `quaternion` (np.ndarray [w,x,y,z]), `primary_objective`, `secondary_objective`, `active_attitude_target`, `attitude_hold_reason`, `relay_axis_label`, `sun_visible`

#### `ADCSModel`
Manages attitude state, slew dynamics, reaction wheel momentum, and detumble.
**Key Attributes**: `current_quaternion`, `target_quaternion`, `wheel_momentum_nms[3]`, `is_slewing`, `slew_angle_deg`, `slew_time_remaining_sec`, `settling`, `is_saturated`, `detumble_phase` ("BDOT"/"WHEEL_DAMP"/"COMPLETE"), `detumble_complete`

**Public Methods**:
- `set_time(time_sec: float) -> None` — Store sim time for saturation timeout
- `bdot_control(b_body_t: np.ndarray, dt_sec: float) -> np.ndarray` — B-dot magnetorquer torque
- `wheel_damp_control() -> np.ndarray` — Reaction wheel damping torque
- `step_detumble(b_body_t: np.ndarray, dt_sec: float) -> np.ndarray` — Two-stage detumble (B-dot → wheel damp)
- `compute_slew(prev_q, new_q) -> Tuple[float, float]` — Slew angle (deg) and duration (sec) with trapezoidal profile
- `command_attitude(target_q, mode, reason, force_update=False, deadband_deg=None) -> bool` — Update target if beyond deadband
- `advance(dt_sec: float) -> None` — Advance attitude toward target with rate/accel limits
- `estimate_torque_vector() -> np.ndarray` — Wheel torque demand from commanded Euler delta
- `update_wheel_momentum(torque_vec, dt_sec, unload_scale, b_projection_scale) -> None` — Integrate momentum + magnetorquer unload
- `check_saturation() -> bool` — Check/set saturation flags, trigger hold/recovery
- `get_saturation_hold_duration_sec() -> float` — Elapsed time in saturation hold
- `get_pointing_error_deg() -> float` — PDR-grade pointing error estimate
- `set_initial_quaternion(quaternion) -> None` — Set initial attitude without slew event

**Quaternion Convention**: Scalar-first [w,x,y,z], body-to-inertial. DCM columns = inertial representations of +X_body, +Y_body, +Z_body.

#### `EPSModel`
Power system in Wh units.
**Attributes**: `battery_wh`, `solar_generation_w`, `load_w`

**Methods**:
- `compute_solar_power(sun_angle_deg: float, illumination_fraction: float) -> float` — P = η·A·G0·cos(θ)·η_charge·η_temp·illumination
- `compute_load(mode: str, adcs_slewing: bool, in_eclipse: bool) -> float` — Per-mode load table (see class docstring)
- `step(dt_sec: float) -> Tuple[float, float]` — Advance battery: net_power·dt, with taper charging above 90% SOC

#### `SAAModel`
South Atlantic Anomaly polygon detection using Shapely.
**Polygon vertices (lon, lat)**: (-90,-5), (-80,-20), (-60,-35), (-40,-35), (-20,-25), (-10,-15), (-20,-5), (-50,0), (-70,0), (-90,-5)

**Methods**:
- `is_inside(lat_deg, lon_deg) -> bool` — Point-in-polygon test
- `detect_entry_exit(prev_inside, curr_inside, time_sec, lat, lon, absolute_date, science_data_step_mb) -> Optional[Tuple[str, Dict]]` — Returns ("ENTRY"/"EXIT", pass_dict) on transitions
- `close_open_pass(final_time_sec, absolute_date) -> None` — Close pass at sim end

#### `GroundStationModel`
Single ground station (GS_INDIA: 15°N, 74°E, 70m, 15° elevation mask).
**Methods**:
- `get_elevation(sat_pos_earth_frame: Vector3D, date: AbsoluteDate) -> float` — Elevation via Orekit TopocentricFrame
- `is_visible(elevation_deg: float) -> bool` — Compare to elevation mask
- `detect_aos_los(prev_visible, curr_visible, time_sec, elevation_deg, absolute_date, downlinked_step_mb) -> Optional[Tuple[str, Dict]]` — AOS/LOS events with stats
- `close_open_contact(final_time_sec, absolute_date) -> None`

#### `PayloadModel`
Proton flux detection payload (active only in DETECTION mode inside SAA).
**Methods**:
- `generate_data(dt_sec: float) -> float` — MB generated = rate_mbps × dt / 8
- `is_operable(battery_wh: float, mode: str) -> bool` — True if mode=DETECTION and battery ≥ saa_detection_min_power_wh

#### `CommunicationsModel`
Downlink over GS_INDIA (active only in RELAY mode while GS visible).
**Methods**:
- `downlink_step(storage_mb: float, dt_sec: float) -> float` — Downlink min(storage, rate·dt/8)
- `record_contact_time(dt_sec: float) -> None` — Accumulate raw visibility time

#### `SpacecraftFSM`
7-state finite state machine with strict transition table.
**States**: COLD_AND_DARK, DETUMBLE, SAFE, SLEWING, SUN_POINTING, DETECTION, RELAY
**ALLOWED_TRANSITIONS** defined as class constant dict.

**Methods**:
- `transition(new_mode: str, reason: str) -> bool` — Execute legal transition, record history
- `accumulate_mode_time(dt_sec: float) -> None` — Dwell time accounting
- `minimum_dwell_complete() -> bool` — Check minimum dwell for DETECTION/RELAY/SAFE
- `evaluate(battery_wh, gs_visible, inside_saa, adcs_saturation_hold_active, saturation_hold_duration_sec, pointing_error_deg, detumble_complete) -> Tuple[str, str]` — Priority-ordered mode selection logic (startup → faults → min dwell → ops priority: RELAY > DETECTION > SUN_POINTING)

#### `TelemetryLogger`
Pandas DataFrame-based recorder with 80+ telemetry columns per timestep + event DataFrames.
**Methods**:
- `log_step(...)` — Append one timestep (42 parameters)
- `log_slew_event(start_time_sec, from_mode, target_mode, angle_deg, estimated_duration_sec, end_time_sec=None)`
- `finalize_slew_event(event_index, end_time_sec)`
- `log_detumble_event(...)`
- `finalize_detumble_event(event_index, end_time_sec)`
- `build_dataframes(saa_pass_rows, gs_contact_rows, mode_transition_rows) -> Tuple[pd.DataFrame, ...]` — Returns 6 DataFrames

#### `ANTARASimulator`
Top-level simulation runner owning all subsystems.
**Key Methods**:
- `run() -> None` — Main simulation loop
- `_init_orekit()` — Initialize Orekit VM and data
- `_init_frames_and_bodies()` — Time scales, frames, Earth, Sun
- `_init_propagator()` — Numerical propagator with gravity (10×10), drag (NRLMSISE-00/DTM2000), SRP with Earth occultation
- `_propagate_state(absolute_date)` — Single propagation step
- `_build_orbit_state(propagated_state, time_sec) -> OrbitState` — Convert Orekit state to mission container
- `_update_attitude_guidance(mode, orbit_state, sun_visible, gs_visible, battery_wh, dt_sec, time_sec)` — Core ADCS update
- `_get_attitude_command(mode, orbit_state, sun_visible, battery_wh, gs_visible) -> AttitudeCommand` — Mode-specific attitude build
- `_begin_slew_transition(...)` / `_complete_slew_transition(...)` — Slew entry/exit
- `_update_storage(generated_mb, downlinked_mb)` — Onboard storage accounting
- `_plot_results(...)` — 9-panel PDR figure suite
- `_print_summary(...)` — Console mission summary

## Internal Logic

### Simulation Loop (`run()`)
1. Initialize time, progress reporting
2. **Per timestep (2s default, 12h default = 21,600 steps)**:
   - Update FSM/ADCS time
   - Propagate orbit to current time
   - Build `OrbitState` (position, lat/lon, eclipse via shadow-cone geometry, beta angle)
   - Compute GS elevation & visibility; SAA inside check; sun visibility; illumination fraction (penumbra ramp)
   - **FSM evaluation**: `fsm.evaluate()` returns requested mode + reason
   - **Mode transition handling**:
     - SLEWING auto-completes when `time >= slew_end_time_sec`
     - COLD_AND_DARK/DETUMBLE transitions are immediate
     - Operational transitions go through SLEWING (computed duration + settling)
   - **Attitude guidance** (`_update_attitude_guidance`):
     - SLEWING: advance ADCS toward target
     - Saturation hold: freeze attitude, advance ADCS
     - DETUMBLE: run physics detumble controller
     - Other modes: command attitude (with deadband), advance ADCS
   - **Power**: compute solar (with startup gating), compute load, step EPS
   - **Science**: generate if DETECTION + inside SAA + pointing valid + settled + operable
   - **Comms**: downlink if RELAY + GS visible + settled + pointing valid + storage > 0
   - **Wheel momentum**: update (torque estimate + magnetorquer unload)
   - **Event detection**: SAA entry/exit, GS AOS/LOS
   - **Physics sanity checks**: one-time warnings for impossible states
   - **Telemetry logging**: 80+ columns per step
   - **Progress print** every 180s sim time

### Attitude Frame Building (`build_frame`)
Constructs body-to-inertial quaternion from:
- Primary axis label (+X/+Y/+Z) locked to target inertial vector
- Secondary axis label locked to reference vector projected orthogonal to primary
- Third axis via cross product (right-hand rule)
- Validates orthogonality, handedness, determinant = 1

### Eclipse Geometry (`_is_in_eclipse_geometry`)
Direct shadow-cone computation (not Orekit event detector):
- Shadow axis = anti-Sun unit vector from Earth center
- Spacecraft in umbra if angle to shadow axis ≤ Earth apparent radius - Sun apparent radius

### Outputs
**CSV files** (via `results_path("master", ...)`):
- `ANTARA_Master_telemetry.csv` — Per-timestep telemetry (80+ columns)
- `ANTARA_Master_saa_passes.csv` — SAA pass events
- `ANTARA_Master_gs_contacts.csv` — GS contact events
- `ANTARA_Master_mode_transitions.csv` — FSM transition log
- `ANTARA_Master_slew_events.csv` — Slew events with angles/durations
- `ANTARA_Master_detumble_events.csv` — Startup detumble events

**Figure** (via `figures_path("master", ...)`):
- `ANTARA_7day_PDR_simulation.png` — 9-panel plot: ground track, SAA/science density, battery, power, mode timeline, science/downlink/storage, slew histogram, wheel momentum, GS contacts

## Usage Example
```bash
# Run from repository root
python src/ANTARA_Master.py
```
Output: `results/master/` CSVs + `figures/master/ANTARA_7day_PDR_simulation.png`

## Notes / TODOs
- **Runtime**: ~minutes for 12h @ 2s step; monitor disk for telemetry CSV size (~21,600 rows × 80 cols)
- **Plotting dependencies**: `matplotlib` + `cartopy` required for plots; guarded by try/except in Science_Sim but Master imports directly
- **Orekit VM**: `orekit.initVM()` called in `_init_orekit()` and also at module top-level (line 12) — harmless but redundant
- **Atmosphere fallback**: Tries NRLMSISE-00 (CSSI space weather) → DTM2000 (Marshall solar activity) → raises RuntimeError
- **SAFE mode behavior**: Freezes current inertial attitude (no slew to nadir) to avoid wheel saturation from repeated large slews
- **SUN_POINTING misnomer**: Despite name, does NOT track Sun — holds fixed RAM/nadir/orbit-normal inertial attitude; submodes (CHARGING/PASSIVE/ECLIPSE) are power-state labels only
- **Geometric self-tests**: Runs at startup (`_run_geometric_self_tests`) to verify quaternion/DCM convention consistency
- **Validation gate**: `_validate_attitude_metrics` commented out in main loop (line 3672) — can be enabled for debugging
- **Random seed**: Configurable via `MissionConfig.rng_seed` (default 42) for reproducible jitter