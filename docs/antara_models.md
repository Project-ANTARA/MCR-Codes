# antara_models.py

## File Overview
**Shared Orekit-free core models** for the ANTARA mission. This module extracts the pure-Python, zero-Java-dependency components from `ANTARA_Science_Sim.py` so they can be imported by lightweight Monte Carlo and validation scripts without initializing the Orekit JVM.

**Key design goal**: Enable `leop_detumble_mc.py` (and validation tests) to run thousands of parallel trials without paying the Orekit import/JVM-init cost per worker process.

Reviewed 2026-09-12. See the [detumble CLI/progress guide](leop_detumble_mc.md) and [Science runtime report](SIMULATION_RUNTIME_FIX_REPORT.md). Progress is implemented in the parent runner, not in these shared physics classes. Science's local FSM now prioritizes fault handling for frozen SLEWING; that change is not part of the shared FSM priority list below.

## Dependencies
- **Standard Library**: `math`, `dataclasses`, `enum`, `pathlib`, `typing`
- **External**: `numpy` only
- **Internal**: None (no `paths.py`, no Orekit)

---

## Exports / Public API

| Symbol | Type | Description |
|--------|------|-------------|
| `BODY_AXIS_VECTORS` | `Dict[str, np.ndarray]` | Canonical body axes (+X=RAM, +Y=nadir, +Z=sun-panel) |
| `_as_flat_vector` | `Callable` | Vector shape normalizer → flat (3,) float array |
| `SunSubmode` | `Enum` | SUN_POINTING power submodes: CHARGING, PASSIVE, ECLIPSE, NONE |
| `MissionConfig` | `dataclass` | All mission parameters (validated in `__post_init__`) |
| `IllegalFSMTransitionError` | `Exception` | Raised on illegal FSM state transitions |
| `SpacecraftFSM` | `class` | 7-state finite state machine with debounced fault logic |
| `ADCSModel` | `class` | Attitude determination & control with quaternion dynamics, B-dot, wheel damping, saturation |
| `EPSModel` | `class` | Shared electrical power model; Science currently defines a separate local version |

---

## Module Constants

### `BODY_AXIS_VECTORS`
```python
{
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}
```
Canonical body-frame axes per ICD Rev B: +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel.

### `SunSubmode(str, Enum)`
Power-state submodes within SUN_POINTING:
- `CHARGING` — Sunlit, battery gaining charge
- `PASSIVE` — Sunlit, battery full, coasting
- `ECLIPSE` — In umbra, solar generation = 0
- `NONE` — Current FSM mode is not SUN_POINTING

---

## Helper Functions

### `_as_flat_vector(vector) -> np.ndarray`
**Purpose**: Single shape-normalization boundary for all vector-like inputs.

**Accepts**:
- Python sequences (list, tuple)
- NumPy arrays with shapes `(3,)`, `(3, 1)`, `(1, 3)`

**Returns**: Flat `np.ndarray` of shape `(3,)` with `dtype=float`.

**Raises**: `ValueError` if result is not shape `(3,)`.

> **Note**: Unlike the version in `ANTARA_Science_Sim.py`, this does **not** handle Orekit `Vector3D` objects (no Orekit dependency).

---

## Classes

### `MissionConfig` (dataclass)
**All numeric mission parameters in one place** — no magic numbers in logic.

**Major parameter groups**:

| Category | Key Parameters |
|----------|----------------|
| **Simulation** | `sim_duration_hours`, `step_sec`, `progress_report_interval_sec`, `telemetry_flush_interval_hours`, `output_tag`, `start_year/month/day` |
| **Orbit** | `altitude_m`, `inclination_deg`, `eccentricity`, `raan_deg`, `arg_perigee_deg`, `true_anomaly_deg` |
| **Spacecraft** | `mass_kg`, `drag_coeff`, `cross_section_m2`, `srp_coeff` |
| **ADCS** | `max_slew_rate_deg_sec`, `wheel_max_torque_nm`, `max_slew_accel_deg_sec2`, `wheel_max_momentum_nms`, `pointing_accuracy_deg`, `settling_time_sec`, `adcs_idle_power_w`, `adcs_slew_power_w`, `inertia_tensor_kgm2`, `bdot_gain`, `bdot_handover_rate_deg_sec`, `wheel_rate_damp_gain`, `detumble_stable_rate_deg_sec`, `earth_B_field_T`, `magnetorquer_dipole_Am2`, `wheel_inertia_kgm2` |
| **EPS** | `battery_capacity_wh`, `initial_battery_wh`, `solar_panel_efficiency`, `solar_panel_area_m2`, `solar_constant_w_m2`, `eps_charge_efficiency`, `solar_temp_derating`, `low_power_enter_wh`, `low_power_exit_wh`, `critical_power_wh`, `saa_detection_min_power_wh`, `min_relaybattery_capacity_wh`, `safe_exit_battery_fraction` |
| **Loads** | `obc_power_w`, `eps_power_w`, `payload_power_w`, `payload_coin_logic_power_w`, `radio_tx_power_w` |
| **Storage** | `onboard_storage_mb`, `initial_storage_mb`, `storage_high_watermark_mb` |
| **Comms** | `downlink_rate_mbps`, `gs_elevation_mask_deg`, `gs_lat/lon/alt_deg`, `gs2_enabled`, `gs2_lat/lon/alt_deg`, `gs2_elevation_mask_deg` |
| **Science** | `energy_channels_mev`, `ap9_baseline_flux`, `detector_efficiency`, `detector_dead_time_sec`, `detector_dark_count_rate_hz`, `saa_depth_flux_min/max`, `ew_asymmetry_amplitude`, `storm_time_flux_multiplier`, `igrf_epoch_year`, `science_histogram_downlink_interval_sec`, `science_packet_bytes_per_bin`, `science_packet_header_bytes`, `enable_periphery_bin_merge`, `detector_coincidence_window_ns`, `detector_dead_time_ns` |
| **Timing/Hysteresis** | `startup_cold_sec`, `initial_body_rate/accel_*`, `detumble_target_attitude_label`, `large_slew_threshold_deg`, `critical_recovery_target_wh`, `attitude_command_deadband_deg`, `safe_hold_deadband_deg`, `ram/relay_track_deadband_deg`, `payload/relay/sun_pointing_valid_deg`, `directional_antenna_half_beamwidth_deg`, `detection/relay/sun_pointing_hard_limit_deg`, `charging_complete_threshold_wh`, `minimum_detection/relay_duration_sec`, `max_saturation_hold_sec`, `minimum_settled_time_before_payload/tx_sec`, `fault_debounce_samples`, `recovery_debounce_samples`, `mode_transition_grace_steps`, `penumbra_transition_sec`, `slew_transition_rate_deg_sec`, `slew_transition_penalty_sec`, `slew_power_penalty_w`, `max_solar_power_w`, `eclipse_heater_power_w`, `battery_taper_soc_start`, `battery_taper_min_fraction`, `wheel_momentum_recovery_fraction`, `rng_seed` |
| **Numerical Propagator** | `min/max_integrator_step_sec`, `integrator_position_tolerance_m`, `integrator_velocity_tolerance_mps` |
| **Plotting** | `mode_colors`, `sun_submode_colors` |

**Validation** (`__post_init__`):
- Orbit parameters in valid ranges
- EPS threshold chain strictly increasing: `critical < low_enter < low_exit < charging_complete < capacity`
- Initial battery within `[0, capacity]`
- Tip-off rates bounded by ±50 deg/s sanity limit
- Energy channel arrays consistent lengths

---

### `IllegalFSMTransitionError(RuntimeError)`
Custom exception raised when `SpacecraftFSM.transition()` is called with a disallowed state change.

---

### `SpacecraftFSM`
**7-state Finite State Machine** with strict transition table and debounced fault/recovery logic.

#### States
`COLD_AND_DARK`, `DETUMBLE`, `SAFE`, `SLEWING`, `SUN_POINTING`, `DETECTION`, `RELAY`

#### Allowed Transitions
```python
{
    "COLD_AND_DARK": {"DETUMBLE"},
    "DETUMBLE": {"SAFE", "SUN_POINTING", "SLEWING"},
    "SAFE": {"SUN_POINTING", "RELAY", "DETECTION", "SLEWING"},
    "SLEWING": {"SAFE", "SUN_POINTING", "RELAY", "DETECTION"},
    "SUN_POINTING": {"SAFE", "RELAY", "DETECTION", "SLEWING"},
    "DETECTION": {"RELAY", "SUN_POINTING", "SAFE", "SLEWING"},
    "RELAY": {"SUN_POINTING", "SAFE", "DETECTION", "SLEWING"},
}
```

#### Constructor
```python
fsm = SpacecraftFSM(config: MissionConfig)
```
Initializes in `COLD_AND_DARK` with all debounce counters cleared.

#### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `set_time` | `(time_sec: float)` | Store sim time for timeout bookkeeping |
| `transition` | `(new_mode: str, reason: str) -> bool` | Execute transition if legal; raises `IllegalFSMTransitionError` if not |
| `accumulate_mode_time` | `(dt_sec: float)` | Add dwell time to current mode |
| `minimum_dwell_complete` | `() -> bool` | Check if DETECTION/RELAY has met minimum dwell |
| `evaluate` | `(battery_wh, gs_visible, inside_saa, adcs_saturation_hold_active, saturation_hold_duration_sec, pointing_budget_error_deg, pointing_budget_limit_deg, detumble_complete) -> Tuple[str, str]` | **Main decision logic** — returns `(next_mode, reason)` |

#### Decision Logic Priority (in `evaluate`)
1. **Slewing progression** — Complete or SAFE on critical battery
2. **Startup sequencing** — `COLD_AND_DARK` → `DETUMBLE` (after `startup_cold_sec`) → `SUN_POINTING` (after `detumble_complete`)
3. **Fault overrides** — SAFE internal logic, post-SAFE recovery hold, critical battery, combined critical+saturation, saturation timeout with pointing divergence, pointing budget fault
4. **Low-power hysteresis** — Entry at `low_power_enter_wh`, exit at `low_power_exit_wh` (debounced)
5. **Minimum dwell enforcement** — DETECTION/RELAY must complete minimum duration
6. **Nominal priority** — RELAY if GS visible + battery > threshold; DETECTION if in SAA + battery > threshold; else SUN_POINTING

#### Debounce Counters
- `fault_debounce_samples` (default 3) — consecutive samples before fault activates
- `recovery_debounce_samples` (default 3) — consecutive samples before recovery clears
- Per-trigger counters in `self.debounce_counters` dict

---

### `ADCSModel`
**Attitude determination and control** with quaternion dynamics, B-dot magnetorquer, reaction wheel damping, and saturation management.

#### Constructor
```python
adcs = ADCSModel(config: MissionConfig)
```
Initializes quaternion to identity, body rates from config tip-off values, wheel momentum to zero.

#### Key Attributes
| Attribute | Type | Description |
|-----------|------|-------------|
| `current_quaternion` | `np.ndarray[4]` | Scalar-first [w,x,y,z], body→inertial |
| `target_quaternion` | `np.ndarray[4]` | Commanded target attitude |
| `wheel_momentum_nms` | `np.ndarray[3]` | Wheel momentum per axis (N·m·s) |
| `body_rate_deg_sec` | `np.ndarray[3]` | Current body rates (deg/s) |
| `body_accel_deg_sec2` | `np.ndarray[3]` | Current body accelerations (deg/s²) |
| `inertia_tensor_kgm2` | `np.ndarray[3]` | Diagonal inertia [Ixx, Iyy, Izz] |
| `is_slewing` | `bool` | Currently executing a slew |
| `slew_angle_deg` | `float` | Remaining slew angle |
| `slew_time_remaining_sec` | `float` | Time to complete slew |
| `settling` | `bool` | Post-slew settling phase |
| `is_saturated` | `bool` | Any wheel at momentum limit |
| `detumble_phase` | `str` | "BDOT", "WHEEL_DAMP", or "COMPLETE" |
| `detumble_complete` | `bool` | Detumble finished successfully |

#### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `set_time` | `(time_sec: float)` | Update internal time for saturation timeout |
| `step_detumble` | `(b_body_t: np.ndarray, dt_sec: float) -> np.ndarray` | Two-stage detumble: B-dot → wheel damp; returns wheel torque demand |
| `bdot_control` | `(b_body_t, dt_sec) -> np.ndarray` | B-dot magnetorquer torque from finite-difference dB/dt |
| `wheel_damp_control` | `() -> np.ndarray` | Per-axis inertia-scaled wheel damping torque |
| `compute_slew` | `(prev_q, new_q) -> Tuple[float, float]` | Slew angle (deg) and duration (sec) between quaternions |
| `command_attitude` | `(target_q, mode, reason, force_update=False, deadband_deg=None) -> bool` | Update target quaternion if beyond deadband; returns `True` if slew started |
| `advance` | `(dt_sec: float)` | Step attitude toward target with trapezoidal slew profile |
| `estimate_torque_vector` | `() -> np.ndarray` | Wheel torque demand from latest Euler-angle change (only during explicit SLEWING) |
| `update_wheel_momentum` | `(torque_vec, dt_sec, unload_scale, b_projection_scale)` | Integrate wheel momentum + continuous magnetorquer unloading |
| `check_saturation` | `() -> bool` | Check wheel momentum limits; trigger recovery if exceeded |
| `get_saturation_hold_duration_sec` | `() -> float` | Elapsed time in saturation hold |
| `get_pointing_error_deg` | `() -> float` | Pointing error estimate based on slew/settle state |
| `set_initial_quaternion` | `(quaternion: np.ndarray)` | Set start quaternion without slew event |

#### Quaternion Convention
- **Scalar-first** `[w, x, y, z]`
- Maps body-frame vectors → inertial coordinates
- DCM columns = inertial representations of +X_body, +Y_body, +Z_body

#### Bug Fixes Included
- **Bug 11**: Explicit re-normalization after every quaternion integration step prevents drift-induced sign flips (~71/hr → 0)
- **Wheel damping**: Per-axis inertia-scaled `tau_i = -k_damp * I_i * omega_i`; implementation reads `config.wheel_rate_damp_gain` (default 0.2), despite an older docstring mentioning `0.4/dt`
- **Moving target commands**: An ongoing slew preserves commanded speed when its target is refreshed in the same mode; acceleration and maximum-rate limits still apply
- **Saturation state**: Saturation freezes the ADCS target at the current quaternion and clears slew/settling flags. This is not proof of convergence to the original mission objective; Science preserves that objective and handles recovery separately

---

## Usage Examples

### Lightweight Monte Carlo (no Orekit)
```python
from antara_models import MissionConfig, ADCSModel, _as_flat_vector
import numpy as np

config = MissionConfig()
adcs = ADCSModel(config)

# Synthetic B-field in body frame
b_body = np.array([10e-6, 20e-6, -5e-6])

# Step detumble controller
wheel_torque = adcs.step_detumble(b_body, dt_sec=0.5)
```

### FSM Evaluation
```python
from antara_models import SpacecraftFSM, MissionConfig

config = MissionConfig()
fsm = SpacecraftFSM(config)

# Run FSM decision logic
mode, reason = fsm.evaluate(
    battery_wh=20.0,
    gs_visible=False,
    inside_saa=True,
    adcs_saturation_hold_active=False,
    saturation_hold_duration_sec=0.0,
    pointing_budget_error_deg=1.0,
    pointing_budget_limit_deg=5.0,
    detumble_complete=True,
)
# A fresh FSM starts in COLD_AND_DARK. Advance startup/time explicitly before
# expecting nominal DETECTION decisions; detumble_complete alone does not skip startup.
```

---

## Consumers (Files That Import From Here)

| File | Imports |
|------|---------|
| `src/ANTARA_Science_Sim.py` | Shared config, ADCS and axis definitions; imported EPS/FSM/exception names are subsequently shadowed by local definitions |
| `src/scenarios.py` | `MissionConfig` directly, avoiding duplicate imports of the simulator |
| `src/leop_detumble_mc.py` | `MissionConfig`, `ADCSModel`, `_as_flat_vector` |
| `src/analyze_detumble_mc.py` | `MissionConfig` |
| `tests/validation/test_comms_link_budget.py` | `MissionConfig` |
| `tests/validation/test_eps_power_budget.py` | `MissionConfig` |
| `tests/validation/test_saa_flux_model.py` | `MissionConfig` |
| `tests/validation/test_adcs_detumble_time.py` | `MissionConfig`, `run_single_trial` |
| `tests/test_fsm_spec_matches_code.py` | `SpacecraftFSM`, `MissionConfig` |

---

## Notes / TODOs

- **Deliberate scope**: Shared models have no Orekit/cartopy/shapely/matplotlib/ppigrf/spacepy dependencies. Science-specific geometry, environment, payload, logging, and runner classes remain in `ANTARA_Science_Sim.py`.
- **Partial extraction**: `EPSModel` is present here, but Science still defines local EPS/FSM/exception classes after importing shared names. Inspect the actual implementation used by a consumer before propagating a fix.
- **Configuration ownership**: Edit Science's `MissionConfig` here; the simulator re-exports it rather than maintaining another dataclass. Other standalone entrypoints may define independent configurations.
- **Test coverage**: `test_fsm_spec_matches_code.py` validates this module's FSM against `fsm_spec.py`, not Science's local FSM. `test_physics_attitude.py` exercises simulator geometry, backend units, and CLI controls.

---

## Change Log

| Date | Change |
|------|--------|
| 2026-09-07 | Created `antara_models.py` (extracted from `ANTARA_Science_Sim.py`), fixed 5 file imports, added `evaluate()` to `SpacecraftFSM`, fixed checkpoint bug in `ANTARA_Science_Sim.py` |

(End of file)
