# ANTARA Mission Simulation - Implementation Plan (Science Sim Only)

> Historical proposal, superseded. Do not implement the code sketches below as
> current instructions: checkpointing, execution, detumble sampling and source
> locations have since changed. See [current status](../../docs/CURRENT_STATUS.md)
> and [current layout](../../docs/SIMULATION_LAYOUT.md), reconciled 2026-09-28.

## Overview
This plan covers 5 stages of enhancements to the **ANTARA Science Simulator** (`src/ANTARA_Science_Sim.py`) only. The Master simulator (`ANTARA_Master.py`) is out of scope.

---

## STAGE 0 — Harness Scaffolding (BLOCKING for Stages 1 and 2)

### 0.1 Telemetry Flush Infrastructure
**Status**: Science Sim's `TelemetryLogger` already has `set_flush_path()` and `flush_telemetry()` methods.
**Action**: Wire it up in `ANTARAScienceSimulator.run()`:
- Add `telemetry_flush_interval_hours: float = 6.0` to `MissionConfig`
- At start of `run()`: `self.logger.set_flush_path(results_path("science", f"{output_tag}_ANTARA_Science_telemetry.csv"))` (with output_tag prefix from 0.2)
- Flush every `config.telemetry_flush_interval_hours` (check `time_sec` modulo)
- At end: read back flushed CSV(s) via `pd.read_csv(chunked=True)` to build final DataFrames instead of relying on `self.logger.telemetry_rows`

### 0.2 Output Tag for Filenames
**Action**: Add `output_tag: Optional[str] = None` to `MissionConfig`
- Modify all `results_path("science", ...)` / `figures_path("science", ...)` calls in `run()` and plotting methods to prefix filename with `f"{output_tag}_"` when set
- Example: `results_path("science", f"{output_tag}_ANTARA_Science_telemetry.csv")` if `output_tag="leop_24h"`

### 0.3 Skip LEOP Flag
**Action**: Add `skip_leop: bool = False` to `ANTARAScienceSimulator.__init__`
When `True`:
```python
self.fsm.current_mode = "SUN_POINTING"
self.fsm.detumble_complete = True
self.adcs.detumble_complete = True
self.adcs.body_rate_deg_sec[:] = 0.0
# Compute default orientation quaternion from initial orbit state
initial_orbit_state = self._build_orbit_state(self.propagator.getInitialState(), 0.0)
q = self._compute_default_orientation_attitude(initial_orbit_state).quaternion
self.adcs.set_initial_quaternion(q)
```
**Note**: Direct state assignment, NOT via `fsm.transition()` — `COLD_AND_DARK -> SUN_POINTING` remains illegal in `ALLOWED_TRANSITIONS`.

### 0.4 Checkpoint/Resume Skeleton
**Action**: Add to `ANTARAScienceSimulator`:
```python
def save_checkpoint(self, path: Path) -> None:
    state = {
        "fsm": {k: v for k, v in self.fsm.__dict__.items() if not k.startswith('_')},
        "adcs": {k: v for k, v in self.adcs.__dict__.items() if not k.startswith('_')},
        "eps": {k: v for k, v in self.eps.__dict__.items() if not k.startswith('_')},
        "payload": {"total_science_data_generated_mb": self.payload.total_science_data_generated_mb,
                    "total_detection_seconds": self.payload.total_detection_seconds,
                    "_cell_histograms": self.payload._cell_histograms,
                    "_pending_bin_counts": self.payload._pending_bin_counts,
                    "_pending_cell_elapsed_sec": self.payload._pending_cell_elapsed_sec,
                    "_previous_cell_id": self.payload._previous_cell_id,
                    "_current_pass_records": self.payload._current_pass_records},
        "saa": {"current_pass": self.saa.current_pass, "completed_passes": self.saa.completed_passes},
        "ground_station": {"current_contact": self.ground_station.current_contact,
                           "completed_contacts": self.ground_station.completed_contacts},
        "ground_station_2": {"current_contact": self.ground_station_2.current_contact if self.ground_station_2 else None,
                             "completed_contacts": self.ground_station_2.completed_contacts if self.ground_station_2 else []},
        "time_sec": self.time_sec,
        "config": self.config,
    }
    with open(path, 'wb') as f:
        pickle.dump(state, f)

@classmethod
def load_checkpoint(cls, path: Path, config: MissionConfig) -> "ANTARAScienceSimulator":
    with open(path, 'rb') as f:
        state = pickle.load(f)
    sim = cls(config)
    # Restore all state...
    return sim
```
**CLI flags**: `--checkpoint-every-hours FLOAT`, `--resume-from PATH`
**Format**: Try pickle first; on version mismatch, fall back to JSON+numpy .npz format.

---

## STAGE 1 — Three Scenarios

### 1.1 Scenario Builder (`src/scenarios.py`)
```python
from typing import Literal, Tuple
from ANTARA_Science_Sim import MissionConfig

def build_scenario(name: Literal["leop_24h", "no_leop_24h", "leop_1yr"]) -> Tuple[MissionConfig, bool]:
    if name == "leop_24h":
        return MissionConfig(sim_duration_hours=24.0, output_tag="leop_24h",
                             telemetry_flush_interval_hours=6.0), False
    elif name == "no_leop_24h":
        return MissionConfig(sim_duration_hours=24.0, output_tag="no_leop_24h",
                             telemetry_flush_interval_hours=6.0), True
    elif name == "leop_1yr":
        return MissionConfig(sim_duration_hours=8760.0, output_tag="leop_1yr",
                             telemetry_flush_interval_hours=6.0,
                             coarse_step_sec=20.0), False  # for multi-rate
```

### 1.2 CLI Scenario Argument
**Action**: Add to `_parse_args()` in `ANTARA_Science_Sim.py`:
```python
parser.add_argument("--scenario", choices=["leop_24h", "no_leop_24h", "leop_1yr"], default=None)
parser.add_argument("--coarse-step-sec", type=float, default=20.0, help="Coarse step for multi-rate (leop_1yr)")
```
When `--scenario` given: call `build_scenario()`, ignore `--duration-hours/--step-sec` (warn if both given).

### 1.3 Multi-Rate Stepping (leop_1yr Performance)
**Profile**: Run 24h with `time.perf_counter()` around main loop, extrapolate.
**Implementation**: In main loop (`run()`):
```python
# Determine step size for this iteration
if (current_mode in {"SLEWING", "DETECTION", "RELAY"} or
    abs(illumination_fraction - self.previous_illumination_fraction) > 0.1):  # near eclipse boundary
    step_sec = self.config.step_sec
else:
    step_sec = self.config.coarse_step_sec  # default 20.0

# For EPS: sub-step at config.step_sec internally
if step_sec > self.config.step_sec:
    n_sub = int(step_sec / self.config.step_sec)
    for _ in range(n_sub):
        self.eps.step(self.config.step_sec)
        # ... other fast sub-steps
else:
    self.eps.step(step_sec)
```
**Validation**: `leop_24h` with multi-rate must match fixed-step baseline within:
- Battery Wh: ≤ 1%
- Mode durations: ≤ 2 coarse steps (40s)

### 1.4 Scenario Comparison (`src/compare_scenarios.py`)
**Action**: Script that loads all three scenarios' CSVs and computes:
- Per-mode duty-cycle %
- Min/avg/max battery_wh
- Saturation-hold rising-edge count
- Total science MB generated & downlinked
- Downlink efficiency %
- Eclipse fraction %
- Average attitude_error_deg per mode
**Outputs**: `scenario_comparison.csv` + 3-panel figure (battery, mode timeline, science/downlink)

---

## STAGE 2 — LEOP Detumble Monte Carlo

### 2.1 New Module `src/leop_detumble_mc.py`
Imports ONLY: `ADCSModel`, `MissionConfig` from `ANTARA_Master` — no Orekit, no simulator.

### 2.2 Synthetic B-field Function
```python
def synthetic_b_body(adcs: ADCSModel, t_sec: float, b_magnitude_t: float,
                      orbital_period_sec: float) -> np.ndarray:
    # Fixed inertial B-vector, rotate at orbital rate, project to body frame
    omega_orb = 2 * np.pi / orbital_period_sec
    b_inertial = np.array([0.0, 0.0, b_magnitude_t])
    angle = omega_orb * t_sec
    rot = rotation_matrix_z(angle)
    b_inertial_rotated = rot @ b_inertial
    return adcs.rotate_inertial_vector_to_body(adcs.current_quaternion, b_inertial_rotated)
```

### 2.3 Dataclass + Single Trial
```python
@dataclass
class DetumbleTrialResult:
    rate0_deg_sec: np.ndarray
    accel0_deg_sec2: np.ndarray
    rate0_mag: float
    accel0_mag: float
    success: bool
    hit_rate_cap: bool
    time_to_handover_sec: float
    time_to_complete_sec: float
    peak_wheel_momentum_nms: float

# Expose rate cap as constant on ADCSModel
ADCSModel.RATE_CAP_DEG_SEC = max(120.0, 10.0 * config.max_slew_rate_deg_sec)  # from _integrate_body_rates

def run_single_trial(rate0, accel0, config, dt_sec=0.5, timeout_sec=...):
    adcs = ADCSModel(config)
    adcs.body_rate_deg_sec = rate0.copy()
    adcs.body_accel_deg_sec2 = accel0.copy()
    # Loop step_detumble until complete or timeout
    # Track handover (BDOT -> WHEEL_DAMP), rate cap hits, peak wheel momentum
```

### 2.4 Monte Carlo Runner
```python
def run_monte_carlo(n_trials, config, dt_sec=0.5, timeout_sec, seed=None, n_workers=None):
    rng = np.random.default_rng(seed)
    # Sample rate0 magnitude ~Uniform(0, 60) deg/s, direction uniform on sphere
    # Sample accel0 magnitude ~Uniform(0, 1.0) deg/s^2 (assumption - pending deployer data)
    # Parallelize with ProcessPoolExecutor
    # Return pd.DataFrame with all DetumbleTrialResult fields
```

### 2.5 Baseline Timeout
Run ONE trial at current defaults (6, -5.5, 4.5 deg/s) → get `time_to_complete_sec` → `timeout_sec = 3 * that`. Print it.

### 2.6 Analysis (`src/analyze_detumble_mc.py`)
- 2D success-probability heatmap over (rate0_mag, accel0_mag) bins
- Logistic regression: success ~ rate0_mag + accel0_mag + interaction
- Report 50%/95% success contours at median and 90th-percentile accel
- time_to_complete_sec vs rate0_mag scatter with trend (successes only)
- Per-axis breakdown given inertia_tensor = (0.016, 0.016, 0.004) — NOT axisymmetric about Z
- Print current config tip-off magnitude (~9.1 deg/s) vs fitted contours, margin in deg/s

---

## STAGE 3 — FSM Decision Spec + Diagrams

### 3.1 Spec Document (`src/fsm_spec.py`)
Create ordered list mirroring `SpacecraftFSM.evaluate()` branch order (Science Sim version with debounce counters):
```python
FSM_SPEC = [
    # 1. SLEWING progression
    ("slew_complete", "time_sec >= slew_end_time_sec", "target_mode"),
    ("slew_critical_battery", "battery_wh < critical_power_wh during SLEWING", "SAFE"),
    ("slew_continue", "default SLEWING", "SLEWING"),
    
    # 2. Startup sequencing
    ("cold_to_detumble", "COLD_AND_DARK and time_sec >= startup_cold_sec", "DETUMBLE"),
    ("detumble_complete", "DETUMBLE and detumble_complete", "SUN_POINTING"),
    
    # 3. Fault overrides (highest priority)
    ("critical_battery", "battery_wh < critical_power_wh", "SAFE"),
    ("combined_fault", "critical_fault_active and wheel_saturation_fault_active", "SAFE"),
    ("saturation_timeout", "saturation_hold_duration > max_saturation_hold_sec and pointing_error > safe_hold_deadband", "SAFE"),
    ("saturation_hold", "adcs_saturation_hold_active", "current_mode"),
    
    # 4. Minimum dwell enforcement
    ("detection_dwell", "DETECTION and not minimum_dwell_complete()", "DETECTION"),
    ("relay_dwell", "RELAY and not minimum_dwell_complete()", "RELAY"),
    ("safe_dwell", "SAFE and not minimum_dwell_complete()", "SAFE"),
    
    # 5. Operational mode selection
    ("relay_priority", "gs_visible and battery_wh > min_relaybattery_capacity_wh", "RELAY"),
    ("detection_priority", "inside_saa and battery_wh > saa_detection_min_power_wh", "DETECTION"),
    ("low_power", "battery_wh < low_power_enter_wh", "SUN_POINTING"),
    ("nominal_sun_pointing", "default", "SUN_POINTING"),
]
```

### 3.2 Drift Guard Test (`tests/test_fsm_spec_matches_code.py`)
- Generate synthetic tuples covering each branch + edge cases at thresholds
- Run through `SpacecraftFSM.evaluate()` 
- Assert result matches first matching guard in `FSM_SPEC`

### 3.3 Diagram Generator (`src/plot_fsm_diagram.py`)
- Build graph from `SpacecraftFSM.ALLOWED_TRANSITIONS` directly
- Nodes colored via `MissionConfig().mode_colors`
- Render with graphviz → PNG + SVG
- Emit Mermaid `stateDiagram-v2` text (`fsm_diagram.mmd`) with:
  - Single-guard edges labeled with guard_name
  - Multi-guard edges labeled "(multiple)" pointing to fsm_spec.py

---

## STAGE 4 — Subsystem Validation Against Real Data

### 4.1 Orbit Propagation (`tests/validation/test_orbit_propagation.py`)
- Propagate equivalent orbit with Skyfield/SGP4
- Compare period (tolerance: ±1s) and RAAN drift vs closed-form SSO precession:
  `dOmega/dt ≈ -1.5 * n * J2 * (Re/p)^2 * cos(i)` (tolerance: ±5%)

### 4.2 IGRF Field (`tests/validation/test_igrf_field.py`)
- Sample `IGRFModel.get_ned_field_tesla()` at ≥10 points spanning SAA
- Compare against NOAA/NCEI IGRF-13 calculator output (hardcode reference values with source/date)
- Tolerance: field magnitude within 5%

### 4.3 ADCS Detumble Time (`tests/validation/test_adcs_detumble_time.py`)
- Analytical B-dot time constant: `τ ≈ I / (k * B²)` 
- Compare against Stage 2 baseline trial's `time_to_handover_sec`
- Tolerance: order-of-magnitude agreement (document why tighter isn't expected)

### 4.4 EPS Power Budget (`tests/validation/test_eps_power_budget.py`)
- Recompute `solar_panel_efficiency` from published cell BOL efficiency (28.5%), packing factor (0.67), harness derating
- Compare against config value 0.193515
- Tolerance: ±2%

### 4.5 SAA/Flux (`tests/validation/test_saa_flux_model.py`)
- Compare `config.ap9_baseline_flux` against literature quiet-time differential proton flux at 600km/L~1.5
- Tolerance: within factor of 2 per channel (document AP9 intrinsic uncertainty)

### 4.6 Comms Link Budget (`tests/validation/test_comms_link_budget.py`)
- Implement Friis link budget: Tx power, cable loss, antenna gains, FSPL at slant range, Rx sensitivity
- Compare against `CommunicationsModel.downlink_step()`'s `sqrt(sin(elevation))` heuristic
- Tolerance: heuristic tracks monotonic shape, within 30% at 45° elevation

### 4.7 Validation Report Generator (`scripts/generate_validation_report.py`)
- Run all 6 validation tests
- Capture pass/fail + printed values
- Write `VALIDATION_REPORT.md` with sections: reference source, method, result, pass/fail, tolerance rationale

---

## STAGE 5 — Integration

### 5.1 CI Script
Order:
1. Stage 0 acceptance checks (telemetry flush, output_tag, skip_leop, checkpoint)
2. Stage 1 three scenarios (reduced to 1 sim hour each for CI; full 1-year nightly)
3. Stage 2 MC at reduced n_trials=100 for CI; full n_trials≥5000 nightly
4. Stage 3 spec/diagram tests
5. Stage 4 validation suite

**Document**: Which parts run every-commit vs nightly/manual.

### 5.2 README Updates
- How to run each scenario (`--scenario leop_24h` etc.)
- How to run MC sweep (`python src/leop_detumble_mc.py`)
- Where FSM diagrams live (`docs/fsm_diagram.png`, `docs/fsm_diagram.mmd`)
- Link to `VALIDATION_REPORT.md`

---

## Key Implementation Notes

### Files to Modify (Science Sim Only)
| Stage | Files |
|-------|-------|
| 0.1 | `src/ANTARA_Science_Sim.py` (MissionConfig + run() flush logic) |
| 0.2 | `src/ANTARA_Science_Sim.py` (MissionConfig + all results_path/figures_path calls) |
| 0.3 | `src/ANTARA_Science_Sim.py` (ANTARAScienceSimulator.__init__) |
| 0.4 | `src/ANTARA_Science_Sim.py` (checkpoint methods + CLI) |
| 1.1 | `src/scenarios.py` (new) |
| 1.2 | `src/ANTARA_Science_Sim.py` (_parse_args + run) |
| 1.3 | `src/ANTARA_Science_Sim.py` (run loop multi-rate) |
| 1.4 | `src/compare_scenarios.py` (new) |
| 2.1-2.6 | `src/leop_detumble_mc.py`, `src/analyze_detumble_mc.py` (new) |
| 3.1-3.3 | `src/fsm_spec.py`, `tests/test_fsm_spec_matches_code.py`, `src/plot_fsm_diagram.py` (new) |
| 4.1-4.7 | `tests/validation/test_*.py` (6 files) + `scripts/generate_validation_report.py` |
| 5.1 | `.github/workflows/ci.yml` (new) |
| 5.2 | `README.md` |

### Testing Strategy
- Unit tests for each new module (Stages 0-3)
- Integration tests for scenario runs (Stage 1)
- Validation tests against real data (Stage 4)
- CI runs smoke tests; nightly runs full scenarios

### Assumptions to Flag
1. **Stage 2 accel bound**: Uniform(0, 1.0) deg/s² is an assumption pending real deployer tip-off data
2. **Stage 4 references**: NOAA IGRF values, AP9 literature values need citations
3. **Checkpoint versioning**: Pickle may break across code changes; JSON fallback needed
4. **Multi-rate validation**: 1%/2-step tolerance needs empirical verification

---

## Dependencies
- Existing: `orekit`, `numpy`, `pandas`, `matplotlib`, `cartopy`, `shapely`, `pyIGRF` (optional), `spacepy` (optional)
- New for Stage 3: `graphviz` (for diagram generation)
- New for Stage 4: `skyfield` or `sgp4` (for orbit validation)

---

## Timeline Estimate
| Stage | Effort | Notes |
|-------|--------|-------|
| 0 | 2-3 days | Core infrastructure, affects all later stages |
| 1 | 2-3 days | Scenario infrastructure + multi-rate tuning |
| 2 | 3-4 days | MC framework + analysis (parallelizable) |
| 3 | 1-2 days | Spec extraction + diagram generation |
| 4 | 3-4 days | 6 validation tests + report generator |
| 5 | 1 day | CI + README |

**Total**: ~12-17 days
