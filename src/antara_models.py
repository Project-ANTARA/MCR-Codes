from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ─── Type aliases (replace Orekit Vector3D) ──────────────────────────────────────
Vector3D = np.ndarray  # flat 3-vector used throughout


# ─── Constants ───────────────────────────────────────────────────────────────────
BODY_AXIS_VECTORS = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=float),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
}


# ─── Helper: vector shape normalisation ──────────────────────────────────────────
def _as_flat_vector(vector) -> np.ndarray:
    """Return a canonical flat 3-vector for diagnostics, geometry, and telemetry.

    Accepts Python sequences and NumPy arrays with shapes (3,), (3, 1), or (1, 3).
    Always returns a flat float array of shape (3,).
    """

    vector = np.asarray(vector, dtype=float)
    if vector.shape == (3, 1):
        vector = vector[:, 0]
    elif vector.shape == (1, 3):
        vector = vector[0]

    vector = vector.reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {vector.shape}")
    return vector


# ─── SunSubmode enum ────────────────────────────────────────────────────────────
class SunSubmode(str, Enum):
    """Power-state submodes of the SUN_POINTING FSM mode."""

    CHARGING = "CHARGING"
    PASSIVE = "PASSIVE"
    ECLIPSE = "ECLIPSE"
    NONE = "NONE"

    def __str__(self) -> str:
        return self.value


# ─── MissionConfig dataclass ────────────────────────────────────────────────────
@dataclass
class MissionConfig:
    """Top-level dataclass holding all mission parameters.
    All numeric constants must live here. No magic numbers in logic.
    """

    """Authoritative spacecraft body frame convention (ICD Rev B):
    +X = RAM face / payload boresight  → aligned to velocity vector (LVLH)
    +Y = directional relay antenna face → aligned to nadir (LVLH)
    +Z = fixed solar panel              → aligned to orbit normal, sun-facing side (LVLH)
    -Y = deployable panel packed face
    -Z = omni antenna face
    -X = PPOD deployment face
    """

    sim_duration_hours: float = 24.0
    step_sec: float = 2.0
    progress_report_interval_sec: float = 180.0
    telemetry_flush_interval_hours: float = 6.0
    output_tag: Optional[str] = None
    start_year: int = 2000
    start_month: int = 3
    start_day: int = 1

    # Orbit definition
    altitude_m: float = 600_000.0
    inclination_deg: float = 97.6
    eccentricity: float = 0.001
    raan_deg: float = 15.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Spacecraft physical properties
    mass_kg: float = 3.0
    drag_coeff: float = 2.2
    cross_section_m2: float = 0.01
    srp_coeff: float = 1.8

    # ADCS performance and actuator limits
    max_slew_rate_deg_sec: float = 5.0
    wheel_max_torque_nm: float = 0.001
    max_slew_accel_deg_sec2: float = 1.0
    wheel_max_momentum_nms: float = 0.01317
    pointing_accuracy_deg: float = 0.4
    settling_time_sec: float = 8.0
    adcs_idle_power_w: float = 0.8
    adcs_slew_power_w: float = 4.5

    # EPS and solar-array properties
    battery_capacity_wh: float = 25.0
    initial_battery_wh: float = 18.0
    solar_panel_efficiency: float = 0.193515
    solar_panel_area_m2: float = 0.085
    solar_constant_w_m2: float = 1361.0
    eps_charge_efficiency: float = 0.88
    solar_temp_derating: float = 0.88

    # EPS charge-state thresholds
    low_power_enter_wh: float = 10.0
    low_power_exit_wh: float = 17.50
    critical_power_wh: float = 7.50
    saa_detection_min_power_wh: float = 10.0
    min_relaybattery_capacity_wh: float = 10.0
    safe_exit_battery_fraction: float = 0.75

    # Electrical loads
    obc_power_w: float = 0.8
    eps_power_w: float = 1.2
    payload_power_w: float = 10.0
    payload_coin_logic_power_w: float = 2.0
    radio_tx_power_w: float = 12.0

    # Storage and data system
    onboard_storage_mb: float = 32_000.0
    initial_storage_mb: float = 10.0
    storage_high_watermark_mb: float = 24_000.0

    # Communications geometry and link rate
    downlink_rate_mbps: float = 0.0192
    gs_elevation_mask_deg: float = 15.0
    gs_lat_deg: float = 15.0
    gs_lon_deg: float = 74.0
    gs_alt_m: float = 70.0

    # Second ground station: Svalbard (78°N, 15°E)
    gs2_enabled: bool = False
    gs2_lat_deg: float = 78.0
    gs2_lon_deg: float = 15.0
    gs2_alt_m: float = 50.0
    gs2_elevation_mask_deg: float = 5.0

    # Payload data generation
    science_data_rate_mbps: float = 0.01152

    # Mode timing, geometric validity, and guidance hysteresis
    startup_cold_sec: float = 120.0
    initial_body_rate_deg_sec_x: float = 6.0
    initial_body_rate_deg_sec_y: float = -5.5
    initial_body_rate_deg_sec_z: float = 4.5
    initial_body_accel_deg_sec2_x: float = 0.12
    initial_body_accel_deg_sec2_y: float = -0.015
    initial_body_accel_deg_sec2_z: float = 0.08
    detumble_target_attitude_label: str = "DEFAULT_ORIENTATION_HOLD"
    inertia_tensor_kgm2: Tuple[float, float, float] = (0.016, 0.016, 0.004)
    bdot_gain: float = 3.0e4
    bdot_handover_rate_deg_sec: float = 2.0
    wheel_rate_damp_gain: float = 0.2
    detumble_stable_rate_deg_sec: float = 0.1
    earth_b_field_model: str = "dipole"
    large_slew_threshold_deg: float = 60.0
    critical_recovery_target_wh: float = 6.5
    attitude_command_deadband_deg: float = 1.5
    safe_hold_deadband_deg: float = 7.5
    ram_track_deadband_deg: float = 5.0
    relay_track_deadband_deg: float = 3.0
    payload_pointing_valid_deg: float = 10.0
    relay_pointing_valid_deg: float = 5.0
    directional_antenna_half_beamwidth_deg: float = 40.0
    detection_pointing_hard_limit_deg: float = 2.0
    relay_pointing_hard_limit_deg: float = 5.0
    sun_pointing_hard_limit_deg: float = 5.0
    charging_complete_threshold_wh: float = 22.5
    minimum_detection_duration_sec: float = 120.0
    minimum_relay_duration_sec: float = 120.0
    max_saturation_hold_sec: float = 300.0
    minimum_settled_time_before_payload_sec: float = 15.0
    minimum_settled_time_before_tx_sec: float = 10.0
    fault_debounce_samples: int = 3
    recovery_debounce_samples: int = 3
    mode_transition_grace_steps: int = 5
    penumbra_transition_sec: float = 30.0
    slew_transition_rate_deg_sec: float = 2.0
    slew_transition_penalty_sec: float = 60.0
    slew_power_penalty_w: float = 5.0
    max_solar_power_w: float = 22.0
    eclipse_heater_power_w: float = 0.8
    battery_taper_soc_start: float = 0.90
    battery_taper_min_fraction: float = 0.20
    wheel_momentum_recovery_fraction: float = 0.80
    magnetorquer_dipole_Am2: float = 0.2
    earth_B_field_T: float = 30.0e-6
    sun_pointing_tracking_noise_deg: float = 2.5
    wheel_inertia_kgm2: float = 1.5e-5
    rng_seed: int = 42

    # Numerical propagator configuration
    min_integrator_step_sec: float = 0.1
    max_integrator_step_sec: float = 120.0
    integrator_position_tolerance_m: float = 10.0
    integrator_velocity_tolerance_mps: float = 0.01

    # Multi-rate stepping (leop_1yr)
    coarse_step_sec: float = 20.0

    # Plotting control
    no_plots: bool = False

    # Plotting and telemetry colors
    mode_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "COLD_AND_DARK": "#111111",
            "DETUMBLE": "#8c564b",
            "SAFE": "#7f7f7f",
            "SLEWING": "#9467bd",
            "SUN_POINTING": "#f1c40f",
            "DETECTION": "#d62728",
            "RELAY": "#1f77b4",
        }
    )
    sun_submode_colors: Dict[str, str] = field(
        default_factory=lambda: {
            SunSubmode.CHARGING: "#f1c40f",
            SunSubmode.PASSIVE: "#ffe680",
            SunSubmode.ECLIPSE: "#4a4a4a",
            SunSubmode.NONE: "#ffffff",
        }
    )

    # ── Science / Payload ─────────────────────────────────────────────────────────
    energy_channels_mev: Tuple[float, ...] = (5.0, 10.0, 20.0, 50.0, 100.0)
    ap9_baseline_flux: Tuple[float, ...] = (3.5e4, 1.8e4, 6.0e3, 9.0e2, 1.1e2)
    detector_efficiency: Tuple[float, ...] = (0.92, 0.90, 0.87, 0.80, 0.68)
    detector_dead_time_sec: float = 1.0e-6
    detector_dark_count_rate_hz: float = 0.5
    saa_depth_flux_min: float = 0.5
    saa_depth_flux_max: float = 2.0
    ew_asymmetry_amplitude: float = 0.15
    storm_time_flux_multiplier: float = 1.5
    igrf_epoch_year: float = 2000.25
    science_histogram_downlink_interval_sec: float = 60.0
    science_packet_bytes_per_bin: float = 4.0
    science_packet_header_bytes: float = 16.0
    enable_periphery_bin_merge: bool = True
    detector_coincidence_window_ns: float = 20.0
    detector_dead_time_ns: float = 250.0

    def __post_init__(self) -> None:
        """Fail fast on physically or logically inconsistent configurations."""
        errors: List[str] = []

        if not (0.0 <= self.eccentricity < 1.0):
            errors.append(f"eccentricity={self.eccentricity} must be in [0, 1)")
        if not (0.0 <= self.raan_deg < 360.0):
            errors.append(f"raan_deg={self.raan_deg} must be in [0, 360)")
        if not (0.0 <= self.inclination_deg <= 180.0):
            errors.append(f"inclination_deg={self.inclination_deg} must be in [0, 180]")
        if self.altitude_m <= 0.0:
            errors.append(f"altitude_m={self.altitude_m} must be positive")
        if self.mass_kg <= 0.0:
            errors.append(f"mass_kg={self.mass_kg} must be positive")
        if self.step_sec <= 0.0:
            errors.append(f"step_sec={self.step_sec} must be positive")
        if self.sim_duration_hours <= 0.0:
            errors.append(f"sim_duration_hours={self.sim_duration_hours} must be positive")

        # EPS hysteresis ordering
        eps_chain = [
            ("critical_power_wh", self.critical_power_wh),
            ("low_power_enter_wh", self.low_power_enter_wh),
            ("low_power_exit_wh", self.low_power_exit_wh),
            ("charging_complete_threshold_wh", self.charging_complete_threshold_wh),
            ("battery_capacity_wh", self.battery_capacity_wh),
        ]
        for (name_a, val_a), (name_b, val_b) in zip(eps_chain, eps_chain[1:]):
            if not (val_a < val_b):
                errors.append(
                    f"EPS threshold ordering violated: {name_a}={val_a} must be < {name_b}={val_b}"
                )
        if not (0.0 <= self.initial_battery_wh <= self.battery_capacity_wh):
            errors.append(
                f"initial_battery_wh={self.initial_battery_wh} must be within "
                f"[0, battery_capacity_wh={self.battery_capacity_wh}]"
            )

        # Tip-off sanity bound
        max_reasonable_tipoff_rate_deg_sec = 50.0
        for axis, rate in (
            ("x", self.initial_body_rate_deg_sec_x),
            ("y", self.initial_body_rate_deg_sec_y),
            ("z", self.initial_body_rate_deg_sec_z),
        ):
            if abs(rate) > max_reasonable_tipoff_rate_deg_sec:
                errors.append(
                    f"initial_body_rate_deg_sec_{axis}={rate} exceeds sanity bound "
                    f"of +/-{max_reasonable_tipoff_rate_deg_sec} deg/s"
                )

        if len(self.energy_channels_mev) != len(self.ap9_baseline_flux):
            errors.append("energy_channels_mev and ap9_baseline_flux length mismatch")
        if len(self.energy_channels_mev) != len(self.detector_efficiency):
            errors.append("energy_channels_mev and detector_efficiency length mismatch")

        if errors:
            raise ValueError("Invalid MissionConfig — " + "; ".join(errors))


# ─── Exception ──────────────────────────────────────────────────────────────────
class IllegalFSMTransitionError(RuntimeError):
    """Raised when the FSM is asked to perform a disallowed transition."""


# ─── SpacecraftFSM ──────────────────────────────────────────────────────────────
class SpacecraftFSM:
    """
    Finite State Machine for ANTARA spacecraft modes.

    States: COLD_AND_DARK, DETUMBLE, SAFE, SLEWING, SUN_POINTING, DETECTION, RELAY

    STRICT transition table (only these are allowed):
      COLD_AND_DARK -> DETUMBLE
      DETUMBLE      -> SAFE, SUN_POINTING, SLEWING
      SAFE          -> SUN_POINTING, RELAY, DETECTION, SLEWING
      SLEWING       -> SAFE, SUN_POINTING, RELAY, DETECTION
      SUN_POINTING  -> SAFE, RELAY, DETECTION, SLEWING
      DETECTION     -> RELAY, SUN_POINTING, SAFE, SLEWING
      RELAY         -> SUN_POINTING, SAFE, DETECTION, SLEWING

    Any attempted illegal transition must raise IllegalFSMTransitionError.

    transition() method signature:
      def transition(self, new_mode: str, reason: str) -> bool

    Decision logic (called each timestep):
      1. Execute the startup sequence COLD_AND_DARK -> DETUMBLE -> SUN_POINTING
      2. SAFE is NOT the nominal operational mode; it is reserved for genuine survival cases
      3. Critical battery may force SAFE
      4. Wheel saturation does not immediately force SAFE; the bus stays in-place unless
         recovery persists too long or attitude hold degrades beyond the safe deadband
      5. Enforce minimum dwell for DETECTION and RELAY before mode exit
      6. If GS visible and battery is adequate: RELAY
      7. If inside SAA and battery is adequate: DETECTION
      8. Otherwise the nominal baseline mode is SUN_POINTING

    Hysteresis is handled by the debounce counters and explicit recovery flags.
    """

    ALLOWED_TRANSITIONS = {
        "COLD_AND_DARK": {"DETUMBLE"},
        "DETUMBLE": {"SAFE", "SUN_POINTING", "SLEWING"},
        "SAFE": {"SUN_POINTING", "RELAY", "DETECTION", "SLEWING"},
        "SLEWING": {"SAFE", "SUN_POINTING", "RELAY", "DETECTION"},
        "SUN_POINTING": {"SAFE", "RELAY", "DETECTION", "SLEWING"},
        "DETECTION": {"RELAY", "SUN_POINTING", "SAFE", "SLEWING"},
        "RELAY": {"SUN_POINTING", "SAFE", "DETECTION", "SLEWING"},
    }

    def __init__(self, config: 'MissionConfig') -> None:
        """Initialise the FSM with the ANTARA startup state."""

        self.config = config
        self.current_mode = "COLD_AND_DARK"
        self.time_sec = 0.0
        self.last_mode_change_time_sec = 0.0
        self.mode_entry_time_sec = 0.0
        self.mode_durations_sec = {mode: 0.0 for mode in self.ALLOWED_TRANSITIONS}
        self.transition_history: List[Dict[str, object]] = []
        self.rejected_transition_history: List[Dict[str, object]] = []
        self.target_mode = self.current_mode
        self.slew_start_time_sec = 0.0
        self.slew_end_time_sec = 0.0
        self.detumble_complete = False
        self.debounce_counters: Dict[str, int] = {}
        self.in_low_power = False
        self.in_critical_recovery = False
        self.post_safe_recovery_hold_active = False
        self.combined_fault_active = False
        self.pointing_budget_fault_active = False
        self.safe_recovery_hold_samples = 0
        self.low_power_hold_samples = 0
        self.critical_fault_active = False
        self.wheel_saturation_fault_active = False

    def set_time(self, time_sec: float) -> None:
        """Store the current simulation time for transition bookkeeping."""

        self.time_sec = time_sec

    def record_rejected_transition(self, from_mode: str, to_mode: str, reason: str) -> None:
        """Record and emit a rejected transition request."""

        message = (
            f"[WARN] Rejected transition {from_mode} -> {to_mode} at t={self.time_sec:.1f}s: {reason}"
        )
        print(message)
        self.rejected_transition_history.append(
            {
                "time_sec": self.time_sec,
                "from_mode": from_mode,
                "to_mode": to_mode,
                "reason": reason,
            }
        )

    def transition(self, new_mode: str, reason: str) -> bool:
        """Execute a mode transition if it is legal and not blocked by hysteresis."""

        if new_mode == self.current_mode:
            return False

        if new_mode not in self.ALLOWED_TRANSITIONS[self.current_mode]:
            raise IllegalFSMTransitionError(f"Illegal transition {self.current_mode} -> {new_mode}")

        old_mode = self.current_mode
        self.current_mode = new_mode
        self.last_mode_change_time_sec = self.time_sec
        self.mode_entry_time_sec = self.time_sec
        self.transition_history.append(
            {
                "time_sec": self.time_sec,
                "from_mode": old_mode,
                "to_mode": new_mode,
                "reason": reason,
            }
        )
        return True

    def accumulate_mode_time(self, dt_sec: float) -> None:
        """Accumulate dwell time in the current mode."""

        self.mode_durations_sec[self.current_mode] += dt_sec

    def _minimum_dwell_for_current_mode_sec(self) -> float:
        """Return the minimum dwell time for the currently active operational mode."""

        if self.current_mode == "DETECTION":
            return self.config.minimum_detection_duration_sec
        if self.current_mode == "RELAY":
            return self.config.minimum_relay_duration_sec
        return 0.0

    def minimum_dwell_complete(self) -> bool:
        """Return whether the current mode has satisfied its minimum dwell duration."""

        required_dwell_sec = self._minimum_dwell_for_current_mode_sec()
        if required_dwell_sec <= 0.0:
            return True
        return (self.time_sec - self.mode_entry_time_sec) >= required_dwell_sec

    def _debounce(self, trigger_name: str, condition: bool, threshold: int) -> bool:
        """Return True once a condition has remained true for the required sample count."""

        if condition:
            self.debounce_counters[trigger_name] = self.debounce_counters.get(trigger_name, 0) + 1
        else:
            self.debounce_counters[trigger_name] = 0
        return self.debounce_counters[trigger_name] >= max(1, threshold)

    def _update_debounce_counters(
        self,
        battery_wh: float,
        adcs_saturation_hold_active: bool,
        saturation_hold_duration_sec: float,
        pointing_budget_error_deg: float,
        pointing_budget_limit_deg: float,
    ) -> None:
        """Refresh persistent trigger counters used by the FSM fault and recovery gates."""

        self.critical_fault_active = self._debounce(
            "critical_battery",
            battery_wh < self.config.critical_power_wh,
            self.config.fault_debounce_samples,
        )
        self.wheel_saturation_fault_active = self._debounce(
            "wheel_saturation",
            adcs_saturation_hold_active,
            self.config.fault_debounce_samples,
        )
        # Pointing budget fault: only evaluate when NOT in SAFE and NOT in saturation hold
        # Once triggered, the fault persists until condition clears (not reset in SAFE)
        if adcs_saturation_hold_active:
            self.debounce_counters["pointing_budget"] = 0
            # Don't clear the fault flag during saturation hold - it should persist
        elif self.current_mode == "SAFE":
            # In SAFE mode, don't evaluate pointing budget fault (already in safe mode)
            # but don't clear the flag either - it should persist until condition clears
            pass
        else:
            self.pointing_budget_fault_active = self._debounce(
                "pointing_budget",
                pointing_budget_error_deg > pointing_budget_limit_deg,
                self.config.fault_debounce_samples,
            )
        self.low_power_hold_samples = self.debounce_counters.get("low_power_recovery", 0)
        self.safe_recovery_hold_samples = self.debounce_counters.get("safe_exit_recovery", 0)

    def _handle_slewing_progression(
        self,
        battery_wh: float,
    ) -> Optional[Tuple[str, str]]:
        """Return the active slewing mode until completion or an overriding SAFE fault."""

        if self.current_mode != "SLEWING":
            return None
        if battery_wh < self.config.critical_power_wh:
            if self._debounce(
                "critical_battery",
                True,
                self.config.fault_debounce_samples,
            ):
                self.in_critical_recovery = True
                self.combined_fault_active = self.wheel_saturation_fault_active
                return "SAFE", "Critical battery during slew"
            return "SLEWING", f"Slewing toward {self.target_mode}"
        if self.time_sec >= self.slew_end_time_sec:
            return self.target_mode, f"Slew complete -> {self.target_mode}"
        return "SLEWING", f"Slewing toward {self.target_mode}"

    def _handle_startup_sequencing(
        self,
        detumble_complete: bool,
    ) -> Optional[Tuple[str, str]]:
        """Handle the startup-only COLD_AND_DARK and DETUMBLE sequencing."""

        if self.current_mode == "COLD_AND_DARK":
            if self.time_sec >= self.config.startup_cold_sec:
                return "DETUMBLE", "Startup cold phase complete"
            return "COLD_AND_DARK", "Remain in cold startup"

        if self.current_mode == "DETUMBLE":
            if detumble_complete:
                return "SUN_POINTING", "Physics detumble complete -> SUN_POINTING"
            return "DETUMBLE", "Remain in detumble startup"

        return None

    def _handle_fault_overrides(
        self,
        battery_wh: float,
        adcs_saturation_hold_active: bool,
        saturation_hold_duration_sec: float,
        pointing_budget_error_deg: float,
        pointing_budget_limit_deg: float,
    ) -> Optional[Tuple[str, str]]:
        """Apply safety overrides, including combined critical-battery and wheel-saturation faults.

        The combined case is intentionally handled first so future ordering edits do
        not accidentally demote it below a single-fault branch.
        """

        combined_fault_condition = self.critical_fault_active and self.wheel_saturation_fault_active
        self.combined_fault_active = combined_fault_condition

        if self.current_mode == "SAFE":
            if combined_fault_condition:
                self.in_critical_recovery = True
                self.post_safe_recovery_hold_active = False
                return "SAFE", "Combined critical battery and wheel saturation survival"

            if self.pointing_budget_fault_active:
                return "SAFE", "Persistent pointing budget violation during freeze"

            if self.in_critical_recovery:
                safe_exit_threshold = self.config.safe_exit_battery_fraction * self.config.battery_capacity_wh
                recovered = (
                    battery_wh >= self.config.critical_recovery_target_wh
                    and battery_wh >= safe_exit_threshold
                )
                if self._debounce("safe_exit_recovery", recovered, self.config.recovery_debounce_samples):
                    self.post_safe_recovery_hold_active = True
                    self.in_critical_recovery = False
                    self.debounce_counters["safe_exit_recovery"] = 0
                    return "SUN_POINTING", "Critical recovery confirmed -> default hold"
                return "SAFE", "Awaiting SAFE recovery conditions"

            if adcs_saturation_hold_active:
                return "SAFE", "Remain in SAFE during wheel saturation recovery"

            return "SUN_POINTING", "SAFE exit -> default hold"

        if self.post_safe_recovery_hold_active and self.current_mode == "SUN_POINTING":
            safe_exit_threshold = self.config.safe_exit_battery_fraction * self.config.battery_capacity_wh
            hold_stable = (
                battery_wh >= self.config.critical_recovery_target_wh
                and battery_wh >= safe_exit_threshold
                and not adcs_saturation_hold_active
                and pointing_budget_error_deg <= self.config.safe_hold_deadband_deg
            )
            if self._debounce("safe_exit_hold", hold_stable, self.config.recovery_debounce_samples):
                self.post_safe_recovery_hold_active = False
                self.debounce_counters["safe_exit_hold"] = 0
                return None
            return "SUN_POINTING", "Post-SAFE recovery hold stabilising"

        if self.critical_fault_active:
            self.in_critical_recovery = True
            self.post_safe_recovery_hold_active = False
            return "SAFE", "Critical battery survival"

        if adcs_saturation_hold_active and self.wheel_saturation_fault_active:
            if battery_wh < self.config.critical_power_wh:
                self.in_critical_recovery = True
                self.combined_fault_active = True
                return "SAFE", "Combined critical battery and wheel saturation survival"
            if (
                saturation_hold_duration_sec > self.config.max_saturation_hold_sec
                and pointing_budget_error_deg > pointing_budget_limit_deg
            ):
                return "SAFE", "Saturation hold timeout with pointing divergence"
            return self.current_mode, "Hold current mode during ADCS saturation unloading"

        if self.pointing_budget_fault_active:
            return "SAFE", "Persistent pointing budget violation"

        return None

    def _select_nominal_mode(
        self,
        battery_wh: float,
        gs_visible: bool,
        inside_saa: bool,
    ) -> Tuple[str, str]:
        """Select the nominal operational mode once startup and fault logic have cleared."""

        if self.in_low_power:
            if self._debounce(
                "low_power_recovery",
                battery_wh > self.config.low_power_exit_wh,
                self.config.recovery_debounce_samples,
            ):
                self.in_low_power = False
                self.debounce_counters["low_power_recovery"] = 0
            else:
                return "SUN_POINTING", "Low-power charging hold"

        if battery_wh < self.config.low_power_enter_wh:
            if self._debounce(
                "low_power_entry",
                True,
                self.config.fault_debounce_samples,
            ):
                self.in_low_power = True
                return "SUN_POINTING", "Low-power charging"
            return "SUN_POINTING", "Low-power charging pending"

        if self.current_mode == "DETECTION" and not self.minimum_dwell_complete():
            return "DETECTION", "Minimum detection dwell not complete"

        if self.current_mode == "RELAY" and not self.minimum_dwell_complete():
            return "RELAY", "Minimum relay dwell not complete"

        if gs_visible and battery_wh > self.config.min_relaybattery_capacity_wh:
            return "RELAY", "Ground station visible with adequate battery"

        if inside_saa and battery_wh > self.config.saa_detection_min_power_wh:
            return "DETECTION", "Inside SAA with sufficient power"

        return "SUN_POINTING", "Nominal SUN_POINTING baseline"

    def evaluate(
        self,
        battery_wh: float,
        gs_visible: bool,
        inside_saa: bool,
        adcs_saturation_hold_active: bool,
        saturation_hold_duration_sec: float,
        pointing_budget_error_deg: float,
        pointing_budget_limit_deg: float,
        detumble_complete: bool,
    ) -> Tuple[str, str]:
        """Evaluate the requested operational mode.

        SAFE is NOT operational nominal mode.
        Eclipse is an attitude condition within SUN_POINTING, not a spacecraft mode.
        """
        self._update_debounce_counters(
            battery_wh,
            adcs_saturation_hold_active,
            saturation_hold_duration_sec,
            pointing_budget_error_deg,
            pointing_budget_limit_deg,
        )

        slew_mode = self._handle_slewing_progression(battery_wh)
        if slew_mode is not None:
            return slew_mode

        startup_mode = self._handle_startup_sequencing(detumble_complete)
        if startup_mode is not None:
            return startup_mode

        fault_mode = self._handle_fault_overrides(
            battery_wh,
            adcs_saturation_hold_active,
            saturation_hold_duration_sec,
            pointing_budget_error_deg,
            pointing_budget_limit_deg,
        )
        if fault_mode is not None:
            return fault_mode

        return self._select_nominal_mode(
            battery_wh,
            gs_visible,
            inside_saa,
        )


# ─── ADCSModel ───────────────────────────────────────────────────────────────────
class ADCSModel:
    """
    Manages attitude state and slew dynamics.

    Attributes:
      - current_quaternion (from Orekit attitude)
      - wheel_momentum_Nms (3-axis float[3])
      - is_slewing (bool)
      - slew_angle_deg (float)
      - slew_time_remaining_sec (float)
      - settling (bool)

    Methods:
      - compute_slew(prev_attitude, new_attitude) -> (angle_deg, time_sec)
      - update_wheel_momentum(torque_vec, dt)
      - check_saturation() -> bool
      - get_pointing_error_deg() -> float

    Quaternion convention:
      - scalar-first [w, x, y, z]
      - maps body-frame vectors into inertial coordinates
      - DCM columns are the inertial representations of +X_body, +Y_body, +Z_body

    Model trapezoidal slew profile: ramp-up, coast, ramp-down.
    Track per-axis (roll, pitch, yaw) from ZYX Euler decomposition.
    Wheel momentum accumulates every timestep; saturation triggers
    a flag (no FDIR needed — just set is_saturated and log it).
    """

    def __init__(self, config: 'MissionConfig') -> None:
        """Initialise ADCS state containers, quaternion state, and running counters."""

        self.config = config
        self.current_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.target_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.current_quaternion_tuple: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        self.wheel_momentum_nms = np.zeros(3, dtype=float)
        self.initial_body_rate_deg_sec = np.array(
            [
                config.initial_body_rate_deg_sec_x,
                config.initial_body_rate_deg_sec_y,
                config.initial_body_rate_deg_sec_z,
            ],
            dtype=float,
        )
        self.initial_body_accel_deg_sec2 = np.array(
            [
                config.initial_body_accel_deg_sec2_x,
                config.initial_body_accel_deg_sec2_y,
                config.initial_body_accel_deg_sec2_z,
            ],
            dtype=float,
        )
        self.body_rate_deg_sec = self.initial_body_rate_deg_sec.copy()
        self.body_accel_deg_sec2 = self.initial_body_accel_deg_sec2.copy()
        self.inertia_tensor_kgm2 = np.array(config.inertia_tensor_kgm2, dtype=float)
        self.is_slewing = False
        self.slew_angle_deg = 0.0
        self.slew_time_remaining_sec = 0.0
        self.settling = False
        self.settling_time_remaining_sec = 0.0
        self.is_saturated = False
        self.saturation_recovery_active = False
        self.saturation_hold_active = False
        self.saturation_start_time_sec: Optional[float] = None
        self.time_sec = 0.0
        self.saturation_event_count = 0
        self.total_slew_time_sec = 0.0
        self.quaternion_continuity_corrections = 0
        self.quaternion_continuity_corrections_by_site: Dict[str, int] = {
            "integration": 0,
            "delta_quaternion": 0,
            "command_attitude": 0,
            "advance": 0,
            "build_frame": 0,
            "unspecified": 0,
        }
        self.rejected_micro_slew_commands = 0
        self.last_euler_delta_deg = np.zeros(3, dtype=float)
        self.current_slew_rate_deg_sec = 0.0
        self.previous_slew_rate_deg_sec = 0.0
        self.current_angular_accel_deg_sec2 = 0.0
        self.attitude_error_deg = 0.0
        self.last_commanded_mode = "COLD_AND_DARK"
        self.last_command_reason = "Initialisation"
        self.commanded_relay_axis = "+Y"
        self.settled_time_sec = 0.0
        self.wheel_speed_rpm_estimate = np.zeros(3, dtype=float)
        self.current_wheel_unload_torque_nm = 0.0
        self.momentum_dump_active = False
        self.detumble_phase = "BDOT"
        self.detumble_complete = False
        self.last_b_body_t: Optional[np.ndarray] = None
        self.current_detumble_body_torque_nm = np.zeros(3, dtype=float)
        self.current_bdot_torque_nm = np.zeros(3, dtype=float)
        self.current_wheel_damp_torque_nm = np.zeros(3, dtype=float)

    def set_time(self, time_sec: float) -> None:
        """Store the simulation time for ADCS recovery timeout bookkeeping."""

        self.time_sec = time_sec

    def _integrate_body_rates(self, torque_vec_nm: np.ndarray, dt_sec: float) -> None:
        """Integrate rigid-body angular rates under the applied body torque."""

        torque_vec_nm = _as_flat_vector(torque_vec_nm)
        omega_rad_sec = np.radians(self.body_rate_deg_sec)
        if not np.all(np.isfinite(omega_rad_sec)):
            self.body_rate_deg_sec[:] = 0.0
            self.body_accel_deg_sec2[:] = 0.0
            return
        inertia = np.maximum(self.inertia_tensor_kgm2, 1.0e-9)
        angular_momentum = inertia * omega_rad_sec
        gyro_term = np.cross(omega_rad_sec, angular_momentum)
        omega_dot_rad_sec2 = (torque_vec_nm - gyro_term) / inertia
        self.body_accel_deg_sec2 = np.degrees(omega_dot_rad_sec2)
        omega_rad_sec = omega_rad_sec + (omega_dot_rad_sec2 * dt_sec)
        updated_body_rate_deg_sec = np.degrees(omega_rad_sec)
        if not np.all(np.isfinite(updated_body_rate_deg_sec)):
            updated_body_rate_deg_sec = np.zeros(3, dtype=float)
        rate_cap_deg_sec = max(120.0, 10.0 * self.config.max_slew_rate_deg_sec)
        self.body_rate_deg_sec = np.clip(updated_body_rate_deg_sec, -rate_cap_deg_sec, rate_cap_deg_sec)

    def _integrate_quaternion_from_body_rates(self, dt_sec: float) -> None:
        """Advance the body-to-inertial quaternion from the current body rates.

        FIX Bug 11: added explicit re-normalization after every integration step.
        Without it, floating-point drift accumulates over thousands of steps and
        triggers _enforce_quaternion_continuity sign flips at ~71/hr (1,717/day).
        """

        omega_deg_sec = _as_flat_vector(self.body_rate_deg_sec)
        omega_mag_deg_sec = float(np.linalg.norm(omega_deg_sec))
        if not math.isfinite(omega_mag_deg_sec) or omega_mag_deg_sec <= 0.0 or dt_sec <= 0.0:
            self._update_quaternion_tuple()
            return
        axis_unit = omega_deg_sec / omega_mag_deg_sec
        step_quaternion = self.quaternion_from_axis_angle(axis_unit, omega_mag_deg_sec * dt_sec)
        self.current_quaternion = self._normalize_quaternion(
            self._quaternion_multiply(self.current_quaternion, step_quaternion)
        )
        self.current_quaternion = self._enforce_quaternion_continuity(
            self.target_quaternion,
            self.current_quaternion,
            call_site="integration",
        )
        # FIX Bug 11: re-normalize after continuity correction to eliminate any
        # residual floating-point drift introduced by the sign flip itself.
        self.current_quaternion = self._normalize_quaternion(self.current_quaternion)
        self._update_quaternion_tuple()

    def bdot_control(self, b_body_t: np.ndarray, dt_sec: float) -> np.ndarray:
        """Return B-dot magnetorquer body torque using finite-difference dB/dt."""

        b_body_t = _as_flat_vector(b_body_t)
        if self.last_b_body_t is None or dt_sec <= 0.0:
            self.last_b_body_t = b_body_t.copy()
            self.current_bdot_torque_nm = np.zeros(3, dtype=float)
            return self.current_bdot_torque_nm

        bdot_t_sec = (b_body_t - self.last_b_body_t) / dt_sec
        self.last_b_body_t = b_body_t.copy()
        b_norm_sq = float(np.dot(b_body_t, b_body_t))
        if b_norm_sq <= 1.0e-18:
            self.current_bdot_torque_nm = np.zeros(3, dtype=float)
            return self.current_bdot_torque_nm

        desired_m_dipole = -self.config.bdot_gain * bdot_t_sec
        dipole_norm = float(np.linalg.norm(desired_m_dipole))
        if dipole_norm > self.config.magnetorquer_dipole_Am2:
            desired_m_dipole *= self.config.magnetorquer_dipole_Am2 / dipole_norm
        self.current_bdot_torque_nm = np.cross(desired_m_dipole, b_body_t)
        return self.current_bdot_torque_nm

    def wheel_damp_control(self) -> np.ndarray:
        """Return wheel torque opposing the current body-rate vector.

        ROOT-CAUSE FIX — Euler-step instability with scalar gain:
        The original used tau_i = -gain * omega_i with a single scalar gain.
        Euler integration is stable only when gain * dt / I_i < 1 on EVERY
        axis.  With gain=0.01, dt=2 s, I_z=0.004 kg*m^2 the ratio was 5.0 —
        5x above the stability limit — causing z-axis rates to grow by a factor
        of 5 per step.  x/y axes (I=0.016) gave ratio 1.25, also unstable.

        Fix: per-axis inertia-scaled damping:
            tau_i = -k_damp * I_i * omega_i,  k_damp = 0.4 / dt
        This gives gain*dt/I_i = 0.4 on every axis regardless of the inertia
        asymmetry, providing identical stability margin on x, y, and z and
        settling each axis in ~5 timesteps.
        """
        k_damp = self.config.wheel_rate_damp_gain
        omega_rad_sec = np.radians(self.body_rate_deg_sec)
        inertia = np.maximum(self.inertia_tensor_kgm2, 1.0e-9)
        desired_torque = -k_damp * inertia * omega_rad_sec
        self.current_wheel_damp_torque_nm = np.clip(
            desired_torque,
            -self.config.wheel_max_torque_nm,
            self.config.wheel_max_torque_nm,
        )
        return self.current_wheel_damp_torque_nm

    def step_detumble(self, b_body_t: np.ndarray, dt_sec: float) -> np.ndarray:
        """Advance the two-stage detumble controller and return wheel torque demand."""

        if self.detumble_complete:
            self.current_bdot_torque_nm = np.zeros(3, dtype=float)
            self.current_wheel_damp_torque_nm = np.zeros(3, dtype=float)
            self.current_detumble_body_torque_nm = np.zeros(3, dtype=float)
            self.body_accel_deg_sec2 = np.zeros(3, dtype=float)
            self.body_rate_deg_sec[:] = 0.0
            return np.zeros(3, dtype=float)

        max_rate_deg_sec = float(np.max(np.abs(self.body_rate_deg_sec)))
        if self.detumble_phase == "BDOT" and max_rate_deg_sec <= self.config.bdot_handover_rate_deg_sec:
            self.detumble_phase = "WHEEL_DAMP"

        bdot_torque = self.bdot_control(b_body_t, dt_sec)
        wheel_torque = np.zeros(3, dtype=float)
        if self.detumble_phase == "WHEEL_DAMP":
            wheel_torque = self.wheel_damp_control()
        else:
            self.current_wheel_damp_torque_nm = np.zeros(3, dtype=float)

        self.current_detumble_body_torque_nm = bdot_torque + wheel_torque
        self._integrate_body_rates(self.current_detumble_body_torque_nm, dt_sec)
        self._integrate_quaternion_from_body_rates(dt_sec)

        if float(np.max(np.abs(self.body_rate_deg_sec))) <= self.config.detumble_stable_rate_deg_sec:
            self.detumble_phase = "COMPLETE"
            self.detumble_complete = True
            self.body_rate_deg_sec[:] = 0.0
            self.body_accel_deg_sec2[:] = 0.0

        return wheel_torque

    def _normalize_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Return a unit-norm scalar-first quaternion."""

        norm = np.linalg.norm(quaternion)
        if norm <= 0.0:
            raise ValueError("Cannot normalise a zero quaternion.")
        return quaternion / norm

    def _enforce_quaternion_continuity(
        self,
        previous_quaternion: np.ndarray,
        new_quaternion: np.ndarray,
        call_site: str = "unspecified",
    ) -> np.ndarray:
        """Flip quaternion sign when needed to avoid double-cover 180 deg artifacts."""

        if float(np.dot(previous_quaternion, new_quaternion)) < 0.0:
            self.quaternion_continuity_corrections += 1
            self.quaternion_continuity_corrections_by_site[call_site] = self.quaternion_continuity_corrections_by_site.get(call_site, 0) + 1
            return -new_quaternion
        return new_quaternion

    def _quaternion_conjugate(self, quaternion: np.ndarray) -> np.ndarray:
        """Return the conjugate of a scalar-first quaternion."""

        return np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )

    def _quaternion_multiply(self, first_quaternion: np.ndarray, second_quaternion: np.ndarray) -> np.ndarray:
        """Multiply two scalar-first quaternions."""

        w1, x1, y1, z1 = first_quaternion
        w2, x2, y2, z2 = second_quaternion
        return np.array(
            [
                (w1 * w2) - (x1 * x2) - (y1 * y2) - (z1 * z2),
                (w1 * x2) + (x1 * w2) + (y1 * z2) - (z1 * y2),
                (w1 * y2) - (x1 * z2) + (y1 * w2) + (z1 * x2),
                (w1 * z2) + (x1 * y2) - (y1 * x2) + (z1 * w2),
            ],
            dtype=float,
        )

    def _quaternion_to_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Convert a scalar-first quaternion into a body-to-inertial rotation matrix."""

        quaternion = self._normalize_quaternion(quaternion)
        w, x, y, z = quaternion
        return np.array(
            [
                [1.0 - (2.0 * ((y * y) + (z * z))), 2.0 * ((x * y) - (z * w)), 2.0 * ((x * z) + (y * w))],
                [2.0 * ((x * y) + (z * w)), 1.0 - (2.0 * ((x * x) + (z * z))), 2.0 * ((y * z) - (x * w))],
                [2.0 * ((x * z) - (y * w)), 2.0 * ((y * z) + (x * w)), 1.0 - (2.0 * ((x * x) + (y * y)))],
            ],
            dtype=float,
        )

    def get_body_to_inertial_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Return the body-to-inertial DCM for the supplied scalar-first quaternion."""

        return self._quaternion_to_matrix(quaternion)

    def get_inertial_to_body_matrix(self, quaternion: np.ndarray) -> np.ndarray:
        """Return the inertial-to-body DCM for the supplied scalar-first quaternion."""

        return self.get_body_to_inertial_matrix(quaternion).T

    def quaternion_from_matrix(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """Convert a body-to-inertial DCM into a scalar-first quaternion."""

        matrix = np.array(rotation_matrix, dtype=float)
        trace_value = float(np.trace(matrix))
        if trace_value > 0.0:
            scale = math.sqrt(trace_value + 1.0) * 2.0
            w = 0.25 * scale
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
        quaternion = np.array([w, x, y, z], dtype=float)
        return self._normalize_quaternion(quaternion)

    def quaternion_from_axis_angle(self, axis_unit: np.ndarray, angle_deg: float) -> np.ndarray:
        """Build a scalar-first quaternion from axis-angle form."""

        axis_unit = _as_flat_vector(axis_unit)
        if not np.all(np.isfinite(axis_unit)):
            raise ValueError("Axis-angle quaternion received a non-finite axis.")
        axis_norm = float(np.linalg.norm(axis_unit))
        if axis_norm <= 0.0:
            raise ValueError("Axis-angle quaternion received a zero-length axis.")
        axis_unit = axis_unit / axis_norm

        if not math.isfinite(angle_deg):
            angle_deg = 0.0
        angle_deg = math.fmod(angle_deg, 360.0)
        half_angle_rad = math.radians(angle_deg) * 0.5
        sine_term = math.sin(half_angle_rad)
        quaternion = np.array(
            [
                math.cos(half_angle_rad),
                axis_unit[0] * sine_term,
                axis_unit[1] * sine_term,
                axis_unit[2] * sine_term,
            ],
            dtype=float,
        )
        return self._normalize_quaternion(quaternion)

    def rotate_body_axis_to_inertial(self, quaternion: np.ndarray, body_axis: np.ndarray) -> np.ndarray:
        """Rotate a body-frame axis into inertial coordinates."""

        return _as_flat_vector(self.get_body_to_inertial_matrix(quaternion) @ _as_flat_vector(body_axis))

    def rotate_inertial_vector_to_body(self, quaternion: np.ndarray, inertial_vector: np.ndarray) -> np.ndarray:
        """Rotate an inertial-frame vector into body coordinates."""

        return _as_flat_vector(self.get_inertial_to_body_matrix(quaternion) @ _as_flat_vector(inertial_vector))

    def compute_delta_quaternion(
        self,
        previous_quaternion: np.ndarray,
        new_quaternion: np.ndarray,
    ) -> np.ndarray:
        """Compute the relative quaternion from previous attitude to new attitude."""

        continuous_new_quaternion = self._enforce_quaternion_continuity(
            previous_quaternion,
            new_quaternion,
            call_site="delta_quaternion",
        )
        return self._normalize_quaternion(
            self._quaternion_multiply(
                continuous_new_quaternion,
                self._quaternion_conjugate(previous_quaternion),
            )
        )

    def compute_slew(self, previous_quaternion: np.ndarray, new_quaternion: np.ndarray) -> Tuple[float, float]:
        """Compute slew angle and commanded duration between two scalar-first quaternions."""

        delta_quaternion = self.compute_delta_quaternion(previous_quaternion, new_quaternion)
        angle_deg = math.degrees(2.0 * math.acos(np.clip(abs(delta_quaternion[0]), -1.0, 1.0)))
        self.attitude_error_deg = angle_deg

        delta_matrix = self._quaternion_to_matrix(delta_quaternion)
        yaw_rad = math.atan2(delta_matrix[1, 0], delta_matrix[0, 0])
        pitch_rad = math.asin(np.clip(-delta_matrix[2, 0], -1.0, 1.0))
        roll_rad = math.atan2(delta_matrix[2, 1], delta_matrix[2, 2])
        self.last_euler_delta_deg = np.degrees(np.array([yaw_rad, pitch_rad, roll_rad], dtype=float))

        max_rate = self.config.max_slew_rate_deg_sec
        max_accel = self.config.max_slew_accel_deg_sec2
        ramp_time_sec = max_rate / max_accel
        triangular_limit_deg = (max_rate * max_rate) / max_accel

        if angle_deg < triangular_limit_deg:
            slew_body_time_sec = 2.0 * math.sqrt(max(angle_deg, 0.0) / max_accel)
        else:
            coast_angle_deg = angle_deg - triangular_limit_deg
            coast_time_sec = coast_angle_deg / max_rate
            slew_body_time_sec = (2.0 * ramp_time_sec) + coast_time_sec

        if angle_deg <= self.config.pointing_accuracy_deg:
            return angle_deg, 0.0
        return angle_deg, slew_body_time_sec + self.config.settling_time_sec

    def command_attitude(
        self,
        target_quaternion: np.ndarray,
        mode: str,
        reason: str,
        force_update: bool = False,
        deadband_deg: Optional[float] = None,
    ) -> bool:
        """Update the target quaternion when the new objective exceeds the command deadband."""

        if self.saturation_hold_active:
            # During wheel saturation recovery the vehicle holds the current inertial
            # attitude and rejects fresh slew commands to prevent momentum growth.
            self.target_quaternion = self.current_quaternion.copy()
            self.last_command_reason = "Saturation hold active"
            self.attitude_error_deg = 0.0
            self.is_slewing = False
            self.settling = False
            self.current_slew_rate_deg_sec = 0.0
            self.slew_time_remaining_sec = 0.0
            return False

        effective_deadband = deadband_deg if deadband_deg is not None else self.config.attitude_command_deadband_deg
        target_quaternion = self._normalize_quaternion(target_quaternion)
        target_quaternion = self._enforce_quaternion_continuity(self.current_quaternion, target_quaternion, call_site="command_attitude")
        delta_from_existing_target_deg, _ = self.compute_slew(self.target_quaternion, target_quaternion)

        if not force_update and delta_from_existing_target_deg < effective_deadband:
            self.rejected_micro_slew_commands += 1
            self.attitude_error_deg, _ = self.compute_slew(self.current_quaternion, self.target_quaternion)
            return False

        self.target_quaternion = target_quaternion
        self.last_commanded_mode = mode
        self.last_command_reason = reason
        self.slew_angle_deg, self.slew_time_remaining_sec = self.compute_slew(
            self.current_quaternion,
            self.target_quaternion,
        )
        self.is_slewing = self.slew_angle_deg > self.config.pointing_accuracy_deg
        self.settling = False
        self.settling_time_remaining_sec = 0.0
        self.current_slew_rate_deg_sec = 0.0
        return self.is_slewing

    def advance(self, dt_sec: float) -> None:
        """Advance the current attitude toward the target using finite slew-rate and acceleration limits."""

        self.attitude_error_deg, _ = self.compute_slew(self.current_quaternion, self.target_quaternion)
        if self.attitude_error_deg <= self.config.pointing_accuracy_deg:
            self.current_quaternion = self.target_quaternion.copy()
            self.current_slew_rate_deg_sec = 0.0
            self.current_angular_accel_deg_sec2 = 0.0
            if self.settling_time_remaining_sec > 0.0:
                self.settling_time_remaining_sec = max(0.0, self.settling_time_remaining_sec - dt_sec)
                self.settling = self.settling_time_remaining_sec > 0.0
                self.settled_time_sec = 0.0
            else:
                self.settling = False
                self.is_slewing = False
                self.slew_time_remaining_sec = 0.0
                self.settled_time_sec += dt_sec
            self._update_quaternion_tuple()
            return

        self.is_slewing = True
        self.settled_time_sec = 0.0
        stopping_angle_deg = (self.current_slew_rate_deg_sec ** 2) / (2.0 * self.config.max_slew_accel_deg_sec2)
        previous_slew_rate_deg_sec = self.current_slew_rate_deg_sec
        if self.attitude_error_deg <= stopping_angle_deg:
            self.current_slew_rate_deg_sec = max(
                0.0,
                self.current_slew_rate_deg_sec - (self.config.max_slew_accel_deg_sec2 * dt_sec),
            )
        else:
            self.current_slew_rate_deg_sec = min(
                self.config.max_slew_rate_deg_sec,
                self.current_slew_rate_deg_sec + (self.config.max_slew_accel_deg_sec2 * dt_sec),
            )
        self.current_angular_accel_deg_sec2 = (
            (self.current_slew_rate_deg_sec - previous_slew_rate_deg_sec) / dt_sec if dt_sec > 0.0 else 0.0
        )
        self.previous_slew_rate_deg_sec = previous_slew_rate_deg_sec

        step_angle_deg = min(
            self.attitude_error_deg,
            max(self.config.pointing_accuracy_deg, self.current_slew_rate_deg_sec * dt_sec),
        )
        delta_quaternion = self.compute_delta_quaternion(self.current_quaternion, self.target_quaternion)
        delta_vector = delta_quaternion[1:]
        delta_vector_norm = float(np.linalg.norm(delta_vector))
        if delta_vector_norm <= 0.0:
            self.current_quaternion = self.target_quaternion.copy()
        else:
            step_axis_unit = delta_vector / delta_vector_norm
            step_quaternion = self.quaternion_from_axis_angle(step_axis_unit, step_angle_deg)
            self.current_quaternion = self._normalize_quaternion(
                self._quaternion_multiply(step_quaternion, self.current_quaternion)
            )
            self.current_quaternion = self._enforce_quaternion_continuity(
                self.target_quaternion,
                self.current_quaternion,
                call_site="advance",
            )

        self.total_slew_time_sec += dt_sec
        self.attitude_error_deg, self.slew_time_remaining_sec = self.compute_slew(
            self.current_quaternion,
            self.target_quaternion,
        )
        if self.attitude_error_deg <= self.config.pointing_accuracy_deg:
            self.current_quaternion = self.target_quaternion.copy()
            self.is_slewing = False
            self.settling = True
            self.settling_time_remaining_sec = self.config.settling_time_sec
            self.current_slew_rate_deg_sec = 0.0
            self.current_angular_accel_deg_sec2 = 0.0
        self._update_quaternion_tuple()

    def estimate_torque_vector(self) -> np.ndarray:
        """Estimate per-axis wheel torque demand from the latest commanded Euler-angle change."""

        # ROOT-CAUSE FIX: do not let tiny hold/repointing updates accumulate wheel
        # momentum unless we are executing an explicit slew command.  The previous
        # logic kept integrating small continuity-driven attitude updates during
        # nominal mode refreshes, which slowly saturated the wheels.
        if not self.is_slewing or self.last_commanded_mode != "SLEWING":
            return np.zeros(3, dtype=float)

        if float(np.max(np.abs(self.last_euler_delta_deg))) <= max(self.config.pointing_accuracy_deg, 0.1):
            return np.zeros(3, dtype=float)

        axis_sign = np.sign(self.last_euler_delta_deg)
        accel_fraction = min(
            1.0,
            abs(self.current_angular_accel_deg_sec2) / max(self.config.max_slew_accel_deg_sec2, 1.0e-9),
        )
        return axis_sign * self.config.wheel_max_torque_nm * accel_fraction

    def update_wheel_momentum(
        self,
        torque_vec: np.ndarray,
        dt_sec: float,
        unload_scale: float,
        b_projection_scale: float,
    ) -> None:
        """Integrate reaction wheel momentum using the applied body torque.

        Momentum unloading is modelled as a continuous magnetorquer dipole acting against
        the stored wheel momentum. The unloading rate is proportional to the available
        magnetic dipole (0.2 A·m² peak) interacting with a representative equatorial B-field
        (~30 µT at 600 km SSO), giving a peak torque of ~6 µN·m.  This produces a time
        constant of roughly wheel_max_momentum / peak_torque ≈ 0.0015 / 6e-6 ≈ 250 s,
        realistic for a 3U CubeSat at this altitude.  The sign always opposes the current
        stored momentum vector on each axis.
        """
        self.wheel_momentum_nms += torque_vec * dt_sec

        # Magnetorquer unloading: continuous bleed opposing stored momentum.
        # Peak unloading torque per axis (N·m): dipole_Am2 × B_T
        peak_unload_torque_nm = (
            self.config.magnetorquer_dipole_Am2
            * self.config.earth_B_field_T
            * unload_scale
            * b_projection_scale
        )
        self.current_wheel_unload_torque_nm = peak_unload_torque_nm
        self.momentum_dump_active = peak_unload_torque_nm > 0.0 and np.any(np.abs(self.wheel_momentum_nms) > 0.0)
        unload_delta = np.sign(self.wheel_momentum_nms) * peak_unload_torque_nm * dt_sec
        # Never overshoot zero — clamp so we don't reverse momentum direction
        self.wheel_momentum_nms = np.where(
            np.abs(unload_delta) >= np.abs(self.wheel_momentum_nms),
            0.0,
            self.wheel_momentum_nms - unload_delta,
        )
        wheel_speed_rad_sec = self.wheel_momentum_nms / max(self.config.wheel_inertia_kgm2, 1.0e-12)
        self.wheel_speed_rpm_estimate = wheel_speed_rad_sec * (60.0 / (2.0 * math.pi))
        self.check_saturation()

    def check_saturation(self) -> bool:
        """Check whether any wheel exceeds the configured momentum capacity."""

        max_wheel_momentum_abs = float(np.max(np.abs(self.wheel_momentum_nms)))
        previously_saturated = self.is_saturated
        self.is_saturated = max_wheel_momentum_abs >= self.config.wheel_max_momentum_nms
        if self.is_saturated and not previously_saturated:
            self.saturation_event_count += 1
            self.saturation_recovery_active = True
            self.saturation_hold_active = True
            self.saturation_start_time_sec = self.time_sec
            # Freeze the current inertial attitude and let the magnetorquer model
            # unload momentum before accepting new slew requests.
            self.target_quaternion = self.current_quaternion.copy()
            self.is_slewing = False
            self.settling = False
            self.slew_time_remaining_sec = 0.0
            self.settling_time_remaining_sec = 0.0
            self.current_slew_rate_deg_sec = 0.0

        recovery_threshold_nms = (
            self.config.wheel_max_momentum_nms * self.config.wheel_momentum_recovery_fraction
        )
        if self.saturation_recovery_active and max_wheel_momentum_abs <= recovery_threshold_nms:
            self.saturation_recovery_active = False
            self.saturation_hold_active = False
            self.saturation_start_time_sec = None
            self.is_saturated = False

        return self.is_saturated

    def get_saturation_hold_duration_sec(self) -> float:
        """Return the elapsed saturation-hold duration used by the spacecraft FSM."""

        if not self.saturation_hold_active or self.saturation_start_time_sec is None:
            return 0.0
        return max(0.0, self.time_sec - self.saturation_start_time_sec)

    def get_pointing_error_deg(self) -> float:
        """Return a simple PDR-grade pointing error estimate based on active slew state."""

        if self.saturation_hold_active:
            return self.config.pointing_accuracy_deg
        if self.is_slewing:
            return max(self.config.pointing_accuracy_deg, self.attitude_error_deg)
        if self.settling:
            return max(self.config.pointing_accuracy_deg, 0.5)
        return self.config.pointing_accuracy_deg

    def _update_quaternion_tuple(self) -> None:
        """Refresh the telemetry-friendly tuple copy of the current quaternion."""

        self.current_quaternion_tuple = tuple(float(component) for component in self.current_quaternion)

    def set_initial_quaternion(self, quaternion: np.ndarray) -> None:
        """Set the starting quaternion without generating an artificial slew event."""

        normalised_quaternion = self._normalize_quaternion(quaternion)
        self.current_quaternion = normalised_quaternion.copy()
        self.target_quaternion = normalised_quaternion.copy()
        self.attitude_error_deg = 0.0
        self._update_quaternion_tuple()


# ─── EPSModel ───────────────────────────────────────────────────────────────────
class EPSModel:
    """
    Power system in Wh units.

    Attributes:
      - battery_wh (float)
      - solar_generation_w (float)
      - load_w (float)

    Methods:
      - compute_solar_power(sun_angle_deg, illumination_fraction) -> watts
      - compute_load(mode, adcs_slewing, in_eclipse) -> watts
      - step(dt_sec) -> (net_power_w, battery_wh)

    Solar model: P = η * A * G0 * cos(θ) * illumination_fraction
    where G0 = 1361 W/m², η = SOLAR_PANEL_EFFICIENCY,
    A = SOLAR_PANEL_AREA_M2, θ = sun incidence on +Z panel.

    Per-mode loads (W):
      COLD_AND_DARK / DETUMBLE: OBC + EPS baseline only
      SAFE:          OBC + EPS + ADCS_IDLE
      SLEWING:       OBC + EPS + ADCS_SLEW + slew penalty
      SUN_POINTING:  OBC + EPS + ADCS_IDLE (or ADCS_SLEW if slewing)
      DETECTION:     OBC + EPS + ADCS_IDLE + PAYLOAD
      RELAY:         OBC + EPS + ADCS_IDLE + RADIO_TX
    """

    def __init__(self, config: 'MissionConfig') -> None:
        """Initialise the EPS energy state."""

        self.config = config
        self.battery_wh = config.initial_battery_wh
        self.solar_generation_w = 0.0
        self.load_w = 0.0

    def compute_solar_power(self, sun_angle_deg: float, illumination_fraction: float) -> float:
        """Compute instantaneous solar generation on the +Z panel.

        P = η × A × G0 × cos(θ) × η_charge × η_temp
        Temperature derating accounts for cell operating temperature (~60°C on-orbit BOL).
        """
        if illumination_fraction <= 0.0:
            self.solar_generation_w = 0.0
            return self.solar_generation_w

        cosine_incidence = max(0.0, math.cos(math.radians(sun_angle_deg)))
        self.solar_generation_w = (
            self.config.solar_panel_efficiency
            * self.config.solar_panel_area_m2
            * self.config.solar_constant_w_m2
            * cosine_incidence
            * self.config.eps_charge_efficiency
            * self.config.solar_temp_derating
            * float(np.clip(illumination_fraction, 0.0, 1.0))
        )
        self.solar_generation_w = min(self.solar_generation_w, self.config.max_solar_power_w)
        return self.solar_generation_w

    def compute_load(self, mode: str, adcs_slewing: bool, in_eclipse: bool, payload_pending: bool = False) -> float:
        """Compute the current spacecraft electrical load for the active mode.

        `mode` is always the top-level FSM mode string (e.g. 'SUN_POINTING').
        Sun submodes (CHARGING / PASSIVE / ECLIPSE) are attitude-layer concepts
        and are never passed here; they do not change the electrical load budget.
        """

        baseline_w = self.config.obc_power_w + self.config.eps_power_w
        if in_eclipse:
            baseline_w += self.config.eclipse_heater_power_w
        adcs_power_w = self.config.adcs_slew_power_w if adcs_slewing else self.config.adcs_idle_power_w
        payload_power_w = self.config.payload_power_w if (mode == "DETECTION" or payload_pending) else 0.0
        if mode in {"COLD_AND_DARK", "DETUMBLE"}:
            self.load_w = baseline_w
        elif mode == "SAFE":
            self.load_w = baseline_w + adcs_power_w
        elif mode == "SLEWING":
            self.load_w = baseline_w + adcs_power_w + self.config.slew_power_penalty_w
        elif mode == "SUN_POINTING":
            self.load_w = baseline_w + adcs_power_w
        elif mode == "DETECTION":
            self.load_w = baseline_w + adcs_power_w + payload_power_w
        elif mode == "RELAY":
            self.load_w = baseline_w + adcs_power_w + self.config.radio_tx_power_w
        else:
            raise ValueError(f"Unknown EPS load mode: {mode}")
        if payload_pending and mode == "SLEWING":
            self.load_w += self.config.payload_coin_logic_power_w
        return self.load_w

    def step(self, dt_sec: float) -> Tuple[float, float]:
        """Advance battery state using net generation minus load over the current timestep."""

        net_power_w = self.solar_generation_w - self.load_w
        soc_fraction = self.battery_wh / max(self.config.battery_capacity_wh, 1.0e-9)
        if net_power_w > 0.0 and soc_fraction >= self.config.battery_taper_soc_start:
            taper_fraction = 1.0 - (
                (soc_fraction - self.config.battery_taper_soc_start)
                / max(1.0 - self.config.battery_taper_soc_start, 1.0e-9)
            )
            taper_fraction = float(np.clip(taper_fraction, self.config.battery_taper_min_fraction, 1.0))
            net_power_w *= taper_fraction
        delta_wh = net_power_w * (dt_sec / 3600.0)
        self.battery_wh = float(
            np.clip(self.battery_wh + delta_wh, 0.0, self.config.battery_capacity_wh)
        )
        return net_power_w, self.battery_wh