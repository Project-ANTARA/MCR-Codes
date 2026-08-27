from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import orekit
import pandas as pd
from paths import figures_path, init_orekit_data, results_path

try:
    from orekit import JArray_double
except ImportError:
    JArray_double = None

from shapely.geometry import Point, Polygon

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.attitudes import NadirPointing
from org.orekit.bodies import CelestialBodyFactory, GeodeticPoint, OneAxisEllipsoid
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import (
    IsotropicRadiationSingleCoefficient,
    SolarRadiationPressure,
)
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.models.earth.atmosphere import DTM2000, NRLMSISE00
from org.orekit.models.earth.atmosphere.data import (
    CssiSpaceWeatherData,
    MarshallSolarActivityFutureEstimation,
)
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions

BODY_AXIS_VECTORS = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=float),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
}


def _as_flat_vector(vector) -> np.ndarray:
    """Return a canonical flat 3-vector for diagnostics, geometry, and telemetry.

    This is the single shape-normalisation boundary for all vector-like inputs.
    It accepts Orekit Vector3D objects, Python sequences, and NumPy arrays with
    shapes (3,), (3, 1), or (1, 3), and always returns a flat float array.
    """

    if isinstance(vector, Vector3D):
        vector = np.array([vector.getX(), vector.getY(), vector.getZ()], dtype=float)

    vector = np.asarray(vector, dtype=float)
    if vector.shape == (3, 1):
        vector = vector[:, 0]
    elif vector.shape == (1, 3):
        vector = vector[0]

    vector = vector.reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {vector.shape}")
    return vector


class SunSubmode(str, Enum):
    """Power-state submodes of the SUN_POINTING FSM mode.

    The spacecraft holds the same orbit-normal inertial attitude in all three
    submodes. Submodes describe the POWER condition, not the attitude target.

    CHARGING - Sunlit and battery actively gaining charge.
    PASSIVE  - Sunlit and battery full. Coasting state.
    ECLIPSE  - Spacecraft in umbra. Solar generation = 0.
    NONE     - Current FSM mode is not SUN_POINTING.
    """

    CHARGING = "CHARGING"
    PASSIVE = "PASSIVE"
    ECLIPSE = "ECLIPSE"
    NONE = "NONE"

    def __str__(self) -> str:
        return self.value


@dataclass
class MissionConfig:
    """Top-level dataclass holding all mission parameters.
    All numeric constants must live here. No magic numbers in logic."""

    """Authoritative spacecraft body frame convention (ICD Rev B):
    +X = RAM face / payload boresight
    +Y = directional relay antenna face (nominally nadir-pointing in default hold)
    +Z = fixed solar panel
    -Y = deployable panel packed face
    -Z = omni antenna face
    -X = PPOD deployment face
    
    Simulation timeline
    The operational philosophy is conservative: the vehicle holds inertially
    unless it must actively Sun-track, recover from an off-nominal state,
    or perform a deliberate mode-transition slew.
    Startup is handled separately through a detumble phase that settles the
    bus onto the nominal default orientation before mission operations begin."""

    sim_duration_hours: float = 12.0
    step_sec: float = 2.0
    progress_report_interval_sec: float = 180.0
    start_year: int = 2000
    start_month: int = 3
    start_day: int = 1

    # Orbit definition
    altitude_m: float = 600_000.0
    inclination_deg: float = 97.6
    eccentricity: float = 0.001
    raan_deg: float = 60.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Spacecraft physical properties
    mass_kg: float = 3.0           # 3U CubeSat typical wet mass
    drag_coeff: float = 2.2
    cross_section_m2: float = 0.01  # 3U face: 10cm × 10cm = 0.01 m²
    srp_coeff: float = 1.8

    # ADCS performance and actuator limits
    # Based on Tensor Tech Datasheet v1.2.3b and ICD v1.3.1b
    max_slew_rate_deg_sec: float = 5.0
    # Datasheet specifies acceleration torque up to 1 mNm
    wheel_max_torque_nm: float = 0.001
    # Conservative derived angular acceleration estimate
    # (not explicitly specified in datasheet)
    max_slew_accel_deg_sec2: float = 1.0
    # Datasheet angular momentum storage
    wheel_max_momentum_nms: float = 0.01317
    # Use control accuracy rather than unrealistic fine pointing
    pointing_accuracy_deg: float = 0.4
    # More realistic settling time for agile CubeSat CMG system
    settling_time_sec: float = 8.0
    # Datasheet power modes
    adcs_idle_power_w: float = 0.8
    adcs_slew_power_w: float = 4.5

    # EPS and solar-array properties
    battery_capacity_wh: float = 25.0   
    initial_battery_wh: float = 15.0    
    solar_panel_efficiency: float = 0.193515  # BOL cell efficiency × packing factor × harness derating
                                            # 28.5% triple-junction × 0.67 packing = ~0.193515
    solar_panel_area_m2: float = 0.085     # 3U: 1 body-mounted + 2 deployable panels, total active area
    solar_constant_w_m2: float = 1361.0   
    eps_charge_efficiency: float = 0.88    # MPPT + regulator losses
    solar_temp_derating: float = 0.88      # temperature derating at operating temp (~60°C BOL)

    # EPS charge-state thresholds
    low_power_enter_wh: float = 10.0   
    low_power_exit_wh: float = 17.50
    critical_power_wh: float = 7.50     
    saa_detection_min_power_wh: float = 10.0
    min_relaybattery_capacity_wh: float = 10.0

    # Electrical loads
    obc_power_w: float = 0.8                
    eps_power_w: float = 1.2                
    payload_power_w: float = 10.0            
    radio_tx_power_w: float = 12.0

    # Storage and data system
    onboard_storage_mb: float = 32_000.0
    initial_storage_mb: float = 314.0
    storage_high_watermark_mb: float = 24_000.0

    # Communications geometry and link rate
    downlink_rate_mbps: float = 0.0096      
    gs_elevation_mask_deg: float = 15.0
    gs_lat_deg: float = 15.0
    gs_lon_deg: float = 74.0
    gs_alt_m: float = 70.0

    # Payload data generation
    # Science payload rate in megabits per second. Converted to MB per step as
    # rate_mbit_per_sec * dt_sec / 8.0.
    science_data_rate_mbps: float = 0.001152

    # Mode timing, geometric validity, and guidance hysteresis
    startup_cold_sec: float = 120.0
    startup_detumble_sec: float = 1000.0
    initial_body_rate_deg_sec_x: float = 0.5
    initial_body_rate_deg_sec_y: float = -0.1
    initial_body_rate_deg_sec_z: float = 0.02
    initial_body_accel_deg_sec2_x: float = 0.02
    initial_body_accel_deg_sec2_y: float = -0.015
    initial_body_accel_deg_sec2_z: float = 0.01
    detumble_target_attitude_label: str = "DEFAULT_ORIENTATION_HOLD"
    inertia_tensor_kgm2: Tuple[float, float, float] = (0.016, 0.016, 0.004)
    bdot_gain: float = 3.0e4
    bdot_handover_rate_deg_sec: float = 2.0
    wheel_rate_damp_gain: float = 0.01
    detumble_stable_rate_deg_sec: float = 0.5
    earth_b_field_model: str = "dipole"
    large_slew_threshold_deg: float = 60.0
    critical_recovery_target_wh: float = 6.5
    startup_sun_incidence_deg: float = 75.0
    attitude_command_deadband_deg: float = 0.5
    safe_hold_deadband_deg: float = 7.5
    ram_track_deadband_deg: float = 1.0
    relay_track_deadband_deg: float = 2.0
    payload_pointing_valid_deg: float = 15.0
    relay_pointing_valid_deg: float = 10.0
    directional_antenna_half_beamwidth_deg: float = 40.0
    charging_complete_threshold_wh: float = 24.5
    minimum_detection_duration_sec: float = 120.0
    minimum_relay_duration_sec: float = 120.0
    minimum_safe_dwell_sec: float = 120.0
    max_saturation_hold_sec: float = 300.0
    minimum_settled_time_before_payload_sec: float = 15.0
    minimum_settled_time_before_tx_sec: float = 10.0
    mode_transition_grace_steps: int = 5
    penumbra_transition_sec: float = 30.0
    slew_transition_rate_deg_sec: float = 2.0
    slew_transition_penalty_sec: float = 60.0
    slew_power_penalty_w: float = 5.0
    max_solar_power_w: float = 24.0
    eclipse_heater_power_w: float = 0.8
    adcs_standby_power_w: float = 0.8
    battery_taper_soc_start: float = 0.90
    battery_taper_min_fraction: float = 0.20
    wheel_momentum_recovery_fraction: float = 0.80
    magnetorquer_dipole_Am2: float = 0.2          # peak dipole per axis (A·m²), realistic 3U
    earth_B_field_T: float = 30.0e-6              # representative SSO B-field at 600 km (T)
    sun_pointing_tracking_noise_deg: float = 2.5  # 1-sigma Gaussian noise on sun incidence angle (deg)
                                                   # accounts for jitter, finite BW, settling error
    wheel_inertia_kgm2: float = 1.5e-5
    rng_seed: int = 42
    # Numerical propagator configuration
    min_integrator_step_sec: float = 0.1
    max_integrator_step_sec: float = 120.0
    integrator_position_tolerance_m: float = 10.0
    integrator_velocity_tolerance_mps: float = 0.01

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


@dataclass
class OrbitState:
    """Holds current propagated state: position, lat, lon, altitude,
    eclipse flag, beta angle."""

    absolute_date: AbsoluteDate
    time_sec: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    in_eclipse: bool
    beta_angle_deg: float
    sat_position_earth_frame: Vector3D
    sat_position_inertial: Vector3D
    sat_velocity_inertial: Vector3D


@dataclass
class AttitudeCommand:
    """Container for a mode-dependent target attitude and its control metadata.
    """

    quaternion: np.ndarray
    primary_objective: str
    secondary_objective: str
    active_attitude_target: str = ""
    attitude_hold_reason: str = ""
    relay_axis_label: str = ""
    sun_visible: bool = True


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

    def __init__(self, config: MissionConfig) -> None:
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
        inertia = np.maximum(self.inertia_tensor_kgm2, 1.0e-9)
        angular_momentum = inertia * omega_rad_sec
        gyro_term = np.cross(omega_rad_sec, angular_momentum)
        omega_dot_rad_sec2 = (torque_vec_nm - gyro_term) / inertia
        self.body_accel_deg_sec2 = np.degrees(omega_dot_rad_sec2)
        omega_rad_sec = omega_rad_sec + (omega_dot_rad_sec2 * dt_sec)
        self.body_rate_deg_sec = np.degrees(omega_rad_sec)

    def _integrate_quaternion_from_body_rates(self, dt_sec: float) -> None:
        """Advance the body-to-inertial quaternion from the current body rates."""

        omega_deg_sec = _as_flat_vector(self.body_rate_deg_sec)
        omega_mag_deg_sec = float(np.linalg.norm(omega_deg_sec))
        if omega_mag_deg_sec <= 0.0 or dt_sec <= 0.0:
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
        )
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
        """Return wheel torque opposing the current body-rate vector."""

        desired_torque = -self.config.wheel_rate_damp_gain * np.radians(self.body_rate_deg_sec)
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
    ) -> np.ndarray:
        """Flip quaternion sign when needed to avoid double-cover 180 deg artifacts."""

        if float(np.dot(previous_quaternion, new_quaternion)) < 0.0:
            self.quaternion_continuity_corrections += 1
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

        continuous_new_quaternion = self._enforce_quaternion_continuity(previous_quaternion, new_quaternion)
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
        target_quaternion = self._enforce_quaternion_continuity(self.current_quaternion, target_quaternion)
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

        if not self.is_slewing:
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


class EPSModel:
    """
    Power system in Wh units.

    Attributes:
      - battery_wh (float)
      - solar_generation_w (float)
      - load_w (float)

    Methods:
      - compute_solar_power(sun_angle_deg, in_sunlight) -> watts
      - compute_load(mode, adcs_slewing, in_eclipse) -> watts
      - step(dt_sec) -> (net_power_w, battery_wh)

    Solar model: P = η * A * G0 * cos(θ) * in_sunlight
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

    def __init__(self, config: MissionConfig) -> None:
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

    def compute_load(self, mode: str, adcs_slewing: bool, in_eclipse: bool) -> float:
        """Compute the current spacecraft electrical load for the active mode.

        `mode` is always the top-level FSM mode string (e.g. 'SUN_POINTING').
        Sun submodes (CHARGING / PASSIVE / ECLIPSE) are attitude-layer concepts
        and are never passed here; they do not change the electrical load budget.
        """

        baseline_w = self.config.obc_power_w + self.config.eps_power_w
        if in_eclipse:
            baseline_w += self.config.eclipse_heater_power_w
        adcs_power_w = self.config.adcs_slew_power_w if adcs_slewing else self.config.adcs_idle_power_w
        if mode in {"COLD_AND_DARK", "DETUMBLE"}:
            self.load_w = baseline_w
        elif mode == "SAFE":
            self.load_w = baseline_w + adcs_power_w
        elif mode == "SLEWING":
            self.load_w = baseline_w + adcs_power_w + self.config.slew_power_penalty_w
        elif mode == "SUN_POINTING":
            self.load_w = baseline_w + adcs_power_w
        elif mode == "DETECTION":
            self.load_w = baseline_w + adcs_power_w + self.config.payload_power_w
        elif mode == "RELAY":
            self.load_w = baseline_w + adcs_power_w + self.config.radio_tx_power_w
        else:
            raise ValueError(f"Unknown EPS load mode: {mode}")
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


class SAAModel:
    """
    South Atlantic Anomaly geographic boundary.

    Use the following polygon vertices (lon, lat in degrees) as the
    SAA boundary — this approximates the 5 MeV+ proton flux contour:

      [(-90, -5), (-80, -20), (-60, -35), (-40, -35),
       (-20, -25), (-10, -15), (-20, -5), (-50, 0),
       (-70, 0), (-90, -5)]

    Methods:
      - is_inside(lat_deg, lon_deg) -> bool  (use shapely Point/Polygon)
      - detect_entry_exit(prev_inside, curr_inside) -> event or None

    Log each SAA pass: entry_time, exit_time, dwell_seconds.
    """

    def __init__(self) -> None:
        """Construct the fixed SAA polygon and clear event state."""

        self.saa_polygon = Polygon(
            [
                (-90.0, -5.0),
                (-80.0, -20.0),
                (-60.0, -35.0),
                (-40.0, -35.0),
                (-20.0, -25.0),
                (-10.0, -15.0),
                (-20.0, -5.0),
                (-50.0, 0.0),
                (-70.0, 0.0),
                (-90.0, -5.0),
            ]
        )
        self.current_pass: Optional[Dict[str, float]] = None
        self.completed_passes: List[Dict[str, float]] = []

    def is_inside(self, lat_deg: float, lon_deg: float) -> bool:
        """Return whether the current sub-satellite point is inside the SAA polygon."""

        return bool(self.saa_polygon.contains(Point(lon_deg, lat_deg)))

    def detect_entry_exit(
        self,
        prev_inside: bool,
        curr_inside: bool,
        time_sec: float,
        lat_deg: float,
        lon_deg: float,
        absolute_date: AbsoluteDate,
        science_data_step_mb: float,
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Detect SAA entry or exit and maintain per-pass statistics."""

        if not prev_inside and curr_inside:
            self.current_pass = {
                "entry_time": absolute_date.toString(),
                "entry_time_sec": time_sec,
                "entry_lat_deg": lat_deg,
                "entry_lon_deg": lon_deg,
                "data_collected_mb": 0.0,
            }
            return "ENTRY", dict(self.current_pass)

        if self.current_pass is not None and curr_inside:
            self.current_pass["data_collected_mb"] += science_data_step_mb

        if prev_inside and not curr_inside and self.current_pass is not None:
            completed_pass = dict(self.current_pass)
            completed_pass["exit_time"] = absolute_date.toString()
            completed_pass["exit_time_sec"] = time_sec
            completed_pass["exit_lat_deg"] = lat_deg
            completed_pass["exit_lon_deg"] = lon_deg
            completed_pass["dwell_sec"] = time_sec - completed_pass["entry_time_sec"]
            self.completed_passes.append(completed_pass)
            self.current_pass = None
            return "EXIT", completed_pass

        return None

    def close_open_pass(self, final_time_sec: float, absolute_date: AbsoluteDate) -> None:
        """Close an open SAA pass at the end of the simulation horizon."""

        if self.current_pass is None:
            return

        completed_pass = dict(self.current_pass)
        completed_pass["exit_time"] = absolute_date.toString()
        completed_pass["exit_time_sec"] = final_time_sec
        completed_pass["exit_lat_deg"] = np.nan
        completed_pass["exit_lon_deg"] = np.nan
        completed_pass["dwell_sec"] = final_time_sec - completed_pass["entry_time_sec"]
        self.completed_passes.append(completed_pass)
        self.current_pass = None


class GroundStationModel:
    """
    Single ground station: GS_INDIA.
    lat=15.0 deg, lon=74.0 deg, alt=70 m.
    Elevation mask: 10 deg.

    Methods:
      - get_elevation(sat_pos_earth_frame, date) -> deg
      - is_visible(elevation_deg) -> bool
      - detect_aos_los(prev_visible, curr_visible) -> event or None

    Use Orekit TopocentricFrame.getElevation().
    Log each contact: AOS_time, LOS_time, max_elevation, downlinked_MB.
    """

    def __init__(self, config: MissionConfig, earth: OneAxisEllipsoid, earth_frame) -> None:
        """Build the GS topocentric frame and initialise contact history."""

        self.config = config
        station_point = GeodeticPoint(
            math.radians(config.gs_lat_deg),
            math.radians(config.gs_lon_deg),
            config.gs_alt_m,
        )
        self.station_point = station_point
        self.frame = TopocentricFrame(earth, station_point, "GS_INDIA")
        self.earth_frame = earth_frame
        self.current_contact: Optional[Dict[str, float]] = None
        self.completed_contacts: List[Dict[str, float]] = []

    def get_elevation(self, sat_pos_earth_frame: Vector3D, date: AbsoluteDate) -> float:
        """Compute instantaneous elevation above the local horizon."""

        return math.degrees(self.frame.getElevation(sat_pos_earth_frame, self.earth_frame, date))

    def is_visible(self, elevation_deg: float) -> bool:
        """Return whether the spacecraft is above the ground-station elevation mask."""

        return elevation_deg >= self.config.gs_elevation_mask_deg

    def detect_aos_los(
        self,
        prev_visible: bool,
        curr_visible: bool,
        time_sec: float,
        elevation_deg: float,
        absolute_date: AbsoluteDate,
        downlinked_step_mb: float,
    ) -> Optional[Tuple[str, Dict[str, float]]]:
        """Detect AOS or LOS transitions while maintaining contact statistics."""

        if not prev_visible and curr_visible:
            self.current_contact = {
                "aos_time": absolute_date.toString(),
                "aos_time_sec": time_sec,
                "max_el_deg": elevation_deg,
                "downlinked_mb": 0.0,
            }
            return "AOS", dict(self.current_contact)

        if self.current_contact is not None and curr_visible:
            self.current_contact["max_el_deg"] = max(self.current_contact["max_el_deg"], elevation_deg)
            self.current_contact["downlinked_mb"] += downlinked_step_mb

        if prev_visible and not curr_visible and self.current_contact is not None:
            completed_contact = dict(self.current_contact)
            completed_contact["los_time"] = absolute_date.toString()
            completed_contact["los_time_sec"] = time_sec
            completed_contact["duration_sec"] = time_sec - completed_contact["aos_time_sec"]
            self.completed_contacts.append(completed_contact)
            self.current_contact = None
            return "LOS", completed_contact

        return None

    def close_open_contact(self, final_time_sec: float, absolute_date: AbsoluteDate) -> None:
        """Close an open contact at the end of the simulation horizon."""

        if self.current_contact is None:
            return

        completed_contact = dict(self.current_contact)
        completed_contact["los_time"] = absolute_date.toString()
        completed_contact["los_time_sec"] = final_time_sec
        completed_contact["duration_sec"] = final_time_sec - completed_contact["aos_time_sec"]
        self.completed_contacts.append(completed_contact)
        self.current_contact = None


class PayloadModel:
    """
    Proton flux detection payload.
    Active ONLY in DETECTION mode while inside SAA.

    Methods:
      - generate_data(dt_sec) -> MB_generated
        (rate = SCIENCE_DATA_RATE_MBPS * dt_sec / 8.0)
      - is_operable(battery_wh, mode) -> bool

    Accumulate total science_data_generated_MB and
    total_detection_seconds across the simulation.
    """

    def __init__(self, config: MissionConfig) -> None:
        """Initialise payload counters."""

        self.config = config
        self.total_science_data_generated_mb = 0.0
        self.total_detection_seconds = 0.0

    def generate_data(self, dt_sec: float) -> float:
        """Generate science data during active SAA detection windows.

        Unit chain: science_data_rate_mbps [Mbit/s] × dt_sec [s] / 8 [bit/byte] = MB generated.
        At 0.001152 Mbps (1.152 kbps) this is 0.000288 MB per 2-second step, consistent with
        a low-rate proton spectrometer producing sparse housekeeping packets.
        """
        data_mb = self.config.science_data_rate_mbps * dt_sec / 8.0
        self.total_science_data_generated_mb += data_mb
        self.total_detection_seconds += dt_sec
        return data_mb

    def is_operable(self, battery_wh: float, mode: str) -> bool:
        """Return whether the payload is allowed to operate."""

        return mode == "DETECTION" and battery_wh >= self.config.saa_detection_min_power_wh


class CommunicationsModel:
    """
    Downlink over GS_INDIA.
    Active ONLY in RELAY mode while GS is visible.

    Methods:
      - downlink_step(storage_mb, dt_sec) -> MB_downlinked
        (rate = DOWNLINK_RATE_MBPS * dt_sec / 8.0)

    Track total_downlinked_MB and total_contact_seconds.
    Contact time is accumulated from raw visibility, while downlinked volume
    still depends on relay pointing validity.
    """

    def __init__(self, config: MissionConfig) -> None:
        """Initialise communications counters."""

        self.config = config
        self.total_downlinked_mb = 0.0
        self.total_contact_seconds = 0.0

    def downlink_step(self, storage_mb: float, dt_sec: float) -> float:
        """Downlink stored data while respecting the available onboard storage."""

        requested_mb = self.config.downlink_rate_mbps * dt_sec / 8.0
        downlinked_mb = min(storage_mb, requested_mb)
        self.total_downlinked_mb += downlinked_mb
        return downlinked_mb

    def record_contact_time(self, dt_sec: float) -> None:
        """Accumulate raw ground-station visibility time."""

        self.total_contact_seconds += dt_sec


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

    Any attempted illegal transition must raise ValueError.

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

    Hysteresis is handled by the dwell rules and the mode-selection logic.
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

    def __init__(self, config: MissionConfig) -> None:
        """Initialise the FSM with the ANTARA startup state."""

        self.config = config
        self.current_mode = "COLD_AND_DARK"
        self.time_sec = 0.0
        self.last_mode_change_time_sec = 0.0
        self.mode_entry_time_sec = 0.0
        self.mode_durations_sec = {mode: 0.0 for mode in self.ALLOWED_TRANSITIONS}
        self.transition_history: List[Dict[str, object]] = []
        self.target_mode = self.current_mode
        self.slew_start_time_sec = 0.0
        self.slew_end_time_sec = 0.0
        self.detumble_complete = False

    def set_time(self, time_sec: float) -> None:
        """Store the current simulation time for transition bookkeeping."""

        self.time_sec = time_sec

    def transition(self, new_mode: str, reason: str) -> bool:
        """Execute a mode transition if it is legal and not blocked by hysteresis."""

        if new_mode == self.current_mode:
            return False

        if new_mode not in self.ALLOWED_TRANSITIONS[self.current_mode]:
            raise ValueError(f"Illegal transition {self.current_mode} -> {new_mode}")

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
        if self.current_mode == "SAFE":
            return self.config.minimum_safe_dwell_sec
        return 0.0

    def minimum_dwell_complete(self) -> bool:
        """Return whether the current mode has satisfied its minimum dwell duration."""

        required_dwell_sec = self._minimum_dwell_for_current_mode_sec()
        if required_dwell_sec <= 0.0:
            return True
        return (self.time_sec - self.mode_entry_time_sec) >= required_dwell_sec

    def evaluate(
        self,
        battery_wh: float,
        gs_visible: bool,
        inside_saa: bool,
        adcs_saturation_hold_active: bool,
        saturation_hold_duration_sec: float,
        pointing_error_deg: float,
        detumble_complete: bool,
    ) -> Tuple[str, str]:
        """Evaluate the requested operational mode.

        SAFE is NOT operational nominal mode.
        Eclipse is an attitude condition within SUN_POINTING, not a spacecraft mode.
        """

        if self.current_mode == "SLEWING":
            if self.time_sec >= self.slew_end_time_sec:
                return self.target_mode, f"Slew complete -> {self.target_mode}"
            if battery_wh < self.config.critical_power_wh:
                return "SAFE", "Critical battery survival"
            return "SLEWING", f"Slewing toward {self.target_mode}"

        # --- Startup sequence (fixed timer, no overrides) ---
        if self.current_mode == "COLD_AND_DARK":
            if self.time_sec >= self.config.startup_cold_sec:
                return "DETUMBLE", "Startup cold phase complete"
            return self.current_mode, "Remain in cold startup"

        if self.current_mode == "DETUMBLE":
            if detumble_complete:
                return "SUN_POINTING", "Physics detumble complete -> SUN_POINTING"
            return self.current_mode, "Remain in detumble startup"

        # --- Fault / survival overrides (highest priority) ---
        if battery_wh < self.config.critical_power_wh:
            return "SAFE", "Critical battery survival"

        if adcs_saturation_hold_active:
            if (
                saturation_hold_duration_sec > self.config.max_saturation_hold_sec
                and pointing_error_deg > self.config.safe_hold_deadband_deg
            ):
                return "SAFE", "Saturation hold timeout with pointing divergence"
            return self.current_mode, "Hold current mode during ADCS saturation unloading"

        # --- Minimum dwell enforcement (before any upward transition) ---
        # DETECTION and RELAY have hard minimum dwell; must not be cut short.
        if self.current_mode == "DETECTION" and not self.minimum_dwell_complete():
            return "DETECTION", "Minimum detection dwell not complete"

        if self.current_mode == "RELAY" and not self.minimum_dwell_complete():
            return "RELAY", "Minimum relay dwell not complete"

        if self.current_mode == "SAFE" and not self.minimum_dwell_complete():
            return "SAFE", "Minimum SAFE dwell not complete"

        # --- Operational mode selection (priority ordered) ---
        if gs_visible and battery_wh > self.config.min_relaybattery_capacity_wh:
            return "RELAY", "Ground station visible with adequate battery"

        if inside_saa and battery_wh > self.config.saa_detection_min_power_wh:
            return "DETECTION", "Inside SAA with sufficient power"

        if battery_wh < self.config.low_power_enter_wh:
            return "SUN_POINTING", "Low-power charging"

        # --- Nominal default: SUN_POINTING (fixed orbit-normal inertial hold) ---
        # SAFE is NOT selected here. Eclipse is handled at the attitude layer.
        return "SUN_POINTING", "Nominal SUN_POINTING baseline"


class TelemetryLogger:
    """
    pandas DataFrame-based telemetry recorder.

    Columns per timestep:
      time_sec, lat_deg, lon_deg, alt_km,
      mode, battery_wh, solar_w, load_w,
      in_sunlight, in_saa, gs_visible,
      slew_angle_deg, wheel_mom_x, wheel_mom_y, wheel_mom_z,
      pointing_error_deg, storage_mb,
      science_data_mb, downlinked_mb,
      sun_visible,
      sun_incidence_angle_deg, relay_pointing_error_deg,
      gs_off_boresight_deg,
      payload_pointing_error_deg, attitude_error_deg,
      cumulative_science_generated_mb, cumulative_downlinked_mb,
      net_storage_delta_mb,
      saturation_hold_active,
      active_attitude_target, attitude_hold_reason,
      sun_submode,
      science_generation_enabled, relay_link_active,
      payload_alignment_cosine, relay_alignment_cosine,
      payload_boresight_x, payload_boresight_y, payload_boresight_z,
      velocity_unit_x, velocity_unit_y, velocity_unit_z,
      relay_boresight_x, relay_boresight_y, relay_boresight_z,
      gs_los_x, gs_los_y, gs_los_z,
      illumination_fraction,
      wheel_speed_rpm_x, wheel_speed_rpm_y, wheel_speed_rpm_z,
      wheel_unload_torque_nm, momentum_dump_active

    Additional event logs (separate DataFrames):
      - saa_passes_df: entry_time, exit_time, dwell_sec, data_collected_mb
      - gs_contacts_df: aos_time, los_time, max_el_deg, downlinked_mb
      - mode_transitions_df: time, from_mode, to_mode, reason
      - slew_events_df: start_time_sec, end_time_sec, from_mode, target_mode, angle_deg, duration_sec
      - detumble_events_df: start_time_sec, end_time_sec, initial_body_rates_deg_sec,
        initial_body_accels_deg_sec2, target_attitude_label, goal
    """

    def __init__(self) -> None:
        """Initialise empty row buffers for timestep and event telemetry."""

        self.telemetry_rows: List[Dict[str, object]] = []
        self.slew_event_rows: List[Dict[str, object]] = []
        self.detumble_event_rows: List[Dict[str, object]] = []

    def log_step(
        self,
        time_sec: float,
        orbit_state: OrbitState,
        mode: str,
        battery_wh: float,
        solar_w: float,
        load_w: float,
        in_sunlight: bool,
        in_saa: bool,
        gs_visible: bool,
        adcs_model: ADCSModel,
        storage_mb: float,
        science_data_mb: float,
        downlinked_mb: float,
        sun_visible: bool,
        sun_incidence_angle_deg: float,
        relay_pointing_error_deg: float,
        gs_off_boresight_deg: float,
        payload_pointing_error_deg: float,
        attitude_error_deg: float,
        cumulative_science_generated_mb: float,
        cumulative_downlinked_mb: float,
        net_storage_delta_mb: float,
        saturation_hold_active: bool,
        active_attitude_target: str,
        attitude_hold_reason: str,
        sun_submode: str,
        science_generation_enabled: bool,
        relay_link_active: bool,
        payload_alignment_cosine: float,
        relay_alignment_cosine: float,
        payload_boresight_inertial: np.ndarray,
        velocity_unit_vector: np.ndarray,
        relay_boresight_vector: np.ndarray,
        gs_los_vector: np.ndarray,
        illumination_fraction: float,
    ) -> None:
        """Append one timestep of spacecraft telemetry to the in-memory log."""

        payload_boresight_inertial = _as_flat_vector(payload_boresight_inertial)
        velocity_unit_vector = _as_flat_vector(velocity_unit_vector)
        relay_boresight_vector = _as_flat_vector(relay_boresight_vector)
        gs_los_vector = _as_flat_vector(gs_los_vector)

        self.telemetry_rows.append(
            {
                "time_sec": time_sec,
                "lat_deg": orbit_state.latitude_deg,
                "lon_deg": orbit_state.longitude_deg,
                "alt_km": orbit_state.altitude_km,
                "mode": mode,
                "battery_wh": battery_wh,
                "solar_w": solar_w,
                "load_w": load_w,
                "in_sunlight": in_sunlight,
                "in_saa": in_saa,
                "gs_visible": gs_visible,
                "slew_angle_deg": adcs_model.slew_angle_deg if adcs_model.is_slewing or adcs_model.settling else 0.0,
                "wheel_mom_x": adcs_model.wheel_momentum_nms[0],
                "wheel_mom_y": adcs_model.wheel_momentum_nms[1],
                "wheel_mom_z": adcs_model.wheel_momentum_nms[2],
                "pointing_error_deg": adcs_model.get_pointing_error_deg(),
                "storage_mb": storage_mb,
                "science_data_mb": science_data_mb,
                "downlinked_mb": downlinked_mb,
                "sun_visible": sun_visible,
                "sun_incidence_angle_deg": sun_incidence_angle_deg,
                "relay_pointing_error_deg": relay_pointing_error_deg,
                "gs_off_boresight_deg": gs_off_boresight_deg,
                "payload_pointing_error_deg": payload_pointing_error_deg,
                "attitude_error_deg": attitude_error_deg,
                "cumulative_science_generated_mb": cumulative_science_generated_mb,
                "cumulative_downlinked_mb": cumulative_downlinked_mb,
                "net_storage_delta_mb": net_storage_delta_mb,
                "saturation_hold_active": saturation_hold_active,
                "active_attitude_target": active_attitude_target,
                "attitude_hold_reason": attitude_hold_reason,
                "sun_submode": sun_submode,
                "science_generation_enabled": science_generation_enabled,
                "relay_link_active": relay_link_active,
                "payload_alignment_cosine": payload_alignment_cosine,
                "relay_alignment_cosine": relay_alignment_cosine,
                "payload_boresight_x": payload_boresight_inertial[0],
                "payload_boresight_y": payload_boresight_inertial[1],
                "payload_boresight_z": payload_boresight_inertial[2],
                "velocity_unit_x": velocity_unit_vector[0],
                "velocity_unit_y": velocity_unit_vector[1],
                "velocity_unit_z": velocity_unit_vector[2],
                "relay_boresight_x": relay_boresight_vector[0],
                "relay_boresight_y": relay_boresight_vector[1],
                "relay_boresight_z": relay_boresight_vector[2],
                "gs_los_x": gs_los_vector[0],
                "gs_los_y": gs_los_vector[1],
                "gs_los_z": gs_los_vector[2],
                "illumination_fraction": illumination_fraction,
                "wheel_speed_rpm_x": adcs_model.wheel_speed_rpm_estimate[0],
                "wheel_speed_rpm_y": adcs_model.wheel_speed_rpm_estimate[1],
                "wheel_speed_rpm_z": adcs_model.wheel_speed_rpm_estimate[2],
                "wheel_unload_torque_nm": adcs_model.current_wheel_unload_torque_nm,
                "momentum_dump_active": adcs_model.momentum_dump_active,
                "beta_angle_deg": orbit_state.beta_angle_deg,
                "in_eclipse": orbit_state.in_eclipse,
            }
        )

    def log_slew_event(
        self,
        start_time_sec: float,
        from_mode: str,
        target_mode: str,
        angle_deg: float,
        estimated_duration_sec: float,
        end_time_sec: Optional[float] = None,
    ) -> None:
        """Append a slew event entry for mode-transition analysis.

        The row is created when SLEWING starts and later finalized with the
        actual end time / duration when the slew completes.
        """

        self.slew_event_rows.append(
            {
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "from_mode": from_mode,
                "target_mode": target_mode,
                "angle_deg": angle_deg,
                "estimated_duration_sec": estimated_duration_sec,
                "duration_sec": estimated_duration_sec if end_time_sec is None else max(0.0, end_time_sec - start_time_sec),
            }
        )

    def finalize_slew_event(self, event_index: int, end_time_sec: float) -> None:
        """Finalize an open slew event with the actual completion time."""

        if event_index < 0 or event_index >= len(self.slew_event_rows):
            return
        event = self.slew_event_rows[event_index]
        start_time_sec = float(event.get("start_time_sec", 0.0))
        event["end_time_sec"] = end_time_sec
        event["duration_sec"] = max(0.0, end_time_sec - start_time_sec)

    def log_detumble_event(
        self,
        start_time_sec: float,
        from_mode: str,
        target_attitude_label: str,
        initial_body_rate_deg_sec: np.ndarray,
        initial_body_accel_deg_sec2: np.ndarray,
        goal: str,
        end_time_sec: Optional[float] = None,
    ) -> None:
        """Append a startup detumble event for separate startup-phase analysis."""

        initial_body_rate_deg_sec = _as_flat_vector(initial_body_rate_deg_sec)
        initial_body_accel_deg_sec2 = _as_flat_vector(initial_body_accel_deg_sec2)
        self.detumble_event_rows.append(
            {
                "time_sec": start_time_sec,
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "duration_sec": 0.0 if end_time_sec is None else max(0.0, end_time_sec - start_time_sec),
                "from_mode": from_mode,
                "to_mode": "DETUMBLE",
                "target_attitude_label": target_attitude_label,
                "goal": goal,
                "initial_body_rate_x_deg_sec": initial_body_rate_deg_sec[0],
                "initial_body_rate_y_deg_sec": initial_body_rate_deg_sec[1],
                "initial_body_rate_z_deg_sec": initial_body_rate_deg_sec[2],
                "initial_body_accel_x_deg_sec2": initial_body_accel_deg_sec2[0],
                "initial_body_accel_y_deg_sec2": initial_body_accel_deg_sec2[1],
                "initial_body_accel_z_deg_sec2": initial_body_accel_deg_sec2[2],
            }
        )

    def finalize_detumble_event(self, event_index: int, end_time_sec: float) -> None:
        """Finalize an open detumble event with the actual completion time."""

        if event_index < 0 or event_index >= len(self.detumble_event_rows):
            return
        event = self.detumble_event_rows[event_index]
        start_time_sec = float(event.get("start_time_sec", event.get("time_sec", 0.0)))
        event["end_time_sec"] = end_time_sec
        event["duration_sec"] = max(0.0, end_time_sec - start_time_sec)

    def build_dataframes(
        self,
        saa_pass_rows: List[Dict[str, object]],
        gs_contact_rows: List[Dict[str, object]],
        mode_transition_rows: List[Dict[str, object]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert raw telemetry buffers into pandas DataFrames."""

        telemetry_df = pd.DataFrame(self.telemetry_rows)
        saa_passes_df = pd.DataFrame(saa_pass_rows)
        gs_contacts_df = pd.DataFrame(gs_contact_rows)
        mode_transitions_df = pd.DataFrame(mode_transition_rows)
        slew_events_df = pd.DataFrame(self.slew_event_rows)
        detumble_events_df = pd.DataFrame(self.detumble_event_rows)
        return telemetry_df, saa_passes_df, gs_contacts_df, mode_transitions_df, slew_events_df, detumble_events_df


class ANTARASimulator:
    """
    Top-level simulation runner.
    Owns all subsystem instances. Runs the main loop.
    """

    def __init__(self, config: Optional[MissionConfig] = None) -> None:
        """Build the mission environment, Orekit propagator, and subsystem models."""

        self.config = config or MissionConfig()
        self.utc = None
        self.start_date = None
        self.inertial_frame = None
        self.earth_frame = None
        self.earth = None
        self.sun = None
        self.propagator = None
        self._init_orekit()
        self._init_frames_and_bodies()
        self._init_propagator()

        self.adcs = ADCSModel(self.config)
        self.eps = EPSModel(self.config)
        self.saa = SAAModel()
        self.ground_station = GroundStationModel(self.config, self.earth, self.earth_frame)
        self.payload = PayloadModel(self.config)
        self.communications = CommunicationsModel(self.config)
        self.fsm = SpacecraftFSM(self.config)
        self.logger = TelemetryLogger()

        self.initial_non_science_storage_mb = self.config.initial_storage_mb
        self.science_storage_mb = 0.0
        self.previous_in_saa = False
        self.previous_gs_visible = False
        self.current_attitude_command: Optional[AttitudeCommand] = None
        self._steps_since_mode_transition = 9999
        self.current_sun_incidence_angle_deg = 90.0
        self.current_relay_pointing_error_deg = 0.0
        self.current_payload_pointing_error_deg = 0.0
        self.current_attitude_error_deg = 0.0
        self.current_gs_off_boresight_deg = 180.0
        self.current_illumination_fraction = 1.0
        self.illumination_start_fraction = 1.0
        self.illumination_target_fraction = 1.0
        self.illumination_transition_start_time_sec = 0.0
        self.previous_sun_visible_geometry = True
        self.target_mode = self.fsm.current_mode
        self.slew_start_time_sec = 0.0
        self.slew_end_time_sec = 0.0
        self.active_slew_event_index: Optional[int] = None
        self.active_detumble_event_index: Optional[int] = None
        self.detection_no_science_duration_sec = 0.0
        self.warning_state: Dict[str, bool] = {}
        self.rng = np.random.default_rng(self.config.rng_seed)
        self.beta_angle_history_deg: List[float] = []
        self.orbital_period_history_min: List[float] = []
        self._run_geometric_self_tests()

    def _get_total_storage_mb(self) -> float:
        """Return total occupied onboard storage including non-science baseline usage."""

        return self.initial_non_science_storage_mb + self.science_storage_mb

    def _run_geometric_self_tests(self) -> None:
        """Abort startup if the quaternion/DCM convention is internally inconsistent."""

        saved_current_quaternion = self.adcs.current_quaternion.copy()
        saved_target_quaternion = self.adcs.target_quaternion.copy()
        saved_current_tuple = self.adcs.current_quaternion_tuple
        saved_continuity = self.adcs.quaternion_continuity_corrections
        saved_current_command = self.current_attitude_command

        try:
            identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            self.adcs.current_quaternion = identity.copy()
            self.adcs.target_quaternion = identity.copy()
            self.adcs._update_quaternion_tuple()
            self._assert_rotation_consistency("Geometric self-test identity", identity)

            yaw_90 = self.adcs.quaternion_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=float), 90.0)
            self._assert_axis_alignment(
                "Geometric self-test yaw",
                yaw_90,
                "+X",
                np.array([0.0, 1.0, 0.0], dtype=float),
                1.0e-6,
            )

            detection_primary = self._unit_numpy(np.array([0.42, 0.73, 0.53], dtype=float))
            detection_secondary = np.array([0.16, 0.91, -0.38], dtype=float)
            detection_quaternion = self.build_frame(
                "+X",
                detection_primary,
                "+Z",
                detection_secondary,
                context="Geometric self-test detection",
            )
            self._assert_axis_alignment(
                "Geometric self-test detection",
                detection_quaternion,
                "+X",
                detection_primary,
                1.0e-6,
            )
            detection_z_target = self._unit_numpy(self._project_orthogonal(detection_secondary, detection_primary))
            self._assert_axis_alignment(
                "Geometric self-test detection",
                detection_quaternion,
                "+Z",
                detection_z_target,
                1.0e-6,
            )

            relay_primary = self._unit_numpy(np.array([-0.31, 0.81, 0.49], dtype=float))
            relay_secondary = np.array([0.44, 0.21, 0.87], dtype=float)
            relay_quaternion = self.build_frame(
                "+Y",
                relay_primary,
                "+Z",
                relay_secondary,
                context="Geometric self-test relay",
            )
            self._assert_axis_alignment(
                "Geometric self-test relay",
                relay_quaternion,
                "+Y",
                relay_primary,
                1.0e-6,
            )
            relay_z_target = self._unit_numpy(self._project_orthogonal(relay_secondary, relay_primary))
            self._assert_axis_alignment(
                "Geometric self-test relay",
                relay_quaternion,
                "+Z",
                relay_z_target,
                1.0e-6,
            )

            roundtrip_quaternion = self.adcs.quaternion_from_axis_angle(
                self._unit_numpy(np.array([1.0, 2.0, 3.0], dtype=float)),
                37.0,
            )
            sample_vector = self._unit_numpy(np.array([0.31, -0.52, 0.79], dtype=float))
            body_vector = self.adcs.rotate_inertial_vector_to_body(roundtrip_quaternion, sample_vector)
            inertial_vector = self.adcs.rotate_body_axis_to_inertial(roundtrip_quaternion, body_vector)
            if float(np.linalg.norm(inertial_vector - sample_vector)) > 1.0e-6:
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        "Geometric self-test round-trip failed",
                        roundtrip_quaternion,
                        {
                            "sample_vector": sample_vector,
                            "roundtrip_vector": inertial_vector,
                        },
                    )
                )

            matrix = self.adcs.get_body_to_inertial_matrix(roundtrip_quaternion)
            reconstructed = self.adcs.quaternion_from_matrix(matrix)
            if min(
                float(np.linalg.norm(reconstructed - roundtrip_quaternion)),
                float(np.linalg.norm(reconstructed + roundtrip_quaternion)),
            ) > 1.0e-6:
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        "Geometric self-test quaternion/DCM mismatch",
                        roundtrip_quaternion,
                        {
                            "reconstructed_quaternion": reconstructed,
                        },
                    )
                )
        finally:
            self.adcs.current_quaternion = saved_current_quaternion
            self.adcs.target_quaternion = saved_target_quaternion
            self.adcs.current_quaternion_tuple = saved_current_tuple
            self.adcs.quaternion_continuity_corrections = saved_continuity
            self.current_attitude_command = saved_current_command

    def _init_orekit(self) -> None:
        """Initialise the Orekit JVM and data providers from the shared data directory."""

        orekit.initVM()
        init_orekit_data()

    def _coerce_double_array(self, array_like):
        """Convert tolerance arrays into the Java double[] type expected by Hipparchus."""

        if JArray_double is not None:
            try:
                return JArray_double.cast_(array_like)
            except Exception:
                pass

        # Some Orekit wrapper versions already return the correct Java array type.
        return array_like

    def _init_frames_and_bodies(self) -> None:
        """Initialise time scales, frames, Earth body shape, and the Sun ephemeris."""

        from org.orekit.frames import ITRFVersion

        self.utc = TimeScalesFactory.getUTC()
        self.start_date = AbsoluteDate(
            self.config.start_year,
            self.config.start_month,
            self.config.start_day,
            0,
            0,
            0.0,
            self.utc,
        )
        self.inertial_frame = FramesFactory.getEME2000()
        self.earth_frame = FramesFactory.getITRF(
            ITRFVersion.ITRF_2014,
            IERSConventions.IERS_2010,
            True,
        )
        self.earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            self.earth_frame,
        )
        self.sun = CelestialBodyFactory.getSun()

    def _build_initial_orbit(self) -> KeplerianOrbit:
        """Create the initial sun-synchronous orbit from the mission configuration."""

        semi_major_axis_m = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + self.config.altitude_m
        return KeplerianOrbit(
            semi_major_axis_m,
            self.config.eccentricity,
            math.radians(self.config.inclination_deg),
            math.radians(self.config.arg_perigee_deg),
            math.radians(self.config.raan_deg),
            math.radians(self.config.true_anomaly_deg),
            PositionAngleType.TRUE,
            self.inertial_frame,
            self.start_date,
            Constants.WGS84_EARTH_MU,
        )

    def _build_atmosphere(self):
        """Construct an Orekit atmosphere model, preferring NRLMSISE-00 and falling back to DTM2000."""

        atmosphere_errors: List[str] = []

        try:
            cssi_data = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt")
            return NRLMSISE00(cssi_data, self.sun, self.earth)
        except Exception as exc:
            atmosphere_errors.append(f"NRLMSISE00 primary attempt failed: {exc}")

        try:
            cssi_data = CssiSpaceWeatherData("SpaceWeather-All-v1.2.txt", self.utc)
            return NRLMSISE00(cssi_data, self.sun, self.earth)
        except Exception as exc:
            atmosphere_errors.append(f"NRLMSISE00 UTC attempt failed: {exc}")

        try:
            strength = MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
            marshall_data = MarshallSolarActivityFutureEstimation(
                "Jan2000F10.txt",
                strength,
            )
            return DTM2000(marshall_data, self.sun, self.earth)
        except Exception as exc:
            atmosphere_errors.append(f"DTM2000 attempt failed: {exc}")

        message = "Unable to construct atmospheric density model.\n" + "\n".join(atmosphere_errors)
        raise RuntimeError(message)

    def _build_srp_model(self):
        """Construct the SRP model with Earth occultation for eclipse-aware illumination."""

        radiation = IsotropicRadiationSingleCoefficient(
            self.config.cross_section_m2,
            self.config.srp_coeff,
        )

        try:
            return SolarRadiationPressure(self.sun, self.earth, radiation)
        except Exception:
            srp_model = SolarRadiationPressure(
                self.sun,
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                radiation,
            )
            try:
                srp_model.addOccultingBody(self.earth)
            except Exception:
                pass
            return srp_model

    def _init_propagator(self) -> None:
        """Build the Orekit numerical propagator and attach required force models."""

        initial_orbit = self._build_initial_orbit()
        tolerances = NumericalPropagator.tolerances(
            self.config.integrator_position_tolerance_m,
            initial_orbit,
            OrbitType.KEPLERIAN,
        )
        min_step = self.config.min_integrator_step_sec
        max_step = self.config.max_integrator_step_sec
        initial_step = self.config.step_sec

        absolute_tolerances = self._coerce_double_array(tolerances[0])
        relative_tolerances = self._coerce_double_array(tolerances[1])

        integrator = DormandPrince853Integrator(
            min_step,
            max_step,
            absolute_tolerances,
            relative_tolerances,
        )
        integrator.setInitialStepSize(initial_step)

        self.propagator = NumericalPropagator(integrator)
        self.propagator.setOrbitType(OrbitType.KEPLERIAN)
        self.propagator.setInitialState(self._make_spacecraft_state(initial_orbit))

        gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
        gravity = HolmesFeatherstoneAttractionModel(self.earth_frame, gravity_provider)
        self.propagator.addForceModel(gravity)

        atmosphere = self._build_atmosphere()
        drag_spacecraft = IsotropicDrag(self.config.cross_section_m2, self.config.drag_coeff)
        self.propagator.addForceModel(DragForce(atmosphere, drag_spacecraft))

        srp_model = self._build_srp_model()
        self.propagator.addForceModel(srp_model)

        # Eclipse classification uses _is_in_eclipse_geometry() (direct shadow-cone
        # geometry) rather than an Orekit event detector, because the event-detector
        # callback model does not fit the fixed-step telemetry loop.
        # self.eclipse_detector is intentionally not registered with the propagator.
        self.eclipse_detector = None

    def _make_spacecraft_state(self, orbit):
        """Create a spacecraft state for the propagator initialisation path."""

        from org.orekit.propagation import SpacecraftState

        return SpacecraftState(orbit, self.config.mass_kg)

    def _set_attitude_provider(self) -> None:
        """Set a stable Orekit attitude provider for propagation-only state generation.

        Always uses the SAFE baseline provider regardless of spacecraft mode,
        because the propagator attitude is used only for perturbation force geometry,
        not for telemetry. The true commanded attitude is tracked independently by
        ADCSModel.
        """

        self.propagator.setAttitudeProvider(NadirPointing(self.inertial_frame, self.earth))

    def _propagate_state(self, absolute_date: AbsoluteDate):
        """Propagate the orbit to the requested date."""

        self._set_attitude_provider()
        return self.propagator.propagate(absolute_date)

    def _vector_difference(self, first_vector: Vector3D, second_vector: Vector3D) -> np.ndarray:
        """Return first_vector - second_vector as a flat NumPy vector."""

        return _as_flat_vector(first_vector) - _as_flat_vector(second_vector)

    def _vector_unit(self, vector: Vector3D) -> np.ndarray:
        """Return a flat unit vector while avoiding wrapper-dependent normalize() calls."""

        vector = _as_flat_vector(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Cannot normalise a zero-length vector.")
        return vector / norm

    def _vector_cross(self, first_vector: Vector3D, second_vector: Vector3D) -> np.ndarray:
        """Return the vector cross product as a flat NumPy vector."""

        return _as_flat_vector(np.cross(_as_flat_vector(first_vector), _as_flat_vector(second_vector)))

    def _vector_to_numpy(self, vector: Vector3D) -> np.ndarray:
        """Convert an Orekit Vector3D into a NumPy vector."""

        return _as_flat_vector(vector)

    def _unit_numpy(self, vector: np.ndarray) -> np.ndarray:
        """Return a unit-length NumPy vector."""

        vector = _as_flat_vector(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Cannot normalise a zero-length NumPy vector.")
        return vector / norm

    def _project_orthogonal(self, vector: np.ndarray, reference_axis: np.ndarray) -> np.ndarray:
        """Project a vector onto the plane orthogonal to the supplied unit axis."""

        vector = _as_flat_vector(vector)
        reference_axis = _as_flat_vector(reference_axis)
        return _as_flat_vector(vector - (float(np.dot(vector, reference_axis)) * reference_axis))

    def _vector_angle_deg(self, first_vector: np.ndarray, second_vector: np.ndarray) -> float:
        """Return the angle between two NumPy vectors in degrees."""

        cosine_term = np.clip(
            float(np.dot(self._unit_numpy(first_vector), self._unit_numpy(second_vector))),
            -1.0,
            1.0,
        )
        return math.degrees(math.acos(cosine_term))

    def _maybe_warn(self, warning_key: str, message: str) -> None:
        """Print a warning only once per warning key to avoid flooding the console."""

        if self.warning_state.get(warning_key, False):
            return
        print(f"[Warning] {message}")
        self.warning_state[warning_key] = True

    def _body_axis_to_inertial(self, quaternion: np.ndarray, axis_label: str) -> np.ndarray:
        """Return the requested signed body axis expressed in inertial coordinates."""

        return _as_flat_vector(self.adcs.rotate_body_axis_to_inertial(quaternion, BODY_AXIS_VECTORS[axis_label]))

    def _relay_boresight_inertial(
        self,
        quaternion: np.ndarray,
        relay_axis_label: Optional[str],
    ) -> np.ndarray:
        """Return the active relay boresight in inertial coordinates."""

        axis_label = relay_axis_label or "+Y"
        boresight = self._body_axis_to_inertial(quaternion, axis_label)
        return boresight

    def _payload_boresight_inertial(self, quaternion: np.ndarray) -> np.ndarray:
        """Return the payload boresight in inertial coordinates."""

        return self._body_axis_to_inertial(quaternion, "+X")

    def _format_vector(self, label: str, vector: np.ndarray) -> str:
        """Format a NumPy vector for diagnostics."""

        vector_array = np.asarray(vector, dtype=float)
        if vector_array.size == 1:
            return f"{label}={float(vector_array.reshape(-1)[0]): .6f}"
        vector = _as_flat_vector(vector_array)
        return f"{label}=({vector[0]: .6f}, {vector[1]: .6f}, {vector[2]: .6f})"

    def _attitude_diagnostic_message(
        self,
        context: str,
        quaternion: np.ndarray,
        target_vectors: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """Build a detailed diagnostics string for frame and quaternion failures."""

        raw_quaternion = np.array(quaternion, dtype=float)
        rotation_matrix = self.adcs.get_body_to_inertial_matrix(raw_quaternion)
        quaternion = self.adcs._normalize_quaternion(raw_quaternion)
        determinant = float(np.linalg.det(rotation_matrix))
        orthogonality_error = float(np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.identity(3)))
        body_axes = {
            "+X": rotation_matrix[:, 0],
            "+Y": rotation_matrix[:, 1],
            "+Z": rotation_matrix[:, 2],
        }

        diagnostic_lines = [
            f"[Attitude Diagnostic] {context}",
            f"Quaternion (scalar-first body->inertial) = {raw_quaternion}",
            f"Quaternion norm = {np.linalg.norm(raw_quaternion):.9f}",
            f"DCM(body->inertial) =\n{rotation_matrix}",
            f"det(DCM) = {determinant:.9f}",
            f"orthogonality error = {orthogonality_error:.3e}",
            self._format_vector("+X_body_in_inertial", body_axes["+X"]),
            self._format_vector("+Y_body_in_inertial", body_axes["+Y"]),
            self._format_vector("+Z_body_in_inertial", body_axes["+Z"]),
        ]
        if target_vectors is not None:
            for label, vector in target_vectors.items():
                diagnostic_lines.append(self._format_vector(label, vector))
        return "\n".join(diagnostic_lines)

    def _assert_rotation_consistency(self, context: str, quaternion: np.ndarray) -> None:
        """Enforce a single quaternion/DCM convention everywhere in the simulator."""

        raw_quaternion = np.array(quaternion, dtype=float)
        quaternion_norm_error = abs(np.linalg.norm(raw_quaternion) - 1.0)
        quaternion = self.adcs._normalize_quaternion(raw_quaternion)
        rotation_matrix = self.adcs.get_body_to_inertial_matrix(quaternion)
        determinant = float(np.linalg.det(rotation_matrix))
        orthogonality_error = float(np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.identity(3)))
        if quaternion_norm_error > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: quaternion norm drift", raw_quaternion))
        if orthogonality_error > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: non-orthogonal DCM", quaternion))
        if abs(determinant - 1.0) > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: improper DCM determinant", quaternion))
        if abs(float(np.dot(rotation_matrix[:, 0], rotation_matrix[:, 1]))) > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: +X/+Y non-orthogonal", quaternion))
        if abs(float(np.dot(rotation_matrix[:, 0], rotation_matrix[:, 2]))) > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: +X/+Z non-orthogonal", quaternion))
        if abs(float(np.dot(rotation_matrix[:, 1], rotation_matrix[:, 2]))) > 1.0e-6:
            raise AssertionError(self._attitude_diagnostic_message(f"{context}: +Y/+Z non-orthogonal", quaternion))

    def _assert_axis_alignment(
        self,
        context: str,
        quaternion: np.ndarray,
        axis_label: str,
        target_vector: np.ndarray,
        max_error_deg: float,
    ) -> float:
        """Assert that a specific body axis aligns with the supplied inertial target."""

        axis_in_inertial = self._body_axis_to_inertial(quaternion, axis_label)
        error_deg = self._vector_angle_deg(axis_in_inertial, target_vector)
        if error_deg > max_error_deg:
            raise AssertionError(
                self._attitude_diagnostic_message(
                    f"{context}: {axis_label} alignment error {error_deg:.6f} deg exceeds {max_error_deg:.6f} deg",
                    quaternion,
                    {
                        f"{axis_label}_body_in_inertial": axis_in_inertial,
                        "target_vector": target_vector,
                    },
                )
            )
        return error_deg

    def build_frame(
        self,
        primary_axis_label: str,
        primary_target_vector: np.ndarray,
        secondary_axis_label: str,
        secondary_reference_vector: np.ndarray,
        context: str = "attitude target",
    ) -> np.ndarray:
        """Build one immutable body frame from a locked axis and a secondary reference.

        The returned quaternion maps body-frame vectors into inertial coordinates.
        The DCM is stored as columns [X_body, Y_body, Z_body].
        """

        allowed_axes = {"+X", "+Y", "+Z"}
        if primary_axis_label not in allowed_axes or secondary_axis_label not in allowed_axes:
            raise ValueError(f"Unsupported body-axis labels: {primary_axis_label}, {secondary_axis_label}")
        if primary_axis_label == secondary_axis_label:
            raise ValueError("Primary and secondary body axes must be distinct.")

        primary_unit = self._unit_numpy(_as_flat_vector(primary_target_vector))
        secondary_projected = self._project_orthogonal(_as_flat_vector(secondary_reference_vector), primary_unit)

        if float(np.linalg.norm(secondary_projected)) <= 1.0e-9:
            for candidate in np.identity(3):
                secondary_projected = self._project_orthogonal(candidate, primary_unit)
                if float(np.linalg.norm(secondary_projected)) > 1.0e-9:
                    break

        secondary_unit = self._unit_numpy(secondary_projected)
        axis_vectors: Dict[str, np.ndarray] = {
            primary_axis_label: primary_unit,
            secondary_axis_label: secondary_unit,
        }
        missing_axis_label = next(iter(sorted(allowed_axes - {primary_axis_label, secondary_axis_label})))
        if missing_axis_label == "+X":
            axis_vectors["+X"] = self._unit_numpy(np.cross(axis_vectors["+Y"], axis_vectors["+Z"]))
        elif missing_axis_label == "+Y":
            axis_vectors["+Y"] = self._unit_numpy(np.cross(axis_vectors["+Z"], axis_vectors["+X"]))
        else:
            axis_vectors["+Z"] = self._unit_numpy(np.cross(axis_vectors["+X"], axis_vectors["+Y"]))

        rotation_matrix = np.column_stack(
            [axis_vectors["+X"], axis_vectors["+Y"], axis_vectors["+Z"]]
        )
        orthogonality_error = float(np.linalg.norm(rotation_matrix.T @ rotation_matrix - np.identity(3)))
        determinant = float(np.linalg.det(rotation_matrix))
        x_from_hand_rule = self._unit_numpy(np.cross(axis_vectors["+Y"], axis_vectors["+Z"]))
        handedness_error = float(np.linalg.norm(x_from_hand_rule - axis_vectors["+X"]))
        if orthogonality_error > 1.0e-6:
            raise AssertionError(
                self._attitude_diagnostic_message(
                    f"{context}: non-orthogonal attitude DCM error={orthogonality_error:.3e}",
                    self.adcs.quaternion_from_matrix(rotation_matrix),
                    {
                        "+X_body_in_inertial": rotation_matrix[:, 0],
                        "+Y_body_in_inertial": rotation_matrix[:, 1],
                        "+Z_body_in_inertial": rotation_matrix[:, 2],
                    },
                )
            )
        if handedness_error > 1.0e-6:
            raise AssertionError(
                self._attitude_diagnostic_message(
                    f"{context}: handedness consistency error={handedness_error:.3e}",
                    self.adcs.quaternion_from_matrix(rotation_matrix),
                    {
                        "+X_body_in_inertial": axis_vectors["+X"],
                        "+Y_body_in_inertial": axis_vectors["+Y"],
                        "+Z_body_in_inertial": axis_vectors["+Z"],
                        "cross(Y,Z)": x_from_hand_rule,
                    },
                )
            )
        if abs(determinant - 1.0) > 1.0e-6:
            raise AssertionError(
                self._attitude_diagnostic_message(
                    f"{context}: improper attitude DCM determinant det={determinant:.6f}",
                    self.adcs.quaternion_from_matrix(rotation_matrix),
                    {
                        "+X_body_in_inertial": rotation_matrix[:, 0],
                        "+Y_body_in_inertial": rotation_matrix[:, 1],
                        "+Z_body_in_inertial": rotation_matrix[:, 2],
                    },
                )
            )
        quaternion = self.adcs.quaternion_from_matrix(rotation_matrix)
        quaternion = self.adcs._enforce_quaternion_continuity(self.adcs.current_quaternion, quaternion)
        self._assert_rotation_consistency(context, quaternion)
        return quaternion

    def _orbit_normal_unit(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the orbit normal direction from the instantaneous position and velocity vectors."""

        return self._unit_numpy(
            np.cross(
                self._vector_to_numpy(orbit_state.sat_position_inertial),
                self._vector_to_numpy(orbit_state.sat_velocity_inertial),
            )
        )

    def _sun_vector_unit(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the Sun direction from spacecraft to Sun in inertial coordinates."""

        sun_position = self.sun.getPVCoordinates(orbit_state.absolute_date, self.inertial_frame).getPosition()
        return self._unit_numpy(
            self._vector_to_numpy(self._vector_difference(sun_position, orbit_state.sat_position_inertial))
        )

    def _nadir_vector_unit(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the nadir direction from spacecraft toward Earth center in inertial coordinates."""

        return self._unit_numpy(-self._vector_to_numpy(orbit_state.sat_position_inertial))

    def _velocity_vector_unit(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the inertial velocity direction."""

        return self._unit_numpy(self._vector_to_numpy(orbit_state.sat_velocity_inertial))

    def _ground_station_vector_unit(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the inertial line-of-sight vector from spacecraft to GS_INDIA."""

        # Use the TopocentricFrame PV transformation directly because some Orekit
        # Python wrapper builds do not expose Transform.transformPosition().
        station_position_inertial = self.ground_station.frame.getPVCoordinates(
            orbit_state.absolute_date,
            self.inertial_frame,
        ).getPosition()
        return self._unit_numpy(
            self._vector_to_numpy(self._vector_difference(station_position_inertial, orbit_state.sat_position_inertial))
        )

    def _compute_earth_b_field_body(self, orbit_state: OrbitState) -> np.ndarray:
        """Return a simple dipole Earth magnetic field vector expressed in the body frame."""

        position_inertial = self._vector_to_numpy(orbit_state.sat_position_inertial)
        radius_m = float(np.linalg.norm(position_inertial))
        if radius_m <= 0.0:
            return np.zeros(3, dtype=float)

        dipole_axis_inertial = np.array([0.0, 0.0, 1.0], dtype=float)
        radial_unit = position_inertial / radius_m
        b_scale_t = self.config.earth_B_field_T * (Constants.WGS84_EARTH_EQUATORIAL_RADIUS / radius_m) ** 3
        b_inertial_t = b_scale_t * (
            (3.0 * float(np.dot(dipole_axis_inertial, radial_unit)) * radial_unit) - dipole_axis_inertial
        )
        return self.adcs.rotate_inertial_vector_to_body(self.adcs.current_quaternion, b_inertial_t)

    def _step_detumble_physics(self, orbit_state: OrbitState, dt_sec: float) -> None:
        """Advance the detumble body-rate dynamics and wheel loading model."""

        b_body_t = self._compute_earth_b_field_body(orbit_state)
        wheel_torque_nm = self.adcs.step_detumble(b_body_t, dt_sec)
        unload_scale = 1.0 if not orbit_state.in_eclipse else 0.5
        b_projection_scale = max(0.2, float(np.linalg.norm(b_body_t)) / max(self.config.earth_B_field_T, 1.0e-12))
        self.adcs.update_wheel_momentum(wheel_torque_nm, dt_sec, unload_scale, b_projection_scale)
        self.fsm.detumble_complete = self.adcs.detumble_complete

    def _compute_default_orientation_attitude(self, orbit_state: OrbitState) -> AttitudeCommand:
        """Build the default inertial hold using the authoritative body-frame semantics.

        The attitude is a fixed RAM/nadir/orbit-normal baseline, but the body
        axes keep their absolute physical meanings:
          +X = payload boresight / RAM face
          +Y = directional relay antenna
          +Z = fixed solar panel
        """

        if (
            self.current_attitude_command is not None
            and self.current_attitude_command.active_attitude_target == "DEFAULT_ORIENTATION_HOLD"
        ):
            return AttitudeCommand(
                quaternion=self.adcs.target_quaternion.copy(),
                primary_objective="Default orientation inertial hold",
                secondary_objective="RAM +X, NADIR +Y, ORBIT-NORMAL +Z inertial hold",
                active_attitude_target="DEFAULT_ORIENTATION_HOLD",
                attitude_hold_reason="Inertial hold of routine RAM/NADIR baseline",
            )

        velocity_unit = self._velocity_vector_unit(orbit_state)
        nadir_unit = self._nadir_vector_unit(orbit_state)
        quaternion = self.build_frame(
            "+X",
            velocity_unit,
            "+Y",
            nadir_unit,
            context="DEFAULT_ORIENTATION_HOLD target",
        )
        return AttitudeCommand(
            quaternion=quaternion,
            primary_objective="+X -> velocity (ram)",
            secondary_objective="+Y -> nadir | +Z -> orbit normal",
            active_attitude_target="DEFAULT_ORIENTATION_HOLD",
            attitude_hold_reason="Entry into RAM/NADIR/orbit-normal inertial baseline",
        )

    def _validate_attitude_metrics(self, mode: str, orbit_state: OrbitState, gs_visible: bool) -> None:
        """Assert that telemetry attitude metrics use the authoritative body-axis mappings."""

        if mode == "SLEWING":
            return

        grace_complete = self._steps_since_mode_transition >= self.config.mode_transition_grace_steps
        self._assert_rotation_consistency("Current ADCS attitude", self.adcs.current_quaternion)
        self._assert_rotation_consistency("Target ADCS attitude", self.adcs.target_quaternion)
        if mode == "DETECTION":
            payload_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            if abs(payload_error_deg - self.current_payload_pointing_error_deg) > 1.0e-6:
                if grace_complete:
                    raise AssertionError(
                        self._attitude_diagnostic_message(
                            "DETECTION telemetry payload error mismatch",
                            self.adcs.current_quaternion,
                            {
                                "payload_boresight_inertial": self._payload_boresight_inertial(self.adcs.current_quaternion),
                                "velocity_target": self._velocity_vector_unit(orbit_state),
                            },
                        )
                    )
            if (
                not self.adcs.is_slewing
                and not self.adcs.settling
                and payload_error_deg > 90.0
                and grace_complete
            ):
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"DETECTION settled payload error {payload_error_deg:.6f} deg exceeds 90 deg",
                        self.adcs.current_quaternion,
                        {
                            "payload_boresight_inertial": self._payload_boresight_inertial(self.adcs.current_quaternion),
                            "velocity_target": self._velocity_vector_unit(orbit_state),
                        },
                    )
                )
        if mode == "RELAY":
            relay_axis_label = self.adcs.commanded_relay_axis or "+Y"
            relay_boresight_inertial = self._relay_boresight_inertial(
                self.adcs.current_quaternion,
                relay_axis_label,
            )
            relay_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                relay_axis_label,
                gs_visible,
            )
            gs_off_boresight_deg = self._compute_gs_off_boresight_deg(
                orbit_state,
                gs_visible,
            )
            if abs(relay_error_deg - self.current_relay_pointing_error_deg) > 1.0e-6:
                if grace_complete:
                    raise AssertionError(
                        self._attitude_diagnostic_message(
                            "RELAY telemetry relay error mismatch",
                            self.adcs.current_quaternion,
                            {
                                "relay_boresight_inertial": relay_boresight_inertial,
                                "nadir_target": self._nadir_vector_unit(orbit_state),
                            },
                        )
                    )
            if (
                not self.adcs.is_slewing
                and not self.adcs.settling
                and self.adcs.settled_time_sec >= self.config.minimum_settled_time_before_tx_sec
                and relay_error_deg > self.config.relay_pointing_valid_deg
                and grace_complete
            ):
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"RELAY settled pointing error {relay_error_deg:.6f} deg exceeds {self.config.relay_pointing_valid_deg:.6f} deg",
                        self.adcs.current_quaternion,
                        {
                            "relay_boresight_inertial": relay_boresight_inertial,
                            "relay_target": self._ground_station_vector_unit(orbit_state) if gs_visible else self._nadir_vector_unit(orbit_state),
                            "gs_off_boresight": np.array([gs_off_boresight_deg], dtype=float),
                        },
                    )
                )
            if (
                gs_visible
                and gs_off_boresight_deg > self.config.directional_antenna_half_beamwidth_deg
                and grace_complete
            ):
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"RELAY GS off-boresight {gs_off_boresight_deg:.6f} deg exceeds beamwidth {self.config.directional_antenna_half_beamwidth_deg:.6f} deg",
                        self.adcs.current_quaternion,
                        {
                            "relay_boresight_inertial": relay_boresight_inertial,
                            "relay_target": self._ground_station_vector_unit(orbit_state) if gs_visible else self._nadir_vector_unit(orbit_state),
                            "gs_target": self._ground_station_vector_unit(orbit_state),
                        },
                    )
                )

    def _update_alignment_telemetry(self, orbit_state: OrbitState, gs_visible: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
        """Compute the current payload and relay alignment metrics for telemetry."""

        payload_boresight_inertial = self._payload_boresight_inertial(self.adcs.current_quaternion)
        velocity_unit_vector = self._velocity_vector_unit(orbit_state)
        payload_alignment_cosine = float(
            np.clip(
                np.dot(self._unit_numpy(payload_boresight_inertial), velocity_unit_vector),
                -1.0,
                1.0,
            )
        )

        gs_los_vector = self._ground_station_vector_unit(orbit_state) if gs_visible else self._nadir_vector_unit(orbit_state)
        relay_boresight_vector = self._relay_boresight_inertial(
            self.adcs.current_quaternion,
            self.adcs.commanded_relay_axis,
        )
        relay_alignment_cosine = float(
            np.clip(
                np.dot(self._unit_numpy(relay_boresight_vector), self._unit_numpy(gs_los_vector)),
                -1.0,
                1.0,
            )
        )
        gs_off_boresight_deg = self._compute_gs_off_boresight_deg(
            orbit_state,
            gs_visible,
        )
        return (
            payload_boresight_inertial,
            velocity_unit_vector,
            relay_boresight_vector,
            gs_los_vector,
            payload_alignment_cosine,
            relay_alignment_cosine,
            gs_off_boresight_deg,
        )

    def _run_physics_sanity_checks(
        self,
        mode: str,
        orbit_state: OrbitState,
        gs_visible: bool,
        inside_saa: bool,
        science_generated_mb: float,
        downlinked_mb: float,
        battery_before_step_wh: float,
    ) -> None:
        """Emit one-time warnings for physically inconsistent states."""

        if self.eps.solar_generation_w > self.config.max_solar_power_w + 1.0e-6:
            self._maybe_warn("solar_power_limit", "Solar generation exceeded configured physical limit.")
        if mode == "DETECTION" and self.current_payload_pointing_error_deg > 90.0:
            self._maybe_warn("payload_error_large", "Payload pointing error exceeded 90 deg during DETECTION.")
        if mode == "RELAY" and self.current_relay_pointing_error_deg > 90.0:
            self._maybe_warn("relay_error_large", "Relay pointing error exceeded 90 deg during RELAY.")
        if self.current_illumination_fraction <= 0.0 and self.eps.battery_wh > battery_before_step_wh + 1.0e-6:
            self._maybe_warn("battery_increase_eclipse", "Battery increased during eclipse.")
        if science_generated_mb > 0.0 and not inside_saa and not self.previous_in_saa:
            self._maybe_warn("science_outside_saa", "Science generation occurred outside the SAA.")
        if downlinked_mb > 0.0 and not gs_visible:
            self._maybe_warn("downlink_without_visibility", "Downlink occurred without GS visibility.")

    def _compute_sun_tracking_attitude(
        self,
        orbit_state: OrbitState,
        sun_visible: bool,
    ) -> AttitudeCommand:
        """Return the SUN_POINTING attitude command.

        SUN_POINTING is a power-state label, not an active Sun-tracking mode.
        The spacecraft continues holding the nominal RAM/NADIR/orbit-normal
        baseline attitude; only the telemetry label changes with sun visibility
        and battery state.
        """
        base_command = self._compute_default_orientation_attitude(orbit_state)
        return AttitudeCommand(
            quaternion=base_command.quaternion.copy(),
            primary_objective="SUN_POINTING power-state hold",
            secondary_objective=(
                "No Sun tracking; +X ram, +Y nadir, +Z orbit-normal inertial hold"
            ),
            active_attitude_target="DEFAULT_ORIENTATION_HOLD",
            attitude_hold_reason="Nominal inertial hold retained; SUN_POINTING is not an attitude-tracking mode",
            sun_visible=sun_visible,
        )

    def _compute_detection_attitude(self, orbit_state: OrbitState) -> AttitudeCommand:
        """Build DETECTION attitude with payload +X aligned to ram on mode entry only."""

        if (
            self.current_attitude_command is not None
            and self.current_attitude_command.active_attitude_target == "DETECTION_INERTIAL_HOLD"
        ):
            return AttitudeCommand(
                quaternion=self.adcs.target_quaternion.copy(),
                primary_objective="+X -> velocity",
                secondary_objective="Hold the entry attitude throughout DETECTION",
                active_attitude_target="DETECTION_INERTIAL_HOLD",
                attitude_hold_reason="Detection attitude is frozen after the entry slew",
            )

        velocity_unit = self._velocity_vector_unit(orbit_state)
        nadir_unit = self._nadir_vector_unit(orbit_state)
        quaternion = self.build_frame(
            "+X",
            velocity_unit,
            "+Y",
            nadir_unit,
            context="DETECTION target",
        )
        self._assert_axis_alignment("DETECTION target", quaternion, "+X", velocity_unit, 1.0e-6)
        return AttitudeCommand(
            quaternion=quaternion,
            primary_objective="+X -> velocity",
            secondary_objective="+Y -> nadir | +Z -> orbit normal at DETECTION entry",
            active_attitude_target="DETECTION_INERTIAL_HOLD",
            attitude_hold_reason="Entry slew establishes the DETECTION inertial hold",
        )

    def _compute_safe_attitude(self, orbit_state: OrbitState) -> AttitudeCommand:
        """Freeze the current inertial attitude in SAFE mode - do not command a slew.

        SAFE is a power/survival mode, not a pointing mode. The correct behaviour
        is to freeze wherever the bus is and minimise ADCS activity. Commanding a
        fresh nadir-lock quaternion on every SAFE entry is operationally wrong and
        drives wheel saturation through repeated large-angle slews.

        Exception: if no attitude has been commanded yet (startup), fall back to the
        default orientation.
        """

        if self.current_attitude_command is None:
            return self._compute_default_orientation_attitude(orbit_state)

        return AttitudeCommand(
            quaternion=self.adcs.target_quaternion.copy(),
            primary_objective="SAFE inertial hold",
            secondary_objective="Freeze current attitude - minimise ADCS activity",
            active_attitude_target="SAFE_INERTIAL_HOLD",
            attitude_hold_reason="SAFE holds current inertial attitude without slewing",
        )

    def _compute_relay_candidate(
        self,
        orbit_state: OrbitState,
        target_unit: np.ndarray,
    ) -> AttitudeCommand:
        """Build the single-antenna RELAY entry attitude using the +Y fixed relay face."""

        z_reference = self._orbit_normal_unit(orbit_state)
        if float(np.linalg.norm(self._project_orthogonal(z_reference, target_unit))) <= 1.0e-9:
            z_reference = self._body_axis_to_inertial(self.adcs.current_quaternion, "+Z")
        z_target = self._unit_numpy(self._project_orthogonal(z_reference, target_unit))
        quaternion = self.build_frame(
            "+Y",
            target_unit,
            "+Z",
            z_reference,
            context="RELAY target",
        )
        self._assert_axis_alignment("RELAY target", quaternion, "+Y", target_unit, 1.0e-6)
        self._assert_axis_alignment("RELAY target", quaternion, "+Z", z_target, 1.0e-6)
        return AttitudeCommand(
            quaternion=quaternion,
            primary_objective="+Y -> ground station LOS",
            secondary_objective="+Z -> orbit-normal projection about +Y",
            active_attitude_target="RELAY_INERTIAL_HOLD",
            attitude_hold_reason="Entry slew establishes the RELAY inertial hold",
            relay_axis_label="+Y",
        )

    def _compute_relay_attitude(self, orbit_state: OrbitState, gs_visible: bool) -> AttitudeCommand:
        """Build RELAY attitude with +Y toward GS on mode entry only."""

        target_unit = self._ground_station_vector_unit(orbit_state) if gs_visible else self._nadir_vector_unit(orbit_state)
        if (
            self.current_attitude_command is not None
            and self.current_attitude_command.active_attitude_target == "RELAY_INERTIAL_HOLD"
        ):
            return AttitudeCommand(
                quaternion=self.adcs.target_quaternion.copy(),
                primary_objective="+Y -> relay boresight hold",
                secondary_objective="Hold the entry attitude throughout RELAY",
                active_attitude_target="RELAY_INERTIAL_HOLD",
                attitude_hold_reason="Relay attitude is frozen after the entry slew",
                relay_axis_label="+Y",
            )

        return self._compute_relay_candidate(orbit_state, target_unit)

    def _get_attitude_command(
        self,
        mode: str,
        orbit_state: OrbitState,
        sun_visible: bool,
        battery_wh: float,
        gs_visible: bool,
    ) -> AttitudeCommand:
        """Return the mode-dependent body-axis attitude objective for the current timestep."""

        if mode in {"COLD_AND_DARK", "DETUMBLE"}:
            command = self._compute_default_orientation_attitude(orbit_state)
            command.primary_objective = "Startup default orientation hold"
            command.secondary_objective = "Passive RAM/NADIR hold during startup"
            return command
        if mode == "SAFE":
            return self._compute_safe_attitude(orbit_state)
        if mode == "SUN_POINTING":
            return self._compute_sun_tracking_attitude(orbit_state, sun_visible)
        if mode == "DETECTION":
            return self._compute_detection_attitude(orbit_state)
        if mode == "RELAY":
            return self._compute_relay_attitude(orbit_state, gs_visible)
        raise ValueError(f"Unsupported attitude mode: {mode}")

    def _compute_beta_angle_deg(self, sat_position: Vector3D, sat_velocity: Vector3D, absolute_date: AbsoluteDate) -> float:
        """Compute orbital beta angle from the orbit normal and Sun direction."""

        sun_position = self.sun.getPVCoordinates(absolute_date, self.inertial_frame).getPosition()
        sun_unit_vec = self._vector_unit(self._vector_difference(sun_position, sat_position))
        orbit_normal = self._vector_unit(self._vector_cross(sat_position, sat_velocity))
        beta_rad = math.asin(np.clip(float(np.dot(orbit_normal, sun_unit_vec)), -1.0, 1.0))
        return math.degrees(beta_rad)

    def _is_in_eclipse_geometry(self, sat_position: Vector3D, absolute_date: AbsoluteDate) -> bool:
        """Classify eclipse using direct Sun-Earth-spacecraft geometry as a wrapper-safe fallback."""

        sun_position = self.sun.getPVCoordinates(absolute_date, self.inertial_frame).getPosition()
        sat_position = _as_flat_vector(sat_position)
        sun_position = _as_flat_vector(sun_position)
        sat_to_sun = _as_flat_vector(sun_position - sat_position)
        # Shadow axis is the anti-sun unit vector from Earth center in inertial coordinates.
        shadow_axis = self._unit_numpy(-sun_position)

        angle_to_shadow_rad = math.acos(
            np.clip(float(np.dot(self._unit_numpy(sat_position), self._unit_numpy(shadow_axis))), -1.0, 1.0)
        )
        earth_ratio = np.clip(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS / float(np.linalg.norm(sat_position)),
            -1.0,
            1.0,
        )
        earth_apparent_radius_rad = math.asin(earth_ratio)
        sun_ratio = np.clip(Constants.SUN_RADIUS / float(np.linalg.norm(sat_to_sun)), -1.0, 1.0)
        sun_apparent_radius_rad = math.asin(sun_ratio)

        return angle_to_shadow_rad <= (earth_apparent_radius_rad - sun_apparent_radius_rad)

    def _update_illumination_fraction(self, sun_visible: bool, time_sec: float) -> float:
        """Smooth the sunlight state across eclipse edges with a simple penumbra ramp."""

        target_fraction = 1.0 if sun_visible else 0.0
        if sun_visible != self.previous_sun_visible_geometry:
            self.illumination_start_fraction = self.current_illumination_fraction
            self.illumination_target_fraction = target_fraction
            self.illumination_transition_start_time_sec = time_sec
            self.previous_sun_visible_geometry = sun_visible

        elapsed_sec = max(0.0, time_sec - self.illumination_transition_start_time_sec)
        transition_fraction = min(1.0, elapsed_sec / max(self.config.penumbra_transition_sec, self.config.step_sec))
        self.current_illumination_fraction = (
            self.illumination_start_fraction
            + (self.illumination_target_fraction - self.illumination_start_fraction) * transition_fraction
        )
        return float(np.clip(self.current_illumination_fraction, 0.0, 1.0))

    def _build_orbit_state(self, propagated_state, time_sec: float) -> OrbitState:
        """Convert the Orekit propagated state into the mission OrbitState container."""

        absolute_date = propagated_state.getDate()
        sat_pos_earth = propagated_state.getPVCoordinates(self.earth_frame).getPosition()
        sat_geo = self.earth.transform(sat_pos_earth, self.earth_frame, absolute_date)

        latitude_deg = math.degrees(sat_geo.getLatitude())
        longitude_deg = math.degrees(sat_geo.getLongitude())
        longitude_deg = ((longitude_deg + 180.0) % 360.0) - 180.0

        sat_pv_inertial = propagated_state.getPVCoordinates(self.inertial_frame)
        sat_pos_inertial = sat_pv_inertial.getPosition()
        sat_vel_inertial = sat_pv_inertial.getVelocity()

        in_eclipse = self._is_in_eclipse_geometry(sat_pos_inertial, absolute_date)
        beta_angle_deg = self._compute_beta_angle_deg(
            sat_pos_inertial,
            sat_vel_inertial,
            absolute_date,
        )

        return OrbitState(
            absolute_date=absolute_date,
            time_sec=time_sec,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_km=sat_geo.getAltitude() / 1000.0,
            in_eclipse=in_eclipse,
            beta_angle_deg=beta_angle_deg,
            sat_position_earth_frame=sat_pos_earth,
            sat_position_inertial=sat_pos_inertial,
            sat_velocity_inertial=sat_vel_inertial,
        )

    def _compute_sun_incidence_angle_deg(self, orbit_state: OrbitState) -> float:
        """Compute the true Sun incidence angle on the +Z panel from the achieved attitude."""
        sun_unit = self._sun_vector_unit(orbit_state)
        body_z_in_inertial = self._body_axis_to_inertial(self.adcs.current_quaternion, "+Z")
        cosine_incidence = np.clip(float(np.dot(self._unit_numpy(body_z_in_inertial), sun_unit)), 0.0, 1.0)
        incidence_deg = math.degrees(math.acos(cosine_incidence))
        jitter_sigma_deg = self.config.sun_pointing_tracking_noise_deg
        if self.adcs.is_slewing:
            jitter_sigma_deg *= 1.5
        incidence_deg += float(self.rng.normal(0.0, jitter_sigma_deg))
        return float(np.clip(incidence_deg, self.config.pointing_accuracy_deg, 90.0))

    def _compute_payload_pointing_error_deg(self, orbit_state: OrbitState) -> float:
        """Return the angle between payload +X boresight and the ram direction."""

        payload_axis_in_inertial = self._payload_boresight_inertial(self.adcs.current_quaternion)
        velocity_unit = self._velocity_vector_unit(orbit_state)
        cosine_term = np.clip(
            float(np.dot(self._unit_numpy(payload_axis_in_inertial), velocity_unit)),
            -1.0,
            1.0,
        )
        return math.degrees(math.acos(cosine_term))

    def _compute_relay_pointing_error_deg(
        self,
        orbit_state: OrbitState,
        relay_axis_label: str,
        gs_visible: bool,
    ) -> float:
        """Return the relay pointing error against the active ground-target vector."""

        target_unit = self._ground_station_vector_unit(orbit_state) if gs_visible else self._nadir_vector_unit(orbit_state)
        relay_axis_in_inertial = self._relay_boresight_inertial(
            self.adcs.current_quaternion,
            relay_axis_label,
        )
        cosine_term = np.clip(
            float(np.dot(self._unit_numpy(relay_axis_in_inertial), target_unit)),
            -1.0,
            1.0,
        )
        return math.degrees(math.acos(cosine_term))

    def _compute_gs_off_boresight_deg(
        self,
        orbit_state: OrbitState,
        gs_visible: bool,
    ) -> float:
        """Return the GS off-boresight angle for the single +Y relay antenna."""

        if not gs_visible:
            return 180.0
        relay_axis_in_inertial = self._body_axis_to_inertial(self.adcs.current_quaternion, "+Y")
        gs_unit = self._ground_station_vector_unit(orbit_state)
        cosine_term = np.clip(
            float(np.dot(self._unit_numpy(relay_axis_in_inertial), gs_unit)),
            -1.0,
            1.0,
        )
        return math.degrees(math.acos(cosine_term))

    def _attitude_deadband_for_mode(self, mode: str) -> float:
        """Return the mode-specific target-update deadband."""

        if mode == "SAFE":
            return self.config.safe_hold_deadband_deg
        if mode == "SUN_POINTING":
            return self.config.attitude_command_deadband_deg
        if mode == "DETECTION":
            return self.config.ram_track_deadband_deg
        if mode == "RELAY":
            return self.config.relay_track_deadband_deg
        return self.config.attitude_command_deadband_deg

    def _derive_sun_submode(self, mode: str, sun_visible: bool, battery_wh: float) -> str:
        """Derive the current SUN_POINTING submode from power state."""

        if mode != "SUN_POINTING":
            return SunSubmode.NONE
        if not sun_visible:
            return SunSubmode.ECLIPSE
        if battery_wh >= self.config.charging_complete_threshold_wh:
            return SunSubmode.PASSIVE
        return SunSubmode.CHARGING

    def _update_attitude_guidance(
        self,
        mode: str,
        orbit_state: OrbitState,
        sun_visible: bool,
        gs_visible: bool,
        battery_wh: float,
        dt_sec: float,
        time_sec: float,
    ) -> None:
        """Update the ADCS target, enforce eclipse-hold behavior, and advance attitude dynamics."""

        if mode == "SLEWING":
            self.adcs.advance(dt_sec)
            self.adcs.is_slewing = time_sec < self.slew_end_time_sec
            self.current_attitude_error_deg = self.adcs.get_pointing_error_deg()
            self.current_sun_incidence_angle_deg = self._compute_sun_incidence_angle_deg(orbit_state)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            return

        if self.adcs.saturation_hold_active:
            self.current_attitude_command = AttitudeCommand(
                quaternion=self.adcs.target_quaternion.copy(),
                primary_objective="Saturation hold",
                secondary_objective="Freeze inertial attitude during wheel unloading",
                active_attitude_target="SATURATION_HOLD",
                attitude_hold_reason="ADCS saturation hold while magnetorquers unload momentum",
                relay_axis_label=self.adcs.commanded_relay_axis,
                sun_visible=sun_visible,
            )
            self.adcs.advance(dt_sec)
            self.current_attitude_error_deg = self.adcs.get_pointing_error_deg()
            self.current_sun_incidence_angle_deg = self._compute_sun_incidence_angle_deg(orbit_state)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            return

        if mode == "DETUMBLE":
            command = self._get_attitude_command(mode, orbit_state, sun_visible, battery_wh, gs_visible)
            self.current_attitude_command = command
            self.adcs.target_quaternion = self.adcs._normalize_quaternion(command.quaternion)
            self.adcs.commanded_relay_axis = command.relay_axis_label or "+Y"
            self._step_detumble_physics(orbit_state, dt_sec)
            self.current_attitude_error_deg, _ = self.adcs.compute_slew(
                self.adcs.current_quaternion,
                self.adcs.target_quaternion,
            )
            self.current_sun_incidence_angle_deg = self._compute_sun_incidence_angle_deg(orbit_state)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            return

        command = self._get_attitude_command(mode, orbit_state, sun_visible, battery_wh, gs_visible)
        if self.current_attitude_command is None:
            self.adcs.set_initial_quaternion(command.quaternion)
            self.current_attitude_command = command
            self.adcs.commanded_relay_axis = command.relay_axis_label or "+Y"
            self.current_attitude_error_deg = 0.0
            self.current_sun_incidence_angle_deg = self._compute_sun_incidence_angle_deg(orbit_state)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            return

        force_update = False

        mode_deadband_deg = self._attitude_deadband_for_mode(mode)
        self.adcs.command_attitude(
            command.quaternion,
            mode,
            command.primary_objective,
            force_update=force_update,
            deadband_deg=mode_deadband_deg,
        )

        self.current_attitude_command = command
        self.adcs.commanded_relay_axis = command.relay_axis_label or "+Y"

        self.adcs.advance(dt_sec)

        self.current_attitude_error_deg = self.adcs.get_pointing_error_deg()
        self.current_sun_incidence_angle_deg = self._compute_sun_incidence_angle_deg(orbit_state)
        self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
        self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
            orbit_state,
            self.adcs.commanded_relay_axis,
            gs_visible,
        )

    def _update_storage(self, generated_mb: float, downlinked_mb: float) -> None:
        """Update onboard storage while enforcing physical lower and upper bounds."""

        available_science_capacity_mb = max(
            0.0,
            self.config.onboard_storage_mb - self.initial_non_science_storage_mb,
        )
        self.science_storage_mb = min(
            available_science_capacity_mb,
            max(0.0, self.science_storage_mb + generated_mb - downlinked_mb),
        )

    def _calculate_slew_duration_sec(self, target_quaternion: np.ndarray) -> Tuple[float, float]:
        """Estimate a mode-transition slew duration from the current and target attitudes."""

        angle_deg, _ = self.adcs.compute_slew(self.adcs.current_quaternion, target_quaternion)
        duration_sec = max(
            self.config.slew_transition_penalty_sec,
            (angle_deg / max(self.config.slew_transition_rate_deg_sec, 1.0e-9)) + self.config.settling_time_sec,
        )
        return angle_deg, duration_sec

    def _begin_slew_transition(
        self,
        from_mode: str,
        target_mode: str,
        reason: str,
        orbit_state: OrbitState,
        sun_visible: bool,
        battery_wh: float,
        gs_visible: bool,
        time_sec: float,
    ) -> bool:
        """Enter SLEWING and command the target attitude for the requested mode."""

        target_command = self._get_attitude_command(target_mode, orbit_state, sun_visible, battery_wh, gs_visible)
        angle_deg, duration_sec = self._calculate_slew_duration_sec(target_command.quaternion)
        self.target_mode = target_mode
        self.slew_start_time_sec = time_sec
        self.slew_end_time_sec = time_sec + duration_sec
        self.fsm.target_mode = target_mode
        self.fsm.slew_start_time_sec = time_sec
        self.fsm.slew_end_time_sec = self.slew_end_time_sec

        self.adcs.command_attitude(
            target_command.quaternion,
            "SLEWING",
            target_command.primary_objective,
            force_update=True,
            deadband_deg=self.config.attitude_command_deadband_deg,
        )
        self.current_attitude_command = target_command

        transitioned = self.fsm.transition("SLEWING", f"Slewing to {target_mode}: {reason}")
        if transitioned:
            self.active_slew_event_index = len(self.logger.slew_event_rows)
            self.logger.log_slew_event(
                start_time_sec=time_sec,
                from_mode=from_mode,
                target_mode=target_mode,
                angle_deg=angle_deg,
                estimated_duration_sec=duration_sec,
            )
        return transitioned

    def _complete_slew_transition(self, time_sec: float) -> bool:
        """Complete the active slew and transition into the target mode."""

        if self.fsm.current_mode != "SLEWING":
            return False
        target_mode = self.target_mode
        if target_mode not in self.fsm.ALLOWED_TRANSITIONS["SLEWING"]:
            target_mode = "SUN_POINTING"
        self.fsm.target_mode = target_mode
        self.fsm.slew_end_time_sec = time_sec
        self.adcs.is_slewing = False
        self.adcs.settling = False
        if self.active_slew_event_index is not None:
            self.logger.finalize_slew_event(self.active_slew_event_index, time_sec)
            self.active_slew_event_index = None
        return self.fsm.transition(target_mode, f"Slew complete -> {target_mode}")

    def _begin_detumble_phase(self, time_sec: float) -> None:
        """Start the startup detumble log when the spacecraft first enters DETUMBLE."""

        self.adcs.detumble_phase = "BDOT"
        self.adcs.detumble_complete = False
        self.adcs.last_b_body_t = None
        self.fsm.detumble_complete = False
        if self.active_detumble_event_index is not None:
            return
        self.active_detumble_event_index = len(self.logger.detumble_event_rows)
        self.logger.log_detumble_event(
            start_time_sec=time_sec,
            from_mode="COLD_AND_DARK",
            target_attitude_label=self.config.detumble_target_attitude_label,
            initial_body_rate_deg_sec=self.adcs.initial_body_rate_deg_sec,
            initial_body_accel_deg_sec2=self.adcs.initial_body_accel_deg_sec2,
            goal="Reduce initial tumble and settle to the nominal default orientation",
        )

    def _complete_detumble_phase(self, time_sec: float) -> None:
        """Finalize the startup detumble log when the spacecraft leaves DETUMBLE."""

        self.fsm.detumble_complete = self.adcs.detumble_complete
        if self.active_detumble_event_index is None:
            return
        self.logger.finalize_detumble_event(self.active_detumble_event_index, time_sec)
        self.active_detumble_event_index = None

    def _finalise_event_logs(
        self,
        final_time_sec: float,
        final_absolute_date: AbsoluteDate,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Close open SAA, GS, detumble, and slew events and convert logs to DataFrames."""

        self.saa.close_open_pass(final_time_sec, final_absolute_date)
        self.ground_station.close_open_contact(final_time_sec, final_absolute_date)
        if self.active_slew_event_index is not None:
            self.logger.finalize_slew_event(self.active_slew_event_index, final_time_sec)
            self.active_slew_event_index = None
        if self.active_detumble_event_index is not None:
            self.logger.finalize_detumble_event(self.active_detumble_event_index, final_time_sec)
            self.active_detumble_event_index = None
        return self.logger.build_dataframes(
            self.saa.completed_passes,
            self.ground_station.completed_contacts,
            self.fsm.transition_history,
        )

    def _plot_results(
        self,
        telemetry_df: pd.DataFrame,
        saa_passes_df: pd.DataFrame,
        gs_contacts_df: pd.DataFrame,
        slew_events_df: pd.DataFrame,
    ) -> None:
        """Render the required 9-panel ANTARA mission visualisation suite."""

        telemetry_df = telemetry_df.copy()
        telemetry_df["time_hours"] = telemetry_df["time_sec"] / 3600.0
        telemetry_df["time_days"] = telemetry_df["time_sec"] / 86400.0
        cumulative_science_mb = telemetry_df["cumulative_science_generated_mb"]
        cumulative_downlinked_mb = telemetry_df["cumulative_downlinked_mb"]
        telemetry_df["cumulative_science_mb"] = cumulative_science_mb
        telemetry_df["cumulative_downlinked_mb"] = cumulative_downlinked_mb
        telemetry_df["mode_code"] = telemetry_df["mode"].map(
            {mode: idx for idx, mode in enumerate(self.config.mode_colors)}
        )
        telemetry_df["wheel_momentum_mag"] = np.linalg.norm(
            telemetry_df[["wheel_mom_x", "wheel_mom_y", "wheel_mom_z"]].to_numpy(),
            axis=1,
        )

        figure = plt.figure(figsize=(24, 18), constrained_layout=True)
        grid = figure.add_gridspec(3, 3)

        map_projection = ccrs.PlateCarree()
        axis1 = figure.add_subplot(grid[0, 0], projection=map_projection)
        axis2 = figure.add_subplot(grid[0, 1], projection=map_projection)
        axis3 = figure.add_subplot(grid[0, 2])
        axis4 = figure.add_subplot(grid[1, 0])
        axis5 = figure.add_subplot(grid[1, 1])
        axis6 = figure.add_subplot(grid[1, 2])
        axis7 = figure.add_subplot(grid[2, 0])
        axis8 = figure.add_subplot(grid[2, 1])
        axis9 = figure.add_subplot(grid[2, 2])

        self._format_map_axis(axis1, f"Panel 1: {self.config.sim_duration_hours:.1f}-Hour Ground Track by Mode")
        for mode, color in self.config.mode_colors.items():
            mode_rows = telemetry_df[telemetry_df["mode"] == mode]
            if mode_rows.empty:
                continue
            axis1.scatter(
                mode_rows["lon_deg"],
                mode_rows["lat_deg"],
                s=2,
                color=color,
                transform=map_projection,
                label=mode,
                alpha=0.75,
            )
        axis1.legend(loc="lower left", fontsize=8)

        self._format_map_axis(axis2, "Panel 2: SAA Overlay and Science Density")
        science_rows = telemetry_df[telemetry_df["science_data_mb"] > 0.0]
        if not science_rows.empty:
            axis2.hexbin(
                science_rows["lon_deg"],
                science_rows["lat_deg"],
                C=science_rows["science_data_mb"],
                reduce_C_function=np.sum,
                gridsize=45,
                cmap="inferno",
                transform=map_projection,
                mincnt=1,
            )
        polygon_lon, polygon_lat = self.saa.saa_polygon.exterior.xy
        axis2.plot(polygon_lon, polygon_lat, color="orange", linewidth=2.0, transform=map_projection)

        axis3.set_title("Panel 3: Battery vs Time")
        axis3.plot(telemetry_df["time_days"], telemetry_df["battery_wh"], color="forestgreen", linewidth=1.2)
        axis3.set_xlabel("Time (days)")
        axis3.set_ylabel("Battery (Wh)")
        self._shade_intervals(axis3, telemetry_df, "in_eclipse", "lightgray", 0.4)
        self._band_events(axis3, saa_passes_df, "entry_time_sec", "exit_time_sec", "red", 0.12)
        self._band_events(axis3, gs_contacts_df, "aos_time_sec", "los_time_sec", "blue", 0.08)
        axis3.grid(True, alpha=0.3)

        axis4.set_title("Panel 4: Solar Generation and Load")
        axis4.plot(telemetry_df["time_days"], telemetry_df["solar_w"], label="Solar Generation", color="goldenrod")
        axis4.plot(telemetry_df["time_days"], telemetry_df["load_w"], label="Electrical Load", color="black")
        self._shade_intervals(axis4, telemetry_df, "in_eclipse", "lightgray", 0.4)
        axis4.set_xlabel("Time (days)")
        axis4.set_ylabel("Power (W)")
        axis4.legend()
        axis4.grid(True, alpha=0.3)

        axis5.set_title("Panel 5: Mode Timeline (top) / Sun Submode (bottom)")
        mode_matrix = np.array([telemetry_df["mode_code"].to_numpy()])
        cmap_mode = plt.matplotlib.colors.ListedColormap(
            [self.config.mode_colors[m] for m in self.config.mode_colors]
        )
        axis5.imshow(
            mode_matrix,
            aspect="auto",
            cmap=cmap_mode,
            extent=[
                telemetry_df["time_days"].iloc[0],
                telemetry_df["time_days"].iloc[-1],
                0.5, 1.0,
            ],
        )
        submode_order = [SunSubmode.NONE, SunSubmode.CHARGING, SunSubmode.PASSIVE, SunSubmode.ECLIPSE]
        submode_to_idx = {s: i for i, s in enumerate(submode_order)}
        submode_codes = telemetry_df["sun_submode"].map(submode_to_idx).fillna(0).astype(int).to_numpy()
        submode_matrix = np.array([submode_codes])
        cmap_submode = plt.matplotlib.colors.ListedColormap(
            [self.config.sun_submode_colors[s] for s in submode_order]
        )
        axis5.imshow(
            submode_matrix,
            aspect="auto",
            cmap=cmap_submode,
            extent=[
                telemetry_df["time_days"].iloc[0],
                telemetry_df["time_days"].iloc[-1],
                0.0, 0.5,
            ],
        )
        axis5.set_yticks([0.25, 0.75])
        axis5.set_yticklabels(["Sun\nSubmode", "Mode"], fontsize=7)
        axis5.set_xlabel("Time (days)")
        axis5.set_xlim(telemetry_df["time_days"].iloc[0], telemetry_df["time_days"].iloc[-1])

        axis6.set_title("Panel 6: Science, Downlink, and Storage")
        axis6.plot(
            telemetry_df["time_days"],
            telemetry_df["cumulative_science_mb"],
            label="Cumulative Science",
            color="red",
        )
        axis6.plot(
            telemetry_df["time_days"],
            telemetry_df["cumulative_downlinked_mb"],
            label="Cumulative Downlinked",
            color="blue",
        )
        axis6.set_xlabel("Time (days)")
        axis6.set_ylabel("Data (MB)")
        axis6.grid(True, alpha=0.3)
        axis6b = axis6.twinx()
        # Show science-only storage (excludes fixed firmware/OS baseline of initial_storage_mb)
        science_only_storage = telemetry_df["storage_mb"] - self.config.initial_storage_mb
        axis6b.plot(
            telemetry_df["time_days"],
            science_only_storage.clip(lower=0.0),
            label="Science Storage",
            color="purple",
            linestyle="--",
        )
        axis6b.set_ylabel("Science Storage (MB)")
        axis6.legend(loc="upper left")
        axis6b.legend(loc="upper right")

        axis7.set_title("Panel 7: Slew Histogram")
        if not slew_events_df.empty:
            axis7.hist(slew_events_df["angle_deg"], bins=20, color="slateblue", alpha=0.85)
            median_slew_deg = float(slew_events_df["angle_deg"].median())
            max_slew_deg = float(slew_events_df["angle_deg"].max())
            axis7.axvline(median_slew_deg, color="green", linestyle="--", label=f"Median {median_slew_deg:.1f} deg")
            axis7.axvline(max_slew_deg, color="red", linestyle="--", label=f"Max {max_slew_deg:.1f} deg")
            axis7.axvline(
                self.config.large_slew_threshold_deg,
                color="black",
                linestyle=":",
                label=f"Large Slew {self.config.large_slew_threshold_deg:.0f} deg",
            )
            axis7.legend(fontsize=8)
        axis7.set_xlabel("Slew Angle (deg)")
        axis7.set_ylabel("Count")
        axis7.grid(True, alpha=0.3)

        axis8.set_title("Panel 8: Wheel Momentum Magnitude")
        axis8.plot(telemetry_df["time_days"], telemetry_df["wheel_momentum_mag"], color="teal")
        axis8.axhline(
            self.config.wheel_max_momentum_nms,
            color="red",
            linestyle="--",
            label="Saturation Limit",
        )
        axis8.set_xlabel("Time (days)")
        axis8.set_ylabel("Wheel Momentum (Nms)")
        axis8.legend()
        axis8.grid(True, alpha=0.3)

        axis9.set_title("Panel 9: GS Contact Timeline")
        if not gs_contacts_df.empty:
            for row_index, (_, row) in enumerate(gs_contacts_df.iterrows()):
                start_day = row["aos_time_sec"] / 86400.0
                duration_day = (row["los_time_sec"] - row["aos_time_sec"]) / 86400.0
                axis9.barh(row_index, duration_day, left=start_day, color="steelblue", alpha=0.8)
                axis9.text(
                    start_day + duration_day,
                    row_index,
                    f" {row['max_el_deg']:.1f} deg, {row['downlinked_mb']:.1f} MB",
                    fontsize=7,
                    va="center",
                )
            axis9.set_yticks(range(len(gs_contacts_df)))
            axis9.set_yticklabels([f"Pass {idx + 1}" for idx in range(len(gs_contacts_df))])
        else:
            axis9.text(0.5, 0.5, "No GS contacts", ha="center", va="center", transform=axis9.transAxes)
        axis9.set_xlabel("Time (days)")
        axis9.grid(True, alpha=0.3)

        output_path = figures_path("master", "ANTARA_7day_PDR_simulation.png")
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

    def _format_map_axis(self, axis, title: str) -> None:
        """Apply consistent coastlines and labels to cartopy map panels."""

        axis.set_title(title)
        axis.set_global()
        axis.coastlines(linewidth=0.7)
        axis.add_feature(cfeature.BORDERS, linewidth=0.3)
        axis.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4)

    def _shade_intervals(self, axis, telemetry_df: pd.DataFrame, column_name: str, color: str, alpha: float) -> None:
        """Shade contiguous telemetry intervals where the specified boolean column is true."""

        time_days = telemetry_df["time_days"].to_numpy()
        flag_values = telemetry_df[column_name].to_numpy(dtype=bool)
        start_idx = None
        for idx, flag in enumerate(flag_values):
            if flag and start_idx is None:
                start_idx = idx
            if not flag and start_idx is not None:
                axis.axvspan(time_days[start_idx], time_days[idx - 1], color=color, alpha=alpha)
                start_idx = None
        if start_idx is not None:
            axis.axvspan(time_days[start_idx], time_days[-1], color=color, alpha=alpha)

    def _band_events(
        self,
        axis,
        event_df: pd.DataFrame,
        start_col: str,
        end_col: str,
        color: str,
        alpha: float,
    ) -> None:
        """Draw vertical event bands for SAA passes or GS contacts."""

        if event_df.empty:
            return
        for _, row in event_df.iterrows():
            if pd.isna(row.get(start_col)) or pd.isna(row.get(end_col)):
                continue
            axis.axvspan(row[start_col] / 86400.0, row[end_col] / 86400.0, color=color, alpha=alpha)

    def _print_summary(
        self,
        telemetry_df: pd.DataFrame,
        saa_passes_df: pd.DataFrame,
        gs_contacts_df: pd.DataFrame,
        mode_transitions_df: pd.DataFrame,
        slew_events_df: pd.DataFrame,
    ) -> None:
        """Print the required ANTARA PDR mission summary to the console."""

        total_sim_seconds = self.config.sim_duration_hours * 3600.0
        avg_orbital_period_min = float(np.mean(self.orbital_period_history_min)) if self.orbital_period_history_min else np.nan
        total_revolutions = int(total_sim_seconds / (avg_orbital_period_min * 60.0)) if avg_orbital_period_min > 0.0 else 0
        beta_min_deg = float(np.min(self.beta_angle_history_deg)) if self.beta_angle_history_deg else np.nan
        beta_max_deg = float(np.max(self.beta_angle_history_deg)) if self.beta_angle_history_deg else np.nan

        total_slew_events = len(slew_events_df)
        total_slew_time_min = self.adcs.total_slew_time_sec / 60.0
        avg_slew_angle_deg = float(slew_events_df["angle_deg"].mean()) if not slew_events_df.empty else 0.0
        max_slew_angle_deg = float(slew_events_df["angle_deg"].max()) if not slew_events_df.empty else 0.0
        average_slew_frequency_per_hour = total_slew_events / max(self.config.sim_duration_hours, 1.0e-9)
        large_slews = int((slew_events_df["angle_deg"] > self.config.large_slew_threshold_deg).sum()) if not slew_events_df.empty else 0
        wheel_momentum_mag = np.linalg.norm(
            telemetry_df[["wheel_mom_x", "wheel_mom_y", "wheel_mom_z"]].to_numpy(),
            axis=1,
        )
        max_wheel_momentum_nms = float(np.max(wheel_momentum_mag)) if len(wheel_momentum_mag) > 0 else 0.0
        adcs_duty_cycle_percent = 100.0 * (
            (telemetry_df["slew_angle_deg"] > 0.0) | (telemetry_df["attitude_error_deg"] > self.config.pointing_accuracy_deg)
        ).mean()

        total_saa_passes = len(saa_passes_df)
        total_saa_dwell_hours = float(saa_passes_df["dwell_sec"].sum() / 3600.0) if not saa_passes_df.empty else 0.0
        total_detection_hours = self.payload.total_detection_seconds / 3600.0
        science_duty_cycle_percent = 100.0 * self.payload.total_detection_seconds / total_sim_seconds

        total_gs_contacts = len(gs_contacts_df)
        total_contact_time_min = self.communications.total_contact_seconds / 60.0
        total_downlinked_mb = self.communications.total_downlinked_mb
        downlink_efficiency_percent = (
            100.0 * total_downlinked_mb / self.payload.total_science_data_generated_mb
            if self.payload.total_science_data_generated_mb > 0.0
            else 0.0
        )

        min_battery_wh = float(telemetry_df["battery_wh"].min())
        max_battery_wh = float(telemetry_df["battery_wh"].max())
        avg_solar_generation_w = float(telemetry_df["solar_w"].mean())
        avg_load_w = float(telemetry_df["load_w"].mean())
        def _infer_adcs_power(row: pd.Series) -> float:
            base_adcs_power_w = row["load_w"] - (self.config.obc_power_w + self.config.eps_power_w)
            if row["mode"] == "DETECTION":
                base_adcs_power_w -= self.config.payload_power_w
            elif row["mode"] == "RELAY":
                base_adcs_power_w -= self.config.radio_tx_power_w
            return max(0.0, float(base_adcs_power_w))

        avg_adcs_power_w = float(telemetry_df.apply(_infer_adcs_power, axis=1).mean()) if not telemetry_df.empty else 0.0
        avg_net_power_w = float((telemetry_df["solar_w"] - telemetry_df["load_w"]).mean())
        eclipse_fraction_percent = 100.0 * telemetry_df["in_eclipse"].mean()
        power_positive_fraction_percent = 100.0 * (telemetry_df["solar_w"] > telemetry_df["load_w"]).mean()
        average_sun_incidence_deg = float(telemetry_df["sun_incidence_angle_deg"].mean())
        max_sun_incidence_deg = float(telemetry_df["sun_incidence_angle_deg"].max())
        relay_rows = telemetry_df[telemetry_df["mode"] == "RELAY"]
        detection_rows = telemetry_df[telemetry_df["mode"] == "DETECTION"]
        average_relay_pointing_error_deg = (
            float(relay_rows["relay_pointing_error_deg"].mean()) if not relay_rows.empty else 0.0
        )
        average_payload_ram_pointing_error_deg = (
            float(detection_rows["payload_pointing_error_deg"].mean()) if not detection_rows.empty else 0.0
        )
        average_attitude_error_per_mode = (
            telemetry_df.groupby("mode")["attitude_error_deg"].mean().to_dict()
            if not telemetry_df.empty
            else {}
        )
        average_wheel_momentum_nms = float(np.mean(wheel_momentum_mag)) if len(wheel_momentum_mag) > 0 else 0.0

        sim_duration_label = (
            f"{self.config.sim_duration_hours:.0f}-hour"
            if self.config.sim_duration_hours < 48.0
            else f"{self.config.sim_duration_hours / 24.0:.0f}-day"
        )
        print(f"\n--- ANTARA Mission Summary ({sim_duration_label} PDR Simulation) ---\n")
        print("Orbit:")
        print(f"  Total revolutions completed: {total_revolutions}")
        print(f"  Average orbital period (min): {avg_orbital_period_min:.2f}")
        print(f"  Beta angle range (deg): {beta_min_deg:.2f} to {beta_max_deg:.2f}")

        print("\nADCS Performance:")
        print(f"  Total mode transitions: {len(mode_transitions_df)}")
        print(f"  Total slew events: {total_slew_events}")
        print(f"  Total slew time (min): {total_slew_time_min:.2f}")
        print(f"  Average slew angle (deg): {avg_slew_angle_deg:.2f}")
        print(f"  Average slew frequency (events/hr): {average_slew_frequency_per_hour:.2f}")
        print(f"  Max slew angle (deg): {max_slew_angle_deg:.2f}")
        print(f"  Large slews > 60 deg (count): {large_slews}")
        print(f"  Max wheel momentum (Nms): {max_wheel_momentum_nms:.4f}")
        print(f"  Wheel saturation events: {self.adcs.saturation_event_count}")
        print(f"  Quaternion continuity corrections: {self.adcs.quaternion_continuity_corrections}")
        print(f"  Rejected micro-slew commands: {self.adcs.rejected_micro_slew_commands}")
        print(f"  ADCS duty cycle (%): {adcs_duty_cycle_percent:.2f}")
        print(f"  Average sun incidence angle (deg): {average_sun_incidence_deg:.2f}")
        print(f"  Max sun incidence angle (deg): {max_sun_incidence_deg:.2f}")
        print(f"  Average relay Earth-pointing error (deg): {average_relay_pointing_error_deg:.2f}")
        print(f"  Average payload ram-pointing error (deg): {average_payload_ram_pointing_error_deg:.2f}")
        print("  Average attitude error per mode (deg):")
        for mode_name in [
            "COLD_AND_DARK",
            "DETUMBLE",
            "SAFE",
            "SLEWING",
            "SUN_POINTING",
            "DETECTION",
            "RELAY",
        ]:
            print(f"    {mode_name}: {average_attitude_error_per_mode.get(mode_name, 0.0):.2f}")

        print("\nScience:")
        print(f"  Total SAA passes: {total_saa_passes}")
        print(f"  Total SAA dwell time (hours): {total_saa_dwell_hours:.2f}")
        print(f"  Total DETECTION time (hours): {total_detection_hours:.2f}")
        print(f"  Total science data generated (MB): {self.payload.total_science_data_generated_mb:.2f}")
        print(f"  Science duty cycle (%): {science_duty_cycle_percent:.2f}")

        print("\nCommunications:")
        print(f"  Total GS contacts: {total_gs_contacts}")
        print(f"  Total contact time (min): {total_contact_time_min:.2f}")
        print(f"  Total data downlinked (MB): {total_downlinked_mb:.2f}")
        print(f"  Downlink efficiency (downlinked / generated %): {downlink_efficiency_percent:.2f}")

        print("\nPower:")
        print(f"  Min battery (Wh): {min_battery_wh:.2f}")
        print(f"  Max battery (Wh): {max_battery_wh:.2f}")
        print(f"  Average solar generation (W): {avg_solar_generation_w:.2f}")
        print(f"  Average load (W): {avg_load_w:.2f}")
        print(f"  Average ADCS power (W): {avg_adcs_power_w:.2f}")
        print(f"  Average net power (W): {avg_net_power_w:.2f}")
        print(f"  Eclipse fraction (%): {eclipse_fraction_percent:.2f}")
        print(f"  Power-positive fraction (%): {power_positive_fraction_percent:.2f}")
        print(f"  Average wheel momentum (Nms): {average_wheel_momentum_nms:.5f}")

        print("\nMode Durations (hours):")
        for mode in [
            "COLD_AND_DARK",
            "DETUMBLE",
            "SAFE",
            "SLEWING",
            "SUN_POINTING",
            "DETECTION",
            "RELAY",
        ]:
            mode_hours = self.fsm.mode_durations_sec[mode] / 3600.0
            mode_duty_cycle_percent = 100.0 * self.fsm.mode_durations_sec[mode] / total_sim_seconds
            print(f"  {mode}: {mode_hours:.2f} ({mode_duty_cycle_percent:.2f}%)")

        if "sun_submode" in telemetry_df.columns:
            print("\nSUN_POINTING Submode Breakdown:")
            for submode_label in [SunSubmode.CHARGING, SunSubmode.PASSIVE, SunSubmode.ECLIPSE]:
                submode_sec = (telemetry_df["sun_submode"] == submode_label).sum() * self.config.step_sec
                submode_hours = submode_sec / 3600.0
                sun_pointing_total_sec = self.fsm.mode_durations_sec["SUN_POINTING"]
                submode_pct = (
                    100.0 * submode_sec / sun_pointing_total_sec
                    if sun_pointing_total_sec > 0.0
                    else 0.0
                )
                print(f"  {submode_label:<8}: {submode_hours:.2f} hr ({submode_pct:.1f}% of SUN_POINTING)")

    def _print_progress(
        self,
        time_sec: float,
        final_time_sec: float,
        mode: str,
        battery_wh: float,
        storage_mb: float,
        solar_w: float,
        load_w: float,
    ) -> None:
        """Print a concise simulation progress line for long mission runs."""

        percent_complete = 100.0 * min(time_sec / final_time_sec, 1.0) if final_time_sec > 0.0 else 100.0
        elapsed_hours = time_sec / 3600.0
        print(
            f"[Progress] {percent_complete:6.2f}% | "
            f"T+{elapsed_hours:7.2f} hr | "
            f"Mode={mode:<13} | "
            f"Battery={battery_wh:5.2f} Wh | "
            f"Solar={solar_w:5.2f} W | "
            f"Load={load_w:5.2f} W | "
            f"Storage={storage_mb:10.3f} MB"
        )

    def run(self) -> None:
        """Run the ANTARA mission simulation for the configured duration.

        The default configuration is 24 hours. The run logs telemetry, renders
        the plots, writes CSV outputs, and prints the mission summary.
        """

        final_time_sec = self.config.sim_duration_hours * 3600.0
        step_sec = self.config.step_sec
        progress_interval_sec = max(step_sec, self.config.progress_report_interval_sec)
        next_progress_report_sec = 0.0

        duration_label = (
            f"{self.config.sim_duration_hours:.0f}-hour"
            if self.config.sim_duration_hours < 48.0
            else f"{self.config.sim_duration_hours / 24.0:.0f}-day"
        )
        print(
            f"Starting ANTARA {duration_label} simulation | "
            f"Duration={self.config.sim_duration_hours:.1f} hr | "
            f"Timestep={step_sec:.1f} s"
        )

        time_sec = 0.0
        while time_sec <= final_time_sec:
            self.fsm.set_time(time_sec)
            self.adcs.set_time(time_sec)
            current_mode = self.fsm.current_mode
            absolute_date = self.start_date.shiftedBy(time_sec)
            propagated_state = self._propagate_state(absolute_date)
            orbit_state = self._build_orbit_state(propagated_state, time_sec)

            osculating_period_min = propagated_state.getOrbit().getKeplerianPeriod() / 60.0
            self.orbital_period_history_min.append(osculating_period_min)
            self.beta_angle_history_deg.append(orbit_state.beta_angle_deg)

            gs_elevation_deg = self.ground_station.get_elevation(
                orbit_state.sat_position_earth_frame,
                orbit_state.absolute_date,
            )
            gs_visible = self.ground_station.is_visible(gs_elevation_deg)
            inside_saa = self.saa.is_inside(orbit_state.latitude_deg, orbit_state.longitude_deg)
            sun_visible = not orbit_state.in_eclipse
            illumination_fraction = self._update_illumination_fraction(sun_visible, time_sec)
            battery_before_step_wh = self.eps.battery_wh

            if self.fsm.current_mode == "SLEWING" and time_sec >= self.slew_end_time_sec:
                self._complete_slew_transition(time_sec)
            current_mode = self.fsm.current_mode

            self._update_attitude_guidance(
                current_mode,
                orbit_state,
                sun_visible,
                gs_visible,
                self.eps.battery_wh,
                step_sec,
                time_sec,
            )
            (
                payload_boresight_inertial,
                velocity_unit_vector,
                relay_boresight_vector,
                gs_los_vector,
                payload_alignment_cosine,
                relay_alignment_cosine,
                gs_off_boresight_deg,
            ) = self._update_alignment_telemetry(orbit_state, gs_visible)
            # self._validate_attitude_metrics(current_mode, orbit_state, gs_visible)
            startup_allows_charging = current_mode not in {"COLD_AND_DARK", "DETUMBLE"}
            self.eps.compute_solar_power(
                self.current_sun_incidence_angle_deg,
                illumination_fraction if startup_allows_charging else 0.0,
            )
            self.eps.compute_load(
                current_mode,
                self.adcs.is_slewing,
                not sun_visible,
            )

            science_data_step_mb = 0.0
            payload_pointing_valid = (
                self.current_payload_pointing_error_deg <= self.config.payload_pointing_valid_deg
            )
            payload_settled = self.adcs.settled_time_sec >= self.config.minimum_settled_time_before_payload_sec
            science_generation_enabled = (
                current_mode == "DETECTION"
                and inside_saa
                and payload_pointing_valid
                and payload_settled
                and self.payload.is_operable(self.eps.battery_wh, current_mode)
            )
            if science_generation_enabled:
                science_data_step_mb = self.payload.generate_data(step_sec)
                if current_mode == "DETECTION":
                    assert science_data_step_mb > 0.0
            if current_mode == "DETECTION" and science_generation_enabled and science_data_step_mb <= 0.0:
                self._maybe_warn("detection_zero_science", "DETECTION active with valid pointing but zero science generated.")

            downlinked_step_mb = 0.0
            relay_pointing_valid = (
                self.current_relay_pointing_error_deg <= self.config.relay_pointing_valid_deg
            )
            relay_settled = self.adcs.settled_time_sec >= self.config.minimum_settled_time_before_tx_sec
            relay_link_active = (
                current_mode == "RELAY"
                and gs_visible
                and relay_settled
                and self.science_storage_mb > 0.0
                and relay_pointing_valid
                and gs_off_boresight_deg <= self.config.directional_antenna_half_beamwidth_deg
            )
            if relay_link_active:
                downlinked_step_mb = self.communications.downlink_step(self.science_storage_mb, step_sec)
                assert downlinked_step_mb > 0.0
            if gs_visible:
                self.communications.record_contact_time(step_sec)

            self._update_storage(science_data_step_mb, downlinked_step_mb)
            if current_mode == "DETECTION" and science_generation_enabled:
                self.detection_no_science_duration_sec = 0.0
            elif current_mode == "DETECTION" and inside_saa and payload_pointing_valid:
                self.detection_no_science_duration_sec += step_sec
                if self.detection_no_science_duration_sec > 60.0:
                    self._maybe_warn(
                        "detection_no_science_long",
                        "DETECTION persisted for >60 s without science generation.",
                    )
            else:
                self.detection_no_science_duration_sec = 0.0

            saa_event = self.saa.detect_entry_exit(
                self.previous_in_saa,
                inside_saa,
                time_sec,
                orbit_state.latitude_deg,
                orbit_state.longitude_deg,
                orbit_state.absolute_date,
                science_data_step_mb,
            )
            gs_event = self.ground_station.detect_aos_los(
                self.previous_gs_visible,
                gs_visible,
                time_sec,
                gs_elevation_deg,
                orbit_state.absolute_date,
                downlinked_step_mb,
            )
            self.previous_in_saa = inside_saa
            self.previous_gs_visible = gs_visible

            if current_mode != "DETUMBLE":
                torque_vector_nm = self.adcs.estimate_torque_vector()
                unload_scale = 1.0 if sun_visible else 0.5
                b_projection_scale = max(0.2, abs(math.sin(math.radians(orbit_state.beta_angle_deg))))
                self.adcs.update_wheel_momentum(torque_vector_nm, step_sec, unload_scale, b_projection_scale)

            self.eps.step(step_sec)
            self.fsm.accumulate_mode_time(step_sec)
            self._run_physics_sanity_checks(
                current_mode,
                orbit_state,
                gs_visible,
                inside_saa,
                science_data_step_mb,
                downlinked_step_mb,
                battery_before_step_wh,
            )

            self.logger.log_step(
                time_sec=time_sec,
                orbit_state=orbit_state,
                mode=current_mode,
                battery_wh=self.eps.battery_wh,
                solar_w=self.eps.solar_generation_w,
                load_w=self.eps.load_w,
                in_sunlight=sun_visible,
                in_saa=inside_saa,
                gs_visible=gs_visible,
                adcs_model=self.adcs,
                storage_mb=self._get_total_storage_mb(),
                science_data_mb=science_data_step_mb,
                downlinked_mb=downlinked_step_mb,
                sun_visible=sun_visible,
                sun_incidence_angle_deg=self.current_sun_incidence_angle_deg,
                relay_pointing_error_deg=self.current_relay_pointing_error_deg,
                gs_off_boresight_deg=self.current_gs_off_boresight_deg,
                payload_pointing_error_deg=self.current_payload_pointing_error_deg,
                attitude_error_deg=self.current_attitude_error_deg,
                cumulative_science_generated_mb=self.payload.total_science_data_generated_mb,
                cumulative_downlinked_mb=self.communications.total_downlinked_mb,
                net_storage_delta_mb=science_data_step_mb - downlinked_step_mb,
                saturation_hold_active=self.adcs.saturation_hold_active,
                active_attitude_target=self.current_attitude_command.active_attitude_target if self.current_attitude_command else "UNINITIALISED",
                attitude_hold_reason=self.current_attitude_command.attitude_hold_reason if self.current_attitude_command else "Initialisation",
                sun_submode=self._derive_sun_submode(current_mode, sun_visible, self.eps.battery_wh),
                science_generation_enabled=science_generation_enabled,
                relay_link_active=relay_link_active,
                payload_alignment_cosine=payload_alignment_cosine,
                relay_alignment_cosine=relay_alignment_cosine,
                payload_boresight_inertial=payload_boresight_inertial,
                velocity_unit_vector=velocity_unit_vector,
                relay_boresight_vector=relay_boresight_vector,
                gs_los_vector=gs_los_vector,
                illumination_fraction=illumination_fraction,
            )

            requested_mode, reason = self.fsm.evaluate(
                self.eps.battery_wh,
                gs_visible,
                inside_saa,
                self.adcs.saturation_hold_active,
                self.adcs.get_saturation_hold_duration_sec(),
                self.current_attitude_error_deg,
                self.adcs.detumble_complete,
            )
            transitioned = False
            if requested_mode != current_mode:
                try:
                    if current_mode == "SLEWING":
                        transitioned = False
                    elif current_mode in {"COLD_AND_DARK", "DETUMBLE"} or requested_mode in {"COLD_AND_DARK", "DETUMBLE"}:
                        transitioned = self.fsm.transition(requested_mode, reason)
                        if transitioned:
                            if current_mode == "COLD_AND_DARK" and requested_mode == "DETUMBLE":
                                self._begin_detumble_phase(time_sec)
                            elif current_mode == "DETUMBLE" and requested_mode != "DETUMBLE":
                                self._complete_detumble_phase(time_sec)
                    else:
                        transitioned = self._begin_slew_transition(
                            from_mode=current_mode,
                            target_mode=requested_mode,
                            reason=reason,
                            orbit_state=orbit_state,
                            sun_visible=sun_visible,
                            battery_wh=self.eps.battery_wh,
                            gs_visible=gs_visible,
                            time_sec=time_sec,
                        )
                except ValueError as exc:
                    transitioned = False
            else:
                transitioned = False

            if transitioned:
                self._steps_since_mode_transition = 0
            else:
                self._steps_since_mode_transition += 1

            if time_sec >= next_progress_report_sec or math.isclose(time_sec, final_time_sec):
                self._print_progress(
                    time_sec=time_sec,
                    final_time_sec=final_time_sec,
                    mode=current_mode,
                    battery_wh=self.eps.battery_wh,
                    storage_mb=self._get_total_storage_mb(),
                    solar_w=self.eps.solar_generation_w,
                    load_w=self.eps.load_w,
                )
                next_progress_report_sec += progress_interval_sec

            time_sec += step_sec

        (
            telemetry_df,
            saa_passes_df,
            gs_contacts_df,
            mode_transitions_df,
            slew_events_df,
            detumble_events_df,
        ) = self._finalise_event_logs(
            final_time_sec,
            self.start_date.shiftedBy(final_time_sec),
        )

        telemetry_df.to_csv(results_path("master", "ANTARA_Master_telemetry.csv"), index=False)
        saa_passes_df.to_csv(results_path("master", "ANTARA_Master_saa_passes.csv"), index=False)
        gs_contacts_df.to_csv(results_path("master", "ANTARA_Master_gs_contacts.csv"), index=False)
        mode_transitions_df.to_csv(results_path("master", "ANTARA_Master_mode_transitions.csv"), index=False)
        slew_events_df.to_csv(results_path("master", "ANTARA_Master_slew_events.csv"), index=False)
        detumble_events_df.to_csv(results_path("master", "ANTARA_Master_detumble_events.csv"), index=False)

        self._plot_results(telemetry_df, saa_passes_df, gs_contacts_df, slew_events_df)
        self._print_summary(
            telemetry_df,
            saa_passes_df,
            gs_contacts_df,
            mode_transitions_df,
            slew_events_df,
        )


if __name__ == "__main__":
    ANTARASimulator().run()
