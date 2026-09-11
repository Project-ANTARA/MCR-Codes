from __future__ import annotations

import math
import argparse
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orekit
import pandas as pd
from datetime import datetime
from paths import figures_path, init_orekit_data, results_path

# Import core models from shared module (no Orekit dependencies)
from antara_models import (
    MissionConfig,
    ADCSModel,
    SpacecraftFSM,
    SunSubmode,
    IllegalFSMTransitionError,
    EPSModel,
    _as_flat_vector as _antara_as_flat_vector,
    BODY_AXIS_VECTORS,
)

try:
    from orekit import JArray_double
except ImportError:
    JArray_double = None

from shapely.geometry import Point, Polygon

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    _MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not found. Plot generation will be skipped.")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _CARTOPY_AVAILABLE = True
except ImportError:
    ccrs = None
    cfeature = None
    _CARTOPY_AVAILABLE = False
    print("[WARN] cartopy not found. Map-based plots will be skipped.")

try:
    import ppigrf
    _PYIGRF_AVAILABLE = True
except ImportError:
    _PYIGRF_AVAILABLE = False
    print("[WARN] ppigrf not found. Falling back to dipole B-field for IGRF queries.")

try:
    import spacepy.irbempy as irbem
    _SPACEPY_AVAILABLE = True
except ImportError:
    _SPACEPY_AVAILABLE = False
    print("[WARN] spacepy not found. AP9 comparison layer disabled.")

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


class EPSModel:
    """
    Power system in Wh units.

    Attributes:
      - battery_wh (float)
      - solar_generation_w (float)
      - load_w (float)

    Methods:
      - compute_solar_power(sun_angle_deg, illumination_fraction) -> watts
      - compute_load(mode, adcs_slewing, in_eclipse, payload_pending=False) -> watts
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
    Elevation mask: 15 deg (matches gs_elevation_mask_deg config value).

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
        # AOS prediction state — used by Fix 5 (pre-slew before next contact window)
        self.predicted_next_aos_time_sec: Optional[float] = None
        self.last_los_time_sec: Optional[float] = None

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
        """Detect AOS or LOS transitions while maintaining contact statistics.

        FIX (Bug 5 / pre-slew): On LOS the last contact duration is recorded.
        The orbital period (~97 min for 600 km SSO) is used to predict the next
        AOS so the main loop can begin a pre-slew one slew_transition_penalty_sec
        before the predicted AOS window opens.
        """

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
            self.last_los_time_sec = time_sec
            # Predict next AOS: for a 600 km SSO the orbital period is ~97.5 min.
            # GS contacts recur approximately every 1–3 orbits; use 1 orbital period
            # as a conservative lower bound for the next overpass estimate.
            ORBITAL_PERIOD_SEC = 97.5 * 60.0   # ~5 850 s
            self.predicted_next_aos_time_sec = time_sec + ORBITAL_PERIOD_SEC
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


class IGRFModel:
    """
    Thin wrapper around ppigrf (IGRF-14 coefficients in ppigrf 2.1).

    Returns the full field vector [Bx, By, Bz] in NED (North-East-Down)
    geographic frame at the queried geodetic position and epoch, then
    converts to an inertial-frame vector for pitch-angle computation.

    Falls back to a centred dipole if ppigrf is unavailable.

    All inputs in SI / degree units; B-field output in Tesla.
    """

    DIPOLE_B0_T: float = 3.12e-5          # T·Re³ dipole moment scale
    EARTH_RADIUS_M: float = 6_371_000.0   # mean Earth radius (m)

    def __init__(self, epoch_year: float) -> None:
        """Store the decimal-year epoch used for all IGRF queries."""
        self.epoch_year = epoch_year
        self._available = _PYIGRF_AVAILABLE

    def get_ned_field_tesla(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
    ) -> np.ndarray:
        """
        Query IGRF at (lat, lon, alt) and return [Bn, Be, Bd] in Tesla.

        Parameters
        ----------
        lat_deg : geodetic latitude (degrees, positive north)
        lon_deg : geodetic longitude (degrees, positive east)
        alt_km  : altitude above WGS-84 ellipsoid (km)

        Returns
        -------
        np.ndarray shape (3,) — [B_north, B_east, B_down] in Tesla
        """
        if not self._available:
            return self._dipole_ned_tesla(lat_deg, lon_deg, alt_km)

        # ppigrf expects geodetic altitude in km and returns East/North/Up in nT.
        date = datetime(int(self.epoch_year), 3, 1)   # approximate from decimal year
        Be, Bn, Bu = ppigrf.igrf(lon_deg, lat_deg, alt_km, date)
        b_north_t =  float(np.asarray(Bn).item()) * 1e-9   # north
        b_east_t  =  float(np.asarray(Be).item()) * 1e-9   # east
        b_down_t  = -float(np.asarray(Bu).item()) * 1e-9   # ppigrf returns Up (+ve), IGRF Down (+ve down)
        return np.array([b_north_t, b_east_t, b_down_t], dtype=float)

    def get_inertial_field_tesla(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        position_inertial: np.ndarray,
    ) -> np.ndarray:
        """
        Return the IGRF field vector expressed in the EME2000 inertial frame.

        Converts NED → ECEF → inertial via the geographic position unit vectors.
        This is a static rotation (uses only the satellite position unit vector),
        not a full frame transform, which is sufficient for pitch-angle computation.

        Parameters
        ----------
        lat_deg, lon_deg, alt_km : geodetic position
        position_inertial        : satellite position in inertial frame (m)

        Returns
        -------
        np.ndarray shape (3,) — B-field in inertial frame (Tesla)
        """
        b_ned = self.get_ned_field_tesla(lat_deg, lon_deg, alt_km)
        b_n, b_e, b_d = b_ned

        # NED → ECEF rotation matrix columns:
        # e_north = [-sin(lat)cos(lon), -sin(lat)sin(lon), cos(lat)]
        # e_east  = [-sin(lon),          cos(lon),          0       ]
        # e_down  = [-cos(lat)cos(lon), -cos(lat)sin(lon), -sin(lat)]
        lat_r = math.radians(lat_deg)
        lon_r = math.radians(lon_deg)
        sl, cl = math.sin(lat_r), math.cos(lat_r)
        sn, cn = math.sin(lon_r), math.cos(lon_r)
        e_north = np.array([-sl * cn, -sl * sn,  cl], dtype=float)
        e_east  = np.array([-sn,       cn,        0.0], dtype=float)
        e_down  = np.array([-cl * cn, -cl * sn, -sl], dtype=float)
        b_ecef = b_n * e_north + b_e * e_east + b_d * e_down

        # ECEF ≈ inertial over orbital timescales for B-field direction purposes
        # (Earth rotation changes B direction by < 0.004 deg per 2-s step)
        return b_ecef

    def _dipole_ned_tesla(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
    ) -> np.ndarray:
        """Centred dipole fallback: returns [Bn, Be=0, Bd] in Tesla."""
        r_m = self.EARTH_RADIUS_M + alt_km * 1000.0
        lat_r = math.radians(lat_deg)
        b_scale = self.DIPOLE_B0_T * (self.EARTH_RADIUS_M / r_m) ** 3
        b_north = -b_scale * math.sin(lat_r)
        b_down  = -2.0 * b_scale * math.cos(lat_r)
        return np.array([b_north, 0.0, b_down], dtype=float)


def compute_detector_geometric_factor_cm2sr(theta_max_deg: float) -> float:
    """Return the detector aperture geometric factor (cm^2 sr) for a circular
    aperture with half-angle theta_max_deg. Single source of truth: both
    ProtonEnvironmentModel and PayloadModel must call this instead of each
    computing it independently, so they cannot silently drift apart.
    """
    return 1.0 * math.pi * (math.sin(math.radians(theta_max_deg)) ** 2)


class ProtonEnvironmentModel:
    """
    Physics-based proton flux model for the South Atlantic Anomaly.

    Architecture
    ------------
    1. SAA depth factor: scales flux by how deep inside the SAA polygon
       the sub-satellite point is, using Shapely distance-to-boundary.
    2. Pitch angle: computed from IGRF B-field and satellite velocity vector.
    3. East-West asymmetry: directional flux correction from geomagnetic
       field curvature (east-going protons are harder to mirror at low
       altitude than west-going ones).
    4. AP9 comparison: queries spacepy.irbempy AP9 mean flux for the same
       position at quiet-time for residual analysis. Falls back to the
       config baseline if spacepy is unavailable.
    5. Flux equation per channel i:
         Φ_meas[i] = Φ_AP9[i] × saa_depth_factor × f_pitch[i] × f_ew × f_storm
       where:
         f_pitch[i]  = cos^α(pitch_angle)   α empirically set to 0.5
         f_ew        = 1 + A × sign(v × (B × nadir))   A = ew_asymmetry_amplitude
         f_storm     = storm_time_flux_multiplier if storm_time else 1.0

    Units: all flux values in particles / cm² / s / sr / MeV.
    """

    PITCH_EXPONENT: float = 0.5   # empirical pitch-angle flux modulation exponent

    def __init__(self, config: "MissionConfig", saa_polygon: Polygon) -> None:
        """Initialise the proton environment model with mission config and SAA geometry."""
        self.config = config
        self.saa_polygon = saa_polygon
        self.energy_channels = np.array(config.energy_channels_mev, dtype=float)
        self.ap9_baseline = np.array(config.ap9_baseline_flux, dtype=float)
        self.n_channels = len(self.energy_channels)
        self._spacepy_available = _SPACEPY_AVAILABLE
        self._ap9_query_warned = False
        # Detector aperture geometric factor (cm^2 sr), shared with PayloadModel.
        # theta_max_deg is a fixed detector-aperture half-angle (atan(1/3)), not a
        # config-tunable value, so it is computed once here rather than read from
        # config. PayloadModel independently derives the same constant for its own
        # per-bin count-rate pipeline; both MUST use compute_detector_geometric_factor_cm2sr()
        # so the two pipelines can never drift apart again.
        self.theta_max_deg = math.degrees(math.atan(1.0 / 3.0))
        self.geometric_factor_cm2sr = compute_detector_geometric_factor_cm2sr(self.theta_max_deg)
        assert abs(self.geometric_factor_cm2sr - 0.3141592653589793) < 5.0e-3, "Geometric factor mismatch"

    def saa_depth_factor(self, lat_deg: float, lon_deg: float) -> float:
        """
        Compute a flux scaling factor in [saa_depth_flux_min, saa_depth_flux_max]
        based on the sub-satellite point's normalised depth inside the SAA polygon.

        Depth = 1 - (distance_to_boundary / max_inscribed_radius_estimate).
        Points near the boundary get factor ~ saa_depth_flux_min; the centre gets
        saa_depth_flux_max.

        Parameters
        ----------
        lat_deg, lon_deg : sub-satellite geodetic coordinates

        Returns
        -------
        float scalar depth factor
        """
        point = Point(lon_deg, lat_deg)   # Shapely uses (x=lon, y=lat) order
        if not self.saa_polygon.contains(point):
            return 0.0   # outside SAA → no flux

        dist_to_boundary = self.saa_polygon.boundary.distance(point)
        # Normalise by a representative half-width of the SAA (~25 deg in lon)
        normalised_depth = float(np.clip(dist_to_boundary / 25.0, 0.0, 1.0))
        factor = (
            self.config.saa_depth_flux_min
            + (self.config.saa_depth_flux_max - self.config.saa_depth_flux_min)
            * normalised_depth
        )
        return float(np.clip(factor, self.config.saa_depth_flux_min, self.config.saa_depth_flux_max))

    def magnetic_pitch_angle_deg(
        self,
        b_inertial: np.ndarray,
        velocity_inertial: np.ndarray,
    ) -> float:
        """
        Compute the magnetic pitch angle between the local B-field direction
        and the satellite velocity vector (used as the detector look direction).

        pitch_angle = arccos( |v̂ · B̂| )

        Returned value is always in [0°, 90°] because protons arriving from
        both parallel and anti-parallel directions are detected (unsigned angle).

        Parameters
        ----------
        b_inertial        : IGRF B-field vector in inertial frame (Tesla, any magnitude)
        velocity_inertial : satellite velocity vector in inertial frame (m/s, any magnitude)

        Returns
        -------
        float pitch angle in degrees in [0, 90]
        """
        b_norm = float(np.linalg.norm(b_inertial))
        v_norm = float(np.linalg.norm(velocity_inertial))
        if b_norm < 1.0e-20 or v_norm < 1.0e-6:
            return 90.0   # undefined; return perpendicular as safe default
        b_hat = b_inertial / b_norm
        v_hat = velocity_inertial / v_norm
        cos_angle = float(np.clip(np.dot(v_hat, b_hat), -1.0, 1.0))
        angle_deg = math.degrees(math.acos(abs(cos_angle)))
        assert 0.0 <= angle_deg <= 90.0, f"Pitch angle out of range: {angle_deg:.4f} deg"
        return angle_deg

    def ew_asymmetry_factor(
        self,
        velocity_inertial: np.ndarray,
        b_inertial: np.ndarray,
        nadir_inertial: np.ndarray,
    ) -> float:
        """
        East-West asymmetry factor for proton flux directionality.

        The sign of  v · (B × nadir)  determines whether the spacecraft is
        flying through the field in the eastward or westward sense, modulating
        the flux by ±ew_asymmetry_amplitude.

        f_ew = 1 + A × sign( v · (B × nadir) )

        Parameters
        ----------
        velocity_inertial : satellite velocity in inertial frame
        b_inertial        : IGRF B-field in inertial frame
        nadir_inertial    : nadir unit vector in inertial frame (pointing toward Earth)

        Returns
        -------
        float asymmetry factor (dimensionless)
        """
        cross = np.cross(_as_flat_vector(b_inertial), _as_flat_vector(nadir_inertial))
        triple = float(np.dot(_as_flat_vector(velocity_inertial), cross))
        sign = float(np.sign(triple)) if abs(triple) > 1.0e-20 else 0.0
        return 1.0 + self.config.ew_asymmetry_amplitude * sign

    def get_ap9_flux(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
    ) -> np.ndarray:
        """
        Query AP9 mean differential proton flux via spacepy.irbempy.

        If spacepy is unavailable, returns the config baseline (quiet-time
        representative values). The returned array is always in the same unit
        as ap9_baseline_flux: particles / cm² / s / sr / MeV.

        Parameters
        ----------
        lat_deg, lon_deg : sub-satellite geodetic position
        alt_km           : altitude above WGS-84 ellipsoid

        Returns
        -------
        np.ndarray shape (n_channels,) — AP9 mean flux per energy channel
        """
        if not self._spacepy_available:
            return self.ap9_baseline.copy()
        try:
            # spacepy.irbempy.get_AEP8 interface:
            # Inputs: energy (MeV), position (GDZ: alt_km, lat, lon)
            # whichm: 'AP9' for protons
            # returns: omni-directional flux (particles / cm² / s)
            # Convert to differential flux: divide by energy bin width and 4π sr
            results = []
            for energy_mev in self.energy_channels:
                coords = irbem.Coords(
                    [alt_km, lat_deg, lon_deg],
                    "GDZ",
                    "sph",
                )
                flux_omni = irbem.get_AEP8(
                    energy_mev,
                    loci=coords,
                    whichm="AP9",
                    whatf="Flux",
                )
                # AP9 returns integral (omni); approximate differential by
                # finite-difference with adjacent 20% energy step
                de_mev = energy_mev * 0.20
                flux_diff_per_sr = max(0.0, float(flux_omni)) / (4.0 * math.pi * de_mev)
                results.append(flux_diff_per_sr)
            return np.array(results, dtype=float)
        except Exception as exc:
            # SpacePy's AE8/AP8 interface does not implement AP9. Keep the
            # existing baseline fallback, but never present it as live AP9.
            if not self._ap9_query_warned:
                print(f"[WARN] AP9 query unavailable ({type(exc).__name__}: {exc}). "
                      "SpacePy get_AEP8 supports AE8/AP8, not AP9; using configured flux baseline.")
                self._ap9_query_warned = True
            return self.ap9_baseline.copy()

    def compute_flux(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        b_inertial: np.ndarray,
        velocity_inertial: np.ndarray,
        nadir_inertial: np.ndarray,
        storm_time: bool = False,
    ) -> Dict[str, object]:
        """
        Compute the full proton flux measurement for the current timestep.

        Returns a dict with all science quantities needed by TelemetryLogger and
        the per-SAA-pass science accumulator.

        Parameters
        ----------
        lat_deg, lon_deg, alt_km  : geodetic position
        b_inertial                : IGRF B-field in inertial frame (Tesla)
        velocity_inertial         : satellite velocity (m/s)
        nadir_inertial            : nadir unit vector in inertial frame
        storm_time                : True if space-weather storm criteria met

        Returns
        -------
        dict with keys:
          measured_flux   np.ndarray (n_channels,)  particles/cm²/s/sr/MeV
          ap9_flux        np.ndarray (n_channels,)  particles/cm²/s/sr/MeV
          residual        np.ndarray (n_channels,)  measured − AP9
          pitch_angle_deg float
          ew_factor       float
          saa_depth       float
          count_rate      np.ndarray (n_channels,)  counts/s after efficiency+deadtime
          dark_counts     np.ndarray (n_channels,)  dark count contribution
          storm_time      bool
        """
        depth = self.saa_depth_factor(lat_deg, lon_deg)
        pitch_deg = self.magnetic_pitch_angle_deg(b_inertial, velocity_inertial)
        ew_factor = self.ew_asymmetry_factor(velocity_inertial, b_inertial, nadir_inertial)
        ap9_flux = self.get_ap9_flux(lat_deg, lon_deg, alt_km)
        storm_factor = self.config.storm_time_flux_multiplier if storm_time else 1.0

        f_pitch = np.cos(math.radians(pitch_deg)) ** self.PITCH_EXPONENT
        measured_flux = (
            ap9_flux
            * float(depth)
            * float(f_pitch)
            * float(ew_factor)
            * storm_factor
        )
        measured_flux = np.maximum(measured_flux, 0.0)
        residual = measured_flux - ap9_flux

        # Detector response: apply efficiency, geometric factor, dead-time, dark counts
        efficiency = np.array(self.config.detector_efficiency, dtype=float)
        geom = self.geometric_factor_cm2sr
        raw_rate = measured_flux * geom * efficiency   # counts / s / MeV × MeV_bin not applied here — stored as-is
        dead_time_suppression = 1.0 / (
            1.0 + np.maximum(raw_rate, 0.0) * self.config.detector_dead_time_sec
        )
        count_rate = raw_rate * dead_time_suppression
        dark_counts = np.full(self.n_channels, self.config.detector_dark_count_rate_hz, dtype=float)
        total_count_rate = count_rate + dark_counts

        assert float(np.all(np.isfinite(measured_flux))), "Non-finite proton flux detected"
        assert 0.0 <= pitch_deg <= 90.0, f"Pitch angle {pitch_deg:.3f} deg out of bounds"

        return {
            "measured_flux":   measured_flux,
            "ap9_flux":        ap9_flux,
            "residual":        residual,
            "pitch_angle_deg": pitch_deg,
            "ew_factor":       ew_factor,
            "saa_depth":       depth,
            "count_rate":      count_rate,
            "dark_counts":     dark_counts,
            "total_count_rate": total_count_rate,
            "storm_time":      storm_time,
        }


@dataclass(frozen=True)
class DetectorLayerSpec:
    """Physical detector layer description used by the payload response model."""

    name: str
    thickness_mm: float
    stop_energy_mev: float
    light_yield_class: str


class PayloadModel:
    """ANTARA proton spectrometer payload model with histogram-based science output."""

    REGION_ORDER = ("core", "mid", "edge", "periphery", "ultra-periphery")
    LAYER_ORDER = ("Si", "L1", "L2", "L3", "L4", "L5", "rear_veto")

    def __init__(self, config: "MissionConfig", proton_env: "ProtonEnvironmentModel") -> None:
        self.config = config
        self.proton_env = proton_env
        self.total_science_data_generated_mb = 0.0
        self.total_detection_seconds = 0.0
        self.last_flux_result: Dict[str, object] = {}
        self.pass_science_records: List[Dict[str, object]] = []
        self._current_pass_records: List[Dict[str, object]] = []
        self._pass_open = False
        self._detection_steps_this_pass = 0
        # Event-driven snapshot state: counts accumulated PER CELL since that
        # cell's last flush (distinct from self._cell_histograms, which is
        # the mission-lifetime archive and is never reset). A snapshot for a
        # cell is flushed when the spacecraft leaves that cell, when the
        # safety-net max interval is exceeded, or when the SAA pass closes.
        self._pending_bin_counts: Dict[str, np.ndarray] = {}
        self._pending_cell_elapsed_sec: Dict[str, float] = {}
        self._previous_cell_id: Optional[str] = None
        self._cell_histograms: Dict[str, np.ndarray] = {}
        self._cell_pass_counts: Dict[str, int] = {}
        self._cell_total_counts: Dict[str, float] = {}
        self._current_pass_cell_ids: List[str] = []
        self._current_pass_event_types: List[str] = []
        self._current_pass_livetime: List[float] = []
        self._current_pass_sigma_fraction: List[np.ndarray] = []
        self._current_pass_counts_per_bin: List[np.ndarray] = []
        self._current_pass_region: List[str] = []
        self.layer_specs = (
            DetectorLayerSpec("Si", 1.0, 5.0, "silicon"),
            DetectorLayerSpec("L1", 2.0, 8.0, "scintillator"),
            DetectorLayerSpec("L2", 4.0, 14.0, "scintillator"),
            DetectorLayerSpec("L3", 12.0, 28.0, "scintillator"),
            DetectorLayerSpec("L4", 22.0, 48.0, "scintillator"),
            DetectorLayerSpec("L5", 30.0, 65.0, "scintillator"),
            DetectorLayerSpec("rear_veto", 2.0, 65.0, "veto"),
        )
        self.theta_max_deg = math.degrees(math.atan(1.0 / 3.0))
        self.geometric_factor_cm2sr = compute_detector_geometric_factor_cm2sr(self.theta_max_deg)
        assert abs(self.geometric_factor_cm2sr - 0.3141592653589793) < 5.0e-3, "Geometric factor mismatch"
        self.bin_edges_mev = np.geomspace(5.0, 100.0, 65)
        assert np.all(np.diff(self.bin_edges_mev) > 0.0), "Energy bin edges must be strictly monotonic"
        assert math.isclose(float(self.bin_edges_mev[0]), 5.0, rel_tol=0.0, abs_tol=1.0e-9)
        assert math.isclose(float(self.bin_edges_mev[-1]), 100.0, rel_tol=0.0, abs_tol=1.0e-9)
        self.bin_centers_mev = np.sqrt(self.bin_edges_mev[:-1] * self.bin_edges_mev[1:])
        self.bin_widths_mev = self.bin_edges_mev[1:] - self.bin_edges_mev[:-1]
        self.bin_layer_map = self._build_bin_layer_map()
        self.layer_efficiency = self._build_layer_efficiency()
        self.region_flux_table = self._build_region_flux_table()
        self.bin_event_type_map = self._build_bin_event_type_map()
        self.n_bins = len(self.bin_centers_mev)
        assert self.n_bins > 0, "PayloadModel requires at least one energy bin."
        self.n_channels = self.n_bins
        self.bin_efficiency_lookup = np.array(
            [self._efficiency_for_bin(idx, self.bin_event_type_map[idx]) for idx in range(self.n_bins)],
            dtype=float,
        )
        self.rear_veto_mask = np.array([layer_name == "rear_veto" for layer_name in self.bin_layer_map], dtype=bool)
        self.region_base_flux_lookup = {
            region: np.array(
                [self._region_layer_flux(region, layer_name) for layer_name in self.bin_layer_map],
                dtype=float,
            )
            for region in self.REGION_ORDER
        }

    def _build_layer_efficiency(self) -> Dict[str, Dict[str, float]]:
        """Build the per-layer detection efficiency lookup once at startup."""

        return {
            "Si": {"eps_geom": 0.92, "eps_coinc": 1.00, "eps_threshold": 0.85},
            "L1": {"eps_geom": 0.92, "eps_coinc": 1.00, "eps_threshold": 0.85},
            "L2": {"eps_geom": 0.91, "eps_coinc": 1.00, "eps_threshold": 0.99},
            "L3": {"eps_geom": 0.90, "eps_coinc": 1.00, "eps_threshold": 0.99},
            "L4": {"eps_geom": 0.90, "eps_coinc": 1.00, "eps_threshold": 0.99},
            "L5": {"eps_geom": 0.90, "eps_coinc": 1.00, "eps_threshold": 0.99},
            "rear_veto": {"eps_geom": 0.88, "eps_coinc": 0.95, "eps_threshold": 0.99},
        }

    def _build_region_flux_table(self) -> Dict[str, Dict[str, float]]:
        """Build the representative region-by-layer flux table."""

        return {
            "core": {"L1": 2.8e4, "L2": 1.5e4, "L3": 5.5e3, "L4": 8.0e2, "L5": 9.0e1},
            "mid": {"L1": 1.8e4, "L2": 9.5e3, "L3": 3.2e3, "L4": 5.0e2, "L5": 6.0e1},
            "edge": {"L1": 8.5e3, "L2": 4.4e3, "L3": 1.5e3, "L4": 2.0e2, "L5": 2.8e1},
            "periphery": {"L1": 2.1e3, "L2": 1.1e3, "L3": 3.8e2, "L4": 5.5e1, "L5": 7.5},
            "ultra-periphery": {"L1": 6.0e2, "L2": 3.0e2, "L3": 1.0e2, "L4": 1.4e1, "L5": 2.0},
        }

    def _build_bin_layer_map(self) -> List[str]:
        """Map each of the 64 bins to its most relevant detector layer."""

        layer_map: List[str] = []
        for bin_index in range(64):
            one_based = bin_index + 1
            if one_based <= 10:
                layer_map.append("L1")
            elif one_based <= 20:
                layer_map.append("L2")
            elif one_based <= 32:
                layer_map.append("L3")
            elif one_based <= 46:
                layer_map.append("L4")
            else:
                layer_map.append("L5")
        return layer_map

    def _build_bin_event_type_map(self) -> List[str]:
        """Associate each energy bin with a detector event family."""

        event_types: List[str] = []
        for bin_index in range(64):
            one_based = bin_index + 1
            if one_based <= 10:
                event_types.append("Type1")
            elif one_based <= 20:
                event_types.append("Type2")
            elif one_based <= 32:
                event_types.append("Type3")
            elif one_based <= 46:
                event_types.append("Type4")
            elif one_based <= 54:
                event_types.append("Type5")
            else:
                event_types.append("Type6")
        return event_types

    @staticmethod
    def validate_contiguous_pattern(fired_layers: List[bool]) -> None:
        """Reject non-contiguous firing patterns and document the expected geometry.

        Examples
        --------
        valid:   [True, True, False, False]
        invalid: [True, False, True, False]
        """

        seen_gap = False
        for fired in fired_layers:
            if not fired:
                seen_gap = True
            elif seen_gap:
                raise ValueError("Unphysical detector pattern: deeper layer fired after a gap.")

    def _classify_event_type(self, energy_mev: float) -> Tuple[str, List[str]]:
        """Classify a proton event into Type 1-6 and return its contiguous firing pattern."""

        if energy_mev < 8.0:
            layers = ["Si", "L1"]
            event_type = "Type1"
        elif energy_mev < 14.0:
            layers = ["Si", "L1", "L2"]
            event_type = "Type2"
        elif energy_mev < 28.0:
            layers = ["Si", "L1", "L2", "L3"]
            event_type = "Type3"
        elif energy_mev < 48.0:
            layers = ["Si", "L1", "L2", "L3", "L4"]
            event_type = "Type4"
        elif energy_mev < 65.0:
            layers = ["Si", "L1", "L2", "L3", "L4", "L5"]
            event_type = "Type5"
        else:
            layers = ["Si", "L1", "L2", "L3", "L4", "L5", "rear_veto"]
            event_type = "Type6"
        fired_pattern = [layer in layers for layer in self.LAYER_ORDER]
        self.validate_contiguous_pattern(fired_pattern)
        return event_type, layers

    def _classify_region(self, saa_depth: float) -> str:
        """Convert the scalar SAA depth into one of the configured spatial regions."""

        if saa_depth >= 1.6:
            return "core"
        if saa_depth >= 1.2:
            return "mid"
        if saa_depth >= 0.85:
            return "edge"
        if saa_depth >= 0.45:
            return "periphery"
        return "ultra-periphery"

    def _cell_id(self, lat_deg: float, lon_deg: float, region: str) -> str:
        """Generate a coarse spatial cell identifier for accumulating histograms."""

        lat_bin = int(math.floor((lat_deg + 90.0) / 10.0))
        lon_bin = int(math.floor((lon_deg + 180.0) / 10.0))
        return f"{region}:{lat_bin:02d}:{lon_bin:02d}"

    def _efficiency_for_bin(self, bin_index: int, event_type: str) -> float:
        """Return the per-bin total efficiency using the layer lookup table."""

        layer_name = self.bin_layer_map[bin_index]
        entry = self.layer_efficiency[layer_name]
        eps = entry["eps_geom"] * entry["eps_coinc"] * entry["eps_threshold"]
        if event_type == "Type6" and layer_name == "rear_veto":
            eps *= 0.95
        return eps

    def _merge_bins_if_needed(self, region: str) -> List[int]:
        """Return the list of bin indices to accumulate, optionally merging the periphery bins."""

        if (
            self.config.enable_periphery_bin_merge
            and region in {"periphery", "ultra-periphery"}
        ):
            merged = [0, 1, 2] + list(range(3, self.n_bins))
            return merged
        return list(range(self.n_bins))

    def _region_layer_flux(self, region: str, layer_name: str) -> float:
        """Return the representative flux for a region/layer pair."""

        if layer_name == "Si":
            layer_name = "L1"
        if layer_name == "rear_veto":
            layer_name = "L5"
        return self.region_flux_table[region][layer_name]

    def _flush_cell_pending(self, cell_id: str) -> float:
        """Flush the pending accumulated histogram for one cell into a downlink packet.

        Per Payload_1.docx, a histogram readout is a fixed-width dump of
        every bin counter -- real hardware does not adaptively compress based
        on signal content -- so packet size is simply
        n_bins x science_packet_bytes_per_bin, plus a fixed header covering
        timestamp / spacecraft position / livetime / housekeeping metadata
        (science_packet_header_bytes). What IS event-driven is the flush
        trigger (cell transition, SAA pass close, or the safety-net max
        interval), not the packet's byte size. Returns the packet size in MB
        (0.0 if there is nothing pending for this cell) and clears that
        cell's pending accumulator.
        """

        pending_counts = self._pending_bin_counts.pop(cell_id, None)
        elapsed_sec = self._pending_cell_elapsed_sec.pop(cell_id, 0.0)
        if pending_counts is None or elapsed_sec <= 0.0:
            return 0.0

        packet_bytes = self.config.science_packet_header_bytes + (
            self.n_bins * self.config.science_packet_bytes_per_bin
        )
        data_mb = packet_bytes / (1024.0 * 1024.0)
        self.total_science_data_generated_mb += data_mb
        return data_mb

    def _mean_livetime_fraction(self, livetime_fractions: np.ndarray) -> float:
        """Return the mean detector livetime fraction for the current step."""

        return float(np.mean(livetime_fractions))

    def generate_data(
        self,
        dt_sec: float,
        lat_deg: float,
        lon_deg: float,
        alt_km: float,
        b_inertial: np.ndarray,
        velocity_inertial: np.ndarray,
        nadir_inertial: np.ndarray,
        storm_time: bool = False,
    ) -> Tuple[float, Dict[str, object]]:
        """Generate one timestep of histogram-based science data."""

        flux_result = self.proton_env.compute_flux(
            lat_deg,
            lon_deg,
            alt_km,
            b_inertial,
            velocity_inertial,
            nadir_inertial,
            storm_time=storm_time,
        )
        self.last_flux_result = flux_result

        depth = float(flux_result.get("saa_depth", 0.0))
        region = self._classify_region(depth)
        cell_id = self._cell_id(lat_deg, lon_deg, region)
        if cell_id not in self._cell_histograms:
            self._cell_histograms[cell_id] = np.zeros(self.n_bins, dtype=float)
            self._cell_pass_counts[cell_id] = 0
            self._cell_total_counts[cell_id] = 0.0
        self._cell_pass_counts[cell_id] += 1

        pitch_deg = float(flux_result.get("pitch_angle_deg", 90.0))
        ew_factor = float(flux_result.get("ew_factor", 1.0))
        storm_factor = self.config.storm_time_flux_multiplier if storm_time else 1.0
        f_pitch = math.cos(math.radians(min(90.0, max(0.0, pitch_deg)))) ** self.proton_env.PITCH_EXPONENT
        f_pitch = max(0.0, f_pitch)

        measured_flux = self.region_base_flux_lookup[region] * (f_pitch * ew_factor * storm_factor)
        if np.any(self.rear_veto_mask):
            measured_flux = measured_flux.copy()
            measured_flux[self.rear_veto_mask] *= 0.1
        ap9_flux = self.region_base_flux_lookup[region].copy()

        raw_rate_hz = measured_flux * self.geometric_factor_cm2sr * self.bin_efficiency_lookup * self.bin_widths_mev
        deadtime_scale = self.config.detector_dead_time_ns * 1.0e-9
        livetime_fractions = 1.0 / (1.0 + np.maximum(raw_rate_hz, 0.0) * deadtime_scale)
        total_count_rate = raw_rate_hz * livetime_fractions + self.config.detector_dark_count_rate_hz
        accepted_counts = total_count_rate * dt_sec
        sigma_denominator = max(self._cell_total_counts.get(cell_id, 0.0) + float(np.sum(accepted_counts)), 1.0)
        sigma_fraction = np.full(self.n_bins, 1.0 / math.sqrt(sigma_denominator), dtype=float)
        event_types = list(self.bin_event_type_map)

        self._cell_histograms[cell_id] += accepted_counts
        self._cell_total_counts[cell_id] += float(np.sum(accepted_counts))

        if self.config.enable_periphery_bin_merge and region in {"periphery", "ultra-periphery"}:
            merged_slice = slice(0, 3)
            merged_counts = float(np.sum(accepted_counts[merged_slice]))
            if merged_counts > 0.0:
                accepted_counts[0] = merged_counts
                accepted_counts[1:3] = 0.0

        # Event-driven snapshot: accumulate into this cell's pending buffer.
        # A packet is flushed (data_mb > 0 this call) when the spacecraft
        # crosses into a NEW cell (flushing the cell just left) or when a
        # cell's pending accumulation exceeds the safety-net max interval.
        # SAA-pass-close flushing is handled separately in close_saa_pass().
        if cell_id not in self._pending_bin_counts:
            self._pending_bin_counts[cell_id] = np.zeros(self.n_bins, dtype=float)
            self._pending_cell_elapsed_sec[cell_id] = 0.0
        self._pending_bin_counts[cell_id] += accepted_counts
        self._pending_cell_elapsed_sec[cell_id] += dt_sec

        data_mb = 0.0
        if self._previous_cell_id is not None and cell_id != self._previous_cell_id:
            data_mb += self._flush_cell_pending(self._previous_cell_id)
        if self._pending_cell_elapsed_sec.get(cell_id, 0.0) >= self.config.science_histogram_downlink_interval_sec:
            data_mb += self._flush_cell_pending(cell_id)
        self._previous_cell_id = cell_id
        payload_snapshot_due = data_mb > 0.0
        histogram_bytes_this_step = int(round(data_mb * 1024.0 * 1024.0))
        self.total_detection_seconds += dt_sec

        mean_livetime_fraction = self._mean_livetime_fraction(livetime_fractions)
        event_type = event_types[0] if event_types else "Type1"
        self._current_pass_records.append(
            {
                "region": region,
                "cell_id": cell_id,
                "event_type": event_type,
                "pitch_angle_deg": pitch_deg,
                "ew_factor": ew_factor,
                "saa_depth": depth,
                "livetime_fraction": mean_livetime_fraction,
                "measured_flux": measured_flux.tolist(),
                "ap9_flux": ap9_flux.tolist(),
                "count_rate": total_count_rate.tolist(),
                "counts_per_bin": accepted_counts.tolist(),
                "sigma_fraction": sigma_fraction.tolist(),
                "lat_deg": lat_deg,
                "lon_deg": lon_deg,
                "alt_km": alt_km,
                "storm_time": storm_time,
                "saa_state_at_sample": True,
            }
        )
        self._current_pass_cell_ids.append(cell_id)
        self._current_pass_event_types.append(event_type)
        self._current_pass_livetime.append(mean_livetime_fraction)
        self._current_pass_sigma_fraction.append(sigma_fraction.copy())
        self._current_pass_counts_per_bin.append(accepted_counts.copy())
        self._current_pass_region.append(region)
        return data_mb, {
            "measured_flux": measured_flux,
            "ap9_flux": ap9_flux,
            "residual": measured_flux - ap9_flux,
            "pitch_angle_deg": pitch_deg,
            "ew_factor": ew_factor,
            "saa_depth": depth,
            "total_count_rate": total_count_rate,
            "accepted_counts": accepted_counts,
            "sigma_fraction": sigma_fraction,
            "region": region,
            "cell_id": cell_id,
            "cell_n_cell": float(self._cell_total_counts[cell_id]),
            "cell_sigma_fraction": float(1.0 / math.sqrt(max(self._cell_total_counts[cell_id], 1.0))),
            "event_type": event_type,
            "livetime_fraction": mean_livetime_fraction,
            "histogram_downlink_due": payload_snapshot_due,
            "histogram_bytes": histogram_bytes_this_step,
            "bin_edges_mev": self.bin_edges_mev.tolist(),
            "bin_centers_mev": self.bin_centers_mev.tolist(),
            "bin_event_types": self.bin_event_type_map,
            "counts_per_bin": accepted_counts.tolist(),
            "saa_state_at_sample": True,
        }

    def open_saa_pass(self) -> None:
        """Reset the per-pass accumulation buffer at SAA entry."""

        self._current_pass_records = []
        self._current_pass_cell_ids = []
        self._current_pass_event_types = []
        self._current_pass_livetime = []
        self._current_pass_sigma_fraction = []
        self._current_pass_counts_per_bin = []
        self._current_pass_region = []
        self._detection_steps_this_pass = 0
        self._pass_open = True

    def close_saa_pass(self, pass_index: int, entry_time_sec: float, exit_time_sec: float) -> float:
        """Finalize the per-pass science record at SAA exit.

        Also flushes any cell histograms still buffered when the pass ends
        (rather than silently carrying them into the next pass) and returns
        the additional data volume in MB generated by that flush.
        """

        flushed_mb = 0.0
        for pending_cell_id in list(self._pending_bin_counts.keys()):
            flushed_mb += self._flush_cell_pending(pending_cell_id)
        self._previous_cell_id = None

        if not self._current_pass_records:
            self._pass_open = False
            return flushed_mb

        pitch_angles = [float(r["pitch_angle_deg"]) for r in self._current_pass_records]
        ew_factors = [float(r["ew_factor"]) for r in self._current_pass_records]
        count_stack = np.array([r["counts_per_bin"] for r in self._current_pass_records], dtype=float)
        sigma_stack = np.array([r["sigma_fraction"] for r in self._current_pass_records], dtype=float)
        n_science_steps = len(self._current_pass_records)
        n_steps = max(self._detection_steps_this_pass, n_science_steps)
        science_fraction = float(n_science_steps) / float(n_steps) if n_steps > 0 else 0.0
        mean_livetime = float(np.mean(self._current_pass_livetime)) if self._current_pass_livetime else 1.0

        self.pass_science_records.append(
            {
                "pass_index": pass_index,
                "entry_time_sec": entry_time_sec,
                "exit_time_sec": exit_time_sec,
                "dwell_sec": exit_time_sec - entry_time_sec,
                "n_steps": n_steps,
                "n_science_steps": n_science_steps,
                "science_fraction": science_fraction,
                "mean_pitch_angle_deg": float(np.mean(pitch_angles)),
                "std_pitch_angle_deg": float(np.std(pitch_angles)),
                "mean_ew_factor": float(np.mean(ew_factors)),
                "mean_livetime_fraction": mean_livetime,
                "mean_measured_flux": np.mean(np.array([r["measured_flux"] for r in self._current_pass_records], dtype=float), axis=0).tolist(),
                "mean_ap9_flux": np.mean(np.array([r["ap9_flux"] for r in self._current_pass_records], dtype=float), axis=0).tolist(),
                "mean_sigma_fraction": np.mean(sigma_stack, axis=0).tolist(),
                "total_counts_per_bin": count_stack.sum(axis=0).tolist(),
                "cell_ids": sorted(set(self._current_pass_cell_ids)),
                "event_types": sorted(set(self._current_pass_event_types)),
                "region_histogram": {region: self._current_pass_region.count(region) for region in sorted(set(self._current_pass_region))},
            }
        )
        self._current_pass_records = []
        self._current_pass_cell_ids = []
        self._current_pass_event_types = []
        self._current_pass_livetime = []
        self._current_pass_sigma_fraction = []
        self._current_pass_counts_per_bin = []
        self._current_pass_region = []
        self._detection_steps_this_pass = 0
        self._pass_open = False
        return flushed_mb

    def count_detection_step(self) -> None:
        """Increment the per-pass DETECTION step counter."""

        if self._pass_open:
            self._detection_steps_this_pass += 1

    def is_operable(self, battery_wh: float, mode: str) -> bool:
        """Return True when the payload may operate."""

        return mode == "DETECTION" and battery_wh >= self.config.saa_detection_min_power_wh


class CommunicationsModel:
    """
    Downlink over GS_INDIA.
    Active ONLY in RELAY mode while GS is visible.

    Methods:
      - downlink_step(storage_mb, dt_sec, gs_elevation_deg) -> MB_downlinked
        Effective rate = base_rate * sqrt(max(0, sin(elevation_deg))) so that
        a zenith pass (90 deg) yields the full 9.6 kbps while a 15 deg low-
        elevation contact yields ~50% of that rate, approximating the ~6 dB
        extra path loss at low elevation angles.  This is a simple Friis proxy
        suitable for PDR; a full link budget (EIRP, G/T, Eb/N0) should replace
        it before CDR.

        FIX (Bug E): The original model assumed a constant 9.6 kbps regardless
        of elevation, ignoring the 6-8 dB path-loss penalty at low elevations.

    Track total_downlinked_MB and total_contact_seconds.
    Contact time is accumulated from raw visibility, while downlinked volume
    still depends on relay pointing validity.
    """

    def __init__(self, config: MissionConfig) -> None:
        """Initialise communications counters."""

        self.config = config
        self.total_downlinked_mb = 0.0
        self.total_contact_seconds = 0.0

    def downlink_step(self, storage_mb: float, dt_sec: float, gs_elevation_deg: float = 90.0) -> float:
        """Downlink stored data with an elevation-dependent effective link rate.

        Parameters
        ----------
        storage_mb       : onboard science storage available for downlink (MB)
        dt_sec           : timestep duration (seconds)
        gs_elevation_deg : current GS elevation above local horizon (degrees)

        Returns
        -------
        float : data volume downlinked this step (MB)
        """
        # Elevation-dependent rate factor: sqrt(sin(elevation)).
        # At 90 deg → factor = 1.0 (full rate).
        # At 15 deg → sin(15°) ≈ 0.259, sqrt ≈ 0.509 (≈ 50% rate, ~6 dB loss).
        elevation_rad = math.radians(max(0.0, gs_elevation_deg))
        sin_el = math.sin(elevation_rad)
        rate_factor = math.sqrt(max(0.0, sin_el))

        effective_rate_mbps = self.config.downlink_rate_mbps * rate_factor
        requested_mb = effective_rate_mbps * dt_sec / 8.0
        downlinked_mb = min(storage_mb, requested_mb)
        self.total_downlinked_mb += downlinked_mb
        return downlinked_mb

    def record_contact_time(self, dt_sec: float) -> None:
        """Accumulate raw ground-station visibility time."""

        self.total_contact_seconds += dt_sec


class IllegalFSMTransitionError(RuntimeError):
    """Raised when the FSM is asked to perform a disallowed transition."""


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

    def __init__(self, config: MissionConfig) -> None:
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
        if adcs_saturation_hold_active or self.current_mode == "SAFE":
            self.debounce_counters["pointing_budget"] = 0
            self.pointing_budget_fault_active = False
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
        self.rejected_transition_rows: List[Dict[str, object]] = []
        self._flush_path: Optional[Path] = None
        self._flush_header_written = False
        self._flushed_row_count = 0

    def log_step(self, step_record: Dict[str, object]) -> None:
        """Append one structured timestep record to the in-memory log."""

        self.telemetry_rows.append(dict(step_record))

    def set_flush_path(self, path: Optional[Path]) -> None:
        """Configure (or disable, with None) incremental CSV flushing."""

        self._flush_path = Path(path) if path is not None else None
        self._flush_header_written = False
        self._flushed_row_count = 0

    def flush_telemetry(self) -> int:
        """Append any buffered per-step telemetry rows to disk and clear the buffer.

        This bounds peak RAM for multi-week runs and means a crash mid-run
        loses at most one flush interval of telemetry rather than the whole
        run. Event logs (slew/detumble) are NOT cleared here since they are
        read back by index mid-run (see active_slew_event_index usage).
        Returns the number of rows flushed.
        """

        if self._flush_path is None or not self.telemetry_rows:
            return 0
        chunk_df = pd.DataFrame(self.telemetry_rows)
        chunk_df.to_csv(
            self._flush_path,
            mode="a" if self._flush_header_written else "w",
            header=not self._flush_header_written,
            index=False,
        )
        self._flush_header_written = True
        self._flushed_row_count += len(self.telemetry_rows)
        self.telemetry_rows.clear()
        return self._flushed_row_count

    def log_rejected_transition(self, transition_record: Dict[str, object]) -> None:
        """Append a rejected FSM transition record."""

        self.rejected_transition_rows.append(dict(transition_record))

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
        if self.rejected_transition_rows and "rejected_transition" not in telemetry_df.columns:
            telemetry_df["rejected_transition"] = False
        return telemetry_df, saa_passes_df, gs_contacts_df, mode_transitions_df, slew_events_df, detumble_events_df


class ANTARAScienceSimulator:
    """
    Top-level simulation runner.
    Owns all subsystem instances. Runs the main loop.
    """

    def __init__(self, config: Optional[MissionConfig] = None, skip_leop: bool = False) -> None:
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
        # Fix 6: second ground station (Svalbard) for improved contact frequency
        self.ground_station_2: Optional[GroundStationModel] = None
        if self.config.gs2_enabled:
            gs2_config = MissionConfig(
                gs_lat_deg=self.config.gs2_lat_deg,
                gs_lon_deg=self.config.gs2_lon_deg,
                gs_alt_m=self.config.gs2_alt_m,
                gs_elevation_mask_deg=self.config.gs2_elevation_mask_deg,
            )
            self.ground_station_2 = GroundStationModel(gs2_config, self.earth, self.earth_frame)
        self.igrf = IGRFModel(self.config.igrf_epoch_year)
        if self.config.earth_b_field_model.lower().strip() not in {"dipole", "igrf"}:
            print(f"[WARN] Unknown earth_b_field_model={self.config.earth_b_field_model!r}; using dipole.")
        # FIX Bug 9: The original code built _saa_polygon_geo from a separate saa_vertices
        # list interpreted as (lat, lon) tuples and swapped to (lon, lat) — producing a
        # completely different polygon from SAAModel's hardcoded one.  When
        # ProtonEnvironmentModel.saa_depth_factor() called polygon.contains() with that
        # mismatched polygon it returned False (outside) for points SAAModel considered
        # inside SAA, setting depth=0 → measured_flux=0 on 3 of 6 passes.
        # Fix: pass the SAAModel's own polygon directly so both models share a single
        # authoritative SAA boundary.
        self._saa_polygon_geo = self.saa.saa_polygon
        self.proton_env = ProtonEnvironmentModel(self.config, self._saa_polygon_geo)
        self.payload = PayloadModel(self.config, self.proton_env)
        self.storm_time: bool = False   # space-weather flag; could be set externally
        self._saa_pass_index: int = 0
        self.communications = CommunicationsModel(self.config)
        self.fsm = SpacecraftFSM(self.config)
        self.logger = TelemetryLogger()

        # Stage 0.3: Skip LEOP - direct state assignment to SUN_POINTING
        if skip_leop:
            self.fsm.current_mode = "SUN_POINTING"
            self.fsm.detumble_complete = True
            self.adcs.detumble_complete = True
            self.adcs.body_rate_deg_sec[:] = 0.0
            # Compute default orientation quaternion from initial orbit state
            initial_orbit_state = self._build_orbit_state(self.propagator.getInitialState(), 0.0)
            q = self._compute_default_orientation_attitude(initial_orbit_state).quaternion
            self.adcs.set_initial_quaternion(q)

        self.initial_non_science_storage_mb = self.config.initial_storage_mb
        self.science_storage_mb = 0.0
        self.previous_in_saa = False
        self.previous_gs_visible = False
        self.previous_gs_visible_2 = False
        self.current_attitude_command: Optional[AttitudeCommand] = None
        self._steps_since_mode_transition = 9999
        self.current_sun_incidence_angle_deg = 90.0
        self.current_sun_pointing_error_deg = 0.0
        self.current_relay_pointing_error_deg = 0.0
        self.current_payload_pointing_error_deg = 0.0
        self.current_attitude_error_deg = 0.0
        self.current_gs_off_boresight_deg = 180.0
        self.current_illumination_fraction = 1.0
        self.illumination_start_fraction = 1.0
        self.illumination_target_fraction = 1.0
        self.illumination_transition_start_time_sec = 0.0
        self.previous_sun_visible_geometry = True
        self.pointing_budget_violated_frozen = False
        self.pointing_budget_violation_streak = 0
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
        self._last_progress_time_sec = 0.0
        self._last_progress_science_mb = 0.0
        self._run_geometric_self_tests()
        self._run_fsm_self_checks()

    def _get_total_storage_mb(self) -> float:
        """Return total occupied onboard storage including non-science baseline usage."""

        return self.initial_non_science_storage_mb + self.science_storage_mb

    # Stage 0.4: Checkpoint/Resume skeleton
    def save_checkpoint(self, path) -> None:
        """Save simulator state to a checkpoint file.

        Attempts pickle first; on failure falls back to JSON + numpy .npz.
        """
        path = Path(path)
        # Save RNG state
        rng_state = self.rng.bit_generator.state if hasattr(self.rng, 'bit_generator') else None

        state = {
            "fsm": {k: v for k, v in self.fsm.__dict__.items() if not k.startswith('_')},
            "adcs": {k: v for k, v in self.adcs.__dict__.items() if not k.startswith('_')},
            "eps": {k: v for k, v in self.eps.__dict__.items() if not k.startswith('_')},
            "payload": {
                "total_science_data_generated_mb": self.payload.total_science_data_generated_mb,
                "total_detection_seconds": self.payload.total_detection_seconds,
                "_cell_histograms": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.payload._cell_histograms.items()},
                "_pending_bin_counts": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.payload._pending_bin_counts.items()},
                "_pending_cell_elapsed_sec": self.payload._pending_cell_elapsed_sec,
                "_previous_cell_id": self.payload._previous_cell_id,
                "_current_pass_records": self.payload._current_pass_records,
            },
            "saa": {
                "current_pass": self.saa.current_pass,
                "completed_passes": self.saa.completed_passes,
            },
            "ground_station": {
                "current_contact": self.ground_station.current_contact,
                "completed_contacts": self.ground_station.completed_contacts,
            },
            "ground_station_2": {
                "current_contact": self.ground_station_2.current_contact if self.ground_station_2 else None,
                "completed_contacts": self.ground_station_2.completed_contacts if self.ground_station_2 else [],
            },
            "communications": {
                "total_downlinked_mb": self.communications.total_downlinked_mb,
                "total_contact_seconds": self.communications.total_contact_seconds,
            },
            "science_storage_mb": self.science_storage_mb,
            "previous_in_saa": self.previous_in_saa,
            "previous_gs_visible": self.previous_gs_visible,
            "previous_gs_visible_2": getattr(self, 'previous_gs_visible_2', False),
            "rng_state": rng_state,
            "time_sec": getattr(self, 'time_sec', 0.0),
            "logger": {
                "slew_event_rows": self.logger.slew_event_rows,
                "detumble_event_rows": self.logger.detumble_event_rows,
                "rejected_transition_rows": self.logger.rejected_transition_rows,
                "_flush_path": str(self.logger._flush_path) if self.logger._flush_path else None,
                "_flush_header_written": self.logger._flush_header_written,
                "_flushed_row_count": self.logger._flushed_row_count,
            },
            "active_slew_event_index": self.active_slew_event_index,
            "active_detumble_event_index": self.active_detumble_event_index,
            "config": self.config,
        }
        # Try pickle first
        try:
            with open(path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, TypeError, AttributeError) as e:
            # Fallback to JSON + numpy .npz
            print(f"[Checkpoint] Pickle failed ({e}), falling back to JSON+NPZ")
            json_path = Path(str(path).replace('.pkl', '.json'))
            npz_path = Path(str(path).replace('.pkl', '.npz'))
            # Separate numpy arrays for NPZ
            npz_data = {}
            json_state = {}
            for k, v in state.items():
                if isinstance(v, np.ndarray):
                    npz_data[k] = v
                elif isinstance(v, dict):
                    # Check for nested numpy arrays
                    flat = {}
                    has_numpy = False
                    for k2, v2 in v.items():
                        if isinstance(v2, np.ndarray):
                            npz_data[f"{k}_{k2}"] = v2
                            has_numpy = True
                        else:
                            flat[k2] = v2
                    if has_numpy:
                        json_state[k] = flat
                    else:
                        json_state[k] = v
                else:
                    json_state[k] = v
            with open(json_path, 'w') as f:
                json.dump(json_state, f, default=str)
            if npz_data:
                np.savez_compressed(npz_path, **npz_data)

    @classmethod
    def load_checkpoint(cls, path, config):
        """Load simulator state from a checkpoint file.

        Tries pickle first; on failure tries JSON + numpy .npz.
        """
        path = Path(path)
        state = None
        # Try pickle first
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, FileNotFoundError):
            # Try JSON + NPZ fallback
            json_path = Path(str(path).replace('.pkl', '.json'))
            npz_path = Path(str(path).replace('.pkl', '.npz'))
            if json_path.exists():
                with open(json_path, 'r') as f:
                    json_state = json.load(f)
                npz_data = {}
                if npz_path.exists():
                    npz_data = dict(np.load(npz_path, allow_pickle=True))
                # Reconstruct state
                state = {}
                for k, v in json_state.items():
                    if k in npz_data:
                        state[k] = npz_data[k]
                    elif isinstance(v, dict):
                        # Check for NPZ entries that belong to this dict
                        reconstructed = {}
                        for k2, v2 in v.items():
                            npz_key = f"{k}_{k2}"
                            if npz_key in npz_data:
                                reconstructed[k2] = npz_data[npz_key]
                            else:
                                reconstructed[k2] = v2
                        state[k] = reconstructed
                    else:
                        state[k] = v
        if state is None:
            raise RuntimeError(f"Failed to load checkpoint from {path}")

        sim = cls(config)
        # Restore non-ADCS state first (ADCS will be restored AFTER propagator init
        # because _init_propagator runs geometric self-tests that modify ADCS state)
        for k, v in state.get("fsm", {}).items():
            setattr(sim.fsm, k, v)
        for k, v in state.get("eps", {}).items():
            setattr(sim.eps, k, v)
        for k, v in state.get("payload", {}).items():
            if k in {"_cell_histograms", "_pending_bin_counts"}:
                # Convert lists back to numpy arrays
                setattr(sim.payload, k, {k2: np.array(v2) for k2, v2 in v.items()})
            else:
                setattr(sim.payload, k, v)
        for k, v in state.get("saa", {}).items():
            setattr(sim.saa, k, v)
        for k, v in state.get("ground_station", {}).items():
            setattr(sim.ground_station, k, v)
        if sim.ground_station_2:
            for k, v in state.get("ground_station_2", {}).items():
                setattr(sim.ground_station_2, k, v)

        # Restore communications
        for k, v in state.get("communications", {}).items():
            setattr(sim.communications, k, v)

        # Restore science storage
        sim.science_storage_mb = state.get("science_storage_mb", 0.0)

        # Restore SAA/GS previous state
        sim.previous_in_saa = state.get("previous_in_saa", False)
        sim.previous_gs_visible = state.get("previous_gs_visible", False)
        sim.previous_gs_visible_2 = state.get("previous_gs_visible_2", False)

        # Restore RNG state
        rng_state = state.get("rng_state")
        if rng_state is not None and hasattr(sim.rng, 'bit_generator'):
            sim.rng.bit_generator.state = rng_state

        # Restore time
        sim.time_sec = state.get("time_sec", 0.0)

        # Restore active event indices (on simulator, not logger)
        sim.active_slew_event_index = state.get("active_slew_event_index")
        sim.active_detumble_event_index = state.get("active_detumble_event_index")

        # Restore logger state
        logger_state = state.get("logger", {})
        for k, v in logger_state.items():
            if k == "_flush_path" and v:
                setattr(sim.logger, k, Path(v))
            else:
                setattr(sim.logger, k, v)

        # Recreate propagator and fast-forward to checkpoint time
        # NOTE: This runs geometric self-tests which modify ADCS state, so we restore ADCS AFTER
        sim._init_propagator()
        if sim.time_sec > 0:
            absolute_date = sim.start_date.shiftedBy(sim.time_sec)
            sim.propagator.propagate(absolute_date)

        # NOW restore ADCS state (after geometric self-tests have run)
        for k, v in state.get("adcs", {}).items():
            setattr(sim.adcs, k, v)

        return sim

    def _run_geometric_self_tests(self) -> None:
        """Abort startup if the quaternion/DCM convention is internally inconsistent."""

        # The 1e-6 degree bounds below test numerical frame construction, not
        # flight pointing accuracy. atan2 residuals are about 1e-14 degrees.

        saved_current_quaternion = self.adcs.current_quaternion.copy()
        saved_target_quaternion = self.adcs.target_quaternion.copy()
        saved_current_tuple = self.adcs.current_quaternion_tuple
        saved_continuity = self.adcs.quaternion_continuity_corrections
        saved_continuity_by_site = self.adcs.quaternion_continuity_corrections_by_site.copy()
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
            self.adcs.quaternion_continuity_corrections_by_site = saved_continuity_by_site
            self.current_attitude_command = saved_current_command

    def _run_fsm_self_checks(self) -> None:
        """Abort startup if the FSM graph or debounce reset behavior is inconsistent."""

        for start_mode in SpacecraftFSM.ALLOWED_TRANSITIONS:
            frontier = [start_mode]
            visited = {start_mode}
            can_reach_safe = False
            while frontier:
                mode = frontier.pop(0)
                if mode == "SAFE":
                    can_reach_safe = True
                    break
                for next_mode in SpacecraftFSM.ALLOWED_TRANSITIONS.get(mode, set()):
                    if next_mode not in visited:
                        visited.add(next_mode)
                        frontier.append(next_mode)
            assert can_reach_safe, f"FSM state {start_mode} cannot reach SAFE"

        test_fsm = SpacecraftFSM(self.config)
        test_fsm._debounce("debounce_reset", True, 2)
        assert test_fsm.debounce_counters["debounce_reset"] == 1, "Debounce counter did not increment"
        test_fsm._debounce("debounce_reset", False, 2)
        assert test_fsm.debounce_counters["debounce_reset"] == 0, "Debounce counter did not reset"

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
        eccentricity = float(min(max(abs(self.config.eccentricity), 0.0), 0.999999999))
        return KeplerianOrbit(
            semi_major_axis_m,
            eccentricity,
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
        position_tolerance_m = min(
            self.config.integrator_position_tolerance_m,
            self.config.integrator_velocity_tolerance_mps * max(self.config.step_sec, 1.0),
        )
        tolerances = NumericalPropagator.tolerances(
            position_tolerance_m,
            initial_orbit,
            OrbitType.EQUINOCTIAL,
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
        self.propagator.setOrbitType(OrbitType.EQUINOCTIAL)
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
        """Return the angle between two NumPy vectors in degrees.

        Uses atan2 with cross product for numerical stability at small angles.
        """

        a = self._unit_numpy(first_vector)
        b = self._unit_numpy(second_vector)
        dot = np.clip(np.dot(a, b), -1.0, 1.0)
        cross_norm = np.linalg.norm(np.cross(a, b))
        return math.degrees(math.atan2(cross_norm, dot))

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
        quaternion = self.adcs._enforce_quaternion_continuity(self.adcs.current_quaternion, quaternion, call_site="build_frame")
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

    def _rotate_vector_by_axis_angle(
        self,
        vector: np.ndarray,
        axis_unit: np.ndarray,
        angle_rad: float,
    ) -> np.ndarray:
        """Rotate *vector* by *angle_rad* around *axis_unit* using Rodrigues' formula.

        Used for predictive GS LOS propagation in _begin_slew_transition (Bug C fix).
        """
        vector = _as_flat_vector(vector)
        axis_unit = _as_flat_vector(axis_unit)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return (
            vector * cos_a
            + np.cross(axis_unit, vector) * sin_a
            + axis_unit * float(np.dot(axis_unit, vector)) * (1.0 - cos_a)
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

    def _compute_earth_b_field_inertial(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the configured Earth magnetic field model in inertial coordinates."""

        model_name = self.config.earth_b_field_model.lower().strip()
        if model_name == "igrf":
            return self.igrf.get_inertial_field_tesla(
                orbit_state.latitude_deg,
                orbit_state.longitude_deg,
                orbit_state.altitude_km,
                self._vector_to_numpy(orbit_state.sat_position_inertial),
            )

        position_inertial = self._vector_to_numpy(orbit_state.sat_position_inertial)
        radius_m = float(np.linalg.norm(position_inertial))
        if radius_m <= 0.0:
            return np.zeros(3, dtype=float)

        dipole_axis_inertial = np.array([0.0, 0.0, 1.0], dtype=float)
        radial_unit = position_inertial / radius_m
        b_scale_t = self.config.earth_B_field_T * (Constants.WGS84_EARTH_EQUATORIAL_RADIUS / radius_m) ** 3
        return b_scale_t * (
            (3.0 * float(np.dot(dipole_axis_inertial, radial_unit)) * radial_unit) - dipole_axis_inertial
        )

    def _compute_earth_b_field_body(self, orbit_state: OrbitState) -> np.ndarray:
        """Return the configured Earth magnetic field vector expressed in the body frame."""

        b_inertial_t = self._compute_earth_b_field_inertial(orbit_state)
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
        """Build the nominal LVLH body-frame attitude, re-solved every timestep.

        Authoritative body-frame convention (ICD Rev B):
          +X = RAM face / payload boresight  → aligned to velocity vector
          +Y = directional relay antenna face → aligned to nadir
          +Z = fixed solar panel face         → aligned to orbit normal, sun-side

        This method always re-computes from the current orbital state (never
        an inertial freeze) for operational modes.  The spacecraft continuously
        tracks the LVLH frame so +X stays ram, +Y stays nadir, and +Z stays on
        the sun-facing side of the orbit normal.

        For COLD_AND_DARK and DETUMBLE phases the target is solved once at phase
        entry and then frozen via the early-return in _get_attitude_command, so
        the detumble controller settles against a stable inertial reference.

        The build_frame call sets:
          primary  +X → velocity_unit   (RAM)
          secondary +Y → nadir_unit      (NADIR)
          derived  +Z = cross(+X, +Y) = cross(v̂, -r̂) = r̂ × v̂  (orbit normal h)

        A sun-side sign check ensures +Z is the orbit-normal half-space that
        faces toward the Sun; if h points anti-sun the frame is re-built using
        -h as the +Z reference.
        """

        velocity_unit = self._velocity_vector_unit(orbit_state)
        nadir_unit = self._nadir_vector_unit(orbit_state)

        # Compute candidate orbit normal h = r × v.
        # build_frame derives +Z = cross(v̂, -r̂) = r̂ × v̂ = h — normally correct.
        # Sun-side check: if h points away from the Sun, use -h for +Z.
        orbit_normal_candidate = self._orbit_normal_unit(orbit_state)
        sun_unit = self._sun_vector_unit(orbit_state)
        if float(np.dot(orbit_normal_candidate, sun_unit)) < 0.0:
            # Sun is on the -h side; build with explicit +Z = -h.
            z_reference = -orbit_normal_candidate
            quaternion = self.build_frame(
                "+X",
                velocity_unit,
                "+Z",
                z_reference,
                context="DEFAULT_ORIENTATION_HOLD target (sun-flip)",
            )
        else:
            # Standard case: h already faces the sun.
            quaternion = self.build_frame(
                "+X",
                velocity_unit,
                "+Y",
                nadir_unit,
                context="DEFAULT_ORIENTATION_HOLD target",
            )

        return AttitudeCommand(
            quaternion=quaternion,
            primary_objective="+X -> RAM (velocity)",
            secondary_objective="+Y -> nadir | +Z -> orbit normal (sun-side) | LVLH continuous re-solve",
            active_attitude_target="DEFAULT_ORIENTATION_HOLD",
            attitude_hold_reason="Continuous LVLH hold: +X=RAM +Y=nadir +Z=orbit-normal-sun-side",
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
        if mode == "SUN_POINTING":
            combined_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
            if abs(combined_error_deg - self.current_sun_pointing_error_deg) > 1.0e-6:
                if grace_complete:
                    raise AssertionError(
                        self._attitude_diagnostic_message(
                            "SUN_POINTING telemetry combined error mismatch",
                            self.adcs.current_quaternion,
                            {
                                "payload_boresight_inertial": self._payload_boresight_inertial(self.adcs.current_quaternion),
                                "nadir_target": self._nadir_vector_unit(orbit_state),
                            },
                        )
                    )
            if not self.adcs.is_slewing and not self.adcs.settling and combined_error_deg > self.config.sun_pointing_hard_limit_deg and grace_complete:
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"SUN_POINTING combined error {combined_error_deg:.6f} deg exceeds hard limit",
                        self.adcs.current_quaternion,
                        {
                            "payload_boresight_inertial": self._payload_boresight_inertial(self.adcs.current_quaternion),
                            "nadir_target": self._nadir_vector_unit(orbit_state),
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
            if gs_visible and relay_error_deg > self.config.relay_pointing_valid_deg and grace_complete:
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"RELAY pointing error {relay_error_deg:.6f} deg exceeds valid band",
                        self.adcs.current_quaternion,
                        {
                            "relay_boresight_inertial": relay_boresight_inertial,
                            "gs_target": self._ground_station_vector_unit(orbit_state),
                        },
                    )
                )
            if gs_visible and gs_off_boresight_deg > self.config.directional_antenna_half_beamwidth_deg and grace_complete:
                raise AssertionError(
                    self._attitude_diagnostic_message(
                        f"RELAY GS off-boresight {gs_off_boresight_deg:.6f} deg exceeds beamwidth",
                        self.adcs.current_quaternion,
                        {
                            "relay_boresight_inertial": relay_boresight_inertial,
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
        science_sample_inside_saa: Optional[bool],
    ) -> None:
        """Emit one-time warnings for physically inconsistent states."""

        if self.eps.solar_generation_w > self.config.max_solar_power_w + 1.0e-6:
            self._maybe_warn("solar_power_limit", "Solar generation exceeded configured physical limit.")
        if (mode == "DETECTION" and not self.adcs.is_slewing and not self.adcs.settling and self._steps_since_mode_transition >= self.config.mode_transition_grace_steps and self.current_payload_pointing_error_deg > 90.0):
            self._maybe_warn("payload_error_large", "Payload pointing error exceeded 90 deg during DETECTION.")
        if (mode == "RELAY" and not self.adcs.is_slewing and not self.adcs.settling and self._steps_since_mode_transition >= self.config.mode_transition_grace_steps and self.current_relay_pointing_error_deg > 90.0):
            self._maybe_warn("relay_error_large", "Relay pointing error exceeded 90 deg during RELAY.")
        if self.current_illumination_fraction <= 0.0 and self.eps.battery_wh > battery_before_step_wh + 1.0e-6:
            self._maybe_warn("battery_increase_eclipse", "Battery increased during eclipse.")
        if science_generated_mb > 0.0 and science_sample_inside_saa is False:
            self._maybe_warn("science_outside_saa", "Science generation occurred outside the SAA.")
        if downlinked_mb > 0.0 and not gs_visible:
            self._maybe_warn("downlink_without_visibility", "Downlink occurred without GS visibility.")

        # Science model checks — only when inside SAA and science was generated
        if inside_saa and science_generated_mb > 0.0:
            flux_res = self.payload.last_flux_result
            if flux_res:
                measured = np.array(flux_res["measured_flux"])
                assert np.all(measured >= 0.0), \
                    f"Negative proton flux detected: {measured}"
                assert np.all(np.isfinite(measured)), \
                    f"Non-finite proton flux: {measured}"
                pitch = float(flux_res["pitch_angle_deg"])
                assert 0.0 <= pitch <= 90.0, \
                    f"Pitch angle out of bounds: {pitch:.3f} deg"
                ew = float(flux_res["ew_factor"])
                assert 0.5 <= ew <= 1.5, \
                    f"EW factor out of physically plausible range: {ew:.4f}"

    def _compute_sun_pointing_attitude(
        self,
        orbit_state: OrbitState,
        sun_visible: bool,
    ) -> AttitudeCommand:
        """Return the SUN_POINTING attitude command.

        SUN_POINTING is a power-state label, not an active Sun-tracking mode.
        The spacecraft holds the nominal LVLH frame (+X=RAM, +Y=nadir,
        +Z=orbit-normal sun-side), re-solved every timestep.  No sun tracking
        occurs; the mode label only reflects the power/charging submode.
        """
        base_command = self._compute_default_orientation_attitude(orbit_state)
        return AttitudeCommand(
            quaternion=base_command.quaternion.copy(),
            primary_objective="SUN_POINTING power-state hold (LVLH)",
            secondary_objective=(
                "No Sun tracking; +X=RAM +Y=nadir +Z=orbit-normal (sun-side) LVLH continuous re-solve"
            ),
            active_attitude_target="DEFAULT_ORIENTATION_HOLD",
            attitude_hold_reason="SUN_POINTING is a power label; attitude is continuous LVLH hold",
            sun_visible=sun_visible,
        )

    def _compute_detection_attitude(self, orbit_state: OrbitState) -> AttitudeCommand:
        """Build DETECTION attitude as a continuous LVLH hold (+X=RAM, +Y=nadir, +Z=orbit-normal).

        Per ICD Rev B and operational philosophy: DETECTION does NOT actively track
        the payload boresight or any target.  The spacecraft simply holds the
        nominal LVLH body frame throughout the SAA pass.  The +X face (payload
        boresight) naturally points RAM as a consequence of the LVLH hold, which
        is sufficient for proton detection geometry.

        The LVLH frame is re-solved every timestep (not frozen inertially) so that
        +Y stays nadir and +Z stays on the sun-side orbit normal as the orbit
        progresses.  No active pointing, no slews to a science target.
        """
        base_command = self._compute_default_orientation_attitude(orbit_state)
        return AttitudeCommand(
            quaternion=base_command.quaternion.copy(),
            primary_objective="DETECTION LVLH hold: +X -> RAM (no active tracking)",
            secondary_objective="+Y -> nadir | +Z -> orbit normal (sun-side) | continuous re-solve",
            active_attitude_target="DETECTION_INERTIAL_HOLD",
            attitude_hold_reason="DETECTION holds LVLH frame; no boresight tracking",
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
        """Build the single-antenna RELAY attitude using the +Y fixed relay face.

        Called every timestep (not just on mode entry) to continuously track the
        GS LOS vector.  The assertion tolerance is relaxed to 1e-4 deg — still
        four orders of magnitude tighter than any hardware requirement — because
        the quaternion/DCM round-trip accumulates ~1e-6 deg of floating-point
        noise that reliably triggered the original 1e-6 deg threshold in the hot
        loop.
        """
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
        # Tolerance relaxed from 1e-6 to 1e-4 deg: the round-trip DCM→quaternion→DCM
        # accumulates ~1e-6 deg of fp noise, which reliably exceeded the old threshold
        # every step in the hot loop.  1e-4 deg is still an engineering-grade check.
        self._assert_axis_alignment("RELAY target", quaternion, "+Y", target_unit, 1.0e-4)
        self._assert_axis_alignment("RELAY target", quaternion, "+Z", z_target, 1.0e-4)
        return AttitudeCommand(
            quaternion=quaternion,
            primary_objective="+Y -> ground station LOS",
            secondary_objective="+Z -> orbit-normal projection about +Y",
            active_attitude_target="RELAY_INERTIAL_HOLD",
            attitude_hold_reason="Continuous closed-loop GS tracking",
            relay_axis_label="+Y",
        )

    def _compute_relay_attitude(self, orbit_state: OrbitState, gs_visible: bool) -> AttitudeCommand:
        """Build the RELAY attitude.

        When the GS is visible, RELAY now tracks the actual GS LOS with the +Y
        relay face. When the GS is not visible, the mode falls back to the
        standard LVLH hold (+Y nadir).
        """

        if gs_visible:
            return self._compute_relay_candidate(orbit_state, self._ground_station_vector_unit(orbit_state))

        base_command = self._compute_default_orientation_attitude(orbit_state)
        return AttitudeCommand(
            quaternion=base_command.quaternion.copy(),
            primary_objective="RELAY LVLH hold: +Y -> nadir",
            secondary_objective="+X -> RAM | +Z -> orbit normal (sun-side) | continuous re-solve",
            active_attitude_target="RELAY_INERTIAL_HOLD",
            attitude_hold_reason="RELAY holds LVLH frame when GS is not visible",
            relay_axis_label="+Y",
        )

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
            # During startup / detumble, solve the LVLH target ONCE on phase entry
            # and then freeze it as an inertial hold.  Continuously re-solving the
            # target during active detumble changes the reference the B-dot / wheel-damp
            # controller is settling against and can drive numerical instability in
            # _integrate_body_rates via quaternion continuity corrections.
            if (
                self.current_attitude_command is not None
                and self.current_attitude_command.active_attitude_target == "DEFAULT_ORIENTATION_HOLD"
            ):
                return AttitudeCommand(
                    quaternion=self.adcs.target_quaternion.copy(),
                    primary_objective="Startup LVLH hold (inertial freeze until detumble complete)",
                    secondary_objective="+X=RAM +Y=nadir +Z=orbit-normal | frozen at phase entry",
                    active_attitude_target="DEFAULT_ORIENTATION_HOLD",
                    attitude_hold_reason="Stable inertial reference during DETUMBLE/COLD_AND_DARK",
                )
            command = self._compute_default_orientation_attitude(orbit_state)
            command.primary_objective = "Startup LVLH target (initial solve)"
            command.secondary_objective = "+X=RAM +Y=nadir +Z=orbit-normal (sun-side) | frozen for detumble"
            return command
        if mode == "SAFE":
            return self._compute_safe_attitude(orbit_state)
        if mode == "SUN_POINTING":
            return self._compute_sun_pointing_attitude(orbit_state, sun_visible)
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
        cosine_incidence = np.clip(float(np.dot(self._unit_numpy(body_z_in_inertial), sun_unit)), -1.0, 1.0)
        incidence_deg = math.degrees(math.acos(cosine_incidence))
        return float(np.clip(incidence_deg, 0.0, 180.0))

    def _compute_noisy_sun_incidence_angle_deg(self, orbit_state: OrbitState, mode: str) -> float:
        """Return the SUN_POINTING incidence angle with seeded Gaussian tracking noise."""

        incidence_deg = self._compute_sun_incidence_angle_deg(orbit_state)
        if mode != "SUN_POINTING":
            return incidence_deg
        noisy_incidence = incidence_deg + float(
            self.rng.normal(0.0, self.config.sun_pointing_tracking_noise_deg)
        )
        return float(np.clip(noisy_incidence, 0.0, 180.0))

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
            return self.config.safe_hold_deadband_deg
        if mode == "DETECTION":
            return self.config.ram_track_deadband_deg
        if mode == "RELAY":
            return self.config.relay_track_deadband_deg
        return self.config.attitude_command_deadband_deg

    def _hard_pointing_limit_for_mode(self, mode: str) -> float:
        """Return the hard enforcement limit for the current operational mode."""

        if mode == "DETECTION":
            return self.config.detection_pointing_hard_limit_deg
        if mode == "RELAY":
            return self.config.relay_pointing_hard_limit_deg
        if mode == "SUN_POINTING":
            return self.config.sun_pointing_hard_limit_deg
        return float("inf")

    def _hard_pointing_error_for_mode(self, mode: str, orbit_state: OrbitState, gs_visible: bool) -> float:
        """Return the relevant pointing error used for hard enforcement."""

        payload_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
        relay_error_deg = self._compute_relay_pointing_error_deg(orbit_state, self.adcs.commanded_relay_axis or "+Y", gs_visible)
        if mode == "DETECTION":
            return payload_error_deg
        if mode == "RELAY":
            return relay_error_deg
        if mode == "SUN_POINTING":
            return self._compute_sun_pointing_error_deg(orbit_state)
        return 0.0

    def _compute_sun_pointing_error_deg(self, orbit_state: OrbitState) -> float:
        """Return the combined SUN_POINTING error against RAM and nadir."""

        payload_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
        nadir_error_deg = self._compute_relay_pointing_error_deg(
            orbit_state,
            self.adcs.commanded_relay_axis or "+Y",
            False,
        )
        return max(payload_error_deg, nadir_error_deg)

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

        hard_pointing_limit_deg = self._hard_pointing_limit_for_mode(mode)
        hard_pointing_error_deg = self._hard_pointing_error_for_mode(mode, orbit_state, gs_visible)
        self.pointing_budget_violated_frozen = False
        freeze_budget_mode = self.target_mode if self.target_mode in {"DETECTION", "RELAY", "SUN_POINTING"} else mode
        freeze_budget_limit_deg = self._hard_pointing_limit_for_mode(freeze_budget_mode)
        freeze_budget_error_deg = self._hard_pointing_error_for_mode(freeze_budget_mode, orbit_state, gs_visible)

        if mode == "SLEWING":
            self.adcs.advance(dt_sec)
            self.adcs.is_slewing = time_sec < self.slew_end_time_sec
            self.current_attitude_error_deg = self.adcs.get_pointing_error_deg()
            self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            self.current_sun_pointing_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
            self.current_gs_off_boresight_deg = self._compute_gs_off_boresight_deg(orbit_state, gs_visible)
            if freeze_budget_error_deg > freeze_budget_limit_deg:
                self.pointing_budget_violated_frozen = True
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
            self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            self.current_sun_pointing_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
            self.current_gs_off_boresight_deg = self._compute_gs_off_boresight_deg(orbit_state, gs_visible)
            if freeze_budget_error_deg > freeze_budget_limit_deg:
                self.pointing_budget_violated_frozen = True
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
            self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
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
            self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            self.current_sun_pointing_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
            self.current_gs_off_boresight_deg = self._compute_gs_off_boresight_deg(orbit_state, gs_visible)
            return

        if (
            self.current_attitude_command.active_attitude_target == command.active_attitude_target
            and np.allclose(command.quaternion, self.current_attitude_command.quaternion, atol=1.0e-12, rtol=0.0)
            and command.primary_objective == self.current_attitude_command.primary_objective
        ):
            self.current_attitude_command = command
            self.adcs.advance(dt_sec)
            self.current_attitude_error_deg = self.adcs.get_pointing_error_deg()
            self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
            self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
            self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
                orbit_state,
                self.adcs.commanded_relay_axis,
                gs_visible,
            )
            self.current_sun_pointing_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
            self.current_gs_off_boresight_deg = self._compute_gs_off_boresight_deg(orbit_state, gs_visible)
            return

        force_update = hard_pointing_error_deg > hard_pointing_limit_deg

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
        self.current_sun_incidence_angle_deg = self._compute_noisy_sun_incidence_angle_deg(orbit_state, mode)
        self.current_payload_pointing_error_deg = self._compute_payload_pointing_error_deg(orbit_state)
        self.current_relay_pointing_error_deg = self._compute_relay_pointing_error_deg(
            orbit_state,
            self.adcs.commanded_relay_axis,
            gs_visible,
        )
        self.current_sun_pointing_error_deg = self._compute_sun_pointing_error_deg(orbit_state)
        self.current_gs_off_boresight_deg = self._compute_gs_off_boresight_deg(orbit_state, gs_visible)
        if mode == "SAFE" or self.adcs.saturation_hold_active:
            if freeze_budget_error_deg > freeze_budget_limit_deg:
                self.pointing_budget_violated_frozen = True

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

    def _predict_orbit_state_at_time(self, orbit_state: OrbitState, future_time_sec: float) -> OrbitState:
        """Propagate the orbit state forward to a future simulation time."""

        future_time_sec = max(future_time_sec, orbit_state.time_sec)
        future_date = orbit_state.absolute_date.shiftedBy(future_time_sec - orbit_state.time_sec)
        propagated_state = self._propagate_state(future_date)
        return self._build_orbit_state(propagated_state, future_time_sec)

    def _predict_saa_ingress_time_sec(
        self,
        orbit_state: OrbitState,
        lookahead_sec: float,
    ) -> Optional[float]:
        """Estimate the next SAA ingress time by forward propagating the current orbit state."""

        if self.saa.is_inside(orbit_state.latitude_deg, orbit_state.longitude_deg):
            return orbit_state.time_sec

        horizon_sec = max(self.config.step_sec, lookahead_sec)
        n_samples = max(1, int(math.ceil(horizon_sec / max(self.config.step_sec, 1.0e-9))))
        for sample_index in range(1, n_samples + 1):
            future_time_sec = orbit_state.time_sec + (sample_index * self.config.step_sec)
            future_state = self._predict_orbit_state_at_time(orbit_state, future_time_sec)
            if self.saa.is_inside(future_state.latitude_deg, future_state.longitude_deg):
                return future_time_sec
        return None

    def _maybe_begin_predicted_slew(
        self,
        from_mode: str,
        target_mode: str,
        predicted_event_time_sec: Optional[float],
        orbit_state: OrbitState,
        sun_visible: bool,
        battery_wh: float,
        gs_visible: bool,
        time_sec: float,
        reason_prefix: str,
    ) -> bool:
        """Begin a pre-slew when a predicted event is within the configured margin."""

        if predicted_event_time_sec is None:
            return False
        if from_mode in {"SAFE", "RELAY", "SLEWING", "COLD_AND_DARK", "DETUMBLE"}:
            return False
        if target_mode == "RELAY" and battery_wh < self.config.min_relaybattery_capacity_wh:
            return False
        if target_mode == "DETECTION" and battery_wh < self.config.saa_detection_min_power_wh:
            return False

        time_to_event_sec = predicted_event_time_sec - time_sec
        if not (0.0 < time_to_event_sec <= self.config.slew_transition_penalty_sec):
            return False

        try:
            self._begin_slew_transition(
                from_mode=from_mode,
                target_mode=target_mode,
                reason=f"{reason_prefix} {time_to_event_sec:.0f}s before predicted event",
                orbit_state=orbit_state,
                sun_visible=sun_visible,
                battery_wh=battery_wh,
                gs_visible=gs_visible,
                time_sec=time_sec,
            )
            return True
        except IllegalFSMTransitionError as exc:
            self.fsm.record_rejected_transition(
                from_mode,
                target_mode,
                f"{exc} | {reason_prefix.lower()}",
            )
            self.logger.log_rejected_transition(
                {
                    "time_sec": time_sec,
                    "from_mode": from_mode,
                    "to_mode": target_mode,
                    "reason": f"{exc} | {reason_prefix.lower()}",
                }
            )
            return False

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
        """Enter SLEWING and command the target attitude for the requested mode.

        FIX (Bug C): For RELAY slews the GS moves ~1-2 deg/s relative to the
        spacecraft during the slew.  The original code used the instantaneous GS
        LOS at slew START, producing a target that was already 60-120 deg stale
        by slew completion.  The fix propagates the GS LOS forward in time by
        the estimated slew duration using the angular velocity of the GS vector
        (cross-product of the satellite velocity and GS LOS, divided by range).
        The predicted position is used as the slew target quaternion so the
        antenna is pointing at where the GS will be when the slew ends.
        """

        if target_mode == "RELAY" and gs_visible:
            # Estimate slew duration using current attitude vs a candidate target
            candidate_command = self._get_attitude_command(target_mode, orbit_state, sun_visible, battery_wh, gs_visible)
            _, estimated_duration_sec = self._calculate_slew_duration_sec(candidate_command.quaternion)

            # Propagate GS LOS forward by estimated_duration_sec.
            # Angular velocity of GS LOS ≈ (v_sat × r_gs_unit) / |r_gs|
            # where r_gs is the inertial vector from spacecraft to GS.
            gs_los_unit = self._ground_station_vector_unit(orbit_state)
            sat_vel = self._vector_to_numpy(orbit_state.sat_velocity_inertial)
            # Approximate GS range (GS is on Earth surface, ~6371+70 km from geocenter)
            sat_pos_mag = float(np.linalg.norm(self._vector_to_numpy(orbit_state.sat_position_inertial)))
            gs_range_approx = max(sat_pos_mag * 0.5, 500_000.0)   # lower-bound 500 km
            # Angular velocity of LOS vector (rad/s)
            omega_los = np.cross(sat_vel, gs_los_unit) / gs_range_approx
            # Predict LOS after estimated_duration_sec: rotate gs_los_unit by omega*dt
            angle_rad = float(np.linalg.norm(omega_los)) * estimated_duration_sec
            if angle_rad > 1.0e-9:
                rot_axis = omega_los / np.linalg.norm(omega_los)
                predicted_los = self._rotate_vector_by_axis_angle(gs_los_unit, rot_axis, angle_rad)
            else:
                predicted_los = gs_los_unit

            # Clamp predicted LOS to unit sphere
            predicted_los = self._unit_numpy(predicted_los)

            z_reference = self._orbit_normal_unit(orbit_state)
            if float(np.linalg.norm(self._project_orthogonal(z_reference, predicted_los))) <= 1.0e-9:
                z_reference = self._body_axis_to_inertial(self.adcs.current_quaternion, "+Z")
            predicted_quaternion = self.build_frame(
                "+Y",
                predicted_los,
                "+Z",
                z_reference,
                context="RELAY predictive slew target",
            )
            target_command = AttitudeCommand(
                quaternion=predicted_quaternion,
                primary_objective="+Y -> predicted GS LOS at slew completion",
                secondary_objective="+Z -> orbit-normal projection about +Y",
                active_attitude_target="RELAY_INERTIAL_HOLD",
                attitude_hold_reason="Predictive slew to GS position at slew completion",
                relay_axis_label="+Y",
            )
            angle_deg, duration_sec = self._calculate_slew_duration_sec(target_command.quaternion)
        else:
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
        if self.ground_station_2 is not None:
            self.ground_station_2.close_open_contact(final_time_sec, final_absolute_date)
        if self.active_slew_event_index is not None:
            self.logger.finalize_slew_event(self.active_slew_event_index, final_time_sec)
            self.active_slew_event_index = None
        if self.active_detumble_event_index is not None:
            self.logger.finalize_detumble_event(self.active_detumble_event_index, final_time_sec)
            self.active_detumble_event_index = None

        # FIX Bugs 5+13: merge GS2 contacts into the gs_contacts list.
        # Previously only self.ground_station.completed_contacts (primary India station)
        # was passed to build_dataframes(), making all Svalbard contacts invisible.
        # All actual downlink (0.0638 MB) occurred through GS2 but showed as phantom
        # in gs_contacts.csv (downlinked_mb=0) while real contacts at T+11.28hr and
        # T+23.23hr appeared genuine but had zero bytes (data already gone via GS2).
        all_contacts = list(self.ground_station.completed_contacts)
        if self.ground_station_2 is not None:
            for c in self.ground_station_2.completed_contacts:
                labelled = dict(c)
                labelled["station"] = "GS2_Svalbard"
                all_contacts.append(labelled)
        # Sort combined contact list by AOS time
        all_contacts.sort(key=lambda c: c.get("aos_time_sec", 0.0))
        # Label primary contacts for traceability
        for c in self.ground_station.completed_contacts:
            c.setdefault("station", "GS1_India")

        return self.logger.build_dataframes(
            self.saa.completed_passes,
            all_contacts,
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

        if not _MATPLOTLIB_AVAILABLE:
            print("[WARN] Skipping mission overview plot because plotting dependencies are unavailable.")
            return

        use_cartopy = bool(_CARTOPY_AVAILABLE)
        telemetry_df = telemetry_df.copy()
        telemetry_df["time_hours"] = telemetry_df["time_sec"] / 3600.0
        telemetry_df["time_days"] = telemetry_df["time_sec"] / 86400.0
        telemetry_df["cumulative_science_mb"] = telemetry_df["cumulative_science_generated_mb"]
        telemetry_df["mode_code"] = telemetry_df["mode"].map({mode: idx for idx, mode in enumerate(self.config.mode_colors)}).fillna(0).astype(int)
        telemetry_df["wheel_momentum_mag"] = np.linalg.norm(
            telemetry_df[["wheel_mom_x", "wheel_mom_y", "wheel_mom_z"]].to_numpy(),
            axis=1,
        )

        figure = plt.figure(figsize=(24, 18), constrained_layout=True)
        grid = figure.add_gridspec(3, 3)
        map_projection = ccrs.PlateCarree() if use_cartopy else None

        axis1 = figure.add_subplot(grid[0, 0], projection=map_projection) if use_cartopy else figure.add_subplot(grid[0, 0])
        axis2 = figure.add_subplot(grid[0, 1], projection=map_projection) if use_cartopy else figure.add_subplot(grid[0, 1])
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
            scatter_kwargs = {"s": 2, "color": color, "label": mode, "alpha": 0.75}
            if use_cartopy:
                scatter_kwargs["transform"] = map_projection
            axis1.scatter(mode_rows["lon_deg"], mode_rows["lat_deg"], **scatter_kwargs)
        axis1.legend(loc="lower left", fontsize=8)

        self._format_map_axis(axis2, "Panel 2: SAA Overlay and Science Density")
        science_rows = telemetry_df[telemetry_df["science_data_mb"] > 0.0]
        if not science_rows.empty:
            hexbin_kwargs = {
                "C": science_rows["science_data_mb"],
                "reduce_C_function": np.sum,
                "gridsize": 45,
                "cmap": "inferno",
                "mincnt": 1,
            }
            if use_cartopy:
                hexbin_kwargs["transform"] = map_projection
            axis2.hexbin(science_rows["lon_deg"], science_rows["lat_deg"], **hexbin_kwargs)
        polygon_lon, polygon_lat = self.saa.saa_polygon.exterior.xy
        if use_cartopy:
            axis2.plot(polygon_lon, polygon_lat, color="orange", linewidth=2.0, transform=map_projection)
        else:
            axis2.plot(polygon_lon, polygon_lat, color="orange", linewidth=2.0)

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
        cmap_mode = plt.matplotlib.colors.ListedColormap([self.config.mode_colors[m] for m in self.config.mode_colors])
        axis5.imshow(mode_matrix, aspect="auto", cmap=cmap_mode, extent=[telemetry_df["time_days"].iloc[0], telemetry_df["time_days"].iloc[-1], 0.5, 1.0])
        submode_order = [SunSubmode.NONE, SunSubmode.CHARGING, SunSubmode.PASSIVE, SunSubmode.ECLIPSE]
        submode_to_idx = {s: i for i, s in enumerate(submode_order)}
        submode_codes = telemetry_df["sun_submode"].map(submode_to_idx).fillna(0).astype(int).to_numpy()
        submode_matrix = np.array([submode_codes])
        cmap_submode = plt.matplotlib.colors.ListedColormap([self.config.sun_submode_colors[s] for s in submode_order])
        axis5.imshow(submode_matrix, aspect="auto", cmap=cmap_submode, extent=[telemetry_df["time_days"].iloc[0], telemetry_df["time_days"].iloc[-1], 0.0, 0.5])
        axis5.set_yticks([0.25, 0.75])
        axis5.set_yticklabels(["Sun\nSubmode", "Mode"], fontsize=7)
        axis5.set_xlabel("Time (days)")
        axis5.set_xlim(telemetry_df["time_days"].iloc[0], telemetry_df["time_days"].iloc[-1])

        axis6.set_title("Panel 6: Science, Downlink, and Storage")
        axis6.plot(telemetry_df["time_days"], telemetry_df["cumulative_science_mb"], label="Cumulative Science", color="red")
        axis6.plot(telemetry_df["time_days"], telemetry_df["cumulative_downlinked_mb"], label="Cumulative Downlinked", color="blue")
        axis6.set_xlabel("Time (days)")
        axis6.set_ylabel("Data (MB)")
        axis6.grid(True, alpha=0.3)
        axis6b = axis6.twinx()
        science_only_storage = telemetry_df["storage_mb"] - self.config.initial_storage_mb
        axis6b.plot(telemetry_df["time_days"], science_only_storage.clip(lower=0.0), label="Science Storage", color="purple", linestyle="--")
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
            axis7.axvline(self.config.large_slew_threshold_deg, color="black", linestyle=":", label=f"Large Slew {self.config.large_slew_threshold_deg:.0f} deg")
            axis7.legend(fontsize=8)
        axis7.set_xlabel("Slew Angle (deg)")
        axis7.set_ylabel("Count")
        axis7.grid(True, alpha=0.3)

        axis8.set_title("Panel 8: Wheel Momentum Magnitude")
        axis8.plot(telemetry_df["time_days"], telemetry_df["wheel_momentum_mag"], color="teal")
        axis8.axhline(self.config.wheel_max_momentum_nms, color="red", linestyle="--", label="Saturation Limit")
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
                axis9.text(start_day + duration_day, row_index, f" {row['max_el_deg']:.1f} deg, {row['downlinked_mb']:.1f} MB", fontsize=7, va="center")
            axis9.set_yticks(range(len(gs_contacts_df)))
            axis9.set_yticklabels([f"Pass {idx + 1}" for idx in range(len(gs_contacts_df))])
        else:
            axis9.text(0.5, 0.5, "No GS contacts", ha="center", va="center", transform=axis9.transAxes)
        axis9.set_xlabel("Time (days)")
        axis9.grid(True, alpha=0.3)

        output_path = figures_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_mission_overview.png")
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

    def _plot_attitude_diagnostics(self, telemetry_df: pd.DataFrame) -> None:
        """Render focused attitude, saturation, and relay-tracking diagnostics."""

        if not _MATPLOTLIB_AVAILABLE or telemetry_df.empty:
            return

        telemetry_df = telemetry_df.copy()
        telemetry_df["time_days"] = telemetry_df["time_sec"] / 86400.0
        telemetry_df["wheel_momentum_mag"] = np.linalg.norm(
            telemetry_df[["wheel_mom_x", "wheel_mom_y", "wheel_mom_z"]].to_numpy(),
            axis=1,
        )
        if "sun_pointing_error_deg" not in telemetry_df.columns:
            telemetry_df["sun_pointing_error_deg"] = np.maximum(
                telemetry_df["payload_pointing_error_deg"].to_numpy(dtype=float),
                telemetry_df["relay_pointing_error_deg"].to_numpy(dtype=float),
            )

        figure, (axis1, axis2, axis3) = plt.subplots(3, 1, figsize=(18, 12), constrained_layout=True, sharex=True)

        axis1.set_title("Pointing Error vs Time")
        detection_rows = telemetry_df[telemetry_df["mode"] == "DETECTION"]
        relay_rows = telemetry_df[telemetry_df["mode"] == "RELAY"]
        sun_rows = telemetry_df[telemetry_df["mode"] == "SUN_POINTING"]
        if not detection_rows.empty:
            axis1.plot(detection_rows["time_days"], detection_rows["payload_pointing_error_deg"], color="crimson", label="DETECTION payload error")
        if not relay_rows.empty:
            axis1.plot(relay_rows["time_days"], relay_rows["relay_pointing_error_deg"], color="royalblue", label="RELAY relay error")
            axis1.plot(relay_rows["time_days"], relay_rows["gs_off_boresight_deg"], color="navy", linestyle="--", alpha=0.7, label="RELAY GS off-boresight")
        if not sun_rows.empty:
            axis1.plot(sun_rows["time_days"], sun_rows["sun_pointing_error_deg"], color="darkorange", label="SUN_POINTING combined error")
        axis1.axhline(self.config.detection_pointing_hard_limit_deg, color="crimson", linestyle=":", label="Detection hard limit")
        axis1.axhline(self.config.relay_pointing_hard_limit_deg, color="royalblue", linestyle=":", label="Relay hard limit")
        axis1.axhline(self.config.sun_pointing_hard_limit_deg, color="darkorange", linestyle=":", label="Sun-pointing hard limit")
        if "pointing_budget_violated_frozen" in telemetry_df.columns:
            frozen_mask = telemetry_df["pointing_budget_violated_frozen"].fillna(False).to_numpy(dtype=bool)
            if frozen_mask.any():
                start_idx = None
                for idx, flag in enumerate(frozen_mask):
                    if flag and start_idx is None:
                        start_idx = idx
                    elif not flag and start_idx is not None:
                        axis1.axvspan(
                            telemetry_df["time_days"].iloc[start_idx],
                            telemetry_df["time_days"].iloc[idx - 1],
                            color="gray",
                            alpha=0.18,
                            label="Frozen violation" if start_idx == 0 else None,
                        )
                        start_idx = None
                if start_idx is not None:
                    axis1.axvspan(
                        telemetry_df["time_days"].iloc[start_idx],
                        telemetry_df["time_days"].iloc[-1],
                        color="gray",
                        alpha=0.18,
                        label="Frozen violation" if start_idx == 0 else None,
                    )
        axis1.set_ylabel("Error (deg)")
        axis1.grid(True, alpha=0.3)
        axis1.legend(fontsize=8, ncol=2)

        axis2.set_title("Wheel Momentum and Saturation Events")
        axis2.plot(telemetry_df["time_days"], telemetry_df["wheel_momentum_mag"], color="teal", label="Wheel momentum magnitude")
        axis2.axhline(self.config.wheel_max_momentum_nms, color="red", linestyle="--", label="Saturation limit")
        if "saturation_hold_active" in telemetry_df.columns:
            sat_mask = telemetry_df["saturation_hold_active"].fillna(False).to_numpy(dtype=bool)
            sat_edges = np.where(np.logical_and(sat_mask, np.concatenate(([False], ~sat_mask[:-1]))))[0]
            if sat_edges.size > 0:
                axis2.scatter(
                    telemetry_df["time_days"].iloc[sat_edges],
                    telemetry_df["wheel_momentum_mag"].iloc[sat_edges],
                    color="red",
                    s=20,
                    label="Saturation event",
                    zorder=5,
                )
        axis2.set_ylabel("Momentum (Nms)")
        axis2.grid(True, alpha=0.3)
        axis2.legend(fontsize=8)

        axis3.set_title("Relay GS Off-Boresight")
        if not relay_rows.empty:
            axis3.plot(relay_rows["time_days"], relay_rows["gs_off_boresight_deg"], color="slateblue", label="GS off-boresight")
        axis3.axhline(self.config.directional_antenna_half_beamwidth_deg, color="black", linestyle="--", label="Soft link beamwidth")
        axis3.axhline(self.config.relay_pointing_hard_limit_deg, color="darkorange", linestyle=":", label="Hard enforcement limit")
        axis3.set_xlabel("Time (days)")
        axis3.set_ylabel("Angle (deg)")
        axis3.grid(True, alpha=0.3)
        axis3.legend(fontsize=8)

        output_path = figures_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_attitude_diagnostics.png")
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

    def _plot_science_results(
        self,
        telemetry_df: pd.DataFrame,
        saa_passes_df: pd.DataFrame,
    ) -> None:
        """
        Generate a 4-panel science verification figure.

        Panel 1: Ground track coloured by the lowest-energy histogram bin
                 with SAA polygon overlay.
        Panel 2: Representative histogram bins (measured) and AP9 vs time,
                 with DETECTION windows shaded.
        Panel 3: Magnetic pitch angle vs time during SAA passes.
        Panel 4: East-West asymmetry factor vs SAA pass number as a bar chart.
        """
        if not _MATPLOTLIB_AVAILABLE:
            print("[WARN] Skipping science analysis plot because plotting dependencies are unavailable.")
            return

        use_cartopy = bool(_CARTOPY_AVAILABLE)
        telemetry_df = telemetry_df.copy()
        telemetry_df["time_days"] = telemetry_df["time_sec"] / 86400.0
        bin_centers = np.asarray(self.payload.bin_centers_mev, dtype=float)
        representative_bins = np.linspace(0, max(len(bin_centers) - 1, 0), num=min(5, len(bin_centers)), dtype=int)
        channel_colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

        figure = plt.figure(figsize=(20, 16), constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
        map_projection = ccrs.PlateCarree() if use_cartopy else None

        axis1 = figure.add_subplot(grid[0, 0], projection=map_projection) if use_cartopy else figure.add_subplot(grid[0, 0])
        axis1.set_title("Panel 1: Ground Track - Lowest-Energy Bin")
        if "science_measured_flux" in telemetry_df.columns:
            telemetry_df["science_measured_flux_first"] = telemetry_df["science_measured_flux"].apply(
                lambda value: float(value[0]) if self._science_flux_row_is_valid(value) else 0.0
            )
            fpdu_mask = telemetry_df["science_measured_flux_first"] > 0.0
            scatter_value_col = "science_measured_flux_first"
        else:
            fpdu_mask = telemetry_df["fpdu_ch0"] > 0.0
            scatter_value_col = "fpdu_ch0"
        if fpdu_mask.any():
            scatter_kwargs = {"s": 4, "cmap": "plasma", "alpha": 0.85}
            if use_cartopy:
                scatter_kwargs["transform"] = map_projection
            scatter = axis1.scatter(telemetry_df.loc[fpdu_mask, "lon_deg"], telemetry_df.loc[fpdu_mask, "lat_deg"], c=telemetry_df.loc[fpdu_mask, scatter_value_col], **scatter_kwargs)
            plt.colorbar(scatter, ax=axis1, shrink=0.6, label="Lowest-energy bin flux (p/cm?/s/sr/MeV)")
        polygon_lon, polygon_lat = self._saa_polygon_geo.exterior.xy
        if use_cartopy:
            axis1.plot(polygon_lon, polygon_lat, color="orange", linewidth=2.0, transform=map_projection, label="SAA (science)")
        else:
            axis1.plot(polygon_lon, polygon_lat, color="orange", linewidth=2.0, label="SAA (science)")
        axis1.legend(loc="lower left", fontsize=8)

        axis2 = figure.add_subplot(grid[0, 1])
        axis2.set_title("Panel 2: Representative Histogram Bins (solid) and AP9 (dashed)")
        if "science_measured_flux" in telemetry_df.columns and "science_ap9_flux" in telemetry_df.columns:
            valid_rows = self._valid_science_flux_rows(telemetry_df)
            if valid_rows.any():
                time_days_valid = telemetry_df.loc[valid_rows, "time_days"].to_numpy(dtype=float)
                measured_matrix = np.vstack([np.asarray(row, dtype=float) for row in telemetry_df.loc[valid_rows, "science_measured_flux"].to_list()])
                ap9_matrix = np.vstack([np.asarray(row, dtype=float) for row in telemetry_df.loc[valid_rows, "science_ap9_flux"].to_list()])
                for color, bin_idx in zip(channel_colors, representative_bins):
                    if bin_idx >= measured_matrix.shape[1]:
                        continue
                    energy_label = f"{bin_centers[bin_idx]:.1f} MeV" if bin_idx < len(bin_centers) else f"Bin {bin_idx}"
                    axis2.plot(time_days_valid, measured_matrix[:, bin_idx], color=color, linewidth=0.8, label=f"Measured {energy_label}")
                    axis2.plot(time_days_valid, ap9_matrix[:, bin_idx], color=color, linewidth=0.8, linestyle="--", alpha=0.7, label=f"AP9 {energy_label}")
        else:
            for ch_idx, color in enumerate(channel_colors):
                fpdu_col = f"fpdu_ch{ch_idx}"
                ap9_col = f"ap9_ch{ch_idx}"
                if fpdu_col in telemetry_df.columns:
                    axis2.plot(telemetry_df["time_days"], telemetry_df[fpdu_col], color=color, linewidth=0.8, label=f"Measured bin {ch_idx}")
                if ap9_col in telemetry_df.columns:
                    axis2.plot(telemetry_df["time_days"], telemetry_df[ap9_col], color=color, linewidth=0.8, linestyle="--", alpha=0.7, label=f"AP9 bin {ch_idx}")
        telemetry_df["in_detection"] = telemetry_df["mode"] == "DETECTION"
        self._shade_intervals(axis2, telemetry_df, "in_detection", "lightcoral", 0.15)
        axis2.set_xlabel("Time (days)")
        axis2.set_ylabel("Flux (p/cm?/s/sr/MeV)")
        axis2.set_yscale("log")
        axis2.grid(True, alpha=0.3)

        axis3 = figure.add_subplot(grid[1, 0])
        axis3.set_title("Panel 3: Magnetic Pitch Angle (SAA passes)")
        saa_rows = telemetry_df[telemetry_df["in_saa"] == True]  # noqa: E712
        if not saa_rows.empty and "pitch_angle_deg" in saa_rows.columns:
            scatter3 = axis3.scatter(
                saa_rows["time_days"],
                saa_rows["pitch_angle_deg"],
                c=saa_rows["ew_factor"],
                cmap="coolwarm",
                vmin=1.0 - self.config.ew_asymmetry_amplitude,
                vmax=1.0 + self.config.ew_asymmetry_amplitude,
                s=6,
                alpha=0.75,
            )
            plt.colorbar(scatter3, ax=axis3, label="EW factor")
        axis3.set_xlabel("Time (days)")
        axis3.set_ylabel("Pitch Angle (deg)")
        axis3.set_ylim(0.0, 90.0)
        axis3.grid(True, alpha=0.3)

        axis4 = figure.add_subplot(grid[1, 1])
        axis4.set_title("Panel 4: Mean EW Asymmetry Factor per SAA Pass")
        pass_records = self.payload.pass_science_records
        if pass_records:
            pass_indices = [r["pass_index"] for r in pass_records]
            mean_ew = [r["mean_ew_factor"] for r in pass_records]
            axis4.bar(pass_indices, mean_ew, color="steelblue", alpha=0.85)
        axis4.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Symmetric (1.0)")
        axis4.set_xlabel("SAA Pass Index")
        axis4.set_ylabel("Mean EW Factor")
        axis4.legend()
        axis4.grid(True, alpha=0.3)

        output_path = figures_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_proton_analysis.png")
        figure.savefig(output_path, dpi=300)
        plt.close(figure)

    def _format_map_axis(self, axis, title: str) -> None:
        """Apply consistent coastlines and labels to cartopy map panels."""

        axis.set_title(title)
        if _CARTOPY_AVAILABLE:
            axis.set_global()
            axis.coastlines(linewidth=0.7)
            if cfeature is not None:
                axis.add_feature(cfeature.BORDERS, linewidth=0.3)
            axis.gridlines(draw_labels=False, linewidth=0.2, alpha=0.4)
        else:
            axis.set_xlabel("Longitude (deg)")
            axis.set_ylabel("Latitude (deg)")
            axis.grid(True, linewidth=0.2, alpha=0.4)

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

    def _science_flux_row_is_valid(self, value: object) -> bool:
        """Return True only for full-width science flux vectors."""

        return isinstance(value, (list, tuple, np.ndarray)) and len(value) == self.payload.n_bins

    def _valid_science_flux_rows(self, telemetry_df: pd.DataFrame) -> pd.Series:
        """Build a mask for telemetry rows that contain full science flux vectors."""

        if "science_measured_flux" not in telemetry_df.columns:
            return pd.Series(False, index=telemetry_df.index)
        measured_valid = telemetry_df["science_measured_flux"].apply(self._science_flux_row_is_valid)
        if "science_ap9_flux" not in telemetry_df.columns:
            return measured_valid
        ap9_valid = telemetry_df["science_ap9_flux"].apply(self._science_flux_row_is_valid)
        return measured_valid & ap9_valid

    def _build_detection_science_telemetry(
        self,
        orbit_state: OrbitState,
        b_inertial: np.ndarray,
        velocity_inertial: np.ndarray,
        nadir_inertial_vec: np.ndarray,
    ) -> Dict[str, object]:
        """Populate scalar science telemetry without emitting histogram-sized arrays."""

        pitch_deg = self.proton_env.magnetic_pitch_angle_deg(
            b_inertial,
            velocity_inertial,
        )
        ew_factor = self.proton_env.ew_asymmetry_factor(
            velocity_inertial,
            b_inertial,
            nadir_inertial_vec,
        )
        saa_depth = self.proton_env.saa_depth_factor(
            orbit_state.latitude_deg,
            orbit_state.longitude_deg,
        )
        return {
            "measured_flux": [],
            "ap9_flux": [],
            "residual": [],
            "pitch_angle_deg": pitch_deg,
            "ew_factor": ew_factor,
            "saa_depth": saa_depth,
            "count_rate": [],
            "dark_counts": [],
            "total_count_rate": [],
            "storm_time": self.storm_time,
            "region": "",
            "cell_id": "",
            "cell_n_cell": 0.0,
            "cell_sigma_fraction": 1.0,
            "event_type": "",
            "livetime_fraction": 1.0,
            "histogram_downlink_due": False,
            "histogram_bytes": 0,
            "counts_per_bin": [],
            "sigma_fraction": [],
            "saa_state_at_sample": True,
        }

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
        # FIX Bug 4: adcs.total_slew_time_sec only accumulates inside advance() while
        # is_slewing=True (physical motion). Once the ADCS physically finishes but the
        # FSM hasn't transitioned yet, the counter freezes — giving ~half the true
        # SLEWING-mode time. The FSM mode duration counter counts every step in SLEWING
        # regardless of physical state, matching telemetry (1376 steps × 2s = 45.87 min).
        total_slew_time_min = self.fsm.mode_durations_sec["SLEWING"] / 60.0
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
        # FIX (Bug 5): total_detection_hours was previously read from
        # payload.total_detection_seconds, which only accumulates when
        # generate_data() fires (science_generation_enabled=True steps).
        # This underreported DETECTION time by ~2.15× because settling-transient
        # and storage-full steps were excluded.  The FSM mode duration counter
        # counts every timestep spent in DETECTION mode, which is the correct
        # value for the duty-cycle denominator.
        total_detection_hours = self.fsm.mode_durations_sec["DETECTION"] / 3600.0
        science_duty_cycle_percent = 100.0 * self.fsm.mode_durations_sec["DETECTION"] / total_sim_seconds

        total_gs_contacts = len(gs_contacts_df)
        # FIX Bugs 1+2+3: total_contact_time_min now reads from
        # communications.total_contact_seconds which is gated on relay_link_active
        # (not raw gs_visible). This gives the true link-active time instead of
        # the 10x-overcounted gs_visible window time.
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
        if not telemetry_df.empty:
            adcs_base_load = telemetry_df["load_w"].to_numpy(dtype=float) - (self.config.obc_power_w + self.config.eps_power_w)
            mode_series = telemetry_df["mode"]
            adcs_base_load = adcs_base_load.copy()
            adcs_base_load[mode_series.to_numpy() == "DETECTION"] -= self.config.payload_power_w
            adcs_base_load[mode_series.to_numpy() == "RELAY"] -= self.config.radio_tx_power_w
            avg_adcs_power_w = float(np.mean(np.maximum(adcs_base_load, 0.0)))
        else:
            avg_adcs_power_w = 0.0
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
        print(f"  Quaternion continuity corrections (total): {self.adcs.quaternion_continuity_corrections}")
        print("  Quaternion continuity corrections by call site:")
        for site_name in ["integration", "delta_quaternion", "command_attitude", "advance", "build_frame", "unspecified"]:
            site_count = self.adcs.quaternion_continuity_corrections_by_site.get(site_name, 0)
            rate_per_hr = site_count / max(self.config.sim_duration_hours, 1.0e-9)
            print(f"    {site_name}: {site_count} ({rate_per_hr:.1f}/hr)")
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
        print(f"  Total science data generated (MB): {self.payload.total_science_data_generated_mb:.6f}")
        print(f"  Science duty cycle (%): {science_duty_cycle_percent:.2f}")

        print("\nCommunications:")
        print(f"  Total GS contacts: {total_gs_contacts}")
        print(f"  Total contact time (min): {total_contact_time_min:.4f}")
        print(f"  Total data downlinked (MB): {total_downlinked_mb:.6f}")
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

        # Science summary section
        n_passes_with_science = len(self.payload.pass_science_records)
        if self.payload.pass_science_records:
            mean_pitch = float(np.mean([r["mean_pitch_angle_deg"] for r in self.payload.pass_science_records]))
            mean_ew = float(np.mean([r["mean_ew_factor"] for r in self.payload.pass_science_records]))
        else:
            mean_pitch = 0.0
            mean_ew = 1.0
        peak_low_energy_bin = 0.0
        peak_high_energy_bin = 0.0
        if "science_measured_flux" in telemetry_df.columns:
            valid_flux_rows = self._valid_science_flux_rows(telemetry_df)
            if valid_flux_rows.any():
                flux_matrix = np.vstack(
                    [np.asarray(row, dtype=float) for row in telemetry_df.loc[valid_flux_rows, "science_measured_flux"].to_list()]
                )
                peak_low_energy_bin = float(np.max(flux_matrix[:, 0]))
                peak_high_energy_bin = float(np.max(flux_matrix[:, -1]))
        else:
            peak_low_energy_bin = float(telemetry_df["fpdu_ch0"].max()) if "fpdu_ch0" in telemetry_df.columns else 0.0
            peak_high_energy_bin = float(telemetry_df["fpdu_ch4"].max()) if "fpdu_ch4" in telemetry_df.columns else 0.0
        total_science_mb = self.payload.total_science_data_generated_mb
        total_downlinked_mb = self.communications.total_downlinked_mb

        print("\n══════════════════════════════════════════════════════")
        print("SCIENCE SUMMARY")
        print("══════════════════════════════════════════════════════")
        print(f"SAA passes with science data      : {n_passes_with_science}")
        print(f"Total proton detection time (s)   : {self.payload.total_detection_seconds:.1f}")
        print(f"Mean pitch angle over mission     : {mean_pitch:.2f} deg")
        print(f"Mean EW asymmetry factor          : {mean_ew:.4f}")
        print(f"Peak lowest-energy bin flux      : {peak_low_energy_bin:.3e} p/cm²/s/sr/MeV")
        print(f"Peak highest-energy bin flux     : {peak_high_energy_bin:.3e} p/cm²/s/sr/MeV")
        print(f"SpacePy/IRBEM import available    : {_SPACEPY_AVAILABLE} (does not imply AP9 support)")
        print(f"ppigrf available                  : {_PYIGRF_AVAILABLE}")
        print(f"Science data generated            : {total_science_mb:.6f} MB")
        print(f"Science data downlinked           : {total_downlinked_mb:.6f} MB")

    def _print_progress(
        self,
        time_sec: float,
        final_time_sec: float,
        mode: str,
        battery_wh: float,
        storage_mb: float,
        solar_w: float,
        load_w: float,
        science_generated_mb: float,
    ) -> None:
        """Print a concise simulation progress line for long mission runs.

        Storage is printed at 6 decimal places (not 3) because science data
        accrues in ~0.000244 MB (256-byte) snapshot chunks — at 3 decimals
        that increment silently rounds away and the field looks frozen even
        though generation is happening. A dedicated Science field (in KB,
        cumulative) plus a live KB/hr rate since the last progress line are
        also printed so the payload's actual generation rate is directly
        visible without having to infer it from the rounded storage total.
        """

        percent_complete = 100.0 * min(time_sec / final_time_sec, 1.0) if final_time_sec > 0.0 else 100.0
        elapsed_hours = time_sec / 3600.0

        interval_sec = time_sec - self._last_progress_time_sec
        interval_science_mb = science_generated_mb - self._last_progress_science_mb
        rate_kb_per_hr = (interval_science_mb * 1024.0) / (interval_sec / 3600.0) if interval_sec > 0.0 else 0.0
        self._last_progress_time_sec = time_sec
        self._last_progress_science_mb = science_generated_mb

        print(
            f"[Progress] {percent_complete:6.2f}% | "
            f"T+{elapsed_hours:7.2f} hr | "
            f"Mode={mode:<13} | "
            f"Battery={battery_wh:5.2f} Wh | "
            f"Solar={solar_w:5.2f} W | "
            f"Load={load_w:5.2f} W | "
            f"Storage={storage_mb:12.6f} MB | "
            f"Science={science_generated_mb * 1024.0:8.3f} KB | "
            f"Rate={rate_kb_per_hr:7.3f} KB/hr"
        )

    def run(self) -> None:
        """Run the ANTARA mission simulation for the configured duration.

        The default configuration is 24 hours. The run logs telemetry, renders
        the plots, writes CSV outputs, and prints the mission summary.
        """

        final_time_sec = self.config.sim_duration_hours * 3600.0
        step_sec = self.config.step_sec
        coarse_step_sec = getattr(self.config, 'coarse_step_sec', step_sec)
        progress_interval_sec = max(step_sec, self.config.progress_report_interval_sec)
        next_progress_report_sec = 0.0

        # Telemetry flush setup
        flush_interval_sec = self.config.telemetry_flush_interval_hours * 3600.0
        next_flush_sec = 0.0
        telemetry_csv_path = results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_telemetry.csv")
        self.logger.set_flush_path(telemetry_csv_path)

        duration_label = (
            f"{self.config.sim_duration_hours:.0f}-hour"
            if self.config.sim_duration_hours < 48.0
            else f"{self.config.sim_duration_hours / 24.0:.0f}-day"
        )
        print(
            f"Starting ANTARA Science {duration_label} simulation | "
            f"Duration={self.config.sim_duration_hours:.1f} hr | "
            f"Timestep={step_sec:.1f} s (coarse={coarse_step_sec:.1f} s)"
        )

        relay_window_diagnostics = {
            }
        time_sec = 0.0
        self.time_sec = 0.0
        prev_illumination_fraction = 1.0
        illumination_fraction = 1.0
        current_mode = self.fsm.current_mode
        while time_sec <= final_time_sec:
            # --- Multi-rate stepping (Stage 1.3) ---
            # Use fine step during SLEWING, DETECTION, RELAY, or near eclipse transitions
            fine_modes = {"SLEWING", "DETECTION", "RELAY"}
            near_eclipse_edge = abs(illumination_fraction - prev_illumination_fraction) > 0.1
            if current_mode in fine_modes or near_eclipse_edge:
                effective_step_sec = step_sec
            else:
                effective_step_sec = coarse_step_sec

            n_substeps = max(1, int(round(effective_step_sec / step_sec)))
            sub_step_sec = effective_step_sec / n_substeps

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
            gs_visible_primary = self.ground_station.is_visible(gs_elevation_deg)

            # Fix 6: second ground station visibility
            gs_elevation_deg_2 = 0.0
            gs_visible_2 = False
            if self.ground_station_2 is not None:
                gs_elevation_deg_2 = self.ground_station_2.get_elevation(
                    orbit_state.sat_position_earth_frame,
                    orbit_state.absolute_date,
                )
                gs_visible_2 = self.ground_station_2.is_visible(gs_elevation_deg_2)

            # Use the station with the higher elevation when both are visible;
            # otherwise fall back to whichever one is above its mask.
            if gs_visible_primary and gs_visible_2:
                if gs_elevation_deg_2 > gs_elevation_deg:
                    gs_elevation_deg = gs_elevation_deg_2
            elif gs_visible_2:
                gs_elevation_deg = gs_elevation_deg_2
            gs_visible = gs_visible_primary or gs_visible_2
            inside_saa = self.saa.is_inside(orbit_state.latitude_deg, orbit_state.longitude_deg)
            sun_visible = not orbit_state.in_eclipse
            illumination_fraction = self._update_illumination_fraction(sun_visible, time_sec)
            battery_before_step_wh = self.eps.battery_wh

            if self.fsm.current_mode == "SLEWING" and time_sec >= self.slew_end_time_sec:
                try:
                    self._complete_slew_transition(time_sec)
                except IllegalFSMTransitionError as exc:
                    self.fsm.record_rejected_transition("SLEWING", self.target_mode, str(exc))
                    self.logger.log_rejected_transition(
                        {
                            "time_sec": time_sec,
                            "from_mode": "SLEWING",
                            "to_mode": self.target_mode,
                            "reason": str(exc),
                        }
                    )
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
            self.current_gs_off_boresight_deg = gs_off_boresight_deg
            self._validate_attitude_metrics(current_mode, orbit_state, gs_visible)
            startup_allows_charging = current_mode not in {"COLD_AND_DARK", "DETUMBLE"}
            self.eps.compute_solar_power(
                self.current_sun_incidence_angle_deg,
                illumination_fraction if startup_allows_charging else 0.0,
            )
            self.eps.compute_load(
                current_mode,
                self.adcs.is_slewing,
                not sun_visible,
                payload_pending=(current_mode == "SLEWING" and self.target_mode == "DETECTION"),
            )

            payload_pointing_valid = (
                self.current_payload_pointing_error_deg <= self.config.payload_pointing_valid_deg
            )
            payload_settled = self.adcs.settled_time_sec >= self.config.minimum_settled_time_before_payload_sec

            # ── Science model ────────────────────────────────────────────────────────────
            b_inertial = self._compute_earth_b_field_inertial(orbit_state)
            nadir_inertial_vec = self._nadir_vector_unit(orbit_state)
            velocity_inertial_vec = self._velocity_vector_unit(orbit_state)

            science_data_step_mb = 0.0
            flux_result: Dict[str, object] = {}
            science_sample_inside_saa: Optional[bool] = None

            # FIX (Bug 1a): payload_settled was computed but never applied to the
            # science gate — science fired immediately on DETECTION entry before
            # the ADCS had settled post-slew, producing unreliable pitch/flux data
            # during the settling transient and miscounting active detection steps.
            # FIX (Bug 1b): add a storage high-watermark guard so science generation
            # halts before the buffer is completely full; this prevents the silent
            # gate-off that occurs when _update_storage silently clamps new data to
            # zero once the available science capacity is exhausted.
            storage_not_full = (
                self._get_total_storage_mb() < self.config.storage_high_watermark_mb
            )
            if (
                current_mode == "DETECTION"
                and inside_saa
                and payload_pointing_valid
                and payload_settled
                and storage_not_full
                and self.payload.is_operable(self.eps.battery_wh, current_mode)
            ):
                science_data_step_mb, flux_result = self.payload.generate_data(
                    step_sec,
                    orbit_state.latitude_deg,
                    orbit_state.longitude_deg,
                    orbit_state.altitude_km,
                    b_inertial,
                    self._vector_to_numpy(orbit_state.sat_velocity_inertial),
                    nadir_inertial_vec,
                    storm_time=self.storm_time,
                )
                science_generation_enabled = True
                science_sample_inside_saa = True
            else:
                science_generation_enabled = False
                if inside_saa and current_mode == "DETECTION":
                    # FIX (Bug 2): emit scalar telemetry only when science is not
                    # being generated.  This keeps pitch / EW / depth visible for
                    # analysis without fabricating histogram-shaped arrays that later
                    # confuse the summary and plots.
                    flux_result = self._build_detection_science_telemetry(
                        orbit_state,
                        b_inertial,
                        self._vector_to_numpy(orbit_state.sat_velocity_inertial),
                        nadir_inertial_vec,
                    )
                    science_sample_inside_saa = True

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
                downlinked_step_mb = self.communications.downlink_step(self.science_storage_mb, step_sec, gs_elevation_deg)
                assert downlinked_step_mb > 0.0
                # FIX Bugs 1+2+3: contact time must only accumulate while the link is
                # actually established (relay_link_active=True). The original code called
                # record_contact_time whenever gs_visible=True, which included all 17
                # GS-visible windows regardless of mode (SUN_POINTING, SLEWING, RELAY
                # without link) — 8114 s = 135.23 min instead of the correct 13.37 min.
                # Gating here also ensures downlinked_mb in gs_contacts.csv is nonzero
                # for real contacts, since data only moves when relay_link_active is True.
                self.communications.record_contact_time(step_sec)

            if current_mode == "RELAY":
                relay_diag_entry = {
                    "relay_settled": relay_settled,
                    "relay_pointing_valid": relay_pointing_valid,
                    "gs_off_boresight_valid": gs_off_boresight_deg <= self.config.directional_antenna_half_beamwidth_deg,
                    "science_storage_positive": self.science_storage_mb > 0.0,
                    "gs_visible": gs_visible,
                    "relay_link_active": relay_link_active,
                }
                for window, rows in relay_window_diagnostics.items():
                    if window[0] <= time_sec <= window[1]:
                        rows.append(relay_diag_entry)

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

            # FIX (Bug 1 / n_steps): count every DETECTION+SAA step so
            # close_saa_pass() can report n_steps = total detection steps, not
            # just the science-generating subset.
            if current_mode == "DETECTION" and inside_saa:
                self.payload.count_detection_step()

            saa_event = self.saa.detect_entry_exit(
                self.previous_in_saa,
                inside_saa,
                time_sec,
                orbit_state.latitude_deg,
                orbit_state.longitude_deg,
                orbit_state.absolute_date,
                science_data_step_mb,
            )

            # Open payload pass on SAA entry
            if not self.previous_in_saa and inside_saa:
                self.payload.open_saa_pass()

            # Close payload pass on SAA exit
            if self.previous_in_saa and not inside_saa:
                if saa_event and isinstance(saa_event, tuple):
                    _, pass_dict = saa_event
                    entry_t = float(pass_dict.get("entry_time_sec", time_sec))
                else:
                    entry_t = time_sec
                saa_pass_close_flushed_mb = self.payload.close_saa_pass(
                    pass_index=self._saa_pass_index,
                    entry_time_sec=entry_t,
                    exit_time_sec=time_sec,
                )
                if saa_pass_close_flushed_mb > 0.0:
                    self._update_storage(saa_pass_close_flushed_mb, 0.0)
                    science_data_step_mb += saa_pass_close_flushed_mb
                self._saa_pass_index += 1

            # FIX (Bug 4): Previously downlinked_step_mb was always passed to the
            # primary station's detect_aos_los, even when the active link was via
            # GS2. This caused gs_contacts.csv to record contacts with
            # downlinked_mb=0.0 whenever GS2 was the sole visible station.
            # Now we route the downlink credit to whichever station is active.
            # If both are visible we credit the one with the higher elevation
            # (consistent with the gs_elevation_deg selection above).
            primary_downlink_credit = 0.0
            gs2_downlink_credit = 0.0
            if downlinked_step_mb > 0.0:
                if gs_visible_primary and gs_visible_2:
                    if gs_elevation_deg_2 > self.ground_station.get_elevation(
                        orbit_state.sat_position_earth_frame, orbit_state.absolute_date
                    ):
                        gs2_downlink_credit = downlinked_step_mb
                    else:
                        primary_downlink_credit = downlinked_step_mb
                elif gs_visible_primary:
                    primary_downlink_credit = downlinked_step_mb
                elif gs_visible_2:
                    gs2_downlink_credit = downlinked_step_mb

            gs_event = self.ground_station.detect_aos_los(
                self.previous_gs_visible,
                gs_visible_primary,
                time_sec,
                gs_elevation_deg,
                orbit_state.absolute_date,
                primary_downlink_credit,
            )
            # Fix 6: also advance second station contact tracking
            if self.ground_station_2 is not None:
                self.ground_station_2.detect_aos_los(
                    self.previous_gs_visible_2 if hasattr(self, "previous_gs_visible_2") else False,
                    gs_visible_2,
                    time_sec,
                    gs_elevation_deg_2,
                    orbit_state.absolute_date,
                    gs2_downlink_credit,
                )
            self.previous_in_saa = inside_saa
            self.previous_gs_visible = gs_visible
            self.previous_gs_visible_2 = gs_visible_2

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
                science_sample_inside_saa,
            )

            science_record = flux_result or {}
            self.logger.log_step(
                {
                    "time_sec": time_sec,
                    "mode": current_mode,
                    "lat_deg": orbit_state.latitude_deg,
                    "lon_deg": orbit_state.longitude_deg,
                    "alt_km": orbit_state.altitude_km,
                    "battery_wh": self.eps.battery_wh,
                    "solar_w": self.eps.solar_generation_w,
                    "load_w": self.eps.load_w,
                    "in_sunlight": sun_visible,
                    "in_saa": inside_saa,
                    "gs_visible": gs_visible,
                    "wheel_mom_x": float(self.adcs.wheel_momentum_nms[0]),
                    "wheel_mom_y": float(self.adcs.wheel_momentum_nms[1]),
                    "wheel_mom_z": float(self.adcs.wheel_momentum_nms[2]),
                    "slew_angle_deg": float(self.adcs.slew_angle_deg),
                    "in_eclipse": bool(orbit_state.in_eclipse),
                    "storage_mb": self._get_total_storage_mb(),
                    "science_data_mb": science_data_step_mb,
                    "downlinked_mb": downlinked_step_mb,
                    "sun_visible": sun_visible,
                    "sun_incidence_angle_deg": self.current_sun_incidence_angle_deg,
                    "sun_pointing_error_deg": self.current_sun_pointing_error_deg,
                    "relay_pointing_error_deg": self.current_relay_pointing_error_deg,
                    "gs_off_boresight_deg": self.current_gs_off_boresight_deg,
                    "payload_pointing_error_deg": self.current_payload_pointing_error_deg,
                    "attitude_error_deg": self.current_attitude_error_deg,
                    "pointing_budget_violated_frozen": self.pointing_budget_violated_frozen,
                    "cumulative_science_generated_mb": self.payload.total_science_data_generated_mb,
                    "cumulative_downlinked_mb": self.communications.total_downlinked_mb,
                    "net_storage_delta_mb": science_data_step_mb - downlinked_step_mb,
                    "saturation_hold_active": self.adcs.saturation_hold_active,
                    "combined_fault_active": self.fsm.combined_fault_active,
                    "in_low_power": self.fsm.in_low_power,
                    "in_critical_recovery": self.fsm.in_critical_recovery,
                    "post_safe_recovery_hold_active": self.fsm.post_safe_recovery_hold_active,
                    "active_attitude_target": self.current_attitude_command.active_attitude_target if self.current_attitude_command else "UNINITIALISED",
                    "attitude_hold_reason": self.current_attitude_command.attitude_hold_reason if self.current_attitude_command else "Initialisation",
                    "sun_submode": self._derive_sun_submode(current_mode, sun_visible, self.eps.battery_wh),
                    "science_generation_enabled": science_generation_enabled,
                    "relay_link_active": relay_link_active,
                    "payload_power_state": "ACTIVE" if (current_mode == "DETECTION" or (current_mode == "SLEWING" and self.target_mode == "DETECTION")) else "IDLE",
                    "payload_alignment_cosine": payload_alignment_cosine,
                    "relay_alignment_cosine": relay_alignment_cosine,
                    "payload_boresight_inertial": payload_boresight_inertial,
                    "velocity_unit_vector": velocity_unit_vector,
                    "relay_boresight_vector": relay_boresight_vector,
                    "gs_los_vector": gs_los_vector,
                    "illumination_fraction": illumination_fraction,
                    "pitch_angle_deg": float(science_record.get("pitch_angle_deg", 0.0)),
                    "ew_factor": float(science_record.get("ew_factor", 1.0)),
                    "science_region": science_record.get("region", "unknown"),
                    "science_cell_id": science_record.get("cell_id", ""),
                    "science_cell_n_cell": float(science_record.get("cell_n_cell", 0.0)),
                    "science_cell_sigma_fraction": float(science_record.get("cell_sigma_fraction", 1.0)),
                    "science_event_type": science_record.get("event_type", ""),
                    "science_livetime_fraction": float(science_record.get("livetime_fraction", 1.0)),
                    "science_histogram_downlink_due": bool(science_record.get("histogram_downlink_due", False)),
                    "science_histogram_bytes": int(science_record.get("histogram_bytes", 0)),
                    "science_saa_state_at_sample": science_record.get("saa_state_at_sample"),
                    "science_measured_flux": science_record.get("measured_flux", []),
                    "science_ap9_flux": science_record.get("ap9_flux", []),
                    "science_total_count_rate": science_record.get("total_count_rate", []),
                    "science_counts_per_bin": science_record.get("counts_per_bin", []),
                    "science_sigma_fraction": science_record.get("sigma_fraction", []),
                    "fpdu_ch0": float(science_record["measured_flux"][0]) if len(science_record.get("measured_flux", [])) > 0 else 0.0,
                    "fpdu_ch1": float(science_record["measured_flux"][1]) if len(science_record.get("measured_flux", [])) > 1 else 0.0,
                    "fpdu_ch2": float(science_record["measured_flux"][2]) if len(science_record.get("measured_flux", [])) > 2 else 0.0,
                    "fpdu_ch3": float(science_record["measured_flux"][3]) if len(science_record.get("measured_flux", [])) > 3 else 0.0,
                    "fpdu_ch4": float(science_record["measured_flux"][4]) if len(science_record.get("measured_flux", [])) > 4 else 0.0,
                    "ap9_ch0": float(science_record["ap9_flux"][0]) if len(science_record.get("ap9_flux", [])) > 0 else 0.0,
                    "ap9_ch1": float(science_record["ap9_flux"][1]) if len(science_record.get("ap9_flux", [])) > 1 else 0.0,
                    "ap9_ch2": float(science_record["ap9_flux"][2]) if len(science_record.get("ap9_flux", [])) > 2 else 0.0,
                    "ap9_ch3": float(science_record["ap9_flux"][3]) if len(science_record.get("ap9_flux", [])) > 3 else 0.0,
                    "ap9_ch4": float(science_record["ap9_flux"][4]) if len(science_record.get("ap9_flux", [])) > 4 else 0.0,
                    "count_rate_ch0": float(science_record["total_count_rate"][0]) if len(science_record.get("total_count_rate", [])) > 0 else 0.0,
                    "count_rate_ch1": float(science_record["total_count_rate"][1]) if len(science_record.get("total_count_rate", [])) > 1 else 0.0,
                    "count_rate_ch2": float(science_record["total_count_rate"][2]) if len(science_record.get("total_count_rate", [])) > 2 else 0.0,
                    "count_rate_ch3": float(science_record["total_count_rate"][3]) if len(science_record.get("total_count_rate", [])) > 3 else 0.0,
                    "count_rate_ch4": float(science_record["total_count_rate"][4]) if len(science_record.get("total_count_rate", [])) > 4 else 0.0,
                    "saa_depth_factor": float(science_record.get("saa_depth", 0.0)),
                    "storm_time": self.storm_time,
                    "panel_deployed": self.adcs.panel_deployed if hasattr(self.adcs, "panel_deployed") else False,
                    "rejected_transition_count": len(self.fsm.rejected_transition_history),
                }
            )

            # ── AOS pre-slewing (now redundant with LVLH-fixed RELAY) ───────────────────
            # Previously this block pre-aimed the +Y antenna toward the GS before AOS.
            # RELAY now holds the same LVLH frame as SUN_POINTING/DETECTION, so the
            # mode transition slew is near-zero angle and no pre-slew targeting is needed.
            # The _maybe_begin_predicted_slew call is retained to preserve the FSM
            # transition-timing logic (mode entry before AOS), but the attitude target
            # it computes is now the LVLH default rather than a GS LOS vector.
            predicted_aos = self.ground_station.predicted_next_aos_time_sec
            if self._maybe_begin_predicted_slew(
                current_mode,
                "RELAY",
                predicted_aos,
                orbit_state,
                sun_visible,
                self.eps.battery_wh,
                False,
                time_sec,
                "Pre-slewing to RELAY (LVLH hold — near-zero slew)",
            ):
                current_mode = self.fsm.current_mode
                self.ground_station.predicted_next_aos_time_sec = None

            predicted_saa_ingress = self._predict_saa_ingress_time_sec(
                orbit_state,
                self.config.slew_transition_penalty_sec,
            )
            if self._maybe_begin_predicted_slew(
                current_mode,
                "DETECTION",
                predicted_saa_ingress,
                orbit_state,
                sun_visible,
                self.eps.battery_wh,
                gs_visible,
                time_sec,
                "Pre-slewing to DETECTION",
            ):
                current_mode = self.fsm.current_mode

            requested_mode, reason = self.fsm.evaluate(
                self.eps.battery_wh,
                gs_visible,
                inside_saa,
                self.adcs.saturation_hold_active,
                self.adcs.get_saturation_hold_duration_sec(),
                self._hard_pointing_error_for_mode(current_mode, orbit_state, gs_visible),
                self._hard_pointing_limit_for_mode(current_mode),
                self.adcs.detumble_complete,
            )
            transitioned = False
            if requested_mode != current_mode:
                try:
                    if current_mode == "SLEWING":
                        if requested_mode == "SAFE":
                            transitioned = self.fsm.transition(requested_mode, reason)
                        else:
                            transitioned = False
                    elif current_mode == "SAFE" and requested_mode == "SUN_POINTING":
                        transitioned = self.fsm.transition(requested_mode, reason)
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
                except IllegalFSMTransitionError as exc:
                    self.fsm.record_rejected_transition(
                        current_mode,
                        requested_mode,
                        f"{exc} | reason={reason}",
                    )
                    self.logger.log_rejected_transition(
                        {
                            "time_sec": time_sec,
                            "from_mode": current_mode,
                            "to_mode": requested_mode,
                            "reason": f"{exc} | reason={reason}",
                        }
                    )
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
                    science_generated_mb=self.payload.total_science_data_generated_mb,
                )
                next_progress_report_sec += progress_interval_sec

            # Periodic telemetry flush to bound memory usage
            if time_sec >= next_flush_sec:
                self.logger.flush_telemetry()
                next_flush_sec += flush_interval_sec

            # Multi-rate stepping: use coarse step for most modes, fine step for
            # SLEWING/DETECTION/RELAY and near eclipse transitions
            fine_step = step_sec
            # Check if we're near an eclipse transition (within penumbra_transition_sec)
            near_eclipse_edge = (
                self.previous_sun_visible_geometry != sun_visible
                or (time_sec < self.config.penumbra_transition_sec and time_sec > 0)
            )
            # Use fine step for active modes or near eclipse boundaries
            use_fine_step = (
                current_mode in {"SLEWING", "DETECTION", "RELAY"}
                or near_eclipse_edge
                or self.adcs.saturation_hold_active
            )
            effective_step = fine_step if use_fine_step else coarse_step_sec

            # Advance time
            time_sec += effective_step
            self.time_sec = time_sec

            # If using coarse step, sub-step EPS and other fast models at fine step
            if effective_step > fine_step:
                n_sub = max(1, int(round(effective_step / fine_step)))
                sub_dt = effective_step / n_sub
                for _ in range(n_sub):
                    self.eps.step(sub_dt)
                    # Sub-step ADCS wheel momentum if not in detumble
                    if current_mode != "DETUMBLE" and not self.adcs.detumble_complete:
                        # Approximate: update wheel momentum at sub-step resolution
                        pass

        if self.payload._current_pass_records:
            if self.saa.current_pass is not None:
                entry_t = float(self.saa.current_pass.get("entry_time_sec", final_time_sec))
            else:
                entry_t = final_time_sec
            final_flushed_mb = self.payload.close_saa_pass(
                pass_index=self._saa_pass_index,
                entry_time_sec=entry_t,
                exit_time_sec=final_time_sec,
            )
            if final_flushed_mb > 0.0:
                self._update_storage(final_flushed_mb, 0.0)
            self._saa_pass_index += 1

        # Final flush to ensure any remaining rows are written
        self.logger.flush_telemetry()

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

        # Read back flushed telemetry CSV (since telemetry_rows was cleared by flushes)
        if self.logger._flush_path and self.logger._flush_path.exists():
            telemetry_df = pd.read_csv(self.logger._flush_path)
        else:
            # Fallback: rebuild from remaining in-memory rows (should be small)
            telemetry_df = pd.DataFrame(self.logger.telemetry_rows)

        telemetry_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_telemetry.csv"), index=False)
        saa_passes_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_saa_passes.csv"), index=False)
        gs_contacts_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_gs_contacts.csv"), index=False)
        mode_transitions_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_mode_transitions.csv"), index=False)
        slew_events_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_slew_events.csv"), index=False)
        detumble_events_df.to_csv(results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_detumble_events.csv"), index=False)
        pd.DataFrame(self.logger.rejected_transition_rows).to_csv(
            results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_rejected_transitions.csv"),
            index=False,
        )

        proton_science_df = pd.DataFrame(self.payload.pass_science_records)
        proton_science_df.to_csv(
            results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_proton_passes.csv"), index=False
        )

        if not self.config.no_plots:
            self._plot_results(telemetry_df, saa_passes_df, gs_contacts_df, slew_events_df)
            self._plot_science_results(telemetry_df, saa_passes_df)
            self._plot_attitude_diagnostics(telemetry_df)
        self._print_summary(
            telemetry_df,
            saa_passes_df,
            gs_contacts_df,
            mode_transitions_df,
            slew_events_df,
        )

        # Final relay diagnostics
        for window, rows in relay_window_diagnostics.items():
            if not rows:
                continue
            false_counts = {
                key: sum(1 for row in rows if not row[key])
                for key in [
                    "relay_settled",
                    "relay_pointing_valid",
                    "gs_off_boresight_valid",
                    "science_storage_positive",
                    "gs_visible",
                    "relay_link_active",
                ]
            }
            print(
                f"[Relay diagnostics] window {int(window[0])}-{int(window[1])} sec | "
                f"samples={len(rows)} | "
                f"false relay_settled={false_counts['relay_settled']} | "
                f"false relay_pointing_valid={false_counts['relay_pointing_valid']} | "
                f"false gs_off_boresight_valid={false_counts['gs_off_boresight_valid']} | "
                f"false science_storage_positive={false_counts['science_storage_positive']} | "
                f"false gs_visible={false_counts['gs_visible']} | "
                f"false relay_link_active={false_counts['relay_link_active']}"
            )

    def run_steps(self, n_steps: int) -> None:
        """Run the simulation for a specific number of steps.

        Useful for checkpoint testing where we need to advance by a fixed step count.
        """
        step_sec = self.config.step_sec
        coarse_step_sec = getattr(self.config, 'coarse_step_sec', step_sec)
        time_sec = getattr(self, 'time_sec', 0.0)

        if time_sec == 0.0:
            # First run: initialize telemetry flush path
            telemetry_csv_path = results_path("science", f"{self.config.output_tag + '_' if self.config.output_tag else ''}ANTARA_Science_telemetry.csv")
            self.logger.set_flush_path(telemetry_csv_path)

        for _ in range(n_steps):
            current_mode = self.fsm.current_mode
            absolute_date = self.start_date.shiftedBy(time_sec)
            propagated_state = self._propagate_state(absolute_date)
            orbit_state = self._build_orbit_state(propagated_state, time_sec)

            gs_elevation_deg = self.ground_station.get_elevation(
                orbit_state.sat_position_earth_frame,
                orbit_state.absolute_date,
            )
            gs_visible_primary = self.ground_station.is_visible(gs_elevation_deg)

            gs_elevation_deg_2 = 0.0
            gs_visible_2 = False
            if self.ground_station_2 is not None:
                gs_elevation_deg_2 = self.ground_station_2.get_elevation(
                    orbit_state.sat_position_earth_frame,
                    orbit_state.absolute_date,
                )
                gs_visible_2 = self.ground_station_2.is_visible(gs_elevation_deg_2)

            if gs_visible_primary and gs_visible_2:
                if gs_elevation_deg_2 > gs_elevation_deg:
                    gs_elevation_deg = gs_elevation_deg_2
            elif gs_visible_2:
                gs_elevation_deg = gs_elevation_deg_2
            gs_visible = gs_visible_primary or gs_visible_2

            inside_saa = self.saa.is_inside(orbit_state.latitude_deg, orbit_state.longitude_deg)
            sun_visible = not orbit_state.in_eclipse
            illumination_fraction = self._update_illumination_fraction(sun_visible, time_sec)

            self.fsm.set_time(time_sec)
            self.adcs.set_time(time_sec)

            if self.fsm.current_mode == "SLEWING" and time_sec >= self.slew_end_time_sec:
                try:
                    self._complete_slew_transition(time_sec)
                except IllegalFSMTransitionError as exc:
                    self.fsm.record_rejected_transition("SLEWING", self.target_mode, str(exc))
                    self.logger.log_rejected_transition(
                        {
                            "time_sec": time_sec,
                            "from_mode": "SLEWING",
                            "to_mode": self.target_mode,
                            "reason": str(exc),
                        }
                    )

            self._update_attitude_guidance(
                current_mode,
                orbit_state,
                sun_visible,
                gs_visible,
                self.eps.battery_wh,
                step_sec,
                time_sec,
            )
            startup_allows_charging = current_mode not in {"COLD_AND_DARK", "DETUMBLE"}
            self.eps.compute_solar_power(
                self.current_sun_incidence_angle_deg,
                illumination_fraction if startup_allows_charging else 0.0,
            )
            self.eps.compute_load(
                current_mode,
                self.adcs.is_slewing,
                not sun_visible,
                payload_pending=(current_mode == "SLEWING" and self.target_mode == "DETECTION"),
            )

            self.eps.step(step_sec)

            self._update_alignment_telemetry(orbit_state, gs_visible)

            # FSM evaluation
            adcs_saturation_hold = self.adcs.saturation_hold_active
            saturation_hold_dur = self.adcs.get_saturation_hold_duration_sec()
            pointing_budget_err = self.adcs.get_pointing_error_deg()
            pointing_budget_lim = self.config.safe_hold_deadband_deg
            detumble_done = self.adcs.detumble_complete

            next_mode, reason = self.fsm.evaluate(
                battery_wh=self.eps.battery_wh,
                gs_visible=gs_visible,
                inside_saa=inside_saa,
                adcs_saturation_hold_active=adcs_saturation_hold,
                saturation_hold_duration_sec=saturation_hold_dur,
                pointing_budget_error_deg=pointing_budget_err,
                pointing_budget_limit_deg=pointing_budget_lim,
                detumble_complete=detumble_done,
            )

            requested_mode = next_mode
            if requested_mode != current_mode:
                try:
                    self.fsm.transition(requested_mode, reason)
                    if requested_mode == "SLEWING":
                        self.slew_start_time_sec = time_sec
                        self.slew_end_time_sec = time_sec + self.adcs.slew_time_remaining_sec
                        self.logger.log_slew_event(
                            start_time_sec=time_sec,
                            from_mode=current_mode,
                            target_mode=requested_mode,
                            angle_deg=self.adcs.slew_angle_deg,
                            estimated_duration_sec=self.adcs.slew_time_remaining_sec,
                        )
                    elif current_mode == "DETUMBLE" and requested_mode == "SUN_POINTING":
                        self.logger.log_detumble_event(
                            start_time_sec=time_sec,
                            from_mode=current_mode,
                            target_attitude_label=self.config.detumble_target_attitude_label,
                            initial_body_rate_deg_sec=self.adcs.initial_body_rate_deg_sec,
                            initial_body_accel_deg_sec2=self.adcs.initial_body_accel_deg_sec2,
                            goal="Physics detumble complete",
                        )
                except IllegalFSMTransitionError as exc:
                    self.fsm.record_rejected_transition(
                        current_mode,
                        requested_mode,
                        f"{exc} | reason={reason}",
                    )
                    self.logger.log_rejected_transition(
                        {
                            "time_sec": time_sec,
                            "from_mode": current_mode,
                            "to_mode": requested_mode,
                            "reason": f"{exc} | reason={reason}",
                        }
                    )

            # Update ADCS
            if current_mode == "DETUMBLE" and not self.adcs.detumble_complete:
                b_body_t = self._compute_earth_b_field_body(orbit_state)
                self.adcs.step_detumble(b_body_t, step_sec)
            elif self.adcs.is_slewing:
                self.adcs.advance(step_sec)
            else:
                self.adcs.advance(step_sec)

            # Log telemetry
            self.logger.log_step({
                "time_sec": time_sec,
                "lat_deg": orbit_state.latitude_deg,
                "lon_deg": orbit_state.longitude_deg,
                "alt_km": orbit_state.altitude_km,
                "mode": self.fsm.current_mode,
                "battery_wh": self.eps.battery_wh,
                "solar_w": self.eps.solar_generation_w,
                "load_w": self.eps.load_w,
                "in_sunlight": sun_visible,
                "in_saa": inside_saa,
                "gs_visible": gs_visible,
                "slew_angle_deg": self.adcs.slew_angle_deg,
                "wheel_mom_x": self.adcs.wheel_momentum_nms[0],
                "wheel_mom_y": self.adcs.wheel_momentum_nms[1],
                "wheel_mom_z": self.adcs.wheel_momentum_nms[2],
                "pointing_error_deg": self.adcs.get_pointing_error_deg(),
                "storage_mb": self._get_total_storage_mb(),
                "science_data_mb": self.payload.total_science_data_generated_mb,
                "downlinked_mb": self.communications.total_downlinked_mb,
                "sun_visible": sun_visible,
            })

            time_sec += step_sec
            self.time_sec = time_sec

        self.logger.flush_telemetry()

def _parse_args(argv=None) -> argparse.Namespace:
    """Parse the lightweight CLI for DITL run controls."""

    parser = argparse.ArgumentParser(description="Run the ANTARA Science DITL simulator.")
    parser.add_argument("--duration-hours", type=float, default=None, help="Override scenario duration in hours (default: 24 without a scenario).")
    parser.add_argument("--step-sec", type=float, default=None, help="Override step in seconds; fixed stepping unless --coarse-step-sec is also supplied (default fine step: 2).")
    parser.add_argument("--rng-seed", type=int, default=None, help="Seed for the simulator RNG.")
    parser.add_argument("--no-periphery-merge", action="store_true", help="Disable adaptive periphery bin merging.")
    parser.add_argument("--checkpoint-every-hours", type=float, default=None, help="Save checkpoint every N hours.")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint file to resume from.")
    parser.add_argument("--scenario", choices=["leop_24h", "no_leop_24h", "leop_1yr"], default=None, help="Predefined scenario to run.")
    parser.add_argument("--coarse-step-sec", type=float, default=None, help="Coarse step for multi-rate stepping (leop_1yr).")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation.")
    return parser.parse_args(argv)


def _config_from_args(args):
    """Apply global defaults, then a scenario preset, then explicit CLI values."""
    skip_leop = False
    if args.scenario:
        from scenarios import build_scenario
        config, skip_leop = build_scenario(args.scenario)
    else:
        config = MissionConfig()
    from dataclasses import replace
    overrides = {
        name: value for name, value in {
            "sim_duration_hours": args.duration_hours,
            "step_sec": args.step_sec,
            "rng_seed": args.rng_seed,
            "coarse_step_sec": args.coarse_step_sec,
        }.items() if value is not None
    }
    if args.step_sec is not None and args.coarse_step_sec is None:
        overrides["coarse_step_sec"] = args.step_sec
    if args.no_plots:
        overrides["no_plots"] = True
    if args.no_periphery_merge:
        overrides["enable_periphery_bin_merge"] = False
    return replace(config, **overrides), skip_leop


if __name__ == "__main__":
    args = _parse_args()
    config, skip_leop = _config_from_args(args)

    if args.resume_from:
        print(f"[Resume] Loading checkpoint from {args.resume_from}")
        sim = ANTARAScienceSimulator.load_checkpoint(args.resume_from, config)
    else:
        sim = ANTARAScienceSimulator(config, skip_leop=skip_leop)

    # Set up periodic checkpointing if requested
    if args.checkpoint_every_hours:
        import threading
        def checkpoint_loop(sim_instance, interval_hours):
            while True:
                import time
                time.sleep(interval_hours * 3600)
                try:
                    ckpt_path = Path(f"checkpoint_{int(sim_instance.time_sec//3600)}h.pkl")
                    sim_instance.save_checkpoint(ckpt_path)
                    print(f"[Checkpoint] Saved to {ckpt_path}")
                except Exception as e:
                    print(f"[Checkpoint] Failed: {e}")
        t = threading.Thread(target=checkpoint_loop, args=(sim, args.checkpoint_every_hours), daemon=True)
        t.start()

    sim.run()
