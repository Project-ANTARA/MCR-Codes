from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import orekit

orekit.initVM()

from orekit import JArray_double
import pandas as pd

# ---------------------------------------------------------------------------
# Project path helpers. In the ANTARA project these live in paths.py and are
# shared with ANTARA_Science_Sim.py (figures_path/results_path/init_orekit_data).
# orekit.initVM() is called explicitly above (before this import), so this
# works regardless of whether paths.py itself boots the JVM as a side effect.
# init_orekit_data() (either the imported one or the local fallback) is
# always called once, right after this block, to load the Orekit data zip
# before any `org.orekit...` symbol is touched below.
# ---------------------------------------------------------------------------
try:
    from paths import figures_path, init_orekit_data, results_path
except ImportError:
    import os

    def _local_path(root: str, category: str, filename: str) -> str:
        directory = os.path.join(root, category)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, filename)

    def results_path(category: str, filename: str) -> str:
        return _local_path("results", category, filename)

    def figures_path(category: str, filename: str) -> str:
        return _local_path("figures", category, filename)

    def init_orekit_data() -> None:
        from orekit.pyhelpers import setup_orekit_curdir

        setup_orekit_curdir()

init_orekit_data()

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    _MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not found. Plot generation will be skipped.")

from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import (
    IsotropicRadiationSingleCoefficient,
    SolarRadiationPressure,
)
from org.orekit.frames import FramesFactory
from org.orekit.models.earth.atmosphere import DTM2000, NRLMSISE00
from org.orekit.models.earth.atmosphere.data import (
    CssiSpaceWeatherData,
    MarshallSolarActivityFutureEstimation,
)
from org.orekit.orbits import KeplerianOrbit, OrbitType, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants, IERSConventions


# ---------------------------------------------------------------------------
# Authoritative spacecraft body frame convention (ICD Rev B, shared with the
# ANTARA DIL sim):
#   +X = RAM face / payload boresight   -> aligned to velocity vector (LVLH)
#   +Y = directional relay antenna face -> aligned to nadir (LVLH)
#   +Z = fixed solar panel              -> aligned to orbit normal, sun-side
#   -Y = deployable panel packed face
#   -Z = omni antenna face
#   -X = PPOD deployment face
# ---------------------------------------------------------------------------
BODY_AXIS_VECTORS = {
    "+X": np.array([1.0, 0.0, 0.0], dtype=float),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=float),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=float),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=float),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=float),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=float),
}

FACE_ORDER = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# Structural adjacency used for the face-to-face conduction network. Each
# face is rail-bolted to its four neighbors; the opposite face is not
# directly adjacent (heat only reaches it via the internal node or two hops).
FACE_ADJACENCY = {
    "+X": ["+Y", "-Y", "+Z", "-Z"],
    "-X": ["+Y", "-Y", "+Z", "-Z"],
    "+Y": ["+X", "-X", "+Z", "-Z"],
    "-Y": ["+X", "-X", "+Z", "-Z"],
    "+Z": ["+X", "-X", "+Y", "-Y"],
    "-Z": ["+X", "-X", "+Y", "-Y"],
}


def _as_flat_vector(vector) -> np.ndarray:
    """Return a canonical flat 3-vector for geometry and telemetry use.

    Single shape-normalisation boundary for Orekit Vector3D objects, Python
    sequences, and NumPy arrays with shapes (3,), (3, 1), or (1, 3).
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


# ---------------------------------------------------------------------------
# Face thermal properties. One dataclass instance per body face. All optical
# and mass/conductance values below are representative CubeSat engineering
# estimates (typical anodized-aluminum / solar-cell / bare-metal properties)
# and are called out as TBD-refine-with-CAD/FEA — they are NOT measured
# ANTARA hardware data.
# ---------------------------------------------------------------------------
@dataclass
class FaceThermalProperties:
    """Thermal-optical and lumped-capacitance properties of a single face.

    solar_absorptivity (alpha_bol): beginning-of-life fraction of incident
        solar flux absorbed.
    solar_absorptivity_eol_delta: additional absorptivity gained by
        end-of-life from on-orbit UV/atomic-oxygen darkening. Effective
        alpha = alpha_bol (+ delta if mission_life_condition == "EOL").
    ir_emissivity (epsilon): emissivity in the IR band, used both for the
        Earth-IR absorption term and for radiative rejection to deep space.
    cell_efficiency: fraction of *absorbed* solar power converted to
        electricity rather than heat (0 for non-solar-cell faces).
    area_m2: projected flat-plate area of the face.
    mass_kg / specific_heat_j_kgk: lumped thermal mass of the face node.
    conductance_to_internal_w_per_k: conductive path (G, W/K) linking this
        face node to the shared internal (bus) thermal node.
    """

    label: str
    normal_body: np.ndarray
    area_m2: float
    solar_absorptivity: float
    solar_absorptivity_eol_delta: float
    ir_emissivity: float
    cell_efficiency: float
    mass_kg: float
    specific_heat_j_kgk: float
    conductance_to_internal_w_per_k: float

    @property
    def thermal_capacitance_j_per_k(self) -> float:
        return self.mass_kg * self.specific_heat_j_kgk

    def effective_solar_absorptivity(self, mission_life_condition: str) -> float:
        alpha = self.solar_absorptivity
        if mission_life_condition.upper() == "EOL":
            alpha += self.solar_absorptivity_eol_delta
        return min(alpha, 1.0)


def _default_face_properties() -> Dict[str, FaceThermalProperties]:
    """Build the default 6-face thermal property set for a 3U CubeSat.

    Geometry assumption: elongated 3U bus (10 x 10 x 34 cm) with the long
    axis along body X (RAM / PPOD deployment axis).
        +X / -X (end caps):  0.10 m x 0.10 m = 0.010 m^2
        +Y / -Y / +Z / -Z (long faces): 0.10 m x 0.34 m = 0.034 m^2
    """

    end_cap_area_m2 = 0.10 * 0.10
    long_face_area_m2 = 0.10 * 0.34

    return {
        "+X": FaceThermalProperties(
            label="+X (RAM / payload boresight)",
            normal_body=BODY_AXIS_VECTORS["+X"],
            area_m2=end_cap_area_m2,
            solar_absorptivity=0.45,        # black-anodized baffle around aperture
            solar_absorptivity_eol_delta=0.10,
            ir_emissivity=0.85,
            cell_efficiency=0.0,
            mass_kg=0.077,
            specific_heat_j_kgk=900.0,       # aluminum 6061-T6
            conductance_to_internal_w_per_k=0.15,
        ),
        "-X": FaceThermalProperties(
            label="-X (PPOD deployment face)",
            normal_body=BODY_AXIS_VECTORS["-X"],
            area_m2=end_cap_area_m2,
            solar_absorptivity=0.30,        # bare / lightly-anodized rail feet
            solar_absorptivity_eol_delta=0.05,
            ir_emissivity=0.10,
            cell_efficiency=0.0,
            mass_kg=0.077,
            specific_heat_j_kgk=900.0,
            conductance_to_internal_w_per_k=0.15,
        ),
        "+Y": FaceThermalProperties(
            label="+Y (relay antenna face)",
            normal_body=BODY_AXIS_VECTORS["+Y"],
            area_m2=long_face_area_m2,
            solar_absorptivity=0.25,        # white paint / anodize
            solar_absorptivity_eol_delta=0.10,
            ir_emissivity=0.85,
            cell_efficiency=0.0,
            mass_kg=0.262,
            specific_heat_j_kgk=900.0,
            conductance_to_internal_w_per_k=0.30,
        ),
        "-Y": FaceThermalProperties(
            label="-Y (deployable panel, packed)",
            normal_body=BODY_AXIS_VECTORS["-Y"],
            area_m2=long_face_area_m2,
            solar_absorptivity=0.85,        # solar cell coverglass, packed side out
            solar_absorptivity_eol_delta=0.08,
            ir_emissivity=0.82,
            cell_efficiency=0.29,
            mass_kg=0.262,
            specific_heat_j_kgk=900.0,
            conductance_to_internal_w_per_k=0.30,
        ),
        "+Z": FaceThermalProperties(
            label="+Z (fixed solar panel, sun-side)",
            normal_body=BODY_AXIS_VECTORS["+Z"],
            area_m2=long_face_area_m2,
            solar_absorptivity=0.92,        # solar cell coverglass
            solar_absorptivity_eol_delta=0.08,
            ir_emissivity=0.85,
            cell_efficiency=0.29,
            mass_kg=0.262,
            specific_heat_j_kgk=900.0,
            conductance_to_internal_w_per_k=0.30,
        ),
        "-Z": FaceThermalProperties(
            label="-Z (omni antenna face)",
            normal_body=BODY_AXIS_VECTORS["-Z"],
            area_m2=long_face_area_m2,
            solar_absorptivity=0.35,        # anodized aluminum
            solar_absorptivity_eol_delta=0.10,
            ir_emissivity=0.80,
            cell_efficiency=0.0,
            mass_kg=0.262,
            specific_heat_j_kgk=900.0,
            conductance_to_internal_w_per_k=0.30,
        ),
    }


@dataclass
class ThermalMissionConfig:
    """Top-level dataclass holding all orbit, spacecraft and thermal-model
    parameters. All numeric constants live here; no magic numbers in logic."""

    sim_duration_hours: float = 24.0
    step_sec: float = 2.0
    progress_report_interval_sec: float = 600.0
    start_year: int = 2000
    start_month: int = 3
    start_day: int = 1

    # Orbit definition (matches ANTARA_Science_Sim.py nominal orbit)
    altitude_m: float = 600_000.0
    inclination_deg: float = 97.6
    eccentricity: float = 0.001
    raan_deg: float = 15.0
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0

    # Spacecraft physical properties (for force-model geometry, not thermal)
    mass_kg: float = 3.0
    drag_coeff: float = 2.2
    cross_section_m2: float = 0.01
    srp_coeff: float = 1.8

    # Numerical integrator tuning for the Orekit propagator
    min_integrator_step_sec: float = 0.1
    max_integrator_step_sec: float = 120.0
    integrator_position_tolerance_m: float = 10.0
    integrator_velocity_tolerance_mps: float = 0.01

    # --- Thermal environment constants --------------------------------
    solar_constant_w_m2: float = 1361.0          # reference value at 1 AU;
                                                  # scaled by true Sun-Earth
                                                  # distance (from Orekit
                                                  # ephemeris) every step, so
                                                  # perihelion/aphelion is
                                                  # captured automatically
                                                  # via start_year/month/day.
    deep_space_temp_k: float = 2.7
    stefan_boltzmann_w_m2_k4: float = 5.670374419e-8

    # Latitude-dependent Earth albedo / IR (used when
    # use_latitude_dependent_environment=True). Falls back to the flat
    # earth_bond_albedo / earth_ir_flux_w_m2 constants otherwise.
    use_latitude_dependent_environment: bool = True
    earth_bond_albedo: float = 0.30              # flat fallback value
    earth_ir_flux_w_m2: float = 237.0            # flat fallback value
    earth_bond_albedo_equator: float = 0.25      # ocean/cloud-average, low latitude
    earth_bond_albedo_pole: float = 0.62         # ice/snow-average, high latitude
    earth_ir_equator_w_m2: float = 260.0
    earth_ir_pole_w_m2: float = 205.0

    # --- Mission life condition (BOL vs EOL optical degradation) ------
    mission_life_condition: str = "BOL"          # "BOL" or "EOL"

    # --- Internal (bus) thermal node -----------------------------------
    internal_mass_kg: float = 1.8
    internal_specific_heat_j_kgk: float = 800.0   # mixed electronics/battery/structure

    # Synthetic time-varying duty-cycle power profile. This is a
    # REPRESENTATIVE stand-in for the DIL sim's actual EPS/FSM mode
    # timeline — swap _internal_dissipation_w() to pull the real mode
    # schedule for mission-grade accuracy.
    baseline_bus_dissipation_w: float = 1.5       # always-on housekeeping load
    charging_bonus_dissipation_w: float = 0.4     # extra EPS loss while sunlit/charging
    eclipse_heater_dissipation_w: float = 0.8     # thermostatic heater load in eclipse
    relay_pass_dissipation_w: float = 3.0         # radio Tx power during a ground pass
    relay_pass_duration_sec: float = 420.0        # ~7 minute pass
    detection_window_dissipation_w: float = 1.2   # payload active power
    detection_window_duration_sec: float = 900.0  # ~15 minute science window

    # --- Face-to-face structural conduction ---------------------------
    face_to_face_conductance_w_per_k: float = 0.05

    # --- Attitude mode --------------------------------------------------
    # "LVLH": nominal controlled attitude (+X=RAM, +Y=nadir, +Z=sun-side
    #         orbit normal), solved every step from the propagated state.
    # "TUMBLING": worst-case uncontrolled/pre-detumble bound. The body axes
    #         spin about a fixed inertial axis at tumble_rate_deg_s, so no
    #         single face is preferentially sun- or nadir-facing.
    attitude_mode: str = "LVLH"
    tumble_rate_deg_s: float = 3.0
    tumble_axis_inertial: Tuple[float, float, float] = (0.3, 0.5, 0.8)

    # --- Numerical integration ------------------------------------------
    thermal_substeps_per_step: int = 10
    initial_temperature_k: float = 293.15  # 20 C

    # --- Predicted-vs-qualification margin ------------------------------
    applied_margin_c: float = 10.0

    @property
    def internal_thermal_capacitance_j_per_k(self) -> float:
        return self.internal_mass_kg * self.internal_specific_heat_j_kgk

    @property
    def orbit_period_sec(self) -> float:
        semi_major_axis_m = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + self.altitude_m
        return 2.0 * math.pi * math.sqrt(semi_major_axis_m ** 3 / Constants.WGS84_EARTH_MU)

    def __post_init__(self) -> None:
        errors: List[str] = []
        if not math.isfinite(self.sim_duration_hours) or self.sim_duration_hours <= 0.0:
            errors.append(f"sim_duration_hours={self.sim_duration_hours} must be > 0")
        if not math.isfinite(self.step_sec) or self.step_sec <= 0.0:
            errors.append(f"step_sec={self.step_sec} must be > 0")
        if self.thermal_substeps_per_step < 1:
            errors.append(f"thermal_substeps_per_step={self.thermal_substeps_per_step} must be >= 1")
        if not (0.0 <= self.eccentricity < 1.0):
            errors.append(f"eccentricity={self.eccentricity} must be in [0, 1)")
        if not (0.0 <= self.earth_bond_albedo <= 1.0):
            errors.append(f"earth_bond_albedo={self.earth_bond_albedo} must be in [0, 1]")
        if self.altitude_m <= 0.0:
            errors.append(f"altitude_m={self.altitude_m} must be > 0")
        if self.mission_life_condition.upper() not in {"BOL", "EOL"}:
            errors.append(f"mission_life_condition={self.mission_life_condition} must be 'BOL' or 'EOL'")
        if self.attitude_mode.upper() not in {"LVLH", "TUMBLING"}:
            errors.append(f"attitude_mode={self.attitude_mode} must be 'LVLH' or 'TUMBLING'")
        if np.linalg.norm(np.asarray(self.tumble_axis_inertial, dtype=float)) <= 0.0:
            errors.append("tumble_axis_inertial must be a non-zero vector")
        if errors:
            raise ValueError("Invalid ThermalMissionConfig — " + "; ".join(errors))


@dataclass
class OrbitState:
    """Propagated orbital state at a single timestep: position, eclipse
    flag, beta angle, and inertial position/velocity for face-flux geometry."""

    absolute_date: AbsoluteDate
    time_sec: float
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    in_eclipse: bool
    beta_angle_deg: float
    sun_distance_au: float
    sat_position_inertial: Vector3D
    sat_velocity_inertial: Vector3D


# ---------------------------------------------------------------------------
# Bounding-case config factory (suggestion #1): tunes the parameters that
# actually drive hot/cold worst cases, rather than relying on one nominal
# run. HOT and COLD each bias every uncertain input in the conservative
# direction; NOMINAL leaves the base config untouched.
# ---------------------------------------------------------------------------
def build_bounding_case_config(case: str, base_config: Optional[ThermalMissionConfig] = None) -> ThermalMissionConfig:
    case = case.upper()
    base_config = base_config or ThermalMissionConfig()

    if case == "NOMINAL":
        return base_config

    if case == "HOT":
        cfg = replace(
            base_config,
            start_year=2000, start_month=1, start_day=3,   # near Earth perihelion -> peak solar constant
            mission_life_condition="EOL",                   # degraded (higher) absorptivity
            earth_bond_albedo=min(base_config.earth_bond_albedo + 0.05, 1.0),
            earth_bond_albedo_equator=min(base_config.earth_bond_albedo_equator + 0.05, 1.0),
            earth_bond_albedo_pole=min(base_config.earth_bond_albedo_pole + 0.05, 1.0),
            baseline_bus_dissipation_w=base_config.baseline_bus_dissipation_w + 0.5,
        )
    elif case == "COLD":
        cfg = replace(
            base_config,
            start_year=2000, start_month=7, start_day=4,   # near Earth aphelion -> minimum solar constant
            mission_life_condition="BOL",                   # clean/undegraded (lower) absorptivity
            earth_bond_albedo=max(base_config.earth_bond_albedo - 0.05, 0.0),
            earth_bond_albedo_equator=max(base_config.earth_bond_albedo_equator - 0.05, 0.0),
            earth_bond_albedo_pole=max(base_config.earth_bond_albedo_pole - 0.05, 0.0),
            baseline_bus_dissipation_w=max(base_config.baseline_bus_dissipation_w - 0.5, 0.0),
            relay_pass_dissipation_w=0.0,     # bound the cold case with payload/radio OFF (safe mode)
            detection_window_dissipation_w=0.0,
        )
    else:
        raise ValueError(f"Unknown bounding case '{case}'. Use HOT, COLD, or NOMINAL.")

    return cfg


class ThermalSimulator:
    """Propagates the ANTARA orbit with Orekit and integrates a 7-node
    (6 face + 1 internal bus) lumped-capacitance thermal network alongside
    it, logging per-face solar / albedo / Earth-IR flux and temperature at
    every timestep."""

    def __init__(self, config: Optional[ThermalMissionConfig] = None) -> None:
        self.config = config or ThermalMissionConfig()
        self.faces: Dict[str, FaceThermalProperties] = _default_face_properties()

        self.utc = None
        self.start_date = None
        self.inertial_frame = None
        self.earth_frame = None
        self.earth = None
        self.sun = None
        self.propagator = None

        self._init_orekit_frames()
        self._init_propagator()
        self._build_earth_view_factor_table()

        # Thermal state: one temperature (K) per face plus "INTERNAL"
        self.temperatures_k: Dict[str, float] = {
            face: self.config.initial_temperature_k for face in FACE_ORDER
        }
        self.temperatures_k["INTERNAL"] = self.config.initial_temperature_k

        self._au_m = (
            Constants.IAU_2015_ASTRONOMICAL_UNIT
            if hasattr(Constants, "IAU_2015_ASTRONOMICAL_UNIT")
            else 1.495978707e11
        )

        self._tumble_axis_unit = self._unit_numpy(np.asarray(self.config.tumble_axis_inertial, dtype=float))

    # ------------------------------------------------------------------
    # Orekit setup (mirrors ANTARA_Science_Sim.py force-model configuration)
    # ------------------------------------------------------------------
    def _init_orekit_frames(self) -> None:
        self.utc = TimeScalesFactory.getUTC()
        self.start_date = AbsoluteDate(
            self.config.start_year,
            self.config.start_month,
            self.config.start_day,
            0, 0, 0.0,
            self.utc,
        )
        self.inertial_frame = FramesFactory.getEME2000()
        self.earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self.earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            self.earth_frame,
        )
        self.sun = CelestialBodyFactory.getSun()

    def _build_initial_orbit(self, raan_deg: Optional[float] = None) -> KeplerianOrbit:
        semi_major_axis_m = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + self.config.altitude_m
        eccentricity = float(min(max(abs(self.config.eccentricity), 0.0), 0.999999999))
        raan = self.config.raan_deg if raan_deg is None else raan_deg
        return KeplerianOrbit(
            semi_major_axis_m,
            eccentricity,
            math.radians(self.config.inclination_deg),
            math.radians(self.config.arg_perigee_deg),
            math.radians(raan),
            math.radians(self.config.true_anomaly_deg),
            PositionAngleType.TRUE,
            self.inertial_frame,
            self.start_date,
            Constants.WGS84_EARTH_MU,
        )

    def _build_atmosphere(self):
        """Construct an Orekit atmosphere model, preferring NRLMSISE-00 and
        falling back to DTM2000 (same fallback chain as the DIL sim)."""

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
            marshall_data = MarshallSolarActivityFutureEstimation("Jan2000F10.txt", strength)
            return DTM2000(marshall_data, self.sun, self.earth)
        except Exception as exc:
            atmosphere_errors.append(f"DTM2000 attempt failed: {exc}")

        raise RuntimeError(
            "Unable to construct atmospheric density model.\n" + "\n".join(atmosphere_errors)
        )

    def _build_srp_model(self):
        """Construct the SRP model with Earth occultation for eclipse-aware
        illumination (used for orbit propagation force accuracy; the thermal
        eclipse flag itself comes from _is_in_eclipse_geometry)."""

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

    def _coerce_double_array(self, array_like):
        """Cast an existing Java double[] (as returned by NumericalPropagator
        .tolerances()) to the typed JArray_double JCC needs for overload
        resolution. Falls back to the raw value for plain Python sequences."""

        try:
            return JArray_double.cast_(array_like)
        except Exception:
            return array_like

    def _init_propagator(self) -> None:
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
        absolute_tolerances = self._coerce_double_array(tolerances[0])
        relative_tolerances = self._coerce_double_array(tolerances[1])

        integrator = DormandPrince853Integrator(
            self.config.min_integrator_step_sec,
            self.config.max_integrator_step_sec,
            absolute_tolerances,
            relative_tolerances,
        )
        integrator.setInitialStepSize(self.config.step_sec)

        self.propagator = NumericalPropagator(integrator)
        self.propagator.setOrbitType(OrbitType.EQUINOCTIAL)
        self.propagator.setInitialState(SpacecraftState(initial_orbit, self.config.mass_kg))

        gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
        gravity = HolmesFeatherstoneAttractionModel(self.earth_frame, gravity_provider)
        self.propagator.addForceModel(gravity)

        atmosphere = self._build_atmosphere()
        drag_spacecraft = IsotropicDrag(self.config.cross_section_m2, self.config.drag_coeff)
        self.propagator.addForceModel(DragForce(atmosphere, drag_spacecraft))

        self.propagator.addForceModel(self._build_srp_model())

    # ------------------------------------------------------------------
    # Vector helpers (shape-safe, mirrors ANTARA_Science_Sim.py conventions)
    # ------------------------------------------------------------------
    def _vector_difference(self, first_vector, second_vector) -> np.ndarray:
        return _as_flat_vector(first_vector) - _as_flat_vector(second_vector)

    def _unit_numpy(self, vector) -> np.ndarray:
        vector = _as_flat_vector(vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("Cannot normalise a zero-length vector.")
        return vector / norm

    def _rotate_vector_rodrigues(self, vector: np.ndarray, axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
        """Rotate `vector` about `axis_unit` by `angle_rad` (Rodrigues' formula)."""

        v = _as_flat_vector(vector)
        k = axis_unit
        return (
            v * math.cos(angle_rad)
            + np.cross(k, v) * math.sin(angle_rad)
            + k * float(np.dot(k, v)) * (1.0 - math.cos(angle_rad))
        )

    # ------------------------------------------------------------------
    # Orbital state / eclipse / beta angle (identical geometry to the DIL sim)
    # ------------------------------------------------------------------
    def _compute_beta_angle_deg(self, sat_position, sat_velocity, absolute_date: AbsoluteDate) -> float:
        sun_position = self.sun.getPVCoordinates(absolute_date, self.inertial_frame).getPosition()
        sun_unit_vec = self._unit_numpy(self._vector_difference(sun_position, sat_position))
        orbit_normal = self._unit_numpy(np.cross(_as_flat_vector(sat_position), _as_flat_vector(sat_velocity)))
        beta_rad = math.asin(np.clip(float(np.dot(orbit_normal, sun_unit_vec)), -1.0, 1.0))
        return math.degrees(beta_rad)

    def _is_in_eclipse_geometry(self, sat_position, absolute_date: AbsoluteDate) -> bool:
        """Conical (penumbra-aware) Sun-Earth-spacecraft shadow test."""

        sun_position = self.sun.getPVCoordinates(absolute_date, self.inertial_frame).getPosition()
        sat_position = _as_flat_vector(sat_position)
        sun_position = _as_flat_vector(sun_position)
        sat_to_sun = _as_flat_vector(sun_position - sat_position)
        shadow_axis = self._unit_numpy(-sun_position)

        angle_to_shadow_rad = math.acos(
            np.clip(float(np.dot(self._unit_numpy(sat_position), self._unit_numpy(shadow_axis))), -1.0, 1.0)
        )
        earth_ratio = np.clip(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS / float(np.linalg.norm(sat_position)), -1.0, 1.0
        )
        earth_apparent_radius_rad = math.asin(earth_ratio)
        sun_ratio = np.clip(Constants.SUN_RADIUS / float(np.linalg.norm(sat_to_sun)), -1.0, 1.0)
        sun_apparent_radius_rad = math.asin(sun_ratio)

        return angle_to_shadow_rad <= (earth_apparent_radius_rad - sun_apparent_radius_rad)

    def _build_orbit_state(self, propagated_state, time_sec: float) -> OrbitState:
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
        beta_angle_deg = self._compute_beta_angle_deg(sat_pos_inertial, sat_vel_inertial, absolute_date)

        sun_position = self.sun.getPVCoordinates(absolute_date, self.inertial_frame).getPosition()
        sun_distance_m = float(np.linalg.norm(self._vector_difference(sun_position, sat_pos_inertial)))
        sun_distance_au = sun_distance_m / self._au_m

        return OrbitState(
            absolute_date=absolute_date,
            time_sec=time_sec,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            altitude_km=sat_geo.getAltitude() / 1000.0,
            in_eclipse=in_eclipse,
            beta_angle_deg=beta_angle_deg,
            sun_distance_au=sun_distance_au,
            sat_position_inertial=sat_pos_inertial,
            sat_velocity_inertial=sat_vel_inertial,
        )

    # ------------------------------------------------------------------
    # Beta-angle targeting (suggestion #1): scan RAAN at the fixed epoch to
    # find the value that gives a desired beta angle (e.g. ~90 deg for a
    # continuous-sun hot case, ~0 deg for a max-eclipse cold case), using
    # the orbit's *analytic* PVCoordinates (no propagation needed since we
    # only need the state at epoch).
    # ------------------------------------------------------------------
    def solve_raan_for_target_beta_deg(self, target_beta_deg: float, coarse_step_deg: float = 2.0) -> float:
        def beta_at_raan(raan_deg: float) -> float:
            orbit = self._build_initial_orbit(raan_deg=raan_deg)
            pv = orbit.getPVCoordinates(self.inertial_frame)
            return self._compute_beta_angle_deg(pv.getPosition(), pv.getVelocity(), self.start_date)

        candidates = np.arange(0.0, 360.0, coarse_step_deg)
        errors = [abs(beta_at_raan(float(r)) - target_beta_deg) for r in candidates]
        best_raan = float(candidates[int(np.argmin(errors))])

        # Local refinement around the coarse best.
        fine_candidates = np.arange(best_raan - coarse_step_deg, best_raan + coarse_step_deg, 0.05)
        fine_errors = [abs(beta_at_raan(float(r) % 360.0) - target_beta_deg) for r in fine_candidates]
        best_fine_raan = float(fine_candidates[int(np.argmin(fine_errors))]) % 360.0
        return best_fine_raan

    # ------------------------------------------------------------------
    # Face normal construction: LVLH-fixed (nominal) or tumbling (worst-case
    # uncontrolled-attitude bound)
    # ------------------------------------------------------------------
    def _build_lvlh_face_normals(self, orbit_state: OrbitState, sun_unit_inertial: np.ndarray) -> Dict[str, np.ndarray]:
        r_vec = _as_flat_vector(orbit_state.sat_position_inertial)
        v_vec = _as_flat_vector(orbit_state.sat_velocity_inertial)

        y_hat = self._unit_numpy(-r_vec)                        # +Y = nadir
        orbit_normal = self._unit_numpy(np.cross(r_vec, v_vec))  # perpendicular to orbital plane
        z_sign = 1.0 if float(np.dot(orbit_normal, sun_unit_inertial)) >= 0.0 else -1.0
        z_hat = z_sign * orbit_normal                            # +Z = sun-side orbit normal
        x_hat = self._unit_numpy(np.cross(y_hat, z_hat))         # +X completes right-handed frame (~RAM)

        return {"+X": x_hat, "-X": -x_hat, "+Y": y_hat, "-Y": -y_hat, "+Z": z_hat, "-Z": -z_hat}

    def _build_tumbling_face_normals(self, time_sec: float) -> Dict[str, np.ndarray]:
        angle_rad = math.radians(self.config.tumble_rate_deg_s * time_sec)
        return {
            label: self._rotate_vector_rodrigues(BODY_AXIS_VECTORS[label], self._tumble_axis_unit, angle_rad)
            for label in FACE_ORDER
        }

    def _build_face_normals(self, orbit_state: OrbitState, sun_unit_inertial: np.ndarray, time_sec: float) -> Dict[str, np.ndarray]:
        if self.config.attitude_mode.upper() == "TUMBLING":
            return self._build_tumbling_face_normals(time_sec)
        return self._build_lvlh_face_normals(orbit_state, sun_unit_inertial)

    # ------------------------------------------------------------------
    # Earth view-factor table (suggestion #7): numerically integrated
    # flat-plate-to-sphere view factor F(phi), where phi is the angle
    # between the face normal and the nadir direction. Uses the exact
    # identity for a Lambertian (cosine-response) flat receiver:
    #
    #     F(phi) = (1/pi) * Integral_over_visible_disk max(0, cos(theta)) dOmega
    #
    # where theta is the angle between the face normal and each sky
    # direction inside the disk Earth subtends (half-angle rho). This
    # correctly captures horizon grazing/clipping, unlike a plain
    # cos(phi)*sin^2(rho) approximation, without needing a closed-form
    # (and error-prone-to-transcribe) elliptic expression. Built once at
    # init; looked up by interpolation at runtime.
    # ------------------------------------------------------------------
    def _nominal_earth_angular_radius_rad(self) -> float:
        r_nominal_m = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + self.config.altitude_m
        return math.asin(np.clip(Constants.WGS84_EARTH_EQUATORIAL_RADIUS / r_nominal_m, -1.0, 1.0))

    def _build_earth_view_factor_table(self, phi_resolution_deg: float = 0.5,
                                        alpha_samples: int = 200, beta_samples: int = 200) -> None:
        rho_rad = self._nominal_earth_angular_radius_rad()

        phi_deg = np.arange(0.0, 180.0 + phi_resolution_deg, phi_resolution_deg)
        phi_rad = np.radians(phi_deg)

        alpha = np.linspace(1.0e-6, rho_rad, alpha_samples)     # angular distance from nadir, within Earth's disk
        beta = np.linspace(0.0, 2.0 * np.pi, beta_samples, endpoint=False)
        d_beta = beta[1] - beta[0]

        sin_alpha = np.sin(alpha)[:, None]     # (alpha_samples, 1)
        cos_alpha = np.cos(alpha)[:, None]
        cos_beta = np.cos(beta)[None, :]       # (1, beta_samples)

        f_table = np.zeros_like(phi_rad)
        for i, phi in enumerate(phi_rad):
            cos_theta = math.cos(phi) * cos_alpha + math.sin(phi) * sin_alpha * cos_beta
            integrand = np.clip(cos_theta, 0.0, None) * sin_alpha   # weight by sin(alpha) for solid angle
            inner = integrand.sum(axis=1) * d_beta                  # integrate over azimuth
            f_table[i] = np.trapz(inner, alpha) / math.pi            # integrate over polar angle

        self._view_factor_phi_deg = phi_deg
        self._view_factor_table = f_table

        # Sanity check: nadir-pointing (phi=0) closed form is F = sin^2(rho).
        analytic_nadir = math.sin(rho_rad) ** 2
        numeric_nadir = float(f_table[0])
        if abs(analytic_nadir - numeric_nadir) > 0.01 * max(analytic_nadir, 1.0e-6):
            print(
                f"[WARN] Earth view-factor table mismatch at nadir: "
                f"analytic={analytic_nadir:.5f} numeric={numeric_nadir:.5f}. "
                f"Consider increasing alpha_samples/beta_samples."
            )

    def _earth_view_factor(self, cos_theta_to_nadir: float) -> float:
        phi_deg = math.degrees(math.acos(np.clip(cos_theta_to_nadir, -1.0, 1.0)))
        return float(np.interp(phi_deg, self._view_factor_phi_deg, self._view_factor_table))

    # ------------------------------------------------------------------
    # Latitude-dependent Earth albedo / IR (suggestion #6)
    # ------------------------------------------------------------------
    def _albedo_for_latitude(self, latitude_deg: float) -> float:
        if not self.config.use_latitude_dependent_environment:
            return self.config.earth_bond_albedo
        x = min(abs(latitude_deg) / 90.0, 1.0)
        return self.config.earth_bond_albedo_equator + (
            self.config.earth_bond_albedo_pole - self.config.earth_bond_albedo_equator
        ) * (x ** 2)

    def _earth_ir_for_latitude(self, latitude_deg: float) -> float:
        if not self.config.use_latitude_dependent_environment:
            return self.config.earth_ir_flux_w_m2
        x = min(abs(latitude_deg) / 90.0, 1.0)
        return self.config.earth_ir_equator_w_m2 + (
            self.config.earth_ir_pole_w_m2 - self.config.earth_ir_equator_w_m2
        ) * (x ** 2)

    # ------------------------------------------------------------------
    # Per-face incident flux (solar / albedo / Earth-IR)
    # ------------------------------------------------------------------
    def _compute_face_fluxes(
        self,
        orbit_state: OrbitState,
        face_normals_inertial: Dict[str, np.ndarray],
        sun_unit_inertial: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Return per-face {q_solar_heat_w, q_solar_electrical_w, q_albedo_w,
        q_ir_w, view_factor_earth} in Watts (already multiplied by face area
        and absorbed fraction)."""

        solar_flux_w_m2 = self.config.solar_constant_w_m2 / (orbit_state.sun_distance_au ** 2)
        albedo = self._albedo_for_latitude(orbit_state.latitude_deg)
        earth_ir_w_m2 = self._earth_ir_for_latitude(orbit_state.latitude_deg)

        nadir_unit = self._unit_numpy(-_as_flat_vector(orbit_state.sat_position_inertial))
        cos_subsat_sun = float(np.dot(nadir_unit, sun_unit_inertial))

        fluxes: Dict[str, Dict[str, float]] = {}
        for face_label in FACE_ORDER:
            face = self.faces[face_label]
            normal = face_normals_inertial[face_label]
            alpha_eff = face.effective_solar_absorptivity(self.config.mission_life_condition)

            cos_sun = float(np.dot(normal, sun_unit_inertial))
            cos_nadir = float(np.dot(normal, nadir_unit))

            view_factor_earth = self._earth_view_factor(cos_nadir)

            if orbit_state.in_eclipse:
                q_solar_absorbed_w = 0.0
            else:
                q_solar_absorbed_w = solar_flux_w_m2 * alpha_eff * face.area_m2 * max(0.0, cos_sun)

            # Solar-cell faces convert part of the absorbed power to
            # electricity rather than heat (suggestion #3).
            q_solar_electrical_w = q_solar_absorbed_w * face.cell_efficiency
            q_solar_heat_w = q_solar_absorbed_w * (1.0 - face.cell_efficiency)

            if orbit_state.in_eclipse or cos_subsat_sun <= 0.0:
                q_albedo_w = 0.0
            else:
                q_albedo_w = (
                    albedo * solar_flux_w_m2 * view_factor_earth * alpha_eff * face.area_m2 * cos_subsat_sun
                )
                # Albedo is reflected *sunlight*; the same electrical
                # conversion fraction applies to it on solar-cell faces.
                q_albedo_w *= (1.0 - face.cell_efficiency)

            q_ir_w = earth_ir_w_m2 * view_factor_earth * face.ir_emissivity * face.area_m2

            fluxes[face_label] = {
                "q_solar_heat_w": q_solar_heat_w,
                "q_solar_electrical_w": q_solar_electrical_w,
                "q_albedo_w": q_albedo_w,
                "q_ir_w": q_ir_w,
                "view_factor_earth": view_factor_earth,
                "solar_flux_w_m2": solar_flux_w_m2,
            }
        return fluxes

    # ------------------------------------------------------------------
    # Time-varying internal dissipation (suggestion #2): a synthetic
    # duty-cycle standing in for the DIL sim's real EPS/FSM power profile.
    # ------------------------------------------------------------------
    def _internal_dissipation_w(self, time_sec: float, in_eclipse: bool) -> float:
        power_w = self.config.baseline_bus_dissipation_w
        power_w += self.config.eclipse_heater_dissipation_w if in_eclipse else self.config.charging_bonus_dissipation_w

        orbit_phase_sec = time_sec % self.config.orbit_period_sec

        if orbit_phase_sec < self.config.relay_pass_duration_sec:
            power_w += self.config.relay_pass_dissipation_w

        detection_start_sec = 0.5 * self.config.orbit_period_sec
        if detection_start_sec <= orbit_phase_sec < (detection_start_sec + self.config.detection_window_duration_sec):
            power_w += self.config.detection_window_dissipation_w

        return power_w

    # ------------------------------------------------------------------
    # Thermal network integration (7-node lumped capacitance, RK4), now
    # including face-to-face structural conduction (suggestion #5)
    # ------------------------------------------------------------------
    def _thermal_derivatives(
        self,
        temperatures_k: Dict[str, float],
        fluxes: Dict[str, Dict[str, float]],
        in_eclipse: bool,
        time_sec: float,
    ) -> Dict[str, float]:
        sigma = self.config.stefan_boltzmann_w_m2_k4
        t_space4 = self.config.deep_space_temp_k ** 4
        g_ff = self.config.face_to_face_conductance_w_per_k

        derivatives: Dict[str, float] = {}
        internal_conduction_sum_w = 0.0

        for face_label in FACE_ORDER:
            face = self.faces[face_label]
            t_face = temperatures_k[face_label]
            t_internal = temperatures_k["INTERNAL"]

            q_absorbed_w = (
                fluxes[face_label]["q_solar_heat_w"]
                + fluxes[face_label]["q_albedo_w"]
                + fluxes[face_label]["q_ir_w"]
            )
            q_radiated_w = face.ir_emissivity * sigma * face.area_m2 * (t_face ** 4 - t_space4)
            q_conduction_internal_w = face.conductance_to_internal_w_per_k * (t_internal - t_face)

            q_conduction_neighbors_w = sum(
                g_ff * (temperatures_k[neighbor] - t_face) for neighbor in FACE_ADJACENCY[face_label]
            )

            internal_conduction_sum_w += q_conduction_internal_w

            derivatives[face_label] = (
                q_absorbed_w - q_radiated_w + q_conduction_internal_w + q_conduction_neighbors_w
            ) / face.thermal_capacitance_j_per_k

        internal_dissipation_w = self._internal_dissipation_w(time_sec, in_eclipse)
        derivatives["INTERNAL"] = (
            internal_dissipation_w - internal_conduction_sum_w
        ) / self.config.internal_thermal_capacitance_j_per_k

        return derivatives

    def _rk4_thermal_step(
        self,
        temperatures_k: Dict[str, float],
        fluxes: Dict[str, Dict[str, float]],
        in_eclipse: bool,
        time_sec: float,
        dt_sec: float,
    ) -> Dict[str, float]:
        """Single RK4 step. Fluxes are held constant across the sub-step
        (valid since dt_sec << the orbital/illumination timescale when
        thermal_substeps_per_step is set high enough). The internal
        dissipation duty-cycle is re-evaluated at each RK4 stage time so a
        window boundary crossed mid-step is not smeared."""

        nodes = list(temperatures_k.keys())

        def add(state: Dict[str, float], deriv: Dict[str, float], scale: float) -> Dict[str, float]:
            return {n: state[n] + scale * deriv[n] for n in nodes}

        k1 = self._thermal_derivatives(temperatures_k, fluxes, in_eclipse, time_sec)
        k2 = self._thermal_derivatives(add(temperatures_k, k1, dt_sec / 2.0), fluxes, in_eclipse, time_sec + dt_sec / 2.0)
        k3 = self._thermal_derivatives(add(temperatures_k, k2, dt_sec / 2.0), fluxes, in_eclipse, time_sec + dt_sec / 2.0)
        k4 = self._thermal_derivatives(add(temperatures_k, k3, dt_sec), fluxes, in_eclipse, time_sec + dt_sec)

        return {
            n: temperatures_k[n] + (dt_sec / 6.0) * (k1[n] + 2.0 * k2[n] + 2.0 * k3[n] + k4[n])
            for n in nodes
        }

    # ------------------------------------------------------------------
    # Main propagation + thermal integration loop
    # ------------------------------------------------------------------
    def run(self, write_outputs: bool = True, verbose: bool = True) -> pd.DataFrame:
        final_time_sec = self.config.sim_duration_hours * 3600.0
        step_sec = self.config.step_sec
        substeps = self.config.thermal_substeps_per_step
        sub_dt_sec = step_sec / substeps

        rows: List[Dict[str, float]] = []
        time_sec = 0.0
        next_progress_report_sec = 0.0

        if verbose:
            print(
                f"[ANTARA Thermal Sim] case attitude={self.config.attitude_mode} | "
                f"life={self.config.mission_life_condition} | "
                f"duration={self.config.sim_duration_hours:.2f} h | "
                f"orbit step={step_sec:.2f} s | thermal sub-dt={sub_dt_sec:.3f} s"
            )

        while time_sec <= final_time_sec + 1.0e-6:
            absolute_date = self.start_date.shiftedBy(time_sec)
            propagated_state = self.propagator.propagate(absolute_date)
            orbit_state = self._build_orbit_state(propagated_state, time_sec)

            sun_position = self.sun.getPVCoordinates(orbit_state.absolute_date, self.inertial_frame).getPosition()
            sun_unit_inertial = self._unit_numpy(
                self._vector_difference(sun_position, orbit_state.sat_position_inertial)
            )

            face_normals_inertial = self._build_face_normals(orbit_state, sun_unit_inertial, time_sec)
            fluxes = self._compute_face_fluxes(orbit_state, face_normals_inertial, sun_unit_inertial)

            row: Dict[str, float] = {
                "time_sec": time_sec,
                "time_hr": time_sec / 3600.0,
                "latitude_deg": orbit_state.latitude_deg,
                "longitude_deg": orbit_state.longitude_deg,
                "altitude_km": orbit_state.altitude_km,
                "in_eclipse": bool(orbit_state.in_eclipse),
                "beta_angle_deg": orbit_state.beta_angle_deg,
                "sun_distance_au": orbit_state.sun_distance_au,
                "internal_dissipation_w": self._internal_dissipation_w(time_sec, orbit_state.in_eclipse),
                "internal_temp_k": self.temperatures_k["INTERNAL"],
                "internal_temp_c": self.temperatures_k["INTERNAL"] - 273.15,
            }
            for face_label in FACE_ORDER:
                row[f"{face_label}_temp_k"] = self.temperatures_k[face_label]
                row[f"{face_label}_temp_c"] = self.temperatures_k[face_label] - 273.15
                row[f"{face_label}_q_solar_heat_w"] = fluxes[face_label]["q_solar_heat_w"]
                row[f"{face_label}_q_solar_electrical_w"] = fluxes[face_label]["q_solar_electrical_w"]
                row[f"{face_label}_q_albedo_w"] = fluxes[face_label]["q_albedo_w"]
                row[f"{face_label}_q_ir_w"] = fluxes[face_label]["q_ir_w"]
                row[f"{face_label}_view_factor_earth"] = fluxes[face_label]["view_factor_earth"]
            rows.append(row)

            if verbose and (time_sec >= next_progress_report_sec or math.isclose(time_sec, final_time_sec)):
                hottest = max(FACE_ORDER, key=lambda f: self.temperatures_k[f])
                coldest = min(FACE_ORDER, key=lambda f: self.temperatures_k[f])
                print(
                    f"  t={time_sec/3600.0:6.2f} h | eclipse={orbit_state.in_eclipse!s:5} | "
                    f"beta={orbit_state.beta_angle_deg:6.2f} deg | "
                    f"internal={self.temperatures_k['INTERNAL']-273.15:6.2f} C | "
                    f"hottest={hottest} ({self.temperatures_k[hottest]-273.15:6.2f} C) | "
                    f"coldest={coldest} ({self.temperatures_k[coldest]-273.15:6.2f} C)"
                )
                next_progress_report_sec += self.config.progress_report_interval_sec

            interval_sec = min(step_sec, final_time_sec - time_sec)
            if interval_sec <= 0.0:
                break
            sub_dt_sec = interval_sec / substeps
            for index in range(substeps):
                self.temperatures_k = self._rk4_thermal_step(
                    self.temperatures_k, fluxes, orbit_state.in_eclipse,
                    time_sec + index * sub_dt_sec, sub_dt_sec,
                )
            time_sec = min(final_time_sec, time_sec + interval_sec)

        telemetry_df = pd.DataFrame(rows)

        if write_outputs:
            telemetry_df.to_csv(results_path("thermal", "ANTARA_Thermal_telemetry.csv"), index=False)
            self._write_worst_case_summary_csv(telemetry_df)
            self._plot_results(telemetry_df)
            self._print_summary(telemetry_df)

        return telemetry_df

    # ------------------------------------------------------------------
    # Convergence check (suggestion #9): halve step_sec / double substeps
    # over a short window and compare worst-case temperatures.
    # ------------------------------------------------------------------
    def run_convergence_check(self, duration_hours: float = 2.0) -> pd.DataFrame:
        base_cfg = replace(self.config, sim_duration_hours=duration_hours, progress_report_interval_sec=1.0e9)

        coarse_cfg = base_cfg
        fine_cfg = replace(
            base_cfg,
            step_sec=base_cfg.step_sec / 2.0,
            thermal_substeps_per_step=base_cfg.thermal_substeps_per_step * 2,
        )

        print(f"\n[Convergence check] running {duration_hours:.2f} h at two resolutions...")
        coarse_df = ThermalSimulator(coarse_cfg).run(write_outputs=False, verbose=False)
        fine_df = ThermalSimulator(fine_cfg).run(write_outputs=False, verbose=False)

        records = []
        for face_label in FACE_ORDER + ["internal"]:
            col = f"{face_label}_temp_c" if face_label != "internal" else "internal_temp_c"
            coarse_max, coarse_min = coarse_df[col].max(), coarse_df[col].min()
            fine_max, fine_min = fine_df[col].max(), fine_df[col].min()
            records.append(
                {
                    "node": face_label,
                    "coarse_hot_c": coarse_max, "fine_hot_c": fine_max, "delta_hot_c": fine_max - coarse_max,
                    "coarse_cold_c": coarse_min, "fine_cold_c": fine_min, "delta_cold_c": fine_min - coarse_min,
                }
            )
        summary_df = pd.DataFrame(records)
        print(summary_df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        max_delta = max(summary_df["delta_hot_c"].abs().max(), summary_df["delta_cold_c"].abs().max())
        print(f"Max |delta| between resolutions: {max_delta:.3f} C "
              f"(step_sec {coarse_cfg.step_sec}s vs {fine_cfg.step_sec}s)")
        return summary_df

    # ------------------------------------------------------------------
    # Outputs: worst-case summary CSV, plots, console summary
    # ------------------------------------------------------------------
    def _write_worst_case_summary_csv(self, telemetry_df: pd.DataFrame) -> None:
        margin_c = self.config.applied_margin_c
        records = []
        for face_label in FACE_ORDER + ["internal"]:
            temp_col_c = f"{face_label}_temp_c" if face_label != "internal" else "internal_temp_c"
            hot_idx = telemetry_df[temp_col_c].idxmax()
            cold_idx = telemetry_df[temp_col_c].idxmin()
            hot_c = telemetry_df.loc[hot_idx, temp_col_c]
            cold_c = telemetry_df.loc[cold_idx, temp_col_c]
            records.append(
                {
                    "node": face_label,
                    "hot_case_temp_c": hot_c,
                    "hot_case_time_hr": telemetry_df.loc[hot_idx, "time_hr"],
                    "hot_case_temp_with_margin_c": hot_c + margin_c,
                    "cold_case_temp_c": cold_c,
                    "cold_case_time_hr": telemetry_df.loc[cold_idx, "time_hr"],
                    "cold_case_temp_with_margin_c": cold_c - margin_c,
                }
            )
        summary_df = pd.DataFrame(records)
        summary_df.to_csv(results_path("thermal", "ANTARA_Thermal_worst_case_summary.csv"), index=False)

    def _shade_eclipse(self, axis, telemetry_df: pd.DataFrame) -> None:
        in_eclipse = telemetry_df["in_eclipse"].to_numpy()
        time_hr = telemetry_df["time_hr"].to_numpy()
        start_idx = None
        for i, flag in enumerate(in_eclipse):
            if flag and start_idx is None:
                start_idx = i
            elif not flag and start_idx is not None:
                axis.axvspan(time_hr[start_idx], time_hr[i - 1], color="0.85", zorder=0)
                start_idx = None
        if start_idx is not None:
            axis.axvspan(time_hr[start_idx], time_hr[-1], color="0.85", zorder=0)

    def _plot_results(self, telemetry_df: pd.DataFrame) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            return

        colors = {
            "+X": "#d62728", "-X": "#9467bd", "+Y": "#1f77b4",
            "-Y": "#2ca02c", "+Z": "#ff7f0e", "-Z": "#17becf",
        }

        # --- Figure 1: face + internal temperatures over time ---------
        fig, axis = plt.subplots(figsize=(12, 6))
        for face_label in FACE_ORDER:
            axis.plot(
                telemetry_df["time_hr"], telemetry_df[f"{face_label}_temp_c"],
                label=f"{face_label} ({self.faces[face_label].label})",
                color=colors[face_label], linewidth=1.4,
            )
        axis.plot(
            telemetry_df["time_hr"], telemetry_df["internal_temp_c"],
            label="Internal (bus)", color="black", linewidth=2.2, linestyle="--",
        )
        self._shade_eclipse(axis, telemetry_df)
        axis.set_xlabel("Mission elapsed time (hours)")
        axis.set_ylabel("Temperature (°C)")
        axis.set_title(f"ANTARA Thermal Simulation — Temperatures ({self.config.attitude_mode} attitude)")
        axis.legend(loc="upper right", fontsize=8, ncol=2)
        axis.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_path("thermal", "ANTARA_Thermal_temperatures.png"), dpi=200)
        plt.close(fig)

        # --- Figure 2: radiative environment (solar / albedo / IR) ----
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        titles = ["Absorbed Solar Flux (post solar-cell derating)", "Absorbed Albedo Flux", "Absorbed Earth-IR Flux"]
        keys = ["q_solar_heat_w", "q_albedo_w", "q_ir_w"]
        for axis, title, key in zip(axes, titles, keys):
            for face_label in FACE_ORDER:
                axis.plot(
                    telemetry_df["time_hr"], telemetry_df[f"{face_label}_{key}"],
                    label=face_label, color=colors[face_label], linewidth=1.2,
                )
            self._shade_eclipse(axis, telemetry_df)
            axis.set_ylabel("Power (W)")
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        axes[-1].set_xlabel("Mission elapsed time (hours)")
        fig.tight_layout()
        fig.savefig(figures_path("thermal", "ANTARA_Thermal_radiation_components.png"), dpi=200)
        plt.close(fig)

        # --- Figure 3: orbital/eclipse/power context -------------------
        fig, (axis1, axis2, axis3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        axis1.plot(telemetry_df["time_hr"], telemetry_df["beta_angle_deg"], color="darkorange")
        self._shade_eclipse(axis1, telemetry_df)
        axis1.set_ylabel("Beta angle (deg)")
        axis1.set_title("Orbital / Eclipse / Internal-Power Context")
        axis1.grid(True, alpha=0.3)

        axis2.plot(telemetry_df["time_hr"], telemetry_df["altitude_km"], color="teal")
        self._shade_eclipse(axis2, telemetry_df)
        axis2.set_ylabel("Altitude (km)")
        axis2.grid(True, alpha=0.3)

        axis3.plot(telemetry_df["time_hr"], telemetry_df["internal_dissipation_w"], color="crimson")
        self._shade_eclipse(axis3, telemetry_df)
        axis3.set_xlabel("Mission elapsed time (hours)")
        axis3.set_ylabel("Internal diss. (W)")
        axis3.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_path("thermal", "ANTARA_Thermal_orbit_context.png"), dpi=200)
        plt.close(fig)

        # --- Figure 4: worst-case hot/cold bar summary (with margin) ---
        fig, axis = plt.subplots(figsize=(9, 5.5))
        node_labels = FACE_ORDER + ["INTERNAL"]
        margin_c = self.config.applied_margin_c
        hot_vals = [telemetry_df[f"{f}_temp_c" if f != 'INTERNAL' else 'internal_temp_c'].max() for f in node_labels]
        cold_vals = [telemetry_df[f"{f}_temp_c" if f != 'INTERNAL' else 'internal_temp_c'].min() for f in node_labels]
        x_positions = np.arange(len(node_labels))
        width = 0.38
        axis.bar(x_positions - width / 2, hot_vals, width, label="Hot case", color="firebrick")
        axis.bar(x_positions + width / 2, cold_vals, width, label="Cold case", color="steelblue")
        axis.scatter(x_positions - width / 2, [v + margin_c for v in hot_vals], marker="_", color="black", zorder=5,
                     label=f"+/-{margin_c:.0f} C margin")
        axis.scatter(x_positions + width / 2, [v - margin_c for v in cold_vals], marker="_", color="black", zorder=5)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(node_labels)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Temperature (°C)")
        axis.set_title(f"Worst-Case Hot / Cold Temperature per Node ({self.config.mission_life_condition})")
        axis.legend(fontsize=8)
        axis.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figures_path("thermal", "ANTARA_Thermal_worst_case_summary.png"), dpi=200)
        plt.close(fig)

    def _print_summary(self, telemetry_df: pd.DataFrame) -> None:
        margin_c = self.config.applied_margin_c
        print("\n" + "=" * 90)
        print(
            f"ANTARA THERMAL SIMULATION — SUMMARY "
            f"[{self.config.attitude_mode} attitude, {self.config.mission_life_condition} optics]"
        )
        print("=" * 90)
        eclipse_fraction_percent = 100.0 * telemetry_df["in_eclipse"].mean()
        print(f"Eclipse fraction of orbit: {eclipse_fraction_percent:.2f} %")
        print(f"Beta angle range: {telemetry_df['beta_angle_deg'].min():.2f} to "
              f"{telemetry_df['beta_angle_deg'].max():.2f} deg")
        print(f"Applied margin: +/-{margin_c:.1f} C")
        print("-" * 90)
        header = f"{'Node':<10}{'Hot (C)':>10}{'@t(hr)':>8}{'Hot+margin':>12}{'Cold (C)':>10}{'@t(hr)':>8}{'Cold-margin':>13}"
        print(header)
        for face_label in FACE_ORDER + ["INTERNAL"]:
            col = f"{face_label}_temp_c" if face_label != "INTERNAL" else "internal_temp_c"
            hot_idx = telemetry_df[col].idxmax()
            cold_idx = telemetry_df[col].idxmin()
            hot_c = telemetry_df.loc[hot_idx, col]
            cold_c = telemetry_df.loc[cold_idx, col]
            print(
                f"{face_label:<10}{hot_c:>10.2f}{telemetry_df.loc[hot_idx, 'time_hr']:>8.2f}"
                f"{hot_c + margin_c:>12.2f}{cold_c:>10.2f}{telemetry_df.loc[cold_idx, 'time_hr']:>8.2f}"
                f"{cold_c - margin_c:>13.2f}"
            )
        print("=" * 90)
        print("Outputs written:")
        print(f"  {results_path('thermal', 'ANTARA_Thermal_telemetry.csv')}")
        print(f"  {results_path('thermal', 'ANTARA_Thermal_worst_case_summary.csv')}")
        if _MATPLOTLIB_AVAILABLE:
            print(f"  {figures_path('thermal', 'ANTARA_Thermal_temperatures.png')}")
            print(f"  {figures_path('thermal', 'ANTARA_Thermal_radiation_components.png')}")
            print(f"  {figures_path('thermal', 'ANTARA_Thermal_orbit_context.png')}")
            print(f"  {figures_path('thermal', 'ANTARA_Thermal_worst_case_summary.png')}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ANTARA CubeSat thermal loading simulator.")
    parser.add_argument("--duration-hours", type=float, default=24.0, help="Simulation duration in hours.")
    parser.add_argument("--step-sec", type=float, default=2.0, help="Orbital propagation step size in seconds.")
    parser.add_argument(
        "--thermal-substeps", type=int, default=10,
        help="Thermal RK4 sub-steps per orbital propagation step.",
    )
    parser.add_argument(
        "--case", choices=["NOMINAL", "HOT", "COLD"], default="NOMINAL",
        help="Bounding thermal case: NOMINAL (as-configured), HOT, or COLD.",
    )
    parser.add_argument(
        "--attitude-mode", choices=["LVLH", "TUMBLING"], default="LVLH",
        help="LVLH = nominal controlled pointing. TUMBLING = worst-case uncontrolled/pre-detumble bound.",
    )
    parser.add_argument(
        "--target-beta-deg", type=float, default=None,
        help="If set, solve for the RAAN that gives this beta angle at epoch "
             "(e.g. 90 for a max-continuous-sun hot case, 0 for a max-eclipse cold case) "
             "before running.",
    )
    parser.add_argument(
        "--internal-dissipation-w", type=float, default=None,
        help="Override the baseline internal (bus) dissipation, in Watts.",
    )
    parser.add_argument(
        "--convergence-check", action="store_true",
        help="Run a short (2h) convergence check at two step-size resolutions instead of / before the full run.",
    )
    parser.add_argument(
        "--convergence-only", action="store_true",
        help="Only run the convergence check; skip the full-duration mission run.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    base_config = ThermalMissionConfig(
        sim_duration_hours=args.duration_hours,
        step_sec=args.step_sec,
        thermal_substeps_per_step=args.thermal_substeps,
        attitude_mode=args.attitude_mode,
    )
    if args.internal_dissipation_w is not None:
        base_config = replace(base_config, baseline_bus_dissipation_w=args.internal_dissipation_w)

    config = build_bounding_case_config(args.case, base_config)

    if args.target_beta_deg is not None:
        probe = ThermalSimulator(config)
        solved_raan_deg = probe.solve_raan_for_target_beta_deg(args.target_beta_deg)
        print(f"[Beta targeting] solved RAAN={solved_raan_deg:.2f} deg for target beta={args.target_beta_deg:.1f} deg")
        config = replace(config, raan_deg=solved_raan_deg)

    simulator = ThermalSimulator(config)

    if args.convergence_check or args.convergence_only:
        simulator.run_convergence_check(duration_hours=2.0)

    if not args.convergence_only:
        simulator.run()
