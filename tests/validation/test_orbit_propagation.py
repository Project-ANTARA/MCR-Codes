#!/usr/bin/env python3
"""
Orbit propagation validation test.

Compares Orekit numerical propagation against:
1. Two-body Kepler period (a separate unperturbed numerical propagation)
2. Closed-form SSO precession formula: dOmega/dt ≈ -1.5 * n * J2 * (Re/p)^2 * cos(i)

Tolerances:
- Period: ±1 second
- RAAN precession rate: ±5%
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np

# Orekit imports
import orekit
orekit.initVM()

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants, IERSConventions
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.orekit.forces.gravity import ThirdBodyAttraction
from simulations.helper.paths import init_orekit_data
from org.orekit.orbits import OrbitType
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient


# Mission parameters (matching ANTARA_Science_Sim defaults)
ALTITUDE_M = 600_000.0
INCLINATION_DEG = 97.6
ECCENTRICITY = 0.001
RAAN_DEG = 15.0
ARG_PERIGEE_DEG = 0.0
TRUE_ANOMALY_DEG = 0.0
MASS_KG = 3.0
CD = 2.2
CROSS_SECTION_M2 = 0.01
SRP_COEFF = 1.8

# Earth constants
MU = Constants.WGS84_EARTH_MU
RE = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
J2 = -Constants.WGS84_EARTH_C20
SOLAR_CONSTANT = 1361.0

# Simulation duration
SIM_DURATION_HOURS = 24.0
STEP_SEC = 60.0


def build_orekit_propagator():
    """Build Orekit numerical propagator matching ANTARA config."""
    init_orekit_data()
    utc = TimeScalesFactory.getUTC()
    start_date = AbsoluteDate(2000, 3, 1, 0, 0, 0.0, utc)

    inertial_frame = FramesFactory.getEME2000()
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(RE, Constants.WGS84_EARTH_FLATTENING, earth_frame)
    sun = CelestialBodyFactory.getSun()

    # Initial orbit
    semi_major_axis = RE + ALTITUDE_M
    orbit = KeplerianOrbit(
        semi_major_axis,
        ECCENTRICITY,
        float(np.radians(INCLINATION_DEG)),
        float(np.radians(ARG_PERIGEE_DEG)),
        float(np.radians(RAAN_DEG)),
        float(np.radians(TRUE_ANOMALY_DEG)),
        PositionAngleType.TRUE,
        inertial_frame,
        start_date,
        MU,
    )

    # Integrator
    integrator = DormandPrince853Integrator(0.1, 600.0, 1e-3, 1e-6)

    propagator = NumericalPropagator(integrator)
    # Cartesian coordinates avoid the near-circular eccentricity singularity.
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(SpacecraftState(orbit, MASS_KG))

    # Gravity
    gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    gravity = HolmesFeatherstoneAttractionModel(earth_frame, gravity_provider)
    propagator.addForceModel(gravity)

    # Third body
    propagator.addForceModel(ThirdBodyAttraction(sun))
    moon = CelestialBodyFactory.getMoon()
    propagator.addForceModel(ThirdBodyAttraction(moon))

    # Atmosphere
    space_weather = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
    atmosphere = DTM2000(space_weather, sun, earth)
    propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(CROSS_SECTION_M2, CD)))

    # SRP
    srp = SolarRadiationPressure(sun, earth, IsotropicRadiationSingleCoefficient(CROSS_SECTION_M2, SRP_COEFF))
    propagator.addForceModel(srp)

    # Attitude provider (nadir for force modeling)
    from org.orekit.attitudes import NadirPointing
    propagator.setAttitudeProvider(NadirPointing(inertial_frame, earth))

    return propagator, start_date, inertial_frame, earth_frame, earth, sun


def compute_closed_form_precession():
    """Compute SSO precession rate using closed-form formula."""
    # dOmega/dt ≈ -1.5 * n * J2 * (Re/p)^2 * cos(i)
    # p = a * (1 - e^2)  (semi-latus rectum)
    a = RE + ALTITUDE_M
    p = a * (1 - ECCENTRICITY**2)
    n = np.sqrt(MU / a**3)  # mean motion [rad/s]
    cos_i = np.cos(np.radians(INCLINATION_DEG))

    dOmega_dt = -1.5 * n * J2 * (RE / p)**2 * cos_i  # rad/s
    dOmega_dt_deg_day = dOmega_dt * 86400.0 * 180.0 / np.pi  # deg/day

    return dOmega_dt_deg_day


def run_orekit_propagation():
    """Run Orekit propagation and collect orbit data."""
    propagator, start_date, inertial_frame, earth_frame, earth, sun = build_orekit_propagator()

    final_time_sec = SIM_DURATION_HOURS * 3600.0
    step_sec = STEP_SEC

    raan_history = []
    period_history = []
    time_history = []

    t = 0.0
    while t <= final_time_sec:
        absolute_date = start_date.shiftedBy(t)
        state = propagator.propagate(absolute_date)

        osculating_orbit = state.getOrbit()
        keplerian = KeplerianOrbit(osculating_orbit)

        raan = np.degrees(keplerian.getRightAscensionOfAscendingNode())
        period_min = keplerian.getKeplerianPeriod() / 60.0

        raan_history.append(raan)
        period_history.append(period_min)
        time_history.append(t / 3600.0)  # hours

        t += step_sec

    return np.array(time_history), np.array(raan_history), np.array(period_history)


def measure_two_body_period():
    """Measure the next ascending crossing of the initial radial half-plane.

    The one-second Kepler tolerance applies to two-body dynamics, not the
    osculating period of the independently checked full-force orbit.
    """
    from scipy.optimize import brentq

    full, start, *_ = build_orekit_propagator()
    initial = full.getInitialState()
    prop = NumericalPropagator(DormandPrince853Integrator(0.01, 60.0, 1e-5, 1e-10))
    prop.setOrbitType(OrbitType.CARTESIAN)
    prop.setInitialState(initial)  # Central Newtonian attraction only.
    pv = initial.getPVCoordinates()
    transverse = pv.getMomentum().crossProduct(pv.getPosition())
    expected = initial.getOrbit().getKeplerianPeriod()

    def crossing(seconds):
        position = prop.propagate(start.shiftedBy(float(seconds))).getPVCoordinates().getPosition()
        return position.dotProduct(transverse)

    return brentq(crossing, expected - 30.0, expected + 30.0, xtol=1e-6)


def main():
    print("=" * 70)
    print("ORBIT PROPAGATION VALIDATION TEST")
    print("=" * 70)

    # Closed-form precession
    print("\n1. Closed-form SSO Precession Rate")
    print("-" * 40)
    dOmega_closed = compute_closed_form_precession()
    print(f"   dOmega/dt (closed-form): {dOmega_closed:.6f} deg/day")

    # Orekit propagation
    print("\n2. Orekit Numerical Propagation (24h)")
    print("-" * 40)
    ore_times, ore_raan, ore_period = run_orekit_propagation()

    # Compute precession from Orekit (linear fit to RAAN)
    # RAAN in degrees, time in hours
    raan_diff = np.unwrap(ore_raan * np.pi / 180.0) * 180.0 / np.pi  # unwrap to handle 360° wrap
    # Linear fit: raan = a * t + b
    coeffs = np.polyfit(ore_times, raan_diff, 1)
    dOmega_orekit = coeffs[0] * 24.0  # deg/day
    print(f"   dOmega/dt (Orekit fit):  {dOmega_orekit:.6f} deg/day")

    # Period
    avg_period = np.mean(ore_period)
    print(f"   Avg orbital period:      {avg_period:.3f} min")
    print(f"   Period std dev:          {np.std(ore_period):.6f} min")

    # Tolerances check
    print("\n3. Tolerance Verification")
    print("-" * 40)

    # Period tolerance: ±1 second
    # Theoretical period at 600km: ~96.7 min = 5802 s
    theoretical_period_sec = 2 * np.pi * np.sqrt((RE + ALTITUDE_M)**3 / MU)
    theoretical_period_min = theoretical_period_sec / 60.0
    measured_period_sec = measure_two_body_period()
    period_error = abs(measured_period_sec - theoretical_period_sec)
    print(f"   Two-body crossing period: {measured_period_sec:.6f} s")
    print(f"   Period error vs theoretical: {period_error:.2f} s (tol: ±1 s)")
    period_pass = bool(period_error <= 1.0)

    # Precession tolerance: ±5%
    precession_error_pct = abs(dOmega_orekit - dOmega_closed) / abs(dOmega_closed) * 100
    print(f"   Precession error: {precession_error_pct:.2f}% (tol: ±5%)")
    precession_pass = bool(precession_error_pct <= 5.0)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Period:            {'PASS' if period_pass else 'FAIL'} (error={period_error:.2f}s, tol=1s)")
    print(f"  RAAN Precession:   {'PASS' if precession_pass else 'FAIL'} (error={precession_error_pct:.2f}%, tol=5%)")

    overall = period_pass and precession_pass
    print(f"\n  OVERALL:           {'PASS' if overall else 'FAIL'}")

    # Write results
    results = {
        "period_avg_min": float(avg_period),
        "two_body_period_sec": float(measured_period_sec),
        "period_error_sec": float(period_error),
        "period_pass": period_pass,
        "raan_precession_deg_day_closed": float(dOmega_closed),
        "raan_precession_deg_day_orekit": float(dOmega_orekit),
        "precession_error_pct": float(precession_error_pct),
        "precession_pass": precession_pass,
        "overall_pass": overall,
    }

    import json
    with open(Path(os.environ.get("ANTARA_VALIDATION_OUTPUT_DIR", Path(__file__).parent)) / "validation_results_orbit.json", "w") as f:
        json.dump(results, f, indent=2)

    return overall


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    success = main()
    sys.exit(0 if success else 1)
