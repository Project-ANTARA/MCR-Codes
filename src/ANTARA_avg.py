# ================================================================
# ANTARA MASTER MISSION SIMULATION FRAMEWORK
# ================================================================
# HIGH-FIDELITY 2-YEAR MONTE CARLO MISSION ANALYSIS
#
# FINAL CONFIGURATION:
#   - 1 Fixed Zenith Panel
#   - 2 Deployed Solar Wings
#   - 25 Wh + 30 Wh Battery Cases
#
# FEATURES:
#   - Full Numerical Orekit Propagation
#   - Drag + SRP + Sun/Moon Perturbations
#   - Eclipse Analysis
#   - Power Generation
#   - Battery SOC
#   - Ground Station Visibility
#   - SAA Crossings
#   - Monte Carlo Uncertainty
#   - CSV Exports
#   - Statistical Analysis
#   - Live Progress Counter
#
# ================================================================

import orekit
orekit.initVM()

import os
import csv
import math
import time
import random
random = random.Random(42)  # Reproducible study-local generator.
import statistics
import numpy as np
import pandas as pd

from paths import init_orekit_data, results_dir

# ================================================================
# OREKIT DATA SETUP
# ================================================================
init_orekit_data()

# ================================================================
# OREKIT IMPORTS
# ================================================================
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants, IERSConventions
from org.orekit.bodies import (
    OneAxisEllipsoid,
    CelestialBodyFactory
)
from org.orekit.attitudes import NadirPointing
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.drag import (
    DragForce,
    IsotropicDrag
)
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.models.earth.atmosphere.data import (
    CssiSpaceWeatherData
)
from org.orekit.forces.radiation import (
    SolarRadiationPressure,
    IsotropicRadiationSingleCoefficient
)
from org.orekit.propagation.events import EclipseDetector
from org.orekit.propagation import SpacecraftState
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# ================================================================
# GLOBAL CONFIGURATION
# ================================================================

MISSION_DURATION_DAYS = 365
STEP_SIZE = 300.0
MONTE_CARLO_RUNS = 3

TOTAL_STEPS_PER_RUN = int(
    (MISSION_DURATION_DAYS * 86400.0) / STEP_SIZE
)

BATTERY_CASES = [25.0]

TOTAL_GLOBAL_STEPS = (
    TOTAL_STEPS_PER_RUN *
    MONTE_CARLO_RUNS *
    len(BATTERY_CASES)
)

# ================================================================
# ORBIT
# ================================================================
ALTITUDE = 600000.0
INCLINATION = 97.4
ECC = 0.001
RAAN = 75.0

# ================================================================
# SPACECRAFT
# ================================================================
MASS = 4.0
CD_BASE = 2.2
CR_BASE = 1.2

# ================================================================
# FINAL CONFIGURATION
# ================================================================
FIXED_PANEL_AREA = 0.030
DEPLOYED_PANEL_AREA = 0.030
NUM_DEPLOYED_PANELS = 2

TOTAL_AREA = 0.076799

# ================================================================
# POWER SYSTEM
# ================================================================
SOLAR_CONSTANT = 1361.0
CELL_EFFICIENCY = 0.28
SYSTEM_EFF = 0.513

LOAD_POWER = 8.0

TRACKING_ERROR_SIGMA = 5.0
TRACKING_BIAS = 2.0

# ================================================================
# MONTE CARLO UNCERTAINTIES
# ================================================================
ALTITUDE_SIGMA = 5000.0
CD_SIGMA = 0.15
CR_SIGMA = 0.10
BATTERY_EFF_SIGMA = 0.03

# ================================================================
# STATISTICS
# ================================================================
Z_99 = 2.576

# ================================================================
# OUTPUT FOLDER
# ================================================================
OUTPUT_DIR = str(results_dir("avg"))

# ================================================================
# TIME
# ================================================================
utc = TimeScalesFactory.getUTC()

# ================================================================
# FRAMES
# ================================================================
inertial_frame = FramesFactory.getEME2000()

earth_frame = FramesFactory.getITRF(
    IERSConventions.IERS_2010,
    True
)

# ================================================================
# EARTH
# ================================================================
earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    earth_frame
)

# ================================================================
# CELESTIAL BODIES
# ================================================================
sun = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()

# ================================================================
# GLOBAL PROGRESS
# ================================================================
global_step_counter = 0
simulation_start_walltime = time.time()

# ================================================================
# HELPER FUNCTIONS
# ================================================================

def safe_normalize(v):

    norm = v.getNorm()

    if norm == 0:
        return Vector3D(0,0,0)

    return v.scalarMultiply(1.0 / norm)

# ================================================================
# SAA CHECK
# ================================================================
def check_saa(lat_deg, lon_deg):

    return (
        -50.0 <= lat_deg <= 0.0
        and
        -90.0 <= lon_deg <= 40.0
    )

# ================================================================
# GROUND STATION CHECK
# ================================================================
def check_gs_bits_goa(lat_deg, lon_deg):

    return (
        0.0 <= lat_deg <= 20.0
        and
        60.0 <= lon_deg <= 80.0
    )

# ================================================================
# STATISTICS
# ================================================================
def compute_statistics(data, metric_name):

    arr = np.array(data)

    mean_val = np.mean(arr)
    std_val = np.std(arr)
    var_val = np.var(arr)

    min_val = np.min(arr)
    max_val = np.max(arr)

    ci = (
        Z_99 *
        std_val /
        math.sqrt(len(arr))
    )

    return {
        "Metric": metric_name,
        "Mean": mean_val,
        "StdDev": std_val,
        "Variance": var_val,
        "Min": min_val,
        "Max": max_val,
        "CI99_Low": mean_val - ci,
        "CI99_High": mean_val + ci
    }

# ================================================================
# POWER MODEL
# ================================================================
def calculate_power(
    state,
    eclipse,
    tracking_sigma
):

    if eclipse:
        return 0.0

    sun_pos = sun.getPVCoordinates(
        state.getDate(),
        inertial_frame
    ).getPosition()

    sat_pos = state.getPVCoordinates().getPosition()

    sun_vec = safe_normalize(
        sun_pos.subtract(sat_pos)
    )

    sun_body = (
        state.getAttitude()
        .getRotation()
        .applyTo(sun_vec)
    )

    # ------------------------------------------------------------
    # FIXED PANEL
    # ------------------------------------------------------------
    fixed_normal = Vector3D(
        0.0,
        0.0,
        -1.0
    )

    cos_theta = Vector3D.dotProduct(
        fixed_normal,
        sun_body
    )

    fixed_power = 0.0

    if cos_theta > 0:

        fixed_power = (
            SOLAR_CONSTANT *
            FIXED_PANEL_AREA *
            cos_theta *
            CELL_EFFICIENCY *
            SYSTEM_EFF
        )

    # ------------------------------------------------------------
    # DEPLOYED PANELS
    # ------------------------------------------------------------
    pointing_error = random.gauss(
        TRACKING_BIAS,
        tracking_sigma
    )

    tracking_loss = math.cos(
        math.radians(pointing_error)
    )

    deployed_power = (
        SOLAR_CONSTANT *
        DEPLOYED_PANEL_AREA *
        NUM_DEPLOYED_PANELS *
        tracking_loss *
        CELL_EFFICIENCY *
        SYSTEM_EFF
    )

    return fixed_power + deployed_power

# ================================================================
# MAIN SIMULATION
# ================================================================
all_summary_rows = []

for battery_capacity in BATTERY_CASES:

    print("\n================================================")
    print(f"RUNNING BATTERY CASE: {battery_capacity} Wh")
    print("================================================")

    timeseries_filename = os.path.join(
        OUTPUT_DIR,
        f"mission_timeseries_{int(battery_capacity)}Wh.csv"
    )

    with open(
        timeseries_filename,
        mode='w',
        newline=''
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "run_id",
            "time_days",
            "altitude_km",
            "latitude_deg",
            "longitude_deg",
            "eclipse",
            "power_generated_w",
            "battery_soc",
            "battery_energy_wh",
            "groundstation_visible",
            "saa_crossing"
        ])

        # ========================================================
        # METRICS
        # ========================================================
        avg_power_runs = []
        eclipse_fraction_runs = []
        min_soc_runs = []
        avg_soc_runs = []
        lifetime_estimate_runs = []

        # ========================================================
        # MONTE CARLO LOOP
        # ========================================================
        for run in range(MONTE_CARLO_RUNS):

            print(
                f"\nMonte Carlo Run "
                f"{run+1}/{MONTE_CARLO_RUNS}"
            )

            # ----------------------------------------------------
            # RANDOMIZATION
            # ----------------------------------------------------
            altitude_offset = random.gauss(
                0,
                ALTITUDE_SIGMA
            )

            cd = random.gauss(
                CD_BASE,
                CD_SIGMA
            )

            cr = random.gauss(
                CR_BASE,
                CR_SIGMA
            )

            battery_eff = random.gauss(
                1.0,
                BATTERY_EFF_SIGMA
            )
            while not 0.0 < battery_eff <= 1.0:
                battery_eff = random.gauss(1.0, BATTERY_EFF_SIGMA)

            tracking_sigma = random.uniform(
                3.0,
                8.0
            )

            # ----------------------------------------------------
            # INITIAL DATE
            # ----------------------------------------------------
            start_date = AbsoluteDate(
                2026,
                1,
                1,
                0,
                0,
                0.0,
                utc
            )

            # ----------------------------------------------------
            # INITIAL ORBIT
            # ----------------------------------------------------
            a0 = (
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS +
                ALTITUDE +
                altitude_offset
            )

            orbit = KeplerianOrbit(
                a0,
                ECC,
                math.radians(INCLINATION),
                0.0,
                math.radians(RAAN),
                0.0,
                PositionAngleType.TRUE,
                inertial_frame,
                start_date,
                Constants.WGS84_EARTH_MU
            )

            # ----------------------------------------------------
            # INTEGRATOR
            # ----------------------------------------------------
            integrator = DormandPrince853Integrator(
                0.1,
                300.0,
                1.0e-3,
                1.0e-6
            )

            propagator = NumericalPropagator(
                integrator
            )

            propagator.setInitialState(
                SpacecraftState(
                    orbit,
                    MASS
                )
            )

            propagator.setAttitudeProvider(
                NadirPointing(
                    inertial_frame,
                    earth
                )
            )

            # ----------------------------------------------------
            # GRAVITY
            # ----------------------------------------------------
            gravity_provider = (
                GravityFieldFactory
                .getNormalizedProvider(5,5)
            )

            propagator.addForceModel(
                HolmesFeatherstoneAttractionModel(
                    earth_frame,
                    gravity_provider
                )
            )

            # ----------------------------------------------------
            # THIRD BODY
            # ----------------------------------------------------
            propagator.addForceModel(
                ThirdBodyAttraction(sun)
            )

            propagator.addForceModel(
                ThirdBodyAttraction(moon)
            )

            # ----------------------------------------------------
            # ATMOSPHERE
            # ----------------------------------------------------
            space_weather = (
                CssiSpaceWeatherData(
                    CssiSpaceWeatherData
                    .DEFAULT_SUPPORTED_NAMES
                )
            )

            atmosphere = DTM2000(
                space_weather,
                sun,
                earth
            )

            propagator.addForceModel(
                DragForce(
                    atmosphere,
                    IsotropicDrag(
                        TOTAL_AREA,
                        cd
                    )
                )
            )

            # ----------------------------------------------------
            # SOLAR RADIATION PRESSURE
            # ----------------------------------------------------
            propagator.addForceModel(
                SolarRadiationPressure(
                    sun,
                    earth,
                    IsotropicRadiationSingleCoefficient(
                        TOTAL_AREA,
                        cr
                    )
                )
            )

            # ----------------------------------------------------
            # ECLIPSE DETECTOR
            # ----------------------------------------------------
            eclipse_detector = EclipseDetector(
                sun,
                Constants.SUN_RADIUS,
                earth
            ).withUmbra()

            # ----------------------------------------------------
            # SIMULATION LOOP
            # ----------------------------------------------------
            total_seconds = (
                MISSION_DURATION_DAYS *
                86400.0
            )

            t = 0.0

            battery_energy = battery_capacity

            min_soc = 1.0

            total_power = 0.0
            total_soc = 0.0

            steps = 0
            eclipse_steps = 0

            while t <= total_seconds:

                date = start_date.shiftedBy(t)

                state = propagator.propagate(date)

                # ------------------------------------------------
                # POSITION
                # ------------------------------------------------
                pv = state.getPVCoordinates()

                geo = earth.transform(
                    pv.getPosition(),
                    inertial_frame,
                    date
                )

                lat = math.degrees(
                    geo.getLatitude()
                )

                lon = math.degrees(
                    geo.getLongitude()
                )

                altitude_km = (
                    pv.getPosition().getNorm()
                    -
                    Constants
                    .WGS84_EARTH_EQUATORIAL_RADIUS
                ) / 1000.0

                # ------------------------------------------------
                # ECLIPSE
                # ------------------------------------------------
                eclipse = (
                    eclipse_detector.g(state)
                    < 0.0
                )

                if eclipse:
                    eclipse_steps += 1

                # ------------------------------------------------
                # POWER
                # ------------------------------------------------
                power_generated = calculate_power(
                    state,
                    eclipse,
                    tracking_sigma
                )

                total_power += power_generated

                # ------------------------------------------------
                # BATTERY
                # ------------------------------------------------
                dt_hr = STEP_SIZE / 3600.0

                power_net = (
                    power_generated -
                    LOAD_POWER
                )

                battery_energy += (
                    power_net *
                    dt_hr *
                    (battery_eff if power_net >= 0.0 else 1.0 / battery_eff)
                )

                battery_energy = max(
                    0.0,
                    min(
                        battery_capacity,
                        battery_energy
                    )
                )

                soc = (
                    battery_energy /
                    battery_capacity
                )

                min_soc = min(
                    min_soc,
                    soc
                )

                total_soc += soc

                # ------------------------------------------------
                # CONOPS
                # ------------------------------------------------
                saa = check_saa(
                    lat,
                    lon
                )

                gs_visible = check_gs_bits_goa(
                    lat,
                    lon
                )

                # ------------------------------------------------
                # CSV WRITE
                # ------------------------------------------------
                writer.writerow([
                    run,
                    t / 86400.0,
                    altitude_km,
                    lat,
                    lon,
                    eclipse,
                    power_generated,
                    soc,
                    battery_energy,
                    gs_visible,
                    saa
                ])

                # ------------------------------------------------
                # GLOBAL PROGRESS TRACKER
                # ------------------------------------------------
                global_step_counter += 1

                if global_step_counter % 500 == 0:

                    progress_percent = (
                        global_step_counter /
                        TOTAL_GLOBAL_STEPS
                    ) * 100.0

                    elapsed = (
                        time.time()
                        -
                        simulation_start_walltime
                    )

                    steps_per_sec = (
                        global_step_counter / elapsed
                        if elapsed > 0 else 0
                    )

                    remaining_steps = (
                        TOTAL_GLOBAL_STEPS -
                        global_step_counter
                    )

                    eta_sec = (
                        remaining_steps /
                        steps_per_sec
                        if steps_per_sec > 0 else 0
                    )

                    eta_hr = eta_sec / 3600.0

                    print(
                        f"[PROGRESS] "
                        f"{progress_percent:6.2f}% | "
                        f"Battery={battery_capacity:.0f}Wh | "
                        f"MC={run+1}/{MONTE_CARLO_RUNS} | "
                        f"Mission Day={t/86400.0:7.2f} | "
                        f"ETA={eta_hr:6.2f} hr"
                    )

                # ------------------------------------------------
                # STEP
                # ------------------------------------------------
                steps += 1
                t += STEP_SIZE

            # ====================================================
            # RUN METRICS
            # ====================================================
            avg_power = total_power / steps

            avg_soc = total_soc / steps

            eclipse_fraction = (
                eclipse_steps / steps
            )

            # crude lifetime estimate
            lifetime_years = (
                altitude_km - 450.0
            ) / 25.0

            avg_power_runs.append(avg_power)

            eclipse_fraction_runs.append(
                eclipse_fraction
            )

            min_soc_runs.append(min_soc)

            avg_soc_runs.append(avg_soc)

            lifetime_estimate_runs.append(
                lifetime_years
            )

        # ========================================================
        # STATISTICS
        # ========================================================
        stats_rows = []

        stats_rows.append(
            compute_statistics(
                avg_power_runs,
                "Average Power"
            )
        )

        stats_rows.append(
            compute_statistics(
                eclipse_fraction_runs,
                "Eclipse Fraction"
            )
        )

        stats_rows.append(
            compute_statistics(
                min_soc_runs,
                "Minimum SOC"
            )
        )

        stats_rows.append(
            compute_statistics(
                avg_soc_runs,
                "Average SOC"
            )
        )

        stats_rows.append(
            compute_statistics(
                lifetime_estimate_runs,
                "Lifetime Estimate"
            )
        )

        # ========================================================
        # EXPORT STATS
        # ========================================================
        stats_df = pd.DataFrame(
            stats_rows
        )

        stats_filename = os.path.join(
            OUTPUT_DIR,
            f"statistics_{int(battery_capacity)}Wh.csv"
        )

        stats_df.to_csv(
            stats_filename,
            index=False
        )

        print(
            f"\nSaved statistics: "
            f"{stats_filename}"
        )

        for row in stats_rows:

            all_summary_rows.append({
                "Battery_Wh": battery_capacity,
                **row
            })

# ================================================================
# FINAL SUMMARY
# ================================================================
summary_df = pd.DataFrame(
    all_summary_rows
)

summary_filename = os.path.join(
    OUTPUT_DIR,
    "mission_summary.csv"
)

summary_df.to_csv(
    summary_filename,
    index=False
)

# ================================================================
# FINAL RUNTIME
# ================================================================
total_runtime_hr = (
    time.time() -
    simulation_start_walltime
) / 3600.0

# ================================================================
# COMPLETE
# ================================================================
print("\n================================================")
print("ANTARA MASTER SIMULATION COMPLETE")
print("================================================")

print(f"\nResults saved in: {OUTPUT_DIR}")

print("\nGenerated Files:")
print("------------------------------------------------")
print("mission_timeseries_25Wh.csv")
print("mission_timeseries_30Wh.csv")
print("statistics_25Wh.csv")
print("statistics_30Wh.csv")
print("mission_summary.csv")
print("------------------------------------------------")

print(
    f"\nTOTAL SIMULATION RUNTIME: "
    f"{total_runtime_hr:.2f} hr"
)

print("================================================")
