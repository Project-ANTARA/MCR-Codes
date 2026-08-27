import orekit
orekit.initVM()

import os
import csv
import math
import time
import random
import numpy as np
import pandas as pd

from collections import Counter

from paths import init_orekit_data, results_dir

# ============================================================
# OREKIT DATA
# ============================================================

init_orekit_data()

# ============================================================
# OREKIT IMPORTS
# ============================================================

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants, IERSConventions

from org.orekit.bodies import (
    OneAxisEllipsoid,
    CelestialBodyFactory
)

from org.orekit.attitudes import NadirPointing

from org.orekit.propagation.numerical import (
    NumericalPropagator
)

from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction
)

from org.orekit.forces.gravity.potential import (
    GravityFieldFactory
)

from org.orekit.forces.drag import (
    DragForce,
    IsotropicDrag
)

from org.orekit.models.earth.atmosphere import (
    DTM2000
)

from org.orekit.models.earth.atmosphere.data import (
    CssiSpaceWeatherData
)

from org.orekit.forces.radiation import (
    SolarRadiationPressure,
    IsotropicRadiationSingleCoefficient
)

from org.orekit.propagation.events import (
    EclipseDetector
)

from org.orekit.propagation import (
    SpacecraftState
)

from org.hipparchus.geometry.euclidean.threed import (
    Vector3D
)

from org.hipparchus.ode.nonstiff import (
    DormandPrince853Integrator
)

# ============================================================
# CONFIGURATION
# ============================================================

MISSION_DURATION_DAYS = 365
STEP_SIZE = 300.0
MONTE_CARLO_RUNS = 2

BATTERY_CAPACITY = 25.0

RAAN_VALUES = [
    0, 30, 60, 90, 120, 150, 180
]

ALTITUDE = 600000.0
INCLINATION = 97.4
ECC = 0.001

MASS = 4.0

CD_BASE = 2.2
CR_BASE = 1.2

FIXED_PANEL_AREA = 0.030
DEPLOYED_PANEL_AREA = 0.030
NUM_DEPLOYED_PANELS = 2

TOTAL_AREA = 0.076799

SOLAR_CONSTANT = 1361.0
CELL_EFFICIENCY = 0.28
SYSTEM_EFF = 0.513

LOAD_POWER = 8.0

TRACKING_BIAS = 2.0

ALTITUDE_SIGMA = 5000.0
CD_SIGMA = 0.15
CR_SIGMA = 0.10
BATTERY_EFF_SIGMA = 0.03

REENTRY_ALTITUDE_KM = 300.0

Z_99 = 2.576

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = str(results_dir("mcr"))

# ============================================================
# TIME + FRAMES
# ============================================================

utc = TimeScalesFactory.getUTC()

inertial_frame = FramesFactory.getEME2000()

earth_frame = FramesFactory.getITRF(
    IERSConventions.IERS_2010,
    True
)

earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    earth_frame
)

sun = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()

# ============================================================
# PROGRESS
# ============================================================

TOTAL_STEPS_PER_RUN = int(
    (MISSION_DURATION_DAYS * 86400.0) / STEP_SIZE
)

TOTAL_GLOBAL_STEPS = (
    TOTAL_STEPS_PER_RUN *
    MONTE_CARLO_RUNS *
    len(RAAN_VALUES)
)

global_step_counter = 0

simulation_start_walltime = time.time()

# ============================================================
# HELPERS
# ============================================================

def safe_normalize(v):

    norm = v.getNorm()

    if norm == 0:
        return Vector3D(0, 0, 0)

    return v.scalarMultiply(1.0 / norm)

# ============================================================
# SAA
# ============================================================

def check_saa(lat_deg, lon_deg):

    return (
        -50.0 <= lat_deg <= 0.0
        and
        -90.0 <= lon_deg <= 40.0
    )

# ============================================================
# GS
# ============================================================

def check_gs_bits_goa(lat_deg, lon_deg):

    return (
        0.0 <= lat_deg <= 20.0
        and
        60.0 <= lon_deg <= 80.0
    )

# ============================================================
# STATISTICS
# ============================================================

def compute_statistics(data, metric_name):

    if len(data) == 0:
        return None

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

# ============================================================
# POWER MODEL
# ============================================================

def calculate_power(
    state,
    eclipse,
    tracking_sigma,
    degradation_factor
):

    if eclipse:
        return 0.0, 0.0, 0.0

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

    fixed_normal = Vector3D(
        0.0,
        0.0,
        -1.0
    )

    cos_theta = Vector3D.dotProduct(
        fixed_normal,
        sun_body
    )

    cos_theta = max(
        -1.0,
        min(1.0, cos_theta)
    )

    incidence_angle = math.degrees(
        math.acos(cos_theta)
    )

    fixed_power = 0.0

    if cos_theta > 0:

        fixed_power = (
            SOLAR_CONSTANT *
            FIXED_PANEL_AREA *
            cos_theta *
            CELL_EFFICIENCY *
            SYSTEM_EFF *
            degradation_factor
        )

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
        SYSTEM_EFF *
        degradation_factor
    )

    total_power = fixed_power + deployed_power

    return (
        total_power,
        incidence_angle,
        tracking_loss
    )

# ============================================================
# CSV FILES
# ============================================================

power_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "power_analysis.csv"
    ),
    "w",
    newline=''
)

battery_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "battery_analysis.csv"
    ),
    "w",
    newline=''
)

conops_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "conops_analysis.csv"
    ),
    "w",
    newline=''
)

orbit_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "orbit_classification.csv"
    ),
    "w",
    newline=''
)

drag_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "drag_lifetime_analysis.csv"
    ),
    "w",
    newline=''
)

master_csv = open(
    os.path.join(
        OUTPUT_DIR,
        "raan_master_trade_matrix.csv"
    ),
    "w",
    newline=''
)

# ============================================================
# WRITERS
# ============================================================

power_writer = csv.writer(power_csv)
battery_writer = csv.writer(battery_csv)
conops_writer = csv.writer(conops_csv)
orbit_writer = csv.writer(orbit_csv)
drag_writer = csv.writer(drag_csv)
master_writer = csv.writer(master_csv)

# ============================================================
# HEADERS
# ============================================================

power_writer.writerow([
    "RAAN",
    "run",
    "time_days",
    "orbit",
    "power_generated_w",
    "power_margin_w",
    "incidence_angle_deg",
    "tracking_loss",
    "eclipse",
    "energy_generated_Wh"
])

battery_writer.writerow([
    "RAAN",
    "run",
    "time_days",
    "orbit",
    "SOC",
    "DoD",
    "battery_energy_Wh",
    "capacity_remaining_Wh",
    "cycle_count",
    "throughput_Wh",
    "brownout"
])

conops_writer.writerow([
    "RAAN",
    "run",
    "time_days",
    "orbit",
    "latitude_deg",
    "longitude_deg",
    "eclipse",
    "SAA",
    "groundstation_visible"
])

orbit_writer.writerow([
    "RAAN",
    "run",
    "orbit_number",
    "orbit_type",
    "SAA_count",
    "GS_count",
    "orbit_period_min"
])

drag_writer.writerow([
    "RAAN",
    "lifetime_days",
    "lifetime_years",
    "avg_decay_rate_km_day",
    "final_altitude_km"
])

master_writer.writerow([
    "RAAN",
    "avg_power",
    "avg_soc",
    "min_soc",
    "battery_cycles",
    "lifetime_years",
    "quiet_orbits",
    "only_gs_orbits",
    "only_saa_orbits",
    "mixed_orbits"
])

# ============================================================
# MAIN LOOP
# ============================================================

for RAAN in RAAN_VALUES:

    print("\n================================================")
    print(f"STARTING RAAN {RAAN} DEG")
    print("================================================")

    raan_avg_power = []
    raan_avg_soc = []
    raan_min_soc = []
    raan_cycles = []
    raan_lifetime = []

    orbit_type_counter = Counter()

    for run in range(MONTE_CARLO_RUNS):

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

        tracking_sigma = random.uniform(
            3.0,
            8.0
        )

        start_date = AbsoluteDate(
            2026,
            1,
            1,
            0,
            0,
            0.0,
            utc
        )

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

        gravity_provider = (
            GravityFieldFactory
            .getNormalizedProvider(5, 5)
        )

        propagator.addForceModel(
            HolmesFeatherstoneAttractionModel(
                earth_frame,
                gravity_provider
            )
        )

        propagator.addForceModel(
            ThirdBodyAttraction(sun)
        )

        propagator.addForceModel(
            ThirdBodyAttraction(moon)
        )

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

        eclipse_detector = EclipseDetector(
            sun,
            Constants.SUN_RADIUS,
            earth
        ).withUmbra()

        battery_energy = BATTERY_CAPACITY
        battery_capacity_remaining = BATTERY_CAPACITY

        throughput_wh = 0.0
        cycle_count = 0.0

        total_power = 0.0
        total_soc = 0.0

        min_soc = 1.0

        orbit_number = 1

        previous_lat = 0.0

        orbit_saa_count = 0
        orbit_gs_count = 0

        steps = 0

        total_seconds = (
            MISSION_DURATION_DAYS *
            86400.0
        )

        t = 0.0

        altitude_history = []

        while t <= total_seconds:

            date = start_date.shiftedBy(t)

            state = propagator.propagate(date)

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
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS
            ) / 1000.0

            altitude_history.append(
                altitude_km
            )

            if altitude_km <= REENTRY_ALTITUDE_KM:

                print(
                    f"RAAN {RAAN} reentered "
                    f"at day {t/86400.0:.2f}"
                )

                break

            if previous_lat < 0 and lat >= 0:

                if (
                    orbit_saa_count == 0
                    and
                    orbit_gs_count == 0
                ):
                    orbit_type = "QUIET"

                elif (
                    orbit_saa_count > 0
                    and
                    orbit_gs_count == 0
                ):
                    orbit_type = "ONLY_SAA"

                elif (
                    orbit_saa_count == 0
                    and
                    orbit_gs_count > 0
                ):
                    orbit_type = "ONLY_GS"

                else:
                    orbit_type = "MIXED"

                orbit_writer.writerow([
                    RAAN,
                    run,
                    orbit_number,
                    orbit_type,
                    orbit_saa_count,
                    orbit_gs_count,
                    97.0
                ])

                orbit_type_counter[
                    orbit_type
                ] += 1

                orbit_number += 1

                orbit_saa_count = 0
                orbit_gs_count = 0

            previous_lat = lat

            eclipse = (
                eclipse_detector.g(state)
                < 0.0
            )

            mission_fraction = (
                t / total_seconds
            )

            degradation_factor = (
                1.0 -
                0.05 * mission_fraction
            )

            (
                power_generated,
                incidence_angle,
                tracking_loss
            ) = calculate_power(
                state,
                eclipse,
                tracking_sigma,
                degradation_factor
            )

            power_margin = (
                power_generated -
                LOAD_POWER
            )

            energy_generated = (
                power_generated *
                STEP_SIZE / 3600.0
            )

            total_power += power_generated

            dt_hr = STEP_SIZE / 3600.0

            power_net = (
                power_generated -
                LOAD_POWER
            )

            delta_energy = (
                power_net *
                dt_hr *
                battery_eff
            )

            battery_energy += delta_energy

            throughput_wh += abs(delta_energy)

            cycle_count = (
                throughput_wh /
                (2.0 * BATTERY_CAPACITY)
            )

            battery_capacity_remaining = (
                BATTERY_CAPACITY *
                (
                    1.0 -
                    0.00005 * cycle_count
                )
            )

            battery_energy = max(
                0.0,
                min(
                    battery_capacity_remaining,
                    battery_energy
                )
            )

            soc = (
                battery_energy /
                battery_capacity_remaining
            )

            dod = 1.0 - soc

            min_soc = min(
                min_soc,
                soc
            )

            total_soc += soc

            brownout = soc < 0.20

            saa = check_saa(
                lat,
                lon
            )

            gs_visible = check_gs_bits_goa(
                lat,
                lon
            )

            if saa:
                orbit_saa_count += 1

            if gs_visible:
                orbit_gs_count += 1

            power_writer.writerow([
                RAAN,
                run,
                t/86400.0,
                orbit_number,
                power_generated,
                power_margin,
                incidence_angle,
                tracking_loss,
                eclipse,
                energy_generated
            ])

            battery_writer.writerow([
                RAAN,
                run,
                t/86400.0,
                orbit_number,
                soc,
                dod,
                battery_energy,
                battery_capacity_remaining,
                cycle_count,
                throughput_wh,
                brownout
            ])

            conops_writer.writerow([
                RAAN,
                run,
                t/86400.0,
                orbit_number,
                lat,
                lon,
                eclipse,
                saa,
                gs_visible
            ])

            global_step_counter += 1

            if global_step_counter % 200 == 0:

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
                    f"RAAN={RAAN} | "
                    f"MC={run+1}/{MONTE_CARLO_RUNS} | "
                    f"Orbit={orbit_number} | "
                    f"Mission Day={t/86400.0:7.2f} | "
                    f"ETA={eta_hr:6.2f} hr"
                )

            steps += 1

            t += STEP_SIZE

        lifetime_days = t / 86400.0

        lifetime_years = (
            lifetime_days / 365.25
        )

        decay_rate = (
            (ALTITUDE / 1000.0 - altitude_km)
            /
            max(1.0, lifetime_days)
        )

        drag_writer.writerow([
            RAAN,
            lifetime_days,
            lifetime_years,
            decay_rate,
            altitude_km
        ])

        avg_power = total_power / max(1, steps)
        avg_soc = total_soc / max(1, steps)

        raan_avg_power.append(avg_power)
        raan_avg_soc.append(avg_soc)
        raan_min_soc.append(min_soc)
        raan_cycles.append(cycle_count)
        raan_lifetime.append(lifetime_years)

    master_writer.writerow([
        RAAN,
        np.mean(raan_avg_power),
        np.mean(raan_avg_soc),
        np.mean(raan_min_soc),
        np.mean(raan_cycles),
        np.mean(raan_lifetime),
        orbit_type_counter["QUIET"],
        orbit_type_counter["ONLY_GS"],
        orbit_type_counter["ONLY_SAA"],
        orbit_type_counter["MIXED"]
    ])

# ============================================================
# FINAL STATISTICS
# ============================================================

stats_rows = []

for stat in [
    compute_statistics(
        raan_avg_power,
        "Average Power"
    ),
    compute_statistics(
        raan_avg_soc,
        "Average SOC"
    ),
    compute_statistics(
        raan_min_soc,
        "Minimum SOC"
    ),
    compute_statistics(
        raan_cycles,
        "Battery Cycles"
    ),
    compute_statistics(
        raan_lifetime,
        "Lifetime Years"
    )
]:

    if stat is not None:
        stats_rows.append(stat)

stats_df = pd.DataFrame(
    stats_rows
)

stats_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "mission_statistics.csv"
    ),
    index=False
)

# ============================================================
# CLOSE FILES
# ============================================================

power_csv.close()
battery_csv.close()
conops_csv.close()
orbit_csv.close()
drag_csv.close()
master_csv.close()

# ============================================================
# FINAL RUNTIME
# ============================================================

total_runtime_hr = (
    time.time()
    -
    simulation_start_walltime
) / 3600.0

print("\n================================================")
print("ANTARA MEGA SYSTEM SIMULATOR COMPLETE")
print("================================================")

print(f"\nTOTAL RUNTIME: {total_runtime_hr:.2f} hr")

print("\nGenerated Files:")
print("------------------------------------------------")
print("power_analysis.csv")
print("battery_analysis.csv")
print("conops_analysis.csv")
print("orbit_classification.csv")
print("drag_lifetime_analysis.csv")
print("raan_master_trade_matrix.csv")
print("mission_statistics.csv")
print("------------------------------------------------")
