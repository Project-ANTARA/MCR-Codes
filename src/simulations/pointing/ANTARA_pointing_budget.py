# Support direct execution and qualified simulations.* imports.
import sys as _sys
from pathlib import Path as _Path
if __package__ in (None, ""):
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import orekit
orekit.initVM()

from simulations.helper.paths import figures_path, init_orekit_data

init_orekit_data()

from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.bodies import OneAxisEllipsoid, GeodeticPoint
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants, IERSConventions
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import Constants
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.attitudes import NadirPointing
from org.hipparchus.geometry.euclidean.threed import Vector3D
from math import radians, degrees
import matplotlib.pyplot as plt
# =============================
# 1️⃣ TIME AND FRAMES
# =============================

utc = TimeScalesFactory.getUTC()
start_date = AbsoluteDate(2026, 1, 1, 12, 0, 0.0, utc)

inertialFrame = FramesFactory.getEME2000()
earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    earthFrame
)

# =============================
# 2️⃣ ORBIT (575 km SSO)
# =============================

a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + 600000.0
e = 0.0
i = radians(97.4)
raan = radians(0.0)
argPerigee = 0.0
meanAnomaly = 0.0

orbit = KeplerianOrbit(
    a, e, i, argPerigee, raan, meanAnomaly,
    PositionAngleType.MEAN,
    inertialFrame,
    start_date,
    Constants.EGM96_EARTH_MU
)

propagator = KeplerianPropagator(orbit)

# =============================
# 3️⃣ NADIR ATTITUDE
# =============================

attitudeProvider = NadirPointing(inertialFrame, earth)
propagator.setAttitudeProvider(attitudeProvider)

# =============================
# 4️⃣ GROUND STATION
# =============================

stationLat = radians(15.0)
stationLon = radians(74.0)
stationAlt = 30.0

stationPoint = GeodeticPoint(stationLat, stationLon, stationAlt)
stationFrame = TopocentricFrame(earth, stationPoint, "GS")

# =============================
# 5️⃣ PROPAGATION LOOP
# =============================

simulation_duration = 86400.0   # 1 day
step = 10.0

currentDate = start_date

times = []
angles = []
elevations = []

in_contact = False
access_start = None
total_access_time = 0.0
pass_durations = []
max_off_boresight = 0.0

elevation_mask = 10.0  # degrees

while currentDate.durationFrom(start_date) < simulation_duration:

    state = propagator.propagate(currentDate)

    # Satellite position in Earth frame
    pv = state.getPVCoordinates(earthFrame)
    satPos = pv.getPosition()

    # Elevation check
    elevation = degrees(
        stationFrame.getElevation(
            satPos, earthFrame, currentDate))

    if elevation > elevation_mask:

        # --- LOS VECTOR IN INERTIAL FRAME ---
        gsPos_inertial = stationFrame.getPVCoordinates(
            currentDate, inertialFrame).getPosition()

        satPos_inertial = state.getPVCoordinates(
            inertialFrame).getPosition()

        los_inertial = gsPos_inertial.subtract(satPos_inertial)

        # --- TRANSFORM LOS INTO SPACECRAFT BODY FRAME ---
        attitude = state.getAttitude()
        rot = attitude.getRotation()

        los_body = rot.applyTo(los_inertial)

        # Antenna boresight (+Z axis)
        boresight = Vector3D.PLUS_K

        angle = degrees(Vector3D.angle(boresight, los_body))

        times.append(currentDate.durationFrom(start_date) / 60.0)
        angles.append(angle)
        elevations.append(elevation)

        if not in_contact:
            access_start = currentDate
            in_contact = True

        max_off_boresight = max(max_off_boresight, angle)

    else:
        if in_contact:
            duration = currentDate.durationFrom(access_start)
            pass_durations.append(duration)
            total_access_time += duration
            in_contact = False

    currentDate = currentDate.shiftedBy(step)

# Catch case if still in contact at end of simulation
if in_contact:
    duration = currentDate.durationFrom(access_start)
    pass_durations.append(duration)
    total_access_time += duration


print("Number of passes:", len(pass_durations))
print("Total access time (minutes):", total_access_time / 60.0)

if pass_durations:
    print("Average pass duration (minutes):",
          (total_access_time / len(pass_durations)) / 60.0)

print("Maximum off-boresight angle (deg):", max_off_boresight)

# =============================
# 7️⃣ PLOT
# =============================

plt.figure()
plt.plot(times, angles)
plt.xlabel("Time since start (minutes)")
plt.ylabel("Off-boresight angle (deg)")
plt.title("Off-Boresight Angle During Pass")
plt.grid(True)
plt.savefig(figures_path("pointing", "boresight1.png"), dpi=300)
plt.show()
