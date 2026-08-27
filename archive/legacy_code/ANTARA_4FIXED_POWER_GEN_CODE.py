import orekit
orekit.initVM()

import os
import math
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

# =========================
# 1. OREKIT DATA
# =========================
if not os.path.isfile("orekit-data.zip"):
    print("Downloading Orekit Data...")
    download_orekit_data_curdir()
setup_orekit_curdir()
print("Orekit data loaded.")

# =========================
# 2. IMPORTS
# =========================
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType, OrbitType
from org.orekit.utils import Constants, IERSConventions
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.attitudes import NadirPointing
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, ThirdBodyAttraction
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient
from org.orekit.propagation.events import EclipseDetector
from org.orekit.propagation import SpacecraftState
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator

# =========================
# 3. USER SETTINGS
# =========================
SIM_DURATION = 12 * 3600.0
STEP_SIZE = 60.0

ALTITUDE = 600000.0
INC = 97.4
RAAN = 0.0

panel_areas = {
    "+X": 0.030, "-X": 0.030,
    "+Y": 0.030, "-Y": 0.030,
    "+Z": 0.008, "-Z": 0.008 
}

SOLAR_CONSTANT = 1361.0
CELL_EFFICIENCY = 0.28   # Triple junction typical

# ---- EOL + REALISTIC LOSSES ----
PACKING_EFF   = 0.75     # 60% as you requested
RADIATION_EFF = 0.80     # EOL degradation
TEMP_EFF      = 0.90     # temperature losses
WIRING_EFF    = 0.95     # EPS + harness

SYSTEM_EFF = PACKING_EFF * RADIATION_EFF * TEMP_EFF * WIRING_EFF

# =========================
# 4. TIME, FRAMES, BODIES
# =========================
utc = TimeScalesFactory.getUTC()
date0 = AbsoluteDate(2026, 3, 20, 0, 0, 0.0, utc)

inertialFrame = FramesFactory.getEME2000()
earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                         Constants.WGS84_EARTH_FLATTENING,
                         earthFrame)

sun = CelestialBodyFactory.getSun()
moon = CelestialBodyFactory.getMoon()

# =========================
# 5. ORBIT
# =========================
a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + ALTITUDE
mu = Constants.EIGEN5C_EARTH_MU

initialOrbit = KeplerianOrbit(
    a, 0.001, math.radians(INC),
    0.0, math.radians(RAAN), 0.0,
    PositionAngleType.TRUE, inertialFrame, date0, mu
)

# =========================
# 6. NUMERICAL PROPAGATOR (PYTHON SAFE – NO jarray)
# =========================
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState

minStep = 0.1
maxStep = 300.0

# Scalar tolerances (THIS IS THE KEY FIX)
absTol = 1.0e-3
relTol = 1.0e-6

integrator = DormandPrince853Integrator(minStep, maxStep, absTol, relTol)

propagator = NumericalPropagator(integrator)
propagator.setInitialState(SpacecraftState(initialOrbit))

# Gravity J2-J5
gravityProvider = GravityFieldFactory.getNormalizedProvider(5, 5)
propagator.addForceModel(HolmesFeatherstoneAttractionModel(earthFrame, gravityProvider))

# Third body
propagator.addForceModel(ThirdBodyAttraction(sun))
propagator.addForceModel(ThirdBodyAttraction(moon))

# SRP (FIXED)
from org.orekit.forces.radiation import SolarRadiationPressure
from org.orekit.forces.radiation import IsotropicRadiationSingleCoefficient
# Drag
atmosphere = HarrisPriester(sun, earth)
propagator.addForceModel(DragForce(atmosphere, IsotropicDrag(0.03, 2.2)))

# SRP (CORRECT for Python)
radiationModel = IsotropicRadiationSingleCoefficient(0.03, 1.2)
srp = SolarRadiationPressure(sun, earth, radiationModel)
propagator.addForceModel(srp)


# Attitude
propagator.setAttitudeProvider(NadirPointing(inertialFrame, earth))

# =========================
# 7. ECLIPSE DETECTOR (REAL)
# =========================
eclipseDetector = EclipseDetector(sun, Constants.SUN_RADIUS, earth).withUmbra()

def in_eclipse(state):
    return eclipseDetector.g(state) < 0.0

# =========================
# 8. POWER MODEL
# =========================
normals = {
    "+X": Vector3D(1.0, 0.0, 0.0), "-X": Vector3D(-1.0, 0.0, 0.0),
    "+Y": Vector3D(0.0, 1.0, 0.0), "-Y": Vector3D(0.0, -1.0, 0.0),
    "+Z": Vector3D(0.0, 0.0, 1.0), "-Z": Vector3D(0.0, 0.0, -1.0)
}

def calculate_power(state, eclipse):
    if eclipse:
        return 0.0

    sun_pos = sun.getPVCoordinates(state.getDate(), inertialFrame).getPosition()
    sat_pos = state.getPVCoordinates().getPosition()

    sun_vec_raw = sun_pos.subtract(sat_pos)
    sun_vec = sun_vec_raw.scalarMultiply(1.0 / sun_vec_raw.getNorm())

    sun_body = state.getAttitude().getRotation().applyTo(sun_vec)

    total = 0.0

    for axis in ["X", "Y", "Z"]:
        cos_p = Vector3D.dotProduct(normals["+"+axis], sun_body)
        cos_m = Vector3D.dotProduct(normals["-"+axis], sun_body)
        cos_eff = max(cos_p, cos_m, 0.0)

        area = panel_areas.get("+"+axis, 0.0)
        total += SOLAR_CONSTANT * area * cos_eff * CELL_EFFICIENCY * SYSTEM_EFF

    return total

# =========================
# 9. SIMULATION
# =========================
print("\nStarting 12-hour simulation...\n")

t = 0.0
data = []

peak_power = 0.0
peak_time = 0.0

eclipse_intervals = []
in_eclipse_prev = False
eclipse_start = None

while t <= SIM_DURATION:
    date = date0.shiftedBy(t)
    state = propagator.propagate(date)

    eclipse = in_eclipse(state)
    power = calculate_power(state, eclipse)

    # ---- Peak power tracking ----
    if power > peak_power:
        peak_power = power
        peak_time = t

    # ---- Eclipse interval tracking ----
    if eclipse and not in_eclipse_prev:
        eclipse_start = t

    if not eclipse and in_eclipse_prev:
        eclipse_intervals.append((eclipse_start, t))

    in_eclipse_prev = eclipse

    data.append((t, power, eclipse))
    t += STEP_SIZE

# Close last eclipse if sim ends in eclipse
if in_eclipse_prev:
    eclipse_intervals.append((eclipse_start, SIM_DURATION))

# =========================
# 10. RESULTS
# =========================
eclipse_steps = sum(1 for d in data if d[2])
total_steps = len(data)

eclipse_pct = 100.0 * eclipse_steps / total_steps
avg_orbit_power = sum(d[1] for d in data) / total_steps
sunlit_powers = [d[1] for d in data if not d[2]]
avg_sunlit_power = sum(sunlit_powers) / len(sunlit_powers)

print("="*40)
print("       FINAL RESULTS")
print("="*40)
print(f"Eclipse fraction     : {eclipse_pct:.1f} %")
print(f"Avg power (orbit)    : {avg_orbit_power:.2f} W")
print(f"Avg power (sunlit)   : {avg_sunlit_power:.2f} W")
print("="*40)
print("\n--- PEAK POWER ---")
print(f"Peak power       : {peak_power:.2f} W")
print(f"Time of peak     : {peak_time/3600:.2f} hours")

print("\n--- ECLIPSE INTERVALS ---")
for idx, (start, end) in enumerate(eclipse_intervals, 1):
    print(f"Eclipse {idx}: {start/60:.1f} min  ->  {end/60:.1f} min  (Duration = {(end-start)/60:.1f} min)")

print("\n--- First 2 Orbits Timeline ---")
for d in data:
    if d[0] > 2 * 5400:   # ~2 LEO orbits
        break
    status = "ECLIPSE" if d[2] else "SUNLIT "
    print(f"T={d[0]/60:6.1f} min | {status} | P={d[1]:6.2f} W")