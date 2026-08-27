import orekit
orekit.initVM()

import os
import math
import random
from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

# =========================
# 1. SETUP & DATA
# =========================
if not os.path.isfile("orekit-data.zip"):
    download_orekit_data_curdir()
setup_orekit_curdir()

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
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
# 2. CONFIGURATION
# =========================
# Simulation length: 4 hours (approx 2.5 orbits) is enough to get a stable average
SIM_DURATION = 4 * 3600.0  
STEP_SIZE = 60.0

# 1 Fixed Zenith (-Z) + 2 Deployed Wings
panel_areas = {"+X":0.0, "-X":0.0, "+Y":0.0, "-Y":0.0, "+Z":0.0, "-Z":0.030}
DEPLOYED_PANEL_AREA = 0.030
NUM_DEPLOYED_PANELS = 1

# Hardware Efficiency
SOLAR_CONSTANT = 1361.0
CELL_EFFICIENCY = 0.28
SYSTEM_EFF = 0.75 * 0.80 * 0.90 * 0.95  # ~0.513

# Tracking Error
TRACKING_ERROR_SIGMA = 5.0
TRACKING_BIAS = 2.0

# Orbit
ALTITUDE = 600000.0
INC = 97.4

# =========================
# 3. HELPER FUNCTIONS
# =========================
def safe_normalize(v):
    """Manually normalizes Vector3D to avoid crash on zero-norm or missing method."""
    norm = v.getNorm()
    if norm == 0.0:
        return Vector3D(0.0, 0.0, 0.0)
    return v.scalarMultiply(1.0 / norm)

normals = {
    "+X": Vector3D(1.0, 0.0, 0.0), "-X": Vector3D(-1.0, 0.0, 0.0),
    "+Y": Vector3D(0.0, 1.0, 0.0), "-Y": Vector3D(0.0, -1.0, 0.0),
    "+Z": Vector3D(0.0, 0.0, 1.0), "-Z": Vector3D(0.0, 0.0, -1.0)
}

def calculate_power(state, eclipse):
    if eclipse:
        return 0.0

    # Sun Vector
    sun_pos = CelestialBodyFactory.getSun().getPVCoordinates(state.getDate(), FramesFactory.getEME2000()).getPosition()
    sat_pos = state.getPVCoordinates().getPosition()
    
    # FIX: Use safe_normalize instead of .normalize()
    sun_vec = safe_normalize(sun_pos.subtract(sat_pos))
    sun_body = state.getAttitude().getRotation().applyTo(sun_vec)

    # 1. Fixed Zenith Panel
    fixed_p = 0.0
    for axis, area in panel_areas.items():
        if area > 0:
            cos_theta = Vector3D.dotProduct(normals[axis], sun_body)
            if cos_theta > 0:
                fixed_p += SOLAR_CONSTANT * area * cos_theta * CELL_EFFICIENCY * SYSTEM_EFF

    # 2. Deployed Wings (with error)
    inst_error = random.gauss(TRACKING_BIAS, TRACKING_ERROR_SIGMA)
    tracking_loss = math.cos(math.radians(inst_error))
    deployed_p = SOLAR_CONSTANT * DEPLOYED_PANEL_AREA * NUM_DEPLOYED_PANELS * tracking_loss * CELL_EFFICIENCY * SYSTEM_EFF

    return fixed_p + deployed_p

# =========================
# 4. OPTIMIZATION FUNCTION
# =========================
def run_simulation(test_raan):
    """Runs a short simulation for a specific RAAN and returns stats."""
    
    # Reset Time
    utc = TimeScalesFactory.getUTC()
    date0 = AbsoluteDate(2026, 3, 20, 0, 0, 0.0, utc) # Equinox

    # Setup Orbit with specific RAAN
    mu = Constants.EIGEN5C_EARTH_MU
    a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + ALTITUDE
    initialOrbit = KeplerianOrbit(a, 0.001, math.radians(INC), 0.0, 
                                  math.radians(test_raan), 0.0, PositionAngleType.TRUE, 
                                  FramesFactory.getEME2000(), date0, mu)

    # Propagator
    integrator = DormandPrince853Integrator(0.1, 300.0, 1.0e-3, 1.0e-6)
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(initialOrbit))
    
    # Forces
    earthFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, earthFrame)
    sun = CelestialBodyFactory.getSun()
    
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(earthFrame, GravityFieldFactory.getNormalizedProvider(5, 5)))
    propagator.addForceModel(ThirdBodyAttraction(sun))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMoon()))
    propagator.addForceModel(DragForce(HarrisPriester(sun, earth), IsotropicDrag(0.03, 2.2)))
    propagator.addForceModel(SolarRadiationPressure(sun, earth, IsotropicRadiationSingleCoefficient(0.03, 1.2)))
    propagator.setAttitudeProvider(NadirPointing(FramesFactory.getEME2000(), earth))

    # Eclipse Detector
    eclipseDetector = EclipseDetector(sun, Constants.SUN_RADIUS, earth).withUmbra()
    
    # Run
    t = 0.0
    total_power = 0.0
    steps = 0
    eclipse_steps = 0
    max_p = 0.0

    # Helper for loop
    def check_eclipse(s): return eclipseDetector.g(s) < 0.0
    
    while t <= SIM_DURATION:
        date = date0.shiftedBy(t)
        state = propagator.propagate(date)
        
        is_eclipse = check_eclipse(state)
        p = calculate_power(state, is_eclipse)
        
        total_power += p
        if p > max_p: max_p = p
        if is_eclipse: eclipse_steps += 1
        
        steps += 1
        t += STEP_SIZE

    avg_p = total_power / steps
    ecl_frac = (eclipse_steps / steps) * 100.0
    
    return avg_p, max_p, ecl_frac

# =========================
# 5. MAIN EXECUTION LOOP
# =========================
print(f"Starting RAAN Optimization Analysis...")
print(f"Date: March 20, 2026 (Equinox)")
print(f"-------------------------------------------------------------")
print(f"{'RAAN (deg)':<10} | {'Avg Power (W)':<15} | {'Max Power (W)':<15} | {'Eclipse %':<10}")
print(f"-------------------------------------------------------------")

results = []
best_raan = 0
max_avg_power = 0

# Loop from 0 to 360 in steps of 15 degrees
for r in range(0, 361, 15):
    avg, peak, ecl = run_simulation(float(r))
    results.append((r, avg, peak, ecl))
    
    print(f"{r:<10} | {avg:<15.2f} | {peak:<15.2f} | {ecl:<10.1f}")

    if avg > max_avg_power:
        max_avg_power = avg
        best_raan = r

print(f"-------------------------------------------------------------")
print(f"\nOPTIMAL RESULT:")
print(f"The best RAAN is {best_raan} degrees.")
print(f"Max Average Power: {max_avg_power:.2f} W")