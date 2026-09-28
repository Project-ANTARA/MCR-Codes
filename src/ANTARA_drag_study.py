import orekit
orekit.initVM()

from paths import init_orekit_data

init_orekit_data()

# ==================================================
# OREKIT IMPORTS
# ==================================================
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.utils import Constants, IERSConventions
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.models.earth.atmosphere.data import CssiSpaceWeatherData
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.propagation import SpacecraftState
from org.orekit.bodies import CelestialBodyFactory

try:
    from org.orekit.models.earth import OneAxisEllipsoid
except ImportError:
    from org.orekit.bodies import OneAxisEllipsoid

import math

# ==================================================
# CONSTANTS
# ==================================================
MU = Constants.WGS84_EARTH_MU
RE = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

MASS = 4.0        # kg
CD = 2.2

TUMBLING_FACTOR = 1   # ⬅️ automatic tumbling
END_ALT = 450000.0     # m

# ==================================================
# LIFETIME FUNCTION
# ==================================================
def lifetime(area, start_alt_m):

    utc = TimeScalesFactory.getUTC()

    # 🌞 SOLAR MAX EPOCH (Cycle 25 peak)
    start_date = AbsoluteDate(2000, 1, 1, 12, 0, 0.0, utc)

    inertial = FramesFactory.getEME2000()
    earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    a0 = RE + start_alt_m
    orbit = KeplerianOrbit(
        a0,
        1e-4,
        math.radians(98.0),
        0.0, 0.0, 0.0,
        PositionAngleType.TRUE,
        inertial,
        start_date,
        MU
    )

    sun = CelestialBodyFactory.getSun()
    earth = OneAxisEllipsoid(
        RE,
        Constants.WGS84_EARTH_FLATTENING,
        earth_frame
    )

    # Atmosphere (solar max included via epoch)
    space_weather = CssiSpaceWeatherData(
        CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES
    )
    atmosphere = DTM2000(space_weather, sun, earth)

    # Fast propagator
    integrator = ClassicalRungeKuttaIntegrator(60.0)
    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(orbit, MASS))

    # 💡 Tumbling-averaged area
    A_eff = TUMBLING_FACTOR * area

    propagator.addForceModel(
        DragForce(atmosphere, IsotropicDrag(A_eff, CD))
    )

    gravity = GravityFieldFactory.getNormalizedProvider(2, 0)
    propagator.addForceModel(
        HolmesFeatherstoneAttractionModel(earth_frame, gravity)
    )

    state = propagator.getInitialState()
    step = 86400.0  # 1 day
    max_days = 30 * 365  # 30-year cap

    for _ in range(max_days):
        state = propagator.propagate(
            state.getDate().shiftedBy(step)
        )

        r = state.getPVCoordinates().getPosition().getNorm()
        alt = r - RE

        if alt < END_ALT:
            return state.getDate().durationFrom(start_date)

    return float('inf')  # Right-censored: threshold not reached within 30 years.

# ==================================================
# ALTITUDE SWEEP
# ==================================================
areas = {
    "4 fixed panels": 0.030,
    "1 fixed + 1 deployed": 0.060,
    "1 fixed + 2 deployed": 0.090,
    "4 deployed only": 0.150
}

altitudes_km = range(500, 601, 10)

print("\n===== LIFETIME vs ALTITUDE (SOLAR MAX + TUMBLING) =====")
print("End altitude: 450 km | Tumbling OFF(max drag area) | Solar Max")
print("------------------------------------------------------")

for name, area in areas.items():
    print(f"\n{name}")
    print("Alt (km) | Years")
    print("---------|------")
    for h in altitudes_km:
        T = lifetime(area, h * 1000.0)
        years = T / (365.25 * 86400.0)
        print(f"{h:<8d} | {years:5.2f}" if math.isfinite(years) else f"{h:<8d} | >30 (censored)")
