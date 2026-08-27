import math
import csv
import orekit
orekit.initVM()

from paths import init_orekit_data, results_path

init_orekit_data()

from org.orekit.frames import FramesFactory
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.utils import Constants, IERSConventions
from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.propagation.events import EclipseDetector

# --- 1. CONFIGURATION ---
start_date_py = AbsoluteDate(2023, 1, 1, 12, 0, 0.0, TimeScalesFactory.getUTC())
duration = 2592000.0
step = 60.0
output_filename = results_path("conops", "ANTARA_conops_log.csv")

# --- 2. PHYSICS & FRAMES ---
inertial_frame = FramesFactory.getEME2000()
earth_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
earth_body = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                              Constants.WGS84_EARTH_FLATTENING,
                              earth_frame)

# --- 3. ORBIT SETUP ---
orbit = KeplerianOrbit(7000000.0, 0.001, math.radians(98.0), 
                       math.radians(0.0), math.radians(90.0), math.radians(0.0),
                       PositionAngleType.MEAN, inertial_frame, start_date_py, Constants.WGS84_EARTH_MU)

propagator = EcksteinHechlerPropagator(orbit, Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                                       Constants.WGS84_EARTH_MU, Constants.WGS84_EARTH_C20,
                                       0.0, 0.0, 0.0, 0.0)

# --- 4. DETECTORS ---
sun = CelestialBodyFactory.getSun()
eclipse_detector = EclipseDetector(sun, Constants.SUN_RADIUS, earth_body).withUmbra()

# --- 5. HELPER FUNCTIONS ---
def check_saa(lat_deg, lon_deg):
    return (-50.0 <= lat_deg <= 0.0) and (-90.0 <= lon_deg <= 40.0)

def check_gs_bits_goa(lat_deg, lon_deg):
    return (0.0 <= lat_deg <= 20.0) and (60.0 <= lon_deg <= 80.0)

# --- 6. SIMULATION LOOP WITH CSV EXPORT ---
print(f"Starting simulation... Data will be saved to '{output_filename}'")

# Open the CSV file in write mode
with open(output_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the Header Row
    writer.writerow(["Time (s)", "Latitude", "Longitude", "Orbit #", "Eclipse", "In SAA", "GS (Goa)"])

    curr_time = 0.0
    orbit_count = 1
    prev_lat = 0.0
    
    # Initialize previous latitude for orbit counting
    initial_pv = orbit.getPVCoordinates()
    initial_geo = earth_body.transform(initial_pv.getPosition(), inertial_frame, start_date_py)
    prev_lat = math.degrees(initial_geo.getLatitude())

    while curr_time <= duration:
        date = start_date_py.shiftedBy(curr_time)
        pv = propagator.propagate(date).getPVCoordinates()
        
        # Transform X,Y,Z to Lat/Lon
        geo_point = earth_body.transform(pv.getPosition(), inertial_frame, date)
        lat = math.degrees(geo_point.getLatitude())
        lon = math.degrees(geo_point.getLongitude())
        
        # 1. ORBIT COUNTING
        if prev_lat < 0 and lat >= 0:
            orbit_count += 1
        
        # 2. ECLIPSE CHECK
        g_eclipse = eclipse_detector.g(propagator.propagate(date))
        in_eclipse = g_eclipse < 0
        
        # 3. AREA CHECKS
        is_saa = check_saa(lat, lon)
        is_gs_visible = check_gs_bits_goa(lat, lon)
        
        # WRITE ROW TO CSV
        writer.writerow([int(curr_time), round(lat, 2), round(lon, 2), orbit_count, in_eclipse, is_saa, is_gs_visible])
        
        # Optional: Print progress to terminal every 10 steps so you know it's running
        if curr_time % 600 == 0:
            print(f"Processed time: {int(curr_time)} s")

        prev_lat = lat
        curr_time += step

print(f"Done! Check your folder for {output_filename}")
