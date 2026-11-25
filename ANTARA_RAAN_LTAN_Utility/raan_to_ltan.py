#!/usr/bin/env python3
"""
ANTARA RAAN to LTAN Conversion Utility

A standalone interactive tool for converting Right Ascension of the Ascending Node (RAAN)
values to Local Time of the Ascending Node (LTAN) for the ANTARA 3U CubeSat mission.

This utility performs an instantaneous RAAN-to-LTAN conversion at a specified epoch using
a low-precision mean-Sun astronomical model. It does not propagate orbits, calculate
J2 precession, launch windows, or ground-station visibility.

Usage:
    python3 raan_to_ltan.py

Author: ANTARA Mission Team
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple


# Default mission parameters
DEFAULT_EPOCH = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
DEFAULT_ALTITUDE_KM = 600.0
DEFAULT_INCLINATION_DEG = 97.6
DEFAULT_ECCENTRICITY = 0.0
DEFAULT_ORBIT_TYPE = "Circular SSO"
DEFAULT_LAUNCH_SITE = "Sriharikota, India"
DEFAULT_LAUNCH_PROVIDER = "TBD"
DEFAULT_GROUND_STATION = "BITS Goa, India"


class MissionParameters(NamedTuple):
    """Container for mission configuration parameters."""
    epoch: datetime
    altitude_km: float
    inclination_deg: float
    eccentricity: float
    orbit_type: str
    launch_site: str
    launch_provider: str
    ground_station: str


class SolarReference(NamedTuple):
    """Container for calculated solar reference values."""
    julian_date: float
    julian_centuries: float
    solar_ra_deg: float


class RAANLTANResult(NamedTuple):
    """Container for a single RAAN-to-LTAN conversion result."""
    original_raan: float
    normalized_raan: float
    ltan_decimal: float
    ltan_hms: str


def julian_date_from_utc(epoch: datetime) -> float:
    """
    Convert a UTC datetime to Julian Date (JD).

    Uses the standard algorithm for Gregorian calendar dates.
    Valid for dates after 1582-10-15.

    Args:
        epoch: UTC datetime object (timezone-aware).

    Returns:
        Julian Date as a float.
    """
    if epoch.tzinfo is None or epoch.tzinfo.utcoffset(epoch) is None:
        raise ValueError("Epoch must be timezone-aware UTC datetime")

    year = epoch.year
    month = epoch.month
    day = epoch.day
    hour = epoch.hour
    minute = epoch.minute
    second = epoch.second + epoch.microsecond / 1_000_000.0

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + a // 4

    jd = (int(365.25 * (year + 4716)) +
          int(30.6001 * (month + 1)) +
          day + b - 1524.5)

    # Add fractional day
    jd += (hour + minute / 60.0 + second / 3600.0) / 24.0

    return jd


def calculate_julian_centuries(julian_date: float) -> float:
    """
    Calculate Julian centuries from J2000.0 epoch.

    Args:
        julian_date: Julian Date.

    Returns:
        Centuries since J2000.0 (T).
    """
    return (julian_date - 2451545.0) / 36525.0


def calculate_mean_sun_right_ascension(julian_centuries: float) -> float:
    """
    Calculate the Sun's mean right ascension (in degrees) at a given epoch.

    This implements a low-precision mean-Sun model based on:
    - Meeus, "Astronomical Algorithms", Chapter 25 (Sun's Position)
    - Uses mean longitude and mean anomaly; applies equation of center
    - Converts ecliptic longitude to equatorial coordinates using mean obliquity

    Expected accuracy: ~0.01 degrees (approx 0.4 seconds of time) for several
    decades around J2000. For high-precision mission design, validate against
    a trusted ephemeris (e.g., JPL DE430, SPICE).

    Args:
        julian_centuries: Centuries since J2000.0 (T).

    Returns:
        Solar mean right ascension in degrees, normalized to [0, 360).
    """
    T = julian_centuries

    # Sun's mean longitude (L0) in degrees (Meeus Eq. 25.2)
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T

    # Sun's mean anomaly (M) in degrees (Meeus Eq. 25.3)
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T

    # Equation of center (C) in degrees (Meeus Eq. 25.4, truncated)
    M_rad = math.radians(M)
    C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M_rad) + \
        (0.019993 - 0.000101 * T) * math.sin(2 * M_rad) + \
        0.000289 * math.sin(3 * M_rad)

    # Sun's true longitude (lambda_sun) in degrees
    lambda_sun = L0 + C

    # Mean obliquity of the ecliptic (epsilon) in degrees (Meeus Eq. 22.3)
    epsilon = (23.4392911111 -
               46.8150 / 3600.0 * T -
               0.00059 / 3600.0 * T * T +
               0.001813 / 3600.0 * T * T * T)

    epsilon_rad = math.radians(epsilon)
    lambda_sun_rad = math.radians(lambda_sun)

    # Convert ecliptic longitude to equatorial right ascension
    # Using atan2(cos(epsilon) * sin(lambda), cos(lambda))
    y = math.cos(epsilon_rad) * math.sin(lambda_sun_rad)
    x = math.cos(lambda_sun_rad)
    ra_rad = math.atan2(y, x)

    ra_deg = math.degrees(ra_rad)

    # Normalize to [0, 360)
    ra_deg = ra_deg % 360.0

    return ra_deg


def normalize_raan(raan_deg: float) -> float:
    """
    Normalize RAAN to the range [0, 360) degrees.

    Args:
        raan_deg: RAAN value in degrees (can be any real number).

    Returns:
        Normalized RAAN in [0, 360).
    """
    if not math.isfinite(raan_deg):
        raise ValueError(f"RAAN value must be finite, got {raan_deg}")
    return raan_deg % 360.0


def calculate_ltan(raan_deg: float, solar_ra_deg: float) -> float:
    """
    Calculate LTAN from RAAN and solar right ascension.

    Formula: LTAN = [12 + (Ω - α☉) / 15] mod 24

    Where:
        Ω = RAAN in degrees
        α☉ = Solar right ascension in degrees
        LTAN = Local Mean Solar Time at ascending node in hours

    Args:
        raan_deg: Normalized RAAN in degrees [0, 360).
        solar_ra_deg: Solar right ascension in degrees [0, 360).

    Returns:
        LTAN in decimal hours, normalized to [0, 24).
    """
    ltan = 12.0 + (raan_deg - solar_ra_deg) / 15.0
    return ltan % 24.0


def decimal_hours_to_hms(decimal_hours: float) -> str:
    """
    Convert decimal hours to HH:MM:SS format.

    Handles midnight rollover correctly (24:00:00 -> 00:00:00).

    Args:
        decimal_hours: Time in decimal hours [0, 24).

    Returns:
        Formatted string "HH:MM:SS".
    """
    if not math.isfinite(decimal_hours):
        raise ValueError(f"Decimal hours must be finite, got {decimal_hours}")

    # Normalize to [0, 24)
    decimal_hours = decimal_hours % 24.0

    total_seconds = round(decimal_hours * 3600)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    # Handle midnight rollover from rounding
    if hours == 24:
        hours = 0

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_raan_input(input_str: str) -> list[float]:
    """
    Parse user input string into a list of RAAN values.

    Accepts comma-separated and/or whitespace-separated values.
    Supports decimal values, negative values, and values > 360.

    Args:
        input_str: Raw input string from user.

    Returns:
        List of parsed RAAN values as floats.

    Raises:
        ValueError: If any value is invalid or non-finite.
    """
    if not input_str.strip():
        raise ValueError("No RAAN values provided")

    # Replace commas with spaces, then split on whitespace
    tokens = input_str.replace(',', ' ').split()

    raan_values = []
    for token in tokens:
        try:
            value = float(token)
            if not math.isfinite(value):
                raise ValueError(f"Non-finite RAAN value: {token}")
            raan_values.append(value)
        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError(f"Invalid RAAN value: '{token}'")
            raise

    return raan_values


def get_user_input(prompt: str, default: str = "") -> str:
    """
    Get user input with optional default value.

    Args:
        prompt: Prompt to display.
        default: Default value if user presses Enter.

    Returns:
        User input string (or default).
    """
    if default:
        full_prompt = f"{prompt} [{default}]: "
    else:
        full_prompt = f"{prompt}: "

    try:
        user_input = input(full_prompt).strip()
        return user_input if user_input else default
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)


def validate_epoch_input(date_str: str, time_str: str) -> datetime:
    """
    Parse and validate epoch date and time strings.

    Args:
        date_str: Date in YYYY-MM-DD format.
        time_str: Time in HH:MM:SS format.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: If format is invalid or date/time is out of range.
    """
    try:
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid date/time format: {e}")


def validate_orbital_params(altitude_km: float, inclination_deg: float,
                             eccentricity: float) -> None:
    """
    Validate orbital parameters for physical plausibility.

    Args:
        altitude_km: Orbit altitude in km.
        inclination_deg: Inclination in degrees.
        eccentricity: Orbital eccentricity.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if altitude_km < 0:
        raise ValueError(f"Altitude must be non-negative, got {altitude_km} km")
    if altitude_km > 100000:
        raise ValueError(f"Altitude unrealistically high: {altitude_km} km")

    if not (0 <= inclination_deg <= 180):
        raise ValueError(f"Inclination must be in [0, 180], got {inclination_deg}")

    if not (0 <= eccentricity < 1):
        raise ValueError(f"Eccentricity must be in [0, 1), got {eccentricity}")


def get_mission_parameters() -> MissionParameters:
    """
    Interactively collect all mission parameters from the user.

    Returns:
        MissionParameters named tuple with all values.
    """
    print("\n" + "=" * 60)
    print("           ANTARA RAAN TO LTAN CONVERTER")
    print("=" * 60)
    print("\nEnter mission parameters (press Enter for defaults):\n")

    # A. Epoch information
    print("--- Epoch (UTC) ---")
    default_date = DEFAULT_EPOCH.strftime("%Y-%m-%d")
    default_time = DEFAULT_EPOCH.strftime("%H:%M:%S")

    while True:
        date_str = get_user_input("Epoch date (YYYY-MM-DD)", default_date)
        time_str = get_user_input("Epoch time (HH:MM:SS)", default_time)
        try:
            epoch = validate_epoch_input(date_str, time_str)
            break
        except ValueError as e:
            print(f"  Error: {e}. Please try again.")

    # B. Orbital parameters
    print("\n--- Orbital Parameters ---")
    while True:
        try:
            alt_str = get_user_input("Orbit altitude (km)", str(DEFAULT_ALTITUDE_KM))
            altitude_km = float(alt_str)

            inc_str = get_user_input("Inclination (deg)", str(DEFAULT_INCLINATION_DEG))
            inclination_deg = float(inc_str)

            ecc_str = get_user_input("Eccentricity", str(DEFAULT_ECCENTRICITY))
            eccentricity = float(ecc_str)

            orbit_type = get_user_input("Orbit type", DEFAULT_ORBIT_TYPE)

            validate_orbital_params(altitude_km, inclination_deg, eccentricity)
            break
        except ValueError as e:
            print(f"  Error: {e}. Please try again.")

    # C. Mission metadata
    print("\n--- Mission Metadata (informational only) ---")
    launch_site = get_user_input("Launch site", DEFAULT_LAUNCH_SITE)
    launch_provider = get_user_input("Launch provider", DEFAULT_LAUNCH_PROVIDER)
    ground_station = get_user_input("Ground station", DEFAULT_GROUND_STATION)

    return MissionParameters(
        epoch=epoch,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        eccentricity=eccentricity,
        orbit_type=orbit_type,
        launch_site=launch_site,
        launch_provider=launch_provider,
        ground_station=ground_station,
    )


def get_raan_values() -> list[float]:
    """
    Interactively collect RAAN values from the user.

    Returns:
        List of original RAAN values as floats.
    """
    print("\n--- RAAN Values ---")
    print("Enter RAAN values (degrees), separated by commas or spaces.")
    print("Supports decimals, negative values, and values > 360.")

    while True:
        input_str = get_user_input("RAAN values")
        try:
            raan_values = parse_raan_input(input_str)
            if not raan_values:
                print("  Error: At least one RAAN value is required. Please try again.")
                continue
            return raan_values
        except ValueError as e:
            print(f"  Error: {e}. Please try again.")


def compute_solar_reference(epoch: datetime) -> SolarReference:
    """
    Compute all solar reference values for the given epoch.

    Args:
        epoch: UTC datetime.

    Returns:
        SolarReference named tuple.
    """
    jd = julian_date_from_utc(epoch)
    jc = calculate_julian_centuries(jd)
    solar_ra = calculate_mean_sun_right_ascension(jc)
    return SolarReference(julian_date=jd, julian_centuries=jc, solar_ra_deg=solar_ra)


def convert_raan_list(raan_values: list[float], solar_ra_deg: float) -> list[RAANLTANResult]:
    """
    Convert a list of RAAN values to LTAN results.

    Args:
        raan_values: List of original RAAN values.
        solar_ra_deg: Solar right ascension in degrees.

    Returns:
        List of RAANLTANResult named tuples.
    """
    results = []
    for original_raan in raan_values:
        normalized_raan = normalize_raan(original_raan)
        ltan_decimal = calculate_ltan(normalized_raan, solar_ra_deg)
        ltan_hms = decimal_hours_to_hms(ltan_decimal)
        results.append(RAANLTANResult(
            original_raan=original_raan,
            normalized_raan=normalized_raan,
            ltan_decimal=ltan_decimal,
            ltan_hms=ltan_hms
        ))
    return results


def format_results_table(results: list[RAANLTANResult]) -> str:
    """
    Format the results as a pretty table string.

    Args:
        results: List of RAANLTANResult.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append("RAAN TO LTAN RESULTS")
    lines.append("-" * 60)
    lines.append(f"{'Original RAAN':>14}  {'Normalized RAAN':>16}  "
                 f"{'LTAN (decimal)':>15}  {'LTAN (HH:MM:SS)':>14}")
    lines.append("-" * 60)

    for r in results:
        lines.append(f"{r.original_raan:14.4f}  {r.normalized_raan:16.4f}  "
                     f"{r.ltan_decimal:15.6f}  {r.ltan_hms:>14}")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_full_output(params: MissionParameters,
                       solar: SolarReference,
                       results: list[RAANLTANResult]) -> str:
    """
    Format the complete output for display and file writing.

    Args:
        params: Mission parameters.
        solar: Solar reference values.
        results: Conversion results.

    Returns:
        Complete formatted output string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("         ANTARA RAAN TO LTAN CONVERTER")
    lines.append("=" * 60)
    lines.append("")

    lines.append("MISSION PARAMETERS")
    lines.append("-" * 60)
    lines.append(f"Epoch (UTC)          : {params.epoch.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Orbit altitude       : {params.altitude_km:.1f} km")
    lines.append(f"Inclination          : {params.inclination_deg:.4f} deg")
    lines.append(f"Eccentricity         : {params.eccentricity:.4f}")
    lines.append(f"Orbit type           : {params.orbit_type}")
    lines.append(f"Launch site          : {params.launch_site}")
    lines.append(f"Launch provider      : {params.launch_provider}")
    lines.append(f"Ground station       : {params.ground_station}")
    lines.append("")

    lines.append("SOLAR REFERENCE")
    lines.append("-" * 60)
    lines.append(f"Julian Date          : {solar.julian_date:.6f}")
    lines.append(f"Julian Centuries     : {solar.julian_centuries:.6f}")
    lines.append(f"Solar RA (mean)      : {solar.solar_ra_deg:.6f} deg")
    lines.append("")

    lines.append(format_results_table(results))
    lines.append("")

    lines.append("SCIENTIFIC ASSUMPTIONS AND LIMITATIONS")
    lines.append("-" * 60)
    lines.append("1. This utility performs an INSTANTANEOUS RAAN-to-LTAN conversion")
    lines.append("   at the specified epoch. It does NOT propagate the orbit.")
    lines.append("2. It does NOT calculate J2 nodal precession or future LTAN drift.")
    lines.append("3. It does NOT calculate launch windows.")
    lines.append("4. It does NOT calculate launch azimuth or launch vehicle injection")
    lines.append("   parameters.")
    lines.append("5. It does NOT calculate ground-station visibility or communication")
    lines.append("   windows.")
    lines.append("6. The BITS Goa ground station does NOT affect the RAAN-to-LTAN")
    lines.append("   conversion.")
    lines.append("7. The launch site and launch provider do NOT affect the conversion")
    lines.append("   once the orbit and epoch are specified.")
    lines.append("8. Orbit altitude and inclination are recorded for mission")
    lines.append("   traceability but are NOT direct inputs to this instantaneous")
    lines.append("   conversion.")
    lines.append("9. The assumed epoch is the reference epoch of the supplied RAAN")
    lines.append("   values. It is NOT necessarily the actual launch or orbital")
    lines.append("   insertion epoch.")
    lines.append("10. If the actual orbital insertion time is unknown, the default")
    lines.append("    epoch must be treated as PROVISIONAL.")
    lines.append("11. The current 97.6° inclination is retained as an ANTARA working")
    lines.append("    assumption. The script does NOT silently modify it or claim")
    lines.append("    that it guarantees a perfectly Sun-synchronous orbit.")
    lines.append("12. The result is dependent on the adopted solar-coordinate")
    lines.append("    convention. Mean LTAN and apparent solar time are NOT")
    lines.append("    interchangeable.")
    lines.append("13. The astronomical model (mean-Sun, low-precision) should be")
    lines.append("    validated against a trusted ephemeris (e.g., JPL DE430, SPICE)")
    lines.append("    before using results for final PDR/CDR mission decisions.")
    lines.append("14. CONVERSION EQUATION: LTAN = [12 + (Ω - α☉)/15] mod 24")
    lines.append("    where Ω = RAAN (deg), α☉ = Solar RA (deg), LTAN in hours.")
    lines.append("15. SOLAR MODEL: Mean-Sun model per Meeus 'Astronomical Algorithms'")
    lines.append("    (Ch. 25). Uses mean longitude, mean anomaly, equation of")
    lines.append("    center, and mean obliquity. Expected accuracy ~0.01 deg.")
    lines.append("")

    return "\n".join(lines)


def write_results_file(output_str: str, params: MissionParameters) -> Path:
    """
    Write results to a text file.

    Args:
        output_str: Formatted output string.
        params: Mission parameters (used for filename context).

    Returns:
        Path to the written file.
    """
    base_filename = "raan_ltan_results.txt"
    filepath = Path.cwd() / base_filename

    # Check if file exists
    if filepath.exists():
        response = get_user_input(
            f"File '{base_filename}' already exists. Overwrite? (y/N)", "n"
        ).lower()
        if response != 'y':
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"raan_ltan_results_{timestamp}.txt"
            filepath = Path.cwd() / base_filename
            print(f"  Using filename: {base_filename}")

    filepath.write_text(output_str, encoding='utf-8')
    return filepath


def run_conversion(params: MissionParameters, raan_values: list[float]) -> tuple[str, Path]:
    """
    Run the complete conversion pipeline.

    Args:
        params: Mission parameters.
        raan_values: List of RAAN values.

    Returns:
        Tuple of (formatted_output_string, output_file_path).
    """
    # Compute solar reference
    solar = compute_solar_reference(params.epoch)

    # Convert RAAN values
    results = convert_raan_list(raan_values, solar.solar_ra_deg)

    # Format output
    output = format_full_output(params, solar, results)

    # Write to file
    filepath = write_results_file(output, params)

    return output, filepath


def main() -> int:
    """
    Main entry point for the interactive RAAN-to-LTAN converter.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        # Get user inputs
        params = get_mission_parameters()
        raan_values = get_raan_values()

        # Run conversion
        output, filepath = run_conversion(params, raan_values)

        # Display results
        print("\n" + output)
        print(f"\nResults saved to: {filepath.absolute()}")

        return 0

    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())