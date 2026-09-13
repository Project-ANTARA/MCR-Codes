#!/usr/bin/env python3
"""
IGRF Magnetic Field validation test.

Compares IGRFModel.get_ned_field_tesla() against independent IRBEM IGRF
evaluation. No external magnetospheric field is included.

Tolerance: field magnitude within 5%
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
from datetime import datetime
import numpy as np
from spacepy import coordinates, irbempy, time

# IGRF model from ANTARA_Science_Sim
from ANTARA_Science_Sim import IGRFModel, MissionConfig


# Retain the original locations and old magnitudes for audit only. The old
# rounded magnitudes had no reproducible calculator output and disagreed with
# both implementations. They are NOT used as acceptance references anymore.
IGRF_REFERENCE_POINTS = [
    # (lat, lon, alt_km, year, B_mag_nT, description)
    # SAA core region
    (-25.0, -50.0, 600, 2000.25, 22500, "SAA core"),
    (-30.0, -45.0, 600, 2000.25, 24800, "SAA core"),
    (-35.0, -40.0, 600, 2000.25, 26200, "SAA core"),
    (-20.0, -55.0, 600, 2000.25, 21800, "SAA edge"),
    # SAA edge / boundary
    (-10.0, -30.0, 600, 2000.25, 28500, "SAA edge"),
    (-5.0, -20.0, 600, 2000.25, 31200, "SAA boundary"),
    (0.0, -10.0, 600, 2000.25, 33500, "SAA boundary"),
    # Outside SAA - equatorial
    (0.0, 0.0, 600, 2000.25, 34200, "Equator, Atlantic"),
    (0.0, 60.0, 600, 2000.25, 35800, "Equator, Indian Ocean"),
    # Mid-latitude
    (30.0, 74.0, 600, 2000.25, 39500, "India (GS)"),
    (45.0, 74.0, 600, 2000.25, 48200, "Mid-latitude"),
    # High-latitude / polar
    (78.0, 15.0, 600, 2000.25, 54800, "Svalbard (GS2)"),
    (-78.0, -15.0, 600, 2000.25, 54800, "Antarctic"),
    # Different altitude (same lat/lon as SAA core)
    (-25.0, -50.0, 400, 2000.25, 28500, "SAA core, 400km"),
    (-25.0, -50.0, 800, 2000.25, 18200, "SAA core, 800km"),
    # Different epoch
    (-25.0, -50.0, 600, 2020.0, 22100, "SAA core, 2020"),
    (-25.0, -50.0, 600, 2025.0, 21700, "SAA core, 2025"),
]


def reference_field_ned(lat, lon, alt, year):
    """IRBEM GEO nT -> geodetic NED nT, independent of ppigrf.

    SpacePy 0.7 bundles IGRF-13, valid through 2025.0. Its endpoint is used
    explicitly for 2025 comparisons; this is not an IGRF-14 epoch validation.
    ANTARA currently maps decimal epochs to March 1 of their integer year.
    """
    date = datetime(int(year), 3, 1) if year < 2025 else datetime(2025, 1, 1)
    ticks = time.Ticktock([date], "UTC")
    loci = coordinates.Coords([[alt, lat, lon]], "GDZ", "sph", use_irbem=True)
    result = irbempy.get_Bfield(ticks, loci, extMag="0", options=[0, 1, 0, 0, 0])
    x, y, z = np.asarray(result["Bvec"])[0]
    phi, lam = np.radians([lat, lon])
    north = -np.sin(phi) * (x*np.cos(lam) + y*np.sin(lam)) + z*np.cos(phi)
    east = -x*np.sin(lam) + y*np.cos(lam)
    down = -np.cos(phi) * (x*np.cos(lam) + y*np.sin(lam)) - z*np.sin(phi)
    return np.array([north, east, down])


def main():
    print("=" * 70)
    print("IGRF MAGNETIC FIELD VALIDATION TEST")
    print("=" * 70)
    print("Reference: independent IRBEM IGRF-13, no external field")
    print(f"Test points: {len(IGRF_REFERENCE_POINTS)} locations spanning SAA and globe")
    print("Epochs: ANTARA March 1 mapping; IRBEM 2025 endpoint uses January 1")

    config = MissionConfig()
    igrf = IGRFModel(config.igrf_epoch_year)
    if not igrf._available:
        raise RuntimeError("ppigrf is required; dipole fallback is not IGRF validation")

    results = []
    all_pass = True

    for i, (lat, lon, alt, year, legacy_mag_nT, desc) in enumerate(IGRF_REFERENCE_POINTS):
        # Get field from IGRF model
        if year != 2000.25:
            # Need to create IGRF with different epoch
            test_igrf = IGRFModel(year)
            b_ned = test_igrf.get_ned_field_tesla(lat, lon, alt)
        else:
            b_ned = igrf.get_ned_field_tesla(lat, lon, alt)

        b_mag_t = float(np.linalg.norm(b_ned))
        b_mag_nT = b_mag_t * 1e9
        reference_ned = reference_field_ned(lat, lon, alt, year)
        ref_mag_nT = float(np.linalg.norm(reference_ned))

        error_pct = abs(b_mag_nT - ref_mag_nT) / ref_mag_nT * 100
        vector_error_pct = float(np.linalg.norm(b_ned * 1e9 - reference_ned) / ref_mag_nT * 100)
        passed = bool(np.isfinite(vector_error_pct) and error_pct <= 5.0 and vector_error_pct <= 5.0)

        status = "PASS" if passed else "FAIL"
        print(f"  {i+1:2d}. {desc:30s} | Ref: {ref_mag_nT:6.0f} nT | Model: {b_mag_nT:6.0f} nT | Error: {error_pct:5.2f}% | {status}")

        results.append({
            "point": i + 1,
            "description": desc,
            "lat": lat,
            "lon": lon,
            "alt_km": alt,
            "year": year,
            "ref_mag_nT": ref_mag_nT,
            "legacy_unverified_mag_nT": legacy_mag_nT,
            "reference_ned_nT": reference_ned.tolist(),
            "model_ned_nT": (b_ned * 1e9).tolist(),
            "vector_error_pct": vector_error_pct,
            "model_mag_nT": b_mag_nT,
            "error_pct": error_pct,
            "passed": passed,
        })

        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"  Passed: {passed_count}/{len(results)}")
    print(f"  Overall: {'PASS' if all_pass else 'FAIL'} (tolerance: 5%)")

    # Save results
    with open(Path(os.environ.get("ANTARA_VALIDATION_OUTPUT_DIR", Path(__file__).parent)) / "validation_results_igrf.json", "w") as f:
        json.dump({
            "overall_pass": all_pass,
            "tolerance_pct": 5.0,
            "results": results,
        }, f, indent=2)

    return all_pass


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np

    success = main()
    sys.exit(0 if success else 1)
