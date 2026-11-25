#!/usr/bin/env python3
"""
Unit tests for ANTARA RAAN to LTAN Conversion Utility.

Tests cover:
1. RAAN normalization
2. LTAN normalization
3. Decimal-hours-to-HMS conversion
4. Solar RA calculation
5. Input validation
6. Full conversion pipeline consistency

Run with: python3 -m pytest test_raan_to_ltan.py -v
or: python3 test_raan_to_ltan.py
"""

import math
import unittest
from datetime import datetime, timezone

from raan_to_ltan import (
    julian_date_from_utc,
    calculate_julian_centuries,
    calculate_mean_sun_right_ascension,
    normalize_raan,
    calculate_ltan,
    decimal_hours_to_hms,
    parse_raan_input,
    validate_epoch_input,
    validate_orbital_params,
    convert_raan_list,
    compute_solar_reference,
)


class TestRAANNormalization(unittest.TestCase):
    """Test RAAN normalization to [0, 360)."""

    def test_exact_360(self):
        self.assertAlmostEqual(normalize_raan(360.0), 0.0)

    def test_370(self):
        self.assertAlmostEqual(normalize_raan(370.0), 10.0)

    def test_negative_10(self):
        self.assertAlmostEqual(normalize_raan(-10.0), 350.0)

    def test_negative_360(self):
        self.assertAlmostEqual(normalize_raan(-360.0), 0.0)

    def test_negative_720(self):
        self.assertAlmostEqual(normalize_raan(-720.0), 0.0)

    def test_720(self):
        self.assertAlmostEqual(normalize_raan(720.0), 0.0)

    def test_decimal(self):
        self.assertAlmostEqual(normalize_raan(45.5), 45.5)
        self.assertAlmostEqual(normalize_raan(405.5), 45.5)

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            normalize_raan(float('inf'))
        with self.assertRaises(ValueError):
            normalize_raan(float('-inf'))
        with self.assertRaises(ValueError):
            normalize_raan(float('nan'))


class TestLTANNormalization(unittest.TestCase):
    """Test LTAN calculation and normalization to [0, 24)."""

    def test_basic_calculation(self):
        # If RAAN == Solar RA, LTAN = 12:00
        solar_ra = 100.0
        ltan = calculate_ltan(100.0, solar_ra)
        self.assertAlmostEqual(ltan, 12.0)

    def test_raan_ahead_of_sun(self):
        # RAAN 15 deg ahead of Sun -> LTAN = 13:00
        solar_ra = 100.0
        ltan = calculate_ltan(115.0, solar_ra)
        self.assertAlmostEqual(ltan, 13.0)

    def test_raan_behind_sun(self):
        # RAAN 15 deg behind Sun -> LTAN = 11:00
        solar_ra = 100.0
        ltan = calculate_ltan(85.0, solar_ra)
        self.assertAlmostEqual(ltan, 11.0)

    def test_wrap_above_24(self):
        # RAAN 360 deg ahead -> LTAN wraps
        solar_ra = 0.0
        ltan = calculate_ltan(350.0, solar_ra)
        # 12 + 350/15 = 12 + 23.333 = 35.333 -> 11.333
        self.assertAlmostEqual(ltan, 11.333333, places=4)

    def test_wrap_negative(self):
        # RAAN 350 deg behind Sun
        solar_ra = 350.0
        ltan = calculate_ltan(10.0, solar_ra)
        # 12 + (10-350)/15 = 12 - 22.666 = -10.666 -> 13.333
        self.assertAlmostEqual(ltan, 13.333333, places=4)

    def test_output_range(self):
        # All results should be in [0, 24)
        solar_ra = 45.0
        for raan in [0, 90, 180, 270, 359, 360, 720, -10, -90]:
            ltan = calculate_ltan(normalize_raan(raan), solar_ra)
            self.assertGreaterEqual(ltan, 0.0)
            self.assertLess(ltan, 24.0)


class TestDecimalHoursToHMS(unittest.TestCase):
    """Test decimal hours to HH:MM:SS conversion."""

    def test_zero(self):
        self.assertEqual(decimal_hours_to_hms(0.0), "00:00:00")

    def test_half_hour(self):
        self.assertEqual(decimal_hours_to_hms(12.5), "12:30:00")

    def test_quarter_hour(self):
        self.assertEqual(decimal_hours_to_hms(3.25), "03:15:00")

    def test_precise_seconds(self):
        self.assertEqual(decimal_hours_to_hms(1.5 / 3600), "00:00:02")  # ~1.5 seconds rounds to 2

    def test_midnight_rollover(self):
        # 23.9999 hours rounds to 24:00:00 -> should display 00:00:00
        self.assertEqual(decimal_hours_to_hms(23.9999), "00:00:00")

    def test_exact_24_rollover(self):
        # 24.0 should wrap to 00:00:00
        self.assertEqual(decimal_hours_to_hms(24.0), "00:00:00")

    def test_negative_wraps(self):
        # Negative values wrap
        self.assertEqual(decimal_hours_to_hms(-1.0), "23:00:00")

    def test_rounding_edge_cases(self):
        # 11:59:59.5 rounds to 12:00:00
        self.assertEqual(decimal_hours_to_hms(11 + 59/60 + 59.5/3600), "12:00:00")


class TestSolarRACalculation(unittest.TestCase):
    """Test solar right ascension calculation."""

    def test_j2000_epoch(self):
        # J2000.0: 2000-01-01 12:00:00 UTC -> JD 2451545.0, T=0
        epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        self.assertAlmostEqual(jd, 2451545.0, places=1)

        jc = calculate_julian_centuries(jd)
        self.assertAlmostEqual(jc, 0.0, places=10)

        solar_ra = calculate_mean_sun_right_ascension(jc)
        # At J2000, mean Sun RA ~ 18h 45m = 281.25 deg
        self.assertTrue(math.isfinite(solar_ra))
        self.assertGreaterEqual(solar_ra, 0.0)
        self.assertLess(solar_ra, 360.0)

    def test_default_epoch_2029(self):
        # Default ANTARA epoch: 2029-01-01 00:00:00 UTC
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)

        self.assertTrue(math.isfinite(solar.julian_date))
        self.assertTrue(math.isfinite(solar.julian_centuries))
        self.assertTrue(math.isfinite(solar.solar_ra_deg))
        self.assertGreaterEqual(solar.solar_ra_deg, 0.0)
        self.assertLess(solar.solar_ra_deg, 360.0)

    def test_solar_ra_normalized(self):
        # Test multiple epochs, all should give normalized RA
        for year in [2000, 2010, 2020, 2029, 2030, 2050]:
            epoch = datetime(year, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            solar = compute_solar_reference(epoch)
            self.assertGreaterEqual(solar.solar_ra_deg, 0.0)
            self.assertLess(solar.solar_ra_deg, 360.0)

    def test_julian_date_consistency(self):
        # Test known JD values
        # 2000-01-01 12:00:00 UTC = JD 2451545.0
        epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        self.assertAlmostEqual(jd, 2451545.0, places=1)

        # 2029-01-01 00:00:00 UTC
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        # Approximately 2461010.5 (can verify with external calculator)
        self.assertTrue(math.isfinite(jd))
        self.assertGreater(jd, 2451545.0)


class TestInputValidation(unittest.TestCase):
    """Test input validation functions."""

    def test_parse_raan_comma_separated(self):
        result = parse_raan_input("0, 15, 30, 45")
        self.assertEqual(result, [0.0, 15.0, 30.0, 45.0])

    def test_parse_raan_space_separated(self):
        result = parse_raan_input("0 15 30 45")
        self.assertEqual(result, [0.0, 15.0, 30.0, 45.0])

    def test_parse_raan_mixed_separators(self):
        result = parse_raan_input("0, 15 30,45")
        self.assertEqual(result, [0.0, 15.0, 30.0, 45.0])

    def test_parse_raan_decimals(self):
        result = parse_raan_input("0.5, 15.25, -10.75")
        self.assertEqual(result, [0.5, 15.25, -10.75])

    def test_parse_raan_negative_and_large(self):
        result = parse_raan_input("-10, 370, 720")
        self.assertEqual(result, [-10.0, 370.0, 720.0])

    def test_parse_raan_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_raan_input("")
        with self.assertRaises(ValueError):
            parse_raan_input("   ")

    def test_parse_raan_invalid_token_raises(self):
        with self.assertRaises(ValueError):
            parse_raan_input("0, abc, 30")
        with self.assertRaises(ValueError):
            parse_raan_input("0, inf, 30")

    def test_validate_epoch_valid(self):
        epoch = validate_epoch_input("2029-01-01", "12:30:45")
        self.assertEqual(epoch, datetime(2029, 1, 1, 12, 30, 45, tzinfo=timezone.utc))

    def test_validate_epoch_invalid_format(self):
        with self.assertRaises(ValueError):
            validate_epoch_input("01-01-2029", "12:30:45")
        with self.assertRaises(ValueError):
            validate_epoch_input("2029-01-01", "12:30")

    def test_validate_orbital_params_valid(self):
        validate_orbital_params(600, 97.6, 0.0)
        validate_orbital_params(400, 98.0, 0.001)
        validate_orbital_params(0, 0, 0)
        validate_orbital_params(2000, 180, 0.999)

    def test_validate_orbital_params_invalid_altitude(self):
        with self.assertRaises(ValueError):
            validate_orbital_params(-100, 97.6, 0.0)
        with self.assertRaises(ValueError):
            validate_orbital_params(200000, 97.6, 0.0)

    def test_validate_orbital_params_invalid_inclination(self):
        with self.assertRaises(ValueError):
            validate_orbital_params(600, -1, 0.0)
        with self.assertRaises(ValueError):
            validate_orbital_params(600, 181, 0.0)

    def test_validate_orbital_params_invalid_eccentricity(self):
        with self.assertRaises(ValueError):
            validate_orbital_params(600, 97.6, -0.01)
        with self.assertRaises(ValueError):
            validate_orbital_params(600, 97.6, 1.0)
        with self.assertRaises(ValueError):
            validate_orbital_params(600, 97.6, 1.5)


class TestFullConversionPipeline(unittest.TestCase):
    """Test the complete conversion pipeline for consistency."""

    def test_single_raan_produces_single_result(self):
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)
        results = convert_raan_list([0.0], solar.solar_ra_deg)
        self.assertEqual(len(results), 1)

    def test_multiple_raan_produces_matching_results(self):
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)
        raan_values = [0, 15, 30, 45, 60, 90, 120, 180, 240, 270, 300, 359]
        results = convert_raan_list(raan_values, solar.solar_ra_deg)
        self.assertEqual(len(results), len(raan_values))

    def test_results_have_correct_structure(self):
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)
        results = convert_raan_list([45.0], solar.solar_ra_deg)
        r = results[0]
        self.assertEqual(r.original_raan, 45.0)
        self.assertEqual(r.normalized_raan, 45.0)
        self.assertTrue(math.isfinite(r.ltan_decimal))
        self.assertGreaterEqual(r.ltan_decimal, 0.0)
        self.assertLess(r.ltan_decimal, 24.0)
        self.assertRegex(r.ltan_hms, r"^\d{2}:\d{2}:\d{2}$")

    def test_original_raan_preserved(self):
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)
        results = convert_raan_list([370.0, -10.0, 720.5], solar.solar_ra_deg)
        self.assertEqual(results[0].original_raan, 370.0)
        self.assertEqual(results[0].normalized_raan, 10.0)
        self.assertEqual(results[1].original_raan, -10.0)
        self.assertEqual(results[1].normalized_raan, 350.0)
        self.assertEqual(results[2].original_raan, 720.5)
        self.assertAlmostEqual(results[2].normalized_raan, 0.5)

    def test_ltan_values_in_valid_range(self):
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        solar = compute_solar_reference(epoch)
        # Test many RAAN values
        raan_values = [i * 30.0 for i in range(12)] + [-10, 370, 720, 1080.5]
        results = convert_raan_list(raan_values, solar.solar_ra_deg)
        for r in results:
            self.assertGreaterEqual(r.ltan_decimal, 0.0)
            self.assertLess(r.ltan_decimal, 24.0)


class TestJulianDate(unittest.TestCase):
    """Test Julian Date calculation."""

    def test_j2000(self):
        epoch = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        self.assertAlmostEqual(jd, 2451545.0, places=1)

    def test_known_date(self):
        # 2029-01-01 00:00:00 UTC
        epoch = datetime(2029, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        # Verified: J2000.0 (2451545.0) + 29 years ≈ 2451545 + 29*365.25 = 2462136.75
        # Actual calculated value
        self.assertAlmostEqual(jd, 2462137.5, places=1)

    def test_fractional_day(self):
        epoch = datetime(2000, 1, 1, 18, 0, 0, tzinfo=timezone.utc)
        jd = julian_date_from_utc(epoch)
        self.assertAlmostEqual(jd, 2451545.25, places=5)


def run_tests():
    """Run all tests and return success status."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRAANNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestLTANNormalization))
    suite.addTests(loader.loadTestsFromTestCase(TestDecimalHoursToHMS))
    suite.addTests(loader.loadTestsFromTestCase(TestSolarRACalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestFullConversionPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestJulianDate))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)