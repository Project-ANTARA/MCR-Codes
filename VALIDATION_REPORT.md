# ANTARA Subsystem Validation Report

> Captured subsystem-validation output; preserved during the 2026-09-12 documentation review without rerunning or changing its PASS/FAIL evidence. The [runtime report](docs/SIMULATION_RUNTIME_FIX_REPORT.md) covers separate targeted regressions and a 60-hour Science run. Those results do not override the subsystem failures recorded here. Use the [README](README.md#subsystem-validation-stage-4) for regeneration instructions.

Generated: 2026-09-12T02:45:34.025420
Repository: ANTARA CubeSat Mission Simulation

## Summary

| Subsystem | Status | Reference | Tolerance |
|-----------|--------|-----------|-----------|
| Orbit Propagation | ❌ FAIL | Skyfield/SGP4 vs Orekit | period ±1s, RAAN precession ±5% |
| IGRF Magnetic Field | ❌ FAIL | NOAA IGRF-13 calculator | field magnitude ±5% |
| ADCS Detumble Time | ✅ PASS | B-dot τ ≈ I/(k·B²) | order-of-magnitude |
| EPS Power Budget | ✅ PASS | BOL cell × packing factor | ±2% |
| SAA/Flux Model | ✅ PASS | AP9 quiet-time at L~1.5 | factor of 2 per channel |
| Comms Link Budget | ✅ PASS | Friis link budget | heuristic within 30% at 45° |

**Overall: ❌ SOME FAILED**

## Detailed Results

### Orbit Propagation

**Script:** `test_orbit_propagation.py`
**Status:** FAIL
**Reference:** Skyfield/SGP4 vs Orekit
**Tolerance:** N/A

#### Stdout
```
======================================================================
ORBIT PROPAGATION VALIDATION TEST
======================================================================

1. Closed-form SSO Precession Rate
----------------------------------------
   dOmega/dt (closed-form): -0.962038 deg/day

2. Orekit Numerical Propagation (24h)
----------------------------------------

```

#### Stderr
```
Traceback (most recent call last):
  File "/home/sammo/Desktop/Projects/ANTARA_CODES-main/tests/validation/test_orbit_propagation.py", line 284, in <module>
    success = main()
  File "/home/sammo/Desktop/Projects/ANTARA_CODES-main/tests/validation/test_orbit_propagation.py", line 213, in main
    ore_times, ore_raan, ore_period = run_orekit_propagation()
  File "/home/sammo/Desktop/Projects/ANTARA_CODES-main/tests/validation/test_orbit_propagation.py", line 150, in run_orekit_propagation
    propagator, start_date, inertial_frame, earth_frame, earth, sun = build_orekit_propagator()
  File "/home/sammo/Desktop/Projects/ANTARA_CODES-main/tests/validation/test_orbit_propagation.py", line 77, in build_orekit_propagator
    utc = TimeScalesFactory.getUTC()
orekit.JavaError: <super: <class 'JavaError'>, <JavaError object>>
    Java stacktrace:
org.orekit.errors.OrekitException: no IERS UTC-TAI history data loaded
	at org.orekit.time.LazyLoadedTimeScales.getUTC(LazyLoadedTimeScales.java:193...
```

### IGRF Magnetic Field

**Script:** `test_igrf_field.py`
**Status:** FAIL
**Reference:** NOAA IGRF-13 calculator
**Tolerance:** N/A

#### Stdout
```
======================================================================
IGRF MAGNETIC FIELD VALIDATION TEST
======================================================================
Reference: NOAA/NCEI IGRF-13 online calculator
Test points: 17 locations spanning SAA and globe
Epoch: 2000.25 (matching sim start 2000-03-01)
   1. SAA core                       | Ref:  22500 nT | Model:  18279 nT | Error: 18.76% | FAIL
   2. SAA core                       | Ref:  24800 nT | Model:  18678 nT | Error: 24.69% | FAIL
   3. SAA core                       | Ref:  26200 nT | Model:  19316 nT | Error: 26.27% | FAIL
   4. SAA edge                       | Ref:  21800 nT | Model:  18240 nT | Error: 16.33% | FAIL
   5. SAA edge                       | Ref:  28500 nT | Model:  19625 nT | Error: 31.14% | FAIL
   6. SAA boundary                   | Ref:  31200 nT | Model:  20784 nT | Error: 33.38% | FAIL
   7. SAA boundary                   | Ref:  33500 nT | Model:  22012 nT | Error: 34.29% | FAIL
   8. Equator, Atlantic              | Ref:  34200 nT | Model:  22823 nT | Error: 33.27% | FAIL
   9. Equator, Indian Ocean          | Ref:  35800 nT | Model:  27078 nT | Error: 24.36% | FAIL
  10. India (GS)                     | Ref:  39500 nT | Model:  35249 nT | Error: 10.76% | FAIL
  11. Mid-latitude                   | Ref:  48200 nT | Model:  40833 nT | Error: 15.29% | FAIL
  12. Svalbard (GS2)                 | Ref:  54800 nT | Model:  42761 nT | Error: 21.97% | FAIL
  13. Antarctic                      | Ref:  54800 nT | Model:  35463 nT | Error: 35.29% | FAIL
  14. SAA core, 400km                | Ref:  28500 nT | Model:  19691 nT | Error: 30.91% | FAIL
  15. SAA core, 800km                | Ref:  18200 nT | Model:  16995 nT | Error:  6.62% | FAIL
  16. SAA core, 2020                 | Ref:  22100 nT | Model:  17942 nT | Error: 18.81% | FAIL
  17. SAA core, 2025                 | Ref:  21700 nT | Model:  17869 nT | Error: 17.66% | FAIL

==============================================...
```

### ADCS Detumble Time

**Script:** `test_adcs_detumble_time.py`
**Status:** PASS
**Reference:** B-dot τ ≈ I/(k·B²)
**Tolerance:** N/A

#### Stdout
```
======================================================================
ADCS DETUMBLE TIME VALIDATION TEST
======================================================================

1. Analytical B-dot Time Constant (Classical Formula)
----------------------------------------
   Classical B-dot time constant: 762.2 s (12.70 min)
   Initial rate: 9.30 deg/s
   Handover threshold: 2.0 deg/s
   Ratio: 4.7x
   Analytical time_to_handover: 1171.5 s (19.53 min)

2. Monte Carlo Baseline Trial (Stage 2)
----------------------------------------
   Running single trial with default tip-off rates...
   MC time_to_handover: 995.0 s (16.58 min)
   Success: True
   Time to complete: 1009.0 s

3. Comparison
----------------------------------------
   MC / Analytical = 0.85x
   Difference: 176.5 s
   Order-of-magnitude: PASS (ratio=0.85)

   Note: Tighter tolerance not expected because:
   - Analytical model assumes constant B-field magnitude
   - Real B-field varies with orbit position (dipole tilt)
   - B-dot uses finite-difference dB/dt (noisy)
   - Wheel-damp phase adds extra time after BDOT handover
   - Discrete timestep integration effects

======================================================================
VALIDATION SUMMARY
======================================================================
  Order-of-magnitude: PASS
  Note: Both ω_orb and non-ω_orb formulas fall within
  the factor-of-10 tolerance band; test validates framework,
  not precise formula correctness.

```

### EPS Power Budget

**Script:** `test_eps_power_budget.py`
**Status:** PASS
**Reference:** BOL cell × packing factor
**Tolerance:** N/A

#### Stdout
```
======================================================================
EPS POWER BUDGET VALIDATION TEST
======================================================================

Published component efficiencies:
  Cell BOL efficiency:       28.5%
  Packing factor:            67.0%
  Harness derating (MPPT):   88.0%
  Thermal derating (60°C):   88.0%

  Recomputed: 0.147872
  Config value: 0.193515

  Error: 23.59%
  Tolerance: 2% -> FAIL

  Component breakdown:
    Cell * Packing = 0.1910
    * Harness = 0.1680
    * Thermal = 0.147872

  Note: Config value 0.193515 = 28.5% * 0.67 = BOL * packing
  Harness (0.88) and thermal (0.88) applied separately in compute_solar_power()
  BOL * Packing = 0.190950
  Config value  = 0.193515
  Error = 1.33%
  Tolerance: 2% -> PASS

======================================================================
VALIDATION SUMMARY
======================================================================
  Overall: PASS (tolerance: 2%)

```

### SAA/Flux Model

**Script:** `test_saa_flux_model.py`
**Status:** PASS
**Reference:** AP9 quiet-time at L~1.5
**Tolerance:** N/A

#### Stdout
```
======================================================================
SAA / FLUX MODEL VALIDATION TEST
======================================================================
Reference: AP9/AE9 v1.5 quiet-time, L=1.5, 600km SSO
Location: 600 km SSO, L~1.5 (sub-SAA peak)

Energy channels: [  5.  10.  20.  50. 100.] MeV
Config flux:      [35000. 18000.  6000.   900.   110.]
Literature nominal: [35000. 18000.  6000.   900.   110.]
Literature range:  [2.0e+04 - 7.0e+04] etc.

  Channel 1: 5.0 MeV
    Config:        3.50e+04 p/cm²/s/sr/MeV
    Lit nominal:   3.50e+04
    Lit range:     [2.00e+04 - 7.00e+04]
    Ratio:         1.00x
    In lit range:  True
    Within factor 2: True
    PASS

  Channel 2: 10.0 MeV
    Config:        1.80e+04 p/cm²/s/sr/MeV
    Lit nominal:   1.80e+04
    Lit range:     [1.00e+04 - 3.50e+04]
    Ratio:         1.00x
    In lit range:  True
    Within factor 2: True
    PASS

  Channel 3: 20.0 MeV
    Config:        6.00e+03 p/cm²/s/sr/MeV
    Lit nominal:   6.00e+03
    Lit range:     [3.00e+03 - 1.20e+04]
    Ratio:         1.00x
    In lit range:  True
    Within factor 2: True
    PASS

  Channel 4: 50.0 MeV
    Config:        9.00e+02 p/cm²/s/sr/MeV
    Lit nominal:   9.00e+02
    Lit range:     [4.00e+02 - 2.00e+03]
    Ratio:         1.00x
    In lit range:  True
    Within factor 2: True
    PASS

  Channel 5: 100.0 MeV
    Config:        1.10e+02 p/cm²/s/sr/MeV
    Lit nominal:   1.10e+02
    Lit range:     [5.00e+01 - 3.00e+02]
    Ratio:         1.00x
    In lit range:  True
    Within factor 2: True
    PASS

======================================================================
VALIDATION SUMMARY
======================================================================
  Channels passed: 5/5
  Overall: PASS (tolerance: factor of 2)

  Note: AP9 quiet-time at L~1.5 carries large intrinsic uncertainty (~factor 2-3)
  due to: limited historical measurements, model parameterization choices,
  solar cycle variation not fully captured, a...
```

### Comms Link Budget

**Script:** `test_comms_link_budget.py`
**Status:** PASS
**Reference:** Friis link budget
**Tolerance:** N/A

#### Stdout
```
======================================================================
COMMS LINK BUDGET VALIDATION TEST
======================================================================

  Tx power: 12.0 W
  Configured rate: 19.2 kbps
  Frequency: 437 MHz (UHF)
  SC antenna: 3.0 dBi (patch/monopole)
  GS antenna: 14.0 dBi (3m dish at UHF)
  Bandwidth: 25 kHz (for 19.2 kbps BPSK)
  Required Eb/N0: 9.6 dB (BPSK, BER=1e-5)

  Elevation: 15°
  Friis max rate:    1891.5 kbps (margin: 19.9 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    9.8 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 20°
  Friis max rate:    2693.9 kbps (margin: 21.5 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    11.2 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 30°
  Friis max rate:    4713.5 kbps (margin: 23.9 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    13.6 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 45°
  Friis max rate:    8430.4 kbps (margin: 26.4 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    16.1 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 60°
  Friis max rate:    12136.9 kbps (margin: 28.0 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    17.9 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 75°
  Friis max rate:    14852.1 kbps (margin: 28.9 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    18.9 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Elevation: 90°
  Friis max rate:    15846.5 kbps (margin: 29.2 dB)
  Configured rate:   19.2 kbps
  Heuristic rate:    19.2 kbps
  Heuristic <= Friis: True
  Margin >= 3dB:     True
  PASS

  Friis monotonic: True
  Heuristic monotonic: True

======================================================================
VALIDATION SUMMARY
======================================================================
  Link margin >=3dB at all elev ...
```

## Notes

- Tolerances are subsystem-specific and documented in each test.
- AP9 flux models carry ~factor 2-3 intrinsic uncertainty (limited measurements).
- B-dot detumble analytical model is order-of-magnitude due to simplified B-field.
- Comms heuristic (sqrt(sin(el))) is a PDR-level approximation; full link budget needed for CDR.
- Period comparison uses theoretical circular orbit at 600km; real orbit has J2 perturbations.

---

*Report generated by `scripts/generate_validation_report.py`*
