# Master interval-weighted reporting repair

Date: 2026-09-28. This changes Master telemetry/summary accounting, not simulation
physics, pointing requirements, orbital propagation or Science checkpoints.

## Cause

Master already clips integration at the horizon and shortens intervals at wheel
events. Its summary nevertheless treated rows as equal-duration samples and
computed submode seconds as row count times the configured step. Shortened
intervals were overrepresented in averages and submode totals could exceed the
time actually spent in that mode.

## Repair

- Master logger requires actual `interval_duration_sec` and records
  `state_time_sec = time_sec + interval_duration_sec`.
- Power, orbital-period, pointing and wheel-momentum means, ADCS/eclipse/power
  fractions, and SUN_POINTING submode durations use actual elapsed-time weights.
- Extrema and event-average slew angles are unchanged.
- Summaries reject missing interval metadata or durations inconsistent with the
  configured horizon.

Master keeps one row per integrated interval, without Science's extra terminal
observation. Its final end-state timestamp, not its final interval-start
timestamp, reaches the horizon. Logs remain in memory; this is not a bounded
annual-memory repair. Master calls `_validate_attitude_metrics` after refreshing
alignment telemetry. The earlier description of this gate as commented out was
incorrect; corrected against the source on September 28.

## Verification

The existing real 11-second wheel-event regression now checks telemetry durations
against every EPS interval. A controlled 2-second/8-second summary fixture verifies
8 W average solar power (rather than the unweighted 5 W), 20% eclipse time
(rather than 50%), 80% ADCS duty, duration-weighted pointing/orbital-period means,
and 20%/80% SUN_POINTING submodes. Missing interval metadata is rejected.

A separate real five-second run with a two-second requested step verifies
intervals `[2, 2, 1]`, start timestamps `[0, 2, 4]` and end-state timestamps
`[2, 4, 5]`. Plots are disabled in these tests and outputs use temporary paths.
These tests do not establish long-run Master fidelity or live payload calibration.

The combined regression suite passed **329 tests**, with 3,809 warnings, in
96.13 seconds using Conda `orekit_env`, `SPACEPY=/tmp`,
`MPLCONFIGDIR=/tmp/antara-mpl` and `PYTHONPATH=src`:

```bash
python -m pytest tests/test_telemetry_integrity.py tests/test_wheel_events.py tests/test_slew_profile.py tests/test_audit_repairs.py tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py tests/test_physics_attitude.py tests/test_scientific_integrity.py tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py -q --disable-warnings --tb=short -p no:cacheprovider
```

Compilation and `git diff --check` passed. Existing CI already collects the
modified wheel-event test module. No annual simulation, commit or push occurred.
