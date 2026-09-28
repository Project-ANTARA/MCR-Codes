# Wheel-Capacity Event Repair

Date: 2026-09-27. Local short-run verification only; no annual simulation,
hosted CI dispatch, commit or push was performed.

## Problem and repair

Acceleration followed by braking can have zero net torque impulse while wheel
momentum exceeds capacity between endpoints. The regression example has a
0.005 N m s unloaded-free peak against a 0.003 N m s capacity. The previous
defensive guard rejected such intervals but required manual timestep refinement.

Science and Master now preview the scalar slew profile on a disposable ADCS
copy, calculate the first capacity crossing, and advance real guidance once to
that boundary. `src/wheel_dynamics.py` integrates constant drive and opposing
unloading per phase, splitting at momentum zero crossings. A one-floating-tick
extension avoids missing the capacity comparison through roundoff; it does not
relax physical limits. The shortened duration is shared by EPS, payload, wheel
integration, statistics and the simulation clock. No interval replay is used.

The existing saturation freeze, pending-objective reacquisition and recovery
thresholds remain. The hidden-peak guard remains a defensive check. Wheel
updates reject mismatched phase duration/mean impulse before committing state.

## Compatibility and CI

- Science checkpoints are **version 6**; version 5 and earlier are rejected.
- `wheel_event_split` marks event-shortened telemetry intervals. Their endpoint
  is `state_time_sec`; the terminal observation has zero duration and no split.
- Fixed-step requests specify a maximum interval. Scenario CI checks exact
  3600-second duration, positive bounded gaps, interval/gap agreement, zero final
  duration and annotations for shortened nonterminal intervals.
- Unit CI includes `tests/test_wheel_events.py`. Annual scheduling, runner and
  duration are unchanged.

## Verification

Using `/home/sammo/miniconda3/envs/orekit_env/bin/python`, with
`SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src`:

```bash
python -m pytest tests/test_wheel_events.py tests/test_slew_profile.py tests/test_audit_repairs.py tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py tests/test_physics_attitude.py tests/test_scientific_integrity.py tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py -q --disable-warnings --tb=short -p no:cacheprovider
```

Result: **226 passed**, 3,809 warnings, 83.64 seconds. Warnings were suppressed
in this summary run, not eliminated. Controlled 11-second Science and Master
tests exercise actual capacity events, interval conservation and saturation;
Science checkpoint/resume at the boundary matches uninterrupted telemetry.
Additional cases cover oblique axes, zero crossings and atomic rejection.

Both real Science one-hour runs used CLI-parsed `--duration-hours 1 --step-sec 2
--no-plots`, isolated outputs and the production constructor/run loop:

| Scenario | Duration | Rows | Maximum gap | Result |
| --- | ---: | ---: | ---: | --- |
| leop_24h | 3600 s | 1801 | 2 s | PASS |
| no_leop_24h | 3600 s | 1801 | 2 s | PASS |

Artifacts: `/tmp/antara-wheel-event-smoke-vg6dvrjm/` (temporary).
Neither smoke exercised DETECTION/RELAY or a capacity split. They ran before
the final split-annotation and impulse-consistency checks were added; the full
226-test run includes those final changes. These are not annual acceptance or
payload-calibration results.

## Remaining limits

The slew remains a scalar kinematic profile with an acceleration-to-torque
proxy, not a coupled inertia/actuator controller. Transverse momentum dynamics,
within-interval moving targets and physical braking during saturation freeze
are not resolved. Recovery release remains endpoint-based. DETUMBLE does not
use this adaptive wheel-event solver. Absolute payload calibration, live AP9,
unbounded event archives and long-run CI acceptance remain separate work.

Historical evidence in `SLEW_PROFILE_REPAIR.md` is retained with its original
scope; its guard-only/version-5 status is superseded by this report.
