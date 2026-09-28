# Slew integration repair — 2026-09-26

## Findings and implementation

The previous profile used the end-of-step rate times the full interval for
displacement. From rest with 1 degree/s² acceleration and a one-second interval,
it moved 1 degree instead of 0.5 degree. It could also reset a nonzero rate
immediately inside the 0.4-degree pointing band, omitting braking. New command
modes could erase motion, and settlement could discard leftover interval time.

`src/slew_profile.py` now integrates piecewise constant acceleration analytically,
splitting acceleration, coast and braking at their actual transition times.
Infeasible stopping distances cause braking overshoot and return, not angle
clipping. Same-axis target reversals brake before reversing. Shared and Master
ADCS use this profile; quaternion angles use cross-component/scalar `atan2`.
Completed motion and remaining settlement time are accounted separately.

Torque remains an interval-average signed acceleration proxy, with the previously
corrected body-frame direction. A complete rest-to-rest interval has zero net
proxy impulse. Geometric assertions, the 1e-6-degree numerical alignment tolerance,
flight pointing budgets and scalar-first body-to-inertial convention are unchanged.

Invalid finite/positive slew limits are rejected before divisions. Shared mission
configuration rejects non-finite scalar inputs and invalid coarse timesteps.
Checkpoint version **5** preserves the in-flight slew direction/rate/cache and
rejects older integration histories. Do not edit an old checkpoint's version.
CI's explicit unit command includes `tests/test_slew_profile.py`.

## Verification

Local Conda `orekit_env`, Python 3.10.21. The combined ten-file regression suite
passed **170 tests** with 3,809 existing wrapper warnings. Compilation of all
`src/*.py` and changed tests, plus `git diff --check`, passed.

Analytic tests cover 1-, 30- and 90-degree maneuvers at 0.01-, 0.1-, 1-, 2- and
20-second steps. The 30-degree rest-to-rest maneuver takes 11 seconds at the
configured 5-degree/s and 1-degree/s² limits. Additional tests cover overshoot,
reverse motion, preserved rate inside the acceptance band, mode changes, zero
time, settlement remainder, invalid configuration and in-flight checkpoint state.

Both one-hour Science smoke runs reached their terminal summaries. Their saved
telemetry was independently rechecked after the tool sessions detached:

| Scenario | Integrated time | Cadence | Rows | Observed modes |
| --- | ---: | ---: | ---: | --- |
| leop_24h | 3,600 s | 2 s | 1,801 | COLD_AND_DARK, DETUMBLE, SUN_POINTING, SLEWING, SAFE |
| no_leop_24h | 3,600 s | 2 s | 1,801 | SUN_POINTING |

Artifacts: `/tmp/antara-profile-smoke-lpmf03p1/`. Runs used the real constructor
and `run()` with CLI-parsed one-hour/two-second overrides, isolated output paths,
and no plots. No simulation physics was mocked. They preceded the final
settlement-remainder/config-validation edits; the final regression run includes
those edits. These smoke runs did not exercise operational DETECTION or RELAY.

```bash
conda activate orekit_env
export SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src
python -m pytest tests/test_slew_profile.py tests/test_audit_repairs.py \
  tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py \
  tests/test_physics_attitude.py tests/test_scientific_integrity.py \
  tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py \
  tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py \
  -q --disable-warnings --tb=short -p no:cacheprovider
python -m py_compile src/*.py tests/test_slew_profile.py tests/test_checkpoint_roundtrip.py
git diff --check
```

## Still open — not evidence of complete physics validation

- The profile is one-dimensional, with a target fixed during each interval.
  Arbitrary changes of slew axis do not model transverse angular momentum.
- Torque magnitude is not inertia-derived; interval-averaged torque does not
  resolve wheel-momentum peaks inside an interval. A coupled actuator/controller
  model and convergence campaign are needed before hardware-capability claims.
- Existing saturation hold is a prescribed freeze/unload policy, not a physical
  braking transient. Its mission policy was preserved.
- Live AP9, calibrated detector response and a stochastic arrival model require
  explicit backend/data and modeling choices; none were invented.
- Event/pass archives and checkpoint size still grow. Chunked telemetry summaries
  alone do not establish bounded annual memory or validate the annual timeout.

No annual simulation, dependency installation, workflow dispatch, commit or push
was performed. This closes the tested numerical profile defects, not every
possible repository defect or the remaining model-design work.

## Follow-up: hidden wheel-momentum peaks

Historical status: the guard-only behavior below was superseded on September 27 by [automatic wheel-event splitting](WHEEL_EVENT_REPAIR.md), with version-6 checkpoints. The results below retain their original scope.

An isolated 30-degree, 11-second rest-to-rest slew has zero average torque but
reaches 0.005 N·m·s under the existing torque proxy. With capacity 0.003 N·m·s,
the old endpoint-only update missed saturation entirely.

The profile now retains executed torque phases (not future planned phases).
Both Science and Master validate these before applying their average torque.
The guard calculates the no-unload prefix peak and subtracts an upper bound on
all possible unloading over the interval. If that lower bound still reaches
capacity while the net-impulse endpoint is below it, the run raises an explicit
`FloatingPointError` asking for a smaller timestep. Wheel state is not mutated
by the check. Invalid timestep/scales, non-finite state/history and overflow are
rejected. Existing endpoint saturation and recovery rules are unchanged.

This is a sufficient detection guard, not automatic adaptive stepping or an
exhaustive event solver. It can leave marginal cases unresolved when the unloading
bound is loose. It also does not repair the inertia/torque-proxy limitation above.
Accepted trajectories and checkpoint format remain version 5; phase history is
recomputed on advancement. Historical runs are not retroactively certified.

Regression cases cover the hidden crossing and a safe 0.006 N·m·s capacity in
both implementations, executed-phase truncation, and one-second stepping into
the existing saturation hold. No annual simulation was run for this follow-up.

Follow-up verification: **176 combined tests passed** (3,809 wrapper warnings);
after adding final unloading/integration overflow guards, **56 slew-profile tests
passed** again (3,448 wrapper warnings). All source compilation and
`git diff --check` passed. No commit or push was performed.

## Follow-up: atomic wheel updates and rejection ordering

Inspection found that `update_wheel_momentum()` changed momentum before checking
derived values. NaN torque could propagate into momentum and the subsequent
comparison could clear `is_saturated`; overflow could leave partial actuator
state. Both implementations now use the shared validated update. Torque shape,
finite/nonnegative time/unloading inputs, positive inertia/capacity, recovery
fraction and finite actuator state are checked. Momentum and RPM are computed
locally and committed only after successful validation. Zero-duration updates
do not change state. The impulse-then-unload approximation and recovery policy
remain unchanged for normal finite inputs.

Both mission loops now perform the hidden-peak guard immediately after attitude
guidance, before payload/storage/EPS accounting. A controlled Science loop test
verifies that rejection neither calls payload/solar/EPS updates nor advances
the mission clock, and that saving the incomplete interval is rejected.
Guidance itself has already run: this is not an atomic retryable mission step.
Restore a valid checkpoint after failure; automatic timestep refinement remains
open and must not retry partial intervals in place.

New shared/Master regressions cover invalid torque/time/scales/configuration,
impulse/RPM/unloading overflow, unchanged full actuator state on failure,
non-finite momentum retaining saturation flags, zero time, and no unloading
overshoot. Tests snapshot serialized actuator state before and after rejection.

Verification: **209 tests passed** in the combined ten-file suite (3,809 existing
wrapper warnings). All-source compilation and `git diff --check` passed. No
annual run, commit or push was performed. Checkpoint format remains version 5.
