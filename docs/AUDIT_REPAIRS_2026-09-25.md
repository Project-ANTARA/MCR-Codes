# Scientific audit repairs — 2026-09-25

## Scope and compatibility

Repairs build on the existing dirty working tree at
`3a822ac65fea5b8e5a42fd0387e79811c22a3706`; earlier execution/checkpoint changes
were preserved. No annual simulations, commits, pushes or CI dispatches are used.
Old payload counts and detumble capability estimates are not comparable without
rerunning the same short inputs. Checkpoint version 3 rejects older physics state.

## Changes

| Finding | Repair |
| --- | --- |
| ADCS minimum 0.4-degree jump / target snap | Remove minimum displacement and target snapping inside the pointing acceptance band; zero time produces zero motion. |
| Braking torque loses sign | Preserve signed angular acceleration; wheel momentum opposes bus torque. Applied in shared and Master implementations. |
| Finite rate divergence clipped | Raise a numerical failure; MC records failure instead of reporting false convergence. Master uses the same guarded rate integrator and inertia-scaled wheel damping. |
| Payload bypasses environment flux | Interpolate the authoritative environment differential spectrum linearly in energy onto detector bin centers, then apply response/deadtime. Validate spectrum shape, coverage and finite nonnegative values. |
| Directional flux ignores achieved pointing | Supply achieved +X boresight as look direction. Legacy velocity argument names remain for compatibility. |
| Unsupported AP9 call | Explicit configured-baseline accessor with one warning; no invented SpacePy AP9 API. Legacy `ap9_flux` fields are baseline values, not independent observations. |
| SAA entry clears first measurement | Open buffers before acquisition/counting; credit the first entry packet. |
| MC wheel momentum stays zero | Integrate reaction torque, track peak and fail on wheel capacity. No unloading is assumed in the isolated trial. |
| Ineffective acceleration sampling | Rate-only trials; reject nonzero initial acceleration. A real disturbance needs a specified torque/time model. The analysis skips degenerate 2D fitting. |
| Infinite failed-baseline timeout | Retain finite 10,000-second bound; validate positive finite steps/timeouts and worker count. |
| Row-weighted reports | Duration-weighted comparison and Science scalar summaries; terminal observation has zero duration. |
| Thermal extra integration | Record state at its timestamp; integrate only remaining time, including fractional last steps. |
| Master old execution path | Stop at horizon, check attitude completion/settlement rather than timer alone, refresh moving slew target, restore validation, and distinguish active link time from visibility. |
| Battery efficiency on discharge | Divide deficit by efficiency; reject sampled efficiencies outside (0,1]. |
| Standalone power mass | Explicit 3 kg Science-mission mass instead of Orekit's default 1000 kg. |
| Censored drag lifetime | Return infinity and print `>30 (censored)` when the threshold was never reached. |
| Orekit download/search mismatch | Search the downloaded data location and report actual candidates. |
| Unseeded supporting studies | Study-local seed 42; power RAAN sweep no longer duplicates 0/360 degrees. |
| Annual telemetry working memory | Chunk scalar summary; constant-space orbital summaries; plots use at most 5,000 display samples. Full telemetry remains on disk. |

## Verification

Use the Conda `orekit_env` interpreter, with `PYTHONPATH=src`, `SPACEPY=/tmp`
and an isolated `MPLCONFIGDIR`. `tests/test_audit_repairs.py` includes negative
inputs, motion/torque, flux response, finite timeout, wheel reaction, interval
weights, streaming chunk boundaries, SAA entry, data discovery, unfinished Master
slews and a real five-second thermal run. CI's unit command includes this file.

Final combined regression run: **104 passed** (35.98 seconds, 3,809 wrapper
deprecation warnings). An earlier new SAA test had an incorrect fixture
constructor; corrected to the actual no-argument API before this run.
Python compilation of all `src/*.py` and `git diff --check` passed.

Executed regression command:

```bash
PYTHONPATH=src SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl python -m pytest \
  tests/test_audit_repairs.py tests/test_execution_integrity.py \
  tests/test_checkpoint_roundtrip.py tests/test_physics_attitude.py \
  tests/test_scientific_integrity.py tests/test_detumble_multiprocessing.py \
  tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py \
  tests/test_runner_installer.py -q --disable-warnings --tb=short -p no:cacheprovider
python -m py_compile src/*.py
git diff --check
```

The isolated three-trial MC CLI (`--workers 1 --timeout-sec 2 --no-progress`)
and `analyze_detumble_mc.py` both exited 0, including an all-timeout analysis.
These are process/analysis smoke checks, not a capability estimate. Existing
regressions also verify serial/spawn equivalence.

Short smoke artifacts: `/tmp/antara-audit-smoke-gyzouw2p/`.
Both `leop_24h` and `no_leop_24h` with 360-second overrides completed with
181 rows, two-second cadence and exactly 360 integrated seconds. Modes were
COLD_AND_DARK/DETUMBLE and SUN_POINTING respectively. Master also completed a
five-second check with plotting suppressed in the harness. A separate 360-second
Master check passed through COLD_AND_DARK, DETUMBLE and SUN_POINTING; artifacts
are in `/tmp/antara-master-audit-4k93cv8b/`. These are not
operational DETECTION/RELAY acceptance runs; targeted subsystem tests cover those
interfaces separately.

## Remaining fidelity limits

- No live AP9, detector calibration, stochastic arrival process or new external
  radiation data was invented. Counts remain expected counts; interpolated
  baseline spectra are modeling assumptions, not validation against flight data.
- Attitude guidance remains a kinematic command model with estimated wheel
  torque, not a sensor/estimator/flexible-body hardware simulation. No flight
  pointing requirement or quaternion convention was changed.
- MC uses a synthetic field and no mission EPS/FSM. Nonzero initial acceleration
  is intentionally unsupported until a physical disturbance definition exists.
- Chunking bounds telemetry-summary working memory, not every event/pass archive
  or checkpoint size. Display decimation can omit short events; analyze the full
  CSV for event investigations. Master and other legacy studies retain separate
  model architectures and need their own longer acceptance campaigns.
- No annual rerun was performed. A completed annual run does not establish model
  fidelity or convergence. The independent value of the fixes is established by
  short regression cases, not a claim that every possible repository defect is gone.

## September 26 follow-up: pass closure and buffered acquisition

Additional controlled DETECTION tests found that a 272-byte packet flushed at
SAA exit or the simulation horizon reached telemetry/storage but was absent from
`saa_passes.csv`. Both boundary tests failed with zero pass credit versus
0.0002593994140625 MB generated. The exit path now flushes the payload before
finalizing the geographic pass; horizon closure adds pending packet credit to
the open pass before event finalization. Counts and packet size are unchanged.

The old zero-science warning also confused packet serialization with acquisition.
It now examines accepted detector counts, so buffered positive counts do not
warn, while genuine zero counts still do. No warning was simply disabled.

A six-second controlled DETECTION checkpoint test compares uninterrupted and
resumed telemetry, SAA-pass CSV and proton-pass CSV. Pending counts survive the
checkpoint and are credited exactly once on closure. These fixtures control SAA
membership and hold attitude while updating pointing telemetry; they test payload
accounting and restart behavior, not flight attitude accuracy. No annual run was
performed.

Follow-up verification: **108 tests passed** in the combined regression suite;
the five boundary/restart cases passed again after adding explicit onboard
storage plus downlink conservation assertions. Compilation and `git diff
--check` passed. Zero-count observations still produce fixed-width packets;
tests verify that those bytes are conserved while the zero-count warning remains
active. No commit or push was performed.

## September 26 follow-up: slew torque axes

Isolated axis probes exposed an X/Z swap: a pure body-X slew reported
`[0, 0, 0.001]` N·m, while body-Z reported `[0.001, 0, 0]` N·m.
Both shared and Master ADCS interpreted diagnostic `[yaw, pitch, roll]` as
body `[X, Y, Z]`. An unrelated `compute_slew()` diagnostic call could also
change the torque used for wheel integration without changing the attitude.

Both implementations now derive the direction from the quaternion delta's
inertial axis, rotate it into the current body frame, and cache the signed
acceleration-based torque during advancement. Diagnostic queries cannot mutate
the cached result. Zero-duration advancement, saturation hold, and attitude
initialization clear or suppress the interval torque. Master retains its existing
nominal-tracking torque scope; Science retains its explicit SLEWING-mode scope.
No quaternion convention, pointing limit, or assertion was weakened.

Checkpoint format advances to **version 4**. Version-3 wheel histories used the
incorrect axes and cannot be repaired by merely adding a cache field; older
snapshots are rejected. Start a fresh run instead of relabeling an old checkpoint.
Current operational guides have been updated; earlier version-3 evidence above
is historical.

Verification: the combined nine-file regression suite passed **118 tests**
(3,809 existing wrapper warnings). Axis tests cover X/Y/Z and an oblique axis
from a nonidentity initial attitude in both implementations, diagnostic-call
independence, and signed braking. These are isolated short dynamics tests, not
annual validation. The torque magnitude remains the existing acceleration-to-
maximum-torque proxy, not inertia-derived rigid-body dynamics. Acceptance-band
rate resets and finite-step braking consistency remain open model limitations.

## Subsequent slew-profile repair

The acceptance-band rate reset and finite-step braking items immediately above
are addressed in the [September 26 slew-profile report](../SLEW_PROFILE_REPAIR.md).
Current checkpoint format is version 5. That follow-up records 170 regression
passes and two one-hour smoke telemetry checks, while retaining explicit
one-axis/torque-proxy limitations. Earlier results in this document remain dated
evidence, not the current checkpoint compatibility contract.
