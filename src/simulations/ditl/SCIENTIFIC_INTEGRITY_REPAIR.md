# Scientific-integrity repair record

**Current status (2026-09-28):** use the [reconciled backlog](../../../docs/CURRENT_STATUS.md).
The itemized backlog below predates the subsequent repairs. Checkpoints are now
version 7, and shared execution, flux/count propagation, reporting and handover
repairs have landed; their older descriptions are retained only as evidence.

Current follow-ups, 2026-09-27: [wheel-event repair](../../../WHEEL_EVENT_REPAIR.md)
records automatic interval splitting and version-6 checkpoint compatibility;
[telemetry integrity](TELEMETRY_INTEGRITY_REPAIR.md) adds reporting continuity
checks. The [slew integration repair](../../../SLEW_PROFILE_REPAIR.md) retains its
earlier evidence and remaining physical-model requirements. The backlog
below is historical; do not read it as a list of unchanged current defects.

Update, 2026-09-25: the next batch unifies execution paths, repairs short-run
checkpoint continuation and endpoint accounting, and fixes per-bin precision and
merged histogram consistency. See [execution/payload evidence](EXECUTION_PAYLOAD_REPAIR.md).
The dated baseline/results below are retained as history; their open execution
items should be read together with that follow-up, not as the current implementation.

## Scope and baseline

Date: 2026-09-14. Baseline: `83ae62c2864840359dd1b09a62b4698100962710`,
with a clean working tree. This is the first implementation batch of the audit
repair plan, **not a declaration that all simulation results are validated**.
No commit, push, workflow dispatch, dependency installation or runner change was
performed. Historical reports retain their original verification scope.

## Corrections in this batch

| Defect | Correction | Regression evidence |
| --- | --- | --- |
| Earth-fixed IGRF vector returned as inertial | Rotate geodetic NED into Earth-fixed coordinates, then apply Orekit's Earth-fixed-to-inertial rotation at the query epoch | Known eastward field at two epochs, norm preservation and fallback round-trip |
| Every fractional epoch replaced by March 1; runtime field epoch frozen | Preserve fractions of calendar years for standalone queries; use current mission UTC date for runtime queries | Integer, fractional and leap-year dates; runtime date overrides configured epoch |
| Link elevation/credit could select GS2 while the antenna still targeted GS1 | One highest-visible-elevation policy for link selection and LOS, with primary tie-breaking | Primary-only, secondary-only, overlapping, tied and no-visible-station cases |
| Primary contact history used combined visibility and potentially GS2's elevation | Record visibility, peak elevation and selected-link credit per station | Overlapping AOS/LOS sequence, handover and independently conserved credits |
| First AOS sample's transmitted data discarded | Include that sample in the newly opened contact | Nonzero AOS credit regression |
| Invalid angular rates or integration overflow replaced by zero, allowing false detumble completion | Reject invalid inputs and divergent integration; retain prior rates on failed integration | NaN, positive/negative infinity, invalid field/timestep, overflow and valid rigid-body update |

The IGRF change affects the explicitly selected `earth_b_field_model="igrf"`
path; the separate configured dipole branch is unchanged. The model's
missing-ppigrf fallback is still rotated correctly. The inertial-query API now
requires `absolute_date`, `earth_frame` and `inertial_frame`; the unused
`position_inertial` argument was removed and its repository caller updated.

No quaternion convention, geometric tolerance, mission pointing budget, finite
slew dynamics, orbital propagator or FSM transition policy was changed.
Station prediction still uses the existing primary-station heuristic; this
batch corrects actual visible-link geometry, not pass forecasting. `run_steps`
uses the same selection policy but is **not yet equivalent to `run`**.

## Verification

Local interpreter: `/home/sammo/miniconda3/envs/orekit_env/bin/python`, Python
3.10.21. Tests ran with `SPACEPY=/tmp`, `MPLCONFIGDIR=/tmp/antara-mpl` and
`PYTHONPATH=src` to keep runtime caches out of developer configuration.

- Initial regression run: **11 failed, 3 passed**, exposing the old paths
  (the new inertial signature initially fails against the old API).
- Expanded combined regression run: **70 passed**, including 21 new integrity
  checks and existing attitude, multiprocessing, scheduling, FSM and runner
  regressions. There were 3,448 Orekit-wrapper deprecation warnings.
- Independent IRBEM reference check: **17/17 passed** at matching decimal dates.
  The existing 2025.0 IGRF-13 endpoint warning remains; this is not validation
  beyond that endpoint or an AP9 validation. No trusted reference was generated
from the simulator itself.
- Python compilation and `git diff --check`: passed.
- All three simulation smoke checks below completed successfully (exit 0).

| Preset | Magnetic model | Actual duration | Cadence | Rows | Observed modes |
| --- | --- | ---: | ---: | ---: | --- |
| `leop_24h` | configured dipole | 3,600 s | 2 s | 1,801 | COLD_AND_DARK, DETUMBLE, SAFE, SLEWING, SUN_POINTING |
| `no_leop_24h` | configured dipole | 3,600 s | 2 s | 1,801 | SUN_POINTING |
| `leop_24h` | IGRF | 36 s | 2 s | 19 | COLD_AND_DARK |

These used `_parse_args`/`_config_from_args` with explicit duration and two-second
step overrides, then the real simulator constructor and `run()`. The verification
harness redirected `paths.RESULTS_DIR` and `paths.FIGURES_DIR` into
`/tmp/antara-integrity-YZvvhu/`, gave each run a distinct output tag, and checked
the generated CSV with pandas. For the third case only,
`config.earth_b_field_model` was explicitly set to `"igrf"`; runtime field queries
execute even in COLD_AND_DARK. No simulation physics was mocked in these runs.

Logs are `leop_24h_dipole.log`, `no_leop_24h_dipole.log` and
`leop_24h_igrf.log` under that temporary directory. None emitted a missing-backend
or AP9 warning. These short runs did **not** exercise operational RELAY or
DETECTION; controlled station-selection/contact tests cover those changed
interfaces, but longer mission acceptance is still pending. No annual or hosted
CI run was performed in this batch.

Commands used for the automated checks:

```bash
conda activate orekit_env
export SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src
python -m pytest tests/test_scientific_integrity.py -q
python -m pytest tests/test_scientific_integrity.py tests/test_physics_attitude.py --disable-warnings --tb=short -q
python -m pytest tests/test_scientific_integrity.py tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py -v --disable-warnings --tb=short
ANTARA_VALIDATION_OUTPUT_DIR=/tmp/antara-integrity-YZvvhu python tests/validation/test_igrf_field.py
python -m py_compile src/simulations/ditl/ANTARA_Science_Sim.py src/simulations/helper/antara_models.py tests/test_scientific_integrity.py tests/validation/test_igrf_field.py
git diff --check
```

The temporary output directory was created with `mktemp -d`; create a fresh
directory for another verification run. Existing scenario outputs were not
overwritten. A separate numerical probe compares the former Earth-fixed vector
with the corrected inertial vector at the same location and date, isolating the
frame correction from the epoch correction.

At latitude −25°, longitude −50°, altitude 600 km, 2000-03-01 00:00 UTC,
the formerly mislabeled Earth-fixed vector differs from the corrected inertial
vector by **111.323536538491°**. Both magnitudes are
**1.827875848672952 × 10⁻⁵ T**: a magnitude-only test would miss this defect.
The largest independent IRBEM NED-vector difference is **0.349079%**, within
the unchanged 5% reference-check tolerance.

## Historical audit backlog and repair order (superseded)

September 25 follow-up: [audit repairs](../../../docs/AUDIT_REPAIRS_2026-09-25.md) addresses
the defects below. This list is retained as historical investigation context,
not a claim that the original defects remain unchanged. Live AP9, calibrated
detector physics and full mission acceptance remain explicit limits.

At the time of the original audit, the following work was open. Its current
disposition is recorded in the linked reconciled backlog, not this old list.

1. **Monte Carlo physical consistency.** Define what sampled initial angular
   acceleration represents before changing dynamics; currently it is overwritten
   without affecting the trial trajectory. Integrate returned wheel torque into
   wheel momentum. Separate rate/acceleration sampling directions if independence
   is intended. Bound failed-baseline timeouts and reject invalid CLI values.
   Verify seeded serial/spawn equivalence and physical—not just serialization—results.
2. **One trustworthy integration/checkpoint path.** Make `run_steps` and `run`
   share timestep semantics; remove duplicate ADCS advancement, fix endpoint
   over-integration and use incremental telemetry consistently. Resume from the
   saved clock without truncating telemetry. Include pending guidance/settlement,
   payload counters, pass indices and buffers in checkpoints. Repair fallback
   array serialization and use atomic, synchronized snapshots. Convert the
   standalone checkpoint check into collected uninterrupted-vs-resumed tests.
3. **Authoritative radiation/count pipeline.** Decide whether to use a documented
   configured baseline or integrate a real radiation backend. Current payload
   counting discards the environment's computed flux in favor of another table;
   the unsupported AP9 path must not be described as live AP9. Test flux scaling,
   Poisson counts, per-bin uncertainty, histogram/bin-merge consistency and final
   pass flushing against independent expectations.
4. **Time-weighted reporting.** Weight duty cycles and average metrics by actual
   interval length, not row count or a fixed fine step. Reconcile final payload
   flushes with telemetry/storage totals. Distinguish contact visibility from
   actual valid transmission in the separate Master study.
5. **Supporting-code correctness.** Align Orekit data download/search paths;
   validate numeric configuration boundaries; audit duplicated ADCS/EPS/FSM
   implementations rather than assuming shared fixes reach them. Review the MCR
   study's unseeded randomness and use of a final power sample as an average.
6. **Long-run acceptance.** Address whole-file telemetry reloads and unbounded
   histories before claiming bounded annual memory use. Progress through short,
   24-hour, 60/72-hour and finally 8,760-hour runs on the dedicated local runner.
   Audit remaining thermal/drag and conservation assumptions separately. The
   full-year nightly must remain full-year; neither completion nor scientific
   fidelity is established by a short smoke run.

The acceleration interpretation and radiation-backend choice require explicit
physical requirements; they will not be guessed to make tests pass.

## Files and handoff

Source: `src/simulations/ditl/ANTARA_Science_Sim.py`, `src/simulations/helper/antara_models.py`.
Regressions/reference check: `tests/test_scientific_integrity.py`,
`tests/test_physics_attitude.py`, `tests/validation/test_igrf_field.py`.
Documentation: this record, `docs/ANTARA_Science_Sim.md`, `README.md`, `AGENTS.md`.
CI: `.github/workflows/ci.yml` adds the new regression file to its explicit unit
test command. Runner selection, dependencies, schedules and timeouts are unchanged.
