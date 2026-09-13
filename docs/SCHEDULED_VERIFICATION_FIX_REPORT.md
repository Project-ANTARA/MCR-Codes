# Scheduled Verification Repair Report

Date: 2026-09-13. Starting revision: `ce4557ce8aac29cf074b742599374a979e1d48e0`; working tree was clean. No commit, push, or remote workflow dispatch was performed.

## Observed failures

- [Nightly run 34743815167](https://github.com/Project-ANTARA/MCR-Codes/actions/runs/34743815167) failed at 9.65 simulated hours with `RELAY pointing error 6.005910 deg`. It ran older revision `3ace53f0a1bf3dd95a24228c231f53ff50143d09`, before the runtime repairs already present in the starting revision. Physics installation and backend verification succeeded.
- [Weekly run 34746578186](https://github.com/Project-ANTARA/MCR-Codes/actions/runs/34746578186) was cancelled during setup, not a failed physics test. Its 08:00 start overlapped the 08:00:05 push. All events previously shared a cancellable workflow/ref concurrency group.
- Local validation independently reproduced missing UTC data. Subsequent harness defects included undefined `org`, NumPy scalars rejected by Java constructors, unsupported orbit conversion, and singular near-circular Keplerian integration.

## Repairs

The workflow now isolates event/schedule/manual-suite concurrency, preserves running scheduled jobs, and exposes manual `nightly`/`weekly` dispatch. Push/PR jobs remain push/PR-only. Regression CI includes the new scheduled-entrypoint tests.

Orbit validation loads existing Orekit data, uses explicit Java imports and native floats, converts via `KeplerianOrbit`, and integrates nonsingular Cartesian coordinates. The J2 formula now uses positive `J2 = -C20`. The full-force osculating period averaged 5789.54 seconds versus the initial two-body 5801.23 seconds: those are not comparable at a one-second tolerance. A separate central-gravity numerical crossing now tests the same Kepler period requirement; the full-force 24-hour RAAN comparison remains. No simulator force models or acceptance tolerances changed.

The old rounded IGRF magnitudes lacked reproducible reference output and disagreed with both ANTARA and independent IRBEM. Locations and legacy values remain recorded for audit. Acceptance now compares ppigrf against IRBEM's separate Fortran implementation, with external field disabled and GEO converted to geodetic NED. Both magnitude and vector must satisfy the original 5% bound. Dipole fallback fails validation. SpacePy 0.7's IGRF-13 endpoint is explicitly used for the 2025 case, not claimed as exact IGRF-14 epoch agreement. Its endpoint warning remains visible. See [SpacePy's model/version explanation](https://github.com/spacepy/spacepy/discussions/801).

Reports now read the actual six JSON filenames, show configured tolerances, retain full subprocess logs, and fail on missing quantitative evidence. All six scripts support `ANTARA_VALIDATION_OUTPUT_DIR`; the report runner supplies it. Default outputs are `results/validation/`; reruns require a fresh `--output-dir` to preserve evidence. CI uploads these actual paths instead of historical tracked reports. Historical root reports and JSON captures were not overwritten.

Scenario comparison honors `--results-dir`, reads current flat outputs, and retains legacy nested support. Nightly requires every scenario and full 24h/24h/8760h monotonic telemetry. Partial annual output cannot silently count as a completed campaign. Sample-weighted comparison statistics are unchanged and remain unsuitable for duration-weighted multirate mission claims.

## Verification

Local physics interpreter: `/home/sammo/miniconda3/envs/orekit_env/bin/python`, Python 3.10.21, Orekit 13.1. CI Python 3.11 execution is still pending publication.

| Check | Result |
| --- | --- |
| All six subsystem scripts via report CLI | PASS, exit 0 |
| Two-body period error | 4.55e-12 s; unchanged 1 s limit |
| Full-force RAAN drift error | 0.2821%; unchanged 5% limit |
| IGRF, 17 locations/epochs | PASS; worst magnitude 0.2221%, vector 0.3044% |
| Combined targeted pytest suite | 47 passed; 3,448 wrapper deprecation warnings |
| Missing scenarios, incomplete duration, stale JSON, failed subprocess, timeout, absent evidence, reversed NED, dipole fallback | Rejected in regression checks |
| Comparison CLI on existing flat output | Exit 0, CSV and PNG produced |
| Full-duration CLI on existing partial annual CSV | Correctly rejected 1014h instead of 8760h, exit 2 |
| Workflow YAML parse and dispatch/concurrency structure | PASS (installed ruamel parser; not hosted execution) |
| Python compilation and `git diff --check` | PASS |

Commands actually run include:

```bash
git status --short
git rev-parse HEAD
gh run list --repo Project-ANTARA/MCR-Codes --limit 15 --json databaseId,event,conclusion,createdAt,displayTitle
gh run view 34743815167 --repo Project-ANTARA/MCR-Codes --log-failed
python tests/validation/test_orbit_propagation.py
python scripts/generate_validation_report.py --output-dir /tmp/antara-validation-PrqQTR
python -m pytest tests/test_scheduled_validation.py tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -q --disable-warnings
python src/compare_scenarios.py --results-dir results/science --output-dir /tmp/antara-nightly-comparison --require-all
python src/compare_scenarios.py --results-dir results/science --output-dir /tmp/antara-nightly-duration-check --require-all --require-full-duration
python -m py_compile scripts/generate_validation_report.py src/compare_scenarios.py tests/test_scheduled_validation.py tests/validation/*.py
git diff --stat
git diff --check
```

Here `python` denotes the local Orekit interpreter; sandbox runs also set `SPACEPY=/tmp`, `MPLCONFIGDIR=/tmp/antara-mpl`, and `PYTHONPATH=src`. A separate base-interpreter invocation only parsed YAML using its installed ruamel library; no physics dependencies were installed into base. Intermediate failing runs and direct numerical probes were retained under `/tmp/antara-validation-3gxNVY` and terminal output. Successful quantitative evidence and full logs are in `/tmp/antara-validation-PrqQTR`.

## Remaining limits / next run

**Subsequent user decision:** keep full-year nightly on the local machine.
The workflow now targets `antara-nightly` with a 23-hour job budget and
20-hour annual-process timeout. The service installer is prepared but requires
the user's sudo prompt; registration and service startup have not occurred.
See [local runner installation/status](LOCAL_NIGHTLY_RUNNER.md). The following
60-minute budget description records the initial investigation, not the updated workflow.

The nightly full-year step still requests 8,760 simulated hours within 60 wall minutes, followed by 5,000 MC trials under a 180-minute job budget. Neither the full-year runtime nor that entire sequence was verified in this repair. The existing annual CSV is incomplete; it was only read, not modified. Prior 60-hour runtime evidence does not establish full-year success.

No scientific assertions were disabled and no tolerances were relaxed. No live AP9 or SGP4 cross-validation is claimed. The workflow's concurrency/dispatch design follows [GitHub's workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax).

After publishing this change on the default branch, trigger the desired suite:

```bash
gh workflow run ci.yml --ref main -f suite=weekly
gh workflow run ci.yml --ref main -f suite=nightly
```

Assessment: **local validation repairs verified; runner installation and full-year execution remain pending**. The user selected the longer-running local runner; no shortened nightly scenario was introduced.
