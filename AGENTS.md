# Repository Guidelines

## Project structure

ANTARA models a 3U CubeSat with Python and Orekit. Run entrypoints from the repository root.

- `src/simulations/`: implementations grouped into `ditl`, `thermal`, `conops`, `power`, `detumble`, `orbit`, and `pointing`. Science Sim in `ditl/` is the authoritative DITL simulator. Master is legacy and excluded from further repair/parity work.
- `src/`: package root only; implementations and module guides have moved into `src/simulations/`. Do not restore flat copies or wrappers. See [layout](docs/SIMULATION_LAYOUT.md).
- `src/simulations/helper/`: shared configuration/models, paths, FSM specification, slew/wheel numerics, telemetry and trusted-pickle migration. Use qualified imports.
- `tests/`: pytest regressions; `tests/validation/`: standalone subsystem reference checks.
- `scripts/`: backend verification and validation-report generation.
- `docs/`: repository-wide index, cross-cutting reports and runner operations. Simulation-specific guides and FSM diagrams live beside their implementations.
- `results/`, `figures/`, `data/`: generated outputs and datasets, generally ignored by Git.
- `notebooks/`, `archive/`, `tmp/`: analysis, legacy code, and scratch work.

## Environment

Local development uses Conda `orekit_env`; CI uses Micromamba `antara` with Python 3.11. Local verification also passed with Python 3.10.21. Do not require local Micromamba or install physics packages only into base Conda.

```bash
conda activate orekit_env
conda install -c conda-forge orekit numpy pandas matplotlib cartopy shapely scipy skyfield pytest pip ppigrf=2.1.0
python -m pip install spacepy==0.7.0 graphviz
python scripts/verify_physics_dependencies.py
```

Orekit is supplied through conda-forge, not `pip install orekit`. Java is required by its wrapper. Preserve JVM initialization order before Java-backed API use. The verification script prints interpreter/module locations and executes IGRF and IRBEM queries.

Prepare Orekit data at repository-root `orekit-data/orekit-data-main/` or `orekit-data.zip`. From the root:

```bash
python -c 'import orekit; orekit.initVM(); from orekit.pyhelpers import download_orekit_data_curdir; download_orekit_data_curdir()'
```

Science calls `init_orekit_data()` without downloading. Searches include the root locations and `data/orekit-data.zip` written by the optional downloader; see [paths documentation](src/simulations/helper/paths.md).

## Development and verification commands

```bash
python -m compileall -q src
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -v
python src/simulations/conops/ANTARA_conops.py
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots
```

The two smoke commands should exit 0; verified runs produced 1,801 telemetry rows over 3600 seconds. Two seconds is the maximum interval: wheel-capacity events may shorten it, recorded by `wheel_event_split`; the final horizon may also clip it. Check interval accounting, not row count alone. Preserve existing tagged results before rerunning. Pytest files follow `test_*.py`; no repository-wide coverage percentage is enforced.

Use `python scripts/generate_validation_report.py` for subsystem reference checks and `python src/simulations/ditl/plot_fsm_diagram.py --output-dir src/simulations/ditl --format both` for diagrams (requires system Graphviz). Master is superseded: retain existing changes and historical evidence but do not develop it further. Annual Science, MCR, and battery-sizing studies are expensive; run targeted checks first.

Detumble Monte Carlo shows parent-only progress/ETA on stderr by default; `--no-progress` hides it, and `--workers` bounds CPU use. Preserve seeded ordering and portable pickle classes. See [detumble usage](src/simulations/detumble/leop_detumble_mc.md). The verified annual-preset run covers 60 hours, not a full year; retain this distinction in documentation and handoffs.

## Configuration and architecture

Use only categorized source paths and qualified `simulations.*` imports. There is one implementation per module; old flat commands are removed. Package initializers must not import simulations: several studies execute immediately on import. Keep `tests/test_simulation_layout.py`, spawn/pickle and checkpoint regressions passing. Existing checkpoint and analysis loaders migrate historical module names for trusted pickles; physics-version compatibility is unchanged. Never launch annual studies to check imports.

Explicit Science CLI values override scenario presets, which override global defaults. `--step-sec` selects fixed stepping unless `--coarse-step-sec` is also supplied. `--no-plots` works with presets; preset-only annual runs retain multirate stepping.

Science uses `MissionConfig` and `ADCSModel` from `antara_models.py`. It defines local `EPSModel`, `SpacecraftFSM`, and `IllegalFSMTransitionError`; do not import the shared names over these local implementations. Other studies maintain separate configurations and subsystem implementations. Inspect actual definitions and consumers before synchronizing a change; the shared FSM drift guard does not cover every local FSM.

## Physics and numerical conventions

- Use four-space indentation, descriptive snake_case functions/variables, PascalCase classes, and existing type annotations. Put new numeric configuration parameters in the appropriate `MissionConfig`.
- Preserve scalar-first `[w,x,y,z]`, body-to-inertial quaternions; +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel.
- Use Orekit `AbsoluteDate` in UTC and elapsed seconds. Route simulation outputs through `results_path()` and `figures_path()`.
- The actual IGRF import is `ppigrf`. Its API takes altitude in km and returns ENU in nT; ANTARA converts to NED in Tesla. Version 2.1.0 uses IGRF-14 coefficients. Missing ppigrf triggers a dipole fallback.
- IGRF inertial queries use the current mission `AbsoluteDate` for both coefficients and Orekit Earth-fixed-to-inertial rotation; apply rotation only to the field vector. Standalone NED queries preserve fractional calendar years. Do not restore the former March 1 approximation or relabel ECEF as inertial.
- Relay LOS, link elevation and downlink credit must share the highest-visible-elevation station selection, with primary-station tie-breaking. Keep contact visibility and peak elevation separate per station, including first-AOS downlink credit.
- A change of selected relay station must reset settlement and reacquire through SLEWING before transmitting; never finish against the previous station's command. Preserve the serializable station label across checkpoints and the real handover/restart and natural SAA-exit tests in `tests/test_execution_integrity.py`.
- Reject non-finite shared ADCS inputs and integration overflow; never turn invalid rates into zero-rate detumble success. Run `tests/test_scientific_integrity.py` for these regressions. Remaining work is tracked in [current status](docs/CURRENT_STATUS.md); the scientific-integrity repair report preserves a superseded historical backlog. Do not equate targeted passes with full simulation validation.
- SpacePy/IRBEM imports do **not** imply live AP9. The legacy AP9 accessor explicitly warns once and returns the configured baseline without calling unsupported SpacePy APIs. Detector counts consume the environment spectrum interpolated onto detector bin centers; look direction is the achieved +X boresight. Never label AP8 verification as AP9 validation.
- Startup alignment uses normalized cross/dot `atan2` with an unchanged `1e-6°` numerical tolerance. Keep geometric self-tests and negative alignment tests; do not equate numerical tolerance with flight pointing accuracy.
- Operational Science payload/relay/antenna/panel angles use that same stable angle helper. Keep tiny-angle regressions and the strict handover checkpoint comparison; do not relax restart tolerances to conceal `acos` conditioning near alignment.
- RELAY must follow the moving LOS during acquisition and tracking; do not complete slews from a timer alone. Preserve finite slew dynamics and continuous valid-tracking settlement.
- SUN_POINTING must check the retained target against current RAM/nadir geometry before advancing, not just the achieved attitude. An out-of-budget target must bypass the deadband and unchanged-command shortcut; retain the 5-degree hard limit.
- Saturation hold is not completed acquisition or valid tracking. Preserve the pending slew objective across unloading; validate the frozen attitude during hold and nominal pointing after release. Do not generate science or transmit during saturation hold.
- Frozen SLEWING uses the pending mode's pointing budget and must reach the existing saturation-timeout SAFE policy before considering timer-based completion. Normal acquisition is not a pointing fault.
- Multirate intervals must advance dynamics, EPS, payload, and the clock consistently. Keep DETUMBLE on the fine step and avoid duplicate EPS updates. Process-pool workers must be module-level callables with serializable arguments; verify spawn as well as serial behavior.
- Science `run()` and `run_steps()` share one loop. Stop integration at the horizon; the final CSV row is a zero-duration observation. Preserve `interval_duration_sec`, `state_time_sec` and final histogram-flush accounting. Geometry/mode belong to the interval start; integrated subsystem state belongs to its end.
- Master also records actual interval duration/end-state time and weights time-based summaries accordingly. It has no separate terminal observation: its final integrated row ends at the horizon. Do not apply Science's zero-terminal-row convention to Master logs. Preserve short Master checks in `tests/test_wheel_events.py`.
- Version-9 checkpoints include bounded per-pass statistics, a verified completed-proton-pass archive prefix, and private guidance/RNG state. Retain referenced JSONL files under results/science as well as telemetry; resume forks the saved prefix. Version-7/8 snapshots are migrated on load; version-6 and older remain rejected. September 28 nominal-frame and slew-energy corrections invalidate version-6 and older histories. Write only at complete-step boundaries, atomically and synchronously; never restore the wall-clock worker thread or lossy JSON fallback. Preserve the telemetry byte prefix, resume to a separate file, and reject incompatible legacy state. Run `tests/test_execution_integrity.py` and `tests/test_checkpoint_roundtrip.py`.
- Nominal Science LVLH must retain +X=RAM and +Y=projected nadir for either Sun hemisphere; +Z is the derived positive orbit normal, not necessarily Sun-facing. Never flip the nadir axis to improve panel illumination. Moving-LOS relay acquisition uses continuous valid alignment, not zero angular rate. Science/Master EPS weight idle/slew power by actual interval motion duration. See [short operational verification](src/simulations/ditl/SHORT_OPERATIONAL_VERIFICATION.md).
- Per-bin uncertainty uses per-bin counts; apply configured bin merging consistently to archived and pending histograms. Expected-count precision is not proof of absolute flux calibration. For the current repair work, use short verification runs only; the developer reports an annual run completed on September 24, but no annual rerun is authorized.
- Run `tests/test_audit_repairs.py` for the September 25 audit repairs. ADCS must not snap through its acceptance band or clip divergent rates. Wheel momentum reacts with opposite sign to bus torque. Detumble trials require zero initial acceleration (a torque/disturbance model is needed otherwise), finite positive steps/timeouts, and finite wheel capacity. Baseline failure retains a finite timeout.
- Science scalar summaries are chunked, elapsed-time weighted JSON; plots above 5,000 rows are display-only decimation. Do not use sampled plots for statistics. Historical pre-repair payload and MC results are not directly comparable to corrected runs. See [audit repairs](docs/AUDIT_REPAIRS_2026-09-25.md).
- Plot sampling uses two bounded CSV passes to select globally spaced row indices and retain both endpoints. Do not repeatedly thin earlier samples as later chunks arrive. This is uniform by row, not elapsed time, and does not guarantee retention of brief events or extrema.
- Reporting rejects non-finite/nonmonotonic timestamps, mismatched interval/end-state metadata and discontinuities across CSV chunks. Completed streaming summaries require a zero-duration final observation. Preserve absolute-only timing tolerances and `tests/test_telemetry_integrity.py`; do not feed display-decimated rows into exact metrics.
- Scalar averages reject infinite measurements and invalid weights. Missing numeric samples are omitted from means but exposed in JSON as `valid_duration_sec`, `missing_duration_sec` and `missing_rows`; all-missing metrics use null, not zero. Normalize averaging weights before multiplying large values.
- Science streaming summaries parse scalar columns explicitly; do not rely on per-chunk inferred dtypes. Register new textual/vector columns in `SCIENCE_NONSCALAR_COLUMNS`. Malformed scalar text must fail, numeric-looking cell IDs must stay metadata, and nullable booleans must retain consistent coverage across chunks.
- Validate Science mode labels before grouping duty cycles; missing/unknown labels must not silently discard time. Parse Boolean flags explicitly (True/False or 0/1), never with string truthiness. Empty optional event CSVs mean no events; malformed event CSVs and empty required telemetry must still fail. Regressions live in `tests/test_scheduled_validation.py` and `tests/test_telemetry_integrity.py`.
- Comparison byte totals require finite nonnegative samples; do not silently skip missing increments or invent zero for absent columns. Sum interval byte increments without duration weighting, including terminal histogram flushes. Efficiency is unavailable when generated bytes are zero or a required counter is absent.
- Scenario comparison CLI streams selected scalar columns with one-row lookahead and cross-chunk hold state. Preserve legacy timestamp weighting and full-duration checks; do not reintroduce whole-file reads or detector arrays into this path. The older `load_scenario_data` helper remains a full-frame API for small callers, not the CLI.
- Bounded comparison still requires a zero-duration terminal observation in current-format telemetry. `--require-full-duration` adds preset endpoint checks; it must not be the only gate against unfinished interval accounting. Legacy timestamp-only files retain inferred terminal weight and weaker completion evidence.
- Credit pending histogram packets to the SAA pass before closing its event, including horizon closure. Acquisition counts and serialized packet bytes are different quantities; a buffered measurement is not zero science. Preserve the boundary and mid-DETECTION checkpoint regressions in `tests/test_audit_repairs.py`.
- Slew torque direction comes from the quaternion rotation axis transformed into the current body frame, never diagnostic yaw/pitch/roll ordering. Cache it during advancement so diagnostic angle queries cannot change wheel momentum. The acceleration-to-torque amplitude remains an approximate proxy, not a coupled rigid-body controller.
- Shared and Master ADCS use `slew_profile.py` for event-resolved acceleration/coast/braking. Do not restore end-rate-times-step displacement or discard a nonzero rate on entering the pointing band or changing command mode. Test `tests/test_slew_profile.py`; retain short-run scope and the one-axis model limitations in [slew repair evidence](SLEW_PROFILE_REPAIR.md).
- Both mission loops preview slew phases on a disposable ADCS copy and shorten the real interval at the first predicted wheel-capacity crossing. Integrate wheel drive/unloading per phase, with identical durations for dynamics, EPS, payload and clock. Preserve the defensive peak guard, existing hold/recovery policy and `tests/test_wheel_events.py`. See [wheel-event repair](WHEEL_EVENT_REPAIR.md); this is not a coupled rigid-body actuator model or an adaptive DETUMBLE solver.
- Run wheel peak guards immediately after guidance, before payload/EPS accounting. Wheel updates validate inputs, proposed momentum and RPM before committing state; failed checks must not clear saturation or partially update actuator state. Preserve `tests/test_scientific_integrity.py` atomic-update checks and the rejected-interval ordering test in `tests/test_execution_integrity.py`.

## CI and change handoff

All five Orekit CI jobs install ppigrf through conda-forge and SpacePy through the activated interpreter's pip, then verify module locations and backend execution. Nightly targets the dedicated local `antara-nightly` runner on the default branch only; other jobs stay GitHub-hosted. Its separate service account/Micromamba environment must not touch the developer's `orekit_env`. The local runner was installed and verified online on 2026-09-13 with its hardened systemd service active; see [runner operation](docs/LOCAL_NIGHTLY_RUNNER.md). Do not rerun the installer on this healthy registered runner. Service-account physics setup remains unverified. The annual process budget is 20 hours inside a 23-hour job; completion remains unverified. Scenario CI checks actual duration and telemetry cadence. The Graphviz job uses a separate minimal environment.

Scheduled runs have isolated concurrency groups and manual `suite=nightly|weekly` dispatch. Validation reports, JSON and full logs go to `results/validation/`; use a fresh `--output-dir` for reruns. Preserve historical root reports. Orbit validation compares like-for-like two-body periods and full-force J2 drift; IGRF uses independent IRBEM with an explicit 2025 endpoint caveat, not undocumented NOAA numbers. Run `tests/test_scheduled_validation.py` for report/CLI regressions. The full-year nightly timeout remains unverified; do not silently shorten the scenario or present partial annual telemetry as complete.

Check `git status --short` and preserve existing edits. Avoid destructive resets and unrequested commits or pushes. Describe the change, relevant verification, and remaining fidelity limits in the handoff or PR. Keep operational docs synchronized with code; retain historical report results with an explicit date/status note. See [README](README.md), [Science guide](src/simulations/ditl/ANTARA_Science_Sim.md), and [repair evidence](OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md).
