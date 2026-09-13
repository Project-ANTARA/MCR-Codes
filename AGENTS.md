# Repository Guidelines

## Project structure

ANTARA models a 3U CubeSat with Python and Orekit. Run entrypoints from the repository root.

- `src/`: simulation entrypoints, `paths.py` I/O helpers, `antara_models.py` shared models, `scenarios.py` presets, and `fsm_spec.py`.
- `tests/`: pytest regressions; `tests/validation/`: standalone subsystem reference checks.
- `scripts/`: backend verification and validation-report generation.
- `docs/`: module guides and FSM diagrams. Root reports record dated investigations.
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

Science calls `init_orekit_data()` without downloading. The optional download branch writes under `data/`, but default searches currently inspect only the two root locations; see [paths documentation](docs/paths.md).

## Development and verification commands

```bash
python -m py_compile src/*.py
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -v
python src/ANTARA_conops.py
python src/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 1 --step-sec 2 --no-plots
python src/ANTARA_Science_Sim.py --scenario no_leop_24h --duration-hours 1 --step-sec 2 --no-plots
```

The two smoke commands should exit 0 and produce 1,801 telemetry rows over 3600 seconds with two-second gaps. Preserve existing tagged results before rerunning. Pytest files follow `test_*.py`; no repository-wide coverage percentage is enforced.

Use `python scripts/generate_validation_report.py` for subsystem reference checks, `python src/ANTARA_Master.py` for the longer high-fidelity simulation, and `python src/plot_fsm_diagram.py --output-dir docs --format both` for diagrams (requires system Graphviz). Annual Science, MCR, and battery-sizing studies are expensive; run targeted checks first.

Detumble Monte Carlo shows parent-only progress/ETA on stderr by default; `--no-progress` hides it, and `--workers` bounds CPU use. Preserve seeded ordering and portable pickle classes. See [detumble usage](docs/leop_detumble_mc.md). The verified annual-preset run covers 60 hours, not a full year; retain this distinction in documentation and handoffs.

## Configuration and architecture

Explicit Science CLI values override scenario presets, which override global defaults. `--step-sec` selects fixed stepping unless `--coarse-step-sec` is also supplied. `--no-plots` works with presets; preset-only annual runs retain multirate stepping.

Science uses `MissionConfig` and `ADCSModel` from `antara_models.py`. It still defines local `EPSModel`, `SpacecraftFSM`, and `IllegalFSMTransitionError` that shadow shared imports. Other studies maintain separate configurations and subsystem implementations. Inspect actual definitions and consumers before synchronizing a change; the shared FSM drift guard does not cover every local FSM.

## Physics and numerical conventions

- Use four-space indentation, descriptive snake_case functions/variables, PascalCase classes, and existing type annotations. Put new numeric configuration parameters in the appropriate `MissionConfig`.
- Preserve scalar-first `[w,x,y,z]`, body-to-inertial quaternions; +X=RAM/payload, +Y=nadir/relay, +Z=sun-panel.
- Use Orekit `AbsoluteDate` in UTC and elapsed seconds. Route simulation outputs through `results_path()` and `figures_path()`.
- The actual IGRF import is `ppigrf`. Its API takes altitude in km and returns ENU in nT; ANTARA converts to NED in Tesla. Version 2.1.0 uses IGRF-14 coefficients. Missing ppigrf triggers a dipole fallback.
- SpacePy/IRBEM imports do **not** imply live AP9. The current unsupported AP9 query warns once and retains the configured flux baseline, which affects primary science flux. Never label AP8 verification as AP9 validation.
- Startup alignment uses normalized cross/dot `atan2` with an unchanged `1e-6°` numerical tolerance. Keep geometric self-tests and negative alignment tests; do not equate numerical tolerance with flight pointing accuracy.
- RELAY must follow the moving LOS during acquisition and tracking; do not complete slews from a timer alone. Preserve finite slew dynamics and continuous valid-tracking settlement.
- SUN_POINTING must check the retained target against current RAM/nadir geometry before advancing, not just the achieved attitude. An out-of-budget target must bypass the deadband and unchanged-command shortcut; retain the 5-degree hard limit.
- Saturation hold is not completed acquisition or valid tracking. Preserve the pending slew objective across unloading; validate the frozen attitude during hold and nominal pointing after release. Do not generate science or transmit during saturation hold.
- Frozen SLEWING uses the pending mode's pointing budget and must reach the existing saturation-timeout SAFE policy before considering timer-based completion. Normal acquisition is not a pointing fault.
- Multirate intervals must advance dynamics, EPS, payload, and the clock consistently. Keep DETUMBLE on the fine step and avoid duplicate EPS updates. Process-pool workers must be module-level callables with serializable arguments; verify spawn as well as serial behavior.

## CI and change handoff

All five Orekit CI jobs install ppigrf through conda-forge and SpacePy through the activated interpreter's pip, then verify module locations and backend execution. Scenario CI checks actual duration and telemetry cadence. The Graphviz job uses a separate minimal environment.

Check `git status --short` and preserve existing edits. Avoid destructive resets and unrequested commits or pushes. Describe the change, relevant verification, and remaining fidelity limits in the handoff or PR. Keep operational docs synchronized with code; retain historical report results with an explicit date/status note. See [README](README.md), [Science guide](docs/ANTARA_Science_Sim.md), and [repair evidence](OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md).
