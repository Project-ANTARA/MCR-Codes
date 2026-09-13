# ANTARA Physics Dependency + Attitude Fix Report

> Historical dependency/startup repair evidence. Current operational guidance is in the [README](README.md), [Science guide](docs/ANTARA_Science_Sim.md), and [detumble guide](docs/leop_detumble_mc.md). The later [runtime report](docs/SIMULATION_RUNTIME_FIX_REPORT.md) records the 60-hour annual-preset saturation recovery and progress-display verification. The original test counts and captured commands below are preserved; they are not the current total regression count or proof of a full-year/hosted-CI run.

## 1. Executive Summary

- SpacePy status: 0.7.0 imports and bundled IRBEM executes successfully. Live AP9 is **not** supplied by this interface.
- IGRF status: verified `ppigrf==2.1.0`, already installed locally; CI now installs it. Corrected altitude units at the existing API call.
- Attitude self-test status: all startup checks pass with normalized cross/dot `atan2`; the existing `1e-6` degree tolerance is unchanged. Negative checks reject meaningful misalignments.
- Scenario duration override status: explicit CLI values now override presets, including duration, step, and plotting. An explicit fine step also sets the coarse step unless separately specified.
- CI status: dependency setup and regression checks updated; workflow YAML verified locally. GitHub Actions itself was not run or dispatched.

## 2. Environment Findings

Initial HEAD: `34bed5f43709d66c77dcc0c0a221b4321d59eab0`.
Initial `git status --short`: ` M src/ANTARA_Science_Sim.py`.
Existing edits were an FSM variable correction and an `atan2` implementation. The FSM edit was preserved exactly; the existing angle implementation was completed by using its already-computed `cross_norm` instead of computing it again. No reset, clean, commit, or push was performed. Existing `AGENTS.md` was read and left unchanged.

| Item | Verified local value |
| --- | --- |
| Python executable | `/home/sammo/miniconda3/envs/orekit_env/bin/python` |
| Python version | 3.10.21 |
| Orekit version | 13.1 |
| SpacePy version | 0.7.0 |
| IGRF distribution / import | ppigrf 2.1.0 / `ppigrf` |
| Module directory | `/home/sammo/miniconda3/envs/orekit_env/lib/python3.10/site-packages/` |
| CI target | Python 3.11, Micromamba `antara` |

The shell initially resolved base Conda Python. All physics execution used the explicit `orekit_env` interpreter. No local installation was needed: all required modules were already present. Micromamba is absent locally and is not required.

SpacePy initially tried to write its configuration under the sandbox's read-only `/home/sammo/.spacepy`. Local verification used its supported `SPACEPY=/tmp` configuration override, plus `MPLCONFIGDIR=/tmp/antara-mpl`. Neither changes the Python environment or installed packages. Normal developer sessions with writable home directories need neither override.

## 3. SpacePy Findings

Actual import: `import spacepy.irbempy as irbem`. It is used only by `ProtonEnvironmentModel.get_ap9_flux`, but that function supplies the baseline multiplied into `compute_flux`'s primary measured proton flux. Thus the layer is **not merely a disconnected plot comparison**. A working alternative baseline would affect primary science results.

The current call attempts `irbem.Coords(...)` followed by `irbem.get_AEP8(..., whichm="AP9", whatf="Flux")`. Installed source establishes three problems:

1. `Coords` belongs to `spacepy.coordinates`, not `irbempy`.
2. Actual signature: `get_AEP8(energy, loci, model='min', fluxtype='diff', particles='e')`.
3. It implements AE8/AP8, not AE9/AP9. SpacePy's `ae9ap9` module explicitly describes itself as a reader, not a model runner.

Consequently, both absent SpacePy and installed SpacePy previously yielded the same configured fallback. Installing SpacePy alone does not activate AP9 or change this baseline. The existing fallback is retained; its exception is now reported once per model instance, and the mission summary distinguishes IRBEM import availability from AP9 support. No AP8-for-AP9 substitution was made.

Conda-forge `linux-64`/`noarch` search returned no `spacepy` package. PyPI provides SpacePy 0.7.0 and a CPython 3.7+ ABI3 manylinux x86-64 wheel compatible with CI Python 3.11. CI installs it using the activated interpreter's `python -m pip install spacepy==0.7.0`. [SpacePy release metadata](https://pypi.org/project/spacepy/) and [IRBEM radiation model documentation](https://prbem.github.io/IRBEM/api/radiation_models.html) support these findings.

Executed backend query: `irbempy.get_AEP8(10.0, [1.0, 1.5], model='min', fluxtype='diff', particles='p')`, yielding finite nonnegative `26673.225401696982`. This verifies bundled IRBEM/AP8 only. ANTARA's actual `get_ap9_flux(-20, -40, 600)` was also executed twice: both returned finite shape `(5,)` baseline `[35000, 18000, 6000, 900, 110]`, with one explanatory warning. AP9 remains unavailable due to an unsupported interface, not a demonstrated missing data download.

## 4. IGRF Findings

| Mapping | Verified value |
| --- | --- |
| Original source import | `import ppigrf` |
| Distribution selected | `ppigrf==2.1.0` |
| API used | `ppigrf.igrf(lon, lat, h, date)` |
| Inputs | Geodetic longitude/latitude in degrees, altitude in **km**, datetime |
| Outputs | East, North, Up arrays in nT; scalar query gives three shape `(1,)` arrays |
| ANTARA output | North, East, Down, shape `(3,)`, Tesla |
| Python compatibility | Package requires Python >=3.6; pure Python wheel; local 3.10 exercised, CI 3.11 compatible by metadata |
| Installation | Conda-forge `ppigrf=2.1.0`, or pip `ppigrf==2.1.0` |

No `pyIGRF14` alias is needed: the actual call is already a `ppigrf` API. The old `pyIGRF.igrf_value` comment and warning were stale. Existing private `_PYIGRF_AVAILABLE` state is retained for compatibility. Missing ppigrf still triggers the dipole fallback and an accurate warning.

The original code multiplied `alt_km` by 1000, evaluating a nominal 600 km spacecraft at 600,000 km. The call now passes kilometers directly. This changes the field to the intended altitude. Installed source uses `IGRF14.shc` by default; documentation now reflects that fact instead of claiming IGRF-13. Epoch handling remains the existing March 1 of the configured integer year.

Compatibility was checked against installed implementation and [ppigrf's published API example and package metadata](https://pypi.org/project/ppigrf/). At longitude 0°, latitude 0°, altitude 600 km, 2025-03-01, the backend returned finite ENU values `[-1636.28286097, 20589.30048612, 10037.32125285]` nT. A regression query at latitude 10°, longitude 20°, 600 km checks ANTARA's units, component ordering, sign, shape, and equality to the backend without asserting an unverified physical reference value.

## 5. Attitude Failure Root Cause

The committed formula is `degrees(acos(clip(dot(unit(a), unit(b)), -1, 1)))`. Near 1, tiny floating-point dot errors become apparent angles of order `sqrt(machine epsilon)` radians. This is a conditioning problem, not evidence of a wrong frame.

Full precision local reconstruction of the reported detection +Z case:

```text
axis   = [-7.672475468727613e-02, 6.143106511060716e-01, -7.853252421486348e-01]
target = [-7.672475468727619e-02, 6.143106511060715e-01, -7.853252421486346e-01]
raw norm(axis)   = 1.00000000000000000e+00
raw norm(target) = 9.99999999999999889e-01
normalized dot  = 1.00000000000000000e+00
normalized cross = [0, 5.551115123125783e-17, 4.857225732735060e-17]
cross norm       = 7.37614539776294124e-17
normalized difference = [6.938893903907228e-17, 0, 0]
acos angle       = 0.00000000000000000e+00 deg
atan2 angle      = 4.22622000366662418e-15 deg
```

The exact CI rounding failure did not reproduce for that particular vector on local NumPy. An independent **identical-vector** input did reproduce the same threshold violation:

```text
v = [0.6904853540677682, -0.42725264633653426, 0.15853969107671423]
a = b = v / norm(v)
a = [0.834610238845396, -0.5164330149879828, 0.1916316525129731]
norm(a) = norm(b) = 9.99999999999999889e-01
dot(a,b)         = 9.99999999999999778e-01
cross(a,b)       = [0, 0, 0]
cross norm       = 0.00000000000000000e+00
difference       = [0, 0, 0]
acos angle       = 1.2074182697257333e-06 deg  (FAIL at 1e-6)
atan2 angle      = 0.0000000000000000e+00 deg (PASS)
```

An identical vector cannot have a real alignment error. Together with the valid frame residuals, this demonstrates the numerical cause without loosening tolerance.

## 6. Attitude Implementation Change

Old committed formula:

```python
degrees(acos(clip(dot(unit(a), unit(b)), -1, 1)))
```

Final formula, completing the existing uncommitted repair:

```python
a = self._unit_numpy(first_vector)
b = self._unit_numpy(second_vector)
dot = np.clip(np.dot(a, b), -1.0, 1.0)
cross_norm = np.linalg.norm(np.cross(a, b))
return math.degrees(math.atan2(cross_norm, dot))
```

Normalization removes input scaling; the existing zero-vector rejection remains. Clipping bounds the cosine term. Cross magnitude preserves first-order information near parallel alignment and atan2 also handles antiparallel vectors. Scalar-first body-to-inertial quaternions and DCM construction are unchanged.

Trace: normalized primary target → orthogonally projected secondary → `build_frame` → quaternion from DCM → `_body_axis_to_inertial` → `_vector_angle_deg` → `_assert_axis_alignment`. `_run_geometric_self_tests` executes in simulator construction before propagation and restores ADCS state afterward.

## 7. Tolerance Justification

Old and chosen angular tolerance: **1e-6 degrees**, unchanged. These hardcoded startup checks test numerical frame construction, not mission pointing requirements. No ADCS accuracy requirement was changed.

Worst built-in alignment residual: `1.80620785248372971e-14` degrees. A seeded 1,000-frame construction sweep found worst residual `9.44256548883864e-13` degrees. The retained bound gives over six orders of magnitude margin above that sweep's residual while rejecting 0.01°, 0.1°, and 1° errors. All three negative cases raised `AssertionError`. Additional regressions cover identical vectors, a real `1e-8` degree angle, scaled vectors, and antiparallel vectors.

## 8. Geometric Self-Test Results

| Mode/Test | Axis | Error deg | Tolerance | PASS/FAIL |
| --- | --- | ---: | ---: | --- |
| Identity rotation | All | 0 | Existing rotation-consistency checks | PASS |
| 90° yaw | +X | 1.27222187258540671e-14 | 1e-6 deg | PASS |
| Detection primary | +X | 1.80620785248372971e-14 | 1e-6 deg | PASS |
| Detection secondary | +Z | 4.22622000366662418e-15 | 1e-6 deg | PASS |
| Relay primary | +Y | 7.11193647390761381e-15 | 1e-6 deg | PASS |
| Relay secondary | +Z | 3.89536803430295105e-15 | 1e-6 deg | PASS |
| Inertial/body round trip | Sample vector | 3.555968236953807e-15 | 1e-6 vector-norm residual | PASS |
| Quaternion/DCM round trip | Quaternion | N/A | 1e-6 quaternion-norm residual | PASS |

The round-trip vector residual was `1.1102230246251565e-16`; quaternion reconstruction residual was zero; identity determinant was 1. These last tests use dimensionless differences, not angular tolerances. There are no separate SAFE, sun-pointing, nadir, or RAM cases in the current startup method. Every actual case was exercised.

## 9. Scenario Override Investigation

Before: providing `--scenario` selected `build_scenario` and ignored duration, step, RNG, coarse step, periphery merge, and no-plots arguments. The warning compared against parser defaults, so explicitly passing a default value was indistinguishable from omitting it. A requested one-hour smoke case could therefore run 24 hours and generate plots.

After: **explicit CLI > scenario preset > global defaults**. Numeric arguments use `None` to represent omission; dataclass replacement applies explicit values and reruns configuration validation. `--no-plots` and `--no-periphery-merge` are honored with presets. Unmodified annual presets still select 8760 hours; explicitly requesting 24 hours now overrides that preset too.

Initial corrected-duration runs exited 0 but exposed default multirate stepping: 185 telemetry rows, times 0–3590 seconds, with mostly 20-second gaps. Final CLI behavior makes explicit `--step-sec 2` select both fine and coarse step 2 unless `--coarse-step-sec` is also supplied. Preset-only multirate behavior remains intact. This is documented in help and covered by tests.

Duplicate warnings came from executing the file as `__main__`, then `scenarios.py` importing it again as `ANTARA_Science_Sim`. Scenarios now imports the same `MissionConfig` directly from `antara_models`, eliminating that second simulator import.

## 10. Scenario Smoke Results

| Scenario | Requested Duration | Actual Duration | Requested Step | Actual Step | Rows | Exit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| leop_24h | 1 hour | 3600 seconds | 2 seconds | 2 seconds | 1801 | 0 |
| no_leop_24h | 1 hour | 3600 seconds | 2 seconds | 2 seconds | 1801 | 0 |

Both logs confirm `Duration=1.0 hr | Timestep=2.0 s (coarse=2.0 s)`. Pandas verified nonempty telemetry, the `mode` column, time range 0–3600 seconds, and every successive gap exactly 2 seconds. There are 1801 rows because both endpoints are logged. CSV telemetry has elapsed `time_sec`, not an absolute timestamp column; timestamps derived from the configured start epoch are **2000-03-01 00:00:00 UTC** through **2000-03-01 01:00:00 UTC** for both scenarios.

LEOP unique modes: `COLD_AND_DARK`, `DETUMBLE`, `SUN_POINTING`, `SLEWING`, `SAFE`. No-LEOP unique mode: `SUN_POINTING`. Neither run enters DETECTION, so these smoke runs do not validate science operation during an SAA pass; direct physics queries provide the backend execution checks. Both final logs contain no dependency warnings or tracebacks.

Logs: `/tmp/antara-leop_24h-physics-fixed.log` and `/tmp/antara-no_leop_24h-physics-fixed.log`. Telemetry: `results/science/leop_24h_ANTARA_Science_telemetry.csv` and `results/science/no_leop_24h_ANTARA_Science_telemetry.csv`.

Existing `leop_24h_*` and `no_leop_24h_*` science results were copied to `/tmp/antara-physics-backup.sMWWGj/` before running. New outputs are in the normal `results/science/` paths. No existing results were deleted; the prior copies remain recoverable from that backup directory for this session.

## 11. CI Changes

Every Orekit job (`smoke-test`, `unit-tests`, `scenario-smoke`, `nightly-full`, `weekly-validation`) now:

- Installs conda-forge `ppigrf=2.1.0` in the existing Micromamba creation block.
- Installs PyPI `spacepy==0.7.0` using `python -m pip` within the existing activated `bash -el` shell.
- Runs `scripts/verify_physics_dependencies.py --require-ci-env`, which prints interpreter/version/module paths, requires `/micromamba/envs/antara/` in the executable path, checks all three modules reside under that interpreter's prefix, and executes IGRF and IRBEM queries.

Using all five Orekit jobs keeps imported science modules consistent and lets unit regressions exercise real backends. The Graphviz-only job is unchanged. Unit CI also runs `tests/test_physics_attitude.py`; scenario output verification now checks approximately one hour of telemetry and exactly two-second gaps. No assertions, failure handling, job conditions, or timeouts were weakened. A terminating newline was added to the workflow.

## 12. Files Modified

| FILE | CHANGE | WHY |
| --- | --- | --- |
| `src/ANTARA_Science_Sim.py` | Finish existing atan2 edit, document unchanged tolerance, correct ppigrf altitude and labels, disclose AP9 fallback, apply CLI overrides | Numerical stability, backend correctness, honest fidelity status, intended CI run controls |
| `src/scenarios.py` | Import shared `MissionConfig` directly | Avoid duplicate simulator import and duplicate warnings |
| `.github/workflows/ci.yml` | Backend installations, interpreter/query checks, regressions, duration/step assertions | Reproducible physics environment and enforced smoke bounds |
| `requirements.txt` | Pin ppigrf 2.1.0 and SpacePy 0.7.0 | Document verified dependencies |
| `scripts/verify_physics_dependencies.py` | New executable backend verification | Fail early on missing/misplaced/unusable modules |
| `tests/test_physics_attitude.py` | New focused regressions | Valid and invalid geometry, IGRF units, scenario precedence |
| `OPENCODE_PHYSICS_ATTITUDE_FIX_REPORT.md` | This report | Evidence and developer handoff |

The pre-existing FSM edit remains in the aggregate diff and is not attributed to this repair. Existing missing terminal newlines were normalized in edited text files.

## 13. Verification Commands

All physics commands below used `/home/sammo/miniconda3/envs/orekit_env/bin/python`, abbreviated here as `$ANTARA_PY`, with `SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src` during sandbox verification. This abbreviation describes commands already executed; it does not refer to base Python.

| Commands / inspection | Result |
| --- | --- |
| `git status --short`, `git rev-parse HEAD`, `pwd`, `sed -n '1,260p' AGENTS.md` | Recorded initial state and read repository rules |
| `git diff -- src/ANTARA_Science_Sim.py` | Identified and preserved two existing edits |
| `rg` for IGRF variants, SpacePy/IRBEM/AP9/AE9, self-tests, CLI, dates, coarse stepping in source/tests/scripts/workflow | Actual imports, call sites, precedence, and tolerances identified |
| `cat .github/workflows/ci.yml`, `cat requirements.txt`, `cat src/scenarios.py`, targeted `sed` reads of simulator, paths, model config, validation test and installed backends | Reviewed source before changing it |
| `which python conda micromamba`, environment listing, inline `find_spec`/distribution metadata probe | Base shell Python identified; local orekit_env modules already installed; Micromamba absent |
| `conda search --override-channels -c conda-forge spacepy` | No matching package; initial sandbox network failure, retried with approval |
| `conda search --override-channels -c conda-forge ppigrf` | 2.1.0 available |
| `$ANTARA_PY -m pip index versions ppigrf` and `... spacepy` | Latest installed releases 2.1.0 and 0.7.0; network query retried with approval |
| Inline backend import/source inspection | First SpacePy import hit read-only home config; successful with supported `SPACEPY=/tmp` override |
| `$ANTARA_PY scripts/verify_physics_dependencies.py` | PASS: interpreter/module locality, IGRF query, compiled IRBEM query |
| Inline `import ANTARA_Science_Sim; print('Simulator import OK')` | PASS; no missing physics dependency warnings |
| Inline startup self-test probe with intercepted alignment checks | PASS; full numerical values recorded above |
| Inline seeded 1,000-frame sweep and identical-vector conditioning probe | PASS; numerical failure isolated without tolerance increase |
| Inline round-trip/identity and two actual `get_ap9_flux` queries | PASS geometry; explicit AP9 baseline fallback confirmed |
| `$ANTARA_PY -m pytest tests/test_physics_attitude.py tests/test_fsm_spec_matches_code.py -q --disable-warnings` | 14 passed; third-party JCC deprecation warnings summarized, not failures |
| `$ANTARA_PY src/ANTARA_Science_Sim.py --help` | PASS; `--no-plots` retained |
| `$ANTARA_PY src/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 0.002 --step-sec 2.0 --no-plots` | PASS, exit 0 before longer runs |
| Both `--scenario leop_24h` and `--scenario no_leop_24h`, with `--duration-hours 1 --step-sec 2.0 --no-plots` | Initial multirate runs exit 0; fixed-step reruns detailed in section 10 |
| Inline pandas telemetry inspection | Initial hidden 20-second cadence identified; final checks in section 10 |
| `$ANTARA_PY -m py_compile src/*.py scripts/verify_physics_dependencies.py tests/test_physics_attitude.py` | PASS |
| Inline PyYAML parse and checks of all five Orekit jobs | PASS: one dependency verification step per job, activated shell and ppigrf configured |
| `git diff --check`, `git diff --stat`, `git diff` | Whitespace check clean; changes reviewed |
| `mktemp -d /tmp/antara-physics-backup.XXXXXX`, `cp -p results/science/leop_24h_* results/science/no_leop_24h_* /tmp/antara-physics-backup.sMWWGj/` | Existing scenario outputs backed up |
| `head`, `tail`, `wc`, `rg` on run logs; process inspection | Monitored smoke execution; final logs retained under `/tmp/antara-*-physics-fixed.log` |

Local developer setup (no Micromamba requirement):

```bash
conda activate orekit_env
conda install -c conda-forge ppigrf=2.1.0
python -m pip install spacepy==0.7.0
python scripts/verify_physics_dependencies.py
PYTHONPATH=src python -c 'import ANTARA_Science_Sim; print("Simulator import OK")'
```

Alternatively, with Orekit already present, `python -m pip install ppigrf==2.1.0 spacepy==0.7.0` installs both verified distributions into the active environment. The verification script prints `sys.executable` and all three module paths. Do not install into base Conda or use `--user`.

## 14. Remaining Warnings

- Environment: local Python is 3.10, not CI's 3.11. Compatibility is supported by distribution metadata/wheels; actual remote CI execution remains unverified. The sandbox required writable SpacePy/Matplotlib configuration locations. No missing SpacePy or IGRF import warnings remain locally.
- Physics fidelity: AP9 is not supported by the current SpacePy call. Its existing configured baseline is retained and now disclosed when queried. No AP9 model output is claimed. Existing approximate IGRF epoch handling and NED-to-inertial transformation were not redesigned.
- Numerical: 3,448 Orekit/JCC `DeprecationWarning` messages were collected by pytest (Java extension types missing `__module__`); no attitude assertion failures. Geometry tolerance was not relaxed.
- CI: workflow syntax and local equivalent checks were verified, but no hosted workflow was dispatched. Long annual/nightly science fidelity was not validated by the one-hour smoke tests.

## 15. Git Diff

`git diff --stat` and the full tracked diff were reviewed. Changes are limited to the files in section 12. Ordinary `git diff` excludes the three new untracked report/test/script files until staged; their contents were inspected separately. The existing FSM change appears in the aggregate tracked diff. No commit or push was performed.

Tracked diff summary: 4 files changed, 105 insertions, 45 deletions. Three additional new files are the report, regression tests, and backend verification script. Final `git diff --check` passed. Generated science CSVs are ignored outputs and are not staged.

## 16. Final Assessment

**READY WITH NON-BLOCKING WARNINGS**

The requested dependency, geometric assertion, and CI smoke override repairs are implemented and locally verified. Both exact one-hour/two-second scenario commands exit 0; all 14 targeted tests pass. AP9 remains an explicitly disclosed configured-baseline fallback, and remote Python 3.11/GitHub Actions execution is not claimed. No commit or push was performed.
