# Simulation layout

Reorganized on 2026-09-28: source files and their module guides are **moved**,
not copied. Each module has one implementation under `src/simulations/`.
The old flat `src/*.py` commands and imports are removed; there are no wrappers.

## Structure

| Folder under `src/simulations/` | Contents |
| --- | --- |
| `ditl/` | Authoritative Science Sim, presets, scenario comparison, FSM tools and diagrams |
| `thermal/` | Thermal simulator and guide |
| `conops/` | Analytical CONOPS timeline and guide |
| `power/` | Power generation, battery sizing and trade studies |
| `detumble/` | Monte Carlo trials, result analysis and guide |
| `orbit/` | Drag and lifetime studies |
| `pointing/` | Pointing-budget study and guide |
| `helper/` | Shared configuration/models, paths, FSM specification, numerics, telemetry and trusted-pickle migration |
| `legacy/` | Superseded Master simulator and historical evidence; no new development |

Simulation-specific guides and repair records live beside their code. `docs/`
retains the documentation index, repository-wide reports and runner operations.
Data, results and figures stay at their existing repository-root locations.

## Commands and imports

Run from the repository root in Conda `orekit_env`:

```bash
python src/simulations/ditl/ANTARA_Science_Sim.py --help
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h --duration-hours 0.01 --step-sec 2 --no-plots
python src/simulations/detumble/leop_detumble_mc.py --help
PYTHONPATH=src python -m simulations.ditl.ANTARA_Science_Sim --help
```

Direct entrypoints bootstrap the `src/` import root. Python callers use qualified
imports, for example `from simulations.helper.antara_models import MissionConfig`.
Package initializers never launch studies. Some study modules execute work on
import: inspect their source instead of importing them just to check the layout.

Checkpoint and detumble analysis loaders map historical pickle module names to
relocated classes. Only load trusted files: this is not safe deserialization.
Checkpoint physics-version checks remain unchanged; relocation does not make
pre-version-7 histories compatible. External consumers of old trusted pickles
can use `simulations.helper.serialization.load_trusted_pickle`.

## Verification scope

Run `tests/test_simulation_layout.py` for source placement, direct CLI help,
repository-root routing, legacy pickle names and spawn/serial Monte Carlo parity.
Checkpoint and scientific regression suites remain required. This reorganization
changes paths and imports, not simulation physics. No annual rerun is authorized.

Local verification on 2026-09-28 (Conda `orekit_env`, Python 3.10.21):

- `PYTHONPATH=src python -m pytest tests/test_*.py -q`: 355 passed.
- Final layout suite after adding the documentation-placement guard: 11 passed.
- `python -m compileall -q src scripts tests` and `git diff --check`: passed.
- Local Markdown links: no broken targets in root reports, `docs/` or simulation guides.

Tests used temporary SpacePy/Matplotlib configuration directories. They exercised
short simulation intervals, checkpoint resume and process spawning, not annual
campaigns. Dependency warnings remain; hosted CI was not dispatched.

## Cleanup scope

The obsolete flat source files and former module-guide locations are removed;
their contents remain in the categorized folders and Git history. There is no
second root-level `simulations/` directory. CI commands, tests and documentation
links use the new locations. The diagram generator defaults to its DITL folder.

Unused imports were removed from active simulations and helpers after checking
their references. Public helper functions, physics fallbacks, legacy studies,
datasets, existing results and dated reports are retained: absence from a short
test run is not evidence that a scientific model or historical artifact is dead.
This cleanup is not a claim that every unreachable branch has been eliminated.

### Follow-up dead-code audit

On 2026-09-28, a repository reference search and inspection of dynamic attribute
lookups identified three uncalled Science private methods, now removed:

- `_classify_event_type`: unused energy-threshold classifier; live detector
  classification continues through `_build_bin_event_type_map`.
- `_merge_bins_if_needed`: unused index-list helper whose branches returned
  identical indices; the real count-merging path remains intact.
- `_rotate_vector_by_axis_angle`: unused remnant of predictive relay guidance;
  moving-LOS acquisition and tracking retain their current implementations.

The overwritten shared `EPSModel`, `SpacecraftFSM` and
`IllegalFSMTransitionError` imports were removed, not the local implementations.
Test/linter/coverage caches are now ignored; the redundant `detumble_*.pkl`
ignore rule was removed because `*.pkl` already covers it. No tracked backup,
reject or Python-bytecode files were found. No datasets, result files, archives,
historical reports or legacy Master code were deleted in this follow-up.

Follow-up verification: 156 tests passed across audit repairs, execution
integrity, checkpoint round trips, physics/attitude, scientific integrity and
layout suites. Compilation, local Markdown links and `git diff --check` passed.
The run reported 3,809 warnings; these were not suppressed in the implementation
or treated as proof of full physics validation. No annual simulation was run.

### Unused state and diagnostic cleanup

A further reference audit removed the permanently empty local
`relay_window_diagnostics` dictionary, its record-building branch and its
unreachable final print loop. No code ever populated its window keys. Active
relay telemetry, link gating, contact accounting and pointing checks remain.

Also removed: the unused cached `_spacepy_available` instance flag (the live
module-level availability report remains), an unused previous-illumination
local, an unused velocity-vector calculation, and the unused acceleration
calculation in detumble analysis `main`. Plot contour calls remain; only their
unused return-value bindings were removed. Checkpoint format and physics are
unchanged; restoring an older extra instance field remains harmless.

Verification: 76 tests passed across execution integrity, checkpoint round trips,
audit repairs, detumble multiprocessing and layout. The extended detumble
CLI regression also passed separately: three short trials were saved and loaded
by the analysis CLI, producing summary CSV and rate-only plot output in a
temporary directory. Compilation and diff whitespace checks passed. No annual
run, historical-output deletion, commit or push was performed.

### Dependency-manifest cleanup

Removed nine repeated dependency entries from `requirements.txt`, preserving
all eleven unique package/version constraints. Replaced obsolete source-build
instructions and the nonexistent `INSTALL.md` reference with the supported
Conda setup in `README.md`. Corrected the claim that Science automatically
downloads Orekit data; its default path helper requires prepared data.

`tests/test_repository_hygiene.py` checks unique distribution names and keeps
the Conda-provided Orekit wrapper out of the pip manifest. The CI regression
command includes this guard. No packages were installed or upgraded.

Verification: with the documented `PYTHONPATH=src`, 66 repository-hygiene,
runner-installer and scheduled-validation tests passed. An initial invocation
without that import path failed collection and was rerun with the correct
environment. `git diff --check` passed; remote CI was not run.
