# Execution, restart and payload checks — 2026-09-25

Historical batch record: the later [audit repair](../../../docs/AUDIT_REPAIRS_2026-09-25.md)
changes payload/ADCS physics and advances checkpoints to version 3. Version-2
results below describe this earlier batch, not the current restart format.

## Scope

Baseline: `3a822ac65fea5b8e5a42fd0387e79811c22a3706`; the working tree was clean.
This batch follows the [scientific-integrity repair plan](SCIENTIFIC_INTEGRITY_REPAIR.md).
The developer reports a successful annual run on September 24. No annual simulation
or workflow was launched here: verification is limited to short runs and controlled
subsystem tests. Runtime completion does not establish detector calibration or
pointing accuracy over every annual condition.

## Findings and changes

1. **Different integration paths:** `run_steps()` had a separate, reduced loop,
   including duplicate ADCS advancement and different science/telemetry behavior.
   It now calls the same production loop as `run()`, with a step-count bound.
2. **Horizon overrun:** a requested six-second run integrated eight seconds and
   reported 133.33% SUN_POINTING occupancy. Integration now stops at the horizon;
   the endpoint is a zero-duration observation, not another dynamics step.
   Calling `run()` again after completion does not append duplicate output.
3. **Restart state loss:** the old snapshot omitted pending commands, relay
   settlement, private payload counters/buffers, pass indices, and other runtime
   state. Version 2 captures Python subsystem state and RNG state while rebuilding
   Orekit runtime objects. Resume uses the saved clock and configuration.
4. **Unsafe snapshots:** the wall-clock checkpoint thread could race updates.
   Checkpoints now run synchronously at completed intervals, using simulated-time
   deadlines. Serialization precedes the atomic file replacement; a serialization
   failure preserves the previous checkpoint. Incomplete steps cannot be saved.
5. **Telemetry loss/corruption:** restarting reset the CSV writer. Version 2 records
   the flushed prefix length/hash, checks it on load, and copies only that prefix
   to a separate resume-tagged file. Later source rows are not included, and the
   original is not truncated. The cross-process CLI check also exposed a redundant
   final pandas rewrite that changed prefix formatting; that rewrite was removed.
6. **Payload precision:** all 64 bins previously received uncertainty computed from
   the pooled count. Each bin now uses its own cumulative expected count. The
   `max(N, 1)` floor remains a reporting convention, not an exact confidence interval.
7. **Histogram disagreement:** mission histograms were accumulated before periphery
   merging, but packets/pass records after merging. Both now use the same merged
   counters. Final pending packet flushes are included in terminal telemetry and
   storage totals, rather than occurring after the last recorded state.

No pointing tolerance, quaternion convention, mission requirement or propagator
was weakened. The radiation backend and Monte Carlo acceleration interpretation
were not guessed or replaced.

## Output and checkpoint compatibility

- `time_sec` identifies interval-start geometry/mode. `state_time_sec` identifies
  the end of that interval's integrated subsystem state. `interval_duration_sec`
  supplies the weight for integration/accounting. This makes the existing
  start-geometry/end-state convention explicit rather than claiming simultaneity.
- A final row has `terminal_snapshot=True`, duration zero, fresh orbital geometry,
  no new detector measurement or transmission, and any final packet-flush credit.
  Exclude it from detector sample statistics. Duration-weighted reporting remains
  a follow-up for existing consumers that still average rows equally.
- Keep the original telemetry CSV with its checkpoint. A missing, changed or
  truncated saved prefix is rejected. Default resume creates a unique output tag;
  an explicitly supplied tag must not already have a telemetry file.
- Only load trusted pickle files. Legacy unversioned pickle/JSON/NPZ checkpoints
  lack necessary state and are rejected; there is no scientifically defensible
  automatic reconstruction of the missing values.
- `--resume-from` uses the checkpoint configuration. Scenario, duration, timestep,
  seed and merge overrides are rejected; `--no-plots` is allowed. Checkpoint files
  are under `results/checkpoints/`, named by tag and elapsed simulation seconds.

## Verification evidence

Environment: local Conda `orekit_env`, Python 3.10.21; repository-root Orekit data.
Initial new regressions: **six failed**, demonstrating the old endpoint, state-loss,
merged-histogram and invalid-step-count behavior. Tests use isolated temporary
output paths and deterministic seeds; existing annual/scenario outputs are untouched.

Combined verification: **85 tests passed**; Python compilation and `git diff --check`
passed. There were 3,448 existing Orekit-wrapper deprecation warnings. The moving-LOS
test's worst reported error was **0.000001207°**, below the unchanged **5°** hard
limit; that idealized geometric check is not a flight-accuracy claim.

The standalone `python tests/test_checkpoint_roundtrip.py` launcher also passed
all **seven** checks after changing it to launch a fresh pytest process before
JVM initialization. An earlier in-process pytest launch crashed the native wrapper;
its diagnostic is retained at `/tmp/antara-execution-NXavrG/hs_err_standalone_bootstrap.log`.
The normal collected test run was unaffected. No simulation assertion was disabled.

Short production-loop runs:

| Scenario | Duration | Integrated time | Rows | Cadence | Observed modes |
| --- | ---: | ---: | ---: | ---: | --- |
| `leop_24h` override | 360 s | 360 s | 181 | 2 s | COLD_AND_DARK, DETUMBLE |
| `no_leop_24h` override | 360 s | 360 s | 181 | 2 s | SUN_POINTING |

The final observation in each run has zero duration. These were real constructor
and `run()` calls using CLI parsing/configuration, with only output paths redirected.
Logs/results are under `/tmp/antara-execution-NXavrG/` (temporary, not committed).

Additional checks:

- Six-second continuous versus segmented versus saved/restored runs compare clocks,
  state and complete telemetry. Cross-process CLI create/resume was also exercised
  at a 12-second horizon; all seven rows agree within `rtol=atol=1e-8`.
- Checkpoint tests cover serialization failure, prefix corruption, changed physics
  configuration, legacy rejection, incomplete-step rejection, continuing the original
  beyond the snapshot, and simulated-time scheduling.
- A controlled 60-second RELAY pass places a station beneath the initial spacecraft
  and follows its moving LOS using real Orekit frames. It checks quaternion norm,
  finite slew-rate bounds, and the existing pointing diagnostics. It is not a
  reproduction of the developer's unprovided annual-run anomaly.
- Synthetic detector inputs isolate count-time scaling, per-bin uncertainty,
  merged archive/packet consistency, and final-flush conservation. This is bookkeeping
  verification, not an independent absolute proton-flux reference.
- Existing negative alignment, stale SUN_POINTING target, saturation recovery,
  moving-target settlement, multiprocessing and scheduled-validation regressions
  remain in the suite.

Reproduce targeted checks from the repository root:

```bash
conda activate orekit_env
SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src python -m pytest \
  tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py \
  tests/test_physics_attitude.py tests/test_scientific_integrity.py \
  tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py \
  tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py \
  --disable-warnings --tb=short -q
python -m py_compile src/simulations/ditl/ANTARA_Science_Sim.py tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py
git diff --check
```

CI's existing unit command now explicitly includes the two execution/checkpoint
test files. No runner, dependency, schedule or annual time-budget change was made.

## Remaining limits

Long-duration restart tolerances, checkpoint memory/size and annual resource use
have not been established. Reports still contain some row-weighted averages.
Native detector rates/fluxes and merged packet bins need explicit spectral metadata
before a full measurement-product audit. The environment-versus-payload flux-table
disagreement and unsupported AP9 path remain open, as do Monte Carlo wheel-momentum,
acceleration semantics and failed-baseline bounds. The developer's specific annual
pointing/payload anomalies still need an example or the relevant telemetry interval.

No commit or push was performed.
