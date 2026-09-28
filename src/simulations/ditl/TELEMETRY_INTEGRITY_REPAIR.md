# Telemetry Time-Accounting Integrity

Date: 2026-09-27. Reporting-only repair; simulation physics and version-6
checkpoint state are unchanged.

## Defect

Scenario completeness previously checked timestamp endpoints and monotonicity,
but not agreement with interval metadata. For example, timestamps `[0, 86400]`
and durations `[2, 0]` could be accepted as a complete day while metrics used only
two seconds of weight. Streaming summaries also checked each chunk separately
without verifying continuity between chunks. Legacy missing timestamps could
be converted into zero-duration weights by `fillna(0)`.

## Changes

- `telemetry_metrics.interval_weights` rejects non-finite or non-increasing
  timestamps, mismatched durations and inconsistent `state_time_sec` endpoints.
- Streaming summaries check continuity across chunks and require the final
  zero-duration observation. Truncated files are rejected, not summarized as
  complete. Validation remains bounded-memory.
- Scenario comparison validates time accounting before publishing artifacts;
  `--require-full-duration` additionally checks terminal duration.
- Timing comparisons use an absolute `1e-7` second tolerance and zero relative
  tolerance, avoiding progressively looser checks at annual timestamps. This
  permits double-precision serialization noise, not physical timing drift.
- CI includes the new regressions and tightens smoke start/end, duration and
  end-state consistency assertions. Runner settings and schedules are unchanged.

Legacy telemetry without duration metadata still derives weights from adjacent
finite increasing timestamps. Fractional wheel-event intervals remain valid.
No row-count or exact fixed-cadence assumption was introduced.

## Verification

Local Conda Python: `/home/sammo/miniconda3/envs/orekit_env/bin/python`.
Environment: `SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src`.

```bash
python -m pytest tests/test_telemetry_integrity.py tests/test_wheel_events.py tests/test_slew_profile.py tests/test_audit_repairs.py tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py tests/test_physics_attitude.py tests/test_scientific_integrity.py tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py -q --disable-warnings --tb=short -p no:cacheprovider
```

Result: **243 passed**, 3,809 warnings, 82.82 seconds. Warnings were suppressed in
the summary output, not eliminated. The 17 new cases cover invalid legacy times,
metadata disagreement, chunk boundaries, truncated output, fractional intervals,
large timestamps and false-complete CLI input. Existing short simulation and
checkpoint regressions also passed.

The earlier one-hour smoke CSVs were rechecked with 127-row chunks: both retain
1,801 rows and exactly 3,600 seconds. This reuses existing evidence; it is not a
new simulation run. Python compilation and `git diff --check` passed.

## Limits

Consistent timestamps do not establish physical fidelity, absolute detector
calibration or annual completion. The comparison tool still loads complete CSVs
into memory; only the streaming summary has bounded working memory. Existing
artifacts from earlier runs are not deleted when a new comparison fails.
No annual run, hosted workflow, commit or push was performed.

## Follow-up: invalid measurements and coverage (2026-09-27)

Inspection found inconsistent scalar handling: `weighted_mean` could return an
infinite average, while the streaming summary silently removed the same infinite
measurement and reported a finite average. Missing samples were also removed
without an indication of how much simulation time the mean actually covered.

Both numeric averaging paths now reject infinities. Weighted means validate
finite nonnegative weights and normalize them before multiplication, avoiding
overflow from finite but large value-times-duration products. Streaming means
combine chunk means by their valid duration instead of accumulating those large
products. These changes affect reporting only, not simulation or checkpoints.

Each numeric JSON metric additionally exposes:

- `valid_duration_sec`: duration contributing finite values to the mean;
- `missing_duration_sec`: simulation duration excluded from that mean;
- `missing_rows`: numeric missing samples, including terminal observations.

An all-missing numeric metric is retained with null mean/min/max, not silently
omitted or reported as zero. Extrema still include finite zero-duration terminal
observations. These fields concern numeric scalar columns, not serialized
detector arrays; they are not a full CSV schema or calibration validator.

Regression cases cover both signs of infinity, negative/non-finite weights,
missing and entirely unavailable measurements across chunk sizes, and finite
`1e308` values whose unnormalized duration products would overflow. The earlier
one-hour CSVs were reprocessed with 127-row chunks: both passed, and battery
coverage was exactly 3,600 seconds with no missing samples. No new one-hour or
annual simulation was launched for this follow-up.

The same combined regression command above, including the nine new cases,
completed with **252 passed**, 3,809 warnings, in 95.84 seconds. Compilation and
`git diff --check` passed. Warning counts are recorded, not treated as resolved.

## Follow-up: chunk-independent scalar parsing (2026-09-27)

The next investigation reproduced a remaining hole: `select_dtypes` silently
excluded numeric columns when malformed text changed pandas' inferred dtype.
It also treated a numeric-looking cell identifier as a measurement in one chunk
but metadata in another, and inconsistently included nullable booleans.

Twelve new tests failed before the repair: nine malformed-value cases spanning
first/middle/final rows and three chunk sizes, plus three metadata/nullable-boolean
cases. All now pass. Science summaries declare their non-scalar fields in
`SCIENCE_NONSCALAR_COLUMNS`; other columns must parse as numeric scalars, with
nullable boolean representations normalized consistently. Errors identify the
offending column. Blank numeric measurements still use the coverage policy above.

This intentionally makes new undeclared text/vector fields an error rather than
silently omitting them. Add such fields to the explicit metadata declaration.
The parser does not validate detector array contents or physical calibration.

Targeted verification: **69 passed**, 3,809 warnings, 26.55 seconds for
`tests/test_telemetry_integrity.py` and `tests/test_audit_repairs.py`. Both prior
one-hour CSVs passed reprocessing with 127-row chunks (1,801 rows and 3,600 seconds
each), without treating cell IDs as numeric metrics. No simulation physics,
checkpoint format, annual schedule or runner configuration changed.

The full combined regression command listed above finished with **264 passed**,
3,809 warnings, in 90.30 seconds. Compilation and `git diff --check` passed.
No annual run, commit or push was performed.

## Next three repairs: comparison input integrity (2026-09-27)

1. **Empty optional events crashed comparison.** Science emits a headerless CSV
   when an event list is empty; pandas raised `EmptyDataError` when comparison
   loaded it. This occurred in both existing one-hour SAA-pass outputs. The reader
   now accepts only missing/headerless optional event logs as zero events. Empty
   telemetry and malformed event CSVs still fail.
2. **Mode grouping silently lost simulation time.** Missing labels disappeared
   in pandas grouping, while unknown labels were omitted from the fixed duty
   dictionary. Both comparison and streaming summaries now require recognized
   Science mode labels on every row. The FSM itself is unchanged.
3. **String Boolean truthiness produced false saturation counts.** `astype(bool)`
   interpreted both `"False"` and `"True"` as true. An explicit parser now supports
   Boolean and 0/1 representations and rejects other values. Comparison rejects
   missing flags; streaming summaries retain missing flags as uncovered duration.
   The same parser handles eclipse fraction and declared Science Boolean scalars.

The initial 11 regression cases all failed before implementation. New guards
also verify that malformed event data is not hidden, supported Boolean encodings
agree, valid duties sum to 100%, and invalid modes/Boolean scalars fail in the
streaming path. No dependency, dynamics, checkpoint or mission-limit changes
were required. Existing CI already collects both modified test modules.

The actual comparison entrypoint was run with the two prior one-hour outputs and
a fresh temporary output directory. It exited successfully, produced a CSV and
figure, and reported both durations as one hour with duty totals of 100%.
Artifacts: `/tmp/antara-comparison-three-fixes-g8ou2pd4/`. The absent annual file
was reported as missing (the command did not request `--require-all`); no annual
completion is implied. Existing result files were not overwritten.

Full combined verification (the regression command above): **285 passed**,
3,809 warnings, 94.25 seconds. This includes 21 added cases for the three repairs.
Compilation and `git diff --check` passed. No annual run, commit or push occurred.

## Follow-up: incomplete byte counters (2026-09-27)

Comparison previously initialized absent science/downlink counters to zero and
used pandas' missing-skipping sum. Consequently incomplete measurements could
appear to establish zero science or a meaningful downlink efficiency. Negative
and infinite values were also accepted.

Counter columns now require finite nonnegative numeric samples. Absent columns
remain unavailable rather than fabricated zero; present incomplete columns raise
an error. Byte increments are still summed without duration weighting, including
zero-duration terminal histogram flushes. An efficiency requires both totals and
strictly positive generated bytes; otherwise it is unavailable. CSV serialization
represents unavailable metrics as blank fields. Finite totals whose accumulation
overflows are rejected, and efficiency divides before multiplying by 100 to avoid
avoidable intermediate overflow.

The initial ten regression cases failed before implementation. Further cases
cover absent downlink data, numeric-string counters, sum overflow and large equal
totals. This repairs comparison reporting only; it does not change storage,
payload acquisition, downlink dynamics or checkpoint format. The existing
`science_downlinked_mb` field name is retained; attribution of transmitted bytes
to preloaded versus newly acquired data is not independently established here.

Verification completed 2026-09-28: the full combined regression command above
passed **298 tests**, with 3,809 warnings, in 94.91 seconds. Compilation and
`git diff --check` passed. The real comparison entrypoint also succeeded on both
existing one-hour CSVs: zero generated bytes remained zero, while efficiency
was unavailable. CSV/figure artifacts are under
`/tmp/antara-byte-counter-check-0srxco1y/`. This reused telemetry, not a new
simulation. No annual run, commit or push was performed.

## Follow-up: bounded-memory scenario comparison (2026-09-28)

The comparison CLI previously loaded full telemetry and three unused event
tables. Annual telemetry also includes serialized detector arrays, so this
post-processing step could exhaust memory despite bounded simulator summaries.

The CLI now selects only ten relevant scalar columns, reads 10,000 rows at a
time, and retains one lookahead row. This preserves inferred intervals in legacy
timestamp-only CSVs and checks current interval metadata across chunk boundaries.
Aggregates retain duration-weighted means, extrema, per-mode seconds/errors,
byte totals including terminal flushes, and the previous saturation flag so a
continuous hold is not recounted at each chunk. Preset completeness checks still
require the configured full duration; no annual duration was shortened.

The CLI no longer reads optional event tables because none of its metrics use
them. The existing `load_scenario_data` API is retained for small full-frame
callers, including its empty-event/error behavior. This supersedes the earlier
whole-file CLI limitation recorded above, not the remaining unbounded simulator
event-history limitation. Missing timestamps now fail instead of fabricating a
24-hour duration. Chunked means can differ by floating-point roundoff, not by
sampling or decimation.

Tests compare every metric against the prior full-frame helpers at chunk sizes
1, 2, 3 and 20, using current and legacy telemetry, multirate/fractional intervals,
missing battery samples, cross-boundary holds, and terminal byte flushes. A reader
guard verifies `chunksize` and selected-column use. Negative cases cover corrupted
interval boundaries, missing timestamps and incomplete annual endpoints; the
annual check uses a tiny synthetic CSV, not an annual simulation.

Verification: **311 passed**, 3,809 warnings, 83.52 seconds with the combined
regression command above. The final scheduled-validation module was rerun after
adding streaming assertions for flat, nested and legacy layouts. Both existing
one-hour CSVs also matched every prior comparison metric at `1e-12` relative and
absolute tolerance using 127-row chunks; absent per-mode columns were compared
as unavailable. The real CLI generated CSV/PNG outputs at
`/tmp/antara-streaming-comparison-3c70q9hr/`. Compilation and `git diff --check`
passed. This is structural bounded-memory verification and short-data parity,
not an annual memory benchmark or a full mission validation. No annual run,
commit or push was performed.

## Follow-up: unfinished bounded comparisons (2026-09-28)

The streaming comparison accepted a positive-duration final row unless
`--require-full-duration` was requested. For timestamps `[0, 2]` and durations
`[2, 2]`, it reported a two-second span while weighting four seconds of subsystem
activity. A truncated current-format file could therefore publish misleading
bounded-study metrics.

The final interval must now have zero duration for every comparison. Bounded
completed studies remain supported; full-preset endpoint checks remain a separate
option. Legacy timestamp-only files infer a zero-duration last observation as
before, so they cannot provide the same completion evidence as explicit metadata.

Four regression cases failed before repair and pass afterward: chunk sizes 1,
2 and 20, plus a CLI test checking failure before artifact publication. Targeted
verification ran `tests/test_scheduled_validation.py` and
`tests/test_telemetry_integrity.py`: **118 passed**, 3,448 warnings, 4.14 seconds.
Compilation and `git diff --check` passed. The broader simulation suite was not
rerun for this reporting-only two-line guard. No simulation physics, checkpoints,
annual scheduling, commit or push was changed.

## Follow-up: recent-history bias in plots (2026-09-28)

The display sampler repeatedly merged new chunks with an already reduced sample
and reduced it again. Early rows were thinned repeatedly while the latest rows
were thinned only once. A 101-row/10-sample regression selected row 66 immediately
after row 0, instead of row 11. This affected visual coverage, not unsampled
summary statistics or simulation physics.

The sampler now counts CSV records using a one-column bounded pass, then selects
globally spaced row indices in a second bounded pass. Both endpoints are retained,
files within the limit are unchanged, and quoted multiline records are parsed
correctly. Limits must be integers of at least two. A row-count change between
passes is rejected; callers must otherwise provide a completed unchanged file.

The initial tests recorded seven failures and three passes before implementation,
including two demonstrated sampling biases and invalid-limit handling. Added
tests cover empty/single-row schema preservation and enforce two chunked reads.
This costs another sequential read but bounds working memory by the sample limit.
Uniform row sampling is not uniform elapsed-time sampling for multirate runs;
brief events and extrema can still be omitted. Figures remain display-only.

Verification: the combined regression command recorded **324 passed**, 3,809
warnings, 99.14 seconds. Both prior one-hour CSVs were sampled to 50 rows and
matched the expected global row indices, retaining 0/3600-second endpoints.
Compilation and `git diff --check` passed. No annual simulation, commit or push
was performed; existing telemetry was reused without modification.
