# Current repair status

Reconciled 2026-09-28 against the reorganized working tree. This is the current
backlog; dated reports and the old implementation plan preserve historical
proposals and results, not a list of still-open defects.

## Implemented and regression-covered

- Physics dependency setup and backend checks; stable `atan2` alignment angles.
- Explicit CLI overrides over scenario defaults and finite attitude acquisition.
- Portable detumble workers, seeded ordering, progress and rate-only semantics.
- Shared Science run/step integration, exact horizon accounting and version-7
  atomic checkpoints with uninterrupted/resumed comparisons.
- Environment-to-detector flux propagation, per-bin uncertainty, histogram
  merging and pass-boundary packet accounting.
- Elapsed-time-weighted summaries, bounded scalar CSV comparison, coverage and
  malformed/unfinished telemetry checks. The legacy full-frame helper remains
  for small callers; the comparison CLI streams.
- Event-resolved slews/wheel limits, torque sign/frame and interval energy fixes.
- Natural SAA exit and two-station relay handover/restart regressions, including
  settlement and transmission gating. These are no longer open coverage gaps.
- Single-copy categorized source layout, colocated guides, trusted old-pickle
  module migration and verified dead-code/dependency cleanup.

Evidence: [operational checks](../src/simulations/ditl/SHORT_OPERATIONAL_VERIFICATION.md),
[execution/payload](../src/simulations/ditl/EXECUTION_PAYLOAD_REPAIR.md),
[telemetry](../src/simulations/ditl/TELEMETRY_INTEGRITY_REPAIR.md),
[audit repairs](AUDIT_REPAIRS_2026-09-25.md), and [layout](SIMULATION_LAYOUT.md).
Passing regressions does not establish exhaustive scientific validation.

## Still open

1. Bound event/pass histories and checkpoint size for long campaigns; streamed
   scalar telemetry does not bound every in-memory history.
2. Broader step-size convergence and independent subsystem reference studies,
   particularly thermal/drag assumptions and absolute payload calibration.
3. Physical requirements for a real AP9 integration and higher-fidelity ADCS.
   SpacePy imports/AP8 checks do not enable live AP9; the configured baseline and
   approximate attitude model remain explicit limitations, not hidden fixes.
4. Verify the service-account physics environment and hosted/self-hosted CI
   execution after relocation. No workflow was dispatched during this cleanup.
5. Independently verify full-year nightly completion within its runtime budget.
   The developer reports an annual run succeeded; no annual rerun was authorized.

Master is superseded and excluded from further acquisition/parity development.
Its historical reports and existing short regression checks are retained.
No commit or push has been performed for this reorganization.
