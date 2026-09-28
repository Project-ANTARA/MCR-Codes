# Short operational verification and repairs

Date: 2026-09-28. Scope: short Science operational runs, Master/Science EPS
interval accounting, and regression checks. No annual run or CI dispatch.

## Defects reproduced and corrected

1. **Moving relay acquisition could stall at finer timesteps.** In a real
   120-second overhead-station case, 4- and 2-second steps reached RELAY, but
   1-second steps stayed in SLEWING with zero downlink. At the end, pointing
   error was only `1.4688877e-5 deg`, tracking had remained valid for 119 seconds,
   and angular rate was `0.00542962 deg/s`. The completion guard incorrectly
   required the stationary-target `is_slewing` flag to clear. Science now permits
   moving acquisition after continuous valid tracking for the configured settling
   duration. Quaternion error checks and saturation rejection remain; angular
   rate is preserved. Invalid tracking also clears previously earned settled time.
2. **Endpoint slew flags mispriced interval energy.** A slew that stopped during
   an interval could receive idle power for that entire interval. Science and
   Master now use actual motion duration divided by actual integration duration
   to interpolate idle/slew ADCS power. Top-level mode loads are unchanged.
   The optional EPS duty argument validates finite values in `[0, 1]`; older
   Boolean callers retain their previous semantics.
3. **Sun-side frame reversal violated the nadir requirement.** A real SAA orbit
   (`true_anomaly_deg=200`, `raan_deg=291.8613431778403`) crashed immediately with
   a `179.980385 deg` SUN_POINTING error. For negative Sun/orbit-normal dot product,
   Science reversed +Z while fixing +X, necessarily reversing +Y. The nominal
   frame now always locks +X to RAM and +Y to projected nadir. Its right-handed
   +Z is the positive orbit normal, which need not face the Sun. Quaternions stay
   scalar-first, body-to-inertial; pointing limits are unchanged.

Checkpoint version **7** rejects version 6 and earlier, avoiding mixed pre/post
repair attitude and energy histories. Historical output files are not rewritten.

## Short operational evidence

Relay tests use actual Orekit propagation, a station at the initial subsatellite
point, production FSM/guidance/EPS/link models and an explicit 100 MB science
backlog. Only output routing is redirected. Tracking noise is disabled for the
step comparison to avoid comparing different random sample sequences.

| Step (s) | Final battery (Wh) | Downlink (MB) | RELAY duration (s) |
| ---: | ---: | ---: | ---: |
| 4 | 23.847597403 | 0.115371840 | 60 |
| 2 | 23.818767612 | 0.114949104 | 60 |
| 1 | 23.808114244 | 0.114737916 | 60 |

All runs account for exactly 120 seconds and conserve backlog plus transmitted
bytes. Successive battery differences decrease from 0.028830 to 0.010653 Wh;
downlink differences decrease from 0.000423 to 0.000211 MB. Screening budgets are
0.1 Wh battery spread and eight seconds of maximum link throughput, with an
additional decreasing-difference check. These are regression criteria, not a
proof of asymptotic convergence, mission accuracy or absolute energy calibration.

The separate real 120-second SAA run, at two-second steps, reaches DETECTION for
60 seconds and produces positive detector counts. Telemetry, storage plus
downlink, SAA pass packet credit and final histogram flushing agree. It uses the
configured proton-flux baseline and explicitly warns that live AP9 is unavailable.
It tests horizon closure inside the SAA, not a natural SAA exit.

## Verification and historical follow-up work

Regressions are in `tests/test_execution_integrity.py`,
`tests/test_physics_attitude.py` and `tests/test_checkpoint_roundtrip.py`, already
included by the existing CI unit command. Negative checks cover incomplete
settlement, saturation, wrong alignment, invalid duty fractions and old snapshots.
Master's attitude validation is active; contrary descriptions in its guide and
interval report were corrected against the source.

Combined verification: **340 passed**, 3,809 warnings, 71.99 seconds under Conda
`orekit_env` with `SPACEPY=/tmp`, `MPLCONFIGDIR=/tmp/antara-mpl`, `PYTHONPATH=src`:

```bash
python -m pytest tests/test_telemetry_integrity.py tests/test_wheel_events.py tests/test_slew_profile.py tests/test_audit_repairs.py tests/test_execution_integrity.py tests/test_checkpoint_roundtrip.py tests/test_physics_attitude.py tests/test_scientific_integrity.py tests/test_detumble_multiprocessing.py tests/test_scheduled_validation.py tests/test_fsm_spec_matches_code.py tests/test_runner_installer.py -q --disable-warnings --tb=short -p no:cacheprovider
```

The initial relay and SAA reproductions failed as described above before their
fixes; they were not skipped or weakened. Warnings were untriaged at this stage;
see the final cleanup verification record for the later classification.

At this initial stage, station handovers and natural SAA exits still needed
integrated checks; the follow-up below closes those gaps. Broader convergence,
bounded archives, calibration and CI acceptance remain separate work. Master
acquisition is no longer a repair target: Master is superseded. See the
[current backlog](../../../docs/CURRENT_STATUS.md).

## Follow-up: natural exit and two-station handover

The SAA test now also runs for 400 simulated seconds (201 rows). It leaves the
actual polygon before the horizon, transitions away from DETECTION, and closes
exactly one pass. Generated packet totals agree across telemetry, storage plus
downlink and pass credit; no pending histogram remains. No polygon, environment,
guidance or FSM method is replaced.

A 240-second two-station test places stations on the actual ground track at
epochs 0 and 180 seconds. Before the repair, switching the highest-elevation
station crashed in RELAY with an `87.669811 deg` pointing error. The spacecraft
was still aimed at the previous station; choosing a new link did not initiate
acquisition. Science now records the selected station as a serializable label,
resets tracking settlement when that selection changes, and initiates a finite
RELAY-to-SLEWING-to-RELAY handover. A selection change during acquisition cannot
complete against the previous station's stale attitude command.

The test checks zero transmission on the handover interval, valid settlement
and pointing for every transmitted increment, positive credit at both stations,
and exact reconciliation of per-station credit with total downlink and remaining
backlog. It also saves at 80 seconds and compares resumed telemetry with the
uninterrupted run. Output routing and a contact-recording observer are the only
patched methods; the observer delegates unchanged to production accounting.

The selection changes at 90 seconds. Final station credits are
`0.06066814896826224 MB` and `0.2092579006060083 MB`; their sum is the total
downlink. The run spends 120 seconds in RELAY and 118 seconds in SLEWING.

Restart comparison also exposed `acos(dot)` conditioning in operational angles:
one nearly aligned relay sample differed by `8.537736e-7 deg` despite otherwise
matching trajectories. Operational relay, antenna off-boresight, payload and panel
incidence angles now use the existing normalized cross/dot `atan2` helper.
The full-frame restart comparison passes at unchanged `rtol=atol=1e-8`.
Direct angle regressions cover `1e-8`, `0.01` and `179.99` degrees, plus the
invisible-station sentinel. This changes angle evaluation, not pointing limits.

Follow-up combined suite: **345 passed**, 3,809 warnings, 101.36 seconds with
the same command/environment listed above. Compilation and `git diff --check`
also passed. Warnings were untriaged in that run; no hosted workflow was dispatched.

This closes these two specific operational coverage gaps, not every possible
station geometry, visibility loss or natural boundary crossing. Bounded archives,
broader convergence/reference studies and CI acceptance remain open. Master is
excluded from further development. No pointing thresholds or assertions were relaxed.
