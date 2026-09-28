# Simulation Runtime and Monte Carlo Repair

## Documentation and progress update (2026-09-12)

Current commands: [README](../../../README.md), [Science guide](ANTARA_Science_Sim.md), [detumble runner/analysis guide](../detumble/leop_detumble_mc.md), and [scenario output guide](scenarios.md). Historical verification counts below retain their original scope.

The Monte Carlo runner now displays parent-only terminal progress on stderr: completed/total, success/timeout counts, rate-cap hits, elapsed time, and estimated time remaining. Baseline/serial execution additionally reports phase, simulated seconds, and body-rate magnitude. Redirected logs are throttled plain text; `--no-progress` retains setup and summary but suppresses the display. Narrow terminals are redrawn across wrapped lines. No new dependency or physics change was introduced.

A normal eight-trial, two-worker run with the new display exited 0, retaining the 1009-second baseline, 3027-second trial timeout, three successes, five timeouts, and three rate-cap hits. Progress regression tests cover interactive/redirected output, counters/ETA, narrow-terminal cleanup, and quiet baseline-only runs, alongside spawn equivalence and portable pickles.

During the documentation refresh, the combined command `PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -q --disable-warnings --tb=short` passed **31 tests** using the local Orekit environment. CLI help for the detumble runner and analyzer, local documentation links, code-fence pairing, and `git diff --check` were also checked. No simulation source or historical subsystem PASS/FAIL data was changed by the documentation refresh.

## Problems and changes

### Annual-run SUN_POINTING failure (follow-up, 2026-09-12)

The annual run reported a 5.097451-degree combined RAM/nadir error near 58 hours. A valid rotation matrix does not prove alignment with the current orbital frame. This error is physically meaningful, unlike the earlier microdegree `acos` conditioning issue.

**Actual reproduced cause:** an initial 60-hour verification attempt reproduced the exact quaternion and 5.097451-degree failure even after adding the stale-target guard below. A focused orbital-window replay exposed `saturation_hold_active=True`, `saturation_recovery_active=True`, `is_slewing=False`, and `settling=False`. Wheel saturation had deliberately replaced the slew target with the achieved inertial attitude. The completion check mistook convergence to this frozen target for completion of the original maneuver, and entered SUN_POINTING during unloading. The nominal pointing assertion then treated intentional off-nominal drift as a tracking implementation failure.

Slew completion now rejects active saturation holds and requires convergence to the pending command as well as the ADCS target. On release, SLEWING restores a pending objective overwritten by saturation. Diagnostics retain rotation and telemetry checks during unloading and assert actual alignment with the frozen target; nominal RAM/LOS limits resume when tracking is enabled. Science generation and transmission are explicitly blocked during saturation hold, so an off-nominal hold cannot be counted as valid observing or downlink.

The first focused correction avoided the assertion but exposed a recovery trap: the local Science FSM's slew timer ran before fault overrides, while SLEWING supplied error zero and limit infinity. The existing 300-second saturation timeout therefore could not request SAFE for a frozen, diverging maneuver. During saturation hold only, the simulator now supplies the pending mode's pointing budget and the local FSM evaluates its existing fault policy before the slew timer. Normal moving slews retain their unbounded acquisition budget. No timeout, debounce count, or recovery threshold changed; shared FSM implementations were not rewritten.

Source inspection found a 7.5-degree SUN_POINTING target-update deadband with a 5-degree hard limit. Recovery inspected only the achieved attitude **before** advancing; the retained target could already be outside the limit. Moreover, an unchanged requested command could bypass recovery even though ADCS had previously rejected that command. A deterministic regression reproduced the failure with actual quaternions: current error 4.9 degrees, retained target error 5.1 degrees, desired error zero. The old advance snapped the remaining 0.2 degrees to the stale target and declared it settled, triggering the assertion at both 2- and 20-second intervals.

This additional guard evaluates the retained target's +X/RAM and +Y/nadir alignment against current orbital geometry, using normalized cross/dot `atan2`. An out-of-budget destination forces a fresh command and bypasses the unchanged-request shortcut. The shortcut also cannot bypass recovery when the achieved attitude already exceeds its mode limit. Normal in-budget deadband behavior is retained. No hard limit, quaternion convention, or slew rate was relaxed. This guard alone did **not** repair the saturation-triggered annual failure.

Regression coverage exercises changed and repeated requests at both cadences. A deliberately settled 5.1-degree attitude still raises the original diagnostic. This is a reproduction of the failure mechanism, not a reconstruction of every internal state from the user's rounded log.

The follow-up targeted suite passed **27 tests**, including the existing RELAY, numerical geometry, multiprocessing, and timestep regressions. New checks cover frozen slew completion rejection, pending-objective restoration, the saturation timeout path into SAFE, changed/repeated stale targets at both cadences, and negative nominal/hold/telemetry assertions. The earlier 20-test and 24-hour results below remain historical evidence from the preceding repair.

The corrected two-hour orbital-window replay (original orbital epoch shifted by 57.8 hours; fresh skip-LEOP state, not a checkpoint) exited 0 with 960 telemetry rows spanning 0–7200 seconds. Saturation hold covered 413 rows, from 956–1780 seconds. The FSM entered SAFE at 1258 seconds for saturation timeout and returned to SUN_POINTING at 1782 seconds after unloading. The final hold flag was false and SUN_POINTING error was 0.009333 degrees. All held rows had science generation and relay-link flags false, with zero generated/downlinked data. Outputs use `sun_recovery_final_window`; the log is `/tmp/antara-sun-recovery-window.log`.

The full-startup verification uses this setup, retaining the annual preset's 2-second fine/20-second coarse cadence. Its diagnostic observer additionally records the maximum settled, non-held SUN_POINTING error without changing control behavior:

```python
from simulations.ditl.ANTARA_Science_Sim import ANTARAScienceSimulator
from simulations.ditl.scenarios import build_scenario

config, skip = build_scenario("leop_1yr")
config.sim_duration_hours = 60.0
config.no_plots = True
config.output_tag = "sun_recovery_final_60h"  # Choose a new tag before rerunning.
sim = ANTARAScienceSimulator(config, skip_leop=skip)
sim.run()
```

Run from the repository root in `orekit_env` with `PYTHONPATH=src`. Sandboxed verification also sets `SPACEPY=/tmp` and `MPLCONFIGDIR=/tmp/antara-mpl`. The normal annual CLI remains `python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_1yr --no-plots`; a 60-hour check is not a completed 8,760-hour validation.

**Final full-startup result: PASS, exit 0.** The run reproduced the supplied pre-failure history and continued through 60 hours. Log: `/tmp/antara-sun-recovery-60h.log`; CSV prefix: `results/science/sun_recovery_final_60h_ANTARA_Science_`.

| Check | Result |
| --- | --- |
| Telemetry | 17,758 rows, 0–216000 seconds; 2- and 20-second intervals |
| Modes | All seven: COLD_AND_DARK, DETUMBLE, SUN_POINTING, SLEWING, DETECTION, RELAY, SAFE |
| Maximum settled, non-held SUN_POINTING error after transition grace | 4.999972891972313 degrees; unchanged limit 5 degrees |
| Saturation hold | 403 samples, 209044–209848 seconds |
| Timeout transition to SAFE | 209346 seconds (58.1517 hours), 302 seconds after hold entry |
| Recovery to SUN_POINTING | 209850 seconds (58.2917 hours) |
| Science and transmission during hold | Both disabled; zero generated/downlinked data in every held row |
| Final state | SUN_POINTING, hold false, combined error 0.144439 degrees |
| Mission science generated / downlinked | 0.023087 / 0.023087 MB |

The worst-error measurement explicitly excludes active maneuvers, settling, saturation hold, and transition grace. Off-nominal pointing drift remains visible in telemetry and drives fault recovery; it is not relabeled as valid tracking. The 27-test suite, changed-file compilation, and `git diff --check` passed. Read the wide telemetry CSV with `pd.read_csv(path, low_memory=False)` to avoid pandas' chunked mixed-type warning for optional science fields.

This follow-up modified `src/simulations/ditl/ANTARA_Science_Sim.py`, `tests/test_physics_attitude.py`, `AGENTS.md`, `docs/ANTARA_Science_Sim.md`, and this report. Existing unrelated worktree changes were preserved. Starting HEAD was `3ace53f0a1bf3dd95a24228c231f53ff50143d09`. No commit or push was performed. The unavailable AP9 baseline warning and intermittent pre-accumulation zero-science warning remain separate from attitude recovery. **Assessment: targeted repair verified, with known physics-fidelity warnings; full-year validation remains outstanding.**

### RELAY tracking failure

The reported 6.005910-degree error was an actual mismatch between the antenna and moving ground-station line of sight, not the earlier small-angle floating-point failure. The source held a one-time predicted LOS throughout acquisition, completed slews from elapsed time, and reset commanded slew speed when targets changed. The prediction used an approximate range rather than continuously following the propagated station geometry. Refreshing the target also repeatedly restarted fixed-target settling.

The repair:

- Commands the current station LOS and refreshes it during RELAY acquisition and tracking.
- Preserves slew speed when updating an ongoing maneuver; configured acceleration/rate bounds remain in use.
- Requires actual attitude convergence and settling before finishing SLEWING; the timer remains a minimum maneuver duration, not proof of alignment.
- Counts consecutive time within configured pointing accuracy when settling on a moving LOS. A new sample no longer automatically discards valid tracking time.
- Resets transition grace bookkeeping on actual slew completion.

No pointing limit, beamwidth, quaternion convention, or alignment assertion was relaxed or removed. The configured relay validity limit remains 5 degrees in this revision.

### Multirate startup and battery accounting

The old coarse path advanced the clock by 20 seconds while passing only 2 seconds into detumble, guidance, and other subsystem updates. It then added another 20 seconds of EPS integration after the existing 2-second battery step. This slowed detumble relative to mission time and distorted the energy history.

Startup and active tracking now use the fine cadence. The selected interval is passed consistently to guidance, EPS, payload, mode accounting, and the clock; the second EPS update was removed. Coarse stepping remains available outside startup/active modes. A regression instruments actual EPS/guidance calls and checks them against telemetry time differences.

### Monte Carlo serialization

`run_monte_carlo` defined a nested `worker`, shadowing an already existing module-level worker. Process pools pickle submitted callables, and a local closure cannot be imported by a child process. The module-level worker also expected configuration, timestep, and timeout fields absent from the original task tuples.

The repair removes the nested worker and supplies explicit serializable arguments to the module-level worker. Results are assembled in seeded trial order so serial and parallel execution agree; the baseline remains included in the requested count. Invalid zero/negative trial counts are rejected.

A second serialization problem was reproduced after fixing dispatch: CLI results referenced `__main__.DetumbleTrialResult` and could not be loaded in a fresh analysis process. The CLI now invokes the importable module's runner, giving saved result classes a stable module name. Existing old pickles are not rewritten.

## Verification

The local interpreter is `/home/sammo/miniconda3/envs/orekit_env/bin/python` (Python 3.10). Sandboxed checks use `SPACEPY=/tmp MPLCONFIGDIR=/tmp/antara-mpl PYTHONPATH=src`. Local Micromamba is not required.

```bash
PYTHONPATH=src python -m pytest tests/test_physics_attitude.py tests/test_detumble_multiprocessing.py tests/test_fsm_spec_matches_code.py -q
python src/simulations/detumble/leop_detumble_mc.py --n-trials 8 --workers 4 --output /tmp/antara-mc-final-normal.pkl
python src/simulations/detumble/leop_detumble_mc.py --n-trials 5000 --workers 4 --timeout-sec 2 --output /tmp/antara-mc-final-5000.pkl
python -m py_compile src/*.py
```

The 5,000-task run tests dispatch, collection, ordering, and serialization with a deliberately short timeout; it is **not a completed 5,000-trial scientific capability study**. The eight-trial run uses the normal baseline-derived timeout (1009-second baseline, 3027-second trial timeout). Tests also explicitly exercise the multiprocessing **spawn** context and load a CLI-generated pickle in a separate process.

A focused contact replay at the original mission epoch around 9.6 hours acquired RELAY after 60 seconds and produced 150 visible-RELAY samples with maximum LOS error about `1.21e-6` degrees. Its initially empty science storage correctly produced no downlink.

The targeted suite passed **20 tests**. Both Monte Carlo CLI checks exited 0 and the analysis module loaded all 8 and 5,000 records in a fresh process. Syntax compilation, `git diff --check`, and CI YAML/test wiring checks passed. The normal-timeout eight-trial run retained the expected 1009-second baseline; 3 of 8 trials completed successfully. The short dispatch test's success fraction is not a scientific performance estimate.

### Final 24-hour scenario

Ran the `leop_24h` preset with its default 2-second fine/20-second coarse configuration, using `no_plots=True` and the separate output tag `relay_repair_final_24h`. All other preset settings were retained. The process exited **0** and reached **86400 seconds**.

| Check | Result |
| --- | --- |
| Telemetry rows | 7,696, spanning 0–86400 seconds |
| Modes visited | COLD_AND_DARK, DETUMBLE, SUN_POINTING, SLEWING, DETECTION, RELAY |
| Detumble start / completion | 120 / 2580 seconds mission time (2460 seconds detumbling) |
| Visible RELAY samples | 323 |
| Maximum visible RELAY LOS error | `1.2074182697257331e-6` degrees |
| Battery range | 15.1281–25 Wh |
| Science generated / downlinked (mission summary) | 0.009079 / 0.009079 MB |
| First GS contact | 34840–35192 seconds, 0.004929 MB downlinked |
| Second GS contact | 77806–78220 seconds, 0.004150 MB downlinked |
| Pointing assertions | All passed; limits unchanged |

Outputs are `results/science/relay_repair_final_24h_ANTARA_Science_*.csv`; the execution log is `/tmp/antara-relay-repair-final.log`. The maximum error above filters to visible-station RELAY samples: outside contact, the target falls back to nadir, so those angles do not measure ground-station tracking. Telemetry's per-step science/downlink fields are not cumulative mission totals.

The originally reported command can now be rerun unchanged:

```bash
conda activate orekit_env
python src/simulations/ditl/ANTARA_Science_Sim.py --scenario leop_24h
python src/simulations/detumble/leop_detumble_mc.py --n-trials 5000 --output detumble_mc_results.pkl
```

Plot generation was disabled only for the long verification run; the plotting path was not modified. The second command requests a full normal-timeout Monte Carlo study, which is more expensive than the 5,000-task short-timeout dispatch verification described above.

## Files

- `src/simulations/ditl/ANTARA_Science_Sim.py`: moving LOS acquisition, settlement, slew completion, consistent timestep accounting.
- `src/simulations/helper/antara_models.py`: preserve speed across ongoing slew target updates.
- `src/simulations/detumble/leop_detumble_mc.py`: importable worker, explicit arguments, deterministic result order, portable pickle classes.
- `tests/test_physics_attitude.py`: runtime settling, maneuver completion, and timestep regressions.
- `tests/test_detumble_multiprocessing.py`: real spawn/serial comparison and CLI pickle round trip.
- `.github/workflows/ci.yml`: runs the multiprocessing regression alongside physics/attitude tests.
- Contributor and simulation docs: explain corrected runtime behavior.

## Remaining limits

SpacePy's current AP9 call still uses the documented configured-baseline fallback; this repair does not claim live AP9 support. A short zero-science warning can occur before the first packet is emitted; subsequent science totals determine whether data were generated. Third-party Orekit/JCC deprecation warnings remain separate from these failures. Hosted GitHub Actions and the full 5,000-trial normal-timeout study were not run as part of this repair.

Existing documentation edits and prior scenario outputs were preserved. Verification uses separate `relay_repair_*` and `relay_contact_replay*` output tags. No commit or push was performed.
