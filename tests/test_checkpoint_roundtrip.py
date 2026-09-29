"""Collected checkpoint safety checks (also runnable as a standalone script)."""
if __name__ == "__main__":
    # Start pytest before loading Orekit/JVM-backed modules. Re-entering pytest
    # inside an already initialized wrapper process can crash native bindings.
    import os
    import sys
    os.execv(sys.executable, [sys.executable, "-m", "pytest", __file__, "-v", "--disable-warnings"])

from dataclasses import replace
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from test_execution_integrity import config, science


def test_checkpoint_write_failure_preserves_previous(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config)
    path = tmp_path / "state.pkl"
    sim.save_checkpoint(path)
    original = path.read_bytes()
    sim.warning_state["unserializable"] = lambda: None
    with pytest.raises((pickle.PicklingError, AttributeError)):
        sim.save_checkpoint(path)
    assert path.read_bytes() == original


def test_checkpoint_prefix_integrity_and_config(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.run_steps(1)
    path = tmp_path / "state.pkl"
    sim.save_checkpoint(path)
    with pytest.raises(ValueError, match="configuration"):
        science.ANTARAScienceSimulator.load_checkpoint(path, replace(config, step_sec=4.))
    sim.logger._flush_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="prefix"):
        science.ANTARAScienceSimulator.load_checkpoint(path, replace(config, output_tag="resumed"))
    assert not (tmp_path / "resumed_ANTARA_Science_telemetry.csv").exists()


def test_checkpoint_only_copies_saved_prefix(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.run_steps(1)
    path = tmp_path / "state.pkl"
    sim.save_checkpoint(path)
    prefix = sim.logger._flush_path.read_bytes()
    sim.run()
    longer = sim.logger._flush_path.read_bytes()
    assert len(longer) > len(prefix)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path)
    assert resumed.logger._flush_path != sim.logger._flush_path
    assert resumed.logger._flush_path.read_bytes() == prefix
    assert sim.logger._flush_path.read_bytes() == longer
    assert resumed.time_sec == 2.


@pytest.mark.parametrize("version", [None, 2, 3, 4, 5, 6])
def test_legacy_checkpoint_rejected(config, tmp_path, version):
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps({"version": version, "time_sec": 123.}))
    with pytest.raises(ValueError, match="version-7"):
        science.ANTARAScienceSimulator.load_checkpoint(path, config)


@pytest.mark.parametrize("version", [7, 8, 9, 10])
def test_checkpoint_rejects_missing_pass_statistics(tmp_path, version):
    path = tmp_path / "incomplete.pkl"
    path.write_bytes(pickle.dumps({"version": version, "components": {"payload": {}}}))
    with pytest.raises(ValueError, match="pass statistics"):
        science.ANTARAScienceSimulator.load_checkpoint(path)


def test_checkpoint_rejects_partial_step(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config)
    sim._at_step_boundary = False
    with pytest.raises(RuntimeError, match="incomplete"):
        sim.save_checkpoint(tmp_path / "unsafe.pkl")


def test_completed_pass_archive_checkpoint_branch(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config)
    sim.payload.pass_science_records.append({"pass_index": 1})
    path = tmp_path / "archive.pkl"
    sim.save_checkpoint(path)
    sim.payload.pass_science_records.append({"pass_index": 2})
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path)
    assert list(resumed.payload.pass_science_records) == [{"pass_index": 1}]
    resumed.payload.pass_science_records.append({"pass_index": 3})
    assert sim.payload.pass_science_records[-1] == {"pass_index": 2}
    sim.payload.pass_science_records.path.write_text("corrupt")
    with pytest.raises(ValueError, match="prefix"):
        science.ANTARAScienceSimulator.load_checkpoint(path)


def test_all_event_archives_resume_with_open_events(config, tmp_path):
    sim = science.ANTARAScienceSimulator(replace(config, gs2_enabled=True))
    for component, field, kind in sim._EVENT_ARCHIVES:
        owner = getattr(sim, component)
        if owner is None:
            continue
        archive = getattr(owner, field)
        if kind is science.OpenEventArchive:
            archive.append({"start_time_sec": 0., "end_time_sec": 1., "duration_sec": 1.})
            archive.append({"start_time_sec": 2., "end_time_sec": None, "duration_sec": 0.})
        else:
            archive.append({"n": 1})
    path = tmp_path / "all_events.pkl"
    sim.save_checkpoint(path)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path)
    for component, field, kind in sim._EVENT_ARCHIVES:
        owner = getattr(sim, component)
        if owner is None:
            continue
        original = getattr(owner, field)
        branch = getattr(getattr(resumed, component), field)
        assert list(original) == list(branch)
        if kind is science.OpenEventArchive:
            original.finalize(1, 3.)
            branch.finalize(1, 4.)
            assert original[-1]["duration_sec"] == 1.
            assert branch[-1]["duration_sec"] == 2.
        else:
            branch.append({"n": 2})
            assert len(original) == 1


@pytest.mark.parametrize("component,field,kind", science.ANTARAScienceSimulator._EVENT_ARCHIVES)
def test_every_archive_prefix_is_verified(config, tmp_path, monkeypatch, component, field, kind):
    sim = science.ANTARAScienceSimulator(replace(config, gs2_enabled=True))
    owner = getattr(sim, component)
    if owner is None:
        pytest.skip("Optional second station disabled")
    archive = getattr(owner, field)
    record = {"start_time_sec": 0., "end_time_sec": 1.} if kind is science.OpenEventArchive else {"n": 1}
    archive.append(record)
    path = tmp_path / "corrupt_event.pkl"
    sim.save_checkpoint(path)
    source = archive.completed if kind is science.OpenEventArchive else archive
    source.path.write_text("corrupt")
    def unexpected_init(*args, **kwargs):
        pytest.fail("Archive verification must precede construction/output creation")
    monkeypatch.setattr(science.ANTARAScienceSimulator, "__init__", unexpected_init)
    with pytest.raises(ValueError, match="prefix"):
        science.ANTARAScienceSimulator.load_checkpoint(path)


def test_checkpoint_preserves_inflight_slew_profile(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config)
    sim.adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = sim.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), 30.)
    sim.adcs.command_attitude(target, "SLEWING", "restart probe", force_update=True)
    sim.adcs.advance(.1)
    path = tmp_path / "slew.pkl"
    sim.save_checkpoint(path)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path)
    np.testing.assert_array_equal(resumed.adcs.slew_axis_inertial, sim.adcs.slew_axis_inertial)
    np.testing.assert_array_equal(resumed.adcs.estimate_torque_vector(), sim.adcs.estimate_torque_vector())
    for adcs in (sim.adcs, resumed.adcs):
        adcs.advance(.2)
    np.testing.assert_array_equal(resumed.adcs.current_quaternion, sim.adcs.current_quaternion)
    assert resumed.adcs.current_slew_rate_deg_sec == sim.adcs.current_slew_rate_deg_sec


def test_periodic_checkpoint_uses_simulation_time(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.configure_checkpointing(2. / 3600.)
    sim.run_steps(1)
    path = tmp_path / "reference_2.000000s.pkl"
    assert path.exists()
    restored = science.ANTARAScienceSimulator.load_checkpoint(path)
    assert restored.time_sec == 2.
    assert restored._next_checkpoint_sec == 4.


def test_cli_checkpoint_resumes_in_another_process(tmp_path):
    import pandas as pd
    root = tmp_path / "cli"
    source_dir = Path(__file__).resolve().parents[1] / "src"
    script = (
        f"import sys; sys.path.insert(0, {str(source_dir)!r}); "
        "from simulations.helper import paths; import runpy; from pathlib import Path; "
        f"paths.RESULTS_DIR=Path({str(root)!r}); "
        "sys.argv=['ANTARA_Science_Sim.py']+sys.argv[1:]; "
        "runpy.run_path('src/simulations/ditl/ANTARA_Science_Sim.py',run_name='__main__')"
    )
    subprocess.run([sys.executable, "-c", script, "--duration-hours", str(6/3600),
                    "--step-sec", "2", "--checkpoint-every-hours", str(2/3600), "--no-plots"],
                   check=True, capture_output=True, text=True, timeout=120)
    checkpoint = root / "checkpoints/science_2.000000s.pkl"
    subprocess.run([sys.executable, "-c", script, "--resume-from", str(checkpoint), "--no-plots"],
                   check=True, capture_output=True, text=True, timeout=120)
    original = pd.read_csv(root / "science/ANTARA_Science_telemetry.csv")
    restored = pd.read_csv(next((root / "science").glob("science_resume_*_ANTARA_Science_telemetry.csv")))
    pd.testing.assert_frame_equal(original, restored, check_exact=False, rtol=1e-8, atol=1e-8)
