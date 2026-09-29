"""Small scientific-accounting regressions; no annual simulations."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import orekit

orekit.initVM()
from simulations.helper.antara_models import ADCSModel, MissionConfig
from simulations.ditl import ANTARA_Science_Sim as science
from simulations.detumble import leop_detumble_mc as mc
from simulations.ditl.compare_scenarios import compute_mode_duty_cycle
from simulations.helper.telemetry_metrics import streaming_summary


def test_adcs_motion_and_zero_step():
    adcs = ADCSModel(MissionConfig())
    adcs.current_quaternion = np.array([1., 0., 0., 0.])
    adcs.target_quaternion = adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), 10.)
    before = adcs.current_quaternion.copy()
    adcs.advance(0.)
    np.testing.assert_array_equal(before, adcs.current_quaternion)
    adcs.advance(.01)
    angle = adcs.compute_slew(before, adcs.current_quaternion)[0]
    assert 0 < angle <= adcs.config.max_slew_rate_deg_sec * .01


def test_braking_torque_changes_sign():
    adcs = ADCSModel(MissionConfig())
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.command_attitude(adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.),
                         "SLEWING", "acceleration probe", force_update=True)
    adcs.advance(.1)
    acceleration = adcs.estimate_torque_vector()
    # Within stopping distance, but not yet in the pointing acceptance band.
    adcs.target_quaternion = adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 1.)
    adcs.current_slew_rate_deg_sec = 5.
    adcs.advance(.1)
    np.testing.assert_allclose(adcs.estimate_torque_vector(), -acceleration, atol=1e-15, rtol=1e-12)


def test_rate_divergence_is_not_clipped():
    adcs = ADCSModel(MissionConfig())
    adcs.body_rate_deg_sec[:] = 0.
    with pytest.raises(FloatingPointError):
        adcs._integrate_body_rates(np.array([1e6, 0., 0.]), 1.)


def test_environment_amplitude_reaches_detector():
    outputs = []
    for amplitude in (1., 1000.):
        env = SimpleNamespace(PITCH_EXPONENT=.5, compute_flux=lambda *a, **kw: {
            "saa_depth": 1., "pitch_angle_deg": 0., "ew_factor": 1.,
            "measured_flux": np.full(5, amplitude), "ap9_flux": np.full(5, amplitude)})
        payload = science.PayloadModel(MissionConfig(), env)
        payload.open_saa_pass()
        _, result = payload.generate_data(2., -25., -50., 600., np.ones(3), np.ones(3), np.ones(3))
        outputs.append(result)
    np.testing.assert_allclose(outputs[1]["measured_flux"], outputs[0]["measured_flux"] * 1000)
    assert sum(outputs[1]["counts_per_bin"]) > sum(outputs[0]["counts_per_bin"])


@pytest.mark.parametrize("dt", [0., -1., float('nan'), float('inf')])
def test_invalid_mc_step_rejected(dt):
    with pytest.raises(ValueError):
        mc.run_single_trial(np.ones(3), np.zeros(3), MissionConfig(), dt, 10.)


def test_mc_acceleration_rejected_and_wheel_momentum_measured():
    with pytest.raises(ValueError, match="acceleration"):
        mc.run_single_trial(np.ones(3), np.ones(3), MissionConfig())
    result = mc.run_single_trial(np.array([.2, .1, .1]), np.zeros(3), MissionConfig(), .1, 20.)
    assert result.peak_wheel_momentum_nms > 0.


def test_failed_baseline_keeps_finite_timeout(monkeypatch):
    timeouts = []
    def fake(*args, **kwargs):
        timeouts.append(args[4])
        return SimpleNamespace(success=False, hit_rate_cap=False, time_to_complete_sec=float('inf'))
    monkeypatch.setattr(mc, "run_single_trial", fake)
    mc.run_monte_carlo(2, MissionConfig(), n_workers=1, show_progress=False)
    assert timeouts == [10000., 10000.]


def test_time_weights_and_streaming_chunks(tmp_path):
    frame = pd.DataFrame({"time_sec": [0., 2., 22.], "interval_duration_sec": [2., 20., 0.],
                          "mode": ["DETECTION", "SUN_POINTING", "SUN_POINTING"],
                          "battery_wh": [10., 20., 30.],
                          "wheel_mom_x": [0., 0., 0.], "wheel_mom_y": [0., 0., 0.], "wheel_mom_z": [0., 0., 0.]})
    assert compute_mode_duty_cycle(frame, 22/3600)["DETECTION"] == pytest.approx(100 * 2/22)
    path = tmp_path / "telemetry.csv"
    frame.to_csv(path, index=False)
    result = streaming_summary(path, chunksize=1)
    assert result["duration_sec"] == 22.
    assert result["metrics"]["battery_wh"]["mean"] == pytest.approx(420/22)
    assert result["metrics"]["battery_wh"]["max"] == 30.


def test_thermal_view_factor_matches_explicit_trapezoidal_sum():
    from simulations.thermal.ThermalSimulator import ThermalSimulator, ThermalMissionConfig
    sim = ThermalSimulator.__new__(ThermalSimulator)
    sim.config = ThermalMissionConfig()
    sim._build_earth_view_factor_table(phi_resolution_deg=30., alpha_samples=200, beta_samples=200)
    alpha = np.linspace(1e-6, sim._nominal_earth_angular_radius_rad(), 200)
    beta = np.linspace(0., 2. * np.pi, 200, endpoint=False)
    expected = []
    for phi in np.radians(sim._view_factor_phi_deg):
        cosine = (np.cos(phi) * np.cos(alpha)[:, None]
                  + np.sin(phi) * np.sin(alpha)[:, None] * np.cos(beta)[None, :])
        inner = (np.clip(cosine, 0., None) * np.sin(alpha)[:, None]).sum(axis=1) * (beta[1] - beta[0])
        expected.append(np.sum(np.diff(alpha) * (inner[1:] + inner[:-1]) / 2.) / np.pi)
    np.testing.assert_allclose(sim._view_factor_table, expected, rtol=1e-14, atol=1e-15)


def test_thermal_exact_horizon():
    from simulations.thermal.ThermalSimulator import ThermalSimulator, ThermalMissionConfig
    sim = ThermalSimulator(ThermalMissionConfig(sim_duration_hours=5/3600, step_sec=2.))
    initial = sim.temperatures_k.copy()
    original = sim._rk4_thermal_step
    intervals = []
    def observed(*args):
        intervals.append(args[-1])
        return original(*args)
    sim._rk4_thermal_step = observed
    frame = sim.run(write_outputs=False, verbose=False)
    assert sum(intervals) == pytest.approx(5.)
    assert frame.time_sec.tolist() == [0., 2., 4., 5.]
    assert frame.internal_temp_k.iloc[0] == initial["INTERNAL"]


def test_saa_entry_credits_first_packet():
    from org.orekit.time import AbsoluteDate
    model = science.SAAModel()
    kind, record = model.detect_entry_exit(False, True, 0., -25., -50., AbsoluteDate.J2000_EPOCH, .25)
    assert kind == "ENTRY"
    assert record["data_collected_mb"] == .25


def test_master_refuses_unfinished_slew():
    from simulations.legacy import ANTARA_Master as master
    sim = master.ANTARASimulator.__new__(master.ANTARASimulator)
    sim.fsm = SimpleNamespace(current_mode="SLEWING")
    sim.adcs = SimpleNamespace(saturation_hold_active=False, is_slewing=True, settling=False)
    assert sim._complete_slew_transition(10000.) is False


def test_downloaded_orekit_data_is_discoverable(monkeypatch, tmp_path):
    from simulations.helper import paths
    data = tmp_path / "data"
    data.mkdir()
    archive = data / "orekit-data.zip"
    archive.touch()
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    called = []
    monkeypatch.setattr(paths, "setup_orekit_curdir", called.append)
    assert paths.init_orekit_data() == archive
    assert called == [str(archive)]


def test_reaction_wheel_opposes_bus_torque():
    adcs = ADCSModel(MissionConfig())
    adcs.wheel_momentum_nms[:] = 0.
    adcs.update_wheel_momentum(np.array([.001, 0., 0.]), 1., 0., 0.)
    np.testing.assert_allclose(adcs.wheel_momentum_nms, [-.001, 0., 0.])
    adcs.update_wheel_momentum(np.array([-.001, 0., 0.]), 1., 0., 0.)
    np.testing.assert_allclose(adcs.wheel_momentum_nms, 0.)


@pytest.mark.parametrize("duration_sec", [2., 4.])
@pytest.mark.parametrize("zero_counts", [False, True])
def test_first_detection_sample_survives_entry(monkeypatch, tmp_path, duration_sec, zero_counts):
    from dataclasses import replace
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    config = MissionConfig(sim_duration_hours=duration_sec/3600, step_sec=2., coarse_step_sec=2., no_plots=True)
    if zero_counts:
        config = replace(config, detector_dark_count_rate_hz=0., ap9_baseline_flux=(0.,) * len(config.energy_channels_mev))
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.fsm.current_mode = sim.target_mode = "DETECTION"
    sim.adcs.settled_time_sec = 1000.
    sim.current_payload_pointing_error_deg = 0.
    # Controlled valid acquisition at an SAA boundary; exercise production ordering.
    monkeypatch.setattr(sim.saa, "is_inside", lambda *args: sim.time_sec < 2.)
    monkeypatch.setattr(sim, "_update_attitude_guidance", lambda mode, orbit, *args, **kwargs:
                        setattr(sim, "current_payload_pointing_error_deg", sim._compute_payload_pointing_error_deg(orbit)))
    sim.run()
    assert len(sim.payload.pass_science_records) == 1
    assert sim.payload.pass_science_records[0]["n_science_steps"] == 1
    generated = sim.payload.total_science_data_generated_mb
    # A zero-count observation still serializes a fixed-width histogram packet.
    assert generated > 0.
    counts = sum(sim.payload.pass_science_records[0]["total_counts_per_bin"])
    assert (counts == 0.) if zero_counts else (counts > 0.)
    assert sum(p["data_collected_mb"] for p in sim.saa.completed_passes) == pytest.approx(generated)
    assert sim.science_storage_mb + sim.communications.total_downlinked_mb == pytest.approx(generated)
    frame = pd.read_csv(tmp_path / "ANTARA_Science_telemetry.csv")
    assert frame.science_data_mb.sum() == pytest.approx(generated)
    assert ("detection_zero_science" in sim.warning_state) == zero_counts


@pytest.mark.parametrize("name,target", [("ANTARA_MCR.py", "delta_energy"), ("ANTARA_avg.py", "battery_energy")])
def test_legacy_energy_balance_without_launching_study(name, target):
    import ast
    from pathlib import Path
    tree = ast.parse((Path(__file__).resolve().parents[1] / "src/simulations/power" / name).read_text())
    expressions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == target for t in node.targets):
            expressions.append(node.value)
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and node.target.id == target:
            expressions.append(node.value)
    expression = next(value for value in expressions if any(isinstance(n, ast.Name) and n.id == "battery_eff" for n in ast.walk(value)))
    code = compile(ast.Expression(expression), name, "eval")
    assert eval(code, {}, dict(power_net=-8., dt_hr=1., battery_eff=.8)) == pytest.approx(-10.)
    assert eval(code, {}, dict(power_net=8., dt_hr=1., battery_eff=.8)) == pytest.approx(6.4)


@pytest.mark.parametrize("checkpoint_version", [7, 8, 9])
def test_checkpoint_mid_detection_conserves_pending_packets(monkeypatch, tmp_path, checkpoint_version):
    from dataclasses import replace
    import pickle
    from copy import deepcopy
    captured = []
    original_add = science.PassStatistics.add
    def capture(self, record):
        captured.append(deepcopy(record))
        original_add(self, record)
    monkeypatch.setattr(science.PassStatistics, "add", capture)
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    # Controlled SAA/attitude fixture, applied at class scope so snapshots do
    # not contain monkeypatched instance callables. All payload I/O is real.
    monkeypatch.setattr(science.SAAModel, "is_inside", lambda *args: True)
    monkeypatch.setattr(science.ANTARAScienceSimulator, "_update_attitude_guidance",
                        lambda self, mode, orbit, *args, **kwargs: setattr(
                            self, "current_payload_pointing_error_deg",
                            self._compute_payload_pointing_error_deg(orbit)))
    config = MissionConfig(sim_duration_hours=6/3600, step_sec=2., coarse_step_sec=2.,
                           no_plots=True, output_tag="pending")
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.fsm.current_mode = sim.target_mode = "DETECTION"
    sim.adcs.settled_time_sec = 1000.
    sim.run_steps(1)
    assert sim.payload.total_science_data_generated_mb == 0.
    assert sum(np.sum(v) for v in sim.payload._pending_bin_counts.values()) > 0.
    checkpoint = tmp_path / "pending.pkl"
    sim.save_checkpoint(checkpoint)
    with checkpoint.open("rb") as stream:
        saved = pickle.load(stream)
    assert saved["version"] == 9
    if checkpoint_version < 9:
        saved["components"]["payload"]["pass_science_records"] = list(
            saved["components"]["payload"]["pass_science_records"])
    if checkpoint_version == 7:
        # Reconstruct the prior payload schema with the actual first sample.
        state = saved["components"]["payload"]
        del state["_pass_statistics"]
        state["_current_pass_records"] = deepcopy(captured)
        for suffix, key in (("cell_ids", "cell_id"), ("event_types", "event_type"),
                            ("livetime", "livetime_fraction"), ("sigma_fraction", "sigma_fraction"),
                            ("counts_per_bin", "counts_per_bin"), ("region", "region")):
            state["_current_pass_" + suffix] = [deepcopy(r[key]) for r in captured]
    if checkpoint_version < 9:
        saved["version"] = checkpoint_version
        with checkpoint.open("wb") as stream:
            pickle.dump(saved, stream)
    sim.run()
    resumed = science.ANTARAScienceSimulator.load_checkpoint(checkpoint, replace(config, output_tag="resumed"))
    resumed.run()
    assert "_current_pass_records" not in vars(resumed.payload)
    assert len(resumed.payload.pass_science_records) == len(sim.payload.pass_science_records)
    for actual, expected in zip(resumed.payload.pass_science_records, sim.payload.pass_science_records):
        for key in ("mean_pitch_angle_deg", "std_pitch_angle_deg", "mean_ew_factor",
                    "mean_measured_flux", "mean_ap9_flux", "mean_sigma_fraction", "total_counts_per_bin"):
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-14)
    for run in (sim, resumed):
        assert run.payload.total_science_data_generated_mb > 0.
        assert run.saa.completed_passes[0]["data_collected_mb"] == pytest.approx(run.payload.total_science_data_generated_mb)
        assert run.science_storage_mb + run.communications.total_downlinked_mb == pytest.approx(run.payload.total_science_data_generated_mb)
        assert not run.payload._pending_bin_counts or all(np.sum(v) == 0 for v in run.payload._pending_bin_counts.values())
    for suffix in ("telemetry", "saa_passes", "proton_passes"):
        a = pd.read_csv(tmp_path / f"pending_ANTARA_Science_{suffix}.csv")
        b = pd.read_csv(tmp_path / f"resumed_ANTARA_Science_{suffix}.csv")
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("implementation", ["shared", "master"])
@pytest.mark.parametrize("axis", [np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                                  np.array([0., 0., 1.]), np.array([-1., 2., -3.])])
def test_slew_torque_body_frame_and_diagnostic_independence(implementation, axis):
    if implementation == "master":
        from simulations.legacy import ANTARA_Master as model
    else:
        from simulations.helper import antara_models as model
    adcs = model.ADCSModel(model.MissionConfig())
    axis = axis / np.linalg.norm(axis)
    initial = adcs.quaternion_from_axis_angle(np.array([0., 1., 0.]), 57.)
    adcs.set_initial_quaternion(initial)
    # Right multiplication expresses the relative target rotation in BODY axes.
    target = adcs._quaternion_multiply(initial, adcs.quaternion_from_axis_angle(axis, 30.))
    adcs.command_attitude(target, "SLEWING", "body-axis probe", force_update=True)
    adcs.advance(.1)
    expected = axis * adcs.config.wheel_max_torque_nm
    np.testing.assert_allclose(adcs.estimate_torque_vector(), expected, atol=1e-15)
    adcs.compute_slew(np.array([1., 0., 0., 0.]),
                      adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 75.))
    np.testing.assert_allclose(adcs.estimate_torque_vector(), expected, atol=1e-15)
    # Returned arrays must not permit callers to modify the cached result.
    adcs.estimate_torque_vector()[:] = 0.
    np.testing.assert_allclose(adcs.estimate_torque_vector(), expected, atol=1e-15)
    adcs.saturation_hold_active = True
    np.testing.assert_array_equal(adcs.estimate_torque_vector(), np.zeros(3))
    held_quaternion = adcs.current_quaternion.copy()
    adcs.advance(.1)
    np.testing.assert_array_equal(adcs.current_quaternion, held_quaternion)
    adcs.saturation_hold_active = False
    np.testing.assert_array_equal(adcs.estimate_torque_vector(), np.zeros(3))
    adcs.advance(0.)
    np.testing.assert_array_equal(adcs.estimate_torque_vector(), np.zeros(3))
    adcs.last_slew_body_torque_nm = expected.copy()
    adcs.set_initial_quaternion(initial)
    np.testing.assert_array_equal(adcs.estimate_torque_vector(), np.zeros(3))
