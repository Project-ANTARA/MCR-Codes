"""Short exact wheel-event and common-clock integration regressions."""
import numpy as np
import pandas as pd
import pytest
from wheel_dynamics import integrate_wheel_profile
from antara_models import ADCSModel, MissionConfig


def test_profile_detects_hidden_peak_and_zero_crossing():
    segments = [(np.array([-.001, 0., 0.]), 5.), (np.array([.001, 0., 0.]), 5.)]
    result, crossing = integrate_wheel_profile(np.zeros(3), segments, 0., .003)
    np.testing.assert_allclose(result, 0., atol=1e-15)
    assert crossing == pytest.approx(3.)
    # Start positive, cross zero, then build negative momentum while unloading.
    result, crossing = integrate_wheel_profile(np.array([.001, 0., 0.]),
                                               [(np.array([.002, 0., 0.]), 2.)], .001, .0015)
    assert result[0] == pytest.approx(-.001 * (2. - 1./3.))
    assert crossing == pytest.approx(1./3. + 1.5)


@pytest.mark.parametrize("implementation", ["shared", "master"])
def test_capacity_event_shortens_motion_and_activates_hold(implementation):
    if implementation == "master":
        import orekit
        orekit.initVM()
        import ANTARA_Master as model
        adcs = model.ADCSModel(model.MissionConfig(wheel_max_momentum_nms=.003))
    else:
        adcs = ADCSModel(MissionConfig(wheel_max_momentum_nms=.003))
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.)
    adcs.command_attitude(target, "SLEWING", "event probe", force_update=True)
    interval = ADCSModel.advance_limited(adcs, 11., 1., 1.)
    unloading = adcs.config.magnetorquer_dipole_Am2 * adcs.config.earth_B_field_T
    expected = .003 / (.001 - unloading)
    assert interval == pytest.approx(expected)
    assert interval < 11.
    assert adcs.current_slew_rate_deg_sec == pytest.approx(interval)
    assert adcs.compute_slew(np.array([1., 0., 0., 0.]), adcs.current_quaternion)[0] == pytest.approx(.5 * interval**2)
    adcs.set_time(interval)
    adcs.update_wheel_momentum(adcs.estimate_torque_vector(), interval, 1., 1., use_slew_profile=True)
    assert adcs.saturation_hold_active
    assert adcs.saturation_start_time_sec == interval
    assert adcs.saturation_event_count == 1
    frozen = adcs.current_quaternion.copy()
    remaining = 11. - interval
    assert ADCSModel.advance_limited(adcs, remaining, 1., 1.) == remaining
    np.testing.assert_array_equal(adcs.current_quaternion, frozen)
    adcs.update_wheel_momentum(adcs.estimate_torque_vector(), remaining, 1., 1.)
    assert adcs.saturation_hold_active
    assert adcs.saturation_event_count == 1


def test_safe_slew_keeps_requested_interval():
    adcs = ADCSModel(MissionConfig())
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.command_attitude(adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.),
                         "SLEWING", "safe", force_update=True)
    assert adcs.advance_limited(11., 1., 1.) == 11.
    adcs.update_wheel_momentum(adcs.estimate_torque_vector(), 11., 1., 1., use_slew_profile=True)
    assert not adcs.saturation_hold_active


def test_profile_mismatch_does_not_mutate_actuators():
    import pickle
    adcs = ADCSModel(MissionConfig())
    adcs.last_slew_torque_segments = [(np.array([.001, 0., 0.]), 1.)]
    before = pickle.dumps(vars(adcs))
    with pytest.raises(ValueError, match="mean torque"):
        adcs.update_wheel_momentum(np.zeros(3), 1., 0., 0., use_slew_profile=True)
    assert pickle.dumps(vars(adcs)) == before


@pytest.mark.parametrize("seed", range(8))
def test_oblique_wheel_event_and_preview_isolation(seed):
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    adcs = ADCSModel(MissionConfig(wheel_max_momentum_nms=.002))
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = adcs.quaternion_from_axis_angle(axis, 60.)
    adcs.command_attitude(target, "SLEWING", "oblique", force_update=True)
    before = adcs.current_quaternion.copy()
    interval = adcs.advance_limited(20., 1., 1.)
    assert interval < 20.
    # Preview must not add motion or time to the real update.
    assert adcs.total_slew_time_sec == pytest.approx(interval)
    assert adcs.compute_slew(before, adcs.current_quaternion)[0] == pytest.approx(.5 * interval**2)
    adcs.update_wheel_momentum(adcs.estimate_torque_vector(), interval, 1., 1., use_slew_profile=True)
    assert adcs.saturation_hold_active


def test_science_event_uses_one_common_clock(monkeypatch, tmp_path):
    import orekit
    orekit.initVM()
    import pandas as pd
    import ANTARA_Science_Sim as science
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    config = MissionConfig(sim_duration_hours=11./3600., step_sec=11., coarse_step_sec=11.,
                           no_plots=True, wheel_max_momentum_nms=.003)
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = sim.adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.)
    sim.current_attitude_command = science.AttitudeCommand(target, "event probe", "")
    sim.adcs.command_attitude(target, "SLEWING", "event probe", force_update=True)
    sim.fsm.current_mode = "SLEWING"
    sim.target_mode = "SUN_POINTING"
    sim.slew_end_time_sec = 1000.
    eps_steps = []
    eps_step = sim.eps.step
    def record_eps(dt):
        eps_steps.append(dt)
        eps_step(dt)
    monkeypatch.setattr(sim.eps, "step", record_eps)
    sim.run()
    frame = pd.read_csv(tmp_path / "ANTARA_Science_telemetry.csv")
    assert 0. < eps_steps[0] < 11.
    assert sum(eps_steps) == pytest.approx(11.)
    np.testing.assert_allclose(np.diff(frame.time_sec), eps_steps)
    np.testing.assert_allclose(frame.interval_duration_sec.iloc[:-1], eps_steps)
    assert sim.time_sec == 11.
    assert frame.wheel_event_split.iloc[0]
    assert not frame.wheel_event_split.iloc[-1]
    assert sim.adcs.saturation_event_count == 1
    assert sim.payload.total_science_data_generated_mb == 0.
    assert sim.communications.total_downlinked_mb == 0.


def test_checkpoint_at_wheel_event_matches_uninterrupted(monkeypatch, tmp_path):
    import orekit
    orekit.initVM()
    import pandas as pd
    from dataclasses import replace
    import ANTARA_Science_Sim as science
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    config = MissionConfig(sim_duration_hours=11./3600., step_sec=11., coarse_step_sec=11.,
                           no_plots=True, wheel_max_momentum_nms=.003, output_tag="event")
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = sim.adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.)
    sim.current_attitude_command = science.AttitudeCommand(target, "event probe", "")
    sim.adcs.command_attitude(target, "SLEWING", "event probe", force_update=True)
    sim.fsm.current_mode = "SLEWING"
    sim.target_mode = "SUN_POINTING"
    sim.slew_end_time_sec = 1000.
    sim.run_steps(1)
    assert sim.adcs.saturation_hold_active
    assert 0. < sim.time_sec < 11.
    path = tmp_path / "event.pkl"
    sim.save_checkpoint(path)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path, replace(config, output_tag="resumed"))
    sim.run()
    resumed.run()
    np.testing.assert_array_equal(sim.adcs.wheel_momentum_nms, resumed.adcs.wheel_momentum_nms)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "event_ANTARA_Science_telemetry.csv"),
                                  pd.read_csv(tmp_path / "resumed_ANTARA_Science_telemetry.csv"))


def test_master_event_preserves_total_eps_duration(monkeypatch, tmp_path, capsys):
    import orekit
    orekit.initVM()
    import ANTARA_Master as master
    monkeypatch.setattr(master, "results_path", lambda *parts: tmp_path / parts[-1])
    sim = master.ANTARASimulator(master.MissionConfig(sim_duration_hours=11./3600., step_sec=11.,
                                                     wheel_max_momentum_nms=.003))
    sim.fsm.current_mode = "SLEWING"
    sim.target_mode = "SUN_POINTING"
    sim.slew_end_time_sec = 1000.
    sim.adcs.last_commanded_mode = "SLEWING"
    intervals = []
    eps_step = sim.eps.step
    def record_eps(dt):
        intervals.append(dt)
        eps_step(dt)
    monkeypatch.setattr(sim.eps, "step", record_eps)
    monkeypatch.setattr(sim, "_plot_results", lambda *args: None)
    sim.run()
    assert 0. < intervals[0] < 11.
    assert sum(intervals) == pytest.approx(11.)
    assert sim.adcs.saturation_event_count == 1
    frame = pd.read_csv(tmp_path / "ANTARA_Master_telemetry.csv")
    np.testing.assert_allclose(frame.interval_duration_sec, intervals)
    np.testing.assert_allclose(frame.state_time_sec, frame.time_sec + frame.interval_duration_sec)
    assert frame.state_time_sec.iloc[-1] == pytest.approx(11.)

    # Exercise the real summary with controlled unequal observation intervals.
    report = frame.iloc[[0, 0]].copy().reset_index(drop=True)
    report["time_sec"] = [0., 2.]
    report["interval_duration_sec"] = [2., 8.]
    report["state_time_sec"] = [2., 10.]
    report["solar_w"] = [0., 10.]
    report["load_w"] = [0., 0.]
    report["in_eclipse"] = [True, False]
    report["mode"] = "SUN_POINTING"
    report["sun_submode"] = [master.SunSubmode.CHARGING, master.SunSubmode.PASSIVE]
    report["attitude_error_deg"] = [0., 10.]
    report["slew_angle_deg"] = 0.
    sim.config.sim_duration_hours = 10 / 3600
    sim.fsm.mode_durations_sec["SUN_POINTING"] = 10.
    sim.orbital_period_history_min = [100., 200.]
    capsys.readouterr()
    empty = pd.DataFrame()
    sim._print_summary(report, empty, empty, empty, empty)
    output = capsys.readouterr().out
    assert "Average orbital period (min): 180.00" in output
    assert "Average solar generation (W): 8.00" in output
    assert "Eclipse fraction (%): 20.00" in output
    assert "ADCS duty cycle (%): 80.00" in output
    assert "SUN_POINTING: 8.00" in output
    assert "(20.0% of SUN_POINTING)" in output
    assert "(80.0% of SUN_POINTING)" in output
    with pytest.raises(ValueError, match="interval_duration_sec"):
        sim._print_summary(report.drop(columns="interval_duration_sec"), empty, empty, empty, empty)


def test_master_partial_final_step_has_exact_end_state(monkeypatch, tmp_path):
    import orekit
    orekit.initVM()
    import ANTARA_Master as master
    monkeypatch.setattr(master, "results_path", lambda *parts: tmp_path / parts[-1])
    sim = master.ANTARASimulator(master.MissionConfig(sim_duration_hours=5 / 3600., step_sec=2.))
    monkeypatch.setattr(sim, "_plot_results", lambda *args: None)
    intervals = []
    original = sim.eps.step
    def record(dt):
        intervals.append(dt)
        original(dt)
    monkeypatch.setattr(sim.eps, "step", record)
    sim.run()
    frame = pd.read_csv(tmp_path / "ANTARA_Master_telemetry.csv")
    assert intervals == [2., 2., 1.]
    assert frame.time_sec.tolist() == [0., 2., 4.]
    assert frame.state_time_sec.tolist() == [2., 4., 5.]
    assert frame.interval_duration_sec.tolist() == intervals
