"""Regression checks for numerical attitude geometry and science CLI controls."""

from datetime import datetime
import math

import numpy as np
import orekit
import ppigrf
import pytest

orekit.initVM()
from ANTARA_Science_Sim import (
    ADCSModel, ANTARAScienceSimulator, IGRFModel, MissionConfig,
    _config_from_args, _parse_args,
)


@pytest.fixture
def geometry():
    sim = ANTARAScienceSimulator.__new__(ANTARAScienceSimulator)
    sim.config = MissionConfig()
    sim.adcs = ADCSModel(sim.config)
    sim.current_attitude_command = None
    sim.warning_state = {}
    return sim


def test_all_geometric_self_tests(geometry):
    geometry._run_geometric_self_tests()


@pytest.mark.parametrize("angle_deg", [1e-8, .01, 179.99])
def test_operational_angles_resolve_small_and_large_offsets(geometry, monkeypatch, angle_deg):
    angle = math.radians(angle_deg)
    sine, cosine = math.sin(angle), math.cos(angle)
    monkeypatch.setattr(geometry, "_velocity_vector_unit", lambda orbit: np.array([cosine, sine, 0.]))
    monkeypatch.setattr(geometry, "_ground_station_vector_unit", lambda orbit: np.array([sine, cosine, 0.]))
    monkeypatch.setattr(geometry, "_sun_vector_unit", lambda orbit: np.array([sine, 0., cosine]))
    values = [geometry._compute_payload_pointing_error_deg(None),
              geometry._compute_relay_pointing_error_deg(None, "+Y", True),
              geometry._compute_gs_off_boresight_deg(None, True),
              geometry._compute_sun_incidence_angle_deg(None)]
    np.testing.assert_allclose(values, angle_deg, rtol=1e-10, atol=1e-12)
    assert geometry._compute_gs_off_boresight_deg(None, False) == 180.


@pytest.mark.parametrize("sun_sign", [-1., 1.])
def test_nominal_frame_never_flips_nadir_for_sun(geometry, monkeypatch, sun_sign):
    monkeypatch.setattr(geometry, "_velocity_vector_unit", lambda orbit: np.array([1., 0., 0.]))
    monkeypatch.setattr(geometry, "_nadir_vector_unit", lambda orbit: np.array([0., 1., 0.]))
    monkeypatch.setattr(geometry, "_orbit_normal_unit", lambda orbit: np.array([0., 0., 1.]))
    monkeypatch.setattr(geometry, "_sun_vector_unit", lambda orbit: np.array([0., 0., sun_sign]))
    command = geometry._compute_default_orientation_attitude(None)
    np.testing.assert_allclose(geometry.adcs.get_body_to_inertial_matrix(command.quaternion), np.eye(3), atol=1e-14)


@pytest.mark.parametrize("offset", [0.01, 0.1, 1.0])
def test_wrong_alignment_rejected(geometry, offset):
    quaternion = geometry.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), offset)
    with pytest.raises(AssertionError, match="alignment error"):
        geometry._assert_axis_alignment("negative test", quaternion, "+X", np.array([1., 0., 0.]), 1e-6)


def test_nearly_parallel_vectors(geometry):
    # acos rounds away this real tiny angle; normalization also handles scaling.
    offset = 1e-8
    target = 7 * np.array([math.cos(math.radians(offset)), math.sin(math.radians(offset)), 0.])
    assert geometry._vector_angle_deg(np.array([3., 0., 0.]), target) == pytest.approx(offset)
    assert geometry._vector_angle_deg(target, -target) == pytest.approx(180.)


def test_identical_vectors_have_zero_angle(geometry):
    # On the local NumPy build, acos(dot(unit(v), unit(v))) reports 1.207e-6
    # degrees for these identical vectors. Do not assert a platform's rounding.
    vector = np.array([0.6904853540677682, -0.42725264633653426, 0.15853969107671423])
    assert geometry._vector_angle_deg(vector, vector) == 0.0


@pytest.mark.parametrize("scenario,skip", [("leop_24h", False), ("no_leop_24h", True), ("leop_1yr", False)])
def test_cli_overrides_presets(scenario, skip):
    config, skip_leop = _config_from_args(_parse_args([
        "--scenario", scenario, "--duration-hours", "1", "--step-sec", "2",
        "--no-plots", "--rng-seed", "17", "--coarse-step-sec", "30",
    ]))
    assert (config.sim_duration_hours, config.step_sec, config.no_plots) == (1, 2, True)
    assert config.rng_seed == 17 and config.coarse_step_sec == 30
    assert skip_leop is skip


def test_presets_and_global_defaults():
    config, _ = _config_from_args(_parse_args([]))
    assert (config.sim_duration_hours, config.step_sec) == (24, 2)
    config, _ = _config_from_args(_parse_args(["--scenario", "leop_1yr"]))
    assert config.sim_duration_hours == 8760
    config, _ = _config_from_args(_parse_args(["--scenario", "leop_1yr", "--duration-hours", "24"]))
    assert config.sim_duration_hours == 24


@pytest.mark.parametrize("scenario", ["leop_24h", "no_leop_24h"])
def test_ci_uses_fixed_two_second_steps(scenario):
    config, _ = _config_from_args(_parse_args([
        "--scenario", scenario, "--duration-hours", "1", "--step-sec", "2", "--no-plots",
    ]))
    assert config.step_sec == config.coarse_step_sec == 2


def test_igrf_altitude_units_and_ned_order():
    model = IGRFModel(2025.)
    east, north, up = ppigrf.igrf(20., 10., 600., datetime(2025, 1, 1))
    actual = model.get_ned_field_tesla(10., 20., 600.)
    assert actual.shape == (3,) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([north.item(), east.item(), -up.item()]) * 1e-9, rtol=1e-12, atol=0)


def test_tracking_updates_preserve_slew_rate(geometry):
    adcs = geometry.adcs
    axis = np.array([0., 0., 1.])
    adcs.command_attitude(adcs.quaternion_from_axis_angle(axis, 60.), "RELAY", "acquire", force_update=True)
    adcs.advance(2.)
    rate = adcs.current_slew_rate_deg_sec
    assert rate > 0
    adcs.command_attitude(adcs.quaternion_from_axis_angle(axis, 62.), "RELAY", "track", force_update=True)
    assert adcs.current_slew_rate_deg_sec == rate


def test_slew_timer_does_not_override_unfinished_attitude(geometry):
    from types import SimpleNamespace
    geometry.fsm = SimpleNamespace(current_mode="SLEWING")
    geometry.adcs.is_slewing = True
    assert not geometry._complete_slew_transition(10000.)
    geometry.adcs.is_slewing = False
    geometry.adcs.settling = True
    assert not geometry._complete_slew_transition(10000.)
    geometry.adcs.settling = False
    geometry.adcs.saturation_hold_active = True
    assert not geometry._complete_slew_transition(10000.)
    geometry.adcs.saturation_hold_active = False
    # Releasing a hold does not mean the pre-hold objective was reached.
    geometry.current_attitude_command = SimpleNamespace(
        quaternion=geometry.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), 30.))
    assert not geometry._complete_slew_transition(10000.)


def test_moving_relay_target_can_settle(geometry, monkeypatch):
    monkeypatch.setattr(geometry, "_compute_relay_pointing_error_deg", lambda *args: 0.1)
    for _ in range(6):
        # Each refreshed command may restart the fixed-target ADCS timer.
        geometry.adcs.settling = True
        geometry.adcs.settling_time_remaining_sec = geometry.config.settling_time_sec
        geometry._update_relay_tracking_settlement(None, True, 2.)
    assert not geometry.adcs.settling
    assert geometry.adcs.settled_time_sec == 12. - geometry.config.settling_time_sec
    monkeypatch.setattr(geometry, "_compute_relay_pointing_error_deg", lambda *args: 10.)
    geometry._update_relay_tracking_settlement(None, True, 2.)
    assert geometry._relay_tracking_valid_sec == 0.
    assert geometry.adcs.settled_time_sec == 0.


@pytest.mark.parametrize("valid_seconds,holding,offset,expected", [
    (8., False, 0.1, True), (7., False, 0.1, False),
    (8., True, 0.1, False), (8., False, 1., False),
])
def test_relay_acquisition_preserves_tracking_rate(geometry, valid_seconds, holding, offset, expected):
    from types import SimpleNamespace
    geometry.target_mode = "RELAY"
    geometry._relay_tracking_valid_sec = valid_seconds
    geometry.active_slew_event_index = None
    geometry.fsm = SimpleNamespace(current_mode="SLEWING",
                                  ALLOWED_TRANSITIONS={"SLEWING": {"RELAY"}},
                                  transition=lambda *args: True)
    geometry.adcs.is_slewing = True
    geometry.adcs.current_slew_rate_deg_sec = .01
    geometry.adcs.saturation_hold_active = holding
    geometry.adcs.target_quaternion = geometry.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), offset)
    assert geometry._complete_slew_transition(100.) is expected
    assert geometry.adcs.current_slew_rate_deg_sec == .01


@pytest.mark.parametrize("holding", [True, False])
def test_pending_slew_restored_only_after_saturation_release(geometry, monkeypatch, holding):
    from types import SimpleNamespace
    from ANTARA_Science_Sim import AttitudeCommand
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    sim = geometry
    orbit = SimpleNamespace(sat_position_inertial=Vector3D(0., -7000000., 0.),
                            sat_velocity_inertial=Vector3D(7500., 0., 0.))
    pending = sim.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), 30.)
    sim.current_attitude_command = AttitudeCommand(pending, "Resume LVLH acquisition", "")
    sim.target_mode = "SUN_POINTING"
    sim.adcs.saturation_hold_active = holding
    monkeypatch.setattr(sim, "_compute_noisy_sun_incidence_angle_deg", lambda *args: 90.)
    sim._update_attitude_guidance("SLEWING", orbit, True, False, 24., 2., 100.)
    if holding:
        np.testing.assert_allclose(sim.adcs.target_quaternion, [1., 0., 0., 0.])
        assert not sim.adcs.is_slewing
    else:
        np.testing.assert_allclose(sim.adcs.target_quaternion, pending)
        assert sim.adcs.is_slewing
        assert sim.adcs.compute_slew(sim.adcs.current_quaternion, pending)[0] > sim.config.pointing_accuracy_deg


def test_saturation_timeout_during_slew_uses_pending_mode_budget(geometry):
    from types import SimpleNamespace
    from ANTARA_Science_Sim import SpacecraftFSM
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    sim = geometry
    sim.target_mode = "SUN_POINTING"
    orbit = SimpleNamespace(sat_position_inertial=Vector3D(0., -7000000., 0.),
                            sat_velocity_inertial=Vector3D(7500., 0., 0.))
    sim.adcs.set_initial_quaternion(sim.adcs.quaternion_from_axis_angle(np.array([0., 0., 1.]), 30.))
    assert sim._hard_pointing_limit_for_mode("SLEWING") == float('inf')
    sim.adcs.saturation_hold_active = True
    limit = sim._hard_pointing_limit_for_mode("SLEWING")
    error = sim._hard_pointing_error_for_mode("SLEWING", orbit, False)
    assert limit == 5. and error == pytest.approx(30.)
    fsm = SpacecraftFSM(sim.config)
    fsm.current_mode = "SLEWING"
    fsm.target_mode = "SUN_POINTING"
    fsm.slew_end_time_sec = 60.
    fsm.set_time(100.)
    for _ in range(sim.config.fault_debounce_samples):
        mode, reason = fsm.evaluate(24., False, False, True, 100., error, limit, True)
    assert mode == "SLEWING"
    fsm.set_time(400.)
    for _ in range(sim.config.fault_debounce_samples):
        mode, reason = fsm.evaluate(24., False, False, True, 400., error, limit, True)
    assert mode == "SAFE"
    assert "Saturation hold timeout" in reason


@pytest.mark.parametrize("dt", [2., 20.])
@pytest.mark.parametrize("repeated_command", [False, True])
def test_sun_guidance_rejects_stale_out_of_band_target(geometry, monkeypatch, dt, repeated_command):
    from types import SimpleNamespace
    from ANTARA_Science_Sim import AttitudeCommand
    from org.hipparchus.geometry.euclidean.threed import Vector3D
    sim = geometry
    orbit = SimpleNamespace(sat_position_inertial=Vector3D(0., -7000000., 0.),
                            sat_velocity_inertial=Vector3D(7500., 0., 0.))
    axis = np.array([0., 0., 1.])
    sim.adcs.set_initial_quaternion(sim.adcs.quaternion_from_axis_angle(axis, 4.9))
    sim.adcs.target_quaternion = sim.adcs.quaternion_from_axis_angle(axis, 5.1)
    sim.adcs.commanded_relay_axis = "+Y"
    sim.target_mode = None
    sim._steps_since_mode_transition = sim.config.mode_transition_grace_steps
    command = AttitudeCommand(np.array([1., 0., 0., 0.]), "RAM/nadir", "", "LVLH", relay_axis_label="+Y")
    # Exercise both the ordinary deadband path and the unchanged-request shortcut.
    sim.current_attitude_command = command if repeated_command else AttitudeCommand(
        sim.adcs.target_quaternion.copy(), "RAM/nadir", "", "LVLH", relay_axis_label="+Y")
    monkeypatch.setattr(sim, "_get_attitude_command", lambda *args: command)
    monkeypatch.setattr(sim, "_compute_noisy_sun_incidence_angle_deg", lambda *args: 90.)
    assert sim._compute_sun_pointing_error_deg(orbit) == pytest.approx(4.9)
    # Old advance() snaps the last 0.2 degrees to the stale 5.1-degree target,
    # marks it settled, and trips the unchanged five-degree assertion.
    sim._update_attitude_guidance("SUN_POINTING", orbit, True, False, 24., dt, 0.)
    sim._validate_attitude_metrics("SUN_POINTING", orbit, False)
    np.testing.assert_allclose(sim.adcs.target_quaternion, command.quaternion)
    assert sim.current_sun_pointing_error_deg <= sim.config.sun_pointing_hard_limit_deg
    # A genuinely settled 5.1-degree error must still fail the diagnostic.
    sim.adcs = ADCSModel(sim.config)
    sim.adcs.set_initial_quaternion(sim.adcs.quaternion_from_axis_angle(axis, 5.1))
    sim.current_sun_pointing_error_deg = sim._compute_sun_pointing_error_deg(orbit)
    with pytest.raises(AssertionError, match="combined error .* exceeds hard limit"):
        sim._validate_attitude_metrics("SUN_POINTING", orbit, False)
    # A commanded saturation hold is off-nominal, but must actually hold its
    # inertial target. Invalid hold geometry and incorrect telemetry still fail.
    sim.adcs.saturation_hold_active = True
    sim._validate_attitude_metrics("SUN_POINTING", orbit, False)
    sim.current_sun_pointing_error_deg = 0.
    with pytest.raises(AssertionError, match="telemetry combined error mismatch"):
        sim._validate_attitude_metrics("SUN_POINTING", orbit, False)
    sim.current_sun_pointing_error_deg = sim._compute_sun_pointing_error_deg(orbit)
    sim.adcs.target_quaternion = command.quaternion
    with pytest.raises(AssertionError, match="Saturation hold.*alignment error"):
        sim._validate_attitude_metrics("SUN_POINTING", orbit, False)


def test_multirate_subsystems_advance_once_per_clock_interval(monkeypatch, tmp_path):
    import ANTARA_Science_Sim as science
    import pandas as pd
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    sim = ANTARAScienceSimulator(MissionConfig(sim_duration_hours=120./3600., no_plots=True), skip_leop=True)
    # Prediction does not affect this no-contact, no-SAA accounting fixture.
    monkeypatch.setattr(sim, "_predict_saa_ingress_time_sec", lambda *args: None)
    eps_steps, guidance_steps = [], []
    eps_step, guidance = sim.eps.step, sim._update_attitude_guidance
    def record_eps(dt):
        eps_steps.append(dt)
        return eps_step(dt)
    def record_guidance(mode, orbit, sun, gs, battery, dt, time, **kwargs):
        result = guidance(mode, orbit, sun, gs, battery, dt, time, **kwargs)
        guidance_steps.append(sim._guidance_interval_sec)
        return result
    monkeypatch.setattr(sim.eps, "step", record_eps)
    monkeypatch.setattr(sim, "_update_attitude_guidance", record_guidance)
    sim.run()
    df = pd.read_csv(tmp_path / 'ANTARA_Science_telemetry.csv')
    np.testing.assert_array_equal(eps_steps, guidance_steps)
    np.testing.assert_array_equal(eps_steps, np.diff(df.time_sec))
    assert sum(eps_steps) == sim.time_sec == 120.
    assert max(eps_steps) == sim.config.coarse_step_sec
