"""Analytic command-profile checks, independent of mission duration."""
import math

import numpy as np
import pytest

from simulations.helper.slew_profile import slew_step
from simulations.helper.antara_models import ADCSModel, MissionConfig


@pytest.mark.parametrize("dt", [.01, .1, 1., 2., 20.])
@pytest.mark.parametrize("distance", [1., 30., 90.])
def test_profile_matches_analytic_duration_and_integral(dt, distance):
    x, v, elapsed = 0., 0., 0.
    expected_duration = (2 * math.sqrt(distance) if distance <= 25.
                         else 10. + (distance - 25.) / 5.)
    for _ in range(10000):
        direction = 1. if distance >= x else -1.
        dx, final_v, moving = slew_step(abs(distance - x), v * direction, 5., 1., dt)
        assert abs(final_v - v * direction) <= dt + 1e-12
        assert abs(final_v) <= 5. + 1e-12
        x += dx * direction
        v = final_v * direction
        elapsed += moving
        if final_v == 0.:
            break
    else:
        pytest.fail("Profile did not finish")
    assert x == pytest.approx(distance, abs=1e-10)
    assert elapsed == pytest.approx(expected_duration, abs=1e-8)


def test_half_acceleration_displacement_and_partial_braking():
    assert slew_step(30., 0., 5., 1., 1.) == pytest.approx((.5, 1., 1.))
    # Target is too close: continue moving while braking, rather than clipping.
    assert slew_step(.1, 2., 5., 1., .5) == pytest.approx((.875, 1.5, .5))
    assert slew_step(1., -2., 5., 1., 1.) == pytest.approx((-1.5, -1., 1.))
    x, v, moving = slew_step(.1, 2., 5., 1., 20.)
    assert (x, v) == pytest.approx((.1, 0.))
    assert moving == pytest.approx(2. + 2. * math.sqrt(1.9))


@pytest.mark.parametrize("values", [(1., 0., 0., 1., 1.), (1., 0., 5., 0., 1.),
                                    (-1., 0., 5., 1., 1.), (1., float('nan'), 5., 1., 1.),
                                    (1., 0., 5., 1., -1.)])
def test_invalid_profile(values):
    with pytest.raises(ValueError):
        slew_step(*values)


def test_acceptance_band_does_not_erase_velocity():
    adcs = ADCSModel(MissionConfig())
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.target_quaternion = adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), .1)
    adcs.current_slew_rate_deg_sec = 2.
    adcs.advance(.1)
    assert adcs.current_slew_rate_deg_sec == pytest.approx(1.9)
    assert adcs.is_slewing
    assert not adcs.settling


def test_coarse_interval_includes_braking_and_settling():
    config = MissionConfig()
    adcs = ADCSModel(config)
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    target = adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.)
    adcs.command_attitude(target, "SLEWING", "test", force_update=True)
    adcs.advance(11. + config.settling_time_sec + 3.)
    assert adcs.compute_slew(adcs.current_quaternion, target)[0] < 1e-10
    assert adcs.current_slew_rate_deg_sec == 0.
    assert adcs.total_slew_time_sec == pytest.approx(11.)
    assert not adcs.settling and not adcs.is_slewing
    assert adcs.settled_time_sec == pytest.approx(3.)
    # Net acceleration impulse is zero over a rest-to-rest slew.
    np.testing.assert_allclose(adcs.estimate_torque_vector(), np.zeros(3), atol=1e-15)


def test_same_axis_target_reversal_brakes_before_reversing():
    adcs = ADCSModel(MissionConfig())
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    axis = np.array([1., 0., 0.])
    adcs.command_attitude(adcs.quaternion_from_axis_angle(axis, 30.), "SLEWING", "out", force_update=True)
    adcs.advance(2.)
    before = adcs.current_quaternion.copy()
    adcs.command_attitude(adcs.quaternion_from_axis_angle(axis, -30.), "RELAY", "back", force_update=True)
    adcs.advance(.1)
    delta = adcs.compute_delta_quaternion(before, adcs.current_quaternion)
    assert delta[1] > 0  # Still moving outward while braking.
    assert adcs.current_slew_rate_deg_sec == pytest.approx(1.9)


def test_settlement_credits_only_remaining_interval():
    adcs = ADCSModel(MissionConfig())
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.settling = True
    adcs.settling_time_remaining_sec = .25
    adcs.advance(1.)
    assert not adcs.settling
    assert adcs.settled_time_sec == pytest.approx(.75)


def test_zero_time_preserves_signed_rate():
    assert slew_step(0., -2., 5., 1., 0.) == (0., -2., 0.)


@pytest.mark.parametrize("field", ["max_slew_rate_deg_sec", "max_slew_accel_deg_sec2",
                                    "wheel_max_torque_nm", "step_sec", "coarse_step_sec",
                                    "sim_duration_hours"])
@pytest.mark.parametrize("value", [0., -1., float('nan'), float('inf')])
def test_invalid_mission_limits(field, value):
    with pytest.raises(ValueError):
        MissionConfig(**{field: value})


@pytest.mark.parametrize("capacity,hidden", [(.003, True), (.006, False)])
@pytest.mark.parametrize("implementation", ["shared", "master"])
def test_hidden_peak_in_zero_net_impulse_slew(capacity, hidden, implementation):
    if implementation == "master":
        import orekit
        orekit.initVM()
        from simulations.legacy import ANTARA_Master as model
        adcs = model.ADCSModel(model.MissionConfig(wheel_max_momentum_nms=capacity))
    else:
        adcs = ADCSModel(MissionConfig(wheel_max_momentum_nms=capacity))
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.command_attitude(adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.),
                         "SLEWING", "peak probe", force_update=True)
    adcs.advance(11.)
    np.testing.assert_allclose(adcs.estimate_torque_vector(), 0., atol=1e-15)
    before = adcs.wheel_momentum_nms.copy()
    if hidden:
        with pytest.raises(FloatingPointError, match="inside timestep"):
            ADCSModel.validate_slew_momentum_peak(adcs, 11., 1., 1.)
    else:
        ADCSModel.validate_slew_momentum_peak(adcs, 11., 1., 1.)
    np.testing.assert_array_equal(before, adcs.wheel_momentum_nms)


def test_fine_step_reaches_existing_saturation_policy():
    adcs = ADCSModel(MissionConfig(wheel_max_momentum_nms=.003))
    adcs.set_initial_quaternion(np.array([1., 0., 0., 0.]))
    adcs.command_attitude(adcs.quaternion_from_axis_angle(np.array([1., 0., 0.]), 30.),
                         "SLEWING", "resolved peak", force_update=True)
    for _ in range(4):
        adcs.advance(1.)
        adcs.validate_slew_momentum_peak(1., 0., 0.)
        adcs.update_wheel_momentum(adcs.estimate_torque_vector(), 1., 0., 0.)
        if adcs.saturation_hold_active:
            break
    assert adcs.saturation_hold_active
    assert adcs.saturation_event_count == 1


def test_executed_torque_history_excludes_future_profile():
    phases = []
    slew_step(30., 0., 5., 1., 2., phases_out=phases)
    assert phases == [(1., 2.)]
