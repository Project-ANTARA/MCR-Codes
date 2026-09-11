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
    east, north, up = ppigrf.igrf(20., 10., 600., datetime(2025, 3, 1))
    actual = model.get_ned_field_tesla(10., 20., 600.)
    assert actual.shape == (3,) and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, np.array([north.item(), east.item(), -up.item()]) * 1e-9, rtol=1e-12, atol=0)
