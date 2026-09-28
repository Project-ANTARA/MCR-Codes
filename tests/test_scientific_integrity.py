"""Regressions for frame/date integrity and fail-fast dynamics."""

from datetime import datetime
from types import SimpleNamespace
import pickle

import numpy as np
import orekit
import pytest

orekit.initVM()
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.utils import IERSConventions

from simulations.ditl import ANTARA_Science_Sim as science
from simulations.helper.antara_models import ADCSModel, MissionConfig
from simulations.helper.paths import init_orekit_data


@pytest.fixture(params=["shared", "master"])
def wheel_model(request):
    if request.param == "master":
        from simulations.legacy import ANTARA_Master as model
        return model.ADCSModel(model.MissionConfig())
    return ADCSModel(MissionConfig())


@pytest.mark.parametrize("case", ["nan_torque", "inf_torque", "shape", "negative_dt", "nan_dt",
                                  "negative_unload", "inf_projection", "negative_field",
                                  "invalid_inertia", "invalid_capacity", "invalid_recovery",
                                  "impulse_overflow", "rpm_overflow", "unload_overflow"])
def test_wheel_failure_is_atomic(wheel_model, case):
    adcs = wheel_model
    torque, dt, unload, projection = np.array([1e-4, 0., 0.]), 1., 1., 1.
    if case == "nan_torque": torque[0] = np.nan
    elif case == "inf_torque": torque[0] = np.inf
    elif case == "shape": torque = np.zeros(2)
    elif case == "negative_dt": dt = -1.
    elif case == "nan_dt": dt = np.nan
    elif case == "negative_unload": unload = -1.
    elif case == "inf_projection": projection = np.inf
    elif case == "negative_field": adcs.config.earth_B_field_T = -1.
    elif case == "invalid_inertia": adcs.config.wheel_inertia_kgm2 = 0.
    elif case == "invalid_capacity": adcs.config.wheel_max_momentum_nms = np.nan
    elif case == "invalid_recovery": adcs.config.wheel_momentum_recovery_fraction = 1.
    elif case == "impulse_overflow": torque[0], dt = 1e308, 10.
    elif case == "rpm_overflow": adcs.config.wheel_inertia_kgm2 = 1e-320
    elif case == "unload_overflow": unload, projection = 1e308, 1e308
    before = pickle.dumps(vars(adcs))
    with pytest.raises((ValueError, FloatingPointError)):
        adcs.update_wheel_momentum(torque, dt, unload, projection)
    assert pickle.dumps(vars(adcs)) == before


def test_invalid_momentum_cannot_clear_saturation(wheel_model):
    adcs = wheel_model
    adcs.wheel_momentum_nms[0] = np.nan
    adcs.is_saturated = True
    adcs.saturation_hold_active = True
    before = pickle.dumps(vars(adcs))
    with pytest.raises(FloatingPointError):
        adcs.check_saturation()
    assert pickle.dumps(vars(adcs)) == before


def test_wheel_unloading_and_zero_time(wheel_model):
    adcs = wheel_model
    adcs.wheel_momentum_nms[:] = [1e-5, -1e-5, 0.]
    before = pickle.dumps(vars(adcs))
    adcs.update_wheel_momentum(np.ones(3), 0., 1., 1.)
    assert pickle.dumps(vars(adcs)) == before
    adcs.update_wheel_momentum(np.zeros(3), 100., 1., 1.)
    np.testing.assert_array_equal(adcs.wheel_momentum_nms, np.zeros(3))
    np.testing.assert_array_equal(adcs.wheel_speed_rpm_estimate, np.zeros(3))


@pytest.mark.parametrize("year,expected", [
    (2000.0, datetime(2000, 1, 1)),
    (2000.25, datetime(2000, 4, 1, 12)),
    (2001.5, datetime(2001, 7, 2, 12)),
])
def test_decimal_igrf_epoch(monkeypatch, year, expected):
    dates = []
    def query(lon, lat, alt, date):
        dates.append(date)
        return np.array([1.]), np.array([2.]), np.array([3.])
    monkeypatch.setattr(science.ppigrf, "igrf", query)
    science.IGRFModel(year).get_ned_field_tesla(10., 20., 600.)
    assert dates == [expected]


@pytest.mark.parametrize("seconds", [0., 21600.])
def test_igrf_rotates_earth_fixed_vector_at_query_date(monkeypatch, seconds):
    init_orekit_data()
    utc = TimeScalesFactory.getUTC()
    date = AbsoluteDate(2020, 6, 1, 0, 0, 0., utc).shiftedBy(seconds)
    earth = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    inertial = FramesFactory.getEME2000()
    dates = []
    def query(lon, lat, alt, query_date):
        dates.append(query_date)
        # At lat=lon=0, East points along Earth-fixed +Y.
        return np.array([10000.]), np.array([0.]), np.array([0.])
    monkeypatch.setattr(science.ppigrf, "igrf", query)
    actual = science.IGRFModel(2000.).get_inertial_field_tesla(
        0., 0., 600., date, earth, inertial)
    expected = earth.getTransformTo(inertial, date).getRotation().applyTo(Vector3D.PLUS_J)
    np.testing.assert_allclose(actual, np.array([expected.getX(), expected.getY(), expected.getZ()]) * 1e-5,
                               rtol=1e-13, atol=1e-18)
    assert np.linalg.norm(actual) == pytest.approx(1e-5)
    assert dates == [datetime(2020, 6, 1, int(seconds / 3600))]
    assert not np.allclose(actual, [0., 1e-5, 0.], atol=1e-9)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_invalid_rates_cannot_report_detumble_success(invalid):
    adcs = ADCSModel(MissionConfig())
    adcs.body_rate_deg_sec = np.array([invalid, 0., 0.])
    with pytest.raises(ValueError, match="finite"):
        adcs.step_detumble(np.array([2e-5, 0., 0.]), .5)
    assert not adcs.detumble_complete
    assert not np.isfinite(adcs.body_rate_deg_sec[0])


def test_nonfinite_integrated_rate_is_not_zeroed():
    adcs = ADCSModel(MissionConfig())
    before = adcs.body_rate_deg_sec.copy()
    with pytest.raises(FloatingPointError):
        adcs._integrate_body_rates(np.array([1e308, 0., 0.]), 1e308)
    np.testing.assert_array_equal(adcs.body_rate_deg_sec, before)
    assert not adcs.detumble_complete


@pytest.mark.parametrize("field,step", [([np.nan, 0., 0.], .5), ([0., 0., 0.], np.inf),
                                       ([0., 0., 0.], -1.)])
def test_invalid_detumble_inputs_rejected(field, step):
    adcs = ADCSModel(MissionConfig())
    with pytest.raises(ValueError, match="finite"):
        adcs.step_detumble(np.array(field), step)
    assert not adcs.detumble_complete


def test_valid_rigid_body_update_preserved():
    adcs = ADCSModel(MissionConfig())
    adcs.body_rate_deg_sec = np.array([1., 2., 3.])
    torque = np.array([1e-6, -2e-6, 3e-6])
    omega = np.radians(adcs.body_rate_deg_sec)
    inertia = adcs.inertia_tensor_kgm2
    acceleration = (torque - np.cross(omega, inertia * omega)) / inertia
    adcs._integrate_body_rates(torque, .1)
    np.testing.assert_allclose(adcs.body_rate_deg_sec, np.degrees(omega + .1 * acceleration))
    np.testing.assert_allclose(adcs.body_accel_deg_sec2, np.degrees(acceleration))


@pytest.mark.parametrize("primary_elevation,secondary_elevation,selected", [
    (20., 50., 2), (-10., 50., 2), (50., 20., 1), (20., 20., 1), (20., -10., 1),
])
def test_relay_aims_at_selected_visible_station(primary_elevation, secondary_elevation, selected):
    def station(elevation, position):
        return SimpleNamespace(
            get_elevation=lambda *args: elevation, is_visible=lambda el: el >= 10.,
            frame=SimpleNamespace(getPVCoordinates=lambda *args: SimpleNamespace(getPosition=lambda: position)))
    sim = science.ANTARAScienceSimulator.__new__(science.ANTARAScienceSimulator)
    sim.ground_station = station(primary_elevation, Vector3D.PLUS_I)
    sim.ground_station_2 = station(secondary_elevation, Vector3D.PLUS_J)
    sim.inertial_frame = None
    orbit = SimpleNamespace(absolute_date=None, sat_position_inertial=Vector3D.ZERO,
                            sat_position_earth_frame=Vector3D.ZERO)
    expected = [1., 0., 0.] if selected == 1 else [0., 1., 0.]
    np.testing.assert_allclose(sim._ground_station_vector_unit(orbit), expected)


def test_per_station_contacts_preserve_aos_credit_and_visibility():
    def station(elevation):
        station = science.GroundStationModel.__new__(science.GroundStationModel)
        station.config = MissionConfig()
        station.current_contact = None
        station.completed_contacts = []
        station.get_elevation = lambda *args: elevation[0]
        return station
    primary_el, secondary_el = [-10.], [50.]
    sim = science.ANTARAScienceSimulator.__new__(science.ANTARAScienceSimulator)
    sim.ground_station, sim.ground_station_2 = station(primary_el), station(secondary_el)
    sim.previous_gs_visible = sim.previous_gs_visible_2 = False
    date = SimpleNamespace(toString=lambda: "test epoch")
    orbit = SimpleNamespace(absolute_date=date, sat_position_earth_frame=None)
    def record(t, downlink):
        sim._record_ground_station_contacts(sim._ground_station_geometry(orbit), t, date, downlink)
    record(0., 2.)
    assert sim.previous_gs_visible is False
    assert sim.previous_gs_visible_2 is True
    assert sim.ground_station_2.current_contact["downlinked_mb"] == 2.
    primary_el[0] = 20.
    record(2., 3.)
    assert sim.ground_station.current_contact["aos_time_sec"] == 2.
    assert sim.ground_station.current_contact["max_el_deg"] == 20.
    assert sim.ground_station_2.current_contact["downlinked_mb"] == 5.
    primary_el[0] = 60.
    record(4., 4.)
    primary_el[0] = secondary_el[0] = -10.
    record(6., 0.)
    primary_contact = sim.ground_station.completed_contacts[0]
    secondary_contact = sim.ground_station_2.completed_contacts[0]
    assert primary_contact["max_el_deg"] == 60.
    assert primary_contact["downlinked_mb"] == 4.
    assert secondary_contact["max_el_deg"] == 50.
    assert secondary_contact["downlinked_mb"] == 5.
    assert not sim._ground_station_geometry(orbit).visible


def test_single_station_configuration():
    sim = science.ANTARAScienceSimulator.__new__(science.ANTARAScienceSimulator)
    sim.ground_station = SimpleNamespace(get_elevation=lambda *args: 20., is_visible=lambda el: True)
    sim.ground_station_2 = None
    orbit = SimpleNamespace(absolute_date=None, sat_position_earth_frame=None)
    geometry = sim._ground_station_geometry(orbit)
    assert geometry.selected_station is sim.ground_station
    assert geometry.visible and not geometry.secondary_visible


def test_missing_igrf_fallback_is_also_rotated(monkeypatch):
    init_orekit_data()
    date = AbsoluteDate(2020, 6, 1, 0, 0, 0., TimeScalesFactory.getUTC())
    earth = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    inertial = FramesFactory.getEME2000()
    model = science.IGRFModel(2000.)
    model._available = False
    monkeypatch.setattr(model, "_dipole_ned_tesla", lambda *args: np.array([0., 1e-5, 0.]))
    actual = model.get_inertial_field_tesla(0., 0., 600., date, earth, inertial)
    earth_vector = inertial.getTransformTo(earth, date).getRotation().applyTo(
        Vector3D(*[float(value) for value in actual]))
    np.testing.assert_allclose([earth_vector.getX(), earth_vector.getY(), earth_vector.getZ()],
                               [0., 1e-5, 0.], atol=1e-18)
