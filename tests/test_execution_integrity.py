"""Short-run accounting and restart regressions; never launch annual scenarios."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import orekit
import pandas as pd
import pytest

orekit.initVM()
import ANTARA_Science_Sim as science


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setattr(science, "results_path", lambda *parts: tmp_path / parts[-1])
    return science.MissionConfig(sim_duration_hours=6 / 3600, step_sec=2., coarse_step_sec=2.,
                                 no_plots=True, rng_seed=17, output_tag="reference")


def telemetry(tmp_path, tag):
    return pd.read_csv(tmp_path / f"{tag}_ANTARA_Science_telemetry.csv")


def test_peak_rejection_precedes_payload_and_eps(config, monkeypatch, tmp_path):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    def guidance(*args, **kwargs):
        sim.adcs.last_commanded_mode = "SLEWING"
    def reject(*args):
        raise FloatingPointError("unresolved wheel crossing")
    calls = []
    monkeypatch.setattr(sim, "_update_attitude_guidance", guidance)
    monkeypatch.setattr(sim.adcs, "validate_slew_momentum_peak", reject)
    monkeypatch.setattr(sim.payload, "generate_data", lambda *a, **k: calls.append("payload"))
    monkeypatch.setattr(sim.eps, "compute_solar_power", lambda *a, **k: calls.append("solar"))
    monkeypatch.setattr(sim.eps, "step", lambda *a, **k: calls.append("eps"))
    initial_storage = sim.science_storage_mb
    with pytest.raises(FloatingPointError, match="wheel crossing"):
        sim.run_steps(1)
    assert calls == []
    assert sim.science_storage_mb == initial_storage
    assert sim.time_sec == 0.
    with pytest.raises(RuntimeError, match="incomplete"):
        sim.save_checkpoint(tmp_path / "invalid.pkl")


def test_exact_endpoint_and_segmented_execution(config, tmp_path):
    reference = science.ANTARAScienceSimulator(config, skip_leop=True)
    reference.run()
    assert reference.time_sec == 6.
    segmented = science.ANTARAScienceSimulator(replace(config, output_tag="segmented"), skip_leop=True)
    segmented.run_steps(1)
    assert segmented.time_sec == 2.
    segmented.run_steps(1)
    assert segmented.time_sec == 4.
    segmented.run()
    assert segmented.time_sec == 6.
    a, b = telemetry(tmp_path, "reference"), telemetry(tmp_path, "segmented")
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-10, atol=1e-10)
    assert list(a.time_sec) == [0., 2., 4., 6.]
    assert a.interval_duration_sec.sum() == 6.
    assert a.interval_duration_sec.iloc[-1] == 0.
    before = a.copy()
    segmented.run()
    pd.testing.assert_frame_equal(before, telemetry(tmp_path, "segmented"))


def test_checkpoint_retains_pending_state_and_continues(config, tmp_path):
    original = science.ANTARAScienceSimulator(config, skip_leop=True)
    original.run_steps(1)
    original._relay_tracking_valid_sec = 3.5
    original.payload._cell_total_counts["probe"] = 42.
    original.payload._cell_pass_counts["probe"] = 2
    original.payload._cell_histograms["probe"] = np.arange(64, dtype=float)
    original._saa_pass_index = 7
    path = tmp_path / "state.pkl"
    original.save_checkpoint(path)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(path, replace(config, output_tag="resumed"))
    assert resumed.time_sec == 2.
    assert resumed._relay_tracking_valid_sec == 3.5
    assert resumed._saa_pass_index == 7
    assert resumed.payload._cell_total_counts == original.payload._cell_total_counts
    np.testing.assert_array_equal(resumed.payload._cell_histograms["probe"], original.payload._cell_histograms["probe"])
    np.testing.assert_array_equal(resumed.current_attitude_command.quaternion, original.current_attitude_command.quaternion)
    original.run()
    resumed.run()
    pd.testing.assert_frame_equal(telemetry(tmp_path, "reference"), telemetry(tmp_path, "resumed"),
                                  check_exact=False, rtol=1e-8, atol=1e-8)


def test_payload_per_bin_uncertainty_and_merge_accounting():
    config = science.MissionConfig(enable_periphery_bin_merge=True)
    env = SimpleNamespace(PITCH_EXPONENT=2., compute_flux=lambda *args, **kwargs: {
        "saa_depth": 0.01, "pitch_angle_deg": 0., "ew_factor": 1.,
        "measured_flux": np.ones(5), "ap9_flux": np.ones(5)})
    payload = science.PayloadModel(config, env)
    _, result = payload.generate_data(2., -25., -50., 600., np.ones(3), np.ones(3), np.ones(3))
    counts = result["counts_per_bin"]
    histogram = payload._cell_histograms[result["cell_id"]]
    np.testing.assert_allclose(histogram, counts)
    np.testing.assert_allclose(result["sigma_fraction"], 1. / np.sqrt(np.maximum(histogram, 1.)))


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_step_count(config, steps):
    sim = science.ANTARAScienceSimulator(config)
    with pytest.raises(ValueError):
        sim.run_steps(steps)


def test_moving_relay_los_with_real_frames(config):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    orbit = sim._build_orbit_state(sim._propagate_state(sim.start_date), 0.)
    station_config = replace(config, gs_lat_deg=orbit.latitude_deg, gs_lon_deg=orbit.longitude_deg)
    sim.ground_station = science.GroundStationModel(station_config, sim.earth, sim.earth_frame)
    sim.fsm.current_mode = sim.target_mode = "RELAY"
    errors = []
    for seconds in range(0, 62, 2):
        orbit = sim._build_orbit_state(sim._propagate_state(sim.start_date.shiftedBy(float(seconds))), float(seconds))
        visible = sim._ground_station_geometry(orbit).visible
        assert visible
        sim.adcs.set_time(float(seconds))
        sim._update_attitude_guidance("RELAY", orbit, not orbit.in_eclipse, visible, 24., 2., float(seconds))
        sim._validate_attitude_metrics("RELAY", orbit, visible)
        errors.append(sim._compute_relay_pointing_error_deg(orbit, "+Y", visible))
        assert np.linalg.norm(sim.adcs.current_quaternion) == pytest.approx(1., abs=1e-12)
        assert sim.adcs.current_slew_rate_deg_sec <= config.max_slew_rate_deg_sec
    assert max(errors) <= config.relay_pointing_hard_limit_deg
    print(f"Moving-LOS worst error: {max(errors):.9f} deg; hard limit: {config.relay_pointing_hard_limit_deg} deg")


def test_payload_accumulation_and_final_flush(config, tmp_path):
    sim = science.ANTARAScienceSimulator(config, skip_leop=True)
    sim.payload.proton_env = SimpleNamespace(PITCH_EXPONENT=2., compute_flux=lambda *a, **kw: {
        "saa_depth": .01, "pitch_angle_deg": 0., "ew_factor": 1.,
        "measured_flux": np.ones(5), "ap9_flux": np.ones(5)})
    payload = sim.payload
    _, first = payload.generate_data(1., -25., -50., 600., np.ones(3), np.ones(3), np.ones(3))
    _, second = payload.generate_data(2., -25., -50., 600., np.ones(3), np.ones(3), np.ones(3))
    np.testing.assert_allclose(second["counts_per_bin"], np.asarray(first["counts_per_bin"]) * 2.)
    cell = second["cell_id"]
    np.testing.assert_allclose(payload._pending_bin_counts[cell], payload._cell_histograms[cell])
    np.testing.assert_allclose(second["sigma_fraction"], 1. / np.sqrt(np.maximum(payload._cell_histograms[cell], 1.)))
    sim.run()
    df = telemetry(tmp_path, "reference")
    assert df.science_data_mb.iloc[-1] > 0.
    assert df.science_data_mb.sum() == pytest.approx(payload.total_science_data_generated_mb)
    assert df.storage_mb.iloc[-1] == pytest.approx(config.initial_storage_mb + payload.total_science_data_generated_mb)
    assert not payload._pending_bin_counts


def test_short_relay_acquisition_timestep_convergence(config, tmp_path):
    """Real moving station/propagation, FSM, guidance, EPS and downlink together."""
    outcomes = []
    for step in (4., 2., 1.):
        tag = f"relay_{int(step)}"
        cfg = replace(config, sim_duration_hours=120 / 3600, step_sec=step,
                      coarse_step_sec=step, output_tag=tag, initial_battery_wh=24.,
                      sun_pointing_tracking_noise_deg=0.)
        sim = science.ANTARAScienceSimulator(cfg, skip_leop=True)
        orbit = sim._build_orbit_state(sim._propagate_state(sim.start_date), 0.)
        station_cfg = replace(cfg, gs_lat_deg=orbit.latitude_deg,
                              gs_lon_deg=orbit.longitude_deg)
        sim.ground_station = science.GroundStationModel(station_cfg, sim.earth, sim.earth_frame)
        # A declared initial backlog allows actual downlink; no production model
        # or transition is mocked, and acquisition starts in SUN_POINTING.
        sim.science_storage_mb = 100.
        sim.run()
        frame = telemetry(tmp_path, tag)
        print("Relay acquisition:", step, sim.adcs.is_slewing, sim.adcs.settling,
              sim.adcs.current_slew_rate_deg_sec, sim.current_relay_pointing_error_deg,
              getattr(sim, "_relay_tracking_valid_sec", 0.))
        assert {"SLEWING", "RELAY"} <= set(frame["mode"])
        assert sim.communications.total_downlinked_mb > 0.
        assert sim.science_storage_mb + sim.communications.total_downlinked_mb == pytest.approx(100.)
        assert frame.interval_duration_sec.sum() == pytest.approx(120.)
        assert frame.state_time_sec.iloc[-1] == pytest.approx(120.)
        assert frame.interval_duration_sec.max() <= step
        assert np.isfinite(frame.battery_wh).all()
        outcomes.append((step, sim.eps.battery_wh, sim.communications.total_downlinked_mb))
    print("Relay convergence (step s, battery Wh, downlink MB):", outcomes)
    # Bound discretization sensitivity over this short, high-elevation contact.
    # These are regression budgets, not flight pointing or calibration claims.
    assert max(x[1] for x in outcomes) - min(x[1] for x in outcomes) < .1
    assert max(x[2] for x in outcomes) - min(x[2] for x in outcomes) < config.downlink_rate_mbps / 8 * 8
    for column in (1, 2):
        assert abs(outcomes[2][column] - outcomes[1][column]) < abs(outcomes[1][column] - outcomes[0][column])


@pytest.mark.parametrize("module_name", ["ANTARA_Science_Sim", "ANTARA_Master"])
def test_eps_uses_interval_slew_fraction(module_name):
    import importlib
    module = importlib.import_module(module_name)
    cfg = module.MissionConfig()
    eps = module.EPSModel(cfg)
    idle = eps.compute_load("RELAY", False, False)
    moving = eps.compute_load("RELAY", True, False)
    # Endpoint state must not override the measured interval duty fraction.
    assert eps.compute_load("RELAY", False, False, slew_fraction=.25) == pytest.approx(idle + .25 * (moving - idle))
    assert eps.compute_load("RELAY", True, False, slew_fraction=.25) == pytest.approx(idle + .25 * (moving - idle))
    for fraction in (-.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="duty fraction"):
            eps.compute_load("RELAY", False, False, slew_fraction=fraction)


@pytest.mark.parametrize("duration", [120., 400.])
def test_short_real_saa_detection_accounting(config, tmp_path, duration):
    """No mocked SAA, flux, guidance or FSM; begin on an orbit inside the SAA."""
    cfg = replace(config, sim_duration_hours=duration / 3600, initial_battery_wh=24.,
                  true_anomaly_deg=200., raan_deg=291.8613431778403,
                  sun_pointing_tracking_noise_deg=0.)
    sim = science.ANTARAScienceSimulator(cfg, skip_leop=True)
    orbit = sim._build_orbit_state(sim._propagate_state(sim.start_date), 0.)
    assert sim.saa.is_inside(orbit.latitude_deg, orbit.longitude_deg)
    sim.run()
    frame = telemetry(tmp_path, "reference")
    assert "DETECTION" in set(frame["mode"])
    assert sim.payload.pass_science_records
    assert sum(sum(p["total_counts_per_bin"]) for p in sim.payload.pass_science_records) > 0.
    generated = sim.payload.total_science_data_generated_mb
    assert generated > 0.
    assert frame.science_data_mb.sum() == pytest.approx(generated)
    assert sim.science_storage_mb + sim.communications.total_downlinked_mb == pytest.approx(generated)
    assert sum(p["data_collected_mb"] for p in sim.saa.completed_passes) == pytest.approx(generated)
    assert not sim.payload._pending_bin_counts
    assert frame.interval_duration_sec.sum() == pytest.approx(duration)
    if duration == 400.:
        final = sim._build_orbit_state(sim._propagate_state(sim.start_date.shiftedBy(duration)), duration)
        assert not sim.saa.is_inside(final.latitude_deg, final.longitude_deg)
        assert len(sim.saa.completed_passes) == 1
        assert sim.saa.completed_passes[0]["exit_time_sec"] < duration


def test_real_station_handover_conserves_contact_credit(config, monkeypatch, tmp_path):
    cfg = replace(config, sim_duration_hours=240 / 3600, initial_battery_wh=24.,
                  sun_pointing_tracking_noise_deg=0.)
    probe = science.ANTARAScienceSimulator(cfg, skip_leop=True)
    coordinates = []
    for seconds in (0., 180.):
        orbit = probe._build_orbit_state(probe._propagate_state(probe.start_date.shiftedBy(seconds)), seconds)
        coordinates.append((orbit.latitude_deg, orbit.longitude_deg))
    cfg = replace(cfg, gs_lat_deg=coordinates[0][0], gs_lon_deg=coordinates[0][1],
                  gs2_enabled=True, gs2_lat_deg=coordinates[1][0], gs2_lon_deg=coordinates[1][1],
                  gs2_alt_m=cfg.gs_alt_m, gs2_elevation_mask_deg=cfg.gs_elevation_mask_deg)
    sim = science.ANTARAScienceSimulator(cfg, skip_leop=True)
    stations = [sim.ground_station, sim.ground_station_2]
    sim.science_storage_mb = 100.
    sim.run_steps(40)
    checkpoint = tmp_path / "before_handover.pkl"
    sim.save_checkpoint(checkpoint)
    assert sim._relay_station_id == "primary"
    initial_credits = [station.current_contact["downlinked_mb"] for station in stations]
    selections = []
    original = sim._record_ground_station_contacts
    def record(geometry, time_sec, date, downlinked_mb):
        selections.append((time_sec, geometry.selected_station, downlinked_mb, sim.fsm.current_mode))
        if downlinked_mb > 0.:
            assert not sim.adcs.saturation_hold_active
            assert sim.adcs.settled_time_sec >= cfg.minimum_settled_time_before_tx_sec
            assert sim.current_relay_pointing_error_deg <= cfg.relay_pointing_valid_deg
        return original(geometry, time_sec, date, downlinked_mb)
    monkeypatch.setattr(sim, "_record_ground_station_contacts", record)
    sim.run()
    frame = telemetry(tmp_path, "reference")
    credits = [sum(c["downlinked_mb"] for c in station.completed_contacts) for station in stations]
    assert selections[0][1] is stations[0]
    assert selections[-1][1] is stations[1]
    for station, credit, initial in zip(stations, credits, initial_credits):
        assert credit == pytest.approx(initial + sum(amount for _, selected, amount, _ in selections if selected is station))
    first_secondary = next(row for row in selections if row[1] is stations[1])
    assert first_secondary[2] == 0.
    assert first_secondary[3] == "SLEWING"
    assert all(credit > 0. for credit in credits)
    assert sum(credits) == pytest.approx(sim.communications.total_downlinked_mb)
    assert sim.science_storage_mb + sum(credits) == pytest.approx(100.)
    assert frame.interval_duration_sec.sum() == pytest.approx(240.)
    print("Handover at (s):", first_secondary[0], "station credits (MB):", credits)
    resumed = science.ANTARAScienceSimulator.load_checkpoint(checkpoint, replace(cfg, output_tag="handover_resume"))
    assert resumed._relay_station_id == "primary"
    resumed.run()
    pd.testing.assert_frame_equal(frame, telemetry(tmp_path, "handover_resume"),
                                  check_exact=False, rtol=1e-8, atol=1e-8)
