"""Bounded payload summaries retain the former sample-weighted results."""
import pickle

import numpy as np
import pytest

from simulations.helper.pass_statistics import PassStatistics


def record(i):
    return {
        "pitch_angle_deg": 20. + np.sin(i), "ew_factor": 1. + i % 7 / 10.,
        "livetime_fraction": .8 + i % 5 / 100.,
        "measured_flux": np.arange(60) + i % 11,
        "ap9_flux": np.arange(60) / 2. + i % 3,
        "sigma_fraction": 1. / np.sqrt(np.arange(60) + i % 5 + 1.),
        "counts_per_bin": np.arange(60) * (i % 2),
        "cell_id": str(i % 3), "event_type": "Type1",
        "region": "core" if i % 2 else "edge",
    }


def test_online_statistics_match_batch_and_pickle_continuation():
    records = [record(i) for i in range(1001)]
    stats = PassStatistics()
    for i, row in enumerate(records):
        stats.add(row)
        if i == 499:
            stats = pickle.loads(pickle.dumps(stats))
    result = stats.summary()
    for key, column in (("mean_pitch_angle_deg", "pitch_angle_deg"),
                        ("mean_ew_factor", "ew_factor"),
                        ("mean_livetime_fraction", "livetime_fraction"),
                        ("mean_measured_flux", "measured_flux"),
                        ("mean_ap9_flux", "ap9_flux"),
                        ("mean_sigma_fraction", "sigma_fraction")):
        np.testing.assert_allclose(result[key], np.mean([r[column] for r in records], axis=0), rtol=1e-13, atol=1e-14)
    assert result["std_pitch_angle_deg"] == pytest.approx(np.std([r["pitch_angle_deg"] for r in records]), rel=1e-13)
    np.testing.assert_array_equal(result["total_counts_per_bin"], np.sum([r["counts_per_bin"] for r in records], axis=0))
    assert result["cell_ids"] == ["0", "1", "2"]
    assert result["region_histogram"] == {"core": 500, "edge": 501}


def test_storage_is_independent_of_sample_count():
    stats = PassStatistics()
    with pytest.raises(ValueError, match="empty"):
        stats.summary()
    sample = record(0)
    for _ in range(100):
        stats.add(sample)
    small = len(pickle.dumps(stats))
    for _ in range(10000):
        stats.add(sample)
    # Only the encoded integer counter can grow, not retained sample arrays.
    assert len(pickle.dumps(stats)) <= small + 16
    assert len(stats.totals) == 4
    assert sum(a.size for a in stats.totals.values()) == 240
    assert stats.count == 10100
