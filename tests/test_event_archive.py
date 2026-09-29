"""Bounded completed-record storage and immutable checkpoint prefixes."""
import pickle

import pandas as pd
import pytest

from simulations.helper.event_archive import EventArchive, OpenEventArchive, event_sample, write_event_csv


def test_open_event_checkpoint_and_indices(tmp_path):
    events = OpenEventArchive(tmp_path / "events.jsonl")
    events.append({"start_time_sec": 0., "end_time_sec": None, "duration_sec": 0.})
    with pytest.raises(ValueError, match="Finalize"):
        events.append({"start_time_sec": 1.})
    snapshot = pickle.dumps(events)
    events.finalize(0, 2.)
    restored = pickle.loads(snapshot).fork(tmp_path / "branch.jsonl")
    assert restored[0]["end_time_sec"] is None
    restored.finalize(0, 3.)
    assert events[0]["duration_sec"] == 2.
    assert restored[0]["duration_sec"] == 3.
    restored.append({"start_time_sec": 4., "end_time_sec": None})
    assert len(restored) == 2
    restored.finalize(1, 5.)
    assert restored[-1]["duration_sec"] == 1.
    with pytest.raises(ValueError, match="current open"):
        restored.finalize(0, 6.)


def test_event_display_sample_is_bounded_and_keeps_endpoints():
    assert event_sample(iter(range(10001)), 10001, 5) == [0, 2500, 5000, 7500, 10000]


def test_missing_coordinates_are_null_but_infinity_is_rejected(tmp_path):
    archive = EventArchive(tmp_path / "events.jsonl")
    archive.append({"exit_lat_deg": float("nan")})
    assert archive[0] == {"exit_lat_deg": None}
    with pytest.raises(ValueError):
        archive.append({"exit_lat_deg": float("inf")})
    assert len(archive) == 1


def test_snapshot_forks_only_saved_prefix(tmp_path):
    archive = EventArchive(tmp_path / "events.jsonl")
    archive.append({"value": 1})
    snapshot = pickle.dumps(archive)
    archive.append({"value": 2})
    restored = pickle.loads(snapshot)
    with pytest.raises(RuntimeError, match="Fork"):
        restored.append({"value": 3})
    branch = restored.fork(tmp_path / "branch.jsonl")
    branch.append({"value": 3})
    assert list(branch) == [{"value": 1}, {"value": 3}]
    assert list(archive) == [{"value": 1}, {"value": 2}]
    assert branch[-1] == {"value": 3}


def test_archive_snapshot_size_is_bounded(tmp_path):
    archive = EventArchive(tmp_path / "events.jsonl")
    archive.append({"bins": list(range(60))})
    initial = len(pickle.dumps(archive))
    for _ in range(1000):
        archive.append({"bins": list(range(60))})
    assert len(pickle.dumps(archive)) < initial + 20
    assert len(archive) == 1001


def test_corrupt_archive_rejected_without_branch(tmp_path):
    archive = EventArchive(tmp_path / "events.jsonl")
    archive.append({"value": 1})
    restored = pickle.loads(pickle.dumps(archive))
    archive.path.write_text('{"value": 9}\n')
    destination = tmp_path / "branch.jsonl"
    with pytest.raises(ValueError, match="changed"):
        restored.fork(destination)
    assert not destination.exists()


@pytest.mark.parametrize("chunksize", [1, 2, 256])
def test_csv_export(tmp_path, chunksize):
    records = [{"n": i, "bins": [1., 2.], "label": "core"} for i in range(7)]
    output = tmp_path / "events.csv"
    write_event_csv(iter(records), output, chunksize)
    assert output.read_text() == pd.DataFrame(records).to_csv(index=False)
    write_event_csv(iter([]), output, chunksize)
    assert output.read_text() == pd.DataFrame().to_csv(index=False)
