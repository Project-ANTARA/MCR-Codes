"""Bounded completed-record storage and immutable checkpoint prefixes."""
import pickle

import pandas as pd
import pytest

from simulations.helper.event_archive import EventArchive, write_event_csv


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
