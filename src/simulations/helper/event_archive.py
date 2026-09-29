"""Append-only JSONL event storage with checkpoint-safe immutable prefixes."""
import hashlib
import json
import os
import math
from copy import deepcopy
from pathlib import Path

import pandas as pd


def _json_record(value):
    """Represent missing floating samples as JSON null, never as zero.

    Horizon-closed SAA events have unavailable exit coordinates. Infinity still
    fails strict JSON serialization rather than being disguised as missing.
    """
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_record(item) for item in value]
    return value


class EventArchive:
    """List-like read access without retaining completed records in RAM.

    Returned dictionaries are detached values. Only append is supported; mutable
    open slew/detumble records must not be placed in this archive.
    """

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.count = 0
        self.size = 0
        self._digest = hashlib.sha256()
        self._writable = True

    def __len__(self):
        return self.count

    def append(self, record):
        if not self._writable:
            raise RuntimeError("Fork a restored event archive before appending")
        encoded = (json.dumps(_json_record(record), allow_nan=False) + "\n").encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("xb" if self.size == 0 else "r+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() != self.size:
                raise ValueError("Event archive changed outside its writer")
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self._digest.update(encoded)
        self.count += 1
        self.size += len(encoded)

    def __iter__(self):
        if not self.count:
            return
        remaining = self.size
        with self.path.open("rb") as stream:
            for _ in range(self.count):
                row = stream.readline(remaining)
                if not row.endswith(b"\n") or not remaining:
                    raise ValueError("Event archive prefix is truncated")
                remaining -= len(row)
                yield json.loads(row)
        if remaining:
            raise ValueError("Event archive record count disagrees with prefix")

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Event archive indices must be integers")
        if index < 0:
            index += self.count
        if index < 0 or index >= self.count:
            raise IndexError(index)
        for i, row in enumerate(self):
            if i == index:
                return row

    def _copy_verified_prefix(self, destination=None):
        digest = hashlib.sha256()
        remaining = self.size
        if remaining:
            with self.path.open("rb") as stream:
                while remaining:
                    block = stream.read(min(remaining, 1024 * 1024))
                    if not block:
                        raise ValueError("Event archive prefix is truncated")
                    digest.update(block)
                    if destination is not None:
                        destination.write(block)
                    remaining -= len(block)
        expected = self._digest.hexdigest() if self._writable else self._sha256
        if digest.hexdigest() != expected:
            raise ValueError("Event archive prefix has changed")
        return digest

    def verify(self):
        self._copy_verified_prefix()

    def __getstate__(self):
        self.verify()
        return {"path": self.path, "count": self.count, "size": self.size,
                "_sha256": self._digest.hexdigest() if self._writable else self._sha256}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._writable = False

    def fork(self, path):
        """Copy only this checkpoint's verified prefix to a new run's archive."""
        result = EventArchive(path)
        if self.size:
            result.path.parent.mkdir(parents=True, exist_ok=True)
            with result.path.open("xb") as stream:
                try:
                    result._digest = self._copy_verified_prefix(stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                except Exception:
                    stream.close()
                    result.path.unlink()
                    raise
        else:
            self.verify()
        result.count, result.size = self.count, self.size
        return result


class OpenEventArchive:
    """Completed events on disk plus at most one mutable, indexed open event."""

    def __init__(self, path):
        self.completed = EventArchive(path)
        self.pending = None

    def __len__(self):
        return len(self.completed) + (self.pending is not None)

    def __iter__(self):
        yield from self.completed
        if self.pending is not None:
            yield dict(self.pending)

    def __getitem__(self, index):
        if not isinstance(index, int):
            raise TypeError("Event archive indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.pending is not None and index == len(self.completed):
            return dict(self.pending)
        return self.completed[index]

    def append(self, record):
        if self.pending is not None:
            raise ValueError("Finalize the open event before starting another")
        if record.get("end_time_sec") is None:
            self.pending = deepcopy(record)
        else:
            self.completed.append(record)

    def finalize(self, index, end_time_sec):
        if self.pending is None or index != len(self.completed):
            raise ValueError("Only the current open event can be finalized")
        record = dict(self.pending)
        record["end_time_sec"] = end_time_sec
        record["duration_sec"] = max(0., end_time_sec - record["start_time_sec"])
        self.completed.append(record)
        self.pending = None

    def verify(self):
        self.completed.verify()

    def fork(self, path):
        result = OpenEventArchive(path)
        result.completed = self.completed.fork(path)
        result.pending = deepcopy(self.pending)
        return result


def event_sample(records, count, limit=5000):
    """Uniform row-index display sample, including endpoints; never statistics."""
    if limit < 2:
        raise ValueError("Display limit must be at least two")
    indices = {i * (count - 1) // (limit - 1) for i in range(limit)} if count > limit else None
    return [row for i, row in enumerate(records) if indices is None or i in indices]


def write_event_csv(records, path, chunksize=256):
    """Export exact records in bounded batches with historical pandas formatting."""
    if chunksize <= 0:
        raise ValueError("Event CSV chunk size must be positive")
    chunk, columns, first = [], None, True
    for record in records:
        if columns is None:
            columns = list(record)
        if set(record) != set(columns):
            raise ValueError("Event archive schema changed within one stream")
        chunk.append(record)
        if len(chunk) == chunksize:
            pd.DataFrame(chunk, columns=columns).to_csv(path, mode="w" if first else "a", header=first, index=False)
            first = False
            chunk.clear()
    if chunk:
        pd.DataFrame(chunk, columns=columns).to_csv(path, mode="w" if first else "a", header=first, index=False)
    elif columns is None:
        pd.DataFrame().to_csv(path, index=False)
