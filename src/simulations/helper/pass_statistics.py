"""Bounded per-pass payload statistics, independent of the number of samples."""
from collections import Counter

import numpy as np


class PassStatistics:
    """Preserve sample-weighted summaries without retaining detector histories.

    Pitch variance uses Welford's population-variance recurrence. Vector totals
    retain one array per reported quantity; labels retain only unique values.
    """

    VECTOR_FIELDS = ("measured_flux", "ap9_flux", "sigma_fraction", "counts_per_bin")

    def __init__(self):
        self.count = 0
        self.pitch_mean = 0.0
        self.pitch_m2 = 0.0
        self.ew_total = 0.0
        self.livetime_total = 0.0
        self.totals = {}
        self.cell_ids = set()
        self.event_types = set()
        self.regions = Counter()

    def add(self, record):
        self.count += 1
        pitch = float(record["pitch_angle_deg"])
        delta = pitch - self.pitch_mean
        self.pitch_mean += delta / self.count
        self.pitch_m2 += delta * (pitch - self.pitch_mean)
        self.ew_total += float(record["ew_factor"])
        self.livetime_total += float(record["livetime_fraction"])
        for key in self.VECTOR_FIELDS:
            value = np.asarray(record[key], dtype=float)
            if key not in self.totals:
                self.totals[key] = value.copy()
            else:
                self.totals[key] += value
        self.cell_ids.add(record["cell_id"])
        self.event_types.add(record["event_type"])
        self.regions[record["region"]] += 1

    def summary(self):
        if not self.count:
            raise ValueError("Cannot summarize an empty science pass")
        return {
            "mean_pitch_angle_deg": self.pitch_mean,
            "std_pitch_angle_deg": float(np.sqrt(max(0.0, self.pitch_m2 / self.count))),
            "mean_ew_factor": self.ew_total / self.count,
            "mean_livetime_fraction": self.livetime_total / self.count,
            "mean_measured_flux": (self.totals["measured_flux"] / self.count).tolist(),
            "mean_ap9_flux": (self.totals["ap9_flux"] / self.count).tolist(),
            "mean_sigma_fraction": (self.totals["sigma_fraction"] / self.count).tolist(),
            "total_counts_per_bin": self.totals["counts_per_bin"].tolist(),
            "cell_ids": sorted(self.cell_ids),
            "event_types": sorted(self.event_types),
            "region_histogram": dict(sorted(self.regions.items())),
        }
