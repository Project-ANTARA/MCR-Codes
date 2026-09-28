"""Elapsed-time weights for interval-start telemetry (terminal observations weigh zero)."""
import numpy as np
import pandas as pd


SCIENCE_MODES = ("COLD_AND_DARK", "DETUMBLE", "SAFE", "SLEWING",
                 "SUN_POINTING", "DETECTION", "RELAY")
SCIENCE_BOOLEAN_COLUMNS = frozenset({
    "terminal_snapshot", "wheel_event_split", "in_sunlight", "in_saa",
    "gs_visible", "in_eclipse", "sun_visible", "pointing_budget_violated_frozen",
    "saturation_hold_active", "combined_fault_active", "in_low_power",
    "in_critical_recovery", "post_safe_recovery_hold_active",
    "science_generation_enabled", "relay_link_active", "science_histogram_downlink_due",
    "science_saa_state_at_sample", "storm_time", "panel_deployed",
})


def validated_modes(frame):
    """Reject labels that would otherwise disappear from grouped statistics."""
    if "mode" not in frame or not frame["mode"].isin(SCIENCE_MODES).all():
        raise ValueError("Telemetry mode must be a recognized Science mode on every row")
    return frame["mode"]


def boolean_measurements(values, column, *, allow_missing=False):
    """Parse explicit Boolean encodings; never use string truthiness."""
    values = pd.Series(values)
    allowed = values.isin([True, False, 0, 1, "True", "False", "0", "1"])
    if allow_missing:
        allowed |= values.isna()
    if not allowed.all():
        raise ValueError(f"Invalid Boolean telemetry in {column}")
    return values.map(lambda value: np.nan if pd.isna(value)
                      else float(value in (True, 1, "True", "1")))


# Science CSV metadata and serialized vectors are not scalar measurements.
# Keep this explicit: pandas' per-chunk dtype inference is not a schema.
SCIENCE_NONSCALAR_COLUMNS = frozenset({
    "mode", "sun_submode", "active_attitude_target", "attitude_hold_reason",
    "payload_power_state", "science_region", "science_cell_id", "science_event_type",
    "payload_boresight_inertial", "velocity_unit_vector", "relay_boresight_vector",
    "gs_los_vector", "science_measured_flux", "science_ap9_flux",
    "science_total_count_rate", "science_counts_per_bin", "science_sigma_fraction",
})


def science_scalar_frame(frame):
    """Parse Science scalar columns strictly, preserving missing measurements.

    New text/vector columns must be declared above rather than silently excluded
    when a malformed scalar happens to change pandas' inferred dtype.
    """
    scalars = {}
    for column in frame.columns:
        if column in SCIENCE_NONSCALAR_COLUMNS:
            continue
        values = frame[column]
        if column in SCIENCE_BOOLEAN_COLUMNS:
            values = boolean_measurements(values, column, allow_missing=True)
        try:
            scalars[column] = pd.to_numeric(values, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid scalar telemetry in {column}") from exc
    return pd.DataFrame(scalars, index=frame.index)


def interval_weights(frame):
    times = None
    if "time_sec" in frame:
        times = frame["time_sec"].astype(float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Telemetry timestamps must be finite and strictly increasing")
    if "interval_duration_sec" in frame:
        weights = frame["interval_duration_sec"].astype(float)
    elif times is not None:
        weights = times.shift(-1).sub(times).fillna(0.)
    else:
        raise ValueError("Time-weighted metrics require timestamps or interval durations")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Invalid telemetry interval durations")
    if times is not None:
        # Absolute tolerance only: relative tolerance at annual timestamps can
        # conceal seconds of missing or double-counted integration.
        if not np.allclose(weights.iloc[:-1], np.diff(times), rtol=0., atol=1e-7):
            raise ValueError("Telemetry interval durations disagree with timestamps")
        if "state_time_sec" in frame:
            ends = frame["state_time_sec"].astype(float)
            if not np.isfinite(ends).all() or not np.allclose(ends, times + weights, rtol=0., atol=1e-7):
                raise ValueError("Telemetry state times disagree with interval endpoints")
    return weights


def weighted_mean(values, weights):
    weights = pd.Series(weights, dtype=float)
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Mean weights must be finite and nonnegative")
    values = pd.Series(values, index=weights.index, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Infinite telemetry measurement in weighted mean")
    valid = values.notna() & (weights > 0)
    denominator = weights[valid].sum()
    if not denominator:
        return float("nan")
    with np.errstate(over="raise", invalid="raise"):
        # Normalize first to avoid overflow from otherwise finite value*seconds.
        mean = float(np.sum(values[valid].to_numpy() * (weights[valid] / denominator).to_numpy()))
    if not np.isfinite(denominator) or not np.isfinite(mean):
        raise FloatingPointError("Weighted telemetry mean overflow")
    return mean


class RunningRange:
    """Pickle-friendly constant-space summary of a scalar history."""
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.minimum = float("inf")
        self.maximum = -float("inf")

    def append(self, value, weight=1.):
        self.count += weight
        self.total += value * weight
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    @property
    def mean(self):
        return self.total / self.count if self.count else float("nan")


def streaming_summary(path, chunksize=10000):
    """Exact duration-weighted scalar CSV summary with bounded working memory.

    Requires current interval metadata; does not guess chunk-boundary durations.
    Array/string science columns remain in the original CSV, not this summary.
    """
    means, durations, minima, maxima = {}, {}, {}, {}
    missing_rows = {}
    modes, submodes = {}, {}
    rows = 0
    elapsed = 0.
    previous_time = previous_end = final_duration = None
    for frame in pd.read_csv(path, chunksize=chunksize):
        if "interval_duration_sec" not in frame:
            raise ValueError("Streaming summary requires interval_duration_sec")
        weights = interval_weights(frame)
        if "time_sec" not in frame:
            raise ValueError("Streaming summary requires time_sec for continuity checks")
        if frame.empty:
            continue
        validated_modes(frame)
        times = frame["time_sec"].astype(float)
        if previous_time is not None and (
                times.iloc[0] <= previous_time or
                not np.isclose(times.iloc[0], previous_end, rtol=0., atol=1e-7)):
            raise ValueError("Telemetry intervals are discontinuous across CSV chunks")
        previous_time = float(times.iloc[-1])
        final_duration = float(weights.iloc[-1])
        previous_end = previous_time + final_duration
        elapsed += weights.sum()
        rows += len(frame)
        scalar_frame = science_scalar_frame(frame)
        scalar_frame["wheel_momentum_norm_nms"] = np.linalg.norm(
            scalar_frame[["wheel_mom_x", "wheel_mom_y", "wheel_mom_z"]], axis=1)
        for column in scalar_frame:
            values = scalar_frame[column]
            if np.isinf(values).any():
                raise ValueError(f"Infinite telemetry measurement in {column}")
            valid = np.isfinite(values)
            missing_rows[column] = missing_rows.get(column, 0) + int((~valid).sum())
            old_duration = durations.get(column, 0.)
            chunk_duration = float(weights[valid].sum())
            durations[column] = old_duration + chunk_duration
            if chunk_duration:
                chunk_mean = weighted_mean(values, weights)
                means[column] = weighted_mean(
                    [means.get(column, 0.), chunk_mean],
                    pd.Series([old_duration, chunk_duration]))
            if valid.any():
                minima[column] = min(minima.get(column, float("inf")), values[valid].min())
                maxima[column] = max(maxima.get(column, -float("inf")), values[valid].max())
        for column, target in (("mode", modes), ("sun_submode", submodes)):
            if column in frame:
                for label, duration in weights.groupby(frame[column]).sum().items():
                    target[label] = target.get(label, 0.) + duration
    if final_duration != 0.:
        raise ValueError("Completed telemetry requires a zero-duration terminal observation")
    return {"rows": rows, "duration_sec": float(elapsed), "mode_seconds": modes,
            "sun_submode_seconds": submodes,
            "metrics": {key: {"mean": means.get(key),
                              "min": float(minima[key]) if key in minima else None,
                              "max": float(maxima[key]) if key in maxima else None,
                              "valid_duration_sec": float(durations[key]),
                              "missing_duration_sec": float(max(0., elapsed - durations[key])),
                              "missing_rows": missing_rows[key]}
                        for key in durations}}


def plotting_sample(path, limit=5000):
    """Uniform row-index display sample with bounded memory and both endpoints.

    Two CSV passes avoid repeatedly thinning early history in favor of recent
    chunks. Row spacing is not elapsed-time spacing; never use for statistics.
    Input must be a completed, unchanged CSV. CSV parsing (not newline counting)
    preserves records containing quoted multiline fields.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 2:
        raise ValueError("Plot sample limit must be an integer of at least two")
    with pd.read_csv(path, chunksize=limit, usecols=[0]) as reader:
        total = sum(len(chunk) for chunk in reader)
    indices = np.linspace(0, total - 1, min(limit, total), dtype=int)
    selected = []
    offset = 0
    columns = None
    with pd.read_csv(path, chunksize=limit) as reader:
        for chunk in reader:
            columns = chunk.columns
            local = indices[(indices >= offset) & (indices < offset + len(chunk))] - offset
            if len(local):
                selected.append(chunk.iloc[local].copy())
            offset += len(chunk)
    if offset != total:
        raise ValueError("Telemetry row count changed while preparing plot sample")
    sample = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame(columns=columns)
    return sample, total > limit
