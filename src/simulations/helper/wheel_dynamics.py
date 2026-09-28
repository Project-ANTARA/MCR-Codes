"""Piecewise-constant wheel drive with constant, zero-opposing unloading."""
import math

import numpy as np


def integrate_wheel_profile(initial, segments, unload, capacity):
    """Return final momentum and first capacity-crossing time (or None).

    Each segment is (bus torque vector, seconds); wheel drive is its negative.
    Unloading opposes each wheel's momentum, stopping at zero unless drive
    exceeds unloading. Zero crossings split a segment analytically.
    """
    momentum = np.asarray(initial, dtype=float).copy()
    if momentum.shape != (3,) or not np.isfinite(momentum).all():
        raise ValueError("Wheel momentum must be a finite three-vector")
    if not math.isfinite(unload) or unload < 0 or not math.isfinite(capacity) or capacity <= 0:
        raise ValueError("Invalid wheel unloading or capacity")
    crossing = 0. if np.any(np.abs(momentum) >= capacity) else None
    elapsed = 0.
    for torque, duration in segments:
        torque = np.asarray(torque, dtype=float)
        if torque.shape != (3,) or not np.isfinite(torque).all() or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid wheel torque segment")
        for axis in range(3):
            h, drive, remaining, local = float(momentum[axis]), -float(torque[axis]), duration, 0.
            for _ in range(2):  # At most one zero crossing under constant drive.
                slope = (drive - math.copysign(unload, h) if h != 0.
                         else math.copysign(max(abs(drive) - unload, 0.), drive))
                zero_time = -h / slope if h != 0. and slope != 0. else math.inf
                reaches_zero = 0. < zero_time <= remaining
                part = zero_time if reaches_zero else remaining
                if slope != 0.:
                    hit = (math.copysign(capacity, slope) - h) / slope
                    if 0. <= hit <= part:
                        event = elapsed + local + hit
                        crossing = event if crossing is None else min(crossing, event)
                h += slope * part
                if not math.isfinite(h):
                    raise FloatingPointError("Wheel profile integration overflow")
                local += part
                remaining -= part
                if not reaches_zero:
                    break
                h = 0.
            momentum[axis] = h
        elapsed += duration
    return momentum, crossing
