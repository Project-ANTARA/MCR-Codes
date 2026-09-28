"""Exact piecewise-constant-acceleration steps for the kinematic slew model.

Angles are degrees. This is a one-dimensional command profile, not a rigid-body
controller. A moving target is treated as fixed for the integration interval.
"""
import math


def validate_slew_config(config):
    """Reject undefined command dynamics before duration/torque divisions."""
    for name in ("max_slew_rate_deg_sec", "max_slew_accel_deg_sec2", "wheel_max_torque_nm"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("pointing_accuracy_deg", "settling_time_sec"):
        value = getattr(config, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def slew_step(distance, velocity, max_rate, acceleration, dt, *, phases_out=None):
    """Return displacement, final signed rate and time spent moving.

    Signed velocity is measured toward the target. If stopping distance exceeds
    remaining distance, brake past the target and return; never discard velocity.
    Optional phases_out receives only executed (acceleration, duration) phases.
    """
    if not all(math.isfinite(x) for x in (distance, velocity, max_rate, acceleration, dt)):
        raise ValueError("Slew profile inputs must be finite")
    if distance < 0 or dt < 0 or max_rate <= 0 or acceleration <= 0:
        raise ValueError("Invalid slew distance, timestep or limits")
    if dt == 0:
        return 0., velocity, 0.
    # Roundoff at a completed stop is not a new triangular maneuver. These
    # numerical bounds are far below the spacecraft pointing acceptance band.
    if distance <= 1e-12 and abs(velocity) <= 1e-12:
        return distance, 0., 0.
    phases = []
    position, speed = 0., velocity

    def phase(accel, duration):
        nonlocal position, speed
        if duration > 0:
            phases.append((accel, duration))
            position += speed * duration + .5 * accel * duration * duration
            speed += accel * duration

    # First recover from motion away from the target or an infeasible stop.
    if speed < 0:
        phase(acceleration, -speed / acceleration)
        speed = 0.
    elif speed > 0 and (speed * speed > 2 * acceleration * distance or
                       math.isclose(speed * speed / (2 * acceleration), distance, rel_tol=1e-12, abs_tol=1e-12)):
        phase(-acceleration, speed / acceleration)
        speed = 0.
        if math.isclose(position, distance, rel_tol=1e-12, abs_tol=1e-12):
            position = distance

    direction = 1. if distance >= position else -1.
    remaining = abs(distance - position)
    along = direction * speed
    if along > max_rate:
        phase(-direction * acceleration, (along - max_rate) / acceleration)
        remaining = abs(distance - position)
        along = max_rate
    peak = min(max_rate, math.sqrt(max(0., acceleration * remaining + .5 * along * along)))
    phase(direction * acceleration, max(0., (peak - along) / acceleration))
    coast_distance = max(0., abs(distance - position) - peak * peak / (2 * acceleration))
    if peak > 0:
        phase(0., coast_distance / peak)
        phase(-direction * acceleration, peak / acceleration)

    position, speed, moving_time = 0., velocity, 0.
    for accel, duration in phases:
        step = min(duration, dt - moving_time)
        if phases_out is not None and step > 0:
            phases_out.append((accel, step))
        position += speed * step + .5 * accel * step * step
        speed += accel * step
        moving_time += step
        if step < duration:
            return position, speed, moving_time
    # All phases actually completed: eliminate only endpoint roundoff.
    return distance, 0., moving_time
