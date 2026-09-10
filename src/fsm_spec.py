"""
FSM Decision Specification for ANTARA Science Simulator.

This mirrors the EXACT branch order inside SpacecraftFSM.evaluate(),
_handle_slewing_progression(), _handle_startup_sequencing(),
_handle_fault_overrides(), and _select_nominal_mode().

ORDER MATTERS - this is the exact priority order evaluated each timestep.
"""

from typing import List, Tuple

# Type: (guard_name, human_description, resulting_mode_if_true)
FSM_SPEC: List[Tuple[str, str, str]] = [
    # ================================================================
    # 1. SLEWING PROGRESSION (highest priority - checked first in evaluate())
    # ================================================================
    (
        "slew_critical_battery",
        "SLEWING active AND battery_wh < critical_power_wh (debounced)",
        "SAFE",
    ),
    (
        "slew_complete",
        "SLEWING active AND time_sec >= slew_end_time_sec",
        "target_mode",  # dynamic: self.target_mode
    ),
    (
        "slew_continue",
        "SLEWING active (default continuation)",
        "SLEWING",
    ),

    # ================================================================
    # 2. STARTUP SEQUENCING (fixed timer, no overrides)
    # ================================================================
    (
        "cold_to_detumble",
        "COLD_AND_DARK AND time_sec >= startup_cold_sec",
        "DETUMBLE",
    ),
    (
        "cold_hold",
        "COLD_AND_DARK (default - remain in cold startup)",
        "COLD_AND_DARK",
    ),
    (
        "detumble_complete",
        "DETUMBLE AND detumble_complete == True",
        "SUN_POINTING",
    ),
    (
        "detumble_hold",
        "DETUMBLE (default - remain in detumble startup)",
        "DETUMBLE",
    ),

    # ================================================================
    # 3. FAULT / SURVIVAL OVERRIDES (highest priority after startup)
    # ================================================================
    # --- SAFE mode internal logic ---
    (
        "safe_combined_fault",
        "current_mode == SAFE AND critical_fault_active AND wheel_saturation_fault_active",
        "SAFE",
    ),
    (
        "safe_pointing_budget",
        "current_mode == SAFE AND pointing_budget_fault_active",
        "SAFE",
    ),
    (
        "safe_critical_recovery_pending",
        "current_mode == SAFE AND in_critical_recovery AND recovery conditions NOT met (debounced)",
        "SAFE",
    ),
    (
        "safe_critical_recovery_confirmed",
        "current_mode == SAFE AND in_critical_recovery AND recovery conditions met (debounced)",
        "SUN_POINTING",
    ),
    (
        "safe_saturation_hold",
        "current_mode == SAFE AND adcs_saturation_hold_active",
        "SAFE",
    ),
    (
        "safe_exit_default",
        "current_mode == SAFE (default - SAFE exit -> default hold)",
        "SUN_POINTING",
    ),

    # --- Post-SAFE recovery hold ---
    (
        "post_safe_recovery_stabilising",
        "post_safe_recovery_hold_active AND current_mode == SUN_POINTING AND hold conditions NOT met (debounced)",
        "SUN_POINTING",
    ),
    (
        "post_safe_recovery_stable",
        "post_safe_recovery_hold_active AND current_mode == SUN_POINTING AND hold conditions met (debounced)",
        "None",  # returns None -> continue to nominal
    ),

    # --- Critical battery (outside SAFE) ---
    (
        "critical_battery",
        "critical_fault_active (battery_wh < critical_power_wh debounced)",
        "SAFE",
    ),

    # --- Combined critical + wheel saturation ---
    (
        "combined_critical_saturation",
        "critical_fault_active AND wheel_saturation_fault_active AND battery_wh < critical_power_wh",
        "SAFE",
    ),
    (
        "saturation_timeout_divergence",
        "adcs_saturation_hold_active AND wheel_saturation_fault_active AND saturation_hold_duration_sec > max_saturation_hold_sec AND pointing_budget_error_deg > pointing_budget_limit_deg",
        "SAFE",
    ),
    (
        "saturation_hold_continue",
        "adcs_saturation_hold_active AND wheel_saturation_fault_active (default - hold current mode during unloading)",
        "current_mode",  # returns self.current_mode
    ),

    # --- Pointing budget ---
    (
        "pointing_budget_fault",
        "pointing_budget_fault_active (pointing_budget_error_deg > pointing_budget_limit_deg debounced)",
        "SAFE",
    ),

    # ================================================================
    # 4. LOW POWER HYSTERESIS (checked before dwell & nominal priority)
    # ================================================================
    (
        "low_power_recovery_pending",
        "in_low_power AND battery_wh > low_power_exit_wh NOT yet debounced",
        "SUN_POINTING",
    ),
    (
        "low_power_exit",
        "in_low_power AND battery_wh > low_power_exit_wh (debounced)",
        "None",  # exits low power, continues to nominal
    ),
    (
        "low_power_entry",
        "battery_wh < low_power_enter_wh (debounced)",
        "SUN_POINTING",
    ),
    (
        "low_power_pending",
        "battery_wh < low_power_enter_wh NOT yet debounced",
        "SUN_POINTING",
    ),

    # ================================================================
    # 5. MINIMUM DWELL ENFORCEMENT (before operational priority)
    # ================================================================
    (
        "detection_dwell",
        "current_mode == DETECTION AND minimum_dwell_complete() == False",
        "DETECTION",
    ),
    (
        "relay_dwell",
        "current_mode == RELAY AND minimum_dwell_complete() == False",
        "RELAY",
    ),

    # ================================================================
    # 6. OPERATIONAL MODE SELECTION (priority ordered)
    # ================================================================
    # --- Operational priority ---
    (
        "relay_priority",
        "gs_visible AND battery_wh > min_relaybattery_capacity_wh",
        "RELAY",
    ),
    (
        "detection_priority",
        "inside_saa AND battery_wh > saa_detection_min_power_wh",
        "DETECTION",
    ),

    # --- Nominal baseline ---
    (
        "nominal_sun_pointing",
        "Default fallback",
        "SUN_POINTING",
    ),
]


# ================================================================
# ALLOWED_TRANSITIONS (mirrors SpacecraftFSM.ALLOWED_TRANSITIONS)
# ================================================================
ALLOWED_TRANSITIONS = {
    "COLD_AND_DARK": {"DETUMBLE"},
    "DETUMBLE": {"SAFE", "SUN_POINTING", "SLEWING"},
    "SAFE": {"SUN_POINTING", "RELAY", "DETECTION", "SLEWING"},
    "SLEWING": {"SAFE", "SUN_POINTING", "RELAY", "DETECTION"},
    "SUN_POINTING": {"SAFE", "RELAY", "DETECTION", "SLEWING"},
    "DETECTION": {"RELAY", "SUN_POINTING", "SAFE", "SLEWING"},
    "RELAY": {"SUN_POINTING", "SAFE", "DETECTION", "SLEWING"},
}


# ================================================================
# MODE COLORS (from MissionConfig)
# ================================================================
MODE_COLORS = {
    "COLD_AND_DARK": "#111111",
    "DETUMBLE": "#8c564b",
    "SAFE": "#7f7f7f",
    "SLEWING": "#9467bd",
    "SUN_POINTING": "#f1c40f",
    "DETECTION": "#d62728",
    "RELAY": "#1f77b4",
}


# ================================================================
# DEBOUNCE THRESHOLDS (from MissionConfig)
# ================================================================
DEBOUNCE_CONFIG = {
    "fault_debounce_samples": 3,
    "recovery_debounce_samples": 3,
    "mode_transition_grace_steps": 5,
    "critical_power_wh": 7.5,
    "low_power_enter_wh": 10.0,
    "low_power_exit_wh": 17.5,
    "safe_exit_battery_fraction": 0.75,
    "critical_recovery_target_wh": 6.5,
    "safe_hold_deadband_deg": 7.5,
    "max_saturation_hold_sec": 300.0,
    "min_relaybattery_capacity_wh": 10.0,
    "saa_detection_min_power_wh": 10.0,
    "startup_cold_sec": 120.0,
}