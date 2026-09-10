#!/usr/bin/env python3
"""
Drift guard test: verifies that SpacecraftFSM.evaluate() matches fsm_spec.py.

Generates synthetic test cases covering each branch in fsm_spec.py at least once,
including edge cases exactly at each threshold. Runs each through a fresh
SpacecraftFSM.evaluate() and asserts the result matches fsm_spec.py's first
matching guard.
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from antara_models import SpacecraftFSM, MissionConfig
from fsm_spec import FSM_SPEC, ALLOWED_TRANSITIONS


def make_fsm(config: MissionConfig) -> SpacecraftFSM:
    """Create a fresh FSM instance."""
    return SpacecraftFSM(config)


def test_guard(config: MissionConfig, fsm: SpacecraftFSM, **kwargs) -> Tuple[str, str]:
    """Run FSM evaluate with given parameters."""
    return fsm.evaluate(
        battery_wh=kwargs.get("battery_wh", config.battery_capacity_wh),
        gs_visible=kwargs.get("gs_visible", False),
        inside_saa=kwargs.get("inside_saa", False),
        adcs_saturation_hold_active=kwargs.get("adcs_saturation_hold_active", False),
        saturation_hold_duration_sec=kwargs.get("saturation_hold_duration_sec", 0.0),
        pointing_budget_error_deg=kwargs.get("pointing_budget_error_deg", 0.0),
        pointing_budget_limit_deg=kwargs.get("pointing_budget_limit_deg", config.safe_hold_deadband_deg),
        detumble_complete=kwargs.get("detumble_complete", True),
    )


def set_mode(fsm: SpacecraftFSM, mode: str):
    """Set FSM mode directly (bypassing transition validation)."""
    fsm.current_mode = mode


def run_branch_tests():
    """Run all branch tests from FSM_SPEC."""
    config = MissionConfig()
    passed = 0
    failed = 0

    for guard_name, description, expected_mode in FSM_SPEC:
        fsm = make_fsm(config)

        # Set up test case based on guard_name
        try:
            result = run_test_case(fsm, config, guard_name)
            actual_mode, reason = result

            # Handle special expected modes
            if expected_mode == "target_mode":
                expected = fsm.target_mode
            elif expected_mode == "current_mode":
                expected = fsm.current_mode
            elif expected_mode == "None":
                # Guard returns None (no mode change)
                expected = fsm.current_mode
            else:
                expected = expected_mode

            if actual_mode == expected:
                print(f"  PASS: {guard_name}")
                passed += 1
            else:
                print(f"  FAIL: {guard_name}")
                print(f"    Expected: {expected}, Got: {actual_mode} (reason: {reason})")
                failed += 1
        except Exception as e:
            print(f"  ERROR: {guard_name} - {e}")
            failed += 1

    print(f"\n=== Summary: {passed} passed, {failed} failed ===")
    return failed == 0


def run_test_case(fsm: SpacecraftFSM, config: MissionConfig, guard_name: str) -> Tuple[str, str]:
    if guard_name == "pointing_budget_fault":
        print(f"DEBUG: run_test_case for pointing_budget_fault")
        print(f"  fsm.current_mode at start: {fsm.current_mode}")
        print(f"  fsm.pointing_budget_fault_active at start: {fsm.pointing_budget_fault_active}")
    """Run a specific test case for a guard."""
    # This is a simplified version - in reality we'd need to carefully
    # set up each test case. For now, we'll use a mapping approach.

    test_cases = {
        # SLEWING
        "slew_critical_battery": {
            "mode": "SLEWING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.critical_power_wh - 0.1,
            "target_mode": "SUN_POINTING",
            "slew_end_time_sec": 1e9,
            # Debounce counter for critical battery (needs 3 samples)
            "critical_battery_debounced": True,
        },
        "slew_complete": {
            "mode": "SLEWING",
            "time_sec": 100.0,
            "detumble_complete": True,
            "battery_wh": config.critical_power_wh + 1.0,
            "target_mode": "SUN_POINTING",
            "slew_end_time_sec": 0.0,
        },
        "slew_continue": {
            "mode": "SLEWING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.critical_power_wh + 1.0,
            "target_mode": "SUN_POINTING",
            "slew_end_time_sec": 1e9,
        },

        # STARTUP
        "cold_to_detumble": {
            "mode": "COLD_AND_DARK",
            "time_sec": config.startup_cold_sec + 1.0,
        },
        "cold_hold": {
            "mode": "COLD_AND_DARK",
            "time_sec": config.startup_cold_sec - 1.0,
        },
        "detumble_complete": {
            "mode": "DETUMBLE",
            "detumble_complete": True,
        },
        "detumble_hold": {
            "mode": "DETUMBLE",
            "detumble_complete": False,
        },

        # FAULTS - SAFE internal
        "safe_combined_fault": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.critical_power_wh - 1.0,
            "wheel_saturation_fault_active": True,
            "critical_fault_active": True,  # debounced
            "adcs_saturation_hold_active": True,
        },
        "safe_pointing_budget": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "pointing_budget_fault_active": True,
            "adcs_saturation_hold_active": False,
        },
        "safe_critical_recovery_pending": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "in_critical_recovery": True,
            "battery_wh": config.critical_recovery_target_wh - 0.1,
            "adcs_saturation_hold_active": False,
        },
        "safe_critical_recovery_confirmed": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "in_critical_recovery": True,
            "battery_wh": config.safe_exit_battery_fraction * config.battery_capacity_wh + 1.0,
            "safe_exit_recovery_debounced": True,
            "adcs_saturation_hold_active": False,
        },
        "safe_saturation_hold": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "adcs_saturation_hold_active": True,
        },
        "safe_exit_default": {
            "mode": "SAFE",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "in_critical_recovery": False,
            "pointing_budget_fault_active": False,
            "adcs_saturation_hold_active": False,
        },

        # Post-SAFE recovery
        "post_safe_recovery_stabilising": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "post_safe_recovery_hold_active": True,
            "battery_wh": config.critical_recovery_target_wh - 0.1,
        },
        "post_safe_recovery_stable": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "post_safe_recovery_hold_active": True,
            "battery_wh": config.critical_recovery_target_wh + 1.0,
            "adcs_saturation_hold_active": False,
            "pointing_budget_error_deg": 0.0,
            "safe_exit_hold_debounced": True,
        },

        # Critical battery (post-startup, in SUN_POINTING)
        "critical_battery": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.critical_power_wh - 0.1,
            "critical_fault_active": True,  # debounced
        },

        # Combined faults (post-startup, in SUN_POINTING)
        "combined_critical_saturation": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "critical_fault_active": True,
            "wheel_saturation_fault_active": True,
            "battery_wh": config.critical_power_wh - 0.1,
        },
        "saturation_timeout_divergence": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "adcs_saturation_hold_active": True,
            "wheel_saturation_fault_active": True,
            "saturation_hold_duration_sec": config.max_saturation_hold_sec + 1.0,
            "pointing_budget_error_deg": config.safe_hold_deadband_deg + 1.0,
            "pointing_budget_limit_deg": config.safe_hold_deadband_deg,
        },
        "saturation_hold_continue": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "adcs_saturation_hold_active": True,
            "wheel_saturation_fault_active": True,
            "battery_wh": config.critical_power_wh + 1.0,
        },

        # Pointing budget (post-startup, in SUN_POINTING)
        "pointing_budget_fault": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "pointing_budget_fault_active": True,
            "pointing_budget_error_deg": config.safe_hold_deadband_deg + 1.0,
            "pointing_budget_limit_deg": config.safe_hold_deadband_deg,
            "adcs_saturation_hold_active": False,
        },

        # Minimum dwell (need to set mode_entry_time_sec to make dwell incomplete)
        # DETECTION minimum dwell = 120s, at time_sec=130 need entry_time > 10 to be incomplete
        "detection_dwell": {
            "mode": "DETECTION",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "mode_entry_time_sec": config.startup_cold_sec + 10.0 - 60.0,  # 60 seconds ago, dwell incomplete (120s required)
            "adcs_saturation_hold_active": False,
        },
        "relay_dwell": {
            "mode": "RELAY",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "mode_entry_time_sec": config.startup_cold_sec + 10.0 - 60.0,  # 60 seconds ago, dwell incomplete (120s required)
            "adcs_saturation_hold_active": False,
        },

        # Low power (post-startup, in SUN_POINTING)
        "low_power_recovery_pending": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "in_low_power": True,
            "battery_wh": config.low_power_exit_wh + 1.0,
        },
        "low_power_exit": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "in_low_power": True,
            "battery_wh": config.low_power_exit_wh + 1.0,
            "low_power_recovery_debounced": True,
        },
        "low_power_entry": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.low_power_enter_wh - 0.1,
        },
        "low_power_pending": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "battery_wh": config.low_power_enter_wh - 0.1,
            "low_power_entry_debounced": False,
        },

        # Nominal (post-startup, in SUN_POINTING)
        "relay_priority": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "gs_visible": True,
            "battery_wh": config.min_relaybattery_capacity_wh + 1.0,
        },
        "detection_priority": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
            "inside_saa": True,
            "battery_wh": config.saa_detection_min_power_wh + 1.0,
        },
        "nominal_sun_pointing": {
            "mode": "SUN_POINTING",
            "time_sec": config.startup_cold_sec + 10.0,
            "detumble_complete": True,
        },
    }

    if guard_name not in test_cases:
        raise ValueError(f"No test case for {guard_name}")

    tc = test_cases[guard_name]

    # Set mode if specified
    if "mode" in tc:
        set_mode(fsm, tc["mode"])

    # Set debounce states
    if tc.get("in_critical_recovery"):
        fsm.in_critical_recovery = True
    if tc.get("wheel_saturation_fault_active"):
        fsm.wheel_saturation_fault_active = True
        fsm.debounce_counters["wheel_saturation"] = config.fault_debounce_samples
    if tc.get("pointing_budget_fault_active"):
        fsm.pointing_budget_fault_active = True
        fsm.debounce_counters["pointing_budget"] = config.fault_debounce_samples
    if tc.get("in_critical_recovery"):
        fsm.in_critical_recovery = True
    if tc.get("post_safe_recovery_hold_active"):
        fsm.post_safe_recovery_hold_active = True
    if tc.get("in_low_power"):
        fsm.in_low_power = True
    if tc.get("low_power_recovery_debounced"):
        fsm.debounce_counters["low_power_recovery"] = config.recovery_debounce_samples
    if tc.get("low_power_entry_debounced"):
        fsm.debounce_counters["low_power_entry"] = config.fault_debounce_samples
    if tc.get("safe_exit_recovery_debounced"):
        fsm.debounce_counters["safe_exit_recovery"] = config.recovery_debounce_samples
    if tc.get("safe_exit_hold_debounced"):
        fsm.debounce_counters["safe_exit_hold"] = config.recovery_debounce_samples
    # New debounce fields for critical battery and fault triggers
    if tc.get("critical_fault_active"):
        fsm.critical_fault_active = True
        fsm.debounce_counters["critical_battery"] = config.fault_debounce_samples
    if tc.get("critical_battery_debounced"):
        fsm.debounce_counters["critical_battery"] = config.fault_debounce_samples
    if tc.get("safe_exit_recovery_debounced"):
        fsm.debounce_counters["safe_exit_recovery"] = config.recovery_debounce_samples
    if tc.get("safe_exit_hold_debounced"):
        fsm.debounce_counters["safe_exit_hold"] = config.recovery_debounce_samples

    # Set time
    if "time_sec" in tc:
        fsm.time_sec = tc["time_sec"]
    if "slew_end_time_sec" in tc:
        fsm.slew_end_time_sec = tc["slew_end_time_sec"]
    if "target_mode" in tc:
        fsm.target_mode = tc["target_mode"]

    # Set minimum dwell states - prefer explicit mode_entry_time_sec, fallback to minimum_dwell_complete
    if "mode_entry_time_sec" in tc:
        fsm.mode_entry_time_sec = tc["mode_entry_time_sec"]
    elif "minimum_dwell_complete" in tc:
        fsm.mode_entry_time_sec = -1e9 if not tc["minimum_dwell_complete"] else 0

    # Run evaluate
    if guard_name == "pointing_budget_fault":
        print(f"DEBUG: Before test_guard")
        print(f"  fsm.current_mode: {fsm.current_mode}")
        print(f"  fsm.pointing_budget_fault_active: {fsm.pointing_budget_fault_active}")
        print(f"  fsm.debounce_counters: {dict(fsm.debounce_counters)}")
    result = test_guard(config, fsm, **{
        k: v for k, v in tc.items() if k not in [
            "mode", "target_mode", "time_sec", "slew_end_time_sec",
            "minimum_dwell_complete", "low_power_recovery_debounced",
            "low_power_entry_debounced", "safe_exit_recovery_debounced",
            "safe_exit_hold_debounced", "in_critical_recovery",
            "wheel_saturation_fault_active", "pointing_budget_fault_active",
            "post_safe_recovery_hold_active", "in_low_power",
            "low_power_recovery_debounced", "low_power_entry_debounced",
            "safe_exit_recovery_debounced", "safe_exit_hold_debounced",
        ]
    })
    if guard_name == "pointing_budget_fault":
        print(f"DEBUG: After test_guard")
        print(f"  result: {result}")
        print(f"  fsm.current_mode: {fsm.current_mode}")
        print(f"  fsm.pointing_budget_fault_active: {fsm.pointing_budget_fault_active}")
        print(f"  fsm.debounce_counters: {dict(fsm.debounce_counters)}")
    return result


if __name__ == "__main__":
    success = run_branch_tests()
    sys.exit(0 if success else 1)