#!/usr/bin/env python3
"""
Checkpoint Round-trip Test.

Tests that save_checkpoint() and load_checkpoint() preserve simulator state correctly.

Test procedure:
1. Build simulator with config
2. Run 50 steps
3. save_checkpoint()
4. Load checkpoint into NEW instance
5. Assert state at checkpoint (step 50) matches exactly
6. Verify both can continue running (smoke test)
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "..", "src")))

from ANTARA_Science_Sim import MissionConfig, ANTARAScienceSimulator


def run_test():
    """Run the checkpoint round-trip test."""
    print("=" * 70)
    print("CHECKPOINT ROUND-TRIP TEST")
    print("=" * 70)
    
    # Create config with shorter duration for testing
    config = MissionConfig(
        sim_duration_hours=1.0,  # 1 hour = 1800 steps at 2s
        step_sec=2.0,
        output_tag="checkpoint_test",
    )
    
    # Build simulator
    print("\n1. Building simulator...")
    sim = ANTARAScienceSimulator(config)
    
    # Run 50 steps
    print("\n2. Running 50 steps...")
    sim.run_steps(50)
    
    # Record state at step 50 (checkpoint point)
    state_50 = {
        "time_sec": sim.time_sec,
        "battery_wh": sim.eps.battery_wh,
        "fsm_mode": sim.fsm.current_mode,
        "adcs_quaternion": sim.adcs.current_quaternion.copy(),
        "adcs_target_quaternion": sim.adcs.target_quaternion.copy(),
        "adcs_continuity_corrections": sim.adcs.quaternion_continuity_corrections,
        "adcs_continuity_by_site": sim.adcs.quaternion_continuity_corrections_by_site.copy(),
        "adcs_wheel_momentum": sim.adcs.wheel_momentum_nms.copy(),
        "adcs_body_rate": sim.adcs.body_rate_deg_sec.copy(),
        "science_storage_mb": sim.science_storage_mb,
    }
    print(f"   State at step 50: time={state_50['time_sec']:.1f}s, "
          f"battery={state_50['battery_wh']:.3f}Wh, "
          f"mode={state_50['fsm_mode']}, "
          f"storage={state_50['science_storage_mb']:.3f}MB")
    
    # Save checkpoint
    print("\n3. Saving checkpoint...")
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        ckpt_path = f.name
    sim.save_checkpoint(ckpt_path)
    print(f"   Checkpoint saved to {ckpt_path}")
    
    # Load checkpoint into NEW instance
    print("\n4. Loading checkpoint into NEW instance...")
    sim2 = ANTARAScienceSimulator.load_checkpoint(ckpt_path, config)
    
    # Record state immediately after load (should be identical to state_50)
    state_50_restored = {
        "time_sec": sim2.time_sec,
        "battery_wh": sim2.eps.battery_wh,
        "fsm_mode": sim2.fsm.current_mode,
        "adcs_quaternion": sim2.adcs.current_quaternion.copy(),
        "adcs_target_quaternion": sim2.adcs.target_quaternion.copy(),
        "adcs_continuity_corrections": sim2.adcs.quaternion_continuity_corrections,
        "adcs_continuity_by_site": sim2.adcs.quaternion_continuity_corrections_by_site.copy(),
        "adcs_wheel_momentum": sim2.adcs.wheel_momentum_nms.copy(),
        "adcs_body_rate": sim2.adcs.body_rate_deg_sec.copy(),
        "science_storage_mb": sim2.science_storage_mb,
    }
    print(f"   State at step 50 (restored): time={state_50_restored['time_sec']:.1f}s, "
          f"battery={state_50_restored['battery_wh']:.3f}Wh, "
          f"mode={state_50_restored['fsm_mode']}, "
          f"storage={state_50_restored['science_storage_mb']:.3f}MB")
    
    # Compare states at checkpoint (should be IDENTICAL)
    print("\n5. Comparing states at checkpoint (step 50)...")
    tolerance = 1e-10  # tight tolerance for exact round-trip
    all_pass = True
    
    # time_sec
    time_diff = abs(state_50["time_sec"] - state_50_restored["time_sec"])
    print(f"   time_sec: diff={time_diff:.6f}s {'PASS' if time_diff < tolerance else 'FAIL'}")
    all_pass &= time_diff < tolerance
    
    # battery_wh
    battery_diff = abs(state_50["battery_wh"] - state_50_restored["battery_wh"])
    print(f"   battery_wh: diff={battery_diff:.6f}Wh {'PASS' if battery_diff < tolerance else 'FAIL'}")
    all_pass &= battery_diff < tolerance
    
    # fsm_mode
    mode_match = state_50["fsm_mode"] == state_50_restored["fsm_mode"]
    print(f"   fsm_mode: original={state_50['fsm_mode']}, "
          f"restored={state_50_restored['fsm_mode']} {'PASS' if mode_match else 'FAIL'}")
    all_pass &= mode_match
    
    # adcs_quaternion
    quat_diff = np.max(np.abs(state_50["adcs_quaternion"] - state_50_restored["adcs_quaternion"]))
    print(f"   adcs_current_quaternion: max diff={quat_diff:.6f} {'PASS' if quat_diff < tolerance else 'FAIL'}")
    all_pass &= quat_diff < tolerance
    
    # adcs_target_quaternion
    target_quat_diff = np.max(np.abs(state_50["adcs_target_quaternion"] - state_50_restored["adcs_target_quaternion"]))
    print(f"   adcs_target_quaternion: max diff={target_quat_diff:.6f} {'PASS' if target_quat_diff < tolerance else 'FAIL'}")
    all_pass &= target_quat_diff < tolerance
    
    # continuity_corrections
    continuity_match = state_50["adcs_continuity_corrections"] == state_50_restored["adcs_continuity_corrections"]
    print(f"   continuity_corrections: original={state_50['adcs_continuity_corrections']}, "
          f"restored={state_50_restored['adcs_continuity_corrections']} {'PASS' if continuity_match else 'FAIL'}")
    all_pass &= continuity_match
    
    # continuity_by_site
    by_site_match = state_50["adcs_continuity_by_site"] == state_50_restored["adcs_continuity_by_site"]
    print(f"   continuity_by_site: {'PASS' if by_site_match else 'FAIL'}")
    all_pass &= by_site_match
    
    # wheel_momentum
    wheel_diff = np.max(np.abs(state_50["adcs_wheel_momentum"] - state_50_restored["adcs_wheel_momentum"]))
    print(f"   wheel_momentum: max diff={wheel_diff:.6f} {'PASS' if wheel_diff < tolerance else 'FAIL'}")
    all_pass &= wheel_diff < tolerance
    
    # body_rate
    rate_diff = np.max(np.abs(state_50["adcs_body_rate"] - state_50_restored["adcs_body_rate"]))
    print(f"   body_rate: max diff={rate_diff:.6f} {'PASS' if rate_diff < tolerance else 'FAIL'}")
    all_pass &= rate_diff < tolerance
    
    # science_storage_mb
    storage_diff = abs(state_50["science_storage_mb"] - state_50_restored["science_storage_mb"])
    print(f"   science_storage_mb: diff={storage_diff:.6f}MB {'PASS' if storage_diff < tolerance else 'FAIL'}")
    all_pass &= storage_diff < tolerance
    
    # Now test that both can continue running (smoke test)
    print("\n6. Running additional 20 steps on both (smoke test)...")
    sim.run_steps(20)
    sim2.run_steps(20)
    
    # Compare after additional steps (looser tolerance for numerical integration differences)
    print("\n7. Comparing states after 20 additional steps (looser tolerance)...")
    loose_tol = 1e-3  # 0.001 for numerical integration differences
    all_pass2 = True
    
    # fsm_mode should match exactly
    mode_match2 = sim.fsm.current_mode == sim2.fsm.current_mode
    print(f"   fsm_mode: original={sim.fsm.current_mode}, restored={sim2.fsm.current_mode} {'PASS' if mode_match2 else 'FAIL'}")
    all_pass2 &= mode_match2
    
    # battery_wh should be close
    battery_diff2 = abs(sim.eps.battery_wh - sim2.eps.battery_wh)
    print(f"   battery_wh: diff={battery_diff2:.6f}Wh {'PASS' if battery_diff2 < loose_tol else 'FAIL'}")
    all_pass2 &= battery_diff2 < loose_tol
    
    # time_sec should match exactly (same number of steps)
    time_diff2 = abs(sim.time_sec - sim2.time_sec)
    print(f"   time_sec: diff={time_diff2:.6f}s {'PASS' if time_diff2 < tolerance else 'FAIL'}")
    all_pass2 &= time_diff2 < tolerance
    
    overall_pass = all_pass and all_pass2
    
    # Cleanup
    try:
        os.unlink(ckpt_path)
        for ext in ['.json', '.npz']:
            alt_path = ckpt_path.replace('.pkl', ext)
            if os.path.exists(alt_path):
                os.unlink(alt_path)
    except:
        pass
    
    print("\n" + "=" * 70)
    if overall_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 70)
    
    return overall_pass


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)