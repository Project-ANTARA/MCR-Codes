#!/usr/bin/env python3
"""
Communications Link Budget validation test.

Implements from-scratch Friis link budget and compares against
CommunicationsModel.downlink_step()'s sqrt(sin(elevation)) heuristic.

The test validates:
1. Friis model predicts sufficient margin at all elevations >= mask (15°)
2. Heuristic (rate * sqrt(sin(el))) is CONSERVATIVE: always <= Friis max rate
3. Both are monotonic increasing with elevation
4. Heuristic provides reasonable conservative throughput estimate

Tolerance: 
- Link margin >= 3 dB at all elevations >= 15° mask
- Heuristic rate <= Friis max rate at all elevations (conservative design)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import numpy as np
from simulations.helper.antara_models import MissionConfig


def friis_link_budget(elevation_deg: float, config: MissionConfig) -> float:
    """
    Compute maximum achievable downlink data rate using Friis transmission equation.
    
    Uses realistic CubeSat parameters for UHF link.
    
    Parameters:
    - elevation_deg: GS elevation angle (degrees)
    - config: MissionConfig with link parameters

    Returns:
    - Maximum achievable data rate in Mbps
    """
    # Physical constants
    c = 299_792_458  # m/s
    k_boltzmann = 1.380649e-23  # J/K

    # Spacecraft parameters
    tx_power_w = config.radio_tx_power_w  # 12 W
    tx_power_dbm = 10 * np.log10(tx_power_w * 1000)  # dBm = 40.8 dBm

    # Frequency: UHF ~437 MHz (typical CubeSat)
    freq_hz = 437e6  # Hz
    wavelength_m = c / freq_hz  # 0.686 m

    # Antenna gains (realistic CubeSat UHF)
    sc_gain_dbi = 3.0    # Spacecraft patch/monopole
    gs_gain_dbi = 14.0   # Ground station 3m dish at UHF

    # Slant range at given elevation
    Re = 6371e3  # m
    h = config.altitude_m  # 600 km
    el_rad = np.radians(elevation_deg)
    slant_range = np.sqrt((Re + h)**2 - Re**2 * np.cos(el_rad)**2) - Re * np.sin(el_rad)

    # Free-space path loss (dB)
    fspl_db = 20 * np.log10(4 * np.pi * slant_range / wavelength_m)

    # Losses
    atm_loss_db = 0.2 / np.sin(el_rad) if elevation_deg > 5 else 10.0
    pol_loss_db = 3.0    # polarization mismatch
    point_loss_db = 2.0  # pointing loss (SC + GS)
    impl_loss_db = 2.0   # implementation loss
    total_losses = atm_loss_db + pol_loss_db + 2*point_loss_db + impl_loss_db

    # Received power (dBm)
    rx_power_dbm = (tx_power_dbm + sc_gain_dbi + gs_gain_dbi 
                    - fspl_db - total_losses)

    # Noise floor
    Tsys = 300.0  # K (typical UHF with LNA)
    bandwidth_hz = 25e3  # Hz (for 19.2 kbps BPSK)
    noise_power_dbm = 10 * np.log10(k_boltzmann * Tsys * bandwidth_hz * 1000)

    # SNR (dB)
    snr_db = rx_power_dbm - noise_power_dbm

    # Required Eb/N0 for BER=1e-5 with BPSK (uncoded)
    req_ebn0_db = 9.6
    
    # Maximum data rate: R = B * SNR / (Eb/N0)
    snr_linear = 10**(snr_db / 10)
    req_ebn0_linear = 10**(req_ebn0_db / 10)
    max_rate_bps = bandwidth_hz * snr_linear / req_ebn0_linear
    max_rate_mbps = max_rate_bps / 1e6

    return max_rate_mbps


def sqrt_sin_heuristic(elevation_deg: float, config: MissionConfig) -> float:
    """CommunicationsModel heuristic: rate * sqrt(sin(el))"""
    base_rate = config.downlink_rate_mbps
    el_rad = np.radians(elevation_deg)
    sin_el = np.sin(el_rad)
    factor = np.sqrt(max(0.0, sin_el))
    return base_rate * factor


def main():
    print("=" * 70)
    print("COMMS LINK BUDGET VALIDATION TEST")
    print("=" * 70)

    config = MissionConfig()

    # Test elevations
    elevations = [15, 20, 30, 45, 60, 75, 90]

    print(f"\n  Tx power: {config.radio_tx_power_w} W")
    print(f"  Configured rate: {config.downlink_rate_mbps*1000:.1f} kbps")
    print(f"  Frequency: 437 MHz (UHF)")
    print(f"  SC antenna: 3.0 dBi (patch/monopole)")
    print(f"  GS antenna: 14.0 dBi (3m dish at UHF)")
    print(f"  Bandwidth: 25 kHz (for 19.2 kbps BPSK)")
    print(f"  Required Eb/N0: 9.6 dB (BPSK, BER=1e-5)")

    results = []
    all_pass = True
    configured_rate_mbps = config.downlink_rate_mbps

    for el in elevations:
        friis_rate = friis_link_budget(el, config)
        heuristic_rate = sqrt_sin_heuristic(el, config)

        # Check 1: Link margin (Friis max rate vs configured rate)
        margin_db = 10 * np.log10(friis_rate / configured_rate_mbps) if friis_rate > 0 else -np.inf
        has_margin = margin_db >= 3.0  # At least 3 dB margin

        # Check 2: Heuristic is conservative (<= Friis max rate)
        heuristic_conservative = heuristic_rate <= friis_rate

        # Margin check
        has_margin_bool = margin_db >= 3.0 and el >= config.gs_elevation_mask_deg

        passed = True
        if not has_margin_bool and el >= config.gs_elevation_mask_deg:
            passed = False
            all_pass = False
        if not heuristic_conservative:
            passed = False
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(f"\n  Elevation: {el:2d}°")
        print(f"  Friis max rate:    {friis_rate*1000:.1f} kbps (margin: {margin_db:.1f} dB)")
        print(f"  Configured rate:   {configured_rate_mbps*1000:.1f} kbps")
        print(f"  Heuristic rate:    {heuristic_rate*1000:.1f} kbps")
        print(f"  Heuristic <= Friis: {heuristic_conservative}")
        print(f"  Margin >= 3dB:     {has_margin_bool}")
        print(f"  {status}")

        results.append({
            "elevation_deg": el,
            "friis_mbps": friis_rate,
            "heuristic_mbps": heuristic_rate,
            "configured_mbps": configured_rate_mbps,
            "margin_db": margin_db,
            "has_margin": bool(has_margin_bool),
            "heuristic_conservative": bool(heuristic_conservative),
        })

    # Monotonicity check
    friis_rates = [r["friis_mbps"] for r in results]
    heuristic_rates = [r["heuristic_mbps"] for r in results]
    friis_mono = all(friis_rates[i] <= friis_rates[i+1] for i in range(len(friis_rates)-1))
    heuristic_mono = all(heuristic_rates[i] <= heuristic_rates[i+1] for i in range(len(heuristic_rates)-1))

    print(f"\n  Friis monotonic: {friis_mono}")
    print(f"  Heuristic monotonic: {heuristic_mono}")

    if not friis_mono:
        all_pass = False
    if not heuristic_mono:
        all_pass = False

    # All elevations >= mask should have margin
    margin_ok = all(r['has_margin'] for r in results if r['elevation_deg'] >= 15)
    if not margin_ok:
        all_pass = False

    # All elevations should have heuristic <= Friis
    conservative_ok = all(r['heuristic_conservative'] for r in results)
    if not conservative_ok:
        all_pass = False

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Link margin >=3dB at all elev >= mask: {'PASS' if margin_ok else 'FAIL'}")
    print(f"  Heuristic conservative (<= Friis): {'PASS' if conservative_ok else 'FAIL'}")
    print(f"  Monotonic (Friis):      {'PASS' if friis_mono else 'FAIL'}")
    print(f"  Monotonic (Heuristic):  {'PASS' if heuristic_mono else 'FAIL'}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")

    # Save (convert numpy types)
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    with open(Path(os.environ.get("ANTARA_VALIDATION_OUTPUT_DIR", Path(__file__).parent)) / "validation_results_comms.json", "w") as f:
        json.dump({
            "overall_pass": all_pass,
            "results": convert(results),
            "monotonic": {"friis": friis_mono, "heuristic": heuristic_mono},
            "configured_rate_mbps": configured_rate_mbps,
        }, f, indent=2)

    return all_pass


if __name__ == "__main__":
    import json
    from pathlib import Path
    import numpy as np
    import sys

    success = main()
    sys.exit(0 if success else 1)
