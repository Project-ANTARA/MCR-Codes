import orekit
orekit.initVM()

from paths import figures_path, init_orekit_data, results_path

init_orekit_data()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# =========================================================
# MISSION PARAMETERS
# =========================================================
LOAD_POWER = 8.0
ORBIT_PERIOD_MIN = 97.0
orbit_period_hr = ORBIT_PERIOD_MIN / 60.0

battery_sizes = [10, 15, 20, 25, 30]

# ---------------------------------------------------------
# RAAN eclipse table (from Orekit eclipse analysis)
# ---------------------------------------------------------
raan_table = {
    0: 36.1, 15: 35.3, 30: 33.2, 45: 29.0, 60: 17.0,
    75: 0.0, 90: 0.0, 105: 0.0, 120: 20.3, 135: 30.3,
    150: 34.0, 165: 36.1, 180: 36.5, 195: 36.5, 210: 34.4,
    225: 31.1, 240: 21.2, 255: 0.0, 270: 0.0, 285: 0.0,
    300: 22.4, 315: 31.1, 330: 34.4, 345: 36.1
}

raan_vals = sorted(raan_table.keys())

# =========================================================
# DoD TRADE COMPUTATION
# =========================================================
results = []

for batt in battery_sizes:
    for r in raan_vals:
        eclipse_fraction = raan_table[r] / 100.0
        eclipse_time = eclipse_fraction * orbit_period_hr
        eclipse_energy = LOAD_POWER * eclipse_time
        dod = eclipse_energy / batt

        results.append([r, batt, dod])

df = pd.DataFrame(results, columns=["RAAN_deg", "Battery_Wh", "DoD"])
df.to_csv(results_path("tradecase", "battery_DoD_trade.csv"), index=False)

print("Saved: battery_DoD_trade.csv")

# =========================================================
# DoD ENVELOPE PLOT
# =========================================================
plt.figure()

for batt in battery_sizes:
    subset = df[df["Battery_Wh"] == batt]
    plt.plot(subset["RAAN_deg"], subset["DoD"], label=f"{batt} Wh")

plt.xlabel("RAAN (deg)")
plt.ylabel("Depth of Discharge (DoD)")
plt.title("Battery DoD Envelope vs RAAN")
plt.legend()
plt.grid(True)
plt.savefig(figures_path("tradecase", "DoD_envelope.png"), dpi=300)
plt.show()

print("Saved: DoD_envelope.png")

# =========================================================
# SOC SIMULATION (WORST RAAN)
# =========================================================
def simulate_soc(battery_wh, eclipse_pct, orbits=10):
    dt = 10.0
    SOC = 1.0
    soc_history = []

    orbit_seconds = ORBIT_PERIOD_MIN * 60
    eclipse_duration = eclipse_pct / 100.0 * orbit_seconds

    for _ in range(orbits):
        t = 0
        while t < orbit_seconds:

            if t < eclipse_duration:
                Pgen = 0
            else:
                Pgen = LOAD_POWER + 1.3  # sunlight recharge margin

            Pnet = Pgen - LOAD_POWER
            SOC += (Pnet * dt / 3600.0) / battery_wh
            SOC = max(0, min(1, SOC))

            soc_history.append(SOC)
            t += dt

    return soc_history

soc = simulate_soc(25, 36.5)

plt.figure()
plt.plot(soc)
plt.xlabel("Time step")
plt.ylabel("SOC")
plt.title("Battery SOC vs Time (Worst Eclipse)")
plt.grid(True)
plt.savefig(figures_path("tradecase", "SOC_simulation.png"), dpi=300)
plt.show()

print("Saved: SOC_simulation.png")

# =========================================================
# BATTERY LIFETIME MARGIN
# =========================================================
required_cycles = 6200

cycle_life_estimates = {
    10: 1500,
    15: 4000,
    20: 6000,
    25: 8000,
    30: 10000
}

summary_rows = []

for batt in battery_sizes:
    max_dod = df[df["Battery_Wh"] == batt]["DoD"].max()
    margin = cycle_life_estimates[batt] - required_cycles
    summary_rows.append([batt, max_dod, margin])

summary_df = pd.DataFrame(
    summary_rows,
    columns=["Battery_Wh", "Max_DoD", "Cycle_Margin"]
)

summary_df.to_csv(results_path("tradecase", "battery_lifetime_summary.csv"), index=False)

print("\nBattery Lifetime Summary")
print(summary_df)
print("Saved: battery_lifetime_summary.csv")
