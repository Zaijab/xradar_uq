import pandas as pd

df = pd.read_csv("cache/times_found.csv")

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs('figures/maneuver_detection', exist_ok=True)
# df = pd.read_csv('times_found.csv')

# Convert delta_v from DU/TU to km/s, keep maneuver_proportion as original
df['dv_km_s'] = (389703 / 382981) * df['delta_v_magnitude'] 
df['dv_rounded'] = df['dv_km_s'].round(3)
df['mp_rounded'] = df['maneuver_proportion'].round(3)

# Filter data: delta_v vs detection rate with fixed maneuver proportion
mp_fixed = 0.044  # Original decimal value
dv_data = df[df['mp_rounded'] == mp_fixed].groupby('dv_rounded')['times_found'].mean().round(3)

# Filter data: maneuver proportion vs detection rate with fixed delta_v (convert to km/s)
dv_fixed_original = 0.048  # Original DU/TU value
dv_fixed = round((389703 / 382981) * dv_fixed_original, 3)  # Convert to km/s
mp_data = df[df['dv_rounded'] == dv_fixed].groupby('mp_rounded')['times_found'].mean().round(3)

# Create plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(dv_data.index, dv_data.values, 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('ΔV Magnitude (km/s)'); ax1.set_ylabel('Detection Rate')
ax1.set_title(f'ΔV vs Detection Rate\n(Maneuver Proportion = {mp_fixed*100:.1f}%)')
ax1.grid(True, alpha=0.3)

ax2.plot(mp_data.index, mp_data.values, 'ro-', linewidth=2, markersize=6)
ax2.set_xlabel('Maneuver Proportion (%)'); ax2.set_ylabel('Detection Rate')
ax2.set_title(f'Maneuver Proportion vs Detection Rate\n(ΔV Magnitude = {dv_fixed:.3f} km/s)')
# Convert x-axis labels to percentages
ax2.set_xticklabels([f'{x*100:.1f}' for x in ax2.get_xticks()])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/maneuver_detection/fixed_value_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Fixed value plots saved. Data points: {len(dv_data)}, {len(mp_data)}")
