import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('figures/maneuver_detection', exist_ok=True)
df = pd.read_csv('cache/times_found.csv')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('figures/maneuver_detection', exist_ok=True)
# df = pd.read_csv('times_found.csv')

# Convert delta_v from DU/TU to km/s, keep maneuver_proportion as original
df['dv_km_s'] = (389703 / 382981) * df['delta_v_magnitude']
df['dv_rounded'] = df['dv_km_s'].round(3)
df['mp_rounded'] = df['maneuver_proportion'].round(3)

# Create pivot table for heatmap
heatmap_data = df.groupby(['dv_rounded', 'mp_rounded'])['times_found'].mean().round(3).unstack()

# Create heatmap
plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0.5, 
                      cbar_kws={'label': 'Detection Rate'}, fmt='.3f')

# Convert x-axis labels to percentages
x_labels = [f'{float(label.get_text())*100:.1f}%' for label in heatmap.get_xticklabels()]
heatmap.set_xticklabels(x_labels)

plt.xlabel('Maneuver Proportion (%)')
plt.ylabel('ΔV Magnitude (km/s)') 
plt.title('Detection Rate Heatmap: ΔV Magnitude vs Maneuver Proportion') 
plt.tight_layout()
plt.savefig('figures/maneuver_detection/detection_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Heatmap saved. Data shape: {heatmap_data.shape}")
print(f"Detection range: {heatmap_data.min().min():.3f} to {heatmap_data.max().max():.3f}")
