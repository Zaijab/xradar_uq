import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

os.makedirs('figures/frontier_analysis', exist_ok=True)
filename = 'mc_1_random_with_prop_prior'
df = pd.read_csv(f'cache/frontier/new/{filename}.csv', index_col=0) #pd.read_csv('cache/frontier/mc_1_pdf.csv', index_col=0) - pd.read_csv('cache/frontier/mc_1_random.csv', index_col=0)
assert df.shape == (20, 10)

du_tu_to_km_s_conversion = 389703 / 382981
row_indices_km_s = df.index * du_tu_to_km_s_conversion
column_labels_percentage = [f'{float(col)*100:.1f}%' for col in df.columns]

df_converted = df.copy()
df_converted.index = row_indices_km_s.round(3)

plt.figure(figsize=(12, 8))
heatmap = sns.heatmap(df_converted, annot=True, cmap='RdYlBu', center=0.5, 
                      cbar_kws={'label': 'Detection Rate'}, fmt='.3f')
heatmap.set_xticklabels(column_labels_percentage)

plt.xlabel('Maneuver Proportion (%)')
plt.ylabel('ΔV Magnitude (km/s)')
plt.title('Detection Rate: ΔV Magnitude vs Maneuver Proportion')
plt.tight_layout()
plt.savefig(f'figures/frontier_analysis/{filename}.png', dpi=300, bbox_inches='tight')
