import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("cache/frontier/mc_1_pdf.csv", index_col=0)

plt.figure(figsize=(8,8))
heatmap = sns.heatmap(df, annot=True, cmap='RdYlBu', center=0.5,
                      cbar_kws={'label': 'Detection Rate'}, fmt='.3f')
x_labels = [f'{float(label.get_text())*100:.1f}%' for label in heatmap.get_xticklabels()]
heatmap.set_xticklabels(x_labels)
y_labels = [f'{float(label.get_text()):.3f}' for label in heatmap.get_yticklabels()]
heatmap.set_yticklabels(y_labels)
plt.xlabel('Maneuver Proportion (%)')
plt.ylabel('ΔV Magnitude (km/s)') 
plt.title('Detection Rate Heatmap: ΔV Magnitude vs Maneuver Proportion') 
plt.tight_layout()
plt.savefig('figures/heatmap/mc_1_pdf.png', dpi=300, bbox_inches='tight')
plt.close()
