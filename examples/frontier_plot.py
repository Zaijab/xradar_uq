import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/detection_boundary', exist_ok=True)
os.makedirs('cache', exist_ok=True)

detection_data = pd.read_csv('cache/heatmap/10_mc_0_001_0_5_20_0_100_10.csv', index_col=0)
assert detection_data.shape == (20, 10)

delta_v_values = detection_data.index.values
maneuver_proportions = detection_data.columns.astype(float).values
detection_threshold = 0.5

plt.figure()
plt.contour(detection_data, levels=jnp.array([0.5]))
plt.savefig('figures/heatmap/contour.png')
plt.close()
