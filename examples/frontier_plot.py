import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures/detection_boundary', exist_ok=True)

def otsu_threshold(data):
    values = data.values.flatten()
    values = values[~np.isnan(values)]
    hist, bin_edges = np.histogram(values, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    total_weight = hist.sum()
    total_mean = (hist * bin_centers).sum() / total_weight
    
    max_variance = 0
    optimal_threshold = bin_centers[0]
    
    for i, threshold in enumerate(bin_centers):
        w0 = hist[:i+1].sum()
        w1 = hist[i+1:].sum()
        
        if w0 == 0 or w1 == 0:
            continue
            
        mean0 = (hist[:i+1] * bin_centers[:i+1]).sum() / w0 if w0 > 0 else 0
        mean1 = (hist[i+1:] * bin_centers[i+1:]).sum() / w1 if w1 > 0 else 0
        
        between_class_variance = w0 * w1 * (mean0 - mean1) ** 2
        
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = threshold
    
    return optimal_threshold

detection_heatmap = pd.read_csv('cache/heatmap/10_mc_0_001_0_5_20_0_100_10.csv', index_col=0)
pdf_frontier_data = pd.read_csv('cache/frontier/mc_1_pdf.csv', index_col=0)
random_frontier_data = pd.read_csv('cache/frontier/mc_1_random.csv', index_col=0)

assert detection_heatmap.shape == (20, 10)
assert pdf_frontier_data.shape[0] >= 2
assert random_frontier_data.shape[0] >= 2

detection_threshold = otsu_threshold(detection_heatmap)

plt.figure()
plt.contour(detection_heatmap, levels=[detection_threshold], colors='black', linewidths=2, alpha=0.7)
plt.plot(pdf_frontier_data.iloc[:, 0], pdf_frontier_data.iloc[:, 1], label='PDF', linewidth=3, color='blue')
plt.plot(random_frontier_data.iloc[:, 0], random_frontier_data.iloc[:, 1], label='Random', linewidth=3, color='red')

all_x_coords = np.concatenate([pdf_frontier_data.iloc[:, 0], random_frontier_data.iloc[:, 0]])
all_y_coords = np.concatenate([pdf_frontier_data.iloc[:, 1], random_frontier_data.iloc[:, 1]])
plt.xlim(all_x_coords.min() - 0.5, all_x_coords.max() + 0.5)
plt.ylim(all_y_coords.min() - 0.5, all_y_coords.max() + 0.5)
plt.legend()
plt.savefig('figures/detection_boundary/frontier_overlay.png')
plt.close()
