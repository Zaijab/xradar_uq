import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Read the CSV data
df = pd.read_csv('cache/times_found_pdf_sensor.csv')

# Create a pivot table for easier analysis
pivot_data = df.pivot_table(
    values='times_found', 
    index='delta_v_magnitude', 
    columns='maneuver_proportion', 
    aggfunc='mean'
)

# Define custody threshold (detection rate below which we "lose custody")
custody_threshold = 0.5

# Find frontier points for each maneuver proportion
frontier_points = []

for maneuver_prop in pivot_data.columns:
    column_data = pivot_data[maneuver_prop].dropna()
    
    # Find the first ΔV where detection drops below threshold
    below_threshold = column_data < custody_threshold
    if below_threshold.any():
        frontier_deltav = column_data[below_threshold].index[0]
        frontier_points.append((maneuver_prop * 100, frontier_deltav))

# Convert to arrays for plotting
frontier_points = np.array(frontier_points)
if len(frontier_points) > 0:
    maneuver_props = frontier_points[:, 0]  # Convert to percentage
    delta_vs = frontier_points[:, 1]

    # Plot the frontier
    plt.figure(figsize=(12, 8))
    plt.plot(maneuver_props, delta_vs, 'r-o', linewidth=3, markersize=8, 
             label='Custody Frontier', markerfacecolor='white', markeredgewidth=2)
    
    # Add some styling
    plt.xlabel('Maneuver Proportion (%)', fontsize=14)
    plt.ylabel('ΔV Magnitude (km/s)', fontsize=14)
    plt.title('Spacecraft Custody Frontier', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Print frontier coordinates
    print("Custody Frontier Points (Maneuver %, ΔV km/s):")
    for mp, dv in zip(maneuver_props, delta_vs):
        print(f"({mp:.1f}%, {dv:.3f})")
    
    plt.tight_layout()
    plt.savefig("figures/frontier/custody_frontier_second_sensor.png")
    plt.show()
