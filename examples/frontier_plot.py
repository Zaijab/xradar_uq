import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_and_plot_frontier_from_csv(csv_path: str):
    """Extract frontier directly from the CSV data."""
    
    # Create directories
    Path("cache/frontier").mkdir(parents=True, exist_ok=True)
    Path("figures/frontier").mkdir(parents=True, exist_ok=True)
    
    # Load and examine the data
    df = pd.read_csv(csv_path)
    print("Data shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Detection rate range:", df['times_found'].min(), "to", df['times_found'].max())
    
    # Convert to physical units
    VU_to_kmps = 389703 / 382981
    df['delta_v_km_s'] = df['delta_v_magnitude'] * VU_to_kmps
    df['maneuver_proportion_pct'] = df['maneuver_proportion'] * 100
    
    # Find frontier points: for each maneuver proportion, find the LOWEST delta_v 
    # where you LOSE custody (detection rate drops below 0.8)
    frontier_points = []
    
    for maneuver_prop in sorted(df['maneuver_proportion'].unique()):
        subset = df[df['maneuver_proportion'] == maneuver_prop].copy()
        subset = subset.sort_values('delta_v_magnitude')
        
        # Find where custody is lost (detection rate < 0.8)
        custody_lost = subset[subset['times_found'] < 0.8]
        
        if len(custody_lost) > 0:
            # Frontier is the LOWEST ΔV where custody is lost
            frontier_dv = custody_lost['delta_v_magnitude'].min()
            
            frontier_points.append({
                'maneuver_proportion': maneuver_prop,
                'delta_v_magnitude': frontier_dv,
                'maneuver_proportion_pct': maneuver_prop * 100,
                'delta_v_km_s': frontier_dv * VU_to_kmps
            })
    
    # Convert to dataframe and save
    frontier_df = pd.DataFrame(frontier_points)
    frontier_df.to_csv("cache/frontier/frontier_points.csv", index=False)
    
    print(f"Extracted {len(frontier_df)} frontier points")
    
    # Plot the frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(frontier_df['maneuver_proportion_pct'], 
           frontier_df['delta_v_km_s'],
           'ro-', linewidth=2, markersize=6, label='Custody Frontier')
    
    ax.set_xlabel('Maneuver Proportion (%)')
    ax.set_ylabel('ΔV Magnitude (km/s)')
    ax.set_title('Custody Frontier')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("figures/frontier/custody_frontier.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return frontier_df
# Run it
frontier_df = extract_and_plot_frontier_from_csv("cache/times_found.csv")
print("\nFrontier summary:")
print(frontier_df.head(10))
