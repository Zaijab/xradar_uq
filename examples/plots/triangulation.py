import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def plot_triangulation_progression(ensemble_angles: Float[Array, "n_samples 2"], 
                                 subdivision_levels: int = 2):
    """Plot convex hull and triangulation refinement progression."""
    
    # Handle azimuth wraparound
    azimuth_range = jnp.max(ensemble_angles[:, 0]) - jnp.min(ensemble_angles[:, 0])
    if azimuth_range > jnp.pi:
        wrapped_angles = jnp.where(ensemble_angles[:, 0] < 0, 
                                 ensemble_angles[:, 0] + 2*jnp.pi, 
                                 ensemble_angles[:, 0])
        hull_points = jnp.column_stack([wrapped_angles, ensemble_angles[:, 1]])
    else:
        hull_points = ensemble_angles
    
    # Create convex hull using PolytopAX
    from polytopax import ConvexHull
    my_hull = ConvexHull.from_points(hull_points)
    
    # Get triangulations at different levels
    triangulations = []
    triangles = fan_triangulate(my_hull)
    triangulations.append(triangles)
    
    for level in range(subdivision_levels):
        triangles = subdivide_triangles(triangles)
        triangulations.append(triangles)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, len(triangulations), figsize=(5*len(triangulations), 10))
    if len(triangulations) == 1:
        axes = axes.reshape(2, 1)
    
    hull_vertices = my_hull.vertices_array()
    centroid = my_hull.centroid()
    
    for col, triangles in enumerate(triangulations):
        n_triangles = triangles.shape[0]
        level_name = f"Level {col}" if col == 0 else f"Subdivision {col}"
        
        # Top row: Show triangulation structure
        ax_top = axes[0, col]
        
        # Plot original data points
        ax_top.scatter(ensemble_angles[:, 0], ensemble_angles[:, 1], 
                      c='red', s=8, alpha=0.6, label='Data Points')
        
        # Plot hull boundary
        hull_closed = np.vstack([hull_vertices, hull_vertices[0]])
        ax_top.plot(hull_closed[:, 0], hull_closed[:, 1], 'k-', linewidth=2, label='Hull')
        
        # Plot triangles
        for i in range(n_triangles):
            tri = triangles[i]
            triangle = patches.Polygon(tri, fill=False, edgecolor='blue', 
                                     linewidth=0.5, alpha=0.7)
            ax_top.add_patch(triangle)
        
        # Plot centroid
        ax_top.scatter(*centroid, c='green', s=50, marker='*', label='Centroid')
        
        ax_top.set_title(f'{level_name}\n{n_triangles} triangles')
        ax_top.set_xlabel('Azimuth (rad)')
        ax_top.set_ylabel('Elevation (rad)')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_aspect('equal')
        if col == 0:
            ax_top.legend()
        
        # Bottom row: Show triangle density
        ax_bottom = axes[1, col]
        
        # Create filled triangles with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, n_triangles))
        for i in range(n_triangles):
            tri = triangles[i]
            triangle = patches.Polygon(tri, facecolor=colors[i], 
                                     edgecolor='white', linewidth=0.2, alpha=0.8)
            ax_bottom.add_patch(triangle)
        
        # Plot original data points on top
        ax_bottom.scatter(ensemble_angles[:, 0], ensemble_angles[:, 1], 
                         c='red', s=8, alpha=0.9, edgecolors='white', linewidth=0.5)
        
        ax_bottom.set_title(f'Density Visualization\n{n_triangles} regions')
        ax_bottom.set_xlabel('Azimuth (rad)')  
        ax_bottom.set_ylabel('Elevation (rad)')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_aspect('equal')
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('figures/triangulation_analysis', exist_ok=True)
    plt.savefig('figures/triangulation_analysis/triangulation_progression.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nTriangulation Statistics:")
    for level, triangles in enumerate(triangulations):
        level_name = f"Level {level}" if level == 0 else f"Subdivision {level}"
        print(f"  {level_name}: {triangles.shape[0]} triangles")
    
    return triangulations

# Usage with your data:
triangulations = plot_triangulation_progression(ensemble_angles, subdivision_levels=3)

# Additional function to show triangle centroids for PDF evaluation
def plot_triangle_centroids(triangulations):
    """Plot centroids of triangles for PDF evaluation points."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['red', 'blue', 'green', 'purple']
    
    for level, triangles in enumerate(triangulations[:4]):  # Show first 4 levels
        # Compute triangle centroids
        centroids = jnp.mean(triangles, axis=1)
        
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c=colors[level], s=20, alpha=0.7, 
                  label=f'Level {level} ({triangles.shape[0]} points)')
    
    ax.set_xlabel('Azimuth (rad)')
    ax.set_ylabel('Elevation (rad)')
    ax.set_title('Triangle Centroids for PDF Evaluation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.savefig('figures/triangulation_analysis/triangle_centroids.png', 
                dpi=150, bbox_inches='tight')
    plt.show()

plot_triangle_centroids(triangulations)
