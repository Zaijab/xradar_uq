import pickle

import jax


def load_rse_results(filepath):
    """Load RSE results from pickle."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    # Convert numpy arrays back to JAX
    return convert_numpy_to_jax(results)

def convert_jax_to_numpy(obj):
    """Recursively convert JAX arrays to numpy."""
    if isinstance(obj, jnp.ndarray):
        return jnp.asarray(obj)  # Ensures numpy conversion
    elif isinstance(obj, dict):
        return {k: convert_jax_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_jax_to_numpy(item) for item in obj)
    else:
        return obj

def convert_numpy_to_jax(obj):
    """Recursively convert numpy arrays to JAX."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return jnp.array(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_to_jax(item) for item in obj)
    else:
        return obj

loaded_results = load_rse_results("rse_results.pkl")
loaded_results.keys()

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)

dynamical_system = CR3BP()
measurement_system = Radar()
stochastic_filter = EnGMF()
initial_state = dynamical_system.initial_state()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

### Insert Here

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
import jax.numpy as jnp
import jax
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def plot_reachable_set_with_particles(reachable_mesh_data, satellite_state, dynamical_system, 
                                     t_propagate, rng_key, figsize=(15, 12)):
    """
    Plot satellite, reachable set boundary, EnGMF particles, and constrained particles.
    All propagated to the same future time to be consistent.
    """
    # Create figure with uniform subplot sizing
    fig = plt.figure(figsize=figsize)
    
    # Create uniform 2x2 grid of subplots
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2) 
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    
    print("Available keys in reachable_mesh_data:")
    for key_name in reachable_mesh_data.keys():
        print(f"  {key_name}: {type(reachable_mesh_data[key_name])}")
        if hasattr(reachable_mesh_data[key_name], 'shape'):
            print(f"    Shape: {reachable_mesh_data[key_name].shape}")
    
    # Extract final positions from trajectories
    if 'combined_trajectories' in reachable_mesh_data:
        trajectories = reachable_mesh_data['combined_trajectories']
        stable_positions = np.array(trajectories[:, -1, :3])
        print(f"Using final positions from combined_trajectories: {stable_positions.shape}")
    elif 'stable_trajectories' in reachable_mesh_data:
        trajectories = reachable_mesh_data['stable_trajectories'] 
        stable_positions = np.array(trajectories[:, -1, :3])
        print(f"Using final positions from stable_trajectories: {stable_positions.shape}")
    elif 'trajectories' in reachable_mesh_data:
        trajectories = reachable_mesh_data['trajectories']
        stable_positions = np.array(trajectories[:, -1, :3])
        print(f"Using final positions from trajectories: {stable_positions.shape}")
    else:
        all_vertices = reachable_mesh_data['vertices']
        if 'vertex_outcomes' in reachable_mesh_data:
            vertex_outcomes = reachable_mesh_data['vertex_outcomes']
            stable_mask = vertex_outcomes < 3
            stable_positions = np.array(all_vertices[stable_mask, :3])
        else:
            stable_positions = np.array(all_vertices[:, :3])
        print(f"Fallback: using vertices (may be initial positions): {stable_positions.shape}")
    
    print(f"Reachable set position range:")
    print(f"  X: [{stable_positions[:, 0].min():.3f}, {stable_positions[:, 0].max():.3f}]")
    print(f"  Y: [{stable_positions[:, 1].min():.3f}, {stable_positions[:, 1].max():.3f}]") 
    print(f"  Z: [{stable_positions[:, 2].min():.3f}, {stable_positions[:, 2].max():.3f}]")
    
    # Compute convex hull for boundary visualization
    try:
        hull_3d = ConvexHull(stable_positions)
        boundary_points = stable_positions[hull_3d.vertices]
    except:
        boundary_points = stable_positions
    
    # Propagate satellite forward to match reachable set time
    sat_propagated = dynamical_system.flow(0.0, t_propagate, jnp.asarray(satellite_state))
    sat_pos = np.array(sat_propagated[:3])
    print(f"Satellite position (propagated): [{sat_pos[0]:.3f}, {sat_pos[1]:.3f}, {sat_pos[2]:.3f}]")
    
    # Generate EnGMF particles around CURRENT satellite position
    current_rng_key = rng_key
    current_rng_key, subkey = jax.random.split(current_rng_key)
    n_particles = 100
    large_cov = 0.02 * np.eye(6)
    current_sat_pos = np.array(satellite_state)
    engmf_particles_6d = np.array(jax.random.multivariate_normal(
        subkey, current_sat_pos, large_cov, shape=(n_particles,)
    ))
    
    print(f"Generated {n_particles} EnGMF particles around current satellite")
    
    # Propagate EnGMF particles forward to match reachable set time
    propagated_particles = []
    for i, particle in enumerate(engmf_particles_6d):
        try:
            prop_particle = dynamical_system.flow(0.0, t_propagate, jnp.asarray(particle))
            propagated_particles.append(prop_particle[:3])
        except Exception as e:
            print(f"Particle {i} propagation failed: {e}")
            continue
    
    print(f"Successfully propagated {len(propagated_particles)}/{n_particles} particles")
    
    if len(propagated_particles) > 10:
        engmf_particles = np.array(propagated_particles)
    else:
        print("Too few particles propagated successfully, using fallback")
        current_rng_key, subkey = jax.random.split(current_rng_key)
        engmf_particles = np.array(jax.random.multivariate_normal(
            subkey, sat_pos, 0.05 * np.eye(3), shape=(50,)
        ))
    
    # Apply proper convex hull point-in-polygon test
    print(f"Applying proper convex hull rejection sampling...")
    
    def is_inside_convex_hull(point, hull_points):
        """Proper convex hull containment test using linear programming approach."""
        try:
            hull = ConvexHull(hull_points)
            # Add the test point to the hull points
            new_points = np.vstack([hull_points, point.reshape(1, -1)])
            new_hull = ConvexHull(new_points)
            # If the new hull has same vertices as original, point is inside
            return len(new_hull.vertices) == len(hull.vertices)
        except:
            # Fallback to distance-based test
            center = np.mean(hull_points, axis=0)
            max_dist = np.max(np.linalg.norm(hull_points - center, axis=1))
            return np.linalg.norm(point - center) <= max_dist * 0.8  # 80% of max radius
    
    # Apply rejection sampling to EnGMF particles
    constrained_particles = []
    for particle in engmf_particles:
        if is_inside_convex_hull(particle, stable_positions):
            constrained_particles.append(particle)
    
    constrained_particles = np.array(constrained_particles)
    print(f"Rejection sampling results: {len(constrained_particles)}/{len(engmf_particles)} particles kept ({len(constrained_particles)/len(engmf_particles)*100:.1f}%)")
    
    # If too few particles survived, use more lenient test
    if len(constrained_particles) < 5:
        print("Too few particles survived, using very lenient discriminator...")
        constrained_particles = []
        center = np.mean(stable_positions, axis=0)
        radius = np.max(np.linalg.norm(stable_positions - center, axis=1)) * 0.9
        
        for particle in engmf_particles:
            if np.linalg.norm(particle - center) <= radius:
                constrained_particles.append(particle)
        constrained_particles = np.array(constrained_particles)
        print(f"Lenient discriminator: {len(constrained_particles)}/{len(engmf_particles)} particles kept")

    def plot_2d_boundary(ax, positions, proj_dims, title):
        """Plot 2D projection with convex hull boundary."""
        proj_positions = positions[:, proj_dims]
        
        # Plot boundary
        try:
            hull_2d = ConvexHull(proj_positions)
            hull_points = proj_positions[hull_2d.vertices]
            hull_points = np.vstack([hull_points, hull_points[0]])  # Close the loop
            ax.plot(hull_points[:, 0], hull_points[:, 1], '#1f77b4', linewidth=3, 
                   alpha=0.8, label='Reachable Set Boundary')
            ax.fill(hull_points[:, 0], hull_points[:, 1], '#1f77b4', 
                   alpha=0.1, label='Reachable Region')
        except:
            ax.scatter(proj_positions[:, 0], proj_positions[:, 1], 
                      c='#1f77b4', alpha=0.3, s=2, marker='.')
        
        # Plot particles
        if len(engmf_particles) > 0:
            ax.scatter(engmf_particles[:, proj_dims[0]], engmf_particles[:, proj_dims[1]],
                      c='#ff7f0e', alpha=0.8, s=35, marker='o', 
                      label='EnGMF Particles', edgecolors='#d62728', linewidth=0.8)
        if len(constrained_particles) > 0:
            ax.scatter(constrained_particles[:, proj_dims[0]], constrained_particles[:, proj_dims[1]],
                      c='#2ca02c', alpha=0.9, s=40, marker='x', linewidth=2,
                      label='Constrained Particles')
        ax.scatter(sat_pos[proj_dims[0]], sat_pos[proj_dims[1]], c='#ffbb00', s=200, marker='*', 
                  label='Satellite', edgecolors='black', linewidth=2, zorder=10)
        
        ax.set_xlabel(f'{["X", "Y", "Z"][proj_dims[0]]} Position')
        ax.set_ylabel(f'{["X", "Y", "Z"][proj_dims[1]]} Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Calculate global bounds for consistent axis limits
    all_points = [stable_positions, sat_pos.reshape(1, -1)]
    if len(engmf_particles) > 0:
        all_points.append(engmf_particles)
    if len(constrained_particles) > 0:
        all_points.append(constrained_particles)
    
    all_points = np.vstack(all_points)
    x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
    y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
    z_min, z_max = all_points[:, 2].min() - 0.1, all_points[:, 2].max() + 0.1
    
    # Plot 2D projections with boundaries
    plot_2d_boundary(ax1, stable_positions, [0, 1], 'XY Projection')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    plot_2d_boundary(ax2, stable_positions, [0, 2], 'XZ Projection')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(z_min, z_max)
    
    plot_2d_boundary(ax3, stable_positions, [1, 2], 'YZ Projection')
    ax3.set_xlim(y_min, y_max)
    ax3.set_ylim(z_min, z_max)
    
    # Compute RMSE metrics
    def compute_rmse(particles, satellite_pos):
        """Compute Root Mean Square Error between particles and satellite."""
        if len(particles) == 0:
            return float('inf')
        diff = particles - satellite_pos
        mse = np.mean(np.sum(diff**2, axis=1))
        return np.sqrt(mse)
    
    # Calculate RMSE for both particle sets
    engmf_rmse = compute_rmse(engmf_particles, sat_pos) if len(engmf_particles) > 0 else float('inf')
    constrained_rmse = compute_rmse(constrained_particles, sat_pos) if len(constrained_particles) > 0 else float('inf')
    
    print(f"\n=== RMSE Analysis ===")
    print(f"EnGMF particles RMSE:        {engmf_rmse:.4f}")
    print(f"Constrained particles RMSE:  {constrained_rmse:.4f}")
    if constrained_rmse > 0 and constrained_rmse != float('inf'):
        print(f"RMSE improvement ratio:      {engmf_rmse/constrained_rmse:.2f}x")
    
    # Plot 3D with mesh boundary
    try:
        hull_3d = ConvexHull(stable_positions)
        
        # Create triangular faces for the convex hull
        faces = []
        for simplex in hull_3d.simplices:
            triangle = stable_positions[simplex]
            faces.append(triangle)
        
        # Plot the hull surface
        poly3d = [[faces[j][i] for i in range(len(faces[j]))] for j in range(len(faces))]
        ax4.add_collection3d(Poly3DCollection(poly3d, alpha=0.15, facecolor='#1f77b4', 
                                             edgecolor='#1f77b4', linewidth=0.5))
        
        # Plot boundary vertices
        ax4.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
                   c='#1f77b4', alpha=0.6, s=8, marker='.', label='Boundary Points')
        
    except Exception as e:
        # Fallback: just plot sample points
        n_sample_3d = min(500, len(stable_positions))
        current_rng_key, subkey = jax.random.split(current_rng_key)
        sample_indices = jax.random.choice(subkey, len(stable_positions), 
                                          shape=(n_sample_3d,), replace=False)
        sample_positions = stable_positions[sample_indices]
        ax4.scatter(sample_positions[:, 0], sample_positions[:, 1], sample_positions[:, 2],
                   c='#1f77b4', alpha=0.3, s=3, marker='.')
    
    # Plot particles with consistent colors and markers
    if len(engmf_particles) > 0:
        ax4.scatter(engmf_particles[:, 0], engmf_particles[:, 1], engmf_particles[:, 2],
                   c='#ff7f0e', alpha=0.8, s=35, marker='o', 
                   label='EnGMF Particles', edgecolors='#d62728', linewidth=0.8)
    if len(constrained_particles) > 0:
        ax4.scatter(constrained_particles[:, 0], constrained_particles[:, 1], constrained_particles[:, 2],
                   c='#2ca02c', alpha=0.9, s=40, marker='x', linewidth=2,
                   label='Constrained Particles')
    ax4.scatter(sat_pos[0], sat_pos[1], sat_pos[2], c='#ffbb00', s=200, marker='*',
               label='Satellite', edgecolors='black', linewidth=2, zorder=10)
    
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position') 
    ax4.set_zlabel('Z Position')
    ax4.set_title('3D Reachable Set Boundary')
    
    # Set consistent bounds for 3D plot
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)
    ax4.set_zlim(z_min, z_max)
    
    # Create explicit legend elements
    legend_elements = [
        mlines.Line2D([], [], color='#1f77b4', linewidth=3, label='Reachable Set Boundary'),
        mpatches.Patch(color='#1f77b4', alpha=0.1, label='Reachable Region'),
        mlines.Line2D([], [], marker='o', color='#ff7f0e', linestyle='None', 
                     markersize=8, markeredgecolor='#d62728', markeredgewidth=1, label='EnGMF Particles'),
        mlines.Line2D([], [], marker='x', color='#2ca02c', linestyle='None', 
                     markersize=8, markeredgewidth=2, label='Constrained Particles'),
        mlines.Line2D([], [], marker='*', color='#ffbb00', linestyle='None', 
                     markersize=12, markeredgecolor='black', markeredgewidth=1, label='Satellite')
    ]
    
    # Position legend outside the plot area
    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    
    # Adjust layout for uniform subplot sizes and legend space
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # Leave space for legend on the right
    
    print(f"Propagation time: {t_propagate}")
    print(f"Reachable set vertices: {len(stable_positions)}")
    print(f"EnGMF particles (propagated): {len(engmf_particles)}")
    print(f"Constrained particles: {len(constrained_particles)}")
    print(f"Satellite position (propagated): [{sat_pos[0]:.3f}, {sat_pos[1]:.3f}, {sat_pos[2]:.3f}]")
    
    return fig

###


reachable_mesh_data = loaded_results['complete_mesh_data']
satellite_state = dynamical_system.initial_state()  # 6D state at current time
t_propagate = 10.0  # Should match the time used in RSE algorithm
key = jax.random.key(42)
fig = plot_reachable_set_with_particles(reachable_mesh_data, satellite_state, 
                                       dynamical_system, t_propagate, key)

fig.savefig('figures/reachability/particles.png')


# The plot now shows everything at the SAME TIME:
# - Reachable set: All positions reachable after t_propagate time with ΔV maneuvers
# - Satellite: Propagated forward by t_propagate (should be near center of reachable set)  
# - EnGMF particles: Started around current satellite, propagated forward by t_propagate
# - Constrained particles: Represent what rejection sampling would produce
# This way the satellite should appear inside/near the center of the reachable set!
