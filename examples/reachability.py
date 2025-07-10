import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from tqdm.auto import tqdm

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF


key = jax.random.key(42)

dynamical_system = CR3BP()
measurement_system = Radar()
stochastic_filter = EnGMF()
initial_state = dynamical_system.initial_state()


delta_v_magnitude = 1e-2
seeds = 900
outer = 400

key, subkey = jax.random.split(key)
seed_vertices = jax.random.multivariate_normal(subkey, shape=(seeds,),mean=jnp.zeros(3), cov=jnp.eye(3))
seed_vertices /= jnp.linalg.norm(seed_vertices, axis=1, keepdims=True)
key, subkey = jax.random.split(key)
radii = jax.random.uniform(subkey, shape=(seeds,1)) ** (1/3)
seed_vertices = radii * seed_vertices
key, subkey = jax.random.split(key)
outer_vertices = jax.random.multivariate_normal(subkey, shape=(outer,),mean=jnp.zeros(3), cov=jnp.eye(3))
outer_vertices /= jnp.linalg.norm(outer_vertices, axis=1, keepdims=True)

import matplotlib.pyplot as plt
from pathlib import Path

def plot_step1_results(interior_vertices, boundary_vertices):
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    LU_to_km = 389703  # km (Earth-Moon distance)
    TU_to_s = 382981   # seconds (≈ 4.348 days)
    VU_to_kmps = LU_to_km / TU_to_s  # ≈ 1.023 km/s
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*(interior_vertices * VU_to_kmps).T, alpha=0.6, label='Interior', s=20)
    ax.scatter(*(boundary_vertices * VU_to_kmps).T, alpha=0.8, label='Boundary', s=30, color='red')
    ax.set_xlabel('ΔVx (km/s)'); ax.set_ylabel('ΔVy (km/s)'); ax.set_zlabel('ΔVz (km/s)')
    ax.legend(); plt.title('Step 1: Seed Vertices in ΔV Sphere')
    plt.savefig("figures/reachability/step1_seed_vertices.png", dpi=150, bbox_inches='tight')

plot_step1_results(seed_vertices, outer_vertices)

import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from pathlib import Path

def build_delaunay_mesh(interior_vertices, boundary_vertices):
    """
    Step 2: Build a mesh from seed vertices using Delaunay triangulation.
    
    Args:
        interior_vertices: Array of shape (n_interior, 3) - interior seed points
        boundary_vertices: Array of shape (n_boundary, 3) - boundary seed points
    
    Returns:
        all_vertices: Combined vertices array
        triangulation: Delaunay triangulation object
        mesh_info: Dictionary with mesh statistics
    """
    
    # Combine all vertices (interior + boundary)
    all_vertices = jnp.concatenate([interior_vertices, boundary_vertices], axis=0)
    
    # Convert to numpy for scipy compatibility
    vertices_np = np.array(all_vertices)
    
    # Build Delaunay triangulation (creates tetrahedra in 3D)
    triangulation = Delaunay(vertices_np)
    
    # Extract mesh information
    n_vertices = len(vertices_np)
    n_tetrahedra = len(triangulation.simplices)
    n_interior = len(interior_vertices)
    n_boundary = len(boundary_vertices)
    
    mesh_info = {
        'n_vertices': n_vertices,
        'n_tetrahedra': n_tetrahedra,
        'n_interior': n_interior,
        'n_boundary': n_boundary,
        'vertices': all_vertices,
        'simplices': triangulation.simplices
    }
    
    print(f"Mesh Statistics:")
    print(f"  Total vertices: {n_vertices}")
    print(f"  Interior vertices: {n_interior}")
    print(f"  Boundary vertices: {n_boundary}")
    print(f"  Tetrahedra: {n_tetrahedra}")
    
    return all_vertices, triangulation, mesh_info

def test_point_inclusion(triangulation):
    """
    Test if arbitrary points are inside/outside the Delaunay mesh.
    """
    print("\n=== Testing Point Inclusion ===")
    
    # Test points
    test_points = np.array([
        [0.0, 0.0, 0.0],      # Center (should be inside)
        [0.5, 0.5, 0.5],      # Interior point (should be inside)
        [1.2, 0.0, 0.0],      # Outside sphere (should be outside)
        [0.95, 0.0, 0.0],     # Near boundary (might be inside/outside)
    ])
    
    for i, point in enumerate(test_points):
        simplex_idx = triangulation.find_simplex(point)
        if simplex_idx >= 0:
            print(f"Point {i+1} {point} is INSIDE (simplex {simplex_idx})")
        else:
            print(f"Point {i+1} {point} is OUTSIDE the mesh")
    
    return test_points

def get_unique_edges(triangulation):
    """
    Extract unique edges from tetrahedra to avoid drawing duplicates.
    """
    edges = set()
    for simplex in triangulation.simplices:
        # Each tetrahedron has 6 edges (4 choose 2)
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                edges.add(edge)
    return list(edges)

def plot_delaunay_mesh(vertices, triangulation, mesh_info):
    """
    Visualize the Delaunay mesh structure with proper connectivity.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    LU_to_km = 389703  # km (Earth-Moon distance)
    TU_to_s = 382981   # seconds (≈ 4.348 days)
    VU_to_kmps = LU_to_km / TU_to_s  # ≈ 1.023 km/s
    
    vertices_km = np.array(vertices) * VU_to_kmps
    n_interior = mesh_info['n_interior']
    
    # Get unique edges for better visualization
    unique_edges = get_unique_edges(triangulation)
    print(f"Total unique edges in mesh: {len(unique_edges)}")
    
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: 3D mesh with all edges
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Plot vertices first
    ax1.scatter(*vertices_km[:n_interior].T, alpha=0.8, label='Interior', s=30, c='blue')
    ax1.scatter(*vertices_km[n_interior:].T, alpha=0.9, label='Boundary', s=40, c='red')
    
    # Draw all edges (sample if too many)
    edge_sample = min(500, len(unique_edges))  # Increase sample size
    for edge in unique_edges[:edge_sample]:
        p1, p2 = vertices_km[edge[0]], vertices_km[edge[1]]
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                'gray', alpha=0.3, linewidth=0.8)
    
    ax1.set_xlabel('ΔVx (km/s)'); ax1.set_ylabel('ΔVy (km/s)'); ax1.set_zlabel('ΔVz (km/s)')
    ax1.legend(); ax1.set_title(f'3D Delaunay Mesh\n({edge_sample}/{len(unique_edges)} edges shown)')
    
    # Plot 2: 2D slice (Z ≈ 0 plane) with triangulation
    ax2 = fig.add_subplot(132)
    
    # Find vertices near Z=0 plane
    z_threshold = 0.1 * VU_to_kmps
    near_z_plane = np.abs(vertices_km[:, 2]) < z_threshold
    
    if np.sum(near_z_plane) > 3:  # Need at least 3 points for triangulation
        slice_vertices = vertices_km[near_z_plane]
        slice_indices = np.where(near_z_plane)[0]
        
        # Create 2D triangulation for the slice
        if len(slice_vertices) > 3:
            from scipy.spatial import Delaunay
            tri_2d = Delaunay(slice_vertices[:, :2])
            ax2.triplot(slice_vertices[:, 0], slice_vertices[:, 1], tri_2d.simplices, 
                       'k-', alpha=0.4, linewidth=0.8)
    
    # Plot all vertices projected to XY
    ax2.scatter(vertices_km[:n_interior, 0], vertices_km[:n_interior, 1], 
               alpha=0.7, label='Interior', s=20, c='blue')
    ax2.scatter(vertices_km[n_interior:, 0], vertices_km[n_interior:, 1], 
               alpha=0.9, label='Boundary', s=30, c='red')
    
    ax2.set_xlabel('ΔVx (km/s)'); ax2.set_ylabel('ΔVy (km/s)')
    ax2.legend(); ax2.set_title('XY Projection (Z≈0 slice)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Connectivity analysis
    ax3 = fig.add_subplot(133)
    ax3.axis('off')
    
    # Calculate connectivity statistics
    vertex_degrees = np.zeros(len(vertices))
    for edge in unique_edges:
        vertex_degrees[edge[0]] += 1
        vertex_degrees[edge[1]] += 1
    
    stats_text = f"""Delaunay Mesh Analysis:

Vertices: {mesh_info['n_vertices']}
  • Interior: {mesh_info['n_interior']}
  • Boundary: {mesh_info['n_boundary']}

Tetrahedra: {mesh_info['n_tetrahedra']}
Unique Edges: {len(unique_edges)}

Connectivity:
  • Avg degree: {vertex_degrees.mean():.1f}
  • Max degree: {int(vertex_degrees.max())}
  • Min degree: {int(vertex_degrees.min())}

Mesh Properties:
  ✓ Space-filling tetrahedra
  ✓ Non-overlapping elements
  ✓ Vertices are seed points
  ✓ Point inclusion testing available
"""
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step2_delaunay_mesh.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return unique_edges

def analyze_mesh_quality(triangulation, vertices):
    """
    Analyze the quality of the Delaunay mesh.
    """
    # Calculate tetrahedra volumes
    volumes = []
    for simplex in triangulation.simplices:
        # Get the 4 vertices of the tetrahedron
        v0, v1, v2, v3 = vertices[simplex]
        
        # Calculate volume using determinant formula
        # V = |det([v1-v0, v2-v0, v3-v0])| / 6
        matrix = np.array([v1-v0, v2-v0, v3-v0]).T
        volume = abs(np.linalg.det(matrix)) / 6.0
        volumes.append(volume)
    
    volumes = np.array(volumes)
    
    print(f"\nMesh Quality Analysis:")
    print(f"  Volume statistics:")
    print(f"    Mean: {volumes.mean():.6f}")
    print(f"    Std:  {volumes.std():.6f}")
    print(f"    Min:  {volumes.min():.6f}")
    print(f"    Max:  {volumes.max():.6f}")
    print(f"  Volume ratio (max/min): {volumes.max()/volumes.min():.2f}")
    
    return volumes

# Execute Step 2
print("=== Step 2: Building Delaunay Mesh ===")
all_vertices, triangulation, mesh_info = build_delaunay_mesh(seed_vertices, outer_vertices)

# Test point inclusion functionality
test_points = test_point_inclusion(triangulation)

# Visualize the mesh with proper connectivity
unique_edges = plot_delaunay_mesh(all_vertices, triangulation, mesh_info)

# Analyze mesh quality
volumes = analyze_mesh_quality(triangulation, np.array(all_vertices))

# Additional mesh analysis
def mesh_inclusion_demo(triangulation, vertices):
    """
    Demonstrate the point inclusion capability with visualization.
    """
    print("\n=== Point Inclusion Demonstration ===")
    
    # Generate random test points
    np.random.seed(42)
    n_test = 100
    test_pts = np.random.uniform(-1.2, 1.2, (n_test, 3))
    
    # Test inclusion
    simplex_indices = triangulation.find_simplex(test_pts)
    inside_pts = test_pts[simplex_indices >= 0]
    outside_pts = test_pts[simplex_indices < 0]
    
    print(f"Tested {n_test} random points:")
    print(f"  Inside mesh: {len(inside_pts)} ({len(inside_pts)/n_test*100:.1f}%)")
    print(f"  Outside mesh: {len(outside_pts)} ({len(outside_pts)/n_test*100:.1f}%)")
    
    # Quick 2D visualization of inclusion test
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    LU_to_km = 389703; TU_to_s = 382981; VU_to_kmps = LU_to_km / TU_to_s
    vertices_km = np.array(vertices) * VU_to_kmps
    
    # Project to XY plane
    ax.scatter(vertices_km[:, 0], vertices_km[:, 1], alpha=0.6, s=20, c='blue', label='Mesh vertices')
    ax.scatter(inside_pts[:, 0] * VU_to_kmps, inside_pts[:, 1] * VU_to_kmps, 
              alpha=0.7, s=15, c='green', label='Points inside', marker='x')
    ax.scatter(outside_pts[:, 0] * VU_to_kmps, outside_pts[:, 1] * VU_to_kmps, 
              alpha=0.7, s=15, c='red', label='Points outside', marker='+')
    
    ax.set_xlabel('ΔVx (km/s)'); ax.set_ylabel('ΔVy (km/s)')
    ax.legend(); ax.set_title('Point Inclusion Test (XY projection)')
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/point_inclusion_demo.png", dpi=150, bbox_inches='tight')
    plt.show()

# Run inclusion demonstration
mesh_inclusion_demo(triangulation, all_vertices)

# Store results for next steps
mesh_data = {
    'vertices': all_vertices,
    'triangulation': triangulation,
    'mesh_info': mesh_info,
    'volumes': volumes,
    'unique_edges': unique_edges,
    'test_points': test_points
}

### Step 3

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffrax import SaveAt
from tqdm.auto import tqdm

def map_deltav_to_6d_states(deltav_vertices, initial_state, delta_v_magnitude):
    """
    Map 3D ΔV vertices to full 6D state space.
    
    According to Komendera paper:
    "RSE then records the topology of this mesh and maps the 3D burn-space points 
    to the full 6D state space via vector addition to X₀."
    
    Args:
        deltav_vertices: Array of shape (n_vertices, 3) - ΔV directions
        initial_state: Array of shape (6,) - initial spacecraft state [r, v]
        delta_v_magnitude: float - magnitude scaling for ΔV
        
    Returns:
        states_6d: Array of shape (n_vertices, 6) - full 6D states
    """
    n_vertices = deltav_vertices.shape[0]
    
    # Scale ΔV vertices by magnitude
    deltav_scaled = deltav_vertices * delta_v_magnitude
    
    # Extract position and velocity from initial state
    r0 = initial_state[:3]  # Initial position
    v0 = initial_state[3:6]  # Initial velocity
    
    # Create 6D states: position unchanged, velocity = v0 + ΔV
    positions = jnp.tile(r0, (n_vertices, 1))  # Same position for all
    velocities = v0 + deltav_scaled  # Add ΔV to initial velocity
    
    states_6d = jnp.concatenate([positions, velocities], axis=1)
    
    print(f"Mapped {n_vertices} ΔV vertices to 6D state space:")
    print(f"  Initial state: {initial_state}")
    print(f"  ΔV magnitude: {delta_v_magnitude}")
    print(f"  Position range: {positions.min():.6f} to {positions.max():.6f}")
    print(f"  Velocity range: {velocities.min():.6f} to {velocities.max():.6f}")
    
    return states_6d

def compute_trajectories_batch(dynamical_system, initial_states, t_final, n_save_points=100):
    """
    Compute t-forward trajectories from each initial state.
    
    Args:
        dynamical_system: Dynamical system (e.g., CR3BP)
        initial_states: Array of shape (n_vertices, 6) - 6D states
        t_final: float - final integration time
        n_save_points: int - number of points to save along each trajectory
        
    Returns:
        trajectories: Array of shape (n_vertices, n_save_points, 6) - all trajectories
        times: Array of shape (n_save_points,) - time points
    """
    
    # Create SaveAt object for trajectory sampling
    times = jnp.linspace(0.0, t_final, n_save_points)
    saveat = SaveAt(ts=times)
    
    print(f"Computing trajectories from {initial_states.shape[0]} vertices...")
    print(f"  Integration time: 0.0 → {t_final}")
    print(f"  Save points: {n_save_points}")
    
    def compute_single_trajectory(initial_state):
        """Compute trajectory for a single initial state."""
        try:
            ts, trajectory = dynamical_system.trajectory(
                initial_time=0.0,
                final_time=t_final,
                state=initial_state,
                saveat=saveat
            )
            return trajectory
        except Exception as e:
            print(f"Warning: Trajectory computation failed: {e}")
            # Return NaN trajectory if integration fails
            return jnp.full((n_save_points, 6), jnp.nan)
    
    # Use vmap to compute all trajectories in parallel
    print("  Using vmap for parallel trajectory computation...")
    trajectories = eqx.filter_vmap(compute_single_trajectory)(initial_states)
    
    # Check for failed integrations
    n_failed = jnp.sum(jnp.isnan(trajectories[:, 0, 0]))
    n_success = initial_states.shape[0] - n_failed
    print(f"  Successful trajectories: {n_success}/{initial_states.shape[0]}")
    if n_failed > 0:
        print(f"  Failed integrations: {n_failed} (marked with NaN)")
    
    return trajectories, times

def remove_unstable_trajectories_rse_step4(trajectories, times, N=10):
    """
    RSE Algorithm Step 4: Remove numerically unstable trajectories.
    
    From Komendera 2015:
    "RSE checks the trajectories produced in step 3 and discards any that
    diverge beyond N times the distance between the main bodies, where N is an 
    algorithm parameter. RSE does this to discard trajectories that make a close 
    approach to the center one of the main bodies (within the impact region) as 
    these may create numerical stability issues due to approaching a singularity."
    
    Args:
        trajectories: Array of shape (n_vertices, n_times, 6)
        times: Array of time points
        N: Algorithm parameter (default=10)
        
    Returns:
        stable_mask: Boolean array indicating which trajectories to keep
        stable_trajectories: Filtered trajectories
        removed_count: Number of removed trajectories
    """
    n_vertices = trajectories.shape[0]
    
    # In CR3BP, distance between main bodies = 1.0 (normalized units)
    distance_between_bodies = 1.0
    divergence_threshold = N * distance_between_bodies
    
    # Check maximum distance during entire trajectory (not just final state)
    # This catches trajectories that diverge at any point during integration
    trajectory_positions = trajectories[:, :, :3]  # Extract positions
    distances_from_origin = jnp.linalg.norm(trajectory_positions, axis=2)
    max_distances = jnp.max(distances_from_origin, axis=1)
    
    # Keep trajectories that don't exceed the divergence threshold
    stable_mask = max_distances <= divergence_threshold
    
    # Remove unstable trajectories
    stable_trajectories = trajectories[stable_mask]
    removed_count = n_vertices - jnp.sum(stable_mask)
    
    print(f"\n=== RSE Step 4: Removing Unstable Trajectories ===")
    print(f"Algorithm parameter N: {N}")
    print(f"Distance between main bodies: {distance_between_bodies}")
    print(f"Divergence threshold: {divergence_threshold}")
    print(f"Total trajectories: {n_vertices}")
    print(f"Removed (unstable): {removed_count} ({removed_count/n_vertices*100:.1f}%)")
    print(f"Kept (stable): {jnp.sum(stable_mask)} ({jnp.sum(stable_mask)/n_vertices*100:.1f}%)")
    
    if removed_count > 0:
        removed_max_distances = max_distances[~stable_mask]
        print(f"Removed trajectory statistics:")
        print(f"  Max distances: {jnp.min(removed_max_distances):.3f} to {jnp.max(removed_max_distances):.3f}")
        print(f"  Mean max distance: {jnp.mean(removed_max_distances):.3f}")
    
    return stable_mask, stable_trajectories, removed_count

def analyze_trajectory_outcomes_rse(trajectories, times, dynamical_system):
    """
    Analyze trajectory end results for RSE refinement heuristics.
    
    This is a simpler classification focused on the paper's approach:
    - Impact: Close to primary bodies at final time
    - Escape: Far from system at final time  
    - In-system: Neither impacted nor escaped
    """
    n_vertices, n_times, state_dim = trajectories.shape
    
    # Get final states
    final_states = trajectories[:, -1, :]
    final_positions = final_states[:, :3]
    final_distances = jnp.linalg.norm(final_positions, axis=1)
    
    # Use fixed thresholds more appropriate for RSE algorithm
    # These are based on the CR3BP system characteristics
    impact_threshold = 0.1   # Close to primary body (normalized units)
    escape_threshold = 2.0   # Far from system center
    
    # Classify outcomes
    impacted = final_distances < impact_threshold
    escaped = final_distances > escape_threshold  
    in_system = ~(impacted | escaped)
    
    # Check for integration failures
    failed = jnp.isnan(final_states[:, 0])
    
    outcomes = {
        'impacted': jnp.sum(impacted),
        'escaped': jnp.sum(escaped),
        'in_system': jnp.sum(in_system), 
        'failed': jnp.sum(failed),
        'impact_mask': impacted,
        'escape_mask': escaped,
        'in_system_mask': in_system,
        'failed_mask': failed,
        'final_distances': final_distances,
        'thresholds': {'impact': impact_threshold, 'escape': escape_threshold}
    }
    
    print(f"\n=== Trajectory Outcome Analysis (Post Step 4) ===")
    print(f"Fixed thresholds - Impact: < {impact_threshold}, Escape: > {escape_threshold}")
    print(f"Outcomes:")
    print(f"  Impacted: {outcomes['impacted']} ({outcomes['impacted']/n_vertices*100:.1f}%)")
    print(f"  Escaped: {outcomes['escaped']} ({outcomes['escaped']/n_vertices*100:.1f}%)")
    print(f"  In-system: {outcomes['in_system']} ({outcomes['in_system']/n_vertices*100:.1f}%)")
    print(f"  Failed: {outcomes['failed']} ({outcomes['failed']/n_vertices*100:.1f}%)")
    
    return outcomes

def plot_trajectories_3d(trajectories, times, outcomes, mesh_info, sample_size=50):
    """
    Visualize the computed trajectories in 3D.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    
    n_vertices = trajectories.shape[0]
    
    # Sample trajectories for visualization (avoid overcrowding)
    if n_vertices > sample_size:
        indices = np.random.choice(n_vertices, sample_size, replace=False)
        traj_sample = trajectories[indices]
        
        # Only sample the mask arrays, keep scalar counts as-is
        sample_outcomes = {}
        for k, v in outcomes.items():
            if k.endswith('_mask'):  # These are boolean arrays
                sample_outcomes[k] = v[indices]
            else:  # These are scalar counts or other data
                sample_outcomes[k] = v
    else:
        traj_sample = trajectories
        sample_outcomes = outcomes
        indices = np.arange(n_vertices)
    
    # Convert to physical units
    LU_to_km = 389703  # km (Earth-Moon distance)
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Full 3D trajectories
    ax1 = fig.add_subplot(231, projection='3d')
    
    for i, traj in enumerate(traj_sample):
        if not sample_outcomes['failed_mask'][i]:
            positions = traj[:, :3] * LU_to_km
            
            # Color by outcome using mask arrays
            if sample_outcomes['escape_mask'][i]:
                color, alpha = 'red', 0.6
            elif sample_outcomes['impact_mask'][i]:
                color, alpha = 'orange', 0.6
            else:
                color, alpha = 'blue', 0.4
                
            ax1.plot(*positions.T, color=color, alpha=alpha, linewidth=1)
    
    # Mark initial position
    initial_pos = traj_sample[0, 0, :3] * LU_to_km
    ax1.scatter(*initial_pos, color='green', s=100, label='Initial position')
    
    ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)'); ax1.set_zlabel('Z (km)')
    ax1.set_title(f'3D Trajectories (n={len(traj_sample)})')
    
    # Add legend for outcomes
    ax1.plot([], [], 'r-', alpha=0.6, label='Escaped')
    ax1.plot([], [], 'orange', alpha=0.6, label='Impacted')
    ax1.plot([], [], 'b-', alpha=0.4, label='In system')
    ax1.legend()
    
    # Plot 2: XY projection
    ax2 = fig.add_subplot(232)
    
    for i, traj in enumerate(traj_sample):
        if not sample_outcomes['failed_mask'][i]:
            positions = traj[:, :3] * LU_to_km
            
            # Color by outcome using mask arrays
            if sample_outcomes['escape_mask'][i]:
                color, alpha = 'red', 0.6
            elif sample_outcomes['impact_mask'][i]:
                color, alpha = 'orange', 0.6
            else:
                color, alpha = 'blue', 0.4
                
            ax2.plot(positions[:, 0], positions[:, 1], color=color, alpha=alpha, linewidth=1)
    
    ax2.scatter(initial_pos[0], initial_pos[1], color='green', s=100, label='Initial position')
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Distance histogram
    ax3 = fig.add_subplot(233)
    final_distances = outcomes['final_distances'] * LU_to_km
    
    ax3.hist(final_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(outcomes['thresholds']['impact'] * LU_to_km, color='orange', 
                linestyle='--', label=f"Impact threshold")
    ax3.axvline(outcomes['thresholds']['escape'] * LU_to_km, color='red', 
                linestyle='--', label=f"Escape threshold")
    ax3.set_xlabel('Final Distance (km)')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Distance Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time evolution of spread
    ax4 = fig.add_subplot(234)
    
    # Calculate position spread over time
    position_std = []
    for t_idx in range(len(times)):
        positions_t = trajectories[:, t_idx, :3]
        # Remove failed trajectories
        valid_positions = positions_t[~outcomes['failed_mask']]
        if len(valid_positions) > 0:
            std_t = jnp.std(jnp.linalg.norm(valid_positions, axis=1))
            position_std.append(std_t)
        else:
            position_std.append(0.0)
    
    ax4.plot(times, np.array(position_std) * LU_to_km)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Position Spread (km)')
    ax4.set_title('Reachable Set Evolution')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Final positions scatter
    ax5 = fig.add_subplot(235)
    final_positions = trajectories[:, -1, :3] * LU_to_km
    
    # Color by outcome
    escaped_pos = final_positions[outcomes['escape_mask']]
    impacted_pos = final_positions[outcomes['impact_mask']]
    in_system_pos = final_positions[outcomes['in_system_mask']]
    
    if len(escaped_pos) > 0:
        ax5.scatter(escaped_pos[:, 0], escaped_pos[:, 1], c='red', alpha=0.6, s=20, label='Escaped')
    if len(impacted_pos) > 0:
        ax5.scatter(impacted_pos[:, 0], impacted_pos[:, 1], c='orange', alpha=0.6, s=20, label='Impacted')
    if len(in_system_pos) > 0:
        ax5.scatter(in_system_pos[:, 0], in_system_pos[:, 1], c='blue', alpha=0.4, s=20, label='In system')
    
    ax5.scatter(initial_pos[0], initial_pos[1], color='green', s=100, marker='*', label='Initial')
    ax5.set_xlabel('X (km)'); ax5.set_ylabel('Y (km)')
    ax5.set_title('Final Positions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_aspect('equal')
    
    # Plot 6: Outcome statistics
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    stats_text = f"""RSE Steps 3-4 Results

Original Vertices: {mesh_data['mesh_info']['n_vertices']}
Stable Trajectories: {n_vertices}
Sample Shown: {len(traj_sample)}

Step 4 Filtering:
  Parameter N: {outcomes.get('N_parameter', 'N/A')}
  Removed (unstable): {outcomes.get('removed_count', 'N/A')}

Outcomes (Stable Only):
  Impacted: {outcomes['impacted']} ({outcomes['impacted']/n_vertices*100:.1f}%)
  Escaped: {outcomes['escaped']} ({outcomes['escaped']/n_vertices*100:.1f}%)
  In System: {outcomes['in_system']} ({outcomes['in_system']/n_vertices*100:.1f}%)
  Failed: {outcomes['failed']} ({outcomes['failed']/n_vertices*100:.1f}%)

Distance Stats (km):
  Mean: {jnp.nanmean(final_distances):.0f}
  Std:  {jnp.nanstd(final_distances):.0f}
  Range: {jnp.nanmin(final_distances):.0f} - {jnp.nanmax(final_distances):.0f}

Fixed Thresholds (km):
  Impact: < {outcomes['thresholds']['impact'] * LU_to_km:.0f}
  Escape: > {outcomes['thresholds']['escape'] * LU_to_km:.0f}

Integration:
  Time span: {times[0]:.1f} → {times[-1]:.1f}
  Time points: {len(times)}

Next: Step 5 - Refinement heuristics
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step3_trajectories.png", dpi=150, bbox_inches='tight')
    plt.show()

# Execute Step 3 + 4
def execute_step3_and_4(mesh_data, dynamical_system, initial_state, delta_v_magnitude, t_final=10.0, N=10):
    """
    Execute Steps 3-4 of RSE algorithm:
    Step 3: Generate t-forward trajectories from each vertex
    Step 4: Remove numerically unstable trajectories
    """
    print("=== RSE Steps 3-4: Trajectories + Stability Filtering ===")
    
    # Get vertices from mesh
    deltav_vertices = mesh_data['vertices']
    
    # Step 3a: Map ΔV vertices to 6D state space
    states_6d = map_deltav_to_6d_states(deltav_vertices, initial_state, delta_v_magnitude)
    
    # Step 3b: Compute trajectories
    trajectories, times = compute_trajectories_batch(
        dynamical_system, states_6d, t_final, n_save_points=100
    )
    
    # Step 4: Remove unstable trajectories (RSE algorithm)
    stable_mask, stable_trajectories, removed_count = remove_unstable_trajectories_rse_step4(
        trajectories, times, N=N
    )
    
    # Step 3c: Analyze outcomes (only on stable trajectories)
    outcomes = analyze_trajectory_outcomes_rse(stable_trajectories, times, dynamical_system)
    
    # Step 3d: Visualize results
    plot_trajectories_3d(stable_trajectories, times, outcomes, mesh_data['mesh_info'])
    
    # Store results
    trajectory_data = {
        # Original data
        'deltav_vertices': deltav_vertices,
        'states_6d': states_6d,
        'all_trajectories': trajectories,  # Before filtering
        'times': times,
        't_final': t_final,
        # Step 4 filtering results
        'stable_mask': stable_mask,
        'stable_trajectories': stable_trajectories,  # After filtering
        'removed_count': removed_count,
        'N_parameter': N,
        # Outcomes (on stable trajectories only)
        'outcomes': outcomes,
    }
    
    print(f"\n✓ Steps 3-4 completed:")
    print(f"  Total trajectories computed: {trajectories.shape[0]}")
    print(f"  Stable trajectories kept: {stable_trajectories.shape[0]}")
    print(f"  Removed (unstable): {removed_count}")
    print(f"  Ready for Step 5: Refinement heuristics")
    
    return trajectory_data

# Execute RSE Steps 3-4 following the paper
trajectory_data = execute_step3_and_4(
    mesh_data=mesh_data,
    dynamical_system=dynamical_system,
    initial_state=initial_state,
    delta_v_magnitude=delta_v_magnitude,
    t_final=10.0,  # Integration time
    N=10           # RSE algorithm parameter
)

### Step 5

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay

def map_vertices_to_outcomes(mesh_data, trajectory_data):
    """
    Map each vertex in the original mesh to its trajectory outcome.
    
    Args:
        mesh_data: Results from Step 2 (mesh construction)
        trajectory_data: Results from Steps 3-4 (trajectories + filtering)
    
    Returns:
        vertex_outcomes: Array mapping each vertex to outcome (0=impact, 1=escape, 2=in-system, 3=removed)
        outcome_mapping: Dictionary for interpretation
    """
    n_original_vertices = mesh_data['mesh_info']['n_vertices']
    stable_mask = trajectory_data['stable_mask']
    outcomes = trajectory_data['outcomes']
    
    # Initialize all vertices as "removed" (3)
    vertex_outcomes = jnp.full(n_original_vertices, 3, dtype=int)
    
    # Map stable vertices to their outcomes
    stable_indices = jnp.where(stable_mask)[0]
    
    # Get outcome masks for stable trajectories
    impact_mask = outcomes['impact_mask']
    escape_mask = outcomes['escape_mask'] 
    in_system_mask = outcomes['in_system_mask']
    
    # Assign outcomes: 0=impact, 1=escape, 2=in-system
    vertex_outcomes = vertex_outcomes.at[stable_indices[impact_mask]].set(0)
    vertex_outcomes = vertex_outcomes.at[stable_indices[escape_mask]].set(1)  
    vertex_outcomes = vertex_outcomes.at[stable_indices[in_system_mask]].set(2)
    
    outcome_mapping = {
        0: 'impact',
        1: 'escape', 
        2: 'in-system',
        3: 'removed'
    }
    
    print(f"=== Vertex Outcome Mapping ===")
    for outcome_id, outcome_name in outcome_mapping.items():
        count = jnp.sum(vertex_outcomes == outcome_id)
        print(f"  {outcome_name}: {count} vertices ({count/n_original_vertices*100:.1f}%)")
    
    return vertex_outcomes, outcome_mapping

def calculate_edge_weights_end_result_heuristic(mesh_data, vertex_outcomes, weight_exponent=2.0):
    """
    Calculate edge weights using the end-result heuristic.
    
    From paper: "The weight of an edge is its length raised to an exponent 
    called the weight exponent. A weight exponent of 2, for example, means 
    that an edge that is double the length of another will be four times as 
    likely to be picked for subdivision."
    
    Args:
        mesh_data: Mesh data from Step 2
        vertex_outcomes: Outcome mapping for each vertex
        weight_exponent: Exponent for edge length weighting
        
    Returns:
        edge_info: Dictionary with edge weights and classifications
    """
    vertices = mesh_data['vertices']  # ΔV vertices (3D)
    unique_edges = mesh_data['unique_edges']
    
    edge_weights = []
    edge_lengths = []
    boundary_crossing = []
    
    print(f"\n=== Edge Weight Calculation ===")
    print(f"Weight exponent: {weight_exponent}")
    print(f"Total edges to analyze: {len(unique_edges)}")
    
    for edge in unique_edges:
        v1_idx, v2_idx = edge
        v1_pos = vertices[v1_idx]
        v2_pos = vertices[v2_idx]
        
        # Calculate edge length in ΔV space
        edge_length = jnp.linalg.norm(v2_pos - v1_pos)
        
        # Check if endpoints have different outcomes
        v1_outcome = vertex_outcomes[v1_idx] 
        v2_outcome = vertex_outcomes[v2_idx]
        is_boundary_crossing = (v1_outcome != v2_outcome)
        
        # Calculate weight: length^weight_exponent
        weight = edge_length ** weight_exponent
        
        edge_weights.append(weight)
        edge_lengths.append(edge_length)
        boundary_crossing.append(is_boundary_crossing)
    
    edge_weights = jnp.array(edge_weights)
    edge_lengths = jnp.array(edge_lengths)
    boundary_crossing = jnp.array(boundary_crossing)
    
    # Statistics
    n_boundary = jnp.sum(boundary_crossing)
    n_same_region = len(boundary_crossing) - n_boundary
    
    print(f"Edge classification:")
    print(f"  Boundary-crossing: {n_boundary} ({n_boundary/len(unique_edges)*100:.1f}%)")
    print(f"  Same-region: {n_same_region} ({n_same_region/len(unique_edges)*100:.1f}%)")
    print(f"Edge length stats:")
    print(f"  Mean: {jnp.mean(edge_lengths):.6f}")
    print(f"  Range: {jnp.min(edge_lengths):.6f} - {jnp.max(edge_lengths):.6f}")
    print(f"Weight stats:")
    print(f"  Mean: {jnp.mean(edge_weights):.6f}")
    print(f"  Range: {jnp.min(edge_weights):.6f} - {jnp.max(edge_weights):.6f}")
    
    edge_info = {
        'edges': unique_edges,
        'weights': edge_weights,
        'lengths': edge_lengths,
        'boundary_crossing': boundary_crossing,
        'weight_exponent': weight_exponent,
        'vertex_outcomes': vertex_outcomes
    }
    
    return edge_info

def sort_edges_by_end_result_heuristic(edge_info):
    """
    Sort edges into two lists based on end-result heuristic.
    
    From paper: "RSE then sorts the edges into two lists, one for those
    whose endpoints have different end results, and one for those whose 
    endpoints have similar end results."
    """
    boundary_mask = edge_info['boundary_crossing']
    weights = edge_info['weights']
    edges = edge_info['edges']
    
    # Split edges into two categories
    boundary_edges = [edges[i] for i in range(len(edges)) if boundary_mask[i]]
    same_region_edges = [edges[i] for i in range(len(edges)) if not boundary_mask[i]]
    
    boundary_weights = weights[boundary_mask]
    same_region_weights = weights[~boundary_mask]
    
    # Sort by weight (descending - higher weights first)
    boundary_sort_indices = jnp.argsort(-boundary_weights)
    same_region_sort_indices = jnp.argsort(-same_region_weights)
    
    sorted_boundary_edges = [boundary_edges[i] for i in boundary_sort_indices]
    sorted_same_region_edges = [same_region_edges[i] for i in same_region_sort_indices]
    
    sorted_boundary_weights = boundary_weights[boundary_sort_indices]
    sorted_same_region_weights = same_region_weights[same_region_sort_indices]
    
    print(f"\n=== Edge List Sorting ===")
    print(f"Boundary-crossing edges (sorted by weight):")
    print(f"  Count: {len(sorted_boundary_edges)}")
    if len(sorted_boundary_weights) > 0:
        print(f"  Top weight: {sorted_boundary_weights[0]:.6f}")
        print(f"  Bottom weight: {sorted_boundary_weights[-1]:.6f}")
    
    print(f"Same-region edges (sorted by weight):")
    print(f"  Count: {len(sorted_same_region_edges)}")
    if len(sorted_same_region_weights) > 0:
        print(f"  Top weight: {sorted_same_region_weights[0]:.6f}")
        print(f"  Bottom weight: {sorted_same_region_weights[-1]:.6f}")
    
    sorted_edge_lists = {
        'boundary_edges': sorted_boundary_edges,
        'boundary_weights': sorted_boundary_weights,
        'same_region_edges': sorted_same_region_edges,
        'same_region_weights': sorted_same_region_weights
    }
    
    return sorted_edge_lists

def select_edges_for_subdivision(sorted_edge_lists, n_edges_to_add=10, fraction=0.9, key=None):
    """
    Select edges for subdivision using the fraction parameter.
    
    From paper: "An additional parameter in the end-result heuristic, called 
    the fraction, allows the user to specify the relative proportion of choices 
    from each list. If the fraction value is 0.9, for example, the algorithm 
    will at each step choose an edge from the different-end-result list with 
    probability 0.9 and from the similar-end-result list with probability 0.1."
    
    Args:
        sorted_edge_lists: Output from sort_edges_by_end_result_heuristic
        n_edges_to_add: Number of edges to select for subdivision
        fraction: Probability of choosing from boundary-crossing list (0.0-1.0)
        key: JAX random key
    """
    if key is None:
        key = jax.random.key(42)
    
    boundary_edges = sorted_edge_lists['boundary_edges']
    same_region_edges = sorted_edge_lists['same_region_edges']
    boundary_weights = sorted_edge_lists['boundary_weights']
    same_region_weights = sorted_edge_lists['same_region_weights']
    
    selected_edges = []
    selection_sources = []  # Track which list each edge came from
    
    print(f"\n=== Edge Selection for Subdivision ===")
    print(f"Selecting {n_edges_to_add} edges")
    print(f"Fraction parameter: {fraction} (boundary) / {1-fraction:.1f} (same-region)")
    
    for i in range(n_edges_to_add):
        key, subkey = jax.random.split(key)
        
        # Decide which list to choose from
        if jax.random.uniform(subkey) < fraction:
            # Choose from boundary-crossing list
            if len(boundary_edges) > 0:
                # Weighted random selection (higher weights more likely)
                if len(boundary_weights) > 1:
                    # Normalize weights to probabilities
                    probs = boundary_weights / jnp.sum(boundary_weights)
                    key, subkey = jax.random.split(key)
                    edge_idx = jax.random.choice(subkey, len(boundary_edges), p=probs)
                else:
                    edge_idx = 0
                
                selected_edges.append(boundary_edges[edge_idx])
                selection_sources.append('boundary')
            else:
                # Fallback to same-region if no boundary edges available
                if len(same_region_edges) > 0:
                    probs = same_region_weights / jnp.sum(same_region_weights) if len(same_region_weights) > 1 else jnp.array([1.0])
                    key, subkey = jax.random.split(key)
                    edge_idx = jax.random.choice(subkey, len(same_region_edges), p=probs)
                    selected_edges.append(same_region_edges[edge_idx])
                    selection_sources.append('same_region_fallback')
        else:
            # Choose from same-region list
            if len(same_region_edges) > 0:
                if len(same_region_weights) > 1:
                    probs = same_region_weights / jnp.sum(same_region_weights)
                    key, subkey = jax.random.split(key)
                    edge_idx = jax.random.choice(subkey, len(same_region_edges), p=probs)
                else:
                    edge_idx = 0
                    
                selected_edges.append(same_region_edges[edge_idx])
                selection_sources.append('same_region')
            else:
                # Fallback to boundary if no same-region edges available
                if len(boundary_edges) > 0:
                    probs = boundary_weights / jnp.sum(boundary_weights) if len(boundary_weights) > 1 else jnp.array([1.0])
                    key, subkey = jax.random.split(key)
                    edge_idx = jax.random.choice(subkey, len(boundary_edges), p=probs)
                    selected_edges.append(boundary_edges[edge_idx])
                    selection_sources.append('boundary_fallback')
    
    # Statistics
    boundary_count = selection_sources.count('boundary') + selection_sources.count('boundary_fallback')
    same_region_count = selection_sources.count('same_region') + selection_sources.count('same_region_fallback')
    
    print(f"Selected edges:")
    print(f"  From boundary list: {boundary_count} ({boundary_count/n_edges_to_add*100:.1f}%)")
    print(f"  From same-region list: {same_region_count} ({same_region_count/n_edges_to_add*100:.1f}%)")
    
    selection_results = {
        'selected_edges': selected_edges,
        'selection_sources': selection_sources,
        'n_boundary_selected': boundary_count,
        'n_same_region_selected': same_region_count,
        'fraction_used': fraction
    }
    
    return selection_results

def visualize_refinement_heuristics(mesh_data, edge_info, sorted_edge_lists, selection_results):
    """
    Visualize the refinement heuristic results.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    
    vertices = mesh_data['vertices']
    vertex_outcomes = edge_info['vertex_outcomes']
    selected_edges = selection_results['selected_edges']
    
    # Convert to physical units for display
    LU_to_km = 389703; TU_to_s = 382981; VU_to_kmps = LU_to_km / TU_to_s
    vertices_km = np.array(vertices) * VU_to_kmps
    
    fig = plt.figure(figsize=(20, 12))
    
    # Color mapping for outcomes
    outcome_colors = {0: 'orange', 1: 'red', 2: 'blue', 3: 'gray'}
    outcome_names = {0: 'Impact', 1: 'Escape', 2: 'In-system', 3: 'Removed'}
    
    # Plot 1: 3D mesh with outcome coloring
    ax1 = fig.add_subplot(231, projection='3d')
    
    for outcome_id in [0, 1, 2, 3]:
        mask = vertex_outcomes == outcome_id
        if jnp.sum(mask) > 0:
            outcome_vertices = vertices_km[mask]
            ax1.scatter(*outcome_vertices.T, c=outcome_colors[outcome_id], 
                       label=outcome_names[outcome_id], alpha=0.7, s=30)
    
    ax1.set_xlabel('ΔVx (km/s)'); ax1.set_ylabel('ΔVy (km/s)'); ax1.set_zlabel('ΔVz (km/s)')
    ax1.set_title('Vertex Outcomes in ΔV Space')
    ax1.legend()
    
    # Plot 2: Edge weight distribution
    ax2 = fig.add_subplot(232)
    
    boundary_weights = sorted_edge_lists['boundary_weights']
    same_region_weights = sorted_edge_lists['same_region_weights']
    
    if len(boundary_weights) > 0:
        ax2.hist(boundary_weights, bins=20, alpha=0.6, label='Boundary-crossing', color='red')
    if len(same_region_weights) > 0:
        ax2.hist(same_region_weights, bins=20, alpha=0.6, label='Same-region', color='blue')
    
    ax2.set_xlabel('Edge Weight')
    ax2.set_ylabel('Count')
    ax2.set_title('Edge Weight Distribution')
    ax2.legend()
    ax2.set_yscale('log')
    
    # Plot 3: 2D projection with selected edges
    ax3 = fig.add_subplot(233)
    
    # Plot vertices colored by outcome
    for outcome_id in [0, 1, 2]:  # Skip removed for clarity
        mask = vertex_outcomes == outcome_id
        if jnp.sum(mask) > 0:
            outcome_vertices = vertices_km[mask]
            ax3.scatter(outcome_vertices[:, 0], outcome_vertices[:, 1], 
                       c=outcome_colors[outcome_id], label=outcome_names[outcome_id], 
                       alpha=0.7, s=20)
    
    # Highlight selected edges
    for edge in selected_edges:
        v1_pos = vertices_km[edge[0]]
        v2_pos = vertices_km[edge[1]]
        ax3.plot([v1_pos[0], v2_pos[0]], [v1_pos[1], v2_pos[1]], 
                'black', alpha=0.8, linewidth=2)
    
    ax3.set_xlabel('ΔVx (km/s)'); ax3.set_ylabel('ΔVy (km/s)')
    ax3.set_title('Selected Edges for Subdivision (XY)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    # Plot 4: Edge length vs weight
    ax4 = fig.add_subplot(234)
    
    all_lengths = edge_info['lengths']
    all_weights = edge_info['weights']
    boundary_mask = edge_info['boundary_crossing']
    
    ax4.scatter(all_lengths[~boundary_mask], all_weights[~boundary_mask], 
               alpha=0.6, c='blue', label='Same-region', s=10)
    ax4.scatter(all_lengths[boundary_mask], all_weights[boundary_mask], 
               alpha=0.6, c='red', label='Boundary-crossing', s=10)
    
    ax4.set_xlabel('Edge Length')
    ax4.set_ylabel('Edge Weight')
    ax4.set_title(f'Weight vs Length (exponent={edge_info["weight_exponent"]})')
    ax4.legend()
    ax4.set_xscale('log'); ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Selection statistics
    ax5 = fig.add_subplot(235)
    
    selection_counts = [selection_results['n_boundary_selected'], 
                       selection_results['n_same_region_selected']]
    selection_labels = ['Boundary-crossing', 'Same-region']
    
    ax5.pie(selection_counts, labels=selection_labels, autopct='%1.1f%%', 
           colors=['red', 'blue'])
    ax5.set_title('Edge Selection Distribution')
    
    # Plot 6: Algorithm parameters and stats
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    stats_text = f"""RSE Step 5: Refinement Heuristics

Algorithm Parameters:
  Weight exponent: {edge_info['weight_exponent']}
  Fraction: {selection_results['fraction_used']}
  Edges to add: {len(selected_edges)}

Edge Classification:
  Total edges: {len(edge_info['edges'])}
  Boundary-crossing: {jnp.sum(edge_info['boundary_crossing'])}
  Same-region: {len(edge_info['edges']) - jnp.sum(edge_info['boundary_crossing'])}

Selection Results:
  From boundary list: {selection_results['n_boundary_selected']}
  From same-region list: {selection_results['n_same_region_selected']}
  
Vertex Outcomes:
  Impact: {jnp.sum(vertex_outcomes == 0)}
  Escape: {jnp.sum(vertex_outcomes == 1)} 
  In-system: {jnp.sum(vertex_outcomes == 2)}
  Removed: {jnp.sum(vertex_outcomes == 3)}

Next: Step 6 - Add new vertices
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step5_refinement_heuristics.png", dpi=150, bbox_inches='tight')
    plt.show()

def execute_step5_refinement_heuristics(mesh_data, trajectory_data, weight_exponent=2.0, 
                                       fraction=0.9, n_edges_to_add=10, key=None):
    """
    Execute RSE Step 5: Apply refinement heuristics to identify areas for refinement.
    """
    print("=== RSE Step 5: Refinement Heuristics ===")
    
    # Map vertices to outcomes
    vertex_outcomes, outcome_mapping = map_vertices_to_outcomes(mesh_data, trajectory_data)
    
    # Calculate edge weights using end-result heuristic
    edge_info = calculate_edge_weights_end_result_heuristic(
        mesh_data, vertex_outcomes, weight_exponent
    )
    
    # Sort edges into boundary-crossing and same-region lists
    sorted_edge_lists = sort_edges_by_end_result_heuristic(edge_info)
    
    # Select edges for subdivision
    selection_results = select_edges_for_subdivision(
        sorted_edge_lists, n_edges_to_add, fraction, key
    )
    
    # Visualize results
    visualize_refinement_heuristics(mesh_data, edge_info, sorted_edge_lists, selection_results)
    
    # Package results
    refinement_data = {
        'vertex_outcomes': vertex_outcomes,
        'outcome_mapping': outcome_mapping,
        'edge_info': edge_info,
        'sorted_edge_lists': sorted_edge_lists,
        'selection_results': selection_results,
        'parameters': {
            'weight_exponent': weight_exponent,
            'fraction': fraction,
            'n_edges_to_add': n_edges_to_add
        }
    }
    
    print(f"\n✓ Step 5 completed: Refinement heuristics applied")
    print(f"✓ Selected {len(selection_results['selected_edges'])} edges for subdivision")
    print(f"✓ Ready for Step 6: Add new vertices to mesh")
    
    return refinement_data

# Usage Example:
refinement_data = execute_step5_refinement_heuristics(
    mesh_data=mesh_data,
    trajectory_data=trajectory_data,
    weight_exponent=2.0,    # Edge length exponent
    fraction=0.9,           # Probability of choosing boundary edges  
    n_edges_to_add=10,      # Number of edges to select
    key=jax.random.key(42)  # Random key for selection
)

### Step 6

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay

def generate_ellipsoidal_vertices(selected_edges, mesh_vertices, sigma=1.0, key=None):
    """
    Generate new vertices in ellipsoidal distributions around selected edges.
    
    From paper: "RSE instead adds new points in an ellipsoidal distribution 
    around the chosen edge. A user-specified edge factor σ controls the width 
    of this distribution... If σ = 1, for example, there is a 68.3% chance that 
    the new vertex will appear in an ellipsoid whose center is the midpoint of 
    the edge, whose semimajor axis is half the length of the edge, and whose 
    semiminor axes are a quarter the length of the edge."
    
    Args:
        selected_edges: List of edge tuples from Step 5
        mesh_vertices: Original mesh vertices (3D ΔV space)
        sigma: Edge factor controlling distribution width
        key: JAX random key
        
    Returns:
        new_vertices: Array of new vertices to add
        ellipsoid_info: Information about ellipsoids for visualization
    """
    if key is None:
        key = jax.random.key(42)
    
    new_vertices = []
    ellipsoid_info = []
    
    print(f"\n=== Generating New Vertices (Step 6) ===")
    print(f"Selected edges: {len(selected_edges)}")
    print(f"Sigma parameter: {sigma}")
    
    for i, edge in enumerate(selected_edges):
        key, subkey = jax.random.split(key)
        
        v1_idx, v2_idx = edge
        v1 = mesh_vertices[v1_idx]
        v2 = mesh_vertices[v2_idx]
        
        # Calculate edge properties
        edge_vector = v2 - v1
        edge_length = jnp.linalg.norm(edge_vector)
        edge_midpoint = (v1 + v2) / 2
        
        # Define ellipsoid parameters (from paper)
        # Center: midpoint of edge
        center = edge_midpoint
        
        # Semimajor axis: half the length of the edge (along edge direction)
        semimajor_axis_length = edge_length / 2
        edge_direction = edge_vector / edge_length
        
        # Semiminor axes: quarter the length of the edge (perpendicular to edge)
        semiminor_axis_length = edge_length / 4
        
        # Scale by sigma parameter
        semimajor_scaled = semimajor_axis_length * sigma
        semiminor_scaled = semiminor_axis_length * sigma
        
        # Create orthonormal basis with edge direction as first axis
        u1 = edge_direction  # Along edge
        
        # Generate two perpendicular directions
        # Use Gram-Schmidt to find orthogonal vectors
        if jnp.abs(u1[0]) < 0.9:
            temp = jnp.array([1.0, 0.0, 0.0])
        else:
            temp = jnp.array([0.0, 1.0, 0.0])
        
        u2 = temp - jnp.dot(temp, u1) * u1
        u2 = u2 / jnp.linalg.norm(u2)
        
        u3 = jnp.cross(u1, u2)
        u3 = u3 / jnp.linalg.norm(u3)
        
        # Create transformation matrix (local to global coordinates)
        # Columns are the basis vectors scaled by axis lengths
        transform_matrix = jnp.column_stack([
            u1 * semimajor_scaled,
            u2 * semiminor_scaled, 
            u3 * semiminor_scaled
        ])
        
        # Sample from unit sphere and transform to ellipsoid
        # This gives the 68.3% probability mentioned in the paper
        unit_sphere_point = jax.random.normal(subkey, shape=(3,))
        unit_sphere_point = unit_sphere_point / jnp.linalg.norm(unit_sphere_point)
        
        # Scale to make it more likely to be within the ellipsoid
        # Using a random radius with appropriate distribution
        key, subkey = jax.random.split(key)
        radius_factor = jax.random.uniform(subkey) ** (1/3)  # Uniform in 3D volume
        
        # Transform to ellipsoid coordinates
        ellipsoid_offset = transform_matrix @ (unit_sphere_point * radius_factor)
        new_vertex = center + ellipsoid_offset
        
        new_vertices.append(new_vertex)
        
        # Store ellipsoid info for visualization
        ellipsoid_info.append({
            'center': center,
            'edge': edge,
            'edge_length': edge_length,
            'transform_matrix': transform_matrix,
            'semimajor_scaled': semimajor_scaled,
            'semiminor_scaled': semiminor_scaled
        })
    
    new_vertices = jnp.array(new_vertices)
    
    print(f"Generated {len(new_vertices)} new vertices")
    if len(new_vertices) > 0:
        print(f"New vertex statistics:")
        print(f"  X range: {jnp.min(new_vertices[:, 0]):.6f} to {jnp.max(new_vertices[:, 0]):.6f}")
        print(f"  Y range: {jnp.min(new_vertices[:, 1]):.6f} to {jnp.max(new_vertices[:, 1]):.6f}")
        print(f"  Z range: {jnp.min(new_vertices[:, 2]):.6f} to {jnp.max(new_vertices[:, 2]):.6f}")
    
    return new_vertices, ellipsoid_info

def rebuild_delaunay_mesh(original_vertices, new_vertices):
    """
    Rebuild the Delaunay triangulation with new vertices added.
    
    From paper: "RSE uses Delaunay triangulation to create a new mesh of the 
    t-reachable set that includes both the old and new vertices added in the 
    previous step."
    """
    print(f"\n=== Rebuilding Delaunay Mesh ===")
    print(f"Original vertices: {len(original_vertices)}")
    print(f"New vertices: {len(new_vertices)}")
    
    # Combine original and new vertices
    if len(new_vertices) > 0:
        all_vertices = jnp.concatenate([original_vertices, new_vertices], axis=0)
    else:
        all_vertices = original_vertices
    
    print(f"Total vertices: {len(all_vertices)}")
    
    # Rebuild Delaunay triangulation
    vertices_np = np.array(all_vertices)
    new_triangulation = Delaunay(vertices_np)
    
    # Calculate new mesh statistics
    n_vertices = len(all_vertices)
    n_tetrahedra = len(new_triangulation.simplices)
    n_original = len(original_vertices)
    n_added = len(new_vertices)
    
    # Get unique edges in new mesh
    new_edges = set()
    for simplex in new_triangulation.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                new_edges.add(edge)
    new_edges = list(new_edges)
    
    refined_mesh_data = {
        'vertices': all_vertices,
        'triangulation': new_triangulation,
        'unique_edges': new_edges,
        'n_vertices': n_vertices,
        'n_tetrahedra': n_tetrahedra,
        'n_original': n_original,
        'n_added': n_added,
        'original_vertices': original_vertices,
        'new_vertices': new_vertices
    }
    
    print(f"Mesh rebuild statistics:")
    print(f"  Vertices: {n_original} → {n_vertices} (+{n_added})")
    print(f"  Tetrahedra: {n_tetrahedra}")
    print(f"  Edges: {len(new_edges)}")
    
    return refined_mesh_data

def analyze_mesh_refinement_quality(original_mesh_data, refined_mesh_data, ellipsoid_info):
    """
    Analyze the quality of the mesh refinement.
    """
    print(f"\n=== Mesh Refinement Quality Analysis ===")
    
    # Volume analysis
    original_vertices = original_mesh_data['vertices']
    refined_vertices = refined_mesh_data['vertices']
    
    # Calculate vertex density changes
    original_volume = 4/3 * jnp.pi  # Unit sphere volume
    density_increase = len(refined_vertices) / len(original_vertices)
    
    print(f"Density analysis:")
    print(f"  Density increase: {density_increase:.2f}x")
    print(f"  Vertex density: {len(refined_vertices)/original_volume:.2f} vertices/unit³")
    
    # Edge length analysis
    original_edges = original_mesh_data['unique_edges']
    refined_edges = refined_mesh_data['unique_edges']
    
    def calculate_edge_lengths(vertices, edges):
        lengths = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            length = jnp.linalg.norm(v2 - v1)
            lengths.append(length)
        return jnp.array(lengths)
    
    original_edge_lengths = calculate_edge_lengths(original_vertices, original_edges)
    refined_edge_lengths = calculate_edge_lengths(refined_vertices, refined_edges)
    
    print(f"Edge length analysis:")
    print(f"  Original mean length: {jnp.mean(original_edge_lengths):.6f}")
    print(f"  Refined mean length: {jnp.mean(refined_edge_lengths):.6f}")
    print(f"  Original min length: {jnp.min(original_edge_lengths):.6f}")
    print(f"  Refined min length: {jnp.min(refined_edge_lengths):.6f}")
    
    # Ellipsoid coverage analysis
    if len(ellipsoid_info) > 0:
        ellipsoid_volumes = []
        for info in ellipsoid_info:
            # Volume of ellipsoid = (4/3)π * a * b * c
            a = info['semimajor_scaled']
            b = c = info['semiminor_scaled']
            volume = (4/3) * jnp.pi * a * b * c
            ellipsoid_volumes.append(volume)
        
        total_ellipsoid_volume = jnp.sum(jnp.array(ellipsoid_volumes))
        print(f"Ellipsoid analysis:")
        print(f"  Mean ellipsoid volume: {jnp.mean(jnp.array(ellipsoid_volumes)):.6f}")
        print(f"  Total ellipsoid volume: {total_ellipsoid_volume:.6f}")
        print(f"  Coverage ratio: {total_ellipsoid_volume/original_volume:.4f}")
    
    quality_metrics = {
        'density_increase': density_increase,
        'edge_length_reduction': jnp.mean(original_edge_lengths) / jnp.mean(refined_edge_lengths),
        'min_edge_improvement': jnp.min(original_edge_lengths) / jnp.min(refined_edge_lengths),
        'total_ellipsoid_volume': total_ellipsoid_volume if len(ellipsoid_info) > 0 else 0
    }
    
    return quality_metrics

def visualize_mesh_refinement(original_mesh_data, refined_mesh_data, ellipsoid_info, 
                             selected_edges, sigma):
    """
    Visualize the mesh refinement results.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    
    original_vertices = original_mesh_data['vertices']
    refined_vertices = refined_mesh_data['vertices']
    new_vertices = refined_mesh_data['new_vertices']
    
    # Convert to physical units
    LU_to_km = 389703; TU_to_s = 382981; VU_to_kmps = LU_to_km / TU_to_s
    
    original_km = np.array(original_vertices) * VU_to_kmps
    refined_km = np.array(refined_vertices) * VU_to_kmps
    if len(new_vertices) > 0:
        new_km = np.array(new_vertices) * VU_to_kmps
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: 3D before/after comparison
    ax1 = fig.add_subplot(241, projection='3d')
    ax1.scatter(*original_km.T, alpha=0.6, s=20, c='blue', label='Original')
    ax1.set_xlabel('ΔVx (km/s)'); ax1.set_ylabel('ΔVy (km/s)'); ax1.set_zlabel('ΔVz (km/s)')
    ax1.set_title('Original Mesh')
    ax1.legend()
    
    ax2 = fig.add_subplot(242, projection='3d')
    ax2.scatter(*original_km.T, alpha=0.4, s=15, c='blue', label='Original')
    if len(new_vertices) > 0:
        ax2.scatter(*new_km.T, alpha=0.8, s=30, c='red', label='New vertices')
    ax2.set_xlabel('ΔVx (km/s)'); ax2.set_ylabel('ΔVy (km/s)'); ax2.set_zlabel('ΔVz (km/s)')
    ax2.set_title('Refined Mesh')
    ax2.legend()
    
    # Plot 3: 2D XY projection with ellipsoids
    ax3 = fig.add_subplot(243)
    ax3.scatter(original_km[:, 0], original_km[:, 1], alpha=0.5, s=15, c='blue', label='Original')
    if len(new_vertices) > 0:
        ax3.scatter(new_km[:, 0], new_km[:, 1], alpha=0.8, s=25, c='red', label='New')
    
    # Draw ellipsoids (projected to XY)
    for i, info in enumerate(ellipsoid_info):
        center = info['center'] * VU_to_kmps
        transform = info['transform_matrix'] * VU_to_kmps
        
        # Draw ellipse in XY projection
        theta = np.linspace(0, 2*np.pi, 50)
        unit_circle = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        ellipse_3d = center[:, None] + transform @ unit_circle
        
        ax3.plot(ellipse_3d[0], ellipse_3d[1], 'green', alpha=0.7, linewidth=2)
        
        # Mark selected edge
        edge = info['edge']
        v1 = original_vertices[edge[0]] * VU_to_kmps
        v2 = original_vertices[edge[1]] * VU_to_kmps
        ax3.plot([v1[0], v2[0]], [v1[1], v2[1]], 'black', linewidth=3, alpha=0.8)
    
    ax3.set_xlabel('ΔVx (km/s)'); ax3.set_ylabel('ΔVy (km/s)')
    ax3.set_title(f'Ellipsoidal Distributions (σ={sigma})')
    ax3.legend(); ax3.grid(True, alpha=0.3); ax3.set_aspect('equal')
    
    # Plot 4: Edge length histogram
    ax4 = fig.add_subplot(244)
    
    def get_edge_lengths(vertices, edges):
        lengths = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            length = np.linalg.norm(v2 - v1)
            lengths.append(length)
        return np.array(lengths)
    
    original_lengths = get_edge_lengths(original_vertices, original_mesh_data['unique_edges'])
    refined_lengths = get_edge_lengths(refined_vertices, refined_mesh_data['unique_edges'])
    
    ax4.hist(original_lengths, bins=30, alpha=0.6, label='Original', color='blue')
    ax4.hist(refined_lengths, bins=30, alpha=0.6, label='Refined', color='red')
    ax4.set_xlabel('Edge Length'); ax4.set_ylabel('Count')
    ax4.set_title('Edge Length Distribution')
    ax4.legend(); ax4.set_yscale('log')
    
    # Plot 5: Vertex density map (2D)
    ax5 = fig.add_subplot(245)
    
    # Create 2D histogram of vertex density
    ax5.hist2d(refined_km[:, 0], refined_km[:, 1], bins=20, alpha=0.7, cmap='Blues')
    ax5.scatter(new_km[:, 0], new_km[:, 1], c='red', s=25, alpha=0.8, label='New vertices')
    ax5.set_xlabel('ΔVx (km/s)'); ax5.set_ylabel('ΔVy (km/s)')
    ax5.set_title('Vertex Density (Refined)')
    ax5.legend(); ax5.set_aspect('equal')
    
    # Plot 6: Ellipsoid size distribution
    ax6 = fig.add_subplot(246)
    
    if len(ellipsoid_info) > 0:
        semimajor_lengths = [info['semimajor_scaled'] for info in ellipsoid_info]
        semiminor_lengths = [info['semiminor_scaled'] for info in ellipsoid_info]
        edge_lengths = [info['edge_length'] for info in ellipsoid_info]
        
        ax6.scatter(edge_lengths, semimajor_lengths, alpha=0.7, label='Semimajor axis')
        ax6.scatter(edge_lengths, semiminor_lengths, alpha=0.7, label='Semiminor axis')
        ax6.set_xlabel('Original Edge Length')
        ax6.set_ylabel('Ellipsoid Axis Length')
        ax6.set_title('Ellipsoid Scaling')
        ax6.legend(); ax6.grid(True, alpha=0.3)
    
    # Plot 7: 3D mesh connectivity (sample)
    ax7 = fig.add_subplot(247, projection='3d')
    
    # Show a sample of tetrahedra edges
    sample_simplices = refined_mesh_data['triangulation'].simplices[:min(50, len(refined_mesh_data['triangulation'].simplices))]
    for simplex in sample_simplices:
        for i in range(4):
            for j in range(i+1, 4):
                v1, v2 = refined_km[simplex[i]], refined_km[simplex[j]]
                ax7.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]], 
                        'gray', alpha=0.2, linewidth=0.5)
    
    if len(new_vertices) > 0:
        ax7.scatter(*new_km.T, c='red', s=30, alpha=0.8, label='New vertices')
    ax7.set_xlabel('ΔVx (km/s)'); ax7.set_ylabel('ΔVy (km/s)'); ax7.set_zlabel('ΔVz (km/s)')
    ax7.set_title('Refined Mesh Connectivity')
    ax7.legend()
    
    # Plot 8: Statistics summary
    ax8 = fig.add_subplot(248)
    ax8.axis('off')
    
    stats_text = f"""RSE Step 6: Mesh Refinement

Algorithm Parameters:
  Sigma (σ): {sigma}
  Selected edges: {len(selected_edges)}

Mesh Statistics:
  Original vertices: {len(original_vertices)}
  New vertices: {len(new_vertices)}
  Total vertices: {len(refined_vertices)}
  Density increase: {len(refined_vertices)/len(original_vertices):.2f}x

Original mesh:
  Tetrahedra: {len(original_mesh_data['triangulation'].simplices)}
  Edges: {len(original_mesh_data['unique_edges'])}

Refined mesh:
  Tetrahedra: {refined_mesh_data['n_tetrahedra']}
  Edges: {len(refined_mesh_data['unique_edges'])}

Ellipsoid Properties:
  Mean edge length: {np.mean([info['edge_length'] for info in ellipsoid_info]):.6f}
  Mean semimajor: {np.mean([info['semimajor_scaled'] for info in ellipsoid_info]):.6f}
  Mean semiminor: {np.mean([info['semiminor_scaled'] for info in ellipsoid_info]):.6f}

Next: Step 7 - Compute trajectories
      from new vertices
"""
    
    ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step6_mesh_refinement.png", dpi=150, bbox_inches='tight')
    plt.show()

def execute_step6_mesh_refinement(mesh_data, refinement_data, sigma=1.0, key=None):
    """
    Execute RSE Step 6: Refine the mesh by adding new vertices.
    """
    print("=== RSE Step 6: Mesh Refinement ===")
    
    # Get selected edges from Step 5
    selected_edges = refinement_data['selection_results']['selected_edges']
    original_vertices = mesh_data['vertices']
    
    # Generate new vertices using ellipsoidal distribution
    new_vertices, ellipsoid_info = generate_ellipsoidal_vertices(
        selected_edges, original_vertices, sigma, key
    )
    
    # Rebuild Delaunay mesh with new vertices
    refined_mesh_data = rebuild_delaunay_mesh(original_vertices, new_vertices)
    
    # Analyze refinement quality
    quality_metrics = analyze_mesh_refinement_quality(mesh_data, refined_mesh_data, ellipsoid_info)
    
    # Visualize results
    visualize_mesh_refinement(mesh_data, refined_mesh_data, ellipsoid_info, selected_edges, sigma)
    
    # Package results for next step
    step6_data = {
        'original_mesh_data': mesh_data,
        'refined_mesh_data': refined_mesh_data,
        'new_vertices': new_vertices,
        'ellipsoid_info': ellipsoid_info,
        'selected_edges': selected_edges,
        'quality_metrics': quality_metrics,
        'sigma': sigma
    }
    
    print(f"\n✓ Step 6 completed: Mesh refined successfully")
    print(f"✓ Added {len(new_vertices)} new vertices")
    print(f"✓ Mesh density increased by {quality_metrics['density_increase']:.2f}x")
    print(f"✓ Ready for Step 7: Compute trajectories from new vertices")
    
    return step6_data

# Usage Example:
step6_data = execute_step6_mesh_refinement(
    mesh_data=mesh_data,
    refinement_data=refinement_data,
    sigma=1.0,                    # Ellipsoid width parameter
    key=jax.random.key(42)        # Random key for vertex generation
)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffrax import SaveAt

def compute_new_vertex_trajectories(step6_data, trajectory_data, dynamical_system, 
                                   initial_state, delta_v_magnitude):
    """
    Step 7: Compute t-forward trajectories for new vertices from Step 6.
    
    From paper: "For each of the new points generated in step 6, RSE generates 
    a t-length trajectory using Mathematica's NDSolve, again removing any that 
    diverge from each other."
    
    Args:
        step6_data: Results from Step 6 (mesh refinement)
        trajectory_data: Results from Steps 3-4 (for time parameters)
        dynamical_system: CR3BP or other dynamical system
        initial_state: Reference initial state
        delta_v_magnitude: ΔV scaling factor
        
    Returns:
        new_trajectory_data: Trajectory results for new vertices
    """
    print("=== RSE Step 7: Update Forward Trajectories ===")
    
    new_vertices = step6_data['new_vertices']
    t_final = trajectory_data['t_final']
    times = trajectory_data['times']
    N_parameter = trajectory_data['N_parameter']
    
    if len(new_vertices) == 0:
        print("No new vertices to process.")
        return {
            'new_vertices_6d': jnp.array([]).reshape(0, 6),
            'new_trajectories': jnp.array([]).reshape(0, len(times), 6),
            'new_stable_mask': jnp.array([], dtype=bool),
            'new_outcomes': {},
            'removed_count': 0
        }
    
    print(f"Processing {len(new_vertices)} new vertices")
    print(f"Integration parameters:")
    print(f"  Time span: 0.0 → {t_final}")
    print(f"  Time points: {len(times)}")
    print(f"  Stability parameter N: {N_parameter}")
    
    # Step 7a: Map new ΔV vertices to 6D state space (same as Step 3)
    def map_deltav_to_6d_states_new(deltav_vertices, initial_state, delta_v_magnitude):
        """Map new ΔV vertices to 6D state space."""
        n_vertices = deltav_vertices.shape[0]
        deltav_scaled = deltav_vertices * delta_v_magnitude
        
        r0 = initial_state[:3]  # Initial position
        v0 = initial_state[3:6]  # Initial velocity
        
        positions = jnp.tile(r0, (n_vertices, 1))
        velocities = v0 + deltav_scaled
        
        states_6d = jnp.concatenate([positions, velocities], axis=1)
        
        print(f"  Mapped {n_vertices} new ΔV vertices to 6D state space")
        print(f"  New velocity range: {jnp.min(velocities):.6f} to {jnp.max(velocities):.6f}")
        
        return states_6d
    
    new_states_6d = map_deltav_to_6d_states_new(new_vertices, initial_state, delta_v_magnitude)
    
    # Step 7b: Compute trajectories for new vertices
    def compute_new_trajectories_batch(dynamical_system, initial_states, times):
        """Compute trajectories for new vertices only."""
        print(f"  Computing trajectories for {len(initial_states)} new vertices...")
        
        # Create SaveAt object matching original trajectories
        saveat = SaveAt(ts=times)
        
        def compute_single_trajectory(initial_state):
            """Compute trajectory for a single new vertex."""
            try:
                ts, trajectory = dynamical_system.trajectory(
                    initial_time=0.0,
                    final_time=times[-1],
                    state=initial_state,
                    saveat=saveat
                )
                return trajectory
            except Exception as e:
                print(f"    Warning: New trajectory computation failed: {e}")
                return jnp.full((len(times), 6), jnp.nan)
        
        # Compute all new trajectories in parallel
        new_trajectories = eqx.filter_vmap(compute_single_trajectory)(initial_states)
        
        # Check for failed integrations
        n_failed = jnp.sum(jnp.isnan(new_trajectories[:, 0, 0]))
        n_success = len(initial_states) - n_failed
        print(f"  New trajectory results: {n_success}/{len(initial_states)} successful")
        if n_failed > 0:
            print(f"  Failed integrations: {n_failed} (marked with NaN)")
        
        return new_trajectories
    
    new_trajectories = compute_new_trajectories_batch(dynamical_system, new_states_6d, times)
    
    # Step 7c: Apply stability filtering (same as Step 4)
    def filter_unstable_trajectories_new(new_trajectories, N_parameter):
        """Apply Step 4 stability filtering to new trajectories."""
        print(f"  Applying stability filtering (N={N_parameter})...")
        
        n_new = new_trajectories.shape[0]
        
        # Distance between main bodies = 1.0 in CR3BP
        divergence_threshold = N_parameter * 1.0
        
        # Check maximum distance during trajectory
        trajectory_positions = new_trajectories[:, :, :3]
        distances_from_origin = jnp.linalg.norm(trajectory_positions, axis=2)
        max_distances = jnp.max(distances_from_origin, axis=1)
        
        # Keep trajectories that don't exceed threshold
        stable_mask = max_distances <= divergence_threshold
        
        # Filter trajectories
        stable_trajectories = new_trajectories[stable_mask]
        removed_count = n_new - jnp.sum(stable_mask)
        
        print(f"  Stability filtering results:")
        print(f"    Divergence threshold: {divergence_threshold}")
        print(f"    New stable trajectories: {jnp.sum(stable_mask)}/{n_new}")
        print(f"    Removed (unstable): {removed_count}")
        
        if removed_count > 0:
            removed_max_distances = max_distances[~stable_mask]
            print(f"    Removed distances: {jnp.min(removed_max_distances):.3f} to {jnp.max(removed_max_distances):.3f}")
        
        return stable_mask, stable_trajectories, removed_count
    
    new_stable_mask, new_stable_trajectories, new_removed_count = filter_unstable_trajectories_new(
        new_trajectories, N_parameter
    )
    
    # Step 7d: Classify outcomes for new stable trajectories
    def classify_new_trajectory_outcomes(stable_trajectories):
        """Classify outcomes for new stable trajectories."""
        if len(stable_trajectories) == 0:
            return {
                'impacted': 0, 'escaped': 0, 'in_system': 0, 'failed': 0,
                'impact_mask': jnp.array([], dtype=bool),
                'escape_mask': jnp.array([], dtype=bool), 
                'in_system_mask': jnp.array([], dtype=bool),
                'failed_mask': jnp.array([], dtype=bool),
                'final_distances': jnp.array([])
            }
        
        print(f"  Classifying outcomes for {len(stable_trajectories)} new stable trajectories...")
        
        # Get final states
        final_states = stable_trajectories[:, -1, :]
        final_positions = final_states[:, :3]
        final_distances = jnp.linalg.norm(final_positions, axis=1)
        
        # Use same thresholds as original analysis (fixed for consistency)
        impact_threshold = 0.1
        escape_threshold = 2.0
        
        # Classify outcomes
        impacted = final_distances < impact_threshold
        escaped = final_distances > escape_threshold
        in_system = ~(impacted | escaped)
        failed = jnp.isnan(final_states[:, 0])
        
        new_outcomes = {
            'impacted': jnp.sum(impacted),
            'escaped': jnp.sum(escaped),
            'in_system': jnp.sum(in_system),
            'failed': jnp.sum(failed),
            'impact_mask': impacted,
            'escape_mask': escaped,
            'in_system_mask': in_system,
            'failed_mask': failed,
            'final_distances': final_distances,
            'thresholds': {'impact': impact_threshold, 'escape': escape_threshold}
        }
        
        n_total = len(stable_trajectories)
        print(f"  New trajectory outcomes:")
        print(f"    Impacted: {new_outcomes['impacted']} ({new_outcomes['impacted']/n_total*100:.1f}%)")
        print(f"    Escaped: {new_outcomes['escaped']} ({new_outcomes['escaped']/n_total*100:.1f}%)")
        print(f"    In-system: {new_outcomes['in_system']} ({new_outcomes['in_system']/n_total*100:.1f}%)")
        print(f"    Failed: {new_outcomes['failed']} ({new_outcomes['failed']/n_total*100:.1f}%)")
        
        return new_outcomes
    
    new_outcomes = classify_new_trajectory_outcomes(new_stable_trajectories)
    
    # Package results
    new_trajectory_data = {
        'new_vertices': new_vertices,
        'new_vertices_6d': new_states_6d,
        'new_trajectories_all': new_trajectories,  # Before filtering
        'new_stable_mask': new_stable_mask,
        'new_stable_trajectories': new_stable_trajectories,  # After filtering
        'new_outcomes': new_outcomes,
        'removed_count': new_removed_count,
        'times': times,
        't_final': t_final,
        'N_parameter': N_parameter
    }
    
    print(f"\n✓ Step 7 completed: New vertex trajectories computed")
    print(f"✓ Processed {len(new_vertices)} new vertices")
    print(f"✓ Generated {jnp.sum(new_stable_mask)} stable new trajectories")
    print(f"✓ Ready for Step 8: Rebuild complete mesh")
    
    return new_trajectory_data

def combine_original_and_new_trajectories(trajectory_data, new_trajectory_data):
    """
    Combine original and new trajectory data for complete mesh analysis.
    """
    print("\n=== Combining Original and New Trajectory Data ===")
    
    # Original data (from Steps 3-4)
    original_stable_trajectories = trajectory_data['stable_trajectories']
    original_outcomes = trajectory_data['outcomes']
    original_stable_mask = trajectory_data['stable_mask']
    
    # New data (from Step 7)
    new_stable_trajectories = new_trajectory_data['new_stable_trajectories']
    new_outcomes = new_trajectory_data['new_outcomes']
    new_stable_mask = new_trajectory_data['new_stable_mask']
    
    print(f"Original stable trajectories: {len(original_stable_trajectories)}")
    print(f"New stable trajectories: {len(new_stable_trajectories)}")
    
    # Combine trajectory arrays
    if len(new_stable_trajectories) > 0:
        combined_trajectories = jnp.concatenate([original_stable_trajectories, new_stable_trajectories], axis=0)
    else:
        combined_trajectories = original_stable_trajectories
    
    # Combine outcome masks
    def combine_masks(orig_mask, new_mask):
        if len(new_mask) > 0:
            return jnp.concatenate([orig_mask, new_mask])
        else:
            return orig_mask
    
    combined_outcomes = {
        'impact_mask': combine_masks(original_outcomes['impact_mask'], new_outcomes['impact_mask']),
        'escape_mask': combine_masks(original_outcomes['escape_mask'], new_outcomes['escape_mask']),
        'in_system_mask': combine_masks(original_outcomes['in_system_mask'], new_outcomes['in_system_mask']),
        'failed_mask': combine_masks(original_outcomes['failed_mask'], new_outcomes['failed_mask']),
        'final_distances': combine_masks(original_outcomes['final_distances'], new_outcomes['final_distances'])
    }
    
    # Calculate combined counts
    combined_outcomes.update({
        'impacted': jnp.sum(combined_outcomes['impact_mask']),
        'escaped': jnp.sum(combined_outcomes['escape_mask']),
        'in_system': jnp.sum(combined_outcomes['in_system_mask']),
        'failed': jnp.sum(combined_outcomes['failed_mask']),
        'thresholds': original_outcomes['thresholds']
    })
    
    n_total = len(combined_trajectories)
    print(f"Combined trajectory statistics:")
    print(f"  Total stable trajectories: {n_total}")
    print(f"  Impacted: {combined_outcomes['impacted']} ({combined_outcomes['impacted']/n_total*100:.1f}%)")
    print(f"  Escaped: {combined_outcomes['escaped']} ({combined_outcomes['escaped']/n_total*100:.1f}%)")
    print(f"  In-system: {combined_outcomes['in_system']} ({combined_outcomes['in_system']/n_total*100:.1f}%)")
    
    combined_data = {
        'combined_trajectories': combined_trajectories,
        'combined_outcomes': combined_outcomes,
        'n_original': len(original_stable_trajectories),
        'n_new': len(new_stable_trajectories),
        'n_total': n_total
    }
    
    return combined_data

def visualize_step7_results(step6_data, new_trajectory_data, combined_data):
    """
    Visualize Step 7 results showing new trajectories and combined mesh.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    
    original_vertices = step6_data['original_mesh_data']['vertices']
    new_vertices = step6_data['new_vertices']
    new_trajectories = new_trajectory_data['new_stable_trajectories']
    new_outcomes = new_trajectory_data['new_outcomes']
    
    # Convert to physical units
    LU_to_km = 389703
    VU_to_kmps = LU_to_km / 382981
    
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: New vertices in ΔV space
    ax1 = fig.add_subplot(231, projection='3d')
    
    if len(original_vertices) > 0:
        orig_km = np.array(original_vertices) * VU_to_kmps
        ax1.scatter(*orig_km.T, alpha=0.4, s=15, c='blue', label='Original')
    
    if len(new_vertices) > 0:
        new_km = np.array(new_vertices) * VU_to_kmps
        ax1.scatter(*new_km.T, alpha=0.8, s=40, c='red', label='New vertices')
    
    ax1.set_xlabel('ΔVx (km/s)'); ax1.set_ylabel('ΔVy (km/s)'); ax1.set_zlabel('ΔVz (km/s)')
    ax1.set_title('Step 7: New Vertices Added')
    ax1.legend()
    
    # Plot 2: New trajectories (3D sample)
    ax2 = fig.add_subplot(232, projection='3d')
    
    if len(new_trajectories) > 0:
        sample_size = min(20, len(new_trajectories))
        sample_indices = np.random.choice(len(new_trajectories), sample_size, replace=False)
        
        for i in sample_indices:
            if not new_outcomes['failed_mask'][i]:
                traj = new_trajectories[i] * LU_to_km
                
                if new_outcomes['escape_mask'][i]:
                    color = 'red'
                elif new_outcomes['impact_mask'][i]:
                    color = 'orange'
                else:
                    color = 'blue'
                
                ax2.plot(*traj[:, :3].T, color=color, alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)'); ax2.set_zlabel('Z (km)')
    ax2.set_title(f'New Trajectories (n={len(new_trajectories)})')
    
    # Plot 3: Outcome comparison (original vs new)
    ax3 = fig.add_subplot(233)
    
    categories = ['Impact', 'Escape', 'In-system']
    if 'combined_outcomes' in combined_data:
        orig_counts = [
            combined_data['combined_outcomes']['impacted'] - new_outcomes['impacted'],
            combined_data['combined_outcomes']['escaped'] - new_outcomes['escaped'], 
            combined_data['combined_outcomes']['in_system'] - new_outcomes['in_system']
        ]
        new_counts = [new_outcomes['impacted'], new_outcomes['escaped'], new_outcomes['in_system']]
    else:
        orig_counts = [0, 0, 0]
        new_counts = [new_outcomes['impacted'], new_outcomes['escaped'], new_outcomes['in_system']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax3.bar(x - width/2, orig_counts, width, label='Original', alpha=0.7, color='blue')
    ax3.bar(x + width/2, new_counts, width, label='New', alpha=0.7, color='red')
    
    ax3.set_xlabel('Outcome Type')
    ax3.set_ylabel('Count')
    ax3.set_title('Trajectory Outcomes: Original vs New')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    
    # Plot 4: Distance distribution comparison
    ax4 = fig.add_subplot(234)
    
    if len(new_outcomes['final_distances']) > 0:
        new_distances_km = new_outcomes['final_distances'] * LU_to_km
        ax4.hist(new_distances_km, bins=20, alpha=0.7, label='New trajectories', color='red')
        
        # Mark thresholds
        impact_thresh_km = new_outcomes['thresholds']['impact'] * LU_to_km
        escape_thresh_km = new_outcomes['thresholds']['escape'] * LU_to_km
        ax4.axvline(impact_thresh_km, color='orange', linestyle='--', label='Impact threshold')
        ax4.axvline(escape_thresh_km, color='red', linestyle='--', label='Escape threshold')
    
    ax4.set_xlabel('Final Distance (km)')
    ax4.set_ylabel('Count')
    ax4.set_title('New Trajectory Final Distances')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Combined mesh growth
    ax5 = fig.add_subplot(235)
    
    mesh_stages = ['Initial', 'After Step 6', 'After Step 7']
    vertex_counts = [
        len(original_vertices),
        len(original_vertices) + len(new_vertices),
        len(original_vertices) + len(new_vertices)  # Same as Step 6
    ]
    trajectory_counts = [
        combined_data.get('n_original', 0),
        combined_data.get('n_original', 0),
        combined_data.get('n_total', 0)
    ]
    
    x = np.arange(len(mesh_stages))
    ax5.bar(x, vertex_counts, alpha=0.7, label='Vertices', color='blue')
    ax5.bar(x, trajectory_counts, alpha=0.7, label='Stable trajectories', color='green')
    
    ax5.set_xlabel('RSE Stage')
    ax5.set_ylabel('Count')
    ax5.set_title('Mesh Growth Progress')
    ax5.set_xticks(x)
    ax5.set_xticklabels(mesh_stages)
    ax5.legend()
    
    # Plot 6: Step 7 summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    stats_text = f"""RSE Step 7: Update Forward Trajectories

New Vertices Processed:
  Added in Step 6: {len(new_vertices)}
  Mapped to 6D: {len(new_vertices)}
  
Trajectory Computation:
  New trajectories: {len(new_trajectory_data.get('new_trajectories_all', []))}
  Integration time: 0.0 → {new_trajectory_data.get('t_final', 0)}
  
Stability Filtering (Step 4):
  Stable: {jnp.sum(new_trajectory_data.get('new_stable_mask', jnp.array([])))}
  Removed: {new_trajectory_data.get('removed_count', 0)}
  
New Trajectory Outcomes:
  Impact: {new_outcomes.get('impacted', 0)}
  Escape: {new_outcomes.get('escaped', 0)}
  In-system: {new_outcomes.get('in_system', 0)}
  Failed: {new_outcomes.get('failed', 0)}

Combined Mesh Status:
  Total stable trajectories: {combined_data.get('n_total', 0)}
  Original: {combined_data.get('n_original', 0)}
  New: {combined_data.get('n_new', 0)}

Ready for Step 8: Rebuild mesh with
all trajectory information
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step7_update_trajectories.png", dpi=150, bbox_inches='tight')
    plt.show()

def execute_step7_update_trajectories(step6_data, trajectory_data, dynamical_system, 
                                     initial_state, delta_v_magnitude):
    """
    Execute RSE Step 7: Update forward trajectories for new vertices.
    """
    print("=== RSE Step 7: Update Forward Trajectories ===")
    
    # Compute trajectories for new vertices
    new_trajectory_data = compute_new_vertex_trajectories(
        step6_data, trajectory_data, dynamical_system, initial_state, delta_v_magnitude
    )
    
    # Combine with original trajectory data
    combined_data = combine_original_and_new_trajectories(trajectory_data, new_trajectory_data)
    
    # Visualize results
    visualize_step7_results(step6_data, new_trajectory_data, combined_data)
    
    # Package complete Step 7 results
    step7_data = {
        'new_trajectory_data': new_trajectory_data,
        'combined_data': combined_data,
        'step6_data': step6_data,  # Pass through for Step 8
        'original_trajectory_data': trajectory_data  # Pass through for Step 8
    }
    
    print(f"\n✓ Step 7 completed: Forward trajectories updated")
    print(f"✓ New vertices: {len(step6_data['new_vertices'])}")
    print(f"✓ New stable trajectories: {len(new_trajectory_data['new_stable_trajectories'])}")
    print(f"✓ Total stable trajectories: {combined_data['n_total']}")
    print(f"✓ Ready for Step 8: Rebuild mesh with complete trajectory data")
    
    return step7_data

# Usage Example:
step7_data = execute_step7_update_trajectories(
    step6_data=step6_data,                    # Results from Step 6
    trajectory_data=trajectory_data,          # Results from Steps 3-4
    dynamical_system=dynamical_system,       # CR3BP system
    initial_state=initial_state,             # Reference state
    delta_v_magnitude=delta_v_magnitude       # ΔV scaling
)

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import Delaunay

def rebuild_complete_mesh_with_trajectories(step7_data):
    """
    Step 8: Rebuild the mesh with complete trajectory information.
    
    From paper: "RSE uses Delaunay triangulation to create a new mesh of the 
    t-reachable set that includes both the old and new vertices added in the 
    previous step."
    
    This step consolidates all vertex and trajectory data into a unified mesh
    structure ready for the next iteration of refinement.
    
    Args:
        step7_data: Complete results from Step 7
        
    Returns:
        complete_mesh_data: Unified mesh with all trajectory information
    """
    print("=== RSE Step 8: Rebuild Complete Mesh ===")
    
    # Extract data from Step 7
    step6_data = step7_data['step6_data']
    new_trajectory_data = step7_data['new_trajectory_data']
    combined_data = step7_data['combined_data']
    original_trajectory_data = step7_data['original_trajectory_data']
    
    # Get all vertices (original + new)
    original_vertices = step6_data['original_mesh_data']['vertices']
    new_vertices = step6_data['new_vertices']
    
    if len(new_vertices) > 0:
        all_vertices = jnp.concatenate([original_vertices, new_vertices], axis=0)
    else:
        all_vertices = original_vertices
    
    print(f"Consolidating mesh data:")
    print(f"  Original vertices: {len(original_vertices)}")
    print(f"  New vertices: {len(new_vertices)}")
    print(f"  Total vertices: {len(all_vertices)}")
    
    # Rebuild Delaunay triangulation with all vertices
    print(f"Rebuilding Delaunay triangulation...")
    vertices_np = np.array(all_vertices)
    complete_triangulation = Delaunay(vertices_np)
    
    # Extract complete mesh topology
    n_vertices = len(all_vertices)
    n_tetrahedra = len(complete_triangulation.simplices)
    
    # Get all unique edges
    all_edges = set()
    for simplex in complete_triangulation.simplices:
        for i in range(4):
            for j in range(i+1, 4):
                edge = tuple(sorted([simplex[i], simplex[j]]))
                all_edges.add(edge)
    all_edges = list(all_edges)
    
    print(f"Complete mesh topology:")
    print(f"  Vertices: {n_vertices}")
    print(f"  Tetrahedra: {n_tetrahedra}")
    print(f"  Edges: {len(all_edges)}")
    
    # Create complete vertex outcome mapping
    def create_complete_vertex_outcomes():
        """Map all vertices to their trajectory outcomes."""
        print(f"Creating complete vertex outcome mapping...")
        
        # Initialize all vertices as "no_trajectory" (4)
        vertex_outcomes = jnp.full(n_vertices, 4, dtype=int)
        
        # Map original vertices
        original_stable_mask = original_trajectory_data['stable_mask']
        original_outcomes = original_trajectory_data['outcomes']
        
        original_stable_indices = jnp.where(original_stable_mask)[0]
        
        # Set outcomes for original stable vertices
        vertex_outcomes = vertex_outcomes.at[original_stable_indices[original_outcomes['impact_mask']]].set(0)  # impact
        vertex_outcomes = vertex_outcomes.at[original_stable_indices[original_outcomes['escape_mask']]].set(1)  # escape
        vertex_outcomes = vertex_outcomes.at[original_stable_indices[original_outcomes['in_system_mask']]].set(2)  # in-system
        
        # Set removed vertices (original unstable)
        original_removed_indices = jnp.where(~original_stable_mask)[0]
        vertex_outcomes = vertex_outcomes.at[original_removed_indices].set(3)  # removed
        
        # Map new vertices
        if len(new_vertices) > 0:
            new_stable_mask = new_trajectory_data['new_stable_mask'] 
            new_outcomes = new_trajectory_data['new_outcomes']
            
            # Base index for new vertices
            new_vertex_base_idx = len(original_vertices)
            
            # Get indices of new stable vertices in the complete vertex array
            new_stable_indices = new_vertex_base_idx + jnp.where(new_stable_mask)[0]
            
            # Set outcomes for new stable vertices
            vertex_outcomes = vertex_outcomes.at[new_stable_indices[new_outcomes['impact_mask']]].set(0)  # impact
            vertex_outcomes = vertex_outcomes.at[new_stable_indices[new_outcomes['escape_mask']]].set(1)  # escape  
            vertex_outcomes = vertex_outcomes.at[new_stable_indices[new_outcomes['in_system_mask']]].set(2)  # in-system
            
            # Set removed new vertices
            new_removed_indices = new_vertex_base_idx + jnp.where(~new_stable_mask)[0]
            vertex_outcomes = vertex_outcomes.at[new_removed_indices].set(3)  # removed
        
        outcome_mapping = {
            0: 'impact',
            1: 'escape', 
            2: 'in-system',
            3: 'removed',
            4: 'no_trajectory'
        }
        
        # Print outcome statistics
        for outcome_id, outcome_name in outcome_mapping.items():
            count = jnp.sum(vertex_outcomes == outcome_id)
            print(f"    {outcome_name}: {count} vertices ({count/n_vertices*100:.1f}%)")
        
        return vertex_outcomes, outcome_mapping
    
    vertex_outcomes, outcome_mapping = create_complete_vertex_outcomes()
    
    # Package complete mesh data
    complete_mesh_data = {
        # Geometric data
        'vertices': all_vertices,
        'triangulation': complete_triangulation,
        'unique_edges': all_edges,
        
        # Mesh statistics
        'n_vertices': n_vertices,
        'n_tetrahedra': n_tetrahedra,
        'n_edges': len(all_edges),
        'n_original': len(original_vertices),
        'n_new': len(new_vertices),
        
        # Trajectory data
        'vertex_outcomes': vertex_outcomes,
        'outcome_mapping': outcome_mapping,
        'combined_trajectories': combined_data['combined_trajectories'],
        'combined_outcomes': combined_data['combined_outcomes'],
        
        # Historical data (for analysis)
        'original_mesh_data': step6_data['original_mesh_data'],
        'refinement_history': {
            'step6_data': step6_data,
            'step7_data': step7_data
        },
        
        # Ready for next iteration
        'ready_for_refinement': True
    }
    
    return complete_mesh_data

def analyze_mesh_convergence(complete_mesh_data, iteration_number=1):
    """
    Analyze mesh convergence and refinement quality.
    """
    print(f"\n=== Mesh Convergence Analysis (Iteration {iteration_number}) ===")
    
    vertex_outcomes = complete_mesh_data['vertex_outcomes']
    all_edges = complete_mesh_data['unique_edges']
    vertices = complete_mesh_data['vertices']
    
    # Boundary crossing analysis
    boundary_crossing_edges = []
    same_region_edges = []
    
    for edge in all_edges:
        v1_outcome = vertex_outcomes[edge[0]]
        v2_outcome = vertex_outcomes[edge[1]]
        
        # Only consider edges between vertices with trajectories (outcomes 0, 1, 2)
        if v1_outcome <= 2 and v2_outcome <= 2:
            if v1_outcome != v2_outcome:
                boundary_crossing_edges.append(edge)
            else:
                same_region_edges.append(edge)
    
    n_boundary = len(boundary_crossing_edges)
    n_same_region = len(same_region_edges)
    n_analyzed = n_boundary + n_same_region
    
    print(f"Boundary analysis:")
    print(f"  Edges analyzed: {n_analyzed}")
    print(f"  Boundary-crossing: {n_boundary} ({n_boundary/n_analyzed*100:.1f}%)")
    print(f"  Same-region: {n_same_region} ({n_same_region/n_analyzed*100:.1f}%)")
    
    # Edge length analysis
    def calculate_edge_lengths(edges):
        lengths = []
        for edge in edges:
            v1, v2 = vertices[edge[0]], vertices[edge[1]]
            length = jnp.linalg.norm(v2 - v1)
            lengths.append(length)
        return jnp.array(lengths) if lengths else jnp.array([])
    
    if n_boundary > 0:
        boundary_lengths = calculate_edge_lengths(boundary_crossing_edges)
        print(f"Boundary edge lengths:")
        print(f"  Mean: {jnp.mean(boundary_lengths):.6f}")
        print(f"  Min: {jnp.min(boundary_lengths):.6f}")
        print(f"  Max: {jnp.max(boundary_lengths):.6f}")
    
    if n_same_region > 0:
        same_region_lengths = calculate_edge_lengths(same_region_edges)
        print(f"Same-region edge lengths:")
        print(f"  Mean: {jnp.mean(same_region_lengths):.6f}")
        print(f"  Min: {jnp.min(same_region_lengths):.6f}")
        print(f"  Max: {jnp.max(same_region_lengths):.6f}")
    
    # Convergence metrics
    vertex_density = complete_mesh_data['n_vertices'] / (4/3 * jnp.pi)  # vertices per unit sphere volume
    boundary_resolution = jnp.mean(boundary_lengths) if n_boundary > 0 else float('inf')
    
    convergence_metrics = {
        'iteration': iteration_number,
        'vertex_density': vertex_density,
        'boundary_crossing_edges': n_boundary,
        'same_region_edges': n_same_region,
        'boundary_resolution': boundary_resolution,
        'total_vertices': complete_mesh_data['n_vertices'],
        'refinement_efficiency': n_boundary / complete_mesh_data['n_new'] if complete_mesh_data['n_new'] > 0 else 0
    }
    
    print(f"Convergence metrics:")
    print(f"  Vertex density: {vertex_density:.2f} vertices/unit³")
    print(f"  Boundary resolution: {boundary_resolution:.6f}")
    print(f"  Refinement efficiency: {convergence_metrics['refinement_efficiency']:.2f}")
    
    return convergence_metrics

def check_refinement_stopping_criteria(convergence_metrics, max_vertices=5000, 
                                      min_boundary_resolution=1e-4, max_iterations=10):
    """
    Check if refinement should stop based on convergence criteria.
    """
    print(f"\n=== Refinement Stopping Criteria ===")
    
    iteration = convergence_metrics['iteration']
    total_vertices = convergence_metrics['total_vertices']
    boundary_resolution = convergence_metrics['boundary_resolution']
    
    stop_reasons = []
    
    # Check stopping criteria
    if total_vertices >= max_vertices:
        stop_reasons.append(f"Max vertices reached ({total_vertices} >= {max_vertices})")
    
    if boundary_resolution <= min_boundary_resolution:
        stop_reasons.append(f"Boundary resolution achieved ({boundary_resolution:.2e} <= {min_boundary_resolution:.2e})")
    
    if iteration >= max_iterations:
        stop_reasons.append(f"Max iterations reached ({iteration} >= {max_iterations})")
    
    should_stop = len(stop_reasons) > 0
    
    print(f"Stopping criteria evaluation:")
    print(f"  Current vertices: {total_vertices} / {max_vertices}")
    print(f"  Current resolution: {boundary_resolution:.2e} / {min_boundary_resolution:.2e}")
    print(f"  Current iteration: {iteration} / {max_iterations}")
    
    if should_stop:
        print(f"✓ STOPPING: {', '.join(stop_reasons)}")
    else:
        print(f"→ CONTINUE: Criteria not met, proceed to next iteration")
    
    return should_stop, stop_reasons

def visualize_complete_mesh(complete_mesh_data, convergence_metrics):
    """
    Visualize the complete mesh after Step 8.
    """
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    
    vertices = complete_mesh_data['vertices']
    vertex_outcomes = complete_mesh_data['vertex_outcomes']
    outcome_mapping = complete_mesh_data['outcome_mapping']
    
    # Convert to physical units
    LU_to_km = 389703; TU_to_s = 382981; VU_to_kmps = LU_to_km / TU_to_s
    vertices_km = np.array(vertices) * VU_to_kmps
    
    # Color mapping
    outcome_colors = {0: 'orange', 1: 'red', 2: 'blue', 3: 'gray', 4: 'lightgray'}
    
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: Complete 3D mesh
    ax1 = fig.add_subplot(231, projection='3d')
    
    for outcome_id in [0, 1, 2, 3]:
        mask = vertex_outcomes == outcome_id
        if jnp.sum(mask) > 0:
            outcome_vertices = vertices_km[mask]
            ax1.scatter(*outcome_vertices.T, c=outcome_colors[outcome_id], 
                       label=outcome_mapping[outcome_id], alpha=0.7, s=20)
    
    ax1.set_xlabel('ΔVx (km/s)'); ax1.set_ylabel('ΔVy (km/s)'); ax1.set_zlabel('ΔVz (km/s)')
    ax1.set_title('Complete Mesh (All Iterations)')
    ax1.legend()
    
    # Plot 2: Boundary crossing edges
    ax2 = fig.add_subplot(232)
    
    # Plot vertices
    for outcome_id in [0, 1, 2]:  # Only trajectories with outcomes
        mask = vertex_outcomes == outcome_id
        if jnp.sum(mask) > 0:
            outcome_vertices = vertices_km[mask]
            ax2.scatter(outcome_vertices[:, 0], outcome_vertices[:, 1], 
                       c=outcome_colors[outcome_id], label=outcome_mapping[outcome_id], 
                       alpha=0.7, s=15)
    
    # Highlight boundary-crossing edges
    boundary_count = 0
    for edge in complete_mesh_data['unique_edges']:
        v1_outcome = vertex_outcomes[edge[0]]
        v2_outcome = vertex_outcomes[edge[1]]
        
        # Only show boundary crossings between trajectory vertices
        if v1_outcome <= 2 and v2_outcome <= 2 and v1_outcome != v2_outcome:
            v1_pos = vertices_km[edge[0]]
            v2_pos = vertices_km[edge[1]]
            ax2.plot([v1_pos[0], v2_pos[0]], [v1_pos[1], v2_pos[1]], 
                    'black', alpha=0.5, linewidth=1)
            boundary_count += 1
            
            if boundary_count > 100:  # Limit for visualization
                break
    
    ax2.set_xlabel('ΔVx (km/s)'); ax2.set_ylabel('ΔVy (km/s)')
    ax2.set_title('Boundary-Crossing Edges (XY)')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_aspect('equal')
    
    # Plot 3: Mesh density evolution
    ax3 = fig.add_subplot(233)
    
    iteration_data = [
        ('Initial', complete_mesh_data['n_original'], 0),
        ('After Refinement', complete_mesh_data['n_vertices'], complete_mesh_data['n_new'])
    ]
    
    iterations = [data[0] for data in iteration_data]
    original_counts = [data[1] for data in iteration_data]
    new_counts = [data[2] for data in iteration_data]
    
    x = np.arange(len(iterations))
    ax3.bar(x, original_counts, label='Original', alpha=0.7, color='blue')
    ax3.bar(x, new_counts, bottom=original_counts, label='Added', alpha=0.7, color='red')
    
    ax3.set_xlabel('Stage')
    ax3.set_ylabel('Vertex Count')
    ax3.set_title('Mesh Growth')
    ax3.set_xticks(x)
    ax3.set_xticklabels(iterations)
    ax3.legend()
    
    # Plot 4: Outcome distribution
    ax4 = fig.add_subplot(234)
    
    outcome_counts = []
    outcome_labels = []
    outcome_colors_list = []
    
    for outcome_id in [0, 1, 2, 3]:
        count = jnp.sum(vertex_outcomes == outcome_id)
        if count > 0:
            outcome_counts.append(count)
            outcome_labels.append(outcome_mapping[outcome_id])
            outcome_colors_list.append(outcome_colors[outcome_id])
    
    ax4.pie(outcome_counts, labels=outcome_labels, colors=outcome_colors_list, autopct='%1.1f%%')
    ax4.set_title('Vertex Outcome Distribution')
    
    # Plot 5: Convergence metrics
    ax5 = fig.add_subplot(235)
    
    metrics = ['Vertices', 'Boundary Edges', 'Density']
    values = [
        complete_mesh_data['n_vertices'],
        convergence_metrics['boundary_crossing_edges'], 
        convergence_metrics['vertex_density']
    ]
    
    # Normalize for visualization
    normalized_values = [v / max(values) for v in values]
    
    ax5.bar(metrics, normalized_values, alpha=0.7, color=['blue', 'red', 'green'])
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('Convergence Metrics')
    ax5.set_ylim(0, 1.1)
    
    # Add actual values as text
    for i, (metric, value) in enumerate(zip(metrics, values)):
        ax5.text(i, normalized_values[i] + 0.05, f'{value:.1f}', 
                ha='center', va='bottom', fontsize=10)
    
    # Plot 6: Step 8 summary
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    
    stats_text = f"""RSE Step 8: Complete Mesh Rebuild

Mesh Consolidation:
  Total vertices: {complete_mesh_data['n_vertices']}
  Original: {complete_mesh_data['n_original']}
  Added this round: {complete_mesh_data['n_new']}
  
Topology:
  Tetrahedra: {complete_mesh_data['n_tetrahedra']}
  Edges: {complete_mesh_data['n_edges']}
  
Trajectory Outcomes:
  Impact: {jnp.sum(vertex_outcomes == 0)}
  Escape: {jnp.sum(vertex_outcomes == 1)}
  In-system: {jnp.sum(vertex_outcomes == 2)}
  Removed: {jnp.sum(vertex_outcomes == 3)}

Convergence:
  Iteration: {convergence_metrics['iteration']}
  Boundary edges: {convergence_metrics['boundary_crossing_edges']}
  Resolution: {convergence_metrics['boundary_resolution']:.2e}
  Density: {convergence_metrics['vertex_density']:.1f}

Status: Ready for next iteration
(or stopping criteria evaluation)
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig("figures/reachability/step8_complete_mesh.png", dpi=150, bbox_inches='tight')
    plt.show()

def execute_step8_rebuild_complete_mesh(step7_data, iteration_number=1, 
                                       stopping_criteria=None):
    """
    Execute RSE Step 8: Rebuild complete mesh with all trajectory information.
    """
    print("=== RSE Step 8: Rebuild Complete Mesh ===")
    
    # Rebuild complete mesh
    complete_mesh_data = rebuild_complete_mesh_with_trajectories(step7_data)
    
    # Analyze convergence
    convergence_metrics = analyze_mesh_convergence(complete_mesh_data, iteration_number)
    
    # Check stopping criteria
    if stopping_criteria is None:
        stopping_criteria = {'max_vertices': 5000, 'min_boundary_resolution': 1e-4, 'max_iterations': 10}
    
    should_stop, stop_reasons = check_refinement_stopping_criteria(
        convergence_metrics, **stopping_criteria
    )
    
    # Visualize complete mesh
    visualize_complete_mesh(complete_mesh_data, convergence_metrics)
    
    # Package Step 8 results
    step8_data = {
        'complete_mesh_data': complete_mesh_data,
        'convergence_metrics': convergence_metrics,
        'should_stop': should_stop,
        'stop_reasons': stop_reasons,
        'iteration_number': iteration_number,
        'ready_for_next_iteration': not should_stop
    }
    
    print(f"\n✓ Step 8 completed: Complete mesh rebuilt")
    print(f"✓ Total vertices: {complete_mesh_data['n_vertices']}")
    print(f"✓ Boundary edges: {convergence_metrics['boundary_crossing_edges']}")
    print(f"✓ Convergence status: {'STOP' if should_stop else 'CONTINUE'}")
    
    if should_stop:
        print(f"✓ RSE Algorithm completed: {', '.join(stop_reasons)}")
    else:
        print(f"✓ Ready for next iteration: Return to Step 5")
    
    return step8_data

# Usage Example:
step8_data = execute_step8_rebuild_complete_mesh(
    step7_data=step7_data,
    iteration_number=1,
    stopping_criteria={
        'max_vertices': 5000,
        'min_boundary_resolution': 1e-4, 
        'max_iterations': 10
    }
)

# Check if we should continue
if step8_data['ready_for_next_iteration']:
    print("Continue to next iteration: Step 5 with new mesh")
else:
    print("RSE Algorithm completed!")

print("✅ RSE Algorithm Steps 1-8 Complete!")
print("🔄 For multiple iterations: Steps 5→6→7→8 repeat until convergence")
print("📊 Each iteration refines boundaries and improves reachable set accuracy")

### Step 9

@jaxtyped(typechecker=typechecker)
def execute_rse_algorithm(
    mesh_data, trajectory_data, dynamical_system, initial_state, delta_v_magnitude,
    max_iterations=5, refinement_params=None, key=None
):
    """Execute complete RSE algorithm with iterative refinement (Steps 1-9).
    
    Args:
        mesh_data: Result from your existing Steps 1-2
        trajectory_data: Result from your existing Steps 3-4
        dynamical_system, initial_state, delta_v_magnitude: System parameters
        max_iterations: Maximum number of refinement iterations
        refinement_params: Parameters for Steps 5-8
        key: JAX random key
    """
    if key is None:
        key = jax.random.key(42)
    if refinement_params is None:
        refinement_params = {
            'weight_exponent': 2.0, 
            'fraction': 0.9, 
            'n_edges_to_add': 10,
            'sigma': 1.0
        }
    
    # Start with your existing mesh and trajectory data from Steps 1-4
    current_mesh_data = mesh_data
    current_trajectory_data = trajectory_data
    convergence_history = []
    
    # Step 9: Iterate Steps 5-8 until convergence
    for iteration in range(max_iterations):
        print(f"\n=== RSE Iteration {iteration + 1}/{max_iterations} ===")
        
        key, subkey = jax.random.split(key)
        
        # Execute one refinement round (Steps 5-8)
        step8_data = execute_refinement_round(
            current_mesh_data, current_trajectory_data, dynamical_system, 
            initial_state, delta_v_magnitude, refinement_params, subkey, iteration + 1
        )
        
        # Check convergence
        should_stop = step8_data['should_stop']
        metrics = step8_data['convergence_metrics'] 
        convergence_history.append(metrics)
        
        print(f"Iteration {iteration + 1} completed:")
        print(f"  Vertices: {metrics['total_vertices']}")
        print(f"  Boundary resolution: {metrics['boundary_resolution']:.2e}")
        print(f"  Should stop: {should_stop}")
        
        if should_stop:
            print(f"Converged after {iteration + 1} iterations")
            break
            
        # Update mesh data for next iteration - ensure correct structure
        complete_mesh = step8_data['complete_mesh_data']
        current_mesh_data = {
            'vertices': complete_mesh['vertices'],
            'triangulation': complete_mesh['triangulation'], 
            'unique_edges': complete_mesh['unique_edges'],
            'mesh_info': {
                'n_vertices': complete_mesh['n_vertices'],
                'n_tetrahedra': complete_mesh['n_tetrahedra'],
                'n_original': complete_mesh['n_original'],
                'n_new': complete_mesh['n_new']
            }
        }
        
        # Update trajectory data to align with refined mesh
        # All vertices in complete_mesh have been filtered and are "stable"
        n_total_vertices = complete_mesh['n_vertices']
        stable_mask = jnp.ones(n_total_vertices, dtype=bool)  # All current vertices are stable
        
        current_trajectory_data = {
            **trajectory_data,
            'stable_trajectories': complete_mesh['combined_trajectories'],
            'outcomes': complete_mesh['combined_outcomes'],
            'stable_mask': stable_mask,
            # Add vertex outcomes directly from Step 8
            'vertex_outcomes': complete_mesh['vertex_outcomes']
        }
    
    return step8_data, convergence_history

@jaxtyped(typechecker=typechecker) 
def execute_refinement_round(mesh_data, trajectory_data, dynamical_system, initial_state, 
                           delta_v_magnitude, params, key, iteration):
    """Execute one round of Steps 5-8."""
    print(f"  Executing Step 5: Refinement heuristics...")
    
    # For iterations > 1, vertex outcomes are already computed in trajectory_data
    if 'vertex_outcomes' in trajectory_data:
        # Use pre-computed vertex outcomes from previous iteration
        vertex_outcomes = trajectory_data['vertex_outcomes']
        outcome_mapping = {0: 'impact', 1: 'escape', 2: 'in-system', 3: 'removed'}
        
        # Calculate edge weights directly
        edge_info = calculate_edge_weights_end_result_heuristic(
            mesh_data, vertex_outcomes, params['weight_exponent']
        )
        
        # Sort edges
        sorted_edge_lists = sort_edges_by_end_result_heuristic(edge_info)
        
        # Select edges for subdivision
        selection_results = select_edges_for_subdivision(
            sorted_edge_lists, params['n_edges_to_add'], params['fraction'], key
        )
        
        refinement_data = {
            'vertex_outcomes': vertex_outcomes,
            'outcome_mapping': outcome_mapping,
            'edge_info': edge_info,
            'sorted_edge_lists': sorted_edge_lists,
            'selection_results': selection_results,
            'parameters': params
        }
    else:
        # First iteration - use the full Step 5 function
        refinement_data = execute_step5_refinement_heuristics(
            mesh_data, trajectory_data, 
            weight_exponent=params['weight_exponent'],
            fraction=params['fraction'], 
            n_edges_to_add=params['n_edges_to_add'],
            key=key
        )
    
    print(f"  Executing Step 6: Mesh refinement...")
    step6_data = execute_step6_mesh_refinement(
        mesh_data, refinement_data, 
        sigma=params['sigma'], 
        key=key
    )
    
    print(f"  Executing Step 7: Update trajectories...")
    step7_data = execute_step7_update_trajectories(
        step6_data, trajectory_data, dynamical_system, 
        initial_state, delta_v_magnitude
    )
    
    print(f"  Executing Step 8: Rebuild complete mesh...")  
    step8_data = execute_step8_rebuild_complete_mesh(
        step7_data, iteration,
        stopping_criteria={'max_vertices': 5000, 'min_boundary_resolution': 1e-4, 'max_iterations': 10}
    )
    
    return step8_data

# Usage example with your existing working code:

# You already have working mesh_data and trajectory_data from Steps 1-4
# Now use Step 9 to iterate Steps 5-8:

final_results, history = execute_rse_algorithm(
    mesh_data=mesh_data,                    # Your existing mesh_data from Steps 1-2
    trajectory_data=trajectory_data,        # Your existing trajectory_data from Steps 3-4
    dynamical_system=dynamical_system,
    initial_state=initial_state, 
    delta_v_magnitude=delta_v_magnitude,
    max_iterations=10,                       # Number of refinement rounds
    refinement_params={
        'weight_exponent': 2.0,
        'fraction': 0.9, 
        'n_edges_to_add': 10,
        'sigma': 1.0
    },
    key=jax.random.key(42)
)

print(f"RSE Algorithm completed:")
print(f"  Final vertices: {final_results['complete_mesh_data']['n_vertices']}")
print(f"  Iterations: {len(history)}")
print(f"  Converged: {final_results['should_stop']}")

