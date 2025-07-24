import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polytopax as ptx
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import (AnglesOnly, DeepSpaceNetwork,
                                           simulate_thrust)
from xradar_uq.stochastic_filters import EnGMF

dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()
angles = AnglesOnly()

key = jax.random.key(42)
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey)

delta_v_magnitude = 2.0

time_range = 0.242
num_particles = 100
key, subkey = jax.random.split(key)
simulated_ensemble = simulate_thrust(subkey, posterior_ensemble, num_particles, delta_v_magnitude)
simulated_trajectories = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, simulated_ensemble)
simulated_ensemble.shape

ensemble_angles = eqx.filter_vmap(angles)(simulated_trajectories)


@jax.jit
def angular_convex_hull(ensemble_angles: Float[Array, "n_samples 2"]) -> Float[Array, "n_hull_vertices 2"]:
    azimuth_range = jnp.max(ensemble_angles[:, 0]) - jnp.min(ensemble_angles[:, 0])
    
    # Handle azimuth wraparound
    wrapped_angles = jnp.where(azimuth_range > jnp.pi,
                               jnp.where(ensemble_angles[:, 0] < 0, 
                                       ensemble_angles[:, 0] + 2*jnp.pi, 
                                       ensemble_angles[:, 0]),
                               ensemble_angles[:, 0])
    points_2d = jnp.column_stack([wrapped_angles, ensemble_angles[:, 1]])

    return ptx.ConvexHull.from_points(points_2d)

my_hull = angular_convex_hull(ensemble_angles)

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def fan_triangulate(hull: ptx.ConvexHull) -> Float[Array, "n_triangles 3 2"]:
    hull_vertices = hull.vertices_array()
    centroid = hull.centroid()
    n_vertices = hull_vertices.shape[0]
    
    # Create triangles: (centroid, vertex_i, vertex_{i+1})
    triangles = jnp.zeros((n_vertices, 3, 2))
    triangles = triangles.at[:, 0, :].set(centroid)  # All triangles share centroid
    triangles = triangles.at[:, 1, :].set(hull_vertices)  # Current vertex
    triangles = triangles.at[:, 2, :].set(jnp.roll(hull_vertices, -1, axis=0))  # Next vertex
    
    return triangles

triangles = fan_triangulate(my_hull)

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def subdivide_triangles(triangles: Float[Array, "n_triangles 3 2"]) -> Float[Array, "4*n_triangles 3 2"]:
    n_triangles = triangles.shape[0]
    
    # Extract vertices A, B, C for all triangles
    A = triangles[:, 0, :]  # shape (n_triangles, 2)
    B = triangles[:, 1, :]  # shape (n_triangles, 2)
    C = triangles[:, 2, :]  # shape (n_triangles, 2)
    
    # Compute edge midpoints
    M_AB = (A + B) / 2
    M_BC = (B + C) / 2  
    M_AC = (A + C) / 2
    
    # Create 4 sub-triangles per original triangle
    new_triangles = jnp.zeros((n_triangles, 4, 3, 2))
    
    # Triangle 1: (A, M_AB, M_AC)
    new_triangles = new_triangles.at[:, 0, 0, :].set(A)
    new_triangles = new_triangles.at[:, 0, 1, :].set(M_AB)
    new_triangles = new_triangles.at[:, 0, 2, :].set(M_AC)
    
    # Triangle 2: (M_AB, B, M_BC)
    new_triangles = new_triangles.at[:, 1, 0, :].set(M_AB)
    new_triangles = new_triangles.at[:, 1, 1, :].set(B)
    new_triangles = new_triangles.at[:, 1, 2, :].set(M_BC)
    
    # Triangle 3: (M_AC, M_BC, C)  
    new_triangles = new_triangles.at[:, 2, 0, :].set(M_AC)
    new_triangles = new_triangles.at[:, 2, 1, :].set(M_BC)
    new_triangles = new_triangles.at[:, 2, 2, :].set(C)
    
    # Triangle 4: (M_AB, M_BC, M_AC) - center triangle
    new_triangles = new_triangles.at[:, 3, 0, :].set(M_AB)
    new_triangles = new_triangles.at[:, 3, 1, :].set(M_BC)
    new_triangles = new_triangles.at[:, 3, 2, :].set(M_AC)
    
    return new_triangles.reshape(4 * n_triangles, 3, 2)

more_triangles = subdivide_triangles(triangles)



# What I have so far:
# True GMM evaluation in 2D using projected normal
# Triangulations of 2D space with subdivision
# What's next:
# Task sensors to look in triangulation without overlap
# Even more next:
# Task sensors to go in highest PDF
# Task randomly
# 


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def place_non_overlapping_sensors(
    key: jax.Array,
    convex_hull_vertices: Float[Array, "n_vertices 2"],
    n_sensors: int,
    sensor_half_width: float = 2.5,
) -> Float[Array, "n_sensors 2"]:
    exclusion_radius = 2 * sensor_half_width
    hull_center = jnp.mean(convex_hull_vertices, axis=0)
    max_hull_radius = jnp.max(jnp.linalg.norm(convex_hull_vertices - hull_center, axis=1))
    
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key = carry
        key, subkey = jax.random.split(key)
        
        candidates = hull_center[None, :] + jax.random.uniform(
            subkey, (1000, 2), minval=-max_hull_radius, maxval=max_hull_radius
        )
        
        distances = jnp.linalg.norm(candidates[:, None, :] - positions[None, :, :], axis=2)
        valid_position_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_position_distances, axis=1)
        valid_mask = min_distances > exclusion_radius
        first_valid_idx = jnp.argmax(valid_mask)
        new_position = candidates[first_valid_idx]
        
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        return (updated_positions, updated_mask, key)
    
    initial_carry = (sensor_positions, placement_mask, key)
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, initial_carry)
    
    assert final_positions.shape == (n_sensors, 2)
    return final_positions

key, subkey = jax.random.split(key)
place_non_overlapping_sensors(subkey, my_hull.vertices, 3)
