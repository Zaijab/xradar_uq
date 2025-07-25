from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polytopax as ptx
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.geometry import (angular_reachable_set, barycentric_subdivision,
                                fan_triangulate)
from xradar_uq.measurement_systems import AnglesOnly, DeepSpaceNetwork
from xradar_uq.statistics import silverman_kde_estimate
from xradar_uq.statistics.silverman_kde import GMM
from xradar_uq.stochastic_filters import EnGMF

dynamical_system = CR3BP()
true_state = dynamical_system.initial_state()

stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()
angles = AnglesOnly()

key = jax.random.key(42)
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey)

delta_v_magnitude = 2.0
time_range = 10*0.242
num_particles = 100

gmm = silverman_kde_estimate(posterior_ensemble)

# Ensemble -> Reachable Set (Conv Hull) -> Triangulation -> Barycentric Subdivision
# Barycentric Subdivision -> Candidate Points



# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def generate_triangulated_candidates(
    convex_hull: ptx.ConvexHull, subdivision_depth: int = 2
) -> Float[Array, "n_candidates 2"]:
    triangles = fan_triangulate(convex_hull)
    for _ in range(subdivision_depth):
        triangles = barycentric_subdivision(triangles)
    centroids = jnp.mean(triangles, axis=1)
    assert centroids.shape[1] == 2
    return centroids

# @jaxtyped(typechecker=typechecker) 
@eqx.filter_jit
def select_random_candidate(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"]
) -> Float[Array, "2"]:
    random_idx = jax.random.choice(key, candidates.shape[0])
    selected_position = candidates[random_idx]
    assert selected_position.shape == (2,)
    return selected_position

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(candidates)
    optimal_idx = jnp.argmax(pdf_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position


# Random placement
hull = angular_reachable_set(subkey, posterior_ensemble, time_range, delta_v_magnitude, num_particles, dynamical_system)
candidates = generate_triangulated_candidates(hull, subdivision_depth=3)
from typing import Callable

from jax.tree_util import Partial

# @jaxtyped(typechecker=typechecker)

# @jaxtyped(typechecker=typechecker) 
@eqx.filter_jit
def select_random_candidate(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"]
) -> Float[Array, "2"]:
    random_idx = jax.random.choice(key, candidates.shape[0])
    selected_position = candidates[random_idx]
    assert selected_position.shape == (2,)
    return selected_position

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(candidates)
    optimal_idx = jnp.argmax(pdf_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position

@eqx.filter_jit
def place_sensors(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"],
    n_sensors: int, exclusion_radius: float, selection: Callable
) -> Float[Array, "n_sensors 2"]:
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_distances, axis=1)
        valid_candidate_mask = min_distances > exclusion_radius
        
        valid_candidates = jnp.where(valid_candidate_mask[:, None], candidates, jnp.inf)
        finite_mask = jnp.isfinite(valid_candidates).all(axis=1)
        filtered_candidates = jnp.where(finite_mask[:, None], valid_candidates, 0.0)
        
        new_position = selection(subkey, filtered_candidates)
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        
        return (updated_positions, updated_mask, key_state)
    
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, (sensor_positions, placement_mask, key))
    assert final_positions.shape == (n_sensors, 2)
    return final_positions

gmm_pdf_selection = Partial(select_pdf_weighted_candidate, gmm=kde_model)
place_sensors(key, all_vertices, 3, jnp.deg2rad(5), gmm_pdf_selection)

###

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    finite_mask = jnp.isfinite(candidates).all(axis=1)
    
    valid_candidates = jnp.where(finite_mask[:, None], candidates, 0.0)
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(valid_candidates)
    masked_weights = jnp.where(finite_mask, pdf_weights, -jnp.inf)
    optimal_idx = jnp.argmax(masked_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position

@eqx.filter_jit
def place_sensors(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"],
    n_sensors: int, exclusion_radius: float, selection: Callable
) -> Float[Array, "n_sensors 2"]:
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_distances, axis=1)
        valid_candidate_mask = min_distances > exclusion_radius
        
        valid_candidates = jnp.where(valid_candidate_mask[:, None], candidates, jnp.inf)
        new_position = selection(subkey, valid_candidates)
        
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        
        return (updated_positions, updated_mask, key_state)
    
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, (sensor_positions, placement_mask, key))
    return final_positions

# Usage:
exclusion_radius = jnp.deg2rad(5.0)
jax.debug.print("Exclusion radius: {r}, Candidate range: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]", 
                r=exclusion_radius, 
                xmin=jnp.min(all_vertices[:, 0]), xmax=jnp.max(all_vertices[:, 0]),
                ymin=jnp.min(all_vertices[:, 1]), ymax=jnp.max(all_vertices[:, 1]))

gmm_pdf_selection = Partial(select_pdf_weighted_candidate, gmm=kde_model)
sensor_positions = place_sensors(key, all_vertices, 3, exclusion_radius, gmm_pdf_selection)
assert jnp.max(jnp.abs(sensor_positions[:, None, :] - sensor_positions[None, :, :]), axis=2) > jnp.deg2rad(5)

sensor_positions
###

# random_selection = Partial(select_random_candidate)
# place_sensors(key, candidates, 3, jnp.deg2rad(5), random_selection)

# Todo:
# Plot Hull + Sensor Locations + Color associated with 2D GMM PDF
# Plot the triangulation
# Explain Barycentric subdivision, arbitrarily small mesh size
# Flesh out paper with plots to fill in the blank
# Run Different Tasking Algorithm on DRO-A
# Plot multiple frontiers
# Make measurement frequency plot: How many measurements for 90%> Tracking
# 

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def test_sensor_exclusion(
    sensor_positions: Float[Array, "n_sensors 2"],
    exclusion_radius: float | Float[Array, ""]
) -> Bool[Array, ""]:
    n_sensors = sensor_positions.shape[0]
    pairwise_distances = jnp.max(jnp.abs(
        sensor_positions[:, None, :] - sensor_positions[None, :, :]
    ), axis=2)
    
    upper_triangular_mask = jnp.triu(jnp.ones((n_sensors, n_sensors)), k=1)
    masked_distances = jnp.where(upper_triangular_mask, pairwise_distances, jnp.inf)
    min_separation = jnp.min(masked_distances)
    
    assert min_separation.shape == ()
    return min_separation <= exclusion_radius

# sensor_positions = jnp.array([[0.10110731, 0.34845498], [0., 0.], [0., 0.]])
# exclusion_radius = jnp.deg2rad(5)
# test_sensor_exclusion(sensor_positions, exclusion_radius)

def sensor_tracking(true_state, prior_ensemble, key, posterior_ensemble):
    true_angles = AnglesOnly()(true_state)
    sensor_placement = place_sensors(key, candidates, 3, jnp.deg2rad(5), gmm_pdf_selection)
    print(sensor_placement)
    distances = jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1)
    return jnp.any(distances <= jnp.deg2rad(5))

sensor_tracking(true_state, posterior_ensemble, subkey, posterior_ensemble)
