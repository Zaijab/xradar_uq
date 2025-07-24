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
from xradar_uq.statistics import silverman_kde_estimate

from xradar_uq.statistics.silverman_kde import GMM
from typing import Callable

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

# Ensemble -> Reachable Set (Conv Hull) -> Triangulation -> Barycentric Subdivision
# Barycentric Subdivision -> Candidate Points


# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def generate_triangulated_candidates(
    convex_hull: ptx.ConvexHull, subdivision_depth: int = 2
) -> Float[Array, "n_candidates 2"]:
    triangles = fan_triangulate(convex_hull)
    for _ in range(subdivision_depth):
        triangles = subdivide_triangles(triangles)
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
candidates = generate_triangulated_candidates(my_hull, subdivision_depth=3)
from jax.tree_util import Partial

# @jaxtyped(typechecker=typechecker)

from typing import Callable

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

gmm_pdf_selection = Partial(select_pdf_weighted_candidate, gmm=gmm)
place_sensors(key, candidates, 3, jnp.deg2rad(5), gmm_pdf_selection)

random_selection = Partial(select_random_candidate)
place_sensors(key, candidates, 3, jnp.deg2rad(5), random_selection)

# Todo:
# Plot Hull + Sensor Locations + Color associated with 2D GMM PDF
# Plot the triangulation
# Explain Barycentric subdivision, arbitrarily small mesh size
# Flesh out paper with plots to fill in the blank
# Run Different Tasking Algorithm on DRO-A
# Plot multiple frontiers
# Make measurement frequency plot: How many measurements for 90%> Tracking
# 
