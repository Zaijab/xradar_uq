from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.evaluate import evaluate_tracking_grid
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.statistics import GMM
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)

delta_v_range = jnp.linspace(0.001, 0.5, 20) # 0.5
maneuver_proportion_range = jnp.linspace(0.0, 1.0, 10) # 0.5

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()
stochastic_filter = EnGMF()

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


import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped
from xradar_uq.geometry import (angular_reachable_set, barycentric_subdivision,
                                fan_triangulate)
from xradar_uq.measurement_systems import AnglesOnly
from xradar_uq.statistics import silverman_kde_estimate


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sensor_tracking_max_pdf(true_state, prior_ensemble, key, posterior_ensemble,
                            dynamical_system, time_horrizon, delta_v_magnitude, num_simulations=100):
    true_angles = AnglesOnly()(true_state)
    hull = angular_reachable_set(key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    triangles = barycentric_subdivision(barycentric_subdivision(fan_triangulate(hull)))
    candidates = jnp.mean(triangles, axis=1)
    gmm = silverman_kde_estimate(posterior_ensemble)
    
    sensor_positions = jnp.zeros((3, 2))
    placement_mask = jnp.zeros(3, dtype=bool)
    
    def place_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, _ = jax.random.split(key_state)
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_mask = jnp.min(jnp.where(mask[None, :], distances, jnp.inf), axis=1) > jnp.deg2rad(5)
        filtered_candidates = jnp.where(valid_mask[:, None], candidates, 0.0)
        optimal_idx = jnp.argmax(eqx.filter_vmap(gmm.positional_pdf)(filtered_candidates))
        return (positions.at[i].set(filtered_candidates[optimal_idx]), mask.at[i].set(True), key_state)
    
    sensor_placement, _, _ = jax.lax.fori_loop(0, 3, place_sensor, (sensor_positions, placement_mask, key))
    assert sensor_placement.shape == (3, 2)
    return jnp.any(jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1) <= jnp.deg2rad(5))

# key, subkey = jax.random.split(key)
# posterior_ensemble = dynamical_system.generate(key)
# prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 0.24, posterior_ensemble)
# true_state = dynamical_system.initial_state()
# true_state = dynamical_system.flow(0.0, 0.24, true_state)

# sensor_tracking_max_pdf(true_state, posterior_ensemble, subkey, posterior_ensemble, dynamical_system, 0.24, 1e-5)


results_single = evaluate_tracking_grid(
    delta_v_range, maneuver_proportion_range, key,
    dynamical_system, measurement_system, stochastic_filter,
    sensor_tracking_max_pdf, mc_iterations = 1
)

# df = pd.DataFrame(
#     jnp.mean(results_single, axis=-1),
#     index=((389703 / 382981) * delta_v_range),
#     columns=maneuver_proportion_range
# )
# return df
