import os

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

import pandas as pd


from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AbstractMeasurementSystem, Radar, tracking_measurability
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.statistics import generate_random_impulse_velocity, silverman_kde_estimate

dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = Radar()

key = jax.random.key(42)
key, subkey = jax.random.split(key)
time_range = 0.242 # TU: 0.242 is approx 1 Day
measurement_time = 1 # How many measurements * time_range
total_fuel = 1.0
delta_v_magnitude = 1e-2
maneuver_proportion = 0.1

true_state = jnp.load("cache/true_state_1000.npy")
posterior_ensemble = jnp.load("cache/posterior_1000_window.npy")
key, subkey = jax.random.split(key)
random_impulse_velocity = generate_random_impulse_velocity(subkey, delta_v_magnitude)


# for i in range(measurement_time):
#     key, update_key, measurement_key, window_center_key, thrust_key = jax.random.split(key, 5)
#     true_state = dynamical_system.flow(0.0, time_range, true_state)

#     prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)
#     predicted_state = jnp.mean(prior_ensemble, axis=0)

posterior_ensemble.shape


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def simulate_thrust(
    key: Key[Array, ""],
    posterior_ensemble: Float[Array, "ensemble_size state_dim"], 
    num_particles: int,
    delta_v_magnitude: float,
) -> Float[Array, "ensemble_size*{num_particles} state_dim"]:
    ensemble_size, state_dim = posterior_ensemble.shape
    keys = jax.random.split(key, ensemble_size * num_particles)
    
    velocity_impulses = eqx.filter_vmap(lambda k: generate_random_impulse_velocity(k, delta_v_magnitude))(keys)
    state_impulses = jnp.concatenate([jnp.zeros((ensemble_size * num_particles, 3)), velocity_impulses], axis=1)
    state_impulses = state_impulses.reshape(ensemble_size, num_particles, state_dim)
    
    expanded_ensemble = jnp.repeat(posterior_ensemble[:, None, :], num_particles, axis=1)
    result = (expanded_ensemble + state_impulses).reshape(-1, state_dim)
    
    assert result.shape == (ensemble_size * num_particles, state_dim)
    return result

# Constrain min num particles in FOV by percentage

import jax.numpy as jnp
import jaxkd as jk

from beartype import beartype as typechecker
import equinox as eqx

@eqx.filter_jit
@typechecker
def compute_sparsity_measures(points: Float[Array, "n d"], k: int) -> Float[Array, "n"]:
    """Compute k-NN sparsity measure for each point."""
    assert points.ndim == 2
    assert k > 0 and k < points.shape[0]
    
    tree = jk.build_tree(points)
    _, distances = jk.query_neighbors(tree, points, k=k+1)  # +1 to exclude self
    kth_distances = distances[:, k]  # k-th neighbor (excluding self)
    
    assert kth_distances.shape == (points.shape[0],)
    return kth_distances


time_range = 0.242 # TU: 0.242 is approx 1 Day
measurement_time = 1000 # How many measurements * time_range

mc_iterations = 1
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, mc_iterations)
delta_v_range = [] #np.logspace(-3, -1, 20)
maneuver_proportion_range = [0.09] #np.linspace(0, 0.2, 10)

index = pd.MultiIndex.from_product(
    [delta_v_range, maneuver_proportion_range, range(mc_iterations)], 
    names=['delta_v_magnitude', 'maneuver_proportion', 'mc_iteration']
)
df = pd.DataFrame(index=index, columns=["times_found"])


for delta_v_magnitude in delta_v_range:
    print(f"{delta_v_magnitude=}")
    for maneuver_proportion in maneuver_proportion_range:
        print(f"{maneuver_proportion=}")
        for mc_iteration_i, subkey in enumerate(subkeys):
            df.loc[(float(delta_v_magnitude), float(maneuver_proportion), mc_iteration_i), ("times_found")]
            total_fuel = 10.0

            true_state = jnp.load("cache/true_state_1000.npy")
            posterior_ensemble = jnp.load("cache/posterior_1000_window.npy")

            key, subkey = jax.random.split(key)
            random_impulse_velocity = generate_random_impulse_velocity(subkey, delta_v_magnitude)

            times_found = 0

            for i in range(measurement_time):
                key, update_key, measurement_key, window_center_key, thrust_key = jax.random.split(key, 5)
                true_state = dynamical_system.flow(0.0, time_range, true_state)

                if jax.random.bernoulli(thrust_key, p=maneuver_proportion):
                    if total_fuel > 0:
                        key, subkey = jax.random.split(key)
                        total_fuel -= delta_v_magnitude
                        true_state = true_state.at[3:].add(random_impulse_velocity)


                prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)



                if tracking_measurability(true_state, jnp.mean(prior_ensemble, axis=0)): # 0.077
                    times_found += 1
                    posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
                else:
                    print("Not FOUND TRYING SECOND SENSOR")
                    num_particles = 100
                    key, subkey = jax.random.split(key)
                    simulated_ensemble = simulate_thrust(subkey, posterior_ensemble, num_particles, delta_v_magnitude)
                    simulated_trajectories = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, simulated_ensemble)
                    sparsity_distance = compute_sparsity_measures(simulated_trajectories,
                                                                  int(jnp.sqrt(simulated_trajectories.shape[0])))

                    if tracking_measurability(true_state, simulated_trajectories[jnp.argmin(sparsity_distance)]) or tracking_measurability(true_state, simulated_trajectories[jnp.argmax(sparsity_distance)]):
                        print("FOUND WITH SECOND SENSOR")
                        times_found += 1
                        posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
                    else:
                        posterior_ensemble = prior_ensemble

            found_proportion = times_found / measurement_time
            print(found_proportion)
            df.loc[(float(delta_v_magnitude), float(maneuver_proportion), mc_iteration_i), ("times_found")] = found_proportion
        break
    break
