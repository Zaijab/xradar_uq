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

true_state = dynamical_system.initial_state()
posterior_ensemble = dynamical_system.generate(subkey)


time_range = 0.242 # TU: 0.242 is approx 1 Day
measurement_time = 1000 # How many measurements * time_range

mc_iterations = 1
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, mc_iterations)
delta_v_range = np.logspace(-3, -1, 20)
maneuver_proportion_range = np.linspace(0, 0.2, 10)
index = pd.MultiIndex.from_product(
    [delta_v_range, maneuver_proportion_range, range(mc_iterations)], 
    names=['delta_v_magnitude', 'maneuver_proportion', 'mc_iteration']
)
df = pd.DataFrame(index=index, columns=["times_found"])


for delta_v_magnitude in jnp.logspace(-3, -1, 20):
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
                predicted_state = jnp.mean(prior_ensemble, axis=0)

                if tracking_measurability(true_state, predicted_state):
                    times_found += 1
                    posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
                else:
                    posterior_ensemble = prior_ensemble

            found_proportion = times_found / measurement_time
            print(found_proportion)
            df.loc[(float(delta_v_magnitude), float(maneuver_proportion), mc_iteration_i), ("times_found")] = found_proportion
        break
    break
    
