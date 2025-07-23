import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from scipy.spatial import ConvexHull

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import DeepSpaceNetwork, simulate_thrust
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.statistics import silverman_kde_estimate

dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()

key = jax.random.key(42)
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey)
print(posterior_ensemble.shape)

delta_v_magnitude = 2.0

time_range = 0.242
num_particles = 100
key, subkey = jax.random.split(key)
simulated_ensemble = simulate_thrust(subkey, posterior_ensemble, num_particles, delta_v_magnitude)
print(simulated_ensemble.shape)

simulated_trajectories = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, simulated_ensemble)
print(simulated_trajectories.shape)

import polytopax as ptx
hull = ptx.ConvexHull.from_points(simulated_trajectories)
