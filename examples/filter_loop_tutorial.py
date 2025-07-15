import equinox as eqx
import jax
from diffrax import SaveAt
import time
import jax.numpy as jnp

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)
time_step = 0.12

# State Space Model
dynamical_system = CR3BP()
measurement_system = Radar()

# Filtering Method
stochastic_filter = EnGMF()

# True State (actual satellite) x^t
true_state = dynamical_system.initial_state() 

key, subkey = jax.random.split(key)

# x^+
posterior_ensemble = dynamical_system.generate(subkey, batch_size=100) 

# Predict Step - Numerically integrate every particle / state
errors = []

for _ in range(10):
    saveat = SaveAt(t1=True)
    # Numerical integration of one state
    true_state = dynamical_system.trajectory(0.0, time_step, true_state, saveat)[1][0]
    # Numerical integration of all states in the batch (vectorized)
    # x^-
    prior_ensemble = eqx.filter_vmap(dynamical_system.trajectory, in_axes=(None, None, 0, None))(0.0, time_step, posterior_ensemble, saveat)[1][:, 0, :] # 100, 6

    # Take a "noisy" measurement of our stat
    key, subkey = jax.random.split(key)
    measurement = measurement_system(true_state, subkey)

    key, subkey = jax.random.split(key)
    posterior_ensemble = stochastic_filter.update(key, prior_ensemble, measurement, measurement_system)
    error = true_state - jnp.mean(posterior_ensemble, axis=0)
    errors.append(error)

rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
rmse * 389703

