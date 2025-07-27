import equinox as eqx
import jax
from diffrax import SaveAt
import time
import jax.numpy as jnp

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.stochastic_filters import EnGMF, EnKF

key = jax.random.key(42)
time_step = 0.12

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()
stochastic_filter = EnGMF()

true_state = dynamical_system.initial_state() 
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey, batch_size=1000) 

errors = []

for _ in range(10):
    true_state = dynamical_system.flow(0.0, time_step, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_step, posterior_ensemble)

    key, subkey = jax.random.split(key)
    measurement = measurement_system(true_state, subkey)

    key, subkey = jax.random.split(key)
    posterior_ensemble = stochastic_filter.update(key, prior_ensemble, measurement, measurement_system)
    error = true_state - jnp.mean(posterior_ensemble, axis=0)
    errors.append(error)

rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
rmse
