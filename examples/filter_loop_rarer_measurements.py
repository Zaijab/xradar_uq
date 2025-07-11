import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import RangeSensor
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
measurement_system = RangeSensor()
stochastic_filter = EnGMF()

true_state = dynamical_system.initial_state()
posterior_ensemble = dynamical_system.generate(subkey, final_time=0.0)


for _ in range(10):
    key, subkey = jax.random.split(key)
    true_state = dynamical_system.flow(0.0, 1.0, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow, in_axes=(None, None, 0))(0.0, jnp.array(1.0), posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)
    errors.append(true_state - jnp.mean(posterior_ensemble, axis=0))

rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
rmse
