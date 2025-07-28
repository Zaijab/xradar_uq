import equinox as eqx
import jax
from diffrax import SaveAt
import time
import jax.numpy as jnp

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.stochastic_filters import EnGMF, EnKF, UKF

key = jax.random.key(42)

dynamical_system = CR3BP()
true_state = dynamical_system.initial_state()


# measurement_system = DeepSpaceNetwork()
# stochastic_filter = UKF()


# key, subkey = jax.random.split(key)
# posterior_ensemble = dynamical_system.generate(subkey, batch_size=13) 
