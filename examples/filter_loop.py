"""
This is an example script which executes the entire filtering loop and calculates the ST-RMSE.
"""

"""
"""

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

"""
Now we shall instantiate the objects we imported.
They will constitute our state space system.
"""


"""
First, the dynamical system object.
"""
dynamical_system = CR3BP()

"""
Next, the measurement system.
It is a `callable` and has a `covariance` property.
"""
measurement_system = RangeSensor()

"""
Lastly, the filter itself.
Python has a builtin function called `filter` already.
In order to avoid overwriting the name `filter`, we call it `stochastic_filter.
"""
stochastic_filter = EnGMF()
true_state = dynamical_system.initial_state()
posterior_ensemble = dynamical_system.generate(subkey)

for _ in range(10):
    key, subkey = jax.random.split(key)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)
