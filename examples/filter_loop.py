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
from diffrax import PIDController

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AnglesOnly, Radar
from xradar_uq.stochastic_filters import EnGMF, EnKF

key = jax.random.key(42)

dynamical_system = CR3BP(stepsize_controller=PIDController(rtol=1e-4, atol=1e-4))
measurement_system = Radar()
stochastic_filter = EnGMF()

total_time = 2.5

mc_iterations = 3

for num_measurements in [400, 200, 100, 50, 25, 15, 10]:
    print(num_measurements, end=": ")
    mc_rmses = []
    for mc_iteration in range(mc_iterations):
        key, subkey = jax.random.split(key)
        true_state = dynamical_system.initial_state()
        posterior_ensemble = dynamical_system.generate(subkey, final_time=0.0, batch_size=1_000)
        errors = []
        time_points = jnp.linspace(0, total_time, num_measurements)
        dt = float(time_points[1] - time_points[0])
        for _ in range(num_measurements - 1):
            key, subkey = jax.random.split(key)
            true_state = dynamical_system.flow(0.0, dt, true_state)
            prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, dt, posterior_ensemble)
            posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)
            error = true_state - jnp.mean(posterior_ensemble, axis=0)
            errors.append(error)
        rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
        print(rmse, end=", ")
        mc_rmses.append(rmse)
    print()
    print(jnp.mean(jnp.asarray(mc_rmses)))
