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
from tqdm.auto import tqdm

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.geometry import reachable_set

key = jax.random.key(42)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from typing import Callable
from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem


import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from scipy.spatial import Delaunay

def _hull_check(test_point, points):
    return Delaunay(points).find_simplex(test_point) >= 0

@jax.jit
@jax.vmap
def sample_gaussian_mixture(key: Key[Array, ""], point: Float[Array, "state_dim"], cov: Float[Array, "state_dim state_dim"]) -> Float[Array, "state_dim"]:
    return jax.random.multivariate_normal(key, mean=point, cov=cov)


stochastic_filter = EnGMF()

true_state = dynamical_system.initial_state()
posterior_ensemble = dynamical_system.generate(subkey, final_time=0.0, batch_size=100)
num_measurements = 10
time_horrizon = 1.0
delta_v_magnitude = 0.0
num_simulations = 1
errors = []

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polytopax as ptx
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AnglesOnly, simulate_thrust


@eqx.filter_jit
@eqx.filter_vmap
def rejection_sample(
    key: jax.Array,
    mean: Float[Array, "state_dim"], 
    cov: Float[Array, "state_dim state_dim"],
    discriminator: Callable[[Float[Array, "state_dim"]], bool],
) -> Float[Array, "state_dim"]:
    
    def cond_fun(carry):
        return ~carry[2]
    
    def body_fun(carry):
        rng, sample, accepted = carry
        rng, subkey = jax.random.split(rng)
        candidate = jax.random.multivariate_normal(subkey, mean, cov)
        pass_discriminator = discriminator(candidate)
        sample = jnp.where(pass_discriminator, candidate, sample)
        accepted = accepted | pass_discriminator
        return (rng, sample, accepted)
    
    carry_init = (key, jnp.zeros_like(mean), False)
    _, final_sample, _ = eqx.internal.while_loop(
        cond_fun, body_fun, carry_init, kind="bounded", max_steps=10 ** 1
    )
    return final_sample

key, subkey, thrust_key = jax.random.split(key, 3)

# current_reachable_set = reachable_set_no_thrust(thrust_key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system, n_directions=100)

def make_hull_discriminator(points: Float[Array, "n_points state_dim"]):
    def discriminator(test_point: Float[Array, "state_dim"]) -> bool:
        result_spec = jax.ShapeDtypeStruct((), jnp.bool_)
        return jax.pure_callback(_hull_check, result_spec, test_point, points, vmap_method='sequential')
    return discriminator

hull_discriminator = make_hull_discriminator(prior_ensemble)
stochastic_filter = EnGMF(sampling_function=jax.tree_util.Partial(rejection_sample, discriminator=hull_discriminator))

for _ in range(num_measurements):
    print(_)
    key, subkey, thrust_key = jax.random.split(key, 3)
    true_state = dynamical_system.flow(0.0, time_horrizon, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horrizon, posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)
    error = true_state - jnp.mean(posterior_ensemble, axis=0)
    errors.append(error)
    break



rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
rmse
