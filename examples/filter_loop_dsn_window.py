# Setup (same as your original)
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped, Int, Bool

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar, tracking_measurability, DeepSpaceNetwork, AbstractMeasurementSystem
from xradar_uq.statistics import generate_random_impulse_velocity, silverman_kde_estimate
from xradar_uq.stochastic_filters import EnGMF



dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()
# measurement_system = Radar(covariance=jnp.diag(jnp.array([0.25, 0.01, 0.01])))

# Define parameter ranges
# delta_v_range = jnp.logspace(-3, -1, 20)
# maneuver_proportion_range = jnp.linspace(0, 0.2, 20)

delta_v_range = jnp.logspace(-3, 0, 20)
maneuver_proportion_range = jnp.linspace(0, 0.5, 20)


# Run optimized computation
key = jax.random.key(42)
results = evaluate_tracking_grid(
    delta_v_range, 
    maneuver_proportion_range, 
    key,
    dynamical_system, 
    measurement_system, 
    stochastic_filter,
    mc_iterations=20
)

# # Convert to DataFrame format matching your original
# import pandas as pd

# n_dv, n_mp, n_mc = results.shape
# index_arrays = []
# for i, dv in enumerate(delta_v_range):
#     for j, mp in enumerate(maneuver_proportion_range):
#         for k in range(n_mc):
#             index_arrays.append([float(dv), float(mp), k])

# index = pd.MultiIndex.from_tuples(
#     index_arrays, 
#     names=['delta_v_magnitude', 'maneuver_proportion', 'mc_iteration']
# )
# df = pd.DataFrame(
#     data=results.reshape(-1), 
#     index=index, 
#     columns=["times_found"]
# )
# df.to_csv('cache/times_found_random_sensor_dsn_20.csv')
# df
