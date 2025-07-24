import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype as typechecker

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.evaluate import evaluate_tracking_grid
from xradar_uq.measurement_systems import (DeepSpaceNetwork,
                                           single_sensor_tracking)
from xradar_uq.stochastic_filters import EnGMF


key = jax.random.key(42)

delta_v_range = jnp.linspace(0.001, 0.5, 20)
maneuver_proportion_range = jnp.linspace(0.0, 1.0, 10)

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()
stochastic_filter = EnGMF()

results_single = evaluate_tracking_grid(
    delta_v_range, maneuver_proportion_range, key,
    dynamical_system, measurement_system, stochastic_filter,
    single_sensor_tracking, mc_iterations = 10
)
df = pd.DataFrame(
    jnp.mean(results_single, axis=-1),
    index=((389703 / 382981) * delta_v_range),
    columns=maneuver_proportion_range
)
return df
