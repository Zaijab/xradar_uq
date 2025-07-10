"""
This is a funciton
RangeSensor r < rthresh
AnglesOnly uses r_0 < r < r_1
Radar uses r_0 < r < r_1
"""

import equinox as eqx
import jax.numpy as jnp


@eqx.filter_jit
def tracking_measurability(state,
                           predicted_state,
                           elevation_fov=jnp.deg2rad(5),
                           azimuth_fov=jnp.deg2rad(5),
                           range_fov=1):

    rho = jnp.linalg.norm(state[:3])
    rho_pred = jnp.linalg.norm(predicted_state[:3])
    elevation = jnp.arcsin(state[2] / rho)
    elevation_pred = jnp.arcsin(predicted_state[2] / rho_pred)
    azimuth = jnp.arctan2(state[1], state[0])
    azimuth_pred = jnp.arctan2(predicted_state[1], predicted_state[0])
    
    return ((jnp.abs(elevation - elevation_pred) <= elevation_fov / 2) &
            (jnp.abs(azimuth - azimuth_pred) <= azimuth_fov / 2))
