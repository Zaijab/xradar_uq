
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from xradar_uq.measurement_systems import tracking_measurability
from xradar_uq.statistics import silverman_kde_estimate


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_random_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], 
    key: Key[Array, ""],
) -> Float[Array, "state_dim"]:
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jax.random.choice(key, pdf_values.shape[0])]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def find_optimal_second_sensor_state(
    prior_ensemble: Float[Array, "batch_size state_dim"], 
) -> Float[Array, "state_dim"]:
    constraint_distance = 5.0
    boundary_resolution = 50
    t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
    side = jnp.floor(t).astype(int)
    local_t = t - side
    azimuth_constraint = jnp.where(side == 0, constraint_distance, jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t, jnp.where(side == 2, -constraint_distance, -constraint_distance + 2*constraint_distance*local_t)))
    elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t, jnp.where(side == 1, constraint_distance, jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t, -constraint_distance)))
    
    az_rad, el_rad = jnp.deg2rad(azimuth_constraint), jnp.deg2rad(elevation_constraint)
    points_3d = jnp.stack([jnp.cos(el_rad)*jnp.cos(az_rad), jnp.cos(el_rad)*jnp.sin(az_rad), jnp.sin(el_rad)], axis=1)
    
    gmm = silverman_kde_estimate(prior_ensemble[:, :3])
    pdf_values = eqx.filter_vmap(gmm.pdf)(points_3d)
    optimal_position = points_3d[jnp.argmax(pdf_values)]
    
    return jnp.concatenate([optimal_position, jnp.zeros(3)])


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def dual_sensor_tracking_random(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    second_predicted_state = find_random_second_sensor_state(prior_ensemble, key)
    return tracking_measurability(true_state, predicted_state) | tracking_measurability(true_state, second_predicted_state)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def dual_sensor_tracking_optimal(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    second_predicted_state = find_optimal_second_sensor_state(prior_ensemble)
    return tracking_measurability(true_state, predicted_state) | tracking_measurability(true_state, second_predicted_state)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def single_sensor_tracking(
    true_state: Float[Array, "state_dim"],
    prior_ensemble: Float[Array, "batch_size state_dim"],
    key: Key[Array, ""],
) -> Bool[Array, ""]:
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    return tracking_measurability(true_state, predicted_state)
