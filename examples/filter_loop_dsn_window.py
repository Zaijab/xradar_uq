# Setup (same as your original)
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped, Int, Bool

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar, tracking_measurability, DeepSpaceNetwork, AbstractMeasurementSystem
from xradar_uq.statistics import generate_random_impulse_velocity
from xradar_uq.stochastic_filters import EnGMF


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def tracking_scan_step(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: EnGMF,
    time_range: float,
    delta_v_magnitude: float | Float[Array, ""],
    maneuver_proportion: float | Float[Array, ""],
    random_impulse_velocity: Float[Array, "3"],
) -> tuple[
    tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]],
    bool | Bool[Array, ""]
]:
    posterior_ensemble, true_state, total_fuel, times_found = carry
    
    # Split keys for different random operations
    update_key, measurement_key, thrust_key = jax.random.split(key, 3)
    
    # Flow true state forward
    true_state_next = dynamical_system.flow(0.0, time_range, true_state)
    
    # Check for maneuver
    should_maneuver = jax.random.bernoulli(thrust_key, p=maneuver_proportion)
    has_fuel = total_fuel > 0
    do_maneuver = should_maneuver & has_fuel
    
    # Apply maneuver conditionally
    true_state_next = jnp.where(
        do_maneuver,
        true_state_next.at[3:].add(random_impulse_velocity),
        true_state_next
    )
    total_fuel_next = jnp.where(do_maneuver, total_fuel - delta_v_magnitude, total_fuel)
    
    # Flow ensemble forward
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)
    predicted_state = jnp.mean(prior_ensemble, axis=0)
    
    # Check tracking measurability
    is_measurable = tracking_measurability(true_state_next, predicted_state)
    
    # Update ensemble conditionally
    posterior_ensemble_next = jnp.where(
        is_measurable,
        stochastic_filter.update(
            update_key, 
            prior_ensemble, 
            measurement_system(true_state_next, measurement_key), 
            measurement_system
        ),
        prior_ensemble
    )
    
    # Update times found counter
    times_found_next = times_found + jnp.where(is_measurable, 1, 0)
    
    new_carry = (posterior_ensemble_next, true_state_next, total_fuel_next, times_found_next)
    return new_carry, is_measurable


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_single_case(
    delta_v_magnitude: float  | Float[Array, ""],
    maneuver_proportion: float  | Float[Array, ""],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: EnGMF,
    time_range: float = 0.242,
    measurement_time: int = 1000,
    initial_fuel: float = 10.0,
) -> float | Float[Array, ""]:
    # Load cached states
    # true_state = jnp.load("cache/true_state_1000.npy")
    # posterior_ensemble = jnp.load("cache/posterior_1000_window.npy")
    key, subkey = jax.random.split(key)
    true_state = dynamical_system.initial_state()
    posterior_ensemble = dynamical_system.generate(subkey)
    
    # Generate random impulse velocity
    key, subkey = jax.random.split(key)
    random_impulse_velocity = generate_random_impulse_velocity(subkey, delta_v_magnitude)
    
    # Generate keys for scan
    keys = jax.random.split(key, measurement_time)
    
    # Initial carry state
    initial_carry = (posterior_ensemble, true_state, initial_fuel, 0)
    
    # Create partial function for scan
    scan_fn = jax.tree_util.Partial(
        tracking_scan_step,
        dynamical_system=dynamical_system,
        measurement_system=measurement_system,
        stochastic_filter=stochastic_filter,
        time_range=time_range,
        delta_v_magnitude=delta_v_magnitude,
        maneuver_proportion=maneuver_proportion,
        random_impulse_velocity=random_impulse_velocity,
    )
    
    # Run scan
    final_carry, detections = jax.lax.scan(scan_fn, initial_carry, keys)
    _, _, _, times_found = final_carry
    
    # Calculate proportion
    found_proportion = times_found / measurement_time
    jax.debug.print("{}", found_proportion)
    return found_proportion


# For vectorizing over parameter ranges
@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_grid(
    delta_v_range: Float[Array, "n_dv"],
    maneuver_proportion_range: Float[Array, "n_mp"],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: EnGMF,
    mc_iterations: int = 1,
) -> Float[Array, "n_dv n_mp mc_iterations"]:
    n_dv, n_mp = len(delta_v_range), len(maneuver_proportion_range)
    
    # Create all parameter combinations
    dv_flat = jnp.repeat(delta_v_range, n_mp * mc_iterations)
    mp_flat = jnp.tile(jnp.repeat(maneuver_proportion_range, mc_iterations), n_dv)
    
    # Generate keys for all combinations
    keys = jax.random.split(key, n_dv * n_mp * mc_iterations)
    
    # Vectorize over flattened arrays
    vectorized_fn = jax.vmap(
        evaluate_tracking_single_case, 
        in_axes=(0, 0, 0, None, None, None)
    )
    
    # Run vectorized computation
    results_flat = vectorized_fn(
        dv_flat, mp_flat, keys,
        dynamical_system, measurement_system, stochastic_filter
    )
    
    # Reshape to desired output format
    results = results_flat.reshape(n_dv, n_mp, mc_iterations)
    return results

# Add this before calling evaluate_tracking_grid
compiled_single_case = jax.jit(evaluate_tracking_single_case)

# Then modify the vectorized_fn line to:
vectorized_fn = jax.vmap(
    compiled_single_case, 
    in_axes=(0, 0, 0, None, None, None)
)

# Setup (same as your original)
dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()

# Define parameter ranges
# delta_v_range = jnp.logspace(-3, -1, 20)
# maneuver_proportion_range = jnp.linspace(0, 0.2, 20)

delta_v_range = jnp.logspace(-3, 0, 20)
maneuver_proportion_range = jnp.linspace(0, 0.5, 50)


# Run optimized computation
key = jax.random.key(42)
results = evaluate_tracking_grid(
    delta_v_range, 
    maneuver_proportion_range, 
    key,
    dynamical_system, 
    measurement_system, 
    stochastic_filter,
    mc_iterations=1
)

# Convert to DataFrame format matching your original
import pandas as pd

n_dv, n_mp, n_mc = results.shape
index_arrays = []
for i, dv in enumerate(delta_v_range):
    for j, mp in enumerate(maneuver_proportion_range):
        for k in range(n_mc):
            index_arrays.append([float(dv), float(mp), k])

index = pd.MultiIndex.from_tuples(
    index_arrays, 
    names=['delta_v_magnitude', 'maneuver_proportion', 'mc_iteration']
)
df = pd.DataFrame(
    data=results.reshape(-1), 
    index=index, 
    columns=["times_found"]
)
df
