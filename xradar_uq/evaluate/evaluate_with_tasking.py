from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Int, Key, jaxtyped

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.statistics import generate_random_impulse_velocity
from xradar_uq.stochastic_filters import EnGMF, AbstractFilter


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def tracking_scan_step(
    carry: tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable, #[[Float[Array, "state_dim"], Float[Array, "batch_size state_dim"], Key[Array, ""], Float[Array, "batch_size state_dim"]], Bool[Array, ""]],
    time_range: float | Float[Array, ""],
    delta_v_magnitude: float | Float[Array, ""],
    maneuver_proportion: float | Float[Array, ""],
    random_impulse_velocity: Float[Array, "3"],
) -> tuple[
    tuple[Float[Array, "batch_size state_dim"], Float[Array, "state_dim"], Float[Array, ""], int | Int[Array, ""]],
    Bool[Array, ""]
]:
    posterior_ensemble, true_state, total_fuel, times_found = carry
    
    update_key, measurement_key, thrust_key, tracking_key = jax.random.split(key, 4)
    
    true_state_next = dynamical_system.flow(0.0, time_range, true_state)
    
    should_maneuver = jax.random.bernoulli(thrust_key, p=maneuver_proportion)
    has_fuel = total_fuel > 0
    do_maneuver = should_maneuver & has_fuel
    
    true_state_next = jnp.where(
        do_maneuver,
        true_state_next.at[3:].add(random_impulse_velocity),
        true_state_next
    )
    total_fuel_next = jnp.where(do_maneuver, total_fuel - delta_v_magnitude, total_fuel)
    
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow, in_axes=(None, None, 0))(0.0, time_range, posterior_ensemble)
    
    is_measurable = tracking_fn(true_state_next, prior_ensemble, tracking_key, posterior_ensemble,
                                dynamical_system, time_range, delta_v_magnitude)
    
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
    
    times_found_next = times_found + jnp.where(is_measurable, 1, 0)
    
    new_carry = (posterior_ensemble_next, true_state_next, total_fuel_next, times_found_next)
    return new_carry, is_measurable


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_single_case(
    delta_v_magnitude: float | Float[Array, ""],
    maneuver_proportion: float | Float[Array, ""],
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable[[Float[Array, "state_dim"], Float[Array, "batch_size state_dim"], Key[Array, ""]], Bool[Array, ""]],
    time_range: float | Float[Array, ""] = 0.242,
    measurement_time: int = 200,
    initial_fuel: float = 1.0,
) -> Float[Array, ""]:
    key, state_key, impulse_key = jax.random.split(key, 3)
    
    true_state = dynamical_system.initial_state()
    posterior_ensemble = dynamical_system.generate(state_key, batch_size=stochastic_filter.ensemble_size)
    random_impulse_velocity = generate_random_impulse_velocity(impulse_key, delta_v_magnitude)
    
    measurement_keys = jax.random.split(key, measurement_time)
    initial_carry = (posterior_ensemble, true_state, initial_fuel, 0)
    
    def scan_fn(carry, key):
        return tracking_scan_step(
            carry, key, dynamical_system, measurement_system, stochastic_filter,
            tracking_fn, time_range, delta_v_magnitude, maneuver_proportion, random_impulse_velocity
        )
    
    final_carry, _ = jax.lax.scan(scan_fn, initial_carry, measurement_keys)
    _, _, _, times_found = final_carry
    
    found_proportion = times_found / measurement_time
    return found_proportion


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def evaluate_tracking_grid(
    delta_v_range: Float[Array, "n_dv"],
    maneuver_proportion_range: Float[Array, "n_mp"],
    time_horrizon,
    key: Key[Array, ""],
    dynamical_system: CR3BP,
    measurement_system: AbstractMeasurementSystem,
    stochastic_filter: AbstractFilter,
    tracking_fn: Callable[[Float[Array, "state_dim"], Float[Array, "batch_size state_dim"], Key[Array, ""]], Bool[Array, ""]],
    mc_iterations: int = 1,
) -> Float[Array, "n_dv n_mp mc_iterations"]:
    n_dv, n_mp = len(delta_v_range), len(maneuver_proportion_range)
    
    mc_keys = jax.random.split(key, mc_iterations)
    
    def evaluate_mc_iteration(mc_key):
        dv_keys = jax.random.split(mc_key, n_dv)
        
        def evaluate_dv(dv_key, dv_val):
            mp_keys = jax.random.split(dv_key, n_mp)
            
            def evaluate_mp(mp_key, mp_val):
                return evaluate_tracking_single_case(
                    dv_val, mp_val, mp_key,
                    dynamical_system, measurement_system, stochastic_filter, tracking_fn, time_horrizon
                )
            
            return eqx.filter_vmap(evaluate_mp)(mp_keys, maneuver_proportion_range)
        
        return eqx.filter_vmap(evaluate_dv)(dv_keys, delta_v_range)
    
    results = eqx.filter_vmap(evaluate_mc_iteration)(mc_keys)
    return jnp.transpose(results, (1, 2, 0))
