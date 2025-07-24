def test_reachable_sets():
    base_key = jax.random.key(42)
    system_key, thrust_key = jax.random.split(base_key)
    
    dynamical_system = CR3BP()
    batch_size = 50
    posterior_ensemble = dynamical_system.generate(system_key, batch_size=batch_size)


    delta_v_magnitude = 2.0
    time_horrizon = 0.242
    num_simulations = 100

    hull = reachable_set(thrust_key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    angle_hull = angular_reachable_set(thrust_key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)

    return hull, angle_hull

hull, angle_hull = test_reachable_sets()

