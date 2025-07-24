def test_reachable_sets():
    base_key = jax.random.key(42)
    system_key, thrust_key = jax.random.split(base_key)
    
    dynamical_system = CR3BP()
    batch_size = 50
    posterior_ensemble = dynamical_system.generate(system_key, batch_size=batch_size)
    assert posterior_ensemble.shape == (50, 6)

    
    delta_v_magnitude = 2.0
    time_horizon = 0.242
    num_simulations = 100

    thrust_ensemble = simulate_thrust(thrust_key, posterior_ensemble, num_simulations, delta_v_magnitude)
    thrust_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horizon, thrust_ensemble)
    hull = ZConvexHull.from_points(thrust_ensemble)

    angles = AnglesOnly()
    ensemble_angles = eqx.filter_vmap(angles)(thrust_ensemble)
    angle_hull = angular_convex_hull(ensemble_angles)
    
    return hull, angle_hull

hull, angle_hull = test_reachable_sets()
