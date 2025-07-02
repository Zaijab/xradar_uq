import os

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import AbstractMeasurementSystem, Radar
from xradar_uq.stochastic_filters import EnGMF

# RangeSensor r < rthresh
# AnglesOnly uses r_0 < r < r_1
# Radar uses r_0 < r < r_1
def tracking_measurability(state, predicted_state, elevation_fov=(5 * jnp.pi / 180), azimuth_fov=(5 * jnp.pi / 180), range_fov=1):

    rho = np.linalg.norm(state[:3])
    rho_pred = np.linalg.norm(predicted_state[:3])
    elevation = np.arcsin(state[2] / rho)
    elevation_pred = np.arcsin(predicted_state[2] / rho_pred)
    azimuth = np.arctan2(state[1], state[0])
    azimuth_pred = np.arctan2(predicted_state[1], predicted_state[0])

    # print(np.abs(elevation - elevation_pred), elevation_fov / 2, np.abs(elevation - elevation_pred) <= elevation_fov / 2)
    # print(np.abs(azimuth - azimuth_pred), azimuth_fov / 2, np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2)
    # print(np.abs(rho - rho_pred), range_fov / 2, np.abs(rho - rho_pred) <= range_fov / 2)
    
    return (np.abs(elevation - elevation_pred) <= elevation_fov / 2 and
            np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2)


def plot_tracking_fov(predicted_state, true_state, step, elevation_fov=(5 * jnp.pi / 180), azimuth_fov=(5 * jnp.pi / 180), range_fov=1.0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Predicted state spherical coordinates
    rho_pred = np.linalg.norm(predicted_state[:3])
    theta_pred = np.arcsin(predicted_state[2] / rho_pred)
    phi_pred = np.arctan2(predicted_state[1], predicted_state[0])
    
    # Box corners in spherical coordinates
    theta_range = [theta_pred - elevation_fov/2, theta_pred + elevation_fov/2]
    phi_range = [phi_pred - azimuth_fov/2, phi_pred + azimuth_fov/2]
    rho_range = [rho_pred - range_fov/2, rho_pred + range_fov/2]
    
    # Generate 8 corners of the box
    corners = []
    for theta in theta_range:
        for phi in phi_range:
            for rho in rho_range:
                x = rho * np.cos(theta) * np.cos(phi)
                y = rho * np.cos(theta) * np.sin(phi)
                z = rho * np.sin(theta)
                corners.append([x, y, z])
    corners = np.array(corners)
    
    # Draw wireframe edges
    edges = [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),(4,5),(4,6),(5,7),(6,7)]
    for edge in edges:
        points = corners[list(edge)]
        ax.plot(points[:,0], points[:,1], points[:,2], 'b-', alpha=0.6)
    
    # Check if true state is measurable
    is_measurable = tracking_measurability(true_state, predicted_state, elevation_fov, azimuth_fov, range_fov)
    true_color = 'green' if is_measurable else 'red'
    
    ax.scatter(*predicted_state[:3], c='blue', s=100, label='Predicted State')
    ax.scatter(*true_state[:3], c=true_color, s=100, label=f'True State ({"Measurable" if is_measurable else "Not Measurable"})')
    
    # Dynamic axis limits to show both states and wireframe
    all_points = np.vstack([corners, predicted_state[:3].reshape(1,-1), true_state[:3].reshape(1,-1)])
    margin = 0.02
    ax.set_xlim(all_points[:,0].min() - margin, all_points[:,0].max() + margin)
    ax.set_ylim(all_points[:,1].min() - margin, all_points[:,1].max() + margin)
    ax.set_zlim(all_points[:,2].min() - margin, all_points[:,2].max() + margin)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); plt.title(f'Optical Sensor FOV - Step {step}')
    os.makedirs('figures', exist_ok=True)
    plt.savefig(f'figures/fov_step_{step:02d}.png', dpi=150, bbox_inches='tight')
    plt.close()

def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    components = distrax.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covs)
    return distrax.MixtureSameFamily(
        mixture_distribution=distrax.Categorical(probs=weights),
        components_distribution=components
    )

    
# Example usage with custody loss
dynamical_system = CR3BP()
stochastic_filter = EnGMF(silverman_bandwidth_scaling = 1.0)

measurement_system = Radar()

key = jax.random.key(42)
key, subkey = jax.random.split(key)

true_state = dynamical_system.initial_state()
posterior_ensemble = dynamical_system.generate(subkey)

# Real time to CR3BP time conversion
TU_seconds = 375730  # Time unit in seconds
TU_days = TU_seconds / (24 * 3600)  # ≈ 4.35 days

def real_time_to_cr3bp(real_seconds):
    return real_seconds / TU_seconds

def cr3bp_time_to_real(cr3bp_time):
    return cr3bp_time * TU_seconds

# Examples from the papers:
print(f"10 minutes real = {real_time_to_cr3bp(10*60):.6f} TU")  # ≈ 0.00160 TU
print(f"2.5 hours real = {real_time_to_cr3bp(2.5*3600):.6f} TU")  # ≈ 0.024 TU
print(f"NRHO period 1.3632 TU = {cr3bp_time_to_real(1.3632)/86400:.2f} days")  # ≈ 5.9 days

def generate_measurement_schedule_cr3bp():
    """Generate measurement times in CR3BP units"""
    
    # NRHO period from Document 25
    orbit_period_tu = 1.3632096570  # TU (about 5.9 real days)
    
    # Tracklet parameters (convert to TU)
    tracklet_duration_tu = real_time_to_cr3bp(2.5 * 3600)  # 2.5 hours → TU
    measurement_interval_tu = real_time_to_cr3bp(10 * 60)   # 10 minutes → TU
    gap_duration_tu = orbit_period_tu / 4  # Quarter orbit gap
    
    print(f"Tracklet duration: {tracklet_duration_tu:.6f} TU")
    print(f"Measurement interval: {measurement_interval_tu:.6f} TU") 
    print(f"Gap between tracklets: {gap_duration_tu:.6f} TU")
    
    schedule = []
    total_time_tu = 5 * orbit_period_tu  # 5 orbits ≈ 30 days
    
    current_time = 0
    while current_time < total_time_tu:
        # Start tracklet
        tracklet_end = current_time + tracklet_duration_tu
        
        # Measurements every 10 minutes during tracklet
        while current_time < tracklet_end and current_time < total_time_tu:
            schedule.append(current_time)
            current_time += measurement_interval_tu
            
        # Gap until next tracklet  
        current_time = tracklet_end + gap_duration_tu
    
    return jnp.array(schedule)

# This gives you measurement times in proper CR3BP units


@jaxtyped(typechecker=typechecker)
def get_kde_mode_center(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    n_candidates: int = 1000,
) -> Float[Array, "state_dim"]:
    """Find the mode (highest density point) of the KDE"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Sample candidates from the KDE itself
    candidates = kde.sample(seed=key, sample_shape=(n_candidates,))
    
    # Evaluate probability density at each candidate
    log_probs = kde.log_prob(candidates)
    
    # Return the candidate with highest density
    max_idx = jnp.argmax(log_probs)
    return candidates[max_idx]

@jaxtyped(typechecker=typechecker)
def get_uncertainty_weighted_kde_center(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    n_samples: int = 1000,
) -> Float[Array, "state_dim"]:
    """Weight pointing toward regions of high local uncertainty"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Sample from the KDE
    samples = kde.sample(seed=key, sample_shape=(n_samples,))
    
    # Compute local uncertainty (inverse of density)
    densities = kde.prob(samples)
    uncertainty_weights = 1.0 / (densities + 1e-8)  # Add small epsilon for numerical stability
    
    # Normalize weights
    weights = uncertainty_weights / jnp.sum(uncertainty_weights)
    
    # Return uncertainty-weighted centroid
    return jnp.sum(weights[:, None] * samples, axis=0)

@jaxtyped(typechecker=typechecker)
def get_kde_mode_optimized(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    n_starts: int = 10,
    learning_rate: float = 0.01,
    n_steps: int = 100
) -> Float[Array, "state_dim"]:
    """Find KDE mode using gradient ascent"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Multiple random starts to avoid local maxima
    keys = jax.random.split(key, n_starts)
    initial_points = jax.vmap(lambda k: kde.sample(seed=k))(keys)
    
    def gradient_ascent_step(point):
        grad_fn = jax.grad(kde.log_prob)
        return point + learning_rate * grad_fn(point)
    
    def find_mode_from_start(start_point):
        def step_fn(point, _):
            new_point = gradient_ascent_step(point)
            return new_point, new_point
        
        final_point, _ = jax.lax.scan(step_fn, start_point, None, length=n_steps)
        return final_point, kde.log_prob(final_point)
    
    # Find mode from each starting point
    final_points, final_probs = jax.vmap(find_mode_from_start)(initial_points)
    
    # Return the best one
    best_idx = jnp.argmax(final_probs)
    return final_points[best_idx]

@jaxtyped(typechecker=typechecker)
def get_information_optimal_center(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    measurement_system: AbstractMeasurementSystem,
    n_candidates: int = 500,
) -> Float[Array, "state_dim"]:
    """Point to maximize expected Fisher Information Gain"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Sample candidate pointing directions
    candidates = kde.sample(seed=key, sample_shape=(n_candidates,))
    
    def expected_information_gain(pointing_state):
        # Fisher Information at this pointing direction
        H = jax.jacfwd(measurement_system)(pointing_state)
        fisher_info = jnp.trace(H.T @ jnp.linalg.solve(measurement_system.covariance, H))
        
        # Weight by probability density
        prob_density = kde.prob(pointing_state)
        
        return fisher_info * prob_density
    
    # Compute expected information for each candidate
    expected_info = jax.vmap(expected_information_gain)(candidates)
    
    # Return candidate with maximum expected information
    best_idx = jnp.argmax(expected_info)
    return candidates[best_idx]

@jaxtyped(typechecker=typechecker)
def get_entropy_reduction_center(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    measurement_system: AbstractMeasurementSystem,
    n_candidates: int = 500,
) -> Float[Array, "state_dim"]:
    """Point to maximize entropy reduction"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Current entropy of the distribution
    sample_points = kde.sample(seed=key, sample_shape=(1000,))
    current_entropy = -jnp.mean(kde.log_prob(sample_points))
    
    # Sample candidate pointing directions
    candidates = kde.sample(seed=key, sample_shape=(n_candidates,))
    
    def entropy_reduction_score(pointing_state):
        # Simulate making a measurement at this pointing direction
        H = jax.jacfwd(measurement_system)(pointing_state)
        measurement_cov = H @ jnp.cov(ensemble.T) @ H.T + measurement_system.covariance
        
        # Approximate entropy reduction (inverse of measurement uncertainty)
        return -jnp.log(jnp.linalg.det(measurement_cov))
    
    scores = jax.vmap(entropy_reduction_score)(candidates)
    best_idx = jnp.argmax(scores)
    return candidates[best_idx]

@jaxtyped(typechecker=typechecker)
def get_quantile_based_center(
    key: Key[Array, ""],
    ensemble: Float[Array, "n_components state_dim"],
    quantile: float = 0.5,  # 0.5 = median
    n_samples: int = 1000,
) -> Float[Array, "state_dim"]:
    """Use quantiles of the KDE distribution"""
    
    kde = silverman_kde_estimate(ensemble)
    
    # Sample from KDE
    samples = kde.sample(seed=key, sample_shape=(n_samples,))
    
    # Compute quantiles along each dimension
    return jnp.quantile(samples, quantile, axis=0)

# We were tracking an object

methods = [
    # predicted_state = jnp.mean(prior_ensemble, axis=0) # 0.955: 0.016

    # predicted_state = get_kde_mode_center(window_center_key, prior_ensemble) # 0.946
    # predicted_state = get_information_optimal_center(window_center_key, prior_ensemble, measurement_system) # 0.943
    # predicted_state = get_entropy_reduction_center(window_center_key, prior_ensemble, measurement_system) # 0.882
    # predicted_state = get_quantile_based_center(window_center_key, prior_ensemble) # 0.956
    # predicted_state = get_uncertainty_weighted_kde_center(window_center_key, prior_ensemble) # 0.919
]
times_found = 0

# TU: 0.242 is approx 1 Day
time_range = 0.242

# How many measurements * time_range
measurement_time = 1000

# We're tracking an object initially
# I guess this is kinda like burn-in lol
# for i in range(measurement_time):
#     print(times_found, i)
#     key, update_key, measurement_key, window_center_key = jax.random.split(key, 4)
#     true_state = dynamical_system.flow(0.0, time_range, true_state)
#     prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)
#     predicted_state = jnp.mean(prior_ensemble, axis=0)
    
#     if tracking_measurability(true_state, predicted_state):
#         times_found += 1
#         posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
#     else:
#         posterior_ensemble = prior_ensemble

# print(times_found / measurement_time)

# Constant Thrust Impulse Velocity random initialization

# mc_iterations = 10
# total_fuel = 10
# delta_v_magnitude = 1
# key, subkey = jax.random.split(key)
# subkeys = jax.random.split(subkey, 10)
# for subkey in subkeys:
#     azimuth_key, elevation_key = jax.random.split(subkey)
#     random_impulse_azimuth = jax.random.uniform(azimuth_key, minval=0, maxval=2 * jnp.pi)
#     random_impulse_elevation = jax.random.uniform(azimuth_key, minval=- jnp.pi / 2, maxval=jnp.pi / 2)

#     vx = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.cos(random_impulse_azimuth)
#     vy = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.sin(random_impulse_azimuth)
#     vz = delta_v_magnitude * jnp.sin(random_impulse_elevation)

#     random_impulse_velocity = jnp.array([vx, vy, vz])
#     random_impulse_velocity = (1e-5 / jnp.linalg.norm(random_impulse_velocity)) * random_impulse_velocity

    
#     for i in range(measurement_time):
#         print(times_found, i)
#         key, update_key, measurement_key, window_center_key = jax.random.split(key, 4)
#         true_state = dynamical_system.flow(0.0, time_range, true_state)

#         if jax.random.bernoulli(subkey, p=0.1):
#             key, subkey = jax.random.split(key)
#             print(random_impulse_velocity)
#             true_state = true_state.at[3:].add(random_impulse_velocity)
        
        
#         prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)
#         predicted_state = jnp.mean(prior_ensemble, axis=0)

#         if tracking_measurability(true_state, predicted_state):
#             times_found += 1
#             posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
#         else:
#             posterior_ensemble = prior_ensemble

#     # print(times_found / measurement_time)



# # Suddenly it maneuvers
# Plot total delta V vs frequency of detection

