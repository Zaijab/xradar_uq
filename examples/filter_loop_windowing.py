import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF


def tracking_measurability(state, predicted_state, elevation_fov=(5 * jnp.pi / 180), azimuth_fov=(5 * jnp.pi / 180), range_fov=1):

    rho = np.linalg.norm(state[:3])
    rho_pred = np.linalg.norm(predicted_state[:3])
    elevation = np.arcsin(state[2] / rho)
    elevation_pred = np.arcsin(predicted_state[2] / rho_pred)
    azimuth = np.arctan2(state[1], state[0])
    azimuth_pred = np.arctan2(predicted_state[1], predicted_state[0])

    print(np.abs(elevation - elevation_pred), elevation_fov / 2, np.abs(elevation - elevation_pred) <= elevation_fov / 2)
    print(np.abs(azimuth - azimuth_pred), azimuth_fov / 2, np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2)
    print(np.abs(rho - rho_pred), range_fov / 2, np.abs(rho - rho_pred) <= range_fov / 2)
    
    return (np.abs(elevation - elevation_pred) <= elevation_fov / 2 and
            np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2 and
            np.abs(rho - rho_pred) <= range_fov / 2)

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

# Example usage with custody loss
dynamical_system = CR3BP()
stochastic_filter = EnGMF()
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
measurement_times_tu = generate_measurement_schedule_cr3bp()

def get_optimal_pointing_direction(posterior_ensemble, measurement_system):
    """
    Determine optimal sensor pointing direction to maximize information gain
    """
    
    # Method 1: Simple ensemble mean (your current approach)
    mean_prediction = jnp.mean(posterior_ensemble, axis=0)
    
    # Method 2: Fisher Information Gain weighted (from papers)
    # Point toward direction that maximizes expected information
    
    # Compute measurement Jacobian for each ensemble member
    jacobians = jax.vmap(jax.jacfwd(measurement_system))(posterior_ensemble)
    ensemble_cov = jnp.cov(posterior_ensemble.T)
    
    # Fisher Information Matrix for each pointing direction
    fisher_info = jax.vmap(lambda H: H.T @ jnp.linalg.inv(measurement_system.covariance) @ H)(jacobians)
    
    # Weight ensemble members by their Fisher information
    weights = jnp.array([jnp.trace(fi @ ensemble_cov) for fi in fisher_info])
    weights = weights / jnp.sum(weights)
    
    # Optimal pointing is Fisher-weighted mean
    optimal_pointing = jnp.sum(weights[:, None] * posterior_ensemble, axis=0)
    
    return optimal_pointing

# Usage in your loop:


# We were tracking an object
for i in range(10):
    print(i)
    key, update_key, measurement_key = jax.random.split(key, 3)
    true_state = dynamical_system.flow(0.0, 1.0, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
    # predicted_state = jnp.mean(posterior_ensemble, axis=0)
    predicted_state = get_optimal_pointing_direction(prior_ensemble, measurement_system)
    plot_tracking_fov(predicted_state, true_state, i)


# # Suddenly it maneuvers
# true_state = true_state.at[3:].add(jnp.array([0.25, 0.0, 0.0]))

# # Now we lost it
# for i in range(10):
#     key, update_key, measurement_key = jax.random.split(key, 3)
#     true_state = dynamical_system.flow(0.0, 1.0, true_state)
#     prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, posterior_ensemble)
#     posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
#     predicted_state = jnp.mean(posterior_ensemble, axis=0)
#     plot_tracking_fov(predicted_state, true_state, i)
