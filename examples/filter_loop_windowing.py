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
# print(f"10 minutes real = {real_time_to_cr3bp(10*60):.6f} TU")  # ≈ 0.00160 TU
# print(f"2.5 hours real = {real_time_to_cr3bp(2.5*3600):.6f} TU")  # ≈ 0.024 TU
# print(f"NRHO period 1.3632 TU = {cr3bp_time_to_real(1.3632)/86400:.2f} days")  # ≈ 5.9 days

# def generate_measurement_schedule_cr3bp():
#     """Generate measurement times in CR3BP units"""
    
#     # NRHO period from Document 25
#     orbit_period_tu = 1.3632096570  # TU (about 5.9 real days)
    
#     # Tracklet parameters (convert to TU)
#     tracklet_duration_tu = real_time_to_cr3bp(2.5 * 3600)  # 2.5 hours → TU
#     measurement_interval_tu = real_time_to_cr3bp(10 * 60)   # 10 minutes → TU
#     gap_duration_tu = orbit_period_tu / 4  # Quarter orbit gap
    
#     print(f"Tracklet duration: {tracklet_duration_tu:.6f} TU")
#     print(f"Measurement interval: {measurement_interval_tu:.6f} TU") 
#     print(f"Gap between tracklets: {gap_duration_tu:.6f} TU")
    
#     schedule = []
#     total_time_tu = 5 * orbit_period_tu  # 5 orbits ≈ 30 days
    
#     current_time = 0
#     while current_time < total_time_tu:
#         # Start tracklet
#         tracklet_end = current_time + tracklet_duration_tu
        
#         # Measurements every 10 minutes during tracklet
#         while current_time < tracklet_end and current_time < total_time_tu:
#             schedule.append(current_time)
#             current_time += measurement_interval_tu
            
#         # Gap until next tracklet  
#         current_time = tracklet_end + gap_duration_tu
    
#     return jnp.array(schedule)

# This gives you measurement times in proper CR3BP units



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

# # How many measurements * time_range
measurement_time = 1000

# # We're tracking an object initially
# # I guess this is kinda like burn-in lol
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

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_state_projections(true_state, posterior_ensemble, prior_ensemble=None, 
                             figsize=(15, 10), title_prefix="State"):
    """
    Plot 3D position state in all 2D projections plus 3D view.
    
    Parameters:
    -----------
    true_state : Array, shape (6,)
        True state vector [x, y, z, vx, vy, vz]
    posterior_ensemble : Array, shape (N, 6) 
        Posterior ensemble of states
    prior_ensemble : Array, shape (N, 6), optional
        Prior ensemble of states
    figsize : tuple
        Figure size
    title_prefix : str
        Prefix for plot titles
    """
    
    # Extract position components (first 3 dimensions)
    true_pos = true_state[:3]
    post_pos = posterior_ensemble[:, :3]
    
    if prior_ensemble is not None:
        prior_pos = prior_ensemble[:, :3]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Define projection pairs and labels
    projections = [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'), (1, 2, 'Y', 'Z')]
    
    # Plot 2D projections
    for i, (dim1, dim2, label1, label2) in enumerate(projections):
        ax = fig.add_subplot(2, 2, i+1)
        
        # Plot prior if available
        if prior_ensemble is not None:
            ax.scatter(prior_pos[:, dim1], prior_pos[:, dim2], 
                      c='red', alpha=0.6, s=20, label='Prior')
        
        # Plot posterior
        ax.scatter(post_pos[:, dim1], post_pos[:, dim2], 
                  c='blue', alpha=0.7, s=20, label='Posterior')
        
        # Plot true state
        ax.scatter(true_pos[dim1], true_pos[dim2], 
                  c='green', s=100, marker='o', 
                  edgecolors='black', linewidth=2, label='True State')
        
        ax.set_xlabel(f'{label1} Position')
        ax.set_ylabel(f'{label2} Position')
        ax.set_title(f'{title_prefix}: {label1}-{label2} Projection')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Plot 3D view
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    
    # Plot prior if available
    if prior_ensemble is not None:
        ax3d.scatter(prior_pos[:, 0], prior_pos[:, 1], prior_pos[:, 2],
                    c='red', alpha=0.6, s=20, label='Prior')
    
    # Plot posterior
    ax3d.scatter(post_pos[:, 0], post_pos[:, 1], post_pos[:, 2],
                c='blue', alpha=0.7, s=20, label='Posterior')
    
    # Plot true state
    ax3d.scatter(true_pos[0], true_pos[1], true_pos[2],
                c='green', s=100, marker='o',
                edgecolors='black', linewidth=2, label='True State')
    
    ax3d.set_xlabel('X Position')
    ax3d.set_ylabel('Y Position')
    ax3d.set_zlabel('Z Position')
    ax3d.set_title(f'{title_prefix}: 3D View')
    ax3d.legend()
    
    plt.tight_layout()
    return fig

# plot_3d_state_projections(true_state, posterior_ensemble)
# plt.savefig("figures/filtering_loop/post_1000_tracking.png")

import pandas as pd

mc_iterations = 1
key, subkey = jax.random.split(key)
subkeys = jax.random.split(subkey, mc_iterations)
delta_v_range = np.logspace(-3, -1, 20)  # 20 different dV values
maneuver_proportion_range = np.linspace(0, 0.2, 10)
index = pd.MultiIndex.from_product(
    [delta_v_range, maneuver_proportion_range, range(mc_iterations)], 
    names=['delta_v_magnitude', 'maneuver_proportion', 'mc_iteration']
)
df = pd.DataFrame(index=index, columns=["times_found"])

# Plot:

# Detection rate vs dV magnitude
# Detection rate vs Frequency of maneuver
# Cumulative tracking performance over time


for delta_v_magnitude in jnp.logspace(-3, -1, 20):
    print(f"{delta_v_magnitude=}")
    for maneuver_proportion in maneuver_proportion_range:
        print(f"{maneuver_proportion=}")
        for mc_iteration_i, subkey in enumerate(subkeys):
            df.loc[(float(delta_v_magnitude), float(maneuver_proportion), mc_iteration_i), ("times_found")]
            total_fuel = 10.0

            true_state = jnp.load("cache/true_state_1000.npy")
            posterior_ensemble = jnp.load("cache/posterior_1000_window.npy")

            times_found = 0
            azimuth_key, elevation_key = jax.random.split(subkey)
            random_impulse_azimuth = jax.random.uniform(azimuth_key, minval=0, maxval=2 * jnp.pi)
            random_impulse_elevation = jax.random.uniform(elevation_key, minval=- jnp.pi / 2, maxval=jnp.pi / 2)

            vx = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.cos(random_impulse_azimuth)
            vy = delta_v_magnitude * jnp.cos(random_impulse_elevation) * jnp.sin(random_impulse_azimuth)
            vz = delta_v_magnitude * jnp.sin(random_impulse_elevation)

            random_impulse_velocity = jnp.array([vx, vy, vz])
            random_impulse_velocity = (delta_v_magnitude / jnp.linalg.norm(random_impulse_velocity)) * random_impulse_velocity

            for i in range(measurement_time):
                key, update_key, measurement_key, window_center_key, thrust_key = jax.random.split(key, 5)
                true_state = dynamical_system.flow(0.0, time_range, true_state)

                if jax.random.bernoulli(thrust_key, p=maneuver_proportion):
                    if total_fuel > 0:
                        key, subkey = jax.random.split(key)
                        total_fuel -= delta_v_magnitude
                        true_state = true_state.at[3:].add(random_impulse_velocity)


                prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, posterior_ensemble)
                predicted_state = jnp.mean(prior_ensemble, axis=0)

                if tracking_measurability(true_state, predicted_state):
                    times_found += 1
                    posterior_ensemble = stochastic_filter.update(update_key, prior_ensemble, measurement_system(true_state, measurement_key), measurement_system)
                else:
                    posterior_ensemble = prior_ensemble

            found_proportion = times_found / measurement_time
            print(found_proportion)
            df.loc[(float(delta_v_magnitude), float(maneuver_proportion), mc_iteration_i), ("times_found")] = found_proportion
