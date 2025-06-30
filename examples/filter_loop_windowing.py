import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.stochastic_filters import EnGMF


def tracking_measurability(state, predicted_state, elevation_fov=0.1, azimuth_fov=0.1, range_fov=0.05):
    rho = np.linalg.norm(state[:3])
    rho_pred = np.linalg.norm(predicted_state[:3])
    elevation = np.arcsin(state[2] / rho)
    elevation_pred = np.arcsin(predicted_state[2] / rho_pred)
    azimuth = np.arctan2(state[1], state[0])
    azimuth_pred = np.arctan2(predicted_state[1], predicted_state[0])
    
    return (np.abs(elevation - elevation_pred) <= elevation_fov / 2 and
            np.abs(azimuth - azimuth_pred) <= azimuth_fov / 2 and
            np.abs(rho - rho_pred) <= range_fov / 2)

def plot_tracking_fov(predicted_state, true_state, step, elevation_fov=0.1, azimuth_fov=0.1, range_fov=0.05):
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
key = jax.random.key(42)
key, subkey = jax.random.split(key)

true_state = dynamical_system.initial_state()
true_state = true_state.at[3:].add(jnp.array([0.25, 0.0, 0.0]))  # Add velocity perturbation
posterior_ensemble = dynamical_system.generate(subkey, final_time=0.0)


predicted_state = dynamical_system.initial_state()  # Nominal prediction

for i in range(10):
    plot_tracking_fov(predicted_state, true_state, i)
    key, subkey = jax.random.split(key)
    true_state = dynamical_system.flow(0.0, 1.0, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 1.0, posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)

