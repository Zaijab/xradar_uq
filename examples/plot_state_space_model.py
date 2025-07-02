"""
This is an example file to show you how to plot basic things in Python.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import RangeSensor
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)
key, subkey = jax.random.split(key)


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

# Example usage with your setup:
def visualize_filtering_state(dynamical_system, key, batch_size=100):
    """Example function showing how to use the visualization."""
    
    # Generate true state and ensembles
    true_state = dynamical_system.initial_state()
    
    # Generate posterior ensemble (after some filtering)
    posterior_ensemble = dynamical_system.generate(key, batch_size=batch_size, final_time=0.0)
    
    # Optionally generate a prior ensemble for comparison
    # This could be your initial ensemble before measurements
    subkey1, subkey2 = jax.random.split(key)
    prior_ensemble = dynamical_system.generate(subkey1, batch_size=batch_size, final_time=0.0)
    
    # Create the visualization
    fig = plot_3d_state_projections(
        true_state=true_state,
        posterior_ensemble=posterior_ensemble, 
        prior_ensemble=prior_ensemble,
        title_prefix="CR3BP State"
    )
    
    plt.show()
    return fig


dynamical_system = CR3BP()
measurement_system = RangeSensor()
stochastic_filter = EnGMF()
true_state = dynamical_system.initial_state()
prior_ensemble = dynamical_system.generate(subkey, final_time=0.0)
posterior_ensemble = dynamical_system.generate(subkey, final_time=0.0)

    

errors = []


total_time = 400
measurement_interval = 1.0

for i in range(400):
    key, subkey = jax.random.split(key)
    true_state = dynamical_system.flow(0.0, 10 / 400, true_state)
    prior_ensemble = eqx.filter_vmap(dynamical_system.flow, in_axes=(None, None, 0))(0.0, jnp.array(1.0), posterior_ensemble)
    posterior_ensemble = stochastic_filter.update(subkey, prior_ensemble, measurement_system(true_state), measurement_system)
    error = true_state - jnp.mean(posterior_ensemble, axis=0)
    errors.append(error)
    print(jnp.mean(error))
    # fig = plot_3d_state_projections(
    #     true_state=true_state,
    #     posterior_ensemble=posterior_ensemble, 
    #     prior_ensemble=prior_ensemble,
    #     title_prefix="CR3BP State"
    # )
    # plt.title(f"{jnp.mean(error)}")
    # plt.savefig(f"figures/filtering_loop/initialization_{i}.png")
    # plt.close()


rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
rmse



# import matplotlib.pyplot as plt


# # Extract position components
# positions = ys[:, :3]  # (100, 3) - x, y, z coordinates

# # 2D trajectory plot (x-y plane)
# plt.figure(figsize=(8, 6))
# plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
# plt.scatter(positions[0, 0], positions[0, 1], c='green', s=50, label='Start')
# plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, label='End')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('CR3BP Trajectory')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.axis('equal')
# plt.savefig('figures/filtering_loop/cr3bp_trajectory.png', dpi=300, bbox_inches='tight')
# plt.show()
