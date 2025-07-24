import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def place_non_overlapping_sensors(
    key: jax.Array,
    convex_hull_vertices: Float[Array, "n_vertices 2"],
    n_sensors: int,
    sensor_half_width: float = 2.5,
) -> Float[Array, "n_sensors 2"]:
    exclusion_radius = 2 * sensor_half_width
    hull_center = jnp.mean(convex_hull_vertices, axis=0)
    max_hull_radius = jnp.max(jnp.linalg.norm(convex_hull_vertices - hull_center, axis=1))
    
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key = carry
        key, subkey = jax.random.split(key)
        
        candidates = hull_center[None, :] + jax.random.uniform(
            subkey, (1000, 2), minval=-max_hull_radius, maxval=max_hull_radius
        )
        
        distances = jnp.linalg.norm(candidates[:, None, :] - positions[None, :, :], axis=2)
        valid_position_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_position_distances, axis=1)
        valid_mask = min_distances > exclusion_radius
        first_valid_idx = jnp.argmax(valid_mask)
        new_position = candidates[first_valid_idx]
        
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        return (updated_positions, updated_mask, key)
    
    initial_carry = (sensor_positions, placement_mask, key)
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, initial_carry)
    
    assert final_positions.shape == (n_sensors, 2)
    return final_positions

key, subkey = jax.random.split(key)
place_non_overlapping_sensors(subkey, my_hull.vertices, 3)



@jaxtyped(typechecker=typechecker)
def visualize_hull_and_sensors(hull_vertices: Float[Array, "n_vertices 2"], 
                              sensor_positions: Float[Array, "n_sensors 2"]) -> None:
    os.makedirs("figures/sensor_placement", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    hull_array = jnp.array(hull_vertices)
    closed_hull = jnp.vstack([hull_array, hull_array[0]])
    ax.plot(closed_hull[:, 0], closed_hull[:, 1], 'b-', linewidth=2, label='Convex Hull')
    ax.fill(closed_hull[:, 0], closed_hull[:, 1], alpha=0.2, color='blue')
    
    sensor_half_width = jnp.deg2rad(2.5)
    for i, pos in enumerate(sensor_positions):
        rect = patches.Rectangle((pos[0] - sensor_half_width, pos[1] - sensor_half_width), 
                                2 * sensor_half_width, 2 * sensor_half_width,
                                facecolor='pink', alpha=0.5, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
              color='red', s=100, marker='o', label='Non-overlapping Sensors', zorder=5)
    
    ax.set_xlabel('Azimuth'); ax.set_ylabel('Elevation')
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout(); plt.savefig("figures/sensor_placement/hull_sensors.png", dpi=150)

hull_vertices_array = my_hull.vertices_array()
key, subkey = jax.random.split(key)
sensor_positions = place_non_overlapping_sensors(subkey, hull_vertices_array, 3)
assert hull_vertices_array.shape[1] == 2 and sensor_positions.shape[1] == 2
visualize_hull_and_sensors(hull_vertices_array, sensor_positions)
