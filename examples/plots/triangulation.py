import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from beartype import beartype as typechecker
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.geometry import angular_reachable_set, fan_triangulate, barycentric_subdivision
from xradar_uq.measurement_systems import AnglesOnly, DeepSpaceNetwork
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.statistics import silverman_kde_estimate, GMM
from typing import Callable
from jax.tree_util import Partial
from jaxtyping import jaxtyped, Float, Array, Key, Bool


dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()
angles = AnglesOnly()

key = jax.random.key(42)
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey)

delta_v_magnitude = 2.0
time_horrizon = 10*0.242
num_simulations = 100

n_directions = 50
angle_hull = angular_reachable_set(subkey, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system, n_directions)

@eqx.filter_jit
def order_vertices_cyclically(vertices):
    centroid = jnp.mean(vertices, axis=0)
    relative_positions = vertices - centroid
    angles = jnp.arctan2(relative_positions[:, 1], relative_positions[:, 0])
    sorted_indices = jnp.argsort(angles)
    return vertices[sorted_indices]

@eqx.filter_jit
def fan_triangulate_ordered(vertices):
    centroid = jnp.mean(vertices, axis=0)
    n_vertices = vertices.shape[0]
    triangles = jnp.zeros((n_vertices, 3, 2))
    triangles = triangles.at[:, 0, :].set(centroid)
    triangles = triangles.at[:, 1, :].set(vertices)
    triangles = triangles.at[:, 2, :].set(jnp.roll(vertices, -1, axis=0))
    return triangles

figures_dir = Path("figures/angular_reachable_set")
figures_dir.mkdir(parents=True, exist_ok=True)

ordered_vertices = order_vertices_cyclically(angle_hull.vertices)
assert ordered_vertices.shape[1] == 2
initial_triangles = fan_triangulate_ordered(ordered_vertices)

refined_triangles = initial_triangles

num_subdivisions = 3
for _ in range(num_subdivisions):
    refined_triangles = barycentric_subdivision(
        refined_triangles
    )
    
assert refined_triangles.shape[2] == 2

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

ax1.plot(jnp.concatenate([ordered_vertices[:, 0], ordered_vertices[0:1, 0]]), 
         jnp.concatenate([ordered_vertices[:, 1], ordered_vertices[0:1, 1]]), 'b-', linewidth=2)
ax1.scatter(ordered_vertices[:, 0], ordered_vertices[:, 1], c='red', s=50, zorder=5)
ax1.set_title("Hull Boundary")
ax1.grid(True, alpha=0.3)

for triangle in initial_triangles:
    triangle_closed = jnp.concatenate([triangle, triangle[0:1]], axis=0)
    ax2.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'b-', alpha=0.7, linewidth=1)
ax2.scatter(ordered_vertices[:, 0], ordered_vertices[:, 1], c='red', s=30, zorder=5)
ax2.set_title("Fan Triangulation")
ax2.grid(True, alpha=0.3)

for triangle in refined_triangles:
    triangle_closed = jnp.concatenate([triangle, triangle[0:1]], axis=0)
    ax3.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'g-', alpha=0.5, linewidth=0.5)
ax3.set_title("Barycentric Subdivision")
ax3.grid(True, alpha=0.3)

all_vertices = jnp.unique(refined_triangles.reshape(-1, 2), axis=0)
kde_model = silverman_kde_estimate(posterior_ensemble)
kde_values = jnp.linalg.norm(all_vertices - jnp.mean(all_vertices, axis=0), axis=1)
scatter = ax4.scatter(all_vertices[:, 0], all_vertices[:, 1], c=kde_values, cmap='viridis', s=30)
plt.colorbar(scatter, ax=ax4, label='Distance from Centroid')
ax4.set_title("Refined Vertices")
ax4.grid(True, alpha=0.3)


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    finite_mask = jnp.isfinite(candidates).all(axis=1)
    
    valid_candidates = jnp.where(finite_mask[:, None], candidates, 0.0)
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(valid_candidates)
    masked_weights = jnp.where(finite_mask, pdf_weights, -jnp.inf)
    optimal_idx = jnp.argmax(masked_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position

@eqx.filter_jit
def place_sensors(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"],
    n_sensors: int, exclusion_radius: float, selection: Callable
) -> Float[Array, "n_sensors 2"]:
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_distances, axis=1)
        valid_candidate_mask = min_distances > exclusion_radius
        
        valid_candidates = jnp.where(valid_candidate_mask[:, None], candidates, jnp.inf)
        new_position = selection(subkey, valid_candidates)
        
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        
        return (updated_positions, updated_mask, key_state)
    
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, (sensor_positions, placement_mask, key))
    return final_positions

# Usage:
exclusion_radius = jnp.deg2rad(5.0)
jax.debug.print("Exclusion radius: {r}, Candidate range: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]", 
                r=exclusion_radius, 
                xmin=jnp.min(all_vertices[:, 0]), xmax=jnp.max(all_vertices[:, 0]),
                ymin=jnp.min(all_vertices[:, 1]), ymax=jnp.max(all_vertices[:, 1]))

gmm_pdf_selection = Partial(select_pdf_weighted_candidate, gmm=kde_model)
sensor_positions = place_sensors(key, all_vertices, 3, exclusion_radius, gmm_pdf_selection)
print(jnp.max(jnp.abs(sensor_positions[:, None, :] - sensor_positions[None, :, :]), axis=2) > jnp.deg2rad(5))

sensor_positions

sensor_window_half_width = jnp.deg2rad(2.5)

for triangle in refined_triangles:
    triangle_closed = jnp.concatenate([triangle, triangle[0:1]], axis=0)
    ax3.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'g-', alpha=0.5, linewidth=0.5)

for sensor_pos in sensor_positions:
    rectangle_corners = jnp.array([
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] - sensor_window_half_width],
        [sensor_pos[0] + sensor_window_half_width, sensor_pos[1] - sensor_window_half_width], 
        [sensor_pos[0] + sensor_window_half_width, sensor_pos[1] + sensor_window_half_width],
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] + sensor_window_half_width],
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] - sensor_window_half_width]
    ])
    ax3.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], 'r-', linewidth=2)
    ax3.scatter(sensor_pos[0], sensor_pos[1], c='red', s=100, marker='x', zorder=10)

ax3.set_title("Barycentric Subdivision with Sensors")

for sensor_pos in sensor_positions:
    rectangle_corners = jnp.array([
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] - sensor_window_half_width],
        [sensor_pos[0] + sensor_window_half_width, sensor_pos[1] - sensor_window_half_width],
        [sensor_pos[0] + sensor_window_half_width, sensor_pos[1] + sensor_window_half_width], 
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] + sensor_window_half_width],
        [sensor_pos[0] - sensor_window_half_width, sensor_pos[1] - sensor_window_half_width]
    ])
    ax4.plot(rectangle_corners[:, 0], rectangle_corners[:, 1], 'r-', linewidth=2)
    ax4.scatter(sensor_pos[0], sensor_pos[1], c='red', s=100, marker='x', zorder=10)

assert sensor_positions.shape[0] == 3
assert sensor_window_half_width == jnp.deg2rad(2.5)

plt.tight_layout()
plt.savefig(figures_dir / f"triangulation_analysis_sensors_{n_directions}_{refined_triangles.shape[0]}.png", dpi=300, bbox_inches='tight')
plt.close()
