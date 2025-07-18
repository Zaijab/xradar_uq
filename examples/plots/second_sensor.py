import jax.numpy as jnp
import matplotlib.pyplot as plt

# Sensor parameters  
window_half_size = 2.5
constraint_distance = 5.0
inner_boundary = constraint_distance - window_half_size  # 2.5°
outer_boundary = constraint_distance + window_half_size  # 7.5°

# Create observable region boundaries
n_points = 200
# Outer boundary (7.5°)
azimuth_outer = jnp.concatenate([
    jnp.full(n_points//4, outer_boundary), jnp.linspace(outer_boundary, -outer_boundary, n_points//4),
    jnp.full(n_points//4, -outer_boundary), jnp.linspace(-outer_boundary, outer_boundary, n_points//4)
])
elevation_outer = jnp.concatenate([
    jnp.linspace(-outer_boundary, outer_boundary, n_points//4), jnp.full(n_points//4, outer_boundary),
    jnp.linspace(outer_boundary, -outer_boundary, n_points//4), jnp.full(n_points//4, -outer_boundary)
])

# Inner boundary (2.5°)
azimuth_inner = jnp.concatenate([
    jnp.full(n_points//4, inner_boundary), jnp.linspace(inner_boundary, -inner_boundary, n_points//4),
    jnp.full(n_points//4, -inner_boundary), jnp.linspace(-inner_boundary, inner_boundary, n_points//4)
])
elevation_inner = jnp.concatenate([
    jnp.linspace(-inner_boundary, inner_boundary, n_points//4), jnp.full(n_points//4, inner_boundary),
    jnp.linspace(inner_boundary, -inner_boundary, n_points//4), jnp.full(n_points//4, -inner_boundary)
])

# Minimal constraint boundary representation
boundary_resolution = 50  # Points per edge
t = jnp.linspace(0, 4, 4*boundary_resolution, endpoint=False)
side = jnp.floor(t).astype(int)
local_t = t - side

# Parametric rectangle boundary
azimuth_constraint = jnp.where(side == 0, constraint_distance, 
                     jnp.where(side == 1, constraint_distance - 2*constraint_distance*local_t,
                     jnp.where(side == 2, -constraint_distance, 
                               -constraint_distance + 2*constraint_distance*local_t)))
elevation_constraint = jnp.where(side == 0, -constraint_distance + 2*constraint_distance*local_t,
                        jnp.where(side == 1, constraint_distance,
                        jnp.where(side == 2, constraint_distance - 2*constraint_distance*local_t,
                                  -constraint_distance)))

angles_rad = jnp.arctan2(elevation_constraint, azimuth_constraint)

# Plot
fig, ax = plt.subplots(figsize=(8, 8))
# Fill observable region for sensor 2 (between 2.5° and 7.5°)
ax.fill(azimuth_outer, elevation_outer, alpha=0.3, color='lightblue', label='Sensor 2 Observable')
ax.fill(azimuth_inner, elevation_inner, alpha=1.0, color='white')  # Cut out inner hole
# Show sensor 2 placement boundary
ax.scatter(azimuth_constraint, elevation_constraint, c=angles_rad, cmap='hsv', s=4, label='Sensor 2 Centers')
# Show sensor 1 window
ax.add_patch(plt.Rectangle((-window_half_size, -window_half_size), 2*window_half_size, 2*window_half_size,
                          fill=False, edgecolor='red', linewidth=2, label='Sensor 1'))
ax.set_xlabel('Azimuth (deg)'); ax.set_ylabel('Elevation (deg)')
ax.set_title('Sensor 2 Observable Region')
ax.legend(); ax.grid(True); ax.set_aspect('equal')
plt.savefig("figures/sensor_figures/second_sensor.png")
plt.show()
