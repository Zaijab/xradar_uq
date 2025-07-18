import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import jax.numpy as jnp

# Import CR3BP to get DRO-A coordinates
# Simulating the CR3BP class from the codebase
class CR3BP:
    def initial_state(self):
        # DRO-A initial state from the codebase
        mean = jnp.array([
            1.021339954388544,
            -0.000000045869005,
            -0.181619950369762,
            0.000000617839352,
            -0.101759879771430,
            0.000001049698173]
        )
        return mean

# Physical constants
R_earth = 6371.0  # km
earth_moon_distance = 384400.0  # km
mu_earth_moon = 0.012150584269940  # CR3BP mass parameter

# NASA DSN locations (lat, lon in degrees)
dsn_stations = {
    'Goldstone': (35.4, -116.9),  # California, USA
    'Madrid': (40.4, -4.2),       # Spain
    'Canberra': (-35.4, 148.98)   # Australia
}

# Get DRO-A initial state from CR3BP
cr3bp = CR3BP()
dro_a_state = cr3bp.initial_state()
dro_a_position_normalized = dro_a_state[:3]  # [x, y, z] in CR3BP units

# Convert DRO-A position from CR3BP to physical coordinates
# CR3BP origin is at Earth-Moon barycenter
# Earth is at [-mu, 0, 0] in CR3BP coordinates
# Moon is at [1-mu, 0, 0] in CR3BP coordinates
earth_position_cr3bp = np.array([-mu_earth_moon, 0.0, 0.0])

# DRO-A position relative to Earth in CR3BP units
dro_a_rel_earth = dro_a_position_normalized - earth_position_cr3bp

# Convert to physical coordinates (km)
dro_a_position_km = dro_a_rel_earth * earth_moon_distance

print(f"DRO-A Position (CR3BP normalized): {dro_a_position_normalized}")
print(f"DRO-A Position relative to Earth (km): {dro_a_position_km}")
print(f"DRO-A Distance from Earth: {np.linalg.norm(dro_a_position_km):.0f} km")

# Use Goldstone as primary tracking station
station_name = 'Goldstone'
observer_lat, observer_lon = dsn_stations[station_name]

# Convert observer location to Cartesian coordinates (ECEF frame)
lat_rad = np.deg2rad(observer_lat)
lon_rad = np.deg2rad(observer_lon)

# Observer position on Earth's surface
obs_x = R_earth * np.cos(lat_rad) * np.cos(lon_rad)
obs_y = R_earth * np.cos(lat_rad) * np.sin(lon_rad)
obs_z = R_earth * np.sin(lat_rad)
observer_pos = np.array([obs_x, obs_y, obs_z])

def azimuth_elevation_to_local_cartesian(azimuth, elevation):
    """
    Convert azimuth-elevation to local Cartesian coordinates.
    Azimuth: 0° = North, 90° = East, measured clockwise from North
    Elevation: 0° = horizon, 90° = zenith
    """
    az_rad = np.deg2rad(azimuth)
    el_rad = np.deg2rad(elevation)
    
    # Local coordinate system (East, North, Up)
    x_local = np.cos(el_rad) * np.sin(az_rad)  # East
    y_local = np.cos(el_rad) * np.cos(az_rad)  # North
    z_local = np.sin(el_rad)                   # Up
    
    return np.array([x_local, y_local, z_local])

def local_to_ecef(local_vector, observer_lat, observer_lon):
    """
    Transform local ENU (East-North-Up) coordinates to ECEF.
    """
    lat_rad = np.deg2rad(observer_lat)
    lon_rad = np.deg2rad(observer_lon)
    
    # Rotation matrix from local ENU to ECEF
    R = np.array([
        [-np.sin(lon_rad), -np.cos(lon_rad)*np.sin(lat_rad), np.cos(lon_rad)*np.cos(lat_rad)],
        [np.cos(lon_rad), -np.sin(lon_rad)*np.sin(lat_rad), np.sin(lon_rad)*np.cos(lat_rad)],
        [0, np.cos(lat_rad), np.sin(lat_rad)]
    ])
    
    if local_vector.ndim == 1:
        return R @ local_vector
    else:
        return R @ local_vector.T

def ecef_to_local_azel(ecef_vector, observer_lat, observer_lon):
    """Convert ECEF vector to local azimuth and elevation."""
    lat_rad = np.deg2rad(observer_lat)
    lon_rad = np.deg2rad(observer_lon)
    
    # Rotation matrix from ECEF to local ENU
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.cos(lon_rad)*np.sin(lat_rad), -np.sin(lon_rad)*np.sin(lat_rad), np.cos(lat_rad)],
        [np.cos(lon_rad)*np.cos(lat_rad), np.sin(lon_rad)*np.cos(lat_rad), np.sin(lat_rad)]
    ])
    
    local_vector = R @ ecef_vector
    
    # Convert to azimuth and elevation
    elevation = np.rad2deg(np.arcsin(local_vector[2]))
    azimuth = np.rad2deg(np.arctan2(local_vector[0], local_vector[1]))
    azimuth = (azimuth + 360) % 360  # Ensure 0-360 range
    
    return azimuth, elevation

# Calculate line of sight from DSN station to DRO-A
los_vector = dro_a_position_km - observer_pos
los_distance = np.linalg.norm(los_vector)
los_unit = los_vector / los_distance

# Calculate actual pointing direction to DRO-A
actual_azimuth, actual_elevation = ecef_to_local_azel(los_unit, observer_lat, observer_lon)

print(f"\nDSN {station_name} to DRO-A:")
print(f"  Distance: {los_distance:.0f} km")
print(f"  Azimuth: {actual_azimuth:.1f}°")
print(f"  Elevation: {actual_elevation:.1f}°")

# Define observation window centered on DRO-A direction
azimuth_center = actual_azimuth
elevation_center = actual_elevation
window_size = 5.0  # degrees

# Create grid of azimuth and elevation points
az_range = np.linspace(azimuth_center - window_size/2, azimuth_center + window_size/2, 6)
el_range = np.linspace(elevation_center - window_size/2, elevation_center + window_size/2, 6)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Draw Earth as a sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
earth_x = R_earth * np.outer(np.cos(u), np.sin(v))
earth_y = R_earth * np.outer(np.sin(u), np.sin(v))
earth_z = R_earth * np.outer(np.ones(np.size(u)), np.cos(v))

ax.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='lightblue', 
                linewidth=0, antialiased=True)

# Plot observer location (DSN station)
ax.scatter(*observer_pos, color='red', s=200, label=f'DSN {station_name}', marker='^')

# Plot DRO-A position
ax.scatter(*dro_a_position_km, color='gold', s=150, label='DRO-A', marker='*')

# Plot direct line of sight to DRO-A
ax.plot([observer_pos[0], dro_a_position_km[0]], 
        [observer_pos[1], dro_a_position_km[1]], 
        [observer_pos[2], dro_a_position_km[2]], 
        color='red', linewidth=4, alpha=0.8, label='Line of Sight to DRO-A')

# Add Moon position for context
moon_position_cr3bp = np.array([1.0 - mu_earth_moon, 0.0, 0.0])
moon_rel_earth = moon_position_cr3bp - earth_position_cr3bp
moon_position_km = moon_rel_earth * earth_moon_distance
ax.scatter(*moon_position_km, color='gray', s=100, label='Moon', alpha=0.7)

# Generate and plot observation rays
ray_length = los_distance * 1.2  # Extend rays beyond DRO-A
ray_colors = ['blue', 'green', 'orange', 'purple', 'brown']

for i, az in enumerate(az_range):
    for j, el in enumerate(el_range):
        # Convert to local Cartesian
        local_direction = azimuth_elevation_to_local_cartesian(az, el)
        
        # Transform to ECEF
        ecef_direction = local_to_ecef(local_direction, observer_lat, observer_lon)
        
        # Create ray from observer position
        ray_end = observer_pos + ray_length * ecef_direction
        
        # Plot ray
        color = ray_colors[(i + j) % len(ray_colors)]
        alpha = 0.7 if (i == len(az_range)//2 and j == len(el_range)//2) else 0.4
        linewidth = 3 if (i == len(az_range)//2 and j == len(el_range)//2) else 1
        
        ax.plot([observer_pos[0], ray_end[0]], 
                [observer_pos[1], ray_end[1]], 
                [observer_pos[2], ray_end[2]], 
                color=color, alpha=alpha, linewidth=linewidth)

# Draw the observation cone boundary
n_boundary = 20
boundary_rays = []

# Create boundary of the observation window
for az in np.linspace(azimuth_center - window_size/2, azimuth_center + window_size/2, n_boundary):
    for el_bound in [elevation_center - window_size/2, elevation_center + window_size/2]:
        local_dir = azimuth_elevation_to_local_cartesian(az, el_bound)
        ecef_dir = local_to_ecef(local_dir, observer_lat, observer_lon)
        boundary_rays.append(observer_pos + ray_length * ecef_dir)

for el in np.linspace(elevation_center - window_size/2, elevation_center + window_size/2, n_boundary):
    for az_bound in [azimuth_center - window_size/2, azimuth_center + window_size/2]:
        local_dir = azimuth_elevation_to_local_cartesian(az_bound, el)
        ecef_dir = local_to_ecef(local_dir, observer_lat, observer_lon)
        boundary_rays.append(observer_pos + ray_length * ecef_dir)

# Plot boundary points
if boundary_rays:
    boundary_points = np.array(boundary_rays)
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], 
               color='red', alpha=0.3, s=10)

# Add coordinate axes
axis_length = R_earth * 1.5
ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', alpha=0.5, linewidth=2, label='X (0°N, 0°E)')
ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', alpha=0.5, linewidth=2, label='Y (0°N, 90°E)')
ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', alpha=0.5, linewidth=2, label='Z (North Pole)')

# Set labels and title
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title(f'NASA DSN Tracking of DRO-A\n'
             f'Station: {station_name} ({observer_lat}°N, {observer_lon}°E)\n'
             f'Target: DRO-A (Az: {actual_azimuth:.1f}°, El: {actual_elevation:.1f}°)\n'
             f'Observation Window: {window_size}° × {window_size}°')

# Set equal aspect ratio
max_range = max(np.abs(dro_a_position_km).max(), R_earth * 2)
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

# Add legend
ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Add text with tracking information
ax.text2D(0.02, 0.98, 
          f"NASA DSN Tracking Parameters:\n"
          f"Station: {station_name}\n"
          f"Target: DRO-A\n"
          f"Range: {los_distance:.0f} km\n"
          f"Azimuth: {actual_azimuth:.1f}° ± {window_size/2}°\n"
          f"Elevation: {actual_elevation:.1f}° ± {window_size/2}°\n"
          f"Window: {window_size}° × {window_size}°",
          transform=ax.transAxes, fontsize=10, verticalalignment='top',
          bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()

# Create directory and save figure
os.makedirs("figures/sensor_figures", exist_ok=True)
plt.savefig("figures/sensor_figures/dsn_dro_a_tracking_3d.png", dpi=300, bbox_inches='tight')
plt.savefig("figures/sensor_figures/dsn_dro_a_tracking_3d.pdf", bbox_inches='tight')

plt.show()

# Print DSN tracking analysis
print(f"\nNASA DSN Tracking Analysis:")
print(f"  Station: {station_name} ({observer_lat}°N, {observer_lon}°E)")
print(f"  Target: DRO-A (Distant Retrograde Orbit)")
print(f"  Station Height: {R_earth:.1f} km (Earth surface)")
print(f"  Target Range: {los_distance:.1f} km")
print(f"  Pointing: Az={actual_azimuth:.1f}°, El={actual_elevation:.1f}°")
print(f"  Observation Window: {window_size}° × {window_size}°")
print(f"  Total Field of View: {window_size**2:.1f} square degrees")

# Calculate solid angle coverage
solid_angle = (np.deg2rad(window_size))**2 * np.cos(np.deg2rad(actual_elevation))
print(f"  Solid Angle: {solid_angle:.6f} steradians")
print(f"  Sky Coverage: {solid_angle/(4*np.pi)*100:.6f}% of full sphere")

# DRO-A orbital characteristics
print(f"\nDRO-A Characteristics:")
print(f"  Position (CR3BP): [{dro_a_position_normalized[0]:.3f}, {dro_a_position_normalized[1]:.3f}, {dro_a_position_normalized[2]:.3f}]")
print(f"  Distance from Earth-Moon barycenter: {np.linalg.norm(dro_a_position_normalized) * earth_moon_distance:.0f} km")
print(f"  Distance from Earth center: {los_distance:.0f} km ({los_distance/earth_moon_distance:.2f} Earth-Moon distances)")
