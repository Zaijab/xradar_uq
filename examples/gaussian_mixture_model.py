from xradar_uq.statistics import GMM, silverman_kde_estimate
from xradar_uq.dynamical_systems import CR3BP
import jax.numpy as jnp
import equinox as eqx
import jax

dynamical_system = CR3BP()

posterior_ensemble = dynamical_system.generate(jax.random.key(0))
gmm = silverman_kde_estimate(posterior_ensemble)

azimuth = jnp.linspace(- jnp.pi, jnp.pi, 10) # -180, 180
elevation = jnp.linspace(- jnp.pi, jnp.pi, 10) # -90, 90
azimuth_mesh, elevation_mesh = jnp.meshgrid(azimuth, elevation, indexing='ij')
coordinate_pairs_flat = jnp.stack([azimuth_mesh.flatten(), elevation_mesh.flatten()], axis=1)

gmm.positional_pdf(jnp.array([jnp.deg2rad(0.0),jnp.deg2rad(180)]))
