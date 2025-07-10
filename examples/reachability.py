import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from beartype import beartype as typechecker
from tqdm.auto import tqdm

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF


key = jax.random.key(42)

dynamical_system = CR3BP()
measurement_system = Radar()
stochastic_filter = EnGMF()
initial_state = dynamical_system.initial_state()


delta_v_magnitude = 1e-2
seeds = 900
outer = 400

key, subkey = jax.random.split(key)
seed_vertices = jax.random.multivariate_normal(subkey, shape=(seeds,),mean=jnp.zeros(3), cov=jnp.eye(3))
seed_vertices /= jnp.linalg.norm(seed_vertices, axis=1, keepdims=True)
key, subkey = jax.random.split(key)
radii = jax.random.uniform(subkey, shape=(seeds,1)) ** (1/3)
seed_vertices = radii * seed_vertices
key, subkey = jax.random.split(key)
outer_vertices = jax.random.multivariate_normal(subkey, shape=(outer,),mean=jnp.zeros(3), cov=jnp.eye(3))
outer_vertices /= jnp.linalg.norm(outer_vertices, axis=1, keepdims=True)

import matplotlib.pyplot as plt
from pathlib import Path

def plot_step1_results(interior_vertices, boundary_vertices):
    Path("figures/reachability").mkdir(parents=True, exist_ok=True)
    LU_to_km = 389703  # km (Earth-Moon distance)
    TU_to_s = 382981   # seconds (≈ 4.348 days)
    VU_to_kmps = LU_to_km / TU_to_s  # ≈ 1.023 km/s
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*(interior_vertices * VU_to_kmps).T, alpha=0.6, label='Interior', s=20)
    ax.scatter(*(boundary_vertices * VU_to_kmps).T, alpha=0.8, label='Boundary', s=30, color='red')
    ax.set_xlabel('ΔVx (km/s)'); ax.set_ylabel('ΔVy (km/s)'); ax.set_zlabel('ΔVz (km/s)')
    ax.legend(); plt.title('Step 1: Seed Vertices in ΔV Sphere')
    plt.savefig("figures/reachability/step1_seed_vertices.png", dpi=150, bbox_inches='tight')

plot_step1_results(seed_vertices, outer_vertices)
