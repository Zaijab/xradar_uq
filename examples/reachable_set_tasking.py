import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polytopax as ptx
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import (AnglesOnly, DeepSpaceNetwork,
                                           simulate_thrust)
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.statistics import silverman_kde_estimate

key = jax.random.key(42)
dynamical_system = CR3BP()
stochastic_filter = EnGMF()
measurement_system = DeepSpaceNetwork()
true_state = dynamical_system.initial_state()
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey)


delta_v_magnitude = 2.0
time_range = 0.242
num_particles = 100

key, subkey = jax.random.split(key)
simulated_ensemble = simulate_thrust(subkey, posterior_ensemble, num_particles, delta_v_magnitude)
simulated_trajectories = eqx.filter_vmap(dynamical_system.flow)(0.0, time_range, simulated_ensemble)
gmm = silverman_kde_estimate(simulated_trajectories)

angles = AnglesOnly()
ensemble_angles = eqx.filter_vmap(angles)(simulated_trajectories)
my_hull = angular_convex_hull(ensemble_angles)

from xradar_uq.statistics.silverman_kde import GMM

from typing import Callable

# @jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def generate_triangulated_candidates(
    convex_hull: ptx.ConvexHull, subdivision_depth: int = 2
) -> Float[Array, "n_candidates 2"]:
    triangles = fan_triangulate(convex_hull)
    for _ in range(subdivision_depth):
        triangles = subdivide_triangles(triangles)
    centroids = jnp.mean(triangles, axis=1)
    assert centroids.shape[1] == 2
    return centroids

# @jaxtyped(typechecker=typechecker) 
@eqx.filter_jit
def select_random_candidate(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"]
) -> Float[Array, "2"]:
    random_idx = jax.random.choice(key, candidates.shape[0])
    selected_position = candidates[random_idx]
    assert selected_position.shape == (2,)
    return selected_position

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(candidates)
    optimal_idx = jnp.argmax(pdf_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position


# Random placement
candidates = generate_triangulated_candidates(my_hull, subdivision_depth=3)
from jax.tree_util import Partial

# @jaxtyped(typechecker=typechecker)

from typing import Callable


gmm_pdf_selection = Partial(select_pdf_weighted_candidate, gmm=gmm)
place_sensors(key, candidates, 3, jnp.deg2rad(5), gmm_pdf_selection)

random_selection = Partial(select_random_candidate)
place_sensors(key, candidates, 3, jnp.deg2rad(5), random_selection)

# Todo:
# Plot Hull + Sensor Locations + Color associated with 2D GMM PDF
# Plot the convex hull of azimuth elevation
# Plot the triangulation

# Explain Barycentric subdivision, arbitrarily small mesh size
# Flesh out paper with plots to fill in the blank

# Run Different Tasking Algorithm on DRO-A

# Plot multiple frontiers
# Make measurement frequency plot: How many measurements for 90%> Tracking
# Use convex hull in 6D to augment EnGMF
