import numpy as np
import jax
import jax.numpy as jnp
points = jnp.array([[0, 0], [0, 1.1], [1, 0], [1, 1]])
from scipy.spatial import Delaunay

import equinox as eqx


import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial import Delaunay
import equinox as eqx

def triangulate_callback(points):
    points_np = np.asarray(points)
    tri = Delaunay(points_np)
    return jnp.array(tri.simplices, dtype=jnp.int32)

@eqx.filter_jit
def triangulate_jit(points):
    n_points = points.shape[0] 
    max_simplices = 2 * n_points - 5  # Upper bound for 2D Delaunay
    result_shape = jax.ShapeDtypeStruct((max_simplices, 3), jnp.int32)
    
    simplices = jax.pure_callback(
        triangulate_callback, 
        result_shape, 
        points,
        vmap_method="sequential"
    )
    return simplices

def get_edges_from_simplices(simplices):
    edges_per_simplex = jnp.array([[0, 1], [1, 2], [2, 0]])
    n_simplices = simplices.shape[0]
    all_edges = simplices[:, edges_per_simplex].reshape(-1, 2)
    sorted_edges = jnp.sort(all_edges, axis=1)
    unique_edges = jnp.unique(sorted_edges, axis=0, size=3*n_simplices, fill_value=-1)
    valid_mask = unique_edges[:, 0] >= 0
    return unique_edges[valid_mask]

# Test
points = jnp.array([[0, 0], [0, 1.1], [1, 0], [1, 1]], dtype=jnp.float32)
simplices = triangulate_jit(points)
edges = get_edges_from_simplices(simplices)
assert edges.shape[0] > 0
