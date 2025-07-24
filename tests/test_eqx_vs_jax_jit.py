import polytopax as ptx
import jax.numpy as jnp
import jax
import equinox as eqx

class ZConvexHull(eqx.Module, ptx.ConvexHull):
    pass

@jax.jit
def f():
    return ZConvexHull.from_points(jnp.arange(12).reshape(4,3))

@eqx.filter_jit
def g():
    return f()

@eqx.filter_jit
def h():
    return ZConvexHull.from_points(jnp.arange(12).reshape(4,3))


h().centroid()
