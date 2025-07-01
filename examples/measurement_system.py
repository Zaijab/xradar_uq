"""
This file shows how to use measurement system objects.
"""

import jax
import jax.numpy as jnp

from xradar_uq.measurement_systems import AnglesOnly

# It takes in a covariance as a parameter.
# This is a square, positive definite matrix whose size is the same as the output space.
# In this example, AnglesOnly is a map which takes in an array of length >= 3
# It returns the azimuth and elevation (2 outputs)
measurement_system = AnglesOnly(
    covariance=jnp.diag(
        jnp.array([(0.1 * jnp.pi / 180) ** 2, (0.1 * jnp.pi / 180) ** 2])
    )
)

# Despite being an object, we can call it like a function
# This is because we implement the `__call__` method
state = jnp.ones(6)
perfect_measurement = measurement_system(state)

# The last call did not take in any randomness (A PsudoRNG Key)
# Hence it returns the measurement without any error
# In applications this is never the case, everything has an error
# To handle randomness in JAX, we need to define a key
seed = 42
key = jax.random.key(seed)

# These keys will be used in random functions and the same input always returns the same output
# So this will return the same number three times
print("Same keys:")
print("\t", jax.random.normal(key))
print("\t", jax.random.normal(key))
print("\t", jax.random.normal(key))
print()

# To propagate randomness, we need to split the keys
# This operation is symmetric, we don't 
key, subkey = jax.random.split(key)

# The proper way to handle them is to use one key for random evaluation of functions
# The other key is used to split to make other keys, so:
print("Diff keys:")
key, subkey = jax.random.split(key)
print("\t", jax.random.normal(subkey))
key, subkey = jax.random.split(key)
print("\t", jax.random.normal(subkey))
key, subkey = jax.random.split(key)
print("\t", jax.random.normal(subkey))
print()

# You don't need to know about how they work internally, but you can print them if you want
print(f"{key=}")

# Also, you can split multiple keys at once
# This is more efficient than splitting twice in a row
key, subkey_1, subkey_2 = jax.random.split(key, 3)

# Anyway, this is all to say, do this when you measure the true_state variables
key, subkey = jax.random.split(key)
measurement_system(state, subkey)
