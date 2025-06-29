"""
It's possible you would want to save the outputs of certain operations to different formats.
In this file we will take a single matrix and output it to various formats.
"""
import jax

key = jax.random.key(42)
data = jax.random.normal(key, (1_000,))

"""
Output array to `*.csv`
These are files which are plaintext and look like

    , col1, col2
row1, 1.0 , 2.0

These are easily compatible with Matlab.
"""

"""
Output array to `*.mat`
"""
