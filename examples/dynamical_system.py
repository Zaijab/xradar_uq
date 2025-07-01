"""
This file shows how to use the dynamical system objects.
"""
# First, import the object
import equinox as eqx
import jax.numpy as jnp
from diffrax import Dopri8, PIDController, SaveAt
from jaxtyping import Array, Float

from xradar_uq.dynamical_systems import CR3BP

# Next, we can instantiate it with a number of options
# These set up parameters for the dynamical system `mu`
# The rest are parameters for the solver
dynamical_system = CR3BP(
    # The non dimensionless mass ratio
    mu=0.012150584269940,
    # dt for the solver
    dt=0.01,
    # @article{prince1981high,
    #         author={Prince, P. J and Dormand, J. R.},
    #         title={High order embedded {R}unge--{K}utta formulae},
    #         journal={J. Comp. Appl. Math},
    #         year={1981},
    #         volume={7},
    #         number={1},
    #         pages={67--75}
    # }
    solver=Dopri8(),
    # The tolerance is calculated as `atol + rtol * y` for the evolving solution `y`.
    # https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller
    stepsize_controller=PIDController(rtol=1e-12, atol=1e-12),
)

# The API of a dynamical system comes from the nomenclature:
# https://en.wikipedia.org/wiki/Dynamical_system#Formal_definition
# So we want a trajectory, flow, and orbit of our dynamical system
# This API also works for discrete systems too, but we won't deal with them here.

# To integrate a system, we at least need the following:

initial_time: float = 0.0
final_time: float = 10.0
state: Float[Array, "6"] = jnp.ones(6)

# Which means we have the flow through x
# This is the result of numerical integration after final_time
new_state = dynamical_system.flow(initial_time, final_time, state)

# The trajectory is the graph of the float
# Of course, we cannot store the continuity of the real line on our computer
# Thus, we make a choice of what points we want to save at
# This is handled by the SaveAt object
ts = jnp.linspace(initial_time, final_time, 100)
saveat = SaveAt(ts=ts)

# The result of the trajectory is both the times and the corresponding ys at those times
# They have the same leading index size
# So, because we saved at ts which is length 100
# ts is of length 100, ys is 100 x 
ts, ys = dynamical_system.trajectory(initial_time, final_time, state, saveat)

# The orbit is is the image of the flow
# This is just the trajectory without the time
ys = dynamical_system.orbit(initial_time, final_time, state, saveat)

# So far, we have propagated states forward one at a time.
# If we wanted to take 10 states and propagate them all
batch = jnp.ones((10, 6))

# Then we cannot naively flow the batch
# new_state = dynamical_system.flow(initial_time, final_time, batch)
# This will yell at you, because it expects (6,) array.
# I made the API strict on purpose so I avoid errors down the line.
# Instead, we need to vectorize the operation over the dimension
# Equinox has a nice vmap operation
# https://docs.kidger.site/equinox/api/transformations/#vectorisation-and-parallelisation
# In theory, I could have this 
new_state = eqx.filter_vmap(dynamical_system.flow)(initial_time, final_time, batch)
