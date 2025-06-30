"""
This is an example file to show you how to plot basic things in Python.
"""

import jax.numpy as jnp
from diffrax import SaveAt
from jaxtyping import Array, Float

from xradar_uq.dynamical_systems import CR3BP

dynamical_system = CR3BP()
initial_state = dynamical_system.initial_state()
t0, t1 = 0.0, 2.0
ts = jnp.linspace(t0, t1, 100)
ts, ys = dynamical_system.trajectory(t0, t1, initial_state, SaveAt(ts=ts))


isinstance(ys, Float[Array, f"{ts.shape[0]} 6"])
# ys: (100 x 6)
# ts: (100,)

import matplotlib.pyplot as plt

import os
os.makedirs("figures", exist_ok=True)

# Extract position components
positions = ys[:, :3]  # (100, 3) - x, y, z coordinates

# 2D trajectory plot (x-y plane)
plt.figure(figsize=(8, 6))
plt.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5)
plt.scatter(positions[0, 0], positions[0, 1], c='green', s=50, label='Start')
plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, label='End')
plt.xlabel('x')
plt.ylabel('y')
plt.title('CR3BP Trajectory')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.savefig('figures/cr3bp_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()
