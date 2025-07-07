import jax
import matplotlib.pyplot as plt
from xradar_uq.dynamical_systems import CR3BP

key = jax.random.key(42)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
initial_state = dynamical_system.initial_state()
ensemble = dynamical_system.generate(subkey)

plt.scatter(ensemble[:, 0], ensemble[:, 1], c="red")
plt.scatter(initial_state[0], initial_state[1], c="green")
plt.savefig("figures/sanity_check_initialization.png")
