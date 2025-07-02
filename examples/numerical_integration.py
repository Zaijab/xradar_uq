from xradar_uq.dynamical_systems import CR3BP
import jax


key = jax.random.key(42)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
initial_state = dynamical_system.initial_state()
ensemble = dynamical_system.generate()
