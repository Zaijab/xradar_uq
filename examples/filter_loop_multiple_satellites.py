import jax

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)
key, subkey = jax.random.split(key)

dynamical_system = CR3BP()
true_states = dynamical_system.generate(subkey, batch_size=3)

stochastic_filters = EnGMF()
measurement_system = Radar()


posterior_ensemble = dynamical_system.generate(subkey)
