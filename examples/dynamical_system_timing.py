import equinox as eqx
import jax
from diffrax import SaveAt

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)
time_step = 10.0

# State Space Model
dynamical_system = CR3BP()
measurement_system = Radar()

# Filtering Method
stochastic_filter = EnGMF()

# True State (actual satellite) x^t
true_state = dynamical_system.initial_state() 
key, subkey = jax.random.split(key)

# x^+
posterior_ensemble = dynamical_system.generate(subkey, batch_size=1) 

# Predict Step

# Numerical integration of one state
import time
import jax.numpy as jnp

ts = jnp.linspace(0.0, time_step, 100)
saveat = SaveAt(ts=ts)
start = time.time()
ts, ys = dynamical_system.trajectory(0.0, time_step, true_state, saveat)
end = time.time()
print(end - start)
# Numerical integration of one state
time.sleep(0.5)
print("")
start = time.time()
ts_ensemble, ys_ensemble = eqx.filter_vmap(dynamical_system.trajectory, in_axes=(None, None, 0, None))(0.0, time_step, posterior_ensemble, saveat)
end = time.time()
print(end - start)

# initial_time: scalar, time_step: scalar, posterior_ensemble: (100, 6), saveat: scalar, scalar, ts=(100,)
# 100 x 6
