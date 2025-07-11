import pickle

def load_rse_results(filepath):
    """Load RSE results from pickle."""
    with open(filepath, 'rb') as f:
        results = pickle.load(f)
    
    # Convert numpy arrays back to JAX
    return convert_numpy_to_jax(results)

def convert_jax_to_numpy(obj):
    """Recursively convert JAX arrays to numpy."""
    if isinstance(obj, jnp.ndarray):
        return jnp.asarray(obj)  # Ensures numpy conversion
    elif isinstance(obj, dict):
        return {k: convert_jax_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_jax_to_numpy(item) for item in obj)
    else:
        return obj

def convert_numpy_to_jax(obj):
    """Recursively convert numpy arrays to JAX."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return jnp.array(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_jax(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_to_jax(item) for item in obj)
    else:
        return obj

loaded_results = load_rse_results("rse_results.pkl")
loaded_results.keys()

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import Radar
from xradar_uq.stochastic_filters import EnGMF

key = jax.random.key(42)

dynamical_system = CR3BP()
measurement_system = Radar()
stochastic_filter = EnGMF()
initial_state = dynamical_system.initial_state()

