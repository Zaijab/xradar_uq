import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
from jaxtyping import Array, Float
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

# @jaxtyped(typechecker=typechecker)
def find_last_reliable_frequency_index(detection_rates: Float[Array, "n_freq"]) -> int:
    n_freq = detection_rates.shape[0]
    consecutive_drops = (detection_rates[:-1] <= 0.9) & (detection_rates[1:] <= 0.9)
    first_consecutive_drop = jnp.argmax(consecutive_drops)
    has_consecutive_drop = jnp.any(consecutive_drops)
    return jnp.where(has_consecutive_drop, jnp.maximum(0, first_consecutive_drop - 1), n_freq - 1)

# @jaxtyped(typechecker=typechecker)  
def compute_frequency_threshold_map(found_proportions: Float[Array, "n_freq n_dv n_mp"]) -> Float[Array, "n_dv n_mp"]:
    n_freq, n_dv, n_mp = found_proportions.shape
    frequency_map = jnp.zeros((n_dv, n_mp))
    
    for i in range(n_dv):
        for j in range(n_mp):
            detection_sequence = found_proportions[:, i, j]
            frequency_map = frequency_map.at[i, j].set(find_last_reliable_frequency_index(detection_sequence))
    
    assert frequency_map.shape == (n_dv, n_mp)
    return frequency_map

measurement_frequency_results = jnp.load('cache/measurement_frequency/results_sweep_engmf.npy')
assert measurement_frequency_results.shape == (30, 20, 11)

minimum_frequency_map = compute_frequency_threshold_map(measurement_frequency_results)
assert minimum_frequency_map.shape == (20, 11)

os.makedirs('figures/measurement_frequency_analysis', exist_ok=True)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(minimum_frequency_map, aspect='auto', cmap='viridis', origin='lower')

for i in range(minimum_frequency_map.shape[0]):
    for j in range(minimum_frequency_map.shape[1]):
        text = ax.text(j, i, f'{int(minimum_frequency_map[i, j])}',
                      ha="center", va="center", color="white", fontsize=8, weight='bold')

ax.set_xlabel('Maneuver Proportion Index')
ax.set_ylabel('Delta V Magnitude Index')
ax.set_title('Last Reliable Measurement Frequency Index\n(Before 2 Consecutive Drops Below 90%)')
plt.colorbar(im, ax=ax, label='Frequency Index')
plt.tight_layout()
plt.savefig('figures/measurement_frequency_analysis/reliable_frequency_heatmap_annotated.png', dpi=300)
plt.show()
