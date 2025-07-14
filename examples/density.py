import jax
import jax.numpy as jnp
import jaxkd as jk
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from beartype import beartype as typechecker
import equinox as eqx

@eqx.filter_jit
@typechecker
def compute_sparsity_measures(points: Float[Array, "n d"], k: int) -> Float[Array, "n"]:
    """Compute k-NN sparsity measure for each point."""
    assert points.ndim == 2
    assert k > 0 and k < points.shape[0]
    
    tree = jk.build_tree(points)
    _, distances = jk.query_neighbors(tree, points, k=k+1)  # +1 to exclude self
    kth_distances = distances[:, k]  # k-th neighbor (excluding self)
    
    assert kth_distances.shape == (points.shape[0],)
    return kth_distances

def generate_varied_density_points(key, n_total=500):
    """Generate points with varying density regions."""
    key1, key2, key3, key4 = jax.random.split(key, 4)
    
    # Dense cluster 1 (bottom-left)
    cluster1 = jax.random.multivariate_normal(
        key1, mean=jnp.array([-2, -2]), cov=0.1 * jnp.eye(2), shape=(150,)
    )
    
    # Dense cluster 2 (top-right) 
    cluster2 = jax.random.multivariate_normal(
        key2, mean=jnp.array([2, 2]), cov=0.15 * jnp.eye(2), shape=(150,)
    )
    
    # Medium density cluster (center)
    cluster3 = jax.random.multivariate_normal(
        key3, mean=jnp.array([0, 0]), cov=0.3 * jnp.eye(2), shape=(100,)
    )
    
    # Sparse scattered points
    sparse_points = jax.random.uniform(
        key4, shape=(100, 2), minval=-4, maxval=4
    )
    
    # Combine all points
    all_points = jnp.concatenate([cluster1, cluster2, cluster3, sparse_points])
    return all_points

# Generate data
key = jax.random.key(42)
points = generate_varied_density_points(key)

# Compute k-NN sparsity measures for different k values
k_values = [3, 5, 10]
fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5))

for i, k in enumerate(k_values):
    sparsity = compute_sparsity_measures(points, k)
    
    scatter = axes[i].scatter(
        points[:, 0], points[:, 1], 
        c=sparsity, cmap='viridis', 
        s=20, alpha=0.7
    )
    
    axes[i].set_title(f'k={k} Nearest Neighbor Distance')
    axes[i].set_xlabel('X')
    axes[i].set_ylabel('Y')
    axes[i].grid(True, alpha=0.3)
    
    # Add colorbar
    plt.colorbar(scatter, ax=axes[i], label='k-NN Distance')

plt.suptitle('Point Sparsity Visualization: Color = k-th Nearest Neighbor Distance', 
         y=1.00, fontsize=14)
plt.tight_layout()
plt.savefig("figures/knn_dist.png")

# Print statistics
print("Sparsity Statistics:")
for k in k_values:
    sparsity = compute_sparsity_measures(points, k)
    print(f"k={k}: min={sparsity.min():.3f}, max={sparsity.max():.3f}, "
          f"mean={sparsity.mean():.3f}, std={sparsity.std():.3f}")
