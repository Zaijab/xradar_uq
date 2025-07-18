import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

@jaxtyped(typechecker=typechecker)
class GMM(eqx.Module):
    means: Float[Array, "num_components state_dim"]
    covs: Float[Array, "num_components state_dim state_dim"]
    weights: Float[Array, "num_components"]
    
    @jaxtyped(typechecker=typechecker)
    def __init__(self, means, covs, weights, max_components=1):
        max_components = max(means.shape[0], max_components)
        pad_width = max_components - means.shape[0]
        self.means = jnp.pad(means, ((0, pad_width), (0, 0)))
        self.covs = jnp.pad(covs, ((0, pad_width), (0, 0), (0, 0)))
        self.weights = jnp.pad(weights, (0, pad_width))
    
    @jaxtyped(typechecker=typechecker)
    def pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute probability density at point x."""
        
        def component_pdf(mean, cov, weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return weight * jnp.exp(log_prob)
        
        component_probs = eqx.filter_vmap(component_pdf)(
            self.means, self.covs, self.weights
        )
        
        return jnp.sum(component_probs)
    
    @jaxtyped(typechecker=typechecker) 
    def log_pdf(self, x: Float[Array, "state_dim"]) -> Float[Array, ""]:
        """Compute log probability density."""
        
        def component_log_pdf(mean, cov, log_weight):
            L = jnp.linalg.cholesky(cov)
            
            diff = x - mean
            y = jax.scipy.linalg.solve_triangular(L, diff, lower=True)
            
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
            quad_form = jnp.sum(y**2)
            
            k = mean.shape[0]
            log_prob = -0.5 * (k * jnp.log(2 * jnp.pi) + log_det + quad_form)
            
            return log_weight + log_prob
        
        log_weights = jnp.log(self.weights)
        log_component_probs = eqx.filter_vmap(component_log_pdf)(
            self.means, self.covs, log_weights
        )
        
        return jax.scipy.special.logsumexp(log_component_probs)

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    return GMM(means, covs, weights)

# Usage:
# my_dist = silverman_kde_estimate(jax.random.normal(jax.random.key(0), (10,2)))
# my_dist.pdf(jnp.array([1.0, 2.0]))
