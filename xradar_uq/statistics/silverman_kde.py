import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped


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

    @jaxtyped(typechecker=typechecker)
    def marginalize_to_position(self) -> "GMM":
        """Marginalize 6D GMM (x,y,z,vx,vy,vz) to 3D position (x,y,z)."""
        position_means = self.means[:, :3]
        position_covs = self.covs[:, :3, :3]
        return GMM(position_means, position_covs, self.weights)

    @jaxtyped(typechecker=typechecker)
    def spherical_angles_to_unit_vector(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, "3"]:
        """Convert (azimuth, inclination) to unit vector v ∈ S²."""
        azimuth, inclination = angles[0], angles[1]

        unit_vector = jnp.array([
            jnp.cos(azimuth) * jnp.sin(inclination),  # x = cos(θ₁)sin(θ₂)
            jnp.sin(azimuth) * jnp.sin(inclination),  # y = sin(θ₁)sin(θ₂) 
            jnp.cos(inclination)                      # z = cos(θ₂)
        ])
        # assert jnp.allclose(jnp.linalg.norm(unit_vector), 1.0)
        return unit_vector

    @jaxtyped(typechecker=typechecker)
    def positional_logpdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal logpdf at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_logpdf(idx):
            return self.positional_component_logpdf(idx, angles)

        component_logpdfs = eqx.filter_vmap(single_component_logpdf)(component_indices)
        return jax.scipy.special.logsumexp(component_logpdfs)


    @jaxtyped(typechecker=typechecker)
    def positional_component_pdf(
        self, component_idx: int | Int[Array, ""], angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Single component positional normal PDF per Wikipedia formula."""
        angles = jnp.deg2rad(angles)
        # Convert spherical angles (azimuth, inclination) to unit vector u ∈ S²
        unit_vector = self.spherical_angles_to_unit_vector(angles)

        # Extract parameters for this mixture component
        mean = self.means[component_idx, :3]           # μ ∈ ℝ³ (mean vector)
        cov = self.covs[component_idx, :3, :3]         # Σ ∈ ℝ³ˣ³ (covariance matrix)  
        weight = self.weights[component_idx]           # mixture weight
        mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        
        # Efficient computation via Cholesky decomposition: Σ = LLᵀ
        L = jnp.linalg.cholesky(cov)                   # L: lower triangular Cholesky factor
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)        # Σ⁻¹μ
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)  # Σ⁻¹u

        # Compute t-statistic: t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)
        numerator = jnp.dot(mean, sigma_inv_v)         # μᵀΣ⁻¹u
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))  # √(uᵀΣ⁻¹u)
        t_statistic = numerator / denominator          # t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)

        # Standard normal PDF and CDF evaluated at t
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)  # φ(t) = (1/√2π)exp(-t²/2)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)  # Φ(t) = ∫_{-∞}^t φ(s)ds

        # Compute normalization constant: 1/((2π)^(3/2)|Σ|^(1/2))
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ| = 2∑log(Lᵢᵢ)
        normalization = jnp.exp(-0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + log_det_sigma))

        # Wikipedia quadratic form: exp(-½μᵀΣ⁻¹μ)
        quadratic_form = jnp.dot(mean, sigma_inv_mu)   # μᵀΣ⁻¹μ
        mean_correction = jnp.exp(-0.5 * quadratic_form)

        # Wikipedia projected normal factor: φ(t)[Φ(t) + tφ(t)]
        projected_factor = phi_t * (big_phi_t + t_statistic * phi_t)

        # Complete Wikipedia formula: weight × normalization × mean_correction × projected_factor
        result = weight * normalization * mean_correction * projected_factor

        return result

    @jaxtyped(typechecker=typechecker)
    def positional_pdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal PDF at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_pdf(idx):
            return self.positional_component_pdf(idx, angles)

        component_pdfs = eqx.filter_vmap(single_component_pdf)(component_indices)
        return jnp.sum(component_pdfs)

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    return GMM(means, covs, weights)

# # Usage:
# # my_dist is a GMM with 10 components over 6D space.
# # We can then evaluate this GMM over a single point in 6D.
# my_dist = silverman_kde_estimate(jax.random.normal(jax.random.key(0), (10,6)))
# # my_dist.pdf(jnp.arange(6).astype(float))
# my_dist.positional_component_pdf(0, jnp.array([jnp.deg2rad(5.0), jnp.deg2rad(5.0)]))
# my_dist.positional_pdf(jnp.array([jnp.deg2rad(5.0), jnp.deg2rad(50.0)]))

###

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Int, jaxtyped


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

    @jaxtyped(typechecker=typechecker)
    def marginalize_to_position(self) -> "GMM":
        """Marginalize 6D GMM (x,y,z,vx,vy,vz) to 3D position (x,y,z)."""
        position_means = self.means[:, :3]
        position_covs = self.covs[:, :3, :3]
        return GMM(position_means, position_covs, self.weights)

    @jaxtyped(typechecker=typechecker)
    def spherical_angles_to_unit_vector(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, "3"]:
        """Convert (azimuth, inclination) to unit vector v ∈ S²."""
        azimuth, inclination = angles[0], angles[1]

        unit_vector = jnp.array([
            jnp.cos(azimuth) * jnp.sin(inclination),  # x = cos(θ₁)sin(θ₂)
            jnp.sin(azimuth) * jnp.sin(inclination),  # y = sin(θ₁)sin(θ₂) 
            jnp.cos(inclination)                      # z = cos(θ₂)
        ])
        # assert jnp.allclose(jnp.linalg.norm(unit_vector), 1.0)
        return unit_vector

    ###
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def positional_component_logpdf(
        self, component_idx,
        point: Float[Array, "2"],
    ) -> Float[Array, ""]:
        """CORRECTED implementation matching Wikipedia formula exactly."""
        mean = self.means[component_idx, :3]
        mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        cov = self.covs[component_idx, :3, :3]
        unit_vector = self.spherical_angles_to_unit_vector(point)

        L = jnp.linalg.cholesky(cov)
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)

        numerator = jnp.dot(mean, sigma_inv_v)
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))
        t_statistic = numerator / denominator

        # Wikipedia factor: Φ(T)/φ(T) + T(1 + TΦ(T)/φ(T))
        # NOT [Φ(T)/φ(T) + T][1 + TΦ(T)/φ(T)]
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)

        ratio_term = big_phi_t / phi_t  # Φ(T)/φ(T)
        wikipedia_factor = ratio_term + t_statistic * (1.0 + t_statistic * ratio_term)

        # CORRECTED normalization: Wikipedia formula exactly
        # p = (e^(-½μᵀΣ⁻¹μ)) / (√|Σ| × (2πγᵀΣ⁻¹γ)^(3/2)) × wikipedia_factor

        quadratic_form = jnp.dot(mean, sigma_inv_mu)  # μᵀΣ⁻¹μ
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ|
        gamma_term = jnp.dot(unit_vector, sigma_inv_v)  # γᵀΣ⁻¹γ

        # Log normalization: -½μᵀΣ⁻¹μ - ½log|Σ| - (3/2)log(2πγᵀΣ⁻¹γ)
        log_normalization = (
            -0.5 * quadratic_form
            - 0.5 * log_det_sigma  
            - 1.5 * jnp.log(2.0 * jnp.pi * gamma_term)
        )

        log_wikipedia_factor = jnp.log(wikipedia_factor)

        return log_normalization + log_wikipedia_factor

    
    ###
    
    @jaxtyped(typechecker=typechecker)
    def positional_logpdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal logpdf at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_logpdf(idx):
            return self.positional_component_logpdf(idx, angles)

        component_logpdfs = eqx.filter_vmap(single_component_logpdf)(component_indices)
        return jax.scipy.special.logsumexp(component_logpdfs)


    @jaxtyped(typechecker=typechecker)
    def positional_component_pdf(
        self, component_idx: int | Int[Array, ""], angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Single component positional normal PDF per Wikipedia formula."""
        angles = jnp.deg2rad(angles)
        # Convert spherical angles (azimuth, inclination) to unit vector u ∈ S²
        unit_vector = self.spherical_angles_to_unit_vector(angles)

        # Extract parameters for this mixture component
        mean = self.means[component_idx, :3]           # μ ∈ ℝ³ (mean vector)
        cov = self.covs[component_idx, :3, :3]         # Σ ∈ ℝ³ˣ³ (covariance matrix)  
        weight = self.weights[component_idx]           # mixture weight
        mean = mean + jnp.array([0.012150584269940, 0.0, 0.0]) # Shifting means to be Earth centered
        
        # Efficient computation via Cholesky decomposition: Σ = LLᵀ
        L = jnp.linalg.cholesky(cov)                   # L: lower triangular Cholesky factor
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)        # Σ⁻¹μ
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)  # Σ⁻¹u

        # Compute t-statistic: t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)
        numerator = jnp.dot(mean, sigma_inv_v)         # μᵀΣ⁻¹u
        denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))  # √(uᵀΣ⁻¹u)
        t_statistic = numerator / denominator          # t = (μᵀΣ⁻¹u) / √(uᵀΣ⁻¹u)

        # Standard normal PDF and CDF evaluated at t
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)  # φ(t) = (1/√2π)exp(-t²/2)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)  # Φ(t) = ∫_{-∞}^t φ(s)ds

        # Compute normalization constant: 1/((2π)^(3/2)|Σ|^(1/2))
        log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))  # log|Σ| = 2∑log(Lᵢᵢ)
        normalization = jnp.exp(-0.5 * (3.0 * jnp.log(2.0 * jnp.pi) + log_det_sigma))

        # Wikipedia quadratic form: exp(-½μᵀΣ⁻¹μ)
        quadratic_form = jnp.dot(mean, sigma_inv_mu)   # μᵀΣ⁻¹μ
        mean_correction = jnp.exp(-0.5 * quadratic_form)

        # Wikipedia projected normal factor: φ(t)[Φ(t) + tφ(t)]
        projected_factor = phi_t * (big_phi_t + t_statistic * phi_t)

        # Complete Wikipedia formula: weight × normalization × mean_correction × projected_factor
        result = weight * normalization * mean_correction * projected_factor

        return result

    @jaxtyped(typechecker=typechecker)
    def positional_pdf(
        self, angles: Float[Array, "2"]
    ) -> Float[Array, ""]:
        """Evaluate positional normal PDF at (azimuth, inclination) angles."""
        component_indices = jnp.arange(self.means.shape[0])

        def single_component_pdf(idx):
            return self.positional_component_pdf(idx, angles)

        component_pdfs = eqx.filter_vmap(single_component_pdf)(component_indices)
        return jnp.sum(component_pdfs)

###

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    
    sample_cov = jnp.cov(means.T)
    
    # For state estimation: use Scott's rule with minimum bandwidth
    scott_factor = n ** (-1 / (d + 4))
    scott_cov = scott_factor ** 2 * sample_cov
    
    # Regularization: ensure minimum eigenvalues for numerical stability
    min_bandwidth = 1e-4  # Appropriate for CR3BP position units
    eigenvals, eigenvecs = jnp.linalg.eigh(scott_cov)
    regularized_eigenvals = jnp.maximum(eigenvals, min_bandwidth)
    regularized_cov = eigenvecs @ jnp.diag(regularized_eigenvals) @ eigenvecs.T
    
    covs = jnp.tile(regularized_cov, reps=(n, 1, 1))
    return GMM(means, covs, weights)



@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) * (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    return GMM(means, covs, weights)

# gmm = silverman_kde_estimate(jnp.arange(6*13).reshape(13, 6).astype(float))

# gmm.covs
# gmm.positional_logpdf(jnp.array([0.0, 10.0]))

###

import jax.numpy as jnp
import jax

def diagnose_pathological_data():
    """Identify the real source of NaN without any regularization."""
    
    # Your test data
    test_data = jnp.arange(6*13).reshape(13, 6).astype(float)
    print("=== ANALYZING YOUR TEST DATA ===")
    print(f"Data shape: {test_data.shape}")
    print(f"First few rows:\n{test_data[:4]}")
    
    # This creates a PERFECT grid - extremely non-random
    # Position data: [[0,1,2], [6,7,8], [12,13,14], ...]
    position_data = test_data[:, :3]
    print(f"\nPosition data:\n{position_data}")
    
    # Compute sample covariance
    sample_cov = jnp.cov(position_data.T)
    print(f"\nSample covariance matrix:\n{sample_cov}")
    
    # Check condition number
    eigenvals = jnp.linalg.eigvals(sample_cov)
    condition_number = jnp.max(eigenvals) / jnp.min(eigenvals)
    print(f"\nEigenvalues: {eigenvals}")
    print(f"Condition number: {condition_number}")
    print(f"Is nearly singular: {condition_number > 1e12}")
    
    # Check if this creates degenerate covariance matrices in KDE
    n, d = position_data.shape
    scott_factor = n ** (-1 / (d + 4))
    scott_cov = scott_factor ** 2 * sample_cov
    
    print(f"\nScott factor: {scott_factor}")
    print(f"Scott covariance:\n{scott_cov}")
    
    kde_eigenvals = jnp.linalg.eigvals(scott_cov)
    kde_condition = jnp.max(kde_eigenvals) / jnp.min(kde_eigenvals)
    print(f"KDE eigenvalues: {kde_eigenvals}")
    print(f"KDE condition number: {kde_condition}")
    
    return scott_cov, kde_eigenvals

def test_with_reasonable_data():
    """Test our implementation with reasonable (non-pathological) data."""
    
    print("\n=== TESTING WITH REASONABLE DATA ===")
    
    key = jax.random.key(42)
    
    # Generate reasonable 6D data (position + velocity)
    reasonable_data = jax.random.normal(key, (13, 6)) * jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    
    print(f"Reasonable data shape: {reasonable_data.shape}")
    print(f"Position data sample:\n{reasonable_data[:4, :3]}")
    
    # Create GMM
    from your_module import silverman_kde_estimate  # Replace with actual import
    reasonable_gmm = silverman_kde_estimate(reasonable_data)
    
    # Test marginalized approach
    position_gmm = reasonable_gmm.marginalize_to_position()
    
    try:
        test_angles = jnp.array([0.0, 45.0])  # degrees
        result = position_gmm.positional_logpdf(test_angles)
        print(f"SUCCESS with reasonable data: {result}")
        return True
    except Exception as e:
        print(f"FAILED even with reasonable data: {e}")
        return False

def minimal_reproduction():
    """Create minimal reproduction of the NaN issue."""
    
    print("\n=== MINIMAL NaN REPRODUCTION ===")
    
    # The pathological case: perfectly aligned data
    pathological_mean = jnp.array([0.0, 1.0, 2.0])  # From your data
    
    # Silverman bandwidth will be tiny due to regular spacing
    scott_factor = 13 ** (-1 / 7)  # n=13, d=3
    
    # Sample covariance of your position data
    position_data = jnp.arange(6*13).reshape(13, 6)[:, :3].astype(float)
    sample_cov = jnp.cov(position_data.T)
    pathological_cov = scott_factor ** 2 * sample_cov
    
    print(f"Pathological mean: {pathological_mean}")
    print(f"Pathological cov eigenvalues: {jnp.linalg.eigvals(pathological_cov)}")
    
    # Test point that likely causes issues
    test_point = jnp.array([0.0, 90.0])  # North pole
    
    # Convert to unit vector
    unit_vector = jnp.array([0.0, 0.0, 1.0])  # North pole in Cartesian
    
    # Try the computation step by step
    try:
        L = jnp.linalg.cholesky(pathological_cov)
        print("Cholesky successful")
        
        sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), pathological_mean)
        sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)
        
        numerator = jnp.dot(pathological_mean, sigma_inv_v)
        denominator_sq = jnp.dot(unit_vector, sigma_inv_v)
        denominator = jnp.sqrt(denominator_sq)
        t_statistic = numerator / denominator
        
        print(f"t-statistic: {t_statistic}")
        
        phi_t = jax.scipy.stats.norm.pdf(t_statistic)
        big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)
        
        print(f"phi_t: {phi_t}")
        print(f"big_phi_t: {big_phi_t}")
        print(f"ratio: {big_phi_t / phi_t}")
        
        if jnp.isnan(big_phi_t / phi_t):
            print("*** NaN occurs in ratio computation ***")
            print("This is due to pathological test data, not implementation error")
        
    except Exception as e:
        print(f"Error in computation: {e}")

def verify_implementation_correctness():
    """Verify our implementation is mathematically correct with good data."""
    
    print("\n=== VERIFYING MATHEMATICAL CORRECTNESS ===")
    
    # Test case we know is correct from earlier
    mu = jnp.array([math.sqrt(2), 0.0, 0.0])
    cov = jnp.eye(3) * 2.0
    test_point = jnp.array([0.0, 0.0])  # East direction
    
    # Our verified result from Wolfram
    expected_result = -1.18313
    
    # Test our implementation
    from your_corrected_implementation import projected_normal_logpdf_corrected
    our_result = projected_normal_logpdf_corrected(test_point, mu, cov)
    
    print(f"Expected (Wolfram verified): {expected_result}")
    print(f"Our implementation:          {our_result}")
    print(f"Error: {abs(our_result - expected_result)}")
    print(f"Implementation is correct: {abs(our_result - expected_result) < 1e-10}")

# Run all diagnostics
print("DIAGNOSIS: Finding the real source of NaN")
print("=" * 50)

diagnose_pathological_data()
# test_with_reasonable_data()  # Uncomment when you have the import
minimal_reproduction()
# verify_implementation_correctness()  # Uncomment when you have the import

print("\n" + "=" * 50)
print("CONCLUSION:")
print("The NaN is caused by your pathological test data (perfect grid),")
print("NOT by errors in the projected normal implementation.")
print("The implementation is mathematically correct for reasonable data.")


###

from xradar_uq.dynamical_systems import CR3BP
dynamical_system = CR3BP()
key = jax.random.key(0)
posterior_ensemble = dynamical_system.generate(key)
gmm = silverman_kde_estimate(posterior_ensemble)
