import jax.numpy as jnp
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
import equinox as eqx
import jax
import math

"""
NON-ISOTROPIC PROJECTED NORMAL TEST CASE

Test Case Setup:
- Covariance: Σ = diag(1, 1, 4) (non-isotropic!)
- Mean: μ = [1, 0, 0] (pointing east)  
- Test point: (azimuth=π/2, elevation=0) → γ = [0, 1, 0] (pointing north)

This gives us t-statistic = 0, which is a nice special case to verify.
"""

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def spherical_to_cartesian(azimuth_elevation: Float[Array, "2"]) -> Float[Array, "3"]:
    azimuth, elevation = azimuth_elevation[0], azimuth_elevation[1]
    return jnp.array([
        jnp.cos(elevation) * jnp.cos(azimuth),
        jnp.cos(elevation) * jnp.sin(azimuth), 
        jnp.sin(elevation)
    ])

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def projected_normal_logpdf_corrected(
    point: Float[Array, "2"], 
    mean: Float[Array, "3"], 
    cov: Float[Array, "3 3"]
) -> Float[Array, ""]:
    """Corrected implementation matching Wikipedia formula exactly."""
    unit_vector = spherical_to_cartesian(point)
    
    L = jnp.linalg.cholesky(cov)
    sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)
    sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)
    
    numerator = jnp.dot(mean, sigma_inv_v)
    denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))
    t_statistic = numerator / denominator
    
    # Wikipedia factor: Φ(T)/φ(T) + T(1 + TΦ(T)/φ(T))
    phi_t = jax.scipy.stats.norm.pdf(t_statistic)
    big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)
    
    ratio_term = big_phi_t / phi_t  # Φ(T)/φ(T)
    wikipedia_factor = ratio_term + t_statistic * (1.0 + t_statistic * ratio_term)
    
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

def symbolic_nonisotropic_calculation():
    """Manual symbolic calculation for non-isotropic case."""
    print("=== NON-ISOTROPIC SYMBOLIC CALCULATION ===")
    print("Σ = diag(1, 1, 4), μ = [1, 0, 0], γ = [0, 1, 0]")
    print("Test point: (azimuth=π/2, elevation=0) → North direction")
    print()
    
    # Step 1: Basic quantities
    mu = jnp.array([1.0, 0.0, 0.0])
    gamma = jnp.array([0.0, 1.0, 0.0])  # North direction
    cov = jnp.diag(jnp.array([1.0, 1.0, 4.0]))
    
    print(f"μ = {mu}")
    print(f"γ = {gamma}")
    print(f"Σ = diag{jnp.diag(cov)}")
    print()
    
    # Step 2: Inverse covariance operations
    # Σ⁻¹ = diag(1, 1, 1/4)
    sigma_inv = jnp.diag(jnp.array([1.0, 1.0, 0.25]))
    print(f"Σ⁻¹ = diag(1, 1, 1/4)")
    
    # μᵀΣ⁻¹μ = [1,0,0] × diag(1,1,1/4) × [1,0,0]ᵀ = 1×1 = 1
    mu_sigma_inv_mu = jnp.dot(mu, sigma_inv @ mu)
    print(f"μᵀΣ⁻¹μ = 1×1 + 0×1 + 0×(1/4) = {mu_sigma_inv_mu}")
    
    # γᵀΣ⁻¹γ = [0,1,0] × diag(1,1,1/4) × [0,1,0]ᵀ = 1×1 = 1
    gamma_sigma_inv_gamma = jnp.dot(gamma, sigma_inv @ gamma)
    print(f"γᵀΣ⁻¹γ = 0×1 + 1×1 + 0×(1/4) = {gamma_sigma_inv_gamma}")
    
    # μᵀΣ⁻¹γ = [1,0,0] × diag(1,1,1/4) × [0,1,0]ᵀ = 1×0 = 0
    mu_sigma_inv_gamma = jnp.dot(mu, sigma_inv @ gamma)
    print(f"μᵀΣ⁻¹γ = 1×0 + 0×1 + 0×0 = {mu_sigma_inv_gamma}")
    print()
    
    # Step 3: t-statistic (special case!)
    # T = (μᵀΣ⁻¹γ) / √(γᵀΣ⁻¹γ) = 0 / √1 = 0
    t_statistic = mu_sigma_inv_gamma / jnp.sqrt(gamma_sigma_inv_gamma)
    print(f"T = 0 / √1 = 0 = {t_statistic}")
    print("Special case: t = 0 means mean and test direction are orthogonal!")
    print()
    
    # Step 4: Normalization components  
    # |Σ| = det(diag(1,1,4)) = 1×1×4 = 4
    det_sigma = 1.0 * 1.0 * 4.0
    print(f"|Σ| = 1×1×4 = {det_sigma}")
    
    # e^(-½μᵀΣ⁻¹μ) = e^(-1/2)
    exp_term = jnp.exp(-0.5 * mu_sigma_inv_mu)
    print(f"e^(-½μᵀΣ⁻¹μ) = e^(-1/2) = {exp_term}")
    
    # √|Σ| = √4 = 2
    sqrt_det = jnp.sqrt(det_sigma)
    print(f"√|Σ| = √4 = {sqrt_det}")
    
    # (2πγᵀΣ⁻¹γ)^(3/2) = (2π × 1)^(3/2) = (2π)^(3/2)
    denom_power = (2 * math.pi * gamma_sigma_inv_gamma) ** 1.5
    print(f"(2πγᵀΣ⁻¹γ)^(3/2) = (2π)^(3/2) = {denom_power}")
    print()
    
    # Step 5: Wikipedia factor components (t = 0 case!)
    # T = 0, so we need Φ(0), φ(0)
    phi_t = jax.scipy.stats.norm.pdf(t_statistic)  # φ(0) = 1/√(2π)
    big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)  # Φ(0) = 1/2
    
    print(f"φ(0) = 1/√(2π) = {phi_t}")
    print(f"Φ(0) = 1/2 = {big_phi_t}")
    
    # Wikipedia factor: Φ(0)/φ(0) + 0×(1 + 0×Φ(0)/φ(0)) = Φ(0)/φ(0) = (1/2)/(1/√(2π)) = √(2π)/2
    ratio = big_phi_t / phi_t
    wikipedia_factor = ratio + t_statistic * (1 + t_statistic * ratio)
    print(f"Φ(0)/φ(0) = (1/2)/(1/√(2π)) = √(2π)/2 = {ratio}")
    print(f"Wikipedia factor = {ratio} + 0×(...) = {wikipedia_factor}")
    print()
    
    # Step 6: Final result
    log_normalization = jnp.log(exp_term) - jnp.log(sqrt_det) - jnp.log(denom_power)
    log_wikipedia = jnp.log(wikipedia_factor)
    
    total_logpdf = log_normalization + log_wikipedia
    
    print(f"log(normalization) = {log_normalization}")
    print(f"log(Wikipedia factor) = {log_wikipedia}")
    print(f"EXACT log PDF = {total_logpdf}")
    
    return {
        'exact_logpdf': total_logpdf,
        't_statistic': t_statistic,
        'mu': mu,
        'gamma': gamma,
        'cov': cov
    }

@jaxtyped(typechecker=typechecker)
def test_nonisotropic_case():
    """Test our implementation against the non-isotropic symbolic result."""
    
    # Run symbolic calculation
    exact_result = symbolic_nonisotropic_calculation()
    
    print("\n=== NON-ISOTROPIC IMPLEMENTATION TEST ===")
    
    # Test point: azimuth=π/2, elevation=0 → [0,1,0] (north)
    test_point = jnp.array([jnp.pi/2, 0.0])
    mean = exact_result['mu']
    cov = exact_result['cov']
    
    # Test our implementation
    computed_logpdf = projected_normal_logpdf_corrected(test_point, mean, cov)
    exact_logpdf = exact_result['exact_logpdf']
    
    print(f"Our implementation: {computed_logpdf}")
    print(f"Exact symbolic:     {exact_logpdf}")
    print(f"Absolute error:     {abs(computed_logpdf - exact_logpdf)}")
    print(f"Relative error:     {abs(computed_logpdf - exact_logpdf) / abs(exact_logpdf)}")
    
    # Tolerance check
    tolerance = 1e-12
    if abs(computed_logpdf - exact_logpdf) < tolerance:
        print("✓ EXACT MATCH within numerical precision!")
    else:
        print("✗ Mismatch detected")
    
    return computed_logpdf, exact_logpdf

def print_wolfram_commands_nonisotropic():
    """Print Wolfram commands for non-isotropic verification."""
    print("\n=== WOLFRAM ALPHA VERIFICATION (NON-ISOTROPIC) ===")
    print("To verify independently:")
    print()
    print("1. t-statistic (should be 0):")
    print("   0 / Sqrt[1]")
    print()
    print("2. Normal CDF and PDF at t=0:")
    print("   CDF[NormalDistribution[0,1], 0]")
    print("   PDF[NormalDistribution[0,1], 0]")
    print()
    print("3. Wikipedia factor (should be √(2π)/2):")
    print("   (1/2) / (1/Sqrt[2*Pi])")
    print("   Simplify[Sqrt[2*Pi]/2]")
    print()
    print("4. Determinant and final calculation:")
    print("   Det[{{1,0,0},{0,1,0},{0,0,4}}]")
    print("   Log[Exp[-1/2] / (2 * (2*Pi)^(3/2))] + Log[Sqrt[2*Pi]/2]")

# RUN THE NON-ISOTROPIC TEST
if __name__ == "__main__":
    test_nonisotropic_case()
    print_wolfram_commands_nonisotropic()

# Execute the test
print("\n" + "="*60)
print("RUNNING NON-ISOTROPIC TEST:")
test_nonisotropic_case()
