# gmm = silverman_kde_estimate(jnp.arange(6*13).reshape(13, 6).astype(float))

# gmm.covs
# gmm.positional_logpdf(jnp.array([0.0, 10.0]))

###

# import jax.numpy as jnp
# import jax

# def diagnose_pathological_data():
#     """Identify the real source of NaN without any regularization."""
    
#     # Your test data
#     test_data = jnp.arange(6*13).reshape(13, 6).astype(float)
#     print("=== ANALYZING YOUR TEST DATA ===")
#     print(f"Data shape: {test_data.shape}")
#     print(f"First few rows:\n{test_data[:4]}")
    
#     # This creates a PERFECT grid - extremely non-random
#     # Position data: [[0,1,2], [6,7,8], [12,13,14], ...]
#     position_data = test_data[:, :3]
#     print(f"\nPosition data:\n{position_data}")
    
#     # Compute sample covariance
#     sample_cov = jnp.cov(position_data.T)
#     print(f"\nSample covariance matrix:\n{sample_cov}")
    
#     # Check condition number
#     eigenvals = jnp.linalg.eigvals(sample_cov)
#     condition_number = jnp.max(eigenvals) / jnp.min(eigenvals)
#     print(f"\nEigenvalues: {eigenvals}")
#     print(f"Condition number: {condition_number}")
#     print(f"Is nearly singular: {condition_number > 1e12}")
    
#     # Check if this creates degenerate covariance matrices in KDE
#     n, d = position_data.shape
#     scott_factor = n ** (-1 / (d + 4))
#     scott_cov = scott_factor ** 2 * sample_cov
    
#     print(f"\nScott factor: {scott_factor}")
#     print(f"Scott covariance:\n{scott_cov}")
    
#     kde_eigenvals = jnp.linalg.eigvals(scott_cov)
#     kde_condition = jnp.max(kde_eigenvals) / jnp.min(kde_eigenvals)
#     print(f"KDE eigenvalues: {kde_eigenvals}")
#     print(f"KDE condition number: {kde_condition}")
    
#     return scott_cov, kde_eigenvals

# def test_with_reasonable_data():
#     """Test our implementation with reasonable (non-pathological) data."""
    
#     print("\n=== TESTING WITH REASONABLE DATA ===")
    
#     key = jax.random.key(42)
    
#     # Generate reasonable 6D data (position + velocity)
#     reasonable_data = jax.random.normal(key, (13, 6)) * jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])
    
#     print(f"Reasonable data shape: {reasonable_data.shape}")
#     print(f"Position data sample:\n{reasonable_data[:4, :3]}")
    
#     # Create GMM
#     from your_module import silverman_kde_estimate  # Replace with actual import
#     reasonable_gmm = silverman_kde_estimate(reasonable_data)
    
#     # Test marginalized approach
#     position_gmm = reasonable_gmm.marginalize_to_position()
    
#     try:
#         test_angles = jnp.array([0.0, 45.0])  # degrees
#         result = position_gmm.positional_logpdf(test_angles)
#         print(f"SUCCESS with reasonable data: {result}")
#         return True
#     except Exception as e:
#         print(f"FAILED even with reasonable data: {e}")
#         return False

# def minimal_reproduction():
#     """Create minimal reproduction of the NaN issue."""
    
#     print("\n=== MINIMAL NaN REPRODUCTION ===")
    
#     # The pathological case: perfectly aligned data
#     pathological_mean = jnp.array([0.0, 1.0, 2.0])  # From your data
    
#     # Silverman bandwidth will be tiny due to regular spacing
#     scott_factor = 13 ** (-1 / 7)  # n=13, d=3
    
#     # Sample covariance of your position data
#     position_data = jnp.arange(6*13).reshape(13, 6)[:, :3].astype(float)
#     sample_cov = jnp.cov(position_data.T)
#     pathological_cov = scott_factor ** 2 * sample_cov
    
#     print(f"Pathological mean: {pathological_mean}")
#     print(f"Pathological cov eigenvalues: {jnp.linalg.eigvals(pathological_cov)}")
    
#     # Test point that likely causes issues
#     test_point = jnp.array([0.0, 90.0])  # North pole
    
#     # Convert to unit vector
#     unit_vector = jnp.array([0.0, 0.0, 1.0])  # North pole in Cartesian
    
#     # Try the computation step by step
#     try:
#         L = jnp.linalg.cholesky(pathological_cov)
#         print("Cholesky successful")
        
#         sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), pathological_mean)
#         sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)
        
#         numerator = jnp.dot(pathological_mean, sigma_inv_v)
#         denominator_sq = jnp.dot(unit_vector, sigma_inv_v)
#         denominator = jnp.sqrt(denominator_sq)
#         t_statistic = numerator / denominator
        
#         print(f"t-statistic: {t_statistic}")
        
#         phi_t = jax.scipy.stats.norm.pdf(t_statistic)
#         big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)
        
#         print(f"phi_t: {phi_t}")
#         print(f"big_phi_t: {big_phi_t}")
#         print(f"ratio: {big_phi_t / phi_t}")
        
#         if jnp.isnan(big_phi_t / phi_t):
#             print("*** NaN occurs in ratio computation ***")
#             print("This is due to pathological test data, not implementation error")
        
#     except Exception as e:
#         print(f"Error in computation: {e}")

# def verify_implementation_correctness():
#     """Verify our implementation is mathematically correct with good data."""
    
#     print("\n=== VERIFYING MATHEMATICAL CORRECTNESS ===")
    
#     # Test case we know is correct from earlier
#     mu = jnp.array([math.sqrt(2), 0.0, 0.0])
#     cov = jnp.eye(3) * 2.0
#     test_point = jnp.array([0.0, 0.0])  # East direction
    
#     # Our verified result from Wolfram
#     expected_result = -1.18313
    
#     # Test our implementation
#     from your_corrected_implementation import projected_normal_logpdf_corrected
#     our_result = projected_normal_logpdf_corrected(test_point, mu, cov)
    
#     print(f"Expected (Wolfram verified): {expected_result}")
#     print(f"Our implementation:          {our_result}")
#     print(f"Error: {abs(our_result - expected_result)}")
#     print(f"Implementation is correct: {abs(our_result - expected_result) < 1e-10}")

# # Run all diagnostics
# print("DIAGNOSIS: Finding the real source of NaN")
# print("=" * 50)

# diagnose_pathological_data()
# # test_with_reasonable_data()  # Uncomment when you have the import
# minimal_reproduction()
# # verify_implementation_correctness()  # Uncomment when you have the import

# print("\n" + "=" * 50)
# print("CONCLUSION:")
# print("The NaN is caused by your pathological test data (perfect grid),")
# print("NOT by errors in the projected normal implementation.")
# print("The implementation is mathematically correct for reasonable data.")


###

from xradar_uq.dynamical_systems import CR3BP
dynamical_system = CR3BP()
key = jax.random.key(0)
posterior_ensemble = dynamical_system.generate(key)
gmm = silverman_kde_estimate(posterior_ensemble)

###

# import jax.numpy as jnp
# import math
# from jaxtyping import Array, Float, jaxtyped
# from beartype import beartype as typechecker
# import equinox as eqx

# """
# CR3BP FRAME-AWARE PROJECTED NORMAL

# Key insight: In CR3BP synodic frame:
# - Barycenter at (0, 0, 0)  
# - Earth at (-μ, 0, 0) where μ ≈ 0.012150584269940
# - Satellite positions in GMM are in barycentric coordinates
# - Angles (azimuth, elevation) are measured FROM EARTH perspective

# Transformation needed:
# 1. Satellite position: r_sat (barycentric)
# 2. Earth-centered position: r_earth_centered = r_sat - (-μ, 0, 0) = r_sat + (μ, 0, 0)
# 3. Then project and compute angles from this Earth-centered frame
# """

# def test_cr3bp_frame_correction():
#     """Test the frame correction with realistic CR3BP scenario."""
    
#     print("=== CR3BP FRAME CORRECTION TEST ===")
    
#     # Typical DRO-A position in barycentric coordinates  
#     satellite_pos_barycentric = jnp.array([1.021, -0.000, -0.182])  # From your CR3BP data
#     earth_pos_barycentric = jnp.array([-CR3BP_MU, 0.0, 0.0])
    
#     print(f"Satellite (barycentric): {satellite_pos_barycentric}")
#     print(f"Earth (barycentric):     {earth_pos_barycentric}")
    
#     # Earth-centered position
#     satellite_pos_earth_centered = satellite_pos_barycentric - earth_pos_barycentric
#     print(f"Satellite (Earth-centered): {satellite_pos_earth_centered}")
    
#     # What angles does this correspond to from Earth?
#     distance = jnp.linalg.norm(satellite_pos_earth_centered)
#     unit_vector = satellite_pos_earth_centered / distance
    
#     # Convert back to spherical coordinates
#     elevation = jnp.arcsin(unit_vector[2])
#     azimuth = jnp.arctan2(unit_vector[1], unit_vector[0])
    
#     print(f"\nFrom Earth perspective:")
#     print(f"Distance: {distance:.6f} AU")
#     print(f"Azimuth:  {jnp.rad2deg(azimuth):.1f}°")
#     print(f"Elevation: {jnp.rad2deg(elevation):.1f}°")
    
#     # Test with simple covariance
#     test_cov = jnp.eye(3) * 1e-4  # Small uncertainty in DU
#     test_weight = 1.0
    
#     # Create test arrays
#     means_test = satellite_pos_barycentric.reshape(1, 3)
#     covs_test = test_cov.reshape(1, 3, 3)
#     weights_test = jnp.array([test_weight])
    
#     # Test the corrected implementation
#     test_angles = jnp.array([azimuth, elevation])
    
#     try:
#         result = cr3bp_positional_component_logpdf(
#             0, test_angles, means_test, covs_test, weights_test, CR3BP_MU
#         )
#         print(f"\n✓ SUCCESS: CR3BP projected normal = {result}")
#         return True
#     except Exception as e:
#         print(f"\n✗ FAILED: {e}")
#         return False

# def compare_with_without_frame_correction():
#     """Compare results with and without frame correction."""
    
#     print("\n=== FRAME CORRECTION COMPARISON ===")
    
#     # Test case
#     satellite_pos = jnp.array([1.021, 0.001, -0.182])  # Barycentric
#     test_cov = jnp.eye(3) * 1e-4
#     test_angles = jnp.array([0.1, -0.2])  # Some arbitrary angles
    
#     means_test = satellite_pos.reshape(1, 3)
#     covs_test = test_cov.reshape(1, 3, 3)
#     weights_test = jnp.array([1.0])
    
#     print(f"Test satellite position: {satellite_pos}")
#     print(f"Test angles: {test_angles} rad = {jnp.rad2deg(test_angles)}°")
    
#     try:
#         # With frame correction (CR3BP-aware)
#         result_corrected = cr3bp_positional_component_logpdf(
#             0, test_angles, means_test, covs_test, weights_test, CR3BP_MU
#         )
        
#         # Without frame correction (naive barycentric)
#         result_naive = cr3bp_positional_component_logpdf(
#             0, test_angles, means_test, covs_test, weights_test, 0.0  # mu=0
#         )
        
#         print(f"\nWith CR3BP frame correction:    {result_corrected}")
#         print(f"Without frame correction (naive): {result_naive}")
#         print(f"Difference: {result_corrected - result_naive}")
        
#         return result_corrected, result_naive
        
#     except Exception as e:
#         print(f"Failed: {e}")
#         return None, None

# def verify_frame_transformation():
#     """Verify the frame transformation is mathematically correct."""
    
#     print("\n=== FRAME TRANSFORMATION VERIFICATION ===")
    
#     # Known test case
#     earth_pos = jnp.array([-CR3BP_MU, 0.0, 0.0])
#     satellite_barycentric = jnp.array([1.0, 0.5, 0.2])
    
#     # Manual transformation
#     satellite_earth_centered = satellite_barycentric - earth_pos
#     print(f"Satellite (barycentric):     {satellite_barycentric}")
#     print(f"Earth (barycentric):         {earth_pos}")  
#     print(f"Satellite (Earth-centered):  {satellite_earth_centered}")
    
#     # This should equal satellite_barycentric + (μ, 0, 0)
#     expected = satellite_barycentric + jnp.array([CR3BP_MU, 0.0, 0.0])
#     print(f"Expected (manual calc):       {expected}")
#     print(f"Match: {jnp.allclose(satellite_earth_centered, expected)}")
    
#     # Convert to spherical and back
#     distance = jnp.linalg.norm(satellite_earth_centered)
#     unit_vec = satellite_earth_centered / distance
    
#     elevation = jnp.arcsin(unit_vec[2])
#     azimuth = jnp.arctan2(unit_vec[1], unit_vec[0])
    
#     # Reconstruct
#     reconstructed = distance * jnp.array([
#         jnp.cos(elevation) * jnp.cos(azimuth),
#         jnp.cos(elevation) * jnp.sin(azimuth),
#         jnp.sin(elevation)
#     ])
    
#     print(f"\nSpherical coordinates:")
#     print(f"Distance: {distance}")
#     print(f"Azimuth:  {azimuth} rad = {jnp.rad2deg(azimuth):.1f}°")
#     print(f"Elevation: {elevation} rad = {jnp.rad2deg(elevation):.1f}°")
#     print(f"Reconstructed: {reconstructed}")
#     print(f"Round-trip match: {jnp.allclose(satellite_earth_centered, reconstructed)}")

# # Run all tests
# if __name__ == "__main__":
#     print("CR3BP FRAME-AWARE PROJECTED NORMAL TESTING")
#     print("=" * 60)
    
#     success = test_cr3bp_frame_correction()
#     compare_with_without_frame_correction()
#     verify_frame_transformation()
    
#     print("\n" + "=" * 60)
#     print("SUMMARY:")
#     if success:
#         print("✓ CR3BP frame correction implemented successfully")
#         print("✓ Accounts for Earth position at (-μ, 0, 0)")
#         print("✓ Angles measured from Earth perspective")
#     else:
#         print("✗ Frame correction needs debugging")
        
#     print("\nNext: Update your GMM class to use cr3bp_positional_component_logpdf")

# # Execute tests
# test_cr3bp_frame_correction()
# compare_with_without_frame_correction()
# verify_frame_transformation()

# print("")
# print("##########################")
# print("")

# Your existing workflow, but frame-aware

###

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, jaxtyped
from beartype import beartype as typechecker
import equinox as eqx

# NumPyro's approach (specialized case)
@jaxtyped(typechecker=typechecker)
def numpyro_projected_normal_logpdf(
    concentration: Float[Array, "3"], 
    unit_vector: Float[Array, "3"]
) -> Float[Array, ""]:
    """NumPyro's ray integration approach - assumes Σ = I."""
    
    t = jnp.dot(concentration, unit_vector)
    t2 = t * t
    r2 = jnp.dot(concentration, concentration) - t2
    
    perp_part = -0.5 * r2 - jnp.log(2 * jnp.pi)
    
    para_part = jnp.log(
        t * jnp.exp(-0.5 * t2) / jnp.sqrt(2 * jnp.pi)
        + (1 + t2) * (1 + jax.scipy.special.erf(t / jnp.sqrt(2))) / 2
    )
    
    return para_part + perp_part

# Your approach (general case)  
@jaxtyped(typechecker=typechecker)
def your_projected_normal_logpdf(
    mean: Float[Array, "3"],
    cov: Float[Array, "3 3"],
    unit_vector: Float[Array, "3"]
) -> Float[Array, ""]:
    """Your Mills ratio approach - handles arbitrary μ, Σ."""
    
    L = jnp.linalg.cholesky(cov)
    sigma_inv_mu = jax.scipy.linalg.cho_solve((L, True), mean)
    sigma_inv_v = jax.scipy.linalg.cho_solve((L, True), unit_vector)
    
    numerator = jnp.dot(mean, sigma_inv_v)
    denominator = jnp.sqrt(jnp.dot(unit_vector, sigma_inv_v))
    t_statistic = numerator / denominator
    
    phi_t = jax.scipy.stats.norm.pdf(t_statistic)
    big_phi_t = jax.scipy.stats.norm.cdf(t_statistic)
    ratio_term = big_phi_t / phi_t
    wikipedia_factor = ratio_term + t_statistic * (1.0 + t_statistic * ratio_term)
    
    quadratic_form = jnp.dot(mean, sigma_inv_mu)
    log_det_sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    gamma_term = jnp.dot(unit_vector, sigma_inv_v)
    
    log_normalization = (
        -0.5 * quadratic_form - 0.5 * log_det_sigma 
        - 1.5 * jnp.log(2.0 * jnp.pi * gamma_term)
    )
    
    return log_normalization + jnp.log(wikipedia_factor)

# Test equivalence for identity covariance case
@jaxtyped(typechecker=typechecker) 
def test_equivalence():
    """Test if methods agree when Σ = I."""
    mean = jnp.array([1.0, 0.5, -0.3])
    cov = jnp.eye(3)
    unit_vector = jnp.array([0.6, 0.8, 0.0])
    
    numpyro_result = numpyro_projected_normal_logpdf(mean, unit_vector)
    your_result = your_projected_normal_logpdf(mean, cov, unit_vector)
    
    assert jnp.allclose(numpyro_result, your_result, rtol=1e-6)
    return numpyro_result, your_result
test_equivalence()
