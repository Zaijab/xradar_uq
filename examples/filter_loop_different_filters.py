import equinox as eqx
import jax
from diffrax import SaveAt
import time
import jax.numpy as jnp

from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.stochastic_filters import EnGMF, EnKF

###

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem


@jaxtyped(typechecker=typechecker)
class UKF(AbstractFilter, strict=True):
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    debug: bool = False
    
    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit  
    def generate_sigma_points(
        self,
        mean: Float[Array, "state_dim"],
        covariance: Float[Array, "state_dim state_dim"],
    ) -> tuple[Float[Array, "2*state_dim+1 state_dim"], Float[Array, "2*state_dim+1"]]:
        n = mean.shape[0]
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        L = jnp.linalg.cholesky((n + lambda_param) * covariance)
        
        sigma_points = jnp.zeros((2*n + 1, n))
        sigma_points = sigma_points.at[0].set(mean)
        sigma_points = sigma_points.at[1:n+1].set(mean[None, :] + L)
        sigma_points = sigma_points.at[n+1:2*n+1].set(mean[None, :] - L)
        
        weights_m = jnp.zeros(2*n + 1)
        weights_c = jnp.zeros(2*n + 1)
        
        weights_m = weights_m.at[0].set(lambda_param / (n + lambda_param))
        weights_c = weights_c.at[0].set(lambda_param / (n + lambda_param) + (1 - self.alpha**2 + self.beta))
        
        weight_other = 1.0 / (2 * (n + lambda_param))
        weights_m = weights_m.at[1:].set(weight_other)
        weights_c = weights_c.at[1:].set(weight_other)
        
        return sigma_points, weights_m, weights_c

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def predict_sigma_points(
        self,
        sigma_points: Float[Array, "2*state_dim+1 state_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "2*state_dim+1 measurement_dim"]:
        return eqx.filter_vmap(measurement_system)(sigma_points)

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_ensemble: Float[Array, "batch_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_dim state_dim"]:
        
        # Convert ensemble to mean and covariance
        prior_mean = jnp.mean(prior_ensemble, axis=0)
        prior_cov = jnp.cov(prior_ensemble.T)
        
        if self.debug:
            assert prior_mean.shape == (prior_ensemble.shape[1],)
            assert prior_cov.shape == (prior_ensemble.shape[1], prior_ensemble.shape[1])
        
        # Generate sigma points
        sigma_points, weights_m, weights_c = self.generate_sigma_points(prior_mean, prior_cov)
        
        # Predict measurements through sigma points
        predicted_measurements = self.predict_sigma_points(sigma_points, measurement_system)
        
        # Predicted measurement mean
        measurement_mean = jnp.sum(weights_m[:, None] * predicted_measurements, axis=0)
        
        # Innovation covariance
        innovation_residuals = predicted_measurements - measurement_mean[None, :]
        innovation_cov = jnp.sum(
            weights_c[:, None, None] * innovation_residuals[:, :, None] * innovation_residuals[:, None, :], 
            axis=0
        ) + measurement_system.covariance
        
        # Cross-covariance
        state_residuals = sigma_points - prior_mean[None, :]
        cross_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * innovation_residuals[:, None, :],
            axis=0
        )
        
        # Kalman gain
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, cross_cov.T).T
        
        # Innovation
        innovation = measurement - measurement_mean
        
        # Posterior mean and covariance
        posterior_mean = prior_mean + kalman_gain @ innovation
        posterior_cov = prior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        
        # Ensure symmetry and positive definiteness
        posterior_cov = (posterior_cov + posterior_cov.T) / 2
        posterior_cov = posterior_cov + 1e-8 * jnp.eye(posterior_cov.shape[0])
        
        if self.debug:
            assert posterior_mean.shape == prior_mean.shape
            assert posterior_cov.shape == prior_cov.shape
        
        # Convert back to ensemble representation
        # Sample from posterior distribution
        batch_size = prior_ensemble.shape[0]
        subkeys = jax.random.split(key, batch_size)
        posterior_ensemble = eqx.filter_vmap(
            lambda k: jax.random.multivariate_normal(k, posterior_mean, posterior_cov)
        )(subkeys)
        
        if self.debug:
            assert posterior_ensemble.shape == prior_ensemble.shape
            
        return posterior_ensemble

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped

from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem


# @jaxtyped(typechecker=typechecker)
class UKF(AbstractFilter, strict=True):
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    debug: bool = False
    
    # @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit  
    def generate_sigma_points(
        self,
        mean: Float[Array, "state_dim"],
        covariance: Float[Array, "state_dim state_dim"],
    ) -> tuple[Float[Array, "2*state_dim+1 state_dim"], Float[Array, "2*state_dim+1"], Float[Array, "2*state_dim+1"]]:
        n = mean.shape[0]
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        L = jnp.linalg.cholesky((n + lambda_param) * covariance)
        
        sigma_points = jnp.zeros((2*n + 1, n))
        sigma_points = sigma_points.at[0].set(mean)
        sigma_points = sigma_points.at[1:n+1].set(mean[None, :] + L)
        sigma_points = sigma_points.at[n+1:2*n+1].set(mean[None, :] - L)
        
        weights_m = jnp.zeros(2*n + 1)
        weights_c = jnp.zeros(2*n + 1)
        
        weights_m = weights_m.at[0].set(lambda_param / (n + lambda_param))
        weights_c = weights_c.at[0].set(lambda_param / (n + lambda_param) + (1 - self.alpha**2 + self.beta))
        
        weight_other = 1.0 / (2 * (n + lambda_param))
        weights_m = weights_m.at[1:].set(weight_other)
        weights_c = weights_c.at[1:].set(weight_other)
        
        return sigma_points, weights_m, weights_c

    # @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def predict_sigma_points(
        self,
        sigma_points: Float[Array, "2*state_dim+1 state_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "2*state_dim+1 measurement_dim"]:
        return eqx.filter_vmap(measurement_system)(sigma_points)

    # @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def update(
        self,
        key: Key[Array, ""],
        prior_ensemble: Float[Array, "batch_dim state_dim"],
        measurement: Float[Array, "measurement_dim"],
        measurement_system: AbstractMeasurementSystem,
    ) -> Float[Array, "batch_dim state_dim"]:
        
        # Convert ensemble to mean and covariance
        prior_mean = jnp.mean(prior_ensemble, axis=0)
        prior_cov = jnp.cov(prior_ensemble.T)
        
        if self.debug:
            assert prior_mean.shape == (prior_ensemble.shape[1],)
            assert prior_cov.shape == (prior_ensemble.shape[1], prior_ensemble.shape[1])
        
        # Generate sigma points
        sigma_points, weights_m, weights_c = self.generate_sigma_points(prior_mean, prior_cov)
        
        # Predict measurements through sigma points
        predicted_measurements = self.predict_sigma_points(sigma_points, measurement_system)
        
        # Predicted measurement mean
        measurement_mean = jnp.sum(weights_m[:, None] * predicted_measurements, axis=0)
        
        # Innovation covariance
        innovation_residuals = predicted_measurements - measurement_mean[None, :]
        innovation_cov = jnp.sum(
            weights_c[:, None, None] * innovation_residuals[:, :, None] * innovation_residuals[:, None, :], 
            axis=0
        ) + measurement_system.covariance
        
        # Cross-covariance
        state_residuals = sigma_points - prior_mean[None, :]
        cross_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * innovation_residuals[:, None, :],
            axis=0
        )
        
        # Kalman gain
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, cross_cov.T).T
        
        # Innovation
        innovation = measurement - measurement_mean
        
        # Posterior mean and covariance
        posterior_mean = prior_mean + kalman_gain @ innovation
        posterior_cov = prior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        
        # Ensure symmetry and positive definiteness
        posterior_cov = (posterior_cov + posterior_cov.T) / 2
        posterior_cov = posterior_cov + 1e-8 * jnp.eye(posterior_cov.shape[0])
        
        if self.debug:
            assert posterior_mean.shape == prior_mean.shape
            assert posterior_cov.shape == prior_cov.shape
        
        # Convert back to ensemble representation
        # Sample from posterior distribution
        batch_size = prior_ensemble.shape[0]
        subkeys = jax.random.split(key, batch_size)
        posterior_ensemble = eqx.filter_vmap(
            lambda k: jax.random.multivariate_normal(k, posterior_mean, posterior_cov)
        )(subkeys)
        
        if self.debug:
            assert posterior_ensemble.shape == prior_ensemble.shape
            
        return posterior_ensemble

###

import equinox as eqx
import jax
import jax.numpy as jnp

from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem


# class UKF(AbstractFilter, strict=True):
#     alpha: float = 1e-3
#     beta: float = 2.0
#     kappa: float = 0.0
#     regularization: float = 1e-6
#     debug: bool = False
    
#     @eqx.filter_jit  
#     def generate_sigma_points(self, mean, covariance):
#         n = mean.shape[0]
#         lambda_param = self.alpha**2 * (n + self.kappa) - n
        
#         # Regularize covariance before Cholesky
#         reg_cov = covariance + self.regularization * jnp.eye(n)
#         reg_cov = (reg_cov + reg_cov.T) / 2
        
#         # Robust Cholesky with eigendecomposition fallback
#         eigenvals, eigenvecs = jnp.linalg.eigh(reg_cov)
#         eigenvals = jnp.maximum(eigenvals, 1e-10)
#         L = eigenvecs @ jnp.diag(jnp.sqrt((n + lambda_param) * eigenvals))
        
#         sigma_points = jnp.zeros((2*n + 1, n))
#         sigma_points = sigma_points.at[0].set(mean)
#         sigma_points = sigma_points.at[1:n+1].set(mean[None, :] + L)
#         sigma_points = sigma_points.at[n+1:2*n+1].set(mean[None, :] - L)
        
#         weights_m = jnp.zeros(2*n + 1)
#         weights_c = jnp.zeros(2*n + 1)
        
#         weights_m = weights_m.at[0].set(lambda_param / (n + lambda_param))
#         weights_c = weights_c.at[0].set(lambda_param / (n + lambda_param) + (1 - self.alpha**2 + self.beta))
        
#         weight_other = 1.0 / (2 * (n + lambda_param))
#         weights_m = weights_m.at[1:].set(weight_other)
#         weights_c = weights_c.at[1:].set(weight_other)
        
#         return sigma_points, weights_m, weights_c

#     @eqx.filter_jit
#     def predict(self, posterior_ensemble, dynamical_system, initial_time, final_time):
#         """UKF Predict Step: Propagate through dynamics using sigma points"""
#         # Convert ensemble to mean and covariance
#         posterior_mean = jnp.mean(posterior_ensemble, axis=0)
#         ensemble_deviations = posterior_ensemble - posterior_mean[None, :]
#         posterior_cov = (ensemble_deviations.T @ ensemble_deviations) / (posterior_ensemble.shape[0] - 1)
#         posterior_cov = posterior_cov + self.regularization * jnp.eye(posterior_cov.shape[0])
        
#         # Generate sigma points
#         sigma_points, weights_m, weights_c = self.generate_sigma_points(posterior_mean, posterior_cov)
        
#         # Propagate sigma points through dynamics
#         propagated_sigma_points = eqx.filter_vmap(dynamical_system.flow)(
#             initial_time, final_time, sigma_points
#         )
        
#         # Compute predicted mean and covariance
#         predicted_mean = jnp.sum(weights_m[:, None] * propagated_sigma_points, axis=0)
        
#         residuals = propagated_sigma_points - predicted_mean[None, :]
#         predicted_cov = jnp.sum(
#             weights_c[:, None, None] * residuals[:, :, None] * residuals[:, None, :], 
#             axis=0
#         )
        
#         # Ensure positive definiteness
#         predicted_cov = (predicted_cov + predicted_cov.T) / 2
#         eigenvals, eigenvecs = jnp.linalg.eigh(posterior_cov)
#         eigenvals = jnp.maximum(eigenvals, self.regularization)
#         predicted_cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
#         return predicted_mean, predicted_cov

#     @eqx.filter_jit
#     def update(self, key, prior_ensemble, measurement, measurement_system):
#         """UKF Update Step: This now assumes prior_ensemble represents predicted distribution"""
#         # This is a compatibility method - convert ensemble to mean/cov
#         prior_mean = jnp.mean(prior_ensemble, axis=0)
#         ensemble_deviations = prior_ensemble - prior_mean[None, :]
#         prior_cov = (ensemble_deviations.T @ ensemble_deviations) / (prior_ensemble.shape[0] - 1)
#         prior_cov = prior_cov + self.regularization * jnp.eye(prior_cov.shape[0])
        
#         # Generate sigma points for measurement update
#         sigma_points, weights_m, weights_c = self.generate_sigma_points(prior_mean, prior_cov)
        
#         # Propagate through measurement function
#         predicted_measurements = eqx.filter_vmap(measurement_system)(sigma_points)
        
#         # Compute measurement statistics
#         measurement_mean = jnp.sum(weights_m[:, None] * predicted_measurements, axis=0)
        
#         innovation_residuals = predicted_measurements - measurement_mean[None, :]
#         innovation_cov = jnp.sum(
#             weights_c[:, None, None] * innovation_residuals[:, :, None] * innovation_residuals[:, None, :], 
#             axis=0
#         ) + measurement_system.covariance
        
#         # Cross-covariance
#         state_residuals = sigma_points - prior_mean[None, :]
#         cross_cov = jnp.sum(
#             weights_c[:, None, None] * state_residuals[:, :, None] * innovation_residuals[:, None, :],
#             axis=0
#         )
        
#         # Kalman gain and update
#         kalman_gain = jax.scipy.linalg.solve(innovation_cov, cross_cov.T).T
#         innovation = measurement - measurement_mean
        
#         posterior_mean = prior_mean + kalman_gain @ innovation
        
#         # Joseph form covariance update
#         posterior_cov = prior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        
#         # Ensure positive definiteness
#         posterior_cov = (posterior_cov + posterior_cov.T) / 2
#         eigenvals, eigenvecs = jnp.linalg.eigh(posterior_cov)
#         eigenvals = jnp.maximum(eigenvals, self.regularization)
#         posterior_cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
#         # Convert back to ensemble
#         batch_size = prior_ensemble.shape[0]
#         subkeys = jax.random.split(key, batch_size)
#         posterior_ensemble = eqx.filter_vmap(
#             lambda k: jax.random.multivariate_normal(k, posterior_mean, posterior_cov)
#         )(subkeys)
            
#         return posterior_ensemble

#     @eqx.filter_jit  
#     def predict_update_cycle(self, key, posterior_ensemble, measurement, 
#                            dynamical_system, measurement_system, 
#                            initial_time, final_time):
#         """Complete UKF cycle: predict through dynamics, then update with measurement"""
#         # Predict step
#         predicted_mean, predicted_cov = self.predict(
#             posterior_ensemble, dynamical_system, initial_time, final_time
#         )
        
#         # Convert to ensemble for update step (temporary)
#         batch_size = posterior_ensemble.shape[0]
#         subkeys = jax.random.split(key, batch_size + 1)
#         key, *sample_keys = subkeys
#         predicted_ensemble = eqx.filter_vmap(
#             lambda k: jax.random.multivariate_normal(k, predicted_mean, predicted_cov)
#         )(jnp.array(sample_keys))
        
#         # Update step
#         return self.update(key, predicted_ensemble, measurement, measurement_system)

###

import equinox as eqx
import jax
import jax.numpy as jnp

from xradar_uq.stochastic_filters import AbstractFilter
from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.dynamical_systems import AbstractDynamicalSystem


class UKF(AbstractFilter, strict=True):
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0
    regularization: float = 1e-6
    debug: bool = False
    
    @eqx.filter_jit  
    def generate_sigma_points(self, mean, covariance):
        n = mean.shape[0]
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        # Regularize covariance
        reg_cov = covariance + self.regularization * jnp.eye(n)
        reg_cov = (reg_cov + reg_cov.T) / 2
        
        # Robust matrix square root
        eigenvals, eigenvecs = jnp.linalg.eigh(reg_cov)
        eigenvals = jnp.maximum(eigenvals, 1e-10)
        L = eigenvecs @ jnp.diag(jnp.sqrt((n + lambda_param) * eigenvals))
        
        sigma_points = jnp.zeros((2*n + 1, n))
        sigma_points = sigma_points.at[0].set(mean)
        sigma_points = sigma_points.at[1:n+1].set(mean[None, :] + L)
        sigma_points = sigma_points.at[n+1:2*n+1].set(mean[None, :] - L)
        
        return sigma_points

    @eqx.filter_jit
    def get_sigma_weights(self, n):
        lambda_param = self.alpha**2 * (n + self.kappa) - n
        
        weights_m = jnp.zeros(2*n + 1)
        weights_c = jnp.zeros(2*n + 1)
        
        weights_m = weights_m.at[0].set(lambda_param / (n + lambda_param))
        weights_c = weights_c.at[0].set(lambda_param / (n + lambda_param) + (1 - self.alpha**2 + self.beta))
        
        weight_other = 1.0 / (2 * (n + lambda_param))
        weights_m = weights_m.at[1:].set(weight_other)
        weights_c = weights_c.at[1:].set(weight_other)
        
        return weights_m, weights_c

    @eqx.filter_jit
    def ensemble_to_sigma_points(self, ensemble):
        """Convert ensemble to sigma points (for initialization)"""
        mean = jnp.mean(ensemble, axis=0)
        deviations = ensemble - mean[None, :]
        cov = (deviations.T @ deviations) / (ensemble.shape[0] - 1)
        cov = cov + self.regularization * jnp.eye(cov.shape[0])
        return self.generate_sigma_points(mean, cov)

    @eqx.filter_jit
    def sigma_points_to_statistics(self, sigma_points):
        """Convert sigma points to mean and covariance"""
        n = sigma_points.shape[1]
        weights_m, weights_c = self.get_sigma_weights(n)
        
        mean = jnp.sum(weights_m[:, None] * sigma_points, axis=0)
        residuals = sigma_points - mean[None, :]
        cov = jnp.sum(
            weights_c[:, None, None] * residuals[:, :, None] * residuals[:, None, :], 
            axis=0
        )
        
        # Ensure positive definiteness
        cov = (cov + cov.T) / 2
        eigenvals, eigenvecs = jnp.linalg.eigh(cov)
        eigenvals = jnp.maximum(eigenvals, self.regularization)
        cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        return mean, cov

    @eqx.filter_jit
    def predict(self, posterior_ensemble, dynamical_system, initial_time, final_time):
        """
        UKF Predict Step: Returns sigma points as 'ensemble'
        
        For compatibility: if input is not sigma points, convert ensemble to sigma points first
        """
        n = posterior_ensemble.shape[1]
        expected_sigma_points = 2 * n + 1
        
        # Check if input is already sigma points or needs conversion
        if posterior_ensemble.shape[0] == expected_sigma_points:
            # Input is already sigma points
            sigma_points = posterior_ensemble
        else:
            # Convert ensemble to sigma points
            sigma_points = self.ensemble_to_sigma_points(posterior_ensemble)
        
        # Propagate sigma points through dynamics
        propagated_sigma_points = eqx.filter_vmap(dynamical_system.flow)(
            initial_time, final_time, sigma_points
        )
        
        return propagated_sigma_points

    @eqx.filter_jit
    def update(self, key, prior_ensemble, measurement, measurement_system):
        """
        UKF Update Step: Takes sigma points as 'ensemble', returns new sigma points
        
        prior_ensemble should be sigma points from predict step
        """
        # Treat prior_ensemble as sigma points
        sigma_points = prior_ensemble
        n = sigma_points.shape[1]
        weights_m, weights_c = self.get_sigma_weights(n)
        
        # Propagate through measurement function  
        predicted_measurements = eqx.filter_vmap(measurement_system)(sigma_points)
        
        # Compute measurement statistics
        measurement_mean = jnp.sum(weights_m[:, None] * predicted_measurements, axis=0)
        
        innovation_residuals = predicted_measurements - measurement_mean[None, :]
        innovation_cov = jnp.sum(
            weights_c[:, None, None] * innovation_residuals[:, :, None] * innovation_residuals[:, None, :], 
            axis=0
        ) + measurement_system.covariance
        
        # Cross-covariance
        prior_mean = jnp.sum(weights_m[:, None] * sigma_points, axis=0)
        state_residuals = sigma_points - prior_mean[None, :]
        cross_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * innovation_residuals[:, None, :],
            axis=0
        )
        
        # Kalman update
        kalman_gain = jax.scipy.linalg.solve(innovation_cov, cross_cov.T).T
        innovation = measurement - measurement_mean
        
        posterior_mean = prior_mean + kalman_gain @ innovation
        
        # Get prior covariance for update
        prior_cov = jnp.sum(
            weights_c[:, None, None] * state_residuals[:, :, None] * state_residuals[:, None, :], 
            axis=0
        )
        prior_cov = prior_cov + self.regularization * jnp.eye(prior_cov.shape[0])
        
        # Covariance update
        posterior_cov = prior_cov - kalman_gain @ innovation_cov @ kalman_gain.T
        
        # Ensure positive definiteness
        posterior_cov = (posterior_cov + posterior_cov.T) / 2
        eigenvals, eigenvecs = jnp.linalg.eigh(posterior_cov)
        eigenvals = jnp.maximum(eigenvals, self.regularization)
        posterior_cov = eigenvecs @ jnp.diag(eigenvals) @ eigenvecs.T
        
        # Generate new sigma points from posterior
        posterior_sigma_points = self.generate_sigma_points(posterior_mean, posterior_cov)
        
        return posterior_sigma_points

###

# import equinox as eqx
# import jax
# import jax.numpy as jnp

# from xradar_uq.dynamical_systems import CR3BP
# from xradar_uq.measurement_systems import DeepSpaceNetwork
# # from xradar_uq.stochastic_filters import UKF  # Your new UKF implementation

# key = jax.random.key(42)
# time_step = 0.12

# dynamical_system = CR3BP()
# measurement_system = DeepSpaceNetwork()
# stochastic_filter = UKF()

# true_state = dynamical_system.initial_state() 
# key, subkey = jax.random.split(key)
# posterior_ensemble = dynamical_system.generate(subkey, batch_size=1000) 

# errors = []

# for _ in range(10):
#     # Propagate true state
#     true_state = dynamical_system.flow(0.0, time_step, true_state)
    
#     # Take measurement
#     key, subkey = jax.random.split(key)
#     measurement = measurement_system(true_state, subkey)

#     # UKF predict and update cycle (proper sigma point propagation)
#     key, subkey = jax.random.split(key)
#     posterior_ensemble = stochastic_filter.predict_update_cycle(
#         subkey, posterior_ensemble, measurement, 
#         dynamical_system, measurement_system, 
#         0.0, time_step
#     )
    
#     error = true_state - jnp.mean(posterior_ensemble, axis=0)
#     errors.append(error)

# rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
# rmse

# Alternative: Manual predict/update steps
# for _ in range(10):
#     true_state = dynamical_system.flow(0.0, time_step, true_state)
#     
#     # UKF Predict: sigma points through dynamics
#     predicted_mean, predicted_cov = stochastic_filter.predict(
#         posterior_ensemble, dynamical_system, 0.0, time_step
#     )
#     
#     # Convert to ensemble for measurement update
#     key, subkey = jax.random.split(key)
#     subkeys = jax.random.split(subkey, 1000)
#     predicted_ensemble = jax.vmap(
#         lambda k: jax.random.multivariate_normal(k, predicted_mean, predicted_cov)
#     )(subkeys)
#     
#     # UKF Update: sigma points through measurement
#     key, subkey = jax.random.split(key)
#     measurement = measurement_system(true_state, subkey)
#     
#     key, subkey = jax.random.split(key)
#     posterior_ensemble = stochastic_filter.update(
#         subkey, predicted_ensemble, measurement, measurement_system
#     )

###


key = jax.random.key(42)
time_step = 0.12

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()
stochastic_filter = UKF()

true_state = dynamical_system.initial_state() 
key, subkey = jax.random.split(key)
posterior_ensemble = dynamical_system.generate(subkey, batch_size=13) 

# errors = []

# for _ in range(10):
#     true_state = dynamical_system.flow(0.0, time_step, true_state)
#     prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_step, posterior_ensemble)

#     key, subkey = jax.random.split(key)
#     measurement = measurement_system(true_state, subkey)

#     key, subkey = jax.random.split(key)
#     posterior_ensemble = stochastic_filter.update(key, prior_ensemble, measurement, measurement_system)
#     error = true_state - jnp.mean(posterior_ensemble, axis=0)
#     errors.append(error)

# rmse = jnp.sqrt(jnp.mean(jnp.asarray(errors) ** 2))
# rmse

# Define parameter ranges
# delta_v_range = jnp.logspace(-3, -1, 20)
# maneuver_proportion_range = jnp.linspace(0, 0.2, 20)
# from xradar_uq.evaluate import evaluate_tracking_grid
# delta_v_range = jnp.logspace(-3, 0, 2)
# maneuver_proportion_range = jnp.linspace(0, 0.5, 2)

# # Run optimized computation
# key = jax.random.key(42)
# results = evaluate_tracking_grid(
#     delta_v_range, 
#     maneuver_proportion_range, 
#     key,
#     dynamical_system, 
#     measurement_system, 
#     stochastic_filter,
#     mc_iterations=1
# )

key = jax.random.key(42)

delta_v_range = jnp.linspace(0.001, 0.5, 2) # 0.5
maneuver_proportion_range = jnp.linspace(0.0, 1.0, 2) # 0.5

# dynamical_system = CR3BP()
# measurement_system = DeepSpaceNetwork()
# stochastic_filter = EnGMF()


# results = evaluate_tracking_grid(
#     delta_v_range, maneuver_proportion_range, key,
#     dynamical_system, measurement_system, stochastic_filter,
#     single_sensor_tracking, mc_iterations = 1
# )

# results

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from xradar_uq.dynamical_systems import CR3BP
from xradar_uq.evaluate import evaluate_tracking_grid
from xradar_uq.measurement_systems import DeepSpaceNetwork
from xradar_uq.statistics import GMM
from xradar_uq.stochastic_filters import EnGMF
from xradar_uq.measurement_systems import AnglesOnly, simulate_thrust

key = jax.random.key(42)

delta_v_range = jnp.linspace(0.001, 0.5, 20) # 0.5
maneuver_proportion_range = jnp.linspace(0.0, 1.0, 10) # 0.5

dynamical_system = CR3BP()
measurement_system = DeepSpaceNetwork()
stochastic_filter = UKF()

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit  
def select_pdf_weighted_candidate(
    key, candidates: Float[Array, "n_candidates 2"], gmm: GMM
) -> Float[Array, "2"]:
    finite_mask = jnp.isfinite(candidates).all(axis=1)
    
    valid_candidates = jnp.where(finite_mask[:, None], candidates, 0.0)
    pdf_weights = eqx.filter_vmap(gmm.positional_pdf)(valid_candidates)
    masked_weights = jnp.where(finite_mask, pdf_weights, -jnp.inf)
    optimal_idx = jnp.argmax(masked_weights)
    selected_position = candidates[optimal_idx]
    assert selected_position.shape == (2,)
    return selected_position

@eqx.filter_jit
def place_sensors(
    key: jax.Array, candidates: Float[Array, "n_candidates 2"],
    n_sensors: int, exclusion_radius: float, selection: Callable
) -> Float[Array, "n_sensors 2"]:
    sensor_positions = jnp.zeros((n_sensors, 2))
    placement_mask = jnp.zeros(n_sensors, dtype=bool)
    
    def place_single_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, subkey = jax.random.split(key_state)
        
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_distances = jnp.where(mask[None, :], distances, jnp.inf)
        min_distances = jnp.min(valid_distances, axis=1)
        valid_candidate_mask = min_distances > exclusion_radius
        
        valid_candidates = jnp.where(valid_candidate_mask[:, None], candidates, jnp.inf)
        new_position = selection(subkey, valid_candidates)
        
        updated_positions = positions.at[i].set(new_position)
        updated_mask = mask.at[i].set(True)
        
        return (updated_positions, updated_mask, key_state)
    
    final_positions, _, _ = jax.lax.fori_loop(0, n_sensors, place_single_sensor, (sensor_positions, placement_mask, key))
    return final_positions


import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, jaxtyped
from xradar_uq.geometry import (angular_reachable_set, barycentric_subdivision,
                                fan_triangulate)
from xradar_uq.measurement_systems import AnglesOnly
from xradar_uq.statistics import silverman_kde_estimate


@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sensor_tracking_max_pdf(true_state, prior_ensemble, key, posterior_ensemble,
                            dynamical_system, time_horrizon, delta_v_magnitude, num_simulations=100):
    true_angles = AnglesOnly()(true_state)
    hull = angular_reachable_set(key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    triangles = barycentric_subdivision(barycentric_subdivision(fan_triangulate(hull)))
    candidates = jnp.mean(triangles, axis=1)
    gmm = silverman_kde_estimate(posterior_ensemble)
    
    sensor_positions = jnp.zeros((3, 2))
    placement_mask = jnp.zeros(3, dtype=bool)
    
    def place_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, _ = jax.random.split(key_state)
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_mask = jnp.min(jnp.where(mask[None, :], distances, jnp.inf), axis=1) > jnp.deg2rad(5)
        filtered_candidates = jnp.where(valid_mask[:, None], candidates, 0.0)
        optimal_idx = jax.random.choice(key, jnp.arange(len(filtered_candidates)))
        # optimal_idx = jnp.argmax(eqx.filter_vmap(gmm.positional_pdf)(filtered_candidates))
        return (positions.at[i].set(filtered_candidates[optimal_idx]), mask.at[i].set(True), key_state)
    
    sensor_placement, _, _ = jax.lax.fori_loop(0, 3, place_sensor, (sensor_positions, placement_mask, key))
    assert sensor_placement.shape == (3, 2)
    return jnp.any(jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1) <= jnp.deg2rad(5))

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sensor_tracking_max_augmented_pdf(true_state, prior_ensemble, key, posterior_ensemble,
                            dynamical_system, time_horrizon, delta_v_magnitude, num_simulations=10):
    true_angles = AnglesOnly()(true_state)
    hull = angular_reachable_set(key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    triangles = barycentric_subdivision(barycentric_subdivision(fan_triangulate(hull)))
    candidates = jnp.mean(triangles, axis=1)
    key, thrust_key = jax.random.split(key)
    thrust_ensemble = simulate_thrust(thrust_key, posterior_ensemble, num_simulations, delta_v_magnitude)
    thrust_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, time_horrizon, thrust_ensemble)
    gmm = silverman_kde_estimate(thrust_ensemble)
    
    ensemble_mean_angles = AnglesOnly()(jnp.mean(posterior_ensemble, axis=0))
    sensor_positions = jnp.zeros((3, 2)).at[0].set(ensemble_mean_angles)
    placement_mask = jnp.zeros(3, dtype=bool).at[0].set(True)
    
    def place_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, _ = jax.random.split(key_state)
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_mask = jnp.min(jnp.where(mask[None, :], distances, jnp.inf), axis=1) > jnp.deg2rad(5)
        filtered_candidates = jnp.where(valid_mask[:, None], candidates, 0.0)
        # random_idx = jax.random.choice(key_state, jnp.arange(len(filtered_candidates)))
        random_idx = jnp.argmax(eqx.filter_vmap(gmm.positional_pdf)(filtered_candidates))
        return (positions.at[i].set(filtered_candidates[random_idx]), mask.at[i].set(True), key_state)
    
    sensor_placement, _, _ = jax.lax.fori_loop(1, 3, place_sensor, (sensor_positions, placement_mask, key))
    assert sensor_placement.shape == (3, 2)
    return jnp.any(jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1) <= jnp.deg2rad(5))

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sensor_tracking_max_pdf(true_state, prior_ensemble, key, posterior_ensemble,
                            dynamical_system, time_horrizon, delta_v_magnitude, num_simulations=10):
    true_angles = AnglesOnly()(true_state)
    hull = angular_reachable_set(key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    triangles = barycentric_subdivision(barycentric_subdivision(fan_triangulate(hull)))
    candidates = jnp.mean(triangles, axis=1)
    key, thrust_key = jax.random.split(key)
    gmm = silverman_kde_estimate(prior_ensemble)
    
    ensemble_mean_angles = AnglesOnly()(jnp.mean(posterior_ensemble, axis=0))
    sensor_positions = jnp.zeros((3, 2)).at[0].set(ensemble_mean_angles)
    placement_mask = jnp.zeros(3, dtype=bool).at[0].set(True)
    
    def place_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, _ = jax.random.split(key_state)
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_mask = jnp.min(jnp.where(mask[None, :], distances, jnp.inf), axis=1) > jnp.deg2rad(5)
        filtered_candidates = jnp.where(valid_mask[:, None], candidates, 0.0)
        # random_idx = jax.random.choice(key_state, jnp.arange(len(filtered_candidates)))
        random_idx = jnp.argmax(eqx.filter_vmap(gmm.positional_pdf)(filtered_candidates))
        return (positions.at[i].set(filtered_candidates[random_idx]), mask.at[i].set(True), key_state)
    
    sensor_placement, _, _ = jax.lax.fori_loop(1, 3, place_sensor, (sensor_positions, placement_mask, key))
    assert sensor_placement.shape == (3, 2)
    return jnp.any(jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1) <= jnp.deg2rad(5))

@jaxtyped(typechecker=typechecker)
@eqx.filter_jit
def sensor_tracking_random(true_state, prior_ensemble, key, posterior_ensemble,
                            dynamical_system, time_horrizon, delta_v_magnitude, num_simulations=10):
    true_angles = AnglesOnly()(true_state)
    hull = angular_reachable_set(key, posterior_ensemble, time_horrizon, delta_v_magnitude, num_simulations, dynamical_system)
    triangles = barycentric_subdivision(barycentric_subdivision(fan_triangulate(hull)))
    candidates = jnp.mean(triangles, axis=1)
    key, thrust_key = jax.random.split(key)
    gmm = silverman_kde_estimate(prior_ensemble)
    
    ensemble_mean_angles = AnglesOnly()(jnp.mean(posterior_ensemble, axis=0))
    sensor_positions = jnp.zeros((3, 2)).at[0].set(ensemble_mean_angles)
    placement_mask = jnp.zeros(3, dtype=bool).at[0].set(True)
    
    def place_sensor(i, carry):
        positions, mask, key_state = carry
        key_state, _ = jax.random.split(key_state)
        distances = jnp.linalg.norm(candidates[:, None] - positions[None], axis=2)
        valid_mask = jnp.min(jnp.where(mask[None, :], distances, jnp.inf), axis=1) > jnp.deg2rad(5)
        filtered_candidates = jnp.where(valid_mask[:, None], candidates, 0.0)
        random_idx = jax.random.choice(key_state, jnp.arange(len(filtered_candidates)))
        # random_idx = jnp.argmax(eqx.filter_vmap(gmm.positional_pdf)(filtered_candidates))
        return (positions.at[i].set(filtered_candidates[random_idx]), mask.at[i].set(True), key_state)
    
    sensor_placement, _, _ = jax.lax.fori_loop(1, 3, place_sensor, (sensor_positions, placement_mask, key))
    assert sensor_placement.shape == (3, 2)
    return jnp.any(jnp.linalg.norm(sensor_placement - true_angles[None, :], axis=1) <= jnp.deg2rad(5))


# key, subkey = jax.random.split(key)
# posterior_ensemble = dynamical_system.generate(key)
# prior_ensemble = eqx.filter_vmap(dynamical_system.flow)(0.0, 0.24, posterior_ensemble)
# true_state = dynamical_system.initial_state()
# true_state = dynamical_system.flow(0.0, 0.24, true_state)
# sensor_tracking_max_pdf(true_state, posterior_ensemble, subkey, posterior_ensemble, dynamical_system, 0.24, 1e-5)


results_single = evaluate_tracking_grid(
    delta_v_range, maneuver_proportion_range, key,
    dynamical_system, measurement_system, stochastic_filter,
    sensor_tracking_max_pdf, mc_iterations = 1
)
results_single

# df = pd.DataFrame(
#     jnp.mean(results_single, axis=-1),
#     index=((389703 / 382981) * delta_v_range),
#     columns=maneuver_proportion_range
# )

# import pandas as pd
# results_single.shape
# df = pd.DataFrame(
#     jnp.mean(results_single, axis=-1),
#     index=((389703 / 382981) * delta_v_range),
#     columns=maneuver_proportion_range
# )
# df.to_csv('~/mc_1_random_with_prior_kde_max.csv')
