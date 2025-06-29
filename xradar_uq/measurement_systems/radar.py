import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped
from xradar_uq.measurement_systems import AbstractMeasurementSystem


class Radar(AbstractMeasurementSystem, strict=True):
    """
    This measurement system takes in an array whose first three indices are the spatial dimensions.
    Then returns the distance, elevation and azimuth.
    """

    covariance: Float[Array, "3 3"] = eqx.field(
        default_factory=lambda: jnp.diag(jnp.array([(0.5) ** 2,
                                                    (0.25 * jnp.pi / 180) ** 2,
                                                    (0.25 * jnp.pi / 180) ** 2]))
    )

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        positions: Float[Array, "state_dim"],
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "3"]:
        x, y, z = positions[0], positions[1], positions[2]
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        alpha = jnp.arctan2(y, x)
        epsilon = jnp.arcsin(z / rho)
        measurements = jnp.array([rho, alpha, epsilon])

        if key is not None:
            measurements = jax.random.multivariate_normal(
                key, mean=measurements, cov=self.covariance
            )

        return measurements

class RadarRFS(AbstractMeasurementSystem, strict=True):
    covariance: Float[Array, "..."]

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        positions: RFS,
        key: Key[Array, ""] | None = None,
    ) -> tuple[Float[Array, "max_objects 3"], Bool[Array, "max_objects"]]:
        x, y, z = positions.state[:, 0], positions.state[:, 1], positions.state[:, 2]
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        alpha = jnp.arctan2(y, x)
        epsilon = jnp.arcsin(z / rho)

        measurements = jnp.stack([rho, alpha, epsilon], axis=1)

        if key is None:
            return measurements, jnp.tile(jnp.asarray(True), measurements.shape[0])

        if key is not None:
            noise_key, detection_key = jax.random.split(key)
            noise_std = jnp.array([1.0, jnp.deg2rad(0.5), jnp.deg2rad(0.5)])
            measurements += jax.random.normal(noise_key, measurements.shape) * noise_std

            detected = jax.random.bernoulli(
                detection_key, p=0.98, shape=(positions.state.shape[0],)
            )
            detected = detected & positions.mask
            return measurements, detected
