import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
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
