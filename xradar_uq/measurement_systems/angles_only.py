import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Bool, Float, Key, jaxtyped

from xradar_uq.measurement_systems import AbstractMeasurementSystem
from xradar_uq.statistics.random_finite_sets import RFS


class AnglesOnly(AbstractMeasurementSystem, strict=True):
    """
    This measurement system takes in an array whose first three indices are the spatial dimensions.
    Then returns the distance, elevation and azimuth.
    """

    covariance: Float[Array, "2"] = eqx.field(
        default_factory=lambda: jnp.diag(jnp.array([(0.1 * jnp.pi / 180) ** 2,
                                                    (0.1 * jnp.pi / 180) ** 2]))
    )

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        positions: Float[Array, "3"] | Float[Array, "6"],
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "2"]:
        x, y, z = positions[0], positions[1], positions[2]
        alpha = jnp.arctan2(y, x)
        epsilon = jnp.arcsin(z / jnp.sqrt(x**2 + y**2 + z**2))
        measurements = jnp.array([alpha, epsilon])

        if key is not None:
            measurements = jax.random.multivariate_normal(
                key, mean=measurements, cov=self.covariance
            )

        return measurements
