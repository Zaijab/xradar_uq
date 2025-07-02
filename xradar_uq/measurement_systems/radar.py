import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, Key, jaxtyped
from xradar_uq.measurement_systems import AbstractMeasurementSystem

# 

class Radar(AbstractMeasurementSystem, strict=True):
    """
    This measurement system takes in an array whose first three indices are the spatial dimensions.
    Then returns the distance, elevation and azimuth.
    """

    mu: float = 0.012150584269940
    covariance: Float[Array, "3 3"] = eqx.field(
        default_factory=lambda: jnp.diag(jnp.array([(2e-3) ** 2,
                                                    (0.05 * jnp.pi / 180) ** 2,
                                                    (0.05 * jnp.pi / 180) ** 2]))
    )


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def __call__(
        self,
        state: Float[Array, "state_dim"],
        key: Key[Array, ""] | None = None,
    ) -> Float[Array, "3"]:
        
        # Satellite position in barycentric coordinates
        satellite_pos = state[:3]
        
        # Earth position in barycentric coordinates
        earth_pos = jnp.array([-self.mu, 0.0, 0.0])
        
        # Satellite position relative to Earth
        relative_pos = satellite_pos - earth_pos
        x, y, z = relative_pos[0], relative_pos[1], relative_pos[2]
        
        # Range, azimuth, elevation from Earth
        rho = jnp.sqrt(x**2 + y**2 + z**2)
        alpha = jnp.arctan2(y, x)          # Azimuth
        epsilon = jnp.arcsin(z / rho)      # Elevation
        
        measurements = jnp.array([rho, alpha, epsilon])

        if key is not None:
            measurements = jax.random.multivariate_normal(
                key, mean=measurements, cov=self.covariance
            )

        return measurements
