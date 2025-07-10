import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from diffrax import (AbstractSolver, AbstractStepSizeController,
                     ConstantStepSize, Dopri8, PIDController)
from jaxtyping import Array, Float, Key, jaxtyped

from xradar_uq.dynamical_systems import AbstractContinuousDynamicalSystem


@jaxtyped(typechecker=typechecker)
class CR3BP(AbstractContinuousDynamicalSystem, strict=True):
    ### Dynamical System Parameters
    mu: float = 0.012150584269940 # Sometimes I see, 0.012150585609624?

    ### Solver Parameters
    dt: float = 0.0001
    solver: AbstractSolver = Dopri8()
    stepsize_controller: AbstractStepSizeController = PIDController(rtol=1e-12, atol=1e-14)

    @property
    def dimension(self):
        return 6

    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def initial_state(
        self,
        key: Key[Array, "..."] | None = None,
        **kwargs,
    ) -> Float[Array, "state_dim"]:
        mean = jnp.array([
            1.021339954388544,
            -0.000000045869005,
            -0.181619950369762,
            0.000000617839352,
            -0.101759879771430,
            0.000001049698173]
        )
        cov = 1.0e-08 *jnp.array([
            [0.067741479217036,  -0.000029214433641,   0.000292500436172,   0.000343197998120,  -0.000801894296500,  -0.000076851751508],
            [-0.000029214433641,   0.067949657828148,  -0.000045655889447,   0.000112485276059,   0.002893878948354,  -0.000038999497288],
            [0.000292500436172,  -0.000045655889447,   0.067754170807105,  -0.000931574297640,   0.000434803811832,   0.000042975146838],
            [0.000343197998120,   0.000112485276059,  -0.000931574297640,   0.950650788374193,   0.004879599683572,   0.000839738344685],
            [-0.000801894296500,   0.002893878948354,   0.000434803811832,   0.004879599683572,   0.955575624017479,  -0.002913896437441],
            [-0.000076851751508,  -0.000038999497288,   0.000042975146838,   0.000839738344685,  -0.002913896437441,   0.954675354567578]])
        
        noise = (
            0
            if key is None
            else jax.random.multivariate_normal(
                key, mean=jnp.zeros(self.dimension), cov=cov,
            )
        )

        return mean + noise


    @jaxtyped(typechecker=typechecker)
    @eqx.filter_jit
    def vector_field(self, t, y, args):
        r, v = y[:3], y[3:6]
        re = jnp.sqrt((r[0] + self.mu)**2 + r[1]**2 + r[2]**2)
        rm = jnp.sqrt((r[0] - 1 + self.mu)**2 + r[1]**2 + r[2]**2)
        assert re.shape == () and rm.shape == ()
        return jnp.concatenate([v, jnp.array([
            r[0] + 2*v[1] - (1-self.mu)*(r[0]+self.mu)/re**3 - self.mu*(r[0]-1+self.mu)/rm**3,
            r[1] - 2*v[0] - (1-self.mu)*r[1]/re**3 - self.mu*r[1]/rm**3,
            -(1-self.mu)*r[2]/re**3 - self.mu*r[2]/rm**3])])
