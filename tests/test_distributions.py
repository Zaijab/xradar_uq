import jax
import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions


def test_emperical_ditribution() -> None:
    samples = jax.random.normal(jax.random.key(0), (10_000,))
    emperical_dist = tfd.Empirical(samples=samples)
    emperical_dist.mean()


def test_jit_compatibility() -> None:

    @jax.jit
    def f(key):
        samples = jax.random.normal(key, (10_000,))
        emperical_dist = tfd.Empirical(samples=samples)
        return emperical_dist.mean()

    print(f(jax.random.key(0)))

test_jit_compatibility()
