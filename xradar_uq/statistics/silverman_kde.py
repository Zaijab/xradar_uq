import distrax
import jax
import equinox as eqx

class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTrer[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def log_prior(self, x):
        return self.cls(*self.args, **self.kwargs).log_prior(x)

prior = FixedDistrax(distrax.MultivariateNormalDiag, mu, sigma)

@eqx.filter_jit
def silverman_kde_estimate(means):
    n, d = means.shape[0], means.shape[1]
    weights = jnp.ones(n) / n
    silverman_beta = (((4) / (d + 2)) ** ((2) / (d + 4))) #* (n ** ((-2) / (d + 4)))
    covs = jnp.tile(silverman_beta * jnp.cov(means.T), reps=(n, 1, 1))
    components = distrax.MultivariateNormalFullCovariance(loc=means, covariance_matrix=covs)
    return distrax.MixtureSameFamily(
        mixture_distribution=distrax.Categorical(probs=weights),
        components_distribution=components
    )
