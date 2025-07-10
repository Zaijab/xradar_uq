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
