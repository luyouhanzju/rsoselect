"""Exponential-family pieces needed by local scoring (examples.R in the GPLM code)."""

import numpy as np
from scipy.special import expit, gammaln

_EPS = 1e-10


class Family:
    name = None

    def link(self, mu): ...            # g
    def inverse(self, eta): ...        # g^{-1}
    def dmu_deta(self, eta): ...       # (g^{-1})'
    def variance(self, mu): ...        # V
    def loglik(self, y, mu): ...       # sum of log-likelihood contributions

    def clip(self, mu):
        return mu

    def neg2loglik(self, y, mu):
        return -2.0 * self.loglik(y, mu)


class Binomial(Family):
    name = "binomial"

    def clip(self, mu):
        return np.clip(mu, _EPS, 1 - _EPS)

    def link(self, mu):
        mu = self.clip(mu)
        return np.log(mu / (1 - mu))

    def inverse(self, eta):
        return self.clip(expit(eta))

    def dmu_deta(self, eta):
        m = expit(eta)
        return np.maximum(m * (1 - m), _EPS)

    def variance(self, mu):
        mu = self.clip(mu)
        return mu * (1 - mu)

    def loglik(self, y, mu):
        mu = self.clip(mu)
        return float(np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu)))


class Poisson(Family):
    name = "poisson"

    def clip(self, mu):
        return np.maximum(mu, _EPS)

    def link(self, mu):
        return np.log(self.clip(mu))

    def inverse(self, eta):
        return self.clip(np.exp(np.minimum(eta, 700)))

    def dmu_deta(self, eta):
        return self.inverse(eta)

    def variance(self, mu):
        return self.clip(mu)

    def loglik(self, y, mu):
        mu = self.clip(mu)
        return float(np.sum(y * np.log(mu) - mu - gammaln(y + 1)))


class Gaussian(Family):
    """Identity link. -2 loglik is profiled over sigma: n log(RSS / n) + const."""
    name = "gaussian"

    def link(self, mu):
        return mu

    def inverse(self, eta):
        return eta

    def dmu_deta(self, eta):
        return np.ones_like(eta)

    def variance(self, mu):
        return np.ones_like(mu)

    def loglik(self, y, mu):
        n = len(y)
        rss = max(float(np.sum((y - mu) ** 2)), 1e-300)
        return -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)


FAMILIES = {"binomial": Binomial, "logistic": Binomial, "poisson": Poisson,
            "gaussian": Gaussian}


def get_family(family):
    if isinstance(family, Family):
        return family
    try:
        return FAMILIES[family]()
    except KeyError:
        raise ValueError(f"family must be one of {sorted(FAMILIES)} or a Family instance")
