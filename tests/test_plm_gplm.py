"""PLM / GPLM tests. The r_* functions are line-by-line ports of the R code
(main_2tau.R, main_2tau_deviance.R, local_scoring.R, examples.R) used as references."""

import numpy as np
import pytest
from scipy.stats import norm

from rsoselect import GPLM, PLM
from rsoselect.smoothers import local_linear


# ------------------------------------------------------------- R reference ports
def r_DD(Z):
    return [np.abs(Z[:, [j]] - Z[:, j][None, :]) for j in range(Z.shape[1])]


def r_getLCSmat(DD, lam, W0=None):
    temp = sum(D * l for D, l in zip(DD, lam))
    K = np.exp(-temp ** 2)
    if W0 is not None:                       # matrix(rep(W0, n), ncol=n, byrow=FALSE)
        K = K * np.repeat(W0[:, None], len(W0), axis=1)
    return K / K.sum(1, keepdims=True)


def r_getHmat(x, lamX, S, W0=None):
    n, p = x.shape
    dW = np.eye(n) if W0 is None else np.diag(np.sqrt(W0))
    I = np.eye(n)
    ImS = I - S
    ImSx = ImS @ x
    Ds = np.diag(np.sqrt(lamX))
    B = (ImSx.T @ dW @ ImS) / n
    T = ImSx @ Ds
    A = Ds @ np.linalg.solve(T.T @ dW @ T / n + np.eye(p), Ds)
    H = ImS @ (I - x @ A @ B)
    return H, A, B


def r_objfun_plm(X, DD, y, lam):
    p = X.shape[1]
    S = r_getLCSmat(DD, lam[p:])
    H, _, _ = r_getHmat(X, lam[:p], S)
    return np.sum((H @ y) ** 2)


def r_MCD_plm(X, DD, y, tau1, tau2):
    eps1 = 1e-4; eps0 = 2 * eps1
    p, q = X.shape[1], len(DD)
    g, l = np.full(p, tau1 / p), np.full(q, tau2 / q)
    grid = [0, 0.2, 0.5, 0.8, 1]
    hist, objs = [], []
    while True:
        go, lo = g.copy(), l.copy()
        for i in range(p):
            g1 = g.copy(); g1[i] = 0
            if g1.sum() > eps0:
                g1 /= g1.sum(); e = np.zeros(p); e[i] = 1
                v = [r_objfun_plm(X, DD, y, np.r_[(m * g1 + (1 - m) * e) * tau1, l]) for m in grid]
                m = grid[int(np.argmin(v))]; g = (m * g1 + (1 - m) * e) * tau1
        for j in range(q):
            l1 = l.copy(); l1[j] = 0
            if l1.sum() > eps0:
                l1 /= l1.sum(); e = np.zeros(q); e[j] = 1
                v = [r_objfun_plm(X, DD, y, np.r_[g, (m * l1 + (1 - m) * e) * tau2]) for m in grid]
                m = grid[int(np.argmin(v))]; l = (m * l1 + (1 - m) * e) * tau2
        hist.append(np.r_[g, l]); objs.append(r_objfun_plm(X, DD, y, np.r_[g, l]))
        if len(hist) > 4:
            return hist[int(np.argmin(objs))]
        if max(np.abs(lo - l).max(), np.abs(go - g).max()) < eps1:
            return np.r_[g, l]


def r_LLsmoothing(z, zt, lam):
    n, q = z.shape
    rows = []
    for i in range(len(zt)):
        temp = z - zt[i]
        XX = np.column_stack([np.ones(n), temp])
        K = np.exp(-np.sum((temp * lam) ** 2, 1))
        M = (XX * K[:, None]).T @ XX
        rows.append(np.linalg.solve(M, (XX * K[:, None]).T)[0])
    return np.array(rows)


def r_family(ex):
    if ex == "poisson":
        return dict(g=np.log, ginv=np.exp, ginvp=np.exp, gp=lambda t: 1 / t,
                    var=lambda e: np.exp(e))
    s = lambda e: np.exp(e) / (1 + np.exp(e))
    return dict(g=lambda t: np.log(t / (1 - t)), ginv=s, ginvp=lambda e: s(e) * (1 - s(e)),
                gp=lambda t: 1 / (t * (1 - t)), var=lambda e: s(e) * (1 - s(e)))


def r_local_scoring_eta(X, DD, Y, lam, fam):
    F = r_family(fam)
    n, p = X.shape
    Ybar = np.full(n, Y.mean())
    eta = F["g"](Ybar); mu = Ybar
    W = F["ginvp"](eta) ** 2 / F["var"](eta); Wpre = W / W.sum()
    Z = eta + (Y - mu) * F["gp"](mu)
    nloop = 0
    while True:
        S = r_getLCSmat(DD, lam[p:], Wpre)
        H, A, B = r_getHmat(X, lam[:p], S, Wpre)
        beta = A @ B @ Z
        eta = X @ beta + S @ (Z - X @ beta)
        mu = F["ginv"](eta)
        W = F["ginvp"](eta) ** 2 / F["var"](eta); Wst = W / W.sum()
        Z = eta + (Y - mu) * F["gp"](mu)
        if (Wst - Wpre).max() < 1e-5 or nloop >= 15:
            break
        Wpre = Wst; nloop += 1
    S = r_getLCSmat(DD, lam[p:], Wst)                     # objfun()
    H, A, B = r_getHmat(X, lam[:p], S, Wst)
    beta = A @ B @ Z
    return X @ beta + S @ (Z - X @ beta)


# --------------------------------------------------------------------- data
def gen_plm(rng, n=100, p=5, q=3):
    d = p + q
    raw = rng.multivariate_normal(np.zeros(d), 0.5 ** np.abs(np.subtract.outer(range(d), range(d))), n)
    x, z = raw[:, :p], 2 * norm.cdf(raw[:, p:]) - 1
    b = np.zeros(p); b[[1, 3]] = 1
    return x, z, x @ b + np.sqrt(2) * np.sin(z[:, 0] * np.pi) + rng.standard_normal(n)


def gen_gplm(rng, fam, n=100, p=4, q=3):
    d = p + q
    raw = rng.multivariate_normal(np.zeros(d), 0.5 ** np.abs(np.subtract.outer(range(d), range(d))), n)
    x, u = raw[:, :p], 2 * norm.cdf(raw[:, p:]) - 1
    b = np.zeros(p); b[[1, 3]] = 1
    eta = x @ b + np.sqrt(2) * np.sin(np.pi * u[:, 0])
    y = rng.poisson(np.exp(eta)) if fam == "poisson" else rng.binomial(1, 1 / (1 + np.exp(-eta)))
    return x, u, y.astype(float)


# -------------------------------------------------------------------- tests
def test_plm_objective_matches_R():
    rng = np.random.default_rng(0)
    x, z, y = gen_plm(rng, n=60)
    m = PLM(r_compat=True); m._prepare(x, z, y)
    th = np.r_[[0.0, 3.0, 0.5, 2.0, 1.0], [1.5, 0.0, 0.7]]
    assert np.isclose(m._objective(th), r_objfun_plm(x, r_DD(z), y, th))
    H, _, _ = r_getHmat(x, th[:5], r_getLCSmat(r_DD(z), th[5:]))
    assert np.isclose(m._naive_stats(th[:5], th[5:])[2], 60 - np.trace(H))


@pytest.mark.parametrize("tau1,tau2", [(5.0, 1.0), (20.0, 2.6), (44.0, 3.0)])
def test_plm_mcd_matches_R(tau1, tau2):
    rng = np.random.default_rng(1)
    x, z, y = gen_plm(rng, n=60)
    m = PLM(r_compat=True, taus1=[tau1], taus2=[tau2], criterion="naive_bic").fit(x, z, y)
    assert np.allclose(np.r_[m.gamma_, m.lambda_], r_MCD_plm(x, r_DD(z), y, tau1, tau2))


def test_local_linear_matches_R():
    rng = np.random.default_rng(2)
    z, zt = rng.uniform(-1, 1, (40, 2)), rng.uniform(-1, 1, (6, 2))
    lam = np.array([1.2, 0.4])
    assert np.allclose(local_linear(zt, z, lam), r_LLsmoothing(z, zt, lam), atol=1e-7)


@pytest.mark.parametrize("fam", ["poisson", "binomial"])
def test_gplm_local_scoring_matches_R(fam):
    rng = np.random.default_rng(3)
    x, u, y = gen_gplm(rng, fam, n=80)
    m = GPLM(family=fam, r_compat=True); m._prepare(x, u, y)
    th = np.r_[[0.5, 2.0, 0.0, 1.5], [1.0, 0.0, 0.8]]
    eta = m._fit_full(th)["eta"]
    assert np.allclose(eta, r_local_scoring_eta(x, r_DD(u), y, th, fam), atol=1e-8)


def test_gaussian_gplm_equals_plm_path():
    rng = np.random.default_rng(4)
    x, z, y = gen_plm(rng, n=80)
    kw = dict(taus1=[2.0, 10.0], taus2=[1.0, 3.0], criterion="naive_bic", refit=False)
    a = PLM(**kw).fit(x, z, y)
    b = GPLM(family="gaussian", **kw).fit(x, z, y)
    for ra, rb in zip(a.path_, b.path_):
        assert np.allclose(ra["gamma"], rb["gamma"]) and np.allclose(ra["lam"], rb["lam"])


def test_plm_selects_true_model_and_predicts():
    rng = np.random.default_rng(1)
    x, z, y = gen_plm(rng, n=150)
    m = PLM(n_taus1=8, n_taus2=5).fit(x, z, y)
    assert list(np.flatnonzero(m.support_x_)) == [1, 3]
    assert list(np.flatnonzero(m.support_z_)) == [0]
    xt, zt, _ = gen_plm(rng, n=500)
    truth = xt[:, 1] + xt[:, 3] + np.sqrt(2) * np.sin(zt[:, 0] * np.pi)
    assert np.mean((m.predict(xt, zt) - truth) ** 2) < 0.3


def test_gplm_poisson_selects_true_model():
    rng = np.random.default_rng(5)
    x, u, y = gen_gplm(rng, "poisson", n=150)
    m = GPLM(family="poisson", n_taus1=5, n_taus2=3).fit(x, u, y)
    assert list(np.flatnonzero(m.support_x_)) == [1, 3]
    assert list(np.flatnonzero(m.support_z_)) == [0]
    assert np.all(m.predict(x, u) > 0)
