import numpy as np
import pytest
from sklearn.model_selection import LeaveOneOut, TimeSeriesSplit

from rsoselect import RSO, mcd
from rsoselect.linear import ridge_df, ridge_rss
from rsoselect.tuning import default_taus


def gen(rng, n, p, r, beta, sigma):
    S = r ** np.abs(np.subtract.outer(range(p), range(p)))
    X = rng.multivariate_normal(np.zeros(p), S, n)
    return X, X @ beta + sigma * rng.standard_normal(n)


# ---------- literal ports of the R code (S = 0) -----------------------------
def r_getHmat(x, gam):
    n, p = x.shape
    I = np.eye(n)
    Ds = np.diag(np.sqrt(gam))
    B = (x.T @ I) / n
    T = x @ Ds
    A = Ds @ np.linalg.solve(T.T @ T / n + np.eye(p), Ds)
    H = I - x @ A @ B
    return H, n - np.trace(H)


def r_MCD_linear(X, y, tau1):
    """MCD() from main_2tau.R, gamma loop only."""
    eps1 = 1e-4; eps0 = 2 * eps1
    p = X.shape[1]
    gam = np.full(p, tau1 / p)
    grid = [0, 0.2, 0.5, 0.8, 1]
    obj = lambda g: np.sum((r_getHmat(X, g)[0] @ y) ** 2)
    hist, objs = [], []
    while True:
        old = gam.copy()
        for i in range(p):
            g1 = gam.copy(); g1[i] = 0
            if g1.sum() > eps0:
                g1 = g1 / g1.sum()
                g2 = np.zeros(p); g2[i] = 1
                vals = [obj((m * g1 + (1 - m) * g2) * tau1) for m in grid]
                m = grid[int(np.argmin(vals))]
                gam = (m * g1 + (1 - m) * g2) * tau1
        hist.append(gam.copy()); objs.append(obj(gam))
        if len(hist) > 4:
            return hist[int(np.argmin(objs))]
        if np.max(np.abs(old - gam)) < eps1:
            return gam


# ---------- tests ------------------------------------------------------------
def test_rss_df_match_R():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 5)); y = X[:, 1] + X[:, 3] + rng.standard_normal(80)
    g = np.array([0.0, 3.0, 0.5, 2.0, 0.0])
    H, df_R = r_getHmat(X, g)
    G, b = X.T @ X, X.T @ y
    assert np.isclose(np.sum((H @ y) ** 2), ridge_rss(G, b, y @ y, g, 80))
    assert np.isclose(df_R, ridge_df(G, b, g, 80))


@pytest.mark.parametrize("tau", [1.0, 5.0, 20.0, 44.0])
def test_mcd_matches_R(tau):
    rng = np.random.default_rng(1)
    beta = np.zeros(5); beta[[1, 3]] = 1
    X, y = gen(rng, 100, 5, 0.5, beta, 1.0)
    g_R = r_MCD_linear(X, y, tau)
    G, b, n = X.T @ X, X.T @ y, 100
    g_py, _, _ = mcd(lambda g: ridge_rss(G, b, y @ y, g, n), [(np.arange(5), tau)],
                     keep_best=False)
    assert np.allclose(g_R, g_py)


@pytest.mark.parametrize("criterion", ["profile_bic", "naive_bic", "loo"])
def test_selects_true_model(criterion):
    rng = np.random.default_rng(2)
    beta = np.zeros(10); beta[[0, 3, 6]] = [2.0, -1.5, 1.0]
    X, y = gen(rng, 300, 10, 0.5, beta, 1.0)
    m = RSO(criterion=criterion).fit(X + 5.0, 3.0 + y)   # nonzero means
    sel = set(np.flatnonzero(m.support_))
    if criterion == "loo":   # LOO ~ AIC: keeps the truth, may add a noise variable
        assert {0, 3, 6} <= sel and len(sel) <= 5
    else:
        assert sel == {0, 3, 6}
    assert np.allclose(m.coef_[[0, 3, 6]], beta[[0, 3, 6]], atol=0.2)


def test_cv_timeseries_and_predict():
    rng = np.random.default_rng(3)
    beta = np.zeros(8); beta[[1, 2]] = [1.0, 1.0]
    X, y = gen(rng, 240, 8, 0.3, beta, 1.0)
    m = RSO(criterion="cv", cv=TimeSeriesSplit(4)).fit(X, y)
    assert {1, 2} <= set(np.flatnonzero(m.support_))
    assert m.predict(X).shape == (240,)


def test_integer_weights_equal_duplication():
    rng = np.random.default_rng(4)
    X, y = gen(rng, 60, 4, 0.3, np.array([1.0, 0, 0.5, 0]), 1.0)
    w = rng.integers(1, 4, 60)
    a = RSO(tau=3.0, refit=False).fit(X, y, sample_weight=w)
    b = RSO(tau=3.0, refit=False).fit(np.repeat(X, w, 0), np.repeat(y, w))
    assert np.allclose(a.coef_, b.coef_) and np.isclose(a.intercept_, b.intercept_)


def test_fixed_tau_and_path_fields():
    rng = np.random.default_rng(5)
    X, y = gen(rng, 50, 3, 0.0, np.array([1.0, 0, 0]), 1.0)
    m = RSO(tau=2.0).fit(X, y)
    assert m.tau_ == 2.0 and np.isclose(m.gamma_.sum(), 2.0) and len(m.path_) == 1
    for key in ["naive_bic", "profile_bic", "rss", "df", "support", "coef"]:
        assert key in m.path_[0]


def test_boundary_warning():
    rng = np.random.default_rng(6)
    X, y = gen(rng, 200, 5, 0.0, np.ones(5), 0.5)
    with pytest.warns(RuntimeWarning):
        RSO(criterion="naive_bic", taus=[0.01, 0.02, 0.03]).fit(X, y)


# ---------- closed-form leave-one-out ---------------------------------------
def brute_loo(m, X, y, w):
    """Delete each row, refit the reported estimator with gamma and support held
    fixed (same centering/scale convention), and score the held-out residual."""
    d = m._prepare(X, y, w)
    Xs, yc, wn, n = d["Xs"], y, d["w"], d["n"]
    idx = np.flatnonzero(m.support_)
    Z = Xs[:, idx]
    pen = np.zeros(len(idx)) if m.refit else n / m.gamma_[idx]
    if m.fit_intercept:
        Z = np.column_stack([np.ones(n), Z]); pen = np.r_[0.0, pen]
    err = 0.0
    for i in range(n):
        k = np.arange(n) != i
        Zk, wk = Z[k], wn[k]
        theta = np.linalg.solve(Zk.T @ (Zk * wk[:, None]) + np.diag(pen), Zk.T @ (wk * yc[k]))
        err += wn[i] * (yc[i] - Z[i] @ theta) ** 2
    return err / wn.sum()


@pytest.mark.parametrize("refit", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_loo_matches_brute_force(refit, fit_intercept):
    rng = np.random.default_rng(7)
    beta = np.zeros(6); beta[[0, 2]] = [1.5, -1.0]
    X, y = gen(rng, 50, 6, 0.4, beta, 1.0)
    w = rng.uniform(0.5, 2.0, 50)
    m = RSO(tau=3.0, refit=refit, fit_intercept=fit_intercept).fit(X + 2.0, y + 1.0, w)
    assert np.isclose(m.path_[0]["loo"], brute_loo(m, X + 2.0, y + 1.0, w))


def test_loo_refit_close_to_exact_loo():
    rng = np.random.default_rng(8)
    beta = np.zeros(8); beta[[1, 4]] = [1.0, -1.0]
    X, y = gen(rng, 80, 8, 0.3, beta, 1.0)
    taus = default_taus(8, 10)
    fast = RSO(criterion="loo", taus=taus).fit(X, y)
    exact = RSO(criterion="cv", cv=LeaveOneOut(), taus=taus).fit(X, y)
    assert {1, 4} <= set(np.flatnonzero(fast.support_))
    # closed form is conditional on gamma_hat, so it can only be optimistic
    assert np.all(fast.criterion_values_ <= exact.criterion_values_ + 1e-12)


def test_loo_msrr_sdf_use():
    """AIPM-style use: regress a vector of ones on factor returns, no intercept."""
    rng = np.random.default_rng(9)
    T, P = 240, 12
    F = 0.04 * rng.standard_normal((T, P))
    F[:, :3] += 0.01                                  # three factors with premia
    m = RSO(criterion="loo", fit_intercept=False, standardize=False, refit=False,
            taus=np.logspace(0, 4, 15)).fit(F, np.ones(T))
    assert m.intercept_ == 0.0
    assert np.isfinite(m.criterion_value_) and set(np.flatnonzero(m.support_)) >= {0, 1, 2}
