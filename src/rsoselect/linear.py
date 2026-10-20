"""
Linear ridge selection operator (RSO).

For a fixed budget tau:

    beta(gamma) = argmin_b (1/n) sum_i w_i (y_i - x_i'b)^2 + sum_j b_j^2 / gamma_j
                = D^{1/2} (D^{1/2} X'WX D^{1/2} / n + I)^{-1} D^{1/2} X'Wy / n,  D = diag(gamma)

    gamma_hat(tau) = argmin  RSS(beta(gamma))  s.t. gamma >= 0, sum(gamma) = tau

solved by modified coordinate descent (rsoselect.mcd). gamma_j = 0 <=> beta_j = 0.
This is getHmat() / MCD() in main_2tau.R with the smoother S set to 0.

tau is tuned over a grid by
    "profile_bic" (default): OLS refit on P(tau) = {j : gamma_j > threshold},
                             BIC = n log RSS_ols + log(n) |P|        (paper eq. 10)
    "naive_bic"            : BIC = n log RSS(gamma_hat) + log(n) tr(H) (paper eq. 7)
    "cv"                   : validation MSE over `cv` splits (e.g. TimeSeriesSplit)
"""

import numpy as np

from .mcd import DEFAULT_GRID, mcd
from .tuning import CRITERIA, bic, check_boundary, default_taus

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:  # usable without sklearn
    class BaseEstimator:  # minimal stand-ins so RSO can still subclass them
        pass

    class RegressorMixin:
        pass


# --------------------------------------------------------------------- algebra
def ridge_beta(G, b, gamma, n):
    """beta(gamma) from G = X'WX, b = X'Wy. Returns (beta, A_active, active)."""
    p = len(gamma)
    beta = np.zeros(p)
    act = gamma > 0
    if not act.any():
        return beta, None, act
    s = np.sqrt(gamma[act])
    M = (s[:, None] * G[np.ix_(act, act)] * s[None, :]) / n + np.eye(act.sum())
    A = s[:, None] * np.linalg.solve(M, np.diag(s)) / n     # D^1/2 M^-1 D^1/2 / n
    beta[act] = A @ b[act]
    return beta, A, act


def ridge_rss(G, b, yWy, gamma, n):
    beta, _, _ = ridge_beta(G, b, gamma, n)
    return float(yWy - 2.0 * beta @ b + beta @ G @ beta)


def ridge_df(G, b, gamma, n):
    """tr(hat matrix) = tr(A X'WX) on the active set."""
    _, A, act = ridge_beta(G, b, gamma, n)
    return 0.0 if A is None else float(np.trace(A @ G[np.ix_(act, act)]))


def ols_subset(G, b, yWy, idx):
    """Weighted OLS on columns idx: returns (beta_idx, rss)."""
    if len(idx) == 0:
        return np.zeros(0), float(yWy)
    Gs, bs = G[np.ix_(idx, idx)], b[idx]
    beta = np.linalg.lstsq(Gs, bs, rcond=None)[0]
    return beta, float(yWy - beta @ bs)


# ------------------------------------------------------------------- estimator
class RSO(BaseEstimator, RegressorMixin):
    """Linear ridge selection operator.

    Parameters
    ----------
    tau : float or None
        Fixed budget sum(gamma). If None (default), tuned over `taus`.
    taus : array-like or None
        Candidate taus. Default: `n_taus` values log-spaced in p*[0.05, 50]
        (see rsoselect.tuning.default_taus).
    n_taus : int, default 30
    criterion : {"profile_bic", "naive_bic", "cv"}, default "profile_bic"
        How tau is chosen (ignored when `tau` is given).
    cv : int or sklearn splitter, default 5
        Used only when criterion="cv". Pass TimeSeriesSplit(...) for time series.
    refit : bool, default True
        Final coefficients: OLS on the selected variables (paper Sec. 4.2) if True;
        the RSO ridge estimate with unselected coefficients set to 0 if False.
        criterion="naive_bic", refit=False, fit_intercept=False, standardize=False
        reproduces the R naiveBIC() path.
    grid : tuple, default (0, .2, .5, .8, 1)
        Mixing weights for the coordinate moves (mygrid in MCD()).
    max_sweeps : int, default 5
    tol : float, default 1e-4
    select_threshold : float, default 1e-2
        Variable j is selected if gamma_j > select_threshold (1e-2 in the R code).
    keep_best : bool, default True
        See rsoselect.mcd.mcd.
    fit_intercept, standardize : bool, default True
        Center / scale X before fitting (intercept unpenalized). Coefficients are
        reported on the original scale. Standardizing makes tau comparable across
        datasets; the R code does neither.

    Attributes
    ----------
    coef_, intercept_, support_ (bool mask), gamma_ (standardized scale),
    tau_, criterion_value_, path_ (list of dicts, one per tau), n_features_in_
    """

    def __init__(self, tau=None, taus=None, n_taus=30, criterion="profile_bic", cv=5,
                 refit=True, grid=DEFAULT_GRID, max_sweeps=5, tol=1e-4,
                 select_threshold=1e-2, keep_best=True, fit_intercept=True,
                 standardize=True):
        self.tau = tau
        self.taus = taus
        self.n_taus = n_taus
        self.criterion = criterion
        self.cv = cv
        self.refit = refit
        self.grid = grid
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.select_threshold = select_threshold
        self.keep_best = keep_best
        self.fit_intercept = fit_intercept
        self.standardize = standardize

    # ---------------------------------------------------------------- internals
    def _prepare(self, X, y, w):
        n, p = X.shape
        w = np.ones(n) if w is None else np.asarray(w, dtype=float).ravel()
        if np.any(w < 0) or w.sum() <= 0:
            raise ValueError("sample_weight must be nonnegative and not all zero")
        w = w * (n / w.sum())                    # mean-1 weights
        if self.fit_intercept:
            x_mean, y_mean = (w @ X) / n, (w @ y) / n
        else:
            x_mean, y_mean = np.zeros(p), 0.0
        Xc, yc = X - x_mean, y - y_mean
        if self.standardize:
            x_scale = np.sqrt((w @ Xc ** 2) / n)
            x_scale[x_scale == 0] = 1.0
        else:
            x_scale = np.ones(p)
        Xs = Xc / x_scale
        Xw = Xs * w[:, None]
        return dict(n=n, p=p, G=Xs.T @ Xw, b=Xw.T @ yc, yWy=float(yc @ (w * yc)),
                    x_mean=x_mean, y_mean=y_mean, x_scale=x_scale)

    def _taus(self, p):
        if self.tau is not None:
            return np.array([float(self.tau)])
        if self.taus is not None:
            return np.asarray(self.taus, dtype=float)
        return default_taus(p, self.n_taus)

    def _path(self, X, y, w, taus):
        """Solve RSO for every tau. Coefficients on the original scale."""
        d = self._prepare(X, y, w)
        n, p, G, b, yWy = d["n"], d["p"], d["G"], d["b"], d["yWy"]
        obj = lambda g: ridge_rss(G, b, yWy, g, n)
        path = []
        for t in taus:
            gamma, rss, sweeps = mcd(obj, [(np.arange(p), t)], grid=self.grid,
                                     max_sweeps=self.max_sweeps, tol=self.tol,
                                     keep_best=self.keep_best)
            support = gamma > self.select_threshold
            idx = np.flatnonzero(support)
            df = ridge_df(G, b, gamma, n)
            beta_ols, rss_ols = ols_subset(G, b, yWy, idx)

            beta_s = np.zeros(p)
            if self.refit:
                beta_s[idx] = beta_ols
            else:
                beta_s[idx] = ridge_beta(G, b, gamma, n)[0][idx]
            coef = beta_s / d["x_scale"]
            path.append(dict(
                tau=float(t), gamma=gamma, support=support, sweeps=sweeps,
                rss=rss, df=df, naive_bic=bic(rss, df, n),
                rss_refit=rss_ols, df_refit=float(len(idx)),
                profile_bic=bic(rss_ols, len(idx), n),
                coef=coef, intercept=float(d["y_mean"] - d["x_mean"] @ coef)))
        return path

    # ---------------------------------------------------------------------- api
    def fit(self, X, y, sample_weight=None):
        if self.criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape
        taus = self._taus(p)
        path = self._path(X, y, sample_weight, taus)

        if len(taus) == 1:
            k, values = 0, None
        elif self.criterion == "cv":
            values = self._cv_scores(X, y, sample_weight, taus)
            k = int(np.argmin(values))
        else:
            values = np.array([r[self.criterion] for r in path])
            k = int(np.argmin(values))
        if values is not None:
            check_boundary(values, taus)
            self.criterion_values_ = values

        best = path[k]
        self.path_ = path
        self.tau_ = best["tau"]
        self.gamma_ = best["gamma"]
        self.support_ = best["support"]
        self.coef_ = best["coef"]
        self.intercept_ = best["intercept"]
        self.criterion_value_ = None if values is None else float(values[k])
        self.n_features_in_ = p
        return self

    def _cv_scores(self, X, y, w, taus):
        if isinstance(self.cv, int):
            from sklearn.model_selection import KFold
            splitter = KFold(self.cv)
        else:
            splitter = self.cv
        w_all = np.ones(len(y)) if w is None else np.asarray(w, dtype=float).ravel()
        err = np.zeros(len(taus))
        tot = 0.0
        for tr, va in splitter.split(X):
            fold = self._path(X[tr], y[tr], w_all[tr], taus)
            for i, r in enumerate(fold):
                res = y[va] - X[va] @ r["coef"] - r["intercept"]
                err[i] += w_all[va] @ res ** 2
            tot += w_all[va].sum()
        return err / tot

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def rso(X, y, **kwargs):
    """Functional shortcut: returns (coef, intercept, support)."""
    m = RSO(**kwargs).fit(X, y)
    return m.coef_, m.intercept_, m.support_
