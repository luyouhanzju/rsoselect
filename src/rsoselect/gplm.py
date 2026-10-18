"""
RSO variable selection for the generalized partially linear model

    g(E[y | x, z]) = x'beta + f(z)

(Lu, Dong, Hu, Wu; R code GPLM_revision/rstudio-export: local_scoring.R, MCD.R,
main_2tau_deviance.R, getVS.R).

For given (gamma, lambda) the model is fit by local scoring (paper Sec. 2.2): starting
from beta = 0, f = g(ybar), repeat
    w_i = (dmu/deta)^2 / V(mu_i),   z_i = eta_i + (y_i - mu_i) / (dmu/deta)
    S   = local constant smoother with weights w and inverse bandwidths lambda
    beta = weighted RSO step on X~ = (I - S) X, z~ = (I - S) z            (eq. 2.7)
    f    = S (z - X beta)
until the weights stabilize. (gamma, lambda) minimize the negative log-likelihood
subject to sum(gamma) = tau1, sum(lambda) = tau2 (eq. 3.10), by modified coordinate
descent.

Tuning (tau1, tau2):
    naive_bic (eq. 4.11): -2 loglik + log(n) (#{beta_k != 0} + tr S)
    refit_bic (eq. 5.17): same, after refitting the selected model (X_P, Z_N) by
                          unpenalized local scoring with the selected bandwidths.
"""

import itertools

import numpy as np

from .families import get_family
from .linear import ridge_beta
from .mcd import DEFAULT_GRID, mcd
from .smoothers import TrainingKernel, local_constant
from .tuning import check_boundary_2d, default_taus, default_taus2


def _key(v):
    return tuple(np.round(np.asarray(v, dtype=float), 12))


class GPLM:
    """Generalized partially linear model with RSO selection.

    Parameters
    ----------
    family : {"binomial", "poisson", "gaussian"} or rsoselect.families.Family
    taus1, taus2, n_taus1, n_taus2 : budget grids as in PLM (defaults 8 x 5 points).
    criterion : {"refit_bic", "naive_bic"}, default "refit_bic"
    refit : bool, default True
        Final estimate from the unpenalized refit of the selected model.
    grid, tol, keep_best : MCD settings.
    max_sweeps : int, default 9 (MCD.R stops after 9 sweeps)
    ls_max_iter : int, default 16, ls_tol : float, default 1e-5
        Local scoring stopping rules (max change of the normalized weights).
    select_threshold : float, default 1e-2
    standardize : bool, default True (ignored when r_compat)
    r_compat : bool, default False
        Reproduce the R code instead of the paper formulas:
          - beta step weighted by sqrt(w / sum w) instead of w        (getHmat)
          - smoother kernel exp(-(sum_j lambda_j |dz_j|)^2), weights cancel (getLCSmat)
          - BIC uses -loglik instead of -2 loglik                       (naiveBICFULL)
          - refit uses uniform bandwidths sum(lambda) / |N|
          - signed convergence check on the weights, no standardization.
    """

    def __init__(self, family="binomial", taus1=None, taus2=None, n_taus1=8, n_taus2=5,
                 criterion="refit_bic", refit=True, grid=DEFAULT_GRID, max_sweeps=9,
                 tol=1e-4, keep_best=True, ls_max_iter=16, ls_tol=1e-5,
                 select_threshold=1e-2, standardize=True, r_compat=False):
        self.family = family
        self.taus1 = taus1
        self.taus2 = taus2
        self.n_taus1 = n_taus1
        self.n_taus2 = n_taus2
        self.criterion = criterion
        self.refit = refit
        self.grid = grid
        self.max_sweeps = max_sweeps
        self.tol = tol
        self.keep_best = keep_best
        self.ls_max_iter = ls_max_iter
        self.ls_tol = ls_tol
        self.select_threshold = select_threshold
        self.standardize = standardize
        self.r_compat = r_compat

    # ---------------------------------------------------------------- data prep
    def _prepare(self, X, Z, y):
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        X = X.reshape(len(X), -1)
        Z = Z.reshape(len(Z), -1)
        y = np.asarray(y, dtype=float).ravel()
        if not (len(X) == len(Z) == len(y)):
            raise ValueError("X, Z and y must have the same number of rows")
        n, p = X.shape
        q = Z.shape[1]
        if p == 0 or q == 0:
            raise ValueError("GPLM needs at least one linear and one nonparametric predictor")
        self.family_ = get_family(self.family)
        std = self.standardize and not self.r_compat
        self.x_scale_ = X.std(0) if std else np.ones(p)
        self.x_scale_[self.x_scale_ == 0] = 1.0
        self.z_center_ = Z.mean(0) if std else np.zeros(q)
        self.z_scale_ = Z.std(0) if std else np.ones(q)
        self.z_scale_[self.z_scale_ == 0] = 1.0
        self._X = X / self.x_scale_
        self._Z = (Z - self.z_center_) / self.z_scale_
        self._y = y
        self._n, self._p, self._q = n, p, q
        self._kernel = "r_code" if self.r_compat else "product"
        self._tk = TrainingKernel(self._Z, kernel=self._kernel)
        self._cache = {}

    # ------------------------------------------------------------ local scoring
    def _working(self, eta):
        fam, y = self.family_, self._y
        mu = fam.inverse(eta)
        d = fam.dmu_deta(eta)
        w = d ** 2 / fam.variance(mu)
        z = eta + (y - mu) / d
        return mu, w, z

    def _step(self, P, N, gamma, lam, w, z, tk):
        """One smoothing + beta update. gamma=None means unpenalized (refit)."""
        n = self._n
        wstd = w / w.sum()
        if self.r_compat:
            S = tk.local_constant(lam)            # row weights in getLCSmat cancel
            omega = np.sqrt(wstd)
        else:
            S = tk.local_constant(lam, w)
            omega = w
        X = self._X[:, P]
        if len(P):
            Xt = X - S @ X
            zt = z - S @ z
            Xw = Xt * omega[:, None]
            G, b = Xt.T @ Xw, Xw.T @ zt
            if gamma is None:
                beta = np.linalg.pinv(G) @ b
            else:
                beta = ridge_beta(G, b, gamma, n)[0]
            lin = X @ beta
        else:
            beta, lin = np.zeros(0), np.zeros(n)
        f = S @ (z - lin)
        return beta, f, lin + f, S

    def _local_scoring(self, P, N, gamma, lam):
        """Fit for fixed (gamma, lambda) on columns P of X and N of Z."""
        fam, y, n = self.family_, self._y, self._n
        tk = self._tk if len(N) == self._q else TrainingKernel(self._Z[:, N], self._kernel)
        eta = np.full(n, float(fam.link(np.array([y.mean()]))[0]))
        mu = np.full(n, y.mean())
        d = fam.dmu_deta(eta)
        w = d ** 2 / fam.variance(mu)
        z = eta + (y - mu) / d
        w_prev = w / w.sum()
        for it in range(self.ls_max_iter):
            _, _, eta, _ = self._step(P, N, gamma, lam, w, z, tk)
            mu, w, z = self._working(eta)
            w_now = w / w.sum()
            diff = (w_now - w_prev).max() if self.r_compat else np.abs(w_now - w_prev).max()
            if diff < self.ls_tol:
                break
            w_prev = w_now
        beta, f, eta, S = self._step(P, N, gamma, lam, w, z, tk)   # objfun() in R
        mu = fam.inverse(eta)
        return dict(beta=beta, eta=eta, w=w, z=z, trS=float(np.trace(S)), iters=it + 1,
                    loglik=fam.loglik(y, mu))

    def _fit_full(self, theta):
        p = self._p
        return self._local_scoring(list(range(p)), list(range(self._q)), theta[:p], theta[p:])

    def _negloglik(self, theta):
        k = _key(theta)
        if k not in self._cache:              # only scalars are cached (memory)
            self._cache[k] = -self._fit_full(theta)["loglik"]
        return self._cache[k]

    def _bic(self, loglik, beta, trS):
        df = np.count_nonzero(np.abs(beta) > 0) + trS
        dev = -loglik if self.r_compat else -2.0 * loglik
        return dev + np.log(self._n) * df, df

    # ---------------------------------------------------------------------- api
    def fit(self, X, Z, y):
        if self.criterion not in ("refit_bic", "naive_bic"):
            raise ValueError("criterion must be 'refit_bic' or 'naive_bic'")
        self._prepare(X, Z, y)
        n, p, q = self._n, self._p, self._q
        taus1 = default_taus(p, self.n_taus1) if self.taus1 is None else np.asarray(self.taus1, float)
        taus2 = default_taus2(q, self.n_taus2) if self.taus2 is None else np.asarray(self.taus2, float)
        thr = self.select_threshold
        obj = self._negloglik

        path = []
        for t1, t2 in itertools.product(taus1, taus2):
            theta, _, sweeps = mcd(obj, [(np.arange(p), t1), (p + np.arange(q), t2)],
                                   grid=self.grid, max_sweeps=self.max_sweeps, tol=self.tol,
                                   keep_best=self.keep_best and not self.r_compat)
            fit = self._fit_full(theta)
            gamma, lam = theta[:p], theta[p:]
            nb, df = self._bic(fit["loglik"], fit["beta"], fit["trS"])
            P = list(np.flatnonzero(gamma > thr))
            N = list(np.flatnonzero(lam > thr))
            if self.r_compat:
                lam_ref = np.full(len(N), lam.sum() / max(len(N), 1))
            else:
                lam_ref = lam[N]
            rf = self._local_scoring(P, N, None, lam_ref)
            rb, rdf = self._bic(rf["loglik"], rf["beta"], rf["trS"])
            path.append(dict(tau1=float(t1), tau2=float(t2), gamma=gamma, lam=lam,
                             support_x=gamma > thr, support_z=lam > thr, sweeps=sweeps,
                             loglik=fit["loglik"], df=df, naive_bic=nb,
                             loglik_refit=rf["loglik"], df_refit=rdf, refit_bic=rb,
                             _fit=fit, _refit=rf, _P=P, _N=N, _lam_ref=lam_ref))

        values = np.array([r[self.criterion] for r in path])
        k = int(np.argmin(values))
        check_boundary_2d(values.reshape(len(taus1), len(taus2)), taus1, taus2)
        best = path[k]
        self.path_ = [{kk: v for kk, v in r.items() if not kk.startswith("_")} for r in path]
        self.criterion_values_ = values.reshape(len(taus1), len(taus2))
        self.tau1_, self.tau2_ = best["tau1"], best["tau2"]
        self.gamma_, self.lambda_ = best["gamma"], best["lam"]
        self.support_x_, self.support_z_ = best["support_x"], best["support_z"]

        beta_s = np.zeros(p)
        if self.refit:
            src = best["_refit"]
            beta_s[best["_P"]] = src["beta"]
            self._N = best["_N"]
            self._lam_pred = best["_lam_ref"]
        else:
            src = best["_fit"]
            beta_s[best["_P"]] = src["beta"][best["_P"]]
            self._N = list(range(q))
            self._lam_pred = np.where(best["support_z"], best["lam"], 0.0)
        self._beta_s = beta_s
        self.coef_ = beta_s / self.x_scale_
        self._w_pred = src["w"]
        self._resid = src["z"] - self._X @ beta_s          # adjusted response minus linear part
        self.eta_train_ = src["eta"]
        self.loglik_ = src["loglik"]
        return self

    # ------------------------------------------------------------------ predict
    def predict_f(self, Z):
        """Estimated f(Z) on the link scale (intercept included)."""
        Z = np.asarray(Z, dtype=float).reshape(-1, self._q)
        Zs = (Z - self.z_center_) / self.z_scale_
        N = self._N
        w = None if self.r_compat else self._w_pred
        S = local_constant(Zs[:, N], self._Z[:, N], self._lam_pred, w=w, kernel=self._kernel)
        return S @ self._resid

    def predict_eta(self, X, Z):
        X = np.asarray(X, dtype=float).reshape(-1, self._p)
        return X @ self.coef_ + self.predict_f(Z)

    def predict(self, X, Z):
        """Predicted mean E[y | X, Z] (probabilities for binomial, rates for poisson)."""
        return self.family_.inverse(self.predict_eta(X, Z))
