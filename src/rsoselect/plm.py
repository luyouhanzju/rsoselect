"""
RSO variable selection for the partially linear model

    y = X beta + f(Z) + eps

(Lu, Dong, Hu, Wu; R code GPLM_revision/rstudio-export-plm/main_2tau.R).

For budgets (tau1, tau2) solve

    min_{gamma, lambda} || R(gamma, lambda) y ||^2
    s.t. gamma >= 0, sum(gamma) = tau1,  lambda >= 0, sum(lambda) = tau2

with S = local constant smoother with inverse bandwidths lambda,
X~ = (I - S) X, y~ = (I - S) y and

    beta(gamma, lambda) = D^1/2 (D^1/2 X~'X~ D^1/2 / n + I)^-1 D^1/2 X~'y~ / n   (paper eq. 3)
    R(gamma, lambda) y  = y~ - X~ beta.

So for fixed lambda this is linear RSO on (X~, y~). Both blocks are optimized by
modified coordinate descent (gamma block first, as in MCD()).

Tuning
    naive_bic   (eq. 7):  n log RSS + log(n) tr(I - R)
    profile_bic (eq. 10): for each distinct selected model (P, N) on the path, refit
                          y = X_P beta + f(Z_N) without ridge penalty using local linear
                          smoothing, choose its own budget tau~ by BIC (eq. 9), and score
                          the model by that BIC. If N is empty the refit is OLS.
Final estimate (refit=True, Sec. 4.2 / 4.4): local linear refit on the selected model
with tau~ chosen by AIC (eq. 11).
"""

import itertools

import numpy as np

from .linear import ridge_beta, ridge_rss
from .mcd import DEFAULT_GRID, mcd
from .smoothers import TrainingKernel, local_constant, local_linear
from .tuning import aic, bic, check_boundary_2d, default_taus, default_taus2


def _key(v):
    return tuple(np.round(np.asarray(v, dtype=float), 12))


class PLM:
    """Partially linear model with RSO selection of linear (X) and nonparametric (Z) predictors.

    Parameters
    ----------
    taus1, taus2 : array-like or None
        Budgets for gamma (linear part) and lambda (inverse bandwidths). Defaults:
        p * logspace(0.05, 50) with `n_taus1` points, q * logspace(0.05, 3) with
        `n_taus2` points (on standardized X, Z).
    criterion : {"profile_bic", "naive_bic"}, default "profile_bic"
    taus_refit : array-like or None
        Grid for the total budget tau~ of the local linear refit bandwidths; default
        logspace(0.25, 6, n_taus_refit) on standardized Z (R: seq(0.5, 10, .5) on
        Z in [-1, 1]). Refits with df >= n - 1 (interpolation) are discarded.
    refit : bool, default True
        Final estimate from the unpenalized local linear refit (AIC-tuned tau~).
        If False, the RSO estimate (local constant, ridge) at the selected taus.
    grid, max_sweeps, tol, keep_best : MCD settings (R: grid (0,.2,.5,.8,1), 5 sweeps).
    refit_max_sweeps : int, default 16
        MCD sweeps for the refit bandwidths (LLMCD in R).
    select_threshold : float, default 1e-2
    standardize : bool, default True
        Scale X columns and center/scale Z columns. Ignored (False) when r_compat.
    r_compat : bool, default False
        Reproduce main_2tau.R instead of the paper formulas: kernel
        exp(-(sum_j lambda_j |dz_j|)^2) instead of the product Gaussian kernel, S = 0
        when sum(lambda) <= 1e-3, no standardization, and the MCD returns the last
        sweep on convergence.
    """

    def __init__(self, taus1=None, taus2=None, n_taus1=12, n_taus2=8,
                 criterion="profile_bic", taus_refit=None, n_taus_refit=10, refit=True,
                 grid=DEFAULT_GRID, max_sweeps=5, refit_max_sweeps=16, tol=1e-4,
                 select_threshold=1e-2, keep_best=True, standardize=True, r_compat=False):
        self.taus1 = taus1
        self.taus2 = taus2
        self.n_taus1 = n_taus1
        self.n_taus2 = n_taus2
        self.criterion = criterion
        self.taus_refit = taus_refit
        self.n_taus_refit = n_taus_refit
        self.refit = refit
        self.grid = grid
        self.max_sweeps = max_sweeps
        self.refit_max_sweeps = refit_max_sweeps
        self.tol = tol
        self.select_threshold = select_threshold
        self.keep_best = keep_best
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
            raise ValueError("PLM needs at least one linear and one nonparametric predictor; "
                             "use RSO for a purely linear model")
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
        self._lc_cache, self._ll_cache, self._profile_cache = {}, {}, {}
        self._tk = TrainingKernel(self._Z, kernel=self._kernel)

    # --------------------------------------------- penalized local-constant fit
    def _lc(self, lam):
        """Cached quantities of the local-constant smoother for one lambda."""
        k = _key(lam)
        if k not in self._lc_cache:
            n = self._n
            if self.r_compat and np.sum(lam) <= 1e-3:
                S = np.zeros((n, n))
            else:
                S = self._tk.local_constant(lam)
            Xt = self._X - S @ self._X
            yt = self._y - S @ self._y
            self._lc_cache[k] = dict(S=S, Xt=Xt, yt=yt, G=Xt.T @ Xt, b=Xt.T @ yt,
                                     yy=float(yt @ yt), XtISXt=Xt.T @ (Xt - S @ Xt))
        return self._lc_cache[k]

    def _objective(self, theta):
        p = self._p
        c = self._lc(theta[p:])
        return ridge_rss(c["G"], c["b"], c["yy"], theta[:p], self._n)

    def _naive_stats(self, gamma, lam):
        c = self._lc(lam)
        n = self._n
        beta, A, act = ridge_beta(c["G"], c["b"], gamma, n)
        res = c["yt"] - c["Xt"] @ beta
        rss = float(res @ res)
        df = float(np.trace(c["S"]))                      # tr(I - R)
        if A is not None:
            df += float(np.trace(A @ c["XtISXt"][np.ix_(act, act)]))
        return beta, rss, df

    # ------------------------------------------- unpenalized local-linear refit
    def _ll_fit(self, P, N, lam):
        k = (P, N, _key(lam))
        if k not in self._ll_cache:
            y, n = self._y, self._n
            S = local_linear(self._Z[:, list(N)], self._Z[:, list(N)], lam)
            yt = y - S @ y
            if len(P):
                XP = self._X[:, list(P)]
                Xt = XP - S @ XP
                Q = Xt.T @ Xt
                Qinv = np.linalg.pinv(Q)
                beta = Qinv @ (Xt.T @ yt)
                res = yt - Xt @ beta
                df = float(np.trace(S) + np.trace(Qinv @ (Xt.T @ (Xt - S @ Xt))))
            else:
                beta = np.zeros(0)
                res = yt
                df = float(np.trace(S))
            self._ll_cache[k] = dict(rss=float(res @ res), df=df, beta=beta)
        return self._ll_cache[k]

    def _refit_taus(self, k):
        """Total budget tau~ for the refit bandwidths. Default logspace(0.25, 6) on
        standardized Z, matching seq(0.5, 10, .5) in the R code for Z in [-1, 1]."""
        if self.taus_refit is not None:
            return np.asarray(self.taus_refit, dtype=float)
        return np.logspace(np.log10(0.25), np.log10(6.0), self.n_taus_refit)

    def _profile(self, P, N):
        """Profile BIC (eq. 10) of a selected model; also keeps the AIC choice (eq. 11)."""
        key = (P, N)
        if key in self._profile_cache:
            return self._profile_cache[key]
        n, y = self._n, self._y
        if len(N) == 0:
            D = np.column_stack([np.ones(n), self._X[:, list(P)]])
            coef = np.linalg.lstsq(D, y, rcond=None)[0]
            r = y - D @ coef
            rss, df = float(r @ r), len(P) + 1.0
            out = dict(pbic=bic(rss, df, n), aic_choice=dict(lam=None, rss=rss, df=df,
                       beta=coef[1:], intercept=float(coef[0])), records=[])
        else:
            recs = []
            for t in self._refit_taus(len(N)):
                lam, _, _ = mcd(lambda l: self._ll_fit(P, N, l)["rss"],
                                [(np.arange(len(N)), t)], grid=self.grid,
                                max_sweeps=self.refit_max_sweeps, tol=self.tol,
                                keep_best=self.keep_best)
                f = self._ll_fit(P, N, lam)
                if f["df"] >= n - 1:          # interpolating fit: log RSS degenerates
                    continue
                recs.append(dict(tau=float(t), lam=lam, rss=f["rss"], df=f["df"],
                                 beta=f["beta"], bic=bic(f["rss"], f["df"], n),
                                 aic=aic(f["rss"], f["df"], n)))
            if not recs:
                out = dict(pbic=np.inf, records=[], aic_choice=None)
            else:
                best_aic = min(recs, key=lambda r: r["aic"])
                out = dict(pbic=min(r["bic"] for r in recs), records=recs,
                           aic_choice=dict(best_aic, intercept=0.0))
        self._profile_cache[key] = out
        return out

    # ---------------------------------------------------------------------- api
    def fit(self, X, Z, y):
        if self.criterion not in ("profile_bic", "naive_bic"):
            raise ValueError("criterion must be 'profile_bic' or 'naive_bic'")
        self._prepare(X, Z, y)
        n, p, q = self._n, self._p, self._q
        taus1 = default_taus(p, self.n_taus1) if self.taus1 is None else np.asarray(self.taus1, float)
        taus2 = default_taus2(q, self.n_taus2) if self.taus2 is None else np.asarray(self.taus2, float)
        thr = self.select_threshold

        path = []
        for t1, t2 in itertools.product(taus1, taus2):
            theta, rss, sweeps = mcd(self._objective,
                                     [(np.arange(p), t1), (p + np.arange(q), t2)],
                                     grid=self.grid, max_sweeps=self.max_sweeps, tol=self.tol,
                                     keep_best=self.keep_best and not self.r_compat)
            gamma, lam = theta[:p], theta[p:]
            beta, rss, df = self._naive_stats(gamma, lam)
            path.append(dict(tau1=float(t1), tau2=float(t2), gamma=gamma, lam=lam,
                             beta=beta, rss=rss, df=df, naive_bic=bic(rss, df, n),
                             support_x=gamma > thr, support_z=lam > thr, sweeps=sweeps))

        models = [(tuple(np.flatnonzero(r["support_x"])), tuple(np.flatnonzero(r["support_z"])))
                  for r in path]
        if self.criterion == "naive_bic":
            values = np.array([r["naive_bic"] for r in path])
        else:
            values = np.array([self._profile(*m)["pbic"] for m in models])
            for r, v in zip(path, values):
                r["profile_bic"] = float(v)
        k = int(np.argmin(values))
        check_boundary_2d(values.reshape(len(taus1), len(taus2)), taus1, taus2)

        best = path[k]
        self.path_ = path
        self.criterion_values_ = values.reshape(len(taus1), len(taus2))
        self.tau1_, self.tau2_ = best["tau1"], best["tau2"]
        self.gamma_, self.lambda_ = best["gamma"], best["lam"]
        self.support_x_, self.support_z_ = best["support_x"], best["support_z"]
        P, N = models[k]
        self._finalize(P, N, best)
        return self

    def _finalize(self, P, N, best):
        p = self._p
        beta_s = np.zeros(p)
        self._N = list(N)
        if self.refit:
            ch = self._profile(P, N)["aic_choice"]
            beta_s[list(P)] = ch["beta"]
            self.intercept_ = ch["intercept"]
            self._smoother = "constant" if len(N) == 0 else "local_linear"
            self.lambda_refit_ = ch["lam"]
            self._lam_pred = ch["lam"]
        else:
            beta_s[list(P)] = best["beta"][list(P)]
            self.intercept_ = 0.0
            lam = np.where(best["support_z"], best["lam"], 0.0)
            self._smoother = "zero" if (self.r_compat and lam.sum() <= 1e-3) else "local_constant"
            self._N = list(range(self._q))
            self._lam_pred = lam
        self._beta_s = beta_s
        self.coef_ = beta_s / self.x_scale_
        self._resid = self._y - self._X @ beta_s - self.intercept_
        self.f_train_ = self._f(self._Z)

    def _f(self, Zs):
        if self._smoother in ("constant", "zero"):
            return np.zeros(len(Zs))
        T, Ztr = Zs[:, self._N], self._Z[:, self._N]
        if self._smoother == "local_linear":
            S = local_linear(T, Ztr, self._lam_pred)
        else:
            S = local_constant(T, Ztr, self._lam_pred, kernel=self._kernel)
        return S @ self._resid

    def predict_f(self, Z):
        """Estimated nonparametric component f(Z) (intercept included in it)."""
        Z = np.asarray(Z, dtype=float).reshape(-1, self._q)
        return self._f((Z - self.z_center_) / self.z_scale_) + self.intercept_

    def predict(self, X, Z):
        X = np.asarray(X, dtype=float).reshape(-1, self._p)
        return X @ self.coef_ + self.predict_f(Z)
