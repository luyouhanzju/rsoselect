"""
Tuning criteria shared by linear RSO, PLM and GPLM.

naive BIC   (paper eq. 7):  n log RSS(tau) + log(n) * tr(hat matrix at gamma_hat(tau))
profile BIC (paper eq. 10): refit without the ridge penalty on the selected set
                            P(tau), then n log RSS_refit + log(n) * df_refit.
                            For the linear model the refit is OLS and df = |P|.
"""

import warnings

import numpy as np

CRITERIA = ("profile_bic", "naive_bic", "cv")


def bic(rss, df, n):
    return n * np.log(max(float(rss), 1e-300)) + np.log(n) * df


def aic(rss, df, n):
    return n * np.log(max(float(rss), 1e-300)) + 2.0 * df


def default_taus(p, n_taus=30, lo=0.05, hi=50.0):
    """Log-spaced grid relative to p.

    On standardized X, uniform gamma = tau/p is ordinary ridge with penalty p/tau
    (loss RSS/n), so this spans heavy shrinkage (ridge penalty 20) to near-OLS
    (ridge penalty 0.02) regardless of the data scale.
    """
    return p * np.logspace(np.log10(lo), np.log10(hi), n_taus)


def check_boundary(values, taus, name="tau"):
    """Warn when the criterion minimum is attained only at an end of the grid.

    Profile BIC is piecewise constant in tau, so a minimum shared with interior
    grid points is not a boundary problem.
    """
    values = np.asarray(values, dtype=float)
    m = len(values)
    if m < 3:
        return
    at_min = np.flatnonzero(np.isclose(values, values.min(), rtol=0, atol=1e-9))
    if np.any((at_min > 0) & (at_min < m - 1)):
        return
    k = int(at_min[0])
    side = "smallest" if k == 0 else "largest"
    warnings.warn(
        f"Selected {name}={taus[k]:.4g} is the {side} value on the grid; "
        f"the optimum may lie outside it. Consider widening `taus`.",
        RuntimeWarning, stacklevel=3)


def check_boundary_2d(values, taus1, taus2, names=("tau1", "tau2")):
    """values has shape (len(taus1), len(taus2)). Warn per axis when every minimizer
    sits on the edge of that axis."""
    V = np.asarray(values, dtype=float)
    at_min = np.argwhere(np.isclose(V, np.nanmin(V), rtol=0, atol=1e-9))
    for axis, (taus, name) in enumerate(zip((taus1, taus2), names)):
        m = len(taus)
        if m < 3:
            continue
        idx = at_min[:, axis]
        if np.any((idx > 0) & (idx < m - 1)):
            continue
        k = int(idx[0])
        side = "smallest" if k == 0 else "largest"
        warnings.warn(
            f"Selected {name}={taus[k]:.4g} is the {side} value on its grid; "
            f"the optimum may lie outside it. Consider widening `taus{axis + 1}`.",
            RuntimeWarning, stacklevel=3)


def default_taus2(q, n_taus=8, lo=0.05, hi=3.0):
    """Grid for the inverse-bandwidth budget. With Z standardized, lambda_j = 1 means
    bandwidth one standard deviation; uniform lambda = tau/q spans h in [1/3, 20] sd."""
    return q * np.logspace(np.log10(lo), np.log10(hi), n_taus)
