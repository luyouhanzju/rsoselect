"""
Modified coordinate descent (MCD) on products of scaled simplices.

Generic version of MCD() in main_2tau.R. The parameter vector is split into
blocks; block b lives on {theta_b >= 0, sum(theta_b) = tau_b}.

    linear RSO : one block  (gamma, tau1)
    PLM / GPLM : two blocks (gamma, tau1) for the linear part,
                            (lambda, tau2) for the inverse bandwidths

For coordinate j of block b, with theta1 = theta_b with j set to 0 and
renormalized to sum 1, the candidates are

    theta_b = (g * theta1 + (1 - g) * e_j) * tau_b,   g in grid

and the one with the smallest objective is kept. g = 1 drops j, g = 0 puts the
whole budget on j.
"""

import numpy as np

DEFAULT_GRID = (0.0, 0.2, 0.5, 0.8, 1.0)


def mcd(objective, blocks, grid=DEFAULT_GRID, max_sweeps=5, tol=1e-4, init=None,
        keep_best=True):
    """
    Parameters
    ----------
    objective : callable(theta) -> float
        Objective on the full parameter vector (all blocks concatenated).
    blocks : list of (indices, tau)
        indices : array of positions in theta belonging to this block.
        tau     : budget for this block.
    grid : sequence of mixing weights g.
    max_sweeps : int
        Max passes over all blocks (the R code stops after 5).
    tol : float
        Converged when max |theta_new - theta_old| < tol (eps1 in R).
    init : array or None
        Starting point. Default: uniform tau_b / |block| within each block.
    keep_best : bool
        Return the sweep with the smallest objective (True), or the last sweep
        on convergence as the R code does (False). The R code returns the best
        sweep when it hits max_sweeps.

    Returns
    -------
    theta : array, obj : float, n_sweeps : int
    """
    d = sum(len(np.atleast_1d(idx)) for idx, _ in blocks)
    blocks = [(np.atleast_1d(np.asarray(idx, dtype=int)), float(tau)) for idx, tau in blocks]
    grid = np.asarray(grid, dtype=float)
    eps0 = 2 * tol

    if init is None:
        theta = np.zeros(d)
        for idx, tau in blocks:
            theta[idx] = tau / len(idx)
    else:
        theta = np.array(init, dtype=float)

    best_theta, best_obj = theta.copy(), np.inf
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        old = theta.copy()
        for idx, tau in blocks:
            m = len(idx)
            for jj in range(m):
                t1 = theta[idx].copy()
                t1[jj] = 0.0
                if t1.sum() <= eps0:           # all budget already on j
                    continue
                t1 /= t1.sum()
                ej = np.zeros(m)
                ej[jj] = 1.0
                cand_objs, cands = [], []
                for g in grid:
                    c = theta.copy()
                    c[idx] = (g * t1 + (1 - g) * ej) * tau
                    cands.append(c)
                    cand_objs.append(objective(c))
                theta = cands[int(np.argmin(cand_objs))]
        obj = objective(theta)
        if obj < best_obj:
            best_obj, best_theta = obj, theta.copy()
        if sweeps == max_sweeps:              # R checks the sweep cap before convergence
            break
        if np.max(np.abs(old - theta)) < tol:
            if not keep_best:
                return theta, obj, sweeps
            break
    return best_theta, best_obj, sweeps
