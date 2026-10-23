# rsoselect

[![tests](https://github.com/luyouhanzju/rsoselect/actions/workflows/tests.yml/badge.svg)](https://github.com/luyouhanzju/rsoselect/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/rsoselect)](https://pypi.org/project/rsoselect/)
[![Python](https://img.shields.io/pypi/pyversions/rsoselect)](https://pypi.org/project/rsoselect/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)

**Variable selection with the ridge selection operator (RSO) for linear, partially linear,
and generalized partially linear models.**

Ridge regression with an individual penalty per coefficient can select variables: if the
penalty on a coefficient goes to infinity, that coefficient is exactly zero. RSO puts a
budget on the inverse penalties and lets the data decide where to spend it
([Wu, 2021](#references)). The same idea, applied to the inverse bandwidths of a kernel
smoother, selects nonparametric predictors as well, so linear and nonlinear predictors
are chosen in one optimization problem ([Lu et al., 2024](#references);
[Lu, Hu and Wu, 2025](#references)).

| model | class | paper |
|---|---|---|
| y = X&beta; + &epsilon; | `RSO` | Wu (2021), *Technometrics* |
| y = X&beta; + f(Z) + &epsilon; | `PLM` | Lu, Dong, Hu and Wu (2024), *JCGS* |
| g(E[y]) = X&beta; + f(Z) | `GPLM` | Lu, Hu and Wu (2025), *Journal of Nonparametric Statistics* |

## Installation

```bash
pip install rsoselect                # core
pip install "rsoselect[sklearn]"     # with scikit-learn integration
```

From source:

```bash
git clone https://github.com/luyouhanzju/rsoselect.git
cd rsoselect
pip install -e .
```

Requires Python 3.9+, NumPy and SciPy. scikit-learn is optional (needed only for
`criterion="cv"` and for using `RSO` inside sklearn pipelines).

## Quick start

```python
from rsoselect import RSO, PLM, GPLM

# Linear model: drop-in replacement for sklearn's Ridge
m = RSO().fit(X, y)
m.support_          # selected predictors (boolean mask)
m.coef_, m.intercept_
m.predict(X_new)

# Partially linear model: X enters linearly, Z nonparametrically
m = PLM().fit(X, Z, y)
m.support_x_, m.support_z_
m.predict(X_new, Z_new)

# Generalized partially linear model (binomial, poisson or gaussian)
m = GPLM(family="binomial").fit(X, Z, y)
m.predict(X_new, Z_new)        # probabilities
m.predict_eta(X_new, Z_new)    # linear predictor
```

## How it works

For a budget &tau;, RSO solves

```
min over gamma:   RSS( beta(gamma) )
subject to:       gamma_k >= 0,  sum_k gamma_k = tau

beta(gamma) = argmin_b  (1/n) ||y - X b||^2 + sum_k b_k^2 / gamma_k
```

so &gamma;<sub>k</sub> = 0 gives &beta;<sub>k</sub> = 0 exactly. In the PLM and GPLM the
nonparametric part is a kernel smoother with inverse bandwidths &lambda;<sub>j</sub>
under a second budget &sum;&lambda;<sub>j</sub> = &tau;<sub>2</sub>; &lambda;<sub>j</sub> = 0
removes Z<sub>j</sub>. The GPLM is fit by local scoring, and (&gamma;, &lambda;) minimize
the negative log-likelihood. All three models are optimized by the same modified
coordinate descent (`rsoselect.mcd`).

The budgets are tuned over a grid:

| model | criteria (default first) |
|---|---|
| `RSO` | `"profile_bic"` (OLS refit on the selected set), `"naive_bic"`, `"cv"`, `"loo"` |
| `PLM` | `"profile_bic"` (local linear refit, eq. 10), `"naive_bic"` (eq. 7) |
| `GPLM` | `"refit_bic"` (eq. 5.17), `"naive_bic"` (eq. 4.11) |

For time series, use `RSO(criterion="cv", cv=TimeSeriesSplit(5))`.

`criterion="loo"` is the closed-form leave-one-out error (PRESS),
`e_i / (1 - h_ii)` with `H` the hat matrix of the reported fit. It costs one fit per
budget and needs no scikit-learn. It holds `gamma` fixed at the full-sample estimate,
so it is optimistic relative to exact leave-one-out, which is
`RSO(criterion="cv", cv=LeaveOneOut())` (one fit per observation and budget). Like AIC,
LOO targets prediction error and may keep a few more variables than BIC.

### Stochastic discount factors (MSRR)

Ridge estimates of SDF weights minimize `mean((1 - F @ lam)**2) + penalty`, where `F`
holds the factor returns (T x P). That is a regression of a vector of ones on `F`
with no intercept, so RSO applies directly, and `"loo"` gives leave-one-period-out
selection of the budget:

```python
P = F_train.shape[1]
z = 10.0 ** np.arange(-10, 4)      # ridge penalties, e.g. Kelly et al.'s grid
m = RSO(criterion="loo", taus=P / z, fit_intercept=False, standardize=False,
        refit=False)
m.fit(F_train, np.ones(len(F_train)))
lam = m.coef_                      # SDF / portfolio weights, zero for dropped factors
r_sdf = F_test @ lam               # out-of-sample SDF portfolio returns
```

Keep `fit_intercept=False`: an intercept would absorb the vector of ones and set every
weight to zero. With uniform `gamma`, the RSO penalty is `(P / tau) * lam @ lam`, so
`taus = P / z` puts the budgets on the same scale as a ridge penalty `z` (the default
`taus` assume standardized columns and are too small for raw returns). `P / m.tau_` is
the ridge-equivalent penalty of the selected budget.

## Main parameters

**`RSO`**

| parameter | default | meaning |
|---|---|---|
| `tau` | `None` | fixed budget; `None` tunes over `taus` |
| `taus` | `p * logspace(0.05, 50, 30)` | candidate budgets (X standardized) |
| `criterion` | `"profile_bic"` | how `tau` is chosen |
| `refit` | `True` | final coefficients by OLS on the selected predictors |
| `select_threshold` | `1e-2` | `gamma_k` above this means selected |

**`PLM` / `GPLM`**

| parameter | default | meaning |
|---|---|---|
| `taus1`, `taus2` | log grids (12 x 8 / 8 x 5) | budgets for `gamma` and `lambda` |
| `criterion` | see table above | how the budgets are chosen |
| `refit` | `True` | final fit without the ridge penalty on the selected model |
| `family` (`GPLM`) | `"binomial"` | `"binomial"`, `"poisson"`, `"gaussian"` |
| `max_sweeps` | `5` / `9` | coordinate descent passes |

Shared optimizer settings: `grid=(0, .2, .5, .8, 1)`, `tol=1e-4`, `standardize=True`.
After fitting, `path_` holds every budget on the grid with its `gamma`, `lambda`,
selected sets and criterion values, which is enough to draw solution paths.

## Reproducing the published results

The defaults implement the estimators as written in the papers. Setting
`r_compat=True` reproduces the original R implementation used for the numerical results
in the papers; the test suite checks this line by line against ports of the R functions.
The two modes differ in:

| | default | `r_compat=True` |
|---|---|---|
| local constant kernel | product Gaussian | `exp(-(sum_j lambda_j abs(dz_j))^2)` |
| GPLM weights in the beta step | `W` | `sqrt(W / sum W)` |
| GPLM weights in the smoother | sample weights `w_k` | unweighted |
| GPLM BIC | `-2 loglik + log(n) df` | `-loglik + log(n) df` |
| GPLM refit bandwidths | selected `lambda` | uniform over selected predictors |
| standardization | X scaled, Z centered and scaled | none |

## Tests

```bash
pip install -e ".[test]"
pytest
```

## References

If you use this package, please cite:

- Wu, Y. (2021). Can't ridge regression perform variable selection? *Technometrics*,
  63(2), 263–271. https://doi.org/10.1080/00401706.2020.1791254
- Lu, Y., Dong, Y., Hu, J. and Wu, Y. (2024). A unified approach to variable selection
  for partially linear models. *Journal of Computational and Graphical Statistics*, 33(1).
  https://doi.org/10.1080/10618600.2023.2216254
- Lu, Y., Hu, J. and Wu, Y. (2025). A new approach to select linear and nonparametric
  predictors simultaneously for generalised partially linear models. *Journal of
  Nonparametric Statistics*, 37(4), 1298–1316. https://doi.org/10.1080/10485252.2025.2556412

```bibtex
@article{wu2021ridge,
  author  = {Wu, Yichao},
  title   = {Can't Ridge Regression Perform Variable Selection?},
  journal = {Technometrics},
  volume  = {63},
  number  = {2},
  pages   = {263--271},
  year    = {2021},
  doi     = {10.1080/00401706.2020.1791254}
}

@article{lu2024plm,
  author  = {Lu, Youhan and Dong, Yushen and Hu, Juan and Wu, Yichao},
  title   = {A Unified Approach to Variable Selection for Partially Linear Models},
  journal = {Journal of Computational and Graphical Statistics},
  volume  = {33},
  number  = {1},
  year    = {2024},
  doi     = {10.1080/10618600.2023.2216254}
}

@article{lu2025gplm,
  author  = {Lu, Youhan and Hu, Juan and Wu, Yichao},
  title   = {A New Approach to Select Linear and Nonparametric Predictors Simultaneously
             for Generalised Partially Linear Models},
  journal = {Journal of Nonparametric Statistics},
  volume  = {37},
  number  = {4},
  pages   = {1298--1316},
  year    = {2025},
  doi     = {10.1080/10485252.2025.2556412}
}
```

An R implementation of linear RSO by other authors is available on CRAN as `RSO`.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
