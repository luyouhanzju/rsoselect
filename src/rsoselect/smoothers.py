"""
Kernel smoother matrices with inverse bandwidths lambda_j = 1 / h_j.

Rows index target points t_i, columns index training points z_k, so the
smoothed value at t_i is S[i] @ v for a training vector v. lambda_j = 0
removes variable j from the kernel.

local_constant(T, Z, lam, w, kernel)
    S[i, k] = w_k K_ik / sum_k' w_k' K_ik'
    kernel="product" : K_ik = exp(-sum_j lam_j^2 (t_ij - z_kj)^2)       (papers)
    kernel="r_code"  : K_ik = exp(-(sum_j lam_j |t_ij - z_kj|)^2)       (getLCSmat in R)

local_linear(T, Z, lam, w)
    product Gaussian kernel, local linear fit in all columns of Z
    (LLsmoothingFAST in R; paper eq. 8).
"""

import numpy as np


def _diffs(T, Z):
    T = np.atleast_2d(np.asarray(T, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    return T[:, None, :] - Z[None, :, :]          # (m, n, q)


def _log_kernel(T, Z, lam, kernel):
    D = _diffs(T, Z)
    lam = np.asarray(lam, dtype=float)
    if D.shape[2] == 0:
        return np.zeros(D.shape[:2])
    if kernel == "product":
        return -np.einsum("mnq,q->mn", D ** 2, lam ** 2)
    if kernel == "r_code":
        return -np.einsum("mnq,q->mn", np.abs(D), lam) ** 2
    raise ValueError("kernel must be 'product' or 'r_code'")


def local_constant(T, Z, lam, w=None, kernel="product"):
    L = _log_kernel(T, Z, lam, kernel)
    L -= L.max(axis=1, keepdims=True)            # row scaling cancels in normalization
    K = np.exp(L)
    if w is not None:
        K = K * np.asarray(w, dtype=float)[None, :]
    return K / K.sum(axis=1, keepdims=True)


def local_linear(T, Z, lam, w=None, jitter=1e-10):
    T = np.atleast_2d(np.asarray(T, dtype=float))
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    m, n, q = T.shape[0], Z.shape[0], Z.shape[1]
    L = _log_kernel(T, Z, lam, "product")
    L -= L.max(axis=1, keepdims=True)
    K = np.exp(L)
    if w is not None:
        K = K * np.asarray(w, dtype=float)[None, :]
    Dm = np.concatenate([np.ones((m, n, 1)), -_diffs(T, Z)], axis=2)   # [1, z_k - t_i]
    M = np.einsum("mn,mna,mnb->mab", K, Dm, Dm)
    M += jitter * np.trace(M, axis1=1, axis2=2)[:, None, None] * np.eye(q + 1)
    e1 = np.zeros(q + 1)
    e1[0] = 1.0
    x = np.linalg.solve(M, np.broadcast_to(e1, (m, q + 1))[..., None])[..., 0]   # M^-1 e1
    return K * np.einsum("mna,ma->mn", Dm, x)


class TrainingKernel:
    """Local-constant smoother on the training points with pairwise distances cached
    once, so repeated calls for different lambdas (inside MCD) are cheap."""

    def __init__(self, Z, kernel="product"):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        D = Z[:, None, :] - Z[None, :, :]
        self.kernel = kernel
        self.q = Z.shape[1]
        self._D = (D ** 2 if kernel == "product" else np.abs(D)).transpose(2, 0, 1).copy()

    def local_constant(self, lam, w=None):
        lam = np.asarray(lam, dtype=float)
        if self.kernel == "product":
            L = -np.tensordot(lam ** 2, self._D, axes=1)
        else:
            L = -np.tensordot(lam, self._D, axes=1) ** 2
        L -= L.max(axis=1, keepdims=True)
        K = np.exp(L)
        if w is not None:
            K = K * np.asarray(w, dtype=float)[None, :]
        return K / K.sum(axis=1, keepdims=True)
