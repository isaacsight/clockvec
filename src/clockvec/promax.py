"""Reference Promax rotation, ported line for line from R's stats::promax.

Promax (Hendrickson & White 1964) is two steps: an orthogonal varimax
rotation (Kaiser 1958), then a least-squares fit of the varimax loadings to
a power of themselves, which relaxes orthogonality. The step the literature
leaves open is whether the varimax stage first rescales each variable to
unit communality -- Kaiser's "normal varimax" -- before rotating and undoes
the scaling afterwards. Kaiser 1958 defines both raw and normal varimax and
recommends the normal one; Hendrickson & White take "the orthogonal
solution" as given and do not say which.

Every convention is a required argument and nothing has a default, for the
reason `mfcc.mfcc` has none: each of these exists because an implementation
chose it silently, and the specific silent choice this module documents is
`kaiser_normalize`, which one widely used package changed between two
releases without exposing it (see `promax_vectors`).

The varimax iteration is the one in R's stats::varimax (Kaiser's criterion
maximized by iterated SVD of the gradient, stopping when the sum of singular
values grows by less than `eps`), not GPArotation's gradient projection.
Both converge to the same optimum; the two differ by the stopping
tolerance, and that difference is what the vectors' tolerance is derived
from.
"""

from __future__ import annotations

import numpy as np


def varimax(
    loadings: np.ndarray,
    *,
    kaiser_normalize: bool,
    eps: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Varimax-rotated loadings and the orthogonal rotation applied.

    Port of R stats::varimax. `kaiser_normalize=True` divides each row by
    its length before iterating and multiplies it back afterwards, which is
    Kaiser 1958's normal varimax. R's default is True; GPArotation's is
    False; the difference is the whole subject of `promax_vectors`.
    """
    x = np.asarray(loadings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"loadings must be 2-D, got shape {x.shape}")
    p, k = x.shape
    if k < 2:
        raise ValueError("varimax needs at least two factors")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    if kaiser_normalize:
        sc = np.sqrt((x**2).sum(axis=1))
        if (sc == 0).any():
            raise ValueError("a variable with zero communality cannot be normalized")
        x = x / sc[:, None]

    rot = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        z = x @ rot
        b = x.T @ (z**3 - z @ np.diag((z**2).sum(axis=0)) / p)
        u, s, vt = np.linalg.svd(b)
        rot = u @ vt
        d_past, d = d, float(s.sum())
        if d < d_past * (1.0 + eps):
            break

    z = x @ rot
    if kaiser_normalize:
        z = z * sc[:, None]
    return z, rot


def promax(
    loadings: np.ndarray,
    *,
    m: float,
    kaiser_normalize: bool,
    eps: float,
    max_iter: int,
) -> dict[str, np.ndarray]:
    """Promax-rotated loadings, the total rotation, and the factor correlations.

    Port of R stats::promax with the normalization exposed. The varimax
    loadings are raised to the power `m` (sign preserved), the varimax
    loadings are regressed onto that target, and the resulting transform is
    column-normalized so the factors keep unit variance.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    z, rot = varimax(loadings, kaiser_normalize=kaiser_normalize, eps=eps, max_iter=max_iter)
    target = z * np.abs(z) ** (m - 1)
    u, *_ = np.linalg.lstsq(z, target, rcond=None)
    d = np.diag(np.linalg.inv(u.T @ u))
    u = u @ np.diag(np.sqrt(d))
    out = z @ u
    total = rot @ u
    ui = np.linalg.inv(total)
    phi = ui @ ui.T
    return {"loadings": out, "rotmat": total, "phi": phi}


def canonicalize(loadings: np.ndarray) -> np.ndarray:
    """Fix the two ambiguities every rotation leaves open.

    A rotated solution is defined only up to column order and column sign;
    two implementations can return the same solution with factors swapped
    or reflected. Comparing raw arrays would report that as a disagreement
    of order 1. Columns are sorted by descending sum of squares and each is
    reflected so its largest-magnitude entry is positive. Any implementation
    can be brought to this form from its public output, so the convention
    costs nothing and is declared in every vector that uses it.
    """
    z = np.asarray(loadings, dtype=np.float64)
    order = np.argsort(-(z**2).sum(axis=0), kind="stable")
    z = z[:, order]
    signs = np.sign(z[np.abs(z).argmax(axis=0), np.arange(z.shape[1])])
    signs[signs == 0] = 1.0
    return z * signs
