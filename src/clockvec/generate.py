"""Synthetic beta-matrix generation.

Synthetic rather than real data, for three reasons: no consent or IRB
question, the vectors can be redistributed freely, and a seed is smaller
than a matrix so a vector stays checkable without hosting gigabytes.

The distributions are not arbitrary. Real methylation beta values are
bimodal, clustering near 0 (unmethylated) and near 1 (methylated) with a
sparser middle. Drawing uniform betas produces matrices that no real array
would ever yield, and several implementations break on them outright:
methylclock #13 is a crash inside `density.default` when too few values
fall in a fitted window, and dnaMethyAge's BMIQ path fits a three-class
beta mixture that assumes the real shape. A generator that ignores this
tests error handling instead of the clock.
"""

from __future__ import annotations

import numpy as np

# Fraction of probes drawn from each mode. Roughly matches the shape of a
# blood 450k/EPIC array: most probes sit at one extreme, a minority are
# intermediate. Exact proportions matter less than avoiding a flat middle.
_UNMETHYLATED_FRAC = 0.45
_METHYLATED_FRAC = 0.40
# remainder is intermediate


def bimodal_betas(
    n_probes: int,
    n_samples: int,
    seed: int,
    *,
    noise: float = 0.0,
) -> np.ndarray:
    """Deterministic bimodal beta matrix in [0, 1], shape (n_probes, n_samples).

    `noise` adds per-sample technical jitter, which is how a replicate pair
    is built: same seed, different noise draw. That is the setup the whole
    reliability literature rests on, so the generator should express it
    directly rather than leaving callers to improvise.
    """
    if not 0 <= noise < 0.5:
        raise ValueError(f"noise must be in [0, 0.5), got {noise}")

    rng = np.random.default_rng(seed)

    n_unmeth = int(n_probes * _UNMETHYLATED_FRAC)
    n_meth = int(n_probes * _METHYLATED_FRAC)
    n_mid = n_probes - n_unmeth - n_meth

    # Beta(a, b) with a < b leans low; a > b leans high; a == b is centred.
    blocks = [
        rng.beta(1.2, 9.0, size=(n_unmeth, n_samples)),
        rng.beta(9.0, 1.2, size=(n_meth, n_samples)),
        rng.beta(3.0, 3.0, size=(n_mid, n_samples)),
    ]
    betas = np.vstack(blocks)

    if noise:
        betas = betas + rng.normal(0.0, noise, size=betas.shape)
        # Clip rather than reject: real arrays also saturate at the bounds,
        # and rejection sampling would break seed-determinism of the shape.
        betas = np.clip(betas, 0.0, 1.0)

    return betas


def replicate_pair(
    n_probes: int, n_samples: int, seed: int, noise: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two matrices that are the same biology measured twice.

    Any clock worth using should return nearly the same answer for both.
    Higgins-Chen et al. 2021 measured deviations up to 9 years across six
    prominent clocks on real replicate pairs, so "nearly" is doing a lot of
    work and this is the function that lets a vector say so.
    """
    a = bimodal_betas(n_probes, n_samples, seed, noise=noise)
    # Offset the seed rather than reusing the generator, so each half is
    # independently reproducible from a seed a reader can write down.
    b = bimodal_betas(n_probes, n_samples, seed + 1_000_000, noise=noise)
    return a, b


def with_missing(
    betas: np.ndarray, fraction: float, seed: int
) -> np.ndarray:
    """Punch NaN holes to exercise each implementation's imputation policy.

    Missing-CpG handling is the single largest declared difference between
    these libraries, and biolearn #101 (GrimAge not matching Clock
    Foundation, unresolved since 2024) is most likely a case of it. A vector
    that never exercises the missing path cannot surface that class of bug.
    """
    if not 0 <= fraction < 1:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")

    rng = np.random.default_rng(seed)
    out = betas.copy()
    mask = rng.random(betas.shape) < fraction
    out[mask] = np.nan
    return out
