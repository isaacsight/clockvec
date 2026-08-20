"""Proof for dnaMethyAge issue #21 — DunedinPACE scoring equivalence.

Question asked (Lingyu Zhan, 2025-11-07, zero replies as of 2026-08-20):

    dnaMethyAge scores DunedinPACE with a plain matrix multiply. Belsky's
    original DunedinPACE package uses

        score = model_intercept +
                rowSums(t(betas.mat[probes, ]) %*% diag(weights))

    Do these two give the same result?

Answer: yes, for the scoring step, and the two forms are algebraically
identical rather than merely close. This file proves it numerically.

Where dnaMethyAge puts the intercept
------------------------------------
R/MethyAge.R appends a synthetic all-ones row to the beta matrix before the
multiply:

    betas <- rbind(betas, Intercept=1)                       # ~line 151
    ...
    m_age <- t(betas) %*% matrix(data=coefs[rownames(betas)])  # ~line 182

and data/DunedinPACE.rda carries a coefficient entry named `Intercept`
(verified 2026-08-20 by extracting strings from the gzipped RData; the
serialized object is named `coefs` and contains an `Intercept` level
alongside 20519 cg-prefixed probe IDs).

So the intercept is not missing. It rides in as one more term of the dot
product, which is exactly what adding it separately does.

DunedinPACE is also correctly absent from every post-transformation branch
in MethyAge.R (lines ~185-215). It should be: DunedinPACE reports a rate of
aging, not an age, so the Horvath log-linear transform must not apply.

The algebra
-----------
Let B be probes x samples, w the weight vector over probes, k the intercept.

    diag(w) scales column p of t(B) by w[p], so

        (t(B) @ diag(w))[s, p] = B[p, s] * w[p]

    and summing across p:

        rowSums(t(B) @ diag(w))[s] = sum_p B[p, s] * w[p] = (t(B) @ w)[s]

    Therefore

        k + rowSums(t(B) @ diag(w))  ==  t(rbind(B, 1)) @ concat(w, k)

Both sides are the same dot product. The `diag`/`rowSums` spelling is a
notational detour, not a different model.

What this does NOT prove
------------------------
Only the scoring step. Full-pipeline agreement additionally depends on
preprocessing, which is where any real numerical difference will live:

  * dnaMethyAge reimplements Belsky's quantile normalization in
    R/preprocessDunedinPACE.R (its header credits PoAmProjector.R) with
    `least_proportion = 0.8`.
  * Missing probes are mean-imputed from `golden_ref`, derived from
    GSE55763 (2,664 blood samples; see dnaMethyAge issue #19, resolved).

If someone reports differing DunedinPACE values between the two packages,
the imputation path is where to look, not the matrix multiply.

Run: python proofs/dunedin_pace_equivalence.py
"""

import numpy as np

SEED = 21  # the issue number, so the vector is self-describing


def belsky_form(betas, weights, intercept):
    """model_intercept + rowSums(t(betas) %*% diag(weights))

    Transcribed literally from PoAmProjector.R, including the detour
    through diag() that the issue asks about.
    """
    scaled = betas.T @ np.diag(weights)  # samples x probes
    return intercept + scaled.sum(axis=1)


def dnamethyage_form(betas, weights, intercept):
    """t(rbind(betas, Intercept=1)) %*% coefs

    Transcribed from MethyAge.R: the intercept enters as an appended
    all-ones probe row rather than as a separate additive term.
    """
    n_samples = betas.shape[1]
    augmented = np.vstack([betas, np.ones((1, n_samples))])
    coefs = np.concatenate([weights, [intercept]])
    return augmented.T @ coefs


def main():
    rng = np.random.default_rng(SEED)

    n_probes, n_samples = 173, 12  # DunedinPACE uses 173 CpGs
    # Beta values are bounded [0, 1]; a Beta(2, 2) draw keeps them in range
    # and away from the degenerate all-0 / all-1 matrices that crash
    # density-based imputation in several implementations.
    betas = rng.beta(2.0, 2.0, size=(n_probes, n_samples))
    weights = rng.normal(0.0, 0.05, size=n_probes)
    intercept = -0.06  # sign and scale are irrelevant to the identity

    a = belsky_form(betas, weights, intercept)
    b = dnamethyage_form(betas, weights, intercept)

    max_abs = np.max(np.abs(a - b))
    max_ulp = np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-300))

    print(f"seed                  {SEED}")
    print(f"probes x samples      {n_probes} x {n_samples}")
    print(f"belsky_form[:3]       {np.array2string(a[:3], precision=17)}")
    print(f"dnamethyage_form[:3]  {np.array2string(b[:3], precision=17)}")
    print(f"max |difference|      {max_abs:.3e}")
    print(f"max relative diff     {max_ulp:.3e}")
    print(f"bitwise identical     {bool(np.array_equal(a, b))}")
    print(f"within 1e-12          {bool(np.allclose(a, b, rtol=0, atol=1e-12))}")

    # The identity is exact in real arithmetic. In floating point the two
    # orders of summation can differ in the last bits, so assert a
    # tolerance rather than bitwise equality — and report which we got.
    assert np.allclose(a, b, rtol=0, atol=1e-12), "forms disagree beyond 1e-12"
    print("\nVERDICT: equivalent. The two spellings compute the same dot product.")


if __name__ == "__main__":
    main()
