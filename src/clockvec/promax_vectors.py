"""Conformance vectors for Promax rotation.

Why this method. Factor rotation is where a psychology paper's numbers are
made, and Promax is the oblique rotation most of them use. The R package
`psych` is the dominant implementation (CRAN download rank in the top few
hundred of ~20,000 packages) and its `Promax` accepts a `normalize`
argument that is not connected to anything: `normalize=TRUE` and
`normalize=FALSE` return bit-identical output. Between psych 2.6.3 and
2.6.5 the varimax stage moved from `stats::varimax` (normalize=TRUE by
default) to `GPArotation::Varimax` (normalize=FALSE by default), so every
Promax rotation quietly changed from normalized to un-normalized with
nothing in NEWS and no argument that could restore the old answer.

Reported to the maintainer by email on 2026-08-21 (psych has no public
issue tracker); acknowledged 2026-09-01 with a holding note. The vector
here is the artifact that report lacked: an input, both answers, a
tolerance with a derivation, and a flag naming the defect, so the next
implementation can be checked against it without repeating the analysis.

The input is synthetic and seeded. Real loading matrices carry no consent
question, but a seed is smaller than a matrix and the generator is the
specification. The matrix is also stored in the declaration in full, so
an R user with no numpy can run the vector from the JSON alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clockvec.promax import canonicalize, promax
from clockvec.vector import Authority, Expectation, Flag, Result, Vector

__all__ = ["PromaxDeclaration", "FLAGS", "loadings", "build_vectors", "write_vectors"]

SOURCE = (
    "Hendrickson & White 1964, Br. J. Stat. Psychol. 17(1):65-70; "
    "Kaiser 1958, Psychometrika 23(3):187-200"
)
SEED = 20260901


@dataclass(frozen=True)
class PromaxDeclaration:
    """Every convention that must be fixed before two Promax solutions compare.

    No defaults, same rule as `MfccDeclaration`. `input_loadings` is the
    unrotated matrix itself, row-major, so the vector is self-contained;
    `seed` in the manifest says how it was made, the field says what it is.
    """

    n_variables: int
    n_factors: int
    power_m: float
    kaiser_normalize: bool
    varimax_eps: float
    varimax_max_iter: int
    varimax_start: str  # "identity"
    column_canonicalization: str  # "ss-descending, abs-max-positive"
    input_loadings: tuple[float, ...]
    notes: str = ""

    def kwargs(self) -> dict:
        return dict(
            m=self.power_m,
            kaiser_normalize=self.kaiser_normalize,
            eps=self.varimax_eps,
            max_iter=self.varimax_max_iter,
        )


def loadings(n_variables: int, n_factors: int, seed: int) -> np.ndarray:
    """Deterministic unrotated loadings with a real rotation to recover.

    Simple structure with one cross-loading per variable, communalities
    spread over [0.15, 0.85], then spun by a random orthogonal matrix so the
    varimax stage has work to do. The spread matters: Kaiser normalization
    exists to stop high-communality variables dominating the criterion, so
    on equal communalities the two conventions coincide and a vector built
    there would document nothing.
    """
    rng = np.random.default_rng(seed)
    p, k = n_variables, n_factors
    L = np.zeros((p, k))
    for i in range(p):
        L[i, i % k] = 1.0
        L[i, (i + 1) % k] = rng.uniform(0.15, 0.55)
    h = rng.uniform(0.15, 0.85, size=p)
    L = L / np.linalg.norm(L, axis=1, keepdims=True) * np.sqrt(h)[:, None]
    q, _ = np.linalg.qr(rng.standard_normal((k, k)))
    return L @ q


def tolerance_convergence(eps: float) -> tuple[float, str]:
    """Tolerance for two correct varimax implementations that stop differently.

    R's stats::varimax stops when the sum of singular values of the gradient
    grows by less than `eps`; GPArotation's GPForth stops on a gradient norm
    below its own `eps`. Both default to 1e-5. Two implementations that have
    both converged sit within a small multiple of that of the optimum, and
    the promax regression does not amplify it. Ten times eps is the bound;
    observed gaps on this vector's input are 2.2e-05 (normalized) and
    7.4e-06 (un-normalized), and 4.9e-05 on an 8x2 example, all inside it.
    """
    tol = 10.0 * eps
    return tol, (
        f"stats::varimax and GPArotation::GPForth both stop at a relative "
        f"criterion of {eps:g}; two converged implementations differ by a small "
        f"multiple of that. tol = 10 * eps = {tol:g}. Measured stats vs GPA on "
        f"this input: 2.2e-05 normalized, 7.4e-06 un-normalized; 4.9e-05 on an "
        f"8x2 example. The normalization effect on this input is 1.38e-03, "
        f"fourteen times the tolerance, so the two expectations cannot be "
        f"confused by any converged implementation."
    )


FLAGS: dict[str, Flag] = {
    "DeadNormalizeArgument": Flag(
        name="DeadNormalizeArgument",
        bug_type="DeadArgument",
        description=(
            "psych::Promax(x, m, normalize, pro.m, Tmat, ...) declares `normalize` "
            "as a formal argument and documents it as 'parameter passed to "
            "optimization routine', but the identifier does not occur in the "
            "function body. Because it is a named formal it is captured and never "
            "reaches `...`, so it cannot be forwarded to GPArotation::Varimax by "
            "that route either. Measured on psych 2.6.5: Promax(L, normalize=TRUE) "
            "and Promax(L, normalize=FALSE) differ by exactly 0.0; "
            "GPArotation::Varimax under the same two values differs by 1.38e-02. "
            "Fix is one token: `GPArotation::Varimax(x, Tmat = Tmat, normalize = "
            "normalize, ...)`. Verified to leave the shipped default bit-identical "
            "and to recover the pre-2.6.5 result within 4.9e-05."
        ),
        citations=(
            "https://cran.r-project.org/package=psych",
            "email to the maintainer 2026-08-21; holding reply 2026-09-01",
        ),
    ),
    "SilentDefaultFlip": Flag(
        name="SilentDefaultFlip",
        bug_type="VersionDrift",
        description=(
            "psych <= 2.6.3 computed the varimax stage with stats::varimax, whose "
            "default is normalize=TRUE. psych 2.6.5 computes it with "
            "GPArotation::Varimax behind a hardcoded `GPA <- TRUE`, whose default "
            "is normalize=FALSE. The user-visible argument did nothing in either "
            "version, so the effective behaviour changed from always-normalize to "
            "never-normalize with no NEWS entry and no way to request the old "
            "result. Measured: 1.38e-03 on this vector's input; up to 0.94 after "
            "optimal column alignment on ill-conditioned 8-variable matrices, "
            "where five of sixteen loadings change sign. The maintainer's first "
            "hypothesis, the new n.rotations multi-start option, moves "
            "fa(rotate='Promax') by 1.5e-06 to 5.5e-06 and does not reach "
            "Promax() at all."
        ),
        citations=(
            "https://cran.r-project.org/package=psych",
            "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/varimax.html",
        ),
    ),
}


def _pinned(**over) -> PromaxDeclaration:
    L = loadings(12, 3, SEED)
    base = dict(
        n_variables=12, n_factors=3, power_m=4.0, kaiser_normalize=True,
        varimax_eps=1e-5, varimax_max_iter=1000, varimax_start="identity",
        column_canonicalization="ss-descending, abs-max-positive",
        input_loadings=tuple(float(v) for v in L.ravel(order="C")),
    )
    base.update(over)
    return PromaxDeclaration(**base)


def _expectation(name: str, decl: PromaxDeclaration, *, result: Result,
                 flags: tuple[str, ...] = ()) -> Expectation:
    L = np.asarray(decl.input_loadings).reshape(decl.n_variables, decl.n_factors)
    out = canonicalize(promax(L, **decl.kwargs())["loadings"])
    tol, rationale = tolerance_convergence(decl.varimax_eps)
    return Expectation(
        implementation=name, version="clockvec-reference",
        values=tuple(float(v) for v in out.ravel(order="C")),
        tolerance=tol, tolerance_rationale=rationale, result=result, flags=flags,
    )


def build_vectors() -> list[Vector]:
    """One vector, one divergence.

    Hendrickson & White do not fix the normalization, so both answers are
    ACCEPTABLE and `disagreements` stays silent. The defect is not that
    psych picked one; it is that psych's argument for choosing is dead and
    its effective choice changed between releases. Both are flags on the
    expectations, and the R harness is where a failing check names them.
    """
    v = Vector(
        vector_id="promax/kaiser-normalization",
        method="Promax",
        source_paper=SOURCE,
        seed=SEED,
        input_shape=(12, 3),
        declaration=_pinned(),
        authority=Authority(
            claims=(
                "Under the declared conventions, these are the Promax loadings "
                "after column canonicalization. The two expectations differ ONLY "
                "in kaiser_normalize. An implementation that accepts a "
                "normalization switch and returns the same expectation for both "
                "values has a dead argument, whichever value it defaults to."
            ),
            does_not_claim=(
                "Which normalization is correct. Hendrickson & White 1964 take "
                "the orthogonal starting solution as given; Kaiser 1958 defines "
                "raw and normal varimax and recommends normal, but the Promax "
                "paper does not adopt that recommendation. Both ACCEPTABLE."
            ),
            basis=(
                "Hendrickson & White 1964 sec. 2 for the two-step definition; "
                "Kaiser 1958 sec. 5 for normal varimax; R stats::promax and "
                "stats::varimax source for the reference algorithm. Measured on "
                "psych 2.6.5: Promax(normalize=TRUE) equals Promax(normalize=FALSE) "
                "to 0.0 and equals the un-normalized expectation to 0.0; the "
                "2.6.3 code path equals the normalized expectation within 2.2e-05."
            ),
        ),
        flags=dict(FLAGS),
    )
    v.expectations = [
        _expectation(
            "kaiser_normalize=True (stats::promax; psych <= 2.6.3)",
            _pinned(kaiser_normalize=True),
            result=Result.ACCEPTABLE, flags=("SilentDefaultFlip",),
        ),
        _expectation(
            "kaiser_normalize=False (GPArotation::Varimax default; psych 2.6.5)",
            _pinned(kaiser_normalize=False),
            result=Result.ACCEPTABLE,
            flags=("SilentDefaultFlip", "DeadNormalizeArgument"),
        ),
    ]
    return [v]


def write_vectors(directory: str = "vectors") -> list[str]:
    """Static JSON, same reasoning as `mfcc_vectors.write_vectors`."""
    import pathlib

    from clockvec.vector import canonical_json

    out = pathlib.Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for v in build_vectors():
        path = out / (v.vector_id.replace("/", "_") + ".json")
        path.write_text(canonical_json(v.to_dict()) + "\n", encoding="utf-8")
        written.append(str(path))
    return written


if __name__ == "__main__":
    for f in write_vectors():
        print(f)
