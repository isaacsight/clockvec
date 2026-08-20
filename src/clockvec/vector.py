"""Conformance vector format.

A vector is an input plus the answers different implementations give for it.
It deliberately does not contain a single "correct" answer. Where the source
paper is unambiguous, one answer is right and the rest are bugs. Where the
paper is silent, the disagreement is the finding, and asserting a winner
would be inventing a standard rather than documenting one.

Content addressing follows the convention already used in
provenance-substrate: canonical JSON, sorted keys, no insignificant
whitespace, SHA-256 over UTF-8 bytes. Two people who generate the same
vector independently get the same id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any


def canonical_json(obj: Any) -> str:
    """Stable serialization. Sorted keys, tight separators, no NaN.

    `allow_nan=False` is deliberate: a NaN in a vector means the generator
    produced something unusable, and silently hashing it would bake a
    non-reproducible value into the id. Fail loudly instead.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_id(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Declaration:
    """The decisions a published paper usually leaves unstated.

    Every field here exists because some implementation made a choice here
    and did not write it down. Recording them is most of the point of the
    format: two implementations that disagree while declaring different
    policies are not in conflict, they are answering different questions.
    """

    missing_cpg_policy: str  # "none" | "mean_impute" | "zero" | "knn"
    imputation_source: str | None  # e.g. "GSE55763" for dnaMethyAge golden_ref
    normalization: str | None  # e.g. "quantile_to_target"
    normalization_reference: str | None
    notes: str = ""


@dataclass(frozen=True)
class Expectation:
    """What one implementation returns, and how close counts as agreeing."""

    implementation: str  # "dnaMethyAge" | "biolearn" | ...
    version: str
    values: list[float]
    tolerance: float
    tolerance_rationale: str


@dataclass
class Vector:
    """One conformance case.

    `underspecified` marks the case where the source paper does not
    determine the answer. Those vectors still ship: they record what each
    implementation assumed, which is the more useful artifact.
    """

    vector_id: str
    clock: str
    source_paper: str
    seed: int
    shape: tuple[int, int]  # (n_probes, n_samples)
    declaration: Declaration
    expectations: list[Expectation] = field(default_factory=list)
    underspecified: bool = False

    def manifest(self) -> dict[str, Any]:
        """The hashed portion. Excludes expectations on purpose.

        The id identifies the *question*, not the answers. Adding a seventh
        implementation's results must not change the identity of the vector
        those results are about, or every prior citation of it breaks.
        """
        return {
            "clock": self.clock,
            "source_paper": self.source_paper,
            "seed": self.seed,
            "shape": list(self.shape),
            "declaration": asdict(self.declaration),
            "underspecified": self.underspecified,
        }

    @property
    def manifest_id(self) -> str:
        return content_id(self.manifest())

    def disagreements(self) -> list[tuple[str, str, float, float]]:
        """Pairs of implementations differing by more than the looser tolerance.

        Returns (impl_a, impl_b, max_abs_diff, tolerance_applied).

        The looser of the two tolerances is used rather than the tighter.
        Claiming a violation under a tolerance one side never accepted would
        manufacture disagreements that nobody actually asserted.
        """
        out: list[tuple[str, str, float, float]] = []
        for i, a in enumerate(self.expectations):
            for b in self.expectations[i + 1 :]:
                if len(a.values) != len(b.values):
                    out.append((a.implementation, b.implementation, float("inf"), 0.0))
                    continue
                tol = max(a.tolerance, b.tolerance)
                diff = max(abs(x - y) for x, y in zip(a.values, b.values))
                if diff > tol:
                    out.append((a.implementation, b.implementation, diff, tol))
        return out
