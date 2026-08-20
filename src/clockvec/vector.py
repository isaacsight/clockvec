"""Conformance vector format.

A vector is an input plus the answers different implementations give for it.
It deliberately does not always name a winner. Where the source paper is
unambiguous, one answer is right and the rest are bugs. Where the paper is
silent, the disagreement is the finding, and asserting a winner would be
inventing a standard rather than documenting one.

Two patterns here are lifted from Wycheproof (C2SP/wycheproof, 144,886 test
cases across 104 algorithms, adopted with no mandate behind it):

  1. A three-valued result. `acceptable` covers cases where the spec is
     genuinely ambiguous, which lets a suite ship without claiming
     certainty it does not have. A field whose canonical methods are
     defined in prose needs this far more than cryptography does.

  2. Flags carrying a bug type and citations. A failing case should not
     only say "wrong" but name which known divergence it reproduces. In
     Wycheproof those citations are CVEs; here they are the GitHub issues
     where researchers hit the divergence one at a time with nothing to
     appeal to.

The format is static files with a versioned schema and no server, because
the surveyed record is unkind to anything else. EVA, LiveBench and CAFASP
were all always-on benchmark services and all three are dead, while CASP
is thirty years old. GA4GH's standard won its field outright and
`ga4gh/benchmarking-tools` has had no substantive commit since April 2020.
Adoption does not fund maintenance, so this has to survive being abandoned.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = "clockvec/v1"


class Result(str, Enum):
    """Whether an implementation's answer conforms.

    ACCEPTABLE is the load-bearing one. Without it, every vector covering a
    method whose paper left a decision unstated would have to either invent
    a correct answer or be dropped, and dropping them would silently
    exclude exactly the cases that cause the most trouble in practice.
    """

    VALID = "valid"
    INVALID = "invalid"
    ACCEPTABLE = "acceptable"


@dataclass(frozen=True)
class Flag:
    """A named failure mode, with the record of where it has bitten.

    `citations` should point at concrete evidence: an issue URL, a DOI, a
    source permalink. A vector that fails and cites biolearn#101 tells a
    maintainer something a bare assertion never could.
    """

    name: str
    bug_type: str  # "SpecAmbiguity" | "ImputationPolicy" | "Normalization" | ...
    description: str
    citations: tuple[str, ...] = ()


@dataclass(frozen=True)
class Declaration:
    """The decisions a published paper usually leaves unstated.

    Every field here exists because some implementation made a choice and
    did not write it down. Recording them is most of the point: two
    implementations that disagree while declaring different policies are
    not in conflict, they are answering different questions.
    """

    missing_cpg_policy: str  # "none" | "mean_impute" | "zero" | "knn"
    imputation_source: str | None  # e.g. "GSE55763" for dnaMethyAge golden_ref
    normalization: str | None  # e.g. "quantile_to_target"
    normalization_reference: str | None
    notes: str = ""


@dataclass(frozen=True)
class Authority:
    """Where this vector does and does not claim to decide anything.

    Modelled on GIAB's confident-region BED. A truth set without a mask is
    not merely incomplete, it is misleading: every call outside the covered
    region becomes unclassifiable while still looking like a verdict.
    Stating the limit is what makes the claim survivable.
    """

    claims: str  # what the vector decides
    does_not_claim: str  # what it is silent on
    basis: str  # why the claim is defensible: paper section, source permalink


@dataclass(frozen=True)
class Expectation:
    """What one implementation returns, and how close counts as agreeing."""

    implementation: str  # "dnaMethyAge" | "biolearn" | ...
    version: str
    values: tuple[float, ...]
    tolerance: float
    tolerance_rationale: str
    result: Result = Result.VALID
    flags: tuple[str, ...] = ()  # Flag.name references


@dataclass
class Vector:
    """One conformance case."""

    vector_id: str
    clock: str
    source_paper: str
    seed: int
    shape: tuple[int, int]  # (n_probes, n_samples)
    declaration: Declaration
    authority: Authority
    expectations: list[Expectation] = field(default_factory=list)
    flags: dict[str, Flag] = field(default_factory=dict)
    schema: str = SCHEMA_VERSION

    def manifest(self) -> dict[str, Any]:
        """The hashed portion. Excludes expectations on purpose.

        The id identifies the *question*, not the answers. Adding a seventh
        implementation's results must not change the identity of the vector
        those results are about, or every prior citation of it breaks.
        """
        return {
            "schema": self.schema,
            "clock": self.clock,
            "source_paper": self.source_paper,
            "seed": self.seed,
            "shape": list(self.shape),
            "declaration": asdict(self.declaration),
            "authority": asdict(self.authority),
        }

    @property
    def manifest_id(self) -> str:
        return content_id(self.manifest())

    def validate(self) -> None:
        """Reject vectors that would mislead. Called before serialization."""
        known = set(self.flags)
        for exp in self.expectations:
            unknown = set(exp.flags) - known
            if unknown:
                raise ValueError(
                    f"{exp.implementation} references undefined flags: {sorted(unknown)}"
                )
            if exp.tolerance < 0:
                raise ValueError(f"{exp.implementation} has negative tolerance")
            if not exp.tolerance_rationale.strip():
                raise ValueError(
                    f"{exp.implementation} declares a tolerance with no rationale. "
                    "An undefended tolerance is the thing that turns a conformance "
                    "suite into a marketing number."
                )

    def disagreements(self) -> list[tuple[str, str, float, float]]:
        """Pairs differing by more than the looser tolerance.

        Returns (impl_a, impl_b, max_abs_diff, tolerance_applied).

        The looser of the two tolerances is used rather than the tighter.
        Judging a pair under a tolerance one side never accepted would
        manufacture disagreements nobody asserted.

        Pairs where either side is ACCEPTABLE are skipped: the vector has
        already declared it does not decide between them, so reporting a
        difference as a conflict would contradict its own authority
        statement.
        """
        out: list[tuple[str, str, float, float]] = []
        for i, a in enumerate(self.expectations):
            for b in self.expectations[i + 1 :]:
                if Result.ACCEPTABLE in (a.result, b.result):
                    continue
                if len(a.values) != len(b.values):
                    out.append((a.implementation, b.implementation, float("inf"), 0.0))
                    continue
                tol = max(a.tolerance, b.tolerance)
                diff = max(abs(x - y) for x, y in zip(a.values, b.values))
                if diff > tol:
                    out.append((a.implementation, b.implementation, diff, tol))
        return out


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
