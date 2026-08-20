"""clockvec — conformance vectors for epigenetic clock implementations.

Six independent libraries implement the same published clocks and disagree
on identical input. This package publishes the inputs, the answers each
implementation gives, and the tolerance under which they count as agreeing.

It does not implement a clock.
"""

__version__ = "0.0.1"

from clockvec.generate import bimodal_betas, replicate_pair, with_missing
from clockvec.vector import Declaration, Expectation, Vector, canonical_json, content_id

__all__ = [
    "bimodal_betas",
    "replicate_pair",
    "with_missing",
    "Declaration",
    "Expectation",
    "Vector",
    "canonical_json",
    "content_id",
]
