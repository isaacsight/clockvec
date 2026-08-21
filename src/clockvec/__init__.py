"""clockvec — conformance vectors for published computational methods.

Independent libraries implement the same published method and disagree on
identical input. This package publishes the inputs, the answers each
implementation gives, and the tolerance under which they count as agreeing.

It started with epigenetic clocks, which is where the name comes from. The
same gap turned out to hold across every field surveyed, so the schema is
domain-agnostic and the coverage is not.

Where a method has no published reference value, this package implements it
from the primary definitions -- see `unifrac` and `mfcc`. That is a last
resort, not a goal: an implementation is only a defensible reference when
the paper fixes the answer and the vector says exactly which paper, which
section, and which decisions the paper left open.
"""

__version__ = "0.0.1"

from clockvec.generate import bimodal_betas, replicate_pair, with_missing
from clockvec.mfcc import MelScale, mel_filterbank, mfcc
from clockvec.vector import (
    Authority,
    Declaration,
    Expectation,
    Flag,
    Result,
    Vector,
    canonical_json,
    content_id,
)

__all__ = [
    "bimodal_betas",
    "replicate_pair",
    "with_missing",
    "Authority",
    "Declaration",
    "Expectation",
    "Flag",
    "MelScale",
    "Result",
    "Vector",
    "canonical_json",
    "content_id",
    "mel_filterbank",
    "mfcc",
]
