# clockvec

**A benchmark says which tool is closer to reality. A conformance vector says
which implementation is correct.** This package publishes the second kind: an
input, the answer the specification requires, and the tolerance under which an
implementation counts as agreeing.

Open-source scientific libraries are overwhelmingly tested against reference
output they generated themselves. A library saves its own result to a golden
file on first run, and every later test compares against that file. If the
implementation was wrong from the beginning, the test passes forever. The test
records what the program did, not what the specification required.

Surveying twelve technical domains for a case where one implementation
validated its numbers against an independent implementation rather than its own
history turned up none.

The name comes from epigenetic clocks, which is where this started. The gap
turned out to hold everywhere, so the schema is domain-agnostic and the
coverage is not.

## Install

```bash
pip install -e ".[dev]"
pytest -q
```

Python >= 3.10. numpy is the only runtime dependency, deliberately: a suite
that needed librosa and torch installed would be un-runnable in exactly the CI
environments these vectors are meant to be vendored into.

## What a vector contains

```python
import json
v = json.load(open("vectors/mfcc_log-base.json"))
v.keys()          # expectations, flags, manifest, manifest_id, vector_id
```

Each expectation names an implementation, the values it produces, a `result`,
a numeric `tolerance`, and — the field that matters most — a
`tolerance_rationale` stating how that number was derived. From
`vectors/mfcc_log-base.json`:

> torchaudio is float32 end to end, so its error scales with coefficient
> magnitude (max|c| = 74.93 here). tol = 16 * eps32 * max|c|; 16 is the
> sqrt(257) accumulation bound for the mel matmul, against a worst observed
> factor of 3.82. An absolute tolerance transferred from another signal would
> be unsatisfiable here.

A tolerance measured on one input is not a tolerance. It has to be derived from
the arithmetic and scale with magnitude where the error does.

## Three-valued results, after Wycheproof

Results are `valid`, `invalid`, or `acceptable`. The third is load-bearing.
Where the source paper is unambiguous, one answer is right and the rest are
bugs. Where the paper is silent, the disagreement *is* the finding, and
asserting a winner would be inventing a standard rather than documenting one.
Three of the four shipped MFCC vectors resolve to `acceptable` on both sides —
log base, mel area normalisation and window placement are conventions the
literature never fixed.

Flags carry a bug type and citations. In Wycheproof those citations are CVEs;
here they are the issues where people hit the divergence one at a time with
nothing to appeal to.

The format is static files with a versioned schema (`clockvec/v1`) and no
server, because the record is unkind to anything else. EVA, LiveBench and
CAFASP were all always-on benchmark services and all three are dead, while CASP
is thirty years old. Adoption does not fund maintenance, so this has to survive
being abandoned.

## A vector nobody can be driven to produce is not a test

`vectors/mfcc_whole-clip-db-clamp.json` records a real one. Appending a +94 dB
burst to a clip changes the coefficients of frames *before* it, because
`power_to_db` clamps against the maximum of the whole array. A function whose
output depends on the rest of the array is not checkable frame by frame.

Neither librosa nor torchaudio can be driven to the unclamped behaviour through
their public APIs — librosa's `feature.mfcc` accepts no `top_db` and forwards
`**kwargs` to `melspectrogram` rather than `power_to_db`, and torchaudio's
`MFCC` hardcodes `top_db = 80.0`. The shipped vectors are runnable against both
only because their signals keep the log-mel dynamic range under 80 dB, where
the clamp is inert. That constraint is enforced in
`mfcc_vectors.build_vectors`, not left to the author to remember.

Every library is driven through its public API only. No private functions, no
monkeypatching: a conformance result obtained by reaching inside an
implementation is a result about a program nobody ships.

```bash
pip install librosa torchaudio
python harness/check_mfcc_libraries.py
```

## Upstream record

The vectors exist to find things. Status as of 2026-08-28:

| Reference | What | Status |
|---|---|---|
| [cdt15/lingam#197](https://github.com/cdt15/lingam/pull/197) | `RESIT.get_error_independence_p_values` silently returned the worst possible answer on every input | **Merged** 2026-08-27 |
| [trac.ffmpeg.org#10209](https://trac.ffmpeg.org/ticket/10209) | EBU Tech 3341 reference values supplied to a ticket stalled since 2023 for want of them | Open |
| [OSGeo/PROJ#4823](https://github.com/OSGeo/PROJ/issues/4823) | Ten GIGS conformance cases disabled since ~2018; the 52xx subset appears to fail on `+init=` syntax rather than accuracy | Open |
| [librosa#2095](https://github.com/librosa/librosa/pull/2095) | Expose `top_db` on features that call `power_to_db` | Open |
| [cggh/scikit-allel#456](https://github.com/cggh/scikit-allel/issues/456) | `patterson_fst` is algebraically identical to `hudson_fst`; proof in `proofs/` | Open |
| [yiluyucheng/dnaMethyAge#21](https://github.com/yiluyucheng/dnaMethyAge/issues/21) | DunedinPACE scoring forms shown equivalent; proof in `proofs/` | Open |

Two findings of mine were downgraded on review — one was a units convention
rather than a defect, one had a legitimate numerical cause. Both corrections
are in the public record. A conformance project that cannot tell a real
divergence from an apparent one is worse than none.

## Status

Early. 80 tests pass; four MFCC vectors ship. The format is settled enough to
build on and the coverage is not yet broad enough to depend on.

## License

MIT. See [LICENSE](./LICENSE).
