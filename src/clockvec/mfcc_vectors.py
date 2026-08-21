"""Conformance vectors for MFCC.

Why this method and not another. Audio has both a solved and an unsolved
sub-area with almost every other variable held constant, which makes it the
cleanest natural experiment in the survey behind this package:

    broadcast loudness   EBU Tech 3341 ships free vectors with expected
                         values and a stated +/-0.1 LU tolerance. Two
                         independent implementations, pyloudnorm and
                         ffmpeg's ebur128, were measured 0.035 dB apart and
                         both hit the published number.

    MFCC                 no vectors, no standard, no reference
                         implementation. librosa and torchaudio at their own
                         defaults were measured 11.00026 apart in native
                         units on the same input.

Same field, same decade, comparable DSP, comparable openness, opposite
outcomes. The difference is that one published free machine-readable vectors
with a tolerance and the other published a 1980 paper.

Every input here is synthetic and seeded. No copyrighted audio is used or
required, which is a licensing precondition for vectors that are meant to be
vendored into other people's test suites.

Inputs are also deliberately short. A vector should be the smallest input
that exhibits the divergence it is about; a long one buries the finding in
data and makes the file harder to review than the bug.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from clockvec.mfcc import MelScale, mfcc
from clockvec.vector import Authority, Expectation, Flag, Result, Vector

__all__ = ["MfccDeclaration", "FLAGS", "hann", "signal", "build_vectors"]

SOURCE = (
    "Davis & Mermelstein 1980, IEEE Trans. Acoust. Speech Signal Process. "
    "28(4):357-366"
)
SEED = 20260821


@dataclass(frozen=True)
class MfccDeclaration:
    """Every convention that must be fixed before two MFCCs are comparable.

    No field has a default, deliberately. A frozen dataclass with no defaults
    refuses to construct unless every convention is stated, which is the same
    rule `unifrac.normalized` follows and for the same reason: each of these
    exists because some implementation chose it silently, and a vector that
    omits one is not checkable by anyone who does not already know which
    library produced it.
    """

    sample_rate: float
    n_mfcc: int
    n_fft: int
    win_length: int
    hop_length: int
    window_placement: str
    center: bool
    pad_mode: str
    n_mels: int
    f_min: float
    f_max: float
    mel_scale: str
    mel_area_normalize: bool
    power: float
    scale_by_n_fft: bool
    log_scale: str
    log_floor: float
    db_clamp_top: float | None
    dct_ortho: bool
    lifter: float
    pre_emphasis: float | None
    notes: str = ""

    def kwargs(self) -> dict:
        """Arguments for `mfcc`, with the window materialized.

        `window` is not a declaration field because it is an array, not a
        decision: what must be declared is `win_length` and the fact that the
        window is a periodic Hann, which `hann` fixes.
        """
        d = {k: v for k, v in asdict(self).items() if k != "notes"}
        d["window"] = hann(self.win_length)
        return d


def hann(n: int) -> np.ndarray:
    """Periodic Hann window, matching scipy's `fftbins=True` and numpy's
    `hanning(n+1)[:-1]`. The symmetric variant differs at every sample and is
    the wrong one for spectral analysis, which is a divergence in its own
    right and the reason this is spelled out rather than imported."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def signal(kind: str, n: int, seed: int = SEED) -> np.ndarray:
    """Deterministic synthetic inputs. `kind` is part of the vector identity."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000.0
    if kind == "tone_plus_noise":
        return 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * rng.standard_normal(n)
    if kind == "quiet_noise":
        return 0.001 * rng.standard_normal(n)
    if kind == "harmonic_stack":
        # The noise floor is not decoration. A pure harmonic stack has almost
        # no energy in the upper mel bands, giving a 127.7 dB log-mel range,
        # and any vector built on it is unrunnable against librosa and
        # torchaudio for the reason in `RUNNABLE_DYNAMIC_RANGE_DB`. With the
        # floor the range is bounded and the vector is checkable.
        stack = sum(0.2 / k * np.sin(2 * np.pi * 220.0 * k * t) for k in range(1, 13))
        return stack + 0.002 * rng.standard_normal(n)
    raise ValueError(f"unknown signal kind {kind!r}")


# A constraint on vector DESIGN, forced by the clamp defect above.
#
# A vector must declare db_clamp_top=None, because a function whose output
# depends on the rest of the array is not checkable frame by frame. But
# neither librosa nor torchaudio can be driven to None through its MFCC API:
# librosa's feature.mfcc takes no top_db and torchaudio's MFCC hardcodes
# 80.0. The only signals on which a no-clamp vector is runnable against
# either library are therefore those whose log-mel dynamic range falls below
# top_db, where the clamp is inert and clamped output equals unclamped
# output exactly.
#
# Measured: tone_plus_noise 34.5 dB (inert, max|d| 0.000e+00); a pure
# harmonic stack 127.7 dB (binds, max|d| 8.093e+01, vector unrunnable).
#
# This is enforced rather than documented because a suite that silently
# emits an unrunnable vector reads as coverage it does not have.
RUNNABLE_DYNAMIC_RANGE_DB = 80.0


def log_mel_dynamic_range_db(y: np.ndarray, decl: "MfccDeclaration") -> float:
    """Peak-to-floor spread of the log-mel spectrogram, in dB."""
    from clockvec.mfcc import frame_signal, mel_filterbank, pad_window, power_spectrum

    frames = frame_signal(y, n_fft=decl.n_fft, hop_length=decl.hop_length,
                          center=decl.center, pad_mode=decl.pad_mode)
    spec = power_spectrum(
        frames,
        window=pad_window(hann(decl.win_length), n_fft=decl.n_fft,
                          placement=decl.window_placement),
        power=decl.power, scale_by_n_fft=decl.scale_by_n_fft,
    )
    fb = mel_filterbank(n_mels=decl.n_mels, n_fft=decl.n_fft,
                        sample_rate=decl.sample_rate, f_min=decl.f_min,
                        f_max=decl.f_max, scale=decl.mel_scale,
                        area_normalize=decl.mel_area_normalize)
    lm = 10.0 * np.log10(np.maximum(fb @ spec, decl.log_floor))
    return float(lm.max() - lm.min())


class UnrunnableVectorError(ValueError):
    """A vector no shipped library could be checked against."""


EPS32 = float(np.finfo(np.float32).eps)  # 1.1920929e-07


def tolerance_float32_pipeline(values) -> tuple[float, str]:
    """Tolerance for an implementation that is float32 END TO END (torchaudio).

    Scales with coefficient magnitude. When the whole pipeline is float32 the
    log-mel array and the coefficients themselves carry float32 RELATIVE
    error, so the absolute error grows with |c|. An absolute tolerance
    measured on one signal does not transfer to another with larger
    coefficients; deriving it here rather than pinning a number is what stops
    a vector from accusing a correct implementation of nonconformance.

    Factor 16 is the sqrt(257) accumulation bound for the mel matmul, which
    dominates. Observed across the vectors in this module: 2.08, 2.08, 3.82.
    """
    mx = max(abs(v) for v in values)
    return 16.0 * EPS32 * mx, (
        f"torchaudio is float32 end to end, so its error scales with "
        f"coefficient magnitude (max|c| = {mx:.2f} here). tol = 16 * eps32 * "
        f"max|c|; 16 is the sqrt(257) accumulation bound for the mel matmul, "
        f"against a worst observed factor of 3.82. An absolute tolerance "
        f"transferred from another signal would be unsatisfiable here."
    )


def tolerance_float32_basis_only() -> tuple[float, str]:
    """Tolerance for a float64 pipeline carrying a float32 mel basis (librosa).

    Magnitude-INDEPENDENT, and that is not an approximation. The float32
    error is relative to the mel ENERGY and enters before the logarithm,
    which converts a relative error d into an absolute error of
    10*log10(1+d) ~= 4.343*d dB. The orthonormal DCT then amplifies a
    per-bin error by at most sqrt(n_mels).

    Bound: 4.343 * eps32 * sqrt(26) = 2.64e-06. Observed 1.471e-07, which is
    18x inside it. Stated as the bound rather than the observation so the
    tolerance does not silently tighten on a signal that happens to be kind.
    """
    return 3.0e-06, (
        "librosa is float64 except filters.mel, which returns float32. That "
        "error is relative to the mel energy and enters BEFORE the log, which "
        "turns it into a magnitude-independent absolute dB error of "
        "10/ln(10) * eps32, amplified by at most sqrt(n_mels) through the "
        "orthonormal DCT: 4.343 * 1.192e-07 * sqrt(26) = 2.64e-06, rounded to "
        "3.0e-06. Measured 1.471e-07. Substituting a float64 basis into "
        "librosa's own path drops the residual to 6.217e-14, confirming the "
        "basis as the sole cause."
    )


FLAGS: dict[str, Flag] = {
    "MelNormNameCollision": Flag(
        name="MelNormNameCollision",
        bug_type="NameCollision",
        description=(
            "librosa.feature.mfcc exposes two distinct norms: `norm` selects the "
            "DCT normalization and `mel_norm` selects the mel filterbank area "
            "normalization. A caller matching torchaudio's `norm=None` writes "
            "`norm=None` and silently retains Slaney area normalization on the "
            "filterbank. Measured mean|d| 11.00026 with the default in place "
            "against 0.00006 with mel_norm=None -- a factor of ~183,000 from one "
            "keyword. Reported and fixed, but the default is still 'slaney' in "
            "librosa 1.0.0, so the divergence from torchaudio's default persists."
        ),
        citations=(
            "https://github.com/librosa/librosa/issues/1842",
            "https://github.com/pytorch/audio/issues/1058",
            "https://github.com/librosa/librosa/pull/1844",
        ),
    ),
    "WholeClipDbClamp": Flag(
        name="WholeClipDbClamp",
        bug_type="StateLeak",
        description=(
            "The log-mel spectrogram is floored at (peak - top_db) where the peak "
            "is reduced over the ENTIRE array, both mel and time axes, so the "
            "coefficients of any frame depend on every other frame present. "
            "librosa: feature.mfcc accepts no top_db and forwards **kwargs to "
            "melspectrogram rather than power_to_db, whose default is 80.0; in "
            "1.0.0 power_to_db's axes='auto' resolves to (-2,-1) for 2-D input, "
            "which is mel and time. torchaudio: transforms.MFCC.__init__ "
            "hardcodes self.top_db = 80.0. Neither is reachable through the MFCC "
            "API. Measured on 200 otherwise byte-identical frames, varying only "
            "what follows them: +59 dB burst -> max|d| 1.9362; +74 dB -> 25.6987; "
            "+94 dB -> 124.0154 with c1..c12 energy exactly 0.00 and all 200 "
            "frames clamped flat. Defeating the clamp (torchaudio log_mels=True) "
            "returns 0.000000000, which isolates the clamp as the sole cause. "
            "Consequence: streaming and batch inference compute different "
            "features from identical audio, by default, in both libraries."
        ),
        citations=(
            "https://github.com/librosa/librosa/blob/main/librosa/core/spectrum.py",
            "https://github.com/pytorch/audio/blob/main/src/torchaudio/transforms/_transforms.py",
        ),
    ),
    "LogBase": Flag(
        name="LogBase",
        bug_type="LogBase",
        description=(
            "librosa and torchaudio use 10*log10; python_speech_features uses the "
            "natural logarithm. A pure factor of 10/ln(10) = 4.34294 on every "
            "coefficient. Large, trivially reconciled once declared, and "
            "undeclared everywhere."
        ),
        citations=("https://github.com/jameslyons/python_speech_features",),
    ),
    "WindowPlacement": Flag(
        name="WindowPlacement",
        bug_type="SpecAmbiguity",
        description=(
            "When win_length < n_fft the analysis window must be padded to n_fft. "
            "Centred padding weights samples [k*hop + (n_fft-win)//2, ...); "
            "left-aligned padding weights [k*hop, ...). With n_fft=512 and "
            "win_length=400 the frames are offset by 56 samples and the frame "
            "counts differ by one. Davis & Mermelstein fixes neither."
        ),
        citations=(),
    ),
    "Float32Basis": Flag(
        name="Float32Basis",
        bug_type="Precision",
        description=(
            "librosa.filters.mel returns float32 and feature.melspectrogram does "
            "not request otherwise, so a float64 pipeline still carries a float32 "
            "mel basis. Substituting a float64 basis into librosa's own path drops "
            "its disagreement with an exact implementation from 1.426e-07 to "
            "6.217e-14; librosa-float32 against librosa-float64 is 1.426e-07, the "
            "same number, so the basis is the sole cause. This is a floor, not a "
            "bug: no implementation can agree with default librosa more closely, "
            "and any vector asserting a tighter tolerance is unsatisfiable."
        ),
        citations=(),
    ),
}


# The convention set everything else is stated relative to. Chosen to match
# what librosa and torchaudio can BOTH be driven to, so a vector under it is
# checkable against either without either being declared the winner.
def _pinned(**over) -> MfccDeclaration:
    base = dict(
        sample_rate=16000.0, n_mfcc=13, n_fft=512, win_length=400, hop_length=160,
        window_placement="center", center=False, pad_mode="constant", n_mels=26,
        f_min=0.0, f_max=8000.0, mel_scale=MelScale.HTK, mel_area_normalize=False,
        power=2.0, scale_by_n_fft=False, log_scale="db", log_floor=1e-10,
        db_clamp_top=None, dct_ortho=True, lifter=0.0, pre_emphasis=None,
    )
    base.update(over)
    return MfccDeclaration(**base)


def _expectation(name: str, decl: MfccDeclaration, y: np.ndarray, *,
                 model: str, result: Result = Result.VALID,
                 flags: tuple[str, ...] = ()) -> Expectation:
    """`model` selects a derived tolerance, never a literal.

    Deriving it is the point. The first draft of this module pinned 2.1e-05
    from a measurement on one signal, which then FAILED torchaudio on a
    second signal whose coefficients were larger -- a correct implementation
    reported as nonconforming by the suite meant to adjudicate it.
    """
    values = tuple(float(v) for v in mfcc(y, **decl.kwargs()).ravel(order="C"))
    if model == "float32_pipeline":
        tol, rationale = tolerance_float32_pipeline(values)
    elif model == "float32_basis":
        tol, rationale = tolerance_float32_basis_only()
    else:
        raise ValueError(f"unknown tolerance model {model!r}")
    return Expectation(
        implementation=name, version="clockvec-reference", values=values,
        tolerance=tol, tolerance_rationale=rationale, result=result, flags=flags,
    )


def build_vectors() -> list[Vector]:
    """The vector set.

    Each vector isolates exactly one divergence. Where the 1980 paper decides
    the question, one answer is VALID and the other INVALID. Where it does
    not, both are ACCEPTABLE and the disagreement is the finding -- asserting
    a winner there would be inventing a standard rather than documenting one.
    """
    vectors: list[Vector] = []
    runnable_checks: list[tuple[Vector, np.ndarray, MfccDeclaration]] = []
    n = 160 * 24 + 512  # 24 frames at hop 160, center=False

    # --- 1. mel filterbank area normalization -----------------------------
    y = signal("tone_plus_noise", n)
    v = Vector(
        vector_id="mfcc/mel-area-normalization",
        method="MFCC",
        source_paper=SOURCE,
        seed=SEED,
        input_shape=(n,),
        declaration=_pinned(),
        authority=Authority(
            claims=(
                "Under the declared convention set, these are the required "
                "coefficients. The two expectations differ ONLY in "
                "mel_area_normalize."
            ),
            does_not_claim=(
                "Which of the two is correct. Davis & Mermelstein 1980 specifies "
                "triangular filters spaced on the mel scale and says nothing about "
                "normalizing their area; Slaney's Auditory Toolbox introduced the "
                "2/(f[i+2]-f[i]) scaling later. Both are defensible readings, so "
                "both are ACCEPTABLE and neither is a bug."
            ),
            basis=(
                "Davis & Mermelstein 1980 sec. II; librosa issue 1842 and "
                "pytorch/audio issue 1058 for the collision that makes the choice "
                "invisible in practice."
            ),
        ),
        flags={"MelNormNameCollision": FLAGS["MelNormNameCollision"],
               "Float32Basis": FLAGS["Float32Basis"]},
    )
    v.expectations = [
        _expectation("area_normalize=False (torchaudio default)", _pinned(), y,
                     model="float32_pipeline",
                     result=Result.ACCEPTABLE, flags=("MelNormNameCollision",)),
        _expectation("area_normalize=True (librosa default)",
                     _pinned(mel_area_normalize=True), y,
                     model="float32_basis",
                     result=Result.ACCEPTABLE,
                     flags=("MelNormNameCollision", "Float32Basis")),
    ]
    runnable_checks.append((v, y, _pinned()))
    vectors.append(v)

    # --- 2. the dB clamp: MFCC is not a function of its frame -------------
    quiet = signal("quiet_noise", n)
    loud = np.concatenate([quiet, 50.0 * np.random.default_rng(SEED + 1).standard_normal(160 * 100)])
    n_out = 13 * (1 + (n - 512) // 160)
    v = Vector(
        vector_id="mfcc/whole-clip-db-clamp",
        method="MFCC",
        source_paper=SOURCE,
        seed=SEED,
        input_shape=(n,),
        declaration=_pinned(db_clamp_top=80.0),
        authority=Authority(
            claims=(
                "The MFCC of a frame is determined by that frame. Davis & "
                "Mermelstein define the coefficients as the DCT of the log mel "
                "energies OF A FRAME; nothing in the definition references other "
                "frames. An implementation whose output for a frame changes when "
                "distant, non-overlapping audio is appended does not compute the "
                "published function. The isolated result is VALID; the "
                "context-dependent result is INVALID."
            ),
            does_not_claim=(
                "That an 80 dB dynamic-range floor is a bad idea, or what the "
                "correct top_db value would be. The defect is the SCOPE of the "
                "reduction, not its existence: a per-frame floor would be a "
                "convention, and this package would record it as one."
            ),
            basis=(
                "Davis & Mermelstein 1980 sec. II. Measured: appending a +94 dB "
                "burst after 200 otherwise byte-identical frames drives c1..c12 "
                "energy from 8066.57 to exactly 0.00 and clamps all 200 frames "
                "flat. With the clamp defeated the same comparison returns "
                "0.000000000, isolating it as the sole cause."
            ),
        ),
        flags={"WholeClipDbClamp": FLAGS["WholeClipDbClamp"]},
    )
    isolated = mfcc(quiet, **_pinned(db_clamp_top=80.0).kwargs())
    in_context = mfcc(loud, **_pinned(db_clamp_top=80.0).kwargs())[:, : isolated.shape[1]]
    v.expectations = [
        Expectation(
            implementation="frame computed in isolation",
            version="clockvec-reference",
            values=tuple(float(x) for x in isolated.ravel(order="C")),
            tolerance=tolerance_float32_basis_only()[0],
            tolerance_rationale=tolerance_float32_basis_only()[1],
            result=Result.VALID,
        ),
        Expectation(
            implementation="same frames, +94 dB burst appended after them",
            version="clockvec-reference",
            values=tuple(float(x) for x in in_context.ravel(order="C")),
            tolerance=tolerance_float32_basis_only()[0],
            tolerance_rationale=(
                tolerance_float32_basis_only()[1]
                + " The disagreement measured on this vector is 113.63, eight "
                "orders of magnitude above the tolerance, so no plausible "
                "tolerance decides this case differently."),
            result=Result.INVALID,
            flags=("WholeClipDbClamp",),
        ),
    ]
    assert len(v.expectations[0].values) == n_out
    vectors.append(v)

    # --- 3. log base ------------------------------------------------------
    y = signal("harmonic_stack", n)
    v = Vector(
        vector_id="mfcc/log-base",
        method="MFCC",
        source_paper=SOURCE,
        seed=SEED,
        input_shape=(n,),
        declaration=_pinned(),
        authority=Authority(
            claims=(
                "The two expectations are related by exactly 10/ln(10) = "
                "4.342944819032518 on every coefficient. An implementation "
                "matching neither has a further divergence, and this vector "
                "localizes it away from the log base."
            ),
            does_not_claim=(
                "Which base is correct. Davis & Mermelstein write 'log' without "
                "fixing a base, and the cepstral literature uses both. Both are "
                "ACCEPTABLE."
            ),
            basis="Davis & Mermelstein 1980 sec. II, eq. for the mel cepstrum.",
        ),
        flags={"LogBase": FLAGS["LogBase"]},
    )
    v.expectations = [
        _expectation("log_scale=db (librosa, torchaudio)", _pinned(), y,
                     model="float32_pipeline",
                     result=Result.ACCEPTABLE, flags=("LogBase",)),
        _expectation("log_scale=ln (python_speech_features)",
                     _pinned(log_scale="ln"), y,
                     model="float32_pipeline",
                     result=Result.ACCEPTABLE, flags=("LogBase",)),
    ]
    runnable_checks.append((v, y, _pinned()))
    vectors.append(v)

    # --- 4. window placement ----------------------------------------------
    y = signal("tone_plus_noise", n)
    v = Vector(
        vector_id="mfcc/window-placement",
        method="MFCC",
        source_paper=SOURCE,
        seed=SEED,
        input_shape=(n,),
        declaration=_pinned(),
        authority=Authority(
            claims=(
                "With n_fft=512 and win_length=400 these two placements analyse "
                "sample ranges offset by 56 samples. An implementation that "
                "matches neither is not merely mis-placed."
            ),
            does_not_claim=(
                "Which placement is correct. Both librosa and torchaudio centre, "
                "so centred is the de facto convention, but the paper does not "
                "address padding a short window at all and a majority is not a "
                "specification. Both ACCEPTABLE."
            ),
            basis="Davis & Mermelstein 1980; librosa util.pad_center.",
        ),
        flags={"WindowPlacement": FLAGS["WindowPlacement"]},
    )
    v.expectations = [
        _expectation("window_placement=center (librosa, torchaudio)", _pinned(), y,
                     model="float32_pipeline",
                     result=Result.ACCEPTABLE, flags=("WindowPlacement",)),
        _expectation("window_placement=left", _pinned(window_placement="left"), y,
                     model="float32_pipeline",
                     result=Result.ACCEPTABLE, flags=("WindowPlacement",)),
    ]
    runnable_checks.append((v, y, _pinned()))
    vectors.append(v)

    for vec in vectors:
        vec.validate()
    _assert_runnable(
        [(v.vector_id, sig, decl) for v, sig, decl in runnable_checks]
    )
    return vectors


def _assert_runnable(cases: list[tuple[str, np.ndarray, MfccDeclaration]]) -> None:
    """Refuse to emit a vector librosa and torchaudio could not be run against."""
    for vector_id, y, decl in cases:
        rng_db = log_mel_dynamic_range_db(y, decl)
        if rng_db >= RUNNABLE_DYNAMIC_RANGE_DB:
            raise UnrunnableVectorError(
                f"{vector_id}: log-mel dynamic range {rng_db:.1f} dB exceeds "
                f"{RUNNABLE_DYNAMIC_RANGE_DB} dB, so librosa's and torchaudio's "
                "unconditional clamp binds and neither can be checked against "
                "this vector. Add a noise floor to the signal or lower its "
                "crest factor."
            )


def write_vectors(directory: str = "vectors") -> list[str]:
    """Emit the vector set as static JSON.

    Static files with a versioned schema and no server, for the reason in
    `vector`'s docstring: every always-on benchmark service in the surveyed
    record is dead, and the survivors are files. This has to keep working
    after nobody is maintaining it.
    """
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
